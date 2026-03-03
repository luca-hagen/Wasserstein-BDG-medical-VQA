"""
Shared utilities: math helpers, embedder, evaluation, VLM judge,
candidate canonicalization, clustering, and Wasserstein distance.
"""
from __future__ import annotations
import re
import base64
import json
import logging
import time
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import requests
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from config import (
    ExperimentConfig,
    OPENROUTER_API_KEY, OPENROUTER_JUDGE_MODEL,
    SYSTEM_JUDGE, SYSTEM_SAMPLE_CORRECT, SYSTEM_SAMPLE_INCORRECT, REPHRASE_PROMPT,
)

log = logging.getLogger(__name__)

try:
    import ot as pot
    HAS_POT = True
except ImportError:
    HAS_POT = False

# =========================================================================
# Type aliases
# =========================================================================
ImageInput = Optional[Union[Image.Image, List[Image.Image]]]


def normalize_images(image: ImageInput) -> List[Image.Image]:
    if image is None:
        return []
    if isinstance(image, Image.Image):
        return [image]
    if isinstance(image, (list, tuple)):
        return [img for img in image if isinstance(img, Image.Image)]
    return []


# =========================================================================
# SapBERT Embedder
# =========================================================================
class SapBERTCLSEmbedder:
    def __init__(self, name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 device=None, max_length=25):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        bs = 128
        all_embs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            toks = self.tokenizer(
                batch, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks)
            cls = out.last_hidden_state[:, 0, :]
            if normalize_embeddings:
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            all_embs.append(cls.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


# =========================================================================
# Math helpers
# =========================================================================
def safe_normalize(p, axis=None, eps=1e-20):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    s = p.sum(axis=axis, keepdims=True)
    return p / np.clip(s, eps, None)


def softmax_from_logs(logits):
    m = np.max(logits)
    ex = np.exp(logits - m)
    return ex / np.sum(ex)


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def ranks_from_scores(scores):
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def reverse_distribution_by_rank(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).copy()
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(p) / len(p)
    p /= s
    asc = np.argsort(p, kind="mergesort")
    masses_desc = np.sort(p, kind="mergesort")[::-1]
    q = np.empty_like(p)
    q[asc] = masses_desc
    q /= q.sum()
    return q


def validate_token_ids(input_ids: torch.Tensor, vocab_size: int, label: str = ""):
    mx, mn = input_ids.max().item(), input_ids.min().item()
    if mx >= vocab_size or mn < 0:
        raise ValueError(f"[{label}] Token IDs out of range: min={mn}, max={mx}, vocab={vocab_size}")


# =========================================================================
# Wasserstein-1
# =========================================================================
def wasserstein_1(p: np.ndarray, q: np.ndarray, D: np.ndarray) -> float:
    p = safe_normalize(p).astype(np.float64)
    q = safe_normalize(q).astype(np.float64)
    D = np.asarray(D, dtype=np.float64)
    if HAS_POT:
        return float(pot.emd2(p, q, D))
    else:
        n = len(p)
        c = D.ravel()
        A_eq = np.zeros((2 * n, n * n))
        for i in range(n):
            A_eq[i, i * n:(i + 1) * n] = 1.0
            for j in range(n):
                A_eq[n + j, i * n + j] = 1.0
        b_eq = np.concatenate([p, q])
        from scipy.optimize import linprog
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        return float(res.fun) if res.success else float("inf")


def wasserstein_proximal(
    p_new: np.ndarray, p_old: np.ndarray, D: np.ndarray,
    alpha: float = 0.3, reg: float = 0.1,
) -> np.ndarray:
    d_nonzero = D[D > 1e-3]
    reg = float(np.median(d_nonzero)) * 0.5 if len(d_nonzero) > 0 else 0.1
    reg = np.clip(reg, 0.1, 1.0)
    A = np.stack([p_new, p_old], axis=1)
    weights = np.array([1.0 - alpha, alpha])
    bary = pot.bregman.barycenter(
        A=A, M=D, reg=reg, weights=weights, numItermax=200, stopThr=1e-2,
    )
    return bary / bary.sum()


# =========================================================================
# Candidate canonicalization & filtering
# =========================================================================
_WS_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)\]])\s*")
_FILLER_PREFIXES = [
    "the answer is", "answer:", "final answer:", "it is", "there is",
    "this is", "final:", "verdict:", "the finding is", "the diagnosis is",
    "based on the image,", "based on the image", "the image shows",
    "i can see", "looking at the image,",
]
_BAD_RE = re.compile(
    r"|".join([
        r"\blook(?:ing)?\s+(?:at|for)\b", r"\bplease\b", r"\bobserve\b",
        r"\bnote that\b", r"\bevaluate\b", r"\bconsider\b", r"\bshould\b",
        r"\bsuggest(?:s|ing)?\b", r"\bhowever\b", r"\btherefore\b",
        r"\bin (?:this|the) image\b", r"\bbased on\b", r"\bI (?:can|would|think)\b",
    ]), re.IGNORECASE,
)


def canonicalize_answer(s: str) -> str:
    if not s:
        return ""
    s = s.strip().split("\n")[0].strip()
    s = _BULLET_RE.sub("", s).strip()
    changed = True
    while changed:
        changed = False
        low = s.lower()
        for prefix in _FILLER_PREFIXES:
            if low.startswith(prefix):
                s = s[len(prefix):].strip()
                changed = True
    s = s.strip("\"'`*_ \t\r\n:;-.,!")
    s = _WS_RE.sub(" ", s).strip()
    return s


def is_bad_candidate(s: str, cfg: ExperimentConfig) -> bool:
    if not s or len(s) < cfg.min_answer_chars:
        return True
    if len(s.split()) > cfg.max_answer_words:
        return True
    if _BAD_RE.search(s):
        return True
    if re.fullmatch(r"\d+\.?\d*", s):
        return True
    return False


# =========================================================================
# Clustering & ground metric
# =========================================================================
def compute_ground_metric(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    sim = cosine_similarity(embeddings)
    if metric == "cosine":
        D = 1.0 - sim
    elif metric == "euclidean":
        D = np.sqrt(np.clip(2.0 - 2.0 * sim, 0, None))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0, None)
    return (D + D.T) / 2.0


def semantic_dedup(
    texts: List[str], embeddings: np.ndarray, threshold: float = 0.90,
) -> Tuple[List[str], np.ndarray]:
    if not texts:
        return texts, embeddings
    kept_indices = [0]
    for i in range(1, len(texts)):
        sims = cosine_similarity(embeddings[i:i + 1], embeddings[kept_indices])[0]
        if sims.max() < threshold:
            kept_indices.append(i)
    return [texts[i] for i in kept_indices], embeddings[kept_indices]


def cluster_and_select(
    pool: List[str], emb: np.ndarray, n_target: int, seed: int,
    density_weighted: bool = True,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if len(pool) <= n_target:
        return pool, emb, np.arange(len(pool))

    km = KMeans(n_clusters=n_target, random_state=seed, n_init="auto")
    labels = km.fit_predict(emb)
    selected_idx = []

    if density_weighted:
        cluster_sizes = np.bincount(labels, minlength=n_target)
        proportional = cluster_sizes / cluster_sizes.sum() * n_target
        slots = np.maximum(1, np.round(proportional).astype(int))
        diff = n_target - slots.sum()
        if diff > 0:
            order = np.argsort(-cluster_sizes)
            for i in range(diff):
                slots[order[i % n_target]] += 1
        elif diff < 0:
            order = np.argsort(cluster_sizes)
            for i in range(-diff):
                if slots[order[i % n_target]] > 1:
                    slots[order[i % n_target]] -= 1
        for c in range(n_target):
            cluster_idx = np.where(labels == c)[0]
            centroid = km.cluster_centers_[c]
            dists = np.linalg.norm(emb[cluster_idx] - centroid, axis=1)
            sorted_idx = cluster_idx[np.argsort(dists)]
            n_pick = min(slots[c], len(sorted_idx))
            selected_idx.extend(sorted_idx[:n_pick].tolist())
        selected_idx = selected_idx[:n_target]
    else:
        for c in range(n_target):
            cluster_idx = np.where(labels == c)[0]
            centroid = km.cluster_centers_[c]
            dists = np.linalg.norm(emb[cluster_idx] - centroid, axis=1)
            selected_idx.append(cluster_idx[np.argmin(dists)])

    selected_idx = np.array(selected_idx)
    return [pool[i] for i in selected_idx], emb[selected_idx], selected_idx


def build_sampling_schedule(cfg: ExperimentConfig) -> List[Dict]:
    schedule = []
    idx = 0
    temps = cfg.temperatures
    for i in range(int(cfg.max_sampling_calls * cfg.frac_correct)):
        schedule.append({"system": SYSTEM_SAMPLE_CORRECT,
                         "temperature": temps[i % len(temps)], "seed_offset": idx})
        idx += 1
    for i in range(int(cfg.max_sampling_calls * cfg.frac_incorrect)):
        schedule.append({"system": SYSTEM_SAMPLE_INCORRECT,
                         "temperature": temps[i % len(temps)], "seed_offset": idx})
        idx += 1
    for i in range(int(cfg.max_sampling_calls * cfg.frac_rephrase)):
        schedule.append({"system": REPHRASE_PROMPT,
                         "temperature": temps[i % len(temps)], "seed_offset": idx})
        idx += 1
    return schedule


# =========================================================================
# Evaluation
# =========================================================================
def evaluate_answer(
    pred: str, gt: str, embedder, threshold: float = 0.80,
) -> Dict:
    pred_n = pred.strip().lower()
    gt_n = gt.strip().lower()
    exact = pred_n == gt_n

    pred_tok = set(pred_n.split())
    gt_tok = set(gt_n.split())
    if pred_tok and gt_tok:
        prec = len(pred_tok & gt_tok) / len(pred_tok)
        rec = len(pred_tok & gt_tok) / len(gt_tok)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    else:
        f1 = float(pred_n == gt_n)

    embs = embedder.encode([pred, gt], normalize_embeddings=True)
    sem_sim = float(np.dot(embs[0], embs[1]))
    soft_match = sem_sim >= threshold

    return {"exact": exact, "f1": f1, "sem_sim": sem_sim, "soft_match": soft_match}


# =========================================================================
# VLM-as-Judge (OpenRouter API)
# =========================================================================
JUDGE_API_ERROR = float("nan")


def _image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def vlm_judge(
    question: str, gt: str, pred: str, image: ImageInput,
    *,
    api_key: str = OPENROUTER_API_KEY,
    judge_model: str = OPENROUTER_JUDGE_MODEL,
    max_retries: int = 3,
) -> float:
    images = normalize_images(image)
    user_text = (
        f"Question: {question}\n\n"
        f"Ground truth answer: {gt}\n"
        f"Predicted answer: {pred}\n\n"
        f"Verdict: "
    )

    user_content = []
    if images:
        b64 = _image_to_base64(images[0])
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    user_content.append({"type": "text", "text": user_text})

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": SYSTEM_JUDGE},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload, timeout=30,
            )
            if not resp.ok:
                log.warning(f"[judge] HTTP {resp.status_code} attempt {attempt + 1}/{max_retries}: {resp.text[:300]}")
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code == 400:
                    payload["messages"][1]["content"] = [{"type": "text", "text": user_text}]
                    continue
                resp.raise_for_status()

            text = resp.json()["choices"][0]["message"]["content"].strip().lower()
            if "incorrect" in text:
                return 0.0
            elif "correct" in text:
                return 1.0
            else:
                continue
        except Exception as e:
            log.warning(f"[judge] Attempt {attempt + 1}/{max_retries} failed: {e}")

    log.error(f"[judge] All {max_retries} retries failed for pred='{pred[:60]}' → NaN")
    return JUDGE_API_ERROR