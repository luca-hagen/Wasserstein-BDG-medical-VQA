"""
Gemma3 / MedGemma Backend: model loading, prompt encoding, sampling, log-prob scoring.

Key differences vs Qwen:
  - Gemma3ForConditionalGeneration
  - System content must be a LIST: [{"type": "text", "text": ...}]
  - apply_chat_template(tokenize=True, return_dict=True) returns full encoding
  - bfloat16 preferred
  - Stop tokens: <eos> (id=1), <end_of_turn>
  - No batched text+image encoding → sequential logprob computation
"""
from __future__ import annotations
import logging
from typing import List, Dict
from itertools import groupby
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

from config import ExperimentConfig, SYSTEM_SAMPLE_CORRECT, SYSTEM_SAMPLE_INCORRECT
from utils import (
    ImageInput, normalize_images, validate_token_ids,
    canonicalize_answer, is_bad_candidate,
    semantic_dedup, cluster_and_select, compute_ground_metric,
    build_sampling_schedule,
)

log = logging.getLogger(__name__)


# =========================================================================
# Stop-token detection
# =========================================================================
def _get_stop_ids(processor) -> list:
    tokenizer = getattr(processor, "tokenizer", processor)
    vocab_size = len(tokenizer)
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for tok_str in ["<end_of_turn>", "<eos>", "<|endoftext|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and 0 <= int(tid) < vocab_size:
                unk = tokenizer.unk_token_id
                if unk is None or int(tid) != int(unk):
                    ids.add(int(tid))
        except Exception:
            pass
    return sorted(ids)


def _was_truncated(new_ids: torch.Tensor, processor, max_new_tokens: int) -> bool:
    if new_ids is None:
        return True
    n = int(new_ids.numel())
    if n == 0 or n < int(max_new_tokens):
        return False
    stop_ids = set(_get_stop_ids(processor))
    return not any(int(t.item()) in stop_ids for t in new_ids)


# =========================================================================
# Encoding helpers
# =========================================================================
def _build_messages(system: str, user_text: str, images: List[Image.Image]) -> list:
    """Gemma 3: system content must be a LIST, not a string."""
    user_content = []
    for img in images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_content},
    ]


def _encode_single(processor, system: str, user_text: str,
                    image: ImageInput = None) -> dict:
    images = normalize_images(image)
    messages = _build_messages(system, user_text, images)
    return processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )


def _to_device(inputs: dict, device) -> dict:
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = (v.to(device=device, dtype=torch.bfloat16)
                       if v.is_floating_point() else v.to(device=device))
        else:
            out[k] = v
    return out


# =========================================================================
# Forward pass
# =========================================================================
@torch.no_grad()
def _forward_logits(model, input_ids, attention_mask, **extra):
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    fwd = {
        k: (v.to(device=device, dtype=torch.bfloat16)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in extra.items()
    }
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attention_mask, **fwd, return_dict=True)
    logits = out.logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("Bad logits (NaN/Inf)")
    return logits


# =========================================================================
# GemmaBackend
# =========================================================================
@dataclass
class CandidateResult:
    Y: List[str]
    D: np.ndarray
    embeddings: np.ndarray
    raw_pool: List[str]
    raw_pool_embeddings: np.ndarray


class GemmaBackend:
    def __init__(self, model_name: str, device: str = "cuda:0",
                 dtype=torch.bfloat16):
        log.info(f"Loading Gemma model: {model_name} → {device}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        tok = getattr(self.processor, "tokenizer", self.processor)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(device).eval()
        self.device = torch.device(device)

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    # ── Sampling ─────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_batch(
        self, system: str, user_text: str, image: ImageInput,
        seeds: List[int], max_new_tokens: int,
        temperature: float, top_p: float, top_k: int,
    ) -> List[Dict]:
        inputs = _encode_single(self.processor, system, user_text, image)
        inputs = _to_device(inputs, self.device)
        prompt_len = inputs["input_ids"].shape[1]
        validate_token_ids(inputs["input_ids"], self.vocab_size, "gemma_batch")

        B = len(seeds)
        expanded = {
            k: v.expand(B, *v.shape[1:]).contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        gen_kwargs = dict(
            do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
            max_new_tokens=max_new_tokens, pad_token_id=pad_id,
            return_dict_in_generate=True,
        )

        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seeds[0]))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seeds[0]))
            out = self.model.generate(**expanded, **gen_kwargs)

        results = []
        for i in range(B):
            new_ids = out.sequences[i][prompt_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append({"text": text, "new_ids": new_ids})
        return results

    def was_truncated(self, new_ids: torch.Tensor, max_new_tokens: int) -> bool:
        return _was_truncated(new_ids, self.processor, max_new_tokens)

    # ── Log-prob scoring (sequential — Gemma can't batch text+image) ─
    @torch.no_grad()
    def logprob_completions(
        self, system: str, user_text: str,
        completions: List[str], image: ImageInput = None,
        length_norm: bool = True,
    ) -> List[float]:
        images = normalize_images(image)
        if not completions:
            return []

        messages = _build_messages(system, user_text, images)
        prompt_enc = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        prompt_len = int(prompt_enc["input_ids"].shape[1])
        prompt_str = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        del prompt_enc

        results = []
        for c in completions:
            enc_kwargs = dict(text=[prompt_str + c], padding=True, return_tensors="pt")
            if images:
                enc_kwargs["images"] = images
            enc = self.processor(**enc_kwargs)
            enc = _to_device(enc, self.device)

            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
            extra = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")}

            logits = _forward_logits(self.model, input_ids, attention_mask, **extra)

            seq_len = int(attention_mask[0].sum().item())
            comp_len = seq_len - prompt_len
            if comp_len <= 0:
                results.append(0.0)
                continue
            pred = logits[0, prompt_len - 1: prompt_len - 1 + comp_len, :]
            lp = F.log_softmax(pred.float(), dim=-1)
            target = input_ids[0, prompt_len: prompt_len + comp_len]
            val = lp[torch.arange(comp_len, device=logits.device), target].sum().item()
            if length_norm:
                val /= comp_len
            results.append(val)
        return results

    @torch.no_grad()
    def logprob_prompts(
        self, system: str, user_texts: List[str],
        completion: str, image: ImageInput = None,
        length_norm: bool = True,
    ) -> List[float]:
        images = normalize_images(image)
        if not user_texts:
            return []

        results = []
        for ut in user_texts:
            messages = _build_messages(system, ut, images)
            prompt_enc = self.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            )
            prompt_len = int(prompt_enc["input_ids"].shape[1])
            prompt_str = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            del prompt_enc

            enc_kwargs = dict(text=[prompt_str + completion], padding=True, return_tensors="pt")
            if images:
                enc_kwargs["images"] = images
            enc = self.processor(**enc_kwargs)
            enc = _to_device(enc, self.device)

            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
            extra = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")}

            logits = _forward_logits(self.model, input_ids, attention_mask, **extra)

            seq_len = int(attention_mask[0].sum().item())
            comp_len = seq_len - prompt_len
            if comp_len <= 0:
                results.append(0.0)
                continue
            pred = logits[0, prompt_len - 1: prompt_len - 1 + comp_len, :]
            lp = F.log_softmax(pred.float(), dim=-1)
            target = input_ids[0, prompt_len: prompt_len + comp_len]
            val = lp[torch.arange(comp_len, device=logits.device), target].sum().item()
            if length_norm:
                val /= comp_len
            results.append(val)
        return results

    # ── Candidate set builder ────────────────────────────────────────
    def build_candidate_set(
        self, question: str, image: ImageInput,
        cfg: ExperimentConfig, embedder,
        base_seed: int = 9000, batch_size: int = 8,
    ) -> CandidateResult:
        schedule = build_sampling_schedule(cfg)
        pool, seen, semantic_pool = [], set(), []

        key_fn = lambda e: (e["system"], e["temperature"])
        schedule_sorted = sorted(schedule, key=key_fn)

        for (sys_prompt, temp), group_iter in groupby(schedule_sorted, key=key_fn):
            if len(pool) >= cfg.n_oversample:
                break
            group = list(group_iter)
            user_text = question
            if sys_prompt == SYSTEM_SAMPLE_CORRECT:
                user_text += "\n\nAnswer: "
            elif sys_prompt == SYSTEM_SAMPLE_INCORRECT:
                user_text += "\n\nIncorrect answer: "

            for batch_start in range(0, len(group), batch_size):
                if len(pool) >= cfg.n_oversample:
                    break
                batch = group[batch_start:batch_start + batch_size]
                seeds = [base_seed + e["seed_offset"] for e in batch]
                samples = self.sample_batch(
                    sys_prompt, user_text, image,
                    seeds=seeds, max_new_tokens=cfg.max_new_tokens_sample,
                    temperature=temp, top_p=cfg.top_p, top_k=cfg.top_k_sample,
                )
                for out in samples:
                    if len(pool) >= cfg.n_oversample:
                        break
                    if self.was_truncated(out["new_ids"], cfg.max_new_tokens_sample):
                        continue
                    ans = canonicalize_answer(out["text"])
                    if is_bad_candidate(ans, cfg):
                        continue
                    key = ans.lower().strip()
                    if key in seen:
                        continue
                    pool.append(ans)
                    seen.add(key)
                    semantic_pool.append(question + "\nAnswer: " + ans)

        if not pool:
            raise RuntimeError(f"No candidates for: {question[:80]}")

        sem_emb = embedder.encode(semantic_pool, normalize_embeddings=True)
        pool, sem_emb = semantic_dedup(pool, sem_emb, threshold=0.99)
        n_target = min(cfg.n_candidates, len(pool))
        if n_target > 1 and n_target % 2 != 0:
            n_target -= 1
        sel_texts, sel_emb, _ = cluster_and_select(pool, sem_emb, n_target, cfg.cluster_seed)
        D = compute_ground_metric(sel_emb, cfg.ground_metric)
        return CandidateResult(
            Y=sel_texts, D=D, embeddings=sel_emb,
            raw_pool=pool, raw_pool_embeddings=sem_emb,
        )