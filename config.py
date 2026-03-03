"""
Shared configuration, environment setup, and prompt templates.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple

# ─── Cache / env ────────────────────────────────────────────────────────
CACHE = "/vol/ideadata/ak95ecuh/hf_cache"
os.environ["HF_HOME"] = CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE, "datasets")
os.environ["XDG_CACHE_HOME"] = CACHE
os.environ["HF_TOKEN"] = "enter your hf_token here"
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

OPENROUTER_API_KEY = "enter your openrouter_key here"
OPENROUTER_JUDGE_MODEL = "x-ai/grok-4-fast"


# =========================================================================
# Experiment config — pure hyperparameters, NO model names
# =========================================================================
@dataclass
class ExperimentConfig:
    # ── Candidate generation ─────────────────────────────────────────
    n_candidates: int = 8
    n_oversample: int = 12
    max_sampling_calls: int = 16
    max_new_tokens_sample: int = 24
    temperatures: Tuple[float, ...] = (0.5, 1.0)
    top_p: float = 0.98
    top_k_sample: int = 100
    frac_correct: float = 1.0
    frac_incorrect: float = 0.0
    frac_rephrase: float = 0.0
    max_answer_words: int = 10
    min_answer_chars: int = 1

    # ── Embedder / clustering ────────────────────────────────────────
    embedder_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    cluster_method: str = "kmeans"
    cluster_seed: int = 0
    ground_metric: str = "cosine"

    # ── BDG hyperparameters ──────────────────────────────────────────
    eta_G: float = 0.4
    eta_V: float = 0.4
    lambda_G: float = 0.4
    lambda_V: float = 0.4
    sigma: float = 5e-3
    max_iter: int = 500
    reverse_incorrect: bool = True

    # ── Wasserstein stopping ─────────────────────────────────────────
    use_wasserstein: bool = True
    w_epsilon: float = 0.2
    delta_l1: float = 0.2

    # ── Wasserstein update ───────────────────────────────────────────
    use_wasserstein_update: bool = False
    w_alpha: float = 0.2
    w_reg: float = 0.1

    # ── Scoring ──────────────────────────────────────────────────────
    use_length_norm: bool = True
    use_contrastive_norm: bool = False
    verifier_y_calibration: bool = False

    # ── Evaluation ───────────────────────────────────────────────────
    semantic_match_threshold: float = 0.9


# =========================================================================
# Prompt templates
# =========================================================================
SYSTEM_SAMPLE_CORRECT = """\
You are a medical expert specializing in radiology and pathology.
Given a medical question, produce a correct answer.
Follow radiological convention: the patient's RIGHT side appears on the LEFT of the image, and vice versa.
To generate a correct answer, first identify precisely what the question is asking.
If the question allows for a yes/no answer, respond strictly with "yes" or "no".
Otherwise respond with a single concise medical term or short phrase — no explanations, no punctuation.
Limit your answer to a maximum of 10 words."""

SYSTEM_SAMPLE_INCORRECT = """\
You are a medical education expert generating exam distractors.
Given a medical question, produce a WRONG but plausible answer that could mislead a student.
To generate a plausible distractor, first identify precisely what the question is asking. The distractor should be related to the correct answer (e.g., same organ system, nearby structure, wrong laterality, or a common anatomic confusion).
Output only the distractor — one short medical term or phrase, no explanation."""

SYSTEM_GEN_CORRECT = """\
You are a medical VQA expert. Your goal is to identify the CORRECT answer to a medical question about a radiology or pathology image.
Important: medical images follow radiological convention — the patient's RIGHT side appears on the LEFT of the image, and vice versa.
Your final answer should be short and precise, without explanation or additional text."""

SYSTEM_GEN_INCORRECT = """\
You are a medical VQA expert playing the role of a distractor selector. Your goal is to identify a WRONG answer to a medical question about a radiology or pathology image.
Your final answer should be short and precise, without explanation or additional text."""

REPHRASE_PROMPT = (
    "You are a medical VQA assistant.\n"
    "Answer the question correctly, but use a DIFFERENT phrasing or synonym "
    "than the most obvious answer.\n"
    "For example, if the obvious answer is 'liver', you might say 'hepatic organ'.\n"
    "Output ONLY the answer — no explanation."
)

SYSTEM_VERIFIER = """\
You are a medical Visual Question Answering (VQA) expert. You will be given a clinical question (text), a medical image, and a student's proposed answer. Verify whether the student's answer correctly answers the question using evidence from the image and the question. Follow radiological convention: the patient's RIGHT side appears on the LEFT side of the image and the patient's LEFT side appears on the RIGHT side of the image. Determine the correct answer internally, then judge the student's answer. Only mark incorrect if the answer is wrong, contradicts the image/question, is too vague to resolve what was asked, or adds specific claims that are unsupported or incorrect. Output exactly one token: C (for correct) or I (for incorrect), no punctuation, no extra text."""

SYSTEM_JUDGE = """\
You are a strict medical visual question answering evaluator.
You are given: a medical question, an image, a ground truth answer (GT), and a predicted answer (Pred).
Decide whether Pred is medically correct for the question.

Rules:
- Accept synonyms, standard abbreviations, and equivalent phrasing.
- If the question requires laterality, location, count, size, grade, or modality details, Pred must match those specifics.
- Pred may be more specific than GT if it logically entails GT.
- If GT is more specific than what the question asks, a less specific but still correct answer is correct.
- Do NOT accept vague answers when the question explicitly demands specificity.
- If Pred contains additional clinically false statements that contradict the image/question, mark incorrect (even if part matches GT).
- If Pred is non-committal (e.g., "unclear", "can't tell") while GT is determinate, mark incorrect.
- Use the image to reject answers that are clearly implausible or contradicted; question scope remains primary.

Output exactly one word: correct or incorrect."""