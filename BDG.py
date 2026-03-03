"""
Bayesian Decoding Game (BDG) with Wasserstein-1 convergence.

Contains:
  - generator_init: score candidates via generator log-probs
  - verifier_init: score candidates via verifier log-probs
  - bdg_wasserstein: the iterative BDG game loop
  - BDGResult dataclass
"""
from __future__ import annotations
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    ExperimentConfig,
    SYSTEM_GEN_CORRECT, SYSTEM_GEN_INCORRECT, SYSTEM_VERIFIER,
)
from utils import (
    ImageInput,
    safe_normalize, softmax_from_logs, entropy, ranks_from_scores,
    reverse_distribution_by_rank, wasserstein_1, wasserstein_proximal,
)

log = logging.getLogger(__name__)


# =========================================================================
# Prompt helpers
# =========================================================================
def _gen_user(question: str, Y: List[str], signal: bool) -> str:
    if signal:
        return f"{question}\n\nAnswer: "
    else:
        return f"{question}\n\nWrong answer: "


def _verifier_user(question: str, candidate: str) -> str:
    return (f"Question:\n{question}\n\nCandidate answer:\n{candidate}\n\n"
            f"Verdict (correct/incorrect): ")


# =========================================================================
# Generator init
# =========================================================================
def generator_init(
    question: str, Y: List[str],
    backend, image: ImageInput, cfg: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score candidates via generator log-probs.

    Returns (G_correct, G_incorrect, contrastive_scores) — each shape (n,).
    """
    lps_c = backend.logprob_completions(
        SYSTEM_GEN_CORRECT, _gen_user(question, Y, True), Y,
        image=image, length_norm=cfg.use_length_norm,
    )
    lps_i = backend.logprob_completions(
        SYSTEM_GEN_INCORRECT, _gen_user(question, Y, False), Y,
        image=image, length_norm=cfg.use_length_norm,
    )

    scd_c = np.array([
        lp_c - np.logaddexp(lp_c, lp_i)
        for lp_c, lp_i in zip(lps_c, lps_i)
    ])
    scores_c = np.array(lps_c)

    G_c = softmax_from_logs(scores_c)
    G_i = (reverse_distribution_by_rank(G_c) if cfg.reverse_incorrect
           else softmax_from_logs(np.array(lps_i)))

    scd_G_c = softmax_from_logs(scd_c)

    if cfg.use_contrastive_norm:
        scd_G_i = (reverse_distribution_by_rank(scd_G_c) if cfg.reverse_incorrect
                    else softmax_from_logs(np.array([
                        lp_i - np.logaddexp(lp_c, lp_i)
                        for lp_c, lp_i in zip(lps_c, lps_i)
                    ])))
        return scd_G_c, scd_G_i, G_c
    else:
        return G_c, G_i, scd_G_c


# =========================================================================
# Verifier init
# =========================================================================
def verifier_init(
    question: str, Y: List[str],
    backend, image: ImageInput, cfg: ExperimentConfig,
) -> np.ndarray:
    """
    Returns V — shape (n, 2) where V[i] = [P(correct|y_i), P(incorrect|y_i)].
    """
    user_texts = [_verifier_user(question, y) for y in Y]

    lps_c = backend.logprob_prompts(
        SYSTEM_VERIFIER, user_texts, "C",
        image=image, length_norm=cfg.use_length_norm,
    )
    lps_i = backend.logprob_prompts(
        SYSTEM_VERIFIER, user_texts, "I",
        image=image, length_norm=cfg.use_length_norm,
    )

    log_sD = np.array(list(zip(lps_c, lps_i)), dtype=np.float64)

    if cfg.verifier_y_calibration:
        m = np.max(log_sD, axis=0, keepdims=True)
        logZ = np.log(np.sum(np.exp(log_sD - m), axis=0, keepdims=True)) + m
        log_sD = log_sD - logZ

    V = np.zeros_like(log_sD)
    for i in range(len(Y)):
        V[i] = softmax_from_logs(log_sD[i])
    return V


# =========================================================================
# BDG Result
# =========================================================================
@dataclass
class BDGResult:
    answer: str
    answer_idx: int
    aG_correct: np.ndarray
    aG_incorrect: np.ndarray
    aV: np.ndarray
    n_iters: int
    converged: bool
    converge_reason: str
    history: List[Dict]
    w1_final: float


# =========================================================================
# BDG Game Loop
# =========================================================================
def bdg_wasserstein(
    Y: List[str],
    G_c_init: np.ndarray,
    G_i_init: np.ndarray,
    V_init: np.ndarray,
    D_semantic: np.ndarray,
    cfg: ExperimentConfig,
    *,
    eps: float = 1e-6,
    verbose: bool = False,
    save_plot: str = None,
) -> BDGResult:
    """
    Wasserstein-BDG: Bayesian Decoding Game with W1 convergence criterion.

    Convergence when ALL of:
      1. sigma-separation (both G and V)
      2. W1(p_G, p_V) < w_epsilon
    OR fallback:
      1. sigma-separation
      2. exact order match + L1 gap < delta
    """
    n = len(Y)
    assert G_c_init.shape == (n,)
    assert G_i_init.shape == (n,)
    assert V_init.shape == (n, 2)
    assert D_semantic.shape == (n, n)

    aG_c = safe_normalize(G_c_init, eps=eps)
    aG_i = safe_normalize(G_i_init, eps=eps)
    aV = safe_normalize(V_init, axis=1, eps=eps)

    eta_G, eta_V = cfg.eta_G, cfg.eta_V
    lambda_G, lambda_V = cfg.lambda_G, cfg.lambda_V
    use_w = cfg.use_wasserstein

    history = []
    t_hist = [0]
    rank_G_hist = [ranks_from_scores(aG_c)]
    rank_V_hist = [ranks_from_scores(aV[:, 0])]
    H_G_hist = [entropy(aG_c)]
    H_V_hist = [entropy(aV[:, 0] / aV[:, 0].sum())]
    gap_hist = [0.0]
    w1_hist = [0.0]

    converged = False
    converge_reason = "max_iter"

    for t in range(1, cfg.max_iter + 1):
        # ── Markovian belief update ──────────────────────────────────
        bG_c = safe_normalize(aV[:, 0], eps=eps)
        bG_i = safe_normalize(aV[:, 1], eps=eps)
        bV = safe_normalize(np.stack([aG_c, aG_i], axis=1), axis=1, eps=eps)

        # ── Generator update ─────────────────────────────────────────
        alpha_t = np.clip(cfg.w_alpha * (1.0 - (t / 50)), 0.0, cfg.w_alpha)
        denom_G = 1.0 / (eta_G * t) + lambda_G

        logits_c = (0.5 * bG_c + lambda_G * np.log(np.clip(aG_c, eps, None))) / denom_G
        aG_c_kl = softmax_from_logs(logits_c)
        aG_c_new = (wasserstein_proximal(aG_c_kl, aG_c, D_semantic, alpha_t, cfg.w_reg)
                     if cfg.use_wasserstein_update else aG_c_kl)

        if cfg.reverse_incorrect:
            aG_i_new = reverse_distribution_by_rank(aG_c_new)
        else:
            logits_i = (0.5 * bG_i + lambda_G * np.log(np.clip(aG_i, eps, None))) / denom_G
            aG_i_kl = softmax_from_logs(logits_i)
            aG_i_new = (wasserstein_proximal(aG_i_kl, aG_i, D_semantic, alpha_t, cfg.w_reg)
                         if cfg.use_wasserstein_update else aG_i_kl)

        # ── Verifier update ──────────────────────────────────────────
        denom_V = 1.0 / (eta_V * t) + lambda_V
        logits_V = (0.5 * bV + lambda_V * np.log(np.clip(aV, eps, None))) / denom_V
        aV_new = np.zeros_like(aV)
        for i in range(n):
            aV_new[i] = softmax_from_logs(logits_V[i])

        # ── Convergence checks ───────────────────────────────────────
        sep_G = np.min(np.abs(aG_c_new - aG_i_new))
        sep_V = np.min(np.abs(aV_new[:, 0] - aV_new[:, 1]))
        sigma_ok = sep_G > cfg.sigma and sep_V > cfg.sigma

        p_G = aG_c_new
        p_V = aV_new[:, 0] / np.sum(aV_new[:, 0])
        w1 = wasserstein_1(p_G, p_V, D_semantic)
        w1_norm = w1 / (sep_V + 1e-6)

        OG = np.argsort(-aG_c_new)
        OV = np.argsort(-aV_new[:, 0])
        order_match = np.array_equal(OG, OV)
        l1_gap = float(np.sum(np.abs(aG_c_new - p_V)))

        # ── Record history ───────────────────────────────────────────
        t_hist.append(t)
        rank_G_hist.append(ranks_from_scores(aG_c_new))
        rank_V_hist.append(ranks_from_scores(aV_new[:, 0]))
        H_G_hist.append(entropy(aG_c_new))
        H_V_hist.append(entropy(p_V))
        gap_hist.append(l1_gap)
        w1_hist.append(w1_norm)
        history.append({
            "t": t, "order_match": order_match, "l1_gap": l1_gap,
            "w1": w1, "sigma_ok": sigma_ok,
            "aG_c": aG_c_new.copy(), "aV_norm": p_V.copy(),
        })

        if verbose and t % 50 == 0:
            log.info(f"  [t={t:03d}] W1={w1:.4f} L1={l1_gap:.4f} "
                     f"order={order_match} σ={sigma_ok}")

        aG_c, aG_i, aV = aG_c_new, aG_i_new, aV_new

        # ── Check convergence ────────────────────────────────────────
        if sigma_ok:
            if use_w and w1_norm < cfg.w_epsilon:
                converged = True
                converge_reason = "wasserstein"
                break
            if not use_w and order_match and l1_gap < cfg.delta_l1:
                converged = True
                converge_reason = "order+gap"
                break

    best_idx = int(np.argmax(aG_c))

    if save_plot:
        _plot_diagnostics(
            Y, t_hist, rank_G_hist, rank_V_hist,
            H_G_hist, H_V_hist, gap_hist, w1_hist,
            converge_reason, save_plot,
        )

    return BDGResult(
        answer=Y[best_idx], answer_idx=best_idx,
        aG_correct=aG_c, aG_incorrect=aG_i, aV=aV,
        n_iters=t_hist[-1], converged=converged,
        converge_reason=converge_reason, history=history,
        w1_final=w1_hist[-1],
    )


# =========================================================================
# Diagnostics plot
# =========================================================================
def _plot_diagnostics(Y, t_hist, rank_G, rank_V, H_G, H_V, gap, w1,
                      reason, path):
    rank_G = np.stack(rank_G)
    rank_V = np.stack(rank_V)
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    ax = axes[0]
    lines = []
    for j, label in enumerate(Y):
        ln, = ax.plot(t_hist, rank_G[:, j], label=label[:20])
        lines.append(ln)
    ax.set_ylabel("Rank"); ax.set_title("Generator ranks"); ax.invert_yaxis()

    ax = axes[1]
    for j, label in enumerate(Y):
        ax.plot(t_hist, rank_V[:, j], label=label[:20], color=lines[j].get_color())
    ax.set_ylabel("Rank"); ax.set_title("Verifier ranks"); ax.invert_yaxis()
    ax.legend(ncol=2, fontsize=7)

    ax = axes[2]
    ax.plot(t_hist, H_G, label="H(G correct)")
    ax.plot(t_hist, H_V, label="H(V correct)", ls="--")
    ax.set_ylabel("Entropy"); ax.legend()

    ax = axes[3]
    ax.plot(t_hist, gap, label="L1 gap"); ax.set_ylabel("L1"); ax.legend()

    ax = axes[4]
    ax.plot(t_hist, w1, label="W1 distance", color="red")
    ax.set_xlabel("Iteration"); ax.set_ylabel("W1")
    ax.set_title(f"Converged: {reason}"); ax.legend()

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)