"""
Wasserstein-BDG for Open-Ended Medical VQA — Main Evaluation Script
====================================================================
Supports mixed model families: generator and verifier can be different
architectures (e.g. --gen-backend gemma --ver-backend qwen).

Usage:
    # Same Qwen model for gen + ver:
    python main.py --gen-backend qwen --gen-model Qwen/Qwen3-VL-4B-Instruct

    # Mixed: MedGemma generator, Qwen verifier:
    python main.py \
        --gen-backend gemma --gen-model google/medgemma-4b-it --gen-device cuda:0 \
        --ver-backend qwen  --ver-model Qwen/Qwen3-VL-4B-Instruct --ver-device cuda:1
"""
from __future__ import annotations
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from datasets import load_dataset

from config import ExperimentConfig
from utils import SapBERTCLSEmbedder, evaluate_answer, vlm_judge, JUDGE_API_ERROR
from BDG import generator_init, verifier_init, bdg_wasserstein

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

METHODS = ["greedy_G", "verifier_D", "contrastive_SC", "bdg_classic", "bdg_wasserstein"]


def _eval_single_method(method, pred, question, gt_answer, image, embedder, cfg):
    ev = evaluate_answer(pred, gt_answer, embedder, cfg.semantic_match_threshold)
    judge_score = vlm_judge(question, gt_answer, pred, image)
    return method, ev, judge_score


# =========================================================================
# Backend factory — one backend per (family, model, device)
# =========================================================================
def _load_backend(family: str, model_name: str, device: str):
    """Instantiate a single backend."""
    import torch
    if family == "qwen":
        from QwenBackend import QwenBackend
        return QwenBackend(model_name, device=device, dtype=torch.float16)
    elif family == "gemma":
        from GemmaBackend import GemmaBackend
        return GemmaBackend(model_name, device=device, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unknown backend family: {family}")


def load_backends(args):
    """
    Load generator and verifier backends.
    If both use the same family+model+device, share the instance.
    """
    log.info(f"Generator: {args.gen_backend} / {args.gen_model} @ {args.gen_device}")
    log.info(f"Verifier:  {args.ver_backend} / {args.ver_model} @ {args.ver_device}")

    gen_backend = _load_backend(args.gen_backend, args.gen_model, args.gen_device)

    same_setup = (args.gen_backend == args.ver_backend
                  and args.gen_model == args.ver_model
                  and args.gen_device == args.ver_device)
    if same_setup:
        log.info("Generator and verifier share the same model instance.")
        ver_backend = gen_backend
    else:
        ver_backend = _load_backend(args.ver_backend, args.ver_model, args.ver_device)

    return gen_backend, ver_backend


# =========================================================================
# Evaluation loop
# =========================================================================
def run_evaluation(
    cfg: ExperimentConfig,
    gen_backend,
    ver_backend,
    embedder,
    dataset,
    n_runs: int = 5,
    max_samples: int = 200,
    do_plot: bool = False,
    plot_dir_classic: str = "bdg_c_runs",
    plot_dir_wass: str = "bdg_w_runs",
):
    all_run_results = []

    for run in range(n_runs):
        RUN_SEED = 9000 + run * 1000
        counts = {m: {"exact": 0, "soft": 0, "f1_sum": 0.0, "sem_sum": 0.0,
                       "judge_sum": 0.0, "judge_n": 0} for m in METHODS}
        n_evaluated = 0
        n_inconsistent = 0
        iter_w1, iter_classic, iter_counter = 0, 0, 0

        for q_idx, sample in enumerate(dataset):
            gt_answer = sample["answer"].strip()
            if gt_answer.lower() in ("yes", "no"):
                continue

            question = sample["question"]
            image = sample.get("image")
            if image is not None and not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")

            # ── 1) Build candidates (gen_backend) ────────────────────
            try:
                cand = gen_backend.build_candidate_set(
                    question, image, cfg, embedder,
                    base_seed=RUN_SEED + q_idx,
                )
            except RuntimeError as e:
                log.warning(f"[{q_idx}] Skipping: {e}")
                continue

            Y, D, n = cand.Y, cand.D, len(cand.Y)

            if n < 2:
                answers = {m: Y[0] for m in METHODS}
            else:
                # ── 2) Generator init (gen_backend) ──────────────────
                G_c, G_i, scd = generator_init(
                    question, Y, gen_backend, image, cfg,
                )
                # ── 3) Verifier init (ver_backend — may differ!) ─────
                V = verifier_init(
                    question, Y, ver_backend, image, cfg,
                )

                # ── 4) Classic BDG ───────────────────────────────────
                cfg_classic = ExperimentConfig(
                    **{**cfg.__dict__,
                       "use_wasserstein": False,
                       "use_wasserstein_update": False}
                )

                if do_plot:
                    res_classic = bdg_wasserstein(
                        Y, G_c, G_i, V, D, cfg_classic,
                        save_plot=f"{plot_dir_classic}/bdg_c_r{run}_{q_idx:04d}.png",
                    )

                    # ── 5) Wasserstein BDG ───────────────────────────────
                    res_wbdg = bdg_wasserstein(
                        Y, G_c, G_i, V, D, cfg,
                        save_plot=f"{plot_dir_wass}/bdg_w_r{run}_{q_idx:04d}.png",
                    )
                else:
                    res_classic = bdg_wasserstein(
                        Y, G_c, G_i, V, D, cfg_classic,
                        save_plot=None,
                    )

                    # ── 5) Wasserstein BDG ───────────────────────────────
                    res_wbdg = bdg_wasserstein(
                        Y, G_c, G_i, V, D, cfg,
                        save_plot=None,
                    )

                answers = {
                    "greedy_G": Y[int(np.argmax(G_c))],
                    "verifier_D": Y[int(np.argmax(V[:, 0]))],
                    "contrastive_SC": Y[int(np.argmax(scd))],
                    "bdg_classic": res_classic.answer,
                    "bdg_wasserstein": res_wbdg.answer,
                }
                iter_w1 += res_wbdg.n_iters
                iter_classic += res_classic.n_iters
                iter_counter += 1

            # ── 6) Evaluate ──────────────────────────────────────────
            n_evaluated += 1
            if answers["greedy_G"] != answers["verifier_D"]:
                n_inconsistent += 1

            with ThreadPoolExecutor(max_workers=len(answers)) as executor:
                futures = {
                    executor.submit(
                        _eval_single_method, m, pred, question,
                        gt_answer, image, embedder, cfg,
                    ): m
                    for m, pred in answers.items()
                }
                for future in as_completed(futures):
                    method, ev, judge_score = future.result()
                    counts[method]["exact"] += int(ev["exact"])
                    counts[method]["soft"] += int(ev["soft_match"])
                    counts[method]["f1_sum"] += ev["f1"]
                    counts[method]["sem_sum"] += ev["sem_sim"]
                    if np.isnan(judge_score):
                        log.warning(f"[{q_idx}] Judge API error for {method}")
                    else:
                        counts[method]["judge_sum"] += judge_score
                        counts[method]["judge_n"] += 1

            # ── 7) Progress log ──────────────────────────────────────
            if n_evaluated % 10 == 0:
                avg_w1_iter = iter_w1 / max(iter_counter, 1)
                avg_classic_iter = iter_classic / max(iter_counter, 1)
                log.info(f"── [{q_idx}] n_eval={n_evaluated} "
                         f"inconsistency={n_inconsistent / n_evaluated:.2%} ──")
                for m in METHODS:
                    c = counts[m]
                    ne = max(n_evaluated, 1)
                    nj = max(c["judge_n"], 1)
                    log.info(
                        f"  {m:20s}  exact={c['exact']/ne:.3f}  "
                        f"soft={c['soft']/ne:.3f}  F1={c['f1_sum']/ne:.3f}  "
                        f"sem={c['sem_sum']/ne:.3f}  "
                        f"judge={c['judge_sum']/nj:.3f}(n={c['judge_n']})  "
                        f"iter: w1={avg_w1_iter:.1f}|cl={avg_classic_iter:.1f}"
                    )

            if n_evaluated >= max_samples:
                break

        # ── Run report ───────────────────────────────────────────────
        run_result = {"run": run, "n_evaluated": n_evaluated}
        ne = max(n_evaluated, 1)
        log.info(f"\n{'=' * 60}")
        log.info(f"FINAL RUN {run} (n={n_evaluated}, "
                 f"inconsistency={n_inconsistent / ne:.2%})")
        log.info(f"{'=' * 60}")
        for m in METHODS:
            c = counts[m]
            run_result[m] = {
                "exact": c["exact"] / ne,
                "soft": c["soft"] / ne,
                "f1": c["f1_sum"] / ne,
                "sem": c["sem_sum"] / ne,
                "judge": c["judge_sum"] / ne,
            }
            log.info(
                f"  {m:20s}  exact={run_result[m]['exact']:.4f}  "
                f"soft={run_result[m]['soft']:.4f}  "
                f"F1={run_result[m]['f1']:.4f}  "
                f"sem={run_result[m]['sem']:.4f}  "
                f"judge={run_result[m]['judge']:.4f}"
            )
        all_run_results.append(run_result)

    # ── Aggregate ────────────────────────────────────────────────────
    metrics = ["exact", "soft", "f1", "sem", "judge"]
    log.info(f"\n{'=' * 60}")
    log.info(f"AGGREGATE ({len(all_run_results)} runs)")
    log.info(f"{'=' * 60}")
    for m in METHODS:
        parts = []
        for metric in metrics:
            vals = [r[m][metric] for r in all_run_results]
            parts.append(f"{metric}={np.mean(vals):.4f}±{np.std(vals):.4f}")
        log.info(f"  {m:20s}  " + "  ".join(parts))

    return all_run_results


# =========================================================================
# CLI
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Wasserstein-BDG Medical VQA — supports mixed model families",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Same Qwen model for gen + ver:
  python main.py --gen-backend qwen --gen-model Qwen/Qwen3-VL-4B-Instruct

  # Mixed: MedGemma generator, Qwen verifier:
  python main.py \\
      --gen-backend gemma --gen-model google/medgemma-4b-it --gen-device cuda:0 \\
      --ver-backend qwen  --ver-model Qwen/Qwen3-VL-4B-Instruct --ver-device cuda:1
""",
    )
    # ── Generator args ───────────────────────────────────────────────
    parser.add_argument("--gen-backend", choices=["qwen", "gemma"], required=True,
                        help="Model family for the generator")
    parser.add_argument("--gen-model", required=True,
                        help="HF model name for the generator")
    parser.add_argument("--gen-device", default="cuda:0")

    # ── Verifier args (defaults to same as generator) ────────────────
    parser.add_argument("--ver-backend", choices=["qwen", "gemma"], default=None,
                        help="Model family for the verifier (default: same as --gen-backend)")
    parser.add_argument("--ver-model", default=None,
                        help="HF model name for the verifier (default: same as --gen-model)")
    parser.add_argument("--ver-device", default=None,
                        help="Device for verifier (default: same as --gen-device)")

    # ── Experiment args ──────────────────────────────────────────────
    parser.add_argument("--dataset", choices=["vqa-rad", "path-vqa"], default="vqa-rad")
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    # ── Fill verifier defaults from generator ────────────────────────
    if args.ver_backend is None:
        args.ver_backend = args.gen_backend
    if args.ver_model is None:
        args.ver_model = args.gen_model
    if args.ver_device is None:
        args.ver_device = args.gen_device

    # ── Config (pure hyperparams, no model names) ────────────────────
    cfg = ExperimentConfig(use_wasserstein_update=False)

    # ── Load backends ────────────────────────────────────────────────
    gen_backend, ver_backend = load_backends(args)

    # ── Load embedder ────────────────────────────────────────────────
    embedder = SapBERTCLSEmbedder(cfg.embedder_name)

    # ── Load dataset ─────────────────────────────────────────────────
    if args.dataset == "vqa-rad":
        ds = load_dataset("flaviagiammarino/vqa-rad")["test"]
    else:
        ds = load_dataset("flaviagiammarino/path-vqa")["test"]

    # ── Run ──────────────────────────────────────────────────────────
    run_evaluation(
        cfg, gen_backend, ver_backend, embedder, ds,
        n_runs=args.n_runs, max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()