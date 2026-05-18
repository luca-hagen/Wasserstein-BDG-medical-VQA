# Wasserstein-BDG for Medical Visual Question Answering

Official implementation of **Wasserstein Equilibrium Decoding for Reliable Medical Visual Question Answering**.

This repository is anonymized for double-blind review.

## What this repository does

This code implements a training-free inference-time decoding method for open-ended Medical Visual Question Answering (Med-VQA). Given a medical image and question, the method:

1. samples candidate answers from a vision-language model,
2. scores candidates with generator and verifier prompts,
3. runs Bayesian Decoding Game (BDG) updates,
4. uses SapBERT embeddings to compute semantic distances between candidates,
5. stops with a Wasserstein-1 convergence criterion when generator and verifier agree semantically.

The implementation compares:

- `greedy_G`
- `verifier_D`
- `contrastive_SC`
- `bdg_classic`
- `bdg_wasserstein`

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── BDG.py
├── QwenBackend.py
├── GemmaBackend.py
└── utils.py
```

## Installation

Create an environment:

```bash
conda create -n wbdg-medvqa python=3.10
conda activate wbdg-medvqa
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The current dependencies are:

```text
datasets
matplotlib
numpy
Pillow
requests
scikit-learn
sentence-transformers
torch
transformers
POT
```

**Note:** remove any accidental trailing `EOF` line from `requirements.txt` before installing.

## Configuration

All main configuration options are in `config.py`.

### 1. Hugging Face cache and token

Set your local Hugging Face cache path:

```python
CACHE = "/your/cache/path"
```

If the selected models require authentication, set a Hugging Face token:

```python
os.environ["HF_TOKEN"] = "hf_xxx"
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
```

If your models are public and already cached, this may not be required.

### 2. Optional OpenRouter judge

The VLM-as-judge evaluation uses OpenRouter:

```python
OPENROUTER_API_KEY = "enter your openrouter_key here"
OPENROUTER_JUDGE_MODEL = "x-ai/grok-4-fast"
```

Set your API key if you want to compute judge accuracy:

```python
OPENROUTER_API_KEY = "sk-or-..."
```

If no key is provided, judge scores may be unavailable or returned as `NaN`. This does not affect BDG decoding itself.

### 3. Core hyperparameters

The main experiment parameters are defined in `ExperimentConfig`:

```python
n_candidates = 8
n_oversample = 12
max_sampling_calls = 16
max_new_tokens_sample = 24
temperatures = (0.5, 1.0)
top_p = 0.98
top_k_sample = 100

embedder_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
ground_metric = "cosine"

eta_G = 0.4
eta_V = 0.4
lambda_G = 0.4
lambda_V = 0.4
sigma = 5e-3
max_iter = 500

use_wasserstein = True
w_epsilon = 0.2
delta_l1 = 0.2

use_length_norm = True
semantic_match_threshold = 0.9
```

For the paper setting, `use_wasserstein_update` is set to `False`; Wasserstein is used as a stopping criterion, not as an update regularizer.

## Datasets

The script loads datasets directly via Hugging Face Datasets:

```python
load_dataset("flaviagiammarino/vqa-rad")["test"]
load_dataset("flaviagiammarino/path-vqa")["test"]
```

Supported CLI values:

```text
--dataset vqa-rad
--dataset path-vqa
```

Yes/no questions are skipped automatically:

```python
if gt_answer.lower() in ("yes", "no"):
    continue
```

## Running experiments

All experiments are launched through `main.py`.

### Qwen model

```bash
python main.py \
  --gen-backend qwen \
  --gen-model Qwen/Qwen3-VL-4B-Instruct \
  --gen-device cuda:0 \
  --dataset vqa-rad \
  --n-runs 5 \
  --max-samples 200
```

### Gemma / MedGemma model

```bash
python main.py \
  --gen-backend gemma \
  --gen-model google/medgemma-4b-it \
  --gen-device cuda:0 \
  --dataset path-vqa \
  --n-runs 5 \
  --max-samples 200
```

### Mixed generator and verifier

By default, the verifier uses the same backend, model, and device as the generator. To use a separate verifier:

```bash
python main.py \
  --gen-backend gemma \
  --gen-model google/medgemma-4b-it \
  --gen-device cuda:0 \
  --ver-backend qwen \
  --ver-model Qwen/Qwen3-VL-4B-Instruct \
  --ver-device cuda:1 \
  --dataset vqa-rad \
  --n-runs 5 \
  --max-samples 200
```

### Quick smoke test

```bash
python main.py \
  --gen-backend qwen \
  --gen-model Qwen/Qwen3-VL-4B-Instruct \
  --dataset vqa-rad \
  --n-runs 1 \
  --max-samples 5
```

This checks model loading, dataset loading, candidate generation, BDG updates, and metric computation.

## Output

The script prints intermediate and final metrics to the console.

Every 10 evaluated open-ended examples, it reports:

- exact match,
- soft semantic match,
- F1,
- semantic similarity,
- judge accuracy,
- average Wasserstein-BDG iterations,
- average classic BDG iterations.

At the end, it prints per-run and aggregate results.

The current version does not automatically save JSON or CSV files. To save results, add the following near the end of `run_evaluation` in `main.py`:

```python
import json

with open("results.json", "w") as f:
    json.dump(all_run_results, f, indent=2)
```

## Reproducibility

Candidate generation is stochastic but seeded. Each run uses:

```python
RUN_SEED = 9000 + run * 1000
```

and each question receives an additional question-index offset:

```python
base_seed = RUN_SEED + q_idx
```

Exact reproducibility may still depend on GPU type, CUDA version, PyTorch version, model implementation, and remote judge availability.

## Troubleshooting

### `pip install -r requirements.txt` fails

Check that `requirements.txt` does not contain an accidental final line:

```text
EOF
```

Remove it if present.

### No candidates are generated

Try increasing:

```python
max_sampling_calls
n_oversample
max_new_tokens_sample
```

or relaxing:

```python
max_answer_words
min_answer_chars
```

### Judge scores are missing or `NaN`

Check:

- `OPENROUTER_API_KEY`,
- network access,
- OpenRouter rate limits,
- judge model availability.

Judge evaluation is optional.

### CUDA out of memory

Try:

- using a smaller model,
- reducing `n_candidates`,
- reducing `max_new_tokens_sample`,
- sharing generator and verifier on the same model instance,
- putting generator and verifier on separate GPUs.

## Double-blind review note

This repository is anonymized. Do not add author names, institutional information, personal links, personal email addresses, or non-anonymized citation metadata until the review process is complete.

## Citation

Citation information will be added after review.

## License

A license will be added upon public release.
