# iCARE Clinical Summarization Reproduction

Checkpointed scripts for reproducing the iCARE Gemma 7B clinical summarization
baseline and scoring saved summaries. Zero-shot and one-shot runs are kept
separate. Optional anonymized-input evaluation verifies completed drafts and
source/output hashes before they enter evaluation.

This repository contains the reproduction tooling only. Clinical transcripts,
expert references, generated summaries, weights and private anonymizer outputs
are not distributed.

## Setup

Use Python 3.11 and install a PyTorch build suited to your hardware:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest test_anonymized_inputs -v
```

Acquire the upstream project and any required data under its own terms:

```bash
git clone https://github.com/ai4mhx/iCARE.git upstream
git -C upstream checkout ce36a39eeb6da1bfb1f572730ad1f0cf0f68f02b
```

The scripts expect upstream `Baselines.py`, `Train_Test_Builder.py` and the
upstream data layout. Place an authorized `google/gemma-7b-it` checkpoint in
`models/gemma-7b-it`. Model access and its license are separate requirements.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python run_baseline.py --setting zeroshot --limit 1
CUDA_VISIBLE_DEVICES=0 python run_baseline.py --setting oneshot --limit 1
```

Select one allocated GPU. Omit `--limit` only after inspecting a pilot. Saved
answers resume after prompt-fingerprint checks. The protocol uses greedy
BF16 generation, at most 100 new tokens per aspect, upstream raw prompts and
the upstream train/test split. It preserves duplicate expert-reference rows.

## Score saved outputs

```bash
python download_metrics.py
python score_results.py
```

Metric downloads are explicit and their URLs/hashes are recorded locally.
Scoring attempts BLEU, METEOR, ROUGE-L, BERTScore, BLEURT and InfoLM on CPU.
BLEURT additionally needs its upstream package and TensorFlow/legacy Keras;
install those in a compatible environment. Missing metrics remain named errors,
never zeros. InfoLM here is a distance: lower is better.

`compare_scores.py` and `make_colored_comparison.py` accept local reference tables
named `reference_scores.csv` and `reference_scores_oneshot.csv`; these are not
bundled. Required columns are `feature,bleu,meteor,rouge,bertscore,bleurt,infolm`.
The original table-to-setting assignment was provisional; verify captions before
claiming reproduction. Output files are written under ignored `results*/`.

## Optional anonymized-input evaluation

`run_baseline.py --anonymized-root /path/to/completed/drafts` uses locally
supplied drafts. `--anonymize-example` also changes the one-shot demonstration;
without it, the original training example remains. Incomplete drafts, hash
mismatches and missing required cases are rejected. Tests use synthetic text.
No private batch coordinator or production anonymizer is included.

## Limits and license

Upstream output slicing is retained alongside token-boundary extraction so
boundary differences remain visible. Hardware, dependencies, input ordering and
model versions can change results. Automated metrics are not clinical validation.
Original wrapper code is MIT licensed; upstream iCARE, metric implementations,
model weights and datasets retain their own terms. See `THIRD_PARTY_NOTICES.md`.
