# Legal Clause Risk Classifier (CUAD, 6 Labels)

This repo is a runnable template for the attached project: multi-label classification of high-risk commercial contract clauses using BERT / Legal-BERT / Longformer with long-document chunking.

## Target labels

1. Cap on Liability
2. Non-Compete
3. License Grant
4. Audit Rights
5. Termination for Convenience
6. Insurance

## What is implemented

- CUAD JSON preprocessing into training rows (`doc_id`, `text`, `labels[]`)
- Sliding-window chunking for long contracts (`max_length`, `stride`)
- Multi-label sequence classification (`sigmoid + BCE`) with:
  - `bert-base-uncased`
  - `nlpaueb/legal-bert-base-uncased`
  - `allenai/longformer-base-4096`
- Document-level split to reduce leakage
- Per-class + macro precision/recall/F1
- Document inference with chunk max-pooling

## Project layout

```text
src/legal_risk_classifier/
  labels.py       # label space + label normalization
  data.py         # CUAD parsing, chunking, dataset, splits
  models.py       # model registry + HF model construction
  metrics.py      # multi-label metrics
  prepare.py      # CUAD JSON -> JSONL
  train.py        # training entrypoint
  evaluate.py     # standalone evaluation entrypoint
  infer.py        # document prediction entrypoint
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 1. Prepare data

Convert `CUAD_v1.json` into row-level JSONL:

```bash
legal-risk-prepare \
  --cuad_json /path/to/CUAD_v1.json \
  --output_jsonl data/cuad_rows.jsonl
```

To keep only rows with at least one of the 6 labels:

```bash
legal-risk-prepare \
  --cuad_json /path/to/CUAD_v1.json \
  --output_jsonl data/cuad_rows_labeled_only.jsonl \
  --drop_unlabeled
```

## 2. Train

Legal-BERT baseline:

```bash
legal-risk-train \
  --rows_jsonl data/cuad_rows.jsonl \
  --model legal-bert \
  --output_dir outputs/legal_bert \
  --max_length 512 \
  --stride 64 \
  --epochs 3 \
  --batch_size 8
```

Longformer long-context run:

```bash
legal-risk-train \
  --rows_jsonl data/cuad_rows.jsonl \
  --model longformer \
  --output_dir outputs/longformer \
  --max_length 4096 \
  --stride 512 \
  --epochs 3 \
  --batch_size 1
```

## 3. Evaluate

```bash
legal-risk-eval \
  --model_dir outputs/legal_bert/best_model \
  --rows_jsonl data/cuad_rows.jsonl \
  --max_length 512 \
  --stride 64
```

## 4. Inference (single contract text file)

```bash
legal-risk-infer \
  --model_dir outputs/legal_bert/best_model \
  --text_file /path/to/contract.txt \
  --max_length 512 \
  --stride 64 \
  --threshold 0.5
```

## Experiment mapping to your project

- **Problem 1 (Embedding):** run `--model bert` vs `--model legal-bert` and compare macro/per-class F1.
- **Problem 2 (Long docs):** run `--model legal-bert --max_length 512` vs `--model longformer --max_length 4096 --stride 512`.

