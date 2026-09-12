# Data & Training Flow

## 1. Dataset preparation
- `legal-risk-prepare` reads `CUAD_v1.json`.
- Extracts paragraphs and maps to 6 target clause labels.
- Writes JSONL rows: `doc_id`, `text`, `labels`.

## 2. Data split
- Rows are split by root document ID (not by chunk).
- Prevents leakage across train/val/test.

## 3. Chunk expansion
- Rows expand into overlapping chunks using tokenizer-aware splitting.
- Chunk labels inherit from parent row labels.

## 4. Training
- `legal-risk-train` builds multi-label classifier head.
- Optimizes BCE objective via AdamW.
- Saves best checkpoint by validation macro F1.

## 5. Evaluation
- `legal-risk-eval` computes per-class and macro precision/recall/F1.
- Output is printed as JSON for reporting/benchmarking.

