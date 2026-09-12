from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ClauseDataset, expand_rows_with_chunking, load_rows_jsonl, tokenizer_for_model
from .labels import TARGET_LABELS
from .metrics import compute_multilabel_metrics
from .models import build_multilabel_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained legal clause classifier.")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--rows_jsonl", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = tokenizer_for_model(str(args.model_dir))
    model = build_multilabel_model(str(args.model_dir), num_labels=len(TARGET_LABELS)).to(args.device)
    model.eval()

    rows = load_rows_jsonl(args.rows_jsonl)
    chunked = expand_rows_with_chunking(rows, tokenizer, args.max_length, args.stride)
    ds = ClauseDataset(chunked, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            y_prob.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(batch["labels"].cpu().numpy())

    y_true_arr = np.concatenate(y_true, axis=0)
    y_prob_arr = np.concatenate(y_prob, axis=0)
    metrics = compute_multilabel_metrics(y_true_arr, y_prob_arr, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

