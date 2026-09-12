from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import chunk_text, tokenizer_for_model
from .labels import TARGET_LABELS
from .models import build_multilabel_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run document-level inference with chunk max-pooling.")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--text_file", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text_file.read_text(encoding="utf-8")
    tokenizer = tokenizer_for_model(str(args.model_dir))
    model = build_multilabel_model(str(args.model_dir), num_labels=len(TARGET_LABELS)).to(args.device)
    model.eval()

    chunks = chunk_text(text, tokenizer, args.max_length, args.stride)
    if not chunks:
        raise ValueError("No text chunks produced from input file.")

    all_probs = []
    with torch.no_grad():
        for chunk in chunks:
            enc = tokenizer(
                chunk,
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            logits = model(
                input_ids=enc["input_ids"].to(args.device),
                attention_mask=enc["attention_mask"].to(args.device),
            ).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy()[0])

    chunk_probs = np.vstack(all_probs)
    pooled = chunk_probs.max(axis=0)
    decisions = pooled >= args.threshold
    result = {
        "chunk_count": len(chunks),
        "threshold": args.threshold,
        "scores": {label: float(pooled[i]) for i, label in enumerate(TARGET_LABELS)},
        "predicted_labels": [label for i, label in enumerate(TARGET_LABELS) if decisions[i]],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

