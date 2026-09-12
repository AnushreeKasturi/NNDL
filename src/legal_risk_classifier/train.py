from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import (
    ClauseDataset,
    convert_cuad_to_rows,
    expand_rows_with_chunking,
    load_rows_jsonl,
    set_seed,
    split_by_document,
    tokenizer_for_model,
)
from .labels import TARGET_LABELS
from .metrics import compute_multilabel_metrics
from .models import build_multilabel_model, resolve_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train legal clause multi-label classifier.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rows_jsonl", type=Path, help="Prepared JSONL rows: doc_id,text,labels[]")
    source.add_argument("--cuad_json", type=Path, help="CUAD_v1.json file")
    parser.add_argument("--model", type=str, default="legal-bert", help="bert | legal-bert | longformer | HF model id")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop_unlabeled", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float,
) -> dict:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.append(labels)
            y_prob.append(probs)
    y_true_arr = np.concatenate(y_true, axis=0)
    y_prob_arr = np.concatenate(y_prob, axis=0)
    return compute_multilabel_metrics(y_true_arr, y_prob_arr, threshold=threshold)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_name = resolve_model_name(args.model)
    tokenizer = tokenizer_for_model(model_name)

    if args.rows_jsonl:
        rows = load_rows_jsonl(args.rows_jsonl)
    else:
        rows = convert_cuad_to_rows(args.cuad_json, include_unlabeled=not args.drop_unlabeled)

    train_rows, val_rows, test_rows = split_by_document(
        rows,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_chunks = expand_rows_with_chunking(train_rows, tokenizer, args.max_length, args.stride)
    val_chunks = expand_rows_with_chunking(val_rows, tokenizer, args.max_length, args.stride)
    test_chunks = expand_rows_with_chunking(test_rows, tokenizer, args.max_length, args.stride)

    train_ds = ClauseDataset(train_chunks, tokenizer, args.max_length)
    val_ds = ClauseDataset(val_chunks, tokenizer, args.max_length)
    test_ds = ClauseDataset(test_chunks, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_multilabel_model(model_name, num_labels=len(TARGET_LABELS)).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_path = args.output_dir / "best_model"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            pbar.set_postfix(loss=running_loss / max(1, len(pbar)))

        val_metrics = _evaluate(model, val_loader, args.device, args.threshold)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "avg_train_loss": running_loss / max(1, len(train_loader)),
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
        )

    best_model = build_multilabel_model(str(best_path), num_labels=len(TARGET_LABELS)).to(args.device)
    test_metrics = _evaluate(best_model, test_loader, args.device, args.threshold)
    summary = {
        "model": model_name,
        "label_space": TARGET_LABELS,
        "split_sizes": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
            "test_chunks": len(test_chunks),
        },
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
    }
    metrics_file = args.output_dir / "metrics.json"
    metrics_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved model to {best_path}")
    print(f"Saved metrics to {metrics_file}")


if __name__ == "__main__":
    main()

