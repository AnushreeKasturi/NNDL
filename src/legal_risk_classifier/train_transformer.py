"""CLI: fine-tune one pretrained encoder on the chunked corpus.

Every model is trained through the same loop, on the same splits, with the
same loss weighting and the same threshold tuning. Chunk width is the one
thing that varies, and only because it is bounded by the model's context
window, which is the variable the Problem 2 experiment is about.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .chunking import chunk_documents
from .cuad import load_documents
from .datasets import TokenizedChunkDataset, label_matrix
from .labels import NUM_LABELS
from .metrics import compute_metrics, positive_weights
from .pretrained import MODELS, PretrainedClassifier, load_tokenizer, resolve
from .splits import load_splits, select
from .training import TrainingConfig, predict, resolve_device, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="legal-bert", help=f"{' | '.join(MODELS)} | any HF id")
    parser.add_argument("--documents", type=Path, default=Path("data/processed/documents.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--window", type=int, default=0, help="override the model's chunk width")
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accumulation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="trade speed for memory")
    parser.add_argument("--limit_documents", type=int, default=0, help="smoke-test on a subset")
    parser.add_argument("--no_pos_weight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = resolve(args.model)
    window = args.window or spec.window_words
    overlap = args.overlap or spec.overlap_words
    max_length = args.max_length or spec.max_length
    output_dir = args.output_dir or Path("outputs") / spec.key

    print(f"{spec.checkpoint}  window={window}w overlap={overlap}w max_length={max_length}t", flush=True)

    documents = load_documents(args.documents)
    assignment = load_splits(args.splits)
    if args.limit_documents:
        documents = documents[: args.limit_documents]

    chunked = {
        split: chunk_documents(
            select(documents, assignment, split), window=window, overlap=overlap
        )
        for split in ("train", "val", "test")
    }
    print({split: len(chunks) for split, chunks in chunked.items()}, flush=True)

    tokenizer = load_tokenizer(spec)
    loaders = {
        split: DataLoader(
            TokenizedChunkDataset(chunks, tokenizer, max_length=max_length),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
        )
        for split, chunks in chunked.items()
    }

    model = PretrainedClassifier(spec, num_labels=NUM_LABELS)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    pos_weight = None if args.no_pos_weight else positive_weights(label_matrix(chunked["train"]))
    if pos_weight is not None:
        print("pos_weight: " + ", ".join(f"{w:.0f}x" for w in pos_weight), flush=True)

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train(model, loaders["train"], loaders["val"], config, pos_weight, output_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    y_true, y_prob = predict(model, loaders["test"], resolve_device(args.device))
    summary = {
        "model": spec.key,
        "checkpoint": spec.checkpoint,
        "chunking": {"window": window, "overlap": overlap, "max_length": max_length},
        "chunks": {split: len(chunks) for split, chunks in chunked.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "pos_weight": None if pos_weight is None else pos_weight.tolist(),
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "thresholds": result.thresholds,
        "test_at_0.5": compute_metrics(y_true, y_prob, thresholds=0.5),
        "test_at_tuned_thresholds": compute_metrics(y_true, y_prob, thresholds=result.thresholds),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tuned = summary["test_at_tuned_thresholds"]
    print(f"\ntest macro F1 @tuned {tuned['macro_f1']:.3f}")
    print(f"test macro AP        {tuned['macro_average_precision']:.3f}")
    for row in tuned["per_class"]:
        print(
            f"  {row['label']:32s} P {row['precision']:.3f}  R {row['recall']:.3f}  "
            f"F1 {row['f1']:.3f}  AP {row['average_precision']:.3f}  n={row['support']}"
        )
    print(f"\nsaved to {output_dir}")


if __name__ == "__main__":
    main()
