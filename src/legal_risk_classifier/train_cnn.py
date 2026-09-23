"""CLI: train the TextCNN baseline.

Establishes the floor the pretrained models have to clear. It trains on CPU in
minutes, which also makes it the fastest way to check that the data pipeline
carries real signal: a model that cannot learn from these chunks would point
at the labels, not the architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .chunking import chunk_documents
from .cuad import load_documents
from .datasets import WordChunkDataset, label_matrix
from .labels import NUM_LABELS
from .metrics import compute_metrics, positive_weights
from .splits import load_splits, select
from .textcnn import TextCNN
from .training import TrainingConfig, predict, resolve_device, train
from .vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--documents", type=Path, default=Path("data/processed/documents.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/cnn"))
    parser.add_argument("--window", type=int, default=300, help="chunk size in words")
    parser.add_argument("--overlap", type=int, default=100, help="chunk overlap in words")
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit_documents", type=int, default=0, help="smoke-test on a subset")
    parser.add_argument(
        "--no_pos_weight",
        action="store_true",
        help="disable class balancing, to demonstrate the collapse it prevents",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.documents)
    assignment = load_splits(args.splits)
    if args.limit_documents:
        documents = documents[: args.limit_documents]

    chunked = {
        split: chunk_documents(
            select(documents, assignment, split), window=args.window, overlap=args.overlap
        )
        for split in ("train", "val", "test")
    }
    print({split: len(chunks) for split, chunks in chunked.items()}, flush=True)

    # Built on the training split only; the corpus-wide vocabulary would leak.
    vocab = Vocabulary.build([chunk.text for chunk in chunked["train"]], min_freq=args.min_freq)
    print(f"vocabulary: {len(vocab):,} types", flush=True)

    loaders = {
        split: DataLoader(
            WordChunkDataset(chunks, vocab, max_tokens=args.max_tokens),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
        )
        for split, chunks in chunked.items()
    }

    model = TextCNN(
        vocab_size=len(vocab),
        num_labels=NUM_LABELS,
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        kernel_sizes=tuple(args.kernel_sizes),
        dropout=args.dropout,
    )

    pos_weight = None if args.no_pos_weight else positive_weights(label_matrix(chunked["train"]))
    if pos_weight is not None:
        print("pos_weight: " + ", ".join(f"{w:.0f}x" for w in pos_weight), flush=True)

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = train(model, loaders["train"], loaders["val"], config, pos_weight, args.output_dir)
    vocab.save(args.output_dir / "vocab.json")

    device = resolve_device(args.device)
    y_true, y_prob = predict(model, loaders["test"], device)
    summary = {
        "model": "textcnn",
        "chunking": {"window": args.window, "overlap": args.overlap},
        "chunks": {split: len(chunks) for split, chunks in chunked.items()},
        "vocab_size": len(vocab),
        "pos_weight": None if pos_weight is None else pos_weight.tolist(),
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "test_at_0.5": compute_metrics(y_true, y_prob, thresholds=0.5),
        "test_at_tuned_thresholds": compute_metrics(y_true, y_prob, thresholds=result.thresholds),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tuned = summary["test_at_tuned_thresholds"]
    print(f"\ntest macro F1 @0.5   {summary['test_at_0.5']['macro_f1']:.3f}")
    print(f"test macro F1 @tuned {tuned['macro_f1']:.3f}")
    print(f"test macro AP        {tuned['macro_average_precision']:.3f}")
    for row in tuned["per_class"]:
        print(
            f"  {row['label']:32s} P {row['precision']:.3f}  R {row['recall']:.3f}  "
            f"F1 {row['f1']:.3f}  AP {row['average_precision']:.3f}  n={row['support']}"
        )
    print(f"\nsaved to {args.output_dir}")


if __name__ == "__main__":
    main()
