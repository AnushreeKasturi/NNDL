"""CLI: turn CUAD_v1.json into the documents file and split assignment.

Chunking deliberately does not happen here. BERT and Longformer want different
window sizes, so chunking is a load-time concern; this step is about extracting
spans once, because parsing the 100MB CUAD JSON is slow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cuad import parse_cuad, save_documents
from .labels import LABEL_NAMES
from .splits import assign_splits, save_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cuad_json", type=Path, default=Path("data/CUAD_v1.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = parse_cuad(args.cuad_json)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    documents_path = args.out_dir / "documents.jsonl"
    splits_path = args.out_dir / "splits.json"

    save_documents(documents, documents_path)
    assignment = assign_splits(
        documents, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    save_splits(assignment, splits_path)

    counts = {name: sum(1 for v in assignment.values() if v == name) for name in ("train", "val", "test")}
    print(f"{len(documents)} documents -> {documents_path}")
    print(f"splits {counts} -> {splits_path}")
    for name in LABEL_NAMES:
        n_docs = sum(1 for doc in documents if doc.spans.get(name))
        n_spans = sum(len(doc.spans.get(name, ())) for doc in documents)
        print(f"  {name:32s} {n_spans:5d} spans across {n_docs:4d} documents")


if __name__ == "__main__":
    main()
