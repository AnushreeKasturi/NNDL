"""Document-level train/validation/test splits, computed once and persisted.

Splitting by chunk would leak: overlapping windows of one contract would land
on both sides of the split, and near-duplicate text would be scored as held
out. Splitting by document avoids that. Writing the assignment to disk means
every model in the comparison is measured on exactly the same test set.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .cuad import Document

SPLIT_NAMES = ("train", "val", "test")


def assign_splits(
    documents: list[Document],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Map each doc_id to 'train', 'val' or 'test'."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum below 1")

    doc_ids = sorted({doc.doc_id for doc in documents})
    random.Random(seed).shuffle(doc_ids)

    n_test = int(len(doc_ids) * test_ratio)
    n_val = int(len(doc_ids) * val_ratio)

    assignment = {}
    for i, doc_id in enumerate(doc_ids):
        if i < n_test:
            assignment[doc_id] = "test"
        elif i < n_test + n_val:
            assignment[doc_id] = "val"
        else:
            assignment[doc_id] = "train"
    return assignment


def save_splits(assignment: dict[str, str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(assignment, indent=2, sort_keys=True), encoding="utf-8")


def load_splits(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select(documents: list[Document], assignment: dict[str, str], split: str) -> list[Document]:
    """The documents belonging to one split."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    return [doc for doc in documents if assignment.get(doc.doc_id) == split]
