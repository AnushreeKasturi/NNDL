"""Checks for chunk-to-document pooling.

Run with `.venv/bin/python tests/test_pooling.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_risk_classifier.chunking import Chunk  # noqa: E402
from legal_risk_classifier.labels import LABEL_TO_INDEX, NUM_LABELS  # noqa: E402
from legal_risk_classifier.pooling import pool_to_documents  # noqa: E402

INSURANCE = LABEL_TO_INDEX["Insurance"]


def _chunks(doc_id: str, n: int, positive_at: int | None = None) -> list[Chunk]:
    return [
        Chunk(
            doc_id=doc_id,
            index=i,
            char_start=0,
            char_end=1,
            text="x",
            labels=("Insurance",) if i == positive_at else (),
        )
        for i in range(n)
    ]


def test_max_pooling_keeps_a_single_positive_chunk() -> None:
    """The reason max is the default: one clause in sixty chunks still counts."""
    chunks = _chunks("contract-a", 60, positive_at=17)
    probs = np.full((60, NUM_LABELS), 0.01, dtype=np.float32)
    probs[17, INSURANCE] = 0.95

    _, doc_true, doc_prob = pool_to_documents(chunks, probs, method="max")
    assert doc_prob[0, INSURANCE] == 0.95
    assert doc_true[0, INSURANCE] == 1.0


def test_mean_pooling_buries_that_same_positive() -> None:
    """Included as the comparison that shows why mean is wrong for this task."""
    chunks = _chunks("contract-a", 60, positive_at=17)
    probs = np.full((60, NUM_LABELS), 0.01, dtype=np.float32)
    probs[17, INSURANCE] = 0.95

    _, _, doc_prob = pool_to_documents(chunks, probs, method="mean")
    assert doc_prob[0, INSURANCE] < 0.03, "a true positive is averaged away"


def test_top_k_sits_between_max_and_mean() -> None:
    chunks = _chunks("contract-a", 10, positive_at=3)
    probs = np.full((10, NUM_LABELS), 0.1, dtype=np.float32)
    probs[3, INSURANCE] = 0.9

    pooled = {
        method: pool_to_documents(chunks, probs, method=method, top_k=3)[2][0, INSURANCE]
        for method in ("max", "top_k", "mean")
    }
    assert pooled["mean"] < pooled["top_k"] < pooled["max"]


def test_documents_are_grouped_and_row_aligned() -> None:
    chunks = _chunks("b-contract", 3, positive_at=0) + _chunks("a-contract", 2)
    probs = np.zeros((5, NUM_LABELS), dtype=np.float32)
    probs[0, INSURANCE] = 0.8

    doc_ids, doc_true, doc_prob = pool_to_documents(chunks, probs, method="max")
    assert doc_ids == ["a-contract", "b-contract"], "sorted, so rows are deterministic"
    assert doc_true.shape == doc_prob.shape == (2, NUM_LABELS)
    assert doc_prob[doc_ids.index("b-contract"), INSURANCE] == 0.8
    assert doc_true[doc_ids.index("a-contract"), INSURANCE] == 0.0


def test_document_truth_is_the_union_of_its_chunk_labels() -> None:
    chunks = [
        Chunk(doc_id="d", index=0, char_start=0, char_end=1, text="x", labels=("Insurance",)),
        Chunk(doc_id="d", index=1, char_start=0, char_end=1, text="x", labels=("Non-Compete",)),
    ]
    _, doc_true, _ = pool_to_documents(chunks, np.zeros((2, NUM_LABELS), dtype=np.float32))
    assert doc_true[0].sum() == 2, "both labels belong to the contract"


def test_mismatched_lengths_and_unknown_methods_are_rejected() -> None:
    chunks = _chunks("d", 3)
    for call, kwargs in (
        (np.zeros((2, NUM_LABELS), dtype=np.float32), {}),
        (np.zeros((3, NUM_LABELS), dtype=np.float32), {"method": "median"}),
    ):
        try:
            pool_to_documents(chunks, call, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs or 'length mismatch'}")


def test_top_k_handles_documents_shorter_than_k() -> None:
    chunks = _chunks("d", 2)
    probs = np.array([[0.2] * NUM_LABELS, [0.6] * NUM_LABELS], dtype=np.float32)
    _, _, doc_prob = pool_to_documents(chunks, probs, method="top_k", top_k=5)
    assert abs(doc_prob[0, INSURANCE] - 0.4) < 1e-6, "k falls back to the chunk count"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
