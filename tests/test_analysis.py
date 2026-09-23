"""Checks for the corpus statistics.

Run with `python tests/test_analysis.py`. Figures are not exercised here;
matplotlib output is checked by looking at it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_risk_classifier.analysis import (  # noqa: E402
    BERT_TOKEN_LIMIT,
    TOKENS_PER_WORD,
    _quantiles,
    chunk_label_stats,
    document_label_stats,
    document_length_stats,
    label_cooccurrence,
    labels_per_document,
    span_length_stats,
    split_balance,
)
from legal_risk_classifier.cuad import Document  # noqa: E402


def _doc(doc_id: str, words: int, spans: dict[str, list[tuple[int, int]]] | None = None) -> Document:
    return Document(
        doc_id=doc_id,
        text="word " * words,
        spans={k: tuple(v) for k, v in (spans or {}).items()},
    )


def test_quantiles_are_ordered_and_bounded() -> None:
    q = _quantiles([float(i) for i in range(1, 101)])
    assert q["min"] == 1.0 and q["max"] == 100.0
    assert q["min"] <= q["p25"] <= q["median"] <= q["p75"] <= q["p95"] <= q["max"]
    assert abs(q["mean"] - 50.5) < 1e-9
    assert _quantiles([]) == {}


def test_length_stats_count_documents_over_each_window() -> None:
    short = int(BERT_TOKEN_LIMIT / TOKENS_PER_WORD) - 50
    stats = document_length_stats([_doc("a", short), _doc("b", 10_000), _doc("c", 20_000)])
    assert stats["documents"] == 3
    assert stats["exceeding_bert_512"] == 2, "only the two long documents overflow 512"
    assert stats["exceeding_longformer_4096"] == 2
    assert stats["words"]["max"] == 20_000
    assert len(stats["_raw_words"]) == 3


def test_document_label_stats_count_documents_and_spans() -> None:
    docs = [
        _doc("a", 100, {"Insurance": [(0, 10), (50, 60)]}),
        _doc("b", 100, {"Insurance": [(0, 10)]}),
        _doc("c", 100, {}),
    ]
    stats = document_label_stats(docs)
    assert stats["Insurance"]["documents"] == 2
    assert stats["Insurance"]["spans"] == 3
    assert abs(stats["Insurance"]["document_rate"] - 2 / 3) < 1e-9
    assert stats["Non-Compete"]["documents"] == 0


def test_pos_weight_is_the_negative_to_positive_ratio() -> None:
    """The weight a balanced BCE loss needs, which the report publishes per label."""
    docs = [_doc("a", 600, {"Audit Rights": [(0, 200)]})]
    stats = chunk_label_stats(docs, window=100, overlap=0)
    per = stats["per_label"]["Audit Rights"]
    expected = (stats["chunks"] - per["positive_chunks"]) / per["positive_chunks"]
    assert abs(per["pos_weight"] - expected) < 1e-9
    assert stats["per_label"]["Insurance"]["pos_weight"] == stats["chunks"], "no positives"


def test_chunk_stats_report_the_unlabelled_share() -> None:
    docs = [_doc("a", 500, {}), _doc("b", 500, {})]
    stats = chunk_label_stats(docs, window=100, overlap=0)
    assert stats["unlabelled_rate"] == 1.0, "nothing is annotated"
    assert stats["chunks"] == stats["unlabelled_chunks"]


def test_span_length_stats_measure_clauses_in_words() -> None:
    text = "alpha beta gamma delta epsilon"
    doc = Document(doc_id="a", text=text, spans={"Insurance": ((0, len("alpha beta gamma")),)})
    stats = span_length_stats([doc])
    assert stats["Insurance"]["spans"] == 1
    assert stats["Insurance"]["median"] == 3.0
    assert stats["Non-Compete"]["spans"] == 0


def test_cooccurrence_diagonal_equals_the_document_count() -> None:
    docs = [
        _doc("a", 100, {"Insurance": [(0, 10)], "Audit Rights": [(20, 30)]}),
        _doc("b", 100, {"Insurance": [(0, 10)]}),
    ]
    matrix = label_cooccurrence(docs)
    assert matrix["Insurance"]["Insurance"] == 2
    assert matrix["Insurance"]["Audit Rights"] == 1
    assert matrix["Audit Rights"]["Insurance"] == 1, "the matrix is symmetric"
    assert matrix["Non-Compete"]["Insurance"] == 0


def test_labels_per_document_distribution_sums_to_the_corpus() -> None:
    docs = [
        _doc("a", 100, {"Insurance": [(0, 10)], "Audit Rights": [(20, 30)]}),
        _doc("b", 100, {"Insurance": [(0, 10)]}),
        _doc("c", 100, {}),
    ]
    distribution = labels_per_document(docs)
    assert distribution == {0: 1, 1: 1, 2: 1}
    assert sum(distribution.values()) == len(docs)


def test_split_balance_reports_per_split_label_rates() -> None:
    docs = [_doc(f"d{i}", 100, {"Insurance": [(0, 10)]} if i < 2 else {}) for i in range(4)]
    assignment = {"d0": "train", "d1": "train", "d2": "test", "d3": "test"}
    balance = split_balance(docs, assignment)
    assert balance["train"]["documents"] == 2
    assert balance["train"]["label_rates"]["Insurance"] == 1.0
    assert balance["test"]["label_rates"]["Insurance"] == 0.0
    assert balance["val"]["documents"] == 0, "an empty split must not divide by zero"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
