"""Checks for span extraction, chunking and splitting.

Run with `python tests/test_data_pipeline.py` (no pytest required).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_risk_classifier.chunking import (  # noqa: E402
    chunk_document,
    labels_for_window,
    overlap_ratio,
    word_offsets,
)
from legal_risk_classifier.cuad import (  # noqa: E402
    Document,
    load_documents,
    merge_spans,
    parse_cuad,
    save_documents,
)
from legal_risk_classifier.labels import label_for_cuad_category  # noqa: E402
from legal_risk_classifier.splits import assign_splits, select  # noqa: E402


def _fake_cuad(text: str, answers: dict[str, list[str]]) -> dict:
    """A CUAD-shaped payload whose spans are located by searching `text`."""
    qas = []
    for category, snippets in answers.items():
        qas.append(
            {
                "id": f"FAKE_CONTRACT__{category}",
                "question": f'Highlight the parts (if any) of this contract related to "{category}".',
                "answers": [{"text": s, "answer_start": text.index(s)} for s in snippets],
                "is_impossible": False,
            }
        )
    return {"version": "1", "data": [{"title": "FAKE_CONTRACT", "paragraphs": [{"context": text, "qas": qas}]}]}


def test_category_parsed_from_id_not_question() -> None:
    """The original defect: CUAD questions never match a label name directly."""
    question = 'Highlight the parts (if any) of this contract related to "Cap On Liability" that should be reviewed by a lawyer.'
    assert label_for_cuad_category(question) is None, "a whole question is not a category"
    assert label_for_cuad_category("Cap On Liability") == "Cap on Liability"
    assert label_for_cuad_category("Parties") is None, "non-target categories are dropped"


def test_spans_extracted_at_correct_offsets() -> None:
    text = "alpha " * 50 + "LIABILITY IS CAPPED AT FEES PAID. " + "beta " * 50 + "INSURANCE SHALL BE MAINTAINED."
    payload = _fake_cuad(
        text,
        {
            "Cap On Liability": ["LIABILITY IS CAPPED AT FEES PAID."],
            "Insurance": ["INSURANCE SHALL BE MAINTAINED."],
            "Parties": ["alpha"],  # a non-target category, must be ignored
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cuad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        docs = parse_cuad(path)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.labels() == ("Cap on Liability", "Insurance")
    start, end = doc.spans["Cap on Liability"][0]
    assert doc.text[start:end] == "LIABILITY IS CAPPED AT FEES PAID."
    assert doc.spans["Non-Compete"] == (), "absent labels stay empty"


def test_merge_spans_joins_overlapping_annotations() -> None:
    assert merge_spans([(0, 10), (5, 20), (30, 40)]) == ((0, 20), (30, 40))
    assert merge_spans([(10, 20), (0, 10)]) == ((0, 20),), "touching spans merge"
    assert merge_spans([]) == ()


def test_overlap_ratio_scores_against_the_shorter_range() -> None:
    assert overlap_ratio((0, 100), (200, 300)) == 0.0, "disjoint"
    assert overlap_ratio((0, 1000), (400, 500)) == 1.0, "short span inside a long chunk"
    assert overlap_ratio((400, 500), (0, 1000)) == 1.0, "short chunk inside a long span"
    assert overlap_ratio((0, 100), (95, 500)) == 0.05, "a sliver at the boundary"


def test_word_offsets_index_back_into_the_text() -> None:
    text = "  one   two\nthree "
    assert [text[a:b] for a, b in word_offsets(text)] == ["one", "two", "three"]


def test_only_chunks_containing_the_clause_are_positive() -> None:
    """The second defect: labels must not be stamped onto every chunk."""
    clause = "THE LICENSOR GRANTS A NON EXCLUSIVE LICENSE TO THE LICENSEE. "
    text = "filler " * 400 + clause + "filler " * 400
    doc = Document(
        doc_id="d",
        text=text,
        spans={"License Grant": ((text.index(clause), text.index(clause) + len(clause)),)},
    )
    chunks = chunk_document(doc, window=100, overlap=20)

    positive = [c for c in chunks if "License Grant" in c.labels]
    assert positive, "the clause must land in at least one chunk"
    assert len(positive) < len(chunks), "the label must not cover every chunk"
    for chunk in positive:
        assert "LICENSE" in chunk.text, "a positive chunk actually contains the clause"
    for chunk in chunks:
        if chunk not in positive:
            assert "LICENSE" not in chunk.text or overlap_ratio(
                (chunk.char_start, chunk.char_end),
                doc.spans["License Grant"][0],
            ) < 0.5


def test_overlap_recovers_a_clause_split_across_a_boundary() -> None:
    """The reason windows overlap: a boundary must not destroy a clause."""
    clause = "AUDIT RIGHTS " * 10
    text = "word " * 100 + clause + "word " * 100
    doc = Document(
        doc_id="d",
        text=text,
        spans={"Audit Rights": ((text.index(clause), text.index(clause) + len(clause)),)},
    )
    # window=100 words puts the clause astride the boundary of the first window.
    without = chunk_document(doc, window=100, overlap=0)
    with_overlap = chunk_document(doc, window=100, overlap=50)
    assert sum("Audit Rights" in c.labels for c in with_overlap) >= sum(
        "Audit Rights" in c.labels for c in without
    ), "overlapping windows recover at least as many positives"


def test_chunks_tile_the_document_without_gaps() -> None:
    doc = Document(doc_id="d", text="word " * 1000, spans={})
    chunks = chunk_document(doc, window=300, overlap=100)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.text.rstrip())
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.char_start < earlier.char_end, "consecutive windows overlap"
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_labels_for_window_needs_meaningful_coverage() -> None:
    spans = {"Insurance": ((1000, 1100),)}
    assert labels_for_window((900, 1200), spans) == ("Insurance",)
    assert labels_for_window((1095, 2000), spans) == (), "5 of 100 characters is not the clause"


def test_chunk_window_arguments_are_validated() -> None:
    doc = Document(doc_id="d", text="word " * 10, spans={})
    for bad in ({"window": 0}, {"window": 10, "overlap": 10}, {"window": 10, "overlap": -1}):
        try:
            chunk_document(doc, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_splits_are_by_document_and_reproducible() -> None:
    docs = [Document(doc_id=f"d{i}", text="word " * 10, spans={}) for i in range(100)]
    first = assign_splits(docs, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert first == assign_splits(docs, val_ratio=0.15, test_ratio=0.15, seed=42), "seeded"
    assert sorted(first) == sorted(d.doc_id for d in docs), "every document is assigned"
    assert len(select(docs, first, "test")) == 15
    assert len(select(docs, first, "val")) == 15
    assert len(select(docs, first, "train")) == 70

    groups = {name: {d.doc_id for d in select(docs, first, name)} for name in ("train", "val", "test")}
    assert not groups["train"] & groups["test"], "no document appears in two splits"
    assert not groups["train"] & groups["val"]


def test_documents_round_trip_through_jsonl() -> None:
    original = [Document(doc_id="d", text="a b c", spans={"Insurance": ((0, 1),)})]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "docs.jsonl"
        save_documents(original, path)
        assert load_documents(path) == original


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
