"""CLI: export everything the results dashboard renders, as one JSON file.

The dashboard is a static page. Keeping its data in a generated file means the
page never contains a number that was typed by hand, and regenerating after a
new training run updates the page without editing it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import build_report
from .chunking import chunk_document
from .cuad import Document, load_documents
from .labels import LABEL_NAMES, LABEL_TO_INDEX
from .runtime import CNNRuntime
from .splits import load_splits, select

# Enough of a contract to read the clause in context without shipping 8,000
# words per document to the browser.
EXCERPT_CHARS = 1400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", type=Path, default=Path("outputs/cnn"))
    parser.add_argument("--documents", type=Path, default=Path("data/processed/documents.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--out", type=Path, default=Path("dashboard/data.json"))
    parser.add_argument("--template", type=Path, default=Path("dashboard/template.html"))
    parser.add_argument("--html", type=Path, default=Path("dashboard/index.html"))
    parser.add_argument("--examples", type=int, default=10, help="contracts to embed")
    return parser.parse_args()


def _excerpt_around(doc: Document, start: int, end: int) -> dict:
    """A readable window around a span, with the span's offsets inside it."""
    margin = max((EXCERPT_CHARS - (end - start)) // 2, 120)
    left = max(start - margin, 0)
    right = min(end + margin, len(doc.text))
    return {
        "text": doc.text[left:right],
        "highlight_start": start - left,
        "highlight_end": min(end, right) - left,
        "doc_offset": left,
    }


def _prediction_summary(runtime: CNNRuntime, doc: Document) -> dict:
    prediction = runtime.predict_document(doc.text, doc.doc_id)
    truth = set(doc.labels())
    predicted = set(prediction.predicted)
    return {
        "scores": prediction.document_scores,
        "predicted": sorted(predicted),
        "truth": sorted(truth),
        "false_positives": sorted(predicted - truth),
        "false_negatives": sorted(truth - predicted),
        "chunk_count": len(prediction.chunks),
        "top_windows": {
            label: [
                {
                    "score": hit.score,
                    "char_start": hit.char_start,
                    "text": hit.text[:900],
                }
                for hit in prediction.hits(label, threshold=runtime.threshold_for(label))[:2]
            ]
            for label in LABEL_NAMES
        },
    }


def choose_examples(
    documents: list[Document], runtime: CNNRuntime, limit: int
) -> list[Document]:
    """Cover every label, and prefer contracts the model gets wrong.

    A page showing only successes is a page nobody believes, and the errors are
    the ones worth discussing.
    """
    scored = []
    for doc in documents:
        summary = _prediction_summary(runtime, doc)
        errors = len(summary["false_positives"]) + len(summary["false_negatives"])
        scored.append((doc, summary, errors))

    chosen: list[tuple[Document, dict]] = []
    covered: set[str] = set()

    # First pass: one contract per label, preferring an interesting one.
    for label in LABEL_NAMES:
        candidates = [
            (doc, summary, errors)
            for doc, summary, errors in scored
            if label in summary["truth"] and doc.doc_id not in {d.doc_id for d, _ in chosen}
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (-row[2], -len(row[1]["truth"])))
        doc, summary, _ = candidates[0]
        chosen.append((doc, summary))
        covered.update(summary["truth"])

    # Second pass: fill the remainder with the most error-prone contracts.
    remaining = sorted(
        (row for row in scored if row[0].doc_id not in {d.doc_id for d, _ in chosen}),
        key=lambda row: -row[2],
    )
    for doc, summary, _ in remaining[: max(limit - len(chosen), 0)]:
        chosen.append((doc, summary))

    return chosen[:limit]


def build_payload(
    documents: list[Document],
    assignment: dict[str, str],
    runtime: CNNRuntime,
    example_count: int,
) -> dict:
    test_documents = select(documents, assignment, "test")
    examples = choose_examples(test_documents, runtime, example_count)

    evaluation_path = runtime.run_dir / "evaluation_test.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.exists()
        else None
    )
    training_path = runtime.run_dir / "training.json"
    training = (
        json.loads(training_path.read_text(encoding="utf-8"))
        if training_path.exists()
        else None
    )

    return {
        "labels": list(LABEL_NAMES),
        "dataset": build_report(documents, assignment),
        "model": {
            "name": "TextCNN",
            "metrics": runtime.metrics,
            "thresholds": {
                name: runtime.threshold_for(name) for name in LABEL_NAMES
            },
            "history": (training or {}).get("history", []),
            "evaluation": evaluation,
        },
        "examples": [
            {
                "doc_id": doc.doc_id,
                "words": len(doc.text.split()),
                "prediction": summary,
                "spans": {
                    label: [
                        _excerpt_around(doc, start, end) for start, end in doc.spans[label][:3]
                    ]
                    for label in LABEL_NAMES
                    if doc.spans[label]
                },
            }
            for doc, summary in examples
        ],
    }


def main() -> None:
    args = parse_args()
    documents = load_documents(args.documents)
    assignment = load_splits(args.splits)
    runtime = CNNRuntime(args.run_dir)

    payload = build_payload(documents, assignment, runtime, args.examples)
    # 510 integers, kept for the length histogram; the verbose key is dropped.
    lengths = payload["dataset"]["document_lengths"]
    lengths["word_counts"] = lengths.pop("_raw_words", [])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(payload, separators=(",", ":"))
    args.out.write_text(serialised, encoding="utf-8")
    print(f"{len(payload['examples'])} examples, {args.out.stat().st_size / 1024:,.0f} KB -> {args.out}")

    # The page ships with its data inlined: one self-contained file, and no
    # fetch that could fail under a strict content policy.
    if args.template.exists():
        page = args.template.read_text(encoding="utf-8").replace("/*DATA*/", serialised)
        args.html.write_text(page, encoding="utf-8")
        print(f"page {args.html.stat().st_size / 1024:,.0f} KB -> {args.html}")
    for example in payload["examples"]:
        prediction = example["prediction"]
        print(
            f"  {example['doc_id'][:44]:44s} "
            f"FP {len(prediction['false_positives'])} FN {len(prediction['false_negatives'])}"
        )


if __name__ == "__main__":
    main()
