"""Read CUAD v1 into documents carrying per-label character spans.

CUAD stores one SQuAD-style "paragraph" per contract, whose context is the
entire contract text. Labels are not document attributes: they are answer
spans, each with a character offset into that context. This module keeps the
spans, because throwing them away is what makes chunk-level supervision
impossible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .labels import LABEL_NAMES, label_for_cuad_category

Span = tuple[int, int]


@dataclass(frozen=True)
class Document:
    """One contract: its full text, plus where each clause type appears in it."""

    doc_id: str
    text: str
    spans: dict[str, tuple[Span, ...]]

    def __post_init__(self) -> None:
        """Normalise spans so every label is present, ordered and merged.

        Callers may pass only the labels they care about; downstream code then
        gets to index `spans[name]` for any label without guarding.
        """
        normalised = {name: merge_spans(list(self.spans.get(name, ()))) for name in LABEL_NAMES}
        object.__setattr__(self, "spans", normalised)

    def labels(self) -> tuple[str, ...]:
        """Labels present anywhere in the document."""
        return tuple(name for name in LABEL_NAMES if self.spans[name])


def merge_spans(spans: list[Span]) -> tuple[Span, ...]:
    """Sort and merge overlapping or touching spans.

    CUAD annotators sometimes record the same clause twice, or record adjacent
    sentences separately. Merging keeps overlap arithmetic downstream honest.
    """
    if not spans:
        return ()
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _category_of(qa: dict) -> str:
    """CUAD encodes the category as the id suffix: '<title>__<Category>'.

    The question text is a full natural-language prompt, so parsing the id is
    both simpler and more reliable.
    """
    return str(qa.get("id", "")).rsplit("__", 1)[-1]


def parse_cuad(path: str | Path) -> list[Document]:
    """Load CUAD v1 JSON into Documents, keeping only our six labels."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    documents: list[Document] = []

    for entry in raw.get("data", []):
        title = str(entry.get("title", "")).strip()
        for paragraph in entry.get("paragraphs", []):
            text = str(paragraph.get("context", ""))
            if not text.strip():
                continue

            collected: dict[str, list[Span]] = {name: [] for name in LABEL_NAMES}
            for qa in paragraph.get("qas", []):
                label = label_for_cuad_category(_category_of(qa))
                if label is None:
                    continue
                for answer in qa.get("answers", []):
                    answer_text = str(answer.get("text", ""))
                    start = answer.get("answer_start")
                    if not answer_text or not isinstance(start, int) or start < 0:
                        continue
                    end = min(start + len(answer_text), len(text))
                    if end > start:
                        collected[label].append((start, end))

            documents.append(
                Document(
                    doc_id=title or f"doc-{len(documents)}",
                    text=text,
                    spans=collected,
                )
            )
    return documents


def save_documents(documents: list[Document], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "text": doc.text,
                        "spans": {k: [list(s) for s in v] for k, v in doc.spans.items()},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_documents(path: str | Path) -> list[Document]:
    documents: list[Document] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            documents.append(
                Document(
                    doc_id=obj["doc_id"],
                    text=obj["text"],
                    spans={
                        name: tuple((int(a), int(b)) for a, b in obj["spans"].get(name, []))
                        for name in LABEL_NAMES
                    },
                )
            )
    return documents
