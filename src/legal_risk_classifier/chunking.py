"""Slice documents into overlapping windows and assign labels by span overlap.

Windows are measured in whitespace words rather than model tokens, on purpose.
A word window is tokenizer-agnostic, so the CNN, BERT, Legal-BERT and Longformer
all train and evaluate on identical chunk boundaries, and a difference in their
scores is a difference between the models rather than between their tokenizers.
Each model still applies its own tokenizer to the chunk text it receives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .cuad import Document, Span
from .labels import LABEL_NAMES

_WORD = re.compile(r"\S+")

# A chunk counts as positive for a label when it covers at least this fraction
# of the overlapping annotated span, or of itself, whichever is the smaller
# target. Ratios guard both directions: a short clause sitting inside a long
# chunk scores 1.0, a chunk fully inside a very long clause also scores 1.0,
# and a few characters clipped at a boundary scores near 0.
DEFAULT_MIN_OVERLAP_RATIO = 0.5


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    char_start: int
    char_end: int
    text: str
    labels: tuple[str, ...]

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}::chunk{self.index}"


def word_offsets(text: str) -> list[Span]:
    """Character (start, end) for every whitespace-delimited word."""
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def overlap_ratio(chunk: Span, span: Span) -> float:
    """Overlap between two char ranges, over the length of the shorter one."""
    covered = min(chunk[1], span[1]) - max(chunk[0], span[0])
    if covered <= 0:
        return 0.0
    shorter = min(chunk[1] - chunk[0], span[1] - span[0])
    return covered / shorter if shorter > 0 else 0.0


def labels_for_window(
    window: Span,
    spans: dict[str, tuple[Span, ...]],
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> tuple[str, ...]:
    """Which labels a character window carries, by span overlap."""
    return tuple(
        name
        for name in LABEL_NAMES
        if any(overlap_ratio(window, span) >= min_overlap_ratio for span in spans.get(name, ()))
    )


def chunk_document(
    doc: Document,
    window: int = 300,
    overlap: int = 100,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> list[Chunk]:
    """Split one document into overlapping word windows carrying chunk labels.

    `window` and `overlap` are counted in words. The overlap exists so a clause
    landing on a boundary still appears whole in a neighbouring window.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be non-negative and smaller than window")

    offsets = word_offsets(doc.text)
    if not offsets:
        return []

    step = window - overlap
    chunks: list[Chunk] = []
    for start_word in range(0, len(offsets), step):
        slice_ = offsets[start_word : start_word + window]
        if not slice_:
            break
        bounds = (slice_[0][0], slice_[-1][1])
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                index=len(chunks),
                char_start=bounds[0],
                char_end=bounds[1],
                text=doc.text[bounds[0] : bounds[1]],
                labels=labels_for_window(bounds, doc.spans, min_overlap_ratio),
            )
        )
        if start_word + window >= len(offsets):
            break
    return chunks


def chunk_documents(documents: list[Document], **kwargs) -> list[Chunk]:
    return [chunk for doc in documents for chunk in chunk_document(doc, **kwargs)]
