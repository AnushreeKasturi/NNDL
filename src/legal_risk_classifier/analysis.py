"""Measure the properties of the corpus that constrain the modelling choices.

Every statistic here exists to answer a design question: how long are the
documents (does 512 tokens suffice), how rare is each label (how badly is the
loss imbalanced), how long are the clauses (is the chunk window sensible), and
do labels co-occur (is multi-label the right framing).
"""

from __future__ import annotations

import statistics
from collections import Counter

from .chunking import chunk_documents, word_offsets
from .cuad import Document
from .labels import LABEL_NAMES

# BERT-family context limit, minus [CLS] and [SEP].
BERT_TOKEN_LIMIT = 512
LONGFORMER_TOKEN_LIMIT = 4096

# Legal English runs longer than the tokenizer's vocabulary was built for:
# subword splitting inflates word counts by roughly this factor. Used only for
# the headline "does it fit" figures; exact counts come from a real tokenizer.
TOKENS_PER_WORD = 1.45


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def at(fraction: float) -> float:
        return float(ordered[min(int(fraction * len(ordered)), len(ordered) - 1)])

    return {
        "min": float(ordered[0]),
        "p25": at(0.25),
        "median": at(0.50),
        "mean": float(statistics.fmean(ordered)),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": float(ordered[-1]),
    }


def document_length_stats(documents: list[Document]) -> dict:
    """Document lengths, and how many exceed each model's context window."""
    word_counts = [len(word_offsets(doc.text)) for doc in documents]
    estimated_tokens = [n * TOKENS_PER_WORD for n in word_counts]
    return {
        "documents": len(documents),
        # Kept for the histogram; the report writer drops it before serialising.
        "_raw_words": word_counts,
        "words": _quantiles([float(n) for n in word_counts]),
        "estimated_tokens": _quantiles(estimated_tokens),
        "exceeding_bert_512": sum(t > BERT_TOKEN_LIMIT for t in estimated_tokens),
        "exceeding_longformer_4096": sum(t > LONGFORMER_TOKEN_LIMIT for t in estimated_tokens),
    }


def span_length_stats(documents: list[Document]) -> dict[str, dict]:
    """How long an annotated clause is, per label, in words.

    This is what decides whether a chunk window can hold a whole clause.
    """
    out: dict[str, dict] = {}
    for name in LABEL_NAMES:
        lengths = [
            float(len(doc.text[start:end].split()))
            for doc in documents
            for start, end in doc.spans[name]
        ]
        out[name] = {"spans": len(lengths), **_quantiles(lengths)}
    return out


def document_label_stats(documents: list[Document]) -> dict[str, dict]:
    """Per label: documents containing it, and total annotated spans."""
    total = len(documents) or 1
    return {
        name: {
            "documents": sum(1 for doc in documents if doc.spans[name]),
            "document_rate": sum(1 for doc in documents if doc.spans[name]) / total,
            "spans": sum(len(doc.spans[name]) for doc in documents),
        }
        for name in LABEL_NAMES
    }


def chunk_label_stats(documents: list[Document], window: int = 300, overlap: int = 100) -> dict:
    """Per label: how many chunks are positive, which is the imbalance the loss sees."""
    chunks = chunk_documents(documents, window=window, overlap=overlap)
    total = len(chunks) or 1
    per_label = {}
    for name in LABEL_NAMES:
        positives = sum(1 for chunk in chunks if name in chunk.labels)
        per_label[name] = {
            "positive_chunks": positives,
            "positive_rate": positives / total,
            # The pos_weight a balanced BCE loss would use for this label.
            "pos_weight": (total - positives) / max(positives, 1),
        }
    return {
        "window": window,
        "overlap": overlap,
        "chunks": len(chunks),
        "unlabelled_chunks": sum(1 for chunk in chunks if not chunk.labels),
        "unlabelled_rate": sum(1 for chunk in chunks if not chunk.labels) / total,
        "per_label": per_label,
    }


def label_cooccurrence(documents: list[Document]) -> dict[str, dict[str, int]]:
    """How often two labels appear in the same document.

    Non-zero off-diagonal entries are what makes this multi-label rather than
    multi-class: a contract is not required to pick one clause type.
    """
    return {
        a: {b: sum(1 for doc in documents if doc.spans[a] and doc.spans[b]) for b in LABEL_NAMES}
        for a in LABEL_NAMES
    }


def labels_per_document(documents: list[Document]) -> dict[int, int]:
    """Distribution of how many distinct labels a document carries."""
    counts = Counter(len(doc.labels()) for doc in documents)
    return dict(sorted(counts.items()))


def split_balance(documents: list[Document], assignment: dict[str, str]) -> dict[str, dict]:
    """Per-label document rate within each split, to confirm no split is skewed."""
    out = {}
    for split in ("train", "val", "test"):
        subset = [doc for doc in documents if assignment.get(doc.doc_id) == split]
        total = len(subset) or 1
        out[split] = {
            "documents": len(subset),
            "label_rates": {
                name: sum(1 for doc in subset if doc.spans[name]) / total for name in LABEL_NAMES
            },
        }
    return out


def build_report(documents: list[Document], assignment: dict[str, str] | None = None, **chunk_kwargs) -> dict:
    """Every statistic, as one JSON-serialisable dictionary."""
    report = {
        "document_lengths": document_length_stats(documents),
        "document_labels": document_label_stats(documents),
        "span_lengths": span_length_stats(documents),
        "chunk_labels": chunk_label_stats(documents, **chunk_kwargs),
        "label_cooccurrence": label_cooccurrence(documents),
        "labels_per_document": labels_per_document(documents),
    }
    if assignment:
        report["split_balance"] = split_balance(documents, assignment)
    return report
