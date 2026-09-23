"""Roll chunk-level predictions up to document level.

Classification happens per chunk, but the question a user asks is about a
contract: does this agreement cap liability? The pooling rule is how those
two levels connect, and it is a modelling decision rather than a formality.

Max-pooling is the default because the task is existential. A clause appearing
in one chunk of a sixty-chunk contract means the contract contains it, and
averaging would bury that single positive under fifty-nine negatives.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .chunking import Chunk
from .labels import LABEL_TO_INDEX, NUM_LABELS


def pool_to_documents(
    chunks: list[Chunk],
    chunk_probs: np.ndarray,
    method: str = "max",
    top_k: int = 3,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (doc_ids, doc_true, doc_prob), all aligned by row.

    `method` is one of:
      max   - the strongest chunk decides. Correct for an existential question.
      mean  - the document average. Included as the comparison that shows why
              max is right, not because it is a good idea here.
      top_k - the mean of the k strongest chunks, a middle ground that resists
              a single over-confident chunk without drowning in negatives.
    """
    if len(chunks) != len(chunk_probs):
        raise ValueError(f"{len(chunks)} chunks but {len(chunk_probs)} predictions")
    if method not in {"max", "mean", "top_k"}:
        raise ValueError(f"unknown pooling method: {method!r}")

    grouped: dict[str, list[int]] = defaultdict(list)
    for row, chunk in enumerate(chunks):
        grouped[chunk.doc_id].append(row)

    doc_ids = sorted(grouped)
    doc_prob = np.zeros((len(doc_ids), NUM_LABELS), dtype=np.float32)
    doc_true = np.zeros((len(doc_ids), NUM_LABELS), dtype=np.float32)

    for i, doc_id in enumerate(doc_ids):
        rows = grouped[doc_id]
        probs = chunk_probs[rows]
        if method == "max":
            doc_prob[i] = probs.max(axis=0)
        elif method == "mean":
            doc_prob[i] = probs.mean(axis=0)
        else:
            k = min(top_k, len(rows))
            doc_prob[i] = np.sort(probs, axis=0)[-k:].mean(axis=0)

        for row in rows:
            for label in chunks[row].labels:
                doc_true[i, LABEL_TO_INDEX[label]] = 1.0

    return doc_ids, doc_true, doc_prob
