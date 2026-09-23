"""Load a trained run and score raw contract text.

Shared by the demo and the dashboard exporter so both agree on how a document
is chunked, scored and pooled. A second implementation of this would drift
from the evaluation path, and the demo would quietly stop matching the
reported numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .chunking import Chunk, chunk_document
from .cuad import Document
from .labels import LABEL_NAMES, LABEL_TO_INDEX
from .textcnn import TextCNN
from .vocab import Vocabulary


@dataclass
class ClauseHit:
    label: str
    score: float
    char_start: int
    char_end: int
    text: str


@dataclass
class DocumentPrediction:
    doc_id: str
    chunks: list[Chunk]
    chunk_probs: np.ndarray
    document_scores: dict[str, float]
    predicted: list[str]

    def hits(self, label: str, threshold: float | None = None) -> list[ClauseHit]:
        """Chunks that fired for one label, strongest first."""
        index = LABEL_TO_INDEX[label]
        cut = threshold if threshold is not None else 0.5
        found = [
            ClauseHit(
                label=label,
                score=float(self.chunk_probs[row, index]),
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                text=chunk.text,
            )
            for row, chunk in enumerate(self.chunks)
            if self.chunk_probs[row, index] >= cut
        ]
        return sorted(found, key=lambda hit: hit.score, reverse=True)


class CNNRuntime:
    """A trained TextCNN run, ready to score text."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        summary = json.loads((self.run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.chunking = summary["chunking"]
        self.metrics = summary

        training_path = self.run_dir / "training.json"
        self.thresholds = (
            json.loads(training_path.read_text(encoding="utf-8"))["thresholds"]
            if training_path.exists()
            else [0.5] * len(LABEL_NAMES)
        )

        self.vocab = Vocabulary.load(self.run_dir / "vocab.json")
        state = torch.load(self.run_dir / "model.pt", map_location="cpu")
        conv_weights = sorted(
            key for key in state if key.startswith("convolutions.") and key.endswith(".weight")
        )
        self.model = TextCNN(
            vocab_size=len(self.vocab),
            num_labels=len(LABEL_NAMES),
            embedding_dim=state["embedding.weight"].shape[1],
            num_filters=state[conv_weights[0]].shape[0],
            kernel_sizes=tuple(state[key].shape[2] for key in conv_weights),
        )
        self.model.load_state_dict(state)
        self.model.eval()

    def threshold_for(self, label: str) -> float:
        return float(self.thresholds[LABEL_TO_INDEX[label]])

    @torch.no_grad()
    def predict_document(self, text: str, doc_id: str = "input") -> DocumentPrediction:
        """Chunk, score, and pool exactly as the evaluation path does."""
        document = Document(doc_id=doc_id, text=text, spans={})
        chunks = chunk_document(
            document,
            window=self.chunking["window"],
            overlap=self.chunking["overlap"],
        )
        if not chunks:
            empty = np.zeros((0, len(LABEL_NAMES)), dtype=np.float32)
            return DocumentPrediction(doc_id, [], empty, dict.fromkeys(LABEL_NAMES, 0.0), [])

        batch = torch.tensor(
            [self.vocab.encode(chunk.text, self.chunking["max_tokens"]) for chunk in chunks]
        )
        probs = torch.sigmoid(self.model(input_ids=batch)).numpy()

        # Max-pooling, matching pooling.pool_to_documents.
        pooled = probs.max(axis=0)
        scores = {name: float(pooled[i]) for i, name in enumerate(LABEL_NAMES)}
        predicted = [name for name in LABEL_NAMES if scores[name] >= self.threshold_for(name)]
        return DocumentPrediction(doc_id, chunks, probs, scores, predicted)
