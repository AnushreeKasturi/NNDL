"""Explain a TextCNN prediction by naming the phrases that caused it.

A convolutional filter over word embeddings is a literal n-gram detector, and
max-over-time pooling records which position fired it hardest. That position
maps back to a specific span of words, so an explanation here is an actual
phrase from the contract — "in no event shall", "shall maintain insurance" —
rather than a diffuse weight over tokens.

This is why the CNN is the better interpretability artifact of the two
architectures. Transformer attention is contested as an explanation; a
convolution's receptive field is not a matter of interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .labels import LABEL_NAMES, LABEL_TO_INDEX
from .textcnn import TextCNN
from .vocab import Vocabulary, tokenize


@dataclass(frozen=True)
class Phrase:
    """An n-gram that contributed to one label's score."""

    text: str
    start_word: int
    end_word: int
    contribution: float


def _filter_positions(model: TextCNN, input_ids: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Per convolution: (max activation, argmax position, kernel width)."""
    embedded = model.embedding(input_ids).transpose(1, 2)
    out = []
    for convolution in model.convolutions:
        activations = torch.relu(convolution(embedded))
        peak = activations.max(dim=2)
        out.append((peak.values.squeeze(0), peak.indices.squeeze(0), convolution.kernel_size[0]))
    return out


@torch.no_grad()
def explain(
    model: TextCNN,
    vocab: Vocabulary,
    text: str,
    label: str,
    max_tokens: int = 400,
    top_n: int = 5,
) -> list[Phrase]:
    """The n-grams that pushed `label` up the most, strongest first.

    Contribution is a filter's pooled activation times the classifier weight
    connecting it to this label, which is exactly that filter's additive term
    in the logit.
    """
    if label not in LABEL_TO_INDEX:
        raise ValueError(f"unknown label: {label!r}")

    words = tokenize(text)[:max_tokens]
    if not words:
        return []

    model.eval()
    input_ids = torch.tensor([vocab.encode(text, max_tokens)])
    per_convolution = _filter_positions(model, input_ids)

    weights = model.classifier.weight[LABEL_TO_INDEX[label]]
    phrases: list[Phrase] = []
    offset = 0
    for values, positions, width in per_convolution:
        contributions = values * weights[offset : offset + values.numel()]
        offset += values.numel()
        for filter_index in torch.argsort(contributions, descending=True)[:top_n].tolist():
            contribution = float(contributions[filter_index])
            if contribution <= 0:
                break
            start = int(positions[filter_index])
            end = min(start + width, len(words))
            if start >= len(words):
                continue  # the filter peaked inside padding
            phrases.append(
                Phrase(
                    text=" ".join(words[start:end]),
                    start_word=start,
                    end_word=end,
                    contribution=contribution,
                )
            )

    phrases.sort(key=lambda p: p.contribution, reverse=True)
    return _deduplicate(phrases)[:top_n]


def _deduplicate(phrases: list[Phrase]) -> list[Phrase]:
    """Different filters often peak on the same phrase; keep the strongest."""
    seen: dict[str, Phrase] = {}
    for phrase in phrases:
        existing = seen.get(phrase.text)
        if existing is None or phrase.contribution > existing.contribution:
            seen[phrase.text] = phrase
    return sorted(seen.values(), key=lambda p: p.contribution, reverse=True)


@torch.no_grad()
def explain_all(
    model: TextCNN,
    vocab: Vocabulary,
    text: str,
    max_tokens: int = 400,
    top_n: int = 5,
) -> dict[str, list[Phrase]]:
    return {
        name: explain(model, vocab, text, name, max_tokens=max_tokens, top_n=top_n)
        for name in LABEL_NAMES
    }
