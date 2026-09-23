"""TextCNN for multi-label clause classification (Kim, 2014).

Convolutions over the embedded sequence act as learned n-gram detectors, and
max-over-time pooling reports whether each detector fired anywhere in the
chunk. That matches the task: a clause is signalled by a phrase such as "in no
event shall" or "shall maintain insurance", and its position within the chunk
does not matter.

The model is also the honest floor for the transformer comparison. If a model
pretrained on 3 billion words cannot beat convolutions over embeddings learned
from 350 contracts, the pretraining is not earning its cost.
"""

from __future__ import annotations

import torch
from torch import nn

from .vocab import PAD_INDEX


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)
        self.min_length = max(kernel_sizes)

    def forward(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        """input_ids (batch, length) -> logits (batch, num_labels)."""
        if input_ids.size(1) < self.min_length:
            # A chunk shorter than the widest kernel would make that convolution
            # undefined; pad rather than fail, since padding embeds to zeros.
            pad = self.min_length - input_ids.size(1)
            input_ids = nn.functional.pad(input_ids, (0, pad), value=PAD_INDEX)

        embedded = self.embedding(input_ids).transpose(1, 2)  # (batch, channels, length)
        pooled = [
            torch.relu(convolution(embedded)).max(dim=2).values
            for convolution in self.convolutions
        ]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))
