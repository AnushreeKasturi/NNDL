"""The pretrained encoders under comparison, behind one interface.

Three models, chosen to isolate two variables rather than to collect
architectures. BERT and Legal-BERT share an architecture, a parameter count
and a context window, differing only in pretraining corpus, so a difference
between them measures domain adaptation and nothing else. Longformer changes
the context window while staying a transformer encoder of comparable size, so
a difference against BERT measures context length.

See docs/model-selection.md for the full justification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    key: str
    checkpoint: str
    max_length: int
    # Chunk width in words, sized so the tokenized chunk fits max_length with
    # room for special tokens. Legal English inflates to roughly 1.45 subword
    # tokens per word.
    window_words: int
    overlap_words: int
    needs_global_attention: bool = False
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "bert": ModelSpec(
        key="bert",
        checkpoint="bert-base-uncased",
        max_length=512,
        window_words=300,
        overlap_words=100,
        note="General-domain baseline. Isolates the effect of legal pretraining.",
    ),
    "legal-bert": ModelSpec(
        key="legal-bert",
        checkpoint="nlpaueb/legal-bert-base-uncased",
        max_length=512,
        window_words=300,
        overlap_words=100,
        note="Same architecture as BERT, pretrained on legislation, contracts and case law.",
    ),
    "longformer": ModelSpec(
        key="longformer",
        checkpoint="allenai/longformer-base-4096",
        max_length=4096,
        window_words=2600,
        overlap_words=400,
        needs_global_attention=True,
        note="Sliding-window attention over an 8x longer context.",
    ),
}


def resolve(name: str) -> ModelSpec:
    """A registry key, or any Hugging Face checkpoint id."""
    if name in MODELS:
        return MODELS[name]
    return ModelSpec(
        key=name,
        checkpoint=name,
        max_length=512,
        window_words=300,
        overlap_words=100,
        note="Supplied directly as a Hugging Face checkpoint.",
    )


class PretrainedClassifier(nn.Module):
    """Adapts a Hugging Face encoder to the interface the training loop expects.

    The loop calls `model(**batch)` and receives a logits tensor, exactly as it
    does for the CNN, which is what lets one loop train every model.
    """

    def __init__(self, spec: ModelSpec, num_labels: int) -> None:
        super().__init__()
        self.spec = spec
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            spec.checkpoint,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        inputs: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        if self.spec.needs_global_attention:
            # Longformer's local attention cannot pool the sequence on its own.
            # [CLS] is given global attention so it can attend to every token
            # and be attended by every token, which is what the classification
            # head reads.
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
            inputs["global_attention_mask"] = global_attention_mask

        return self.backbone(**inputs).logits

    def gradient_checkpointing_enable(self) -> None:
        self.backbone.gradient_checkpointing_enable()


def load_tokenizer(spec: ModelSpec):
    return AutoTokenizer.from_pretrained(spec.checkpoint, use_fast=True)
