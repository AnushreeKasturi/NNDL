from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from legal_risk_classifier.data import chunk_text, tokenizer_for_model
from legal_risk_classifier.labels import TARGET_LABELS
from legal_risk_classifier.models import build_multilabel_model


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    return tokenizer_for_model(model_name)


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_multilabel_model(model_name, num_labels=len(TARGET_LABELS))
    model.to(device)
    model.eval()
    return model, device


def infer_document(
    text: str,
    model_name: str,
    max_length: int,
    stride: int,
    threshold: float,
) -> dict:
    tokenizer = _load_tokenizer(model_name)
    model, device = _load_model(model_name)
    chunks = chunk_text(text, tokenizer=tokenizer, max_length=max_length, stride=stride)
    if not chunks:
        raise ValueError("No chunks produced for document.")

    probs = []
    with torch.no_grad():
        for chunk in chunks:
            enc = tokenizer(
                chunk,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            probs.append(torch.sigmoid(logits).cpu().numpy()[0])

    pooled = np.vstack(probs).max(axis=0)
    predicted_labels = [label for i, label in enumerate(TARGET_LABELS) if pooled[i] >= threshold]
    return {
        "chunk_count": len(chunks),
        "threshold": threshold,
        "scores": {label: float(pooled[i]) for i, label in enumerate(TARGET_LABELS)},
        "predicted_labels": predicted_labels,
    }

