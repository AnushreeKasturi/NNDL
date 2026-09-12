from __future__ import annotations

from transformers import AutoModelForSequenceClassification


MODEL_CHOICES = {
    "bert": "bert-base-uncased",
    "legal-bert": "nlpaueb/legal-bert-base-uncased",
    "longformer": "allenai/longformer-base-4096",
}


def resolve_model_name(choice_or_name: str) -> str:
    return MODEL_CHOICES.get(choice_or_name, choice_or_name)


def build_multilabel_model(model_name: str, num_labels: int) -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

