"""Checks for the pretrained-encoder layer.

Uses a tiny randomly-initialised BERT so the suite stays offline-friendly and
fast; the point is the wiring, not the weights. Run with
`.venv/bin/python tests/test_pretrained.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_risk_classifier.labels import NUM_LABELS  # noqa: E402
from legal_risk_classifier.pretrained import (  # noqa: E402
    MODELS,
    ModelSpec,
    PretrainedClassifier,
    resolve,
)

TOKENS_PER_WORD = 1.45


def _tiny(**overrides) -> PretrainedClassifier:
    """A 2-layer BERT with random weights, built without touching the network."""
    from transformers import BertConfig, BertForSequenceClassification

    spec = ModelSpec(
        key="tiny",
        checkpoint="tiny",
        max_length=overrides.pop("max_length", 64),
        window_words=20,
        overlap_words=5,
        needs_global_attention=overrides.pop("needs_global_attention", False),
    )
    model = PretrainedClassifier.__new__(PretrainedClassifier)
    torch.nn.Module.__init__(model)
    model.spec = spec
    model.backbone = BertForSequenceClassification(
        BertConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=spec.max_length,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
    )
    return model


def test_registry_covers_the_three_compared_models() -> None:
    assert set(MODELS) == {"bert", "legal-bert", "longformer"}
    assert MODELS["bert"].checkpoint == "bert-base-uncased"
    assert MODELS["legal-bert"].checkpoint == "nlpaueb/legal-bert-base-uncased"
    assert MODELS["longformer"].checkpoint == "allenai/longformer-base-4096"


def test_bert_and_legal_bert_differ_only_in_pretraining() -> None:
    """Problem 1 is only answerable if everything else is held equal."""
    bert, legal = MODELS["bert"], MODELS["legal-bert"]
    assert bert.max_length == legal.max_length == 512
    assert bert.window_words == legal.window_words
    assert bert.overlap_words == legal.overlap_words
    assert bert.needs_global_attention == legal.needs_global_attention is False
    assert bert.checkpoint != legal.checkpoint, "the pretraining corpus is the only difference"


def test_chunk_widths_fit_their_context_windows() -> None:
    """A window that overflows would truncate, silently undoing the experiment."""
    for spec in MODELS.values():
        estimated = spec.window_words * TOKENS_PER_WORD
        assert estimated < spec.max_length - 2, (
            f"{spec.key}: {spec.window_words} words is about {estimated:.0f} tokens, "
            f"over the {spec.max_length} limit"
        )
        assert 0 <= spec.overlap_words < spec.window_words


def test_longformer_gets_a_longer_window_than_bert() -> None:
    """Problem 2 requires the context actually to differ."""
    assert MODELS["longformer"].window_words > 4 * MODELS["bert"].window_words


def test_only_longformer_requests_global_attention() -> None:
    assert MODELS["longformer"].needs_global_attention is True
    assert not MODELS["bert"].needs_global_attention
    assert not MODELS["legal-bert"].needs_global_attention


def test_resolve_accepts_a_registry_key_or_a_checkpoint_id() -> None:
    assert resolve("legal-bert") is MODELS["legal-bert"]
    custom = resolve("some-org/some-model")
    assert custom.checkpoint == "some-org/some-model" and custom.max_length == 512


def test_forward_returns_bare_logits_like_the_cnn() -> None:
    """The training loop expects a tensor, not a ModelOutput."""
    model = _tiny().eval()
    batch = {
        "input_ids": torch.randint(0, 64, (3, 16)),
        "attention_mask": torch.ones(3, 16, dtype=torch.long),
    }
    with torch.no_grad():
        logits = model(**batch)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (3, NUM_LABELS)


def test_forward_ignores_extra_batch_keys() -> None:
    """Tokenizers emit token_type_ids and friends; the loop passes them through."""
    model = _tiny().eval()
    with torch.no_grad():
        logits = model(
            input_ids=torch.randint(0, 64, (2, 16)),
            attention_mask=torch.ones(2, 16, dtype=torch.long),
            token_type_ids=torch.zeros(2, 16, dtype=torch.long),
        )
    assert logits.shape == (2, NUM_LABELS)


def test_global_attention_marks_cls_only() -> None:
    """Longformer's head reads [CLS], which local attention alone cannot fill."""
    captured: dict[str, torch.Tensor] = {}
    model = _tiny(needs_global_attention=True).eval()
    original = model.backbone.forward

    def spy(**kwargs):
        captured.update(kwargs)
        kwargs.pop("global_attention_mask", None)
        return original(**kwargs)

    model.backbone.forward = spy
    with torch.no_grad():
        model(input_ids=torch.randint(0, 64, (2, 16)), attention_mask=torch.ones(2, 16, dtype=torch.long))

    mask = captured["global_attention_mask"]
    assert mask.shape == (2, 16)
    assert (mask[:, 0] == 1).all(), "[CLS] is global"
    assert mask[:, 1:].sum() == 0, "nothing else is"


def test_no_global_attention_mask_for_bert() -> None:
    captured: dict[str, torch.Tensor] = {}
    model = _tiny(needs_global_attention=False).eval()
    original = model.backbone.forward

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    model.backbone.forward = spy
    with torch.no_grad():
        model(input_ids=torch.randint(0, 64, (2, 16)), attention_mask=torch.ones(2, 16, dtype=torch.long))
    assert "global_attention_mask" not in captured


def test_pretrained_model_trains_through_the_shared_loop() -> None:
    """The same loop must drive a transformer and the CNN alike."""
    from torch.utils.data import DataLoader, Dataset

    from legal_risk_classifier.training import TrainingConfig, train

    class Fake(Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            labels = torch.zeros(NUM_LABELS)
            labels[0] = float(index % 2)
            return {
                "input_ids": torch.randint(0, 64, (16,)),
                "attention_mask": torch.ones(16, dtype=torch.long),
                "labels": labels,
            }

    loader = DataLoader(Fake(), batch_size=4)
    result = train(_tiny(), loader, loader, TrainingConfig(epochs=2, device="cpu", learning_rate=1e-3))
    assert len(result.history) == 2
    assert len(result.thresholds) == NUM_LABELS


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
