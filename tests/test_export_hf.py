"""Checks for Hugging Face packaging.

Uses a tiny randomly-initialised BERT so the suite stays offline and fast.
Run with `.venv/bin/python tests/test_export_hf.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_risk_classifier.export_hf import export_cnn, export_transformer, render_card  # noqa: E402
from legal_risk_classifier.labels import LABEL_NAMES, NUM_LABELS  # noqa: E402
from legal_risk_classifier.pretrained import ModelSpec, PretrainedClassifier  # noqa: E402

SUMMARY = {
    "model": "legal-bert",
    "checkpoint": "tiny",
    "chunking": {"window": 300, "overlap": 100, "max_length": 64},
    "thresholds": [0.31, 0.42, 0.53, 0.64, 0.25, 0.77],
    "test_at_tuned_thresholds": {
        "macro_f1": 0.663,
        "per_class": [
            {"label": name, "precision": 0.7, "recall": 0.8, "f1": 0.75, "support": 50}
            for name in LABEL_NAMES
        ],
    },
}


def _tiny_model() -> PretrainedClassifier:
    from transformers import BertConfig, BertForSequenceClassification

    spec = ModelSpec(key="tiny", checkpoint="tiny", max_length=64, window_words=20, overlap_words=5)
    model = PretrainedClassifier.__new__(PretrainedClassifier)
    torch.nn.Module.__init__(model)
    model.spec = spec
    model.backbone = BertForSequenceClassification(
        BertConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=64, max_position_embeddings=64, num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
    )
    return model


def _exported(tmp: Path) -> tuple[Path, PretrainedClassifier]:
    """Write a run directory and export it, the way the CLI would."""
    model = _tiny_model()
    run, out = tmp / "run", tmp / "hf"
    run.mkdir(); out.mkdir()
    torch.save(model.state_dict(), run / "model.pt")
    (run / "metrics.json").write_text(json.dumps(SUMMARY), encoding="utf-8")

    # Stand in for the real checkpoint, which the test must not download.
    import legal_risk_classifier.export_hf as module

    original = module.resolve
    module.resolve = lambda name: model.spec
    module.PretrainedClassifier = lambda spec, num_labels: _tiny_model()
    saved_tokenizer = module.load_tokenizer
    module.load_tokenizer = lambda spec: _DummyTokenizer()
    try:
        export_transformer(run, out, SUMMARY)
    finally:
        module.resolve = original
        module.PretrainedClassifier = PretrainedClassifier
        module.load_tokenizer = saved_tokenizer
    return out, model


class _DummyTokenizer:
    def save_pretrained(self, path):  # noqa: D401 - stands in for a real tokenizer
        Path(path, "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_export_writes_a_loadable_model_directory() -> None:
    """A stranger must be able to call from_pretrained on what we upload."""
    from transformers import AutoModelForSequenceClassification

    with tempfile.TemporaryDirectory() as tmp:
        out, _ = _exported(Path(tmp))
        names = {p.name for p in out.iterdir()}
        assert "config.json" in names
        assert "model.safetensors" in names or "pytorch_model.bin" in names

        loaded = AutoModelForSequenceClassification.from_pretrained(out)
        assert loaded.config.num_labels == NUM_LABELS


def test_exported_model_is_self_describing() -> None:
    """Without id2label a caller has to know our label ordering by hand."""
    from transformers import AutoModelForSequenceClassification

    with tempfile.TemporaryDirectory() as tmp:
        out, _ = _exported(Path(tmp))
        config = AutoModelForSequenceClassification.from_pretrained(out).config
        assert [config.id2label[i] for i in range(NUM_LABELS)] == list(LABEL_NAMES)
        assert config.label2id[LABEL_NAMES[0]] == 0
        assert config.problem_type == "multi_label_classification"


def test_the_backbone_prefix_is_stripped() -> None:
    """The run checkpoint has backbone.* keys; an uploaded model must not."""
    with tempfile.TemporaryDirectory() as tmp:
        out, model = _exported(Path(tmp))
        assert any(k.startswith("backbone.") for k in model.state_dict()), "wrapper is prefixed"

        from safetensors.torch import load_file

        exported = load_file(out / "model.safetensors")
        assert exported, "something was written"
        assert not any(k.startswith("backbone.") for k in exported), "prefix removed on export"


def test_card_carries_labels_thresholds_and_a_disclaimer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        card = render_card(SUMMARY, Path(tmp), "user/model")

    assert card.startswith("---"), "Hub frontmatter comes first"
    for name in LABEL_NAMES:
        assert name in card
    assert "0.31" in card and "0.77" in card, "per-label thresholds are published"
    assert "0.663" in card, "the real macro F1, not a placeholder"
    assert "not legal advice" in card
    assert "±0.03" in card, "the variance caveat travels with the model"
    assert "from_pretrained" in card, "transformer card shows transformers usage"


def test_cnn_card_does_not_promise_from_pretrained() -> None:
    """The TextCNN is not a transformers architecture; the card must say so."""
    summary = dict(SUMMARY, model="textcnn")
    summary.pop("checkpoint", None)
    with tempfile.TemporaryDirectory() as tmp:
        card = render_card(summary, Path(tmp), None)
    assert "CNNRuntime" in card
    assert "AutoModelForSequenceClassification" not in card


def test_cnn_export_copies_what_the_loader_needs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run, out = Path(tmp) / "run", Path(tmp) / "hf"
        run.mkdir(); out.mkdir()
        for name in ("model.pt", "vocab.json", "metrics.json", "training.json"):
            (run / name).write_text("{}", encoding="utf-8")
        export_cnn(run, out)
        assert {p.name for p in out.iterdir()} == {"model.pt", "vocab.json", "metrics.json", "training.json"}


def test_cnn_export_tolerates_a_missing_optional_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run, out = Path(tmp) / "run", Path(tmp) / "hf"
        run.mkdir(); out.mkdir()
        (run / "model.pt").write_text("{}", encoding="utf-8")
        export_cnn(run, out)
        assert {p.name for p in out.iterdir()} == {"model.pt"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
