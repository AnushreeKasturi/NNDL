"""Checks for the vocabulary, the CNN, the metrics and the training loop.

Run with `.venv/bin/python tests/test_model_layer.py`. Requires torch and
scikit-learn, unlike the data-layer tests.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_risk_classifier.chunking import Chunk  # noqa: E402
from legal_risk_classifier.datasets import WordChunkDataset, label_matrix  # noqa: E402
from legal_risk_classifier.labels import LABEL_NAMES, NUM_LABELS  # noqa: E402
from legal_risk_classifier.metrics import (  # noqa: E402
    compute_metrics,
    positive_weights,
    tune_thresholds,
)
from legal_risk_classifier.textcnn import TextCNN  # noqa: E402
from legal_risk_classifier.training import TrainingConfig, predict, train  # noqa: E402
from legal_risk_classifier.vocab import PAD_INDEX, UNK_INDEX, Vocabulary, tokenize  # noqa: E402


def _chunk(text: str, labels: tuple[str, ...] = ()) -> Chunk:
    return Chunk(doc_id="d", index=0, char_start=0, char_end=len(text), text=text, labels=labels)


def test_tokenizer_keeps_legal_section_references() -> None:
    assert tokenize("Section 10.2 of the AGREEMENT") == ["section", "10.2", "of", "the", "agreement"]
    assert tokenize("indemnify, and hold harmless;") == ["indemnify", "and", "hold", "harmless"]


def test_vocabulary_pads_truncates_and_maps_unknowns() -> None:
    vocab = Vocabulary.build(["alpha beta alpha beta", "alpha beta"], min_freq=2)
    assert vocab.stoi["<pad>"] == PAD_INDEX and vocab.stoi["<unk>"] == UNK_INDEX

    padded = vocab.encode("alpha", max_tokens=4)
    assert len(padded) == 4 and padded[1:] == [PAD_INDEX] * 3, "right-padded"
    assert len(vocab.encode("alpha beta alpha beta alpha", max_tokens=3)) == 3, "truncated"
    assert vocab.encode("zzzz", max_tokens=1) == [UNK_INDEX], "unseen words map to <unk>"


def test_vocabulary_honours_min_freq() -> None:
    vocab = Vocabulary.build(["common common rare"], min_freq=2)
    assert "common" in vocab.stoi
    assert "rare" not in vocab.stoi, "a singleton is dropped"


def test_vocabulary_round_trips() -> None:
    vocab = Vocabulary.build(["alpha beta alpha beta"], min_freq=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocab.json"
        vocab.save(path)
        assert Vocabulary.load(path).itos == vocab.itos


def test_label_matrix_is_multi_hot() -> None:
    chunks = [_chunk("a", ("Insurance",)), _chunk("b", ()), _chunk("c", LABEL_NAMES[:2])]
    matrix = label_matrix(chunks)
    assert matrix.shape == (3, NUM_LABELS)
    assert matrix[0].sum() == 1 and matrix[1].sum() == 0 and matrix[2].sum() == 2


def test_cnn_output_shape_and_short_input_handling() -> None:
    model = TextCNN(vocab_size=50, num_labels=NUM_LABELS, embedding_dim=16, num_filters=8)
    assert model(torch.randint(2, 50, (4, 40))).shape == (4, NUM_LABELS)
    # Shorter than the widest kernel: must pad rather than raise.
    assert model(torch.randint(2, 50, (2, 2))).shape == (2, NUM_LABELS)


def test_cnn_pooling_is_position_invariant() -> None:
    """Max-over-time means a clause counts wherever it sits in the chunk."""
    model = TextCNN(vocab_size=50, num_labels=NUM_LABELS, embedding_dim=16, num_filters=8).eval()
    phrase = torch.tensor([7, 8, 9, 10])
    # Pad both sides, wider than the largest kernel, so every window that sees
    # the phrase sees identical context and only its position changes.
    def place(before: int, after: int) -> torch.Tensor:
        return torch.cat(
            [torch.full((before,), PAD_INDEX), phrase, torch.full((after,), PAD_INDEX)]
        ).unsqueeze(0)

    early, late = place(6, 26), place(26, 6)
    with torch.no_grad():
        assert torch.allclose(model(early), model(late), atol=1e-5)


def test_positive_weights_match_the_negative_to_positive_ratio() -> None:
    y_true = np.zeros((100, NUM_LABELS), dtype=np.float32)
    y_true[:10, 0] = 1.0  # 10 positives, 90 negatives
    weights = positive_weights(y_true)
    assert abs(weights[0] - 9.0) < 1e-6
    assert weights[1] == 100.0, "a label with no positives must not divide by zero"


def test_metrics_expose_the_all_negative_failure() -> None:
    """The failure mode this project has to detect: predicting nothing."""
    y_true = np.zeros((100, NUM_LABELS), dtype=np.float32)
    y_true[:3, 0] = 1.0
    y_prob = np.zeros_like(y_true)  # predicts nothing, anywhere

    result = compute_metrics(y_true, y_prob, thresholds=0.5)
    assert result["macro_f1"] == 0.0, "F1 exposes it even though accuracy would be 97%"
    assert result["per_class"][0]["support"] == 3
    assert result["per_class"][0]["predicted"] == 0


def test_metrics_accept_per_label_thresholds() -> None:
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]] * 5, dtype=np.float32)
    y_true = np.pad(y_true, ((0, 0), (0, NUM_LABELS - 2)))
    y_prob = np.full_like(y_true, 0.4)
    assert compute_metrics(y_true, y_prob, thresholds=0.5)["per_class"][0]["predicted"] == 0
    per_label = [0.3] + [0.5] * (NUM_LABELS - 1)
    assert compute_metrics(y_true, y_prob, thresholds=per_label)["per_class"][0]["predicted"] == 10


def test_metrics_reject_mismatched_shapes() -> None:
    try:
        compute_metrics(np.zeros((4, NUM_LABELS)), np.zeros((5, NUM_LABELS)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on a shape mismatch")


def test_tuned_thresholds_beat_a_fixed_half() -> None:
    """Rare labels need a low cut; 0.5 is arbitrary under 2% positive rates."""
    rng = np.random.default_rng(0)
    y_true = np.zeros((400, NUM_LABELS), dtype=np.float32)
    y_true[:20, 0] = 1.0
    y_prob = rng.uniform(0.0, 0.25, size=y_true.shape)
    y_prob[:20, 0] += 0.2  # positives score higher, but never reach 0.5

    thresholds = tune_thresholds(y_true, y_prob)
    assert thresholds[0] < 0.5
    tuned = compute_metrics(y_true, y_prob, thresholds=thresholds)["per_class"][0]["f1"]
    fixed = compute_metrics(y_true, y_prob, thresholds=0.5)["per_class"][0]["f1"]
    assert fixed == 0.0 and tuned > fixed


def test_training_loop_learns_a_separable_signal() -> None:
    """End-to-end: the loop must reduce loss and restore the best weights."""
    vocab = Vocabulary.build(["insurance clause here", "unrelated filler text"], min_freq=1)
    positive = [_chunk("insurance clause here", ("Insurance",)) for _ in range(24)]
    negative = [_chunk("unrelated filler text", ()) for _ in range(24)]
    dataset = WordChunkDataset(positive + negative, vocab, max_tokens=8)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TextCNN(vocab_size=len(vocab), num_labels=NUM_LABELS, embedding_dim=16, num_filters=16)
    config = TrainingConfig(epochs=12, learning_rate=5e-2, patience=12, device="cpu")
    result = train(model, loader, loader, config)

    assert result.history[-1].train_loss < result.history[0].train_loss, "loss falls"
    assert len(result.thresholds) == NUM_LABELS

    y_true, y_prob = predict(model, loader, "cpu")
    assert y_true.shape == y_prob.shape == (48, NUM_LABELS)
    assert ((y_prob >= 0) & (y_prob <= 1)).all(), "predict returns probabilities"

    # Only Insurance has positives here, so macro F1 over all six labels caps
    # at 1/6 even when the signal is learned perfectly. Score the label that
    # carries support.
    insurance = compute_metrics(y_true, y_prob, thresholds=result.thresholds)["per_class"][
        LABEL_NAMES.index("Insurance")
    ]
    assert insurance["f1"] > 0.9, "a separable signal must be learnable"
    assert insurance["average_precision"] > 0.9


def test_training_writes_a_resumable_artifact() -> None:
    vocab = Vocabulary.build(["alpha beta"], min_freq=1)
    loader = DataLoader(
        WordChunkDataset([_chunk("alpha beta", ("Insurance",))] * 4, vocab, max_tokens=8),
        batch_size=2,
    )
    model = TextCNN(vocab_size=len(vocab), num_labels=NUM_LABELS, embedding_dim=8, num_filters=4)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        train(model, loader, loader, TrainingConfig(epochs=1, device="cpu"), output_dir=out)
        assert (out / "model.pt").exists() and (out / "training.json").exists()
        reloaded = TextCNN(vocab_size=len(vocab), num_labels=NUM_LABELS, embedding_dim=8, num_filters=4)
        reloaded.load_state_dict(torch.load(out / "model.pt"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
