"""Checks for filter attribution, the inference runtime and the demo callback.

Run with `.venv/bin/python tests/test_interpretability.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_risk_classifier.attribution import explain, explain_all  # noqa: E402
from legal_risk_classifier.labels import LABEL_NAMES, LABEL_TO_INDEX, NUM_LABELS  # noqa: E402
from legal_risk_classifier.textcnn import TextCNN  # noqa: E402
from legal_risk_classifier.training import TrainingConfig, train  # noqa: E402
from legal_risk_classifier.vocab import Vocabulary  # noqa: E402

CLAUSE = "the supplier shall maintain comprehensive general liability insurance"
FILLER = "this agreement is written in the english language and duly executed"


def _trained_pair(tmp: Path) -> tuple[TextCNN, Vocabulary]:
    """A tiny CNN actually trained to separate CLAUSE from FILLER."""
    from torch.utils.data import DataLoader

    from legal_risk_classifier.chunking import Chunk
    from legal_risk_classifier.datasets import WordChunkDataset

    vocab = Vocabulary.build([CLAUSE, FILLER], min_freq=1)

    def chunk(text: str, positive: bool) -> Chunk:
        return Chunk(
            doc_id="d",
            index=0,
            char_start=0,
            char_end=len(text),
            text=text,
            labels=("Insurance",) if positive else (),
        )

    rows = [chunk(CLAUSE, True) for _ in range(16)] + [chunk(FILLER, False) for _ in range(16)]
    loader = DataLoader(WordChunkDataset(rows, vocab, max_tokens=16), batch_size=8, shuffle=True)
    model = TextCNN(len(vocab), NUM_LABELS, embedding_dim=24, num_filters=24)
    train(model, loader, loader, TrainingConfig(epochs=15, learning_rate=5e-2, device="cpu", patience=15))
    return model, vocab


def test_explain_returns_phrases_from_the_input_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model, vocab = _trained_pair(Path(tmp))
    phrases = explain(model, vocab, CLAUSE, "Insurance", max_tokens=16, top_n=5)

    assert phrases, "a trained model must attribute something"
    words = CLAUSE.split()
    for phrase in phrases:
        assert phrase.text in " ".join(words), "a phrase is a real span of the input"
        assert phrase.contribution > 0, "only positive contributions are reported"
    assert phrases == sorted(phrases, key=lambda p: p.contribution, reverse=True)


def test_explain_finds_the_discriminating_words() -> None:
    """The point of the feature: the phrase should be why, not noise."""
    with tempfile.TemporaryDirectory() as tmp:
        model, vocab = _trained_pair(Path(tmp))
    phrases = explain(model, vocab, CLAUSE, "Insurance", max_tokens=16, top_n=6)
    joined = " ".join(phrase.text for phrase in phrases)
    assert "insurance" in joined or "liability" in joined, f"expected the clause words, got {joined!r}"


def test_phrases_are_deduplicated_keeping_the_strongest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model, vocab = _trained_pair(Path(tmp))
    phrases = explain(model, vocab, CLAUSE, "Insurance", max_tokens=16, top_n=10)
    texts = [phrase.text for phrase in phrases]
    assert len(texts) == len(set(texts)), "different filters peaking on one phrase collapse"


def test_explain_handles_empty_and_unknown_input() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model, vocab = _trained_pair(Path(tmp))
    assert explain(model, vocab, "", "Insurance") == []
    assert explain(model, vocab, "   ", "Insurance") == []
    try:
        explain(model, vocab, CLAUSE, "Not A Label")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown label")


def test_explain_all_covers_every_label() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model, vocab = _trained_pair(Path(tmp))
    assert set(explain_all(model, vocab, CLAUSE, max_tokens=16)) == set(LABEL_NAMES)


def test_phrase_word_offsets_are_within_the_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model, vocab = _trained_pair(Path(tmp))
    words = CLAUSE.split()
    for phrase in explain(model, vocab, CLAUSE, "Insurance", max_tokens=16, top_n=6):
        assert 0 <= phrase.start_word < phrase.end_word <= len(words)


def _write_run(tmp: Path) -> Path:
    """A minimal run directory the runtime can load."""
    model, vocab = _trained_pair(tmp)
    run_dir = tmp / "run"
    run_dir.mkdir()
    torch.save(model.state_dict(), run_dir / "model.pt")
    vocab.save(run_dir / "vocab.json")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": "textcnn",
                "chunking": {"window": 8, "overlap": 2, "max_tokens": 16},
                "test_at_tuned_thresholds": {"macro_f1": 0.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training.json").write_text(
        json.dumps({"thresholds": [0.5] * NUM_LABELS}), encoding="utf-8"
    )
    return run_dir


def test_runtime_rebuilds_the_model_from_a_run_directory() -> None:
    from legal_risk_classifier.runtime import CNNRuntime

    with tempfile.TemporaryDirectory() as tmp:
        runtime = CNNRuntime(_write_run(Path(tmp)))
        assert runtime.threshold_for("Insurance") == 0.5

        prediction = runtime.predict_document(CLAUSE + " " + FILLER)
        assert prediction.chunk_probs.shape[1] == NUM_LABELS
        assert set(prediction.document_scores) == set(LABEL_NAMES)
        assert "Insurance" in prediction.predicted, "the clause it was trained on must flag"


def test_runtime_pools_with_the_maximum_over_chunks() -> None:
    """The runtime must agree with the evaluation path, not approximate it."""
    from legal_risk_classifier.runtime import CNNRuntime

    with tempfile.TemporaryDirectory() as tmp:
        runtime = CNNRuntime(_write_run(Path(tmp)))
        text = " ".join([FILLER] * 6 + [CLAUSE] + [FILLER] * 6)
        prediction = runtime.predict_document(text)
        index = LABEL_TO_INDEX["Insurance"]
        assert abs(
            prediction.document_scores["Insurance"] - float(prediction.chunk_probs[:, index].max())
        ) < 1e-6


def test_runtime_handles_empty_text() -> None:
    from legal_risk_classifier.runtime import CNNRuntime

    with tempfile.TemporaryDirectory() as tmp:
        runtime = CNNRuntime(_write_run(Path(tmp)))
        prediction = runtime.predict_document("   ")
        assert prediction.chunks == [] and prediction.predicted == []


def test_hits_are_sorted_and_respect_the_threshold() -> None:
    from legal_risk_classifier.runtime import CNNRuntime

    with tempfile.TemporaryDirectory() as tmp:
        runtime = CNNRuntime(_write_run(Path(tmp)))
        prediction = runtime.predict_document(" ".join([CLAUSE, FILLER] * 4))
        hits = prediction.hits("Insurance", threshold=0.5)
        assert all(hit.score >= 0.5 for hit in hits)
        assert hits == sorted(hits, key=lambda hit: hit.score, reverse=True)
        assert prediction.hits("Insurance", threshold=1.1) == [], "nothing clears an impossible cut"


def test_demo_callback_returns_three_renderable_fields() -> None:
    from legal_risk_classifier.demo import analyse
    from legal_risk_classifier.runtime import CNNRuntime

    with tempfile.TemporaryDirectory() as tmp:
        runtime = CNNRuntime(_write_run(Path(tmp)))
        scores, passages, triggers = analyse(runtime, CLAUSE, "Insurance")
        assert len(scores) == NUM_LABELS
        assert all("thr" in key for key in scores), "each row shows its own threshold"
        assert isinstance(passages, str) and passages
        assert isinstance(triggers, str) and triggers

        empty_scores, message, _ = analyse(runtime, "  ", "Insurance")
        assert empty_scores == {} and "Paste" in message


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
