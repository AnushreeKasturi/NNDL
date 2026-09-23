"""CLI: score a trained checkpoint at chunk and document level.

Reported separately because they answer different questions. Chunk-level says
whether the model can recognise a clause in front of it; document-level says
whether the system finds the clause somewhere in a contract, which is what the
tool would actually be used for.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .chunking import chunk_documents
from .cuad import load_documents
from .datasets import TokenizedChunkDataset, WordChunkDataset
from .labels import NUM_LABELS
from .metrics import compute_metrics, tune_thresholds
from .pooling import pool_to_documents
from .splits import load_splits, select
from .training import predict, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", type=Path, required=True, help="a training output directory")
    parser.add_argument("--documents", type=Path, default=Path("data/processed/documents.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pooling", default="max", choices=("max", "mean", "top_k"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _rebuild(run_dir: Path):
    """Reconstruct the model and its chunking from a run directory."""
    summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    chunking = summary["chunking"]
    state = torch.load(run_dir / "model.pt", map_location="cpu")

    if summary["model"] == "textcnn":
        from .textcnn import TextCNN
        from .vocab import Vocabulary

        vocab = Vocabulary.load(run_dir / "vocab.json")
        # The architecture is recoverable from the checkpoint's own shapes,
        # so a run directory stays self-describing.
        conv_weights = sorted(
            key for key in state if key.startswith("convolutions.") and key.endswith(".weight")
        )
        model = TextCNN(
            vocab_size=len(vocab),
            num_labels=NUM_LABELS,
            embedding_dim=state["embedding.weight"].shape[1],
            num_filters=state[conv_weights[0]].shape[0],
            kernel_sizes=tuple(state[key].shape[2] for key in conv_weights),
        )
        model.load_state_dict(state)
        return model, summary, chunking, ("cnn", vocab)

    from .pretrained import PretrainedClassifier, load_tokenizer, resolve

    spec = resolve(summary["checkpoint"])
    model = PretrainedClassifier(spec, num_labels=NUM_LABELS)
    model.load_state_dict(state)
    tokenizer_dir = run_dir / "tokenizer"
    if tokenizer_dir.exists():
        spec = replace(spec, checkpoint=str(tokenizer_dir))
    return model, summary, chunking, ("hf", load_tokenizer(spec))


def main() -> None:
    args = parse_args()
    model, summary, chunking, (kind, encoder) = _rebuild(args.run_dir)

    documents = select(load_documents(args.documents), load_splits(args.splits), args.split)
    chunks = chunk_documents(
        documents, window=chunking["window"], overlap=chunking["overlap"]
    )
    dataset = (
        WordChunkDataset(chunks, encoder, max_tokens=chunking["max_tokens"])
        if kind == "cnn"
        else TokenizedChunkDataset(chunks, encoder, max_length=chunking["max_length"])
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    model.to(device)
    y_true, y_prob = predict(model, loader, device)

    # training.json is written by the shared loop for every model, so it is the
    # one place validation-tuned thresholds are guaranteed to be found. Falling
    # back to 0.5 silently would report a different operating point than the
    # training run did.
    training_path = args.run_dir / "training.json"
    thresholds = (
        json.loads(training_path.read_text(encoding="utf-8"))["thresholds"]
        if training_path.exists()
        else 0.5
    )
    chunk_metrics = compute_metrics(y_true, y_prob, thresholds=thresholds)

    doc_ids, doc_true, doc_prob = pool_to_documents(chunks, y_prob, method=args.pooling)
    # Document-level thresholds are their own operating point: a pooled max is
    # systematically higher than a chunk probability, so reusing the chunk
    # thresholds would over-predict.
    doc_thresholds = tune_thresholds(doc_true, doc_prob)
    document_metrics = compute_metrics(doc_true, doc_prob, thresholds=doc_thresholds)

    result = {
        "run_dir": str(args.run_dir),
        "model": summary["model"],
        "split": args.split,
        "pooling": args.pooling,
        "counts": {"documents": len(doc_ids), "chunks": len(chunks)},
        "chunk_level": chunk_metrics,
        "document_level": document_metrics,
    }
    out = args.out or args.run_dir / f"evaluation_{args.split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"{summary['model']}  {args.split}  ({len(chunks)} chunks, {len(doc_ids)} documents)")
    print(f"  chunk    macro F1 {chunk_metrics['macro_f1']:.3f}  AP {chunk_metrics['macro_average_precision']:.3f}")
    print(f"  document macro F1 {document_metrics['macro_f1']:.3f}  AP {document_metrics['macro_average_precision']:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
