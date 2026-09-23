"""CLI: package a run directory for the Hugging Face Hub, and optionally push it.

A training run saves what the training loop needs, which is not what a stranger
needs. For the transformer runs the checkpoint is a wrapper state dict with
`backbone.` prefixed keys, so uploading it unchanged would give people a file
they cannot load. This converts a run into the standard layout, writes a model
card carrying the run's real numbers, and labels the outputs so the model is
self-describing once loaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .labels import LABEL_NAMES
from .pretrained import PretrainedClassifier, load_tokenizer, resolve

CARD_HEADER = """---
license: mit
language: en
tags:
  - legal
  - contracts
  - multi-label-classification
  - cuad
datasets:
  - theatticusproject/cuad
pipeline_tag: text-classification
---
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None, help="default: <run_dir>/hf")
    parser.add_argument("--repo_id", type=str, default=None, help="e.g. username/model-name")
    parser.add_argument("--push", action="store_true", help="upload to the Hub")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def _load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def export_transformer(run_dir: Path, out_dir: Path, summary: dict) -> None:
    """Convert the wrapper checkpoint into a standard Hugging Face model directory."""
    spec = resolve(summary["checkpoint"])
    model = PretrainedClassifier(spec, num_labels=len(LABEL_NAMES))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))

    backbone = model.backbone
    # Without these the loaded model reports LABEL_0 ... LABEL_5 and the caller
    # has to know our ordering by hand.
    backbone.config.id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    backbone.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    backbone.config.problem_type = "multi_label_classification"
    backbone.save_pretrained(out_dir)

    tokenizer_dir = run_dir / "tokenizer"
    load_tokenizer(
        resolve(str(tokenizer_dir)) if tokenizer_dir.exists() else spec
    ).save_pretrained(out_dir)


def export_cnn(run_dir: Path, out_dir: Path) -> None:
    """Copy what the custom architecture needs to be rebuilt.

    The TextCNN is not a transformers architecture, so there is no
    `from_pretrained` for it. The card says so and points at the loader.
    """
    for name in ("model.pt", "vocab.json", "metrics.json", "training.json"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)


def render_card(summary: dict, run_dir: Path, repo_id: str | None) -> str:
    tuned = summary.get("test_at_tuned_thresholds", {})
    per_class = {row["label"]: row for row in tuned.get("per_class", [])}
    chunking = summary.get("chunking", {})
    is_cnn = summary["model"] == "textcnn"

    evaluation_path = run_dir / "evaluation_test.json"
    document = None
    if evaluation_path.exists():
        document = json.loads(evaluation_path.read_text(encoding="utf-8")).get("document_level")

    lines = [
        CARD_HEADER,
        f"# Clause risk detection — {summary.get('checkpoint', 'TextCNN')}",
        "",
        "Multi-label classifier flagging six risk-relevant clause types in commercial",
        "contracts, fine-tuned on [CUAD v1](https://www.atticusprojectai.org/cuad).",
        "",
        "Source and training code: https://github.com/ManasDasri/NNDL",
        "",
        "## Labels",
        "",
        "| Index | Label |",
        "| ---: | --- |",
    ]
    lines += [f"| {i} | {name} |" for i, name in enumerate(LABEL_NAMES)]

    lines += [
        "",
        "Six independent sigmoid outputs, not a softmax: contracts routinely carry",
        "several of these clause types at once.",
        "",
        "## Results",
        "",
        "Held-out test split, split by contract so no document appears in training.",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in LABEL_NAMES:
        row = per_class.get(name)
        if row:
            lines.append(
                f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1']:.3f} | {row['support']} |"
            )

    lines += ["", f"**Chunk-level macro F1: {tuned.get('macro_f1', float('nan')):.3f}**"]
    if document:
        lines.append(f"**Document-level macro F1: {document['macro_f1']:.3f}** (max-pooled over windows)")

    lines += [
        "",
        "## How it was trained",
        "",
        f"- Contracts are split into overlapping windows of {chunking.get('window', '?')} words "
        f"with {chunking.get('overlap', '?')} words of overlap, because 97% of CUAD contracts",
        "  exceed a 512-token context and 70% exceed 4,096.",
        "- A window takes a label when it meaningfully overlaps an annotated clause span.",
        "- Loss is BCE weighted by each label's negative-to-positive ratio, which runs 22x to",
        "  68x. 85% of windows carry no label at all, so an unweighted loss collapses to",
        "  predicting nothing.",
        "- Splits are assigned per contract, never per window.",
        "",
        "## Thresholds",
        "",
        "Decision thresholds are tuned per label on the validation split rather than fixed at",
        "0.5, because each label has its own positive rate. Using 0.5 will cost you recall.",
        "",
        "| Label | Threshold |",
        "| --- | ---: |",
    ]
    thresholds = summary.get("thresholds") or [0.5] * len(LABEL_NAMES)
    lines += [f"| {name} | {thresholds[i]:.2f} |" for i, name in enumerate(LABEL_NAMES)]

    lines += ["", "## Usage", ""]
    if is_cnn:
        lines += [
            "This is a TextCNN, not a transformers architecture, so there is no",
            "`from_pretrained` for it. Load it with the training repository:",
            "",
            "```python",
            "from legal_risk_classifier.runtime import CNNRuntime",
            "",
            "runtime = CNNRuntime('path/to/this/download')",
            "prediction = runtime.predict_document(contract_text)",
            "print(prediction.predicted, prediction.document_scores)",
            "```",
        ]
    else:
        lines += [
            "```python",
            "import torch",
            "from transformers import AutoModelForSequenceClassification, AutoTokenizer",
            "",
            f"name = \"{repo_id or 'your-username/your-model'}\"",
            "tokenizer = AutoTokenizer.from_pretrained(name)",
            "model = AutoModelForSequenceClassification.from_pretrained(name).eval()",
            "",
            "batch = tokenizer(window_text, truncation=True, max_length=512, return_tensors='pt')",
            "with torch.no_grad():",
            "    scores = torch.sigmoid(model(**batch).logits)[0]",
            "",
            "for i, score in enumerate(scores):",
            "    print(model.config.id2label[i], round(float(score), 3))",
            "```",
            "",
            "Pass one window at a time and take the maximum per label across a contract's",
            "windows; that is how the document-level figure above is computed.",
        ]

    lines += [
        "",
        "## Limitations",
        "",
        "- Trained on 358 contracts. The rarest label, Non-Compete, appears in around 1.5% of",
        "  windows and is correspondingly the weakest.",
        "- Run-to-run variance is about ±0.03 macro F1 on identical settings, so small",
        "  differences between checkpoints are not meaningful.",
        "- CUAD is US commercial contracts. Behaviour on other jurisdictions or contract",
        "  families is untested.",
        "- **This is a research artifact and not legal advice.** It is a triage aid for",
        "  locating clauses a lawyer should read, not a substitute for reading them.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = _load_summary(args.run_dir)
    out_dir = args.out_dir or args.run_dir / "hf"
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary["model"] == "textcnn":
        export_cnn(args.run_dir, out_dir)
    else:
        export_transformer(args.run_dir, out_dir, summary)

    (out_dir / "README.md").write_text(
        render_card(summary, args.run_dir, args.repo_id), encoding="utf-8"
    )

    files = sorted(p.name for p in out_dir.iterdir())
    size = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file()) / 1e6
    print(f"packaged {summary['model']} -> {out_dir}  ({size:.0f} MB)")
    for name in files:
        print(f"  {name}")

    if not args.push:
        print("\nnot pushed. Re-run with --push --repo_id <username>/<name> once logged in.")
        return

    if not args.repo_id:
        raise SystemExit("--push needs --repo_id")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=args.repo_id, repo_type="model")
    print(f"\npushed -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
