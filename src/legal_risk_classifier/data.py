from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .labels import LABEL_TO_INDEX, TARGET_LABELS, normalize_label


@dataclass(frozen=True)
class LabeledText:
    doc_id: str
    text: str
    labels: tuple[str, ...]


class ClauseDataset(Dataset):
    def __init__(
        self,
        rows: list[LabeledText],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        enc = self.tokenizer(
            row.text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        target = torch.zeros(len(TARGET_LABELS), dtype=torch.float32)
        for label in row.labels:
            target[LABEL_TO_INDEX[label]] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": target,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
) -> list[str]:
    if max_length <= stride:
        raise ValueError("max_length must be greater than stride.")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    step = max_length - stride
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + max_length]
        if not window:
            continue
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + max_length >= len(token_ids):
            break
    return chunks


def _iter_cuad_paragraphs(raw: dict) -> Iterable[tuple[str, str, list[dict]]]:
    data = raw.get("data", [])
    for doc_i, doc in enumerate(data):
        title = str(doc.get("title", f"doc-{doc_i}"))
        paragraphs = doc.get("paragraphs", [])
        for p_i, paragraph in enumerate(paragraphs):
            context = str(paragraph.get("context", "")).strip()
            qas = paragraph.get("qas", [])
            doc_id = f"{title}::p{p_i}"
            if context:
                yield doc_id, context, qas


def convert_cuad_to_rows(
    cuad_json_path: str | Path,
    include_unlabeled: bool = True,
) -> list[LabeledText]:
    raw = json.loads(Path(cuad_json_path).read_text(encoding="utf-8"))
    out: list[LabeledText] = []
    for doc_id, context, qas in _iter_cuad_paragraphs(raw):
        labels: set[str] = set()
        for qa in qas:
            question = str(qa.get("question", "")).strip()
            label = normalize_label(question)
            if label is None:
                continue
            answers = qa.get("answers", [])
            has_positive = any(str(a.get("text", "")).strip() for a in answers)
            if has_positive:
                labels.add(label)
        if labels or include_unlabeled:
            out.append(LabeledText(doc_id=doc_id, text=context, labels=tuple(sorted(labels))))
    return out


def load_rows_jsonl(path: str | Path) -> list[LabeledText]:
    rows: list[LabeledText] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            doc_id = str(obj.get("doc_id", f"row-{line_num}"))
            raw_labels = obj.get("labels", [])
            labels = []
            for raw_label in raw_labels:
                label = normalize_label(str(raw_label))
                if label:
                    labels.append(label)
            if text:
                rows.append(LabeledText(doc_id=doc_id, text=text, labels=tuple(sorted(set(labels)))))
    return rows


def save_rows_jsonl(rows: list[LabeledText], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps({"doc_id": row.doc_id, "text": row.text, "labels": list(row.labels)}, ensure_ascii=False)
                + "\n"
            )


def split_by_document(
    rows: list[LabeledText],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[LabeledText], list[LabeledText], list[LabeledText]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Invalid split ratios.")

    doc_ids = sorted({r.doc_id.split("::")[0] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    n_docs = len(doc_ids)
    n_test = int(n_docs * test_ratio)
    n_val = int(n_docs * val_ratio)

    test_ids = set(doc_ids[:n_test])
    val_ids = set(doc_ids[n_test : n_test + n_val])

    train, val, test = [], [], []
    for row in rows:
        root_doc_id = row.doc_id.split("::")[0]
        if root_doc_id in test_ids:
            test.append(row)
        elif root_doc_id in val_ids:
            val.append(row)
        else:
            train.append(row)
    return train, val, test


def expand_rows_with_chunking(
    rows: list[LabeledText],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
) -> list[LabeledText]:
    out: list[LabeledText] = []
    for row in rows:
        chunks = chunk_text(row.text, tokenizer=tokenizer, max_length=max_length, stride=stride)
        if not chunks:
            continue
        for i, chunk in enumerate(chunks):
            out.append(
                LabeledText(
                    doc_id=f"{row.doc_id}::chunk{i}",
                    text=chunk,
                    labels=row.labels,
                )
            )
    return out


def tokenizer_for_model(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

