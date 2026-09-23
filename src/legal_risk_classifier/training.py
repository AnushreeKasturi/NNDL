"""The training loop, shared by every model in the comparison.

One loop for all four models is what makes the comparison a comparison: the
CNN, BERT, Legal-BERT and Longformer see identical chunks, identical splits,
identical loss weighting and identical early stopping, so a score difference
is attributable to the model rather than to its training recipe.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics, tune_thresholds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainingConfig:
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    grad_accumulation: int = 1
    max_grad_norm: float = 1.0
    patience: int = 3
    seed: int = 42
    device: str = "auto"


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_macro_f1: float
    val_macro_average_precision: float | None


@dataclass
class TrainingResult:
    best_epoch: int
    best_val_macro_f1: float
    history: list[EpochRecord] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_prob) over a loader."""
    model.eval()
    true_batches, prob_batches = [], []
    for batch in loader:
        targets = batch.pop("labels")
        logits = model(**{k: v.to(device) for k, v in batch.items()})
        prob_batches.append(torch.sigmoid(logits.float()).cpu().numpy())
        true_batches.append(targets.numpy())
    return np.concatenate(true_batches), np.concatenate(prob_batches)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    pos_weight: np.ndarray | None = None,
    output_dir: Path | None = None,
) -> TrainingResult:
    """Train, early-stop on validation macro F1, restore the best weights."""
    device = resolve_device(config.device)
    set_seed(config.seed)
    model.to(device)

    weight = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    result = TrainingResult(best_epoch=0, best_val_macro_f1=-1.0)
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            targets = batch.pop("labels").to(device)
            logits = model(**{k: v.to(device) for k, v in batch.items()})
            loss = criterion(logits, targets) / config.grad_accumulation
            loss.backward()

            if step % config.grad_accumulation == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += float(loss.item()) * config.grad_accumulation
            steps += 1

        y_true, y_prob = predict(model, val_loader, device)
        val = compute_metrics(y_true, y_prob, thresholds=0.5)
        record = EpochRecord(
            epoch=epoch,
            train_loss=total_loss / max(steps, 1),
            val_macro_f1=val["macro_f1"],
            val_macro_average_precision=val["macro_average_precision"],
        )
        result.history.append(record)
        print(json.dumps(asdict(record)), flush=True)

        if record.val_macro_f1 > result.best_val_macro_f1:
            result.best_val_macro_f1 = record.val_macro_f1
            result.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"early stop: no improvement for {config.patience} epochs", flush=True)
                break

    model.load_state_dict(best_state)

    # Thresholds are tuned on validation, never on test.
    y_true, y_prob = predict(model, val_loader, device)
    result.thresholds = tune_thresholds(y_true, y_prob).tolist()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, output_dir / "model.pt")
        (output_dir / "training.json").write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "best_epoch": result.best_epoch,
                    "best_val_macro_f1": result.best_val_macro_f1,
                    "thresholds": result.thresholds,
                    "history": [asdict(r) for r in result.history],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return result
