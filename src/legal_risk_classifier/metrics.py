"""Metrics for a heavily imbalanced multi-label problem.

Accuracy is excluded deliberately: with 85% of chunks unlabelled and per-label
positive rates near 2%, a model predicting all-negative scores above 95% on
every label. Per-class precision, recall and F1 are the honest view, and
average precision is reported alongside because it is threshold-free.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from .labels import LABEL_NAMES


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | float = 0.5,
) -> dict:
    """Per-class and averaged scores. `thresholds` may be per-label."""
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")

    threshold_vector = np.broadcast_to(np.asarray(thresholds, dtype=float), (y_true.shape[1],))
    y_pred = (y_prob >= threshold_vector).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(LABEL_NAMES))
    )

    per_class = []
    for i, name in enumerate(LABEL_NAMES):
        column_true, column_prob = y_true[:, i], y_prob[:, i]
        per_class.append(
            {
                "label": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "average_precision": (
                    float(average_precision_score(column_true, column_prob))
                    if column_true.sum() > 0
                    else None
                ),
                "support": int(support[i]),
                "predicted": int(y_pred[:, i].sum()),
                "threshold": float(threshold_vector[i]),
            }
        )

    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    scored = [c["average_precision"] for c in per_class if c["average_precision"] is not None]
    return {
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "micro_precision": float(micro[0]),
        "micro_recall": float(micro[1]),
        "micro_f1": float(micro[2]),
        "macro_average_precision": float(np.mean(scored)) if scored else None,
        "per_class": per_class,
    }


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Per-label threshold maximising F1 on the given split.

    A single 0.5 cut is wrong here: each label has its own positive rate, so
    each has its own best operating point. Tune on validation, never on test.
    """
    candidates = np.arange(0.05, 0.96, 0.01) if grid is None else grid
    chosen = np.full(y_true.shape[1], 0.5)
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        scores = [
            precision_recall_fscore_support(
                y_true[:, i], (y_prob[:, i] >= t).astype(int), average="binary", zero_division=0
            )[2]
            for t in candidates
        ]
        chosen[i] = float(candidates[int(np.argmax(scores))])
    return chosen


def positive_weights(y_true: np.ndarray) -> np.ndarray:
    """Negative-to-positive ratio per label, for BCEWithLogitsLoss(pos_weight=...).

    Without this the loss is dominated by negatives and the model converges on
    predicting nothing, which the dataset report quantifies at 23x to 66x.
    """
    positives = y_true.sum(axis=0)
    negatives = y_true.shape[0] - positives
    return negatives / np.maximum(positives, 1)
