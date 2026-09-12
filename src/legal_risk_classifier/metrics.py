from __future__ import annotations

import numpy as np

from .labels import TARGET_LABELS


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape.")

    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    per_class = []
    for i, label in enumerate(TARGET_LABELS):
        per_class.append(
            {
                "label": label,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(y_true[:, i].sum()),
            }
        )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }

