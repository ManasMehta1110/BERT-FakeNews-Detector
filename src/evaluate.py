from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _extract_predictions(eval_pred) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return preds, labels


def compute_metrics(eval_pred) -> Dict[str, float]:
    preds, labels = _extract_predictions(eval_pred)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }


def evaluate_predictions(
    labels: np.ndarray,
    preds: np.ndarray,
) -> Dict[str, object]:
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }
    matrix = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return {"metrics": metrics, "confusion_matrix": matrix}
