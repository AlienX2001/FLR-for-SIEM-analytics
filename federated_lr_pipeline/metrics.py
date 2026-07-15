from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float(accuracy_score(labels, predictions))


def classification_report_dict(
    labels: np.ndarray, predictions: np.ndarray, class_names: list[str]
) -> dict[str, object]:
    if len(labels) == 0:
        return {}
    return classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
