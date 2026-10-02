from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from prevention_relevance.models import RankingRecord


TRUE_VALUES = {"1", "true", "yes", "relevant"}
FALSE_VALUES = {"0", "false", "no", "not_relevant", "irrelevant"}


def _parse_relevance(value: object, *, location: str) -> int:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return 1
    if normalized in FALSE_VALUES:
        return 0
    raise ValueError(
        f"{location} must be one of {sorted(TRUE_VALUES | FALSE_VALUES)}"
    )


def load_relevance_labels(path: Path) -> dict[str, int]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"Could not open relevance-label CSV {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"indicator_id", "relevant"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Relevance-label CSV must contain columns {sorted(required)}"
            )
        labels: dict[str, int] = {}
        for line_number, row in enumerate(reader, start=2):
            indicator_id = str(row.get("indicator_id", "")).strip()
            if not indicator_id:
                raise ValueError(f"{path}:{line_number}: indicator_id cannot be empty")
            if indicator_id in labels:
                raise ValueError(
                    f"{path}:{line_number}: duplicate indicator_id {indicator_id!r}"
                )
            labels[indicator_id] = _parse_relevance(
                row.get("relevant"), location=f"{path}:{line_number}:relevant"
            )
    if not labels:
        raise ValueError("Relevance-label CSV is empty")
    return labels


def _ranking_metric(labels: Sequence[int], *, k: int) -> tuple[float, float]:
    cutoff = min(k, len(labels))
    if cutoff == 0:
        return 0.0, 0.0
    selected = labels[:cutoff]
    relevant_selected = sum(selected)
    total_relevant = sum(labels)
    return (
        relevant_selected / cutoff,
        relevant_selected / total_relevant if total_relevant else 0.0,
    )


def _ndcg(labels: Sequence[int], scores: Sequence[float]) -> float:
    if not labels or not any(labels):
        return 0.0
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    ideal = sorted(labels, reverse=True)

    def dcg(values: Sequence[int]) -> float:
        return sum(value / np.log2(index + 2) for index, value in enumerate(values))

    return float(dcg([labels[index] for index in order]) / dcg(ideal))


def compute_relevance_metrics(
    rankings: Sequence[RankingRecord],
    labels: Mapping[str, int],
    *,
    top_k: int,
) -> dict[str, Any]:
    by_id = {record.indicator_id: record for record in rankings}
    unknown = sorted(set(labels) - set(by_id))
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(
            f"Relevance labels reference {len(unknown)} unknown Indicator(s): {preview}"
        )
    evaluated = [record for record in rankings if record.indicator_id in labels]
    truth = [labels[record.indicator_id] for record in evaluated]
    predicted = [int(record.relevance_decision == "relevant") for record in evaluated]
    scores = [record.relevance_score for record in evaluated]
    precision_at_k, recall_at_k = _ranking_metric(truth, k=top_k)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).tolist()
    metrics: dict[str, Any] = {
        "labeled_indicator_count": len(evaluated),
        "positive_label_count": int(sum(truth)),
        "negative_label_count": int(len(truth) - sum(truth)),
        "decision_positive_count": int(sum(predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "confusion_matrix": {
            "labels": ["not_relevant", "relevant"],
            "values": matrix,
        },
        f"precision_at_{top_k}": float(precision_at_k),
        f"recall_at_{top_k}": float(recall_at_k),
        "ndcg": _ndcg(truth, scores),
    }
    if len(set(truth)) == 2:
        metrics["average_precision"] = float(average_precision_score(truth, scores))
        metrics["roc_auc"] = float(roc_auc_score(truth, scores))
    else:
        metrics["average_precision"] = None
        metrics["roc_auc"] = None
    return metrics
