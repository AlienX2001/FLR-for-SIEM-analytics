from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from federated_lr_pipeline.model import observed_bounds, softmax

LOGGER = logging.getLogger(__name__)


class EnsembleFusion(ABC):
    @abstractmethod
    def fit(self, specialist_logits: Any, labels: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_logits(
        self,
        logits_by_label: dict[str, dict[str, np.ndarray]],
        *,
        coverage_by_label: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "EnsembleFusion":
        raise NotImplementedError


class ManualLogitFusion(EnsembleFusion):
    def __init__(
        self,
        labels: list[str],
        subcategories_by_label: dict[str, list[str]],
        weights_by_label: dict[str, dict[str, float]],
        coverage_aware: bool = True,
    ) -> None:
        self.labels = labels
        self.subcategories_by_label = subcategories_by_label
        self.weights_by_label = weights_by_label
        self.coverage_aware = coverage_aware

    def fit(self, specialist_logits: Any, labels: Any) -> None:
        return None

    def predict_logits(
        self,
        logits_by_label: dict[str, dict[str, np.ndarray]],
        *,
        coverage_by_label: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        label_logits: list[np.ndarray] = []
        for label in self.labels:
            label_weights = self.weights_by_label.get(label, {})
            subcategories = self.subcategories_by_label[label]
            fused: np.ndarray | None = None
            for subcategory in subcategories:
                sub_logits = np.asarray(logits_by_label[label][subcategory], dtype=float)
                if self.coverage_aware and coverage_by_label is not None:
                    try:
                        coverage = np.asarray(
                            coverage_by_label[label][subcategory],
                            dtype=bool,
                        )
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing coverage for {label}/{subcategory}"
                        ) from exc
                    if coverage.shape != sub_logits.shape:
                        raise ValueError(
                            f"Coverage shape {coverage.shape} does not match logits "
                            f"shape {sub_logits.shape} for {label}/{subcategory}"
                        )
                    sub_logits = np.where(coverage, sub_logits, 0.0)
                contribution = label_weights.get(subcategory, 1.0) * sub_logits
                fused = contribution if fused is None else fused + contribution
            if fused is None:
                raise ValueError(f"Label {label} has no configured subcategories")
            fused = fused + label_weights.get("bias", 0.0)
            label_logits.append(fused)
        return np.column_stack(label_logits)

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "manual",
            "labels": self.labels,
            "subcategories_by_label": self.subcategories_by_label,
            "weights_by_label": self.weights_by_label,
            "coverage_aware": self.coverage_aware,
        }
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "ManualLogitFusion":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("type") != "manual":
            raise ValueError("ManualLogitFusion can only load manual fusion configs")
        return cls(
            labels=list(payload["labels"]),
            subcategories_by_label={
                str(label): list(subcategories)
                for label, subcategories in payload["subcategories_by_label"].items()
            },
            weights_by_label={
                str(label): {str(k): float(v) for k, v in weights.items()}
                for label, weights in payload["weights_by_label"].items()
            },
            coverage_aware=bool(payload.get("coverage_aware", False)),
        )


class MetaLogitFusion(EnsembleFusion):
    def fit(self, specialist_logits: Any, labels: Any) -> None:
        raise NotImplementedError("MetaLogitFusion is reserved for a later implementation")

    def predict_logits(
        self,
        logits_by_label: dict[str, dict[str, np.ndarray]],
        *,
        coverage_by_label: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        del coverage_by_label
        raise NotImplementedError("MetaLogitFusion is reserved for a later implementation")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("MetaLogitFusion is reserved for a later implementation")

    @classmethod
    def load(cls, path: str | Path) -> "MetaLogitFusion":
        raise NotImplementedError("MetaLogitFusion is reserved for a later implementation")


def fused_probabilities(
    fusion: EnsembleFusion,
    logits_by_label: dict[str, dict[str, np.ndarray]],
    *,
    coverage_by_label: dict[str, dict[str, np.ndarray]] | None = None,
    log_context: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    label_logits = fusion.predict_logits(
        logits_by_label,
        coverage_by_label=coverage_by_label,
    )
    if log_context is not None:
        lower_bound, upper_bound = observed_bounds(label_logits)
        LOGGER.info(
            "%s softmax input bounds: lower=%s upper=%s",
            log_context,
            lower_bound,
            upper_bound,
        )
    return label_logits, softmax(label_logits)
