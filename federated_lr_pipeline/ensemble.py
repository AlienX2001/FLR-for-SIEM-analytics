from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from federated_lr_pipeline.model import softmax


class EnsembleFusion(ABC):
    @abstractmethod
    def fit(self, specialist_logits: Any, labels: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_logits(self, logits_by_label: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
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
    ) -> None:
        self.labels = labels
        self.subcategories_by_label = subcategories_by_label
        self.weights_by_label = weights_by_label

    def fit(self, specialist_logits: Any, labels: Any) -> None:
        return None

    def predict_logits(self, logits_by_label: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
        label_logits: list[np.ndarray] = []
        for label in self.labels:
            label_weights = self.weights_by_label.get(label, {})
            subcategories = self.subcategories_by_label[label]
            fused: np.ndarray | None = None
            for subcategory in subcategories:
                sub_logits = np.asarray(logits_by_label[label][subcategory], dtype=float)
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
        )


class MetaLogitFusion(EnsembleFusion):
    def fit(self, specialist_logits: Any, labels: Any) -> None:
        raise NotImplementedError("MetaLogitFusion is reserved for a later implementation")

    def predict_logits(self, logits_by_label: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
        raise NotImplementedError("MetaLogitFusion is reserved for a later implementation")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("MetaLogitFusion is reserved for a later implementation")

    @classmethod
    def load(cls, path: str | Path) -> "MetaLogitFusion":
        raise NotImplementedError("MetaLogitFusion is reserved for a later implementation")


def fused_probabilities(
    fusion: EnsembleFusion,
    logits_by_label: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    label_logits = fusion.predict_logits(logits_by_label)
    return label_logits, softmax(label_logits)
