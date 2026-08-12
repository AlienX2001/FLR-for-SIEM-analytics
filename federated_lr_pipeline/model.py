from __future__ import annotations

import numpy as np


def observed_bounds(values: np.ndarray) -> tuple[float | None, float | None]:
    """Return the observed minimum and maximum, or undefined bounds for no values."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None, None
    return float(np.min(array)), float(np.max(array))


def initialize_weights(
    num_features: int, num_classes: int, seed: int, scale: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=scale, size=(num_features, num_classes))
    bias = np.zeros(num_classes, dtype=float)
    return weights.astype(float), bias


def softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    denominator = np.sum(exp_logits, axis=1, keepdims=True)
    return exp_logits / denominator
