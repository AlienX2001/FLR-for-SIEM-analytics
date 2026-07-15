from __future__ import annotations

import numpy as np


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

