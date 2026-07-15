from __future__ import annotations

import numpy as np
from scipy import sparse

from federated_lr_pipeline.local_training import (
    binary_logits,
    sigmoid,
    train_binary_logistic_regression,
)


def _binary_loss(
    X: sparse.csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    regularization: float,
) -> float:
    probabilities = sigmoid(binary_logits(X, weights, bias))
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    loss = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    return float(np.mean(loss)) + 0.5 * regularization * float(np.sum(weights * weights))


def test_binary_specialist_loss_decreases_on_synthetic_dataset() -> None:
    X = sparse.csr_matrix(
        [
            [2.0, 0.0],
            [1.5, 0.0],
            [0.0, 2.0],
            [0.0, 1.5],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1])
    weights = np.zeros(2, dtype=float)
    bias = 0.0
    before = _binary_loss(X, y, weights, bias, regularization=1e-4)

    result = train_binary_logistic_regression(
        X,
        y,
        weights,
        bias,
        learning_rate=0.2,
        batch_size=2,
        epochs=40,
        regularization=1e-4,
        seed=5,
    )

    assert result.loss < before
    assert result.accuracy == 1.0


def test_binary_specialist_supports_one_vs_rest_targets() -> None:
    X = sparse.csr_matrix(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 2.5],
        ],
        dtype=np.float32,
    )
    multiclass_labels = np.array([0, 1, 2, 0, 1, 2])
    target_class = 2
    y = (multiclass_labels == target_class).astype(int)

    result = train_binary_logistic_regression(
        X,
        y,
        np.zeros(3, dtype=float),
        0.0,
        learning_rate=0.15,
        batch_size=3,
        epochs=50,
        regularization=1e-4,
        seed=10,
        positive_class_weight=1.5,
        negative_class_weight=0.75,
    )

    assert result.weights.shape == (3,)
    assert isinstance(result.bias, float)
    assert result.accuracy == 1.0
