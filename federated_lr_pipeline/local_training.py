from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from federated_lr_pipeline.vocab import tokenize

LOGGER = logging.getLogger(__name__)


@dataclass
class BinaryTrainingResult:
    weights: np.ndarray
    bias: float
    num_examples: int
    loss: float
    accuracy: float


def build_feature_matrix(
    texts: list[str],
    vocab_tokens: list[str],
    mode: str = "tf",
    *,
    token_counters: list[Mapping[str, int]] | None = None,
    log_every: int = 0,
    org_index: int | None = None,
    round_index: int | None = None,
) -> sparse.csr_matrix:
    if mode not in {"tf", "tfidf"}:
        raise ValueError("mode must be 'tf' or 'tfidf'")

    if token_counters is not None and len(token_counters) != len(texts):
        raise ValueError("token_counters length must match texts length")

    counters = token_counters if token_counters is not None else build_token_counters(
        texts,
        log_every=log_every,
        org_index=org_index,
        round_index=round_index,
        mode=mode,
    )
    return build_feature_matrix_from_counters(counters, vocab_tokens, mode=mode)


def build_token_counters(
    texts: list[str],
    *,
    log_every: int = 0,
    org_index: int | None = None,
    round_index: int | None = None,
    mode: str = "tf",
) -> list[Counter[str]]:
    counters: list[Counter[str]] = []
    total_rows = len(texts)
    for row_index, text in enumerate(texts):
        counters.append(Counter(tokenize(text)))
        completed_rows = row_index + 1
        if log_every > 0 and (
            completed_rows % log_every == 0 or completed_rows == total_rows
        ):
            LOGGER.info(
                "Round %s org %s: tokenized %s/%s local rows for %s features",
                "?" if round_index is None else round_index + 1,
                "?" if org_index is None else org_index,
                completed_rows,
                total_rows,
                mode,
            )
    return counters


def build_feature_matrix_from_counters(
    token_counters: list[Mapping[str, int]],
    vocab_tokens: list[str],
    mode: str = "tf",
) -> sparse.csr_matrix:
    if mode not in {"tf", "tfidf"}:
        raise ValueError("mode must be 'tf' or 'tfidf'")

    n_rows = len(token_counters)
    n_cols = len(vocab_tokens)
    if n_rows == 0 or n_cols == 0:
        return sparse.csr_matrix((n_rows, n_cols), dtype=np.float32)

    token_to_index = {token: index for index, token in enumerate(vocab_tokens)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    for row_index, counts in enumerate(token_counters):
        for token, count in counts.items():
            column_index = token_to_index.get(token)
            if column_index is None or count == 0:
                continue
            row_indices.append(row_index)
            column_indices.append(column_index)
            values.append(float(count))

    if mode == "tfidf" and len(vocab_tokens):
        document_frequency = np.bincount(
            np.asarray(column_indices, dtype=int),
            minlength=n_cols,
        ).astype(np.float32)
    X = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(column_indices, dtype=np.int64),
            ),
        ),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    )
    if mode == "tfidf" and n_cols:
        n_docs = max(1, n_rows)
        idf = np.log((1.0 + n_docs) / (1.0 + document_frequency)) + 1.0
        X = X.multiply(idf.astype(np.float32)).tocsr()

    return X


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values, dtype=float)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def binary_logits(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = X @ weights
    if hasattr(logits, "toarray"):
        logits = logits.toarray()
    return np.asarray(logits, dtype=float).ravel() + float(bias)


def train_binary_logistic_regression(
    X: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    *,
    learning_rate: float = 0.05,
    batch_size: int = 64,
    epochs: int = 1,
    regularization: float = 1e-4,
    seed: int = 42,
    positive_class_weight: float = 1.0,
    negative_class_weight: float = 1.0,
    log_every: int = 0,
    org_index: int | None = None,
    round_index: int | None = None,
) -> BinaryTrainingResult:
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(labels) != X.shape[0]:
        raise ValueError("labels length must match X rows")
    if initial_weights.shape[0] != X.shape[1]:
        raise ValueError("initial_weights length must match X columns")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs < 0:
        raise ValueError("epochs must be non-negative")

    weights = initial_weights.astype(float, copy=True)
    bias = float(initial_bias)
    y = labels.astype(float)
    rng = np.random.default_rng(seed)
    n_examples = X.shape[0]
    total_updates = epochs * int(np.ceil(n_examples / batch_size)) if n_examples else 0
    update_counter = 0

    for epoch_index in range(epochs):
        order = rng.permutation(n_examples)
        for start in range(0, n_examples, batch_size):
            batch_indices = order[start : start + batch_size]
            if len(batch_indices) == 0:
                continue
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            logits = binary_logits(X_batch, weights, bias)
            probabilities = sigmoid(logits)
            sample_weights = np.where(
                y_batch == 1.0, positive_class_weight, negative_class_weight
            )
            residual = (probabilities - y_batch) * sample_weights
            batch_n = float(len(batch_indices))
            grad_w = np.asarray(X_batch.T @ residual, dtype=float).ravel() / batch_n
            grad_w += regularization * weights
            grad_b = float(np.sum(residual) / batch_n)
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
            update_counter += 1
            if log_every > 0 and (
                update_counter % log_every == 0 or update_counter == total_updates
            ):
                LOGGER.info(
                    "Round %s org %s: binary local update %s/%s "
                    "(epoch %s/%s, batch rows %s-%s)",
                    "?" if round_index is None else round_index + 1,
                    "?" if org_index is None else org_index,
                    update_counter,
                    total_updates,
                    epoch_index + 1,
                    epochs,
                    int(start),
                    int(min(start + batch_size, n_examples)),
                )

    logits = binary_logits(X, weights, bias)
    probabilities = sigmoid(logits)
    sample_weights = np.where(y == 1.0, positive_class_weight, negative_class_weight)
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    per_sample_loss = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    loss = float(np.mean(sample_weights * per_sample_loss))
    loss += 0.5 * regularization * float(np.sum(weights * weights))
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = float(np.mean(predictions == labels.astype(int))) if n_examples else 0.0
    return BinaryTrainingResult(
        weights=weights,
        bias=bias,
        num_examples=n_examples,
        loss=loss,
        accuracy=accuracy,
    )
