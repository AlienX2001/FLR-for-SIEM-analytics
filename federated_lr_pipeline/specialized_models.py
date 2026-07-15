from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from federated_lr_pipeline.config import PipelineConfig
from federated_lr_pipeline.data import OrgDataset
from federated_lr_pipeline.ensemble import ManualLogitFusion, fused_probabilities
from federated_lr_pipeline.feature_schemas import (
    CROSS_CATEGORY_TOKENS,
    SUBCATEGORY_NAMES,
    SUBCATEGORY_SCHEMAS,
)
from federated_lr_pipeline.local_training import (
    binary_logits,
    build_feature_matrix,
    build_token_counters,
    train_binary_logistic_regression,
)
from federated_lr_pipeline.metrics import accuracy, classification_report_dict
from federated_lr_pipeline.model import initialize_weights
from federated_lr_pipeline.prf import tag_namespaced_vocabulary
from federated_lr_pipeline.utils import json_default, write_json, write_jsonl
from federated_lr_pipeline.vocab import (
    LocalVocabulary,
    construct_global_vocabulary,
    generate_local_vocabulary_from_token_counters,
)

LOGGER = logging.getLogger(__name__)
SUBTOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
FeatureMatrixCache = dict[tuple[str, str, int, str], tuple[tuple[str, ...], sparse.csr_matrix]]


@dataclass(frozen=True)
class LabelBranchConfig:
    label: str
    subcategories: list[str]
    weights: dict[str, float]


@dataclass(frozen=True)
class HierarchicalModelConfig:
    labels: list[str]
    branches: dict[str, LabelBranchConfig]
    fusion: str = "logit"
    fusion_mode: str = "manual"


@dataclass
class SpecialistState:
    label: str
    subcategory: str
    org_texts: list[list[str]]
    org_token_counters: list[list[Counter[str]]]
    local_vocabularies: list[LocalVocabulary]
    org_vocab_tokens: list[list[str]]
    org_tag_lists: list[list[str]]
    global_tags: list[str]
    org_index_vectors: list[list[int]]
    weights: np.ndarray
    bias: float
    missing_columns_by_org: dict[int, list[str]]


@dataclass(frozen=True)
class SpecialistUpdate:
    org_index: int
    index_vector: list[int]
    weights: np.ndarray
    bias: float
    num_examples: int
    loss: float
    accuracy: float


@dataclass(frozen=True)
class PreparedSpecialistJob:
    org_position: int
    org_index: int
    index_vector: list[int]
    X: Any
    train_targets: np.ndarray
    initial_weights: np.ndarray
    initial_bias: float
    learning_rate: float
    batch_size: int
    epochs: int
    regularization: float
    seed: int
    positive_class_weight: float
    negative_class_weight: float
    log_every: int
    round_index: int


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _field_name(column_name: str) -> str:
    return re.sub(r"\s+", "_", str(column_name).strip().lower())


def _field_value(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value).strip().lower())


def _row_has(row: Mapping[str, Any] | pd.Series, column: str) -> bool:
    if isinstance(row, Mapping):
        return column in row
    return column in row.index


def _row_get(row: Mapping[str, Any] | pd.Series, column: str) -> Any:
    return row[column]


def field_aware_tokens(column_name: str, value: Any) -> list[str]:
    if _is_missing_value(value):
        return []
    field = _field_name(column_name)
    value_text = _field_value(value)
    if not value_text:
        return []

    tokens = [f"{field}={value_text}"]
    for subtoken in SUBTOKEN_RE.findall(value_text):
        subtoken = subtoken.lower()
        if len(subtoken) >= 2 and subtoken != value_text:
            tokens.append(f"{field}:{subtoken}")
    return list(dict.fromkeys(tokens))


def build_specialized_texts_for_org(
    dataset: OrgDataset,
    subcategory: str,
    attributes: list[str],
) -> tuple[list[str], list[str]]:
    dataframe = dataset.logs_df
    present_columns = [attribute for attribute in attributes if attribute in dataframe.columns]
    missing_columns = [attribute for attribute in attributes if attribute not in dataframe.columns]
    if missing_columns:
        LOGGER.warning(
            "Organization %s %s subcategory missing %s configured column(s): %s",
            dataset.org_index,
            subcategory,
            len(missing_columns),
            ", ".join(missing_columns),
        )

    texts: list[str] = []
    for values in dataframe.loc[:, present_columns].itertuples(index=False, name=None):
        row_tokens: list[str] = []
        for column, value in zip(present_columns, values):
            row_tokens.extend(field_aware_tokens(column, value))
        texts.append(" ".join(row_tokens))
    return texts, missing_columns


def _row_text(row: Mapping[str, Any] | pd.Series, columns: list[str]) -> str:
    tokens: list[str] = []
    for column in columns:
        if _row_has(row, column):
            tokens.extend(field_aware_tokens(column, _row_get(row, column)))
    return " ".join(tokens)


def _value_contains(
    row: Mapping[str, Any] | pd.Series,
    columns: list[str],
    needles: list[str],
) -> bool:
    haystack = " ".join(
        str(_row_get(row, column)).lower()
        for column in columns
        if _row_has(row, column) and not _is_missing_value(_row_get(row, column))
    )
    return any(needle in haystack for needle in needles)


def _numeric_value(row: Mapping[str, Any] | pd.Series, columns: list[str]) -> float:
    for column in columns:
        if not _row_has(row, column) or _is_missing_value(_row_get(row, column)):
            continue
        value = pd.to_numeric(_row_get(row, column), errors="coerce")
        if not pd.isna(value):
            return float(value)
    return 0.0


def cross_tokens_for_row(row: Mapping[str, Any] | pd.Series) -> list[str]:
    tokens: list[str] = []
    total_size = _numeric_value(row, ["total_size", "bytes_out", "total_sum"])
    has_sensitive_file = _value_contains(
        row,
        ["file_path", "process_command_line", "process_exe", "llm_tool_input"],
        ["secret", "password", "credential", "token", "key", "sensitive"],
    )
    has_external_target = any(
        _row_has(row, column) and not _is_missing_value(_row_get(row, column))
        for column in ["dst_ip", "remote_address", "http_host", "tls_sni"]
    )
    has_encoded_command = _value_contains(
        row,
        ["process_command_line", "process_name", "process_exe"],
        ["encodedcommand", "base64", "powershell", "cmd.exe"],
    )
    has_llm_file_tool = _value_contains(
        row,
        ["llm_tool_name", "llm_tool_input", "tool_name", "tool_input"],
        ["file_read", "read_file", "secret_read"],
    )
    has_failed_login = _value_contains(
        row,
        ["sub_label", "sub_label_cat", "process_command_line", "label"],
        ["failed_login", "bruteforce", "failed login"],
    )

    if has_sensitive_file and total_size >= 5000:
        tokens.append("cross:sensitive_file_read_AND_large_upload_same_host_15m")
    if has_encoded_command and has_external_target:
        tokens.append("cross:encoded_command_AND_first_seen_domain_same_host_15m")
    if has_llm_file_tool and has_sensitive_file:
        tokens.append("cross:llm_file_read_tool_AND_system_sensitive_file_read_same_user_15m")
    if has_failed_login:
        tokens.append("cross:failed_login_burst_AND_successful_login_same_user_15m")
    if has_sensitive_file and has_external_target:
        tokens.append("cross:secret_read_tool_AND_external_post_same_user_15m")
    return list(dict.fromkeys(tokens))


def build_cross_texts_for_org(dataset: OrgDataset) -> tuple[list[str], list[str]]:
    columns = list(dataset.logs_df.columns)
    texts: list[str] = []
    for values in dataset.logs_df.itertuples(index=False, name=None):
        row = dict(zip(columns, values))
        texts.append(" ".join(cross_tokens_for_row(row)))
    return texts, []


def build_subcategory_texts(
    org_datasets: list[OrgDataset],
    subcategories: Sequence[str] | None = None,
) -> tuple[dict[str, list[list[str]]], dict[str, dict[int, list[str]]]]:
    selected_subcategories = list(subcategories) if subcategories is not None else list(SUBCATEGORY_NAMES)
    texts_by_subcategory: dict[str, list[list[str]]] = {
        subcategory: [] for subcategory in selected_subcategories
    }
    missing_by_subcategory: dict[str, dict[int, list[str]]] = {
        subcategory: {} for subcategory in selected_subcategories
    }
    for subcategory in selected_subcategories:
        if subcategory not in SUBCATEGORY_NAMES:
            raise ValueError(f"Unsupported subcategory: {subcategory}")
        for dataset in org_datasets:
            if subcategory == "cross":
                texts, missing = build_cross_texts_for_org(dataset)
            else:
                texts, missing = build_specialized_texts_for_org(
                    dataset,
                    subcategory,
                    SUBCATEGORY_SCHEMAS[subcategory],
                )
            texts_by_subcategory[subcategory].append(texts)
            missing_by_subcategory[subcategory][dataset.org_index] = missing
    return texts_by_subcategory, missing_by_subcategory


def build_subcategory_token_counters(
    texts_by_subcategory: dict[str, list[list[str]]],
) -> dict[str, list[list[Counter[str]]]]:
    return {
        subcategory: [build_token_counters(texts) for texts in org_texts]
        for subcategory, org_texts in texts_by_subcategory.items()
    }


def active_subcategories_for_hierarchy(hierarchy: HierarchicalModelConfig) -> list[str]:
    active = {
        subcategory
        for branch in hierarchy.branches.values()
        for subcategory in branch.subcategories
    }
    return [subcategory for subcategory in SUBCATEGORY_NAMES if subcategory in active]


def select_items(values: Sequence[Any], indices: np.ndarray) -> list[Any]:
    return [values[int(index)] for index in indices]


def _default_subcategories(label: str) -> list[str]:
    lowered = label.lower()
    if lowered == "benign":
        return ["system", "network", "cross"]
    if "credential" in lowered or "identity" in lowered:
        return ["system", "network", "identity", "cross"]
    if "exfil" in lowered or "data" in lowered:
        return ["system", "network", "llm", "cross"]
    if "llm" in lowered or "prompt" in lowered:
        return ["llm", "system", "network", "cross"]
    return ["system", "network", "cross"]


def _default_weights(label: str, subcategories: list[str]) -> dict[str, float]:
    lowered = label.lower()
    weights = {subcategory: 1.0 for subcategory in subcategories}
    weights["bias"] = 0.0 if lowered == "benign" else -1.0
    if "credential" in lowered:
        weights.update({"system": 0.8, "network": 0.7, "identity": 0.8, "cross": 1.2})
    elif "exfil" in lowered or "data" in lowered:
        weights.update({"system": 0.8, "network": 0.8, "llm": 0.5, "cross": 1.2})
    elif "llm" in lowered:
        weights.update({"llm": 1.0, "system": 0.5, "network": 0.5, "cross": 1.1})
    return {key: float(value) for key, value in weights.items() if key == "bias" or key in subcategories}


def _load_hierarchical_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_hierarchical_config(
    observed_labels: list[str],
    config_path: Path | None,
    *,
    fusion_mode: str,
) -> HierarchicalModelConfig:
    payload = _load_hierarchical_payload(config_path)
    label_payload = payload.get("labels", {})
    weight_payload = payload.get("ensemble", {}).get("weights", {})
    labels = [label for label in observed_labels if label in observed_labels]
    branches: dict[str, LabelBranchConfig] = {}

    for label in labels:
        configured = label_payload.get(label, {})
        subcategories = configured.get("subcategories", _default_subcategories(label))
        subcategories = [str(subcategory) for subcategory in subcategories]
        invalid = [subcategory for subcategory in subcategories if subcategory not in SUBCATEGORY_NAMES]
        if invalid:
            raise ValueError(
                f"Label {label} references unsupported subcategories: {', '.join(invalid)}"
            )
        weights = _default_weights(label, subcategories)
        for key, value in weight_payload.get(label, {}).items():
            if key == "bias" or key in subcategories:
                weights[str(key)] = float(value)
        branches[label] = LabelBranchConfig(
            label=label,
            subcategories=subcategories,
            weights=weights,
        )

    return HierarchicalModelConfig(
        labels=labels,
        branches=branches,
        fusion=payload.get("ensemble", {}).get("fusion", "logit"),
        fusion_mode=fusion_mode,
    )


def initialize_specialist(
    *,
    label: str,
    subcategory: str,
    org_datasets: list[OrgDataset],
    org_texts: list[list[str]],
    missing_columns_by_org: dict[int, list[str]],
    splits: list[Any],
    num_features: int,
    min_df: int,
    max_df: float,
    vocabulary_source: str,
    prf_key: bytes,
    seed: int,
    org_token_counters: list[list[Counter[str]]] | None = None,
) -> SpecialistState:
    if org_token_counters is None:
        org_token_counters = [build_token_counters(texts) for texts in org_texts]
    local_vocabularies: list[LocalVocabulary] = []
    for dataset, counters, split in zip(org_datasets, org_token_counters, splits):
        if subcategory == "cross":
            active_tokens = set()
            for counter in select_items(counters, split.train_indices):
                active_tokens.update(counter.keys())
            df = {token: int(token in active_tokens) for token in CROSS_CATEGORY_TOKENS}
            local_vocabularies.append(
                LocalVocabulary(
                    tokens=list(CROSS_CATEGORY_TOKENS),
                    document_frequency=df,
                    effective_min_df=1,
                    effective_max_df_count=len(split.train_indices),
                    used_fallback=False,
                )
            )
            continue
        vocabulary_counters = (
            select_items(counters, split.train_indices)
            if vocabulary_source == "train"
            else counters
        )
        local_vocabularies.append(
            generate_local_vocabulary_from_token_counters(
                vocabulary_counters,
                num_features=num_features,
                min_df=min_df,
                max_df=max_df,
                org_index=dataset.org_index,
            )
        )

    org_vocab_tokens = [vocabulary.tokens for vocabulary in local_vocabularies]
    org_tag_lists = [
        tag_namespaced_vocabulary(
            tokens,
            prf_key,
            label=label,
            subcategory=subcategory,
        )
        for tokens in org_vocab_tokens
    ]
    global_tags, org_index_vectors = construct_global_vocabulary(org_tag_lists)
    weights, bias_vector = initialize_weights(
        len(global_tags),
        1,
        seed=seed + abs(hash((label, subcategory))) % 100_000,
    )
    return SpecialistState(
        label=label,
        subcategory=subcategory,
        org_texts=org_texts,
        org_token_counters=org_token_counters,
        local_vocabularies=local_vocabularies,
        org_vocab_tokens=org_vocab_tokens,
        org_tag_lists=org_tag_lists,
        global_tags=global_tags,
        org_index_vectors=org_index_vectors,
        weights=weights[:, 0],
        bias=float(bias_vector[0]),
        missing_columns_by_org=missing_columns_by_org,
    )


def initialize_all_specialists(
    *,
    hierarchy: HierarchicalModelConfig,
    org_datasets: list[OrgDataset],
    texts_by_subcategory: dict[str, list[list[str]]],
    token_counters_by_subcategory: dict[str, list[list[Counter[str]]]],
    missing_by_subcategory: dict[str, dict[int, list[str]]],
    splits: list[Any],
    config: PipelineConfig,
    prf_key: bytes,
) -> dict[str, dict[str, SpecialistState]]:
    specialists: dict[str, dict[str, SpecialistState]] = {}
    for label in hierarchy.labels:
        specialists[label] = {}
        for subcategory in hierarchy.branches[label].subcategories:
            specialists[label][subcategory] = initialize_specialist(
                label=label,
                subcategory=subcategory,
                org_datasets=org_datasets,
                org_texts=texts_by_subcategory[subcategory],
                org_token_counters=token_counters_by_subcategory[subcategory],
                missing_columns_by_org=missing_by_subcategory[subcategory],
                splits=splits,
                num_features=config.num_features,
                min_df=config.min_df,
                max_df=config.max_df,
                vocabulary_source=config.vocabulary_source,
                prf_key=prf_key,
                seed=config.seed,
            )
    return specialists


def _binary_class_weights(targets: np.ndarray) -> tuple[float, float]:
    positives = float(np.sum(targets == 1))
    negatives = float(np.sum(targets == 0))
    total = positives + negatives
    if positives == 0 or negatives == 0:
        return 1.0, 1.0
    return total / (2.0 * positives), total / (2.0 * negatives)


def _aggregate_specialist_updates(
    previous_weights: np.ndarray,
    previous_bias: float,
    updates: list[SpecialistUpdate],
    *,
    weighting: str,
) -> tuple[np.ndarray, float]:
    if not updates:
        return previous_weights.copy(), previous_bias
    update_weights = [
        float(update.num_examples if weighting == "sample_size" else 1.0)
        for update in updates
    ]
    total = float(sum(update_weights))
    if total <= 0:
        raise ValueError("Cannot aggregate specialist updates with zero total weight")

    numerator = previous_weights.astype(float, copy=True) * total
    bias_numerator = 0.0
    for update, alpha in zip(updates, update_weights):
        bias_numerator += alpha * update.bias
        indices = np.asarray(update.index_vector, dtype=int)
        if update.weights.shape != (len(indices),):
            raise ValueError(
                f"Update from org {update.org_index} has incompatible weight shape"
            )
        np.add.at(
            numerator,
            indices,
            alpha * (update.weights - previous_weights[indices]),
        )
    return numerator / total, bias_numerator / total


def _feature_matrix_for_rows(
    *,
    state: SpecialistState,
    org_position: int,
    row_indices: np.ndarray,
    mode: str,
    log_every: int = 0,
    round_index: int | None = None,
    feature_matrix_cache: FeatureMatrixCache | None = None,
    cache_partition: str | None = None,
) -> sparse.csr_matrix:
    texts = select_items(state.org_texts[org_position], row_indices)
    counters = select_items(state.org_token_counters[org_position], row_indices)
    vocab_tokens = state.org_vocab_tokens[org_position]
    if feature_matrix_cache is not None and cache_partition is not None:
        key = (cache_partition, state.subcategory, org_position, mode)
        vocab_key = tuple(vocab_tokens)
        cached = feature_matrix_cache.get(key)
        if cached is not None and cached[0] == vocab_key:
            return cached[1]
        X = build_feature_matrix(
            texts,
            vocab_tokens,
            mode=mode,
            token_counters=counters,
            log_every=log_every,
            org_index=org_position,
            round_index=round_index,
        )
        feature_matrix_cache[key] = (vocab_key, X)
        return X
    return build_feature_matrix(
        texts,
        vocab_tokens,
        mode=mode,
        token_counters=counters,
        log_every=log_every,
        org_index=org_position,
        round_index=round_index,
    )


def _run_prepared_specialist_job(job: PreparedSpecialistJob) -> tuple[SpecialistUpdate, dict[str, Any]]:
    local_result = train_binary_logistic_regression(
        job.X,
        job.train_targets,
        job.initial_weights,
        job.initial_bias,
        learning_rate=job.learning_rate,
        batch_size=job.batch_size,
        epochs=job.epochs,
        regularization=job.regularization,
        seed=job.seed,
        positive_class_weight=job.positive_class_weight,
        negative_class_weight=job.negative_class_weight,
        log_every=job.log_every,
        org_index=job.org_index,
        round_index=job.round_index,
    )
    update = SpecialistUpdate(
        org_index=job.org_index,
        index_vector=job.index_vector,
        weights=local_result.weights,
        bias=local_result.bias,
        num_examples=local_result.num_examples,
        loss=local_result.loss,
        accuracy=local_result.accuracy,
    )
    metrics = {
        "org_index": job.org_index,
        "train_loss": local_result.loss,
        "local_train_accuracy": local_result.accuracy,
        "num_train_examples": local_result.num_examples,
        "num_features": len(job.index_vector),
        "positive_class_weight": job.positive_class_weight,
        "negative_class_weight": job.negative_class_weight,
    }
    return update, metrics


def train_specialist_round(
    *,
    state: SpecialistState,
    org_datasets: list[OrgDataset],
    encoded_labels_by_org: list[np.ndarray],
    label_index: int,
    splits: list[Any],
    mode: str,
    round_index: int,
    total_rounds: int,
    config: PipelineConfig,
    feature_matrix_cache: FeatureMatrixCache | None = None,
) -> dict[str, Any]:
    all_train_targets = np.concatenate(
        [
            (labels[split.train_indices] == label_index).astype(int)
            for labels, split in zip(encoded_labels_by_org, splits)
        ]
    )
    pos_weight, neg_weight = (
        _binary_class_weights(all_train_targets)
        if config.class_weight == "balanced"
        else (1.0, 1.0)
    )
    prepared_jobs: list[PreparedSpecialistJob] = []
    test_counts_by_org: dict[int, int] = {}

    for org_position, (dataset, labels, split, vocab_tokens, index_vector) in enumerate(
        zip(
            org_datasets,
            encoded_labels_by_org,
            splits,
            state.org_vocab_tokens,
            state.org_index_vectors,
        )
    ):
        train_targets = (labels[split.train_indices] == label_index).astype(int)
        LOGGER.info(
            "Round %s/%s %s/%s org %s: building %s matrix "
            "(train_rows=%s, test_rows=%s, local_features=%s)",
            round_index + 1,
            total_rounds,
            state.label,
            state.subcategory,
            dataset.org_index,
            mode,
            len(split.train_indices),
            len(split.test_indices),
            len(vocab_tokens),
        )
        X = _feature_matrix_for_rows(
            state=state,
            org_position=org_position,
            row_indices=split.train_indices,
            mode=mode,
            log_every=config.local_progress_interval,
            round_index=round_index,
            feature_matrix_cache=feature_matrix_cache,
            cache_partition="train",
        )
        local_initial_weights = (
            state.weights[index_vector] if index_vector else state.weights[:0]
        )
        test_counts_by_org[dataset.org_index] = len(split.test_indices)
        prepared_jobs.append(
            PreparedSpecialistJob(
                org_position=org_position,
                org_index=dataset.org_index,
                index_vector=index_vector,
                X=X,
                train_targets=train_targets,
                initial_weights=local_initial_weights,
                initial_bias=state.bias,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                epochs=config.local_epochs,
                regularization=config.regularization,
                seed=config.seed + round_index * 1009 + org_position,
                positive_class_weight=pos_weight,
                negative_class_weight=neg_weight,
                log_every=config.local_progress_interval,
                round_index=round_index,
            )
        )
    if config.num_workers > 1 and len(prepared_jobs) > 1:
        worker_count = min(config.num_workers, len(prepared_jobs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_run_prepared_specialist_job, prepared_jobs))
    else:
        results = [_run_prepared_specialist_job(job) for job in prepared_jobs]

    updates = [update for update, _ in results]
    local_metrics = []
    for update, metrics in results:
        metrics["num_test_examples"] = test_counts_by_org[update.org_index]
        local_metrics.append(metrics)
    state.weights, state.bias = _aggregate_specialist_updates(
        state.weights,
        state.bias,
        updates,
        weighting=config.aggregation_weighting,
    )
    return {
        "local_metrics": local_metrics,
        "num_global_features": len(state.global_tags),
        "num_features_per_org": [len(tokens) for tokens in state.org_vocab_tokens],
    }


def logits_for_org_rows(
    state: SpecialistState,
    org_position: int,
    row_indices: np.ndarray,
    *,
    feature_matrix_cache: FeatureMatrixCache | None = None,
    cache_partition: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    X = _feature_matrix_for_rows(
        state=state,
        org_position=org_position,
        row_indices=row_indices,
        mode="tf",
        feature_matrix_cache=feature_matrix_cache,
        cache_partition=cache_partition,
    )
    index_vector = state.org_index_vectors[org_position]
    local_weights = state.weights[index_vector] if index_vector else state.weights[:0]
    return X, binary_logits(X, local_weights, state.bias)


def collect_logits_for_rows(
    specialists: dict[str, dict[str, SpecialistState]],
    org_position: int,
    row_indices: np.ndarray,
    *,
    include_features: bool = False,
    feature_matrix_cache: FeatureMatrixCache | None = None,
    cache_partition: str | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    logits: dict[str, dict[str, np.ndarray]] = {}
    features: dict[str, dict[str, np.ndarray]] = {}
    for label, by_subcategory in specialists.items():
        logits[label] = {}
        features[label] = {}
        for subcategory, state in by_subcategory.items():
            X, sub_logits = logits_for_org_rows(
                state,
                org_position,
                row_indices,
                feature_matrix_cache=feature_matrix_cache,
                cache_partition=cache_partition,
            )
            if include_features:
                features[label][subcategory] = X
            logits[label][subcategory] = sub_logits
    return logits, features


def evaluate_hierarchical_ensemble(
    *,
    specialists: dict[str, dict[str, SpecialistState]],
    fusion: ManualLogitFusion,
    encoded_labels_by_org: list[np.ndarray],
    splits: list[Any],
    class_names: list[str],
    feature_matrix_cache: FeatureMatrixCache | None = None,
) -> dict[str, Any]:
    labels_all: list[int] = []
    predictions_all: list[int] = []
    per_org: list[dict[str, Any]] = []
    for org_position, (labels, split) in enumerate(zip(encoded_labels_by_org, splits)):
        logits_by_label, _ = collect_logits_for_rows(
            specialists,
            org_position,
            split.test_indices,
            include_features=False,
            feature_matrix_cache=feature_matrix_cache,
            cache_partition="test",
        )
        _, probabilities = fused_probabilities(fusion, logits_by_label)
        predictions = np.argmax(probabilities, axis=1)
        test_labels = labels[split.test_indices]
        labels_all.extend(test_labels.tolist())
        predictions_all.extend(predictions.tolist())
        per_org.append(
            {
                "org_index": org_position,
                "test_accuracy": accuracy(test_labels, predictions),
                "num_test_examples": len(test_labels),
            }
        )
    labels_array = np.asarray(labels_all, dtype=int)
    predictions_array = np.asarray(predictions_all, dtype=int)
    return {
        "test_accuracy": accuracy(labels_array, predictions_array),
        "classification_report": classification_report_dict(
            labels_array,
            predictions_array,
            class_names,
        ),
        "per_org": per_org,
    }


def top_contributions(
    *,
    state: SpecialistState,
    org_position: int,
    feature_row: Any,
    debug_plaintext_vocab: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    vocab_tokens = state.org_vocab_tokens[org_position]
    index_vector = state.org_index_vectors[org_position]
    tags = state.org_tag_lists[org_position]
    if not vocab_tokens:
        return []
    local_weights = state.weights[index_vector] if index_vector else state.weights[:0]
    candidates: list[dict[str, Any]] = []
    if sparse.issparse(feature_row):
        row = feature_row.tocsr()
        indices = row.indices
        values = row.data
    else:
        dense_row = np.asarray(feature_row).ravel()
        indices = np.flatnonzero(dense_row)
        values = dense_row[indices]
    for local_index, feature_value in zip(indices, values):
        contribution = float(feature_value) * float(local_weights[local_index])
        item: dict[str, Any] = {
            "tag": tags[local_index],
            "gv_index": int(index_vector[local_index]),
            "feature_value": float(feature_value),
            "weight": float(local_weights[local_index]),
            "contribution": float(contribution),
        }
        if debug_plaintext_vocab:
            item["token"] = vocab_tokens[local_index]
        candidates.append(item)
    candidates.sort(key=lambda item: (-abs(item["contribution"]), item["tag"]))
    for rank, item in enumerate(candidates[:limit], start=1):
        item["rank"] = rank
    return candidates[:limit]


def _dict_from_row(values: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {label: float(values[index]) for index, label in enumerate(labels)}


def generate_hierarchical_predictions(
    *,
    org_datasets: list[OrgDataset],
    specialists: dict[str, dict[str, SpecialistState]],
    fusion: ManualLogitFusion,
    label_classes: list[str],
    risk_threshold: float,
    debug_plaintext_vocab: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for org_position, dataset in enumerate(org_datasets):
        row_indices = np.arange(len(dataset.labels), dtype=int)
        logits_by_label, _ = collect_logits_for_rows(
            specialists,
            org_position,
            row_indices,
            include_features=False,
            feature_matrix_cache={},
            cache_partition="predict_all",
        )
        label_logits, probabilities = fused_probabilities(fusion, logits_by_label)
        predictions = np.argmax(probabilities, axis=1)

        for row_index in range(len(dataset.labels)):
            predicted_index = int(predictions[row_index])
            predicted_label = label_classes[predicted_index]
            max_probability = float(np.max(probabilities[row_index]))
            high_risk = bool(max_probability >= risk_threshold)
            subcategory_logits = {
                label: {
                    subcategory: float(logits[row_index])
                    for subcategory, logits in subcategory_logits.items()
                }
                for label, subcategory_logits in logits_by_label.items()
            }
            contributions_for_prediction: dict[str, dict[str, list[dict[str, Any]]]] = {}
            if high_risk:
                contributions_for_prediction[predicted_label] = {}
                for subcategory, state in specialists[predicted_label].items():
                    X_row, _ = logits_for_org_rows(
                        state,
                        org_position,
                        np.asarray([row_index], dtype=int),
                    )
                    contributions_for_prediction[predicted_label][subcategory] = top_contributions(
                        state=state,
                        org_position=org_position,
                        feature_row=X_row[0],
                        debug_plaintext_vocab=debug_plaintext_vocab,
                    )
            record: dict[str, Any] = {
                "org_index": int(dataset.org_index),
                "row_index": int(row_index),
                "internal_log_id": dataset.internal_log_ids[row_index],
                "predicted_label": predicted_label,
                "probabilities": _dict_from_row(probabilities[row_index], label_classes),
                "label_logits": _dict_from_row(label_logits[row_index], label_classes),
                "subcategory_logits": subcategory_logits,
                "top_contributions": contributions_for_prediction,
                "high_risk": high_risk,
                "max_risk_probability": max_probability,
            }
            source_log_id = dataset.source_log_id_at(row_index)
            if source_log_id is not None:
                record["source_log_id"] = source_log_id
            records.append(record)
    return records


def save_hierarchical_artifacts(
    *,
    output_dir: Path,
    specialists: dict[str, dict[str, SpecialistState]],
    debug_plaintext_vocab: bool,
) -> None:
    manifest: dict[str, Any] = {}
    for label, by_subcategory in specialists.items():
        manifest[label] = {}
        for subcategory, state in by_subcategory.items():
            safe_label = label.replace("/", "_")
            prefix = f"{safe_label}_{subcategory}"
            write_json(
                output_dir / f"{prefix}_global_vocabulary_tags.json",
                state.global_tags,
            )
            np.save(output_dir / f"final_{prefix}_weights.npy", state.weights)
            np.save(output_dir / f"final_{prefix}_bias.npy", np.asarray([state.bias]))
            manifest[label][subcategory] = {
                "global_vocabulary_tags": f"{prefix}_global_vocabulary_tags.json",
                "weights": f"final_{prefix}_weights.npy",
                "bias": f"final_{prefix}_bias.npy",
                "num_global_features": len(state.global_tags),
                "missing_columns_by_org": state.missing_columns_by_org,
                "per_org_artifacts": [],
            }
            for org_index, (tags, index_vector, vocabulary) in enumerate(
                zip(state.org_tag_lists, state.org_index_vectors, state.local_vocabularies)
            ):
                org_prefix = output_dir / f"org_{org_index}_{prefix}"
                lv_tags_name = f"org_{org_index}_{prefix}_lv_tags.json"
                index_vector_name = f"org_{org_index}_{prefix}_gv_index_vector.json"
                write_json(output_dir / lv_tags_name, tags)
                write_json(output_dir / index_vector_name, index_vector)
                per_org_artifact: dict[str, Any] = {
                    "org_index": org_index,
                    "lv_tags": lv_tags_name,
                    "gv_index_vector": index_vector_name,
                }
                if debug_plaintext_vocab:
                    lv_tokens_name = f"org_{org_index}_{prefix}_lv_tokens.json"
                    debug_lv_tokens_name = f"org_{org_index}_{prefix}_debug_lv_tokens.json"
                    write_json(output_dir / lv_tokens_name, vocabulary.tokens)
                    write_json(output_dir / debug_lv_tokens_name, vocabulary.tokens)
                    per_org_artifact["lv_tokens"] = lv_tokens_name
                    per_org_artifact["debug_lv_tokens"] = debug_lv_tokens_name
                manifest[label][subcategory]["per_org_artifacts"].append(per_org_artifact)
    write_json(output_dir / "hierarchical_model_manifest.json", manifest)


def write_hierarchical_inference_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    write_jsonl(output_dir / "predictions.jsonl", records)
    high_risk_records = [record for record in records if record["high_risk"]]
    write_jsonl(output_dir / "high_risk_logs.jsonl", high_risk_records)
    explanations = [
        {
            "org_index": record["org_index"],
            "row_index": record["row_index"],
            "internal_log_id": record["internal_log_id"],
            "predicted_label": record["predicted_label"],
            "max_risk_probability": record["max_risk_probability"],
            "label_logits": record["label_logits"],
            "subcategory_logits": record["subcategory_logits"],
            "top_contributions": record["top_contributions"],
            **(
                {"source_log_id": record["source_log_id"]}
                if "source_log_id" in record
                else {}
            ),
        }
        for record in high_risk_records
    ]
    write_jsonl(output_dir / "explanations.jsonl", explanations)

    fieldnames = sorted({key for record in records for key in record})
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key, value in list(row.items()):
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, sort_keys=True, default=json_default)
            writer.writerow(row)
