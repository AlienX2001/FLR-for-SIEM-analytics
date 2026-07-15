from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from federated_lr_pipeline.config import PipelineConfig
from federated_lr_pipeline.data import OrgDataset, load_all_orgs
from federated_lr_pipeline.ensemble import ManualLogitFusion, fused_probabilities
from federated_lr_pipeline.specialized_models import (
    SpecialistState,
    build_subcategory_token_counters,
    build_subcategory_texts,
    collect_logits_for_rows,
    top_contributions,
    logits_for_org_rows,
)
from federated_lr_pipeline.utils import ensure_dir, json_default, setup_logging, write_json, write_jsonl
from federated_lr_pipeline.vocab import LocalVocabulary

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestingArtifacts:
    artifact_dir: Path | None
    run_config: dict[str, Any]
    label_classes: list[str]
    specialists: dict[str, dict[str, SpecialistState]]
    fusion: ManualLogitFusion


@dataclass(frozen=True)
class TestingInferenceResult:
    records: list[dict[str, Any]]
    y_true_indices: np.ndarray
    y_pred_indices: np.ndarray
    subcategory_pred_indices: dict[str, np.ndarray]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _prompt_for_missing_artifact(description: str, default_path: Path) -> Path:
    message = f"Missing required testing artifact:\n{default_path}"
    if not sys.stdin.isatty():
        raise FileNotFoundError(
            f"{message}\nProvide --model-artifact-dir with the required file or pass "
            f"an explicit path for {description}."
        )
    print(message)
    raw_value = input(f"Enter path for {description}: ").strip()
    if not raw_value:
        raise FileNotFoundError(f"No path provided for required artifact: {description}")
    return Path(raw_value)


def _resolve_artifact(
    *,
    artifact_dir: Path | None,
    filename: str,
    description: str,
    explicit_path: Path | None = None,
    required: bool = True,
) -> Path | None:
    candidate = explicit_path
    if candidate is None and artifact_dir is not None:
        candidate = artifact_dir / filename
    if candidate is None:
        if not required:
            return None
        replacement = _prompt_for_missing_artifact(description, Path(filename))
        if not replacement.exists():
            raise FileNotFoundError(f"Provided artifact does not exist: {replacement}")
        return replacement
    if candidate.exists():
        return candidate
    if not required:
        return None
    replacement = _prompt_for_missing_artifact(description, candidate)
    if not replacement.exists():
        raise FileNotFoundError(f"Provided artifact does not exist: {replacement}")
    return replacement


def _optional_artifact(artifact_dir: Path | None, filename: str) -> Path | None:
    if artifact_dir is None:
        return None
    candidate = artifact_dir / filename
    return candidate if candidate.exists() else None


def _load_label_classes(path: Path) -> list[str]:
    payload = _read_json(path)
    label_classes = [str(label) for label in payload]
    if not label_classes:
        raise ValueError("Loaded label encoder classes are empty")
    if len(label_classes) != len(set(label_classes)):
        raise ValueError("Loaded label encoder classes contain duplicates")
    return label_classes


def _validate_groundtruth_labels(
    org_datasets: list[OrgDataset],
    label_classes: list[str],
) -> None:
    known = set(label_classes)
    unknown: dict[int, list[str]] = {}
    for dataset in org_datasets:
        missing = sorted({str(label) for label in dataset.labels if str(label) not in known})
        if missing:
            unknown[dataset.org_index] = missing
    if unknown:
        raise ValueError(
            "Testing groundtruth contains labels that were not present during training: "
            + json.dumps(unknown, sort_keys=True)
        )


def _fusion_from_hierarchy(path: Path, label_classes: list[str]) -> ManualLogitFusion:
    payload = _read_json(path)
    label_payload = payload.get("labels", {})
    subcategories_by_label = {
        label: list(label_payload[label]["subcategories"]) for label in label_classes
    }
    weights_by_label = {
        label: {
            str(key): float(value)
            for key, value in label_payload[label].get("weights", {}).items()
        }
        for label in label_classes
    }
    return ManualLogitFusion(
        labels=label_classes,
        subcategories_by_label=subcategories_by_label,
        weights_by_label=weights_by_label,
    )


def _apply_testing_fusion_override(
    fusion: ManualLogitFusion,
    config: PipelineConfig,
) -> ManualLogitFusion:
    if config.ensemble_method is None:
        return fusion

    weights_by_label: dict[str, dict[str, float]] = {}
    for label in fusion.labels:
        subcategories = fusion.subcategories_by_label[label]
        if config.ensemble_method == "average_logits":
            denominator = float(len(subcategories))
            weights_by_label[label] = {
                subcategory: 1.0 / denominator for subcategory in subcategories
            }
            weights_by_label[label]["bias"] = 0.0
            continue

        raw_weights: dict[str, float] = {}
        for subcategory in subcategories:
            if subcategory == "network":
                raw_weights[subcategory] = config.network_logit_weight
            elif subcategory == "system":
                raw_weights[subcategory] = config.system_logit_weight
            elif subcategory in {"cross", "inter_category"}:
                raw_weights[subcategory] = config.inter_logit_weight
            else:
                raw_weights[subcategory] = 1.0
        denominator = float(sum(raw_weights.values()))
        if denominator == 0.0:
            raise ValueError("Weighted logit fusion requires nonzero total logit weight")
        weights_by_label[label] = {
            subcategory: weight / denominator
            for subcategory, weight in raw_weights.items()
        }
        weights_by_label[label]["bias"] = 0.0

    LOGGER.info(
        "Testing mode overriding saved fusion with %s",
        config.ensemble_method,
    )
    return ManualLogitFusion(
        labels=fusion.labels,
        subcategories_by_label=fusion.subcategories_by_label,
        weights_by_label=weights_by_label,
    )


def _state_prefix(label: str, subcategory: str) -> str:
    return f"{label.replace('/', '_')}_{subcategory}"


def _manifest_entry_path(
    *,
    artifact_dir: Path | None,
    manifest_entry: dict[str, Any],
    key: str,
    fallback_filename: str,
    description: str,
    required: bool = True,
) -> Path | None:
    filename = manifest_entry.get(key, fallback_filename)
    if filename is None:
        return None if not required else _resolve_artifact(
            artifact_dir=artifact_dir,
            filename=fallback_filename,
            description=description,
            required=True,
        )
    path = Path(str(filename))
    if not path.is_absolute() and artifact_dir is not None:
        path = artifact_dir / path
    if path.exists():
        return path
    if not required:
        return None
    replacement = _prompt_for_missing_artifact(description, path)
    if not replacement.exists():
        raise FileNotFoundError(f"Provided artifact does not exist: {replacement}")
    return replacement


def _per_org_manifest_by_index(
    manifest_entry: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    per_org = manifest_entry.get("per_org_artifacts", [])
    result: dict[int, dict[str, Any]] = {}
    for entry in per_org:
        result[int(entry["org_index"])] = dict(entry)
    return result


def _load_org_artifact_path(
    *,
    artifact_dir: Path | None,
    org_index: int,
    prefix: str,
    per_org_entry: dict[str, Any],
    key: str,
    fallback_suffixes: list[str],
    description: str,
    required: bool = True,
) -> Path | None:
    candidates: list[Path] = []
    if key in per_org_entry:
        candidate = Path(str(per_org_entry[key]))
        candidates.append(candidate if candidate.is_absolute() or artifact_dir is None else artifact_dir / candidate)
    for suffix in fallback_suffixes:
        if artifact_dir is not None:
            candidates.append(artifact_dir / f"org_{org_index}_{prefix}_{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not required:
        return None
    default_path = candidates[0] if candidates else Path(f"org_{org_index}_{prefix}_{fallback_suffixes[0]}")
    replacement = _prompt_for_missing_artifact(description, default_path)
    if not replacement.exists():
        raise FileNotFoundError(f"Provided artifact does not exist: {replacement}")
    return replacement


def _load_specialist_state(
    *,
    artifact_dir: Path | None,
    label: str,
    subcategory: str,
    manifest_entry: dict[str, Any],
    org_datasets: list[OrgDataset],
    org_texts: list[list[str]],
    org_token_counters: list[list[Any]],
) -> SpecialistState:
    prefix = _state_prefix(label, subcategory)
    global_tags_path = _manifest_entry_path(
        artifact_dir=artifact_dir,
        manifest_entry=manifest_entry,
        key="global_vocabulary_tags",
        fallback_filename=f"{prefix}_global_vocabulary_tags.json",
        description=f"{label}/{subcategory} global vocabulary tags",
    )
    weights_path = _manifest_entry_path(
        artifact_dir=artifact_dir,
        manifest_entry=manifest_entry,
        key="weights",
        fallback_filename=f"final_{prefix}_weights.npy",
        description=f"{label}/{subcategory} weights",
    )
    bias_path = _manifest_entry_path(
        artifact_dir=artifact_dir,
        manifest_entry=manifest_entry,
        key="bias",
        fallback_filename=f"final_{prefix}_bias.npy",
        description=f"{label}/{subcategory} bias",
    )
    assert global_tags_path is not None
    assert weights_path is not None
    assert bias_path is not None

    global_tags = [str(tag) for tag in _read_json(global_tags_path)]
    weights = np.asarray(np.load(weights_path), dtype=float)
    if weights.ndim == 2 and weights.shape[1] == 1:
        weights = weights[:, 0]
    if weights.ndim != 1:
        raise ValueError(
            f"{label}/{subcategory} weights must be a 1-D binary specialist vector; "
            f"got shape {weights.shape}"
        )
    bias_values = np.asarray(np.load(bias_path), dtype=float).reshape(-1)
    if bias_values.size != 1:
        raise ValueError(
            f"{label}/{subcategory} bias must contain exactly one value; "
            f"got shape {np.asarray(np.load(bias_path)).shape}"
        )

    per_org_entries = _per_org_manifest_by_index(manifest_entry)
    local_vocabularies: list[LocalVocabulary] = []
    org_vocab_tokens: list[list[str]] = []
    org_tag_lists: list[list[str]] = []
    org_index_vectors: list[list[int]] = []

    for org_position, dataset in enumerate(org_datasets):
        per_org_entry = per_org_entries.get(dataset.org_index, {})
        tokens_path = _load_org_artifact_path(
            artifact_dir=artifact_dir,
            org_index=dataset.org_index,
            prefix=prefix,
            per_org_entry=per_org_entry,
            key="lv_tokens",
            fallback_suffixes=["lv_tokens.json", "debug_lv_tokens.json"],
            description=f"org {dataset.org_index} {label}/{subcategory} local vocabulary tokens",
        )
        tags_path = _load_org_artifact_path(
            artifact_dir=artifact_dir,
            org_index=dataset.org_index,
            prefix=prefix,
            per_org_entry=per_org_entry,
            key="lv_tags",
            fallback_suffixes=["lv_tags.json"],
            description=f"org {dataset.org_index} {label}/{subcategory} local vocabulary tags",
        )
        index_vector_path = _load_org_artifact_path(
            artifact_dir=artifact_dir,
            org_index=dataset.org_index,
            prefix=prefix,
            per_org_entry=per_org_entry,
            key="gv_index_vector",
            fallback_suffixes=["gv_index_vector.json"],
            description=f"org {dataset.org_index} {label}/{subcategory} GV index vector",
        )
        assert tokens_path is not None
        assert tags_path is not None
        assert index_vector_path is not None

        tokens = [str(token) for token in _read_json(tokens_path)]
        tags = [str(tag) for tag in _read_json(tags_path)]
        index_vector = [int(index) for index in _read_json(index_vector_path)]
        local_vocabularies.append(
            LocalVocabulary(
                tokens=tokens,
                document_frequency={},
                effective_min_df=1,
                effective_max_df_count=len(org_texts[org_position]),
                used_fallback=False,
            )
        )
        org_vocab_tokens.append(tokens)
        org_tag_lists.append(tags)
        org_index_vectors.append(index_vector)

    missing_columns_by_org = {
        int(org_index): list(columns)
        for org_index, columns in manifest_entry.get("missing_columns_by_org", {}).items()
    }
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
        weights=weights,
        bias=float(bias_values[0]),
        missing_columns_by_org=missing_columns_by_org,
    )


def validate_testing_artifacts(
    *,
    org_datasets: list[OrgDataset],
    label_classes: list[str],
    specialists: dict[str, dict[str, SpecialistState]],
    fusion: ManualLogitFusion,
) -> None:
    if len(org_datasets) == 0:
        raise ValueError("Testing requires at least one organization")
    if fusion.labels != label_classes:
        raise ValueError(
            "Manual fusion label order does not match label_encoder_classes.json: "
            f"{fusion.labels} != {label_classes}"
        )
    _validate_groundtruth_labels(org_datasets, label_classes)

    for label in label_classes:
        if label not in specialists:
            raise ValueError(f"Missing specialist artifacts for label: {label}")
        expected_subcategories = fusion.subcategories_by_label.get(label)
        if not expected_subcategories:
            raise ValueError(f"Missing fusion subcategories for label: {label}")
        for subcategory in expected_subcategories:
            if subcategory not in specialists[label]:
                raise ValueError(f"Missing specialist artifacts for {label}/{subcategory}")
            state = specialists[label][subcategory]
            if len(state.weights) != len(state.global_tags):
                raise ValueError(
                    f"{label}/{subcategory} shape mismatch: weights rows={len(state.weights)} "
                    f"but GV tags={len(state.global_tags)}"
                )
            if not np.isfinite(state.bias):
                raise ValueError(f"{label}/{subcategory} bias is not finite")
            if len(state.org_vocab_tokens) != len(org_datasets):
                raise ValueError(
                    f"{label}/{subcategory} has {len(state.org_vocab_tokens)} org LV files "
                    f"but testing has {len(org_datasets)} organizations"
                )
            for org_position, dataset in enumerate(org_datasets):
                tokens = state.org_vocab_tokens[org_position]
                tags = state.org_tag_lists[org_position]
                index_vector = state.org_index_vectors[org_position]
                if len(tokens) != len(index_vector):
                    raise ValueError(
                        f"org {dataset.org_index} {label}/{subcategory} LV token count "
                        f"{len(tokens)} does not match index vector count {len(index_vector)}"
                    )
                if len(tags) != len(index_vector):
                    raise ValueError(
                        f"org {dataset.org_index} {label}/{subcategory} LV tag count "
                        f"{len(tags)} does not match index vector count {len(index_vector)}"
                    )
                out_of_bounds = [
                    index for index in index_vector if index < 0 or index >= len(state.global_tags)
                ]
                if out_of_bounds:
                    raise ValueError(
                        f"org {dataset.org_index} {label}/{subcategory} has GV indices "
                        f"outside [0, {len(state.global_tags) - 1}]"
                    )


def load_testing_artifacts(
    config: PipelineConfig,
    org_datasets: list[OrgDataset],
) -> TestingArtifacts:
    artifact_dir = config.model_artifact_dir
    if artifact_dir is not None and not artifact_dir.exists():
        raise FileNotFoundError(f"--model-artifact-dir does not exist: {artifact_dir}")

    run_config_path = _resolve_artifact(
        artifact_dir=artifact_dir,
        filename="run_config.json",
        description="training run config",
        explicit_path=config.run_config,
    )
    label_classes_path = _resolve_artifact(
        artifact_dir=artifact_dir,
        filename="label_encoder_classes.json",
        description="label encoder classes",
        explicit_path=config.label_encoder_classes,
    )
    manifest_path = _resolve_artifact(
        artifact_dir=artifact_dir,
        filename="hierarchical_model_manifest.json",
        description="hierarchical model manifest",
        explicit_path=config.hierarchical_model_manifest,
    )
    assert run_config_path is not None
    assert label_classes_path is not None
    assert manifest_path is not None

    run_config = dict(_read_json(run_config_path))
    label_classes = _load_label_classes(label_classes_path)
    manifest = dict(_read_json(manifest_path))

    fusion_path = _resolve_artifact(
        artifact_dir=artifact_dir,
        filename="manual_logit_fusion.json",
        description="manual logit fusion config",
        explicit_path=config.manual_logit_fusion,
        required=False,
    )
    if fusion_path is not None:
        fusion = ManualLogitFusion.load(fusion_path)
    else:
        hierarchy_path = _optional_artifact(artifact_dir, "hierarchical_config.json")
        if hierarchy_path is None:
            raise FileNotFoundError(
                "Missing required testing artifact:\nmanual_logit_fusion.json\n"
                "hierarchical_config.json was also unavailable, so fusion weights "
                "could not be reconstructed."
            )
        fusion = _fusion_from_hierarchy(hierarchy_path, label_classes)
    fusion = _apply_testing_fusion_override(fusion, config)

    active_subcategories = sorted(
        {
            subcategory
            for label in fusion.labels
            for subcategory in fusion.subcategories_by_label[label]
        }
    )
    texts_by_subcategory, _ = build_subcategory_texts(
        org_datasets,
        subcategories=active_subcategories,
    )
    token_counters_by_subcategory = build_subcategory_token_counters(texts_by_subcategory)
    specialists: dict[str, dict[str, SpecialistState]] = {}
    for label in fusion.labels:
        if label not in manifest:
            raise ValueError(f"hierarchical_model_manifest.json is missing label {label}")
        specialists[label] = {}
        for subcategory in fusion.subcategories_by_label[label]:
            if subcategory not in manifest[label]:
                raise ValueError(
                    f"hierarchical_model_manifest.json is missing {label}/{subcategory}"
                )
            specialists[label][subcategory] = _load_specialist_state(
                artifact_dir=artifact_dir,
                label=label,
                subcategory=subcategory,
                manifest_entry=dict(manifest[label][subcategory]),
                org_datasets=org_datasets,
                org_texts=texts_by_subcategory[subcategory],
                org_token_counters=token_counters_by_subcategory[subcategory],
            )

    validate_testing_artifacts(
        org_datasets=org_datasets,
        label_classes=label_classes,
        specialists=specialists,
        fusion=fusion,
    )
    return TestingArtifacts(
        artifact_dir=artifact_dir,
        run_config=run_config,
        label_classes=label_classes,
        specialists=specialists,
        fusion=fusion,
    )


def _dict_from_row(values: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {label: float(values[index]) for index, label in enumerate(labels)}


def _subcategory_predictions(
    *,
    logits_by_label: dict[str, dict[str, np.ndarray]],
    label_classes: list[str],
    subcategory: str,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    row_count = 0
    for by_subcategory in logits_by_label.values():
        for logits in by_subcategory.values():
            row_count = len(logits)
            break
        if row_count:
            break
    for label in label_classes:
        if subcategory in logits_by_label.get(label, {}):
            columns.append(np.asarray(logits_by_label[label][subcategory], dtype=float))
        else:
            columns.append(np.zeros(row_count, dtype=float))
    return np.argmax(np.column_stack(columns), axis=1)


def _all_subcategories(specialists: dict[str, dict[str, SpecialistState]]) -> list[str]:
    names = {
        subcategory
        for by_subcategory in specialists.values()
        for subcategory in by_subcategory
    }
    return sorted(names)


def run_testing_inference(
    *,
    org_datasets: list[OrgDataset],
    artifacts: TestingArtifacts,
    risk_threshold: float,
    debug_plaintext_vocab: bool,
) -> TestingInferenceResult:
    label_to_index = {label: index for index, label in enumerate(artifacts.label_classes)}
    records: list[dict[str, Any]] = []
    y_true_indices: list[int] = []
    y_pred_indices: list[int] = []
    subcategory_predictions: dict[str, list[int]] = {
        subcategory: [] for subcategory in _all_subcategories(artifacts.specialists)
    }

    for org_position, dataset in enumerate(org_datasets):
        row_indices = np.arange(len(dataset.labels), dtype=int)
        feature_matrix_cache = {}
        logits_by_label, _ = collect_logits_for_rows(
            artifacts.specialists,
            org_position,
            row_indices,
            include_features=False,
            feature_matrix_cache=feature_matrix_cache,
            cache_partition="testing_all",
        )
        label_logits, probabilities = fused_probabilities(artifacts.fusion, logits_by_label)
        predictions = np.argmax(probabilities, axis=1)
        for subcategory in subcategory_predictions:
            subcategory_predictions[subcategory].extend(
                _subcategory_predictions(
                    logits_by_label=logits_by_label,
                    label_classes=artifacts.label_classes,
                    subcategory=subcategory,
                ).tolist()
            )

        for row_index in range(len(dataset.labels)):
            true_label = str(dataset.labels[row_index])
            true_index = label_to_index[true_label]
            predicted_index = int(predictions[row_index])
            predicted_label = artifacts.label_classes[predicted_index]
            subcategory_logits = {
                label: {
                    subcategory: float(logits[row_index])
                    for subcategory, logits in by_subcategory.items()
                }
                for label, by_subcategory in logits_by_label.items()
            }
            max_probability = float(np.max(probabilities[row_index]))
            contributions_for_prediction: dict[str, dict[str, list[dict[str, Any]]]] = {}
            if max_probability >= risk_threshold:
                contributions_for_prediction[predicted_label] = {}
                for subcategory, state in artifacts.specialists[predicted_label].items():
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
                "true_label": true_label,
                "predicted_label": predicted_label,
                "ensemble_predicted_label": predicted_label,
                "correct": bool(predicted_label == true_label),
                "probabilities": _dict_from_row(probabilities[row_index], artifacts.label_classes),
                "ensemble_probabilities": _dict_from_row(
                    probabilities[row_index],
                    artifacts.label_classes,
                ),
                "label_logits": _dict_from_row(label_logits[row_index], artifacts.label_classes),
                "subcategory_logits": subcategory_logits,
                "top_contributions": contributions_for_prediction,
                "top_contributing_features": contributions_for_prediction,
                "high_risk": bool(max_probability >= risk_threshold),
                "max_risk_probability": max_probability,
                "ensemble_max_risk_probability": max_probability,
            }
            source_log_id = dataset.source_log_id_at(row_index)
            if source_log_id is not None:
                record["source_log_id"] = source_log_id
            records.append(record)
            y_true_indices.append(true_index)
            y_pred_indices.append(predicted_index)

    return TestingInferenceResult(
        records=records,
        y_true_indices=np.asarray(y_true_indices, dtype=int),
        y_pred_indices=np.asarray(y_pred_indices, dtype=int),
        subcategory_pred_indices={
            subcategory: np.asarray(predictions, dtype=int)
            for subcategory, predictions in subcategory_predictions.items()
        },
    )


def _metrics_for_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_classes: list[str],
) -> dict[str, Any]:
    labels = list(range(len(label_classes)))
    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1)
    per_class_accuracy = {
        label: (
            float(matrix[index, index] / row_sums[index])
            if row_sums[index] > 0
            else 0.0
        )
        for index, label in enumerate(label_classes)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_classes,
            zero_division=0,
            output_dict=True,
        ),
        "per_class_accuracy": per_class_accuracy,
    }


def compute_testing_metrics(
    result: TestingInferenceResult,
    label_classes: list[str],
) -> dict[str, Any]:
    subcategory_metrics = {
        subcategory: _metrics_for_predictions(
            result.y_true_indices,
            predictions,
            label_classes,
        )
        for subcategory, predictions in result.subcategory_pred_indices.items()
    }
    metrics: dict[str, Any] = {
        "ensemble_model": _metrics_for_predictions(
            result.y_true_indices,
            result.y_pred_indices,
            label_classes,
        ),
        "subcategories": subcategory_metrics,
    }
    if "network" in subcategory_metrics:
        metrics["network_model"] = subcategory_metrics["network"]
    if "system" in subcategory_metrics:
        metrics["system_model"] = subcategory_metrics["system"]
    if "cross" in subcategory_metrics:
        metrics["inter_category_model"] = subcategory_metrics["cross"]
    return metrics


def _classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_classes: list[str],
) -> str:
    labels = list(range(len(label_classes)))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_classes,
        zero_division=0,
    )


def _write_predictions_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key, value in list(row.items()):
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, sort_keys=True, default=json_default)
            writer.writerow(row)


def _write_confusion_matrix_csv(
    path: Path,
    result: TestingInferenceResult,
    label_classes: list[str],
) -> None:
    matrix = confusion_matrix(
        result.y_true_indices,
        result.y_pred_indices,
        labels=list(range(len(label_classes))),
    )
    dataframe = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in label_classes],
        columns=[f"pred_{label}" for label in label_classes],
    )
    dataframe.to_csv(path)


def write_testing_outputs(
    *,
    output_dir: Path,
    config: PipelineConfig,
    artifacts: TestingArtifacts,
    result: TestingInferenceResult,
    metrics: dict[str, Any],
    risk_threshold: float,
) -> None:
    write_json(
        output_dir / "testing_run_config.json",
        {
            **config.to_json_dict(),
            "risk_threshold_used": risk_threshold,
            "loaded_training_run_config": artifacts.run_config,
        },
    )
    write_json(output_dir / "testing_metrics.json", metrics)
    with (output_dir / "testing_classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            _classification_report_text(
                result.y_true_indices,
                result.y_pred_indices,
                artifacts.label_classes,
            )
        )
        handle.write("\n")
    _write_confusion_matrix_csv(
        output_dir / "testing_confusion_matrix.csv",
        result,
        artifacts.label_classes,
    )
    write_jsonl(output_dir / "testing_predictions.jsonl", result.records)
    _write_predictions_csv(output_dir / "testing_predictions.csv", result.records)
    high_risk_records = [record for record in result.records if record["high_risk"]]
    write_jsonl(output_dir / "testing_high_risk_logs.jsonl", high_risk_records)

    explanations = [
        {
            "org_index": record["org_index"],
            "row_index": record["row_index"],
            "internal_log_id": record["internal_log_id"],
            "true_label": record["true_label"],
            "ensemble_predicted_label": record["ensemble_predicted_label"],
            "correct": record["correct"],
            "ensemble_max_risk_probability": record["ensemble_max_risk_probability"],
            "model_logits": {
                "ensemble": record["label_logits"],
                "by_label_subcategory": record["subcategory_logits"],
            },
            "top_contributing_features": record["top_contributing_features"],
            **(
                {"source_log_id": record["source_log_id"]}
                if "source_log_id" in record
                else {}
            ),
        }
        for record in high_risk_records
    ]
    write_jsonl(output_dir / "testing_explanations.jsonl", explanations)


def _effective_risk_threshold(config: PipelineConfig, run_config: dict[str, Any]) -> float:
    if "--risk-threshold" in config.testing_override_parameters:
        return config.risk_threshold
    return float(run_config.get("risk_threshold", config.risk_threshold))


def run_testing_mode(config: PipelineConfig) -> None:
    setup_logging()
    for flag in config.testing_ignored_parameters:
        LOGGER.warning("Ignoring training parameter %s in --testing mode", flag)

    output_dir = ensure_dir(config.output_dir)
    LOGGER.info("Loading row-aligned organization data for testing")
    org_datasets = load_all_orgs(
        config.org_data,
        config.org_groundtruth,
        text_column=config.text_column,
        text_columns=config.text_columns,
        label_column=config.label_column,
    )
    LOGGER.info("Loading pretrained hierarchical artifacts")
    artifacts = load_testing_artifacts(config, org_datasets)
    risk_threshold = _effective_risk_threshold(config, artifacts.run_config)

    LOGGER.info("Running testing-mode hierarchical inference")
    result = run_testing_inference(
        org_datasets=org_datasets,
        artifacts=artifacts,
        risk_threshold=risk_threshold,
        debug_plaintext_vocab=config.debug_plaintext_vocab,
    )
    LOGGER.info("Computing testing metrics")
    metrics = compute_testing_metrics(result, artifacts.label_classes)
    LOGGER.info("Writing testing outputs")
    write_testing_outputs(
        output_dir=output_dir,
        config=config,
        artifacts=artifacts,
        result=result,
        metrics=metrics,
        risk_threshold=risk_threshold,
    )
    LOGGER.info("Wrote testing outputs to %s", output_dir)
