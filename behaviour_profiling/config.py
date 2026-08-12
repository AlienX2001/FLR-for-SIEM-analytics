from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class BehaviourConfigError(ValueError):
    pass


@dataclass(frozen=True)
class FieldConfig:
    entity: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    numeric: tuple[str, ...] = ()
    text: tuple[str, ...] = ()
    ignored: tuple[str, ...] = (
        "label",
        "sub_label",
        "sub_label_cat",
    )
    auto_infer: bool = True
    auto_ignore_timestamps: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    minimum_rows: int = 10
    minimum_entity_rows: int = 20
    maximum_entities: int = 1_000
    numeric_inference_ratio: float = 0.98
    categorical_minimum_count: int = 1
    maximum_categorical_values: int = 10_000
    token_minimum_count: int = 2
    maximum_tokens_per_field: int = 10_000


@dataclass(frozen=True)
class DetectionConfig:
    anomaly_score_threshold: float = 1.0
    numeric_standard_deviations: float = 4.0
    enforce_observed_numeric_range: bool = True
    maximum_unseen_token_ratio: float = 0.25
    missing_profiled_field_is_anomaly: bool = True
    unknown_entity_is_anomaly: bool = True


@dataclass(frozen=True)
class ScoreWeights:
    unknown_entity: float = 1.0
    missing_field: float = 1.0
    unknown_categorical_value: float = 1.0
    numeric_outlier: float = 1.0
    unseen_text_tokens: float = 1.0
    malformed_row: float = 1.0
    insufficient_evidence: float = 1.0


@dataclass(frozen=True)
class BehaviourProfileConfig:
    version: int = 1
    profile_name: str = "benign_behavior_profile"
    fields: FieldConfig = field(default_factory=FieldConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TOP_LEVEL_KEYS = {"version", "profile_name", "fields", "training", "detection", "score_weights"}
FIELD_KEYS = {
    "entity",
    "categorical",
    "numeric",
    "text",
    "ignored",
    "auto_infer",
    "auto_ignore_timestamps",
}
TRAINING_KEYS = {
    "minimum_rows",
    "minimum_entity_rows",
    "maximum_entities",
    "numeric_inference_ratio",
    "categorical_minimum_count",
    "maximum_categorical_values",
    "token_minimum_count",
    "maximum_tokens_per_field",
}
DETECTION_KEYS = {
    "anomaly_score_threshold",
    "numeric_standard_deviations",
    "enforce_observed_numeric_range",
    "maximum_unseen_token_ratio",
    "missing_profiled_field_is_anomaly",
    "unknown_entity_is_anomaly",
}
WEIGHT_KEYS = {
    "unknown_entity",
    "missing_field",
    "unknown_categorical_value",
    "numeric_outlier",
    "unseen_text_tokens",
    "malformed_row",
    "insufficient_evidence",
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BehaviourConfigError(f"{path}: expected an object")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        prefix = f"{path}." if path else ""
        raise BehaviourConfigError(f"{prefix}{unknown[0]}: unknown configuration field")


def _string_tuple(value: Any, path: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise BehaviourConfigError(f"{path}: expected a list of column names")
    stripped = tuple(item.strip() for item in value)
    if any(not item for item in stripped):
        raise BehaviourConfigError(f"{path}: column names must not be empty")
    if len(set(stripped)) != len(stripped):
        raise BehaviourConfigError(f"{path}: duplicate column name")
    return stripped


def _bool(payload: Mapping[str, Any], key: str, default: bool, path: str) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise BehaviourConfigError(f"{path}.{key}: expected a boolean")
    return value


def _number(payload: Mapping[str, Any], key: str, default: float, path: str) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BehaviourConfigError(f"{path}.{key}: expected a number")
    return float(value)


def _integer(payload: Mapping[str, Any], key: str, default: int, path: str) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BehaviourConfigError(f"{path}.{key}: expected an integer")
    return value


def config_from_mapping(payload: Mapping[str, Any]) -> BehaviourProfileConfig:
    _reject_unknown(payload, TOP_LEVEL_KEYS, "")
    version = payload.get("version", 1)
    if version != 1:
        raise BehaviourConfigError(f"version: unsupported schema version {version!r}")
    profile_name = payload.get("profile_name", "benign_behavior_profile")
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise BehaviourConfigError("profile_name: expected a nonempty string")

    fields_raw = _mapping(payload.get("fields"), "fields")
    training_raw = _mapping(payload.get("training"), "training")
    detection_raw = _mapping(payload.get("detection"), "detection")
    weights_raw = _mapping(payload.get("score_weights"), "score_weights")
    _reject_unknown(fields_raw, FIELD_KEYS, "fields")
    _reject_unknown(training_raw, TRAINING_KEYS, "training")
    _reject_unknown(detection_raw, DETECTION_KEYS, "detection")
    _reject_unknown(weights_raw, WEIGHT_KEYS, "score_weights")

    field_defaults = FieldConfig()
    fields = FieldConfig(
        entity=_string_tuple(fields_raw.get("entity"), "fields.entity", field_defaults.entity),
        categorical=_string_tuple(
            fields_raw.get("categorical"), "fields.categorical", field_defaults.categorical
        ),
        numeric=_string_tuple(fields_raw.get("numeric"), "fields.numeric", field_defaults.numeric),
        text=_string_tuple(fields_raw.get("text"), "fields.text", field_defaults.text),
        ignored=_string_tuple(fields_raw.get("ignored"), "fields.ignored", field_defaults.ignored),
        auto_infer=_bool(fields_raw, "auto_infer", field_defaults.auto_infer, "fields"),
        auto_ignore_timestamps=_bool(
            fields_raw,
            "auto_ignore_timestamps",
            field_defaults.auto_ignore_timestamps,
            "fields",
        ),
    )
    groups = {
        "entity": set(fields.entity),
        "categorical": set(fields.categorical),
        "numeric": set(fields.numeric),
        "text": set(fields.text),
        "ignored": set(fields.ignored),
    }
    for left, left_values in groups.items():
        for right, right_values in groups.items():
            if left >= right:
                continue
            overlap = sorted(left_values & right_values)
            if overlap:
                raise BehaviourConfigError(
                    f"fields.{left}: {overlap[0]!r} is also configured in fields.{right}"
                )

    training_defaults = TrainingConfig()
    training = TrainingConfig(
        minimum_rows=_integer(training_raw, "minimum_rows", training_defaults.minimum_rows, "training"),
        minimum_entity_rows=_integer(
            training_raw,
            "minimum_entity_rows",
            training_defaults.minimum_entity_rows,
            "training",
        ),
        maximum_entities=_integer(
            training_raw, "maximum_entities", training_defaults.maximum_entities, "training"
        ),
        numeric_inference_ratio=_number(
            training_raw,
            "numeric_inference_ratio",
            training_defaults.numeric_inference_ratio,
            "training",
        ),
        categorical_minimum_count=_integer(
            training_raw,
            "categorical_minimum_count",
            training_defaults.categorical_minimum_count,
            "training",
        ),
        maximum_categorical_values=_integer(
            training_raw,
            "maximum_categorical_values",
            training_defaults.maximum_categorical_values,
            "training",
        ),
        token_minimum_count=_integer(
            training_raw,
            "token_minimum_count",
            training_defaults.token_minimum_count,
            "training",
        ),
        maximum_tokens_per_field=_integer(
            training_raw,
            "maximum_tokens_per_field",
            training_defaults.maximum_tokens_per_field,
            "training",
        ),
    )
    if training.minimum_rows <= 0:
        raise BehaviourConfigError("training.minimum_rows: must be positive")
    if training.minimum_entity_rows <= 0:
        raise BehaviourConfigError("training.minimum_entity_rows: must be positive")
    if training.maximum_entities < 0:
        raise BehaviourConfigError("training.maximum_entities: must be nonnegative")
    if not 0.0 <= training.numeric_inference_ratio <= 1.0:
        raise BehaviourConfigError("training.numeric_inference_ratio: must be between 0 and 1")
    for key in (
        "categorical_minimum_count",
        "maximum_categorical_values",
        "token_minimum_count",
        "maximum_tokens_per_field",
    ):
        if getattr(training, key) <= 0:
            raise BehaviourConfigError(f"training.{key}: must be positive")

    detection_defaults = DetectionConfig()
    detection = DetectionConfig(
        anomaly_score_threshold=_number(
            detection_raw,
            "anomaly_score_threshold",
            detection_defaults.anomaly_score_threshold,
            "detection",
        ),
        numeric_standard_deviations=_number(
            detection_raw,
            "numeric_standard_deviations",
            detection_defaults.numeric_standard_deviations,
            "detection",
        ),
        enforce_observed_numeric_range=_bool(
            detection_raw,
            "enforce_observed_numeric_range",
            detection_defaults.enforce_observed_numeric_range,
            "detection",
        ),
        maximum_unseen_token_ratio=_number(
            detection_raw,
            "maximum_unseen_token_ratio",
            detection_defaults.maximum_unseen_token_ratio,
            "detection",
        ),
        missing_profiled_field_is_anomaly=_bool(
            detection_raw,
            "missing_profiled_field_is_anomaly",
            detection_defaults.missing_profiled_field_is_anomaly,
            "detection",
        ),
        unknown_entity_is_anomaly=_bool(
            detection_raw,
            "unknown_entity_is_anomaly",
            detection_defaults.unknown_entity_is_anomaly,
            "detection",
        ),
    )
    if detection.anomaly_score_threshold <= 0:
        raise BehaviourConfigError("detection.anomaly_score_threshold: must be positive")
    if detection.numeric_standard_deviations <= 0:
        raise BehaviourConfigError("detection.numeric_standard_deviations: must be positive")
    if not 0.0 <= detection.maximum_unseen_token_ratio <= 1.0:
        raise BehaviourConfigError(
            "detection.maximum_unseen_token_ratio: must be between 0 and 1"
        )

    weight_defaults = ScoreWeights()
    weights = ScoreWeights(
        **{
            key: _number(weights_raw, key, getattr(weight_defaults, key), "score_weights")
            for key in WEIGHT_KEYS
        }
    )
    if any(getattr(weights, key) < 0 for key in WEIGHT_KEYS):
        raise BehaviourConfigError("score_weights: weights must be nonnegative")
    if not fields.auto_infer and not (fields.categorical or fields.numeric or fields.text):
        raise BehaviourConfigError(
            "fields: configure at least one profiled field when auto_infer is false"
        )

    return BehaviourProfileConfig(
        version=1,
        profile_name=profile_name.strip(),
        fields=fields,
        training=training,
        detection=detection,
        score_weights=weights,
    )


def load_config(path: str | Path | None) -> BehaviourProfileConfig:
    if path is None:
        return BehaviourProfileConfig()
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    return config_from_mapping(_mapping(payload, str(source)))
