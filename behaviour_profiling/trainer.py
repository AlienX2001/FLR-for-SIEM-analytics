from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

from behaviour_profiling.config import BehaviourProfileConfig
from behaviour_profiling.csv_stream import collect_union_headers, iter_csv_records, read_header
from behaviour_profiling.preprocessing_adapter import preprocess_behaviour_row

LOGGER = logging.getLogger(__name__)
PROFILE_SCHEMA_VERSION = 1


@dataclass
class RunningStats:
    count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def to_dict(self, *, missing_count: int) -> dict[str, float | int]:
        variance = self.m2 / self.count if self.count else 0.0
        return {
            "count": self.count,
            "missing_count": missing_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "standard_deviation": math.sqrt(max(0.0, variance)),
        }


@dataclass
class FieldAccumulator:
    row_count: int = 0
    missing_count: int = 0
    nonempty_count: int = 0
    numeric_stats: RunningStats = field(default_factory=RunningStats)
    categorical_counts: Counter[str] = field(default_factory=Counter)
    token_counts: Counter[str] = field(default_factory=Counter)
    categorical_overflow: bool = False


@dataclass
class ScopeAccumulator:
    row_count: int = 0
    fields: dict[str, FieldAccumulator] = field(default_factory=dict)


def _bounded_increment(counter: Counter[str], value: str, limit: int) -> None:
    counter[value] += 1
    if len(counter) <= 2 * limit:
        return
    retained = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    counter.clear()
    counter.update(dict(retained))


def _parse_finite_number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _looks_like_timestamp(column: str) -> bool:
    normalized = column.strip().lower()
    return (
        "timestamp" in normalized
        or normalized.startswith("event_time")
        or normalized.endswith("_time")
        or normalized in {"time", "datetime", "date"}
    )


def _candidate_fields(headers: list[str], config: BehaviourProfileConfig) -> list[str]:
    explicitly_profiled = (
        set(config.fields.categorical) | set(config.fields.numeric) | set(config.fields.text)
    )
    ignored = set(config.fields.ignored) | set(config.fields.entity)
    result: list[str] = []
    for column in headers:
        if column in ignored:
            continue
        if column in explicitly_profiled:
            result.append(column)
            continue
        if not config.fields.auto_infer:
            continue
        if config.fields.auto_ignore_timestamps and _looks_like_timestamp(column):
            continue
        result.append(column)
    return result


def _validate_configured_fields(headers: list[str], config: BehaviourProfileConfig) -> None:
    available = set(headers)
    configured = (
        set(config.fields.entity)
        | set(config.fields.categorical)
        | set(config.fields.numeric)
        | set(config.fields.text)
    )
    missing = sorted(configured - available)
    if missing:
        raise ValueError(f"Configured behavior field is absent from all logs: {missing[0]}")


def _update_scope(
    scope: ScopeAccumulator,
    *,
    fields: list[str],
    canonical_fields: dict[str, str],
    field_tokens: dict[str, tuple[str, ...]],
    config: BehaviourProfileConfig,
) -> None:
    scope.row_count += 1
    categorical_limit = config.training.maximum_categorical_values
    token_limit = config.training.maximum_tokens_per_field
    for column in fields:
        accumulator = scope.fields.setdefault(column, FieldAccumulator())
        accumulator.row_count += 1
        value = canonical_fields.get(column)
        if value is None or value == "":
            accumulator.missing_count += 1
            continue
        accumulator.nonempty_count += 1
        number = _parse_finite_number(value)
        if number is not None:
            accumulator.numeric_stats.update(number)
        if value not in accumulator.categorical_counts and len(
            accumulator.categorical_counts
        ) >= categorical_limit:
            accumulator.categorical_overflow = True
        _bounded_increment(accumulator.categorical_counts, value, categorical_limit)
        for token in field_tokens.get(column, ()):
            _bounded_increment(accumulator.token_counts, token, token_limit)


def _entity_key(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _resolve_field_types(
    global_scope: ScopeAccumulator,
    fields: list[str],
    config: BehaviourProfileConfig,
) -> dict[str, str]:
    explicit = {
        **{field: "categorical" for field in config.fields.categorical},
        **{field: "numeric" for field in config.fields.numeric},
        **{field: "text" for field in config.fields.text},
    }
    resolved: dict[str, str] = {}
    for column in fields:
        if column in explicit:
            resolved[column] = explicit[column]
            continue
        accumulator = global_scope.fields[column]
        if accumulator.nonempty_count == 0:
            continue
        numeric_ratio = accumulator.numeric_stats.count / accumulator.nonempty_count
        if numeric_ratio >= config.training.numeric_inference_ratio:
            resolved[column] = "numeric"
        elif not accumulator.categorical_overflow:
            resolved[column] = "categorical"
        else:
            resolved[column] = "text"
    return resolved


def _finalize_scope(
    scope: ScopeAccumulator,
    field_types: dict[str, str],
    config: BehaviourProfileConfig,
) -> dict[str, Any]:
    categorical: dict[str, Any] = {}
    numeric: dict[str, Any] = {}
    text: dict[str, Any] = {}
    for column, field_type in field_types.items():
        accumulator = scope.fields.get(column, FieldAccumulator(row_count=scope.row_count))
        if field_type == "numeric":
            if accumulator.numeric_stats.count:
                numeric[column] = accumulator.numeric_stats.to_dict(
                    missing_count=accumulator.missing_count
                )
            continue
        if field_type == "categorical":
            retained = [
                (value, count)
                for value, count in accumulator.categorical_counts.items()
                if count >= config.training.categorical_minimum_count
            ]
            retained.sort(key=lambda item: item[0])
            categorical[column] = {
                "count": accumulator.nonempty_count,
                "missing_count": accumulator.missing_count,
                "allowed_values": [value for value, _ in retained],
                "value_counts": {value: count for value, count in retained},
            }
            continue
        retained_tokens = [
            (token, count)
            for token, count in accumulator.token_counts.items()
            if count >= config.training.token_minimum_count
        ]
        retained_tokens.sort(key=lambda item: item[0])
        text[column] = {
            "count": accumulator.nonempty_count,
            "missing_count": accumulator.missing_count,
            "allowed_tokens": [token for token, _ in retained_tokens],
            "token_counts": {token: count for token, count in retained_tokens},
        }
    return {
        "row_count": scope.row_count,
        "categorical_fields": categorical,
        "numeric_fields": numeric,
        "text_fields": text,
    }


def _label_column(path: Path, requested: str | None, encoding: str, delimiter: str) -> str:
    columns = read_header(path, encoding=encoding, delimiter=delimiter)
    if requested is not None:
        if requested not in columns:
            raise ValueError(f"Label column {requested!r} is absent from {path}")
        return requested
    if len(columns) == 1:
        return columns[0]
    if "label" in columns:
        return "label"
    raise ValueError(f"Could not detect label column in {path}; provide --label-column")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def train_profile(
    *,
    log_paths: list[Path],
    output_path: Path,
    config: BehaviourProfileConfig,
    groundtruth_paths: list[Path] | None = None,
    label_column: str | None = None,
    benign_label: str = "Benign",
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> dict[str, Any]:
    if not log_paths:
        raise ValueError("At least one benign log CSV is required")
    if groundtruth_paths is not None and len(groundtruth_paths) != len(log_paths):
        raise ValueError("The number of ground-truth CSVs must match the number of log CSVs")
    headers = collect_union_headers(log_paths, encoding=encoding, delimiter=delimiter)
    if not headers:
        raise ValueError("Training log CSVs do not contain a header")
    _validate_configured_fields(headers, config)
    fields = _candidate_fields(headers, config)
    if not fields:
        raise ValueError("No fields are available for behavior profiling")

    global_scope = ScopeAccumulator()
    entity_scopes: dict[str, tuple[tuple[str, ...], ScopeAccumulator]] = {}
    total_rows = 0
    selected_rows = 0
    non_benign_rows = 0
    malformed_rows = 0
    benign_key = benign_label.strip().casefold()

    for file_index, log_path in enumerate(log_paths):
        LOGGER.info("Learning benign behavior from %s", log_path)
        log_records = iter_csv_records(log_path, encoding=encoding, delimiter=delimiter)
        groundtruth_records = None
        current_label_column = None
        if groundtruth_paths is not None:
            groundtruth_path = groundtruth_paths[file_index]
            current_label_column = _label_column(
                groundtruth_path, label_column, encoding, delimiter
            )
            groundtruth_records = iter_csv_records(
                groundtruth_path, encoding=encoding, delimiter=delimiter
            )

        if groundtruth_records is None:
            paired_records = ((record, None) for record in log_records)
        else:
            sentinel = object()
            paired_records = zip_longest(log_records, groundtruth_records, fillvalue=sentinel)

        for log_record, truth_record in paired_records:
            if groundtruth_records is not None and (
                not hasattr(log_record, "row") or not hasattr(truth_record, "row")
            ):
                raise ValueError(
                    f"Row count mismatch between {log_path} and {groundtruth_paths[file_index]}"
                )
            total_rows += 1
            if log_record.malformed:
                malformed_rows += 1
                continue
            if truth_record is not None:
                assert current_label_column is not None
                observed_label = truth_record.row.get(current_label_column, "")
                if observed_label.strip().casefold() != benign_key:
                    non_benign_rows += 1
                    continue
            selected_rows += 1
            preprocessed = preprocess_behaviour_row(log_record.row)
            canonical = dict(preprocessed.canonical_fields)
            tokens = dict(preprocessed.field_tokens)
            _update_scope(
                global_scope,
                fields=fields,
                canonical_fields=canonical,
                field_tokens=tokens,
                config=config,
            )
            if config.fields.entity:
                entity_values = tuple(canonical.get(column, "") for column in config.fields.entity)
                if all(entity_values):
                    key = _entity_key(entity_values)
                    if key in entity_scopes:
                        entity_scope = entity_scopes[key][1]
                    elif len(entity_scopes) < config.training.maximum_entities:
                        entity_scope = ScopeAccumulator()
                        entity_scopes[key] = (entity_values, entity_scope)
                    else:
                        entity_scope = None
                    if entity_scope is not None:
                        _update_scope(
                            entity_scope,
                            fields=fields,
                            canonical_fields=canonical,
                            field_tokens=tokens,
                            config=config,
                        )

    if selected_rows < config.training.minimum_rows:
        raise ValueError(
            f"Only {selected_rows} benign rows were available; "
            f"training.minimum_rows requires {config.training.minimum_rows}"
        )
    field_types = _resolve_field_types(global_scope, fields, config)
    if not field_types:
        raise ValueError("No nonempty fields could be profiled from benign rows")
    finalized_entities: dict[str, Any] = {}
    for key, (entity_values, scope) in sorted(entity_scopes.items()):
        if scope.row_count < config.training.minimum_entity_rows:
            continue
        finalized_entities[key] = {
            "entity_values": list(entity_values),
            "profile": _finalize_scope(scope, field_types, config),
        }

    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "module": "behaviour_profiling",
        "profile_name": config.profile_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "field_types": field_types,
        "global_profile": _finalize_scope(global_scope, field_types, config),
        "entity_profiles": finalized_entities,
        "training_summary": {
            "total_rows_read": total_rows,
            "benign_rows_used": selected_rows,
            "non_benign_rows_skipped": non_benign_rows,
            "malformed_rows_skipped": malformed_rows,
            "input_files": [str(path) for path in log_paths],
            "groundtruth_files": (
                [str(path) for path in groundtruth_paths]
                if groundtruth_paths is not None
                else []
            ),
            "benign_label": benign_label if groundtruth_paths is not None else None,
            "profiled_fields": sorted(field_types),
            "entity_profiles_created": len(finalized_entities),
            "entity_profile_limit_reached": len(entity_scopes)
            >= config.training.maximum_entities
            if config.fields.entity
            else False,
        },
    }
    _atomic_write_json(output_path, profile)
    LOGGER.info(
        "Wrote behavior profile %s using %s benign rows and %s fields",
        output_path,
        selected_rows,
        len(field_types),
    )
    return profile


def load_profile(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported behavior profile schema version {payload.get('schema_version')!r}"
        )
    if payload.get("module") != "behaviour_profiling":
        raise ValueError("Artifact is not a behaviour_profiling profile")
    for key in ("config", "field_types", "global_profile", "entity_profiles"):
        if key not in payload:
            raise ValueError(f"Behavior profile is missing required field {key!r}")
    return payload

