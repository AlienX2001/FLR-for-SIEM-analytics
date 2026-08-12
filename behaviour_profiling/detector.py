from __future__ import annotations

import json
import logging
import math
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from behaviour_profiling.config import BehaviourProfileConfig, config_from_mapping
from behaviour_profiling.csv_stream import (
    AtomicAnomalyCsvWriter,
    AtomicJsonlWriter,
    collect_union_headers,
    iter_csv_records,
)
from behaviour_profiling.models import (
    AnomalyEvidence,
    DetectionResult,
    DetectionSummary,
    PreprocessedBehaviourRow,
    SourceRecord,
)
from behaviour_profiling.preprocessing_adapter import preprocess_behaviour_row
from behaviour_profiling.trainer import load_profile

LOGGER = logging.getLogger(__name__)

UNKNOWN_ENTITY = "UNKNOWN_ENTITY"
MISSING_PROFILED_FIELD = "MISSING_PROFILED_FIELD"
UNKNOWN_CATEGORICAL_VALUE = "UNKNOWN_CATEGORICAL_VALUE"
NUMERIC_VALUE_MALFORMED = "NUMERIC_VALUE_MALFORMED"
NUMERIC_OUTSIDE_BENIGN_PROFILE = "NUMERIC_OUTSIDE_BENIGN_PROFILE"
UNSEEN_TEXT_TOKENS = "UNSEEN_TEXT_TOKENS"
MALFORMED_ROW = "MALFORMED_ROW"
INSUFFICIENT_PROFILE_EVIDENCE = "INSUFFICIENT_PROFILE_EVIDENCE"


def _entity_key(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _add_missing_evidence(
    evidence: list[AnomalyEvidence],
    field: str,
    details: Mapping[str, Any],
    config: BehaviourProfileConfig,
) -> None:
    if not config.detection.missing_profiled_field_is_anomaly:
        return
    if int(details.get("missing_count", 0)) > 0:
        return
    evidence.append(
        AnomalyEvidence(
            reason_code=MISSING_PROFILED_FIELD,
            field=field,
            details="profiled field is missing or empty",
            score=config.score_weights.missing_field,
        )
    )


def _evaluate_categorical(
    field: str,
    details: Mapping[str, Any],
    row: PreprocessedBehaviourRow,
    config: BehaviourProfileConfig,
    evidence: list[AnomalyEvidence],
) -> bool:
    value = row.canonical_fields.get(field)
    if value is None or value == "":
        _add_missing_evidence(evidence, field, details, config)
        return False
    if value not in set(details.get("allowed_values", [])):
        evidence.append(
            AnomalyEvidence(
                reason_code=UNKNOWN_CATEGORICAL_VALUE,
                field=field,
                details="value was not observed often enough in benign training data",
                score=config.score_weights.unknown_categorical_value,
            )
        )
    return True


def _evaluate_numeric(
    field: str,
    details: Mapping[str, Any],
    row: PreprocessedBehaviourRow,
    config: BehaviourProfileConfig,
    evidence: list[AnomalyEvidence],
) -> bool:
    value = row.canonical_fields.get(field)
    if value is None or value == "":
        _add_missing_evidence(evidence, field, details, config)
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number):
        evidence.append(
            AnomalyEvidence(
                reason_code=NUMERIC_VALUE_MALFORMED,
                field=field,
                details="value cannot be interpreted as a finite number",
                score=config.score_weights.numeric_outlier,
            )
        )
        return True

    minimum = float(details["minimum"])
    maximum = float(details["maximum"])
    mean = float(details["mean"])
    standard_deviation = float(details["standard_deviation"])
    outside_range = config.detection.enforce_observed_numeric_range and not (
        minimum <= number <= maximum
    )
    z_score = (
        abs(number - mean) / standard_deviation if standard_deviation > 0.0 else 0.0
    )
    outside_z_score = z_score > config.detection.numeric_standard_deviations
    if outside_range or outside_z_score:
        failures: list[str] = []
        if outside_range:
            failures.append("outside observed benign range")
        if outside_z_score:
            failures.append(
                f"z-score {z_score:.4g} exceeds {config.detection.numeric_standard_deviations:g}"
            )
        evidence.append(
            AnomalyEvidence(
                reason_code=NUMERIC_OUTSIDE_BENIGN_PROFILE,
                field=field,
                details="; ".join(failures),
                score=config.score_weights.numeric_outlier,
            )
        )
    return True


def _evaluate_text(
    field: str,
    details: Mapping[str, Any],
    row: PreprocessedBehaviourRow,
    config: BehaviourProfileConfig,
    evidence: list[AnomalyEvidence],
) -> bool:
    tokens = row.field_tokens.get(field, ())
    if not tokens:
        _add_missing_evidence(evidence, field, details, config)
        return False
    allowed = set(details.get("allowed_tokens", []))
    unseen_count = sum(token not in allowed for token in tokens)
    unseen_ratio = unseen_count / len(tokens)
    if unseen_ratio > config.detection.maximum_unseen_token_ratio:
        evidence.append(
            AnomalyEvidence(
                reason_code=UNSEEN_TEXT_TOKENS,
                field=field,
                details=(
                    f"{unseen_count}/{len(tokens)} normalized tokens were unseen; "
                    f"ratio {unseen_ratio:.4g} exceeds "
                    f"{config.detection.maximum_unseen_token_ratio:g}"
                ),
                score=config.score_weights.unseen_text_tokens,
            )
        )
    return True


def score_preprocessed_row(
    row: PreprocessedBehaviourRow,
    profile: Mapping[str, Any],
    config: BehaviourProfileConfig,
) -> DetectionResult:
    evidence: list[AnomalyEvidence] = []
    profile_scope = "global"
    selected_profile = profile["global_profile"]
    global_profile = profile["global_profile"]

    if config.fields.entity:
        entity_values = tuple(row.canonical_fields.get(field, "") for field in config.fields.entity)
        entity_profile = None
        if all(entity_values):
            entity_profile = profile.get("entity_profiles", {}).get(_entity_key(entity_values))
        if entity_profile is not None:
            selected_profile = entity_profile["profile"]
            profile_scope = "entity"
        elif config.detection.unknown_entity_is_anomaly:
            evidence.append(
                AnomalyEvidence(
                    reason_code=UNKNOWN_ENTITY,
                    field=",".join(config.fields.entity),
                    details="entity was not represented by a sufficiently trained benign profile",
                    score=config.score_weights.unknown_entity,
                )
            )

    evaluated_fields = 0
    categorical_fields = dict(global_profile.get("categorical_fields", {}))
    categorical_fields.update(selected_profile.get("categorical_fields", {}))
    numeric_fields = dict(global_profile.get("numeric_fields", {}))
    numeric_fields.update(selected_profile.get("numeric_fields", {}))
    text_fields = dict(global_profile.get("text_fields", {}))
    text_fields.update(selected_profile.get("text_fields", {}))
    for field, details in categorical_fields.items():
        evaluated_fields += int(
            _evaluate_categorical(field, details, row, config, evidence)
        )
    for field, details in numeric_fields.items():
        evaluated_fields += int(_evaluate_numeric(field, details, row, config, evidence))
    for field, details in text_fields.items():
        evaluated_fields += int(_evaluate_text(field, details, row, config, evidence))

    if evaluated_fields == 0 and not evidence:
        evidence.append(
            AnomalyEvidence(
                reason_code=INSUFFICIENT_PROFILE_EVIDENCE,
                field=None,
                details="row contains no usable fields from the trained behavior profile",
                score=config.score_weights.insufficient_evidence,
            )
        )
    score = float(sum(item.score for item in evidence))
    return DetectionResult(
        is_anomaly=score >= config.detection.anomaly_score_threshold,
        anomaly_score=score,
        profile_scope=profile_scope,
        evidence=tuple(evidence),
    )


def _malformed_result(record: SourceRecord, config: BehaviourProfileConfig) -> DetectionResult:
    evidence = AnomalyEvidence(
        reason_code=MALFORMED_ROW,
        field=None,
        details=record.malformed_reason or "malformed CSV row",
        score=max(
            config.score_weights.malformed_row,
            config.detection.anomaly_score_threshold,
        ),
    )
    return DetectionResult(
        is_anomaly=True,
        anomaly_score=evidence.score,
        profile_scope="none",
        evidence=(evidence,),
    )


def _jsonl_record(record: SourceRecord, result: DetectionResult) -> dict[str, Any]:
    return {
        "source_file": str(record.source_file),
        "source_row_number": record.source_row_number,
        "source_line_number": record.source_line_number,
        "is_anomaly": result.is_anomaly,
        "anomaly_score": result.anomaly_score,
        "profile_scope": result.profile_scope,
        "evidence": [item.to_dict() for item in result.evidence],
        "event": record.row,
    }


def detect_anomalies(
    *,
    log_paths: list[Path],
    profile_path: Path,
    output_path: Path,
    results_jsonl_path: Path | None = None,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> DetectionSummary:
    if not log_paths:
        raise ValueError("At least one detection log CSV is required")
    profile = load_profile(profile_path)
    config = config_from_mapping(profile["config"])
    headers = collect_union_headers(log_paths, encoding=encoding, delimiter=delimiter)
    total_rows = 0
    anomaly_rows = 0
    malformed_rows = 0
    reason_counts: Counter[str] = Counter()

    with ExitStack() as stack:
        anomaly_writer = stack.enter_context(
            AtomicAnomalyCsvWriter(
                output_path,
                headers,
                encoding=encoding,
                delimiter=delimiter,
            )
        )
        results_writer = (
            stack.enter_context(AtomicJsonlWriter(results_jsonl_path))
            if results_jsonl_path is not None
            else None
        )
        for path in log_paths:
            LOGGER.info("Detecting behavior anomalies in %s", path)
            for record in iter_csv_records(path, encoding=encoding, delimiter=delimiter):
                total_rows += 1
                if record.malformed:
                    malformed_rows += 1
                    result = _malformed_result(record, config)
                else:
                    preprocessed = preprocess_behaviour_row(record.row)
                    result = score_preprocessed_row(preprocessed, profile, config)
                reason_counts.update(item.reason_code for item in result.evidence)
                if result.is_anomaly:
                    anomaly_rows += 1
                    anomaly_writer.write(record, result)
                if results_writer is not None:
                    results_writer.write(_jsonl_record(record, result))

    summary = DetectionSummary(
        total_rows=total_rows,
        anomaly_rows=anomaly_rows,
        normal_rows=total_rows - anomaly_rows,
        malformed_rows=malformed_rows,
        reasons=dict(sorted(reason_counts.items())),
    )
    LOGGER.info(
        "Behavior detection complete: total=%s anomalies=%s normal=%s",
        summary.total_rows,
        summary.anomaly_rows,
        summary.normal_rows,
    )
    return summary
