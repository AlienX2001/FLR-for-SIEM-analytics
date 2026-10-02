from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ioc_generation.extractors import extract_iocs
from ioc_generation.normalize import (
    detect_source_log_id_column,
    detect_text_column,
    normalize_log_row,
    source_log_id_from_row,
)
from ioc_generation.stix import make_bundle, make_indicator
from ioc_generation.utils import read_jsonl, write_csv, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def load_org_frames(org_data: list[Path]) -> list[pd.DataFrame]:
    return [pd.read_csv(path) for path in org_data]


def _explanations_by_id(explanations_path: Path) -> dict[str, dict[str, Any]]:
    explanations = read_jsonl(explanations_path)
    return {
        str(record["internal_log_id"]): record
        for record in explanations
        if "internal_log_id" in record
    }


def _resolve_org_positions(
    high_risk_records: list[dict[str, Any]],
    *,
    org_count: int,
    org_indexes: list[int] | None,
) -> dict[int, int]:
    if org_indexes is not None:
        if len(org_indexes) != org_count:
            raise ValueError(
                "--org-indexes must contain exactly one index per --org-data file"
            )
        if any(index < 0 for index in org_indexes):
            raise ValueError("--org-indexes values must be non-negative")
        if len(set(org_indexes)) != len(org_indexes):
            raise ValueError("--org-indexes values must be unique")
        return {org_index: position for position, org_index in enumerate(org_indexes)}

    observed_indexes = {
        int(record["org_index"])
        for record in high_risk_records
        if "org_index" in record
    }
    if org_count == 1 and len(observed_indexes) == 1:
        return {next(iter(observed_indexes)): 0}
    return {position: position for position in range(org_count)}


def generate_ioc_outputs(
    *,
    high_risk_logs: Path,
    explanations: Path,
    org_data: list[Path],
    output_dir: Path,
    text_column: str | None = None,
    org_indexes: list[int] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    high_risk_records = read_jsonl(high_risk_logs)
    explanations_by_id = _explanations_by_id(explanations)
    org_frames = load_org_frames(org_data)
    text_columns = [detect_text_column(frame, text_column) for frame in org_frames]
    source_columns = [detect_source_log_id_column(frame) for frame in org_frames]
    org_positions = _resolve_org_positions(
        high_risk_records,
        org_count=len(org_frames),
        org_indexes=org_indexes,
    )

    ioc_records: list[dict[str, Any]] = []
    indicators_by_id: dict[str, dict[str, Any]] = {}
    skipped_unmapped_records = 0

    for high_risk_record in high_risk_records:
        org_index = int(high_risk_record["org_index"])
        row_index = int(high_risk_record["row_index"])
        internal_log_id = str(high_risk_record["internal_log_id"])
        position = org_positions.get(org_index)
        if position is None:
            if org_indexes is not None:
                skipped_unmapped_records += 1
                continue
            raise ValueError(
                f"org_index {org_index} has no corresponding --org-data file; "
                "provide --org-indexes to define the mapping"
            )
        frame = org_frames[position]
        if row_index < 0 or row_index >= len(frame):
            raise ValueError(f"row_index {row_index} is out of range for organization {org_index}")

        row = frame.iloc[row_index]
        normalized_log = normalize_log_row(row, text_columns[position])
        source_log_id = source_log_id_from_row(row, source_columns[position])
        explanation = explanations_by_id.get(internal_log_id, {})
        predicted_label = high_risk_record.get(
            "predicted_label",
            high_risk_record.get(
                "ensemble_predicted_label",
                explanation.get(
                    "predicted_label", explanation.get("ensemble_predicted_label")
                ),
            ),
        )
        risk_probability = high_risk_record.get(
            "max_risk_probability",
            high_risk_record.get(
                "ensemble_max_risk_probability",
                explanation.get(
                    "max_risk_probability",
                    explanation.get("ensemble_max_risk_probability"),
                ),
            ),
        )

        for candidate in extract_iocs(normalized_log):
            indicator = make_indicator(
                candidate,
                internal_log_id=internal_log_id,
                org_index=org_index,
                row_index=row_index,
                predicted_label=str(predicted_label) if predicted_label is not None else None,
                risk_probability=float(risk_probability)
                if risk_probability is not None
                else None,
            )
            indicators_by_id[indicator["id"]] = indicator
            contribution_evidence = explanation.get(
                "top_contributions",
                explanation.get("top_contributing_features", []),
            )
            record: dict[str, Any] = {
                "org_index": org_index,
                "row_index": row_index,
                "internal_log_id": internal_log_id,
                "indicator_id": indicator["id"],
                "indicator_type": candidate.indicator_type,
                "indicator_value": candidate.value,
                "predicted_label": predicted_label,
                "max_risk_probability": risk_probability,
                "evidence_by_label_subcategory": json.dumps(
                    contribution_evidence, sort_keys=True
                ),
            }
            if source_log_id is not None:
                record["source_log_id"] = source_log_id
            ioc_records.append(record)

    if skipped_unmapped_records:
        LOGGER.info(
            "Skipped %s high-risk record(s) for organizations not selected by "
            "--org-indexes",
            skipped_unmapped_records,
        )

    indicators = [indicators_by_id[key] for key in sorted(indicators_by_id)]
    write_json(output_dir / "ioc_bundle.json", make_bundle(indicators))
    write_jsonl(output_dir / "ioc_records.jsonl", ioc_records)
    summary_records = [
        {
            "indicator_type": indicator_type,
            "count": count,
        }
        for indicator_type, count in sorted(
            pd.Series([record["indicator_type"] for record in ioc_records])
            .value_counts()
            .to_dict()
            .items()
        )
    ]
    if not summary_records:
        summary_records = [{"indicator_type": "none", "count": 0}]
    write_csv(output_dir / "ioc_summary.csv", summary_records)
