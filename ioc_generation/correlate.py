from __future__ import annotations

import json
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


def load_org_frames(org_data: list[Path]) -> list[pd.DataFrame]:
    return [pd.read_csv(path) for path in org_data]


def _explanations_by_id(explanations_path: Path) -> dict[str, dict[str, Any]]:
    explanations = read_jsonl(explanations_path)
    return {
        str(record["internal_log_id"]): record
        for record in explanations
        if "internal_log_id" in record
    }


def generate_ioc_outputs(
    *,
    high_risk_logs: Path,
    explanations: Path,
    org_data: list[Path],
    output_dir: Path,
    text_column: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    high_risk_records = read_jsonl(high_risk_logs)
    explanations_by_id = _explanations_by_id(explanations)
    org_frames = load_org_frames(org_data)
    text_columns = [detect_text_column(frame, text_column) for frame in org_frames]
    source_columns = [detect_source_log_id_column(frame) for frame in org_frames]

    ioc_records: list[dict[str, Any]] = []
    indicators_by_id: dict[str, dict[str, Any]] = {}

    for high_risk_record in high_risk_records:
        org_index = int(high_risk_record["org_index"])
        row_index = int(high_risk_record["row_index"])
        internal_log_id = str(high_risk_record["internal_log_id"])
        if org_index < 0 or org_index >= len(org_frames):
            raise ValueError(f"org_index {org_index} is out of range for --org-data")
        frame = org_frames[org_index]
        if row_index < 0 or row_index >= len(frame):
            raise ValueError(f"row_index {row_index} is out of range for organization {org_index}")

        row = frame.iloc[row_index]
        normalized_log = normalize_log_row(row, text_columns[org_index])
        source_log_id = source_log_id_from_row(row, source_columns[org_index])
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
