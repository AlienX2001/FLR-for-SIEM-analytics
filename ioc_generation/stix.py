from __future__ import annotations

import uuid
from typing import Any

from ioc_generation.extractors import IndicatorCandidate

NAMESPACE = uuid.UUID("7ad079d2-5865-4d7a-9f46-7c4cf70aa1f3")
DETERMINISTIC_TIMESTAMP = "2024-01-01T00:00:00Z"


def deterministic_stix_id(stix_type: str, value: str) -> str:
    return f"{stix_type}--{uuid.uuid5(NAMESPACE, f'{stix_type}:{value}')}"


def indicator_pattern(candidate: IndicatorCandidate) -> str:
    value = candidate.value.replace("\\", "\\\\").replace("'", "\\'")
    if candidate.indicator_type == "ipv4":
        return f"[ipv4-addr:value = '{value}']"
    if candidate.indicator_type == "domain":
        return f"[domain-name:value = '{value}']"
    if candidate.indicator_type == "url":
        return f"[url:value = '{value}']"
    if candidate.indicator_type == "md5":
        return f"[file:hashes.'MD5' = '{value}']"
    if candidate.indicator_type == "sha1":
        return f"[file:hashes.'SHA-1' = '{value}']"
    if candidate.indicator_type == "sha256":
        return f"[file:hashes.'SHA-256' = '{value}']"
    raise ValueError(f"Unsupported indicator type: {candidate.indicator_type}")


def make_indicator(
    candidate: IndicatorCandidate,
    *,
    internal_log_id: str,
    org_index: int,
    row_index: int,
    predicted_label: str | None = None,
    risk_probability: float | None = None,
) -> dict[str, Any]:
    name = f"{candidate.indicator_type}:{candidate.value}"
    indicator = {
        "type": "indicator",
        "spec_version": "2.1",
        "id": deterministic_stix_id("indicator", f"{candidate.indicator_type}:{candidate.value}"),
        "created": DETERMINISTIC_TIMESTAMP,
        "modified": DETERMINISTIC_TIMESTAMP,
        "name": name,
        "description": (
            f"Generated from high-risk federated LR prediction {internal_log_id} "
            f"(org {org_index}, row {row_index})."
        ),
        "pattern": indicator_pattern(candidate),
        "pattern_type": "stix",
        "labels": ["federated-lr", "high-risk-log", candidate.indicator_type],
        "x_federated_lr_internal_log_id": internal_log_id,
        "x_federated_lr_org_index": int(org_index),
        "x_federated_lr_row_index": int(row_index),
    }
    if predicted_label is not None:
        indicator["x_federated_lr_predicted_label"] = predicted_label
    if risk_probability is not None:
        indicator["x_federated_lr_risk_probability"] = float(risk_probability)
    return indicator


def make_bundle(indicators: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "bundle",
        "id": deterministic_stix_id(
            "bundle", "|".join(indicator["id"] for indicator in indicators)
        ),
        "objects": indicators,
    }
