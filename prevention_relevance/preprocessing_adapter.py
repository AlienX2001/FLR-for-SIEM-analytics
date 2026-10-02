from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from federated_lr_pipeline.feature_schemas import SUBCATEGORY_SCHEMAS
from federated_lr_pipeline.specialized_models import (
    TARGET_LEAKAGE_COLUMNS,
    _field_value,
    _is_missing_value,
    _row_text,
    field_aware_tokens,
)

from prevention_relevance.models import PreprocessedChangeRow


def _resolve_fields(
    raw_row: Mapping[str, Any],
    field_mappings: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    canonical: dict[str, str] = {}
    conflicts: dict[str, tuple[str, ...]] = {}
    for logical_name, aliases in field_mappings.items():
        values = [
            _field_value(raw_row[column])
            for column in aliases
            if column in raw_row and not _is_missing_value(raw_row[column])
        ]
        distinct = tuple(dict.fromkeys(value for value in values if value))
        if len(distinct) > 1:
            conflicts[logical_name] = distinct
        elif distinct:
            canonical[logical_name] = distinct[0]
    return canonical, conflicts


def preprocess_change_row(
    raw_row: Mapping[str, Any],
    *,
    field_mappings: Mapping[str, tuple[str, ...]],
) -> PreprocessedChangeRow:
    """Adapt shared federated field-aware preprocessing for graph extraction."""
    safe_row = {
        str(column): value
        for column, value in raw_row.items()
        if str(column).strip().lower() not in TARGET_LEAKAGE_COLUMNS
    }
    canonical_fields, conflicts = _resolve_fields(safe_row, field_mappings)

    normalized_tokens: list[str] = []
    for column, value in safe_row.items():
        normalized_tokens.extend(field_aware_tokens(column, value))
    normalized_text = " ".join(dict.fromkeys(normalized_tokens))

    def tokens_for(subcategory: str) -> tuple[str, ...]:
        text = _row_text(safe_row, SUBCATEGORY_SCHEMAS[subcategory])
        return tuple(token for token in text.split() if token)

    return PreprocessedChangeRow(
        canonical_fields=canonical_fields,
        normalized_text=normalized_text,
        system_tokens=tokens_for("system"),
        network_tokens=tokens_for("network"),
        identity_tokens=tokens_for("identity"),
        llm_tokens=tokens_for("llm"),
        cloud_tokens=tokens_for("cloud"),
        conflicts=conflicts,
    )

