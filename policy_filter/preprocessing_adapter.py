from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from federated_lr_pipeline.feature_schemas import SUBCATEGORY_SCHEMAS
from federated_lr_pipeline.local_training import build_token_counters
from federated_lr_pipeline.specialized_models import (
    _field_value,
    _is_missing_value,
    _row_text,
    cross_tokens_for_row,
    field_aware_tokens,
)

from policy_filter.models import PolicyDocument, PreprocessedPolicyRow, ResolvedField


def _nonempty_text(value: Any) -> str | None:
    if _is_missing_value(value):
        return None
    text = str(value)
    return text if text.strip() else None


def canonicalize_policy_value(value: Any) -> str:
    if _is_missing_value(value):
        return ""
    return _field_value(value)


def normalized_event_fields(
    raw_row: Mapping[str, str],
    *,
    excluded_fields: set[str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(column), canonicalize_policy_value(value))
        for column, value in sorted(raw_row.items())
        if column not in excluded_fields
    )


def resolve_fields(
    raw_row: Mapping[str, str],
    field_mappings: Mapping[str, tuple[str, ...]],
) -> dict[str, ResolvedField]:
    resolved: dict[str, ResolvedField] = {}
    for logical_name, aliases in field_mappings.items():
        candidates: list[tuple[str, str, str]] = []
        for column_name in aliases:
            if column_name not in raw_row:
                continue
            original_value = _nonempty_text(raw_row[column_name])
            if original_value is None:
                continue
            candidates.append((column_name, original_value, _field_value(original_value)))
        if not candidates:
            resolved[logical_name] = ResolvedField(logical_name=logical_name, value=None)
            continue
        canonical_values = {canonical for _, _, canonical in candidates}
        first_column, first_original, first_canonical = candidates[0]
        if len(canonical_values) > 1:
            resolved[logical_name] = ResolvedField(
                logical_name=logical_name,
                value=None,
                source_column=first_column,
                original_value=first_original,
                conflicting_columns=tuple(column for column, _, _ in candidates),
                conflicting_values=tuple(original for _, original, _ in candidates),
            )
            continue
        resolved[logical_name] = ResolvedField(
            logical_name=logical_name,
            value=first_canonical,
            source_column=first_column,
            original_value=first_original,
        )
    return resolved


def _canonical_fields(
    resolved: Mapping[str, ResolvedField],
    *,
    preserve_empty_fields: bool,
) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for logical_name, field in resolved.items():
        if field.is_conflicting:
            continue
        if field.value is None:
            if preserve_empty_fields:
                canonical[logical_name] = ""
            continue
        canonical[logical_name] = field.value
        if field.source_column:
            canonical[field.source_column] = field.value
    return canonical


def _field_text(canonical_fields: Mapping[str, str]) -> str:
    tokens: list[str] = []
    for logical_name, value in canonical_fields.items():
        tokens.extend(field_aware_tokens(logical_name, value))
    return " ".join(tokens)


def _subcategory_evidence(
    canonical_fields: Mapping[str, str],
    subcategory: str,
) -> dict[str, Any]:
    attributes = SUBCATEGORY_SCHEMAS.get(subcategory, [])
    text = _row_text(canonical_fields, attributes)
    return {
        "text": text,
        "tokens": tuple(token for token in text.split() if token),
        "fields": {
            key: value for key, value in canonical_fields.items() if key in attributes
        },
    }


def _detected_categories(
    canonical_fields: Mapping[str, str],
    policy: PolicyDocument,
) -> frozenset[str]:
    category = canonical_fields.get("category")
    if category:
        normalized = category.lower()
        categories = {
            category_name
            for category_name, aliases in policy.category_aliases.items()
            if normalized == category_name.lower() or normalized in aliases
        }
        if categories:
            return frozenset(
                item for item in categories if item in {"system", "network"}
            )
        return frozenset()

    system_fields = {
        "user",
        "user_groups",
        "program",
        "program_path",
        "parent_program",
        "command_line",
    }
    network_fields = {
        "source_ip",
        "destination_ip",
        "domain",
        "protocol",
        "source_port",
        "destination_port",
        "direction",
    }
    categories: set[str] = set()
    if any(canonical_fields.get(field) for field in system_fields):
        categories.add("system")
    if any(canonical_fields.get(field) for field in network_fields):
        categories.add("network")
    return frozenset(categories)


def preprocess_policy_row(
    raw_row: Mapping[str, str],
    policy: PolicyDocument,
) -> PreprocessedPolicyRow:
    resolved = resolve_fields(raw_row, policy.field_mappings)
    conflicts = {
        logical_name: field
        for logical_name, field in resolved.items()
        if field.is_conflicting
    }
    preserve_empty_fields = bool(
        policy.preprocessing.get("preserve_empty_fields", False)
    )
    canonical_fields = _canonical_fields(
        resolved,
        preserve_empty_fields=preserve_empty_fields,
    )
    normalized_tokens = _field_text(canonical_fields).split()
    text_column = policy.preprocessing.get("text_column")
    if isinstance(text_column, str) and text_column in raw_row:
        normalized_tokens.extend(
            field_aware_tokens(text_column, raw_row[text_column])
        )
    normalized_text = " ".join(dict.fromkeys(normalized_tokens))
    counters = build_token_counters([normalized_text])[0]
    cross_tokens = (
        cross_tokens_for_row(canonical_fields)
        if bool(policy.preprocessing.get("include_cross_category", True))
        else []
    )
    return PreprocessedPolicyRow(
        normalized_text=normalized_text,
        token_counts=counters,
        canonical_fields=canonical_fields,
        resolved_fields=resolved,
        system_evidence=_subcategory_evidence(canonical_fields, "system"),
        network_evidence=_subcategory_evidence(canonical_fields, "network"),
        cross_evidence={
            "text": " ".join(cross_tokens),
            "tokens": tuple(cross_tokens),
            "fields": dict(canonical_fields),
        },
        detected_categories=_detected_categories(canonical_fields, policy),
        conflicts=conflicts,
    )
