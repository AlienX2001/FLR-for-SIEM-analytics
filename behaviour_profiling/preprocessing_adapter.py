from __future__ import annotations

from collections.abc import Mapping

from federated_lr_pipeline.specialized_models import (
    _field_value,
    _is_missing_value,
    field_aware_tokens,
)

from behaviour_profiling.models import PreprocessedBehaviourRow


def preprocess_behaviour_row(raw_row: Mapping[str, str]) -> PreprocessedBehaviourRow:
    """Adapt federated field normalization without invoking model feature generation."""
    canonical_fields: dict[str, str] = {}
    tokens_by_field: dict[str, tuple[str, ...]] = {}
    normalized_tokens: list[str] = []
    for column, value in raw_row.items():
        if _is_missing_value(value):
            continue
        canonical_value = _field_value(value)
        canonical_fields[str(column)] = canonical_value
        tokens = tuple(field_aware_tokens(str(column), value))
        tokens_by_field[str(column)] = tokens
        normalized_tokens.extend(tokens)
    return PreprocessedBehaviourRow(
        normalized_text=" ".join(normalized_tokens),
        canonical_fields=canonical_fields,
        field_tokens=tokens_by_field,
    )

