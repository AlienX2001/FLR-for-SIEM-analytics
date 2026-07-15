from __future__ import annotations

from typing import Any

import pandas as pd

TEXT_COLUMN_CANDIDATES = ("message", "log", "raw_log", "text")
SOURCE_ID_CANDIDATES = ("log_id", "logid", "event_id", "id")


def detect_text_column(df: pd.DataFrame, requested: str | None = None) -> str | None:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Requested text column '{requested}' is not present")
        return requested
    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    return None


def detect_source_log_id_column(df: pd.DataFrame) -> str | None:
    lower_to_original = {str(column).lower(): str(column) for column in df.columns}
    for candidate in SOURCE_ID_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def normalize_log_row(row: pd.Series, text_column: str | None = None) -> str:
    if text_column is not None:
        value = row[text_column]
        return "" if pd.isna(value) else str(value)
    parts: list[str] = []
    for key, value in row.items():
        if pd.isna(value):
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def source_log_id_from_row(row: pd.Series, source_column: str | None) -> Any | None:
    if source_column is None:
        return None
    value = row[source_column]
    if pd.isna(value):
        return None
    return value
