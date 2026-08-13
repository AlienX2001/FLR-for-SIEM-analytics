from __future__ import annotations

from typing import Any

import pandas as pd

TEXT_COLUMN_CANDIDATES = ("message", "log", "raw_log", "text")
SOURCE_ID_CANDIDATES = ("log_id", "logid", "event_id", "id")
STRUCTURED_IOC_COLUMNS = frozenset(
    {
        "src_ip",
        "source_ip",
        "dst_ip",
        "destination_ip",
        "local_ip",
        "local_address",
        "remote_ip",
        "remote_address",
        "domain",
        "destination_domain",
        "dns_query",
        "hostname_query",
        "http_host",
        "tls_sni",
        "sni",
        "url",
        "uri",
        "http_uri",
        "download",
        "download_url",
        "request_url",
        "md5",
        "sha1",
        "sha_1",
        "sha256",
        "sha_256",
        "file_hash",
        "hash",
    }
)


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
    structured_parts: list[str] = []
    for key, value in row.items():
        normalized_key = str(key).strip().lower()
        if normalized_key not in STRUCTURED_IOC_COLUMNS:
            continue
        if pd.isna(value) or not str(value).strip():
            continue
        structured_parts.append(f"{key}={value}")
    if structured_parts:
        return " ".join(structured_parts)
    if text_column is None:
        return ""
    value = row[text_column]
    return "" if pd.isna(value) else str(value)


def source_log_id_from_row(row: pd.Series, source_column: str | None) -> Any | None:
    if source_column is None:
        return None
    value = row[source_column]
    if pd.isna(value):
        return None
    return value
