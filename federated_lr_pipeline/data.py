from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_COLUMN_CANDIDATES = ("message", "log", "raw_log", "text")
SOURCE_ID_CANDIDATES = ("log_id", "logid", "event_id", "id")
ALL_COLUMNS_TEXT = "__all_columns__"


@dataclass
class OrgDataset:
    org_index: int
    log_path: Path
    groundtruth_path: Path
    logs_df: pd.DataFrame
    groundtruth_df: pd.DataFrame
    text_column: str
    text_columns: list[str]
    label_column: str
    texts: list[str]
    labels: list[Any]
    row_indices: list[int]
    internal_log_ids: list[str]
    source_log_id_column: str | None

    def source_log_id_at(self, row_index: int) -> Any | None:
        if self.source_log_id_column is None:
            return None
        value = self.logs_df.iloc[row_index][self.source_log_id_column]
        if pd.isna(value):
            return None
        return value


def detect_text_columns(
    logs_df: pd.DataFrame,
    requested: str | None = None,
    requested_columns: list[str] | None = None,
) -> list[str]:
    if requested_columns is not None:
        missing = [column for column in requested_columns if column not in logs_df.columns]
        if missing:
            raise ValueError(
                "Requested text columns are not present in log CSV: "
                + ", ".join(missing)
            )
        return requested_columns
    if requested is not None:
        if requested not in logs_df.columns:
            raise ValueError(f"Requested text column '{requested}' is not present in log CSV")
        return [requested]
    for column in TEXT_COLUMN_CANDIDATES:
        if column in logs_df.columns:
            return [column]
    return list(logs_df.columns)


def summarize_text_columns(text_columns: list[str], logs_df: pd.DataFrame) -> str:
    if text_columns == list(logs_df.columns):
        return ALL_COLUMNS_TEXT
    if len(text_columns) == 1:
        return text_columns[0]
    return ",".join(text_columns)


def build_texts_from_logs(logs_df: pd.DataFrame, text_columns: list[str]) -> list[str]:
    text_frame = logs_df.loc[:, text_columns]
    text_rows: list[str] = []
    for row in text_frame.itertuples(index=False, name=None):
        values = [str(value) for value in row if not pd.isna(value) and str(value).strip()]
        text_rows.append(" ".join(values))
    return text_rows


def detect_label_column(groundtruth_df: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None:
        if requested not in groundtruth_df.columns:
            raise ValueError(
                f"Requested label column '{requested}' is not present in groundtruth CSV"
            )
        return requested
    if len(groundtruth_df.columns) == 1:
        return str(groundtruth_df.columns[0])
    if "label" in groundtruth_df.columns:
        return "label"
    raise ValueError(
        "Could not auto-detect label column. Provide --label-column or use a 'label' column."
    )


def detect_source_log_id_column(logs_df: pd.DataFrame) -> str | None:
    lower_to_original = {str(column).lower(): str(column) for column in logs_df.columns}
    for candidate in SOURCE_ID_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def load_org_dataset(
    log_path: str | Path,
    groundtruth_path: str | Path,
    org_index: int,
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    label_column: str | None = None,
) -> OrgDataset:
    log_path = Path(log_path)
    groundtruth_path = Path(groundtruth_path)
    logs_df = pd.read_csv(log_path)
    groundtruth_df = pd.read_csv(groundtruth_path)

    if len(logs_df) != len(groundtruth_df):
        raise ValueError(
            f"Row count mismatch for organization {org_index}: "
            f"{log_path} has {len(logs_df)} rows but {groundtruth_path} has "
            f"{len(groundtruth_df)} rows"
        )

    resolved_text_columns = detect_text_columns(logs_df, text_column, text_columns)
    resolved_text_column = summarize_text_columns(resolved_text_columns, logs_df)
    resolved_label_column = detect_label_column(groundtruth_df, label_column)
    source_id_column = detect_source_log_id_column(logs_df)

    texts = build_texts_from_logs(logs_df, resolved_text_columns)
    labels = groundtruth_df[resolved_label_column].tolist()
    row_indices = list(range(len(logs_df)))
    internal_ids = [f"org_{org_index}_row_{row_index}" for row_index in row_indices]

    return OrgDataset(
        org_index=org_index,
        log_path=log_path,
        groundtruth_path=groundtruth_path,
        logs_df=logs_df,
        groundtruth_df=groundtruth_df,
        text_column=resolved_text_column,
        text_columns=resolved_text_columns,
        label_column=resolved_label_column,
        texts=texts,
        labels=labels,
        row_indices=row_indices,
        internal_log_ids=internal_ids,
        source_log_id_column=source_id_column,
    )


def load_all_orgs(
    log_paths: list[Path],
    groundtruth_paths: list[Path],
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    label_column: str | None = None,
) -> list[OrgDataset]:
    return [
        load_org_dataset(
            log_path=log_path,
            groundtruth_path=groundtruth_path,
            org_index=org_index,
            text_column=text_column,
            text_columns=text_columns,
            label_column=label_column,
        )
        for org_index, (log_path, groundtruth_path) in enumerate(
            zip(log_paths, groundtruth_paths)
        )
    ]
