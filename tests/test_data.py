from __future__ import annotations

from pathlib import Path

import pandas as pd

from federated_lr_pipeline.data import load_org_dataset


def test_loader_uses_all_columns_when_text_column_not_detected(tmp_path: Path) -> None:
    logs = tmp_path / "logs.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "user": ["admin"],
            "event": ["failed login from 10.0.0.1"],
        }
    ).to_csv(logs, index=False)
    pd.DataFrame({"label": ["malicious"]}).to_csv(labels, index=False)

    dataset = load_org_dataset(logs, labels, org_index=0)

    assert dataset.text_column == "__all_columns__"
    assert dataset.text_columns == ["timestamp", "user", "event"]
    assert dataset.texts == ["2026-01-01T00:00:00Z admin failed login from 10.0.0.1"]


def test_loader_concatenates_requested_text_columns(tmp_path: Path) -> None:
    logs = tmp_path / "logs.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "event_type": ["process_start"],
            "command_line": ["curl http://example.com/a"],
        }
    ).to_csv(logs, index=False)
    pd.DataFrame({"label": ["malicious"]}).to_csv(labels, index=False)

    dataset = load_org_dataset(
        logs,
        labels,
        org_index=0,
        text_columns=["event_type", "command_line"],
    )

    assert dataset.text_column == "event_type,command_line"
    assert dataset.text_columns == ["event_type", "command_line"]
    assert dataset.texts == ["process_start curl http://example.com/a"]


def test_loader_canonicalizes_numeric_columns_and_lowercases_labels(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "event_time_epoch": ["1701468825.819"],
            "src_port": ["443.0"],
            "user_uid": ["001000"],
            "user_euid": ["001001"],
            "group_gid": ["002000"],
            "group_egid": ["002001"],
            "normalized_id": ["000123"],
        }
    ).to_csv(logs, index=False)
    pd.DataFrame({"label": [" Credential Access "]}).to_csv(labels, index=False)

    dataset = load_org_dataset(logs, labels, org_index=0)

    assert dataset.logs_df.loc[0, "event_time_epoch"] == 1701468825.819
    assert dataset.logs_df.loc[0, "src_port"] == 443
    assert dataset.logs_df.loc[0, "user_uid"] == "001000"
    assert dataset.logs_df.loc[0, "user_euid"] == "001001"
    assert dataset.logs_df.loc[0, "group_gid"] == "002000"
    assert dataset.logs_df.loc[0, "group_egid"] == "002001"
    assert dataset.logs_df.loc[0, "normalized_id"] == "000123"
    assert dataset.labels == ["credential access"]
