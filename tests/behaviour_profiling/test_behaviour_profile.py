from __future__ import annotations

import csv
import json
from pathlib import Path

from federated_lr_pipeline.specialized_models import field_aware_tokens

from behaviour_profiling.cli import run
from behaviour_profiling.config import config_from_mapping
from behaviour_profiling.detector import (
    NUMERIC_OUTSIDE_BENIGN_PROFILE,
    UNKNOWN_CATEGORICAL_VALUE,
    UNKNOWN_ENTITY,
    detect_anomalies,
    score_preprocessed_row,
)
from behaviour_profiling.preprocessing_adapter import preprocess_behaviour_row
from behaviour_profiling.trainer import load_profile, train_profile


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _config():
    return config_from_mapping(
        {
            "version": 1,
            "profile_name": "test-profile",
            "fields": {
                "entity": ["host"],
                "categorical": ["process_name"],
                "numeric": ["duration"],
                "text": ["command_line"],
                "ignored": ["event_time_iso", "label"],
                "auto_infer": False,
            },
            "training": {
                "minimum_rows": 2,
                "minimum_entity_rows": 2,
                "maximum_entities": 10,
                "categorical_minimum_count": 1,
                "token_minimum_count": 1,
            },
            "detection": {
                "anomaly_score_threshold": 1.0,
                "maximum_unseen_token_ratio": 0.25,
            },
        }
    )


def _training_files(tmp_path: Path) -> tuple[Path, Path]:
    logs = tmp_path / "logs.csv"
    labels = tmp_path / "labels.csv"
    _write_csv(
        logs,
        ["event_time_iso", "host", "process_name", "duration", "command_line"],
        [
            {
                "event_time_iso": "2026-01-01T00:00:00Z",
                "host": "HOST-01",
                "process_name": "browser.exe",
                "duration": "1.0",
                "command_line": "browser.exe --safe",
            },
            {
                "event_time_iso": "2026-01-01T00:01:00Z",
                "host": "HOST-01",
                "process_name": "powershell.exe",
                "duration": "99",
                "command_line": "powershell.exe -enc bad",
            },
            {
                "event_time_iso": "2026-01-01T00:02:00Z",
                "host": "HOST-01",
                "process_name": "browser.exe",
                "duration": "2.0",
                "command_line": "browser.exe --safe",
            },
        ],
    )
    _write_csv(labels, ["label"], [{"label": "Benign"}, {"label": "Attack"}, {"label": "Benign"}])
    return logs, labels


def test_preprocessing_adapter_reuses_federated_field_tokens() -> None:
    row = {"process_name": "PowerShell.EXE", "duration": "12"}
    result = preprocess_behaviour_row(row)

    assert list(result.field_tokens["process_name"]) == field_aware_tokens(
        "process_name", "PowerShell.EXE"
    )
    assert result.canonical_fields["process_name"] == "powershell.exe"


def test_training_uses_only_row_aligned_benign_groundtruth(tmp_path: Path) -> None:
    logs, labels = _training_files(tmp_path)
    output = tmp_path / "profile.json"

    profile = train_profile(
        log_paths=[logs],
        groundtruth_paths=[labels],
        label_column="label",
        benign_label="Benign",
        output_path=output,
        config=_config(),
    )

    assert profile["training_summary"]["benign_rows_used"] == 2
    assert profile["training_summary"]["non_benign_rows_skipped"] == 1
    assert profile["global_profile"]["numeric_fields"]["duration"]["maximum"] == 2.0
    assert profile["global_profile"]["categorical_fields"]["process_name"][
        "allowed_values"
    ] == ["browser.exe"]
    assert output.exists()


def test_detector_flags_unknown_categorical_numeric_and_entity_behavior(
    tmp_path: Path,
) -> None:
    logs, labels = _training_files(tmp_path)
    profile_path = tmp_path / "profile.json"
    train_profile(
        log_paths=[logs],
        groundtruth_paths=[labels],
        label_column="label",
        output_path=profile_path,
        config=_config(),
    )
    profile = load_profile(profile_path)
    config = _config()

    normal = score_preprocessed_row(
        preprocess_behaviour_row(
            {
                "host": "HOST-01",
                "process_name": "browser.exe",
                "duration": "1.5",
                "command_line": "browser.exe --safe",
            }
        ),
        profile,
        config,
    )
    unusual = score_preprocessed_row(
        preprocess_behaviour_row(
            {
                "host": "HOST-01",
                "process_name": "shell.exe",
                "duration": "10",
                "command_line": "browser.exe --safe",
            }
        ),
        profile,
        config,
    )
    unknown_entity = score_preprocessed_row(
        preprocess_behaviour_row(
            {
                "host": "HOST-02",
                "process_name": "browser.exe",
                "duration": "1.5",
                "command_line": "browser.exe --safe",
            }
        ),
        profile,
        config,
    )

    assert normal.is_anomaly is False
    assert {item.reason_code for item in unusual.evidence} == {
        UNKNOWN_CATEGORICAL_VALUE,
        NUMERIC_OUTSIDE_BENIGN_PROFILE,
    }
    assert unknown_entity.is_anomaly is True
    assert UNKNOWN_ENTITY in {item.reason_code for item in unknown_entity.evidence}


def test_missing_value_seen_during_benign_training_is_not_anomalous(tmp_path: Path) -> None:
    logs = tmp_path / "logs.csv"
    _write_csv(
        logs,
        ["process_name", "duration"],
        [
            {"process_name": "browser.exe", "duration": "1"},
            {"process_name": "browser.exe", "duration": ""},
        ],
    )
    config = config_from_mapping(
        {
            "fields": {
                "categorical": ["process_name"],
                "numeric": ["duration"],
                "auto_infer": False,
            },
            "training": {"minimum_rows": 2},
        }
    )
    profile_path = tmp_path / "profile.json"
    profile = train_profile(
        log_paths=[logs],
        output_path=profile_path,
        config=config,
    )

    result = score_preprocessed_row(
        preprocess_behaviour_row({"process_name": "browser.exe", "duration": ""}),
        profile,
        config,
    )

    assert result.is_anomaly is False


def test_detection_writes_only_anomalies_and_all_results_jsonl(tmp_path: Path) -> None:
    logs, labels = _training_files(tmp_path)
    profile_path = tmp_path / "profile.json"
    train_profile(
        log_paths=[logs],
        groundtruth_paths=[labels],
        label_column="label",
        output_path=profile_path,
        config=_config(),
    )
    detection_logs = tmp_path / "detect.csv"
    _write_csv(
        detection_logs,
        ["host", "process_name", "duration", "command_line"],
        [
            {
                "host": "HOST-01",
                "process_name": "browser.exe",
                "duration": "1.5",
                "command_line": "browser.exe --safe",
            },
            {
                "host": "HOST-01",
                "process_name": "malware.exe",
                "duration": "1.5",
                "command_line": "browser.exe --safe",
            },
        ],
    )
    anomalies = tmp_path / "anomalies.csv"
    results = tmp_path / "results.jsonl"

    summary = detect_anomalies(
        log_paths=[detection_logs],
        profile_path=profile_path,
        output_path=anomalies,
        results_jsonl_path=results,
    )

    assert summary.total_rows == 2
    assert summary.anomaly_rows == 1
    assert len(anomalies.read_text(encoding="utf-8").splitlines()) == 2
    result_records = [json.loads(line) for line in results.read_text().splitlines()]
    assert [record["is_anomaly"] for record in result_records] == [False, True]


def test_cli_train_and_detect(tmp_path: Path) -> None:
    logs, labels = _training_files(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config().to_dict()), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    anomaly_path = tmp_path / "anomalies.csv"

    assert run(
        [
            "train",
            "--logs",
            str(logs),
            "--groundtruth",
            str(labels),
            "--label-column",
            "label",
            "--config",
            str(config_path),
            "--output",
            str(profile_path),
        ]
    ) == 0
    assert run(
        [
            "detect",
            "--logs",
            str(logs),
            "--profile",
            str(profile_path),
            "--output",
            str(anomaly_path),
        ]
    ) == 0
    assert profile_path.exists()
    assert anomaly_path.exists()
