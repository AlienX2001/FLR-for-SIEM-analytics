from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import federated_lr_pipeline.run as run_module
from federated_lr_pipeline.config import PipelineConfig
from federated_lr_pipeline.config import parse_args
from federated_lr_pipeline.run import run_pipeline


def _write_testing_org(tmp_path: Path, org_name: str) -> tuple[Path, Path]:
    rows = []
    labels = []
    for index in range(6):
        benign = index % 2 == 0
        rows.append(
            {
                "event_time_epoch": str(1767225600 + index),
                "event_time_iso": f"2026-01-01T00:00:0{index}Z",
                "protocol_name": "tcp",
                "src_ip": "10.0.0.10" if benign else "203.0.113.50",
                "dst_ip": "10.0.0.20" if benign else "10.0.0.10",
                "src_port": "51514" if benign else "51222",
                "dst_port": "443" if benign else "445",
                "network_direction": "outbound" if benign else "inbound",
                "flow_duration": "30" if benign else "220",
                "total_size": "120" if benign else "9000",
                "protocol_tcp": "1",
                "source": "auditd",
                "entity_id": f"{org_name}-{index}",
                "entity_type": "process",
                "process_pid": str(1000 + index),
                "process_ppid": "4" if benign else "100",
                "process_name": "svchost" if benign else "cmd",
                "process_exe": (
                    r"C:\Windows\System32\svchost.exe"
                    if benign
                    else r"C:\Windows\System32\cmd.exe"
                ),
                "process_command_line": (
                    "svchost.exe -k netsvcs"
                    if benign
                    else "cmd.exe /c type secret.txt"
                ),
                "user_uid": "1000" if benign else "0",
                "file_path": (
                    r"C:\Windows\System32\svchost.exe"
                    if benign
                    else r"C:\Users\Public\secret.txt"
                ),
                "file_permissions": "read_execute",
            }
        )
        labels.append("benign" if benign else "malicious")
    logs_path = tmp_path / f"{org_name}_logs.csv"
    labels_path = tmp_path / f"{org_name}_labels.csv"
    pd.DataFrame(rows).to_csv(logs_path, index=False)
    pd.DataFrame({"label": labels}).to_csv(labels_path, index=False)
    return logs_path, labels_path


def _train_small_model(tmp_path: Path) -> tuple[list[Path], list[Path], Path]:
    org0_logs, org0_labels = _write_testing_org(tmp_path, "org0")
    org1_logs, org1_labels = _write_testing_org(tmp_path, "org1")
    output_dir = tmp_path / "trained"
    run_pipeline(
        PipelineConfig(
            org_data=[org0_logs, org1_logs],
            org_groundtruth=[org0_labels, org1_labels],
            num_features=15,
            federation_iterations=1,
            min_df=1,
            max_df=1.0,
            output_dir=output_dir,
            seed=7,
            batch_size=2,
            learning_rate=0.05,
            local_epochs=1,
            test_size=0.33,
            risk_threshold=0.0,
            debug_plaintext_vocab=True,
        )
    )
    return [org0_logs, org1_logs], [org0_labels, org1_labels], output_dir


def _testing_config(
    *,
    org_logs: list[Path],
    org_labels: list[Path],
    artifact_dir: Path,
    output_dir: Path,
) -> PipelineConfig:
    return PipelineConfig(
        org_data=org_logs,
        org_groundtruth=org_labels,
        num_features=0,
        federation_iterations=0,
        output_dir=output_dir,
        testing=True,
        model_artifact_dir=artifact_dir,
        risk_threshold=0.0,
        debug_plaintext_vocab=True,
    )


def test_testing_mode_skips_training_and_writes_metrics(tmp_path: Path, monkeypatch) -> None:
    org_logs, org_labels, artifact_dir = _train_small_model(tmp_path)
    output_dir = tmp_path / "testing"

    def fail_if_training(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("testing mode must not call train_specialist_round")

    monkeypatch.setattr(run_module, "train_specialist_round", fail_if_training)

    run_pipeline(
        _testing_config(
            org_logs=org_logs,
            org_labels=org_labels,
            artifact_dir=artifact_dir,
            output_dir=output_dir,
        )
    )

    metrics = json.loads((output_dir / "testing_metrics.json").read_text())
    assert "ensemble_model" in metrics
    assert "network_model" in metrics
    assert "system_model" in metrics
    assert "inter_category_model" in metrics
    assert (output_dir / "testing_classification_report.txt").exists()
    assert (output_dir / "testing_confusion_matrix.csv").exists()
    assert not list(output_dir.glob("final_*_weights.npy"))

    predictions = (output_dir / "testing_predictions.jsonl").read_text().splitlines()
    assert predictions
    first_prediction = json.loads(predictions[0])
    assert "true_label" in first_prediction
    assert "ensemble_predicted_label" in first_prediction
    assert "correct" in first_prediction

    explanations = (output_dir / "testing_explanations.jsonl").read_text().splitlines()
    assert explanations
    first_explanation = json.loads(explanations[0])
    assert "true_label" in first_explanation
    assert "ensemble_predicted_label" in first_explanation
    assert "top_contributing_features" in first_explanation


def test_testing_mode_missing_artifact_has_clear_error(tmp_path: Path, monkeypatch) -> None:
    org_logs, org_labels, artifact_dir = _train_small_model(tmp_path)
    (artifact_dir / "label_encoder_classes.json").unlink()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    with pytest.raises(FileNotFoundError, match="Missing required testing artifact"):
        run_pipeline(
            _testing_config(
                org_logs=org_logs,
                org_labels=org_labels,
                artifact_dir=artifact_dir,
                output_dir=tmp_path / "testing",
            )
        )


def test_testing_mode_validates_row_alignment(tmp_path: Path) -> None:
    org_logs, org_labels, artifact_dir = _train_small_model(tmp_path)
    bad_labels = tmp_path / "bad_labels.csv"
    pd.read_csv(org_labels[0]).iloc[:-1].to_csv(bad_labels, index=False)

    with pytest.raises(ValueError, match="Row count mismatch"):
        run_pipeline(
            _testing_config(
                org_logs=[org_logs[0], org_logs[1]],
                org_labels=[bad_labels, org_labels[1]],
                artifact_dir=artifact_dir,
                output_dir=tmp_path / "testing",
            )
        )


def test_testing_mode_validates_pretrained_model_shapes(tmp_path: Path) -> None:
    org_logs, org_labels, artifact_dir = _train_small_model(tmp_path)
    tags = json.loads((artifact_dir / "benign_system_global_vocabulary_tags.json").read_text())
    np.save(artifact_dir / "final_benign_system_weights.npy", np.zeros(len(tags) + 1))

    with pytest.raises(ValueError, match="shape mismatch"):
        run_pipeline(
            _testing_config(
                org_logs=org_logs,
                org_labels=org_labels,
                artifact_dir=artifact_dir,
                output_dir=tmp_path / "testing",
            )
        )


def test_testing_mode_parser_ignores_training_only_parameters() -> None:
    config = parse_args(
        [
            "--testing",
            "--org-data",
            "logs.csv",
            "--org-groundtruth",
            "labels.csv",
            "--model-artifact-dir",
            "trained",
            "--output-dir",
            "testing",
            "--num-features",
            "-1",
            "--federation-iterations",
            "-2",
            "--min-df",
            "0",
            "--max-df",
            "-1",
            "--batch-size",
            "0",
            "--local-epochs",
            "-1",
            "--regularization",
            "-1",
        ]
    )

    assert config.testing
    assert "--num-features" in config.testing_ignored_parameters
    assert "--federation-iterations" in config.testing_ignored_parameters
    assert "--min-df" in config.testing_ignored_parameters
    assert "--max-df" in config.testing_ignored_parameters
    assert "--batch-size" in config.testing_ignored_parameters
    assert "--local-epochs" in config.testing_ignored_parameters
    assert "--regularization" in config.testing_ignored_parameters
