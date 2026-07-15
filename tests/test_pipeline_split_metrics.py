from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from federated_lr_pipeline.config import PipelineConfig
from federated_lr_pipeline.run import run_pipeline


def _write_org(tmp_path: Path, org_name: str) -> tuple[Path, Path]:
    rows = []
    labels = []
    for index in range(10):
        if index % 2 == 0:
            rows.append(
                {
                    "event_time_epoch": str(1767225600 + index),
                    "event_time_iso": f"2026-01-01T00:00:0{index}Z",
                    "protocol_name": "tcp",
                    "protocol_number": "6",
                    "network_direction": "outbound",
                    "src_ip": "10.0.0.10",
                    "dst_ip": "10.0.0.20",
                    "src_port": "51514",
                    "dst_port": "443",
                    "local_address": "10.0.0.10",
                    "local_port": "51514",
                    "remote_address": "10.0.0.20",
                    "remote_port": "443",
                    "flow_duration": "32",
                    "duration": "32",
                    "rate": "1.0",
                    "srate": "0.5",
                    "drate": "0.5",
                    "header_length": "20",
                    "total_size": "120",
                    "total_sum": "240",
                    "packet_number": "3",
                    "iat": "10",
                    "tcp_ack": "1",
                    "protocol_https": "1",
                    "protocol_tcp": "1",
                    "label": "benign_network",
                    "sub_label": "normal",
                    "sub_label_cat": "benign",
                    "source": "auditd",
                    "entity_id": f"{org_name}-entity-{index}",
                    "entity_type": "process",
                    "source_entity_id": "parent-safe",
                    "target_entity_id": "child-safe",
                    "process_pid": str(1000 + index),
                    "process_ppid": "4",
                    "process_tgid": str(1000 + index),
                    "process_name": "svchost",
                    "process_exe": r"C:\Windows\System32\svchost.exe",
                    "process_command_line": "svchost.exe -k netsvcs",
                    "user_uid": "1000",
                    "user_euid": "1000",
                    "group_gid": "1000",
                    "group_egid": "1000",
                    "file_path": r"C:\Windows\System32\svchost.exe",
                    "file_subtype": "exe",
                    "file_permissions": "read_execute",
                    "file_mode": "755",
                }
            )
            labels.append("benign")
        else:
            rows.append(
                {
                    "event_time_epoch": str(1767225600 + index),
                    "event_time_iso": f"2026-01-01T00:00:0{index}Z",
                    "protocol_name": "tcp",
                    "protocol_number": "6",
                    "network_direction": "inbound",
                    "src_ip": "203.0.113.50",
                    "dst_ip": "10.0.0.10",
                    "src_port": "51222",
                    "dst_port": "445",
                    "local_address": "10.0.0.10",
                    "local_port": "445",
                    "remote_address": "203.0.113.50",
                    "remote_port": "51222",
                    "flow_duration": "220",
                    "duration": "220",
                    "rate": "9.0",
                    "srate": "7.0",
                    "drate": "2.0",
                    "header_length": "40",
                    "total_size": "9000",
                    "total_sum": "12000",
                    "packet_number": "40",
                    "iat": "2",
                    "tcp_syn": "1",
                    "tcp_ack": "1",
                    "syn_count": "4",
                    "ack_count": "8",
                    "protocol_tcp": "1",
                    "label": "malicious_network",
                    "sub_label": "lateral",
                    "sub_label_cat": "malicious",
                    "source": "auditd",
                    "entity_id": f"{org_name}-entity-{index}",
                    "entity_type": "process",
                    "source_entity_id": "parent-suspicious",
                    "target_entity_id": "child-suspicious",
                    "process_pid": str(2000 + index),
                    "process_ppid": "100",
                    "process_tgid": str(2000 + index),
                    "process_name": "cmd",
                    "process_exe": r"C:\Windows\System32\cmd.exe",
                    "process_command_line": "cmd.exe /c whoami",
                    "user_uid": "0",
                    "user_euid": "0",
                    "group_gid": "0",
                    "group_egid": "0",
                    "file_path": r"C:\Windows\System32\cmd.exe",
                    "file_subtype": "exe",
                    "file_permissions": "read_execute",
                    "file_mode": "755",
                }
            )
            labels.append("malicious")
    logs_path = tmp_path / f"{org_name}_logs.csv"
    labels_path = tmp_path / f"{org_name}_labels.csv"
    pd.DataFrame(rows).to_csv(logs_path, index=False)
    pd.DataFrame({"label": labels}).to_csv(labels_path, index=False)
    return logs_path, labels_path


def test_pipeline_reports_held_out_test_metrics_and_class_weights(tmp_path: Path) -> None:
    org0_logs, org0_labels = _write_org(tmp_path, "org0")
    org1_logs, org1_labels = _write_org(tmp_path, "org1")
    output_dir = tmp_path / "outputs"

    run_pipeline(
        PipelineConfig(
            org_data=[org0_logs, org1_logs],
            org_groundtruth=[org0_labels, org1_labels],
            num_features=20,
            federation_iterations=2,
            min_df=1,
            max_df=1.0,
            output_dir=output_dir,
            seed=123,
            batch_size=4,
            learning_rate=0.1,
            local_epochs=1,
            test_size=0.2,
            class_weight="balanced",
            risk_threshold=0.0,
            debug_plaintext_vocab=True,
        )
    )

    metrics = json.loads((output_dir / "training_metrics.json").read_text())

    assert metrics["split"]["test_size"] == 0.2
    assert metrics["class_weight"]["mode"] == "balanced"
    assert metrics["rounds"]
    final_round = metrics["rounds"][-1]
    assert "ensemble_metrics" in final_round
    assert "specialists" in final_round
    assert set(final_round["specialists"]) == {"benign", "malicious"}
    assert "global_training_metrics" not in final_round
    assert "test_accuracy" in final_round["ensemble_metrics"]
    assert final_round["ensemble_metrics"]["per_org"][0]["num_test_examples"] == 2
    assert (output_dir / "hierarchical_model_manifest.json").exists()
    assert (output_dir / "manual_logit_fusion.json").exists()
    assert (output_dir / "final_benign_system_weights.npy").exists()
    assert (output_dir / "final_malicious_network_weights.npy").exists()
    assert (output_dir / "benign_cross_global_vocabulary_tags.json").exists()

    predictions = (output_dir / "predictions.jsonl").read_text().splitlines()
    assert predictions
    first_prediction = json.loads(predictions[0])
    assert "predicted_label" in first_prediction
    assert "probabilities" in first_prediction
    assert "label_logits" in first_prediction
    assert "subcategory_logits" in first_prediction
    assert "top_contributions" in first_prediction

    explanations = (output_dir / "explanations.jsonl").read_text().splitlines()
    assert explanations
    first_explanation = json.loads(explanations[0])
    assert "label_logits" in first_explanation
    assert "subcategory_logits" in first_explanation
    assert "top_contributions" in first_explanation
