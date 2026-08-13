from __future__ import annotations

from pathlib import Path

from federated_lr_pipeline.feature_schemas import SUBCATEGORY_SCHEMAS
from federated_lr_pipeline.specialized_models import _row_text, cross_tokens_for_row
from policy_filter.preprocessing_adapter import preprocess_policy_row
from policy_filter.schema import load_policy
from policy_filter_helpers import write_policy


def parity_policy() -> dict:
    return {
        "version": 1,
        "organization_id": "parity",
        "default_timezone": "UTC",
        "field_mappings": {
            "timestamp": ["event_time_iso"],
            "host": ["host"],
            "user": ["user_uid"],
            "program": ["process_name"],
            "program_path": ["process_exe"],
            "command_line": ["process_command_line"],
            "source_ip": ["src_ip"],
            "destination_ip": ["dst_ip"],
            "domain": ["domain"],
            "protocol": ["protocol_name"],
            "destination_port": ["dst_port"],
            "direction": ["network_direction"],
        },
        "category_aliases": {"system": ["system"], "network": ["network"]},
        "system_policies": [],
        "network_policies": [],
    }


def test_adapter_system_network_and_cross_evidence_matches_federated_helpers(tmp_path: Path) -> None:
    raw_row = {
        "event_time_iso": "2026-07-20T09:00:00Z",
        "host": "linux-01",
        "user_uid": "1000",
        "process_name": "curl",
        "process_exe": "/usr/bin/curl",
        "process_command_line": "curl https://example.com/secret",
        "src_ip": "10.0.0.10",
        "dst_ip": "203.0.113.20",
        "domain": "example.com",
        "protocol_name": "TCP",
        "dst_port": "443",
        "network_direction": "outbound",
        "total_size": "9000",
        "file_path": "/home/alice/secret.txt",
    }
    policy = load_policy(write_policy(tmp_path, parity_policy()))

    adapted = preprocess_policy_row(raw_row, policy)

    assert adapted.system_evidence["text"] == _row_text(
        adapted.canonical_fields,
        SUBCATEGORY_SCHEMAS["system"],
    )
    assert adapted.network_evidence["text"] == _row_text(
        adapted.canonical_fields,
        SUBCATEGORY_SCHEMAS["network"],
    )
    assert adapted.cross_evidence["tokens"] == tuple(cross_tokens_for_row(adapted.canonical_fields))
    assert adapted.token_counts


def test_canonical_values_drive_matching_but_raw_values_are_separate(tmp_path: Path) -> None:
    policy = load_policy(write_policy(tmp_path, parity_policy()))
    adapted = preprocess_policy_row({"process_name": "Curl"}, policy)

    assert adapted.canonical_fields["program"] == "curl"
    assert adapted.resolved_fields["program"].original_value == "Curl"


def test_configured_preprocessing_options_are_applied(tmp_path: Path) -> None:
    payload = parity_policy()
    payload["preprocessing"] = {
        "text_column": "message",
        "include_cross_category": False,
        "preserve_empty_fields": True,
    }
    policy = load_policy(write_policy(tmp_path, payload))

    adapted = preprocess_policy_row(
        {"message": "Administrative Notice", "process_name": ""},
        policy,
    )

    assert "message=administrative_notice" in adapted.normalized_text
    assert adapted.canonical_fields["program"] == ""
    assert adapted.cross_evidence["tokens"] == ()
