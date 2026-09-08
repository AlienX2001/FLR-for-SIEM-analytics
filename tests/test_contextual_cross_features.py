from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from federated_lr_pipeline.config import parse_args
from federated_lr_pipeline.prf import derive_prf_key, hmac_sha256_tag
from federated_lr_pipeline.specialized_models import (
    BenignNoveltyBaseline,
    build_benign_novelty_baselines,
    build_subcategory_texts,
    contextual_feature_tokens_for_rows,
    contextual_cross_tokens_for_rows,
    cross_tokens_for_row,
)


FAILED_THEN_SUCCESS_TOKEN = (
    "cross:failed_login_burst_and_successful_login_same_user_15m"
)


def _failed(timestamp: str | None, *, user: str = "alice") -> dict[str, str]:
    row = {"user_uid": user, "login_result": "failed_login"}
    if timestamp is not None:
        row["event_time_iso"] = timestamp
    return row


def _success(timestamp: str, *, user: str = "alice") -> dict[str, str]:
    return {
        "event_time_iso": timestamp,
        "user_uid": user,
        "login_result": "success",
    }


def test_causal_context_correlates_prior_event_for_same_user() -> None:
    tokens = contextual_cross_tokens_for_rows(
        [
            _failed("2026-01-01T10:00:00Z"),
            _success("2026-01-01T10:05:00Z"),
        ]
    )

    assert FAILED_THEN_SUCCESS_TOKEN not in tokens[0]
    assert FAILED_THEN_SUCCESS_TOKEN in tokens[1]


def test_causal_context_does_not_cross_scope_values() -> None:
    tokens = contextual_cross_tokens_for_rows(
        [
            _failed("2026-01-01T10:00:00Z", user="alice"),
            _success("2026-01-01T10:05:00Z", user="bob"),
        ]
    )

    assert all(FAILED_THEN_SUCCESS_TOKEN not in row_tokens for row_tokens in tokens)


def test_context_window_boundary_is_inclusive_and_then_expires() -> None:
    tokens = contextual_cross_tokens_for_rows(
        [
            _failed("2026-01-01T10:00:00Z"),
            _success("2026-01-01T10:15:00Z"),
            _success("2026-01-01T10:15:01Z"),
        ]
    )

    assert FAILED_THEN_SUCCESS_TOKEN in tokens[1]
    assert FAILED_THEN_SUCCESS_TOKEN not in tokens[2]


def test_causal_window_statistics_emit_network_behavior_tokens() -> None:
    rows = [
        {
            "event_time_iso": f"2026-01-01T10:00:{index:02d}Z",
            "src_ip": "10.0.0.10",
            "dst_ip": f"10.0.1.{index + 1}",
            "dst_port": str(8000 + index),
            "rate": "20",
            "total_size": "600",
        }
        for index in range(10)
    ]

    tokens = contextual_cross_tokens_for_rows(rows)

    assert (
        "cross:window_flow_burst_and_high_beacon_rate_same_src_ip_15m"
        in tokens[-1]
    )
    assert (
        "cross:window_destination_fanout_and_high_beacon_rate_same_src_ip_15m"
        in tokens[-1]
    )
    assert (
        "cross:window_port_fanout_and_high_beacon_rate_same_src_ip_15m"
        in tokens[-1]
    )
    assert (
        "cross:window_high_volume_and_high_beacon_rate_same_src_ip_15m"
        in tokens[-1]
    )
    assert all(
        "cross:window_flow_burst_and_high_beacon_rate_same_src_ip_15m"
        not in row_tokens
        for row_tokens in tokens[:-1]
    )


def test_causal_window_detects_repeated_destination() -> None:
    rows = [
        {
            "event_time_iso": f"2026-01-01T10:00:0{index}Z",
            "src_ip": "10.0.0.10",
            "dst_ip": "10.0.1.20",
            "dst_port": "443",
            "rate": "20",
        }
        for index in range(5)
    ]

    tokens = contextual_cross_tokens_for_rows(rows)

    assert (
        "cross:window_repeated_destination_and_high_beacon_rate_same_src_ip_15m"
        in tokens[-1]
    )


def test_context_processing_preserves_original_row_alignment() -> None:
    tokens = contextual_cross_tokens_for_rows(
        [
            _success("2026-01-01T10:05:00Z"),
            _failed("2026-01-01T10:00:00Z"),
        ]
    )

    assert FAILED_THEN_SUCCESS_TOKEN in tokens[0]
    assert FAILED_THEN_SUCCESS_TOKEN not in tokens[1]


def test_missing_timestamp_uses_only_single_row_evidence() -> None:
    tokens = contextual_cross_tokens_for_rows(
        [
            _failed(None),
            _success("2026-01-01T10:05:00Z"),
        ]
    )

    assert all(FAILED_THEN_SUCCESS_TOKEN not in row_tokens for row_tokens in tokens)


def test_matching_epoch_and_iso_timestamps_are_accepted() -> None:
    rows = [
        {
            **_failed("2026-01-01T10:00:00Z"),
            "event_time_epoch": "1767261600",
        },
        {
            **_success("2026-01-01T10:01:00Z"),
            "event_time_epoch": "1767261660",
        },
    ]

    tokens = contextual_cross_tokens_for_rows(rows)

    assert FAILED_THEN_SUCCESS_TOKEN in tokens[1]


def test_context_configuration_defaults_to_fixed_15_minute_window(tmp_path) -> None:
    config = parse_args(
        [
            "--org-data",
            "logs.csv",
            "--org-groundtruth",
            "labels.csv",
            "--num-features",
            "10",
            "--federation-iterations",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert config.context_window_minutes == 15.0
    assert config.to_json_dict()["cross_context"] == {
        "version": 2,
        "causal": True,
        "window_minutes": 15.0,
        "timestamp_epoch_field": "event_time_epoch",
        "timestamp_iso_field": "event_time_iso",
        "specialist_window_features": {
            "enabled": True,
            "version": 1,
            "schema_sha256": config.to_json_dict()["cross_context"][
                "specialist_window_features"
            ]["schema_sha256"],
            "scope_values_emitted": False,
            "absolute_timestamps_emitted": False,
        },
    }


def test_context_configuration_rejects_window_without_fixed_vocabulary(
    tmp_path,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--org-data",
                "logs.csv",
                "--org-groundtruth",
                "labels.csv",
                "--num-features",
                "10",
                "--federation-iterations",
                "1",
                "--output-dir",
                str(tmp_path),
                "--context-window-minutes",
                "10",
            ]
        )


def test_novel_domain_signal_uses_tagged_benign_training_baseline() -> None:
    prf_key = derive_prf_key(42)
    baseline = BenignNoveltyBaseline(
        benign_row_count=10,
        tagged_values={
            "domains": frozenset(
                {
                    hmac_sha256_tag(
                        prf_key,
                        "benign-baseline|domains|known.example",
                    )
                }
            ),
            "snis": frozenset(),
            "destination_ips": frozenset(),
        },
    )
    common = {
        "host": "host-01",
        "process_command_line": "powershell.exe -EncodedCommand SQBFAFgA",
    }

    known = cross_tokens_for_row(
        {**common, "domain": "known.example"},
        benign_novelty_baseline=baseline,
        prf_key=prf_key,
    )
    novel = cross_tokens_for_row(
        {**common, "domain": "novel.example"},
        benign_novelty_baseline=baseline,
        prf_key=prf_key,
    )

    token = "cross:encoded_command_and_first_seen_domain_same_host_15m"
    assert token not in known
    assert token in novel


def test_benign_novelty_baseline_uses_only_benign_training_rows() -> None:
    prf_key = derive_prf_key(42)
    dataset = SimpleNamespace(
        org_index=0,
        labels=["benign", "credential access", "benign"],
        logs_df=pd.DataFrame(
            {
                "domain": [
                    "known.example",
                    "attack-only.example",
                    "heldout-benign.example",
                ]
            }
        ),
    )
    split = SimpleNamespace(train_indices=[0, 1], test_indices=[2])

    baseline = build_benign_novelty_baselines(
        [dataset],
        [split],
        prf_key=prf_key,
    )[0]

    assert baseline.benign_row_count == 1
    assert baseline.tagged_values["domains"] == frozenset(
        {
            hmac_sha256_tag(
                prf_key,
                "benign-baseline|domains|known.example",
            )
        }
    )


def test_label_remnant_columns_cannot_activate_cross_signals() -> None:
    tokens = cross_tokens_for_row(
        {
            "host": "host-01",
            "label": "encoded command large upload",
            "sub_label": "encoded command large upload",
            "sub_label_cat": "encoded powershell followed by a large upload",
        }
    )

    assert tokens == []


def test_same_network_zone_correlates_distinct_endpoints_in_same_subnet() -> None:
    token = "cross:encoded_command_and_outbound_ssh_same_network_zone_15m"
    tokens = contextual_cross_tokens_for_rows(
        [
            {
                "event_time_iso": "2026-01-01T10:00:00Z",
                "src_ip": "10.20.30.10",
                "process_command_line": "powershell.exe -EncodedCommand SQBFAFgA",
            },
            {
                "event_time_iso": "2026-01-01T10:05:00Z",
                "src_ip": "10.20.30.200",
                "dst_ip": "203.0.113.20",
                "dst_port": "22",
                "network_direction": "outbound",
            },
        ]
    )

    assert token in tokens[1]


def test_same_network_zone_extracts_ip_from_endpoint_values() -> None:
    token = "cross:encoded_command_and_outbound_ssh_same_network_zone_15m"
    tokens = contextual_cross_tokens_for_rows(
        [
            {
                "event_time_iso": "2026-01-01T10:00:00Z",
                "endpoint": "10.20.30.10:51515",
                "process_command_line": "powershell.exe -EncodedCommand SQBFAFgA",
            },
            {
                "event_time_iso": "2026-01-01T10:05:00Z",
                "endpoint": "10.20.30.200:22",
                "dst_port": "22",
                "network_direction": "outbound",
            },
        ]
    )

    assert token in tokens[1]


def test_same_network_zone_extracts_bracketed_ipv6_endpoint_values() -> None:
    token = "cross:encoded_command_and_outbound_ssh_same_network_zone_15m"
    tokens = contextual_cross_tokens_for_rows(
        [
            {
                "event_time_iso": "2026-01-01T10:00:00Z",
                "source_endpoint": "[2001:db8:1::10]:51515",
                "process_command_line": "powershell.exe -EncodedCommand SQBFAFgA",
            },
            {
                "event_time_iso": "2026-01-01T10:05:00Z",
                "destination_endpoint": "[2001:db8:1::20]:22",
                "dst_port": "22",
                "network_direction": "outbound",
            },
        ]
    )

    assert token in tokens[1]


def test_specialist_context_summarizes_causal_network_window_without_identifiers() -> None:
    rows = [
        {
            "event_time_iso": "2026-01-01T10:00:00Z",
            "src_ip": "10.0.0.10",
            "dst_ip": "10.0.1.20",
            "dst_port": "443",
            "protocol_name": "tcp",
            "total_size": "100",
            "packet_number": "2",
        },
        {
            "event_time_iso": "2026-01-01T10:05:00Z",
            "src_ip": "10.0.0.10",
            "dst_ip": "10.0.1.21",
            "dst_port": "80",
            "protocol_name": "tcp",
            "total_size": "200",
            "packet_number": "3",
        },
    ]

    contextual = contextual_feature_tokens_for_rows(rows)
    second = contextual.specialist_tokens["network"][1]

    assert "context:same_src_ip:event_count_15m=b2_le_3" in second
    assert "context:same_src_ip:distinct_destination_count_15m=b2_le_3" in second
    assert "context:same_src_ip:total_size_15m=b5_le_1023" in second
    assert "context:same_src_ip:interarrival_seconds_15m=b7_le_300" in second
    assert not any("10.0.0.10" in token for token in second)
    assert not any("2026-01-01" in token for token in second)


def test_specialist_context_window_boundary_is_inclusive() -> None:
    row = {
        "src_ip": "10.0.0.10",
        "dst_ip": "10.0.1.20",
        "dst_port": "443",
    }
    inclusive = contextual_feature_tokens_for_rows(
        [
            {**row, "event_time_iso": "2026-01-01T10:00:00Z"},
            {**row, "event_time_iso": "2026-01-01T10:15:00Z"},
        ]
    )
    expired = contextual_feature_tokens_for_rows(
        [
            {**row, "event_time_iso": "2026-01-01T10:00:00Z"},
            {**row, "event_time_iso": "2026-01-01T10:15:01Z"},
        ]
    )

    assert (
        "context:same_src_ip:event_count_15m=b2_le_3"
        in inclusive.specialist_tokens["network"][1]
    )
    assert (
        "context:same_src_ip:event_count_15m=b1_le_1"
        in expired.specialist_tokens["network"][1]
    )


def test_specialist_context_does_not_cross_entity_scopes() -> None:
    contextual = contextual_feature_tokens_for_rows(
        [
            {
                "event_time_iso": "2026-01-01T10:00:00Z",
                "src_ip": "10.0.0.10",
                "dst_ip": "10.0.1.20",
            },
            {
                "event_time_iso": "2026-01-01T10:01:00Z",
                "src_ip": "10.0.2.10",
                "dst_ip": "10.0.3.20",
            },
        ]
    )

    assert (
        "context:same_src_ip:event_count_15m=b1_le_1"
        in contextual.specialist_tokens["network"][1]
    )


def test_specialist_context_captures_failed_then_successful_login() -> None:
    contextual = contextual_feature_tokens_for_rows(
        [
            {
                **_failed("2026-01-01T10:00:00Z"),
                "process_name": "login",
            },
            {
                **_success("2026-01-01T10:05:00Z"),
                "process_name": "login",
            },
        ]
    )

    token = "context:same_user:failed_then_success_15m=true"
    assert token in contextual.specialist_tokens["system"][1]
    assert token in contextual.specialist_tokens["identity"][1]


def test_missing_timestamp_does_not_create_specialist_context() -> None:
    contextual = contextual_feature_tokens_for_rows(
        [{"src_ip": "10.0.0.10", "dst_ip": "10.0.1.20"}]
    )

    assert contextual.specialist_tokens["network"] == [[]]


def test_subcategory_texts_append_context_to_native_specialist_view() -> None:
    dataset = SimpleNamespace(
        org_index=0,
        labels=["benign", "benign"],
        logs_df=pd.DataFrame(
            [
                {
                    "event_time_iso": "2026-01-01T10:00:00Z",
                    "src_ip": "10.0.0.10",
                    "dst_ip": "10.0.1.20",
                    "dst_port": "443",
                },
                {
                    "event_time_iso": "2026-01-01T10:01:00Z",
                    "src_ip": "10.0.0.10",
                    "dst_ip": "10.0.1.21",
                    "dst_port": "80",
                },
            ]
        ),
    )

    texts, _ = build_subcategory_texts(
        [dataset],
        subcategories=["network"],
        context_window_minutes=15.0,
    )

    assert "src_ip=10.0.0.10" in texts["network"][0][1]
    assert "context:same_src_ip:event_count_15m=b2_le_3" in texts["network"][0][1]
