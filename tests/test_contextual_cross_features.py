from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from federated_lr_pipeline.config import parse_args
from federated_lr_pipeline.prf import derive_prf_key, hmac_sha256_tag
from federated_lr_pipeline.specialized_models import (
    BenignNoveltyBaseline,
    build_benign_novelty_baselines,
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
        "version": 1,
        "causal": True,
        "window_minutes": 15.0,
        "timestamp_epoch_field": "event_time_epoch",
        "timestamp_iso_field": "event_time_iso",
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
