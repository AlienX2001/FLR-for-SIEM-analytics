from __future__ import annotations

import pytest

from federated_lr_pipeline.config import parse_args
from federated_lr_pipeline.specialized_models import contextual_cross_tokens_for_rows


FAILED_THEN_SUCCESS_TOKEN = (
    "cross:failed_login_burst_AND_successful_login_same_user_15m"
)


def _failed(timestamp: str | None, *, user: str = "alice") -> dict[str, str]:
    row = {"user_uid": user, "sub_label": "failed_login"}
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
