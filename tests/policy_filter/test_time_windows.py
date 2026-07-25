from __future__ import annotations

from policy_filter_helpers import base_policy, decision_for, system_row


def test_overnight_authorized_time_window(tmp_path) -> None:
    payload = base_policy()
    payload["system_policies"][0]["authorized_identities"][0]["authorized_time_windows"] = [
        {
            "days": ["Monday"],
            "start": "22:00",
            "end": "06:00",
            "timezone": "UTC",
        }
    ]

    assert decision_for(
        tmp_path,
        system_row(timestamp="2026-07-20T23:30:00Z"),
        payload,
    ).action == "suppress"
    assert decision_for(
        tmp_path,
        system_row(timestamp="2026-07-21T05:30:00Z"),
        payload,
    ).action == "suppress"
    assert decision_for(
        tmp_path,
        system_row(timestamp="2026-07-21T12:00:00Z"),
        payload,
    ).reason_code == "OUTSIDE_AUTHORIZED_TIME"
