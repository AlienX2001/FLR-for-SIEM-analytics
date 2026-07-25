from __future__ import annotations

import pytest

import policy_filter.matcher as matcher
from policy_filter.matcher import decide
from policy_filter.preprocessing_adapter import preprocess_policy_row
from policy_filter.schema import PolicyValidationError, load_policy
from policy_filter_helpers import network_policy
from policy_filter_helpers import base_policy, decision_for, system_row, write_policy


def combined_policy() -> dict:
    payload = base_policy()
    payload["network_policies"] = network_policy()["network_policies"]
    return payload


def combined_row(**overrides: str) -> dict[str, str]:
    row = system_row()
    row.update(
        {
            "src_ip": "10.0.0.10",
            "dst_ip": "203.0.113.20",
            "protocol": "tcp",
            "dst_port": "443",
            "direction": "outbound",
        }
    )
    row.update(overrides)
    return row


def test_combined_row_suppressed_only_when_system_and_network_allow(tmp_path) -> None:
    assert decision_for(tmp_path, combined_row(), combined_policy()).action == "suppress"
    assert decision_for(
        tmp_path,
        combined_row(dst_ip="198.51.101.99"),
        combined_policy(),
    ).reason_code == "UNAUTHORIZED_REMOTE_IP"
    assert decision_for(
        tmp_path,
        combined_row(user="mallory"),
        combined_policy(),
    ).reason_code == "UNAUTHORIZED_USER"


def test_conflicting_aliases_forward(tmp_path) -> None:
    payload = base_policy()
    payload["field_mappings"]["user"] = ["user", "account"]
    row = system_row()
    row["account"] = "mallory"

    decision = decision_for(tmp_path, row, payload)

    assert decision.action == "forward"
    assert decision.reason_code == "CONFLICTING_FIELD_VALUES"


def test_unknown_category_and_no_policy_forward(tmp_path) -> None:
    payload = base_policy()
    payload["category_aliases"] = {"system": ["system"], "network": ["network"]}
    assert decision_for(tmp_path, {"event_category": "mystery"}, payload).reason_code == "UNKNOWN_CATEGORY"

    payload = base_policy()
    payload["system_policies"] = []
    assert decision_for(tmp_path, system_row(), payload).reason_code == "NO_APPLICABLE_POLICY"


def test_unexpected_matching_exception_fails_open(tmp_path, monkeypatch) -> None:
    policy = load_policy(write_policy(tmp_path, base_policy()))
    row = preprocess_policy_row(system_row(), policy)

    def explode(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("boom")

    monkeypatch.setattr(matcher, "evaluate_system", explode)

    decision = decide(row, policy)

    assert decision.action == "forward"
    assert decision.reason_code == "INTERNAL_FILTER_ERROR"


def test_invalid_policy_rejected_before_logs(tmp_path) -> None:
    payload = base_policy()
    payload["preprocessing"] = {"unsupported": True}

    with pytest.raises(PolicyValidationError, match="unsupported preprocessing option"):
        load_policy(write_policy(tmp_path, payload))
