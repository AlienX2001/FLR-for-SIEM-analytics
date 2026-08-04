from __future__ import annotations

from pathlib import Path

import pytest

from policy_filter.matcher import decide
from policy_filter.prefilters import (
    combine_prefilters_with_policy_decision,
    evaluate_event_id_prefilter,
    evaluate_severity_prefilter,
)
from policy_filter.preprocessing_adapter import preprocess_policy_row
from policy_filter.schema import PolicyValidationError, load_policy
from policy_filter_helpers import base_policy, system_row, write_policy


def _policy_with_prefilters() -> dict:
    payload = base_policy()
    payload["prefilters"] = {
        "event_id": {
            "suppress_ids": [1000, "4624"],
            "always_forward_ids": [1102, "4688"],
            "require_policy_match_for_suppression": True,
        },
        "severity": {
            "suppress_values": ["debug", "info"],
            "always_forward_values": ["warning", "critical"],
            "case_insensitive": True,
            "require_policy_match_for_suppression": True,
        },
    }
    return payload


def _decision(
    tmp_path: Path,
    row: dict[str, str],
    *,
    event_id_field: str | None = None,
    severity_field: str | None = None,
    payload: dict | None = None,
):
    policy = load_policy(write_policy(tmp_path, payload or _policy_with_prefilters()))
    base = decide(preprocess_policy_row(row, policy), policy)
    event_decision = (
        evaluate_event_id_prefilter(
            row,
            field_name=event_id_field,
            policy=policy.prefilters.event_id,
        )
        if event_id_field is not None
        else None
    )
    severity_decision = (
        evaluate_severity_prefilter(
            row,
            field_name=severity_field,
            policy=policy.prefilters.severity,
        )
        if severity_field is not None
        else None
    )
    return combine_prefilters_with_policy_decision(
        base,
        tuple(item for item in (event_decision, severity_decision) if item is not None),
    )


def test_event_id_path_is_not_executed_when_field_is_omitted(tmp_path: Path) -> None:
    policy = base_policy()
    row = system_row(eventid="1102")

    decision = _decision(tmp_path, row, payload=policy)

    assert decision.action == "suppress"


def test_event_id_exact_matches_and_canonical_strings(tmp_path: Path) -> None:
    policy = load_policy(write_policy(tmp_path, _policy_with_prefilters()))

    suppress = evaluate_event_id_prefilter(
        system_row(eventid="4624"),
        field_name="eventid",
        policy=policy.prefilters.event_id,
    )
    forward = evaluate_event_id_prefilter(
        system_row(eventid="4688"),
        field_name="eventid",
        policy=policy.prefilters.event_id,
    )
    neutral = evaluate_event_id_prefilter(
        system_row(eventid="9999"),
        field_name="eventid",
        policy=policy.prefilters.event_id,
    )

    assert suppress.suppress_candidate is True
    assert suppress.normalized_value == "4624"
    assert forward.must_forward is True
    assert neutral.is_neutral is True


def test_event_id_always_forward_and_missing_or_malformed_fail_open(tmp_path: Path) -> None:
    assert _decision(tmp_path, system_row(eventid="1102"), event_id_field="eventid").reason_code == "EVENT_ID_ALWAYS_FORWARD"
    assert _decision(tmp_path, system_row(), event_id_field="eventid").reason_code == "EVENT_ID_MISSING"
    assert _decision(tmp_path, system_row(eventid="46 24"), event_id_field="eventid").reason_code == "EVENT_ID_MALFORMED"


def test_event_id_policy_mismatch_overrides_suppress_candidate(tmp_path: Path) -> None:
    decision = _decision(
        tmp_path,
        system_row(user="mallory", eventid="4624"),
        event_id_field="eventid",
    )

    assert decision.action == "forward"
    assert decision.reason_code == "UNAUTHORIZED_USER"


def test_prefilter_can_suppress_without_policy_only_when_explicitly_configured(tmp_path: Path) -> None:
    payload = _policy_with_prefilters()
    payload["system_policies"] = []
    payload["network_policies"] = []
    payload["prefilters"]["event_id"]["require_policy_match_for_suppression"] = False

    decision = _decision(
        tmp_path,
        {"src_ip": "10.1.1.20", "dst_ip": "10.1.1.5", "eventid": "4624"},
        event_id_field="eventid",
        payload=payload,
    )

    assert decision.action == "suppress"


def test_event_id_overlap_fails_validation(tmp_path: Path) -> None:
    payload = _policy_with_prefilters()
    payload["prefilters"]["event_id"]["always_forward_ids"].append("1000")

    with pytest.raises(PolicyValidationError, match="both suppressed and always-forwarded"):
        load_policy(write_policy(tmp_path, payload))


def test_severity_path_is_not_executed_when_field_is_omitted(tmp_path: Path) -> None:
    decision = _decision(tmp_path, system_row(severity="critical"), payload=base_policy())

    assert decision.action == "suppress"


def test_severity_case_insensitive_exact_matching(tmp_path: Path) -> None:
    low = _decision(tmp_path, system_row(severity="INFO"), severity_field="severity")
    high = _decision(tmp_path, system_row(severity="Critical"), severity_field="severity")
    neutral = _decision(tmp_path, system_row(severity="medium"), severity_field="severity")

    assert low.action == "suppress"
    assert high.reason_code == "SEVERITY_ALWAYS_FORWARD"
    assert neutral.action == "suppress"


def test_severity_missing_and_policy_mismatch_fail_open(tmp_path: Path) -> None:
    assert _decision(tmp_path, system_row(), severity_field="severity").reason_code == "SEVERITY_MISSING"
    decision = _decision(
        tmp_path,
        system_row(user="mallory", severity="debug"),
        severity_field="severity",
    )

    assert decision.action == "forward"
    assert decision.reason_code == "UNAUTHORIZED_USER"


def test_severity_overlap_fails_validation(tmp_path: Path) -> None:
    payload = _policy_with_prefilters()
    payload["prefilters"]["severity"]["always_forward_values"].append("INFO")

    with pytest.raises(PolicyValidationError, match="both suppressed and always-forwarded"):
        load_policy(write_policy(tmp_path, payload))
