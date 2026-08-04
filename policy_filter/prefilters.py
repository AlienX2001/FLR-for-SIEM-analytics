from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any

from policy_filter import reasons
from policy_filter.models import (
    EventIdPrefilterPolicy,
    FilterDecision,
    PolicyDocument,
    PrefilterDecision,
    SeverityPrefilterPolicy,
)

EVENT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


class PrefilterValueError(ValueError):
    pass


def canonical_event_id(value: Any) -> str:
    if value is None:
        raise PrefilterValueError("missing Event ID")
    if isinstance(value, bool):
        raise PrefilterValueError("malformed Event ID")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise PrefilterValueError("malformed Event ID")
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise PrefilterValueError("missing Event ID")
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None and math.isfinite(number) and number.is_integer() and re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        return str(int(number))
    if not EVENT_ID_RE.fullmatch(text):
        raise PrefilterValueError("malformed Event ID")
    return text


def canonical_severity_value(value: Any, *, case_insensitive: bool) -> str:
    if value is None:
        raise PrefilterValueError("missing severity")
    text = str(value).strip()
    if not text:
        raise PrefilterValueError("missing severity")
    if any(ord(character) < 32 for character in text):
        raise PrefilterValueError("malformed severity")
    return text.lower() if case_insensitive else text


def evaluate_event_id_prefilter(
    raw_row: dict[str, str],
    *,
    field_name: str | None,
    policy: EventIdPrefilterPolicy | None,
) -> PrefilterDecision | None:
    if field_name is None:
        return None
    if field_name not in raw_row:
        return PrefilterDecision(
            must_forward=True,
            suppress_candidate=False,
            reason_codes=(reasons.EVENT_ID_MISSING,),
            reason_details=(f"Event-ID field {field_name!r} is absent",),
        )
    try:
        normalized = canonical_event_id(raw_row.get(field_name))
    except PrefilterValueError as exc:
        reason = reasons.EVENT_ID_MISSING if "missing" in str(exc) else reasons.EVENT_ID_MALFORMED
        return PrefilterDecision(
            must_forward=True,
            suppress_candidate=False,
            reason_codes=(reason,),
            reason_details=(f"{reason}: {exc}",),
        )
    assert policy is not None
    if normalized in policy.always_forward_ids:
        return PrefilterDecision(
            must_forward=True,
            suppress_candidate=False,
            reason_codes=(reasons.EVENT_ID_ALWAYS_FORWARD,),
            reason_details=(f"Event ID {normalized} is configured to always forward",),
            normalized_value=normalized,
            require_policy_match_for_suppression=policy.require_policy_match_for_suppression,
        )
    if normalized in policy.suppress_ids:
        return PrefilterDecision(
            must_forward=False,
            suppress_candidate=True,
            reason_codes=(reasons.EVENT_ID_SUPPRESS_MATCH,),
            reason_details=(f"Event ID {normalized} matched suppress_ids",),
            normalized_value=normalized,
            require_policy_match_for_suppression=policy.require_policy_match_for_suppression,
        )
    return PrefilterDecision(
        must_forward=False,
        suppress_candidate=False,
        reason_codes=(reasons.EVENT_ID_NEUTRAL,),
        reason_details=(f"Event ID {normalized} did not match prefilter lists",),
        normalized_value=normalized,
        require_policy_match_for_suppression=policy.require_policy_match_for_suppression,
    )


def evaluate_severity_prefilter(
    raw_row: dict[str, str],
    *,
    field_name: str | None,
    policy: SeverityPrefilterPolicy | None,
) -> PrefilterDecision | None:
    if field_name is None:
        return None
    if field_name not in raw_row:
        return PrefilterDecision(
            must_forward=True,
            suppress_candidate=False,
            reason_codes=(reasons.SEVERITY_MISSING,),
            reason_details=(f"severity field {field_name!r} is absent",),
        )
    assert policy is not None
    try:
        normalized = canonical_severity_value(
            raw_row.get(field_name),
            case_insensitive=policy.case_insensitive,
        )
    except PrefilterValueError as exc:
        reason = reasons.SEVERITY_MISSING if "missing" in str(exc) else reasons.SEVERITY_MALFORMED
        return PrefilterDecision(
            must_forward=True,
            suppress_candidate=False,
            reason_codes=(reason,),
            reason_details=(f"{reason}: {exc}",),
        )
    if normalized in policy.always_forward_values:
        return PrefilterDecision(
            must_forward=True,
            suppress_candidate=False,
            reason_codes=(reasons.SEVERITY_ALWAYS_FORWARD,),
            reason_details=(f"severity {normalized!r} is configured to always forward",),
            normalized_value=normalized,
            require_policy_match_for_suppression=policy.require_policy_match_for_suppression,
        )
    if normalized in policy.suppress_values:
        return PrefilterDecision(
            must_forward=False,
            suppress_candidate=True,
            reason_codes=(reasons.SEVERITY_SUPPRESS_MATCH,),
            reason_details=(f"severity {normalized!r} matched suppress_values",),
            normalized_value=normalized,
            require_policy_match_for_suppression=policy.require_policy_match_for_suppression,
        )
    return PrefilterDecision(
        must_forward=False,
        suppress_candidate=False,
        reason_codes=(reasons.SEVERITY_NEUTRAL,),
        reason_details=(f"severity {normalized!r} did not match prefilter lists",),
        normalized_value=normalized,
        require_policy_match_for_suppression=policy.require_policy_match_for_suppression,
    )


def enabled_prefilter_decisions(
    raw_row: dict[str, str],
    policy: PolicyDocument,
    *,
    event_id_field: str | None,
    severity_field: str | None,
) -> tuple[PrefilterDecision, ...]:
    decisions = []
    event_decision = evaluate_event_id_prefilter(
        raw_row,
        field_name=event_id_field,
        policy=policy.prefilters.event_id,
    )
    if event_decision is not None:
        decisions.append(event_decision)
    severity_decision = evaluate_severity_prefilter(
        raw_row,
        field_name=severity_field,
        policy=policy.prefilters.severity,
    )
    if severity_decision is not None:
        decisions.append(severity_decision)
    return tuple(decisions)


def combine_prefilters_with_policy_decision(
    base_decision: FilterDecision,
    prefilter_decisions: tuple[PrefilterDecision, ...],
) -> FilterDecision:
    if not prefilter_decisions:
        return base_decision

    detail_parts = [base_decision.reason_details]
    for decision in prefilter_decisions:
        detail_parts.extend(decision.reason_details)
    combined_details = "; ".join(part for part in detail_parts if part)

    for decision in prefilter_decisions:
        if decision.must_forward:
            return FilterDecision(
                "forward",
                base_decision.category,
                decision.reason_codes[0],
                combined_details,
                base_decision.matched_policy_id,
            )

    suppress_candidates = [
        decision for decision in prefilter_decisions if decision.suppress_candidate
    ]
    if not suppress_candidates:
        return replace(base_decision, reason_details=combined_details)

    if base_decision.action == "suppress":
        return replace(base_decision, reason_details=combined_details)

    all_allow_without_policy = all(
        not decision.require_policy_match_for_suppression
        for decision in suppress_candidates
    )
    if all_allow_without_policy and base_decision.reason_code == reasons.NO_APPLICABLE_POLICY:
        return FilterDecision(
            "suppress",
            base_decision.category,
            "ALLOWED",
            combined_details,
            base_decision.matched_policy_id,
        )

    return replace(base_decision, reason_details=combined_details)
