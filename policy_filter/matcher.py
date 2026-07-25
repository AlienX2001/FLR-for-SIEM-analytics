from __future__ import annotations

import ipaddress
import logging
from datetime import datetime, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

from federated_lr_pipeline.specialized_models import _field_value

from policy_filter import reasons
from policy_filter.models import (
    AllowedProgram,
    AuthorizedConnection,
    AuthorizedIdentity,
    FilterDecision,
    MatchResult,
    NetworkPolicy,
    PolicyDocument,
    PreprocessedPolicyRow,
    SystemPolicy,
    TimeWindow,
)

LOGGER = logging.getLogger(__name__)


def _field(row: PreprocessedPolicyRow, name: str) -> str | None:
    value = row.canonical_fields.get(name)
    return value if value else None


def _contains_any(value: str | None, options: Iterable[str]) -> bool:
    if value is None:
        return False
    normalized = value.lower()
    return any(normalized == option.lower() for option in options)


def _parse_ip(value: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not value:
        return None
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _parse_port(value: str | None) -> int | None:
    if not value:
        return None
    try:
        port = int(value)
    except ValueError:
        return None
    return port if 0 <= port <= 65535 else None


def _parse_event_time(
    value: str | None,
    default_timezone: str,
) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("z", "+00:00").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(default_timezone))
    return parsed.astimezone(timezone.utc)


def _policy_date_valid(
    timestamp: datetime | None,
    valid_from: datetime | None,
    valid_until: datetime | None,
) -> bool:
    if valid_from is None and valid_until is None:
        return True
    if timestamp is None:
        return False
    if valid_from is not None and timestamp < valid_from:
        return False
    if valid_until is not None and timestamp > valid_until:
        return False
    return True


def _window_contains(timestamp: datetime, window: TimeWindow) -> bool:
    local = timestamp.astimezone(ZoneInfo(window.timezone))
    minutes = local.hour * 60 + local.minute
    if window.start_minutes <= window.end_minutes:
        return local.weekday() in window.days and window.start_minutes <= minutes <= window.end_minutes
    if minutes >= window.start_minutes and local.weekday() in window.days:
        return True
    previous_weekday = (local.weekday() - 1) % 7
    return minutes <= window.end_minutes and previous_weekday in window.days


def _timestamp_in_windows(
    row: PreprocessedPolicyRow,
    windows: tuple[TimeWindow, ...],
    default_timezone: str,
) -> tuple[bool, str | None]:
    if not windows:
        return True, None
    timestamp = _parse_event_time(_field(row, "timestamp"), default_timezone)
    if timestamp is None:
        return False, reasons.INVALID_TIMESTAMP
    if any(_window_contains(timestamp, window) for window in windows):
        return True, None
    return False, reasons.OUTSIDE_AUTHORIZED_TIME


def _user_groups(value: str | None) -> set[str]:
    if not value:
        return set()
    return {
        item.strip().lower()
        for item in value.replace(";", ",").split(",")
        if item.strip()
    }


def _program_name_from_path(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", maxsplit=1)[-1].lower()


def _path_equal(observed: str | None, allowed: str) -> bool:
    if observed is None:
        return False
    return (
        observed.replace("\\", "/").rstrip("/").lower()
        == _field_value(allowed).replace("\\", "/").rstrip("/").lower()
    )


def _program_matches(program: AllowedProgram, row: PreprocessedPolicyRow) -> MatchResult:
    observed_program = _field(row, "program")
    observed_path = _field(row, "program_path")
    observed_parent = _field(row, "parent_program")
    command_line = _field(row, "command_line") or ""
    program_candidates = {
        item
        for item in (observed_program, _program_name_from_path(observed_path))
        if item
    }
    if program.name.lower() not in {candidate.lower() for candidate in program_candidates}:
        return MatchResult(
            "NOT_ALLOWED",
            reasons.UNAUTHORIZED_PROGRAM,
            f"program {observed_program or observed_path or '<missing>'} is not allowed",
        )
    if program.paths and observed_path is None:
        return MatchResult("NOT_ALLOWED", reasons.MISSING_REQUIRED_FIELD, "program_path is required")
    if program.paths and not any(_path_equal(observed_path, allowed) for allowed in program.paths):
        return MatchResult(
            "NOT_ALLOWED",
            reasons.UNAUTHORIZED_PROGRAM_PATH,
            "program path is not authorized",
        )
    if program.allowed_parent_programs and observed_parent is None:
        return MatchResult("NOT_ALLOWED", reasons.MISSING_REQUIRED_FIELD, "parent_program is required")
    if program.allowed_parent_programs and not _contains_any(observed_parent, program.allowed_parent_programs):
        return MatchResult(
            "NOT_ALLOWED",
            reasons.UNAUTHORIZED_PARENT_PROGRAM,
            "parent program is not authorized",
        )
    if program.command_line is not None:
        allowed_patterns = program.command_line.allowed_patterns
        prohibited_patterns = program.command_line.prohibited_patterns
        if allowed_patterns and not any(pattern.search(command_line) for pattern in allowed_patterns):
            return MatchResult(
                "NOT_ALLOWED",
                reasons.COMMAND_LINE_POLICY_VIOLATION,
                "command line did not match any allowed pattern",
            )
        if any(pattern.search(command_line) for pattern in prohibited_patterns):
            return MatchResult(
                "NOT_ALLOWED",
                reasons.COMMAND_LINE_POLICY_VIOLATION,
                "command line matched a prohibited pattern",
            )
    return MatchResult("ALLOWED", "ALLOWED", "program allowed")


def _system_policy_applies(policy: SystemPolicy, row: PreprocessedPolicyRow) -> tuple[bool, str | None]:
    host = _field(row, "host")
    local_ip = _parse_ip(_field(row, "local_ip") or _field(row, "source_ip"))
    if policy.hosts and host is not None and _contains_any(host, policy.hosts):
        return True, None
    if policy.local_ips and local_ip is not None and local_ip in policy.local_ips:
        return True, None
    if policy.hosts and host is None and not policy.local_ips:
        return False, reasons.MISSING_REQUIRED_FIELD
    if policy.local_ips and local_ip is None and not policy.hosts:
        return False, reasons.MISSING_REQUIRED_FIELD
    if policy.hosts and host is None and policy.local_ips and local_ip is None:
        return False, reasons.MISSING_REQUIRED_FIELD
    if policy.hosts and host is not None:
        return False, reasons.UNAUTHORIZED_HOST
    if policy.local_ips and local_ip is not None:
        return False, reasons.UNAUTHORIZED_LOCAL_IP
    return False, reasons.NO_APPLICABLE_POLICY


def _identity_matches(identity: AuthorizedIdentity, row: PreprocessedPolicyRow) -> bool:
    user = _field(row, "user")
    groups = _user_groups(_field(row, "user_groups"))
    if user is not None and _contains_any(user, identity.users):
        return True
    return bool(groups & {group.lower() for group in identity.groups})


def _choose_failure(
    failures: list[MatchResult],
    priority: tuple[str, ...],
) -> MatchResult | None:
    if not failures:
        return None
    for reason_code in priority:
        for failure in failures:
            if failure.reason_code == reason_code:
                return failure
    return failures[0]


def evaluate_system(row: PreprocessedPolicyRow, policy: PolicyDocument) -> MatchResult:
    timestamp = _parse_event_time(_field(row, "timestamp"), policy.default_timezone)
    best_failure: MatchResult | None = None
    applicable_found = False
    for system_policy in policy.system_policies:
        if not system_policy.enabled:
            continue
        applies, apply_reason = _system_policy_applies(system_policy, row)
        if not applies:
            if best_failure is None and apply_reason != reasons.NO_APPLICABLE_POLICY:
                best_failure = MatchResult(
                    "NOT_ALLOWED",
                    apply_reason or reasons.SYSTEM_POLICY_MISMATCH,
                    "system policy selector did not match",
                    system_policy.policy_id,
                )
            continue
        applicable_found = True
        if not _policy_date_valid(timestamp, system_policy.valid_from, system_policy.valid_until):
            best_failure = MatchResult(
                "NOT_ALLOWED",
                reasons.OUTSIDE_AUTHORIZED_TIME if timestamp else reasons.INVALID_TIMESTAMP,
                "system policy validity dates do not include the event timestamp",
                system_policy.policy_id,
            )
            continue
        matching_identities = [
            identity for identity in system_policy.authorized_identities if _identity_matches(identity, row)
        ]
        if not matching_identities:
            best_failure = MatchResult(
                "NOT_ALLOWED",
                reasons.UNAUTHORIZED_USER,
                "no authorized identity matched user or groups",
                system_policy.policy_id,
            )
            continue
        for identity in matching_identities:
            in_time, time_reason = _timestamp_in_windows(
                row,
                identity.authorized_time_windows,
                policy.default_timezone,
            )
            if not in_time:
                best_failure = MatchResult(
                    "NOT_ALLOWED",
                    time_reason or reasons.OUTSIDE_AUTHORIZED_TIME,
                    "identity time window did not authorize the event",
                    system_policy.policy_id,
                )
                continue
            program_failures = []
            for program in identity.allowed_programs:
                result = _program_matches(program, row)
                if result.state == "ALLOWED":
                    return MatchResult(
                        "ALLOWED",
                        "ALLOWED",
                        "system row fully matched allow-list policy",
                        system_policy.policy_id,
                    )
                program_failures.append(result)
            if program_failures:
                chosen = _choose_failure(
                    program_failures,
                    (
                        reasons.MISSING_REQUIRED_FIELD,
                        reasons.UNAUTHORIZED_PROGRAM_PATH,
                        reasons.UNAUTHORIZED_PARENT_PROGRAM,
                        reasons.COMMAND_LINE_POLICY_VIOLATION,
                        reasons.UNAUTHORIZED_PROGRAM,
                    ),
                )
                assert chosen is not None
                best_failure = MatchResult(
                    "NOT_ALLOWED",
                    chosen.reason_code,
                    chosen.reason_details,
                    system_policy.policy_id,
                )
    if applicable_found and best_failure is not None:
        return best_failure
    if best_failure is not None:
        return best_failure
    return MatchResult(
        "NOT_APPLICABLE",
        reasons.NO_APPLICABLE_POLICY,
        "no enabled system policy applies",
    )


def _domain_normalize(value: str | None) -> str | None:
    if not value:
        return None
    domain = value.strip().rstrip(".").lower()
    return domain or None


def _domain_matches(domain: str | None, patterns: tuple[str, ...]) -> bool:
    normalized = _domain_normalize(domain)
    if normalized is None:
        return False
    for pattern in patterns:
        if pattern.startswith("*."):
            suffix = pattern[2:]
            if normalized.endswith("." + suffix) and normalized != suffix:
                return True
        elif normalized == pattern:
            return True
    return False


def _network_selector_applies(policy: NetworkPolicy, row: PreprocessedPolicyRow) -> tuple[bool, str | None]:
    host = _field(row, "host")
    local_ip = _parse_ip(_field(row, "local_ip") or _field(row, "source_ip"))
    if policy.local_hosts and host is not None and _contains_any(host, policy.local_hosts):
        return True, None
    if policy.local_ips and local_ip is not None and local_ip in policy.local_ips:
        return True, None
    if policy.local_networks and local_ip is not None and any(local_ip in network for network in policy.local_networks):
        return True, None
    if policy.local_hosts and host is not None:
        return False, reasons.UNAUTHORIZED_HOST
    if (policy.local_ips or policy.local_networks) and local_ip is not None:
        return False, reasons.UNAUTHORIZED_LOCAL_IP
    if policy.local_hosts or policy.local_ips or policy.local_networks:
        return False, reasons.MISSING_REQUIRED_FIELD
    return False, reasons.NO_APPLICABLE_POLICY


def _derive_direction(policy: NetworkPolicy, row: PreprocessedPolicyRow) -> tuple[str | None, str | None]:
    explicit = _field(row, "direction")
    if explicit:
        lowered = explicit.lower()
        if lowered in {"inbound", "outbound", "internal"}:
            return lowered, None
        return None, None
    source = _parse_ip(_field(row, "source_ip"))
    destination = _parse_ip(_field(row, "destination_ip"))
    if source is None or destination is None or not policy.local_networks:
        return None, None
    source_local = any(source in network for network in policy.local_networks)
    destination_local = any(destination in network for network in policy.local_networks)
    if source_local and destination_local:
        return "internal", "derived direction as internal from local_networks"
    if source_local and not destination_local:
        return "outbound", "derived direction as outbound from local_networks"
    if destination_local and not source_local:
        return "inbound", "derived direction as inbound from local_networks"
    return None, None


def _remote_ip_for_direction(direction: str | None, row: PreprocessedPolicyRow) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if direction == "inbound":
        return _parse_ip(_field(row, "source_ip"))
    return _parse_ip(_field(row, "destination_ip"))


def _connection_matches(
    connection: AuthorizedConnection,
    direction: str | None,
    row: PreprocessedPolicyRow,
    policy: PolicyDocument,
) -> MatchResult:
    if direction is None:
        return MatchResult("NOT_ALLOWED", reasons.UNKNOWN_NETWORK_DIRECTION, "direction is missing or ambiguous")
    if connection.direction != "any" and connection.direction != direction:
        return MatchResult("NOT_ALLOWED", reasons.UNKNOWN_NETWORK_DIRECTION, "direction did not match")
    remote_ip = _remote_ip_for_direction(direction, row)
    remote_domain = _field(row, "domain")
    remote_matched = False
    if remote_ip is not None:
        remote_matched = remote_ip in connection.remote_ips or any(
            remote_ip in network for network in connection.remote_cidrs
        )
    if not remote_matched and connection.remote_domains:
        remote_matched = _domain_matches(remote_domain, connection.remote_domains)
    if not remote_matched:
        if connection.remote_domains and remote_domain:
            return MatchResult("NOT_ALLOWED", reasons.UNAUTHORIZED_REMOTE_DOMAIN, "remote domain is not authorized")
        if connection.remote_domains and remote_domain is None and remote_ip is None:
            return MatchResult("NOT_ALLOWED", reasons.MISSING_REQUIRED_FIELD, "remote domain or IP is required")
        return MatchResult("NOT_ALLOWED", reasons.UNAUTHORIZED_REMOTE_IP, "remote IP is not authorized")
    if connection.protocols:
        protocol = _field(row, "protocol")
        if protocol is None:
            return MatchResult("NOT_ALLOWED", reasons.MISSING_REQUIRED_FIELD, "protocol is required")
        if protocol.lower() not in connection.protocols:
            return MatchResult("NOT_ALLOWED", reasons.UNAUTHORIZED_PROTOCOL, "protocol is not authorized")
    if connection.destination_ports:
        port = _parse_port(_field(row, "destination_port"))
        if port is None:
            return MatchResult("NOT_ALLOWED", reasons.MISSING_REQUIRED_FIELD, "destination_port is required")
        if not any(port_matcher.contains(port) for port_matcher in connection.destination_ports):
            return MatchResult("NOT_ALLOWED", reasons.UNAUTHORIZED_PORT, "destination port is not authorized")
    in_time, time_reason = _timestamp_in_windows(row, connection.authorized_time_windows, policy.default_timezone)
    if not in_time:
        return MatchResult(
            "NOT_ALLOWED",
            time_reason or reasons.OUTSIDE_AUTHORIZED_TIME,
            "connection time window did not authorize the event",
        )
    return MatchResult("ALLOWED", "ALLOWED", "network connection fully matched allow-list policy")


def evaluate_network(row: PreprocessedPolicyRow, policy: PolicyDocument) -> MatchResult:
    timestamp = _parse_event_time(_field(row, "timestamp"), policy.default_timezone)
    best_failure: MatchResult | None = None
    applicable_found = False
    for network_policy in policy.network_policies:
        if not network_policy.enabled:
            continue
        applies, apply_reason = _network_selector_applies(network_policy, row)
        if not applies:
            if best_failure is None and apply_reason != reasons.NO_APPLICABLE_POLICY:
                best_failure = MatchResult(
                    "NOT_ALLOWED",
                    apply_reason or reasons.NETWORK_POLICY_MISMATCH,
                    "network policy selector did not match",
                    network_policy.policy_id,
                )
            continue
        applicable_found = True
        if not _policy_date_valid(timestamp, network_policy.valid_from, network_policy.valid_until):
            best_failure = MatchResult(
                "NOT_ALLOWED",
                reasons.OUTSIDE_AUTHORIZED_TIME if timestamp else reasons.INVALID_TIMESTAMP,
                "network policy validity dates do not include the event timestamp",
                network_policy.policy_id,
            )
            continue
        direction, derivation = _derive_direction(network_policy, row)
        connection_failures = []
        for connection in network_policy.authorized_connections:
            result = _connection_matches(connection, direction, row, policy)
            if result.state == "ALLOWED":
                details = "network row fully matched allow-list policy"
                if derivation:
                    details = f"{details}; {derivation}"
                return MatchResult("ALLOWED", "ALLOWED", details, network_policy.policy_id)
            connection_failures.append(result)
        if connection_failures:
            chosen = _choose_failure(
                connection_failures,
                (
                    reasons.UNKNOWN_NETWORK_DIRECTION,
                    reasons.MISSING_REQUIRED_FIELD,
                    reasons.INVALID_TIMESTAMP,
                    reasons.OUTSIDE_AUTHORIZED_TIME,
                    reasons.UNAUTHORIZED_PROTOCOL,
                    reasons.UNAUTHORIZED_PORT,
                    reasons.UNAUTHORIZED_REMOTE_DOMAIN,
                    reasons.UNAUTHORIZED_REMOTE_IP,
                ),
            )
            assert chosen is not None
            best_failure = MatchResult(
                "NOT_ALLOWED",
                chosen.reason_code,
                chosen.reason_details,
                network_policy.policy_id,
            )
    if applicable_found and best_failure is not None:
        return best_failure
    if best_failure is not None:
        return best_failure
    return MatchResult(
        "NOT_APPLICABLE",
        reasons.NO_APPLICABLE_POLICY,
        "no enabled network policy applies",
    )


def _category(row: PreprocessedPolicyRow) -> str:
    categories = row.detected_categories
    if categories == {"system"}:
        return "system"
    if categories == {"network"}:
        return "network"
    if categories == {"system", "network"}:
        return "system_and_network"
    return "unknown"


def decide(row: PreprocessedPolicyRow, policy: PolicyDocument) -> FilterDecision:
    try:
        if row.conflicts:
            details = "; ".join(
                f"{name}: {', '.join(field.conflicting_columns)}"
                for name, field in row.conflicts.items()
            )
            return FilterDecision(
                "forward",
                _category(row),
                reasons.CONFLICTING_FIELD_VALUES,
                details,
            )
        category = _category(row)
        if category == "unknown":
            return FilterDecision("forward", "unknown", reasons.UNKNOWN_CATEGORY, "no known category or evidence")
        if category == "system":
            result = evaluate_system(row, policy)
            if result.state == "ALLOWED":
                return FilterDecision("suppress", "system", "ALLOWED", result.reason_details, result.matched_policy_id)
            return FilterDecision("forward", "system", result.reason_code, result.reason_details, result.matched_policy_id)
        if category == "network":
            result = evaluate_network(row, policy)
            if result.state == "ALLOWED":
                return FilterDecision("suppress", "network", "ALLOWED", result.reason_details, result.matched_policy_id)
            return FilterDecision("forward", "network", result.reason_code, result.reason_details, result.matched_policy_id)

        system_result = evaluate_system(row, policy)
        network_result = evaluate_network(row, policy)
        if system_result.state == "ALLOWED" and network_result.state == "ALLOWED":
            matched = ",".join(
                policy_id
                for policy_id in (system_result.matched_policy_id, network_result.matched_policy_id)
                if policy_id
            )
            return FilterDecision("suppress", "system_and_network", "ALLOWED", "system and network policies allowed row", matched)
        failure = network_result if system_result.state == "ALLOWED" else system_result
        return FilterDecision(
            "forward",
            "system_and_network",
            failure.reason_code,
            failure.reason_details,
            failure.matched_policy_id,
        )
    except Exception:
        LOGGER.exception("Unexpected policy-filter matching failure")
        return FilterDecision(
            "forward",
            "malformed",
            reasons.INTERNAL_FILTER_ERROR,
            "internal filter error while evaluating row",
        )
