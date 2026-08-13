from __future__ import annotations

import ipaddress
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from policy_filter.models import (
    AllowedProgram,
    AuthorizedConnection,
    AuthorizedIdentity,
    CommandLinePolicy,
    EventIdPrefilterPolicy,
    NetworkPolicy,
    PolicyDocument,
    PortMatcher,
    PrefilterPolicy,
    SeverityPrefilterPolicy,
    SystemPolicy,
    TimeWindow,
)
from policy_filter.prefilters import (
    PrefilterValueError,
    canonical_event_id,
    canonical_severity_value,
)

LOGGER = logging.getLogger(__name__)

SUPPORTED_VERSION = 1
SUPPORTED_PREPROCESSING_OPTIONS = {
    "text_column",
    "include_cross_category",
    "preserve_empty_fields",
}
TOP_LEVEL_FIELDS = {
    "version",
    "organization_id",
    "default_timezone",
    "preprocessing",
    "field_mappings",
    "category_aliases",
    "system_policies",
    "network_policies",
    "prefilters",
}
PREFILTER_FIELDS = {"event_id", "severity"}
EVENT_ID_PREFILTER_FIELDS = {
    "suppress_ids",
    "always_forward_ids",
    "require_policy_match_for_suppression",
}
SEVERITY_PREFILTER_FIELDS = {
    "suppress_values",
    "always_forward_values",
    "case_insensitive",
    "require_policy_match_for_suppression",
}
SYSTEM_POLICY_FIELDS = {
    "policy_id",
    "enabled",
    "valid_from",
    "valid_until",
    "hosts",
    "local_ips",
    "authorized_identities",
}
IDENTITY_FIELDS = {
    "identity_id",
    "users",
    "groups",
    "authorized_time_windows",
    "allowed_programs",
}
PROGRAM_FIELDS = {
    "name",
    "paths",
    "allowed_parent_programs",
    "command_line",
}
COMMAND_LINE_FIELDS = {"allowed_patterns", "prohibited_patterns"}
NETWORK_POLICY_FIELDS = {
    "policy_id",
    "enabled",
    "valid_from",
    "valid_until",
    "local_hosts",
    "local_ips",
    "local_networks",
    "authorized_connections",
}
CONNECTION_FIELDS = {
    "connection_id",
    "direction",
    "remote_ips",
    "remote_cidrs",
    "remote_domains",
    "protocols",
    "destination_ports",
    "authorized_time_windows",
}
TIME_WINDOW_FIELDS = {"days", "start", "end", "timezone"}
WEEKDAY_TO_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


class PolicyValidationError(ValueError):
    pass


class _NoDuplicateSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping_without_duplicate_keys(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise PolicyValidationError(f"{key_node.start_mark}: duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicateSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_without_duplicate_keys,
)


def _fail(path: str, message: str) -> PolicyValidationError:
    return PolicyValidationError(f"{path}: {message}")


def _parse_preprocessing(value: Any, errors: list[str]) -> dict[str, Any]:
    mapping = _as_dict(value, "preprocessing")
    for key in sorted(set(mapping) - SUPPORTED_PREPROCESSING_OPTIONS):
        errors.append(f"preprocessing.{key}: unsupported preprocessing option")

    text_column = mapping.get("text_column")
    if text_column is not None and (
        not isinstance(text_column, str) or not text_column.strip()
    ):
        errors.append("preprocessing.text_column: expected null or a nonempty string")
    include_cross_category = mapping.get("include_cross_category", True)
    if not isinstance(include_cross_category, bool):
        errors.append("preprocessing.include_cross_category: expected a boolean")
        include_cross_category = True
    preserve_empty_fields = mapping.get("preserve_empty_fields", False)
    if not isinstance(preserve_empty_fields, bool):
        errors.append("preprocessing.preserve_empty_fields: expected a boolean")
        preserve_empty_fields = False
    return {
        "text_column": text_column.strip() if isinstance(text_column, str) else None,
        "include_cross_category": include_cross_category,
        "preserve_empty_fields": preserve_empty_fields,
    }


def _warn_or_raise(
    *,
    strict: bool,
    path: str,
    message: str,
    errors: list[str],
) -> None:
    full = f"{path}: {message}"
    if strict:
        errors.append(full)
    else:
        LOGGER.warning("Ignoring unsupported policy field: %s", full)


def _check_unknown_fields(
    mapping: dict[str, Any],
    allowed: set[str],
    *,
    strict: bool,
    path: str,
    errors: list[str],
) -> None:
    for key in sorted(set(mapping) - allowed):
        _warn_or_raise(
            strict=strict,
            path=f"{path}.{key}" if path else key,
            message="unknown field",
            errors=errors,
        )


def _as_dict(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _fail(path, "must be a mapping")
    return value


def _as_list(value: Any, path: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise _fail(path, "must be a list")
    return value


def _str_tuple(value: Any, path: str) -> tuple[str, ...]:
    result: list[str] = []
    for index, item in enumerate(_as_list(value, path)):
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
        else:
            raise _fail(f"{path}[{index}]", "empty string is not allowed")
    return tuple(result)


def _parse_bool(value: Any, path: str, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise _fail(path, f"must be a boolean, got {type(value).__name__}")


def _parse_ip(value: Any, path: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        return ipaddress.ip_address(str(value).strip())
    except ValueError as exc:
        raise _fail(path, f"invalid IP address {value!r}") from exc


def _parse_network(value: Any, path: str) -> ipaddress.IPv4Network | ipaddress.IPv6Network:
    try:
        return ipaddress.ip_network(str(value).strip(), strict=False)
    except ValueError as exc:
        raise _fail(path, f"invalid CIDR {value!r}") from exc


def _parse_port(value: Any, path: str) -> PortMatcher:
    text = str(value).strip()
    if "-" in text:
        start_text, end_text = text.split("-", maxsplit=1)
    else:
        start_text = end_text = text
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise _fail(path, f"invalid port value {value!r}") from exc
    if start < 0 or end > 65535 or start > end:
        raise _fail(path, f"invalid port range {value!r}")
    return PortMatcher(start=start, end=end)


def _parse_time(value: Any, path: str) -> int:
    text = str(value).strip()
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text)
    if match is None:
        raise _fail(path, f"invalid time {value!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def _parse_datetime(value: Any, path: str, default_timezone: str) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail(path, f"invalid datetime {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(default_timezone))
    return parsed.astimezone(timezone.utc)


def _validate_timezone(name: str, path: str) -> str:
    try:
        ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise _fail(path, f"invalid timezone {name!r}") from exc
    return name


def _parse_windows(
    value: Any,
    path: str,
    default_timezone: str,
    *,
    strict: bool,
    errors: list[str],
) -> tuple[TimeWindow, ...]:
    windows: list[TimeWindow] = []
    for index, item in enumerate(_as_list(value, path)):
        item_path = f"{path}[{index}]"
        mapping = _as_dict(item, item_path)
        _check_unknown_fields(
            mapping,
            TIME_WINDOW_FIELDS,
            strict=strict,
            path=item_path,
            errors=errors,
        )
        day_values = _as_list(mapping.get("days"), f"{item_path}.days")
        if not day_values:
            raise _fail(f"{item_path}.days", "at least one weekday is required")
        days: set[int] = set()
        for day_index, day in enumerate(day_values):
            key = str(day).strip().lower()
            if key not in WEEKDAY_TO_INDEX:
                raise _fail(f"{item_path}.days[{day_index}]", f"unknown weekday {day!r}")
            days.add(WEEKDAY_TO_INDEX[key])
        timezone_name = _validate_timezone(
            str(mapping.get("timezone") or default_timezone),
            f"{item_path}.timezone",
        )
        windows.append(
            TimeWindow(
                days=frozenset(days),
                start_minutes=_parse_time(mapping.get("start"), f"{item_path}.start"),
                end_minutes=_parse_time(mapping.get("end"), f"{item_path}.end"),
                timezone=timezone_name,
                path=item_path,
            )
        )
    return tuple(windows)


def _parse_command_line(
    value: Any,
    path: str,
    *,
    strict: bool,
    errors: list[str],
) -> CommandLinePolicy | None:
    if value is None:
        return None
    mapping = _as_dict(value, path)
    _check_unknown_fields(
        mapping,
        COMMAND_LINE_FIELDS,
        strict=strict,
        path=path,
        errors=errors,
    )
    allowed = []
    prohibited = []
    for key, target in (("allowed_patterns", allowed), ("prohibited_patterns", prohibited)):
        for index, pattern in enumerate(_as_list(mapping.get(key), f"{path}.{key}")):
            try:
                target.append(re.compile(str(pattern)))
            except re.error as exc:
                raise _fail(f"{path}.{key}[{index}]", f"invalid regular expression: {exc}") from exc
    return CommandLinePolicy(
        allowed_patterns=tuple(allowed),
        prohibited_patterns=tuple(prohibited),
    )


def _parse_program(
    value: Any,
    path: str,
    *,
    strict: bool,
    errors: list[str],
) -> AllowedProgram:
    mapping = _as_dict(value, path)
    _check_unknown_fields(
        mapping,
        PROGRAM_FIELDS,
        strict=strict,
        path=path,
        errors=errors,
    )
    name = str(mapping.get("name") or "").strip()
    if not name:
        raise _fail(f"{path}.name", "program name is required")
    return AllowedProgram(
        name=name,
        paths=_str_tuple(mapping.get("paths"), f"{path}.paths"),
        allowed_parent_programs=_str_tuple(
            mapping.get("allowed_parent_programs"),
            f"{path}.allowed_parent_programs",
        ),
        command_line=_parse_command_line(
            mapping.get("command_line"),
            f"{path}.command_line",
            strict=strict,
            errors=errors,
        ),
    )


def _parse_identity(
    value: Any,
    path: str,
    default_timezone: str,
    *,
    strict: bool,
    errors: list[str],
) -> AuthorizedIdentity:
    mapping = _as_dict(value, path)
    _check_unknown_fields(
        mapping,
        IDENTITY_FIELDS,
        strict=strict,
        path=path,
        errors=errors,
    )
    identity_id = str(mapping.get("identity_id") or "").strip()
    if not identity_id:
        raise _fail(f"{path}.identity_id", "identity_id is required")
    users = _str_tuple(mapping.get("users"), f"{path}.users")
    groups = _str_tuple(mapping.get("groups"), f"{path}.groups")
    if not users and not groups:
        raise _fail(path, "enabled identity policies require at least one user or group")
    programs = tuple(
        _parse_program(
            item,
            f"{path}.allowed_programs[{index}]",
            strict=strict,
            errors=errors,
        )
        for index, item in enumerate(_as_list(mapping.get("allowed_programs"), f"{path}.allowed_programs"))
    )
    if not programs:
        raise _fail(f"{path}.allowed_programs", "at least one allowed program is required")
    return AuthorizedIdentity(
        identity_id=identity_id,
        users=users,
        groups=groups,
        authorized_time_windows=_parse_windows(
            mapping.get("authorized_time_windows"),
            f"{path}.authorized_time_windows",
            default_timezone,
            strict=strict,
            errors=errors,
        ),
        allowed_programs=programs,
    )


def _parse_system_policy(
    value: Any,
    index: int,
    default_timezone: str,
    *,
    strict: bool,
    errors: list[str],
) -> SystemPolicy:
    path = f"system_policies[{index}]"
    mapping = _as_dict(value, path)
    _check_unknown_fields(
        mapping,
        SYSTEM_POLICY_FIELDS,
        strict=strict,
        path=path,
        errors=errors,
    )
    policy_id = str(mapping.get("policy_id") or "").strip()
    if not policy_id:
        raise _fail(f"{path}.policy_id", "policy_id is required")
    enabled = _parse_bool(mapping.get("enabled"), f"{path}.enabled", default=True)
    hosts = _str_tuple(mapping.get("hosts"), f"{path}.hosts")
    local_ips = tuple(
        _parse_ip(item, f"{path}.local_ips[{item_index}]")
        for item_index, item in enumerate(_as_list(mapping.get("local_ips"), f"{path}.local_ips"))
    )
    if enabled and not hosts and not local_ips:
        raise _fail(path, "policies require at least one host or local-IP selector")
    identities = tuple(
        _parse_identity(
            item,
            f"{path}.authorized_identities[{identity_index}]",
            default_timezone,
            strict=strict,
            errors=errors,
        )
        for identity_index, item in enumerate(
            _as_list(mapping.get("authorized_identities"), f"{path}.authorized_identities")
        )
    )
    if enabled and not identities:
        raise _fail(f"{path}.authorized_identities", "at least one identity is required")
    return SystemPolicy(
        policy_id=policy_id,
        enabled=enabled,
        valid_from=_parse_datetime(mapping.get("valid_from"), f"{path}.valid_from", default_timezone),
        valid_until=_parse_datetime(mapping.get("valid_until"), f"{path}.valid_until", default_timezone),
        hosts=hosts,
        local_ips=local_ips,
        authorized_identities=identities,
    )


def _normalize_domain_pattern(value: Any, path: str) -> str:
    domain = str(value).strip().rstrip(".").lower()
    if not domain:
        raise _fail(path, "empty domain is not allowed")
    if domain.startswith("*."):
        suffix = domain[2:]
        if "." not in suffix or suffix.startswith("."):
            raise _fail(path, f"invalid wildcard domain {value!r}")
        return domain
    if "." not in domain:
        raise _fail(path, f"invalid domain {value!r}")
    return domain


def _parse_connection(
    value: Any,
    path: str,
    default_timezone: str,
    *,
    strict: bool,
    errors: list[str],
) -> AuthorizedConnection:
    mapping = _as_dict(value, path)
    _check_unknown_fields(
        mapping,
        CONNECTION_FIELDS,
        strict=strict,
        path=path,
        errors=errors,
    )
    connection_id = str(mapping.get("connection_id") or "").strip()
    if not connection_id:
        raise _fail(f"{path}.connection_id", "connection_id is required")
    direction = str(mapping.get("direction") or "").strip().lower()
    if direction not in {"inbound", "outbound", "internal", "any"}:
        raise _fail(f"{path}.direction", f"unsupported direction {direction!r}")
    remote_ips = tuple(
        _parse_ip(item, f"{path}.remote_ips[{item_index}]")
        for item_index, item in enumerate(_as_list(mapping.get("remote_ips"), f"{path}.remote_ips"))
    )
    remote_cidrs = tuple(
        _parse_network(item, f"{path}.remote_cidrs[{item_index}]")
        for item_index, item in enumerate(_as_list(mapping.get("remote_cidrs"), f"{path}.remote_cidrs"))
    )
    remote_domains = tuple(
        _normalize_domain_pattern(item, f"{path}.remote_domains[{item_index}]")
        for item_index, item in enumerate(
            _as_list(mapping.get("remote_domains"), f"{path}.remote_domains")
        )
    )
    if not remote_ips and not remote_cidrs and not remote_domains:
        raise _fail(path, "enabled network connections require a remote IP, CIDR, or domain")
    return AuthorizedConnection(
        connection_id=connection_id,
        direction=direction,
        remote_ips=remote_ips,
        remote_cidrs=remote_cidrs,
        remote_domains=remote_domains,
        protocols=tuple(str(protocol).strip().lower() for protocol in _str_tuple(mapping.get("protocols"), f"{path}.protocols")),
        destination_ports=tuple(
            _parse_port(item, f"{path}.destination_ports[{item_index}]")
            for item_index, item in enumerate(
                _as_list(mapping.get("destination_ports"), f"{path}.destination_ports")
            )
        ),
        authorized_time_windows=_parse_windows(
            mapping.get("authorized_time_windows"),
            f"{path}.authorized_time_windows",
            default_timezone,
            strict=strict,
            errors=errors,
        ),
    )


def _parse_network_policy(
    value: Any,
    index: int,
    default_timezone: str,
    *,
    strict: bool,
    errors: list[str],
) -> NetworkPolicy:
    path = f"network_policies[{index}]"
    mapping = _as_dict(value, path)
    _check_unknown_fields(
        mapping,
        NETWORK_POLICY_FIELDS,
        strict=strict,
        path=path,
        errors=errors,
    )
    policy_id = str(mapping.get("policy_id") or "").strip()
    if not policy_id:
        raise _fail(f"{path}.policy_id", "policy_id is required")
    enabled = _parse_bool(mapping.get("enabled"), f"{path}.enabled", default=True)
    local_hosts = _str_tuple(mapping.get("local_hosts"), f"{path}.local_hosts")
    local_ips = tuple(
        _parse_ip(item, f"{path}.local_ips[{item_index}]")
        for item_index, item in enumerate(_as_list(mapping.get("local_ips"), f"{path}.local_ips"))
    )
    local_networks = tuple(
        _parse_network(item, f"{path}.local_networks[{item_index}]")
        for item_index, item in enumerate(
            _as_list(mapping.get("local_networks"), f"{path}.local_networks")
        )
    )
    if enabled and not local_hosts and not local_ips and not local_networks:
        raise _fail(path, "policies require at least one host or local-IP selector")
    connections = tuple(
        _parse_connection(
            item,
            f"{path}.authorized_connections[{connection_index}]",
            default_timezone,
            strict=strict,
            errors=errors,
        )
        for connection_index, item in enumerate(
            _as_list(mapping.get("authorized_connections"), f"{path}.authorized_connections")
        )
    )
    if enabled and not connections:
        raise _fail(f"{path}.authorized_connections", "at least one connection is required")
    connection_ids = [connection.connection_id for connection in connections]
    duplicates = sorted({item for item in connection_ids if connection_ids.count(item) > 1})
    if duplicates:
        raise _fail(f"{path}.authorized_connections", f"duplicate connection IDs: {duplicates}")
    return NetworkPolicy(
        policy_id=policy_id,
        enabled=enabled,
        valid_from=_parse_datetime(mapping.get("valid_from"), f"{path}.valid_from", default_timezone),
        valid_until=_parse_datetime(mapping.get("valid_until"), f"{path}.valid_until", default_timezone),
        local_hosts=local_hosts,
        local_ips=local_ips,
        local_networks=local_networks,
        authorized_connections=connections,
    )


def _event_id_set(value: Any, path: str) -> frozenset[str]:
    normalized: set[str] = set()
    for index, item in enumerate(_as_list(value, path)):
        try:
            normalized.add(canonical_event_id(item))
        except PrefilterValueError as exc:
            raise _fail(f"{path}[{index}]", str(exc)) from exc
    return frozenset(normalized)


def _severity_set(value: Any, path: str, *, case_insensitive: bool) -> frozenset[str]:
    normalized: set[str] = set()
    for index, item in enumerate(_as_list(value, path)):
        try:
            normalized.add(
                canonical_severity_value(
                    item,
                    case_insensitive=case_insensitive,
                )
            )
        except PrefilterValueError as exc:
            raise _fail(f"{path}[{index}]", str(exc)) from exc
    return frozenset(normalized)


def _parse_event_id_prefilter(
    value: Any,
    *,
    strict: bool,
    errors: list[str],
) -> EventIdPrefilterPolicy:
    path = "prefilters.event_id"
    mapping = _as_dict(value, path)
    _check_unknown_fields(
        mapping,
        EVENT_ID_PREFILTER_FIELDS,
        strict=strict,
        path=path,
        errors=errors,
    )
    suppress_ids = _event_id_set(mapping.get("suppress_ids"), f"{path}.suppress_ids")
    always_forward_ids = _event_id_set(
        mapping.get("always_forward_ids"),
        f"{path}.always_forward_ids",
    )
    if not suppress_ids and not always_forward_ids:
        raise _fail(path, "event_id prefilter must define suppress_ids or always_forward_ids")
    overlap = sorted(suppress_ids & always_forward_ids)
    if overlap:
        raise _fail(path, f"Event IDs cannot be both suppressed and always-forwarded: {overlap}")
    return EventIdPrefilterPolicy(
        suppress_ids=suppress_ids,
        always_forward_ids=always_forward_ids,
        require_policy_match_for_suppression=_parse_bool(
            mapping.get("require_policy_match_for_suppression"),
            f"{path}.require_policy_match_for_suppression",
            default=True,
        ),
    )


def _parse_severity_prefilter(
    value: Any,
    *,
    strict: bool,
    errors: list[str],
) -> SeverityPrefilterPolicy:
    path = "prefilters.severity"
    mapping = _as_dict(value, path)
    _check_unknown_fields(
        mapping,
        SEVERITY_PREFILTER_FIELDS,
        strict=strict,
        path=path,
        errors=errors,
    )
    case_insensitive = _parse_bool(
        mapping.get("case_insensitive"),
        f"{path}.case_insensitive",
        default=True,
    )
    suppress_values = _severity_set(
        mapping.get("suppress_values"),
        f"{path}.suppress_values",
        case_insensitive=case_insensitive,
    )
    always_forward_values = _severity_set(
        mapping.get("always_forward_values"),
        f"{path}.always_forward_values",
        case_insensitive=case_insensitive,
    )
    if not suppress_values and not always_forward_values:
        raise _fail(path, "severity prefilter must define suppress_values or always_forward_values")
    overlap = sorted(suppress_values & always_forward_values)
    if overlap:
        raise _fail(path, f"severity values cannot be both suppressed and always-forwarded: {overlap}")
    return SeverityPrefilterPolicy(
        suppress_values=suppress_values,
        always_forward_values=always_forward_values,
        case_insensitive=case_insensitive,
        require_policy_match_for_suppression=_parse_bool(
            mapping.get("require_policy_match_for_suppression"),
            f"{path}.require_policy_match_for_suppression",
            default=True,
        ),
    )


def _parse_prefilters(
    value: Any,
    *,
    strict: bool,
    errors: list[str],
) -> PrefilterPolicy:
    mapping = _as_dict(value, "prefilters")
    _check_unknown_fields(
        mapping,
        PREFILTER_FIELDS,
        strict=strict,
        path="prefilters",
        errors=errors,
    )
    return PrefilterPolicy(
        event_id=(
            _parse_event_id_prefilter(
                mapping["event_id"],
                strict=strict,
                errors=errors,
            )
            if "event_id" in mapping
            else None
        ),
        severity=(
            _parse_severity_prefilter(
                mapping["severity"],
                strict=strict,
                errors=errors,
            )
            if "severity" in mapping
            else None
        ),
    )


def _load_raw_policy(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                payload = json.load(handle)
            else:
                payload = yaml.load(handle, Loader=_NoDuplicateSafeLoader)
    except OSError:
        raise
    except Exception as exc:
        raise PolicyValidationError(f"{path}: failed to parse policy: {exc}") from exc
    if not isinstance(payload, dict):
        raise PolicyValidationError(f"{path}: policy must be a mapping")
    return payload


def load_policy(path: str | Path, *, strict: bool = False) -> PolicyDocument:
    policy_path = Path(path)
    raw = _load_raw_policy(policy_path)
    errors: list[str] = []
    _check_unknown_fields(raw, TOP_LEVEL_FIELDS, strict=strict, path="", errors=errors)

    version = int(raw.get("version", 0))
    if version != SUPPORTED_VERSION:
        raise PolicyValidationError(f"version: unsupported schema version {version!r}")

    default_timezone = str(raw.get("default_timezone") or "UTC")
    _validate_timezone(default_timezone, "default_timezone")
    preprocessing = _parse_preprocessing(raw.get("preprocessing"), errors)

    field_mapping_payload = _as_dict(raw.get("field_mappings"), "field_mappings")
    field_mappings = {
        str(logical): _str_tuple(aliases, f"field_mappings.{logical}")
        for logical, aliases in field_mapping_payload.items()
    }
    category_payload = _as_dict(raw.get("category_aliases"), "category_aliases")
    category_aliases = {
        str(category): tuple(alias.lower() for alias in _str_tuple(aliases, f"category_aliases.{category}"))
        for category, aliases in category_payload.items()
    }

    system_policies = tuple(
        _parse_system_policy(
            item,
            index,
            default_timezone,
            strict=strict,
            errors=errors,
        )
        for index, item in enumerate(_as_list(raw.get("system_policies"), "system_policies"))
    )
    network_policies = tuple(
        _parse_network_policy(
            item,
            index,
            default_timezone,
            strict=strict,
            errors=errors,
        )
        for index, item in enumerate(_as_list(raw.get("network_policies"), "network_policies"))
    )
    prefilters = _parse_prefilters(
        raw.get("prefilters"),
        strict=strict,
        errors=errors,
    )
    policy_ids = [policy.policy_id for policy in system_policies + network_policies]
    duplicates = sorted({item for item in policy_ids if policy_ids.count(item) > 1})
    if duplicates:
        errors.append(f"policy_id: duplicate policy IDs: {duplicates}")
    if errors:
        raise PolicyValidationError("; ".join(errors))
    return PolicyDocument(
        version=version,
        organization_id=str(raw.get("organization_id") or ""),
        default_timezone=default_timezone,
        preprocessing=preprocessing,
        field_mappings=field_mappings,
        category_aliases=category_aliases,
        system_policies=system_policies,
        network_policies=network_policies,
        prefilters=prefilters,
        raw=raw,
    )
