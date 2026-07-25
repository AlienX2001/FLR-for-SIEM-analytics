from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network
from pathlib import Path
from re import Pattern
from typing import Any, Literal, Mapping

IPAddress = IPv4Address | IPv6Address
IPNetwork = IPv4Network | IPv6Network
FilterAction = Literal["suppress", "forward"]
EvaluatedCategory = Literal["system", "network", "system_and_network", "unknown", "malformed"]
MatchState = Literal["ALLOWED", "NOT_ALLOWED", "NOT_APPLICABLE"]


@dataclass(frozen=True)
class ResolvedField:
    logical_name: str
    value: str | None
    source_column: str | None = None
    original_value: str | None = None
    conflicting_columns: tuple[str, ...] = ()
    conflicting_values: tuple[str, ...] = ()

    @property
    def is_missing(self) -> bool:
        return self.value is None or self.value == ""

    @property
    def is_conflicting(self) -> bool:
        return bool(self.conflicting_columns)


@dataclass(frozen=True)
class PreprocessedPolicyRow:
    normalized_text: str
    token_counts: Mapping[str, int]
    canonical_fields: Mapping[str, str]
    resolved_fields: Mapping[str, ResolvedField]
    system_evidence: Mapping[str, Any]
    network_evidence: Mapping[str, Any]
    cross_evidence: Mapping[str, Any]
    detected_categories: frozenset[str]
    conflicts: Mapping[str, ResolvedField] = field(default_factory=dict)


@dataclass(frozen=True)
class CsvLogRecord:
    source_file: Path
    source_row_number: int
    source_line_number: int
    row: dict[str, str]
    malformed: bool = False
    malformed_reason: str | None = None


@dataclass(frozen=True)
class FilterDecision:
    action: FilterAction
    category: EvaluatedCategory
    reason_code: str
    reason_details: str
    matched_policy_id: str | None = None


@dataclass(frozen=True)
class MatchResult:
    state: MatchState
    reason_code: str
    reason_details: str
    matched_policy_id: str | None = None


@dataclass(frozen=True)
class TimeWindow:
    days: frozenset[int]
    start_minutes: int
    end_minutes: int
    timezone: str
    path: str


@dataclass(frozen=True)
class CommandLinePolicy:
    allowed_patterns: tuple[Pattern[str], ...] = ()
    prohibited_patterns: tuple[Pattern[str], ...] = ()


@dataclass(frozen=True)
class AllowedProgram:
    name: str
    paths: tuple[str, ...] = ()
    allowed_parent_programs: tuple[str, ...] = ()
    command_line: CommandLinePolicy | None = None


@dataclass(frozen=True)
class AuthorizedIdentity:
    identity_id: str
    users: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    authorized_time_windows: tuple[TimeWindow, ...] = ()
    allowed_programs: tuple[AllowedProgram, ...] = ()


@dataclass(frozen=True)
class SystemPolicy:
    policy_id: str
    enabled: bool
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    hosts: tuple[str, ...] = ()
    local_ips: tuple[IPAddress, ...] = ()
    authorized_identities: tuple[AuthorizedIdentity, ...] = ()


@dataclass(frozen=True)
class PortMatcher:
    start: int
    end: int

    def contains(self, port: int) -> bool:
        return self.start <= port <= self.end


@dataclass(frozen=True)
class AuthorizedConnection:
    connection_id: str
    direction: str
    remote_ips: tuple[IPAddress, ...] = ()
    remote_cidrs: tuple[IPNetwork, ...] = ()
    remote_domains: tuple[str, ...] = ()
    protocols: tuple[str, ...] = ()
    destination_ports: tuple[PortMatcher, ...] = ()
    authorized_time_windows: tuple[TimeWindow, ...] = ()


@dataclass(frozen=True)
class NetworkPolicy:
    policy_id: str
    enabled: bool
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    local_hosts: tuple[str, ...] = ()
    local_ips: tuple[IPAddress, ...] = ()
    local_networks: tuple[IPNetwork, ...] = ()
    authorized_connections: tuple[AuthorizedConnection, ...] = ()


@dataclass(frozen=True)
class PolicyDocument:
    version: int
    organization_id: str
    default_timezone: str
    preprocessing: Mapping[str, Any]
    field_mappings: Mapping[str, tuple[str, ...]]
    category_aliases: Mapping[str, tuple[str, ...]]
    system_policies: tuple[SystemPolicy, ...]
    network_policies: tuple[NetworkPolicy, ...]
    raw: Mapping[str, Any]
