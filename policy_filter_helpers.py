from __future__ import annotations

from pathlib import Path

import yaml

from policy_filter.matcher import decide
from policy_filter.preprocessing_adapter import preprocess_policy_row
from policy_filter.schema import load_policy


def write_policy(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def base_policy() -> dict:
    return {
        "version": 1,
        "organization_id": "test",
        "default_timezone": "UTC",
        "field_mappings": {
            "timestamp": ["timestamp"],
            "category": ["event_category"],
            "host": ["host"],
            "local_ip": ["local_ip"],
            "user": ["user"],
            "user_groups": ["groups"],
            "program": ["process"],
            "program_path": ["process_path"],
            "parent_program": ["parent"],
            "command_line": ["cmdline"],
            "source_ip": ["src_ip"],
            "destination_ip": ["dst_ip"],
            "domain": ["domain"],
            "protocol": ["protocol"],
            "destination_port": ["dst_port"],
            "direction": ["direction"],
        },
        "category_aliases": {
            "system": ["system", "endpoint"],
            "network": ["network", "flow"],
        },
        "system_policies": [
            {
                "policy_id": "workstation_access",
                "enabled": True,
                "hosts": ["HOST-01"],
                "authorized_identities": [
                    {
                        "identity_id": "alice_access",
                        "users": ["alice"],
                        "groups": ["finance-users"],
                        "authorized_time_windows": [
                            {
                                "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                                "start": "08:00",
                                "end": "18:00",
                                "timezone": "UTC",
                            }
                        ],
                        "allowed_programs": [
                            {
                                "name": "browser.exe",
                                "paths": ["C:\\Program Files\\Browser\\browser.exe"],
                                "allowed_parent_programs": ["explorer.exe"],
                            },
                            {
                                "name": "powershell.exe",
                                "command_line": {
                                    "allowed_patterns": [
                                        r"^powershell\.exe_-file_c:\\approvedscripts\\[^\\]+\.ps1$"
                                    ],
                                    "prohibited_patterns": [r"encodedcommand"],
                                },
                            },
                        ],
                    }
                ],
            }
        ],
        "network_policies": [],
    }


def system_row(**overrides: str) -> dict[str, str]:
    row = {
        "timestamp": "2026-07-20T09:00:00Z",
        "host": "HOST-01",
        "user": "alice",
        "groups": "",
        "process": "browser.exe",
        "process_path": "C:\\Program Files\\Browser\\browser.exe",
        "parent": "explorer.exe",
        "cmdline": "",
    }
    row.update(overrides)
    return row


def network_policy() -> dict:
    return {
        "version": 1,
        "organization_id": "test",
        "default_timezone": "UTC",
        "field_mappings": {
            "timestamp": ["timestamp"],
            "host": ["host"],
            "source_ip": ["src_ip"],
            "destination_ip": ["dst_ip"],
            "domain": ["domain"],
            "protocol": ["protocol"],
            "destination_port": ["dst_port"],
            "direction": ["direction"],
        },
        "category_aliases": {"network": ["network"], "system": ["system"]},
        "system_policies": [],
        "network_policies": [
            {
                "policy_id": "workstation_network",
                "enabled": True,
                "local_hosts": ["HOST-01"],
                "local_networks": ["10.0.0.0/24"],
                "authorized_connections": [
                    {
                        "connection_id": "web_ip",
                        "direction": "outbound",
                        "remote_ips": ["203.0.113.20"],
                        "protocols": ["tcp"],
                        "destination_ports": [443],
                    },
                    {
                        "connection_id": "office_domain",
                        "direction": "outbound",
                        "remote_domains": ["*.office.com"],
                        "protocols": ["tcp"],
                        "destination_ports": ["8000-8100", 443],
                    },
                    {
                        "connection_id": "cidr",
                        "direction": "outbound",
                        "remote_cidrs": ["198.51.100.0/24"],
                        "protocols": ["udp"],
                        "destination_ports": [53],
                    },
                ],
            }
        ],
    }


def network_row(**overrides: str) -> dict[str, str]:
    row = {
        "timestamp": "2026-07-20T09:00:00Z",
        "host": "HOST-01",
        "src_ip": "10.0.0.10",
        "dst_ip": "203.0.113.20",
        "domain": "",
        "protocol": "tcp",
        "dst_port": "443",
        "direction": "outbound",
    }
    row.update(overrides)
    return row


def decision_for(tmp_path: Path, row: dict[str, str], payload: dict | None = None):
    policy = load_policy(write_policy(tmp_path, payload or base_policy()))
    return decide(preprocess_policy_row(row, policy), policy)
