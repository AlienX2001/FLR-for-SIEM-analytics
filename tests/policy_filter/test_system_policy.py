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


def decision_for(tmp_path: Path, row: dict[str, str], payload: dict | None = None):
    policy = load_policy(write_policy(tmp_path, payload or base_policy()))
    return decide(preprocess_policy_row(row, policy), policy)


def test_authorized_system_row_is_suppressed(tmp_path: Path) -> None:
    decision = decision_for(tmp_path, system_row())

    assert decision.action == "suppress"
    assert decision.category == "system"
    assert decision.matched_policy_id == "workstation_access"


def test_unknown_and_missing_host_are_forwarded(tmp_path: Path) -> None:
    assert decision_for(tmp_path, system_row(host="HOST-99")).reason_code == "UNAUTHORIZED_HOST"
    assert decision_for(tmp_path, system_row(host="")).reason_code == "MISSING_REQUIRED_FIELD"


def test_unauthorized_user_is_forwarded(tmp_path: Path) -> None:
    decision = decision_for(tmp_path, system_row(user="mallory"))

    assert decision.action == "forward"
    assert decision.reason_code == "UNAUTHORIZED_USER"


def test_outside_time_window_and_malformed_timestamp_are_forwarded(tmp_path: Path) -> None:
    assert decision_for(tmp_path, system_row(timestamp="2026-07-20T22:00:00Z")).reason_code == "OUTSIDE_AUTHORIZED_TIME"
    assert decision_for(tmp_path, system_row(timestamp="not-a-date")).reason_code == "INVALID_TIMESTAMP"
    assert decision_for(tmp_path, system_row(timestamp="")).reason_code == "INVALID_TIMESTAMP"


def test_program_path_parent_and_command_line_failures_are_forwarded(tmp_path: Path) -> None:
    assert decision_for(tmp_path, system_row(process_path="C:\\Temp\\browser.exe")).reason_code == "UNAUTHORIZED_PROGRAM_PATH"
    assert decision_for(tmp_path, system_row(parent="cmd.exe")).reason_code == "UNAUTHORIZED_PARENT_PROGRAM"
    assert decision_for(
        tmp_path,
        system_row(process="powershell.exe", process_path="", parent="", cmdline="powershell.exe -EncodedCommand abc"),
    ).reason_code == "COMMAND_LINE_POLICY_VIOLATION"


def test_alias_change_in_yaml_changes_resolution(tmp_path: Path) -> None:
    payload = base_policy()
    payload["field_mappings"]["user"] = ["account"]
    row = system_row()
    row.pop("user")
    row["account"] = "alice"

    assert decision_for(tmp_path, row, payload).action == "suppress"


def test_adding_user_in_yaml_changes_decision(tmp_path: Path) -> None:
    payload = base_policy()
    payload["system_policies"][0]["authorized_identities"][0]["users"].append("mallory")

    assert decision_for(tmp_path, system_row(user="mallory"), payload).action == "suppress"
