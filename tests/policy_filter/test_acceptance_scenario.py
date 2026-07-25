from __future__ import annotations

import csv
from pathlib import Path

import yaml

from policy_filter.cli import run
from policy_filter.config import FilterCliConfig


def test_acceptance_scenario(tmp_path: Path) -> None:
    policy = {
        "version": 1,
        "organization_id": "example",
        "default_timezone": "UTC",
        "field_mappings": {
            "timestamp": ["timestamp"],
            "host": ["host"],
            "user": ["user"],
            "program": ["process"],
            "destination_ip": ["dst_ip"],
            "destination_port": ["dst_port"],
            "protocol": ["protocol"],
            "direction": ["direction"],
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
                        "authorized_time_windows": [
                            {
                                "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                                "start": "08:00",
                                "end": "18:00",
                                "timezone": "UTC",
                            }
                        ],
                        "allowed_programs": [{"name": "browser.exe"}],
                    }
                ],
            }
        ],
        "network_policies": [
            {
                "policy_id": "workstation_network",
                "enabled": True,
                "local_hosts": ["HOST-01"],
                "authorized_connections": [
                    {
                        "connection_id": "web_access",
                        "direction": "outbound",
                        "remote_ips": ["203.0.113.20"],
                        "protocols": ["tcp"],
                        "destination_ports": [443],
                    }
                ],
            }
        ],
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    logs = tmp_path / "logs.csv"
    logs.write_text(
        "timestamp,host,user,process,dst_ip,dst_port,protocol,direction\n"
        "2026-07-20T09:00:00Z,HOST-01,alice,browser.exe,203.0.113.20,443,tcp,outbound\n"
        "2026-07-20T09:05:00Z,HOST-01,alice,powershell.exe,203.0.113.20,443,tcp,outbound\n"
        "2026-07-20T09:10:00Z,HOST-01,alice,browser.exe,198.51.100.40,443,tcp,outbound\n"
        "2026-07-20T22:00:00Z,HOST-01,alice,browser.exe,203.0.113.20,443,tcp,outbound\n",
        encoding="utf-8",
    )
    output = tmp_path / "forwarded.csv"

    assert run(FilterCliConfig(logs=[logs], policy=policy_path, output=output)) == 0

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["source_row_number"] for row in rows] == ["2", "3", "4"]
    assert [row["filter_reason"] for row in rows] == [
        "UNAUTHORIZED_PROGRAM",
        "UNAUTHORIZED_REMOTE_IP",
        "OUTSIDE_AUTHORIZED_TIME",
    ]
    assert rows[0]["process"] == "powershell.exe"
