from __future__ import annotations

from pathlib import Path

import yaml

from policy_filter.matcher import decide
from policy_filter.preprocessing_adapter import preprocess_policy_row
from policy_filter.schema import load_policy


def write_policy(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "network_policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


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
    policy = load_policy(write_policy(tmp_path, payload or network_policy()))
    return decide(preprocess_policy_row(row, policy), policy)


def test_authorized_outbound_ip_connection_is_suppressed(tmp_path: Path) -> None:
    decision = decision_for(tmp_path, network_row())

    assert decision.action == "suppress"
    assert decision.category == "network"


def test_authorized_cidr_connection_is_suppressed(tmp_path: Path) -> None:
    decision = decision_for(
        tmp_path,
        network_row(dst_ip="198.51.100.42", protocol="udp", dst_port="53"),
    )

    assert decision.action == "suppress"


def test_domain_protocol_port_and_unknown_remote_failures(tmp_path: Path) -> None:
    assert decision_for(
        tmp_path,
        network_row(dst_ip="", domain="api.office.com", dst_port="443"),
    ).action == "suppress"
    assert decision_for(
        tmp_path,
        network_row(dst_ip="", domain="api.office.com", dst_port="22"),
    ).reason_code == "UNAUTHORIZED_PORT"
    assert decision_for(
        tmp_path,
        network_row(protocol="udp"),
    ).reason_code == "UNAUTHORIZED_PROTOCOL"
    assert decision_for(
        tmp_path,
        network_row(dst_ip="", domain="unknown.example.com"),
    ).reason_code == "UNAUTHORIZED_REMOTE_DOMAIN"
    assert decision_for(
        tmp_path,
        network_row(dst_port=""),
    ).reason_code == "MISSING_REQUIRED_FIELD"


def test_unknown_or_derived_direction(tmp_path: Path) -> None:
    assert decision_for(tmp_path, network_row(direction="sideways")).reason_code == "UNKNOWN_NETWORK_DIRECTION"
    derived = decision_for(tmp_path, network_row(direction=""))
    assert derived.action == "suppress"
    assert "derived direction" in derived.reason_details


def test_adding_domain_in_yaml_changes_decision(tmp_path: Path) -> None:
    payload = network_policy()
    payload["network_policies"][0]["authorized_connections"][1]["remote_domains"].append("example.org")

    assert decision_for(
        tmp_path,
        network_row(dst_ip="", domain="example.org"),
        payload,
    ).action == "suppress"
