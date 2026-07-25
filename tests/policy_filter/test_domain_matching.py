from __future__ import annotations

from policy_filter_helpers import decision_for, network_row, network_policy


def test_wildcard_domain_matches_only_subdomains(tmp_path) -> None:
    assert decision_for(
        tmp_path,
        network_row(dst_ip="", domain="api.example.com"),
        _policy_with_domain("*.example.com"),
    ).action == "suppress"
    assert decision_for(
        tmp_path,
        network_row(dst_ip="", domain="example.com"),
        _policy_with_domain("*.example.com"),
    ).reason_code == "UNAUTHORIZED_REMOTE_DOMAIN"
    assert decision_for(
        tmp_path,
        network_row(dst_ip="", domain="maliciousexample.com"),
        _policy_with_domain("*.example.com"),
    ).reason_code == "UNAUTHORIZED_REMOTE_DOMAIN"


def test_domain_matching_is_case_insensitive_and_trailing_dot_safe(tmp_path) -> None:
    assert decision_for(
        tmp_path,
        network_row(dst_ip="", domain="API.OFFICE.COM."),
        network_policy(),
    ).action == "suppress"


def _policy_with_domain(domain: str) -> dict:
    payload = network_policy()
    payload["network_policies"][0]["authorized_connections"] = [
        {
            "connection_id": "domain",
            "direction": "outbound",
            "remote_domains": [domain],
            "protocols": ["tcp"],
            "destination_ports": [443],
        }
    ]
    return payload
