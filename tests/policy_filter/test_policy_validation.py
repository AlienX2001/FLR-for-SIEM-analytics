from __future__ import annotations

import pytest

from policy_filter.schema import PolicyValidationError, load_policy
from policy_filter_helpers import network_policy
from policy_filter_helpers import base_policy, write_policy


def test_duplicate_policy_ids_are_rejected(tmp_path) -> None:
    payload = base_policy()
    payload["network_policies"] = network_policy()["network_policies"]
    payload["network_policies"][0]["policy_id"] = payload["system_policies"][0]["policy_id"]

    with pytest.raises(PolicyValidationError, match="duplicate policy IDs"):
        load_policy(write_policy(tmp_path, payload))


def test_invalid_ip_cidr_port_timezone_weekday_and_regex_are_rejected(tmp_path) -> None:
    payload = network_policy()
    payload["network_policies"][0]["authorized_connections"][0]["destination_ports"] = ["9000-8000"]
    with pytest.raises(PolicyValidationError, match="invalid port range"):
        load_policy(write_policy(tmp_path, payload))

    payload = network_policy()
    payload["network_policies"][0]["local_networks"] = ["not-a-cidr"]
    with pytest.raises(PolicyValidationError, match="invalid CIDR"):
        load_policy(write_policy(tmp_path, payload))

    payload = base_policy()
    payload["system_policies"][0]["local_ips"] = ["999.1.1.1"]
    with pytest.raises(PolicyValidationError, match="invalid IP address"):
        load_policy(write_policy(tmp_path, payload))

    payload = base_policy()
    payload["system_policies"][0]["authorized_identities"][0]["authorized_time_windows"][0]["timezone"] = "Mars/Base"
    with pytest.raises(PolicyValidationError, match="invalid timezone"):
        load_policy(write_policy(tmp_path, payload))

    payload = base_policy()
    payload["system_policies"][0]["authorized_identities"][0]["authorized_time_windows"][0]["days"] = ["Funday"]
    with pytest.raises(PolicyValidationError, match="unknown weekday"):
        load_policy(write_policy(tmp_path, payload))

    payload = base_policy()
    payload["system_policies"][0]["authorized_identities"][0]["allowed_programs"][0]["command_line"] = {
        "allowed_patterns": ["["],
        "prohibited_patterns": [],
    }
    with pytest.raises(PolicyValidationError, match="invalid regular expression"):
        load_policy(write_policy(tmp_path, payload))


def test_strict_unknown_fields_are_rejected(tmp_path) -> None:
    payload = base_policy()
    payload["system_policies"][0]["unknown"] = True

    with pytest.raises(PolicyValidationError, match="unknown field"):
        load_policy(write_policy(tmp_path, payload), strict=True)
