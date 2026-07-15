from __future__ import annotations

from ioc_generation.extractors import IndicatorCandidate
from ioc_generation.stix import deterministic_stix_id, make_bundle, make_indicator


def test_stix_indicator_is_valid_like_and_deterministic() -> None:
    candidate = IndicatorCandidate("ipv4", "10.0.0.5")

    indicator_a = make_indicator(
        candidate,
        internal_log_id="org_0_row_3",
        org_index=0,
        row_index=3,
        predicted_label="malicious",
        risk_probability=0.91,
    )
    indicator_b = make_indicator(
        candidate,
        internal_log_id="org_0_row_3",
        org_index=0,
        row_index=3,
        predicted_label="malicious",
        risk_probability=0.91,
    )

    assert indicator_a == indicator_b
    assert indicator_a["type"] == "indicator"
    assert indicator_a["spec_version"] == "2.1"
    assert indicator_a["id"].startswith("indicator--")
    assert indicator_a["pattern"] == "[ipv4-addr:value = '10.0.0.5']"
    assert deterministic_stix_id("indicator", "ipv4:10.0.0.5") == indicator_a["id"]


def test_stix_bundle_contains_indicators() -> None:
    indicator = make_indicator(
        IndicatorCandidate("domain", "example.com"),
        internal_log_id="org_1_row_0",
        org_index=1,
        row_index=0,
    )

    bundle = make_bundle([indicator])

    assert bundle["type"] == "bundle"
    assert bundle["id"].startswith("bundle--")
    assert bundle["objects"] == [indicator]
