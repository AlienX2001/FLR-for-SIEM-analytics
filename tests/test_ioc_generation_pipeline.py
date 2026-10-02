from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ioc_generation.correlate import generate_ioc_outputs
from ioc_generation.utils import write_jsonl


def test_ioc_generation_does_not_emit_structured_metadata_or_url_path_iocs(
    tmp_path: Path,
) -> None:
    logs_path = tmp_path / "logs.csv"
    pd.DataFrame(
        [
            {
                "normalized_id": "368c710a8a0a763186eab09971d12717",
                "entity_id": "000062ac9da6cdaebe3fc1477043fc57",
                "src_ip": "192.168.219.134",
                "remote_address": "https://ellechina.online/01_logo_HLW-300x168.jpg",
                "download": "http://200.98.142.12/system/MA-1.0.0.0/fbclient.dll",
                "sub_label_cat": "suspicious callback domain",
            }
        ]
    ).to_csv(logs_path, index=False)

    high_risk_path = tmp_path / "high_risk.jsonl"
    explanations_path = tmp_path / "explanations.jsonl"
    write_jsonl(
        high_risk_path,
        [
            {
                "org_index": 0,
                "row_index": 0,
                "internal_log_id": "org_0_row_0",
                "predicted_label": "command and control",
                "max_risk_probability": 0.99,
            }
        ],
    )
    write_jsonl(
        explanations_path,
        [
            {
                "org_index": 0,
                "row_index": 0,
                "internal_log_id": "org_0_row_0",
                "top_contributions": {},
            }
        ],
    )

    output_dir = tmp_path / "iocgen"
    generate_ioc_outputs(
        high_risk_logs=high_risk_path,
        explanations=explanations_path,
        org_data=[logs_path],
        output_dir=output_dir,
    )

    records = [
        json.loads(line)
        for line in (output_dir / "ioc_records.jsonl").read_text().splitlines()
    ]
    extracted = {
        (record["indicator_type"], record["indicator_value"]) for record in records
    }

    assert ("ipv4", "192.168.219.134") in extracted
    assert ("url", "https://ellechina.online/01_logo_HLW-300x168.jpg") in extracted
    assert ("domain", "ellechina.online") in extracted
    assert ("url", "http://200.98.142.12/system/MA-1.0.0.0/fbclient.dll") in extracted
    assert ("ipv4", "200.98.142.12") in extracted

    assert ("md5", "368c710a8a0a763186eab09971d12717") not in extracted
    assert ("md5", "000062ac9da6cdaebe3fc1477043fc57") not in extracted
    assert ("domain", "-300x168.jpg") not in extracted
    assert ("domain", "300x168.jpg") not in extracted
    assert ("ipv4", "1.0.0.0") not in extracted

    bundle = json.loads((output_dir / "ioc_bundle.json").read_text())
    patterns = {indicator["pattern"] for indicator in bundle["objects"]}
    assert "[file:hashes.'MD5' = '368c710a8a0a763186eab09971d12717']" not in patterns
    assert "[domain-name:value = '-300x168.jpg']" not in patterns
    assert "[ipv4-addr:value = '1.0.0.0']" not in patterns


def test_single_org_data_accepts_retained_nonzero_org_index(tmp_path: Path) -> None:
    logs_path = tmp_path / "org_1_logs.csv"
    pd.DataFrame(
        {
            "message": ["connection to 203.0.113.10"],
            "log_id": ["source-1"],
        }
    ).to_csv(logs_path, index=False)
    high_risk_path = tmp_path / "high_risk.jsonl"
    explanations_path = tmp_path / "explanations.jsonl"
    write_jsonl(
        high_risk_path,
        [
            {
                "org_index": 1,
                "row_index": 0,
                "internal_log_id": "org_1_row_0",
                "ensemble_predicted_label": "credential access",
                "ensemble_max_risk_probability": 0.9,
            }
        ],
    )
    write_jsonl(
        explanations_path,
        [{"internal_log_id": "org_1_row_0", "top_contributions": {}}],
    )

    output_dir = tmp_path / "iocs"
    generate_ioc_outputs(
        high_risk_logs=high_risk_path,
        explanations=explanations_path,
        org_data=[logs_path],
        output_dir=output_dir,
    )

    records = [
        json.loads(line)
        for line in (output_dir / "ioc_records.jsonl").read_text().splitlines()
    ]
    assert records[0]["org_index"] == 1
    assert records[0]["internal_log_id"] == "org_1_row_0"
    assert records[0]["source_log_id"] == "source-1"


def test_explicit_org_indexes_map_noncontiguous_subset(tmp_path: Path) -> None:
    org_1 = tmp_path / "org_1.csv"
    org_4 = tmp_path / "org_4.csv"
    pd.DataFrame({"message": ["203.0.113.1"]}).to_csv(org_1, index=False)
    pd.DataFrame({"message": ["203.0.113.4"]}).to_csv(org_4, index=False)
    high_risk_path = tmp_path / "high_risk.jsonl"
    explanations_path = tmp_path / "explanations.jsonl"
    write_jsonl(
        high_risk_path,
        [
            {
                "org_index": 0,
                "row_index": 0,
                "internal_log_id": "org_0_row_0",
                "predicted_label": "credential access",
                "max_risk_probability": 0.9,
            },
            {
                "org_index": 4,
                "row_index": 0,
                "internal_log_id": "org_4_row_0",
                "predicted_label": "credential access",
                "max_risk_probability": 0.9,
            }
        ],
    )
    write_jsonl(explanations_path, [])

    output_dir = tmp_path / "iocs"
    generate_ioc_outputs(
        high_risk_logs=high_risk_path,
        explanations=explanations_path,
        org_data=[org_1, org_4],
        org_indexes=[1, 4],
        output_dir=output_dir,
    )

    records = [
        json.loads(line)
        for line in (output_dir / "ioc_records.jsonl").read_text().splitlines()
    ]
    assert {record["indicator_value"] for record in records} == {"203.0.113.4"}
    assert all(record["org_index"] == 4 for record in records)
