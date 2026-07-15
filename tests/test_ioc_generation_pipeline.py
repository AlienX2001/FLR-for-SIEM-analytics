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
