from __future__ import annotations

import json
from pathlib import Path

from prevention_relevance.config import load_config
from prevention_relevance.stix_graph import (
    build_historical_ioc_graph,
    iter_stix_objects,
    parse_indicator_pattern,
)


def _indicator(indicator_id: str, pattern: str) -> dict[str, object]:
    return {
        "type": "indicator",
        "id": indicator_id,
        "name": "test",
        "pattern": pattern,
        "x_validation_status": "validated",
        "x_federated_lr_predicted_label": "credential access",
    }


def test_exact_stix_patterns_are_typed() -> None:
    assert parse_indicator_pattern("[ipv4-addr:value = '203.0.113.20']") == (
        "ip_address",
        "203.0.113.20",
    )
    assert parse_indicator_pattern(
        "[file:hashes.'SHA-256' = '" + "a" * 64 + "']"
    ) == ("sha256", "a" * 64)
    assert parse_indicator_pattern("[domain-name:value MATCHES '.*']") is None


def test_bundle_reader_streams_objects(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    objects = [
        _indicator("indicator--00000000-0000-4000-8000-000000000001", "[domain-name:value = 'a.example']"),
        _indicator("indicator--00000000-0000-4000-8000-000000000002", "[domain-name:value = 'b.example']"),
    ]
    path.write_text(json.dumps({"type": "bundle", "objects": objects}), encoding="utf-8")
    assert [item["id"] for item in iter_stix_objects(path)] == [
        item["id"] for item in objects
    ]


def test_graph_enriches_ioc_with_event_evidence(tmp_path: Path) -> None:
    indicator_id = "indicator--00000000-0000-4000-8000-000000000001"
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "type": "bundle",
                "objects": [
                    _indicator(
                        indicator_id,
                        "[domain-name:value = 'updates.example.com']",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "indicator_id": indicator_id,
                "internal_log_id": "org_0_row_1",
                "predicted_label": "credential access",
                "evidence_by_label_subcategory": {
                    "credential access": {
                        "system": [{"token": "process_name=powershell.exe"}]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    graph = build_historical_ioc_graph(
        bundle,
        config=load_config(),
        ioc_records_path=records,
    )

    ego = graph.ego_subgraph(indicator_id, hops=2)
    assert "process_name=powershell.exe" in {
        node.value for node in ego.nodes.values() if node.node_type == "feature"
    }
    assert graph.metadata["ioc_record_count"] == 1
