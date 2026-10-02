from __future__ import annotations

from pathlib import Path

from federated_lr_pipeline.feature_schemas import SUBCATEGORY_SCHEMAS
from federated_lr_pipeline.specialized_models import _row_text

from prevention_relevance.config import load_config
from prevention_relevance.preprocessing_adapter import preprocess_change_row
from prevention_relevance.proposed_change_graph import build_proposed_change_graph


def test_adapter_reuses_federated_field_aware_preprocessing() -> None:
    row = {
        "host": "HOST-01",
        "process_name": "PowerShell.EXE",
        "process_exe": r"C:\Windows\System32\WindowsPowerShell\powershell.exe",
        "dst_ip": "203.0.113.20",
        "dst_port": "443",
        "protocol_name": "TCP",
        "label": "must-not-be-a-feature",
    }
    config = load_config()
    adapted = preprocess_change_row(row, field_mappings=config.field_mappings)
    safe_row = {key: value for key, value in row.items() if key != "label"}

    assert adapted.system_tokens == tuple(
        _row_text(safe_row, SUBCATEGORY_SCHEMAS["system"]).split()
    )
    assert adapted.network_tokens == tuple(
        _row_text(safe_row, SUBCATEGORY_SCHEMAS["network"]).split()
    )
    assert "label=must-not-be-a-feature" not in adapted.normalized_text


def test_proposed_graph_extracts_typed_nodes_and_relations(tmp_path: Path) -> None:
    source = tmp_path / "proposed.csv"
    source.write_text(
        "host,platform,process_name,process_exe,dst_ip,dst_port,protocol_name,domain\n"
        "HOST-01,Windows,powershell.exe,C:\\\\Windows\\\\powershell.exe,"
        "203.0.113.20,443,TCP,Updates.Example.COM.\n",
        encoding="utf-8",
    )
    graph = build_proposed_change_graph(
        [(source, None)],
        config=load_config(),
        environment_id="planned",
    )

    assert {graph.nodes[node_id].value for node_id in graph.node_ids("domain")} == {
        "updates.example.com"
    }
    assert {graph.nodes[node_id].value for node_id in graph.node_ids("port")} == {
        "443"
    }
    relations = {edge.relation for edge in graph.edges.values()}
    assert {"runs", "connects_to", "uses_protocol", "uses_port"} <= relations
    assert graph.metadata["row_count"] == 1


def test_conflicting_aliases_are_omitted_conservatively(tmp_path: Path) -> None:
    source = tmp_path / "conflict.csv"
    source.write_text(
        "dst_ip,destination_ip,protocol_name\n203.0.113.1,198.51.100.2,tcp\n",
        encoding="utf-8",
    )
    graph = build_proposed_change_graph(
        [(source, "network")],
        config=load_config(),
        environment_id="planned",
    )
    assert graph.node_ids("ip_address") == []
    assert graph.metadata["conflicting_field_count"] == 1
