from __future__ import annotations

from prevention_relevance.config import load_config
from prevention_relevance.graph import TypedGraph
from prevention_relevance.ranking import rank_indicators


def _historical_indicator(pattern_value: str) -> tuple[TypedGraph, str]:
    graph = TypedGraph(graph_id="history", graph_type="ioc_knowledge")
    indicator_id = "indicator--00000000-0000-4000-8000-000000000001"
    graph.add_node(
        "indicator",
        indicator_id,
        node_id=indicator_id,
        attributes={
            "name": "test indicator",
            "pattern": f"[domain-name:value = '{pattern_value}']",
            "validation_status": "validated",
        },
    )
    observable = graph.add_node("domain", pattern_value)
    graph.add_edge(indicator_id, "indicates", observable)
    return graph, indicator_id


def test_exact_observable_match_is_relevant() -> None:
    environment = TypedGraph(graph_id="new", graph_type="proposed_change")
    environment.add_node("domain", "updates.example.com")
    historical, _ = _historical_indicator("updates.example.com")

    record = rank_indicators(environment, historical, config=load_config())[0]
    assert record.direct_score == 1.0
    assert record.relevance_score == 0.45
    assert record.relevance_decision == "relevant"


def test_generic_evidence_alone_cannot_pass_eligibility_gate() -> None:
    environment = TypedGraph(graph_id="new", graph_type="proposed_change")
    environment.add_node("protocol", "tcp")
    historical = TypedGraph(graph_id="history", graph_type="ioc_knowledge")
    indicator_id = "indicator--00000000-0000-4000-8000-000000000001"
    historical.add_node(
        "indicator",
        indicator_id,
        node_id=indicator_id,
        attributes={"validation_status": "validated"},
    )
    protocol = historical.add_node("protocol", "tcp")
    historical.add_edge(indicator_id, "associated_with", protocol)

    record = rank_indicators(environment, historical, config=load_config())[0]
    assert record.eligible is False
    assert record.relevance_score == 0.0


def test_matching_typed_relation_contributes_to_score() -> None:
    environment = TypedGraph(graph_id="new", graph_type="proposed_change")
    process = environment.add_node("process", "powershell.exe")
    file_path = environment.add_node("file_path", r"C:\Finance\quarterly.xlsx")
    environment.add_edge(process, "accesses", file_path)

    historical = TypedGraph(graph_id="history", graph_type="ioc_knowledge")
    indicator_id = "indicator--00000000-0000-4000-8000-000000000001"
    historical.add_node(
        "indicator",
        indicator_id,
        node_id=indicator_id,
        attributes={"validation_status": "validated"},
    )
    event = historical.add_node("event", "event-1")
    candidate_process = historical.add_node("process", "powershell.exe")
    candidate_file = historical.add_node("file_path", r"C:\Finance\quarterly.xlsx")
    historical.add_edge(indicator_id, "derived_from", event)
    historical.add_edge(event, "supported_by", candidate_process)
    historical.add_edge(event, "supported_by", candidate_file)
    historical.add_edge(candidate_process, "accesses", candidate_file)

    record = rank_indicators(environment, historical, config=load_config())[0]
    assert record.node_score == 1.0
    assert record.edge_score == 1.0
    assert record.distinctive_matches == 2
    assert record.eligible is True
    assert record.relevance_score == 0.4
