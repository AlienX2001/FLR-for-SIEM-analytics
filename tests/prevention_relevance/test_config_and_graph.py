from __future__ import annotations

from pathlib import Path

import pytest

from prevention_relevance.config import load_config
from prevention_relevance.graph import TypedGraph


def test_graph_ids_and_serialization_are_deterministic() -> None:
    graph = TypedGraph(graph_id="new", graph_type="proposed_change")
    first = graph.add_node("domain", "Example.COM.")
    second = graph.add_node("domain-name", "example.com")
    assert first == second
    environment = graph.add_node("environment", "new")
    edge = graph.add_edge(environment, "contains", first)

    restored = TypedGraph.from_dict(graph.to_dict())
    assert restored.nodes[first].value == "example.com"
    assert restored.edges[edge].target == first


def test_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("version: 1\nversion: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_config(path)


def test_config_normalizes_component_weights_and_validates_thresholds(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        """
version: 1
similarity:
  component_weights: {direct: 2, node: 1, edge: 1, context: 0}
""",
        encoding="utf-8",
    )
    config = load_config(valid)
    assert config.component_weights == {
        "direct": 0.5,
        "node": 0.25,
        "edge": 0.25,
        "context": 0.0,
    }

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "version: 1\nsimilarity: {review_threshold: 0.8, high_threshold: 0.4}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="thresholds"):
        load_config(invalid)
