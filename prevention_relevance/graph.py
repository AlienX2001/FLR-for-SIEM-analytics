from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from prevention_relevance.models import GraphEdge, GraphNode
from prevention_relevance.ontology import (
    canonical_node_type,
    canonical_value,
    deterministic_edge_id,
    deterministic_node_id,
)


def _merged_attributes(
    existing: Mapping[str, Any], incoming: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if value is None:
            continue
        if key not in merged or merged[key] is None or merged[key] == "":
            merged[key] = value
        elif key == "occurrence_count":
            merged[key] = int(merged[key]) + int(value)
        elif key in {"risk_probability", "max_risk_probability"}:
            merged[key] = max(float(merged[key]), float(value))
        elif merged[key] != value:
            old_values = merged[key] if isinstance(merged[key], list) else [merged[key]]
            new_values = value if isinstance(value, list) else [value]
            merged[key] = sorted(
                {str(item) for item in old_values + new_values if item is not None}
            )
    return merged


class TypedGraph:
    """A deterministic, JSON-serializable directed property graph."""

    def __init__(self, *, graph_id: str, graph_type: str) -> None:
        self.graph_id = str(graph_id)
        self.graph_type = str(graph_type)
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self.metadata: dict[str, Any] = {}
        self._outgoing: dict[str, set[str]] = {}
        self._incoming: dict[str, set[str]] = {}

    def add_node(
        self,
        node_type: str,
        value: object,
        *,
        attributes: Mapping[str, Any] | None = None,
        increment_occurrence: bool = False,
        node_id: str | None = None,
    ) -> str:
        kind = canonical_node_type(node_type)
        canonical = canonical_value(kind, value)
        resolved_id = node_id or deterministic_node_id(kind, canonical)
        incoming = dict(attributes or {})
        if increment_occurrence:
            incoming["occurrence_count"] = 1
        current = self.nodes.get(resolved_id)
        if current is None:
            self.nodes[resolved_id] = GraphNode(
                node_id=resolved_id,
                node_type=kind,
                value=canonical,
                attributes=incoming,
            )
        else:
            if current.node_type != kind or current.value != canonical:
                raise ValueError(f"Node ID collision for {resolved_id}")
            self.nodes[resolved_id] = GraphNode(
                node_id=resolved_id,
                node_type=kind,
                value=canonical,
                attributes=_merged_attributes(current.attributes, incoming),
            )
        return resolved_id

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        increment_occurrence: bool = False,
    ) -> str:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both edge endpoints must exist before adding an edge")
        normalized_relation = str(relation).strip().lower().replace(" ", "_")
        edge_id = deterministic_edge_id(source, normalized_relation, target)
        incoming = dict(attributes or {})
        if increment_occurrence:
            incoming["occurrence_count"] = 1
        current = self.edges.get(edge_id)
        if current is None:
            self.edges[edge_id] = GraphEdge(
                edge_id=edge_id,
                source=source,
                relation=normalized_relation,
                target=target,
                attributes=incoming,
            )
            self._outgoing.setdefault(source, set()).add(edge_id)
            self._incoming.setdefault(target, set()).add(edge_id)
        else:
            self.edges[edge_id] = GraphEdge(
                edge_id=edge_id,
                source=source,
                relation=normalized_relation,
                target=target,
                attributes=_merged_attributes(current.attributes, incoming),
            )
        return edge_id

    def node_ids(self, node_type: str | None = None) -> list[str]:
        if node_type is None:
            return sorted(self.nodes)
        kind = canonical_node_type(node_type)
        return sorted(
            node_id
            for node_id, node in self.nodes.items()
            if node.node_type == kind
        )

    def incident_edges(self, node_id: str) -> list[GraphEdge]:
        edge_ids = self._outgoing.get(node_id, set()) | self._incoming.get(
            node_id, set()
        )
        return [self.edges[edge_id] for edge_id in sorted(edge_ids)]

    def neighbors(self, node_id: str) -> list[str]:
        result: set[str] = set()
        for edge in self.incident_edges(node_id):
            result.add(edge.target if edge.source == node_id else edge.source)
        return sorted(result)

    def ego_subgraph(self, root_id: str, *, hops: int) -> "TypedGraph":
        if root_id not in self.nodes:
            raise KeyError(root_id)
        selected = {root_id}
        queue: deque[tuple[str, int]] = deque([(root_id, 0)])
        while queue:
            node_id, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbor in self.neighbors(node_id):
                if neighbor in selected:
                    continue
                selected.add(neighbor)
                queue.append((neighbor, depth + 1))

        result = TypedGraph(graph_id=f"{self.graph_id}:{root_id}", graph_type="ego")
        for node_id in sorted(selected):
            node = self.nodes[node_id]
            result.nodes[node_id] = node
        selected_edges = {
            edge.edge_id
            for node_id in selected
            for edge in self.incident_edges(node_id)
            if edge.source in selected and edge.target in selected
        }
        for edge_id in sorted(selected_edges):
            edge = self.edges[edge_id]
            result.edges[edge_id] = edge
            result._outgoing.setdefault(edge.source, set()).add(edge_id)
            result._incoming.setdefault(edge.target, set()).add(edge_id)
        result.metadata = {"root_id": root_id, "hops": hops}
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "graph_id": self.graph_id,
            "graph_type": self.graph_type,
            "metadata": dict(self.metadata),
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.node_type,
                    "value": node.value,
                    "attributes": dict(node.attributes),
                }
                for node in sorted(self.nodes.values(), key=lambda item: item.node_id)
            ],
            "edges": [
                {
                    "id": edge.edge_id,
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                    "attributes": dict(edge.attributes),
                }
                for edge in sorted(self.edges.values(), key=lambda item: item.edge_id)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypedGraph":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported graph schema version")
        graph = cls(
            graph_id=str(payload["graph_id"]),
            graph_type=str(payload["graph_type"]),
        )
        graph.metadata = dict(payload.get("metadata", {}))
        for raw_node in payload.get("nodes", []):
            node = GraphNode(
                node_id=str(raw_node["id"]),
                node_type=str(raw_node["type"]),
                value=str(raw_node["value"]),
                attributes=dict(raw_node.get("attributes", {})),
            )
            graph.nodes[node.node_id] = node
        for raw_edge in payload.get("edges", []):
            edge = GraphEdge(
                edge_id=str(raw_edge["id"]),
                source=str(raw_edge["source"]),
                relation=str(raw_edge["relation"]),
                target=str(raw_edge["target"]),
                attributes=dict(raw_edge.get("attributes", {})),
            )
            if edge.source not in graph.nodes or edge.target not in graph.nodes:
                raise ValueError(f"Edge {edge.edge_id} references an unknown node")
            graph.edges[edge.edge_id] = edge
            graph._outgoing.setdefault(edge.source, set()).add(edge.edge_id)
            graph._incoming.setdefault(edge.target, set()).add(edge.edge_id)
        return graph

    def update_metadata(self, values: Mapping[str, Any]) -> None:
        self.metadata.update(values)

    def extend(self, other: "TypedGraph") -> None:
        for node in other.nodes.values():
            self.add_node(
                node.node_type,
                node.value,
                attributes=node.attributes,
                node_id=node.node_id,
            )
        for edge in other.edges.values():
            self.add_edge(
                edge.source,
                edge.relation,
                edge.target,
                attributes=edge.attributes,
            )

    def induced_subgraph(self, node_ids: Iterable[str]) -> "TypedGraph":
        selected = set(node_ids)
        result = TypedGraph(graph_id=f"{self.graph_id}:subgraph", graph_type="subgraph")
        for node_id in sorted(selected):
            if node_id in self.nodes:
                result.nodes[node_id] = self.nodes[node_id]
        for edge_id, edge in sorted(self.edges.items()):
            if edge.source in result.nodes and edge.target in result.nodes:
                result.edges[edge_id] = edge
                result._outgoing.setdefault(edge.source, set()).add(edge_id)
                result._incoming.setdefault(edge.target, set()).add(edge_id)
        return result
