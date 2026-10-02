from __future__ import annotations

import ipaddress
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from prevention_relevance.config import RelevanceConfig
from prevention_relevance.graph import TypedGraph
from prevention_relevance.models import (
    EdgeMatchEvidence,
    GraphEdge,
    GraphNode,
    MatchEvidence,
    SimilarityBreakdown,
)
from prevention_relevance.ontology import (
    ADMINISTRATIVE_RELATIONS,
    DISTINCTIVE_NODE_TYPES,
    MATCHABLE_NODE_TYPES,
    basename,
)


@dataclass(frozen=True)
class EnvironmentIndex:
    nodes_by_key: Mapping[tuple[str, str], tuple[GraphNode, ...]]
    nodes_by_type: Mapping[str, tuple[GraphNode, ...]]
    process_nodes_by_basename: Mapping[str, tuple[GraphNode, ...]]
    url_nodes_by_hostname: Mapping[str, tuple[GraphNode, ...]]
    edges_by_relation: Mapping[str, tuple[GraphEdge, ...]]


def build_environment_index(graph: TypedGraph) -> EnvironmentIndex:
    by_key: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
    by_type: dict[str, list[GraphNode]] = defaultdict(list)
    by_basename: dict[str, list[GraphNode]] = defaultdict(list)
    by_url_hostname: dict[str, list[GraphNode]] = defaultdict(list)
    by_relation: dict[str, list[GraphEdge]] = defaultdict(list)
    for node in _matchable_nodes(graph):
        by_key[(node.node_type, node.value)].append(node)
        by_type[node.node_type].append(node)
        if node.node_type in {"process", "process_path"}:
            by_basename[basename(node.value)].append(node)
        if node.node_type == "url":
            try:
                hostname = (urlsplit(node.value).hostname or "").lower().rstrip(".")
            except ValueError:
                hostname = ""
            if hostname:
                by_url_hostname[hostname].append(node)
    for edge in graph.edges.values():
        by_relation[edge.relation].append(edge)

    def frozen(values):  # noqa: ANN001, ANN202
        return {
            key: tuple(sorted(items, key=lambda item: item.node_id))
            for key, items in values.items()
        }

    return EnvironmentIndex(
        nodes_by_key=frozen(by_key),
        nodes_by_type=frozen(by_type),
        process_nodes_by_basename=frozen(by_basename),
        url_nodes_by_hostname=frozen(by_url_hostname),
        edges_by_relation={
            key: tuple(sorted(items, key=lambda item: item.edge_id))
            for key, items in by_relation.items()
        },
    )


def _relation_equivalent(left: str, right: str, config: RelevanceConfig) -> bool:
    if left == right:
        return True
    for canonical, aliases in config.relation_aliases.items():
        group = {canonical, *aliases}
        if left in group and right in group:
            return True
    return False


def node_similarity(
    candidate: GraphNode,
    environment: GraphNode,
    config: RelevanceConfig,
) -> float:
    exact = float(config.match_scores["exact"])
    if candidate.node_type == environment.node_type and candidate.value == environment.value:
        return exact

    if candidate.node_type == "ip_address" and environment.node_type == "network":
        try:
            if ipaddress.ip_address(candidate.value) in ipaddress.ip_network(
                environment.value, strict=False
            ):
                return float(config.match_scores["network_membership"])
        except ValueError:
            return 0.0
    if candidate.node_type == "network" and environment.node_type == "ip_address":
        try:
            if ipaddress.ip_address(environment.value) in ipaddress.ip_network(
                candidate.value, strict=False
            ):
                return float(config.match_scores["network_membership"])
        except ValueError:
            return 0.0

    process_types = {candidate.node_type, environment.node_type}
    if process_types <= {"process", "process_path"} and (
        candidate.node_type in {"process", "process_path"}
        and environment.node_type in {"process", "process_path"}
    ):
        if basename(candidate.value) == basename(environment.value):
            return float(config.match_scores["process_basename"])

    if {candidate.node_type, environment.node_type} == {"domain", "url"}:
        url_node = candidate if candidate.node_type == "url" else environment
        domain_node = candidate if candidate.node_type == "domain" else environment
        try:
            if (urlsplit(url_node.value).hostname or "").lower().rstrip(".") == domain_node.value:
                return float(config.match_scores["url_host"])
        except ValueError:
            return 0.0
    return 0.0


def _matchable_nodes(graph: TypedGraph) -> list[GraphNode]:
    return [
        node
        for node in graph.nodes.values()
        if node.node_type in MATCHABLE_NODE_TYPES
    ]


def candidate_document_frequencies(
    candidate_graphs: Iterable[TypedGraph],
) -> Counter[tuple[str, str]]:
    frequencies: Counter[tuple[str, str]] = Counter()
    for graph in candidate_graphs:
        frequencies.update(
            {(node.node_type, node.value) for node in _matchable_nodes(graph)}
        )
    return frequencies


def _node_weight(
    node: GraphNode,
    *,
    frequencies: Mapping[tuple[str, str], int],
    candidate_count: int,
    config: RelevanceConfig,
) -> float:
    base = float(config.type_weights.get(node.node_type, 1.0))
    frequency = int(frequencies.get((node.node_type, node.value), 1))
    inverse_frequency = math.log((candidate_count + 1.0) / (frequency + 1.0)) + 1.0
    return base * inverse_frequency


def _best_node_matches(
    candidate_nodes: Sequence[GraphNode],
    environment_index: EnvironmentIndex,
    *,
    frequencies: Mapping[tuple[str, str], int],
    candidate_count: int,
    config: RelevanceConfig,
) -> tuple[dict[str, tuple[str, float]], tuple[MatchEvidence, ...], float]:
    mapping: dict[str, tuple[str, float]] = {}
    evidence: list[MatchEvidence] = []
    weighted_score = 0.0
    total_weight = 0.0
    for candidate in candidate_nodes:
        weight = _node_weight(
            candidate,
            frequencies=frequencies,
            candidate_count=candidate_count,
            config=config,
        )
        total_weight += weight
        matches = _indexed_node_matches(candidate, environment_index, config)
        best_node, best_score = matches[0] if matches else (None, 0.0)
        weighted_score += weight * best_score
        if best_node is not None and best_score > 0:
            mapping[candidate.node_id] = (best_node.node_id, best_score)
            evidence.append(
                MatchEvidence(
                    candidate_node_id=candidate.node_id,
                    candidate_type=candidate.node_type,
                    candidate_value=candidate.value,
                    environment_node_id=best_node.node_id,
                    environment_value=best_node.value,
                    score=best_score,
                    weight=weight,
                )
            )
    return mapping, tuple(evidence), weighted_score / total_weight if total_weight else 0.0


def _indexed_node_matches(
    candidate: GraphNode,
    environment_index: EnvironmentIndex,
    config: RelevanceConfig,
) -> tuple[tuple[GraphNode, float], ...]:
    possible: dict[str, GraphNode] = {
        node.node_id: node
        for node in environment_index.nodes_by_key.get(
            (candidate.node_type, candidate.value), ()
        )
    }
    if candidate.node_type == "ip_address":
        for node in environment_index.nodes_by_type.get("network", ()):
            possible[node.node_id] = node
    elif candidate.node_type == "network":
        for node in environment_index.nodes_by_type.get("ip_address", ()):
            possible[node.node_id] = node
    elif candidate.node_type in {"process", "process_path"}:
        for node in environment_index.process_nodes_by_basename.get(
            basename(candidate.value), ()
        ):
            possible[node.node_id] = node
    elif candidate.node_type == "domain":
        for node in environment_index.url_nodes_by_hostname.get(candidate.value, ()):
            possible[node.node_id] = node
    elif candidate.node_type == "url":
        try:
            hostname = (urlsplit(candidate.value).hostname or "").lower().rstrip(".")
        except ValueError:
            hostname = ""
        for node in environment_index.nodes_by_key.get(("domain", hostname), ()):
            possible[node.node_id] = node
    matches = [
        (node, node_similarity(candidate, node, config))
        for node in possible.values()
    ]
    return tuple(
        sorted(
            (item for item in matches if item[1] > 0),
            key=lambda item: (-item[1], item[0].node_id),
        )
    )


def _direct_observable_nodes(candidate: TypedGraph, indicator_id: str) -> list[GraphNode]:
    return [
        candidate.nodes[edge.target]
        for edge in candidate.edges.values()
        if edge.source == indicator_id
        and edge.relation == "indicates"
        and edge.target in candidate.nodes
    ]


def _context_nodes(candidate: TypedGraph) -> list[GraphNode]:
    return [
        node
        for node in candidate.nodes.values()
        if bool(node.attributes.get("applicability_requirement", False))
    ]


def _scored_edges(candidate: TypedGraph) -> list[GraphEdge]:
    return [
        edge
        for edge in candidate.edges.values()
        if edge.relation not in ADMINISTRATIVE_RELATIONS
        and candidate.nodes[edge.source].node_type in MATCHABLE_NODE_TYPES
        and candidate.nodes[edge.target].node_type in MATCHABLE_NODE_TYPES
    ]


def _edge_score(
    candidate: TypedGraph,
    environment: TypedGraph,
    *,
    environment_index: EnvironmentIndex,
    config: RelevanceConfig,
    frequencies: Mapping[tuple[str, str], int],
    candidate_count: int,
) -> tuple[float, tuple[EdgeMatchEvidence, ...]]:
    evidence: list[EdgeMatchEvidence] = []
    weighted_score = 0.0
    total_weight = 0.0
    for edge in _scored_edges(candidate):
        source_node = candidate.nodes[edge.source]
        target_node = candidate.nodes[edge.target]
        edge_weight = (
            _node_weight(
                source_node,
                frequencies=frequencies,
                candidate_count=candidate_count,
                config=config,
            )
            + _node_weight(
                target_node,
                frequencies=frequencies,
                candidate_count=candidate_count,
                config=config,
            )
        ) / 2.0
        total_weight += edge_weight
        source_matches = _indexed_node_matches(source_node, environment_index, config)
        target_matches = _indexed_node_matches(target_node, environment_index, config)
        possible_edges = {
            environment_edge.edge_id: environment_edge
            for relation, edges in environment_index.edges_by_relation.items()
            if _relation_equivalent(edge.relation, relation, config)
            for environment_edge in edges
        }
        best_edge: GraphEdge | None = None
        best_score = 0.0
        source_scores = {node.node_id: score for node, score in source_matches}
        target_scores = {node.node_id: score for node, score in target_matches}
        for environment_edge in possible_edges.values():
            score = source_scores.get(environment_edge.source, 0.0) * target_scores.get(
                environment_edge.target, 0.0
            )
            if score > best_score or (
                score == best_score
                and score > 0
                and best_edge is not None
                and environment_edge.edge_id < best_edge.edge_id
            ):
                best_score = score
                best_edge = environment_edge
        weighted_score += edge_weight * best_score
        if best_edge is not None and best_score > 0:
            evidence.append(
                EdgeMatchEvidence(
                    candidate_edge_id=edge.edge_id,
                    candidate_relation=edge.relation,
                    environment_edge_id=best_edge.edge_id,
                    score=best_score,
                    weight=edge_weight,
                )
            )
    return (
        weighted_score / total_weight if total_weight else 0.0,
        tuple(evidence),
    )


def _is_generic(node: GraphNode, config: RelevanceConfig) -> bool:
    configured = config.generic_values.get(node.node_type, ())
    return node.value in {str(item).strip().lower() for item in configured}


def score_indicator(
    environment: TypedGraph,
    candidate: TypedGraph,
    *,
    environment_index: EnvironmentIndex | None = None,
    indicator_id: str,
    frequencies: Mapping[tuple[str, str], int],
    candidate_count: int,
    config: RelevanceConfig,
) -> SimilarityBreakdown:
    resolved_environment_index = environment_index or build_environment_index(environment)
    direct_nodes = _direct_observable_nodes(candidate, indicator_id)
    direct_matches = tuple(
        node.node_id for node in direct_nodes if node.node_id in environment.nodes
    )
    direct_score = 1.0 if direct_matches else 0.0

    context_nodes = _context_nodes(candidate)
    excluded = {node.node_id for node in direct_nodes + context_nodes}
    candidate_nodes = [
        node for node in _matchable_nodes(candidate) if node.node_id not in excluded
    ]
    _, node_evidence, node_score = _best_node_matches(
        candidate_nodes,
        resolved_environment_index,
        frequencies=frequencies,
        candidate_count=candidate_count,
        config=config,
    )
    _, context_evidence, context_score = _best_node_matches(
        context_nodes,
        resolved_environment_index,
        frequencies=frequencies,
        candidate_count=candidate_count,
        config=config,
    )
    edge_score, edge_evidence = _edge_score(
        candidate,
        environment,
        environment_index=resolved_environment_index,
        config=config,
        frequencies=frequencies,
        candidate_count=candidate_count,
    )

    all_evidence = tuple(sorted(node_evidence + context_evidence, key=lambda item: (-item.weight * item.score, item.candidate_node_id)))
    distinctive_matches = sum(
        item.candidate_type in DISTINCTIVE_NODE_TYPES
        and not _is_generic(candidate.nodes[item.candidate_node_id], config)
        for item in all_evidence
    )

    candidate_platforms = [node for node in context_nodes if node.node_type == "platform"]
    environment_platforms = environment.node_ids("platform")
    incompatibility_penalty = 0.0
    if candidate_platforms and environment_platforms:
        platform_matches = [
            item for item in context_evidence if item.candidate_type == "platform"
        ]
        if not platform_matches:
            incompatibility_penalty = 1.0

    eligible = bool(direct_score) or (
        distinctive_matches >= config.min_distinctive_matches
        and (edge_score > 0 or context_score > 0)
    )
    weights = config.component_weights
    relevance = (
        weights["direct"] * direct_score
        + weights["node"] * node_score
        + weights["edge"] * edge_score
        + weights["context"] * context_score
        - config.incompatibility_weight * incompatibility_penalty
    )
    if not eligible:
        relevance = 0.0
    relevance = min(1.0, max(0.0, relevance))
    matched_candidate_ids = {item.candidate_node_id for item in all_evidence}
    unmatched_nodes = tuple(
        f"{node.node_type}:{node.value}"
        for node in sorted(
            candidate_nodes + context_nodes,
            key=lambda item: (item.node_type, item.value),
        )
        if node.node_id not in matched_candidate_ids
    )
    return SimilarityBreakdown(
        indicator_id=indicator_id,
        direct_score=direct_score,
        node_score=node_score,
        edge_score=edge_score,
        context_score=context_score,
        incompatibility_penalty=incompatibility_penalty,
        relevance_score=relevance,
        eligible=eligible,
        distinctive_matches=distinctive_matches,
        matched_nodes=all_evidence,
        matched_edges=edge_evidence,
        unmatched_nodes=unmatched_nodes,
        direct_matches=direct_matches,
    )
