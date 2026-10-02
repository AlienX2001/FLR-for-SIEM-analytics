from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    value: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    source: str
    relation: str
    target: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessedChangeRow:
    canonical_fields: Mapping[str, str]
    normalized_text: str
    system_tokens: tuple[str, ...]
    network_tokens: tuple[str, ...]
    identity_tokens: tuple[str, ...]
    llm_tokens: tuple[str, ...]
    cloud_tokens: tuple[str, ...]
    conflicts: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchEvidence:
    candidate_node_id: str
    candidate_type: str
    candidate_value: str
    environment_node_id: str
    environment_value: str
    score: float
    weight: float


@dataclass(frozen=True)
class EdgeMatchEvidence:
    candidate_edge_id: str
    candidate_relation: str
    environment_edge_id: str
    score: float
    weight: float


@dataclass(frozen=True)
class SimilarityBreakdown:
    indicator_id: str
    direct_score: float
    node_score: float
    edge_score: float
    context_score: float
    incompatibility_penalty: float
    relevance_score: float
    eligible: bool
    distinctive_matches: int
    matched_nodes: tuple[MatchEvidence, ...] = ()
    matched_edges: tuple[EdgeMatchEvidence, ...] = ()
    unmatched_nodes: tuple[str, ...] = ()
    direct_matches: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankingRecord:
    rank: int
    indicator_id: str
    indicator_name: str
    indicator_pattern: str
    predicted_label: str | None
    validation_status: str
    relevance_decision: str
    relevance_score: float
    priority_score: float
    risk_probability: float | None
    direct_score: float
    node_score: float
    edge_score: float
    context_score: float
    incompatibility_penalty: float
    eligible: bool
    distinctive_matches: int
    explanation: Mapping[str, Any]

