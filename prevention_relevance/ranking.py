from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace

from prevention_relevance.config import RelevanceConfig
from prevention_relevance.graph import TypedGraph
from prevention_relevance.models import RankingRecord
from prevention_relevance.similarity import (
    build_environment_index,
    candidate_document_frequencies,
    score_indicator,
)


def _scalar(value: object, default: str = "") -> str:
    if isinstance(value, list):
        return str(value[0]) if value else default
    return default if value is None else str(value)


def _attack_label(graph: TypedGraph, indicator_id: str) -> str | None:
    labels = sorted(
        graph.nodes[edge.target].value
        for edge in graph.incident_edges(indicator_id)
        if edge.source == indicator_id
        and edge.relation == "associated_with"
        and graph.nodes[edge.target].node_type == "attack_class"
    )
    return labels[0] if labels else None


def _decision(score: float, eligible: bool, config: RelevanceConfig) -> str:
    if not eligible:
        return "not_selected"
    if score >= config.high_threshold:
        return "relevant"
    if score >= config.review_threshold:
        return "analyst_review"
    return "not_selected"


def _explanation(breakdown) -> dict[str, object]:  # noqa: ANN001
    return {
        "direct_matches": list(breakdown.direct_matches),
        "matched_nodes": [asdict(item) for item in breakdown.matched_nodes],
        "matched_edges": [asdict(item) for item in breakdown.matched_edges],
        "unmatched_nodes": list(breakdown.unmatched_nodes),
        "eligibility": {
            "eligible": breakdown.eligible,
            "distinctive_matches": breakdown.distinctive_matches,
        },
    }


def rank_indicators(
    environment: TypedGraph,
    historical: TypedGraph,
    *,
    config: RelevanceConfig,
    limit: int | None = None,
) -> list[RankingRecord]:
    indicator_ids = historical.node_ids("indicator")
    if config.require_validated:
        indicator_ids = [
            indicator_id
            for indicator_id in indicator_ids
            if _scalar(
                historical.nodes[indicator_id].attributes.get("validation_status"),
                "unreviewed",
            )
            == "validated"
        ]
    frequencies: Counter[tuple[str, str]] = candidate_document_frequencies(
        historical.ego_subgraph(indicator_id, hops=config.ego_hops)
        for indicator_id in indicator_ids
    )
    candidate_count = max(1, len(indicator_ids))
    environment_index = build_environment_index(environment)
    records: list[RankingRecord] = []
    for indicator_id in indicator_ids:
        candidate = historical.ego_subgraph(indicator_id, hops=config.ego_hops)
        indicator = historical.nodes[indicator_id]
        breakdown = score_indicator(
            environment,
            candidate,
            indicator_id=indicator_id,
            environment_index=environment_index,
            frequencies=frequencies,
            candidate_count=candidate_count,
            config=config,
        )
        risk_raw = indicator.attributes.get("risk_probability")
        risk = float(risk_raw) if isinstance(risk_raw, (int, float)) else None
        validation_status = _scalar(
            indicator.attributes.get("validation_status"), "unreviewed"
        )
        validation_factor = {
            "validated": 1.0,
            "unreviewed": 0.75,
            "rejected": 0.0,
        }.get(validation_status, 0.75)
        confidence_factor = 1.0 if risk is None else 0.5 + 0.5 * min(1.0, max(0.0, risk))
        decision = (
            "not_selected"
            if validation_status == "rejected"
            else _decision(breakdown.relevance_score, breakdown.eligible, config)
        )
        records.append(
            RankingRecord(
                rank=0,
                indicator_id=indicator_id,
                indicator_name=_scalar(indicator.attributes.get("name"), indicator.value),
                indicator_pattern=_scalar(indicator.attributes.get("pattern")),
                predicted_label=_attack_label(historical, indicator_id),
                validation_status=validation_status,
                relevance_decision=decision,
                relevance_score=breakdown.relevance_score,
                priority_score=(
                    breakdown.relevance_score * validation_factor * confidence_factor
                ),
                risk_probability=risk,
                direct_score=breakdown.direct_score,
                node_score=breakdown.node_score,
                edge_score=breakdown.edge_score,
                context_score=breakdown.context_score,
                incompatibility_penalty=breakdown.incompatibility_penalty,
                eligible=breakdown.eligible,
                distinctive_matches=breakdown.distinctive_matches,
                explanation=_explanation(breakdown),
            )
        )
    records.sort(
        key=lambda item: (
            -item.priority_score,
            -item.relevance_score,
            item.indicator_id,
        )
    )
    ranked = [replace(record, rank=index) for index, record in enumerate(records, start=1)]
    return ranked if limit is None else ranked[:limit]
