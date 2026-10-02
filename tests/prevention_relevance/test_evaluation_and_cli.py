from __future__ import annotations

import json
from pathlib import Path

import pytest

from prevention_relevance.evaluation import compute_relevance_metrics
from prevention_relevance.models import RankingRecord
from prevention_relevance.run import main


def _record(indicator_id: str, rank: int, score: float, decision: str) -> RankingRecord:
    return RankingRecord(
        rank=rank,
        indicator_id=indicator_id,
        indicator_name=indicator_id,
        indicator_pattern="",
        predicted_label=None,
        validation_status="validated",
        relevance_decision=decision,
        relevance_score=score,
        priority_score=score,
        risk_probability=None,
        direct_score=score,
        node_score=0.0,
        edge_score=0.0,
        context_score=0.0,
        incompatibility_penalty=0.0,
        eligible=score > 0,
        distinctive_matches=1,
        explanation={},
    )


def test_evaluation_reports_classification_and_ranking_metrics() -> None:
    rankings = [
        _record("indicator--1", 1, 0.9, "relevant"),
        _record("indicator--2", 2, 0.1, "not_selected"),
    ]
    metrics = compute_relevance_metrics(
        rankings,
        {"indicator--1": 1, "indicator--2": 0},
        top_k=1,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_1"] == 1.0
    assert metrics["average_precision"] == 1.0


def test_evaluation_rejects_unknown_indicator_label() -> None:
    with pytest.raises(ValueError, match="unknown Indicator"):
        compute_relevance_metrics(
            [_record("indicator--1", 1, 0.9, "relevant")],
            {"indicator--missing": 1},
            top_k=1,
        )


def test_cli_runs_complete_example(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    examples = root / "prevention_relevance" / "examples"
    output = tmp_path / "output"
    exit_code = main(
        [
            "--proposed-system-logs",
            str(examples / "proposed_system.csv"),
            "--proposed-network-logs",
            str(examples / "proposed_network.csv"),
            "--ioc-bundle",
            str(examples / "example_ioc_bundle.json"),
            "--config",
            str(examples / "similarity.yaml"),
            "--relevance-labels",
            str(examples / "relevance_labels.csv"),
            "--output-dir",
            str(output),
            "--top-k",
            "2",
        ]
    )
    assert exit_code == 0
    expected = {
        "proposed_change_graph.json",
        "historical_ioc_graph.json",
        "ioc_relevance_rankings.csv",
        "ioc_relevance_rankings.jsonl",
        "prevention_explanations.jsonl",
        "prevention_summary.json",
        "prevention_metrics.json",
        "prevention_run_config.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    metrics = json.loads((output / "prevention_metrics.json").read_text())
    assert metrics["labeled_indicator_count"] == 2
    rankings = [
        json.loads(line)
        for line in (output / "ioc_relevance_rankings.jsonl").read_text().splitlines()
    ]
    assert rankings[0]["indicator_pattern"] == (
        "[domain-name:value = 'updates.example.com']"
    )
