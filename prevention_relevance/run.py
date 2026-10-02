from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from prevention_relevance.config import load_config
from prevention_relevance.evaluation import (
    compute_relevance_metrics,
    load_relevance_labels,
)
from prevention_relevance.proposed_change_graph import build_proposed_change_graph
from prevention_relevance.ranking import rank_indicators
from prevention_relevance.serialization import (
    ranking_to_csv_row,
    ranking_to_dict,
    write_csv,
    write_graph_json,
    write_json,
    write_jsonl,
)
from prevention_relevance.stix_graph import build_historical_ioc_graph

LOGGER = logging.getLogger(__name__)

RANKING_COLUMNS = [
    "rank",
    "indicator_id",
    "indicator_name",
    "indicator_pattern",
    "predicted_label",
    "validation_status",
    "relevance_decision",
    "relevance_score",
    "priority_score",
    "risk_probability",
    "direct_score",
    "node_score",
    "edge_score",
    "context_score",
    "incompatibility_penalty",
    "eligible",
    "distinctive_matches",
    "explanation",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank historical STIX Indicators against a graph representation of a "
            "proposed environment change."
        )
    )
    parser.add_argument(
        "--proposed-logs",
        nargs="+",
        type=Path,
        help="Proposed-change CSVs without an explicit system/network category hint.",
    )
    parser.add_argument(
        "--proposed-system-logs",
        nargs="+",
        type=Path,
        help="Proposed-change CSVs representing system telemetry.",
    )
    parser.add_argument(
        "--proposed-network-logs",
        nargs="+",
        type=Path,
        help="Proposed-change CSVs representing network telemetry.",
    )
    parser.add_argument("--ioc-bundle", required=True, type=Path)
    parser.add_argument(
        "--ioc-records",
        type=Path,
        help="Optional ioc_generation/ioc_records.jsonl context.",
    )
    parser.add_argument(
        "--explanations",
        type=Path,
        help="Optional federated inference explanations.jsonl context.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--relevance-labels",
        type=Path,
        help="Optional analyst labels CSV with indicator_id,relevant columns.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--environment-id", default="proposed_environment")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument(
        "--assume-validated",
        action="store_true",
        help="Treat all input Indicators as analyst validated.",
    )
    parser.add_argument(
        "--omit-historical-graph-output",
        action="store_true",
        help="Do not serialize the potentially large historical graph.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def _log_sources(args: argparse.Namespace) -> list[tuple[Path, str | None]]:
    sources: list[tuple[Path, str | None]] = []
    sources.extend((path, None) for path in (args.proposed_logs or []))
    sources.extend((path, "system") for path in (args.proposed_system_logs or []))
    sources.extend((path, "network") for path in (args.proposed_network_logs or []))
    return sources


def _validate_args(args: argparse.Namespace) -> list[tuple[Path, str | None]]:
    sources = _log_sources(args)
    if not sources:
        raise ValueError(
            "At least one of --proposed-logs, --proposed-system-logs, or "
            "--proposed-network-logs is required"
        )
    if len(args.delimiter) != 1:
        raise ValueError("--delimiter must be exactly one character")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    paths = [path for path, _ in sources]
    paths.extend(
        path
        for path in (
            args.ioc_bundle,
            args.ioc_records,
            args.explanations,
            args.config,
            args.relevance_labels,
        )
        if path is not None
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Input file(s) do not exist: {', '.join(missing)}")
    return sources


def _summary(rankings, historical, proposed, *, top_k: int) -> dict[str, object]:  # noqa: ANN001
    decisions = Counter(record.relevance_decision for record in rankings)
    validation = Counter(record.validation_status for record in rankings)
    selected = [
        record
        for record in rankings
        if record.relevance_decision in {"relevant", "analyst_review"}
    ]
    return {
        "proposed_graph": {
            "node_count": len(proposed.nodes),
            "edge_count": len(proposed.edges),
            "row_count": proposed.metadata.get("row_count", 0),
        },
        "historical_graph": {
            "node_count": len(historical.nodes),
            "edge_count": len(historical.edges),
            "indicator_count": historical.metadata.get("indicator_count", 0),
        },
        "ranking": {
            "ranked_indicator_count": len(rankings),
            "decision_counts": dict(sorted(decisions.items())),
            "validation_counts": dict(sorted(validation.items())),
            "selected_count": len(selected),
            "top_k": top_k,
            "top_indicators": [
                {
                    "rank": record.rank,
                    "indicator_id": record.indicator_id,
                    "decision": record.relevance_decision,
                    "relevance_score": record.relevance_score,
                    "priority_score": record.priority_score,
                }
                for record in rankings[:top_k]
            ],
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    sources = _validate_args(args)
    config = load_config(args.config)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building proposed-change graph from %s CSV file(s)", len(sources))
    proposed = build_proposed_change_graph(
        sources,
        config=config,
        environment_id=args.environment_id,
        encoding=args.encoding,
        delimiter=args.delimiter,
    )
    LOGGER.info(
        "Proposed-change graph contains %s nodes and %s edges",
        len(proposed.nodes),
        len(proposed.edges),
    )

    LOGGER.info("Building historical IoC graph from STIX")
    historical = build_historical_ioc_graph(
        args.ioc_bundle,
        config=config,
        ioc_records_path=args.ioc_records,
        explanations_path=args.explanations,
        assume_validated=args.assume_validated,
    )
    LOGGER.info("Scoring %s Indicators", historical.metadata.get("indicator_count", 0))
    rankings = rank_indicators(proposed, historical, config=config)

    write_graph_json(output_dir / "proposed_change_graph.json", proposed)
    if not args.omit_historical_graph_output:
        write_graph_json(output_dir / "historical_ioc_graph.json", historical)
    write_jsonl(
        output_dir / "ioc_relevance_rankings.jsonl",
        (ranking_to_dict(record) for record in rankings),
    )
    write_csv(
        output_dir / "ioc_relevance_rankings.csv",
        (ranking_to_csv_row(record) for record in rankings),
        fieldnames=RANKING_COLUMNS,
    )
    write_jsonl(
        output_dir / "prevention_explanations.jsonl",
        (
            {
                "indicator_id": record.indicator_id,
                "rank": record.rank,
                "relevance_decision": record.relevance_decision,
                "relevance_score": record.relevance_score,
                "priority_score": record.priority_score,
                "predicted_label": record.predicted_label,
                "explanation": record.explanation,
            }
            for record in rankings
        ),
    )
    summary = _summary(
        rankings,
        historical,
        proposed,
        top_k=args.top_k,
    )
    if args.relevance_labels is not None:
        labels = load_relevance_labels(args.relevance_labels)
        metrics = compute_relevance_metrics(rankings, labels, top_k=args.top_k)
        summary["evaluation"] = metrics
        write_json(output_dir / "prevention_metrics.json", metrics)
    write_json(output_dir / "prevention_summary.json", summary)
    write_json(
        output_dir / "prevention_run_config.json",
        {
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "resolved_similarity_config": asdict(config),
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = run(args)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2
    ranking = summary["ranking"]
    print(f"Ranked Indicators: {ranking['ranked_indicator_count']}")
    print(f"Decision counts: {ranking['decision_counts']}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
