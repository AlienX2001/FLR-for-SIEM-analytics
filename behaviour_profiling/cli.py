from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from behaviour_profiling.config import BehaviourConfigError, load_config
from behaviour_profiling.detector import detect_anomalies
from behaviour_profiling.trainer import train_profile

LOGGER = logging.getLogger(__name__)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--logs", nargs="+", required=True, type=Path)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--log-level", default="INFO")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train benign behavior profiles and detect anomalous log rows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Build a profile from benign logs")
    _common_arguments(train_parser)
    train_parser.add_argument("--output", required=True, type=Path)
    train_parser.add_argument("--config", type=Path, default=None)
    train_parser.add_argument("--groundtruth", nargs="+", type=Path, default=None)
    train_parser.add_argument("--label-column", default=None)
    train_parser.add_argument("--benign-label", default="Benign")

    detect_parser = subparsers.add_parser("detect", help="Detect rows outside a profile")
    _common_arguments(detect_parser)
    detect_parser.add_argument("--profile", required=True, type=Path)
    detect_parser.add_argument("--output", required=True, type=Path)
    detect_parser.add_argument("--results-jsonl", type=Path, default=None)
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.delimiter) != 1:
        parser.error("--delimiter must be a single character")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "train":
            profile = train_profile(
                log_paths=args.logs,
                output_path=args.output,
                config=load_config(args.config),
                groundtruth_paths=args.groundtruth,
                label_column=args.label_column,
                benign_label=args.benign_label,
                encoding=args.encoding,
                delimiter=args.delimiter,
            )
            summary = profile["training_summary"]
            print(f"behavior profile: {args.output}")
            print(f"total rows read: {summary['total_rows_read']}")
            print(f"benign rows used: {summary['benign_rows_used']}")
            print(f"profiled fields: {len(summary['profiled_fields'])}")
            print(f"entity profiles: {summary['entity_profiles_created']}")
            return 0

        summary = detect_anomalies(
            log_paths=args.logs,
            profile_path=args.profile,
            output_path=args.output,
            results_jsonl_path=args.results_jsonl,
            encoding=args.encoding,
            delimiter=args.delimiter,
        )
        percentage = (
            100.0 * summary.anomaly_rows / summary.total_rows
            if summary.total_rows
            else 0.0
        )
        print(f"total rows read: {summary.total_rows}")
        print(f"normal rows: {summary.normal_rows}")
        print(f"anomaly rows: {summary.anomaly_rows}")
        print(f"anomaly percentage: {percentage:.2f}%")
        print(f"malformed rows: {summary.malformed_rows}")
        print("reason counts:")
        for reason, count in summary.reasons.items():
            print(f"  {reason}: {count}")
        return 0
    except (OSError, ValueError, BehaviourConfigError) as exc:
        LOGGER.error("%s", exc)
        return 2


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main(sys.argv[1:])

