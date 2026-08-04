from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from policy_filter.aggregation import DuplicateAggregator
from policy_filter.config import FilterCliConfig
from policy_filter.csv_stream import (
    ForwardedCsvWriter,
    collect_union_headers,
    iter_csv_records,
    malformed_decision,
    summarize,
)
from policy_filter.matcher import decide
from policy_filter.prefilters import (
    combine_prefilters_with_policy_decision,
    evaluate_event_id_prefilter,
    evaluate_severity_prefilter,
)
from policy_filter.preprocessing_adapter import preprocess_policy_row
from policy_filter.schema import PolicyValidationError, load_policy

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> FilterCliConfig:
    parser = argparse.ArgumentParser(description="Policy-based pre-SIEM log filter.")
    parser.add_argument("--logs", nargs="+", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--default-timezone", default=None)
    parser.add_argument("--strict-policy-validation", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--event-id-field", default=None)
    parser.add_argument("--severity-field", default=None)
    parser.add_argument("--timestamp-epoch-field", default=None)
    parser.add_argument("--timestamp-iso-field", default=None)
    parser.add_argument("--aggregation-window-minutes", type=int, default=0)
    args = parser.parse_args(argv)
    if len(args.delimiter) != 1:
        parser.error("--delimiter must be a single character")
    return FilterCliConfig(
        logs=args.logs,
        policy=args.policy,
        output=args.output,
        encoding=args.encoding,
        delimiter=args.delimiter,
        default_timezone=args.default_timezone,
        strict_policy_validation=args.strict_policy_validation,
        log_level=args.log_level,
        event_id_field=args.event_id_field,
        severity_field=args.severity_field,
        timestamp_epoch_field=args.timestamp_epoch_field,
        timestamp_iso_field=args.timestamp_iso_field,
        aggregation_window_minutes=args.aggregation_window_minutes,
    )


def run(config: FilterCliConfig) -> int:
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        policy = load_policy(config.policy, strict=config.strict_policy_validation)
        if config.default_timezone is not None:
            try:
                ZoneInfo(config.default_timezone)
            except ZoneInfoNotFoundError as exc:
                raise PolicyValidationError(
                    f"--default-timezone: invalid timezone {config.default_timezone!r}"
                ) from exc
            policy = replace(policy, default_timezone=config.default_timezone)
        if config.event_id_field is not None and policy.prefilters.event_id is None:
            raise PolicyValidationError(
                "--event-id-field requires policy prefilters.event_id configuration"
            )
        if config.severity_field is not None and policy.prefilters.severity is None:
            raise PolicyValidationError(
                "--severity-field requires policy prefilters.severity configuration"
            )
        aggregation_enabled = config.aggregation_window_minutes > 0
        if aggregation_enabled and not (
            config.timestamp_epoch_field or config.timestamp_iso_field
        ):
            raise PolicyValidationError(
                "--aggregation-window-minutes requires --timestamp-epoch-field or --timestamp-iso-field"
            )
        original_columns = collect_union_headers(
            config.logs,
            encoding=config.encoding,
            delimiter=config.delimiter,
        )
    except (OSError, PolicyValidationError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    total = 0
    suppressed = 0
    forwarded = 0
    reasons_by_code: Counter[str] = Counter()
    event_id_counts: Counter[str] | None = Counter() if config.event_id_field is not None else None
    severity_counts: Counter[str] | None = Counter() if config.severity_field is not None else None
    aggregator = (
        DuplicateAggregator(
            window_minutes=config.aggregation_window_minutes,
            timestamp_epoch_field=config.timestamp_epoch_field,
            timestamp_iso_field=config.timestamp_iso_field,
            default_timezone=policy.default_timezone,
            event_id_enabled=config.event_id_field is not None,
            severity_enabled=config.severity_field is not None,
        )
        if aggregation_enabled
        else None
    )
    try:
        with ForwardedCsvWriter(
            config.output,
            original_columns,
            delimiter=config.delimiter,
            encoding=config.encoding,
            aggregation_enabled=aggregation_enabled,
        ) as writer:
            for path in config.logs:
                LOGGER.info("Filtering CSV file %s", path)
                for record in iter_csv_records(
                    path,
                    encoding=config.encoding,
                    delimiter=config.delimiter,
                ):
                    total += 1
                    if record.malformed:
                        decision = malformed_decision(record.malformed_reason)
                        event_decision = None
                        severity_decision = None
                    else:
                        try:
                            preprocessed = preprocess_policy_row(record.row, policy)
                            base_decision = decide(preprocessed, policy)
                            event_decision = (
                                evaluate_event_id_prefilter(
                                    record.row,
                                    field_name=config.event_id_field,
                                    policy=policy.prefilters.event_id,
                                )
                                if config.event_id_field is not None
                                else None
                            )
                            severity_decision = (
                                evaluate_severity_prefilter(
                                    record.row,
                                    field_name=config.severity_field,
                                    policy=policy.prefilters.severity,
                                )
                                if config.severity_field is not None
                                else None
                            )
                            if event_decision is not None and event_id_counts is not None:
                                event_id_counts.update(event_decision.reason_codes)
                            if severity_decision is not None and severity_counts is not None:
                                severity_counts.update(severity_decision.reason_codes)
                            decision = combine_prefilters_with_policy_decision(
                                base_decision,
                                tuple(
                                    item
                                    for item in (event_decision, severity_decision)
                                    if item is not None
                                ),
                            )
                        except Exception:
                            LOGGER.exception("Unexpected row preprocessing failure")
                            decision = malformed_decision("row preprocessing failed")
                            event_decision = None
                            severity_decision = None
                    if decision.action == "suppress":
                        suppressed += 1
                        continue
                    forwarded += 1
                    reasons_by_code[decision.reason_code] += 1
                    if aggregator is None:
                        writer.write_forwarded(record, decision)
                    else:
                        for output in aggregator.add(
                            record,
                            decision,
                            normalized_event_id=(
                                event_decision.normalized_value
                                if event_decision is not None
                                else None
                            ),
                            normalized_severity=(
                                severity_decision.normalized_value
                                if severity_decision is not None
                                else None
                            ),
                        ):
                            writer.write_aggregate(output)
            if aggregator is not None:
                for output in aggregator.flush_all():
                    writer.write_aggregate(output)
    except OSError as exc:
        LOGGER.error("%s", exc)
        return 2

    print(
        summarize(
            total,
            suppressed,
            forwarded,
            reasons_by_code,
            output_rows_after_aggregation=(
                aggregator.stats.output_rows if aggregator is not None else forwarded
            ),
            aggregate_groups=aggregator.stats.aggregate_groups if aggregator is not None else 0,
            max_occurrence_count=(
                aggregator.stats.max_occurrence_count if aggregator is not None else 0
            ),
            event_id_counts=event_id_counts,
            severity_counts=severity_counts,
            timestamp_parse_failures=(
                aggregator.stats.timestamp_parse_failures if aggregator is not None else 0
            ),
            timestamp_conflicts=aggregator.stats.timestamp_conflicts if aggregator is not None else 0,
            out_of_order_timestamps=(
                aggregator.stats.out_of_order_timestamps if aggregator is not None else 0
            ),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
