from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from policy_filter.config import FilterCliConfig
from policy_filter.csv_stream import (
    ForwardedCsvWriter,
    collect_union_headers,
    iter_csv_records,
    malformed_decision,
    summarize,
)
from policy_filter.matcher import decide
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
    try:
        with ForwardedCsvWriter(
            config.output,
            original_columns,
            delimiter=config.delimiter,
            encoding=config.encoding,
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
                    else:
                        try:
                            preprocessed = preprocess_policy_row(record.row, policy)
                            decision = decide(preprocessed, policy)
                        except Exception:
                            LOGGER.exception("Unexpected row preprocessing failure")
                            decision = malformed_decision("row preprocessing failed")
                    if decision.action == "suppress":
                        suppressed += 1
                        continue
                    forwarded += 1
                    reasons_by_code[decision.reason_code] += 1
                    writer.write_forwarded(record, decision)
    except OSError as exc:
        LOGGER.error("%s", exc)
        return 2

    print(summarize(total, suppressed, forwarded, reasons_by_code))
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
