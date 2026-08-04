from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from policy_filter import reasons
from policy_filter.models import (
    AggregateOutputRecord,
    CsvLogRecord,
    FilterDecision,
    TimestampParseResult,
)
from policy_filter.preprocessing_adapter import normalized_event_fields

FINGERPRINT_METADATA_EXCLUSIONS = {
    "source_file",
    "source_row_number",
    "source_line_number",
    "source_row_number_first",
    "source_row_number_last",
    "source_line_number_first",
    "source_line_number_last",
    "source_row_numbers",
    "source_line_numbers",
    "filter_action",
    "filter_reason",
    "filter_reason_details",
    "evaluated_category",
    "matched_policy_id",
    "occurrence_count",
    "first_seen",
    "last_seen",
    "aggregation_window_minutes",
    "aggregation_status",
}


def compact_ranges(values: Iterable[int]) -> str:
    ordered = sorted(values)
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous:
            continue
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = value
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ";".join(ranges)


def format_utc_timestamp(timestamp: datetime | None) -> str:
    if timestamp is None:
        return ""
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _parse_epoch(value: Any, field_name: str) -> datetime:
    if _missing(value):
        raise ValueError(f"{field_name!r} is missing")
    try:
        seconds = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field_name!r} is not numeric") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{field_name!r} is not a valid nonnegative epoch second value")
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _parse_iso(value: Any, field_name: str, default_timezone: str) -> datetime:
    if _missing(value):
        raise ValueError(f"{field_name!r} is missing")
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("z", "+00:00").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name!r} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(default_timezone))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"default timezone {default_timezone!r} is invalid") from exc
    return parsed.astimezone(timezone.utc)


def parse_aggregation_timestamp(
    row: dict[str, str],
    *,
    epoch_field: str | None,
    iso_field: str | None,
    default_timezone: str,
    tolerance_seconds: float = 1.0,
) -> TimestampParseResult:
    try:
        epoch_timestamp = None
        iso_timestamp = None
        if epoch_field is not None:
            if epoch_field not in row:
                raise ValueError(f"{epoch_field!r} is absent")
            epoch_timestamp = _parse_epoch(row.get(epoch_field), epoch_field)
        if iso_field is not None:
            if iso_field not in row:
                raise ValueError(f"{iso_field!r} is absent")
            iso_timestamp = _parse_iso(row.get(iso_field), iso_field, default_timezone)
        if epoch_timestamp is not None and iso_timestamp is not None:
            delta = abs((epoch_timestamp - iso_timestamp).total_seconds())
            if delta > tolerance_seconds:
                return TimestampParseResult(
                    None,
                    reasons.AGGREGATION_TIMESTAMP_CONFLICT,
                    f"epoch and ISO timestamps differ by {delta:.3f} seconds",
                )
            return TimestampParseResult(epoch_timestamp)
        if epoch_timestamp is not None:
            return TimestampParseResult(epoch_timestamp)
        if iso_timestamp is not None:
            return TimestampParseResult(iso_timestamp)
    except ValueError as exc:
        return TimestampParseResult(
            None,
            reasons.AGGREGATION_TIMESTAMP_INVALID,
            str(exc),
        )
    return TimestampParseResult(
        None,
        reasons.AGGREGATION_TIMESTAMP_INVALID,
        "no aggregation timestamp field configured",
    )


def append_detail(decision: FilterDecision, reason_code: str, reason_detail: str) -> FilterDecision:
    detail = f"{decision.reason_details}; {reason_code}: {reason_detail}"
    return FilterDecision(
        decision.action,
        decision.category,
        decision.reason_code,
        detail,
        decision.matched_policy_id,
    )


@dataclass
class AggregationStats:
    output_rows: int = 0
    aggregate_groups: int = 0
    max_occurrence_count: int = 0
    timestamp_parse_failures: int = 0
    timestamp_conflicts: int = 0
    out_of_order_timestamps: int = 0


@dataclass
class _AggregateGroup:
    key: tuple[Any, ...]
    first_record: CsvLogRecord
    decision: FilterDecision
    first_seen: datetime
    last_seen: datetime
    source_row_numbers: list[int]
    source_line_numbers: list[int]
    occurrence_count: int = 1

    def add(self, record: CsvLogRecord, timestamp: datetime) -> None:
        self.last_seen = max(self.last_seen, timestamp)
        self.source_row_numbers.append(record.source_row_number)
        self.source_line_numbers.append(record.source_line_number)
        self.occurrence_count += 1

    def to_output(self, window_minutes: int) -> AggregateOutputRecord:
        return AggregateOutputRecord(
            source_file=self.first_record.source_file,
            source_row_number_first=min(self.source_row_numbers),
            source_row_number_last=max(self.source_row_numbers),
            source_line_number_first=min(self.source_line_numbers),
            source_line_number_last=max(self.source_line_numbers),
            source_row_numbers=compact_ranges(self.source_row_numbers),
            source_line_numbers=compact_ranges(self.source_line_numbers),
            decision=self.decision,
            occurrence_count=self.occurrence_count,
            first_seen=format_utc_timestamp(self.first_seen),
            last_seen=format_utc_timestamp(self.last_seen),
            aggregation_window_minutes=window_minutes,
            aggregation_status="aggregated" if self.occurrence_count > 1 else "individual",
            row=self.first_record.row,
        )


class DuplicateAggregator:
    def __init__(
        self,
        *,
        window_minutes: int,
        timestamp_epoch_field: str | None,
        timestamp_iso_field: str | None,
        default_timezone: str,
        event_id_enabled: bool,
        severity_enabled: bool,
    ) -> None:
        if window_minutes <= 0:
            raise ValueError("aggregation window must be positive")
        self.window_minutes = window_minutes
        self.window = timedelta(minutes=window_minutes)
        self.timestamp_epoch_field = timestamp_epoch_field
        self.timestamp_iso_field = timestamp_iso_field
        self.default_timezone = default_timezone
        self.event_id_enabled = event_id_enabled
        self.severity_enabled = severity_enabled
        self.stats = AggregationStats()
        self._groups: dict[tuple[Any, ...], _AggregateGroup] = {}
        self._last_seen_by_key: dict[tuple[Any, ...], datetime] = {}
        self._max_seen: datetime | None = None

    def _fingerprint(
        self,
        record: CsvLogRecord,
        decision: FilterDecision,
        *,
        normalized_event_id: str | None,
        normalized_severity: str | None,
    ) -> tuple[Any, ...]:
        excluded = {
            field
            for field in (self.timestamp_epoch_field, self.timestamp_iso_field)
            if field is not None
        }
        excluded.update(FINGERPRINT_METADATA_EXCLUSIONS)
        parts: list[Any] = [
            str(record.source_file),
            decision.category,
            decision.reason_code,
            decision.matched_policy_id or "",
        ]
        if self.event_id_enabled:
            parts.append(normalized_event_id or "")
        if self.severity_enabled:
            parts.append(normalized_severity or "")
        parts.extend(normalized_event_fields(record.row, excluded_fields=excluded))
        return tuple(parts)

    def _output_individual(
        self,
        record: CsvLogRecord,
        decision: FilterDecision,
        timestamp: datetime | None,
    ) -> AggregateOutputRecord:
        timestamp_text = format_utc_timestamp(timestamp)
        return AggregateOutputRecord(
            source_file=record.source_file,
            source_row_number_first=record.source_row_number,
            source_row_number_last=record.source_row_number,
            source_line_number_first=record.source_line_number,
            source_line_number_last=record.source_line_number,
            source_row_numbers=str(record.source_row_number),
            source_line_numbers=str(record.source_line_number),
            decision=decision,
            occurrence_count=1,
            first_seen=timestamp_text,
            last_seen=timestamp_text,
            aggregation_window_minutes=self.window_minutes,
            aggregation_status="individual",
            row=record.row,
        )

    def _flush_expired(self) -> list[AggregateOutputRecord]:
        if self._max_seen is None:
            return []
        expired_keys = [
            key
            for key, group in self._groups.items()
            if group.first_seen + self.window < self._max_seen
        ]
        return [self._flush_key(key) for key in expired_keys]

    def _flush_key(self, key: tuple[Any, ...]) -> AggregateOutputRecord:
        group = self._groups.pop(key)
        output = group.to_output(self.window_minutes)
        self._record_output(output)
        return output

    def _record_output(self, output: AggregateOutputRecord) -> None:
        self.stats.output_rows += 1
        if output.occurrence_count > 1:
            self.stats.aggregate_groups += 1
        self.stats.max_occurrence_count = max(
            self.stats.max_occurrence_count,
            output.occurrence_count,
        )

    def add(
        self,
        record: CsvLogRecord,
        decision: FilterDecision,
        *,
        normalized_event_id: str | None,
        normalized_severity: str | None,
    ) -> list[AggregateOutputRecord]:
        parsed = parse_aggregation_timestamp(
            record.row,
            epoch_field=self.timestamp_epoch_field,
            iso_field=self.timestamp_iso_field,
            default_timezone=self.default_timezone,
        )
        if parsed.timestamp is None:
            if parsed.reason_code == reasons.AGGREGATION_TIMESTAMP_CONFLICT:
                self.stats.timestamp_conflicts += 1
            else:
                self.stats.timestamp_parse_failures += 1
            output = self._output_individual(
                record,
                append_detail(
                    decision,
                    parsed.reason_code or reasons.AGGREGATION_TIMESTAMP_INVALID,
                    parsed.reason_details or "timestamp could not be parsed",
                ),
                None,
            )
            self._record_output(output)
            return [output]

        timestamp = parsed.timestamp
        if self._max_seen is None or timestamp > self._max_seen:
            self._max_seen = timestamp

        key = self._fingerprint(
            record,
            decision,
            normalized_event_id=normalized_event_id,
            normalized_severity=normalized_severity,
        )
        last_seen = self._last_seen_by_key.get(key)
        if last_seen is not None and timestamp < last_seen:
            self.stats.out_of_order_timestamps += 1
            output = self._output_individual(
                record,
                append_detail(
                    decision,
                    reasons.AGGREGATION_OUT_OF_ORDER_TIMESTAMP,
                    "timestamp is earlier than a previously seen duplicate-key timestamp",
                ),
                timestamp,
            )
            self._record_output(output)
            return [output] + self._flush_expired()

        self._last_seen_by_key[key] = timestamp
        outputs: list[AggregateOutputRecord] = []
        group = self._groups.get(key)
        if group is None:
            self._groups[key] = _AggregateGroup(
                key=key,
                first_record=record,
                decision=decision,
                first_seen=timestamp,
                last_seen=timestamp,
                source_row_numbers=[record.source_row_number],
                source_line_numbers=[record.source_line_number],
            )
            return self._flush_expired()

        if timestamp - group.first_seen <= self.window:
            group.add(record, timestamp)
            return self._flush_expired()

        outputs.append(self._flush_key(key))
        self._groups[key] = _AggregateGroup(
            key=key,
            first_record=record,
            decision=decision,
            first_seen=timestamp,
            last_seen=timestamp,
            source_row_numbers=[record.source_row_number],
            source_line_numbers=[record.source_line_number],
        )
        outputs.extend(self._flush_expired())
        return outputs

    def flush_all(self) -> list[AggregateOutputRecord]:
        return [self._flush_key(key) for key in list(self._groups)]
