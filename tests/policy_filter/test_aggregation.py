from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from policy_filter.aggregation import DuplicateAggregator, compact_ranges, parse_aggregation_timestamp
from policy_filter.cli import run
from policy_filter.config import FilterCliConfig
from policy_filter.csv_stream import iter_csv_records
from policy_filter.models import CsvLogRecord, FilterDecision
from policy_filter_helpers import base_policy, write_policy


def _record(
    tmp_path: Path,
    *,
    row_number: int,
    line_number: int,
    source_name: str = "logs.csv",
    **fields: str,
) -> CsvLogRecord:
    return CsvLogRecord(
        source_file=tmp_path / source_name,
        source_row_number=row_number,
        source_line_number=line_number,
        row={
            "event_time_iso": "2026-07-20T10:00:00Z",
            "host": "HOST-99",
            "src_ip": "10.1.1.20",
            "dst_ip": "10.1.1.5",
            "dst_port": "443",
            "firewall_action": "allow",
            **fields,
        },
    )


def _decision(reason: str = "NO_APPLICABLE_POLICY") -> FilterDecision:
    return FilterDecision(
        "forward",
        "network",
        reason,
        "forwarded for test",
        None,
    )


def _aggregator(tmp_path: Path, *, window: int = 15, epoch: bool = False) -> DuplicateAggregator:
    return DuplicateAggregator(
        window_minutes=window,
        timestamp_epoch_field="event_time_epoch" if epoch else None,
        timestamp_iso_field=None if epoch else "event_time_iso",
        default_timezone="UTC",
        event_id_enabled=False,
        severity_enabled=False,
    )


def _outputs(aggregator: DuplicateAggregator, records: list[CsvLogRecord]):
    written = []
    for record in records:
        written.extend(
            aggregator.add(
                record,
                _decision(),
                normalized_event_id=None,
                normalized_severity=None,
            )
        )
    written.extend(aggregator.flush_all())
    return written


def test_aggregation_disabled_preserves_original_output_columns(tmp_path: Path) -> None:
    policy_path = write_policy(tmp_path, base_policy())
    logs = tmp_path / "logs.csv"
    logs.write_text(
        "timestamp,host,user,process\n"
        "2026-07-20T09:00:00Z,HOST-99,alice,browser.exe\n",
        encoding="utf-8",
    )
    output = tmp_path / "forwarded.csv"

    assert run(FilterCliConfig(logs=[logs], policy=policy_path, output=output)) == 0

    with output.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert "source_row_number" in header
    assert "occurrence_count" not in header


def test_aggregation_requires_timestamp_field(tmp_path: Path) -> None:
    policy_path = write_policy(tmp_path, base_policy())
    logs = tmp_path / "logs.csv"
    logs.write_text("timestamp,host\n2026-07-20T09:00:00Z,HOST-99\n", encoding="utf-8")

    exit_code = run(
        FilterCliConfig(
            logs=[logs],
            policy=policy_path,
            output=tmp_path / "out.csv",
            aggregation_window_minutes=15,
        )
    )

    assert exit_code == 2


def test_duplicates_aggregate_with_inclusive_boundary_and_beyond_starts_new_group(tmp_path: Path) -> None:
    aggregator = _aggregator(tmp_path)
    records = [
        _record(tmp_path, row_number=1, line_number=2, event_time_iso="2026-07-20T10:00:00Z"),
        _record(tmp_path, row_number=2, line_number=3, event_time_iso="2026-07-20T10:15:00Z"),
        _record(tmp_path, row_number=3, line_number=4, event_time_iso="2026-07-20T10:15:01Z"),
    ]

    outputs = _outputs(aggregator, records)

    assert [output.occurrence_count for output in outputs] == [2, 1]
    assert outputs[0].first_seen == "2026-07-20T10:00:00Z"
    assert outputs[0].last_seen == "2026-07-20T10:15:00Z"


def test_non_duplicates_and_different_source_files_do_not_merge(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, row_number=1, line_number=2, dst_port="443"),
        _record(tmp_path, row_number=2, line_number=3, dst_port="8443"),
        _record(tmp_path, row_number=1, line_number=2, source_name="other.csv", dst_port="443"),
    ]

    outputs = _outputs(_aggregator(tmp_path), records)

    assert [output.occurrence_count for output in outputs] == [1, 1, 1]


def test_interleaved_duplicate_streams_aggregate_independently(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, row_number=1, line_number=2, dst_port="443", event_time_iso="2026-07-20T10:00:00Z"),
        _record(tmp_path, row_number=2, line_number=3, dst_port="53", event_time_iso="2026-07-20T10:00:01Z"),
        _record(tmp_path, row_number=3, line_number=4, dst_port="443", event_time_iso="2026-07-20T10:01:00Z"),
        _record(tmp_path, row_number=4, line_number=5, dst_port="53", event_time_iso="2026-07-20T10:01:01Z"),
    ]

    outputs = sorted(_outputs(_aggregator(tmp_path), records), key=lambda output: output.row["dst_port"])

    assert [output.occurrence_count for output in outputs] == [2, 2]


def test_epoch_iso_offset_and_conflict_parsing() -> None:
    parsed_epoch = parse_aggregation_timestamp(
            {"event_time_epoch": "1784541600"},
        epoch_field="event_time_epoch",
        iso_field=None,
        default_timezone="UTC",
    )
    parsed_iso = parse_aggregation_timestamp(
        {"event_time_iso": "2026-07-20T03:00:00-07:00"},
        epoch_field=None,
        iso_field="event_time_iso",
        default_timezone="UTC",
    )
    parsed_both = parse_aggregation_timestamp(
        {
                "event_time_epoch": "1784541600",
            "event_time_iso": "2026-07-20T10:00:00Z",
        },
        epoch_field="event_time_epoch",
        iso_field="event_time_iso",
        default_timezone="UTC",
    )
    conflict = parse_aggregation_timestamp(
        {
                "event_time_epoch": "1784541600",
            "event_time_iso": "2026-07-20T10:00:05Z",
        },
        epoch_field="event_time_epoch",
        iso_field="event_time_iso",
        default_timezone="UTC",
    )

    assert parsed_epoch.timestamp == datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)
    assert parsed_iso.timestamp == datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)
    assert parsed_both.timestamp == datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)
    assert conflict.reason_code == "AGGREGATION_TIMESTAMP_CONFLICT"


def test_malformed_and_out_of_order_timestamps_are_written_individually(tmp_path: Path) -> None:
    aggregator = _aggregator(tmp_path)
    records = [
        _record(tmp_path, row_number=1, line_number=2, event_time_iso="2026-07-20T10:10:00Z"),
        _record(tmp_path, row_number=2, line_number=3, event_time_iso="not-a-date"),
        _record(tmp_path, row_number=3, line_number=4, event_time_iso="2026-07-20T10:05:00Z"),
    ]

    outputs = _outputs(aggregator, records)

    assert [output.occurrence_count for output in outputs] == [1, 1, 1]
    assert aggregator.stats.timestamp_parse_failures == 1
    assert aggregator.stats.out_of_order_timestamps == 1
    assert "AGGREGATION_TIMESTAMP_INVALID" in outputs[0].decision.reason_details
    assert "AGGREGATION_OUT_OF_ORDER_TIMESTAMP" in outputs[1].decision.reason_details


def test_source_locations_are_compacted_and_embedded_newline_start_lines_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "logs.csv"
    path.write_text(
        'event_time_iso,host,message\n'
        '2026-07-20T10:00:00Z,HOST-99,"first line\nsecond line"\n'
        '2026-07-20T10:01:00Z,HOST-99,"first line\nsecond line"\n',
        encoding="utf-8",
    )
    records = list(iter_csv_records(path))
    outputs = _outputs(_aggregator(tmp_path), records)

    assert [record.source_line_number for record in records] == [2, 4]
    assert outputs[0].occurrence_count == 2
    assert outputs[0].source_line_numbers == "2;4"
    assert compact_ranges([4, 5, 8, 10, 11, 12]) == "4-5;8;10-12"


def test_only_forwarded_rows_are_aggregated_by_cli(tmp_path: Path) -> None:
    policy_path = write_policy(tmp_path, base_policy())
    logs = tmp_path / "logs.csv"
    logs.write_text(
        "event_time_iso,timestamp,host,user,process,process_path,parent\n"
        "2026-07-20T10:00:00Z,2026-07-20T09:00:00Z,HOST-01,alice,browser.exe,C:\\Program Files\\Browser\\browser.exe,explorer.exe\n"
        "2026-07-20T10:01:00Z,2026-07-20T22:00:00Z,HOST-01,alice,browser.exe,C:\\Program Files\\Browser\\browser.exe,explorer.exe\n"
        "2026-07-20T10:02:00Z,2026-07-20T22:00:00Z,HOST-01,alice,browser.exe,C:\\Program Files\\Browser\\browser.exe,explorer.exe\n",
        encoding="utf-8",
    )
    output = tmp_path / "forwarded.csv"

    assert run(
        FilterCliConfig(
            logs=[logs],
            policy=policy_path,
            output=output,
            timestamp_iso_field="event_time_iso",
            aggregation_window_minutes=15,
        )
    ) == 0

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["occurrence_count"] == "2"
    assert rows[0]["source_row_numbers"] == "2-3"


def test_thousand_duplicate_firewall_events_collapse_to_one_row(tmp_path: Path) -> None:
    policy_path = write_policy(tmp_path, base_policy())
    logs = tmp_path / "firewall.csv"
    lines = ["event_time_iso,host,firewall_action,src_ip,dst_ip,dst_port"]
    base = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc).timestamp()
    for index in range(1000):
        seconds = int(round(index * 900 / 999))
        stamp = datetime.fromtimestamp(base + seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        lines.append(f"{stamp},HOST-99,allow,10.1.1.20,10.1.1.5,443")
    logs.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = tmp_path / "forwarded.csv"

    assert run(
        FilterCliConfig(
            logs=[logs],
            policy=policy_path,
            output=output,
            timestamp_iso_field="event_time_iso",
            aggregation_window_minutes=15,
        )
    ) == 0

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["occurrence_count"] == "1000"
    assert rows[0]["aggregation_status"] == "aggregated"
    assert rows[0]["firewall_action"] == "allow"
