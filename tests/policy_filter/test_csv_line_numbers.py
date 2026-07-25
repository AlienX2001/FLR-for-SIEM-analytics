from __future__ import annotations

import csv
from pathlib import Path

from policy_filter.cli import run
from policy_filter.config import FilterCliConfig
from policy_filter.csv_stream import collect_union_headers, iter_csv_records
from policy_filter_helpers import base_policy, write_policy


def test_physical_line_numbers_with_embedded_newline(tmp_path: Path) -> None:
    path = tmp_path / "logs.csv"
    path.write_text(
        'timestamp,host,user,process,notes\n'
        '2026-07-20T09:00:00Z,HOST-01,alice,browser.exe,"first line\nsecond line"\n'
        '2026-07-20T09:05:00Z,HOST-02,bob,browser.exe,normal\n',
        encoding="utf-8",
    )

    records = list(iter_csv_records(path))

    assert records[0].source_row_number == 1
    assert records[0].source_line_number == 2
    assert records[1].source_row_number == 2
    assert records[1].source_line_number == 4
    assert records[0].row["notes"] == "first line\nsecond line"


def test_multiple_files_union_headers_and_forwarded_output_preserves_values(tmp_path: Path) -> None:
    policy_path = write_policy(tmp_path, base_policy())
    logs_a = tmp_path / "a.csv"
    logs_b = tmp_path / "b.csv"
    logs_a.write_text(
        "timestamp,host,user,process,process_path,parent\n"
        "2026-07-20T22:00:00Z,HOST-01,alice,browser.exe,C:\\Temp\\browser.exe,explorer.exe\n",
        encoding="utf-8",
    )
    logs_b.write_text(
        "timestamp,host,user,process,extra\n"
        "2026-07-20T09:00:00Z,HOST-99,alice,browser.exe,=formula\n",
        encoding="utf-8",
    )
    output = tmp_path / "forwarded.csv"

    assert collect_union_headers([logs_a, logs_b], encoding="utf-8", delimiter=",") == [
        "timestamp",
        "host",
        "user",
        "process",
        "process_path",
        "parent",
        "extra",
    ]
    exit_code = run(
        FilterCliConfig(
            logs=[logs_a, logs_b],
            policy=policy_path,
            output=output,
        )
    )

    assert exit_code == 0
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["source_row_number"] == "1"
    assert rows[0]["source_line_number"] == "2"
    assert rows[0]["process_path"] == "C:\\Temp\\browser.exe"
    assert rows[1]["extra"] == "=formula"
