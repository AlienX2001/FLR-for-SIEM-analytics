from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from policy_filter import reasons
from policy_filter.models import CsvLogRecord, FilterDecision

METADATA_COLUMNS = [
    "source_file",
    "source_row_number",
    "source_line_number",
    "filter_action",
    "filter_reason",
    "filter_reason_details",
    "evaluated_category",
    "matched_policy_id",
]


def _metadata_safe(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def read_header(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> list[str]:
    with Path(path).open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return []
    return [str(column) for column in header]


def collect_union_headers(
    paths: list[Path],
    *,
    encoding: str,
    delimiter: str,
) -> list[str]:
    union: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for column in read_header(path, encoding=encoding, delimiter=delimiter):
            if column not in seen:
                seen.add(column)
                union.append(column)
    return union


def iter_csv_records(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> Iterable[CsvLogRecord]:
    source = Path(path)
    with source.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return
        source_row_number = 0
        while True:
            start_line = reader.line_num + 1
            try:
                row_values = next(reader)
            except StopIteration:
                break
            source_row_number += 1
            malformed = len(row_values) != len(header)
            malformed_reason = None
            if malformed:
                malformed_reason = (
                    f"expected {len(header)} fields but found {len(row_values)}"
                )
            row: dict[str, str] = {}
            for index, column in enumerate(header):
                row[column] = row_values[index] if index < len(row_values) else ""
            yield CsvLogRecord(
                source_file=source,
                source_row_number=source_row_number,
                source_line_number=start_line,
                row=row,
                malformed=malformed,
                malformed_reason=malformed_reason,
            )


class ForwardedCsvWriter:
    def __init__(
        self,
        output_path: str | Path,
        original_columns: list[str],
        *,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        self.output_path = Path(output_path)
        self.original_columns = original_columns
        self.delimiter = delimiter
        self.encoding = encoding
        self._temp_name: str | None = None
        self._handle = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> "ForwardedCsvWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.",
            suffix=".tmp",
            dir=str(self.output_path.parent),
            text=True,
        )
        self._temp_name = temp_name
        self._handle = os.fdopen(fd, "w", encoding=self.encoding, newline="")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=METADATA_COLUMNS + self.original_columns,
            delimiter=self.delimiter,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        return self

    def write_forwarded(self, record: CsvLogRecord, decision: FilterDecision) -> None:
        if self._writer is None:
            raise RuntimeError("writer is not open")
        row = {
            "source_file": _metadata_safe(str(record.source_file)),
            "source_row_number": _metadata_safe(record.source_row_number),
            "source_line_number": _metadata_safe(record.source_line_number),
            "filter_action": "forward",
            "filter_reason": _metadata_safe(decision.reason_code),
            "filter_reason_details": _metadata_safe(decision.reason_details),
            "evaluated_category": _metadata_safe(decision.category),
            "matched_policy_id": _metadata_safe(decision.matched_policy_id or ""),
        }
        for column in self.original_columns:
            row[column] = record.row.get(column, "")
        self._writer.writerow(row)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self._handle is not None
        assert self._temp_name is not None
        self._handle.close()
        if exc_type is None:
            os.replace(self._temp_name, self.output_path)
        else:
            try:
                os.unlink(self._temp_name)
            except OSError:
                pass


def malformed_decision(details: str | None) -> FilterDecision:
    return FilterDecision(
        action="forward",
        category="malformed",
        reason_code=reasons.MALFORMED_ROW,
        reason_details=details or "malformed CSV row",
    )


def summarize(total: int, suppressed: int, forwarded: int, reasons_by_code: Counter[str]) -> str:
    percentage = (100.0 * forwarded / total) if total else 0.0
    lines = [
        f"total rows read: {total}",
        f"total rows suppressed: {suppressed}",
        f"total rows forwarded: {forwarded}",
        f"forwarding percentage: {percentage:.2f}%",
        "filter_reason counts:",
    ]
    for reason_code, count in sorted(reasons_by_code.items()):
        lines.append(f"  {reason_code}: {count}")
    return "\n".join(lines)
