from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from behaviour_profiling.models import DetectionResult, SourceRecord

ANOMALY_METADATA_COLUMNS = [
    "source_file",
    "source_row_number",
    "source_line_number",
    "behaviour_action",
    "anomaly_score",
    "profile_scope",
    "anomaly_reasons",
    "anomaly_reason_details",
]


def read_header(path: str | Path, *, encoding: str, delimiter: str) -> list[str]:
    with Path(path).open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return [str(column) for column in next(reader, [])]


def collect_union_headers(
    paths: list[Path], *, encoding: str, delimiter: str
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for column in read_header(path, encoding=encoding, delimiter=delimiter):
            if column not in seen:
                seen.add(column)
                result.append(column)
    return result


def iter_csv_records(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> Iterable[SourceRecord]:
    source = Path(path)
    with source.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        if header is None:
            return
        row_number = 0
        while True:
            starting_line = reader.line_num + 1
            try:
                values = next(reader)
            except StopIteration:
                break
            row_number += 1
            malformed = len(values) != len(header)
            row = {
                str(column): values[index] if index < len(values) else ""
                for index, column in enumerate(header)
            }
            yield SourceRecord(
                source_file=source,
                source_row_number=row_number,
                source_line_number=starting_line,
                row=row,
                malformed=malformed,
                malformed_reason=(
                    f"expected {len(header)} fields but found {len(values)}"
                    if malformed
                    else None
                ),
            )


class AtomicJsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None
        self._temporary_path: str | None = None

    def __enter__(self) -> "AtomicJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent, text=True
        )
        self._temporary_path = temporary_path
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8")
        return self

    def write(self, payload: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("writer is not open")
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self._handle is not None and self._temporary_path is not None
        self._handle.close()
        if exc_type is None:
            os.replace(self._temporary_path, self.path)
        else:
            os.unlink(self._temporary_path)


class AtomicAnomalyCsvWriter:
    def __init__(
        self,
        path: str | Path,
        original_columns: list[str],
        *,
        encoding: str,
        delimiter: str,
    ) -> None:
        self.path = Path(path)
        self.original_columns = original_columns
        self.encoding = encoding
        self.delimiter = delimiter
        self._handle = None
        self._writer: csv.DictWriter[str] | None = None
        self._temporary_path: str | None = None

    def __enter__(self) -> "AtomicAnomalyCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent, text=True
        )
        self._temporary_path = temporary_path
        self._handle = os.fdopen(descriptor, "w", encoding=self.encoding, newline="")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=ANOMALY_METADATA_COLUMNS + self.original_columns,
            delimiter=self.delimiter,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        return self

    def write(self, record: SourceRecord, result: DetectionResult) -> None:
        if self._writer is None:
            raise RuntimeError("writer is not open")
        evidence = [item.to_dict() for item in result.evidence]
        output: dict[str, Any] = {
            "source_file": str(record.source_file),
            "source_row_number": record.source_row_number,
            "source_line_number": record.source_line_number,
            "behaviour_action": "anomaly",
            "anomaly_score": result.anomaly_score,
            "profile_scope": result.profile_scope,
            "anomaly_reasons": ";".join(item.reason_code for item in result.evidence),
            "anomaly_reason_details": json.dumps(evidence, sort_keys=True),
        }
        output.update({column: record.row.get(column, "") for column in self.original_columns})
        self._writer.writerow(output)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self._handle is not None and self._temporary_path is not None
        self._handle.close()
        if exc_type is None:
            os.replace(self._temporary_path, self.path)
        else:
            os.unlink(self._temporary_path)

