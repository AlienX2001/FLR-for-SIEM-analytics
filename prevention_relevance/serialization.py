from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from prevention_relevance.graph import TypedGraph


def _json_default(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_text_writer(path: Path):  # noqa: ANN202
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )


def write_json(path: Path, value: object) -> None:
    handle = _atomic_text_writer(path)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                default=_json_default,
            )
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_graph_json(path: Path, graph: TypedGraph) -> None:
    """Serialize a graph without constructing a second graph-sized dictionary."""
    handle = _atomic_text_writer(path)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write("{\n")
            header = {
                "schema_version": 1,
                "graph_id": graph.graph_id,
                "graph_type": graph.graph_type,
                "metadata": graph.metadata,
            }
            for index, (key, value) in enumerate(header.items()):
                separator = "," if index < len(header) - 1 else ","
                handle.write(
                    f"  {json.dumps(key)}: "
                    f"{json.dumps(value, sort_keys=True, default=_json_default)}{separator}\n"
                )
            handle.write('  "nodes": [\n')
            nodes = sorted(graph.nodes.values(), key=lambda item: item.node_id)
            for index, node in enumerate(nodes):
                value = {
                    "id": node.node_id,
                    "type": node.node_type,
                    "value": node.value,
                    "attributes": dict(node.attributes),
                }
                suffix = "," if index < len(nodes) - 1 else ""
                handle.write(
                    "    "
                    + json.dumps(value, sort_keys=True, default=_json_default)
                    + suffix
                    + "\n"
                )
            handle.write("  ],\n")
            handle.write('  "edges": [\n')
            edges = sorted(graph.edges.values(), key=lambda item: item.edge_id)
            for index, edge in enumerate(edges):
                value = {
                    "id": edge.edge_id,
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                    "attributes": dict(edge.attributes),
                }
                suffix = "," if index < len(edges) - 1 else ""
                handle.write(
                    "    "
                    + json.dumps(value, sort_keys=True, default=_json_default)
                    + suffix
                    + "\n"
                )
            handle.write("  ]\n}\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, values: Iterable[object]) -> None:
    handle = _atomic_text_writer(path)
    temporary = Path(handle.name)
    try:
        with handle:
            for value in values:
                handle.write(
                    json.dumps(
                        value,
                        sort_keys=True,
                        ensure_ascii=True,
                        default=_json_default,
                    )
                )
                handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    handle = _atomic_text_writer(path)
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def ranking_to_dict(record: object) -> dict[str, Any]:
    if not is_dataclass(record):
        raise TypeError("Ranking record must be a dataclass")
    return asdict(record)


def ranking_to_csv_row(record: object) -> dict[str, Any]:
    row = ranking_to_dict(record)
    row["explanation"] = json.dumps(
        row["explanation"], sort_keys=True, ensure_ascii=True
    )
    return row
