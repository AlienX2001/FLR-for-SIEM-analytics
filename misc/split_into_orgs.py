#!/usr/bin/env python3
"""Split aligned raw-log and groundtruth CSV files into organization-specific sets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ID_FIELDS = ["normalized_id", "entity_id", "source_entity_id", "target_entity_id"]
MISSING_VALUES = {"", "nan", "none", "null", "<na>"}


def clean_id(value: Any) -> str:
    """Normalize missing values for ID lookup without changing stored row data."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_VALUES:
        return ""
    return text


def load_data(raw_path: Path, groundtruth_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw log CSV not found: {raw_path}")
    if not groundtruth_path.exists():
        raise FileNotFoundError(f"Groundtruth CSV not found: {groundtruth_path}")

    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    groundtruth = pd.read_csv(groundtruth_path, dtype=str, keep_default_na=False)
    return raw, groundtruth


def validate_alignment(raw: pd.DataFrame, groundtruth: pd.DataFrame) -> None:
    if len(raw) != len(groundtruth):
        raise ValueError(
            "Raw log and groundtruth row counts differ: "
            f"raw={len(raw)}, groundtruth={len(groundtruth)}"
        )


def org_names(num_orgs: int) -> list[str]:
    if num_orgs <= 0:
        raise ValueError("--num_orgs must be a positive integer")
    return [f"org_{index}" for index in range(1, num_orgs + 1)]


def assign_ids_to_orgs(
    raw: pd.DataFrame, num_orgs: int, seed: int
) -> tuple[dict[str, str], dict[str, int]]:
    """Assign every non-empty normalized_id to exactly one deterministic org."""
    names = org_names(num_orgs)
    if "normalized_id" not in raw.columns:
        return {}, {name: 0 for name in names}

    unique_ids = sorted({clean_id(value) for value in raw["normalized_id"]})
    unique_ids = [value for value in unique_ids if value]

    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    id_to_org: dict[str, str] = {}
    id_counts = {name: 0 for name in names}
    for index, normalized_id in enumerate(unique_ids):
        org = names[index % num_orgs]
        id_to_org[normalized_id] = org
        id_counts[org] += 1

    return id_to_org, id_counts


def get_row_orgs(
    row_values: tuple[Any, ...],
    field_names: list[str],
    id_to_org: dict[str, str],
) -> tuple[set[str], str]:
    """Return all orgs referenced by known IDs in a row, plus its normalized_id."""
    referenced_orgs: set[str] = set()
    normalized_id = ""

    for field_name, value in zip(field_names, row_values):
        candidate_id = clean_id(value)
        if field_name == "normalized_id":
            normalized_id = candidate_id
        if candidate_id and candidate_id in id_to_org:
            referenced_orgs.add(id_to_org[candidate_id])

    return referenced_orgs, normalized_id


def split_rows(
    raw: pd.DataFrame,
    id_to_org: dict[str, str],
    num_orgs: int,
) -> tuple[dict[str, list[int]], list[int], list[int]]:
    """
    Route rows by referenced organizations.

    A row is removed when the IDs visible in normalized_id/entity_id/source/target
    resolve to more than one organization. This drops inter-organization entity
    relationships and communication rows while preserving the matching
    groundtruth row by using the same row index lists for both files.
    """
    names = org_names(num_orgs)
    org_indices = {name: [] for name in names}
    removed_inter_org: list[int] = []
    unassigned: list[int] = []

    present_fields = [field for field in ID_FIELDS if field in raw.columns]
    if present_fields:
        id_frame = raw[present_fields]
        row_iter = id_frame.itertuples(index=False, name=None)
    else:
        row_iter = iter(())

    for row_index, row_values in enumerate(row_iter):
        row_orgs, normalized_id = get_row_orgs(row_values, present_fields, id_to_org)

        if len(row_orgs) > 1:
            removed_inter_org.append(row_index)
        elif len(row_orgs) == 1:
            org_indices[next(iter(row_orgs))].append(row_index)
        elif normalized_id and normalized_id in id_to_org:
            org_indices[id_to_org[normalized_id]].append(row_index)
        else:
            unassigned.append(row_index)

    if not present_fields:
        unassigned.extend(range(len(raw)))

    return org_indices, removed_inter_org, unassigned


def write_outputs(
    raw: pd.DataFrame,
    groundtruth: pd.DataFrame,
    output_dir: Path,
    org_indices: dict[str, list[int]],
    removed_inter_org: list[int],
    unassigned: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for org, indices in org_indices.items():
        org_dir = output_dir / org
        org_dir.mkdir(parents=True, exist_ok=True)
        raw.iloc[indices].to_csv(org_dir / "raw.csv", index=False)
        groundtruth.iloc[indices].to_csv(org_dir / "groundtruth.csv", index=False)

    raw.iloc[removed_inter_org].to_csv(output_dir / "removed_inter_org_raw.csv", index=False)
    groundtruth.iloc[removed_inter_org].to_csv(
        output_dir / "removed_inter_org_groundtruth.csv", index=False
    )
    raw.iloc[unassigned].to_csv(output_dir / "unassigned_raw.csv", index=False)
    groundtruth.iloc[unassigned].to_csv(output_dir / "unassigned_groundtruth.csv", index=False)


def write_summary(
    output_dir: Path,
    total_rows: int,
    org_indices: dict[str, list[int]],
    removed_inter_org: list[int],
    unassigned: list[int],
    id_counts: dict[str, int],
    num_orgs: int,
    seed: int,
) -> dict[str, Any]:
    rows_written = {org: len(indices) for org, indices in org_indices.items()}
    accounted_rows = sum(rows_written.values()) + len(removed_inter_org) + len(unassigned)
    summary: dict[str, Any] = {
        "total_input_rows": total_rows,
        "num_orgs": num_orgs,
        "seed": seed,
        "rows_written_to_each_organization": rows_written,
        "removed_inter_org_rows": len(removed_inter_org),
        "unassigned_rows": len(unassigned),
        "unique_normalized_ids_assigned_to_each_organization": id_counts,
        "sanity_check": {
            "accounted_rows": accounted_rows,
            "matches_total_input_rows": accounted_rows == total_rows,
        },
    }

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split aligned raw-log and groundtruth CSV files into organizations."
    )
    parser.add_argument("--raw", required=True, type=Path, help="Aligned raw log CSV.")
    parser.add_argument("--groundtruth", required=True, type=Path, help="Aligned groundtruth CSV.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--num_orgs", type=int, default=3, help="Number of organizations.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic assignment seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        raw, groundtruth = load_data(args.raw, args.groundtruth)
        validate_alignment(raw, groundtruth)
        id_to_org, id_counts = assign_ids_to_orgs(raw, args.num_orgs, args.seed)
        org_indices, removed_inter_org, unassigned = split_rows(
            raw, id_to_org, args.num_orgs
        )
        write_outputs(
            raw,
            groundtruth,
            args.output_dir,
            org_indices,
            removed_inter_org,
            unassigned,
        )
        summary = write_summary(
            args.output_dir,
            len(raw),
            org_indices,
            removed_inter_org,
            unassigned,
            id_counts,
            args.num_orgs,
            args.seed,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Split summary")
    for org, count in summary["rows_written_to_each_organization"].items():
        print(f"{org}: {count} rows")
    print(f"Removed inter-organization rows: {summary['removed_inter_org_rows']}")
    print(f"Unassigned rows: {summary['unassigned_rows']}")
    print(
        "Sanity check: "
        f"{summary['sanity_check']['accounted_rows']} / {summary['total_input_rows']} rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
