from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_lr_pipeline.data import read_log_csv
from federated_lr_pipeline.local_training import (
    build_feature_matrix_from_counters,
    build_token_counters,
)
from federated_lr_pipeline.prf import derive_prf_key
from federated_lr_pipeline.specialized_models import (
    BENIGN_NOVELTY_BASELINE_FILENAME,
    benign_novelty_baselines_from_json,
    build_cross_texts_for_org,
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a saved run's sparse cross-category TF matrix and export "
            "its nonzero row/feature values."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("outputs/final-bs512"),
        help="Training artifact directory. Default: outputs/final-bs512",
    )
    parser.add_argument(
        "--org-index",
        type=int,
        default=0,
        help="Zero-based organization index to reconstruct. Default: 0",
    )
    parser.add_argument(
        "--label",
        default="benign",
        help=(
            "Label branch used to locate cross artifacts in the manifest. Cross "
            "TF values are shared across label specialists. Default: benign"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --artifact-dir.",
    )
    return parser.parse_args()


def reconstruct_cross_tf(
    *,
    artifact_dir: Path,
    org_index: int,
    label: str,
    output_dir: Path,
) -> tuple[Path, Path, Path, tuple[int, int], int, list[float]]:
    artifact_dir = artifact_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _read_json(artifact_dir / "run_config.json")
    manifest = _read_json(artifact_dir / "hierarchical_model_manifest.json")
    if label not in manifest or "cross" not in manifest[label]:
        raise ValueError(f"Manifest has no cross specialist for label {label!r}")

    org_data = config.get("org_data")
    if not isinstance(org_data, list) or not 0 <= org_index < len(org_data):
        raise ValueError(
            f"Organization index {org_index} is outside run_config.org_data"
        )

    cross_entry = manifest[label]["cross"]
    per_org_artifacts = cross_entry.get("per_org_artifacts", [])
    try:
        org_entry = next(
            entry
            for entry in per_org_artifacts
            if int(entry["org_index"]) == org_index
        )
    except StopIteration as exc:
        raise ValueError(
            f"Manifest has no cross artifacts for organization {org_index}"
        ) from exc

    raw_path = _resolve_project_path(str(org_data[org_index]))
    logs_df = read_log_csv(raw_path)
    dataset = SimpleNamespace(org_index=org_index, logs_df=logs_df)

    baseline_metadata = config.get("benign_novelty_baseline", {})
    baseline_filename = str(
        baseline_metadata.get("filename", BENIGN_NOVELTY_BASELINE_FILENAME)
    )
    baselines = benign_novelty_baselines_from_json(
        _read_json(artifact_dir / baseline_filename)
    )
    if org_index not in baselines:
        raise ValueError(
            f"Benign novelty baseline has no entry for organization {org_index}"
        )

    cross_context = config.get("cross_context", {})
    context_window_minutes = float(
        cross_context.get(
            "window_minutes",
            config.get("context_window_minutes", 15.0),
        )
    )
    epoch_field = cross_context.get(
        "timestamp_epoch_field",
        config.get("context_timestamp_epoch_field", "event_time_epoch"),
    )
    iso_field = cross_context.get(
        "timestamp_iso_field",
        config.get("context_timestamp_iso_field", "event_time_iso"),
    )
    prf_key = derive_prf_key(int(config["seed"]))

    texts, _ = build_cross_texts_for_org(
        dataset,
        context_window_minutes=context_window_minutes,
        context_timestamp_epoch_field=epoch_field,
        context_timestamp_iso_field=iso_field,
        benign_novelty_baseline=baselines[org_index],
        prf_key=prf_key,
    )
    counters = build_token_counters(texts)
    tokens = _read_json(artifact_dir / org_entry["lv_tokens"])
    global_indices = _read_json(artifact_dir / org_entry["gv_index_vector"])
    if len(tokens) != len(global_indices):
        raise ValueError(
            "Cross local-vocabulary token count does not match GV index-vector count"
        )

    matrix = build_feature_matrix_from_counters(counters, tokens, mode="tf")
    matrix_path = output_dir / f"org_{org_index}_cross_tf.npz"
    sparse.save_npz(matrix_path, matrix)

    nonzero_path = output_dir / f"org_{org_index}_cross_tf_nonzero.csv"
    coordinate_matrix = matrix.tocoo()
    with nonzero_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "org_index",
                "row_index",
                "internal_log_id",
                "local_feature_index",
                "global_feature_index",
                "cross_token",
                "tf",
            ]
        )
        for row, column, value in zip(
            coordinate_matrix.row,
            coordinate_matrix.col,
            coordinate_matrix.data,
        ):
            writer.writerow(
                [
                    org_index,
                    int(row),
                    f"org_{org_index}_row_{int(row)}",
                    int(column),
                    int(global_indices[column]),
                    tokens[column],
                    float(value),
                ]
            )

    active_rows = np.asarray(matrix.getnnz(axis=0)).ravel()
    total_tf = np.asarray(matrix.sum(axis=0)).ravel()
    summary_path = output_dir / f"org_{org_index}_cross_tf_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "local_feature_index",
                "global_feature_index",
                "cross_token",
                "active_rows",
                "total_tf",
            ]
        )
        for column, token in enumerate(tokens):
            writer.writerow(
                [
                    column,
                    global_indices[column],
                    token,
                    int(active_rows[column]),
                    float(total_tf[column]),
                ]
            )

    observed_values = sorted({float(value) for value in matrix.data})
    return (
        matrix_path,
        nonzero_path,
        summary_path,
        matrix.shape,
        matrix.nnz,
        observed_values,
    )


def main() -> None:
    args = _parse_args()
    artifact_dir = args.artifact_dir
    if not artifact_dir.is_absolute():
        artifact_dir = (Path.cwd() / artifact_dir).resolve()
    output_dir = args.output_dir or artifact_dir
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    matrix_path, nonzero_path, summary_path, shape, nnz, values = (
        reconstruct_cross_tf(
            artifact_dir=artifact_dir,
            org_index=args.org_index,
            label=args.label,
            output_dir=output_dir,
        )
    )
    print(f"Matrix shape: {shape}")
    print(f"Nonzero TF values: {nnz}")
    print(f"Observed nonzero values: {values}")
    print(f"Matrix: {matrix_path}")
    print(f"Nonzero entries: {nonzero_path}")
    print(f"Feature summary: {summary_path}")


if __name__ == "__main__":
    main()
