from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_lr_pipeline.data import NUMERIC_LOG_COLUMNS, _canonical_numeric_value
from federated_lr_pipeline.feature_schemas import NETWORK_ATTRIBUTES, SYSTEM_ATTRIBUTES
from federated_lr_pipeline.specialized_models import field_aware_tokens
from federated_lr_pipeline.vocab import tokenize


ARTIFACT_DIR = PROJECT_ROOT / "outputs" / "orgAB"
EXAMPLE_DIR = PROJECT_ROOT / "examples"

EXAMPLE_SPECS = (
    {
        "org_index": 0,
        "dataset": "CICAPT-IIOT2024",
        "slug": "cicapt_iiot2024",
        "rows": ((1, "system"), (9, "system"), (912, "network"), (918, "network")),
    },
    {
        "org_index": 1,
        "dataset": "CIC-IDS2018",
        "slug": "cic_ids2018",
        "rows": (
            (71750, "system"),
            (143662, "system"),
            (14, "network"),
            (22, "network"),
        ),
    },
)


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _generated_words(row: pd.Series, subcategory: str, row_index: int) -> list[str]:
    attributes = SYSTEM_ATTRIBUTES if subcategory == "system" else NETWORK_ATTRIBUTES
    field_tokens: list[str] = []
    for column in attributes:
        if column not in row.index:
            continue
        value = row[column]
        if value == "" or pd.isna(value):
            continue
        if column in NUMERIC_LOG_COLUMNS:
            value = _canonical_numeric_value(value, column=column, row_index=row_index)
        field_tokens.extend(field_aware_tokens(column, value))
    return list(dict.fromkeys(tokenize(" ".join(field_tokens))))


def main() -> None:
    run_config = json.loads((ARTIFACT_DIR / "run_config.json").read_text(encoding="utf-8"))
    org_data = run_config["org_data"]
    org_groundtruth = run_config["org_groundtruth"]
    label_column = run_config["label_column"]

    for spec in EXAMPLE_SPECS:
        org_index = int(spec["org_index"])
        dataset_name = str(spec["dataset"])
        slug = str(spec["slug"])
        selected_rows = tuple(spec["rows"])

        log_path = (PROJECT_ROOT / org_data[org_index]).resolve()
        groundtruth_path = (PROJECT_ROOT / org_groundtruth[org_index]).resolve()
        logs = pd.read_csv(log_path, dtype=str, keep_default_na=False)
        groundtruth = pd.read_csv(groundtruth_path, dtype=str, keep_default_na=False)
        if len(logs) != len(groundtruth):
            raise ValueError(f"Row count mismatch for organization {org_index}")

        csv_records: list[dict[str, Any]] = []
        log_and_vocab_records: list[dict[str, Any]] = []
        vocabulary_usage: dict[tuple[str, str], set[str]] = defaultdict(set)
        vocabulary_tags: dict[tuple[str, str], str] = {}

        artifact_maps: dict[str, dict[str, str]] = {}
        for subcategory in ("system", "network"):
            token_path = ARTIFACT_DIR / f"org_{org_index}_benign_{subcategory}_lv_tokens.json"
            tag_path = ARTIFACT_DIR / f"org_{org_index}_benign_{subcategory}_lv_tags.json"
            local_tokens = json.loads(token_path.read_text(encoding="utf-8"))
            local_tags = json.loads(tag_path.read_text(encoding="utf-8"))
            if len(local_tokens) != len(local_tags):
                raise ValueError(f"Token/tag length mismatch for {token_path.name}")
            artifact_maps[subcategory] = dict(zip(local_tokens, local_tags))

        for row_index, subcategory in selected_rows:
            raw_row = logs.iloc[row_index]
            original_label = groundtruth.iloc[row_index][label_column]
            normalized_label = str(original_label).strip().lower()
            if normalized_label != "benign":
                raise ValueError(
                    f"Expected a benign example at org {org_index}, row {row_index}; "
                    f"found {original_label!r}"
                )

            internal_log_id = f"org_{org_index}_row_{row_index}"
            generated_words = _generated_words(raw_row, subcategory, row_index)
            tag_map = artifact_maps[subcategory]
            retained_words = [word for word in generated_words if word in tag_map]
            pruned_words = [word for word in generated_words if word not in tag_map]

            raw_values = {column: raw_row[column] for column in logs.columns}
            csv_records.append(
                {
                    "org_index": org_index,
                    "row_index": row_index,
                    "source_data_row_number": row_index + 1,
                    "source_line_number": row_index + 2,
                    "internal_log_id": internal_log_id,
                    "example_category": subcategory,
                    "ground_truth_label": original_label,
                    **raw_values,
                }
            )
            log_and_vocab_records.append(
                {
                    "org_index": org_index,
                    "row_index": row_index,
                    "source_data_row_number": row_index + 1,
                    "source_line_number": row_index + 2,
                    "internal_log_id": internal_log_id,
                    "subcategory": subcategory,
                    "ground_truth_label": normalized_label,
                    "raw_log": raw_values,
                    "generated_vocabulary_words": generated_words,
                    "retained_local_vocabulary_words": retained_words,
                    "generated_but_pruned_words": pruned_words,
                }
            )
            for word in retained_words:
                key = (subcategory, word)
                vocabulary_usage[key].add(internal_log_id)
                vocabulary_tags[key] = tag_map[word]

        prefix = f"org_{org_index}_{slug}"
        pd.DataFrame(csv_records).to_csv(
            EXAMPLE_DIR / f"{prefix}_selected_logs.csv",
            index=False,
        )
        _json_dump(
            EXAMPLE_DIR / f"{prefix}_logs_and_vocab.json",
            {
                "dataset": dataset_name,
                "org_index": org_index,
                "source_log_csv": str(log_path),
                "source_groundtruth_csv": str(groundtruth_path),
                "model_artifact_directory": str(ARTIFACT_DIR),
                "notes": [
                    "generated_vocabulary_words contains all unique base field-aware tokens for the row.",
                    "retained_local_vocabulary_words contains the generated words retained by the saved benign specialist local vocabulary.",
                    "generated_but_pruned_words did not become local model coordinates and therefore have no saved PRF tag artifact.",
                    "Causal 15-minute context tokens are not reconstructed in this example export.",
                ],
                "records": log_and_vocab_records,
            },
        )

        vocabulary = []
        for subcategory, word in sorted(vocabulary_tags):
            vocabulary.append(
                {
                    "subcategory": subcategory,
                    "prf_namespace": f"{subcategory}|{word}",
                    "vocabulary_word": word,
                    "prf_tag": vocabulary_tags[(subcategory, word)],
                    "used_by_internal_log_ids": sorted(vocabulary_usage[(subcategory, word)]),
                }
            )
        _json_dump(
            EXAMPLE_DIR / f"{prefix}_vocab_and_prfs.json",
            {
                "dataset": dataset_name,
                "org_index": org_index,
                "prf": "HMAC-SHA256",
                "prf_namespace_format": "subcategory|canonical_token",
                "source_artifacts": {
                    subcategory: {
                        "tokens": f"org_{org_index}_benign_{subcategory}_lv_tokens.json",
                        "tags": f"org_{org_index}_benign_{subcategory}_lv_tags.json",
                    }
                    for subcategory in ("system", "network")
                },
                "vocabulary": vocabulary,
            },
        )


if __name__ == "__main__":
    main()
