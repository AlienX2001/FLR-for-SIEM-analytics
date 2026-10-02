from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_lr_pipeline.prf import (  # noqa: E402
    derive_prf_key,
    hmac_sha256_tag,
    namespaced_token,
)


EXAMPLES_DIR = PROJECT_ROOT / "examples"
ARTIFACT_DIR = PROJECT_ROOT / "outputs" / "orgAB"
EXPORTS = (
    (
        EXAMPLES_DIR / "orgA_LV.csv",
        EXAMPLES_DIR / "org_0_cicapt_iiot2024_vocab_and_prfs.json",
        EXAMPLES_DIR / "orgA_PRFs.csv",
    ),
    (
        EXAMPLES_DIR / "orgB_LV.csv",
        EXAMPLES_DIR / "org_1_cic_ids2018_vocab_and_prfs.json",
        EXAMPLES_DIR / "orgB_PRFs.csv",
    ),
)


def _subcategory_for_row(row_index: int) -> str:
    if row_index in (0, 1):
        return "system"
    if row_index in (2, 3):
        return "network"
    raise ValueError(
        "Example LV files must contain two system rows followed by two network rows"
    )


def main() -> None:
    run_config = json.loads((ARTIFACT_DIR / "run_config.json").read_text(encoding="utf-8"))
    prf_key = derive_prf_key(int(run_config["seed"]))

    for lv_path, reference_path, output_path in EXPORTS:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_tags = {
            (entry["subcategory"], entry["vocabulary_word"]): entry["prf_tag"]
            for entry in reference["vocabulary"]
        }

        with lv_path.open("r", encoding="utf-8", newline="") as source:
            lv_rows = list(csv.reader(source))

        output_rows: list[list[str]] = []
        reference_count = 0
        derived_count = 0
        for row_index, vocabulary_words in enumerate(lv_rows):
            subcategory = _subcategory_for_row(row_index)
            prf_row: list[str] = []
            for vocabulary_word in vocabulary_words:
                expected_tag = hmac_sha256_tag(
                    prf_key,
                    namespaced_token(subcategory, vocabulary_word),
                )
                reference_tag = reference_tags.get((subcategory, vocabulary_word))
                if reference_tag is not None:
                    if reference_tag != expected_tag:
                        raise ValueError(
                            f"Reference PRF mismatch for {subcategory}|{vocabulary_word}"
                        )
                    prf_row.append(reference_tag)
                    reference_count += 1
                else:
                    prf_row.append(expected_tag)
                    derived_count += 1
            output_rows.append(prf_row)

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            csv.writer(destination, quoting=csv.QUOTE_ALL).writerows(output_rows)

        print(
            f"{output_path.name}: rows={len(output_rows)}, "
            f"tags_from_reference={reference_count}, derived_tags={derived_count}"
        )


if __name__ == "__main__":
    main()
