from __future__ import annotations

import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
PRF_TAG_RE = re.compile(r"[0-9a-f]{64}")


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.reader(source))


def _write_csv(path: Path, rows: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        csv.writer(destination).writerows(rows)


def main() -> None:
    org_prf_paths = {
        "orgA": EXAMPLES_DIR / "orgA_PRFs.csv",
        "orgB": EXAMPLES_DIR / "orgB_PRFs.csv",
    }
    org_lv_paths = {
        "orgA": EXAMPLES_DIR / "orgA_LV.csv",
        "orgB": EXAMPLES_DIR / "orgB_LV.csv",
    }

    prf_rows_by_org: dict[str, list[list[str]]] = {}
    all_tags: set[str] = set()
    for organization, prf_path in org_prf_paths.items():
        prf_rows = _read_csv(prf_path)
        lv_rows = _read_csv(org_lv_paths[organization])
        if len(prf_rows) != len(lv_rows) or any(
            len(prf_row) != len(lv_row)
            for prf_row, lv_row in zip(prf_rows, lv_rows)
        ):
            raise ValueError(
                f"{prf_path.name} does not have the same shape as "
                f"{org_lv_paths[organization].name}"
            )
        for row_index, row in enumerate(prf_rows):
            for column_index, tag in enumerate(row):
                if PRF_TAG_RE.fullmatch(tag) is None:
                    raise ValueError(
                        f"Invalid PRF tag in {prf_path.name} at "
                        f"row {row_index}, column {column_index}"
                    )
                all_tags.add(tag)
        prf_rows_by_org[organization] = prf_rows

    global_vocabulary = sorted(all_tags)
    tag_to_index = {tag: index for index, tag in enumerate(global_vocabulary)}
    (EXAMPLES_DIR / "GV.txt").write_text(
        "".join(f"{tag}\n" for tag in global_vocabulary),
        encoding="utf-8",
    )

    for organization, prf_rows in prf_rows_by_org.items():
        index_rows = [
            [tag_to_index[tag] for tag in prf_row]
            for prf_row in prf_rows
        ]
        _write_csv(
            EXAMPLES_DIR / f"{organization}_GV_index_vectors.csv",
            index_rows,
        )

    print(
        f"GV.txt: {len(global_vocabulary)} unique sorted tags; "
        + ", ".join(
            f"{organization}={sum(map(len, rows))} mapped positions"
            for organization, rows in prf_rows_by_org.items()
        )
    )


if __name__ == "__main__":
    main()
