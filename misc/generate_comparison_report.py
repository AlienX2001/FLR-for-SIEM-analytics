from pathlib import Path
import argparse
import json


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def comparison(path: Path) -> None:
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for k in data.keys():
        true_positives += data[k]["intersection_count"]
        false_negatives += len(data[k]["only_in_groundtruth"])
        false_positives += len(data[k]["only_in_test"])

    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    print(f"File: {file_path}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reports from IoC comparison JSONs",
        epilog=(
            "Example: python generate_comparison_report.py "
            "--file comparison.json"
        ),
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to the comparison JSON file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison(args.file)


if __name__ == "__main__":
    raise SystemExit(main())
