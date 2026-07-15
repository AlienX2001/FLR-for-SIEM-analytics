from pathlib import Path
import argparse
import json

def comparison(path: Path) -> None:
    file_path = Path(path)
    file = file_path.open("r", encoding="utf-8")
    data = json.load(file)
    groundtruth_count = 0
    test_count = 0
    intersection_count = 0
    only_in_groundtruth_count = 0
    only_in_test_count = 0
    for k in data.keys():
        groundtruth_count += data[k]['groundtruth_count']
        test_count += data[k]['test_count']
        intersection_count += data[k]['intersection_count']
        only_in_groundtruth_count += len(data[k]['only_in_groundtruth'])
        only_in_test_count += len(data[k]['only_in_test'])
    print(f"File: {file_path}")
    print(f"Indicators in groundtruth: {groundtruth_count}")
    print(f"Indicators in testing: {test_count}")
    print(f"Intersection of indicators: {intersection_count}")
    print(f"Indicators only in groundtruth: {only_in_groundtruth_count}")
    print(f"Indicators only in test: {only_in_test_count}")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reports from IoC comparison jsons",
        epilog=(
            "Example: python generate_comparison_report.py "
            "--file comparison.json"
        ),
    )
    parser.add_argument("--file", required=True, type=Path, help="Path to the comparison JSON file")
    return parser.parse_args()

def main():
    args = parse_args()
    print(comparison(args.file))


if __name__ == "__main__":
    raise SystemExit(main())

