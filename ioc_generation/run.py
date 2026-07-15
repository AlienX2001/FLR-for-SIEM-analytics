from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ioc_generation.correlate import generate_ioc_outputs
from ioc_generation.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate STIX v2 IoC templates from high-risk LR predictions."
    )
    parser.add_argument("--high-risk-logs", required=True, type=Path)
    parser.add_argument("--explanations", required=True, type=Path)
    parser.add_argument("--org-data", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-column", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    output_dir = ensure_dir(args.output_dir)
    LOGGER.info("Generating IoCs from high-risk row-aligned predictions")
    generate_ioc_outputs(
        high_risk_logs=args.high_risk_logs,
        explanations=args.explanations,
        org_data=args.org_data,
        output_dir=output_dir,
        text_column=args.text_column,
    )
    LOGGER.info("Wrote IoC outputs to %s", output_dir)


if __name__ == "__main__":
    main()
