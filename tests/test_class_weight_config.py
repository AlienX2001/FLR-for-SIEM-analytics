from __future__ import annotations

import pytest

from federated_lr_pipeline.config import parse_args


def _training_args() -> list[str]:
    return [
        "--org-data",
        "logs.csv",
        "--org-groundtruth",
        "labels.csv",
        "--num-features",
        "10",
        "--federation-iterations",
        "1",
        "--output-dir",
        "output",
    ]


def test_class_weight_tuning_options_are_parsed() -> None:
    config = parse_args(
        _training_args()
        + [
            "--class-weight-strength",
            "0.5",
            "--class-weight-max-ratio",
            "8",
        ]
    )

    assert config.class_weight_strength == 0.5
    assert config.class_weight_max_ratio == 8.0


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--class-weight-strength", "-0.1"),
        ("--class-weight-strength", "1.1"),
        ("--class-weight-max-ratio", "0.5"),
        ("--class-weight-max-ratio", "nan"),
    ],
)
def test_invalid_class_weight_tuning_options_are_rejected(
    flag: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(_training_args() + [flag, value])
