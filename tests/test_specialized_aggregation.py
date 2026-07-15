from __future__ import annotations

import numpy as np

from federated_lr_pipeline.specialized_models import (
    SpecialistUpdate,
    _aggregate_specialist_updates,
)


def test_specialist_aggregation_handles_missing_rows_and_weights_by_samples() -> None:
    previous_weights = np.array([1.0, 2.0, 3.0])
    previous_bias = 0.5
    updates = [
        SpecialistUpdate(
            org_index=0,
            index_vector=[0, 2],
            weights=np.array([2.0, 4.0]),
            bias=1.0,
            num_examples=2,
            loss=0.0,
            accuracy=1.0,
        ),
        SpecialistUpdate(
            org_index=1,
            index_vector=[1],
            weights=np.array([4.0]),
            bias=0.0,
            num_examples=1,
            loss=0.0,
            accuracy=1.0,
        ),
    ]

    weights, bias = _aggregate_specialist_updates(
        previous_weights,
        previous_bias,
        updates,
        weighting="sample_size",
    )

    np.testing.assert_allclose(weights[0], (2 * 2.0 + previous_weights[0]) / 3)
    np.testing.assert_allclose(weights[1], (2 * previous_weights[1] + 4.0) / 3)
    np.testing.assert_allclose(weights[2], (2 * 4.0 + previous_weights[2]) / 3)
    np.testing.assert_allclose(bias, (2 * 1.0 + 0.0) / 3)
