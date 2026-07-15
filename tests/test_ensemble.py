from __future__ import annotations

import numpy as np

from federated_lr_pipeline.ensemble import ManualLogitFusion, fused_probabilities
from federated_lr_pipeline.model import softmax


def test_manual_logit_fusion_combines_label_logits_before_softmax() -> None:
    fusion = ManualLogitFusion(
        labels=["benign", "data_exfiltration"],
        subcategories_by_label={
            "benign": ["system", "network"],
            "data_exfiltration": ["system", "network", "cross"],
        },
        weights_by_label={
            "benign": {"bias": 0.5, "system": 1.0, "network": 1.0},
            "data_exfiltration": {
                "bias": -1.0,
                "system": 0.8,
                "network": 0.8,
                "cross": 1.2,
            },
        },
    )
    logits = {
        "benign": {
            "system": np.array([1.0]),
            "network": np.array([0.5]),
        },
        "data_exfiltration": {
            "system": np.array([1.5]),
            "network": np.array([1.0]),
            "cross": np.array([2.0]),
        },
    }

    label_logits, probabilities = fused_probabilities(fusion, logits)

    expected = np.array([[2.0, 3.4]])
    np.testing.assert_allclose(label_logits, expected)
    np.testing.assert_allclose(probabilities, softmax(expected))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.array([1.0]))


def test_manual_logit_fusion_save_and_load(tmp_path) -> None:
    path = tmp_path / "fusion.json"
    fusion = ManualLogitFusion(
        labels=["a"],
        subcategories_by_label={"a": ["system"]},
        weights_by_label={"a": {"bias": -1.0, "system": 2.0}},
    )
    fusion.save(path)

    loaded = ManualLogitFusion.load(path)

    logits = {"a": {"system": np.array([3.0])}}
    np.testing.assert_allclose(loaded.predict_logits(logits), np.array([[5.0]]))
