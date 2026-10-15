from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from federated_lr_pipeline.config import PipelineConfig
from federated_lr_pipeline.data import OrgDataset
from federated_lr_pipeline.local_training import build_token_counters
from federated_lr_pipeline.prf import derive_prf_key
from federated_lr_pipeline.specialized_models import (
    SpecialistState,
    SpecialistUpdate,
    _aggregate_specialist_updates,
    _binary_class_weights,
    train_specialist_round,
)
from federated_lr_pipeline.vocab import LocalVocabulary


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


def test_round_zero_returns_tfidf_weights_to_server_tf_coordinates() -> None:
    texts = ["common rare", "common"]
    vocabulary = ["common", "rare"]
    initial_weights = np.array([1.5, -2.0])
    dataset = OrgDataset(
        org_index=0,
        log_path=None,  # type: ignore[arg-type]
        groundtruth_path=None,  # type: ignore[arg-type]
        logs_df=None,  # type: ignore[arg-type]
        groundtruth_df=None,  # type: ignore[arg-type]
        text_column="",
        text_columns=[],
        label_column="label",
        texts=texts,
        labels=["target", "other"],
        row_indices=[0, 1],
        internal_log_ids=["org_0_row_0", "org_0_row_1"],
        source_log_id_column=None,
    )
    state = SpecialistState(
        label="target",
        subcategory="network",
        org_texts=[texts],
        org_token_counters=[build_token_counters(texts)],
        local_vocabularies=[
            LocalVocabulary(
                tokens=vocabulary,
                document_frequency={"common": 2, "rare": 1},
                effective_min_df=1,
                effective_max_df_count=2,
                used_fallback=False,
            )
        ],
        org_vocab_tokens=[vocabulary],
        org_tag_lists=[["tag-common", "tag-rare"]],
        global_tags=["tag-common", "tag-rare"],
        org_index_vectors=[[0, 1]],
        weights=initial_weights.copy(),
        bias=0.25,
        missing_columns_by_org={0: []},
        prf_key=derive_prf_key(42),
    )
    config = PipelineConfig(
        org_data=[],
        org_groundtruth=[],
        num_features=2,
        federation_iterations=1,
        local_epochs=0,
        class_weight="none",
    )

    metrics = train_specialist_round(
        state=state,
        org_datasets=[dataset],
        encoded_labels_by_org=[np.array([0, 1])],
        label_index=0,
        splits=[
            SimpleNamespace(
                train_indices=np.array([0, 1]),
                test_indices=np.array([], dtype=int),
            )
        ],
        mode="tfidf",
        round_index=0,
        total_rounds=1,
        config=config,
    )

    np.testing.assert_allclose(state.weights, initial_weights)
    assert metrics["local_metrics"][0]["local_feature_mode"] == "tfidf"
    assert metrics["local_metrics"][0]["weight_coordinate_system"] == "tf"


def test_binary_class_weight_tuning_tempers_and_caps_local_imbalance() -> None:
    targets = np.array([1] + [0] * 99)

    assert _binary_class_weights(targets, strength=0.0) == (1.0, 1.0)

    positive_weight, negative_weight = _binary_class_weights(
        targets,
        strength=1.0,
        max_ratio=5.0,
    )

    np.testing.assert_allclose(positive_weight / negative_weight, 5.0)
    np.testing.assert_allclose(
        (positive_weight + 99.0 * negative_weight) / 100.0,
        1.0,
    )


def test_specialist_round_computes_class_weights_per_organization() -> None:
    texts_by_org = [["shared"] * 4, ["shared"] * 4]
    datasets = [
        OrgDataset(
            org_index=org_index,
            log_path=None,  # type: ignore[arg-type]
            groundtruth_path=None,  # type: ignore[arg-type]
            logs_df=None,  # type: ignore[arg-type]
            groundtruth_df=None,  # type: ignore[arg-type]
            text_column="",
            text_columns=[],
            label_column="label",
            texts=texts,
            labels=[],
            row_indices=list(range(4)),
            internal_log_ids=[f"org_{org_index}_row_{index}" for index in range(4)],
            source_log_id_column=None,
        )
        for org_index, texts in enumerate(texts_by_org)
    ]
    vocabulary = ["shared"]
    state = SpecialistState(
        label="target",
        subcategory="system",
        org_texts=texts_by_org,
        org_token_counters=[build_token_counters(texts) for texts in texts_by_org],
        local_vocabularies=[
            LocalVocabulary(
                tokens=vocabulary,
                document_frequency={"shared": 4},
                effective_min_df=1,
                effective_max_df_count=4,
            )
            for _ in datasets
        ],
        org_vocab_tokens=[vocabulary, vocabulary],
        org_tag_lists=[["tag-shared"], ["tag-shared"]],
        global_tags=["tag-shared"],
        org_index_vectors=[[0], [0]],
        weights=np.array([0.0]),
        bias=0.0,
        missing_columns_by_org={0: [], 1: []},
        prf_key=derive_prf_key(42),
    )
    config = PipelineConfig(
        org_data=[],
        org_groundtruth=[],
        num_features=1,
        federation_iterations=1,
        local_epochs=0,
        class_weight="balanced",
        class_weight_strength=1.0,
    )

    metrics = train_specialist_round(
        state=state,
        org_datasets=datasets,
        encoded_labels_by_org=[
            np.array([0, 1, 1, 1]),
            np.array([0, 0, 0, 1]),
        ],
        label_index=0,
        splits=[
            SimpleNamespace(
                train_indices=np.arange(4),
                test_indices=np.array([], dtype=int),
            ),
            SimpleNamespace(
                train_indices=np.arange(4),
                test_indices=np.array([], dtype=int),
            ),
        ],
        mode="tf",
        round_index=0,
        total_rounds=1,
        config=config,
    )

    org0, org1 = metrics["local_metrics"]
    assert (org0["positive_class_count"], org0["negative_class_count"]) == (1, 3)
    assert (org1["positive_class_count"], org1["negative_class_count"]) == (3, 1)
    np.testing.assert_allclose(org0["positive_class_weight"], 2.0)
    np.testing.assert_allclose(org0["negative_class_weight"], 2.0 / 3.0)
    np.testing.assert_allclose(org1["positive_class_weight"], 2.0 / 3.0)
    np.testing.assert_allclose(org1["negative_class_weight"], 2.0)
