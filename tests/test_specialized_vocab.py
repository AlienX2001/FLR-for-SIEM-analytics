from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from federated_lr_pipeline.data import OrgDataset
from federated_lr_pipeline.prf import derive_prf_key, tag_namespaced_vocabulary
from federated_lr_pipeline.local_training import build_token_counters
from federated_lr_pipeline.specialized_models import (
    initialize_specialist,
    logits_for_org_rows,
    top_contributions,
)


@dataclass(frozen=True)
class Split:
    train_indices: np.ndarray
    test_indices: np.ndarray


def _dataset(org_index: int = 0) -> OrgDataset:
    return OrgDataset(
        org_index=org_index,
        log_path=None,  # type: ignore[arg-type]
        groundtruth_path=None,  # type: ignore[arg-type]
        logs_df=None,  # type: ignore[arg-type]
        groundtruth_df=None,  # type: ignore[arg-type]
        text_column="",
        text_columns=[],
        label_column="label",
        texts=[""],
        labels=["benign"],
        row_indices=[0],
        internal_log_ids=[f"org_{org_index}_row_0"],
        source_log_id_column=None,
    )


def test_prf_namespacing_shares_same_token_across_labels() -> None:
    key = derive_prf_key(42)

    data_tag = tag_namespaced_vocabulary(
        ["large_upload"],
        key,
        subcategory="network",
    )[0]
    credential_tag = tag_namespaced_vocabulary(
        ["large_upload"],
        key,
        subcategory="network",
    )[0]

    assert data_tag == credential_tag


def test_prf_namespacing_separates_subcategories() -> None:
    key = derive_prf_key(42)

    network_tag = tag_namespaced_vocabulary(
        ["shared=value"], key, subcategory="network"
    )[0]
    system_tag = tag_namespaced_vocabulary(
        ["shared=value"], key, subcategory="system"
    )[0]

    assert network_tag != system_tag


def test_labels_in_same_subcategory_share_gv_axis() -> None:
    dataset = _dataset()
    split = Split(train_indices=np.array([0]), test_indices=np.array([], dtype=int))
    key = derive_prf_key(42)

    data_state = initialize_specialist(
        label="data_exfiltration",
        subcategory="network",
        org_datasets=[dataset],
        org_texts=[["dst_port=443 shared=value"]],
        missing_columns_by_org={0: []},
        splits=[split],
        num_features=10,
        min_df=1,
        max_df=1.0,
        vocabulary_source="train",
        prf_key=key,
        seed=42,
    )
    credential_state = initialize_specialist(
        label="credential_attack",
        subcategory="network",
        org_datasets=[dataset],
        org_texts=[["dst_port=443 shared=value"]],
        missing_columns_by_org={0: []},
        splits=[split],
        num_features=10,
        min_df=1,
        max_df=1.0,
        vocabulary_source="train",
        prf_key=key,
        seed=42,
    )

    assert data_state.org_vocab_tokens[0] == credential_state.org_vocab_tokens[0]
    assert data_state.global_tags == credential_state.global_tags
    assert data_state.org_index_vectors == credential_state.org_index_vectors


def test_inference_uses_global_token_missing_from_organization_lv() -> None:
    key = derive_prf_key(42)
    split = Split(train_indices=np.array([0]), test_indices=np.array([], dtype=int))
    state = initialize_specialist(
        label="credential_access",
        subcategory="network",
        org_datasets=[_dataset(0), _dataset(1)],
        org_texts=[["alpha"], ["beta"]],
        missing_columns_by_org={0: [], 1: []},
        splits=[split, split],
        num_features=1,
        min_df=1,
        max_df=1.0,
        vocabulary_source="train",
        prf_key=key,
        seed=42,
    )
    assert "beta" not in state.org_vocab_tokens[0]
    beta_tag = tag_namespaced_vocabulary(
        ["beta"], key, subcategory="network"
    )[0]
    beta_index = state.global_tags.index(beta_tag)
    state.weights[:] = 0.0
    state.weights[beta_index] = 2.5
    state.bias = 0.0

    query_counter = build_token_counters(["beta"])[0]
    state.org_texts[0] = ["beta"]
    state.org_token_counters[0] = [query_counter]
    X, logits = logits_for_org_rows(state, 0, np.array([0]))

    assert X.shape == (1, len(state.global_tags))
    assert X[0, beta_index] == 1.0
    np.testing.assert_allclose(logits, [2.5])
    contributions = top_contributions(
        state=state,
        org_position=0,
        feature_row=X[0],
        row_token_counter=query_counter,
        debug_plaintext_vocab=True,
    )
    assert contributions[0]["token"] == "beta"
    assert contributions[0]["tag"] == beta_tag
    assert contributions[0]["gv_index"] == beta_index
    assert contributions[0]["contribution"] == 2.5


def test_inference_still_drops_token_absent_from_global_vocabulary() -> None:
    key = derive_prf_key(42)
    split = Split(train_indices=np.array([0]), test_indices=np.array([], dtype=int))
    state = initialize_specialist(
        label="credential_access",
        subcategory="network",
        org_datasets=[_dataset(0)],
        org_texts=[["alpha"]],
        missing_columns_by_org={0: []},
        splits=[split],
        num_features=1,
        min_df=1,
        max_df=1.0,
        vocabulary_source="train",
        prf_key=key,
        seed=42,
    )
    state.bias = 0.75
    state.org_texts[0] = ["never-seen"]
    state.org_token_counters[0] = build_token_counters(["never-seen"])

    X, logits = logits_for_org_rows(state, 0, np.array([0]))

    assert X.nnz == 0
    np.testing.assert_allclose(logits, [0.75])
