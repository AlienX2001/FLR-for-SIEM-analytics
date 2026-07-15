from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from federated_lr_pipeline.data import OrgDataset
from federated_lr_pipeline.prf import derive_prf_key, tag_namespaced_vocabulary
from federated_lr_pipeline.specialized_models import initialize_specialist


@dataclass(frozen=True)
class Split:
    train_indices: np.ndarray
    test_indices: np.ndarray


def _dataset() -> OrgDataset:
    return OrgDataset(
        org_index=0,
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
        internal_log_ids=["org_0_row_0"],
        source_log_id_column=None,
    )


def test_prf_namespacing_separates_same_token_across_labels() -> None:
    key = derive_prf_key(42)

    data_tag = tag_namespaced_vocabulary(
        ["large_upload"],
        key,
        label="data_exfiltration",
        subcategory="network",
    )[0]
    credential_tag = tag_namespaced_vocabulary(
        ["large_upload"],
        key,
        label="credential_attack",
        subcategory="network",
    )[0]

    assert data_tag != credential_tag


def test_label_subcategory_specialists_have_separate_gvs() -> None:
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
    assert data_state.global_tags != credential_state.global_tags
    assert data_state.org_index_vectors is not credential_state.org_index_vectors
