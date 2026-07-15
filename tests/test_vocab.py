from __future__ import annotations

from federated_lr_pipeline.prf import derive_prf_key, tag_vocabulary
from federated_lr_pipeline.vocab import construct_global_vocabulary, generate_local_vocabulary


def test_local_vocabulary_respects_df_top_k_and_tie_breaking() -> None:
    texts = [
        "beta alpha common",
        "beta gamma common",
        "alpha gamma common",
        "delta common",
    ]

    vocab = generate_local_vocabulary(texts, num_features=3, min_df=2, max_df=3)

    assert vocab.tokens == ["alpha", "beta", "gamma"]
    assert "common" not in vocab.tokens
    assert "delta" not in vocab.tokens


def test_global_vocabulary_union_sorting_and_index_vectors() -> None:
    key = derive_prf_key(7)
    org0 = tag_vocabulary(["alpha", "beta"], key)
    org1 = tag_vocabulary(["beta", "gamma"], key)

    gv, index_vectors = construct_global_vocabulary([org0, org1])

    assert gv == sorted(set(org0 + org1))
    assert len(gv) == 3
    assert index_vectors == [
        [gv.index(org0[0]), gv.index(org0[1])],
        [gv.index(org1[0]), gv.index(org1[1])],
    ]
