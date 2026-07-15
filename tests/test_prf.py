from __future__ import annotations

from federated_lr_pipeline.prf import derive_prf_key, hmac_sha256_tag


def test_prf_is_deterministic_for_same_word_and_key() -> None:
    key = derive_prf_key(42)

    assert hmac_sha256_tag(key, "failed") == hmac_sha256_tag(key, "failed")


def test_prf_differs_for_different_words() -> None:
    key = derive_prf_key(42)

    assert hmac_sha256_tag(key, "failed") != hmac_sha256_tag(key, "success")


def test_prf_output_is_sha256_hex() -> None:
    key = derive_prf_key(42)
    tag = hmac_sha256_tag(key, "token")

    assert len(tag) == 64
    int(tag, 16)
