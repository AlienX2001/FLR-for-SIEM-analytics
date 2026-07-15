from __future__ import annotations

import hashlib
import hmac


def derive_prf_key(seed: int) -> bytes:
    return hashlib.sha256(f"federated-lr-prf-key:{seed}".encode("utf-8")).digest()


def hmac_sha256_tag(key: bytes, word: str) -> str:
    return hmac.new(key, word.encode("utf-8"), hashlib.sha256).hexdigest()


def tag_vocabulary(tokens: list[str], key: bytes) -> list[str]:
    return [hmac_sha256_tag(key, token) for token in tokens]


def namespaced_token(label: str, subcategory: str, token: str) -> str:
    return f"{label}|{subcategory}|{token}"


def tag_namespaced_vocabulary(
    tokens: list[str],
    key: bytes,
    *,
    label: str,
    subcategory: str,
) -> list[str]:
    return [
        hmac_sha256_tag(key, namespaced_token(label, subcategory, token))
        for token in tokens
    ]
