from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)
TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True)
class LocalVocabulary:
    tokens: list[str]
    document_frequency: dict[str, int]
    effective_min_df: int
    effective_max_df_count: int
    used_fallback: bool = False


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_part in str(text).lower().split():
        part = raw_part.strip().strip("\"'`“”‘’()[]{}<>;,")
        if not part:
            continue
        if "=" in part or ":" in part:
            cleaned = part.strip(".,;")
            if len(cleaned) >= 2:
                tokens.append(cleaned)
            continue
        tokens.extend(
            token for token in TOKEN_SPLIT_RE.split(part) if len(token) >= 2
        )
    return tokens


def compute_document_frequency_from_tokenized(
    tokenized_documents: Iterable[Iterable[str]],
) -> Counter[str]:
    df: Counter[str] = Counter()
    for tokens in tokenized_documents:
        df.update(set(tokens))
    return df


def compute_document_frequency(texts: list[str]) -> Counter[str]:
    return compute_document_frequency_from_tokenized(tokenize(text) for text in texts)


def resolve_max_df_count(max_df: float, document_count: int) -> int:
    if document_count <= 0:
        return 0
    if 0 < max_df <= 1:
        return max(1, int(np.floor(max_df * document_count)))
    return int(max_df)


def _select_candidates(
    df: Counter[str], min_df: int, max_df_count: int, num_features: int
) -> list[str]:
    candidates = [
        token
        for token, count in df.items()
        if count >= min_df and count <= max_df_count
    ]
    candidates.sort(key=lambda token: (-df[token], token))
    return candidates[:num_features]


def generate_local_vocabulary(
    texts: list[str],
    num_features: int,
    min_df: int = 2,
    max_df: float = 0.95,
    org_index: int | None = None,
) -> LocalVocabulary:
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if min_df < 1:
        raise ValueError("min_df must be at least 1")
    if max_df <= 0:
        raise ValueError("max_df must be positive")

    df = compute_document_frequency(texts)
    max_df_count = resolve_max_df_count(max_df, len(texts))
    tokens = _select_candidates(df, min_df, max_df_count, num_features)
    used_fallback = False

    requested_available = min(num_features, len(df))
    if len(tokens) < requested_available and df:
        used_fallback = True
        LOGGER.warning(
            "Organization %s had too few vocabulary terms satisfying min_df=%s, max_df=%s. "
            "Falling back to min_df=1 and max_df=1.0.",
            org_index if org_index is not None else "?",
            min_df,
            max_df,
        )
        min_df = 1
        max_df_count = len(texts)
        tokens = _select_candidates(df, min_df, max_df_count, num_features)

    return LocalVocabulary(
        tokens=tokens,
        document_frequency={token: int(df[token]) for token in sorted(df)},
        effective_min_df=min_df,
        effective_max_df_count=max_df_count,
        used_fallback=used_fallback,
    )


def generate_local_vocabulary_from_token_counters(
    token_counters: list[Mapping[str, int]],
    num_features: int,
    min_df: int = 2,
    max_df: float = 0.95,
    org_index: int | None = None,
) -> LocalVocabulary:
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if min_df < 1:
        raise ValueError("min_df must be at least 1")
    if max_df <= 0:
        raise ValueError("max_df must be positive")

    df = compute_document_frequency_from_tokenized(counter.keys() for counter in token_counters)
    max_df_count = resolve_max_df_count(max_df, len(token_counters))
    tokens = _select_candidates(df, min_df, max_df_count, num_features)
    used_fallback = False

    requested_available = min(num_features, len(df))
    if len(tokens) < requested_available and df:
        used_fallback = True
        LOGGER.warning(
            "Organization %s had too few vocabulary terms satisfying min_df=%s, max_df=%s. "
            "Falling back to min_df=1 and max_df=1.0.",
            org_index if org_index is not None else "?",
            min_df,
            max_df,
        )
        min_df = 1
        max_df_count = len(token_counters)
        tokens = _select_candidates(df, min_df, max_df_count, num_features)

    return LocalVocabulary(
        tokens=tokens,
        document_frequency={token: int(df[token]) for token in sorted(df)},
        effective_min_df=min_df,
        effective_max_df_count=max_df_count,
        used_fallback=used_fallback,
    )


def construct_global_vocabulary(
    org_tag_lists: list[list[str]],
) -> tuple[list[str], list[list[int]]]:
    global_tags = sorted({tag for tags in org_tag_lists for tag in tags})
    tag_to_index = {tag: index for index, tag in enumerate(global_tags)}
    index_vectors = [[tag_to_index[tag] for tag in tags] for tags in org_tag_lists]
    return global_tags, index_vectors
