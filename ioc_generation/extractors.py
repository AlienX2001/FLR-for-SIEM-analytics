from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit


URL_RE = re.compile(r"\bhttps?://[^\s\"'<>]+", re.IGNORECASE)
IPV4_RE = re.compile(
    r"(?<![A-Za-z0-9_.:/\\-])(?:\d{1,3}\.){3}\d{1,3}(?![A-Za-z0-9_.\\-])"
)
HASH_RE = re.compile(r"(?<![a-fA-F0-9])(?:[a-fA-F0-9]{64}|[a-fA-F0-9]{40}|[a-fA-F0-9]{32})(?![a-fA-F0-9])")
DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9_.:/\\-])"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}"
    r"(?![A-Za-z0-9_.\\-])"
)
IDENTIFIER_HASH_KEYS = {
    "id",
    "uuid",
    "guid",
    "normalized_id",
    "log_id",
    "logid",
    "event_id",
    "source_log_id",
    "internal_log_id",
    "entity_id",
    "source_entity_id",
    "target_entity_id",
    "x_otx_pulse_id",
}


@dataclass(frozen=True, order=True)
class IndicatorCandidate:
    indicator_type: str
    value: str


def _clean(value: str) -> str:
    return value.strip().strip(".,;:)]}'\"")


def _valid_ipv4(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


def _valid_domain(value: str) -> bool:
    domain = value.rstrip(".").lower()
    if len(domain) > 253 or "." not in domain or _valid_ipv4(domain):
        return False
    labels = domain.split(".")
    if len(labels[-1]) < 2 or not labels[-1].isalpha():
        return False
    return all(
        label
        and len(label) <= 63
        and not label.startswith("-")
        and not label.endswith("-")
        and all(character.isalnum() or character == "-" for character in label)
        for label in labels
    )


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    characters = list(text)
    for start, end in spans:
        for index in range(start, end):
            characters[index] = " "
    return "".join(characters)


def _hash_context_key(text: str, match_start: int) -> str | None:
    prefix = text[max(0, match_start - 120) : match_start]
    match = re.search(r"([A-Za-z_][A-Za-z0-9_.-]{0,80})\s*[=:]\s*[\"']?$", prefix)
    if match is None:
        return None
    return match.group(1).lower()


def _is_identifier_hash_context(text: str, match_start: int) -> bool:
    key = _hash_context_key(text, match_start)
    if key is None:
        return False
    if "hash" in key or "md5" in key or "sha" in key:
        return False
    return key in IDENTIFIER_HASH_KEYS or key.endswith("_id") or key.endswith(".id")


def _hash_type(value: str) -> str:
    length = len(value)
    if length == 32:
        return "md5"
    if length == 40:
        return "sha1"
    if length == 64:
        return "sha256"
    raise ValueError(f"Unsupported hash length: {length}")


def extract_iocs(text: str) -> list[IndicatorCandidate]:
    candidates: set[IndicatorCandidate] = set()
    url_spans: list[tuple[int, int]] = []
    for match in URL_RE.finditer(text):
        url_spans.append(match.span())
        cleaned_url = _clean(match.group(0))
        parsed = urlsplit(cleaned_url)
        host = (parsed.hostname or "").strip("[]").lower()
        if not host:
            continue
        if _valid_ipv4(host) or _valid_domain(host):
            candidates.add(IndicatorCandidate("url", cleaned_url))
        if _valid_ipv4(host):
            candidates.add(IndicatorCandidate("ipv4", host))
        elif _valid_domain(host):
            candidates.add(IndicatorCandidate("domain", host))

    masked_text = _mask_spans(text, url_spans)
    for match in IPV4_RE.findall(masked_text):
        cleaned = _clean(match)
        if _valid_ipv4(cleaned):
            candidates.add(IndicatorCandidate("ipv4", cleaned))
    for match in HASH_RE.finditer(masked_text):
        if _is_identifier_hash_context(masked_text, match.start()):
            continue
        cleaned = _clean(match.group(0)).lower()
        candidates.add(IndicatorCandidate(_hash_type(cleaned), cleaned))
    for match in DOMAIN_RE.findall(masked_text):
        cleaned = _clean(match).lower()
        if _valid_domain(cleaned):
            candidates.add(IndicatorCandidate("domain", cleaned))
    return sorted(candidates)
