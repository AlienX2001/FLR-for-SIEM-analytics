from __future__ import annotations

import hashlib
import ipaddress
import posixpath
import re
from pathlib import PureWindowsPath
from urllib.parse import SplitResult, urlsplit, urlunsplit


NODE_TYPE_ALIASES = {
    "ipv4": "ip_address",
    "ipv4-addr": "ip_address",
    "ipv6": "ip_address",
    "ipv6-addr": "ip_address",
    "domain-name": "domain",
    "process_name": "process",
    "executable": "process",
    "filename": "file_path",
}

HASH_TYPES = {"md5", "sha1", "sha256", "imphash", "pehash"}
MATCHABLE_NODE_TYPES = frozenset(
    {
        "platform",
        "host",
        "process",
        "process_path",
        "file_path",
        "service",
        "protocol",
        "port",
        "ip_address",
        "network",
        "domain",
        "url",
        "email",
        "vulnerability",
        "identity",
        "feature",
        *HASH_TYPES,
    }
)

ADMINISTRATIVE_RELATIONS = frozenset(
    {
        "contains",
        "indicates",
        "associated_with",
        "supported_by",
        "derived_from",
        "applicable_to",
    }
)

DISTINCTIVE_NODE_TYPES = frozenset(
    {
        "process",
        "process_path",
        "file_path",
        "service",
        "ip_address",
        "network",
        "domain",
        "url",
        "email",
        "vulnerability",
        "feature",
        *HASH_TYPES,
    }
)


def canonical_node_type(node_type: str) -> str:
    normalized = str(node_type).strip().lower().replace(" ", "_")
    return NODE_TYPE_ALIASES.get(normalized, normalized)


def canonical_domain(value: str) -> str:
    text = str(value).strip().lower().rstrip(".")
    if not text or "." not in text:
        raise ValueError(f"Invalid domain value: {value!r}")
    try:
        ipaddress.ip_address(text)
    except ValueError:
        pass
    else:
        raise ValueError(f"IP address is not a domain: {value!r}")
    try:
        ascii_domain = text.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError(f"Invalid domain value: {value!r}") from exc
    labels = ascii_domain.split(".")
    if any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or re.fullmatch(r"[a-z0-9-]+", label) is None
        for label in labels
    ):
        raise ValueError(f"Invalid domain value: {value!r}")
    return ascii_domain


def canonical_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid HTTP(S) URL: {value!r}")
    try:
        host = ipaddress.ip_address(parsed.hostname).compressed
        if ":" in host:
            host = f"[{host}]"
    except ValueError:
        host = canonical_domain(parsed.hostname)
    port = parsed.port
    default_port = 80 if parsed.scheme.lower() == "http" else 443
    netloc = host if port in {None, default_port} else f"{host}:{port}"
    path = posixpath.normpath(parsed.path or "/")
    if parsed.path.endswith("/") and not path.endswith("/"):
        path += "/"
    normalized = SplitResult(
        parsed.scheme.lower(),
        netloc,
        path,
        parsed.query,
        "",
    )
    return urlunsplit(normalized)


def canonical_path(value: str) -> str:
    text = str(value).strip().strip('"\'')
    if not text:
        raise ValueError("Path cannot be empty")
    if "\\" in text or re.match(r"^[A-Za-z]:", text):
        return str(PureWindowsPath(text)).replace("/", "\\").lower()
    return posixpath.normpath(text).lower()


def canonical_hash(value: str, node_type: str) -> str:
    text = str(value).strip().lower()
    expected_lengths = {"md5": 32, "sha1": 40, "sha256": 64}
    expected = expected_lengths.get(node_type)
    if expected is not None and (
        len(text) != expected or re.fullmatch(r"[0-9a-f]+", text) is None
    ):
        raise ValueError(f"Invalid {node_type} value")
    return text


def canonical_value(node_type: str, value: object) -> str:
    kind = canonical_node_type(node_type)
    text = str(value).strip()
    if not text:
        raise ValueError(f"{kind} value cannot be empty")
    if kind == "ip_address":
        return ipaddress.ip_address(text.strip("[]")).compressed
    if kind == "network":
        return ipaddress.ip_network(text, strict=False).with_prefixlen
    if kind == "domain":
        return canonical_domain(text)
    if kind == "url":
        return canonical_url(text)
    if kind in HASH_TYPES:
        return canonical_hash(text, kind)
    if kind == "port":
        port = int(text)
        if not 0 <= port <= 65535:
            raise ValueError(f"Invalid port: {value!r}")
        return str(port)
    if kind in {"process_path", "file_path"}:
        return canonical_path(text)
    return re.sub(r"\s+", "_", text.lower())


def deterministic_node_id(node_type: str, value: object) -> str:
    kind = canonical_node_type(node_type)
    canonical = canonical_value(kind, value)
    digest = hashlib.sha256(f"{kind}|{canonical}".encode("utf-8")).hexdigest()
    return f"{kind}--{digest}"


def deterministic_edge_id(source: str, relation: str, target: str) -> str:
    normalized_relation = str(relation).strip().lower().replace(" ", "_")
    digest = hashlib.sha256(
        f"{source}|{normalized_relation}|{target}".encode("utf-8")
    ).hexdigest()
    return f"edge--{digest}"


def basename(value: str) -> str:
    return re.split(r"[/\\]", value.rstrip("/\\"))[-1].lower()
