#!/usr/bin/env python3
"""Compare IoCs extracted from two STIX 2.x JSON or JSONL files.

Usage example:
  python compare_ioc_stix.py --test-file test_iocs.jsonl --groundtruth-file groundtruth_iocs.json
"""

from __future__ import annotations

import argparse
import ast
import ipaddress
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple
from urllib.parse import urlparse


IndicatorMap = Dict[str, Set[str]]

OUTPUT_FILE = Path("ioc_comparison_results.json")

# This is intentionally a practical subset of the STIX pattern grammar. It
# extracts useful literals from common IoC patterns without making a full STIX
# pattern parser a runtime dependency.
OBJECT_PATH_RE = r"[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9_.\-'\"\[\]*]+"
VALUE_RE = r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|[^\]\s(),]+"

IN_RE = re.compile(
    rf"(?P<path>{OBJECT_PATH_RE})\s+IN\s*\((?P<values>.*?)\)",
    re.IGNORECASE | re.DOTALL,
)
COMPARISON_RE = re.compile(
    rf"(?P<path>{OBJECT_PATH_RE})\s*"
    rf"(?P<operator>=|LIKE|MATCHES|ISSUBSET|ISSUPERSET)\s*"
    rf"(?P<value>{VALUE_RE})",
    re.IGNORECASE | re.DOTALL,
)
LIST_VALUE_RE = re.compile(VALUE_RE, re.DOTALL)


def warn(message: str) -> None:
    """Print a warning without interrupting the comparison."""
    print(f"[WARN] {message}", file=sys.stderr)


def load_stix_file(path: str | Path) -> List[Dict[str, Any]]:
    """Load a STIX .json or .jsonl file and return a flat list of STIX objects."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return list(iter_stix_objects(data))

    if suffix == ".jsonl":
        objects: List[Dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    warn(f"Skipping malformed JSON in {file_path} line {line_number}: {exc}")
                    continue
                objects.extend(iter_stix_objects(data))
        return objects

    raise ValueError(f"Unsupported STIX file extension for {file_path}; expected .json or .jsonl")


def iter_stix_objects(stix_data: Any) -> Iterable[Dict[str, Any]]:
    """Yield STIX objects from a bundle, object list, or single object."""
    if isinstance(stix_data, dict):
        objects = stix_data.get("objects")
        if isinstance(objects, list):
            for obj in objects:
                if isinstance(obj, dict):
                    yield obj
                else:
                    warn("Skipping non-object entry in STIX bundle objects list")
            return

        yield stix_data
        return

    if isinstance(stix_data, list):
        for obj in stix_data:
            if isinstance(obj, dict):
                yield obj
            else:
                warn("Skipping non-object entry in top-level STIX object list")
        return

    warn(f"Unsupported STIX JSON root type: {type(stix_data).__name__}")


def extract_indicators(stix_data: Any) -> IndicatorMap:
    """Extract normalized IoCs from STIX indicator objects."""
    indicators: DefaultDict[str, Set[str]] = defaultdict(set)

    for obj in iter_stix_objects(stix_data):
        if obj.get("type") != "indicator":
            continue

        pattern = obj.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            indicator_id = obj.get("id", "<unknown indicator>")
            warn(f"Skipping indicator {indicator_id}: missing or invalid pattern")
            continue

        try:
            parsed = parse_indicator_pattern(pattern)
        except Exception as exc:  # Defensive: malformed patterns should not stop comparison.
            indicator_id = obj.get("id", "<unknown indicator>")
            warn(f"Skipping malformed pattern in {indicator_id}: {exc}")
            continue

        if not parsed:
            indicator_id = obj.get("id", "<unknown indicator>")
            warn(f"Skipping unsupported pattern in {indicator_id}: {pattern}")
            continue

        for indicator_type, value in parsed:
            normalized = normalize_indicator(value, indicator_type)
            if normalized:
                indicators[indicator_type].add(normalized)

    return dict(indicators)


def parse_indicator_pattern(pattern: str) -> List[Tuple[str, str]]:
    """Extract (indicator_type, value) pairs from common STIX pattern clauses."""
    results: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for match in IN_RE.finditer(pattern):
        path = match.group("path")
        for raw_value in split_stix_value_list(match.group("values")):
            add_pattern_value(results, seen, path, raw_value)

    for match in COMPARISON_RE.finditer(pattern):
        path = match.group("path")
        raw_value = match.group("value")
        add_pattern_value(results, seen, path, raw_value)

    return results


def add_pattern_value(
    results: List[Tuple[str, str]],
    seen: Set[Tuple[str, str]],
    stix_path: str,
    raw_value: str,
) -> None:
    """Classify and collect one STIX path/value pair."""
    value = parse_stix_literal(raw_value)
    indicator_type = indicator_type_from_stix_path(stix_path, value)
    for item in expand_indicator_value(indicator_type, value):
        if item[1] and item not in seen:
            seen.add(item)
            results.append(item)


def expand_indicator_value(indicator_type: str, value: str) -> List[Tuple[str, str]]:
    """Add comparable host indicators for URL-shaped IoCs."""
    if indicator_type == "url":
        host_item = host_indicator_from_url(value)
        if not host_item:
            return [(indicator_type, value)]
        if is_host_only_url(value):
            return [host_item]
        return [(indicator_type, value), host_item]

    if indicator_type == "domain" and looks_like_url(value):
        host_item = host_indicator_from_url(value)
        if host_item:
            return [host_item]

    return [(indicator_type, value)]


def split_stix_value_list(raw_values: str) -> List[str]:
    """Split values from a STIX IN (...) expression."""
    return [m.group(0).strip() for m in LIST_VALUE_RE.finditer(raw_values) if m.group(0).strip()]


def parse_stix_literal(raw_value: str) -> str:
    """Turn a STIX quoted or unquoted literal into a Python string."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
            return str(parsed).strip()
        except (SyntaxError, ValueError):
            return value[1:-1].replace(r"\'", "'").replace(r"\"", '"').replace(r"\\", "\\").strip()
    return value.strip()


def indicator_type_from_stix_path(stix_path: str, value: str) -> str:
    """Map a STIX cyber observable path to a concise IoC type name."""
    object_type, _, property_path = stix_path.partition(":")
    object_type = object_type.lower().strip()
    parts = split_stix_path_parts(property_path)
    lower_parts = [p.lower() for p in parts]

    if object_type == "ipv4-addr" and "value" in lower_parts:
        return "ipv4"
    if object_type == "ipv6-addr" and "value" in lower_parts:
        return "ipv6"
    if object_type == "domain-name" and "value" in lower_parts:
        return "domain"
    if object_type == "url" and "value" in lower_parts:
        return "url"
    if object_type == "email-addr" and "value" in lower_parts:
        return "email"
    if object_type == "mac-addr" and "value" in lower_parts:
        return "mac"

    hash_type = hash_type_from_path(object_type, lower_parts)
    if hash_type:
        return hash_type

    if object_type == "file" and "name" in lower_parts:
        return "file_name"
    if object_type == "directory" and "path" in lower_parts:
        return "directory_path"
    if object_type == "autonomous-system" and "number" in lower_parts:
        return "autonomous_system"
    if object_type == "windows-registry-key" and "key" in lower_parts:
        return "windows_registry_key"
    if object_type == "mutex" and "name" in lower_parts:
        return "mutex"
    if object_type == "user-account" and any(p in lower_parts for p in ("account_login", "user_id")):
        return "user_account"
    if object_type == "x509-certificate" and "serial_number" in lower_parts:
        return "x509_serial_number"

    inferred_type = infer_indicator_type(value)
    if inferred_type:
        return inferred_type

    if lower_parts:
        return sanitize_indicator_type(f"{object_type}_{lower_parts[-1]}")
    return sanitize_indicator_type(object_type)


def split_stix_path_parts(property_path: str) -> List[str]:
    """Split a STIX property path while respecting quoted segments."""
    cleaned = re.sub(r"\[(?:\*|\d+)\]", "", property_path.strip())
    parts: List[str] = []
    current: List[str] = []
    quote: str | None = None
    escape = False

    for char in cleaned:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\":
            current.append(char)
            escape = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            current.append(char)
            quote = char
            continue
        if char == ".":
            append_path_part(parts, "".join(current))
            current = []
            continue
        current.append(char)

    append_path_part(parts, "".join(current))
    return parts


def append_path_part(parts: List[str], value: str) -> None:
    """Normalize one STIX path segment before appending it."""
    part = value.strip()
    if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
        part = parse_stix_literal(part)
    if part:
        parts.append(part)


def hash_type_from_path(object_type: str, lower_parts: Sequence[str]) -> str | None:
    """Return a hash indicator type when a STIX path points at hashes."""
    if "hashes" not in lower_parts:
        return None

    hash_index = lower_parts.index("hashes")
    if hash_index + 1 >= len(lower_parts):
        return "hash"

    algorithm = lower_parts[hash_index + 1]
    normalized = re.sub(r"[^a-z0-9]", "", algorithm.lower())
    known_hashes = {
        "md5": "md5",
        "sha1": "sha1",
        "sha256": "sha256",
        "sha512": "sha512",
        "sha384": "sha384",
        "sha224": "sha224",
        "ssdeep": "ssdeep",
        "tlsh": "tlsh",
        "imphash": "imphash",
    }

    if object_type in {"file", "artifact", "x509-certificate"}:
        return known_hashes.get(normalized, normalized or "hash")
    return known_hashes.get(normalized)


def infer_indicator_type(value: str) -> str | None:
    """Infer a common IoC type from a value when the STIX path is generic."""
    stripped = value.strip()
    lower = stripped.lower()

    try:
        ip_obj = ipaddress.ip_address(stripped)
        return "ipv4" if ip_obj.version == 4 else "ipv6"
    except ValueError:
        pass

    if "@" in stripped and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", stripped):
        return "email"
    if lower.startswith(("http://", "https://")):
        return "url"
    if re.fullmatch(r"[a-fA-F0-9]{32}", stripped):
        return "md5"
    if re.fullmatch(r"[a-fA-F0-9]{40}", stripped):
        return "sha1"
    if re.fullmatch(r"[a-fA-F0-9]{64}", stripped):
        return "sha256"
    return None


def sanitize_indicator_type(value: str) -> str:
    """Convert an arbitrary STIX type/path into a JSON-key-friendly name."""
    sanitized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return sanitized or "unknown"


def normalize_indicator(value: str, indicator_type: str) -> str:
    """Normalize an IoC value before set comparison."""
    normalized = value.strip()
    if not normalized:
        return ""

    if indicator_type == "domain":
        return normalize_domain(normalized)

    if indicator_type in {
        "url",
        "email",
        "mac",
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "ssdeep",
        "tlsh",
        "imphash",
    }:
        normalized = normalized.lower()

    if indicator_type in {"ipv4", "ipv6"}:
        try:
            if "/" in normalized:
                return str(ipaddress.ip_network(normalized, strict=False))
            return str(ipaddress.ip_address(normalized))
        except ValueError:
            return normalized

    return normalized


def looks_like_url(value: str) -> bool:
    """Return True when a value appears to include a URL scheme."""
    return re.match(r"^[a-z][a-z0-9+.-]*://", value.strip(), re.IGNORECASE) is not None


def is_host_only_url(value: str) -> bool:
    """Return True for URLs that only carry scheme and host, not a path IoC."""
    if not looks_like_url(value):
        return False

    parsed = urlparse(value.strip())
    return (
        bool(parsed.hostname)
        and parsed.path in {"", "/"}
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )


def host_indicator_from_url(value: str) -> Tuple[str, str] | None:
    """Extract a domain/IP indicator from a URL value."""
    parsed = urlparse(value.strip())
    host = parsed.hostname
    if not host:
        return None

    try:
        ip_obj = ipaddress.ip_address(host)
        return ("ipv4" if ip_obj.version == 4 else "ipv6", str(ip_obj))
    except ValueError:
        return "domain", host


def normalize_domain(value: str) -> str:
    """Normalize domains and URL-shaped domain values for comparison."""
    stripped = value.strip().lower()
    if not stripped:
        return ""

    if looks_like_url(stripped):
        parsed = urlparse(stripped)
        if parsed.hostname:
            stripped = parsed.hostname

    stripped = stripped.rstrip(".")
    if stripped.startswith("www."):
        stripped = stripped[4:]
    return stripped


def compare_indicators(test_indicators: IndicatorMap, groundtruth_indicators: IndicatorMap) -> Dict[str, Any]:
    """Compare extracted IoC sets by indicator type."""
    results: Dict[str, Any] = {}
    all_types = sorted(set(test_indicators) | set(groundtruth_indicators))

    for indicator_type in all_types:
        test_values = test_indicators.get(indicator_type, set())
        groundtruth_values = groundtruth_indicators.get(indicator_type, set())
        intersection = test_values & groundtruth_values
        only_in_test = test_values - groundtruth_values
        only_in_groundtruth = groundtruth_values - test_values

        results[indicator_type] = {
            "test_count": len(test_values),
            "groundtruth_count": len(groundtruth_values),
            "intersection_count": len(intersection),
            "disjoint_count": len(only_in_test) + len(only_in_groundtruth),
            "only_in_test": sorted(only_in_test),
            "only_in_groundtruth": sorted(only_in_groundtruth),
        }

    return results


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print a compact count table followed by exact differences."""
    if not results:
        print("No indicators were extracted from either file.")
        return

    rows = [
        (
            indicator_type,
            data["test_count"],
            data["groundtruth_count"],
            data["intersection_count"],
            data["disjoint_count"],
        )
        for indicator_type, data in results.items()
    ]

    headers = ("type", "test", "groundtruth", "intersection", "disjoint")
    widths = [
        max(len(str(row[index])) for row in rows + [headers])
        for index in range(len(headers))
    ]

    print(format_row(headers, widths))
    print(format_row(tuple("-" * width for width in widths), widths))
    for row in rows:
        print(format_row(row, widths))

    print("\nDifferences:")
    for indicator_type, data in results.items():
        print(f"\n[{indicator_type}]")
        print(f"only_in_test: {json.dumps(data['only_in_test'], ensure_ascii=False)}")
        print(f"only_in_groundtruth: {json.dumps(data['only_in_groundtruth'], ensure_ascii=False)}")


def format_row(row: Sequence[Any], widths: Sequence[int]) -> str:
    """Format one console table row."""
    return " | ".join(str(value).ljust(width) for value, width in zip(row, widths))


def write_results(results: Dict[str, Any], output_path: Path = OUTPUT_FILE) -> None:
    """Write comparison results as JSON."""
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare IoCs from two STIX 2.x JSON or JSONL files.",
        epilog=(
            "Example: python compare_ioc_stix.py "
            "--test-file test_iocs.jsonl --groundtruth-file groundtruth_iocs.json"
        ),
    )
    parser.add_argument("--test-file", required=True, type=Path, help="Path to the test IoC STIX JSON or JSONL file")
    parser.add_argument(
        "--groundtruth-file",
        required=True,
        type=Path,
        help="Path to the ground-truth IoC STIX JSON or JSONL file",
    )
    return parser.parse_args()


def main() -> int:
    """Load, extract, compare, print, and write IoC comparison results."""
    args = parse_args()

    try:
        test_data = load_stix_file(args.test_file)
        groundtruth_data = load_stix_file(args.groundtruth_file)
    except OSError as exc:
        print(f"Error reading input file: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON input file: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error loading input file: {exc}", file=sys.stderr)
        return 1

    test_indicators = extract_indicators(test_data)
    groundtruth_indicators = extract_indicators(groundtruth_data)
    results = compare_indicators(test_indicators, groundtruth_indicators)

    print_summary_table(results)
    write_results(results)
    print(f"\nWrote JSON results to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
