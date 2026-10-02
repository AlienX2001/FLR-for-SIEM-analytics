from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from prevention_relevance.config import RelevanceConfig
from prevention_relevance.graph import TypedGraph

LOGGER = logging.getLogger(__name__)

PATTERN_RE = re.compile(
    r"^\[\s*([a-zA-Z0-9-]+):([^=]+?)\s*=\s*'((?:\\.|[^'])*)'\s*\]$"
)
SUBCATEGORIES = {"system", "network", "identity", "llm", "cloud", "cross"}


def _unescape_stix_string(value: str) -> str:
    return value.replace("\\'", "'").replace("\\\\", "\\")


def parse_indicator_pattern(pattern: str) -> tuple[str, str] | None:
    """Parse the exact-equality STIX patterns emitted or accepted by this project."""
    match = PATTERN_RE.fullmatch(str(pattern).strip())
    if match is None:
        return None
    object_type = match.group(1).lower()
    property_path = match.group(2).strip().lower().replace('"', "").replace("'", "")
    value = _unescape_stix_string(match.group(3))

    if object_type in {"ipv4-addr", "ipv6-addr"} and property_path == "value":
        return "ip_address", value
    if object_type == "domain-name" and property_path == "value":
        return "domain", value
    if object_type == "url" and property_path == "value":
        return "url", value
    if object_type == "email-addr" and property_path == "value":
        return "email", value
    if object_type == "vulnerability" and property_path in {"name", "value"}:
        return "vulnerability", value
    if object_type in {"file", "artifact"} and "hashes." in property_path:
        algorithm = property_path.rsplit(".", maxsplit=1)[-1].replace("-", "")
        if algorithm in {"md5", "sha1", "sha256", "imphash", "pehash"}:
            return algorithm, value
    return None


def _iter_bundle_objects(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Incrementally decode objects from a top-level STIX bundle."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        objects_start: int | None = None
        eof = False
        while objects_start is None:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"STIX bundle {path} has no objects array")
            buffer += chunk
            key_index = buffer.find('"objects"')
            if key_index >= 0:
                objects_start = buffer.find("[", key_index)
            if objects_start is None and len(buffer) > chunk_size * 2:
                buffer = buffer[-chunk_size:]
        buffer = buffer[objects_start + 1 :]

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Malformed STIX bundle object in {path}")
                    chunk = handle.read(chunk_size)
                    if chunk:
                        buffer += chunk
                    else:
                        eof = True
            if not isinstance(value, dict):
                raise ValueError(f"STIX bundle {path} contains a non-object entry")
            yield value
            buffer = buffer[end:]


def iter_stix_objects(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid STIX JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"STIX JSONL entry {path}:{line_number} is not an object")
                yield value
        return

    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(65536)
    if re.search(r'"type"\s*:\s*"bundle"', prefix):
        yield from _iter_bundle_objects(path)
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read STIX input {path}: {exc}") from exc
    values = payload if isinstance(payload, list) else [payload]
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"STIX input {path} contains a non-object entry")
        yield value


def _validation_status(obj: Mapping[str, Any], assume_validated: bool) -> str:
    if assume_validated:
        return "validated"
    raw = obj.get("x_validation_status", obj.get("x_validated"))
    if raw is True or str(raw).strip().lower() in {"validated", "true", "approved"}:
        return "validated"
    if raw is False or str(raw).strip().lower() in {"rejected", "false"}:
        return "rejected"
    return "unreviewed"


def _indicator_node(graph: TypedGraph, obj: Mapping[str, Any], assume_validated: bool) -> str:
    stix_id = str(obj.get("id", "")).strip()
    if not stix_id:
        raise ValueError("STIX Indicator is missing id")
    risk = obj.get("x_federated_lr_risk_probability")
    attributes: dict[str, Any] = {
        "name": str(obj.get("name", stix_id)),
        "pattern": str(obj.get("pattern", "")),
        "validation_status": _validation_status(obj, assume_validated),
        "labels": [str(item) for item in obj.get("labels", [])],
        "stix_type": "indicator",
    }
    if risk is not None:
        attributes["risk_probability"] = float(risk)
    return graph.add_node("indicator", stix_id, attributes=attributes, node_id=stix_id)


def _add_attack_context(
    graph: TypedGraph,
    indicator_id: str,
    label: object,
    config: RelevanceConfig,
) -> None:
    normalized_label = str(label).strip().lower()
    if not normalized_label:
        return
    attack = graph.add_node("attack_class", normalized_label)
    graph.add_edge(indicator_id, "associated_with", attack)
    for node_type, values in config.attack_applicability.get(normalized_label, {}).items():
        for value in values:
            try:
                requirement = graph.add_node(
                    node_type,
                    value,
                    attributes={"applicability_requirement": True},
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid attack applicability value {normalized_label}.{node_type}: {value!r}"
                ) from exc
            graph.add_edge(attack, "applicable_to", requirement)


def _add_stix_object(
    graph: TypedGraph,
    obj: Mapping[str, Any],
    *,
    config: RelevanceConfig,
    assume_validated: bool,
) -> tuple[str | None, tuple[str, str, str] | None]:
    object_type = str(obj.get("type", "")).lower()
    stix_id = str(obj.get("id", "")).strip()
    if object_type == "indicator":
        indicator = _indicator_node(graph, obj, assume_validated)
        parsed = parse_indicator_pattern(str(obj.get("pattern", "")))
        if parsed is not None:
            node_type, value = parsed
            try:
                observable = graph.add_node(node_type, value)
            except (TypeError, ValueError):
                observable = None
            if observable:
                graph.add_edge(indicator, "indicates", observable)
        label = obj.get("x_federated_lr_predicted_label")
        if label is not None:
            _add_attack_context(graph, indicator, label, config)
        return indicator, None
    if object_type == "relationship":
        source = str(obj.get("source_ref", ""))
        target = str(obj.get("target_ref", ""))
        relation = str(obj.get("relationship_type", "related_to"))
        return None, (source, relation, target)
    if not stix_id:
        return None, None

    name = str(obj.get("name", obj.get("value", stix_id)))
    node_type = {
        "attack-pattern": "attack_technique",
        "identity": "identity",
        "malware": "process",
        "tool": "process",
        "software": "process",
        "infrastructure": "service",
        "vulnerability": "vulnerability",
    }.get(object_type, "stix_object")
    try:
        node = graph.add_node(
            node_type,
            name,
            attributes={"stix_type": object_type, "stix_id": stix_id},
            node_id=stix_id,
        )
    except (TypeError, ValueError):
        return None, None
    return node, None


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL entry {path}:{line_number} is not an object")
            yield record


def _walk_evidence(value: object) -> tuple[set[str], set[str]]:
    categories: set[str] = set()
    tokens: set[str] = set()

    def visit(item: object, parent_key: str | None = None) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized_key = str(key).strip().lower().replace("inter_category", "cross")
                if normalized_key in SUBCATEGORIES:
                    categories.add(normalized_key)
                if normalized_key == "token" and isinstance(nested, str) and nested.strip():
                    tokens.add(nested.strip().lower())
                visit(nested, normalized_key)
        elif isinstance(item, list):
            for nested in item:
                visit(nested, parent_key)

    visit(value)
    return categories, tokens


def _enrich_from_ioc_records(
    graph: TypedGraph,
    path: Path,
    *,
    config: RelevanceConfig,
) -> int:
    count = 0
    for record in _iter_jsonl(path):
        indicator_id = str(record.get("indicator_id", "")).strip()
        if not indicator_id or indicator_id not in graph.nodes:
            continue
        count += 1
        risk = record.get("max_risk_probability")
        attributes = {"occurrence_count": 1}
        if risk is not None:
            attributes["risk_probability"] = float(risk)
        indicator = graph.nodes[indicator_id]
        graph.add_node(
            "indicator",
            indicator.value,
            attributes=attributes,
            node_id=indicator_id,
        )
        label = record.get("predicted_label")
        if label is not None:
            _add_attack_context(graph, indicator_id, label, config)
        internal_id = str(record.get("internal_log_id", "")).strip()
        event = None
        if internal_id:
            event = graph.add_node("event", internal_id)
            graph.add_edge(indicator_id, "derived_from", event)
        raw_evidence = record.get("evidence_by_label_subcategory")
        if isinstance(raw_evidence, str):
            try:
                raw_evidence = json.loads(raw_evidence)
            except json.JSONDecodeError:
                raw_evidence = None
        categories, tokens = _walk_evidence(raw_evidence)
        for category in categories:
            evidence = graph.add_node("evidence_category", category)
            graph.add_edge(event or indicator_id, "supported_by", evidence)
        if config.include_feature_tokens:
            for token in tokens:
                feature = graph.add_node("feature", token)
                graph.add_edge(event or indicator_id, "supported_by", feature)
    return count


def _enrich_from_explanations(
    graph: TypedGraph,
    path: Path,
    *,
    config: RelevanceConfig,
) -> int:
    if not config.include_feature_tokens:
        return 0
    count = 0
    event_nodes = {
        node.value: node.node_id
        for node in graph.nodes.values()
        if node.node_type == "event"
    }
    for record in _iter_jsonl(path):
        internal_id = str(record.get("internal_log_id", "")).strip()
        event = event_nodes.get(internal_id)
        if event is None:
            continue
        raw_evidence = record.get(
            "top_contributions", record.get("top_contributing_features")
        )
        categories, tokens = _walk_evidence(raw_evidence)
        for category in categories:
            evidence = graph.add_node("evidence_category", category)
            graph.add_edge(event, "supported_by", evidence)
        for token in tokens:
            feature = graph.add_node("feature", token)
            graph.add_edge(event, "supported_by", feature)
        count += 1
    return count


def build_historical_ioc_graph(
    stix_path: Path,
    *,
    config: RelevanceConfig,
    ioc_records_path: Path | None = None,
    explanations_path: Path | None = None,
    assume_validated: bool = False,
) -> TypedGraph:
    graph = TypedGraph(graph_id="historical_ioc_knowledge", graph_type="ioc_knowledge")
    pending_relationships: list[tuple[str, str, str]] = []
    object_count = 0
    indicator_count = 0
    for obj in iter_stix_objects(stix_path):
        object_count += 1
        node_id, relationship = _add_stix_object(
            graph,
            obj,
            config=config,
            assume_validated=assume_validated,
        )
        indicator_count += int(node_id is not None and str(obj.get("type", "")).lower() == "indicator")
        if relationship is not None:
            pending_relationships.append(relationship)

    relationship_count = 0
    for source, relation, target in pending_relationships:
        if source in graph.nodes and target in graph.nodes:
            graph.add_edge(source, relation, target)
            relationship_count += 1
    record_count = (
        _enrich_from_ioc_records(graph, ioc_records_path, config=config)
        if ioc_records_path is not None
        else 0
    )
    explanation_count = (
        _enrich_from_explanations(graph, explanations_path, config=config)
        if explanations_path is not None
        else 0
    )
    graph.update_metadata(
        {
            "source_stix": str(stix_path),
            "stix_object_count": object_count,
            "indicator_count": indicator_count,
            "stix_relationship_count": relationship_count,
            "ioc_record_count": record_count,
            "explanation_record_count": explanation_count,
            "assume_validated": assume_validated,
        }
    )
    LOGGER.info(
        "Built historical IoC graph: indicators=%s nodes=%s edges=%s",
        indicator_count,
        len(graph.nodes),
        len(graph.edges),
    )
    return graph

