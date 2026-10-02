from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from prevention_relevance.ontology import (
    MATCHABLE_NODE_TYPES,
    canonical_node_type,
    canonical_value,
)


DEFAULT_FIELD_MAPPINGS: dict[str, tuple[str, ...]] = {
    "host": ("host", "hostname", "computer_name", "Computer", "device_id"),
    "platform": ("platform", "os", "os_name", "host_os", "EventData.HostOSName"),
    "process": ("process_name", "process", "image_name"),
    "process_path": ("process_exe", "process_path", "executable_path", "image_path"),
    "parent_process": ("parent_process", "parent_image", "process_parent_name"),
    "file_path": ("file_path", "path", "target_file", "object_name"),
    "identity": ("user", "username", "user_uid", "account_name", "principal"),
    "service": ("service", "service_name", "application", "app_protocol"),
    "source_ip": ("src_ip", "source_ip", "local_ip", "local_address"),
    "destination_ip": ("dst_ip", "destination_ip", "remote_ip", "remote_address"),
    "domain": ("domain", "destination_domain", "dns_query", "http_host", "tls_sni", "sni"),
    "url": ("url", "uri", "http_uri", "request_url", "download_url"),
    "protocol": ("protocol_name", "protocol", "proto", "transport_protocol"),
    "source_port": ("src_port", "source_port", "local_port"),
    "destination_port": ("dst_port", "destination_port", "remote_port"),
    "direction": ("network_direction", "direction", "traffic_direction"),
    "md5": ("md5",),
    "sha1": ("sha1", "sha_1"),
    "sha256": ("sha256", "sha_256", "file_hash"),
}

DEFAULT_TYPE_WEIGHTS = {
    "platform": 1.5,
    "host": 0.5,
    "process": 2.0,
    "process_path": 2.5,
    "file_path": 2.5,
    "service": 1.5,
    "protocol": 0.4,
    "port": 0.7,
    "ip_address": 2.5,
    "network": 1.5,
    "domain": 3.0,
    "url": 3.5,
    "md5": 4.0,
    "sha1": 4.0,
    "sha256": 4.0,
    "email": 2.5,
    "vulnerability": 3.0,
    "identity": 1.0,
    "feature": 1.0,
}

DEFAULT_COMPONENT_WEIGHTS = {
    "direct": 0.45,
    "node": 0.15,
    "edge": 0.25,
    "context": 0.15,
}

DEFAULT_MATCH_SCORES = {
    "exact": 1.0,
    "network_membership": 0.9,
    "process_basename": 0.7,
    "url_host": 0.65,
}

DEFAULT_GENERIC_VALUES = {
    "protocol": ("tcp", "udp"),
    "port": ("53", "80", "443"),
    "process": ("sh", "bash", "cmd.exe"),
}

DEFAULT_RELATION_ALIASES = {
    "runs": ("executes", "uses_process"),
    "connects_to": ("communicates_with", "contacts"),
    "uses_port": ("destination_port", "listens_on"),
    "uses_protocol": ("protocol",),
    "accesses": ("reads", "writes"),
}


@dataclass(frozen=True)
class RelevanceConfig:
    version: int
    component_weights: Mapping[str, float]
    type_weights: Mapping[str, float]
    match_scores: Mapping[str, float]
    high_threshold: float
    review_threshold: float
    incompatibility_weight: float
    ego_hops: int
    min_distinctive_matches: int
    field_mappings: Mapping[str, tuple[str, ...]]
    generic_values: Mapping[str, tuple[str, ...]]
    relation_aliases: Mapping[str, tuple[str, ...]]
    attack_applicability: Mapping[str, Mapping[str, tuple[str, ...]]]
    require_validated: bool
    include_feature_tokens: bool


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load relevance configuration {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Relevance configuration must be an object")
    return payload


def _string_tuple_mapping(
    value: object,
    *,
    path: str,
    defaults: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, tuple[str, ...]]:
    if value is None:
        return dict(defaults or {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    result = dict(defaults or {})
    for raw_key, raw_values in value.items():
        key = str(raw_key).strip()
        if not key or not isinstance(raw_values, list):
            raise ValueError(f"{path}.{raw_key} must be a list")
        values = tuple(str(item).strip() for item in raw_values if str(item).strip())
        if not values:
            raise ValueError(f"{path}.{raw_key} cannot be empty")
        result[key] = values
    return result


def _float_mapping(
    value: object,
    *,
    path: str,
    defaults: Mapping[str, float],
    required_keys: set[str] | None = None,
) -> dict[str, float]:
    if value is None:
        return dict(defaults)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    result = dict(defaults)
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}.{key} must be numeric") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{path}.{key} must be finite and non-negative")
        result[key] = number
    if required_keys and set(result) != required_keys:
        unknown = set(result) - required_keys
        missing = required_keys - set(result)
        raise ValueError(
            f"{path} has invalid keys; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    return result


def _bool(value: object, *, path: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def load_config(path: Path | None = None) -> RelevanceConfig:
    payload: Mapping[str, Any] = {} if path is None else _load_payload(path)
    allowed_top = {
        "version",
        "similarity",
        "field_mappings",
        "generic_values",
        "relation_aliases",
        "attack_applicability",
        "require_validated",
        "include_feature_tokens",
    }
    unknown = set(payload) - allowed_top
    if unknown:
        raise ValueError(f"Unknown relevance configuration fields: {sorted(unknown)}")
    version = int(payload.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported relevance configuration version: {version}")

    similarity = payload.get("similarity", {})
    if not isinstance(similarity, Mapping):
        raise ValueError("similarity must be an object")
    allowed_similarity = {
        "component_weights",
        "type_weights",
        "match_scores",
        "high_threshold",
        "review_threshold",
        "incompatibility_weight",
        "ego_hops",
        "min_distinctive_matches",
    }
    unknown_similarity = set(similarity) - allowed_similarity
    if unknown_similarity:
        raise ValueError(f"Unknown similarity fields: {sorted(unknown_similarity)}")

    component_weights = _float_mapping(
        similarity.get("component_weights"),
        path="similarity.component_weights",
        defaults=DEFAULT_COMPONENT_WEIGHTS,
        required_keys={"direct", "node", "edge", "context"},
    )
    if sum(component_weights.values()) <= 0:
        raise ValueError("similarity.component_weights must have a positive total")
    total = sum(component_weights.values())
    component_weights = {key: value / total for key, value in component_weights.items()}
    raw_type_weights = _float_mapping(
        similarity.get("type_weights"),
        path="similarity.type_weights",
        defaults=DEFAULT_TYPE_WEIGHTS,
    )
    type_weights: dict[str, float] = {}
    for raw_type, weight in raw_type_weights.items():
        node_type = canonical_node_type(raw_type)
        if node_type in type_weights and type_weights[node_type] != weight:
            raise ValueError(
                f"similarity.type_weights defines conflicting aliases for {node_type}"
            )
        type_weights[node_type] = weight
    unknown_types = set(type_weights) - MATCHABLE_NODE_TYPES
    if unknown_types:
        raise ValueError(f"Unknown similarity.type_weights node types: {sorted(unknown_types)}")
    match_scores = _float_mapping(
        similarity.get("match_scores"),
        path="similarity.match_scores",
        defaults=DEFAULT_MATCH_SCORES,
        required_keys=set(DEFAULT_MATCH_SCORES),
    )
    if any(value > 1 for value in match_scores.values()):
        raise ValueError("similarity.match_scores values must be in [0, 1]")

    high_threshold = float(similarity.get("high_threshold", 0.4))
    review_threshold = float(similarity.get("review_threshold", 0.2))
    if not 0 <= review_threshold <= high_threshold <= 1:
        raise ValueError(
            "similarity thresholds must satisfy 0 <= review_threshold <= high_threshold <= 1"
        )
    incompatibility_weight = float(similarity.get("incompatibility_weight", 0.25))
    if not 0 <= incompatibility_weight <= 1:
        raise ValueError("similarity.incompatibility_weight must be in [0, 1]")
    ego_hops = int(similarity.get("ego_hops", 2))
    min_distinctive_matches = int(similarity.get("min_distinctive_matches", 1))
    if ego_hops < 1:
        raise ValueError("similarity.ego_hops must be at least 1")
    if min_distinctive_matches < 1:
        raise ValueError("similarity.min_distinctive_matches must be at least 1")

    field_mappings = _string_tuple_mapping(
        payload.get("field_mappings"),
        path="field_mappings",
        defaults=DEFAULT_FIELD_MAPPINGS,
    )
    raw_generic_values = _string_tuple_mapping(
        payload.get("generic_values"),
        path="generic_values",
        defaults=DEFAULT_GENERIC_VALUES,
    )
    generic_values: dict[str, tuple[str, ...]] = {}
    for raw_type, values in raw_generic_values.items():
        node_type = canonical_node_type(raw_type)
        if node_type not in MATCHABLE_NODE_TYPES:
            raise ValueError(f"generic_values.{raw_type} has unsupported node type")
        try:
            generic_values[node_type] = tuple(
                canonical_value(node_type, value) for value in values
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"generic_values.{raw_type} contains an invalid value") from exc
    raw_relation_aliases = _string_tuple_mapping(
        payload.get("relation_aliases"),
        path="relation_aliases",
        defaults=DEFAULT_RELATION_ALIASES,
    )
    relation_aliases = {
        str(relation).strip().lower().replace(" ", "_"): tuple(
            str(alias).strip().lower().replace(" ", "_") for alias in aliases
        )
        for relation, aliases in raw_relation_aliases.items()
    }

    raw_applicability = payload.get("attack_applicability", {})
    if not isinstance(raw_applicability, Mapping):
        raise ValueError("attack_applicability must be an object")
    attack_applicability: dict[str, dict[str, tuple[str, ...]]] = {}
    for raw_label, raw_requirements in raw_applicability.items():
        label = str(raw_label).strip().lower()
        if not label or not isinstance(raw_requirements, Mapping):
            raise ValueError(f"attack_applicability.{raw_label} must be an object")
        requirements: dict[str, tuple[str, ...]] = {}
        for raw_type, raw_values in raw_requirements.items():
            node_type = canonical_node_type(str(raw_type))
            if node_type not in MATCHABLE_NODE_TYPES:
                raise ValueError(
                    f"attack_applicability.{label}.{raw_type} has unsupported node type"
                )
            if not isinstance(raw_values, list) or not raw_values:
                raise ValueError(
                    f"attack_applicability.{label}.{raw_type} must be a nonempty list"
                )
            requirements[node_type] = tuple(str(value) for value in raw_values)
        attack_applicability[label] = requirements

    return RelevanceConfig(
        version=version,
        component_weights=component_weights,
        type_weights=type_weights,
        match_scores=match_scores,
        high_threshold=high_threshold,
        review_threshold=review_threshold,
        incompatibility_weight=incompatibility_weight,
        ego_hops=ego_hops,
        min_distinctive_matches=min_distinctive_matches,
        field_mappings=field_mappings,
        generic_values=generic_values,
        relation_aliases=relation_aliases,
        attack_applicability=attack_applicability,
        require_validated=_bool(
            payload.get("require_validated"),
            path="require_validated",
            default=False,
        ),
        include_feature_tokens=_bool(
            payload.get("include_feature_tokens"),
            path="include_feature_tokens",
            default=True,
        ),
    )
