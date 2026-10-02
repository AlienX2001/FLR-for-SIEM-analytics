from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from federated_lr_pipeline.specialized_models import _endpoint_ip_value

from prevention_relevance.config import RelevanceConfig
from prevention_relevance.graph import TypedGraph
from prevention_relevance.ontology import basename, canonical_domain
from prevention_relevance.preprocessing_adapter import preprocess_change_row

LOGGER = logging.getLogger(__name__)


def _safe_node(
    graph: TypedGraph,
    node_type: str,
    value: object | None,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> str | None:
    if value is None or not str(value).strip():
        return None
    try:
        return graph.add_node(
            node_type,
            value,
            attributes=attributes,
            increment_occurrence=True,
        )
    except (TypeError, ValueError):
        return None


def _ip_node(graph: TypedGraph, value: str | None) -> str | None:
    if value is None:
        return None
    address = _endpoint_ip_value(value)
    return _safe_node(graph, "ip_address", address.compressed if address else None)


def _domain_node(graph: TypedGraph, value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return _safe_node(graph, "domain", canonical_domain(value))
    except ValueError:
        return None


def _add_contains(graph: TypedGraph, environment: str, node_id: str | None) -> None:
    if node_id is not None:
        graph.add_edge(environment, "contains", node_id, increment_occurrence=True)


def _add_row_to_graph(
    graph: TypedGraph,
    environment: str,
    raw_row: Mapping[str, Any],
    *,
    config: RelevanceConfig,
    source_file: Path,
    category_hint: str | None,
) -> tuple[int, int]:
    processed = preprocess_change_row(raw_row, field_mappings=config.field_mappings)
    fields = processed.canonical_fields
    attributes = {"source_file": str(source_file)}
    if category_hint:
        attributes["category_hint"] = category_hint

    host = _safe_node(graph, "host", fields.get("host"), attributes=attributes)
    platform = _safe_node(graph, "platform", fields.get("platform"))
    process = _safe_node(graph, "process", fields.get("process"))
    process_path = _safe_node(graph, "process_path", fields.get("process_path"))
    parent_process = _safe_node(graph, "process", fields.get("parent_process"))
    file_path = _safe_node(graph, "file_path", fields.get("file_path"))
    identity = _safe_node(graph, "identity", fields.get("identity"))
    service = _safe_node(graph, "service", fields.get("service"))
    protocol = _safe_node(graph, "protocol", fields.get("protocol"))
    source_port = _safe_node(graph, "port", fields.get("source_port"))
    destination_port = _safe_node(graph, "port", fields.get("destination_port"))
    source_ip = _ip_node(graph, fields.get("source_ip"))
    destination_ip = _ip_node(graph, fields.get("destination_ip"))
    domain = _domain_node(graph, fields.get("domain"))
    url = _safe_node(graph, "url", fields.get("url"))
    if process is None and process_path is not None:
        process = _safe_node(graph, "process", basename(graph.nodes[process_path].value))

    for node_id in (
        host,
        platform,
        process,
        process_path,
        parent_process,
        file_path,
        identity,
        service,
        protocol,
        source_port,
        destination_port,
        source_ip,
        destination_ip,
        domain,
        url,
    ):
        _add_contains(graph, environment, node_id)

    if host and platform:
        graph.add_edge(host, "has_platform", platform, increment_occurrence=True)
    if host and process:
        graph.add_edge(host, "runs", process, increment_occurrence=True)
    if process and process_path:
        graph.add_edge(process, "executable_at", process_path, increment_occurrence=True)
    if process and parent_process:
        graph.add_edge(process, "launched_by", parent_process, increment_occurrence=True)
    if process and file_path:
        graph.add_edge(process, "accesses", file_path, increment_occurrence=True)
    if host and identity:
        graph.add_edge(host, "uses_identity", identity, increment_occurrence=True)

    actor = process or host or environment
    targets = tuple(dict.fromkeys(item for item in (url, domain, destination_ip) if item))
    for target in targets:
        graph.add_edge(actor, "connects_to", target, increment_occurrence=True)
        if source_ip:
            graph.add_edge(source_ip, "connects_to", target, increment_occurrence=True)
    for owner in (service, actor):
        if owner and protocol:
            graph.add_edge(owner, "uses_protocol", protocol, increment_occurrence=True)
        if owner and destination_port:
            graph.add_edge(owner, "uses_port", destination_port, increment_occurrence=True)
    if source_ip and source_port:
        graph.add_edge(source_ip, "uses_port", source_port, increment_occurrence=True)
    if url:
        try:
            url_host = urlsplit(graph.nodes[url].value).hostname
        except ValueError:
            url_host = None
        url_domain = _domain_node(graph, url_host) if url_host else None
        url_ip = _ip_node(graph, url_host) if url_host else None
        if url_domain:
            graph.add_edge(url, "has_host", url_domain, increment_occurrence=True)
        elif url_ip:
            graph.add_edge(url, "has_host", url_ip, increment_occurrence=True)

    for hash_type in ("md5", "sha1", "sha256"):
        hash_node = _safe_node(graph, hash_type, fields.get(hash_type))
        _add_contains(graph, environment, hash_node)
        if file_path and hash_node:
            graph.add_edge(file_path, "has_hash", hash_node, increment_occurrence=True)

    if config.include_feature_tokens:
        token_groups = {
            "system": processed.system_tokens,
            "network": processed.network_tokens,
            "identity": processed.identity_tokens,
            "llm": processed.llm_tokens,
            "cloud": processed.cloud_tokens,
        }
        for category, tokens in token_groups.items():
            for token in tokens:
                feature = _safe_node(
                    graph,
                    "feature",
                    token,
                    attributes={"category": category},
                )
                if feature:
                    graph.add_edge(
                        environment,
                        "has_feature",
                        feature,
                        increment_occurrence=True,
                    )

    return 1, len(processed.conflicts)


def build_proposed_change_graph(
    log_sources: Iterable[tuple[Path, str | None]],
    *,
    config: RelevanceConfig,
    environment_id: str,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> TypedGraph:
    graph = TypedGraph(graph_id=environment_id, graph_type="proposed_change")
    environment = graph.add_node(
        "environment",
        environment_id,
        attributes={"profile_type": "proposed_change"},
    )
    row_count = 0
    conflict_count = 0
    source_count = 0
    for source_file, category_hint in log_sources:
        source_count += 1
        try:
            handle = source_file.open("r", encoding=encoding, newline="")
        except OSError as exc:
            raise ValueError(f"Could not open proposed-change log {source_file}: {exc}") from exc
        with handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"Proposed-change CSV {source_file} has no header")
            for row in reader:
                read, conflicts = _add_row_to_graph(
                    graph,
                    environment,
                    row,
                    config=config,
                    source_file=source_file,
                    category_hint=category_hint,
                )
                row_count += read
                conflict_count += conflicts
    graph.update_metadata(
        {
            "environment_node_id": environment,
            "source_file_count": source_count,
            "row_count": row_count,
            "conflicting_field_count": conflict_count,
            "uses_federated_field_aware_preprocessing": True,
        }
    )
    if conflict_count:
        LOGGER.warning(
            "Skipped %s ambiguous logical field value(s) while building G_new",
            conflict_count,
        )
    return graph
