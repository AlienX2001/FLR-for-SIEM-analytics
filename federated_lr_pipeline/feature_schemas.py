from __future__ import annotations

import hashlib
import json

NETWORK_ATTRIBUTES = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "local_address",
    "local_port",
    "remote_address",
    "remote_port",
    "protocol_name",
    "protocol_number",
    "network_direction",
    "flow_duration",
    "duration",
    "rate",
    "srate",
    "drate",
    "header_length",
    "total_size",
    "total_sum",
    "packet_number",
    "iat",
    "tcp_fin",
    "tcp_syn",
    "tcp_rst",
    "tcp_psh",
    "tcp_ack",
    "tcp_urg",
    "tcp_ece",
    "tcp_cwr",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "protocol_http",
    "protocol_https",
    "protocol_dns",
    "protocol_tcp",
    "protocol_udp",
    "protocol_icmp",
]

SYSTEM_ATTRIBUTES = [
    "source",
    "entity_id",
    "entity_type",
    "source_entity_id",
    "target_entity_id",
    "process_pid",
    "process_ppid",
    "process_tgid",
    "process_name",
    "process_exe",
    "process_command_line",
    "user_uid",
    "user_euid",
    "group_gid",
    "group_egid",
    "file_path",
    "file_subtype",
    "file_permissions",
    "file_mode",
]

CROSS_ATTRIBUTES = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "network_direction",
    "flow_duration",
    "total_size",
    "process_pid",
    "process_ppid",
    "process_name",
    "process_exe",
    "process_command_line",
]

LLM_ATTRIBUTES = [
    "llm_provider",
    "llm_model",
    "llm_prompt",
    "llm_response",
    "llm_tool_name",
    "llm_tool_input",
    "llm_tool_output",
    "prompt",
    "response",
    "tool_name",
    "tool_input",
    "tool_output",
    "session_id",
    "user_uid",
]

IDENTITY_ATTRIBUTES = [
    "user_uid",
    "user_euid",
    "group_gid",
    "group_egid",
    "source_entity_id",
    "target_entity_id",
    "identity",
    "principal",
    "account_name",
    "login_result",
    "auth_method",
    "session_id",
    "src_ip",
    "dst_ip",
]

CLOUD_ATTRIBUTES = [
    "cloud_provider",
    "cloud_account",
    "cloud_region",
    "cloud_service",
    "cloud_action",
    "cloud_resource",
    "cloud_identity",
    "source",
    "entity_id",
    "user_uid",
    "src_ip",
    "dst_ip",
]

CROSS_VOCABULARY_VERSION = 3
CROSS_VOCABULARY_SIZE = 1000
CROSS_VOCABULARY_LOCAL_EQUALS_GLOBAL = True

# Ten left signals are paired with ten right signals across ten scopes, keeping
# the fixed vocabulary at exactly 1,000 features. Window signals are computed
# causally from preceding events and never from labels or inference results.
CROSS_LEFT_SIGNALS = (
    "encoded_command",
    "sensitive_file_read",
    "llm_file_read_tool",
    "failed_login_burst",
    "secret_read_tool",
    "window_flow_burst",
    "window_destination_fanout",
    "window_port_fanout",
    "window_high_volume",
    "window_repeated_destination",
)

CROSS_RIGHT_SIGNALS = (
    "large_upload",
    "external_post",
    "first_seen_domain",
    "rare_destination_ip",
    "high_beacon_rate",
    "outbound_ssh",
    "internal_port_scan",
    "new_tls_sni",
    "successful_login",
    "system_sensitive_file_read",
)

CROSS_VOCABULARY_SCOPES = (
    "same_host",
    "same_user",
    "same_session",
    "same_process_tree",
    "same_src_ip",
    "same_dst_ip",
    "same_entity",
    "same_process_pid",
    "same_parent_process",
    "same_network_zone",
)


def make_cross_category_token(left_signal: str, right_signal: str, scope: str) -> str:
    return f"cross:{left_signal}_and_{right_signal}_{scope}_15m"


CROSS_CATEGORY_TOKENS = [
    make_cross_category_token(left_signal, right_signal, scope)
    for left_signal in CROSS_LEFT_SIGNALS
    for right_signal in CROSS_RIGHT_SIGNALS
    for scope in CROSS_VOCABULARY_SCOPES
]

if len(CROSS_CATEGORY_TOKENS) != CROSS_VOCABULARY_SIZE:
    raise RuntimeError(
        f"Cross vocabulary must contain {CROSS_VOCABULARY_SIZE} tokens"
    )
if len(set(CROSS_CATEGORY_TOKENS)) != CROSS_VOCABULARY_SIZE:
    raise RuntimeError("Cross vocabulary contains duplicate tokens")
CROSS_VOCABULARY_SHA256 = hashlib.sha256(
    "\n".join(CROSS_CATEGORY_TOKENS).encode("utf-8")
).hexdigest()

SUBCATEGORY_SCHEMAS = {
    "network": NETWORK_ATTRIBUTES,
    "system": SYSTEM_ATTRIBUTES,
    "llm": LLM_ATTRIBUTES,
    "identity": IDENTITY_ATTRIBUTES,
    "cloud": CLOUD_ATTRIBUTES,
    "cross": CROSS_ATTRIBUTES,
}

# Coverage is based on evidence native to a specialist. In particular, source
# and destination IPs alone are network evidence, not identity or cloud evidence.
SUBCATEGORY_COVERAGE_ATTRIBUTES = {
    "network": frozenset(NETWORK_ATTRIBUTES),
    "system": frozenset(SYSTEM_ATTRIBUTES),
    "llm": frozenset(
        {
            "llm_provider",
            "llm_model",
            "llm_prompt",
            "llm_response",
            "llm_tool_name",
            "llm_tool_input",
            "llm_tool_output",
            "prompt",
            "response",
            "tool_name",
            "tool_input",
            "tool_output",
        }
    ),
    "identity": frozenset(
        {
            "user_uid",
            "user_euid",
            "group_gid",
            "group_egid",
            "identity",
            "principal",
            "account_name",
            "login_result",
            "auth_method",
            "session_id",
        }
    ),
    "cloud": frozenset(
        {
            "cloud_provider",
            "cloud_account",
            "cloud_region",
            "cloud_service",
            "cloud_action",
            "cloud_resource",
            "cloud_identity",
        }
    ),
    "cross": frozenset({"cross"}),
}

SUBCATEGORY_NAMES = ["system", "network", "llm", "identity", "cloud", "cross"]


# Context features summarize the causal [event_time - 15 minutes, event_time]
# window. Scope values partition state but are never emitted, preventing host,
# user, process, address, and timestamp identifiers from entering the LR input.
CONTEXT_FEATURE_VERSION = 1
CONTEXT_FEATURE_SCOPES = (
    "same_host",
    "same_user",
    "same_session",
    "same_process_tree",
    "same_src_ip",
    "same_dst_ip",
    "same_entity",
    "same_process_pid",
    "same_parent_process",
    "same_network_zone",
    "same_cloud_identity",
    "same_cloud_account",
    "same_container",
    "same_cluster",
)

CONTEXT_SCOPES_BY_SUBCATEGORY = {
    "network": (
        "same_host",
        "same_src_ip",
        "same_dst_ip",
        "same_network_zone",
    ),
    "system": (
        "same_host",
        "same_user",
        "same_session",
        "same_process_tree",
        "same_entity",
        "same_process_pid",
        "same_parent_process",
    ),
    "identity": (
        "same_host",
        "same_user",
        "same_session",
        "same_src_ip",
    ),
    "llm": (
        "same_host",
        "same_user",
        "same_session",
    ),
    "cloud": (
        "same_host",
        "same_cloud_identity",
        "same_cloud_account",
        "same_src_ip",
    ),
}

CONTEXT_COUNT_BUCKET_UPPER_BOUNDS = (
    0,
    1,
    3,
    7,
    15,
    31,
    63,
    127,
    255,
    511,
    1023,
    2047,
    4095,
)
CONTEXT_MAGNITUDE_BUCKET_UPPER_BOUNDS = (
    0,
    1,
    15,
    63,
    255,
    1023,
    4095,
    16383,
    65535,
    262143,
    1048575,
    4194303,
    16777215,
)
CONTEXT_INTERVAL_BUCKET_UPPER_BOUNDS = (
    0,
    1,
    5,
    15,
    30,
    60,
    120,
    300,
    600,
    900,
)
CONTEXT_FEATURE_METRICS = (
    "event_count",
    "interarrival_seconds",
    "window_span_seconds",
    "network_event_count",
    "system_event_count",
    "identity_event_count",
    "llm_event_count",
    "cloud_event_count",
    "distinct_source_count",
    "distinct_destination_count",
    "distinct_source_port_count",
    "distinct_destination_port_count",
    "distinct_protocol_count",
    "distinct_process_count",
    "total_size",
    "max_size",
    "total_packets",
    "max_packets",
    "inbound_count",
    "outbound_count",
    "failed_login_count",
    "successful_login_count",
    "sensitive_read_count",
    "large_upload_count",
    "new_destination_in_window",
    "repeated_destination_in_window",
    "failed_then_success",
    "sensitive_read_then_large_upload",
    "prior_system_activity",
    "prior_network_activity",
)

_CONTEXT_FEATURE_SCHEMA_PAYLOAD = {
    "version": CONTEXT_FEATURE_VERSION,
    "scopes": CONTEXT_FEATURE_SCOPES,
    "scopes_by_subcategory": CONTEXT_SCOPES_BY_SUBCATEGORY,
    "metrics": CONTEXT_FEATURE_METRICS,
    "count_bucket_upper_bounds": CONTEXT_COUNT_BUCKET_UPPER_BOUNDS,
    "magnitude_bucket_upper_bounds": CONTEXT_MAGNITUDE_BUCKET_UPPER_BOUNDS,
    "interval_bucket_upper_bounds": CONTEXT_INTERVAL_BUCKET_UPPER_BOUNDS,
}
CONTEXT_FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        _CONTEXT_FEATURE_SCHEMA_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

# Backward-compatible aliases for older tests/imports.
INTER_CATEGORY_ATTRIBUTES = CROSS_ATTRIBUTES
SPECIALIZED_MODEL_SCHEMAS = SUBCATEGORY_SCHEMAS
SPECIALIZED_MODEL_NAMES = ["network", "system", "cross"]
