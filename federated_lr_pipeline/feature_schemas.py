from __future__ import annotations

import hashlib

NETWORK_ATTRIBUTES = [
    "event_time_epoch",
    "event_time_iso",
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
    "event_time_epoch",
    "event_time_iso",
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
    "event_time_epoch",
    "event_time_iso",
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
    "event_time_epoch",
    "event_time_iso",
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
    "event_time_epoch",
    "event_time_iso",
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
    "event_time_epoch",
    "event_time_iso",
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

CROSS_VOCABULARY_VERSION = 2
CROSS_VOCABULARY_SIZE = 1000
CROSS_VOCABULARY_LOCAL_EQUALS_GLOBAL = True

# Each left signal receives exactly 20 right signals across 10 scopes: 200
# features per left signal and 1,000 fixed cross features in total.
CROSS_LEFT_SIGNALS = (
    "encoded_command",
    "sensitive_file_read",
    "llm_file_read_tool",
    "failed_login_burst",
    "secret_read_tool",
)

CROSS_RIGHT_SIGNALS = (
    "large_upload",
    "external_post",
    "external_tls_post",
    "first_seen_domain",
    "rare_destination_ip",
    "dns_tunnel_pattern",
    "high_beacon_rate",
    "outbound_ssh",
    "outbound_rdp",
    "suspicious_user_agent",
    "http_post_to_ip",
    "domain_generation_pattern",
    "internal_port_scan",
    "smb_lateral_connection",
    "high_connection_fanout",
    "cleartext_credential_post",
    "cloud_storage_upload",
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

SUBCATEGORY_NAMES = ["system", "network", "llm", "identity", "cloud", "cross"]

# Backward-compatible aliases for older tests/imports.
INTER_CATEGORY_ATTRIBUTES = CROSS_ATTRIBUTES
SPECIALIZED_MODEL_SCHEMAS = SUBCATEGORY_SCHEMAS
SPECIALIZED_MODEL_NAMES = ["network", "system", "cross"]
