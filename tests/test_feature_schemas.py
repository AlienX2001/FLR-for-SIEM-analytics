from __future__ import annotations

from federated_lr_pipeline.feature_schemas import CROSS_CATEGORY_TOKENS
from federated_lr_pipeline.specialized_models import cross_tokens_for_row


def test_cross_category_tokens_are_expanded_15m_fixed_vocabulary() -> None:
    assert len(CROSS_CATEGORY_TOKENS) == 1000
    assert len(set(CROSS_CATEGORY_TOKENS)) == 1000
    assert all(token.startswith("cross:") for token in CROSS_CATEGORY_TOKENS)
    assert all(token.endswith("_15m") for token in CROSS_CATEGORY_TOKENS)
    assert CROSS_CATEGORY_TOKENS[:5] == [
        "cross:sensitive_file_read_AND_large_upload_same_host_15m",
        "cross:encoded_command_AND_first_seen_domain_same_host_15m",
        "cross:llm_file_read_tool_AND_system_sensitive_file_read_same_user_15m",
        "cross:failed_login_burst_AND_successful_login_same_user_15m",
        "cross:secret_read_tool_AND_external_post_same_user_15m",
    ]


def test_cross_token_extractor_emits_expanded_vocabulary_conditions() -> None:
    tokens = cross_tokens_for_row(
        {
            "host": "HOST-01",
            "user_uid": "1000",
            "process_pid": "4242",
            "process_ppid": "4000",
            "src_ip": "10.0.0.10",
            "dst_ip": "8.8.8.8",
            "dst_port": "22",
            "network_direction": "outbound",
            "process_command_line": "powershell.exe -EncodedCommand SQBFAFgA",
        }
    )

    assert "cross:encoded_command_AND_outbound_ssh_same_host_15m" in tokens
    assert "cross:encoded_command_AND_outbound_ssh_same_user_15m" in tokens
    assert "cross:encoded_command_AND_outbound_ssh_same_process_pid_15m" in tokens
    assert set(tokens).issubset(set(CROSS_CATEGORY_TOKENS))


def test_cross_token_extractor_preserves_original_five_conditions() -> None:
    tokens = cross_tokens_for_row(
        {
            "host": "HOST-01",
            "user_uid": "1000",
            "process_command_line": "powershell.exe -EncodedCommand read secret token",
            "process_exe": "cmd.exe",
            "file_path": "/tmp/secret-token.txt",
            "llm_tool_name": "file_read",
            "llm_tool_input": "/tmp/secret-token.txt",
            "sub_label": "failed_login",
            "label": "bruteforce",
            "total_size": "6000",
            "dst_ip": "8.8.8.8",
            "tls_sni": "new.example.com",
        }
    )

    for token in CROSS_CATEGORY_TOKENS[:5]:
        assert token in tokens


def test_new_tls_sni_cross_signal_is_reachable() -> None:
    tokens = cross_tokens_for_row(
        {
            "host": "HOST-01",
            "process_command_line": "powershell.exe -EncodedCommand SQBFAFgA",
            "tls_sni": "first-seen.example",
        }
    )

    assert "cross:encoded_command_AND_new_tls_sni_same_host_15m" in tokens
