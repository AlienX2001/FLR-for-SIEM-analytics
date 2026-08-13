from __future__ import annotations

from collections import Counter

from federated_lr_pipeline.feature_schemas import (
    CROSS_CATEGORY_TOKENS,
    CROSS_LEFT_SIGNALS,
    CROSS_RIGHT_SIGNALS,
    CROSS_VOCABULARY_SCOPES,
    CROSS_VOCABULARY_SIZE,
)
from federated_lr_pipeline.local_training import (
    build_feature_matrix,
    build_token_counters,
)
from federated_lr_pipeline.specialized_models import (
    BenignNoveltyBaseline,
    CROSS_TOKEN_SPECS,
    cross_tokens_for_row,
)
from federated_lr_pipeline.prf import derive_prf_key
from federated_lr_pipeline.vocab import tokenize


def test_cross_category_tokens_are_canonical_fixed_vocabulary() -> None:
    assert len(CROSS_CATEGORY_TOKENS) == CROSS_VOCABULARY_SIZE == 1000
    assert len(set(CROSS_CATEGORY_TOKENS)) == 1000
    assert all(token.startswith("cross:") for token in CROSS_CATEGORY_TOKENS)
    assert all(token.endswith("_15m") for token in CROSS_CATEGORY_TOKENS)
    assert all(token == token.lower() for token in CROSS_CATEGORY_TOKENS)
    assert all(tokenize(token) == [token] for token in CROSS_CATEGORY_TOKENS)


def test_cross_vocabulary_is_balanced_without_partial_groups() -> None:
    left_counts = Counter(left for _, left, _, _ in CROSS_TOKEN_SPECS)
    right_counts = Counter(right for _, _, right, _ in CROSS_TOKEN_SPECS)
    scope_counts = Counter(scope for _, _, _, scope in CROSS_TOKEN_SPECS)

    assert left_counts == Counter({signal: 200 for signal in CROSS_LEFT_SIGNALS})
    assert right_counts == Counter({signal: 50 for signal in CROSS_RIGHT_SIGNALS})
    assert scope_counts == Counter({scope: 100 for scope in CROSS_VOCABULARY_SCOPES})


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

    assert "cross:encoded_command_and_outbound_ssh_same_host_15m" in tokens
    assert "cross:encoded_command_and_outbound_ssh_same_user_15m" in tokens
    assert "cross:encoded_command_and_outbound_ssh_same_process_pid_15m" in tokens
    assert set(tokens).issubset(set(CROSS_CATEGORY_TOKENS))


def test_original_seed_conditions_remain_in_balanced_vocabulary() -> None:
    expected = {
        "cross:sensitive_file_read_and_large_upload_same_host_15m",
        "cross:encoded_command_and_first_seen_domain_same_host_15m",
        "cross:llm_file_read_tool_and_system_sensitive_file_read_same_user_15m",
        "cross:failed_login_burst_and_successful_login_same_user_15m",
        "cross:secret_read_tool_and_external_post_same_user_15m",
    }

    assert expected.issubset(set(CROSS_CATEGORY_TOKENS))


def test_emitted_cross_token_produces_nonzero_feature_matrix() -> None:
    token = "cross:encoded_command_and_outbound_ssh_same_host_15m"
    counters = build_token_counters([token])

    matrix = build_feature_matrix(
        [token],
        CROSS_CATEGORY_TOKENS,
        token_counters=counters,
    )

    assert matrix.shape == (1, 1000)
    assert matrix.nnz == 1
    assert matrix[0, CROSS_CATEGORY_TOKENS.index(token)] == 1.0


def test_new_tls_sni_cross_signal_is_reachable() -> None:
    prf_key = derive_prf_key(42)
    tokens = cross_tokens_for_row(
        {
            "host": "HOST-01",
            "process_command_line": "powershell.exe -EncodedCommand SQBFAFgA",
            "tls_sni": "first-seen.example",
        },
        benign_novelty_baseline=BenignNoveltyBaseline(
            benign_row_count=1,
            tagged_values={"snis": frozenset()},
        ),
        prf_key=prf_key,
    )

    assert "cross:encoded_command_and_new_tls_sni_same_host_15m" in tokens
