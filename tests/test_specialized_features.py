from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from federated_lr_pipeline.data import load_org_dataset
from federated_lr_pipeline.feature_schemas import NETWORK_ATTRIBUTES, SUBCATEGORY_SCHEMAS
from federated_lr_pipeline.specialized_models import (
    build_specialized_texts_for_org,
    field_aware_tokens,
    representation_ambiguity_diagnostics,
    token_counter_has_subcategory_coverage,
)


def test_field_aware_tokens_include_full_value_and_subtokens() -> None:
    tokens = field_aware_tokens("process_exe", r"C:\Windows\System32\cmd.exe")

    assert r"process_exe=c:\windows\system32\cmd.exe" in tokens
    assert "process_exe:windows" in tokens
    assert "process_exe:system32" in tokens
    assert "process_exe:cmd" in tokens
    assert "process_exe:exe" in tokens


def test_specialized_feature_selection_uses_only_model_attributes(tmp_path: Path) -> None:
    logs = tmp_path / "logs.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "protocol_name": ["TCP"],
            "dst_port": [443],
            "total_size": [1234],
            "source": ["auditd"],
            "process_pid": [4688],
            "process_exe": [r"C:\Windows\System32\cmd.exe"],
            "leaky_label_text": ["malicious"],
        }
    ).to_csv(logs, index=False)
    pd.DataFrame({"label": ["malicious"]}).to_csv(labels, index=False)
    dataset = load_org_dataset(logs, labels, org_index=0)

    network_texts, network_missing = build_specialized_texts_for_org(
        dataset,
        "network",
        ["protocol_name", "dst_port", "total_size"],
    )
    system_texts, system_missing = build_specialized_texts_for_org(
        dataset,
        "system",
        ["source", "process_pid", "process_exe"],
    )
    inter_texts, inter_missing = build_specialized_texts_for_org(
        dataset,
        "inter_category",
        ["protocol_name", "dst_port", "source", "process_pid"],
    )

    assert not network_missing
    assert not system_missing
    assert not inter_missing
    assert "protocol_name=tcp" in network_texts[0]
    assert "dst_port=443" in network_texts[0]
    assert "total_size:bucket=1024_to_lt_2048" in network_texts[0]
    assert "process_pid:bucket=4096_to_lt_8192" not in network_texts[0]
    assert "process_pid:bucket=4096_to_lt_8192" in system_texts[0]
    assert "dst_port=443" not in system_texts[0]
    assert "protocol_name=tcp" in inter_texts[0]
    assert "process_pid:bucket=4096_to_lt_8192" in inter_texts[0]
    assert "leaky_label_text" not in network_texts[0]
    assert "leaky_label_text" not in system_texts[0]
    assert "leaky_label_text" not in inter_texts[0]


def test_missing_columns_are_reported_gracefully(tmp_path: Path) -> None:
    logs = tmp_path / "logs.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"protocol_name": ["UDP"]}).to_csv(logs, index=False)
    pd.DataFrame({"label": ["benign"]}).to_csv(labels, index=False)
    dataset = load_org_dataset(logs, labels, org_index=0)

    texts, missing = build_specialized_texts_for_org(
        dataset,
        "network",
        ["protocol_name", "dst_port", "protocol_dns"],
    )

    assert texts == ["protocol_name=udp"]
    assert missing == ["dst_port", "protocol_dns"]


def test_remnant_label_columns_are_not_model_attributes() -> None:
    assert "label" not in NETWORK_ATTRIBUTES
    assert "sub_label" not in NETWORK_ATTRIBUTES
    assert "sub_label_cat" not in NETWORK_ATTRIBUTES


def test_timestamps_are_not_specialist_lr_attributes() -> None:
    for attributes in SUBCATEGORY_SCHEMAS.values():
        assert "event_time_epoch" not in attributes
        assert "event_time_iso" not in attributes


def test_specialist_text_does_not_encode_timestamp_values(tmp_path: Path) -> None:
    logs = tmp_path / "logs.csv"
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "event_time_epoch": [1767261600],
            "event_time_iso": ["2026-01-01T10:00:00Z"],
            "protocol_name": ["TCP"],
        }
    ).to_csv(logs, index=False)
    pd.DataFrame({"label": ["benign"]}).to_csv(labels, index=False)
    dataset = load_org_dataset(logs, labels, org_index=0)

    texts, _ = build_specialized_texts_for_org(
        dataset,
        "network",
        SUBCATEGORY_SCHEMAS["network"],
    )

    assert "protocol_name=tcp" in texts[0]
    assert "event_time_epoch" not in texts[0]
    assert "event_time_iso" not in texts[0]


def test_high_cardinality_numeric_fields_use_fixed_buckets() -> None:
    assert field_aware_tokens("rate", 1025) == [
        "rate:bucket=1024_to_lt_2048"
    ]
    assert field_aware_tokens("rate", 1900) == [
        "rate:bucket=1024_to_lt_2048"
    ]
    assert field_aware_tokens("rate", 2048) == [
        "rate:bucket=2048_to_lt_4096"
    ]
    assert field_aware_tokens("process_pid", 4688) == [
        "process_pid:bucket=4096_to_lt_8192"
    ]


def test_ports_use_service_and_range_buckets() -> None:
    assert field_aware_tokens("dst_port", 443) == [
        "dst_port:class=system",
        "dst_port:service=https",
        "dst_port=443",
    ]
    assert field_aware_tokens("src_port", 55000) == [
        "src_port:class=dynamic"
    ]


def test_low_cardinality_numeric_categories_remain_exact() -> None:
    assert field_aware_tokens("protocol_number", 6) == ["protocol_number=6"]
    assert field_aware_tokens("tcp_syn", 1) == ["tcp_syn=1"]
    assert field_aware_tokens("protocol_https", 0) == ["protocol_https=0"]


def test_string_identifiers_are_not_bucketed() -> None:
    assert field_aware_tokens("user_uid", "1001") == ["user_uid=1001"]
    assert field_aware_tokens("user_euid", "1001") == ["user_euid=1001"]
    assert field_aware_tokens("group_gid", "100") == ["group_gid=100"]
    assert field_aware_tokens("group_egid", "100") == ["group_egid=100"]


def test_representation_diagnostics_detect_conflicting_labels() -> None:
    diagnostics = representation_ambiguity_diagnostics(
        org_datasets=[
            SimpleNamespace(
                org_index=0,
                labels=["benign", "collection", "discovery", "collection"],
            )
        ],
        splits=[SimpleNamespace(train_indices=np.arange(4, dtype=int))],
        token_counters_by_subcategory={
            "network": [
                [
                    Counter({"protocol_name=tcp": 1}),
                    Counter({"dst_port:class=system": 1}),
                    Counter({"dst_port:class=system": 1}),
                    Counter({"dst_port:class=dynamic": 1}),
                ]
            ]
        },
        subcategories=["network"],
    )

    attack = diagnostics["per_org"][0]["non_benign_training_rows"]
    assert attack["num_rows"] == 3
    assert attack["num_unique_representations"] == 2
    assert attack["rows_in_ambiguous_representations"] == 2
    assert attack["exact_representation_majority_ceiling"] == 2 / 3


def test_identity_coverage_requires_identity_native_evidence() -> None:
    assert not token_counter_has_subcategory_coverage(
        Counter({"src_ip=10.0.0.1": 1, "dst_ip=10.0.0.2": 1}),
        "identity",
    )
    assert token_counter_has_subcategory_coverage(
        Counter({"src_ip=10.0.0.1": 1, "user_uid=1001": 1}),
        "identity",
    )
