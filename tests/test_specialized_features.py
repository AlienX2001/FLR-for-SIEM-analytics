from __future__ import annotations

from pathlib import Path

import pandas as pd

from federated_lr_pipeline.data import load_org_dataset
from federated_lr_pipeline.specialized_models import (
    build_specialized_texts_for_org,
    field_aware_tokens,
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
    assert "process_pid=4688" not in network_texts[0]
    assert "process_pid=4688" in system_texts[0]
    assert "dst_port=443" not in system_texts[0]
    assert "protocol_name=tcp" in inter_texts[0]
    assert "process_pid=4688" in inter_texts[0]
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
