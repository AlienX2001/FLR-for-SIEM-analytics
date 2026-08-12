from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class SourceRecord:
    source_file: Path
    source_row_number: int
    source_line_number: int
    row: dict[str, str]
    malformed: bool = False
    malformed_reason: str | None = None


@dataclass(frozen=True)
class PreprocessedBehaviourRow:
    normalized_text: str
    canonical_fields: Mapping[str, str]
    field_tokens: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class AnomalyEvidence:
    reason_code: str
    field: str | None
    details: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "reason_code": self.reason_code,
            "field": self.field,
            "details": self.details,
            "score": self.score,
        }


@dataclass(frozen=True)
class DetectionResult:
    is_anomaly: bool
    anomaly_score: float
    profile_scope: str
    evidence: tuple[AnomalyEvidence, ...]


@dataclass(frozen=True)
class DetectionSummary:
    total_rows: int
    anomaly_rows: int
    normal_rows: int
    malformed_rows: int
    reasons: Mapping[str, int]

