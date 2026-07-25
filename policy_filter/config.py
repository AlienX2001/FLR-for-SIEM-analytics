from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilterCliConfig:
    logs: list[Path]
    policy: Path
    output: Path
    encoding: str = "utf-8"
    delimiter: str = ","
    default_timezone: str | None = None
    strict_policy_validation: bool = False
    log_level: str = "INFO"
