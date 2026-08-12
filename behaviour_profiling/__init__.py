"""Standalone benign-behavior profiling and anomaly detection."""

from behaviour_profiling.detector import detect_anomalies
from behaviour_profiling.trainer import train_profile

__all__ = ["detect_anomalies", "train_profile"]
