"""Shared constants for the alpha-validity audit pipeline."""

import os
from pathlib import Path

LOCAL_AUDIT_ROOT = Path(os.environ.get("LOCAL_AUDIT_ROOT", "/home/hr/quant-alpha-audit"))
DATA_RAW = LOCAL_AUDIT_ROOT / "data" / "raw"
DATA_SILVER = LOCAL_AUDIT_ROOT / "data" / "silver"
DATA_GOLD = LOCAL_AUDIT_ROOT / "data" / "gold"
WORK = LOCAL_AUDIT_ROOT / "work"
ARTIFACTS = LOCAL_AUDIT_ROOT / "artifacts"
REPORTS = LOCAL_AUDIT_ROOT / "reports"
LOGS = LOCAL_AUDIT_ROOT / "logs"
MANIFESTS = LOCAL_AUDIT_ROOT / "manifests"
CHECKPOINTS = LOCAL_AUDIT_ROOT / "checkpoints"
SNAPSHOTS = LOCAL_AUDIT_ROOT / "snapshots"
BUNDLES = LOCAL_AUDIT_ROOT / "bundles"

CANDLES_DIR = DATA_RAW / "okx_candles"
FUNDING_DIR = DATA_RAW / "okx_funding"

# audit window: two full years ending at the snapshot cutoff day (UTC)
AUDIT_END_DATE = "2026-07-18"
AUDIT_START_DATE = "2024-07-18"

DECISION_DELAY_BARS = (
    1  # bars between feature timestamp and executable decision (one_bar_delay in prod config)
)

for _p in (
    DATA_RAW,
    DATA_SILVER,
    DATA_GOLD,
    WORK,
    ARTIFACTS,
    REPORTS,
    LOGS,
    MANIFESTS,
    CHECKPOINTS,
    CANDLES_DIR,
    FUNDING_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)


def latest_snapshot_id() -> str:
    return (SNAPSHOTS / "LATEST").read_text().strip()


def snapshot_dir() -> Path:
    return SNAPSHOTS / latest_snapshot_id()
