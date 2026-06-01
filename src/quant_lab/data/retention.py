from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

HIGH_FREQUENCY_HOT_HOURS = 24
HIGH_FREQUENCY_ARCHIVE_ROOT = Path("archive") / "high_frequency"


def high_frequency_hot_cutoff(
    *,
    now: datetime | None = None,
    hot_hours: int = HIGH_FREQUENCY_HOT_HOURS,
) -> datetime:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current - timedelta(hours=max(int(hot_hours), 1))


def high_frequency_archive_path(
    lake_root: str | Path,
    dataset_relative_path: str | Path,
    *,
    ts: datetime,
    symbol: str = "unknown",
    filename: str,
) -> Path:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    dataset = Path(dataset_relative_path)
    return (
        Path(lake_root)
        / HIGH_FREQUENCY_ARCHIVE_ROOT
        / dataset
        / f"date={ts.date().isoformat()}"
        / f"hour={ts.hour:02d}"
        / f"symbol={symbol.replace('/', '-') or 'unknown'}"
        / filename
    )

