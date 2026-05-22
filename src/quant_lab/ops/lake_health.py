from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import write_parquet_dataset

LAKE_FILE_HEALTH_DATASET = Path("gold") / "lake_file_health_daily"
DEFAULT_DATASET_PATHS = {
    "okx_public_ws": Path("bronze") / "okx_public_ws",
    "trade_print": Path("silver") / "trade_print",
    "orderbook_snapshot": Path("silver") / "orderbook_snapshot",
    "market_bar": Path("silver") / "market_bar",
    "cost_bucket_daily": Path("gold") / "cost_bucket_daily",
    "feature_value": Path("gold") / "feature_value",
    "alpha_evidence": Path("gold") / "alpha_evidence",
    "gate_decision": Path("gold") / "gate_decision",
    "risk_permission": Path("gold") / "risk_permission",
    "strategy_health_daily": Path("gold") / "strategy_health_daily",
    "api_request_metrics": Path("bronze") / "api_request_metrics",
    "job_run_history": Path("gold") / "job_run_history",
    "lake_file_health_daily": Path("gold") / "lake_file_health_daily",
}


@dataclass(frozen=True)
class LakeDatasetFileHealth:
    dataset: str
    path: str
    parquet_file_count: int
    total_size_bytes: int
    small_file_count: int
    small_file_ratio: float
    partition_dir_count: int
    largest_file_bytes: int
    status: str
    warning: str | None


def lake_file_health_rows(
    lake_root: str | Path,
    *,
    dataset_names: Iterable[str] | None = None,
    small_file_threshold_bytes: int = 1_000_000,
) -> list[dict[str, Any]]:
    root = Path(lake_root)
    names = list(dataset_names) if dataset_names is not None else sorted(DEFAULT_DATASET_PATHS)
    created_at = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for name in names:
        relative = DEFAULT_DATASET_PATHS.get(name, Path(name))
        path = root / relative
        files = _parquet_files(path)
        sizes = _file_sizes(files)
        file_count = len(files)
        small_count = sum(1 for size in sizes if size < small_file_threshold_bytes)
        small_ratio = small_count / file_count if file_count else 0.0
        partition_dirs = _partition_dir_count(path)
        status, warning = _dataset_file_status(
            file_count=file_count,
            small_file_ratio=small_ratio,
            partition_dir_count=partition_dirs,
        )
        rows.append(
            {
                "day": created_at.date().isoformat(),
                "dataset": name,
                "path": str(path),
                "parquet_file_count": file_count,
                "total_size_bytes": sum(sizes),
                "small_file_count": small_count,
                "small_file_ratio": small_ratio,
                "partition_dir_count": partition_dirs,
                "largest_file_bytes": max(sizes) if sizes else 0,
                "status": status,
                "warning": warning,
                "created_at": created_at,
            }
        )
    return rows


def write_lake_file_health_daily(lake_root: str | Path) -> dict[str, Any]:
    rows = lake_file_health_rows(lake_root)
    df = pl.DataFrame(rows)
    write_parquet_dataset(df, Path(lake_root) / LAKE_FILE_HEALTH_DATASET)
    return _lake_file_health_summary(rows)


def lake_file_health_summary(lake_root: str | Path) -> dict[str, Any]:
    """Return lake file health without writing a daily health dataset."""

    return _lake_file_health_summary(lake_file_health_rows(lake_root))


def _lake_file_health_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_count": len(rows),
        "total_parquet_files": sum(int(row["parquet_file_count"]) for row in rows),
        "warning_count": sum(1 for row in rows if row["status"] != "OK"),
        "rows": rows,
    }


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if not path.exists():
        return []
    return [item for item in path.rglob("*.parquet") if item.is_file()]


def _file_sizes(files: list[Path]) -> list[int]:
    sizes: list[int] = []
    for file_path in files:
        try:
            sizes.append(file_path.stat().st_size)
        except OSError:
            continue
    return sizes


def _partition_dir_count(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    try:
        return sum(1 for item in path.rglob("*=*") if item.is_dir())
    except OSError:
        return 0


def _dataset_file_status(
    *,
    file_count: int,
    small_file_ratio: float,
    partition_dir_count: int,
) -> tuple[str, str | None]:
    if file_count == 0:
        return "MISSING", "dataset has no parquet files"
    if file_count > 10_000 and partition_dir_count == 0:
        return "CRITICAL", "large unpartitioned parquet file set"
    if file_count > 10_000:
        return "WARN", "large parquet file set; compaction recommended"
    if file_count > 1_000 and small_file_ratio > 0.8:
        return "WARN", "small-file ratio is high"
    return "OK", None
