from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import polars as pl

from quant_lab.data.lake import append_parquet_dataset, read_parquet_dataset

API_METRICS_DATASET = Path("bronze") / "api_request_metrics"
JOB_RUN_HISTORY_DATASET = Path("gold") / "job_run_history"

F = TypeVar("F")


@dataclass(frozen=True)
class JobRunRecord:
    job_name: str
    status: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    error_type: str | None = None
    error_message: str | None = None


def record_api_request(
    *,
    lake_root: str | Path,
    method: str,
    path: str,
    status_code: int,
    duration_seconds: float,
    client_host: str | None = None,
    user_agent: str | None = None,
    request_ts: datetime | None = None,
) -> None:
    timestamp = request_ts or datetime.now(UTC)
    row = {
        "day": timestamp.date().isoformat(),
        "request_ts": timestamp,
        "method": method,
        "path": path,
        "status_code": int(status_code),
        "duration_ms": round(float(duration_seconds) * 1000.0, 3),
        "client_host": client_host,
        "user_agent": _safe_user_agent(user_agent),
    }
    append_parquet_dataset(
        pl.DataFrame([row]),
        Path(lake_root) / API_METRICS_DATASET,
        partition_by=["day", "path"],
        target_rows_per_file=10_000,
        file_prefix="api",
    )


def record_job_run(
    *,
    lake_root: str | Path,
    job_name: str,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    error: BaseException | None = None,
) -> None:
    duration = max((finished_at - started_at).total_seconds(), 0.0)
    row = {
        "day": started_at.date().isoformat(),
        "job_name": job_name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration, 3),
        "error_type": type(error).__name__ if error else None,
        "error_message": _safe_error_message(str(error)) if error else None,
    }
    append_parquet_dataset(
        pl.DataFrame([row]),
        Path(lake_root) / JOB_RUN_HISTORY_DATASET,
        partition_by=["day", "job_name"],
        target_rows_per_file=10_000,
        file_prefix="job",
    )


def run_with_job_metrics(
    *,
    lake_root: str | Path,
    job_name: str,
    func: Callable[[], F],
) -> F:
    started = datetime.now(UTC)
    error: BaseException | None = None
    try:
        result = func()
    except BaseException as exc:
        error = exc
        raise
    finally:
        finished = datetime.now(UTC)
        try:
            record_job_run(
                lake_root=lake_root,
                job_name=job_name,
                status="failed" if error else "succeeded",
                started_at=started,
                finished_at=finished,
                error=error,
            )
        except Exception:
            # Metrics must never make the production job fail.
            pass
    return result


def api_metrics_summary(
    lake_root: str | Path,
    *,
    day: str | None = None,
) -> dict[str, Any]:
    df = read_parquet_dataset(Path(lake_root) / API_METRICS_DATASET)
    if df.is_empty():
        return {
            "request_count": 0,
            "by_path": {},
            "by_status_code": {},
            "latency_ms": {},
        }
    scoped = df.filter(pl.col("day") == day) if day and "day" in df.columns else df
    if scoped.is_empty():
        return {
            "request_count": 0,
            "by_path": {},
            "by_status_code": {},
            "latency_ms": {},
        }
    latency = {}
    if "duration_ms" in scoped.columns:
        metrics = scoped.select(
            [
                pl.col("duration_ms").cast(pl.Float64, strict=False).median().alias("p50"),
                pl.col("duration_ms").cast(pl.Float64, strict=False).quantile(0.95).alias("p95"),
                pl.col("duration_ms").cast(pl.Float64, strict=False).max().alias("max"),
            ]
        ).to_dicts()[0]
        latency = {key: _float_or_none(value) for key, value in metrics.items()}
    return {
        "request_count": scoped.height,
        "by_path": _count_by(scoped, "path"),
        "by_status_code": _count_by(scoped, "status_code"),
        "latency_ms": latency,
    }


def job_run_summary(
    lake_root: str | Path,
    *,
    day: str | None = None,
) -> dict[str, Any]:
    df = read_parquet_dataset(Path(lake_root) / JOB_RUN_HISTORY_DATASET)
    if df.is_empty():
        return {"run_count": 0, "jobs": []}
    scoped = df.filter(pl.col("day") == day) if day and "day" in df.columns else df
    if scoped.is_empty():
        return {"run_count": 0, "jobs": []}
    grouped = (
        scoped.group_by("job_name")
        .agg(
            [
                pl.len().alias("run_count"),
                (pl.col("status") == "failed").sum().alias("failure_count"),
                pl.col("duration_seconds").cast(pl.Float64, strict=False).mean().alias("avg_s"),
                pl.col("duration_seconds").cast(pl.Float64, strict=False).max().alias("max_s"),
            ]
        )
        .sort("job_name")
    )
    return {"run_count": scoped.height, "jobs": grouped.to_dicts()}


def _count_by(df: pl.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {
        str(row[column]): int(row["count"])
        for row in df.group_by(column).len(name="count").to_dicts()
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _safe_user_agent(value: str | None) -> str | None:
    if not value:
        return None
    return value[:200]


def _safe_error_message(value: str) -> str:
    lowered = value.lower()
    if any(token in lowered for token in ["secret", "passphrase", "token", "ok-access"]):
        return "[REDACTED]"
    try:
        json.loads(value)
        return "[REDACTED_JSON]"
    except json.JSONDecodeError:
        return value[:500]
