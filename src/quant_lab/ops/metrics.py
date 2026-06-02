from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import polars as pl

from quant_lab.data.lake import (
    append_parquet_dataset,
    read_parquet_dataset,
    read_parquet_lazy,
    upsert_parquet_dataset,
)

API_METRICS_DATASET = Path("bronze") / "api_request_metrics"
JOB_RUN_HISTORY_DATASET = Path("gold") / "job_run_history"
_API_METRICS_LOCK = threading.Lock()
_API_METRICS_BUFFERS: dict[str, list[dict[str, Any]]] = {}
_API_METRICS_LAST_FLUSH: dict[str, float] = {}
_API_METRICS_FLUSH_THREADS: dict[str, threading.Thread] = {}
_API_METRICS_FLUSH_TIMERS: dict[str, threading.Timer] = {}

F = TypeVar("F")


@atexit.register
def _flush_all_api_request_metrics() -> None:
    for root_key in list(_API_METRICS_BUFFERS):
        try:
            _cancel_api_request_metrics_timer(root_key)
            _wait_api_request_metrics_flush(root_key)
            flush_api_request_metrics(root_key)
        except Exception:
            pass


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
    cache_hit: bool | None = None,
    rows_returned: int | None = None,
    response_bytes: int | None = None,
    lake_scan_ms: float | None = None,
    serialize_ms: float | None = None,
    source_signature_ms: float | None = None,
    response_cache_hit: bool | None = None,
    dependency_meta_missing: bool | None = None,
    error_type: str | None = None,
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
        "cache_hit": bool(cache_hit) if cache_hit is not None else None,
        "rows_returned": int(rows_returned) if rows_returned is not None else None,
        "response_bytes": int(response_bytes) if response_bytes is not None else None,
        "lake_scan_ms": round(float(lake_scan_ms), 3) if lake_scan_ms is not None else None,
        "serialize_ms": round(float(serialize_ms), 3) if serialize_ms is not None else None,
        "source_signature_ms": (
            round(float(source_signature_ms), 3) if source_signature_ms is not None else None
        ),
        "response_cache_hit": (
            bool(response_cache_hit) if response_cache_hit is not None else None
        ),
        "dependency_meta_missing": (
            bool(dependency_meta_missing) if dependency_meta_missing is not None else None
        ),
        "error_type": error_type,
    }
    root_key = str(Path(lake_root))
    should_flush = False
    should_schedule_timer = False
    now_monotonic = time.monotonic()
    with _API_METRICS_LOCK:
        buffer = _API_METRICS_BUFFERS.setdefault(root_key, [])
        buffer_was_empty = not buffer
        buffer.append(row)
        last_flush = _API_METRICS_LAST_FLUSH.setdefault(root_key, now_monotonic)
        should_flush = (
            len(buffer) >= _api_metrics_flush_rows()
            or (now_monotonic - last_flush) >= _api_metrics_flush_seconds()
        )
        should_schedule_timer = (
            buffer_was_empty and not should_flush and _api_metrics_async_flush_enabled()
        )
    if should_flush:
        _cancel_api_request_metrics_timer(root_key)
        if _api_metrics_async_flush_enabled():
            _schedule_api_request_metrics_flush(lake_root)
        else:
            flush_api_request_metrics(lake_root)
    elif should_schedule_timer:
        _schedule_api_request_metrics_timer(lake_root)


def flush_api_request_metrics(lake_root: str | Path) -> int:
    root_key = str(Path(lake_root))
    with _API_METRICS_LOCK:
        rows = _API_METRICS_BUFFERS.pop(root_key, [])
        _API_METRICS_LAST_FLUSH[root_key] = time.monotonic()
    if not rows:
        return 0
    result = append_parquet_dataset(
        pl.DataFrame(rows),
        Path(lake_root) / API_METRICS_DATASET,
        partition_by=["day"],
        target_rows_per_file=10_000,
        file_prefix="api",
    )
    return result.rows_written


def _schedule_api_request_metrics_flush(lake_root: str | Path) -> None:
    root_key = str(Path(lake_root))
    with _API_METRICS_LOCK:
        existing = _API_METRICS_FLUSH_THREADS.get(root_key)
        if existing is not None and existing.is_alive():
            return
        thread = threading.Thread(
            target=_api_metrics_flush_worker,
            args=(root_key,),
            name="quant-lab-api-metrics-flush",
            daemon=True,
        )
        _API_METRICS_FLUSH_THREADS[root_key] = thread
        thread.start()


def _schedule_api_request_metrics_timer(lake_root: str | Path) -> None:
    root_key = str(Path(lake_root))
    with _API_METRICS_LOCK:
        existing = _API_METRICS_FLUSH_TIMERS.get(root_key)
        if existing is not None and existing.is_alive():
            return
        timer = threading.Timer(
            _api_metrics_flush_seconds(),
            _api_metrics_timer_worker,
            args=(root_key,),
        )
        timer.daemon = True
        _API_METRICS_FLUSH_TIMERS[root_key] = timer
    timer.start()


def _api_metrics_timer_worker(root_key: str) -> None:
    try:
        flush_api_request_metrics(root_key)
    finally:
        schedule_next = False
        with _API_METRICS_LOCK:
            current = _API_METRICS_FLUSH_TIMERS.get(root_key)
            if current is threading.current_thread():
                _API_METRICS_FLUSH_TIMERS.pop(root_key, None)
            schedule_next = bool(_API_METRICS_BUFFERS.get(root_key))
        if schedule_next:
            _schedule_api_request_metrics_timer(root_key)


def _cancel_api_request_metrics_timer(lake_root: str | Path) -> None:
    root_key = str(Path(lake_root))
    with _API_METRICS_LOCK:
        timer = _API_METRICS_FLUSH_TIMERS.pop(root_key, None)
    if timer is not None and timer is not threading.current_thread():
        timer.cancel()


def _api_metrics_flush_worker(root_key: str) -> None:
    try:
        flush_api_request_metrics(root_key)
    finally:
        with _API_METRICS_LOCK:
            current = _API_METRICS_FLUSH_THREADS.get(root_key)
            if current is threading.current_thread():
                _API_METRICS_FLUSH_THREADS.pop(root_key, None)


def _wait_api_request_metrics_flush(lake_root: str | Path) -> None:
    root_key = str(Path(lake_root))
    with _API_METRICS_LOCK:
        thread = _API_METRICS_FLUSH_THREADS.get(root_key)
    if thread is None or thread is threading.current_thread():
        return
    thread.join(timeout=_api_metrics_flush_join_seconds())


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
    upsert_parquet_dataset(
        pl.DataFrame([row]),
        Path(lake_root) / JOB_RUN_HISTORY_DATASET,
        key_columns=["job_name", "started_at", "finished_at"],
        max_rows=_job_run_history_max_rows(),
        max_rows_sort_by=["finished_at", "started_at"],
        max_rows_descending=True,
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
    since_minutes: int | None = None,
) -> dict[str, Any]:
    _wait_api_request_metrics_flush(lake_root)
    flush_api_request_metrics(lake_root)
    lazy = _lazy_dataset_or_none(Path(lake_root) / API_METRICS_DATASET, schema_union=True)
    if lazy is None:
        return _empty_api_metrics_summary()
    schema_names = _lazy_schema_names(lazy)
    day = _normalize_summary_day(day)
    scoped = lazy.filter(pl.col("day") == day) if day and "day" in schema_names else lazy
    scoped = _filter_since_minutes(
        scoped,
        schema_names=schema_names,
        timestamp_column="request_ts",
        since_minutes=since_minutes,
    )
    request_count = _lazy_count(scoped)
    if request_count == 0:
        return _empty_api_metrics_summary()
    latency = {}
    latency_by_path = {}
    slow_paths = []
    if "duration_ms" in schema_names:
        metrics = (
            scoped.select(
                [
                    pl.col("duration_ms").cast(pl.Float64, strict=False).median().alias("p50"),
                    pl.col("duration_ms")
                    .cast(pl.Float64, strict=False)
                    .quantile(0.90)
                    .alias("p90"),
                    pl.col("duration_ms")
                    .cast(pl.Float64, strict=False)
                    .quantile(0.95)
                    .alias("p95"),
                    pl.col("duration_ms")
                    .cast(pl.Float64, strict=False)
                    .quantile(0.99)
                    .alias("p99"),
                    pl.col("duration_ms").cast(pl.Float64, strict=False).max().alias("max"),
                ]
            )
            .collect()
            .to_dicts()[0]
        )
        latency = {key: _float_or_none(value) for key, value in metrics.items()}
        latency_by_path = _latency_by_path_lazy(scoped, schema_names=schema_names)
        slow_paths = _slow_paths(latency_by_path)
    return {
        "request_count": request_count,
        "by_path": _count_by_lazy(scoped, "path", schema_names=schema_names),
        "by_status_code": _count_by_lazy(scoped, "status_code", schema_names=schema_names),
        "latency_ms": latency,
        "latency_by_path_ms": latency_by_path,
        "slow_paths": slow_paths,
        "cache_hit_count": _sum_bool_lazy(scoped, "cache_hit", schema_names=schema_names),
        "rows_returned_total": _sum_numeric_lazy(
            scoped,
            "rows_returned",
            schema_names=schema_names,
        ),
        "response_bytes_total": _sum_numeric_lazy(
            scoped,
            "response_bytes",
            schema_names=schema_names,
        ),
        "lake_scan_ms_total": _sum_numeric_lazy(scoped, "lake_scan_ms", schema_names=schema_names),
        "serialize_ms_total": _sum_numeric_lazy(scoped, "serialize_ms", schema_names=schema_names),
        "source_signature_ms_total": _sum_numeric_lazy(
            scoped,
            "source_signature_ms",
            schema_names=schema_names,
        ),
        "response_cache_hit_count": _sum_bool_lazy(
            scoped,
            "response_cache_hit",
            schema_names=schema_names,
        ),
        "dependency_meta_missing_count": _sum_bool_lazy(
            scoped,
            "dependency_meta_missing",
            schema_names=schema_names,
        ),
        "by_error_type": _count_by_non_empty_lazy(scoped, "error_type", schema_names=schema_names),
    }


def job_run_summary(
    lake_root: str | Path,
    *,
    day: str | None = None,
    since_minutes: int | None = None,
) -> dict[str, Any]:
    lazy = _lazy_dataset_or_none(Path(lake_root) / JOB_RUN_HISTORY_DATASET)
    if lazy is None:
        return {"run_count": 0, "jobs": []}
    schema_names = _lazy_schema_names(lazy)
    day = _normalize_summary_day(day)
    scoped = lazy.filter(pl.col("day") == day) if day and "day" in schema_names else lazy
    timestamp_column = "finished_at" if "finished_at" in schema_names else "started_at"
    scoped = _filter_since_minutes(
        scoped,
        schema_names=schema_names,
        timestamp_column=timestamp_column,
        since_minutes=since_minutes,
    )
    run_count = _lazy_count(scoped)
    if run_count == 0:
        return {"run_count": 0, "jobs": []}
    required = {"job_name", "status", "duration_seconds"}
    if not required.issubset(schema_names):
        return {"run_count": 0, "jobs": []}
    grouped = (
        scoped.group_by("job_name")
        .agg(
            [
                pl.len().alias("run_count"),
                (pl.col("status") == "failed").sum().alias("failure_count"),
                pl.col("duration_seconds").cast(pl.Float64, strict=False).mean().alias("avg_s"),
                pl.col("duration_seconds")
                .cast(pl.Float64, strict=False)
                .quantile(0.95)
                .alias("p95_s"),
                pl.col("duration_seconds").cast(pl.Float64, strict=False).max().alias("max_s"),
                pl.col("duration_seconds")
                .cast(pl.Float64, strict=False)
                .sort_by("finished_at")
                .last()
                .alias("latest_duration_s")
                if "finished_at" in schema_names
                else pl.lit(None).alias("latest_duration_s"),
                pl.col("status").sort_by("finished_at").last().alias("latest_status")
                if "finished_at" in schema_names
                else pl.lit(None).alias("latest_status"),
                pl.col("finished_at").max().alias("latest_finished_at")
                if "finished_at" in schema_names
                else pl.lit(None).alias("latest_finished_at"),
            ]
        )
        .sort("job_name")
        .collect()
    )
    return {"run_count": run_count, "jobs": grouped.to_dicts()}


def _normalize_summary_day(day: str | None) -> str | None:
    if day is None:
        return None
    normalized = str(day).strip().lower()
    if not normalized:
        return None
    if normalized in {"auto", "today"}:
        return datetime.now(UTC).date().isoformat()
    return str(day).strip()


def _filter_since_minutes(
    lazy: pl.LazyFrame,
    *,
    schema_names: set[str],
    timestamp_column: str,
    since_minutes: int | None,
) -> pl.LazyFrame:
    if since_minutes is None or since_minutes <= 0 or timestamp_column not in schema_names:
        return lazy
    cutoff = datetime.now(UTC) - timedelta(minutes=since_minutes)
    return lazy.filter(
        pl.col(timestamp_column)
        .cast(pl.Utf8)
        .str.to_datetime(time_zone="UTC", strict=False)
        >= cutoff
    )


def _empty_api_metrics_summary() -> dict[str, Any]:
    return {
        "request_count": 0,
        "by_path": {},
        "by_status_code": {},
        "latency_ms": {},
        "latency_by_path_ms": {},
        "slow_paths": [],
        "cache_hit_count": 0,
        "rows_returned_total": 0,
        "response_bytes_total": 0,
        "lake_scan_ms_total": 0.0,
        "serialize_ms_total": 0.0,
        "source_signature_ms_total": 0.0,
        "response_cache_hit_count": 0,
        "dependency_meta_missing_count": 0,
        "by_error_type": {},
    }


def _lazy_dataset_or_none(path: Path, *, schema_union: bool = False) -> pl.LazyFrame | None:
    if schema_union:
        try:
            union = _read_parquet_dataset_with_schema_union(path)
        except Exception:
            union = pl.DataFrame()
        if not union.is_empty():
            return union.lazy()
    try:
        lazy = read_parquet_lazy(path)
        lazy.collect_schema()
        return lazy
    except Exception:
        try:
            fallback = read_parquet_dataset(path)
        except Exception:
            return None
        if fallback.is_empty():
            return None
        return fallback.lazy()


def _read_parquet_dataset_with_schema_union(path: Path) -> pl.DataFrame:
    files = _parquet_metric_files(path)
    if not files:
        return pl.DataFrame()
    frames: list[pl.DataFrame] = []
    for file_path in files:
        try:
            frame = pl.read_parquet(file_path)
        except Exception:
            continue
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _parquet_metric_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path] if not _is_internal_metric_file(path) else []
    if not path.exists():
        return []
    return sorted(
        candidate
        for candidate in path.rglob("*.parquet")
        if not _is_internal_metric_file(candidate)
    )


def _is_internal_metric_file(path: Path) -> bool:
    return (
        any(part == "._tmp" or part.startswith("__") for part in path.parts)
        or path.name.startswith(".")
        or path.name.endswith(".tmp.parquet")
    )


def _lazy_schema_names(lazy: pl.LazyFrame) -> set[str]:
    return set(lazy.collect_schema().names())


def _lazy_count(lazy: pl.LazyFrame) -> int:
    try:
        value = lazy.select(pl.len().alias("count")).collect().item(0, "count")
    except Exception:
        return 0
    return int(value or 0)


def _count_by_lazy(
    lazy: pl.LazyFrame,
    column: str,
    *,
    schema_names: set[str],
) -> dict[str, int]:
    if column not in schema_names:
        return {}
    grouped = lazy.group_by(column).len(name="count").collect()
    return {str(row[column]): int(row["count"]) for row in grouped.to_dicts()}


def _count_by_non_empty_lazy(
    lazy: pl.LazyFrame,
    column: str,
    *,
    schema_names: set[str],
) -> dict[str, int]:
    if column not in schema_names:
        return {}
    text = pl.col(column).cast(pl.Utf8, strict=False)
    try:
        grouped = (
            lazy.filter(text.is_not_null() & (text.str.strip_chars() != ""))
            .group_by(column)
            .len(name="count")
            .collect()
        )
    except Exception:
        return {}
    return {str(row[column]): int(row["count"]) for row in grouped.to_dicts()}


def _sum_bool_lazy(
    lazy: pl.LazyFrame,
    column: str,
    *,
    schema_names: set[str],
) -> int:
    if column not in schema_names:
        return 0
    try:
        value = (
            lazy.select(pl.col(column).fill_null(False).cast(pl.Int64).sum().alias("sum"))
            .collect()
            .item(0, "sum")
        )
    except Exception:
        return 0
    return int(value or 0)


def _sum_numeric_lazy(
    lazy: pl.LazyFrame,
    column: str,
    *,
    schema_names: set[str],
) -> float:
    if column not in schema_names:
        return 0.0
    try:
        value = (
            lazy.select(pl.col(column).cast(pl.Float64, strict=False).sum().alias("sum"))
            .collect()
            .item(0, "sum")
        )
    except Exception:
        return 0.0
    return round(float(value or 0.0), 3)


def _latency_by_path_lazy(
    lazy: pl.LazyFrame,
    *,
    schema_names: set[str],
) -> dict[str, dict[str, float | int | None]]:
    if "path" not in schema_names or "duration_ms" not in schema_names:
        return {}
    duration = pl.col("duration_ms").cast(pl.Float64, strict=False)
    aggregations: list[pl.Expr] = [
        pl.len().alias("count"),
        duration.median().alias("p50"),
        duration.quantile(0.90).alias("p90"),
        duration.quantile(0.95).alias("p95"),
        duration.quantile(0.99).alias("p99"),
        duration.max().alias("max"),
    ]
    if "cache_hit" in schema_names:
        aggregations.append(
            pl.col("cache_hit")
            .fill_null(False)
            .cast(pl.Int64)
            .sum()
            .alias("cache_hit_count")
        )
    if "rows_returned" in schema_names:
        aggregations.append(
            pl.col("rows_returned")
            .cast(pl.Float64, strict=False)
            .sum()
            .alias("rows_returned_total")
        )
    if "response_bytes" in schema_names:
        aggregations.append(
            pl.col("response_bytes")
            .cast(pl.Float64, strict=False)
            .sum()
            .alias("response_bytes_total")
        )
    if "lake_scan_ms" in schema_names:
        aggregations.append(
            pl.col("lake_scan_ms").cast(pl.Float64, strict=False).sum().alias("lake_scan_ms_total")
        )
    if "serialize_ms" in schema_names:
        aggregations.append(
            pl.col("serialize_ms").cast(pl.Float64, strict=False).sum().alias("serialize_ms_total")
        )
    if "source_signature_ms" in schema_names:
        aggregations.append(
            pl.col("source_signature_ms")
            .cast(pl.Float64, strict=False)
            .sum()
            .alias("source_signature_ms_total")
        )
    if "response_cache_hit" in schema_names:
        aggregations.append(
            pl.col("response_cache_hit")
            .fill_null(False)
            .cast(pl.Int64)
            .sum()
            .alias("response_cache_hit_count")
        )
    if "dependency_meta_missing" in schema_names:
        aggregations.append(
            pl.col("dependency_meta_missing")
            .fill_null(False)
            .cast(pl.Int64)
            .sum()
            .alias("dependency_meta_missing_count")
        )
    if "error_type" in schema_names:
        aggregations.append(
            pl.col("error_type").is_not_null().cast(pl.Int64).sum().alias("error_count")
        )
    if "status_code" in schema_names:
        status = pl.col("status_code").cast(pl.Int64, strict=False)
        aggregations.append((status >= 500).sum().alias("server_error_count"))
        aggregations.append(((status >= 400) & (status < 500)).sum().alias("client_error_count"))
    grouped = lazy.group_by("path").agg(aggregations).sort("path").collect()
    result: dict[str, dict[str, float | int | None]] = {}
    for row in grouped.to_dicts():
        metrics: dict[str, float | int | None] = {
            "count": int(row.get("count") or 0),
            "p50": _float_or_none(row.get("p50")),
            "p90": _float_or_none(row.get("p90")),
            "p95": _float_or_none(row.get("p95")),
            "p99": _float_or_none(row.get("p99")),
            "max": _float_or_none(row.get("max")),
        }
        for metric in (
            "cache_hit_count",
            "rows_returned_total",
            "response_bytes_total",
            "lake_scan_ms_total",
            "serialize_ms_total",
            "source_signature_ms_total",
            "response_cache_hit_count",
            "dependency_meta_missing_count",
            "error_count",
        ):
            if metric in row:
                metrics[metric] = _float_or_none(row.get(metric)) or 0
        if "server_error_count" in row:
            metrics["server_error_count"] = int(row.get("server_error_count") or 0)
        if "client_error_count" in row:
            metrics["client_error_count"] = int(row.get("client_error_count") or 0)
        result[str(row["path"])] = metrics
    return result


def _slow_paths(
    latency_by_path: dict[str, dict[str, float | int | None]],
    *,
    limit: int = 10,
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for path, metrics in latency_by_path.items():
        rows.append({"path": path, **metrics})
    return sorted(
        rows,
        key=lambda row: (
            float(row["p95"] or 0),
            float(row["max"] or 0),
            int(row["count"] or 0),
        ),
        reverse=True,
    )[:limit]


def _float_or_none(value: Any) -> float | None:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _safe_user_agent(value: str | None) -> str | None:
    if not value:
        return None
    return value[:200]


def _api_metrics_flush_rows() -> int:
    return _positive_int_env("QUANT_LAB_API_METRICS_FLUSH_ROWS", 1_000)


def _api_metrics_flush_seconds() -> float:
    raw_value = os.environ.get("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "300")
    try:
        value = float(raw_value)
    except ValueError:
        return 300.0
    return max(value, 1.0)


def _api_metrics_flush_join_seconds() -> float:
    raw_value = os.environ.get("QUANT_LAB_API_METRICS_FLUSH_JOIN_SECONDS", "5")
    try:
        value = float(raw_value)
    except ValueError:
        return 5.0
    return max(value, 0.0)


def _api_metrics_async_flush_enabled() -> bool:
    value = os.environ.get("QUANT_LAB_API_METRICS_ASYNC_FLUSH")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _job_run_history_max_rows() -> int:
    return _positive_int_env("QUANT_LAB_JOB_RUN_HISTORY_MAX_ROWS", 10_000)


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(value, 1)


def _safe_error_message(value: str) -> str:
    lowered = value.lower()
    if any(token in lowered for token in ["secret", "passphrase", "token", "ok-access"]):
        return "[REDACTED]"
    try:
        json.loads(value)
        return "[REDACTED_JSON]"
    except json.JSONDecodeError:
        return value[:500]
