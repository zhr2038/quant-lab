from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset

LAKE_FILE_INDEX = Path("bronze") / "lake_file_index"
TIMESTAMP_COLUMNS = ("ts", "timestamp", "received_at", "created_at", "minute_ts")
INDEX_VERSION = "lake_file_index.v0.2"


def build_lake_file_index(
    lake_root: str | Path,
    dataset_paths: Iterable[str | Path],
) -> pl.DataFrame:
    root = Path(lake_root)
    existing = read_parquet_dataset(root / LAKE_FILE_INDEX)
    existing_by_key = _existing_rows_by_key(existing)
    rows: list[dict] = []
    indexed_datasets: list[str] = []
    for dataset_path in dataset_paths:
        absolute = root / Path(dataset_path)
        dataset = str(absolute.relative_to(root)).replace("\\", "/")
        indexed_datasets.append(dataset)
        rows.extend(_index_dataset(root, absolute, existing_by_key=existing_by_key))
    frame = pl.DataFrame(rows, infer_schema_length=None)
    output = _merged_index_frame(existing, frame, indexed_datasets=indexed_datasets)
    if not output.is_empty():
        write_parquet_dataset(output, root / LAKE_FILE_INDEX)
    return frame


def recent_files_for_dataset(path: Path, *, since: datetime) -> list[Path] | None:
    lake_root = _infer_lake_root(path)
    if lake_root is None:
        return None
    index = read_parquet_dataset(lake_root / LAKE_FILE_INDEX)
    if index.is_empty() or "path" not in index.columns or "max_ts" not in index.columns:
        return None
    try:
        relative_dataset = str(path.relative_to(lake_root)).replace("\\", "/")
    except ValueError:
        return None
    scoped = index.filter(
        (pl.col("dataset") == relative_dataset)
        & (
            pl.col("max_ts")
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(time_zone="UTC", strict=False)
            >= since
        )
    )
    return [lake_root / item for item in scoped.get_column("path").to_list()]


def old_files_for_dataset(path: Path, *, before: datetime) -> list[Path] | None:
    lake_root = _infer_lake_root(path)
    if lake_root is None:
        return None
    index = read_parquet_dataset(lake_root / LAKE_FILE_INDEX)
    required = {"path", "dataset", "max_ts", "mtime_ns"}
    if index.is_empty() or not required.issubset(set(index.columns)):
        return None
    try:
        relative_dataset = str(path.relative_to(lake_root)).replace("\\", "/")
    except ValueError:
        return None
    cutoff_ns = int(before.timestamp() * 1_000_000_000)
    scoped = (
        index.with_columns(
            [
                pl.col("max_ts")
                .cast(pl.Utf8, strict=False)
                .str.to_datetime(time_zone="UTC", strict=False)
                .alias("_max_ts"),
                pl.col("mtime_ns").cast(pl.Int64, strict=False).alias("_mtime_ns"),
            ]
        )
        .filter(pl.col("dataset") == relative_dataset)
        .filter(
            (pl.col("_max_ts").is_not_null() & (pl.col("_max_ts") < before))
            | (
                pl.col("_max_ts").is_null()
                & pl.col("_mtime_ns").is_not_null()
                & (pl.col("_mtime_ns") < cutoff_ns)
            )
        )
    )
    return [lake_root / item for item in scoped.get_column("path").to_list()]


def files_fully_within_time_range(
    path: Path,
    *,
    since: datetime,
    before: datetime,
) -> list[Path] | None:
    """Return indexed files whose complete timestamp range is inside a window."""
    lake_root = _infer_lake_root(path)
    if lake_root is None:
        return None
    index = read_parquet_dataset(lake_root / LAKE_FILE_INDEX)
    required = {"path", "dataset", "min_ts", "max_ts"}
    if index.is_empty() or not required.issubset(set(index.columns)):
        return None
    try:
        relative_dataset = str(path.relative_to(lake_root)).replace("\\", "/")
    except ValueError:
        return None
    if before <= since:
        return []
    scoped = (
        index.with_columns(
            [
                pl.col("min_ts")
                .cast(pl.Utf8, strict=False)
                .str.to_datetime(time_zone="UTC", strict=False)
                .alias("_min_ts"),
                pl.col("max_ts")
                .cast(pl.Utf8, strict=False)
                .str.to_datetime(time_zone="UTC", strict=False)
                .alias("_max_ts"),
            ]
        )
        .filter(pl.col("dataset") == relative_dataset)
        .filter(
            pl.col("_min_ts").is_not_null()
            & pl.col("_max_ts").is_not_null()
            & (pl.col("_min_ts") >= since)
            & (pl.col("_max_ts") < before)
        )
    )
    return [lake_root / item for item in scoped.get_column("path").to_list()]


def _index_dataset(
    lake_root: Path,
    dataset_path: Path,
    *,
    existing_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict]:
    if not dataset_path.exists() or not dataset_path.is_dir():
        return []
    rows = []
    dataset = str(dataset_path.relative_to(lake_root)).replace("\\", "/")
    indexed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    for file_path in sorted(dataset_path.rglob("*.parquet")):
        if not file_path.is_file() or _is_internal_file(file_path):
            continue
        try:
            stat = file_path.stat()
        except Exception:
            continue
        relative_path = str(file_path.relative_to(lake_root)).replace("\\", "/")
        existing = existing_by_key.get((dataset, relative_path))
        if (
            existing is not None
            and _int_value(existing.get("mtime_ns")) == stat.st_mtime_ns
            and _int_value(existing.get("file_size")) == stat.st_size
        ):
            rows.append(_reuse_index_row(existing, stat=stat, indexed_at=indexed_at))
            continue
        try:
            min_ts, max_ts, row_count = _file_time_bounds(file_path)
        except Exception:
            continue
        rows.append(
            {
                "dataset": dataset,
                "path": relative_path,
                "min_ts": _iso(min_ts),
                "max_ts": _iso(max_ts),
                "row_count": row_count,
                "file_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "source_sha": f"{stat.st_mtime_ns}:{stat.st_size}",
                "indexed_at": indexed_at,
                "index_version": INDEX_VERSION,
                "reused_from_previous_index": False,
            }
        )
    return rows


def _existing_rows_by_key(frame: pl.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if frame.is_empty() or not {"dataset", "path"}.issubset(set(frame.columns)):
        return {}
    return {
        (str(row.get("dataset") or ""), str(row.get("path") or "")): row for row in frame.to_dicts()
    }


def _reuse_index_row(row: dict[str, Any], *, stat, indexed_at: str) -> dict[str, Any]:
    reused = dict(row)
    reused["file_size"] = stat.st_size
    reused["mtime_ns"] = stat.st_mtime_ns
    reused["source_sha"] = str(reused.get("source_sha") or f"{stat.st_mtime_ns}:{stat.st_size}")
    reused["indexed_at"] = str(reused.get("indexed_at") or indexed_at)
    reused["index_version"] = str(reused.get("index_version") or INDEX_VERSION)
    reused["reused_from_previous_index"] = True
    return reused


def _merged_index_frame(
    existing: pl.DataFrame,
    updated: pl.DataFrame,
    *,
    indexed_datasets: list[str],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    if not existing.is_empty() and {"dataset"}.issubset(set(existing.columns)):
        untouched = existing.filter(~pl.col("dataset").is_in(indexed_datasets))
        if not untouched.is_empty():
            frames.append(untouched)
    if not updated.is_empty():
        frames.append(updated)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _file_time_bounds(file_path: Path) -> tuple[datetime | None, datetime | None, int]:
    lazy = pl.scan_parquet(str(file_path), hive_partitioning=False)
    columns = set(lazy.collect_schema().names())
    ts_column = next((column for column in TIMESTAMP_COLUMNS if column in columns), None)
    if ts_column is None:
        count = lazy.select(pl.len().alias("rows")).collect().item(0, "rows")
        return None, None, int(count or 0)
    frame = lazy.select(
        [
            _timestamp_expr(ts_column).min().alias("min_ts"),
            _timestamp_expr(ts_column).max().alias("max_ts"),
            pl.len().alias("rows"),
        ]
    ).collect()
    return (
        _coerce_datetime(frame.item(0, "min_ts")),
        _coerce_datetime(frame.item(0, "max_ts")),
        int(frame.item(0, "rows") or 0),
    )


def _timestamp_expr(column: str) -> pl.Expr:
    return pl.coalesce(
        [
            pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(time_zone="UTC", strict=False),
        ]
    )


def _coerce_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value in (None, ""):
        return None
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except Exception:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso(value: datetime | None) -> str:
    return value.isoformat().replace("+00:00", "Z") if value else ""


def _infer_lake_root(path: Path) -> Path | None:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part in {"bronze", "silver", "gold"} and idx > 0:
            return Path(*parts[:idx])
    return None


def _is_internal_file(path: Path) -> bool:
    return path.name.startswith(".") or any(
        part.startswith("__") or part.startswith(".") for part in path.parts
    )
