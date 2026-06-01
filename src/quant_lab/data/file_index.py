from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset

LAKE_FILE_INDEX = Path("bronze") / "lake_file_index"
TIMESTAMP_COLUMNS = ("ts", "timestamp", "received_at", "created_at", "minute_ts")


def build_lake_file_index(
    lake_root: str | Path,
    dataset_paths: Iterable[str | Path],
) -> pl.DataFrame:
    root = Path(lake_root)
    rows: list[dict] = []
    for dataset_path in dataset_paths:
        absolute = root / Path(dataset_path)
        rows.extend(_index_dataset(root, absolute))
    frame = pl.DataFrame(rows, infer_schema_length=None)
    if not frame.is_empty():
        upsert_parquet_dataset(
            frame,
            root / LAKE_FILE_INDEX,
            key_columns=["dataset", "path"],
        )
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


def _index_dataset(lake_root: Path, dataset_path: Path) -> list[dict]:
    if not dataset_path.exists() or not dataset_path.is_dir():
        return []
    rows = []
    dataset = str(dataset_path.relative_to(lake_root)).replace("\\", "/")
    for file_path in sorted(dataset_path.rglob("*.parquet")):
        if not file_path.is_file() or _is_internal_file(file_path):
            continue
        try:
            stat = file_path.stat()
            min_ts, max_ts, row_count = _file_time_bounds(file_path)
        except Exception:
            continue
        rows.append(
            {
                "dataset": dataset,
                "path": str(file_path.relative_to(lake_root)).replace("\\", "/"),
                "min_ts": _iso(min_ts),
                "max_ts": _iso(max_ts),
                "row_count": row_count,
                "file_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "source_sha": f"{stat.st_mtime_ns}:{stat.st_size}",
            }
        )
    return rows


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
