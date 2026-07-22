import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import MarketBar, require_utc
from quant_lab.data.market_bar_time import (
    DEFAULT_MARKET_BAR_TIMEFRAME,
    market_bar_close_ts,
)
from quant_lab.symbols import normalize_symbol

MARKET_BAR_DATASET = Path("silver") / "market_bar"
MARKET_BAR_HEALTH_DATASET = Path("silver") / "market_bar_health"
MARKET_BAR_PRIMARY_KEY = ["venue", "symbol", "timeframe", "ts"]
PARQUET_MAGIC = b"PAR1"
MIN_PARQUET_SIZE_BYTES = 12
logger = logging.getLogger(__name__)
_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.Lock] = {}
MARKET_BAR_SCHEMA = {
    "venue": pl.Utf8,
    "symbol": pl.Utf8,
    "market_type": pl.Utf8,
    "timeframe": pl.Utf8,
    "ts": pl.Datetime(time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "quote_volume": pl.Float64,
    "source": pl.Utf8,
    "ingest_ts": pl.Datetime(time_zone="UTC"),
    "is_closed": pl.Boolean,
}


class LakeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: Path
    created_by: str = Field(default="quant-lab", min_length=1)


@dataclass(frozen=True)
class AppendParquetResult:
    dataset_path: str
    rows_written: int
    file_count: int
    partition_by: list[str]
    auto_compact_triggered: bool = False
    compact_source_file_count: int = 0
    compact_output_file_count: int = 0


@dataclass(frozen=True)
class CompactParquetResult:
    dataset_path: str
    source_file_count: int
    output_file_count: int
    rows: int
    partition_by: list[str]
    target_rows_per_file: int
    max_source_files_per_batch: int
    max_source_batch_bytes: int = 0


@dataclass(frozen=True)
class RepairParquetPartitionResult:
    dataset_path: str
    bad_file_count: int
    repaired_file_count: int
    repaired_rows: int
    removed_bad_file_count: int
    partition_by: list[str]


def write_parquet_dataset(
    df: pl.DataFrame,
    dataset_path: str | Path,
    partition_by: str | Sequence[str] | None = None,
) -> Path:
    path = Path(dataset_path)
    with _dataset_lock(path):
        return _write_parquet_dataset_unlocked(df, path, partition_by=partition_by)


def write_single_file_parquet_dataset_in_place(
    df: pl.DataFrame,
    dataset_path: str | Path,
) -> Path:
    """Atomically replace one-file dataset contents without replacing its directory.

    This is intended for datasets whose systemd write boundary is the dataset
    directory itself. Lock and staging files stay inside that directory, so the
    parent can remain read-only.
    """

    path = Path(dataset_path)
    path.mkdir(parents=True, exist_ok=True)
    _ensure_lake_dir_permissions(path)
    lock_anchor = path / "_in_place_payload"
    with _dataset_lock(lock_anchor):
        staging = path / f"._data_write_{uuid.uuid4().hex}.tmp"
        target = path / "data.parquet"
        try:
            _sort_dataframe(df).write_parquet(staging)
            os.replace(_replaceable_path(staging), _replaceable_path(target))
            _ensure_internal_tree_permissions(path)
            _remove_stale_parquet_files(path, keep=target)
        finally:
            try:
                staging.unlink()
            except FileNotFoundError:
                pass
    return path


def write_snapshot_meta(
    dataset_path: str | Path,
    *,
    dataset_name: str,
    frame: pl.DataFrame,
    schema_version: str | None = None,
    generated_at: str | datetime | None = None,
    expires_at: str | datetime | None = None,
) -> Path:
    """Write an atomic dataset snapshot sidecar for small API dependency tables."""

    path = Path(dataset_path)
    path.mkdir(parents=True, exist_ok=True)
    generated_at_text = _snapshot_text(
        generated_at
    ) or _frame_latest_snapshot_text(
        frame,
        (
            "snapshot_generated_at",
            "generated_at",
            "generated_at_utc",
            "updated_at",
            "created_at",
            "as_of_ts",
            "as_of_date",
            "latest_bundle_ts",
            "bundle_ts",
            "ingest_ts",
            "event_ts",
            "ts_utc",
            "ts",
            "entry_ts",
            "exit_ts",
            "paper_date",
            "date",
            "day",
        ),
    )
    expires_at_text = _snapshot_text(expires_at) or _frame_latest_snapshot_text(
        frame,
        ("expires_at",),
    )
    payload = {
        "dataset": str(dataset_name),
        "generated_at": generated_at_text,
        "expires_at": expires_at_text,
        "row_count": int(frame.height),
        "source_sha": _snapshot_source_sha(
            str(dataset_name),
            frame,
            generated_at=generated_at_text,
            expires_at=expires_at_text,
        ),
        "file_count": sum(1 for candidate in path.rglob("*.parquet") if candidate.is_file()),
        "schema_version": schema_version or _frame_first_snapshot_text(frame, "schema_version")
        or str(dataset_name),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    tmp_path = path / "._snapshot_meta.tmp"
    meta_path = path / "_snapshot_meta.json"
    tmp_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_snapshot_json),
        encoding="utf-8",
    )
    os.replace(_replaceable_path(tmp_path), _replaceable_path(meta_path))
    return meta_path


def _write_parquet_dataset_unlocked(
    df: pl.DataFrame,
    dataset_path: str | Path,
    partition_by: str | Sequence[str] | None = None,
    *,
    preserve_files: Sequence[str] = (),
) -> Path:
    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_lake_dir_permissions(path.parent)

    sorted_df = _sort_dataframe(df)

    if partition_by:
        staging = path.parent / f"__{path.name}_write_{uuid.uuid4().hex}"
        backup = path.parent / f"__{path.name}_backup_{uuid.uuid4().hex}"
        try:
            sorted_df.write_parquet(staging, partition_by=partition_by, mkdir=True)
            _copy_preserved_dataset_files(path, staging, preserve_files)
            _ensure_internal_tree_permissions(staging)
            if path.exists():
                _replace_path(path, backup)
            try:
                _replace_path(staging, path)
                _ensure_internal_tree_permissions(path)
            except Exception:
                if backup.exists() and not path.exists():
                    _replace_path(backup, path)
                raise
            _remove_internal_path(backup)
        except Exception:
            _remove_internal_path(staging)
            if backup.exists() and not path.exists():
                _replace_path(backup, path)
            raise
        return path

    staging = path.parent / f"__{path.name}_write_{uuid.uuid4().hex}"
    backup = path.parent / f"__{path.name}_backup_{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True, exist_ok=False)
        _ensure_lake_dir_permissions(staging)
        sorted_df.write_parquet(staging / "data.parquet")
        _copy_preserved_dataset_files(path, staging, preserve_files)
        if path.exists():
            _replace_path(path, backup)
        try:
            _replace_path(staging, path)
            _ensure_lake_dir_permissions(path)
        except Exception:
            if backup.exists() and not path.exists():
                _replace_path(backup, path)
            raise
        _remove_internal_path(backup)
    except Exception:
        _remove_internal_path(staging)
        if backup.exists() and not path.exists():
            _replace_path(backup, path)
        raise
    return path


def append_parquet_dataset(
    df: pl.DataFrame,
    dataset_path: str | Path,
    *,
    partition_by: str | Sequence[str] | None = None,
    target_rows_per_file: int | None = None,
    file_prefix: str = "part",
) -> AppendParquetResult:
    """Append a batch without reading or rewriting existing dataset files.

    This is intended for high-frequency append-only datasets such as WebSocket
    trade prints and order book snapshots. Low-frequency gold/silver tables that
    need logical upsert semantics should keep using ``upsert_parquet_dataset``.
    """

    path = Path(dataset_path)
    if df.is_empty():
        return AppendParquetResult(str(path), 0, 0, _partition_columns(partition_by))
    with _dataset_lock(path):
        return _append_parquet_dataset_unlocked(
            df,
            path,
            partition_by=partition_by,
            target_rows_per_file=target_rows_per_file,
            file_prefix=file_prefix,
        )


def _append_parquet_dataset_unlocked(
    df: pl.DataFrame,
    dataset_path: Path,
    *,
    partition_by: str | Sequence[str] | None = None,
    target_rows_per_file: int | None = None,
    file_prefix: str = "part",
    auto_compact: bool = True,
) -> AppendParquetResult:
    path = Path(dataset_path)
    path.mkdir(parents=True, exist_ok=True)
    _ensure_lake_dir_permissions(path)
    frame = _sort_dataframe(df)
    partition_columns = _partition_columns(partition_by)
    chunks = _partitioned_chunks(frame, partition_columns)
    rows_written = 0
    file_count = 0
    compact_source_file_count = 0
    compact_output_file_count = 0
    max_rows = max(int(target_rows_per_file or frame.height), 1)
    touched_dirs: set[Path] = set()
    for partition_values, partition_frame in chunks:
        partition_dir = _partition_dir(path, partition_values)
        partition_dir.mkdir(parents=True, exist_ok=True)
        _ensure_lake_dir_permissions(partition_dir)
        touched_dirs.add(partition_dir)
        for offset in range(0, partition_frame.height, max_rows):
            chunk = partition_frame.slice(offset, max_rows)
            if chunk.is_empty():
                continue
            final_file = partition_dir / _append_file_name(file_prefix)
            temp_file = _dataset_temp_file(path)
            try:
                chunk.write_parquet(temp_file)
                _replace_path(temp_file, final_file)
            except Exception:
                try:
                    temp_file.unlink()
                except FileNotFoundError:
                    pass
                raise
            rows_written += chunk.height
            file_count += 1
    if auto_compact:
        for compact_dir in sorted(touched_dirs):
            result = _auto_compact_append_dataset_unlocked(
                compact_dir,
                target_rows_per_file=max_rows,
            )
            if result is None:
                continue
            compact_source_file_count += result.source_file_count
            compact_output_file_count += result.output_file_count
    return AppendParquetResult(
        str(path),
        rows_written,
        file_count,
        partition_columns,
        auto_compact_triggered=compact_source_file_count > 0,
        compact_source_file_count=compact_source_file_count,
        compact_output_file_count=compact_output_file_count,
    )


def compact_parquet_dataset(
    dataset_path: str | Path,
    *,
    partition_by: str | Sequence[str] | None = None,
    target_rows_per_file: int = 250_000,
    max_source_files_per_batch: int = 5_000,
    max_source_batch_bytes: int | None = None,
) -> CompactParquetResult:
    """Rewrite a dataset into fewer deterministic partitioned parquet files."""

    path = Path(dataset_path)
    partition_columns = _partition_columns(partition_by)
    batch_bytes = _resolve_max_source_batch_bytes(max_source_batch_bytes)
    with _dataset_lock(path, timeout_seconds=120.0):
        files = _parquet_files(path)
        source_file_count = len(files)
        if not files:
            return CompactParquetResult(
                dataset_path=str(path),
                source_file_count=0,
                output_file_count=0,
                rows=0,
                partition_by=partition_columns,
                target_rows_per_file=target_rows_per_file,
                max_source_files_per_batch=max_source_files_per_batch,
                max_source_batch_bytes=batch_bytes,
            )
        staging = path.parent / f"__{path.name}_compact_{uuid.uuid4().hex}"
        output_file_count = 0
        rows = 0
        try:
            for batch_files in _parquet_file_batches(
                files,
                max_source_files_per_batch=max_source_files_per_batch,
                max_source_batch_bytes=batch_bytes,
            ):
                frame = _read_parquet_files(batch_files, schema_union=True)
                if frame.is_empty():
                    continue
                result = _append_parquet_dataset_unlocked(
                    frame,
                    staging,
                    partition_by=partition_columns,
                    target_rows_per_file=target_rows_per_file,
                    file_prefix="compact",
                    auto_compact=False,
                )
                output_file_count += result.file_count
                rows += result.rows_written
            if rows == 0:
                _remove_existing_dataset(path)
                path.mkdir(parents=True, exist_ok=True)
                _ensure_lake_dir_permissions(path)
                return CompactParquetResult(
                    dataset_path=str(path),
                    source_file_count=source_file_count,
                    output_file_count=0,
                    rows=0,
                    partition_by=partition_columns,
                    target_rows_per_file=target_rows_per_file,
                    max_source_files_per_batch=max_source_files_per_batch,
                    max_source_batch_bytes=batch_bytes,
                )
            backup = path.parent / f"__{path.name}_backup_{uuid.uuid4().hex}"
            if path.exists():
                _replace_path(path, backup)
            _replace_path(staging, path)
            _remove_internal_path(backup)
            return CompactParquetResult(
                dataset_path=str(path),
                source_file_count=source_file_count,
                output_file_count=output_file_count,
                rows=rows,
                partition_by=partition_columns,
                target_rows_per_file=target_rows_per_file,
                max_source_files_per_batch=max_source_files_per_batch,
                max_source_batch_bytes=batch_bytes,
            )
        except Exception:
            _remove_internal_path(staging)
            raise


def compact_parquet_directory_files(
    directory_path: str | Path,
    *,
    target_rows_per_file: int = 250_000,
    max_source_files_per_batch: int = 5_000,
    max_source_batch_bytes: int | None = None,
    include_existing_compact_files: bool = False,
    max_total_source_files: int | None = None,
) -> CompactParquetResult:
    """Compact only Parquet files directly inside a directory.

    Hot append datasets may contain both direct batch files and older partition
    directories. Direct compaction rewrites only the current directory's files,
    preserving partition subdirectories and avoiding an expensive full dataset
    rewrite.
    """

    path = Path(directory_path)
    with _dataset_lock(path, timeout_seconds=120.0):
        files = _direct_compaction_source_files(
            path,
            include_existing_compact_files=include_existing_compact_files,
        )
        if max_total_source_files is not None and max_total_source_files > 0:
            files = files[:max_total_source_files]
        return _compact_direct_parquet_files_unlocked(
            path,
            files,
            target_rows_per_file=target_rows_per_file,
            max_source_files_per_batch=max_source_files_per_batch,
            max_source_batch_bytes=max_source_batch_bytes,
        )


def _auto_compact_append_dataset_unlocked(
    dataset_path: Path,
    *,
    target_rows_per_file: int,
) -> CompactParquetResult | None:
    threshold = _int_env("QUANT_LAB_APPEND_AUTO_COMPACT_FILES", 64)
    if threshold <= 0:
        return None
    files = _direct_compaction_source_files(dataset_path)
    file_count_before = len(files)
    if file_count_before <= threshold:
        return None
    min_total_bytes = _int_env("QUANT_LAB_APPEND_AUTO_COMPACT_MIN_TOTAL_BYTES", 0)
    total_source_bytes = _file_size_sum(files)
    if min_total_bytes > 0 and total_source_bytes < min_total_bytes:
        logger.debug(
            "skip_append_auto_compact dataset_path=%s partition_dir=%s "
            "file_count_before=%s total_source_bytes=%s min_total_bytes=%s",
            dataset_path,
            dataset_path,
            file_count_before,
            total_source_bytes,
            min_total_bytes,
        )
        return None
    max_source_files = _int_env(
        "QUANT_LAB_APPEND_AUTO_COMPACT_MAX_SOURCE_FILES",
        max(threshold, 512),
    )
    result = _compact_direct_parquet_files_unlocked(
        dataset_path,
        files,
        target_rows_per_file=_int_env(
            "QUANT_LAB_APPEND_AUTO_COMPACT_TARGET_ROWS",
            target_rows_per_file,
        ),
        max_source_files_per_batch=max_source_files,
        max_source_batch_bytes=None,
    )
    file_count_after = len(_direct_compaction_source_files(dataset_path))
    logger.info(
        "append_auto_compact dataset_path=%s partition_dir=%s "
        "file_count_before=%s file_count_after=%s "
        "compact_source_file_count=%s compact_output_file_count=%s "
        "total_source_bytes=%s",
        dataset_path,
        dataset_path,
        file_count_before,
        file_count_after,
        result.source_file_count,
        result.output_file_count,
        total_source_bytes,
    )
    return result


def _compact_direct_parquet_files_unlocked(
    path: Path,
    files: Sequence[Path],
    *,
    target_rows_per_file: int,
    max_source_files_per_batch: int,
    max_source_batch_bytes: int | None,
) -> CompactParquetResult:
    batch_bytes = _resolve_max_source_batch_bytes(max_source_batch_bytes)
    source_file_count = len(files)
    if not files:
        return CompactParquetResult(
            dataset_path=str(path),
            source_file_count=0,
            output_file_count=0,
            rows=0,
            partition_by=[],
            target_rows_per_file=target_rows_per_file,
            max_source_files_per_batch=max_source_files_per_batch,
            max_source_batch_bytes=batch_bytes,
        )

    staging = path / f"__direct_compact_{uuid.uuid4().hex}"
    output_file_count = 0
    rows = 0
    try:
        for batch_files in _parquet_file_batches(
            files,
            max_source_files_per_batch=max_source_files_per_batch,
            max_source_batch_bytes=batch_bytes,
        ):
            frame = _read_parquet_files(batch_files, schema_union=True)
            if frame.is_empty():
                continue
            result = _append_parquet_dataset_unlocked(
                frame,
                staging,
                partition_by=None,
                target_rows_per_file=target_rows_per_file,
                file_prefix="compact",
                auto_compact=False,
            )
            output_file_count += result.file_count
            rows += result.rows_written

        if rows == 0:
            for source_file in files:
                try:
                    source_file.unlink()
                except FileNotFoundError:
                    pass
            return CompactParquetResult(
                dataset_path=str(path),
                source_file_count=source_file_count,
                output_file_count=0,
                rows=0,
                partition_by=[],
                target_rows_per_file=target_rows_per_file,
                max_source_files_per_batch=max_source_files_per_batch,
                max_source_batch_bytes=batch_bytes,
            )

        for source_file in files:
            source_file.unlink()
        for compacted_file in sorted(
            candidate
            for candidate in staging.glob("*.parquet")
            if _is_valid_parquet_file(candidate)
        ):
            _replace_path(compacted_file, path / compacted_file.name)
        _remove_internal_path(staging)
        return CompactParquetResult(
            dataset_path=str(path),
            source_file_count=source_file_count,
            output_file_count=output_file_count,
            rows=rows,
            partition_by=[],
            target_rows_per_file=target_rows_per_file,
            max_source_files_per_batch=max_source_files_per_batch,
            max_source_batch_bytes=batch_bytes,
        )
    except Exception:
        _remove_internal_path(staging)
        raise


def _file_size_sum(files: Sequence[Path]) -> int:
    total = 0
    for file_path in files:
        try:
            total += file_path.stat().st_size
        except OSError:
            continue
    return total


def repair_parquet_partition_values(
    dataset_path: str | Path,
    *,
    partition_by: str | Sequence[str] | None,
    bad_values: Sequence[str] = ("__null__", "__empty__"),
    target_rows_per_file: int = 250_000,
    max_source_files_per_batch: int = 5_000,
    max_source_batch_bytes: int | None = None,
) -> RepairParquetPartitionResult:
    """Rewrite files from invalid hive partition directories into valid directories.

    This is a targeted maintenance primitive for append-only datasets. It avoids
    a full dataset rewrite by reading only files under partition directories such
    as ``day=__null__`` and rewriting those rows with repairable partition values.
    """

    path = Path(dataset_path)
    partition_columns = _partition_columns(partition_by)
    batch_bytes = _resolve_max_source_batch_bytes(max_source_batch_bytes)
    with _dataset_lock(path, timeout_seconds=120.0):
        bad_files = _bad_partition_files(path, partition_columns, bad_values)
        bad_file_count = len(bad_files)
        if not bad_files:
            _remove_empty_bad_partition_directories(path, partition_columns, bad_values)
            return RepairParquetPartitionResult(
                dataset_path=str(path),
                bad_file_count=0,
                repaired_file_count=0,
                repaired_rows=0,
                removed_bad_file_count=0,
                partition_by=partition_columns,
            )

        staging = path.parent / f"__{path.name}_repair_{uuid.uuid4().hex}"
        repaired_file_count = 0
        repaired_rows = 0
        try:
            for batch_files in _parquet_file_batches(
                bad_files,
                max_source_files_per_batch=max_source_files_per_batch,
                max_source_batch_bytes=batch_bytes,
            ):
                frame = _read_parquet_files(batch_files)
                if frame.is_empty():
                    continue
                repaired = _repair_partition_frame(frame, partition_columns)
                result = _append_parquet_dataset_unlocked(
                    repaired,
                    staging,
                    partition_by=partition_columns,
                    target_rows_per_file=target_rows_per_file,
                    file_prefix="repair",
                    auto_compact=False,
                )
                repaired_file_count += result.file_count
                repaired_rows += result.rows_written

            moved_file_count = _move_repaired_staging_files(staging, path)
            removed = 0
            touched_dirs = {file.parent for file in bad_files}
            for source_file in bad_files:
                try:
                    source_file.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
            _remove_empty_directories(touched_dirs, stop_at=path)
            _remove_empty_bad_partition_directories(path, partition_columns, bad_values)
            _remove_internal_path(staging)
            return RepairParquetPartitionResult(
                dataset_path=str(path),
                bad_file_count=bad_file_count,
                repaired_file_count=moved_file_count,
                repaired_rows=repaired_rows,
                removed_bad_file_count=removed,
                partition_by=partition_columns,
            )
        except Exception:
            _remove_internal_path(staging)
            raise


def read_parquet_dataset(dataset_path: str | Path) -> pl.DataFrame:
    files = _parquet_files(dataset_path)
    if not files:
        return pl.DataFrame()
    return _read_parquet_files(files)


def count_parquet_rows(dataset_path: str | Path) -> int:
    files = _parquet_files(dataset_path)
    if not files:
        return 0
    try:
        return int(_scan_parquet_files(files).select(pl.len().alias("rows")).collect().item())
    except Exception:
        return _read_parquet_files(files).height


def invalid_parquet_files(dataset_path: str | Path) -> list[Path]:
    return [path for path in _all_parquet_files(dataset_path) if not _is_valid_parquet_file(path)]


def upsert_parquet_dataset(
    df: pl.DataFrame,
    dataset_path: str | Path,
    key_columns: Sequence[str],
    *,
    max_rows: int | None = None,
    max_rows_sort_by: Sequence[str] | None = None,
    max_rows_descending: bool = True,
    streaming_upsert: bool = False,
    preserve_files: Sequence[str] = (),
) -> int:
    path = Path(dataset_path)
    with _dataset_lock(path):
        if streaming_upsert and max_rows is None:
            try:
                return _streaming_upsert_parquet_dataset_unlocked(
                    df,
                    path,
                    key_columns=key_columns,
                    preserve_files=preserve_files,
                )
            except Exception:
                logger.warning(
                    "streaming upsert failed for %s; using in-memory fallback",
                    path,
                    exc_info=True,
                )
        existing_df = read_parquet_dataset(path)
        frames = [frame for frame in [existing_df, df] if not frame.is_empty()]
        combined = pl.concat(frames, how="diagonal_relaxed") if frames else df
        if not combined.is_empty():
            available_keys = [column for column in key_columns if column in combined.columns]
            if available_keys:
                combined = combined.unique(subset=available_keys, keep="last", maintain_order=True)
            combined = _limit_dataframe_rows(
                combined,
                max_rows=max_rows,
                sort_by=max_rows_sort_by,
                descending=max_rows_descending,
            )
        _write_parquet_dataset_unlocked(
            combined,
            path,
            preserve_files=preserve_files,
        )
        return combined.height


def _streaming_upsert_parquet_dataset_unlocked(
    df: pl.DataFrame,
    dataset_path: Path,
    *,
    key_columns: Sequence[str],
    preserve_files: Sequence[str] = (),
) -> int:
    """Upsert a small batch into a large Parquet history with bounded memory."""

    files = _parquet_files(dataset_path)
    if not files:
        _write_parquet_dataset_unlocked(
            df,
            dataset_path,
            preserve_files=preserve_files,
        )
        return df.height
    existing_schema = _scan_parquet_files(files, schema_union=True).collect_schema().names()
    available_keys = [
        column for column in key_columns if column in df.columns and column in existing_schema
    ]
    if not available_keys:
        raise ValueError("streaming upsert requires at least one shared key column")
    incoming = df.unique(subset=available_keys, keep="last", maintain_order=True)

    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_lake_dir_permissions(path.parent)
    staging = path.parent / f"__{path.name}_stream_upsert_{uuid.uuid4().hex}"
    backup = path.parent / f"__{path.name}_backup_{uuid.uuid4().hex}"
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        staging.mkdir(parents=True, exist_ok=False)
        _ensure_lake_dir_permissions(staging)
        incoming_path = staging / "incoming.parquet"
        output_path = staging / "data.parquet"
        temp_directory = staging / "duckdb_tmp"
        temp_directory.mkdir(parents=True, exist_ok=False)
        incoming.write_parquet(incoming_path)

        connection = duckdb.connect(database=":memory:", read_only=False)
        connection.execute("SET threads = 1")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET memory_limit = '512MB'")
        connection.execute(
            f"SET temp_directory = {_duckdb_sql_literal(temp_directory)}"
        )
        existing_paths = ",".join(_duckdb_sql_literal(file) for file in files)
        using_columns = ",".join(_duckdb_identifier(column) for column in available_keys)
        incoming_sql = _duckdb_sql_literal(incoming_path)
        output_sql = _duckdb_sql_literal(output_path)
        query = f"""
            WITH incoming AS (
                SELECT * FROM read_parquet({incoming_sql})
            ), retained AS (
                SELECT existing.*
                FROM read_parquet([{existing_paths}], union_by_name=true) AS existing
                ANTI JOIN incoming USING ({using_columns})
            )
            SELECT * FROM retained
            UNION ALL BY NAME
            SELECT * FROM incoming
        """
        connection.execute(
            f"COPY ({query}) TO {output_sql} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        connection.close()
        connection = None
        incoming_path.unlink()
        _remove_internal_path(temp_directory)
        rows = int(pl.scan_parquet(output_path).select(pl.len()).collect().item())
        _copy_preserved_dataset_files(path, staging, preserve_files)
        _ensure_internal_tree_permissions(staging)
        if path.exists():
            _replace_path(path, backup)
        try:
            _replace_path(staging, path)
            _ensure_internal_tree_permissions(path)
        except Exception:
            if backup.exists() and not path.exists():
                _replace_path(backup, path)
            raise
        _remove_internal_path(backup)
        return rows
    except Exception:
        if connection is not None:
            connection.close()
        _remove_internal_path(staging)
        if backup.exists() and not path.exists():
            _replace_path(backup, path)
        raise


def _duckdb_sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _duckdb_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _limit_dataframe_rows(
    df: pl.DataFrame,
    *,
    max_rows: int | None,
    sort_by: Sequence[str] | None,
    descending: bool,
) -> pl.DataFrame:
    if max_rows is None or max_rows <= 0 or df.height <= max_rows:
        return df
    sort_columns = [column for column in (sort_by or []) if column in df.columns]
    limited = df
    if sort_columns:
        try:
            limited = limited.sort(sort_columns, descending=descending)
        except Exception:
            limited = df
    return limited.head(max_rows)


def validate_market_bars(records: Sequence[MarketBar | dict]) -> list[MarketBar]:
    validated = [
        record if isinstance(record, MarketBar) else MarketBar(**record) for record in records
    ]
    seen_keys: set[tuple[str, str, str, datetime]] = set()
    duplicate_keys: list[tuple[str, str, str, datetime]] = []

    for record in validated:
        key = (record.venue, record.symbol, record.timeframe, record.ts)
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)

    if duplicate_keys:
        rendered = ", ".join(
            f"{venue}/{symbol}/{timeframe}/{ts.isoformat()}"
            for venue, symbol, timeframe, ts in duplicate_keys
        )
        raise ValueError(f"duplicate market_bar primary key: {rendered}")

    return validated


def market_bars_to_polars(records: Sequence[MarketBar | dict]) -> pl.DataFrame:
    validated = validate_market_bars(records)
    rows = [record.model_dump() for record in validated]
    return pl.DataFrame(rows, schema=MARKET_BAR_SCHEMA, orient="row")


def write_market_bars(lake_root: str | Path, records: Sequence[MarketBar | dict]) -> int:
    dataset_path = Path(lake_root) / MARKET_BAR_DATASET
    if not records:
        return count_parquet_rows(dataset_path)
    new_df = market_bars_to_polars(records)
    if new_df.is_empty():
        return count_parquet_rows(dataset_path)
    rows = upsert_parquet_dataset(new_df, dataset_path, key_columns=MARKET_BAR_PRIMARY_KEY)
    try:
        _write_market_bar_health(Path(lake_root), new_df, row_count=rows)
    except Exception:
        logger.exception("failed to update market_bar health metadata")
    return rows


def _write_market_bar_health(lake_root: Path, new_df: pl.DataFrame, *, row_count: int) -> None:
    new_latest = _frame_latest_datetime(new_df, "ts")
    if new_latest is None:
        return
    path = lake_root / MARKET_BAR_HEALTH_DATASET
    existing = read_parquet_dataset(path)
    existing_latest = _frame_latest_datetime(existing, "latest_ts")
    new_timeframe = _frame_latest_value(new_df, "timeframe", sort_column="ts")
    existing_timeframe = _frame_latest_value(existing, "latest_timeframe", sort_column="latest_ts")
    latest = max([value for value in [existing_latest, new_latest] if value is not None])
    if existing_latest is not None and existing_latest >= new_latest:
        timeframe = existing_timeframe or DEFAULT_MARKET_BAR_TIMEFRAME
        latest_close = _frame_latest_datetime(existing, "latest_close_ts") or market_bar_close_ts(
            latest,
            timeframe,
        )
    else:
        timeframe = new_timeframe or DEFAULT_MARKET_BAR_TIMEFRAME
        latest_close = market_bar_close_ts(latest, timeframe)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "dataset": "market_bar",
                    "row_count": row_count,
                    "latest_ts": latest,
                    "latest_timeframe": timeframe,
                    "latest_close_ts": latest_close,
                    "updated_at": datetime.now(UTC),
                }
            ]
        ),
        path,
    )


def _frame_latest_datetime(frame: pl.DataFrame, column: str) -> datetime | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    try:
        value = frame.select(
            pl.col(column)
            .cast(pl.Utf8)
            .str.to_datetime(time_zone="UTC", strict=False)
            .max()
            .alias(column)
        ).item()
    except Exception:
        return None
    if not isinstance(value, datetime):
        return None
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def _frame_latest_value(
    frame: pl.DataFrame,
    column: str,
    *,
    sort_column: str,
) -> str | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    try:
        scoped = frame
        if sort_column in scoped.columns:
            scoped = scoped.sort(sort_column)
        value = scoped.tail(1).item(0, column)
    except Exception:
        return None
    text = str(value or "").strip()
    return text or None


def read_market_bars(
    lake_root: str | Path,
    venue: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> list[MarketBar]:
    start_utc = require_utc(start)
    end_utc = require_utc(end)
    if end_utc < start_utc:
        raise ValueError("end must be greater than or equal to start")
    normalized_venue = _normalize_market_venue(venue)

    files = _parquet_files(Path(lake_root) / MARKET_BAR_DATASET)
    if not files:
        return []

    lazy = _scan_parquet_files(files)
    try:
        schema = lazy.collect_schema()
    except Exception:
        df = read_parquet_dataset(Path(lake_root) / MARKET_BAR_DATASET)
        if df.is_empty():
            return []
        normalized = _normalize_market_bar_frame(df)
        filtered = (
            normalized.filter(
                (pl.col("venue").cast(pl.Utf8).str.to_lowercase() == normalized_venue)
                & (pl.col("symbol") == normalize_symbol(symbol))
                & (pl.col("timeframe") == timeframe)
                & (pl.col("ts") >= start_utc)
                & (pl.col("ts") <= end_utc)
            )
            .sort("ts")
            .select(list(MARKET_BAR_SCHEMA))
        )
        return validate_market_bars(filtered.to_dicts())

    filtered = (
        _normalize_market_bar_lazy_frame(lazy, schema)
        .filter(
            (pl.col("venue").cast(pl.Utf8).str.to_lowercase() == normalized_venue)
            & (pl.col("symbol") == normalize_symbol(symbol))
            & (pl.col("timeframe") == timeframe)
            & (pl.col("ts") >= start_utc)
            & (pl.col("ts") <= end_utc)
        )
        .sort("ts")
        .select(list(MARKET_BAR_SCHEMA))
    )
    return validate_market_bars(filtered.collect().to_dicts())


def _normalize_market_venue(value: str) -> str:
    return str(value or "").strip().lower()


def read_parquet_lazy(path: str | Path) -> pl.LazyFrame:
    files = _parquet_files(path)
    if files:
        return _scan_parquet_files(files)
    return pl.scan_parquet(str(path))


def scan_parquet_with_duckdb(path: str | Path) -> duckdb.DuckDBPyRelation:
    files = _parquet_files(path)
    if not files:
        raise FileNotFoundError(f"No Parquet files found under dataset path: {path}")
    connection = duckdb.connect(database=":memory:", read_only=False)
    return connection.read_parquet([str(file) for file in files])


def query_dataset_sql(lake_root: str | Path, dataset_name: str, sql: str) -> pl.DataFrame:
    dataset_path = Path(lake_root) / dataset_name
    files = _parquet_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"No Parquet files found for dataset: {dataset_name}")

    query = sql.strip()
    if ";" in query.rstrip(";"):
        raise ValueError("query_dataset_sql accepts a single read-only SELECT statement")
    query = query.rstrip(";")
    if not query.lower().startswith("select"):
        raise ValueError("query_dataset_sql only accepts read-only SELECT statements")

    connection = duckdb.connect(database=":memory:", read_only=False)
    connection.read_parquet([str(path) for path in files]).create_view("dataset")
    return pl.from_arrow(connection.execute(query).to_arrow_table())


def _parquet_files(dataset_path: str | Path) -> list[Path]:
    return [path for path in _all_parquet_files(dataset_path) if _is_valid_parquet_file(path)]


def _direct_parquet_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path)
    if not path.exists() or not path.is_dir():
        return []
    return sorted(
        candidate
        for candidate in path.glob("*.parquet")
        if not _is_internal_lake_file(candidate) and _is_valid_parquet_file(candidate)
    )


def _direct_compaction_source_files(
    dataset_path: str | Path,
    *,
    include_existing_compact_files: bool = False,
) -> list[Path]:
    """Return direct append files that still need compaction.

    Hot append datasets accumulate small ``part_``/``api_`` style files between
    compaction runs. Previously direct compaction also re-read existing
    ``compact_*.parquet`` outputs on every run, causing large historical tables
    to be decompressed repeatedly and pushing production memory into swap.
    """

    if include_existing_compact_files:
        return _direct_parquet_files(dataset_path)
    return [
        file
        for file in _direct_parquet_files(dataset_path)
        if _is_direct_compaction_source_file(file)
    ]


def _is_direct_compaction_source_file(path: Path) -> bool:
    name = path.name
    return not (name.startswith("compact_") or name == "data.parquet")


def _all_parquet_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path)
    if path.is_file() and path.suffix == ".parquet":
        return [] if _is_internal_lake_file(path) else [path]
    if not path.exists():
        return []
    return sorted(
        candidate for candidate in path.rglob("*.parquet") if not _is_internal_lake_file(candidate)
    )


def _bad_partition_files(
    dataset_path: Path,
    partition_columns: Sequence[str],
    bad_values: Sequence[str],
) -> list[Path]:
    if not partition_columns:
        return []
    bad_parts = {
        f"{column}={bad_value}" for column in partition_columns for bad_value in bad_values
    }
    return [
        path
        for path in _parquet_files(dataset_path)
        if any(part in bad_parts for part in path.relative_to(dataset_path).parts)
    ]


def _is_internal_lake_file(path: Path) -> bool:
    return (
        any(part == "._tmp" or part.startswith("__") for part in path.parts)
        or path.name.startswith(".")
        or path.name.endswith(".tmp.parquet")
    )


def _read_parquet_files(
    files: Sequence[Path],
    *,
    schema_union: bool = False,
) -> pl.DataFrame:
    try:
        return _scan_parquet_files(files, schema_union=schema_union).collect()
    except pl.exceptions.SchemaError:
        frames = [pl.read_parquet(path) for path in files]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    except Exception as exc:
        if not _is_parquet_schema_compat_error(exc):
            raise
        frames = [pl.read_parquet(path) for path in files]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    except TypeError:
        return pl.read_parquet([str(path) for path in files])


def _scan_parquet_files(
    files: Sequence[Path],
    *,
    schema_union: bool = False,
) -> pl.LazyFrame:
    paths = [str(path) for path in files]
    try:
        schema = _parquet_schema_union(files) if schema_union else {}
        return pl.scan_parquet(
            paths,
            hive_partitioning=False,
            schema=schema or None,
            missing_columns="insert",
            extra_columns="ignore",
        )
    except TypeError:
        return pl.scan_parquet(paths)


def _parquet_schema_union(files: Sequence[Path]) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {}
    for path in files:
        try:
            file_schema = pl.read_parquet_schema(path)
        except Exception:
            continue
        for name, dtype in file_schema.items():
            schema.setdefault(str(name), dtype)
    return schema


def _is_parquet_schema_compat_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in [
            "extra column",
            "outside of expected schema",
            "schema",
            "not found",
            "did not match",
        ]
    )


def _is_valid_parquet_file(path: Path) -> bool:
    try:
        readable_path = _replaceable_path(path)
        if os.stat(readable_path).st_size < MIN_PARQUET_SIZE_BYTES:
            logger.warning("Ignoring undersized parquet file: %s", path)
            return False
        with open(readable_path, "rb") as file:
            header = file.read(len(PARQUET_MAGIC))
            file.seek(-len(PARQUET_MAGIC), os.SEEK_END)
            footer = file.read(len(PARQUET_MAGIC))
        if header != PARQUET_MAGIC or footer != PARQUET_MAGIC:
            logger.warning("Ignoring invalid parquet file header/footer: %s", path)
            return False
        return True
    except OSError as exc:
        logger.warning("Ignoring unreadable parquet file %s: %s", path, exc)
        return False


def _replace_path(source: Path, target: Path) -> None:
    os.replace(_replaceable_path(source), _replaceable_path(target))


def _dataset_temp_file(dataset_path: Path) -> Path:
    file_name = f"{uuid.uuid4().hex}.tmp.parquet"
    errors: list[str] = []
    for temp_dir in _dataset_temp_dirs(dataset_path):
        temp_file = temp_dir / file_name
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            _ensure_lake_dir_permissions(temp_dir)
            fd = os.open(
                _replaceable_path(temp_file),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.close(fd)
            temp_file.unlink()
            return temp_file
        except OSError as exc:
            errors.append(f"{temp_dir}: {exc}")
            try:
                if temp_file.exists() and temp_file.stat().st_size == 0:
                    temp_file.unlink()
            except OSError:
                pass
    rendered = "; ".join(errors) if errors else "no temp directory candidates"
    raise PermissionError(f"unable to create dataset temp file for {dataset_path}: {rendered}")


def _dataset_temp_dirs(dataset_path: Path) -> list[Path]:
    return [
        dataset_path / "._tmp",
        dataset_path.parent / f".{dataset_path.name}._tmp",
    ]


def _ensure_lake_dir_permissions(path: Path) -> None:
    """Keep lake-created directories group-writable for service/user handoff."""

    if os.name == "nt":
        return
    try:
        current_mode = stat.S_IMODE(path.stat().st_mode)
        desired_mode = current_mode | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_ISGID
        if desired_mode != current_mode:
            os.chmod(_replaceable_path(path), desired_mode)
    except OSError as exc:
        logger.debug("failed to adjust lake directory permissions for %s: %s", path, exc)


def _ensure_internal_tree_permissions(path: Path) -> None:
    if os.name == "nt" or not path.exists():
        return
    for directory in [path, *(item for item in path.rglob("*") if item.is_dir())]:
        _ensure_lake_dir_permissions(directory)


def _make_path_removable(path: Path) -> None:
    if os.name == "nt" or not path.exists():
        return
    targets = [path]
    if path.is_dir():
        targets.extend(path.rglob("*"))
    for target in targets:
        try:
            mode = stat.S_IMODE(target.stat().st_mode)
            writable_mode = mode | stat.S_IWUSR
            if target.is_dir():
                writable_mode |= stat.S_IXUSR | stat.S_IWGRP | stat.S_IXGRP
            if writable_mode != mode:
                os.chmod(_replaceable_path(target), writable_mode)
        except OSError:
            continue


def _replaceable_path(path: Path) -> str | Path:
    if os.name != "nt":
        return path
    resolved = str(Path(path).resolve())
    if resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def _sort_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    preferred = [
        "source_path",
        "run_id",
        "venue",
        "symbol",
        "timeframe",
        "ts",
        "cost_day",
        "bucket_index",
        "dataset",
        "ingest_ts",
    ]
    sort_columns = [column for column in preferred if column in df.columns]
    if not sort_columns:
        sort_columns = sorted(df.columns)
    try:
        return df.sort(sort_columns)
    except Exception:
        return df


def _frame_latest_snapshot_text(frame: pl.DataFrame, columns: Sequence[str]) -> str:
    if frame.is_empty():
        return ""
    for column in columns:
        if column not in frame.columns:
            continue
        values = [
            _snapshot_text(value)
            for value in frame.get_column(column).drop_nulls().to_list()
        ]
        values = [value for value in values if value]
        if values:
            return max(values)
    return ""


def _frame_first_snapshot_text(frame: pl.DataFrame, column: str) -> str:
    if frame.is_empty() or column not in frame.columns:
        return ""
    for value in frame.get_column(column).drop_nulls().to_list():
        text = _snapshot_text(value)
        if text:
            return text
    return ""


def _snapshot_source_sha(
    dataset_name: str,
    frame: pl.DataFrame,
    *,
    generated_at: str,
    expires_at: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(dataset_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(generated_at.encode("utf-8"))
    digest.update(b"\0")
    digest.update(expires_at.encode("utf-8"))
    digest.update(b"\0")
    columns = sorted(str(column) for column in frame.columns)
    digest.update(json.dumps(columns, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(frame.height).encode("ascii"))
    if frame.is_empty():
        return digest.hexdigest()
    if columns:
        # Wide V5 evidence frames can contain millions of scalar values. Turning
        # every value into Python objects for a metadata checksum consumed several
        # gigabytes, so hash fixed-width row fingerprints instead.
        row_hashes = frame.select(columns).hash_rows(
            seed=0,
            seed_1=1,
            seed_2=2,
            seed_3=3,
        ).sort()
        for buffer in row_hashes.to_arrow().buffers():
            if buffer is not None:
                digest.update(memoryview(buffer))
    return digest.hexdigest()


def _snapshot_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        timestamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan", "nat"} else text


def _snapshot_json(value: Any) -> str:
    if isinstance(value, datetime):
        return _snapshot_text(value)
    return str(value)


def _partition_columns(partition_by: str | Sequence[str] | None) -> list[str]:
    if partition_by is None:
        return []
    if isinstance(partition_by, str):
        return [partition_by]
    return [str(column) for column in partition_by]


def _partitioned_chunks(
    df: pl.DataFrame,
    partition_columns: list[str],
) -> list[tuple[dict[str, Any], pl.DataFrame]]:
    available = [column for column in partition_columns if column in df.columns]
    if not available:
        return [({}, df)]
    chunks: list[tuple[dict[str, Any], pl.DataFrame]] = []
    for key, group in df.group_by(available, maintain_order=True):
        values = key if isinstance(key, tuple) else (key,)
        chunks.append((dict(zip(available, values, strict=False)), group))
    return chunks


def _repair_partition_frame(df: pl.DataFrame, partition_columns: Sequence[str]) -> pl.DataFrame:
    repaired = _fill_event_timestamp_columns(df)
    for column in partition_columns:
        if column == "day":
            repaired = _fill_day_column(repaired)
        elif column == "symbol":
            repaired = _fill_symbol_column(repaired)
        else:
            repaired = _fill_text_column(repaired, column, fallback="unknown")
    return repaired


def _fill_event_timestamp_columns(df: pl.DataFrame) -> pl.DataFrame:
    if "ts" not in df.columns:
        return df
    candidates = [
        column
        for column in ["ts", "received_at", "ingest_ts", "created_at", "updated_at"]
        if column in df.columns
    ]
    if not candidates:
        return df
    expressions = [_non_empty_string_expr(column) for column in candidates]
    return df.with_columns(pl.coalesce(expressions).alias("ts"))


def _fill_day_column(df: pl.DataFrame) -> pl.DataFrame:
    candidates = []
    if "day" in df.columns:
        candidates.append(_non_empty_string_expr("day"))
    candidates.extend(
        _non_empty_string_expr(column).str.slice(0, 10)
        for column in ["ts", "received_at", "ingest_ts", "created_at", "updated_at"]
        if column in df.columns
    )
    if not candidates:
        return df.with_columns(pl.lit("unknown").alias("day"))
    return df.with_columns(
        pl.coalesce(candidates + [pl.lit("unknown", dtype=pl.Utf8)]).alias("day")
    )


def _fill_symbol_column(df: pl.DataFrame) -> pl.DataFrame:
    candidates = []
    if "symbol" in df.columns:
        candidates.append(_non_empty_string_expr("symbol"))
    if "inst_id" in df.columns:
        candidates.append(_non_empty_string_expr("inst_id"))
    if not candidates:
        return df.with_columns(pl.lit("unknown").alias("symbol"))
    return df.with_columns(
        pl.coalesce(candidates + [pl.lit("unknown", dtype=pl.Utf8)])
        .map_elements(normalize_symbol, return_dtype=pl.Utf8)
        .alias("symbol")
    )


def _fill_text_column(df: pl.DataFrame, column: str, *, fallback: str) -> pl.DataFrame:
    if column not in df.columns:
        return df.with_columns(pl.lit(fallback).alias(column))
    return df.with_columns(
        pl.coalesce([_non_empty_string_expr(column), pl.lit(fallback, dtype=pl.Utf8)]).alias(column)
    )


def _non_empty_string_expr(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Utf8).str.strip_chars()
    return pl.when(value.is_not_null() & (value != "")).then(value).otherwise(None)


def _remove_empty_directories(directories: set[Path], *, stop_at: Path) -> None:
    stop = stop_at.resolve()
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        current = directory
        while current != stop and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def _remove_empty_bad_partition_directories(
    dataset_path: Path,
    partition_columns: Sequence[str],
    bad_values: Sequence[str],
) -> None:
    if not dataset_path.exists():
        return
    bad_parts = {
        f"{column}={bad_value}" for column in partition_columns for bad_value in bad_values
    }
    candidates = [
        path
        for path in dataset_path.rglob("*")
        if path.is_dir() and any(part in bad_parts for part in path.relative_to(dataset_path).parts)
    ]
    _remove_empty_directories(set(candidates), stop_at=dataset_path)


def _move_repaired_staging_files(staging: Path, dataset_path: Path) -> int:
    moved = 0
    if not staging.exists():
        return moved
    for repaired_file in sorted(staging.rglob("*.parquet")):
        if not _is_valid_parquet_file(repaired_file):
            continue
        target = dataset_path / repaired_file.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        _ensure_lake_dir_permissions(target.parent)
        _replace_path(repaired_file, target)
        moved += 1
    return moved


def _chunks(values: Sequence[Path], size: int) -> list[Sequence[Path]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _parquet_file_batches(
    files: Sequence[Path],
    *,
    max_source_files_per_batch: int,
    max_source_batch_bytes: int,
) -> list[list[Path]]:
    max_files = max(max_source_files_per_batch, 1)
    max_bytes = max(max_source_batch_bytes, 0)
    if max_bytes <= 0:
        return [list(chunk) for chunk in _chunks(files, max_files)]

    batches: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for file_path in files:
        file_size = _safe_file_size(file_path)
        if current and (
            len(current) >= max_files or current_bytes + file_size > max_bytes
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(file_path)
        current_bytes += file_size
    if current:
        batches.append(current)
    return batches


def _resolve_max_source_batch_bytes(value: int | None) -> int:
    if value is not None and value > 0:
        return value
    return _int_env("QUANT_LAB_COMPACT_MAX_SOURCE_BATCH_BYTES", 128 * 1024 * 1024)


def _safe_file_size(path: Path) -> int:
    try:
        return max(path.stat().st_size, 1)
    except OSError:
        return 1


def _partition_dir(dataset_path: Path, partition_values: dict[str, Any]) -> Path:
    path = dataset_path
    for column, value in partition_values.items():
        safe_value = _safe_partition_value(value)
        path = path / f"{column}={safe_value}"
    return path


def _safe_partition_value(value: Any) -> str:
    if value is None:
        return "__null__"
    text = str(value).strip()
    if not text:
        return "__empty__"
    safe_chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_", ".", ":", "T", "Z"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def _append_file_name(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}_{os.getpid()}_{time.monotonic_ns()}.parquet"


def _int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _normalize_market_bar_frame(df: pl.DataFrame) -> pl.DataFrame:
    normalized = df
    if "quote_volume" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit(None, dtype=pl.Float64).alias("quote_volume"))
    if "is_closed" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit(True).alias("is_closed"))

    return normalized.with_columns(
        [
            pl.col("symbol").map_elements(normalize_symbol, return_dtype=pl.Utf8),
            _datetime_column(normalized, "ts"),
            _datetime_column(normalized, "ingest_ts"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
            pl.col("is_closed").cast(pl.Boolean),
        ]
    )


def _normalize_market_bar_lazy_frame(lazy: pl.LazyFrame, schema: Any) -> pl.LazyFrame:
    columns = set(schema.names()) if hasattr(schema, "names") else set(schema)
    normalized = lazy
    if "quote_volume" not in columns:
        normalized = normalized.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("quote_volume")
        )
    if "is_closed" not in columns:
        normalized = normalized.with_columns(pl.lit(True).alias("is_closed"))

    return normalized.with_columns(
        [
            pl.col("symbol").map_elements(normalize_symbol, return_dtype=pl.Utf8),
            _lazy_datetime_column(schema, "ts"),
            _lazy_datetime_column(schema, "ingest_ts"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
            pl.col("is_closed").cast(pl.Boolean),
        ]
    )


def _datetime_column(df: pl.DataFrame, column: str) -> pl.Expr:
    expression = pl.col(column)
    if df.schema.get(column) == pl.String:
        return expression.str.to_datetime(time_zone="UTC", strict=False).alias(column)
    return expression.cast(pl.Datetime(time_zone="UTC")).alias(column)


def _lazy_datetime_column(schema: Any, column: str) -> pl.Expr:
    expression = pl.col(column)
    if schema.get(column) == pl.String:
        return expression.str.to_datetime(time_zone="UTC", strict=False).alias(column)
    return expression.cast(pl.Datetime(time_zone="UTC"), strict=False).alias(column)


def _remove_existing_dataset(path: Path) -> None:
    if path.is_file():
        path.unlink()
        return
    if path.is_dir():
        for parquet_file in path.rglob("*.parquet"):
            try:
                parquet_file.unlink()
            except FileNotFoundError:
                pass
        for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _remove_internal_path(path: Path) -> None:
    """Remove an internal staging/backup path without touching user datasets."""

    if not path.exists():
        return
    if not _is_internal_lake_file(path):
        raise ValueError(f"refusing to remove non-internal lake path: {path}")
    try:
        _make_path_removable(path)
        if path.is_dir():
            shutil.rmtree(_replaceable_path(path))
        else:
            path.unlink()
    except OSError as exc:
        logger.warning("failed to remove internal lake path %s: %s", path, exc)


def _remove_stale_parquet_files(path: Path, keep: Path) -> None:
    for parquet_file in _all_parquet_files(path):
        if parquet_file == keep:
            continue
        parquet_file.unlink()
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()


def _copy_preserved_dataset_files(
    source_root: Path,
    destination_root: Path,
    file_names: Sequence[str],
) -> None:
    for file_name in file_names:
        relative = Path(file_name)
        if (
            not file_name
            or relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != file_name
            or file_name in {".", ".."}
        ):
            raise ValueError(f"invalid preserved dataset file name: {file_name!r}")
        source = source_root / relative
        if source.is_symlink():
            raise ValueError(f"preserved dataset file must not be a symlink: {file_name}")
        if source.is_file():
            shutil.copy2(source, destination_root / relative)


@contextmanager
def _dataset_lock(dataset_path: Path, *, timeout_seconds: float = 30.0) -> object:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_lake_dir_permissions(dataset_path.parent)
    lock_path = dataset_path.parent / f".{dataset_path.name}.lock"
    process_lock = _process_lock(lock_path)
    process_lock.acquire()
    start = time.monotonic()
    fd: int | None = None
    try:
        while fd is None:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("ascii"))
            except FileExistsError:
                if _lock_is_stale(lock_path):
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() - start > timeout_seconds:
                    raise TimeoutError(
                        f"timed out waiting for dataset lock: {dataset_path}"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    finally:
        process_lock.release()


def _lock_is_stale(lock_path: Path, *, stale_seconds: float = 600.0) -> bool:
    try:
        stat = lock_path.stat()
    except FileNotFoundError:
        return False
    age_seconds = time.time() - stat.st_mtime
    try:
        payload = lock_path.read_text(encoding="ascii").strip()
    except OSError:
        payload = ""
    if not payload:
        return age_seconds > 5.0
    try:
        pid = int(payload)
    except ValueError:
        return age_seconds > 5.0
    if pid <= 0:
        return age_seconds > 5.0
    if os.name == "nt":
        if pid == os.getpid():
            return age_seconds > stale_seconds
        return (not _windows_pid_exists(pid)) or age_seconds > stale_seconds
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return True
    return age_seconds > stale_seconds


def _process_lock(lock_path: Path) -> threading.Lock:
    key = str(lock_path.resolve())
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _windows_pid_exists(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return str(pid) in result.stdout
