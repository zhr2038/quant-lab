import logging
import os
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
from quant_lab.symbols import normalize_symbol

MARKET_BAR_DATASET = Path("silver") / "market_bar"
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


@dataclass(frozen=True)
class CompactParquetResult:
    dataset_path: str
    source_file_count: int
    output_file_count: int
    rows: int
    partition_by: list[str]
    target_rows_per_file: int
    max_source_files_per_batch: int


def write_parquet_dataset(
    df: pl.DataFrame,
    dataset_path: str | Path,
    partition_by: str | Sequence[str] | None = None,
) -> Path:
    path = Path(dataset_path)
    with _dataset_lock(path):
        return _write_parquet_dataset_unlocked(df, path, partition_by=partition_by)


def _write_parquet_dataset_unlocked(
    df: pl.DataFrame,
    dataset_path: str | Path,
    partition_by: str | Sequence[str] | None = None,
) -> Path:
    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_df = _sort_dataframe(df)

    if partition_by:
        _remove_existing_dataset(path)
        path.mkdir(parents=True, exist_ok=True)
        sorted_df.write_parquet(path, partition_by=partition_by, mkdir=True)
        return path

    if path.is_file():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    final_file = path / "data.parquet"
    temp_file = _dataset_temp_file(path)
    try:
        sorted_df.write_parquet(temp_file)
        _replace_path(temp_file, final_file)
        _remove_stale_parquet_files(path, keep=final_file)
    finally:
        if temp_file.exists():
            temp_file.unlink()
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
) -> AppendParquetResult:
    path = Path(dataset_path)
    path.mkdir(parents=True, exist_ok=True)
    frame = _sort_dataframe(df)
    partition_columns = _partition_columns(partition_by)
    chunks = _partitioned_chunks(frame, partition_columns)
    rows_written = 0
    file_count = 0
    max_rows = max(int(target_rows_per_file or frame.height), 1)
    for partition_values, partition_frame in chunks:
        partition_dir = _partition_dir(path, partition_values)
        partition_dir.mkdir(parents=True, exist_ok=True)
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
    return AppendParquetResult(str(path), rows_written, file_count, partition_columns)


def compact_parquet_dataset(
    dataset_path: str | Path,
    *,
    partition_by: str | Sequence[str] | None = None,
    target_rows_per_file: int = 250_000,
    max_source_files_per_batch: int = 5_000,
) -> CompactParquetResult:
    """Rewrite a dataset into fewer deterministic partitioned parquet files."""

    path = Path(dataset_path)
    partition_columns = _partition_columns(partition_by)
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
            )
        staging = path.parent / f"__{path.name}_compact_{uuid.uuid4().hex}"
        output_file_count = 0
        rows = 0
        try:
            for batch_files in _chunks(files, max(max_source_files_per_batch, 1)):
                frame = _read_parquet_files(batch_files)
                if frame.is_empty():
                    continue
                result = _append_parquet_dataset_unlocked(
                    frame,
                    staging,
                    partition_by=partition_columns,
                    target_rows_per_file=target_rows_per_file,
                    file_prefix="compact",
                )
                output_file_count += result.file_count
                rows += result.rows_written
            if rows == 0:
                _remove_existing_dataset(path)
                path.mkdir(parents=True, exist_ok=True)
                return CompactParquetResult(
                    dataset_path=str(path),
                    source_file_count=source_file_count,
                    output_file_count=0,
                    rows=0,
                    partition_by=partition_columns,
                    target_rows_per_file=target_rows_per_file,
                    max_source_files_per_batch=max_source_files_per_batch,
                )
            backup = path.parent / f"__{path.name}_backup_{uuid.uuid4().hex}"
            if path.exists():
                _replace_path(path, backup)
            _replace_path(staging, path)
            _remove_existing_dataset(backup)
            return CompactParquetResult(
                dataset_path=str(path),
                source_file_count=source_file_count,
                output_file_count=output_file_count,
                rows=rows,
                partition_by=partition_columns,
                target_rows_per_file=target_rows_per_file,
                max_source_files_per_batch=max_source_files_per_batch,
            )
        except Exception:
            _remove_existing_dataset(staging)
            raise


def read_parquet_dataset(dataset_path: str | Path) -> pl.DataFrame:
    files = _parquet_files(dataset_path)
    if not files:
        return pl.DataFrame()
    return _read_parquet_files(files)


def invalid_parquet_files(dataset_path: str | Path) -> list[Path]:
    return [path for path in _all_parquet_files(dataset_path) if not _is_valid_parquet_file(path)]


def upsert_parquet_dataset(
    df: pl.DataFrame,
    dataset_path: str | Path,
    key_columns: Sequence[str],
) -> int:
    path = Path(dataset_path)
    with _dataset_lock(path):
        existing_df = read_parquet_dataset(path)
        frames = [frame for frame in [existing_df, df] if not frame.is_empty()]
        combined = pl.concat(frames, how="diagonal_relaxed") if frames else df
        if not combined.is_empty():
            available_keys = [column for column in key_columns if column in combined.columns]
            if available_keys:
                combined = combined.unique(subset=available_keys, keep="last", maintain_order=True)
        _write_parquet_dataset_unlocked(combined, path)
        return combined.height


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
        return read_parquet_dataset(dataset_path).height
    new_df = market_bars_to_polars(records)
    if new_df.is_empty():
        return read_parquet_dataset(dataset_path).height
    return upsert_parquet_dataset(new_df, dataset_path, key_columns=MARKET_BAR_PRIMARY_KEY)


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

    dataset_path = Path(lake_root) / MARKET_BAR_DATASET
    df = read_parquet_dataset(dataset_path)
    if df.is_empty():
        return []

    normalized = _normalize_market_bar_frame(df)
    filtered = (
        normalized.filter(
            (pl.col("venue") == venue)
            & (pl.col("symbol") == normalize_symbol(symbol))
            & (pl.col("timeframe") == timeframe)
            & (pl.col("ts") >= start_utc)
            & (pl.col("ts") <= end_utc)
        )
        .sort("ts")
        .select(list(MARKET_BAR_SCHEMA))
    )
    return validate_market_bars(filtered.to_dicts())


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


def _all_parquet_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path)
    if path.is_file() and path.suffix == ".parquet":
        return [] if _is_internal_lake_file(path) else [path]
    if not path.exists():
        return []
    return sorted(
        candidate for candidate in path.rglob("*.parquet") if not _is_internal_lake_file(candidate)
    )


def _is_internal_lake_file(path: Path) -> bool:
    return (
        any(part == "._tmp" or part.startswith("__") for part in path.parts)
        or path.name.startswith(".")
        or path.name.endswith(".tmp.parquet")
    )


def _read_parquet_files(files: Sequence[Path]) -> pl.DataFrame:
    try:
        return _scan_parquet_files(files).collect()
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


def _scan_parquet_files(files: Sequence[Path]) -> pl.LazyFrame:
    try:
        return pl.scan_parquet(
            [str(path) for path in files],
            hive_partitioning=False,
            missing_columns="insert",
            extra_columns="ignore",
        )
    except TypeError:
        return pl.scan_parquet([str(path) for path in files])


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
        if path.stat().st_size < MIN_PARQUET_SIZE_BYTES:
            logger.warning("Ignoring undersized parquet file: %s", path)
            return False
        with path.open("rb") as file:
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
    temp_dir = dataset_path / "._tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f"{uuid.uuid4().hex}.tmp.parquet"


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


def _chunks(values: Sequence[Path], size: int) -> list[Sequence[Path]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


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


def _datetime_column(df: pl.DataFrame, column: str) -> pl.Expr:
    expression = pl.col(column)
    if df.schema.get(column) == pl.String:
        return expression.str.to_datetime(time_zone="UTC", strict=False).alias(column)
    return expression.cast(pl.Datetime(time_zone="UTC")).alias(column)


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


def _remove_stale_parquet_files(path: Path, keep: Path) -> None:
    for parquet_file in _all_parquet_files(path):
        if parquet_file == keep:
            continue
        parquet_file.unlink()
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()


@contextmanager
def _dataset_lock(dataset_path: Path, *, timeout_seconds: float = 30.0) -> object:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
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
