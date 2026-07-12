from __future__ import annotations

import gc
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.file_index import files_fully_within_time_range, old_files_for_dataset
from quant_lab.data.lake import read_parquet_lazy, upsert_parquet_dataset

HF_DATASETS = {
    "okx_public_ws": Path("bronze") / "okx_public_ws",
    "trade_print": Path("silver") / "trade_print",
    "orderbook_snapshot": Path("silver") / "orderbook_snapshot",
}
ORDERBOOK_SPREAD_ROLLUP = Path("silver") / "orderbook_spread_1m"
TRADE_ACTIVITY_ROLLUP = Path("silver") / "trade_activity_1m"
ORDERBOOK_SPREAD_ROLLUP_KEYS = ["symbol", "channel", "minute_ts"]
TRADE_ACTIVITY_ROLLUP_KEYS = ["symbol", "minute_ts"]
TOP_BOOK_LEVEL_RE = r'^\s*\[\s*\[\s*"?([^",\]\s]+)"?\s*,\s*"?([^",\]\s]+)"?'
MAX_ARCHIVED_FILE_SAMPLES = 50
SILVER_ROLLUP_COVERAGE = {
    "trade_print": (TRADE_ACTIVITY_ROLLUP, "minute_ts"),
    "orderbook_snapshot": (ORDERBOOK_SPREAD_ROLLUP, "minute_ts"),
}


@dataclass
class MarketDataCompactionResult:
    lake_root: str
    dry_run: bool
    started_at: datetime
    finished_at: datetime | None = None
    archived_files: list[str] = field(default_factory=list)
    archived_file_count: int = 0
    archived_files_truncated: bool = False
    archived_bytes: int = 0
    archived_by_dataset: dict[str, int] = field(default_factory=dict)
    rollup_rows: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_root": self.lake_root,
            "dry_run": self.dry_run,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "archived_files": self.archived_files,
            "archived_file_count": self.archived_file_count,
            "archived_files_truncated": self.archived_files_truncated,
            "archived_bytes": self.archived_bytes,
            "archived_by_dataset": self.archived_by_dataset,
            "rollup_rows": self.rollup_rows,
            "warnings": self.warnings,
        }


def compact_market_data(
    lake_root: str | Path,
    *,
    hot_hours: int = 24,
    dry_run: bool = True,
    now: datetime | None = None,
    rollup_lookback_hours: int | None = None,
) -> MarketDataCompactionResult:
    current = now or datetime.now(UTC)
    root = Path(lake_root)
    result = MarketDataCompactionResult(lake_root=str(root), dry_run=dry_run, started_at=current)
    _build_and_write_rollups(
        root,
        dry_run=dry_run,
        result=result,
        since=_rollup_since(current, rollup_lookback_hours),
    )
    _archive_old_high_frequency_files(
        root, hot_hours=hot_hours, dry_run=dry_run, now=current, result=result
    )
    result.finished_at = datetime.now(UTC)
    return result


def build_market_data_1m_rollups(
    lake_root: str | Path,
    *,
    dry_run: bool = True,
    now: datetime | None = None,
    lookback_hours: int | None = None,
) -> MarketDataCompactionResult:
    current = now or datetime.now(UTC)
    root = Path(lake_root)
    result = MarketDataCompactionResult(lake_root=str(root), dry_run=dry_run, started_at=current)
    _build_and_write_rollups(
        root,
        dry_run=dry_run,
        result=result,
        since=_rollup_since(current, lookback_hours),
    )
    result.finished_at = datetime.now(UTC)
    return result


def _build_and_write_rollups(
    root: Path,
    *,
    dry_run: bool,
    result: MarketDataCompactionResult,
    since: datetime | None,
) -> None:
    trade_rollup = build_trade_activity_1m_rollup(root, since=since, warnings=result.warnings)
    if not dry_run:
        if not trade_rollup.is_empty():
            upsert_parquet_dataset(
                trade_rollup,
                root / TRADE_ACTIVITY_ROLLUP,
                key_columns=TRADE_ACTIVITY_ROLLUP_KEYS,
            )
    result.rollup_rows["trade_activity_1m"] = trade_rollup.height
    del trade_rollup
    gc.collect()

    orderbook_rollup = build_orderbook_spread_1m_rollup(
        root,
        since=since,
        warnings=result.warnings,
    )
    if not dry_run:
        if not orderbook_rollup.is_empty():
            upsert_parquet_dataset(
                orderbook_rollup,
                root / ORDERBOOK_SPREAD_ROLLUP,
                key_columns=ORDERBOOK_SPREAD_ROLLUP_KEYS,
            )
    result.rollup_rows["orderbook_spread_1m"] = orderbook_rollup.height


def build_trade_activity_1m_rollup(
    lake_root: str | Path,
    *,
    since: datetime | None = None,
    warnings: list[str] | None = None,
) -> pl.DataFrame:
    path = Path(lake_root) / HF_DATASETS["trade_print"]
    try:
        lazy = _source_lazy(path, since=since, warnings=warnings)
        if lazy is None:
            return pl.DataFrame()
        schema = set(lazy.collect_schema().names())
    except Exception:
        return pl.DataFrame()
    if "symbol" not in schema or "ts" not in schema:
        return pl.DataFrame()
    size_field = _first_existing(schema, ("size", "qty", "amount", "volume"))
    size_value = (
        pl.col(size_field).cast(pl.Float64, strict=False)
        if size_field is not None
        else pl.lit(None).cast(pl.Float64)
    )
    size_expr = size_value.sum().alias("size_sum")
    side_field = _first_existing(schema, ("side", "taker_side", "direction", "aggressor_side"))
    buy_size_field = _first_existing(
        schema, ("taker_buy_size", "taker_buy_volume", "buy_size", "buy_volume")
    )
    sell_size_field = _first_existing(
        schema, ("taker_sell_size", "taker_sell_volume", "sell_size", "sell_volume")
    )
    if buy_size_field is not None:
        taker_buy_expr = (
            pl.col(buy_size_field).cast(pl.Float64, strict=False).sum().alias("taker_buy_size_sum")
        )
    elif side_field is not None and size_field is not None:
        side_text = pl.col(side_field).cast(pl.Utf8, strict=False).str.to_lowercase()
        taker_buy_expr = (
            pl.when(side_text.str.contains("buy|bid", literal=False))
            .then(size_value)
            .otherwise(0.0)
            .sum()
            .alias("taker_buy_size_sum")
        )
    else:
        taker_buy_expr = pl.lit(None).cast(pl.Float64).alias("taker_buy_size_sum")
    if sell_size_field is not None:
        taker_sell_expr = (
            pl.col(sell_size_field)
            .cast(pl.Float64, strict=False)
            .sum()
            .alias("taker_sell_size_sum")
        )
    elif side_field is not None and size_field is not None:
        side_text = pl.col(side_field).cast(pl.Utf8, strict=False).str.to_lowercase()
        taker_sell_expr = (
            pl.when(side_text.str.contains("sell|ask", literal=False))
            .then(size_value)
            .otherwise(0.0)
            .sum()
            .alias("taker_sell_size_sum")
        )
    else:
        taker_sell_expr = pl.lit(None).cast(pl.Float64).alias("taker_sell_size_sum")
    ts_expr = _timestamp_expr("ts")
    filtered = lazy.with_columns(ts_expr.alias("_ts"))
    if since is not None:
        filtered = filtered.filter(pl.col("_ts") >= since)
    return (
        filtered.with_columns(pl.col("_ts").dt.truncate("1m").alias("minute_ts"))
        .group_by(["symbol", "minute_ts"])
        .agg(
            [
                pl.len().alias("trade_count"),
                size_expr,
                taker_buy_expr,
                taker_sell_expr,
                pl.col("_ts").max().alias("latest_trade_ts"),
            ]
        )
        .sort(["symbol", "minute_ts"])
        .pipe(_collect_rollup)
    )


def build_orderbook_spread_1m_rollup(
    lake_root: str | Path,
    *,
    since: datetime | None = None,
    warnings: list[str] | None = None,
) -> pl.DataFrame:
    path = Path(lake_root) / HF_DATASETS["orderbook_snapshot"]
    try:
        lazy = _source_lazy(path, since=since, warnings=warnings)
        if lazy is None:
            return pl.DataFrame()
        schema = set(lazy.collect_schema().names())
    except Exception:
        return pl.DataFrame()
    if not {"symbol", "ts"}.issubset(schema):
        return pl.DataFrame()
    channel_expr = pl.col("channel") if "channel" in schema else pl.lit("").alias("channel")
    bid_size_field = _first_existing(schema, ("bid_size", "bid_qty", "bid_volume"))
    ask_size_field = _first_existing(schema, ("ask_size", "ask_qty", "ask_volume"))
    imbalance_field = _first_existing(
        schema, ("orderbook_imbalance", "imbalance", "bid_ask_imbalance", "book_imbalance")
    )
    json_columns = ["asks_json", "bids_json"] if {"asks_json", "bids_json"}.issubset(schema) else []
    if "spread_bps" in schema:
        selected_columns = [
            "symbol",
            "ts",
            "spread_bps",
            *json_columns,
            *([imbalance_field] if imbalance_field is not None else []),
            *([bid_size_field] if bid_size_field is not None else []),
            *([ask_size_field] if ask_size_field is not None else []),
            *(["channel"] if "channel" in schema else []),
        ]
        spread_expr = pl.col("spread_bps").cast(pl.Float64, strict=False).alias("spread_bps")
    elif {"asks_json", "bids_json"}.issubset(schema):
        selected_columns = [
            "symbol",
            "ts",
            "asks_json",
            "bids_json",
            *([imbalance_field] if imbalance_field is not None else []),
            *([bid_size_field] if bid_size_field is not None else []),
            *([ask_size_field] if ask_size_field is not None else []),
            *(["channel"] if "channel" in schema else []),
        ]
        spread_expr = _spread_bps_expr("asks_json", "bids_json")
    else:
        return pl.DataFrame()
    imbalance_expr = _orderbook_imbalance_expr(
        schema,
        imbalance_field=imbalance_field,
        bid_size_field=bid_size_field,
        ask_size_field=ask_size_field,
        allow_json_fallback={"asks_json", "bids_json"}.issubset(schema),
    )
    frame = (
        lazy.select(selected_columns)
        .with_columns(
            [
                _timestamp_expr("ts").alias("_ts"),
                channel_expr,
                spread_expr,
                imbalance_expr,
            ]
        )
        .filter(pl.col("_ts") >= since if since is not None else pl.lit(True))
        .with_columns(pl.col("_ts").dt.truncate("1m").alias("minute_ts"))
        .filter(pl.col("spread_bps").is_not_null())
        .group_by(["symbol", "channel", "minute_ts"])
        .agg(
            [
                pl.col("spread_bps").mean().alias("spread_bps"),
                pl.col("orderbook_imbalance").mean().alias("orderbook_imbalance"),
                pl.col("_ts").max().alias("ts"),
            ]
        )
        .sort(["symbol", "channel", "minute_ts"])
        .pipe(_collect_rollup)
    )
    return frame


def _collect_rollup(lazy: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lazy.collect(engine="streaming")
    except TypeError:
        try:
            return lazy.collect(streaming=True)
        except TypeError:
            return lazy.collect()


def _first_existing(schema: set[str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in schema:
            return name
    return None


def _orderbook_imbalance_expr(
    schema: set[str],
    *,
    imbalance_field: str | None,
    bid_size_field: str | None,
    ask_size_field: str | None,
    allow_json_fallback: bool = True,
) -> pl.Expr:
    if imbalance_field is not None:
        return pl.col(imbalance_field).cast(pl.Float64, strict=False).alias("orderbook_imbalance")
    if bid_size_field is not None and ask_size_field is not None:
        bid = pl.col(bid_size_field).cast(pl.Float64, strict=False)
        ask = pl.col(ask_size_field).cast(pl.Float64, strict=False)
        return (
            pl.when((bid + ask) > 0)
            .then((bid - ask) / (bid + ask))
            .otherwise(None)
            .alias("orderbook_imbalance")
        )
    if allow_json_fallback and {"asks_json", "bids_json"}.issubset(schema):
        return _book_imbalance_json_expr("asks_json", "bids_json")
    return pl.lit(None).cast(pl.Float64).alias("orderbook_imbalance")


def _rollup_since(now: datetime, lookback_hours: int | None) -> datetime | None:
    if lookback_hours is None:
        return None
    try:
        hours = int(lookback_hours)
    except (TypeError, ValueError):
        return None
    if hours <= 0:
        return None
    return now - timedelta(hours=hours)


def _timestamp_expr(column: str) -> pl.Expr:
    return pl.coalesce(
        [
            pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(time_zone="UTC", strict=False),
        ]
    )


def _spread_bps_expr(asks_column: str, bids_column: str) -> pl.Expr:
    ask = _top_book_number_expr(asks_column, 1)
    bid = _top_book_number_expr(bids_column, 1)
    mid = (ask + bid) / 2.0
    return (
        pl.when((ask.is_not_null()) & (bid.is_not_null()) & (ask > bid) & (mid > 0))
        .then(((ask - bid) / mid) * 10_000.0)
        .otherwise(None)
        .alias("spread_bps")
    )


def _book_imbalance_json_expr(asks_column: str, bids_column: str) -> pl.Expr:
    ask_size = _top_book_number_expr(asks_column, 2)
    bid_size = _top_book_number_expr(bids_column, 2)
    total = ask_size + bid_size
    return (
        pl.when(
            (ask_size.is_not_null())
            & (bid_size.is_not_null())
            & (total.is_not_null())
            & (total > 0)
        )
        .then((bid_size - ask_size) / total)
        .otherwise(None)
        .alias("orderbook_imbalance")
    )


def _top_book_number_expr(column: str, group_index: int) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.extract(TOP_BOOK_LEVEL_RE, group_index)
        .cast(pl.Float64, strict=False)
    )


def _source_lazy(
    path: Path,
    *,
    since: datetime | None,
    warnings: list[str] | None = None,
) -> pl.LazyFrame | None:
    if since is None:
        return read_parquet_lazy(path)
    files = _recent_parquet_files(path, since=since, warnings=warnings)
    if not files:
        return None
    try:
        return pl.scan_parquet(
            [str(file_path) for file_path in files],
            hive_partitioning=False,
            missing_columns="insert",
            extra_columns="ignore",
        )
    except TypeError:
        return pl.scan_parquet([str(file_path) for file_path in files], hive_partitioning=False)


def _recent_parquet_files(
    path: Path,
    *,
    since: datetime,
    warnings: list[str] | None = None,
) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    indexed_files = _recent_parquet_files_from_index(path, since=since)
    mtime_files = _recent_parquet_files_by_mtime(path, since=since)
    if indexed_files is not None:
        # The lake file index can lag behind a hot append stream.  Merge the
        # indexed set with mtime-recent files so rollups stay fresh without
        # rebuilding min/max timestamp metadata for every source parquet file.
        merged: dict[str, Path] = {}
        missing_from_disk = 0
        for file_path in indexed_files:
            if file_path.is_file() and not _is_internal_file(file_path):
                merged[str(file_path)] = file_path
            else:
                missing_from_disk += 1
        if missing_from_disk and warnings is not None:
            warnings.append(
                f"file_index_stale_dropped_missing_files:{path.as_posix()}:{missing_from_disk}"
            )
        missing_from_index = 0
        for file_path in mtime_files:
            key = str(file_path)
            if key not in merged:
                missing_from_index += 1
                merged[key] = file_path
        if missing_from_index and warnings is not None:
            warnings.append(
                f"file_index_stale_merged_recent_mtime_files:{path.as_posix()}:{missing_from_index}"
            )
        return sorted(merged.values())
    if warnings is not None:
        warnings.append(f"file_index_missing_fallback_rglob:{path.as_posix()}")
    return mtime_files


def _recent_parquet_files_by_mtime(path: Path, *, since: datetime) -> list[Path]:
    cutoff = since.timestamp()
    files: list[Path] = []
    for candidate in sorted(path.rglob("*.parquet")):
        if not candidate.is_file() or _is_internal_file(candidate):
            continue
        try:
            if candidate.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        files.append(candidate)
    return files


def _recent_parquet_files_from_index(path: Path, *, since: datetime) -> list[Path] | None:
    try:
        from quant_lab.data.file_index import recent_files_for_dataset

        return recent_files_for_dataset(path, since=since)
    except Exception:
        return None


def _is_internal_file(path: Path) -> bool:
    if path.name.startswith(".") or path.name.endswith(".tmp.parquet"):
        return True
    return any(part.startswith("__") or part.startswith(".") for part in path.parts)


def _archive_old_high_frequency_files(
    root: Path,
    *,
    hot_hours: int,
    dry_run: bool,
    now: datetime,
    result: MarketDataCompactionResult,
) -> None:
    cutoff = now - timedelta(hours=max(int(hot_hours), 1))
    for dataset_name, relative_path in HF_DATASETS.items():
        dataset_root = root / relative_path
        if not dataset_root.exists():
            continue
        archive_root = root / "archive" / "high_frequency" / relative_path
        coverage_spec = SILVER_ROLLUP_COVERAGE.get(dataset_name)
        preserve_source_layout = coverage_spec is not None
        if coverage_spec is None:
            indexed_files = _old_archive_files_from_index(dataset_root, before=cutoff)
            if indexed_files is None:
                result.warnings.append(f"archive_fallback_rglob:{dataset_root.as_posix()}")
                files = sorted(dataset_root.rglob("*.parquet"))
            else:
                files = sorted(indexed_files)
        else:
            rollup_path, timestamp_column = coverage_spec
            coverage = _rollup_coverage(root / rollup_path, timestamp_column=timestamp_column)
            if coverage is None:
                result.warnings.append(
                    f"archive_skipped_rollup_coverage_unavailable:{dataset_root.as_posix()}"
                )
                continue
            coverage_start, coverage_end = coverage
            effective_before = min(cutoff, coverage_end + timedelta(minutes=1))
            indexed_files = _covered_archive_files_from_index(
                dataset_root,
                since=coverage_start,
                before=effective_before,
            )
            if indexed_files is None:
                result.warnings.append(
                    f"archive_skipped_file_index_unavailable:{dataset_root.as_posix()}"
                )
                continue
            files = sorted(indexed_files)
        _archive_dataset_files(
            dataset_root,
            archive_root,
            dataset_name=str(relative_path).replace("\\", "/"),
            files=files,
            cutoff=cutoff,
            dry_run=dry_run,
            preserve_source_layout=preserve_source_layout,
            result=result,
        )


def _archive_dataset_files(
    dataset_root: Path,
    archive_root: Path,
    *,
    dataset_name: str,
    files: list[Path],
    cutoff: datetime,
    dry_run: bool,
    preserve_source_layout: bool,
    result: MarketDataCompactionResult,
) -> None:
    for path in files:
        if not path.exists() or not path.is_file() or _is_internal_file(path):
            continue
        try:
            stat = path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, UTC)
        except OSError as exc:
            result.warnings.append(f"stat_failed:{path}:{exc}")
            continue
        if mtime >= cutoff:
            continue
        base_dest = archive_root / f"date={mtime.date().isoformat()}" / f"hour={mtime.hour:02d}"
        if preserve_source_layout:
            try:
                relative_source = path.relative_to(dataset_root)
            except ValueError:
                result.warnings.append(f"archive_source_outside_dataset:{path}")
                continue
            dest = base_dest / relative_source
        else:
            symbol = _first_symbol(path) or "unknown"
            dest = base_dest / f"symbol={symbol}" / path.name
        if dest.exists():
            result.warnings.append(f"archive_destination_exists:{dest}")
            continue
        if dry_run:
            _record_archived_file(result, path, dataset_name=dataset_name, size=stat.st_size)
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
        except OSError as exc:
            result.warnings.append(f"archive_move_failed:{path}:{exc}")
            continue
        _record_archived_file(result, path, dataset_name=dataset_name, size=stat.st_size)
    if not dry_run:
        _remove_empty_source_directories(dataset_root)


def _record_archived_file(
    result: MarketDataCompactionResult,
    path: Path,
    *,
    dataset_name: str,
    size: int,
) -> None:
    result.archived_file_count += 1
    result.archived_bytes += max(int(size), 0)
    result.archived_by_dataset[dataset_name] = result.archived_by_dataset.get(dataset_name, 0) + 1
    if len(result.archived_files) < MAX_ARCHIVED_FILE_SAMPLES:
        result.archived_files.append(str(path))
    else:
        result.archived_files_truncated = True


def _remove_empty_source_directories(dataset_root: Path) -> None:
    directories = sorted(
        (path for path in dataset_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue


def _rollup_coverage(
    path: Path,
    *,
    timestamp_column: str,
) -> tuple[datetime, datetime] | None:
    try:
        lazy = read_parquet_lazy(path)
        if timestamp_column not in lazy.collect_schema().names():
            return None
        bounds = (
            lazy.select(
                [
                    _timestamp_expr(timestamp_column).min().alias("_min_ts"),
                    _timestamp_expr(timestamp_column).max().alias("_max_ts"),
                ]
            )
            .collect()
            .row(0, named=True)
        )
    except Exception:
        return None
    start = bounds.get("_min_ts")
    end = bounds.get("_max_ts")
    if not isinstance(start, datetime) or not isinstance(end, datetime) or end < start:
        return None
    return start, end


def _old_archive_files_from_index(path: Path, *, before: datetime) -> list[Path] | None:
    try:
        return old_files_for_dataset(path, before=before)
    except Exception:
        return None


def _covered_archive_files_from_index(
    path: Path,
    *,
    since: datetime,
    before: datetime,
) -> list[Path] | None:
    try:
        return files_fully_within_time_range(path, since=since, before=before)
    except Exception:
        return None


def _spread_bps(row: dict[str, Any]) -> float | None:
    ask = _best_price(row.get("asks_json"))
    bid = _best_price(row.get("bids_json"))
    if ask is None or bid is None or ask <= bid:
        return None
    mid = (ask + bid) / 2.0
    return ((ask - bid) / mid) * 10_000.0


def _best_price(value: Any) -> float | None:
    data = value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if isinstance(first, list) and first:
        first = first[0]
    try:
        return float(first)
    except (TypeError, ValueError):
        return None


def _book_imbalance(row: dict[str, Any]) -> float | None:
    bid_size = _best_size(row.get("bids_json"))
    ask_size = _best_size(row.get("asks_json"))
    if bid_size is None or ask_size is None:
        return None
    total = bid_size + ask_size
    return (bid_size - ask_size) / total if total > 0 else None


def _best_size(value: Any) -> float | None:
    data = value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if isinstance(first, list):
        if len(first) < 2:
            return None
        first = first[1]
    try:
        return float(first)
    except (TypeError, ValueError):
        return None


def _first_symbol(path: Path) -> str | None:
    try:
        frame = pl.scan_parquet(str(path)).select("symbol").head(1).collect()
    except Exception:
        return None
    if frame.is_empty():
        return None
    text = str(frame.item() or "").strip()
    return text.replace("/", "-") if text else None
