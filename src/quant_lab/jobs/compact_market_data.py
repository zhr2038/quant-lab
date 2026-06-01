from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

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


@dataclass
class MarketDataCompactionResult:
    lake_root: str
    dry_run: bool
    started_at: datetime
    finished_at: datetime | None = None
    archived_files: list[str] = field(default_factory=list)
    rollup_rows: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_root": self.lake_root,
            "dry_run": self.dry_run,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "archived_files": self.archived_files,
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
    _archive_old_okx_public_ws(
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
    trade_rollup = build_trade_activity_1m_rollup(root, since=since)
    orderbook_rollup = build_orderbook_spread_1m_rollup(root, since=since)
    if not dry_run:
        if not trade_rollup.is_empty():
            upsert_parquet_dataset(
                trade_rollup,
                root / TRADE_ACTIVITY_ROLLUP,
                key_columns=TRADE_ACTIVITY_ROLLUP_KEYS,
            )
        if not orderbook_rollup.is_empty():
            upsert_parquet_dataset(
                orderbook_rollup,
                root / ORDERBOOK_SPREAD_ROLLUP,
                key_columns=ORDERBOOK_SPREAD_ROLLUP_KEYS,
            )
    result.rollup_rows["trade_activity_1m"] = trade_rollup.height
    result.rollup_rows["orderbook_spread_1m"] = orderbook_rollup.height


def build_trade_activity_1m_rollup(
    lake_root: str | Path,
    *,
    since: datetime | None = None,
) -> pl.DataFrame:
    path = Path(lake_root) / HF_DATASETS["trade_print"]
    try:
        lazy = _source_lazy(path, since=since)
        if lazy is None:
            return pl.DataFrame()
        schema = set(lazy.collect_schema().names())
    except Exception:
        return pl.DataFrame()
    if "symbol" not in schema or "ts" not in schema:
        return pl.DataFrame()
    size_expr = (
        pl.col("size").cast(pl.Float64, strict=False).sum().alias("size_sum")
        if "size" in schema
        else pl.lit(None).cast(pl.Float64).alias("size_sum")
    )
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
                pl.col("_ts").max().alias("latest_trade_ts"),
            ]
        )
        .sort(["symbol", "minute_ts"])
        .collect()
    )


def build_orderbook_spread_1m_rollup(
    lake_root: str | Path,
    *,
    since: datetime | None = None,
) -> pl.DataFrame:
    path = Path(lake_root) / HF_DATASETS["orderbook_snapshot"]
    try:
        lazy = _source_lazy(path, since=since)
        if lazy is None:
            return pl.DataFrame()
        schema = set(lazy.collect_schema().names())
    except Exception:
        return pl.DataFrame()
    if not {"symbol", "ts", "asks_json", "bids_json"}.issubset(schema):
        return pl.DataFrame()
    channel_expr = pl.col("channel") if "channel" in schema else pl.lit("").alias("channel")
    selected_columns = [
        "symbol",
        "ts",
        "asks_json",
        "bids_json",
        *(["channel"] if "channel" in schema else []),
    ]
    frame = (
        lazy.select(selected_columns)
        .with_columns(
            [
                _timestamp_expr("ts").alias("_ts"),
                channel_expr,
                pl.struct(["asks_json", "bids_json"])
                .map_elements(_spread_bps, return_dtype=pl.Float64)
                .alias("spread_bps"),
            ]
        )
        .filter(pl.col("_ts") >= since if since is not None else pl.lit(True))
        .with_columns(pl.col("_ts").dt.truncate("1m").alias("minute_ts"))
        .filter(pl.col("spread_bps").is_not_null())
        .group_by(["symbol", "channel", "minute_ts"])
        .agg(
            [
                pl.col("spread_bps").mean().alias("spread_bps"),
                pl.col("_ts").max().alias("ts"),
            ]
        )
        .sort(["symbol", "channel", "minute_ts"])
        .collect()
    )
    return frame


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


def _source_lazy(path: Path, *, since: datetime | None) -> pl.LazyFrame | None:
    if since is None:
        return read_parquet_lazy(path)
    files = _recent_parquet_files(path, since=since)
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


def _recent_parquet_files(path: Path, *, since: datetime) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
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


def _is_internal_file(path: Path) -> bool:
    if path.name.startswith(".") or path.name.endswith(".tmp.parquet"):
        return True
    return any(part.startswith("__") or part.startswith(".") for part in path.parts)


def _archive_old_okx_public_ws(
    root: Path,
    *,
    hot_hours: int,
    dry_run: bool,
    now: datetime,
    result: MarketDataCompactionResult,
) -> None:
    dataset_root = root / HF_DATASETS["okx_public_ws"]
    if not dataset_root.exists():
        return
    cutoff = now - timedelta(hours=max(int(hot_hours), 1))
    archive_root = root / "archive" / "high_frequency" / "bronze" / "okx_public_ws"
    for path in sorted(dataset_root.rglob("*.parquet")):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        except OSError as exc:
            result.warnings.append(f"stat_failed:{path}:{exc}")
            continue
        if mtime >= cutoff:
            continue
        symbol = _first_symbol(path) or "unknown"
        dest = (
            archive_root
            / f"date={mtime.date().isoformat()}"
            / f"hour={mtime.hour:02d}"
            / f"symbol={symbol}"
            / path.name
        )
        result.archived_files.append(str(path))
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))


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


def _first_symbol(path: Path) -> str | None:
    try:
        frame = pl.scan_parquet(str(path)).select("symbol").head(1).collect()
    except Exception:
        return None
    if frame.is_empty():
        return None
    text = str(frame.item() or "").strip()
    return text.replace("/", "-") if text else None
