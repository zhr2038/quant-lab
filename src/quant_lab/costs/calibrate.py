import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.costs.health import build_cost_health_daily, publish_cost_health_daily
from quant_lab.costs.model import DEFAULT_FALLBACK_COST_BPS, CostBucketDaily
from quant_lab.data.lake import read_parquet_dataset, read_parquet_lazy, write_parquet_dataset
from quant_lab.ingest.okx_readonly_private import (
    BRONZE_BILLS_DATASET,
    BRONZE_FILLS_DATASET,
    normalize_okx_bills,
    normalize_okx_fills,
)
from quant_lab.symbols import normalize_symbol

FILL_EVENT_DATASET = Path("silver") / "fill_event"
ACCOUNT_BILL_DATASET = Path("silver") / "account_bill"
ORDERBOOK_SNAPSHOT_DATASET = Path("silver") / "orderbook_snapshot"
TRADE_PRINT_DATASET = Path("silver") / "trade_print"
V5_TRADE_EVENT_DATASET = Path("silver") / "v5_trade_event"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
COST_HEALTH_DAILY_DATASET = Path("gold") / "cost_health_daily"
DEFAULT_MIN_SAMPLE_COUNT = 30
PRIVATE_COST_LOOKBACK_DAYS = 7

COST_BUCKET_DAILY_SCHEMA = {
    "day": pl.Utf8,
    "symbol": pl.Utf8,
    "regime": pl.Utf8,
    "event_type": pl.Utf8,
    "notional_bucket": pl.Utf8,
    "sample_count": pl.Int64,
    "fee_bps_p50": pl.Float64,
    "fee_bps_p75": pl.Float64,
    "fee_bps_p90": pl.Float64,
    "slippage_bps_p50": pl.Float64,
    "slippage_bps_p75": pl.Float64,
    "slippage_bps_p90": pl.Float64,
    "spread_bps_p50": pl.Float64,
    "spread_bps_p75": pl.Float64,
    "spread_bps_p90": pl.Float64,
    "total_cost_bps_p50": pl.Float64,
    "total_cost_bps_p75": pl.Float64,
    "total_cost_bps_p90": pl.Float64,
    "fallback_level": pl.Utf8,
    "source": pl.Utf8,
    "cost_source": pl.Utf8,
    "actual_fill_count": pl.Int64,
    "mixed_fill_count": pl.Int64,
    "proxy_sample_count": pl.Int64,
    "cost_model_version": pl.Utf8,
    "created_at": pl.Utf8,
}


class CostCalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    day: str
    rows_written: int = Field(ge=0)
    health_rows_written: int = Field(default=0, ge=0)
    dataset_path: str
    health_dataset_path: str = ""
    sources: list[str]
    cost_health_status: str | None = None


def calibrate_costs_for_day(
    lake_root: str | Path,
    day: str,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
) -> CostCalibrationResult:
    root = Path(lake_root)
    market_bars = _read_day_dataset(root, MARKET_BAR_DATASET, day)
    fill_events = _private_fill_events_for_day(root, day)
    account_bills = _private_account_bills_for_day(root, day)
    v5_trade_events = _v5_trade_events_for_day(root, day)
    rows = build_cost_bucket_daily_rows(
        fill_events=fill_events,
        account_bills=account_bills,
        order_events=_filter_recent_window(
            read_parquet_dataset(root / "silver" / "order_event"),
            day=day,
            lookback_days=PRIVATE_COST_LOOKBACK_DAYS,
        ),
        orderbook_snapshots=_read_day_dataset(root, ORDERBOOK_SNAPSHOT_DATASET, day),
        trade_prints=_read_day_dataset(root, TRADE_PRINT_DATASET, day),
        v5_trade_events=v5_trade_events,
        market_bars=market_bars,
        day=day,
        min_sample_count=min_sample_count,
    )
    rows_written = publish_cost_bucket_daily(root, rows)
    cost_frame = _cost_bucket_daily_frame(rows)
    health = build_cost_health_daily(
        cost_frame,
        day=day,
        min_sample_count=min_sample_count,
        expected_symbols=sorted(
            set(_symbols_from_public_data(pl.DataFrame(), market_bars))
            | _symbols_from_fill_events(fill_events)
            | _symbols_from_v5_trade_events(v5_trade_events)
        ),
        private_fill_rows=fill_events.height,
        private_bill_rows=account_bills.height,
        v5_trade_rows=v5_trade_events.height,
        fee_bps_missing_count=_fee_missing_count(fill_events)
        + _v5_trade_fee_missing_count(v5_trade_events),
    )
    health_rows_written = publish_cost_health_daily(root, health)
    return CostCalibrationResult(
        lake_root=str(root),
        day=day,
        rows_written=rows_written,
        health_rows_written=health_rows_written,
        dataset_path=str(root / COST_BUCKET_DAILY_DATASET),
        health_dataset_path=str(root / COST_HEALTH_DAILY_DATASET),
        sources=sorted({row.source for row in rows}),
        cost_health_status=health.status,
    )


def build_cost_bucket_daily_rows(
    fill_events: pl.DataFrame,
    account_bills: pl.DataFrame,
    orderbook_snapshots: pl.DataFrame,
    trade_prints: pl.DataFrame,
    market_bars: pl.DataFrame,
    day: str,
    order_events: pl.DataFrame | None = None,
    v5_trade_events: pl.DataFrame | None = None,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
) -> list[CostBucketDaily]:
    spread_samples = _spread_samples_by_symbol(orderbook_snapshots)
    reference_prices = _reference_prices_by_order(
        order_events if order_events is not None else pl.DataFrame()
    )
    fill_samples = [
        *_fill_samples(fill_events, reference_prices),
        *_v5_trade_fill_samples(v5_trade_events if v5_trade_events is not None else pl.DataFrame()),
    ]

    if fill_samples:
        actual_rows = _actual_fill_rows(
            fill_samples=fill_samples,
            bills_present=not account_bills.is_empty(),
            spread_samples=spread_samples,
            day=day,
            min_sample_count=min_sample_count,
        )
        actual_symbols = {row.symbol for row in actual_rows}
        proxy_rows = _public_spread_proxy_rows(
            spread_samples={
                symbol: samples
                for symbol, samples in spread_samples.items()
                if symbol not in actual_symbols
            },
            day=day,
        )
        return sorted(
            [*actual_rows, *proxy_rows],
            key=lambda row: (row.symbol, row.notional_bucket),
        )

    if spread_samples:
        return _public_spread_proxy_rows(spread_samples=spread_samples, day=day)

    symbols = _symbols_from_public_data(trade_prints, market_bars)
    if symbols:
        return [_global_default_row(day=day, symbol=symbol) for symbol in symbols]

    return [_global_default_row(day=day, symbol="GLOBAL")]


def publish_cost_bucket_daily(lake_root: str | Path, rows: Sequence[CostBucketDaily]) -> int:
    dataset_path = Path(lake_root) / COST_BUCKET_DAILY_DATASET
    df = _cost_bucket_daily_frame(rows)
    existing = read_parquet_dataset(dataset_path)
    days = sorted({row.day for row in rows})
    if not existing.is_empty() and "day" in existing.columns and days:
        existing = existing.filter(~pl.col("day").is_in(days))
    frames = [frame for frame in [existing, df] if not frame.is_empty()]
    combined = pl.concat(frames, how="diagonal_relaxed") if frames else df
    if not combined.is_empty():
        combined = combined.unique(
            subset=["day", "symbol", "regime", "event_type", "notional_bucket"],
            keep="last",
            maintain_order=True,
        )
    write_parquet_dataset(combined, dataset_path)
    return combined.height


def _actual_fill_rows(
    fill_samples: list[dict[str, Any]],
    bills_present: bool,
    spread_samples: dict[str, list[float]],
    day: str,
    min_sample_count: int,
) -> list[CostBucketDaily]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in fill_samples:
        groups.setdefault((sample["symbol"], sample["notional_bucket"]), []).append(sample)

    rows: list[CostBucketDaily] = []
    for (symbol, notional_bucket), samples in sorted(groups.items()):
        rows.append(
            _actual_fill_row(
                symbol=symbol,
                notional_bucket=notional_bucket,
                samples=samples,
                bills_present=bills_present,
                spread_samples=spread_samples,
                day=day,
                min_sample_count=min_sample_count,
            )
        )
    for symbol, samples in sorted(_group_all_fill_samples(fill_samples).items()):
        rows.append(
            _actual_fill_row(
                symbol=symbol,
                notional_bucket="all",
                samples=samples,
                bills_present=bills_present,
                spread_samples=spread_samples,
                day=day,
                min_sample_count=min_sample_count,
            )
        )
    return sorted(rows, key=lambda row: (row.symbol, row.notional_bucket))


def _actual_fill_row(
    *,
    symbol: str,
    notional_bucket: str,
    samples: list[dict[str, Any]],
    bills_present: bool,
    spread_samples: dict[str, list[float]],
    day: str,
    min_sample_count: int,
) -> CostBucketDaily:
    fee_samples = [sample["fee_bps"] for sample in samples if sample["fee_bps"] is not None]
    slippage_samples = [
        sample["slippage_bps"] for sample in samples if sample["slippage_bps"] is not None
    ]
    fee_missing = len(fee_samples) != len(samples)
    spread_values = spread_samples.get(symbol, [])
    spread_fallback = not spread_values
    sample_too_small = len(samples) < min_sample_count
    slippage_unknown = len(slippage_samples) != len(samples)

    source = _actual_fill_source(
        samples=samples,
        fee_missing=fee_missing,
        sample_too_small=sample_too_small,
        slippage_unknown=slippage_unknown,
    )
    fallback_parts = []
    if not bills_present:
        fallback_parts.append("BILLS_MISSING")
    if fee_missing:
        fallback_parts.append("FEE_MISSING")
    if sample_too_small:
        fallback_parts.append("SAMPLE_TOO_SMALL")
    if slippage_unknown:
        fallback_parts.append("SLIPPAGE_UNKNOWN")
    if spread_fallback:
        fallback_parts.append("SPREAD_MISSING")
    else:
        fallback_parts.append("SPREAD_PROXY")
    if _uses_private_fill_lookback(samples, day):
        fallback_parts.append("PRIVATE_FILL_LOOKBACK")

    fee_p50, fee_p75, fee_p90 = _percentiles_or_zero(fee_samples)
    slippage_p50, slippage_p75, slippage_p90 = _percentiles_or_zero(slippage_samples)
    spread_p50, spread_p75, spread_p90 = _percentiles_or_zero(spread_values)

    return CostBucketDaily(
        day=day,
        symbol=symbol,
        regime="realized",
        event_type="actual_fill",
        notional_bucket=notional_bucket,
        sample_count=len(samples),
        fee_bps_p50=fee_p50,
        fee_bps_p75=fee_p75,
        fee_bps_p90=fee_p90,
        slippage_bps_p50=slippage_p50,
        slippage_bps_p75=slippage_p75,
        slippage_bps_p90=slippage_p90,
        spread_bps_p50=spread_p50,
        spread_bps_p75=spread_p75,
        spread_bps_p90=spread_p90,
        total_cost_bps_p50=fee_p50 + slippage_p50 + spread_p50,
        total_cost_bps_p75=fee_p75 + slippage_p75 + spread_p75,
        total_cost_bps_p90=fee_p90 + slippage_p90 + spread_p90,
        fallback_level=";".join(fallback_parts) if fallback_parts else "NONE",
        source=source,
        cost_source=source,
        actual_fill_count=len(samples) if source == "actual_fills" else 0,
        mixed_fill_count=len(samples) if source != "actual_fills" else 0,
        proxy_sample_count=len(spread_values),
    )


def _public_spread_proxy_rows(
    spread_samples: dict[str, list[float]],
    day: str,
) -> list[CostBucketDaily]:
    rows: list[CostBucketDaily] = []
    for symbol, samples in sorted(spread_samples.items()):
        spread_p50, spread_p75, spread_p90 = _percentiles_or_zero(samples)
        rows.append(
            CostBucketDaily(
                day=day,
                symbol=symbol,
                regime="public_proxy",
                event_type="spread_proxy",
                notional_bucket="all",
                sample_count=len(samples),
                fee_bps_p50=0.0,
                fee_bps_p75=0.0,
                fee_bps_p90=0.0,
                slippage_bps_p50=0.0,
                slippage_bps_p75=0.0,
                slippage_bps_p90=0.0,
                spread_bps_p50=spread_p50,
                spread_bps_p75=spread_p75,
                spread_bps_p90=spread_p90,
                total_cost_bps_p50=spread_p50,
                total_cost_bps_p75=spread_p75,
                total_cost_bps_p90=spread_p90,
                fallback_level="FEE_MISSING;SLIPPAGE_UNKNOWN;PUBLIC_SPREAD_PROXY",
                source="public_spread_proxy",
                cost_source="public_spread_proxy",
                actual_fill_count=0,
                mixed_fill_count=0,
                proxy_sample_count=len(samples),
            )
        )
    return rows


def _global_default_row(day: str, symbol: str) -> CostBucketDaily:
    return CostBucketDaily(
        day=day,
        symbol=symbol,
        regime="global_default",
        event_type="global_default",
        notional_bucket="all",
        sample_count=0,
        fee_bps_p50=0.0,
        fee_bps_p75=0.0,
        fee_bps_p90=0.0,
        slippage_bps_p50=0.0,
        slippage_bps_p75=0.0,
        slippage_bps_p90=0.0,
        spread_bps_p50=0.0,
        spread_bps_p75=0.0,
        spread_bps_p90=0.0,
        total_cost_bps_p50=DEFAULT_FALLBACK_COST_BPS,
        total_cost_bps_p75=DEFAULT_FALLBACK_COST_BPS,
        total_cost_bps_p90=DEFAULT_FALLBACK_COST_BPS,
        fallback_level="GLOBAL_DEFAULT",
        source="global_default",
        cost_source="global_default",
    )


def _fill_samples(
    fill_events: pl.DataFrame,
    reference_prices: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in fill_events.to_dicts():
        price = _optional_float(row.get("fill_price"))
        size = _optional_float(row.get("fill_size"))
        if price is None or size is None:
            continue
        notional = abs(price * size)
        if notional <= 0:
            continue
        fee = _optional_float(row.get("fee"))
        fee_currency = str(row.get("fee_currency") or "")
        order_id = str(row.get("order_id") or "")
        side = str(row.get("side") or "").lower()
        symbol = normalize_symbol(row["inst_id"])
        reference_price = (reference_prices or {}).get(order_id)
        fee_abs_usdt = _fee_abs_usdt(
            fee=fee,
            fee_currency=fee_currency,
            symbol=symbol,
            fill_price=price,
        )
        samples.append(
            {
                "symbol": symbol,
                "source_kind": "okx_readonly_private",
                "notional": notional,
                "notional_bucket": _notional_bucket(notional),
                "trade_id": str(row.get("trade_id") or ""),
                "order_id": order_id,
                "side": side,
                "fill_px": price,
                "fill_qty": size,
                "fee": fee,
                "fee_ccy": fee_currency,
                "ts": row.get("ts"),
                "fee_bps": fee_abs_usdt / notional * 10_000
                if fee_abs_usdt is not None
                else None,
                "slippage_bps": _slippage_bps(
                    fill_price=price,
                    reference_price=reference_price,
                    side=side,
                ),
            }
        )
    return samples


def _v5_trade_fill_samples(v5_trade_events: pl.DataFrame) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in v5_trade_events.to_dicts():
        symbol = _symbol_from_trade_row(row)
        if not symbol:
            continue
        price = _first_float(row, ["price", "fill_price", "fill_px", "px", "avg_price"])
        qty = _first_float(row, ["qty", "quantity", "size", "fill_size", "fill_sz", "sz", "amount"])
        notional = _first_float(row, ["notional", "notional_usdt", "quote_notional", "turnover"])
        if notional is None and price is not None and qty is not None:
            notional = abs(price * qty)
        if notional is None or notional <= 0:
            continue
        fee_usdt = _first_float(row, ["fee_usdt", "fee_abs_usdt"])
        fee = _first_float(row, ["fee", "commission", "fee_abs"])
        fee_ccy = str(row.get("fee_ccy") or row.get("fee_currency") or "")
        fee_abs_usdt = abs(fee_usdt) if fee_usdt is not None else _fee_abs_usdt(
            fee=fee,
            fee_currency=fee_ccy,
            symbol=symbol,
            fill_price=price,
        )
        slippage = _first_float(
            row,
            [
                "realized_slippage_bps",
                "estimated_slippage_bps",
                "slippage_bps",
                "slip_bps",
            ],
        )
        if slippage is None:
            slippage_usdt = _first_float(row, ["slippage_usdt", "realized_slippage_usdt"])
            if slippage_usdt is not None and abs(notional) > 0:
                slippage = abs(slippage_usdt) / abs(notional) * 10_000
        samples.append(
            {
                "symbol": symbol,
                "source_kind": "v5_trades_csv",
                "notional": abs(notional),
                "notional_bucket": _notional_bucket(abs(notional)),
                "trade_id": str(row.get("trade_id") or row.get("tradeId") or ""),
                "order_id": str(row.get("order_id") or row.get("ordId") or ""),
                "side": str(row.get("side") or ""),
                "action": str(row.get("action") or ""),
                "fill_px": price,
                "fill_qty": qty,
                "fee": fee,
                "fee_ccy": fee_ccy,
                "fee_usdt": fee_abs_usdt,
                "ts": row.get("ts_utc") or row.get("ts") or row.get("timestamp"),
                "fee_bps": fee_abs_usdt / abs(notional) * 10_000
                if fee_abs_usdt is not None
                else None,
                "slippage_bps": slippage if slippage is not None and slippage >= 0 else None,
            }
        )
    return samples


def _symbol_from_trade_row(row: dict[str, Any]) -> str:
    for key in ["symbol", "normalized_symbol", "inst_id", "instId", "instrument", "pair"]:
        value = row.get(key)
        if value:
            return normalize_symbol(value)
    payload = row.get("raw_payload_json")
    if isinstance(payload, str) and payload.strip():
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            for key in ["symbol", "normalized_symbol", "inst_id", "instId", "instrument", "pair"]:
                value = loaded.get(key)
                if value:
                    return normalize_symbol(value)
    return ""


def _group_all_fill_samples(fill_samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in fill_samples:
        groups.setdefault(sample["symbol"], []).append(sample)
    return groups


def _reference_prices_by_order(order_events: pl.DataFrame) -> dict[str, float]:
    if order_events.is_empty():
        return {}
    references: dict[str, float] = {}
    for row in order_events.to_dicts():
        order_id = str(row.get("order_id") or "")
        if not order_id:
            continue
        value = _optional_float(row.get("reference_price"))
        if value is None:
            value = _optional_float(row.get("reference_px"))
        if value is not None:
            references[order_id] = value
    return references


def _slippage_bps(
    *,
    fill_price: float,
    reference_price: float | None,
    side: str,
) -> float | None:
    if reference_price is None or reference_price <= 0:
        return None
    if side == "sell":
        return max((reference_price - fill_price) / reference_price * 10_000, 0.0)
    return max((fill_price - reference_price) / reference_price * 10_000, 0.0)


def _spread_samples_by_symbol(orderbook_snapshots: pl.DataFrame) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    for row in orderbook_snapshots.to_dicts():
        bid = _best_price(row.get("bids_json"), best="bid")
        ask = _best_price(row.get("asks_json"), best="ask")
        if bid is None or ask is None or ask <= bid:
            continue
        mid = (ask + bid) / 2
        if mid <= 0:
            continue
        samples.setdefault(normalize_symbol(row["symbol"]), []).append((ask - bid) / mid * 10_000)
    return samples


def _best_price(raw_json: Any, best: str) -> float | None:
    if raw_json is None:
        return None
    try:
        levels = json.loads(str(raw_json))
    except json.JSONDecodeError:
        return None
    prices = [_optional_float(level[0]) for level in levels if isinstance(level, list) and level]
    clean_prices = [price for price in prices if price is not None]
    if not clean_prices:
        return None
    return max(clean_prices) if best == "bid" else min(clean_prices)


def _filter_day(df: pl.DataFrame, day: str) -> pl.DataFrame:
    if df.is_empty() or "ts" not in df.columns:
        return df
    return df.filter(pl.col("ts").cast(pl.Utf8).str.starts_with(day))


def _read_day_dataset(
    root: Path,
    dataset: Path,
    day: str,
    timestamp_column: str = "ts",
) -> pl.DataFrame:
    dataset_path = root / dataset
    try:
        lazy = read_parquet_lazy(dataset_path)
        columns = lazy.collect_schema().names()
        if timestamp_column not in columns:
            return lazy.collect(engine="streaming")
        return (
            lazy.filter(pl.col(timestamp_column).cast(pl.Utf8).str.starts_with(day))
            .collect(engine="streaming")
        )
    except Exception:
        return _filter_day(read_parquet_dataset(dataset_path), day)


def _symbols_from_public_data(trade_prints: pl.DataFrame, market_bars: pl.DataFrame) -> list[str]:
    symbols: set[str] = set()
    for df in [trade_prints, market_bars]:
        if not df.is_empty() and "symbol" in df.columns:
            symbols.update(
                normalize_symbol(symbol)
                for symbol in df["symbol"].drop_nulls().to_list()
            )
    return sorted(symbols)


def _symbols_from_fill_events(fill_events: pl.DataFrame) -> set[str]:
    if fill_events.is_empty() or "inst_id" not in fill_events.columns:
        return set()
    return {
        normalize_symbol(symbol)
        for symbol in fill_events["inst_id"].drop_nulls().to_list()
        if str(symbol).strip()
    }


def _symbols_from_v5_trade_events(v5_trade_events: pl.DataFrame) -> set[str]:
    if v5_trade_events.is_empty():
        return set()
    symbols: set[str] = set()
    for row in v5_trade_events.to_dicts():
        symbol = _symbol_from_trade_row(row)
        if symbol:
            symbols.add(symbol)
    return symbols


def _private_fill_events_for_day(root: Path, day: str) -> pl.DataFrame:
    silver = read_parquet_dataset(root / FILL_EVENT_DATASET)
    bronze = _bronze_private_fills_frame(read_parquet_dataset(root / BRONZE_FILLS_DATASET))
    return _filter_recent_window(
        _dedupe_frame(
            _concat_frames([silver, bronze]),
            key_columns=["venue", "inst_id", "trade_id", "order_id", "ts"],
        ),
        day=day,
        lookback_days=PRIVATE_COST_LOOKBACK_DAYS,
    )


def _private_account_bills_for_day(root: Path, day: str) -> pl.DataFrame:
    silver = read_parquet_dataset(root / ACCOUNT_BILL_DATASET)
    bronze = _bronze_private_bills_frame(read_parquet_dataset(root / BRONZE_BILLS_DATASET))
    return _filter_recent_window(
        _dedupe_frame(
            _concat_frames([silver, bronze]),
            key_columns=["venue", "bill_id", "ccy", "ts"],
        ),
        day=day,
        lookback_days=PRIVATE_COST_LOOKBACK_DAYS,
    )


def _v5_trade_events_for_day(root: Path, day: str) -> pl.DataFrame:
    return _filter_recent_window(
        read_parquet_dataset(root / V5_TRADE_EVENT_DATASET),
        day=day,
        lookback_days=PRIVATE_COST_LOOKBACK_DAYS,
        timestamp_columns=("ts_utc", "ts", "timestamp", "time", "created_at"),
    )


def _filter_recent_window(
    df: pl.DataFrame,
    *,
    day: str,
    lookback_days: int,
    timestamp_columns: tuple[str, ...] = ("ts",),
) -> pl.DataFrame:
    if df.is_empty():
        return df
    available = [column for column in timestamp_columns if column in df.columns]
    if not available:
        return df
    end = datetime.fromisoformat(day).replace(tzinfo=UTC) + timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    rows = [
        row
        for row in df.to_dicts()
        if (ts := _first_parseable_timestamp(row, available)) is not None and start <= ts < end
    ]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, schema=df.schema, orient="row")


def _first_parseable_timestamp(
    row: dict[str, Any],
    timestamp_columns: Sequence[str],
) -> datetime | None:
    for column in timestamp_columns:
        parsed = _parse_utc_timestamp(row.get(column))
        if parsed is not None:
            return parsed
    return None


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _uses_private_fill_lookback(samples: list[dict[str, Any]], day: str) -> bool:
    for sample in samples:
        ts = _parse_utc_timestamp(sample.get("ts"))
        if ts is not None and ts.date().isoformat() != day:
            return True
    return False


def _bronze_private_fills_frame(bronze: pl.DataFrame) -> pl.DataFrame:
    raw_items = _raw_items_from_bronze(bronze)
    if not raw_items:
        return pl.DataFrame()
    try:
        rows = [record.model_dump(mode="json") for record in normalize_okx_fills(raw_items)]
    except (KeyError, TypeError, ValueError):
        return pl.DataFrame()
    return pl.DataFrame(rows)


def _bronze_private_bills_frame(bronze: pl.DataFrame) -> pl.DataFrame:
    raw_items = _raw_items_from_bronze(bronze)
    if not raw_items:
        return pl.DataFrame()
    try:
        rows = [record.model_dump(mode="json") for record in normalize_okx_bills(raw_items)]
    except (KeyError, TypeError, ValueError):
        return pl.DataFrame()
    return pl.DataFrame(rows)


def _raw_items_from_bronze(bronze: pl.DataFrame) -> list[dict[str, Any]]:
    if bronze.is_empty() or "raw_json" not in bronze.columns:
        return []
    items: list[dict[str, Any]] = []
    for raw_json in bronze["raw_json"].drop_nulls().to_list():
        try:
            loaded = json.loads(str(raw_json))
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            items.append(loaded)
    return items


def _concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    usable = [frame for frame in frames if not frame.is_empty()]
    if not usable:
        return pl.DataFrame()
    return pl.concat(usable, how="diagonal_relaxed")


def _dedupe_frame(df: pl.DataFrame, key_columns: list[str]) -> pl.DataFrame:
    if df.is_empty():
        return df
    keys = [column for column in key_columns if column in df.columns]
    if not keys:
        return df
    return df.unique(subset=keys, keep="last", maintain_order=True)


def _fee_missing_count(fill_events: pl.DataFrame) -> int:
    if fill_events.is_empty() or "fee" not in fill_events.columns:
        return 0
    return fill_events.filter(pl.col("fee").is_null()).height


def _v5_trade_fee_missing_count(v5_trade_events: pl.DataFrame) -> int:
    if v5_trade_events.is_empty():
        return 0
    count = 0
    for row in v5_trade_events.to_dicts():
        if _first_float(row, ["fee_usdt", "fee_abs_usdt", "fee", "commission", "fee_abs"]) is None:
            count += 1
    return count


def _cost_bucket_daily_frame(rows: Sequence[CostBucketDaily]) -> pl.DataFrame:
    return pl.DataFrame(
        [row.model_dump(mode="json") for row in rows],
        schema=COST_BUCKET_DAILY_SCHEMA,
        orient="row",
    )


def _percentiles_or_zero(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return (
        _percentile(values, 0.50),
        _percentile(values, 0.75),
        _percentile(values, 0.90),
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return float(sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight)


def _notional_bucket(notional: float) -> str:
    if notional < 1_000:
        return "0-1k"
    if notional < 10_000:
        return "1k-10k"
    if notional < 100_000:
        return "10k-100k"
    return "100k+"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _fee_abs_usdt(
    *,
    fee: float | None,
    fee_currency: str,
    symbol: str,
    fill_price: float | None,
) -> float | None:
    if fee is None:
        return None
    normalized_currency = fee_currency.upper().strip()
    base, quote = _symbol_parts(symbol)
    fee_abs = abs(fee)
    if normalized_currency in {"", quote, "USDT", "USDC", "USD"}:
        return fee_abs
    if normalized_currency == base and fill_price is not None:
        return fee_abs * fill_price
    return fee_abs


def _symbol_parts(symbol: str) -> tuple[str, str]:
    normalized = normalize_symbol(symbol)
    if "-" not in normalized:
        return normalized, ""
    base, quote = normalized.split("-", 1)
    return base, quote


def _actual_fill_source(
    *,
    samples: list[dict[str, Any]],
    fee_missing: bool,
    sample_too_small: bool,
    slippage_unknown: bool,
) -> str:
    source_kinds = {str(sample.get("source_kind") or "") for sample in samples}
    if fee_missing:
        return "actual_okx_fills_fee_missing"
    if slippage_unknown or sample_too_small or "v5_trades_csv" in source_kinds:
        return "mixed_actual_proxy"
    return "actual_fills"
