import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.costs.model import DEFAULT_FALLBACK_COST_BPS, CostBucketDaily
from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset

FILL_EVENT_DATASET = Path("silver") / "fill_event"
ACCOUNT_BILL_DATASET = Path("silver") / "account_bill"
ORDERBOOK_SNAPSHOT_DATASET = Path("silver") / "orderbook_snapshot"
TRADE_PRINT_DATASET = Path("silver") / "trade_print"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"

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
    "cost_model_version": pl.Utf8,
    "created_at": pl.Utf8,
}


class CostCalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    day: str
    rows_written: int = Field(ge=0)
    dataset_path: str
    sources: list[str]


def calibrate_costs_for_day(lake_root: str | Path, day: str) -> CostCalibrationResult:
    root = Path(lake_root)
    rows = build_cost_bucket_daily_rows(
        fill_events=_filter_day(read_parquet_dataset(root / FILL_EVENT_DATASET), day),
        account_bills=_filter_day(read_parquet_dataset(root / ACCOUNT_BILL_DATASET), day),
        orderbook_snapshots=_filter_day(
            read_parquet_dataset(root / ORDERBOOK_SNAPSHOT_DATASET),
            day,
        ),
        trade_prints=_filter_day(read_parquet_dataset(root / TRADE_PRINT_DATASET), day),
        market_bars=_filter_day(read_parquet_dataset(root / MARKET_BAR_DATASET), day),
        day=day,
    )
    rows_written = publish_cost_bucket_daily(root, rows)
    return CostCalibrationResult(
        lake_root=str(root),
        day=day,
        rows_written=rows_written,
        dataset_path=str(root / COST_BUCKET_DAILY_DATASET),
        sources=sorted({row.source for row in rows}),
    )


def build_cost_bucket_daily_rows(
    fill_events: pl.DataFrame,
    account_bills: pl.DataFrame,
    orderbook_snapshots: pl.DataFrame,
    trade_prints: pl.DataFrame,
    market_bars: pl.DataFrame,
    day: str,
) -> list[CostBucketDaily]:
    spread_samples = _spread_samples_by_symbol(orderbook_snapshots)
    fill_samples = _fill_samples(fill_events)

    if fill_samples:
        return _actual_fill_rows(
            fill_samples=fill_samples,
            bills_present=not account_bills.is_empty(),
            spread_samples=spread_samples,
            day=day,
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
    return upsert_parquet_dataset(
        df,
        dataset_path,
        key_columns=["day", "symbol", "regime", "event_type", "notional_bucket"],
    )


def _actual_fill_rows(
    fill_samples: list[dict[str, Any]],
    bills_present: bool,
    spread_samples: dict[str, list[float]],
    day: str,
) -> list[CostBucketDaily]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in fill_samples:
        groups.setdefault((sample["symbol"], sample["notional_bucket"]), []).append(sample)

    rows: list[CostBucketDaily] = []
    for (symbol, notional_bucket), samples in groups.items():
        fee_samples = [sample["fee_bps"] for sample in samples if sample["fee_bps"] is not None]
        fee_missing = len(fee_samples) != len(samples)
        spread_values = spread_samples.get(symbol, [])
        spread_fallback = not spread_values

        source = (
            "actual_okx_fills_and_bills"
            if bills_present and not fee_missing
            else "actual_okx_fills_fee_missing"
        )
        fallback_parts = ["SLIPPAGE_UNKNOWN"]
        if not bills_present:
            fallback_parts.append("BILLS_MISSING")
        if fee_missing:
            fallback_parts.append("FEE_MISSING")
        if spread_fallback:
            fallback_parts.append("SPREAD_MISSING")
        else:
            fallback_parts.append("SPREAD_PROXY")

        fee_p50, fee_p75, fee_p90 = _percentiles_or_zero(fee_samples)
        spread_p50, spread_p75, spread_p90 = _percentiles_or_zero(spread_values)

        rows.append(
            CostBucketDaily(
                day=day,
                symbol=symbol,
                regime="realized",
                event_type="actual_fill",
                notional_bucket=notional_bucket,
                sample_count=len(samples),
                fee_bps_p50=fee_p50,
                fee_bps_p75=fee_p75,
                fee_bps_p90=fee_p90,
                slippage_bps_p50=0.0,
                slippage_bps_p75=0.0,
                slippage_bps_p90=0.0,
                spread_bps_p50=spread_p50,
                spread_bps_p75=spread_p75,
                spread_bps_p90=spread_p90,
                total_cost_bps_p50=fee_p50 + spread_p50,
                total_cost_bps_p75=fee_p75 + spread_p75,
                total_cost_bps_p90=fee_p90 + spread_p90,
                fallback_level=";".join(fallback_parts),
                source=source,
            )
        )
    return sorted(rows, key=lambda row: (row.symbol, row.notional_bucket))


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
    )


def _fill_samples(fill_events: pl.DataFrame) -> list[dict[str, Any]]:
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
        samples.append(
            {
                "symbol": str(row["inst_id"]),
                "notional": notional,
                "notional_bucket": _notional_bucket(notional),
                "fee_bps": abs(fee) / notional * 10_000 if fee is not None else None,
            }
        )
    return samples


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
        samples.setdefault(str(row["symbol"]), []).append((ask - bid) / mid * 10_000)
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


def _symbols_from_public_data(trade_prints: pl.DataFrame, market_bars: pl.DataFrame) -> list[str]:
    symbols: set[str] = set()
    for df in [trade_prints, market_bars]:
        if not df.is_empty() and "symbol" in df.columns:
            symbols.update(str(symbol) for symbol in df["symbol"].drop_nulls().to_list())
    return sorted(symbols)


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
