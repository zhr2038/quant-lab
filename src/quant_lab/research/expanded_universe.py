from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

MARKET_BAR_DATASET = Path("silver") / "market_bar"
ORDERBOOK_SNAPSHOT_DATASET = Path("silver") / "orderbook_snapshot"
STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"
PULLBACK_BY_SYMBOL_DATASET = Path("gold") / "v5_entry_quality_history_pullback_by_symbol"
LATE_ENTRY_BY_SYMBOL_DATASET = Path("gold") / "v5_late_entry_chase_threshold_by_symbol"

EXPANDED_UNIVERSE_SHADOW_DATASET = Path("gold") / "expanded_crypto_universe_shadow"
SYMBOL_QUALITY_SCORE_DATASET = Path("gold") / "symbol_quality_score"
EXPANDED_CANDIDATE_OUTCOMES_DATASET = (
    Path("gold") / "expanded_crypto_candidate_outcomes_by_symbol"
)
EXPANDED_RECOMMENDATIONS_DATASET = Path("gold") / "expanded_crypto_recommendations"

SOURCE_NAME = "research.expanded_crypto_universe_shadow.v0.1"
SCHEMA_VERSION = "expanded_crypto_universe_shadow.v0.1"
RECOMMENDATION_SCHEMA_VERSION = "expanded_crypto_recommendations.v0.1"
CURRENT_V5_UNIVERSE = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"}
STABLE_BASES = {
    "USDT",
    "USDC",
    "USD",
    "DAI",
    "FDUSD",
    "TUSD",
    "USDD",
    "USDE",
    "PYUSD",
    "BUSD",
}
MEME_BASES = {
    "DOGE",
    "SHIB",
    "PEPE",
    "FLOKI",
    "BONK",
    "WIF",
    "MEME",
    "TURBO",
    "BABYDOGE",
    "AIDOGE",
}
LEVERAGED_SUFFIXES = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")

QUALITY_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "symbol": pl.Utf8,
    "quote_volume_24h": pl.Float64,
    "avg_spread_bps": pl.Float64,
    "min_notional_ok": pl.Boolean,
    "data_coverage": pl.Float64,
    "avg_24h_net_bps": pl.Float64,
    "avg_48h_net_bps": pl.Float64,
    "win_rate_24h": pl.Float64,
    "win_rate_48h": pl.Float64,
    "f3_dominant_negative_score": pl.Float64,
    "f4_confirmed_win_rate": pl.Float64,
    "f5_confirmed_win_rate": pl.Float64,
    "pullback_shadow_avg_24h": pl.Float64,
    "late_chase_loss_rate": pl.Float64,
    "negative_expectancy_bps": pl.Float64,
    "btc_correlation": pl.Float64,
    "quality_score": pl.Float64,
    "recommendation": pl.Utf8,
    "blocking_reasons": pl.Utf8,
    "source": pl.Utf8,
}

OUTCOME_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "horizon_hours": pl.Int64,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "decision": pl.Utf8,
    "source": pl.Utf8,
}

SHADOW_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "rank": pl.Int64,
    "symbol": pl.Utf8,
    "is_current_v5_symbol": pl.Boolean,
    "quote_volume_24h": pl.Float64,
    "avg_spread_bps": pl.Float64,
    "data_coverage": pl.Float64,
    "btc_correlation": pl.Float64,
    "quality_score": pl.Float64,
    "recommendation": pl.Utf8,
    "blocking_reasons": pl.Utf8,
    "min_shadow_days_required": pl.Int64,
    "notes": pl.Utf8,
    "source": pl.Utf8,
}

RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "top_symbols_json": pl.Utf8,
    "candidate_replace_eth_json": pl.Utf8,
    "candidate_replace_bnb_json": pl.Utf8,
    "current_universe_json": pl.Utf8,
    "warnings_json": pl.Utf8,
    "min_stable_output_days": pl.Int64,
    "source": pl.Utf8,
}


class ExpandedUniverseBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    quality_rows: int = Field(ge=0)
    shadow_rows: int = Field(ge=0)
    outcome_rows: int = Field(ge=0)
    recommendation_rows: int = Field(ge=0)
    top_symbols: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_expanded_crypto_universe_shadow(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
    max_candidates: int = 30,
    min_quote_volume_24h: float = 1_000_000.0,
    max_spread_bps: float = 20.0,
    min_coverage_bars: int = 24 * 30,
    min_price: float = 0.01,
    blacklist: list[str] | None = None,
) -> ExpandedUniverseBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    generated_at = datetime.now(UTC)
    market = read_parquet_dataset(root / MARKET_BAR_DATASET)
    orderbook = read_parquet_dataset(root / ORDERBOOK_SNAPSHOT_DATASET)
    evidence = read_parquet_dataset(root / STRATEGY_EVIDENCE_DATASET)
    pullback = read_parquet_dataset(root / PULLBACK_BY_SYMBOL_DATASET)
    late_entry = read_parquet_dataset(root / LATE_ENTRY_BY_SYMBOL_DATASET)

    warnings: list[str] = []
    if market.is_empty():
        warnings.append("market_bar_empty")
    if orderbook.is_empty():
        warnings.append("orderbook_snapshot_empty")
    if evidence.is_empty():
        warnings.append("strategy_evidence_empty")

    quality = build_symbol_quality_score(
        market_bars=market,
        orderbook_snapshots=orderbook,
        strategy_evidence=evidence,
        pullback_by_symbol=pullback,
        late_entry_by_symbol=late_entry,
        as_of_date=day,
        generated_at=generated_at,
        min_quote_volume_24h=min_quote_volume_24h,
        max_spread_bps=max_spread_bps,
        min_coverage_bars=min_coverage_bars,
        min_price=min_price,
        blacklist=blacklist or [],
    )
    outcomes = build_expanded_candidate_outcomes_by_symbol(
        strategy_evidence=evidence,
        as_of_date=day,
        generated_at=generated_at,
    )
    shadow = build_expanded_universe_shadow(
        quality,
        as_of_date=day,
        generated_at=generated_at,
        max_candidates=max_candidates,
    )
    recommendations = build_expanded_crypto_recommendations(
        quality,
        shadow,
        as_of_date=day,
        generated_at=generated_at,
        warnings=warnings,
    )

    write_parquet_dataset(quality, root / SYMBOL_QUALITY_SCORE_DATASET)
    write_parquet_dataset(outcomes, root / EXPANDED_CANDIDATE_OUTCOMES_DATASET)
    write_parquet_dataset(shadow, root / EXPANDED_UNIVERSE_SHADOW_DATASET)
    write_parquet_dataset(recommendations, root / EXPANDED_RECOMMENDATIONS_DATASET)

    return ExpandedUniverseBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        quality_rows=quality.height,
        shadow_rows=shadow.height,
        outcome_rows=outcomes.height,
        recommendation_rows=recommendations.height,
        top_symbols=shadow["symbol"].head(max_candidates).to_list()
        if not shadow.is_empty()
        else [],
        warnings=warnings,
    )


def build_symbol_quality_score(
    *,
    market_bars: pl.DataFrame,
    orderbook_snapshots: pl.DataFrame,
    strategy_evidence: pl.DataFrame,
    pullback_by_symbol: pl.DataFrame,
    late_entry_by_symbol: pl.DataFrame,
    as_of_date: date,
    generated_at: datetime | None = None,
    min_quote_volume_24h: float = 1_000_000.0,
    max_spread_bps: float = 20.0,
    min_coverage_bars: int = 24 * 30,
    min_price: float = 0.01,
    blacklist: list[str] | None = None,
) -> pl.DataFrame:
    if market_bars.is_empty():
        return pl.DataFrame(schema=QUALITY_SCHEMA)
    generated = generated_at or datetime.now(UTC)
    barred = {normalize_symbol(symbol) for symbol in (blacklist or [])}
    bars_by_symbol = _market_rows_by_symbol(market_bars)
    latest_ts = _latest_market_ts(bars_by_symbol)
    end = latest_ts or datetime.combine(as_of_date + timedelta(days=1), time.min, tzinfo=UTC)
    start_24h = end - timedelta(hours=24)
    start_30d = end - timedelta(days=30)
    spreads = _avg_spread_by_symbol(orderbook_snapshots, since=start_24h)
    evidence_metrics = _strategy_evidence_metrics(strategy_evidence)
    pullback_metrics = _pullback_metrics(pullback_by_symbol)
    late_metrics = _late_entry_metrics(late_entry_by_symbol)
    btc_returns = _returns_by_ts(bars_by_symbol.get("BTC-USDT", []), since=start_30d)

    rows: list[dict[str, Any]] = []
    for symbol, rows_for_symbol in sorted(bars_by_symbol.items()):
        base, quote = _symbol_parts(symbol)
        latest_close = _latest_close(rows_for_symbol)
        quote_volume_24h = _quote_volume(rows_for_symbol, since=start_24h)
        coverage_count = _coverage_count(rows_for_symbol, since=start_30d)
        data_coverage = min(coverage_count / max(min_coverage_bars, 1), 1.0)
        avg_spread = spreads.get(symbol)
        metrics = evidence_metrics.get(symbol, {})
        blocking = _blocking_reasons(
            symbol=symbol,
            base=base,
            quote=quote,
            latest_close=latest_close,
            quote_volume_24h=quote_volume_24h,
            avg_spread_bps=avg_spread,
            data_coverage=data_coverage,
            min_quote_volume_24h=min_quote_volume_24h,
            max_spread_bps=max_spread_bps,
            min_price=min_price,
            blacklisted=symbol in barred,
        )
        btc_correlation = _correlation(
            _returns_by_ts(rows_for_symbol, since=start_30d),
            btc_returns,
        )
        quality_score = _quality_score(
            quote_volume_24h=quote_volume_24h,
            avg_spread_bps=avg_spread,
            data_coverage=data_coverage,
            avg_24h_net_bps=_float(metrics.get("avg_24h_net_bps")),
            avg_48h_net_bps=_float(metrics.get("avg_48h_net_bps")),
            win_rate_24h=_float(metrics.get("win_rate_24h")),
            win_rate_48h=_float(metrics.get("win_rate_48h")),
            btc_correlation=btc_correlation,
            blocking_reasons=blocking,
            min_quote_volume_24h=min_quote_volume_24h,
            max_spread_bps=max_spread_bps,
        )
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": SCHEMA_VERSION,
                "symbol": symbol,
                "quote_volume_24h": quote_volume_24h,
                "avg_spread_bps": avg_spread,
                "min_notional_ok": latest_close is not None and latest_close >= min_price,
                "data_coverage": data_coverage,
                "avg_24h_net_bps": _float(metrics.get("avg_24h_net_bps")),
                "avg_48h_net_bps": _float(metrics.get("avg_48h_net_bps")),
                "win_rate_24h": _float(metrics.get("win_rate_24h")),
                "win_rate_48h": _float(metrics.get("win_rate_48h")),
                "f3_dominant_negative_score": _float(
                    metrics.get("f3_dominant_negative_score")
                ),
                "f4_confirmed_win_rate": _float(metrics.get("f4_confirmed_win_rate")),
                "f5_confirmed_win_rate": _float(metrics.get("f5_confirmed_win_rate")),
                "pullback_shadow_avg_24h": _float(
                    pullback_metrics.get(symbol, {}).get("pullback_shadow_avg_24h")
                ),
                "late_chase_loss_rate": _float(
                    late_metrics.get(symbol, {}).get("late_chase_loss_rate")
                ),
                "negative_expectancy_bps": _negative_expectancy_bps(metrics),
                "btc_correlation": btc_correlation,
                "quality_score": quality_score,
                "recommendation": _recommendation(symbol, blocking, quality_score, metrics),
                "blocking_reasons": safe_json_dumps(blocking),
                "source": SOURCE_NAME,
            }
        )
    if not rows:
        return pl.DataFrame(schema=QUALITY_SCHEMA)
    return (
        pl.DataFrame(rows, schema=QUALITY_SCHEMA, orient="row")
        .sort(["quality_score", "quote_volume_24h"], descending=[True, True])
    )


def build_expanded_candidate_outcomes_by_symbol(
    *,
    strategy_evidence: pl.DataFrame,
    as_of_date: date,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    if strategy_evidence.is_empty():
        return pl.DataFrame(schema=OUTCOME_SCHEMA)
    generated = generated_at or datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for row in strategy_evidence.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if not _is_usdt_symbol(symbol):
            continue
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": SCHEMA_VERSION,
                "symbol": symbol,
                "strategy_candidate": str(row.get("strategy_candidate") or ""),
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "sample_count": _int(row.get("sample_count")) or 0,
                "complete_sample_count": _int(row.get("complete_sample_count")) or 0,
                "avg_net_bps": _float(row.get("avg_net_bps")),
                "p25_net_bps": _float(row.get("p25_net_bps")),
                "win_rate": _float(row.get("win_rate")),
                "cost_source_mix": str(row.get("cost_source_mix") or ""),
                "decision": str(row.get("decision") or "RESEARCH_ONLY"),
                "source": SOURCE_NAME,
            }
        )
    if not rows:
        return pl.DataFrame(schema=OUTCOME_SCHEMA)
    return pl.DataFrame(rows, schema=OUTCOME_SCHEMA, orient="row").sort(
        ["symbol", "strategy_candidate", "horizon_hours"]
    )


def build_expanded_universe_shadow(
    quality: pl.DataFrame,
    *,
    as_of_date: date,
    generated_at: datetime | None = None,
    max_candidates: int = 30,
) -> pl.DataFrame:
    if quality.is_empty():
        return pl.DataFrame(schema=SHADOW_SCHEMA)
    generated = generated_at or datetime.now(UTC)
    candidate_rows = [
        row
        for row in quality.to_dicts()
        if str(row.get("recommendation") or "").startswith("candidate_replace_")
        or str(row.get("recommendation") or "") in {"shadow_only", "keep_current"}
    ]
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            float(row.get("quality_score") or 0.0),
            float(row.get("quote_volume_24h") or 0.0),
        ),
        reverse=True,
    )[: max(max_candidates, 1)]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_rows, start=1):
        symbol = str(row.get("symbol") or "")
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": SCHEMA_VERSION,
                "rank": index,
                "symbol": symbol,
                "is_current_v5_symbol": symbol in CURRENT_V5_UNIVERSE,
                "quote_volume_24h": _float(row.get("quote_volume_24h")),
                "avg_spread_bps": _float(row.get("avg_spread_bps")),
                "data_coverage": _float(row.get("data_coverage")),
                "btc_correlation": _float(row.get("btc_correlation")),
                "quality_score": _float(row.get("quality_score")),
                "recommendation": str(row.get("recommendation") or ""),
                "blocking_reasons": str(row.get("blocking_reasons") or "[]"),
                "min_shadow_days_required": 7,
                "notes": _shadow_notes(row),
                "source": SOURCE_NAME,
            }
        )
    if not rows:
        return pl.DataFrame(schema=SHADOW_SCHEMA)
    return pl.DataFrame(rows, schema=SHADOW_SCHEMA, orient="row")


def build_expanded_crypto_recommendations(
    quality: pl.DataFrame,
    shadow: pl.DataFrame,
    *,
    as_of_date: date,
    generated_at: datetime | None = None,
    warnings: list[str] | None = None,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    shadow_rows = shadow.to_dicts() if not shadow.is_empty() else []
    quality_rows = quality.to_dicts() if not quality.is_empty() else []
    by_symbol = {str(row.get("symbol") or ""): row for row in quality_rows}
    row = {
        "as_of_date": as_of_date.isoformat(),
        "generated_at": generated,
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "top_symbols_json": safe_json_dumps(
            [
                {
                    "symbol": row.get("symbol"),
                    "recommendation": row.get("recommendation"),
                    "quality_score": row.get("quality_score"),
                    "blocking_reasons": _loads_list(row.get("blocking_reasons")),
                }
                for row in shadow_rows[:30]
            ]
        ),
        "candidate_replace_eth_json": safe_json_dumps(
            _replacement_candidates(shadow_rows, "candidate_replace_eth")
        ),
        "candidate_replace_bnb_json": safe_json_dumps(
            _replacement_candidates(shadow_rows, "candidate_replace_bnb")
        ),
        "current_universe_json": safe_json_dumps(
            {symbol: by_symbol.get(symbol, {}) for symbol in sorted(CURRENT_V5_UNIVERSE)}
        ),
        "warnings_json": safe_json_dumps(warnings or []),
        "min_stable_output_days": 7,
        "source": SOURCE_NAME,
    }
    return pl.DataFrame([row], schema=RECOMMENDATION_SCHEMA, orient="row")


def _market_rows_by_symbol(market: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    rows_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in market.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if not _is_usdt_symbol(symbol):
            continue
        ts = _parse_dt(row.get("ts"))
        close = _float(row.get("close"))
        if ts is None or close is None or close <= 0:
            continue
        rows_by_symbol[symbol].append(row | {"symbol": symbol, "ts": ts})
    for rows in rows_by_symbol.values():
        rows.sort(key=lambda item: item["ts"])
    return dict(rows_by_symbol)


def _latest_market_ts(rows_by_symbol: dict[str, list[dict[str, Any]]]) -> datetime | None:
    values = [rows[-1]["ts"] for rows in rows_by_symbol.values() if rows]
    return max(values) if values else None


def _quote_volume(rows: list[dict[str, Any]], *, since: datetime) -> float:
    total = 0.0
    for row in rows:
        ts = row.get("ts")
        if not isinstance(ts, datetime) or ts < since:
            continue
        quote_volume = _float(row.get("quote_volume"))
        if quote_volume is None:
            close = _float(row.get("close"))
            volume = _float(row.get("volume"))
            quote_volume = close * volume if close is not None and volume is not None else 0.0
        total += max(quote_volume, 0.0)
    return total


def _coverage_count(rows: list[dict[str, Any]], *, since: datetime) -> int:
    return len(
        {
            row["ts"].replace(minute=0, second=0, microsecond=0)
            for row in rows
            if isinstance(row.get("ts"), datetime) and row["ts"] >= since
        }
    )


def _latest_close(rows: list[dict[str, Any]]) -> float | None:
    return _float(rows[-1].get("close")) if rows else None


def _returns_by_ts(rows: list[dict[str, Any]], *, since: datetime) -> dict[datetime, float]:
    returns: dict[datetime, float] = {}
    previous_close: float | None = None
    for row in rows:
        ts = row.get("ts")
        close = _float(row.get("close"))
        if not isinstance(ts, datetime) or close is None:
            continue
        if previous_close is not None and previous_close > 0 and ts >= since:
            returns[ts] = close / previous_close - 1.0
        previous_close = close
    return returns


def _avg_spread_by_symbol(orderbook: pl.DataFrame, *, since: datetime) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    if orderbook.is_empty():
        return {}
    for row in orderbook.to_dicts():
        ts = _parse_dt(row.get("ts") or row.get("ingest_ts"))
        if ts is None or ts < since:
            continue
        symbol = normalize_symbol(row.get("symbol"))
        if not _is_usdt_symbol(symbol):
            continue
        bid = _best_price(row.get("bids_json"), best="bid")
        ask = _best_price(row.get("asks_json"), best="ask")
        if bid is None or ask is None or ask <= bid:
            continue
        mid = (bid + ask) / 2.0
        if mid <= 0:
            continue
        values[symbol].append((ask - bid) / mid * 10_000.0)
    return {symbol: statistics.fmean(items) for symbol, items in values.items() if items}


def _strategy_evidence_metrics(strategy_evidence: pl.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = defaultdict(dict)
    if strategy_evidence.is_empty():
        return {}
    rows_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strategy_evidence.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if _is_usdt_symbol(symbol):
            rows_by_symbol[symbol].append(row)
    for symbol, rows in rows_by_symbol.items():
        for horizon in (24, 48):
            horizon_rows = [row for row in rows if _int(row.get("horizon_hours")) == horizon]
            if horizon_rows:
                metrics[symbol][f"avg_{horizon}h_net_bps"] = _weighted_average(
                    horizon_rows,
                    "avg_net_bps",
                )
                metrics[symbol][f"win_rate_{horizon}h"] = _weighted_average(
                    horizon_rows,
                    "win_rate",
                )
        f3_rows = [
            row
            for row in rows
            if "f3" in str(row.get("strategy_candidate") or "").lower()
        ]
        if f3_rows:
            negative_values = [
                abs(value)
                for row in f3_rows
                if (value := _float(row.get("avg_net_bps"))) is not None and value < 0
            ]
            metrics[symbol]["f3_dominant_negative_score"] = (
                statistics.fmean(negative_values) if negative_values else 0.0
            )
        for factor in ("f4", "f5"):
            factor_rows = [
                row
                for row in rows
                if factor in str(row.get("strategy_candidate") or "").lower()
            ]
            if factor_rows:
                metrics[symbol][f"{factor}_confirmed_win_rate"] = _weighted_average(
                    factor_rows,
                    "win_rate",
                )
    return dict(metrics)


def _pullback_metrics(pullback: pl.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = defaultdict(dict)
    if pullback.is_empty():
        return {}
    for row in pullback.to_dicts():
        if _int(row.get("horizon_hours")) != 24:
            continue
        symbol = normalize_symbol(row.get("symbol") or row.get("group_key"))
        value = _float(row.get("avg_net_bps"))
        if _is_usdt_symbol(symbol) and value is not None:
            metrics[symbol]["pullback_shadow_avg_24h"] = value
    return dict(metrics)


def _late_entry_metrics(late_entry: pl.DataFrame) -> dict[str, dict[str, float]]:
    if late_entry.is_empty():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in late_entry.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if _is_usdt_symbol(symbol):
            grouped[symbol].append(row)
    output: dict[str, dict[str, float]] = {}
    for symbol, rows in grouped.items():
        blocked = sum(_int(row.get("would_block_count")) or 0 for row in rows)
        losses = sum(_int(row.get("would_block_loss_count")) or 0 for row in rows)
        output[symbol] = {"late_chase_loss_rate": losses / blocked if blocked else 0.0}
    return output


def _blocking_reasons(
    *,
    symbol: str,
    base: str,
    quote: str,
    latest_close: float | None,
    quote_volume_24h: float,
    avg_spread_bps: float | None,
    data_coverage: float,
    min_quote_volume_24h: float,
    max_spread_bps: float,
    min_price: float,
    blacklisted: bool,
) -> list[str]:
    reasons: list[str] = []
    if quote != "USDT":
        reasons.append("not_usdt_spot")
    if base in STABLE_BASES:
        reasons.append("stablecoin")
    if base.endswith(LEVERAGED_SUFFIXES):
        reasons.append("leveraged_token")
    if base in MEME_BASES:
        reasons.append("high_risk_meme")
    if blacklisted:
        reasons.append("configured_blacklist")
    if latest_close is None or latest_close < min_price:
        reasons.append("dust_or_ultra_low_price")
    if quote_volume_24h < min_quote_volume_24h:
        reasons.append("low_quote_volume")
    if avg_spread_bps is None:
        reasons.append("spread_not_observed")
    elif avg_spread_bps > max_spread_bps:
        reasons.append("high_spread")
    if data_coverage < 1.0:
        reasons.append("insufficient_30d_1h_coverage")
    return reasons


def _recommendation(
    symbol: str,
    blocking_reasons: list[str],
    quality_score: float,
    metrics: dict[str, Any],
) -> str:
    blocking = set(blocking_reasons)
    if "low_quote_volume" in blocking:
        return "reject_low_liquidity"
    if "high_spread" in blocking or "spread_not_observed" in blocking:
        return "reject_high_spread"
    if blocking & {
        "stablecoin",
        "leveraged_token",
        "high_risk_meme",
        "dust_or_ultra_low_price",
        "configured_blacklist",
        "insufficient_30d_1h_coverage",
    }:
        return "shadow_only" if symbol in CURRENT_V5_UNIVERSE else "reject_negative_expectancy"
    if _float(metrics.get("f3_dominant_negative_score")) and (
        _float(metrics.get("f3_dominant_negative_score")) or 0.0
    ) > 100.0:
        return "reject_f3_noise"
    if (_float(metrics.get("avg_24h_net_bps")) or 0.0) < -25.0 and (
        _float(metrics.get("avg_48h_net_bps")) or 0.0
    ) < -25.0:
        return "reject_negative_expectancy"
    if symbol in CURRENT_V5_UNIVERSE:
        return "keep_current" if quality_score >= 45.0 else "shadow_only"
    if quality_score >= 70.0:
        return "candidate_replace_eth"
    if quality_score >= 60.0:
        return "candidate_replace_bnb"
    return "shadow_only"


def _quality_score(
    *,
    quote_volume_24h: float,
    avg_spread_bps: float | None,
    data_coverage: float,
    avg_24h_net_bps: float | None,
    avg_48h_net_bps: float | None,
    win_rate_24h: float | None,
    win_rate_48h: float | None,
    btc_correlation: float | None,
    blocking_reasons: list[str],
    min_quote_volume_24h: float,
    max_spread_bps: float,
) -> float:
    liquidity = min(quote_volume_24h / max(min_quote_volume_24h * 5.0, 1.0), 1.0) * 25.0
    spread = (
        max(0.0, 1.0 - (avg_spread_bps or max_spread_bps) / max(max_spread_bps, 1.0))
        * 20.0
    )
    coverage = max(min(data_coverage, 1.0), 0.0) * 20.0
    edge_values = [value for value in [avg_24h_net_bps, avg_48h_net_bps] if value is not None]
    edge = max(min((statistics.fmean(edge_values) if edge_values else 0.0) / 100.0, 1.0), -1.0)
    edge_score = (edge + 1.0) / 2.0 * 15.0
    win_values = [value for value in [win_rate_24h, win_rate_48h] if value is not None]
    win_score = max(min((statistics.fmean(win_values) if win_values else 0.5), 1.0), 0.0) * 10.0
    corr_abs = abs(btc_correlation) if btc_correlation is not None else 1.0
    diversification = max(0.0, 1.0 - corr_abs) * 10.0
    penalty = min(len(blocking_reasons) * 12.0, 40.0)
    raw_score = liquidity + spread + coverage + edge_score + win_score + diversification - penalty
    return round(max(0.0, raw_score), 4)


def _negative_expectancy_bps(metrics: dict[str, Any]) -> float | None:
    values = []
    for key in ["avg_24h_net_bps", "avg_48h_net_bps"]:
        value = _float(metrics.get(key))
        if value is not None and value < 0:
            values.append(value)
    return statistics.fmean(values) if values else 0.0


def _weighted_average(rows: list[dict[str, Any]], column: str) -> float | None:
    values: list[tuple[float, int]] = []
    for row in rows:
        value = _float(row.get(column))
        if value is None:
            continue
        weight = _int(row.get("complete_sample_count") or row.get("sample_count")) or 1
        values.append((value, max(weight, 1)))
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / total_weight


def _correlation(left: dict[datetime, float], right: dict[datetime, float]) -> float | None:
    keys = sorted(set(left) & set(right))
    if len(keys) < 5:
        return None
    left_values = [left[key] for key in keys]
    right_values = [right[key] for key in keys]
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_values, right_values, strict=True)
    )
    left_var = sum((a - left_mean) ** 2 for a in left_values)
    right_var = sum((b - right_mean) ** 2 for b in right_values)
    denom = math.sqrt(left_var * right_var)
    return None if denom == 0 else numerator / denom


def _best_price(raw_json: Any, *, best: str) -> float | None:
    if raw_json is None:
        return None
    try:
        levels = json.loads(str(raw_json))
    except json.JSONDecodeError:
        return None
    prices = [_float(level[0]) for level in levels if isinstance(level, list) and level]
    clean = [price for price in prices if price is not None and price > 0]
    if not clean:
        return None
    return max(clean) if best == "bid" else min(clean)


def _symbol_parts(symbol: str) -> tuple[str, str]:
    normalized = normalize_symbol(symbol)
    if "-" not in normalized:
        return normalized, ""
    base, quote = normalized.split("-", 1)
    return base, quote


def _is_usdt_symbol(symbol: str) -> bool:
    return bool(symbol and symbol.endswith("-USDT") and "-" in symbol)


def _shadow_notes(row: dict[str, Any]) -> str:
    recommendation = str(row.get("recommendation") or "")
    reasons = _loads_list(row.get("blocking_reasons"))
    if recommendation.startswith("candidate_replace_"):
        return "仅研究候选；至少连续 7 天稳定输出后再讨论替换 V5 币池。"
    if reasons:
        return "阻断原因：" + ",".join(str(reason) for reason in reasons)
    return "当前仅 shadow 观察，不影响 V5 实盘。"


def _replacement_candidates(
    rows: list[dict[str, Any]],
    recommendation: str,
) -> list[dict[str, Any]]:
    return [
        {
            "symbol": row.get("symbol"),
            "quality_score": row.get("quality_score"),
            "quote_volume_24h": row.get("quote_volume_24h"),
            "avg_spread_bps": row.get("avg_spread_bps"),
            "btc_correlation": row.get("btc_correlation"),
            "blocking_reasons": _loads_list(row.get("blocking_reasons")),
        }
        for row in rows
        if row.get("recommendation") == recommendation
    ]


def _loads_list(value: Any) -> list[Any]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_day(value: str | date | None) -> date:
    if value is None:
        return datetime.now(UTC).date()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value)[:10])


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
