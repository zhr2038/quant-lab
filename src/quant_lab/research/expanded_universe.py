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

from quant_lab.data.lake import (
    read_parquet_dataset,
    upsert_parquet_dataset,
    write_parquet_dataset,
)
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

MARKET_BAR_DATASET = Path("silver") / "market_bar"
ORDERBOOK_SNAPSHOT_DATASET = Path("silver") / "orderbook_snapshot"
SPOT_UNIVERSE_CANDIDATES_DATASET = Path("bronze") / "okx_public_rest" / "spot_universe_candidates"
STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
PULLBACK_BY_SYMBOL_DATASET = Path("gold") / "v5_entry_quality_history_pullback_by_symbol"
LATE_ENTRY_BY_SYMBOL_DATASET = Path("gold") / "v5_late_entry_chase_threshold_by_symbol"

EXPANDED_UNIVERSE_SHADOW_DATASET = Path("gold") / "expanded_crypto_universe_shadow"
EXPANDED_UNIVERSE_CANDIDATE_DATASET = Path("gold") / "expanded_universe_candidate"
EXPANDED_UNIVERSE_QUALITY_DATASET = Path("gold") / "expanded_universe_quality"
EXPANDED_UNIVERSE_CANDIDATE_EVENT_DATASET = (
    Path("gold") / "expanded_universe_candidate_event"
)
EXPANDED_UNIVERSE_CANDIDATE_LABEL_DATASET = (
    Path("gold") / "expanded_universe_candidate_label"
)
EXPANDED_UNIVERSE_PROMOTION_QUEUE_DATASET = (
    Path("gold") / "expanded_universe_promotion_queue"
)
SYMBOL_QUALITY_SCORE_DATASET = Path("gold") / "symbol_quality_score"
EXPANDED_CANDIDATE_OUTCOMES_DATASET = (
    Path("gold") / "expanded_crypto_candidate_outcomes_by_symbol"
)
EXPANDED_RECOMMENDATIONS_DATASET = Path("gold") / "expanded_crypto_recommendations"
EXPANDED_UNIVERSE_WATCHLIST_DATASET = Path("gold") / "expanded_universe_watchlist"
EXPANDED_UNIVERSE_CANDIDATE_MATURITY_DATASET = (
    Path("gold") / "expanded_universe_candidate_maturity"
)

SOURCE_NAME = "research.expanded_crypto_universe_automation.v0.1"
SCHEMA_VERSION = "expanded_crypto_universe_shadow.v0.1"
AUTOMATION_SCHEMA_VERSION = "expanded_crypto_universe_automation.v0.1"
RECOMMENDATION_SCHEMA_VERSION = "expanded_crypto_recommendations.v0.1"
EXPANDED_MATURITY_SCHEMA_VERSION = "expanded_universe_candidate_maturity.v0.1"
EXPANDED_WATCHLIST_SCHEMA_VERSION = "expanded_universe_watchlist.v0.1"
EXPANDED_UNIVERSE_TYPE = "expanded_paper"
EXPANDED_PAPER_UNIVERSE_RECOMMENDATION = "candidate_for_expanded_paper_universe"
SHORT_HORIZONS = (4, 8, 12)
SEED_EXPANDED_SYMBOLS = (
    "TRX-USDT",
    "HYPE-USDT",
    "SUI-USDT",
    "XAUT-USDT",
    "PAXG-USDT",
    "ZEC-USDT",
)
QUALITY_WATCHLIST_SYMBOLS = (
    "TRX-USDT",
    "XAUT-USDT",
)
OUTCOME_WATCHLIST_SYMBOLS = ("NEAR-USDT", "WLD-USDT", "OKB-USDT")
REJECT_WATCHLIST_SYMBOLS = ("HYPE-USDT", "SUI-USDT", "ZEC-USDT", "FIL-USDT")
EXPANDED_STRATEGY_CANDIDATES = (
    "Alpha6Factor",
    "v5.f4_volume_expansion_entry",
    "v5.f3_dominant_entry",
    "v5.alt_impulse_shadow",
    "v5.pullback_reversal_shadow",
    "v5.late_entry_chase_shadow",
)
LABEL_HORIZONS = (4, 8, 12, 24, 48, 72)
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

CANDIDATE_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "symbol": pl.Utf8,
    "universe_type": pl.Utf8,
    "active_trading": pl.Boolean,
    "bar_coverage": pl.Float64,
    "spread_bps_p75": pl.Float64,
    "quote_volume_24h": pl.Float64,
    "min_notional_ok": pl.Boolean,
    "candidate_state": pl.Utf8,
    "blocking_reasons": pl.Utf8,
    "source": pl.Utf8,
}

CANDIDATE_EVENT_SCHEMA: dict[str, Any] = {
    "candidate_id": pl.Utf8,
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "symbol": pl.Utf8,
    "universe_type": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "final_score": pl.Float64,
    "f3": pl.Float64,
    "f4": pl.Float64,
    "f5": pl.Float64,
    "alpha6_score": pl.Float64,
    "alpha6_side": pl.Utf8,
    "cost_bps": pl.Float64,
    "cost_source": pl.Utf8,
    "regime_state": pl.Utf8,
    "risk_level": pl.Utf8,
    "replacement_target_candidate": pl.Utf8,
    "expansion_state": pl.Utf8,
    "source": pl.Utf8,
}

CANDIDATE_LABEL_SCHEMA: dict[str, Any] = {
    "candidate_id": pl.Utf8,
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "label_ts": pl.Datetime(time_zone="UTC"),
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "symbol": pl.Utf8,
    "universe_type": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "horizon_hours": pl.Int64,
    "entry_close": pl.Float64,
    "label_close": pl.Float64,
    "gross_bps": pl.Float64,
    "net_bps_after_cost": pl.Float64,
    "win": pl.Boolean,
    "mfe_bps": pl.Float64,
    "mae_bps": pl.Float64,
    "label_status": pl.Utf8,
    "cost_bps": pl.Float64,
    "cost_source": pl.Utf8,
    "replacement_target_candidate": pl.Utf8,
    "expansion_state": pl.Utf8,
    "source": pl.Utf8,
}

EXPANDED_STRATEGY_EVIDENCE_SCHEMA: dict[str, Any] = {
    "strategy": pl.Utf8,
    "evidence_version": pl.Utf8,
    "as_of_date": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "candidate_name": pl.Utf8,
    "source_type": pl.Utf8,
    "symbol": pl.Utf8,
    "universe_type": pl.Utf8,
    "replacement_target_candidate": pl.Utf8,
    "expansion_state": pl.Utf8,
    "regime_state": pl.Utf8,
    "horizon_hours": pl.Int64,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "median_net_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "start_ts": pl.Datetime(time_zone="UTC"),
    "end_ts": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

PROMOTION_QUEUE_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "universe_type": pl.Utf8,
    "promotion_state": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "horizon_hours": pl.Int64,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "live_block_reasons": pl.Utf8,
    "replacement_target_candidate": pl.Utf8,
    "expansion_state": pl.Utf8,
    "min_shadow_days_required": pl.Int64,
    "human_approval_required": pl.Boolean,
    "max_live_notional_usdt": pl.Float64,
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

MATURITY_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "universe_type": pl.Utf8,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "positive_short_horizon_count": pl.Int64,
    "positive_short_horizons": pl.Utf8,
    "best_short_horizon_hours": pl.Int64,
    "best_short_avg_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "p25_net_bps": pl.Float64,
    "maturity_state": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "maturity_reasons": pl.Utf8,
    "cost_source_mix": pl.Utf8,
    "source": pl.Utf8,
}

WATCHLIST_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "watchlist_type": pl.Utf8,
    "symbol": pl.Utf8,
    "quality_score": pl.Float64,
    "recommendation": pl.Utf8,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "positive_short_horizon_count": pl.Int64,
    "best_short_horizon_hours": pl.Int64,
    "best_short_avg_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "p25_net_bps": pl.Float64,
    "maturity_state": pl.Utf8,
    "watch_reason": pl.Utf8,
    "source": pl.Utf8,
}


class ExpandedUniverseBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    candidate_rows: int = Field(ge=0)
    quality_rows: int = Field(ge=0)
    event_rows: int = Field(ge=0)
    label_rows: int = Field(ge=0)
    strategy_evidence_rows: int = Field(ge=0)
    promotion_rows: int = Field(ge=0)
    maturity_rows: int = Field(ge=0)
    watchlist_rows: int = Field(ge=0)
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
    market, market_end = _read_recent_market_bars(root, min_coverage_bars=min_coverage_bars)
    orderbook = _read_recent_orderbook_snapshots(root, since=_orderbook_since(day, market_end))
    spot_candidates = read_parquet_dataset(root / SPOT_UNIVERSE_CANDIDATES_DATASET)
    evidence = read_parquet_dataset(root / STRATEGY_EVIDENCE_DATASET)
    costs = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET)
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
        spot_universe_candidates=spot_candidates,
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
    candidates = build_expanded_universe_candidates(
        quality,
        market_bars=market,
        spot_universe_candidates=spot_candidates,
        as_of_date=day,
        generated_at=generated_at,
    )
    current_events = build_expanded_candidate_events(
        candidates,
        market_bars=market,
        cost_bucket_daily=costs,
        as_of_date=day,
        generated_at=generated_at,
    )
    existing_events = read_parquet_dataset(root / EXPANDED_UNIVERSE_CANDIDATE_EVENT_DATASET)
    events = _dedupe_frame(
        _concat_optional(existing_events, current_events),
        key_columns=["candidate_id"],
    )
    labels = build_expanded_candidate_labels(
        events,
        market_bars=market,
        as_of_date=day,
        generated_at=generated_at,
    )
    expanded_evidence = build_expanded_strategy_evidence(
        labels,
        as_of_date=day,
        generated_at=generated_at,
    )
    maturity = build_expanded_universe_candidate_maturity(
        expanded_evidence,
        as_of_date=day,
        generated_at=generated_at,
    )
    promotion_queue = build_expanded_universe_promotion_queue(
        expanded_evidence,
        candidates=candidates,
        maturity=maturity,
        as_of_date=day,
        generated_at=generated_at,
    )
    outcomes = build_expanded_candidate_outcomes_by_symbol(
        strategy_evidence=_concat_optional(evidence, expanded_evidence),
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
    watchlist = build_expanded_universe_watchlist(
        quality,
        maturity,
        as_of_date=day,
        generated_at=generated_at,
    )

    write_parquet_dataset(candidates, root / EXPANDED_UNIVERSE_CANDIDATE_DATASET)
    write_parquet_dataset(quality, root / SYMBOL_QUALITY_SCORE_DATASET)
    write_parquet_dataset(quality, root / EXPANDED_UNIVERSE_QUALITY_DATASET)
    if not current_events.is_empty():
        upsert_parquet_dataset(
            current_events,
            root / EXPANDED_UNIVERSE_CANDIDATE_EVENT_DATASET,
            key_columns=["candidate_id"],
        )
    if not labels.is_empty():
        upsert_parquet_dataset(
            labels,
            root / EXPANDED_UNIVERSE_CANDIDATE_LABEL_DATASET,
            key_columns=["candidate_id", "horizon_hours"],
        )
    write_parquet_dataset(promotion_queue, root / EXPANDED_UNIVERSE_PROMOTION_QUEUE_DATASET)
    write_parquet_dataset(maturity, root / EXPANDED_UNIVERSE_CANDIDATE_MATURITY_DATASET)
    write_parquet_dataset(watchlist, root / EXPANDED_UNIVERSE_WATCHLIST_DATASET)
    if not expanded_evidence.is_empty():
        upsert_parquet_dataset(
            expanded_evidence,
            root / STRATEGY_EVIDENCE_DATASET,
            key_columns=[
                "as_of_date",
                "strategy_candidate",
                "symbol",
                "regime_state",
                "horizon_hours",
                "source_type",
                "universe_type",
            ],
        )
    write_parquet_dataset(outcomes, root / EXPANDED_CANDIDATE_OUTCOMES_DATASET)
    write_parquet_dataset(shadow, root / EXPANDED_UNIVERSE_SHADOW_DATASET)
    write_parquet_dataset(recommendations, root / EXPANDED_RECOMMENDATIONS_DATASET)

    return ExpandedUniverseBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        candidate_rows=candidates.height,
        quality_rows=quality.height,
        event_rows=events.height,
        label_rows=labels.height,
        strategy_evidence_rows=expanded_evidence.height,
        promotion_rows=promotion_queue.height,
        maturity_rows=maturity.height,
        watchlist_rows=watchlist.height,
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
    spot_universe_candidates: pl.DataFrame | None = None,
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
    spot_candidate_metrics = _spot_candidate_metrics(
        spot_universe_candidates if spot_universe_candidates is not None else pl.DataFrame()
    )
    evidence_metrics = _strategy_evidence_metrics(strategy_evidence)
    pullback_metrics = _pullback_metrics(pullback_by_symbol)
    late_metrics = _late_entry_metrics(late_entry_by_symbol)
    btc_returns = _returns_by_ts(bars_by_symbol.get("BTC-USDT", []), since=start_30d)

    rows: list[dict[str, Any]] = []
    for symbol, rows_for_symbol in sorted(bars_by_symbol.items()):
        base, quote = _symbol_parts(symbol)
        latest_close = _latest_close(rows_for_symbol)
        candidate_metrics = spot_candidate_metrics.get(symbol, {})
        quote_volume_24h = max(
            _quote_volume(rows_for_symbol, since=start_24h),
            _float(candidate_metrics.get("quote_volume_24h")) or 0.0,
        )
        coverage_count = _coverage_count(rows_for_symbol, since=start_30d)
        data_coverage = min(coverage_count / max(min_coverage_bars, 1), 1.0)
        avg_spread = spreads.get(symbol)
        if avg_spread is None:
            avg_spread = _float(candidate_metrics.get("avg_spread_bps"))
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


def build_expanded_universe_candidates(
    quality: pl.DataFrame,
    *,
    market_bars: pl.DataFrame,
    spot_universe_candidates: pl.DataFrame,
    as_of_date: date,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    quality_by_symbol = {
        normalize_symbol(row.get("symbol")): row for row in quality.to_dicts()
    }
    market_symbols = set(_market_rows_by_symbol(market_bars))
    spot_metrics = _spot_candidate_metrics(spot_universe_candidates)
    selected_symbols = sorted(set(SEED_EXPANDED_SYMBOLS) | set(quality_by_symbol))
    rows: list[dict[str, Any]] = []
    for symbol in selected_symbols:
        symbol = normalize_symbol(symbol)
        if not _is_usdt_symbol(symbol):
            continue
        quality_row = quality_by_symbol.get(symbol, {})
        spot = spot_metrics.get(symbol, {})
        blocking = _loads_list(quality_row.get("blocking_reasons"))
        has_market = symbol in market_symbols
        recommendation = str(quality_row.get("recommendation") or "")
        candidate_state = _candidate_state_from_quality(
            recommendation=recommendation,
            blocking_reasons=blocking,
            has_market=has_market,
        )
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": AUTOMATION_SCHEMA_VERSION,
                "symbol": symbol,
                "universe_type": EXPANDED_UNIVERSE_TYPE,
                "active_trading": has_market or symbol in spot_metrics,
                "bar_coverage": _float(quality_row.get("data_coverage")) or 0.0,
                "spread_bps_p75": _float(
                    quality_row.get("avg_spread_bps") or spot.get("avg_spread_bps")
                ),
                "quote_volume_24h": _float(
                    quality_row.get("quote_volume_24h") or spot.get("quote_volume_24h")
                )
                or 0.0,
                "min_notional_ok": bool(quality_row.get("min_notional_ok", has_market)),
                "candidate_state": candidate_state,
                "blocking_reasons": safe_json_dumps(blocking),
                "source": SOURCE_NAME,
            }
        )
    if not rows:
        return pl.DataFrame(schema=CANDIDATE_SCHEMA)
    return pl.DataFrame(rows, schema=CANDIDATE_SCHEMA, orient="row").sort(
        ["candidate_state", "symbol"]
    )


def build_expanded_candidate_events(
    candidates: pl.DataFrame,
    *,
    market_bars: pl.DataFrame,
    cost_bucket_daily: pl.DataFrame,
    as_of_date: date,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    if candidates.is_empty() or market_bars.is_empty():
        return pl.DataFrame(schema=CANDIDATE_EVENT_SCHEMA)
    generated = generated_at or datetime.now(UTC)
    bars_by_symbol = _market_rows_by_symbol(market_bars)
    costs_by_symbol = _latest_cost_by_symbol(cost_bucket_daily)
    candidate_context = {
        normalize_symbol(row.get("symbol")): row for row in candidates.to_dicts()
    }
    rows: list[dict[str, Any]] = []
    for symbol, context in sorted(candidate_context.items()):
        symbol_bars = bars_by_symbol.get(symbol, [])
        if not symbol_bars:
            continue
        latest = symbol_bars[-1]
        ts = _parse_dt(latest.get("ts"))
        if ts is None:
            continue
        factors = _expanded_factor_snapshot(symbol_bars)
        cost = costs_by_symbol.get(symbol, {})
        cost_bps = _float(cost.get("total_cost_bps_p75")) or _float(
            cost.get("selected_total_cost_bps")
        )
        cost_source = str(
            cost.get("cost_source") or cost.get("source") or "public_spread_proxy"
        )
        if cost_bps is None:
            cost_bps = 30.0
            cost_source = "conservative_default"
        for strategy_candidate in EXPANDED_STRATEGY_CANDIDATES:
            candidate_id = _expanded_candidate_id(symbol, strategy_candidate, ts)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "ts_utc": ts,
                    "generated_at": generated,
                    "schema_version": AUTOMATION_SCHEMA_VERSION,
                    "symbol": symbol,
                    "universe_type": EXPANDED_UNIVERSE_TYPE,
                    "strategy_candidate": strategy_candidate,
                    "final_score": _strategy_final_score(strategy_candidate, factors),
                    "f3": factors.get("f3"),
                    "f4": factors.get("f4"),
                    "f5": factors.get("f5"),
                    "alpha6_score": factors.get("alpha6_score"),
                    "alpha6_side": "long"
                    if (factors.get("alpha6_score") or 0.0) >= 0
                    else "short_shadow_only",
                    "cost_bps": cost_bps,
                    "cost_source": cost_source,
                    "regime_state": _expanded_regime_state(factors),
                    "risk_level": _expanded_risk_level(factors),
                    "replacement_target_candidate": _replacement_target(context),
                    "expansion_state": str(context.get("candidate_state") or "RESEARCH"),
                    "source": SOURCE_NAME,
                }
            )
    if not rows:
        return pl.DataFrame(schema=CANDIDATE_EVENT_SCHEMA)
    return pl.DataFrame(rows, schema=CANDIDATE_EVENT_SCHEMA, orient="row").sort(
        ["symbol", "strategy_candidate"]
    )


def build_expanded_candidate_labels(
    events: pl.DataFrame,
    *,
    market_bars: pl.DataFrame,
    as_of_date: date,
    generated_at: datetime | None = None,
    horizons: tuple[int, ...] = LABEL_HORIZONS,
) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(schema=CANDIDATE_LABEL_SCHEMA)
    generated = generated_at or datetime.now(UTC)
    bars_by_symbol = _market_rows_by_symbol(market_bars)
    rows: list[dict[str, Any]] = []
    for event in events.to_dicts():
        symbol = normalize_symbol(event.get("symbol"))
        symbol_bars = bars_by_symbol.get(symbol, [])
        event_ts = _parse_dt(event.get("ts_utc"))
        entry_bar = _bar_at_or_before(symbol_bars, event_ts)
        entry_close = _float(entry_bar.get("close")) if entry_bar else None
        if event_ts is None or entry_close is None or entry_close <= 0:
            continue
        for horizon in horizons:
            label_ts = event_ts + timedelta(hours=horizon)
            future_bar = _bar_at_or_after(symbol_bars, label_ts)
            window = _bars_between(symbol_bars, event_ts, label_ts)
            label_close = _float(future_bar.get("close")) if future_bar else None
            gross = (
                (label_close / entry_close - 1.0) * 10_000.0
                if label_close is not None and label_close > 0
                else None
            )
            cost_bps = _float(event.get("cost_bps")) or 0.0
            net = gross - cost_bps if gross is not None else None
            mfe, mae = _mfe_mae_bps(window, entry_close)
            rows.append(
                {
                    "candidate_id": str(event.get("candidate_id") or ""),
                    "ts_utc": event_ts,
                    "decision_ts": event_ts,
                    "label_ts": label_ts,
                    "generated_at": generated,
                    "schema_version": AUTOMATION_SCHEMA_VERSION,
                    "symbol": symbol,
                    "universe_type": EXPANDED_UNIVERSE_TYPE,
                    "strategy_candidate": str(event.get("strategy_candidate") or ""),
                    "horizon_hours": horizon,
                    "entry_close": entry_close,
                    "label_close": label_close,
                    "gross_bps": gross,
                    "net_bps_after_cost": net,
                    "win": net is not None and net > 0,
                    "mfe_bps": mfe,
                    "mae_bps": mae,
                    "label_status": "complete" if net is not None else "pending",
                    "cost_bps": cost_bps,
                    "cost_source": str(event.get("cost_source") or "unknown"),
                    "replacement_target_candidate": str(
                        event.get("replacement_target_candidate") or ""
                    ),
                    "expansion_state": str(event.get("expansion_state") or "RESEARCH"),
                    "source": SOURCE_NAME,
                }
            )
    if not rows:
        return pl.DataFrame(schema=CANDIDATE_LABEL_SCHEMA)
    return pl.DataFrame(rows, schema=CANDIDATE_LABEL_SCHEMA, orient="row").sort(
        ["symbol", "strategy_candidate", "horizon_hours"]
    )


def build_expanded_strategy_evidence(
    labels: pl.DataFrame,
    *,
    as_of_date: date,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    if labels.is_empty():
        return pl.DataFrame(schema=EXPANDED_STRATEGY_EVIDENCE_SCHEMA)
    generated = generated_at or datetime.now(UTC)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in labels.to_dicts():
        grouped[
            (
                str(row.get("strategy_candidate") or ""),
                normalize_symbol(row.get("symbol")),
                _int(row.get("horizon_hours")) or 0,
            )
        ].append(row)
    rows: list[dict[str, Any]] = []
    for (candidate, symbol, horizon), group_rows in sorted(grouped.items()):
        if not candidate or not _is_usdt_symbol(symbol):
            continue
        complete = [row for row in group_rows if row.get("label_status") == "complete"]
        net_values = [
            value
            for row in complete
            if (value := _float(row.get("net_bps_after_cost"))) is not None
        ]
        wins = [bool(row.get("win")) for row in complete if row.get("win") is not None]
        cost_mix = _cost_source_mix(group_rows)
        avg_net = statistics.fmean(net_values) if net_values else None
        p25 = _quantile(net_values, 0.25)
        win_rate = sum(wins) / len(wins) if wins else None
        decision, reasons = _expanded_decision(
            sample_count=len(group_rows),
            complete_sample_count=len(complete),
            avg_net_bps=avg_net,
            p25_net_bps=p25,
            win_rate=win_rate,
            cost_source_mix=cost_mix,
        )
        ts_values = [_parse_dt(row.get("ts_utc")) for row in group_rows]
        label_values = [_parse_dt(row.get("label_ts")) for row in group_rows]
        rows.append(
            {
                "strategy": "v5",
                "evidence_version": AUTOMATION_SCHEMA_VERSION,
                "as_of_date": as_of_date.isoformat(),
                "strategy_candidate": candidate,
                "candidate_name": candidate,
                "source_type": "expanded_universe_candidate_label",
                "symbol": symbol,
                "universe_type": EXPANDED_UNIVERSE_TYPE,
                "replacement_target_candidate": str(
                    group_rows[0].get("replacement_target_candidate") or ""
                ),
                "expansion_state": _promotion_state_from_decision(decision),
                "regime_state": "expanded_universe",
                "horizon_hours": horizon,
                "sample_count": len(group_rows),
                "complete_sample_count": len(complete),
                "avg_net_bps": avg_net,
                "median_net_bps": statistics.median(net_values) if net_values else None,
                "p25_net_bps": p25,
                "win_rate": win_rate,
                "cost_source_mix": safe_json_dumps(cost_mix),
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
                "start_ts": min([ts for ts in ts_values if ts], default=None),
                "end_ts": max([ts for ts in label_values if ts], default=None),
                "created_at": generated,
                "source": SOURCE_NAME,
            }
        )
    if not rows:
        return pl.DataFrame(schema=EXPANDED_STRATEGY_EVIDENCE_SCHEMA)
    return pl.DataFrame(rows, schema=EXPANDED_STRATEGY_EVIDENCE_SCHEMA, orient="row")


def build_expanded_universe_candidate_maturity(
    strategy_evidence: pl.DataFrame,
    *,
    as_of_date: date,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    if strategy_evidence.is_empty():
        return pl.DataFrame(schema=MATURITY_SCHEMA)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in strategy_evidence.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        candidate = str(row.get("strategy_candidate") or "")
        if _is_usdt_symbol(symbol) and candidate:
            grouped[(symbol, candidate)].append(row)

    rows: list[dict[str, Any]] = []
    for (symbol, candidate), group_rows in grouped.items():
        state, reasons, stats = _maturity_state_from_evidence_rows(group_rows)
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": EXPANDED_MATURITY_SCHEMA_VERSION,
                "symbol": symbol,
                "strategy_candidate": candidate,
                "universe_type": EXPANDED_UNIVERSE_TYPE,
                "sample_count": stats["sample_count"],
                "complete_sample_count": stats["complete_sample_count"],
                "positive_short_horizon_count": stats["positive_short_horizon_count"],
                "positive_short_horizons": safe_json_dumps(stats["positive_short_horizons"]),
                "best_short_horizon_hours": stats["best_short_horizon_hours"],
                "best_short_avg_net_bps": stats["best_short_avg_net_bps"],
                "win_rate": stats["win_rate"],
                "p25_net_bps": stats["p25_net_bps"],
                "maturity_state": state,
                "recommended_mode": _recommended_mode_from_maturity_state(state),
                "maturity_reasons": safe_json_dumps(reasons),
                "cost_source_mix": safe_json_dumps(
                    _merge_cost_source_mix(group_rows),
                ),
                "source": SOURCE_NAME,
            }
        )
    if not rows:
        return pl.DataFrame(schema=MATURITY_SCHEMA)
    return pl.DataFrame(rows, schema=MATURITY_SCHEMA, orient="row").sort(
        ["maturity_state", "symbol", "strategy_candidate"],
        descending=[False, False, False],
    )


def build_expanded_universe_watchlist(
    quality: pl.DataFrame,
    maturity: pl.DataFrame,
    *,
    as_of_date: date,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    quality_by_symbol = {
        normalize_symbol(row.get("symbol")): row for row in quality.to_dicts()
    } if not quality.is_empty() else {}
    maturity_by_symbol = _best_maturity_by_symbol(maturity)
    rows: list[dict[str, Any]] = []
    for watchlist_type, symbols in (
        ("quality_watchlist", QUALITY_WATCHLIST_SYMBOLS),
        ("outcome_watchlist", OUTCOME_WATCHLIST_SYMBOLS),
        ("reject_list", REJECT_WATCHLIST_SYMBOLS),
    ):
        for raw_symbol in symbols:
            symbol = normalize_symbol(raw_symbol)
            quality_row = quality_by_symbol.get(symbol, {})
            maturity_row = maturity_by_symbol.get(symbol, {})
            recommendation = str(quality_row.get("recommendation") or "")
            if watchlist_type == "reject_list":
                recommendation = "reject_low_priority_current_weak"
            rows.append(
                {
                    "as_of_date": as_of_date.isoformat(),
                    "generated_at": generated,
                    "schema_version": EXPANDED_WATCHLIST_SCHEMA_VERSION,
                    "watchlist_type": watchlist_type,
                    "symbol": symbol,
                    "quality_score": _float(quality_row.get("quality_score")),
                    "recommendation": recommendation,
                    "sample_count": _int(maturity_row.get("sample_count")) or 0,
                    "complete_sample_count": _int(
                        maturity_row.get("complete_sample_count")
                    ) or 0,
                    "positive_short_horizon_count": _int(
                        maturity_row.get("positive_short_horizon_count")
                    ) or 0,
                    "best_short_horizon_hours": _int(
                        maturity_row.get("best_short_horizon_hours")
                    ) or 0,
                    "best_short_avg_net_bps": _float(
                        maturity_row.get("best_short_avg_net_bps")
                    ),
                    "win_rate": _float(maturity_row.get("win_rate")),
                    "p25_net_bps": _float(maturity_row.get("p25_net_bps")),
                    "maturity_state": str(
                        maturity_row.get("maturity_state") or "RESEARCH"
                    ),
                    "watch_reason": _watch_reason(
                        watchlist_type=watchlist_type,
                        has_quality=bool(quality_row),
                        has_maturity=bool(maturity_row),
                    ),
                    "source": SOURCE_NAME,
                }
            )
    return pl.DataFrame(rows, schema=WATCHLIST_SCHEMA, orient="row")


def build_expanded_universe_promotion_queue(
    strategy_evidence: pl.DataFrame,
    *,
    candidates: pl.DataFrame,
    maturity: pl.DataFrame | None = None,
    as_of_date: date,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    if strategy_evidence.is_empty():
        return pl.DataFrame(schema=PROMOTION_QUEUE_SCHEMA)
    candidate_context = {
        normalize_symbol(row.get("symbol")): row for row in candidates.to_dicts()
    }
    maturity_context = {
        (
            normalize_symbol(row.get("symbol")),
            str(row.get("strategy_candidate") or ""),
        ): row
        for row in (maturity.to_dicts() if maturity is not None and not maturity.is_empty() else [])
    }
    rows: list[dict[str, Any]] = []
    for row in strategy_evidence.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        decision = str(row.get("decision") or "RESEARCH_ONLY").upper()
        candidate = str(row.get("strategy_candidate") or "")
        maturity_row = maturity_context.get((symbol, candidate), {})
        maturity_state = str(maturity_row.get("maturity_state") or "")
        promotion_state = (
            _promotion_state_from_maturity_state(maturity_state)
            if maturity_state
            else _promotion_state_from_decision(decision)
        )
        recommended_mode = _recommended_mode_from_promotion_state(promotion_state)
        context = candidate_context.get(symbol, {})
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": AUTOMATION_SCHEMA_VERSION,
                "symbol": symbol,
                "strategy_candidate": candidate,
                "universe_type": EXPANDED_UNIVERSE_TYPE,
                "promotion_state": promotion_state,
                "recommended_mode": recommended_mode,
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "sample_count": _int(row.get("sample_count")) or 0,
                "complete_sample_count": _int(row.get("complete_sample_count")) or 0,
                "avg_net_bps": _float(row.get("avg_net_bps")),
                "p25_net_bps": _float(row.get("p25_net_bps")),
                "win_rate": _float(row.get("win_rate")),
                "cost_source_mix": str(row.get("cost_source_mix") or "{}"),
                "live_block_reasons": safe_json_dumps(
                    ["expanded_universe_not_live_approved"]
                ),
                "replacement_target_candidate": str(
                    row.get("replacement_target_candidate")
                    or _replacement_target(context)
                ),
                "expansion_state": promotion_state,
                "min_shadow_days_required": 7,
                "human_approval_required": True,
                "max_live_notional_usdt": 0.0,
                "source": SOURCE_NAME,
            }
        )
    if not rows:
        return pl.DataFrame(schema=PROMOTION_QUEUE_SCHEMA)
    return pl.DataFrame(rows, schema=PROMOTION_QUEUE_SCHEMA, orient="row").sort(
        ["promotion_state", "symbol", "strategy_candidate", "horizon_hours"]
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
        if str(row.get("recommendation") or "")
        in {EXPANDED_PAPER_UNIVERSE_RECOMMENDATION, "shadow_only", "keep_current"}
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
    warning_values = list(warnings or [])
    warning_values.append("replacement_requires_strategy_evidence_and_paper_evidence")
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
        "warnings_json": safe_json_dumps(sorted(set(warning_values))),
        "min_stable_output_days": 7,
        "source": SOURCE_NAME,
    }
    return pl.DataFrame([row], schema=RECOMMENDATION_SCHEMA, orient="row")


def _concat_optional(*frames: pl.DataFrame) -> pl.DataFrame:
    clean = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(clean, how="diagonal_relaxed") if clean else pl.DataFrame()


def _dedupe_frame(frame: pl.DataFrame, *, key_columns: list[str]) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    keys = [column for column in key_columns if column in frame.columns]
    return frame.unique(subset=keys, keep="last", maintain_order=True) if keys else frame


def _candidate_state_from_quality(
    *,
    recommendation: str,
    blocking_reasons: list[Any],
    has_market: bool,
) -> str:
    if not has_market:
        return "DISCOVERED"
    hard_reasons = {
        "stablecoin",
        "leveraged_token",
        "high_risk_meme",
        "dust_or_ultra_low_price",
        "configured_blacklist",
    }
    if set(str(reason) for reason in blocking_reasons) & hard_reasons:
        return "RESEARCH"
    if recommendation == EXPANDED_PAPER_UNIVERSE_RECOMMENDATION:
        return "SHADOW"
    if recommendation in {"shadow_only", "keep_current"}:
        return "SHADOW"
    return "RESEARCH"


def _latest_cost_by_symbol(cost_bucket_daily: pl.DataFrame) -> dict[str, dict[str, Any]]:
    if cost_bucket_daily.is_empty() or "symbol" not in cost_bucket_daily.columns:
        return {}
    rows: list[tuple[datetime, dict[str, Any]]] = []
    for row in cost_bucket_daily.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if not _is_usdt_symbol(symbol):
            continue
        ts = _parse_dt(row.get("created_at") or row.get("as_of_ts"))
        if ts is None:
            day = str(row.get("day") or "").strip()
            try:
                ts = datetime.combine(date.fromisoformat(day[:10]), time.min, tzinfo=UTC)
            except ValueError:
                ts = datetime.min.replace(tzinfo=UTC)
        row = row | {"symbol": symbol}
        rows.append((ts, row))
    latest: dict[str, tuple[datetime, dict[str, Any]]] = {}
    source_rank = {
        "actual_fills": 4,
        "actual_okx_fills_and_bills": 4,
        "mixed_actual_proxy": 3,
        "public_spread_proxy": 2,
        "global_default": 1,
    }
    for ts, row in rows:
        symbol = str(row["symbol"])
        source = str(row.get("cost_source") or row.get("source") or "")
        rank = source_rank.get(source, 0)
        existing = latest.get(symbol)
        existing_source = (
            str(existing[1].get("cost_source") or existing[1].get("source") or "")
            if existing
            else ""
        )
        existing_rank = source_rank.get(existing_source, 0)
        if existing is None or (ts, rank) >= (existing[0], existing_rank):
            latest[symbol] = (ts, row)
    return {symbol: row for symbol, (_, row) in latest.items()}


def _expanded_factor_snapshot(rows: list[dict[str, Any]]) -> dict[str, float]:
    closes = [_float(row.get("close")) for row in rows]
    volumes = [_float(row.get("quote_volume") or row.get("volume")) for row in rows]
    clean_closes = [value for value in closes if value is not None and value > 0]
    clean_volumes = [value for value in volumes if value is not None and value >= 0]
    latest = clean_closes[-1] if clean_closes else 0.0
    prev = clean_closes[-2] if len(clean_closes) >= 2 else latest
    if len(clean_closes) >= 25:
        prev_24 = clean_closes[-25]
    elif clean_closes:
        prev_24 = clean_closes[0]
    else:
        prev_24 = latest
    return_1h = (latest / prev - 1.0) * 100.0 if prev else 0.0
    return_24h = (latest / prev_24 - 1.0) * 100.0 if prev_24 else 0.0
    recent_volume = clean_volumes[-1] if clean_volumes else 0.0
    mean_volume = statistics.fmean(clean_volumes[-24:]) if clean_volumes else 1.0
    f4 = recent_volume / mean_volume - 1.0 if mean_volume else 0.0
    f3 = return_24h
    f5 = return_1h
    alpha6 = (f3 * 0.45) + (f4 * 20.0) + (f5 * 0.35)
    return {
        "f3": round(f3, 6),
        "f4": round(f4, 6),
        "f5": round(f5, 6),
        "alpha6_score": round(alpha6, 6),
    }


def _strategy_final_score(strategy_candidate: str, factors: dict[str, float]) -> float:
    candidate = strategy_candidate.lower()
    if "f3" in candidate:
        return float(factors.get("f3") or 0.0)
    if "f4" in candidate:
        return float(factors.get("f4") or 0.0) * 100.0
    if "late_entry" in candidate:
        return max(float(factors.get("f3") or 0.0), 0.0)
    if "pullback" in candidate:
        return -abs(float(factors.get("f5") or 0.0))
    return float(factors.get("alpha6_score") or 0.0)


def _expanded_regime_state(factors: dict[str, float]) -> str:
    f3 = float(factors.get("f3") or 0.0)
    f5 = float(factors.get("f5") or 0.0)
    if f3 > 3.0 and f5 >= 0:
        return "risk_on_momentum"
    if f3 < -3.0:
        return "risk_off_pullback"
    return "neutral"


def _expanded_risk_level(factors: dict[str, float]) -> str:
    f3 = abs(float(factors.get("f3") or 0.0))
    f5 = abs(float(factors.get("f5") or 0.0))
    if max(f3, f5) > 8.0:
        return "high"
    if max(f3, f5) > 3.0:
        return "medium"
    return "low"


def _replacement_target(row: dict[str, Any]) -> str:
    return ""


def _expanded_candidate_id(symbol: str, strategy_candidate: str, ts: datetime) -> str:
    safe_candidate = strategy_candidate.replace(".", "_").replace("/", "_")
    return f"{EXPANDED_UNIVERSE_TYPE}:{symbol}:{safe_candidate}:{ts.isoformat()}"


def _bar_at_or_before(
    rows: list[dict[str, Any]],
    target_ts: datetime | None,
) -> dict[str, Any] | None:
    if target_ts is None:
        return None
    best = None
    for row in rows:
        ts = row.get("ts")
        if isinstance(ts, datetime) and ts <= target_ts:
            best = row
        if isinstance(ts, datetime) and ts > target_ts:
            break
    return best


def _bar_at_or_after(
    rows: list[dict[str, Any]],
    target_ts: datetime | None,
) -> dict[str, Any] | None:
    if target_ts is None:
        return None
    for row in rows:
        ts = row.get("ts")
        if isinstance(ts, datetime) and ts >= target_ts:
            return row
    return None


def _bars_between(
    rows: list[dict[str, Any]],
    start_ts: datetime,
    end_ts: datetime,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if isinstance(row.get("ts"), datetime) and start_ts <= row["ts"] <= end_ts
    ]


def _mfe_mae_bps(
    rows: list[dict[str, Any]],
    entry_close: float,
) -> tuple[float | None, float | None]:
    if entry_close <= 0 or not rows:
        return None, None
    highs = [
        _float(row.get("high") or row.get("close"))
        for row in rows
        if _float(row.get("high") or row.get("close")) is not None
    ]
    lows = [
        _float(row.get("low") or row.get("close"))
        for row in rows
        if _float(row.get("low") or row.get("close")) is not None
    ]
    mfe = (max(highs) / entry_close - 1.0) * 10_000.0 if highs else None
    mae = (min(lows) / entry_close - 1.0) * 10_000.0 if lows else None
    return mfe, mae


def _cost_source_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        source = str(row.get("cost_source") or "missing")
        counts[source] += 1
    return dict(sorted(counts.items()))


def _expanded_decision(
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    p25_net_bps: float | None,
    win_rate: float | None,
    cost_source_mix: dict[str, int],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if sample_count < 10:
        reasons.append("insufficient_total_samples")
        return "RESEARCH_ONLY", reasons
    if complete_sample_count < 5:
        reasons.append("insufficient_complete_samples")
        return "RESEARCH_ONLY", reasons
    avg = avg_net_bps if avg_net_bps is not None else 0.0
    p25 = p25_net_bps if p25_net_bps is not None else 0.0
    win = win_rate if win_rate is not None else 0.0
    has_global_default = "global_default" in {
        source.lower() for source in cost_source_mix
    }
    if complete_sample_count >= 30 and avg < 0 and win < 0.45 and p25 < -50:
        reasons.extend(["non_positive_after_cost_edge", "win_rate_below_threshold"])
        return "KILL", reasons
    if complete_sample_count >= 10 and avg > 0 and win >= 0.55 and p25 > -50:
        if has_global_default:
            reasons.append("cost_source_not_trusted")
            return "KEEP_SHADOW", reasons
        reasons.append("expanded_universe_paper_only")
        return "PAPER_READY", reasons
    if complete_sample_count >= 5 and avg >= 0:
        reasons.append("positive_shadow_edge_needs_more_samples")
        return "KEEP_SHADOW", reasons
    reasons.append("shadow_collect_more_samples")
    return "SHADOW", reasons


def _maturity_state_from_evidence_rows(
    rows: list[dict[str, Any]],
) -> tuple[str, list[str], dict[str, Any]]:
    sample_count = max((_int(row.get("sample_count")) or 0 for row in rows), default=0)
    complete_sample_count = max(
        (_int(row.get("complete_sample_count")) or 0 for row in rows),
        default=0,
    )
    short_rows = [
        row
        for row in rows
        if (_int(row.get("horizon_hours")) or 0) in SHORT_HORIZONS
        and (_int(row.get("complete_sample_count")) or 0) > 0
    ]
    positive_short_rows = [
        row for row in short_rows if (_float(row.get("avg_net_bps")) or 0.0) > 0.0
    ]
    positive_short_horizons = sorted(
        {
            _int(row.get("horizon_hours")) or 0
            for row in positive_short_rows
            if _int(row.get("horizon_hours")) is not None
        }
    )
    best_short_row = max(
        short_rows,
        key=lambda row: _float(row.get("avg_net_bps")) or float("-inf"),
        default={},
    )
    best_all_row = max(
        rows,
        key=lambda row: _float(row.get("avg_net_bps")) or float("-inf"),
        default={},
    )
    p25 = _float(best_all_row.get("p25_net_bps"))
    win = _float(best_all_row.get("win_rate"))
    avg = _float(best_all_row.get("avg_net_bps"))
    state = "RESEARCH"
    reasons: list[str] = []
    if complete_sample_count < 10:
        reasons.append("insufficient_complete_samples")
    elif (
        complete_sample_count >= 30
        and (win or 0.0) > 0.55
        and (p25 if p25 is not None else float("-inf")) > -50.0
        and (avg or 0.0) > 0.0
    ):
        state = "PAPER_READY"
        reasons.append("mature_positive_after_cost_evidence")
    elif len(positive_short_horizons) >= 2:
        state = "KEEP_SHADOW"
        reasons.append("two_positive_short_horizons")
    else:
        reasons.append("needs_more_outcome_evidence")
    stats = {
        "sample_count": sample_count,
        "complete_sample_count": complete_sample_count,
        "positive_short_horizon_count": len(positive_short_horizons),
        "positive_short_horizons": positive_short_horizons,
        "best_short_horizon_hours": _int(best_short_row.get("horizon_hours")) or 0,
        "best_short_avg_net_bps": _float(best_short_row.get("avg_net_bps")),
        "win_rate": win,
        "p25_net_bps": p25,
    }
    return state, reasons, stats


def _merge_cost_source_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        raw = row.get("cost_source_mix")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    counts[str(key)] += _int(value) or 0
                continue
        source = str(row.get("cost_source") or "missing")
        counts[source] += max(_int(row.get("complete_sample_count")) or 1, 1)
    return dict(sorted(counts.items()))


def _best_maturity_by_symbol(maturity: pl.DataFrame) -> dict[str, dict[str, Any]]:
    if maturity.is_empty():
        return {}
    rank = {"PAPER_READY": 3, "KEEP_SHADOW": 2, "RESEARCH": 1}
    best: dict[str, dict[str, Any]] = {}
    for row in maturity.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        current = best.get(symbol)
        score = (
            rank.get(str(row.get("maturity_state") or ""), 0),
            _int(row.get("complete_sample_count")) or 0,
            _float(row.get("best_short_avg_net_bps")) or float("-inf"),
        )
        current_score = (
            rank.get(str(current.get("maturity_state") or ""), 0),
            _int(current.get("complete_sample_count")) or 0,
            _float(current.get("best_short_avg_net_bps")) or float("-inf"),
        ) if current else (-1, -1, float("-inf"))
        if current is None or score > current_score:
            best[symbol] = row
    return best


def _watch_reason(*, watchlist_type: str, has_quality: bool, has_maturity: bool) -> str:
    if watchlist_type == "quality_watchlist":
        if has_quality and has_maturity:
            return "quality_score_and_outcome_monitor"
        if has_quality:
            return "quality_score_monitor"
        return "quality_watch_symbol_not_observed"
    if watchlist_type == "reject_list":
        if has_maturity:
            return "low_priority_current_outcomes_weak"
        if has_quality:
            return "low_priority_quality_not_enough"
        return "low_priority_symbol_not_observed"
    if has_maturity:
        return "early_outcome_signal_monitor"
    return "outcome_watch_symbol_not_observed"


def _promotion_state_from_decision(decision: str) -> str:
    decision = str(decision or "").upper()
    if decision == "PAPER_READY":
        return "PAPER"
    if decision in {"KEEP_SHADOW", "REGIME_SHADOW", "SHADOW"}:
        return "SHADOW"
    if decision == "KILL":
        return "KILL"
    return "RESEARCH"


def _promotion_state_from_maturity_state(maturity_state: str) -> str:
    state = str(maturity_state or "").upper()
    if state == "PAPER_READY":
        return "PAPER"
    if state == "KEEP_SHADOW":
        return "SHADOW"
    return "RESEARCH"


def _recommended_mode_from_maturity_state(maturity_state: str) -> str:
    return _recommended_mode_from_promotion_state(
        _promotion_state_from_maturity_state(maturity_state)
    )


def _recommended_mode_from_promotion_state(promotion_state: str) -> str:
    if promotion_state == "PAPER":
        return "paper"
    if promotion_state == "SHADOW":
        return "shadow"
    if promotion_state == "KILL":
        return "none"
    return "research"


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


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


def _read_recent_market_bars(
    lake_root: Path,
    *,
    min_coverage_bars: int,
) -> tuple[pl.DataFrame, datetime | None]:
    dataset_path = lake_root / MARKET_BAR_DATASET
    scan = _scan_dataset(dataset_path, max_files=800)
    if scan is None:
        return pl.DataFrame(), None
    try:
        max_ts = (
            scan.select(pl.col("ts").max().alias("max_ts"))
            .collect(engine="streaming")
            .item()
        )
    except Exception:
        return pl.DataFrame(), None
    latest_ts = _parse_dt(max_ts)
    if latest_ts is None:
        return pl.DataFrame(), None
    lookback_hours = max(min_coverage_bars + 24, 24 * 31)
    since = latest_ts - timedelta(hours=lookback_hours)
    columns = [
        "symbol",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]
    try:
        frame = (
            scan.filter(
                (pl.col("ts") >= pl.lit(since))
                & (pl.col("timeframe").is_null() | (pl.col("timeframe") == "1H"))
            )
            .select(columns)
            .collect(engine="streaming")
        )
    except Exception:
        frame = pl.DataFrame()
    return frame, latest_ts


def _read_recent_orderbook_snapshots(lake_root: Path, *, since: datetime) -> pl.DataFrame:
    dataset_path = lake_root / ORDERBOOK_SNAPSHOT_DATASET
    scan = _scan_dataset(dataset_path, max_files=300)
    if scan is None:
        return pl.DataFrame()
    timestamp = pl.coalesce([pl.col("ts"), pl.col("ingest_ts")])
    columns = ["symbol", "ts", "ingest_ts", "bids_json", "asks_json"]
    try:
        return (
            scan.filter(timestamp >= pl.lit(since))
            .select(columns)
            .collect(engine="streaming")
        )
    except Exception:
        return pl.DataFrame()


def _orderbook_since(as_of_date: date, market_end: datetime | None) -> datetime:
    if market_end is not None:
        return market_end - timedelta(hours=24)
    return datetime.combine(as_of_date + timedelta(days=1), time.min, tzinfo=UTC) - timedelta(
        hours=24
    )


def _scan_dataset(dataset_path: Path, *, max_files: int | None = None) -> pl.LazyFrame | None:
    if not dataset_path.exists():
        return None
    parquet_paths = list(dataset_path.rglob("*.parquet"))
    if not parquet_paths:
        return None
    if max_files is not None and len(parquet_paths) > max_files:
        parquet_paths = sorted(
            parquet_paths,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:max_files]
    return pl.scan_parquet(
        [str(path) for path in parquet_paths],
        missing_columns="insert",
        extra_columns="ignore",
        low_memory=True,
        cache=False,
    )


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


def _spot_candidate_metrics(frame: pl.DataFrame) -> dict[str, dict[str, float]]:
    if frame.is_empty() or "symbol" not in frame.columns:
        return {}
    metrics: dict[str, dict[str, float]] = {}
    sort_columns = [column for column in ["generated_at", "rank"] if column in frame.columns]
    rows = (
        frame.sort(sort_columns, descending=[True, False][: len(sort_columns)]).to_dicts()
        if sort_columns
        else frame.to_dicts()
    )
    for row in rows:
        symbol = normalize_symbol(row.get("symbol"))
        if not _is_usdt_symbol(symbol) or symbol in metrics:
            continue
        quote_volume = _float(row.get("quote_volume_24h"))
        spread = _float(row.get("avg_spread_bps") or row.get("spread_bps"))
        metrics[symbol] = {
            "quote_volume_24h": quote_volume or 0.0,
            "avg_spread_bps": spread if spread is not None else float("nan"),
        }
    return metrics


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
    if quality_score >= 60.0:
        return EXPANDED_PAPER_UNIVERSE_RECOMMENDATION
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
    if recommendation == EXPANDED_PAPER_UNIVERSE_RECOMMENDATION:
        return "质量候选；只能进入 expanded paper/shadow 研究，不输出 ETH/BNB 替换建议。"
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
