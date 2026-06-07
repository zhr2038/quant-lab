from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from quant_lab.backtest.cost_model import conservative_cost_for_symbol
from quant_lab.backtest.datasets import (
    HORIZONS,
    boolish,
    coerce_dt,
    entry_price_from_row,
    first_value,
    float_or_none,
    future_net_bps_from_market,
    iso_utc,
    market_rows_by_symbol,
    normalize_strategy_symbol,
    rows,
)
from quant_lab.backtest.engine import BacktestEngine
from quant_lab.backtest.event_replay import (
    V5_DECISION_REPLAY_EQUITY_FIELDS,
    V5_DECISION_REPLAY_TRADES_FIELDS,
)
from quant_lab.backtest.label_backtest import (
    BACKTEST_LABEL_SUMMARY_FIELDS,
    label_summary_md,
)
from quant_lab.backtest.metrics import frame_with_schema, max_drawdown_bps, summarize_net_bps

BOTTOM_ZONE_BACKTEST_FIELDS = [
    "strategy_id",
    "symbol",
    "ts_utc",
    "entry_px",
    "bottom_zone_state",
    "would_enter",
    "support_zone",
    "anchored_vwap",
    "orderbook_imbalance",
    "taker_sell_exhaustion",
    "volatility_climax",
    "spread_normalization",
    "cost_bps",
    "cost_model",
    "future_4h_net_bps",
    "future_8h_net_bps",
    "future_12h_net_bps",
    "future_24h_net_bps",
    "label_status",
    "data_leakage_check",
    "live_order_effect",
]

RESEARCH_PROMOTION_DECISION_FIELDS = [
    "strategy_id",
    "symbol",
    "horizon_hours",
    "current_stage",
    "recommended_stage",
    "sample_count",
    "complete_sample_count",
    "avg_net_bps",
    "p25_net_bps",
    "win_rate",
    "recent_7d_avg_net_bps",
    "paper_days",
    "paper_entries",
    "paper_avg_net_bps",
    "actual_or_mixed_cost_coverage",
    "max_drawdown_bps",
    "decision_reasons",
    "live_order_effect",
]

BACKTEST_VS_PAPER_CONSISTENCY_FIELDS = [
    "strategy_id",
    "symbol",
    "horizon_hours",
    "backtest_sample_count",
    "backtest_complete_sample_count",
    "backtest_avg_net_bps",
    "backtest_p25_net_bps",
    "paper_strategy_id",
    "paper_days",
    "paper_entries",
    "paper_avg_net_bps",
    "consistency_status",
    "recommendation",
    "decision_reasons",
    "live_order_effect",
]

BACKTEST_REGIME_BREAKDOWN_FIELDS = [
    "strategy_id",
    "symbol",
    "regime",
    "horizon_hours",
    "sample_count",
    "complete_sample_count",
    "avg_net_bps",
    "p25_net_bps",
    "win_rate",
    "recommendation",
    "live_order_effect",
]

BACKTEST_CSV_SCHEMAS = {
    "reports/backtest_label_summary.csv": BACKTEST_LABEL_SUMMARY_FIELDS,
    "reports/v5_decision_replay_trades.csv": V5_DECISION_REPLAY_TRADES_FIELDS,
    "reports/v5_decision_replay_equity.csv": V5_DECISION_REPLAY_EQUITY_FIELDS,
    "reports/backtest_regime_breakdown.csv": BACKTEST_REGIME_BREAKDOWN_FIELDS,
    "reports/bottom_zone_backtest.csv": BOTTOM_ZONE_BACKTEST_FIELDS,
    "reports/research_promotion_decision.csv": RESEARCH_PROMOTION_DECISION_FIELDS,
    "reports/backtest_vs_paper_consistency.csv": BACKTEST_VS_PAPER_CONSISTENCY_FIELDS,
}


@dataclass(frozen=True)
class BacktestReportBundle:
    label_summary: pl.DataFrame
    label_summary_md: str
    replay_trades: pl.DataFrame
    replay_equity: pl.DataFrame
    replay_summary_md: str
    regime_breakdown: pl.DataFrame
    bottom_zone_backtest: pl.DataFrame
    bottom_zone_summary_md: str
    promotion_decision: pl.DataFrame
    promotion_decision_md: str
    backtest_vs_paper_consistency: pl.DataFrame
    backtest_vs_paper_consistency_md: str


def build_backtest_report_bundle(frames: dict[str, pl.DataFrame]) -> BacktestReportBundle:
    engine_result = BacktestEngine().run(frames)
    bottom = build_bottom_zone_backtest(
        bottom_zone_reversal_shadow=frames.get("bottom_zone_reversal_shadow", pl.DataFrame()),
        market_bars=frames.get("market_bar", pl.DataFrame()),
        cost_bucket_daily=frames.get("cost_bucket_daily", pl.DataFrame()),
    )
    consistency = build_backtest_vs_paper_consistency(
        label_summary=engine_result.label_summary,
        paper_daily=frames.get("paper_strategy_daily", pl.DataFrame()),
        bnb_paper_daily=frames.get("bnb_paper_strategy_daily", pl.DataFrame()),
    )
    promotion = build_research_promotion_decision(
        label_summary=engine_result.label_summary,
        paper_daily=frames.get("paper_strategy_daily", pl.DataFrame()),
        bnb_paper_daily=frames.get("bnb_paper_strategy_daily", pl.DataFrame()),
        backtest_vs_paper_consistency=consistency,
    )
    return BacktestReportBundle(
        label_summary=engine_result.label_summary,
        label_summary_md=label_summary_md(engine_result.label_summary),
        replay_trades=engine_result.replay_trades,
        replay_equity=engine_result.replay_equity,
        replay_summary_md=engine_result.replay_summary_md,
        regime_breakdown=build_backtest_regime_breakdown(engine_result.label_summary),
        bottom_zone_backtest=bottom,
        bottom_zone_summary_md=bottom_zone_backtest_summary_md(bottom),
        promotion_decision=promotion,
        promotion_decision_md=research_promotion_decision_md(promotion),
        backtest_vs_paper_consistency=consistency,
        backtest_vs_paper_consistency_md=backtest_vs_paper_consistency_md(consistency),
    )


def build_backtest_regime_breakdown(label_summary: pl.DataFrame) -> pl.DataFrame:
    out: list[dict[str, Any]] = []
    for row in rows(label_summary):
        out.append(
            {
                "strategy_id": row.get("strategy_id"),
                "symbol": row.get("symbol"),
                "regime": row.get("regime"),
                "horizon_hours": row.get("horizon_hours"),
                "sample_count": row.get("sample_count"),
                "complete_sample_count": row.get("complete_sample_count"),
                "avg_net_bps": row.get("avg_net_bps"),
                "p25_net_bps": row.get("p25_net_bps"),
                "win_rate": row.get("win_rate"),
                "recommendation": row.get("recommendation"),
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return frame_with_schema(out, BACKTEST_REGIME_BREAKDOWN_FIELDS)


def build_bottom_zone_backtest(
    *,
    bottom_zone_reversal_shadow: pl.DataFrame | None,
    market_bars: pl.DataFrame | None,
    cost_bucket_daily: pl.DataFrame | None,
) -> pl.DataFrame:
    bars_by_symbol = market_rows_by_symbol(market_bars)
    out: list[dict[str, Any]] = []
    for row in rows(bottom_zone_reversal_shadow):
        symbol = normalize_strategy_symbol(row.get("symbol"))
        ts = coerce_dt(first_value(row, ("ts_utc", "decision_ts", "generated_at")))
        entry_px = entry_price_from_row(row)
        if entry_px is None or ts is None or symbol == "UNKNOWN":
            continue
        cost = conservative_cost_for_symbol(cost_bucket_daily, symbol=symbol)
        futures = {
            horizon: future_net_bps_from_market(
                bars_by_symbol=bars_by_symbol,
                symbol=symbol,
                ts=ts,
                entry_px=entry_px,
                horizon_hours=horizon,
                cost_bps=cost.cost_bps,
            )
            for horizon in (4, 8, 12, 24)
        }
        out.append(
            {
                "strategy_id": "BOTTOM_ZONE_PROBE_BACKTEST",
                "symbol": symbol,
                "ts_utc": iso_utc(ts),
                "entry_px": entry_px,
                "bottom_zone_state": row.get("bottom_zone_state"),
                "would_enter": boolish(row.get("would_probe_paper")),
                "support_zone": first_value(row, ("support_low_24h", "support_low_72h")),
                "anchored_vwap": first_value(row, ("vwap_24h", "anchored_vwap")),
                "orderbook_imbalance": first_value(row, ("orderbook_imbalance", "market_pressure_state")),
                "taker_sell_exhaustion": first_value(row, ("taker_sell_exhaustion", "trade_count_60m")),
                "volatility_climax": first_value(row, ("volatility_climax", "return_24h_bps")),
                "spread_normalization": first_value(row, ("spread_normalization", "avg_spread_bps_15m")),
                "cost_bps": cost.cost_bps,
                "cost_model": cost.cost_model,
                "future_4h_net_bps": futures[4],
                "future_8h_net_bps": futures[8],
                "future_12h_net_bps": futures[12],
                "future_24h_net_bps": futures[24],
                "label_status": "partial_complete" if any(v is not None for v in futures.values()) else "pending",
                "data_leakage_check": "pass_future_prices_used_only_for_labels",
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return frame_with_schema(out, BOTTOM_ZONE_BACKTEST_FIELDS)


def bottom_zone_backtest_summary_md(frame: pl.DataFrame) -> str:
    rows_in = rows(frame)
    stats = summarize_net_bps([row.get("future_24h_net_bps") for row in rows_in])
    lines = [
        "# Bottom Zone Backtest",
        "",
        "Read-only backtest for BOTTOM_ZONE_PROBE_BACKTEST.",
        "Signals are evaluated with future labels only after each decision timestamp.",
        "",
        f"- rows: {frame.height}",
        f"- 24h_complete_sample_count: {stats['complete_sample_count']}",
        f"- 24h_avg_net_bps: {stats['avg_net_bps']}",
        f"- 24h_win_rate: {stats['win_rate']}",
        "- live_order_effect: read_only_no_live_order",
    ]
    return "\n".join(lines) + "\n"


def build_research_promotion_decision(
    *,
    label_summary: pl.DataFrame,
    paper_daily: pl.DataFrame | None = None,
    bnb_paper_daily: pl.DataFrame | None = None,
    backtest_vs_paper_consistency: pl.DataFrame | None = None,
) -> pl.DataFrame:
    paper_index = _paper_daily_index(paper_daily, bnb_paper_daily)
    conflict_index = _paper_conflict_index(backtest_vs_paper_consistency)
    out: list[dict[str, Any]] = []
    for row in rows(label_summary):
        strategy_id = str(row.get("strategy_id") or "")
        paper = paper_index.get(strategy_id, {})
        sample_count = int(float_or_none(row.get("sample_count")) or 0)
        complete = int(float_or_none(row.get("complete_sample_count")) or 0)
        avg = float_or_none(row.get("avg_net_bps"))
        p25 = float_or_none(row.get("p25_net_bps"))
        win = float_or_none(row.get("win_rate"))
        paper_days = int(float_or_none(paper.get("paper_days")) or 0)
        paper_entries = int(float_or_none(paper.get("entry_count")) or 0)
        paper_avg = float_or_none(
            first_value(paper, ("avg_paper_pnl_bps", "avg_paper_pnl_bps_24h"))
        )
        cost_ok = _actual_or_mixed_cost(row, paper)
        stage, reasons = _promotion_stage(
            sample_count=sample_count,
            complete=complete,
            avg=avg,
            p25=p25,
            win=win,
            paper_days=paper_days,
            paper_entries=paper_entries,
            paper_avg=paper_avg,
            actual_or_mixed_cost_coverage=cost_ok,
        )
        conflict_key = _promotion_conflict_key(row)
        conflict = conflict_index.get(conflict_key)
        if conflict:
            stage = "QUARANTINE"
            reasons = [
                "QUARANTINE_BACKTEST_PAPER_CONFLICT",
                *[reason for reason in reasons if reason != "paper_days_or_entries_insufficient"],
            ]
        out.append(
            {
                "strategy_id": strategy_id,
                "symbol": row.get("symbol"),
                "horizon_hours": row.get("horizon_hours"),
                "current_stage": _current_stage(strategy_id),
                "recommended_stage": stage,
                "sample_count": sample_count,
                "complete_sample_count": complete,
                "avg_net_bps": avg,
                "p25_net_bps": p25,
                "win_rate": win,
                "recent_7d_avg_net_bps": None,
                "paper_days": paper_days,
                "paper_entries": paper_entries,
                "paper_avg_net_bps": paper_avg,
                "actual_or_mixed_cost_coverage": cost_ok,
                "max_drawdown_bps": max_drawdown_bps([row.get("avg_net_bps")]),
                "decision_reasons": ";".join(reasons),
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return frame_with_schema(out, RESEARCH_PROMOTION_DECISION_FIELDS)


def build_backtest_vs_paper_consistency(
    *,
    label_summary: pl.DataFrame,
    paper_daily: pl.DataFrame | None = None,
    bnb_paper_daily: pl.DataFrame | None = None,
) -> pl.DataFrame:
    paper_by_strategy, paper_by_symbol = _paper_daily_lookup(paper_daily, bnb_paper_daily)
    out: list[dict[str, Any]] = []
    for row in rows(label_summary):
        avg = float_or_none(row.get("avg_net_bps"))
        if avg is None or avg <= 0:
            continue
        strategy_id = str(row.get("strategy_id") or "")
        symbol = normalize_strategy_symbol(row.get("symbol"))
        paper = paper_by_strategy.get(strategy_id)
        if paper is None and _strategy_uses_symbol_paper_proxy(strategy_id, symbol):
            paper = paper_by_symbol.get(symbol)
        if paper is None:
            continue
        horizon = int(float_or_none(row.get("horizon_hours")) or 0)
        paper_avg = _paper_avg_for_horizon(paper, horizon)
        if paper_avg is None:
            continue
        conflict = paper_avg < 0
        out.append(
            {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "horizon_hours": horizon,
                "backtest_sample_count": row.get("sample_count"),
                "backtest_complete_sample_count": row.get("complete_sample_count"),
                "backtest_avg_net_bps": avg,
                "backtest_p25_net_bps": row.get("p25_net_bps"),
                "paper_strategy_id": paper.get("strategy_id"),
                "paper_days": paper.get("paper_days_to_date") or paper.get("paper_days"),
                "paper_entries": paper.get("entry_count"),
                "paper_avg_net_bps": paper_avg,
                "consistency_status": (
                    "backtest_positive_paper_negative"
                    if conflict
                    else "backtest_positive_paper_non_negative"
                ),
                "recommendation": (
                    "QUARANTINE_BACKTEST_PAPER_CONFLICT"
                    if conflict
                    else "CONSISTENT_KEEP_RESEARCH_PIPELINE"
                ),
                "decision_reasons": (
                    "backtest_positive_but_v5_paper_negative"
                    if conflict
                    else "backtest_positive_and_v5_paper_non_negative"
                ),
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return frame_with_schema(out, BACKTEST_VS_PAPER_CONSISTENCY_FIELDS)


def backtest_vs_paper_consistency_md(frame: pl.DataFrame) -> str:
    conflicts = frame.filter(pl.col("recommendation") == "QUARANTINE_BACKTEST_PAPER_CONFLICT") if not frame.is_empty() else pl.DataFrame()
    lines = [
        "# Backtest vs Paper Consistency",
        "",
        "Read-only consistency check between deduped backtest labels and V5 paper telemetry.",
        "Rows with positive backtest but negative paper are quarantined from PAPER promotion.",
        "",
        f"- rows: {frame.height}",
        f"- quarantine_conflict_rows: {conflicts.height}",
        "- live_order_effect: read_only_no_live_order",
    ]
    for row in rows(conflicts)[:12]:
        lines.append(
            "- "
            f"{row.get('strategy_id')} {row.get('symbol')} h={row.get('horizon_hours')} "
            f"backtest_avg={row.get('backtest_avg_net_bps')} paper_avg={row.get('paper_avg_net_bps')} "
            f"recommendation={row.get('recommendation')}"
        )
    return "\n".join(lines) + "\n"


def research_promotion_decision_md(frame: pl.DataFrame) -> str:
    lines = [
        "# Research Promotion Decision",
        "",
        "Read-only promotion gate for research -> shadow -> paper -> live-small review.",
        "LIVE_SMALL rows are review-only and do not alter V5 live configuration.",
        "",
        f"- decision_rows: {frame.height}",
    ]
    for row in rows(frame)[:20]:
        lines.append(
            "- "
            f"{row.get('strategy_id')} {row.get('symbol')} h={row.get('horizon_hours')} "
            f"stage={row.get('recommended_stage')} reasons={row.get('decision_reasons')}"
        )
    return "\n".join(lines) + "\n"


def _paper_daily_index(*frames: pl.DataFrame | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for frame in frames:
        for row in rows(frame):
            strategy_id = str(row.get("strategy_id") or "").strip()
            if strategy_id:
                out[strategy_id] = row
    return out


def _paper_daily_lookup(
    *frames: pl.DataFrame | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_strategy: dict[str, dict[str, Any]] = {}
    by_symbol: dict[str, dict[str, Any]] = {}
    for frame in frames:
        for row in rows(frame):
            strategy_id = str(row.get("strategy_id") or "").strip()
            symbol = normalize_strategy_symbol(row.get("symbol"))
            if not strategy_id and symbol == "UNKNOWN":
                continue
            existing = by_strategy.get(strategy_id) if strategy_id else None
            if strategy_id and (existing is None or _paper_row_sort_key(row) >= _paper_row_sort_key(existing)):
                by_strategy[strategy_id] = row
            if symbol != "UNKNOWN":
                existing_symbol = by_symbol.get(symbol)
                if existing_symbol is None or _paper_row_sort_key(row) >= _paper_row_sort_key(existing_symbol):
                    by_symbol[symbol] = row
    return by_strategy, by_symbol


def _paper_row_sort_key(row: dict[str, Any]) -> tuple[str, float]:
    return (
        str(first_value(row, ("paper_date", "ts_utc", "generated_at")) or ""),
        float_or_none(row.get("entry_count")) or 0.0,
    )


def _strategy_uses_symbol_paper_proxy(strategy_id: str, symbol: str) -> bool:
    text = strategy_id.upper()
    return symbol == "BNB-USDT" and (
        "BNB_STRONG_ALPHA6_BYPASS" in text
        or "FINAL_SCORE_ALPHA6_CONFLICT" in text
    )


def _paper_avg_for_horizon(row: dict[str, Any], horizon: int) -> float | None:
    candidates = []
    if horizon > 0:
        candidates.append(f"avg_paper_pnl_bps_{horizon}h")
    candidates.extend(
        [
            "avg_paper_pnl_bps",
            "paper_avg_net_bps",
            "avg_net_bps",
        ]
    )
    return float_or_none(first_value(row, tuple(candidates)))


def _promotion_conflict_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("strategy_id") or ""),
        normalize_strategy_symbol(row.get("symbol")),
        int(float_or_none(row.get("horizon_hours")) or 0),
    )


def _paper_conflict_index(frame: pl.DataFrame | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows(frame):
        if str(row.get("recommendation") or "") != "QUARANTINE_BACKTEST_PAPER_CONFLICT":
            continue
        out[
            (
                str(row.get("strategy_id") or ""),
                normalize_strategy_symbol(row.get("symbol")),
                int(float_or_none(row.get("horizon_hours")) or 0),
            )
        ] = row
    return out


def _actual_or_mixed_cost(row: dict[str, Any], paper: dict[str, Any]) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            row.get("cost_model"),
            paper.get("cost_source_mix"),
            paper.get("latest_cost_source"),
        )
    ).lower()
    return "actual" in text or "mixed" in text


def _promotion_stage(
    *,
    sample_count: int,
    complete: int,
    avg: float | None,
    p25: float | None,
    win: float | None,
    paper_days: int,
    paper_entries: int,
    paper_avg: float | None,
    actual_or_mixed_cost_coverage: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    avg_value = avg if avg is not None else -10**9
    if sample_count < 30 or avg_value <= 0:
        if sample_count < 30:
            reasons.append("sample_count_lt_30")
        if avg_value <= 0:
            reasons.append("avg_net_bps_not_positive")
        return "RESEARCH", reasons or ["collect_more_samples"]
    if complete < 50 or p25 is None or p25 <= -50 or win is None or win <= 0.55:
        if complete < 50:
            reasons.append("complete_sample_count_lt_50")
        if p25 is None or p25 <= -50:
            reasons.append("p25_not_above_minus_50")
        if win is None or win <= 0.55:
            reasons.append("win_rate_lte_0_55")
        return "SHADOW", reasons
    if (
        paper_days >= 14
        and paper_entries >= 20
        and (paper_avg is not None and paper_avg > 0)
        and actual_or_mixed_cost_coverage
    ):
        return "LIVE_SMALL_REVIEW_ONLY", ["paper_live_small_thresholds_met_review_only"]
    reasons.extend(
        [
            "paper_days_or_entries_insufficient",
            "live_small_review_requires_actual_or_mixed_cost",
        ]
    )
    return "PAPER", reasons


def _current_stage(strategy_id: str) -> str:
    text = strategy_id.upper()
    if "PAPER" in text:
        return "PAPER"
    if "SHADOW" in text:
        return "SHADOW"
    return "RESEARCH"
