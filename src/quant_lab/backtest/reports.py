from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import polars as pl

from quant_lab.backtest.cost_model import conservative_cost_for_symbol
from quant_lab.backtest.datasets import (
    boolish,
    coerce_dt,
    entry_price_from_row,
    first_float,
    first_value,
    float_or_none,
    future_net_bps_from_market,
    iso_utc,
    market_rows_by_symbol,
    normalize_strategy_symbol,
    price_at_or_after,
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

BACKTEST_VS_PAPER_GAP_FIELDS = [
    "strategy_id",
    "symbol",
    "horizon_hours",
    "backtest_sample_count",
    "backtest_complete_sample_count",
    "backtest_avg_net_bps",
    "backtest_p25_net_bps",
    "backtest_cost_model",
    "backtest_data_leakage_check",
    "paper_strategy_id",
    "paper_days",
    "paper_entries",
    "paper_closed_entries",
    "paper_avg_net_bps",
    "paper_cost_source_mix",
    "arrival_mid_coverage",
    "gap_bps",
    "absolute_gap_bps",
    "paper_to_backtest_ratio",
    "gap_status",
    "root_causes",
    "recommendation",
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

FACTOR_FORWARD_VALIDATION_FIELDS = [
    "as_of_date",
    "factor_id",
    "factor_family",
    "candidate_state",
    "symbol",
    "regime",
    "horizon_hours",
    "sample_count",
    "rank_ic",
    "pearson_ic",
    "long_short_bps",
    "p25_net_bps",
    "hit_rate",
    "recent_7d_score",
    "regime_stability",
    "cost_adjusted_score",
    "recommendation",
    "data_leakage_check",
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
    "reports/backtest_vs_paper_gap_report.csv": BACKTEST_VS_PAPER_GAP_FIELDS,
    "reports/factor_forward_validation.csv": FACTOR_FORWARD_VALIDATION_FIELDS,
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
    backtest_vs_paper_gap_report: pl.DataFrame
    backtest_vs_paper_gap_report_md: str


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
    gap_report = build_backtest_vs_paper_gap_report(
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
        backtest_vs_paper_gap_report=gap_report,
        backtest_vs_paper_gap_report_md=backtest_vs_paper_gap_report_md(gap_report),
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


def build_factor_forward_validation(
    *,
    factor_candidates: pl.DataFrame | None,
    factor_values: pl.DataFrame | None,
    market_bars: pl.DataFrame | None,
    market_regime: pl.DataFrame | None = None,
    cost_bucket_daily: pl.DataFrame | None = None,
    horizon_hours: tuple[int, ...] = (1, 4, 8),
) -> pl.DataFrame:
    candidate_rows = [
        row
        for row in rows(factor_candidates)
        if str(row.get("candidate_state") or "") in {"PAPER_READY", "KEEP_SHADOW"}
        and str(row.get("factor_id") or "").strip()
    ]
    candidate_by_factor = {str(row.get("factor_id")): row for row in candidate_rows}
    if not candidate_by_factor:
        return frame_with_schema([], FACTOR_FORWARD_VALIDATION_FIELDS)

    bars_by_symbol = market_rows_by_symbol(market_bars)
    regime_rows = _forward_regime_rows(market_regime)
    samples: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for value_row in rows(factor_values):
        factor_id = str(value_row.get("factor_id") or "")
        candidate = candidate_by_factor.get(factor_id)
        if candidate is None:
            continue
        if "is_valid" in value_row and not boolish(value_row.get("is_valid")):
            continue
        symbol = normalize_strategy_symbol(value_row.get("symbol"))
        ts = coerce_dt(first_value(value_row, ("ts", "feature_ts", "created_at")))
        factor_value = first_float(
            value_row,
            ("value", "normalized_value", "rank_value", "raw_value"),
        )
        if symbol == "UNKNOWN" or ts is None or factor_value is None:
            continue
        entry_px = price_at_or_after(bars_by_symbol, symbol, ts)
        if entry_px is None or entry_px <= 0:
            continue
        cost = conservative_cost_for_symbol(cost_bucket_daily, symbol=symbol)
        regime = (
            _canonical_forward_regime(
                first_value(
                    value_row,
                    ("regime", "regime_state", "market_regime", "current_regime"),
                )
            )
            or _forward_regime_for_ts(regime_rows, ts)
            or _derived_market_regime(bars_by_symbol.get(symbol, []), ts)
        )
        for horizon in sorted({int(item) for item in horizon_hours if int(item) > 0}):
            future_net = future_net_bps_from_market(
                bars_by_symbol=bars_by_symbol,
                symbol=symbol,
                ts=ts,
                entry_px=entry_px,
                horizon_hours=horizon,
                cost_bps=cost.cost_bps,
            )
            if future_net is None:
                continue
            samples.setdefault((factor_id, symbol, regime, horizon), []).append(
                {"ts": ts, "value": factor_value, "future_net_bps": future_net}
            )

    out: list[dict[str, Any]] = []
    for key, sample_rows in sorted(samples.items()):
        factor_id, symbol, regime, horizon = key
        candidate = candidate_by_factor[factor_id]
        out.append(
            _factor_forward_summary_row(
                candidate=candidate,
                symbol=symbol,
                regime=regime,
                horizon=horizon,
                samples=sample_rows,
            )
        )
    out = _enrich_factor_forward_rows(out)
    return frame_with_schema(out, FACTOR_FORWARD_VALIDATION_FIELDS)


def factor_forward_validation_md(frame: pl.DataFrame) -> str:
    row_list = rows(frame)
    passed = [
        row
        for row in row_list
        if str(row.get("recommendation") or "") == "FORWARD_VALIDATION_PASS"
    ]
    lines = [
        "# Factor Forward Validation",
        "",
        "Read-only OOS/recent/regime validation for PAPER_READY and KEEP_SHADOW factors.",
        "A factor bridge candidate remains blocked unless this report has a passing row.",
        "",
        f"- rows: {len(row_list)}",
        f"- pass_rows: {len(passed)}",
        "- live_order_effect: none_read_only_research",
    ]
    for row in passed[:12]:
        lines.append(
            "- "
            f"{row.get('factor_id')} {row.get('symbol')} {row.get('regime')} "
            f"h={row.get('horizon_hours')} rank_ic={row.get('rank_ic')} "
            f"pearson_ic={row.get('pearson_ic')} "
            f"long_short_bps={row.get('long_short_bps')}"
        )
    return "\n".join(lines) + "\n"


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
                "orderbook_imbalance": first_value(
                    row,
                    ("orderbook_imbalance", "market_pressure_state"),
                ),
                "taker_sell_exhaustion": first_value(
                    row,
                    ("taker_sell_exhaustion", "trade_count_60m"),
                ),
                "volatility_climax": first_value(row, ("volatility_climax", "return_24h_bps")),
                "spread_normalization": first_value(
                    row,
                    ("spread_normalization", "avg_spread_bps_15m"),
                ),
                "cost_bps": cost.cost_bps,
                "cost_model": cost.cost_model,
                "future_4h_net_bps": futures[4],
                "future_8h_net_bps": futures[8],
                "future_12h_net_bps": futures[12],
                "future_24h_net_bps": futures[24],
                "label_status": (
                    "partial_complete"
                    if any(v is not None for v in futures.values())
                    else "pending"
                ),
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
        duplicate_rate = float_or_none(row.get("duplicate_rate")) or 0.0
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
        if duplicate_rate > 0.05:
            stage = "QUARANTINE"
            reasons = [
                "label_duplicate_rate_gt_5pct",
                *[reason for reason in reasons if reason != "paper_days_or_entries_insufficient"],
            ]
        forced_stage = _forced_research_stage(
            strategy_id=strategy_id,
            symbol=normalize_strategy_symbol(row.get("symbol")),
            sample_count=sample_count,
            avg=avg,
        )
        if forced_stage is not None:
            stage, forced_reason = forced_stage
            reasons = [forced_reason, *reasons]
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
        horizon = int(float_or_none(row.get("horizon_hours")) or 0)
        paper = _select_paper_row_for_horizon(paper_by_strategy.get(strategy_id, []), horizon)
        if paper is None and _strategy_uses_symbol_paper_proxy(strategy_id, symbol):
            paper = _select_paper_row_for_horizon(paper_by_symbol.get(symbol, []), horizon)
        if paper is None:
            continue
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
    conflicts = (
        frame.filter(pl.col("recommendation") == "QUARANTINE_BACKTEST_PAPER_CONFLICT")
        if not frame.is_empty()
        else pl.DataFrame()
    )
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
            f"backtest_avg={row.get('backtest_avg_net_bps')} "
            f"paper_avg={row.get('paper_avg_net_bps')} "
            f"recommendation={row.get('recommendation')}"
        )
    return "\n".join(lines) + "\n"


def build_backtest_vs_paper_gap_report(
    *,
    label_summary: pl.DataFrame,
    paper_daily: pl.DataFrame | None = None,
    bnb_paper_daily: pl.DataFrame | None = None,
) -> pl.DataFrame:
    paper_by_strategy, paper_by_symbol = _paper_daily_lookup(
        paper_daily, bnb_paper_daily
    )
    output: list[dict[str, Any]] = []
    for row in rows(label_summary):
        strategy_id = str(row.get("strategy_id") or "")
        symbol = normalize_strategy_symbol(row.get("symbol"))
        horizon = int(float_or_none(row.get("horizon_hours")) or 0)
        paper = _select_paper_row_for_horizon(
            paper_by_strategy.get(strategy_id, []), horizon
        )
        if paper is None and _strategy_uses_symbol_paper_proxy(strategy_id, symbol):
            paper = _select_paper_row_for_horizon(paper_by_symbol.get(symbol, []), horizon)
        paper = paper or {}
        backtest_avg = float_or_none(row.get("avg_net_bps"))
        paper_avg = _paper_avg_for_horizon(paper, horizon) if paper else None
        gap = (
            paper_avg - backtest_avg
            if paper_avg is not None and backtest_avg is not None
            else None
        )
        paper_entries = int(
            float_or_none(
                first_value(
                    paper,
                    (
                        "entry_count",
                        "would_enter_count",
                        "cumulative_would_enter_count",
                    ),
                )
            )
            or 0
        )
        paper_closed = int(
            float_or_none(
                first_value(
                    paper,
                    (
                        "closed_entries",
                        "paper_pnl_observed_count",
                        "cumulative_paper_pnl_observed_count",
                    ),
                )
            )
            or 0
        )
        paper_days = int(
            float_or_none(
                first_value(paper, ("paper_days_to_date", "paper_days"))
            )
            or 0
        )
        arrival_coverage = float_or_none(paper.get("arrival_mid_coverage"))
        causes: list[str] = []
        if not paper:
            causes.append("paper_tracker_or_daily_evidence_missing")
        if paper_days < 14:
            causes.append("paper_days_lt_14")
        if paper_closed < 20:
            causes.append("paper_closed_entries_lt_20")
        if arrival_coverage is None or arrival_coverage < 0.8:
            causes.append("arrival_mid_coverage_lt_0_80_or_missing")
        backtest_cost = str(row.get("cost_model") or "not_observable")
        paper_cost = str(paper.get("cost_source_mix") or "not_observable")
        if paper and not _cost_models_comparable(backtest_cost, paper_cost):
            causes.append("backtest_paper_cost_source_mismatch")
        if "one_bar_delay" not in str(row.get("data_leakage_check") or "") and (
            "visible_at_decision_time"
            not in str(row.get("data_leakage_check") or "")
        ):
            causes.append("backtest_entry_timing_not_proven")
        if gap is not None and abs(gap) > 100.0:
            causes.append("absolute_return_gap_gt_100bps")
        if (
            backtest_avg is not None
            and paper_avg is not None
            and backtest_avg * paper_avg < 0
        ):
            causes.append("return_sign_reversal")
        gap_status = _gap_status(
            paper=paper,
            paper_closed=paper_closed,
            arrival_coverage=arrival_coverage,
            gap=gap,
            sign_reversal="return_sign_reversal" in causes,
        )
        output.append(
            {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "horizon_hours": horizon,
                "backtest_sample_count": row.get("sample_count"),
                "backtest_complete_sample_count": row.get("complete_sample_count"),
                "backtest_avg_net_bps": backtest_avg,
                "backtest_p25_net_bps": row.get("p25_net_bps"),
                "backtest_cost_model": backtest_cost,
                "backtest_data_leakage_check": row.get("data_leakage_check"),
                "paper_strategy_id": paper.get("strategy_id"),
                "paper_days": paper_days,
                "paper_entries": paper_entries,
                "paper_closed_entries": paper_closed,
                "paper_avg_net_bps": paper_avg,
                "paper_cost_source_mix": paper_cost,
                "arrival_mid_coverage": arrival_coverage,
                "gap_bps": gap,
                "absolute_gap_bps": abs(gap) if gap is not None else None,
                "paper_to_backtest_ratio": (
                    paper_avg / backtest_avg
                    if paper_avg is not None and backtest_avg not in (None, 0.0)
                    else None
                ),
                "gap_status": gap_status,
                "root_causes": ";".join(causes) or "no_material_gap_detected",
                "recommendation": _gap_recommendation(gap_status),
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return frame_with_schema(output, BACKTEST_VS_PAPER_GAP_FIELDS)


def backtest_vs_paper_gap_report_md(frame: pl.DataFrame) -> str:
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows(frame):
        status_counts[str(row.get("gap_status") or "UNKNOWN")] += 1
    lines = [
        "# Backtest vs Paper Gap Report",
        "",
        "Read-only attribution of return, cost, timing, coverage and sample gaps.",
        "This report cannot promote a strategy to Canary or Live.",
        "",
        f"- rows: {frame.height}",
        *(f"- {key}: {value}" for key, value in sorted(status_counts.items())),
        "- live_order_effect: read_only_no_live_order",
    ]
    return "\n".join(lines) + "\n"


def _cost_models_comparable(backtest_cost: str, paper_cost: str) -> bool:
    left = backtest_cost.lower()
    right = paper_cost.lower()
    if right in {"", "not_observable", "[]"}:
        return False
    source_classes = ("actual", "mixed", "bootstrap", "proxy", "configured", "default")
    return any(token in left and token in right for token in source_classes)


def _gap_status(
    *,
    paper: dict[str, Any],
    paper_closed: int,
    arrival_coverage: float | None,
    gap: float | None,
    sign_reversal: bool,
) -> str:
    if not paper:
        return "PAPER_MISSING"
    if paper_closed < 20 or arrival_coverage is None or arrival_coverage < 0.8:
        return "PAPER_EVIDENCE_INSUFFICIENT"
    if gap is None:
        return "GAP_NOT_OBSERVABLE"
    if sign_reversal or abs(gap) > 100.0:
        return "MATERIAL_GAP"
    return "CONSISTENT_WITHIN_100BPS"


def _gap_recommendation(status: str) -> str:
    return {
        "PAPER_MISSING": "START_OR_REPAIR_GENERIC_PAPER_TRACKER",
        "PAPER_EVIDENCE_INSUFFICIENT": "CONTINUE_PAPER_OBSERVATION",
        "GAP_NOT_OBSERVABLE": "REPAIR_GAP_INPUTS",
        "MATERIAL_GAP": "QUARANTINE_AND_ATTRIBUTE_GAP",
        "CONSISTENT_WITHIN_100BPS": "KEEP_PAPER_OBSERVATION",
    }.get(status, "REVIEW_GAP_REPORT")


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
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        for row in rows(frame):
            strategy_id = str(row.get("strategy_id") or "").strip()
            symbol = normalize_strategy_symbol(row.get("symbol"))
            if not strategy_id and symbol == "UNKNOWN":
                continue
            if strategy_id:
                by_strategy.setdefault(strategy_id, []).append(row)
            if symbol != "UNKNOWN":
                by_symbol.setdefault(symbol, []).append(row)
    for bucket in (by_strategy, by_symbol):
        for key, values in bucket.items():
            bucket[key] = sorted(values, key=_paper_row_sort_key, reverse=True)
    return by_strategy, by_symbol


def _paper_row_sort_key(row: dict[str, Any]) -> tuple[str, float]:
    return (
        str(first_value(row, ("paper_date", "ts_utc", "generated_at")) or ""),
        float_or_none(row.get("entry_count")) or 0.0,
    )


def _select_paper_row_for_horizon(
    candidates: list[dict[str, Any]],
    horizon: int,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    for row in candidates:
        if _paper_avg_for_horizon(row, horizon) is not None:
            return row
    return candidates[0]


def _strategy_uses_symbol_paper_proxy(strategy_id: str, symbol: str) -> bool:
    text = strategy_id.upper()
    return symbol == "BNB-USDT" and (
        "BNB_STRONG_ALPHA6_BYPASS" in text
        or "FINAL_SCORE_ALPHA6_CONFLICT" in text
        or "BNB_F3_DOMINANT" in text
        or "BNB_RISK_ON_BUY" in text
    )


def _paper_avg_for_horizon(row: dict[str, Any], horizon: int) -> float | None:
    horizon_values = _json_dict(row.get("avg_paper_pnl_bps_by_horizon"))
    if horizon > 0 and horizon_values:
        for key in (f"{horizon}h", str(horizon), horizon):
            value = float_or_none(horizon_values.get(key))
            if value is not None:
                return value
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


def _json_dict(value: Any) -> dict[Any, Any]:
    if isinstance(value, dict):
        return value
    if value is None or value == "":
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _forced_research_stage(
    *,
    strategy_id: str,
    symbol: str,
    sample_count: int,
    avg: float | None,
) -> tuple[str, str] | None:
    text = strategy_id.upper()
    if "BNB_RISK_ON_BUY" in text:
        return "KILL_AS_ENTRY", "forced_rule_bnb_risk_on_buy_kill_as_entry"
    if "BNB_F3_DOMINANT" in text or (
        symbol == "BNB-USDT" and "F3_DOMINANT" in text
    ):
        return "KILL_AS_ENTRY", "forced_rule_bnb_f3_dominant_kill_as_entry"
    if "RISK_ON_MULTI_BUY" in text and sample_count >= 100 and (avg is not None and avg < 0):
        return "KILL_AS_LIVE_ENTRY", "forced_rule_risk_on_multi_buy_negative"
    if "FUTURES_PROXY" in text or (
        "FUTURES" in text and ("PROXY" in text or "HEDGE" in text or "DOWNTREND" in text)
    ):
        return "KILL_AS_PROXY", "forced_rule_futures_proxy_kill_as_proxy"
    if ("SOL_PROTECT" in text or "OLD_PULLBACK" in text) and (avg is not None and avg < 0):
        return "LOW_PRIORITY_OR_KILL", "forced_rule_long_term_negative_low_priority_or_kill"
    return None


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


def _factor_forward_summary_row(
    *,
    candidate: dict[str, Any],
    symbol: str,
    regime: str,
    horizon: int,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = [
        (sample["ts"], value, label)
        for sample in samples
        if (value := float_or_none(sample.get("value"))) is not None
        and (label := float_or_none(sample.get("future_net_bps"))) is not None
    ]
    labels = [label for _ts, _value, label in pairs]
    values = [value for _ts, value, _label in pairs]
    rank_ic = _forward_rank_ic(values, labels)
    pearson_ic = _forward_pearson(values, labels)
    long_short, p25, hit_rate = _forward_top_bottom_stats(pairs)
    recent_score = _forward_recent_score(pairs)
    return {
        "as_of_date": candidate.get("as_of_date"),
        "factor_id": candidate.get("factor_id"),
        "factor_family": candidate.get("factor_family"),
        "candidate_state": candidate.get("candidate_state"),
        "symbol": symbol,
        "regime": regime,
        "horizon_hours": horizon,
        "sample_count": len(pairs),
        "rank_ic": _round_float(rank_ic),
        "pearson_ic": _round_float(pearson_ic),
        "long_short_bps": _round_float(long_short),
        "p25_net_bps": _round_float(p25),
        "hit_rate": _round_float(hit_rate),
        "recent_7d_score": _round_float(recent_score),
        "regime_stability": None,
        "cost_adjusted_score": _round_float(long_short),
        "recommendation": None,
        "data_leakage_check": "pass_future_prices_used_only_for_labels",
        "live_order_effect": "none_read_only_research",
    }


def _enrich_factor_forward_rows(rows_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stability_by_factor: dict[str, float | None] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_in:
        grouped[str(row.get("factor_id") or "")].append(row)
    for factor_id, factor_rows in grouped.items():
        scored = [
            row
            for row in factor_rows
            if int(float_or_none(row.get("sample_count")) or 0) >= 30
            and str(row.get("regime") or "")
            in {"RISK_OFF", "SIDEWAYS", "RISK_ON_CONFIRMED", "TREND_UP"}
        ]
        if not factor_id or not scored:
            stability_by_factor[factor_id] = None
            continue
        positive = sum(
            1
            for row in scored
            if (float_or_none(row.get("rank_ic")) or 0.0) > 0
            and (float_or_none(row.get("cost_adjusted_score")) or 0.0) > 0
        )
        negative = sum(
            1
            for row in scored
            if (float_or_none(row.get("rank_ic")) or 0.0) < 0
            or (float_or_none(row.get("cost_adjusted_score")) or 0.0) < 0
        )
        stability_by_factor[factor_id] = (positive - negative) / len(scored)
    out: list[dict[str, Any]] = []
    for row in rows_in:
        enriched = dict(row)
        stability = stability_by_factor.get(str(row.get("factor_id") or ""))
        enriched["regime_stability"] = _round_float(stability)
        enriched["recommendation"] = _factor_forward_recommendation(
            sample_count=int(float_or_none(row.get("sample_count")) or 0),
            rank_ic=float_or_none(row.get("rank_ic")),
            long_short_bps=float_or_none(row.get("long_short_bps")),
            p25_net_bps=float_or_none(row.get("p25_net_bps")),
            hit_rate=float_or_none(row.get("hit_rate")),
            regime_stability=stability,
            cost_adjusted_score=float_or_none(row.get("cost_adjusted_score")),
        )
        out.append(enriched)
    return out


def _forward_regime_rows(frame: pl.DataFrame | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows(frame):
        ts = coerce_dt(first_value(row, ("as_of_ts", "created_at", "ts", "date", "as_of_date")))
        regime = _canonical_forward_regime(
            first_value(row, ("current_regime", "regime_state", "market_regime", "state"))
        )
        if ts is None or regime is None:
            continue
        item = dict(row)
        item["_ts"] = ts
        item["_regime"] = regime
        out.append(item)
    out.sort(key=lambda item: item["_ts"])
    return out


def _forward_regime_for_ts(regime_rows: list[dict[str, Any]], ts: Any) -> str | None:
    parsed = coerce_dt(ts)
    if parsed is None:
        return None
    candidates = [row for row in regime_rows if row["_ts"] <= parsed]
    return str(candidates[-1].get("_regime") or "") if candidates else None


def _derived_market_regime(bars: list[dict[str, Any]], ts: Any) -> str:
    parsed = coerce_dt(ts)
    if parsed is None:
        return "SIDEWAYS"
    current = _bar_close_at_or_after(bars, parsed)
    prior = _bar_close_at_or_after(bars, parsed - timedelta(hours=4))
    if current is None or prior is None or prior <= 0:
        return "SIDEWAYS"
    ret_bps = (current / prior - 1.0) * 10000.0
    if ret_bps <= -80:
        return "RISK_OFF"
    if ret_bps >= 120:
        return "RISK_ON_CONFIRMED"
    if ret_bps >= 50:
        return "TREND_UP"
    return "SIDEWAYS"


def _bar_close_at_or_after(bars: list[dict[str, Any]], ts: Any) -> float | None:
    parsed = coerce_dt(ts)
    if parsed is None:
        return None
    for row in bars:
        row_ts = coerce_dt(row.get("_ts") or row.get("ts") or row.get("ts_utc"))
        value = float_or_none(row.get("close")) or float_or_none(row.get("_close"))
        if row_ts is not None and row_ts >= parsed and value is not None:
            return value
    return None


def _canonical_forward_regime(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text or text in {"UNKNOWN", "NONE", "NAN"}:
        return None
    if "RISK_ON_CONFIRMED" in text:
        return "RISK_ON_CONFIRMED"
    if "RISK_ON" in text or "IMPULSE" in text:
        return "RISK_ON_CONFIRMED"
    if "RISK_OFF" in text or "DOWN" in text:
        return "RISK_OFF"
    if "UP" in text or text in {"TREND", "TRENDING"}:
        return "TREND_UP"
    if "SIDE" in text or "CHOP" in text or "PROTECT" in text or "LOW_VOL" in text:
        return "SIDEWAYS"
    if text in {"RISK_OFF", "SIDEWAYS", "RISK_ON_CONFIRMED", "TREND_UP"}:
        return text
    return "SIDEWAYS"


def _forward_top_bottom_stats(
    pairs: list[tuple[Any, float, float]],
    *,
    quantile: float = 0.2,
) -> tuple[float | None, float | None, float | None]:
    if not pairs:
        return None, None, None
    ordered = sorted(pairs, key=lambda item: item[1])
    bucket_size = max(1, int(len(ordered) * quantile))
    bottom = [label for _ts, _value, label in ordered[:bucket_size]]
    top = [label for _ts, _value, label in ordered[-bucket_size:]]
    return (
        (_forward_mean(top) - _forward_mean(bottom)) if top and bottom else None,
        _forward_percentile(top, 0.25) if top else None,
        (sum(1 for value in top if value > 0) / len(top)) if top else None,
    )


def _forward_rank_ic(values: list[float], labels: list[float]) -> float | None:
    if len(values) < 3 or len(values) != len(labels):
        return None
    return _forward_pearson(_forward_ranks(values), _forward_ranks(labels))


def _forward_ranks(values: list[float]) -> list[float]:
    ranked = [0.0] * len(values)
    for rank, (index, _value) in enumerate(
        sorted(enumerate(values), key=lambda item: item[1]),
        start=1,
    ):
        ranked[index] = float(rank)
    return ranked


def _forward_pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = _forward_mean(left)
    right_mean = _forward_mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=False))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = (left_var * right_var) ** 0.5
    return numerator / denominator if denominator > 0 else None


def _forward_recent_score(pairs: list[tuple[Any, float, float]]) -> float | None:
    parsed_pairs: list[tuple[Any, float]] = []
    for raw_ts, _value, label in pairs:
        parsed = coerce_dt(raw_ts)
        if parsed is not None:
            parsed_pairs.append((parsed, label))
    if not parsed_pairs:
        return None
    latest = max(ts for ts, _label in parsed_pairs)
    recent = [label for ts, label in parsed_pairs if ts >= latest - timedelta(days=7)]
    return _forward_mean(recent) if recent else None


def _factor_forward_recommendation(
    *,
    sample_count: int,
    rank_ic: float | None,
    long_short_bps: float | None,
    p25_net_bps: float | None,
    hit_rate: float | None,
    regime_stability: float | None,
    cost_adjusted_score: float | None,
) -> str:
    if sample_count < 30:
        return "NEEDS_MORE_FORWARD_SAMPLES"
    if (
        (rank_ic or 0.0) > 0.02
        and (long_short_bps or 0.0) > 0
        and (p25_net_bps is not None and p25_net_bps > -50)
        and (hit_rate or 0.0) > 0.50
        and (regime_stability or 0.0) > 0
        and (cost_adjusted_score or 0.0) > 0
    ):
        return "FORWARD_VALIDATION_PASS"
    return "FORWARD_VALIDATION_WEAK_OR_MIXED"


def _forward_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _forward_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


def _round_float(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _current_stage(strategy_id: str) -> str:
    text = strategy_id.upper()
    if "PAPER" in text:
        return "PAPER"
    if "SHADOW" in text:
        return "SHADOW"
    return "RESEARCH"
