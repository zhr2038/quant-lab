from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

SOURCE_NAME = "quant_lab.bnb_swing_exit_policy_review"
SCHEMA_VERSION = "bnb_swing_exit_policy_review.v0.2"
SUMMARY_SCHEMA_VERSION = "bnb_swing_exit_policy_summary.v0.3"
CONSISTENCY_SCHEMA_VERSION = "bnb_exit_policy_v5_vs_quant_lab_consistency.v0.2"
DEFAULT_ROUNDTRIP_COST_BPS = 30.0
HORIZONS = (4, 8, 12, 24)
FIXED_HOLD_FROM_ENTRY_HOURS = (6, 12, 24)
DELAYED_EXIT_HOURS = (6, 12, 24)
V5_PRICE_BEFORE_TOLERANCE = timedelta(minutes=5)
V5_PRICE_AFTER_TOLERANCE = timedelta(hours=2)
CONSISTENCY_TOLERANCE_BPS = 0.05
PROFIT_LOCK_BPS = (30, 50)
ATR_LOOKBACK_BARS = 14
ATR_MULTIPLIER = 2.0
MIN_SAMPLE_COUNT_FOR_EXIT_CHANGE = 10
MIN_SHADOW_HELP_RATE_FOR_EXIT_REVIEW = 0.60
MIN_AVG_SHADOW_IMPROVEMENT_BPS_FOR_EXIT_REVIEW = 50.0

V5_TRADE_EVENT_DATASET = Path("silver") / "v5_trade_event"
V5_BNB_PROFIT_LOCK_SHADOW_DATASET = Path("silver") / "v5_bnb_profit_lock_shadow"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
BNB_SWING_EXIT_POLICY_REVIEW_DATASET = Path("gold") / "bnb_swing_exit_policy_review"
BNB_SWING_EXIT_POLICY_SUMMARY_DATASET = Path("gold") / "bnb_swing_exit_policy_summary"
BNB_EXIT_POLICY_V5_VS_QUANT_LAB_CONSISTENCY_DATASET = (
    Path("gold") / "bnb_exit_policy_v5_vs_quant_lab_consistency"
)

REVIEW_SCHEMA: dict[str, Any] = {
    "contract_version": pl.Utf8,
    "schema_version": pl.Utf8,
    "quant_lab_git_commit": pl.Utf8,
    "source_version": pl.Utf8,
    "generated_at_utc": pl.Datetime(time_zone="UTC"),
    "generated_from_bundle_id": pl.Utf8,
    "as_of_date": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "run_id": pl.Utf8,
    "source_entry_id": pl.Utf8,
    "entry_ts": pl.Datetime(time_zone="UTC"),
    "entry_px": pl.Float64,
    "highest_px_after_entry": pl.Float64,
    "max_unrealized_bps": pl.Float64,
    "actual_exit_ts": pl.Datetime(time_zone="UTC"),
    "actual_exit_px": pl.Float64,
    "actual_exit_net_bps": pl.Float64,
    "fixed_hold_6h_from_entry_net_bps": pl.Float64,
    "fixed_hold_12h_from_entry_net_bps": pl.Float64,
    "fixed_hold_24h_from_entry_net_bps": pl.Float64,
    "fixed_hold_4h_net_bps": pl.Float64,
    "fixed_hold_8h_net_bps": pl.Float64,
    "fixed_hold_12h_net_bps": pl.Float64,
    "fixed_hold_24h_net_bps": pl.Float64,
    "profit_lock_30bps_exit": pl.Float64,
    "profit_lock_50bps_exit": pl.Float64,
    "delayed_exit_6h_from_actual_exit_net_bps": pl.Float64,
    "delayed_exit_12h_from_actual_exit_net_bps": pl.Float64,
    "delayed_exit_24h_from_actual_exit_net_bps": pl.Float64,
    "delayed_exit_6h_net_bps": pl.Float64,
    "delayed_exit_12h_net_bps": pl.Float64,
    "delayed_exit_24h_net_bps": pl.Float64,
    "trailing_atr_exit": pl.Float64,
    "best_exit_policy": pl.Utf8,
    "best_shadow_exit_policy": pl.Utf8,
    "best_exit_net_bps": pl.Float64,
    "delta_vs_actual_bps": pl.Float64,
    "exit_reason": pl.Utf8,
    "selected_roundtrip_cost_bps": pl.Float64,
    "diagnosis": pl.Utf8,
    "status": pl.Utf8,
    "duplicate_group_key": pl.Utf8,
    "duplicate_row_count": pl.Int64,
    "selected_for_summary": pl.Boolean,
    "summary_eligible": pl.Boolean,
    "review_row_source": pl.Utf8,
    "v5_vs_quant_lab_consistency_status": pl.Utf8,
    "v5_vs_quant_lab_mismatch_reason": pl.Utf8,
    "v5_fixed_hold_6h_from_entry_net_bps": pl.Float64,
    "quant_lab_fixed_hold_6h_from_entry_net_bps": pl.Float64,
    "diff_fixed_hold_6h_from_entry_net_bps": pl.Float64,
    "v5_fixed_hold_12h_from_entry_net_bps": pl.Float64,
    "quant_lab_fixed_hold_12h_from_entry_net_bps": pl.Float64,
    "diff_fixed_hold_12h_from_entry_net_bps": pl.Float64,
    "v5_fixed_hold_24h_from_entry_net_bps": pl.Float64,
    "quant_lab_fixed_hold_24h_from_entry_net_bps": pl.Float64,
    "diff_fixed_hold_24h_from_entry_net_bps": pl.Float64,
    "v5_delayed_exit_6h_from_actual_exit_net_bps": pl.Float64,
    "quant_lab_delayed_exit_6h_from_actual_exit_net_bps": pl.Float64,
    "diff_delayed_exit_6h_from_actual_exit_net_bps": pl.Float64,
    "v5_delayed_exit_12h_from_actual_exit_net_bps": pl.Float64,
    "quant_lab_delayed_exit_12h_from_actual_exit_net_bps": pl.Float64,
    "diff_delayed_exit_12h_from_actual_exit_net_bps": pl.Float64,
    "v5_delayed_exit_24h_from_actual_exit_net_bps": pl.Float64,
    "quant_lab_delayed_exit_24h_from_actual_exit_net_bps": pl.Float64,
    "diff_delayed_exit_24h_from_actual_exit_net_bps": pl.Float64,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

CONSISTENCY_SCHEMA: dict[str, Any] = {
    "contract_version": pl.Utf8,
    "schema_version": pl.Utf8,
    "quant_lab_git_commit": pl.Utf8,
    "source_version": pl.Utf8,
    "generated_at_utc": pl.Datetime(time_zone="UTC"),
    "generated_from_bundle_id": pl.Utf8,
    "as_of_date": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "run_id": pl.Utf8,
    "source_entry_id": pl.Utf8,
    "entry_ts": pl.Datetime(time_zone="UTC"),
    "actual_exit_ts": pl.Datetime(time_zone="UTC"),
    "duplicate_group_key": pl.Utf8,
    "duplicate_row_count": pl.Int64,
    "v5_shadow_row_present": pl.Boolean,
    "quant_lab_recomputed_row_present": pl.Boolean,
    "consistency_status": pl.Utf8,
    "mismatch_reason": pl.Utf8,
    "selected_for_summary_allowed": pl.Boolean,
    "compared_field_count": pl.Int64,
    "mismatch_field_count": pl.Int64,
    "v5_fixed_hold_6h_from_entry_net_bps": pl.Float64,
    "quant_lab_fixed_hold_6h_from_entry_net_bps": pl.Float64,
    "diff_fixed_hold_6h_from_entry_net_bps": pl.Float64,
    "v5_fixed_hold_12h_from_entry_net_bps": pl.Float64,
    "quant_lab_fixed_hold_12h_from_entry_net_bps": pl.Float64,
    "diff_fixed_hold_12h_from_entry_net_bps": pl.Float64,
    "v5_fixed_hold_24h_from_entry_net_bps": pl.Float64,
    "quant_lab_fixed_hold_24h_from_entry_net_bps": pl.Float64,
    "diff_fixed_hold_24h_from_entry_net_bps": pl.Float64,
    "v5_delayed_exit_6h_from_actual_exit_net_bps": pl.Float64,
    "quant_lab_delayed_exit_6h_from_actual_exit_net_bps": pl.Float64,
    "diff_delayed_exit_6h_from_actual_exit_net_bps": pl.Float64,
    "v5_delayed_exit_12h_from_actual_exit_net_bps": pl.Float64,
    "quant_lab_delayed_exit_12h_from_actual_exit_net_bps": pl.Float64,
    "diff_delayed_exit_12h_from_actual_exit_net_bps": pl.Float64,
    "v5_delayed_exit_24h_from_actual_exit_net_bps": pl.Float64,
    "quant_lab_delayed_exit_24h_from_actual_exit_net_bps": pl.Float64,
    "diff_delayed_exit_24h_from_actual_exit_net_bps": pl.Float64,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

SUMMARY_SCHEMA: dict[str, Any] = {
    "contract_version": pl.Utf8,
    "schema_version": pl.Utf8,
    "quant_lab_git_commit": pl.Utf8,
    "source_version": pl.Utf8,
    "generated_at_utc": pl.Datetime(time_zone="UTC"),
    "generated_from_bundle_id": pl.Utf8,
    "as_of_date": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "sample_count": pl.Int64,
    "min_sample_count_for_exit_change": pl.Int64,
    "avg_actual_exit_net_bps": pl.Float64,
    "avg_max_unrealized_bps": pl.Float64,
    "avg_delta_best_vs_actual_bps": pl.Float64,
    "profit_lock_better_count": pl.Int64,
    "delayed_exit_better_count": pl.Int64,
    "trailing_better_count": pl.Int64,
    "shadow_help_count": pl.Int64,
    "shadow_help_rate": pl.Float64,
    "avg_best_shadow_improvement_bps": pl.Float64,
    "recommendation": pl.Utf8,
    "review_reason": pl.Utf8,
    "sample_count_gate_met_for_exit_change_review": pl.Boolean,
    "best_exit_policy_mix": pl.Utf8,
    "best_shadow_exit_policy_mix": pl.Utf8,
    "status": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


class BnbSwingExitPolicyReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of_date: str
    review_rows: int = Field(ge=0)
    summary_rows: int = Field(ge=0)
    consistency_rows: int = Field(ge=0)
    status: str
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Context:
    as_of_date: date
    generated_at: datetime
    generated_from_bundle_id: str
    git_commit: str | None


def build_and_publish_bnb_swing_exit_policy_review(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> BnbSwingExitPolicyReviewResult:
    root = Path(lake_root)
    ctx = _Context(
        as_of_date=_parse_day(as_of_date),
        generated_at=datetime.now(UTC),
        generated_from_bundle_id=_latest_bundle_id(root),
        git_commit=_git_commit(),
    )
    trades = read_parquet_dataset(root / V5_TRADE_EVENT_DATASET)
    profit_lock_shadow = read_parquet_dataset(root / V5_BNB_PROFIT_LOCK_SHADOW_DATASET)
    market = read_parquet_dataset(root / MARKET_BAR_DATASET)
    review = build_bnb_swing_exit_policy_review(
        trade_events=trades,
        profit_lock_shadow=profit_lock_shadow,
        market_bars=market,
        ctx=ctx,
    )
    consistency = build_bnb_exit_policy_v5_vs_quant_lab_consistency(review, ctx=ctx)
    summary = build_bnb_swing_exit_policy_summary(review, ctx=ctx)
    write_parquet_dataset(review, root / BNB_SWING_EXIT_POLICY_REVIEW_DATASET)
    write_parquet_dataset(
        consistency,
        root / BNB_EXIT_POLICY_V5_VS_QUANT_LAB_CONSISTENCY_DATASET,
    )
    write_parquet_dataset(summary, root / BNB_SWING_EXIT_POLICY_SUMMARY_DATASET)
    status = (
        str(summary["status"][0])
        if not summary.is_empty() and "status" in summary.columns
        else "RESEARCH_ONLY"
    )
    warnings = [] if review.height else ["bnb_swing_trade_missing"]
    if (
        not consistency.is_empty()
        and "consistency_status" in consistency.columns
        and "MISMATCH" in set(consistency["consistency_status"].to_list())
    ):
        warnings.append("bnb_exit_policy_v5_quant_lab_mismatch")
    return BnbSwingExitPolicyReviewResult(
        as_of_date=ctx.as_of_date.isoformat(),
        review_rows=review.height,
        summary_rows=summary.height,
        consistency_rows=consistency.height,
        status=status,
        warnings=warnings,
    )


def build_bnb_swing_exit_policy_review(
    *,
    trade_events: pl.DataFrame,
    profit_lock_shadow: pl.DataFrame | None = None,
    market_bars: pl.DataFrame,
    ctx: _Context,
) -> pl.DataFrame:
    shadow_frame = profit_lock_shadow if profit_lock_shadow is not None else pl.DataFrame()
    if trade_events.is_empty() and shadow_frame.is_empty():
        return pl.DataFrame(schema=REVIEW_SCHEMA)
    bars = _market_bar_index(market_bars)
    bnb_trades = _bnb_trade_rows(trade_events)
    entries = [row for row in bnb_trades if _is_bnb_swing_entry(row)]
    exits = [row for row in bnb_trades if _is_bnb_exit(row)]
    rows: list[dict[str, Any]] = []
    for shadow in _bnb_profit_lock_shadow_rows(shadow_frame):
        shadow_row = _review_row_from_profit_lock_shadow(shadow, ctx=ctx)
        if shadow_row is None:
            continue
        rows.append(shadow_row)
    for entry in entries:
        entry_ts = _parse_datetime(entry.get("ts_utc") or entry.get("ts"))
        entry_px = _float_or_none(entry.get("price") or entry.get("fill_px"))
        if entry_ts is None or entry_px is None or entry_px <= 0:
            continue
        exit_row = _matching_exit(entry, exits)
        actual_exit_ts = (
            _parse_datetime(exit_row.get("ts_utc") or exit_row.get("ts"))
            if exit_row
            else None
        )
        actual_exit_px = (
            _float_or_none(exit_row.get("price") or exit_row.get("fill_px")) if exit_row else None
        )
        cost_bps = _roundtrip_cost_bps(entry, exit_row, entry_px)
        actual_net = _actual_exit_net_bps(
            entry_px=entry_px,
            exit_px=actual_exit_px,
            cost_bps=cost_bps,
            entry=entry,
            exit_row=exit_row,
        )
        window_end = actual_exit_ts or entry_ts + timedelta(hours=24)
        highest_px = _highest_after_entry(
            bars.get("BNB-USDT", []),
            entry_ts=entry_ts,
            end_ts=window_end,
            fallback=actual_exit_px or entry_px,
        )
        max_unrealized = (
            (highest_px / entry_px - 1.0) * 10_000.0 if highest_px is not None else None
        )
        fixed_hold = {
            horizon: _fixed_hold_net_bps(
                bars.get("BNB-USDT", []),
                entry_ts=entry_ts,
                entry_px=entry_px,
                horizon_hours=horizon,
                cost_bps=cost_bps,
            )
            for horizon in HORIZONS
        }
        profit_lock = {
            bps: _profit_lock_exit_net_bps(
                bars.get("BNB-USDT", []),
                entry_ts=entry_ts,
                end_ts=window_end,
                entry_px=entry_px,
                threshold_bps=bps,
                cost_bps=cost_bps,
            )
            for bps in PROFIT_LOCK_BPS
        }
        delayed_exit = {
            hours: _delayed_exit_net_bps(
                bars.get("BNB-USDT", []),
                actual_exit_ts=actual_exit_ts,
                entry_px=entry_px,
                delay_hours=hours,
                cost_bps=cost_bps,
            )
            for hours in DELAYED_EXIT_HOURS
        }
        fixed_hold_from_entry = {
            hours: _fixed_hold_net_bps(
                bars.get("BNB-USDT", []),
                entry_ts=entry_ts,
                entry_px=entry_px,
                horizon_hours=hours,
                cost_bps=cost_bps,
            )
            for hours in FIXED_HOLD_FROM_ENTRY_HOURS
        }
        trailing_atr = _trailing_atr_exit_net_bps(
            bars.get("BNB-USDT", []),
            entry_ts=entry_ts,
            end_ts=window_end,
            entry_px=entry_px,
            cost_bps=cost_bps,
        )
        alternatives = {
            "actual_exit": actual_net,
            **{
                f"fixed_hold_{horizon}h_from_entry": value
                for horizon, value in fixed_hold.items()
            },
            **{
                f"fixed_hold_{hours}h_from_entry": value
                for hours, value in fixed_hold_from_entry.items()
            },
            "profit_lock_30bps": profit_lock[30],
            "profit_lock_50bps": profit_lock[50],
            **{
                f"delayed_exit_{hours}h_from_actual_exit": value
                for hours, value in delayed_exit.items()
            },
            "trailing_atr": trailing_atr,
        }
        best_policy, best_value = _best_policy(alternatives)
        delta = (
            best_value - actual_net if best_value is not None and actual_net is not None else None
        )
        exit_reason = str(
            _field(entry, "exit_reason", "close_reason", "reason")
            or _field(exit_row or {}, "exit_reason", "close_reason", "reason")
            or ""
        )
        diagnosis = _diagnosis(
            actual_exit_net_bps=actual_net,
            max_unrealized_bps=max_unrealized,
            best_exit_policy=best_policy,
            delta_vs_actual_bps=delta,
            exit_reason=exit_reason,
        )
        rows.append(
            _common(ctx)
            | {
                "as_of_date": ctx.as_of_date.isoformat(),
                "strategy_candidate": "v5.bnb_swing_exit_policy_review",
                "symbol": "BNB-USDT",
                "run_id": str(entry.get("run_id") or ""),
                "source_entry_id": _source_entry_id(entry, entry_ts),
                "entry_ts": entry_ts,
                "entry_px": entry_px,
                "highest_px_after_entry": highest_px,
                "max_unrealized_bps": max_unrealized,
                "actual_exit_ts": actual_exit_ts,
                "actual_exit_px": actual_exit_px,
                "actual_exit_net_bps": actual_net,
                "fixed_hold_6h_from_entry_net_bps": fixed_hold_from_entry[6],
                "fixed_hold_12h_from_entry_net_bps": fixed_hold_from_entry[12],
                "fixed_hold_24h_from_entry_net_bps": fixed_hold_from_entry[24],
                "fixed_hold_4h_net_bps": fixed_hold[4],
                "fixed_hold_8h_net_bps": fixed_hold[8],
                "fixed_hold_12h_net_bps": fixed_hold[12],
                "fixed_hold_24h_net_bps": fixed_hold[24],
                "profit_lock_30bps_exit": profit_lock[30],
                "profit_lock_50bps_exit": profit_lock[50],
                "delayed_exit_6h_from_actual_exit_net_bps": delayed_exit[6],
                "delayed_exit_12h_from_actual_exit_net_bps": delayed_exit[12],
                "delayed_exit_24h_from_actual_exit_net_bps": delayed_exit[24],
                "delayed_exit_6h_net_bps": delayed_exit[6],
                "delayed_exit_12h_net_bps": delayed_exit[12],
                "delayed_exit_24h_net_bps": delayed_exit[24],
                "trailing_atr_exit": trailing_atr,
                "best_exit_policy": best_policy,
                "best_shadow_exit_policy": best_policy,
                "best_exit_net_bps": best_value,
                "delta_vs_actual_bps": delta,
                "exit_reason": exit_reason,
                "selected_roundtrip_cost_bps": cost_bps,
                "diagnosis": diagnosis,
                "status": "REVIEW",
                "review_row_source": "quant_lab_recomputed",
                "created_at": ctx.generated_at,
                "source": SOURCE_NAME,
            }
        )
    if not rows:
        return pl.DataFrame(schema=REVIEW_SCHEMA)
    return pl.DataFrame(_dedupe_review_rows(rows), schema=REVIEW_SCHEMA, orient="row")


def build_bnb_swing_exit_policy_summary(
    review: pl.DataFrame,
    *,
    ctx: _Context,
) -> pl.DataFrame:
    rows = _selected_review_rows(review)
    sample_count = len(rows)
    reasons = ["read_only_research_no_live_exit_change"]
    if not rows:
        if not review.is_empty():
            reasons.append("bnb_exit_policy_v5_quant_lab_mismatch_excluded")
        else:
            reasons.append("no_bnb_swing_roundtrip")
    profit_better = [
        row
        for row in rows
        if str(row.get("best_exit_policy") or "").startswith("profit_lock")
        and (_float_or_none(row.get("delta_vs_actual_bps")) or 0.0) > 0
    ]
    trailing_better = [
        row
        for row in rows
        if str(row.get("best_exit_policy") or "") == "trailing_atr"
        and (_float_or_none(row.get("delta_vs_actual_bps")) or 0.0) > 0
    ]
    delayed_better = [
        row
        for row in rows
        if str(row.get("best_exit_policy") or "").startswith("delayed_exit")
        and (_float_or_none(row.get("delta_vs_actual_bps")) or 0.0) > 0
    ]
    shadow_help_rows = [
        row
        for row in rows
        if str(row.get("best_shadow_exit_policy") or "")
        not in {"", "actual_exit", "not_observable"}
        and (_float_or_none(row.get("delta_vs_actual_bps")) or 0.0) > 0
    ]
    improvement_values = [
        value
        for value in (_float_or_none(row.get("delta_vs_actual_bps")) for row in rows)
        if value is not None
    ]
    shadow_help_rate = len(shadow_help_rows) / sample_count if sample_count else None
    avg_best_shadow_improvement = _mean(improvement_values)
    if profit_better:
        reasons.append("profit_lock_would_improve_exit")
    if delayed_better:
        reasons.append("delayed_exit_would_improve_exit")
    if trailing_better:
        reasons.append("atr_trailing_variant_would_improve_exit")
    sample_gate_met = sample_count >= MIN_SAMPLE_COUNT_FOR_EXIT_CHANGE
    clear_shadow_advantage = (
        shadow_help_rate is not None
        and shadow_help_rate >= MIN_SHADOW_HELP_RATE_FOR_EXIT_REVIEW
        and avg_best_shadow_improvement is not None
        and avg_best_shadow_improvement >= MIN_AVG_SHADOW_IMPROVEMENT_BPS_FOR_EXIT_REVIEW
    )
    if not sample_gate_met:
        reasons.append("insufficient_sample_count_for_exit_change")
        recommendation = "collect_more_samples"
        review_reason = "sample_count_lt_10"
        decision = "RESEARCH_ONLY"
    elif clear_shadow_advantage:
        reasons.append("sample_gate_met_shadow_exit_outperforms_actual")
        recommendation = "REVIEW_EXIT_POLICY"
        review_reason = "sample_gate_met_shadow_exit_outperforms_actual"
        decision = "REVIEW_EXIT_POLICY"
    else:
        reasons.append("sample_gate_met_no_clear_shadow_advantage")
        recommendation = "no_change_recommended"
        review_reason = "sample_gate_met_no_clear_shadow_advantage"
        decision = "RESEARCH_ONLY"
    return pl.DataFrame(
        [
            _common(ctx, schema_version=SUMMARY_SCHEMA_VERSION)
            | {
                "as_of_date": ctx.as_of_date.isoformat(),
                "strategy_candidate": "v5.bnb_swing_exit_policy_review",
                "symbol": "BNB-USDT",
                "sample_count": sample_count,
                "min_sample_count_for_exit_change": MIN_SAMPLE_COUNT_FOR_EXIT_CHANGE,
                "avg_actual_exit_net_bps": _mean(row.get("actual_exit_net_bps") for row in rows),
                "avg_max_unrealized_bps": _mean(row.get("max_unrealized_bps") for row in rows),
                "avg_delta_best_vs_actual_bps": _mean(
                    row.get("delta_vs_actual_bps") for row in rows
                ),
                "profit_lock_better_count": len(profit_better),
                "delayed_exit_better_count": len(delayed_better),
                "trailing_better_count": len(trailing_better),
                "shadow_help_count": len(shadow_help_rows),
                "shadow_help_rate": shadow_help_rate,
                "avg_best_shadow_improvement_bps": avg_best_shadow_improvement,
                "recommendation": recommendation,
                "review_reason": review_reason,
                "sample_count_gate_met_for_exit_change_review": sample_gate_met,
                "best_exit_policy_mix": safe_json_dumps(
                    _counts(row.get("best_exit_policy") for row in rows)
                ),
                "best_shadow_exit_policy_mix": safe_json_dumps(
                    _counts(row.get("best_shadow_exit_policy") for row in rows)
                ),
                "status": "REVIEW" if rows else "RESEARCH_ONLY",
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
                "created_at": ctx.generated_at,
                "source": SOURCE_NAME,
            }
        ],
        schema=SUMMARY_SCHEMA,
        orient="row",
    )


def build_bnb_exit_policy_v5_vs_quant_lab_consistency(
    review: pl.DataFrame,
    *,
    ctx: _Context,
) -> pl.DataFrame:
    if review.is_empty():
        return pl.DataFrame(schema=CONSISTENCY_SCHEMA)
    rows: list[dict[str, Any]] = []
    for row in review.to_dicts():
        compared = 0
        mismatches = 0
        for hours in FIXED_HOLD_FROM_ENTRY_HOURS:
            diff = _float_or_none(
                row.get(f"diff_fixed_hold_{hours}h_from_entry_net_bps")
            )
            if diff is None:
                continue
            compared += 1
            if abs(diff) > CONSISTENCY_TOLERANCE_BPS:
                mismatches += 1
        for hours in DELAYED_EXIT_HOURS:
            diff = _float_or_none(
                row.get(f"diff_delayed_exit_{hours}h_from_actual_exit_net_bps")
            )
            if diff is None:
                continue
            compared += 1
            if abs(diff) > CONSISTENCY_TOLERANCE_BPS:
                mismatches += 1
        rows.append(
            _common(ctx, schema_version=CONSISTENCY_SCHEMA_VERSION)
            | {
                "as_of_date": str(row.get("as_of_date") or ctx.as_of_date.isoformat()),
                "strategy_candidate": str(
                    row.get("strategy_candidate") or "v5.bnb_swing_exit_policy_review"
                ),
                "symbol": str(row.get("symbol") or "BNB-USDT"),
                "run_id": str(row.get("run_id") or ""),
                "source_entry_id": str(row.get("source_entry_id") or ""),
                "entry_ts": _parse_datetime(row.get("entry_ts")),
                "actual_exit_ts": _parse_datetime(row.get("actual_exit_ts")),
                "duplicate_group_key": str(row.get("duplicate_group_key") or ""),
                "duplicate_row_count": int(_float_or_none(row.get("duplicate_row_count")) or 0),
                "v5_shadow_row_present": _observable(
                    row.get("v5_fixed_hold_6h_from_entry_net_bps")
                )
                or _observable(row.get("v5_fixed_hold_12h_from_entry_net_bps"))
                or _observable(row.get("v5_fixed_hold_24h_from_entry_net_bps"))
                or _observable(
                    row.get("v5_delayed_exit_6h_from_actual_exit_net_bps")
                )
                or _observable(row.get("v5_delayed_exit_12h_from_actual_exit_net_bps"))
                or _observable(row.get("v5_delayed_exit_24h_from_actual_exit_net_bps")),
                "quant_lab_recomputed_row_present": _observable(
                    row.get("quant_lab_fixed_hold_6h_from_entry_net_bps")
                )
                or _observable(row.get("quant_lab_fixed_hold_12h_from_entry_net_bps"))
                or _observable(row.get("quant_lab_fixed_hold_24h_from_entry_net_bps"))
                or _observable(
                    row.get("quant_lab_delayed_exit_6h_from_actual_exit_net_bps")
                )
                or _observable(row.get("quant_lab_delayed_exit_12h_from_actual_exit_net_bps"))
                or _observable(row.get("quant_lab_delayed_exit_24h_from_actual_exit_net_bps")),
                "consistency_status": str(
                    row.get("v5_vs_quant_lab_consistency_status") or "NOT_APPLICABLE"
                ),
                "mismatch_reason": str(row.get("v5_vs_quant_lab_mismatch_reason") or ""),
                "selected_for_summary_allowed": bool(row.get("summary_eligible")),
                "compared_field_count": compared,
                "mismatch_field_count": mismatches,
                "v5_fixed_hold_6h_from_entry_net_bps": _float_or_none(
                    row.get("v5_fixed_hold_6h_from_entry_net_bps")
                ),
                "quant_lab_fixed_hold_6h_from_entry_net_bps": _float_or_none(
                    row.get("quant_lab_fixed_hold_6h_from_entry_net_bps")
                ),
                "diff_fixed_hold_6h_from_entry_net_bps": _float_or_none(
                    row.get("diff_fixed_hold_6h_from_entry_net_bps")
                ),
                "v5_fixed_hold_12h_from_entry_net_bps": _float_or_none(
                    row.get("v5_fixed_hold_12h_from_entry_net_bps")
                ),
                "quant_lab_fixed_hold_12h_from_entry_net_bps": _float_or_none(
                    row.get("quant_lab_fixed_hold_12h_from_entry_net_bps")
                ),
                "diff_fixed_hold_12h_from_entry_net_bps": _float_or_none(
                    row.get("diff_fixed_hold_12h_from_entry_net_bps")
                ),
                "v5_fixed_hold_24h_from_entry_net_bps": _float_or_none(
                    row.get("v5_fixed_hold_24h_from_entry_net_bps")
                ),
                "quant_lab_fixed_hold_24h_from_entry_net_bps": _float_or_none(
                    row.get("quant_lab_fixed_hold_24h_from_entry_net_bps")
                ),
                "diff_fixed_hold_24h_from_entry_net_bps": _float_or_none(
                    row.get("diff_fixed_hold_24h_from_entry_net_bps")
                ),
                "v5_delayed_exit_6h_from_actual_exit_net_bps": _float_or_none(
                    row.get("v5_delayed_exit_6h_from_actual_exit_net_bps")
                ),
                "quant_lab_delayed_exit_6h_from_actual_exit_net_bps": _float_or_none(
                    row.get("quant_lab_delayed_exit_6h_from_actual_exit_net_bps")
                ),
                "diff_delayed_exit_6h_from_actual_exit_net_bps": _float_or_none(
                    row.get("diff_delayed_exit_6h_from_actual_exit_net_bps")
                ),
                "v5_delayed_exit_12h_from_actual_exit_net_bps": _float_or_none(
                    row.get("v5_delayed_exit_12h_from_actual_exit_net_bps")
                ),
                "quant_lab_delayed_exit_12h_from_actual_exit_net_bps": _float_or_none(
                    row.get("quant_lab_delayed_exit_12h_from_actual_exit_net_bps")
                ),
                "diff_delayed_exit_12h_from_actual_exit_net_bps": _float_or_none(
                    row.get("diff_delayed_exit_12h_from_actual_exit_net_bps")
                ),
                "v5_delayed_exit_24h_from_actual_exit_net_bps": _float_or_none(
                    row.get("v5_delayed_exit_24h_from_actual_exit_net_bps")
                ),
                "quant_lab_delayed_exit_24h_from_actual_exit_net_bps": _float_or_none(
                    row.get("quant_lab_delayed_exit_24h_from_actual_exit_net_bps")
                ),
                "diff_delayed_exit_24h_from_actual_exit_net_bps": _float_or_none(
                    row.get("diff_delayed_exit_24h_from_actual_exit_net_bps")
                ),
                "created_at": ctx.generated_at,
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=CONSISTENCY_SCHEMA, orient="row")


def bnb_swing_exit_policy_summary_md(summary: pl.DataFrame, review: pl.DataFrame) -> str:
    if summary.is_empty():
        return (
            "# BNB Swing Exit Policy Review\n\n"
            "- status: RESEARCH_ONLY\n"
            "- reason: no summary rows\n"
            "- safety: read-only research, no V5 live exit changes\n"
        )
    row = summary.to_dicts()[0]
    lines = [
        "# BNB Swing Exit Policy Review",
        "",
        "- safety: read-only research, no V5 live exit changes",
        f"- status: {row.get('status')}",
        f"- decision: {row.get('decision')}",
        f"- sample_count: {row.get('sample_count')}",
        f"- min_sample_count_for_exit_change: {row.get('min_sample_count_for_exit_change')}",
        f"- avg_actual_exit_net_bps: {_fmt(row.get('avg_actual_exit_net_bps'))}",
        f"- avg_max_unrealized_bps: {_fmt(row.get('avg_max_unrealized_bps'))}",
        f"- avg_delta_best_vs_actual_bps: {_fmt(row.get('avg_delta_best_vs_actual_bps'))}",
        f"- profit_lock_better_count: {row.get('profit_lock_better_count')}",
        f"- delayed_exit_better_count: {row.get('delayed_exit_better_count')}",
        f"- trailing_better_count: {row.get('trailing_better_count')}",
        f"- shadow_help_count: {row.get('shadow_help_count')}",
        f"- shadow_help_rate: {_fmt(row.get('shadow_help_rate'))}",
        f"- avg_best_shadow_improvement_bps: {_fmt(row.get('avg_best_shadow_improvement_bps'))}",
        f"- sample_count_gate_met_for_exit_change_review: "
        f"{row.get('sample_count_gate_met_for_exit_change_review')}",
        f"- recommendation: {row.get('recommendation')}",
        f"- review_reason: {row.get('review_reason')}",
        f"- best_exit_policy_mix: {row.get('best_exit_policy_mix')}",
        f"- best_shadow_exit_policy_mix: {row.get('best_shadow_exit_policy_mix')}",
        f"- decision_reasons: {row.get('decision_reasons')}",
    ]
    latest: dict[str, Any] = {}
    if not review.is_empty():
        latest = _latest_selected_review_row(review)
    if latest:
        lines.extend(
            [
                "",
                "## Latest BNB Swing",
                f"- entry_ts: {latest.get('entry_ts')}",
                f"- entry_px: {_fmt(latest.get('entry_px'))}",
                f"- actual_exit_net_bps: {_fmt(latest.get('actual_exit_net_bps'))}",
                f"- max_unrealized_bps: {_fmt(latest.get('max_unrealized_bps'))}",
                f"- best_exit_policy: {latest.get('best_exit_policy')}",
                f"- source_entry_id: {latest.get('source_entry_id')}",
                f"- duplicate_group_key: {latest.get('duplicate_group_key')}",
                f"- duplicate_row_count: {latest.get('duplicate_row_count')}",
                f"- fixed_hold_12h_from_entry_net_bps: "
                f"{_fmt(latest.get('fixed_hold_12h_from_entry_net_bps'))}",
                f"- delayed_exit_12h_from_actual_exit_net_bps: "
                f"{_fmt(latest.get('delayed_exit_12h_from_actual_exit_net_bps'))}",
                f"- v5_vs_quant_lab_consistency_status: "
                f"{latest.get('v5_vs_quant_lab_consistency_status')}",
                f"- v5_vs_quant_lab_mismatch_reason: "
                f"{latest.get('v5_vs_quant_lab_mismatch_reason')}",
                f"- delta_vs_actual_bps: {_fmt(latest.get('delta_vs_actual_bps'))}",
                f"- diagnosis: {latest.get('diagnosis')}",
            ]
        )
    return "\n".join(lines) + "\n"


def _dedupe_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        key = _duplicate_group_key(row)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    selected: list[dict[str, Any]] = []
    for key in order:
        candidates = grouped[key]
        consistency = _consistency_for_duplicate_group(candidates)
        summary_eligible = consistency["v5_vs_quant_lab_consistency_status"] != "MISMATCH"
        best = max(
            enumerate(candidates),
            key=lambda item: (_review_completeness_score(item[1]), item[0]),
        )[1]
        selected.append(
            best
            | {
                "duplicate_group_key": key,
                "duplicate_row_count": len(candidates),
                "selected_for_summary": summary_eligible,
                "summary_eligible": summary_eligible,
            }
            | consistency
        )
    return selected


def _selected_review_rows(review: pl.DataFrame) -> list[dict[str, Any]]:
    if review.is_empty():
        return []
    rows = review.to_dicts()
    if "selected_for_summary" in review.columns:
        return [row for row in rows if bool(row.get("selected_for_summary"))]
    return _dedupe_review_rows(rows)


def _latest_selected_review_row(review: pl.DataFrame) -> dict[str, Any]:
    rows = _selected_review_rows(review)
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            _parse_datetime(row.get("actual_exit_ts") or row.get("entry_ts"))
            or datetime.min.replace(tzinfo=UTC),
            _review_completeness_score(row),
        ),
    )


def _duplicate_group_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            _ts_key(row.get("entry_ts")),
            _float_key(row.get("entry_px")),
            _ts_key(row.get("actual_exit_ts")),
            _float_key(row.get("actual_exit_px")),
        ]
    )


def _review_completeness_score(row: dict[str, Any]) -> tuple[int, int, int, int, float]:
    max_unrealized = _float_or_none(row.get("max_unrealized_bps"))
    return (
        1 if _observable(row.get("run_id")) else 0,
        1 if _observable(row.get("source_entry_id")) else 0,
        1 if _float_or_none(row.get("highest_px_after_entry")) is not None else 0,
        1 if _reasonable_bps(max_unrealized) else 0,
        max_unrealized if _reasonable_bps(max_unrealized) else float("-inf"),
    )


def _consistency_for_duplicate_group(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    v5_row = _best_source_row(candidates, "v5_profit_lock_shadow")
    quant_row = _best_source_row(candidates, "quant_lab_recomputed")
    result: dict[str, Any] = {
        "v5_vs_quant_lab_consistency_status": "NOT_APPLICABLE",
        "v5_vs_quant_lab_mismatch_reason": "",
    }
    mismatch_reasons: list[str] = []
    mismatches = 0
    for hours in FIXED_HOLD_FROM_ENTRY_HOURS:
        field = f"fixed_hold_{hours}h_from_entry_net_bps"
        v5_value = _float_or_none(v5_row.get(field) if v5_row else None)
        quant_value = _float_or_none(quant_row.get(field) if quant_row else None)
        diff = (
            quant_value - v5_value
            if quant_value is not None and v5_value is not None
            else None
        )
        result[f"v5_fixed_hold_{hours}h_from_entry_net_bps"] = v5_value
        result[f"quant_lab_fixed_hold_{hours}h_from_entry_net_bps"] = quant_value
        result[f"diff_fixed_hold_{hours}h_from_entry_net_bps"] = diff
        if diff is None:
            continue
        if abs(diff) > CONSISTENCY_TOLERANCE_BPS:
            mismatches += 1
            mismatch_reasons.append(f"{field}_mismatch")
    for hours in DELAYED_EXIT_HOURS:
        field = f"delayed_exit_{hours}h_from_actual_exit_net_bps"
        v5_value = _float_or_none(v5_row.get(field) if v5_row else None)
        quant_value = _float_or_none(quant_row.get(field) if quant_row else None)
        diff = (
            quant_value - v5_value
            if quant_value is not None and v5_value is not None
            else None
        )
        result[f"v5_delayed_exit_{hours}h_from_actual_exit_net_bps"] = v5_value
        result[f"quant_lab_delayed_exit_{hours}h_from_actual_exit_net_bps"] = quant_value
        result[f"diff_delayed_exit_{hours}h_from_actual_exit_net_bps"] = diff
        if diff is None:
            continue
        if abs(diff) > CONSISTENCY_TOLERANCE_BPS:
            mismatches += 1
            mismatch_reasons.append(f"{field}_mismatch")
    if v5_row and quant_row:
        status = "MISMATCH" if mismatches else "MATCH"
    elif v5_row:
        status = "V5_ONLY"
    elif quant_row:
        status = "QUANT_LAB_ONLY"
    else:
        status = "NOT_APPLICABLE"
    result["v5_vs_quant_lab_consistency_status"] = status
    result["v5_vs_quant_lab_mismatch_reason"] = ";".join(mismatch_reasons)
    return result


def _best_source_row(candidates: list[dict[str, Any]], source: str) -> dict[str, Any] | None:
    rows = [row for row in candidates if str(row.get("review_row_source") or "") == source]
    if not rows:
        return None
    return max(enumerate(rows), key=lambda item: (_review_completeness_score(item[1]), item[0]))[1]


def _reasonable_bps(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and -1_000.0 <= value <= 10_000.0


def _ts_key(value: Any) -> str:
    parsed = _parse_datetime(value)
    return parsed.isoformat() if parsed is not None else ""


def _float_key(value: Any) -> str:
    number = _float_or_none(value)
    return f"{number:.8f}" if number is not None else ""


def _bnb_trade_rows(trade_events: pl.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in trade_events.to_dicts():
        symbol = normalize_symbol(row.get("normalized_symbol") or row.get("symbol"))
        if symbol != "BNB-USDT":
            continue
        ts = _parse_datetime(row.get("ts_utc") or row.get("ts"))
        price = _float_or_none(row.get("price") or row.get("fill_px"))
        if ts is None or price is None:
            continue
        rows.append(row | {"_ts": ts, "_price": price})
    rows.sort(key=lambda item: item["_ts"])
    return rows


def _bnb_profit_lock_shadow_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        symbol = normalize_symbol(row.get("normalized_symbol") or row.get("symbol"))
        if symbol != "BNB-USDT":
            continue
        entry_ts = _parse_datetime(_field(row, "entry_ts", "ts_utc", "ts"))
        entry_px = _float_or_none(_field(row, "entry_px", "entry_price", "fill_px", "price"))
        if entry_ts is None or entry_px is None:
            continue
        rows.append(row | {"_ts": entry_ts, "_price": entry_px})
    rows.sort(key=lambda item: item["_ts"])
    return rows


def _review_row_from_profit_lock_shadow(
    row: dict[str, Any],
    *,
    ctx: _Context,
) -> dict[str, Any] | None:
    entry_ts = _parse_datetime(_field(row, "entry_ts", "ts_utc", "ts"))
    entry_px = _float_or_none(_field(row, "entry_px", "entry_price", "fill_px", "price"))
    if entry_ts is None or entry_px is None or entry_px <= 0:
        return None
    actual_exit_ts = _parse_datetime(_field(row, "actual_exit_ts", "exit_ts"))
    actual_exit_px = _float_or_none(_field(row, "actual_exit_px", "exit_px"))
    actual_net = _float_or_none(_field(row, "actual_exit_net_bps", "actual_net_bps"))
    delayed = {
        hours: _float_or_none(
            _field(
                row,
                f"delayed_exit_{hours}h_from_actual_exit_net_bps",
                f"delayed_exit_{hours}h",
                f"delayed_exit_{hours}h_net_bps",
                f"delayed_exit_{hours}h_bps",
            )
        )
        for hours in DELAYED_EXIT_HOURS
    }
    fixed_hold_from_entry = {
        hours: _float_or_none(
            _field(
                row,
                f"fixed_hold_{hours}h_from_entry_net_bps",
                f"fixed_hold_{hours}h_net_bps",
            )
        )
        for hours in FIXED_HOLD_FROM_ENTRY_HOURS
    }
    profit_lock = {
        bps: _float_or_none(
            _field(row, f"profit_lock_{bps}bps_exit", f"profit_lock_{bps}bps_net_bps")
        )
        for bps in PROFIT_LOCK_BPS
    }
    trailing_atr = _float_or_none(_field(row, "trailing_atr_exit", "atr_trailing_exit"))
    alternatives = {
        "actual_exit": actual_net,
        "profit_lock_30bps": profit_lock[30],
        "profit_lock_50bps": profit_lock[50],
        **{
            f"fixed_hold_{hours}h_from_entry": value
            for hours, value in fixed_hold_from_entry.items()
        },
        **{
            f"delayed_exit_{hours}h_from_actual_exit": value
            for hours, value in delayed.items()
        },
        "trailing_atr": trailing_atr,
    }
    best_policy = _normalize_exit_policy_name(
        str(_field(row, "best_shadow_exit_policy", "best_exit_policy") or "").strip()
    )
    best_value = _float_or_none(_field(row, "best_exit_net_bps", "best_shadow_exit_net_bps"))
    if not best_policy or best_value is None:
        best_policy, best_value = _best_policy(alternatives)
    delta = (
        best_value - actual_net if best_value is not None and actual_net is not None else None
    )
    exit_reason = str(_field(row, "exit_reason", "actual_exit_reason") or "")
    max_unrealized = _float_or_none(_field(row, "max_unrealized_bps"))
    diagnosis = _diagnosis(
        actual_exit_net_bps=actual_net,
        max_unrealized_bps=max_unrealized,
        best_exit_policy=best_policy,
        delta_vs_actual_bps=delta,
        exit_reason=exit_reason,
    )
    return _common(ctx) | {
        "as_of_date": ctx.as_of_date.isoformat(),
        "strategy_candidate": "v5.bnb_swing_exit_policy_review",
        "symbol": "BNB-USDT",
        "run_id": str(_field(row, "run_id") or ""),
        "source_entry_id": str(_field(row, "source_entry_id", "trade_id", "order_id") or ""),
        "entry_ts": entry_ts,
        "entry_px": entry_px,
        "highest_px_after_entry": _float_or_none(_field(row, "highest_px_after_entry")),
        "max_unrealized_bps": max_unrealized,
        "actual_exit_ts": actual_exit_ts,
        "actual_exit_px": actual_exit_px,
        "actual_exit_net_bps": actual_net,
        "fixed_hold_6h_from_entry_net_bps": fixed_hold_from_entry[6],
        "fixed_hold_12h_from_entry_net_bps": fixed_hold_from_entry[12],
        "fixed_hold_24h_from_entry_net_bps": fixed_hold_from_entry[24],
        "fixed_hold_4h_net_bps": _float_or_none(_field(row, "fixed_hold_4h_net_bps")),
        "fixed_hold_8h_net_bps": _float_or_none(_field(row, "fixed_hold_8h_net_bps")),
        "fixed_hold_12h_net_bps": fixed_hold_from_entry[12],
        "fixed_hold_24h_net_bps": fixed_hold_from_entry[24],
        "profit_lock_30bps_exit": profit_lock[30],
        "profit_lock_50bps_exit": profit_lock[50],
        "delayed_exit_6h_from_actual_exit_net_bps": delayed[6],
        "delayed_exit_12h_from_actual_exit_net_bps": delayed[12],
        "delayed_exit_24h_from_actual_exit_net_bps": delayed[24],
        "delayed_exit_6h_net_bps": delayed[6],
        "delayed_exit_12h_net_bps": delayed[12],
        "delayed_exit_24h_net_bps": delayed[24],
        "trailing_atr_exit": trailing_atr,
        "best_exit_policy": best_policy,
        "best_shadow_exit_policy": best_policy,
        "best_exit_net_bps": best_value,
        "delta_vs_actual_bps": delta,
        "exit_reason": exit_reason,
        "selected_roundtrip_cost_bps": _float_or_none(
            _field(row, "selected_roundtrip_cost_bps", "cost_bps")
        )
        or DEFAULT_ROUNDTRIP_COST_BPS,
        "diagnosis": diagnosis,
        "status": "REVIEW",
        "review_row_source": "v5_profit_lock_shadow",
        "created_at": ctx.generated_at,
        "source": SOURCE_NAME,
    }


def _is_bnb_swing_entry(row: dict[str, Any]) -> bool:
    action = str(_field(row, "action", "intent", "event_type") or "").lower()
    side = str(_field(row, "side", "order_side") or "").lower()
    if any(token in action for token in ["exit", "close", "reduce"]):
        return False
    if not (side in {"buy", "long"} or any(token in action for token in ["entry", "open"])):
        return False
    haystack = " ".join(
        str(_field(row, field) or "")
        for field in [
            "strategy_id",
            "strategy_candidate",
            "proposal_id",
            "entry_reason",
            "source_strategy_candidate",
            "raw_payload_json",
        ]
    ).lower()
    return not haystack or "bnb" in haystack or "swing" in haystack or "f3" in haystack


def _is_bnb_exit(row: dict[str, Any]) -> bool:
    action = str(_field(row, "action", "intent", "event_type") or "").lower()
    side = str(_field(row, "side", "order_side") or "").lower()
    return side in {"sell", "short"} or any(
        token in action for token in ["exit", "close", "reduce"]
    )


def _matching_exit(entry: dict[str, Any], exits: list[dict[str, Any]]) -> dict[str, Any] | None:
    entry_ts = entry.get("_ts")
    run_id = str(entry.get("run_id") or "")
    candidates = [
        row
        for row in exits
        if row.get("_ts") is not None
        and row["_ts"] > entry_ts
        and (not run_id or not row.get("run_id") or str(row.get("run_id")) == run_id)
    ]
    if not candidates and entry_ts is not None:
        candidates = [row for row in exits if row.get("_ts") is not None and row["_ts"] > entry_ts]
    return candidates[0] if candidates else None


def _actual_exit_net_bps(
    *,
    entry_px: float,
    exit_px: float | None,
    cost_bps: float,
    entry: dict[str, Any],
    exit_row: dict[str, Any] | None,
) -> float | None:
    explicit = _first_float(
        [entry, exit_row or {}],
        [
            "actual_exit_net_bps",
            "realized_net_bps",
            "net_bps",
            "pnl_bps",
            "paper_pnl_bps",
        ],
    )
    if explicit is not None:
        return explicit
    if exit_px is None or entry_px <= 0:
        return None
    return (exit_px / entry_px - 1.0) * 10_000.0 - cost_bps


def _roundtrip_cost_bps(
    entry: dict[str, Any],
    exit_row: dict[str, Any] | None,
    entry_px: float,
) -> float:
    explicit = _first_float(
        [entry, exit_row or {}],
        ["selected_roundtrip_cost_bps", "roundtrip_all_in_cost_bps", "roundtrip_cost_bps"],
    )
    if explicit is not None:
        return max(explicit, 0.0)
    entry_notional = _float_or_none(entry.get("notional_usdt"))
    if entry_notional is None:
        qty = _float_or_none(entry.get("qty") or entry.get("fill_size"))
        entry_notional = abs(qty * entry_px) if qty is not None else None
    fees = [
        _float_or_none(entry.get("fee_usdt")),
        _float_or_none((exit_row or {}).get("fee_usdt")),
    ]
    fee_sum = sum(abs(value) for value in fees if value is not None)
    if entry_notional is not None and entry_notional > 0 and fee_sum > 0:
        return fee_sum / entry_notional * 10_000.0
    return DEFAULT_ROUNDTRIP_COST_BPS


def _highest_after_entry(
    rows: list[dict[str, Any]],
    *,
    entry_ts: datetime,
    end_ts: datetime,
    fallback: float,
) -> float | None:
    window = [row for row in rows if entry_ts < row["ts"] <= end_ts]
    values = [_float_or_none(row.get("high")) for row in window]
    observed = [value for value in values if value is not None]
    if fallback is not None:
        return max([fallback, *observed])
    return max(observed) if observed else None


def _fixed_hold_net_bps(
    rows: list[dict[str, Any]],
    *,
    entry_ts: datetime,
    entry_px: float,
    horizon_hours: int,
    cost_bps: float,
) -> float | None:
    close = _market_close_at_or_after(rows, entry_ts + timedelta(hours=horizon_hours))
    if close is None or entry_px <= 0:
        return None
    return (close / entry_px - 1.0) * 10_000.0 - cost_bps


def _profit_lock_exit_net_bps(
    rows: list[dict[str, Any]],
    *,
    entry_ts: datetime,
    end_ts: datetime,
    entry_px: float,
    threshold_bps: int,
    cost_bps: float,
) -> float | None:
    threshold_px = entry_px * (1.0 + threshold_bps / 10_000.0)
    for row in rows:
        high = _float_or_none(row.get("high")) or 0.0
        if entry_ts < row["ts"] <= end_ts and high >= threshold_px:
            return float(threshold_bps) - cost_bps
    return None


def _delayed_exit_net_bps(
    rows: list[dict[str, Any]],
    *,
    actual_exit_ts: datetime | None,
    entry_px: float,
    delay_hours: int,
    cost_bps: float,
) -> float | None:
    if actual_exit_ts is None or entry_px <= 0:
        return None
    close = _market_close_at_or_after(rows, actual_exit_ts + timedelta(hours=delay_hours))
    if close is None:
        return None
    return (close / entry_px - 1.0) * 10_000.0 - cost_bps


def _trailing_atr_exit_net_bps(
    rows: list[dict[str, Any]],
    *,
    entry_ts: datetime,
    end_ts: datetime,
    entry_px: float,
    cost_bps: float,
) -> float | None:
    if entry_px <= 0:
        return None
    ordered = sorted(rows, key=lambda row: row["ts"])
    high_water = entry_px
    for index, row in enumerate(ordered):
        ts = row["ts"]
        if ts <= entry_ts:
            continue
        if ts > end_ts:
            break
        high = _float_or_none(row.get("high")) or _float_or_none(row.get("close"))
        low = _float_or_none(row.get("low")) or _float_or_none(row.get("close"))
        if high is None or low is None:
            continue
        high_water = max(high_water, high)
        atr = _average_range(ordered[max(0, index - ATR_LOOKBACK_BARS + 1) : index + 1])
        if atr is None or atr <= 0:
            continue
        stop_px = high_water - ATR_MULTIPLIER * atr
        if high_water > entry_px and low <= stop_px:
            return (stop_px / entry_px - 1.0) * 10_000.0 - cost_bps
    return None


def _best_policy(alternatives: dict[str, float | None]) -> tuple[str, float | None]:
    observed = {key: value for key, value in alternatives.items() if value is not None}
    if not observed:
        return ("", None)
    return max(observed.items(), key=lambda item: item[1])


def _normalize_exit_policy_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    if text.startswith("delayed_exit_") and text.endswith("h"):
        return f"{text}_from_actual_exit"
    if text.startswith("fixed_hold_") and text.endswith("h"):
        return f"{text}_from_entry"
    if text.startswith("profit_lock_") and text.endswith("bps_exit"):
        return text.removesuffix("_exit")
    return text


def _diagnosis(
    *,
    actual_exit_net_bps: float | None,
    max_unrealized_bps: float | None,
    best_exit_policy: str,
    delta_vs_actual_bps: float | None,
    exit_reason: str,
) -> str:
    if actual_exit_net_bps is None:
        return "actual_exit_not_observable"
    if max_unrealized_bps is not None and max_unrealized_bps > 50 and actual_exit_net_bps < 0:
        if best_exit_policy.startswith("profit_lock"):
            return "profit_lock_too_late"
        if best_exit_policy == "trailing_atr":
            return "trailing_variant_may_improve"
        return "gave_back_unrealized_profit"
    if delta_vs_actual_bps is not None and delta_vs_actual_bps > 0:
        return "alternative_exit_would_improve"
    if "atr" in exit_reason.lower() or "trailing" in exit_reason.lower():
        return "actual_trailing_not_worse_in_sample"
    return "actual_exit_not_worse_in_sample"


def _market_bar_index(market_bars: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if market_bars.is_empty() or "close" not in market_bars.columns:
        return {}
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in market_bars.to_dicts():
        symbol = normalize_symbol(row.get("symbol") or row.get("inst_id"))
        ts = _parse_datetime(row.get("ts"))
        close = _float_or_none(row.get("close"))
        if not symbol or ts is None or close is None:
            continue
        high = _float_or_none(row.get("high"))
        low = _float_or_none(row.get("low"))
        rows_by_symbol.setdefault(symbol, []).append(
            {
                "ts": ts,
                "close": close,
                "high": high if high is not None else close,
                "low": low if low is not None else close,
            }
        )
    for rows in rows_by_symbol.values():
        rows.sort(key=lambda item: item["ts"])
    return rows_by_symbol


def _market_close_at_or_after(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    return _market_close_near_v5_target(rows, ts)


def _market_close_near_v5_target(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    candidates: list[tuple[float, int, datetime, float]] = []
    lower = ts - V5_PRICE_BEFORE_TOLERANCE
    upper = ts + V5_PRICE_AFTER_TOLERANCE
    for row in rows:
        row_ts = row["ts"]
        if row_ts < lower or row_ts > upper:
            continue
        close = _float_or_none(row.get("close"))
        if close is None:
            continue
        candidates.append(
            (
                abs((row_ts - ts).total_seconds()),
                0 if row_ts >= ts else 1,
                row_ts,
                close,
            )
        )
    if not candidates:
        return None
    return min(candidates)[3]


def _average_range(rows: list[dict[str, Any]]) -> float | None:
    ranges = []
    for row in rows:
        high = _float_or_none(row.get("high"))
        low = _float_or_none(row.get("low"))
        if high is not None and low is not None:
            ranges.append(max(high - low, 0.0))
    return sum(ranges) / len(ranges) if ranges else None


def _field(row: dict[str, Any], *names: str) -> Any:
    payload = _payload(row)
    for name in names:
        value = row.get(name)
        if _observable(value):
            return value
        value = payload.get(name)
        if _observable(value):
            return value
    return None


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("raw_payload_json")
    if not value:
        return {}
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _first_float(rows: list[dict[str, Any]], fields: list[str]) -> float | None:
    for row in rows:
        for field in fields:
            value = _float_or_none(_field(row, field))
            if value is not None:
                return value
    return None


def _observable(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "none", "null", "nan", "not_observable"}


def _float_or_none(value: Any) -> float | None:
    if not _observable(value):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not _observable(value):
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _mean(values: Any) -> float | None:
    observed = [_float_or_none(value) for value in values]
    numbers = [value for value in observed if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _source_entry_id(entry: dict[str, Any], entry_ts: datetime) -> str:
    return str(
        _field(entry, "trade_id", "order_id", "source_event_key", "event_id")
        or f"{entry.get('run_id') or ''}:{entry_ts.isoformat()}"
    )


def _common(ctx: _Context, *, schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
    return {
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
        "schema_version": schema_version,
        "quant_lab_git_commit": ctx.git_commit or "not_observable",
        "source_version": _source_version("bnb_swing_exit_policy", ctx.git_commit),
        "generated_at_utc": ctx.generated_at,
        "generated_from_bundle_id": ctx.generated_from_bundle_id,
    }


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    return "None" if number is None else f"{number:.4f}"


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value and value != "auto":
        return date.fromisoformat(str(value))
    return datetime.now(UTC).date()


def _latest_bundle_id(root: Path) -> str:
    frame = read_parquet_dataset(root / V5_TRADE_EVENT_DATASET)
    candidates = []
    for row in frame.to_dicts() if not frame.is_empty() else []:
        bundle_ts = _parse_datetime(row.get("bundle_ts") or row.get("ingest_ts"))
        bundle_name = str(row.get("bundle_name") or row.get("source_bundle") or "")
        if bundle_ts is not None:
            candidates.append((bundle_ts, bundle_name or bundle_ts.isoformat()))
    return max(candidates, key=lambda item: item[0])[1] if candidates else "not_observable"


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            cwd=root,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def _source_version(component: str, git_commit: str | None) -> str:
    return f"{component}:{git_commit or __version__}"
