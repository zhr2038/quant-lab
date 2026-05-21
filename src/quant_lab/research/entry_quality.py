from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

SOURCE_NAME = "quant_lab"
ENTRY_QUALITY_SCHEMA_VERSION = "entry_quality.v0.1"
DEFAULT_WINDOW_HOURS = 24
ENTRY_QUALITY_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"}
LATE_CHASE_THRESHOLDS_BPS = (100, 150, 200, 250, 300, 400)
PULLBACK_HORIZON_HOURS = (4, 8, 12, 24, 48, 72)
MIN_ROUNDTRIP_COST_BPS = 30.0

V5_TRADE_EVENT_DATASET = Path("silver") / "v5_trade_event"
V5_ORDER_LIFECYCLE_DATASET = Path("silver") / "v5_order_lifecycle"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
V5_CANDIDATE_EVENT_DATASET = Path("silver") / "v5_candidate_event"
V5_CANDIDATE_LABEL_DATASET = Path("gold") / "v5_candidate_label"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"

MISSED_LOW_AUDIT_DATASET = Path("gold") / "v5_missed_low_audit"
MISSED_LOW_BY_SYMBOL_DATASET = Path("gold") / "v5_missed_low_by_symbol"
MISSED_LOW_BY_ENTRY_REASON_DATASET = Path("gold") / "v5_missed_low_by_entry_reason"
LATE_ENTRY_CHASE_SHADOW_DATASET = Path("gold") / "v5_late_entry_chase_shadow"
LATE_ENTRY_CHASE_THRESHOLD_ADVISORY_DATASET = (
    Path("gold") / "v5_late_entry_chase_threshold_advisory"
)
PULLBACK_REVERSAL_SHADOW_DATASET = Path("gold") / "v5_pullback_reversal_shadow"
PULLBACK_REVERSAL_READINESS_DATASET = Path("gold") / "v5_pullback_reversal_readiness"
ENTRY_QUALITY_ADVISORY_DATASET = Path("gold") / "v5_entry_quality_advisory"


COMMON_SCHEMA = {
    "contract_version": pl.Utf8,
    "schema_version": pl.Utf8,
    "generated_at_utc": pl.Datetime(time_zone="UTC"),
    "generated_from_bundle_id": pl.Utf8,
    "as_of_date": pl.Utf8,
    "window_hours": pl.Int64,
    "source": pl.Utf8,
    "mode": pl.Utf8,
}

MISSED_LOW_AUDIT_SCHEMA = COMMON_SCHEMA | {
    "run_id": pl.Utf8,
    "source_event_key": pl.Utf8,
    "symbol": pl.Utf8,
    "entry_ts": pl.Datetime(time_zone="UTC"),
    "entry_px": pl.Float64,
    "entry_reason": pl.Utf8,
    "probe_type": pl.Utf8,
    "side": pl.Utf8,
    "intent": pl.Utf8,
    "entry_vs_pre_4h_low_bps": pl.Float64,
    "entry_vs_pre_8h_low_bps": pl.Float64,
    "entry_vs_pre_12h_low_bps": pl.Float64,
    "entry_vs_pre_24h_low_bps": pl.Float64,
    "entry_position_in_24h_range": pl.Float64,
    "realized_net_bps": pl.Float64,
    "exit_reason": pl.Utf8,
    "diagnosis": pl.Utf8,
}

MISSED_LOW_AGG_SCHEMA = COMMON_SCHEMA | {
    "group_key": pl.Utf8,
    "sample_count": pl.Int64,
    "loss_count": pl.Int64,
    "profit_count": pl.Int64,
    "late_chase_loss_count": pl.Int64,
    "late_but_trend_profitable_count": pl.Int64,
    "avg_entry_vs_pre_24h_low_bps": pl.Float64,
    "avg_entry_position_in_24h_range": pl.Float64,
    "avg_realized_net_bps": pl.Float64,
    "diagnosis_mix": pl.Utf8,
}

LATE_ENTRY_CHASE_SHADOW_SCHEMA = COMMON_SCHEMA | {
    "strategy_candidate": pl.Utf8,
    "source_type": pl.Utf8,
    "run_id": pl.Utf8,
    "candidate_id": pl.Utf8,
    "source_event_key": pl.Utf8,
    "symbol": pl.Utf8,
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "entry_or_candidate_px": pl.Float64,
    "recent_12h_low": pl.Float64,
    "recent_24h_low": pl.Float64,
    "entry_vs_12h_low_bps": pl.Float64,
    "entry_vs_24h_low_bps": pl.Float64,
    "entry_position_in_12h_range": pl.Float64,
    "f4_volume_expansion": pl.Float64,
    "f5_rsi_trend_confirm": pl.Float64,
    "late_chase_risk": pl.Boolean,
    "would_block_if_enabled": pl.Boolean,
    "realized_net_bps": pl.Float64,
    "forward_24h_net_bps": pl.Float64,
    "forward_48h_net_bps": pl.Float64,
    "outcome_class": pl.Utf8,
}

LATE_ENTRY_THRESHOLD_SCHEMA = COMMON_SCHEMA | {
    "threshold_bps": pl.Int64,
    "would_block_count": pl.Int64,
    "would_block_loss_count": pl.Int64,
    "would_block_profit_count": pl.Int64,
    "false_positive_rate": pl.Float64,
    "ready_for_live_guard": pl.Boolean,
    "advisory": pl.Utf8,
}

PULLBACK_REVERSAL_SHADOW_SCHEMA = COMMON_SCHEMA | {
    "strategy_candidate": pl.Utf8,
    "run_id": pl.Utf8,
    "candidate_id": pl.Utf8,
    "source_event_key": pl.Utf8,
    "symbol": pl.Utf8,
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "regime_state": pl.Utf8,
    "risk_level": pl.Utf8,
    "current_px": pl.Float64,
    "pre_24h_low": pl.Float64,
    "pre_24h_high": pl.Float64,
    "pullback_from_24h_high_bps": pl.Float64,
    "recent_2h_no_new_low": pl.Boolean,
    "f4_volume_expansion": pl.Float64,
    "f5_rsi_trend_confirm": pl.Float64,
    "selected_roundtrip_cost_bps": pl.Float64,
    "horizon_hours": pl.Int64,
    "gross_bps": pl.Float64,
    "net_bps_after_cost": pl.Float64,
    "mfe_bps": pl.Float64,
    "mae_bps": pl.Float64,
    "win": pl.Boolean,
    "label_status": pl.Utf8,
}

PULLBACK_REVERSAL_READINESS_SCHEMA = COMMON_SCHEMA | {
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "sample_count": pl.Int64,
    "recent_7d_sample_count": pl.Int64,
    "avg_24h_net_bps": pl.Float64,
    "win_rate_24h": pl.Float64,
    "p25_24h_net_bps": pl.Float64,
    "avg_mae_bps": pl.Float64,
    "ready_for_paper": pl.Boolean,
    "ready_for_live_probe": pl.Boolean,
    "readiness_reasons": pl.Utf8,
}

ENTRY_QUALITY_ADVISORY_SCHEMA = COMMON_SCHEMA | {
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "readiness_status": pl.Utf8,
    "sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "advisory_reasons": pl.Utf8,
    "ready_for_live": pl.Boolean,
}


class EntryQualityBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    missed_low_audit_rows: int = Field(ge=0)
    missed_low_by_symbol_rows: int = Field(ge=0)
    missed_low_by_entry_reason_rows: int = Field(ge=0)
    late_entry_chase_shadow_rows: int = Field(ge=0)
    late_entry_chase_threshold_rows: int = Field(ge=0)
    pullback_reversal_shadow_rows: int = Field(ge=0)
    pullback_reversal_readiness_rows: int = Field(ge=0)
    entry_quality_advisory_rows: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _BuildContext:
    as_of_date: date
    generated_at: datetime
    generated_from_bundle_id: str
    window_hours: int


def build_and_publish_entry_quality(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> EntryQualityBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    ctx = _BuildContext(
        as_of_date=day,
        generated_at=datetime.now(UTC),
        generated_from_bundle_id=_latest_bundle_id(root),
        window_hours=max(int(window_hours), 1),
    )
    trades = read_parquet_dataset(root / V5_TRADE_EVENT_DATASET)
    lifecycles = read_parquet_dataset(root / V5_ORDER_LIFECYCLE_DATASET)
    market = _normalize_market_bars(read_parquet_dataset(root / MARKET_BAR_DATASET))
    candidates = read_parquet_dataset(root / V5_CANDIDATE_EVENT_DATASET)
    labels = read_parquet_dataset(root / V5_CANDIDATE_LABEL_DATASET)
    costs = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET)

    warnings: list[str] = []
    if market.is_empty():
        warnings.append("market_bar_empty")
    if trades.is_empty() and lifecycles.is_empty():
        warnings.append("v5_actual_entry_events_empty")
    if candidates.is_empty():
        warnings.append("v5_candidate_event_empty")

    actual_entries = _actual_entry_rows(trades, lifecycles)
    missed = build_missed_low_audit(actual_entries, market, ctx)
    missed_by_symbol = aggregate_missed_low(missed, group_column="symbol", ctx=ctx)
    missed_by_reason = aggregate_missed_low(missed, group_column="entry_reason", ctx=ctx)
    late_shadow = build_late_entry_chase_shadow(
        actual_entries=actual_entries,
        candidates=candidates,
        labels=labels,
        market_bars=market,
        ctx=ctx,
    )
    late_threshold = build_late_entry_chase_threshold_advisory(late_shadow, ctx=ctx)
    pullback = build_pullback_reversal_shadow(
        candidates=candidates,
        market_bars=market,
        costs=costs,
        ctx=ctx,
    )
    readiness = build_pullback_reversal_readiness(pullback, ctx=ctx)
    advisory = build_entry_quality_advisory(
        missed_low=missed,
        late_threshold=late_threshold,
        pullback_readiness=readiness,
        ctx=ctx,
    )

    return EntryQualityBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        missed_low_audit_rows=_publish_daily(
            root,
            MISSED_LOW_AUDIT_DATASET,
            missed,
            ["as_of_date", "source_event_key", "symbol", "entry_ts"],
        ),
        missed_low_by_symbol_rows=_publish_daily(
            root,
            MISSED_LOW_BY_SYMBOL_DATASET,
            missed_by_symbol,
            ["as_of_date", "group_key"],
        ),
        missed_low_by_entry_reason_rows=_publish_daily(
            root,
            MISSED_LOW_BY_ENTRY_REASON_DATASET,
            missed_by_reason,
            ["as_of_date", "group_key"],
        ),
        late_entry_chase_shadow_rows=_publish_daily(
            root,
            LATE_ENTRY_CHASE_SHADOW_DATASET,
            late_shadow,
            ["as_of_date", "source_type", "source_event_key", "symbol"],
        ),
        late_entry_chase_threshold_rows=_publish_daily(
            root,
            LATE_ENTRY_CHASE_THRESHOLD_ADVISORY_DATASET,
            late_threshold,
            ["as_of_date", "threshold_bps"],
        ),
        pullback_reversal_shadow_rows=_publish_daily(
            root,
            PULLBACK_REVERSAL_SHADOW_DATASET,
            pullback,
            ["as_of_date", "source_event_key", "symbol", "horizon_hours"],
        ),
        pullback_reversal_readiness_rows=_publish_daily(
            root,
            PULLBACK_REVERSAL_READINESS_DATASET,
            readiness,
            ["as_of_date", "strategy_candidate", "symbol"],
        ),
        entry_quality_advisory_rows=_publish_daily(
            root,
            ENTRY_QUALITY_ADVISORY_DATASET,
            advisory,
            ["as_of_date", "strategy_candidate", "symbol"],
        ),
        warnings=warnings,
    )


def build_missed_low_audit(
    actual_entries: list[dict[str, Any]],
    market_bars: pl.DataFrame,
    ctx: _BuildContext,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    market_by_symbol = _market_rows_by_symbol(market_bars)
    for entry in actual_entries:
        symbol = normalize_symbol(entry.get("symbol")) or "UNKNOWN"
        entry_ts = _coerce_datetime(entry.get("entry_ts"))
        entry_px = _float_or_none(entry.get("entry_px"))
        if symbol == "UNKNOWN" or entry_ts is None or entry_px is None or entry_px <= 0:
            continue
        windows = {
            hours: _pre_window_stats(market_by_symbol.get(symbol, []), entry_ts, hours)
            for hours in (4, 8, 12, 24)
        }
        realized_net_bps = _float_or_none(entry.get("realized_net_bps"))
        diagnosis = _missed_low_diagnosis(
            entry_vs_24h_low_bps=_entry_vs_low_bps(entry_px, windows[24].get("low")),
            position_24h=_position_in_range(
                entry_px, windows[24].get("low"), windows[24].get("high")
            ),
            realized_net_bps=realized_net_bps,
        )
        rows.append(
            _common(ctx, mode="audit")
            | {
                "run_id": str(entry.get("run_id") or ""),
                "source_event_key": _source_event_key(entry, symbol=symbol, ts=entry_ts),
                "symbol": symbol,
                "entry_ts": entry_ts,
                "entry_px": entry_px,
                "entry_reason": str(entry.get("entry_reason") or ""),
                "probe_type": str(entry.get("probe_type") or ""),
                "side": str(entry.get("side") or ""),
                "intent": str(entry.get("intent") or ""),
                "entry_vs_pre_4h_low_bps": _entry_vs_low_bps(entry_px, windows[4].get("low")),
                "entry_vs_pre_8h_low_bps": _entry_vs_low_bps(entry_px, windows[8].get("low")),
                "entry_vs_pre_12h_low_bps": _entry_vs_low_bps(entry_px, windows[12].get("low")),
                "entry_vs_pre_24h_low_bps": _entry_vs_low_bps(entry_px, windows[24].get("low")),
                "entry_position_in_24h_range": _position_in_range(
                    entry_px, windows[24].get("low"), windows[24].get("high")
                ),
                "realized_net_bps": realized_net_bps,
                "exit_reason": str(entry.get("exit_reason") or ""),
                "diagnosis": diagnosis,
            }
        )
    if not rows:
        return pl.DataFrame(schema=MISSED_LOW_AUDIT_SCHEMA)
    return pl.DataFrame(rows, schema=MISSED_LOW_AUDIT_SCHEMA, orient="row")


def aggregate_missed_low(
    missed_low: pl.DataFrame,
    *,
    group_column: str,
    ctx: _BuildContext,
) -> pl.DataFrame:
    if missed_low.is_empty() or group_column not in missed_low.columns:
        return pl.DataFrame(schema=MISSED_LOW_AGG_SCHEMA)
    rows: list[dict[str, Any]] = []
    for group_key, group_rows in _group_dicts(missed_low.to_dicts(), group_column).items():
        diagnoses = Counter(str(row.get("diagnosis") or "") for row in group_rows)
        realized = [_float_or_none(row.get("realized_net_bps")) for row in group_rows]
        rows.append(
            _common(ctx, mode="audit")
            | {
                "group_key": str(group_key or "UNKNOWN"),
                "sample_count": len(group_rows),
                "loss_count": sum(
                    1 for value in realized if value is not None and value < 0.0
                ),
                "profit_count": sum(
                    1 for value in realized if value is not None and value >= 0.0
                ),
                "late_chase_loss_count": diagnoses.get("late_chase_loss", 0),
                "late_but_trend_profitable_count": diagnoses.get(
                    "late_but_trend_profitable", 0
                ),
                "avg_entry_vs_pre_24h_low_bps": _mean(
                    row.get("entry_vs_pre_24h_low_bps") for row in group_rows
                ),
                "avg_entry_position_in_24h_range": _mean(
                    row.get("entry_position_in_24h_range") for row in group_rows
                ),
                "avg_realized_net_bps": _mean(realized),
                "diagnosis_mix": safe_json_dumps(dict(diagnoses)),
            }
        )
    return pl.DataFrame(rows, schema=MISSED_LOW_AGG_SCHEMA, orient="row")


def build_late_entry_chase_shadow(
    *,
    actual_entries: list[dict[str, Any]],
    candidates: pl.DataFrame,
    labels: pl.DataFrame,
    market_bars: pl.DataFrame,
    ctx: _BuildContext,
) -> pl.DataFrame:
    market_by_symbol = _market_rows_by_symbol(market_bars)
    label_context = _label_context(labels)
    rows: list[dict[str, Any]] = []
    for entry in actual_entries:
        subject = _late_subject_from_actual(entry)
        if subject:
            rows.append(
                _late_shadow_row(subject, market_by_symbol, label_context, ctx=ctx)
            )
    for candidate in candidates.to_dicts() if not candidates.is_empty() else []:
        subject = _late_subject_from_candidate(candidate)
        if subject:
            rows.append(
                _late_shadow_row(subject, market_by_symbol, label_context, ctx=ctx)
            )
    rows = [row for row in rows if row]
    if not rows:
        return pl.DataFrame(schema=LATE_ENTRY_CHASE_SHADOW_SCHEMA)
    return pl.DataFrame(rows, schema=LATE_ENTRY_CHASE_SHADOW_SCHEMA, orient="row")


def build_late_entry_chase_threshold_advisory(
    late_shadow: pl.DataFrame,
    *,
    ctx: _BuildContext,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    shadow_rows = late_shadow.to_dicts() if not late_shadow.is_empty() else []
    for threshold in LATE_CHASE_THRESHOLDS_BPS:
        blocked = [
            row
            for row in shadow_rows
            if (_float_or_none(row.get("entry_vs_12h_low_bps")) or 0.0) > threshold
            and (_float_or_none(row.get("entry_position_in_12h_range")) or 0.0) > 0.70
        ]
        loss_count = sum(
            1
            for row in blocked
            if _outcome_value(row) is not None and (_outcome_value(row) or 0.0) < 0
        )
        profit_count = sum(
            1
            for row in blocked
            if _outcome_value(row) is not None and (_outcome_value(row) or 0.0) >= 0
        )
        total = len(blocked)
        rows.append(
            _common(ctx, mode="advisory")
            | {
                "threshold_bps": int(threshold),
                "would_block_count": total,
                "would_block_loss_count": loss_count,
                "would_block_profit_count": profit_count,
                "false_positive_rate": (profit_count / total) if total else None,
                "ready_for_live_guard": False,
                "advisory": "shadow_only_collect_more_samples",
            }
        )
    return pl.DataFrame(rows, schema=LATE_ENTRY_THRESHOLD_SCHEMA, orient="row")


def build_pullback_reversal_shadow(
    *,
    candidates: pl.DataFrame,
    market_bars: pl.DataFrame,
    costs: pl.DataFrame,
    ctx: _BuildContext,
) -> pl.DataFrame:
    if candidates.is_empty() or market_bars.is_empty():
        return pl.DataFrame(schema=PULLBACK_REVERSAL_SHADOW_SCHEMA)
    market_by_symbol = _market_rows_by_symbol(market_bars)
    cost_by_symbol = _roundtrip_cost_by_symbol(costs)
    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dicts():
        symbol = normalize_symbol(candidate.get("symbol") or candidate.get("normalized_symbol"))
        if symbol not in ENTRY_QUALITY_SYMBOLS:
            continue
        ts = _coerce_datetime(candidate.get("ts_utc") or candidate.get("ts"))
        if ts is None:
            continue
        current_px = _candidate_price(candidate, market_by_symbol.get(symbol, []), ts)
        if current_px is None:
            continue
        bars = market_by_symbol.get(symbol, [])
        pre_24h = _pre_window_stats(bars, ts, 24)
        pullback_bps = _pullback_from_high_bps(current_px, pre_24h.get("high"))
        f4 = _float_or_none(candidate.get("f4_volume_expansion"))
        f5 = _float_or_none(candidate.get("f5_rsi_trend_confirm"))
        if not _pullback_reversal_candidate_ok(
            row=candidate,
            current_px=current_px,
            pre_24h=pre_24h,
            pullback_bps=pullback_bps,
            f4=f4,
            f5=f5,
            bars=bars,
            ts=ts,
        ):
            continue
        roundtrip_cost = max(
            cost_by_symbol.get(symbol, MIN_ROUNDTRIP_COST_BPS),
            MIN_ROUNDTRIP_COST_BPS,
        )
        for horizon in PULLBACK_HORIZON_HOURS:
            label = _forward_label(bars, ts, current_px, horizon, roundtrip_cost)
            rows.append(
                _common(ctx, mode="shadow")
                | {
                    "strategy_candidate": _pullback_candidate_name(symbol),
                    "run_id": str(candidate.get("run_id") or ""),
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "source_event_key": _source_event_key(candidate, symbol=symbol, ts=ts),
                    "symbol": symbol,
                    "ts_utc": ts,
                    "regime_state": str(candidate.get("regime_state") or ""),
                    "risk_level": str(candidate.get("risk_level") or ""),
                    "current_px": current_px,
                    "pre_24h_low": pre_24h.get("low"),
                    "pre_24h_high": pre_24h.get("high"),
                    "pullback_from_24h_high_bps": pullback_bps,
                    "recent_2h_no_new_low": _recent_2h_no_new_low(bars, ts),
                    "f4_volume_expansion": f4,
                    "f5_rsi_trend_confirm": f5,
                    "selected_roundtrip_cost_bps": roundtrip_cost,
                    "horizon_hours": horizon,
                    **label,
                }
            )
    if not rows:
        return pl.DataFrame(schema=PULLBACK_REVERSAL_SHADOW_SCHEMA)
    return pl.DataFrame(rows, schema=PULLBACK_REVERSAL_SHADOW_SCHEMA, orient="row")


def build_pullback_reversal_readiness(
    pullback: pl.DataFrame,
    *,
    ctx: _BuildContext,
) -> pl.DataFrame:
    if pullback.is_empty():
        return pl.DataFrame(schema=PULLBACK_REVERSAL_READINESS_SCHEMA)
    rows: list[dict[str, Any]] = []
    shadow_rows = [
        row for row in pullback.to_dicts() if int(row.get("horizon_hours") or 0) == 24
    ]
    grouped = _group_dicts(shadow_rows, "symbol")
    recent_cutoff = datetime.combine(ctx.as_of_date - timedelta(days=7), time.min, tzinfo=UTC)
    for symbol, group_rows in grouped.items():
        net_values = [_float_or_none(row.get("net_bps_after_cost")) for row in group_rows]
        net_values = [value for value in net_values if value is not None]
        mae_values = [_float_or_none(row.get("mae_bps")) for row in group_rows]
        mae_values = [value for value in mae_values if value is not None]
        sample_count = len(group_rows)
        recent_count = sum(
            1
            for row in group_rows
            if (_coerce_datetime(row.get("ts_utc")) or datetime.min.replace(tzinfo=UTC))
            >= recent_cutoff
        )
        avg_net = _mean(net_values)
        win_rate = (
            sum(1 for value in net_values if value > 0) / len(net_values)
            if net_values
            else None
        )
        p25 = _quantile(net_values, 0.25)
        avg_mae = _mean(mae_values)
        reasons = _pullback_readiness_reasons(
            sample_count=sample_count,
            recent_count=recent_count,
            avg_net=avg_net,
            win_rate=win_rate,
            p25=p25,
            avg_mae=avg_mae,
        )
        ready_for_paper = not reasons
        rows.append(
            _common(ctx, mode="advisory")
            | {
                "strategy_candidate": _pullback_candidate_name(str(symbol)),
                "symbol": str(symbol),
                "sample_count": sample_count,
                "recent_7d_sample_count": recent_count,
                "avg_24h_net_bps": avg_net,
                "win_rate_24h": win_rate,
                "p25_24h_net_bps": p25,
                "avg_mae_bps": avg_mae,
                "ready_for_paper": ready_for_paper,
                "ready_for_live_probe": False,
                "readiness_reasons": safe_json_dumps(
                    reasons if reasons else ["ready_for_paper_shadow_only"]
                ),
            }
        )
    return pl.DataFrame(rows, schema=PULLBACK_REVERSAL_READINESS_SCHEMA, orient="row")


def build_entry_quality_advisory(
    *,
    missed_low: pl.DataFrame,
    late_threshold: pl.DataFrame,
    pullback_readiness: pl.DataFrame,
    ctx: _BuildContext,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    if not missed_low.is_empty():
        rows.append(
            _common(ctx, mode="advisory")
            | {
                "strategy_candidate": "v5.entry_quality_missed_low_audit",
                "symbol": "ALL",
                "recommended_mode": "audit",
                "readiness_status": "AUDIT_READY",
                "sample_count": missed_low.height,
                "avg_net_bps": _mean(missed_low.get_column("realized_net_bps").to_list()),
                "win_rate": None,
                "advisory_reasons": safe_json_dumps(
                    ["read_only_audit", "does_not_block_live_orders"]
                ),
                "ready_for_live": False,
            }
        )
    if not late_threshold.is_empty():
        total_blocked = sum(
            int(row.get("would_block_count") or 0) for row in late_threshold.to_dicts()
        )
        if total_blocked > 0:
            rows.append(
                _common(ctx, mode="advisory")
                | {
                    "strategy_candidate": "v5.late_entry_chase_guard_shadow",
                    "symbol": "ALL",
                    "recommended_mode": "shadow",
                    "readiness_status": "SHADOW_ONLY",
                    "sample_count": total_blocked,
                    "avg_net_bps": None,
                    "win_rate": None,
                    "advisory_reasons": safe_json_dumps(
                        ["ready_for_live_guard=false", "threshold_sensitivity_only"]
                    ),
                    "ready_for_live": False,
                }
            )
    for row in pullback_readiness.to_dicts() if not pullback_readiness.is_empty() else []:
        ready_for_paper = bool(row.get("ready_for_paper"))
        rows.append(
            _common(ctx, mode="advisory")
            | {
                "strategy_candidate": row.get("strategy_candidate"),
                "symbol": row.get("symbol"),
                "recommended_mode": "paper" if ready_for_paper else "shadow",
                "readiness_status": "READY_FOR_PAPER" if ready_for_paper else "SHADOW_ONLY",
                "sample_count": int(row.get("sample_count") or 0),
                "avg_net_bps": _float_or_none(row.get("avg_24h_net_bps")),
                "win_rate": _float_or_none(row.get("win_rate_24h")),
                "advisory_reasons": str(row.get("readiness_reasons") or "[]"),
                "ready_for_live": False,
            }
        )
    if not rows:
        return pl.DataFrame(schema=ENTRY_QUALITY_ADVISORY_SCHEMA)
    return pl.DataFrame(rows, schema=ENTRY_QUALITY_ADVISORY_SCHEMA, orient="row")


def _actual_entry_rows(trades: pl.DataFrame, lifecycles: pl.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in trades.to_dicts() if not trades.is_empty() else []:
        if not _is_open_long_like(row):
            continue
        symbol = normalize_symbol(row.get("normalized_symbol") or row.get("symbol"))
        ts = _coerce_datetime(row.get("ts_utc") or row.get("ts"))
        price = _float_or_none(row.get("price") or row.get("fill_px") or row.get("fill_price"))
        if symbol and ts and price:
            rows.append(
                {
                    **row,
                    "symbol": symbol,
                    "entry_ts": ts,
                    "entry_px": price,
                    "intent": str(row.get("intent") or row.get("action") or "OPEN_LONG"),
                    "entry_reason": str(
                        row.get("entry_reason")
                        or row.get("action")
                        or row.get("final_decision")
                        or ""
                    ),
                    "realized_net_bps": _float_or_none(
                        row.get("realized_net_bps") or row.get("net_bps") or row.get("pnl_bps")
                    ),
                }
            )
    for row in lifecycles.to_dicts() if not lifecycles.is_empty() else []:
        if not _is_open_long_like(row):
            continue
        symbol = normalize_symbol(row.get("normalized_symbol") or row.get("symbol"))
        ts = _coerce_datetime(
            row.get("ts_utc")
            or row.get("last_fill_ts")
            or row.get("submit_ts")
            or row.get("decision_ts")
        )
        price = _float_or_none(row.get("avg_fill_px") or row.get("fill_px") or row.get("entry_px"))
        if symbol and ts and price:
            rows.append(
                {
                    **row,
                    "symbol": symbol,
                    "entry_ts": ts,
                    "entry_px": price,
                    "intent": str(row.get("intent") or "OPEN_LONG"),
                    "entry_reason": str(row.get("entry_reason") or row.get("intent") or ""),
                    "realized_net_bps": _float_or_none(
                        row.get("realized_net_bps")
                        or row.get("net_bps")
                        or row.get("total_realized_cost_bps")
                    ),
                }
            )
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _source_event_key(row, symbol=str(row.get("symbol") or ""), ts=row.get("entry_ts"))
        unique[key] = row
    return list(unique.values())


def _late_subject_from_actual(row: dict[str, Any]) -> dict[str, Any] | None:
    symbol = normalize_symbol(row.get("symbol"))
    ts = _coerce_datetime(row.get("entry_ts"))
    price = _float_or_none(row.get("entry_px"))
    if not symbol or ts is None or price is None:
        return None
    return {
        "strategy_candidate": "v5.late_entry_chase_guard_shadow",
        "source_type": "actual_entry",
        "run_id": row.get("run_id"),
        "candidate_id": "",
        "source_event_key": _source_event_key(row, symbol=symbol, ts=ts),
        "symbol": symbol,
        "ts_utc": ts,
        "entry_or_candidate_px": price,
        "f4_volume_expansion": _float_or_none(row.get("f4_volume_expansion")),
        "f5_rsi_trend_confirm": _float_or_none(row.get("f5_rsi_trend_confirm")),
        "realized_net_bps": _float_or_none(row.get("realized_net_bps")),
    }


def _late_subject_from_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    symbol = normalize_symbol(row.get("normalized_symbol") or row.get("symbol"))
    ts = _coerce_datetime(row.get("ts_utc") or row.get("ts"))
    if not symbol or ts is None:
        return None
    price = _float_or_none(
        row.get("entry_close")
        or row.get("candidate_px")
        or row.get("price")
        or row.get("close")
        or row.get("last_price")
    )
    return {
        "strategy_candidate": "v5.late_entry_chase_guard_shadow",
        "source_type": "candidate_event",
        "run_id": row.get("run_id"),
        "candidate_id": str(row.get("candidate_id") or ""),
        "source_event_key": _source_event_key(row, symbol=symbol, ts=ts),
        "symbol": symbol,
        "ts_utc": ts,
        "entry_or_candidate_px": price,
        "f4_volume_expansion": _float_or_none(row.get("f4_volume_expansion")),
        "f5_rsi_trend_confirm": _float_or_none(row.get("f5_rsi_trend_confirm")),
        "realized_net_bps": None,
    }


def _late_shadow_row(
    subject: dict[str, Any],
    market_by_symbol: dict[str, list[dict[str, Any]]],
    label_context: dict[str, dict[int, dict[str, Any]]],
    *,
    ctx: _BuildContext,
) -> dict[str, Any]:
    symbol = str(subject["symbol"])
    ts = subject["ts_utc"]
    px = _float_or_none(subject.get("entry_or_candidate_px"))
    bars = market_by_symbol.get(symbol, [])
    if px is None:
        px = _market_close_at_or_before(bars, ts)
    low12 = _pre_window_stats(bars, ts, 12)
    low24 = _pre_window_stats(bars, ts, 24)
    f4 = _float_or_none(subject.get("f4_volume_expansion"))
    f5 = _float_or_none(subject.get("f5_rsi_trend_confirm"))
    entry_vs_12h_low = _entry_vs_low_bps(px, low12.get("low"))
    entry_pos_12h = _position_in_range(px, low12.get("low"), low12.get("high"))
    late_chase = (
        (entry_vs_12h_low or 0.0) > 250.0
        and (entry_pos_12h or 0.0) > 0.70
        and (f4 is None or f4 < 0.50)
        and (f5 is None or f5 < 0.45)
    )
    candidate_labels = label_context.get(str(subject.get("candidate_id") or ""), {})
    forward_24 = _float_or_none(candidate_labels.get(24, {}).get("net_bps_after_cost"))
    forward_48 = _float_or_none(candidate_labels.get(48, {}).get("net_bps_after_cost"))
    realized = _float_or_none(subject.get("realized_net_bps"))
    outcome = _classify_outcome(realized if realized is not None else forward_24)
    return _common(ctx, mode="shadow") | {
        "strategy_candidate": "v5.late_entry_chase_guard_shadow",
        "source_type": subject.get("source_type"),
        "run_id": str(subject.get("run_id") or ""),
        "candidate_id": str(subject.get("candidate_id") or ""),
        "source_event_key": str(subject.get("source_event_key") or ""),
        "symbol": symbol,
        "ts_utc": ts,
        "entry_or_candidate_px": px,
        "recent_12h_low": low12.get("low"),
        "recent_24h_low": low24.get("low"),
        "entry_vs_12h_low_bps": entry_vs_12h_low,
        "entry_vs_24h_low_bps": _entry_vs_low_bps(px, low24.get("low")),
        "entry_position_in_12h_range": entry_pos_12h,
        "f4_volume_expansion": f4,
        "f5_rsi_trend_confirm": f5,
        "late_chase_risk": late_chase,
        "would_block_if_enabled": late_chase,
        "realized_net_bps": realized,
        "forward_24h_net_bps": forward_24,
        "forward_48h_net_bps": forward_48,
        "outcome_class": outcome,
    }


def _is_open_long_like(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in ["intent", "action", "final_decision", "side"]
    )
    return "open_long" in text or ("entry" in text and "buy" in text) or " buy" in f" {text}"


def _normalize_market_bars(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    normalized = frame
    if "symbol" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("symbol").map_elements(normalize_symbol, return_dtype=pl.Utf8).alias("symbol")
        )
    if "ts" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("ts").cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False).alias("ts")
        )
    if "timeframe" in normalized.columns:
        normalized = normalized.filter(pl.col("timeframe").cast(pl.Utf8).str.to_uppercase() == "1H")
    return normalized


def _market_rows_by_symbol(market_bars: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    if market_bars.is_empty():
        return rows_by_symbol
    for row in market_bars.sort("ts").to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        ts = _coerce_datetime(row.get("ts"))
        if not symbol or ts is None:
            continue
        rows_by_symbol.setdefault(symbol, []).append(
            {
                **row,
                "symbol": symbol,
                "ts": ts,
                "open": _float_or_none(row.get("open")),
                "high": _float_or_none(row.get("high")),
                "low": _float_or_none(row.get("low")),
                "close": _float_or_none(row.get("close")),
            }
        )
    return rows_by_symbol


def _pre_window_stats(
    bars: list[dict[str, Any]],
    entry_ts: datetime,
    hours: int,
) -> dict[str, float | None]:
    start = entry_ts - timedelta(hours=hours)
    selected = [
        row
        for row in bars
        if start <= row["ts"] < entry_ts
        and _float_or_none(row.get("low")) is not None
        and _float_or_none(row.get("high")) is not None
    ]
    if not selected:
        return {"low": None, "high": None}
    return {
        "low": min(float(row["low"]) for row in selected),
        "high": max(float(row["high"]) for row in selected),
    }


def _entry_vs_low_bps(price: float | None, low: float | None) -> float | None:
    if price is None or low is None or low <= 0:
        return None
    return (price / low - 1.0) * 10_000.0


def _position_in_range(
    price: float | None,
    low: float | None,
    high: float | None,
) -> float | None:
    if price is None or low is None or high is None or high <= low:
        return None
    return max(0.0, min(1.0, (price - low) / (high - low)))


def _missed_low_diagnosis(
    *,
    entry_vs_24h_low_bps: float | None,
    position_24h: float | None,
    realized_net_bps: float | None,
) -> str:
    if (entry_vs_24h_low_bps or 0.0) > 250.0 and realized_net_bps is not None:
        return "late_chase_loss" if realized_net_bps < 0.0 else "late_but_trend_profitable"
    if (entry_vs_24h_low_bps or 0.0) < 100.0 or (position_24h is not None and position_24h < 0.35):
        return "early_entry"
    return "normal_entry"


def _label_context(labels: pl.DataFrame) -> dict[str, dict[int, dict[str, Any]]]:
    context: dict[str, dict[int, dict[str, Any]]] = {}
    if labels.is_empty() or "candidate_id" not in labels.columns:
        return context
    for row in labels.to_dicts():
        candidate_id = str(row.get("candidate_id") or "")
        horizon = int(_float_or_none(row.get("horizon_hours")) or 0)
        if candidate_id and horizon:
            context.setdefault(candidate_id, {})[horizon] = row
    return context


def _candidate_price(
    row: dict[str, Any],
    bars: list[dict[str, Any]],
    ts: datetime,
) -> float | None:
    price = _float_or_none(
        row.get("entry_close")
        or row.get("candidate_px")
        or row.get("current_px")
        or row.get("price")
        or row.get("close")
        or row.get("last_price")
    )
    return price if price is not None else _market_close_at_or_before(bars, ts)


def _market_close_at_or_before(bars: list[dict[str, Any]], ts: datetime) -> float | None:
    close = None
    for row in bars:
        if row["ts"] <= ts:
            close = _float_or_none(row.get("close"))
        if row["ts"] > ts:
            break
    return close


def _pullback_reversal_candidate_ok(
    *,
    row: dict[str, Any],
    current_px: float,
    pre_24h: dict[str, float | None],
    pullback_bps: float | None,
    f4: float | None,
    f5: float | None,
    bars: list[dict[str, Any]],
    ts: datetime,
) -> bool:
    if str(row.get("regime_state") or row.get("risk_level") or "").lower() == "risk_off":
        return False
    low = pre_24h.get("low")
    if low is None or pullback_bps is None:
        return False
    if pullback_bps < 100.0 or pullback_bps > 500.0:
        return False
    if current_px <= low * 1.005:
        return False
    if not _recent_2h_no_new_low(bars, ts):
        return False
    if f5 is not None and f5 < -0.10:
        return False
    if f4 is not None and f4 < -0.50:
        return False
    spread = _float_or_none(row.get("estimated_spread_bps") or row.get("spread_bps"))
    return spread is None or spread < 50.0


def _pullback_from_high_bps(price: float, high: float | None) -> float | None:
    if high is None or high <= 0:
        return None
    return (high / price - 1.0) * 10_000.0


def _recent_2h_no_new_low(bars: list[dict[str, Any]], ts: datetime) -> bool:
    recent_start = ts - timedelta(hours=2)
    previous_start = ts - timedelta(hours=24)
    recent = [row for row in bars if recent_start <= row["ts"] <= ts]
    previous = [row for row in bars if previous_start <= row["ts"] < recent_start]
    if not recent or not previous:
        return True
    recent_low = min(_float_or_none(row.get("low")) or math.inf for row in recent)
    previous_low = min(_float_or_none(row.get("low")) or math.inf for row in previous)
    return recent_low > previous_low


def _forward_label(
    bars: list[dict[str, Any]],
    ts: datetime,
    current_px: float,
    horizon_hours: int,
    roundtrip_cost_bps: float,
) -> dict[str, Any]:
    end = ts + timedelta(hours=horizon_hours)
    future = [row for row in bars if ts < row["ts"] <= end]
    if not future:
        return {
            "gross_bps": None,
            "net_bps_after_cost": None,
            "mfe_bps": None,
            "mae_bps": None,
            "win": None,
            "label_status": "pending",
        }
    end_close = _float_or_none(future[-1].get("close"))
    highs = [_float_or_none(row.get("high")) for row in future]
    lows = [_float_or_none(row.get("low")) for row in future]
    highs = [value for value in highs if value is not None]
    lows = [value for value in lows if value is not None]
    gross = ((end_close / current_px - 1.0) * 10_000.0) if end_close else None
    net = (gross - roundtrip_cost_bps) if gross is not None else None
    return {
        "gross_bps": gross,
        "net_bps_after_cost": net,
        "mfe_bps": ((max(highs) / current_px - 1.0) * 10_000.0) if highs else None,
        "mae_bps": ((min(lows) / current_px - 1.0) * 10_000.0) if lows else None,
        "win": None if net is None else net > 0.0,
        "label_status": "complete",
    }


def _roundtrip_cost_by_symbol(costs: pl.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    if costs.is_empty():
        return output
    for row in costs.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        cost = _float_or_none(row.get("roundtrip_all_in_cost_bps"))
        if cost is None:
            one_way = _float_or_none(
                row.get("one_way_all_in_cost_bps")
                or row.get("total_cost_bps_p75")
                or row.get("selected_total_cost_bps")
            )
            cost = one_way * 2.0 if one_way is not None else None
        if cost is None:
            continue
        output[symbol] = max(output.get(symbol, 0.0), float(cost))
    return output


def _pullback_readiness_reasons(
    *,
    sample_count: int,
    recent_count: int,
    avg_net: float | None,
    win_rate: float | None,
    p25: float | None,
    avg_mae: float | None,
) -> list[str]:
    reasons: list[str] = []
    if sample_count < 50:
        reasons.append("insufficient_sample_count")
    if recent_count < 10:
        reasons.append("insufficient_recent_7d_samples")
    if avg_net is None or avg_net <= 50.0:
        reasons.append("weak_24h_avg_net_bps")
    if win_rate is None or win_rate <= 0.55:
        reasons.append("weak_24h_win_rate")
    if p25 is None or p25 <= -50.0:
        reasons.append("weak_24h_p25_net_bps")
    if avg_mae is None or avg_mae <= -120.0:
        reasons.append("excessive_avg_mae_bps")
    return reasons


def _pullback_candidate_name(symbol: str) -> str:
    base = normalize_symbol(symbol).split("-")[0].lower()
    return f"v5.pullback_reversal_shadow_{base}"


def _common(ctx: _BuildContext, *, mode: str) -> dict[str, Any]:
    return {
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
        "schema_version": ENTRY_QUALITY_SCHEMA_VERSION,
        "generated_at_utc": ctx.generated_at,
        "generated_from_bundle_id": ctx.generated_from_bundle_id,
        "as_of_date": ctx.as_of_date.isoformat(),
        "window_hours": ctx.window_hours,
        "source": SOURCE_NAME,
        "mode": mode,
    }


def _publish_daily(
    root: Path,
    relative_path: Path,
    frame: pl.DataFrame,
    key_columns: list[str],
) -> int:
    if frame.is_empty():
        return read_parquet_dataset(root / relative_path).height
    return upsert_parquet_dataset(frame, root / relative_path, key_columns=key_columns)


def _latest_bundle_id(root: Path) -> str:
    for dataset in [
        root / "gold" / "strategy_health_daily",
        root / "silver" / "v5_candidate_event",
        root / "silver" / "v5_trade_event",
    ]:
        frame = read_parquet_dataset(dataset)
        if frame.is_empty():
            continue
        for column in [
            "latest_bundle_sha256",
            "bundle_sha256",
            "bundle_name",
            "source_bundle_ts",
            "bundle_ts",
        ]:
            if column in frame.columns:
                values = [str(value) for value in frame.get_column(column).drop_nulls().to_list()]
                if values:
                    return max(values)
    return ""


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return datetime.now(UTC).date()
    return date.fromisoformat(str(value)[:10])


def _source_event_key(row: dict[str, Any], *, symbol: str, ts: Any) -> str:
    for key in [
        "event_key",
        "source_event_key",
        "lifecycle_id",
        "trade_id",
        "order_id",
        "candidate_id",
    ]:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    dt = _coerce_datetime(ts)
    material = "|".join(
        [
            str(row.get("run_id") or ""),
            symbol,
            dt.isoformat() if dt else "",
            str(row.get("row_index") or ""),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"entry_quality:{digest}"


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mean(values: Any) -> float | None:
    numbers = [_float_or_none(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int((len(ordered) - 1) * q)
    return ordered[index]


def _group_dicts(rows: list[dict[str, Any]], column: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(column) or "UNKNOWN"), []).append(row)
    return grouped


def _classify_outcome(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "profit" if value >= 0.0 else "loss"


def _outcome_value(row: dict[str, Any]) -> float | None:
    for column in ["realized_net_bps", "forward_24h_net_bps", "forward_48h_net_bps"]:
        value = _float_or_none(row.get(column))
        if value is not None:
            return value
    return None
