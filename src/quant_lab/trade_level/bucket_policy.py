from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import polars as pl

from quant_lab.opportunity_cost.ledger import (
    build_opportunity_cost_by_bucket,
)

TRADE_LEVEL_BUCKET_POLICY_SCHEMA_VERSION = "trade_level_bucket_policy.v0.1"

TRADE_LEVEL_BUCKET_POLICY_SCHEMA = {
    "schema_version": pl.Utf8,
    "policy_date": pl.Date,
    "bucket_key": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "regime": pl.Utf8,
    "risk_level": pl.Utf8,
    "rank_bucket": pl.Utf8,
    "alpha6_bucket": pl.Utf8,
    "expected_edge_ratio_bucket": pl.Utf8,
    "cost_source": pl.Utf8,
    "cost_gate_bucket": pl.Utf8,
    "sample_count": pl.Int64,
    "false_block_count": pl.Int64,
    "loss_saved_count": pl.Int64,
    "false_allow_count": pl.Int64,
    "correct_allow_count": pl.Int64,
    "missed_profit_bps_sum": pl.Float64,
    "loss_saved_bps_sum": pl.Float64,
    "veto_net_value_bps": pl.Float64,
    "recent_7d_veto_net_value_bps": pl.Float64,
    "high_confidence_false_block_count": pl.Int64,
    "high_confidence_loss_saved_count": pl.Int64,
    "policy_action": pl.Utf8,
    "policy_reason": pl.Utf8,
    "policy_confidence": pl.Utf8,
    "min_required_observability": pl.Float64,
    "max_single_order_usdt": pl.Float64,
    "daily_trade_limit": pl.Int64,
    "expires_at": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


def build_trade_level_bucket_policy(
    *,
    opportunity_cost_events: pl.DataFrame | None = None,
    opportunity_cost_by_bucket: pl.DataFrame | None = None,
    policy_date: str | date | None = None,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    created = created_at or datetime.now(UTC)
    day = _parse_day(policy_date, created)
    event_frame = opportunity_cost_events if opportunity_cost_events is not None else pl.DataFrame()
    bucket_input = (
        opportunity_cost_by_bucket if opportunity_cost_by_bucket is not None else pl.DataFrame()
    )
    bucket_frame = _historical_bucket_frame(
        opportunity_cost_events=event_frame,
        opportunity_cost_by_bucket=bucket_input,
        policy_date=day,
        created_at=created,
    )
    if bucket_frame.is_empty():
        return pl.DataFrame(schema=TRADE_LEVEL_BUCKET_POLICY_SCHEMA)
    expires_at = datetime.combine(day + timedelta(days=1), time.min, tzinfo=UTC)
    rows = [
        _policy_row(row, policy_date=day, expires_at=expires_at, created=created)
        for row in bucket_frame.to_dicts()
    ]
    rows.sort(
        key=lambda row: (
            _policy_priority(row.get("policy_action")),
            -(_int(row.get("sample_count")) or 0),
            _text(row.get("bucket_key")),
        )
    )
    return _frame(rows, TRADE_LEVEL_BUCKET_POLICY_SCHEMA)


def _historical_bucket_frame(
    *,
    opportunity_cost_events: pl.DataFrame,
    opportunity_cost_by_bucket: pl.DataFrame,
    policy_date: date,
    created_at: datetime,
) -> pl.DataFrame:
    if not opportunity_cost_events.is_empty() and "day" in opportunity_cost_events.columns:
        try:
            historical_events = opportunity_cost_events.filter(
                pl.col("day").cast(pl.Date) < policy_date
            )
        except Exception:
            historical_events = pl.DataFrame()
        if not historical_events.is_empty():
            return build_opportunity_cost_by_bucket(historical_events, created_at=created_at)
    return opportunity_cost_by_bucket


def _policy_row(
    row: dict[str, Any],
    *,
    policy_date: date,
    expires_at: datetime,
    created: datetime,
) -> dict[str, Any]:
    sample_count = _int(row.get("sample_count")) or 0
    false_blocks = _int(row.get("false_block_count")) or 0
    loss_saved = _int(row.get("loss_saved_count")) or 0
    missed = _float(row.get("missed_profit_bps_sum")) or 0.0
    saved = _float(row.get("loss_saved_bps_sum")) or 0.0
    veto_net = _float(row.get("veto_net_value_bps"))
    veto_net = veto_net if veto_net is not None else saved - missed
    high_false_blocks = _int(row.get("high_confidence_false_block_count")) or 0
    high_loss_saved = _int(row.get("high_confidence_loss_saved_count")) or 0
    action = _text(row.get("policy_action") or row.get("recommended_trade_level_decision")).upper()
    if action not in {"RISK_BLOCK", "MICRO_CANARY_REVIEW", "MICRO_CANARY_ALLOW", "PAPER_ONLY"}:
        action = _policy_action(
            sample_count=sample_count,
            false_block_count=false_blocks,
            loss_saved_count=loss_saved,
            missed_profit_bps_sum=missed,
            loss_saved_bps_sum=saved,
            veto_net_value_bps=veto_net,
            high_confidence_false_block_count=high_false_blocks,
            high_confidence_loss_saved_count=high_loss_saved,
        )
    reason = _policy_reason(
        action=action,
        sample_count=sample_count,
        false_block_count=false_blocks,
        loss_saved_count=loss_saved,
        missed_profit_bps_sum=missed,
        loss_saved_bps_sum=saved,
        veto_net_value_bps=veto_net,
        high_confidence_false_block_count=high_false_blocks,
        high_confidence_loss_saved_count=high_loss_saved,
    )
    confidence = _text(row.get("policy_confidence"))
    if not confidence:
        confidence = "high" if action in {"RISK_BLOCK", "MICRO_CANARY_REVIEW"} else "low"
    return {
        "schema_version": TRADE_LEVEL_BUCKET_POLICY_SCHEMA_VERSION,
        "policy_date": policy_date,
        "bucket_key": _text(row.get("bucket_key")),
        "symbol": _text(row.get("symbol")),
        "strategy_candidate": _text(row.get("strategy_candidate")),
        "regime": _text(row.get("regime")),
        "risk_level": _text(row.get("risk_level")),
        "rank_bucket": _text(row.get("rank_bucket")),
        "alpha6_bucket": _text(row.get("alpha6_bucket")),
        "expected_edge_ratio_bucket": _text(row.get("expected_edge_ratio_bucket")),
        "cost_source": _text(row.get("cost_source")),
        "cost_gate_bucket": _text(row.get("cost_gate_bucket")),
        "sample_count": sample_count,
        "false_block_count": false_blocks,
        "loss_saved_count": loss_saved,
        "false_allow_count": _int(row.get("false_allow_count")) or 0,
        "correct_allow_count": _int(row.get("correct_allow_count")) or 0,
        "missed_profit_bps_sum": missed,
        "loss_saved_bps_sum": saved,
        "veto_net_value_bps": veto_net,
        "recent_7d_veto_net_value_bps": _float(row.get("recent_7d_veto_net_value_bps")) or veto_net,
        "high_confidence_false_block_count": high_false_blocks,
        "high_confidence_loss_saved_count": high_loss_saved,
        "policy_action": action,
        "policy_reason": reason,
        "policy_confidence": confidence,
        "min_required_observability": (
            0.8 if action in {"MICRO_CANARY_REVIEW", "MICRO_CANARY_ALLOW"} else 0.0
        ),
        "max_single_order_usdt": 5.0 if action == "MICRO_CANARY_ALLOW" else 0.0,
        "daily_trade_limit": 1 if action == "MICRO_CANARY_ALLOW" else 0,
        "expires_at": expires_at,
        "created_at": created,
        "source": "quant_lab.trade_level.bucket_policy",
    }


def _policy_action(
    *,
    sample_count: int,
    false_block_count: int,
    loss_saved_count: int,
    missed_profit_bps_sum: float,
    loss_saved_bps_sum: float,
    veto_net_value_bps: float,
    high_confidence_false_block_count: int,
    high_confidence_loss_saved_count: int,
) -> str:
    if (
        sample_count >= 5
        and loss_saved_count >= 3
        and loss_saved_bps_sum > missed_profit_bps_sum
        and veto_net_value_bps > 0.0
        and high_confidence_loss_saved_count >= 3
    ):
        return "RISK_BLOCK"
    if (
        sample_count >= 5
        and false_block_count >= 3
        and missed_profit_bps_sum > loss_saved_bps_sum
        and veto_net_value_bps < 0.0
        and high_confidence_false_block_count >= 3
    ):
        return "MICRO_CANARY_REVIEW"
    return "PAPER_ONLY"


def _policy_reason(
    *,
    action: str,
    sample_count: int,
    false_block_count: int,
    loss_saved_count: int,
    missed_profit_bps_sum: float,
    loss_saved_bps_sum: float,
    veto_net_value_bps: float,
    high_confidence_false_block_count: int,
    high_confidence_loss_saved_count: int,
) -> str:
    if action == "RISK_BLOCK":
        return "loss_saved_bucket_positive_veto_value"
    if action == "MICRO_CANARY_REVIEW":
        return "false_block_bucket_negative_veto_value_manual_review_only"
    if action == "MICRO_CANARY_ALLOW":
        return "explicit_micro_canary_allow_policy"
    missing: list[str] = []
    if sample_count < 5:
        missing.append("sample_count_lt_5")
    if false_block_count < 3 and loss_saved_count < 3:
        missing.append("insufficient_directional_outcomes")
    if high_confidence_false_block_count < 3 and high_confidence_loss_saved_count < 3:
        missing.append("insufficient_high_confidence_outcomes")
    if veto_net_value_bps == 0.0 or missed_profit_bps_sum == loss_saved_bps_sum:
        missing.append("veto_value_flat")
    return ";".join(missing) or "paper_only_until_bucket_policy_clear"


def _policy_priority(value: Any) -> int:
    return {
        "RISK_BLOCK": 0,
        "MICRO_CANARY_REVIEW": 1,
        "MICRO_CANARY_ALLOW": 2,
        "PAPER_ONLY": 3,
    }.get(_text(value).upper(), 4)


def _parse_day(value: str | date | None, created: datetime) -> date:
    if isinstance(value, date):
        return value
    if value and str(value).lower() != "auto":
        return date.fromisoformat(str(value))
    return created.date()


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row").select(
        [pl.col(name).cast(dtype, strict=False).alias(name) for name, dtype in schema.items()]
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None
