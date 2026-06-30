from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import UTC, date, datetime
from typing import Any

import polars as pl

OPPORTUNITY_COST_EVENT_SCHEMA_VERSION = "quant_lab_opportunity_cost_event.v0.1"
OPPORTUNITY_COST_DAILY_SCHEMA_VERSION = "quant_lab_opportunity_cost_daily.v0.1"
OPPORTUNITY_COST_BY_BUCKET_SCHEMA_VERSION = "opportunity_cost_by_bucket.v0.1"

OPPORTUNITY_COST_EVENT_SCHEMA = {
    "schema_version": pl.Utf8,
    "event_id": pl.Utf8,
    "sample_id": pl.Utf8,
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "day": pl.Date,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "v5_would_open": pl.Boolean,
    "quant_lab_decision": pl.Utf8,
    "quant_lab_would_block": pl.Boolean,
    "quant_lab_would_allow": pl.Boolean,
    "actual_action": pl.Utf8,
    "actual_submitted": pl.Boolean,
    "after_cost_bps": pl.Float64,
    "label_horizon_hours": pl.Int64,
    "best_hindsight_action": pl.Utf8,
    "false_block": pl.Boolean,
    "loss_saved": pl.Boolean,
    "false_allow": pl.Boolean,
    "correct_allow": pl.Boolean,
    "missed_profit_bps": pl.Float64,
    "loss_saved_bps": pl.Float64,
    "false_allow_loss_bps": pl.Float64,
    "captured_profit_bps": pl.Float64,
    "benefit_bps": pl.Float64,
    "regret_bps": pl.Float64,
    "regret_type": pl.Utf8,
    "high_confidence_v5": pl.Boolean,
    "bucket_key": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

OPPORTUNITY_COST_DAILY_SCHEMA = {
    "schema_version": pl.Utf8,
    "day": pl.Date,
    "total_v5_would_open_count": pl.Int64,
    "quant_lab_would_block_count": pl.Int64,
    "quant_lab_would_allow_count": pl.Int64,
    "false_block_count": pl.Int64,
    "false_block_profit_bps_sum": pl.Float64,
    "false_block_profit_bps_mean": pl.Float64,
    "false_block_profit_bps_median": pl.Float64,
    "loss_saved_count": pl.Int64,
    "loss_saved_bps_sum": pl.Float64,
    "loss_saved_bps_mean": pl.Float64,
    "loss_saved_bps_median": pl.Float64,
    "false_allow_count": pl.Int64,
    "false_allow_loss_bps_sum": pl.Float64,
    "correct_allow_count": pl.Int64,
    "captured_profit_bps_sum": pl.Float64,
    "high_confidence_false_block_count": pl.Int64,
    "high_confidence_loss_saved_count": pl.Int64,
    "veto_net_value_bps": pl.Float64,
    "veto_precision": pl.Float64,
    "veto_false_block_rate": pl.Float64,
    "veto_loss_saved_rate": pl.Float64,
    "opportunity_cost_status": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

OPPORTUNITY_COST_BY_BUCKET_SCHEMA = {
    "schema_version": pl.Utf8,
    "bucket_key": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "regime": pl.Utf8,
    "risk_level": pl.Utf8,
    "rank_bucket": pl.Utf8,
    "alpha6_bucket": pl.Utf8,
    "expected_edge_ratio_bucket": pl.Utf8,
    "cost_gate_bucket": pl.Utf8,
    "sample_count": pl.Int64,
    "false_block_count": pl.Int64,
    "loss_saved_count": pl.Int64,
    "false_allow_count": pl.Int64,
    "correct_allow_count": pl.Int64,
    "missed_profit_bps_sum": pl.Float64,
    "loss_saved_bps_sum": pl.Float64,
    "veto_net_value_bps": pl.Float64,
    "high_confidence_false_block_count": pl.Int64,
    "opportunity_exception_candidate": pl.Boolean,
    "recommended_trade_level_decision": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


def build_opportunity_cost_frames(
    events: pl.DataFrame,
    labels: pl.DataFrame,
    judgments: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> dict[str, pl.DataFrame]:
    created = created_at or datetime.now(UTC)
    event_frame = build_opportunity_cost_events(
        events,
        labels,
        judgments,
        created_at=created,
    )
    daily = build_opportunity_cost_daily(event_frame, created_at=created)
    buckets = build_opportunity_cost_by_bucket(event_frame, created_at=created)
    return {
        "quant_lab_opportunity_cost_event": event_frame,
        "quant_lab_opportunity_cost_daily": daily,
        "opportunity_cost_by_bucket": buckets,
    }


def build_opportunity_cost_events(
    events: pl.DataFrame,
    labels: pl.DataFrame,
    judgments: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    created = created_at or datetime.now(UTC)
    if events.is_empty():
        return pl.DataFrame(schema=OPPORTUNITY_COST_EVENT_SCHEMA)
    labels_by_event = {str(row.get("event_id") or ""): row for row in labels.to_dicts()}
    judgments_by_event = {
        str(row.get("event_id") or ""): row for row in judgments.to_dicts()
    }
    rows = []
    for event in events.to_dicts():
        event_id = _text(event.get("event_id"))
        label = labels_by_event.get(event_id, {})
        judgment = judgments_by_event.get(event_id, {})
        after_cost, horizon = _preferred_label_value(label)
        quant_lab_decision = _text(judgment.get("trade_level_decision")) or "UNKNOWN"
        quant_lab_would_block = quant_lab_decision not in {
            "MICRO_CANARY_ALLOW",
            "LIVE_SMALL_ALLOW",
        }
        v5_would_open = _bool(event.get("v5_would_open")) is True
        quant_lab_would_allow = not quant_lab_would_block
        false_block = v5_would_open and quant_lab_would_block and _positive(after_cost)
        loss_saved = v5_would_open and quant_lab_would_block and _negative(after_cost)
        false_allow = quant_lab_would_allow and _negative(after_cost)
        correct_allow = quant_lab_would_allow and _positive(after_cost)
        high_confidence = _bool(judgment.get("v5_high_confidence_opportunity")) is True
        decision_ts = _timestamp(event.get("decision_ts"))
        bucket = _bucket_fields(event)
        rows.append(
            {
                "schema_version": OPPORTUNITY_COST_EVENT_SCHEMA_VERSION,
                "event_id": event_id,
                "sample_id": event_id,
                "decision_ts": decision_ts,
                "day": decision_ts.date() if decision_ts else None,
                "symbol": _text(event.get("symbol")),
                "strategy_candidate": _text(event.get("strategy_candidate")),
                "v5_would_open": v5_would_open,
                "quant_lab_decision": quant_lab_decision,
                "quant_lab_would_block": quant_lab_would_block,
                "quant_lab_would_allow": quant_lab_would_allow,
                "actual_action": "SUBMITTED" if _bool(event.get("actual_submitted")) else "NONE",
                "actual_submitted": _bool(event.get("actual_submitted")) is True,
                "after_cost_bps": after_cost,
                "label_horizon_hours": horizon,
                "best_hindsight_action": _best_hindsight_action(after_cost),
                "false_block": false_block,
                "loss_saved": loss_saved,
                "false_allow": false_allow,
                "correct_allow": correct_allow,
                "missed_profit_bps": after_cost if false_block else 0.0,
                "loss_saved_bps": abs(after_cost) if loss_saved and after_cost is not None else 0.0,
                "false_allow_loss_bps": (
                    abs(after_cost) if false_allow and after_cost is not None else 0.0
                ),
                "captured_profit_bps": after_cost if correct_allow else 0.0,
                "benefit_bps": abs(after_cost) if loss_saved and after_cost is not None else 0.0,
                "regret_bps": _regret_bps(
                    after_cost=after_cost,
                    false_block=false_block,
                    false_allow=false_allow,
                ),
                "regret_type": _regret_type(
                    false_block=false_block,
                    loss_saved=loss_saved,
                    false_allow=false_allow,
                    correct_allow=correct_allow,
                    after_cost=after_cost,
                ),
                "high_confidence_v5": high_confidence,
                "bucket_key": bucket["bucket_key"],
                "created_at": created,
                "source": "quant_lab.opportunity_cost.event",
            }
        )
    return _frame(rows, OPPORTUNITY_COST_EVENT_SCHEMA)


def build_opportunity_cost_daily(
    event_frame: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    created = created_at or datetime.now(UTC)
    if event_frame.is_empty():
        return pl.DataFrame(schema=OPPORTUNITY_COST_DAILY_SCHEMA)
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in event_frame.to_dicts():
        day = row.get("day")
        if isinstance(day, date):
            grouped[day].append(row)
    rows = [_daily_row(day, rows_for_day, created) for day, rows_for_day in grouped.items()]
    rows.sort(key=lambda row: row["day"])
    return _frame(rows, OPPORTUNITY_COST_DAILY_SCHEMA)


def build_opportunity_cost_by_bucket(
    event_frame: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    created = created_at or datetime.now(UTC)
    if event_frame.is_empty():
        return pl.DataFrame(schema=OPPORTUNITY_COST_BY_BUCKET_SCHEMA)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_frame.to_dicts():
        grouped[_text(row.get("bucket_key"))].append(row)
    rows = [
        _bucket_row(bucket_key, rows_for_bucket, created)
        for bucket_key, rows_for_bucket in grouped.items()
    ]
    rows.sort(
        key=lambda row: (
            row["opportunity_exception_candidate"] is not True,
            row["veto_net_value_bps"],
        )
    )
    return _frame(rows, OPPORTUNITY_COST_BY_BUCKET_SCHEMA)


def _daily_row(day: date, rows: list[dict[str, Any]], created: datetime) -> dict[str, Any]:
    v5_open_rows = [row for row in rows if row.get("v5_would_open")]
    blocked = [row for row in v5_open_rows if row.get("quant_lab_would_block")]
    allowed = [row for row in v5_open_rows if row.get("quant_lab_would_allow")]
    false_blocks = [row for row in rows if row.get("false_block")]
    loss_saved = [row for row in rows if row.get("loss_saved")]
    false_allows = [row for row in rows if row.get("false_allow")]
    correct_allows = [row for row in rows if row.get("correct_allow")]
    false_block_values = [_float(row.get("missed_profit_bps")) or 0.0 for row in false_blocks]
    loss_saved_values = [_float(row.get("loss_saved_bps")) or 0.0 for row in loss_saved]
    false_allow_values = [_float(row.get("false_allow_loss_bps")) or 0.0 for row in false_allows]
    captured_values = [_float(row.get("captured_profit_bps")) or 0.0 for row in correct_allows]
    veto_net = sum(loss_saved_values) - sum(false_block_values)
    return {
        "schema_version": OPPORTUNITY_COST_DAILY_SCHEMA_VERSION,
        "day": day,
        "total_v5_would_open_count": len(v5_open_rows),
        "quant_lab_would_block_count": len(blocked),
        "quant_lab_would_allow_count": len(allowed),
        "false_block_count": len(false_blocks),
        "false_block_profit_bps_sum": sum(false_block_values),
        "false_block_profit_bps_mean": _mean(false_block_values),
        "false_block_profit_bps_median": _median(false_block_values),
        "loss_saved_count": len(loss_saved),
        "loss_saved_bps_sum": sum(loss_saved_values),
        "loss_saved_bps_mean": _mean(loss_saved_values),
        "loss_saved_bps_median": _median(loss_saved_values),
        "false_allow_count": len(false_allows),
        "false_allow_loss_bps_sum": sum(false_allow_values),
        "correct_allow_count": len(correct_allows),
        "captured_profit_bps_sum": sum(captured_values),
        "high_confidence_false_block_count": sum(
            1 for row in false_blocks if row.get("high_confidence_v5")
        ),
        "high_confidence_loss_saved_count": sum(
            1 for row in loss_saved if row.get("high_confidence_v5")
        ),
        "veto_net_value_bps": veto_net,
        "veto_precision": _safe_ratio(len(loss_saved), len(blocked)),
        "veto_false_block_rate": _safe_ratio(len(false_blocks), len(blocked)),
        "veto_loss_saved_rate": _safe_ratio(len(loss_saved), len(blocked)),
        "opportunity_cost_status": _opportunity_cost_status(veto_net, blocked),
        "created_at": created,
        "source": "quant_lab.opportunity_cost.daily",
    }


def _bucket_row(bucket_key: str, rows: list[dict[str, Any]], created: datetime) -> dict[str, Any]:
    parts = bucket_key.split("|")
    v5_open_rows = [row for row in rows if row.get("v5_would_open")]
    false_blocks = [row for row in rows if row.get("false_block")]
    loss_saved = [row for row in rows if row.get("loss_saved")]
    false_allows = [row for row in rows if row.get("false_allow")]
    correct_allows = [row for row in rows if row.get("correct_allow")]
    missed = sum(_float(row.get("missed_profit_bps")) or 0.0 for row in false_blocks)
    saved = sum(_float(row.get("loss_saved_bps")) or 0.0 for row in loss_saved)
    veto_net = saved - missed
    high_conf_false_blocks = sum(1 for row in false_blocks if row.get("high_confidence_v5"))
    exception = high_conf_false_blocks >= 3 and missed > saved
    return {
        "schema_version": OPPORTUNITY_COST_BY_BUCKET_SCHEMA_VERSION,
        "bucket_key": bucket_key,
        "symbol": _part(parts, 0),
        "strategy_candidate": _part(parts, 1),
        "regime": _part(parts, 2),
        "risk_level": _part(parts, 3),
        "rank_bucket": _part(parts, 4),
        "alpha6_bucket": _part(parts, 5),
        "expected_edge_ratio_bucket": _part(parts, 6),
        "cost_gate_bucket": _part(parts, 7),
        "sample_count": len(v5_open_rows),
        "false_block_count": len(false_blocks),
        "loss_saved_count": len(loss_saved),
        "false_allow_count": len(false_allows),
        "correct_allow_count": len(correct_allows),
        "missed_profit_bps_sum": missed,
        "loss_saved_bps_sum": saved,
        "veto_net_value_bps": veto_net,
        "high_confidence_false_block_count": high_conf_false_blocks,
        "opportunity_exception_candidate": exception,
        "recommended_trade_level_decision": "MICRO_CANARY_REVIEW" if exception else "",
        "created_at": created,
        "source": "quant_lab.opportunity_cost.bucket",
    }


def _bucket_fields(event: dict[str, Any]) -> dict[str, str]:
    parts = [
        _text(event.get("symbol")) or "UNKNOWN_SYMBOL",
        _text(event.get("strategy_candidate")) or "UNKNOWN_STRATEGY",
        _text(event.get("regime")) or "UNKNOWN_REGIME",
        _text(event.get("risk_level")) or "UNKNOWN_RISK",
        _rank_bucket(event),
        _alpha_bucket(event),
        _edge_ratio_bucket(event),
        "cost_gate_verified" if _bool(event.get("cost_gate_verified")) else "cost_gate_unverified",
    ]
    return {"bucket_key": "|".join(parts)}


def _preferred_label_value(label: dict[str, Any]) -> tuple[float | None, int | None]:
    for horizon, field in [
        (24, "label_24h_after_cost_bps"),
        (8, "label_8h_after_cost_bps"),
        (4, "label_4h_after_cost_bps"),
    ]:
        value = _float(label.get(field))
        if value is not None:
            return value, horizon
    return None, None


def _best_hindsight_action(after_cost: float | None) -> str:
    if after_cost is None:
        return "UNKNOWN"
    return "ALLOW" if after_cost > 0.0 else "BLOCK"


def _regret_bps(
    *,
    after_cost: float | None,
    false_block: bool,
    false_allow: bool,
) -> float:
    if after_cost is None:
        return 0.0
    if false_block:
        return after_cost
    if false_allow:
        return abs(after_cost)
    return 0.0


def _regret_type(
    *,
    false_block: bool,
    loss_saved: bool,
    false_allow: bool,
    correct_allow: bool,
    after_cost: float | None,
) -> str:
    if after_cost is None:
        return "pending_label"
    if false_block:
        return "false_block"
    if loss_saved:
        return "loss_saved"
    if false_allow:
        return "false_allow"
    if correct_allow:
        return "correct_allow"
    return "not_v5_open"


def _opportunity_cost_status(veto_net: float, blocked: list[dict[str, Any]]) -> str:
    if not blocked:
        return "NO_BLOCKED_OPPORTUNITIES"
    if veto_net > 0.0:
        return "VETO_VALUE_POSITIVE"
    if veto_net < 0.0:
        return "VETO_VALUE_NEGATIVE_REVIEW_EXCEPTIONS"
    return "VETO_VALUE_FLAT"


def _rank_bucket(row: dict[str, Any]) -> str:
    rank = _int(row.get("rank"))
    if rank == 1:
        return "rank_1"
    if rank is not None and rank <= 3:
        return "rank_2_3"
    return "rank_other"


def _alpha_bucket(row: dict[str, Any]) -> str:
    score = _float(row.get("alpha6_score"))
    if score is None:
        return "alpha_missing"
    if score >= 0.95:
        return "alpha_ge_0_95"
    if score >= 0.85:
        return "alpha_ge_0_85"
    return "alpha_lt_0_85"


def _edge_ratio_bucket(row: dict[str, Any]) -> str:
    ratio = _float(row.get("edge_required_ratio"))
    if ratio is None:
        return "edge_ratio_missing"
    if ratio >= 3.0:
        return "edge_ratio_ge_3"
    if ratio >= 1.5:
        return "edge_ratio_ge_1_5"
    return "edge_ratio_lt_1_5"


def _part(parts: list[str], index: int) -> str:
    return parts[index] if index < len(parts) else ""


def _positive(value: float | None) -> bool:
    return value is not None and value > 0.0


def _negative(value: float | None) -> bool:
    return value is not None and value < 0.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row").select(
        [pl.col(name).cast(dtype, strict=False).alias(name) for name, dtype in schema.items()]
    )


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value in (None, ""):
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


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


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None
