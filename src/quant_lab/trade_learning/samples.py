from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

V5_TRADE_LEARNING_SAMPLE_SCHEMA_VERSION = "v5_trade_learning_sample.v0.1"
V5_TRADE_LEARNING_SAMPLE_SCHEMA = {
    "schema_version": pl.Utf8,
    "sample_id": pl.Utf8,
    "event_id": pl.Utf8,
    "sample_type": pl.Utf8,
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "run_id": pl.Utf8,
    "symbol": pl.Utf8,
    "side": pl.Utf8,
    "intent": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "v5_final_score": pl.Float64,
    "rank": pl.Int64,
    "alpha6_score": pl.Float64,
    "alpha6_side": pl.Utf8,
    "expected_edge_bps": pl.Float64,
    "required_edge_bps": pl.Float64,
    "edge_required_ratio": pl.Float64,
    "cost_bps": pl.Float64,
    "cost_gate_verified": pl.Boolean,
    "risk_level": pl.Utf8,
    "regime_at_decision": pl.Utf8,
    "arrival_mid": pl.Float64,
    "arrival_spread_bps": pl.Float64,
    "quant_lab_permission_at_decision": pl.Utf8,
    "quant_lab_would_block": pl.Boolean,
    "actual_order_submitted": pl.Boolean,
    "actual_fill_px": pl.Float64,
    "actual_all_in_bps": pl.Float64,
    "exit_reason": pl.Utf8,
    "hold_minutes": pl.Float64,
    "net_bps": pl.Float64,
    "net_pnl_usdt": pl.Float64,
    "outcome_label": pl.Utf8,
    "learning_eligible": pl.Boolean,
    "quant_lab_false_block_candidate": pl.Boolean,
    "feature_as_of_ts": pl.Datetime(time_zone="UTC"),
    "label_end_ts": pl.Datetime(time_zone="UTC"),
    "label_horizon_hours": pl.Int64,
    "cost_model_version_at_decision": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


def build_v5_trade_learning_samples(
    events: pl.DataFrame,
    labels: pl.DataFrame,
    judgments: pl.DataFrame,
    *,
    v5_trades: pl.DataFrame | None = None,
    order_lifecycles: pl.DataFrame | None = None,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    created = created_at or datetime.now(UTC)
    if events.is_empty():
        return pl.DataFrame(schema=V5_TRADE_LEARNING_SAMPLE_SCHEMA)

    labels_by_event = {str(row.get("event_id") or ""): row for row in labels.to_dicts()}
    judgments_by_event = {
        str(row.get("event_id") or ""): row for row in judgments.to_dicts()
    }
    fill_lookup = _fill_lookup(v5_trades if v5_trades is not None else pl.DataFrame())
    lifecycle_lookup = _lifecycle_lookup(
        order_lifecycles if order_lifecycles is not None else pl.DataFrame()
    )
    rows: list[dict[str, Any]] = []
    for event in events.to_dicts():
        event_id = _text(event.get("event_id"))
        label = labels_by_event.get(event_id, {})
        judgment = judgments_by_event.get(event_id, {})
        value, horizon = _preferred_label_value(label)
        decision_ts = _timestamp(event.get("decision_ts"))
        actual_submitted = _bool(event.get("actual_submitted")) is True
        quant_lab_would_block = _quant_lab_would_block(judgment)
        sample_type = _sample_type(actual_submitted=actual_submitted, net_bps=value)
        fill = fill_lookup.get((_text(event.get("run_id")), _text(event.get("symbol"))), {})
        lifecycle = lifecycle_lookup.get(
            (_text(event.get("run_id")), _text(event.get("symbol"))), {}
        )
        label_end_ts = (
            decision_ts + timedelta(hours=horizon)
            if decision_ts is not None and horizon is not None
            else None
        )
        learning_eligible = (
            bool(event.get("v5_would_open"))
            and value is not None
            and decision_ts is not None
            and bool(event.get("symbol"))
            and bool(event.get("strategy_candidate"))
        )
        false_block_candidate = (
            bool(event.get("v5_would_open"))
            and quant_lab_would_block
            and value is not None
            and value > 0.0
        )
        rows.append(
            {
                "schema_version": V5_TRADE_LEARNING_SAMPLE_SCHEMA_VERSION,
                "sample_id": event_id,
                "event_id": event_id,
                "sample_type": sample_type,
                "decision_ts": decision_ts,
                "run_id": _text(event.get("run_id")),
                "symbol": _text(event.get("symbol")),
                "side": _text(event.get("side")),
                "intent": _text(event.get("intent")),
                "strategy_candidate": _text(event.get("strategy_candidate")),
                "v5_final_score": _float(event.get("v5_final_score")),
                "rank": _int(event.get("rank")),
                "alpha6_score": _float(event.get("alpha6_score")),
                "alpha6_side": _text(event.get("alpha6_side")),
                "expected_edge_bps": _float(event.get("expected_edge_bps")),
                "required_edge_bps": _float(event.get("required_edge_bps")),
                "edge_required_ratio": _float(event.get("edge_required_ratio")),
                "cost_bps": _float(event.get("cost_bps")),
                "cost_gate_verified": _bool(event.get("cost_gate_verified")),
                "risk_level": _text(event.get("risk_level")),
                "regime_at_decision": _text(event.get("regime")),
                "arrival_mid": _float(event.get("arrival_mid")),
                "arrival_spread_bps": _float(event.get("arrival_spread_bps")),
                "quant_lab_permission_at_decision": _text(
                    event.get("quant_lab_permission")
                ),
                "quant_lab_would_block": quant_lab_would_block,
                "actual_order_submitted": actual_submitted,
                "actual_fill_px": _float(fill.get("entry_price")),
                "actual_all_in_bps": _first_float(
                    event.get("actual_all_in_cost_bps"),
                    lifecycle.get("realized_total_cost_bps"),
                    event.get("selected_cost_bps"),
                ),
                "exit_reason": _text(lifecycle.get("exit_reason") or fill.get("exit_reason")),
                "hold_minutes": float(horizon * 60) if horizon is not None else None,
                "net_bps": value,
                "net_pnl_usdt": _float(fill.get("net_pnl_usdt")),
                "outcome_label": _outcome_label(value),
                "learning_eligible": learning_eligible,
                "quant_lab_false_block_candidate": false_block_candidate,
                "feature_as_of_ts": decision_ts,
                "label_end_ts": label_end_ts,
                "label_horizon_hours": horizon,
                "cost_model_version_at_decision": _text(
                    event.get("cost_model_version")
                    or event.get("cost_model_version_at_decision")
                ),
                "created_at": created,
                "source": "quant_lab.trade_learning.samples",
            }
        )
    return _frame(rows, V5_TRADE_LEARNING_SAMPLE_SCHEMA)


def _fill_lookup(frame: pl.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if frame.is_empty():
        return lookup
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in frame.to_dicts():
        key = (_text(row.get("run_id")), _text(row.get("symbol") or row.get("normalized_symbol")))
        if not all(key):
            continue
        grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: _timestamp(row.get("ts_utc")) or datetime.min.replace(tzinfo=UTC),
        )
        entry = next(
            (
                row
                for row in ordered
                if _text(row.get("side")).lower() == "buy"
                or "entry" in _text(row.get("action")).lower()
            ),
            ordered[0],
        )
        exit_row = next(
            (
                row
                for row in reversed(ordered)
                if _text(row.get("side")).lower() == "sell"
                or "exit" in _text(row.get("action")).lower()
            ),
            {},
        )
        lookup[key] = {
            "entry_price": _float(entry.get("price") or entry.get("fill_price")),
            "exit_reason": _payload_text(exit_row, "exit_reason", "reason"),
            "net_pnl_usdt": _first_float(
                exit_row.get("net_pnl_usdt"),
                exit_row.get("pnl_usdt"),
                exit_row.get("realized_pnl_usdt"),
            ),
        }
    return lookup


def _lifecycle_lookup(frame: pl.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if frame.is_empty():
        return lookup
    for row in sorted(
        frame.to_dicts(),
        key=lambda item: _timestamp(item.get("ts_utc")) or datetime.min.replace(tzinfo=UTC),
    ):
        key = (_text(row.get("run_id")), _text(row.get("symbol") or row.get("normalized_symbol")))
        if not all(key):
            continue
        payload = _payload(row)
        existing = lookup.get(key, {})
        lookup[key] = {
            **existing,
            "realized_total_cost_bps": _first_float(
                row.get("realized_total_cost_bps"),
                row.get("total_realized_cost_bps"),
                payload.get("realized_total_cost_bps"),
                payload.get("total_realized_cost_bps"),
                existing.get("realized_total_cost_bps"),
            ),
            "exit_reason": _text(
                row.get("exit_reason")
                or payload.get("exit_reason")
                or payload.get("reason")
                or existing.get("exit_reason")
            ),
        }
    return lookup


def _quant_lab_would_block(judgment: dict[str, Any]) -> bool:
    decision = _text(judgment.get("trade_level_decision"))
    return decision not in {"MICRO_CANARY_ALLOW", "LIVE_SMALL_ALLOW"}


def _sample_type(*, actual_submitted: bool, net_bps: float | None) -> str:
    if net_bps is None:
        return "LIVE_PENDING" if actual_submitted else "COUNTERFACTUAL_PENDING"
    if actual_submitted:
        return "LIVE_SUCCESS" if net_bps > 0.0 else "LIVE_FAILURE"
    return "COUNTERFACTUAL_SUCCESS" if net_bps > 0.0 else "COUNTERFACTUAL_FAILURE"


def _outcome_label(net_bps: float | None) -> str:
    if net_bps is None:
        return "PENDING"
    return "PROFITABLE" if net_bps > 0.0 else "UNPROFITABLE"


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


def _payload_text(row: dict[str, Any], *fields: str) -> str:
    payload = _payload(row)
    for field in fields:
        value = _text(row.get(field) or payload.get(field))
        if value:
            return value
    return ""


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("raw_payload_json")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


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


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            return parsed
    return None


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
