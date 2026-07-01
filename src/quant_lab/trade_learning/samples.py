from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

V5_TRADE_LEARNING_SAMPLE_SCHEMA_VERSION = "v5_trade_learning_sample.v0.2"
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
    "actual_exit_ts": pl.Datetime(time_zone="UTC"),
    "actual_exit_reason": pl.Utf8,
    "actual_hold_minutes": pl.Float64,
    "actual_roundtrip_net_bps": pl.Float64,
    "actual_roundtrip_net_pnl_usdt": pl.Float64,
    "actual_outcome_label": pl.Utf8,
    "exit_reason": pl.Utf8,
    "hold_minutes": pl.Float64,
    "net_bps": pl.Float64,
    "net_pnl_usdt": pl.Float64,
    "outcome_label": pl.Utf8,
    "learning_eligible": pl.Boolean,
    "quant_lab_false_block_candidate": pl.Boolean,
    "feature_as_of_ts": pl.Datetime(time_zone="UTC"),
    "label_4h_after_cost_bps": pl.Float64,
    "label_8h_after_cost_bps": pl.Float64,
    "label_24h_after_cost_bps": pl.Float64,
    "fixed_horizon_net_bps": pl.Float64,
    "fixed_horizon_outcome_label": pl.Utf8,
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
    lifecycle_frame = order_lifecycles if order_lifecycles is not None else pl.DataFrame()
    fill_lookup = _fill_lookup(
        v5_trades if v5_trades is not None else pl.DataFrame(),
        lifecycle_frame,
    )
    lifecycle_lookup = _lifecycle_lookup(
        lifecycle_frame,
    )
    rows: list[dict[str, Any]] = []
    for event in events.to_dicts():
        event_id = _text(event.get("event_id"))
        label = labels_by_event.get(event_id, {})
        judgment = judgments_by_event.get(event_id, {})
        label_values = _label_values(label)
        fixed_value, horizon = _preferred_label_value(label)
        decision_ts = _timestamp(event.get("decision_ts"))
        actual_submitted = _bool(event.get("actual_submitted")) is True
        quant_lab_would_block = _quant_lab_would_block(judgment)
        fill = fill_lookup.get((_text(event.get("run_id")), _text(event.get("symbol"))), {})
        lifecycle = lifecycle_lookup.get(
            (_text(event.get("run_id")), _text(event.get("symbol"))), {}
        )
        actual = _actual_roundtrip_outcome(fill, lifecycle, decision_ts)
        actual_net_bps = _float(actual.get("actual_roundtrip_net_bps"))
        actual_net_pnl = _float(actual.get("actual_roundtrip_net_pnl_usdt"))
        primary_value = actual_net_bps if actual_submitted else fixed_value
        primary_pnl = actual_net_pnl if actual_submitted else None
        sample_type = _sample_type(actual_submitted=actual_submitted, net_bps=primary_value)
        label_end_ts = (
            decision_ts + timedelta(hours=horizon)
            if decision_ts is not None and horizon is not None
            else None
        )
        learning_eligible = (
            bool(event.get("v5_would_open"))
            and primary_value is not None
            and decision_ts is not None
            and bool(event.get("symbol"))
            and bool(event.get("strategy_candidate"))
        )
        false_block_candidate = (
            bool(event.get("v5_would_open"))
            and quant_lab_would_block
            and primary_value is not None
            and primary_value > 0.0
        )
        actual_exit_reason = _text(
            actual.get("actual_exit_reason")
            or lifecycle.get("exit_reason")
            or fill.get("exit_reason")
        )
        actual_hold_minutes = _float(actual.get("actual_hold_minutes"))
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
                    fill.get("realized_total_cost_bps"),
                    fill.get("actual_all_in_bps"),
                    event.get("selected_cost_bps"),
                ),
                "actual_exit_ts": _timestamp(actual.get("actual_exit_ts")),
                "actual_exit_reason": actual_exit_reason,
                "actual_hold_minutes": actual_hold_minutes,
                "actual_roundtrip_net_bps": actual_net_bps,
                "actual_roundtrip_net_pnl_usdt": actual_net_pnl,
                "actual_outcome_label": _outcome_label(actual_net_bps),
                "exit_reason": actual_exit_reason,
                "hold_minutes": (
                    actual_hold_minutes
                    if actual_submitted and actual_hold_minutes is not None
                    else float(horizon * 60)
                    if horizon is not None
                    else None
                ),
                "net_bps": primary_value,
                "net_pnl_usdt": primary_pnl,
                "outcome_label": _outcome_label(primary_value),
                "learning_eligible": learning_eligible,
                "quant_lab_false_block_candidate": false_block_candidate,
                "feature_as_of_ts": decision_ts,
                "label_4h_after_cost_bps": label_values.get(4),
                "label_8h_after_cost_bps": label_values.get(8),
                "label_24h_after_cost_bps": label_values.get(24),
                "fixed_horizon_net_bps": fixed_value,
                "fixed_horizon_outcome_label": _outcome_label(fixed_value),
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


def _fill_lookup(
    frame: pl.DataFrame,
    lifecycles: pl.DataFrame | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    lifecycle_exits = _lifecycle_exit_lookup(
        lifecycles if lifecycles is not None else pl.DataFrame()
    )
    if frame.is_empty() and not lifecycle_exits:
        return lookup
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in frame.to_dicts():
        key = (_text(row.get("run_id")), _text(row.get("symbol") or row.get("normalized_symbol")))
        if not all(key):
            continue
        grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        lookup[key] = _merge_exit_details(_run_fill_summary(rows), lifecycle_exits.get(key, {}))
    for key, exit_details in lifecycle_exits.items():
        lookup.setdefault(key, _merge_exit_details({}, exit_details))
    _pair_cross_run_exits(lookup)
    return lookup


def _run_fill_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: _row_ts(row) or datetime.min.replace(tzinfo=UTC))
    entry_rows = [row for row in ordered if _is_entry_row(row)]
    exit_rows = [row for row in ordered if _is_exit_row(row)]
    entry = entry_rows[0] if entry_rows else {}
    exit_row = exit_rows[-1] if exit_rows else {}
    exit_payload = _payload(exit_row)
    return {
        "entry_ts": _min_ts(entry_rows),
        "exit_ts": _max_ts(exit_rows),
        "entry_side": _text(entry.get("side") or _payload(entry).get("side")),
        "entry_price": _weighted_price(entry_rows),
        "exit_price": _weighted_price(exit_rows),
        "entry_qty": _sum_rows(entry_rows, "qty", "filled_qty"),
        "exit_qty": _sum_rows(exit_rows, "qty", "filled_qty"),
        "entry_notional_usdt": _sum_notional(entry_rows),
        "exit_notional_usdt": _sum_notional(exit_rows),
        "entry_fee_usdt": _sum_fee_usdt(entry_rows),
        "exit_fee_usdt": _sum_fee_usdt(exit_rows),
        "exit_reason": _payload_text(exit_row, "exit_reason", "reason", "source_reason"),
        "actual_roundtrip_net_bps": _first_float(
            exit_row.get("actual_roundtrip_net_bps"),
            exit_row.get("roundtrip_net_bps"),
            exit_row.get("realized_net_bps"),
            exit_row.get("net_bps"),
            exit_payload.get("actual_roundtrip_net_bps"),
            exit_payload.get("roundtrip_net_bps"),
            exit_payload.get("realized_net_bps"),
            exit_payload.get("net_bps"),
        ),
        "realized_total_cost_bps": _first_float(
            exit_row.get("realized_total_cost_bps"),
            exit_row.get("total_realized_cost_bps"),
            exit_payload.get("realized_total_cost_bps"),
            exit_payload.get("total_realized_cost_bps"),
        ),
        "net_pnl_usdt": _first_float(
            exit_row.get("net_pnl_usdt"),
            exit_row.get("pnl_usdt"),
            exit_row.get("realized_pnl_usdt"),
            exit_payload.get("net_pnl_usdt"),
            exit_payload.get("pnl_usdt"),
            exit_payload.get("realized_pnl_usdt"),
        ),
    }


def _lifecycle_exit_lookup(frame: pl.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if frame.is_empty():
        return lookup
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in frame.to_dicts():
        if not _is_exit_row(row):
            continue
        key = (_text(row.get("run_id")), _text(row.get("symbol") or row.get("normalized_symbol")))
        if not all(key):
            continue
        grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: _row_ts(row) or datetime.min.replace(tzinfo=UTC))
        latest = ordered[-1]
        payload = _payload(latest)
        lookup[key] = {
            "exit_ts": _timestamp(
                latest.get("last_fill_ts")
                or latest.get("first_fill_ts")
                or latest.get("ts_utc")
                or payload.get("last_fill_ts")
                or payload.get("first_fill_ts")
                or payload.get("ts_utc")
            ),
            "exit_price": _weighted_price(ordered),
            "exit_qty": _sum_rows(ordered, "qty", "filled_qty"),
            "exit_notional_usdt": _sum_notional(ordered),
            "exit_fee_usdt": _sum_fee_usdt(ordered),
            "exit_reason": _text(
                latest.get("exit_reason")
                or latest.get("source_reason")
                or payload.get("exit_reason")
                or payload.get("source_reason")
                or payload.get("reason")
            ),
            "realized_total_cost_bps": _first_float(
                latest.get("realized_total_cost_bps"),
                latest.get("total_realized_cost_bps"),
                payload.get("realized_total_cost_bps"),
                payload.get("total_realized_cost_bps"),
            ),
        }
    return lookup


def _merge_exit_details(base: dict[str, Any], exit_details: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for field in [
        "exit_ts",
        "exit_price",
        "exit_qty",
        "exit_notional_usdt",
        "exit_fee_usdt",
        "exit_reason",
        "realized_total_cost_bps",
        "actual_roundtrip_net_bps",
        "net_pnl_usdt",
    ]:
        if merged.get(field) in (None, "") and exit_details.get(field) not in (None, ""):
            merged[field] = exit_details.get(field)
    return merged


def _pair_cross_run_exits(lookup: dict[tuple[str, str], dict[str, Any]]) -> None:
    exit_candidates_by_symbol: dict[str, list[tuple[tuple[str, str], dict[str, Any]]]] = {}
    for key, row in lookup.items():
        if _timestamp(row.get("exit_ts")) is None or _float(row.get("exit_price")) is None:
            continue
        exit_candidates_by_symbol.setdefault(key[1], []).append((key, row))
    used: set[tuple[str, str]] = set()
    for key, row in sorted(
        lookup.items(),
        key=lambda item: _timestamp(item[1].get("entry_ts")) or datetime.max.replace(tzinfo=UTC),
    ):
        if _timestamp(row.get("entry_ts")) is None or _float(row.get("entry_price")) is None:
            continue
        if _timestamp(row.get("exit_ts")) is not None and _float(row.get("exit_price")) is not None:
            continue
        candidate = _matching_exit_candidate(
            row,
            exit_candidates_by_symbol.get(key[1], []),
            used,
        )
        if candidate is None:
            continue
        exit_key, exit_row = candidate
        used.add(exit_key)
        lookup[key] = _merge_exit_details(row, exit_row)


def _matching_exit_candidate(
    entry: dict[str, Any],
    candidates: list[tuple[tuple[str, str], dict[str, Any]]],
    used: set[tuple[str, str]],
) -> tuple[tuple[str, str], dict[str, Any]] | None:
    entry_ts = _timestamp(entry.get("entry_ts"))
    if entry_ts is None:
        return None
    viable = []
    for key, candidate in candidates:
        if key in used:
            continue
        exit_ts = _timestamp(candidate.get("exit_ts"))
        if exit_ts is None or exit_ts <= entry_ts:
            continue
        if not _similar_size(entry, candidate):
            continue
        viable.append((exit_ts, key, candidate))
    if not viable:
        return None
    _, key, candidate = min(viable, key=lambda item: item[0])
    return key, candidate


def _similar_size(entry: dict[str, Any], exit_row: dict[str, Any]) -> bool:
    entry_qty = _float(entry.get("entry_qty"))
    exit_qty = _float(exit_row.get("exit_qty"))
    if entry_qty not in (None, 0.0) and exit_qty is not None:
        return abs(entry_qty - exit_qty) / entry_qty <= 0.05
    entry_notional = _float(entry.get("entry_notional_usdt"))
    exit_notional = _float(exit_row.get("exit_notional_usdt"))
    if entry_notional not in (None, 0.0) and exit_notional is not None:
        return abs(entry_notional - exit_notional) / entry_notional <= 0.15
    return True


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
        is_exit = _is_exit_row(row)
        lookup[key] = {
            **existing,
            "realized_total_cost_bps": _first_float(
                row.get("realized_total_cost_bps"),
                row.get("total_realized_cost_bps"),
                payload.get("realized_total_cost_bps"),
                payload.get("total_realized_cost_bps"),
                existing.get("realized_total_cost_bps"),
            ),
            "actual_roundtrip_net_bps": _first_float(
                row.get("actual_roundtrip_net_bps") if is_exit else None,
                row.get("roundtrip_net_bps") if is_exit else None,
                row.get("realized_net_bps") if is_exit else None,
                row.get("net_bps") if is_exit else None,
                payload.get("actual_roundtrip_net_bps") if is_exit else None,
                payload.get("roundtrip_net_bps") if is_exit else None,
                payload.get("realized_net_bps") if is_exit else None,
                payload.get("net_bps") if is_exit else None,
                existing.get("actual_roundtrip_net_bps"),
            ),
            "actual_roundtrip_net_pnl_usdt": _first_float(
                row.get("actual_roundtrip_net_pnl_usdt") if is_exit else None,
                row.get("net_pnl_usdt") if is_exit else None,
                row.get("pnl_usdt") if is_exit else None,
                row.get("realized_pnl_usdt") if is_exit else None,
                payload.get("actual_roundtrip_net_pnl_usdt") if is_exit else None,
                payload.get("net_pnl_usdt") if is_exit else None,
                payload.get("pnl_usdt") if is_exit else None,
                payload.get("realized_pnl_usdt") if is_exit else None,
                existing.get("actual_roundtrip_net_pnl_usdt"),
            ),
            "exit_ts": _timestamp(
                (row.get("exit_ts") if is_exit else None)
                or (row.get("ts_utc") if is_exit else None)
                or (row.get("ts") if is_exit else None)
                or (payload.get("exit_ts") if is_exit else None)
                or (payload.get("ts_utc") if is_exit else None)
                or existing.get("exit_ts")
            ),
            "exit_reason": _text(
                (row.get("exit_reason") if is_exit else None)
                or (payload.get("exit_reason") if is_exit else None)
                or (payload.get("reason") if is_exit else None)
                or existing.get("exit_reason")
            ),
        }
    return lookup


def _is_entry_row(row: dict[str, Any]) -> bool:
    payload = _payload(row)
    side = _text(row.get("side") or payload.get("side")).lower()
    action = _text(
        row.get("action")
        or row.get("intent")
        or row.get("event_type")
        or payload.get("intent")
    ).lower()
    return side == "buy" or any(token in action for token in ("entry", "open_long", "open"))


def _is_exit_row(row: dict[str, Any]) -> bool:
    payload = _payload(row)
    side = _text(row.get("side") or payload.get("side")).lower()
    action = _text(
        row.get("action")
        or row.get("intent")
        or row.get("event_type")
        or payload.get("intent")
        or payload.get("action")
    ).lower()
    exit_reason = _text(row.get("exit_reason") or payload.get("exit_reason"))
    return (
        side == "sell"
        or any(token in action for token in ("exit", "close", "sell"))
        or bool(exit_reason)
    )


def _row_ts(row: dict[str, Any]) -> datetime | None:
    payload = _payload(row)
    return _timestamp(
        row.get("ts_utc")
        or row.get("ts")
        or row.get("timestamp")
        or row.get("last_fill_ts")
        or row.get("first_fill_ts")
        or payload.get("ts_utc")
        or payload.get("ts")
        or payload.get("last_fill_ts")
        or payload.get("first_fill_ts")
    )


def _min_ts(rows: list[dict[str, Any]]) -> datetime | None:
    values = [_row_ts(row) for row in rows]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _max_ts(rows: list[dict[str, Any]]) -> datetime | None:
    values = [_row_ts(row) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _weighted_price(rows: list[dict[str, Any]]) -> float | None:
    weighted = 0.0
    qty_sum = 0.0
    prices: list[float] = []
    for row in rows:
        price = _row_price(row)
        if price is None:
            continue
        qty = _first_float(row.get("qty"), row.get("filled_qty"), _payload(row).get("filled_qty"))
        prices.append(price)
        if qty is not None and qty > 0.0:
            weighted += price * qty
            qty_sum += qty
    if qty_sum > 0.0:
        return weighted / qty_sum
    return prices[-1] if prices else None


def _row_price(row: dict[str, Any]) -> float | None:
    payload = _payload(row)
    return _first_float(
        row.get("price"),
        row.get("fill_price"),
        row.get("fill_px"),
        row.get("avg_fill_px"),
        payload.get("price"),
        payload.get("fill_price"),
        payload.get("fill_px"),
        payload.get("avg_fill_px"),
    )


def _sum_rows(rows: list[dict[str, Any]], *fields: str) -> float | None:
    total = 0.0
    found = False
    for row in rows:
        payload = _payload(row)
        row_values = [row.get(field) for field in fields]
        payload_values = [payload.get(field) for field in fields]
        value = _first_float(*row_values, *payload_values)
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def _sum_notional(rows: list[dict[str, Any]]) -> float | None:
    total = 0.0
    found = False
    for row in rows:
        payload = _payload(row)
        value = _first_float(row.get("notional_usdt"), payload.get("notional_usdt"))
        if value is None:
            qty = _first_float(row.get("qty"), row.get("filled_qty"), payload.get("filled_qty"))
            price = _row_price(row)
            value = qty * price if qty is not None and price is not None else None
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def _sum_fee_usdt(rows: list[dict[str, Any]]) -> float | None:
    total = 0.0
    found = False
    for row in rows:
        payload = _payload(row)
        value = _first_float(row.get("fee_usdt"), payload.get("fee_usdt"))
        if value is None:
            continue
        total += abs(value)
        found = True
    return total if found else None


def _actual_roundtrip_outcome(
    fill: dict[str, Any],
    lifecycle: dict[str, Any],
    decision_ts: datetime | None,
) -> dict[str, Any]:
    exit_ts = _timestamp(lifecycle.get("exit_ts") or fill.get("exit_ts"))
    entry_ts = _timestamp(fill.get("entry_ts")) or decision_ts
    cost_bps = _first_float(
        lifecycle.get("realized_total_cost_bps"),
        fill.get("realized_total_cost_bps"),
        fill.get("actual_all_in_bps"),
    )
    net_bps = _first_float(
        lifecycle.get("actual_roundtrip_net_bps"),
        fill.get("actual_roundtrip_net_bps"),
    )
    net_pnl = _first_float(
        lifecycle.get("actual_roundtrip_net_pnl_usdt"),
        fill.get("net_pnl_usdt"),
        _roundtrip_net_pnl_usdt(fill),
    )
    if net_bps is None:
        entry_notional = _float(fill.get("entry_notional_usdt"))
        if net_pnl is not None and entry_notional not in (None, 0.0):
            net_bps = (net_pnl / entry_notional) * 10_000.0
        else:
            gross_bps = _gross_roundtrip_bps(fill)
            if gross_bps is not None:
                net_bps = gross_bps - (cost_bps or 0.0)
    hold_minutes = None
    if entry_ts is not None and exit_ts is not None:
        hold_minutes = max(0.0, (exit_ts - entry_ts).total_seconds() / 60.0)
    return {
        "actual_exit_ts": exit_ts,
        "actual_exit_reason": _text(lifecycle.get("exit_reason") or fill.get("exit_reason")),
        "actual_hold_minutes": hold_minutes,
        "actual_roundtrip_net_bps": net_bps,
        "actual_roundtrip_net_pnl_usdt": net_pnl,
    }


def _roundtrip_net_pnl_usdt(fill: dict[str, Any]) -> float | None:
    entry_notional = _float(fill.get("entry_notional_usdt"))
    exit_notional = _float(fill.get("exit_notional_usdt"))
    if entry_notional is None or exit_notional is None:
        return None
    entry_fee = _float(fill.get("entry_fee_usdt")) or 0.0
    exit_fee = _float(fill.get("exit_fee_usdt")) or 0.0
    entry_side = _text(fill.get("entry_side")).lower()
    if entry_side == "sell":
        return entry_notional - exit_notional - entry_fee - exit_fee
    return exit_notional - entry_notional - entry_fee - exit_fee


def _gross_roundtrip_bps(fill: dict[str, Any]) -> float | None:
    entry_price = _float(fill.get("entry_price"))
    exit_price = _float(fill.get("exit_price"))
    if entry_price in (None, 0.0) or exit_price is None:
        return None
    entry_side = _text(fill.get("entry_side")).lower()
    if entry_side == "sell":
        return ((entry_price - exit_price) / entry_price) * 10_000.0
    return ((exit_price - entry_price) / entry_price) * 10_000.0


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


def _label_values(label: dict[str, Any]) -> dict[int, float | None]:
    return {
        4: _float(label.get("label_4h_after_cost_bps")),
        8: _float(label.get("label_8h_after_cost_bps")),
        24: _float(label.get("label_24h_after_cost_bps")),
    }


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
