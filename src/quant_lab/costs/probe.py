from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import polars as pl

from quant_lab.symbols import normalize_symbol


def canonical_cost_probe_roundtrip_events(roundtrip_events: pl.DataFrame | None) -> pl.DataFrame:
    """Return one final/canonical row per cost-probe roundtrip key."""

    if roundtrip_events is None or roundtrip_events.is_empty():
        return pl.DataFrame()

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in roundtrip_events.to_dicts():
        payload = _payload_dict(row)
        key = _roundtrip_group_key(row, payload)
        grouped.setdefault(key, []).append(row)

    canonical_rows: list[dict[str, Any]] = []
    for rows in grouped.values():
        ordered = sorted(rows, key=_canonical_sort_key)
        selected = dict(ordered[-1])
        selected_payload = _payload_dict(selected)
        selected_event_id = str(
            _first_value(selected, selected_payload, ["event_id"]) or ""
        ).strip()
        superseded = [
            str(_first_value(row, _payload_dict(row), ["event_id"]) or "").strip()
            for row in ordered[:-1]
        ]
        selected["revision"] = len(ordered)
        selected["supersedes_event_id"] = ";".join(item for item in superseded if item)
        selected["terminal"] = _roundtrip_terminal(selected, selected_payload)
        selected["canonical"] = True
        selected["canonical_priority"] = _roundtrip_priority(selected, selected_payload)
        if selected_event_id:
            selected["event_id"] = selected_event_id
        canonical_rows.append(selected)

    if not canonical_rows:
        return pl.DataFrame()
    return pl.DataFrame(canonical_rows, infer_schema_length=None)


def cost_probe_terminal_fill_count_by_symbol(
    order_events: pl.DataFrame | None,
    roundtrip_events: pl.DataFrame | None,
) -> dict[str, int]:
    """Count entry/exit fills from canonical, model-eligible cost-probe roundtrips."""

    canonical_roundtrips = canonical_cost_probe_roundtrip_events(roundtrip_events)
    if canonical_roundtrips.is_empty():
        return {}

    order_rows = (
        order_events.to_dicts()
        if order_events is not None and not order_events.is_empty()
        else []
    )
    counts: dict[str, int] = {}
    for row in canonical_roundtrips.to_dicts():
        payload = _payload_dict(row)
        if not eligible_cost_probe_roundtrip(row, payload):
            continue
        symbol = cost_probe_symbol(row, payload)
        if not symbol:
            continue
        fill_count = 0
        for leg in ("entry", "exit"):
            matched = _latest_matching_order(order_rows, row, payload, leg=leg)
            if matched is not None:
                matched_payload = _payload_dict(matched)
                if cost_probe_order_is_filled(matched, matched_payload):
                    fill_count += 1
                    continue
            if _roundtrip_leg_has_fill(row, payload, leg=leg):
                fill_count += 1
        if fill_count == 0:
            fill_count = 2
        counts[symbol] = counts.get(symbol, 0) + fill_count
    return counts


def cost_probe_private_fill_keys(
    order_events: pl.DataFrame | None,
    roundtrip_events: pl.DataFrame | None,
) -> tuple[set[str], set[str]]:
    order_ids: set[str] = set()
    trade_ids: set[str] = set()
    if order_events is not None and not order_events.is_empty():
        for row in order_events.to_dicts():
            payload = _payload_dict(row)
            order_ids.update(cost_probe_order_identifiers(row, payload))
            trade_ids.update(cost_probe_trade_identifiers(row, payload))
    canonical_roundtrips = canonical_cost_probe_roundtrip_events(roundtrip_events)
    for row in canonical_roundtrips.to_dicts() if not canonical_roundtrips.is_empty() else []:
        payload = _payload_dict(row)
        for key in ("entry_order_id", "exit_order_id", "order_id", "exchange_order_id"):
            value = str(_first_value(row, payload, [key]) or "").strip()
            if value:
                order_ids.add(value)
        trade_ids.update(cost_probe_trade_identifiers(row, payload))
        for state_key in ("entry_state", "exit_state"):
            state = payload.get(state_key)
            if isinstance(state, dict):
                order_ids.update(cost_probe_order_identifiers(state, state))
                trade_ids.update(cost_probe_trade_identifiers(state, state))
    return order_ids, trade_ids


def cost_probe_private_fill_count_by_symbol(
    private_fills: pl.DataFrame | None,
    order_events: pl.DataFrame | None,
    roundtrip_events: pl.DataFrame | None,
) -> dict[str, int]:
    if private_fills is None or private_fills.is_empty():
        return {}
    order_ids, trade_ids = cost_probe_private_fill_keys(order_events, roundtrip_events)
    if not order_ids and not trade_ids:
        return {}
    counts: dict[str, int] = {}
    for row in private_fills.to_dicts():
        if not private_fill_matches_cost_probe(row, order_ids=order_ids, trade_ids=trade_ids):
            continue
        symbol = row_symbol(row)
        if symbol:
            counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def private_fill_matches_cost_probe(
    row: Mapping[str, Any],
    *,
    order_ids: set[str],
    trade_ids: set[str],
) -> bool:
    payload = _payload_dict(row)
    row_order_ids = cost_probe_order_identifiers(row, payload)
    row_trade_ids = cost_probe_trade_identifiers(row, payload)
    return bool(row_order_ids.intersection(order_ids) or row_trade_ids.intersection(trade_ids))


def row_symbol(row: Mapping[str, Any]) -> str:
    payload = _payload_dict(row)
    return cost_probe_symbol(row, payload)


def eligible_cost_probe_roundtrip(
    row: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> bool:
    payload = payload or _payload_dict(row)
    status = str(
        _first_value(row, payload, ["roundtrip_status", "status", "state"]) or ""
    ).strip().lower()
    if status not in {"closed", "closed_flat"}:
        return False
    if _probe_bool(row, payload, ["no_order_submitted"]) is True:
        return False
    required = (
        ("execution_completed", "completed"),
        ("flat_verified",),
        ("exchange_flat_verified",),
        ("local_flat_verified",),
        ("reconcile_ok",),
        ("cost_evidence_complete",),
        ("eligible_for_cost_model",),
    )
    return all(_probe_bool(row, payload, keys) is True for keys in required)


def cost_probe_order_is_filled(
    row: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> bool:
    payload = payload or _payload_dict(row)
    status = str(
        _first_value(row, payload, ["order_status", "status", "state"]) or ""
    ).strip().lower()
    qty = _first_float(row, payload, ["filled_qty", "fill_qty", "fillSz", "accFillSz"])
    price = _first_float(row, payload, ["avg_px", "avgPx", "fill_px", "fillPx"])
    return (
        status in {"filled", "partially_filled", "partial_fill", "partially-filled"}
        and qty is not None
        and qty > 0
        and price is not None
        and price > 0
    )


def cost_probe_symbol(row: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    for key in ("normalized_symbol", "symbol", "inst_id", "instId", "instrument", "pair"):
        value = _first_value(row, payload, [key])
        if value:
            return normalize_symbol(value)
    return ""


def cost_probe_order_identifiers(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> set[str]:
    ids: set[str] = set()
    for key in (
        "order_id",
        "ordId",
        "exchange_order_id",
        "client_order_id",
        "clOrdId",
        "cl_ord_id",
        "order_key",
    ):
        value = str(_first_value(row, payload, [key]) or "").strip()
        if value:
            ids.add(value)
    for fill in _probe_fill_items(payload):
        for key in ("ordId", "order_id", "orderId"):
            value = str(fill.get(key) or "").strip()
            if value:
                ids.add(value)
    return ids


def cost_probe_trade_identifiers(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> set[str]:
    ids: set[str] = set()
    for key in ("trade_id", "tradeId", "trade_ids"):
        value = str(_first_value(row, payload, [key]) or "").strip()
        if value:
            ids.update(part.strip() for part in value.replace(",", ";").split(";") if part.strip())
    for fill in _probe_fill_items(payload):
        for key in ("tradeId", "trade_id"):
            value = str(fill.get(key) or "").strip()
            if value:
                ids.add(value)
    return ids


def _roundtrip_group_key(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    roundtrip_id = str(
        _first_value(row, payload, ["roundtrip_id", "roundtrip_key"]) or ""
    ).strip()
    authorization_id = str(_first_value(row, payload, ["authorization_id"]) or "").strip()
    entry_order_id = str(_first_value(row, payload, ["entry_order_id"]) or "").strip()
    exit_order_id = str(_first_value(row, payload, ["exit_order_id"]) or "").strip()
    if not roundtrip_id and (entry_order_id or exit_order_id):
        roundtrip_id = f"{entry_order_id}:{exit_order_id}"
    if not any((roundtrip_id, authorization_id, entry_order_id, exit_order_id)):
        roundtrip_id = str(_first_value(row, payload, ["event_id"]) or id(row)).strip()
    return roundtrip_id, authorization_id, entry_order_id, exit_order_id


def _canonical_sort_key(row: Mapping[str, Any]) -> tuple[int, datetime, str]:
    payload = _payload_dict(row)
    timestamp = _first_timestamp(
        row,
        payload,
        ("event_ts", "closed_at", "generated_at", "bundle_ts", "ingest_ts", "ts", "timestamp"),
    )
    return (
        _roundtrip_priority(row, payload),
        timestamp or datetime.min.replace(tzinfo=UTC),
        str(_first_value(row, payload, ["event_id"]) or ""),
    )


def _roundtrip_priority(row: Mapping[str, Any], payload: Mapping[str, Any]) -> int:
    status = str(
        _first_value(row, payload, ["roundtrip_status", "status", "state"]) or ""
    ).strip().lower()
    state = str(_first_value(row, payload, ["source_state"]) or "").strip().lower()
    event_type = str(_first_value(row, payload, ["event_type"]) or "").strip().lower()
    if eligible_cost_probe_roundtrip(row, payload) or state == "closed_flat":
        return 100
    if status in {"closed", "closed_flat"} or "roundtrip:closed" in event_type:
        return 90
    if "kill" in status or "kill" in state:
        return 70
    if "recovery" in status or "recovery" in state:
        return 60
    if "incomplete" in status or "incomplete" in state or "incomplete" in event_type:
        return 40
    return 10


def _roundtrip_terminal(row: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    priority = _roundtrip_priority(row, payload)
    return priority >= 60


def _latest_matching_order(
    order_rows: Sequence[dict[str, Any]],
    roundtrip_row: Mapping[str, Any],
    roundtrip_payload: Mapping[str, Any],
    *,
    leg: str,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    leg_order_ids = _roundtrip_leg_order_ids(roundtrip_row, roundtrip_payload, leg=leg)
    roundtrip_authorization = str(
        _first_value(roundtrip_row, roundtrip_payload, ["authorization_id"]) or ""
    ).strip()
    for order_row in order_rows:
        order_payload = _payload_dict(order_row)
        if not cost_probe_order_is_filled(order_row, order_payload):
            continue
        order_leg = str(_first_value(order_row, order_payload, ["leg"]) or "").lower()
        if leg == "entry" and order_leg not in {"entry", "buy", "open", "open_long"}:
            continue
        if leg == "exit" and order_leg not in {"exit", "sell", "close", "close_long"}:
            continue
        order_ids = cost_probe_order_identifiers(order_row, order_payload)
        if leg_order_ids and order_ids.intersection(leg_order_ids):
            matches.append(order_row)
            continue
        order_authorization = str(
            _first_value(order_row, order_payload, ["authorization_id"]) or ""
        ).strip()
        if roundtrip_authorization and order_authorization == roundtrip_authorization:
            matches.append(order_row)
    if not matches:
        return None
    return sorted(matches, key=lambda item: str(item.get("event_ts") or ""))[-1]


def _roundtrip_leg_order_ids(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    leg: str,
) -> set[str]:
    ids: set[str] = set()
    key = "entry_order_id" if leg == "entry" else "exit_order_id"
    value = str(_first_value(row, payload, [key]) or "").strip()
    if value:
        ids.add(value)
    roundtrip_id = str(_first_value(row, payload, ["roundtrip_id"]) or "").strip()
    if ":" in roundtrip_id:
        entry_id, exit_id = roundtrip_id.split(":", 1)
        ids.add(entry_id if leg == "entry" else exit_id)
    return {item for item in ids if item}


def _roundtrip_leg_has_fill(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    leg: str,
) -> bool:
    qty = _first_float(
        row,
        payload,
        [f"{leg}_filled_qty", f"{leg}_fill_qty", f"{leg}_qty"],
    )
    if qty is not None and qty > 0:
        return True
    state = payload.get(f"{leg}_state")
    if isinstance(state, dict):
        return cost_probe_order_is_filled(state, state)
    return False


def _payload_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("raw_payload_json", "raw_json"):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def _first_value(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    sources: list[Mapping[str, Any]] = [row, payload]
    for nested in ("raw", "flat_verification"):
        value = payload.get(nested)
        if isinstance(value, dict):
            sources.append(value)
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None and str(value).strip() != "":
                return value
    return None


def _first_float(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> float | None:
    value = _first_value(row, payload, keys)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _probe_bool(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> bool | None:
    value = _first_value(row, payload, keys)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    rendered = str(value).strip().lower()
    if rendered in {"1", "true", "yes", "y", "on"}:
        return True
    if rendered in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _probe_fill_items(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    for source in (payload, payload.get("raw") if isinstance(payload.get("raw"), dict) else {}):
        fills = source.get("_fills") if isinstance(source, dict) else None
        if isinstance(fills, list):
            items.extend(item for item in fills if isinstance(item, dict))
    data = payload.get("data")
    if isinstance(data, list):
        items.extend(item for item in data if isinstance(item, dict))
    return items


def _first_timestamp(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> datetime | None:
    for key in keys:
        value = _first_value(row, payload, [key])
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        if value in {None, "", "null", "None"}:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None
