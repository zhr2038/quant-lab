from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import polars as pl

from quant_lab.symbols import normalize_symbol

COST_PROBE_FILL_BILL_MATCH_FIELDS = [
    "generated_at",
    "symbol",
    "authorization_id",
    "roundtrip_id",
    "entry_order_id",
    "exit_order_id",
    "entry_trade_id",
    "exit_trade_id",
    "entry_bill_id",
    "exit_bill_id",
    "entry_fee_from_fill",
    "entry_fee_from_bill",
    "exit_fee_from_fill",
    "exit_fee_from_bill",
    "fee_diff_usdt",
    "bill_match_status",
]


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


def canonical_cost_probe_live_execution_status(
    live_execution_status: pl.DataFrame | None,
) -> pl.DataFrame:
    """Return one operator-facing status row per cost-probe authorization."""

    if live_execution_status is None or live_execution_status.is_empty():
        return pl.DataFrame()

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in live_execution_status.to_dicts():
        payload = _payload_dict(row)
        key = _live_status_group_key(row, payload)
        grouped.setdefault(key, []).append(row)

    canonical_rows: list[dict[str, Any]] = []
    for rows in grouped.values():
        ordered = sorted(rows, key=_live_status_sort_key)
        selected = dict(ordered[-1])
        selected_payload = _payload_dict(selected)
        superseded = [
            str(_first_value(row, _payload_dict(row), ["stable_row_key"]) or "").strip()
            for row in ordered[:-1]
        ]
        selected["revision"] = len(ordered)
        selected["supersedes_stable_row_key"] = ";".join(item for item in superseded if item)
        selected["canonical"] = True
        selected["canonical_priority"] = _live_status_priority(selected, selected_payload)
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


def build_cost_probe_fill_bill_match(
    order_events: pl.DataFrame | None,
    roundtrip_events: pl.DataFrame | None,
    private_fills: pl.DataFrame | None,
    private_bills: pl.DataFrame | None,
    *,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    generated_text = generated.isoformat().replace("+00:00", "Z")
    canonical_roundtrips = canonical_cost_probe_roundtrip_events(roundtrip_events)
    if canonical_roundtrips.is_empty():
        return _empty_fill_bill_match_frame()

    order_rows = (
        order_events.to_dicts()
        if order_events is not None and not order_events.is_empty()
        else []
    )
    fill_rows = (
        private_fills.to_dicts()
        if private_fills is not None and not private_fills.is_empty()
        else []
    )
    bill_rows = (
        private_bills.to_dicts()
        if private_bills is not None and not private_bills.is_empty()
        else []
    )
    rows: list[dict[str, Any]] = []

    for roundtrip in canonical_roundtrips.to_dicts():
        payload = _payload_dict(roundtrip)
        if not eligible_cost_probe_roundtrip(roundtrip, payload):
            continue
        symbol = cost_probe_symbol(roundtrip, payload)
        entry = _cost_probe_leg_bill_match(
            leg="entry",
            roundtrip=roundtrip,
            payload=payload,
            order_rows=order_rows,
            fill_rows=fill_rows,
            bill_rows=bill_rows,
            symbol=symbol,
        )
        exit_ = _cost_probe_leg_bill_match(
            leg="exit",
            roundtrip=roundtrip,
            payload=payload,
            order_rows=order_rows,
            fill_rows=fill_rows,
            bill_rows=bill_rows,
            symbol=symbol,
        )
        fee_diff = _fee_diff(
            entry["fee_from_fill"],
            entry["fee_from_bill"],
            exit_["fee_from_fill"],
            exit_["fee_from_bill"],
        )
        authorization_id = str(_first_value(roundtrip, payload, ["authorization_id"]) or "")
        roundtrip_id = str(
            _first_value(roundtrip, payload, ["roundtrip_id", "roundtrip_key"]) or ""
        )
        rows.append(
            {
                "generated_at": generated_text,
                "symbol": symbol,
                "authorization_id": authorization_id,
                "roundtrip_id": roundtrip_id,
                "entry_order_id": ";".join(sorted(entry["order_ids"])),
                "exit_order_id": ";".join(sorted(exit_["order_ids"])),
                "entry_trade_id": ";".join(sorted(entry["trade_ids"])),
                "exit_trade_id": ";".join(sorted(exit_["trade_ids"])),
                "entry_bill_id": ";".join(sorted(entry["bill_ids"])),
                "exit_bill_id": ";".join(sorted(exit_["bill_ids"])),
                "entry_fee_from_fill": _format_number(entry["fee_from_fill"]),
                "entry_fee_from_bill": _format_number(entry["fee_from_bill"]),
                "exit_fee_from_fill": _format_number(exit_["fee_from_fill"]),
                "exit_fee_from_bill": _format_number(exit_["fee_from_bill"]),
                "fee_diff_usdt": _format_number(fee_diff),
                "bill_match_status": _bill_match_status(entry, exit_, fee_diff),
            }
        )

    if not rows:
        return _empty_fill_bill_match_frame()
    return pl.DataFrame(rows, infer_schema_length=None).select(COST_PROBE_FILL_BILL_MATCH_FIELDS)


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


def _live_status_group_key(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    authorization_id = str(_first_value(row, payload, ["authorization_id"]) or "").strip()
    symbol_value = str(_first_value(row, payload, ["manual_probe_symbol", "symbol"]) or "").strip()
    symbol = normalize_symbol(symbol_value) if symbol_value else ""
    if not authorization_id:
        authorization_id = str(_first_value(row, payload, ["stable_row_key"]) or "").strip()
    if not authorization_id:
        authorization_id = str(_first_value(row, payload, ["bundle_sha256"]) or id(row)).strip()
    return authorization_id, symbol


def _live_status_sort_key(row: Mapping[str, Any]) -> tuple[int, datetime, str]:
    payload = _payload_dict(row)
    timestamp = _first_timestamp(
        row,
        payload,
        (
            "generated_at_utc",
            "generated_at",
            "event_ts",
            "authorization_consumed_at",
            "bundle_ts",
            "ingest_ts",
            "ts",
            "timestamp",
        ),
    )
    return (
        _live_status_priority(row, payload),
        timestamp or datetime.min.replace(tzinfo=UTC),
        str(_first_value(row, payload, ["stable_row_key", "event_id"]) or ""),
    )


def _live_status_priority(row: Mapping[str, Any], payload: Mapping[str, Any]) -> int:
    status = str(_first_value(row, payload, ["status", "state"]) or "").strip().upper()
    source_state = str(_first_value(row, payload, ["source_state"]) or "").strip().upper()
    combined = f"{status} {source_state}"
    if (
        "CLOSED_FLAT" in combined
        or (
            _probe_bool(row, payload, ["execution_completed"]) is True
            and _probe_bool(row, payload, ["flat_verified"]) is True
            and _probe_bool(row, payload, ["reconcile_ok"]) is True
        )
    ):
        return 100
    if "INCOMPLETE_KILL_SWITCH" in combined or (
        "INCOMPLETE" in combined and "KILL" in combined
    ):
        return 90
    if "RECOVERY_REQUIRED" in combined or _probe_bool(row, payload, ["recovery_required"]):
        return 80
    if "AUTH_CONSUMED" in combined or _probe_bool(row, payload, ["authorization_consumed"]):
        return 70
    if "AUTH_VALIDATED" in combined or _probe_bool(row, payload, ["authorization_validated"]):
        return 60
    if "PREFLIGHT_READY" in combined or "READY_FOR_MANUAL_AUTHORIZATION" in combined:
        return 50
    return 10


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


def _empty_fill_bill_match_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={field: pl.Utf8 for field in COST_PROBE_FILL_BILL_MATCH_FIELDS})


def _cost_probe_leg_bill_match(
    *,
    leg: str,
    roundtrip: Mapping[str, Any],
    payload: Mapping[str, Any],
    order_rows: Sequence[dict[str, Any]],
    fill_rows: Sequence[dict[str, Any]],
    bill_rows: Sequence[dict[str, Any]],
    symbol: str,
) -> dict[str, Any]:
    order_ids = _roundtrip_leg_order_ids(roundtrip, payload, leg=leg)
    trade_ids = _roundtrip_leg_trade_ids(roundtrip, payload, leg=leg)
    private_matches = _matching_private_fills_for_leg(
        fill_rows,
        symbol=symbol,
        order_ids=order_ids,
        trade_ids=trade_ids,
    )
    if not private_matches:
        matched_order = _latest_matching_order(order_rows, roundtrip, payload, leg=leg)
        if matched_order is not None:
            matched_payload = _payload_dict(matched_order)
            order_ids.update(cost_probe_order_identifiers(matched_order, matched_payload))
            trade_ids.update(cost_probe_trade_identifiers(matched_order, matched_payload))
            private_matches = [matched_order]

    for fill in private_matches:
        fill_payload = _payload_dict(fill)
        order_ids.update(cost_probe_order_identifiers(fill, fill_payload))
        trade_ids.update(cost_probe_trade_identifiers(fill, fill_payload))

    bill_matches = _matching_bill_rows(
        bill_rows,
        fill_rows=private_matches,
        order_ids=order_ids,
        trade_ids=trade_ids,
    )
    return {
        "order_ids": {item for item in order_ids if item},
        "trade_ids": {item for item in trade_ids if item},
        "bill_ids": _bill_ids(bill_matches),
        "fee_from_fill": _sum_optional(_fee_from_fill(row) for row in private_matches),
        "fee_from_bill": _sum_optional(_fee_from_bill(row) for row in bill_matches),
    }


def _roundtrip_leg_trade_ids(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    leg: str,
) -> set[str]:
    ids: set[str] = set()
    for key in (f"{leg}_trade_id", f"{leg}_trade_ids"):
        value = str(_first_value(row, payload, [key]) or "").strip()
        if value:
            ids.update(part.strip() for part in value.replace(",", ";").split(";") if part.strip())
    state = payload.get(f"{leg}_state")
    if isinstance(state, dict):
        ids.update(cost_probe_trade_identifiers(state, state))
    return ids


def _matching_private_fills_for_leg(
    rows: Sequence[dict[str, Any]],
    *,
    symbol: str,
    order_ids: set[str],
    trade_ids: set[str],
) -> list[dict[str, Any]]:
    matches: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        payload = _payload_dict(row)
        row_symbol_value = cost_probe_symbol(row, payload)
        if symbol and row_symbol_value and row_symbol_value != symbol:
            continue
        row_order_ids = cost_probe_order_identifiers(row, payload)
        row_trade_ids = cost_probe_trade_identifiers(row, payload)
        if not (row_order_ids.intersection(order_ids) or row_trade_ids.intersection(trade_ids)):
            continue
        key = (
            ";".join(sorted(row_order_ids)),
            ";".join(sorted(row_trade_ids)),
            str(_first_value(row, payload, ["ts", "event_ts", "timestamp"]) or ""),
        )
        matches[key] = row
    return list(matches.values())


def _matching_bill_rows(
    rows: Sequence[dict[str, Any]],
    *,
    fill_rows: Sequence[dict[str, Any]],
    order_ids: set[str],
    trade_ids: set[str],
) -> list[dict[str, Any]]:
    matches: dict[tuple[str, str, str], dict[str, Any]] = {}
    for bill in rows:
        bill_payload = _payload_dict(bill)
        bill_order_ids = cost_probe_order_identifiers(bill, bill_payload)
        bill_trade_ids = cost_probe_trade_identifiers(bill, bill_payload)
        direct_match = bool(
            bill_order_ids.intersection(order_ids) or bill_trade_ids.intersection(trade_ids)
        )
        inferred_match = any(_bill_matches_fill_fee_and_time(bill, fill) for fill in fill_rows)
        if not (direct_match or inferred_match):
            continue
        if _fee_from_bill(bill) is None:
            continue
        key = (
            str(_first_value(bill, bill_payload, ["bill_id", "billId"]) or ""),
            str(_first_value(bill, bill_payload, ["ts", "event_ts", "timestamp"]) or ""),
            str(_first_value(bill, bill_payload, ["amount", "balChg", "fee", "fee_usdt"]) or ""),
        )
        matches[key] = bill
    return list(matches.values())


def _bill_matches_fill_fee_and_time(bill: Mapping[str, Any], fill: Mapping[str, Any]) -> bool:
    bill_fee = _fee_from_bill(bill)
    fill_fee = _fee_from_fill(fill)
    if bill_fee is None or fill_fee is None:
        return False
    tolerance = max(0.000001, abs(fill_fee) * 0.02)
    if abs(bill_fee - fill_fee) > tolerance:
        return False
    bill_ts = _row_timestamp(bill)
    fill_ts = _row_timestamp(fill)
    if bill_ts is None or fill_ts is None:
        return False
    return abs((bill_ts - fill_ts).total_seconds()) <= 15 * 60


def _fee_from_fill(row: Mapping[str, Any]) -> float | None:
    payload = _payload_dict(row)
    fee_usdt = _first_float(row, payload, ["fee_usdt", "fee_abs_usdt"])
    if fee_usdt is not None:
        return abs(fee_usdt)
    fee = _first_float(row, payload, ["fee", "commission", "fee_abs"])
    fee_ccy = str(_first_value(row, payload, ["fee_currency", "fee_ccy", "feeCcy"]) or "").upper()
    if fee is not None and (fee_ccy in {"", "USDT", "USD"}):
        return abs(fee)
    symbol = cost_probe_symbol(row, payload)
    price = _first_float(row, payload, ["fill_price", "fillPx", "fill_px", "avg_px", "px"])
    if fee is not None and price is not None and fee_ccy == _base_ccy(symbol):
        return abs(fee) * abs(price)
    return None


def _fee_from_bill(row: Mapping[str, Any]) -> float | None:
    payload = _payload_dict(row)
    fee_usdt = _first_float(row, payload, ["fee_usdt", "fee_abs_usdt"])
    if fee_usdt is not None:
        return abs(fee_usdt)
    fee = _first_float(row, payload, ["fee"])
    if fee is None:
        return None
    fee_ccy = str(
        _first_value(row, payload, ["fee_currency", "fee_ccy", "feeCcy", "ccy"]) or ""
    ).upper()
    if fee_ccy in {"", "USDT", "USD"}:
        return abs(fee)
    symbol = cost_probe_symbol(row, payload)
    price = _first_float(row, payload, ["fill_price", "fillPx", "fill_px", "avg_px", "px"])
    if price is not None and fee_ccy == _base_ccy(symbol):
        return abs(fee) * abs(price)
    return None


def _base_ccy(symbol: str) -> str:
    normalized = normalize_symbol(symbol) if symbol else ""
    if "-" not in normalized:
        return normalized
    return normalized.split("-", 1)[0]


def _row_timestamp(row: Mapping[str, Any]) -> datetime | None:
    payload = _payload_dict(row)
    return _first_timestamp(row, payload, ("ts", "event_ts", "timestamp", "created_at"))


def _bill_ids(rows: Sequence[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        payload = _payload_dict(row)
        value = str(_first_value(row, payload, ["bill_id", "billId"]) or "").strip()
        if value:
            ids.add(value)
    return ids


def _sum_optional(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return sum(observed)


def _fee_diff(
    entry_fill: float | None,
    entry_bill: float | None,
    exit_fill: float | None,
    exit_bill: float | None,
) -> float | None:
    if any(value is None for value in (entry_fill, entry_bill, exit_fill, exit_bill)):
        return None
    return abs((entry_bill or 0.0) + (exit_bill or 0.0) - (entry_fill or 0.0) - (exit_fill or 0.0))


def _bill_match_status(
    entry: Mapping[str, Any],
    exit_: Mapping[str, Any],
    fee_diff: float | None,
) -> str:
    entry_has_fill = entry.get("fee_from_fill") is not None
    exit_has_fill = exit_.get("fee_from_fill") is not None
    entry_has_bill = entry.get("fee_from_bill") is not None
    exit_has_bill = exit_.get("fee_from_bill") is not None
    if not entry_has_fill and not exit_has_fill:
        return "NO_COST_PROBE_FILLS"
    if not entry_has_bill and not exit_has_bill:
        return "BILL_NOT_OBSERVED"
    if entry_has_bill != entry_has_fill or exit_has_bill != exit_has_fill:
        return "PARTIAL"
    fill_fee_total = (entry.get("fee_from_fill") or 0.0) + (exit_.get("fee_from_fill") or 0.0)
    tolerance = max(0.000001, fill_fee_total * 0.02)
    if fee_diff is not None and fee_diff <= tolerance:
        return "PASS"
    return "FEE_MISMATCH"


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.12g}"


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
