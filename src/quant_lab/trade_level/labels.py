from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from quant_lab.trade_level.judgment import event_id_for_row

TRADE_OPPORTUNITY_LABEL_SCHEMA_VERSION = "trade_opportunity_label.v0.1"
TRADE_OPPORTUNITY_LABEL_SCHEMA = {
    "schema_version": pl.Utf8,
    "event_id": pl.Utf8,
    "candidate_id": pl.Utf8,
    "run_id": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "label_4h_after_cost_bps": pl.Float64,
    "label_8h_after_cost_bps": pl.Float64,
    "label_24h_after_cost_bps": pl.Float64,
    "hit_4h": pl.Boolean,
    "hit_8h": pl.Boolean,
    "hit_24h": pl.Boolean,
    "max_adverse_bps": pl.Float64,
    "max_favorable_bps": pl.Float64,
    "label_status": pl.Utf8,
    "label_reason": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

_HORIZONS = (4, 8, 24)


def build_trade_opportunity_labels(
    events: pl.DataFrame,
    candidate_labels: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    created = created_at or datetime.now(UTC)
    if events.is_empty():
        return pl.DataFrame(schema=TRADE_OPPORTUNITY_LABEL_SCHEMA)

    labels_by_key = _candidate_labels_by_key(candidate_labels)
    rows: list[dict[str, Any]] = []
    for event in events.to_dicts():
        event_id = str(event.get("event_id") or event_id_for_row(event))
        label_rows = _matching_labels(event, labels_by_key)
        row = {
            "schema_version": TRADE_OPPORTUNITY_LABEL_SCHEMA_VERSION,
            "event_id": event_id,
            "candidate_id": _text(event.get("candidate_id")),
            "run_id": _text(event.get("run_id")),
            "symbol": _text(event.get("symbol")),
            "strategy_candidate": _text(event.get("strategy_candidate")),
            "decision_ts": _timestamp(event.get("decision_ts")),
            "label_4h_after_cost_bps": None,
            "label_8h_after_cost_bps": None,
            "label_24h_after_cost_bps": None,
            "hit_4h": None,
            "hit_8h": None,
            "hit_24h": None,
            "max_adverse_bps": None,
            "max_favorable_bps": None,
            "label_status": "missing",
            "label_reason": "candidate_label_missing",
            "created_at": created,
            "source": "quant_lab.trade_level.labels",
        }
        complete_count = 0
        pending_reasons: list[str] = []
        adverse: list[float] = []
        favorable: list[float] = []
        for label in label_rows:
            horizon = _int(label.get("horizon_hours"))
            if horizon not in _HORIZONS:
                continue
            status = _text(label.get("label_status")).lower()
            reason = _text(label.get("label_reason"))
            net = _float(label.get("net_bps_after_cost"))
            if status == "complete" and net is not None:
                complete_count += 1
                row[f"label_{horizon}h_after_cost_bps"] = net
                row[f"hit_{horizon}h"] = (
                    bool(label.get("win")) if label.get("win") is not None else net > 0.0
                )
            elif reason:
                pending_reasons.append(reason)
            mae = _float(label.get("mae_bps"))
            mfe = _float(label.get("mfe_bps"))
            if mae is not None:
                adverse.append(mae)
            if mfe is not None:
                favorable.append(mfe)
        if complete_count:
            row["label_status"] = "complete" if complete_count == len(_HORIZONS) else "partial"
            row["label_reason"] = "ok" if complete_count == len(_HORIZONS) else "partial_horizons"
        elif pending_reasons:
            row["label_status"] = "pending"
            row["label_reason"] = ";".join(sorted(set(pending_reasons)))
        if adverse:
            row["max_adverse_bps"] = min(adverse)
        if favorable:
            row["max_favorable_bps"] = max(favorable)
        rows.append(row)
    return _label_frame(rows)


def _candidate_labels_by_key(
    frame: pl.DataFrame,
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    if frame.is_empty():
        return grouped
    for row in frame.to_dicts():
        for key in _label_keys(row):
            grouped.setdefault(key, []).append(row)
    return grouped


def _matching_labels(
    event: dict[str, Any],
    labels_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for key in _label_keys(event):
        for row in labels_by_key.get(key, []):
            marker = id(row)
            if marker not in seen:
                seen.add(marker)
                rows.append(row)
    return rows


def _label_keys(row: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    candidate_id = _text(row.get("candidate_id"))
    run_id = _text(row.get("run_id"))
    symbol = _text(row.get("symbol"))
    strategy = _text(row.get("strategy_candidate"))
    keys: list[tuple[str, str, str, str]] = []
    if candidate_id:
        keys.append((candidate_id, run_id, symbol, strategy))
        keys.append((candidate_id, "", "", ""))
    if run_id and symbol and strategy:
        keys.append(("", run_id, symbol, strategy))
    if run_id and symbol:
        keys.append(("", run_id, symbol, ""))
    return keys


def _label_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=TRADE_OPPORTUNITY_LABEL_SCHEMA)
    return pl.DataFrame(rows, schema=TRADE_OPPORTUNITY_LABEL_SCHEMA, orient="row").select(
        [
            pl.col(name).cast(dtype, strict=False).alias(name)
            for name, dtype in TRADE_OPPORTUNITY_LABEL_SCHEMA.items()
        ]
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
