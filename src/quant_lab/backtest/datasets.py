from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

import polars as pl

from quant_lab.symbols import normalize_symbol

HORIZONS = (4, 8, 12, 24, 48, 72)


def rows(frame: pl.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.is_empty():
        return []
    return [dict(row) for row in frame.to_dicts()]


def normalize_strategy_symbol(value: Any) -> str:
    return normalize_symbol(value) or "UNKNOWN"


def coerce_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.isdigit():
            raw = int(text)
            if raw > 10_000_000_000:
                raw = raw // 1000
            return datetime.fromtimestamp(raw, tz=UTC)
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except (OSError, ValueError):
        return None


def iso_utc(value: Any) -> str:
    parsed = coerce_dt(value)
    if parsed is None:
        return ""
    return parsed.isoformat().replace("+00:00", "Z")


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "buy", "allow", "allowed", "paper_ready"}


def first_value(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, "", "not_observable", "unknown", "nan"):
            return value
    return None


def first_float(row: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = float_or_none(row.get(name))
        if value is not None:
            return value
    return None


def market_rows_by_symbol(market_bars: pl.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows(market_bars):
        symbol = normalize_strategy_symbol(row.get("symbol"))
        ts = coerce_dt(first_value(row, ("ts", "ts_utc", "timestamp", "bar_ts")))
        close = float_or_none(row.get("close"))
        if symbol == "UNKNOWN" or ts is None or close is None or close <= 0:
            continue
        item = dict(row)
        item["_ts"] = ts
        item["_close"] = close
        out.setdefault(symbol, []).append(item)
    for values in out.values():
        values.sort(key=lambda item: item["_ts"])
    return out


def price_at_or_after(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    symbol: str,
    ts: datetime,
) -> float | None:
    for row in bars_by_symbol.get(normalize_strategy_symbol(symbol), []):
        if row["_ts"] >= ts:
            return float_or_none(row.get("close")) or row.get("_close")
    return None


def entry_price_from_row(row: dict[str, Any]) -> float | None:
    return first_float(
        row,
        (
            "entry_px",
            "entry_price",
            "entry_close",
            "close",
            "candidate_close",
            "price",
            "last_px",
        ),
    )


def future_net_bps_from_market(
    *,
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    symbol: str,
    ts: datetime,
    entry_px: float,
    horizon_hours: int,
    cost_bps: float,
) -> float | None:
    future_px = price_at_or_after(
        bars_by_symbol,
        symbol,
        ts + timedelta(hours=int(horizon_hours)),
    )
    if future_px is None or future_px <= 0 or entry_px <= 0:
        return None
    return (future_px / entry_px - 1.0) * 10_000.0 - cost_bps


def empty_frame(fields: list[str]) -> pl.DataFrame:
    return pl.DataFrame(schema={field: pl.Utf8 for field in fields})
