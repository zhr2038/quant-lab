from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

import polars as pl

from quant_lab.symbols import normalize_symbol


BOTTOM_ZONE_SCHEMA_VERSION = "bottom_zone_reversal_shadow.v0.1"
BOTTOM_ZONE_FIELDS = [
    "generated_at",
    "schema_version",
    "symbol",
    "ts_utc",
    "close",
    "support_low_24h",
    "support_low_72h",
    "distance_to_24h_low_bps",
    "distance_to_72h_low_bps",
    "rebound_from_24h_low_bps",
    "return_4h_bps",
    "return_24h_bps",
    "return_72h_bps",
    "vwap_24h",
    "close_vs_vwap_24h_bps",
    "avg_spread_bps_15m",
    "trade_count_15m",
    "trade_count_60m",
    "bounce_probability_score",
    "bottom_zone_state",
    "would_probe_paper",
    "no_probe_reason",
    "response_action",
    "live_order_effect",
]


def build_bottom_zone_reversal_shadow(
    *,
    market_bars: pl.DataFrame,
    orderbook_spread_1m: pl.DataFrame | None = None,
    trade_activity_1m: pl.DataFrame | None = None,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    """Create bottom-zone reversal shadow rows without affecting execution."""

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    if market_bars.is_empty():
        return _empty_frame()
    bars_by_symbol = _rows_by_symbol(market_bars, ts_fields=("ts",))
    spread_frame = orderbook_spread_1m if orderbook_spread_1m is not None else pl.DataFrame()
    trade_frame = trade_activity_1m if trade_activity_1m is not None else pl.DataFrame()
    spreads_by_symbol = _rows_by_symbol(spread_frame, ts_fields=("minute_ts", "ts"))
    trades_by_symbol = _rows_by_symbol(trade_frame, ts_fields=("minute_ts", "latest_trade_ts", "ts"))
    rows: list[dict[str, Any]] = []
    for symbol, bars in sorted(bars_by_symbol.items()):
        latest = bars[-1] if bars else None
        if latest is None:
            continue
        ts = _coerce_dt(latest.get("_ts"))
        close = _float(latest.get("close"))
        if ts is None or close is None or close <= 0:
            continue
        low_24h = _min_value(bars, ts, hours=24, field="low")
        low_72h = _min_value(bars, ts, hours=72, field="low")
        return_4h = _return_bps(bars, ts, hours=4)
        return_24h = _return_bps(bars, ts, hours=24)
        return_72h = _return_bps(bars, ts, hours=72)
        vwap_24h = _vwap(bars, ts, hours=24)
        close_vs_vwap = (close / vwap_24h - 1.0) * 10000.0 if vwap_24h else None
        distance_24h = (close / low_24h - 1.0) * 10000.0 if low_24h and low_24h > 0 else None
        distance_72h = (close / low_72h - 1.0) * 10000.0 if low_72h and low_72h > 0 else None
        spread_15m = _avg_value(
            spreads_by_symbol.get(symbol, []),
            ts,
            minutes=15,
            field="spread_bps",
        )
        trade_15m = _sum_value(
            trades_by_symbol.get(symbol, []),
            ts,
            minutes=15,
            field="trade_count",
        )
        trade_60m = _sum_value(
            trades_by_symbol.get(symbol, []),
            ts,
            minutes=60,
            field="trade_count",
        )
        score = _bounce_probability_score(
            return_24h=return_24h,
            return_4h=return_4h,
            distance_to_24h_low_bps=distance_24h,
            close_vs_vwap_24h_bps=close_vs_vwap,
            avg_spread_bps_15m=spread_15m,
            trade_count_60m=trade_60m,
        )
        state, would_probe, reason = _bottom_state(
            score=score,
            return_24h=return_24h,
            distance_to_24h_low_bps=distance_24h,
            avg_spread_bps_15m=spread_15m,
        )
        rows.append(
            {
                "generated_at": generated.isoformat().replace("+00:00", "Z"),
                "schema_version": BOTTOM_ZONE_SCHEMA_VERSION,
                "symbol": symbol,
                "ts_utc": ts.isoformat().replace("+00:00", "Z"),
                "close": close,
                "support_low_24h": low_24h,
                "support_low_72h": low_72h,
                "distance_to_24h_low_bps": distance_24h,
                "distance_to_72h_low_bps": distance_72h,
                "rebound_from_24h_low_bps": distance_24h,
                "return_4h_bps": return_4h,
                "return_24h_bps": return_24h,
                "return_72h_bps": return_72h,
                "vwap_24h": vwap_24h,
                "close_vs_vwap_24h_bps": close_vs_vwap,
                "avg_spread_bps_15m": spread_15m,
                "trade_count_15m": trade_15m,
                "trade_count_60m": trade_60m,
                "bounce_probability_score": score,
                "bottom_zone_state": state,
                "would_probe_paper": would_probe,
                "no_probe_reason": reason,
                "response_action": "paper_tracking" if would_probe else "shadow_tracking",
                "live_order_effect": "read_only_no_live_order",
            }
        )
    if not rows:
        return _empty_frame()
    return pl.DataFrame(rows, infer_schema_length=None).select(BOTTOM_ZONE_FIELDS)


def bottom_zone_reversal_summary_md(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "\n".join(
            [
                "# Bottom Zone Reversal Shadow",
                "",
                "No bottom-zone rows are observable in the current export.",
                "",
                "Live order effect: none. This report is read-only.",
            ]
        )
    rows = frame.to_dicts()
    probe_count = sum(1 for row in rows if _truthy(row.get("would_probe_paper")))
    state_counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("bottom_zone_state") or "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1
    leaders = sorted(
        rows,
        key=lambda row: _float(row.get("bounce_probability_score")) or -1,
        reverse=True,
    )[:5]
    lines = [
        "# Bottom Zone Reversal Shadow",
        "",
        f"- rows: {len(rows)}",
        f"- paper probe candidates: {probe_count}",
        f"- state mix: {state_counts}",
        "- live_order_effect: read_only_no_live_order",
        "",
        "Top candidates:",
    ]
    for row in leaders:
        lines.append(
            "- "
            f"{row.get('symbol')} score={row.get('bounce_probability_score')} "
            f"state={row.get('bottom_zone_state')} reason={row.get('no_probe_reason')}"
        )
    return "\n".join(lines)


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={field: pl.Utf8 for field in BOTTOM_ZONE_FIELDS})


def _rows_by_symbol(frame: pl.DataFrame, *, ts_fields: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    if frame is None or frame.is_empty() or "symbol" not in frame.columns:
        return {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for raw in frame.to_dicts():
        symbol = normalize_symbol(raw.get("symbol"))
        if not symbol or symbol == "UNKNOWN":
            continue
        ts = _first_ts(raw, ts_fields)
        if ts is None:
            continue
        item = dict(raw)
        item["_ts"] = ts
        rows.setdefault(symbol, []).append(item)
    for values in rows.values():
        values.sort(key=lambda row: row["_ts"])
    return rows


def _first_ts(row: dict[str, Any], fields: Iterable[str]) -> datetime | None:
    for field in fields:
        ts = _coerce_dt(row.get(field))
        if ts is not None:
            return ts
    return None


def _coerce_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _window_rows(rows: list[dict[str, Any]], ts: datetime, *, hours: int) -> list[dict[str, Any]]:
    start = ts - timedelta(hours=hours)
    return [row for row in rows if start <= row.get("_ts") <= ts]


def _minute_window(rows: list[dict[str, Any]], ts: datetime, *, minutes: int) -> list[dict[str, Any]]:
    start = ts - timedelta(minutes=minutes)
    return [row for row in rows if start <= row.get("_ts") <= ts]


def _min_value(rows: list[dict[str, Any]], ts: datetime, *, hours: int, field: str) -> float | None:
    values = [_float(row.get(field)) for row in _window_rows(rows, ts, hours=hours)]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _avg_value(rows: list[dict[str, Any]], ts: datetime, *, minutes: int, field: str) -> float | None:
    values = [_float(row.get(field)) for row in _minute_window(rows, ts, minutes=minutes)]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _sum_value(rows: list[dict[str, Any]], ts: datetime, *, minutes: int, field: str) -> float | None:
    values = [_float(row.get(field)) for row in _minute_window(rows, ts, minutes=minutes)]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _latest_close(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    candidates = [row for row in rows if row.get("_ts") <= ts and _float(row.get("close")) is not None]
    if not candidates:
        return None
    return _float(candidates[-1].get("close"))


def _return_bps(rows: list[dict[str, Any]], ts: datetime, *, hours: int) -> float | None:
    latest = _latest_close(rows, ts)
    prior = _latest_close(rows, ts - timedelta(hours=hours))
    if latest is None or prior is None or prior <= 0:
        return None
    return (latest / prior - 1.0) * 10000.0


def _vwap(rows: list[dict[str, Any]], ts: datetime, *, hours: int) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in _window_rows(rows, ts, hours=hours):
        close = _float(row.get("close"))
        volume = _float(row.get("volume"))
        quote_volume = _float(row.get("quote_volume"))
        if close is None:
            continue
        if quote_volume is not None and volume and volume > 0:
            numerator += quote_volume
            denominator += volume
        elif volume and volume > 0:
            numerator += close * volume
            denominator += volume
    return numerator / denominator if denominator > 0 else None


def _bounce_probability_score(
    *,
    return_24h: float | None,
    return_4h: float | None,
    distance_to_24h_low_bps: float | None,
    close_vs_vwap_24h_bps: float | None,
    avg_spread_bps_15m: float | None,
    trade_count_60m: float | None,
) -> float:
    score = 0.0
    if return_24h is not None:
        score += 0.30 if return_24h <= -300 else 0.15 if return_24h <= -150 else -0.10
    if distance_to_24h_low_bps is not None:
        score += 0.25 if 20 <= distance_to_24h_low_bps <= 180 else -0.10 if distance_to_24h_low_bps > 350 else 0.05
    if return_4h is not None:
        score += 0.20 if return_4h > 30 else -0.15 if return_4h < -120 else 0.0
    if close_vs_vwap_24h_bps is not None:
        score += 0.15 if close_vs_vwap_24h_bps > -80 else -0.05
    if avg_spread_bps_15m is not None:
        score += 0.10 if avg_spread_bps_15m <= 12 else -0.10 if avg_spread_bps_15m >= 30 else 0.0
    if trade_count_60m is not None:
        score += 0.10 if trade_count_60m >= 30 else -0.05
    return round(max(0.0, min(1.0, score)), 4)


def _bottom_state(
    *,
    score: float,
    return_24h: float | None,
    distance_to_24h_low_bps: float | None,
    avg_spread_bps_15m: float | None,
) -> tuple[str, bool, str]:
    reasons: list[str] = []
    if return_24h is None:
        reasons.append("return_24h_not_observable")
    elif return_24h > -120:
        reasons.append("not_enough_downside")
    if distance_to_24h_low_bps is None:
        reasons.append("distance_to_low_not_observable")
    elif distance_to_24h_low_bps > 300:
        reasons.append("too_far_from_24h_low")
    if avg_spread_bps_15m is not None and avg_spread_bps_15m >= 35:
        reasons.append("spread_too_wide")
    if score >= 0.65 and not reasons:
        return "BOTTOM_PROBE_ALLOWED", True, ""
    if score >= 0.45:
        return "CAPITULATION_WATCH", False, ";".join(reasons or ["score_lt_probe_threshold"])
    if return_24h is not None and return_24h <= -250:
        return "RISK_OFF_NO_CATCH", False, ";".join(reasons or ["bounce_confirmation_missing"])
    return "NO_BOTTOM_EDGE", False, ";".join(reasons or ["score_too_low"])
