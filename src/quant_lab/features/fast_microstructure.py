from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from quant_lab.symbols import normalize_symbol

FAST_MICROSTRUCTURE_SCHEMA_VERSION = "fast_microstructure_features.v0.1"
FAST_MICROSTRUCTURE_FIELDS = [
    "generated_at",
    "schema_version",
    "symbol",
    "ts_utc",
    "close",
    "latest_spread_bps",
    "avg_spread_bps_5m",
    "avg_spread_bps_15m",
    "orderbook_imbalance_1m",
    "orderbook_imbalance_5m",
    "spread_widening_bps",
    "spread_bps_change_5m",
    "trade_count_5m",
    "trade_count_15m",
    "trade_count_60m",
    "size_sum_5m",
    "size_sum_15m",
    "size_sum_60m",
    "activity_ratio_15m_to_60m",
    "taker_buy_size_sum_15m",
    "taker_sell_size_sum_15m",
    "taker_buy_sell_imbalance_5m",
    "cvd_5m",
    "cvd_divergence",
    "cvd_size_15m",
    "vwap_1h",
    "close_vs_vwap_1h_bps",
    "return_1h_bps",
    "return_4h_bps",
    "realized_vol_4h_bps",
    "liquidity_quality",
    "pressure_bias",
    "response_action",
    "live_order_effect",
]


def build_fast_microstructure_features(
    *,
    market_bars: pl.DataFrame,
    orderbook_spread_1m: pl.DataFrame | None = None,
    trade_activity_1m: pl.DataFrame | None = None,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    """Build read-only fast microstructure diagnostics from preloaded export frames."""

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    if market_bars.is_empty():
        return _empty_frame()

    bars_by_symbol = _rows_by_symbol(market_bars, ts_fields=("ts", "minute_ts"))
    spread_frame = orderbook_spread_1m if orderbook_spread_1m is not None else pl.DataFrame()
    trade_frame = trade_activity_1m if trade_activity_1m is not None else pl.DataFrame()
    spreads_by_symbol = _rows_by_symbol(spread_frame, ts_fields=("minute_ts", "ts"))
    trades_by_symbol = _rows_by_symbol(
        trade_frame,
        ts_fields=("minute_ts", "latest_trade_ts", "ts"),
    )
    microstructure_symbols = (
        set(spreads_by_symbol).union(trades_by_symbol).intersection(bars_by_symbol)
    )
    symbols = sorted(microstructure_symbols or set(bars_by_symbol))
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        bars = bars_by_symbol.get(symbol, [])
        latest = bars[-1] if bars else None
        if latest is None:
            continue
        spread_rows = spreads_by_symbol.get(symbol, [])
        trade_rows = trades_by_symbol.get(symbol, [])
        ts = _feature_ts(latest, spread_rows, trade_rows)
        close = _latest_close(bars, ts) if ts is not None else None
        if ts is None or close is None:
            continue
        spread_5m = _avg_value(spread_rows, ts, minutes=5, field="spread_bps")
        spread_15m = _avg_value(spread_rows, ts, minutes=15, field="spread_bps")
        spread_60m = _avg_value(spread_rows, ts, minutes=60, field="spread_bps")
        latest_spread = _latest_value(spread_rows, ts, field="spread_bps")
        orderbook_imbalance_1m = _latest_value(spread_rows, ts, field="orderbook_imbalance")
        orderbook_imbalance_5m = _avg_value(spread_rows, ts, minutes=5, field="orderbook_imbalance")
        spread_5m_prior = _value_at_or_before(
            spread_rows,
            ts - timedelta(minutes=5),
            field="spread_bps",
        )
        trade_5m = _sum_value(trade_rows, ts, minutes=5, field="trade_count")
        trade_15m = _sum_value(trade_rows, ts, minutes=15, field="trade_count")
        trade_60m = _sum_value(trade_rows, ts, minutes=60, field="trade_count")
        size_5m = _sum_value(trade_rows, ts, minutes=5, field="size_sum")
        size_15m = _sum_value(trade_rows, ts, minutes=15, field="size_sum")
        size_60m = _sum_value(trade_rows, ts, minutes=60, field="size_sum")
        taker_buy_15m = _sum_first_observed(
            trade_rows,
            ts,
            minutes=15,
            fields=("taker_buy_size_sum", "buy_size_sum"),
        )
        taker_sell_15m = _sum_first_observed(
            trade_rows,
            ts,
            minutes=15,
            fields=("taker_sell_size_sum", "sell_size_sum"),
        )
        taker_buy_5m = _sum_first_observed(
            trade_rows,
            ts,
            minutes=5,
            fields=("taker_buy_size_sum", "buy_size_sum", "taker_buy_volume", "buy_volume"),
        )
        taker_sell_5m = _sum_first_observed(
            trade_rows,
            ts,
            minutes=5,
            fields=("taker_sell_size_sum", "sell_size_sum", "taker_sell_volume", "sell_volume"),
        )
        cvd_5m = None
        taker_imbalance_5m = None
        if taker_buy_5m is not None and taker_sell_5m is not None:
            cvd_5m = taker_buy_5m - taker_sell_5m
            total_5m = taker_buy_5m + taker_sell_5m
            if total_5m > 0:
                taker_imbalance_5m = cvd_5m / total_5m
        cvd = None
        if taker_buy_15m is not None and taker_sell_15m is not None:
            cvd = taker_buy_15m - taker_sell_15m
        vwap_1h = _vwap(bars, ts, hours=1)
        ret_1h = _return_bps(bars, ts, hours=1)
        ret_4h = _return_bps(bars, ts, hours=4)
        vol_4h = _realized_vol_bps(bars, ts, hours=4)
        activity_ratio = None
        if trade_15m is not None and trade_60m and trade_60m > 0:
            activity_ratio = trade_15m / max(trade_60m / 4.0, 1e-9)
        spread_widening = None
        if spread_15m is not None and spread_60m is not None:
            spread_widening = spread_15m - spread_60m
        spread_bps_change_5m = None
        if latest_spread is not None and spread_5m_prior is not None:
            spread_bps_change_5m = latest_spread - spread_5m_prior
        close_vs_vwap = None
        if vwap_1h and close:
            close_vs_vwap = (close / vwap_1h - 1.0) * 10000.0
        cvd_divergence = _cvd_divergence(ret_1h, taker_imbalance_5m)
        rows.append(
            {
                "generated_at": generated.isoformat().replace("+00:00", "Z"),
                "schema_version": FAST_MICROSTRUCTURE_SCHEMA_VERSION,
                "symbol": symbol,
                "ts_utc": ts.isoformat().replace("+00:00", "Z"),
                "close": close,
                "latest_spread_bps": latest_spread,
                "avg_spread_bps_5m": spread_5m,
                "avg_spread_bps_15m": spread_15m,
                "orderbook_imbalance_1m": orderbook_imbalance_1m,
                "orderbook_imbalance_5m": orderbook_imbalance_5m,
                "spread_widening_bps": spread_widening,
                "spread_bps_change_5m": spread_bps_change_5m,
                "trade_count_5m": trade_5m,
                "trade_count_15m": trade_15m,
                "trade_count_60m": trade_60m,
                "size_sum_5m": size_5m,
                "size_sum_15m": size_15m,
                "size_sum_60m": size_60m,
                "activity_ratio_15m_to_60m": activity_ratio,
                "taker_buy_size_sum_15m": taker_buy_15m,
                "taker_sell_size_sum_15m": taker_sell_15m,
                "taker_buy_sell_imbalance_5m": taker_imbalance_5m,
                "cvd_5m": cvd_5m,
                "cvd_divergence": cvd_divergence,
                "cvd_size_15m": cvd,
                "vwap_1h": vwap_1h,
                "close_vs_vwap_1h_bps": close_vs_vwap,
                "return_1h_bps": ret_1h,
                "return_4h_bps": ret_4h,
                "realized_vol_4h_bps": vol_4h,
                "liquidity_quality": _liquidity_quality(spread_15m, trade_60m),
                "pressure_bias": _pressure_bias(ret_1h, close_vs_vwap, cvd),
                "response_action": "diagnostic_only",
                "live_order_effect": "read_only_no_live_order",
            }
        )
    if not rows:
        return _empty_frame()
    return pl.DataFrame(rows, infer_schema_length=None).select(FAST_MICROSTRUCTURE_FIELDS)


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={field: pl.Utf8 for field in FAST_MICROSTRUCTURE_FIELDS})


def _rows_by_symbol(
    frame: pl.DataFrame,
    *,
    ts_fields: Iterable[str],
) -> dict[str, list[dict[str, Any]]]:
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
        value = _coerce_dt(row.get(field))
        if value is not None:
            return value
    return None


def _feature_ts(
    latest_bar: dict[str, Any],
    spread_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
) -> datetime | None:
    candidates = [
        _coerce_dt(row.get("_ts"))
        for row in (spread_rows[-1:] + trade_rows[-1:] + [latest_bar])
    ]
    candidates = [value for value in candidates if value is not None]
    return max(candidates) if candidates else None


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
    if not math.isfinite(number):
        return None
    return number


def _window_rows(rows: list[dict[str, Any]], ts: datetime, *, minutes: int) -> list[dict[str, Any]]:
    start = ts - timedelta(minutes=minutes)
    return [row for row in rows if start <= row.get("_ts") <= ts]


def _avg_value(
    rows: list[dict[str, Any]],
    ts: datetime,
    *,
    minutes: int,
    field: str,
) -> float | None:
    values = [_float(row.get(field)) for row in _window_rows(rows, ts, minutes=minutes)]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _latest_value(rows: list[dict[str, Any]], ts: datetime, *, field: str) -> float | None:
    candidates = [
        row for row in rows if row.get("_ts") <= ts and _float(row.get(field)) is not None
    ]
    if not candidates:
        return None
    return _float(candidates[-1].get(field))


def _value_at_or_before(rows: list[dict[str, Any]], ts: datetime, *, field: str) -> float | None:
    candidates = [
        row for row in rows if row.get("_ts") <= ts and _float(row.get(field)) is not None
    ]
    if not candidates:
        return None
    return _float(candidates[-1].get(field))


def _sum_value(
    rows: list[dict[str, Any]],
    ts: datetime,
    *,
    minutes: int,
    field: str,
) -> float | None:
    values = [_float(row.get(field)) for row in _window_rows(rows, ts, minutes=minutes)]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _sum_first_observed(
    rows: list[dict[str, Any]],
    ts: datetime,
    *,
    minutes: int,
    fields: Iterable[str],
) -> float | None:
    total = 0.0
    seen = False
    for row in _window_rows(rows, ts, minutes=minutes):
        for field in fields:
            value = _float(row.get(field))
            if value is not None:
                total += value
                seen = True
                break
    return total if seen else None


def _cvd_divergence(return_1h_bps: float | None, taker_imbalance_5m: float | None) -> float | None:
    if return_1h_bps is None or taker_imbalance_5m is None:
        return None
    if abs(return_1h_bps) < 1e-9:
        return 0.0
    if return_1h_bps < 0.0 and taker_imbalance_5m > 0.0:
        return taker_imbalance_5m
    if return_1h_bps > 0.0 and taker_imbalance_5m < 0.0:
        return taker_imbalance_5m
    return 0.0


def _bars_window(rows: list[dict[str, Any]], ts: datetime, *, hours: int) -> list[dict[str, Any]]:
    start = ts - timedelta(hours=hours)
    return [row for row in rows if start <= row.get("_ts") <= ts]


def _vwap(rows: list[dict[str, Any]], ts: datetime, *, hours: int) -> float | None:
    window = _bars_window(rows, ts, hours=hours)
    numerator = 0.0
    denominator = 0.0
    for row in window:
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


def _return_bps(rows: list[dict[str, Any]], ts: datetime, *, hours: int) -> float | None:
    latest_close = _latest_close(rows, ts)
    start_close = _latest_close(rows, ts - timedelta(hours=hours))
    if latest_close is None or start_close is None or start_close <= 0:
        return None
    return (latest_close / start_close - 1.0) * 10000.0


def _latest_close(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    candidates = [
        row
        for row in rows
        if row.get("_ts") <= ts and _float(row.get("close")) is not None
    ]
    if not candidates:
        return None
    return _float(candidates[-1].get("close"))


def _realized_vol_bps(rows: list[dict[str, Any]], ts: datetime, *, hours: int) -> float | None:
    window = _bars_window(rows, ts, hours=hours)
    returns: list[float] = []
    for prev, current in zip(window, window[1:], strict=False):
        prev_close = _float(prev.get("close"))
        close = _float(current.get("close"))
        if prev_close and close:
            returns.append((close / prev_close - 1.0) * 10000.0)
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return variance**0.5


def _liquidity_quality(spread_15m: float | None, trade_60m: float | None) -> str:
    if spread_15m is None and trade_60m is None:
        return "not_observable"
    if (spread_15m is not None and spread_15m <= 8.0) and (trade_60m is None or trade_60m >= 30):
        return "good"
    if spread_15m is not None and spread_15m >= 25.0:
        return "thin"
    return "mixed"


def _pressure_bias(
    return_1h_bps: float | None,
    close_vs_vwap_1h_bps: float | None,
    cvd_size_15m: float | None,
) -> str:
    bullish_votes = 0
    bearish_votes = 0
    if return_1h_bps is not None:
        bullish_votes += int(return_1h_bps > 30)
        bearish_votes += int(return_1h_bps < -30)
    if close_vs_vwap_1h_bps is not None:
        bullish_votes += int(close_vs_vwap_1h_bps > 20)
        bearish_votes += int(close_vs_vwap_1h_bps < -20)
    if cvd_size_15m is not None:
        bullish_votes += int(cvd_size_15m > 0)
        bearish_votes += int(cvd_size_15m < 0)
    if bullish_votes > bearish_votes:
        return "buy_pressure"
    if bearish_votes > bullish_votes:
        return "sell_pressure"
    return "neutral"
