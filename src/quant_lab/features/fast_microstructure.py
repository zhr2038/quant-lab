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
    "side_inferred",
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
        spread_5m = _avg_spread_bps(spread_rows, ts, minutes=5)
        spread_15m = _avg_spread_bps(spread_rows, ts, minutes=15)
        spread_60m = _avg_spread_bps(spread_rows, ts, minutes=60)
        latest_spread = _latest_spread_bps(spread_rows, ts)
        orderbook_imbalance_1m = _latest_orderbook_imbalance(spread_rows, ts)
        orderbook_imbalance_5m = _avg_orderbook_imbalance(spread_rows, ts, minutes=5)
        spread_5m_prior = _spread_at_or_before(
            spread_rows,
            ts - timedelta(minutes=5),
        )
        trade_5m = _sum_value(trade_rows, ts, minutes=5, field="trade_count")
        trade_15m = _sum_value(trade_rows, ts, minutes=15, field="trade_count")
        trade_60m = _sum_value(trade_rows, ts, minutes=60, field="trade_count")
        size_5m = _sum_value(trade_rows, ts, minutes=5, field="size_sum")
        size_15m = _sum_value(trade_rows, ts, minutes=15, field="size_sum")
        size_60m = _sum_value(trade_rows, ts, minutes=60, field="size_sum")
        taker_buy_15m, taker_sell_15m, side_inferred_15m = _buy_sell_sums(
            trade_rows,
            spread_rows,
            ts,
            minutes=15,
        )
        taker_buy_5m, taker_sell_5m, side_inferred_5m = _buy_sell_sums(
            trade_rows,
            spread_rows,
            ts,
            minutes=5,
        )
        side_inferred = side_inferred_15m or side_inferred_5m
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
                "side_inferred": side_inferred,
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


def _avg_spread_bps(rows: list[dict[str, Any]], ts: datetime, *, minutes: int) -> float | None:
    values = [_spread_bps(row) for row in _window_rows(rows, ts, minutes=minutes)]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _latest_spread_bps(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    candidates = [row for row in rows if row.get("_ts") <= ts and _spread_bps(row) is not None]
    return _spread_bps(candidates[-1]) if candidates else None


def _spread_at_or_before(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    candidates = [row for row in rows if row.get("_ts") <= ts and _spread_bps(row) is not None]
    return _spread_bps(candidates[-1]) if candidates else None


def _spread_bps(row: dict[str, Any]) -> float | None:
    explicit = _first_float(
        row,
        (
            "spread_bps",
            "avg_spread_bps",
            "latest_spread_bps",
            "spread",
        ),
    )
    if explicit is not None:
        return explicit
    bid = _first_float(row, ("bid", "best_bid", "bid_px", "bid_price"))
    ask = _first_float(row, ("ask", "best_ask", "ask_px", "ask_price"))
    if bid is None or ask is None or ask <= 0 or bid <= 0:
        return None
    mid = _mid_price(row)
    if mid is None or mid <= 0:
        mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 10000.0


def _avg_orderbook_imbalance(
    rows: list[dict[str, Any]],
    ts: datetime,
    *,
    minutes: int,
) -> float | None:
    values = [_orderbook_imbalance(row) for row in _window_rows(rows, ts, minutes=minutes)]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _latest_orderbook_imbalance(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    candidates = [
        row for row in rows if row.get("_ts") <= ts and _orderbook_imbalance(row) is not None
    ]
    return _orderbook_imbalance(candidates[-1]) if candidates else None


def _orderbook_imbalance(row: dict[str, Any]) -> float | None:
    explicit = _first_float(row, ("orderbook_imbalance", "imbalance", "book_imbalance"))
    if explicit is not None:
        return explicit
    bid_size = _first_float(row, ("bid_size", "best_bid_size", "bid_qty", "bid_sz"))
    ask_size = _first_float(row, ("ask_size", "best_ask_size", "ask_qty", "ask_sz"))
    if bid_size is None or ask_size is None:
        return None
    total = bid_size + ask_size
    if total <= 0:
        return None
    return (bid_size - ask_size) / total


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


def _buy_sell_sums(
    trade_rows: list[dict[str, Any]],
    spread_rows: list[dict[str, Any]],
    ts: datetime,
    *,
    minutes: int,
) -> tuple[float | None, float | None, bool]:
    window = _window_rows(trade_rows, ts, minutes=minutes)
    buy = _sum_first_observed(
        window,
        ts,
        minutes=minutes,
        fields=("taker_buy_size_sum", "buy_size_sum", "taker_buy_volume", "buy_volume"),
    )
    sell = _sum_first_observed(
        window,
        ts,
        minutes=minutes,
        fields=("taker_sell_size_sum", "sell_size_sum", "taker_sell_volume", "sell_volume"),
    )
    if buy is not None or sell is not None:
        return buy or 0.0, sell or 0.0, False

    inferred_buy = 0.0
    inferred_sell = 0.0
    inferred = False
    for row in window:
        size = _trade_size(row)
        if size is None or size <= 0:
            continue
        side = _trade_side(row)
        if side == "buy":
            inferred_buy += size
            inferred = True
            continue
        if side == "sell":
            inferred_sell += size
            inferred = True
            continue
        price = _first_float(row, ("latest_trade_px", "trade_px", "price", "last_px", "close"))
        mid = _mid_price(row) or _latest_mid_price(spread_rows, row.get("_ts") or ts)
        if price is None or mid is None:
            continue
        if price >= mid:
            inferred_buy += size
        else:
            inferred_sell += size
        inferred = True
    if not inferred:
        return None, None, False
    return inferred_buy, inferred_sell, True


def _trade_side(row: dict[str, Any]) -> str | None:
    raw = str(
        row.get("taker_side")
        or row.get("aggressor_side")
        or row.get("side")
        or row.get("direction")
        or ""
    ).lower()
    if raw in {"b", "buy", "bid", "buyer", "taker_buy"} or "buy" in raw:
        return "buy"
    if raw in {"s", "sell", "ask", "seller", "taker_sell"} or "sell" in raw:
        return "sell"
    return None


def _trade_size(row: dict[str, Any]) -> float | None:
    return _first_float(row, ("size_sum", "size", "qty", "quantity", "amount", "volume"))


def _mid_price(row: dict[str, Any]) -> float | None:
    mid = _first_float(row, ("mid", "mid_px", "arrival_mid", "book_mid"))
    if mid is not None:
        return mid
    bid = _first_float(row, ("bid", "best_bid", "bid_px", "bid_price"))
    ask = _first_float(row, ("ask", "best_ask", "ask_px", "ask_price"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return None


def _latest_mid_price(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    candidates = [row for row in rows if row.get("_ts") <= ts and _mid_price(row) is not None]
    return _mid_price(candidates[-1]) if candidates else None


def _first_float(row: dict[str, Any], fields: Iterable[str]) -> float | None:
    for field in fields:
        value = _float(row.get(field))
        if value is not None:
            return value
    return None


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
