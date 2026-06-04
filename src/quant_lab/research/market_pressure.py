from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import polars as pl

from quant_lab.symbols import normalize_symbol


MARKET_PRESSURE_SCHEMA_VERSION = "market_pressure_score.v0.1"
MAJOR_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]
MARKET_PRESSURE_FIELDS = [
    "generated_at",
    "schema_version",
    "ts_utc",
    "market_pressure_state",
    "market_pressure_score",
    "broad_market_positive_count",
    "btc_24h_return_bps",
    "eth_24h_return_bps",
    "sol_24h_return_bps",
    "bnb_24h_return_bps",
    "avg_major_24h_return_bps",
    "downside_symbol_count",
    "avg_spread_bps_15m",
    "total_trade_count_60m",
    "capitulation_watch_count",
    "bottom_probe_allowed_count",
    "risk_off_no_catch_count",
    "response_action",
    "live_order_effect",
]


def build_market_pressure_score(
    *,
    bottom_zone_reversal_shadow: pl.DataFrame,
    fast_microstructure_features: pl.DataFrame | None = None,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    if bottom_zone_reversal_shadow.is_empty():
        return _empty_frame()
    rows = bottom_zone_reversal_shadow.to_dicts()
    fast_frame = fast_microstructure_features if fast_microstructure_features is not None else pl.DataFrame()
    fast_rows = fast_frame.to_dicts()
    latest_ts = _latest_iso([*rows, *fast_rows]) or generated.isoformat().replace("+00:00", "Z")
    by_symbol = {normalize_symbol(row.get("symbol")): row for row in rows}
    returns = {
        symbol: _float((by_symbol.get(symbol) or {}).get("return_24h_bps"))
        for symbol in MAJOR_SYMBOLS
    }
    observed_returns = [value for value in returns.values() if value is not None]
    broad_positive = sum(1 for value in observed_returns if value > 0)
    avg_return = sum(observed_returns) / len(observed_returns) if observed_returns else None
    downside_count = sum(1 for value in observed_returns if value <= -150)
    spread_values = [_float(row.get("avg_spread_bps_15m")) for row in fast_rows]
    spread_values = [value for value in spread_values if value is not None]
    avg_spread = sum(spread_values) / len(spread_values) if spread_values else None
    trade_counts = [_float(row.get("trade_count_60m")) for row in fast_rows]
    trade_counts = [value for value in trade_counts if value is not None]
    total_trade_count = sum(trade_counts) if trade_counts else None
    state_counts = _state_counts(rows)
    pressure_score = _pressure_score(
        avg_return=avg_return,
        broad_positive=broad_positive,
        avg_spread=avg_spread,
        bottom_probe_allowed_count=state_counts["BOTTOM_PROBE_ALLOWED"],
        capitulation_watch_count=state_counts["CAPITULATION_WATCH"],
        risk_off_no_catch_count=state_counts["RISK_OFF_NO_CATCH"],
    )
    state = _pressure_state(
        avg_return=avg_return,
        broad_positive=broad_positive,
        bottom_probe_allowed_count=state_counts["BOTTOM_PROBE_ALLOWED"],
        capitulation_watch_count=state_counts["CAPITULATION_WATCH"],
        risk_off_no_catch_count=state_counts["RISK_OFF_NO_CATCH"],
        avg_spread=avg_spread,
    )
    frame = pl.DataFrame(
        [
            {
                "generated_at": generated.isoformat().replace("+00:00", "Z"),
                "schema_version": MARKET_PRESSURE_SCHEMA_VERSION,
                "ts_utc": latest_ts,
                "market_pressure_state": state,
                "market_pressure_score": pressure_score,
                "broad_market_positive_count": broad_positive,
                "btc_24h_return_bps": returns.get("BTC-USDT"),
                "eth_24h_return_bps": returns.get("ETH-USDT"),
                "sol_24h_return_bps": returns.get("SOL-USDT"),
                "bnb_24h_return_bps": returns.get("BNB-USDT"),
                "avg_major_24h_return_bps": avg_return,
                "downside_symbol_count": downside_count,
                "avg_spread_bps_15m": avg_spread,
                "total_trade_count_60m": total_trade_count,
                "capitulation_watch_count": state_counts["CAPITULATION_WATCH"],
                "bottom_probe_allowed_count": state_counts["BOTTOM_PROBE_ALLOWED"],
                "risk_off_no_catch_count": state_counts["RISK_OFF_NO_CATCH"],
                "response_action": "shadow_tracking",
                "live_order_effect": "read_only_no_live_order",
            }
        ],
        infer_schema_length=None,
    )
    return frame.select(MARKET_PRESSURE_FIELDS)


def market_pressure_summary_md(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "\n".join(
            [
                "# Market Pressure Score",
                "",
                "No market pressure state is observable in the current export.",
                "",
                "Live order effect: none. This report is read-only.",
            ]
        )
    row = frame.to_dicts()[0]
    return "\n".join(
        [
            "# Market Pressure Score",
            "",
            f"- state: {row.get('market_pressure_state')}",
            f"- score: {row.get('market_pressure_score')}",
            f"- broad_market_positive_count: {row.get('broad_market_positive_count')}",
            f"- avg_major_24h_return_bps: {row.get('avg_major_24h_return_bps')}",
            f"- bottom_probe_allowed_count: {row.get('bottom_probe_allowed_count')}",
            f"- capitulation_watch_count: {row.get('capitulation_watch_count')}",
            f"- risk_off_no_catch_count: {row.get('risk_off_no_catch_count')}",
            "- live_order_effect: read_only_no_live_order",
        ]
    )


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={field: pl.Utf8 for field in MARKET_PRESSURE_FIELDS})


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _latest_iso(rows: list[dict[str, Any]]) -> str | None:
    values = [str(row.get("ts_utc") or "").strip() for row in rows if str(row.get("ts_utc") or "").strip()]
    return max(values) if values else None


def _state_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "BOTTOM_PROBE_ALLOWED": 0,
        "CAPITULATION_WATCH": 0,
        "RISK_OFF_NO_CATCH": 0,
    }
    for row in rows:
        state = str(row.get("bottom_zone_state") or "").strip().upper()
        if state in counts:
            counts[state] += 1
    return counts


def _pressure_score(
    *,
    avg_return: float | None,
    broad_positive: int,
    avg_spread: float | None,
    bottom_probe_allowed_count: int,
    capitulation_watch_count: int,
    risk_off_no_catch_count: int,
) -> float:
    score = 0.5
    if avg_return is not None:
        score += 0.20 if avg_return > 80 else -0.25 if avg_return < -250 else -0.10 if avg_return < -120 else 0.0
    score += 0.10 * min(max(broad_positive - 1, 0), 3)
    score += 0.12 * min(bottom_probe_allowed_count, 3)
    score += 0.05 * min(capitulation_watch_count, 3)
    score -= 0.12 * min(risk_off_no_catch_count, 3)
    if avg_spread is not None and avg_spread > 30:
        score -= 0.15
    return round(max(0.0, min(1.0, score)), 4)


def _pressure_state(
    *,
    avg_return: float | None,
    broad_positive: int,
    bottom_probe_allowed_count: int,
    capitulation_watch_count: int,
    risk_off_no_catch_count: int,
    avg_spread: float | None,
) -> str:
    if broad_positive >= 3 and (avg_return or 0.0) > 80:
        return "RISK_ON_CONFIRMED"
    if bottom_probe_allowed_count >= 2 and (avg_spread is None or avg_spread <= 25):
        return "BOTTOM_PROBE_ALLOWED"
    if capitulation_watch_count >= 1 and risk_off_no_catch_count <= 2:
        return "CAPITULATION_WATCH"
    return "RISK_OFF_NO_CATCH"
