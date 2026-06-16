from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

import polars as pl

from quant_lab.backtest.cost_model import conservative_cost_for_symbol
from quant_lab.symbols import normalize_symbol

FAST_MICROSTRUCTURE_SCHEMA_VERSION = "fast_microstructure_features.v0.1"
FAST_MICROSTRUCTURE_TARGET_SYMBOLS = (
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "BNB-USDT",
    "WLD-USDT",
    "HYPE-USDT",
    "XRP-USDT",
    "ZEC-USDT",
)
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
    "bid_depth_recovery",
    "spread_normalization",
    "side_inferred",
    "cvd_size_15m",
    "vwap_1h",
    "close_vs_vwap_1h_bps",
    "return_1h_bps",
    "return_4h_bps",
    "realized_vol_4h_bps",
    "liquidity_quality",
    "pressure_bias",
    "missing_reason",
    "response_action",
    "live_order_effect",
]
FAST_MICROSTRUCTURE_FORWARD_FEATURES = (
    "orderbook_imbalance_1m",
    "orderbook_imbalance_5m",
    "taker_buy_sell_imbalance_5m",
    "cvd_5m",
    "cvd_divergence",
    "spread_bps_change_5m",
)
FAST_MICROSTRUCTURE_FORWARD_HORIZONS = (1, 4, 8)
FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS = 2000
FAST_MICROSTRUCTURE_FORWARD_AGGREGATE_REGIME = "ALL_REGIMES"
FAST_MICROSTRUCTURE_FORWARD_REGIMES = (
    "RISK_OFF",
    "SIDEWAYS",
    "RISK_ON_CONFIRMED",
    "TREND_UP",
)
FAST_MICROSTRUCTURE_FORWARD_TEST_FIELDS = [
    "generated_at",
    "feature_name",
    "symbol",
    "regime",
    "horizon_hours",
    "sample_count",
    "rank_ic",
    "long_short_bps",
    "p25_net_bps",
    "hit_rate",
    "recent_7d_score",
    "lookback_bars",
    "build_elapsed_ms",
    "recommendation",
    "data_leakage_check",
    "live_order_effect",
]
FAST_MICROSTRUCTURE_STRATEGY_CANDIDATE_FIELDS = [
    "generated_at",
    "feature_name",
    "symbol",
    "regime",
    "horizon_hours",
    "forward_sample_count",
    "rank_ic",
    "long_short_bps",
    "p25_net_bps",
    "hit_rate",
    "recent_7d_score",
    "lookback_bars",
    "candidate_strategy_id",
    "recommended_stage",
    "review_blocking_reasons",
    "data_leakage_check",
    "live_order_effect",
]
FAST_MICROSTRUCTURE_STRATEGY_REVIEW_BLOCKING_REASONS = (
    "needs_strategy_formulation",
    "needs_paper_tracking",
    "needs_cost_validation",
)
FAST_MICROSTRUCTURE_AGGREGATE_REVIEW_BLOCKING_REASONS = (
    *FAST_MICROSTRUCTURE_STRATEGY_REVIEW_BLOCKING_REASONS,
    "needs_specific_regime_validation",
)


def build_fast_microstructure_features(
    *,
    market_bars: pl.DataFrame,
    orderbook_spread_1m: pl.DataFrame | None = None,
    trade_activity_1m: pl.DataFrame | None = None,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    """Build read-only fast microstructure diagnostics from preloaded export frames."""

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    bars_by_symbol = _rows_by_symbol(market_bars, ts_fields=("ts", "minute_ts"))
    spread_frame = orderbook_spread_1m if orderbook_spread_1m is not None else pl.DataFrame()
    trade_frame = trade_activity_1m if trade_activity_1m is not None else pl.DataFrame()
    spreads_by_symbol = _rows_by_symbol(spread_frame, ts_fields=("minute_ts", "ts"))
    trades_by_symbol = _rows_by_symbol(
        trade_frame,
        ts_fields=("minute_ts", "latest_trade_ts", "ts"),
    )
    symbols = list(FAST_MICROSTRUCTURE_TARGET_SYMBOLS)
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        bars = bars_by_symbol.get(symbol, [])
        latest = bars[-1] if bars else None
        spread_rows = spreads_by_symbol.get(symbol, [])
        trade_rows = trades_by_symbol.get(symbol, [])
        if latest is None and not spread_rows and not trade_rows:
            rows.append(
                _diagnostic_row(
                    generated=generated,
                    symbol=symbol,
                    ts=None,
                    missing_reasons=["missing_market_bar"],
                )
            )
            continue
        ts = _feature_ts(latest, spread_rows, trade_rows)
        close = _latest_close(bars, ts) if ts is not None and bars else None
        if ts is None:
            rows.append(
                _diagnostic_row(
                    generated=generated,
                    symbol=symbol,
                    ts=ts,
                    missing_reasons=["missing_feature_ts_or_close"],
                )
            )
            continue
        missing_reasons: list[str] = []
        if latest is None:
            missing_reasons.append("missing_market_bar")
        if not spread_rows:
            missing_reasons.append("missing_orderbook_rollup")
        if not trade_rows:
            missing_reasons.append("missing_trade_rollup")
        spread_5m = _avg_spread_bps(spread_rows, ts, minutes=5)
        spread_15m = _avg_spread_bps(spread_rows, ts, minutes=15)
        spread_60m = _avg_spread_bps(spread_rows, ts, minutes=60)
        latest_spread = _latest_spread_bps(spread_rows, ts)
        orderbook_imbalance_1m = _latest_orderbook_imbalance(spread_rows, ts)
        orderbook_imbalance_5m = _avg_orderbook_imbalance(spread_rows, ts, minutes=5)
        if spread_rows and (
            latest_spread is None
            or orderbook_imbalance_1m is None
            or orderbook_imbalance_5m is None
        ):
            missing_reasons.append("missing_orderbook_core_fields")
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
        if trade_rows and (trade_5m is None or taker_imbalance_5m is None or cvd_5m is None):
            missing_reasons.append("missing_trade_core_fields")
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
        bid_depth_recovery = _bid_depth_recovery(spread_rows, ts)
        spread_normalization = None
        if latest_spread is not None and spread_15m is not None:
            spread_normalization = spread_15m - latest_spread
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
                "bid_depth_recovery": bid_depth_recovery,
                "spread_normalization": spread_normalization,
                "side_inferred": side_inferred,
                "cvd_size_15m": cvd,
                "vwap_1h": vwap_1h,
                "close_vs_vwap_1h_bps": close_vs_vwap,
                "return_1h_bps": ret_1h,
                "return_4h_bps": ret_4h,
                "realized_vol_4h_bps": vol_4h,
                "liquidity_quality": _liquidity_quality(spread_15m, trade_60m),
                "pressure_bias": _pressure_bias(ret_1h, close_vs_vwap, cvd),
                "missing_reason": ";".join(missing_reasons) if missing_reasons else "none",
                "response_action": "diagnostic_only",
                "live_order_effect": "read_only_no_live_order",
            }
        )
    if not rows:
        return _empty_frame()
    return pl.DataFrame(rows, infer_schema_length=None).select(FAST_MICROSTRUCTURE_FIELDS)


def build_fast_microstructure_forward_test(
    *,
    market_bars: pl.DataFrame,
    orderbook_spread_1m: pl.DataFrame | None = None,
    trade_activity_1m: pl.DataFrame | None = None,
    market_regime: pl.DataFrame | None = None,
    cost_bucket_daily: pl.DataFrame | None = None,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    """Validate fast microstructure diagnostics against future net returns.

    The returned frame is read-only research evidence. Future prices are used only
    after each feature timestamp to build labels.
    """

    started = perf_counter()
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    lookback_bars = _forward_lookback_bars()
    bars_by_symbol = _rows_by_symbol(market_bars, ts_fields=("ts", "minute_ts"))
    spreads_by_symbol = _rows_by_symbol(
        orderbook_spread_1m if orderbook_spread_1m is not None else pl.DataFrame(),
        ts_fields=("minute_ts", "ts"),
    )
    trades_by_symbol = _rows_by_symbol(
        trade_activity_1m if trade_activity_1m is not None else pl.DataFrame(),
        ts_fields=("minute_ts", "latest_trade_ts", "ts"),
    )
    regime_rows = _regime_rows(market_regime)
    samples: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}

    for symbol in FAST_MICROSTRUCTURE_TARGET_SYMBOLS:
        bars = bars_by_symbol.get(symbol, [])
        if not bars:
            continue
        cost = conservative_cost_for_symbol(cost_bucket_daily, symbol=symbol)
        spread_rows = spreads_by_symbol.get(symbol, [])
        trade_rows = trades_by_symbol.get(symbol, [])
        for bar in bars[-lookback_bars:]:
            ts = _coerce_dt(bar.get("_ts"))
            close = _float(bar.get("close"))
            if ts is None or close is None or close <= 0:
                continue
            feature_values = _microstructure_feature_values_at(
                bars=bars,
                spread_rows=spread_rows,
                trade_rows=trade_rows,
                ts=ts,
            )
            if not any(value is not None for value in feature_values.values()):
                continue
            regime = (
                _canonical_forward_regime(
                    _first_text(
                        bar,
                        (
                            "regime",
                            "regime_state",
                            "market_regime",
                            "current_regime",
                            "risk_level",
                        ),
                    )
                )
                or _regime_for_ts(regime_rows, ts)
                or _derived_regime(bars, ts)
            )
            for horizon in FAST_MICROSTRUCTURE_FORWARD_HORIZONS:
                future_net = _future_net_bps(
                    bars=bars,
                    ts=ts,
                    entry_px=close,
                    horizon_hours=horizon,
                    cost_bps=cost.cost_bps,
                )
                if future_net is None:
                    continue
                for feature_name, value in feature_values.items():
                    if value is None:
                        continue
                    sample = {"ts": ts, "value": value, "future_net_bps": future_net}
                    samples.setdefault((feature_name, symbol, regime, horizon), []).append(sample)
                    samples.setdefault(
                        (
                            feature_name,
                            symbol,
                            FAST_MICROSTRUCTURE_FORWARD_AGGREGATE_REGIME,
                            horizon,
                        ),
                        [],
                    ).append(sample)

    elapsed_ms = round((perf_counter() - started) * 1000.0, 3)
    out = [
        _forward_summary_row(
            generated,
            key,
            values,
            lookback_bars=lookback_bars,
            build_elapsed_ms=elapsed_ms,
        )
        for key, values in sorted(samples.items())
    ]
    if not out:
        return pl.DataFrame(
            schema={field: pl.Utf8 for field in FAST_MICROSTRUCTURE_FORWARD_TEST_FIELDS}
        )
    return pl.DataFrame(out, infer_schema_length=None).select(
        FAST_MICROSTRUCTURE_FORWARD_TEST_FIELDS
    )


def fast_microstructure_forward_summary_md(frame: pl.DataFrame) -> str:
    rows = frame.to_dicts() if frame is not None and not frame.is_empty() else []
    passed = [
        row for row in rows if str(row.get("recommendation") or "") == "FORWARD_VALIDATION_PASS"
    ]
    specific_passed = [row for row in passed if _is_specific_regime_forward_pass(row)]
    aggregate_passed = [row for row in passed if not _is_specific_regime_forward_pass(row)]
    lines = [
        "# Fast Microstructure Forward Test",
        "",
        "Read-only validation of fast microstructure diagnostics against future net returns.",
        "Future prices are used only after each feature timestamp for label construction.",
        "",
        f"- rows: {len(rows)}",
        f"- pass_rows: {len(passed)}",
        f"- aggregate_pass_rows: {len(aggregate_passed)}",
        f"- strategy_candidate_eligible_pass_rows: {len(specific_passed)}",
        "- live_order_effect: read_only_no_live_order",
    ]
    if passed and not specific_passed:
        lines.append(
            "- strategy_candidate_note: aggregate ALL_REGIMES passes stay validation-only; "
            "specific-regime passes are required before SHADOW_REVIEW candidates are emitted."
        )
    for row in passed[:12]:
        lines.append(
            "- "
            f"{row.get('feature_name')} {row.get('symbol')} {row.get('regime')} "
            f"h={row.get('horizon_hours')} rank_ic={row.get('rank_ic')} "
            f"long_short_bps={row.get('long_short_bps')}"
        )
    return "\n".join(lines) + "\n"


def build_fast_microstructure_strategy_candidates(
    fast_microstructure_forward_test: pl.DataFrame | None,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in _frame_rows(fast_microstructure_forward_test):
        if str(row.get("recommendation") or "") != "FORWARD_VALIDATION_PASS":
            continue
        feature_name = str(row.get("feature_name") or "").strip()
        symbol = normalize_symbol(row.get("symbol")) or ""
        regime = str(row.get("regime") or "").strip()
        horizon_hours = _int(row.get("horizon_hours"))
        if not feature_name or not symbol or not regime or horizon_hours is None:
            continue
        aggregate_only = regime.upper() == FAST_MICROSTRUCTURE_FORWARD_AGGREGATE_REGIME
        blocking_reasons = (
            FAST_MICROSTRUCTURE_AGGREGATE_REVIEW_BLOCKING_REASONS
            if aggregate_only
            else FAST_MICROSTRUCTURE_STRATEGY_REVIEW_BLOCKING_REASONS
        )
        rows.append(
            {
                "generated_at": row.get("generated_at"),
                "feature_name": feature_name,
                "symbol": symbol,
                "regime": regime,
                "horizon_hours": horizon_hours,
                "forward_sample_count": _int(row.get("sample_count")),
                "rank_ic": _round(_float(row.get("rank_ic"))),
                "long_short_bps": _round(_float(row.get("long_short_bps"))),
                "p25_net_bps": _round(_float(row.get("p25_net_bps"))),
                "hit_rate": _round(_float(row.get("hit_rate"))),
                "recent_7d_score": _round(_float(row.get("recent_7d_score"))),
                "lookback_bars": _int(row.get("lookback_bars")),
                "candidate_strategy_id": _fast_strategy_candidate_id(
                    feature_name=feature_name,
                    symbol=symbol,
                    regime=regime,
                    horizon_hours=horizon_hours,
                ),
                "recommended_stage": "VALIDATION_ONLY" if aggregate_only else "SHADOW_REVIEW",
                "review_blocking_reasons": json.dumps(
                    list(blocking_reasons),
                    separators=(",", ":"),
                ),
                "data_leakage_check": row.get("data_leakage_check") or "",
                "live_order_effect": "read_only_no_live_order",
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={field: pl.Utf8 for field in FAST_MICROSTRUCTURE_STRATEGY_CANDIDATE_FIELDS}
        )
    rows.sort(key=_fast_strategy_candidate_rank_key)
    return pl.DataFrame(rows, infer_schema_length=None).select(
        FAST_MICROSTRUCTURE_STRATEGY_CANDIDATE_FIELDS
    )


def _is_specific_regime_forward_pass(row: dict[str, Any]) -> bool:
    if str(row.get("recommendation") or "") != "FORWARD_VALIDATION_PASS":
        return False
    regime = str(row.get("regime") or "").strip().upper()
    return bool(regime) and regime != FAST_MICROSTRUCTURE_FORWARD_AGGREGATE_REGIME


def _frame_rows(frame: pl.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.is_empty():
        return []
    return frame.to_dicts()


def _fast_strategy_candidate_id(
    *,
    feature_name: str,
    symbol: str,
    regime: str,
    horizon_hours: int,
) -> str:
    return ".".join(
        [
            "v5",
            "fast_microstructure",
            _slug(feature_name),
            _slug(symbol),
            _slug(regime),
            f"{horizon_hours}h",
        ]
    )


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    out = [character if character.isalnum() else "_" for character in text]
    return "_".join(part for part in "".join(out).split("_") if part) or "unknown"


def _fast_strategy_candidate_rank_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("symbol") or ""),
        str(row.get("regime") or ""),
        _int(row.get("horizon_hours")) or 0,
        str(row.get("feature_name") or ""),
    )


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={field: pl.Utf8 for field in FAST_MICROSTRUCTURE_FIELDS})


def _diagnostic_row(
    *,
    generated: datetime,
    symbol: str,
    ts: datetime | None,
    missing_reasons: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {field: None for field in FAST_MICROSTRUCTURE_FIELDS}
    row.update(
        {
            "generated_at": generated.isoformat().replace("+00:00", "Z"),
            "schema_version": FAST_MICROSTRUCTURE_SCHEMA_VERSION,
            "symbol": symbol,
            "ts_utc": (ts or generated).isoformat().replace("+00:00", "Z"),
            "liquidity_quality": "not_observable",
            "pressure_bias": "not_observable",
            "missing_reason": ";".join(missing_reasons) if missing_reasons else "none",
            "response_action": "diagnostic_only",
            "live_order_effect": "read_only_no_live_order",
        }
    )
    return row


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
    latest_bar: dict[str, Any] | None,
    spread_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
) -> datetime | None:
    microstructure_candidates = [
        _coerce_dt(row.get("_ts"))
        for row in (spread_rows[-1:] + trade_rows[-1:])
    ]
    microstructure_candidates = [value for value in microstructure_candidates if value is not None]
    if microstructure_candidates:
        return max(microstructure_candidates)
    candidates = [_coerce_dt(latest_bar.get("_ts"))] if latest_bar is not None else []
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


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


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


def _bid_depth_recovery(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    latest = _latest_bid_depth(rows, ts)
    prior = _avg_bid_depth(rows, ts - timedelta(minutes=5), minutes=15)
    if latest is None or prior is None or prior <= 0:
        return None
    return latest / prior - 1.0


def _latest_bid_depth(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    candidates = [
        _bid_depth(row)
        for row in rows
        if row.get("_ts") <= ts and _bid_depth(row) is not None
    ]
    return candidates[-1] if candidates else None


def _avg_bid_depth(rows: list[dict[str, Any]], ts: datetime, *, minutes: int) -> float | None:
    values = [_bid_depth(row) for row in _window_rows(rows, ts, minutes=minutes)]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _bid_depth(row: dict[str, Any]) -> float | None:
    return _first_float(
        row,
        (
            "bid_depth",
            "bid_depth_1pct",
            "bid_depth_usdt",
            "bid_size",
            "best_bid_size",
            "bid_qty",
            "bid_sz",
        ),
    )


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


def _microstructure_feature_values_at(
    *,
    bars: list[dict[str, Any]],
    spread_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    ts: datetime,
) -> dict[str, float | None]:
    latest_spread = _latest_spread_bps(spread_rows, ts)
    spread_5m_prior = _spread_at_or_before(spread_rows, ts - timedelta(minutes=5))
    spread_bps_change_5m = None
    if latest_spread is not None and spread_5m_prior is not None:
        spread_bps_change_5m = latest_spread - spread_5m_prior
    taker_buy_5m, taker_sell_5m, _side_inferred = _buy_sell_sums(
        trade_rows,
        spread_rows,
        ts,
        minutes=5,
    )
    cvd_5m = None
    taker_imbalance_5m = None
    if taker_buy_5m is not None and taker_sell_5m is not None:
        cvd_5m = taker_buy_5m - taker_sell_5m
        total = taker_buy_5m + taker_sell_5m
        if total > 0:
            taker_imbalance_5m = cvd_5m / total
    return_1h = _return_bps(bars, ts, hours=1)
    return {
        "orderbook_imbalance_1m": _latest_orderbook_imbalance(spread_rows, ts),
        "orderbook_imbalance_5m": _avg_orderbook_imbalance(spread_rows, ts, minutes=5),
        "taker_buy_sell_imbalance_5m": taker_imbalance_5m,
        "cvd_5m": cvd_5m,
        "cvd_divergence": _cvd_divergence(return_1h, taker_imbalance_5m),
        "spread_bps_change_5m": spread_bps_change_5m,
    }


def _future_net_bps(
    *,
    bars: list[dict[str, Any]],
    ts: datetime,
    entry_px: float,
    horizon_hours: int,
    cost_bps: float,
) -> float | None:
    future_px = _close_at_or_after(bars, ts + timedelta(hours=horizon_hours))
    if future_px is None or future_px <= 0 or entry_px <= 0:
        return None
    return (future_px / entry_px - 1.0) * 10000.0 - cost_bps


def _close_at_or_after(rows: list[dict[str, Any]], ts: datetime) -> float | None:
    for row in rows:
        row_ts = _coerce_dt(row.get("_ts"))
        value = _float(row.get("close"))
        if row_ts is not None and row_ts >= ts and value is not None:
            return value
    return None


def _regime_rows(frame: pl.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.is_empty():
        return []
    out: list[dict[str, Any]] = []
    for raw in frame.to_dicts():
        ts = _first_ts(raw, ("as_of_ts", "created_at", "ts", "date", "as_of_date"))
        if ts is None:
            continue
        regime = _canonical_forward_regime(
            _first_text(raw, ("current_regime", "regime_state", "market_regime", "state"))
        )
        if regime is None:
            continue
        item = dict(raw)
        item["_ts"] = ts
        item["_regime"] = regime
        out.append(item)
    out.sort(key=lambda row: row["_ts"])
    return out


def _regime_for_ts(regime_rows: list[dict[str, Any]], ts: datetime) -> str | None:
    candidates = [row for row in regime_rows if row["_ts"] <= ts]
    if not candidates:
        return None
    return str(candidates[-1].get("_regime") or "") or None


def _derived_regime(bars: list[dict[str, Any]], ts: datetime) -> str:
    ret_4h = _return_bps(bars, ts, hours=4)
    if ret_4h is None:
        return "SIDEWAYS"
    if ret_4h <= -80:
        return "RISK_OFF"
    if ret_4h >= 120:
        return "RISK_ON_CONFIRMED"
    if ret_4h >= 50:
        return "TREND_UP"
    return "SIDEWAYS"


def _canonical_forward_regime(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text or text in {"UNKNOWN", "NONE", "NAN"}:
        return None
    if "RISK_ON_CONFIRMED" in text:
        return "RISK_ON_CONFIRMED"
    if "RISK_ON" in text or "IMPULSE" in text:
        return "RISK_ON_CONFIRMED"
    if "RISK_OFF" in text or "DOWN" in text:
        return "RISK_OFF"
    if "UP" in text or text in {"TREND", "TRENDING"}:
        return "TREND_UP"
    if "SIDE" in text or "CHOP" in text or "PROTECT" in text or "LOW_VOL" in text:
        return "SIDEWAYS"
    if text in FAST_MICROSTRUCTURE_FORWARD_REGIMES:
        return text
    return "SIDEWAYS"


def _first_text(row: dict[str, Any], fields: Iterable[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value not in (None, "", "not_observable", "unknown", "UNKNOWN", "nan"):
            return str(value)
    return None


def _forward_summary_row(
    generated: datetime,
    key: tuple[str, str, str, int],
    samples: list[dict[str, Any]],
    *,
    lookback_bars: int,
    build_elapsed_ms: float,
) -> dict[str, Any]:
    feature_name, symbol, regime, horizon = key
    values = [_float(sample.get("value")) for sample in samples]
    labels = [_float(sample.get("future_net_bps")) for sample in samples]
    pairs = [
        (sample["ts"], value, label)
        for sample, value, label in zip(samples, values, labels, strict=False)
        if value is not None and label is not None
    ]
    labels_only = [label for _ts, _value, label in pairs]
    long_short, p25, hit_rate = _top_bottom_stats(pairs)
    rank_ic = _rank_ic([value for _ts, value, _label in pairs], labels_only)
    recent_score = _recent_score(pairs)
    recommendation = _forward_recommendation(
        sample_count=len(pairs),
        rank_ic=rank_ic,
        long_short_bps=long_short,
        p25_net_bps=p25,
        hit_rate=hit_rate,
    )
    return {
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "feature_name": feature_name,
        "symbol": symbol,
        "regime": regime,
        "horizon_hours": horizon,
        "sample_count": len(pairs),
        "rank_ic": _round(rank_ic),
        "long_short_bps": _round(long_short),
        "p25_net_bps": _round(p25),
        "hit_rate": _round(hit_rate),
        "recent_7d_score": _round(recent_score),
        "lookback_bars": int(lookback_bars),
        "build_elapsed_ms": float(build_elapsed_ms),
        "recommendation": recommendation,
        "data_leakage_check": "pass_future_prices_used_only_for_labels",
        "live_order_effect": "read_only_no_live_order",
    }


def _forward_lookback_bars() -> int:
    raw = (
        os.environ.get("FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS")
        or os.environ.get("QUANT_LAB_FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS")
        or str(FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS)
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS must be an integer") from exc
    if value < 1:
        raise ValueError("FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS must be >= 1")
    return value


def _top_bottom_stats(
    pairs: list[tuple[datetime, float, float]],
    *,
    quantile: float = 0.2,
) -> tuple[float | None, float | None, float | None]:
    if not pairs:
        return None, None, None
    ordered = sorted(pairs, key=lambda item: item[1])
    bucket_size = max(1, int(len(ordered) * quantile))
    bottom = [label for _ts, _value, label in ordered[:bucket_size]]
    top = [label for _ts, _value, label in ordered[-bucket_size:]]
    long_short = _mean(top) - _mean(bottom) if top and bottom else None
    p25 = _percentile(top, 0.25) if top else None
    hit_rate = sum(1 for value in top if value > 0) / len(top) if top else None
    return long_short, p25, hit_rate


def _rank_ic(values: list[float], labels: list[float]) -> float | None:
    if len(values) < 3 or len(values) != len(labels):
        return None
    return _pearson(_ranks(values), _ranks(labels))


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    for rank, (index, _value) in enumerate(ordered, start=1):
        ranks[index] = float(rank)
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=False))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = (left_var * right_var) ** 0.5
    if denominator <= 0:
        return None
    return numerator / denominator


def _recent_score(pairs: list[tuple[datetime, float, float]]) -> float | None:
    if not pairs:
        return None
    latest = max(ts for ts, _value, _label in pairs)
    recent = [label for ts, _value, label in pairs if ts >= latest - timedelta(days=7)]
    return _mean(recent) if recent else None


def _forward_recommendation(
    *,
    sample_count: int,
    rank_ic: float | None,
    long_short_bps: float | None,
    p25_net_bps: float | None,
    hit_rate: float | None,
) -> str:
    if sample_count < 30:
        return "NEEDS_MORE_FORWARD_SAMPLES"
    if (
        (rank_ic or 0.0) > 0.02
        and (long_short_bps or 0.0) > 0
        and (p25_net_bps is not None and p25_net_bps > -50)
        and (hit_rate or 0.0) > 0.50
    ):
        return "FORWARD_VALIDATION_PASS"
    return "FORWARD_VALIDATION_WEAK_OR_MIXED"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


def _round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
