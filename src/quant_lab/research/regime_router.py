from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, read_parquet_lazy, write_parquet_dataset
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

MARKET_REGIME_DATASET = Path("gold") / "market_regime_daily"
STRATEGY_REGIME_MATRIX_DATASET = Path("gold") / "strategy_regime_matrix"
REGIME_STRATEGY_ADVISORY_DATASET = Path("gold") / "regime_strategy_advisory"

SOURCE_NAME = "research.regime_router.v0.1"
MARKET_REGIME_SCHEMA_VERSION = "market_regime_daily.v0.1"
STRATEGY_REGIME_MATRIX_SCHEMA_VERSION = "strategy_regime_matrix.v0.1"
REGIME_ADVISORY_SCHEMA_VERSION = "regime_strategy_advisory.v0.1"

MAJOR_SYMBOLS = ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT")
REGIME_MATCHES: dict[str, set[str]] = {
    "TREND_UP": {"TREND_UP", "TREND", "ALT_IMPULSE", "SOL_STRENGTH"},
    "TREND_DOWN": {"TREND_DOWN", "RISK_OFF"},
    "SIDEWAYS": {"SIDEWAYS", "CHOP", "PROTECT", "LOW_VOL"},
    "HIGH_VOL": {"HIGH_VOL", "TREND_UP", "TREND_DOWN", "ALT_IMPULSE"},
    "LOW_VOL": {"LOW_VOL", "SIDEWAYS", "CHOP"},
    "RISK_OFF": {"RISK_OFF", "TREND_DOWN"},
    "ALT_IMPULSE": {"ALT_IMPULSE", "IMPULSE", "TREND_UP", "TREND"},
    "BTC_LEADERSHIP": {"BTC_LEADERSHIP", "TREND_UP", "TREND"},
    "SOL_STRENGTH": {"SOL_STRENGTH", "TREND_UP", "TREND", "PROTECT"},
    "LIQUIDITY_THIN": {"LIQUIDITY_THIN"},
}

MARKET_REGIME_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "current_regime": pl.Utf8,
    "active_regimes_json": pl.Utf8,
    "regime_confidence": pl.Float64,
    "btc_24h_return_bps": pl.Float64,
    "eth_24h_return_bps": pl.Float64,
    "sol_24h_return_bps": pl.Float64,
    "bnb_24h_return_bps": pl.Float64,
    "broad_market_positive_count": pl.Int64,
    "realized_vol_bps": pl.Float64,
    "avg_spread_bps": pl.Float64,
    "liquidity_thin": pl.Boolean,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}

STRATEGY_REGIME_MATRIX_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "raw_strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "regime_state": pl.Utf8,
    "canonical_regime": pl.Utf8,
    "horizon_hours": pl.Int64,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "median_net_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "mae_bps": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}

REGIME_ADVISORY_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "current_regime": pl.Utf8,
    "active_regimes_json": pl.Utf8,
    "allowed_strategy_candidates": pl.Utf8,
    "blocked_strategy_candidates": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "regime_confidence": pl.Float64,
    "live_block_reasons": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}


class RegimeRouterBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    market_regime_rows: int = Field(ge=0)
    strategy_regime_matrix_rows: int = Field(ge=0)
    regime_strategy_advisory_rows: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_regime_router(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> RegimeRouterBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    market_regime = build_market_regime_daily(root, as_of_date=day)
    matrix = build_strategy_regime_matrix(root, as_of_date=day)
    advisory = build_regime_strategy_advisory(market_regime, matrix, as_of_date=day)
    market_rows = _replace_as_of_date(root / MARKET_REGIME_DATASET, market_regime, day)
    matrix_rows = _replace_as_of_date(root / STRATEGY_REGIME_MATRIX_DATASET, matrix, day)
    advisory_rows = _replace_as_of_date(root / REGIME_STRATEGY_ADVISORY_DATASET, advisory, day)
    warnings: list[str] = []
    if market_regime.is_empty():
        warnings.append("market_regime_empty")
    if matrix.is_empty():
        warnings.append("strategy_regime_matrix_empty")
    if advisory.is_empty():
        warnings.append("regime_strategy_advisory_empty")
    return RegimeRouterBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        market_regime_rows=market_rows,
        strategy_regime_matrix_rows=matrix_rows,
        regime_strategy_advisory_rows=advisory_rows,
        warnings=warnings,
    )


def build_market_regime_daily(lake_root: str | Path, *, as_of_date: date) -> pl.DataFrame:
    created_at = datetime.now(UTC)
    bars = _recent_market_bars(Path(lake_root), as_of_date=as_of_date)
    if bars.is_empty():
        return pl.DataFrame(schema=MARKET_REGIME_SCHEMA)
    rows = _latest_major_market_rows(bars, as_of_date=as_of_date)
    returns = {symbol: _return_bps(symbol_rows) for symbol, symbol_rows in rows.items()}
    broad_positive = sum(1 for value in returns.values() if value is not None and value > 0)
    realized_vol = _realized_vol_bps(rows)
    spread = _avg_spread_bps(Path(lake_root), as_of_date=as_of_date)
    active = _active_regimes(returns, realized_vol, spread)
    current = _primary_regime(active)
    confidence = _regime_confidence(active, returns, realized_vol, spread)
    payload = {
        "as_of_date": as_of_date.isoformat(),
        "current_regime": current,
        "active_regimes_json": safe_json_dumps(active),
        "regime_confidence": confidence,
        "btc_24h_return_bps": returns.get("BTC-USDT"),
        "eth_24h_return_bps": returns.get("ETH-USDT"),
        "sol_24h_return_bps": returns.get("SOL-USDT"),
        "bnb_24h_return_bps": returns.get("BNB-USDT"),
        "broad_market_positive_count": broad_positive,
        "realized_vol_bps": realized_vol,
        "avg_spread_bps": spread,
        "liquidity_thin": bool(spread is not None and spread > 20.0),
        "created_at": created_at,
        "source": SOURCE_NAME,
        "schema_version": MARKET_REGIME_SCHEMA_VERSION,
    }
    return pl.DataFrame([payload], schema=MARKET_REGIME_SCHEMA, orient="row")


def build_strategy_regime_matrix(lake_root: str | Path, *, as_of_date: date) -> pl.DataFrame:
    root = Path(lake_root)
    created_at = datetime.now(UTC)
    summary = _read_latest_as_of_dataset(root / "gold" / "strategy_evidence", as_of_date)
    board = _read_latest_as_of_dataset(root / "gold" / "alpha_discovery_board", as_of_date)
    pullback = _read_latest_as_of_dataset(
        root / "gold" / "v5_pullback_reversal_shadow",
        as_of_date,
    )
    rows: list[dict[str, Any]] = []
    rows.extend(_matrix_rows_from_summary(summary, created_at=created_at, as_of_date=as_of_date))
    rows.extend(_matrix_rows_from_board(board, created_at=created_at, as_of_date=as_of_date))
    rows.extend(_matrix_rows_from_pullback(pullback, created_at=created_at, as_of_date=as_of_date))
    rows = _dedupe_matrix_rows(rows)
    if not rows:
        return pl.DataFrame(schema=STRATEGY_REGIME_MATRIX_SCHEMA)
    return pl.DataFrame(rows, schema=STRATEGY_REGIME_MATRIX_SCHEMA, orient="row")


def build_regime_strategy_advisory(
    market_regime: pl.DataFrame,
    matrix: pl.DataFrame,
    *,
    as_of_date: date,
) -> pl.DataFrame:
    if market_regime.is_empty():
        return pl.DataFrame(schema=REGIME_ADVISORY_SCHEMA)
    regime_row = market_regime.to_dicts()[0]
    current = str(regime_row.get("current_regime") or "SIDEWAYS").upper()
    active = _json_list(regime_row.get("active_regimes_json")) or [current]
    confidence = _float(regime_row.get("regime_confidence")) or 0.0
    allowed: list[str] = []
    blocked: list[str] = []
    reasons = {"regime_router_read_only", "not_live_validated", "no_live_small_from_regime_router"}
    if confidence < 0.60:
        reasons.add("low_regime_confidence")
    if not matrix.is_empty():
        for row in matrix.to_dicts():
            candidate = str(row.get("strategy_candidate") or "").strip()
            if not candidate:
                continue
            decision, item_reasons = _regime_candidate_decision(row, current, active, confidence)
            reasons.update(item_reasons)
            if decision in {"KILL", "BLOCKED"}:
                blocked.append(candidate)
            elif decision in {"PAPER_READY", "KEEP_SHADOW", "REGIME_SHADOW"}:
                allowed.append(candidate)
            else:
                blocked.append(candidate)
    allowed = sorted(set(allowed))
    blocked = sorted(set(blocked) - set(allowed))
    mode = _regime_recommended_mode(allowed, matrix, confidence)
    payload = {
        "as_of_date": as_of_date.isoformat(),
        "current_regime": current,
        "active_regimes_json": safe_json_dumps(active),
        "allowed_strategy_candidates": safe_json_dumps(allowed),
        "blocked_strategy_candidates": safe_json_dumps(blocked),
        "recommended_mode": mode,
        "regime_confidence": confidence,
        "live_block_reasons": safe_json_dumps(sorted(reasons)),
        "created_at": datetime.now(UTC),
        "source": SOURCE_NAME,
        "schema_version": REGIME_ADVISORY_SCHEMA_VERSION,
    }
    return pl.DataFrame([payload], schema=REGIME_ADVISORY_SCHEMA, orient="row")


def _latest_major_market_rows(
    bars: pl.DataFrame,
    *,
    as_of_date: date,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in MAJOR_SYMBOLS}
    end = datetime.combine(as_of_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    start = end - timedelta(hours=48)
    for row in bars.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if symbol not in rows_by_symbol:
            continue
        ts = _dt(row.get("ts"))
        if ts is None or ts < start or ts >= end:
            continue
        if str(row.get("timeframe") or "1H").upper() not in {"1H", "60M", "1HOUR"}:
            continue
        rows_by_symbol[symbol].append(row | {"_ts": ts})
    for symbol in rows_by_symbol:
        rows_by_symbol[symbol].sort(key=lambda item: item["_ts"])
    return rows_by_symbol


def _recent_market_bars(lake_root: Path, *, as_of_date: date) -> pl.DataFrame:
    lazy, columns = _safe_lazy_frame(lake_root / "silver" / "market_bar")
    if lazy is None or not {"symbol", "ts", "close"}.issubset(columns):
        return pl.DataFrame()
    end = datetime.combine(as_of_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    start = end - timedelta(hours=48)
    frame = lazy.with_columns(
        [
            _lazy_datetime("ts").alias("_ts"),
            _lazy_normalized_symbol("symbol").alias("_symbol"),
        ]
    ).filter(
        (pl.col("_symbol").is_in(MAJOR_SYMBOLS))
        & (pl.col("_ts") >= start)
        & (pl.col("_ts") < end)
    )
    if "timeframe" in columns:
        frame = frame.with_columns(
            pl.col("timeframe").cast(pl.Utf8).str.to_uppercase().alias("_timeframe")
        ).filter(pl.col("_timeframe").is_in(["1H", "60M", "1HOUR"]))
    select_columns = [
        column
        for column in ["symbol", "timeframe", "ts", "open", "high", "low", "close", "volume"]
        if column in columns
    ]
    return _collect_lazy(frame.select([*select_columns, "_ts"]))


def _return_bps(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return None
    latest = _float(rows[-1].get("close"))
    start_index = max(0, len(rows) - 25)
    base = _float(rows[start_index].get("close"))
    if latest is None or base is None or base <= 0:
        return None
    return (latest / base - 1.0) * 10000.0


def _realized_vol_bps(rows_by_symbol: dict[str, list[dict[str, Any]]]) -> float | None:
    btc_rows = rows_by_symbol.get("BTC-USDT", [])
    returns: list[float] = []
    for prev, current in zip(btc_rows, btc_rows[1:], strict=False):
        prev_close = _float(prev.get("close"))
        close = _float(current.get("close"))
        if prev_close and close:
            returns.append((close / prev_close - 1.0) * 10000.0)
    if len(returns) < 4:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return variance**0.5


def _avg_spread_bps(lake_root: Path, *, as_of_date: date) -> float | None:
    rollup_spread = _avg_spread_bps_from_rollup(lake_root, as_of_date=as_of_date)
    if rollup_spread is not None:
        return rollup_spread
    lazy, columns = _safe_lazy_frame(lake_root / "silver" / "orderbook_snapshot")
    if lazy is None or "symbol" not in columns:
        return None
    end = datetime.combine(as_of_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    start = end - timedelta(hours=24)
    timestamp_expressions = [
        _lazy_datetime(column) for column in ["ts", "ingest_ts"] if column in columns
    ]
    if not timestamp_expressions:
        return None
    spread_expression = _spread_bps_expression(columns)
    if spread_expression is None:
        return None
    frame = lazy.with_columns(
        [
            pl.coalesce(timestamp_expressions).alias("_ts"),
            _lazy_normalized_symbol("symbol").alias("_symbol"),
            spread_expression.alias("_spread_bps"),
        ]
    ).filter(
        (pl.col("_symbol").is_in(MAJOR_SYMBOLS))
        & (pl.col("_ts") >= start)
        & (pl.col("_ts") < end)
        & pl.col("_spread_bps").is_not_null()
    )
    mean_frame = _collect_lazy(
        frame.select(pl.col("_spread_bps").mean().alias("avg_spread_bps"))
    )
    if mean_frame.is_empty():
        return None
    return _float(mean_frame.item(0, "avg_spread_bps"))


def _avg_spread_bps_from_rollup(lake_root: Path, *, as_of_date: date) -> float | None:
    lazy, columns = _safe_lazy_frame(lake_root / "silver" / "orderbook_spread_1m")
    if lazy is None or "symbol" not in columns or "spread_bps" not in columns:
        return None
    end = datetime.combine(as_of_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    start = end - timedelta(hours=24)
    timestamp_expressions = [
        _lazy_datetime(column) for column in ["ts", "minute_ts"] if column in columns
    ]
    if not timestamp_expressions:
        return None
    frame = lazy.with_columns(
        [
            pl.coalesce(timestamp_expressions).alias("_ts"),
            _lazy_normalized_symbol("symbol").alias("_symbol"),
            pl.col("spread_bps").cast(pl.Float64, strict=False).alias("_spread_bps"),
        ]
    ).filter(
        (pl.col("_symbol").is_in(MAJOR_SYMBOLS))
        & (pl.col("_ts") >= start)
        & (pl.col("_ts") < end)
        & pl.col("_spread_bps").is_not_null()
    )
    mean_frame = _collect_lazy(
        frame.select(pl.col("_spread_bps").mean().alias("avg_spread_bps"))
    )
    if mean_frame.is_empty():
        return None
    return _float(mean_frame.item(0, "avg_spread_bps"))


def _active_regimes(
    returns: dict[str, float | None],
    realized_vol: float | None,
    spread: float | None,
) -> list[str]:
    btc = returns.get("BTC-USDT") or 0.0
    eth = returns.get("ETH-USDT") or 0.0
    sol = returns.get("SOL-USDT") or 0.0
    bnb = returns.get("BNB-USDT") or 0.0
    broad_positive = sum(1 for value in [btc, eth, sol, bnb] if value > 0.0)
    active: list[str] = []
    if btc <= -300 or (broad_positive <= 1 and sum([btc, eth, sol, bnb]) / 4.0 <= -180):
        active.append("RISK_OFF")
    if btc >= 150 or broad_positive >= 3:
        active.append("TREND_UP")
    elif btc <= -150:
        active.append("TREND_DOWN")
    else:
        active.append("SIDEWAYS")
    if realized_vol is not None:
        active.append("HIGH_VOL" if realized_vol >= 90 else "LOW_VOL")
    if sol >= max(btc, eth, bnb) + 100 and sol > 150:
        active.append("SOL_STRENGTH")
    alt_avg = (eth + sol + bnb) / 3.0
    if alt_avg > btc + 80 and sum(1 for value in [eth, sol, bnb] if value > 120) >= 2:
        active.append("ALT_IMPULSE")
    if btc > alt_avg + 120 and btc > 150:
        active.append("BTC_LEADERSHIP")
    if spread is not None and spread > 20.0:
        active.append("LIQUIDITY_THIN")
    return _unique_preserve_order(active)


def _primary_regime(active: list[str]) -> str:
    for candidate in [
        "RISK_OFF",
        "ALT_IMPULSE",
        "SOL_STRENGTH",
        "BTC_LEADERSHIP",
        "TREND_UP",
        "TREND_DOWN",
        "SIDEWAYS",
        "HIGH_VOL",
        "LOW_VOL",
        "LIQUIDITY_THIN",
    ]:
        if candidate in active:
            return candidate
    return "SIDEWAYS"


def _regime_confidence(
    active: list[str],
    returns: dict[str, float | None],
    realized_vol: float | None,
    spread: float | None,
) -> float:
    observed_returns = [value for value in returns.values() if value is not None]
    if len(observed_returns) < 2:
        return 0.35
    score = 0.55
    if any(abs(value) > 150 for value in observed_returns):
        score += 0.15
    if "RISK_OFF" in active or "ALT_IMPULSE" in active or "SOL_STRENGTH" in active:
        score += 0.10
    if realized_vol is not None:
        score += 0.05
    if spread is not None:
        score += 0.05
    return min(score, 0.9)


def _matrix_rows_from_summary(
    frame: pl.DataFrame,
    *,
    created_at: datetime,
    as_of_date: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts() if not frame.is_empty() else []:
        rows.append(_matrix_row_from_aggregate(row, created_at=created_at, as_of_date=as_of_date))
    return rows


def _matrix_rows_from_board(
    frame: pl.DataFrame,
    *,
    created_at: datetime,
    as_of_date: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts() if not frame.is_empty() else []:
        rows.append(_matrix_row_from_aggregate(row, created_at=created_at, as_of_date=as_of_date))
    return rows


def _matrix_rows_from_pullback(
    frame: pl.DataFrame,
    *,
    created_at: datetime,
    as_of_date: date,
) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    horizon_rows = [
        row
        for row in frame.to_dicts()
        if int(_float(row.get("horizon_hours")) or 0) == 24
        and str(row.get("rule_version") or "") == "confirmed_reversal_v0.2"
    ]
    output: list[dict[str, Any]] = []
    for (symbol, regime), group in _group_by(horizon_rows, ["symbol", "regime_state"]).items():
        net = [_float(row.get("net_bps_after_cost")) for row in group]
        net_values = [value for value in net if value is not None]
        mae = [_float(row.get("mae_bps")) for row in group]
        mae_values = [value for value in mae if value is not None]
        output.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "strategy_candidate": "pullback_reversal_v2",
                "raw_strategy_candidate": "v5.pullback_reversal_shadow",
                "symbol": normalize_symbol(symbol) or str(symbol or "UNKNOWN"),
                "regime_state": str(regime or "UNKNOWN"),
                "canonical_regime": _canonical_regime(regime),
                "horizon_hours": 24,
                "sample_count": len(group),
                "complete_sample_count": len(net_values),
                "avg_net_bps": _mean(net_values),
                "median_net_bps": _quantile(net_values, 0.50),
                "p25_net_bps": _quantile(net_values, 0.25),
                "win_rate": _win_rate(net_values),
                "mae_bps": _mean(mae_values),
                "cost_source_mix": safe_json_dumps(
                    _counter(row.get("cost_quality") for row in group)
                ),
                "decision": _matrix_decision(
                    "pullback_reversal_v2",
                    _canonical_regime(regime),
                    len(group),
                    len(net_values),
                    _mean(net_values),
                    _win_rate(net_values),
                ),
                "decision_reasons": "[]",
                "created_at": created_at,
                "source": SOURCE_NAME,
                "schema_version": STRATEGY_REGIME_MATRIX_SCHEMA_VERSION,
            }
        )
    return output


def _matrix_row_from_aggregate(
    row: dict[str, Any],
    *,
    created_at: datetime,
    as_of_date: date,
) -> dict[str, Any]:
    raw_candidate = str(row.get("strategy_candidate") or row.get("candidate_name") or "").strip()
    symbol = normalize_symbol(row.get("symbol")) or str(row.get("symbol") or "UNKNOWN")
    regime = str(row.get("regime_state") or row.get("regime") or "UNKNOWN")
    candidate = _router_candidate_name(raw_candidate, symbol)
    canonical = _canonical_regime(regime)
    sample_count = int(_float(row.get("sample_count")) or 0)
    complete = int(_float(row.get("complete_sample_count")) or 0)
    avg = _float(row.get("avg_net_bps"))
    win = _float(row.get("win_rate"))
    decision = _matrix_decision(
        candidate,
        canonical,
        sample_count,
        complete,
        avg,
        win,
        original_decision=str(row.get("decision") or ""),
    )
    return {
        "as_of_date": str(row.get("as_of_date") or as_of_date.isoformat()),
        "strategy_candidate": candidate,
        "raw_strategy_candidate": raw_candidate,
        "symbol": symbol,
        "regime_state": regime,
        "canonical_regime": canonical,
        "horizon_hours": int(_float(row.get("horizon_hours")) or 0),
        "sample_count": sample_count,
        "complete_sample_count": complete,
        "avg_net_bps": avg,
        "median_net_bps": _float(row.get("median_net_bps")),
        "p25_net_bps": _float(row.get("p25_net_bps")),
        "win_rate": win,
        "mae_bps": _float(row.get("mae_bps") or row.get("avg_mae_bps")),
        "cost_source_mix": str(row.get("cost_source_mix") or "{}"),
        "decision": decision,
        "decision_reasons": str(row.get("decision_reasons") or "[]"),
        "created_at": created_at,
        "source": SOURCE_NAME,
        "schema_version": STRATEGY_REGIME_MATRIX_SCHEMA_VERSION,
    }


def _router_candidate_name(raw_candidate: str, symbol: str) -> str:
    if raw_candidate == "v5.sol_protect_alpha6_low_exception":
        return "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1"
    if raw_candidate == "v5.f4_volume_expansion_entry" and symbol == "SOL-USDT":
        return "SOL_F4_VOLUME_EXPANSION_PAPER_V1"
    if raw_candidate == "v5.f3_dominant_entry" and symbol == "ETH-USDT":
        return "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
    if raw_candidate == "v5.alt_impulse_shadow":
        return "ALT_IMPULSE_REGIME_SHADOW_V1"
    if raw_candidate in {"v5.btc_leadership_probe_strict", "v5.btc_broad_leadership"}:
        return "BTC_STRICT_PROBE_MONITOR_V1"
    if raw_candidate == "v5.late_entry_chase_guard_shadow":
        return "late_entry_chase_guard_shadow"
    if raw_candidate.startswith("v5.pullback_reversal_shadow"):
        return "pullback_reversal_v2"
    return raw_candidate or "UNKNOWN"


def _matrix_decision(
    candidate: str,
    canonical_regime: str,
    sample_count: int,
    complete_count: int,
    avg_net_bps: float | None,
    win_rate: float | None,
    *,
    original_decision: str = "",
) -> str:
    original = original_decision.strip().upper()
    if candidate == "BTC_STRICT_PROBE_MONITOR_V1":
        return "KILL"
    if candidate == "pullback_reversal_v2" and canonical_regime in {"TREND_DOWN", "RISK_OFF"}:
        return "KILL"
    if sample_count < 10 or complete_count < 5:
        return "RESEARCH_ONLY"
    if candidate == "ALT_IMPULSE_REGIME_SHADOW_V1":
        if canonical_regime in {"ALT_IMPULSE", "TREND_UP"} and (avg_net_bps or 0.0) > 0:
            return "PAPER_READY" if complete_count >= 10 else "REGIME_SHADOW"
        return "REGIME_SHADOW"
    if original in {"KILL", "DEAD"}:
        return "KILL"
    if avg_net_bps is not None and avg_net_bps < 0 and (win_rate is None or win_rate < 0.45):
        return "KILL"
    if original == "LIVE_SMALL_READY":
        return "PAPER_READY"
    if original in {"PAPER_READY", "KEEP_SHADOW", "REGIME_SHADOW"}:
        return original
    if avg_net_bps is not None and avg_net_bps > 0:
        return "KEEP_SHADOW"
    return "RESEARCH_ONLY"


def _regime_candidate_decision(
    row: dict[str, Any],
    current: str,
    active: list[str],
    confidence: float,
) -> tuple[str, list[str]]:
    candidate = str(row.get("strategy_candidate") or "")
    row_regime = str(row.get("canonical_regime") or "UNKNOWN").upper()
    decision = str(row.get("decision") or "RESEARCH_ONLY").upper()
    reasons: list[str] = []
    if candidate == "pullback_reversal_v2" and current in {"TREND_DOWN", "RISK_OFF"}:
        return "BLOCKED", ["pullback_reversal_blocked_in_down_or_risk_off"]
    if candidate == "ALT_IMPULSE_REGIME_SHADOW_V1" and not (
        current in {"ALT_IMPULSE", "TREND_UP"} or {"ALT_IMPULSE", "TREND_UP"} & set(active)
    ):
        return "BLOCKED", ["alt_impulse_only_validated_in_alt_impulse_or_trend_up"]
    if not _regime_matches(current, row_regime, str(row.get("symbol") or "")):
        return "BLOCKED", ["strategy_regime_not_current"]
    sample_count = int(_float(row.get("sample_count")) or 0)
    complete = int(_float(row.get("complete_sample_count")) or 0)
    if sample_count < 10 or complete < 5:
        return "RESEARCH_ONLY", ["strategy_current_regime_samples_insufficient"]
    if confidence < 0.60 and decision == "PAPER_READY":
        return "KEEP_SHADOW", ["low_regime_confidence_shadow_only"]
    if decision == "LIVE_SMALL_READY":
        return "PAPER_READY", ["regime_router_never_live_small"]
    return decision, reasons


def _regime_matches(current: str, row_regime: str, symbol: str) -> bool:
    if row_regime in REGIME_MATCHES.get(current, {current}):
        return True
    if current == "SOL_STRENGTH" and symbol == "SOL-USDT":
        return row_regime in {"TREND_UP", "TREND", "PROTECT", "SOL_STRENGTH"}
    if current == "ALT_IMPULSE" and row_regime in {"IMPULSE", "ALT_IMPULSE", "TREND"}:
        return True
    return False


def _regime_recommended_mode(allowed: list[str], matrix: pl.DataFrame, confidence: float) -> str:
    if not allowed:
        return "research"
    if confidence < 0.60:
        return "shadow"
    decisions = {
        str(row.get("decision") or "").upper()
        for row in matrix.to_dicts()
        if str(row.get("strategy_candidate") or "") in set(allowed)
    }
    if "PAPER_READY" in decisions:
        return "paper"
    return "shadow"


def _canonical_regime(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text == "UNKNOWN":
        return "UNKNOWN"
    if "RISK_OFF" in text:
        return "RISK_OFF"
    if "IMPULSE" in text:
        return "ALT_IMPULSE"
    if "SOL" in text and "STRENGTH" in text:
        return "SOL_STRENGTH"
    if "BTC" in text and "LEAD" in text:
        return "BTC_LEADERSHIP"
    if "DOWN" in text:
        return "TREND_DOWN"
    if "UP" in text or text == "TREND":
        return "TREND_UP"
    if "SIDE" in text or "CHOP" in text:
        return "SIDEWAYS"
    if "PROTECT" in text:
        return "SIDEWAYS"
    if "HIGH_VOL" in text:
        return "HIGH_VOL"
    if "LOW_VOL" in text:
        return "LOW_VOL"
    return text


def _latest_as_of(frame: pl.DataFrame, as_of_date: date) -> pl.DataFrame:
    if frame.is_empty() or "as_of_date" not in frame.columns:
        return frame
    dates = [
        str(value)[:10]
        for value in frame.get_column("as_of_date").drop_nulls().to_list()
        if str(value)[:10] <= as_of_date.isoformat()
    ]
    if not dates:
        return pl.DataFrame(schema=frame.schema)
    latest = max(dates)
    return frame.filter(pl.col("as_of_date").cast(pl.Utf8).str.slice(0, 10) == latest)


def _read_latest_as_of_dataset(path: Path, as_of_date: date) -> pl.DataFrame:
    lazy, columns = _safe_lazy_frame(path)
    if lazy is None:
        return pl.DataFrame()
    if "as_of_date" not in columns:
        return _collect_lazy(lazy.tail(50_000))
    day_text = as_of_date.isoformat()
    latest_frame = _collect_lazy(
        lazy.with_columns(
            pl.col("as_of_date").cast(pl.Utf8).str.slice(0, 10).alias("_as_of_day")
        )
        .filter((pl.col("_as_of_day") <= day_text) & pl.col("_as_of_day").is_not_null())
        .select(pl.col("_as_of_day").max().alias("latest_as_of_date"))
    )
    if latest_frame.is_empty():
        return pl.DataFrame()
    latest = latest_frame.item(0, "latest_as_of_date")
    if latest is None:
        return pl.DataFrame()
    return _collect_lazy(
        lazy.with_columns(
            pl.col("as_of_date").cast(pl.Utf8).str.slice(0, 10).alias("_as_of_day")
        )
        .filter(pl.col("_as_of_day") == str(latest))
        .drop("_as_of_day")
    )


def _safe_lazy_frame(path: Path) -> tuple[pl.LazyFrame | None, set[str]]:
    try:
        lazy = read_parquet_lazy(path)
        columns = set(lazy.collect_schema().names())
    except Exception:
        return None, set()
    return lazy, columns


def _collect_lazy(lazy: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lazy.collect()
    except pl.exceptions.SchemaError:
        return pl.DataFrame()
    except Exception as exc:
        if "not found" in str(exc).lower() or "no such file" in str(exc).lower():
            return pl.DataFrame()
        raise


def _lazy_datetime(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False)


def _lazy_normalized_symbol(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace("^OKX:", "")
        .str.replace("/", "-")
        .str.to_uppercase()
    )


def _spread_bps_expression(columns: set[str]) -> pl.Expr | None:
    expressions: list[pl.Expr] = []
    if "spread_bps" in columns:
        expressions.append(pl.col("spread_bps").cast(pl.Float64, strict=False))
    bid = _coalesced_float_column(columns, ["bid_px", "best_bid", "bid", "bid_price"])
    ask = _coalesced_float_column(columns, ["ask_px", "best_ask", "ask", "ask_price"])
    if bid is not None and ask is not None:
        computed = (
            pl.when(
                bid.is_not_null()
                & ask.is_not_null()
                & (ask > bid)
                & ((ask + bid) > 0)
            )
            .then((ask - bid) / ((ask + bid) / 2.0) * 10000.0)
            .otherwise(None)
        )
        expressions.append(computed)
    if not expressions:
        return None
    return pl.coalesce(expressions)


def _coalesced_float_column(columns: set[str], candidates: list[str]) -> pl.Expr | None:
    expressions = [
        pl.col(column).cast(pl.Float64, strict=False) for column in candidates if column in columns
    ]
    if not expressions:
        return None
    return pl.coalesce(expressions)


def _dedupe_matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("strategy_candidate") or ""),
            str(row.get("symbol") or ""),
            str(row.get("canonical_regime") or ""),
            int(row.get("horizon_hours") or 0),
        )
        current = by_key.get(key)
        if current is None or int(row.get("complete_sample_count") or 0) >= int(
            current.get("complete_sample_count") or 0
        ):
            by_key[key] = row
    return list(by_key.values())


def _replace_as_of_date(path: Path, new_frame: pl.DataFrame, as_of_date: date) -> int:
    existing = read_parquet_dataset(path)
    if not existing.is_empty() and "as_of_date" in existing.columns:
        existing = existing.filter(
            pl.col("as_of_date").cast(pl.Utf8).str.slice(0, 10) != as_of_date.isoformat()
        )
    frames = [frame for frame in [existing, new_frame] if not frame.is_empty()]
    combined = pl.concat(frames, how="diagonal_relaxed") if frames else new_frame
    write_parquet_dataset(combined, path)
    return combined.height


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value is None or str(value).strip().lower() == "auto":
        return datetime.now(UTC).date()
    return date.fromisoformat(str(value)[:10])


def _dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _win_rate(values: list[float]) -> float | None:
    return sum(1 for value in values if value > 0) / len(values) if values else None


def _counter(values: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        text = str(value or "").strip()
        if text:
            counter[text] += 1
    return dict(counter)


def _group_by(
    rows: list[dict[str, Any]],
    keys: list[str],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(item) for item in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return _json_list(parsed)


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
