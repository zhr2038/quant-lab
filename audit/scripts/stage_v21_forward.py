"""Run the one frozen Audit v2.1 low-vol forward-paper hypothesis."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.factors import low_vol_480  # noqa: E402
from audit.auditlib.forward_v21 import (  # noqa: E402
    BAR_CLOSE_AVAILABLE_DELAY_HOURS,
    DECISION_SCHEMA,
    ENTRY_FEE_BPS,
    ENTRY_PRICE_RULE,
    ENTRY_SLIPPAGE_BPS,
    EQUITY_SCHEMA,
    EVENT_SCHEMA,
    EXIT_FEE_BPS,
    EXIT_SLIPPAGE_BPS,
    HOLDING_HOURS,
    RUNNER_VERSION,
    STRATEGY_ID,
    TRADE_SCHEMA,
    append_only_events,
    arithmetic_sum_return,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    build_cohort_equity,
    compounded_return,
    empty_frame,
    forward_available_days,
    forward_review_status,
    make_event,
    merge_immutable_decisions,
    merge_trade_states,
    performance_counts,
    portfolio_period_returns,
    resolve_trade_state,
    select_next_bar_close,
    stable_decision_id,
    stable_trade_id,
    utc,
    validate_parameter_lock,
)
from audit.auditlib.portfolio_backtest import _capped_weights  # noqa: E402
from audit.auditlib.universe import UNIVERSES, build_daily_universe  # noqa: E402

OKX_HISTORY = "https://www.okx.com/api/v5/market/history-candles"
BAR_COLUMNS = ["symbol", "ts", "open", "high", "low", "close", "volume", "quote_volume"]
BAR_SCHEMA = {
    "symbol": pl.Utf8,
    "ts": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "quote_volume": pl.Float64,
}
BENCHMARK_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "btc_equity": pl.Float64,
    "equal_weight_universe_equity": pl.Float64,
    "cash_equity": pl.Float64,
}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_or_empty(path: Path, schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else empty_frame(schema)


def _fetch_symbol(
    session: requests.Session,
    symbol: str,
    fetch_after: datetime,
    as_of: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = int(utc(as_of).timestamp() * 1000)
    after_ms = int(utc(fetch_after).timestamp() * 1000)
    as_of_ms = int(utc(as_of).timestamp() * 1000)
    maximum_pages = max(2, min(100, math.ceil((as_of_ms - after_ms) / 3_600_000 / 100) + 2))
    for _page in range(maximum_pages):
        response: requests.Response | None = None
        params = {"instId": symbol, "bar": "1H", "limit": "100", "after": str(cursor)}
        for retry in range(5):
            try:
                response = session.get(OKX_HISTORY, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                if str(payload.get("code")) != "0":
                    raise RuntimeError(f"OKX {payload.get('code')}: {payload.get('msg')}")
                break
            except Exception:
                if retry == 4:
                    raise
                time.sleep(2**retry)
        assert response is not None
        data = response.json().get("data") or []
        if not data:
            break
        oldest = min(int(item[0]) for item in data)
        for item in data:
            timestamp_ms = int(item[0])
            if after_ms < timestamp_ms <= as_of_ms and str(item[8]) == "1":
                rows.append(
                    {
                        "symbol": symbol,
                        "ts": datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                        "quote_volume": float(item[7]),
                    }
                )
        if oldest <= after_ms or len(data) < 100:
            break
        cursor = oldest - 1
        time.sleep(0.11)
    return rows


def load_or_fetch_market(
    *,
    root: Path,
    historical_bars: pl.DataFrame,
    fetch_after: datetime,
    as_of: datetime,
    no_fetch: bool,
) -> pl.DataFrame:
    cache = root / "state/forward_v21_market_bars.parquet"
    existing = pl.read_parquet(cache) if cache.exists() else empty_frame(BAR_SCHEMA)
    rows: list[dict[str, Any]] = []
    if not no_fetch:
        symbols = historical_bars["symbol"].unique().sort().to_list()
        latest_by_symbol = {
            str(row["symbol"]): utc(row["latest_ts"])
            for row in existing.group_by("symbol")
            .agg(pl.col("ts").max().alias("latest_ts"))
            .iter_rows(named=True)
        }
        with requests.Session() as session:
            for index, symbol in enumerate(symbols, start=1):
                symbol_name = str(symbol)
                symbol_fetch_after = latest_by_symbol.get(symbol_name, fetch_after)
                if symbol_fetch_after >= as_of:
                    continue
                rows.extend(
                    _fetch_symbol(
                        session,
                        symbol_name,
                        symbol_fetch_after,
                        as_of,
                    )
                )
                if index % 20 == 0:
                    print(f"forward public candles: {index}/{len(symbols)} symbols")
                time.sleep(0.11)
    fresh = pl.DataFrame(rows, schema=BAR_SCHEMA) if rows else empty_frame(BAR_SCHEMA)
    frames = [frame for frame in (existing, fresh) if not frame.is_empty()]
    combined = (
        pl.concat(frames, how="vertical_relaxed")
        .unique(subset=["symbol", "ts"], keep="last")
        .sort(["symbol", "ts"])
        if frames
        else empty_frame(BAR_SCHEMA)
    )
    atomic_write_parquet(combined, cache)
    return combined


def _available_cutoff(
    historical: pl.DataFrame, forward: pl.DataFrame, requested_as_of: datetime
) -> datetime:
    eligible = forward.filter(
        pl.col("ts")
        + pl.duration(hours=BAR_CLOSE_AVAILABLE_DELAY_HOURS)
        <= requested_as_of
    )
    btc = eligible.filter(pl.col("symbol") == "BTC-USDT")
    if not btc.is_empty():
        return utc(btc["ts"].max()) + timedelta(
            hours=BAR_CLOSE_AVAILABLE_DELAY_HOURS
        )
    if not eligible.is_empty():
        return utc(eligible["ts"].max()) + timedelta(
            hours=BAR_CLOSE_AVAILABLE_DELAY_HOURS
        )
    return utc(historical["ts"].max()) + timedelta(
        hours=BAR_CLOSE_AVAILABLE_DELAY_HOURS
    )


def _initial_trade(
    *,
    decision_id: str,
    symbol: str,
    weight: float,
    entry_ts: datetime,
    entry_price: float,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "trade_id": stable_trade_id(decision_id, symbol, entry_ts),
        "decision_id": decision_id,
        "symbol": symbol,
        "target_weight": float(weight),
        "entry_ts": entry_ts,
        "entry_price": float(entry_price),
        "scheduled_exit_ts": entry_ts + timedelta(hours=HOLDING_HOURS),
        "actual_exit_ts": None,
        "exit_price": None,
        "exit_delay_bars": None,
        "mark_ts": entry_ts,
        "mark_price": float(entry_price),
        "entry_fee_bps": ENTRY_FEE_BPS,
        "entry_slippage_bps": ENTRY_SLIPPAGE_BPS,
        "exit_fee_bps": EXIT_FEE_BPS,
        "exit_slippage_bps": EXIT_SLIPPAGE_BPS,
        "gross_return": None,
        "net_return": None,
        "unrealized_gross_return": 0.0,
        "unrealized_net_return": -(ENTRY_FEE_BPS + ENTRY_SLIPPAGE_BPS) / 10_000.0,
        "weighted_gross_contribution": None,
        "weighted_net_contribution": None,
        "status": "OPEN",
        "invalidation_reason": "",
        "parameter_lock_hash": str(lock["sha256"]),
        "code_commit": str(lock["code_commit"]),
        "snapshot_id": str(lock["data_snapshot_id"]),
        "runner_version": RUNNER_VERSION,
    }


def _decision_times(
    forward: pl.DataFrame, cutoff: datetime, available_cutoff: datetime
) -> list[datetime]:
    cutoff = utc(cutoff)
    anchor = cutoff.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    eligible = (
        forward.filter(pl.col("symbol") == "BTC-USDT")
        .with_columns(
            (
                pl.col("ts")
                + pl.duration(hours=BAR_CLOSE_AVAILABLE_DELAY_HOURS)
            ).alias("_feature_available_ts")
        )
        .filter(
            (pl.col("_feature_available_ts") > cutoff)
            & (
                pl.col("_feature_available_ts")
                + pl.duration(hours=1)
                <= available_cutoff
            )
        )["_feature_available_ts"]
        .unique()
        .sort()
        .to_list()
    )
    if not eligible:
        return []
    return [
        utc(value)
        for value in eligible
        if int((utc(value) - anchor).total_seconds() // 3600) % 120 == 0
    ]


def _btc_state(btc: pl.DataFrame, feature_ts: datetime) -> str:
    row = btc.filter(pl.col("ts") == feature_ts)
    if row.is_empty() or row["btc_ma_60d"][0] is None:
        return "UNAVAILABLE"
    return "UP" if float(row["close"][0]) >= float(row["btc_ma_60d"][0]) else "DOWN"


def build_new_decisions(
    *,
    lock: Mapping[str, Any],
    cutoff: datetime,
    available_cutoff: datetime,
    combined_bars: pl.DataFrame,
    forward_bars: pl.DataFrame,
    existing: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, int, int]:
    signals = low_vol_480(combined_bars)
    universe = build_daily_universe(combined_bars, UNIVERSES["top20"])
    btc = (
        combined_bars.filter(pl.col("symbol") == "BTC-USDT")
        .sort("ts")
        .with_columns(pl.col("close").rolling_mean(1440).alias("btc_ma_60d"))
        .select(["ts", "close", "btc_ma_60d"])
    )
    existing_ids = set(existing["decision_id"].to_list()) if not existing.is_empty() else set()
    decision_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    coverage_expected = 0
    coverage_available = 0
    for feature_cutoff_ts in _decision_times(
        forward_bars, cutoff, available_cutoff
    ):
        feature_bar_ts = feature_cutoff_ts - timedelta(
            hours=BAR_CLOSE_AVAILABLE_DELAY_HOURS
        )
        decision_id = stable_decision_id(
            STRATEGY_ID,
            feature_cutoff_ts,
            str(lock["data_snapshot_id"]),
            str(lock["sha256"]),
        )
        if decision_id in existing_ids:
            continue
        entry_ts = feature_cutoff_ts + timedelta(hours=1)
        membership = universe.filter(
            pl.col("date") == feature_cutoff_ts.date()
        ).sort("rank")
        factor_values = (
            signals.filter(pl.col("feature_ts") == feature_bar_ts)
            .join(membership.select(["symbol", "rank"]), on="symbol", how="inner")
            .drop_nulls("signal")
            .sort(["signal", "symbol"], descending=[True, False])
        )
        btc_state = _btc_state(btc, feature_bar_ts)
        target = (
            factor_values.head(3)
            if btc_state == "UP" and factor_values.height >= 3
            else factor_values.head(0)
        )
        raw_weights = (
            _capped_weights(target["signal"].to_numpy(), 3, 0.50, "score")
            if target.height == 3
            else np.asarray([], dtype=float)
        )
        selected: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        if btc_state == "UP" and target.height < 3:
            rejected.append({"symbol": "*", "reason": "INSUFFICIENT_RANKED_UNIVERSE"})
        for row, weight in zip(target.iter_rows(named=True), raw_weights, strict=True):
            symbol = str(row["symbol"])
            coverage_expected += 1
            price = select_next_bar_close(
                combined_bars, symbol, feature_cutoff_ts
            )
            if price is None:
                rejected.append({"symbol": symbol, "reason": "MISSING_EXACT_NEXT_BAR_CLOSE"})
                continue
            coverage_available += 1
            actual_entry_ts, entry_price = price
            selected.append(
                {
                    "symbol": symbol,
                    "raw_score": float(row["signal"]),
                    "weight": float(weight),
                    "entry_ts": actual_entry_ts,
                    "entry_price": entry_price,
                }
            )
        invested_weight = float(sum(item["weight"] for item in selected))
        if invested_weight > 1.0 + 1e-12:
            raise RuntimeError("frozen target weights exceed one")
        cash_weight = max(0.0, 1.0 - invested_weight)
        decision_status = "INVESTED" if selected else "CASH"
        decision = {
            "decision_id": decision_id,
            "strategy_id": STRATEGY_ID,
            "decision_ts": feature_cutoff_ts,
            "feature_cutoff_ts": feature_cutoff_ts,
            "entry_ts": entry_ts,
            "available_data_cutoff": available_cutoff,
            "btc_trend_state": btc_state,
            "universe": _json(membership["symbol"].to_list()),
            "factor_values": _json(
                factor_values.select(["symbol", "signal", "rank"]).to_dicts()
            ),
            "ranked_symbols": _json(factor_values["symbol"].to_list()),
            "selected_symbols": _json([item["symbol"] for item in selected]),
            "rejected_symbols": _json(rejected),
            "target_weights": _json(
                {item["symbol"]: item["weight"] for item in selected}
            ),
            "cash_weight": cash_weight,
            "invested_weight": invested_weight,
            "decision_status": decision_status,
            "decision_delay_bars": 1,
            "entry_price_rule": ENTRY_PRICE_RULE,
            "parameter_lock_hash": str(lock["sha256"]),
            "code_commit": str(lock["code_commit"]),
            "snapshot_id": str(lock["data_snapshot_id"]),
            "hypothesis_type": "POST_HOC_HYPOTHESIS",
            "parameters_locked": True,
            "runner_version": RUNNER_VERSION,
        }
        decision_rows.append(decision)
        event_rows.append(
            make_event(
                event_type="DECISION_CREATED",
                event_ts=feature_cutoff_ts,
                decision_id=decision_id,
                payload={
                    "btc_trend_state": btc_state,
                    "decision_status": decision_status,
                    "selected_symbols": [item["symbol"] for item in selected],
                    "rejected_symbols": rejected,
                    "target_weights": {
                        item["symbol"]: item["weight"] for item in selected
                    },
                    "cash_weight": cash_weight,
                },
                parameter_lock_hash=str(lock["sha256"]),
                code_commit=str(lock["code_commit"]),
            )
        )
        for item in selected:
            trade = _initial_trade(
                decision_id=decision_id,
                symbol=item["symbol"],
                weight=item["weight"],
                entry_ts=item["entry_ts"],
                entry_price=item["entry_price"],
                lock=lock,
            )
            trade_rows.append(trade)
            event_rows.append(
                make_event(
                    event_type="ENTRY_RECORDED",
                    event_ts=item["entry_ts"],
                    decision_id=decision_id,
                    trade_id=trade["trade_id"],
                    payload={
                        "symbol": item["symbol"],
                        "target_weight": item["weight"],
                        "entry_price": item["entry_price"],
                        "scheduled_exit_ts": trade["scheduled_exit_ts"].isoformat(),
                        "entry_total_cost_bps": ENTRY_FEE_BPS + ENTRY_SLIPPAGE_BPS,
                    },
                    parameter_lock_hash=str(lock["sha256"]),
                    code_commit=str(lock["code_commit"]),
                )
            )
    return (
        pl.DataFrame(decision_rows, schema=DECISION_SCHEMA)
        if decision_rows
        else empty_frame(DECISION_SCHEMA),
        pl.DataFrame(trade_rows, schema=TRADE_SCHEMA)
        if trade_rows
        else empty_frame(TRADE_SCHEMA),
        pl.DataFrame(event_rows, schema=EVENT_SCHEMA)
        if event_rows
        else empty_frame(EVENT_SCHEMA),
        coverage_expected,
        coverage_available,
    )


def update_trade_states(
    *,
    existing: pl.DataFrame,
    new: pl.DataFrame,
    bars: pl.DataFrame,
    available_cutoff: datetime,
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    prior = {row["trade_id"]: row for row in existing.iter_rows(named=True)}
    base = merge_trade_states(existing, new)
    resolved_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for trade in base.iter_rows(named=True):
        resolved = resolve_trade_state(trade, bars, available_cutoff)
        resolved_rows.append(resolved)
        old = prior.get(resolved["trade_id"])
        if resolved["status"] == "CLOSED" and (old is None or old["status"] != "CLOSED"):
            event_rows.append(
                make_event(
                    event_type="EXIT_RECORDED",
                    event_ts=resolved["actual_exit_ts"],
                    decision_id=resolved["decision_id"],
                    trade_id=resolved["trade_id"],
                    payload={
                        "actual_exit_ts": resolved["actual_exit_ts"].isoformat(),
                        "exit_delay_bars": resolved["exit_delay_bars"],
                        "exit_price": resolved["exit_price"],
                        "gross_return": resolved["gross_return"],
                        "net_return": resolved["net_return"],
                        "exit_total_cost_bps": EXIT_FEE_BPS + EXIT_SLIPPAGE_BPS,
                    },
                    parameter_lock_hash=str(lock["sha256"]),
                    code_commit=str(lock["code_commit"]),
                )
            )
        elif resolved["status"] == "OPEN" and (
            old is None
            or old.get("mark_ts") != resolved.get("mark_ts")
            or old.get("mark_price") != resolved.get("mark_price")
        ):
            event_rows.append(
                make_event(
                    event_type="MARK_UPDATED",
                    event_ts=resolved["mark_ts"],
                    decision_id=resolved["decision_id"],
                    trade_id=resolved["trade_id"],
                    payload={
                        "mark_price": resolved["mark_price"],
                        "unrealized_gross_return": resolved["unrealized_gross_return"],
                        "unrealized_net_return": resolved["unrealized_net_return"],
                    },
                    parameter_lock_hash=str(lock["sha256"]),
                    code_commit=str(lock["code_commit"]),
                )
            )
    updates = (
        pl.DataFrame(resolved_rows, schema=TRADE_SCHEMA)
        if resolved_rows
        else empty_frame(TRADE_SCHEMA)
    )
    events = (
        pl.DataFrame(event_rows, schema=EVENT_SCHEMA)
        if event_rows
        else empty_frame(EVENT_SCHEMA)
    )
    return merge_trade_states(base, updates), events


def _benchmark_leg(
    benchmark: str,
    decision_id: str,
    symbol: str,
    weight: float,
    entry: tuple[datetime, float],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    trade = _initial_trade(
        decision_id=f"{benchmark}:{decision_id}",
        symbol=symbol,
        weight=weight,
        entry_ts=entry[0],
        entry_price=entry[1],
        lock=lock,
    )
    trade["trade_id"] = stable_trade_id(trade["decision_id"], symbol, entry[0])
    return trade


def _benchmark_state(
    *,
    benchmark: str,
    decisions: pl.DataFrame,
    bars: pl.DataFrame,
    universe: pl.DataFrame,
    available_cutoff: datetime,
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    decision_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    for decision in decisions.sort("decision_ts").iter_rows(named=True):
        feature_ts = utc(decision["feature_cutoff_ts"])
        if benchmark == "btc":
            symbols = ["BTC-USDT"]
        else:
            symbols = (
                universe.filter(pl.col("date") == feature_ts.date())
                .sort("rank")["symbol"]
                .to_list()
            )
        expected_weight = 1.0 / len(symbols) if symbols else 0.0
        local: list[dict[str, Any]] = []
        for symbol in symbols:
            entry = select_next_bar_close(bars, str(symbol), feature_ts)
            if entry is not None:
                local.append(
                    _benchmark_leg(
                        benchmark,
                        str(decision["decision_id"]),
                        str(symbol),
                        expected_weight,
                        entry,
                        lock,
                    )
                )
        invested = expected_weight * len(local)
        decision_rows.append(
            {
                "decision_id": f"{benchmark}:{decision['decision_id']}",
                "entry_ts": decision["entry_ts"],
                "cash_weight": max(0.0, 1.0 - invested),
                "invested_weight": invested,
            }
        )
        trade_rows.extend(local)
    decision_frame = pl.DataFrame(decision_rows) if decision_rows else pl.DataFrame()
    trade_frame = (
        pl.DataFrame(trade_rows, schema=TRADE_SCHEMA)
        if trade_rows
        else empty_frame(TRADE_SCHEMA)
    )
    resolved = [
        resolve_trade_state(row, bars, available_cutoff)
        for row in trade_frame.iter_rows(named=True)
    ]
    resolved_frame = (
        pl.DataFrame(resolved, schema=TRADE_SCHEMA)
        if resolved
        else empty_frame(TRADE_SCHEMA)
    )
    return decision_frame, resolved_frame


def _timeline(
    combined: pl.DataFrame, cutoff: datetime, available_cutoff: datetime
) -> list[datetime]:
    timestamps = (
        combined.filter(
            (pl.col("symbol") == "BTC-USDT")
            & (
                pl.col("ts")
                + pl.duration(hours=BAR_CLOSE_AVAILABLE_DELAY_HOURS)
                > cutoff
            )
            & (
                pl.col("ts")
                + pl.duration(hours=BAR_CLOSE_AVAILABLE_DELAY_HOURS)
                <= available_cutoff
            )
        )
        .select(
            (
                pl.col("ts")
                + pl.duration(hours=BAR_CLOSE_AVAILABLE_DELAY_HOURS)
            ).alias("_available_ts")
        )["_available_ts"]
        .unique()
        .sort()
        .to_list()
    )
    return [utc(value) for value in timestamps] or [available_cutoff]


def build_equity_and_benchmarks(
    *,
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    combined: pl.DataFrame,
    cutoff: datetime,
    available_cutoff: datetime,
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    universe = build_daily_universe(combined, UNIVERSES["top20"])
    timeline = _timeline(combined, cutoff, available_cutoff)
    strategy_rows = build_cohort_equity(decisions, trades, combined, timeline)
    btc_decisions, btc_trades = _benchmark_state(
        benchmark="btc",
        decisions=decisions,
        bars=combined,
        universe=universe,
        available_cutoff=available_cutoff,
        lock=lock,
    )
    uni_decisions, uni_trades = _benchmark_state(
        benchmark="universe",
        decisions=decisions,
        bars=combined,
        universe=universe,
        available_cutoff=available_cutoff,
        lock=lock,
    )
    btc_rows = build_cohort_equity(btc_decisions, btc_trades, combined, timeline)
    uni_rows = build_cohort_equity(uni_decisions, uni_trades, combined, timeline)
    peak = 1.0
    equity_rows: list[dict[str, Any]] = []
    for strategy, btc, uni in zip(strategy_rows, btc_rows, uni_rows, strict=True):
        peak = max(peak, float(strategy["equity"]))
        equity_rows.append(
            {
                "timestamp": strategy["timestamp"],
                "strategy_equity": float(strategy["equity"]),
                "btc_equity": float(btc["equity"]),
                "equal_weight_universe_equity": float(uni["equity"]),
                "cash_equity": 1.0,
                "drawdown": float(strategy["equity"]) / peak - 1.0,
                "gross_exposure": float(strategy["gross_exposure"]),
                "cash_weight": float(strategy["cash_weight"]),
                "open_trade_count": int(strategy["open_trade_count"]),
            }
        )
    equity = pl.DataFrame(equity_rows, schema=EQUITY_SCHEMA)
    benchmarks = equity.select(list(BENCHMARK_SCHEMA))
    return equity, benchmarks, btc_trades, uni_trades


def _period_return_rows(decisions: pl.DataFrame, trades: pl.DataFrame) -> list[dict[str, Any]]:
    decision_map = {row["decision_id"]: row for row in decisions.iter_rows(named=True)}
    rows: list[dict[str, Any]] = []
    for decision_id, group in trades.group_by("decision_id", maintain_order=True):
        key = decision_id[0] if isinstance(decision_id, tuple) else decision_id
        legs = group.to_dicts()
        if legs and all(item["status"] == "CLOSED" for item in legs):
            result = portfolio_period_returns(
                legs, float(decision_map[str(key)]["cash_weight"]), realized=True
            )
            rows.append({"decision_id": str(key), **result})
    return rows


def _trade_diagnostics(trades: pl.DataFrame) -> dict[str, float | int]:
    closed = trades.filter(pl.col("status") == "CLOSED")
    open_trades = trades.filter(pl.col("status") == "OPEN")
    wins = closed.filter(pl.col("net_return") > 0)
    losses = closed.filter(pl.col("net_return") < 0)
    positive = float(wins["net_return"].sum()) if not wins.is_empty() else 0.0
    negative = float(losses["net_return"].sum()) if not losses.is_empty() else 0.0
    return {
        "win_rate": float(wins.height / closed.height) if closed.height else 0.0,
        "average_trade_net_return": (
            float(closed["net_return"].mean()) if not closed.is_empty() else 0.0
        ),
        "profit_factor": (
            positive / abs(negative) if negative < 0 else (float("inf") if positive > 0 else 0.0)
        ),
        "unrealized_weighted_net_return": (
            float(
                (
                    open_trades["target_weight"]
                    * open_trades["unrealized_net_return"]
                ).sum()
            )
            if not open_trades.is_empty()
            else 0.0
        ),
    }


def _coverage_counts(decisions: pl.DataFrame) -> tuple[int, int]:
    expected = 0
    available = 0
    for row in decisions.iter_rows(named=True):
        selected = json.loads(row["selected_symbols"])
        rejected = json.loads(row["rejected_symbols"])
        available += len(selected)
        expected += len(selected)
        expected += sum(
            1
            for item in rejected
            if item.get("reason") == "MISSING_EXACT_NEXT_BAR_CLOSE"
        )
    return expected, available


def _concentration(trades: pl.DataFrame) -> tuple[float, str]:
    closed = trades.filter(pl.col("status") == "CLOSED")
    if closed.is_empty():
        return 0.0, ""
    by_symbol = closed.group_by("symbol").agg(
        pl.col("weighted_net_contribution").sum().alias("contribution")
    )
    denominator = float(by_symbol["contribution"].abs().sum())
    if denominator <= 1e-15:
        return 0.0, ""
    top = by_symbol.with_columns(
        (pl.col("contribution").abs() / denominator).alias("share")
    ).sort("share", descending=True).head(1)
    return float(top["share"][0]), str(top["symbol"][0])


def _performance(
    *,
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    equity: pl.DataFrame,
    cutoff: datetime,
    available_cutoff: datetime,
    requested_as_of: datetime,
    coverage_expected: int,
    coverage_available: int,
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    counts = performance_counts(decisions, trades)
    periods = _period_return_rows(decisions, trades)
    gross_periods = [float(item["gross_return"]) for item in periods]
    net_periods = [float(item["net_return"]) for item in periods]
    realized_gross = compounded_return(gross_periods)
    realized_net = compounded_return(net_periods)
    arithmetic_net = arithmetic_sum_return(net_periods)
    diagnostics = _trade_diagnostics(trades)
    available_days = forward_available_days(cutoff, available_cutoff)
    strategy_return = float(equity["strategy_equity"][-1]) - 1.0
    btc_return = float(equity["btc_equity"][-1]) - 1.0
    universe_return = float(equity["equal_weight_universe_equity"][-1]) - 1.0
    max_drawdown = float(equity["drawdown"].min())
    top_share, top_symbol = _concentration(trades)
    data_coverage = (
        coverage_available / coverage_expected if coverage_expected else 0.0
    )
    delay_hours = max(0.0, (requested_as_of - available_cutoff).total_seconds() / 3600.0)
    exit_delay_count = trades.filter(
        (pl.col("status") == "OPEN") & (pl.col("scheduled_exit_ts") <= available_cutoff)
    ).height
    feed_delay_count = int(delay_hours > 2.0)
    unhandled_delay_count = exit_delay_count + feed_delay_count
    status = forward_review_status(
        available_days=available_days,
        completed_periods=counts["completed_independent_period_count"],
        entry_count=counts["entry_count"],
        data_coverage=data_coverage,
        runner_error_count=0,
        unhandled_delay_count=unhandled_delay_count,
        strategy_net_return=realized_net,
        excess_vs_btc=strategy_return - btc_return,
        excess_vs_universe=strategy_return - universe_return,
        drawdown_within_lock=abs(max_drawdown)
        <= float(lock["maximum_drawdown_review_limit"]),
        concentration_within_lock=top_share
        <= float(lock["maximum_symbol_contribution_share_review_limit"]),
    )
    closed_weight = float(
        trades.filter(pl.col("status") == "CLOSED")["target_weight"].sum()
    ) if counts["closed_trade_count"] else 0.0
    open_weight = float(
        trades.filter(pl.col("status") == "OPEN")["target_weight"].sum()
    ) if counts["open_trade_count"] else 0.0
    row: dict[str, Any] = {
        "forward_v21_start_cutoff": cutoff.isoformat(),
        "requested_as_of": requested_as_of.isoformat(),
        "available_market_data_cutoff": available_cutoff.isoformat(),
        "forward_available_days": available_days,
        **counts,
        "realized_gross_compounded_return": realized_gross,
        "realized_net_compounded_return": realized_net,
        "realized_net_arithmetic_sum_return": arithmetic_net,
        "unrealized_weighted_net_return": diagnostics["unrealized_weighted_net_return"],
        "marked_strategy_compounded_return": strategy_return,
        "btc_benchmark_compounded_return": btc_return,
        "equal_weight_universe_benchmark_compounded_return": universe_return,
        "cash_benchmark_compounded_return": 0.0,
        "strategy_excess_vs_btc": strategy_return - btc_return,
        "strategy_excess_vs_universe": strategy_return - universe_return,
        "entry_fee_return": closed_weight * ENTRY_FEE_BPS / 10_000.0,
        "entry_slippage_return": closed_weight * ENTRY_SLIPPAGE_BPS / 10_000.0,
        "exit_fee_return": closed_weight * EXIT_FEE_BPS / 10_000.0,
        "exit_slippage_return": closed_weight * EXIT_SLIPPAGE_BPS / 10_000.0,
        "open_entry_cost_return": open_weight
        * (ENTRY_FEE_BPS + ENTRY_SLIPPAGE_BPS)
        / 10_000.0,
        "win_rate": diagnostics["win_rate"],
        "average_trade_net_return": diagnostics["average_trade_net_return"],
        "profit_factor": diagnostics["profit_factor"],
        "max_drawdown": max_drawdown,
        "top_symbol": top_symbol,
        "top_symbol_contribution_share": top_share,
        "data_coverage": data_coverage,
        "data_delay_hours": delay_hours,
        "runner_error_count": 0,
        "unhandled_delay_count": unhandled_delay_count,
        "parameter_lock_hash": lock["sha256"],
        "code_commit": lock["code_commit"],
        "runner_version": RUNNER_VERSION,
        "hypothesis_type": "POST_HOC_HYPOTHESIS",
        "parameters_locked": True,
        "conclusion": status,
    }
    status_payload = {
        "schema_version": "quant_lab_forward_v21_status.v1",
        **row,
        "portfolio_validity": "INCONCLUSIVE",
        "deployment_readiness": "INCONCLUSIVE",
        "production_alpha": "FROZEN",
        "live_order_effect": "none",
        "automatic_promotion": False,
        "legacy_v2_status": "INVALIDATED_BY_RUNNER_BUG",
        "may_merge_with_legacy_v2": False,
        "minimum_review_threshold": {
            "available_days": 30,
            "completed_independent_periods": 6,
            "coin_entries": 12,
            "data_coverage": 0.95,
            "maximum_drawdown": lock["maximum_drawdown_review_limit"],
            "maximum_symbol_contribution_share": lock[
                "maximum_symbol_contribution_share_review_limit"
            ],
            "maximum_possible_status": "PAPER_REVIEW_READY",
            "live_status_permitted": False,
        },
    }
    return pl.DataFrame([row]), status_payload


def _legacy_invalidation_event(root: Path, lock: Mapping[str, Any]) -> pl.DataFrame:
    status = _load_json(root / "artifacts/legacy_forward_v2_status.json")
    event = make_event(
        event_type="LEGACY_V2_RECORD_INVALIDATED",
        event_ts=utc(status["invalidated_at"]),
        decision_id="legacy_v2_forward_record",
        payload={
            "status": status["status"],
            "preserved": status["preserved"],
            "may_merge_with_v21": status["may_merge_with_v21"],
            "source_files": status["files"],
        },
        parameter_lock_hash=str(lock["sha256"]),
        code_commit=str(lock["code_commit"]),
    )
    return pl.DataFrame([event], schema=EVENT_SCHEMA)


def run(
    *,
    root: Path,
    v1_root: Path,
    requested_as_of: datetime,
    resume: bool,
    dry_run: bool,
    no_fetch: bool,
) -> dict[str, Any]:
    lock = _load_json(root / "manifests/parameter_lock_v21.json")
    cutoff_manifest = _load_json(root / "manifests/forward_v21_cutoff.json")
    validate_parameter_lock(lock)
    cutoff = utc(cutoff_manifest["forward_v21_start_cutoff"])
    if cutoff_manifest["parameter_lock_hash"] != lock["sha256"]:
        raise RuntimeError("cutoff and parameter lock hashes differ")
    if requested_as_of <= cutoff and not no_fetch:
        raise ValueError("requested as-of must be later than the v2.1 cutoff")
    artifact = root / "artifacts"
    paths = {
        "decisions": artifact / "forward_v21_decisions.parquet",
        "trades": artifact / "forward_v21_trades.parquet",
        "events": artifact / "forward_v21_events.parquet",
        "equity": artifact / "forward_v21_equity.parquet",
        "benchmarks": artifact / "forward_v21_benchmarks.parquet",
        "performance": artifact / "forward_v21_performance.csv",
        "status": artifact / "forward_v21_status.json",
    }
    if paths["decisions"].exists() and not resume and not dry_run:
        raise RuntimeError("forward v2.1 state exists; use --resume")
    historical_path = v1_root / "data/silver/bars_1h.parquet"
    if not historical_path.exists():
        raise FileNotFoundError(historical_path)
    historical = pl.read_parquet(historical_path).select(BAR_COLUMNS)
    v1_cutoff = utc(cutoff_manifest["v1_data_cutoff"])
    forward = load_or_fetch_market(
        root=root,
        historical_bars=historical,
        fetch_after=v1_cutoff,
        as_of=requested_as_of,
        no_fetch=no_fetch,
    )
    available_cutoff = _available_cutoff(historical, forward, requested_as_of)
    combined = (
        pl.concat([historical, forward], how="vertical_relaxed")
        .unique(subset=["symbol", "ts"], keep="last")
        .sort(["symbol", "ts"])
    )
    existing_decisions = _read_or_empty(paths["decisions"], DECISION_SCHEMA)
    existing_trades = _read_or_empty(paths["trades"], TRADE_SCHEMA)
    existing_events = _read_or_empty(paths["events"], EVENT_SCHEMA)
    new_decisions, new_trades, creation_events, _expected, _available = build_new_decisions(
        lock=lock,
        cutoff=cutoff,
        available_cutoff=available_cutoff,
        combined_bars=combined,
        forward_bars=forward,
        existing=existing_decisions,
    )
    decisions = merge_immutable_decisions(existing_decisions, new_decisions)
    expected, available = _coverage_counts(decisions)
    trades, state_events = update_trade_states(
        existing=existing_trades,
        new=new_trades,
        bars=combined,
        available_cutoff=available_cutoff,
        lock=lock,
    )
    events = append_only_events(existing_events, _legacy_invalidation_event(root, lock))
    events = append_only_events(events, creation_events)
    events = append_only_events(events, state_events)
    equity, benchmarks, _btc_trades, _universe_trades = build_equity_and_benchmarks(
        decisions=decisions,
        trades=trades,
        combined=combined,
        cutoff=cutoff,
        available_cutoff=available_cutoff,
        lock=lock,
    )
    performance, status = _performance(
        decisions=decisions,
        trades=trades,
        equity=equity,
        cutoff=cutoff,
        available_cutoff=available_cutoff,
        requested_as_of=requested_as_of,
        coverage_expected=expected,
        coverage_available=available,
        lock=lock,
    )
    if dry_run:
        print(
            f"dry-run decisions={decisions.height} entries={trades.height} "
            f"available_days={status['forward_available_days']:.6f}"
        )
        return status
    atomic_write_parquet(decisions, paths["decisions"])
    atomic_write_parquet(trades, paths["trades"])
    atomic_write_parquet(events, paths["events"])
    atomic_write_parquet(equity, paths["equity"])
    atomic_write_parquet(benchmarks, paths["benchmarks"])
    atomic_write_csv(performance, paths["performance"])
    atomic_write_json(status, paths["status"])
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get("AUDIT_V21_ROOT", "/home/hr/quant-alpha-audit-v2.1")
        ),
    )
    parser.add_argument(
        "--v1-root",
        type=Path,
        default=Path(os.environ.get("AUDIT_V1_ROOT", "/home/hr/quant-alpha-audit")),
    )
    parser.add_argument("--as-of", default=datetime.now(UTC).replace(microsecond=0).isoformat())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--no-fetch", action="store_true", help="Use the immutable local cache only"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.report_only:
        status = _load_json(root / "artifacts/forward_v21_status.json")
    else:
        status = run(
            root=root,
            v1_root=args.v1_root.resolve(),
            requested_as_of=utc(args.as_of),
            resume=args.resume,
            dry_run=args.dry_run,
            no_fetch=args.no_fetch,
        )
    print(f"forward_v21_status={status['conclusion']}")
    print(f"forward_available_days={status['forward_available_days']:.6f}")
    print(f"decision_count={status['decision_count']}")
    print(f"entry_count={status['entry_count']}")
    print(f"closed_trade_count={status['closed_trade_count']}")


if __name__ == "__main__":
    main()
