"""Run Audit v2.2 realtime or recovery forward-paper accounting.

Realtime mode considers only the latest due schedule.  Recovery mode may rebuild
older schedules, but every recovered row is permanently ineligible for formal
forward evidence.  Decision creation and entry recording are separate passes, so
a newly frozen decision can never receive a same-invocation historical fill.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.factors import low_vol_480  # noqa: E402
from audit.auditlib.forward_v22 import (  # noqa: E402
    BAR_CLOSE_DELAY_HOURS,
    BENCHMARK_DECISION_SCHEMA,
    BENCHMARK_TRADE_SCHEMA,
    BTC_TREND_LOOKBACK_HOURS,
    DECISION_SCHEMA,
    ENTRY_COST_BPS,
    EVENT_SCHEMA,
    EXIT_COST_BPS,
    FACTOR_LOOKBACK_HOURS,
    HOLDING_HOURS,
    RUNNER_VERSION,
    STRATEGY_ID,
    STRATEGY_VERSION,
    TRADE_SCHEMA,
    append_hash_chain,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    build_benchmark_equity,
    build_realized_equity,
    canonical_json,
    classify_decision_origin,
    compounded_return,
    cost_adjusted_return,
    due_schedule_times,
    empty_frame,
    evaluate_forward_status,
    make_event,
    merge_benchmark_decision_states,
    merge_benchmark_trade_states,
    merge_immutable_rows,
    merge_trade_states,
    payload_digest,
    runtime_identity,
    sha256_file,
    stable_benchmark_id,
    stable_decision_id,
    stable_trade_id,
    utc,
    validate_event_chain,
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
BENCHMARK_TYPES = (
    "BTC_BUY_AND_HOLD",
    "DYNAMIC_UNIVERSE_EQUAL_WEIGHT",
    "CASH",
)


class RunnerIntegrityError(RuntimeError):
    """Evidence, parameter, or code identity cannot be trusted."""


class DataIncompleteError(RuntimeError):
    """Required causal market data is not yet available."""


def _json(value: Any) -> str:
    return canonical_json(value)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read(path: Path, schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else empty_frame(schema)


def _frame(rows: Sequence[Mapping[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=dict(schema)) if rows else empty_frame(schema)


def _integrity_boundary(
    integrity_message: str, function: Any, *args: Any, **kwargs: Any
) -> Any:
    try:
        return function(*args, **kwargs)
    except (ValueError, RuntimeError) as exc:
        raise RunnerIntegrityError(integrity_message) from exc


def _fetch_symbol(
    session: requests.Session, symbol: str, fetch_after: datetime, as_of: datetime
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = int(utc(as_of).timestamp() * 1000)
    after_ms = int(utc(fetch_after).timestamp() * 1000)
    as_of_ms = int(utc(as_of).timestamp() * 1000)
    page_count = max(2, min(100, math.ceil((as_of_ms - after_ms) / 360_000_000) + 2))
    for _page in range(page_count):
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


def contiguous_fetch_start(
    *,
    historical: pl.DataFrame,
    existing: pl.DataFrame,
    symbol: str,
    fallback: datetime,
) -> datetime:
    history = historical.filter(pl.col("symbol") == symbol)
    start = utc(history["ts"].max()) if not history.is_empty() else utc(fallback)
    if existing.is_empty():
        return start
    cached = sorted(
        utc(value)
        for value in existing.filter(
            (pl.col("symbol") == symbol) & (pl.col("ts") > start)
        )["ts"].to_list()
    )
    expected = start + timedelta(hours=BAR_CLOSE_DELAY_HOURS)
    for timestamp in cached:
        if timestamp < expected:
            continue
        if timestamp != expected:
            break
        start = timestamp
        expected = start + timedelta(hours=BAR_CLOSE_DELAY_HOURS)
    return start


def load_or_fetch_market(
    *,
    root: Path,
    historical: pl.DataFrame,
    fetch_after: datetime,
    as_of: datetime,
    no_fetch: bool,
    persist_cache: bool,
) -> tuple[pl.DataFrame, Path]:
    cache = root / "state/forward_v22_market_bars.parquet"
    existing = pl.read_parquet(cache) if cache.exists() else empty_frame(BAR_SCHEMA)
    rows: list[dict[str, Any]] = []
    if not no_fetch:
        symbols = historical["symbol"].unique().sort().to_list()
        with requests.Session() as session:
            for index, raw_symbol in enumerate(symbols, start=1):
                symbol = str(raw_symbol)
                start = contiguous_fetch_start(
                    historical=historical,
                    existing=existing,
                    symbol=symbol,
                    fallback=fetch_after,
                )
                if start >= as_of:
                    continue
                rows.extend(_fetch_symbol(session, symbol, start, as_of))
                if index % 20 == 0:
                    print(f"forward_v22_public_candles={index}/{len(symbols)}")
                time.sleep(0.11)
    fresh = _frame(rows, BAR_SCHEMA)
    frames = [frame for frame in (existing, fresh) if not frame.is_empty()]
    combined = (
        pl.concat(frames, how="vertical_relaxed")
        .unique(subset=["symbol", "ts"], keep="last")
        .sort(["symbol", "ts"])
        if frames
        else empty_frame(BAR_SCHEMA)
    )
    if persist_cache:
        atomic_write_parquet(combined, cache)
    return combined, cache


def available_market_cutoff(bars: pl.DataFrame, requested_as_of: datetime) -> datetime:
    eligible = bars.filter(
        pl.col("ts") + pl.duration(hours=BAR_CLOSE_DELAY_HOURS) <= utc(requested_as_of)
    )
    btc = eligible.filter(pl.col("symbol") == "BTC-USDT")
    source = btc if not btc.is_empty() else eligible
    if source.is_empty():
        raise DataIncompleteError("no completed market bar is available")
    return utc(source["ts"].max()) + timedelta(hours=BAR_CLOSE_DELAY_HOURS)


def forward_market_fetch_start(cutoff: datetime) -> datetime:
    """Include the bar that first becomes observable strictly after cutoff."""
    return utc(cutoff) - timedelta(hours=BAR_CLOSE_DELAY_HOURS)


def build_market_snapshot(
    *, input_paths: Sequence[Path], market_cutoff: datetime, recorded_at: datetime
) -> dict[str, Any]:
    files = []
    for path in input_paths:
        if not path.is_file():
            continue
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "market_data_cutoff": utc(market_cutoff).isoformat(),
        "files": sorted(files, key=lambda item: item["path"]),
    }
    snapshot_id = f"market_v22_{payload_digest(payload)[:24]}"
    return {
        "snapshot_id": snapshot_id,
        "created_at": utc(recorded_at).isoformat(),
        **payload,
    }


def update_snapshot_registry(root: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "manifests/market_data_snapshot_v22.json"
    registry = (
        _load(path)
        if path.exists()
        else {
            "schema_version": "quant_lab_market_data_snapshot_registry.v1",
            "snapshots": [],
        }
    )
    rows = {str(item["snapshot_id"]): item for item in registry.get("snapshots", [])}
    prior = rows.get(str(snapshot["snapshot_id"]))
    if prior is not None:
        comparable = dict(snapshot)
        comparable["created_at"] = prior.get("created_at")
        if prior != comparable:
            raise RunnerIntegrityError("immutable market snapshot changed")
    else:
        rows[str(snapshot["snapshot_id"])] = dict(snapshot)
    registry["snapshots"] = sorted(rows.values(), key=lambda item: item["created_at"])
    registry["current_snapshot_id"] = snapshot["snapshot_id"]
    registry["integrity_alert"] = len(rows) > 1
    atomic_write_json(registry, path)
    return registry


def _exact_next_close(
    bars: pl.DataFrame, symbol: str, feature_cutoff_ts: datetime
) -> tuple[datetime, float] | None:
    bar_open = utc(feature_cutoff_ts)
    row = bars.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("ts") == bar_open)
        & pl.col("close").is_finite()
    )
    if row.is_empty():
        return None
    return bar_open + timedelta(hours=BAR_CLOSE_DELAY_HOURS), float(row["close"][0])


def _first_exit_close(
    bars: pl.DataFrame, symbol: str, scheduled_exit_ts: datetime, cutoff: datetime
) -> tuple[datetime, float] | None:
    rows = (
        bars.filter(
            (pl.col("symbol") == symbol)
            & pl.col("close").is_finite()
            & (pl.col("ts") + pl.duration(hours=BAR_CLOSE_DELAY_HOURS) >= scheduled_exit_ts)
            & (pl.col("ts") + pl.duration(hours=BAR_CLOSE_DELAY_HOURS) <= cutoff)
        )
        .sort("ts")
        .head(1)
    )
    if rows.is_empty():
        return None
    return utc(rows["ts"][0]) + timedelta(hours=BAR_CLOSE_DELAY_HOURS), float(
        rows["close"][0]
    )


def _decision_schedule_candidates(
    *,
    mode: str,
    cutoff: datetime,
    observed_cutoff: datetime,
    recorded_at: datetime,
    existing: pl.DataFrame,
) -> list[datetime]:
    end = min(utc(observed_cutoff), utc(recorded_at))
    due = due_schedule_times(cutoff, end)
    existing_times = (
        {utc(value) for value in existing["scheduled_run_ts"].to_list()}
        if not existing.is_empty()
        else set()
    )
    missing = [value for value in due if value not in existing_times]
    if mode == "realtime":
        return missing[-1:] if missing and missing[-1] == due[-1] else []
    if mode == "recovery":
        return missing
    raise ValueError(f"unsupported mode: {mode}")


def _symbols_with_complete_window(
    bars: pl.DataFrame, *, end_bar_ts: datetime, hours: int
) -> set[str]:
    end = utc(end_bar_ts)
    start = end - timedelta(hours=hours - 1)
    coverage = (
        bars.filter((pl.col("ts") >= start) & (pl.col("ts") <= end))
        .group_by("symbol")
        .agg(
            pl.col("ts").n_unique().alias("bar_count"),
            pl.col("ts").min().alias("first_bar"),
            pl.col("ts").max().alias("last_bar"),
        )
        .filter(
            (pl.col("bar_count") == hours)
            & (pl.col("first_bar") == start)
            & (pl.col("last_bar") == end)
        )
    )
    return {str(value) for value in coverage["symbol"].to_list()}


def _btc_state(bars: pl.DataFrame, feature_bar_ts: datetime) -> str:
    complete = _symbols_with_complete_window(
        bars.filter(pl.col("symbol") == "BTC-USDT"),
        end_bar_ts=feature_bar_ts,
        hours=BTC_TREND_LOOKBACK_HOURS,
    )
    if "BTC-USDT" not in complete:
        return "UNAVAILABLE"
    btc = (
        bars.filter(pl.col("symbol") == "BTC-USDT")
        .sort("ts")
        .with_columns(
            pl.col("close").rolling_mean(BTC_TREND_LOOKBACK_HOURS).alias("btc_ma_60d")
        )
        .filter(pl.col("ts") == feature_bar_ts)
    )
    if btc.is_empty() or btc["btc_ma_60d"][0] is None:
        return "UNAVAILABLE"
    return "UP" if float(btc["close"][0]) >= float(btc["btc_ma_60d"][0]) else "DOWN"


def create_decisions(
    *,
    mode: str,
    lock: Mapping[str, Any],
    cutoff: datetime,
    observed_cutoff: datetime,
    recorded_at: datetime,
    bars: pl.DataFrame,
    existing: pl.DataFrame,
    snapshot: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    schedules = _decision_schedule_candidates(
        mode=mode,
        cutoff=cutoff,
        observed_cutoff=observed_cutoff,
        recorded_at=recorded_at,
        existing=existing,
    )
    decision_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    input_paths = [item["path"] for item in snapshot["files"]]
    input_hashes = {item["path"]: item["sha256"] for item in snapshot["files"]}
    for scheduled in schedules:
        feature_bar_ts = scheduled - timedelta(hours=BAR_CLOSE_DELAY_HOURS)
        causal = bars.filter(pl.col("ts") <= feature_bar_ts)
        signals = low_vol_480(causal)
        universe = build_daily_universe(causal, UNIVERSES["top20"])
        membership = universe.filter(pl.col("date") == scheduled.date()).sort("rank")
        complete_symbols = _symbols_with_complete_window(
            causal, end_bar_ts=feature_bar_ts, hours=FACTOR_LOOKBACK_HOURS
        )
        factor = (
            signals.filter(pl.col("feature_ts") == feature_bar_ts)
            .join(membership.select(["symbol", "rank"]), on="symbol", how="inner")
            .filter(pl.col("symbol").is_in(sorted(complete_symbols)))
            .drop_nulls("signal")
            .sort(["signal", "symbol"], descending=[True, False])
        )
        btc_state = _btc_state(causal, feature_bar_ts)
        feature_coverage = factor.height / membership.height if membership.height else 0.0
        data_complete = (
            btc_state != "UNAVAILABLE"
            and membership.height >= 3
            and feature_coverage >= float(lock["minimum_data_coverage"])
        )
        target = (
            factor.head(3)
            if data_complete and btc_state == "UP" and factor.height >= 3
            else factor.head(0)
        )
        weights = (
            _capped_weights(target["signal"].to_numpy(), 3, 0.50, "score")
            if target.height == 3
            else np.asarray([], dtype=float)
        )
        selected = [str(value) for value in target["symbol"].to_list()]
        target_weights = {
            symbol: float(weight) for symbol, weight in zip(selected, weights, strict=True)
        }
        cash_weight = max(0.0, 1.0 - sum(target_weights.values()))
        origin, late, timely_eligible, latency = classify_decision_origin(
            mode=mode,
            scheduled_run_ts=scheduled,
            recorded_at=recorded_at,
            max_latency=int(lock["max_decision_latency_seconds"]),
        )
        eligible = timely_eligible and data_complete
        decision_id = stable_decision_id(
            scheduled, str(snapshot["snapshot_id"]), str(lock["sha256"])
        )
        row = {
            "decision_id": decision_id,
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
            "decision_ts": scheduled,
            "scheduled_run_ts": scheduled,
            "recorded_at": recorded_at,
            "decision_latency_seconds": latency,
            "decision_origin": origin,
            "late_reconstructed": late,
            "eligible_for_forward_evidence": eligible,
            "feature_cutoff_ts": scheduled,
            "observed_market_data_cutoff": observed_cutoff,
            "market_data_cutoff": observed_cutoff,
            "market_data_snapshot_id": str(snapshot["snapshot_id"]),
            "input_file_paths": _json(input_paths),
            "input_file_sha256": _json(input_hashes),
            "feature_data_coverage": feature_coverage,
            "data_quality_status": "PASS" if data_complete else "INCOMPLETE",
            "btc_trend_state": btc_state,
            "universe": _json([str(value) for value in membership["symbol"].to_list()]),
            "factor_scores": _json(
                factor.select(["symbol", "signal", "rank"]).to_dicts()
            ),
            "ranked_symbols": _json([str(value) for value in factor["symbol"].to_list()]),
            "selected_symbols": _json(selected),
            "target_weights": _json(target_weights),
            "cash_weight": cash_weight,
            "parameter_lock_hash": str(lock["sha256"]),
            "strategy_code_hash": str(lock["strategy_code_hash"]),
            "git_commit": str(lock["strategy_code_commit"]),
            "working_tree_clean": True,
            "status": (
                "DATA_INCOMPLETE"
                if not data_complete
                else "DECISION_CREATED"
                if selected
                else "CASH"
            ),
        }
        decision_rows.append(row)
        event_rows.append(
            make_event(
                event_type="DECISION_CREATED",
                event_ts=scheduled,
                recorded_at=recorded_at,
                decision_id=decision_id,
                payload={
                    "origin": origin,
                    "late_reconstructed": late,
                    "eligible_for_forward_evidence": eligible,
                    "feature_data_coverage": feature_coverage,
                    "data_quality_status": "PASS" if data_complete else "INCOMPLETE",
                    "btc_trend_state": btc_state,
                    "universe": json.loads(row["universe"]),
                    "factor_scores": json.loads(row["factor_scores"]),
                    "selected_symbols": selected,
                    "target_weights": target_weights,
                    "cash_weight": cash_weight,
                    "market_data_snapshot_id": snapshot["snapshot_id"],
                },
                lock=lock,
            )
        )
    return _frame(decision_rows, DECISION_SCHEMA), _frame(event_rows, EVENT_SCHEMA)


def create_entries(
    *,
    decisions: pl.DataFrame,
    existing_trades: pl.DataFrame,
    existing_benchmarks: pl.DataFrame,
    existing_benchmark_trades: pl.DataFrame,
    bars: pl.DataFrame,
    observed_cutoff: datetime,
    recorded_at: datetime,
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    trade_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    benchmark_trade_rows: list[dict[str, Any]] = []
    strategy_events: list[dict[str, Any]] = []
    benchmark_events: list[dict[str, Any]] = []
    existing_trade_keys = {
        (str(row["decision_id"]), str(row["symbol"]))
        for row in existing_trades.iter_rows(named=True)
    }
    existing_benchmark_ids = (
        set(existing_benchmarks["benchmark_id"].to_list())
        if not existing_benchmarks.is_empty()
        else set()
    )
    existing_benchmark_trade_ids = (
        set(existing_benchmark_trades["benchmark_trade_id"].to_list())
        if not existing_benchmark_trades.is_empty()
        else set()
    )
    for decision in decisions.sort("scheduled_run_ts").iter_rows(named=True):
        entry_due = utc(decision["scheduled_run_ts"]) + timedelta(hours=BAR_CLOSE_DELAY_HOURS)
        if observed_cutoff < entry_due:
            continue
        selected = json.loads(decision["selected_symbols"])
        weights = json.loads(decision["target_weights"])
        entry_latency = max(0.0, (recorded_at - entry_due).total_seconds())
        trade_eligible = bool(decision["eligible_for_forward_evidence"]) and entry_latency <= float(
            lock["max_decision_latency_seconds"]
        )
        for symbol in selected:
            key = (str(decision["decision_id"]), str(symbol))
            if key in existing_trade_keys:
                continue
            entry = _exact_next_close(bars, str(symbol), decision["feature_cutoff_ts"])
            if entry is None:
                continue
            trade_id = stable_trade_id(str(decision["decision_id"]), str(symbol), entry[0])
            trade = {
                "trade_id": trade_id,
                "decision_id": str(decision["decision_id"]),
                "symbol": str(symbol),
                "target_weight": float(weights[str(symbol)]),
                "entry_ts": entry[0],
                "entry_price": entry[1],
                "scheduled_exit_ts": entry[0] + timedelta(hours=HOLDING_HOURS),
                "actual_exit_ts": None,
                "exit_price": None,
                "entry_cost_bps": ENTRY_COST_BPS,
                "exit_cost_bps": EXIT_COST_BPS,
                "gross_return": None,
                "net_return": None,
                "eligible_for_forward_evidence": trade_eligible,
                "market_data_snapshot_id": str(decision["market_data_snapshot_id"]),
                "parameter_lock_hash": str(lock["sha256"]),
                "strategy_code_hash": str(lock["strategy_code_hash"]),
                "git_commit": str(lock["strategy_code_commit"]),
                "status": "OPEN",
            }
            trade_rows.append(trade)
            existing_trade_keys.add(key)
            strategy_events.append(
                make_event(
                    event_type="ENTRY_RECORDED",
                    event_ts=entry[0],
                    recorded_at=recorded_at,
                    decision_id=str(decision["decision_id"]),
                    trade_id=trade_id,
                    payload={
                        "symbol": symbol,
                        "target_weight": weights[str(symbol)],
                        "entry_price": entry[1],
                        "eligible_for_forward_evidence": trade_eligible,
                    },
                    lock=lock,
                )
            )

        benchmark_specs = {
            "BTC_BUY_AND_HOLD": ["BTC-USDT"],
            "DYNAMIC_UNIVERSE_EQUAL_WEIGHT": json.loads(decision["universe"]),
            "CASH": ["CASH"],
        }
        for benchmark_type, symbols in benchmark_specs.items():
            benchmark_id = stable_benchmark_id(str(decision["decision_id"]), benchmark_type)
            if benchmark_id in existing_benchmark_ids:
                continue
            expected_weight = 1.0 / len(symbols) if symbols else 0.0
            actual_symbols: list[str] = []
            actual_weights: dict[str, float] = {}
            for symbol in symbols:
                if symbol == "CASH":
                    entry = (entry_due, 1.0)
                    entry_cost = 0.0
                    exit_cost = 0.0
                else:
                    entry = _exact_next_close(bars, str(symbol), decision["feature_cutoff_ts"])
                    entry_cost = ENTRY_COST_BPS
                    exit_cost = EXIT_COST_BPS
                if entry is None:
                    continue
                benchmark_trade_id = payload_digest(
                    [benchmark_id, symbol, entry[0].isoformat()]
                )
                if benchmark_trade_id in existing_benchmark_trade_ids:
                    continue
                actual_symbols.append(str(symbol))
                actual_weights[str(symbol)] = expected_weight
                benchmark_trade_rows.append(
                    {
                        "benchmark_trade_id": benchmark_trade_id,
                        "benchmark_id": benchmark_id,
                        "benchmark_type": benchmark_type,
                        "symbol": str(symbol),
                        "weight": expected_weight,
                        "entry_ts": entry[0],
                        "entry_price": entry[1],
                        "scheduled_exit_ts": entry[0] + timedelta(hours=HOLDING_HOURS),
                        "actual_exit_ts": None,
                        "exit_price": None,
                        "gross_return": None,
                        "net_return": None,
                        "entry_cost_bps": entry_cost,
                        "exit_cost_bps": exit_cost,
                        "data_snapshot_hash": str(decision["market_data_snapshot_id"]),
                        "eligible_for_forward_evidence": trade_eligible,
                        "status": "OPEN",
                    }
                )
                existing_benchmark_trade_ids.add(benchmark_trade_id)
                benchmark_events.append(
                    make_event(
                        event_type="BENCHMARK_ENTRY_RECORDED",
                        event_ts=entry[0],
                        recorded_at=recorded_at,
                        decision_id=str(decision["decision_id"]),
                        trade_id=benchmark_trade_id,
                        payload={
                            "benchmark_id": benchmark_id,
                            "benchmark_type": benchmark_type,
                            "symbol": symbol,
                            "weight": expected_weight,
                            "entry_price": entry[1],
                        },
                        lock=lock,
                    )
                )
            benchmark_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_type": benchmark_type,
                    "decision_id": str(decision["decision_id"]),
                    "decision_ts": decision["decision_ts"],
                    "entry_ts": entry_due,
                    "scheduled_exit_ts": entry_due + timedelta(hours=HOLDING_HOURS),
                    "actual_exit_ts": None,
                    "entry_price": 1.0,
                    "exit_price": None,
                    "weights": _json(actual_weights),
                    "symbols": _json(actual_symbols),
                    "gross_return": None,
                    "net_return": None,
                    "data_snapshot_hash": str(decision["market_data_snapshot_id"]),
                    "market_data_cutoff": decision["market_data_cutoff"],
                    "code_commit": str(lock["strategy_code_commit"]),
                    "parameter_lock_hash": str(lock["sha256"]),
                    "strategy_code_hash": str(lock["strategy_code_hash"]),
                    "eligible_for_forward_evidence": trade_eligible,
                    "status": "OPEN",
                }
            )
            existing_benchmark_ids.add(benchmark_id)
            benchmark_events.append(
                make_event(
                    event_type="BENCHMARK_DECISION_CREATED",
                    event_ts=decision["decision_ts"],
                    recorded_at=recorded_at,
                    decision_id=str(decision["decision_id"]),
                    trade_id=benchmark_id,
                    payload={
                        "benchmark_type": benchmark_type,
                        "symbols": actual_symbols,
                        "weights": actual_weights,
                        "data_snapshot_hash": decision["market_data_snapshot_id"],
                    },
                    lock=lock,
                )
            )
    strategy_event_frame = _frame(strategy_events, EVENT_SCHEMA)
    benchmark_event_frame = _frame(benchmark_events, EVENT_SCHEMA)
    return (
        _frame(trade_rows, TRADE_SCHEMA),
        _frame(benchmark_rows, BENCHMARK_DECISION_SCHEMA),
        _frame(benchmark_trade_rows, BENCHMARK_TRADE_SCHEMA),
        pl.concat([strategy_event_frame, benchmark_event_frame], how="diagonal_relaxed")
        if strategy_events or benchmark_events
        else empty_frame(EVENT_SCHEMA),
    )


def update_strategy_exits(
    *,
    trades: pl.DataFrame,
    bars: pl.DataFrame,
    observed_cutoff: datetime,
    recorded_at: datetime,
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    updates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for prior in trades.iter_rows(named=True):
        row = dict(prior)
        if row["status"] == "CLOSED" or observed_cutoff < row["scheduled_exit_ts"]:
            updates.append(row)
            continue
        exit_value = _first_exit_close(
            bars, str(row["symbol"]), row["scheduled_exit_ts"], observed_cutoff
        )
        if exit_value is None:
            updates.append(row)
            continue
        gross = exit_value[1] / float(row["entry_price"]) - 1.0
        row.update(
            {
                "actual_exit_ts": exit_value[0],
                "exit_price": exit_value[1],
                "gross_return": gross,
                "net_return": cost_adjusted_return(
                    gross, float(row["entry_cost_bps"]), float(row["exit_cost_bps"])
                ),
                "status": "CLOSED",
            }
        )
        updates.append(row)
        events.append(
            make_event(
                event_type="EXIT_RECORDED",
                event_ts=exit_value[0],
                recorded_at=recorded_at,
                decision_id=str(row["decision_id"]),
                trade_id=str(row["trade_id"]),
                payload={
                    "symbol": row["symbol"],
                    "exit_price": row["exit_price"],
                    "gross_return": row["gross_return"],
                    "net_return": row["net_return"],
                },
                lock=lock,
            )
        )
    return _frame(updates, TRADE_SCHEMA), _frame(events, EVENT_SCHEMA)


def update_benchmark_exits(
    *,
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    bars: pl.DataFrame,
    observed_cutoff: datetime,
    recorded_at: datetime,
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    trade_updates: list[dict[str, Any]] = []
    decision_updates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for prior in trades.iter_rows(named=True):
        row = dict(prior)
        if row["status"] == "CLOSED" or observed_cutoff < row["scheduled_exit_ts"]:
            trade_updates.append(row)
            continue
        if row["symbol"] == "CASH":
            exit_value = (utc(row["scheduled_exit_ts"]), 1.0)
        else:
            exit_value = _first_exit_close(
                bars, str(row["symbol"]), row["scheduled_exit_ts"], observed_cutoff
            )
        if exit_value is None:
            trade_updates.append(row)
            continue
        gross = exit_value[1] / float(row["entry_price"]) - 1.0
        row.update(
            {
                "actual_exit_ts": exit_value[0],
                "exit_price": exit_value[1],
                "gross_return": gross,
                "net_return": cost_adjusted_return(
                    gross, float(row["entry_cost_bps"]), float(row["exit_cost_bps"])
                ),
                "status": "CLOSED",
            }
        )
        trade_updates.append(row)
        events.append(
            make_event(
                event_type="BENCHMARK_EXIT_RECORDED",
                event_ts=exit_value[0],
                recorded_at=recorded_at,
                decision_id=str(decisions.filter(pl.col("benchmark_id") == row["benchmark_id"])[
                    "decision_id"
                ][0]),
                trade_id=str(row["benchmark_trade_id"]),
                payload={
                    "benchmark_id": row["benchmark_id"],
                    "benchmark_type": row["benchmark_type"],
                    "symbol": row["symbol"],
                    "exit_price": row["exit_price"],
                    "net_return": row["net_return"],
                },
                lock=lock,
            )
        )
    trade_frame = _frame(trade_updates, BENCHMARK_TRADE_SCHEMA)
    for prior in decisions.iter_rows(named=True):
        row = dict(prior)
        if row["status"] == "CLOSED":
            decision_updates.append(row)
            continue
        local = trade_frame.filter(pl.col("benchmark_id") == row["benchmark_id"])
        if local.is_empty() or not local.select(pl.col("status").eq("CLOSED").all()).item():
            decision_updates.append(row)
            continue
        gross = float((local["weight"] * local["gross_return"]).sum())
        net = float((local["weight"] * local["net_return"]).sum())
        row.update(
            {
                "actual_exit_ts": local["actual_exit_ts"].max(),
                "exit_price": 1.0 + gross,
                "gross_return": gross,
                "net_return": net,
                "status": "CLOSED",
            }
        )
        decision_updates.append(row)
    return (
        _frame(decision_updates, BENCHMARK_DECISION_SCHEMA),
        trade_frame,
        _frame(events, EVENT_SCHEMA),
    )


def _completed_decision_ids(
    decisions: pl.DataFrame, trades: pl.DataFrame, observed_cutoff: datetime
) -> list[str]:
    completed: list[str] = []
    for decision in decisions.filter(
        pl.col("eligible_for_forward_evidence")
    ).iter_rows(named=True):
        selected = json.loads(decision["selected_symbols"])
        horizon_end = utc(decision["scheduled_run_ts"]) + timedelta(
            hours=BAR_CLOSE_DELAY_HOURS + HOLDING_HOURS
        )
        if observed_cutoff < horizon_end:
            continue
        local = trades.filter(pl.col("decision_id") == decision["decision_id"])
        if not selected or (
            local.height == len(selected)
            and local.select(pl.col("status").eq("CLOSED").all()).item()
        ):
            completed.append(str(decision["decision_id"]))
    return completed


def performance_and_status(
    *,
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    events: pl.DataFrame,
    benchmarks: pl.DataFrame,
    benchmark_events: pl.DataFrame,
    cutoff: datetime,
    observed_cutoff: datetime,
    snapshot: Mapping[str, Any],
    lock: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame, pl.DataFrame]:
    validate_event_chain(events)
    validate_event_chain(benchmark_events)
    realtime_count = decisions.filter(pl.col("decision_origin") == "REALTIME").height
    reconstructed_count = decisions.filter(
        pl.col("decision_origin") == "RECOVERY_RECONSTRUCTION"
    ).height
    eligible_decisions = decisions.filter(pl.col("eligible_for_forward_evidence"))
    ineligible_count = decisions.height - eligible_decisions.height
    eligible_trades = trades.filter(pl.col("eligible_for_forward_evidence"))
    completed_ids = _completed_decision_ids(decisions, trades, observed_cutoff)
    period_returns: list[float] = []
    for decision_id in completed_ids:
        local = eligible_trades.filter(pl.col("decision_id") == decision_id)
        period_returns.append(
            float((local["target_weight"] * local["net_return"]).sum())
            if not local.is_empty()
            else 0.0
        )
    strategy_return = compounded_return(period_returns)
    eligible_benchmarks = benchmarks.filter(
        pl.col("eligible_for_forward_evidence") & pl.col("decision_id").is_in(completed_ids)
    )

    def benchmark_return(name: str) -> float:
        rows = eligible_benchmarks.filter(
            (pl.col("benchmark_type") == name) & (pl.col("status") == "CLOSED")
        )
        return compounded_return(rows["net_return"].to_list()) if not rows.is_empty() else 0.0

    btc_return = benchmark_return("BTC_BUY_AND_HOLD")
    universe_return = benchmark_return("DYNAMIC_UNIVERSE_EQUAL_WEIGHT")
    cash_return = benchmark_return("CASH")
    equity = build_realized_equity(
        decisions, trades, cutoff, completed_decision_ids=completed_ids
    )
    benchmark_equity = build_benchmark_equity(benchmarks, cutoff)
    equity_values = equity["strategy_equity"].to_list()
    peak = 1.0
    max_drawdown = 0.0
    for value in equity_values:
        peak = max(peak, float(value))
        max_drawdown = min(max_drawdown, float(value) / peak - 1.0)
    closed = eligible_trades.filter(pl.col("status") == "CLOSED")
    top_symbol = ""
    top_share = 0.0
    if not closed.is_empty():
        contributions = closed.group_by("symbol").agg(
            (pl.col("target_weight") * pl.col("net_return")).sum().alias("contribution")
        )
        denominator = float(contributions["contribution"].abs().sum())
        if denominator > 1e-15:
            top = contributions.with_columns(
                (pl.col("contribution").abs() / denominator).alias("share")
            ).sort("share", descending=True).head(1)
            top_symbol = str(top["symbol"][0])
            top_share = float(top["share"][0])
    entry_due = eligible_decisions.filter(
        pl.col("scheduled_run_ts") + pl.duration(hours=BAR_CLOSE_DELAY_HOURS)
        <= observed_cutoff
    )
    expected_trades = sum(len(json.loads(value)) for value in entry_due["selected_symbols"])
    actual_trades = eligible_trades.height
    fill_coverage = min(1.0, actual_trades / expected_trades) if expected_trades else 1.0
    realtime_decisions = decisions.filter(pl.col("decision_origin") == "REALTIME")
    feature_coverage = (
        float(realtime_decisions["feature_data_coverage"].min())
        if not realtime_decisions.is_empty()
        else 1.0
    )
    incomplete_decisions = decisions.filter(
        pl.col("data_quality_status") != "PASS"
    ).height
    coverage = min(fill_coverage, feature_coverage)
    missing_fills = max(0, expected_trades - actual_trades)
    exit_gap_count = closed.filter(
        pl.col("actual_exit_ts") > pl.col("scheduled_exit_ts")
    ).height
    forward_days = max(0.0, (observed_cutoff - utc(cutoff)).total_seconds() / 86400.0)
    integrity_errors = list(identity.get("errors", []))
    mixed = eligible_decisions.filter(
        (pl.col("decision_origin") != "REALTIME") | pl.col("late_reconstructed")
    ).height
    if mixed:
        integrity_errors.append("INELIGIBLE_ORIGIN_MIXED_IN_FORMAL_DECISIONS")
    if eligible_trades.filter(
        ~pl.col("decision_id").is_in(eligible_decisions["decision_id"])
    ).height:
        integrity_errors.append("INELIGIBLE_DECISION_TRADE_MIX")
    metrics = {
        "forward_days": forward_days,
        "completed_independent_cycles": len(completed_ids),
        "actual_symbol_trades": actual_trades,
        "data_coverage": coverage,
        "base_cost_net_return": strategy_return,
        "excess_vs_btc": strategy_return - btc_return,
        "excess_vs_dynamic_universe": strategy_return - universe_return,
        "max_drawdown": max_drawdown,
        "maximum_single_symbol_contribution": top_share,
        "unhandled_market_gap_count": exit_gap_count,
        "unexplained_missing_fill_count": missing_fills,
        "system_error_count": 0,
        "data_completeness_unknown": incomplete_decisions > 0,
    }
    paper_status = evaluate_forward_status(
        metrics=metrics, lock=lock, integrity_errors=integrity_errors
    )
    row = {
        "audit_version": "v2.2",
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "forward_v22_cutoff": utc(cutoff).isoformat(),
        "market_data_cutoff": observed_cutoff.isoformat(),
        "market_data_snapshot_id": snapshot["snapshot_id"],
        "real_time_decision_count": realtime_count,
        "reconstructed_decision_count": reconstructed_count,
        "eligible_decision_count": eligible_decisions.height,
        "ineligible_decision_count": ineligible_count,
        "cash_decision_count": eligible_decisions.filter(pl.col("status") == "CASH").height,
        "actual_entry_count": actual_trades,
        "completed_independent_cycles": len(completed_ids),
        "actual_symbol_trades": actual_trades,
        "forward_days": forward_days,
        "data_coverage": coverage,
        "feature_data_incomplete_count": incomplete_decisions,
        "strategy_net_return": strategy_return,
        "btc_benchmark_return": btc_return,
        "dynamic_universe_benchmark_return": universe_return,
        "cash_benchmark_return": cash_return,
        "strategy_excess_vs_btc": strategy_return - btc_return,
        "strategy_excess_vs_dynamic_universe": strategy_return - universe_return,
        "max_drawdown": max_drawdown,
        "top_symbol": top_symbol,
        "maximum_single_symbol_contribution": top_share,
        "paper_status": paper_status,
        "parameter_lock_hash": lock["sha256"],
        "strategy_code_commit": lock["strategy_code_commit"],
        "strategy_code_hash": lock["strategy_code_hash"],
        "reporting_code_hash": lock["reporting_code_hash"],
        "runner_version": RUNNER_VERSION,
    }
    status = {
        "schema_version": "quant_lab_forward_v22_status.v1",
        **row,
        "runner_integrity": "PASS" if not integrity_errors else "FAIL",
        "git_head_check": identity.get("current_head") == lock["strategy_code_commit"],
        "strategy_code_hash_check": identity.get("strategy_code_hash")
        == lock["strategy_code_hash"],
        "working_tree_clean": bool(identity.get("working_tree_clean")),
        "integrity_errors": integrity_errors,
        "recovery_records_excluded": True,
        "benchmark_records_immutable": True,
        "portfolio_validity": "INCONCLUSIVE",
        "deployment_readiness": "INCONCLUSIVE",
        "production_alpha": "FROZEN",
        "live": "NOT_ALLOWED",
        "live_order_effect": "none",
        "automatic_promotion": False,
        "thresholds": {
            key: lock[key]
            for key in (
                "minimum_forward_days",
                "minimum_completed_cycles",
                "minimum_symbol_trades",
                "minimum_data_coverage",
                "maximum_single_symbol_contribution",
                "maximum_drawdown",
            )
        },
    }
    return pl.DataFrame([row]), status, equity, benchmark_equity


def _paths(root: Path) -> dict[str, Path]:
    artifact = root / "artifacts"
    return {
        "decisions": artifact / "forward_v22_decisions.parquet",
        "trades": artifact / "forward_v22_trades.parquet",
        "events": artifact / "forward_v22_events.parquet",
        "equity": artifact / "forward_v22_equity.parquet",
        "benchmark_decisions": artifact / "forward_v22_benchmark_decisions.parquet",
        "benchmark_trades": artifact / "forward_v22_benchmark_trades.parquet",
        "benchmark_events": artifact / "forward_v22_benchmark_events.parquet",
        "benchmark_equity": artifact / "forward_v22_benchmark_equity.parquet",
        "performance": artifact / "forward_v22_performance.csv",
        "status": artifact / "forward_v22_status.json",
    }


def _prior_status(root: Path) -> dict[str, Any]:
    path = root / "artifacts/forward_v22_status.json"
    if not path.is_file():
        return {}
    try:
        return _load(path)
    except (OSError, ValueError, TypeError):
        return {}


def _failure_status(
    *,
    root: Path,
    lock: Mapping[str, Any],
    cutoff: datetime,
    identity: Mapping[str, Any] | None,
    paper_status: str,
    failure_code: str,
    failure_fingerprint: str,
    dry_run: bool,
) -> dict[str, Any]:
    previous = _prior_status(root)
    identity = dict(identity or {})
    defaults = {
        "schema_version": "quant_lab_forward_v22_status.v1",
        "audit_version": "v2.2",
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "forward_v22_cutoff": utc(cutoff).isoformat(),
        "real_time_decision_count": 0,
        "reconstructed_decision_count": 0,
        "eligible_decision_count": 0,
        "ineligible_decision_count": 0,
        "actual_entry_count": 0,
        "completed_independent_cycles": 0,
        "actual_symbol_trades": 0,
        "forward_days": 0.0,
        "data_coverage": 0.0,
        "feature_data_incomplete_count": 0,
        "strategy_net_return": 0.0,
        "btc_benchmark_return": 0.0,
        "dynamic_universe_benchmark_return": 0.0,
        "cash_benchmark_return": 0.0,
        "strategy_excess_vs_btc": 0.0,
        "strategy_excess_vs_dynamic_universe": 0.0,
        "max_drawdown": 0.0,
        "top_symbol": "",
        "maximum_single_symbol_contribution": 0.0,
        "market_data_cutoff": "",
        "market_data_snapshot_id": "",
        "parameter_lock_hash": lock.get("sha256", ""),
        "strategy_code_commit": lock.get("strategy_code_commit", ""),
        "strategy_code_hash": lock.get("strategy_code_hash", ""),
        "reporting_code_hash": lock.get("reporting_code_hash", ""),
        "runner_version": RUNNER_VERSION,
        "recovery_records_excluded": True,
        "benchmark_records_immutable": True,
        "portfolio_validity": "INCONCLUSIVE",
        "deployment_readiness": "INCONCLUSIVE",
        "production_alpha": "FROZEN",
        "live": "NOT_ALLOWED",
        "live_order_effect": "none",
        "automatic_promotion": False,
        "thresholds": {
            key: lock.get(key)
            for key in (
                "minimum_forward_days",
                "minimum_completed_cycles",
                "minimum_symbol_trades",
                "minimum_data_coverage",
                "maximum_single_symbol_contribution",
                "maximum_drawdown",
            )
        },
    }
    status = {
        **defaults,
        **previous,
        "paper_status": paper_status,
        "runner_integrity": "FAIL"
        if paper_status == "FAIL_RUNNER_INTEGRITY"
        else "PASS",
        "git_head_check": identity.get("current_head")
        == lock.get("strategy_code_commit"),
        "strategy_code_hash_check": identity.get("strategy_code_hash")
        == lock.get("strategy_code_hash"),
        "working_tree_clean": bool(identity.get("working_tree_clean")),
        "integrity_errors": list(identity.get("errors", []))
        if paper_status == "FAIL_RUNNER_INTEGRITY"
        else [],
        "failure_code": failure_code,
        "failure_fingerprint": failure_fingerprint,
        "evidence_mutated": False,
    }
    if not dry_run:
        atomic_write_json(status, root / "artifacts/forward_v22_status.json")
    return status


def _integrity_failure(
    *,
    root: Path,
    lock: Mapping[str, Any],
    cutoff: datetime,
    identity: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    errors = list(identity.get("errors", []))
    return _failure_status(
        root=root,
        lock=lock,
        cutoff=cutoff,
        identity=identity,
        paper_status="FAIL_RUNNER_INTEGRITY",
        failure_code=errors[0] if errors else "RUNNER_IDENTITY_INVALID",
        failure_fingerprint=payload_digest(errors),
        dry_run=dry_run,
    )


def run(
    *,
    root: Path,
    v1_root: Path,
    repo: Path,
    mode: str,
    requested_as_of: datetime,
    recorded_at: datetime | None,
    resume: bool,
    dry_run: bool,
    no_fetch: bool,
) -> dict[str, Any]:
    lock = _load(root / "manifests/parameter_lock_v22.json")
    cutoff_manifest = _load(root / "manifests/forward_v22_cutoff.json")
    try:
        validate_parameter_lock(lock)
    except Exception as exc:
        raise RunnerIntegrityError("parameter lock validation failed") from exc
    cutoff = utc(cutoff_manifest["forward_v22_cutoff"])
    if cutoff_manifest["parameter_lock_hash"] != lock["sha256"]:
        raise RunnerIntegrityError("cutoff and parameter lock hashes differ")
    identity = runtime_identity(repo, lock)
    if not identity["ok"]:
        return _integrity_failure(
            root=root, lock=lock, cutoff=cutoff, identity=identity, dry_run=dry_run
        )
    wall_clock = utc(recorded_at or datetime.now(UTC).replace(microsecond=0))
    requested = utc(requested_as_of)
    if mode == "realtime" and (
        requested < wall_clock - timedelta(seconds=int(lock["max_decision_latency_seconds"]))
        or requested > wall_clock + timedelta(minutes=5)
    ):
        invalid = dict(identity)
        invalid["errors"] = [*identity["errors"], "AS_OF_CANNOT_BACKDATE_REALTIME"]
        invalid["ok"] = False
        return _integrity_failure(
            root=root, lock=lock, cutoff=cutoff, identity=invalid, dry_run=dry_run
        )
    paths = _paths(root)
    evidence_exists = any(
        paths[name].exists()
        for name in (
            "decisions",
            "trades",
            "events",
            "benchmark_decisions",
            "benchmark_trades",
            "benchmark_events",
        )
    )
    if evidence_exists and not resume and not dry_run:
        raise RuntimeError("v2.2 evidence exists; use --resume")
    historical_path = v1_root / "data/silver/bars_1h.parquet"
    if not historical_path.is_file():
        raise DataIncompleteError("the locked historical market input is unavailable")
    historical = pl.read_parquet(historical_path).select(BAR_COLUMNS)
    forward, cache_path = load_or_fetch_market(
        root=root,
        historical=historical,
        fetch_after=forward_market_fetch_start(cutoff),
        as_of=requested,
        no_fetch=no_fetch,
        persist_cache=not dry_run,
    )
    frames = [historical, forward] if not forward.is_empty() else [historical]
    bars = (
        pl.concat(frames, how="vertical_relaxed")
        .unique(subset=["symbol", "ts"], keep="last")
        .sort(["symbol", "ts"])
    )
    observed_cutoff = available_market_cutoff(bars, requested)
    snapshot = build_market_snapshot(
        input_paths=(historical_path, cache_path),
        market_cutoff=observed_cutoff,
        recorded_at=wall_clock,
    )

    decisions = _read(paths["decisions"], DECISION_SCHEMA)
    trades = _read(paths["trades"], TRADE_SCHEMA)
    events = _read(paths["events"], EVENT_SCHEMA)
    benchmark_decisions = _read(paths["benchmark_decisions"], BENCHMARK_DECISION_SCHEMA)
    benchmark_trades = _read(paths["benchmark_trades"], BENCHMARK_TRADE_SCHEMA)
    benchmark_events = _read(paths["benchmark_events"], EVENT_SCHEMA)
    try:
        validate_event_chain(events)
        validate_event_chain(benchmark_events)
    except ValueError as exc:
        raise RunnerIntegrityError("existing event hash chain is invalid") from exc

    entry_trades, new_benchmarks, new_benchmark_trades, entry_events = create_entries(
        decisions=decisions,
        existing_trades=trades,
        existing_benchmarks=benchmark_decisions,
        existing_benchmark_trades=benchmark_trades,
        bars=bars,
        observed_cutoff=observed_cutoff,
        recorded_at=wall_clock,
        lock=lock,
    )
    strategy_entry_ids = (
        set(entry_trades["trade_id"].to_list())
        if not entry_trades.is_empty()
        else set()
    )
    strategy_entry_events = entry_events.filter(pl.col("trade_id").is_in(strategy_entry_ids))
    benchmark_entry_events = entry_events.filter(~pl.col("trade_id").is_in(strategy_entry_ids))
    trades = _integrity_boundary(
        "strategy trade history changed", merge_trade_states, trades, entry_trades
    )
    benchmark_decisions = _integrity_boundary(
        "benchmark decision history changed",
        merge_benchmark_decision_states,
        benchmark_decisions,
        new_benchmarks,
    )
    benchmark_trades = _integrity_boundary(
        "benchmark trade history changed",
        merge_benchmark_trade_states,
        benchmark_trades,
        new_benchmark_trades,
    )

    trade_updates, exit_events = update_strategy_exits(
        trades=trades,
        bars=bars,
        observed_cutoff=observed_cutoff,
        recorded_at=wall_clock,
        lock=lock,
    )
    trades = _integrity_boundary(
        "closed strategy trade changed", merge_trade_states, trades, trade_updates
    )
    benchmark_decision_updates, benchmark_trade_updates, benchmark_exit_events = (
        update_benchmark_exits(
            decisions=benchmark_decisions,
            trades=benchmark_trades,
            bars=bars,
            observed_cutoff=observed_cutoff,
            recorded_at=wall_clock,
            lock=lock,
        )
    )
    benchmark_decisions = _integrity_boundary(
        "closed benchmark decision changed",
        merge_benchmark_decision_states,
        benchmark_decisions,
        benchmark_decision_updates,
    )
    benchmark_trades = _integrity_boundary(
        "closed benchmark trade changed",
        merge_benchmark_trade_states,
        benchmark_trades,
        benchmark_trade_updates,
    )
    events = _integrity_boundary(
        "strategy entry event chain changed",
        append_hash_chain,
        events,
        strategy_entry_events,
    )
    events = _integrity_boundary(
        "strategy exit event chain changed", append_hash_chain, events, exit_events
    )
    benchmark_events = _integrity_boundary(
        "benchmark entry event chain changed",
        append_hash_chain,
        benchmark_events,
        benchmark_entry_events,
    )
    benchmark_events = _integrity_boundary(
        "benchmark exit event chain changed",
        append_hash_chain,
        benchmark_events,
        benchmark_exit_events,
    )

    new_decisions, decision_events = create_decisions(
        mode=mode,
        lock=lock,
        cutoff=cutoff,
        observed_cutoff=observed_cutoff,
        recorded_at=wall_clock,
        bars=bars,
        existing=decisions,
        snapshot=snapshot,
    )
    decisions = _integrity_boundary(
        "decision history changed",
        merge_immutable_rows,
        decisions,
        new_decisions,
        schema=DECISION_SCHEMA,
        id_field="decision_id",
        sort_fields=("scheduled_run_ts",),
        label="decision",
    )
    events = _integrity_boundary(
        "decision event chain changed", append_hash_chain, events, decision_events
    )
    performance, status, equity, benchmark_equity = _integrity_boundary(
        "formal evidence accounting failed integrity validation",
        performance_and_status,
        decisions=decisions,
        trades=trades,
        events=events,
        benchmarks=benchmark_decisions,
        benchmark_events=benchmark_events,
        cutoff=cutoff,
        observed_cutoff=observed_cutoff,
        snapshot=snapshot,
        lock=lock,
        identity=identity,
    )
    if dry_run:
        return status
    update_snapshot_registry(root, snapshot)
    atomic_write_parquet(decisions, paths["decisions"])
    atomic_write_parquet(trades, paths["trades"])
    atomic_write_parquet(events, paths["events"])
    atomic_write_parquet(equity, paths["equity"])
    atomic_write_parquet(benchmark_decisions, paths["benchmark_decisions"])
    atomic_write_parquet(benchmark_trades, paths["benchmark_trades"])
    atomic_write_parquet(benchmark_events, paths["benchmark_events"])
    atomic_write_parquet(benchmark_equity, paths["benchmark_equity"])
    atomic_write_csv(performance, paths["performance"])
    atomic_write_json(status, paths["status"])
    return status


def _record_exception_status(
    *,
    root: Path,
    repo: Path,
    error: Exception,
    dry_run: bool,
) -> dict[str, Any]:
    try:
        lock = _load(root / "manifests/parameter_lock_v22.json")
    except Exception:
        lock = {}
    try:
        cutoff = utc(
            _load(root / "manifests/forward_v22_cutoff.json")["forward_v22_cutoff"]
        )
    except Exception:
        cutoff = datetime.now(UTC).replace(microsecond=0)
    identity = runtime_identity(repo, lock) if lock else {}
    if isinstance(error, RunnerIntegrityError):
        paper_status = "FAIL_RUNNER_INTEGRITY"
        failure_code = "RUNNER_INTEGRITY_EXCEPTION"
        identity = dict(identity)
        identity["errors"] = [
            *identity.get("errors", []),
            failure_code,
        ]
    elif isinstance(error, DataIncompleteError):
        paper_status = "INCONCLUSIVE_DATA_INCOMPLETE"
        failure_code = "FORWARD_DATA_INCOMPLETE"
    else:
        paper_status = "INCONCLUSIVE_SYSTEM_ERROR"
        failure_code = "FORWARD_RUNNER_SYSTEM_ERROR"
    fingerprint = payload_digest(
        {"exception_type": type(error).__name__, "message": str(error)}
    )
    return _failure_status(
        root=root,
        lock=lock,
        cutoff=cutoff,
        identity=identity,
        paper_status=paper_status,
        failure_code=failure_code,
        failure_fingerprint=fingerprint,
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("realtime", "recovery"), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("AUDIT_V22_ROOT", "/home/hr/quant-alpha-audit-v2.2")),
    )
    parser.add_argument(
        "--v1-root",
        type=Path,
        default=Path(os.environ.get("AUDIT_V1_ROOT", "/home/hr/quant-alpha-audit")),
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--as-of")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    failed = False
    try:
        if args.report_only:
            status = _load(root / "artifacts/forward_v22_status.json")
        else:
            status = run(
                root=root,
                v1_root=args.v1_root.resolve(),
                repo=args.repo.resolve(),
                mode=args.mode,
                requested_as_of=utc(
                    args.as_of or datetime.now(UTC).replace(microsecond=0).isoformat()
                ),
                recorded_at=None,
                resume=args.resume,
                dry_run=args.dry_run,
                no_fetch=args.no_fetch,
            )
    except Exception as exc:
        failed = True
        status = _record_exception_status(
            root=root,
            repo=args.repo.resolve(),
            error=exc,
            dry_run=args.dry_run,
        )
    failed = failed or status.get("paper_status") == "FAIL_RUNNER_INTEGRITY"
    print(f"forward_v22_status={status['paper_status']}")
    print(f"real_time_decision_count={status['real_time_decision_count']}")
    print(f"reconstructed_decision_count={status['reconstructed_decision_count']}")
    print(f"eligible_decision_count={status['eligible_decision_count']}")
    print(f"actual_entry_count={status['actual_entry_count']}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
