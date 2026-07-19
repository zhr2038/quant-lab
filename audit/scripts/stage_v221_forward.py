# ruff: noqa: E501
"""Run Audit v2.2.1 realtime or recovery Forward Paper accounting.

Realtime and recovery share causal computation but never share formal metrics.
The hourly systemd invocation is separately logged from the locked 120-hour
strategy schedule. All writes are atomic and evidence is committed only after
identity, market, benchmark, and hash-chain validation succeeds.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.factors import low_vol_480  # noqa: E402
from audit.auditlib.forward_v221 import (  # noqa: E402
    BAR_CLOSE_DELAY_HOURS,
    BENCHMARK_COVERAGE_SCHEMA,
    BENCHMARK_DECISION_SCHEMA,
    BENCHMARK_EQUITY_SCHEMA,
    BENCHMARK_TRADE_SCHEMA,
    BENCHMARK_TYPES,
    BTC_TREND_LOOKBACK_HOURS,
    DECISION_SCHEMA,
    ENTRY_COST_BPS,
    EQUITY_SCHEMA,
    EVENT_SCHEMA,
    EXIT_COST_BPS,
    FACTOR_LOOKBACK_HOURS,
    HOLDING_HOURS,
    MARKET_LOOKBACK_DAYS,
    RUNNER_VERSION,
    SCHEDULE_EVENT_SCHEMA,
    STRATEGY_ID,
    STRATEGY_VERSION,
    TRADE_SCHEMA,
    append_hash_chain,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    benchmark_entry_records,
    canonical_json,
    classify_decision_origin,
    compounded_return,
    cost_adjusted_return,
    due_strategy_schedules,
    empty_frame,
    evaluate_forward_status,
    evidence_partitions,
    expected_timer_slots,
    formal_benchmark_period_returns,
    formal_cycle_period_returns,
    frame,
    make_event,
    make_schedule_event,
    merge_state_rows,
    next_strategy_schedule,
    payload_digest,
    runtime_identity,
    schedule_statistics,
    sha256_file,
    sha256_text,
    stable_benchmark_id,
    stable_decision_id,
    stable_trade_id,
    timer_slot,
    utc,
    validate_hash_chain,
    validate_parameter_lock,
)
from audit.auditlib.portfolio_backtest import _capped_weights  # noqa: E402
from audit.auditlib.universe import UNIVERSES, build_daily_universe  # noqa: E402

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


class RunnerIntegrityError(RuntimeError):
    """Locked code, deployment identity, or prior evidence cannot be trusted."""


class DataIncompleteError(RuntimeError):
    """Required causal market input is unavailable."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read(path: Path, schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else empty_frame(schema)


def _paths(root: Path) -> dict[str, Path]:
    artifact = root / "artifacts"
    return {
        "decisions": artifact / "forward_v221_decisions.parquet",
        "trades": artifact / "forward_v221_trades.parquet",
        "events": artifact / "forward_v221_events.parquet",
        "equity": artifact / "forward_v221_equity.parquet",
        "benchmark_decisions": artifact / "forward_v221_benchmark_decisions.parquet",
        "benchmark_trades": artifact / "forward_v221_benchmark_trades.parquet",
        "benchmark_events": artifact / "forward_v221_benchmark_events.parquet",
        "benchmark_equity": artifact / "forward_v221_benchmark_equity.parquet",
        "benchmark_coverage": artifact / "forward_v221_benchmark_coverage.parquet",
        "schedule_events": artifact / "forward_v221_schedule_events.parquet",
        "performance": artifact / "forward_v221_performance.csv",
        "status": artifact / "forward_v221_status.json",
    }


def _market_bars(path: Path, as_of: datetime) -> tuple[pl.DataFrame, dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise DataIncompleteError(f"market source is unavailable: {path}")
    before_hash = sha256_file(path)
    requested = utc(as_of)
    start = requested - timedelta(days=MARKET_LOOKBACK_DAYS)
    schema = pl.scan_parquet(path).collect_schema()
    required = set(BAR_COLUMNS)
    if not required.issubset(schema.names()):
        raise DataIncompleteError(f"market schema is missing: {sorted(required - set(schema.names()))}")
    lazy = pl.scan_parquet(path)
    predicates = [
        pl.col("ts") >= start,
        pl.col("ts") + pl.duration(hours=BAR_CLOSE_DELAY_HOURS) <= requested,
    ]
    if "venue" in schema:
        predicates.append(pl.col("venue") == "okx")
    if "market_type" in schema:
        predicates.append(pl.col("market_type") == "SPOT")
    if "timeframe" in schema:
        predicates.append(pl.col("timeframe") == "1H")
    if "is_closed" in schema:
        predicates.append(pl.col("is_closed"))
    condition = predicates[0]
    for predicate in predicates[1:]:
        condition &= predicate
    bars = (
        lazy.filter(condition)
        .select(BAR_COLUMNS)
        .collect(engine="streaming")
        .unique(subset=["symbol", "ts"], keep="last")
        .sort(["symbol", "ts"])
    )
    after_hash = sha256_file(path)
    if before_hash != after_hash:
        raise RunnerIntegrityError("market source changed during snapshot read")
    if bars.is_empty():
        raise DataIncompleteError("no completed 1H spot bars are available")
    btc = bars.filter(pl.col("symbol") == "BTC-USDT")
    if btc.is_empty():
        raise DataIncompleteError("BTC-USDT market bars are unavailable")
    market_cutoff = utc(btc["ts"].max()) + timedelta(hours=BAR_CLOSE_DELAY_HOURS)
    buffer = io.BytesIO()
    bars.write_parquet(buffer, compression="zstd")
    snapshot_bytes = buffer.getvalue()
    snapshot_hash = payload_digest(
        {
            "source_path": str(path.resolve()),
            "source_sha256": before_hash,
            "market_data_cutoff": market_cutoff.isoformat(),
            "projected_sha256": __import__("hashlib").sha256(snapshot_bytes).hexdigest(),
            "row_count": bars.height,
        }
    )
    snapshot_id = f"market_v221_{snapshot_hash[:24]}"
    snapshot_path = Path("state/market_snapshots") / f"{snapshot_id}.parquet"
    snapshot = {
        "snapshot_id": snapshot_id,
        "market_data_cutoff": market_cutoff,
        "source_path": str(path.resolve()),
        "source_sha256": before_hash,
        "projected_path": snapshot_path.as_posix(),
        "projected_sha256": __import__("hashlib").sha256(snapshot_bytes).hexdigest(),
        "row_count": bars.height,
        "symbol_count": bars.select(pl.col("symbol").n_unique()).item(),
        "first_bar_ts": utc(bars["ts"].min()).isoformat(),
        "last_bar_ts": utc(bars["ts"].max()).isoformat(),
    }
    return bars, snapshot, snapshot_bytes


def _persist_snapshot(root: Path, snapshot: Mapping[str, Any], raw: bytes) -> Path:
    path = root / str(snapshot["projected_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if __import__("hashlib").sha256(path.read_bytes()).hexdigest() != snapshot["projected_sha256"]:
            raise RunnerIntegrityError("immutable market snapshot changed")
        return path
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_bytes(raw)
    os.replace(partial, path)
    return path


def _update_snapshot_registry(root: Path, snapshot: Mapping[str, Any], persisted: Path) -> None:
    path = root / "manifests/market_data_snapshot_v221.json"
    registry = (
        _load(path)
        if path.exists()
        else {"schema_version": "quant_lab_market_data_snapshot_v221.v1", "snapshots": []}
    )
    normalized = {
        **{
            key: (utc(value).isoformat() if isinstance(value, datetime) else value)
            for key, value in snapshot.items()
        },
        "projected_path": str(persisted.resolve()),
    }
    by_id = {str(row["snapshot_id"]): row for row in registry.get("snapshots", [])}
    prior = by_id.get(str(snapshot["snapshot_id"]))
    if prior is not None and prior != normalized:
        raise RunnerIntegrityError("market snapshot registry entry changed")
    by_id[str(snapshot["snapshot_id"])] = normalized
    registry["snapshots"] = sorted(by_id.values(), key=lambda row: row["market_data_cutoff"])
    registry["current_snapshot_id"] = snapshot["snapshot_id"]
    atomic_write_json(registry, path)


def _complete_window_symbols(
    bars: pl.DataFrame, *, end_bar_ts: datetime, hours: int
) -> set[str]:
    end = utc(end_bar_ts)
    start = end - timedelta(hours=hours - 1)
    coverage = (
        bars.filter((pl.col("ts") >= start) & (pl.col("ts") <= end))
        .group_by("symbol")
        .agg(
            pl.col("ts").n_unique().alias("count"),
            pl.col("ts").min().alias("first"),
            pl.col("ts").max().alias("last"),
        )
        .filter(
            (pl.col("count") == hours)
            & (pl.col("first") == start)
            & (pl.col("last") == end)
        )
    )
    return set(coverage["symbol"].to_list())


def _btc_state(bars: pl.DataFrame, feature_bar_ts: datetime) -> str:
    complete = _complete_window_symbols(
        bars.filter(pl.col("symbol") == "BTC-USDT"),
        end_bar_ts=feature_bar_ts,
        hours=BTC_TREND_LOOKBACK_HOURS,
    )
    if "BTC-USDT" not in complete:
        return "UNAVAILABLE"
    row = (
        bars.filter(pl.col("symbol") == "BTC-USDT")
        .sort("ts")
        .with_columns(
            pl.col("close").rolling_mean(BTC_TREND_LOOKBACK_HOURS).alias("btc_ma_60d")
        )
        .filter(pl.col("ts") == utc(feature_bar_ts))
    )
    if row.is_empty() or row["btc_ma_60d"][0] is None:
        return "UNAVAILABLE"
    return "UP" if float(row["close"][0]) >= float(row["btc_ma_60d"][0]) else "DOWN"


def _exact_close(
    bars: pl.DataFrame, symbol: str, close_ts: datetime
) -> tuple[datetime, float] | None:
    bar_ts = utc(close_ts) - timedelta(hours=BAR_CLOSE_DELAY_HOURS)
    row = bars.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("ts") == bar_ts)
        & pl.col("close").is_finite()
    )
    if row.is_empty():
        return None
    return utc(close_ts), float(row["close"][0])


def _decision_candidates(
    *,
    mode: str,
    cutoff: datetime,
    observed_cutoff: datetime,
    recorded_at: datetime,
    existing: pl.DataFrame,
) -> list[datetime]:
    due = due_strategy_schedules(cutoff, min(utc(observed_cutoff), utc(recorded_at)))
    seen = set(existing["scheduled_run_ts"].to_list()) if not existing.is_empty() else set()
    missing = [schedule for schedule in due if schedule not in seen]
    if mode == "recovery":
        return missing
    if mode != "realtime":
        raise ValueError(f"unsupported mode: {mode}")
    return missing[-1:] if missing else []


def create_decisions(
    *,
    effective_mode: str,
    run_schedule_id: str,
    lock: Mapping[str, Any],
    cutoff: datetime,
    observed_cutoff: datetime,
    recorded_at: datetime,
    bars: pl.DataFrame,
    existing: pl.DataFrame,
    snapshot: Mapping[str, Any],
    root: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    snapshot_path = root / str(snapshot["projected_path"])
    input_paths = [str(snapshot["source_path"]), str(snapshot_path.resolve())]
    input_hashes = {
        str(snapshot["source_path"]): str(snapshot["source_sha256"]),
        str(snapshot_path.resolve()): str(snapshot["projected_sha256"]),
    }
    for scheduled in _decision_candidates(
        mode=effective_mode,
        cutoff=cutoff,
        observed_cutoff=observed_cutoff,
        recorded_at=recorded_at,
        existing=existing,
    ):
        feature_bar_ts = scheduled - timedelta(hours=BAR_CLOSE_DELAY_HOURS)
        causal = bars.filter(pl.col("ts") <= feature_bar_ts)
        signals = low_vol_480(causal)
        universe = build_daily_universe(causal, UNIVERSES["top20"])
        membership = universe.filter(pl.col("date") == scheduled.date()).sort("rank")
        complete_symbols = _complete_window_symbols(
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
        origin, late, timely, latency = classify_decision_origin(
            mode=effective_mode,
            scheduled_run_ts=scheduled,
            recorded_at=recorded_at,
            max_latency=int(lock["max_decision_latency_seconds"]),
        )
        decision_id = stable_decision_id(
            scheduled, str(snapshot["snapshot_id"]), str(lock["sha256"])
        )
        row = {
            "decision_id": decision_id,
            "schedule_id": run_schedule_id,
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
            "decision_ts": scheduled,
            "scheduled_run_ts": scheduled,
            "recorded_at": recorded_at,
            "decision_latency_seconds": latency,
            "decision_origin": origin,
            "late_reconstructed": late,
            "eligible_for_forward_evidence": timely,
            "cycle_eligible_for_forward_evidence": False,
            "btc_benchmark_cycle_complete": False,
            "universe_benchmark_cycle_complete": False,
            "cash_benchmark_cycle_complete": False,
            "all_benchmarks_complete": False,
            "feature_cutoff_ts": scheduled,
            "observed_market_data_cutoff": observed_cutoff,
            "market_data_cutoff": observed_cutoff,
            "market_data_snapshot_id": str(snapshot["snapshot_id"]),
            "input_file_paths": canonical_json(input_paths),
            "input_file_sha256": canonical_json(input_hashes),
            "feature_data_coverage": feature_coverage,
            "data_quality_status": "PASS" if data_complete else "INCOMPLETE",
            "btc_trend_state": btc_state,
            "universe": canonical_json(membership["symbol"].to_list()),
            "factor_scores": canonical_json(
                factor.select(["symbol", "signal", "rank"]).to_dicts()
            ),
            "ranked_symbols": canonical_json(factor["symbol"].to_list()),
            "selected_symbols": canonical_json(selected),
            "target_weights": canonical_json(target_weights),
            "cash_weight": max(0.0, 1.0 - sum(target_weights.values())),
            "parameter_lock_hash": str(lock["sha256"]),
            "strategy_code_hash": str(lock["strategy_code_hash"]),
            "git_commit": str(lock["strategy_code_commit"]),
            "working_tree_clean": True,
            "status": "DATA_INCOMPLETE" if not data_complete else "DECISION_CREATED" if selected else "CASH",
        }
        rows.append(row)
        events.append(
            make_event(
                event_type="DECISION_CREATED",
                event_ts=scheduled,
                recorded_at=recorded_at,
                decision_id=decision_id,
                payload={
                    "decision_origin": origin,
                    "eligible_for_forward_evidence": timely,
                    "data_quality_status": row["data_quality_status"],
                    "btc_trend_state": btc_state,
                    "universe": membership["symbol"].to_list(),
                    "selected_symbols": selected,
                    "target_weights": target_weights,
                    "market_data_snapshot_id": snapshot["snapshot_id"],
                },
                lock=lock,
            )
        )
    return frame(rows, DECISION_SCHEMA), frame(events, EVENT_SCHEMA)


def create_entries(
    *,
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    benchmark_decisions: pl.DataFrame,
    bars: pl.DataFrame,
    observed_cutoff: datetime,
    recorded_at: datetime,
    snapshot: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    strategy_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    benchmark_trade_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    strategy_events: list[dict[str, Any]] = []
    benchmark_events: list[dict[str, Any]] = []
    existing_strategy = {(row["decision_id"], row["symbol"]) for row in trades.iter_rows(named=True)}
    existing_benchmarks = set(benchmark_decisions["benchmark_id"].to_list()) if not benchmark_decisions.is_empty() else set()
    for decision in decisions.sort("scheduled_run_ts").iter_rows(named=True):
        entry_due = utc(decision["scheduled_run_ts"]) + timedelta(hours=BAR_CLOSE_DELAY_HOURS)
        if observed_cutoff < entry_due:
            continue
        selected = json.loads(decision["selected_symbols"])
        weights = json.loads(decision["target_weights"])
        entry_latency = max(0.0, (utc(recorded_at) - entry_due).total_seconds())
        entry_timely = entry_latency <= float(lock["max_decision_latency_seconds"])
        evidence_eligible = bool(decision["eligible_for_forward_evidence"] and entry_timely)
        for symbol in selected:
            key = (decision["decision_id"], symbol)
            if key in existing_strategy:
                continue
            value = _exact_close(bars, symbol, entry_due)
            trade_id = stable_trade_id(str(decision["decision_id"]), symbol, entry_due)
            row = {
                "trade_id": trade_id,
                "decision_id": str(decision["decision_id"]),
                "decision_origin": str(decision["decision_origin"]),
                "symbol": symbol,
                "target_weight": float(weights[symbol]),
                "entry_ts": entry_due,
                "entry_price": value[1] if value else None,
                "scheduled_exit_ts": entry_due + timedelta(hours=HOLDING_HOURS),
                "actual_exit_ts": None,
                "exit_price": None,
                "entry_cost_bps": ENTRY_COST_BPS,
                "exit_cost_bps": EXIT_COST_BPS,
                "gross_return": None,
                "net_return": None,
                "entry_market_data_snapshot_id": str(snapshot["snapshot_id"]),
                "exit_market_data_snapshot_id": "",
                "eligible_for_forward_evidence": evidence_eligible,
                "cycle_eligible_for_forward_evidence": False,
                "parameter_lock_hash": str(lock["sha256"]),
                "strategy_code_hash": str(lock["strategy_code_hash"]),
                "git_commit": str(lock["strategy_code_commit"]),
                "status": "OPEN" if value else "MISSING_ENTRY",
            }
            strategy_rows.append(row)
            existing_strategy.add(key)
            strategy_events.append(
                make_event(
                    event_type="ENTRY_RECORDED" if value else "ENTRY_MISSING",
                    event_ts=entry_due,
                    recorded_at=recorded_at,
                    decision_id=str(decision["decision_id"]),
                    trade_id=trade_id,
                    payload={"symbol": symbol, "entry_price": value[1] if value else None, "eligible": evidence_eligible},
                    lock=lock,
                )
            )
        specs = {
            "BTC_BUY_AND_HOLD": ["BTC-USDT"],
            "DYNAMIC_UNIVERSE_EQUAL_WEIGHT": json.loads(decision["universe"]),
            "CASH": ["CASH"],
        }
        for benchmark_type, symbols in specs.items():
            benchmark_id = stable_benchmark_id(str(decision["decision_id"]), benchmark_type)
            if benchmark_id in existing_benchmarks:
                continue
            prices = {
                symbol: (entry_due, 1.0)
                if symbol == "CASH"
                else _exact_close(bars, symbol, entry_due)
                for symbol in symbols
            }
            entry_snapshot_decision = dict(decision)
            entry_snapshot_decision["market_data_snapshot_id"] = str(
                snapshot["snapshot_id"]
            )
            entry_snapshot_decision["market_data_cutoff"] = observed_cutoff
            benchmark, local_trades, coverage = benchmark_entry_records(
                decision=entry_snapshot_decision,
                benchmark_type=benchmark_type,
                expected_symbols=symbols,
                entry_prices=prices,
                entry_due=entry_due,
                lock=lock,
            )
            if not entry_timely:
                benchmark["eligible_for_forward_evidence"] = False
                for local in local_trades:
                    local["eligible_for_forward_evidence"] = False
            benchmark_rows.append(benchmark)
            benchmark_trade_rows.extend(local_trades)
            coverage_rows.append(coverage)
            existing_benchmarks.add(benchmark_id)
            benchmark_events.append(
                make_event(
                    event_type="BENCHMARK_DECISION_CREATED",
                    event_ts=decision["decision_ts"],
                    recorded_at=recorded_at,
                    decision_id=str(decision["decision_id"]),
                    trade_id=benchmark_id,
                    payload={
                        "benchmark_type": benchmark_type,
                        "expected_symbols": symbols,
                        "filled_symbol_count": benchmark["filled_symbol_count"],
                        "missing_symbol_count": benchmark["missing_symbol_count"],
                        "fill_coverage": benchmark["fill_coverage"],
                        "cash_residual": benchmark["cash_residual"],
                    },
                    lock=lock,
                )
            )
            for local in local_trades:
                benchmark_events.append(
                    make_event(
                        event_type="BENCHMARK_ENTRY_RECORDED" if local["entry_price_available"] else "BENCHMARK_ENTRY_MISSING",
                        event_ts=entry_due,
                        recorded_at=recorded_at,
                        decision_id=str(decision["decision_id"]),
                        trade_id=str(local["benchmark_trade_id"]),
                        payload={
                            "benchmark_type": benchmark_type,
                            "symbol": local["symbol"],
                            "expected_weight": local["expected_weight"],
                            "realized_weight": local["realized_weight"],
                            "missing_reason": local["missing_reason"],
                        },
                        lock=lock,
                    )
                )
    return (
        frame(strategy_rows, TRADE_SCHEMA),
        frame(benchmark_rows, BENCHMARK_DECISION_SCHEMA),
        frame(benchmark_trade_rows, BENCHMARK_TRADE_SCHEMA),
        frame(coverage_rows, BENCHMARK_COVERAGE_SCHEMA),
        frame(strategy_events, EVENT_SCHEMA),
        frame(benchmark_events, EVENT_SCHEMA),
    )


def update_exits(
    *,
    trades: pl.DataFrame,
    benchmark_decisions: pl.DataFrame,
    benchmark_trades: pl.DataFrame,
    bars: pl.DataFrame,
    observed_cutoff: datetime,
    recorded_at: datetime,
    snapshot: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    strategy_updates: list[dict[str, Any]] = []
    benchmark_trade_updates: list[dict[str, Any]] = []
    benchmark_updates: list[dict[str, Any]] = []
    strategy_events: list[dict[str, Any]] = []
    benchmark_events: list[dict[str, Any]] = []
    for prior in trades.iter_rows(named=True):
        row = dict(prior)
        if row["status"] != "OPEN" or observed_cutoff < row["scheduled_exit_ts"]:
            strategy_updates.append(row)
            continue
        value = _exact_close(bars, str(row["symbol"]), row["scheduled_exit_ts"])
        if value is None:
            row["status"] = "MISSING_EXIT"
            strategy_updates.append(row)
            continue
        gross = value[1] / float(row["entry_price"]) - 1.0
        row.update(
            {
                "actual_exit_ts": value[0],
                "exit_price": value[1],
                "gross_return": gross,
                "net_return": cost_adjusted_return(gross, row["entry_cost_bps"], row["exit_cost_bps"]),
                "exit_market_data_snapshot_id": str(snapshot["snapshot_id"]),
                "status": "CLOSED",
            }
        )
        strategy_updates.append(row)
        strategy_events.append(
            make_event(
                event_type="EXIT_RECORDED",
                event_ts=value[0],
                recorded_at=recorded_at,
                decision_id=str(row["decision_id"]),
                trade_id=str(row["trade_id"]),
                payload={"symbol": row["symbol"], "exit_price": row["exit_price"], "net_return": row["net_return"]},
                lock=lock,
            )
        )
    for prior in benchmark_trades.iter_rows(named=True):
        row = dict(prior)
        if row["status"] != "OPEN" or observed_cutoff < row["scheduled_exit_ts"]:
            benchmark_trade_updates.append(row)
            continue
        value = (utc(row["scheduled_exit_ts"]), 1.0) if row["symbol"] == "CASH" else _exact_close(bars, str(row["symbol"]), row["scheduled_exit_ts"])
        if value is None:
            row["status"] = "MISSING_EXIT"
            benchmark_trade_updates.append(row)
            continue
        gross = value[1] / float(row["entry_price"]) - 1.0
        row.update(
            {
                "actual_exit_ts": value[0],
                "exit_price": value[1],
                "gross_return": gross,
                "net_return": cost_adjusted_return(gross, row["entry_cost_bps"], row["exit_cost_bps"]),
                "exit_market_data_snapshot_id": str(snapshot["snapshot_id"]),
                "status": "CLOSED",
            }
        )
        benchmark_trade_updates.append(row)
        benchmark_events.append(
            make_event(
                event_type="BENCHMARK_EXIT_RECORDED",
                event_ts=value[0],
                recorded_at=recorded_at,
                decision_id=str(row["decision_id"]),
                trade_id=str(row["benchmark_trade_id"]),
                payload={"benchmark_type": row["benchmark_type"], "symbol": row["symbol"], "net_return": row["net_return"]},
                lock=lock,
            )
        )
    benchmark_trade_frame = frame(benchmark_trade_updates, BENCHMARK_TRADE_SCHEMA)
    for prior in benchmark_decisions.iter_rows(named=True):
        row = dict(prior)
        if row["status"] in {"CLOSED", "INCOMPLETE_ENTRY", "INCOMPLETE_EXIT"}:
            benchmark_updates.append(row)
            continue
        if observed_cutoff < row["scheduled_exit_ts"]:
            benchmark_updates.append(row)
            continue
        local = benchmark_trade_frame.filter(pl.col("benchmark_id") == row["benchmark_id"])
        filled = local.filter(pl.col("entry_price_available"))
        if filled.filter(pl.col("status") == "MISSING_EXIT").height:
            row["status"] = "INCOMPLETE_EXIT"
            benchmark_updates.append(row)
            continue
        if filled.is_empty() or not filled.select(pl.col("status").eq("CLOSED").all()).item():
            benchmark_updates.append(row)
            continue
        if filled.select(pl.col("actual_exit_ts").n_unique()).item() != 1 or utc(filled["actual_exit_ts"][0]) != utc(row["scheduled_exit_ts"]):
            row["status"] = "INCOMPLETE_EXIT"
            benchmark_updates.append(row)
            continue
        gross = float((filled["realized_weight"] * filled["gross_return"]).sum())
        net = float((filled["realized_weight"] * filled["net_return"]).sum())
        row.update(
            {
                "actual_exit_ts": row["scheduled_exit_ts"],
                "gross_return": gross,
                "net_return": net,
                "exit_market_data_snapshot_id": str(snapshot["snapshot_id"]),
                "cycle_complete": True,
                "status": "CLOSED",
            }
        )
        benchmark_updates.append(row)
    return (
        frame(strategy_updates, TRADE_SCHEMA),
        frame(benchmark_updates, BENCHMARK_DECISION_SCHEMA),
        benchmark_trade_frame,
        frame(strategy_events, EVENT_SCHEMA),
        frame(benchmark_events, EVENT_SCHEMA),
    )


def finalize_cycles(
    *,
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    benchmarks: pl.DataFrame,
    observed_cutoff: datetime,
    lock: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str], int]:
    decision_updates: list[dict[str, Any]] = []
    trade_updates = trades.to_dicts()
    benchmark_updates = benchmarks.to_dicts()
    completed: list[str] = []
    incomplete_benchmark_cycles = 0
    for prior in decisions.iter_rows(named=True):
        row = dict(prior)
        if not bool(row["eligible_for_forward_evidence"]):
            decision_updates.append(row)
            continue
        horizon_end = utc(row["scheduled_run_ts"]) + timedelta(hours=BAR_CLOSE_DELAY_HOURS + HOLDING_HOURS)
        if observed_cutoff < horizon_end:
            decision_updates.append(row)
            continue
        selected = json.loads(row["selected_symbols"])
        local_trades = trades.filter(pl.col("decision_id") == row["decision_id"])
        strategy_complete = not selected or (
            local_trades.height == len(selected)
            and local_trades.select(pl.col("status").eq("CLOSED").all()).item()
            and local_trades.select(pl.col("actual_exit_ts").n_unique()).item() == 1
            and utc(local_trades["actual_exit_ts"][0]) == horizon_end
        )
        local_benchmarks = benchmarks.filter(pl.col("decision_id") == row["decision_id"])
        benchmark_types = set(local_benchmarks["benchmark_type"].to_list())
        complete_types = set(
            local_benchmarks.filter(
                pl.col("cycle_complete") & (pl.col("status") == "CLOSED")
            )["benchmark_type"].to_list()
        )
        row["btc_benchmark_cycle_complete"] = "BTC_BUY_AND_HOLD" in complete_types
        row["universe_benchmark_cycle_complete"] = (
            "DYNAMIC_UNIVERSE_EQUAL_WEIGHT" in complete_types
        )
        row["cash_benchmark_cycle_complete"] = "CASH" in complete_types
        benchmark_entry_snapshots = set(
            local_benchmarks["entry_market_data_snapshot_id"].to_list()
        )
        benchmark_exit_snapshots = set(
            local_benchmarks["exit_market_data_snapshot_id"].to_list()
        )
        strategy_entry_snapshots = set(
            local_trades["entry_market_data_snapshot_id"].to_list()
        )
        strategy_exit_snapshots = set(
            local_trades["exit_market_data_snapshot_id"].to_list()
        )
        snapshots_match = (
            len(benchmark_entry_snapshots) == 1
            and len(benchmark_exit_snapshots) == 1
            and (
                not selected
                or (
                    strategy_entry_snapshots == benchmark_entry_snapshots
                    and strategy_exit_snapshots == benchmark_exit_snapshots
                )
            )
        )
        benchmarks_complete = (
            benchmark_types == set(BENCHMARK_TYPES)
            and local_benchmarks.select(pl.col("cycle_complete").all()).item()
            and local_benchmarks.select(pl.col("status").eq("CLOSED").all()).item()
            and local_benchmarks.select(pl.col("entry_ts").n_unique()).item() == 1
            and utc(local_benchmarks["entry_ts"][0]) == utc(row["scheduled_run_ts"]) + timedelta(hours=1)
            and local_benchmarks.select(pl.col("actual_exit_ts").n_unique()).item() == 1
            and utc(local_benchmarks["actual_exit_ts"][0]) == horizon_end
            and snapshots_match
        )
        row["all_benchmarks_complete"] = bool(benchmarks_complete)
        thresholds_ok = True
        for benchmark in local_benchmarks.iter_rows(named=True):
            required = 1.0 if benchmark["benchmark_type"] in {"BTC_BUY_AND_HOLD", "CASH"} else float(lock["minimum_benchmark_fill_coverage"])
            thresholds_ok = thresholds_ok and float(benchmark["fill_coverage"]) + 1e-12 >= required
        eligible_cycle = bool(strategy_complete and benchmarks_complete and thresholds_ok and row["data_quality_status"] == "PASS")
        row["cycle_eligible_for_forward_evidence"] = eligible_cycle
        row["status"] = "COMPLETE" if eligible_cycle else "CYCLE_INCOMPLETE"
        if eligible_cycle:
            completed.append(str(row["decision_id"]))
        elif not benchmarks_complete or not thresholds_ok:
            incomplete_benchmark_cycles += 1
        decision_updates.append(row)
        for trade in trade_updates:
            if trade["decision_id"] == row["decision_id"]:
                trade["cycle_eligible_for_forward_evidence"] = eligible_cycle
        for benchmark in benchmark_updates:
            if benchmark["decision_id"] == row["decision_id"]:
                benchmark["eligible_for_forward_evidence"] = bool(benchmark["eligible_for_forward_evidence"] and eligible_cycle)
    return (
        frame(decision_updates, DECISION_SCHEMA),
        frame(trade_updates, TRADE_SCHEMA),
        frame(benchmark_updates, BENCHMARK_DECISION_SCHEMA),
        completed,
        incomplete_benchmark_cycles,
    )


def coverage_frame(benchmarks: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark in benchmarks.iter_rows(named=True):
        local = trades.filter(pl.col("benchmark_id") == benchmark["benchmark_id"])
        details = [
            {
                "symbol": row["symbol"],
                "expected_weight": row["expected_weight"],
                "entry_price_available": row["entry_price_available"],
                "entry_price": row["entry_price"],
                "fill_status": row["fill_status"],
                "missing_reason": row["missing_reason"],
                "realized_weight": row["realized_weight"],
            }
            for row in local.iter_rows(named=True)
        ]
        rows.append(
            {
                "benchmark_id": benchmark["benchmark_id"],
                "decision_id": benchmark["decision_id"],
                "benchmark_type": benchmark["benchmark_type"],
                "expected_symbol_count": benchmark["expected_symbol_count"],
                "filled_symbol_count": benchmark["filled_symbol_count"],
                "missing_symbol_count": benchmark["missing_symbol_count"],
                "fill_coverage": benchmark["fill_coverage"],
                "invested_weight": benchmark["invested_weight"],
                "cash_residual": benchmark["cash_residual"],
                "cycle_complete": benchmark["cycle_complete"],
                "eligible_for_forward_evidence": benchmark["eligible_for_forward_evidence"],
                "missing_symbols": canonical_json([row["symbol"] for row in details if not row["entry_price_available"]]),
                "entry_details": canonical_json(details),
                "status": benchmark["status"],
            }
        )
    return frame(rows, BENCHMARK_COVERAGE_SCHEMA)


def performance(
    *,
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    benchmarks: pl.DataFrame,
    coverage: pl.DataFrame,
    schedule_events: pl.DataFrame,
    cutoff: datetime,
    observed_cutoff: datetime,
    snapshot: Mapping[str, Any],
    lock: Mapping[str, Any],
    identity: Mapping[str, Any],
    deployment: Mapping[str, Any],
    health: Mapping[str, Any],
    incomplete_benchmark_cycles: int,
) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame, pl.DataFrame]:
    partitions = evidence_partitions(decisions, trades, schedule_events)
    formal = partitions["formal_decisions"]
    recovery = partitions["recovery_decisions"]
    formal_trades = partitions["formal_trades"]
    recovery_trades = partitions["recovery_trades"]
    cycles = formal.filter(pl.col("cycle_eligible_for_forward_evidence"))
    formal_returns = formal_cycle_period_returns(decisions, trades)
    cycle_ids = [decision_id for decision_id, _value in formal_returns]
    period_returns = [value for _decision_id, value in formal_returns]
    strategy_return = compounded_return(period_returns)
    cycle_benchmarks = benchmarks.filter(
        pl.col("decision_id").is_in(cycle_ids)
        & pl.col("cycle_complete")
        & (pl.col("status") == "CLOSED")
    )

    def benchmark_return(name: str) -> float:
        values = [
            value
            for _decision_id, value in formal_benchmark_period_returns(
                decisions, benchmarks, name
            )
        ]
        return compounded_return(values)

    btc_return = benchmark_return("BTC_BUY_AND_HOLD")
    universe_return = benchmark_return("DYNAMIC_UNIVERSE_EQUAL_WEIGHT")
    cash_return = benchmark_return("CASH")
    schedule = schedule_statistics(schedule_events, cutoff=cutoff, through=observed_cutoff)
    formal_feature_coverage = float(formal["feature_data_coverage"].min()) if not formal.is_empty() else 0.0
    expected_strategy_trades = sum(len(json.loads(value)) for value in formal["selected_symbols"])
    filled_strategy_trades = formal_trades.height
    strategy_fill_coverage = filled_strategy_trades / expected_strategy_trades if expected_strategy_trades else (1.0 if not formal.is_empty() else 0.0)
    formal_data_coverage = min(formal_feature_coverage, strategy_fill_coverage)
    recovery_data_coverage = partitions["recovery_data_coverage"]
    formal_incomplete = partitions["formal_realtime_incomplete_decision_count"]
    recovery_incomplete = partitions["recovery_incomplete_decision_count"]
    formal_coverage = coverage.filter(pl.col("decision_id").is_in(formal["decision_id"]))
    min_benchmark_coverage = float(formal_coverage["fill_coverage"].min()) if not formal_coverage.is_empty() else 0.0
    dynamic = formal_coverage.filter(pl.col("benchmark_type") == "DYNAMIC_UNIVERSE_EQUAL_WEIGHT")
    cash_residual = float(dynamic["cash_residual"].max()) if not dynamic.is_empty() else 0.0
    completed_trades = formal_trades.filter(pl.col("cycle_eligible_for_forward_evidence") & (pl.col("status") == "CLOSED"))
    top_symbol, top_share = "", 0.0
    if not completed_trades.is_empty():
        contributions = completed_trades.group_by("symbol").agg((pl.col("target_weight") * pl.col("net_return")).sum().alias("value"))
        denominator = float(contributions["value"].abs().sum())
        if denominator > 1e-15:
            top = contributions.with_columns((pl.col("value").abs() / denominator).alias("share")).sort("share", descending=True).head(1)
            top_symbol, top_share = str(top["symbol"][0]), float(top["share"][0])
    equity_rows = [{"timestamp": utc(cutoff), "strategy_equity": 1.0, "completed_cycle_count": 0}]
    value = 1.0
    for index, (decision_id, period_return) in enumerate(zip(cycle_ids, period_returns, strict=True), start=1):
        value *= 1.0 + period_return
        local = cycle_benchmarks.filter(pl.col("decision_id") == decision_id)
        equity_rows.append({"timestamp": local["actual_exit_ts"].max(), "strategy_equity": value, "completed_cycle_count": index})
    equity = pl.DataFrame(equity_rows, schema=EQUITY_SCHEMA)
    benchmark_rows = [{"timestamp": utc(cutoff), "btc_equity": 1.0, "dynamic_universe_equity": 1.0, "cash_equity": 1.0, "completed_cycle_count": 0}]
    states = {name: 1.0 for name in BENCHMARK_TYPES}
    for index, decision_id in enumerate(cycle_ids, start=1):
        local = cycle_benchmarks.filter(pl.col("decision_id") == decision_id)
        if set(local["benchmark_type"].to_list()) != set(BENCHMARK_TYPES):
            continue
        for row in local.iter_rows(named=True):
            states[row["benchmark_type"]] *= 1.0 + float(row["net_return"])
        benchmark_rows.append({
            "timestamp": local["actual_exit_ts"].max(),
            "btc_equity": states["BTC_BUY_AND_HOLD"],
            "dynamic_universe_equity": states["DYNAMIC_UNIVERSE_EQUAL_WEIGHT"],
            "cash_equity": states["CASH"],
            "completed_cycle_count": index,
        })
    benchmark_equity = pl.DataFrame(benchmark_rows, schema=BENCHMARK_EQUITY_SCHEMA)
    peaks, max_drawdown = 1.0, 0.0
    for item in equity["strategy_equity"].to_list():
        peaks = max(peaks, float(item))
        max_drawdown = min(max_drawdown, float(item) / peaks - 1.0)
    metrics = {
        **schedule,
        "integrity_errors": list(identity.get("errors", [])),
        "timer_installed": deployment.get("timer_installed", False),
        "timer_enabled": deployment.get("timer_enabled", False),
        "timer_active": deployment.get("timer_active", False),
        "health_check_passed": health.get("health_status") == "PASS",
        "cutoff_exists": True,
        "formal_realtime_decision_count": formal.height,
        "formal_realtime_data_coverage": formal_data_coverage,
        "formal_realtime_incomplete_decision_count": formal_incomplete,
        "completed_independent_cycles": cycles.height,
        "actual_symbol_trades": completed_trades.height,
        "minimum_benchmark_fill_coverage": min_benchmark_coverage,
        "incomplete_formal_benchmark_cycle_count": incomplete_benchmark_cycles,
        "unexplained_critical_schedule_gap_count": schedule["missed_schedule_count"],
        "base_cost_net_return": strategy_return,
        "excess_vs_btc": strategy_return - btc_return,
        "excess_vs_dynamic_universe": strategy_return - universe_return,
        "max_drawdown": max_drawdown,
        "maximum_single_symbol_contribution": top_share,
    }
    paper_status = evaluate_forward_status(metrics=metrics, lock=lock)
    row = {
        "audit_version": "v2.2.1",
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "forward_v221_cutoff": utc(cutoff).isoformat(),
        "next_legal_strategy_decision": next_strategy_schedule(observed_cutoff).isoformat(),
        "market_data_cutoff": utc(observed_cutoff).isoformat(),
        "market_data_snapshot_id": snapshot["snapshot_id"],
        "formal_realtime_decision_count": formal.height,
        "formal_realtime_trade_count": formal_trades.height,
        "formal_realtime_data_coverage": formal_data_coverage,
        "formal_realtime_incomplete_decision_count": formal_incomplete,
        "formal_realtime_runner_errors": partitions["formal_realtime_runner_errors"],
        "recovery_decision_count": recovery.height,
        "recovery_trade_count": recovery_trades.height,
        "recovery_data_coverage": recovery_data_coverage,
        "recovery_incomplete_decision_count": recovery_incomplete,
        "recovery_runner_errors": partitions["recovery_runner_errors"],
        "recovery_audit_warning": bool(recovery.height or recovery_incomplete),
        **schedule,
        "cash_decision_count": formal.filter(pl.col("status").is_in(["CASH", "COMPLETE"])).filter(pl.col("selected_symbols") == "[]").height,
        "actual_entry_count": formal_trades.height,
        "completed_independent_cycles": cycles.height,
        "actual_symbol_trades": completed_trades.height,
        "btc_benchmark_complete_cycle_count": cycle_benchmarks.filter(pl.col("benchmark_type") == "BTC_BUY_AND_HOLD").height,
        "dynamic_universe_benchmark_complete_cycle_count": cycle_benchmarks.filter(pl.col("benchmark_type") == "DYNAMIC_UNIVERSE_EQUAL_WEIGHT").height,
        "cash_benchmark_complete_cycle_count": cycle_benchmarks.filter(pl.col("benchmark_type") == "CASH").height,
        "minimum_benchmark_fill_coverage": min_benchmark_coverage,
        "benchmark_cash_residual": cash_residual,
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
        "schema_version": "quant_lab_forward_v221_status.v1",
        **row,
        "forward_start_status": "FORWARD_V221_READY",
        "research_node": deployment["research_node"],
        "timer_installed": deployment["timer_installed"],
        "timer_enabled": deployment["timer_enabled"],
        "timer_active": deployment["timer_active"],
        "next_timer_trigger": health.get(
            "next_trigger", deployment["next_timer_trigger"]
        ),
        "runner_integrity": "PASS" if identity["ok"] else "FAIL",
        "git_head_match": identity["current_head"] == identity["expected_head"],
        "strategy_code_hash_match": identity["strategy_code_hash"] == identity["expected_strategy_code_hash"],
        "working_tree_clean": identity["working_tree_clean"],
        "unit_hash_match": bool(identity["service_unit_hash_match"] and identity["timer_unit_hash_match"] and identity["runner_script_hash_match"]),
        "integrity_errors": identity["errors"],
        "recovery_records_excluded": True,
        "benchmark_records_immutable": True,
        "portfolio_validity": "INCONCLUSIVE",
        "deployment_readiness": "INCONCLUSIVE",
        "execution_mode": "PAPER",
        "approval_state": "PAPER_ONLY",
        "production_alpha": "FROZEN",
        "live_opening_enabled": False,
        "live": "NOT_ALLOWED",
        "live_order_effect": "none",
        "automatic_promotion": False,
    }
    return pl.DataFrame([row], infer_schema_length=None), status, equity, benchmark_equity


def _schedule_prefix(
    *,
    existing: pl.DataFrame,
    lock: Mapping[str, Any],
    cutoff: datetime,
    actual_start: datetime,
    mode: str,
) -> tuple[pl.DataFrame, datetime, str, bool]:
    slot, persistent = timer_slot(actual_start)
    additions: list[dict[str, Any]] = []
    if mode == "realtime":
        existing_expected = set(existing.filter(pl.col("event_type") == "SCHEDULE_EXPECTED")["schedule_id"].to_list()) if not existing.is_empty() else set()
        terminal = set(existing.filter(pl.col("event_type").is_in(["RUN_COMPLETED", "RUN_FAILED", "RUN_MISSED"]))["schedule_id"].to_list()) if not existing.is_empty() else set()
        for expected in expected_timer_slots(cutoff, slot):
            schedule_id = sha256_text(
                f"{STRATEGY_ID}|timer|{expected.isoformat()}"
            )
            if schedule_id not in existing_expected:
                additions.append(make_schedule_event(event_type="SCHEDULE_EXPECTED", expected_run_ts=expected, actual_start_ts=None, actual_finish_ts=None, run_mode="REALTIME", run_status="EXPECTED", exit_code=0, eligible=False, market_data_cutoff=None, error_code="", lock=lock))
            if expected < slot and schedule_id not in terminal:
                additions.append(make_schedule_event(event_type="RUN_MISSED", expected_run_ts=expected, actual_start_ts=None, actual_finish_ts=slot, run_mode="REALTIME", run_status="MISSED", exit_code=0, eligible=False, market_data_cutoff=None, error_code="NO_RUN_OBSERVED", lock=lock))
    run_mode = "REALTIME" if mode == "realtime" else "RECOVERY"
    latency = max(0.0, (utc(actual_start) - slot).total_seconds())
    eligible_run = mode == "realtime" and not persistent and latency <= float(lock["max_decision_latency_seconds"])
    additions.append(make_schedule_event(event_type="RUN_STARTED", expected_run_ts=slot, actual_start_ts=actual_start, actual_finish_ts=None, run_mode=run_mode, run_status="STARTED", exit_code=0, eligible=False, market_data_cutoff=None, error_code="", lock=lock))
    if not eligible_run and mode == "realtime":
        additions.append(make_schedule_event(event_type="RUN_LATE", expected_run_ts=slot, actual_start_ts=actual_start, actual_finish_ts=None, run_mode=run_mode, run_status="LATE", exit_code=0, eligible=False, market_data_cutoff=None, error_code="PERSISTENT_CATCHUP" if persistent else "LATENCY_EXCEEDED", lock=lock))
    result = append_hash_chain(existing, frame(additions, SCHEDULE_EVENT_SCHEMA), schema=SCHEDULE_EVENT_SCHEMA, id_field="schedule_event_id")
    schedule_id = sha256_text(f"{STRATEGY_ID}|timer|{slot.isoformat()}")
    return result, slot, schedule_id, eligible_run


def run(
    *,
    root: Path,
    repo: Path,
    market_bar: Path,
    mode: str,
    requested_as_of: datetime,
    recorded_at: datetime | None,
    resume: bool,
    dry_run: bool,
    allow_no_cutoff: bool,
) -> dict[str, Any]:
    paths = _paths(root)
    lock = _load(root / "manifests/parameter_lock_v221.json")
    validate_parameter_lock(lock)
    deployment_path = root / "manifests/systemd_deployment_v221.json"
    cutoff_path = root / "manifests/forward_v221_cutoff.json"
    if not deployment_path.exists() or not cutoff_path.exists():
        if allow_no_cutoff and dry_run:
            _bars, snapshot, _raw = _market_bars(market_bar, requested_as_of)
            initial = (
                _load(paths["status"])
                if paths["status"].exists()
                else {
                    "paper_status": "INCONCLUSIVE_SYSTEM_NOT_DEPLOYED",
                    "formal_realtime_decision_count": 0,
                    "recovery_decision_count": 0,
                    "schedule_coverage": 0.0,
                    "completed_independent_cycles": 0,
                    "forward_start_status": "NOT_READY",
                    "production_alpha": "FROZEN",
                    "live": "NOT_ALLOWED",
                }
            )
            return {
                **initial,
                "market_data_cutoff": utc(snapshot["market_data_cutoff"]).isoformat(),
                "market_data_snapshot_id": snapshot["snapshot_id"],
                "dry_run": True,
                "evidence_mutated": False,
            }
        raise RunnerIntegrityError("deployment manifest and formal cutoff are required")
    deployment = _load(deployment_path)
    cutoff_manifest = _load(cutoff_path)
    cutoff = utc(cutoff_manifest["forward_v221_cutoff"])
    health = _load(root / "state/forward_v221_health_latest.json")
    identity = runtime_identity(
        repo=repo,
        lock=lock,
        installed_service=Path(deployment["service_unit_path"]),
        installed_timer=Path(deployment["timer_unit_path"]),
        deployment_manifest=deployment,
        cutoff_manifest=cutoff_manifest,
    )
    if not identity["ok"]:
        status = _load(paths["status"])
        status.update({"paper_status": "FAIL_RUNNER_INTEGRITY", "runner_integrity": "FAIL", "integrity_errors": identity["errors"], "evidence_mutated": False})
        if not dry_run:
            atomic_write_json(status, paths["status"])
        return status
    actual_start = utc(recorded_at or datetime.now(UTC).replace(microsecond=0))
    requested = utc(requested_as_of)
    if mode == "realtime" and (requested < actual_start - timedelta(seconds=int(lock["max_decision_latency_seconds"])) or requested > actual_start + timedelta(minutes=5)):
        raise RunnerIntegrityError("AS_OF_CANNOT_BACKDATE_REALTIME")
    evidence_exists = any(paths[name].exists() and _read(paths[name], schema).height for name, schema in (("decisions", DECISION_SCHEMA), ("trades", TRADE_SCHEMA), ("events", EVENT_SCHEMA)))
    if evidence_exists and not resume and not dry_run:
        raise RuntimeError("v2.2.1 evidence exists; use --resume")
    schedule_events = _read(paths["schedule_events"], SCHEDULE_EVENT_SCHEMA)
    validate_hash_chain(schedule_events, schema=SCHEDULE_EVENT_SCHEMA)
    schedule_events, slot, run_schedule_id, eligible_run = _schedule_prefix(existing=schedule_events, lock=lock, cutoff=cutoff, actual_start=actual_start, mode=mode)
    bars, snapshot, snapshot_bytes = _market_bars(market_bar, requested)
    observed_cutoff = utc(snapshot["market_data_cutoff"])
    decisions = _read(paths["decisions"], DECISION_SCHEMA)
    trades = _read(paths["trades"], TRADE_SCHEMA)
    events = _read(paths["events"], EVENT_SCHEMA)
    benchmark_decisions = _read(paths["benchmark_decisions"], BENCHMARK_DECISION_SCHEMA)
    benchmark_trades = _read(paths["benchmark_trades"], BENCHMARK_TRADE_SCHEMA)
    benchmark_events = _read(paths["benchmark_events"], EVENT_SCHEMA)
    validate_hash_chain(events, schema=EVENT_SCHEMA)
    validate_hash_chain(benchmark_events, schema=EVENT_SCHEMA)
    effective_mode = mode if eligible_run or mode == "recovery" else "recovery"
    strategy_exit_updates, benchmark_updates, benchmark_trade_updates, exit_events, benchmark_exit_events = update_exits(
        trades=trades, benchmark_decisions=benchmark_decisions, benchmark_trades=benchmark_trades,
        bars=bars, observed_cutoff=observed_cutoff, recorded_at=actual_start, snapshot=snapshot, lock=lock,
    )
    trades = merge_state_rows(trades, strategy_exit_updates, schema=TRADE_SCHEMA, id_field="trade_id", immutable_fields=("trade_id", "decision_id", "decision_origin", "symbol", "target_weight", "entry_ts", "entry_price", "scheduled_exit_ts", "entry_cost_bps", "exit_cost_bps", "entry_market_data_snapshot_id", "eligible_for_forward_evidence", "parameter_lock_hash", "strategy_code_hash", "git_commit"), terminal_statuses=frozenset({"CLOSED", "MISSING_ENTRY", "MISSING_EXIT"}), sort_fields=("entry_ts", "symbol"))
    benchmark_decisions = merge_state_rows(benchmark_decisions, benchmark_updates, schema=BENCHMARK_DECISION_SCHEMA, id_field="benchmark_id", immutable_fields=("benchmark_id", "benchmark_type", "decision_id", "decision_origin", "decision_ts", "entry_ts", "scheduled_exit_ts", "expected_symbols", "expected_symbol_count", "expected_weight_per_symbol", "filled_symbol_count", "missing_symbol_count", "fill_coverage", "invested_weight", "cash_residual", "symbols", "weights", "entry_market_data_snapshot_id", "market_data_cutoff", "code_commit", "parameter_lock_hash", "strategy_code_hash"), terminal_statuses=frozenset({"CLOSED", "INCOMPLETE_ENTRY", "INCOMPLETE_EXIT"}), sort_fields=("decision_ts", "benchmark_type"))
    benchmark_trades = merge_state_rows(benchmark_trades, benchmark_trade_updates, schema=BENCHMARK_TRADE_SCHEMA, id_field="benchmark_trade_id", immutable_fields=("benchmark_trade_id", "benchmark_id", "benchmark_type", "decision_id", "decision_origin", "symbol", "expected_weight", "entry_price_available", "entry_ts", "entry_price", "fill_status", "missing_reason", "realized_weight", "scheduled_exit_ts", "entry_cost_bps", "exit_cost_bps", "entry_market_data_snapshot_id", "eligible_for_forward_evidence"), terminal_statuses=frozenset({"CLOSED", "MISSING_ENTRY", "MISSING_EXIT"}), sort_fields=("entry_ts", "benchmark_type", "symbol"))
    events = append_hash_chain(events, exit_events, schema=EVENT_SCHEMA, id_field="event_id")
    benchmark_events = append_hash_chain(benchmark_events, benchmark_exit_events, schema=EVENT_SCHEMA, id_field="event_id")
    entry_updates, new_benchmarks, new_benchmark_trades, _entry_coverage, entry_events, benchmark_entry_events = create_entries(
        decisions=decisions, trades=trades, benchmark_decisions=benchmark_decisions, bars=bars,
        observed_cutoff=observed_cutoff, recorded_at=actual_start, snapshot=snapshot, lock=lock,
    )
    trades = merge_state_rows(trades, entry_updates, schema=TRADE_SCHEMA, id_field="trade_id", immutable_fields=("trade_id", "decision_id", "decision_origin", "symbol", "target_weight", "entry_ts", "entry_price", "scheduled_exit_ts", "entry_cost_bps", "exit_cost_bps", "entry_market_data_snapshot_id", "eligible_for_forward_evidence", "parameter_lock_hash", "strategy_code_hash", "git_commit"), terminal_statuses=frozenset({"CLOSED", "MISSING_ENTRY", "MISSING_EXIT"}), sort_fields=("entry_ts", "symbol"))
    benchmark_decisions = merge_state_rows(benchmark_decisions, new_benchmarks, schema=BENCHMARK_DECISION_SCHEMA, id_field="benchmark_id", immutable_fields=tuple(BENCHMARK_DECISION_SCHEMA), terminal_statuses=frozenset({"CLOSED", "INCOMPLETE_ENTRY", "INCOMPLETE_EXIT"}), sort_fields=("decision_ts", "benchmark_type"))
    benchmark_trades = merge_state_rows(benchmark_trades, new_benchmark_trades, schema=BENCHMARK_TRADE_SCHEMA, id_field="benchmark_trade_id", immutable_fields=tuple(BENCHMARK_TRADE_SCHEMA), terminal_statuses=frozenset({"CLOSED", "MISSING_ENTRY", "MISSING_EXIT"}), sort_fields=("entry_ts", "benchmark_type", "symbol"))
    events = append_hash_chain(events, entry_events, schema=EVENT_SCHEMA, id_field="event_id")
    benchmark_events = append_hash_chain(benchmark_events, benchmark_entry_events, schema=EVENT_SCHEMA, id_field="event_id")
    new_decisions, decision_events = create_decisions(
        effective_mode=effective_mode, run_schedule_id=run_schedule_id, lock=lock, cutoff=cutoff,
        observed_cutoff=observed_cutoff, recorded_at=actual_start, bars=bars, existing=decisions,
        snapshot=snapshot, root=root,
    )
    decisions = merge_state_rows(decisions, new_decisions, schema=DECISION_SCHEMA, id_field="decision_id", immutable_fields=tuple(DECISION_SCHEMA), terminal_statuses=frozenset({"COMPLETE", "CYCLE_INCOMPLETE"}), sort_fields=("scheduled_run_ts",))
    events = append_hash_chain(events, decision_events, schema=EVENT_SCHEMA, id_field="event_id")
    decisions, trades, benchmark_decisions, completed_ids, incomplete_benchmarks = finalize_cycles(
        decisions=decisions, trades=trades, benchmarks=benchmark_decisions, observed_cutoff=observed_cutoff, lock=lock,
    )
    coverage = coverage_frame(benchmark_decisions, benchmark_trades)
    finish = datetime.now(UTC).replace(microsecond=0)
    schedule_events = append_hash_chain(
        schedule_events,
        frame([make_schedule_event(event_type="RUN_COMPLETED", expected_run_ts=slot, actual_start_ts=actual_start, actual_finish_ts=finish, run_mode="REALTIME" if mode == "realtime" else "RECOVERY", run_status="COMPLETED", exit_code=0, eligible=eligible_run and mode == "realtime", market_data_cutoff=observed_cutoff, error_code="", lock=lock)], SCHEDULE_EVENT_SCHEMA),
        schema=SCHEDULE_EVENT_SCHEMA,
        id_field="schedule_event_id",
    )
    performance_frame, status, equity, benchmark_equity = performance(
        decisions=decisions, trades=trades, benchmarks=benchmark_decisions, coverage=coverage,
        schedule_events=schedule_events, cutoff=cutoff, observed_cutoff=observed_cutoff, snapshot=snapshot,
        lock=lock, identity=identity, deployment=deployment, health=health,
        incomplete_benchmark_cycles=incomplete_benchmarks,
    )
    if dry_run:
        return {**status, "dry_run": True, "evidence_mutated": False}
    evidence_changed = any((not new_decisions.is_empty(), not entry_updates.is_empty(), not strategy_exit_updates.is_empty(), not new_benchmarks.is_empty(), not benchmark_updates.is_empty()))
    if evidence_changed:
        persisted = _persist_snapshot(root, snapshot, snapshot_bytes)
        _update_snapshot_registry(root, snapshot, persisted)
    atomic_write_parquet(decisions, paths["decisions"])
    atomic_write_parquet(trades, paths["trades"])
    atomic_write_parquet(events, paths["events"])
    atomic_write_parquet(equity, paths["equity"])
    atomic_write_parquet(benchmark_decisions, paths["benchmark_decisions"])
    atomic_write_parquet(benchmark_trades, paths["benchmark_trades"])
    atomic_write_parquet(benchmark_events, paths["benchmark_events"])
    atomic_write_parquet(benchmark_equity, paths["benchmark_equity"])
    atomic_write_parquet(coverage, paths["benchmark_coverage"])
    atomic_write_parquet(schedule_events, paths["schedule_events"])
    atomic_write_csv(performance_frame, paths["performance"])
    atomic_write_json(status, paths["status"])
    return status


def _record_failure(
    *, root: Path, repo: Path, mode: str, error: Exception, dry_run: bool
) -> dict[str, Any]:
    paths = _paths(root)
    prior = _load(paths["status"]) if paths["status"].exists() else {}
    try:
        lock = _load(root / "manifests/parameter_lock_v221.json")
        schedule_events = _read(paths["schedule_events"], SCHEDULE_EVENT_SCHEMA)
        now = datetime.now(UTC).replace(microsecond=0)
        slot, _persistent = timer_slot(now)
        failed_event = make_schedule_event(
            event_type="RUN_FAILED",
            expected_run_ts=slot,
            actual_start_ts=now,
            actual_finish_ts=now,
            run_mode="REALTIME" if mode == "realtime" else "RECOVERY",
            run_status="FAILED",
            exit_code=2,
            eligible=False,
            market_data_cutoff=None,
            error_code=type(error).__name__,
            lock=lock,
        )
        schedule_events = append_hash_chain(
            schedule_events,
            frame([failed_event], SCHEDULE_EVENT_SCHEMA),
            schema=SCHEDULE_EVENT_SCHEMA,
            id_field="schedule_event_id",
        )
        if not dry_run:
            atomic_write_parquet(schedule_events, paths["schedule_events"])
    except Exception:
        pass
    if mode == "recovery":
        prior.update(
            {
                "recovery_audit_warning": True,
                "recovery_runner_errors": int(prior.get("recovery_runner_errors", 0)) + 1,
                "last_recovery_error": type(error).__name__,
                "evidence_mutated": False,
            }
        )
    else:
        paper = "FAIL_RUNNER_INTEGRITY" if isinstance(error, RunnerIntegrityError) else "INCONCLUSIVE_DATA_INCOMPLETE" if isinstance(error, DataIncompleteError) else "INCONCLUSIVE_SYSTEM_NOT_DEPLOYED"
        prior.update(
            {
                "paper_status": paper,
                "formal_realtime_runner_errors": int(prior.get("formal_realtime_runner_errors", 0)) + 1,
                "failure_code": type(error).__name__,
                "failure_fingerprint": payload_digest({"type": type(error).__name__, "message": str(error)}),
                "evidence_mutated": False,
            }
        )
    if not dry_run:
        atomic_write_json(prior, paths["status"])
    return prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("realtime", "recovery"), required=True)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("AUDIT_V221_ROOT", "/var/lib/quant-lab/forward_v221")))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--market-bar", type=Path, default=Path(os.environ.get("QUANT_LAB_FORWARD_MARKET_BAR_PATH", "/var/lib/quant-lab/lake/silver/market_bar/data.parquet")))
    parser.add_argument("--as-of")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allow-no-cutoff", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    failed = False
    try:
        if args.report_only:
            status = _load(root / "artifacts/forward_v221_status.json")
        else:
            status = run(
                root=root,
                repo=args.repo.resolve(),
                market_bar=args.market_bar.resolve(),
                mode=args.mode,
                requested_as_of=utc(args.as_of or datetime.now(UTC).replace(microsecond=0)),
                recorded_at=None,
                resume=args.resume,
                dry_run=args.dry_run,
                allow_no_cutoff=args.allow_no_cutoff,
            )
    except Exception as exc:
        failed = True
        status = _record_failure(root=root, repo=args.repo.resolve(), mode=args.mode, error=exc, dry_run=args.dry_run)
    print(f"forward_v221_status={status.get('paper_status', 'UNKNOWN')}")
    print(f"formal_realtime_decision_count={status.get('formal_realtime_decision_count', 0)}")
    print(f"recovery_decision_count={status.get('recovery_decision_count', 0)}")
    print(f"schedule_coverage={status.get('schedule_coverage', 0.0)}")
    print(f"completed_independent_cycles={status.get('completed_independent_cycles', 0)}")
    if failed or status.get("paper_status") == "FAIL_RUNNER_INTEGRITY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
