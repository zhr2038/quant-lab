# ruff: noqa: E501
"""Audit v2.2.1 immutable Forward Paper primitives.

The module keeps the locked v2.2 strategy formula and 120-hour anchor while
adding three controls required for credible forward evidence:

* missing dynamic-universe benchmark weights remain explicit cash;
* recovery observations are separately accounted and cannot drive paper state;
* hourly runner schedules are append-only evidence, independent from strategy
  rebalance schedules.

Nothing in this module can submit an exchange order or enable production Alpha.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

AUDIT_VERSION = "v2.2.1"
RUNNER_VERSION = "quant_lab_low_vol_forward.v2.2.1"
STRATEGY_ID = "low_vol_btc_60d_trend_top3_score_120h_v221"
STRATEGY_VERSION = "2.2.1"
STRATEGY_SCHEDULE_ANCHOR = datetime(2026, 7, 18, 22, 0, tzinfo=UTC)
SCHEDULE_TIMEZONE = "UTC"
TIMER_MINUTE = 10

HOLDING_HOURS = 120
BTC_TREND_LOOKBACK_HOURS = 1440
FACTOR_LOOKBACK_HOURS = 480
MAX_DECISION_LATENCY_SECONDS = 3600
ENTRY_COST_BPS = 15.0
EXIT_COST_BPS = 15.0
ROUND_TRIP_COST_BPS = 30.0
BAR_CLOSE_DELAY_HOURS = 1
MARKET_LOOKBACK_DAYS = 70

BENCHMARK_TYPES = (
    "BTC_BUY_AND_HOLD",
    "DYNAMIC_UNIVERSE_EQUAL_WEIGHT",
    "CASH",
)
DECISION_ORIGINS = frozenset(
    {"REALTIME", "RECOVERY_RECONSTRUCTION", "MANUAL_REPLAY"}
)
SCHEDULE_EVENT_TYPES = frozenset(
    {
        "SCHEDULE_EXPECTED",
        "RUN_STARTED",
        "RUN_COMPLETED",
        "RUN_FAILED",
        "RUN_MISSED",
        "RUN_LATE",
    }
)
PAPER_STATUSES = frozenset(
    {
        "FAIL_RUNNER_INTEGRITY",
        "FAIL_DATA_QUALITY",
        "FAIL_PERFORMANCE",
        "FAIL_BENCHMARK_UNDERPERFORMANCE",
        "FAIL_RISK_OR_DRAWDOWN",
        "FAIL_CONCENTRATION",
        "INCONCLUSIVE_SYSTEM_NOT_DEPLOYED",
        "INCONCLUSIVE_SAMPLE_INSUFFICIENT",
        "INCONCLUSIVE_DATA_INCOMPLETE",
        "CONTINUE_PAPER",
        "PAPER_REVIEW_READY",
    }
)

DECISION_SCHEMA = {
    "decision_id": pl.Utf8,
    "schedule_id": pl.Utf8,
    "strategy_id": pl.Utf8,
    "strategy_version": pl.Utf8,
    "decision_ts": pl.Datetime("us", "UTC"),
    "scheduled_run_ts": pl.Datetime("us", "UTC"),
    "recorded_at": pl.Datetime("us", "UTC"),
    "decision_latency_seconds": pl.Float64,
    "decision_origin": pl.Utf8,
    "late_reconstructed": pl.Boolean,
    "eligible_for_forward_evidence": pl.Boolean,
    "cycle_eligible_for_forward_evidence": pl.Boolean,
    "btc_benchmark_cycle_complete": pl.Boolean,
    "universe_benchmark_cycle_complete": pl.Boolean,
    "cash_benchmark_cycle_complete": pl.Boolean,
    "all_benchmarks_complete": pl.Boolean,
    "feature_cutoff_ts": pl.Datetime("us", "UTC"),
    "observed_market_data_cutoff": pl.Datetime("us", "UTC"),
    "market_data_cutoff": pl.Datetime("us", "UTC"),
    "market_data_snapshot_id": pl.Utf8,
    "input_file_paths": pl.Utf8,
    "input_file_sha256": pl.Utf8,
    "feature_data_coverage": pl.Float64,
    "data_quality_status": pl.Utf8,
    "btc_trend_state": pl.Utf8,
    "universe": pl.Utf8,
    "factor_scores": pl.Utf8,
    "ranked_symbols": pl.Utf8,
    "selected_symbols": pl.Utf8,
    "target_weights": pl.Utf8,
    "cash_weight": pl.Float64,
    "parameter_lock_hash": pl.Utf8,
    "strategy_code_hash": pl.Utf8,
    "git_commit": pl.Utf8,
    "working_tree_clean": pl.Boolean,
    "status": pl.Utf8,
}

TRADE_SCHEMA = {
    "trade_id": pl.Utf8,
    "decision_id": pl.Utf8,
    "decision_origin": pl.Utf8,
    "symbol": pl.Utf8,
    "target_weight": pl.Float64,
    "entry_ts": pl.Datetime("us", "UTC"),
    "entry_price": pl.Float64,
    "scheduled_exit_ts": pl.Datetime("us", "UTC"),
    "actual_exit_ts": pl.Datetime("us", "UTC"),
    "exit_price": pl.Float64,
    "entry_cost_bps": pl.Float64,
    "exit_cost_bps": pl.Float64,
    "gross_return": pl.Float64,
    "net_return": pl.Float64,
    "entry_market_data_snapshot_id": pl.Utf8,
    "exit_market_data_snapshot_id": pl.Utf8,
    "eligible_for_forward_evidence": pl.Boolean,
    "cycle_eligible_for_forward_evidence": pl.Boolean,
    "parameter_lock_hash": pl.Utf8,
    "strategy_code_hash": pl.Utf8,
    "git_commit": pl.Utf8,
    "status": pl.Utf8,
}

EVENT_SCHEMA = {
    "event_id": pl.Utf8,
    "event_type": pl.Utf8,
    "event_ts": pl.Datetime("us", "UTC"),
    "recorded_at": pl.Datetime("us", "UTC"),
    "decision_id": pl.Utf8,
    "trade_id": pl.Utf8,
    "payload_hash": pl.Utf8,
    "payload_json": pl.Utf8,
    "parameter_lock_hash": pl.Utf8,
    "strategy_code_hash": pl.Utf8,
    "git_commit": pl.Utf8,
    "previous_event_hash": pl.Utf8,
    "event_hash": pl.Utf8,
}

BENCHMARK_DECISION_SCHEMA = {
    "benchmark_id": pl.Utf8,
    "benchmark_type": pl.Utf8,
    "decision_id": pl.Utf8,
    "decision_origin": pl.Utf8,
    "decision_ts": pl.Datetime("us", "UTC"),
    "entry_ts": pl.Datetime("us", "UTC"),
    "scheduled_exit_ts": pl.Datetime("us", "UTC"),
    "actual_exit_ts": pl.Datetime("us", "UTC"),
    "expected_symbols": pl.Utf8,
    "expected_symbol_count": pl.Int64,
    "expected_weight_per_symbol": pl.Float64,
    "filled_symbol_count": pl.Int64,
    "missing_symbol_count": pl.Int64,
    "fill_coverage": pl.Float64,
    "invested_weight": pl.Float64,
    "cash_residual": pl.Float64,
    "symbols": pl.Utf8,
    "weights": pl.Utf8,
    "gross_return": pl.Float64,
    "net_return": pl.Float64,
    "entry_market_data_snapshot_id": pl.Utf8,
    "exit_market_data_snapshot_id": pl.Utf8,
    "market_data_cutoff": pl.Datetime("us", "UTC"),
    "code_commit": pl.Utf8,
    "parameter_lock_hash": pl.Utf8,
    "strategy_code_hash": pl.Utf8,
    "eligible_for_forward_evidence": pl.Boolean,
    "cycle_complete": pl.Boolean,
    "status": pl.Utf8,
}

BENCHMARK_TRADE_SCHEMA = {
    "benchmark_trade_id": pl.Utf8,
    "benchmark_id": pl.Utf8,
    "benchmark_type": pl.Utf8,
    "decision_id": pl.Utf8,
    "decision_origin": pl.Utf8,
    "symbol": pl.Utf8,
    "expected_weight": pl.Float64,
    "entry_price_available": pl.Boolean,
    "entry_ts": pl.Datetime("us", "UTC"),
    "entry_price": pl.Float64,
    "fill_status": pl.Utf8,
    "missing_reason": pl.Utf8,
    "realized_weight": pl.Float64,
    "scheduled_exit_ts": pl.Datetime("us", "UTC"),
    "actual_exit_ts": pl.Datetime("us", "UTC"),
    "exit_price": pl.Float64,
    "gross_return": pl.Float64,
    "net_return": pl.Float64,
    "entry_cost_bps": pl.Float64,
    "exit_cost_bps": pl.Float64,
    "entry_market_data_snapshot_id": pl.Utf8,
    "exit_market_data_snapshot_id": pl.Utf8,
    "eligible_for_forward_evidence": pl.Boolean,
    "status": pl.Utf8,
}

BENCHMARK_COVERAGE_SCHEMA = {
    "benchmark_id": pl.Utf8,
    "decision_id": pl.Utf8,
    "benchmark_type": pl.Utf8,
    "expected_symbol_count": pl.Int64,
    "filled_symbol_count": pl.Int64,
    "missing_symbol_count": pl.Int64,
    "fill_coverage": pl.Float64,
    "invested_weight": pl.Float64,
    "cash_residual": pl.Float64,
    "cycle_complete": pl.Boolean,
    "eligible_for_forward_evidence": pl.Boolean,
    "missing_symbols": pl.Utf8,
    "entry_details": pl.Utf8,
    "status": pl.Utf8,
}

SCHEDULE_EVENT_SCHEMA = {
    "schedule_event_id": pl.Utf8,
    "event_type": pl.Utf8,
    "schedule_id": pl.Utf8,
    "strategy_id": pl.Utf8,
    "expected_run_ts": pl.Datetime("us", "UTC"),
    "scheduled_run_ts": pl.Datetime("us", "UTC"),
    "actual_start_ts": pl.Datetime("us", "UTC"),
    "actual_finish_ts": pl.Datetime("us", "UTC"),
    "recorded_at": pl.Datetime("us", "UTC"),
    "latency_seconds": pl.Float64,
    "run_mode": pl.Utf8,
    "run_status": pl.Utf8,
    "exit_code": pl.Int64,
    "runner_version": pl.Utf8,
    "git_commit": pl.Utf8,
    "strategy_code_hash": pl.Utf8,
    "parameter_lock_hash": pl.Utf8,
    "service_unit_hash": pl.Utf8,
    "timer_unit_hash": pl.Utf8,
    "market_data_cutoff": pl.Datetime("us", "UTC"),
    "eligible_for_forward_evidence": pl.Boolean,
    "error_code": pl.Utf8,
    "previous_event_hash": pl.Utf8,
    "event_hash": pl.Utf8,
}

RUNTIME_HEALTH_SCHEMA = {
    "checked_at": pl.Datetime("us", "UTC"),
    "research_node": pl.Utf8,
    "timer_installed": pl.Boolean,
    "timer_enabled": pl.Boolean,
    "timer_active": pl.Boolean,
    "service_last_result": pl.Utf8,
    "next_trigger": pl.Utf8,
    "last_successful_run": pl.Datetime("us", "UTC"),
    "last_failed_run": pl.Datetime("us", "UTC"),
    "working_tree_clean": pl.Boolean,
    "git_head_match": pl.Boolean,
    "code_hash_match": pl.Boolean,
    "parameter_lock_match": pl.Boolean,
    "unit_hash_match": pl.Boolean,
    "disk_free_bytes": pl.Int64,
    "market_data_staleness_seconds": pl.Float64,
    "health_status": pl.Utf8,
}

EQUITY_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "strategy_equity": pl.Float64,
    "completed_cycle_count": pl.Int64,
}

BENCHMARK_EQUITY_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "btc_equity": pl.Float64,
    "dynamic_universe_equity": pl.Float64,
    "cash_equity": pl.Float64,
    "completed_cycle_count": pl.Int64,
}


def utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_digest(value: Any) -> str:
    return sha256_text(canonical_json(value))


def empty_frame(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict(schema))


def frame(rows: Sequence[Mapping[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=dict(schema)) if rows else empty_frame(schema)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def atomic_write_parquet(data: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    data.write_parquet(partial, compression="zstd")
    os.replace(partial, path)


def atomic_write_csv(data: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    data.write_csv(partial)
    os.replace(partial, path)


def source_hash_entries(repo: Path, paths: Iterable[str]) -> list[dict[str, str]]:
    repo = repo.resolve()
    result: list[dict[str, str]] = []
    for raw in sorted(set(paths)):
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"locked source must be repository-relative: {raw}")
        path = (repo / relative).resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"locked source escapes repository: {raw}") from exc
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        result.append({"path": relative.as_posix(), "sha256": sha256_file(path)})
    return result


def composite_source_hash(entries: Sequence[Mapping[str, str]]) -> str:
    normalized = [
        {"path": str(item["path"]), "sha256": str(item["sha256"])}
        for item in sorted(entries, key=lambda row: str(row["path"]))
    ]
    return payload_digest(normalized)


def parameter_lock_digest(lock: Mapping[str, Any]) -> str:
    return payload_digest({key: value for key, value in lock.items() if key != "sha256"})


def deployment_manifest_digest(manifest: Mapping[str, Any]) -> str:
    return payload_digest(
        {key: value for key, value in manifest.items() if key != "deployment_manifest_sha256"}
    )


def validate_parameter_lock(lock: Mapping[str, Any]) -> None:
    expected = {
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "hypothesis_type": "POST_HOC_HYPOTHESIS",
        "approval_state": "PAPER_ONLY",
        "parameters_locked": True,
        "factor": "low_vol_20d",
        "factor_formula": "-rolling_std(ret_1h, 480)",
        "factor_lookback_hours": FACTOR_LOOKBACK_HOURS,
        "btc_trend_filter": True,
        "btc_trend_lookback_hours": BTC_TREND_LOOKBACK_HOURS,
        "btc_trend_description": "BTC 60-day SMA trend filter",
        "top_n": 3,
        "weighting": "score",
        "rebalance_hours": HOLDING_HOURS,
        "holding_hours": HOLDING_HOURS,
        "decision_delay_bars": 1,
        "strategy_type": "long_only_spot",
        "entry_cost_bps": ENTRY_COST_BPS,
        "exit_cost_bps": EXIT_COST_BPS,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "max_decision_latency_seconds": MAX_DECISION_LATENCY_SECONDS,
        "minimum_forward_days": 30,
        "minimum_completed_cycles": 6,
        "minimum_symbol_trades": 12,
        "minimum_data_coverage": 0.95,
        "minimum_benchmark_fill_coverage": 0.95,
        "minimum_schedule_coverage": 0.95,
        "maximum_single_symbol_contribution": 0.50,
        "maximum_drawdown": 0.30,
        "schedule_anchor": STRATEGY_SCHEDULE_ANCHOR.isoformat(),
        "schedule_timezone": SCHEDULE_TIMEZONE,
        "runner_version": RUNNER_VERSION,
        "execution_mode": "PAPER",
        "production_alpha": "FROZEN",
        "live_opening_enabled": False,
        "live_order_effect": "none",
        "automatic_promotion": False,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": lock.get(key)}
        for key, expected_value in expected.items()
        if lock.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"parameter lock violates v2.2.1 strategy: {mismatches}")
    if len(str(lock.get("strategy_code_commit", ""))) != 40:
        raise ValueError("strategy_code_commit must be a full 40-character SHA")
    locked = lock.get("locked_source_files")
    if not isinstance(locked, list) or not locked:
        raise ValueError("locked_source_files must be non-empty")
    if composite_source_hash(locked) != lock.get("strategy_code_hash"):
        raise ValueError("strategy_code_hash mismatch")
    for key in ("service_unit_hash", "timer_unit_hash", "runner_script_hash"):
        value = str(lock.get(key, ""))
        if len(value) != 64:
            raise ValueError(f"{key} must be SHA256")
    if lock.get("sha256") != parameter_lock_digest(lock):
        raise ValueError("parameter lock sha256 mismatch")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def runtime_identity(
    *,
    repo: Path,
    lock: Mapping[str, Any],
    installed_service: Path | None = None,
    installed_timer: Path | None = None,
    deployment_manifest: Mapping[str, Any] | None = None,
    cutoff_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed identity check; it never mutates evidence."""
    errors: list[str] = []
    try:
        validate_parameter_lock(lock)
    except Exception as exc:
        errors.append(f"PARAMETER_LOCK_INVALID:{exc}")
    try:
        head = _git(repo, "rev-parse", "HEAD")
        dirty = _git(repo, "status", "--porcelain")
    except Exception as exc:
        head, dirty = "", "unknown"
        errors.append(f"GIT_STATE_UNAVAILABLE:{exc}")
    if head and head != str(lock.get("strategy_code_commit", "")):
        errors.append("GIT_HEAD_MISMATCH")
    if dirty:
        errors.append("WORKING_TREE_DIRTY")
    current_entries: list[dict[str, str]] = []
    try:
        current_entries = source_hash_entries(
            repo, [str(item["path"]) for item in lock.get("locked_source_files", [])]
        )
        expected_map = {
            str(item["path"]): str(item["sha256"])
            for item in lock.get("locked_source_files", [])
        }
        if {item["path"]: item["sha256"] for item in current_entries} != expected_map:
            errors.append("LOCKED_SOURCE_HASH_MISMATCH")
        if composite_source_hash(current_entries) != lock.get("strategy_code_hash"):
            errors.append("STRATEGY_CODE_HASH_MISMATCH")
    except Exception as exc:
        errors.append(f"LOCKED_SOURCE_UNAVAILABLE:{exc}")
    runner = repo / "scripts/run_forward_v221_realtime.sh"
    if not runner.is_file() or sha256_file(runner) != lock.get("runner_script_hash"):
        errors.append("RUNNER_SCRIPT_HASH_MISMATCH")
    for label, path, key in (
        ("SERVICE", installed_service, "service_unit_hash"),
        ("TIMER", installed_timer, "timer_unit_hash"),
    ):
        if path is not None and (
            not path.is_file() or sha256_file(path) != str(lock.get(key, ""))
        ):
            errors.append(f"{label}_UNIT_HASH_MISMATCH")
    deployment_hash = ""
    if deployment_manifest is not None:
        deployment_hash = deployment_manifest_digest(deployment_manifest)
        if deployment_hash != deployment_manifest.get("deployment_manifest_sha256"):
            errors.append("DEPLOYMENT_MANIFEST_HASH_MISMATCH")
        if deployment_manifest.get("parameter_lock_hash") != lock.get("sha256"):
            errors.append("DEPLOYMENT_PARAMETER_LOCK_MISMATCH")
        if deployment_manifest.get("service_unit_sha256") != lock.get("service_unit_hash"):
            errors.append("DEPLOYMENT_SERVICE_HASH_MISMATCH")
        if deployment_manifest.get("timer_unit_sha256") != lock.get("timer_unit_hash"):
            errors.append("DEPLOYMENT_TIMER_HASH_MISMATCH")
    if cutoff_manifest is not None:
        if cutoff_manifest.get("parameter_lock_hash") != lock.get("sha256"):
            errors.append("CUTOFF_PARAMETER_LOCK_MISMATCH")
        if cutoff_manifest.get("deployment_manifest_sha256") != deployment_hash:
            errors.append("CUTOFF_DEPLOYMENT_MANIFEST_MISMATCH")
        if cutoff_manifest.get("schedule_anchor") != lock.get("schedule_anchor"):
            errors.append("CUTOFF_SCHEDULE_ANCHOR_MISMATCH")
    current_hash = composite_source_hash(current_entries) if current_entries else ""
    return {
        "ok": not errors,
        "status": "PASS" if not errors else "FAIL_RUNNER_INTEGRITY",
        "errors": errors,
        "current_head": head,
        "expected_head": str(lock.get("strategy_code_commit", "")),
        "working_tree_clean": not bool(dirty),
        "strategy_code_hash": current_hash,
        "expected_strategy_code_hash": str(lock.get("strategy_code_hash", "")),
        "service_unit_hash_match": "SERVICE_UNIT_HASH_MISMATCH" not in errors,
        "timer_unit_hash_match": "TIMER_UNIT_HASH_MISMATCH" not in errors,
        "runner_script_hash_match": "RUNNER_SCRIPT_HASH_MISMATCH" not in errors,
        "deployment_manifest_sha256": deployment_hash,
    }


def next_strategy_schedule(after: datetime) -> datetime:
    value = utc(after)
    if value < STRATEGY_SCHEDULE_ANCHOR:
        return STRATEGY_SCHEDULE_ANCHOR
    elapsed = (value - STRATEGY_SCHEDULE_ANCHOR).total_seconds()
    index = int(elapsed // (HOLDING_HOURS * 3600)) + 1
    return STRATEGY_SCHEDULE_ANCHOR + timedelta(hours=index * HOLDING_HOURS)


def due_strategy_schedules(after: datetime, through: datetime) -> list[datetime]:
    start = next_strategy_schedule(after)
    end = utc(through)
    if start > end:
        return []
    count = int((end - start).total_seconds() // (HOLDING_HOURS * 3600)) + 1
    return [start + timedelta(hours=index * HOLDING_HOURS) for index in range(count)]


def timer_slot(actual_start: datetime) -> tuple[datetime, bool]:
    """Return the nominal hourly :10 slot and whether it is a persistent catch-up."""
    value = utc(actual_start).replace(second=0, microsecond=0)
    current = value.replace(minute=TIMER_MINUTE)
    persistent_catchup = value.minute < TIMER_MINUTE
    if persistent_catchup:
        current -= timedelta(hours=1)
    return current, persistent_catchup


def expected_timer_slots(cutoff: datetime, through: datetime) -> list[datetime]:
    start = utc(cutoff).replace(second=0, microsecond=0)
    candidate = start.replace(minute=TIMER_MINUTE)
    if candidate <= start:
        candidate += timedelta(hours=1)
    end = utc(through)
    if candidate > end:
        return []
    count = int((end - candidate).total_seconds() // 3600) + 1
    return [candidate + timedelta(hours=index) for index in range(count)]


def classify_decision_origin(
    *, mode: str, scheduled_run_ts: datetime, recorded_at: datetime, max_latency: int
) -> tuple[str, bool, bool, float]:
    scheduled = utc(scheduled_run_ts)
    recorded = utc(recorded_at)
    latency = max(0.0, (recorded - scheduled).total_seconds())
    if mode == "recovery":
        return "RECOVERY_RECONSTRUCTION", True, False, latency
    if mode != "realtime":
        raise ValueError(f"unsupported mode: {mode}")
    late = latency > float(max_latency)
    return ("RECOVERY_RECONSTRUCTION" if late else "REALTIME", late, not late, latency)


def stable_decision_id(
    scheduled_run_ts: datetime, snapshot_id: str, parameter_lock_hash: str
) -> str:
    return sha256_text(
        "|".join(
            (
                STRATEGY_ID,
                utc(scheduled_run_ts).isoformat(),
                snapshot_id,
                parameter_lock_hash,
            )
        )
    )


def stable_trade_id(decision_id: str, symbol: str, entry_ts: datetime) -> str:
    return sha256_text("|".join((decision_id, symbol, utc(entry_ts).isoformat())))


def stable_benchmark_id(decision_id: str, benchmark_type: str) -> str:
    return sha256_text("|".join((decision_id, benchmark_type)))


def cost_adjusted_return(gross_return: float, entry_bps: float, exit_bps: float) -> float:
    return (
        (1.0 - float(entry_bps) / 10_000.0)
        * (1.0 + float(gross_return))
        * (1.0 - float(exit_bps) / 10_000.0)
        - 1.0
    )


def compounded_return(values: Iterable[float]) -> float:
    result = 1.0
    for value in values:
        result *= 1.0 + float(value)
    return result - 1.0


def benchmark_entry_records(
    *,
    decision: Mapping[str, Any],
    benchmark_type: str,
    expected_symbols: Sequence[str],
    entry_prices: Mapping[str, tuple[datetime, float] | None],
    entry_due: datetime,
    lock: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Lock expected benchmark membership and retain every missing weight as cash."""
    expected = [str(symbol) for symbol in expected_symbols]
    if not expected:
        raise ValueError("benchmark expected_symbols cannot be empty")
    expected_weight = 1.0 / len(expected)
    benchmark_id = stable_benchmark_id(str(decision["decision_id"]), benchmark_type)
    trade_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    realized_weights: dict[str, float] = {}
    filled = 0
    for symbol in expected:
        value = entry_prices.get(symbol)
        available = value is not None
        if available:
            assert value is not None
            entry_ts, entry_price = utc(value[0]), float(value[1])
            fill_status, missing_reason, realized_weight = "FILLED", "", expected_weight
            filled += 1
            realized_weights[symbol] = realized_weight
        else:
            entry_ts, entry_price = utc(entry_due), None
            fill_status, missing_reason, realized_weight = (
                "MISSING",
                "ENTRY_PRICE_UNAVAILABLE",
                0.0,
            )
        entry_cost = 0.0 if symbol == "CASH" else ENTRY_COST_BPS
        exit_cost = 0.0 if symbol == "CASH" else EXIT_COST_BPS
        trade_rows.append(
            {
                "benchmark_trade_id": payload_digest(
                    [benchmark_id, symbol, utc(entry_due).isoformat()]
                ),
                "benchmark_id": benchmark_id,
                "benchmark_type": benchmark_type,
                "decision_id": str(decision["decision_id"]),
                "decision_origin": str(decision["decision_origin"]),
                "symbol": symbol,
                "expected_weight": expected_weight,
                "entry_price_available": available,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "fill_status": fill_status,
                "missing_reason": missing_reason,
                "realized_weight": realized_weight,
                "scheduled_exit_ts": utc(entry_due) + timedelta(hours=HOLDING_HOURS),
                "actual_exit_ts": None,
                "exit_price": None,
                "gross_return": None,
                "net_return": None,
                "entry_cost_bps": entry_cost,
                "exit_cost_bps": exit_cost,
                "entry_market_data_snapshot_id": str(decision["market_data_snapshot_id"]),
                "exit_market_data_snapshot_id": "",
                "eligible_for_forward_evidence": bool(
                    decision["eligible_for_forward_evidence"]
                ),
                "status": "OPEN" if available else "MISSING_ENTRY",
            }
        )
        details.append(
            {
                "symbol": symbol,
                "expected_weight": expected_weight,
                "entry_price_available": available,
                "entry_price": entry_price,
                "fill_status": fill_status,
                "missing_reason": missing_reason,
                "realized_weight": realized_weight,
            }
        )
    count = len(expected)
    missing = count - filled
    coverage = filled / count
    invested = sum(realized_weights.values())
    cash_residual = max(0.0, 1.0 - invested)
    if not math.isclose(invested + cash_residual, 1.0, abs_tol=1e-12):
        raise ValueError("benchmark invested weight and cash residual do not sum to one")
    required = 1.0 if benchmark_type in {"BTC_BUY_AND_HOLD", "CASH"} else float(
        lock["minimum_benchmark_fill_coverage"]
    )
    entry_complete = coverage + 1e-12 >= required
    benchmark = {
        "benchmark_id": benchmark_id,
        "benchmark_type": benchmark_type,
        "decision_id": str(decision["decision_id"]),
        "decision_origin": str(decision["decision_origin"]),
        "decision_ts": decision["decision_ts"],
        "entry_ts": utc(entry_due),
        "scheduled_exit_ts": utc(entry_due) + timedelta(hours=HOLDING_HOURS),
        "actual_exit_ts": None,
        "expected_symbols": canonical_json(expected),
        "expected_symbol_count": count,
        "expected_weight_per_symbol": expected_weight,
        "filled_symbol_count": filled,
        "missing_symbol_count": missing,
        "fill_coverage": coverage,
        "invested_weight": invested,
        "cash_residual": cash_residual,
        "symbols": canonical_json([row["symbol"] for row in details if row["entry_price_available"]]),
        "weights": canonical_json(realized_weights),
        "gross_return": None,
        "net_return": None,
        "entry_market_data_snapshot_id": str(decision["market_data_snapshot_id"]),
        "exit_market_data_snapshot_id": "",
        "market_data_cutoff": decision["market_data_cutoff"],
        "code_commit": str(lock["strategy_code_commit"]),
        "parameter_lock_hash": str(lock["sha256"]),
        "strategy_code_hash": str(lock["strategy_code_hash"]),
        "eligible_for_forward_evidence": bool(
            decision["eligible_for_forward_evidence"] and entry_complete
        ),
        "cycle_complete": False,
        "status": "OPEN" if entry_complete else "INCOMPLETE_ENTRY",
    }
    coverage_row = {
        "benchmark_id": benchmark_id,
        "decision_id": str(decision["decision_id"]),
        "benchmark_type": benchmark_type,
        "expected_symbol_count": count,
        "filled_symbol_count": filled,
        "missing_symbol_count": missing,
        "fill_coverage": coverage,
        "invested_weight": invested,
        "cash_residual": cash_residual,
        "cycle_complete": False,
        "eligible_for_forward_evidence": False,
        "missing_symbols": canonical_json(
            [row["symbol"] for row in details if not row["entry_price_available"]]
        ),
        "entry_details": canonical_json(details),
        "status": "OPEN" if entry_complete else "INCOMPLETE_ENTRY",
    }
    return benchmark, trade_rows, coverage_row


def _normal(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc(value).isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def merge_state_rows(
    existing: pl.DataFrame,
    updates: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    id_field: str,
    immutable_fields: Sequence[str],
    terminal_statuses: frozenset[str],
    sort_fields: Sequence[str],
) -> pl.DataFrame:
    rows = {str(row[id_field]): row for row in existing.iter_rows(named=True)}
    for incoming in updates.iter_rows(named=True):
        key = str(incoming[id_field])
        prior = rows.get(key)
        if prior is None:
            rows[key] = incoming
            continue
        changed = [
            field
            for field in immutable_fields
            if _normal(prior.get(field)) != _normal(incoming.get(field))
        ]
        if changed:
            raise ValueError(f"immutable {id_field} fields changed: {changed}")
        if str(prior.get("status")) in terminal_statuses:
            all_changed = [
                field
                for field in schema
                if _normal(prior.get(field)) != _normal(incoming.get(field))
            ]
            if all_changed:
                raise ValueError(f"terminal {id_field} changed: {all_changed}")
        else:
            rows[key] = incoming
    if not rows:
        return empty_frame(schema)
    return pl.DataFrame(list(rows.values()), schema=dict(schema)).sort(list(sort_fields))


def _event_core(row: Mapping[str, Any], schema: Mapping[str, pl.DataType]) -> dict[str, Any]:
    return {
        key: utc(value).isoformat() if isinstance(value, datetime) else value
        for key, value in row.items()
        if key in schema and key not in {"previous_event_hash", "event_hash"}
    }


def append_hash_chain(
    existing: pl.DataFrame,
    additions: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    id_field: str,
    previous_field: str = "previous_event_hash",
    hash_field: str = "event_hash",
) -> pl.DataFrame:
    rows = existing.to_dicts()
    previous = str(rows[-1][hash_field]) if rows else ""
    by_id = {str(row[id_field]): row for row in rows}
    additions = additions.sort("recorded_at") if not additions.is_empty() else additions
    for incoming in additions.iter_rows(named=True):
        key = str(incoming[id_field])
        if key in by_id:
            old_core = _event_core(by_id[key], schema)
            new_core = _event_core(incoming, schema)
            if new_core != old_core:
                raise ValueError("append-only event changed")
            continue
        row = dict(incoming)
        row[previous_field] = previous
        row[hash_field] = sha256_text(previous + canonical_json(_event_core(row, schema)))
        previous = row[hash_field]
        rows.append(row)
        by_id[key] = row
    result = pl.DataFrame(rows, schema=dict(schema)) if rows else empty_frame(schema)
    validate_hash_chain(
        result,
        schema=schema,
        previous_field=previous_field,
        hash_field=hash_field,
    )
    return result


def validate_hash_chain(
    events: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    previous_field: str = "previous_event_hash",
    hash_field: str = "event_hash",
) -> None:
    previous = ""
    for row in events.iter_rows(named=True):
        if str(row[previous_field]) != previous:
            raise ValueError("event previous hash mismatch")
        expected = sha256_text(previous + canonical_json(_event_core(row, schema)))
        if str(row[hash_field]) != expected:
            raise ValueError("event hash mismatch")
        previous = expected


def make_event(
    *,
    event_type: str,
    event_ts: datetime,
    recorded_at: datetime,
    decision_id: str,
    trade_id: str = "",
    payload: Mapping[str, Any] | None,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    payload_json = canonical_json(dict(payload or {}))
    payload_hash = sha256_text(payload_json)
    event_id = sha256_text(
        "|".join(
            (
                event_type,
                decision_id,
                trade_id,
                utc(event_ts).isoformat(),
                payload_hash,
                str(lock["sha256"]),
            )
        )
    )
    return {
        "event_id": event_id,
        "event_type": event_type,
        "event_ts": utc(event_ts),
        "recorded_at": utc(recorded_at),
        "decision_id": decision_id,
        "trade_id": trade_id,
        "payload_hash": payload_hash,
        "payload_json": payload_json,
        "parameter_lock_hash": str(lock["sha256"]),
        "strategy_code_hash": str(lock["strategy_code_hash"]),
        "git_commit": str(lock["strategy_code_commit"]),
        "previous_event_hash": "",
        "event_hash": "",
    }


def make_schedule_event(
    *,
    event_type: str,
    expected_run_ts: datetime,
    actual_start_ts: datetime | None,
    actual_finish_ts: datetime | None,
    run_mode: str,
    run_status: str,
    exit_code: int,
    eligible: bool,
    market_data_cutoff: datetime | None,
    error_code: str,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    if event_type not in SCHEDULE_EVENT_TYPES:
        raise ValueError(f"unsupported schedule event type: {event_type}")
    expected = utc(expected_run_ts)
    schedule_id = sha256_text(f"{STRATEGY_ID}|timer|{expected.isoformat()}")
    start = utc(actual_start_ts) if actual_start_ts else None
    finish = utc(actual_finish_ts) if actual_finish_ts else None
    latency = max(0.0, (start - expected).total_seconds()) if start else 0.0
    recorded = finish or start or expected
    event_id = sha256_text(
        "|".join(
            (
                schedule_id,
                event_type,
                start.isoformat() if start else "",
                finish.isoformat() if finish else "",
                run_status,
                str(exit_code),
                error_code,
            )
        )
    )
    return {
        "schedule_event_id": event_id,
        "event_type": event_type,
        "schedule_id": schedule_id,
        "strategy_id": STRATEGY_ID,
        "expected_run_ts": expected,
        "scheduled_run_ts": expected,
        "actual_start_ts": start,
        "actual_finish_ts": finish,
        "recorded_at": recorded,
        "latency_seconds": latency,
        "run_mode": run_mode,
        "run_status": run_status,
        "exit_code": int(exit_code),
        "runner_version": RUNNER_VERSION,
        "git_commit": str(lock["strategy_code_commit"]),
        "strategy_code_hash": str(lock["strategy_code_hash"]),
        "parameter_lock_hash": str(lock["sha256"]),
        "service_unit_hash": str(lock["service_unit_hash"]),
        "timer_unit_hash": str(lock["timer_unit_hash"]),
        "market_data_cutoff": utc(market_data_cutoff) if market_data_cutoff else None,
        "eligible_for_forward_evidence": bool(eligible),
        "error_code": error_code,
        "previous_event_hash": "",
        "event_hash": "",
    }


def schedule_statistics(
    events: pl.DataFrame, *, cutoff: datetime, through: datetime
) -> dict[str, Any]:
    formal = events.filter(pl.col("run_mode") == "REALTIME")
    expected = formal.filter(pl.col("event_type") == "SCHEDULE_EXPECTED")
    started = formal.filter(pl.col("event_type") == "RUN_STARTED")
    completed = formal.filter(pl.col("event_type") == "RUN_COMPLETED")
    failed = formal.filter(pl.col("event_type") == "RUN_FAILED")
    missed = formal.filter(pl.col("event_type") == "RUN_MISSED")
    late = formal.filter(pl.col("event_type") == "RUN_LATE")
    eligible = completed.filter(pl.col("eligible_for_forward_evidence"))
    expected_ids = set(expected["schedule_id"].to_list()) if not expected.is_empty() else set()
    eligible_ids = set(eligible["schedule_id"].to_list()) if not eligible.is_empty() else set()
    coverage = len(eligible_ids) / len(expected_ids) if expected_ids else 0.0
    calendar_days = max(0.0, (utc(through) - utc(cutoff)).total_seconds() / 86400.0)
    eligible_days = min(calendar_days, len(eligible_ids) / 24.0)
    observed_dates = {
        utc(value).date()
        for value in completed["actual_finish_ts"].drop_nulls().to_list()
    }
    longest_gap = 0.0
    current_gap = 0.0
    for slot in expected_timer_slots(cutoff, through):
        schedule_id = sha256_text(f"{STRATEGY_ID}|timer|{slot.isoformat()}")
        if schedule_id in eligible_ids:
            current_gap = 0.0
        else:
            current_gap += 1.0
            longest_gap = max(longest_gap, current_gap)
    return {
        "expected_schedule_count": len(expected_ids),
        "started_schedule_count": started.select(pl.col("schedule_id").n_unique()).item()
        if not started.is_empty()
        else 0,
        "completed_schedule_count": completed.select(pl.col("schedule_id").n_unique()).item()
        if not completed.is_empty()
        else 0,
        "failed_schedule_count": failed.select(pl.col("schedule_id").n_unique()).item()
        if not failed.is_empty()
        else 0,
        "missed_schedule_count": missed.select(pl.col("schedule_id").n_unique()).item()
        if not missed.is_empty()
        else 0,
        "late_schedule_count": late.select(pl.col("schedule_id").n_unique()).item()
        if not late.is_empty()
        else 0,
        "eligible_realtime_schedule_count": len(eligible_ids),
        "schedule_coverage": coverage,
        "forward_calendar_days": calendar_days,
        "eligible_forward_days": eligible_days,
        "runner_observed_days": len(observed_dates),
        "runner_online_hours": float(len(eligible_ids)),
        "longest_runner_gap_hours": longest_gap,
    }


def evidence_partitions(
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    schedule_events: pl.DataFrame,
) -> dict[str, Any]:
    """Return explicitly disjoint formal and recovery accounting scopes."""
    formal = decisions.filter(
        (pl.col("decision_origin") == "REALTIME")
        & pl.col("eligible_for_forward_evidence")
    )
    recovery = decisions.filter(
        pl.col("decision_origin") == "RECOVERY_RECONSTRUCTION"
    )
    formal_trades = trades.filter(
        (pl.col("decision_origin") == "REALTIME")
        & pl.col("eligible_for_forward_evidence")
        & pl.col("entry_price").is_not_null()
    )
    recovery_trades = trades.filter(
        (pl.col("decision_origin") == "RECOVERY_RECONSTRUCTION")
        & pl.col("entry_price").is_not_null()
    )
    formal_coverage = (
        float(formal["feature_data_coverage"].min()) if not formal.is_empty() else 0.0
    )
    recovery_coverage = (
        float(recovery["feature_data_coverage"].min())
        if not recovery.is_empty()
        else 0.0
    )
    formal_errors = schedule_events.filter(
        (pl.col("run_mode") == "REALTIME") & (pl.col("event_type") == "RUN_FAILED")
    ).height
    recovery_errors = schedule_events.filter(
        (pl.col("run_mode") == "RECOVERY") & (pl.col("event_type") == "RUN_FAILED")
    ).height
    return {
        "formal_decisions": formal,
        "recovery_decisions": recovery,
        "formal_trades": formal_trades,
        "recovery_trades": recovery_trades,
        "formal_realtime_decision_count": formal.height,
        "formal_realtime_trade_count": formal_trades.height,
        "formal_realtime_data_coverage": formal_coverage,
        "formal_realtime_incomplete_decision_count": formal.filter(
            pl.col("data_quality_status") != "PASS"
        ).height,
        "formal_realtime_runner_errors": formal_errors,
        "recovery_decision_count": recovery.height,
        "recovery_trade_count": recovery_trades.height,
        "recovery_data_coverage": recovery_coverage,
        "recovery_incomplete_decision_count": recovery.filter(
            pl.col("data_quality_status") != "PASS"
        ).height,
        "recovery_runner_errors": recovery_errors,
    }


def formal_cycle_period_returns(
    decisions: pl.DataFrame, trades: pl.DataFrame
) -> list[tuple[str, float]]:
    """Return only complete eligible REALTIME cycle returns, never recovery rows."""
    formal_cycles = decisions.filter(
        (pl.col("decision_origin") == "REALTIME")
        & pl.col("eligible_for_forward_evidence")
        & pl.col("cycle_eligible_for_forward_evidence")
    ).sort("scheduled_run_ts")
    formal_trades = trades.filter(
        (pl.col("decision_origin") == "REALTIME")
        & pl.col("eligible_for_forward_evidence")
        & pl.col("cycle_eligible_for_forward_evidence")
        & (pl.col("status") == "CLOSED")
    )
    result: list[tuple[str, float]] = []
    for decision_id in formal_cycles["decision_id"].to_list():
        local = formal_trades.filter(pl.col("decision_id") == decision_id)
        value = (
            float((local["target_weight"] * local["net_return"]).sum())
            if not local.is_empty()
            else 0.0
        )
        result.append((str(decision_id), value))
    return result


def formal_benchmark_period_returns(
    decisions: pl.DataFrame,
    benchmarks: pl.DataFrame,
    benchmark_type: str,
) -> list[tuple[str, float]]:
    """Return benchmark results only for complete formal REALTIME strategy cycles."""
    if benchmark_type not in BENCHMARK_TYPES:
        raise ValueError(f"unsupported benchmark type: {benchmark_type}")
    formal_cycles = decisions.filter(
        (pl.col("decision_origin") == "REALTIME")
        & pl.col("eligible_for_forward_evidence")
        & pl.col("cycle_eligible_for_forward_evidence")
    ).sort("scheduled_run_ts")
    formal_benchmarks = benchmarks.filter(
        (pl.col("decision_origin") == "REALTIME")
        & pl.col("eligible_for_forward_evidence")
        & pl.col("cycle_complete")
        & (pl.col("status") == "CLOSED")
        & (pl.col("benchmark_type") == benchmark_type)
    )
    result: list[tuple[str, float]] = []
    for decision_id in formal_cycles["decision_id"].to_list():
        local = formal_benchmarks.filter(pl.col("decision_id") == decision_id)
        if local.height != 1 or local["net_return"][0] is None:
            continue
        result.append((str(decision_id), float(local["net_return"][0])))
    return result


def evaluate_forward_status(
    *, metrics: Mapping[str, Any], lock: Mapping[str, Any]
) -> str:
    if metrics.get("integrity_errors"):
        return "FAIL_RUNNER_INTEGRITY"
    if not all(
        (
            bool(metrics.get("timer_installed")),
            bool(metrics.get("timer_enabled")),
            bool(metrics.get("timer_active")),
            bool(metrics.get("health_check_passed")),
            bool(metrics.get("cutoff_exists")),
        )
    ):
        return "INCONCLUSIVE_SYSTEM_NOT_DEPLOYED"
    expected_schedules = int(metrics.get("expected_schedule_count", 0))
    formal_decisions = int(metrics.get("formal_realtime_decision_count", 0))
    formal_cycles = int(metrics.get("completed_independent_cycles", 0))
    if (
        (expected_schedules > 0 and float(metrics.get("schedule_coverage", 0.0))
         < float(lock["minimum_schedule_coverage"]))
        or float(metrics.get("longest_runner_gap_hours", 0.0)) > 120.0
        or int(metrics.get("unexplained_critical_schedule_gap_count", 0)) > 0
        or (formal_decisions > 0 and float(metrics.get("formal_realtime_data_coverage", 0.0))
            < float(lock["minimum_data_coverage"]))
        or int(metrics.get("formal_realtime_incomplete_decision_count", 0)) > 0
        or (formal_cycles > 0 and float(metrics.get("minimum_benchmark_fill_coverage", 0.0))
            < float(lock["minimum_benchmark_fill_coverage"]))
        or int(metrics.get("incomplete_formal_benchmark_cycle_count", 0)) > 0
    ):
        return "FAIL_DATA_QUALITY"
    sample_ready = all(
        (
            float(metrics.get("eligible_forward_days", 0.0))
            >= float(lock["minimum_forward_days"]),
            formal_cycles >= int(lock["minimum_completed_cycles"]),
            int(metrics.get("actual_symbol_trades", 0))
            >= int(lock["minimum_symbol_trades"]),
        )
    )
    if not sample_ready:
        return "INCONCLUSIVE_SAMPLE_INSUFFICIENT"
    if float(metrics.get("base_cost_net_return", 0.0)) <= 0.0:
        return "FAIL_PERFORMANCE"
    if (
        float(metrics.get("excess_vs_btc", 0.0)) <= 0.0
        or float(metrics.get("excess_vs_dynamic_universe", 0.0)) <= 0.0
    ):
        return "FAIL_BENCHMARK_UNDERPERFORMANCE"
    if abs(float(metrics.get("max_drawdown", 0.0))) > float(lock["maximum_drawdown"]):
        return "FAIL_RISK_OR_DRAWDOWN"
    if float(metrics.get("maximum_single_symbol_contribution", 0.0)) > float(
        lock["maximum_single_symbol_contribution"]
    ):
        return "FAIL_CONCENTRATION"
    return "PAPER_REVIEW_READY"
