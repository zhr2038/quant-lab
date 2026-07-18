"""Audit v2.2 forward-paper integrity primitives.

This module is deliberately exchange-agnostic.  It provides immutable schemas,
hash chains, runtime identity checks, benchmark accounting, and the preregistered
paper status machine.  It never submits or mutates an exchange order.
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

RUNNER_VERSION = "quant_lab_low_vol_forward.v2.2"
STRATEGY_ID = "low_vol_btc_60d_trend_top3_score_120h_v22"
STRATEGY_VERSION = "2.2"
HOLDING_HOURS = 120
BTC_TREND_LOOKBACK_HOURS = 1440
FACTOR_LOOKBACK_HOURS = 480
MAX_DECISION_LATENCY_SECONDS = 3600
ENTRY_COST_BPS = 15.0
EXIT_COST_BPS = 15.0
ROUND_TRIP_COST_BPS = 30.0
BAR_CLOSE_DELAY_HOURS = 1

DECISION_ORIGINS = frozenset(
    {"REALTIME", "RECOVERY_RECONSTRUCTION", "MANUAL_REPLAY"}
)
PAPER_STATUSES = frozenset(
    {
        "INCONCLUSIVE_SAMPLE_INSUFFICIENT",
        "INCONCLUSIVE_DATA_INCOMPLETE",
        "INCONCLUSIVE_SYSTEM_ERROR",
        "FAIL_PERFORMANCE",
        "FAIL_BENCHMARK_UNDERPERFORMANCE",
        "FAIL_RISK_OR_DRAWDOWN",
        "FAIL_CONCENTRATION",
        "FAIL_DATA_QUALITY",
        "FAIL_RUNNER_INTEGRITY",
        "CONTINUE_PAPER",
        "PAPER_REVIEW_READY",
    }
)

DECISION_SCHEMA = {
    "decision_id": pl.Utf8,
    "strategy_id": pl.Utf8,
    "strategy_version": pl.Utf8,
    "decision_ts": pl.Datetime("us", "UTC"),
    "scheduled_run_ts": pl.Datetime("us", "UTC"),
    "recorded_at": pl.Datetime("us", "UTC"),
    "decision_latency_seconds": pl.Float64,
    "decision_origin": pl.Utf8,
    "late_reconstructed": pl.Boolean,
    "eligible_for_forward_evidence": pl.Boolean,
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
    "eligible_for_forward_evidence": pl.Boolean,
    "market_data_snapshot_id": pl.Utf8,
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
    "decision_ts": pl.Datetime("us", "UTC"),
    "entry_ts": pl.Datetime("us", "UTC"),
    "scheduled_exit_ts": pl.Datetime("us", "UTC"),
    "actual_exit_ts": pl.Datetime("us", "UTC"),
    "entry_price": pl.Float64,
    "exit_price": pl.Float64,
    "weights": pl.Utf8,
    "symbols": pl.Utf8,
    "gross_return": pl.Float64,
    "net_return": pl.Float64,
    "data_snapshot_hash": pl.Utf8,
    "market_data_cutoff": pl.Datetime("us", "UTC"),
    "code_commit": pl.Utf8,
    "parameter_lock_hash": pl.Utf8,
    "strategy_code_hash": pl.Utf8,
    "eligible_for_forward_evidence": pl.Boolean,
    "status": pl.Utf8,
}

BENCHMARK_TRADE_SCHEMA = {
    "benchmark_trade_id": pl.Utf8,
    "benchmark_id": pl.Utf8,
    "benchmark_type": pl.Utf8,
    "symbol": pl.Utf8,
    "weight": pl.Float64,
    "entry_ts": pl.Datetime("us", "UTC"),
    "entry_price": pl.Float64,
    "scheduled_exit_ts": pl.Datetime("us", "UTC"),
    "actual_exit_ts": pl.Datetime("us", "UTC"),
    "exit_price": pl.Float64,
    "gross_return": pl.Float64,
    "net_return": pl.Float64,
    "entry_cost_bps": pl.Float64,
    "exit_cost_bps": pl.Float64,
    "data_snapshot_hash": pl.Utf8,
    "eligible_for_forward_evidence": pl.Boolean,
    "status": pl.Utf8,
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

IMMUTABLE_TRADE_FIELDS = (
    "trade_id",
    "decision_id",
    "symbol",
    "target_weight",
    "entry_ts",
    "entry_price",
    "scheduled_exit_ts",
    "entry_cost_bps",
    "exit_cost_bps",
    "eligible_for_forward_evidence",
    "market_data_snapshot_id",
    "parameter_lock_hash",
    "strategy_code_hash",
    "git_commit",
)

IMMUTABLE_BENCHMARK_TRADE_FIELDS = (
    "benchmark_trade_id",
    "benchmark_id",
    "benchmark_type",
    "symbol",
    "weight",
    "entry_ts",
    "entry_price",
    "scheduled_exit_ts",
    "entry_cost_bps",
    "exit_cost_bps",
    "data_snapshot_hash",
    "eligible_for_forward_evidence",
)

IMMUTABLE_BENCHMARK_DECISION_FIELDS = (
    "benchmark_id",
    "benchmark_type",
    "decision_id",
    "decision_ts",
    "entry_ts",
    "scheduled_exit_ts",
    "entry_price",
    "weights",
    "symbols",
    "data_snapshot_hash",
    "market_data_cutoff",
    "code_commit",
    "parameter_lock_hash",
    "strategy_code_hash",
    "eligible_for_forward_evidence",
)


def empty_frame(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict(schema))


def utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


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


def parameter_lock_digest(lock: Mapping[str, Any]) -> str:
    return payload_digest({key: value for key, value in lock.items() if key != "sha256"})


def source_hash_entries(repo: Path, paths: Iterable[str]) -> list[dict[str, str]]:
    repo = repo.resolve()
    entries: list[dict[str, str]] = []
    for raw in sorted(set(paths)):
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"locked source path must be repository-relative: {raw}")
        path = (repo / relative).resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"locked source escapes repository: {raw}") from exc
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        entries.append({"path": relative.as_posix(), "sha256": sha256_file(path)})
    return entries


def composite_source_hash(entries: Sequence[Mapping[str, str]]) -> str:
    normalized = [
        {"path": str(entry["path"]), "sha256": str(entry["sha256"])}
        for entry in sorted(entries, key=lambda item: str(item["path"]))
    ]
    return payload_digest(normalized)


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
        "universe": "dynamic_top20_v1",
        "maximum_single_position_weight": 0.50,
        "rebalance_hours": HOLDING_HOURS,
        "holding_hours": HOLDING_HOURS,
        "schedule_anchor_rule": "first_full_hour_strictly_after_cutoff_then_120h",
        "decision_delay_bars": 1,
        "strategy_type": "long_only_spot",
        "live_order_effect": "none",
        "automatic_promotion": False,
        "entry_cost_bps": ENTRY_COST_BPS,
        "exit_cost_bps": EXIT_COST_BPS,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "max_decision_latency_seconds": MAX_DECISION_LATENCY_SECONDS,
        "minimum_forward_days": 30,
        "minimum_completed_cycles": 6,
        "minimum_symbol_trades": 12,
        "minimum_data_coverage": 0.95,
        "maximum_single_symbol_contribution": 0.50,
        "maximum_drawdown": 0.30,
        "runner_version": RUNNER_VERSION,
    }
    mismatches = {
        key: {"expected": value, "actual": lock.get(key)}
        for key, value in expected.items()
        if lock.get(key) != value
    }
    if mismatches:
        raise ValueError(f"parameter lock violates v2.2 hypothesis: {mismatches}")
    if len(str(lock.get("strategy_code_commit", ""))) != 40:
        raise ValueError("strategy_code_commit must be a full 40-character SHA")
    locked = lock.get("locked_source_files")
    if not isinstance(locked, list) or not locked:
        raise ValueError("locked_source_files must be a non-empty list")
    if composite_source_hash(locked) != lock.get("strategy_code_hash"):
        raise ValueError("strategy_code_hash does not match locked_source_files")
    if lock.get("sha256") != parameter_lock_digest(lock):
        raise ValueError("parameter lock sha256 mismatch")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def runtime_identity(repo: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fail-closed identity receipt without mutating evidence."""
    errors: list[str] = []
    try:
        validate_parameter_lock(lock)
    except Exception as exc:
        errors.append(f"PARAMETER_LOCK_INVALID:{exc}")
    head = ""
    dirty = ""
    try:
        head = _git(repo, "rev-parse", "HEAD")
        dirty = _git(repo, "status", "--porcelain")
    except Exception as exc:
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
        expected_entries = {
            str(item["path"]): str(item["sha256"])
            for item in lock.get("locked_source_files", [])
        }
        current_map = {item["path"]: item["sha256"] for item in current_entries}
        if current_map != expected_entries:
            errors.append("LOCKED_SOURCE_HASH_MISMATCH")
        if composite_source_hash(current_entries) != lock.get("strategy_code_hash"):
            errors.append("STRATEGY_CODE_HASH_MISMATCH")
    except Exception as exc:
        errors.append(f"LOCKED_SOURCE_UNAVAILABLE:{exc}")
    return {
        "ok": not errors,
        "status": "PASS" if not errors else "FAIL_RUNNER_INTEGRITY",
        "errors": errors,
        "current_head": head,
        "expected_head": str(lock.get("strategy_code_commit", "")),
        "working_tree_clean": not bool(dirty),
        "strategy_code_hash": (
            composite_source_hash(current_entries) if current_entries else ""
        ),
        "expected_strategy_code_hash": str(lock.get("strategy_code_hash", "")),
    }


def schedule_anchor(cutoff: datetime) -> datetime:
    value = utc(cutoff)
    floor = value.replace(minute=0, second=0, microsecond=0)
    return floor + timedelta(hours=1)


def due_schedule_times(cutoff: datetime, as_of: datetime) -> list[datetime]:
    anchor = schedule_anchor(cutoff)
    end = utc(as_of)
    if anchor > end:
        return []
    count = int((end - anchor).total_seconds() // (HOLDING_HOURS * 3600)) + 1
    return [anchor + timedelta(hours=HOLDING_HOURS * index) for index in range(count)]


def classify_decision_origin(
    *, mode: str, scheduled_run_ts: datetime, recorded_at: datetime, max_latency: int
) -> tuple[str, bool, bool, float]:
    scheduled = utc(scheduled_run_ts)
    recorded = utc(recorded_at)
    latency = max(0.0, (recorded - scheduled).total_seconds())
    if mode == "recovery":
        return "RECOVERY_RECONSTRUCTION", True, False, latency
    if mode == "manual_replay":
        return "MANUAL_REPLAY", True, False, latency
    if mode != "realtime":
        raise ValueError(f"unknown forward mode: {mode}")
    late = latency > float(max_latency)
    return (
        "RECOVERY_RECONSTRUCTION" if late else "REALTIME",
        late,
        not late,
        latency,
    )


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


def _event_core(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(row["event_id"]),
        "event_type": str(row["event_type"]),
        "event_ts": utc(row["event_ts"]).isoformat(),
        "recorded_at": utc(row["recorded_at"]).isoformat(),
        "decision_id": str(row.get("decision_id") or ""),
        "trade_id": str(row.get("trade_id") or ""),
        "payload_hash": str(row["payload_hash"]),
        "payload_json": str(row["payload_json"]),
        "parameter_lock_hash": str(row["parameter_lock_hash"]),
        "strategy_code_hash": str(row["strategy_code_hash"]),
        "git_commit": str(row["git_commit"]),
    }


def make_event(
    *,
    event_type: str,
    event_ts: datetime,
    recorded_at: datetime,
    decision_id: str,
    trade_id: str = "",
    payload: Mapping[str, Any] | None = None,
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


def validate_event_chain(events: pl.DataFrame) -> None:
    previous = ""
    for row in events.iter_rows(named=True):
        if str(row["previous_event_hash"]) != previous:
            raise ValueError("event chain previous hash mismatch")
        expected = sha256_text(previous + canonical_json(_event_core(row)))
        if str(row["event_hash"]) != expected:
            raise ValueError("event chain hash mismatch")
        previous = expected


def append_hash_chain(existing: pl.DataFrame, additions: pl.DataFrame) -> pl.DataFrame:
    if not existing.is_empty():
        validate_event_chain(existing)
    rows = existing.to_dicts()
    by_id = {str(row["event_id"]): row for row in rows}
    previous = str(rows[-1]["event_hash"]) if rows else ""
    ordered = (
        additions.sort(["recorded_at", "event_ts", "event_id"])
        if not additions.is_empty()
        else additions
    )
    for raw in ordered.iter_rows(named=True):
        event_id = str(raw["event_id"])
        prior = by_id.get(event_id)
        if prior is not None:
            prior_core = _event_core(prior)
            incoming_core = _event_core(raw)
            incoming_core["recorded_at"] = prior_core["recorded_at"]
            if incoming_core != prior_core:
                raise ValueError("immutable event payload changed")
            continue
        row = dict(raw)
        row["previous_event_hash"] = previous
        row["event_hash"] = sha256_text(previous + canonical_json(_event_core(row)))
        previous = row["event_hash"]
        rows.append(row)
        by_id[event_id] = row
    result = pl.DataFrame(rows, schema=EVENT_SCHEMA) if rows else empty_frame(EVENT_SCHEMA)
    validate_event_chain(result)
    return result


def _normal(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc(value).isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _assert_fields(
    old: Mapping[str, Any], new: Mapping[str, Any], fields: Sequence[str], label: str
) -> None:
    changed = [name for name in fields if _normal(old.get(name)) != _normal(new.get(name))]
    if changed:
        raise ValueError(f"immutable {label} fields changed: {changed}")


def merge_immutable_rows(
    existing: pl.DataFrame,
    additions: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    id_field: str,
    sort_fields: Sequence[str],
    label: str,
) -> pl.DataFrame:
    rows = {str(row[id_field]): row for row in existing.iter_rows(named=True)}
    for row in additions.iter_rows(named=True):
        key = str(row[id_field])
        if key in rows:
            _assert_fields(rows[key], row, tuple(schema), label)
        else:
            rows[key] = row
    if not rows:
        return empty_frame(schema)
    return pl.DataFrame(list(rows.values()), schema=dict(schema)).sort(list(sort_fields))


def merge_trade_states(existing: pl.DataFrame, updates: pl.DataFrame) -> pl.DataFrame:
    rows = {str(row["trade_id"]): row for row in existing.iter_rows(named=True)}
    for row in updates.iter_rows(named=True):
        key = str(row["trade_id"])
        prior = rows.get(key)
        if prior is None:
            rows[key] = row
            continue
        _assert_fields(prior, row, IMMUTABLE_TRADE_FIELDS, "trade")
        if prior["status"] == "CLOSED":
            _assert_fields(prior, row, tuple(TRADE_SCHEMA), "closed trade")
        else:
            rows[key] = row
    if not rows:
        return empty_frame(TRADE_SCHEMA)
    return pl.DataFrame(list(rows.values()), schema=TRADE_SCHEMA).sort(["entry_ts", "symbol"])


def merge_benchmark_trade_states(existing: pl.DataFrame, updates: pl.DataFrame) -> pl.DataFrame:
    rows = {str(row["benchmark_trade_id"]): row for row in existing.iter_rows(named=True)}
    for row in updates.iter_rows(named=True):
        key = str(row["benchmark_trade_id"])
        prior = rows.get(key)
        if prior is None:
            rows[key] = row
            continue
        _assert_fields(prior, row, IMMUTABLE_BENCHMARK_TRADE_FIELDS, "benchmark trade")
        if prior["status"] == "CLOSED":
            _assert_fields(prior, row, tuple(BENCHMARK_TRADE_SCHEMA), "closed benchmark trade")
        else:
            rows[key] = row
    if not rows:
        return empty_frame(BENCHMARK_TRADE_SCHEMA)
    return pl.DataFrame(list(rows.values()), schema=BENCHMARK_TRADE_SCHEMA).sort(
        ["entry_ts", "benchmark_type", "symbol"]
    )


def merge_benchmark_decision_states(
    existing: pl.DataFrame, updates: pl.DataFrame
) -> pl.DataFrame:
    rows = {str(row["benchmark_id"]): row for row in existing.iter_rows(named=True)}
    for row in updates.iter_rows(named=True):
        key = str(row["benchmark_id"])
        prior = rows.get(key)
        if prior is None:
            rows[key] = row
            continue
        _assert_fields(
            prior, row, IMMUTABLE_BENCHMARK_DECISION_FIELDS, "benchmark decision"
        )
        if prior["status"] == "CLOSED":
            _assert_fields(
                prior, row, tuple(BENCHMARK_DECISION_SCHEMA), "closed benchmark decision"
            )
        else:
            rows[key] = row
    if not rows:
        return empty_frame(BENCHMARK_DECISION_SCHEMA)
    return pl.DataFrame(list(rows.values()), schema=BENCHMARK_DECISION_SCHEMA).sort(
        ["decision_ts", "benchmark_type"]
    )


def evaluate_forward_status(
    *, metrics: Mapping[str, Any], lock: Mapping[str, Any], integrity_errors: Sequence[str] = ()
) -> str:
    if integrity_errors:
        return "FAIL_RUNNER_INTEGRITY"
    if int(metrics.get("system_error_count", 0)) > 0:
        return "INCONCLUSIVE_SYSTEM_ERROR"
    if bool(metrics.get("data_completeness_unknown", False)):
        return "INCONCLUSIVE_DATA_INCOMPLETE"
    sample_ready = all(
        (
            float(metrics.get("forward_days", 0.0)) >= float(lock["minimum_forward_days"]),
            int(metrics.get("completed_independent_cycles", 0))
            >= int(lock["minimum_completed_cycles"]),
            int(metrics.get("actual_symbol_trades", 0))
            >= int(lock["minimum_symbol_trades"]),
        )
    )
    if not sample_ready:
        return "INCONCLUSIVE_SAMPLE_INSUFFICIENT"
    if (
        float(metrics.get("data_coverage", 0.0)) < float(lock["minimum_data_coverage"])
        or int(metrics.get("unhandled_market_gap_count", 0)) > 0
        or int(metrics.get("unexplained_missing_fill_count", 0)) > 0
    ):
        return "FAIL_DATA_QUALITY"
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


def build_realized_equity(
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    cutoff: datetime,
    completed_decision_ids: Iterable[str] | None = None,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = [
        {"timestamp": utc(cutoff), "strategy_equity": 1.0, "completed_cycle_count": 0}
    ]
    equity = 1.0
    count = 0
    completed = (
        {str(value) for value in completed_decision_ids}
        if completed_decision_ids is not None
        else None
    )
    eligible = decisions.filter(pl.col("eligible_for_forward_evidence"))
    for decision in eligible.sort("scheduled_run_ts").iter_rows(named=True):
        if completed is not None and str(decision["decision_id"]) not in completed:
            continue
        local = trades.filter(pl.col("decision_id") == decision["decision_id"])
        if local.is_empty():
            if completed is None and decision["status"] != "CASH_COMPLETE":
                continue
            period_return = 0.0
            timestamp = utc(decision["scheduled_run_ts"]) + timedelta(
                hours=BAR_CLOSE_DELAY_HOURS + HOLDING_HOURS
            )
        elif not local.select(pl.col("status").eq("CLOSED").all()).item():
            continue
        else:
            period_return = float(
                (local["target_weight"] * local["net_return"]).sum()
            )
            timestamp = utc(local["actual_exit_ts"].max())
        equity *= 1.0 + period_return
        count += 1
        rows.append(
            {"timestamp": timestamp, "strategy_equity": equity, "completed_cycle_count": count}
        )
    return pl.DataFrame(rows, schema=EQUITY_SCHEMA)


def build_benchmark_equity(
    benchmarks: pl.DataFrame, cutoff: datetime
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "timestamp": utc(cutoff),
            "btc_equity": 1.0,
            "dynamic_universe_equity": 1.0,
            "cash_equity": 1.0,
            "completed_cycle_count": 0,
        }
    ]
    state = {"BTC_BUY_AND_HOLD": 1.0, "DYNAMIC_UNIVERSE_EQUAL_WEIGHT": 1.0, "CASH": 1.0}
    completed = benchmarks.filter(
        pl.col("eligible_for_forward_evidence") & (pl.col("status") == "CLOSED")
    )
    by_decision: dict[str, list[dict[str, Any]]] = {}
    for row in completed.sort(["decision_ts", "benchmark_type"]).iter_rows(named=True):
        by_decision.setdefault(str(row["decision_id"]), []).append(row)
    count = 0
    for group in by_decision.values():
        if {str(row["benchmark_type"]) for row in group} != set(state):
            continue
        timestamp = max(utc(row["actual_exit_ts"]) for row in group)
        for row in group:
            state[str(row["benchmark_type"])] *= 1.0 + float(row["net_return"])
        count += 1
        rows.append(
            {
                "timestamp": timestamp,
                "btc_equity": state["BTC_BUY_AND_HOLD"],
                "dynamic_universe_equity": state["DYNAMIC_UNIVERSE_EQUAL_WEIGHT"],
                "cash_equity": state["CASH"],
                "completed_cycle_count": count,
            }
        )
    return pl.DataFrame(rows, schema=BENCHMARK_EQUITY_SCHEMA)


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    frame.write_parquet(partial, compression="zstd")
    os.replace(partial, path)


def atomic_write_csv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    frame.write_csv(partial)
    os.replace(partial, path)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
