# ruff: noqa: E501
"""Create the locked v2.2.1 identity and empty, not-yet-deployed evidence store."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v221 import (  # noqa: E402
    AUDIT_VERSION,
    BENCHMARK_COVERAGE_SCHEMA,
    BENCHMARK_DECISION_SCHEMA,
    BENCHMARK_EQUITY_SCHEMA,
    BENCHMARK_TRADE_SCHEMA,
    DECISION_SCHEMA,
    ENTRY_COST_BPS,
    EQUITY_SCHEMA,
    EVENT_SCHEMA,
    EXIT_COST_BPS,
    FACTOR_LOOKBACK_HOURS,
    HOLDING_HOURS,
    MAX_DECISION_LATENCY_SECONDS,
    ROUND_TRIP_COST_BPS,
    RUNNER_VERSION,
    RUNTIME_HEALTH_SCHEMA,
    SCHEDULE_EVENT_SCHEMA,
    STRATEGY_ID,
    STRATEGY_SCHEDULE_ANCHOR,
    STRATEGY_VERSION,
    TRADE_SCHEMA,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    composite_source_hash,
    empty_frame,
    parameter_lock_digest,
    payload_digest,
    sha256_file,
    source_hash_entries,
    utc,
    validate_parameter_lock,
)

STRATEGY_SOURCE_FILES = (
    "audit/auditlib/factors.py",
    "audit/auditlib/forward_v221.py",
    "audit/auditlib/portfolio_backtest.py",
    "audit/auditlib/universe.py",
    "audit/scripts/stage_v221_deployment.py",
    "audit/scripts/stage_v221_forward.py",
    "schemas/forward_parameter_lock_v221.schema.json",
    "scripts/run_forward_v221_realtime.sh",
)
REPORTING_SOURCE_FILES = (
    "audit/scripts/stage_v221_bundle.py",
    "audit/scripts/stage_v221_consistency.py",
    "audit/scripts/stage_v221_init.py",
    "audit/scripts/stage_v221_report.py",
    "audit/scripts/stage_v221_test.py",
    "scripts/run_alpha_audit_v221.sh",
    "scripts/run_forward_v221_recovery.sh",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def build_lock(repo: Path, created_at: datetime) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise RuntimeError("working tree must be clean before parameter lock creation")
    strategy_entries = source_hash_entries(repo, STRATEGY_SOURCE_FILES)
    reporting_entries = source_hash_entries(repo, REPORTING_SOURCE_FILES)
    lock: dict[str, Any] = {
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "hypothesis_type": "POST_HOC_HYPOTHESIS",
        "approval_state": "PAPER_ONLY",
        "parameters_locked": True,
        "factor": "low_vol_20d",
        "factor_formula": "-rolling_std(ret_1h, 480)",
        "factor_lookback_hours": FACTOR_LOOKBACK_HOURS,
        "btc_trend_filter": True,
        "btc_trend_lookback_hours": 1440,
        "btc_trend_description": "BTC 60-day SMA trend filter",
        "top_n": 3,
        "weighting": "score",
        "universe": "dynamic_top20_v1",
        "maximum_single_position_weight": 0.50,
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
        "strategy_code_commit": head,
        "strategy_code_hash": composite_source_hash(strategy_entries),
        "reporting_code_hash": composite_source_hash(reporting_entries),
        "locked_source_files": strategy_entries,
        "reporting_source_files": reporting_entries,
        "service_unit_hash": sha256_file(
            repo / "deploy/systemd/quant-lab-forward-v221.service"
        ),
        "timer_unit_hash": sha256_file(
            repo / "deploy/systemd/quant-lab-forward-v221.timer"
        ),
        "runner_script_hash": sha256_file(
            repo / "scripts/run_forward_v221_realtime.sh"
        ),
        "schedule_anchor": STRATEGY_SCHEDULE_ANCHOR.isoformat(),
        "schedule_timezone": "UTC",
        "runner_version": RUNNER_VERSION,
        "execution_mode": "PAPER",
        "production_alpha": "FROZEN",
        "live_opening_enabled": False,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "universe_definition_hash": sha256_file(repo / "audit/auditlib/universe.py"),
        "cost_model_hash": payload_digest(
            {
                "entry_cost_bps": ENTRY_COST_BPS,
                "exit_cost_bps": EXIT_COST_BPS,
                "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            }
        ),
        "created_at": utc(created_at).isoformat(),
    }
    lock["sha256"] = parameter_lock_digest(lock)
    validate_parameter_lock(lock)
    return lock


def _initial_status(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "quant_lab_forward_v221_status.v1",
        "audit_version": AUDIT_VERSION,
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "forward_start_status": "NOT_READY",
        "paper_status": "INCONCLUSIVE_SYSTEM_NOT_DEPLOYED",
        "forward_v221_cutoff": "",
        "next_legal_strategy_decision": "",
        "research_node": "",
        "timer_installed": False,
        "timer_enabled": False,
        "timer_active": False,
        "next_timer_trigger": "",
        "formal_realtime_decision_count": 0,
        "formal_realtime_trade_count": 0,
        "formal_realtime_data_coverage": 0.0,
        "formal_realtime_incomplete_decision_count": 0,
        "formal_realtime_runner_errors": 0,
        "recovery_decision_count": 0,
        "recovery_trade_count": 0,
        "recovery_data_coverage": 0.0,
        "recovery_incomplete_decision_count": 0,
        "recovery_runner_errors": 0,
        "recovery_audit_warning": False,
        "expected_schedule_count": 0,
        "completed_schedule_count": 0,
        "failed_schedule_count": 0,
        "missed_schedule_count": 0,
        "late_schedule_count": 0,
        "eligible_realtime_schedule_count": 0,
        "schedule_coverage": 0.0,
        "forward_calendar_days": 0.0,
        "eligible_forward_days": 0.0,
        "runner_observed_days": 0,
        "runner_online_hours": 0.0,
        "longest_runner_gap_hours": 0.0,
        "actual_entry_count": 0,
        "completed_independent_cycles": 0,
        "actual_symbol_trades": 0,
        "btc_benchmark_complete_cycle_count": 0,
        "dynamic_universe_benchmark_complete_cycle_count": 0,
        "cash_benchmark_complete_cycle_count": 0,
        "minimum_benchmark_fill_coverage": 0.0,
        "benchmark_cash_residual": 0.0,
        "strategy_net_return": 0.0,
        "btc_benchmark_return": 0.0,
        "dynamic_universe_benchmark_return": 0.0,
        "cash_benchmark_return": 0.0,
        "strategy_excess_vs_btc": 0.0,
        "strategy_excess_vs_dynamic_universe": 0.0,
        "parameter_lock_hash": lock["sha256"],
        "strategy_code_commit": lock["strategy_code_commit"],
        "strategy_code_hash": lock["strategy_code_hash"],
        "reporting_code_hash": lock["reporting_code_hash"],
        "service_unit_hash": lock["service_unit_hash"],
        "timer_unit_hash": lock["timer_unit_hash"],
        "runner_script_hash": lock["runner_script_hash"],
        "runner_integrity": "PASS",
        "working_tree_clean": True,
        "execution_mode": "PAPER",
        "approval_state": "PAPER_ONLY",
        "production_alpha": "FROZEN",
        "live_opening_enabled": False,
        "live": "NOT_ALLOWED",
        "live_order_effect": "none",
        "automatic_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get("AUDIT_V221_ROOT", "/home/hr/quant-alpha-audit-v2.2.1")
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--created-at")
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve()
    created_at = utc(args.created_at or datetime.now(UTC).replace(microsecond=0))
    consistency = root / "artifacts/v22_consistency_check.json"
    if not consistency.is_file() or json.loads(consistency.read_text())["status"] != "PASS":
        raise RuntimeError("v2.2 consistency must pass before v2.2.1 init")
    lock = build_lock(repo, created_at)
    atomic_write_json(lock, root / "manifests/parameter_lock_v221.json")
    strategy_lines = "\n".join(
        f"{item['sha256']}  {item['path']}" for item in lock["locked_source_files"]
    )
    _write_text(root / "manifests/strategy_code_hashes_v221.txt", strategy_lines + "\n")
    deployment_lines = "\n".join(
        (
            f"{lock['service_unit_hash']}  deploy/systemd/quant-lab-forward-v221.service",
            f"{lock['timer_unit_hash']}  deploy/systemd/quant-lab-forward-v221.timer",
            f"{lock['runner_script_hash']}  scripts/run_forward_v221_realtime.sh",
        )
    )
    _write_text(root / "manifests/deployment_hashes_v221.txt", deployment_lines + "\n")
    atomic_write_json(
        {
            "schema_version": "quant_lab_forward_v22_supersession.v1",
            "status": "SUPERSEDED_BY_V221",
            "v22_bundle_sha256": "cf2d7514c68f8e38b15e609003d32b1d43e044c5ca893838db72bfc961c9ea9d",
            "v22_formal_decisions_migrated": 0,
            "v22_recovery_decisions_migrated": 0,
            "v22_evidence_deleted": False,
            "v221_strategy_id": STRATEGY_ID,
            "recorded_at": created_at.isoformat(),
        },
        root / "artifacts/v22_forward_supersession.json",
    )
    schemas = {
        "forward_v221_decisions.parquet": DECISION_SCHEMA,
        "forward_v221_trades.parquet": TRADE_SCHEMA,
        "forward_v221_events.parquet": EVENT_SCHEMA,
        "forward_v221_equity.parquet": EQUITY_SCHEMA,
        "forward_v221_benchmark_decisions.parquet": BENCHMARK_DECISION_SCHEMA,
        "forward_v221_benchmark_trades.parquet": BENCHMARK_TRADE_SCHEMA,
        "forward_v221_benchmark_events.parquet": EVENT_SCHEMA,
        "forward_v221_benchmark_equity.parquet": BENCHMARK_EQUITY_SCHEMA,
        "forward_v221_benchmark_coverage.parquet": BENCHMARK_COVERAGE_SCHEMA,
        "forward_v221_schedule_events.parquet": SCHEDULE_EVENT_SCHEMA,
        "forward_v221_runtime_health.parquet": RUNTIME_HEALTH_SCHEMA,
    }
    for name, schema in schemas.items():
        path = root / "artifacts" / name
        if not path.exists():
            atomic_write_parquet(empty_frame(schema), path)
    status = _initial_status(lock)
    atomic_write_json(status, root / "artifacts/forward_v221_status.json")
    atomic_write_csv(pl.DataFrame([status], infer_schema_length=None), root / "artifacts/forward_v221_performance.csv")
    print(f"strategy_id={STRATEGY_ID}")
    print(f"strategy_code_commit={lock['strategy_code_commit']}")
    print(f"strategy_code_hash={lock['strategy_code_hash']}")
    print(f"parameter_lock_hash={lock['sha256']}")
    print("forward_start_status=NOT_READY")
    print("paper_status=INCONCLUSIVE_SYSTEM_NOT_DEPLOYED")


if __name__ == "__main__":
    main()
