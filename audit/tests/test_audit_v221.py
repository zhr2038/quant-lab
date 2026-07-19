# ruff: noqa: E501
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from audit.auditlib.forward_v221 import (
    BENCHMARK_DECISION_SCHEMA,
    DECISION_SCHEMA,
    SCHEDULE_EVENT_SCHEMA,
    STRATEGY_ID,
    STRATEGY_SCHEDULE_ANCHOR,
    TRADE_SCHEMA,
    append_hash_chain,
    benchmark_entry_records,
    composite_source_hash,
    deployment_manifest_digest,
    evaluate_forward_status,
    evidence_partitions,
    expected_timer_slots,
    formal_benchmark_period_returns,
    formal_cycle_period_returns,
    frame,
    make_schedule_event,
    merge_state_rows,
    next_strategy_schedule,
    parameter_lock_digest,
    runtime_identity,
    schedule_statistics,
    sha256_file,
    source_hash_entries,
    timer_slot,
)
from audit.auditlib.portfolio_backtest import _capped_weights
from audit.scripts.stage_v221_deployment import create_cutoff
from audit.scripts.stage_v221_forward import _locked_capped_score_weights, finalize_cycles

NOW = datetime(2026, 7, 19, 2, 10, tzinfo=UTC)


def test_locked_score_weight_formula_is_numerically_identical_to_v22() -> None:
    rng = np.random.default_rng(20260719)
    for _ in range(100):
        scores = rng.normal(size=3)
        expected = _capped_weights(scores, 3, 0.50, "score")
        actual = _locked_capped_score_weights(scores, 3, 0.50)
        np.testing.assert_array_equal(actual, expected)


def _minimal_lock() -> dict[str, object]:
    return {
        "minimum_benchmark_fill_coverage": 0.95,
        "minimum_schedule_coverage": 0.95,
        "minimum_data_coverage": 0.95,
        "minimum_forward_days": 30,
        "minimum_completed_cycles": 6,
        "minimum_symbol_trades": 12,
        "maximum_single_symbol_contribution": 0.50,
        "maximum_drawdown": 0.30,
        "max_decision_latency_seconds": 3600,
        "strategy_code_commit": "a" * 40,
        "strategy_code_hash": "b" * 64,
        "sha256": "c" * 64,
        "service_unit_hash": "d" * 64,
        "timer_unit_hash": "e" * 64,
    }


def _decision(
    decision_id: str,
    *,
    origin: str = "REALTIME",
    eligible: bool = True,
    cycle_eligible: bool = False,
    coverage: float = 1.0,
    selected: list[str] | None = None,
) -> dict[str, object]:
    selected = selected or []
    weights = {symbol: 1.0 / len(selected) for symbol in selected} if selected else {}
    return {
        "decision_id": decision_id,
        "schedule_id": "schedule",
        "strategy_id": STRATEGY_ID,
        "strategy_version": "2.2.1",
        "decision_ts": NOW,
        "scheduled_run_ts": NOW,
        "recorded_at": NOW + timedelta(minutes=10),
        "decision_latency_seconds": 600.0,
        "decision_origin": origin,
        "late_reconstructed": origin != "REALTIME",
        "eligible_for_forward_evidence": eligible,
        "cycle_eligible_for_forward_evidence": cycle_eligible,
        "btc_benchmark_cycle_complete": cycle_eligible,
        "universe_benchmark_cycle_complete": cycle_eligible,
        "cash_benchmark_cycle_complete": cycle_eligible,
        "all_benchmarks_complete": cycle_eligible,
        "feature_cutoff_ts": NOW,
        "observed_market_data_cutoff": NOW,
        "market_data_cutoff": NOW,
        "market_data_snapshot_id": "decision-snapshot",
        "input_file_paths": "[]",
        "input_file_sha256": "{}",
        "feature_data_coverage": coverage,
        "data_quality_status": "PASS" if coverage >= 0.95 else "INCOMPLETE",
        "btc_trend_state": "UP",
        "universe": "[]",
        "factor_scores": "[]",
        "ranked_symbols": "[]",
        "selected_symbols": json.dumps(selected),
        "target_weights": json.dumps(weights),
        "cash_weight": 1.0 - sum(weights.values()),
        "parameter_lock_hash": "c" * 64,
        "strategy_code_hash": "b" * 64,
        "git_commit": "a" * 40,
        "working_tree_clean": True,
        "status": "COMPLETE" if cycle_eligible else "CASH",
    }


def _trade(
    trade_id: str,
    decision_id: str,
    *,
    origin: str = "REALTIME",
    eligible: bool = True,
    cycle_eligible: bool = False,
    net_return: float = 0.10,
) -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "decision_id": decision_id,
        "decision_origin": origin,
        "symbol": "BTC-USDT",
        "target_weight": 1.0,
        "entry_ts": NOW + timedelta(hours=1),
        "entry_price": 100.0,
        "scheduled_exit_ts": NOW + timedelta(hours=121),
        "actual_exit_ts": NOW + timedelta(hours=121),
        "exit_price": 110.0,
        "entry_cost_bps": 15.0,
        "exit_cost_bps": 15.0,
        "gross_return": net_return + 0.003,
        "net_return": net_return,
        "entry_market_data_snapshot_id": "entry-snapshot",
        "exit_market_data_snapshot_id": "exit-snapshot",
        "eligible_for_forward_evidence": eligible,
        "cycle_eligible_for_forward_evidence": cycle_eligible,
        "parameter_lock_hash": "c" * 64,
        "strategy_code_hash": "b" * 64,
        "git_commit": "a" * 40,
        "status": "CLOSED",
    }


def _benchmark_inputs(missing: int, benchmark_type: str = "DYNAMIC_UNIVERSE_EQUAL_WEIGHT"):
    symbols = [f"S{index:02d}" for index in range(20)]
    if benchmark_type == "BTC_BUY_AND_HOLD":
        symbols = ["BTC-USDT"]
    if benchmark_type == "CASH":
        symbols = ["CASH"]
    available_symbols = symbols[:-missing] if missing else symbols
    prices = {
        symbol: (NOW + timedelta(hours=1), 1.0 if symbol == "CASH" else 100.0)
        for symbol in available_symbols
    }
    decision = _decision("decision")
    return benchmark_entry_records(
        decision=decision,
        benchmark_type=benchmark_type,
        expected_symbols=symbols,
        entry_prices=prices,
        entry_due=NOW + timedelta(hours=1),
        lock=_minimal_lock(),
    )


def test_dynamic_benchmark_all_twenty_symbols_has_full_coverage() -> None:
    benchmark, trades, coverage = _benchmark_inputs(0)
    assert benchmark["expected_symbol_count"] == 20
    assert benchmark["filled_symbol_count"] == 20
    assert benchmark["fill_coverage"] == 1.0
    assert benchmark["cash_residual"] == pytest.approx(0.0)
    assert len(trades) == 20
    assert coverage["status"] == "OPEN"


def test_dynamic_benchmark_one_missing_keeps_weight_as_cash_without_redistribution() -> None:
    benchmark, trades, coverage = _benchmark_inputs(1)
    assert benchmark["fill_coverage"] == pytest.approx(0.95)
    assert benchmark["invested_weight"] == pytest.approx(0.95)
    assert benchmark["cash_residual"] == pytest.approx(0.05)
    assert benchmark["invested_weight"] + benchmark["cash_residual"] == pytest.approx(1.0)
    filled = [row for row in trades if row["entry_price_available"]]
    missing = [row for row in trades if not row["entry_price_available"]]
    assert len(filled) == 19 and len(missing) == 1
    assert all(row["realized_weight"] == pytest.approx(0.05) for row in filled)
    assert missing[0]["realized_weight"] == 0.0
    assert missing[0]["net_return"] is None
    assert json.loads(coverage["missing_symbols"]) == ["S19"]


def test_dynamic_benchmark_two_missing_fails_data_quality() -> None:
    benchmark, _trades, _coverage = _benchmark_inputs(2)
    assert benchmark["fill_coverage"] == pytest.approx(0.90)
    metrics = _ready_metrics()
    metrics["minimum_benchmark_fill_coverage"] = benchmark["fill_coverage"]
    metrics["completed_independent_cycles"] = 1
    assert evaluate_forward_status(metrics=metrics, lock=_minimal_lock()) == "FAIL_DATA_QUALITY"


def test_missing_benchmark_is_not_silently_treated_as_zero_return() -> None:
    benchmark, trades, _coverage = _benchmark_inputs(1)
    missing = [row for row in trades if row["status"] == "MISSING_ENTRY"]
    assert missing and missing[0]["gross_return"] is None and missing[0]["net_return"] is None
    assert benchmark["gross_return"] is None and benchmark["net_return"] is None


def test_btc_missing_is_incomplete_and_cash_is_always_complete_at_entry() -> None:
    btc, _btc_trades, _ = _benchmark_inputs(1, "BTC_BUY_AND_HOLD")
    cash, cash_trades, _ = _benchmark_inputs(0, "CASH")
    assert btc["fill_coverage"] == 0.0
    assert btc["status"] == "INCOMPLETE_ENTRY"
    assert not btc["eligible_for_forward_evidence"]
    assert cash["fill_coverage"] == 1.0
    assert cash["status"] == "OPEN"
    assert cash_trades[0]["entry_cost_bps"] == 0.0


def test_closed_benchmark_cannot_be_modified() -> None:
    benchmark, _trades, _ = _benchmark_inputs(0, "CASH")
    benchmark.update({"status": "CLOSED", "cycle_complete": True, "actual_exit_ts": NOW + timedelta(hours=121), "gross_return": 0.0, "net_return": 0.0, "exit_market_data_snapshot_id": "exit"})
    existing = frame([benchmark], BENCHMARK_DECISION_SCHEMA)
    changed = dict(benchmark)
    changed["net_return"] = 0.1
    with pytest.raises(ValueError, match="terminal benchmark_id changed"):
        merge_state_rows(
            existing,
            frame([changed], BENCHMARK_DECISION_SCHEMA),
            schema=BENCHMARK_DECISION_SCHEMA,
            id_field="benchmark_id",
            immutable_fields=("benchmark_id",),
            terminal_statuses=frozenset({"CLOSED"}),
            sort_fields=("decision_ts",),
        )


def _closed_benchmarks(
    decision: dict[str, object], *, missing_btc: bool = False
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for benchmark_type, symbols in (
        ("BTC_BUY_AND_HOLD", ["BTC-USDT"]),
        ("DYNAMIC_UNIVERSE_EQUAL_WEIGHT", [f"S{index:02d}" for index in range(20)]),
        ("CASH", ["CASH"]),
    ):
        available = (
            {}
            if missing_btc and benchmark_type == "BTC_BUY_AND_HOLD"
            else {
                symbol: (
                    NOW + timedelta(hours=1),
                    1.0 if symbol == "CASH" else 100.0,
                )
                for symbol in symbols
            }
        )
        benchmark, _trades, _coverage = benchmark_entry_records(
            decision=decision,
            benchmark_type=benchmark_type,
            expected_symbols=symbols,
            entry_prices=available,
            entry_due=NOW + timedelta(hours=1),
            lock=_minimal_lock(),
        )
        if benchmark["status"] == "OPEN":
            benchmark.update(
                {
                    "actual_exit_ts": NOW + timedelta(hours=121),
                    "gross_return": 0.01,
                    "net_return": 0.007,
                    "exit_market_data_snapshot_id": "exit-snapshot",
                    "cycle_complete": True,
                    "status": "CLOSED",
                }
            )
        rows.append(benchmark)
    return frame(rows, BENCHMARK_DECISION_SCHEMA)


def test_all_three_closed_benchmarks_are_required_for_cycle_completion() -> None:
    decision = _decision("complete", selected=["BTC-USDT"])
    strategy_trade = _trade("trade", "complete")
    strategy_trade.update(
        {
            "entry_market_data_snapshot_id": "decision-snapshot",
            "exit_market_data_snapshot_id": "exit-snapshot",
        }
    )
    decisions, trades, benchmarks, completed, incomplete = finalize_cycles(
        decisions=frame([decision], DECISION_SCHEMA),
        trades=frame([strategy_trade], TRADE_SCHEMA),
        benchmarks=_closed_benchmarks(decision),
        observed_cutoff=NOW + timedelta(hours=121),
        lock=_minimal_lock(),
    )
    row = decisions.to_dicts()[0]
    assert row["btc_benchmark_cycle_complete"]
    assert row["universe_benchmark_cycle_complete"]
    assert row["cash_benchmark_cycle_complete"]
    assert row["all_benchmarks_complete"]
    assert row["cycle_eligible_for_forward_evidence"]
    assert completed == ["complete"] and incomplete == 0
    assert trades["cycle_eligible_for_forward_evidence"].all()
    assert benchmarks["eligible_for_forward_evidence"].all()


def test_missing_btc_benchmark_makes_cycle_ineligible() -> None:
    decision = _decision("missing-btc", selected=["BTC-USDT"])
    strategy_trade = _trade("trade", "missing-btc")
    strategy_trade.update(
        {
            "entry_market_data_snapshot_id": "decision-snapshot",
            "exit_market_data_snapshot_id": "exit-snapshot",
        }
    )
    decisions, _trades, _benchmarks, completed, incomplete = finalize_cycles(
        decisions=frame([decision], DECISION_SCHEMA),
        trades=frame([strategy_trade], TRADE_SCHEMA),
        benchmarks=_closed_benchmarks(decision, missing_btc=True),
        observed_cutoff=NOW + timedelta(hours=121),
        lock=_minimal_lock(),
    )
    row = decisions.to_dicts()[0]
    assert not row["btc_benchmark_cycle_complete"]
    assert not row["all_benchmarks_complete"]
    assert not row["cycle_eligible_for_forward_evidence"]
    assert completed == [] and incomplete == 1


def test_recovery_rows_are_disjoint_from_formal_counts_and_returns() -> None:
    decisions = frame(
        [
            _decision("formal", cycle_eligible=True, selected=["BTC-USDT"]),
            _decision("recovery", origin="RECOVERY_RECONSTRUCTION", eligible=False, cycle_eligible=True, coverage=0.1, selected=["BTC-USDT"]),
        ],
        DECISION_SCHEMA,
    )
    trades = frame(
        [
            _trade("formal-trade", "formal", cycle_eligible=True, net_return=-0.02),
            _trade("recovery-trade", "recovery", origin="RECOVERY_RECONSTRUCTION", eligible=False, cycle_eligible=True, net_return=10.0),
        ],
        TRADE_SCHEMA,
    )
    partitions = evidence_partitions(decisions, trades, frame([], SCHEDULE_EVENT_SCHEMA))
    assert partitions["formal_realtime_decision_count"] == 1
    assert partitions["recovery_decision_count"] == 1
    assert partitions["formal_realtime_trade_count"] == 1
    assert partitions["recovery_trade_count"] == 1
    assert partitions["formal_realtime_data_coverage"] == 1.0
    assert partitions["recovery_data_coverage"] == 0.1
    assert formal_cycle_period_returns(decisions, trades) == [("formal", pytest.approx(-0.02))]


def test_recovery_error_cannot_trigger_formal_data_quality_failure() -> None:
    lock = _minimal_lock()
    metrics = _ready_metrics()
    metrics["recovery_runner_errors"] = 99
    metrics["recovery_incomplete_decision_count"] = 99
    assert evaluate_forward_status(metrics=metrics, lock=lock) == "PAPER_REVIEW_READY"


def test_recovery_benchmark_return_cannot_enter_formal_excess() -> None:
    formal_decision = _decision("formal", cycle_eligible=True)
    recovery_decision = _decision(
        "recovery",
        origin="RECOVERY_RECONSTRUCTION",
        eligible=False,
        cycle_eligible=True,
    )
    formal_rows = _closed_benchmarks(formal_decision).to_dicts()
    recovery_rows = _closed_benchmarks(recovery_decision).to_dicts()
    for row in recovery_rows:
        row["net_return"] = 99.0
        row["eligible_for_forward_evidence"] = False
        row["decision_origin"] = "RECOVERY_RECONSTRUCTION"
    returns = formal_benchmark_period_returns(
        frame([formal_decision, recovery_decision], DECISION_SCHEMA),
        frame(formal_rows + recovery_rows, BENCHMARK_DECISION_SCHEMA),
        "BTC_BUY_AND_HOLD",
    )
    assert returns == [("formal", pytest.approx(0.007))]


def _schedule_events(completed_slots: int, expected_slots: int) -> pl.DataFrame:
    lock = _minimal_lock()
    rows = []
    cutoff = datetime(2026, 7, 19, 0, 5, tzinfo=UTC)
    slots = expected_timer_slots(cutoff, cutoff + timedelta(hours=expected_slots))
    for index, slot in enumerate(slots):
        rows.append(make_schedule_event(event_type="SCHEDULE_EXPECTED", expected_run_ts=slot, actual_start_ts=None, actual_finish_ts=None, run_mode="REALTIME", run_status="EXPECTED", exit_code=0, eligible=False, market_data_cutoff=None, error_code="", lock=lock))
        if index < completed_slots:
            rows.append(make_schedule_event(event_type="RUN_STARTED", expected_run_ts=slot, actual_start_ts=slot + timedelta(minutes=1), actual_finish_ts=None, run_mode="REALTIME", run_status="STARTED", exit_code=0, eligible=False, market_data_cutoff=None, error_code="", lock=lock))
            rows.append(make_schedule_event(event_type="RUN_COMPLETED", expected_run_ts=slot, actual_start_ts=slot + timedelta(minutes=1), actual_finish_ts=slot + timedelta(minutes=2), run_mode="REALTIME", run_status="COMPLETED", exit_code=0, eligible=True, market_data_cutoff=slot, error_code="", lock=lock))
        else:
            rows.append(make_schedule_event(event_type="RUN_MISSED", expected_run_ts=slot, actual_start_ts=None, actual_finish_ts=slot + timedelta(hours=1), run_mode="REALTIME", run_status="MISSED", exit_code=0, eligible=False, market_data_cutoff=None, error_code="NO_RUN", lock=lock))
    return append_hash_chain(frame([], SCHEDULE_EVENT_SCHEMA), frame(rows, SCHEDULE_EVENT_SCHEMA), schema=SCHEDULE_EVENT_SCHEMA, id_field="schedule_event_id")


def test_expected_hourly_schedule_and_coverage_are_correct() -> None:
    cutoff = datetime(2026, 7, 19, 0, 5, tzinfo=UTC)
    events = _schedule_events(19, 20)
    stats = schedule_statistics(events, cutoff=cutoff, through=cutoff + timedelta(hours=20))
    assert stats["expected_schedule_count"] == 20
    assert stats["eligible_realtime_schedule_count"] == 19
    assert stats["missed_schedule_count"] == 1
    assert stats["schedule_coverage"] == pytest.approx(0.95)
    assert stats["eligible_forward_days"] != pytest.approx(stats["forward_calendar_days"])


def test_persistent_pre_ten_minute_run_is_catchup_not_realtime() -> None:
    slot, persistent = timer_slot(datetime(2026, 7, 19, 4, 3, tzinfo=UTC))
    assert persistent is True
    assert slot == datetime(2026, 7, 19, 3, 10, tzinfo=UTC)


def test_late_schedule_event_is_ineligible_and_counted_separately() -> None:
    cutoff = datetime(2026, 7, 19, 0, 5, tzinfo=UTC)
    slot = datetime(2026, 7, 19, 1, 10, tzinfo=UTC)
    rows = [
        make_schedule_event(
            event_type="SCHEDULE_EXPECTED",
            expected_run_ts=slot,
            actual_start_ts=None,
            actual_finish_ts=None,
            run_mode="REALTIME",
            run_status="EXPECTED",
            exit_code=0,
            eligible=False,
            market_data_cutoff=None,
            error_code="",
            lock=_minimal_lock(),
        ),
        make_schedule_event(
            event_type="RUN_LATE",
            expected_run_ts=slot,
            actual_start_ts=slot + timedelta(hours=2),
            actual_finish_ts=None,
            run_mode="REALTIME",
            run_status="LATE",
            exit_code=0,
            eligible=False,
            market_data_cutoff=None,
            error_code="LATENCY_EXCEEDED",
            lock=_minimal_lock(),
        ),
    ]
    events = append_hash_chain(
        frame([], SCHEDULE_EVENT_SCHEMA),
        frame(rows, SCHEDULE_EVENT_SCHEMA),
        schema=SCHEDULE_EVENT_SCHEMA,
        id_field="schedule_event_id",
    )
    stats = schedule_statistics(events, cutoff=cutoff, through=slot)
    assert stats["late_schedule_count"] == 1
    assert stats["eligible_realtime_schedule_count"] == 0
    assert stats["schedule_coverage"] == 0.0


def test_strategy_schedule_keeps_original_120h_anchor() -> None:
    assert STRATEGY_SCHEDULE_ANCHOR == datetime(2026, 7, 18, 22, tzinfo=UTC)
    assert next_strategy_schedule(datetime(2026, 7, 19, tzinfo=UTC)) == datetime(2026, 7, 23, 22, tzinfo=UTC)


def _ready_metrics() -> dict[str, object]:
    return {
        "integrity_errors": [],
        "timer_installed": True,
        "timer_enabled": True,
        "timer_active": True,
        "health_check_passed": True,
        "cutoff_exists": True,
        "expected_schedule_count": 100,
        "schedule_coverage": 1.0,
        "longest_runner_gap_hours": 0.0,
        "unexplained_critical_schedule_gap_count": 0,
        "formal_realtime_decision_count": 6,
        "formal_realtime_data_coverage": 1.0,
        "formal_realtime_incomplete_decision_count": 0,
        "completed_independent_cycles": 6,
        "actual_symbol_trades": 12,
        "minimum_benchmark_fill_coverage": 1.0,
        "incomplete_formal_benchmark_cycle_count": 0,
        "eligible_forward_days": 30.0,
        "base_cost_net_return": 0.1,
        "excess_vs_btc": 0.02,
        "excess_vs_dynamic_universe": 0.01,
        "max_drawdown": -0.1,
        "maximum_single_symbol_contribution": 0.4,
    }


@pytest.mark.parametrize("field", ["timer_installed", "timer_enabled", "timer_active", "health_check_passed", "cutoff_exists"])
def test_not_deployed_state_is_explicit(field: str) -> None:
    metrics = _ready_metrics()
    metrics[field] = False
    assert evaluate_forward_status(metrics=metrics, lock=_minimal_lock()) == "INCONCLUSIVE_SYSTEM_NOT_DEPLOYED"


def test_deployed_but_sample_insufficient_is_inconclusive() -> None:
    metrics = _ready_metrics()
    metrics["eligible_forward_days"] = 1.0
    assert evaluate_forward_status(metrics=metrics, lock=_minimal_lock()) == "INCONCLUSIVE_SAMPLE_INSUFFICIENT"


def test_sample_sufficient_but_loss_is_fail_performance() -> None:
    metrics = _ready_metrics()
    metrics["base_cost_net_return"] = 0.0
    assert evaluate_forward_status(metrics=metrics, lock=_minimal_lock()) == "FAIL_PERFORMANCE"


def test_underperforming_benchmark_is_explicit_failure() -> None:
    metrics = _ready_metrics()
    metrics["excess_vs_btc"] = 0.0
    assert evaluate_forward_status(metrics=metrics, lock=_minimal_lock()) == "FAIL_BENCHMARK_UNDERPERFORMANCE"


def test_schedule_coverage_below_threshold_is_data_quality_failure() -> None:
    metrics = _ready_metrics()
    metrics["schedule_coverage"] = 0.94
    assert evaluate_forward_status(metrics=metrics, lock=_minimal_lock()) == "FAIL_DATA_QUALITY"


def test_all_preregistered_gates_can_reach_paper_review_ready() -> None:
    assert evaluate_forward_status(metrics=_ready_metrics(), lock=_minimal_lock()) == "PAPER_REVIEW_READY"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _synthetic_lock(repo: Path) -> dict[str, object]:
    entries = source_hash_entries(repo, ["logic.py"])
    lock: dict[str, object] = {
        "strategy_id": STRATEGY_ID,
        "strategy_version": "2.2.1",
        "hypothesis_type": "POST_HOC_HYPOTHESIS",
        "approval_state": "PAPER_ONLY",
        "parameters_locked": True,
        "factor": "low_vol_20d",
        "factor_formula": "-rolling_std(ret_1h, 480)",
        "factor_lookback_hours": 480,
        "btc_trend_filter": True,
        "btc_trend_lookback_hours": 1440,
        "btc_trend_description": "BTC 60-day SMA trend filter",
        "top_n": 3,
        "weighting": "score",
        "rebalance_hours": 120,
        "holding_hours": 120,
        "decision_delay_bars": 1,
        "strategy_type": "long_only_spot",
        "entry_cost_bps": 15.0,
        "exit_cost_bps": 15.0,
        "round_trip_cost_bps": 30.0,
        "max_decision_latency_seconds": 3600,
        "minimum_forward_days": 30,
        "minimum_completed_cycles": 6,
        "minimum_symbol_trades": 12,
        "minimum_data_coverage": 0.95,
        "minimum_benchmark_fill_coverage": 0.95,
        "minimum_schedule_coverage": 0.95,
        "maximum_single_symbol_contribution": 0.50,
        "maximum_drawdown": 0.30,
        "strategy_code_commit": _git(repo, "rev-parse", "HEAD"),
        "strategy_code_hash": composite_source_hash(entries),
        "reporting_code_hash": "f" * 64,
        "locked_source_files": entries,
        "service_unit_hash": sha256_file(repo / "service"),
        "timer_unit_hash": sha256_file(repo / "timer"),
        "runner_script_hash": sha256_file(repo / "scripts/run_forward_v221_realtime.sh"),
        "schedule_anchor": STRATEGY_SCHEDULE_ANCHOR.isoformat(),
        "schedule_timezone": "UTC",
        "runner_version": "quant_lab_low_vol_forward.v2.2.1",
        "execution_mode": "PAPER",
        "production_alpha": "FROZEN",
        "live_opening_enabled": False,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "created_at": NOW.isoformat(),
    }
    lock["sha256"] = parameter_lock_digest(lock)
    return lock


def _fake_git_repo(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "logic.py").write_text("VALUE = 1\n")
    (repo / "service").write_text("service\n")
    (repo / "timer").write_text("timer\n")
    runner = repo / "scripts/run_forward_v221_realtime.sh"
    runner.write_text("#!/bin/sh\nexit 0\n")
    runner.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.email", "audit@example.invalid")
    _git(repo, "config", "user.name", "Audit Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo, _synthetic_lock(repo)


def test_installed_unit_hash_mismatch_is_runner_integrity_failure(
    tmp_path: Path,
) -> None:
    repo, lock = _fake_git_repo(tmp_path)
    installed_service = tmp_path / "installed.service"
    installed_timer = tmp_path / "installed.timer"
    installed_service.write_text("tampered service\n")
    shutil.copy2(repo / "timer", installed_timer)
    identity = runtime_identity(
        repo=repo,
        lock=lock,
        installed_service=installed_service,
        installed_timer=installed_timer,
    )
    assert identity["status"] == "FAIL_RUNNER_INTEGRITY"
    assert "SERVICE_UNIT_HASH_MISMATCH" in identity["errors"]


def test_timer_not_active_cannot_generate_cutoff(tmp_path: Path) -> None:
    repo, lock = _fake_git_repo(tmp_path)
    root = tmp_path / "root"
    (root / "manifests").mkdir(parents=True)
    (root / "state").mkdir()
    (root / "manifests/parameter_lock_v221.json").write_text(json.dumps(lock))
    deployment = {
        "research_node_always_on": True,
        "timer_installed": True,
        "timer_enabled": True,
        "timer_active": False,
        "working_tree_clean": True,
        "timer_enabled_at": NOW.isoformat(),
    }
    (root / "manifests/systemd_deployment_v221.json").write_text(json.dumps(deployment))
    (root / "state/forward_v221_health_latest.json").write_text(json.dumps({"health_status": "PASS", "checked_at": NOW.isoformat()}))
    args = Namespace(root=root, repo=repo, created_at=(NOW + timedelta(minutes=1)).isoformat())
    with pytest.raises(RuntimeError, match="deployment gates"):
        create_cutoff(args)
    assert not (root / "manifests/forward_v221_cutoff.json").exists()


def test_cutoff_creation_is_idempotent_after_deployment(tmp_path: Path) -> None:
    repo, lock = _fake_git_repo(tmp_path)
    root = tmp_path / "root"
    (root / "manifests").mkdir(parents=True)
    (root / "state").mkdir()
    (root / "artifacts").mkdir()
    (root / "manifests/parameter_lock_v221.json").write_text(json.dumps(lock))
    deployment = {
        "research_node_always_on": True,
        "timer_installed": True,
        "timer_enabled": True,
        "timer_active": True,
        "working_tree_clean": True,
        "research_node": "audit-node",
        "timer_enabled_at": NOW.isoformat(),
        "next_timer_trigger": "2026-07-19 03:10:00 UTC",
        "service_unit_path": str(repo / "service"),
        "timer_unit_path": str(repo / "timer"),
        "service_unit_sha256": lock["service_unit_hash"],
        "timer_unit_sha256": lock["timer_unit_hash"],
        "runner_script_sha256": lock["runner_script_hash"],
        "parameter_lock_hash": lock["sha256"],
    }
    deployment["deployment_manifest_sha256"] = deployment_manifest_digest(deployment)
    (root / "manifests/systemd_deployment_v221.json").write_text(json.dumps(deployment))
    (root / "state/forward_v221_health_latest.json").write_text(
        json.dumps(
            {
                "health_status": "PASS",
                "checked_at": (NOW + timedelta(minutes=1)).isoformat(),
            }
        )
    )
    first = create_cutoff(
        Namespace(root=root, repo=repo, created_at=(NOW + timedelta(minutes=2)).isoformat())
    )
    second = create_cutoff(
        Namespace(root=root, repo=repo, created_at=(NOW + timedelta(hours=2)).isoformat())
    )
    assert first == second


def test_units_are_non_root_and_timer_is_persistent() -> None:
    repo = Path(__file__).resolve().parents[2]
    service = (repo / "deploy/systemd/quant-lab-forward-v221.service").read_text()
    timer = (repo / "deploy/systemd/quant-lab-forward-v221.timer").read_text()
    assert "User=quantlab" in service
    assert "Group=quantlab" in service
    assert "UMask=0077" in service
    assert "OnCalendar=*-*-* *:10:00" in timer
    assert "Persistent=true" in timer


def test_install_is_idempotent_and_uninstall_preserves_evidence(tmp_path: Path) -> None:
    source_repo = Path(__file__).resolve().parents[2]
    fake_repo = tmp_path / "install-repo"
    (fake_repo / "deploy/systemd").mkdir(parents=True)
    shutil.copy2(source_repo / "deploy/systemd/quant-lab-forward-v221.service", fake_repo / "deploy/systemd/quant-lab-forward-v221.service")
    shutil.copy2(source_repo / "deploy/systemd/quant-lab-forward-v221.timer", fake_repo / "deploy/systemd/quant-lab-forward-v221.timer")
    (fake_repo / "scripts").mkdir()
    shutil.copy2(source_repo / "scripts/run_forward_v221_realtime.sh", fake_repo / "scripts/run_forward_v221_realtime.sh")
    (fake_repo / "logic.py").write_text("VALUE = 1\n")
    shutil.copy2(fake_repo / "deploy/systemd/quant-lab-forward-v221.service", fake_repo / "service")
    shutil.copy2(fake_repo / "deploy/systemd/quant-lab-forward-v221.timer", fake_repo / "timer")
    subprocess.run(["git", "init", "-q", str(fake_repo)], check=True)
    _git(fake_repo, "config", "user.email", "audit@example.invalid")
    _git(fake_repo, "config", "user.name", "Audit Test")
    _git(fake_repo, "add", ".")
    _git(fake_repo, "commit", "-qm", "fixture")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(_synthetic_lock(fake_repo)))
    market = tmp_path / "market.parquet"
    market.write_bytes(b"fixture")
    dest = tmp_path / "dest"
    root = "/var/lib/quant-lab/forward_v221"
    env = {
        **os.environ,
        "QUANT_LAB_FORWARD_INSTALL_TEST_MODE": "1",
        "QUANT_LAB_FORWARD_DESTDIR": str(dest),
    }
    install = source_repo / "deploy/install_forward_v221_systemd.sh"
    command = [str(install), "--repo", str(fake_repo), "--root", root, "--python", sys.executable, "--market-bar", str(market), "--parameter-lock", str(lock)]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    marker = dest / root.lstrip("/") / "artifacts/evidence.marker"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("preserve")
    subprocess.run([str(source_repo / "deploy/uninstall_forward_v221_systemd.sh")], env=env, check=True, capture_output=True, text=True)
    assert marker.read_text() == "preserve"


def test_dirty_repo_is_rejected_by_installer(tmp_path: Path) -> None:
    source_repo = Path(__file__).resolve().parents[2]
    fake_repo = tmp_path / "dirty-repo"
    (fake_repo / "deploy/systemd").mkdir(parents=True)
    shutil.copy2(source_repo / "deploy/systemd/quant-lab-forward-v221.service", fake_repo / "deploy/systemd/quant-lab-forward-v221.service")
    shutil.copy2(source_repo / "deploy/systemd/quant-lab-forward-v221.timer", fake_repo / "deploy/systemd/quant-lab-forward-v221.timer")
    (fake_repo / "scripts").mkdir()
    shutil.copy2(source_repo / "scripts/run_forward_v221_realtime.sh", fake_repo / "scripts/run_forward_v221_realtime.sh")
    (fake_repo / "logic.py").write_text("VALUE = 1\n")
    shutil.copy2(fake_repo / "deploy/systemd/quant-lab-forward-v221.service", fake_repo / "service")
    shutil.copy2(fake_repo / "deploy/systemd/quant-lab-forward-v221.timer", fake_repo / "timer")
    subprocess.run(["git", "init", "-q", str(fake_repo)], check=True)
    _git(fake_repo, "config", "user.email", "audit@example.invalid")
    _git(fake_repo, "config", "user.name", "Audit Test")
    _git(fake_repo, "add", ".")
    _git(fake_repo, "commit", "-qm", "fixture")
    lock_payload = _synthetic_lock(fake_repo)
    (fake_repo / "scripts/run_forward_v221_realtime.sh").write_text("dirty\n")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(lock_payload))
    market = tmp_path / "market.parquet"
    market.write_bytes(b"fixture")
    env = {**os.environ, "QUANT_LAB_FORWARD_INSTALL_TEST_MODE": "1", "QUANT_LAB_FORWARD_DESTDIR": str(tmp_path / "dest")}
    result = subprocess.run([str(source_repo / "deploy/install_forward_v221_systemd.sh"), "--repo", str(fake_repo), "--root", "/var/lib/quant-lab/forward_v221", "--python", sys.executable, "--market-bar", str(market), "--parameter-lock", str(lock)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "dirty" in result.stderr.lower()


def test_realtime_wrapper_blocks_concurrent_invocation(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    mock_python = tmp_path / "python-mock"
    mock_python.write_text("#!/bin/sh\nsleep 2\n")
    mock_python.chmod(0o755)
    env = {
        **os.environ,
        "AUDIT_V221_ROOT": str(tmp_path / "state"),
        "QUANT_LAB_FORWARD_REPO": str(tmp_path / "repo"),
        "QUANT_LAB_FORWARD_PYTHON": str(mock_python),
        "QUANT_LAB_FORWARD_MARKET_BAR_PATH": str(tmp_path / "market"),
    }
    first = subprocess.Popen([str(repo / "scripts/run_forward_v221_realtime.sh")], env=env)
    try:
        time.sleep(0.2)
        second = subprocess.run([str(repo / "scripts/run_forward_v221_realtime.sh")], env=env, capture_output=True, text=True)
        assert second.returncode == 75
        assert "concurrent" in second.stderr
    finally:
        first.wait(timeout=5)
