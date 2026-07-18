from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v22 import (
    BENCHMARK_DECISION_SCHEMA,
    DECISION_SCHEMA,
    EVENT_SCHEMA,
    PAPER_STATUSES,
    STRATEGY_ID,
    TRADE_SCHEMA,
    append_hash_chain,
    build_realized_equity,
    classify_decision_origin,
    composite_source_hash,
    due_schedule_times,
    empty_frame,
    evaluate_forward_status,
    make_event,
    merge_benchmark_decision_states,
    parameter_lock_digest,
    runtime_identity,
    source_hash_entries,
    validate_event_chain,
    validate_parameter_lock,
)
from audit.scripts.stage_v22_forward import (
    BAR_SCHEMA,
    DataIncompleteError,
    RunnerIntegrityError,
    _decision_schedule_candidates,
    _record_exception_status,
    build_market_snapshot,
    create_decisions,
    create_entries,
    forward_market_fetch_start,
    run,
)
from audit.scripts.stage_v22_report import _dashboard, _executive, _final_decisions


def _lock(*, entries: list[dict[str, str]] | None = None, commit: str = "a" * 40) -> dict:
    locked = entries or [{"path": "runner.py", "sha256": "b" * 64}]
    lock = {
        "strategy_id": "low_vol_btc_60d_trend_top3_score_120h_v22",
        "strategy_version": "2.2",
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
        "universe": "dynamic_top20_v1",
        "maximum_single_position_weight": 0.5,
        "rebalance_hours": 120,
        "holding_hours": 120,
        "schedule_anchor_rule": "first_full_hour_strictly_after_cutoff_then_120h",
        "decision_delay_bars": 1,
        "strategy_type": "long_only_spot",
        "live_order_effect": "none",
        "automatic_promotion": False,
        "entry_cost_bps": 15.0,
        "exit_cost_bps": 15.0,
        "round_trip_cost_bps": 30.0,
        "max_decision_latency_seconds": 3600,
        "minimum_forward_days": 30,
        "minimum_completed_cycles": 6,
        "minimum_symbol_trades": 12,
        "minimum_data_coverage": 0.95,
        "maximum_single_symbol_contribution": 0.5,
        "maximum_drawdown": 0.3,
        "strategy_code_commit": commit,
        "strategy_code_hash": composite_source_hash(locked),
        "reporting_code_hash": "c" * 64,
        "locked_source_files": locked,
        "reporting_source_files": [{"path": "report.py", "sha256": "d" * 64}],
        "universe_definition_hash": "e" * 64,
        "cost_model_hash": "f" * 64,
        "runner_version": "quant_lab_low_vol_forward.v2.2",
        "created_at": "2026-07-19T00:00:00+00:00",
    }
    lock["sha256"] = parameter_lock_digest(lock)
    return lock


def _event(lock: dict, *, event_type: str = "DECISION_CREATED", suffix: str = "1") -> dict:
    return make_event(
        event_type=event_type,
        event_ts=datetime(2026, 7, 19, int(suffix), tzinfo=UTC),
        recorded_at=datetime(2026, 7, 19, int(suffix), 1, tzinfo=UTC),
        decision_id=f"decision-{suffix}",
        payload={"suffix": suffix},
        lock=lock,
    )


def _benchmark_row(status: str = "OPEN") -> dict:
    start = datetime(2026, 7, 19, tzinfo=UTC)
    return {
        "benchmark_id": "bench-1",
        "benchmark_type": "BTC_BUY_AND_HOLD",
        "decision_id": "decision-1",
        "decision_ts": start,
        "entry_ts": start + timedelta(hours=1),
        "scheduled_exit_ts": start + timedelta(hours=121),
        "actual_exit_ts": start + timedelta(hours=121) if status == "CLOSED" else None,
        "entry_price": 1.0,
        "exit_price": 1.1 if status == "CLOSED" else None,
        "weights": '{"BTC-USDT":1.0}',
        "symbols": '["BTC-USDT"]',
        "gross_return": 0.1 if status == "CLOSED" else None,
        "net_return": 0.097 if status == "CLOSED" else None,
        "data_snapshot_hash": "snapshot-1",
        "market_data_cutoff": start,
        "code_commit": "a" * 40,
        "parameter_lock_hash": "b" * 64,
        "strategy_code_hash": "c" * 64,
        "eligible_for_forward_evidence": True,
        "status": status,
    }


def _metrics(**overrides: object) -> dict:
    result = {
        "forward_days": 30.0,
        "completed_independent_cycles": 6,
        "actual_symbol_trades": 12,
        "data_coverage": 0.95,
        "base_cost_net_return": 0.10,
        "excess_vs_btc": 0.03,
        "excess_vs_dynamic_universe": 0.02,
        "max_drawdown": -0.10,
        "maximum_single_symbol_contribution": 0.40,
        "unhandled_market_gap_count": 0,
        "unexplained_missing_fill_count": 0,
        "system_error_count": 0,
        "data_completeness_unknown": False,
    }
    result.update(overrides)
    return result


def test_locked_strategy_uses_explicit_60_day_btc_semantics() -> None:
    lock = _lock()
    validate_parameter_lock(lock)
    assert STRATEGY_ID == "low_vol_btc_60d_trend_top3_score_120h_v22"
    assert lock["btc_trend_lookback_hours"] == 1440
    assert 1440 == 60 * 24
    assert "60-day" in lock["btc_trend_description"]


def test_any_parameter_change_changes_lock_hash() -> None:
    original = _lock()
    changed = dict(original)
    changed["top_n"] = 4
    changed["sha256"] = parameter_lock_digest(changed)
    assert changed["sha256"] != original["sha256"]
    with pytest.raises(ValueError, match="violates v2.2 hypothesis"):
        validate_parameter_lock(changed)


def test_realtime_and_recovery_origin_eligibility() -> None:
    scheduled = datetime(2026, 7, 19, tzinfo=UTC)
    timely = classify_decision_origin(
        mode="realtime",
        scheduled_run_ts=scheduled,
        recorded_at=scheduled + timedelta(minutes=30),
        max_latency=3600,
    )
    assert timely[:3] == ("REALTIME", False, True)
    late = classify_decision_origin(
        mode="realtime",
        scheduled_run_ts=scheduled,
        recorded_at=scheduled + timedelta(seconds=3601),
        max_latency=3600,
    )
    assert late[:3] == ("RECOVERY_RECONSTRUCTION", True, False)
    recovery = classify_decision_origin(
        mode="recovery",
        scheduled_run_ts=scheduled,
        recorded_at=scheduled + timedelta(minutes=1),
        max_latency=3600,
    )
    assert recovery[:3] == ("RECOVERY_RECONSTRUCTION", True, False)


def test_realtime_selects_only_latest_due_schedule_without_history_scan() -> None:
    cutoff = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    as_of = cutoff + timedelta(hours=500)
    due = due_schedule_times(cutoff, as_of)
    selected = _decision_schedule_candidates(
        mode="realtime",
        cutoff=cutoff,
        observed_cutoff=as_of,
        recorded_at=as_of,
        existing=empty_frame(DECISION_SCHEMA),
    )
    assert len(due) > 1
    assert selected == [due[-1]]


def test_realtime_waits_until_the_decision_bar_is_observable() -> None:
    cutoff = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    scheduled = due_schedule_times(cutoff, cutoff + timedelta(hours=1))[0]
    selected = _decision_schedule_candidates(
        mode="realtime",
        cutoff=cutoff,
        observed_cutoff=scheduled - timedelta(seconds=1),
        recorded_at=scheduled,
        existing=empty_frame(DECISION_SCHEMA),
    )
    assert selected == []


def test_fetch_window_includes_first_bar_that_closes_after_cutoff() -> None:
    cutoff = datetime(2026, 7, 19, 20, 46, 56, tzinfo=UTC)
    first_schedule = due_schedule_times(cutoff, cutoff + timedelta(hours=1))[0]
    feature_bar_open = first_schedule - timedelta(hours=1)
    assert forward_market_fetch_start(cutoff) < feature_bar_open
    assert feature_bar_open + timedelta(hours=1) > cutoff


def test_recovery_can_rebuild_but_never_create_formal_evidence() -> None:
    scheduled = datetime(2026, 7, 19, tzinfo=UTC)
    origin, late, eligible, _ = classify_decision_origin(
        mode="recovery",
        scheduled_run_ts=scheduled,
        recorded_at=scheduled,
        max_latency=3600,
    )
    assert origin == "RECOVERY_RECONSTRUCTION"
    assert late is True
    assert eligible is False


def test_reconstructed_trade_is_excluded_from_formal_equity() -> None:
    start = datetime(2026, 7, 19, tzinfo=UTC)
    decision = {
        "decision_id": "recovery-decision",
        "strategy_id": STRATEGY_ID,
        "strategy_version": "2.2",
        "decision_ts": start,
        "scheduled_run_ts": start,
        "recorded_at": start + timedelta(days=1),
        "decision_latency_seconds": 86400.0,
        "decision_origin": "RECOVERY_RECONSTRUCTION",
        "late_reconstructed": True,
        "eligible_for_forward_evidence": False,
        "feature_cutoff_ts": start,
        "observed_market_data_cutoff": start,
        "market_data_cutoff": start,
        "market_data_snapshot_id": "snapshot",
        "input_file_paths": "[]",
        "input_file_sha256": "{}",
        "btc_trend_state": "UP",
        "universe": '["BTC-USDT"]',
        "factor_scores": "[]",
        "ranked_symbols": '["BTC-USDT"]',
        "selected_symbols": '["BTC-USDT"]',
        "target_weights": '{"BTC-USDT":1.0}',
        "cash_weight": 0.0,
        "parameter_lock_hash": "a" * 64,
        "strategy_code_hash": "b" * 64,
        "git_commit": "c" * 40,
        "working_tree_clean": True,
        "status": "DECISION_CREATED",
    }
    trade = {
        "trade_id": "recovery-trade",
        "decision_id": "recovery-decision",
        "symbol": "BTC-USDT",
        "target_weight": 1.0,
        "entry_ts": start + timedelta(hours=1),
        "entry_price": 100.0,
        "scheduled_exit_ts": start + timedelta(hours=121),
        "actual_exit_ts": start + timedelta(hours=121),
        "exit_price": 200.0,
        "entry_cost_bps": 15.0,
        "exit_cost_bps": 15.0,
        "gross_return": 1.0,
        "net_return": 0.997,
        "eligible_for_forward_evidence": False,
        "market_data_snapshot_id": "snapshot",
        "parameter_lock_hash": "a" * 64,
        "strategy_code_hash": "b" * 64,
        "git_commit": "c" * 40,
        "status": "CLOSED",
    }
    equity = build_realized_equity(
        pl.DataFrame([decision], schema=DECISION_SCHEMA),
        pl.DataFrame([trade], schema=TRADE_SCHEMA),
        start,
    )
    assert equity.height == 1
    assert equity["strategy_equity"][0] == 1.0


def test_completed_realtime_cash_cycle_is_kept_in_formal_equity() -> None:
    start = datetime(2026, 7, 19, tzinfo=UTC)
    decision = {
        "decision_id": "cash-decision",
        "strategy_id": STRATEGY_ID,
        "strategy_version": "2.2",
        "decision_ts": start,
        "scheduled_run_ts": start,
        "recorded_at": start,
        "decision_latency_seconds": 0.0,
        "decision_origin": "REALTIME",
        "late_reconstructed": False,
        "eligible_for_forward_evidence": True,
        "feature_cutoff_ts": start,
        "observed_market_data_cutoff": start,
        "market_data_cutoff": start,
        "market_data_snapshot_id": "snapshot",
        "input_file_paths": "[]",
        "input_file_sha256": "{}",
        "btc_trend_state": "DOWN",
        "universe": "[]",
        "factor_scores": "[]",
        "ranked_symbols": "[]",
        "selected_symbols": "[]",
        "target_weights": "{}",
        "cash_weight": 1.0,
        "parameter_lock_hash": "a" * 64,
        "strategy_code_hash": "b" * 64,
        "git_commit": "c" * 40,
        "working_tree_clean": True,
        "status": "CASH",
    }
    equity = build_realized_equity(
        pl.DataFrame([decision], schema=DECISION_SCHEMA),
        empty_frame(TRADE_SCHEMA),
        start,
        completed_decision_ids=["cash-decision"],
    )
    assert equity.height == 2
    assert equity["strategy_equity"].to_list() == [1.0, 1.0]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"forward_days": 29.9}, "INCONCLUSIVE_SAMPLE_INSUFFICIENT"),
        ({"base_cost_net_return": 0.0}, "FAIL_PERFORMANCE"),
        ({"excess_vs_btc": 0.0}, "FAIL_BENCHMARK_UNDERPERFORMANCE"),
        ({"max_drawdown": -0.31}, "FAIL_RISK_OR_DRAWDOWN"),
        ({"maximum_single_symbol_contribution": 0.51}, "FAIL_CONCENTRATION"),
        ({"data_coverage": 0.94}, "FAIL_DATA_QUALITY"),
        ({}, "PAPER_REVIEW_READY"),
    ],
)
def test_explicit_forward_status_machine(overrides: dict, expected: str) -> None:
    status = evaluate_forward_status(metrics=_metrics(**overrides), lock=_lock())
    assert status == expected
    assert status in PAPER_STATUSES


def test_integrity_failure_takes_priority() -> None:
    assert (
        evaluate_forward_status(
            metrics=_metrics(), lock=_lock(), integrity_errors=["GIT_HEAD_MISMATCH"]
        )
        == "FAIL_RUNNER_INTEGRITY"
    )


def test_event_chain_detects_tampering_and_is_idempotent() -> None:
    lock = _lock()
    first = append_hash_chain(
        empty_frame(EVENT_SCHEMA), pl.DataFrame([_event(lock)], schema=EVENT_SCHEMA)
    )
    same = append_hash_chain(first, pl.DataFrame([_event(lock)], schema=EVENT_SCHEMA))
    assert same.equals(first)
    corrupted = same.with_columns(pl.lit("bad").alias("payload_hash"))
    with pytest.raises(ValueError, match="event chain hash mismatch"):
        validate_event_chain(corrupted)


def test_closed_benchmark_is_immutable() -> None:
    closed = pl.DataFrame([_benchmark_row("CLOSED")], schema=BENCHMARK_DECISION_SCHEMA)
    changed = _benchmark_row("CLOSED")
    changed["exit_price"] = 1.2
    with pytest.raises(ValueError, match="closed benchmark decision"):
        merge_benchmark_decision_states(
            closed, pl.DataFrame([changed], schema=BENCHMARK_DECISION_SCHEMA)
        )


def test_market_file_change_produces_new_snapshot_without_overwriting_old(tmp_path: Path) -> None:
    path = tmp_path / "bars.parquet"
    path.write_bytes(b"first")
    cutoff = datetime(2026, 7, 19, tzinfo=UTC)
    first = build_market_snapshot(
        input_paths=[path], market_cutoff=cutoff, recorded_at=cutoff
    )
    path.write_bytes(b"second")
    second = build_market_snapshot(
        input_paths=[path], market_cutoff=cutoff, recorded_at=cutoff + timedelta(hours=1)
    )
    assert first["snapshot_id"] != second["snapshot_id"]
    assert first["files"][0]["sha256"] != second["files"][0]["sha256"]


def _git_repo(tmp_path: Path) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "runner.py").write_text("print('locked')\n", encoding="utf-8")
    (repo / "report.py").write_text("print('report')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "audit@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Audit Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "locked"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    entries = source_hash_entries(repo, ["runner.py"])
    return repo, _lock(entries=entries, commit=commit)


def test_runtime_identity_rejects_head_dirty_and_locked_hash_mismatch(tmp_path: Path) -> None:
    repo, lock = _git_repo(tmp_path)
    assert runtime_identity(repo, lock)["ok"] is True
    (repo / "runner.py").write_text("print('changed')\n", encoding="utf-8")
    dirty = runtime_identity(repo, lock)
    assert "WORKING_TREE_DIRTY" in dirty["errors"]
    assert "LOCKED_SOURCE_HASH_MISMATCH" in dirty["errors"]
    subprocess.run(["git", "-C", str(repo), "add", "runner.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "changed"], check=True)
    changed_head = runtime_identity(repo, lock)
    assert "GIT_HEAD_MISMATCH" in changed_head["errors"]


def test_reporting_change_does_not_change_strategy_code_hash(tmp_path: Path) -> None:
    repo, lock = _git_repo(tmp_path)
    before = composite_source_hash(source_hash_entries(repo, ["runner.py"]))
    (repo / "report.py").write_text("print('report v2')\n", encoding="utf-8")
    after = composite_source_hash(source_hash_entries(repo, ["runner.py"]))
    assert before == after == lock["strategy_code_hash"]


def test_decision_and_entry_stages_are_separate() -> None:
    lock = _lock()
    cutoff = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    scheduled = due_schedule_times(cutoff, cutoff + timedelta(hours=1))[0]
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]
    start = scheduled - timedelta(hours=1500)
    rows = []
    for index in range(1501):
        timestamp = start + timedelta(hours=index)
        for offset, symbol in enumerate(symbols):
            price = 100.0 + offset * 10.0 + index * (0.02 + offset * 0.001)
            rows.append(
                {
                    "symbol": symbol,
                    "ts": timestamp,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1000.0,
                    "quote_volume": 100_000.0,
                }
            )
    bars = pl.DataFrame(rows, schema=BAR_SCHEMA)
    snapshot = {
        "snapshot_id": "snapshot-1",
        "files": [{"path": "/data/bars", "sha256": "a" * 64}],
    }
    decisions, _events = create_decisions(
        mode="realtime",
        lock=lock,
        cutoff=cutoff,
        observed_cutoff=scheduled,
        recorded_at=scheduled + timedelta(minutes=5),
        bars=bars,
        existing=empty_frame(DECISION_SCHEMA),
        snapshot=snapshot,
    )
    assert decisions.height == 1
    assert decisions["feature_cutoff_ts"][0] == scheduled
    assert decisions["observed_market_data_cutoff"][0] == scheduled
    entries, benchmarks, benchmark_trades, _ = create_entries(
        decisions=empty_frame(DECISION_SCHEMA),
        existing_trades=pl.DataFrame(schema={}),
        existing_benchmarks=empty_frame(BENCHMARK_DECISION_SCHEMA),
        existing_benchmark_trades=pl.DataFrame(schema={}),
        bars=bars,
        observed_cutoff=scheduled + timedelta(hours=1),
        recorded_at=scheduled + timedelta(hours=1),
        lock=lock,
    )
    assert entries.is_empty()
    assert benchmarks.is_empty()
    assert benchmark_trades.is_empty()
    entries, benchmarks, benchmark_trades, _ = create_entries(
        decisions=decisions,
        existing_trades=pl.DataFrame(schema={}),
        existing_benchmarks=empty_frame(BENCHMARK_DECISION_SCHEMA),
        existing_benchmark_trades=pl.DataFrame(schema={}),
        bars=bars,
        observed_cutoff=scheduled + timedelta(hours=1),
        recorded_at=scheduled + timedelta(hours=1),
        lock=lock,
    )
    assert entries.height == 3
    assert entries["entry_ts"].unique().to_list() == [scheduled + timedelta(hours=1)]
    assert set(benchmarks["benchmark_type"]) == {
        "BTC_BUY_AND_HOLD",
        "DYNAMIC_UNIVERSE_EQUAL_WEIGHT",
        "CASH",
    }
    assert benchmark_trades["entry_ts"].unique().to_list() == [
        scheduled + timedelta(hours=1)
    ]


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (OSError("disk unavailable"), "INCONCLUSIVE_SYSTEM_ERROR"),
        (DataIncompleteError("bars unavailable"), "INCONCLUSIVE_DATA_INCOMPLETE"),
        (RunnerIntegrityError("event chain invalid"), "FAIL_RUNNER_INTEGRITY"),
    ],
)
def test_runner_exception_writes_explicit_status_without_touching_evidence(
    tmp_path: Path, error: Exception, expected_status: str
) -> None:
    repo, lock = _git_repo(tmp_path)
    root = tmp_path / "audit"
    artifact = root / "artifacts"
    manifest = root / "manifests"
    artifact.mkdir(parents=True)
    manifest.mkdir(parents=True)
    (manifest / "parameter_lock_v22.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    (manifest / "forward_v22_cutoff.json").write_text(
        json.dumps(
            {
                "forward_v22_cutoff": "2026-07-19T00:00:00+00:00",
                "parameter_lock_hash": lock["sha256"],
            }
        ),
        encoding="utf-8",
    )
    evidence = artifact / "forward_v22_events.parquet"
    evidence.write_bytes(b"immutable evidence sentinel")
    prior = {
        "real_time_decision_count": 4,
        "reconstructed_decision_count": 2,
        "eligible_decision_count": 4,
        "actual_entry_count": 12,
    }
    (artifact / "forward_v22_status.json").write_text(
        json.dumps(prior), encoding="utf-8"
    )
    status = _record_exception_status(
        root=root,
        repo=repo,
        error=error,
        dry_run=False,
    )
    assert status["paper_status"] == expected_status
    assert status["real_time_decision_count"] == 4
    assert status["reconstructed_decision_count"] == 2
    assert status["evidence_mutated"] is False
    assert evidence.read_bytes() == b"immutable evidence sentinel"


def test_zero_sample_initialization_writes_all_v22_outputs_idempotently(tmp_path: Path) -> None:
    repo, lock = _git_repo(tmp_path)
    root = tmp_path / "audit-v22"
    v1_root = tmp_path / "audit-v1"
    (root / "manifests").mkdir(parents=True)
    (v1_root / "data/silver").mkdir(parents=True)
    cutoff = datetime(2026, 7, 19, 12, 30, tzinfo=UTC)
    (root / "manifests/parameter_lock_v22.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    (root / "manifests/forward_v22_cutoff.json").write_text(
        json.dumps(
            {
                "forward_v22_cutoff": cutoff.isoformat(),
                "parameter_lock_hash": lock["sha256"],
            }
        ),
        encoding="utf-8",
    )
    bars = pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "ts": cutoff - timedelta(hours=2),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1.0,
                "quote_volume": 100.0,
            }
        ],
        schema=BAR_SCHEMA,
    )
    bars.write_parquet(v1_root / "data/silver/bars_1h.parquet")
    first = run(
        root=root,
        v1_root=v1_root,
        repo=repo,
        mode="realtime",
        requested_as_of=cutoff,
        recorded_at=cutoff,
        resume=False,
        dry_run=False,
        no_fetch=True,
    )
    assert first["paper_status"] == "INCONCLUSIVE_SAMPLE_INSUFFICIENT"
    events_before = pl.read_parquet(root / "artifacts/forward_v22_events.parquet")
    second = run(
        root=root,
        v1_root=v1_root,
        repo=repo,
        mode="realtime",
        requested_as_of=cutoff,
        recorded_at=cutoff,
        resume=True,
        dry_run=False,
        no_fetch=True,
    )
    assert second == first
    assert pl.read_parquet(root / "artifacts/forward_v22_events.parquet").equals(events_before)
    backdated = run(
        root=root,
        v1_root=v1_root,
        repo=repo,
        mode="realtime",
        requested_as_of=cutoff,
        recorded_at=cutoff + timedelta(hours=2),
        resume=True,
        dry_run=False,
        no_fetch=True,
    )
    assert backdated["paper_status"] == "FAIL_RUNNER_INTEGRITY"
    assert "AS_OF_CANNOT_BACKDATE_REALTIME" in backdated["integrity_errors"]
    assert pl.read_parquet(root / "artifacts/forward_v22_events.parquet").equals(events_before)
    expected = {
        "forward_v22_decisions.parquet",
        "forward_v22_trades.parquet",
        "forward_v22_events.parquet",
        "forward_v22_equity.parquet",
        "forward_v22_benchmark_decisions.parquet",
        "forward_v22_benchmark_trades.parquet",
        "forward_v22_benchmark_events.parquet",
        "forward_v22_benchmark_equity.parquet",
        "forward_v22_performance.csv",
        "forward_v22_status.json",
    }
    assert expected <= {path.name for path in (root / "artifacts").iterdir()}


def test_dashboard_markdown_and_final_json_share_canonical_forward_fields() -> None:
    lock = _lock()
    status = {
        "paper_status": "INCONCLUSIVE_SAMPLE_INSUFFICIENT",
        "forward_v22_cutoff": "2026-07-19T00:00:00+00:00",
        "market_data_snapshot_id": "snapshot",
        "runner_integrity": "PASS",
        "git_head_check": True,
        "strategy_code_hash_check": True,
        "working_tree_clean": True,
        "parameter_lock_hash": lock["sha256"],
        "real_time_decision_count": 2,
        "reconstructed_decision_count": 3,
        "eligible_decision_count": 2,
        "ineligible_decision_count": 3,
        "cash_decision_count": 1,
        "actual_entry_count": 3,
        "completed_independent_cycles": 1,
        "actual_symbol_trades": 3,
        "forward_days": 5.0,
        "data_coverage": 1.0,
        "strategy_net_return": 0.01,
        "btc_benchmark_return": 0.02,
        "dynamic_universe_benchmark_return": 0.0,
        "strategy_excess_vs_btc": -0.01,
        "strategy_excess_vs_dynamic_universe": 0.01,
        "maximum_single_symbol_contribution": 0.4,
    }
    low_vol = {
        "raw_cross_sectional_ic": 0.0965,
        "within_symbol_permutation_null_ic": 0.077,
        "incremental_ic_above_symbol_persistent_component": 0.0195,
    }
    dashboard = _dashboard(status, lock, low_vol)
    executive = _executive(status, lock)
    final = _final_decisions(status, lock)
    candidate = final["low_vol_btc_60d_trend"]
    for value in (
        lock["strategy_id"],
        status["forward_v22_cutoff"],
        status["paper_status"],
        str(status["real_time_decision_count"]),
        str(status["reconstructed_decision_count"]),
    ):
        assert value in dashboard
        assert value in executive
    assert candidate["paper_status"] == status["paper_status"]
    assert candidate["real_time_decision_count"] == 2
    assert candidate["reconstructed_decision_count"] == 3
