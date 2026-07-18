"""Create the Audit v2.2 parameter lock and fresh forward cutoff.

Run this only after all strategy-affecting code is committed.  The command is
fail-closed on branch, Git cleanliness, v2.1 consistency, and source hashes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v22 import (  # noqa: E402
    ENTRY_COST_BPS,
    EXIT_COST_BPS,
    FACTOR_LOOKBACK_HOURS,
    MAX_DECISION_LATENCY_SECONDS,
    ROUND_TRIP_COST_BPS,
    RUNNER_VERSION,
    STRATEGY_ID,
    STRATEGY_VERSION,
    atomic_write_json,
    composite_source_hash,
    parameter_lock_digest,
    payload_digest,
    source_hash_entries,
    utc,
    validate_parameter_lock,
)
from audit.auditlib.universe import UNIVERSES  # noqa: E402

STRATEGY_SOURCE_PATHS = (
    "audit/auditlib/forward_v22.py",
    "audit/scripts/stage_v22_forward.py",
    "audit/auditlib/factors.py",
    "audit/auditlib/universe.py",
    "audit/auditlib/portfolio_backtest.py",
    "schemas/forward_parameter_lock_v22.schema.json",
    "scripts/run_forward_v22_realtime.sh",
)

REPORTING_SOURCE_PATHS = (
    "audit/scripts/stage_v22_consistency.py",
    "audit/scripts/stage_v22_report.py",
    "audit/scripts/stage_v22_bundle.py",
    "audit/scripts/stage_v22_test.py",
    "scripts/run_alpha_audit_v22.sh",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_hash_list(path: Path, entries: list[dict[str, str]], composite: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{item['sha256']}  {item['path']}\n" for item in entries)
    body += f"{composite}  COMPOSITE_SHA256\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable hash list already exists with other content: {path}")
        return
    path.write_text(body, encoding="utf-8")


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if _load(path) != payload:
            raise RuntimeError(f"immutable v2.2 manifest already exists with other content: {path}")
        return
    atomic_write_json(payload, path)


def initialize(*, root: Path, repo: Path, now: datetime | None = None) -> tuple[dict, dict]:
    consistency = _load(root / "artifacts/v21_consistency_check.json")
    if consistency.get("status") != "PASS":
        raise RuntimeError("v2.1 consistency gate has not passed")
    branch = _git(repo, "branch", "--show-current")
    if branch != "audit/alpha-validity-v2.2":
        raise RuntimeError(f"unexpected quant-lab branch: {branch}")
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise RuntimeError("v2.2 lock requires a clean quant-lab worktree")
    commit = _git(repo, "rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("strategy commit is not a full 40-character SHA")
    commit_time = utc(_git(repo, "show", "-s", "--format=%cI", commit))

    strategy_entries = source_hash_entries(repo, STRATEGY_SOURCE_PATHS)
    reporting_entries = source_hash_entries(repo, REPORTING_SOURCE_PATHS)
    strategy_hash = composite_source_hash(strategy_entries)
    reporting_hash = composite_source_hash(reporting_entries)
    hashes_completed_at = utc(now or datetime.now(UTC).replace(microsecond=0))

    universe_spec = vars(UNIVERSES["top20"])
    universe_hash = payload_digest(universe_spec)
    cost_model = {
        "entry_cost_bps": ENTRY_COST_BPS,
        "exit_cost_bps": EXIT_COST_BPS,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
    }
    cost_hash = payload_digest(cost_model)
    lock_created_at = utc(now or datetime.now(UTC).replace(microsecond=0))
    lock = {
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
        "rebalance_hours": 120,
        "holding_hours": 120,
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
        "strategy_code_commit": commit,
        "strategy_code_hash": strategy_hash,
        "reporting_code_hash": reporting_hash,
        "locked_source_files": strategy_entries,
        "reporting_source_files": reporting_entries,
        "universe_definition_hash": universe_hash,
        "cost_model_hash": cost_hash,
        "runner_version": RUNNER_VERSION,
        "created_at": lock_created_at.isoformat(),
    }
    lock["sha256"] = parameter_lock_digest(lock)
    validate_parameter_lock(lock)
    _write_once(root / "manifests/parameter_lock_v22.json", lock)
    _write_hash_list(
        root / "manifests/strategy_code_hashes_v22.txt", strategy_entries, strategy_hash
    )
    _write_hash_list(
        root / "manifests/reporting_code_hashes_v22.txt", reporting_entries, reporting_hash
    )

    cutoff_created_at = utc(now or datetime.now(UTC).replace(microsecond=0))
    cutoff_value = max(commit_time, hashes_completed_at, lock_created_at, cutoff_created_at)
    cutoff = {
        "schema_version": "quant_lab_forward_v22_cutoff.v1",
        "forward_v22_cutoff": cutoff_value.isoformat(),
        "strictly_after_cutoff": True,
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "strategy_code_commit": commit,
        "strategy_code_commit_time": commit_time.isoformat(),
        "strategy_code_hash": strategy_hash,
        "locked_source_hashes_completed_at": hashes_completed_at.isoformat(),
        "parameter_lock_created_at": lock_created_at.isoformat(),
        "cutoff_manifest_created_at": cutoff_created_at.isoformat(),
        "parameter_lock_hash": lock["sha256"],
        "v21_forward_status": "SUPERSEDED_BY_V22",
        "v21_forward_cutoff": consistency["forward_cutoff"],
        "v21_formal_decision_count": consistency["formal_decision_count"],
        "v21_formal_trade_count": consistency["formal_trade_count"],
        "v21_bundle_sha256": consistency["bundle_sha256"],
    }
    _write_once(root / "manifests/forward_v22_cutoff.json", cutoff)
    _write_once(
        root / "artifacts/v21_forward_supersession.json",
        {
            "schema_version": "quant_lab_forward_supersession.v1",
            "source": "Audit v2.1 Forward",
            "status": "SUPERSEDED_BY_V22",
            "preserved": True,
            "may_merge_with_v22": False,
            "formal_decision_count": consistency["formal_decision_count"],
            "formal_trade_count": consistency["formal_trade_count"],
            "replacement_strategy_id": STRATEGY_ID,
            "replacement_parameter_lock_hash": lock["sha256"],
            "replacement_cutoff": cutoff_value.isoformat(),
        },
    )
    return lock, cutoff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--now", help="Test-only UTC timestamp; omit for the real lock")
    args = parser.parse_args()
    lock, cutoff = initialize(
        root=args.root.resolve(),
        repo=args.repo.resolve(),
        now=utc(args.now) if args.now else None,
    )
    print(f"parameter_lock_hash={lock['sha256']}")
    print(f"strategy_code_commit={lock['strategy_code_commit']}")
    print(f"strategy_code_hash={lock['strategy_code_hash']}")
    print(f"forward_v22_cutoff={cutoff['forward_v22_cutoff']}")


if __name__ == "__main__":
    main()
