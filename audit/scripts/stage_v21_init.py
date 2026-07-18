"""Create the immutable Audit v2.1 forward cutoff and parameter lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v21 import (  # noqa: E402
    ENTRY_PRICE_RULE,
    EXIT_PRICE_RULE,
    RUNNER_VERSION,
    STRATEGY_ID,
    atomic_write_json,
    parameter_lock_digest,
    utc,
    validate_new_cutoff,
    validate_parameter_lock,
)
from audit.auditlib.universe import UNIVERSES  # noqa: E402

LEGACY_FILES = (
    "forward_paper_decisions.parquet",
    "forward_paper_positions.parquet",
    "forward_paper_performance.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _load_v2_manifest(v2_dir: Path) -> dict:
    path = v2_dir / "artifacts/run_manifest_v2.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("audit_version") != "v2":
        raise ValueError("input manifest is not Audit v2")
    return payload


def _write_once(path: Path, payload: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable manifest already exists with other content: {path}")
        return
    atomic_write_json(payload, path)


def initialize(
    *,
    root: Path,
    v2_dir: Path,
    repo: Path,
    cutoff: datetime | None = None,
) -> tuple[dict, dict]:
    manifest = _load_v2_manifest(v2_dir)
    branch = _git(repo, "branch", "--show-current")
    if branch != "audit/alpha-validity-v2.1":
        raise RuntimeError(f"unexpected branch for v2.1 initialization: {branch}")
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise RuntimeError(
            "parameter lock requires a clean worktree so code_commit binds all runner code"
        )
    commit = _git(repo, "rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("quant-lab HEAD is not a full commit SHA")
    fix_completed_at = utc(_git(repo, "show", "-s", "--format=%cI", commit))
    created_at = utc(cutoff or datetime.now(UTC).replace(microsecond=0))
    validate_new_cutoff(created_at, fix_completed_at)

    v1_snapshot = str(manifest["data"]["v1_snapshot_id"])
    universe_spec = vars(UNIVERSES["top20"])
    universe_hash = hashlib.sha256(
        json.dumps(universe_spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cost_model = {
        "entry_fee_bps": 10.0,
        "entry_slippage_bps": 5.0,
        "exit_fee_bps": 10.0,
        "exit_slippage_bps": 5.0,
        "round_trip_cost_bps": 30.0,
    }
    cost_hash = hashlib.sha256(
        json.dumps(cost_model, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    snapshot_digest = hashlib.sha256(
        f"{v1_snapshot}|{created_at.isoformat()}|{commit}".encode()
    ).hexdigest()
    data_snapshot_id = f"forward_v21_{snapshot_digest[:20]}"
    lock = {
        "strategy_id": STRATEGY_ID,
        "hypothesis_type": "POST_HOC_HYPOTHESIS",
        "parameters_locked": True,
        "factor": "low_vol_20d",
        "factor_formula": "-rolling_std(ret_1h,480)",
        "factor_lookback_hours": 480,
        "universe": "top20",
        "universe_definition_hash": universe_hash,
        "top_n": 3,
        "weighting": "score",
        "max_single_weight": 0.5,
        "rebalance_hours": 120,
        "holding_hours": 120,
        "rebalance_anchor_rule": "first_full_hour_strictly_after_cutoff",
        "btc_trend_filter": True,
        "btc_trend_lookback": "1440h",
        "btc_trend_lookback_hours": 1440,
        "btc_trend_formula": "BTC-USDT close >= trailing_1440h_mean(close)",
        "decision_delay_bars": 1,
        "entry_price_rule": ENTRY_PRICE_RULE,
        "exit_price_rule": EXIT_PRICE_RULE,
        "strategy_type": "long_only_spot",
        "round_trip_cost_bps": 30.0,
        "cost_model": cost_model,
        "cost_model_hash": cost_hash,
        "maximum_drawdown_review_limit": 0.30,
        "maximum_symbol_contribution_share_review_limit": 0.50,
        "data_snapshot_id": data_snapshot_id,
        "v1_snapshot_id": v1_snapshot,
        "code_commit": commit,
        "runner_version": RUNNER_VERSION,
        "created_at": created_at.isoformat(),
    }
    lock["sha256"] = parameter_lock_digest(lock)
    validate_parameter_lock(lock)
    cutoff_manifest = {
        "schema_version": "quant_lab_forward_v21_cutoff.v1",
        "forward_v21_start_cutoff": created_at.isoformat(),
        "runner_fix_completed_at": fix_completed_at.isoformat(),
        "strictly_after_cutoff": True,
        "code_commit": commit,
        "parameter_lock_hash": lock["sha256"],
        "data_snapshot_id": data_snapshot_id,
        "cost_model_hash": cost_hash,
        "runner_version": RUNNER_VERSION,
        "legacy_v2_status": "INVALIDATED_BY_RUNNER_BUG",
        "v1_data_cutoff": manifest["data"]["v1_cutoff"],
        "legacy_v2_cutoff": manifest["data"]["v2_cutoff"],
    }

    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    _write_once(root / "manifests/parameter_lock_v21.json", lock)
    _write_once(root / "manifests/forward_v21_cutoff.json", cutoff_manifest)

    legacy_hashes: dict[str, str] = {}
    for name in LEGACY_FILES:
        source = v2_dir / "artifacts" / name
        if not source.exists():
            raise FileNotFoundError(source)
        destination = root / "artifacts" / f"legacy_v2_invalidated_{name}"
        source_hash = _sha256(source)
        if destination.exists() and _sha256(destination) != source_hash:
            raise RuntimeError(f"legacy preservation copy changed: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        legacy_hashes[name] = source_hash
    _write_once(
        root / "artifacts/legacy_forward_v2_status.json",
        {
            "status": "INVALIDATED_BY_RUNNER_BUG",
            "preserved": True,
            "may_merge_with_v21": False,
            "reason": (
                "v2 marked old decisions at latest price, summed returns, charged only "
                "one side of cost, and did not compute benchmarks"
            ),
            "source_manifest_sha256": _sha256(
                v2_dir / "artifacts/run_manifest_v2.json"
            ),
            "files": legacy_hashes,
            "invalidated_at": created_at.isoformat(),
            "replacement_parameter_lock_hash": lock["sha256"],
        },
    )
    return lock, cutoff_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cutoff")
    args = parser.parse_args()
    cutoff = utc(args.cutoff) if args.cutoff else None
    lock, manifest = initialize(
        root=args.root.resolve(),
        v2_dir=args.v2_dir.resolve(),
        repo=args.repo.resolve(),
        cutoff=cutoff,
    )
    print(f"parameter_lock_hash={lock['sha256']}")
    print(f"forward_v21_start_cutoff={manifest['forward_v21_start_cutoff']}")


if __name__ == "__main__":
    main()
