# ruff: noqa: E501
"""Record deployment identity, health, and the deployment-gated v2.2.1 cutoff."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v221 import (  # noqa: E402
    RUNTIME_HEALTH_SCHEMA,
    STRATEGY_ID,
    STRATEGY_SCHEDULE_ANCHOR,
    STRATEGY_VERSION,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    deployment_manifest_digest,
    empty_frame,
    next_strategy_schedule,
    payload_digest,
    runtime_identity,
    sha256_file,
    utc,
    validate_parameter_lock,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    repo = args.repo.resolve()
    lock = _load(root / "manifests/parameter_lock_v221.json")
    validate_parameter_lock(lock)
    service = args.installed_service.resolve()
    timer = args.installed_timer.resolve()
    if not service.is_file() or not timer.is_file():
        raise RuntimeError("installed service/timer unit is missing")
    manifest: dict[str, Any] = {
        "schema_version": "quant_lab_forward_v221_deployment.v1",
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "research_node": args.node,
        "research_node_always_on": bool(args.node_always_on),
        "repo_path": str(repo),
        "repo_head": _git(repo, "rev-parse", "HEAD"),
        "working_tree_clean": not bool(_git(repo, "status", "--porcelain")),
        "parameter_lock_hash": lock["sha256"],
        "strategy_code_hash": lock["strategy_code_hash"],
        "service_unit_path": str(service),
        "service_unit_sha256": sha256_file(service),
        "timer_unit_path": str(timer),
        "timer_unit_sha256": sha256_file(timer),
        "runner_script_path": str(repo / "scripts/run_forward_v221_realtime.sh"),
        "runner_script_sha256": sha256_file(
            repo / "scripts/run_forward_v221_realtime.sh"
        ),
        "timer_installed": bool(args.timer_installed),
        "timer_enabled": bool(args.timer_enabled),
        "timer_active": bool(args.timer_active),
        "timer_enabled_at": utc(args.timer_enabled_at).isoformat(),
        "next_timer_trigger": args.next_trigger,
        "installed_at": utc(args.installed_at).isoformat(),
        "execution_user": "quantlab",
        "execution_group": "quantlab",
        "execution_mode": "PAPER",
        "production_alpha": "FROZEN",
        "live_opening_enabled": False,
        "live_order_effect": "none",
        "v5_production_modified": False,
    }
    manifest["deployment_manifest_sha256"] = deployment_manifest_digest(manifest)
    identity = runtime_identity(
        repo=repo,
        lock=lock,
        installed_service=service,
        installed_timer=timer,
        deployment_manifest=manifest,
    )
    if not identity["ok"]:
        raise RuntimeError(f"deployment identity failed: {identity['errors']}")
    destination = root / "manifests/systemd_deployment_v221.json"
    if destination.exists():
        prior = _load(destination)
        immutable = {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "installed_at",
                "next_timer_trigger",
                "deployment_manifest_sha256",
            }
        }
        comparable = {
            key: value
            for key, value in prior.items()
            if key
            not in {
                "installed_at",
                "next_timer_trigger",
                "deployment_manifest_sha256",
            }
        }
        if immutable != comparable:
            raise RuntimeError("immutable deployment identity changed")
        return prior
    atomic_write_json(manifest, destination)
    return manifest


def record_health(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    repo = args.repo.resolve()
    lock = _load(root / "manifests/parameter_lock_v221.json")
    deployment = _load(root / "manifests/systemd_deployment_v221.json")
    cutoff_path = root / "manifests/forward_v221_cutoff.json"
    cutoff = _load(cutoff_path) if cutoff_path.exists() else None
    checked_at = utc(args.checked_at)
    identity = runtime_identity(
        repo=repo,
        lock=lock,
        installed_service=args.installed_service.resolve(),
        installed_timer=args.installed_timer.resolve(),
        deployment_manifest=deployment,
        cutoff_manifest=cutoff,
    )
    healthy = all(
        (
            identity["ok"],
            bool(args.timer_installed),
            bool(args.timer_enabled),
            bool(args.timer_active),
            args.service_last_result in {"success", "inactive", "never-run"},
            float(args.market_data_staleness_seconds) <= 7200.0,
            int(args.disk_free_bytes) >= 2_000_000_000,
        )
    )
    row = {
        "checked_at": checked_at,
        "research_node": deployment["research_node"],
        "timer_installed": bool(args.timer_installed),
        "timer_enabled": bool(args.timer_enabled),
        "timer_active": bool(args.timer_active),
        "service_last_result": args.service_last_result,
        "next_trigger": args.next_trigger,
        "last_successful_run": utc(args.last_successful_run)
        if args.last_successful_run
        else None,
        "last_failed_run": utc(args.last_failed_run) if args.last_failed_run else None,
        "working_tree_clean": bool(identity["working_tree_clean"]),
        "git_head_match": identity["current_head"] == identity["expected_head"],
        "code_hash_match": identity["strategy_code_hash"]
        == identity["expected_strategy_code_hash"],
        "parameter_lock_match": not any(
            "PARAMETER_LOCK" in error for error in identity["errors"]
        ),
        "unit_hash_match": bool(
            identity["service_unit_hash_match"]
            and identity["timer_unit_hash_match"]
            and identity["runner_script_hash_match"]
        ),
        "disk_free_bytes": int(args.disk_free_bytes),
        "market_data_staleness_seconds": float(args.market_data_staleness_seconds),
        "health_status": "PASS" if healthy else "FAIL",
    }
    path = root / "artifacts/forward_v221_runtime_health.parquet"
    existing = pl.read_parquet(path) if path.exists() else empty_frame(RUNTIME_HEALTH_SCHEMA)
    update = pl.DataFrame([row], schema=RUNTIME_HEALTH_SCHEMA)
    combined = pl.concat([existing, update], how="vertical_relaxed").unique(
        subset=["checked_at"], keep="last"
    ).sort("checked_at")
    atomic_write_parquet(combined, path)
    latest = {
        **{key: (utc(value).isoformat() if isinstance(value, datetime) else value) for key, value in row.items()},
        "cutoff_exists": cutoff is not None,
        "identity_errors": identity["errors"],
    }
    atomic_write_json(latest, root / "state/forward_v221_health_latest.json")
    if not healthy:
        raise RuntimeError(f"forward health check failed: {identity['errors']}")
    return latest


def create_cutoff(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    repo = args.repo.resolve()
    lock = _load(root / "manifests/parameter_lock_v221.json")
    deployment = _load(root / "manifests/systemd_deployment_v221.json")
    health = _load(root / "state/forward_v221_health_latest.json")
    validate_parameter_lock(lock)
    if health.get("health_status") != "PASS":
        raise RuntimeError("health must pass before cutoff creation")
    if not all(
        (
            deployment.get("research_node_always_on"),
            deployment.get("timer_installed"),
            deployment.get("timer_enabled"),
            deployment.get("timer_active"),
            deployment.get("working_tree_clean"),
        )
    ):
        raise RuntimeError("deployment gates are incomplete")
    identity = runtime_identity(
        repo=repo,
        lock=lock,
        installed_service=Path(deployment["service_unit_path"]),
        installed_timer=Path(deployment["timer_unit_path"]),
        deployment_manifest=deployment,
    )
    if not identity["ok"]:
        raise RuntimeError(f"identity failed before cutoff: {identity['errors']}")
    path = root / "manifests/forward_v221_cutoff.json"
    if path.exists():
        prior = _load(path)
        immutable_bindings = {
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
            "strategy_code_commit": lock["strategy_code_commit"],
            "strategy_code_hash": lock["strategy_code_hash"],
            "parameter_lock_hash": lock["sha256"],
            "deployment_manifest_sha256": deployment["deployment_manifest_sha256"],
            "schedule_anchor": STRATEGY_SCHEDULE_ANCHOR.isoformat(),
        }
        mismatches = {
            key: {"expected": value, "actual": prior.get(key)}
            for key, value in immutable_bindings.items()
            if prior.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"immutable cutoff binding changed: {mismatches}")
        expected_hash = payload_digest(
            {key: value for key, value in prior.items() if key != "sha256"}
        )
        if prior.get("sha256") != expected_hash:
            raise RuntimeError("immutable cutoff digest changed")
        return prior
    commit_time = utc(_git(repo, "show", "-s", "--format=%cI", lock["strategy_code_commit"]))
    gates = [
        commit_time,
        utc(lock["created_at"]),
        utc(deployment["timer_enabled_at"]),
        utc(health["checked_at"]),
        utc(args.created_at),
    ]
    cutoff = max(gates).replace(microsecond=0)
    manifest: dict[str, Any] = {
        "schema_version": "quant_lab_forward_v221_cutoff.v1",
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "forward_v221_cutoff": cutoff.isoformat(),
        "strictly_after_cutoff": True,
        "strategy_code_commit": lock["strategy_code_commit"],
        "strategy_code_commit_time": commit_time.isoformat(),
        "strategy_code_hash": lock["strategy_code_hash"],
        "parameter_lock_hash": lock["sha256"],
        "parameter_lock_created_at": lock["created_at"],
        "locked_source_hashes_completed_at": lock["created_at"],
        "systemd_timer_enabled_at": deployment["timer_enabled_at"],
        "health_check_passed_at": health["checked_at"],
        "deployment_manifest_sha256": deployment["deployment_manifest_sha256"],
        "service_unit_sha256": deployment["service_unit_sha256"],
        "timer_unit_sha256": deployment["timer_unit_sha256"],
        "runner_script_sha256": deployment["runner_script_sha256"],
        "schedule_anchor": STRATEGY_SCHEDULE_ANCHOR.isoformat(),
        "schedule_timezone": "UTC",
        "next_legal_strategy_decision": next_strategy_schedule(cutoff).isoformat(),
        "v22_status": "SUPERSEDED_BY_V221",
        "v22_formal_decisions_migrated": 0,
        "v22_recovery_decisions_migrated": 0,
        "created_at": cutoff.isoformat(),
    }
    manifest["sha256"] = payload_digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    atomic_write_json(manifest, path)
    status_path = root / "artifacts/forward_v221_status.json"
    status = (
        _load(status_path)
        if status_path.exists()
        else {
            "schema_version": "quant_lab_forward_v221_status.v1",
            "audit_version": "v2.2.1",
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
            "formal_realtime_decision_count": 0,
            "formal_realtime_trade_count": 0,
            "recovery_decision_count": 0,
            "recovery_trade_count": 0,
            "expected_schedule_count": 0,
            "completed_schedule_count": 0,
            "missed_schedule_count": 0,
            "late_schedule_count": 0,
            "schedule_coverage": 0.0,
            "forward_calendar_days": 0.0,
            "eligible_forward_days": 0.0,
            "actual_entry_count": 0,
            "completed_independent_cycles": 0,
            "actual_symbol_trades": 0,
            "strategy_net_return": 0.0,
            "btc_benchmark_return": 0.0,
            "dynamic_universe_benchmark_return": 0.0,
            "production_alpha": "FROZEN",
            "live": "NOT_ALLOWED",
        }
    )
    status.update(
        {
            "forward_start_status": "FORWARD_V221_READY",
            "paper_status": "INCONCLUSIVE_SAMPLE_INSUFFICIENT",
            "forward_v221_cutoff": cutoff.isoformat(),
            "next_legal_strategy_decision": manifest["next_legal_strategy_decision"],
            "research_node": deployment["research_node"],
            "timer_installed": True,
            "timer_enabled": True,
            "timer_active": True,
            "next_timer_trigger": deployment["next_timer_trigger"],
            "deployment_manifest_sha256": deployment["deployment_manifest_sha256"],
        }
    )
    atomic_write_json(status, status_path)
    atomic_write_csv(
        pl.DataFrame([status], infer_schema_length=None),
        root / "artifacts/forward_v221_performance.csv",
    )
    return manifest


def _bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "active", "enabled"}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get("AUDIT_V221_ROOT", "/var/lib/quant-lab/forward_v221")
        ),
    )
    common.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    common.add_argument(
        "--installed-service",
        type=Path,
        default=Path("/etc/systemd/system/quant-lab-forward-v221.service"),
    )
    common.add_argument(
        "--installed-timer",
        type=Path,
        default=Path("/etc/systemd/system/quant-lab-forward-v221.timer"),
    )

    manifest_parser = sub.add_parser("manifest", parents=[common])
    manifest_parser.add_argument("--node", required=True)
    manifest_parser.add_argument("--node-always-on", type=_bool, required=True)
    manifest_parser.add_argument("--timer-installed", type=_bool, required=True)
    manifest_parser.add_argument("--timer-enabled", type=_bool, required=True)
    manifest_parser.add_argument("--timer-active", type=_bool, required=True)
    manifest_parser.add_argument("--timer-enabled-at", required=True)
    manifest_parser.add_argument("--next-trigger", required=True)
    manifest_parser.add_argument("--installed-at", required=True)

    health_parser = sub.add_parser("health", parents=[common])
    health_parser.add_argument("--timer-installed", type=_bool, required=True)
    health_parser.add_argument("--timer-enabled", type=_bool, required=True)
    health_parser.add_argument("--timer-active", type=_bool, required=True)
    health_parser.add_argument("--service-last-result", required=True)
    health_parser.add_argument("--next-trigger", required=True)
    health_parser.add_argument("--last-successful-run")
    health_parser.add_argument("--last-failed-run")
    health_parser.add_argument("--disk-free-bytes", type=int, required=True)
    health_parser.add_argument("--market-data-staleness-seconds", type=float, required=True)
    health_parser.add_argument("--checked-at", required=True)

    cutoff_parser = sub.add_parser("cutoff", parents=[common])
    cutoff_parser.add_argument("--created-at", required=True)

    args = parser.parse_args()
    if args.command == "manifest":
        payload = create_manifest(args)
        print(f"deployment_manifest_sha256={payload['deployment_manifest_sha256']}")
    elif args.command == "health":
        payload = record_health(args)
        print(f"health_status={payload['health_status']}")
    else:
        payload = create_cutoff(args)
        print(f"forward_v221_cutoff={payload['forward_v221_cutoff']}")
        print(f"next_legal_strategy_decision={payload['next_legal_strategy_decision']}")


if __name__ == "__main__":
    main()
