"""Validate the immutable Audit v2.2 bundle before starting v2.2.1."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v221 import atomic_write_json, utc  # noqa: E402

EXPECTED_BUNDLE_SHA256 = "cf2d7514c68f8e38b15e609003d32b1d43e044c5ca893838db72bfc961c9ea9d"
REQUIRED = {
    "reports/audit_dashboard_v22.html",
    "reports/executive_summary_v22.md",
    "reports/realtime_forward_integrity.md",
    "reports/forward_status_machine.md",
    "reports/immutable_benchmark_design.md",
    "reports/code_identity_lock.md",
    "artifacts/final_decisions_v22.json",
    "artifacts/forward_v22_decisions.parquet",
    "artifacts/forward_v22_trades.parquet",
    "artifacts/forward_v22_events.parquet",
    "artifacts/forward_v22_equity.parquet",
    "artifacts/forward_v22_benchmark_decisions.parquet",
    "artifacts/forward_v22_benchmark_trades.parquet",
    "artifacts/forward_v22_benchmark_events.parquet",
    "artifacts/forward_v22_benchmark_equity.parquet",
    "artifacts/forward_v22_status.json",
    "manifests/parameter_lock_v22.json",
    "manifests/forward_v22_cutoff.json",
    "manifests/strategy_code_hashes_v22.txt",
    "manifests/market_data_snapshot_v22.json",
    "artifacts/run_manifest_v22.json",
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _json(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("expected JSON object")
    return value


def validate(bundle: Path, repo: Path, checked_at: datetime) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    bundle_hash = _sha256(bundle.read_bytes())
    if bundle_hash != EXPECTED_BUNDLE_SHA256:
        errors.append("V22_BUNDLE_SHA256_MISMATCH")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        name_set = set(names)
        if archive.testzip() is not None:
            errors.append("V22_ZIP_CRC_FAILURE")
        unsafe = [
            name
            for name in names
            if PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or "\\" in name
        ]
        if unsafe:
            errors.append("V22_ZIP_UNSAFE_PATH")
        missing = sorted(REQUIRED - name_set)
        if missing:
            errors.append("V22_REQUIRED_FILES_MISSING")
        manifest = _json(archive.read("artifacts/run_manifest_v22.json"))
        hash_errors: list[str] = []
        size_errors: list[str] = []
        for item in manifest.get("files", []):
            path = str(item["path"])
            if path not in name_set:
                hash_errors.append(path)
                continue
            raw = archive.read(path)
            if _sha256(raw) != str(item["sha256"]):
                hash_errors.append(path)
            if len(raw) != int(item["size_bytes"]):
                size_errors.append(path)
        if hash_errors:
            errors.append("V22_MANIFEST_HASH_MISMATCH")
        if size_errors:
            errors.append("V22_MANIFEST_SIZE_MISMATCH")
        lock = _json(archive.read("manifests/parameter_lock_v22.json"))
        cutoff = _json(archive.read("manifests/forward_v22_cutoff.json"))
        status = _json(archive.read("artifacts/forward_v22_status.json"))
        decisions = pl.read_parquet(
            io.BytesIO(archive.read("artifacts/forward_v22_decisions.parquet"))
        )
        trades = pl.read_parquet(
            io.BytesIO(archive.read("artifacts/forward_v22_trades.parquet"))
        )
        formal_decisions = decisions.filter(
            (pl.col("decision_origin") == "REALTIME")
            & pl.col("eligible_for_forward_evidence")
        )
        recovery_decisions = decisions.filter(
            pl.col("decision_origin") == "RECOVERY_RECONSTRUCTION"
        )
        formal_trades = trades.filter(pl.col("eligible_for_forward_evidence"))
        timer_files = sorted(
            name
            for name in names
            if "forward-v22" in name and (name.endswith(".service") or name.endswith(".timer"))
        )
        if not timer_files:
            warnings.append("V22_FORMAL_TIMER_NOT_PRESENT")
        try:
            _git(repo, "cat-file", "-e", f"{lock['strategy_code_commit']}^{{commit}}")
            commit_exists = True
        except subprocess.CalledProcessError:
            commit_exists = False
            errors.append("V22_GIT_COMMIT_UNAVAILABLE")
        elapsed_hours = max(
            0.0,
            (utc(checked_at) - utc(cutoff["forward_v22_cutoff"])).total_seconds()
            / 3600.0,
        )
        result = {
            "schema_version": "quant_lab_v22_consistency_check.v1",
            "audit_version": "v2.2.1",
            "checked_at": utc(checked_at).isoformat(),
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "warnings": warnings,
            "bundle_path": str(bundle.resolve()),
            "bundle_sha256": bundle_hash,
            "zip_entry_count": len(names),
            "zip_crc_ok": "V22_ZIP_CRC_FAILURE" not in errors,
            "required_files_present": not missing,
            "manifest_hashes_match": not hash_errors,
            "manifest_sizes_match": not size_errors,
            "v22_strategy_id": lock["strategy_id"],
            "v22_strategy_version": lock["strategy_version"],
            "v22_git_commit": lock["strategy_code_commit"],
            "v22_git_commit_available": commit_exists,
            "v22_strategy_code_hash": lock["strategy_code_hash"],
            "v22_parameter_lock_hash": lock["sha256"],
            "v22_forward_cutoff": cutoff["forward_v22_cutoff"],
            "v22_market_data_snapshot_id": status["market_data_snapshot_id"],
            "formal_decision_count": formal_decisions.height,
            "recovery_decision_count": recovery_decisions.height,
            "formal_trade_count": formal_trades.height,
            "completed_cycle_count": int(status["completed_independent_cycles"]),
            "v22_timer_unit_in_bundle": bool(timer_files),
            "v22_timer_files": timer_files,
            "v22_timer_installed": False,
            "v22_timer_enabled": False,
            "v22_missed_schedule_window_evidence": "NOT_RECORDED",
            "v22_unobservable_elapsed_hours": elapsed_hours,
            "v22_forward_deployment_status": "NOT_DEPLOYED",
            "supersession_target": "SUPERSEDED_BY_V221",
        }
    return result


def report(payload: dict[str, Any]) -> str:
    return f"""# Audit v2.2 Consistency Check

Status: **{payload['status']}**

## Immutable bundle

- Bundle SHA256: `{payload['bundle_sha256']}`
- ZIP CRC: `{payload['zip_crc_ok']}`
- Required files: `{payload['required_files_present']}`
- Manifest hashes/sizes: `{payload['manifest_hashes_match']}` / `{payload['manifest_sizes_match']}`
- Locked commit: `{payload['v22_git_commit']}`
- Strategy code hash: `{payload['v22_strategy_code_hash']}`
- Forward cutoff: `{payload['v22_forward_cutoff']}`

## Evidence counts

- Formal REALTIME decisions: `{payload['formal_decision_count']}`
- Recovery decisions: `{payload['recovery_decision_count']}`
- Formal trades: `{payload['formal_trade_count']}`
- Completed cycles: `{payload['completed_cycle_count']}`

## Scheduling gap confirmed

v2.2 contains no formal service/timer unit and no append-only hourly schedule ledger.
Timer installed/enabled therefore evaluates to `false/false`; missed hourly windows are
`NOT_RECORDED`, not silently treated as zero. The elapsed unobservable period at this
check was `{payload['v22_unobservable_elapsed_hours']:.3f}` hours.

The v2.2 evidence remains immutable and will be marked `SUPERSEDED_BY_V221`; none of
its rows are copied into v2.2.1 formal evidence.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            os.environ.get(
                "AUDIT_V22_BUNDLE",
                "/mnt/c/Users/HR/Downloads/alpha_audit_v22_bundle_20260719_053423.zip",
            )
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get("AUDIT_V221_ROOT", "/home/hr/quant-alpha-audit-v2.2.1")
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    payload = validate(
        args.bundle.resolve(),
        args.repo.resolve(),
        datetime.now(UTC).replace(microsecond=0),
    )
    args.root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(payload, args.root / "artifacts/v22_consistency_check.json")
    report_path = args.root / "reports/v22_consistency_check.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report(payload), encoding="utf-8")
    print(f"v22_consistency={payload['status']}")
    print(f"formal_decision_count={payload['formal_decision_count']}")
    print(f"recovery_decision_count={payload['recovery_decision_count']}")
    print(f"timer_evidence={payload['v22_missed_schedule_window_evidence']}")
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
