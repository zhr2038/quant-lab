"""Validate, manifest, and package the complete Audit v2.2 deliverable."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v22 import (  # noqa: E402
    STRATEGY_ID,
    atomic_write_json,
    sha256_file,
    validate_event_chain,
    validate_parameter_lock,
)

REQUIRED_REPORTS = (
    "audit_dashboard_v22.html",
    "executive_summary_v22.md",
    "v21_consistency_check.md",
    "realtime_forward_integrity.md",
    "forward_status_machine.md",
    "immutable_benchmark_design.md",
    "code_identity_lock.md",
    "low_vol_interpretation_v22.md",
    "v5_decision_receipt_v22.md",
    "test_report_v22.md",
    "reproduction_guide_v22.md",
)
REQUIRED_ARTIFACTS = (
    "final_decisions_v22.json",
    "v21_consistency_check.json",
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
    "run_manifest_v22.json",
)
REQUIRED_MANIFESTS = (
    "parameter_lock_v22.json",
    "forward_v22_cutoff.json",
    "strategy_code_hashes_v22.txt",
    "reporting_code_hashes_v22.txt",
    "market_data_snapshot_v22.json",
    "report_consistency_v22.json",
)
SCRIPT_SOURCES = (
    "audit/scripts/stage_v22_consistency.py",
    "audit/scripts/stage_v22_init.py",
    "audit/scripts/stage_v22_forward.py",
    "audit/scripts/stage_v22_report.py",
    "audit/scripts/stage_v22_bundle.py",
    "audit/scripts/stage_v22_test.py",
    "scripts/run_forward_v22_realtime.sh",
    "scripts/run_forward_v22_recovery.sh",
    "scripts/run_alpha_audit_v22.sh",
)
SECRET_VALUE_PATTERNS = (
    re.compile(rb"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE) KEY-----"),
    re.compile(rb"OK-ACCESS-(?:KEY|SIGN|PASSPHRASE)\s*[:=]\s*[A-Za-z0-9+/=_-]{8,}"),
    re.compile(
        rb"(?:api[_-]?secret|passphrase|database[_-]?password)\s*[:=]\s*[^<\s][^\r\n]{7,}",
        re.I,
    ),
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _require(root: Path, directory: str, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (root / directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {directory} deliverables: {missing}")


def _prepare_support(root: Path, repo: Path, v5_repo: Path) -> None:
    script_dir = root / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    for raw in SCRIPT_SOURCES:
        source = repo / raw
        target = script_dir / raw.replace("/", "__")
        shutil.copy2(source, target)
    _write(
        root / "manifests/quant_lab_git_commits_v22.txt",
        _git(repo, "log", "--oneline", "c1797b8780f2abb98e04d5bf10ade064f168411b..HEAD"),
    )
    _write(
        root / "manifests/quant_lab_git_diff_v22.patch",
        _git(repo, "diff", "--binary", "c1797b8780f2abb98e04d5bf10ade064f168411b..HEAD"),
    )
    _write(
        root / "manifests/v5_git_commits_v22.txt",
        _git(v5_repo, "log", "--oneline", "3df1c67cc44cc8be364ec5d3798ea0d3595c0abc..HEAD"),
    )
    _write(
        root / "manifests/v5_git_diff_v22.patch",
        _git(v5_repo, "diff", "--binary", "3df1c67cc44cc8be364ec5d3798ea0d3595c0abc..HEAD"),
    )


def _validate(root: Path) -> None:
    _require(root, "reports", REQUIRED_REPORTS)
    _require(root, "artifacts", REQUIRED_ARTIFACTS)
    _require(root, "manifests", REQUIRED_MANIFESTS)
    _require(root, "schemas", ("v5_decision_receipt_v22.schema.json",))
    consistency = _load(root / "artifacts/v21_consistency_check.json")
    if consistency.get("status") != "PASS":
        raise RuntimeError("v2.1 consistency check is not PASS")
    lock = _load(root / "manifests/parameter_lock_v22.json")
    validate_parameter_lock(lock)
    if lock["strategy_id"] != STRATEGY_ID or lock["btc_trend_lookback_hours"] != 1440:
        raise RuntimeError("parameter lock does not preserve 60-day strategy identity")
    status = _load(root / "artifacts/forward_v22_status.json")
    if status.get("production_alpha") != "FROZEN" or status.get("live") != "NOT_ALLOWED":
        raise RuntimeError("production safety status changed")
    if status.get("runner_integrity") != "PASS":
        raise RuntimeError("formal runner integrity is not PASS")
    reports = _load(root / "manifests/report_consistency_v22.json")
    if reports.get("status") != "PASS":
        raise RuntimeError("cross-format report consistency is not PASS")
    tests = _load(root / "artifacts/test_execution_v22.json")
    if tests.get("status") != "PASS":
        raise RuntimeError("test execution status is not PASS")
    decisions = pl.read_parquet(root / "artifacts/forward_v22_decisions.parquet")
    recovery = decisions.filter(pl.col("decision_origin") == "RECOVERY_RECONSTRUCTION")
    if not recovery.is_empty() and recovery["eligible_for_forward_evidence"].any():
        raise RuntimeError("recovery decision entered formal evidence")
    trades = pl.read_parquet(root / "artifacts/forward_v22_trades.parquet")
    if not recovery.is_empty():
        recovery_ids = recovery["decision_id"]
        mixed = trades.filter(
            pl.col("decision_id").is_in(recovery_ids)
            & pl.col("eligible_for_forward_evidence")
        )
        if not mixed.is_empty():
            raise RuntimeError("recovery trade entered formal evidence")
    validate_event_chain(pl.read_parquet(root / "artifacts/forward_v22_events.parquet"))
    validate_event_chain(
        pl.read_parquet(root / "artifacts/forward_v22_benchmark_events.parquet")
    )


def _readme(bundle_name: str) -> str:
    return f"""Audit v2.2 — README FIRST

1. Open reports/audit_dashboard_v22.html.
2. Read reports/realtime_forward_integrity.md.
3. Machine decisions: artifacts/final_decisions_v22.json.
4. Locked strategy: manifests/parameter_lock_v22.json.
5. RECOVERY_RECONSTRUCTION is audit-only and never formal Forward evidence.
6. Production Alpha remains FROZEN and Live is NOT_ALLOWED.

Bundle: {bundle_name}
"""


def _candidate_files(root: Path) -> list[Path]:
    files = [root / "README_FIRST_V22.txt"]
    for directory in ("reports", "artifacts", "manifests", "schemas", "scripts", "logs"):
        base = root / directory
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _secret_scan(files: list[Path]) -> None:
    hits: list[str] = []
    for path in files:
        raw = path.read_bytes()
        for pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(raw):
                hits.append(path.as_posix())
                break
    if hits:
        raise RuntimeError(f"secret-like values detected in bundle candidates: {hits}")


def _manifest(root: Path, repo: Path, v5_repo: Path, files: list[Path]) -> dict[str, Any]:
    managed = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
        if path.relative_to(root).as_posix() != "artifacts/run_manifest_v22.json"
    ]
    status = _load(root / "artifacts/forward_v22_status.json")
    lock = _load(root / "manifests/parameter_lock_v22.json")
    return {
        "audit_version": "v2.2",
        "generated_at": datetime.now(UTC).isoformat(),
        "quant_lab_branch": _git(repo, "branch", "--show-current"),
        "quant_lab_head": _git(repo, "rev-parse", "HEAD"),
        "v5_branch": _git(v5_repo, "branch", "--show-current"),
        "v5_head": _git(v5_repo, "rev-parse", "HEAD"),
        "strategy_id": lock["strategy_id"],
        "strategy_code_commit": lock["strategy_code_commit"],
        "strategy_code_hash": lock["strategy_code_hash"],
        "parameter_lock_hash": lock["sha256"],
        "forward_v22_cutoff": status["forward_v22_cutoff"],
        "paper_status": status["paper_status"],
        "manifest_scope": "all bundle files except this self-referential manifest",
        "files": managed,
        "safety": {
            "production_mutations": 0,
            "deployments": 0,
            "live_orders": 0,
            "production_alpha": "FROZEN",
            "live": "NOT_ALLOWED",
            "recovery_records_excluded": True,
        },
    }


def _zip(root: Path, bundle: Path, files: list[Path]) -> None:
    partial = bundle.with_name(f".{bundle.name}.{os.getpid()}.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            relative = path.relative_to(root).as_posix()
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"unsafe archive path: {relative}")
            info = zipfile.ZipInfo(relative)
            info.date_time = datetime.now().timetuple()[:6]
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o640) << 16
            zf.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    os.replace(partial, bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--v5-repo", type=Path, default=Path("/home/hr/quant-alpha-audit-v2.2/repos/V5-prod")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve()
    v5_repo = args.v5_repo.resolve()
    _prepare_support(root, repo, v5_repo)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"alpha_audit_v22_bundle_{timestamp}.zip"
    readme = _readme(bundle_name)
    _write(root / "README_FIRST_V22.txt", readme)
    _write(root / "bundles/README_FIRST_V22.txt", readme)
    placeholder = root / "artifacts/run_manifest_v22.json"
    if not placeholder.exists():
        atomic_write_json({}, placeholder)
    files = _candidate_files(root)
    manifest = _manifest(root, repo, v5_repo, files)
    atomic_write_json(manifest, placeholder)
    _validate(root)
    files = _candidate_files(root)
    _secret_scan(files)
    bundle = root / "bundles" / bundle_name
    _zip(root, bundle, files)
    digest = sha256_file(bundle)
    _write(bundle.with_suffix(bundle.suffix + ".sha256"), f"{digest}  {bundle.name}")
    print(f"bundle={bundle}")
    print(f"bundle_sha256={digest}")
    print(f"bundle_files={len(files)}")


if __name__ == "__main__":
    main()
