# ruff: noqa: E501
"""Validate, manifest, and package the complete Audit v2.2.1 deliverable."""

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

from audit.auditlib.forward_v221 import (  # noqa: E402
    BENCHMARK_COVERAGE_SCHEMA,
    EVENT_SCHEMA,
    SCHEDULE_EVENT_SCHEMA,
    STRATEGY_ID,
    STRATEGY_VERSION,
    atomic_write_json,
    sha256_file,
    validate_hash_chain,
    validate_parameter_lock,
)

REQUIRED_REPORTS = (
    "audit_dashboard_v221.html",
    "executive_summary_v221.md",
    "v22_consistency_check.md",
    "benchmark_coverage_fix.md",
    "recovery_isolation.md",
    "schedule_integrity.md",
    "systemd_deployment.md",
    "forward_start_readiness.md",
    "test_report_v221.md",
    "reproduction_guide_v221.md",
)
REQUIRED_ARTIFACTS = (
    "final_decisions_v221.json",
    "v22_consistency_check.json",
    "forward_v221_decisions.parquet",
    "forward_v221_trades.parquet",
    "forward_v221_events.parquet",
    "forward_v221_equity.parquet",
    "forward_v221_benchmark_decisions.parquet",
    "forward_v221_benchmark_trades.parquet",
    "forward_v221_benchmark_events.parquet",
    "forward_v221_benchmark_equity.parquet",
    "forward_v221_benchmark_coverage.parquet",
    "forward_v221_schedule_events.parquet",
    "forward_v221_runtime_health.parquet",
    "forward_v221_performance.csv",
    "forward_v221_status.json",
    "test_execution_v221.json",
    "run_manifest_v221.json",
)
REQUIRED_MANIFESTS = (
    "parameter_lock_v221.json",
    "forward_v221_cutoff.json",
    "strategy_code_hashes_v221.txt",
    "deployment_hashes_v221.txt",
    "systemd_deployment_v221.json",
    "report_consistency_v221.json",
)
DEPLOY_FILES = (
    "systemd/quant-lab-forward-v221.service",
    "systemd/quant-lab-forward-v221.timer",
    "install_forward_v221_systemd.sh",
    "uninstall_forward_v221_systemd.sh",
    "check_forward_v221_health.sh",
)
SUPPORT_SOURCES = (
    "audit/auditlib/forward_v221.py",
    "audit/scripts/stage_v221_bundle.py",
    "audit/scripts/stage_v221_consistency.py",
    "audit/scripts/stage_v221_deployment.py",
    "audit/scripts/stage_v221_forward.py",
    "audit/scripts/stage_v221_init.py",
    "audit/scripts/stage_v221_report.py",
    "audit/scripts/stage_v221_test.py",
    "scripts/run_alpha_audit_v221.sh",
    "scripts/run_forward_v221_realtime.sh",
    "scripts/run_forward_v221_recovery.sh",
    "schemas/forward_parameter_lock_v221.schema.json",
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE) KEY-----"),
    re.compile(rb"OK-ACCESS-(?:KEY|SIGN|PASSPHRASE)\s*[:=]\s*[A-Za-z0-9+/=_-]{8,}"),
    re.compile(
        rb"(?:api[_-]?secret|passphrase|database[_-]?password)\s*[:=]\s*[^<\s][^\r\n]{7,}",
        re.I,
    ),
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _require(root: Path, directory: str, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (root / directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {directory} deliverables: {missing}")


def _prepare_support(root: Path, repo: Path) -> None:
    deploy_root = root / "deploy"
    for relative in DEPLOY_FILES:
        source = repo / "deploy" / relative
        target = deploy_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    script_root = root / "scripts"
    script_root.mkdir(parents=True, exist_ok=True)
    for relative in SUPPORT_SOURCES:
        source = repo / relative
        target = script_root / relative.replace("/", "__")
        shutil.copy2(source, target)
    schema_root = root / "schemas"
    schema_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        repo / "schemas/forward_parameter_lock_v221.schema.json",
        schema_root / "forward_parameter_lock_v221.schema.json",
    )
    base = "1cea72b1218d3839bcd7fa77c7cb64adc2f10205"
    _write(
        root / "manifests/quant_lab_git_commits_v221.txt",
        _git(repo, "log", "--oneline", f"{base}..HEAD"),
    )
    _write(
        root / "manifests/quant_lab_git_diff_v221.patch",
        _git(repo, "diff", "--binary", f"{base}..HEAD"),
    )
    _write(root / "manifests/quant_lab_git_status_v221.txt", _git(repo, "status", "--short"))


def _validate(root: Path) -> None:
    _require(root, "reports", REQUIRED_REPORTS)
    _require(root, "artifacts", REQUIRED_ARTIFACTS)
    _require(root, "manifests", REQUIRED_MANIFESTS)
    _require(root, "deploy", DEPLOY_FILES)
    consistency = _load(root / "artifacts/v22_consistency_check.json")
    if consistency.get("status") != "PASS":
        raise RuntimeError("v2.2 consistency is not PASS")
    lock = _load(root / "manifests/parameter_lock_v221.json")
    validate_parameter_lock(lock)
    if (
        lock["strategy_id"] != STRATEGY_ID
        or lock["strategy_version"] != STRATEGY_VERSION
        or lock["btc_trend_lookback_hours"] != 1440
    ):
        raise RuntimeError("locked strategy identity changed")
    status = _load(root / "artifacts/forward_v221_status.json")
    if status.get("forward_start_status") != "FORWARD_V221_READY":
        raise RuntimeError("formal Forward is not deployment-ready")
    if status.get("production_alpha") != "FROZEN" or status.get("live") != "NOT_ALLOWED":
        raise RuntimeError("production safety boundary changed")
    if not all(
        (
            status.get("timer_installed"),
            status.get("timer_enabled"),
            status.get("timer_active"),
            status.get("runner_integrity") == "PASS",
            status.get("working_tree_clean"),
            status.get("unit_hash_match"),
        )
    ):
        raise RuntimeError("deployment or identity gate is incomplete")
    reports = _load(root / "manifests/report_consistency_v221.json")
    if reports.get("status") != "PASS":
        raise RuntimeError("cross-format report consistency is not PASS")
    tests = _load(root / "artifacts/test_execution_v221.json")
    if tests.get("status") != "PASS":
        raise RuntimeError("test execution status is not PASS")
    decisions = pl.read_parquet(root / "artifacts/forward_v221_decisions.parquet")
    trades = pl.read_parquet(root / "artifacts/forward_v221_trades.parquet")
    recovery = decisions.filter(pl.col("decision_origin") == "RECOVERY_RECONSTRUCTION")
    if not recovery.is_empty() and recovery["eligible_for_forward_evidence"].any():
        raise RuntimeError("Recovery decision entered formal evidence")
    if not recovery.is_empty():
        mixed = trades.filter(
            pl.col("decision_id").is_in(recovery["decision_id"])
            & pl.col("eligible_for_forward_evidence")
        )
        if not mixed.is_empty():
            raise RuntimeError("Recovery trade entered formal evidence")
    validate_hash_chain(
        pl.read_parquet(root / "artifacts/forward_v221_events.parquet"),
        schema=EVENT_SCHEMA,
    )
    validate_hash_chain(
        pl.read_parquet(root / "artifacts/forward_v221_benchmark_events.parquet"),
        schema=EVENT_SCHEMA,
    )
    validate_hash_chain(
        pl.read_parquet(root / "artifacts/forward_v221_schedule_events.parquet"),
        schema=SCHEDULE_EVENT_SCHEMA,
    )
    coverage = pl.read_parquet(
        root / "artifacts/forward_v221_benchmark_coverage.parquet"
    ).cast(BENCHMARK_COVERAGE_SCHEMA)
    if not coverage.is_empty():
        invariant = coverage.filter(
            (pl.col("invested_weight") + pl.col("cash_residual") - 1.0).abs()
            > 1e-9
        )
        if not invariant.is_empty():
            raise RuntimeError("benchmark cash invariant failed")


def _readme(bundle_name: str) -> str:
    return f"""Audit v2.2.1 — README FIRST

1. 打开 reports/audit_dashboard_v221.html
2. 查看 reports/forward_start_readiness.md
3. 查看 artifacts/final_decisions_v221.json
4. 查看 manifests/parameter_lock_v221.json
5. 查看 manifests/forward_v221_cutoff.json
6. Recovery数据不属于正式Forward证据
7. 当前不得恢复实盘Alpha

Bundle: {bundle_name}
"""


def _candidate_files(root: Path) -> list[Path]:
    files = [root / "README_FIRST_V221.txt"]
    for directory in (
        "reports",
        "artifacts",
        "manifests",
        "deploy",
        "scripts",
        "schemas",
        "logs",
    ):
        base = root / directory
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _secret_scan(files: list[Path]) -> None:
    hits: list[str] = []
    for path in files:
        raw = path.read_bytes()
        if any(pattern.search(raw) for pattern in SECRET_PATTERNS):
            hits.append(path.as_posix())
    if hits:
        raise RuntimeError(f"secret-like values detected: {hits}")


def _manifest(root: Path, repo: Path, files: list[Path]) -> dict[str, Any]:
    status = _load(root / "artifacts/forward_v221_status.json")
    lock = _load(root / "manifests/parameter_lock_v221.json")
    managed = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
        if path.relative_to(root).as_posix() != "artifacts/run_manifest_v221.json"
    ]
    return {
        "audit_version": "v2.2.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "quant_lab_branch": _git(repo, "branch", "--show-current"),
        "quant_lab_head": _git(repo, "rev-parse", "HEAD"),
        "quant_lab_working_tree_clean": not bool(_git(repo, "status", "--porcelain")),
        "strategy_id": lock["strategy_id"],
        "strategy_version": lock["strategy_version"],
        "strategy_code_commit": lock["strategy_code_commit"],
        "strategy_code_hash": lock["strategy_code_hash"],
        "parameter_lock_hash": lock["sha256"],
        "forward_v221_cutoff": status["forward_v221_cutoff"],
        "next_legal_strategy_decision": status["next_legal_strategy_decision"],
        "forward_start_status": status["forward_start_status"],
        "paper_status": status["paper_status"],
        "research_node": status["research_node"],
        "timer_active": status["timer_active"],
        "manifest_scope": "all bundle files except this self-referential manifest",
        "files": managed,
        "safety": {
            "v5_production_modified": False,
            "live_orders": 0,
            "production_alpha": "FROZEN",
            "live": "NOT_ALLOWED",
            "recovery_records_excluded": True,
        },
    }


def _zip(root: Path, bundle: Path, files: list[Path]) -> None:
    partial = bundle.with_name(f".{bundle.name}.{os.getpid()}.partial")
    with zipfile.ZipFile(
        partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"unsafe archive path: {relative}")
            info = zipfile.ZipInfo(relative)
            info.date_time = datetime.now().timetuple()[:6]
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o640) << 16
            archive.writestr(
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
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve()
    _prepare_support(root, repo)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"alpha_audit_v221_bundle_{timestamp}.zip"
    readme = _readme(bundle_name)
    _write(root / "README_FIRST_V221.txt", readme)
    _write(root / "bundles/README_FIRST_V221.txt", readme)
    manifest_path = root / "artifacts/run_manifest_v221.json"
    if not manifest_path.exists():
        atomic_write_json({}, manifest_path)
    files = _candidate_files(root)
    atomic_write_json(_manifest(root, repo, files), manifest_path)
    _validate(root)
    files = _candidate_files(root)
    _secret_scan(files)
    bundle = root / "bundles" / bundle_name
    bundle.parent.mkdir(parents=True, exist_ok=True)
    _zip(root, bundle, files)
    digest = sha256_file(bundle)
    _write(bundle.with_suffix(bundle.suffix + ".sha256"), f"{digest}  {bundle.name}")
    print(f"bundle={bundle}")
    print(f"bundle_sha256={digest}")
    print(f"bundle_files={len(files)}")


if __name__ == "__main__":
    main()
