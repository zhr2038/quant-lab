"""Validate and package the final Audit v2.1 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_REPORTS = (
    "audit_dashboard_v21.html",
    "executive_summary_v21.md",
    "forward_runner_fix_report.md",
    "forward_paper_report_v21.md",
    "funding_window_correction_v21.md",
    "permutation_and_controls_report_v21.md",
    "symbol_concentration_v21.md",
    "v5_decision_receipt_design.md",
    "test_report_v21.md",
    "reproduction_guide_v21.md",
)
REQUIRED_ARTIFACTS = (
    "final_decisions_v21.json",
    "forward_v21_decisions.parquet",
    "forward_v21_trades.parquet",
    "forward_v21_events.parquet",
    "forward_v21_equity.parquet",
    "forward_v21_benchmarks.parquet",
    "forward_v21_performance.csv",
    "forward_v21_status.json",
    "funding_frequency_summary_v21.csv",
    "funding_window_comparison_v21.csv",
    "funding_signal_results_v21.csv",
    "null_control_summary_v21.csv",
    "robustness_perturbation_summary_v21.csv",
    "symbol_contribution_by_partition_v21.csv",
    "test_execution_v21.json",
    "browser_qa_v21.json",
)
REQUIRED_MANIFESTS = (
    "parameter_lock_v21.json",
    "forward_v21_cutoff.json",
    "provisional_initialization_rejection_v21.json",
    "code_hashes_v21.txt",
    "report_consistency_v21.json",
)
SOURCE_SCRIPTS = (
    "scripts/run_alpha_audit_v21.sh",
    "scripts/run_low_vol_forward_paper.sh",
    "audit/scripts/stage_v21_init.py",
    "audit/scripts/stage_v21_forward.py",
    "audit/scripts/stage_v21_analysis.py",
    "audit/scripts/stage_v21_test.py",
    "audit/scripts/stage_v21_report.py",
    "audit/scripts/stage_v21_bundle.py",
)
CODE_HASH_PATHS = (
    "audit/auditlib/forward_v21.py",
    "audit/auditlib/funding_v2.py",
    "audit/auditlib/contribution_v21.py",
    "audit/auditlib/permutation_v21.py",
    "audit/auditlib/report_v21.py",
    "audit/auditlib/portfolio_backtest.py",
    *SOURCE_SCRIPTS,
    "schemas/v5_decision_receipt.schema.json",
    "docs/v5_decision_receipt_design.md",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _require(root: Path, directory: str, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (root / directory / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required {directory} files: {missing}")


def _validate(root: Path, repo: Path) -> None:
    _require(root, "reports", REQUIRED_REPORTS)
    _require(root, "artifacts", REQUIRED_ARTIFACTS)
    _require(root, "manifests", REQUIRED_MANIFESTS)
    if not (root / "schemas/v5_decision_receipt.schema.json").is_file():
        raise RuntimeError("missing receipt schema in audit root")
    consistency = json.loads(
        (root / "manifests/report_consistency_v21.json").read_text()
    )
    if not consistency.get("consistent"):
        raise RuntimeError("report consistency is not PASS")
    tests = json.loads((root / "artifacts/test_execution_v21.json").read_text())
    if tests.get("overall_status") != "PASS":
        raise RuntimeError("test execution is not PASS")
    if tests.get("v5_regression", {}).get("status") != "PASS":
        raise RuntimeError("fresh V5 read-only regression is not PASS")
    browser = json.loads((root / "artifacts/browser_qa_v21.json").read_text())
    if browser.get("status") != "PASS":
        raise RuntimeError("browser QA is not PASS")
    decisions = json.loads((root / "artifacts/final_decisions_v21.json").read_text())
    if decisions.get("live_or_live_small_permitted") is not False:
        raise RuntimeError("final decisions permit Live")
    if decisions["current_v5"].get("production_alpha") != "FROZEN":
        raise RuntimeError("production Alpha is not frozen")
    if decisions["forward_v21"].get("portfolio_validity") == "PASS":
        raise RuntimeError("forward portfolio was automatically marked PASS")
    if _git(repo, "branch", "--show-current") != "audit/alpha-validity-v2.1":
        raise RuntimeError("bundle must be generated from audit/alpha-validity-v2.1")


def _prepare_support_files(root: Path, repo: Path) -> None:
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for relative in SOURCE_SCRIPTS:
        source = repo / relative
        destination = scripts_dir / relative.replace("/", "__")
        shutil.copy2(source, destination)
    schema_dir = root / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        repo / "schemas/v5_decision_receipt.schema.json",
        schema_dir / "v5_decision_receipt.schema.json",
    )

    code_hashes = [
        f"{_sha(repo / relative)}  {relative}" for relative in CODE_HASH_PATHS
    ]
    (root / "manifests/code_hashes_v21.txt").write_text(
        "\n".join(code_hashes) + "\n", encoding="utf-8"
    )
    base = "0f1ddfb50d288cdfdfd97cbd21e7289c9aca9f2d"
    (root / "manifests/git_diff_v21.patch").write_text(
        _git(repo, "diff", "--binary", f"{base}..HEAD") + "\n",
        encoding="utf-8",
    )
    (root / "manifests/git_commits_v21.txt").write_text(
        _git(repo, "log", "--oneline", "--decorate", f"{base}..HEAD") + "\n",
        encoding="utf-8",
    )


def _write_manifest(root: Path, repo: Path) -> dict:
    included: list[dict] = []
    readme = root / "README_FIRST_V21.txt"
    if readme.is_file():
        included.append(
            {
                "path": readme.name,
                "size_bytes": readme.stat().st_size,
                "sha256": _sha(readme),
            }
        )
    for directory in ("reports", "artifacts", "manifests", "schemas", "scripts"):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or path.name == "run_manifest_v21.json":
                continue
            included.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
    lock = json.loads((root / "manifests/parameter_lock_v21.json").read_text())
    cutoff = json.loads((root / "manifests/forward_v21_cutoff.json").read_text())
    status = json.loads((root / "artifacts/forward_v21_status.json").read_text())
    payload = {
        "audit_version": "v2.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": _git(repo, "branch", "--show-current"),
        "final_commit": _git(repo, "rev-parse", "HEAD"),
        "runner_code_commit": lock["code_commit"],
        "parameter_lock_hash": lock["sha256"],
        "forward_v21_start_cutoff": cutoff["forward_v21_start_cutoff"],
        "available_market_data_cutoff": status["available_market_data_cutoff"],
        "forward_available_days": status["forward_available_days"],
        "legacy_v2_status": "INVALIDATED_BY_RUNNER_BUG",
        "v2_bundle_sha256": "410a18e4b001a8c395b922357e9f22f7a1e94da2baaa37444e68467f8f874a51",
        "manifest_scope": "all bundle files except this self-referential manifest",
        "files": included,
        "safety": {
            "production_mutations": 0,
            "live_orders": 0,
            "deployment": False,
            "production_alpha": "FROZEN",
            "live_order_effect": "none",
        },
    }
    path = root / "artifacts/run_manifest_v21.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _secret_scan(root: Path) -> None:
    patterns = (
        re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(
            rb"(?i)(api[_-]?key|api[_-]?secret|passphrase|password|database[_-]?password)"
            rb"\s*[\"']?\s*[:=]\s*[\"'][^\"']{4,}[\"']"
        ),
    )
    findings: list[str] = []
    for directory in ("reports", "artifacts", "manifests", "schemas", "scripts"):
        for path in (root / directory).rglob("*"):
            if not path.is_file():
                continue
            data = path.read_bytes()
            if any(pattern.search(data) for pattern in patterns):
                findings.append(path.relative_to(root).as_posix())
    if findings:
        raise RuntimeError(f"secret-like content found in bundle: {findings}")


def _readme(root: Path, bundle_name: str) -> str:
    value = f"""Audit v2.1 evidence bundle: {bundle_name}

1. 先打开 reports/audit_dashboard_v21.html
2. 查看 reports/forward_runner_fix_report.md
3. 查看 artifacts/final_decisions_v21.json
4. 查看 manifests/parameter_lock_v21.json
5. Forward v2旧记录已失效
6. Forward v2.1数据不得与旧数据合并
7. 当前结果不得用于恢复实盘Alpha

Audit v2.1 is research-only. No production deployment, order, position change,
risk-permission change, service restart or Alpha enablement is included.
"""
    path = root / "README_FIRST_V21.txt"
    path.write_text(value, encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--downloads", type=Path, default=Path("/mnt/c/Users/HR/Downloads")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve()
    _prepare_support_files(root, repo)
    _validate(root, repo)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    bundle_name = f"alpha_audit_v21_bundle_{timestamp}.zip"
    readme = _readme(root, bundle_name)
    manifest = _write_manifest(root, repo)
    _secret_scan(root)
    bundle_dir = root / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / bundle_name
    with zipfile.ZipFile(
        bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive.write(root / "README_FIRST_V21.txt", "README_FIRST_V21.txt")
        for directory in ("reports", "artifacts", "manifests", "schemas", "scripts"):
            for path in sorted((root / directory).rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
    downloads = args.downloads.resolve()
    downloads.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_path, downloads / bundle_name)
    (bundle_dir / "README_FIRST_V21.txt").write_text(readme, encoding="utf-8")
    (downloads / "README_FIRST_V21.txt").write_text(readme, encoding="utf-8")
    sidecar = f"{_sha(bundle_path)}  {bundle_name}\n"
    (bundle_path.with_suffix(bundle_path.suffix + ".sha256")).write_text(sidecar)
    (downloads / f"{bundle_name}.sha256").write_text(sidecar)
    print(f"bundle={bundle_path}")
    print(f"downloads_bundle={downloads / bundle_name}")
    print(f"sha256={_sha(bundle_path)}")
    print(f"manifest_files={len(manifest['files'])}")


if __name__ == "__main__":
    main()
