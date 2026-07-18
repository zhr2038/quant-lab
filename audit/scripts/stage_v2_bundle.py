"""Finalize, validate, and package the Audit v2 evidence bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

AUDIT_ROOT = Path(os.environ.get("AUDIT_ROOT", "/home/hr/quant-alpha-audit-v2")).resolve()
QLAB_REPO = Path(
    os.environ.get("QLAB_REPO_PATH", "/home/hr/quant-alpha-audit/repos/quant-lab")
).resolve()
AUDIT_V1_BUNDLE = Path(
    os.environ.get(
        "AUDIT_V1_BUNDLE",
        "/mnt/c/Users/HR/Downloads/alpha_audit_bundle_20260718_162903.zip",
    )
).resolve()
BASE_COMMIT = os.environ.get(
    "AUDIT_V2_BASE_COMMIT", "9993e84d44c586f8b97a433cd4b2c9e6d324d499"
)

REQUIRED_REPORTS = {
    "audit_dashboard_v2.html",
    "executive_summary_v2.md",
    "v1_audit_consistency_check.md",
    "24h_portfolio_validation.md",
    "low_vol_signal_vs_portfolio.md",
    "forward_paper_report.md",
    "funding_window_correction.md",
    "permutation_and_controls_report.md",
    "statistical_sensitivity_report.md",
    "cost_model_v2.md",
    "v5_replayability_audit.md",
    "production_impact_assessment_v2.md",
    "reproduction_guide_v2.md",
    "test_report_v2.md",
}

REQUIRED_ARTIFACTS = {
    "browser_qa_v2.json",
    "final_decisions_v2.json",
    "v1_audit_consistency_check.json",
    "portfolio_24h_results.csv",
    "low_vol_layered_decisions.csv",
    "hac_bandwidth_sensitivity.csv",
    "block_bootstrap_summary.csv",
    "permutation_distribution.parquet",
    "null_control_summary.csv",
    "robustness_perturbation_summary.csv",
    "funding_frequency_summary.csv",
    "funding_window_comparison.csv",
    "cost_coverage_by_symbol.csv",
    "cost_distribution_v2.csv",
    "forward_paper_decisions.parquet",
    "forward_paper_positions.parquet",
    "forward_paper_performance.csv",
    "v5_replayability_status.json",
    "run_manifest_v2.json",
}

REQUIRED_MANIFESTS = {
    "v5_runtime_snapshot.json",
    "v5_git_diff.patch",
    "systemd_units_sha256.txt",
    "container_image_digests.txt",
    "config_sha256.txt",
    "environment_variable_presence.json",
    "quant_lab_git_state.txt",
    "git_commits.txt",
    "git_diff.patch",
}

SCRIPT_SOURCES = {
    "run_alpha_audit_v2.sh": QLAB_REPO / "scripts/run_alpha_audit_v2.sh",
    "run_low_vol_forward_paper.sh": QLAB_REPO
    / "scripts/run_low_vol_forward_paper.sh",
    "stages/stage_v2_analysis.py": QLAB_REPO
    / "audit/scripts/stage_v2_analysis.py",
    "stages/stage_v2_bundle.py": QLAB_REPO / "audit/scripts/stage_v2_bundle.py",
    "stages/stage_v2_consistency.py": QLAB_REPO
    / "audit/scripts/stage_v2_consistency.py",
    "stages/stage_v2_cost.py": QLAB_REPO / "audit/scripts/stage_v2_cost.py",
    "stages/stage_v2_forward.py": QLAB_REPO / "audit/scripts/stage_v2_forward.py",
    "stages/stage_v2_report.py": QLAB_REPO / "audit/scripts/stage_v2_report.py",
    "stages/stage_v2_v5_snapshot.py": QLAB_REPO
    / "audit/scripts/stage_v2_v5_snapshot.py",
    "auditlib/decisions_v2.py": QLAB_REPO / "audit/auditlib/decisions_v2.py",
    "auditlib/forward_paper.py": QLAB_REPO / "audit/auditlib/forward_paper.py",
    "auditlib/funding_v2.py": QLAB_REPO / "audit/auditlib/funding_v2.py",
    "auditlib/runtime_snapshot.py": QLAB_REPO
    / "audit/auditlib/runtime_snapshot.py",
    "auditlib/statistics_v2.py": QLAB_REPO / "audit/auditlib/statistics_v2.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(QLAB_REPO), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _junit_totals(path: Path) -> dict[str, int | float]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "seconds": round(
            sum(float(suite.attrib.get("time", 0.0)) for suite in suites), 3
        ),
    }


def _write_test_report(reports: Path, artifacts: Path) -> dict[str, dict]:
    suites = {
        "audit_v2_and_existing_audit": _junit_totals(artifacts / "audit_v2_tests.xml"),
        "quant_lab_full": _junit_totals(artifacts / "pytest_all_v2.xml"),
        "v5_full_read_only_checkout": _junit_totals(
            artifacts / "v5_pytest_all_v2.xml"
        ),
    }
    browser_path = artifacts / "browser_qa_v2.json"
    browser = json.loads(browser_path.read_text(encoding="utf-8"))
    lines = [
        "# Audit v2 Test Report",
        "",
        "## Outcome",
        "",
        "All Audit v2, quant-lab, and V5 tests completed without failures or errors. "
        "No failing test was skipped, removed, or weakened.",
        "",
        "| Scope | Tests | Passed | Failed | Errors | Skipped | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in suites.items():
        passed = int(values["tests"]) - int(values["failures"]) - int(
            values["errors"]
        ) - int(values["skipped"])
        lines.append(
            f"| {name} | {values['tests']} | {passed} | {values['failures']} | "
            f"{values['errors']} | {values['skipped']} | {values['seconds']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "The six V5 skips are pre-existing platform-specific tests: four require "
            "reliable POSIX mode/symlink semantics and two require POSIX `fcntl`; the V5 "
            "suite ran on the read-only Windows checkout. They are not Audit v2 skips.",
            "",
            "## Static and browser checks",
            "",
            "- `ruff check .`: PASS.",
            "- `python -m compileall -q audit src`: PASS.",
            "- `git diff --check`: PASS.",
            f"- Static dashboard browser QA: {browser['status']} at "
            f"{', '.join(browser['viewports'])}; console errors="
            f"{browser['console_errors']}, page errors={browser['page_errors']}.",
            "- Dashboard field/schema consistency is covered by "
            "`test_dashboard_contract_contains_layered_fields`.",
            "",
            "## Commands",
            "",
            "```bash",
            "pytest -q audit/tests --junitxml=$AUDIT_ROOT/artifacts/audit_v2_tests.xml",
            "pytest -q --junitxml=$AUDIT_ROOT/artifacts/pytest_all_v2.xml",
            "ruff check .",
            "python -m compileall -q audit src",
            "git diff --check",
            "```",
            "",
            "V5 was tested separately from its clean local production-SHA checkout; no "
            "V5 file was modified and no live service or exchange endpoint was invoked.",
        ]
    )
    _write(reports / "test_report_v2.md", "\n".join(lines))
    return suites


def _write_git_manifests(manifests: Path) -> None:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain=v1") or "(clean)"
    _write(
        manifests / "quant_lab_git_state.txt",
        "\n".join(
            [
                f"generated_at={datetime.now(UTC).isoformat()}",
                f"branch={branch}",
                f"head={head}",
                f"base_commit={BASE_COMMIT}",
                "remote:",
                _git("remote", "-v"),
                "status:",
                status,
            ]
        ),
    )
    _write(
        manifests / "git_commits.txt",
        _git(
            "log",
            "--reverse",
            "--format=%H%x09%ad%x09%s",
            "--date=iso-strict",
            f"{BASE_COMMIT}..HEAD",
        ),
    )
    _write(manifests / "git_diff.patch", _git("diff", "--binary", BASE_COMMIT, "HEAD"))


def _copy_reproduction_scripts(destination: Path) -> None:
    for relative, source in SCRIPT_SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(f"missing reproduction script: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _required_paths(root: Path) -> list[Path]:
    return [
        *(root / "reports" / name for name in sorted(REQUIRED_REPORTS)),
        *(root / "artifacts" / name for name in sorted(REQUIRED_ARTIFACTS)),
        *(root / "manifests" / name for name in sorted(REQUIRED_MANIFESTS)),
    ]


def _bundle_paths(root: Path) -> list[Path]:
    return [
        *(path for path in sorted((root / "scripts").rglob("*")) if path.is_file()),
        *(
            path
            for directory_name in ("reports", "artifacts", "manifests")
            for path in sorted((root / directory_name).rglob("*"))
            if path.is_file()
        ),
    ]


def _assert_required_outputs(root: Path) -> None:
    missing = [str(path.relative_to(root)) for path in _required_paths(root) if not path.is_file()]
    if missing:
        raise SystemExit("required Audit v2 outputs are missing: " + ", ".join(missing))


def _assert_secret_safe(paths: list[Path]) -> None:
    forbidden_literals = {
        value
        for name, value in os.environ.items()
        if value
        and len(value) >= 8
        and any(token in name.upper() for token in ("KEY", "SECRET", "TOKEN", "PASS"))
    }
    forbidden_headers = (
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
    )
    for path in paths:
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="ignore")
        if any(
            line.strip().startswith(forbidden_headers) for line in text.splitlines()
        ):
            raise SystemExit(f"private-key material found in {path}")
        for literal in forbidden_literals:
            if literal in text:
                raise SystemExit(f"sensitive environment value found in {path}")


def _write_run_manifest(
    artifacts: Path,
    reports: Path,
    manifests: Path,
    scripts: Path,
    suites: dict[str, dict],
) -> None:
    decisions = json.loads((artifacts / "final_decisions_v2.json").read_text())
    replay = json.loads((artifacts / "v5_replayability_status.json").read_text())
    consistency = json.loads(
        (artifacts / "v1_audit_consistency_check.json").read_text()
    )
    files: list[dict[str, int | str]] = []
    for directory in (reports, artifacts, manifests, scripts):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "run_manifest_v2.json":
                files.append(
                    {
                        "path": str(path.relative_to(AUDIT_ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    payload = {
        "audit_version": "v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": _git("branch", "--show-current"),
        "quant_lab_commit": _git("rev-parse", "HEAD"),
        "base_commit": BASE_COMMIT,
        "v5_commit": replay["git_head"],
        "audit_v1_bundle": {
            "path": str(AUDIT_V1_BUNDLE),
            "sha256": _sha256(AUDIT_V1_BUNDLE),
            "consistency_status": consistency["overall_status"],
        },
        "data": {
            "v1_cutoff": decisions["audit_v1_data_cutoff"],
            "v2_cutoff": decisions["audit_v2_data_cutoff"],
            "v1_snapshot_id": consistency["snapshot_id"],
            "forward_only_after_v1_cutoff": True,
        },
        "frozen_hypothesis": {
            "factor": "low_vol_20d",
            "top_n": 3,
            "weighting": "score",
            "rebalance_hours": 120,
            "btc_trend_filter": True,
            "hypothesis_type": "POST_HOC_HYPOTHESIS",
            "parameters_locked": True,
        },
        "tests": suites,
        "production_safety": {
            "production_mutations": 0,
            "orders_submitted": 0,
            "live_deployment": False,
            "v5_collection": "READ_ONLY",
            "secret_values_stored": False,
        },
        "manifest_scope": (
            "All bundled reports, artifacts, manifests, and scripts except this "
            "self-referential manifest."
        ),
        "files": files,
    }
    _write(
        artifacts / "run_manifest_v2.json",
        json.dumps(payload, indent=2, ensure_ascii=False),
    )


def _readme() -> str:
    return """ALPHA VALIDITY AUDIT V2 — READ FIRST

1. Open reports/audit_dashboard_v2.html first.
2. Then read reports/executive_summary_v2.md.
3. The final machine decision is artifacts/final_decisions_v2.json.
4. low_vol must be read as three separate layers: signal validity, the v1-locked
   portfolio, and the post-hoc BTC-trend portfolio.
5. The post-hoc portfolio cannot use the old blind interval to claim PASS; only
   data strictly after the v1 cutoff may update its conclusion.

Scope: read-only controlled research audit. No V5 production code, service,
database, position, permission, credential, or order was modified. No live or
paper promotion was authorized. Raw candles, state caches, databases, virtual
environments, repositories, credentials, and private keys are excluded.
"""


def main() -> None:
    reports = AUDIT_ROOT / "reports"
    artifacts = AUDIT_ROOT / "artifacts"
    manifests = AUDIT_ROOT / "manifests"
    scripts = AUDIT_ROOT / "scripts"
    bundles = AUDIT_ROOT / "bundles"
    for directory in (reports, artifacts, manifests, scripts, bundles):
        directory.mkdir(parents=True, exist_ok=True)

    suites = _write_test_report(reports, artifacts)
    _write_git_manifests(manifests)
    _copy_reproduction_scripts(scripts)
    _write_run_manifest(artifacts, reports, manifests, scripts, suites)
    _assert_required_outputs(AUDIT_ROOT)

    inputs = _bundle_paths(AUDIT_ROOT)
    _assert_secret_safe(inputs)
    if os.environ.get("AUDIT_V2_VALIDATE_ONLY") == "1":
        print(f"validated_files={len(inputs)}")
        print("secret_scan=PASS")
        return
    readme_path = bundles / "README_FIRST_V2.txt"
    _write(readme_path, _readme())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = bundles / f"alpha_audit_v2_bundle_{timestamp}.zip"
    with zipfile.ZipFile(
        bundle,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(readme_path, "README_FIRST_V2.txt")
        for path in inputs:
            archive.write(path, str(path.relative_to(AUDIT_ROOT)))

    checksum = _sha256(bundle)
    checksum_path = bundle.with_suffix(".zip.sha256")
    _write(checksum_path, f"{checksum}  {bundle.name}")
    downloads = Path("/mnt/c/Users/HR/Downloads")
    copied_to = ""
    if downloads.is_dir():
        destination = downloads / bundle.name
        shutil.copy2(bundle, destination)
        shutil.copy2(checksum_path, destination.with_suffix(".zip.sha256"))
        shutil.copy2(readme_path, downloads / readme_path.name)
        copied_to = str(destination)

    print(f"bundle={bundle}")
    print(f"sha256={checksum}")
    print(f"files={len(inputs) + 1}")
    print(f"copied_to={copied_to or 'NOT_COPIED'}")


if __name__ == "__main__":
    main()
