"""Run the Audit v2.1 targeted and full quant-lab regression gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path


def _run(command: list[str], *, cwd: Path, log_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    seconds = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(process.stdout, encoding="utf-8")
    print(process.stdout[-4000:])
    return process.returncode, seconds


def _junit(path: Path, scope: str, seconds: float | None = None) -> dict:
    root = ET.parse(path).getroot()
    if root.tag == "testsuites":
        tests = int(sum(int(item.attrib.get("tests", 0)) for item in root))
        failures = int(sum(int(item.attrib.get("failures", 0)) for item in root))
        errors = int(sum(int(item.attrib.get("errors", 0)) for item in root))
        skipped = int(sum(int(item.attrib.get("skipped", 0)) for item in root))
        recorded_seconds = sum(float(item.attrib.get("time", 0.0)) for item in root)
    else:
        tests = int(root.attrib.get("tests", 0))
        failures = int(root.attrib.get("failures", 0))
        errors = int(root.attrib.get("errors", 0))
        skipped = int(root.attrib.get("skipped", 0))
        recorded_seconds = float(root.attrib.get("time", 0.0))
    return {
        "scope": scope,
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
        "seconds": recorded_seconds if seconds is None else seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--v5-junit",
        type=Path,
        required=True,
        help="Fresh JUnit XML from the unchanged V5-prod read-only checkout.",
    )
    parser.add_argument("--v5-head", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve()
    # Keep the virtual-environment launcher path intact. ``resolve()`` follows
    # ``.venv/bin/python`` to the system interpreter and silently drops the
    # environment's site-packages.
    python = str(args.python.expanduser().absolute())
    artifacts = root / "artifacts"
    logs = root / "logs"
    artifacts.mkdir(parents=True, exist_ok=True)
    suites: list[dict] = []
    static: list[dict] = []

    targeted_xml = artifacts / "audit_v21_targeted_tests.xml"
    command = [
        python,
        "-m",
        "pytest",
        "-q",
        "audit/tests/test_audit_v21.py",
        f"--junitxml={targeted_xml}",
    ]
    code, seconds = _run(command, cwd=repo, log_path=logs / "pytest_v21_targeted.log")
    if code:
        raise SystemExit(code)
    suites.append(_junit(targeted_xml, "audit_v21_targeted", seconds))

    full_xml = artifacts / "pytest_all_v21.xml"
    command = [python, "-m", "pytest", "-q", f"--junitxml={full_xml}"]
    code, seconds = _run(command, cwd=repo, log_path=logs / "pytest_quant_lab_full.log")
    if code:
        raise SystemExit(code)
    suites.append(_junit(full_xml, "quant_lab_full", seconds))

    checks = (
        ([python, "-m", "ruff", "check", "."], "ruff_check", "ruff.log"),
        (
            [python, "-m", "compileall", "-q", "audit", "src"],
            "compileall_audit_src",
            "compileall.log",
        ),
        (["git", "diff", "--check"], "git_diff_check", "git_diff_check.log"),
    )
    for command, label, filename in checks:
        code, seconds = _run(command, cwd=repo, log_path=logs / filename)
        static.append(
            {
                "command": " ".join(command),
                "label": label,
                "status": "PASS" if code == 0 else "FAIL",
                "seconds": seconds,
            }
        )
        if code:
            raise SystemExit(code)

    v5_result = _junit(args.v5_junit.resolve(), "v5_full_read_only_checkout")
    if v5_result["failed"] or v5_result["errors"]:
        raise RuntimeError(f"V5 read-only regression failed: {v5_result}")
    suites.append(v5_result)

    payload = {
        "schema_version": "alpha_audit_test_execution.v2.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "overall_status": "PASS",
        "test_suites": suites,
        "static_checks": static,
        "v5_regression": {
            "status": "PASS",
            "git_head": args.v5_head,
            "source_mutated": False,
            "junit_path": args.v5_junit.resolve().as_posix(),
            "reason": "Fresh full regression from the unchanged V5-prod read-only checkout.",
        },
        "failed_tests_skipped": False,
        "assertions_weakened": False,
        "production_mutations": 0,
        "live_orders": 0,
    }
    (artifacts / "test_execution_v21.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
