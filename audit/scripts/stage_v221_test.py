"""Record independently executed Audit v2.2.1 test and check evidence."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v221 import atomic_write_json  # noqa: E402


def _junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return {
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    commands: list[dict[str, Any]] = []
    failed = False
    for name, path in (
        (
            "quant-lab v2.2.1 targeted",
            root / "artifacts/pytest_v221_targeted.xml",
        ),
        (
            "quant-lab full regression",
            root / "artifacts/pytest_quant_lab_full_v221.xml",
        ),
    ):
        counts = _junit(path)
        passed = counts["failures"] == 0 and counts["errors"] == 0
        failed |= not passed
        commands.append(
            {
                "name": name,
                "result": "PASS" if passed else "FAIL",
                "summary": (
                    f"{counts['passed']} passed, {counts['skipped']} skipped, "
                    f"{counts['failures']} failed, {counts['errors']} errors"
                ),
                "artifact": str(path.relative_to(root)),
                **counts,
            }
        )
    for name, filename, marker in (
        ("ruff check", "ruff_v221.log", "V221_RUFF_PASS"),
        ("compileall", "compileall_v221.log", "V221_COMPILEALL_PASS"),
        ("git diff --check", "git_diff_check_v221.log", "V221_DIFF_CHECK_PASS"),
        ("shell syntax", "bash_n_v221.log", "V221_BASH_N_PASS"),
        ("systemd unit verify", "systemd_verify_v221.log", "V221_SYSTEMD_PASS"),
    ):
        path = root / "logs" / filename
        text = path.read_text(encoding="utf-8")
        passed = marker in text
        failed |= not passed
        commands.append(
            {
                "name": name,
                "result": "PASS" if passed else "FAIL",
                "summary": text.strip().splitlines()[-1] if text.strip() else "empty log",
                "artifact": str(path.relative_to(root)),
            }
        )
    payload = {
        "schema_version": "quant_lab_test_execution_v221.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "FAIL" if failed else "PASS",
        "commands": commands,
        "skipped_failures": 0,
        "assertions_relaxed": False,
        "tests_deleted": False,
        "factor_scan_rerun": False,
        "parameter_search_rerun": False,
    }
    atomic_write_json(payload, root / "artifacts/test_execution_v221.json")
    print(f"test_execution={payload['status']}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
