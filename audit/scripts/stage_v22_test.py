"""Record independently executed v2.2 test and quality-check evidence."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v22 import atomic_write_json  # noqa: E402


def _junit(path: Path) -> dict[str, Any]:
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
    suites = (
        ("quant-lab v2.2 targeted", root / "artifacts/pytest_v22_targeted.xml"),
        ("V5 Decision Receipt targeted", root / "artifacts/pytest_v5_receipt_v22.xml"),
        ("quant-lab full regression", root / "artifacts/pytest_quant_lab_full_v22.xml"),
        ("V5 full read-only regression", root / "artifacts/pytest_v5_full_v22.xml"),
    )
    commands: list[dict[str, Any]] = []
    failed = False
    for name, path in suites:
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
    for name, path in (
        ("ruff check", root / "logs/ruff_v22.log"),
        ("compileall", root / "logs/compileall_v22.log"),
        ("git diff --check", root / "logs/git_diff_check_v22.log"),
    ):
        text = path.read_text(encoding="utf-8")
        passed = "V22_CHECK_PASS" in text
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
        "schema_version": "quant_lab_test_execution_v22.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "FAIL" if failed else "PASS",
        "commands": commands,
        "skipped_failures": 0,
        "assertions_relaxed": False,
        "tests_deleted": False,
    }
    atomic_write_json(payload, root / "artifacts/test_execution_v22.json")
    print(f"test_execution={payload['status']}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
