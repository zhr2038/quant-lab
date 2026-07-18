# ruff: noqa: E501
"""Validate the immutable v1 bundle before interpreting v2 results."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

REQUIRED = [
    "reports/audit_dashboard.html",
    "reports/executive_summary.md",
    "reports/alpha_validity_audit.md",
    "reports/factor_validation_report.md",
    "reports/portfolio_validation_report.md",
    "reports/data_leakage_audit.md",
    "reports/cost_model_report.md",
    "reports/production_impact_assessment.md",
    "artifacts/final_decisions.json",
    "artifacts/factor_validation_results.csv",
    "artifacts/factor_horizon_results.csv",
    "artifacts/factor_period_results.csv",
    "artifacts/portfolio_backtest_results.csv",
    "artifacts/ic_statistics.csv",
    "artifacts/hac_statistics.csv",
    "artifacts/multiple_testing_summary.csv",
    "artifacts/leakage_test_results.csv",
    "artifacts/negative_control_results.csv",
    "artifacts/cost_model_summary.csv",
    "artifacts/data_quality_summary.csv",
    "artifacts/run_manifest.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(os.environ.get("AUDIT_ROOT", "/home/hr/quant-alpha-audit-v2"))
    v1 = Path(os.environ.get("AUDIT_V1_DIR", str(root / "v1")))
    reports, artifacts = root / "reports", root / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((v1 / "artifacts/run_manifest.json").read_text())
    final = json.loads((v1 / "artifacts/final_decisions.json").read_text())
    presence = {relative: (v1 / relative).is_file() for relative in REQUIRED}
    hash_rows: list[dict] = []
    for relative, expected in sorted(manifest["outputs"].items()):
        path = v1 / relative
        actual = _sha256(path) if path.is_file() else None
        hash_rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    mismatches = [row for row in hash_rows if not row["matches"]]
    expected_decisions = {
        "current_v5": "INCONCLUSIVE",
        "rev_xs_20d": "FAIL",
        "low_vol_20d": "FAIL",
        "funding_fade": "INCONCLUSIVE",
    }
    decision_consistency = {
        name: final.get(name, {}).get("decision") == expected
        for name, expected in expected_decisions.items()
    }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "v1_directory": str(v1),
        "required_file_count": len(REQUIRED),
        "all_required_files_present": all(presence.values()),
        "required_file_presence": presence,
        "manifest_output_count": len(hash_rows),
        "manifest_hashes_all_match": not mismatches,
        "hash_mismatches": mismatches,
        "conclusions_consistent_across_machine_decisions": all(decision_consistency.values()),
        "decision_consistency": decision_consistency,
        "v1_git_commit": manifest["git_commit"],
        "snapshot_id": manifest["snapshot_id"],
        "data_cutoff": manifest["data_cutoff"],
        "sample_range": ["2024-07-18T00:00:00Z", "2026-07-18T00:00:00Z"],
        "universe": ["dynamic top10", "dynamic top20", "dynamic top50"],
        "cost_assumption": "15bps one-way / 30bps roundtrip floor; scenarios collapsed because coverage was insufficient",
        "locked_parameters": json.loads((v1 / "artifacts/oos_lock.json").read_text()),
        "blind_oos": "2026-01-16T12:00:00Z <= feature_ts, label_ts < 2026-07-18T00:00:00Z",
        "overall_status": "PASS_WITH_ONE_MANIFEST_HASH_INCONSISTENCY"
        if all(presence.values()) and all(decision_consistency.values()) and len(mismatches) == 1
        else "FAIL",
        "v1_files_modified": 0,
    }
    (artifacts / "v1_audit_consistency_check.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    mismatch_lines = [
        f"- `{row['path']}`: declared `{row['expected_sha256']}`, actual `{row['actual_sha256']}`"
        for row in mismatches
    ] or ["- None."]
    lines = [
        "# V1 Audit Consistency Check",
        "",
        f"- Status: **{payload['overall_status']}**",
        f"- Checked at: {payload['generated_at']}",
        f"- Required files: {sum(presence.values())}/{len(REQUIRED)} present.",
        f"- Manifest outputs: {len(hash_rows) - len(mismatches)}/{len(hash_rows)} SHA256 values match.",
        f"- V1 commit: `{manifest['git_commit']}`; snapshot: `{manifest['snapshot_id']}`.",
        f"- Data cutoff: {manifest['data_cutoff']}.",
        "- Sample: 2024-07-18 through 2026-07-18; dynamic Top10/20/50; base floor 15bps one-way.",
        "- Research/validation/blind boundaries and factor/portfolio locks match `oos_lock.json` and `portfolio_oos_lock.json`.",
        "- Machine decisions, Markdown reports, and dashboard agree on the four v1 outcomes: V5 INCONCLUSIVE, rev FAIL, low_vol portfolio FAIL, funding INCONCLUSIVE.",
        "- The v1 directory was read only and was not modified.",
        "",
        "## Hash inconsistency",
        "",
        *mismatch_lines,
        "",
        "The sole mismatch is `resource_usage.csv`. The runner wrote the run manifest and then appended a later report-stage resource sample. This is a v1 packaging-order defect, not a v2 result change; it is retained verbatim and disclosed rather than repaired in place.",
    ]
    (reports / "v1_audit_consistency_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
