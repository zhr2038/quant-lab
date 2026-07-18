"""Independently validate the immutable Audit v2.1 bundle before v2.2 work."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v22 import atomic_write_json  # noqa: E402

REQUIRED = (
    "reports/audit_dashboard_v21.html",
    "reports/executive_summary_v21.md",
    "reports/forward_runner_fix_report.md",
    "reports/funding_window_correction_v21.md",
    "reports/test_report_v21.md",
    "artifacts/final_decisions_v21.json",
    "artifacts/forward_v21_decisions.parquet",
    "artifacts/forward_v21_trades.parquet",
    "artifacts/forward_v21_events.parquet",
    "artifacts/forward_v21_equity.parquet",
    "artifacts/forward_v21_benchmarks.parquet",
    "artifacts/forward_v21_status.json",
    "manifests/parameter_lock_v21.json",
    "manifests/forward_v21_cutoff.json",
    "manifests/code_hashes_v21.txt",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _json(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    payload = json.loads(zf.read(name))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} is not a JSON object")
    return payload


def check(bundle: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    bundle_sha256 = _sha256_file(bundle)
    with zipfile.ZipFile(bundle) as zf:
        bad_crc = zf.testzip()
        if bad_crc:
            errors.append(f"ZIP_CRC_ERROR:{bad_crc}")
        names = zf.namelist()
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            errors.append(f"DUPLICATE_PATHS:{duplicates}")
        unsafe = sorted(name for name in names if not _safe_name(name))
        if unsafe:
            errors.append(f"UNSAFE_PATHS:{unsafe}")
        missing = sorted(set(REQUIRED) - set(names))
        if missing:
            errors.append(f"MISSING_REQUIRED:{missing}")

        manifest = _json(zf, "artifacts/run_manifest_v21.json")
        managed = {str(item["path"]): item for item in manifest.get("files", [])}
        actual_managed = set(names) - {"artifacts/run_manifest_v21.json"}
        if set(managed) != actual_managed:
            errors.append("MANIFEST_FILE_SET_MISMATCH")
        hash_mismatches: list[str] = []
        size_mismatches: list[str] = []
        for name, expected in managed.items():
            if name not in names:
                continue
            raw = zf.read(name)
            if len(raw) != int(expected["size_bytes"]):
                size_mismatches.append(name)
            if _sha256_bytes(raw) != str(expected["sha256"]):
                hash_mismatches.append(name)
        if hash_mismatches:
            errors.append(f"SHA256_MISMATCH:{hash_mismatches}")
        if size_mismatches:
            errors.append(f"SIZE_MISMATCH:{size_mismatches}")

        lock = _json(zf, "manifests/parameter_lock_v21.json")
        cutoff = _json(zf, "manifests/forward_v21_cutoff.json")
        status = _json(zf, "artifacts/forward_v21_status.json")
        decisions = pl.read_parquet(io.BytesIO(zf.read("artifacts/forward_v21_decisions.parquet")))
        trades = pl.read_parquet(io.BytesIO(zf.read("artifacts/forward_v21_trades.parquet")))
        events = pl.read_parquet(io.BytesIO(zf.read("artifacts/forward_v21_events.parquet")))
        final = _json(zf, "artifacts/final_decisions_v21.json")
        dashboard = zf.read("reports/audit_dashboard_v21.html").decode("utf-8")
        executive = zf.read("reports/executive_summary_v21.md").decode("utf-8")

        expected_verdicts = {
            "rev_xs_20d": ("FAIL", "FAIL", "FAIL"),
            "funding_fade": ("INCONCLUSIVE", "INCONCLUSIVE", "INCONCLUSIVE"),
        }
        for factor, expected in expected_verdicts.items():
            row = final.get(factor, {})
            actual = (
                row.get("signal_validity"),
                row.get("portfolio_validity"),
                row.get("deployment_readiness"),
            )
            if actual != expected:
                errors.append(f"FINAL_DECISION_MISMATCH:{factor}:{actual}")
        low_vol = final.get("low_vol_20d", {})
        if (
            low_vol.get("signal_validity") != "PASS"
            or low_vol.get("locked_portfolio_validity") != "FAIL"
            or low_vol.get("deployment_readiness") != "INCONCLUSIVE"
        ):
            errors.append("FINAL_DECISION_MISMATCH:low_vol_20d")
        post_hoc = final.get("low_vol_btc_trend", {})
        if (
            post_hoc.get("hypothesis_type") != "POST_HOC_HYPOTHESIS"
            or post_hoc.get("portfolio_validity") != "INCONCLUSIVE"
            or post_hoc.get("deployment_readiness") != "INCONCLUSIVE"
        ):
            errors.append("FINAL_DECISION_MISMATCH:low_vol_btc_trend")
        current_v5 = final.get("current_v5", {})
        if (
            current_v5.get("replayability") != "PARTIALLY_REPLAYABLE"
            or current_v5.get("deployment_readiness") != "FAIL"
        ):
            errors.append("FINAL_DECISION_MISMATCH:current_v5")

        if manifest.get("audit_version") != "v2.1":
            errors.append("AUDIT_VERSION_MISMATCH")
        if manifest.get("branch") != "audit/alpha-validity-v2.1":
            errors.append("V21_BRANCH_MISMATCH")
        if manifest.get("runner_code_commit") != lock.get("code_commit"):
            errors.append("RUNNER_COMMIT_LOCK_MISMATCH")
        if cutoff.get("code_commit") != lock.get("code_commit"):
            errors.append("CUTOFF_COMMIT_LOCK_MISMATCH")
        if cutoff.get("parameter_lock_hash") != lock.get("sha256"):
            errors.append("CUTOFF_PARAMETER_LOCK_MISMATCH")
        if manifest.get("parameter_lock_hash") != lock.get("sha256"):
            errors.append("MANIFEST_PARAMETER_LOCK_MISMATCH")
        if manifest.get("forward_v21_start_cutoff") != cutoff.get("forward_v21_start_cutoff"):
            errors.append("FORWARD_CUTOFF_MISMATCH")
        if manifest.get("available_market_data_cutoff") != status.get(
            "available_market_data_cutoff"
        ):
            errors.append("DATA_CUTOFF_MISMATCH")
        if status.get("runner_version") != lock.get("runner_version"):
            errors.append("RUNNER_VERSION_MISMATCH")
        if int(status.get("decision_count", -1)) != decisions.height:
            errors.append("DECISION_COUNT_MISMATCH")
        if int(status.get("entry_count", -1)) != trades.height:
            errors.append("TRADE_COUNT_MISMATCH")
        if int(status.get("decision_count", -1)) != 0 or int(status.get("entry_count", -1)) != 0:
            warnings.append("v2.1 contains formal records; v2.2 cutoff reset needs manual review")
        report_text = f"{dashboard}\n{executive}".lower()
        equivalent_markers = {
            "POST_HOC": ("post-hoc", "post_hoc"),
            "PARTIALLY_REPLAYABLE": ("partially_replayable",),
            "FORWARD_INCONCLUSIVE": ("inconclusive", "样本不足"),
        }
        for label, alternatives in equivalent_markers.items():
            if not any(marker in report_text for marker in alternatives):
                errors.append(f"REPORT_MARKER_MISSING:{label}")

    return {
        "schema_version": "quant_lab_v21_consistency_check.v1",
        "checked_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "bundle_path": str(bundle.resolve()),
        "bundle_sha256": bundle_sha256,
        "zip_entry_count": len(names),
        "manifest_managed_file_count": len(managed),
        "crc_ok": bad_crc is None,
        "all_required_files_present": not missing,
        "all_manifest_hashes_match": not hash_mismatches,
        "all_manifest_sizes_match": not size_mismatches,
        "v21_branch": manifest.get("branch"),
        "v21_final_head": manifest.get("final_commit"),
        "v21_runner_commit": manifest.get("runner_code_commit"),
        "parameter_lock_commit": lock.get("code_commit"),
        "parameter_lock_hash": lock.get("sha256"),
        "forward_cutoff": cutoff.get("forward_v21_start_cutoff"),
        "data_cutoff": status.get("available_market_data_cutoff"),
        "runner_version": status.get("runner_version"),
        "formal_decision_count": decisions.height,
        "formal_trade_count": trades.height,
        "event_count": events.height,
        "v21_forward_status": status.get("conclusion"),
        "supersession_status": "SUPERSEDED_BY_V22",
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL",
    }


def report(payload: dict[str, Any]) -> str:
    checks = [
        ("ZIP CRC", payload["crc_ok"]),
        ("Required files", payload["all_required_files_present"]),
        ("Manifest SHA256", payload["all_manifest_hashes_match"]),
        ("Manifest sizes", payload["all_manifest_sizes_match"]),
        ("Cross-format conclusions", not payload["errors"]),
    ]
    rows = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks
    )
    errors = "\n".join(f"- {item}" for item in payload["errors"]) or "- None"
    warnings = "\n".join(f"- {item}" for item in payload["warnings"]) or "- None"
    return f"""# Audit v2.1 Consistency Check

- Status: **{payload['status']}**
- Bundle SHA256: `{payload['bundle_sha256']}`
- v2.1 final HEAD: `{payload['v21_final_head']}`
- Runner/lock commit: `{payload['v21_runner_commit']}`
- Forward cutoff: `{payload['forward_cutoff']}`
- Data cutoff: `{payload['data_cutoff']}`
- Runner version: `{payload['runner_version']}`

| Check | Result |
|---|---|
{rows}

- Formal decisions: **{payload['formal_decision_count']}**
- Formal trades: **{payload['formal_trade_count']}**
- v2.1 forward status: **{payload['v21_forward_status']}**

v2.1 files remain immutable and are marked **SUPERSEDED_BY_V22**. They are not
deleted, rewritten, or merged into v2.2 evidence.

## Errors

{errors}

## Warnings

{warnings}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    payload = check(args.bundle.resolve())
    root = args.root.resolve()
    atomic_write_json(payload, root / "artifacts/v21_consistency_check.json")
    path = root / "reports/v21_consistency_check.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report(payload), encoding="utf-8")
    print(f"v21_consistency={payload['status']}")
    print(f"formal_decision_count={payload['formal_decision_count']}")
    print(f"formal_trade_count={payload['formal_trade_count']}")
    if payload["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
