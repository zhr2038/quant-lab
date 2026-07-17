from __future__ import annotations

import csv
import hashlib
import io
import json
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from quant_lab.export import daily as daily_export
from quant_lab.export_plane.contracts import (
    ExportSnapshotManifest,
    ExportTask,
    ExportValidationReport,
)
from quant_lab.export_plane.signatures import sha256_file

DERIVED_CSV_MEMBERS = (
    "reports/api_auth_production_slo.csv",
    "reports/paper_runtime_freshness.csv",
    "reports/paper_proposal_propagation_status.csv",
)
DERIVED_JSON_MEMBER = "reports/system_acceptance_complete_status.json"
REQUIRED_MEMBERS = {
    "manifest.json",
    "provenance.json",
    "data_quality.json",
    "README.md",
    "executive_summary.md",
    "expert_questions.md",
    "diagnostics/export_timing.csv",
    "diagnostics/export_timing.json",
    *DERIVED_CSV_MEMBERS,
    DERIVED_JSON_MEMBER,
}
MAX_MEMBERS = 20_000
MAX_MEMBER_BYTES = 4 * 1024**3
MAX_UNCOMPRESSED_BYTES = 20 * 1024**3
MAX_COMPRESSION_RATIO = 2_000.0
MAX_METADATA_MEMBER_BYTES = 8 * 1024**2


def validate_export_pack_locally(
    pack_path: str | Path,
    *,
    task: ExportTask,
    snapshot: ExportSnapshotManifest,
    pack_id: str,
) -> ExportValidationReport:
    path = Path(pack_path)
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    total_uncompressed = 0
    peak_ratio = 0.0
    names: list[str] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        checks["member_count"] = len(infos) <= MAX_MEMBERS
        checks["no_duplicate_members"] = len(names) == len(set(names))
        checks["safe_member_paths"] = all(_safe_member_name(name) for name in names)
        checks["no_symlinks"] = all(not _zipinfo_is_symlink(item) for item in infos)
        checks["required_members"] = REQUIRED_MEMBERS.issubset(names)
        for info in infos:
            total_uncompressed += info.file_size
            ratio = info.file_size / max(info.compress_size, 1)
            peak_ratio = max(peak_ratio, ratio)
            if info.file_size > MAX_MEMBER_BYTES:
                failures.append(f"member_too_large:{info.filename}")
            if ratio > MAX_COMPRESSION_RATIO:
                failures.append(f"compression_ratio_exceeded:{info.filename}")
        checks["total_uncompressed_size"] = total_uncompressed <= MAX_UNCOMPRESSED_BYTES
        checks["member_size_limits"] = not any(
            item.startswith("member_too_large:") for item in failures
        )
        checks["compression_ratio_limits"] = not any(
            item.startswith("compression_ratio_exceeded:") for item in failures
        )
        manifest = _json_member(archive, "manifest.json")
        checks["manifest_schema"] = isinstance(manifest, dict) and bool(manifest.get("files"))
        checks["snapshot_id"] = manifest.get("export_snapshot_id") == snapshot.snapshot_id
        checks["quant_lab_commit"] = manifest.get("quant_lab_commit") == task.quant_lab_commit
        checks["v5_bundle_sha"] = (
            str(manifest.get("selected_v5_bundle_sha256") or "").lower()
            == task.selected_v5_bundle_sha256
        )
        checks["acceptance_set_id"] = manifest.get("acceptance_set_id") == task.acceptance_set_id
        checks["authoritative_input"] = bool(manifest.get("authoritative_snapshot"))
        checks.update(
            _verify_derived_reports(
                archive,
                manifest=manifest,
                task=task,
                snapshot=snapshot,
                failures=failures,
            )
        )
        member_checks = _verify_manifest_members(
            archive,
            manifest,
            set(names),
            failures,
        )
        checks.update(member_checks)
    for name, passed in checks.items():
        if not passed:
            failures.append(f"check_failed:{name}")
    return ExportValidationReport(
        pack_id=pack_id,
        task_id=task.task_id,
        snapshot_id=snapshot.snapshot_id,
        validated_at=datetime.now(UTC),
        valid=not failures,
        checks=checks,
        failures=sorted(set(failures)),
        warnings=warnings,
        zip_sha256=sha256_file(path),
        zip_size_bytes=path.stat().st_size,
        member_count=len(names),
        total_uncompressed_bytes=total_uncompressed,
        peak_compression_ratio=round(peak_ratio, 3),
    )


def _safe_member_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(
        normalized
        and not path.is_absolute()
        and ".." not in path.parts
        and not normalized.startswith("/")
        and all(part not in {"", "."} for part in path.parts)
    )


def _zipinfo_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


def _json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        info = archive.getinfo(name)
        if info.file_size > MAX_METADATA_MEMBER_BYTES:
            return {}
        with archive.open(info) as handle:
            raw = handle.read(MAX_METADATA_MEMBER_BYTES + 1)
        if len(raw) > MAX_METADATA_MEMBER_BYTES:
            return {}
        value = json.loads(raw.decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
        return {}
    return value if isinstance(value, dict) else {}


def _csv_member_rows(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    try:
        info = archive.getinfo(name)
        if info.file_size > MAX_METADATA_MEMBER_BYTES:
            return []
        with archive.open(info) as handle:
            raw = handle.read(MAX_METADATA_MEMBER_BYTES + 1)
        if len(raw) > MAX_METADATA_MEMBER_BYTES:
            return []
        return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    except (KeyError, UnicodeDecodeError, csv.Error, RuntimeError):
        return []


def _verify_derived_reports(
    archive: zipfile.ZipFile,
    *,
    manifest: dict[str, Any],
    task: ExportTask,
    snapshot: ExportSnapshotManifest,
    failures: list[str],
) -> dict[str, bool]:
    reports: list[tuple[str, dict[str, Any]]] = []
    parseable = True
    for name in DERIVED_CSV_MEMBERS:
        rows = _csv_member_rows(archive, name)
        if not rows:
            failures.append(f"derived_report_missing_or_empty:{name}")
            parseable = False
            continue
        reports.extend((name, row) for row in rows)
    complete_status = _json_member(archive, DERIVED_JSON_MEMBER)
    if not complete_status:
        failures.append(f"derived_report_missing_or_empty:{DERIVED_JSON_MEMBER}")
        parseable = False
    else:
        reports.append((DERIVED_JSON_MEMBER, complete_status))

    expected_bundle_sha = task.selected_v5_bundle_sha256.lower()
    source_matches = parseable and all(
        str(row.get("source_bundle_sha256") or "").strip().lower() == expected_bundle_sha
        for _, row in reports
    )
    if not source_matches:
        for name, row in reports:
            if str(row.get("source_bundle_sha256") or "").strip().lower() != expected_bundle_sha:
                failures.append(f"derived_report_source_bundle_mismatch:{name}")

    identity_pairs = (
        ("proposal_snapshot_id", snapshot.proposal_snapshot_id, False),
        ("proposal_snapshot_sha256", snapshot.proposal_snapshot_sha256, True),
        ("proposal_content_snapshot_id", snapshot.proposal_content_snapshot_id, False),
        (
            "proposal_content_snapshot_sha256",
            snapshot.proposal_content_snapshot_sha256,
            True,
        ),
    )
    identity_matches = parseable and all(
        _derived_identity_matches(row, identity_pairs) for _, row in reports
    )
    if not identity_matches:
        for name, row in reports:
            if not _derived_identity_matches(row, identity_pairs):
                failures.append(f"derived_report_snapshot_identity_mismatch:{name}")

    manifest_generated_at = _parse_utc(manifest.get("generated_at"))
    fresh = bool(parseable and manifest_generated_at is not None)
    for name, row in reports:
        report_generated_at = _parse_utc(row.get("generated_at"))
        age_seconds = (
            (manifest_generated_at - report_generated_at).total_seconds()
            if manifest_generated_at is not None and report_generated_at is not None
            else None
        )
        row_fresh = bool(
            str(row.get("derived_report_status") or "").strip().upper() == "FRESH"
            and age_seconds is not None
            and -300 <= age_seconds <= daily_export.DERIVED_REPORT_MAX_AGE_SECONDS
        )
        if not row_fresh:
            failures.append(f"derived_report_stale:{name}")
            fresh = False

    return {
        "derived_reports_parseable": parseable,
        "derived_reports_source_v5_bundle": source_matches,
        "derived_reports_snapshot_identity": identity_matches,
        "derived_reports_fresh": fresh,
    }


def _derived_identity_matches(
    row: dict[str, Any],
    identity_pairs: tuple[tuple[str, str | None, bool], ...],
) -> bool:
    for field, expected, normalize_lower in identity_pairs:
        if expected is None:
            continue
        observed = str(row.get(field) or "").strip()
        expected_text = str(expected).strip()
        if normalize_lower:
            observed = observed.lower()
            expected_text = expected_text.lower()
        if observed != expected_text:
            return False
    return True


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _verify_manifest_members(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
    names: set[str],
    failures: list[str],
) -> dict[str, bool]:
    rows = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    expected = {str(row.get("path") or ""): row for row in rows if isinstance(row, dict)}
    if set(expected) != names:
        failures.append("manifest_member_set_mismatch")
        return {
            "manifest_members": False,
            "member_sha256": False,
            "member_row_counts": False,
            "secret_scan": False,
            "crc": False,
        }
    sha_valid = True
    row_counts_valid = True
    secrets_valid = True
    crc_valid = True
    for name, row in expected.items():
        expected_sha = str(row.get("sha256") or "").lower()
        digest = hashlib.sha256()
        newline_count = 0
        try:
            with archive.open(name) as handle:
                if daily_export._is_text_member(name):  # noqa: SLF001
                    for raw_line in handle:
                        digest.update(raw_line)
                        newline_count += raw_line.count(b"\n")
                        line = raw_line.decode("utf-8", "replace")
                        if daily_export._line_may_contain_secret(line):  # noqa: SLF001
                            high, medium = daily_export._secret_severity_counts(line)  # noqa: SLF001
                            if high or medium:
                                failures.append(f"possible_secret:{name}")
                                secrets_valid = False
                else:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        except (OSError, RuntimeError, zipfile.BadZipFile):
            failures.append(f"member_crc_error:{name}")
            crc_valid = False
            sha_valid = False
            continue
        if expected_sha and digest.hexdigest() != expected_sha:
            failures.append(f"member_sha_mismatch:{name}")
            sha_valid = False
        expected_rows = row.get("rows")
        if name.endswith(".csv") and isinstance(expected_rows, int):
            observed_rows = max(newline_count - 1, 0)
            if observed_rows != expected_rows:
                failures.append(f"member_row_count_mismatch:{name}")
                row_counts_valid = False
    return {
        "manifest_members": True,
        "member_sha256": sha_valid,
        "member_row_counts": row_counts_valid,
        "secret_scan": secrets_valid,
        "crc": crc_valid,
    }
