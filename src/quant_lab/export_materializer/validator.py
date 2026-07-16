from __future__ import annotations

import hashlib
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

REQUIRED_MEMBERS = {
    "manifest.json",
    "provenance.json",
    "data_quality.json",
    "README.md",
    "executive_summary.md",
    "expert_questions.md",
    "diagnostics/export_timing.csv",
    "diagnostics/export_timing.json",
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
