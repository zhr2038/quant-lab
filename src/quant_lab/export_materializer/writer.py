from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_lab.export.daily import export_daily_pack
from quant_lab.export_materializer.validator import validate_export_pack_locally
from quant_lab.export_plane.contracts import (
    ExportPackFile,
    ExportPackManifest,
    ExportSnapshotManifest,
    ExportTask,
    ExportValidationReport,
)
from quant_lab.export_plane.signatures import sha256_file
from quant_lab.export_plane.status import atomic_write_json


@dataclass(frozen=True)
class MaterializationResult:
    pack_path: Path
    pack_manifest: ExportPackManifest
    validation_report: ExportValidationReport
    worker_report: dict[str, Any]


def materialize_snapshot_pack(
    *,
    snapshot_root: str | Path,
    task: ExportTask,
    snapshot: ExportSnapshotManifest,
    work_root: str | Path,
    worker_id: str,
    worker_commit: str,
) -> MaterializationResult:
    if worker_commit != task.expected_worker_commit:
        raise RuntimeError("worker_code_mismatch")
    root = Path(snapshot_root)
    lake_root = root / "lake"
    if not lake_root.is_dir():
        raise FileNotFoundError(f"snapshot lake is missing: {lake_root}")
    work = Path(work_root)
    work.mkdir(parents=True, exist_ok=True)
    staging = work / "staging"
    output = work / "output"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    staging.mkdir(parents=True)
    output.mkdir(parents=True)
    started = time.perf_counter()
    with _temporary_env("QUANT_LAB_EXPORT_ACTIVE_STAGING_DIR", str(staging)):
        result = export_daily_pack(
            export_date=task.export_date,
            lake_root=lake_root,
            out_dir=output,
            profile="expert",
            command_line=["quant-export-worker", task.task_id],
            refresh_risk_permission=False,
            pre_export_v5_refresh=False,
            allow_stale_v5=False,
            expected_v5_bundle_sha256=task.selected_v5_bundle_sha256,
            acceptance_set_id=task.acceptance_set_id,
            materialization_mode=True,
            member_staging_dir=staging,
            source_snapshot_id=snapshot.snapshot_id,
            sealed_acceptance_context=_sealed_acceptance_context(snapshot),
        )
    pack_path = Path(result.zip_path)
    pack_sha = sha256_file(pack_path)
    pack_id = f"expert-pack-{pack_sha[:24]}"
    generated_at = datetime.now(UTC)
    files = _pack_files(pack_path)
    pack_manifest = ExportPackManifest(
        pack_id=pack_id,
        task_id=task.task_id,
        snapshot_id=snapshot.snapshot_id,
        export_date=task.export_date,
        generated_at=generated_at,
        quant_lab_commit=task.quant_lab_commit,
        worker_commit=worker_commit,
        selected_v5_bundle_sha256=task.selected_v5_bundle_sha256,
        acceptance_set_id=task.acceptance_set_id,
        files=files,
    )
    materialize_elapsed = time.perf_counter() - started
    validation_started = time.perf_counter()
    validation = validate_export_pack_locally(
        pack_path,
        task=task,
        snapshot=snapshot,
        pack_id=pack_id,
    )
    elapsed = time.perf_counter() - started
    worker_report = {
        "schema_version": "quant_lab_export_worker_report.v1",
        "task_id": task.task_id,
        "snapshot_id": snapshot.snapshot_id,
        "worker_id": worker_id,
        "worker_commit": worker_commit,
        "total_elapsed_seconds": round(elapsed, 3),
        "materialize_seconds": round(materialize_elapsed, 3),
        "validation_seconds": round(time.perf_counter() - validation_started, 3),
        "peak_rss_bytes": _peak_rss_bytes(),
        "input_files": len(snapshot.files),
        "input_bytes": snapshot.total_input_bytes,
        "members_generated": len(files),
        "members_reused": 0,
        "pack_size_bytes": pack_path.stat().st_size,
        "cache_hit_count": 0,
        "warnings": list(result.warnings),
        "live_order_effect": "none_read_only_nas_materialization",
    }
    atomic_write_json(work / "pack_manifest.json", pack_manifest.model_dump(mode="json"))
    atomic_write_json(work / "validation_report.json", validation.model_dump(mode="json"))
    atomic_write_json(work / "worker_report.json", worker_report)
    return MaterializationResult(
        pack_path=pack_path,
        pack_manifest=pack_manifest,
        validation_report=validation,
        worker_report=worker_report,
    )


def _sealed_acceptance_context(snapshot: ExportSnapshotManifest) -> dict[str, Any]:
    context = {
        "quant_lab_production_commit": snapshot.quant_lab_commit,
        "quant_lab_current_main_commit": snapshot.quant_lab_current_main_commit,
        "current_main_production_relationship": (
            snapshot.current_main_production_relationship
        ),
        "proposal_snapshot_id": snapshot.proposal_snapshot_id,
        "proposal_snapshot_sha256": snapshot.proposal_snapshot_sha256,
        "proposal_content_snapshot_id": snapshot.proposal_content_snapshot_id,
        "proposal_content_snapshot_sha256": (
            snapshot.proposal_content_snapshot_sha256
        ),
        "snapshot_generated_at": snapshot.snapshot_generated_at,
        "v5_observed_proposal_snapshot_id": (
            snapshot.v5_observed_proposal_snapshot_id
        ),
        "v5_observed_proposal_snapshot_sha256": (
            snapshot.v5_observed_proposal_snapshot_sha256
        ),
        "v5_observed_proposal_content_snapshot_id": (
            snapshot.v5_observed_proposal_content_snapshot_id
        ),
        "v5_observed_proposal_content_snapshot_sha256": (
            snapshot.v5_observed_proposal_content_snapshot_sha256
        ),
        "selected_v5_bundle_built_at": snapshot.selected_v5_bundle_built_at,
    }
    missing = [key for key, value in context.items() if value in (None, "")]
    if missing:
        raise RuntimeError(
            "sealed_snapshot_acceptance_context_missing:" + ",".join(sorted(missing))
        )
    return context


def _pack_files(path: Path) -> list[ExportPackFile]:
    with zipfile.ZipFile(path) as archive:
        manifest_bytes = _bounded_zip_member(archive, "manifest.json", 8 * 1024 * 1024)
        manifest = json.loads(manifest_bytes)
        rows = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("pack_manifest_files_missing")
        infos = {info.filename: info for info in archive.infolist()}
        expected = {
            str(row.get("path") or ""): row for row in rows if isinstance(row, dict)
        }
        if set(expected) != set(infos):
            raise RuntimeError("pack_manifest_member_set_mismatch")
        result: list[ExportPackFile] = []
        for name, row in sorted(expected.items()):
            digest = str(row.get("sha256") or "").lower()
            if name == "manifest.json" and len(digest) != 64:
                digest = hashlib.sha256(manifest_bytes).hexdigest()
            result.append(
                ExportPackFile(
                    path=name,
                    sha256=digest,
                    size_bytes=infos[name].file_size,
                    row_count=row.get("rows") if isinstance(row.get("rows"), int) else None,
                )
            )
    return result


def _bounded_zip_member(archive: zipfile.ZipFile, name: str, limit: int) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > limit:
        raise RuntimeError(f"pack_metadata_member_too_large:{name}")
    with archive.open(info) as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise RuntimeError(f"pack_metadata_member_too_large:{name}")
    return value


def _peak_rss_bytes() -> int:
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if os.name == "posix" and sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return 0


@contextmanager
def _temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
