from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_lab.export_materializer.writer import MaterializationResult
from quant_lab.export_plane.contracts import (
    ExportPackIndexEntry,
    ExportSnapshotManifest,
    ExportTask,
    ExportWorkerReceipt,
)
from quant_lab.export_plane.signatures import (
    load_signing_key,
    sha256_file,
    sign_payload,
)
from quant_lab.export_plane.status import atomic_write_json


def accept_materialized_pack(
    *,
    result: MaterializationResult,
    task: ExportTask,
    snapshot: ExportSnapshotManifest,
    accepted_root: str | Path,
    index_path: str | Path,
    worker_id: str,
    worker_signing_key_path: str | Path,
    worker_key_id: str,
    cache_hits: int,
    downloaded_bytes: int,
) -> tuple[ExportWorkerReceipt, Path]:
    if not result.validation_report.valid:
        raise RuntimeError("pack_validation_failed")
    accepted = Path(accepted_root)
    accepted.mkdir(parents=True, exist_ok=True)
    pack_sha = result.validation_report.zip_sha256
    pack_id = result.pack_manifest.pack_id
    final_dir = accepted / f"{task.export_date:%Y/%m/%d}" / pack_id
    if final_dir.is_dir():
        receipt = ExportWorkerReceipt.model_validate_json(
            (final_dir / "receipt.json").read_text(encoding="utf-8")
        )
        if receipt.pack_sha256 != pack_sha:
            raise RuntimeError("accepted_pack_id_sha256_conflict")
        return receipt, final_dir

    incoming_parent = accepted / ".incoming"
    incoming_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{task.task_id}.", dir=incoming_parent))
    try:
        pack_name = result.pack_path.name
        pack_target = temporary / pack_name
        _stream_copy(result.pack_path, pack_target)
        if sha256_file(pack_target) != pack_sha:
            raise RuntimeError("accepted_copy_sha256_mismatch")
        worker_report = {
            **result.worker_report,
            "cache_hit_count": cache_hits,
            "downloaded_input_bytes": downloaded_bytes,
        }
        pack_manifest_path = atomic_write_json(
            temporary / "pack_manifest.json",
            result.pack_manifest.model_dump(mode="json"),
        )
        validation_path = atomic_write_json(
            temporary / "validation_report.json",
            result.validation_report.model_dump(mode="json"),
        )
        atomic_write_json(temporary / "worker_report.json", worker_report)
        atomic_write_json(
            temporary / "snapshot_manifest.json",
            snapshot.model_dump(mode="json"),
        )
        summaries = _bounded_pack_summaries(pack_target)
        accepted_at = datetime.now(UTC)
        provisional = ExportWorkerReceipt(
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            worker_id=worker_id,
            worker_commit=task.expected_worker_commit,
            pack_id=pack_id,
            pack_name=pack_name,
            pack_sha256=pack_sha,
            pack_size_bytes=pack_target.stat().st_size,
            pack_manifest_sha256=sha256_file(pack_manifest_path),
            pack_state="accepted",
            nas_artifact_validated=True,
            validation_report_sha256=sha256_file(validation_path),
            authoritative_input_snapshot=True,
            selected_v5_bundle_sha256=task.selected_v5_bundle_sha256,
            acceptance_set_id=task.acceptance_set_id,
            download_relative_path=(
                f"{task.export_date:%Y/%m/%d}/{pack_id}/{pack_name}"
            ),
            generated_at=result.pack_manifest.generated_at,
            accepted_at=accepted_at,
            manifest_summary=summaries["manifest_summary"],
            data_quality_summary=summaries["data_quality_summary"],
            expert_questions=summaries["expert_questions"],
            validation_summary={
                "valid": True,
                "member_count": result.validation_report.member_count,
                "total_uncompressed_bytes": result.validation_report.total_uncompressed_bytes,
                "failure_count": 0,
            },
            worker_report_summary={
                key: worker_report.get(key)
                for key in (
                    "total_elapsed_seconds",
                    "materialize_seconds",
                    "validation_seconds",
                    "peak_rss_bytes",
                    "input_files",
                    "input_bytes",
                    "members_generated",
                    "cache_hit_count",
                    "downloaded_input_bytes",
                )
            },
            signature_key_id=worker_key_id,
            signature="A" * 88,
        )
        receipt = ExportWorkerReceipt.model_validate(
            {
                **provisional.model_dump(mode="json"),
                "signature": sign_payload(
                    provisional,
                    load_signing_key(worker_signing_key_path),
                ),
            }
        )
        atomic_write_json(temporary / "receipt.json", receipt.model_dump(mode="json"))
        _set_accepted_permissions(temporary)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final_dir)
        _update_index(index_path, receipt, task.export_date)
        return receipt, final_dir
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _stream_copy(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _set_accepted_permissions(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o550 if path.is_dir() else 0o440)
    root.chmod(0o550)


def _bounded_pack_summaries(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        manifest = _read_bounded_json(archive, "manifest.json")
        quality = _read_bounded_json(archive, "data_quality.json")
        questions = _read_bounded_text(archive, "expert_questions.md")
    manifest_keys = (
        "export_date",
        "generated_at",
        "quant_lab_commit",
        "selected_v5_bundle_sha256",
        "acceptance_set_id",
        "authoritative_snapshot",
        "export_snapshot_id",
    )
    quality_keys = (
        "status",
        "critical_count",
        "warning_count",
        "stale_dataset_count",
        "missing_dataset_count",
    )
    return {
        "manifest_summary": {key: manifest.get(key) for key in manifest_keys},
        "data_quality_summary": {key: quality.get(key) for key in quality_keys},
        "expert_questions": [
            line.lstrip("-0123456789. ").strip()
            for line in questions.splitlines()
            if line.strip().startswith(("-", "1.", "2.", "3.", "4.", "5."))
        ][:20],
    }


def _read_bounded_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    text = _read_bounded_text(archive, name)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_bounded_text(archive: zipfile.ZipFile, name: str, limit: int = 256 * 1024) -> str:
    try:
        with archive.open(name) as handle:
            data = handle.read(limit + 1)
    except KeyError:
        return ""
    if len(data) > limit:
        raise RuntimeError(f"receipt_summary_member_too_large:{name}")
    return data.decode("utf-8", "replace")


def mark_control_plane_receipt_verified(index_path: str | Path, pack_id: str) -> None:
    path = Path(index_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [ExportPackIndexEntry.model_validate(row) for row in payload.get("packs", [])]
    updated: list[ExportPackIndexEntry] = []
    found = False
    for row in rows:
        if row.pack_id == pack_id:
            row = row.model_copy(update={"control_plane_receipt_verified": True})
            found = True
        updated.append(row)
    if not found:
        raise RuntimeError("accepted_pack_missing_from_index")
    atomic_write_json(
        path,
        {
            "schema_version": "quant_lab_export_accepted_index.v1",
            "updated_at": datetime.now(UTC).isoformat(),
            "packs": [item.model_dump(mode="json") for item in updated],
        },
        mode=0o640,
    )


def _update_index(
    index_path: str | Path,
    receipt: ExportWorkerReceipt,
    export_date,
) -> None:
    path = Path(index_path)
    rows: list[ExportPackIndexEntry] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [ExportPackIndexEntry.model_validate(row) for row in payload.get("packs", [])]
    except (OSError, ValueError, TypeError):
        rows = []
    row = ExportPackIndexEntry(
        pack_id=receipt.pack_id,
        task_id=receipt.task_id,
        pack_name=receipt.pack_name,
        export_date=export_date,
        generated_at=receipt.generated_at,
        accepted_at=receipt.accepted_at,
        pack_sha256=receipt.pack_sha256,
        pack_size_bytes=receipt.pack_size_bytes,
        snapshot_id=receipt.snapshot_id,
        authoritative_input_snapshot=True,
        nas_artifact_validated=True,
        control_plane_receipt_verified=False,
        download_ready=True,
        download_relative_path=receipt.download_relative_path,
        selected_v5_bundle_sha256=receipt.selected_v5_bundle_sha256,
        acceptance_set_id=receipt.acceptance_set_id,
        worker_id=receipt.worker_id,
        worker_commit=receipt.worker_commit,
        pack_state="accepted",
        manifest_summary=receipt.manifest_summary,
        data_quality_summary=receipt.data_quality_summary,
        expert_questions=receipt.expert_questions,
        validation_summary=receipt.validation_summary,
        worker_report_summary=receipt.worker_report_summary,
    )
    deduped = {item.pack_id: item for item in rows}
    deduped[row.pack_id] = row
    atomic_write_json(
        path,
        {
            "schema_version": "quant_lab_export_accepted_index.v1",
            "updated_at": datetime.now(UTC).isoformat(),
            "packs": [
                item.model_dump(mode="json")
                for item in sorted(
                    deduped.values(),
                    key=lambda value: value.accepted_at,
                    reverse=True,
                )
            ],
        },
        mode=0o640,
    )
