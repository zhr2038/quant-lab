from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_lab.export_plane.cloud_index import load_cloud_index, write_cloud_index
from quant_lab.export_plane.contracts import (
    ExportPackIndexEntry,
    ExportSnapshotManifest,
    ExportTask,
    ExportTaskState,
    ExportTaskStatus,
    ExportWorkerReceipt,
)
from quant_lab.export_plane.signatures import load_public_key, verify_payload
from quant_lab.export_plane.snapshot import verify_snapshot_manifest_digest
from quant_lab.export_plane.status import (
    atomic_write_json,
    ensure_queue_layout,
    find_task_directory,
)


def import_export_receipts(
    *,
    queue_root: str | Path,
    worker_public_key_path: str | Path,
    worker_key_id: str,
    max_receipts: int = 20,
) -> dict[str, Any]:
    root = ensure_queue_layout(queue_root)
    public_key = load_public_key(worker_public_key_path)
    imported: list[str] = []
    rejected: list[dict[str, str]] = []
    index = load_cloud_index(root)
    indexed = {item.pack_id: item for item in index}
    inbox = root / "receipts" / "inbox"
    for receipt_dir in sorted(inbox.iterdir())[: max(1, max_receipts)]:
        if not receipt_dir.is_dir():
            continue
        try:
            receipt = ExportWorkerReceipt.model_validate_json(
                (receipt_dir / "receipt.json").read_text(encoding="utf-8")
            )
            if receipt.signature_key_id != worker_key_id:
                raise ValueError("unknown NAS worker signing key")
            verify_payload(receipt, receipt.signature, public_key)
            task_dir = find_task_directory(root, receipt.task_id)
            if task_dir is None:
                raise ValueError("receipt references an unknown task")
            task = ExportTask.model_validate_json((task_dir / "task.json").read_text())
            manifest = ExportSnapshotManifest.model_validate_json(
                (root / "snapshots" / task.snapshot_id / "manifest.json").read_text()
            )
            verify_snapshot_manifest_digest(manifest)
            _validate_receipt_binding(receipt, task, manifest)
            _release_snapshot_bytes(root, manifest)
            row = ExportPackIndexEntry(
                pack_id=receipt.pack_id,
                task_id=receipt.task_id,
                pack_name=receipt.pack_name,
                export_date=task.export_date,
                generated_at=receipt.generated_at,
                accepted_at=receipt.accepted_at,
                pack_sha256=receipt.pack_sha256,
                pack_size_bytes=receipt.pack_size_bytes,
                snapshot_id=receipt.snapshot_id,
                authoritative_input_snapshot=receipt.authoritative_input_snapshot,
                nas_artifact_validated=receipt.nas_artifact_validated,
                control_plane_receipt_verified=True,
                download_ready=True,
                download_relative_path=receipt.download_relative_path,
                selected_v5_bundle_sha256=receipt.selected_v5_bundle_sha256,
                acceptance_set_id=receipt.acceptance_set_id,
                worker_id=receipt.worker_id,
                worker_commit=receipt.worker_commit,
                manifest_summary=receipt.manifest_summary,
                data_quality_summary=receipt.data_quality_summary,
                expert_questions=receipt.expert_questions,
                validation_summary=receipt.validation_summary,
                worker_report_summary=receipt.worker_report_summary,
            )
            prior = indexed.get(row.pack_id)
            if prior is not None and prior.pack_sha256 != row.pack_sha256:
                raise ValueError("pack_id replayed with a different SHA256")
            indexed[row.pack_id] = row
            now = datetime.now(UTC)
            status = ExportTaskStatus(
                task_id=task.task_id,
                snapshot_id=task.snapshot_id,
                state=ExportTaskState.DOWNLOAD_READY,
                requested_at=task.requested_at,
                updated_at=now,
                heartbeat_at=now,
                worker_id=receipt.worker_id,
                attempt=1,
                max_attempts=task.max_attempts,
                current_stage="download_ready",
                completed_members=int(receipt.worker_report_summary.get("members_generated") or 0),
                total_members=int(receipt.worker_report_summary.get("members_generated") or 0),
                input_bytes=manifest.total_input_bytes,
                output_bytes=receipt.pack_size_bytes,
                nas_pack_id=receipt.pack_id,
                nas_pack_sha256=receipt.pack_sha256,
                nas_download_path=receipt.download_relative_path,
            )
            atomic_write_json(
                root / "status" / f"{task.task_id}.json",
                status.model_dump(mode="json"),
            )
            if task_dir.parent.name != "completed":
                completed_dir = root / "completed" / task.task_id
                if completed_dir.exists():
                    _remove_tree(completed_dir)
                os.replace(task_dir, completed_dir)
            destination = root / "receipts" / "imported" / receipt_dir.name
            if destination.exists():
                _remove_tree(destination)
            os.replace(receipt_dir, destination)
            imported.append(receipt.task_id)
        except Exception as exc:
            rejected.append({"receipt": receipt_dir.name, "error": f"{type(exc).__name__}: {exc}"})
            destination = root / "receipts" / "rejected" / receipt_dir.name
            if destination.exists():
                _remove_tree(destination)
            os.replace(receipt_dir, destination)
            atomic_write_json(
                destination / "rejection.json",
                {"rejected_at": datetime.now(UTC).isoformat(), "error": rejected[-1]["error"]},
            )
    write_cloud_index(root, list(indexed.values()))
    return {"imported": imported, "rejected": rejected, "pack_count": len(indexed)}


def _validate_receipt_binding(
    receipt: ExportWorkerReceipt,
    task: ExportTask,
    manifest: ExportSnapshotManifest,
) -> None:
    checks = {
        "task_id": receipt.task_id == task.task_id,
        "snapshot_id": receipt.snapshot_id == task.snapshot_id == manifest.snapshot_id,
        "worker_commit": receipt.worker_commit == task.expected_worker_commit,
        "v5_sha": (
            receipt.selected_v5_bundle_sha256
            == task.selected_v5_bundle_sha256
            == manifest.selected_v5_bundle_sha256
        ),
        "acceptance_set": (
            receipt.acceptance_set_id == task.acceptance_set_id == manifest.acceptance_set_id
        ),
        "authoritative": receipt.authoritative_input_snapshot,
        "nas_validated": receipt.nas_artifact_validated,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("receipt binding mismatch: " + ",".join(failed))
    summary_bytes = len(
        json.dumps(
            {
                "manifest": receipt.manifest_summary,
                "quality": receipt.data_quality_summary,
                "questions": receipt.expert_questions,
                "validation": receipt.validation_summary,
                "worker": receipt.worker_report_summary,
            },
            ensure_ascii=True,
        ).encode()
    )
    if summary_bytes > 256 * 1024:
        raise ValueError("receipt summaries exceed 256 KiB")


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _release_snapshot_bytes(root: Path, manifest: ExportSnapshotManifest) -> None:
    snapshot_dir = root / "snapshots" / manifest.snapshot_id
    files_dir = snapshot_dir / "files"
    release_marker = snapshot_dir / "RELEASED.json"
    if not files_dir.exists() and release_marker.is_file():
        return
    if files_dir.exists():
        import shutil

        shutil.rmtree(files_dir)
    atomic_write_json(
        release_marker,
        {
            "snapshot_id": manifest.snapshot_id,
            "manifest_sha256": manifest.manifest_sha256,
            "released_at": datetime.now(UTC).isoformat(),
            "reason": "nas_receipt_verified_snapshot_bytes_no_longer_required",
        },
        mode=0o440,
    )
