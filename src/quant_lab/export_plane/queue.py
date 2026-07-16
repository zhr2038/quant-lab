from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from quant_lab import __version__
from quant_lab.export_plane.contracts import ExportTask, ExportTaskState, ExportTaskStatus
from quant_lab.export_plane.signatures import (
    canonical_json_bytes,
    load_signing_key,
    sha256_bytes,
    sign_payload,
)
from quant_lab.export_plane.snapshot import seal_export_snapshot
from quant_lab.export_plane.status import (
    atomic_write_json,
    ensure_queue_layout,
    find_task_directory,
)


def create_export_task(
    *,
    export_date: date,
    lake_root: str | Path,
    queue_root: str | Path,
    signing_key_path: str | Path,
    signature_key_id: str,
    export_mode: str = "authoritative",
    expected_worker_commit: str | None = None,
    acceptance_set_id: str | None = None,
) -> tuple[ExportTask, ExportTaskStatus, bool]:
    queue = ensure_queue_layout(queue_root)
    manifest, _ = seal_export_snapshot(
        export_date=export_date,
        lake_root=lake_root,
        queue_root=queue,
        signing_key_path=signing_key_path,
        signature_key_id=signature_key_id,
        acceptance_set_id=acceptance_set_id,
    )
    worker_commit = expected_worker_commit or manifest.quant_lab_commit
    idempotency_key = sha256_bytes(
        canonical_json_bytes(
            {
                "snapshot_id": manifest.snapshot_id,
                "manifest_sha256": manifest.manifest_sha256,
                "export_mode": export_mode,
                "worker_commit": worker_commit,
            }
        )
    )
    task_id = f"export-{export_date:%Y%m%d}-{idempotency_key[:20]}"
    existing_dir = find_task_directory(queue, task_id)
    if existing_dir is not None:
        task = ExportTask.model_validate_json((existing_dir / "task.json").read_text())
        status_path = queue / "status" / f"{task_id}.json"
        status = ExportTaskStatus.model_validate_json(status_path.read_text())
        return task, status, False

    if not (queue / "snapshots" / manifest.snapshot_id / "files").is_dir():
        manifest, _ = seal_export_snapshot(
            export_date=export_date,
            lake_root=lake_root,
            queue_root=queue,
            signing_key_path=signing_key_path,
            signature_key_id=signature_key_id,
            acceptance_set_id=acceptance_set_id,
            rehydrate_released=True,
        )

    now = datetime.now(UTC)
    provisional = ExportTask(
        task_id=task_id,
        snapshot_id=manifest.snapshot_id,
        export_date=export_date,
        export_mode=export_mode,
        quant_lab_commit=manifest.quant_lab_commit,
        quant_lab_version=__version__,
        expected_worker_commit=worker_commit,
        report_schema_version="quant_lab.expert_pack.v1",
        selected_v5_bundle_sha256=manifest.selected_v5_bundle_sha256,
        acceptance_set_id=manifest.acceptance_set_id,
        snapshot_manifest_sha256=manifest.manifest_sha256,
        requested_at=now,
        idempotency_key=idempotency_key,
        signature_key_id=signature_key_id,
        signature="A" * 88,
    )
    task = ExportTask.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "signature": sign_payload(provisional, load_signing_key(signing_key_path)),
        }
    )
    status = ExportTaskStatus(
        task_id=task_id,
        snapshot_id=manifest.snapshot_id,
        state=ExportTaskState.PENDING,
        requested_at=now,
        updated_at=now,
        max_attempts=task.max_attempts,
        current_stage="pending",
        input_bytes=manifest.total_input_bytes,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{task_id}.", dir=queue / "pending"))
    try:
        atomic_write_json(temporary / "task.json", task.model_dump(mode="json"))
        os.chmod(temporary, 0o2770)
        os.replace(temporary, queue / "pending" / task_id)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    atomic_write_json(queue / "status" / f"{task_id}.json", status.model_dump(mode="json"))
    return task, status, True
