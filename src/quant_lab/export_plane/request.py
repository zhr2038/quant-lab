from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from quant_lab.export_plane.contracts import ExportRequest
from quant_lab.export_plane.queue import create_export_task
from quant_lab.export_plane.status import atomic_write_json, ensure_queue_layout


def submit_export_request(
    *,
    queue_root: str | Path,
    export_date: date,
    export_mode: str,
    requested_by: str = "web",
) -> tuple[ExportRequest, bool]:
    root = ensure_queue_layout(queue_root)
    for state in ("pending", "processing"):
        for existing in sorted((root / "requests" / state).glob("*.json"), reverse=True):
            request = ExportRequest.model_validate_json(existing.read_text(encoding="utf-8"))
            if request.export_date == export_date and request.export_mode == export_mode:
                return request, False
    now = datetime.now(UTC)
    request_id = f"export-request-{export_date:%Y%m%d}-{export_mode}-{now:%H%M%S%f}"
    request = ExportRequest(
        request_id=request_id,
        export_date=export_date,
        export_mode=export_mode,
        requested_at=now,
        requested_by=requested_by,
    )
    path = root / "requests" / "pending" / f"{request_id}.json"
    atomic_write_json(path, request.model_dump(mode="json"))
    atomic_write_json(
        root / "requests" / "status" / path.name,
        {
            "request_id": request_id,
            "state": "pending",
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return request, True


def process_export_requests(
    *,
    queue_root: str | Path,
    lake_root: str | Path,
    signing_key_path: str | Path,
    signature_key_id: str,
    expected_worker_commit: str,
    max_requests: int = 5,
) -> dict[str, Any]:
    root = ensure_queue_layout(queue_root)
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    pending = root / "requests" / "pending"
    for request_path in sorted(pending.glob("*.json"))[: max(1, max_requests)]:
        processing = root / "requests" / "processing" / request_path.name
        try:
            os.replace(request_path, processing)
        except FileNotFoundError:
            continue
        try:
            request = ExportRequest.model_validate_json(processing.read_text(encoding="utf-8"))
            atomic_write_json(
                root / "requests" / "status" / request_path.name,
                {
                    "request_id": request.request_id,
                    "state": "snapshot_preparing",
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            task, status, created = create_export_task(
                export_date=request.export_date,
                lake_root=lake_root,
                queue_root=root,
                signing_key_path=signing_key_path,
                signature_key_id=signature_key_id,
                export_mode=request.export_mode,
                expected_worker_commit=expected_worker_commit,
            )
            result = {
                "request_id": request.request_id,
                "state": status.state.value,
                "task_id": task.task_id,
                "snapshot_id": task.snapshot_id,
                "task_created": created,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            atomic_write_json(root / "requests" / "status" / request_path.name, result)
            destination = root / "requests" / "completed" / request_path.name
            os.replace(processing, destination)
            completed.append(result)
        except Exception as exc:
            error = {
                "request": request_path.stem,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
            failed.append(error)
            destination = root / "requests" / "failed" / request_path.name
            if destination.exists():
                destination.unlink()
            os.replace(processing, destination)
            atomic_write_json(
                root / "requests" / "status" / request_path.name,
                {
                    "request_id": request_path.stem,
                    "state": "failed",
                    "last_error": error["error"],
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
    return {"completed": completed, "failed": failed}


def cancel_pending_request(queue_root: str | Path, request_id: str) -> bool:
    root = ensure_queue_layout(queue_root)
    source = root / "requests" / "pending" / f"{request_id}.json"
    if not source.is_file():
        return False
    destination = root / "requests" / "completed" / source.name
    if destination.exists():
        destination.unlink()
    os.replace(source, destination)
    atomic_write_json(
        root / "requests" / "status" / source.name,
        {
            "request_id": request_id,
            "state": "cancelled",
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return True
