from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from quant_lab.export_plane.contracts import (
    ExportSnapshotManifest,
    ExportTask,
    ExportTaskState,
    ExportTaskStatus,
)
from quant_lab.export_plane.snapshot_gc import gc_export_snapshot_payloads
from quant_lab.export_plane.status import atomic_write_json, ensure_queue_layout

COMMIT = "a" * 40
V5_SHA = "b" * 64


def test_gc_releases_old_terminal_snapshot_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    queue = ensure_queue_layout(tmp_path / "queue")
    now = datetime(2026, 7, 19, tzinfo=UTC)
    snapshot_id = "export-snapshot-old-failed"
    task = _write_task(queue, snapshot_id, "failed", now - timedelta(days=2))
    snapshot = _write_snapshot(queue / "snapshots" / snapshot_id, snapshot_id, b"payload")

    result = gc_export_snapshot_payloads(queue, dry_run=False, now=now)

    assert result.released_snapshot_count == 1
    assert result.released_invalid_snapshot_count == 0
    assert result.bytes_released == len(b"payload")
    assert not (snapshot / "files").exists()
    assert (snapshot / "manifest.json").is_file()
    assert (snapshot / "SEALED").is_file()
    assert (queue / "failed" / task.task_id / "task.json").is_file()
    assert (queue / "status" / f"{task.task_id}.json").is_file()
    marker = json.loads((snapshot / "RELEASED.json").read_text(encoding="utf-8"))
    assert marker["state"] == "released"
    assert marker["reason"] == "terminal_grace_elapsed"
    assert marker["manifest_preserved"] is True
    assert (queue / "audit" / "snapshot_gc.jsonl").is_file()


def test_gc_protects_active_snapshot_even_with_terminal_reference(tmp_path: Path) -> None:
    queue = ensure_queue_layout(tmp_path / "queue")
    now = datetime(2026, 7, 19, tzinfo=UTC)
    snapshot_id = "export-snapshot-active"
    _write_task(queue, snapshot_id, "failed", now - timedelta(days=2), suffix="old")
    _write_task(queue, snapshot_id, "pending", now - timedelta(minutes=5), suffix="retry")
    snapshot = _write_snapshot(queue / "snapshots" / snapshot_id, snapshot_id, b"active")

    result = gc_export_snapshot_payloads(
        queue,
        terminal_grace_hours=0,
        max_terminal_payload_bytes=0,
        dry_run=False,
        now=now,
    )

    assert result.active_snapshot_count == 1
    assert result.released_snapshot_count == 0
    assert (snapshot / "files" / "payload.bin").read_bytes() == b"active"


def test_gc_capacity_releases_oldest_recent_terminal_payload(tmp_path: Path) -> None:
    queue = ensure_queue_layout(tmp_path / "queue")
    now = datetime(2026, 7, 19, tzinfo=UTC)
    old_id = "export-snapshot-capacity-old"
    new_id = "export-snapshot-capacity-new"
    _write_task(queue, old_id, "failed", now - timedelta(hours=2), suffix="old")
    _write_task(queue, new_id, "failed", now - timedelta(hours=1), suffix="new")
    old_snapshot = _write_snapshot(queue / "snapshots" / old_id, old_id, b"1234")
    new_snapshot = _write_snapshot(queue / "snapshots" / new_id, new_id, b"5678")

    result = gc_export_snapshot_payloads(
        queue,
        terminal_grace_hours=24,
        max_terminal_payload_bytes=4,
        dry_run=False,
        now=now,
    )

    assert result.released_snapshot_count == 1
    assert not (old_snapshot / "files").exists()
    assert (new_snapshot / "files").is_dir()
    marker = json.loads((old_snapshot / "RELEASED.json").read_text(encoding="utf-8"))
    assert marker["reason"] == "terminal_capacity_limit"


def test_gc_releases_terminal_invalid_snapshot_payload_copy(tmp_path: Path) -> None:
    queue = ensure_queue_layout(tmp_path / "queue")
    now = datetime(2026, 7, 19, tzinfo=UTC)
    snapshot_id = "export-snapshot-invalid-copy"
    _write_task(queue, snapshot_id, "failed", now - timedelta(days=2))
    invalid = _write_snapshot(
        queue / "audit" / "invalid_snapshots" / f"{snapshot_id}.20260717T000000Z",
        snapshot_id,
        b"invalid-copy-payload",
    )

    result = gc_export_snapshot_payloads(queue, dry_run=False, now=now)

    assert result.released_snapshot_count == 0
    assert result.released_invalid_snapshot_count == 1
    assert not (invalid / "files").exists()
    assert (invalid / "manifest.json").is_file()
    assert (invalid / "RELEASED.json").is_file()


def test_gc_dry_run_reports_without_removing_payload(tmp_path: Path) -> None:
    queue = ensure_queue_layout(tmp_path / "queue")
    now = datetime(2026, 7, 19, tzinfo=UTC)
    snapshot_id = "export-snapshot-dry-run"
    _write_task(queue, snapshot_id, "cancelled", now - timedelta(days=2))
    snapshot = _write_snapshot(queue / "snapshots" / snapshot_id, snapshot_id, b"keep")

    result = gc_export_snapshot_payloads(queue, dry_run=True, now=now)

    assert result.dry_run is True
    assert result.released_snapshot_count == 1
    assert (snapshot / "files" / "payload.bin").is_file()
    assert not (snapshot / "RELEASED.json").exists()


def _write_snapshot(root: Path, snapshot_id: str, payload: bytes) -> Path:
    files = root / "files"
    files.mkdir(parents=True)
    (files / "payload.bin").write_bytes(payload)
    manifest = ExportSnapshotManifest(
        snapshot_id=snapshot_id,
        export_date=date(2026, 7, 17),
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        quant_lab_commit=COMMIT,
        quant_lab_version="0.1.0",
        v5_commit="c" * 40,
        selected_v5_bundle_name="v5.tar.gz",
        selected_v5_bundle_sha256=V5_SHA,
        acceptance_set_id="acceptance-test",
        risk_permission_identity="risk:test",
        paper_lifecycle_identity="paper:test",
        environment_fingerprint="d" * 64,
        schema_fingerprint="e" * 64,
        files=[
            {
                "relative_path": "payload.bin",
                "sha256": "f" * 64,
                "size_bytes": len(payload),
                "mtime_ns": 1,
                "dataset": "test",
                "media_type": "other",
            }
        ],
        total_input_bytes=len(payload),
        manifest_sha256="1" * 64,
        signature_key_id="cloud-export-v1",
        signature="A" * 88,
    )
    atomic_write_json(root / "manifest.json", manifest.model_dump(mode="json"))
    (root / "SEALED").write_text(manifest.manifest_sha256 + "\n", encoding="utf-8")
    return root


def _write_task(
    queue: Path,
    snapshot_id: str,
    state: str,
    updated_at: datetime,
    *,
    suffix: str = "task",
) -> ExportTask:
    task_id = f"export-20260717-{suffix}-{snapshot_id.removeprefix('export-snapshot-')}"
    task = ExportTask(
        task_id=task_id,
        snapshot_id=snapshot_id,
        export_date=date(2026, 7, 17),
        export_mode="authoritative",
        quant_lab_commit=COMMIT,
        quant_lab_version="0.1.0",
        expected_worker_commit=COMMIT,
        report_schema_version="quant_lab.expert_pack.v1",
        selected_v5_bundle_sha256=V5_SHA,
        acceptance_set_id="acceptance-test",
        snapshot_manifest_sha256="1" * 64,
        requested_at=updated_at,
        idempotency_key="2" * 64,
        signature_key_id="cloud-export-v1",
        signature="A" * 88,
    )
    task_root = queue / state / task.task_id
    task_root.mkdir(parents=True)
    atomic_write_json(task_root / "task.json", task.model_dump(mode="json"))
    status = ExportTaskStatus(
        task_id=task.task_id,
        snapshot_id=snapshot_id,
        state=ExportTaskState(state),
        requested_at=updated_at,
        updated_at=updated_at,
        current_stage=state,
    )
    atomic_write_json(
        queue / "status" / f"{task.task_id}.json",
        status.model_dump(mode="json"),
    )
    return task
