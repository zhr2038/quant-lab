from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.export_plane.contracts import (
    ExportDatasetReference,
    ExportPackIndexEntry,
    ExportSnapshotManifest,
    ExportTask,
    ExportWorkerReceipt,
)
from quant_lab.export_plane.receipt import import_export_receipts
from quant_lab.export_plane.request import submit_export_request
from quant_lab.export_plane.signatures import canonical_json_bytes, sha256_bytes, sign_payload
from quant_lab.export_plane.status import atomic_write_json, ensure_queue_layout
from quant_lab.export_worker.retention import enforce_accepted_retention

COMMIT = "c" * 40
V5_SHA = "d" * 64
FILE_SHA = "e" * 64
NOW = datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC)


def _write_key_pair(root: Path) -> tuple[Path, Path, Ed25519PrivateKey]:
    key = Ed25519PrivateKey.generate()
    private_path = root / "worker-private.pem"
    public_path = root / "worker-public.pem"
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path, key


def _snapshot() -> ExportSnapshotManifest:
    reference = ExportDatasetReference(
        relative_path="lake/gold/example.parquet",
        sha256=FILE_SHA,
        size_bytes=7,
        mtime_ns=1,
        dataset="example",
        media_type="parquet",
    )
    values = {
        "snapshot_id": "export-snapshot-test",
        "export_date": date(2026, 7, 16),
        "created_at": NOW,
        "quant_lab_commit": COMMIT,
        "quant_lab_version": "0.1.0",
        "v5_commit": "f" * 40,
        "selected_v5_bundle_name": "v5-test.tar.gz",
        "selected_v5_bundle_sha256": V5_SHA,
        "acceptance_set_id": "acceptance-test",
        "risk_permission_identity": "risk-test",
        "paper_lifecycle_identity": "paper-test",
        "environment_fingerprint": "1" * 64,
        "schema_fingerprint": "2" * 64,
        "files": [reference],
        "total_input_bytes": 7,
        "authoritative_input_snapshot": True,
        "manifest_sha256": "0" * 64,
        "signature_key_id": "cloud-v1",
        "signature_algorithm": "ed25519",
        "signature": "A" * 88,
    }
    provisional = ExportSnapshotManifest.model_validate(values)
    digest_values = provisional.model_dump(mode="json")
    digest_values.pop("signature")
    digest_values.pop("manifest_sha256")
    return provisional.model_copy(
        update={"manifest_sha256": sha256_bytes(canonical_json_bytes(digest_values))}
    )


def _task(snapshot: ExportSnapshotManifest) -> ExportTask:
    return ExportTask(
        task_id="export-20260716-test",
        snapshot_id=snapshot.snapshot_id,
        export_date=snapshot.export_date,
        export_mode="authoritative",
        quant_lab_commit=COMMIT,
        quant_lab_version="0.1.0",
        expected_worker_commit=COMMIT,
        report_schema_version="quant_lab.expert_pack.v1",
        selected_v5_bundle_sha256=V5_SHA,
        acceptance_set_id="acceptance-test",
        snapshot_manifest_sha256=snapshot.manifest_sha256,
        requested_at=NOW,
        idempotency_key="3" * 64,
        signature_key_id="cloud-v1",
        signature="A" * 88,
    )


def _receipt(task: ExportTask, key: Ed25519PrivateKey) -> ExportWorkerReceipt:
    provisional = ExportWorkerReceipt(
        task_id=task.task_id,
        snapshot_id=task.snapshot_id,
        worker_id="nas-worker-test",
        worker_commit=COMMIT,
        pack_id="expert-pack-test",
        pack_name="expert-test.zip",
        pack_sha256="4" * 64,
        pack_size_bytes=123,
        pack_manifest_sha256="5" * 64,
        validation_report_sha256="6" * 64,
        selected_v5_bundle_sha256=V5_SHA,
        acceptance_set_id="acceptance-test",
        download_relative_path="2026/07/16/expert-pack-test/expert-test.zip",
        generated_at=NOW,
        accepted_at=NOW,
        signature_key_id="nas-worker-v1",
        signature="A" * 88,
    )
    return provisional.model_copy(update={"signature": sign_payload(provisional, key)})


def _install_receipt(root: Path, receipt: ExportWorkerReceipt, name: str = "receipt-1") -> None:
    target = root / "receipts" / "inbox" / name
    target.mkdir(parents=True)
    atomic_write_json(target / "receipt.json", receipt.model_dump(mode="json"))


def _install_control_plane(root: Path) -> tuple[ExportTask, ExportSnapshotManifest]:
    queue = ensure_queue_layout(root)
    snapshot = _snapshot()
    task = _task(snapshot)
    snapshot_dir = queue / "snapshots" / snapshot.snapshot_id
    snapshot_dir.mkdir(parents=True)
    atomic_write_json(snapshot_dir / "manifest.json", snapshot.model_dump(mode="json"))
    snapshot_file = snapshot_dir / "files" / snapshot.files[0].relative_path
    snapshot_file.parent.mkdir(parents=True)
    snapshot_file.write_bytes(b"example")
    task_dir = queue / "running" / task.task_id
    task_dir.mkdir(parents=True)
    atomic_write_json(task_dir / "task.json", task.model_dump(mode="json"))
    return task, snapshot


def test_signed_receipt_import_is_idempotent(tmp_path: Path) -> None:
    _, public_path, key = _write_key_pair(tmp_path)
    task, _ = _install_control_plane(tmp_path / "queue")
    receipt = _receipt(task, key)
    _install_receipt(tmp_path / "queue", receipt)

    first = import_export_receipts(
        queue_root=tmp_path / "queue",
        worker_public_key_path=public_path,
        worker_key_id="nas-worker-v1",
    )
    _install_receipt(tmp_path / "queue", receipt, "receipt-replay")
    second = import_export_receipts(
        queue_root=tmp_path / "queue",
        worker_public_key_path=public_path,
        worker_key_id="nas-worker-v1",
    )

    assert first == {"imported": [task.task_id], "rejected": [], "pack_count": 1}
    assert second == {"imported": [task.task_id], "rejected": [], "pack_count": 1}
    index = json.loads((tmp_path / "queue" / "cloud_index.json").read_text())
    assert len(index["packs"]) == 1
    assert index["packs"][0]["control_plane_receipt_verified"] is True
    snapshot_dir = tmp_path / "queue" / "snapshots" / receipt.snapshot_id
    assert not (snapshot_dir / "files").exists()
    assert (snapshot_dir / "RELEASED.json").is_file()


def test_tampered_receipt_is_rejected(tmp_path: Path) -> None:
    _, public_path, key = _write_key_pair(tmp_path)
    task, _ = _install_control_plane(tmp_path / "queue")
    receipt = _receipt(task, key).model_copy(update={"pack_size_bytes": 124})
    _install_receipt(tmp_path / "queue", receipt)

    result = import_export_receipts(
        queue_root=tmp_path / "queue",
        worker_public_key_path=public_path,
        worker_key_id="nas-worker-v1",
    )

    assert result["imported"] == []
    assert result["pack_count"] == 0
    assert "signature verification failed" in result["rejected"][0]["error"]


def _index_row(
    *,
    pack_id: str,
    accepted_at: datetime,
    ai_consumed: bool,
) -> ExportPackIndexEntry:
    return ExportPackIndexEntry(
        pack_id=pack_id,
        pack_name=f"{pack_id}.zip",
        export_date=accepted_at.date(),
        generated_at=accepted_at,
        accepted_at=accepted_at,
        pack_sha256="7" * 64,
        pack_size_bytes=100,
        snapshot_id=f"snapshot-{pack_id}",
        authoritative_input_snapshot=True,
        nas_artifact_validated=True,
        control_plane_receipt_verified=True,
        download_ready=True,
        download_relative_path=f"2026/07/16/{pack_id}/{pack_id}.zip",
        selected_v5_bundle_sha256=V5_SHA,
        acceptance_set_id=f"acceptance-{pack_id}",
        worker_id="nas-worker-test",
        worker_commit=COMMIT,
        ai_consumed=ai_consumed,
    )


def test_retention_removes_consumed_pack_but_pins_unconsumed_pack(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    old = datetime.now(UTC) - timedelta(days=120)
    consumed = _index_row(pack_id="consumed", accepted_at=old, ai_consumed=True)
    pending_ai = _index_row(pack_id="pending-ai", accepted_at=old, ai_consumed=False)
    for row in (consumed, pending_ai):
        pack = accepted / row.download_relative_path
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * row.pack_size_bytes)
    index = tmp_path / "accepted_index.json"
    atomic_write_json(
        index,
        {"packs": [consumed.model_dump(mode="json"), pending_ai.model_dump(mode="json")]},
    )

    result = enforce_accepted_retention(
        accepted_root=accepted,
        index_path=index,
        retention_days=90,
        max_total_bytes=100,
        min_keep_packs=0,
        audit_log_path=tmp_path / "retention.jsonl",
    )

    assert [row["pack_id"] for row in result["removed"]] == ["consumed"]
    assert not (accepted / consumed.download_relative_path).exists()
    assert (accepted / pending_ai.download_relative_path).is_file()
    assert "pending-ai" in index.read_text(encoding="utf-8")


def test_duplicate_active_export_request_is_idempotent(tmp_path: Path) -> None:
    first, first_created = submit_export_request(
        queue_root=tmp_path,
        export_date=date(2026, 7, 16),
        export_mode="authoritative",
    )
    second, second_created = submit_export_request(
        queue_root=tmp_path,
        export_date=date(2026, 7, 16),
        export_mode="authoritative",
    )

    assert first_created is True
    assert second_created is False
    assert second.request_id == first.request_id
    assert len(list((tmp_path / "requests" / "pending").glob("*.json"))) == 1
