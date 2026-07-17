from __future__ import annotations

import io
import json
import os
import tarfile
import threading
import time
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from quant_lab.export import daily as daily_export
from quant_lab.export_materializer import writer as materializer_writer
from quant_lab.export_materializer.validator import validate_export_pack_locally
from quant_lab.export_plane import snapshot as snapshot_module
from quant_lab.export_plane.cloud_index import export_plane_status, write_cloud_index
from quant_lab.export_plane.contracts import (
    ExportDatasetReference,
    ExportPackIndexEntry,
    ExportSnapshotManifest,
    ExportTask,
    ExportTaskState,
    ExportTaskStatus,
)
from quant_lab.export_plane.signatures import (
    load_public_key,
    load_signing_key,
    sha256_bytes,
    sign_payload,
    verify_payload,
)
from quant_lab.export_worker import accepted as accepted_module
from quant_lab.export_worker import runner as export_runner
from quant_lab.export_worker.runner import _ssh
from quant_lab.export_worker.sync import sync_snapshot_blobs

COMMIT = "a" * 40
V5_SHA = "b" * 64


def _keys(tmp_path: Path) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def _snapshot(reference: ExportDatasetReference | None = None) -> ExportSnapshotManifest:
    files = [reference] if reference else [
        ExportDatasetReference(
            relative_path="lake/gold/example/part.parquet",
            sha256="c" * 64,
            size_bytes=3,
            mtime_ns=1,
            row_count=1,
            dataset="example",
            media_type="parquet",
        )
    ]
    return ExportSnapshotManifest(
        snapshot_id="export-snapshot-test",
        export_date=date(2026, 7, 16),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
        quant_lab_commit=COMMIT,
        quant_lab_current_main_commit=COMMIT,
        current_main_production_relationship="MATCH",
        quant_lab_version="0.1.0",
        v5_commit="d" * 40,
        selected_v5_bundle_name="v5.tar.gz",
        selected_v5_bundle_sha256=V5_SHA,
        acceptance_set_id="acceptance-test",
        risk_permission_identity="risk:test",
        paper_lifecycle_identity="paper:test",
        proposal_snapshot_id="proposal-snapshot:test",
        proposal_snapshot_sha256="2" * 64,
        proposal_content_snapshot_id="proposal-content-snapshot:test",
        proposal_content_snapshot_sha256="3" * 64,
        snapshot_generated_at=datetime(2026, 7, 15, 23, 55, tzinfo=UTC),
        v5_observed_proposal_snapshot_id="proposal-snapshot:test",
        v5_observed_proposal_snapshot_sha256="2" * 64,
        v5_observed_proposal_content_snapshot_id="proposal-content-snapshot:test",
        v5_observed_proposal_content_snapshot_sha256="3" * 64,
        selected_v5_bundle_built_at=datetime(2026, 7, 16, tzinfo=UTC),
        environment_fingerprint="e" * 64,
        schema_fingerprint="f" * 64,
        files=files,
        total_input_bytes=sum(item.size_bytes for item in files),
        manifest_sha256="1" * 64,
        signature_key_id="cloud-export-v1",
        signature="A" * 88,
    )


def _task() -> ExportTask:
    return ExportTask(
        task_id="export-20260716-test",
        snapshot_id="export-snapshot-test",
        export_date=date(2026, 7, 16),
        export_mode="authoritative",
        quant_lab_commit=COMMIT,
        quant_lab_version="0.1.0",
        expected_worker_commit=COMMIT,
        report_schema_version="quant_lab.expert_pack.v1",
        selected_v5_bundle_sha256=V5_SHA,
        acceptance_set_id="acceptance-test",
        snapshot_manifest_sha256="1" * 64,
        requested_at=datetime(2026, 7, 16, tzinfo=UTC),
        idempotency_key="2" * 64,
        signature_key_id="cloud-export-v1",
        signature="A" * 88,
    )


def test_contract_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExportTask.model_validate({**_task().model_dump(mode="json"), "secret": "no"})


def test_ed25519_signature_detects_tampering(tmp_path: Path) -> None:
    private_path, public_path = _keys(tmp_path)
    task = _task()
    signature = sign_payload(task, load_signing_key(private_path))
    verify_payload(task, signature, load_public_key(public_path))
    changed = task.model_copy(update={"acceptance_set_id": "acceptance:changed"})
    with pytest.raises(ValueError, match="signature verification failed"):
        verify_payload(changed, signature, load_public_key(public_path))


def test_snapshot_manifest_digest_uses_normalized_nested_models(tmp_path: Path) -> None:
    private_path, public_path = _keys(tmp_path)
    unsigned = _snapshot().model_dump(mode="json")
    unsigned["manifest_sha256"] = "0" * 64
    unsigned["signature"] = "A" * 88

    manifest = snapshot_module._finalize_snapshot_manifest(  # noqa: SLF001
        unsigned,
        private_path,
    )

    snapshot_module.verify_snapshot_manifest_digest(manifest)
    verify_payload(manifest, manifest.signature, load_public_key(public_path))


def test_snapshot_acceptance_context_binds_current_proposal_and_v5(
    monkeypatch,
) -> None:
    content_id = "proposal-content-snapshot:stable"
    content_sha = "c" * 64
    proposal = pl.DataFrame(
        {
            "proposal_snapshot_id": ["proposal-snapshot:current"],
            "proposal_snapshot_sha256": ["d" * 64],
            "proposal_content_snapshot_id": [content_id],
            "proposal_content_snapshot_sha256": [content_sha],
            "snapshot_generated_at": ["2026-07-16T01:00:00+00:00"],
        }
    )
    contract = pl.DataFrame(
        {
            "bundle_sha256": [V5_SHA],
            "proposal_snapshot_id": ["proposal-snapshot:observed"],
            "proposal_snapshot_sha256": ["e" * 64],
            "proposal_content_snapshot_id": [content_id],
            "proposal_content_snapshot_sha256": [content_sha],
        }
    )
    monkeypatch.setattr(
        snapshot_module.readers,
        "read_dataset",
        lambda _root, name: proposal
        if name == "paper_strategy_proposal_snapshot"
        else contract,
    )
    monkeypatch.setattr(
        snapshot_module.daily_export,
        "_latest_v5_contract_status_row",
        lambda frame: frame.tail(1).to_dicts()[0],
    )
    monkeypatch.setattr(
        snapshot_module.daily_export,
        "_current_main_commit_full",
        lambda: COMMIT,
    )

    context = snapshot_module._snapshot_acceptance_context(  # noqa: SLF001
        Path("/lake"),
        v5_context={"selected_v5_bundle_built_at": "2026-07-16T02:00:00+00:00"},
        quant_lab_commit=COMMIT,
        selected_v5_bundle_sha256=V5_SHA,
    )

    assert context["current_main_production_relationship"] == "MATCH"
    assert context["proposal_content_snapshot_sha256"] == content_sha
    assert context["v5_observed_proposal_content_snapshot_sha256"] == content_sha


def test_snapshot_acceptance_context_is_read_from_copied_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    private_path, _public_path = _keys(tmp_path)
    lake_root = tmp_path / "lake"
    source = lake_root / "gold" / "example" / "data.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"value":1}\n', encoding="utf-8")
    bundle = tmp_path / "inbox" / "v5" / "bundle.tar.gz"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"v5-bundle")
    bundle_sha = sha256_bytes(bundle.read_bytes())
    captured: dict[str, Path] = {}
    content_sha = "c" * 64

    monkeypatch.setattr(snapshot_module, "_git_commit", lambda: COMMIT)
    monkeypatch.setattr(
        snapshot_module.daily_export,
        "_observe_v5_before_export",
        lambda _root: {
            "selected_v5_bundle_path": str(bundle),
            "selected_v5_bundle_sha256": bundle_sha,
            "selected_v5_bundle_built_at": "2026-07-16T02:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        snapshot_module.daily_export,
        "_selected_v5_bundle_git_commit",
        lambda _context: "b" * 40,
    )
    monkeypatch.setattr(
        snapshot_module,
        "_export_source_files",
        lambda _root: [("example", source, "lake/gold/example/data.json")],
    )
    monkeypatch.setattr(snapshot_module, "_dataset_identity", lambda *_args: "identity")
    monkeypatch.setattr(snapshot_module, "_environment_fingerprint", lambda: "d" * 64)
    monkeypatch.setattr(snapshot_module, "_schema_fingerprint", lambda: "e" * 64)

    def acceptance_context(copied_lake: Path, **_kwargs) -> dict[str, object]:
        captured["root"] = copied_lake
        assert (copied_lake / "gold" / "example" / "data.json").read_text(
            encoding="utf-8"
        ) == '{"value":1}\n'
        return {
            "quant_lab_current_main_commit": COMMIT,
            "current_main_production_relationship": "MATCH",
            "proposal_snapshot_id": "proposal-snapshot:current",
            "proposal_snapshot_sha256": "f" * 64,
            "proposal_content_snapshot_id": "proposal-content-snapshot:stable",
            "proposal_content_snapshot_sha256": content_sha,
            "snapshot_generated_at": datetime(2026, 7, 16, 1, tzinfo=UTC),
            "v5_observed_proposal_snapshot_id": "proposal-snapshot:observed",
            "v5_observed_proposal_snapshot_sha256": "1" * 64,
            "v5_observed_proposal_content_snapshot_id": (
                "proposal-content-snapshot:stable"
            ),
            "v5_observed_proposal_content_snapshot_sha256": content_sha,
            "selected_v5_bundle_built_at": datetime(2026, 7, 16, 2, tzinfo=UTC),
        }

    monkeypatch.setattr(snapshot_module, "_snapshot_acceptance_context", acceptance_context)

    _manifest, snapshot_dir = snapshot_module.seal_export_snapshot(
        export_date=date(2026, 7, 16),
        lake_root=lake_root,
        queue_root=tmp_path / "queue",
        signing_key_path=private_path,
        signature_key_id="cloud-export-v1",
    )

    assert captured["root"] != lake_root
    assert captured["root"].parent.name == "files"
    assert captured["root"].parent.parent.name.startswith(".sealing.")
    assert (snapshot_dir / "files" / "lake" / "gold" / "example" / "data.json").exists()


def test_snapshot_sealing_rejects_missing_v5_commit_with_explicit_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lake_root = tmp_path / "lake"
    lake_root.mkdir()
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"v5-bundle")
    bundle_sha = sha256_bytes(bundle.read_bytes())

    monkeypatch.setattr(snapshot_module, "_git_commit", lambda: COMMIT)
    monkeypatch.setattr(
        snapshot_module.daily_export,
        "_observe_v5_before_export",
        lambda _root: {
            "selected_v5_bundle_path": str(bundle),
            "selected_v5_bundle_sha256": bundle_sha,
        },
    )
    monkeypatch.setattr(
        snapshot_module.daily_export,
        "_selected_v5_bundle_git_commit",
        lambda _context: None,
    )

    with pytest.raises(
        RuntimeError,
        match="selected V5 bundle does not expose a full git commit",
    ):
        snapshot_module.seal_export_snapshot(
            export_date=date(2026, 7, 17),
            lake_root=lake_root,
            queue_root=tmp_path / "queue",
            signing_key_path=tmp_path / "unused.key",
            signature_key_id="cloud-export-v1",
        )


def test_materializer_requires_signed_acceptance_context() -> None:
    snapshot = _snapshot().model_copy(
        update={
            "quant_lab_current_main_commit": None,
            "current_main_production_relationship": "UNOBSERVABLE",
            "proposal_content_snapshot_id": None,
            "proposal_content_snapshot_sha256": None,
            "snapshot_generated_at": None,
            "v5_observed_proposal_content_snapshot_id": None,
            "v5_observed_proposal_content_snapshot_sha256": None,
            "selected_v5_bundle_built_at": None,
        }
    )
    with pytest.raises(RuntimeError, match="sealed_snapshot_acceptance_context_missing"):
        materializer_writer._sealed_acceptance_context(snapshot)  # noqa: SLF001


def test_materializer_uses_signed_acceptance_context() -> None:
    snapshot = _snapshot().model_copy(
        update={
            "quant_lab_current_main_commit": COMMIT,
            "current_main_production_relationship": "MATCH",
            "proposal_snapshot_id": "proposal-snapshot:current",
            "proposal_snapshot_sha256": "c" * 64,
            "proposal_content_snapshot_id": "proposal-content-snapshot:stable",
            "proposal_content_snapshot_sha256": "d" * 64,
            "snapshot_generated_at": datetime(2026, 7, 16, 1, tzinfo=UTC),
            "v5_observed_proposal_snapshot_id": "proposal-snapshot:observed",
            "v5_observed_proposal_snapshot_sha256": "e" * 64,
            "v5_observed_proposal_content_snapshot_id": (
                "proposal-content-snapshot:stable"
            ),
            "v5_observed_proposal_content_snapshot_sha256": "d" * 64,
            "selected_v5_bundle_built_at": datetime(2026, 7, 16, 2, tzinfo=UTC),
        }
    )

    context = materializer_writer._sealed_acceptance_context(snapshot)  # noqa: SLF001

    assert context["quant_lab_production_commit"] == COMMIT
    assert context["quant_lab_current_main_commit"] == COMMIT
    assert context["proposal_content_snapshot_sha256"] == "d" * 64
    assert context["snapshot_generated_at"] == "2026-07-16T01:00:00Z"
    assert context["selected_v5_bundle_built_at"] == "2026-07-16T02:00:00Z"
    json.dumps(context)


def test_snapshot_contract_rejects_inconsistent_content_identity() -> None:
    payload = _snapshot().model_dump(mode="json")
    payload.update(
        {
            "quant_lab_current_main_commit": COMMIT,
            "current_main_production_relationship": "MATCH",
            "proposal_content_snapshot_id": "proposal-content-snapshot:current",
            "proposal_content_snapshot_sha256": "c" * 64,
            "snapshot_generated_at": "2026-07-16T01:00:00+00:00",
            "v5_observed_proposal_content_snapshot_id": (
                "proposal-content-snapshot:other"
            ),
            "v5_observed_proposal_content_snapshot_sha256": "d" * 64,
            "selected_v5_bundle_built_at": "2026-07-16T02:00:00+00:00",
        }
    )

    with pytest.raises(ValidationError, match="proposal content snapshot identity"):
        ExportSnapshotManifest.model_validate(payload)


def test_snapshot_blob_sync_downloads_once_then_hits_cache(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"snapshot-data")
    reference = ExportDatasetReference(
        relative_path="lake/gold/example/data.bin",
        sha256=sha256_bytes(source.read_bytes()),
        size_bytes=source.stat().st_size,
        mtime_ns=source.stat().st_mtime_ns,
        dataset="example",
        media_type="other",
    )
    snapshot = _snapshot(reference)
    calls: list[str] = []

    def fetch(relative: str, target: Path) -> None:
        calls.append(relative)
        target.write_bytes(source.read_bytes())

    first = sync_snapshot_blobs(
        snapshot,
        data_root=tmp_path / "data",
        fetch_blob=fetch,
        min_free_disk_bytes=0,
        max_snapshot_bytes=1024,
    )
    second = sync_snapshot_blobs(
        snapshot,
        data_root=tmp_path / "data",
        fetch_blob=fetch,
        min_free_disk_bytes=0,
        max_snapshot_bytes=1024,
    )
    assert first.downloaded_files == 1
    assert second.downloaded_files == 0
    assert second.cache_hits == 1
    assert calls == [reference.relative_path]


def test_snapshot_blob_sync_batches_all_missing_files(tmp_path: Path) -> None:
    payloads = {
        "lake/gold/one/data.bin": b"one",
        "lake/gold/two/data.bin": b"two",
    }
    references = [
        ExportDatasetReference(
            relative_path=relative_path,
            sha256=sha256_bytes(payload),
            size_bytes=len(payload),
            mtime_ns=1,
            dataset=relative_path.split("/")[2],
            media_type="other",
        )
        for relative_path, payload in payloads.items()
    ]
    base = _snapshot()
    snapshot = ExportSnapshotManifest.model_validate(
        {
            **base.model_dump(mode="json"),
            "files": [item.model_dump(mode="json") for item in references],
            "total_input_bytes": sum(len(value) for value in payloads.values()),
        }
    )
    batches: list[list[str]] = []

    def fetch_batch(items: list[ExportDatasetReference], target: Path) -> None:
        batches.append([item.relative_path for item in items])
        for item in items:
            destination = target / item.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payloads[item.relative_path])

    result = sync_snapshot_blobs(
        snapshot,
        data_root=tmp_path / "data",
        fetch_blob=lambda *_args: pytest.fail("single-file fetch should not run"),
        fetch_blobs=fetch_batch,
        batch_fetch_workers=2,
        min_free_disk_bytes=0,
        max_snapshot_bytes=1024,
    )

    assert sorted(batches) == [[item] for item in sorted(payloads)]
    assert result.cache_hits == 0
    assert result.downloaded_files == 2
    assert result.downloaded_bytes == 6
    for relative_path, payload in payloads.items():
        assert (result.snapshot_root / "files" / relative_path).read_bytes() == payload


def test_snapshot_blob_sync_preserves_completed_batches_after_peer_failure(
    tmp_path: Path,
) -> None:
    payloads = {
        "lake/gold/one/data.bin": b"one",
        "lake/gold/two/data.bin": b"two",
    }
    references = [
        ExportDatasetReference(
            relative_path=relative_path,
            sha256=sha256_bytes(payload),
            size_bytes=len(payload),
            mtime_ns=1,
            dataset=relative_path.split("/")[2],
            media_type="other",
        )
        for relative_path, payload in payloads.items()
    ]
    base = _snapshot()
    snapshot = ExportSnapshotManifest.model_validate(
        {
            **base.model_dump(mode="json"),
            "files": [item.model_dump(mode="json") for item in references],
            "total_input_bytes": sum(len(value) for value in payloads.values()),
        }
    )
    failed_once = {"value": False}

    def fetch_batch(items: list[ExportDatasetReference], target: Path) -> None:
        item = items[0]
        if item.relative_path.endswith("two/data.bin") and not failed_once["value"]:
            failed_once["value"] = True
            raise EOFError("simulated network interruption")
        destination = target / item.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[item.relative_path])

    with pytest.raises(EOFError, match="simulated network interruption"):
        sync_snapshot_blobs(
            snapshot,
            data_root=tmp_path / "data",
            fetch_blob=lambda *_args: pytest.fail("single-file fetch should not run"),
            fetch_blobs=fetch_batch,
            batch_fetch_workers=2,
            min_free_disk_bytes=0,
            max_snapshot_bytes=1024,
        )

    result = sync_snapshot_blobs(
        snapshot,
        data_root=tmp_path / "data",
        fetch_blob=lambda *_args: pytest.fail("single-file fetch should not run"),
        fetch_blobs=fetch_batch,
        batch_fetch_workers=2,
        min_free_disk_bytes=0,
        max_snapshot_bytes=1024,
    )

    assert result.cache_hits == 1
    assert result.downloaded_files == 1
    for relative_path, payload in payloads.items():
        assert (result.snapshot_root / "files" / relative_path).read_bytes() == payload


def test_snapshot_blob_sync_rejects_bad_sha(tmp_path: Path) -> None:
    reference = ExportDatasetReference(
        relative_path="lake/data.bin",
        sha256=sha256_bytes(b"correct"),
        size_bytes=7,
        mtime_ns=1,
        dataset="example",
        media_type="other",
    )

    def fetch(_relative: str, target: Path) -> None:
        target.write_bytes(b"wrong!!")

    with pytest.raises(RuntimeError, match="blob_sha256_mismatch"):
        sync_snapshot_blobs(
            _snapshot(reference),
            data_root=tmp_path / "data",
            fetch_blob=fetch,
            min_free_disk_bytes=0,
            max_snapshot_bytes=1024,
        )
    assert not list((tmp_path / "data").rglob("*.partial"))


@pytest.mark.skipif(os.name == "nt", reason="Windows forbids replacing an open file")
def test_snapshot_copy_keeps_open_inode_when_source_is_atomically_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "snapshot" / "source.json"
    destination.parent.mkdir()
    source.write_bytes(b'{"version":1}\n')
    original_copy = snapshot_module.shutil.copyfileobj

    def copy_then_replace(source_handle, target_handle, *, length: int) -> None:
        original_copy(source_handle, target_handle, length=length)
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(b'{"version":2}\n')
        replacement.replace(source)

    monkeypatch.setattr(snapshot_module.shutil, "copyfileobj", copy_then_replace)
    reference = snapshot_module._copy_stable_reference(  # noqa: SLF001
        "example",
        source,
        destination,
        "lake/example/source.json",
    )

    assert destination.read_bytes() == b'{"version":1}\n'
    assert source.read_bytes() == b'{"version":2}\n'
    assert reference.sha256 == sha256_bytes(b'{"version":1}\n')


def test_snapshot_copy_rejects_in_place_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "snapshot.json"
    source.write_bytes(b'{"version":1}\n')
    original_copy = snapshot_module.shutil.copyfileobj

    def copy_then_mutate(source_handle, target_handle, *, length: int) -> None:
        original_copy(source_handle, target_handle, length=length)
        with source.open("ab") as mutable:
            mutable.write(b"changed\n")
            mutable.flush()

    monkeypatch.setattr(snapshot_module.shutil, "copyfileobj", copy_then_mutate)
    with pytest.raises(RuntimeError, match="changed while copying"):
        snapshot_module._copy_snapshot_input(source, destination)  # noqa: SLF001


def test_worker_failure_publishes_terminal_cloud_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    task = _task()
    (work / "task.json").write_text(task.model_dump_json(), encoding="utf-8")
    published = []
    monkeypatch.setattr(
        export_runner,
        "_upload_status",
        lambda _config, status, _work: published.append(status),
    )
    monkeypatch.setattr(export_runner, "_scp_to", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        export_runner,
        "_ssh",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    export_runner._mark_failed(  # noqa: SLF001
        SimpleNamespace(
            worker_id="nas-export-worker-01",
            worker_commit=COMMIT,
            remote_queue_root="/queue",
            worker_status_path=tmp_path / "worker-status.json",
        ),
        task.task_id,
        work,
        ValueError("snapshot manifest SHA256 mismatch"),
    )

    assert len(published) == 1
    assert published[0].state.value == "failed"
    assert published[0].current_stage == "failed"
    assert published[0].last_error == "ValueError: snapshot manifest SHA256 mismatch"


def test_export_plane_status_ignores_stale_cancelled_task_and_completed_request(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 16, 14, 46, tzinfo=UTC)
    pack = ExportPackIndexEntry(
        pack_id="expert-pack-current",
        task_id="export-current",
        pack_name="current.zip",
        export_date=now.date(),
        generated_at=now,
        accepted_at=now,
        pack_sha256="1" * 64,
        pack_size_bytes=10,
        snapshot_id="export-snapshot-current",
        authoritative_input_snapshot=True,
        nas_artifact_validated=True,
        control_plane_receipt_verified=True,
        download_ready=True,
        download_relative_path="2026/07/16/expert-pack-current/current.zip",
        selected_v5_bundle_sha256=V5_SHA,
        acceptance_set_id="acceptance-current",
        worker_id="worker",
        worker_commit=COMMIT,
    )
    write_cloud_index(tmp_path, [pack])
    stale = ExportTaskStatus(
        task_id="export-cancelled",
        snapshot_id="export-snapshot-cancelled",
        state=ExportTaskState.SYNCING,
        requested_at=now,
        updated_at=now,
        current_stage="syncing",
    )
    status_root = tmp_path / "status"
    status_root.mkdir(exist_ok=True)
    (status_root / "export-cancelled.json").write_text(
        stale.model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / "cancelled" / stale.task_id).mkdir(parents=True)
    completed_request = {
        "request_id": "export-request-current",
        "task_id": pack.task_id,
        "task_created": True,
        "state": "pending",
        "updated_at": (now.replace(minute=45)).isoformat(),
    }
    request_status = tmp_path / "requests" / "status" / "export-request-current.json"
    request_status.parent.mkdir(parents=True, exist_ok=True)
    request_status.write_text(json.dumps(completed_request), encoding="utf-8")
    completed_request_path = (
        tmp_path / "requests" / "completed" / "export-request-current.json"
    )
    completed_request_path.parent.mkdir(parents=True, exist_ok=True)
    completed_request_path.write_text("{}", encoding="utf-8")

    result = export_plane_status(tmp_path, export_date=now.date())

    assert result["state"] == "download_ready"
    assert result["task"] is None
    assert result["request"] is None


def test_export_plane_status_keeps_genuinely_running_task_visible(tmp_path: Path) -> None:
    now = datetime(2026, 7, 16, 14, 46, tzinfo=UTC)
    active = ExportTaskStatus(
        task_id="export-running",
        snapshot_id="export-snapshot-running",
        state=ExportTaskState.SYNCING,
        requested_at=now,
        updated_at=now,
        current_stage="syncing",
    )
    status_root = tmp_path / "status"
    status_root.mkdir(parents=True)
    (status_root / "export-running.json").write_text(
        active.model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / "running" / active.task_id).mkdir(parents=True)

    result = export_plane_status(tmp_path)

    assert result["state"] == "syncing"
    assert result["task"]["task_id"] == active.task_id


def _valid_pack(
    path: Path,
    *,
    extra_name: str | None = None,
    derived_bundle_sha: str = V5_SHA,
    derived_report_status: str = "FRESH",
) -> None:
    generated_at = "2026-07-16T00:05:00+00:00"
    derived_header = (
        "generated_at,source_bundle_sha256,proposal_snapshot_id,"
        "proposal_snapshot_sha256,proposal_content_snapshot_id,"
        "proposal_content_snapshot_sha256,derived_report_age_seconds,"
        "derived_report_status\n"
    )
    derived_row = (
        f"{generated_at},{derived_bundle_sha},proposal-snapshot:test,{'2' * 64},"
        f"proposal-content-snapshot:test,{'3' * 64},0,{derived_report_status}\n"
    )
    members = {
        "provenance.json": "{}\n",
        "data_quality.json": '{"status":"OK"}\n',
        "README.md": "read only\n",
        "executive_summary.md": "summary\n",
        "expert_questions.md": "- question\n",
        "diagnostics/export_timing.csv": "stage,seconds\nall,1\n",
        "diagnostics/export_timing.json": "{}\n",
        "reports/api_auth_production_slo.csv": derived_header + derived_row,
        "reports/paper_runtime_freshness.csv": derived_header + derived_row,
        "reports/paper_proposal_propagation_status.csv": derived_header + derived_row,
        "reports/system_acceptance_complete_status.json": json.dumps(
            {
                "generated_at": generated_at,
                "source_bundle_sha256": derived_bundle_sha,
                "proposal_snapshot_id": "proposal-snapshot:test",
                "proposal_snapshot_sha256": "2" * 64,
                "proposal_content_snapshot_id": "proposal-content-snapshot:test",
                "proposal_content_snapshot_sha256": "3" * 64,
                "derived_report_age_seconds": 0,
                "derived_report_status": derived_report_status,
            }
        ),
    }
    files = [
        {
            "path": name,
            "sha256": sha256_bytes(value.encode()),
            "rows": None,
        }
        for name, value in members.items()
    ]
    files.append({"path": "manifest.json", "sha256": None, "rows": None})
    if extra_name:
        members[extra_name] = "unsafe\n"
        files.append(
            {
                "path": extra_name,
                "sha256": sha256_bytes(b"unsafe\n"),
                "rows": None,
            }
        )
    manifest = {
        "files": files,
        "export_snapshot_id": "export-snapshot-test",
        "quant_lab_commit": COMMIT,
        "selected_v5_bundle_sha256": V5_SHA,
        "acceptance_set_id": "acceptance-test",
        "authoritative_snapshot": True,
        "generated_at": generated_at,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
        archive.writestr("manifest.json", json.dumps(manifest))


def test_validator_accepts_contract_pack(tmp_path: Path) -> None:
    pack = tmp_path / "valid.zip"
    _valid_pack(pack)
    report = validate_export_pack_locally(
        pack,
        task=_task(),
        snapshot=_snapshot(),
        pack_id="expert-pack-test",
    )
    assert report.valid is True
    assert report.failures == []


def test_validator_rejects_derived_report_bundle_sha_mismatch(tmp_path: Path) -> None:
    pack = tmp_path / "derived-mismatch.zip"
    _valid_pack(pack, derived_bundle_sha="c" * 64)

    report = validate_export_pack_locally(
        pack,
        task=_task(),
        snapshot=_snapshot(),
        pack_id="expert-pack-test",
    )

    assert report.valid is False
    assert report.checks["derived_reports_source_v5_bundle"] is False
    assert any(
        failure.startswith("derived_report_source_bundle_mismatch:")
        for failure in report.failures
    )


def test_validator_rejects_stale_derived_reports(tmp_path: Path) -> None:
    pack = tmp_path / "derived-stale.zip"
    _valid_pack(pack, derived_report_status="STALE_DERIVED_REPORT")

    report = validate_export_pack_locally(
        pack,
        task=_task(),
        snapshot=_snapshot(),
        pack_id="expert-pack-test",
    )

    assert report.valid is False
    assert report.checks["derived_reports_fresh"] is False
    assert any(failure.startswith("derived_report_stale:") for failure in report.failures)


def test_validator_rejects_zip_slip(tmp_path: Path) -> None:
    pack = tmp_path / "unsafe.zip"
    _valid_pack(pack, extra_name="../escape.txt")
    report = validate_export_pack_locally(
        pack,
        task=_task(),
        snapshot=_snapshot(),
        pack_id="expert-pack-test",
    )
    assert report.valid is False
    assert "check_failed:safe_member_paths" in report.failures


def test_receipt_summary_extracts_large_bounded_data_quality_member(tmp_path: Path) -> None:
    pack = tmp_path / "quality.zip"
    quality = {
        "status": "WARNING",
        "critical_count": 1,
        "warning_count": 8,
        "stale_dataset_count": 2,
        "missing_dataset_count": 3,
        "checks": [{"detail": "x" * 1024} for _ in range(300)],
    }
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("data_quality.json", json.dumps(quality))
        archive.writestr("expert_questions.md", "- question\n")

    summaries = accepted_module._bounded_pack_summaries(pack)  # noqa: SLF001

    assert summaries["data_quality_summary"] == {
        "status": "WARNING",
        "critical_count": 1,
        "warning_count": 8,
        "stale_dataset_count": 2,
        "missing_dataset_count": 3,
    }


def test_receipt_summary_rejects_unbounded_data_quality_member(tmp_path: Path) -> None:
    pack = tmp_path / "quality-too-large.zip"
    quality = {"status": "WARNING", "detail": "x" * (2 * 1024 * 1024)}
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("data_quality.json", json.dumps(quality))
        archive.writestr("expert_questions.md", "- question\n")

    with pytest.raises(RuntimeError, match="receipt_summary_member_too_large:data_quality.json"):
        accepted_module._bounded_pack_summaries(pack)  # noqa: SLF001


def test_accepted_publish_keeps_source_root_writable_until_atomic_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    temporary = tmp_path / "accepted" / ".incoming" / "task"
    final_dir = tmp_path / "accepted" / "2026" / "07" / "16" / "pack"
    nested = temporary / "nested"
    nested.mkdir(parents=True)
    payload = nested / "pack.zip"
    payload.write_bytes(b"pack")
    permission_calls: list[tuple[Path, int]] = []
    real_set_permissions = accepted_module._set_accepted_permissions

    def record_permissions(root, *, root_mode=0o550):
        permission_calls.append((Path(root), root_mode))
        real_set_permissions(root, root_mode=root_mode)

    monkeypatch.setattr(accepted_module, "_set_accepted_permissions", record_permissions)

    accepted_module._publish_accepted_directory(temporary, final_dir)  # noqa: SLF001

    assert permission_calls == [(temporary, 0o750)]
    assert (final_dir / "nested" / "pack.zip").read_bytes() == b"pack"
    if os.name != "nt":
        assert final_dir.stat().st_mode & 0o777 == 0o550
        assert (final_dir / "nested").stat().st_mode & 0o777 == 0o550
        assert (final_dir / "nested" / "pack.zip").stat().st_mode & 0o777 == 0o440


def test_failed_accepted_publish_can_remove_read_only_temporary_tree(tmp_path: Path) -> None:
    temporary = tmp_path / "accepted" / ".incoming" / "task"
    nested = temporary / "nested"
    nested.mkdir(parents=True)
    payload = nested / "pack.zip"
    payload.write_bytes(b"pack")
    accepted_module._set_accepted_permissions(temporary)  # noqa: SLF001

    accepted_module._remove_temporary_directory(temporary)  # noqa: SLF001

    assert not temporary.exists()


def test_existing_accepted_pack_repairs_missing_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pack_sha = "f" * 64
    pack_id = "expert-pack-existing"
    export_date = date(2026, 7, 16)
    accepted_root = tmp_path / "accepted"
    final_dir = accepted_root / "2026" / "07" / "16" / pack_id
    final_dir.mkdir(parents=True)
    (final_dir / "receipt.json").write_text("{}", encoding="utf-8")
    receipt = SimpleNamespace(pack_sha256=pack_sha)
    calls: list[tuple[Path, object, date]] = []
    monkeypatch.setattr(
        accepted_module.ExportWorkerReceipt,
        "model_validate_json",
        lambda _value: receipt,
    )
    monkeypatch.setattr(
        accepted_module,
        "_update_index",
        lambda path, value, day: calls.append((Path(path), value, day)),
    )
    result = SimpleNamespace(
        validation_report=SimpleNamespace(valid=True, zip_sha256=pack_sha),
        pack_manifest=SimpleNamespace(pack_id=pack_id),
    )

    returned_receipt, returned_dir = accepted_module.accept_materialized_pack(
        result=result,
        task=SimpleNamespace(export_date=export_date),
        snapshot=SimpleNamespace(),
        accepted_root=accepted_root,
        index_path=tmp_path / "accepted_index.json",
        worker_id="worker",
        worker_signing_key_path=tmp_path / "unused-key",
        worker_key_id="worker-key",
        cache_hits=0,
        downloaded_bytes=0,
    )

    assert returned_receipt is receipt
    assert returned_dir == final_dir
    assert calls == [(tmp_path / "accepted_index.json", receipt, export_date)]


def test_remote_worker_command_is_serialized_as_one_quoted_argument(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("quant_lab.export_worker.runner.subprocess.run", fake_run)
    config = SimpleNamespace(
        ssh_host="cloud.example",
        ssh_port=22,
        ssh_user="quant-export",
        ssh_key_path=tmp_path / "id_ed25519",
        known_hosts_path=tmp_path / "known_hosts",
    )
    script = "set -eu\nprintf '%s\\n' 'task with spaces'"

    _ssh(config, ["sh", "-lc", script])

    command = captured["command"]
    assert isinstance(command, list)
    assert command[-1].startswith("sh -lc '")
    assert "task with spaces" in command[-1]
    assert captured["kwargs"] == {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 120,
    }


def test_worker_ssh_options_reuse_one_authenticated_connection(tmp_path: Path) -> None:
    config = SimpleNamespace(
        ssh_host="cloud.example",
        ssh_port=22,
        ssh_user="quant-export",
        ssh_key_path=tmp_path / "id_ed25519",
        known_hosts_path=tmp_path / "known_hosts",
    )

    options = export_runner._ssh_options(config)  # noqa: SLF001
    joined = " ".join(options)

    assert "ControlMaster=auto" in joined
    assert "ControlPersist=300" in joined
    assert "ControlPath=/tmp/quant-export-ssh-%C" in joined
    assert "ServerAliveInterval=30" in joined
    assert "ServerAliveCountMax=3" in joined


def test_snapshot_transfer_ssh_options_use_independent_connections(tmp_path: Path) -> None:
    config = SimpleNamespace(
        ssh_host="cloud.example",
        ssh_port=22,
        ssh_user="quant-export",
        ssh_key_path=tmp_path / "id_ed25519",
        known_hosts_path=tmp_path / "known_hosts",
    )

    options = export_runner._snapshot_transfer_ssh_options(config)  # noqa: SLF001
    joined = " ".join(options)

    assert "ControlMaster=no" in joined
    assert "ControlPersist" not in joined
    assert "ControlPath" not in joined
    assert "ServerAliveInterval=30" in joined
    assert "ServerAliveCountMax=3" in joined


def test_snapshot_batch_fetch_uses_independent_connection_options(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = b"snapshot"
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        info = tarfile.TarInfo("lake/data.bin")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    captured: dict[str, list[str]] = {}

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            captured["command"] = command
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(archive_bytes.getvalue())

        def wait(self, timeout):
            assert timeout == 3600
            return 0

        def kill(self):
            raise AssertionError("successful transfer must not be killed")

    monkeypatch.setattr(export_runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        export_runner,
        "_snapshot_transfer_ssh_options",
        lambda _config: ["independent-transfer-option"],
    )
    config = SimpleNamespace(
        ssh_host="cloud.example",
        ssh_user="quant-export",
        remote_queue_root="/queue",
    )

    export_runner._tar_snapshot_files_from(  # noqa: SLF001
        config,
        snapshot_id="snapshot-1",
        relative_paths=["lake/data.bin"],
        target_root=tmp_path / "target",
    )

    assert "independent-transfer-option" in captured["command"]
    assert (tmp_path / "target" / "lake" / "data.bin").read_bytes() == payload


def test_worker_uploads_are_group_readable_for_cloud_services(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("quant_lab.export_worker.runner.subprocess.run", fake_run)
    config = SimpleNamespace(
        ssh_host="cloud.example",
        ssh_port=22,
        ssh_user="quant-export",
        ssh_key_path=tmp_path / "id_ed25519",
        known_hosts_path=tmp_path / "known_hosts",
    )
    local = tmp_path / "status.json"
    local.write_text("{}\n", encoding="utf-8")

    export_runner._scp_to(  # noqa: SLF001
        config,
        local,
        "/queue/status/task.json.partial",
    )

    assert commands[0][0] == "scp"
    assert commands[1][0] == "ssh"
    assert commands[1][-1] == "chmod 0660 /queue/status/task.json.partial"


def test_worker_status_uploads_are_serialized(tmp_path: Path, monkeypatch) -> None:
    active_uploads = 0
    max_active_uploads = 0
    active_lock = threading.Lock()
    start = threading.Barrier(3)

    def fake_scp_to(_config, _local, _remote):
        nonlocal active_uploads, max_active_uploads
        with active_lock:
            active_uploads += 1
            max_active_uploads = max(max_active_uploads, active_uploads)
        time.sleep(0.05)
        with active_lock:
            active_uploads -= 1

    monkeypatch.setattr(export_runner, "_scp_to", fake_scp_to)
    monkeypatch.setattr(
        export_runner,
        "_ssh",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(export_runner, "_write_worker_status", lambda *_args, **_kwargs: None)
    config = SimpleNamespace(remote_queue_root="/queue", worker_id="worker-1")
    task = _task()
    work = tmp_path / "work"
    work.mkdir()
    statuses = [
        export_runner._status(task, ExportTaskState.SYNCING, config, "syncing"),
        export_runner._status(task, ExportTaskState.MATERIALIZING, config, "materializing"),
    ]

    def upload(status):
        start.wait()
        export_runner._upload_status(config, status, work)  # noqa: SLF001

    threads = [threading.Thread(target=upload, args=(status,)) for status in statuses]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert max_active_uploads == 1


def test_worker_streams_snapshot_batch_without_extracting_unexpected_members(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = b"snapshot"
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo("lake/gold/example/data.bin")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(archive_bytes.getvalue())
            self.killed = False

        def wait(self, timeout: int) -> int:
            assert timeout in {10, 3600}
            return 0

        def kill(self) -> None:
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr(
        "quant_lab.export_worker.runner.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    config = SimpleNamespace(
        ssh_host="cloud.example",
        ssh_port=22,
        ssh_user="quant-export",
        ssh_key_path=tmp_path / "id_ed25519",
        known_hosts_path=tmp_path / "known_hosts",
        remote_queue_root="/queue",
    )
    target = tmp_path / "incoming"

    export_runner._tar_snapshot_files_from(  # noqa: SLF001
        config,
        snapshot_id="export-snapshot-test",
        relative_paths=["lake/gold/example/data.bin"],
        target_root=target,
    )

    assert (target / "lake/gold/example/data.bin").read_bytes() == payload
    assert process.killed is False


def test_worker_rejects_unexpected_snapshot_batch_member(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo("../escape.bin")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(archive_bytes.getvalue())
            self.killed = False

        def wait(self, timeout: int) -> int:
            assert timeout in {10, 3600}
            return 0

        def kill(self) -> None:
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr(
        "quant_lab.export_worker.runner.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    config = SimpleNamespace(
        ssh_host="cloud.example",
        ssh_port=22,
        ssh_user="quant-export",
        ssh_key_path=tmp_path / "id_ed25519",
        known_hosts_path=tmp_path / "known_hosts",
        remote_queue_root="/queue",
    )

    with pytest.raises(RuntimeError, match="unexpected_snapshot_batch_member"):
        export_runner._tar_snapshot_files_from(  # noqa: SLF001
            config,
            snapshot_id="export-snapshot-test",
            relative_paths=["lake/gold/example/data.bin"],
            target_root=tmp_path / "incoming",
        )

    assert process.killed is True
    assert not (tmp_path / "escape.bin").exists()


def test_current_main_commit_uses_github_pr_base_sha_when_remote_ref_is_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps({"pull_request": {"base": {"sha": COMMIT}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_export, "_git_ref_commit_full", lambda _ref: None)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))

    assert daily_export._current_main_commit_full() == COMMIT
