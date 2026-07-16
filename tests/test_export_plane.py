from __future__ import annotations

import io
import json
import os
import tarfile
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from quant_lab.export import daily as daily_export
from quant_lab.export_materializer.validator import validate_export_pack_locally
from quant_lab.export_plane import snapshot as snapshot_module
from quant_lab.export_plane.contracts import (
    ExportDatasetReference,
    ExportSnapshotManifest,
    ExportTask,
)
from quant_lab.export_plane.signatures import (
    load_public_key,
    load_signing_key,
    sha256_bytes,
    sign_payload,
    verify_payload,
)
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
        quant_lab_version="0.1.0",
        v5_commit="d" * 40,
        selected_v5_bundle_name="v5.tar.gz",
        selected_v5_bundle_sha256=V5_SHA,
        acceptance_set_id="acceptance-test",
        risk_permission_identity="risk:test",
        paper_lifecycle_identity="paper:test",
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
        min_free_disk_bytes=0,
        max_snapshot_bytes=1024,
    )

    assert batches == [list(payloads)]
    assert result.cache_hits == 0
    assert result.downloaded_files == 2
    assert result.downloaded_bytes == 6
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


def _valid_pack(path: Path, *, extra_name: str | None = None) -> None:
    members = {
        "provenance.json": "{}\n",
        "data_quality.json": '{"status":"OK"}\n',
        "README.md": "read only\n",
        "executive_summary.md": "summary\n",
        "expert_questions.md": "- question\n",
        "diagnostics/export_timing.csv": "stage,seconds\nall,1\n",
        "diagnostics/export_timing.json": "{}\n",
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
