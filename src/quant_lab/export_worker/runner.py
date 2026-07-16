from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from quant_lab.export_materializer.writer import materialize_snapshot_pack
from quant_lab.export_plane.contracts import (
    ExportSnapshotManifest,
    ExportTask,
    ExportTaskState,
    ExportTaskStatus,
)
from quant_lab.export_plane.signatures import load_public_key, verify_payload
from quant_lab.export_plane.snapshot import verify_snapshot_manifest_digest
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.export_worker.accepted import (
    accept_materialized_pack,
    mark_control_plane_receipt_verified,
)
from quant_lab.export_worker.retention import enforce_accepted_retention
from quant_lab.export_worker.sync import sync_snapshot_blobs

LOG = logging.getLogger("quant_export_worker")
TASK_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,180}$")
_STOP = False
_STATUS_UPLOAD_LOCK = threading.RLock()


class Config:
    def __init__(self) -> None:
        self.ssh_host = _required("QLAB_SSH_HOST")
        self.ssh_port = int(os.getenv("QLAB_SSH_PORT", "22"))
        self.ssh_user = _required("QLAB_SSH_USER")
        self.ssh_key_path = Path(os.getenv("QLAB_SSH_KEY_PATH", "/run/secrets/qlab_ssh_key"))
        self.known_hosts_path = Path(
            os.getenv("QLAB_SSH_KNOWN_HOSTS", "/run/secrets/known_hosts")
        )
        self.remote_queue_root = os.getenv(
            "QLAB_REMOTE_QUEUE_ROOT",
            "/var/lib/quant-lab/export_queue",
        ).rstrip("/")
        self.worker_id = os.getenv("WORKER_ID", socket.gethostname())
        self.data_dir = Path(os.getenv("WORKER_DATA_DIR", "/data"))
        self.accepted_root = Path(os.getenv("NAS_EXPORT_ACCEPTED_ROOT", "/data/accepted"))
        self.index_path = Path(
            os.getenv("NAS_EXPORT_INDEX_PATH", "/data/accepted_index.json")
        )
        self.worker_status_path = Path(
            os.getenv("NAS_EXPORT_WORKER_STATUS_PATH", "/data/status/worker.json")
        )
        self.worker_signing_key = Path(
            os.getenv("WORKER_SIGNING_KEY_PATH", "/run/secrets/nas_export_signing_key")
        )
        self.worker_key_id = os.getenv("WORKER_KEY_ID", "nas-export-v1")
        self.cloud_public_key = Path(
            os.getenv("CLOUD_PUBLIC_KEY_PATH", "/run/secrets/cloud_export_public_key")
        )
        self.cloud_key_id = os.getenv("CLOUD_KEY_ID", "cloud-export-v1")
        self.worker_commit = _required("BUILD_GIT_COMMIT").lower()
        self.poll_interval = max(5, int(os.getenv("POLL_INTERVAL_SECONDS", "30")))
        self.heartbeat_seconds = max(10, int(os.getenv("HEARTBEAT_SECONDS", "30")))
        self.run_once = _bool_env("RUN_ONCE", False)
        self.max_snapshot_bytes = int(os.getenv("MAX_SNAPSHOT_BYTES", str(20 * 1024**3)))
        self.max_pack_bytes = int(os.getenv("MAX_PACK_BYTES", str(10 * 1024**3)))
        self.min_free_disk_bytes = int(os.getenv("MIN_FREE_DISK_BYTES", str(20 * 1024**3)))
        self.max_task_age_seconds = int(os.getenv("MAX_TASK_AGE_SECONDS", str(48 * 3600)))
        self.receipt_wait_seconds = int(os.getenv("RECEIPT_WAIT_SECONDS", "900"))
        self.heavy_lock_path = Path(
            os.getenv("NAS_HEAVY_JOB_LOCK_PATH", "/runtime/heavy-job.lock")
        )
        self.retention_days = int(os.getenv("NAS_EXPORT_RETENTION_DAYS", "90"))
        self.max_total_bytes = int(float(os.getenv("NAS_EXPORT_MAX_TOTAL_GB", "200")) * 1024**3)
        self.min_keep_packs = int(os.getenv("NAS_EXPORT_MIN_KEEP_PACKS", "30"))
        for required_path in (
            self.ssh_key_path,
            self.known_hosts_path,
            self.worker_signing_key,
            self.cloud_public_key,
        ):
            if not required_path.is_file():
                raise FileNotFoundError(f"required worker secret is missing: {required_path}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_root.mkdir(parents=True, exist_ok=True)
        self.heavy_lock_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    config = Config()
    LOG.info(
        "worker_start worker_id=%s worker_commit=%s storage=nas_local",
        config.worker_id,
        config.worker_commit,
    )
    _write_worker_status(config, state="idle", current_stage="waiting_for_task")
    while not _STOP:
        try:
            task_id = claim_next_task(config)
            if task_id is None:
                _reconcile_control_plane_verifications(config)
                if config.run_once:
                    return 0
                _sleep(config.poll_interval)
                continue
            process_claimed_task(config, task_id)
        except Exception:  # noqa: BLE001
            LOG.exception("worker_loop_error")
            if config.run_once:
                return 2
            _sleep(config.poll_interval)
        if config.run_once:
            return 0
    return 0


def claim_next_task(config: Config) -> str | None:
    root = shlex.quote(config.remote_queue_root)
    script = f"""
set -eu
umask 0007
root={root}
task=""
for candidate in "$root"/pending/*; do
  [ -d "$candidate" ] || continue
  task=$(basename "$candidate")
  break
done
[ -n "$task" ] || exit 3
case "$task" in *[!A-Za-z0-9_.:-]*|'') exit 4 ;; esac
mv "$root/pending/$task" "$root/running/$task"
printf '%s\n' "$task"
"""
    result = _ssh(config, ["sh", "-lc", script], check=False)
    if result.returncode == 3:
        return None
    if result.returncode != 0:
        raise RuntimeError(f"claim_failed:{_tail(result.stderr)}")
    task_id = result.stdout.strip().splitlines()[-1]
    if not TASK_ID_RE.fullmatch(task_id):
        raise ValueError("unsafe_remote_task_id")
    return task_id


def process_claimed_task(config: Config, task_id: str) -> None:
    work = config.data_dir / "work" / task_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        task_path = work / "task.json"
        _scp_from(config, _remote_task_path(config, task_id), task_path)
        task = ExportTask.model_validate_json(task_path.read_text(encoding="utf-8"))
        _verify_task(config, task, task_id)
        snapshot_path = work / "snapshot_manifest.json"
        remote_snapshot = (
            f"{config.remote_queue_root}/snapshots/{task.snapshot_id}/manifest.json"
        )
        _scp_from(config, remote_snapshot, snapshot_path)
        snapshot = ExportSnapshotManifest.model_validate_json(
            snapshot_path.read_text(encoding="utf-8")
        )
        _verify_snapshot(config, task, snapshot)
        status = _status(task, ExportTaskState.SYNCING, config, "syncing")
        _upload_status(config, status, work)

        with _heartbeat(config, task, work), _heavy_job_lock(config.heavy_lock_path):
            sync_result = sync_snapshot_blobs(
                snapshot,
                data_root=config.data_dir,
                fetch_blob=lambda relative, target: _scp_from(
                    config,
                    (
                        f"{config.remote_queue_root}/snapshots/{task.snapshot_id}/files/"
                        f"{relative}"
                    ),
                    target,
                ),
                fetch_blobs=lambda references, target: _tar_snapshot_files_from(
                    config,
                    snapshot_id=task.snapshot_id,
                    relative_paths=[item.relative_path for item in references],
                    target_root=target,
                ),
                min_free_disk_bytes=config.min_free_disk_bytes,
                max_snapshot_bytes=config.max_snapshot_bytes,
            )
            status = _status(
                task,
                ExportTaskState.MATERIALIZING,
                config,
                "materializing",
                input_bytes=snapshot.total_input_bytes,
            )
            _upload_status(config, status, work)
            result = materialize_snapshot_pack(
                snapshot_root=sync_result.snapshot_root / "files",
                task=task,
                snapshot=snapshot,
                work_root=work / "materializer",
                worker_id=config.worker_id,
                worker_commit=config.worker_commit,
            )
            if result.pack_path.stat().st_size > config.max_pack_bytes:
                raise RuntimeError("pack_output_limit_exceeded")
            status = _status(
                task,
                ExportTaskState.VALIDATING_ON_NAS,
                config,
                "validating_on_nas",
                input_bytes=snapshot.total_input_bytes,
                output_bytes=result.pack_path.stat().st_size,
            )
            _upload_status(config, status, work)
            receipt, _ = accept_materialized_pack(
                result=result,
                task=task,
                snapshot=snapshot,
                accepted_root=config.accepted_root,
                index_path=config.index_path,
                worker_id=config.worker_id,
                worker_signing_key_path=config.worker_signing_key,
                worker_key_id=config.worker_key_id,
                cache_hits=sync_result.cache_hits,
                downloaded_bytes=sync_result.downloaded_bytes,
            )

        status = _status(
            task,
            ExportTaskState.ACCEPTED_ON_NAS,
            config,
            "accepted_on_nas",
            input_bytes=snapshot.total_input_bytes,
            output_bytes=receipt.pack_size_bytes,
            nas_pack_id=receipt.pack_id,
            nas_pack_sha256=receipt.pack_sha256,
            nas_download_path=receipt.download_relative_path,
        )
        _upload_status(config, status, work)
        _upload_receipt(config, task_id, receipt, work)
        if _wait_for_control_plane_verification(config, task_id):
            mark_control_plane_receipt_verified(config.index_path, receipt.pack_id)
        enforce_accepted_retention(
            accepted_root=config.accepted_root,
            index_path=config.index_path,
            retention_days=config.retention_days,
            max_total_bytes=config.max_total_bytes,
            min_keep_packs=config.min_keep_packs,
            audit_log_path=config.data_dir / "audit" / "retention.jsonl",
        )
        LOG.info(
            "task_completed task_id=%s pack_id=%s cache_hits=%s downloaded_bytes=%s",
            task_id,
            receipt.pack_id,
            sync_result.cache_hits,
            sync_result.downloaded_bytes,
        )
        _write_worker_status(
            config,
            state="completed",
            current_stage="waiting_for_task",
            task_id=task_id,
            pack_id=receipt.pack_id,
            pack_size_bytes=receipt.pack_size_bytes,
            cache_hit_count=sync_result.cache_hits,
            downloaded_input_bytes=sync_result.downloaded_bytes,
        )
        shutil.rmtree(work, ignore_errors=True)
    except Exception as exc:
        LOG.exception("task_failed task_id=%s", task_id)
        _write_worker_status(
            config,
            state="failed",
            current_stage="failed",
            task_id=task_id,
            last_error=f"{type(exc).__name__}: {exc}"[:2000],
        )
        _mark_failed(config, task_id, work, exc)
        raise


def _verify_task(config: Config, task: ExportTask, task_id: str) -> None:
    if task.task_id != task_id:
        raise ValueError("task_id_mismatch")
    if task.signature_key_id != config.cloud_key_id:
        raise ValueError("unknown_cloud_signing_key")
    verify_payload(task, task.signature, load_public_key(config.cloud_public_key))
    if task.expected_worker_commit != config.worker_commit:
        raise ValueError("worker_code_mismatch")
    age = datetime.now(UTC) - task.requested_at
    if age < timedelta(seconds=-300) or age > timedelta(seconds=config.max_task_age_seconds):
        raise ValueError("task_replay_or_expired")


def _verify_snapshot(
    config: Config,
    task: ExportTask,
    snapshot: ExportSnapshotManifest,
) -> None:
    if snapshot.signature_key_id != config.cloud_key_id:
        raise ValueError("unknown_cloud_snapshot_signing_key")
    verify_payload(snapshot, snapshot.signature, load_public_key(config.cloud_public_key))
    verify_snapshot_manifest_digest(snapshot)
    checks = (
        task.snapshot_id == snapshot.snapshot_id,
        task.snapshot_manifest_sha256 == snapshot.manifest_sha256,
        task.quant_lab_commit == snapshot.quant_lab_commit,
        task.selected_v5_bundle_sha256 == snapshot.selected_v5_bundle_sha256,
        task.acceptance_set_id == snapshot.acceptance_set_id,
    )
    if not all(checks):
        raise ValueError("task_snapshot_binding_mismatch")


def _upload_receipt(config: Config, task_id: str, receipt, work: Path) -> None:
    receipt_path = work / "receipt.json"
    atomic_write_json(receipt_path, receipt.model_dump(mode="json"), mode=0o600)
    remote_inbox = f"{config.remote_queue_root}/receipts/inbox"
    remote_temp = f"{remote_inbox}/.{task_id}.{os.getpid()}.partial"
    remote_final = f"{remote_inbox}/{task_id}"
    _ssh(config, ["mkdir", "-p", remote_temp])
    _scp_to(config, receipt_path, f"{remote_temp}/receipt.json")
    script = (
        f"test ! -e {shlex.quote(remote_final)} && "
        f"mv {shlex.quote(remote_temp)} {shlex.quote(remote_final)}"
    )
    result = _ssh(config, ["sh", "-lc", script], check=False)
    if result.returncode != 0:
        _ssh(config, ["rm", "-rf", remote_temp], check=False)
        existing = _ssh(config, ["test", "-f", f"{remote_final}/receipt.json"], check=False)
        if existing.returncode != 0:
            raise RuntimeError(f"receipt_upload_failed:{_tail(result.stderr)}")


def _wait_for_control_plane_verification(config: Config, task_id: str) -> bool:
    deadline = time.monotonic() + config.receipt_wait_seconds
    status_path = f"{config.remote_queue_root}/status/{task_id}.json"
    while time.monotonic() < deadline and not _STOP:
        result = _ssh(config, ["cat", status_path], check=False)
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = {}
            if payload.get("state") == ExportTaskState.DOWNLOAD_READY:
                return True
            if payload.get("state") == ExportTaskState.FAILED:
                return False
        _sleep(min(15, config.poll_interval))
    return False


def _reconcile_control_plane_verifications(config: Config) -> None:
    try:
        payload = json.loads(config.index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for row in payload.get("packs", []):
        if not isinstance(row, dict) or row.get("control_plane_receipt_verified"):
            continue
        task_id = str(row.get("task_id") or "")
        pack_id = str(row.get("pack_id") or "")
        if not TASK_ID_RE.fullmatch(task_id) or not pack_id:
            continue
        status_path = f"{config.remote_queue_root}/status/{task_id}.json"
        result = _ssh(config, ["cat", status_path], check=False)
        if result.returncode != 0:
            continue
        try:
            state = json.loads(result.stdout).get("state")
        except json.JSONDecodeError:
            continue
        if state == ExportTaskState.DOWNLOAD_READY:
            mark_control_plane_receipt_verified(config.index_path, pack_id)


def _status(
    task: ExportTask,
    state: ExportTaskState,
    config: Config,
    stage: str,
    **updates: Any,
) -> ExportTaskStatus:
    now = datetime.now(UTC)
    return ExportTaskStatus(
        task_id=task.task_id,
        snapshot_id=task.snapshot_id,
        state=state,
        requested_at=task.requested_at,
        updated_at=now,
        claimed_at=now,
        heartbeat_at=now,
        lease_expires_at=now + timedelta(seconds=task.lease_seconds),
        worker_id=config.worker_id,
        attempt=1,
        max_attempts=task.max_attempts,
        current_stage=stage,
        **updates,
    )


def _upload_status(config: Config, status: ExportTaskStatus, work: Path) -> None:
    with _STATUS_UPLOAD_LOCK:
        _upload_status_locked(config, status, work)


def _upload_status_locked(config: Config, status: ExportTaskStatus, work: Path) -> None:
    local = work / "status.json"
    atomic_write_json(local, status.model_dump(mode="json"), mode=0o600)
    remote = f"{config.remote_queue_root}/status/{status.task_id}.json"
    partial = f"{remote}.{config.worker_id}.partial"
    _scp_to(config, local, partial)
    _ssh(config, ["mv", partial, remote])
    _write_worker_status(
        config,
        state=status.state.value,
        current_stage=status.current_stage,
        task_id=status.task_id,
        input_bytes=status.input_bytes,
        output_bytes=status.output_bytes,
    )


def _write_worker_status(config: Config, **fields: Any) -> None:
    atomic_write_json(
        config.worker_status_path,
        {
            "schema_version": "quant_lab_nas_export_worker_status.v1",
            "worker_id": config.worker_id,
            "worker_commit": config.worker_commit,
            "heartbeat_at": datetime.now(UTC).isoformat(),
            **fields,
        },
        mode=0o640,
    )


@contextmanager
def _heartbeat(
    config: Config,
    task: ExportTask,
    work: Path,
):
    stop = threading.Event()

    def run() -> None:
        while not stop.wait(config.heartbeat_seconds):
            try:
                with _STATUS_UPLOAD_LOCK:
                    status = ExportTaskStatus.model_validate_json(
                        (work / "status.json").read_text(encoding="utf-8")
                    )
                    now = datetime.now(UTC)
                    status = status.model_copy(
                        update={
                            "updated_at": now,
                            "heartbeat_at": now,
                            "lease_expires_at": now + timedelta(seconds=task.lease_seconds),
                        }
                    )
                    _upload_status_locked(config, status, work)
            except Exception:
                LOG.warning("heartbeat_failed task_id=%s", task.task_id, exc_info=True)

    thread = threading.Thread(target=run, name=f"heartbeat-{task.task_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


@contextmanager
def _heavy_job_lock(path: Path):
    import fcntl

    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _mark_failed(config: Config, task_id: str, work: Path, exc: Exception) -> None:
    last_error = f"{type(exc).__name__}: {exc}"[:4000]
    error = {
        "task_id": task_id,
        "failed_at": datetime.now(UTC).isoformat(),
        "worker_id": config.worker_id,
        "error_code": str(exc)[:200],
        "error_type": type(exc).__name__,
    }
    local = work / "worker_error.json"
    atomic_write_json(local, error, mode=0o600)
    try:
        task = ExportTask.model_validate_json((work / "task.json").read_text(encoding="utf-8"))
        failed_status = _status(
            task,
            ExportTaskState.FAILED,
            config,
            "failed",
            last_error=last_error,
        )
        _upload_status(config, failed_status, work)
    except Exception:
        LOG.warning("failed_to_publish_failed_status task_id=%s", task_id, exc_info=True)
    try:
        _scp_to(
            config,
            local,
            f"{config.remote_queue_root}/running/{task_id}/worker_error.json",
        )
        root = shlex.quote(config.remote_queue_root)
        task = shlex.quote(task_id)
        _ssh(
            config,
            [
                "sh",
                "-lc",
                f"test ! -e {root}/failed/{task} && mv {root}/running/{task} {root}/failed/{task}",
            ],
            check=False,
        )
    except Exception:
        LOG.warning("failed_to_publish_worker_error task_id=%s", task_id, exc_info=True)


def _remote_task_path(config: Config, task_id: str) -> str:
    return f"{config.remote_queue_root}/running/{task_id}/task.json"


def _ssh(
    config: Config,
    remote_command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    serialized = " ".join(shlex.quote(value) for value in remote_command)
    command = [
        "ssh",
        *_ssh_options(config),
        f"{config.ssh_user}@{config.ssh_host}",
        serialized,
    ]
    return subprocess.run(command, check=check, capture_output=True, text=True, timeout=120)


def _scp_from(config: Config, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial = local_path.with_name(local_path.name + ".partial")
    partial.unlink(missing_ok=True)
    command = [
        "scp",
        *_scp_options(config),
        f"{config.ssh_user}@{config.ssh_host}:{remote_path}",
        str(partial),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600)
    os.replace(partial, local_path)


def _scp_to(config: Config, local_path: Path, remote_path: str) -> None:
    command = [
        "scp",
        *_scp_options(config),
        str(local_path),
        f"{config.ssh_user}@{config.ssh_host}:{remote_path}",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    _ssh(config, ["chmod", "0660", remote_path])


def _tar_snapshot_files_from(
    config: Config,
    *,
    snapshot_id: str,
    relative_paths: list[str],
    target_root: Path,
) -> None:
    if not relative_paths:
        return
    expected = set(relative_paths)
    if len(expected) != len(relative_paths):
        raise RuntimeError("duplicate_snapshot_batch_path")
    remote_root = f"{config.remote_queue_root}/snapshots/{snapshot_id}/files"
    remote_command = [
        "tar",
        "-C",
        remote_root,
        "-cf",
        "-",
        "--null",
        "--verbatim-files-from",
        "--files-from=-",
    ]
    serialized = " ".join(shlex.quote(value) for value in remote_command)
    command = [
        "ssh",
        *_ssh_options(config),
        f"{config.ssh_user}@{config.ssh_host}",
        serialized,
    ]
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(mode="w+b") as stderr:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            process.stdin.write(
                b"\0".join(path.encode("utf-8") for path in relative_paths) + b"\0"
            )
            process.stdin.close()
            seen: set[str] = set()
            with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
                for member in archive:
                    if member.name not in expected or member.name in seen:
                        raise RuntimeError(f"unexpected_snapshot_batch_member:{member.name}")
                    if not member.isfile():
                        raise RuntimeError(f"non_file_snapshot_batch_member:{member.name}")
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"unreadable_snapshot_batch_member:{member.name}")
                    destination = target_root / member.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    seen.add(member.name)
            return_code = process.wait(timeout=3600)
            if return_code != 0:
                stderr.seek(0)
                detail = stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"snapshot_batch_fetch_failed:{_tail(detail)}")
            missing = expected - seen
            if missing:
                raise RuntimeError(f"snapshot_batch_members_missing:{len(missing)}")
        except Exception:
            process.kill()
            process.wait(timeout=10)
            raise


def _ssh_options(config: Config) -> list[str]:
    return [
        "-p",
        str(config.ssh_port),
        "-i",
        str(config.ssh_key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={config.known_hosts_path}",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=300",
        "-o",
        "ControlPath=/tmp/quant-export-ssh-%C",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]


def _scp_options(config: Config) -> list[str]:
    result = _ssh_options(config)
    result[0] = "-P"
    return result


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _request_stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _sleep(seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while not _STOP and time.monotonic() < deadline:
        time.sleep(min(1, deadline - time.monotonic()))


def _tail(value: str, limit: int = 1000) -> str:
    return value.strip()[-limit:]


if __name__ == "__main__":
    raise SystemExit(main())
