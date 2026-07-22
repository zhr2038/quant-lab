from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - Linux NAS always provides resource.
    resource = None  # type: ignore[assignment]

from quant_lab.research_plane.contracts import (
    DEFAULT_FACTOR_FACTORY_MAX_INPUT_UNCOMPRESSED_BYTES,
    DEFAULT_FACTOR_FACTORY_MAX_RESULT_BYTES,
    DEFAULT_FACTOR_FACTORY_MAX_SNAPSHOT_BYTES,
    DEFAULT_FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES,
    DEFAULT_FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES,
    DEFAULT_RESEARCH_MAX_RESULT_BYTES,
    RESEARCH_SNAPSHOT_ADAPTER,
    RESEARCH_TASK_ADAPTER,
    AlphaFactorySnapshotManifest,
    AlphaFactoryTask,
    FactorFactorySnapshotManifest,
    FactorFactoryTask,
    FactorResearchSnapshotManifest,
    FactorResearchTask,
    ResearchTaskEnvelope,
    ResearchTaskLease,
    ResearchTaskState,
    ResearchTaskStatus,
)
from quant_lab.research_plane.result import validate_research_task_snapshot
from quant_lab.research_plane.signatures import (
    load_public_key,
    load_signing_key,
)
from quant_lab.research_worker.alpha_factory import compute_alpha_factory_from_snapshot
from quant_lab.research_worker.entry_quality_history import (
    compute_entry_quality_history_from_snapshot,
)
from quant_lab.research_worker.factor_factory import compute_factor_factory_result
from quant_lab.research_worker.factor_research import compute_factor_research_result
from quant_lab.research_worker.result_writer import (
    write_alpha_factory_result_bundle,
    write_entry_quality_history_result_bundle,
    write_factor_factory_result_bundle,
    write_factor_research_result_bundle,
)
from quant_lab.transfer.snapshot_sync import sync_snapshot_blobs

LOG = logging.getLogger("quant_research_worker")
STOP = threading.Event()
_STATUS_UPLOAD_LOCK = threading.RLock()
_LEASE_UPLOAD_LOCK = threading.RLock()
_HANDOFF_READY_MARKER = ".HANDOFF_READY"
_NON_RETRYABLE_TASK_ERRORS = frozenset({"worker_code_mismatch"})


@dataclass(frozen=True)
class Config:
    cloud_host: str
    cloud_user: str
    cloud_port: int
    ssh_key_path: Path
    known_hosts_path: Path
    cloud_queue_root: str
    data_root: Path
    task_public_key_path: Path
    task_key_id: str
    worker_signing_key_path: Path
    worker_key_id: str
    worker_id: str
    worker_commit: str
    run_once: bool
    poll_seconds: int
    heartbeat_seconds: int
    min_free_disk_bytes: int
    max_snapshot_bytes: int
    max_result_bytes: int
    heavy_job_lock: Path
    batch_fetch_workers: int
    factor_factory_max_result_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_RESULT_BYTES
    factor_factory_max_snapshot_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_SNAPSHOT_BYTES
    factor_factory_max_value_partition_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES
    factor_factory_enabled: bool = False
    factor_factory_max_file_count: int = 20_000
    factor_factory_max_uncompressed_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES
    factor_factory_max_input_uncompressed_bytes: int = (
        DEFAULT_FACTOR_FACTORY_MAX_INPUT_UNCOMPRESSED_BYTES
    )
    ssh_timeout_seconds: int = 90
    scp_timeout_seconds: int = 900

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            cloud_host=_required("QUANT_RESEARCH_CLOUD_HOST"),
            cloud_user=_required("QUANT_RESEARCH_CLOUD_USER"),
            cloud_port=int(os.environ.get("QUANT_RESEARCH_CLOUD_PORT", "22")),
            ssh_key_path=Path(_required("QUANT_RESEARCH_SSH_KEY_PATH")),
            known_hosts_path=Path(_required("QUANT_RESEARCH_KNOWN_HOSTS_PATH")),
            cloud_queue_root=os.environ.get(
                "QUANT_LAB_RESEARCH_QUEUE_ROOT", "/var/lib/quant-lab/research_queue"
            ),
            data_root=Path(os.environ.get("QUANT_RESEARCH_DATA_ROOT", "/data")),
            task_public_key_path=Path(_required("QUANT_RESEARCH_TASK_PUBLIC_KEY_PATH")),
            task_key_id=_required("QUANT_RESEARCH_TASK_KEY_ID"),
            worker_signing_key_path=Path(_required("QUANT_RESEARCH_WORKER_SIGNING_KEY_PATH")),
            worker_key_id=_required("QUANT_RESEARCH_WORKER_KEY_ID"),
            worker_id=os.environ.get("QUANT_RESEARCH_WORKER_ID", "nas-research-worker-01"),
            worker_commit=_required("QUANT_RESEARCH_WORKER_COMMIT"),
            run_once=_bool_env("RUN_ONCE", False),
            poll_seconds=max(5, int(os.environ.get("POLL_SECONDS", "30"))),
            heartbeat_seconds=max(10, int(os.environ.get("HEARTBEAT_SECONDS", "30"))),
            min_free_disk_bytes=int(os.environ.get("MIN_FREE_DISK_BYTES", str(5 * 1024**3))),
            max_snapshot_bytes=int(os.environ.get("MAX_SNAPSHOT_BYTES", str(250 * 1024**3))),
            max_result_bytes=int(
                os.environ.get("MAX_RESULT_BYTES", str(DEFAULT_RESEARCH_MAX_RESULT_BYTES))
            ),
            factor_factory_max_result_bytes=int(
                os.environ.get(
                    "FACTOR_FACTORY_MAX_RESULT_BYTES",
                    str(DEFAULT_FACTOR_FACTORY_MAX_RESULT_BYTES),
                )
            ),
            factor_factory_max_snapshot_bytes=int(
                os.environ.get(
                    "FACTOR_FACTORY_MAX_SNAPSHOT_BYTES",
                    str(DEFAULT_FACTOR_FACTORY_MAX_SNAPSHOT_BYTES),
                )
            ),
            factor_factory_max_value_partition_bytes=int(
                os.environ.get(
                    "FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES",
                    str(DEFAULT_FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES),
                )
            ),
            factor_factory_enabled=_bool_env("QUANT_RESEARCH_FACTOR_FACTORY_ENABLED", False),
            factor_factory_max_file_count=int(
                os.environ.get("FACTOR_FACTORY_MAX_FILE_COUNT", "20000")
            ),
            factor_factory_max_uncompressed_bytes=int(
                os.environ.get(
                    "FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES",
                    str(DEFAULT_FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES),
                )
            ),
            factor_factory_max_input_uncompressed_bytes=int(
                os.environ.get(
                    "FACTOR_FACTORY_MAX_INPUT_UNCOMPRESSED_BYTES",
                    str(DEFAULT_FACTOR_FACTORY_MAX_INPUT_UNCOMPRESSED_BYTES),
                )
            ),
            heavy_job_lock=Path(
                os.environ.get("QUANT_HEAVY_JOB_LOCK", "/runtime/quant-runtime/heavy-job.lock")
            ),
            batch_fetch_workers=max(1, min(4, int(os.environ.get("BATCH_FETCH_WORKERS", "3")))),
            ssh_timeout_seconds=max(10, int(os.environ.get("SSH_TIMEOUT_SECONDS", "90"))),
            scp_timeout_seconds=max(30, int(os.environ.get("SCP_TIMEOUT_SECONDS", "900"))),
        )


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    config = Config.from_env()
    _validate_config(config)
    while not STOP.is_set():
        recover_expired_leases(config)
        task_id = claim_next_task(config)
        if task_id is None:
            if config.run_once:
                return 0
            STOP.wait(config.poll_seconds)
            continue
        try:
            process_claimed_task(config, task_id)
        except Exception as exc:
            LOG.exception("research task failed task_id=%s error=%s", task_id, type(exc).__name__)
            _handle_failure(config, task_id, exc)
            if config.run_once:
                return 1
        if config.run_once:
            return 0
    return 0


def claim_next_task(config: Config) -> str | None:
    root = shlex.quote(config.cloud_queue_root)
    allow_factor_factory = "1" if config.factor_factory_enabled else "0"
    script = (
        "set -eu; "
        f"root={root}; "
        f"allow_factor_factory={allow_factor_factory}; "
        'task=""; '
        'for candidate in $(find "$root/pending" -mindepth 1 -maxdepth 1 -type d '
        "-printf '%f\\n' 2>/dev/null | LC_ALL=C sort); do "
        "case \"$candidate\" in *[!A-Za-z0-9_.:-]*|'') continue;; esac; "
        'if [ "$allow_factor_factory" != "1" ] && '
        'grep -Eq \'"task_type"[[:space:]]*:[[:space:]]*"factor_factory"\' '
        '"$root/pending/$candidate/task.json"; then continue; fi; '
        'task="$candidate"; break; done; '
        '[ -n "$task" ] || exit 44; '
        "case \"$task\" in *[!A-Za-z0-9_.:-]*|'') exit 45;; esac; "
        'mv "$root/pending/$task" "$root/running/$task"; '
        'date -u +%s > "$root/running/$task/.lease_claim_epoch"; '
        f"printf '%s' {shlex.quote(config.worker_id)} > \"$root/running/$task/.lease_worker\"; "
        "printf '%s' \"$task\""
    )
    result = _ssh(config, script, check=False)
    if result.returncode == 44:
        return None
    if result.returncode != 0:
        raise RuntimeError(f"research_task_claim_failed:{_tail(result.stderr)}")
    task_id = result.stdout.strip()
    _require_identifier(task_id)
    return task_id


def recover_expired_leases(config: Config, *, now: datetime | None = None) -> int:
    """Recover abandoned running tasks without racing a live heartbeat."""
    current_time = now or datetime.now(UTC)
    root = shlex.quote(config.cloud_queue_root)
    result = _ssh(
        config,
        f"find {root}/running -mindepth 1 -maxdepth 1 -type d "
        "-printf '%f\\n' 2>/dev/null | LC_ALL=C sort",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"research_running_scan_failed:{_tail(result.stderr)}")
    recovered = 0
    for task_id in filter(None, (line.strip() for line in result.stdout.splitlines())):
        _require_identifier(task_id)
        work = config.data_root / "work" / task_id
        work.mkdir(parents=True, exist_ok=True)
        status = _read_local_or_remote_status(config, task_id, work)
        if status is None:
            LOG.warning("cannot recover task without valid status task_id=%s", task_id)
            continue
        if _recover_handoff_visibility(config, task_id):
            continue
        if status.state in {
            ResearchTaskState.PUBLISHING,
            ResearchTaskState.COMPLETED,
            ResearchTaskState.REJECTED,
        }:
            continue
        if not _lease_is_expired(config, task_id, status, current_time):
            continue
        _discard_incomplete_result_partials(config, task_id)
        status_path = work / "status.previous.json"
        expected_status_sha = hashlib.sha256(status_path.read_bytes()).hexdigest()
        retry = status.attempt < status.max_attempts
        destination_state = "pending" if retry else "failed"
        transition = _conditional_remote_transition(
            config,
            task_id=task_id,
            expected_status_sha=expected_status_sha,
            destination_state=destination_state,
        )
        if not transition:
            continue
        next_status = status.model_copy(
            update={
                "state": ResearchTaskState.PENDING if retry else ResearchTaskState.FAILED,
                "worker_id": None if retry else status.worker_id,
                "heartbeat_at": current_time,
                "completed_at": None if retry else current_time,
                "lease_expires_at": None,
                "last_error": "LEASE_EXPIRED",
                "import_status": "retry_pending" if retry else "worker_failed_lease_expired",
            }
        )
        _upload_status(config, next_status, work)
        recovered += 1
        LOG.warning(
            "recovered expired research lease task_id=%s destination=%s attempt=%s",
            task_id,
            destination_state,
            status.attempt,
        )
    return recovered


def _lease_is_expired(
    config: Config,
    task_id: str,
    status: ResearchTaskStatus,
    now: datetime,
) -> bool:
    lease = _read_local_or_remote_lease(config, task_id, config.data_root / "work" / task_id)
    if lease is not None:
        return lease.lease_expires_at <= now
    if status.lease_expires_at is not None:
        return status.lease_expires_at <= now
    claim_epoch = _read_remote_claim_epoch(config, task_id)
    if claim_epoch is None:
        return False
    grace_seconds = max(120, config.heartbeat_seconds * 3)
    return claim_epoch + grace_seconds <= int(now.timestamp())


def _read_remote_claim_epoch(config: Config, task_id: str) -> int | None:
    _require_identifier(task_id)
    path = f"{config.cloud_queue_root}/running/{task_id}/.lease_claim_epoch"
    result = _ssh(config, f"cat {shlex.quote(path)}", check=False)
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _remote_exists(config: Config, path: str) -> bool:
    return _ssh(config, f"test -e {shlex.quote(path)}", check=False).returncode == 0


def _conditional_remote_transition(
    config: Config,
    *,
    task_id: str,
    expected_status_sha: str,
    destination_state: str,
) -> bool:
    _require_identifier(task_id)
    if destination_state not in {"pending", "failed"}:
        raise ValueError("invalid lease recovery destination")
    root = config.cloud_queue_root
    source = f"{root}/running/{task_id}"
    destination = f"{root}/{destination_state}/{task_id}"
    status_path = f"{root}/status/{task_id}.json"
    script = (
        "set -eu; "
        f"test -d {shlex.quote(source)} || exit 47; "
        f"test ! -e {shlex.quote(destination)} || exit 48; "
        f"test ! -e {shlex.quote(root + '/results/inbox/' + task_id)} || exit 49; "
        f"actual=$(sha256sum {shlex.quote(status_path)} | cut -d' ' -f1); "
        f'test "$actual" = {shlex.quote(expected_status_sha)} || exit 46; '
        f"mv {shlex.quote(source)} {shlex.quote(destination)}; "
        f"rm -f -- {shlex.quote(root + '/lease/' + task_id + '.json')}"
    )
    result = _ssh(config, script, check=False)
    if result.returncode in {46, 47, 48, 49}:
        return False
    if result.returncode != 0:
        raise RuntimeError(f"research_lease_recovery_failed:{_tail(result.stderr)}")
    return True


def process_claimed_task(config: Config, task_id: str) -> None:
    work = config.data_root / "work" / task_id
    work.mkdir(parents=True, exist_ok=True)
    task_path = work / "task.json"
    _scp_from(config, _remote_task_path(config, "running", task_id, "task.json"), task_path)
    task = RESEARCH_TASK_ADAPTER.validate_json(task_path.read_text("utf-8"))
    task_public_key = load_public_key(config.task_public_key_path)
    if task.task_id != task_id:
        raise ValueError("research_task_id_mismatch")
    if isinstance(task, FactorFactoryTask) and not config.factor_factory_enabled:
        raise RuntimeError("factor_factory_worker_disabled")
    if task.quant_lab_commit != config.worker_commit:
        raise ValueError("worker_code_mismatch")
    snapshot_dir = work / "snapshot-control"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_dir / "manifest.json"
    _scp_from(
        config,
        f"{config.cloud_queue_root}/snapshots/{task.snapshot_id}/manifest.json",
        manifest_path,
    )
    snapshot = RESEARCH_SNAPSHOT_ADAPTER.validate_json(manifest_path.read_text("utf-8"))
    validate_research_task_snapshot(
        task,
        snapshot,
        task_public_key=task_public_key,
        expected_key_id=config.task_key_id,
        expected_quant_lab_commit=config.worker_commit,
    )

    claimed_at = datetime.now(UTC)
    start_date, end_date, mode, cost_mode = _task_status_dimensions(task)
    status = ResearchTaskStatus(
        task_id=task.task_id,
        snapshot_id=task.snapshot_id,
        task_type=task.task_type,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        cost_mode=cost_mode,
        state=ResearchTaskState.SYNCING,
        worker_id=config.worker_id,
        requested_at=task.requested_at,
        claimed_at=claimed_at,
        heartbeat_at=claimed_at,
        lease_expires_at=claimed_at + timedelta(seconds=task.lease_seconds),
        attempt=_current_attempt(config, task_id) + 1,
        max_attempts=task.max_attempts,
        input_bytes=snapshot.total_input_bytes,
        import_status="waiting_for_nas_result",
    )
    _upload_status(config, status, work)
    lease = ResearchTaskLease(
        task_id=task.task_id,
        snapshot_id=task.snapshot_id,
        task_type=task.task_type,
        worker_id=config.worker_id,
        claimed_at=claimed_at,
        heartbeat_at=claimed_at,
        lease_expires_at=claimed_at + timedelta(seconds=task.lease_seconds),
    )
    _upload_lease(config, lease, work)
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(config, task, lease, work, heartbeat_stop),
        daemon=True,
    )
    heartbeat.start()
    try:
        if isinstance(snapshot, FactorFactorySnapshotManifest):
            if snapshot.total_input_bytes > config.factor_factory_max_snapshot_bytes:
                raise ValueError("factor_factory_snapshot_input_size_limit_exceeded")
            if (
                snapshot.estimated_uncompressed_bytes
                > config.factor_factory_max_input_uncompressed_bytes
            ):
                raise ValueError("factor_factory_input_uncompressed_size_limit_exceeded")
        sync_result = sync_snapshot_blobs(
            snapshot,
            data_root=config.data_root,
            fetch_blob=lambda relative, destination: _scp_from(
                config,
                f"{config.cloud_queue_root}/snapshots/{snapshot.snapshot_id}/files/{relative}",
                destination,
            ),
            fetch_blobs=lambda references, destination: _fetch_snapshot_batch(
                config, snapshot.snapshot_id, references, destination
            ),
            batch_fetch_workers=config.batch_fetch_workers,
            min_free_disk_bytes=config.min_free_disk_bytes,
            max_snapshot_bytes=config.max_snapshot_bytes,
        )
        status = status.model_copy(
            update={
                "state": ResearchTaskState.COMPUTING,
                "heartbeat_at": datetime.now(UTC),
                "downloaded_bytes": sync_result.downloaded_bytes,
                "cache_hit_bytes": sum(item.size_bytes for item in snapshot.files)
                - sync_result.downloaded_bytes,
            }
        )
        _upload_status(config, status, work)
        started = time.perf_counter()
        with _heavy_job_lock(config.heavy_job_lock):
            if isinstance(task, FactorFactoryTask):
                if not isinstance(snapshot, FactorFactorySnapshotManifest):
                    raise ValueError("research_task_snapshot_type_mismatch")
                compute = compute_factor_factory_result(
                    sync_result.snapshot_root,
                    snapshot,
                    task,
                    stage_callback=lambda stage: _upload_factor_factory_stage(
                        config,
                        status,
                        work,
                        stage,
                    ),
                    max_input_uncompressed_bytes=(
                        config.factor_factory_max_input_uncompressed_bytes
                    ),
                )
            elif isinstance(task, AlphaFactoryTask):
                if not isinstance(snapshot, AlphaFactorySnapshotManifest):
                    raise ValueError("research_task_snapshot_type_mismatch")
                compute = compute_alpha_factory_from_snapshot(
                    sync_result.snapshot_root,
                    snapshot,
                    task,
                )
            elif isinstance(task, FactorResearchTask):
                if not isinstance(snapshot, FactorResearchSnapshotManifest):
                    raise ValueError("research_task_snapshot_type_mismatch")
                compute = compute_factor_research_result(
                    sync_result.snapshot_root,
                    snapshot,
                    task,
                )
            else:
                compute = compute_entry_quality_history_from_snapshot(
                    sync_result.snapshot_root,
                    snapshot,
                    task,
                )
            status = status.model_copy(
                update={
                    "state": ResearchTaskState.VALIDATING_ON_NAS,
                    "heartbeat_at": datetime.now(UTC),
                }
            )
            _upload_status(config, status, work)
            peak_rss = _peak_rss_bytes()
            signing_key = load_signing_key(config.worker_signing_key_path)
            if isinstance(task, FactorFactoryTask):
                if not isinstance(snapshot, FactorFactorySnapshotManifest):
                    raise ValueError("research_task_snapshot_type_mismatch")
                result_root, manifest, receipt = write_factor_factory_result_bundle(
                    config.data_root / "results",
                    task=task,
                    snapshot=snapshot,
                    compute=compute,
                    worker_id=config.worker_id,
                    worker_commit=config.worker_commit,
                    worker_key_id=config.worker_key_id,
                    worker_signing_key=signing_key,
                    claimed_at=claimed_at,
                    input_bytes=snapshot.total_input_bytes,
                    cache_hit_bytes=status.cache_hit_bytes,
                    downloaded_bytes=status.downloaded_bytes,
                    peak_rss_bytes=peak_rss,
                    compute_duration_seconds=time.perf_counter() - started,
                    max_result_bytes=config.factor_factory_max_result_bytes,
                    max_value_partition_bytes=(config.factor_factory_max_value_partition_bytes),
                    max_file_count=config.factor_factory_max_file_count,
                    max_uncompressed_bytes=config.factor_factory_max_uncompressed_bytes,
                )
            elif isinstance(task, AlphaFactoryTask):
                if not isinstance(snapshot, AlphaFactorySnapshotManifest):
                    raise ValueError("research_task_snapshot_type_mismatch")
                result_root, manifest, receipt = write_alpha_factory_result_bundle(
                    config.data_root / "results",
                    task=task,
                    snapshot=snapshot,
                    compute=compute,
                    worker_id=config.worker_id,
                    worker_commit=config.worker_commit,
                    worker_key_id=config.worker_key_id,
                    worker_signing_key=signing_key,
                    claimed_at=claimed_at,
                    input_bytes=snapshot.total_input_bytes,
                    cache_hit_bytes=status.cache_hit_bytes,
                    downloaded_bytes=status.downloaded_bytes,
                    peak_rss_bytes=peak_rss,
                    compute_duration_seconds=time.perf_counter() - started,
                    max_result_bytes=config.max_result_bytes,
                )
            elif isinstance(task, FactorResearchTask):
                if not isinstance(snapshot, FactorResearchSnapshotManifest):
                    raise ValueError("research_task_snapshot_type_mismatch")
                result_root, manifest, receipt = write_factor_research_result_bundle(
                    config.data_root / "results",
                    task=task,
                    snapshot=snapshot,
                    compute=compute,
                    worker_id=config.worker_id,
                    worker_commit=config.worker_commit,
                    worker_key_id=config.worker_key_id,
                    worker_signing_key=signing_key,
                    claimed_at=claimed_at,
                    input_bytes=snapshot.total_input_bytes,
                    cache_hit_bytes=status.cache_hit_bytes,
                    downloaded_bytes=status.downloaded_bytes,
                    peak_rss_bytes=peak_rss,
                    compute_duration_seconds=time.perf_counter() - started,
                    max_result_bytes=config.max_result_bytes,
                )
            else:
                result_root, manifest, receipt = write_entry_quality_history_result_bundle(
                    config.data_root / "results",
                    task=task,
                    snapshot=snapshot,
                    artifacts=compute.artifacts,
                    worker_id=config.worker_id,
                    worker_commit=config.worker_commit,
                    worker_key_id=config.worker_key_id,
                    worker_signing_key=signing_key,
                    claimed_at=claimed_at,
                    input_bytes=snapshot.total_input_bytes,
                    cache_hit_bytes=status.cache_hit_bytes,
                    downloaded_bytes=status.downloaded_bytes,
                    peak_rss_bytes=peak_rss,
                    compute_duration_seconds=time.perf_counter() - started,
                    max_result_bytes=config.max_result_bytes,
                )
        if manifest.output_bytes > _max_result_bytes_for_task(config, task):
            raise RuntimeError("research_result_size_limit_exceeded")
        status = status.model_copy(
            update={
                "state": ResearchTaskState.UPLOADING,
                "heartbeat_at": datetime.now(UTC),
                "output_rows": receipt.output_rows,
                "output_bytes": manifest.output_bytes,
                "peak_rss_bytes": manifest.peak_rss_bytes,
                "compute_duration_seconds": manifest.compute_duration_seconds,
                "anti_leakage_status": receipt.anti_leakage_status,
            }
        )
        _upload_status(config, status, work)
        status = _handoff_result_to_cloud(
            config,
            task,
            status,
            work,
            result_root,
            heartbeat_stop,
            heartbeat,
        )
    finally:
        if heartbeat.is_alive():
            _stop_heartbeat(config, heartbeat_stop, heartbeat)


def _task_status_dimensions(task: ResearchTaskEnvelope) -> tuple[Any, Any, str, str]:
    if isinstance(task, FactorFactoryTask):
        return (
            task.as_of_date,
            task.as_of_date,
            "PARITY_FULL/bootstrap_full",
            f"point_in_task_{task.cost_quantile}",
        )
    if isinstance(task, AlphaFactoryTask):
        return (
            task.as_of_date,
            task.as_of_date,
            "alpha_factory",
            "research",
        )
    if isinstance(task, FactorResearchTask):
        return (
            task.start_date,
            task.end_date,
            "factor_research",
            "research",
        )
    return (
        task.parameters.start_date,
        task.parameters.end_date,
        task.parameters.mode,
        task.parameters.cost_mode,
    )


def _upload_factor_factory_stage(
    config: Config,
    status: ResearchTaskStatus,
    work: Path,
    stage: str,
) -> None:
    allowed = {
        ResearchTaskState.COMPUTING_VALUES.value,
        ResearchTaskState.COMPUTING_LABELS.value,
        ResearchTaskState.COMPUTING_EVIDENCE.value,
        ResearchTaskState.COMPUTING_CORRELATION.value,
    }
    if stage not in allowed:
        raise ValueError(f"unknown_factor_factory_stage:{stage}")
    _upload_status(
        config,
        status.model_copy(
            update={
                "state": ResearchTaskState(stage),
                "heartbeat_at": datetime.now(UTC),
            }
        ),
        work,
    )


def _max_result_bytes_for_task(config: Config, task: ResearchTaskEnvelope) -> int:
    if isinstance(task, FactorFactoryTask):
        return config.factor_factory_max_result_bytes
    return config.max_result_bytes


def _heartbeat_loop(
    config: Config,
    task: ResearchTaskEnvelope,
    initial: ResearchTaskLease,
    work: Path,
    stop: threading.Event,
) -> None:
    lease = initial
    while not stop.wait(config.heartbeat_seconds):
        try:
            now = datetime.now(UTC)
            lease = lease.model_copy(
                update={
                    "heartbeat_at": now,
                    "lease_expires_at": now + timedelta(seconds=task.lease_seconds),
                    "sequence": lease.sequence + 1,
                }
            )
            _upload_lease(config, lease, work)
        except Exception as exc:
            LOG.warning(
                "heartbeat upload failed task_id=%s error=%s",
                task.task_id,
                type(exc).__name__,
            )


def _fetch_snapshot_batch(
    config: Config,
    snapshot_id: str,
    references: list[Any],
    destination: Path,
) -> None:
    for reference in references:
        target = destination / reference.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        _scp_from(
            config,
            f"{config.cloud_queue_root}/snapshots/{snapshot_id}/files/{reference.relative_path}",
            target,
        )


def _upload_result_partial(config: Config, task_id: str, result_root: Path) -> str:
    remote_partial = (
        f"{config.cloud_queue_root}/results/inbox/.{task_id}.{uuid.uuid4().hex}.partial"
    )
    try:
        _scp_to(config, result_root, remote_partial, recursive=True)
        quoted_partial = shlex.quote(remote_partial)
        _ssh(
            config,
            f"find {quoted_partial} -type d -exec chmod 2770 {{}} + && "
            f"find {quoted_partial} -type f -exec chmod 0660 {{}} +",
        )
    except Exception:
        with contextlib.suppress(Exception):
            _ssh(config, f"rm -rf -- {shlex.quote(remote_partial)}", check=False)
        raise
    return remote_partial


def _mark_result_partial_ready(config: Config, remote_partial: str) -> None:
    marker = f"{remote_partial}/{_HANDOFF_READY_MARKER}"
    quoted_marker = shlex.quote(marker)
    _ssh(config, f"touch {quoted_marker} && chmod 0660 {quoted_marker}")


def _handoff_result_to_cloud(
    config: Config,
    task: ResearchTaskEnvelope,
    status: ResearchTaskStatus,
    _work: Path,
    result_root: Path,
    heartbeat_stop: threading.Event,
    heartbeat: threading.Thread,
) -> ResearchTaskStatus:
    remote_partial = _upload_result_partial(config, task.task_id, result_root)
    ready = False
    exposed = False
    try:
        _mark_result_partial_ready(config, remote_partial)
        ready = True
        _stop_heartbeat(config, heartbeat_stop, heartbeat)
        _finalize_result_upload(config, task.task_id, remote_partial)
        exposed = True
        completed_at = datetime.now(UTC)
        cloud_status = status.model_copy(
            update={
                "state": ResearchTaskState.VALIDATING_ON_CLOUD,
                "heartbeat_at": completed_at,
                "lease_expires_at": completed_at + timedelta(seconds=task.lease_seconds),
                "import_status": "result_uploaded",
            }
        )
        return cloud_status
    finally:
        if not ready and not exposed:
            with contextlib.suppress(Exception):
                _ssh(config, f"rm -rf -- {shlex.quote(remote_partial)}", check=False)


def _finalize_result_upload(config: Config, task_id: str, remote_partial: str) -> None:
    remote_final = f"{config.cloud_queue_root}/results/inbox/{task_id}"
    _ssh(
        config,
        f"test ! -e {shlex.quote(remote_final)} && mv {shlex.quote(remote_partial)} "
        f"{shlex.quote(remote_final)} || rm -rf {shlex.quote(remote_partial)}",
    )


def _recover_handoff_visibility(config: Config, task_id: str) -> bool:
    """Expose a complete hidden result or report that cloud already owns it."""
    inbox = f"{config.cloud_queue_root}/results/inbox/{task_id}"
    if _remote_exists(config, inbox):
        return True
    for remote_partial in _list_result_partials(config, task_id):
        marker = f"{remote_partial}/{_HANDOFF_READY_MARKER}"
        if not _remote_exists(config, marker):
            continue
        _finalize_result_upload(config, task_id, remote_partial)
        if _remote_exists(config, inbox):
            LOG.warning(
                "recovered complete hidden research result task_id=%s",
                task_id,
            )
            return True
    return False


def _list_result_partials(config: Config, task_id: str) -> list[str]:
    _require_identifier(task_id)
    inbox_root = f"{config.cloud_queue_root}/results/inbox"
    prefix = f".{task_id}."
    suffix = ".partial"
    command = (
        f"find {shlex.quote(inbox_root)} -mindepth 1 -maxdepth 1 -type d "
        f"-name {shlex.quote(prefix + '*' + suffix)} -print 2>/dev/null | LC_ALL=C sort"
    )
    result = _ssh(config, command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"research_partial_scan_failed:{_tail(result.stderr)}")
    partials: list[str] = []
    for raw_path in result.stdout.splitlines():
        path = raw_path.strip()
        name = Path(path).name
        if not path or not name.startswith(prefix) or not name.endswith(suffix):
            continue
        partials.append(path)
    return partials


def _discard_incomplete_result_partials(config: Config, task_id: str) -> None:
    for remote_partial in _list_result_partials(config, task_id):
        marker = f"{remote_partial}/{_HANDOFF_READY_MARKER}"
        if _remote_exists(config, marker):
            continue
        _ssh(config, f"rm -rf -- {shlex.quote(remote_partial)}", check=False)


def _stop_heartbeat(
    config: Config,
    stop: threading.Event,
    heartbeat: threading.Thread,
) -> None:
    stop.set()
    heartbeat.join(timeout=max(config.ssh_timeout_seconds, config.scp_timeout_seconds) + 5)
    if heartbeat.is_alive():
        raise RuntimeError("research_heartbeat_did_not_stop")


def _handle_failure(config: Config, task_id: str, exc: Exception) -> None:
    work = config.data_root / "work" / task_id
    work.mkdir(parents=True, exist_ok=True)
    try:
        result_uploaded = _recover_handoff_visibility(config, task_id)
    except Exception as check_exc:
        LOG.warning(
            "cannot verify result ownership; leave running task untouched task_id=%s error=%s",
            task_id,
            type(check_exc).__name__,
        )
        return
    if result_uploaded:
        LOG.warning(
            "research result already uploaded; cloud owns task state task_id=%s",
            task_id,
        )
        return
    current = _read_local_or_remote_status(config, task_id, work)
    attempt = current.attempt if current is not None else 0
    if current is None or current.state == ResearchTaskState.PENDING:
        attempt += 1
    max_attempts = current.max_attempts if current is not None else 3
    rejected = isinstance(exc, ValueError) and str(exc) in _NON_RETRYABLE_TASK_ERRORS
    retry = not rejected and attempt < max_attempts
    now = datetime.now(UTC)
    if current is not None:
        status = current.model_copy(
            update={
                "state": (
                    ResearchTaskState.REJECTED
                    if rejected
                    else ResearchTaskState.PENDING
                    if retry
                    else ResearchTaskState.FAILED
                ),
                "worker_id": current.worker_id or config.worker_id,
                "claimed_at": current.claimed_at or now,
                "heartbeat_at": now,
                "completed_at": None if retry else now,
                "lease_expires_at": None,
                "attempt": attempt,
                "last_error": f"{type(exc).__name__}:{str(exc)[:800]}",
                "import_status": (
                    "worker_rejected_code_mismatch"
                    if rejected
                    else "retry_pending"
                    if retry
                    else "worker_failed"
                ),
            }
        )
        with contextlib.suppress(Exception):
            _upload_status(config, status, work)
    source = f"{config.cloud_queue_root}/running/{task_id}"
    destination_state = "pending" if retry else "failed"
    destination = f"{config.cloud_queue_root}/{destination_state}/{task_id}"
    with contextlib.suppress(Exception):
        _ssh(
            config,
            f"test -d {shlex.quote(source)} && test ! -e {shlex.quote(destination)} && "
            f"mv {shlex.quote(source)} {shlex.quote(destination)} || true",
        )


def _current_attempt(config: Config, task_id: str) -> int:
    work = config.data_root / "work" / task_id
    status = _read_local_or_remote_status(config, task_id, work)
    return status.attempt if status is not None else 0


def _read_local_or_remote_status(
    config: Config,
    task_id: str,
    work: Path,
) -> ResearchTaskStatus | None:
    path = work / "status.previous.json"
    try:
        path.unlink(missing_ok=True)
        _scp_from(
            config,
            f"{config.cloud_queue_root}/status/{task_id}.json",
            path,
            check=False,
        )
        return ResearchTaskStatus.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError, RuntimeError):
        return None


def _read_local_or_remote_lease(
    config: Config,
    task_id: str,
    work: Path,
) -> ResearchTaskLease | None:
    path = work / "lease.previous.json"
    try:
        path.unlink(missing_ok=True)
        _scp_from(
            config,
            f"{config.cloud_queue_root}/lease/{task_id}.json",
            path,
            check=False,
        )
        return ResearchTaskLease.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError, RuntimeError):
        return None


def _upload_status(config: Config, status: ResearchTaskStatus, work: Path) -> None:
    token = uuid.uuid4().hex
    with _STATUS_UPLOAD_LOCK:
        local = work / "status.upload.json"
        local_tmp = work / f".{local.name}.{token}.tmp"
        local_tmp.write_text(status.model_dump_json(indent=2), encoding="utf-8")
        os.replace(local_tmp, local)
        upload = work / f".status.upload.{token}.json"
        upload.write_bytes(local.read_bytes())
    remote_tmp = f"{config.cloud_queue_root}/status/.{status.task_id}.{token}.tmp"
    remote_final = f"{config.cloud_queue_root}/status/{status.task_id}.json"
    inbox = f"{config.cloud_queue_root}/results/inbox/{status.task_id}"
    try:
        _scp_to(config, upload, remote_tmp)
        result = _ssh(
            config,
            f"if test -e {shlex.quote(inbox)}; then rm -f -- {shlex.quote(remote_tmp)}; "
            f"exit 52; fi; mv {shlex.quote(remote_tmp)} {shlex.quote(remote_final)}",
            check=False,
        )
        if result.returncode == 52:
            raise RuntimeError("research_status_owned_by_cloud")
        if result.returncode != 0:
            raise RuntimeError(f"research_status_publish_failed:{_tail(result.stderr)}")
    except Exception:
        with contextlib.suppress(Exception):
            _ssh(config, f"rm -f -- {shlex.quote(remote_tmp)}", check=False)
        raise
    finally:
        upload.unlink(missing_ok=True)


def _upload_lease(config: Config, lease: ResearchTaskLease, work: Path) -> None:
    token = uuid.uuid4().hex
    with _LEASE_UPLOAD_LOCK:
        local = work / "lease.upload.json"
        local_tmp = work / f".{local.name}.{token}.tmp"
        local_tmp.write_text(lease.model_dump_json(indent=2), encoding="utf-8")
        os.replace(local_tmp, local)
        upload = work / f".lease.upload.{token}.json"
        upload.write_bytes(local.read_bytes())
    remote_tmp = f"{config.cloud_queue_root}/lease/.{lease.task_id}.{token}.tmp"
    remote_final = f"{config.cloud_queue_root}/lease/{lease.task_id}.json"
    inbox = f"{config.cloud_queue_root}/results/inbox/{lease.task_id}"
    try:
        _scp_to(config, upload, remote_tmp)
        result = _ssh(
            config,
            f"if test -e {shlex.quote(inbox)}; then rm -f -- {shlex.quote(remote_tmp)}; "
            f"exit 52; fi; mv {shlex.quote(remote_tmp)} {shlex.quote(remote_final)}",
            check=False,
        )
        if result.returncode == 52:
            raise RuntimeError("research_lease_owned_by_cloud")
        if result.returncode != 0:
            raise RuntimeError(f"research_lease_publish_failed:{_tail(result.stderr)}")
    except Exception:
        with contextlib.suppress(Exception):
            _ssh(config, f"rm -f -- {shlex.quote(remote_tmp)}", check=False)
        raise
    finally:
        upload.unlink(missing_ok=True)


@contextlib.contextmanager
def _heavy_job_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(Exception):
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _validate_config(config: Config) -> None:
    if len(config.worker_commit) != 40 or any(
        character not in "0123456789abcdef" for character in config.worker_commit
    ):
        raise ValueError("QUANT_RESEARCH_WORKER_COMMIT must be a full git SHA")
    for path in (
        config.ssh_key_path,
        config.known_hosts_path,
        config.task_public_key_path,
        config.worker_signing_key_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.factor_factory_max_result_bytes <= 0:
        raise ValueError("FACTOR_FACTORY_MAX_RESULT_BYTES must be positive")
    if config.factor_factory_max_value_partition_bytes <= 0:
        raise ValueError("FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES must be positive")
    if config.factor_factory_max_file_count <= 0:
        raise ValueError("FACTOR_FACTORY_MAX_FILE_COUNT must be positive")
    if config.factor_factory_max_snapshot_bytes <= 0:
        raise ValueError("FACTOR_FACTORY_MAX_SNAPSHOT_BYTES must be positive")
    if config.factor_factory_max_uncompressed_bytes <= 0:
        raise ValueError("FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES must be positive")
    if config.factor_factory_max_input_uncompressed_bytes <= 0:
        raise ValueError("FACTOR_FACTORY_MAX_INPUT_UNCOMPRESSED_BYTES must be positive")
    config.data_root.mkdir(parents=True, exist_ok=True)


def _remote_task_path(config: Config, state: str, task_id: str, name: str) -> str:
    return f"{config.cloud_queue_root}/{state}/{task_id}/{name}"


def _ssh(config: Config, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [
                "ssh",
                "-p",
                str(config.cloud_port),
                "-i",
                str(config.ssh_key_path),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={config.known_hosts_path}",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                f"{config.cloud_user}@{config.cloud_host}",
                command,
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=config.ssh_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ssh_timeout") from exc
    if check and result.returncode != 0:
        raise RuntimeError(f"ssh_failed:{_tail(result.stderr)}")
    return result


def _scp_from(
    config: Config,
    remote_path: str,
    local_path: Path,
    *,
    check: bool = True,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "scp",
                "-P",
                str(config.cloud_port),
                "-i",
                str(config.ssh_key_path),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={config.known_hosts_path}",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                f"{config.cloud_user}@{config.cloud_host}:{remote_path}",
                str(local_path),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=config.scp_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("scp_from_timeout") from exc
    if result.returncode != 0 and check:
        raise RuntimeError(f"scp_from_failed:{_tail(result.stderr)}")


def _scp_to(
    config: Config,
    local_path: Path,
    remote_path: str,
    *,
    recursive: bool = False,
) -> None:
    command = ["scp"]
    if recursive:
        command.append("-r")
    command.extend(
        [
            "-P",
            str(config.cloud_port),
            "-i",
            str(config.ssh_key_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={config.known_hosts_path}",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            str(local_path),
            f"{config.cloud_user}@{config.cloud_host}:{remote_path}",
        ]
    )
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=config.scp_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("scp_to_timeout") from exc
    if result.returncode != 0:
        raise RuntimeError(f"scp_to_failed:{_tail(result.stderr)}")


def _peak_rss_bytes() -> int:
    if resource is None:
        return 0
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > 10_000_000 else value * 1024


def _request_stop(_signum: int, _frame: Any) -> None:
    STOP.set()


def _required(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require_identifier(value: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not value or any(character not in allowed for character in value):
        raise ValueError("unsafe research task id")


def _tail(value: str, limit: int = 1200) -> str:
    return str(value or "")[-limit:].replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
