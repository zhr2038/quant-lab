from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from quant_lab.export_plane.status import atomic_write_json
from quant_lab.research_plane.contracts import (
    RESEARCH_SNAPSHOT_ADAPTER,
    RESEARCH_TASK_ADAPTER,
)
from quant_lab.research_plane.factor_factory_snapshot import (
    cleanup_stale_factor_factory_rehydrate_partials,
)
from quant_lab.research_plane.snapshot_lock import snapshot_payload_lock
from quant_lab.research_plane.status import ensure_research_queue_layout

DEFAULT_SNAPSHOT_RETENTION_DAYS = 7
DEFAULT_SNAPSHOT_PAYLOAD_CAP_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class SnapshotGcResult:
    scanned_snapshot_count: int
    active_snapshot_count: int
    released_snapshot_count: int
    released_snapshot_ids: tuple[str, ...]
    bytes_before: int
    bytes_released: int
    bytes_after: int
    dry_run: bool


def gc_research_snapshot_payloads(
    queue_root: str | Path,
    *,
    retention_days: int = DEFAULT_SNAPSHOT_RETENTION_DAYS,
    max_payload_bytes: int = DEFAULT_SNAPSHOT_PAYLOAD_CAP_BYTES,
    dry_run: bool = False,
    now: datetime | None = None,
) -> SnapshotGcResult:
    if retention_days < 0:
        raise ValueError("snapshot retention_days must be non-negative")
    if max_payload_bytes < 0:
        raise ValueError("snapshot max_payload_bytes must be non-negative")
    queue = ensure_research_queue_layout(queue_root)
    observed_at = now or datetime.now(UTC)
    cleanup_stale_factor_factory_rehydrate_partials(queue, now=observed_at)
    active = _active_snapshot_ids(queue)
    snapshots = _snapshot_records(queue)
    bytes_before = sum(record[2] for record in snapshots)
    bytes_remaining = bytes_before
    released: list[str] = []
    bytes_released = 0
    cutoff = observed_at - timedelta(days=retention_days)

    completed = _completed_snapshot_ids(queue)
    release_candidates: list[tuple[datetime, str, int, str]] = []
    for generated_at, snapshot_id, payload_bytes in snapshots:
        if snapshot_id in active or payload_bytes <= 0:
            continue
        if snapshot_id in completed:
            release_candidates.append(
                (generated_at, snapshot_id, payload_bytes, "completed_import")
            )
        elif generated_at <= cutoff:
            release_candidates.append(
                (generated_at, snapshot_id, payload_bytes, "retention_expired")
            )

    selected = {snapshot_id for _, snapshot_id, _, _ in release_candidates}
    bytes_remaining -= sum(payload_bytes for _, _, payload_bytes, _ in release_candidates)
    if bytes_remaining > max_payload_bytes:
        for generated_at, snapshot_id, payload_bytes in snapshots:
            if snapshot_id in active or payload_bytes <= 0 or snapshot_id in selected:
                continue
            release_candidates.append((generated_at, snapshot_id, payload_bytes, "capacity_limit"))
            selected.add(snapshot_id)
            bytes_remaining -= payload_bytes
            if bytes_remaining <= max_payload_bytes:
                break

    for _, snapshot_id, payload_bytes, reason in sorted(release_candidates):
        if not dry_run:
            if not release_snapshot_payload(queue, snapshot_id, reason=reason, now=observed_at):
                continue
        released.append(snapshot_id)
        bytes_released += payload_bytes

    result = SnapshotGcResult(
        scanned_snapshot_count=len(snapshots),
        active_snapshot_count=len(active),
        released_snapshot_count=len(released),
        released_snapshot_ids=tuple(released),
        bytes_before=bytes_before,
        bytes_released=bytes_released,
        bytes_after=max(bytes_before - bytes_released, 0),
        dry_run=dry_run,
    )
    _append_gc_audit(queue, {"action": "gc_run", **asdict(result)}, observed_at)
    return result


def release_snapshot_payload(
    queue_root: str | Path,
    snapshot_id: str,
    *,
    reason: str,
    now: datetime | None = None,
) -> bool:
    queue = ensure_research_queue_layout(queue_root)
    try:
        with snapshot_payload_lock(queue, snapshot_id, timeout_seconds=0):
            return _release_snapshot_payload_locked(
                queue,
                snapshot_id,
                reason=reason,
                now=now,
            )
    except TimeoutError:
        return False


def _release_snapshot_payload_locked(
    queue: Path,
    snapshot_id: str,
    *,
    reason: str,
    now: datetime | None,
) -> bool:
    if snapshot_id in _active_snapshot_ids(queue):
        return False
    snapshot_root = queue / "snapshots" / snapshot_id
    manifest_path = snapshot_root / "manifest.json"
    seal_path = snapshot_root / "SEALED"
    files_root = snapshot_root / "files"
    if not manifest_path.is_file() or not seal_path.is_file() or not files_root.exists():
        return False
    manifest = RESEARCH_SNAPSHOT_ADAPTER.validate_json(manifest_path.read_text("utf-8"))
    observed_at = now or datetime.now(UTC)
    _make_tree_writable(snapshot_root)
    marker = {
        "schema_version": "quant_lab_research_snapshot_payload_release.v1",
        "snapshot_id": snapshot_id,
        "manifest_sha256": manifest.manifest_sha256,
        "released_at": observed_at.isoformat(),
        "released_bytes": manifest.total_input_bytes,
        "reason": reason,
        "state": "release_started",
        "manifest_preserved": True,
        "signature_preserved": True,
    }
    atomic_write_json(snapshot_root / "FILES_RELEASED.json", marker)
    shutil.rmtree(files_root)
    atomic_write_json(
        snapshot_root / "FILES_RELEASED.json",
        marker | {"state": "released"},
    )
    _make_tree_read_only(snapshot_root)
    _append_gc_audit(
        queue,
        {"action": "snapshot_payload_released", **marker, "state": "released"},
        observed_at,
    )
    return True


def _snapshot_records(queue: Path) -> list[tuple[datetime, str, int]]:
    records: list[tuple[datetime, str, int]] = []
    for snapshot_root in sorted((queue / "snapshots").iterdir()):
        if not snapshot_root.is_dir() or snapshot_root.name.startswith("."):
            continue
        manifest_path = snapshot_root / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = RESEARCH_SNAPSHOT_ADAPTER.validate_json(manifest_path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        payload_bytes = manifest.total_input_bytes if (snapshot_root / "files").exists() else 0
        records.append((manifest.generated_at, manifest.snapshot_id, payload_bytes))
    records.sort()
    return records


def _active_snapshot_ids(queue: Path) -> set[str]:
    snapshot_ids: set[str] = set()
    for state in ("pending", "running"):
        for task_path in (queue / state).glob("*/task.json"):
            try:
                snapshot_ids.add(
                    RESEARCH_TASK_ADAPTER.validate_json(task_path.read_text("utf-8")).snapshot_id
                )
            except (OSError, ValueError):
                continue
    for manifest_path in (queue / "results" / "inbox").glob("*/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        snapshot_id = str(payload.get("snapshot_id") or "")
        if snapshot_id:
            snapshot_ids.add(snapshot_id)
    for marker_path in (queue / "snapshots").glob(".rehydrate.*.partial/REHYDRATE.json"):
        try:
            payload = json.loads(marker_path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        snapshot_id = str(payload.get("snapshot_id") or "")
        if snapshot_id:
            snapshot_ids.add(snapshot_id)
    return snapshot_ids


def _completed_snapshot_ids(queue: Path) -> set[str]:
    snapshot_ids: set[str] = set()
    for task_path in (queue / "completed").glob("*/task.json"):
        try:
            snapshot_ids.add(
                RESEARCH_TASK_ADAPTER.validate_json(task_path.read_text("utf-8")).snapshot_id
            )
        except (OSError, ValueError):
            continue
    return snapshot_ids


def _append_gc_audit(queue: Path, payload: dict[str, Any], observed_at: datetime) -> None:
    audit_root = queue / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    path = audit_root / "snapshot_gc.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"observed_at": observed_at.isoformat(), **payload},
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n"
        )


def _make_tree_writable(path: Path) -> None:
    path.chmod(0o750)
    for candidate in path.rglob("*"):
        candidate.chmod(0o640 if candidate.is_file() else 0o750)


def _make_tree_read_only(path: Path) -> None:
    for candidate in sorted(path.rglob("*"), reverse=True):
        candidate.chmod(0o440 if candidate.is_file() else 0o550)
    path.chmod(0o550)
