from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from quant_lab.export_plane.cloud_index import load_cloud_index
from quant_lab.export_plane.contracts import ExportSnapshotManifest, ExportTask
from quant_lab.export_plane.status import atomic_write_json, ensure_queue_layout

DEFAULT_TERMINAL_GRACE_HOURS = 24
DEFAULT_TERMINAL_PAYLOAD_CAP_BYTES = 5 * 1024**3
_ACTIVE_STATES = ("pending", "running")
_TERMINAL_STATES = ("failed", "expired", "cancelled")


@dataclass(frozen=True)
class ExportSnapshotGcResult:
    scanned_snapshot_count: int
    active_snapshot_count: int
    terminal_snapshot_count: int
    released_snapshot_count: int
    released_invalid_snapshot_count: int
    released_snapshot_ids: tuple[str, ...]
    released_paths: tuple[str, ...]
    bytes_before: int
    bytes_released: int
    bytes_after: int
    dry_run: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _PayloadRecord:
    snapshot_id: str
    snapshot_root: Path
    files_root: Path
    manifest_sha256: str
    payload_bytes: int
    observed_at: datetime
    location: str


def gc_export_snapshot_payloads(
    queue_root: str | Path,
    *,
    terminal_grace_hours: int = DEFAULT_TERMINAL_GRACE_HOURS,
    max_terminal_payload_bytes: int = DEFAULT_TERMINAL_PAYLOAD_CAP_BYTES,
    dry_run: bool = True,
    now: datetime | None = None,
) -> ExportSnapshotGcResult:
    """Release regenerable Export Snapshot payloads without deleting audit identity."""
    if terminal_grace_hours < 0:
        raise ValueError("terminal_grace_hours must be non-negative")
    if max_terminal_payload_bytes < 0:
        raise ValueError("max_terminal_payload_bytes must be non-negative")

    queue = ensure_queue_layout(queue_root)
    observed_at = _as_utc(now or datetime.now(UTC))
    warnings: list[str] = []
    active = _active_snapshot_ids(queue, warnings)
    terminal = _terminal_snapshot_refs(queue, warnings)
    indexed = {item.snapshot_id for item in load_cloud_index(queue)}
    records = _payload_records(queue, warnings)
    bytes_before = sum(item.payload_bytes for item in records)
    cutoff = observed_at - timedelta(hours=terminal_grace_hours)

    selected: dict[Path, str] = {}
    terminal_records = [
        record
        for record in records
        if record.snapshot_id in terminal and record.snapshot_id not in active
    ]
    for record in terminal_records:
        if terminal[record.snapshot_id]["updated_at"] <= cutoff:
            selected[record.files_root] = "terminal_grace_elapsed"

    for record in records:
        if record.snapshot_id in active:
            continue
        if record.location == "snapshot" and record.snapshot_id in indexed:
            selected[record.files_root] = "verified_pack_indexed"

    remaining_terminal_bytes = sum(
        record.payload_bytes
        for record in terminal_records
        if record.files_root not in selected
    )
    if remaining_terminal_bytes > max_terminal_payload_bytes:
        for record in sorted(
            terminal_records,
            key=lambda item: terminal[item.snapshot_id]["updated_at"],
        ):
            if record.files_root in selected:
                continue
            selected[record.files_root] = "terminal_capacity_limit"
            remaining_terminal_bytes -= record.payload_bytes
            if remaining_terminal_bytes <= max_terminal_payload_bytes:
                break

    released_ids: list[str] = []
    released_paths: list[str] = []
    released_invalid = 0
    bytes_released = 0
    by_path = {item.files_root: item for item in records}
    for files_root, reason in sorted(selected.items(), key=lambda item: str(item[0])):
        record = by_path[files_root]
        task_info = terminal.get(record.snapshot_id, {})
        if dry_run:
            released = True
        else:
            released = _release_payload(
                queue,
                record,
                reason=reason,
                task_ids=tuple(task_info.get("task_ids", ())),
                task_states=tuple(task_info.get("task_states", ())),
                now=observed_at,
                warnings=warnings,
            )
        if not released:
            continue
        released_ids.append(record.snapshot_id)
        released_paths.append(str(record.files_root))
        bytes_released += record.payload_bytes
        if record.location == "invalid_audit":
            released_invalid += 1

    result = ExportSnapshotGcResult(
        scanned_snapshot_count=len(records),
        active_snapshot_count=len(active),
        terminal_snapshot_count=len(terminal),
        released_snapshot_count=len(released_ids) - released_invalid,
        released_invalid_snapshot_count=released_invalid,
        released_snapshot_ids=tuple(released_ids),
        released_paths=tuple(released_paths),
        bytes_before=bytes_before,
        bytes_released=bytes_released,
        bytes_after=max(bytes_before - bytes_released, 0),
        dry_run=dry_run,
        warnings=tuple(warnings),
    )
    _append_gc_audit(queue, {"action": "gc_run", **asdict(result)}, observed_at)
    return result


def _payload_records(queue: Path, warnings: list[str]) -> list[_PayloadRecord]:
    records: list[_PayloadRecord] = []
    snapshots_root = queue / "snapshots"
    for snapshot_root in sorted(snapshots_root.iterdir()):
        if not snapshot_root.is_dir() or snapshot_root.name.startswith("."):
            continue
        record = _snapshot_payload_record(snapshot_root, "snapshot", warnings)
        if record is not None:
            records.append(record)

    invalid_root = queue / "audit" / "invalid_snapshots"
    if invalid_root.is_dir():
        for snapshot_root in sorted(invalid_root.iterdir()):
            if not snapshot_root.is_dir() or snapshot_root.name.startswith("."):
                continue
            record = _snapshot_payload_record(snapshot_root, "invalid_audit", warnings)
            if record is not None:
                records.append(record)
    return records


def _snapshot_payload_record(
    snapshot_root: Path,
    location: str,
    warnings: list[str],
) -> _PayloadRecord | None:
    files_root = snapshot_root / "files"
    manifest_path = snapshot_root / "manifest.json"
    if not files_root.is_dir() or not manifest_path.is_file():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ExportSnapshotManifest.model_validate(raw)
    except (OSError, ValueError, TypeError) as exc:
        warnings.append(f"invalid_snapshot_manifest:{manifest_path}:{type(exc).__name__}")
        return None
    return _PayloadRecord(
        snapshot_id=manifest.snapshot_id,
        snapshot_root=snapshot_root,
        files_root=files_root,
        manifest_sha256=manifest.manifest_sha256,
        payload_bytes=_path_size(files_root),
        observed_at=_as_utc(manifest.created_at),
        location=location,
    )


def _active_snapshot_ids(queue: Path, warnings: list[str]) -> set[str]:
    result: set[str] = set()
    for state in _ACTIVE_STATES:
        for task_path in (queue / state).glob("*/task.json"):
            task = _read_task(task_path, warnings)
            if task is not None:
                result.add(task.snapshot_id)
    for receipt_path in (queue / "receipts" / "inbox").glob("*/receipt.json"):
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f"invalid_receipt:{receipt_path}:{type(exc).__name__}")
            continue
        snapshot_id = str(payload.get("snapshot_id") or "")
        if snapshot_id:
            result.add(snapshot_id)
    return result


def _terminal_snapshot_refs(
    queue: Path,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for state in _TERMINAL_STATES:
        for task_path in (queue / state).glob("*/task.json"):
            task = _read_task(task_path, warnings)
            if task is None:
                continue
            status_path = queue / "status" / f"{task.task_id}.json"
            updated_at = task.requested_at
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("updated_at"):
                    updated_at = _parse_datetime(status["updated_at"])
            except (OSError, ValueError, TypeError):
                pass
            row = result.setdefault(
                task.snapshot_id,
                {"updated_at": _as_utc(updated_at), "task_ids": [], "task_states": []},
            )
            row["updated_at"] = max(row["updated_at"], _as_utc(updated_at))
            row["task_ids"].append(task.task_id)
            row["task_states"].append(state)
    return result


def _read_task(path: Path, warnings: list[str]) -> ExportTask | None:
    try:
        return ExportTask.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"invalid_export_task:{path}:{type(exc).__name__}")
        return None


def _release_payload(
    queue: Path,
    record: _PayloadRecord,
    *,
    reason: str,
    task_ids: tuple[str, ...],
    task_states: tuple[str, ...],
    now: datetime,
    warnings: list[str],
) -> bool:
    allowed_root = (
        queue / "snapshots"
        if record.location == "snapshot"
        else queue / "audit" / "invalid_snapshots"
    ).resolve()
    snapshot_root = record.snapshot_root.resolve()
    files_root = record.files_root
    try:
        snapshot_root.relative_to(allowed_root)
    except ValueError:
        warnings.append(f"refused_snapshot_outside_queue:{record.snapshot_root}")
        return False
    if files_root.is_symlink() or files_root.resolve().parent != snapshot_root:
        warnings.append(f"refused_unsafe_snapshot_payload:{files_root}")
        return False
    marker_path = record.snapshot_root / "RELEASED.json"
    marker = {
        "schema_version": "quant_lab_export_snapshot_payload_release.v1",
        "snapshot_id": record.snapshot_id,
        "manifest_sha256": record.manifest_sha256,
        "released_at": now.isoformat(),
        "released_bytes": record.payload_bytes,
        "reason": reason,
        "location": record.location,
        "task_ids": list(task_ids),
        "task_states": list(task_states),
        "manifest_preserved": True,
        "signature_preserved": True,
        "state": "release_started",
    }
    try:
        atomic_write_json(marker_path, marker, mode=0o660)
        shutil.rmtree(files_root)
        atomic_write_json(marker_path, marker | {"state": "released"}, mode=0o660)
    except OSError as exc:
        warnings.append(f"snapshot_payload_release_failed:{files_root}:{exc}")
        return False
    _append_gc_audit(
        queue,
        {"action": "payload_released", **marker, "state": "released"},
        now,
    )
    return True


def _append_gc_audit(queue: Path, payload: dict[str, Any], now: datetime) -> None:
    audit_root = queue / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    with (audit_root / "snapshot_gc.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"observed_at": now.isoformat(), **payload},
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )


def _path_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def _parse_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
