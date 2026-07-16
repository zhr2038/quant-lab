from __future__ import annotations

import secrets
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from quant_lab.export_plane.contracts import ExportPackIndexEntry, ExportTaskStatus
from quant_lab.export_plane.signatures import signed_download_token
from quant_lab.export_plane.status import atomic_write_json, ensure_queue_layout, read_json

INDEX_FILE = "cloud_index.json"
_ACTIVE_TASK_STATES = ("pending", "running")
_ACTIVE_REQUEST_STATES = ("pending", "processing")
_TERMINAL_TASK_STATES = {"download_ready", "failed", "expired", "cancelled"}


def load_cloud_index(queue_root: str | Path) -> list[ExportPackIndexEntry]:
    root = ensure_queue_layout(queue_root)
    payload = read_json(root / INDEX_FILE)
    rows = payload.get("packs") if isinstance(payload.get("packs"), list) else []
    result: list[ExportPackIndexEntry] = []
    for row in rows:
        try:
            result.append(ExportPackIndexEntry.model_validate(row))
        except (TypeError, ValueError):
            continue
    return sorted(result, key=lambda item: item.accepted_at, reverse=True)


def write_cloud_index(queue_root: str | Path, rows: list[ExportPackIndexEntry]) -> Path:
    root = ensure_queue_layout(queue_root)
    deduped: dict[str, ExportPackIndexEntry] = {}
    for row in sorted(rows, key=lambda item: item.accepted_at):
        deduped[row.pack_id] = row
    payload = {
        "schema_version": "quant_lab_export_cloud_index.v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "packs": [
            row.model_dump(mode="json")
            for row in sorted(deduped.values(), key=lambda item: item.accepted_at, reverse=True)
        ],
    }
    return atomic_write_json(root / INDEX_FILE, payload)


def export_plane_status(
    queue_root: str | Path,
    *,
    export_date: date | None = None,
    nas_base_url: str | None = None,
    download_secret: bytes | None = None,
    download_key_id: str = "nas-download-v1",
    download_ttl_seconds: int = 1800,
) -> dict[str, Any]:
    root = ensure_queue_layout(queue_root)
    statuses: list[ExportTaskStatus] = []
    for path in sorted((root / "status").glob("*.json"), reverse=True):
        try:
            statuses.append(ExportTaskStatus.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    packs = load_cloud_index(root)
    request_statuses: list[dict[str, Any]] = []
    for path in sorted((root / "requests" / "status").glob("*.json"), reverse=True):
        value = read_json(path)
        if value:
            request_statuses.append(value)
    statuses.sort(key=lambda value: value.updated_at, reverse=True)
    request_statuses.sort(key=_request_status_updated_at, reverse=True)
    active_task_ids = {
        path.name
        for state in _ACTIVE_TASK_STATES
        for path in (root / state).iterdir()
        if path.is_dir()
    }
    active_request_ids = {
        path.stem
        for state in _ACTIVE_REQUEST_STATES
        for path in (root / "requests" / state).glob("*.json")
        if path.is_file()
    }
    if export_date is not None:
        requested = [item for item in packs if item.export_date == export_date]
    else:
        requested = packs
    latest = requested[0] if requested else (packs[0] if packs else None)
    pack_rows = [
        _pack_status_row(
            item,
            nas_base_url=nas_base_url,
            download_secret=download_secret,
            download_key_id=download_key_id,
            download_ttl_seconds=download_ttl_seconds,
        )
        for item in packs[:30]
    ]
    latest_row = (
        _pack_status_row(
            latest,
            nas_base_url=nas_base_url,
            download_secret=download_secret,
            download_key_id=download_key_id,
            download_ttl_seconds=download_ttl_seconds,
        )
        if latest is not None
        else None
    )
    active = next(
        (
            item
            for item in statuses
            if item.task_id in active_task_ids
            and item.state.value not in _TERMINAL_TASK_STATES
        ),
        None,
    )
    active_request = next(
        (
            item
            for item in request_statuses
            if str(item.get("request_id") or "") in active_request_ids
            or str(item.get("task_id") or "") in active_task_ids
        ),
        None,
    )
    terminal_events: list[tuple[datetime, str, str, Any]] = [
        (item.updated_at, item.state.value, "task", item)
        for item in statuses
        if item.state.value in _TERMINAL_TASK_STATES
    ]
    terminal_events.extend(
        (
            _request_status_updated_at(item),
            str(item.get("state") or ""),
            "request",
            item,
        )
        for item in request_statuses
        if str(item.get("state") or "") in {"failed", "cancelled"}
    )
    terminal_event = max(terminal_events, default=None, key=lambda item: item[0])
    if terminal_event is not None and latest is not None:
        if terminal_event[0] <= latest.accepted_at:
            terminal_event = None

    current_task: ExportTaskStatus | None = active
    current_request: dict[str, Any] | None = active_request
    if active is not None:
        effective_state = active.state.value
    elif active_request is not None:
        effective_state = str(active_request.get("state") or "pending")
    elif terminal_event is not None:
        effective_state = terminal_event[1]
        if terminal_event[2] == "task":
            current_task = terminal_event[3]
        else:
            current_request = terminal_event[3]
    elif latest is not None:
        effective_state = "download_ready"
    else:
        effective_state = "idle"
    return {
        "export_plane": "nas_local",
        "state": effective_state,
        "task": current_task.model_dump(mode="json") if current_task is not None else None,
        "request": current_request,
        "latest_pack": latest_row,
        "packs": pack_rows,
        "pack_count": len(packs),
        "authoritative_input_snapshot": bool(latest and latest.authoritative_input_snapshot),
        "nas_artifact_validated": bool(latest and latest.nas_artifact_validated),
        "control_plane_receipt_verified": bool(
            latest and latest.control_plane_receipt_verified
        ),
        "download_ready": bool(latest and latest.download_ready),
        "nas_online": None,
        "nas_online_reason": "cloud_control_plane_does_not_probe_private_nas",
        "storage_location": "nas_only",
        "cloud_zip_present": False,
        "live_order_effect": "none_read_only_export_control_plane",
    }


def _request_status_updated_at(value: dict[str, Any]) -> datetime:
    raw = value.get("updated_at") or value.get("requested_at")
    if raw:
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.astimezone(UTC)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=UTC)


def _pack_status_row(
    item: ExportPackIndexEntry,
    *,
    nas_base_url: str | None,
    download_secret: bytes | None,
    download_key_id: str,
    download_ttl_seconds: int,
) -> dict[str, Any]:
    row = item.model_dump(mode="json")
    row["download_url"] = None
    if nas_base_url and download_secret and item.download_ready:
        expires_at = int(time.time()) + max(60, min(download_ttl_seconds, 86_400))
        nonce = secrets.token_urlsafe(12)
        token = signed_download_token(
            pack_id=item.pack_id,
            pack_sha256=item.pack_sha256,
            expires_at=expires_at,
            nonce=nonce,
            key_id=download_key_id,
            secret=download_secret,
        )
        query = urlencode(
            {
                "sha256": item.pack_sha256,
                "expires": expires_at,
                "nonce": nonce,
                "key_id": download_key_id,
                "signature": token,
            }
        )
        row["download_url"] = f"{nas_base_url.rstrip('/')}/download/{item.pack_id}?{query}"
        row["download_expires_at"] = datetime.fromtimestamp(expires_at, UTC).isoformat()
    return row
