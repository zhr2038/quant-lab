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
            for item in sorted(statuses, key=lambda value: value.updated_at, reverse=True)
            if item.state.value
            not in {"download_ready", "failed", "expired", "cancelled"}
        ),
        None,
    )
    request_state = (
        str(request_statuses[0].get("state") or "") if request_statuses else ""
    )
    effective_state = (
        active.state.value
        if active is not None
        else (
            request_state
            if request_state not in {"", "cancelled", "failed"}
            else ("download_ready" if latest else request_state or "idle")
        )
    )
    return {
        "export_plane": "nas_local",
        "state": effective_state,
        "task": active.model_dump(mode="json") if active is not None else None,
        "request": request_statuses[0] if request_statuses else None,
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
