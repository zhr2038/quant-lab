from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from quant_lab.export_plane.contracts import ExportPackIndexEntry
from quant_lab.export_plane.status import atomic_write_json


def enforce_accepted_retention(
    *,
    accepted_root: str | Path,
    index_path: str | Path,
    retention_days: int,
    max_total_bytes: int,
    min_keep_packs: int,
    audit_log_path: str | Path,
) -> dict[str, Any]:
    root = Path(accepted_root).resolve()
    index = Path(index_path)
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        rows = [ExportPackIndexEntry.model_validate(row) for row in payload.get("packs", [])]
    except (OSError, ValueError, TypeError):
        return {"removed": [], "remaining": 0, "total_bytes": 0}
    rows.sort(key=lambda item: item.accepted_at, reverse=True)
    keep_ids = {item.pack_id for item in rows[: max(0, min_keep_packs)]}
    keep_ids.update(item.pack_id for item in rows if not item.ai_consumed)
    cutoff = datetime.now(UTC) - timedelta(days=max(1, retention_days))
    total = sum(item.pack_size_bytes for item in rows)
    removed: list[dict[str, Any]] = []
    kept: list[ExportPackIndexEntry] = []
    for row in rows:
        over_age = row.accepted_at < cutoff
        over_capacity = total > max_total_bytes
        if row.pack_id in keep_ids or not (over_age or over_capacity):
            kept.append(row)
            continue
        pack_path = _safe_pack_path(root, row.download_relative_path)
        pack_dir = pack_path.parent
        if pack_dir.is_dir():
            shutil.rmtree(pack_dir)
        total -= row.pack_size_bytes
        removed.append(
            {
                "pack_id": row.pack_id,
                "pack_sha256": row.pack_sha256,
                "removed_at": datetime.now(UTC).isoformat(),
                "reason": "retention_age" if over_age else "retention_capacity",
            }
        )
    atomic_write_json(
        index,
        {
            "schema_version": "quant_lab_export_accepted_index.v1",
            "updated_at": datetime.now(UTC).isoformat(),
            "packs": [item.model_dump(mode="json") for item in kept],
        },
        mode=0o640,
    )
    if removed:
        _append_audit(audit_log_path, removed)
    return {"removed": removed, "remaining": len(kept), "total_bytes": total}


def _safe_pack_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe retention pack path")
    path = (root / relative.as_posix()).resolve()
    path.relative_to(root)
    return path


def _append_audit(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
