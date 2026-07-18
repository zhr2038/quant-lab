"""Deterministic, hash-validated stage checkpoints for resumable audits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(root: Path, stage: str) -> Path:
    return Path(root) / "checkpoints" / f"{stage}.done.json"


def write_checkpoint(
    *,
    root: Path,
    stage: str,
    started_at: str,
    git_commit: str,
    snapshot_id: str,
    command: str,
    outputs: Iterable[Path],
    finished_at: str | None = None,
    status: str = "completed",
) -> Path:
    """Atomically write a checkpoint with hashes of every declared output."""
    resolved = [Path(path).resolve() for path in outputs]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint outputs missing: {missing}")
    payload = {
        "stage": stage,
        "started_at": started_at,
        "finished_at": finished_at or _utc_now(),
        "git_commit": git_commit,
        "snapshot_id": snapshot_id,
        "command": command,
        "status": status,
        "outputs": [str(path) for path in resolved],
        "sha256": {str(path): sha256_file(path) for path in resolved},
    }
    path = checkpoint_path(Path(root), stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_checkpoint(root: Path, stage: str) -> dict | None:
    path = checkpoint_path(Path(root), stage)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_stage_complete(root: Path, stage: str) -> bool:
    """Return true only for a completed checkpoint whose outputs still match."""
    payload = load_checkpoint(Path(root), stage)
    if not payload or payload.get("stage") != stage or payload.get("status") != "completed":
        return False
    outputs = payload.get("outputs") or []
    hashes = payload.get("sha256") or {}
    if not outputs or set(outputs) != set(hashes):
        return False
    for raw_path in outputs:
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != hashes.get(raw_path):
            return False
    return True
