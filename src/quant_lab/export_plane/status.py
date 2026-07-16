from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

QUEUE_STATES = (
    "pending",
    "running",
    "completed",
    "failed",
    "expired",
    "cancelled",
)


def ensure_queue_layout(root: str | Path) -> Path:
    queue_root = Path(root)
    for relative in (
        *QUEUE_STATES,
        "receipts/inbox",
        "receipts/imported",
        "receipts/rejected",
        "snapshots",
        "status",
        "requests/pending",
        "requests/processing",
        "requests/completed",
        "requests/failed",
        "requests/status",
    ):
        path = queue_root / relative
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o2770)
        except OSError:
            pass
    return queue_root


def atomic_write_json(path: str | Path, value: Any, *, mode: int = 0o660) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, target)
        _fsync_directory(target.parent)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def task_directory(root: str | Path, state: str, task_id: str) -> Path:
    if state not in QUEUE_STATES:
        raise ValueError(f"unknown queue state: {state}")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not task_id or any(char not in allowed for char in task_id):
        raise ValueError("unsafe task_id")
    return Path(root) / state / task_id


def find_task_directory(root: str | Path, task_id: str) -> Path | None:
    for state in QUEUE_STATES:
        candidate = task_directory(root, state, task_id)
        if candidate.is_dir():
            return candidate
    return None


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
