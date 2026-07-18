from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from quant_lab.export_plane.status import atomic_write_json, read_json
from quant_lab.research_plane.contracts import ResearchTaskState, ResearchTaskStatus

TASK_DIRECTORY_STATES = (
    "pending",
    "running",
    "completed",
    "failed",
    "expired",
    "cancelled",
)


def ensure_research_queue_layout(root: str | Path) -> Path:
    queue_root = Path(root)
    for relative in (
        *TASK_DIRECTORY_STATES,
        "requests/pending",
        "requests/processing",
        "requests/completed",
        "requests/failed",
        "requests/status",
        "results/inbox",
        "results/imported",
        "results/rejected",
        "snapshots",
        "audit",
        "status",
        "validation",
    ):
        path = queue_root / relative
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o2770)
        except OSError:
            pass
    return queue_root


def research_task_directory(root: str | Path, state: str, task_id: str) -> Path:
    if state not in TASK_DIRECTORY_STATES:
        raise ValueError(f"unknown research queue state: {state}")
    _require_identifier(task_id)
    return Path(root) / state / task_id


def find_research_task_directory(root: str | Path, task_id: str) -> Path | None:
    for state in TASK_DIRECTORY_STATES:
        candidate = research_task_directory(root, state, task_id)
        if candidate.is_dir():
            return candidate
    return None


def write_research_status(root: str | Path, status: ResearchTaskStatus) -> Path:
    queue_root = ensure_research_queue_layout(root)
    return atomic_write_json(
        queue_root / "status" / f"{status.task_id}.json",
        status.model_dump(mode="json"),
    )


def read_research_status(root: str | Path, task_id: str) -> ResearchTaskStatus | None:
    payload = read_json(Path(root) / "status" / f"{task_id}.json")
    if not payload:
        return None
    try:
        return ResearchTaskStatus.model_validate(payload)
    except ValueError:
        return None


def entry_quality_history_plane_status(root: str | Path) -> dict[str, Any]:
    queue_root = Path(root)
    if not queue_root.is_dir():
        return {
            "schema_version": "quant_lab_entry_quality_research_plane_status.v1",
            "task_type": "entry_quality_history",
            "state": "idle",
            "task": None,
            "recent": [],
            "nas_offline_behavior": "wait_no_local_fallback",
            "research_only": True,
            "live_order_effect": "none",
        }
    statuses: list[ResearchTaskStatus] = []
    for path in (queue_root / "status").glob("*.json"):
        try:
            statuses.append(ResearchTaskStatus.model_validate_json(path.read_text("utf-8")))
        except (OSError, ValueError):
            continue
    statuses.sort(key=_status_sort_key, reverse=True)
    latest = statuses[0] if statuses else None
    active = next(
        (
            status
            for status in statuses
            if status.state
            not in {
                ResearchTaskState.COMPLETED,
                ResearchTaskState.REJECTED,
                ResearchTaskState.FAILED,
                ResearchTaskState.EXPIRED,
                ResearchTaskState.CANCELLED,
            }
        ),
        None,
    )
    selected = active or latest
    return {
        "schema_version": "quant_lab_entry_quality_research_plane_status.v1",
        "task_type": "entry_quality_history",
        "state": selected.state.value if selected else "idle",
        "task": selected.model_dump(mode="json") if selected else None,
        "recent": [status.model_dump(mode="json") for status in statuses[:12]],
        "nas_offline_behavior": "wait_no_local_fallback",
        "research_only": True,
        "live_order_effect": "none",
    }


def _status_sort_key(status: ResearchTaskStatus) -> datetime:
    return status.completed_at or status.heartbeat_at or status.claimed_at or status.requested_at


def _require_identifier(value: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not value or len(value) > 180 or any(character not in allowed for character in value):
        raise ValueError("unsafe research task id")
