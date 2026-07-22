from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_lab.export_plane.status import atomic_write_json, read_json
from quant_lab.research_plane.contracts import (
    RESEARCH_RESULT_ADAPTER,
    RESEARCH_SNAPSHOT_ADAPTER,
    RESEARCH_TASK_ADAPTER,
    FactorFactoryResultManifest,
    FactorFactorySnapshotManifest,
    FactorFactoryTask,
    ResearchTaskLease,
    ResearchTaskState,
    ResearchTaskStatus,
    V5CandidateEvidenceResultManifest,
    V5CandidateEvidenceSnapshotManifest,
    V5CandidateEvidenceTask,
)

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
        "lease",
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


def write_research_lease(root: str | Path, lease: ResearchTaskLease) -> Path:
    queue_root = ensure_research_queue_layout(root)
    return atomic_write_json(
        queue_root / "lease" / f"{lease.task_id}.json",
        lease.model_dump(mode="json"),
    )


def read_research_lease(root: str | Path, task_id: str) -> ResearchTaskLease | None:
    payload = read_json(Path(root) / "lease" / f"{task_id}.json")
    if not payload:
        return None
    try:
        return ResearchTaskLease.model_validate(payload)
    except ValueError:
        return None


def entry_quality_history_plane_status(root: str | Path) -> dict[str, Any]:
    return _research_plane_status_for_type(root, "entry_quality_history")


def alpha_factory_plane_status(root: str | Path) -> dict[str, Any]:
    return _research_plane_status_for_type(root, "alpha_factory")


def factor_research_plane_status(root: str | Path) -> dict[str, Any]:
    return _research_plane_status_for_type(root, "factor_research")


def factor_factory_plane_status(root: str | Path) -> dict[str, Any]:
    return _research_plane_status_for_type(root, "factor_factory")


def v5_candidate_evidence_plane_status(root: str | Path) -> dict[str, Any]:
    return _research_plane_status_for_type(root, "v5_candidate_evidence")


def research_plane_status(root: str | Path) -> dict[str, Any]:
    entry_quality = entry_quality_history_plane_status(root)
    alpha_factory = alpha_factory_plane_status(root)
    factor_research = factor_research_plane_status(root)
    factor_factory = factor_factory_plane_status(root)
    v5_candidate_evidence = v5_candidate_evidence_plane_status(root)
    return {
        "schema_version": "quant_lab_research_plane_status.v2",
        "state": _aggregate_state(
            [
                entry_quality,
                alpha_factory,
                factor_research,
                factor_factory,
                v5_candidate_evidence,
            ]
        ),
        "tasks": {
            "entry_quality_history": entry_quality,
            "alpha_factory": alpha_factory,
            "factor_research": factor_research,
            "factor_factory": factor_factory,
            "v5_candidate_evidence": v5_candidate_evidence,
        },
        "nas_offline_behavior": "wait_no_local_fallback",
        "research_only": True,
        "live_order_effect": "none",
    }


def _research_plane_status_for_type(
    root: str | Path,
    task_type: str,
) -> dict[str, Any]:
    queue_root = Path(root)
    schema_version = "quant_lab_research_plane_task_status.v2"
    if not queue_root.is_dir():
        return {
            "schema_version": schema_version,
            "task_type": task_type,
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
            status = ResearchTaskStatus.model_validate_json(path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if status.task_type == task_type:
            statuses.append(status)
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
    request_status_name = {
        "factor_factory": "factor_factory_request.json",
        "v5_candidate_evidence": "v5_candidate_evidence_request.json",
    }.get(task_type)
    request_status = (
        read_json(queue_root / "status" / request_status_name)
        if request_status_name is not None
        else None
    )
    selected_payload = selected.model_dump(mode="json") if selected else None
    if selected_payload is not None:
        lease = read_research_lease(queue_root, selected.task_id)
        if lease is not None:
            selected_payload["worker_heartbeat_at"] = lease.heartbeat_at.isoformat()
            selected_payload["worker_lease_expires_at"] = lease.lease_expires_at.isoformat()
            selected_payload["worker_lease_sequence"] = lease.sequence
        if task_type == "factor_factory":
            selected_payload.update(_factor_factory_status_details(queue_root, selected))
        elif task_type == "v5_candidate_evidence":
            selected_payload.update(
                _v5_candidate_evidence_status_details(queue_root, selected)
            )
    state = selected.state.value if selected else "idle"
    snapshot_root = queue_root / "snapshots"
    transient_snapshot_task = task_type in {"factor_factory", "v5_candidate_evidence"}
    snapshot_rehydrating = transient_snapshot_task and any(
        snapshot_root.glob(".rehydrate.*.partial")
    )
    snapshot_sealing = transient_snapshot_task and any(snapshot_root.glob(".sealing.*.partial"))
    if snapshot_rehydrating:
        state = "snapshot_rehydrating"
    elif snapshot_sealing:
        state = "snapshot_sealing"
    elif (
        active is None
        and request_status
        and request_status.get("state")
        in {
            "already_current",
            "already_current_no_update",
        }
    ):
        state = "up_to_date"
    elif (
        active is None
        and request_status
        and request_status.get("state") == "generation_integrity_failed"
    ):
        state = "generation_integrity_failed"
    payload = {
        "schema_version": schema_version,
        "task_type": task_type,
        "state": state,
        "task": selected_payload,
        "request": request_status,
        "recent": [status.model_dump(mode="json") for status in statuses[:12]],
        "nas_offline_behavior": "wait_no_local_fallback",
        "research_only": True,
        "live_order_effect": "none",
    }
    if task_type == "factor_factory":
        request = request_status or {}
        payload_state = request.get("snapshot_payload_state")
        if snapshot_rehydrating:
            payload_state = "rehydrating"
        elif snapshot_sealing:
            payload_state = "sealing"
        payload.update(
            {
                "health_state": request.get("health_state") or state,
                "request_outcome": request.get("request_outcome") or request.get("state"),
                "input_fingerprint": request.get("input_fingerprint"),
                "fingerprint_matches_generation": bool(
                    request.get("fingerprint_matches_generation", False)
                ),
                "snapshot_materialized": bool(request.get("snapshot_materialized", False)),
                "snapshot_rehydrated": bool(request.get("snapshot_rehydrated", False)),
                "snapshot_payload_state": payload_state,
                "compressed_input_bytes": request.get("compressed_input_bytes"),
                "estimated_uncompressed_input_bytes": request.get(
                    "estimated_uncompressed_input_bytes"
                ),
                "already_current_at": request.get("already_current_at"),
                "no_update_reason": request.get("no_update_reason"),
            }
        )
    elif task_type == "v5_candidate_evidence":
        request = request_status or {}
        latest_completed = next(
            (
                status
                for status in statuses
                if status.state == ResearchTaskState.COMPLETED
                and status.gold_generation_id is not None
            ),
            None,
        )
        completed_at = latest_completed.completed_at if latest_completed is not None else None
        if completed_at is None and request.get("current_generation_published_at"):
            try:
                completed_at = datetime.fromisoformat(
                    str(request["current_generation_published_at"]).replace("Z", "+00:00")
                ).astimezone(UTC)
            except ValueError:
                completed_at = None
        generation_id = (
            latest_completed.gold_generation_id
            if latest_completed is not None
            else request.get("current_generation_id")
        )
        payload_state = request.get("snapshot_payload_state")
        if snapshot_rehydrating:
            payload_state = "rehydrating"
        elif snapshot_sealing:
            payload_state = "sealing"
        payload.update(
            {
                "health_state": request.get("health_state") or state,
                "request_outcome": request.get("request_outcome") or request.get("state"),
                "candidate_evidence_generation_id": generation_id,
                "generation_age_seconds": (
                    max(0, int((datetime.now(UTC) - completed_at).total_seconds()))
                    if completed_at is not None
                    else None
                ),
                "last_completed_at": completed_at.isoformat() if completed_at else None,
                "input_fingerprint": request.get("input_fingerprint"),
                "pending_running_state": (
                    active.state.value if active is not None else None
                ),
                "fingerprint_matches_generation": bool(
                    request.get("fingerprint_matches_generation", False)
                ),
                "snapshot_materialized": bool(request.get("snapshot_materialized", False)),
                "snapshot_rehydrated": bool(request.get("snapshot_rehydrated", False)),
                "snapshot_payload_state": payload_state,
                "compressed_input_bytes": request.get("compressed_input_bytes"),
                "estimated_uncompressed_input_bytes": request.get(
                    "estimated_uncompressed_input_bytes"
                ),
                "already_current_at": request.get("already_current_at"),
            }
        )
    return payload


def _aggregate_state(statuses: list[dict[str, Any]]) -> str:
    states = [str(status.get("state") or "idle") for status in statuses]
    for state in ("failed", "rejected", "expired"):
        if state in states:
            return state
    for state in (
        "publishing",
        "validating_on_cloud",
        "uploading",
        "validating_on_nas",
        "computing_correlation",
        "computing_evidence",
        "computing_samples",
        "computing_labels",
        "computing_values",
        "computing",
        "syncing",
        "claimed",
        "pending",
    ):
        if state in states:
            return state
    return "completed" if "completed" in states else "idle"


def _factor_factory_status_details(
    queue_root: Path,
    status: ResearchTaskStatus,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    task_directory = find_research_task_directory(queue_root, status.task_id)
    if task_directory is not None:
        try:
            task = RESEARCH_TASK_ADAPTER.validate_json(
                (task_directory / "task.json").read_text("utf-8")
            )
        except (OSError, ValueError):
            task = None
        if isinstance(task, FactorFactoryTask):
            details.update(
                {
                    "factor_plan_digest": task.factor_plan_digest,
                    "as_of_date": task.as_of_date.isoformat(),
                    "feature_set": task.feature_set,
                    "feature_version": task.feature_version,
                    "factor_version": task.factor_version,
                    "timeframe": task.timeframe,
                    "horizon_bars": list(task.horizon_bars),
                }
            )
            try:
                snapshot = RESEARCH_SNAPSHOT_ADAPTER.validate_json(
                    (queue_root / "snapshots" / task.snapshot_id / "manifest.json").read_text(
                        "utf-8"
                    )
                )
            except (OSError, ValueError):
                snapshot = None
            if isinstance(snapshot, FactorFactorySnapshotManifest):
                snapshot_root = queue_root / "snapshots" / task.snapshot_id
                payload_state = "sealed"
                rehydrate_partials = (queue_root / "snapshots").glob(
                    f".rehydrate.{task.snapshot_id}.*.partial"
                )
                if any(rehydrate_partials):
                    payload_state = "rehydrating"
                elif (snapshot_root / "FILES_RELEASED.json").is_file():
                    payload_state = "released"
                elif (snapshot_root / "FILES_REHYDRATED.json").is_file():
                    payload_state = "rehydrated"
                elif (snapshot_root / "files").is_dir():
                    payload_state = "materialized"
                details.update(
                    {
                        "factor_count": snapshot.factor_plan.factor_count,
                        "input_fingerprint": (
                            snapshot.input_fingerprint.model_dump(mode="json")
                            if snapshot.input_fingerprint is not None
                            else None
                        ),
                        "snapshot_payload_state": payload_state,
                        "compressed_input_bytes": (
                            snapshot.compressed_input_bytes or snapshot.total_input_bytes
                        ),
                        "estimated_uncompressed_input_bytes": (
                            snapshot.estimated_uncompressed_input_bytes
                            if snapshot.estimated_uncompressed_input_bytes is not None
                            else snapshot.estimated_uncompressed_bytes
                        ),
                    }
                )
    for result_state in ("inbox", "imported"):
        manifest_path = queue_root / "results" / result_state / status.task_id / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = RESEARCH_RESULT_ADAPTER.validate_json(manifest_path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, FactorFactoryResultManifest):
            continue
        rows = {item.dataset_name: item.row_count for item in manifest.outputs}
        details.update(
            {
                "factor_count": manifest.factor_count,
                "value_rows": sum(item.row_count for item in manifest.value_partitions),
                "evidence_rows": rows.get("factor_evidence", 0),
                "correlation_rows": rows.get("factor_correlation_daily", 0),
                "generation_id": manifest.generation_id,
                "generation_age_seconds": max(
                    0,
                    int((datetime.now(UTC) - manifest.completed_at).total_seconds()),
                ),
            }
        )
        break
    return details


def _v5_candidate_evidence_status_details(
    queue_root: Path,
    status: ResearchTaskStatus,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    task_directory = find_research_task_directory(queue_root, status.task_id)
    if task_directory is not None:
        try:
            task = RESEARCH_TASK_ADAPTER.validate_json(
                (task_directory / "task.json").read_text("utf-8")
            )
        except (OSError, ValueError):
            task = None
        if isinstance(task, V5CandidateEvidenceTask):
            details.update(
                {
                    "as_of_date": task.as_of_date.isoformat(),
                    "lookback_days": task.lookback_days,
                    "horizon_hours": list(task.horizon_hours),
                    "input_fingerprint_digest": task.input_fingerprint_digest,
                    "previous_generation_id": task.previous_generation_id,
                }
            )
            try:
                snapshot = RESEARCH_SNAPSHOT_ADAPTER.validate_json(
                    (queue_root / "snapshots" / task.snapshot_id / "manifest.json").read_text(
                        "utf-8"
                    )
                )
            except (OSError, ValueError):
                snapshot = None
            if isinstance(snapshot, V5CandidateEvidenceSnapshotManifest):
                snapshot_root = queue_root / "snapshots" / task.snapshot_id
                payload_state = "sealed"
                if any(
                    (queue_root / "snapshots").glob(
                        f".rehydrate.{task.snapshot_id}.*.partial"
                    )
                ):
                    payload_state = "rehydrating"
                elif (snapshot_root / "FILES_RELEASED.json").is_file():
                    payload_state = "released"
                elif (snapshot_root / "FILES_REHYDRATED.json").is_file():
                    payload_state = "rehydrated"
                elif (snapshot_root / "files").is_dir():
                    payload_state = "materialized"
                details.update(
                    {
                        "input_fingerprint": {
                            "schema_version": "v5_candidate_evidence_input_fingerprint.v1",
                            "projection_version": snapshot.projection_version,
                            "input_fingerprint_digest": snapshot.input_fingerprint_digest,
                            "candidate_event_digest": snapshot.candidate_event_digest,
                            "market_bar_digest": snapshot.market_bar_digest,
                            "run_summary_digest": snapshot.run_summary_digest,
                            "candidate_event_row_count": snapshot.candidate_event_row_count,
                            "market_bar_row_count": snapshot.market_bar_row_count,
                            "run_summary_row_count": snapshot.run_summary_row_count,
                        },
                        "snapshot_payload_state": payload_state,
                        "compressed_input_bytes": snapshot.total_input_bytes,
                        "estimated_uncompressed_input_bytes": (
                            snapshot.estimated_uncompressed_bytes
                        ),
                    }
                )
    for result_state in ("inbox", "imported"):
        manifest_path = queue_root / "results" / result_state / status.task_id / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = RESEARCH_RESULT_ADAPTER.validate_json(manifest_path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, V5CandidateEvidenceResultManifest):
            continue
        details.update(
            {
                "candidate_evidence_generation_id": manifest.generation_id,
                "generation_age_seconds": max(
                    0,
                    int((datetime.now(UTC) - manifest.completed_at).total_seconds()),
                ),
                "last_completed_at": manifest.completed_at.isoformat(),
                "label_rows": sum(
                    item.row_count
                    for item in manifest.outputs
                    if item.dataset_name == "v5_candidate_label_delta"
                ),
                "sample_rows": sum(
                    item.row_count
                    for item in manifest.outputs
                    if item.dataset_name == "strategy_evidence_sample_delta"
                ),
            }
        )
        break
    return details


def _status_sort_key(status: ResearchTaskStatus) -> datetime:
    return status.completed_at or status.heartbeat_at or status.claimed_at or status.requested_at


def _require_identifier(value: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not value or len(value) > 180 or any(character not in allowed for character in value):
        raise ValueError("unsafe research task id")
