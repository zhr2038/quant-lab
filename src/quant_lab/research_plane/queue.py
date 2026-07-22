from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.research.alpha_factory.factory import (
    ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET,
    alpha_factory_template_registry_digest,
    prepare_alpha_factory_control_state,
)
from quant_lab.research.alpha_factory.factory import (
    SCHEMA_VERSION as ALPHA_FACTORY_SCHEMA_VERSION,
)
from quant_lab.research.entry_quality import (
    ENTRY_QUALITY_SCHEMA_VERSION,
    latest_entry_quality_bundle_id,
    normalize_entry_quality_history_request,
)
from quant_lab.research.factor_research.contracts import (
    SCHEMA_VERSION as FACTOR_RESEARCH_SCHEMA_VERSION,
)
from quant_lab.research.factor_research.registry import (
    EXECUTABLE_HYPOTHESIS_STATUSES,
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
    hypotheses_from_registry,
    hypothesis_registry_digest,
    plan_factor_research_trials,
    prepare_factor_research_control_state,
    recover_retryable_data_quality_hypotheses,
    trial_ledger_digest,
    trial_ledger_frame,
)
from quant_lab.research.second_stage_alpha_factory import (
    SCHEMA_VERSION as SECOND_STAGE_ALPHA_FACTORY_SCHEMA_VERSION,
)
from quant_lab.research_plane.contracts import (
    ALPHA_FACTORY_TASK_TYPE,
    ENTRY_QUALITY_HISTORY_TASK_TYPE,
    FACTOR_FACTORY_TASK_TYPE,
    FACTOR_RESEARCH_TASK_TYPE,
    RESEARCH_TASK_SCHEMA,
    AlphaFactoryTask,
    AlphaFactoryTaskParameters,
    EntryQualityHistoryTaskParameters,
    FactorFactoryTask,
    FactorResearchTask,
    ResearchTask,
    ResearchTaskState,
    ResearchTaskStatus,
)
from quant_lab.research_plane.factor_factory_snapshot import (
    FactorFactorySnapshotPreflight,
    load_factor_factory_generation_binding,
    materialize_factor_factory_snapshot,
    preflight_factor_factory_snapshot,
)
from quant_lab.research_plane.factor_research_publish import (
    current_factor_research_generation_binding,
)
from quant_lab.research_plane.signatures import model_content_sha256, sign_model
from quant_lab.research_plane.snapshot import (
    factor_research_source_identity,
    seal_alpha_factory_snapshot,
    seal_entry_quality_history_snapshot,
    seal_factor_research_snapshot,
)
from quant_lab.research_plane.status import (
    ensure_research_queue_layout,
    find_research_task_directory,
    write_research_status,
)

FACTOR_FACTORY_GENERATION_POINTER = Path("gold") / "factor_factory_generation.json"
FACTOR_FACTORY_NO_UPDATE_POINTER = Path("gold") / "factor_factory_no_update_state.json"


@dataclass(frozen=True)
class FactorFactoryTaskRequestResult:
    state: str
    task_created: bool
    snapshot_materialized: bool
    current_generation_id: str | None
    reason: str
    snapshot_id: str
    task: FactorFactoryTask | None = None
    status: ResearchTaskStatus | None = None
    snapshot_rehydrated: bool = False

    def __iter__(self) -> Iterator[FactorFactoryTask | ResearchTaskStatus]:
        if self.task is None or self.status is None:
            raise TypeError(f"factor factory request {self.state} did not create a task")
        yield self.task
        yield self.status

    def model_dump(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "task_created": self.task_created,
            "snapshot_materialized": self.snapshot_materialized,
            "snapshot_rehydrated": self.snapshot_rehydrated,
            "current_generation_id": self.current_generation_id,
            "reason": self.reason,
            "snapshot_id": self.snapshot_id,
            "task": self.task.model_dump(mode="json") if self.task is not None else None,
            "status": self.status.model_dump(mode="json") if self.status is not None else None,
        }


def create_factor_factory_task(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str,
    feature_set: str = "core",
    feature_version: str = "v0.1",
    factor_version: str = "v0.1",
    timeframe: str = "1H",
    horizon_bars: tuple[int, ...] = (4, 8, 24, 72),
    decision_delay_bars: int = 1,
    max_factors: int = 200,
    min_samples: int = 100,
    top_quantile: float = 0.2,
    cost_quantile: str = "p75",
    lease_seconds: int = 4 * 60 * 60,
    max_attempts: int = 3,
    max_input_bytes: int = 2 * 1024**3,
    max_input_rows: int = 150_000_000,
    max_pending_tasks: int = 1,
    min_recompute_interval_seconds: int = 0,
) -> FactorFactoryTaskRequestResult:
    """Create one coalesced signed full-history Factor Factory NAS task."""

    if max_pending_tasks != 1:
        raise ValueError("factor_factory_max_pending_tasks_must_equal_one")
    if min_recompute_interval_seconds < 0:
        raise ValueError("factor_factory_min_recompute_interval_must_be_non_negative")
    queue = ensure_research_queue_layout(queue_root)
    preflight = preflight_factor_factory_snapshot(
        lake_root,
        queue,
        as_of_date=as_of_date,
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        horizon_bars=horizon_bars,
        decision_delay_bars=decision_delay_bars,
        max_factors=max_factors,
        min_samples=min_samples,
        top_quantile=top_quantile,
        cost_quantile=cost_quantile,
        quant_lab_commit=quant_lab_commit,
    )
    lake = Path(lake_root).resolve(strict=True)
    current = _read_factor_factory_pointer(lake / FACTOR_FACTORY_GENERATION_POINTER)
    if _factor_factory_pointer_matches_preflight(current, preflight):
        result = FactorFactoryTaskRequestResult(
            state="already_current",
            task_created=False,
            snapshot_materialized=False,
            current_generation_id=str(current.get("generation_id") or "") or None,
            reason="factor_factory_inputs_unchanged",
            snapshot_id=preflight.snapshot_id,
        )
        _write_factor_factory_request_result(queue, result)
        return result
    no_update = _read_factor_factory_pointer(lake / FACTOR_FACTORY_NO_UPDATE_POINTER)
    if _factor_factory_pointer_matches_preflight(no_update, preflight):
        result = FactorFactoryTaskRequestResult(
            state="already_current_no_update",
            task_created=False,
            snapshot_materialized=False,
            current_generation_id=None,
            reason=str(no_update.get("reason") or "factor_factory_empty_input_unchanged"),
            snapshot_id=preflight.snapshot_id,
        )
        _write_factor_factory_request_result(queue, result)
        return result
    interval_pointer = current or no_update
    if _factor_factory_recompute_interval_active(
        interval_pointer,
        min_recompute_interval_seconds=min_recompute_interval_seconds,
    ):
        result = FactorFactoryTaskRequestResult(
            state="recompute_deferred",
            task_created=False,
            snapshot_materialized=False,
            current_generation_id=(
                str(interval_pointer.get("generation_id") or "") or None
                if interval_pointer
                else None
            ),
            reason="factor_factory_minimum_recompute_interval_active",
            snapshot_id=preflight.snapshot_id,
        )
        _write_factor_factory_request_result(queue, result)
        return result

    previous_id, previous_digest, _ = load_factor_factory_generation_binding(lake)
    materialization = materialize_factor_factory_snapshot(
        preflight,
        signing_key=signing_key,
        signature_key_id=signature_key_id,
        max_input_bytes=max_input_bytes,
        max_input_rows=max_input_rows,
    )
    snapshot = materialization.manifest
    task_seed = model_content_sha256(
        {
            "schema_version": "quant_lab_factor_factory_task_identity.v1",
            "task_type": FACTOR_FACTORY_TASK_TYPE,
            "snapshot_id": snapshot.snapshot_id,
            "factor_plan_digest": snapshot.factor_plan_digest,
            "source_input_digest": snapshot.source_input_digest,
            "cost_input_digest": snapshot.cost_input_digest,
            "previous_generation_id": previous_id,
            "previous_generation_digest": previous_digest,
            "parameters": {
                "as_of_date": as_of_date,
                "feature_set": snapshot.feature_set,
                "feature_version": snapshot.feature_version,
                "factor_version": snapshot.factor_version,
                "timeframe": snapshot.timeframe,
                "horizon_bars": snapshot.horizon_bars,
                "decision_delay_bars": snapshot.decision_delay_bars,
                "max_factors": snapshot.max_factors,
                "min_samples": snapshot.min_samples,
                "top_quantile": snapshot.top_quantile,
                "cost_quantile": snapshot.cost_quantile,
                "result_mode": snapshot.result_mode,
                "history_mode": snapshot.history_mode,
            },
            "quant_lab_commit": quant_lab_commit,
            "signature_key_id": signature_key_id,
        }
    )[:24]
    task_id = f"factor-factory-{task_seed}"
    existing = find_research_task_directory(queue, task_id)
    if existing is not None:
        task = FactorFactoryTask.model_validate_json((existing / "task.json").read_text("utf-8"))
        status = ResearchTaskStatus.model_validate_json(
            (queue / "status" / f"{task_id}.json").read_text("utf-8")
        )
        result = FactorFactoryTaskRequestResult(
            state="task_created",
            task_created=False,
            snapshot_materialized=materialization.snapshot_materialized,
            snapshot_rehydrated=materialization.snapshot_rehydrated,
            current_generation_id=previous_id,
            reason="factor_factory_task_already_exists",
            snapshot_id=snapshot.snapshot_id,
            task=task,
            status=status,
        )
        _write_factor_factory_request_result(queue, result)
        return result
    _coalesce_factor_factory_pending(queue, successor_task_id=task_id)
    requested_at = datetime.now(UTC)
    provisional = FactorFactoryTask(
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        as_of_date=as_of_date,
        feature_set=snapshot.feature_set,
        feature_version=snapshot.feature_version,
        factor_version=snapshot.factor_version,
        timeframe=snapshot.timeframe,
        horizon_bars=snapshot.horizon_bars,
        decision_delay_bars=snapshot.decision_delay_bars,
        max_factors=snapshot.max_factors,
        min_samples=snapshot.min_samples,
        top_quantile=snapshot.top_quantile,
        cost_quantile=snapshot.cost_quantile,
        factor_plan_digest=snapshot.factor_plan_digest,
        source_input_digest=snapshot.source_input_digest,
        cost_input_digest=snapshot.cost_input_digest,
        quant_lab_commit=quant_lab_commit,
        snapshot_manifest_sha256=snapshot.manifest_sha256,
        previous_generation_id=previous_id,
        previous_generation_digest=previous_digest,
        requested_at=requested_at,
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
        signature_key_id=signature_key_id,
        signature="pending",
    )
    task = provisional.model_copy(update={"signature": sign_model(provisional, signing_key)})
    temporary = queue / "pending" / f".{task_id}.{uuid.uuid4().hex}.partial"
    final = queue / "pending" / task_id
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        (temporary / "task.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")
        (temporary / "snapshot_id").write_text(snapshot.snapshot_id + "\n", encoding="ascii")
        for path in (temporary / "task.json", temporary / "snapshot_id"):
            path.chmod(0o660)
        temporary.chmod(0o2770)
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    status = ResearchTaskStatus(
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        task_type=FACTOR_FACTORY_TASK_TYPE,
        start_date=(snapshot.feature_min_ts or snapshot.generated_at).date(),
        end_date=(snapshot.feature_max_ts or snapshot.generated_at).date(),
        mode="PARITY_FULL/bootstrap_full",
        cost_mode=f"point_in_task_{cost_quantile}",
        state=ResearchTaskState.PENDING,
        requested_at=requested_at,
        max_attempts=max_attempts,
        input_bytes=snapshot.total_input_bytes,
        import_status="waiting_for_nas",
    )
    write_research_status(queue, status)
    result = FactorFactoryTaskRequestResult(
        state="task_created",
        task_created=True,
        snapshot_materialized=materialization.snapshot_materialized,
        snapshot_rehydrated=materialization.snapshot_rehydrated,
        current_generation_id=previous_id,
        reason="factor_factory_task_enqueued",
        snapshot_id=snapshot.snapshot_id,
        task=task,
        status=status,
    )
    _write_factor_factory_request_result(queue, result)
    return result


def _read_factor_factory_pointer(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"factor_factory_pointer_invalid:{path.name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"factor_factory_pointer_invalid:{path.name}")
    return payload


def _factor_factory_pointer_matches_preflight(
    pointer: dict[str, Any],
    preflight: FactorFactorySnapshotPreflight,
) -> bool:
    if not pointer:
        return False
    expected = preflight.identity_payload
    return (
        all(
            pointer.get(name) == expected[name]
            for name in (
                "quant_lab_commit",
                "factor_plan_digest",
                "source_input_digest",
                "cost_input_digest",
                "feature_set",
                "feature_version",
                "factor_version",
                "timeframe",
                "horizon_bars",
                "decision_delay_bars",
                "max_factors",
                "min_samples",
                "top_quantile",
                "cost_quantile",
                "result_mode",
                "history_mode",
            )
        )
        and str(pointer.get("snapshot_id") or "") == preflight.snapshot_id
    )


def _factor_factory_recompute_interval_active(
    pointer: dict[str, Any],
    *,
    min_recompute_interval_seconds: int,
) -> bool:
    if not pointer or min_recompute_interval_seconds <= 0:
        return False
    value = pointer.get("published_at") or pointer.get("observed_at")
    if not value:
        return False
    try:
        observed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    return datetime.now(UTC) - observed.astimezone(UTC) < timedelta(
        seconds=min_recompute_interval_seconds
    )


def _write_factor_factory_request_result(
    queue: Path,
    result: FactorFactoryTaskRequestResult,
) -> None:
    atomic_write_json(
        queue / "status" / "factor_factory_request.json",
        {
            "schema_version": "quant_lab_factor_factory_request_status.v1",
            **{
                key: value
                for key, value in result.model_dump().items()
                if key not in {"task", "status"}
            },
            "health_state": (
                "up_to_date"
                if result.state in {"already_current", "already_current_no_update"}
                else result.state
            ),
            "observed_at": datetime.now(UTC).isoformat(),
        },
    )


def create_factor_research_task(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str,
    start_date: date | None = None,
    end_date: date | None = None,
    max_history_days: int = 730,
    selected_v5_bundle_id: str | None = None,
    lease_seconds: int = 4 * 60 * 60,
    max_attempts: int = 3,
) -> tuple[FactorResearchTask, ResearchTaskStatus]:
    """Create one bounded, deterministic factor-research task from cloud control state."""
    queue = ensure_research_queue_layout(queue_root)
    root = Path(lake_root)
    selected_bundle = selected_v5_bundle_id
    if selected_bundle is None:
        selected_bundle = latest_entry_quality_bundle_id(root)
    selected_bundle = str(selected_bundle or "").strip()
    if not selected_bundle:
        raise RuntimeError("selected_v5_bundle_id_unavailable")
    existing_registry = read_parquet_dataset(root / RESEARCH_HYPOTHESIS_REGISTRY_DATASET)
    effective_registry = prepare_factor_research_control_state(existing_registry)
    effective_registry = recover_retryable_data_quality_hypotheses(
        effective_registry,
        read_parquet_dataset(root / RESEARCH_TRIAL_LEDGER_DATASET),
    )
    hypotheses = hypotheses_from_registry(effective_registry)
    executable = [item for item in hypotheses if item.status in EXECUTABLE_HYPOTHESIS_STATUSES]
    if not executable:
        raise RuntimeError("no_approved_factor_research_hypotheses")
    minimum_history_days = max(
        requirement.min_history_days
        for hypothesis in executable
        for requirement in hypothesis.data_requirements
    )
    if max_history_days < minimum_history_days:
        raise ValueError("factor_research_history_shorter_than_hypothesis_requirement")
    max_horizon_hours = max(
        horizon for hypothesis in executable for horizon in hypothesis.expected_horizons
    )
    # A date-only task created during the UTC day cannot assume that day's
    # closing bars already exist. Keep the full as-of day outside every label
    # horizon so the final decision has a completed forward label.
    latest_complete_end = as_of_date - timedelta(days=(max_horizon_hours + 23) // 24 + 1)
    resolved_end = end_date or latest_complete_end
    if resolved_end > latest_complete_end:
        raise ValueError("factor_research_label_horizon_incomplete")
    resolved_start = start_date or (resolved_end - timedelta(days=max_history_days - 1))
    if resolved_end < resolved_start or (resolved_end - resolved_start).days >= max_history_days:
        raise ValueError("factor_research_window_out_of_range")
    registry_digest = hypothesis_registry_digest(effective_registry)
    source_input_digest, data_snapshot_id = factor_research_source_identity(
        root,
        start_date=resolved_start,
        end_date=resolved_end,
        hypotheses=executable,
    )
    task_seed = model_content_sha256(
        {
            "schema_version": "quant_lab_factor_research_task_identity.v1",
            "task_type": FACTOR_RESEARCH_TASK_TYPE,
            "as_of_date": as_of_date.isoformat(),
            "start_date": resolved_start.isoformat(),
            "end_date": resolved_end.isoformat(),
            "max_history_days": max_history_days,
            "hypothesis_registry_digest": registry_digest,
            "source_input_digest": source_input_digest,
            "hypotheses": [
                [item.hypothesis_id, item.hypothesis_version, item.definition_digest]
                for item in executable
            ],
            "quant_lab_commit": quant_lab_commit,
            "selected_v5_bundle_id": selected_bundle,
        }
    )[:24]
    task_id = f"factor-research-{task_seed}"
    trials = plan_factor_research_trials(
        hypotheses,
        start_date=resolved_start,
        end_date=resolved_end,
        code_commit=quant_lab_commit,
        data_snapshot_id=data_snapshot_id,
        nas_task_id=task_id,
    )
    ledger = trial_ledger_frame(trials)
    ledger_digest = trial_ledger_digest(ledger)
    snapshot = seal_factor_research_snapshot(
        root,
        queue,
        as_of_date=as_of_date,
        start_date=resolved_start,
        end_date=resolved_end,
        max_history_days=max_history_days,
        selected_v5_bundle_id=selected_bundle,
        effective_registry=effective_registry,
        trial_ledger=ledger,
        hypotheses=executable,
        signing_key=signing_key,
        signature_key_id=signature_key_id,
        quant_lab_commit=quant_lab_commit,
        expected_source_input_digest=source_input_digest,
    )
    if (
        snapshot.hypothesis_registry_digest != registry_digest
        or snapshot.trial_ledger_digest != ledger_digest
        or snapshot.source_input_digest != source_input_digest
    ):
        raise RuntimeError("factor_research_snapshot_control_identity_mismatch")
    existing = find_research_task_directory(queue, task_id)
    if existing is not None:
        task = FactorResearchTask.model_validate_json((existing / "task.json").read_text("utf-8"))
        status = ResearchTaskStatus.model_validate_json(
            (queue / "status" / f"{task_id}.json").read_text("utf-8")
        )
        return task, status

    requested_at = datetime.now(UTC)
    provisional = FactorResearchTask(
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        as_of_date=as_of_date,
        start_date=resolved_start,
        end_date=resolved_end,
        max_history_days=max_history_days,
        hypothesis_ids=snapshot.hypothesis_ids,
        trial_ids=snapshot.trial_ids,
        test_count=snapshot.test_count,
        quant_lab_commit=quant_lab_commit,
        factor_research_schema_version=FACTOR_RESEARCH_SCHEMA_VERSION,
        hypothesis_registry_digest=registry_digest,
        trial_ledger_digest=ledger_digest,
        source_input_digest=source_input_digest,
        selected_v5_bundle_id=selected_bundle,
        snapshot_manifest_sha256=snapshot.manifest_sha256,
        requested_at=requested_at,
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
        signature_key_id=signature_key_id,
        signature="pending",
    )
    task = provisional.model_copy(update={"signature": sign_model(provisional, signing_key)})
    temporary = queue / "pending" / f".{task_id}.{uuid.uuid4().hex}.partial"
    final = queue / "pending" / task_id
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        (temporary / "task.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")
        (temporary / "snapshot_id").write_text(snapshot.snapshot_id + "\n", encoding="ascii")
        for path in (temporary / "task.json", temporary / "snapshot_id"):
            path.chmod(0o660)
        temporary.chmod(0o2770)
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    status = ResearchTaskStatus(
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        task_type=FACTOR_RESEARCH_TASK_TYPE,
        start_date=resolved_start,
        end_date=resolved_end,
        mode=FACTOR_RESEARCH_TASK_TYPE,
        cost_mode="research_point_in_time_p75",
        state=ResearchTaskState.PENDING,
        requested_at=requested_at,
        max_attempts=max_attempts,
        input_bytes=snapshot.total_input_bytes,
        import_status="waiting_for_nas",
    )
    write_research_status(queue, status)
    return task, status


def _coalesce_factor_factory_pending(queue: Path, *, successor_task_id: str) -> None:
    """Keep at most one pending successor without interrupting an active worker."""

    for status_path in sorted((queue / "status").glob("*.json")):
        try:
            status = ResearchTaskStatus.model_validate_json(status_path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if (
            status.task_type != FACTOR_FACTORY_TASK_TYPE
            or status.task_id == successor_task_id
            or status.state is not ResearchTaskState.PENDING
        ):
            continue
        pending = queue / "pending" / status.task_id
        if not pending.is_dir():
            continue
        cancelled = queue / "cancelled" / status.task_id
        if cancelled.exists():
            raise RuntimeError("factor_factory_cancelled_destination_exists")
        os.replace(pending, cancelled)
        now = datetime.now(UTC)
        write_research_status(
            queue,
            status.model_copy(
                update={
                    "state": ResearchTaskState.CANCELLED,
                    "completed_at": now,
                    "import_status": "superseded_before_claim",
                    "last_error": f"superseded_by:{successor_task_id}",
                }
            ),
        )


def create_alpha_factory_task(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    lookback_days: int = 30,
    max_candidates: int = 200,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str,
    selected_v5_bundle_id: str | None = None,
    lease_seconds: int = 7200,
    max_attempts: int = 3,
) -> tuple[AlphaFactoryTask, ResearchTaskStatus]:
    """Create one deterministic, signed Alpha Factory research task."""
    queue = ensure_research_queue_layout(queue_root)
    selected_bundle = selected_v5_bundle_id
    if selected_bundle is None:
        selected_bundle = latest_entry_quality_bundle_id(lake_root)
    selected_bundle = str(selected_bundle or "").strip()
    if not selected_bundle:
        raise RuntimeError("selected_v5_bundle_id_unavailable")
    parameters = AlphaFactoryTaskParameters(
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        max_candidates=max_candidates,
    )
    root = Path(lake_root)
    existing_registry = read_parquet_dataset(root / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET)
    effective_registry = prepare_alpha_factory_control_state(existing_registry)
    registry_digest = alpha_factory_template_registry_digest(effective_registry)
    factor_binding = current_factor_research_generation_binding(
        root,
        alpha_as_of_date=parameters.as_of_date,
    )
    task_seed = model_content_sha256(
        {
            "task_type": ALPHA_FACTORY_TASK_TYPE,
            "factor_generation_id": factor_binding["factor_generation_id"],
            "factor_generation_digest": factor_binding["factor_generation_digest"],
            "hypothesis_registry_digest": factor_binding["hypothesis_registry_digest"],
            "trial_ledger_digest": factor_binding["trial_ledger_digest"],
            "lookback_days": parameters.lookback_days,
            "max_candidates": parameters.max_candidates,
            "template_registry_digest": registry_digest,
            "quant_lab_commit": quant_lab_commit,
            "signature_key_id": signature_key_id,
        }
    )[:24]
    task_id = f"alpha-factory-{task_seed}"
    existing = find_research_task_directory(queue, task_id)
    if existing is not None:
        task = AlphaFactoryTask.model_validate_json((existing / "task.json").read_text("utf-8"))
        status = ResearchTaskStatus.model_validate_json(
            (queue / "status" / f"{task_id}.json").read_text("utf-8")
        )
        return task, status
    snapshot = seal_alpha_factory_snapshot(
        root,
        queue,
        as_of_date=parameters.as_of_date,
        lookback_days=parameters.lookback_days,
        max_candidates=parameters.max_candidates,
        selected_v5_bundle_id=selected_bundle,
        effective_registry=effective_registry,
        signing_key=signing_key,
        signature_key_id=signature_key_id,
        quant_lab_commit=quant_lab_commit,
        factor_generation_binding=factor_binding,
    )
    if snapshot.template_registry_digest != registry_digest:
        raise RuntimeError("alpha_factory_snapshot_registry_digest_mismatch")
    requested_at = datetime.now(UTC)
    provisional = AlphaFactoryTask(
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        as_of_date=parameters.as_of_date,
        lookback_days=parameters.lookback_days,
        max_candidates=parameters.max_candidates,
        quant_lab_commit=quant_lab_commit,
        alpha_factory_schema_version=ALPHA_FACTORY_SCHEMA_VERSION,
        second_stage_schema_version=SECOND_STAGE_ALPHA_FACTORY_SCHEMA_VERSION,
        template_registry_digest=registry_digest,
        **factor_binding,
        selected_v5_bundle_id=selected_bundle,
        snapshot_manifest_sha256=snapshot.manifest_sha256,
        requested_at=requested_at,
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
        signature_key_id=signature_key_id,
        signature="pending",
    )
    task = provisional.model_copy(update={"signature": sign_model(provisional, signing_key)})
    temporary = queue / "pending" / f".{task_id}.{uuid.uuid4().hex}.partial"
    final = queue / "pending" / task_id
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        (temporary / "task.json").write_text(
            task.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (temporary / "snapshot_id").write_text(
            snapshot.snapshot_id + "\n",
            encoding="ascii",
        )
        for path in (temporary / "task.json", temporary / "snapshot_id"):
            path.chmod(0o660)
        temporary.chmod(0o2770)
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    status = ResearchTaskStatus(
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        task_type=ALPHA_FACTORY_TASK_TYPE,
        start_date=parameters.as_of_date,
        end_date=parameters.as_of_date,
        mode=ALPHA_FACTORY_TASK_TYPE,
        cost_mode="research",
        state=ResearchTaskState.PENDING,
        requested_at=requested_at,
        max_attempts=max_attempts,
        input_bytes=snapshot.total_input_bytes,
        import_status="waiting_for_nas",
    )
    write_research_status(queue, status)
    return task, status


def create_entry_quality_history_task(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    start_date: date,
    end_date: date,
    mode: str = "recent_30d",
    cost_mode: str = "conservative",
    window_hours: int = 24,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str,
    selected_v5_bundle_id: str | None = None,
    lease_seconds: int = 3600,
    max_attempts: int = 3,
) -> tuple[ResearchTask, ResearchTaskStatus]:
    queue = ensure_research_queue_layout(queue_root)
    selected_bundle = selected_v5_bundle_id
    if selected_bundle is None:
        selected_bundle = latest_entry_quality_bundle_id(lake_root)
    selected_bundle = str(selected_bundle or "").strip()
    if not selected_bundle:
        raise RuntimeError("selected_v5_bundle_id_unavailable")
    normalized_start, normalized_end, normalized_mode, normalized_cost_mode = (
        normalize_entry_quality_history_request(
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            cost_mode=cost_mode,
        )
    )
    parameters = EntryQualityHistoryTaskParameters(
        start_date=normalized_start,
        end_date=normalized_end,
        mode=normalized_mode,
        cost_mode=normalized_cost_mode,
        window_hours=window_hours,
    )
    snapshot = seal_entry_quality_history_snapshot(
        lake_root,
        queue,
        start_date=parameters.start_date,
        end_date=parameters.end_date,
        selected_v5_bundle_id=selected_bundle,
        signing_key=signing_key,
        signature_key_id=signature_key_id,
        quant_lab_commit=quant_lab_commit,
    )
    task_seed = model_content_sha256(
        {
            "task_type": ENTRY_QUALITY_HISTORY_TASK_TYPE,
            "snapshot_id": snapshot.snapshot_id,
            "start_date": parameters.start_date.isoformat(),
            "end_date": parameters.end_date.isoformat(),
            "mode": parameters.mode,
            "cost_mode": parameters.cost_mode,
            "window_hours": parameters.window_hours,
            "quant_lab_commit": quant_lab_commit,
            "signature_key_id": signature_key_id,
        }
    )[:24]
    task_id = f"entry-quality-history-{task_seed}"
    existing = find_research_task_directory(queue, task_id)
    if existing is not None:
        task = ResearchTask.model_validate_json((existing / "task.json").read_text("utf-8"))
        status = ResearchTaskStatus.model_validate_json(
            (queue / "status" / f"{task_id}.json").read_text("utf-8")
        )
        return task, status

    requested_at = datetime.now(UTC)
    provisional = ResearchTask(
        schema_version=RESEARCH_TASK_SCHEMA,
        task_type=ENTRY_QUALITY_HISTORY_TASK_TYPE,
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        start_date=parameters.start_date,
        end_date=parameters.end_date,
        mode=parameters.mode,
        cost_mode=parameters.cost_mode,
        window_hours=parameters.window_hours,
        quant_lab_commit=quant_lab_commit,
        entry_quality_schema_version=ENTRY_QUALITY_SCHEMA_VERSION,
        selected_v5_bundle_id=selected_bundle,
        snapshot_manifest_sha256=snapshot.manifest_sha256,
        requested_at=requested_at,
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
        signature_key_id=signature_key_id,
        signature="pending",
    )
    task = provisional.model_copy(update={"signature": sign_model(provisional, signing_key)})
    temporary = queue / "pending" / f".{task_id}.{uuid.uuid4().hex}.partial"
    final = queue / "pending" / task_id
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        (temporary / "task.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")
        (temporary / "snapshot_id").write_text(snapshot.snapshot_id + "\n", encoding="ascii")
        for path in (temporary / "task.json", temporary / "snapshot_id"):
            path.chmod(0o660)
        temporary.chmod(0o2770)
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    status = ResearchTaskStatus(
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        start_date=parameters.start_date,
        end_date=parameters.end_date,
        mode=parameters.mode,
        cost_mode=parameters.cost_mode,
        state=ResearchTaskState.PENDING,
        requested_at=requested_at,
        max_attempts=max_attempts,
        input_bytes=snapshot.total_input_bytes,
        import_status="waiting_for_nas",
    )
    write_research_status(queue, status)
    return task, status
