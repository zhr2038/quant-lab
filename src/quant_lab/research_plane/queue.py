from __future__ import annotations

import os
import shutil
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.data.lake import read_parquet_dataset
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
from quant_lab.research.second_stage_alpha_factory import (
    SCHEMA_VERSION as SECOND_STAGE_ALPHA_FACTORY_SCHEMA_VERSION,
)
from quant_lab.research_plane.contracts import (
    ALPHA_FACTORY_TASK_TYPE,
    ENTRY_QUALITY_HISTORY_TASK_TYPE,
    RESEARCH_TASK_SCHEMA,
    AlphaFactoryTask,
    AlphaFactoryTaskParameters,
    EntryQualityHistoryTaskParameters,
    ResearchTask,
    ResearchTaskState,
    ResearchTaskStatus,
)
from quant_lab.research_plane.signatures import model_content_sha256, sign_model
from quant_lab.research_plane.snapshot import (
    seal_alpha_factory_snapshot,
    seal_entry_quality_history_snapshot,
)
from quant_lab.research_plane.status import (
    ensure_research_queue_layout,
    find_research_task_directory,
    write_research_status,
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
    existing_registry = read_parquet_dataset(
        root / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET
    )
    effective_registry = prepare_alpha_factory_control_state(existing_registry)
    registry_digest = alpha_factory_template_registry_digest(effective_registry)
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
    )
    if snapshot.template_registry_digest != registry_digest:
        raise RuntimeError("alpha_factory_snapshot_registry_digest_mismatch")
    task_seed = model_content_sha256(
        {
            "task_type": ALPHA_FACTORY_TASK_TYPE,
            "snapshot_id": snapshot.snapshot_id,
            "as_of_date": parameters.as_of_date.isoformat(),
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
        task = AlphaFactoryTask.model_validate_json(
            (existing / "task.json").read_text("utf-8")
        )
        status = ResearchTaskStatus.model_validate_json(
            (queue / "status" / f"{task_id}.json").read_text("utf-8")
        )
        return task, status

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
