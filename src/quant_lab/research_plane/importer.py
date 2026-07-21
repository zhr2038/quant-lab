from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from quant_lab.data.file_index import build_lake_file_index
from quant_lab.data.lake import count_parquet_rows, read_parquet_dataset
from quant_lab.research.alpha_factory.factory import (
    ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET,
    alpha_factory_template_registry_digest,
    prepare_alpha_factory_control_state,
)
from quant_lab.research.entry_quality import (
    ENTRY_QUALITY_HISTORY_OUTPUT_SPECS,
    EntryQualityHistoryArtifacts,
    publish_entry_quality_history_result,
)
from quant_lab.research.factor_research.registry import (
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
    hypothesis_registry_digest,
    trial_ledger_digest,
)
from quant_lab.research_plane.alpha_factory_publish import (
    publish_alpha_factory_generation,
    verify_alpha_factory_generation,
)
from quant_lab.research_plane.contracts import (
    DEFAULT_FACTOR_FACTORY_MAX_RESULT_BYTES,
    DEFAULT_RESEARCH_MAX_RESULT_BYTES,
    RESEARCH_RECEIPT_ADAPTER,
    RESEARCH_RESULT_ADAPTER,
    RESEARCH_SNAPSHOT_ADAPTER,
    RESEARCH_TASK_ADAPTER,
    AlphaFactoryResultManifest,
    AlphaFactorySnapshotManifest,
    AlphaFactoryTask,
    AlphaFactoryWorkerReceipt,
    FactorFactoryResultManifest,
    FactorFactorySnapshotManifest,
    FactorFactoryTask,
    FactorFactoryWorkerReceipt,
    FactorResearchResultManifest,
    FactorResearchSnapshotManifest,
    FactorResearchTask,
    FactorResearchWorkerReceipt,
    ResearchResultManifest,
    ResearchSnapshotManifest,
    ResearchTask,
    ResearchTaskEnvelope,
    ResearchTaskState,
    ResearchTaskStatus,
    ResearchValidationEvent,
    ResearchWorkerReceipt,
)
from quant_lab.research_plane.factor_factory_publish import (
    FACTOR_FACTORY_DATASETS,
    publish_factor_factory_generation,
    verify_factor_factory_generation,
)
from quant_lab.research_plane.factor_factory_result import (
    validate_factor_factory_result_bundle,
)
from quant_lab.research_plane.factor_research_publish import (
    FACTOR_RESEARCH_GENERATION_POINTER,
    current_factor_research_generation_binding,
    publish_factor_research_generation,
    verify_factor_research_generation,
)
from quant_lab.research_plane.result import (
    ValidatedEntryQualityHistoryResult,
    validate_alpha_factory_result_bundle,
    validate_entry_quality_history_result_bundle,
    validate_factor_research_result_bundle,
    validate_research_task_snapshot,
)
from quant_lab.research_plane.signatures import sha256_file
from quant_lab.research_plane.snapshot_gc import release_snapshot_payload
from quant_lab.research_plane.status import (
    TASK_DIRECTORY_STATES,
    ensure_research_queue_layout,
    read_research_status,
    write_research_status,
)


@dataclass(frozen=True)
class ResearchImportResult:
    task_id: str
    state: str
    generation_id: str
    published_rows: dict[str, int]
    idempotent: bool


@dataclass(frozen=True)
class ResearchImportValidationResult:
    task_id: str
    snapshot_id: str
    generation_id: str
    output_rows: int
    output_bytes: int
    anti_leakage_status: str


def validate_entry_quality_history_result_for_import(
    queue_root: str | Path,
    task_id: str,
    *,
    task_public_key: Ed25519PublicKey,
    worker_public_key: Ed25519PublicKey,
    expected_task_key_id: str,
    expected_worker_key_id: str,
    expected_quant_lab_commit: str,
    max_result_bytes: int = DEFAULT_RESEARCH_MAX_RESULT_BYTES,
    factor_factory_max_result_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_RESULT_BYTES,
    factor_factory_max_value_partition_bytes: int = 256 * 1024**2,
    factor_factory_max_file_count: int = 20_000,
    factor_factory_max_uncompressed_bytes: int = 16 * 1024**3,
) -> ResearchImportValidationResult:
    """Validate one inbox result without changing queue state or publishing Gold."""

    queue = Path(queue_root)
    inbox = queue / "results" / "inbox" / task_id
    running = queue / "running" / task_id
    if not inbox.is_dir():
        raise FileNotFoundError(f"research result inbox missing: {task_id}")
    if not running.is_dir():
        raise ValueError("research_result_task_not_running")
    task = RESEARCH_TASK_ADAPTER.validate_json((running / "task.json").read_text("utf-8"))
    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = RESEARCH_SNAPSHOT_ADAPTER.validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    validate_research_task_snapshot(
        task,
        snapshot,
        task_public_key=task_public_key,
        expected_key_id=expected_task_key_id,
        expected_quant_lab_commit=expected_quant_lab_commit,
        snapshot_root=snapshot_root,
    )
    if _task_is_superseded(queue, task):
        raise ValueError("research_result_superseded_by_newer_snapshot")
    manifest = _load_result_manifest(inbox)
    receipt = RESEARCH_RECEIPT_ADAPTER.validate_json((inbox / "receipt.json").read_text("utf-8"))
    if isinstance(task, FactorFactoryTask):
        if not all(
            (
                isinstance(snapshot, FactorFactorySnapshotManifest),
                isinstance(manifest, FactorFactoryResultManifest),
                isinstance(receipt, FactorFactoryWorkerReceipt),
            )
        ):
            raise ValueError("research_result_task_type_mismatch")
        validated_factory = validate_factor_factory_result_bundle(
            inbox,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_public_key,
            expected_worker_key_id=expected_worker_key_id,
            max_result_bytes=factor_factory_max_result_bytes,
            max_value_partition_bytes=factor_factory_max_value_partition_bytes,
            max_file_count=factor_factory_max_file_count,
            max_uncompressed_bytes=factor_factory_max_uncompressed_bytes,
        )
        output_rows = sum(item.row_count for item in validated_factory.manifest.outputs) + sum(
            item.row_count for item in validated_factory.manifest.value_partitions
        )
    elif isinstance(task, AlphaFactoryTask):
        if not all(
            (
                isinstance(snapshot, AlphaFactorySnapshotManifest),
                isinstance(manifest, AlphaFactoryResultManifest),
                isinstance(receipt, AlphaFactoryWorkerReceipt),
            )
        ):
            raise ValueError("research_result_task_type_mismatch")
        validated_alpha = validate_alpha_factory_result_bundle(
            inbox,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_public_key,
            expected_worker_key_id=expected_worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        output_rows = sum(item.row_count for item in validated_alpha.manifest.outputs)
    elif isinstance(task, FactorResearchTask):
        if not all(
            (
                isinstance(snapshot, FactorResearchSnapshotManifest),
                isinstance(manifest, FactorResearchResultManifest),
                isinstance(receipt, FactorResearchWorkerReceipt),
            )
        ):
            raise ValueError("research_result_task_type_mismatch")
        validated_factor = validate_factor_research_result_bundle(
            inbox,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_public_key,
            expected_worker_key_id=expected_worker_key_id,
            max_result_bytes=max_result_bytes,
            snapshot_root=snapshot_root,
        )
        output_rows = sum(item.row_count for item in validated_factor.manifest.outputs)
    else:
        if not all(
            (
                isinstance(snapshot, ResearchSnapshotManifest),
                isinstance(manifest, ResearchResultManifest),
                isinstance(receipt, ResearchWorkerReceipt),
            )
        ):
            raise ValueError("research_result_task_type_mismatch")
        validated_entry = validate_entry_quality_history_result_bundle(
            inbox,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_public_key,
            expected_worker_key_id=expected_worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        output_rows = sum(frame.height for frame in validated_entry.frames.values())
    return ResearchImportValidationResult(
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        generation_id=manifest.generation_id,
        output_rows=output_rows,
        output_bytes=manifest.output_bytes,
        anti_leakage_status=manifest.anti_leakage_status,
    )


def validate_pending_entry_quality_history_results(
    queue_root: str | Path,
    *,
    task_public_key: Ed25519PublicKey,
    worker_public_key: Ed25519PublicKey,
    expected_task_key_id: str,
    expected_worker_key_id: str,
    expected_quant_lab_commit: str,
    max_result_bytes: int = DEFAULT_RESEARCH_MAX_RESULT_BYTES,
    factor_factory_max_result_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_RESULT_BYTES,
    factor_factory_max_value_partition_bytes: int = 256 * 1024**2,
    factor_factory_max_file_count: int = 20_000,
    factor_factory_max_uncompressed_bytes: int = 16 * 1024**3,
) -> list[ResearchImportValidationResult]:
    """Validate every current inbox result without creating or moving queue files."""

    inbox = Path(queue_root) / "results" / "inbox"
    if not inbox.is_dir():
        return []
    results: list[ResearchImportValidationResult] = []
    for candidate in sorted(inbox.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        results.append(
            validate_entry_quality_history_result_for_import(
                queue_root,
                candidate.name,
                task_public_key=task_public_key,
                worker_public_key=worker_public_key,
                expected_task_key_id=expected_task_key_id,
                expected_worker_key_id=expected_worker_key_id,
                expected_quant_lab_commit=expected_quant_lab_commit,
                max_result_bytes=max_result_bytes,
                factor_factory_max_result_bytes=factor_factory_max_result_bytes,
                factor_factory_max_value_partition_bytes=(factor_factory_max_value_partition_bytes),
                factor_factory_max_file_count=factor_factory_max_file_count,
                factor_factory_max_uncompressed_bytes=(factor_factory_max_uncompressed_bytes),
            )
        )
    return results


def import_entry_quality_history_result(
    lake_root: str | Path,
    queue_root: str | Path,
    task_id: str,
    *,
    task_public_key: Ed25519PublicKey,
    worker_public_key: Ed25519PublicKey,
    expected_task_key_id: str,
    expected_worker_key_id: str,
    expected_quant_lab_commit: str,
    max_result_bytes: int = DEFAULT_RESEARCH_MAX_RESULT_BYTES,
    factor_factory_max_result_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_RESULT_BYTES,
    factor_factory_max_value_partition_bytes: int = 256 * 1024**2,
    factor_factory_max_file_count: int = 20_000,
    factor_factory_max_uncompressed_bytes: int = 16 * 1024**3,
) -> ResearchImportResult:
    queue = ensure_research_queue_layout(queue_root)
    task_envelope = _load_task_envelope(queue, task_id)
    if isinstance(task_envelope, FactorFactoryTask):
        return _import_factor_factory_result(
            lake_root,
            queue,
            task_id,
            task=task_envelope,
            task_public_key=task_public_key,
            worker_public_key=worker_public_key,
            expected_task_key_id=expected_task_key_id,
            expected_worker_key_id=expected_worker_key_id,
            expected_quant_lab_commit=expected_quant_lab_commit,
            max_result_bytes=factor_factory_max_result_bytes,
            max_value_partition_bytes=factor_factory_max_value_partition_bytes,
            max_file_count=factor_factory_max_file_count,
            max_uncompressed_bytes=factor_factory_max_uncompressed_bytes,
        )
    if isinstance(task_envelope, AlphaFactoryTask):
        return _import_alpha_factory_result(
            lake_root,
            queue,
            task_id,
            task=task_envelope,
            task_public_key=task_public_key,
            worker_public_key=worker_public_key,
            expected_task_key_id=expected_task_key_id,
            expected_worker_key_id=expected_worker_key_id,
            expected_quant_lab_commit=expected_quant_lab_commit,
            max_result_bytes=max_result_bytes,
        )
    if isinstance(task_envelope, FactorResearchTask):
        return _import_factor_research_result(
            lake_root,
            queue,
            task_id,
            task=task_envelope,
            task_public_key=task_public_key,
            worker_public_key=worker_public_key,
            expected_task_key_id=expected_task_key_id,
            expected_worker_key_id=expected_worker_key_id,
            expected_quant_lab_commit=expected_quant_lab_commit,
            max_result_bytes=max_result_bytes,
        )
    lake = Path(lake_root)
    inbox = queue / "results" / "inbox" / task_id
    imported = queue / "results" / "imported" / task_id
    if imported.is_dir() and not inbox.exists():
        manifest = _load_result_manifest(imported)
        _verify_published_generation(
            lake,
            manifest,
            _published_row_counts(lake, manifest.generation_id),
        )
        _finalize_committed_import(queue, task_id, manifest)
        return ResearchImportResult(
            task_id=task_id,
            state="completed",
            generation_id=manifest.generation_id,
            published_rows=_published_row_counts(lake, manifest.generation_id),
            idempotent=True,
        )
    if imported.is_dir() and inbox.is_dir():
        if sha256_file(imported / "manifest.json") != sha256_file(inbox / "manifest.json"):
            raise ValueError("research_result_duplicate_payload_conflict")
        shutil.rmtree(inbox)
        manifest = _load_result_manifest(imported)
        _verify_published_generation(
            lake,
            manifest,
            _published_row_counts(lake, manifest.generation_id),
        )
        _finalize_committed_import(queue, task_id, manifest)
        return ResearchImportResult(
            task_id=task_id,
            state="completed",
            generation_id=manifest.generation_id,
            published_rows=_published_row_counts(lake, manifest.generation_id),
            idempotent=True,
        )
    if not inbox.is_dir():
        raise FileNotFoundError(f"research result inbox missing: {task_id}")

    running = queue / "running" / task_id
    if not running.is_dir():
        raise ValueError("research_result_task_not_running")
    task = ResearchTask.model_validate_json((running / "task.json").read_text("utf-8"))
    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = ResearchSnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    status = read_research_status(queue, task_id) or _initial_import_status(task, snapshot)
    publication_committed = False
    strict_validation_passed = False
    manifest: ResearchResultManifest | None = None
    try:
        status = status.model_copy(
            update={
                "state": ResearchTaskState.VALIDATING_ON_CLOUD,
                "heartbeat_at": datetime.now(UTC),
                "import_status": "validating",
                "last_error": None,
            }
        )
        write_research_status(queue, status)
        (queue / "lease" / f"{task_id}.json").unlink(missing_ok=True)
        validate_research_task_snapshot(
            task,
            snapshot,
            task_public_key=task_public_key,
            expected_key_id=expected_task_key_id,
            expected_quant_lab_commit=expected_quant_lab_commit,
            snapshot_root=snapshot_root,
        )
        if _task_is_superseded(queue, task):
            raise ValueError("research_result_superseded_by_newer_snapshot")
        manifest = _load_result_manifest(inbox)
        receipt = ResearchWorkerReceipt.model_validate_json(
            (inbox / "receipt.json").read_text("utf-8")
        )
        validated = validate_entry_quality_history_result_bundle(
            inbox,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_public_key,
            expected_worker_key_id=expected_worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        strict_validation_passed = True
        _write_validation_event(
            queue,
            task_id,
            "strict_result_validation",
            "PASS",
            "25 checks passed",
        )
        artifacts = _artifacts_from_validated(validated)
        status = status.model_copy(
            update={
                "state": ResearchTaskState.PUBLISHING,
                "heartbeat_at": datetime.now(UTC),
                "import_status": "publishing",
                "output_rows": receipt.output_rows,
                "output_bytes": manifest.output_bytes,
                "peak_rss_bytes": manifest.peak_rss_bytes,
                "compute_duration_seconds": manifest.compute_duration_seconds,
                "anti_leakage_status": manifest.anti_leakage_status,
            }
        )
        write_research_status(queue, status)
        if _generation_is_published(lake, manifest):
            published_rows = _published_row_counts(lake, manifest.generation_id)
        else:
            published_rows = publish_entry_quality_history_result(
                lake,
                artifacts,
                generation_id=manifest.generation_id,
                snapshot_id=manifest.snapshot_id,
                task_id=manifest.task_id,
                reports=validated.reports,
            )
        publication_committed = True
        _verify_published_generation(lake, manifest, published_rows)
        _finalize_committed_import(queue, task_id, manifest, status=status)
        return ResearchImportResult(
            task_id=task_id,
            state=ResearchTaskState.COMPLETED.value,
            generation_id=manifest.generation_id,
            published_rows=published_rows,
            idempotent=False,
        )
    except Exception as exc:
        if publication_committed and manifest is not None:
            _record_finalize_pending(queue, status, manifest.generation_id, exc)
        elif strict_validation_passed and manifest is not None:
            _record_publish_retry(queue, status, manifest.generation_id, exc)
            return ResearchImportResult(
                task_id=task_id,
                state="publish_retry_pending",
                generation_id=manifest.generation_id,
                published_rows={},
                idempotent=False,
            )
        else:
            _reject_result(queue, task, status, inbox, running, exc)
        raise


def _import_alpha_factory_result(
    lake_root: str | Path,
    queue: Path,
    task_id: str,
    *,
    task: AlphaFactoryTask,
    task_public_key: Ed25519PublicKey,
    worker_public_key: Ed25519PublicKey,
    expected_task_key_id: str,
    expected_worker_key_id: str,
    expected_quant_lab_commit: str,
    max_result_bytes: int,
) -> ResearchImportResult:
    lake = Path(lake_root)
    inbox = queue / "results" / "inbox" / task_id
    imported = queue / "results" / "imported" / task_id
    if imported.is_dir() and inbox.is_dir():
        if sha256_file(imported / "manifest.json") != sha256_file(inbox / "manifest.json"):
            raise ValueError("research_result_duplicate_payload_conflict")
        shutil.rmtree(inbox)
    if imported.is_dir() and not inbox.exists():
        manifest = _load_result_manifest(imported)
        if not isinstance(manifest, AlphaFactoryResultManifest):
            raise ValueError("research_result_task_type_mismatch")
        published_rows = verify_alpha_factory_generation(
            lake,
            manifest.generation_id,
        )
        _finalize_committed_import(queue, task_id, manifest)
        return ResearchImportResult(
            task_id=task_id,
            state="completed",
            generation_id=manifest.generation_id,
            published_rows=published_rows,
            idempotent=True,
        )
    if not inbox.is_dir():
        raise FileNotFoundError(f"research result inbox missing: {task_id}")
    running = queue / "running" / task_id
    if not running.is_dir():
        raise ValueError("research_result_task_not_running")
    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = RESEARCH_SNAPSHOT_ADAPTER.validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    if not isinstance(snapshot, AlphaFactorySnapshotManifest):
        raise ValueError("research_result_task_type_mismatch")
    status = read_research_status(queue, task_id) or _initial_import_status(task, snapshot)
    publication_committed = False
    strict_validation_passed = False
    manifest: AlphaFactoryResultManifest | None = None
    try:
        status = status.model_copy(
            update={
                "state": ResearchTaskState.VALIDATING_ON_CLOUD,
                "heartbeat_at": datetime.now(UTC),
                "import_status": "validating",
                "last_error": None,
            }
        )
        write_research_status(queue, status)
        (queue / "lease" / f"{task_id}.json").unlink(missing_ok=True)
        validate_research_task_snapshot(
            task,
            snapshot,
            task_public_key=task_public_key,
            expected_key_id=expected_task_key_id,
            expected_quant_lab_commit=expected_quant_lab_commit,
            snapshot_root=snapshot_root,
        )
        if _task_is_superseded(queue, task):
            raise ValueError("research_result_superseded_by_newer_snapshot")
        current_registry = prepare_alpha_factory_control_state(
            read_parquet_dataset(lake / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET)
        )
        if (
            alpha_factory_template_registry_digest(current_registry)
            != task.template_registry_digest
        ):
            raise ValueError("alpha_factory_result_superseded_by_registry_change")
        current_factor_binding = current_factor_research_generation_binding(
            lake,
            alpha_as_of_date=task.as_of_date,
        )
        for field, current_value in current_factor_binding.items():
            if getattr(task, field) != current_value:
                raise ValueError("alpha_factory_result_superseded_by_factor_generation")
        parsed_manifest = _load_result_manifest(inbox)
        parsed_receipt = RESEARCH_RECEIPT_ADAPTER.validate_json(
            (inbox / "receipt.json").read_text("utf-8")
        )
        if not isinstance(parsed_manifest, AlphaFactoryResultManifest) or not isinstance(
            parsed_receipt,
            AlphaFactoryWorkerReceipt,
        ):
            raise ValueError("research_result_task_type_mismatch")
        manifest = parsed_manifest
        validated = validate_alpha_factory_result_bundle(
            inbox,
            manifest=manifest,
            receipt=parsed_receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_public_key,
            expected_worker_key_id=expected_worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        strict_validation_passed = True
        anti_leakage_check_count = len(
            json.loads(validated.reports["alpha_factory_anti_leakage.json"].decode("utf-8"))[
                "checks"
            ]
        )
        _write_validation_event(
            queue,
            task_id,
            "strict_alpha_factory_result_validation",
            "PASS",
            f"{anti_leakage_check_count} checks passed; promotion and evidence remain cloud-owned",
        )
        status = status.model_copy(
            update={
                "state": ResearchTaskState.PUBLISHING,
                "heartbeat_at": datetime.now(UTC),
                "import_status": "publishing",
                "output_rows": parsed_receipt.output_rows,
                "output_bytes": manifest.output_bytes,
                "peak_rss_bytes": manifest.peak_rss_bytes,
                "compute_duration_seconds": manifest.compute_duration_seconds,
                "anti_leakage_status": manifest.anti_leakage_status,
            }
        )
        write_research_status(queue, status)
        try:
            published_rows = verify_alpha_factory_generation(
                lake,
                manifest.generation_id,
            )
        except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError):
            published_rows = publish_alpha_factory_generation(lake, validated)
        publication_committed = True
        verify_alpha_factory_generation(lake, manifest.generation_id, published_rows)
        _finalize_committed_import(queue, task_id, manifest, status=status)
        return ResearchImportResult(
            task_id=task_id,
            state=ResearchTaskState.COMPLETED.value,
            generation_id=manifest.generation_id,
            published_rows=published_rows,
            idempotent=False,
        )
    except Exception as exc:
        if publication_committed and manifest is not None:
            _record_finalize_pending(queue, status, manifest.generation_id, exc)
        elif strict_validation_passed and manifest is not None:
            _record_publish_retry(queue, status, manifest.generation_id, exc)
            return ResearchImportResult(
                task_id=task_id,
                state="publish_retry_pending",
                generation_id=manifest.generation_id,
                published_rows={},
                idempotent=False,
            )
        else:
            _reject_result(queue, task, status, inbox, running, exc)
        raise


def _import_factor_factory_result(
    lake_root: str | Path,
    queue: Path,
    task_id: str,
    *,
    task: FactorFactoryTask,
    task_public_key: Ed25519PublicKey,
    worker_public_key: Ed25519PublicKey,
    expected_task_key_id: str,
    expected_worker_key_id: str,
    expected_quant_lab_commit: str,
    max_result_bytes: int,
    max_value_partition_bytes: int,
    max_file_count: int,
    max_uncompressed_bytes: int,
) -> ResearchImportResult:
    lake = Path(lake_root)
    inbox = queue / "results" / "inbox" / task_id
    imported = queue / "results" / "imported" / task_id
    if imported.is_dir() and inbox.is_dir():
        if sha256_file(imported / "manifest.json") != sha256_file(inbox / "manifest.json"):
            raise ValueError("research_result_duplicate_payload_conflict")
        shutil.rmtree(inbox)
    if imported.is_dir() and not inbox.exists():
        manifest = _load_result_manifest(imported)
        if not isinstance(manifest, FactorFactoryResultManifest):
            raise ValueError("research_result_task_type_mismatch")
        if manifest.completed_no_update:
            published_rows = {
                name: count_parquet_rows(lake / path)
                for name, path in FACTOR_FACTORY_DATASETS.items()
            }
        else:
            published_rows = verify_factor_factory_generation(
                lake,
                manifest.generation_id,
            )
        _finalize_committed_import(queue, task_id, manifest)
        return ResearchImportResult(
            task_id=task_id,
            state="completed",
            generation_id=manifest.generation_id,
            published_rows=published_rows,
            idempotent=True,
        )
    if not inbox.is_dir():
        raise FileNotFoundError(f"research result inbox missing: {task_id}")
    running = queue / "running" / task_id
    if not running.is_dir():
        raise ValueError("research_result_task_not_running")
    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = RESEARCH_SNAPSHOT_ADAPTER.validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    if not isinstance(snapshot, FactorFactorySnapshotManifest):
        raise ValueError("research_result_task_type_mismatch")
    status = read_research_status(queue, task_id) or _initial_import_status(task, snapshot)
    publication_committed = False
    strict_validation_passed = False
    manifest: FactorFactoryResultManifest | None = None
    try:
        status = status.model_copy(
            update={
                "state": ResearchTaskState.VALIDATING_ON_CLOUD,
                "heartbeat_at": datetime.now(UTC),
                "import_status": "validating",
                "last_error": None,
            }
        )
        write_research_status(queue, status)
        (queue / "lease" / f"{task_id}.json").unlink(missing_ok=True)
        validate_research_task_snapshot(
            task,
            snapshot,
            task_public_key=task_public_key,
            expected_key_id=expected_task_key_id,
            expected_quant_lab_commit=expected_quant_lab_commit,
            snapshot_root=snapshot_root,
        )
        if _task_is_superseded(queue, task):
            raise ValueError("research_result_superseded_by_newer_snapshot")
        parsed_manifest = _load_result_manifest(inbox)
        parsed_receipt = RESEARCH_RECEIPT_ADAPTER.validate_json(
            (inbox / "receipt.json").read_text("utf-8")
        )
        if not isinstance(parsed_manifest, FactorFactoryResultManifest) or not isinstance(
            parsed_receipt,
            FactorFactoryWorkerReceipt,
        ):
            raise ValueError("research_result_task_type_mismatch")
        manifest = parsed_manifest
        validated = validate_factor_factory_result_bundle(
            inbox,
            manifest=manifest,
            receipt=parsed_receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_public_key,
            expected_worker_key_id=expected_worker_key_id,
            max_result_bytes=max_result_bytes,
            max_value_partition_bytes=max_value_partition_bytes,
            max_file_count=max_file_count,
            max_uncompressed_bytes=max_uncompressed_bytes,
        )
        strict_validation_passed = True
        anti_leakage_check_count = len(
            json.loads(validated.reports["factor_factory_anti_leakage.json"].decode("utf-8"))[
                "checks"
            ]
        )
        _write_validation_event(
            queue,
            task_id,
            "strict_factor_factory_result_validation",
            "PASS",
            f"{anti_leakage_check_count} checks passed; candidate and Gold publication "
            "remain cloud-owned",
        )
        status = status.model_copy(
            update={
                "state": ResearchTaskState.PUBLISHING,
                "heartbeat_at": datetime.now(UTC),
                "import_status": "publishing",
                "output_rows": parsed_receipt.output_rows,
                "output_bytes": manifest.output_bytes,
                "peak_rss_bytes": manifest.peak_rss_bytes,
                "compute_duration_seconds": manifest.compute_duration_seconds,
                "anti_leakage_status": manifest.anti_leakage_status,
            }
        )
        write_research_status(queue, status)
        if manifest.completed_no_update:
            publish_result = publish_factor_factory_generation(lake, validated)
            published_rows = dict(publish_result["row_counts"])
        else:
            try:
                published_rows = verify_factor_factory_generation(
                    lake,
                    manifest.generation_id,
                )
            except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError):
                publish_result = publish_factor_factory_generation(lake, validated)
                published_rows = dict(publish_result["row_counts"])
            verify_factor_factory_generation(
                lake,
                manifest.generation_id,
                expected_rows=published_rows,
            )
            build_lake_file_index(lake, FACTOR_FACTORY_DATASETS.values())
        publication_committed = True
        _finalize_committed_import(queue, task_id, manifest, status=status)
        return ResearchImportResult(
            task_id=task_id,
            state=ResearchTaskState.COMPLETED.value,
            generation_id=manifest.generation_id,
            published_rows=published_rows,
            idempotent=False,
        )
    except Exception as exc:
        if publication_committed and manifest is not None:
            _record_finalize_pending(queue, status, manifest.generation_id, exc)
        elif strict_validation_passed and manifest is not None:
            _record_publish_retry(queue, status, manifest.generation_id, exc)
            return ResearchImportResult(
                task_id=task_id,
                state="publish_retry_pending",
                generation_id=manifest.generation_id,
                published_rows={},
                idempotent=False,
            )
        else:
            _reject_result(queue, task, status, inbox, running, exc)
            raise


def _import_factor_research_result(
    lake_root: str | Path,
    queue: Path,
    task_id: str,
    *,
    task: FactorResearchTask,
    task_public_key: Ed25519PublicKey,
    worker_public_key: Ed25519PublicKey,
    expected_task_key_id: str,
    expected_worker_key_id: str,
    expected_quant_lab_commit: str,
    max_result_bytes: int,
) -> ResearchImportResult:
    lake = Path(lake_root)
    inbox = queue / "results" / "inbox" / task_id
    imported = queue / "results" / "imported" / task_id
    if imported.is_dir() and inbox.is_dir():
        if sha256_file(imported / "manifest.json") != sha256_file(inbox / "manifest.json"):
            raise ValueError("research_result_duplicate_payload_conflict")
        shutil.rmtree(inbox)
    if imported.is_dir() and not inbox.exists():
        manifest = _load_result_manifest(imported)
        if not isinstance(manifest, FactorResearchResultManifest):
            raise ValueError("research_result_task_type_mismatch")
        published_rows = verify_factor_research_generation(lake, manifest.generation_id)
        _finalize_committed_import(queue, task_id, manifest)
        return ResearchImportResult(
            task_id=task_id,
            state="completed",
            generation_id=manifest.generation_id,
            published_rows=published_rows,
            idempotent=True,
        )
    if not inbox.is_dir():
        raise FileNotFoundError(f"research result inbox missing: {task_id}")
    running = queue / "running" / task_id
    if not running.is_dir():
        raise ValueError("research_result_task_not_running")
    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = RESEARCH_SNAPSHOT_ADAPTER.validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    if not isinstance(snapshot, FactorResearchSnapshotManifest):
        raise ValueError("research_result_task_type_mismatch")
    status = read_research_status(queue, task_id) or _initial_import_status(task, snapshot)
    publication_committed = False
    strict_validation_passed = False
    manifest: FactorResearchResultManifest | None = None
    try:
        status = status.model_copy(
            update={
                "state": ResearchTaskState.VALIDATING_ON_CLOUD,
                "heartbeat_at": datetime.now(UTC),
                "import_status": "validating",
                "last_error": None,
            }
        )
        write_research_status(queue, status)
        (queue / "lease" / f"{task_id}.json").unlink(missing_ok=True)
        validate_research_task_snapshot(
            task,
            snapshot,
            task_public_key=task_public_key,
            expected_key_id=expected_task_key_id,
            expected_quant_lab_commit=expected_quant_lab_commit,
            snapshot_root=snapshot_root,
        )
        if _task_is_superseded(queue, task):
            raise ValueError("research_result_superseded_by_newer_snapshot")
        parsed_manifest = _load_result_manifest(inbox)
        parsed_receipt = RESEARCH_RECEIPT_ADAPTER.validate_json(
            (inbox / "receipt.json").read_text("utf-8")
        )
        if not isinstance(parsed_manifest, FactorResearchResultManifest) or not isinstance(
            parsed_receipt,
            FactorResearchWorkerReceipt,
        ):
            raise ValueError("research_result_task_type_mismatch")
        manifest = parsed_manifest
        validated = validate_factor_research_result_bundle(
            inbox,
            manifest=manifest,
            receipt=parsed_receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_public_key,
            expected_worker_key_id=expected_worker_key_id,
            max_result_bytes=max_result_bytes,
            snapshot_root=snapshot_root,
        )
        strict_validation_passed = True
        anti_leakage_check_count = len(
            json.loads(validated.reports["factor_research_anti_leakage.json"].decode("utf-8"))[
                "checks"
            ]
        )
        _write_validation_event(
            queue,
            task_id,
            "strict_factor_research_result_validation",
            "PASS",
            f"{anti_leakage_check_count} checks passed; "
            "hypothesis and trial state remain cloud-owned",
        )
        status = status.model_copy(
            update={
                "state": ResearchTaskState.PUBLISHING,
                "heartbeat_at": datetime.now(UTC),
                "import_status": "publishing",
                "output_rows": parsed_receipt.output_rows,
                "output_bytes": manifest.output_bytes,
                "peak_rss_bytes": manifest.peak_rss_bytes,
                "compute_duration_seconds": manifest.compute_duration_seconds,
                "anti_leakage_status": manifest.anti_leakage_status,
            }
        )
        write_research_status(queue, status)
        try:
            published_rows = verify_factor_research_generation(lake, manifest.generation_id)
        except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            current_registry = read_parquet_dataset(lake / RESEARCH_HYPOTHESIS_REGISTRY_DATASET)
            has_published_generation = (lake / FACTOR_RESEARCH_GENERATION_POINTER).is_file()
            if (
                has_published_generation or not current_registry.is_empty()
            ) and hypothesis_registry_digest(current_registry) != task.hypothesis_registry_digest:
                raise ValueError(
                    "factor_research_result_superseded_by_hypothesis_registry_change"
                ) from exc
            if (
                _snapshot_factor_research_trial_ledger_digest(snapshot_root, task)
                != task.trial_ledger_digest
            ):
                raise ValueError(
                    "factor_research_result_superseded_by_trial_ledger_change"
                ) from exc
            published_rows = publish_factor_research_generation(
                lake,
                validated,
                snapshot_root=snapshot_root,
            )
        publication_committed = True
        verify_factor_research_generation(lake, manifest.generation_id, published_rows)
        _finalize_committed_import(queue, task_id, manifest, status=status)
        return ResearchImportResult(
            task_id=task_id,
            state=ResearchTaskState.COMPLETED.value,
            generation_id=manifest.generation_id,
            published_rows=published_rows,
            idempotent=False,
        )
    except Exception as exc:
        if publication_committed and manifest is not None:
            _record_finalize_pending(queue, status, manifest.generation_id, exc)
        elif strict_validation_passed and manifest is not None:
            _record_publish_retry(queue, status, manifest.generation_id, exc)
            return ResearchImportResult(
                task_id=task_id,
                state="publish_retry_pending",
                generation_id=manifest.generation_id,
                published_rows={},
                idempotent=False,
            )
        else:
            _reject_result(queue, task, status, inbox, running, exc)
            raise


def _snapshot_factor_research_trial_ledger_digest(
    snapshot_root: str | Path,
    task: FactorResearchTask,
) -> str:
    """Bind publication to the immutable, signed trial plan for this task."""
    ledger = read_parquet_dataset(Path(snapshot_root) / "files" / RESEARCH_TRIAL_LEDGER_DATASET)
    if ledger.is_empty() or "trial_id" not in ledger.columns:
        raise ValueError("factor_research_current_trial_ledger_missing")
    expected = set(task.trial_ids)
    current = ledger.filter(ledger.get_column("trial_id").is_in(sorted(expected)))
    observed = set(current.get_column("trial_id").to_list())
    if observed != expected:
        raise ValueError("factor_research_current_trial_ledger_incomplete")
    return trial_ledger_digest(current)


def import_pending_entry_quality_history_results(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    task_public_key: Ed25519PublicKey,
    worker_public_key: Ed25519PublicKey,
    expected_task_key_id: str,
    expected_worker_key_id: str,
    expected_quant_lab_commit: str,
    max_result_bytes: int = DEFAULT_RESEARCH_MAX_RESULT_BYTES,
    factor_factory_max_result_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_RESULT_BYTES,
    factor_factory_max_value_partition_bytes: int = 256 * 1024**2,
    factor_factory_max_file_count: int = 20_000,
    factor_factory_max_uncompressed_bytes: int = 16 * 1024**3,
) -> list[ResearchImportResult]:
    queue = ensure_research_queue_layout(queue_root)
    results: list[ResearchImportResult] = []
    for candidate in sorted((queue / "results" / "inbox").iterdir()):
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        results.append(
            import_entry_quality_history_result(
                lake_root,
                queue,
                candidate.name,
                task_public_key=task_public_key,
                worker_public_key=worker_public_key,
                expected_task_key_id=expected_task_key_id,
                expected_worker_key_id=expected_worker_key_id,
                expected_quant_lab_commit=expected_quant_lab_commit,
                max_result_bytes=max_result_bytes,
                factor_factory_max_result_bytes=factor_factory_max_result_bytes,
                factor_factory_max_value_partition_bytes=(factor_factory_max_value_partition_bytes),
                factor_factory_max_file_count=factor_factory_max_file_count,
                factor_factory_max_uncompressed_bytes=(factor_factory_max_uncompressed_bytes),
            )
        )
    return results


def _artifacts_from_validated(
    validated: ValidatedEntryQualityHistoryResult,
) -> EntryQualityHistoryArtifacts:
    frames = validated.frames
    manifest = validated.manifest
    return EntryQualityHistoryArtifacts(
        start_date=manifest.start_date,
        end_date=manifest.end_date,
        mode=manifest.mode,
        cost_mode=manifest.cost_mode,
        generated_at=manifest.generated_at,
        generated_from_bundle_id=manifest.selected_v5_bundle_id,
        missed_low_audit=frames["v5_entry_quality_history_missed_low_audit"],
        missed_low_by_symbol=frames["v5_entry_quality_history_missed_low_by_symbol"],
        missed_low_by_entry_reason=frames["v5_entry_quality_history_missed_low_by_entry_reason"],
        late_entry_chase_shadow=frames["v5_entry_quality_history_late_entry_chase_shadow"],
        late_entry_threshold_sensitivity=frames[
            "v5_entry_quality_history_late_entry_chase_threshold_sensitivity"
        ],
        pullback_reversal_shadow=frames["v5_entry_quality_history_pullback_reversal_shadow"],
        pullback_by_symbol=frames["v5_entry_quality_history_pullback_by_symbol"],
        pullback_by_regime=frames["v5_entry_quality_history_pullback_by_regime"],
        pullback_by_horizon=frames["v5_entry_quality_history_pullback_by_horizon"],
        anti_leakage_check=frames["v5_entry_quality_history_anti_leakage_check"],
        metrics=frames["v5_entry_quality_history_metrics"],
        reports=validated.reports,
        warnings=tuple(manifest.warnings),
    )


def _task_is_superseded(queue: Path, task: ResearchTaskEnvelope) -> bool:
    for state in TASK_DIRECTORY_STATES:
        if state in {"failed", "expired", "cancelled"}:
            continue
        directory = queue / state
        for candidate in directory.glob("*/task.json"):
            try:
                other = RESEARCH_TASK_ADAPTER.validate_json(candidate.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if other.task_id == task.task_id or other.task_type != task.task_type:
                continue
            if isinstance(task, AlphaFactoryTask) and isinstance(other, AlphaFactoryTask):
                same_scope = other.as_of_date == task.as_of_date
            elif isinstance(task, FactorFactoryTask) and isinstance(other, FactorFactoryTask):
                same_scope = (
                    other.feature_set,
                    other.feature_version,
                    other.factor_version,
                    other.timeframe,
                ) == (
                    task.feature_set,
                    task.feature_version,
                    task.factor_version,
                    task.timeframe,
                )
            elif isinstance(task, FactorResearchTask) and isinstance(other, FactorResearchTask):
                same_scope = bool(set(other.hypothesis_ids).intersection(task.hypothesis_ids))
            elif isinstance(task, ResearchTask) and isinstance(other, ResearchTask):
                same_scope = other.mode == task.mode and other.cost_mode == task.cost_mode
            else:
                same_scope = False
            if same_scope and other.requested_at > task.requested_at:
                return True
    return False


def _verify_published_generation(
    lake: Path,
    manifest: ResearchResultManifest,
    published_rows: dict[str, int],
) -> None:
    pointer = json.loads(
        (lake / "gold" / "entry_quality_history_generation.json").read_text("utf-8")
    )
    if pointer.get("generation_id") != manifest.generation_id:
        raise RuntimeError("research_import_generation_pointer_mismatch")
    if pointer.get("snapshot_id") != manifest.snapshot_id:
        raise RuntimeError("research_import_snapshot_pointer_mismatch")
    if pointer.get("row_counts") != published_rows:
        raise RuntimeError("research_import_row_count_pointer_mismatch")
    for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS:
        metadata_path = lake / spec.relative_path / "_research_generation.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"research_import_generation_metadata_missing:{spec.dataset_name}")
        metadata = json.loads(metadata_path.read_text("utf-8"))
        if (
            metadata.get("generation_id") != manifest.generation_id
            or metadata.get("snapshot_id") != manifest.snapshot_id
            or metadata.get("task_id") != manifest.task_id
        ):
            raise RuntimeError(f"research_import_dataset_generation_mismatch:{spec.dataset_name}")


def _published_row_counts(lake: Path, generation_id: str) -> dict[str, int]:
    pointer_path = lake / "gold" / "entry_quality_history_generation.json"
    if not pointer_path.is_file():
        return {}
    payload = json.loads(pointer_path.read_text("utf-8"))
    if payload.get("generation_id") != generation_id:
        return {}
    return {str(key): int(value) for key, value in dict(payload.get("row_counts") or {}).items()}


def _generation_is_published(lake: Path, manifest: ResearchResultManifest) -> bool:
    pointer_path = lake / "gold" / "entry_quality_history_generation.json"
    if not pointer_path.is_file():
        return False
    try:
        payload = json.loads(pointer_path.read_text("utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("generation_id") == manifest.generation_id
        and payload.get("snapshot_id") == manifest.snapshot_id
        and payload.get("task_id") == manifest.task_id
    )


def _finalize_committed_import(
    queue: Path,
    task_id: str,
    manifest: (
        ResearchResultManifest
        | AlphaFactoryResultManifest
        | FactorResearchResultManifest
        | FactorFactoryResultManifest
    ),
    *,
    status: ResearchTaskStatus | None = None,
) -> None:
    current = status or read_research_status(queue, task_id)
    if current is None:
        raise RuntimeError("research_import_status_missing_after_publish")
    inbox = queue / "results" / "inbox" / task_id
    imported = queue / "results" / "imported" / task_id
    imported.parent.mkdir(parents=True, exist_ok=True)
    if inbox.exists() and not imported.exists():
        os.replace(inbox, imported)
    running = queue / "running" / task_id
    completed = queue / "completed" / task_id
    if running.exists() and not completed.exists():
        os.replace(running, completed)
    if (
        current.state == ResearchTaskState.COMPLETED
        and current.gold_generation_id == manifest.generation_id
        and imported.is_dir()
        and completed.is_dir()
    ):
        return
    now = datetime.now(UTC)
    completed_status = current.model_copy(
        update={
            "state": ResearchTaskState.COMPLETED,
            "heartbeat_at": now,
            "completed_at": now,
            "lease_expires_at": None,
            "import_status": "imported",
            "last_error": None,
            "gold_generation_id": manifest.generation_id,
        }
    )
    write_research_status(queue, completed_status)
    _write_validation_event(
        queue,
        task_id,
        "atomic_gold_publish",
        "PASS",
        manifest.generation_id,
    )
    try:
        release_snapshot_payload(
            queue,
            manifest.snapshot_id,
            reason="completed_import",
        )
    except Exception as exc:
        _write_validation_event(
            queue,
            task_id,
            "snapshot_payload_release",
            "RETRY",
            f"{type(exc).__name__}:{str(exc)[:800]}",
        )


def _record_finalize_pending(
    queue: Path,
    status: ResearchTaskStatus,
    generation_id: str,
    exc: Exception,
) -> None:
    pending = status.model_copy(
        update={
            "state": ResearchTaskState.PUBLISHING,
            "heartbeat_at": datetime.now(UTC),
            "lease_expires_at": None,
            "import_status": "finalize_pending",
            "last_error": f"{type(exc).__name__}:{str(exc)[:800]}",
            "gold_generation_id": generation_id,
        }
    )
    write_research_status(queue, pending)


def _record_publish_retry(
    queue: Path,
    status: ResearchTaskStatus,
    generation_id: str,
    exc: Exception,
) -> None:
    detail = f"{type(exc).__name__}:{str(exc)[:800]}"
    pending = status.model_copy(
        update={
            "state": ResearchTaskState.PUBLISHING,
            "heartbeat_at": datetime.now(UTC),
            "lease_expires_at": None,
            "import_status": "publish_retry_pending",
            "last_error": detail,
            "gold_generation_id": generation_id,
        }
    )
    write_research_status(queue, pending)
    _write_validation_event(
        queue,
        status.task_id,
        "atomic_gold_publish",
        "RETRY",
        detail,
    )


def _load_result_manifest(
    root: Path,
) -> (
    ResearchResultManifest
    | AlphaFactoryResultManifest
    | FactorResearchResultManifest
    | FactorFactoryResultManifest
):
    return RESEARCH_RESULT_ADAPTER.validate_json((root / "manifest.json").read_text("utf-8"))


def _load_task_envelope(queue: Path, task_id: str) -> ResearchTaskEnvelope:
    for state in TASK_DIRECTORY_STATES:
        path = queue / state / task_id / "task.json"
        if not path.is_file():
            continue
        return RESEARCH_TASK_ADAPTER.validate_json(path.read_text("utf-8"))
    raise FileNotFoundError(f"research task missing: {task_id}")


def _initial_import_status(
    task: ResearchTaskEnvelope,
    snapshot: (
        ResearchSnapshotManifest
        | AlphaFactorySnapshotManifest
        | FactorResearchSnapshotManifest
        | FactorFactorySnapshotManifest
    ),
) -> ResearchTaskStatus:
    if isinstance(task, FactorFactoryTask):
        start_date = task.as_of_date
        end_date = task.as_of_date
        mode = "PARITY_FULL/bootstrap_full"
        cost_mode = f"point_in_task_{task.cost_quantile}"
    elif isinstance(task, AlphaFactoryTask):
        start_date = task.as_of_date
        end_date = task.as_of_date
        mode = "alpha_factory"
        cost_mode = "research"
    elif isinstance(task, FactorResearchTask):
        start_date = task.start_date
        end_date = task.end_date
        mode = "factor_research"
        cost_mode = "research"
    else:
        start_date = task.start_date
        end_date = task.end_date
        mode = task.mode
        cost_mode = task.cost_mode
    return ResearchTaskStatus(
        task_id=task.task_id,
        snapshot_id=task.snapshot_id,
        task_type=task.task_type,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        cost_mode=cost_mode,
        state=ResearchTaskState.VALIDATING_ON_CLOUD,
        requested_at=task.requested_at,
        max_attempts=task.max_attempts,
        input_bytes=snapshot.total_input_bytes,
        import_status="validating",
    )


def _reject_result(
    queue: Path,
    task: ResearchTaskEnvelope,
    status: ResearchTaskStatus,
    inbox: Path,
    running: Path,
    exc: Exception,
) -> None:
    detail = f"{type(exc).__name__}:{str(exc)[:800]}"
    now = datetime.now(UTC)
    rejected_status = status.model_copy(
        update={
            "state": ResearchTaskState.REJECTED,
            "heartbeat_at": now,
            "completed_at": now,
            "lease_expires_at": None,
            "import_status": "rejected",
            "last_error": detail,
        }
    )
    write_research_status(queue, rejected_status)
    _write_validation_event(queue, task.task_id, "strict_result_validation", "FAIL", detail)
    rejected = queue / "results" / "rejected" / task.task_id
    if inbox.exists() and not rejected.exists():
        os.replace(inbox, rejected)
    failed = queue / "failed" / task.task_id
    if running.exists() and not failed.exists():
        os.replace(running, failed)


def _write_validation_event(
    queue: Path,
    task_id: str,
    check_name: str,
    status: str,
    detail: str,
) -> None:
    event = ResearchValidationEvent(
        task_id=task_id,
        stage="cloud",
        check_name=check_name,
        status=status,
        detail=detail[:1200],
        observed_at=datetime.now(UTC),
    )
    path = queue / "validation" / f"{task_id}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(event.model_dump_json() + "\n")
