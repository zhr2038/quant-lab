from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from quant_lab.research.entry_quality import (
    ENTRY_QUALITY_HISTORY_OUTPUT_SPECS,
    ENTRY_QUALITY_HISTORY_REPORT_NAMES,
    ENTRY_QUALITY_SCHEMA_VERSION,
)
from quant_lab.research_plane.contracts import (
    ResearchResultManifest,
    ResearchSnapshotManifest,
    ResearchTask,
    ResearchWorkerReceipt,
)
from quant_lab.research_plane.signatures import sha256_bytes, sha256_file, verify_payload
from quant_lab.research_plane.snapshot import verify_snapshot_manifest

FORBIDDEN_LIVE_STATE = "LIVE_SMALL_READY"
REQUIRED_ANTI_LEAKAGE_CHECKS = frozenset(
    {
        "history_window_respected",
        "label_ts_after_decision_ts",
        "forward_label_end_boundary",
        "candidate_label_identity",
        "market_future_data_excluded",
        "closed_bar_inputs_only",
        "walk_forward_semantics",
        "horizon_completion",
        "bundle_source_identity",
        "read_only_no_live_action",
    }
)


@dataclass(frozen=True)
class ValidatedEntryQualityHistoryResult:
    manifest: ResearchResultManifest
    receipt: ResearchWorkerReceipt
    frames: dict[str, pl.DataFrame]
    reports: dict[str, bytes]


def validate_research_task_snapshot(
    task: ResearchTask,
    snapshot: ResearchSnapshotManifest,
    *,
    task_public_key: Ed25519PublicKey,
    expected_key_id: str,
    expected_quant_lab_commit: str | None = None,
    snapshot_root: Path | None = None,
) -> None:
    if task.signature_key_id != expected_key_id:
        raise ValueError("research_task_unknown_signature_key")
    if snapshot.signature_key_id != expected_key_id:
        raise ValueError("research_snapshot_unknown_signature_key")
    verify_payload(task, task.signature, task_public_key)
    verify_payload(snapshot, snapshot.signature, task_public_key)
    verify_snapshot_manifest(snapshot, final_root=snapshot_root)
    if task.snapshot_id != snapshot.snapshot_id:
        raise ValueError("research_task_snapshot_id_mismatch")
    if task.snapshot_manifest_sha256 != snapshot.manifest_sha256:
        raise ValueError("research_task_snapshot_digest_mismatch")
    if task.quant_lab_commit != snapshot.quant_lab_commit:
        raise ValueError("research_task_snapshot_commit_mismatch")
    if task.entry_quality_schema_version != snapshot.entry_quality_schema_version:
        raise ValueError("research_task_snapshot_schema_mismatch")
    if task.selected_v5_bundle_id != snapshot.selected_v5_bundle_id:
        raise ValueError("research_task_snapshot_bundle_mismatch")
    if expected_quant_lab_commit is not None and task.quant_lab_commit != expected_quant_lab_commit:
        raise ValueError("research_task_current_commit_mismatch")


def validate_entry_quality_history_result_bundle(
    bundle_root: str | Path,
    *,
    manifest: ResearchResultManifest,
    receipt: ResearchWorkerReceipt,
    task: ResearchTask,
    snapshot: ResearchSnapshotManifest,
    worker_public_key: Ed25519PublicKey,
    expected_worker_key_id: str,
    max_result_bytes: int,
) -> ValidatedEntryQualityHistoryResult:
    root = Path(bundle_root).resolve(strict=True)
    if manifest.worker_key_id != expected_worker_key_id:
        raise ValueError("research_result_unknown_worker_key")
    if receipt.worker_key_id != expected_worker_key_id:
        raise ValueError("research_receipt_unknown_worker_key")
    verify_payload(manifest, manifest.signature, worker_public_key)
    verify_payload(receipt, receipt.signature, worker_public_key)
    _validate_result_binding(manifest, receipt, task, snapshot)

    manifest_path = _safe_bundle_path(root, "manifest.json")
    if receipt.result_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("research_receipt_manifest_sha256_mismatch")

    expected_specs = {spec.dataset_name: spec for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS}
    actual_outputs = {item.dataset_name: item for item in manifest.outputs}
    if set(actual_outputs) != set(expected_specs):
        raise ValueError("research_result_output_set_mismatch")
    if len(manifest.outputs) != len(expected_specs):
        raise ValueError("research_result_duplicate_output")
    report_paths = {item.relative_path for item in manifest.reports}
    expected_reports = {f"reports/{name}" for name in ENTRY_QUALITY_HISTORY_REPORT_NAMES}
    if report_paths != expected_reports or len(manifest.reports) != len(expected_reports):
        raise ValueError("research_result_report_set_mismatch")

    declared_bytes = sum(item.size_bytes for item in manifest.outputs) + sum(
        item.size_bytes for item in manifest.reports
    )
    if declared_bytes != manifest.output_bytes or declared_bytes > max_result_bytes:
        raise ValueError("research_result_size_limit_exceeded")

    frames: dict[str, pl.DataFrame] = {}
    for dataset_name, spec in expected_specs.items():
        output = actual_outputs[dataset_name]
        _validate_output_contract(output, spec)
        path = _safe_bundle_path(root, output.relative_path)
        if path.stat().st_size != output.size_bytes or sha256_file(path) != output.sha256:
            raise ValueError(f"research_result_file_integrity_mismatch:{dataset_name}")
        schema = pl.read_parquet_schema(path)
        if list(schema.items()) != list(spec.schema.items()):
            raise ValueError(f"research_result_schema_mismatch:{dataset_name}")
        if schema_fingerprint(schema) != output.schema_fingerprint:
            raise ValueError(f"research_result_schema_fingerprint_mismatch:{dataset_name}")
        actual_rows = int(pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item())
        if actual_rows != output.row_count:
            raise ValueError(f"research_result_row_count_mismatch:{dataset_name}")
        frame = pl.read_parquet(path)
        _validate_frame_scope(frame, dataset_name=dataset_name, task=task)
        _reject_forbidden_frame_value(frame, dataset_name)
        frames[dataset_name] = frame

    reports: dict[str, bytes] = {}
    for report in manifest.reports:
        path = _safe_bundle_path(root, report.relative_path)
        if path.stat().st_size != report.size_bytes or sha256_file(path) != report.sha256:
            raise ValueError(f"research_result_report_integrity_mismatch:{report.relative_path}")
        payload = path.read_bytes()
        if FORBIDDEN_LIVE_STATE.encode("ascii") in payload:
            raise ValueError(f"research_result_live_state_forbidden:{report.relative_path}")
        reports[Path(report.relative_path).name] = payload

    _validate_anti_leakage(frames["v5_entry_quality_history_anti_leakage_check"])
    output_rows = sum(frame.height for frame in frames.values())
    if receipt.output_rows != output_rows:
        raise ValueError("research_receipt_output_rows_mismatch")
    return ValidatedEntryQualityHistoryResult(
        manifest=manifest,
        receipt=receipt,
        frames=frames,
        reports=reports,
    )


def _validate_result_binding(
    manifest: ResearchResultManifest,
    receipt: ResearchWorkerReceipt,
    task: ResearchTask,
    snapshot: ResearchSnapshotManifest,
) -> None:
    if manifest.task_id != task.task_id or receipt.task_id != task.task_id:
        raise ValueError("research_result_task_mismatch")
    if manifest.snapshot_id != snapshot.snapshot_id or receipt.snapshot_id != snapshot.snapshot_id:
        raise ValueError("research_result_snapshot_mismatch")
    if manifest.snapshot_manifest_sha256 != snapshot.manifest_sha256:
        raise ValueError("research_result_snapshot_digest_mismatch")
    if manifest.quant_lab_commit != task.quant_lab_commit:
        raise ValueError("research_result_quant_lab_commit_mismatch")
    if (
        manifest.worker_commit != task.quant_lab_commit
        or receipt.worker_commit != task.quant_lab_commit
    ):
        raise ValueError("research_result_worker_code_mismatch")
    if manifest.entry_quality_schema_version != ENTRY_QUALITY_SCHEMA_VERSION:
        raise ValueError("research_result_schema_version_mismatch")
    if manifest.entry_quality_schema_version != task.entry_quality_schema_version:
        raise ValueError("research_result_task_schema_mismatch")
    if manifest.selected_v5_bundle_id != task.selected_v5_bundle_id:
        raise ValueError("research_result_bundle_id_mismatch")
    expected_parameters = (
        task.start_date,
        task.end_date,
        task.mode,
        task.cost_mode,
        task.window_hours,
    )
    actual_parameters = (
        manifest.start_date,
        manifest.end_date,
        manifest.mode,
        manifest.cost_mode,
        manifest.window_hours,
    )
    if actual_parameters != expected_parameters:
        raise ValueError("research_result_task_parameters_mismatch")
    if manifest.input_bytes != snapshot.total_input_bytes:
        raise ValueError("research_result_input_bytes_mismatch")
    if manifest.cache_hit_bytes + manifest.downloaded_bytes != snapshot.total_input_bytes:
        raise ValueError("research_result_cache_accounting_mismatch")
    if receipt.input_bytes != manifest.input_bytes:
        raise ValueError("research_receipt_input_bytes_mismatch")
    if receipt.downloaded_bytes != manifest.downloaded_bytes:
        raise ValueError("research_receipt_downloaded_bytes_mismatch")
    if receipt.cache_hit_bytes != manifest.cache_hit_bytes:
        raise ValueError("research_receipt_cache_hit_bytes_mismatch")
    if receipt.anti_leakage_status != manifest.anti_leakage_status:
        raise ValueError("research_receipt_anti_leakage_mismatch")
    if receipt.completed_at != manifest.completed_at:
        raise ValueError("research_receipt_completed_at_mismatch")


def _validate_output_contract(output: Any, spec: Any) -> None:
    if output.publish_mode != spec.publish_mode:
        raise ValueError(f"research_result_publish_mode_mismatch:{spec.dataset_name}")
    if output.primary_keys != list(spec.primary_keys):
        raise ValueError(f"research_result_primary_key_mismatch:{spec.dataset_name}")
    if output.window_keys != list(spec.window_keys):
        raise ValueError(f"research_result_window_key_mismatch:{spec.dataset_name}")
    if output.empty_result_semantics != spec.empty_result_semantics:
        raise ValueError(f"research_result_empty_semantics_mismatch:{spec.dataset_name}")


def _validate_frame_scope(
    frame: pl.DataFrame,
    *,
    dataset_name: str,
    task: ResearchTask,
) -> None:
    expected_text = {
        "start_date": task.start_date.isoformat(),
        "end_date": task.end_date.isoformat(),
        "window_mode": task.mode,
        "cost_mode": task.cost_mode,
        "generated_from_bundle_id": task.selected_v5_bundle_id,
        "schema_version": task.entry_quality_schema_version,
    }
    for column, expected in expected_text.items():
        if column not in frame.columns or frame.is_empty():
            continue
        values = {str(value) for value in frame.get_column(column).drop_nulls().to_list()}
        if values != {expected}:
            raise ValueError(f"research_result_scope_mismatch:{dataset_name}:{column}")
    if "window_hours" in frame.columns and not frame.is_empty():
        values = {int(value) for value in frame.get_column("window_hours").drop_nulls().to_list()}
        if values != {task.window_hours}:
            raise ValueError(f"research_result_scope_mismatch:{dataset_name}:window_hours")
    if "quant_lab_git_commit" in frame.columns and not frame.is_empty():
        for value in frame.get_column("quant_lab_git_commit").drop_nulls().cast(pl.Utf8).to_list():
            if not task.quant_lab_commit.startswith(str(value)):
                raise ValueError(
                    f"research_result_scope_mismatch:{dataset_name}:quant_lab_git_commit"
                )
    start_dt = datetime.combine(task.start_date, time.min, tzinfo=UTC)
    end_dt = datetime.combine(task.end_date + timedelta(days=1), time.min, tzinfo=UTC)
    for column in ("entry_ts", "ts_utc"):
        if column not in frame.columns:
            continue
        invalid = frame.filter(
            pl.col(column).is_not_null()
            & ((pl.col(column) < start_dt) | (pl.col(column) >= end_dt))
        )
        if not invalid.is_empty():
            raise ValueError(f"research_result_time_window_mismatch:{dataset_name}:{column}")


def _validate_anti_leakage(frame: pl.DataFrame) -> None:
    if frame.is_empty() or not {"check_name", "status", "violation_count"}.issubset(frame.columns):
        raise ValueError("research_result_anti_leakage_missing")
    names = {str(value) for value in frame.get_column("check_name").drop_nulls().to_list()}
    if not REQUIRED_ANTI_LEAKAGE_CHECKS.issubset(names):
        raise ValueError("research_result_anti_leakage_incomplete")
    statuses = {str(value).upper() for value in frame.get_column("status").to_list()}
    violations = sum(int(value or 0) for value in frame.get_column("violation_count").to_list())
    if statuses != {"PASS"} or violations != 0:
        raise ValueError("research_result_anti_leakage_failed")


def _reject_forbidden_frame_value(frame: pl.DataFrame, dataset_name: str) -> None:
    for column, dtype in frame.schema.items():
        if dtype != pl.Utf8:
            continue
        values = frame.get_column(column).drop_nulls().cast(pl.Utf8).to_list()
        if any(FORBIDDEN_LIVE_STATE in str(value) for value in values):
            raise ValueError(f"research_result_live_state_forbidden:{dataset_name}:{column}")


def _safe_bundle_path(root: Path, relative_path: str) -> Path:
    unresolved = root / relative_path
    try:
        parts = unresolved.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("research_result_path_escape") from exc
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("research_result_symlink_forbidden")
    candidate = unresolved.resolve(strict=True)
    if root not in candidate.parents:
        raise ValueError("research_result_path_escape")
    if not candidate.is_file():
        raise ValueError("research_result_non_file_forbidden")
    return candidate


def schema_fingerprint(schema: Any) -> str:
    payload = json.dumps(
        [(str(name), str(dtype)) for name, dtype in schema.items()],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)
