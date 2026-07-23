from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.alpha_factory.factory import (
    ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS,
)
from quant_lab.research.alpha_factory.factory import (
    SCHEMA_VERSION as ALPHA_FACTORY_SCHEMA_VERSION,
)
from quant_lab.research.entry_quality import (
    ENTRY_QUALITY_HISTORY_OUTPUT_SPECS,
    ENTRY_QUALITY_HISTORY_REPORT_NAMES,
    ENTRY_QUALITY_SCHEMA_VERSION,
)
from quant_lab.research.factor_research.contracts import FactorResearchDecision
from quant_lab.research.factor_research.outputs import FACTOR_RESEARCH_OUTPUT_SPECS
from quant_lab.research.factor_research.registry import (
    RESEARCH_TRIAL_LEDGER_DATASET,
    trial_ledger_digest,
)
from quant_lab.research.second_stage_alpha_factory import (
    SCHEMA_VERSION as SECOND_STAGE_ALPHA_FACTORY_SCHEMA_VERSION,
)
from quant_lab.research_plane.contracts import (
    ALPHA_FACTORY_RECEIPT_SCHEMA,
    ALPHA_FACTORY_RESULT_SCHEMA,
    FACTOR_RESEARCH_RECEIPT_SCHEMA,
    FACTOR_RESEARCH_RESULT_SCHEMA,
    AlphaFactoryResultManifest,
    AlphaFactorySnapshotManifest,
    AlphaFactoryTask,
    AlphaFactoryWorkerReceipt,
    FactorFactorySnapshotManifest,
    FactorFactoryTask,
    FactorResearchResultManifest,
    FactorResearchSnapshotManifest,
    FactorResearchTask,
    FactorResearchWorkerReceipt,
    ResearchResultManifest,
    ResearchSnapshotManifest,
    ResearchTask,
    ResearchWorkerReceipt,
    TradeLevelHistorySnapshotManifest,
    TradeLevelHistoryTask,
    V5CandidateEvidenceSnapshotManifest,
    V5CandidateEvidenceTask,
)
from quant_lab.research_plane.factor_factory_snapshot import (
    verify_factor_factory_snapshot_manifest,
)
from quant_lab.research_plane.signatures import sha256_bytes, sha256_file, verify_payload
from quant_lab.research_plane.snapshot import (
    verify_alpha_factory_snapshot_manifest,
    verify_factor_research_snapshot_manifest,
    verify_snapshot_manifest,
)
from quant_lab.research_plane.v5_candidate_evidence_snapshot import (
    verify_v5_candidate_evidence_snapshot_manifest,
)

FORBIDDEN_LIVE_STATE = "LIVE_SMALL_READY"
ALPHA_FACTORY_FORBIDDEN_LIVE_STATES = frozenset(
    {"LIVE_SMALL_READY", "LIVE", "CANARY", "ENFORCE", "AUTO_PROMOTE"}
)
ALPHA_FACTORY_RESULT_DECISIONS = frozenset({"RESEARCH", "KEEP_SHADOW", "KILL", "PAPER_READY"})
ALPHA_FACTORY_DECISIONS_BY_DATASET = {
    "second_stage_alpha_factory_summary": frozenset(
        {"RESEARCH_ONLY", "KEEP_SHADOW", "KILL", "PAPER_READY"}
    ),
    "exit_policy_review_sample": frozenset({"RESEARCH_ONLY", "REVIEW_EXIT_POLICY"}),
    "exit_policy_review_summary": frozenset({"RESEARCH_ONLY", "REVIEW_EXIT_POLICY"}),
    "alpha_factory_result": ALPHA_FACTORY_RESULT_DECISIONS,
}
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
ALPHA_FACTORY_REQUIRED_REPORTS = frozenset(
    {
        "reports/factor_strategy_bridge_candidates.csv",
        "reports/alpha_factory_worker_report.json",
        "reports/alpha_factory_anti_leakage.json",
    }
)
ALPHA_FACTORY_WINDOWED_AS_OF_DATASETS = frozenset({"second_stage_alpha_factory_sample"})
FACTOR_RESEARCH_REQUIRED_REPORTS = frozenset(
    {
        "reports/factor_research_worker_report.json",
        "reports/factor_research_anti_leakage.json",
    }
)
FACTOR_RESEARCH_ALLOWED_DECISIONS = frozenset(item.value for item in FactorResearchDecision)
FACTOR_RESEARCH_ALLOWED_CANDIDATE_STATES = frozenset(
    {
        "PAPER_CANDIDATE",
        "SIGNAL_VALID",
        "PORTFOLIO_FAIL",
        "SIGNAL_CANDIDATE",
        "REJECTED",
        "RESEARCH",
    }
)


@dataclass(frozen=True)
class ValidatedEntryQualityHistoryResult:
    manifest: ResearchResultManifest
    receipt: ResearchWorkerReceipt
    frames: dict[str, pl.DataFrame]
    reports: dict[str, bytes]


@dataclass(frozen=True)
class ValidatedAlphaFactoryResult:
    manifest: AlphaFactoryResultManifest
    receipt: AlphaFactoryWorkerReceipt
    output_paths: dict[str, Path]
    reports: dict[str, bytes]


@dataclass(frozen=True)
class ValidatedFactorResearchResult:
    manifest: FactorResearchResultManifest
    receipt: FactorResearchWorkerReceipt
    output_paths: dict[str, Path]
    reports: dict[str, bytes]


def validate_research_task_snapshot(
    task: (
        ResearchTask
        | AlphaFactoryTask
        | FactorResearchTask
        | FactorFactoryTask
        | V5CandidateEvidenceTask
        | TradeLevelHistoryTask
    ),
    snapshot: (
        ResearchSnapshotManifest
        | AlphaFactorySnapshotManifest
        | FactorResearchSnapshotManifest
        | FactorFactorySnapshotManifest
        | V5CandidateEvidenceSnapshotManifest
        | TradeLevelHistorySnapshotManifest
    ),
    *,
    task_public_key: Ed25519PublicKey,
    expected_key_id: str,
    expected_quant_lab_commit: str | None = None,
    snapshot_root: Path | None = None,
) -> None:
    matching_types = (
        (isinstance(task, ResearchTask) and isinstance(snapshot, ResearchSnapshotManifest))
        or (
            isinstance(task, AlphaFactoryTask)
            and isinstance(snapshot, AlphaFactorySnapshotManifest)
        )
        or (
            isinstance(task, FactorResearchTask)
            and isinstance(snapshot, FactorResearchSnapshotManifest)
        )
        or (
            isinstance(task, FactorFactoryTask)
            and isinstance(snapshot, FactorFactorySnapshotManifest)
        )
        or (
            isinstance(task, V5CandidateEvidenceTask)
            and isinstance(snapshot, V5CandidateEvidenceSnapshotManifest)
        )
        or (
            isinstance(task, TradeLevelHistoryTask)
            and isinstance(snapshot, TradeLevelHistorySnapshotManifest)
        )
    )
    if not matching_types:
        raise ValueError("research_task_snapshot_type_mismatch")
    if task.signature_key_id != expected_key_id:
        raise ValueError("research_task_unknown_signature_key")
    if snapshot.signature_key_id != expected_key_id:
        raise ValueError("research_snapshot_unknown_signature_key")
    verify_payload(task, task.signature, task_public_key)
    if not isinstance(
        snapshot,
        (
            FactorFactorySnapshotManifest,
            V5CandidateEvidenceSnapshotManifest,
            TradeLevelHistorySnapshotManifest,
        ),
    ):
        verify_payload(snapshot, snapshot.signature, task_public_key)
    if isinstance(snapshot, AlphaFactorySnapshotManifest):
        verify_alpha_factory_snapshot_manifest(snapshot, final_root=snapshot_root)
    elif isinstance(snapshot, FactorResearchSnapshotManifest):
        verify_factor_research_snapshot_manifest(snapshot, final_root=snapshot_root)
    elif isinstance(snapshot, FactorFactorySnapshotManifest):
        verify_factor_factory_snapshot_manifest(
            snapshot,
            final_root=snapshot_root,
            public_key=task_public_key,
        )
    elif isinstance(snapshot, V5CandidateEvidenceSnapshotManifest):
        if snapshot_root is None:
            verify_payload(snapshot, snapshot.signature, task_public_key)
        else:
            verify_v5_candidate_evidence_snapshot_manifest(
                snapshot,
                final_root=snapshot_root,
                public_key=task_public_key,
            )
    elif isinstance(snapshot, TradeLevelHistorySnapshotManifest):
        if snapshot_root is None:
            verify_payload(snapshot, snapshot.signature, task_public_key)
        else:
            from quant_lab.research_plane.trade_level_history_snapshot import (  # noqa: PLC0415
                verify_trade_level_history_snapshot_manifest,
            )

            verify_trade_level_history_snapshot_manifest(
                snapshot,
                final_root=snapshot_root,
                public_key=task_public_key,
            )
    else:
        verify_snapshot_manifest(snapshot, final_root=snapshot_root)
    if task.snapshot_id != snapshot.snapshot_id:
        raise ValueError("research_task_snapshot_id_mismatch")
    if task.snapshot_manifest_sha256 != snapshot.manifest_sha256:
        raise ValueError("research_task_snapshot_digest_mismatch")
    if task.quant_lab_commit != snapshot.quant_lab_commit:
        raise ValueError("research_task_snapshot_commit_mismatch")
    if isinstance(task, AlphaFactoryTask) and isinstance(
        snapshot,
        AlphaFactorySnapshotManifest,
    ):
        if task.alpha_factory_schema_version != snapshot.alpha_factory_schema_version:
            raise ValueError("research_task_snapshot_alpha_schema_mismatch")
        if task.second_stage_schema_version != snapshot.second_stage_schema_version:
            raise ValueError("research_task_snapshot_second_stage_schema_mismatch")
        if task.template_registry_digest != snapshot.template_registry_digest:
            raise ValueError("research_task_snapshot_registry_digest_mismatch")
        factor_binding_fields = (
            "factor_generation_id",
            "factor_generation_digest",
            "factor_generation_as_of_date",
            "factor_generation_published_at",
            "hypothesis_registry_digest",
            "trial_ledger_digest",
            "factor_generation_fresh",
            "factor_generation_hypothesis_ids",
        )
        if tuple(getattr(task, field) for field in factor_binding_fields) != tuple(
            getattr(snapshot, field) for field in factor_binding_fields
        ):
            raise ValueError("research_task_snapshot_factor_generation_mismatch")
        if (
            task.as_of_date,
            task.lookback_days,
            task.max_candidates,
        ) != (
            snapshot.as_of_date,
            snapshot.lookback_days,
            snapshot.max_candidates,
        ):
            raise ValueError("research_task_snapshot_parameters_mismatch")
    elif isinstance(task, FactorResearchTask) and isinstance(
        snapshot,
        FactorResearchSnapshotManifest,
    ):
        expected_identity = (
            task.factor_research_schema_version,
            task.hypothesis_registry_digest,
            task.trial_ledger_digest,
            task.source_input_digest,
            task.as_of_date,
            task.start_date,
            task.end_date,
            task.max_history_days,
            task.hypothesis_ids,
            task.trial_ids,
            task.test_count,
        )
        observed_identity = (
            snapshot.factor_research_schema_version,
            snapshot.hypothesis_registry_digest,
            snapshot.trial_ledger_digest,
            snapshot.source_input_digest,
            snapshot.as_of_date,
            snapshot.start_date,
            snapshot.end_date,
            snapshot.max_history_days,
            snapshot.hypothesis_ids,
            snapshot.trial_ids,
            snapshot.test_count,
        )
        if observed_identity != expected_identity:
            raise ValueError("research_task_snapshot_factor_research_identity_mismatch")
    elif isinstance(task, FactorFactoryTask) and isinstance(
        snapshot,
        FactorFactorySnapshotManifest,
    ):
        expected_identity = (
            task.parameters.model_dump(exclude={"as_of_date"}),
            task.factor_plan_digest,
            task.source_input_digest,
            task.cost_input_digest,
        )
        observed_identity = (
            {
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
            snapshot.factor_plan_digest,
            snapshot.source_input_digest,
            snapshot.cost_input_digest,
        )
        if snapshot.schema_version == "quant_lab_factor_factory_snapshot.v1":
            expected_identity += (
                task.previous_generation_id,
                task.previous_generation_digest,
                task.as_of_date,
            )
            observed_identity += (
                snapshot.previous_generation_id,
                snapshot.previous_generation_digest,
                snapshot.as_of_date,
            )
        if observed_identity != expected_identity:
            raise ValueError("research_task_snapshot_factor_factory_identity_mismatch")
    elif isinstance(task, V5CandidateEvidenceTask) and isinstance(
        snapshot,
        V5CandidateEvidenceSnapshotManifest,
    ):
        expected_identity = (
            task.input_fingerprint_digest,
            task.as_of_date,
            task.mode,
            task.lookback_days,
            task.horizon_hours,
            task.include_historical_outcomes,
            task.candidate_label_schema_version,
            task.strategy_evidence_version,
            task.projection_version,
        )
        observed_identity = (
            snapshot.input_fingerprint_digest,
            snapshot.as_of_date,
            snapshot.mode,
            snapshot.lookback_days,
            snapshot.horizon_hours,
            snapshot.include_historical_outcomes,
            snapshot.candidate_label_schema_version,
            snapshot.strategy_evidence_version,
            snapshot.projection_version,
        )
        if observed_identity != expected_identity:
            raise ValueError("research_task_snapshot_v5_candidate_evidence_identity_mismatch")
    elif isinstance(task, TradeLevelHistoryTask) and isinstance(
        snapshot,
        TradeLevelHistorySnapshotManifest,
    ):
        expected_identity = (
            task.input_fingerprint_digest,
            task.as_of_date,
            task.history_mode,
            task.candidate_evidence_generation_id,
            task.candidate_evidence_generation_digest,
            task.candidate_evidence_input_fingerprint,
            task.trade_event_schema_version,
            task.trade_label_schema_version,
            task.similarity_schema_version,
            task.similarity_availability_policy,
        )
        observed_identity = (
            snapshot.input_fingerprint_digest,
            snapshot.as_of_date,
            snapshot.history_mode,
            snapshot.candidate_evidence_generation_id,
            snapshot.candidate_evidence_generation_digest,
            snapshot.candidate_evidence_input_fingerprint,
            snapshot.trade_event_schema_version,
            snapshot.trade_label_schema_version,
            snapshot.similarity_schema_version,
            snapshot.similarity_availability_policy,
        )
        if observed_identity != expected_identity:
            raise ValueError(
                "research_task_snapshot_trade_level_history_identity_mismatch"
            )
    elif isinstance(task, ResearchTask) and isinstance(snapshot, ResearchSnapshotManifest):
        if task.entry_quality_schema_version != snapshot.entry_quality_schema_version:
            raise ValueError("research_task_snapshot_schema_mismatch")
    else:  # pragma: no cover - guarded above; keeps the type boundary explicit.
        raise ValueError("research_task_snapshot_type_mismatch")
    if not isinstance(
        task,
        (
            FactorFactoryTask,
            V5CandidateEvidenceTask,
            TradeLevelHistoryTask,
        ),
    ) and (
        task.selected_v5_bundle_id != snapshot.selected_v5_bundle_id
    ):
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


def validate_alpha_factory_result_bundle(
    bundle_root: str | Path,
    *,
    manifest: AlphaFactoryResultManifest,
    receipt: AlphaFactoryWorkerReceipt,
    task: AlphaFactoryTask,
    snapshot: AlphaFactorySnapshotManifest,
    worker_public_key: Ed25519PublicKey,
    expected_worker_key_id: str,
    max_result_bytes: int,
) -> ValidatedAlphaFactoryResult:
    root = Path(bundle_root).resolve(strict=True)
    if manifest.schema_version != ALPHA_FACTORY_RESULT_SCHEMA:
        raise ValueError("alpha_factory_result_schema_version_mismatch")
    if receipt.schema_version != ALPHA_FACTORY_RECEIPT_SCHEMA:
        raise ValueError("alpha_factory_receipt_schema_version_mismatch")
    if manifest.worker_key_id != expected_worker_key_id:
        raise ValueError("alpha_factory_result_unknown_worker_key")
    if receipt.worker_key_id != expected_worker_key_id:
        raise ValueError("alpha_factory_receipt_unknown_worker_key")
    verify_payload(manifest, manifest.signature, worker_public_key)
    verify_payload(receipt, receipt.signature, worker_public_key)
    _validate_alpha_factory_result_binding(manifest, receipt, task, snapshot)

    manifest_path = _safe_bundle_path(root, "manifest.json")
    if receipt.result_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("alpha_factory_receipt_manifest_sha256_mismatch")
    expected_specs = {spec.dataset_name: spec for spec in ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS}
    actual_outputs = {item.dataset_name: item for item in manifest.outputs}
    if set(actual_outputs) != set(expected_specs) or len(manifest.outputs) != len(expected_specs):
        raise ValueError("alpha_factory_result_output_set_mismatch")
    report_paths = {item.relative_path for item in manifest.reports}
    if report_paths != ALPHA_FACTORY_REQUIRED_REPORTS or len(manifest.reports) != len(
        ALPHA_FACTORY_REQUIRED_REPORTS
    ):
        raise ValueError("alpha_factory_result_report_set_mismatch")
    declared_bytes = sum(item.size_bytes for item in manifest.outputs) + sum(
        item.size_bytes for item in manifest.reports
    )
    if declared_bytes != manifest.output_bytes or declared_bytes > max_result_bytes:
        raise ValueError("alpha_factory_result_size_limit_exceeded")

    output_paths: dict[str, Path] = {}
    for dataset_name, spec in expected_specs.items():
        output = actual_outputs[dataset_name]
        _validate_output_contract(output, spec)
        path = _safe_bundle_path(root, output.relative_path)
        if path.stat().st_size != output.size_bytes or sha256_file(path) != output.sha256:
            raise ValueError(f"alpha_factory_result_file_integrity_mismatch:{dataset_name}")
        schema = pl.read_parquet_schema(path)
        if list(schema.items()) != list(spec.schema.items()):
            raise ValueError(f"alpha_factory_result_schema_mismatch:{dataset_name}")
        if schema_fingerprint(schema) != output.schema_fingerprint:
            raise ValueError(f"alpha_factory_result_schema_fingerprint_mismatch:{dataset_name}")
        lazy = pl.scan_parquet(path)
        actual_rows = int(lazy.select(pl.len()).collect(engine="streaming").item())
        if actual_rows != output.row_count:
            raise ValueError(f"alpha_factory_result_row_count_mismatch:{dataset_name}")
        _validate_alpha_frame_scope(lazy, dataset_name=dataset_name, task=task)
        _validate_lazy_unique_keys(lazy, spec.primary_keys, dataset_name)
        _validate_alpha_frame_safety(lazy, dataset_name)
        output_paths[dataset_name] = path

    _validate_alpha_candidate_result_identity(output_paths)
    reports: dict[str, bytes] = {}
    for report in manifest.reports:
        path = _safe_bundle_path(root, report.relative_path)
        if path.stat().st_size != report.size_bytes or sha256_file(path) != report.sha256:
            raise ValueError(
                f"alpha_factory_result_report_integrity_mismatch:{report.relative_path}"
            )
        payload = path.read_bytes()
        if any(state.encode("ascii") in payload for state in ALPHA_FACTORY_FORBIDDEN_LIVE_STATES):
            raise ValueError(f"alpha_factory_result_live_state_forbidden:{report.relative_path}")
        reports[Path(report.relative_path).name] = payload
    _validate_alpha_anti_leakage_report(
        reports["alpha_factory_anti_leakage.json"],
        task=task,
        snapshot=snapshot,
    )
    _validate_alpha_worker_report(
        reports["alpha_factory_worker_report.json"],
        task=task,
        snapshot=snapshot,
    )
    _validate_factor_bridge_report(root / "reports" / "factor_strategy_bridge_candidates.csv")
    output_rows = sum(item.row_count for item in manifest.outputs)
    if receipt.output_rows != output_rows:
        raise ValueError("alpha_factory_receipt_output_rows_mismatch")
    return ValidatedAlphaFactoryResult(
        manifest=manifest,
        receipt=receipt,
        output_paths=output_paths,
        reports=reports,
    )


def validate_factor_research_result_bundle(
    bundle_root: str | Path,
    *,
    manifest: FactorResearchResultManifest,
    receipt: FactorResearchWorkerReceipt,
    task: FactorResearchTask,
    snapshot: FactorResearchSnapshotManifest,
    worker_public_key: Ed25519PublicKey,
    expected_worker_key_id: str,
    max_result_bytes: int,
    snapshot_root: Path | None = None,
) -> ValidatedFactorResearchResult:
    root = Path(bundle_root).resolve(strict=True)
    if manifest.schema_version != FACTOR_RESEARCH_RESULT_SCHEMA:
        raise ValueError("factor_research_result_schema_version_mismatch")
    if receipt.schema_version != FACTOR_RESEARCH_RECEIPT_SCHEMA:
        raise ValueError("factor_research_receipt_schema_version_mismatch")
    if manifest.worker_key_id != expected_worker_key_id:
        raise ValueError("factor_research_result_unknown_worker_key")
    if receipt.worker_key_id != expected_worker_key_id:
        raise ValueError("factor_research_receipt_unknown_worker_key")
    verify_payload(manifest, manifest.signature, worker_public_key)
    verify_payload(receipt, receipt.signature, worker_public_key)
    _validate_factor_research_result_binding(manifest, receipt, task, snapshot)

    manifest_path = _safe_bundle_path(root, "manifest.json")
    if receipt.result_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("factor_research_receipt_manifest_sha256_mismatch")
    expected_specs = {spec.dataset_name: spec for spec in FACTOR_RESEARCH_OUTPUT_SPECS}
    actual_outputs = {item.dataset_name: item for item in manifest.outputs}
    if set(actual_outputs) != set(expected_specs) or len(manifest.outputs) != len(expected_specs):
        raise ValueError("factor_research_result_output_set_mismatch")
    report_paths = {item.relative_path for item in manifest.reports}
    if report_paths != FACTOR_RESEARCH_REQUIRED_REPORTS or len(manifest.reports) != len(
        FACTOR_RESEARCH_REQUIRED_REPORTS
    ):
        raise ValueError("factor_research_result_report_set_mismatch")
    declared_bytes = sum(item.size_bytes for item in manifest.outputs) + sum(
        item.size_bytes for item in manifest.reports
    )
    if declared_bytes != manifest.output_bytes or declared_bytes > max_result_bytes:
        raise ValueError("factor_research_result_size_limit_exceeded")

    output_paths: dict[str, Path] = {}
    for dataset_name, spec in expected_specs.items():
        output = actual_outputs[dataset_name]
        _validate_output_contract(output, spec)
        path = _safe_bundle_path(root, output.relative_path)
        if path.stat().st_size != output.size_bytes or sha256_file(path) != output.sha256:
            raise ValueError(f"factor_research_result_file_integrity_mismatch:{dataset_name}")
        schema = pl.read_parquet_schema(path)
        if list(schema.items()) != list(spec.schema.items()):
            raise ValueError(f"factor_research_result_schema_mismatch:{dataset_name}")
        if schema_fingerprint(schema) != output.schema_fingerprint:
            raise ValueError(f"factor_research_result_schema_fingerprint_mismatch:{dataset_name}")
        lazy = pl.scan_parquet(path)
        actual_rows = int(lazy.select(pl.len()).collect(engine="streaming").item())
        if actual_rows != output.row_count:
            raise ValueError(f"factor_research_result_row_count_mismatch:{dataset_name}")
        _validate_factor_research_scope(lazy, dataset_name=dataset_name, task=task)
        _validate_factor_research_unique_keys(lazy, spec.primary_keys, dataset_name)
        _validate_factor_research_safety(lazy, dataset_name)
        output_paths[dataset_name] = path

    _validate_factor_research_membership(output_paths, task=task)
    if snapshot_root is not None:
        _validate_factor_research_trial_ledger(Path(snapshot_root), task=task, snapshot=snapshot)

    reports: dict[str, bytes] = {}
    for report in manifest.reports:
        path = _safe_bundle_path(root, report.relative_path)
        if path.stat().st_size != report.size_bytes or sha256_file(path) != report.sha256:
            raise ValueError(
                f"factor_research_result_report_integrity_mismatch:{report.relative_path}"
            )
        payload = path.read_bytes()
        if any(state.encode("ascii") in payload for state in ALPHA_FACTORY_FORBIDDEN_LIVE_STATES):
            raise ValueError(f"factor_research_result_live_state_forbidden:{report.relative_path}")
        reports[Path(report.relative_path).name] = payload
    _validate_factor_research_anti_leakage_report(
        reports["factor_research_anti_leakage.json"], task=task, snapshot=snapshot
    )
    _validate_factor_research_worker_report(
        reports["factor_research_worker_report.json"],
        task=task,
        snapshot=snapshot,
        outputs=actual_outputs,
    )
    output_rows = sum(item.row_count for item in manifest.outputs)
    if receipt.output_rows != output_rows:
        raise ValueError("factor_research_receipt_output_rows_mismatch")
    return ValidatedFactorResearchResult(
        manifest=manifest,
        receipt=receipt,
        output_paths=output_paths,
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


def _validate_alpha_factory_result_binding(
    manifest: AlphaFactoryResultManifest,
    receipt: AlphaFactoryWorkerReceipt,
    task: AlphaFactoryTask,
    snapshot: AlphaFactorySnapshotManifest,
) -> None:
    if manifest.task_id != task.task_id or receipt.task_id != task.task_id:
        raise ValueError("alpha_factory_result_task_mismatch")
    if manifest.snapshot_id != snapshot.snapshot_id or receipt.snapshot_id != snapshot.snapshot_id:
        raise ValueError("alpha_factory_result_snapshot_mismatch")
    if manifest.snapshot_manifest_sha256 != snapshot.manifest_sha256:
        raise ValueError("alpha_factory_result_snapshot_digest_mismatch")
    if manifest.quant_lab_commit != task.quant_lab_commit:
        raise ValueError("alpha_factory_result_quant_lab_commit_mismatch")
    if (
        manifest.worker_commit != task.quant_lab_commit
        or receipt.worker_commit != task.quant_lab_commit
    ):
        raise ValueError("alpha_factory_result_worker_code_mismatch")
    if manifest.alpha_factory_schema_version != ALPHA_FACTORY_SCHEMA_VERSION:
        raise ValueError("alpha_factory_result_schema_version_mismatch")
    if manifest.alpha_factory_schema_version != task.alpha_factory_schema_version:
        raise ValueError("alpha_factory_result_task_schema_mismatch")
    if manifest.second_stage_schema_version != SECOND_STAGE_ALPHA_FACTORY_SCHEMA_VERSION:
        raise ValueError("alpha_factory_result_second_stage_schema_mismatch")
    if manifest.second_stage_schema_version != task.second_stage_schema_version:
        raise ValueError("alpha_factory_result_task_second_stage_schema_mismatch")
    if manifest.template_registry_digest != task.template_registry_digest:
        raise ValueError("alpha_factory_result_registry_digest_mismatch")
    factor_binding_fields = (
        "factor_generation_id",
        "factor_generation_digest",
        "factor_generation_as_of_date",
        "factor_generation_published_at",
        "hypothesis_registry_digest",
        "trial_ledger_digest",
        "factor_generation_fresh",
        "factor_generation_hypothesis_ids",
    )
    task_binding = tuple(getattr(task, field) for field in factor_binding_fields)
    if tuple(getattr(snapshot, field) for field in factor_binding_fields) != task_binding:
        raise ValueError("alpha_factory_snapshot_factor_generation_mismatch")
    if tuple(getattr(manifest, field) for field in factor_binding_fields) != task_binding:
        raise ValueError("alpha_factory_result_factor_generation_mismatch")
    if manifest.selected_v5_bundle_id != task.selected_v5_bundle_id:
        raise ValueError("alpha_factory_result_bundle_id_mismatch")
    if (
        manifest.as_of_date,
        manifest.lookback_days,
        manifest.max_candidates,
    ) != (
        task.as_of_date,
        task.lookback_days,
        task.max_candidates,
    ):
        raise ValueError("alpha_factory_result_task_parameters_mismatch")
    if manifest.input_bytes != snapshot.total_input_bytes:
        raise ValueError("alpha_factory_result_input_bytes_mismatch")
    if manifest.cache_hit_bytes + manifest.downloaded_bytes != snapshot.total_input_bytes:
        raise ValueError("alpha_factory_result_cache_accounting_mismatch")
    if receipt.input_bytes != manifest.input_bytes:
        raise ValueError("alpha_factory_receipt_input_bytes_mismatch")
    if receipt.downloaded_bytes != manifest.downloaded_bytes:
        raise ValueError("alpha_factory_receipt_downloaded_bytes_mismatch")
    if receipt.cache_hit_bytes != manifest.cache_hit_bytes:
        raise ValueError("alpha_factory_receipt_cache_hit_bytes_mismatch")
    if receipt.anti_leakage_status != manifest.anti_leakage_status:
        raise ValueError("alpha_factory_receipt_anti_leakage_mismatch")
    if receipt.anti_leakage_violation_count != manifest.anti_leakage_violation_count:
        raise ValueError("alpha_factory_receipt_anti_leakage_count_mismatch")
    if receipt.completed_at != manifest.completed_at:
        raise ValueError("alpha_factory_receipt_completed_at_mismatch")


def _validate_factor_research_result_binding(
    manifest: FactorResearchResultManifest,
    receipt: FactorResearchWorkerReceipt,
    task: FactorResearchTask,
    snapshot: FactorResearchSnapshotManifest,
) -> None:
    if manifest.task_id != task.task_id or receipt.task_id != task.task_id:
        raise ValueError("factor_research_result_task_mismatch")
    if manifest.snapshot_id != snapshot.snapshot_id or receipt.snapshot_id != snapshot.snapshot_id:
        raise ValueError("factor_research_result_snapshot_mismatch")
    if manifest.snapshot_manifest_sha256 != snapshot.manifest_sha256:
        raise ValueError("factor_research_result_snapshot_digest_mismatch")
    if manifest.quant_lab_commit != task.quant_lab_commit:
        raise ValueError("factor_research_result_quant_lab_commit_mismatch")
    if (
        manifest.worker_commit != task.quant_lab_commit
        or receipt.worker_commit != task.quant_lab_commit
    ):
        raise ValueError("factor_research_result_worker_code_mismatch")
    expected_identity = (
        task.factor_research_schema_version,
        task.hypothesis_registry_digest,
        task.trial_ledger_digest,
        task.source_input_digest,
        task.selected_v5_bundle_id,
        task.as_of_date,
        task.start_date,
        task.end_date,
        task.max_history_days,
        task.hypothesis_ids,
        task.trial_ids,
        task.test_count,
    )
    observed_identity = (
        manifest.factor_research_schema_version,
        manifest.hypothesis_registry_digest,
        manifest.trial_ledger_digest,
        manifest.source_input_digest,
        manifest.selected_v5_bundle_id,
        manifest.as_of_date,
        manifest.start_date,
        manifest.end_date,
        manifest.max_history_days,
        manifest.hypothesis_ids,
        manifest.trial_ids,
        manifest.test_count,
    )
    if observed_identity != expected_identity:
        raise ValueError("factor_research_result_identity_mismatch")
    if manifest.input_bytes != snapshot.total_input_bytes:
        raise ValueError("factor_research_result_input_bytes_mismatch")
    if manifest.cache_hit_bytes + manifest.downloaded_bytes != snapshot.total_input_bytes:
        raise ValueError("factor_research_result_cache_accounting_mismatch")
    if receipt.input_bytes != manifest.input_bytes:
        raise ValueError("factor_research_receipt_input_bytes_mismatch")
    if receipt.downloaded_bytes != manifest.downloaded_bytes:
        raise ValueError("factor_research_receipt_downloaded_bytes_mismatch")
    if receipt.cache_hit_bytes != manifest.cache_hit_bytes:
        raise ValueError("factor_research_receipt_cache_hit_bytes_mismatch")
    if receipt.anti_leakage_status != manifest.anti_leakage_status:
        raise ValueError("factor_research_receipt_anti_leakage_mismatch")
    if receipt.anti_leakage_violation_count != manifest.anti_leakage_violation_count:
        raise ValueError("factor_research_receipt_anti_leakage_count_mismatch")
    if receipt.completed_at != manifest.completed_at:
        raise ValueError("factor_research_receipt_completed_at_mismatch")


def _validate_alpha_frame_scope(
    lazy: pl.LazyFrame,
    *,
    dataset_name: str,
    task: AlphaFactoryTask,
) -> None:
    schema = lazy.collect_schema()
    if "as_of_date" in schema:
        null_count = int(
            lazy.select(pl.col("as_of_date").null_count()).collect(engine="streaming").item()
        )
        if null_count:
            raise ValueError(f"alpha_factory_result_scope_null:{dataset_name}:as_of_date")
        values = (
            lazy.select(pl.col("as_of_date").cast(pl.Utf8).unique())
            .collect(engine="streaming")
            .get_column("as_of_date")
            .to_list()
        )
        if values:
            try:
                scoped_days = {date.fromisoformat(str(value)) for value in values}
            except ValueError as exc:
                raise ValueError(
                    f"alpha_factory_result_scope_invalid:{dataset_name}:as_of_date"
                ) from exc
            if dataset_name in ALPHA_FACTORY_WINDOWED_AS_OF_DATASETS:
                first_allowed = task.as_of_date - timedelta(days=task.lookback_days)
                in_scope = all(
                    first_allowed <= scoped_day <= task.as_of_date for scoped_day in scoped_days
                )
            else:
                in_scope = scoped_days == {task.as_of_date}
            if not in_scope:
                raise ValueError(f"alpha_factory_result_scope_mismatch:{dataset_name}:as_of_date")
    if "candidate_id" in schema:
        null_count = int(
            lazy.select(pl.col("candidate_id").null_count()).collect(engine="streaming").item()
        )
        if null_count:
            raise ValueError(f"alpha_factory_result_null_candidate_id:{dataset_name}")


def _validate_lazy_unique_keys(
    lazy: pl.LazyFrame,
    keys: tuple[str, ...],
    dataset_name: str,
) -> None:
    schema = lazy.collect_schema()
    missing = [key for key in keys if key not in schema]
    if missing:
        raise ValueError(f"alpha_factory_result_primary_key_missing:{dataset_name}")
    null_counts = (
        lazy.select([pl.col(key).null_count().alias(key) for key in keys])
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if any(int(value or 0) for value in null_counts.values()):
        raise ValueError(f"alpha_factory_result_primary_key_null:{dataset_name}")
    duplicate = (
        lazy.group_by(list(keys))
        .len()
        .filter(pl.col("len") > 1)
        .limit(1)
        .collect(engine="streaming")
    )
    if not duplicate.is_empty():
        raise ValueError(f"alpha_factory_result_duplicate_primary_key:{dataset_name}")


def _validate_alpha_frame_safety(lazy: pl.LazyFrame, dataset_name: str) -> None:
    schema = lazy.collect_schema()
    forbidden_decisions = ALPHA_FACTORY_FORBIDDEN_LIVE_STATES
    if "max_live_notional_usdt" in schema:
        invalid = lazy.filter(pl.col("max_live_notional_usdt").fill_null(0.0) != 0.0).limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(f"alpha_factory_result_nonzero_live_notional:{dataset_name}")
    if "safety_mode" in schema:
        invalid = lazy.filter(pl.col("safety_mode").fill_null("") != "paper_shadow_only").limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(f"alpha_factory_result_unsafe_safety_mode:{dataset_name}")
    for column in ("decision", "candidate_state", "promotion_state", "recommended_mode"):
        if column not in schema:
            continue
        invalid = lazy.filter(
            pl.col(column).cast(pl.Utf8).str.to_uppercase().is_in(forbidden_decisions)
        ).limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(f"alpha_factory_result_live_state_forbidden:{dataset_name}:{column}")
    if "decision" in schema:
        allowed = ALPHA_FACTORY_DECISIONS_BY_DATASET.get(
            dataset_name,
            ALPHA_FACTORY_RESULT_DECISIONS,
        )
        decision_text = pl.col("decision").cast(pl.Utf8)
        invalid = lazy.filter(
            decision_text.is_null() | ~decision_text.is_in(sorted(allowed))
        ).limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(f"alpha_factory_result_unknown_decision:{dataset_name}")
    if "candidate_state" in schema:
        invalid = lazy.filter(pl.col("candidate_state") != "RESEARCH").limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(f"alpha_factory_result_unknown_candidate_state:{dataset_name}")
    if {"strategy_candidate", "decision"}.issubset(schema):
        invalid_futures = lazy.filter(
            pl.col("strategy_candidate").str.to_lowercase().str.contains("futures")
            & (pl.col("decision") == "PAPER_READY")
        ).limit(1)
        if not invalid_futures.collect(engine="streaming").is_empty():
            raise ValueError("alpha_factory_result_futures_proxy_paper_ready")
    if {"template_name", "decision"}.issubset(schema):
        invalid_bridge = lazy.filter(
            pl.col("template_name").str.to_lowercase().str.contains("factor_strategy_bridge")
            & (pl.col("decision") != "RESEARCH")
        ).limit(1)
        if not invalid_bridge.collect(engine="streaming").is_empty():
            raise ValueError("alpha_factory_result_factor_bridge_not_research")
    if {"template_name", "parameter_json"}.issubset(schema):
        bridge_rows = (
            lazy.filter(
                pl.col("template_name").str.to_lowercase().str.contains("factor_strategy_bridge")
            )
            .select("parameter_json")
            .collect(engine="streaming")
        )
        for row in bridge_rows.to_dicts():
            try:
                parameters = json.loads(str(row.get("parameter_json") or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("alpha_factory_result_factor_bridge_parameters_invalid") from exc
            if parameters.get("strategy_review_only") is not True:
                raise ValueError("alpha_factory_result_factor_bridge_not_strategy_review_only")
    if {"strategy_candidate", "futures_data_available"}.issubset(schema):
        invalid_futures_data = lazy.filter(
            pl.col("strategy_candidate").str.to_lowercase().str.contains("futures")
            & pl.col("futures_data_available").fill_null(True)
        ).limit(1)
        if not invalid_futures_data.collect(engine="streaming").is_empty():
            raise ValueError("alpha_factory_result_futures_data_claim_forbidden")
    if {"strategy_candidate", "funding_available"}.issubset(schema):
        invalid_funding = lazy.filter(
            pl.col("strategy_candidate").str.to_lowercase().str.contains("futures")
            & pl.col("funding_available").fill_null(True)
        ).limit(1)
        if not invalid_funding.collect(engine="streaming").is_empty():
            raise ValueError("alpha_factory_result_funding_claim_forbidden")


def _validate_alpha_candidate_result_identity(output_paths: dict[str, Path]) -> None:
    candidates = (
        pl.scan_parquet(output_paths["alpha_factory_candidate"])
        .select("candidate_id")
        .collect(engine="streaming")
        .get_column("candidate_id")
        .to_list()
    )
    results = (
        pl.scan_parquet(output_paths["alpha_factory_result"])
        .select("candidate_id")
        .collect(engine="streaming")
        .get_column("candidate_id")
        .to_list()
    )
    if set(candidates) != set(results) or len(candidates) != len(results):
        raise ValueError("alpha_factory_candidate_result_identity_mismatch")


def _validate_alpha_anti_leakage_report(
    payload: bytes,
    *,
    task: AlphaFactoryTask,
    snapshot: AlphaFactorySnapshotManifest,
) -> None:
    try:
        report = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("alpha_factory_anti_leakage_invalid_json") from exc
    if report.get("task_id") != task.task_id or report.get("snapshot_id") != snapshot.snapshot_id:
        raise ValueError("alpha_factory_anti_leakage_binding_mismatch")
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("alpha_factory_anti_leakage_missing")
    names = {str(item.get("check_name") or "") for item in checks if isinstance(item, dict)}
    from quant_lab.research_worker.alpha_factory import (  # noqa: PLC0415
        ALPHA_FACTORY_ANTI_LEAKAGE_CHECKS,
    )

    if names != set(ALPHA_FACTORY_ANTI_LEAKAGE_CHECKS):
        raise ValueError("alpha_factory_anti_leakage_incomplete")
    statuses = {str(item.get("status") or "").upper() for item in checks}
    violations = sum(int(item.get("violation_count") or 0) for item in checks)
    if report.get("status") != "PASS" or statuses != {"PASS"} or violations != 0:
        raise ValueError("alpha_factory_anti_leakage_failed")
    if report.get("violation_count") != 0:
        raise ValueError("alpha_factory_anti_leakage_count_mismatch")


def _validate_factor_bridge_report(path: Path) -> None:
    frame = pl.read_csv(path, infer_schema_length=None)
    if frame.is_empty():
        return
    if "live_order_effect" not in frame.columns:
        raise ValueError("alpha_factory_factor_bridge_live_effect_missing")
    invalid = frame.filter(
        ~pl.col("live_order_effect")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .is_in(["none", "none_read_only_research"])
    )
    if not invalid.is_empty():
        raise ValueError("alpha_factory_factor_bridge_live_effect_forbidden")
    if "eligible_for_alpha_factory" in frame.columns:
        invalid_eligible = frame.filter(
            pl.col("eligible_for_alpha_factory")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .is_in(["true", "1", "yes"])
        )
        if not invalid_eligible.is_empty():
            raise ValueError("alpha_factory_factor_bridge_direct_promotion_forbidden")


def _validate_alpha_worker_report(
    payload: bytes,
    *,
    task: AlphaFactoryTask,
    snapshot: AlphaFactorySnapshotManifest,
) -> None:
    try:
        report = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("alpha_factory_worker_report_invalid_json") from exc
    expected = {
        "task_id": task.task_id,
        "snapshot_id": snapshot.snapshot_id,
        "quant_lab_commit": task.quant_lab_commit,
        "template_registry_digest": task.template_registry_digest,
        "as_of_date": task.as_of_date.isoformat(),
        "factor_bridge_source": "snapshot_recompute",
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
    }
    if task.factor_generation_id is not None:
        expected.update(
            {
                "factor_generation_id": task.factor_generation_id,
                "factor_generation_digest": task.factor_generation_digest,
                "factor_generation_as_of_date": (
                    task.factor_generation_as_of_date.isoformat()
                    if task.factor_generation_as_of_date is not None
                    else None
                ),
                "factor_generation_published_at": (
                    task.factor_generation_published_at.isoformat()
                    if task.factor_generation_published_at is not None
                    else None
                ),
                "hypothesis_registry_digest": task.hypothesis_registry_digest,
                "trial_ledger_digest": task.trial_ledger_digest,
                "factor_generation_fresh": task.factor_generation_fresh,
                "factor_generation_hypothesis_ids": list(
                    task.factor_generation_hypothesis_ids or ()
                ),
            }
        )
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"alpha_factory_worker_report_mismatch:{field}")


def _validate_factor_research_scope(
    lazy: pl.LazyFrame,
    *,
    dataset_name: str,
    task: FactorResearchTask,
) -> None:
    schema = lazy.collect_schema()
    if "as_of_date" in schema:
        values = (
            lazy.select(pl.col("as_of_date").cast(pl.Utf8).unique())
            .collect(engine="streaming")
            .get_column("as_of_date")
            .to_list()
        )
        if values and set(values) != {task.as_of_date.isoformat()}:
            raise ValueError(f"factor_research_result_scope_mismatch:{dataset_name}:as_of_date")
    expected_data_snapshot_id = f"factor-input-{task.source_input_digest[:24]}"
    if "data_snapshot_id" in schema:
        values = (
            lazy.select(pl.col("data_snapshot_id").cast(pl.Utf8).unique())
            .collect(engine="streaming")
            .get_column("data_snapshot_id")
            .to_list()
        )
        if values and set(values) != {expected_data_snapshot_id}:
            raise ValueError(
                f"factor_research_result_scope_mismatch:{dataset_name}:data_snapshot_id"
            )
    if "hypothesis_id" in schema:
        values = set(
            lazy.select(pl.col("hypothesis_id").cast(pl.Utf8).unique())
            .collect(engine="streaming")
            .get_column("hypothesis_id")
            .to_list()
        )
        if not values.issubset(set(task.hypothesis_ids)):
            raise ValueError(
                f"factor_research_result_hypothesis_membership_mismatch:{dataset_name}"
            )
    if "trial_id" in schema:
        values = set(
            lazy.select(pl.col("trial_id").cast(pl.Utf8).unique())
            .collect(engine="streaming")
            .get_column("trial_id")
            .to_list()
        )
        if values != set(task.trial_ids):
            raise ValueError(f"factor_research_result_trial_membership_mismatch:{dataset_name}")


def _validate_factor_research_unique_keys(
    lazy: pl.LazyFrame,
    keys: tuple[str, ...],
    dataset_name: str,
) -> None:
    schema = lazy.collect_schema()
    if any(key not in schema for key in keys):
        raise ValueError(f"factor_research_result_primary_key_missing:{dataset_name}")
    nulls = (
        lazy.select([pl.col(key).null_count().alias(key) for key in keys])
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if any(int(value or 0) for value in nulls.values()):
        raise ValueError(f"factor_research_result_primary_key_null:{dataset_name}")
    duplicate = (
        lazy.group_by(list(keys))
        .len()
        .filter(pl.col("len") > 1)
        .limit(1)
        .collect(engine="streaming")
    )
    if not duplicate.is_empty():
        raise ValueError(f"factor_research_result_duplicate_primary_key:{dataset_name}")


def _validate_factor_research_safety(lazy: pl.LazyFrame, dataset_name: str) -> None:
    schema = lazy.collect_schema()
    required_literals: dict[str, Any] = {
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0.0,
    }
    for column, expected in required_literals.items():
        if column not in schema:
            continue
        invalid = lazy.filter(pl.col(column).is_null() | (pl.col(column) != expected)).limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(
                f"factor_research_result_safety_literal_mismatch:{dataset_name}:{column}"
            )
    for column in ("decision", "candidate_state", "deployment_readiness"):
        if column not in schema:
            continue
        invalid = lazy.filter(
            pl.col(column)
            .cast(pl.Utf8)
            .str.to_uppercase()
            .is_in(sorted(ALPHA_FACTORY_FORBIDDEN_LIVE_STATES))
        ).limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(f"factor_research_result_live_state_forbidden:{dataset_name}:{column}")
    if "decision" in schema:
        invalid = lazy.filter(
            pl.col("decision").is_null()
            | ~pl.col("decision").cast(pl.Utf8).is_in(sorted(FACTOR_RESEARCH_ALLOWED_DECISIONS))
        ).limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(f"factor_research_result_unknown_decision:{dataset_name}")
    if "candidate_state" in schema:
        invalid = lazy.filter(
            pl.col("candidate_state").is_null()
            | ~pl.col("candidate_state")
            .cast(pl.Utf8)
            .is_in(sorted(FACTOR_RESEARCH_ALLOWED_CANDIDATE_STATES))
        ).limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(f"factor_research_result_unknown_candidate_state:{dataset_name}")


def _validate_factor_research_membership(
    output_paths: dict[str, Path],
    *,
    task: FactorResearchTask,
) -> None:
    definitions = pl.scan_parquet(output_paths["factor_definition"])
    definition_hypotheses = set(
        definitions.select("hypothesis_id")
        .collect(engine="streaming")
        .get_column("hypothesis_id")
        .to_list()
    )
    if definition_hypotheses != set(task.hypothesis_ids):
        raise ValueError("factor_research_result_definition_hypothesis_mismatch")
    factor_ids = set(
        definitions.select("factor_id")
        .collect(engine="streaming")
        .get_column("factor_id")
        .to_list()
    )
    if not factor_ids or len(factor_ids) > 54:
        raise ValueError("factor_research_result_factor_count_invalid")
    for dataset_name in (
        "factor_value",
        "factor_evidence",
        "factor_attribution",
        "factor_portfolio_validation",
        "factor_candidate",
    ):
        observed = set(
            pl.scan_parquet(output_paths[dataset_name])
            .select("factor_id")
            .collect(engine="streaming")
            .get_column("factor_id")
            .to_list()
        )
        if not observed.issubset(factor_ids):
            raise ValueError(f"factor_research_result_factor_identity_mismatch:{dataset_name}")
    evidence_rows = int(
        pl.scan_parquet(output_paths["factor_evidence"])
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if evidence_rows != task.test_count:
        raise ValueError("factor_research_result_evidence_count_mismatch")
    candidate_count = int(
        pl.scan_parquet(output_paths["factor_candidate"])
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if candidate_count > len(factor_ids):
        raise ValueError("factor_research_result_candidate_limit_exceeded")


def _validate_factor_research_trial_ledger(
    snapshot_root: Path,
    *,
    task: FactorResearchTask,
    snapshot: FactorResearchSnapshotManifest,
) -> None:
    ledger = read_parquet_dataset(snapshot_root / "files" / RESEARCH_TRIAL_LEDGER_DATASET)
    if trial_ledger_digest(ledger) != snapshot.trial_ledger_digest:
        raise ValueError("factor_research_result_trial_ledger_digest_mismatch")
    if set(ledger.get_column("trial_id").to_list()) != set(task.trial_ids):
        raise ValueError("factor_research_result_trial_ledger_membership_mismatch")
    if set(ledger.get_column("nas_task_id").to_list()) != {task.task_id}:
        raise ValueError("factor_research_result_trial_ledger_task_mismatch")


def _validate_factor_research_anti_leakage_report(
    payload: bytes,
    *,
    task: FactorResearchTask,
    snapshot: FactorResearchSnapshotManifest,
) -> None:
    try:
        report = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("factor_research_anti_leakage_invalid_json") from exc
    if report.get("task_id") != task.task_id or report.get("snapshot_id") != snapshot.snapshot_id:
        raise ValueError("factor_research_anti_leakage_binding_mismatch")
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("factor_research_anti_leakage_missing")
    from quant_lab.research_worker.factor_research import (  # noqa: PLC0415
        FACTOR_RESEARCH_ANTI_LEAKAGE_CHECKS,
    )

    names = {str(item.get("check_name") or "") for item in checks if isinstance(item, dict)}
    if names != set(FACTOR_RESEARCH_ANTI_LEAKAGE_CHECKS):
        raise ValueError("factor_research_anti_leakage_incomplete")
    statuses = {str(item.get("status") or "").upper() for item in checks}
    violations = sum(int(item.get("violation_count") or 0) for item in checks)
    if (
        report.get("status") != "PASS"
        or report.get("violation_count") != 0
        or statuses != {"PASS"}
        or violations != 0
    ):
        raise ValueError("factor_research_anti_leakage_failed")
    if report.get("research_only") is not True or report.get("live_order_effect") != "none":
        raise ValueError("factor_research_anti_leakage_safety_mismatch")


def _validate_factor_research_worker_report(
    payload: bytes,
    *,
    task: FactorResearchTask,
    snapshot: FactorResearchSnapshotManifest,
    outputs: dict[str, Any],
) -> None:
    try:
        report = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("factor_research_worker_report_invalid_json") from exc
    expected = {
        "task_id": task.task_id,
        "snapshot_id": snapshot.snapshot_id,
        "source_input_digest": task.source_input_digest,
        "hypothesis_registry_digest": task.hypothesis_registry_digest,
        "trial_ledger_digest": task.trial_ledger_digest,
        "hypothesis_count": len(task.hypothesis_ids),
        "test_count": task.test_count,
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"factor_research_worker_report_mismatch:{field}")
    row_counts = report.get("output_rows")
    expected_rows = {name: item.row_count for name, item in outputs.items()}
    if row_counts != expected_rows:
        raise ValueError("factor_research_worker_report_output_rows_mismatch")


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
        series = frame.get_column(column)
        if series.null_count():
            raise ValueError(f"research_result_scope_null:{dataset_name}:{column}")
        values = {str(value) for value in series.to_list()}
        if values != {expected}:
            raise ValueError(f"research_result_scope_mismatch:{dataset_name}:{column}")
    if "window_hours" in frame.columns and not frame.is_empty():
        values = {int(value) for value in frame.get_column("window_hours").drop_nulls().to_list()}
        if values != {task.window_hours}:
            raise ValueError(f"research_result_scope_mismatch:{dataset_name}:window_hours")
    if "quant_lab_git_commit" in frame.columns and not frame.is_empty():
        commit_series = frame.get_column("quant_lab_git_commit")
        if commit_series.null_count():
            raise ValueError(f"research_result_scope_null:{dataset_name}:quant_lab_git_commit")
        values = {str(value) for value in commit_series.cast(pl.Utf8).to_list()}
        if values != {task.quant_lab_commit}:
            raise ValueError(f"research_result_scope_mismatch:{dataset_name}:quant_lab_git_commit")
    if "source_version" in frame.columns and not frame.is_empty():
        source_series = frame.get_column("source_version")
        if source_series.null_count():
            raise ValueError(f"research_result_scope_null:{dataset_name}:source_version")
        expected_source_version = f"entry_quality:{task.quant_lab_commit}"
        values = {str(value) for value in source_series.cast(pl.Utf8).to_list()}
        if values != {expected_source_version}:
            raise ValueError(f"research_result_scope_mismatch:{dataset_name}:source_version")
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
