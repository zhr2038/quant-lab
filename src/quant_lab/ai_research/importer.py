from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.ai_research.contracts import (
    LIVE_ORDER_EFFECT,
    AIResearchResult,
    AIResearchTask,
    canonical_json,
    compute_task_packet_sha256,
)
from quant_lab.data.lake import upsert_parquet_dataset

AI_RUN_DATASET = Path("gold") / "ai_research_run"
AI_FINDING_DATASET = Path("gold") / "ai_research_finding"
AI_FACTOR_PROPOSAL_DATASET = Path("gold") / "ai_factor_proposal"
AI_PAPER_DRAFT_DATASET = Path("gold") / "ai_paper_strategy_draft"
AI_EXPERIMENT_DATASET = Path("gold") / "ai_experiment_proposal"
AI_CODE_REVIEW_DATASET = Path("gold") / "ai_code_review_target"

AI_RUN_SCHEMA: dict[str, Any] = {
    "task_id": pl.Utf8,
    "source_pack_sha256": pl.Utf8,
    "packet_sha256": pl.Utf8,
    "model": pl.Utf8,
    "reasoning_effort": pl.Utf8,
    "worker_id": pl.Utf8,
    "started_at": pl.Datetime(time_zone="UTC"),
    "completed_at": pl.Datetime(time_zone="UTC"),
    "system_state": pl.Utf8,
    "stage2_allowed": pl.Boolean,
    "executive_summary": pl.Utf8,
    "route_sections_json": pl.Utf8,
    "prompt_version": pl.Utf8,
    "source_pack_name": pl.Utf8,
    "preflight_status": pl.Utf8,
    "preflight_checked_at": pl.Datetime(time_zone="UTC"),
    "preflight_blockers_json": pl.Utf8,
    "preflight_warnings_json": pl.Utf8,
    "primary_bottleneck_id": pl.Utf8,
    "root_cause_tree_json": pl.Utf8,
    "next_actions_json": pl.Utf8,
    "continuity_status": pl.Utf8,
    "continuity_json": pl.Utf8,
    "stage1_attempts": pl.Int64,
    "stage2_attempts": pl.Int64,
    "validation_events_json": pl.Utf8,
    "finding_count": pl.Int64,
    "factor_proposal_count": pl.Int64,
    "paper_draft_count": pl.Int64,
    "experiment_count": pl.Int64,
    "code_review_target_count": pl.Int64,
    "usage_json": pl.Utf8,
    "warnings_json": pl.Utf8,
    "schema_version": pl.Utf8,
    "diagnostic_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

AI_FINDING_SCHEMA: dict[str, Any] = {
    "task_id": pl.Utf8,
    "finding_group": pl.Utf8,
    "finding_id": pl.Utf8,
    "category": pl.Utf8,
    "status": pl.Utf8,
    "severity": pl.Utf8,
    "summary": pl.Utf8,
    "explanation": pl.Utf8,
    "confidence": pl.Float64,
    "recommended_action": pl.Utf8,
    "evidence_refs_json": pl.Utf8,
    "model": pl.Utf8,
    "completed_at": pl.Datetime(time_zone="UTC"),
    "diagnostic_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "source": pl.Utf8,
}

AI_FACTOR_SCHEMA: dict[str, Any] = {
    "task_id": pl.Utf8,
    "proposal_id": pl.Utf8,
    "factor_name": pl.Utf8,
    "factor_family": pl.Utf8,
    "description": pl.Utf8,
    "template": pl.Utf8,
    "input_features_json": pl.Utf8,
    "parameters_json": pl.Utf8,
    "direction": pl.Int64,
    "lookback_bars": pl.Int64,
    "availability_lag_bars": pl.Int64,
    "expected_horizon_bars_json": pl.Utf8,
    "hypothesis": pl.Utf8,
    "economic_rationale": pl.Utf8,
    "falsification_conditions_json": pl.Utf8,
    "evidence_refs_json": pl.Utf8,
    "known_overlap_risk": pl.Utf8,
    "research_thread_id": pl.Utf8,
    "source_finding_ids_json": pl.Utf8,
    "model": pl.Utf8,
    "completed_at": pl.Datetime(time_zone="UTC"),
    "proposal_state": pl.Utf8,
    "requires_human_review": pl.Boolean,
    "diagnostic_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "source": pl.Utf8,
}

AI_PAPER_DRAFT_SCHEMA: dict[str, Any] = {
    "task_id": pl.Utf8,
    "draft_id": pl.Utf8,
    "strategy_family": pl.Utf8,
    "symbols_json": pl.Utf8,
    "timeframe": pl.Utf8,
    "direction": pl.Utf8,
    "entry_match": pl.Utf8,
    "entry_clauses_json": pl.Utf8,
    "exit_match": pl.Utf8,
    "exit_clauses_json": pl.Utf8,
    "max_holding_bars": pl.Int64,
    "min_holding_bars": pl.Int64,
    "cooldown_bars": pl.Int64,
    "required_market_fields_json": pl.Utf8,
    "hypothesis": pl.Utf8,
    "falsification_conditions_json": pl.Utf8,
    "evidence_refs_json": pl.Utf8,
    "mode": pl.Utf8,
    "research_thread_id": pl.Utf8,
    "source_finding_ids_json": pl.Utf8,
    "model": pl.Utf8,
    "completed_at": pl.Datetime(time_zone="UTC"),
    "proposal_state": pl.Utf8,
    "requires_human_review": pl.Boolean,
    "diagnostic_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "source": pl.Utf8,
}

AI_EXPERIMENT_SCHEMA: dict[str, Any] = {
    "task_id": pl.Utf8,
    "proposal_id": pl.Utf8,
    "objective": pl.Utf8,
    "hypothesis": pl.Utf8,
    "control": pl.Utf8,
    "treatment": pl.Utf8,
    "required_datasets_json": pl.Utf8,
    "success_metrics_json": pl.Utf8,
    "minimum_complete_samples": pl.Int64,
    "mode": pl.Utf8,
    "risks_json": pl.Utf8,
    "evidence_refs_json": pl.Utf8,
    "research_thread_id": pl.Utf8,
    "source_finding_ids_json": pl.Utf8,
    "falsification_conditions_json": pl.Utf8,
    "stopping_conditions_json": pl.Utf8,
    "regime_slices_json": pl.Utf8,
    "model": pl.Utf8,
    "completed_at": pl.Datetime(time_zone="UTC"),
    "proposal_state": pl.Utf8,
    "requires_human_review": pl.Boolean,
    "diagnostic_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "source": pl.Utf8,
}

AI_CODE_REVIEW_SCHEMA: dict[str, Any] = {
    "task_id": pl.Utf8,
    "target_id": pl.Utf8,
    "repository": pl.Utf8,
    "path_or_component": pl.Utf8,
    "reason": pl.Utf8,
    "expected_evidence": pl.Utf8,
    "priority": pl.Utf8,
    "source_finding_ids_json": pl.Utf8,
    "origin_stage": pl.Utf8,
    "model": pl.Utf8,
    "completed_at": pl.Datetime(time_zone="UTC"),
    "requires_human_review": pl.Boolean,
    "diagnostic_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "source": pl.Utf8,
}


def import_ai_research_results(
    queue_root: str | Path,
    *,
    lake_root: str | Path,
    max_results: int = 20,
) -> dict[str, Any]:
    queue = Path(queue_root)
    inbox = queue / "results" / "inbox"
    imported = queue / "results" / "imported"
    rejected = queue / "results" / "rejected"
    imported.mkdir(parents=True, exist_ok=True)
    rejected.mkdir(parents=True, exist_ok=True)

    candidates = (
        sorted(
            (
                item
                for item in inbox.iterdir()
                if item.is_dir() and (item / "result.json").is_file()
            ),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        if inbox.exists()
        else []
    )

    summary: dict[str, Any] = {
        "queue_root": str(queue),
        "lake_root": str(Path(lake_root)),
        "examined": 0,
        "imported": 0,
        "rejected": 0,
        "task_ids": [],
        "errors": [],
    }

    for result_dir in candidates[: max(0, int(max_results))]:
        summary["examined"] += 1
        try:
            result = _load_result(result_dir / "result.json")
            task = _load_task_for_result(queue, result.task_id)
            _validate_result_against_task(result, task)
            _publish_result(result, task=task, lake_root=Path(lake_root))
            _atomic_write_json(
                queue / "state" / "latest_research_context.json",
                _research_context_payload(result),
            )
            destination = imported / result.task_id
            _replace_directory(result_dir, destination)
            _atomic_write_json(
                destination / "import_summary.json",
                {
                    "task_id": result.task_id,
                    "imported_at": datetime.now(UTC).isoformat(),
                    "model": result.model,
                    "system_state": result.diagnosis.system_state,
                    "factor_proposals": len(result.proposals.factor_proposals)
                    if result.proposals
                    else 0,
                    "paper_strategy_drafts": len(result.proposals.paper_strategy_drafts)
                    if result.proposals
                    else 0,
                    "experiment_proposals": len(result.proposals.experiment_proposals)
                    if result.proposals
                    else 0,
                    "diagnostic_only": True,
                    "live_order_effect": LIVE_ORDER_EFFECT,
                },
            )
            summary["imported"] += 1
            summary["task_ids"].append(result.task_id)
        except Exception as exc:  # noqa: BLE001 - result isolation is intentional
            destination = rejected / result_dir.name
            _replace_directory(result_dir, destination)
            _atomic_write_json(
                destination / "import_error.json",
                {
                    "rejected_at": datetime.now(UTC).isoformat(),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            summary["rejected"] += 1
            summary["errors"].append(
                {
                    "task_id": result_dir.name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return summary


def _publish_result(
    result: AIResearchResult,
    *,
    task: AIResearchTask,
    lake_root: Path,
) -> None:
    completed = result.completed_at.astimezone(UTC)
    effective_preflight = result.effective_preflight or task.preflight
    proposals = result.proposals
    finding_groups = {
        "primary_bottleneck": result.diagnosis.primary_bottlenecks,
        "contradiction": result.diagnosis.contradictions,
        "missing_evidence": result.diagnosis.missing_evidence,
    }
    finding_count = sum(len(items) for items in finding_groups.values())
    stage2_code_targets = proposals.code_review_targets if proposals else []
    code_review_target_count = len(result.diagnosis.code_review_targets) + len(
        stage2_code_targets
    )

    run_rows = [
        {
            "task_id": result.task_id,
            "source_pack_sha256": result.source_pack_sha256,
            "packet_sha256": result.packet_sha256,
            "model": result.model,
            "reasoning_effort": result.reasoning_effort,
            "worker_id": result.worker_id,
            "started_at": result.started_at.astimezone(UTC),
            "completed_at": completed,
            "system_state": result.diagnosis.system_state,
            "stage2_allowed": result.diagnosis.stage2_allowed,
            "executive_summary": result.diagnosis.executive_summary,
            "route_sections_json": canonical_json(result.diagnosis.route_sections),
            "prompt_version": result.prompt_version,
            "source_pack_name": task.source_pack_name,
            "preflight_status": (
                effective_preflight.status if effective_preflight else "NOT_AVAILABLE"
            ),
            "preflight_checked_at": (
                effective_preflight.checked_at.astimezone(UTC)
                if effective_preflight
                else None
            ),
            "preflight_blockers_json": canonical_json(
                effective_preflight.blockers if effective_preflight else []
            ),
            "preflight_warnings_json": canonical_json(
                effective_preflight.warnings if effective_preflight else []
            ),
            "primary_bottleneck_id": result.diagnosis.primary_bottleneck_id,
            "root_cause_tree_json": canonical_json(
                [item.model_dump(mode="json") for item in result.diagnosis.root_cause_tree]
            ),
            "next_actions_json": canonical_json(
                [item.model_dump(mode="json") for item in result.diagnosis.next_actions]
            ),
            "continuity_status": result.diagnosis.continuity.status,
            "continuity_json": canonical_json(
                result.diagnosis.continuity.model_dump(mode="json")
            ),
            "stage1_attempts": result.stage1_attempts,
            "stage2_attempts": result.stage2_attempts,
            "validation_events_json": result.validation_events_json,
            "finding_count": finding_count,
            "factor_proposal_count": len(proposals.factor_proposals) if proposals else 0,
            "paper_draft_count": len(proposals.paper_strategy_drafts) if proposals else 0,
            "experiment_count": len(proposals.experiment_proposals) if proposals else 0,
            "code_review_target_count": code_review_target_count,
            "usage_json": result.usage_json,
            "warnings_json": canonical_json(result.warnings),
            "schema_version": result.schema_version,
            "diagnostic_only": True,
            "live_order_effect": LIVE_ORDER_EFFECT,
            "created_at": datetime.now(UTC),
            "source": "ai_research.importer.v2",
        }
    ]
    _upsert_rows(
        run_rows,
        schema=AI_RUN_SCHEMA,
        path=lake_root / AI_RUN_DATASET,
        keys=["task_id"],
    )

    finding_rows: list[dict[str, Any]] = []
    for group_name, findings in finding_groups.items():
        for finding in findings:
            finding_rows.append(
                {
                    "task_id": result.task_id,
                    "finding_group": group_name,
                    "finding_id": finding.finding_id,
                    "category": finding.category,
                    "status": finding.status,
                    "severity": finding.severity,
                    "summary": finding.summary,
                    "explanation": finding.explanation,
                    "confidence": finding.confidence,
                    "recommended_action": finding.recommended_action,
                    "evidence_refs_json": canonical_json(
                        [item.model_dump(mode="json") for item in finding.evidence_refs]
                    ),
                    "model": result.model,
                    "completed_at": completed,
                    "diagnostic_only": True,
                    "live_order_effect": LIVE_ORDER_EFFECT,
                    "source": "ai_research.importer.v2",
                }
            )
    _upsert_rows(
        finding_rows,
        schema=AI_FINDING_SCHEMA,
        path=lake_root / AI_FINDING_DATASET,
        keys=["task_id", "finding_group", "finding_id"],
    )

    factor_proposals = proposals.factor_proposals if proposals else []
    paper_strategy_drafts = proposals.paper_strategy_drafts if proposals else []
    experiment_proposals = proposals.experiment_proposals if proposals else []

    factor_rows = [
        {
            "task_id": result.task_id,
            "proposal_id": item.proposal_id,
            "factor_name": item.factor_name,
            "factor_family": item.factor_family,
            "description": item.description,
            "template": item.template,
            "input_features_json": canonical_json(item.input_features),
            "parameters_json": canonical_json(
                [parameter.model_dump(mode="json") for parameter in item.parameters]
            ),
            "direction": item.direction,
            "lookback_bars": item.lookback_bars,
            "availability_lag_bars": item.availability_lag_bars,
            "expected_horizon_bars_json": canonical_json(item.expected_horizon_bars),
            "hypothesis": item.hypothesis,
            "economic_rationale": item.economic_rationale,
            "falsification_conditions_json": canonical_json(item.falsification_conditions),
            "evidence_refs_json": canonical_json(
                [reference.model_dump(mode="json") for reference in item.evidence_refs]
            ),
            "known_overlap_risk": item.known_overlap_risk,
            "research_thread_id": item.research_thread_id,
            "source_finding_ids_json": canonical_json(item.source_finding_ids),
            "model": result.model,
            "completed_at": completed,
            "proposal_state": "AI_RESEARCH_DRAFT",
            "requires_human_review": True,
            "diagnostic_only": True,
            "live_order_effect": LIVE_ORDER_EFFECT,
            "source": "ai_research.importer.v2",
        }
        for item in factor_proposals
    ]
    _upsert_rows(
        factor_rows,
        schema=AI_FACTOR_SCHEMA,
        path=lake_root / AI_FACTOR_PROPOSAL_DATASET,
        keys=["task_id", "proposal_id"],
    )

    paper_rows = [
        {
            "task_id": result.task_id,
            "draft_id": item.draft_id,
            "strategy_family": item.strategy_family,
            "symbols_json": canonical_json(item.symbols),
            "timeframe": item.timeframe,
            "direction": item.direction,
            "entry_match": item.entry_match,
            "entry_clauses_json": canonical_json(
                [clause.model_dump(mode="json") for clause in item.entry_clauses]
            ),
            "exit_match": item.exit_match,
            "exit_clauses_json": canonical_json(
                [clause.model_dump(mode="json") for clause in item.exit_clauses]
            ),
            "max_holding_bars": item.max_holding_bars,
            "min_holding_bars": item.min_holding_bars,
            "cooldown_bars": item.cooldown_bars,
            "required_market_fields_json": canonical_json(item.required_market_fields),
            "hypothesis": item.hypothesis,
            "falsification_conditions_json": canonical_json(item.falsification_conditions),
            "evidence_refs_json": canonical_json(
                [reference.model_dump(mode="json") for reference in item.evidence_refs]
            ),
            "mode": item.mode,
            "research_thread_id": item.research_thread_id,
            "source_finding_ids_json": canonical_json(item.source_finding_ids),
            "model": result.model,
            "completed_at": completed,
            "proposal_state": "AI_RESEARCH_DRAFT",
            "requires_human_review": True,
            "diagnostic_only": True,
            "live_order_effect": LIVE_ORDER_EFFECT,
            "source": "ai_research.importer.v2",
        }
        for item in paper_strategy_drafts
    ]
    _upsert_rows(
        paper_rows,
        schema=AI_PAPER_DRAFT_SCHEMA,
        path=lake_root / AI_PAPER_DRAFT_DATASET,
        keys=["task_id", "draft_id"],
    )

    experiment_rows = [
        {
            "task_id": result.task_id,
            "proposal_id": item.proposal_id,
            "objective": item.objective,
            "hypothesis": item.hypothesis,
            "control": item.control,
            "treatment": item.treatment,
            "required_datasets_json": canonical_json(item.required_datasets),
            "success_metrics_json": canonical_json(item.success_metrics),
            "minimum_complete_samples": item.minimum_complete_samples,
            "mode": item.mode,
            "risks_json": canonical_json(item.risks),
            "evidence_refs_json": canonical_json(
                [reference.model_dump(mode="json") for reference in item.evidence_refs]
            ),
            "research_thread_id": item.research_thread_id,
            "source_finding_ids_json": canonical_json(item.source_finding_ids),
            "falsification_conditions_json": canonical_json(item.falsification_conditions),
            "stopping_conditions_json": canonical_json(item.stopping_conditions),
            "regime_slices_json": canonical_json(item.regime_slices),
            "model": result.model,
            "completed_at": completed,
            "proposal_state": "AI_RESEARCH_DRAFT",
            "requires_human_review": True,
            "diagnostic_only": True,
            "live_order_effect": LIVE_ORDER_EFFECT,
            "source": "ai_research.importer.v2",
        }
        for item in experiment_proposals
    ]
    _upsert_rows(
        experiment_rows,
        schema=AI_EXPERIMENT_SCHEMA,
        path=lake_root / AI_EXPERIMENT_DATASET,
        keys=["task_id", "proposal_id"],
    )

    stage1_code_targets = [
        (item, "STAGE1_DIAGNOSTIC") for item in result.diagnosis.code_review_targets
    ]
    stage2_code_targets_with_origin = [
        (item, "STAGE2_RESEARCH") for item in stage2_code_targets
    ]
    code_rows = [
        {
            "task_id": result.task_id,
            "target_id": item.target_id,
            "repository": item.repository,
            "path_or_component": item.path_or_component,
            "reason": item.reason,
            "expected_evidence": item.expected_evidence,
            "priority": item.priority,
            "source_finding_ids_json": canonical_json(item.source_finding_ids),
            "origin_stage": origin_stage,
            "model": result.model,
            "completed_at": completed,
            "requires_human_review": True,
            "diagnostic_only": True,
            "live_order_effect": LIVE_ORDER_EFFECT,
            "source": "ai_research.importer.v2",
        }
        for item, origin_stage in stage1_code_targets + stage2_code_targets_with_origin
    ]
    _upsert_rows(
        code_rows,
        schema=AI_CODE_REVIEW_SCHEMA,
        path=lake_root / AI_CODE_REVIEW_DATASET,
        keys=["task_id", "target_id"],
    )


def _research_context_payload(result: AIResearchResult) -> dict[str, Any]:
    findings = (
        result.diagnosis.primary_bottlenecks
        + result.diagnosis.contradictions
        + result.diagnosis.missing_evidence
    )
    return {
        "task_id": result.task_id,
        "completed_at": result.completed_at.astimezone(UTC).isoformat(),
        "system_state": result.diagnosis.system_state,
        "executive_summary": result.diagnosis.executive_summary,
        "findings": [
            {
                "finding_id": item.finding_id,
                "category": item.category,
                "severity": item.severity,
                "summary": item.summary,
            }
            for item in findings[:24]
        ],
        "next_action_ids": [item.action_id for item in result.diagnosis.next_actions],
    }


def _upsert_rows(
    rows: list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    path: Path,
    keys: list[str],
) -> None:
    if not rows:
        return
    frame = (
        pl.DataFrame(rows, infer_schema_length=None)
        .cast(schema, strict=False)
        .select(list(schema))
    )
    upsert_parquet_dataset(frame, path, keys)


def _load_result(path: Path) -> AIResearchResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AIResearchResult.model_validate(payload)


def _load_task_for_result(queue_root: Path, task_id: str) -> AIResearchTask:
    for state in ("completed", "running", "pending", "failed"):
        candidate = queue_root / state / task_id / "task.json"
        if candidate.is_file():
            return AIResearchTask.model_validate_json(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"task.json not found for result {task_id}")


def _validate_result_against_task(result: AIResearchResult, task: AIResearchTask) -> None:
    computed_packet_sha = compute_task_packet_sha256(task)
    if computed_packet_sha != task.packet_sha256:
        raise ValueError("stored task packet_sha256 is invalid")
    if result.task_id != task.task_id:
        raise ValueError("result task_id does not match task")
    if result.source_pack_sha256 != task.source_pack_sha256:
        raise ValueError("result source_pack_sha256 does not match task")
    if result.packet_sha256 != task.packet_sha256:
        raise ValueError("result packet_sha256 does not match task")
    effective_preflight = result.effective_preflight or task.preflight
    if task.source_location == "nas_accepted" and result.effective_preflight is None:
        raise ValueError("NAS result is missing the materialized effective preflight")
    if (
        task.preflight is not None
        and result.effective_preflight is not None
        and result.effective_preflight != task.preflight
    ):
        raise ValueError("result effective preflight does not match embedded task preflight")
    if (
        effective_preflight
        and effective_preflight.status == "BLOCK"
        and result.diagnosis.stage2_allowed
    ):
        raise ValueError("blocked deterministic preflight cannot enter Stage 2")
    previous_context = task.previous_research_context
    continuity = result.diagnosis.continuity
    if previous_context is None:
        if continuity.status != "FIRST_RUN" or continuity.previous_task_id is not None:
            raise ValueError("continuity must be FIRST_RUN without previous research context")
    elif continuity.previous_task_id != previous_context.task_id:
        raise ValueError("continuity previous_task_id does not match task context")
    evidence_members = _result_evidence_members(result, task)
    if effective_preflight and effective_preflight.available_sections:
        if set(effective_preflight.available_sections) != set(evidence_members):
            raise ValueError(
                "effective preflight sections do not match materialized evidence manifest"
            )
    available_sections = set(evidence_members)
    routed = set(result.diagnosis.route_sections)
    unknown_sections = routed - available_sections
    if unknown_sections:
        raise ValueError(f"diagnosis routed unknown sections: {sorted(unknown_sections)}")
    stage1_references = [
            reference
            for finding in (
                *result.diagnosis.primary_bottlenecks,
                *result.diagnosis.contradictions,
                *result.diagnosis.missing_evidence,
            )
            for reference in finding.evidence_refs
        ]
    stage1_references.extend(
        reference
        for node in result.diagnosis.root_cause_tree
        for reference in node.evidence_refs
    )
    stage1_references.extend(
        reference
        for action in result.diagnosis.next_actions
        for reference in action.evidence_refs
    )
    _validate_evidence_references(
        stage1_references,
        task=task,
        allowed_sections=available_sections,
        evidence_members=evidence_members,
    )
    if result.proposals is None:
        return
    allowed_stage2_sections = set(result.diagnosis.route_sections)
    if "core_state" in available_sections:
        allowed_stage2_sections.add("core_state")
    for proposal in result.proposals.factor_proposals:
        if proposal.template not in task.allowed_factor_templates:
            raise ValueError(f"factor template not allowed by task: {proposal.template}")
    stage2_references = [
        reference
        for proposal in (
            *result.proposals.factor_proposals,
            *result.proposals.paper_strategy_drafts,
            *result.proposals.experiment_proposals,
        )
        for reference in proposal.evidence_refs
    ]
    _validate_evidence_references(
        stage2_references,
        task=task,
        allowed_sections=allowed_stage2_sections,
        evidence_members=evidence_members,
    )


def _validate_evidence_references(
    references: list[Any],
    *,
    task: AIResearchTask,
    allowed_sections: set[str],
    evidence_members: dict[str, set[str]] | None = None,
) -> None:
    members_by_section = evidence_members or {
        section: {document.source_member for document in documents}
        for section, documents in task.sections.items()
    }
    for reference in references:
        if reference.section not in allowed_sections:
            raise ValueError(f"evidence section was not routed: {reference.section}")
        if reference.source_member not in members_by_section.get(reference.section, set()):
            raise ValueError(
                "evidence source member is not present in the task: "
                f"{reference.section}/{reference.source_member}"
            )


def _result_evidence_members(
    result: AIResearchResult,
    task: AIResearchTask,
) -> dict[str, set[str]]:
    if task.sections:
        return {
            section: {document.source_member for document in documents}
            for section, documents in task.sections.items()
        }
    if task.source_location != "nas_accepted" or not result.evidence_manifest:
        raise ValueError("NAS AI result is missing its bounded evidence manifest")
    members: dict[str, set[str]] = {}
    for item in result.evidence_manifest:
        members.setdefault(item.section, set()).add(item.source_member)
    return members


def _replace_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(source, destination)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass
