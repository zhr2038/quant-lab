from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_lab.ai_research.contracts import (
    AIResearchResult,
    AIResearchTask,
    CodeReviewTarget,
    EvidenceDocument,
    EvidenceManifestEntry,
    EvidenceReference,
    FactorProposal,
    ResearchFinding,
    Stage1Diagnosis,
    Stage2ProposalSet,
    TaskPreflight,
    compute_task_packet_sha256,
)
from quant_lab.ai_research.importer import (
    AI_CODE_REVIEW_DATASET,
    AI_RUN_DATASET,
    _publish_result,
    _validate_result_against_task,
    import_ai_research_results,
)
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset


def _task() -> AIResearchTask:
    provisional = AIResearchTask(
        task_id="task-import",
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
        source_pack_name="expert.zip",
        source_pack_sha256="a" * 64,
        packet_sha256="0" * 64,
        sections={
            "factor_research": [
                EvidenceDocument(
                    source_member="reports/factor_evidence.csv",
                    source_format="csv",
                    content_sha256="b" * 64,
                    source_size_bytes=10,
                    content={"rows": []},
                )
            ]
        },
        allowed_factor_templates=["feature"],
    )
    return provisional.model_copy(
        update={"packet_sha256": compute_task_packet_sha256(provisional)}
    )


def _reference(source_member: str = "reports/factor_evidence.csv") -> EvidenceReference:
    return EvidenceReference(
        section="factor_research",
        source_member=source_member,
        fields=["rank_ic_mean"],
        claim="Cited evidence.",
    )


def _result(task: AIResearchTask, reference: EvidenceReference) -> AIResearchResult:
    diagnosis = Stage1Diagnosis(
        task_id=task.task_id,
        system_state="READY_FOR_PROPOSALS",
        executive_summary="Evidence is sufficient for a draft.",
        stage2_allowed=True,
        route_sections=["factor_research"],
        primary_bottlenecks=[
            ResearchFinding(
                finding_id="finding-1",
                category="factor_research",
                status="observed",
                severity="info",
                summary="Observed evidence.",
                explanation="The source row is cited.",
                confidence=0.8,
                evidence_refs=[reference],
                recommended_action="Run a falsifiable Paper experiment.",
            )
        ],
    )
    proposals = Stage2ProposalSet(
        task_id=task.task_id,
        executive_summary="One diagnostic factor draft.",
        factor_proposals=[
            FactorProposal(
                proposal_id="factor-1",
                factor_name="test_factor",
                factor_family="test",
                description="Diagnostic draft.",
                template="feature",
                input_features=["close_return_4"],
                direction=1,
                lookback_bars=4,
                availability_lag_bars=1,
                expected_horizon_bars=[4],
                hypothesis="The feature may retain after-cost information.",
                economic_rationale="Test only.",
                falsification_conditions=["after-cost return is non-positive"],
                evidence_refs=[reference],
                known_overlap_risk="May overlap with return factors.",
                research_thread_id="thread-factor-1",
                source_finding_ids=["finding-1"],
            )
        ],
    )
    return AIResearchResult(
        task_id=task.task_id,
        source_pack_sha256=task.source_pack_sha256,
        packet_sha256=task.packet_sha256,
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        worker_id="nas-1",
        started_at=datetime(2026, 7, 14, 1, tzinfo=UTC),
        completed_at=datetime(2026, 7, 14, 2, tzinfo=UTC),
        diagnosis=diagnosis,
        proposals=proposals,
    )


def test_import_validation_accepts_exact_evidence_members() -> None:
    task = _task()
    _validate_result_against_task(_result(task, _reference()), task)


def test_import_ignores_result_directory_until_atomic_payload_is_present(tmp_path) -> None:
    queue_root = tmp_path / "queue"
    incomplete = queue_root / "results" / "inbox" / "task-uploading"
    incomplete.mkdir(parents=True)

    summary = import_ai_research_results(
        queue_root,
        lake_root=tmp_path / "lake",
    )

    assert summary["examined"] == 0
    assert summary["rejected"] == 0
    assert incomplete.is_dir()


def test_import_validation_rejects_hallucinated_evidence_members() -> None:
    task = _task()
    with pytest.raises(ValueError, match="not present"):
        _validate_result_against_task(
            _result(task, _reference("reports/not-in-task.csv")),
            task,
        )


def test_blocked_preflight_cannot_enter_stage2() -> None:
    task = _task().model_copy(
        update={
            "preflight": TaskPreflight(
                status="BLOCK",
                checked_at=datetime(2026, 7, 14, tzinfo=UTC),
                missing_core_members=["provenance.json"],
                blockers=["missing_core_member:provenance.json"],
                truncated_document_count=0,
            )
        }
    )
    task = task.model_copy(update={"packet_sha256": compute_task_packet_sha256(task)})

    with pytest.raises(ValueError, match="blocked deterministic preflight"):
        _validate_result_against_task(_result(task, _reference()), task)


def test_nas_result_requires_and_publishes_materialized_effective_preflight(tmp_path) -> None:
    embedded = _task()
    provisional = embedded.model_copy(
        update={
            "source_pack_id": "expert-pack-test",
            "source_snapshot_id": "snapshot-test",
            "source_location": "nas_accepted",
            "sections": {},
            "preflight": None,
            "packet_sha256": "0" * 64,
        }
    )
    task = provisional.model_copy(
        update={"packet_sha256": compute_task_packet_sha256(provisional)}
    )
    result = _result(task, _reference()).model_copy(
        update={
            "effective_preflight": TaskPreflight(
                status="WARN",
                checked_at=datetime(2026, 7, 14, 1, tzinfo=UTC),
                available_sections=["factor_research"],
                truncated_document_count=0,
                warnings=["paper_runtime_stale"],
            ),
            "evidence_manifest": [
                EvidenceManifestEntry(
                    section="factor_research",
                    source_member="reports/factor_evidence.csv",
                    content_sha256="b" * 64,
                )
            ],
        }
    )

    _validate_result_against_task(result, task)
    _publish_result(result, task=task, lake_root=tmp_path)

    row = read_parquet_dataset(tmp_path / AI_RUN_DATASET).to_dicts()[0]
    assert row["preflight_status"] == "WARN"
    assert row["preflight_checked_at"] == datetime(2026, 7, 14, 1, tzinfo=UTC)
    assert row["preflight_blockers_json"] == "[]"
    assert row["preflight_warnings_json"] == '["paper_runtime_stale"]'


def test_nas_result_without_materialized_preflight_is_rejected() -> None:
    embedded = _task()
    provisional = embedded.model_copy(
        update={
            "source_pack_id": "expert-pack-test",
            "source_snapshot_id": "snapshot-test",
            "source_location": "nas_accepted",
            "sections": {},
            "preflight": None,
            "packet_sha256": "0" * 64,
        }
    )
    task = provisional.model_copy(
        update={"packet_sha256": compute_task_packet_sha256(provisional)}
    )

    with pytest.raises(ValueError, match="materialized effective preflight"):
        _validate_result_against_task(_result(task, _reference()), task)


def test_stage1_code_target_is_published_when_stage2_is_blocked(tmp_path) -> None:
    task = _task()
    reference = _reference()
    finding = ResearchFinding(
        finding_id="finding-data",
        category="data_quality",
        status="observed",
        severity="critical",
        summary="Evidence refresh is incomplete.",
        explanation="The source evidence is incomplete.",
        confidence=0.9,
        evidence_refs=[reference],
        recommended_action="Repair the evidence refresh.",
    )
    diagnosis = Stage1Diagnosis(
        task_id=task.task_id,
        system_state="BLOCKED_DATA_QUALITY",
        executive_summary="Stage 2 is blocked.",
        stage2_allowed=False,
        primary_bottlenecks=[finding],
        primary_bottleneck_id=finding.finding_id,
        code_review_targets=[
            CodeReviewTarget(
                target_id="review-refresh",
                repository="quant-lab",
                path_or_component="src/quant_lab/reports",
                reason="Find why evidence refresh is incomplete.",
                expected_evidence="A fresh complete evidence row.",
                priority="P0",
                source_finding_ids=[finding.finding_id],
            )
        ],
    )
    result = AIResearchResult(
        task_id=task.task_id,
        source_pack_sha256=task.source_pack_sha256,
        packet_sha256=task.packet_sha256,
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        worker_id="nas-1",
        started_at=datetime(2026, 7, 14, 1, tzinfo=UTC),
        completed_at=datetime(2026, 7, 14, 2, tzinfo=UTC),
        diagnosis=diagnosis,
        proposals=None,
    )

    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "task_id": "old-task",
                    "completed_at": datetime(2026, 7, 13, tzinfo=UTC),
                    "system_state": "REVIEW_REQUIRED",
                }
            ]
        ),
        tmp_path / AI_RUN_DATASET,
    )

    _publish_result(result, task=task, lake_root=tmp_path)

    rows = read_parquet_dataset(tmp_path / AI_CODE_REVIEW_DATASET).to_dicts()
    assert rows[0]["target_id"] == "review-refresh"
    assert rows[0]["origin_stage"] == "STAGE1_DIAGNOSTIC"
    run_rows = read_parquet_dataset(tmp_path / AI_RUN_DATASET)
    assert run_rows.height == 2
    assert "root_cause_tree_json" in run_rows.columns
