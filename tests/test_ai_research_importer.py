from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_lab.ai_research.contracts import (
    AIResearchResult,
    AIResearchTask,
    EvidenceDocument,
    EvidenceReference,
    FactorProposal,
    ResearchFinding,
    Stage1Diagnosis,
    Stage2ProposalSet,
    compute_task_packet_sha256,
)
from quant_lab.ai_research.importer import _validate_result_against_task


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


def test_import_validation_rejects_hallucinated_evidence_members() -> None:
    task = _task()
    with pytest.raises(ValueError, match="not present"):
        _validate_result_against_task(
            _result(task, _reference("reports/not-in-task.csv")),
            task,
        )
