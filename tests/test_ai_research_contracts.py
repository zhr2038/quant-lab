from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from quant_lab.ai_research.contracts import (
    PROHIBITED_ACTIONS,
    AIResearchResult,
    AIResearchTask,
    EvidenceReference,
    KeyValue,
    LegacyStage2ProposalSet,
    ResearchFinding,
    ResearchHypothesisDraft,
    RootCauseNode,
    Stage1Diagnosis,
    Stage2ProposalSet,
    strict_output_schema,
)


def _evidence() -> EvidenceReference:
    return EvidenceReference(
        section="factor_research",
        source_member="reports/factor_evidence.csv",
        row_keys=[KeyValue(key="factor_id", value="core.test")],
        fields=["rank_ic_mean", "long_short_mean_bps"],
        claim="The cited row contains the measurements used by the finding.",
    )


def test_observed_finding_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        ResearchFinding(
            finding_id="missing-evidence",
            category="factor_research",
            status="observed",
            severity="warning",
            summary="Observed claim without a citation.",
            explanation="This must be rejected.",
            confidence=0.8,
            evidence_refs=[],
            recommended_action="Do not accept it.",
        )


def test_stage2_gate_requires_ready_state_and_route() -> None:
    with pytest.raises(ValidationError):
        Stage1Diagnosis(
            task_id="task-1",
            system_state="REVIEW_REQUIRED",
            executive_summary="Not ready.",
            stage2_allowed=True,
            route_sections=["factor_research"],
        )


def test_hypothesis_draft_is_bounded_and_never_auto_executes() -> None:
    proposal = ResearchHypothesisDraft(
        hypothesis_id="hypothesis-1",
        title="Lower-cost momentum persistence",
        hypothesis_family="behavioral_underreaction",
        research_question="Does underreaction persist after beta and liquidity controls?",
        economic_return_payer="Late information adopters may pay earlier informed participants.",
        persistence_mechanism="Information may diffuse slowly across the observed universe.",
        beta_exclusion_design="Residualize returns against timestamp-aligned BTC beta.",
        liquidity_exclusion_design="Match observations by spread and depth quantiles.",
        symbol_fixed_effect_exclusion_design="Estimate within-symbol effects with fixed effects.",
        required_datasets=["factor_value", "market_bar"],
        required_fields=["available_time", "forward_return_bps", "spread_bps"],
        data_availability_status="AVAILABLE_VERIFIED",
        data_availability_notes="The cited report exposes the required fields.",
        expected_horizon_bars=[4, 8],
        falsification_conditions=["The residual after-cost effect is non-positive."],
        stopping_conditions=["Stop after the independent holdout fails twice."],
        known_overlap_risks=["Existing liquidity-adjusted momentum family."],
        max_variants=2,
        evidence_refs=[_evidence()],
        research_thread_id="thread-hypothesis-1",
        source_finding_ids=["finding-1"],
    )
    assert proposal.research_only is True
    assert proposal.live_order_effect == "none_read_only_research"
    assert proposal.proposal_state == "AI_RESEARCH_DRAFT"
    assert proposal.automatic_execution is False
    assert proposal.automatic_promotion is False

    with pytest.raises(ValidationError):
        ResearchHypothesisDraft.model_validate(
            {**proposal.model_dump(), "max_variants": 4}
        )

    with pytest.raises(ValidationError):
        Stage2ProposalSet(
            task_id="task-too-many-hypotheses",
            executive_summary="The contract must stay bounded.",
            research_hypothesis_drafts=[proposal] * 4,
        )


def test_result_cannot_contain_proposals_when_stage2_is_blocked() -> None:
    diagnosis = Stage1Diagnosis(
        task_id="task-2",
        system_state="BLOCKED_DATA_QUALITY",
        executive_summary="Freshness is insufficient.",
        stage2_allowed=False,
        route_sections=[],
        primary_bottlenecks=[
            ResearchFinding(
                finding_id="stale-data",
                category="data_quality",
                status="observed",
                severity="critical",
                summary="The source is stale.",
                explanation="The manifest marks the relevant dataset stale.",
                confidence=0.99,
                evidence_refs=[_evidence()],
                recommended_action="Refresh evidence before proposing changes.",
            )
        ],
    )
    proposals = Stage2ProposalSet(
        task_id="task-2",
        executive_summary="No proposal should be accepted.",
    )
    with pytest.raises(ValidationError):
        AIResearchResult(
            task_id="task-2",
            source_pack_sha256="a" * 64,
            packet_sha256="b" * 64,
            model="gpt-5.6-sol",
            reasoning_effort="xhigh",
            worker_id="nas-1",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            diagnosis=diagnosis,
            proposals=proposals,
        )


def test_result_contract_still_reads_legacy_stage2_history() -> None:
    diagnosis = Stage1Diagnosis(
        task_id="task-legacy-result",
        system_state="READY_FOR_PROPOSALS",
        executive_summary="Historical result.",
        stage2_allowed=True,
        route_sections=["factor_research"],
    )
    result = AIResearchResult(
        task_id="task-legacy-result",
        source_pack_sha256="a" * 64,
        packet_sha256="b" * 64,
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        worker_id="nas-legacy",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        diagnosis=diagnosis,
        proposals=LegacyStage2ProposalSet(
            task_id="task-legacy-result",
            executive_summary="Historical proposal payload.",
        ),
        prompt_version="quant_lab.ai_research.prompt.v3",
    )

    assert isinstance(result.proposals, LegacyStage2ProposalSet)


def test_task_contract_still_reads_v4_hypothesis_research_history() -> None:
    task = AIResearchTask(
        prompt_version="quant_lab.ai_research.prompt.v4",
        task_id="task-v4-history",
        created_at=datetime.now(UTC),
        source_pack_name="expert.zip",
        source_pack_sha256="a" * 64,
        packet_sha256="b" * 64,
        sections={"factor_research": []},
        allowed_hypothesis_families=["behavioral_underreaction"],
    )

    assert task.prompt_version == "quant_lab.ai_research.prompt.v4"


def test_strict_schema_rejects_unknown_keys_and_requires_all_properties() -> None:
    schema = strict_output_schema(Stage1Diagnosis)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    finding_schema = schema["$defs"]["ResearchFinding"]
    assert finding_schema["additionalProperties"] is False
    assert set(finding_schema["required"]) == set(finding_schema["properties"])


def test_stage_outputs_cannot_remove_prohibited_actions() -> None:
    with pytest.raises(ValidationError, match="prohibited_actions"):
        Stage1Diagnosis(
            task_id="task-safety",
            system_state="REVIEW_REQUIRED",
            executive_summary="Review only.",
            stage2_allowed=False,
            prohibited_actions=list(PROHIBITED_ACTIONS[:-1]),
        )

    with pytest.raises(ValidationError, match="prohibited_actions"):
        Stage2ProposalSet(
            task_id="task-safety",
            executive_summary="Draft only.",
            prohibited_actions=["live_order"],
        )


def test_result_requires_ordered_utc_timestamps() -> None:
    diagnosis = Stage1Diagnosis(
        task_id="task-time",
        system_state="REVIEW_REQUIRED",
        executive_summary="Review only.",
        stage2_allowed=False,
    )
    with pytest.raises(ValidationError, match="completed_at"):
        AIResearchResult(
            task_id="task-time",
            source_pack_sha256="a" * 64,
            packet_sha256="b" * 64,
            model="gpt-5.6-sol",
            reasoning_effort="xhigh",
            worker_id="nas-1",
            started_at=datetime(2026, 7, 14, 2, tzinfo=UTC),
            completed_at=datetime(2026, 7, 14, 1, tzinfo=UTC),
            diagnosis=diagnosis,
            proposals=None,
        )


def test_stage1_root_cause_must_reference_current_finding() -> None:
    with pytest.raises(ValidationError, match="unknown findings"):
        Stage1Diagnosis(
            task_id="task-root-cause",
            system_state="REVIEW_REQUIRED",
            executive_summary="Review only.",
            stage2_allowed=False,
            root_cause_tree=[
                RootCauseNode(
                    node_id="root-1",
                    label="Unlinked root cause.",
                    causal_role="primary",
                    source_finding_ids=["missing-finding"],
                    evidence_refs=[_evidence()],
                )
            ],
        )
