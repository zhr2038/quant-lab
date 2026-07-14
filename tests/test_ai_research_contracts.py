from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from quant_lab.ai_research.contracts import (
    PROHIBITED_ACTIONS,
    AIResearchResult,
    EvidenceReference,
    FactorProposal,
    KeyValue,
    ResearchFinding,
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


def test_factor_proposal_is_research_only_and_has_lag() -> None:
    proposal = FactorProposal(
        proposal_id="factor-1",
        factor_name="volume_momentum_ratio",
        factor_family="volume_price_confirm",
        description="A constrained test proposal.",
        template="safe_divide",
        input_features=["close_return_4", "spread_bps"],
        parameters=[],
        direction=1,
        lookback_bars=4,
        availability_lag_bars=1,
        expected_horizon_bars=[4, 8],
        hypothesis="The ratio may separate efficient momentum from noisy moves.",
        economic_rationale="Momentum with lower spread should be easier to monetize after cost.",
        falsification_conditions=["after-cost spread is non-positive"],
        evidence_refs=[_evidence()],
        known_overlap_risk="May overlap with existing liquidity-adjusted momentum.",
        research_thread_id="thread-factor-1",
        source_finding_ids=["finding-1"],
    )
    assert proposal.research_only is True
    assert proposal.live_order_effect == "none_read_only_research"

    with pytest.raises(ValidationError):
        FactorProposal.model_validate(
            {**proposal.model_dump(), "availability_lag_bars": 0}
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
