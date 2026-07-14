from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AI_TASK_SCHEMA_VERSION = "quant_lab.ai_research_task.v1"
AI_STAGE1_SCHEMA_VERSION = "quant_lab.ai_research_diagnosis.v1"
AI_STAGE2_SCHEMA_VERSION = "quant_lab.ai_research_proposals.v1"
AI_RESULT_SCHEMA_VERSION = "quant_lab.ai_research_result.v1"
AI_PROMPT_VERSION = "quant_lab.ai_research.prompt.v1"

LIVE_ORDER_EFFECT = "none_read_only_research"
PROHIBITED_ACTIONS = (
    "live_order",
    "cancel_order",
    "modify_position",
    "modify_exchange_state",
    "modify_v5_live_config",
    "modify_risk_permission",
    "automatic_strategy_promotion",
)

FactorTemplate = Literal[
    "feature",
    "neg_feature",
    "product",
    "difference",
    "safe_divide",
    "vol_adjusted",
    "range_vol_ratio",
    "range_location",
    "liquidity_adjusted",
]

ResearchCategory = Literal[
    "data_quality",
    "factor_research",
    "cost_model",
    "entry_false_block",
    "exit_quality",
    "paper_lifecycle",
    "strategy_evidence",
    "operations",
    "unknown",
]

FindingStatus = Literal["observed", "hypothesis", "insufficient_evidence"]
FindingSeverity = Literal["info", "warning", "critical"]
ResearchMode = Literal["backtest", "shadow", "paper"]
PaperDirection = Literal["long", "short"]
RuleMatch = Literal["all", "any"]
RuleOperator = Literal[
    "gt",
    "gte",
    "lt",
    "lte",
    "crosses_above",
    "crosses_below",
    "consecutive",
    "rank_gte",
    "rank_lte",
    "quantile_gte",
    "quantile_lte",
    "momentum_gt",
    "momentum_lt",
    "return_gt",
    "return_lt",
    "volatility_lt",
    "volatility_gt",
    "volume_zscore_gt",
    "volume_zscore_lt",
    "regime_in",
    "take_profit",
    "stop_loss",
    "trailing_exit",
    "max_holding_bars",
    "signal_invalid",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class KeyValue(StrictModel):
    key: str = Field(min_length=1, max_length=160)
    value: str = Field(min_length=1, max_length=1000)


class EvidenceReference(StrictModel):
    section: str = Field(min_length=1, max_length=120)
    source_member: str = Field(min_length=1, max_length=500)
    row_keys: list[KeyValue] = Field(default_factory=list, max_length=12)
    fields: list[str] = Field(default_factory=list, max_length=32)
    claim: str = Field(min_length=1, max_length=2000)


class ResearchFinding(StrictModel):
    finding_id: str = Field(min_length=1, max_length=120)
    category: ResearchCategory
    status: FindingStatus
    severity: FindingSeverity
    summary: str = Field(min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=3000)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[EvidenceReference] = Field(default_factory=list, max_length=12)
    recommended_action: str = Field(min_length=1, max_length=1500)

    @model_validator(mode="after")
    def observed_requires_evidence(self) -> ResearchFinding:
        if self.status == "observed" and not self.evidence_refs:
            raise ValueError("observed findings require at least one evidence reference")
        return self


class Stage1Diagnosis(StrictModel):
    schema_version: Literal[AI_STAGE1_SCHEMA_VERSION] = AI_STAGE1_SCHEMA_VERSION
    task_id: str = Field(min_length=1, max_length=160)
    system_state: Literal[
        "READY_FOR_PROPOSALS",
        "BLOCKED_DATA_QUALITY",
        "BLOCKED_INSUFFICIENT_EVIDENCE",
        "REVIEW_REQUIRED",
    ]
    executive_summary: str = Field(min_length=1, max_length=3000)
    stage2_allowed: bool
    route_sections: list[str] = Field(default_factory=list, max_length=12)
    primary_bottlenecks: list[ResearchFinding] = Field(default_factory=list, max_length=12)
    contradictions: list[ResearchFinding] = Field(default_factory=list, max_length=12)
    missing_evidence: list[ResearchFinding] = Field(default_factory=list, max_length=12)
    data_quality_warnings: list[str] = Field(default_factory=list, max_length=32)
    prohibited_actions: list[str] = Field(default_factory=lambda: list(PROHIBITED_ACTIONS))

    @model_validator(mode="after")
    def enforce_stage2_gate(self) -> Stage1Diagnosis:
        if self.stage2_allowed and not self.route_sections:
            raise ValueError("stage2_allowed requires at least one routed evidence section")
        if self.stage2_allowed and self.system_state != "READY_FOR_PROPOSALS":
            raise ValueError("stage2_allowed requires system_state=READY_FOR_PROPOSALS")
        _require_prohibited_actions(self.prohibited_actions)
        return self


class FactorParameter(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=500)


class FactorProposal(StrictModel):
    proposal_id: str = Field(min_length=1, max_length=160)
    factor_name: str = Field(min_length=1, max_length=160)
    factor_family: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    template: FactorTemplate
    input_features: list[str] = Field(min_length=1, max_length=8)
    parameters: list[FactorParameter] = Field(default_factory=list, max_length=16)
    direction: Literal[-1, 1]
    lookback_bars: int = Field(ge=1, le=10_000)
    availability_lag_bars: int = Field(ge=1, le=100)
    expected_horizon_bars: list[int] = Field(min_length=1, max_length=8)
    hypothesis: str = Field(min_length=1, max_length=2000)
    economic_rationale: str = Field(min_length=1, max_length=2000)
    falsification_conditions: list[str] = Field(min_length=1, max_length=12)
    evidence_refs: list[EvidenceReference] = Field(min_length=1, max_length=12)
    known_overlap_risk: str = Field(min_length=1, max_length=1000)
    research_only: Literal[True] = True
    live_order_effect: Literal[LIVE_ORDER_EFFECT] = LIVE_ORDER_EFFECT

    @field_validator("expected_horizon_bars")
    @classmethod
    def horizons_are_positive(cls, value: list[int]) -> list[int]:
        cleaned = sorted(set(int(item) for item in value if int(item) > 0))
        if not cleaned:
            raise ValueError("expected_horizon_bars must contain positive values")
        return cleaned


class RuleClauseDraft(StrictModel):
    operator: RuleOperator
    field: str | None
    reference_field: str | None
    value: str | None
    values: list[str] = Field(default_factory=list, max_length=32)
    window: int | None = Field(default=None, ge=1, le=512)
    periods: int | None = Field(default=None, ge=1, le=512)
    rationale: str = Field(min_length=1, max_length=1000)


class PaperStrategyDraft(StrictModel):
    draft_id: str = Field(min_length=1, max_length=160)
    strategy_family: str = Field(min_length=1, max_length=120)
    symbols: list[str] = Field(min_length=1, max_length=16)
    timeframe: str = Field(min_length=1, max_length=24)
    direction: PaperDirection
    entry_match: RuleMatch
    entry_clauses: list[RuleClauseDraft] = Field(min_length=1, max_length=16)
    exit_match: RuleMatch
    exit_clauses: list[RuleClauseDraft] = Field(min_length=1, max_length=16)
    max_holding_bars: int = Field(ge=1, le=10_000)
    min_holding_bars: int = Field(ge=0, le=10_000)
    cooldown_bars: int = Field(ge=0, le=10_000)
    required_market_fields: list[str] = Field(default_factory=list, max_length=64)
    hypothesis: str = Field(min_length=1, max_length=2000)
    falsification_conditions: list[str] = Field(min_length=1, max_length=12)
    evidence_refs: list[EvidenceReference] = Field(min_length=1, max_length=12)
    mode: Literal["shadow", "paper"]
    research_only: Literal[True] = True
    live_order_effect: Literal[LIVE_ORDER_EFFECT] = LIVE_ORDER_EFFECT

    @model_validator(mode="after")
    def holding_period_is_valid(self) -> PaperStrategyDraft:
        if self.min_holding_bars > self.max_holding_bars:
            raise ValueError("min_holding_bars cannot exceed max_holding_bars")
        return self


class ExperimentProposal(StrictModel):
    proposal_id: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=1000)
    hypothesis: str = Field(min_length=1, max_length=2000)
    control: str = Field(min_length=1, max_length=1500)
    treatment: str = Field(min_length=1, max_length=1500)
    required_datasets: list[str] = Field(min_length=1, max_length=32)
    success_metrics: list[str] = Field(min_length=1, max_length=16)
    minimum_complete_samples: int = Field(ge=10, le=1_000_000)
    mode: ResearchMode
    risks: list[str] = Field(default_factory=list, max_length=12)
    evidence_refs: list[EvidenceReference] = Field(min_length=1, max_length=12)
    research_only: Literal[True] = True
    live_order_effect: Literal[LIVE_ORDER_EFFECT] = LIVE_ORDER_EFFECT


class CodeReviewTarget(StrictModel):
    repository: Literal["quant-lab", "V5-prod"]
    path_or_component: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=1500)
    expected_evidence: str = Field(min_length=1, max_length=1000)
    priority: Literal["P0", "P1", "P2", "P3"]


class Stage2ProposalSet(StrictModel):
    schema_version: Literal[AI_STAGE2_SCHEMA_VERSION] = AI_STAGE2_SCHEMA_VERSION
    task_id: str = Field(min_length=1, max_length=160)
    executive_summary: str = Field(min_length=1, max_length=3000)
    factor_proposals: list[FactorProposal] = Field(default_factory=list, max_length=8)
    paper_strategy_drafts: list[PaperStrategyDraft] = Field(default_factory=list, max_length=6)
    experiment_proposals: list[ExperimentProposal] = Field(default_factory=list, max_length=8)
    code_review_targets: list[CodeReviewTarget] = Field(default_factory=list, max_length=12)
    no_action_reasons: list[str] = Field(default_factory=list, max_length=24)
    prohibited_actions: list[str] = Field(default_factory=lambda: list(PROHIBITED_ACTIONS))

    @model_validator(mode="after")
    def preserve_safety_boundary(self) -> Stage2ProposalSet:
        _require_prohibited_actions(self.prohibited_actions)
        return self


class EvidenceDocument(StrictModel):
    source_member: str = Field(min_length=1, max_length=500)
    source_format: Literal["json", "csv", "markdown", "text"]
    content_sha256: str = Field(min_length=64, max_length=64)
    source_size_bytes: int = Field(ge=1)
    truncated: bool = False
    content: Any


class AIResearchTask(StrictModel):
    schema_version: Literal[AI_TASK_SCHEMA_VERSION] = AI_TASK_SCHEMA_VERSION
    prompt_version: Literal[AI_PROMPT_VERSION] = AI_PROMPT_VERSION
    task_id: str = Field(min_length=1, max_length=160)
    created_at: datetime
    source_pack_name: str = Field(min_length=1, max_length=500)
    source_pack_sha256: str = Field(min_length=64, max_length=64)
    packet_sha256: str = Field(min_length=64, max_length=64)
    sections: dict[str, list[EvidenceDocument]]
    allowed_factor_templates: list[FactorTemplate]
    prohibited_actions: list[str] = Field(default_factory=lambda: list(PROHIBITED_ACTIONS))
    warnings: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_task_boundary(self) -> AIResearchTask:
        _require_utc(self.created_at, field_name="created_at")
        _require_prohibited_actions(self.prohibited_actions)
        if not self.sections:
            raise ValueError("AI research task requires at least one evidence section")
        if not self.allowed_factor_templates:
            raise ValueError("AI research task requires at least one factor template")
        return self


class AIResearchResult(StrictModel):
    schema_version: Literal[AI_RESULT_SCHEMA_VERSION] = AI_RESULT_SCHEMA_VERSION
    task_id: str = Field(min_length=1, max_length=160)
    source_pack_sha256: str = Field(min_length=64, max_length=64)
    packet_sha256: str = Field(min_length=64, max_length=64)
    model: str = Field(min_length=1, max_length=160)
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"]
    worker_id: str = Field(min_length=1, max_length=160)
    started_at: datetime
    completed_at: datetime
    diagnosis: Stage1Diagnosis
    proposals: Stage2ProposalSet | None
    stage1_response_id: str | None = Field(default=None, max_length=300)
    stage2_response_id: str | None = Field(default=None, max_length=300)
    usage_json: str = "{}"
    warnings: list[str] = Field(default_factory=list, max_length=128)
    diagnostic_only: Literal[True] = True
    live_order_effect: Literal[LIVE_ORDER_EFFECT] = LIVE_ORDER_EFFECT

    @model_validator(mode="after")
    def proposal_gate_matches_diagnosis(self) -> AIResearchResult:
        if self.diagnosis.stage2_allowed and self.proposals is None:
            raise ValueError("stage2_allowed result requires proposals")
        if not self.diagnosis.stage2_allowed and self.proposals is not None:
            raise ValueError("blocked diagnosis must not contain proposals")
        if self.diagnosis.task_id != self.task_id:
            raise ValueError("diagnosis task_id mismatch")
        if self.proposals is not None and self.proposals.task_id != self.task_id:
            raise ValueError("proposal task_id mismatch")
        _require_utc(self.started_at, field_name="started_at")
        _require_utc(self.completed_at, field_name="completed_at")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def task_packet_payload(task: AIResearchTask | dict[str, Any]) -> dict[str, Any]:
    if isinstance(task, AIResearchTask):
        payload = task.model_dump(mode="json", exclude={"packet_sha256"})
    else:
        payload = dict(task)
        payload.pop("packet_sha256", None)
    return payload


def compute_task_packet_sha256(task: AIResearchTask | dict[str, Any]) -> str:
    return sha256_json(task_packet_payload(task))


def strict_output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a Responses API compatible strict JSON schema.

    Structured Outputs requires every object to reject unknown keys and works best
    when all properties are listed in ``required``. Nullable fields remain nullable
    through their Pydantic-generated ``anyOf`` branches.
    """

    schema = model.model_json_schema()

    def patch(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["additionalProperties"] = False
                    node["required"] = list(properties)
            for value in node.values():
                patch(value)
        elif isinstance(node, list):
            for value in node:
                patch(value)

    patch(schema)
    return schema


def _require_prohibited_actions(value: list[str]) -> None:
    if len(value) != len(PROHIBITED_ACTIONS) or set(value) != set(PROHIBITED_ACTIONS):
        raise ValueError("prohibited_actions must preserve the complete safety boundary")


def _require_utc(value: datetime, *, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} must use UTC")
