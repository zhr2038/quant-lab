"""New research semantics without changing the bytes of archived v1 contracts."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from quant_lab.contracts.models import require_utc
from quant_lab.decision.contracts import Advice, AnalysisResult, ForwardSummary
from quant_lab.decision.contracts_base import Contract
from quant_lab.decision.current_cost_contracts import CurrentCostObservation

EXPERIMENT_VERSION = "trend-reference-1h-v2"
STRATEGY_VERSION = "context-trend-24h-v1"


class ResearchEligibility(Contract):
    research_evaluable: bool
    cost_calibrated: bool
    live_execution_eligible: Literal[False] = False
    scope: Literal["independent_paper_comparison_only"] = "independent_paper_comparison_only"


class AdviceV2(Advice):
    experiment_version: str = Field(default=EXPERIMENT_VERSION, min_length=1, max_length=128)
    strategy_version: str = Field(default=STRATEGY_VERSION, min_length=1, max_length=128)
    eligibility: ResearchEligibility

    @model_validator(mode="after")
    def eligibility_has_evidence(self):
        calibrated = (
            not isinstance(self.cost, CurrentCostObservation)
            and self.cost.quality == "calibrated"
            and self.cost.actual_sample_count > 0
        )
        if self.eligibility.cost_calibrated != calibrated:
            raise ValueError("cost calibration flag must reflect original cost evidence")
        if self.eligibility.research_evaluable:
            if self.action == "NO_VIEW" or self.cost.roundtrip_bps is None:
                raise ValueError("incomplete advice cannot be research evaluable")
            if isinstance(self.cost, CurrentCostObservation):
                if (
                    self.cost.current.status != "ESTIMATED"
                    or self.cost.current.valid_until <= self.generated_at
                ):
                    raise ValueError("research estimate must be currently valid")
            elif not self.cost.trusted_for_paper:
                raise ValueError("legacy cost lacks research evidence")
        if self.action == "REVIEW_ENTRY" and not self.eligibility.research_evaluable:
            raise ValueError("research entry requires explicit eligibility")
        return self


class ScopedForwardSummary(ForwardSummary):
    experiment: str = Field(min_length=1, max_length=128)
    strategy_version: str = Field(min_length=1, max_length=128)
    cost_versions: list[str] = Field(min_length=1, max_length=4)
    published_from: datetime
    published_until: datetime
    non_overlapping_opportunities: int = Field(ge=0)
    overlapping_price_observations: int = Field(ge=0)
    actual_trades: None = None

    @field_validator("published_from", "published_until")
    @classmethod
    def utc_scope(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def valid_scope(self):
        if self.published_from > self.published_until:
            raise ValueError("invalid forward summary time interval")
        if self.non_overlapping_opportunities > self.registered_opportunities:
            raise ValueError("independent opportunity count exceeds registered count")
        if (
            self.overlapping_price_observations
            != self.registered_horizon_observations - self.non_overlapping_opportunities
        ):
            raise ValueError("overlapping label counts do not reconcile")
        return self


class AnalysisResultV2(AnalysisResult):
    schema_version: Literal["qlab.decision.result.v2"] = "qlab.decision.result.v2"
    experiment_version: str = Field(default=EXPERIMENT_VERSION, min_length=1, max_length=128)
    advice: list[AdviceV2] = Field(min_length=8, max_length=8)
    forward: ScopedForwardSummary

    @model_validator(mode="after")
    def bound_experiment(self):
        if any(
            a.experiment_version != self.experiment_version
            or a.strategy_version != self.forward.strategy_version
            for a in self.advice
        ):
            raise ValueError("mixed experiment or strategy versions")
        if self.forward.experiment != self.experiment_version:
            raise ValueError("forward experiment mismatch")
        return self


def parse_result(value: dict) -> AnalysisResult | AnalysisResultV2:
    schema = value.get("schema_version")
    if schema == "qlab.decision.result.v2":
        return AnalysisResultV2.model_validate(value)
    if schema == "qlab.decision.result.v1":
        return AnalysisResult.model_validate(value)
    raise ValueError("unsupported result schema")


def parse_advice(value: dict) -> Advice | AdviceV2:
    if "eligibility" in value:
        return AdviceV2.model_validate(value)
    return Advice.model_validate(value)
