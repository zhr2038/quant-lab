from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ResearchFamily(StrEnum):
    DEFENSIVE_QUALITY = "DEFENSIVE_QUALITY"
    ABSOLUTE_TIMING_BREADTH = "ABSOLUTE_TIMING_BREADTH"
    DERIVATIVES_CROWDING = "DERIVATIVES_CROWDING"
    LIQUIDITY_MICROSTRUCTURE = "LIQUIDITY_MICROSTRUCTURE"


class HypothesisStatus(StrEnum):
    DRAFT = "DRAFT"
    DATA_BLOCKED = "DATA_BLOCKED"
    APPROVED_FOR_RESEARCH = "APPROVED_FOR_RESEARCH"
    RUNNING = "RUNNING"
    SIGNAL_VALID = "SIGNAL_VALID"
    PORTFOLIO_FAIL = "PORTFOLIO_FAIL"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class DiscoverySource(StrEnum):
    HUMAN = "HUMAN"
    AI_DRAFT = "AI_DRAFT"
    AUDIT_IMPORT = "AUDIT_IMPORT"


class StrategyFit(StrEnum):
    CROSS_SECTIONAL_RANK = "cross_sectional_rank"
    ABSOLUTE_TIMING = "absolute_timing"
    LONG_ONLY_SPOT = "long_only_spot"
    MARKET_NEUTRAL_RESEARCH_ONLY = "market_neutral_research_only"


class FactorResearchDecision(StrEnum):
    DATA_BLOCKED = "DATA_BLOCKED"
    REJECTED_DATA_QUALITY = "REJECTED_DATA_QUALITY"
    REJECTED_LEAKAGE = "REJECTED_LEAKAGE"
    REJECTED_DUPLICATE = "REJECTED_DUPLICATE"
    REJECTED_NO_SIGNAL = "REJECTED_NO_SIGNAL"
    REJECTED_MULTIPLE_TESTING = "REJECTED_MULTIPLE_TESTING"
    REJECTED_OVERFIT = "REJECTED_OVERFIT"
    SIGNAL_CANDIDATE = "SIGNAL_CANDIDATE"
    SIGNAL_VALID = "SIGNAL_VALID"
    PORTFOLIO_FAIL = "PORTFOLIO_FAIL"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    FORWARD_RUNNING = "FORWARD_RUNNING"
    PAPER_REVIEW_READY = "PAPER_REVIEW_READY"
    RETIRED = "RETIRED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INCONCLUSIVE_OVERFIT_DIAGNOSTICS = "INCONCLUSIVE_OVERFIT_DIAGNOSTICS"


class TrialKind(StrEnum):
    EXPLORATORY = "EXPLORATORY"
    CONFIRMATORY = "CONFIRMATORY"


class TrialStatus(StrEnum):
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    INVALIDATED = "INVALIDATED"


class DataRequirement(StrictModel):
    dataset_name: str = Field(min_length=1, max_length=180)
    required_columns: tuple[str, ...] = Field(min_length=1, max_length=64)
    min_history_days: int = Field(ge=1, le=1095)
    real_source_required: bool = True
    proxy_allowed: Literal[False] = False

    @field_validator("required_columns")
    @classmethod
    def columns_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not cleaned:
            raise ValueError("required_columns must not be empty")
        return cleaned


class RecipeParameter(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=500)


class FeatureRecipe(StrictModel):
    recipe_id: str = Field(min_length=1, max_length=180)
    operation: Literal[
        "rolling_volatility",
        "rolling_downside_volatility",
        "volatility_change",
        "beta_stability",
        "liquidity_stability",
        "crash_sensitivity",
        "market_breadth",
        "market_dispersion",
        "funding_crowding",
        "microstructure_liquidity",
    ]
    source_fields: tuple[str, ...] = Field(min_length=1, max_length=16)
    parameters: tuple[RecipeParameter, ...] = Field(default=(), max_length=16)


class UniverseDefinition(StrictModel):
    universe_id: str = Field(min_length=1, max_length=180)
    symbols: tuple[str, ...] = Field(default=(), max_length=200)
    dynamic_source_dataset: str | None = Field(default=None, max_length=180)
    inclusion_rule: str = Field(min_length=1, max_length=1000)
    exclusion_rule: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def has_source(self) -> UniverseDefinition:
        if not self.symbols and not self.dynamic_source_dataset:
            raise ValueError("universe requires symbols or dynamic_source_dataset")
        return self


class NeutralizationPlan(StrictModel):
    neutralization_id: str = Field(min_length=1, max_length=180)
    controls: tuple[
        Literal[
            "symbol_fixed_effect",
            "market_beta",
            "liquidity",
            "long_run_volatility",
            "momentum",
            "regime",
        ],
        ...,
    ] = Field(default=(), max_length=8)


class BenchmarkPlan(StrictModel):
    benchmark_id: str = Field(min_length=1, max_length=180)
    benchmarks: tuple[Literal["BTC", "DYNAMIC_UNIVERSE_EQUAL_WEIGHT"], ...] = Field(
        min_length=1, max_length=2
    )


class CostModelPlan(StrictModel):
    cost_model_id: str = Field(min_length=1, max_length=180)
    research_quantile: Literal["p50", "p75", "p90"] = "p75"
    require_point_in_time: bool = True
    proxy_can_support_deployment: Literal[False] = False


class ResearchHypothesis(StrictModel):
    hypothesis_id: str = Field(min_length=1, max_length=180)
    hypothesis_version: int = Field(ge=1)
    research_thread_id: str = Field(min_length=1, max_length=180)
    title: str = Field(min_length=1, max_length=500)
    factor_family: ResearchFamily
    economic_mechanism: str = Field(min_length=1, max_length=3000)
    who_pays_the_edge: str = Field(min_length=1, max_length=2000)
    why_edge_may_persist: str = Field(min_length=1, max_length=2000)
    why_edge_may_decay: str = Field(min_length=1, max_length=2000)
    strategy_fit: tuple[StrategyFit, ...] = Field(min_length=1, max_length=4)
    data_requirements: tuple[DataRequirement, ...] = Field(min_length=1, max_length=12)
    available_data_confirmed: bool
    blocked_missing_data: tuple[str, ...] = Field(default=(), max_length=24)
    feature_recipes: tuple[FeatureRecipe, ...] = Field(min_length=1, max_length=3)
    expected_direction: Literal[-1, 1]
    expected_horizons: tuple[int, ...] = Field(min_length=1, max_length=3)
    allowed_variants: tuple[str, ...] = Field(min_length=1, max_length=3)
    variant_budget: int = Field(ge=1, le=3)
    universe_definition: UniverseDefinition
    neutralization_plan: NeutralizationPlan
    benchmark_plan: BenchmarkPlan
    cost_model: CostModelPlan
    falsification_conditions: tuple[str, ...] = Field(min_length=1, max_length=16)
    stopping_conditions: tuple[str, ...] = Field(min_length=1, max_length=16)
    success_conditions: tuple[str, ...] = Field(min_length=1, max_length=16)
    known_overlap: tuple[str, ...] = Field(default=(), max_length=24)
    related_hypotheses: tuple[str, ...] = Field(default=(), max_length=24)
    related_existing_factors: tuple[str, ...] = Field(default=(), max_length=32)
    discovery_source: DiscoverySource
    status: HypothesisStatus
    approved_by: str | None = Field(default=None, max_length=180)
    approved_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    research_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0

    @field_validator("expected_horizons")
    @classmethod
    def horizons_are_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        cleaned = tuple(dict.fromkeys(int(item) for item in value if int(item) > 0))
        if not cleaned or len(cleaned) > 3:
            raise ValueError("expected_horizons must contain one to three positive values")
        return cleaned

    @model_validator(mode="after")
    def validate_hypothesis(self) -> ResearchHypothesis:
        _require_identifier(self.hypothesis_id, "hypothesis_id")
        _require_identifier(self.research_thread_id, "research_thread_id")
        _require_utc(self.created_at, "created_at")
        _require_utc(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if len(self.allowed_variants) > self.variant_budget:
            raise ValueError("allowed_variants exceeds variant_budget")
        if len(self.feature_recipes) > self.variant_budget:
            raise ValueError("feature_recipes exceeds variant_budget")
        active = self.status in {
            HypothesisStatus.APPROVED_FOR_RESEARCH,
            HypothesisStatus.RUNNING,
            HypothesisStatus.SIGNAL_VALID,
            HypothesisStatus.PORTFOLIO_FAIL,
            HypothesisStatus.PAPER_CANDIDATE,
        }
        if active:
            if not self.approved_by or self.approved_at is None:
                raise ValueError("active hypothesis requires approved_by and approved_at")
            _require_utc(self.approved_at, "approved_at")
            if self.discovery_source == DiscoverySource.AI_DRAFT:
                raise ValueError("AI_DRAFT cannot become executable without human re-registration")
        if self.status == HypothesisStatus.DATA_BLOCKED:
            if self.available_data_confirmed or not self.blocked_missing_data:
                raise ValueError("DATA_BLOCKED requires explicit missing data")
        elif not self.available_data_confirmed and active:
            raise ValueError("active hypothesis requires confirmed data")
        return self

    @property
    def definition_digest(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"status", "approved_by", "approved_at", "created_at", "updated_at"},
        )
        return _sha256_json(payload)


class ResearchTrial(StrictModel):
    trial_id: str = Field(min_length=1, max_length=180)
    hypothesis_id: str = Field(min_length=1, max_length=180)
    hypothesis_version: int = Field(ge=1)
    test_family_id: str = Field(min_length=1, max_length=180)
    factor_formula_hash: str = Field(min_length=64, max_length=64)
    feature_recipe_hash: str = Field(min_length=64, max_length=64)
    direction: Literal[-1, 1]
    lookback: int = Field(ge=1, le=1095 * 24)
    horizon: int = Field(ge=1, le=1095 * 24)
    universe_id: str = Field(min_length=1, max_length=180)
    neutralization_id: str = Field(min_length=1, max_length=180)
    cost_model_id: str = Field(min_length=1, max_length=180)
    portfolio_rule_id: str = Field(min_length=1, max_length=180)
    split_definition: str = Field(min_length=1, max_length=500)
    blind_period_id: str = Field(min_length=1, max_length=180)
    random_seed: int = Field(ge=0, le=2**31 - 1)
    code_commit: str = Field(min_length=40, max_length=40)
    data_snapshot_id: str = Field(min_length=1, max_length=180)
    nas_task_id: str = Field(min_length=1, max_length=180)
    trial_kind: TrialKind
    parameter_locked_at: datetime
    blind_opened_at: datetime | None = None
    blind_invalidated: bool = False
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: TrialStatus
    decision: FactorResearchDecision
    failure_reason: str | None = Field(default=None, max_length=2000)
    counts_toward_multiple_testing: Literal[True] = True
    research_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0

    @model_validator(mode="after")
    def validate_trial(self) -> ResearchTrial:
        for field_name in (
            "trial_id",
            "hypothesis_id",
            "test_family_id",
            "universe_id",
            "neutralization_id",
            "cost_model_id",
            "portfolio_rule_id",
            "blind_period_id",
            "data_snapshot_id",
            "nas_task_id",
        ):
            _require_identifier(getattr(self, field_name), field_name)
        _require_sha(self.factor_formula_hash, "factor_formula_hash")
        _require_sha(self.feature_recipe_hash, "feature_recipe_hash")
        _require_commit(self.code_commit, "code_commit")
        for field_name in (
            "parameter_locked_at",
            "blind_opened_at",
            "submitted_at",
            "started_at",
            "finished_at",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_utc(value, field_name)
        if self.trial_kind == TrialKind.CONFIRMATORY:
            if self.blind_opened_at is not None and self.parameter_locked_at > self.blind_opened_at:
                raise ValueError("confirmatory parameters must be locked before blind is opened")
            if self.blind_invalidated and self.status != TrialStatus.INVALIDATED:
                raise ValueError("invalidated blind trial must have INVALIDATED status")
        if self.status in {TrialStatus.FAILED, TrialStatus.REJECTED, TrialStatus.CANCELLED}:
            if not self.failure_reason:
                raise ValueError("failed, rejected, or cancelled trial requires failure_reason")
        if self.finished_at is not None and self.finished_at < self.submitted_at:
            raise ValueError("finished_at cannot precede submitted_at")
        return self

    @property
    def identity_digest(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={
                "status",
                "decision",
                "failure_reason",
                "blind_opened_at",
                "blind_invalidated",
                "started_at",
                "finished_at",
            },
        )
        return _sha256_json(payload)


def build_trial_id(
    *,
    hypothesis_id: str,
    hypothesis_version: int,
    test_family_id: str,
    factor_formula_hash: str,
    feature_recipe_hash: str,
    direction: int,
    lookback: int,
    horizon: int,
    universe_id: str,
    neutralization_id: str,
    cost_model_id: str,
    portfolio_rule_id: str,
    split_definition: str,
    blind_period_id: str,
    random_seed: int,
    code_commit: str,
    data_snapshot_id: str,
) -> str:
    payload = {
        "hypothesis_id": hypothesis_id,
        "hypothesis_version": hypothesis_version,
        "test_family_id": test_family_id,
        "factor_formula_hash": factor_formula_hash,
        "feature_recipe_hash": feature_recipe_hash,
        "direction": direction,
        "lookback": lookback,
        "horizon": horizon,
        "universe_id": universe_id,
        "neutralization_id": neutralization_id,
        "cost_model_id": cost_model_id,
        "portfolio_rule_id": portfolio_rule_id,
        "split_definition": split_definition,
        "blind_period_id": blind_period_id,
        "random_seed": random_seed,
        "code_commit": code_commit,
        "data_snapshot_id": data_snapshot_id,
    }
    return f"factor-trial-{_sha256_json(payload)[:24]}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_identifier(value: str, field_name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not value or len(value) > 180 or any(character not in allowed for character in value):
        raise ValueError(f"unsafe {field_name}")


def _require_sha(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be lowercase sha256")


def _require_commit(value: str, field_name: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a full lowercase git commit")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be UTC")


def utc_midnight(value: date) -> datetime:
    return datetime(value.year, value.month, value.day, tzinfo=UTC)
