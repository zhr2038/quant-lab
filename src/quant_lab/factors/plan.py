from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_lab.factors.registry import (
    FactorSpec,
    FactorStatus,
    FactorTemplate,
    apply_factor_semantic_lineage,
    discover_factor_specs,
)

FACTOR_PLAN_SCHEMA_VERSION = "quant_lab_factor_plan.v1"
FACTOR_FACTORY_DECISION_POLICY = "factor_factory.main_49ad71f.v1"

FactorParameterValue = str | int | float | bool | None


class StrictPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EffectiveFactorExpression(StrictPlanModel):
    template: FactorTemplate
    input_features: tuple[str, ...] = Field(min_length=1)
    params: dict[str, FactorParameterValue] = Field(default_factory=dict)


class EffectiveFactorSpec(StrictPlanModel):
    factor_id: str = Field(min_length=1, max_length=180)
    factor_name: str = Field(min_length=1, max_length=180)
    factor_family: str = Field(min_length=1, max_length=180)
    factor_version: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1024)
    feature_set: str = Field(min_length=1, max_length=80)
    feature_version: str = Field(min_length=1, max_length=80)
    timeframe: str = Field(min_length=1, max_length=32)
    input_features: tuple[str, ...] = Field(min_length=1)
    template: FactorTemplate
    params: dict[str, FactorParameterValue] = Field(default_factory=dict)
    expression: EffectiveFactorExpression
    expression_hash: str = Field(min_length=64, max_length=64)
    factor_hash: str = Field(min_length=64, max_length=64)
    factor_formula_hash: str = Field(min_length=64, max_length=64)
    formula_hash: str = Field(min_length=64, max_length=64)
    operator_graph_hash: str = Field(min_length=64, max_length=64)
    canonical_factor_id: str = Field(min_length=1, max_length=180)
    feature_dependencies: tuple[str, ...] = Field(min_length=1)
    duplicate_of: str = Field(max_length=180)
    correlation_cluster_id: str = Field(min_length=1, max_length=180)
    effective_independence_weight: float = Field(gt=0.0, le=1.0)
    independence_weight: float = Field(gt=0.0, le=1.0)
    status: FactorStatus
    owner: str = Field(min_length=1, max_length=120)
    lookback_bars: int = Field(ge=1)
    availability_lag_bars: int = Field(ge=0)
    warmup_bars: int = Field(ge=0)
    required_bars: int = Field(ge=1)
    causal: Literal[True] = True
    normalization: str | None = None
    direction: Literal[-1, 1]
    min_cross_section: int = Field(ge=1)
    clip_abs: float | None = Field(default=None, gt=0.0)
    enabled: bool
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_semantics(self) -> EffectiveFactorSpec:
        for field_name in (
            "expression_hash",
            "factor_hash",
            "factor_formula_hash",
            "formula_hash",
            "operator_graph_hash",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.expression.template != self.template:
            raise ValueError("factor plan expression template mismatch")
        if self.expression.input_features != self.input_features:
            raise ValueError("factor plan expression input_features mismatch")
        if self.expression.params != self.params:
            raise ValueError("factor plan expression params mismatch")
        if self.factor_hash != self.expression_hash:
            raise ValueError("factor plan factor_hash mismatch")
        if self.formula_hash != self.factor_formula_hash:
            raise ValueError("factor plan formula_hash mismatch")
        if self.independence_weight != self.effective_independence_weight:
            raise ValueError("factor plan independence weight mismatch")
        if self.required_bars != (
            self.lookback_bars + self.availability_lag_bars + self.warmup_bars
        ):
            raise ValueError("factor plan required_bars mismatch")
        reconstructed = self._identity_factor_spec()
        expected = {
            "expression_hash": reconstructed.expression_hash,
            "factor_formula_hash": reconstructed.factor_formula_hash,
            "operator_graph_hash": reconstructed.operator_graph_hash,
            "canonical_factor_id": reconstructed.canonical_factor_id,
            "feature_dependencies": reconstructed.feature_dependencies,
        }
        observed = {
            "expression_hash": self.expression_hash,
            "factor_formula_hash": self.factor_formula_hash,
            "operator_graph_hash": self.operator_graph_hash,
            "canonical_factor_id": self.canonical_factor_id,
            "feature_dependencies": self.feature_dependencies,
        }
        if observed != expected:
            raise ValueError("factor plan semantic identity mismatch")
        return self

    def _identity_factor_spec(self) -> FactorSpec:
        return FactorSpec(
            factor_id=self.factor_id,
            factor_name=self.factor_name,
            factor_family=self.factor_family,
            factor_version=self.factor_version,
            description=self.description,
            feature_set=self.feature_set,
            feature_version=self.feature_version,
            timeframe=self.timeframe,
            input_features=self.input_features,
            template=self.template,
            params=dict(self.params),
            lookback_bars=self.lookback_bars,
            availability_lag_bars=self.availability_lag_bars,
            warmup_bars=self.warmup_bars,
            causal=self.causal,
            normalization=self.normalization,
            status=self.status,
            owner=self.owner,
            enabled=self.enabled,
            direction=self.direction,
            min_cross_section=self.min_cross_section,
            clip_abs=self.clip_abs,
            tags=self.tags,
        )

    def to_factor_spec(self) -> FactorSpec:
        return FactorSpec(
            factor_id=self.factor_id,
            factor_name=self.factor_name,
            factor_family=self.factor_family,
            factor_version=self.factor_version,
            description=self.description,
            feature_set=self.feature_set,
            feature_version=self.feature_version,
            timeframe=self.timeframe,
            input_features=self.input_features,
            template=self.template,
            params=dict(self.params),
            lookback_bars=self.lookback_bars,
            availability_lag_bars=self.availability_lag_bars,
            warmup_bars=self.warmup_bars,
            causal=self.causal,
            normalization=self.normalization,
            status=self.status,
            owner=self.owner,
            expression_hash=self.expression_hash,
            canonical_factor_id=self.canonical_factor_id,
            factor_formula_hash=self.factor_formula_hash,
            feature_dependencies=self.feature_dependencies,
            operator_graph_hash=self.operator_graph_hash,
            duplicate_of=self.duplicate_of,
            correlation_cluster_id=self.correlation_cluster_id,
            effective_independence_weight=self.effective_independence_weight,
            enabled=self.enabled,
            direction=self.direction,
            min_cross_section=self.min_cross_section,
            clip_abs=self.clip_abs,
            tags=self.tags,
        )

    @classmethod
    def from_factor_spec(cls, spec: FactorSpec) -> EffectiveFactorSpec:
        params = {
            str(key): _factor_parameter_value(value) for key, value in sorted(spec.params.items())
        }
        return cls(
            factor_id=spec.factor_id,
            factor_name=spec.factor_name,
            factor_family=spec.factor_family,
            factor_version=spec.factor_version,
            description=spec.description,
            feature_set=spec.feature_set,
            feature_version=spec.feature_version,
            timeframe=spec.timeframe,
            input_features=spec.input_features,
            template=spec.template,
            params=params,
            expression=EffectiveFactorExpression(
                template=spec.template,
                input_features=spec.input_features,
                params=params,
            ),
            expression_hash=spec.expression_hash,
            factor_hash=spec.expression_hash,
            factor_formula_hash=spec.factor_formula_hash,
            formula_hash=spec.factor_formula_hash,
            operator_graph_hash=spec.operator_graph_hash,
            canonical_factor_id=spec.canonical_factor_id,
            feature_dependencies=spec.feature_dependencies,
            duplicate_of=spec.duplicate_of,
            correlation_cluster_id=spec.correlation_cluster_id,
            effective_independence_weight=spec.effective_independence_weight,
            independence_weight=spec.effective_independence_weight,
            status=spec.status,
            owner=spec.owner,
            lookback_bars=spec.lookback_bars,
            availability_lag_bars=spec.availability_lag_bars,
            warmup_bars=spec.warmup_bars,
            required_bars=spec.required_bars,
            causal=True,
            normalization=spec.normalization,
            direction=spec.direction,
            min_cross_section=spec.min_cross_section,
            clip_abs=spec.clip_abs,
            enabled=spec.enabled,
            tags=spec.tags,
        )


class EffectiveFactorPlan(StrictPlanModel):
    schema_version: Literal["quant_lab_factor_plan.v1"] = FACTOR_PLAN_SCHEMA_VERSION
    feature_set: str = Field(min_length=1, max_length=80)
    feature_version: str = Field(min_length=1, max_length=80)
    factor_version: str = Field(min_length=1, max_length=80)
    timeframe: str = Field(min_length=1, max_length=32)
    max_factors: int = Field(ge=1, le=200)
    include_legacy_enumeration: Literal[True] = True
    decision_policy: Literal["factor_factory.main_49ad71f.v1"] = FACTOR_FACTORY_DECISION_POLICY
    feature_names: tuple[str, ...]
    factor_specs: tuple[EffectiveFactorSpec, ...]
    factor_count: int = Field(ge=0, le=200)
    plan_digest: str = Field(min_length=64, max_length=64)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("factor plan created_at must be UTC")
        return value

    @model_validator(mode="after")
    def validate_plan(self) -> EffectiveFactorPlan:
        _require_commit(self.quant_lab_commit)
        _require_sha256(self.plan_digest, "plan_digest")
        if self.factor_count != len(self.factor_specs):
            raise ValueError("factor plan factor_count mismatch")
        if self.factor_count > self.max_factors:
            raise ValueError("factor plan exceeds max_factors")
        if tuple(sorted(set(self.feature_names))) != self.feature_names:
            raise ValueError("factor plan feature_names must be sorted and unique")
        ids = [item.factor_id for item in self.factor_specs]
        if len(ids) != len(set(ids)):
            raise ValueError("factor plan factor_id values must be unique")
        available = set(self.feature_names)
        for item in self.factor_specs:
            if (
                item.feature_set != self.feature_set
                or item.feature_version != self.feature_version
                or item.factor_version != self.factor_version
                or item.timeframe != self.timeframe
            ):
                raise ValueError("factor plan spec scope mismatch")
            if not set(item.input_features).issubset(available):
                raise ValueError(f"factor plan input feature missing: {item.factor_id}")
        recomputed = apply_factor_semantic_lineage(
            item._identity_factor_spec() for item in self.factor_specs
        )
        lineage = {
            item.factor_id: (
                item.duplicate_of,
                item.correlation_cluster_id,
                item.effective_independence_weight,
            )
            for item in recomputed
        }
        for item in self.factor_specs:
            if lineage[item.factor_id] != (
                item.duplicate_of,
                item.correlation_cluster_id,
                item.effective_independence_weight,
            ):
                raise ValueError("factor plan semantic lineage mismatch")
        if factor_plan_digest(self) != self.plan_digest:
            raise ValueError("factor plan digest mismatch")
        return self

    def factor_spec_models(self) -> list[FactorSpec]:
        return apply_factor_semantic_lineage(item.to_factor_spec() for item in self.factor_specs)


def build_effective_factor_plan(
    available_features: list[str] | tuple[str, ...],
    *,
    feature_set: str,
    feature_version: str,
    factor_version: str,
    timeframe: str,
    max_factors: int,
    quant_lab_commit: str,
    created_at: datetime | None = None,
) -> EffectiveFactorPlan:
    feature_names = tuple(
        sorted({str(item).strip() for item in available_features if str(item).strip()})
    )
    specs = discover_factor_specs(
        feature_names,
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        max_factors=max_factors,
        include_legacy_enumeration=True,
    )
    provisional = EffectiveFactorPlan.model_construct(
        schema_version=FACTOR_PLAN_SCHEMA_VERSION,
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        max_factors=max_factors,
        feature_names=feature_names,
        factor_specs=tuple(EffectiveFactorSpec.from_factor_spec(item) for item in specs),
        factor_count=len(specs),
        plan_digest="0" * 64,
        quant_lab_commit=quant_lab_commit,
        created_at=created_at or datetime.now(UTC),
    )
    payload = provisional.model_dump(mode="json")
    payload["plan_digest"] = factor_plan_digest(provisional)
    return EffectiveFactorPlan.model_validate(payload)


def factor_plan_digest(plan: EffectiveFactorPlan) -> str:
    payload = plan.model_dump(mode="json", exclude={"plan_digest"})
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _factor_parameter_value(value: object) -> FactorParameterValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported factor parameter type: {type(value).__name__}")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be lowercase sha256")


def _require_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("quant_lab_commit must be a full lowercase git commit")
