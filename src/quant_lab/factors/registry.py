from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class FactorStatus(StrEnum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    SHADOW = "shadow"
    PRODUCTION = "production"
    RETIRED = "retired"


class FactorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_id: str = Field(min_length=1)
    factor_name: str = Field(min_length=1)
    factor_family: str = Field(min_length=1)
    factor_version: str = Field(min_length=1)
    description: str = Field(min_length=1)

    feature_set: str = Field(default="core", min_length=1)
    feature_version: str = Field(default="v0.1", min_length=1)
    timeframe: str = Field(default="1H", min_length=1)

    input_features: tuple[str, ...]
    template: FactorTemplate
    params: dict[str, Any] = Field(default_factory=dict)

    lookback_bars: int = Field(default=1, ge=1)
    availability_lag_bars: int = Field(default=1, ge=0)
    warmup_bars: int = Field(default=0, ge=0)
    causal: bool = True
    normalization: str | None = "cross_sectional_zscore"
    status: FactorStatus = FactorStatus.CANDIDATE
    owner: str = Field(default="quant", min_length=1)
    expression_hash: str = Field(default="", min_length=1)

    enabled: bool = True
    direction: int = Field(default=1)
    min_cross_section: int = Field(default=3, ge=1)
    clip_abs: float | None = Field(default=10.0, gt=0)
    tags: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def populate_expression_hash(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("expression_hash"):
            return data
        updated = dict(data)
        updated["expression_hash"] = _expression_hash(updated)
        return updated

    @field_validator("direction")
    @classmethod
    def direction_is_valid(cls, value: int) -> int:
        if value not in {-1, 1}:
            raise ValueError("direction must be either -1 or 1")
        return value

    @field_validator("causal")
    @classmethod
    def factor_must_be_causal(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("factor specs must be causal")
        return value

    @field_validator("input_features")
    @classmethod
    def input_features_not_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not cleaned:
            raise ValueError("input_features must not be empty")
        return cleaned

    @property
    def required_bars(self) -> int:
        return self.lookback_bars + self.warmup_bars + self.availability_lag_bars

    @property
    def expression(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "input_features": list(self.input_features),
            "params": self.params,
        }

    def definition_row(self, *, created_at: datetime, source: str) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "factor_name": self.factor_name,
            "factor_family": self.factor_family,
            "factor_version": self.factor_version,
            "description": self.description,
            "feature_set": self.feature_set,
            "feature_version": self.feature_version,
            "timeframe": self.timeframe,
            "input_features_json": json.dumps(list(self.input_features), sort_keys=True),
            "template": self.template,
            "params_json": json.dumps(self.params, sort_keys=True),
            "expression_json": json.dumps(self.expression, sort_keys=True),
            "expression_hash": self.expression_hash,
            "status": self.status.value,
            "lookback_bars": self.lookback_bars,
            "availability_lag_bars": self.availability_lag_bars,
            "warmup_bars": self.warmup_bars,
            "required_bars": self.required_bars,
            "causal": self.causal,
            "normalization": self.normalization,
            "owner": self.owner,
            "direction": self.direction,
            "min_cross_section": self.min_cross_section,
            "clip_abs": self.clip_abs,
            "enabled": self.enabled,
            "tags_json": json.dumps(list(self.tags), sort_keys=True),
            "created_at": created_at,
            "source": source,
        }


def default_factor_registry(
    *,
    feature_set: str = "core",
    feature_version: str = "v0.1",
    factor_version: str = "v0.1",
    timeframe: str = "1H",
) -> list[FactorSpec]:
    def spec(
        *,
        factor_id: str,
        factor_name: str,
        factor_family: str,
        description: str,
        input_features: tuple[str, ...],
        template: FactorTemplate,
        params: dict[str, Any],
        direction: int = 1,
        tags: tuple[str, ...] = (),
        lookback_bars: int | None = None,
    ) -> FactorSpec:
        return FactorSpec(
            factor_id=factor_id,
            factor_name=factor_name,
            factor_family=factor_family,
            factor_version=factor_version,
            description=description,
            feature_set=feature_set,
            feature_version=feature_version,
            timeframe=timeframe,
            input_features=input_features,
            template=template,
            params=params,
            lookback_bars=lookback_bars or _infer_lookback_bars(input_features),
            direction=direction,
            tags=tags,
        )

    return [
        spec(
            factor_id="core.close_return_24",
            factor_name="close_return_24",
            factor_family="momentum",
            description="24-bar close-to-close momentum.",
            input_features=("close_return_24",),
            template="feature",
            params={"feature": "close_return_24"},
            tags=("price", "momentum"),
        ),
        spec(
            factor_id="core.short_reversal_1",
            factor_name="short_reversal_1",
            factor_family="reversal",
            description="Negative one-bar return as short-horizon reversal.",
            input_features=("close_return_1",),
            template="neg_feature",
            params={"feature": "close_return_1"},
            tags=("price", "reversal"),
        ),
        spec(
            factor_id="core.volume_zscore_24",
            factor_name="volume_zscore_24",
            factor_family="volume",
            description="24-bar rolling volume z-score.",
            input_features=("volume_zscore_24",),
            template="feature",
            params={"feature": "volume_zscore_24"},
            tags=("volume",),
        ),
        spec(
            factor_id="core.range_bps",
            factor_name="range_bps",
            factor_family="volatility",
            description="Intrabar high-low range in basis points.",
            input_features=("range_bps",),
            template="feature",
            params={"feature": "range_bps"},
            tags=("volatility",),
        ),
        spec(
            factor_id="core.close_position_in_range",
            factor_name="close_position_in_range",
            factor_family="price_location",
            description="Close location inside high-low range.",
            input_features=("close_position_in_range",),
            template="feature",
            params={"feature": "close_position_in_range"},
            tags=("price", "location"),
        ),
        spec(
            factor_id="core.liquidity_proxy",
            factor_name="liquidity_proxy",
            factor_family="liquidity",
            description="Log quote-volume liquidity proxy.",
            input_features=("liquidity_proxy",),
            template="feature",
            params={"feature": "liquidity_proxy"},
            tags=("liquidity",),
        ),
        spec(
            factor_id="core.momentum_vol_adjusted_24",
            factor_name="momentum_vol_adjusted_24",
            factor_family="risk_adjusted_momentum",
            description="24-bar momentum divided by 24-bar rolling volatility.",
            input_features=("close_return_24", "rolling_volatility_24"),
            template="vol_adjusted",
            params={
                "return_feature": "close_return_24",
                "vol_feature": "rolling_volatility_24",
            },
            tags=("momentum", "volatility_adjusted"),
        ),
        spec(
            factor_id="core.volume_momentum_4",
            factor_name="volume_momentum_4",
            factor_family="volume_price_confirm",
            description="4-bar momentum confirmed by 24-bar volume z-score.",
            input_features=("close_return_4", "volume_zscore_24"),
            template="product",
            params={"left": "close_return_4", "right": "volume_zscore_24"},
            tags=("momentum", "volume"),
        ),
        spec(
            factor_id="core.range_vol_ratio",
            factor_name="range_vol_ratio",
            factor_family="volatility_expansion",
            description="Intrabar range bps divided by rolling volatility.",
            input_features=("range_bps", "rolling_volatility_24"),
            template="range_vol_ratio",
            params={"range_feature": "range_bps", "vol_feature": "rolling_volatility_24"},
            tags=("range", "volatility"),
        ),
        spec(
            factor_id="core.range_close_location",
            factor_name="range_close_location",
            factor_family="breakout_location",
            description="Range expansion weighted by close location inside the bar.",
            input_features=("range_bps", "close_position_in_range"),
            template="range_location",
            params={
                "range_feature": "range_bps",
                "location_feature": "close_position_in_range",
            },
            tags=("breakout", "range"),
        ),
        spec(
            factor_id="core.liquidity_adjusted_momentum_24",
            factor_name="liquidity_adjusted_momentum_24",
            factor_family="liquidity_adjusted_momentum",
            description="24-bar momentum divided by liquidity proxy.",
            input_features=("close_return_24", "liquidity_proxy"),
            template="liquidity_adjusted",
            params={
                "return_feature": "close_return_24",
                "liquidity_feature": "liquidity_proxy",
            },
            tags=("momentum", "liquidity"),
        ),
        spec(
            factor_id="core.mean_reversion_vol_adjusted_4",
            factor_name="mean_reversion_vol_adjusted_4",
            factor_family="risk_adjusted_reversal",
            description="Negative 4-bar return divided by 24-bar rolling volatility.",
            input_features=("close_return_4", "rolling_volatility_24"),
            template="vol_adjusted",
            params={"return_feature": "close_return_4", "vol_feature": "rolling_volatility_24"},
            direction=-1,
            tags=("reversal", "volatility_adjusted"),
        ),
    ]


def discover_factor_specs(
    available_features: Iterable[str],
    *,
    feature_set: str = "core",
    feature_version: str = "v0.1",
    factor_version: str = "v0.1",
    timeframe: str = "1H",
    max_factors: int = 200,
) -> list[FactorSpec]:
    available = {str(name).strip() for name in available_features if str(name).strip()}
    specs: list[FactorSpec] = []

    for item in default_factor_registry(
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
    ):
        if set(item.input_features).issubset(available):
            specs.append(item)

    existing_ids = {item.factor_id for item in specs}
    for feature_name in sorted(available):
        factor_id = f"auto.single.{feature_name}"
        if factor_id in existing_ids:
            continue
        specs.append(
            FactorSpec(
                factor_id=factor_id,
                factor_name=feature_name,
                factor_family="auto_single_feature",
                factor_version=factor_version,
                description=f"Auto-discovered single-feature factor from {feature_name}.",
                feature_set=feature_set,
                feature_version=feature_version,
                timeframe=timeframe,
                input_features=(feature_name,),
                template="feature",
                params={"feature": feature_name},
                lookback_bars=_infer_lookback_bars((feature_name,)),
                tags=("auto", "single"),
            )
        )
        existing_ids.add(factor_id)
        if len(specs) >= max_factors:
            return specs[:max_factors]

    return specs[:max_factors]


def _expression_hash(values: dict[str, Any]) -> str:
    payload = {
        "template": values.get("template"),
        "input_features": list(values.get("input_features") or ()),
        "params": values.get("params") or {},
        "direction": values.get("direction", 1),
        "clip_abs": values.get("clip_abs", 10.0),
        "lookback_bars": values.get("lookback_bars", 1),
        "availability_lag_bars": values.get("availability_lag_bars", 1),
        "warmup_bars": values.get("warmup_bars", 0),
        "causal": values.get("causal", True),
        "normalization": values.get("normalization", "cross_sectional_zscore"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _infer_lookback_bars(input_features: Iterable[str]) -> int:
    lookbacks = [1]
    for feature in input_features:
        tokens = str(feature).replace("-", "_").split("_")
        for token in reversed(tokens):
            if token.isdigit():
                lookbacks.append(max(1, int(token)))
                break
    return max(lookbacks)
