from __future__ import annotations

from datetime import datetime, timedelta
from math import isclose
from typing import Literal

from pydantic import Field, field_validator, model_validator

from quant_lab.contracts.models import require_utc
from quant_lab.decision.contracts_base import Contract as Contract
from quant_lab.decision.contracts_base import CostObservation as CostObservation
from quant_lab.decision.current_cost_contracts import CurrentCostObservation

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
HORIZONS = (4, 24)
EXPERIMENT_VERSION = "trend-reference-1h-v1"
Action = Literal["DEFER", "REVIEW_ENTRY", "KEEP_BASELINE", "NO_VIEW"]


class HourBar(Contract):
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    ts: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    ingest_ts: datetime

    @field_validator("ts", "ingest_ts")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def consistent(self) -> HourBar:
        if self.ts.minute or self.ts.second or self.ts.microsecond:
            raise ValueError("hour bar must start on a UTC hour")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("inconsistent OHLC")
        if self.low > self.high:
            raise ValueError("low exceeds high")
        return self


class InputSnapshot(Contract):
    schema_version: Literal["qlab.decision.input.v1"] = "qlab.decision.input.v1"
    snapshot_id: str = Field(pattern=r"^input-[a-f0-9]{64}$")
    generated_at: datetime
    producer_commit: str = Field(min_length=7, max_length=64)
    bars: list[HourBar] = Field(max_length=4_096)
    costs: list[CurrentCostObservation | CostObservation] = Field(max_length=4)
    warnings: list[str] = Field(default_factory=list, max_length=40)
    signature: str = Field(min_length=1, max_length=200)

    @field_validator("generated_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def closed_and_unique(self) -> InputSnapshot:
        keys = {(bar.symbol, bar.ts) for bar in self.bars}
        if len(keys) != len(self.bars):
            raise ValueError("duplicate input bars")
        if len({cost.symbol for cost in self.costs}) != len(self.costs):
            raise ValueError("duplicate costs")
        for bar in self.bars:
            if bar.ts + timedelta(hours=1) > self.generated_at:
                raise ValueError("unclosed input bar")
            if bar.ingest_ts > self.generated_at:
                raise ValueError("input was not available at generation time")
        return self


class Distribution(Contract):
    samples: int = Field(ge=0)
    non_overlapping_samples: int = Field(ge=0)
    calendar_days: int = Field(ge=0)
    first_signal_at: datetime | None = None
    last_signal_at: datetime | None = None
    gross_mean_bps: float | None = None
    gross_p10_bps: float | None = None
    gross_p50_bps: float | None = None
    gross_p90_bps: float | None = None
    net_mean_bps: float | None = None
    net_p10_bps: float | None = None
    net_p50_bps: float | None = None
    net_p90_bps: float | None = None
    double_cost_mean_bps: float | None = None
    historical_positive_fraction: float | None = Field(default=None, ge=0, le=1)
    chronological_tail_samples: int = Field(default=0, ge=0)
    chronological_tail_net_mean_bps: float | None = None
    cost_basis: Literal["current_roundtrip_cost_scenario"] = "current_roundtrip_cost_scenario"
    interpretation: Literal["historical_description_not_calibrated_forecast"] = (
        "historical_description_not_calibrated_forecast"
    )

    @field_validator("first_signal_at", "last_signal_at")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None

    @model_validator(mode="after")
    def counts_and_quantiles(self) -> Distribution:
        if not self.chronological_tail_samples <= self.non_overlapping_samples <= self.samples:
            raise ValueError("inconsistent sample counts")
        for prefix in ("gross", "net"):
            values = [getattr(self, f"{prefix}_{q}_bps") for q in ("p10", "p50", "p90")]
            if all(value is not None for value in values) and values != sorted(values):
                raise ValueError("quantiles are out of order")
        if self.non_overlapping_samples == 0 and self.gross_mean_bps is not None:
            raise ValueError("empty sample cannot have an estimated mean")
        return self


class Advice(Contract):
    advice_id: str = Field(pattern=r"^advice-[a-f0-9]{64}$")
    opportunity_id: str = Field(pattern=r"^opportunity-[a-f0-9]{64}$")
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    horizon_hours: Literal[4, 24]
    generated_at: datetime
    market_asof: datetime | None
    expires_at: datetime
    reference_entry_at: datetime | None
    reference_exit_at: datetime | None
    action: Action
    reason_codes: list[str] = Field(min_length=1, max_length=30)
    explanation: str = Field(min_length=1, max_length=1_000)
    invalidation_conditions: list[str] = Field(min_length=1, max_length=20)
    context: Literal["trend_up", "trend_down", "unavailable"]
    last_close: float | None = Field(default=None, gt=0)
    trend_24h_bps: float | None = None
    volatility_24h_bps: float | None = Field(default=None, ge=0)
    cost: CurrentCostObservation | CostObservation
    distribution: Distribution
    input_snapshot_id: str = Field(pattern=r"^input-[a-f0-9]{64}$")
    data_snapshot_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    experiment_version: Literal["trend-reference-1h-v1"] = EXPERIMENT_VERSION
    evidence_tier: Literal["historical_descriptive"] = "historical_descriptive"
    consumption_scope: Literal["research_only"] = "research_only"
    live_order_effect: Literal["none"] = "none"
    baseline_action: Literal["preserve_v5_decision"] = "preserve_v5_decision"
    calibrated_profit_probability: None = None

    @field_validator(
        "generated_at", "market_asof", "expires_at", "reference_entry_at", "reference_exit_at"
    )
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None

    @model_validator(mode="after")
    def coherent_times(self) -> Advice:
        if self.expires_at < self.generated_at:
            raise ValueError("expiry precedes generation")
        if self.expires_at > self.generated_at + timedelta(hours=1):
            raise ValueError("advice exceeds maximum lifetime")
        if self.market_asof is not None and self.market_asof > self.generated_at:
            raise ValueError("future market context")
        if self.reference_entry_at is not None:
            if self.reference_exit_at != self.reference_entry_at + timedelta(
                hours=self.horizon_hours
            ):
                raise ValueError("reference horizon mismatch")
            if self.action != "NO_VIEW" and self.generated_at >= self.reference_entry_at:
                raise ValueError("reference entry was already known")
            if self.action != "NO_VIEW" and self.expires_at > self.reference_entry_at:
                raise ValueError("advice cannot remain valid after reference entry")
        cost = self.cost.roundtrip_bps
        if cost is not None:
            for name in ("mean", "p10", "p50", "p90"):
                gross = getattr(self.distribution, f"gross_{name}_bps")
                net = getattr(self.distribution, f"net_{name}_bps")
                if gross is not None and (net is None or not isclose(gross - cost, net)):
                    raise ValueError("net result must subtract roundtrip cost exactly once")
        return self


class ForwardGroup(Contract):
    horizon_hours: Literal[4, 24]
    action: Action
    observations: int = Field(ge=0)
    gross_mean_bps: float | None = None
    net_mean_bps: float | None = None


class ForwardSummary(Contract):
    interpretation: Literal["prospective_asset_labels_not_v5_account_pnl"] = (
        "prospective_asset_labels_not_v5_account_pnl"
    )
    started_at: datetime | None = None
    registered_opportunities: int = Field(default=0, ge=0)
    registered_horizon_observations: int = Field(default=0, ge=0)
    matured_observations: int = Field(default=0, ge=0)
    waiting_observations: int = Field(default=0, ge=0)
    missing_label_observations: int = Field(default=0, ge=0)
    net_mean_bps: None = None
    by_group: list[ForwardGroup] = Field(default_factory=list, max_length=8)
    v5_received_count: None = None
    incremental_account_pnl: None = None
    baseline_comparison_status: Literal["not_connected"] = "not_connected"

    @field_validator("started_at")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None

    @model_validator(mode="after")
    def counts(self) -> ForwardSummary:
        if self.registered_horizon_observations != (
            self.matured_observations + self.waiting_observations + self.missing_label_observations
        ):
            raise ValueError("forward observation counts do not reconcile")
        if self.registered_opportunities > self.registered_horizon_observations:
            raise ValueError("forward opportunity count exceeds observations")
        return self


class AnalysisResult(Contract):
    schema_version: Literal["qlab.decision.result.v1"] = "qlab.decision.result.v1"
    result_id: str = Field(pattern=r"^result-[a-f0-9]{64}$")
    generated_at: datetime
    input_snapshot_id: str = Field(pattern=r"^input-[a-f0-9]{64}$")
    worker_commit: str = Field(min_length=7, max_length=64)
    experiment_version: Literal["trend-reference-1h-v1"] = EXPERIMENT_VERSION
    advice: list[Advice] = Field(min_length=8, max_length=8)
    forward: ForwardSummary
    history_rows: int = Field(ge=0, le=100_000)
    runtime_seconds: float = Field(ge=0)
    peak_rss_mib: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list, max_length=40)
    signature: str = Field(min_length=1, max_length=200)

    @field_validator("generated_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def exact_scope(self) -> AnalysisResult:
        if {(a.symbol, a.horizon_hours) for a in self.advice} != {
            (symbol, horizon) for symbol in SYMBOLS for horizon in HORIZONS
        }:
            raise ValueError("result must contain each symbol/horizon exactly once")
        if any(a.input_snapshot_id != self.input_snapshot_id for a in self.advice):
            raise ValueError("mixed input provenance")
        if any(a.generated_at > self.generated_at for a in self.advice):
            raise ValueError("result precedes its advice")
        return self
