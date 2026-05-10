from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GateStatus(StrEnum):
    DEAD = "DEAD"
    QUARANTINE = "QUARANTINE"
    PAPER_READY = "PAPER_READY"
    LIVE_READY = "LIVE_READY"


class RiskAction(StrEnum):
    ALLOW = "ALLOW"
    SELL_ONLY = "SELL_ONLY"
    ABORT = "ABORT"


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    normalized = value.astimezone(UTC)
    return normalized


class MarketBar(ContractModel):
    venue: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    market_type: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    ts: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    quote_volume: float | None = Field(default=None, ge=0)
    source: str = Field(min_length=1)
    ingest_ts: datetime
    is_closed: bool = True

    @field_validator("ts", "ingest_ts")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def validate_bar(self) -> "MarketBar":
        if not self.is_closed:
            raise ValueError("market bars used by quant-lab must be closed")
        if self.high < self.low:
            raise ValueError("bar high must be greater than or equal to low")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("bar high/low are inconsistent with open/close")
        return self


class FeatureValue(ContractModel):
    feature_set: str = Field(min_length=1)
    feature_name: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    ts: datetime
    value: float | None
    lookback_bars: int = Field(ge=0)
    input_dataset_version: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    created_at: datetime

    @field_validator("ts", "created_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        return require_utc(value)

    @field_validator("value")
    @classmethod
    def value_is_finite_or_null(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("feature value must be finite or null")
        return value

    @model_validator(mode="after")
    def validate_created_after_feature_ts(self) -> "FeatureValue":
        if self.created_at < self.ts:
            raise ValueError("created_at must not be earlier than feature timestamp")
        return self


class CostEstimate(ContractModel):
    symbol: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    notional_usdt: float = Field(gt=0)
    cost_bps: float = Field(ge=0)
    fallback_level: str = Field(min_length=1)
    bucket_id: str | None = None


class FillEvent(ContractModel):
    venue: str = "okx"
    inst_type: str = Field(min_length=1)
    inst_id: str = Field(min_length=1)
    trade_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    side: str = Field(min_length=1)
    fill_price: float = Field(ge=0)
    fill_size: float = Field(ge=0)
    fee: float
    fee_currency: str = Field(min_length=1)
    ts: datetime
    source: str = Field(default="okx_readonly_private", min_length=1)
    liquidity: str | None = None

    @field_validator("ts")
    @classmethod
    def ts_is_utc(cls, value: datetime) -> datetime:
        return require_utc(value)


class AccountBill(ContractModel):
    venue: str = "okx"
    bill_id: str = Field(min_length=1)
    ccy: str = Field(min_length=1)
    amount: float
    balance: float
    bill_type: str = Field(min_length=1)
    sub_type: str = Field(min_length=1)
    ts: datetime
    source: str = Field(default="okx_readonly_private", min_length=1)

    @field_validator("ts")
    @classmethod
    def ts_is_utc(cls, value: datetime) -> datetime:
        return require_utc(value)


class AlphaEvidence(ContractModel):
    alpha_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    cost_model_version: str = Field(min_length=1)
    universe_id: str = Field(min_length=1)
    start_ts: datetime
    end_ts: datetime
    coverage: float = Field(ge=0, le=1)
    ic_mean: float
    ic_tstat: float
    rank_ic_mean: float
    rank_ic_tstat: float
    edge_cost_ratio: float
    oos_sharpe: float
    oos_sortino: float | None = None
    oos_cagr: float | None = None
    oos_max_drawdown: float = Field(ge=0)
    profit_factor: float | None = None
    turnover: float = Field(ge=0)
    cost_ratio: float | None = None
    profitable_folds_ratio: float = Field(ge=0, le=1)
    train_oos_decay: float = Field(ge=0)
    pbo_score: float | None = None
    paper_days: int = Field(ge=0)
    paper_slippage_coverage: float = Field(ge=0, le=1)
    created_at: datetime

    @field_validator("start_ts", "end_ts", "created_at")
    @classmethod
    def evidence_timestamps_are_utc(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def validate_window(self) -> "AlphaEvidence":
        if self.end_ts <= self.start_ts:
            raise ValueError("end_ts must be after start_ts")
        return self

    @classmethod
    def example_live_ready(cls) -> "AlphaEvidence":
        return cls(
            alpha_id="example-alpha",
            version="v1",
            data_version="okx-market-bar-2026-05-10",
            feature_version="features-v1",
            cost_model_version="costs-v1",
            universe_id="okx-spot-major",
            start_ts=datetime(2026, 1, 1, tzinfo=UTC),
            end_ts=datetime(2026, 5, 10, tzinfo=UTC),
            coverage=0.99,
            ic_mean=0.04,
            ic_tstat=3.1,
            rank_ic_mean=0.05,
            rank_ic_tstat=3.4,
            edge_cost_ratio=2.4,
            oos_sharpe=1.2,
            oos_sortino=1.5,
            oos_cagr=0.18,
            oos_max_drawdown=0.12,
            profit_factor=1.4,
            turnover=0.35,
            cost_ratio=0.42,
            profitable_folds_ratio=0.75,
            train_oos_decay=0.25,
            pbo_score=0.2,
            paper_days=21,
            paper_slippage_coverage=0.92,
            created_at=datetime(2026, 5, 10, tzinfo=UTC),
        )


class GateDecision(ContractModel):
    alpha_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    gate_version: str = Field(min_length=1)
    status: GateStatus
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any]
    next_action: str = Field(min_length=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        return require_utc(value)


class RiskPermission(ContractModel):
    strategy: str = Field(min_length=1)
    version: str = Field(min_length=1)
    permission: RiskAction
    allowed_modes: list[str] = Field(default_factory=list)
    max_gross_exposure: float = Field(ge=0)
    max_single_weight: float = Field(ge=0)
    cost_model_version: str = Field(min_length=1)
    gate_version: str = Field(min_length=1)
    reasons: list[str] = Field(default_factory=list)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        return require_utc(value)
