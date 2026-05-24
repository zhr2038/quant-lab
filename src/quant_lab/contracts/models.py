from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_lab.contracts.v5_quant_lab import (
    RISK_PERMISSION_CONTRACT_VERSION,
    V5_COST_ESTIMATE_RESPONSE_SCHEMA_VERSION,
    V5_QUANT_LAB_CONTRACT_VERSION,
    V5_RISK_PERMISSION_RESPONSE_SCHEMA_VERSION,
)
from quant_lab.symbols import normalize_symbol


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


class RiskPermissionStatus(StrEnum):
    ACTIVE_ALLOW = "ACTIVE_ALLOW"
    ACTIVE_SELL_ONLY = "ACTIVE_SELL_ONLY"
    ACTIVE_ABORT = "ACTIVE_ABORT"
    STALE_ALLOW = "STALE_ALLOW"
    STALE_SELL_ONLY = "STALE_SELL_ONLY"
    STALE_ABORT = "STALE_ABORT"
    EXPIRED_ALLOW = "EXPIRED_ALLOW"
    EXPIRED_SELL_ONLY = "EXPIRED_SELL_ONLY"
    EXPIRED_ABORT = "EXPIRED_ABORT"
    NO_FRESH_PERMISSION = "NO_FRESH_PERMISSION"


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

    @field_validator("symbol")
    @classmethod
    def normalize_market_symbol(cls, value: str) -> str:
        return normalize_symbol(value)

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
    timeframe: str = Field(default="1H", min_length=1)
    ts: datetime
    value: float | None
    lookback_bars: int = Field(ge=0)
    input_dataset_version: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    created_at: datetime
    source: str = Field(default="market_bar", min_length=1)
    is_valid: bool = True
    invalid_reason: str | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_feature_symbol(cls, value: str) -> str:
        return normalize_symbol(value)

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
        if self.value is None and self.is_valid and self.invalid_reason is None:
            raise ValueError("null feature values must be marked invalid or carry invalid_reason")
        if self.invalid_reason and self.is_valid:
            raise ValueError("feature with invalid_reason must not be marked valid")
        return self


class CostEstimate(ContractModel):
    contract_version: str = V5_QUANT_LAB_CONTRACT_VERSION
    schema_version: str = V5_COST_ESTIMATE_RESPONSE_SCHEMA_VERSION
    symbol: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    notional_usdt: float = Field(gt=0)
    quantile: str = Field(default="p75", pattern="^p(50|75|90)$")
    requested_quantile: str | None = Field(default=None, pattern="^p(50|75|90)$")
    fee_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)
    spread_bps: float = Field(default=0.0, ge=0)
    total_cost_bps: float = Field(default=0.0, ge=0)
    cost_bps: float = Field(default=0.0, ge=0)
    fallback_level: str = Field(min_length=1)
    source: str = Field(default="unknown", min_length=1)
    sample_count: int = Field(default=0, ge=0)
    cost_model_version: str = Field(default="unknown", min_length=1)
    bucket_id: str | None = None
    normalized_symbol: str | None = None
    requested_regime: str | None = None
    matched_regime: str | None = None
    cost_source: str | None = None
    total_cost_bps_p50: float | None = Field(default=None, ge=0)
    total_cost_bps_p75: float | None = Field(default=None, ge=0)
    total_cost_bps_p90: float | None = Field(default=None, ge=0)
    selected_total_cost_bps: float | None = Field(default=None, ge=0)
    fallback_reason: str | None = None
    degraded_reason: str | None = None
    degraded_cost_model: bool = False
    sample_size: int | None = Field(default=None, ge=0)
    as_of_ts: datetime | None = None
    fee_source: str = "unknown"
    spread_source: str = "unknown"
    slippage_source: str = "unknown"
    delay_cost_bps: float = Field(default=0.0, ge=0)
    delay_cost_source: str = "none"
    uncertainty_buffer_bps: float = Field(default=0.0, ge=0)
    one_way_all_in_cost_bps: float = Field(default=0.0, ge=0)
    roundtrip_all_in_cost_bps: float = Field(default=0.0, ge=0)
    cost_quality: str = "unknown"
    cost_trusted_for_paper: bool = False
    cost_trusted_for_live: bool = False
    cost_trusted_for_live_canary: bool = False
    cost_trusted_for_live_scale: bool = False
    cost_trust_level: str = "BLOCK"
    cost_trust_block_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def sync_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            if "total_cost_bps" not in normalized and "cost_bps" in normalized:
                normalized["total_cost_bps"] = normalized["cost_bps"]
            if "cost_bps" not in normalized and "total_cost_bps" in normalized:
                normalized["cost_bps"] = normalized["total_cost_bps"]
            normalized["symbol"] = normalize_symbol(normalized.get("symbol"))
            normalized.setdefault("normalized_symbol", normalized["symbol"])
            normalized.setdefault("requested_regime", normalized.get("regime"))
            normalized.setdefault("matched_regime", normalized.get("regime"))
            normalized.setdefault("cost_source", normalized.get("source"))
            normalized.setdefault("requested_quantile", normalized.get("quantile"))
            normalized.setdefault("selected_total_cost_bps", normalized.get("total_cost_bps"))
            normalized.setdefault("fallback_reason", normalized.get("fallback_level"))
            normalized.setdefault("degraded_reason", "none")
            normalized.setdefault("sample_size", normalized.get("sample_count"))
            normalized.setdefault("fee_source", "unknown")
            normalized.setdefault("spread_source", "unknown")
            normalized.setdefault("slippage_source", "unknown")
            normalized.setdefault("delay_cost_bps", 0.0)
            normalized.setdefault("delay_cost_source", "none")
            normalized.setdefault("uncertainty_buffer_bps", 0.0)
            normalized.setdefault(
                "one_way_all_in_cost_bps",
                _cost_estimate_one_way_all_in(normalized),
            )
            normalized.setdefault(
                "roundtrip_all_in_cost_bps",
                float(normalized.get("one_way_all_in_cost_bps") or 0.0) * 2.0,
            )
            normalized.setdefault("cost_quality", _cost_estimate_quality(normalized))
            normalized.setdefault(
                "cost_trusted_for_paper",
                _cost_estimate_trusted_for_paper(normalized),
            )
            normalized.setdefault("degraded_cost_model", _cost_estimate_is_degraded(normalized))
            live_trust = _cost_estimate_live_trust(normalized)
            normalized["cost_trusted_for_live_canary"] = live_trust[
                "cost_trusted_for_live_canary"
            ]
            normalized["cost_trusted_for_live_scale"] = live_trust["cost_trusted_for_live_scale"]
            normalized["cost_trust_level"] = live_trust["cost_trust_level"]
            normalized["cost_trust_block_reasons"] = live_trust["cost_trust_block_reasons"]
            normalized["cost_trusted_for_live"] = live_trust["cost_trusted_for_live_canary"]
            for suffix in ("p50", "p75", "p90"):
                normalized.setdefault(f"total_cost_bps_{suffix}", normalized.get("total_cost_bps"))
            return normalized
        return data

    @field_validator("as_of_ts")
    @classmethod
    def as_of_is_utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None


def _cost_estimate_is_degraded(data: dict[str, Any]) -> bool:
    source = str(data.get("cost_source") or data.get("source") or "").lower()
    fallback_level = str(data.get("fallback_level") or "")
    fallback_reason = str(data.get("fallback_reason") or "")
    degraded_reason = str(data.get("degraded_reason") or "")
    return (
        source in {"global_default", "public_spread_proxy"}
        or fallback_level not in {"", "NONE", "actual_okx_fills_and_bills"}
        or fallback_reason not in {"", "NONE"}
        or degraded_reason not in {"", "none", "None"}
    )


def _cost_estimate_one_way_all_in(data: dict[str, Any]) -> float:
    fields = [
        "fee_bps",
        "spread_bps",
        "slippage_bps",
        "delay_cost_bps",
        "uncertainty_buffer_bps",
    ]
    total = 0.0
    for field in fields:
        try:
            total += float(data.get(field) or 0.0)
        except (TypeError, ValueError):
            continue
    if total > 0:
        return total
    try:
        return float(data.get("total_cost_bps") or data.get("cost_bps") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _cost_estimate_quality(data: dict[str, Any]) -> str:
    source = str(data.get("cost_source") or data.get("source") or "").lower()
    sample_count = int(float(data.get("sample_count") or data.get("sample_size") or 0))
    if source == "global_default":
        return "global_default"
    if source in {"public_spread_proxy", "public_proxy"}:
        return "public_proxy_only"
    if sample_count < 30:
        return "small_sample"
    if source in {"mixed_actual_proxy", "actual_okx_fills_fee_missing"}:
        return "mixed_actual_proxy"
    if source in {"actual_fills", "actual_okx_fills_and_bills"}:
        return "actual"
    return "unknown"


def _cost_estimate_trusted_for_paper(data: dict[str, Any]) -> bool:
    source = str(data.get("cost_source") or data.get("source") or "").lower()
    return (
        source not in {"", "global_default"}
        and data.get("degraded_reason") != "cost_bucket_stale"
    )


def _cost_estimate_trusted_for_live(data: dict[str, Any]) -> bool:
    return bool(_cost_estimate_live_trust(data)["cost_trusted_for_live_canary"])


def _cost_estimate_live_trust(data: dict[str, Any]) -> dict[str, Any]:
    source = str(data.get("cost_source") or data.get("source") or "").lower()
    sample_count = _cost_estimate_sample_count(data)
    fallback_level = str(data.get("fallback_level") or "")
    fallback_reason = str(data.get("fallback_reason") or "")
    fallback_text = f"{fallback_level};{fallback_reason}".upper()
    degraded_reason = str(data.get("degraded_reason") or "none").lower()
    degraded_cost_model = bool(data.get("degraded_cost_model")) or _cost_estimate_is_degraded(data)
    slippage_source = str(data.get("slippage_source") or "").lower()
    stale = degraded_reason == "cost_bucket_stale" or fallback_reason == "cost_bucket_stale"
    reasons: list[str] = []
    if stale:
        reasons.append("stale_cost_bucket")
    if sample_count < 30:
        reasons.append("sample_count_lt_30")
    if source in {"public_spread_proxy", "public_proxy"}:
        reasons.append("source_public_proxy_only")
    if source == "global_default":
        reasons.append("source_global_default")
    if degraded_cost_model:
        reasons.append("degraded_cost_model")
    if _cost_estimate_has_unsafe_fallback(fallback_level, fallback_reason):
        reasons.append("fallback_not_live_safe")
    if "FEE_MISSING" in fallback_text or source == "actual_okx_fills_fee_missing":
        reasons.append("fee_missing")
    if "NOTIONAL_BUCKET" in fallback_text:
        reasons.append("notional_bucket_fallback")
    if slippage_source not in {
        "v5_order_lifecycle_arrival_mid",
        "actual_order_lifecycle_arrival_mid",
        "actual_arrival_mid",
        "actual_slippage",
        "okx_order_lifecycle_arrival_mid",
    }:
        reasons.append("slippage_not_actual")

    canary = (
        source in {"actual_fills", "actual_okx_fills_and_bills", "mixed_actual_proxy"}
        and sample_count >= 30
        and not stale
        and degraded_reason not in {"global_default_cost", "cost_bucket_stale"}
    )
    scale = (
        source in {"actual_fills", "actual_okx_fills_and_bills"}
        and sample_count >= 100
        and not stale
        and _cost_estimate_safe_for_scale(fallback_level, fallback_reason)
        and not degraded_cost_model
        and "NOTIONAL_BUCKET" not in fallback_text
        and "FEE_MISSING" not in fallback_text
        and slippage_source
        in {
            "v5_order_lifecycle_arrival_mid",
            "actual_order_lifecycle_arrival_mid",
            "actual_arrival_mid",
            "actual_slippage",
            "okx_order_lifecycle_arrival_mid",
        }
    )
    if scale:
        level = "SCALE_READY"
    elif canary:
        level = "CANARY"
    elif source and source not in {"global_default"} and not stale:
        level = "PAPER_ONLY"
    else:
        level = "BLOCK"
    return {
        "cost_trusted_for_live_canary": canary,
        "cost_trusted_for_live_scale": scale,
        "cost_trust_level": level,
        "cost_trust_block_reasons": sorted(set(reasons)),
    }


def _cost_estimate_sample_count(data: dict[str, Any]) -> int:
    try:
        return int(float(data.get("sample_count") or data.get("sample_size") or 0))
    except (TypeError, ValueError):
        return 0


def _cost_estimate_safe_for_scale(fallback_level: str, fallback_reason: str) -> bool:
    return _cost_estimate_normalized_token(fallback_level) in {
        "",
        "NONE",
        "ACTUAL_OKX_FILLS_AND_BILLS",
    } and _cost_estimate_normalized_token(fallback_reason) in {"", "NONE"}


def _cost_estimate_has_unsafe_fallback(fallback_level: str, fallback_reason: str) -> bool:
    return not (
        _cost_estimate_normalized_token(fallback_level)
        in {"", "NONE", "ACTUAL_OKX_FILLS_AND_BILLS"}
        and _cost_estimate_normalized_token(fallback_reason) in {"", "NONE"}
    )


def _cost_estimate_normalized_token(value: str) -> str:
    return str(value or "").strip().upper()


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

    @field_validator("inst_id")
    @classmethod
    def normalize_inst_id(cls, value: str) -> str:
        return normalize_symbol(value)

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


class OrderEvent(ContractModel):
    venue: str = "okx"
    inst_type: str = Field(min_length=1)
    inst_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    side: str = Field(min_length=1)
    state: str = Field(min_length=1)
    ts: datetime
    source: str = Field(default="okx_readonly_private", min_length=1)
    order_type: str | None = None
    avg_price: float | None = Field(default=None, ge=0)
    accumulated_fill_size: float | None = Field(default=None, ge=0)
    fee: float | None = None
    reference_price: float | None = Field(default=None, ge=0)

    @field_validator("inst_id")
    @classmethod
    def normalize_inst_id(cls, value: str) -> str:
        return normalize_symbol(value)

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
    evidence_status: str = Field(default="ok", min_length=1)
    role: str = Field(default="strategy_alpha", min_length=1)

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


class AlphaResearchSpec(ContractModel):
    alpha_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    feature_set: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    feature_names: list[str] = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    label_horizon_bars: int = Field(gt=0)
    decision_delay_bars: int = Field(default=1, ge=1)
    universe_id: str = Field(min_length=1)
    strategy: str = Field(default="v5", min_length=1)
    min_coverage: float = Field(default=0.95, ge=0, le=1)
    min_samples: int = Field(default=100, ge=1)
    cost_quantile: str = Field(default="p75", pattern="^p(50|75|90)$")
    long_only: bool = True
    top_quantile: float = Field(default=0.2, gt=0, lt=1)
    bottom_quantile: float | None = Field(default=None, gt=0, lt=1)
    start: datetime | None = None
    end: datetime | None = None

    @field_validator("feature_names")
    @classmethod
    def normalize_feature_names(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("feature_names must contain at least one feature")
        return normalized

    @field_validator("start", "end")
    @classmethod
    def optional_timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None

    @model_validator(mode="after")
    def validate_window(self) -> "AlphaResearchSpec":
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("end must be after start")
        return self


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
    schema_version: str = V5_RISK_PERMISSION_RESPONSE_SCHEMA_VERSION
    strategy: str = Field(min_length=1)
    version: str = Field(min_length=1)
    permission: RiskAction
    allowed_modes: list[str] = Field(default_factory=list)
    max_gross_exposure: float = Field(ge=0)
    max_single_weight: float = Field(ge=0)
    max_gross_exposure_usdt: float | None = Field(default=None, ge=0)
    max_single_order_usdt: float | None = Field(default=None, ge=0)
    cost_model_version: str = Field(min_length=1)
    gate_version: str = Field(min_length=1)
    reasons: list[str] = Field(default_factory=list)
    reason: str | None = None
    created_at: datetime
    as_of_ts: datetime | None = None
    source_bundle_ts: datetime | None = None
    expires_at: datetime | None = None
    telemetry_latest_ts: datetime | None = None
    permission_freshness_sec: int | None = Field(default=None, ge=0)
    contract_version: str = RISK_PERMISSION_CONTRACT_VERSION
    permission_status: RiskPermissionStatus | None = None
    enforceable: bool | None = None
    risk_reason_codes: list[str] = Field(default_factory=list)
    system_safety_status: str | None = None
    core_alpha_gate_status: str | None = None
    core_alpha_dead: bool = False
    baseline_status: str | None = None
    baseline_alpha_id: str | None = None
    baseline_role: str | None = None
    baseline_not_live_eligible: bool = False
    baseline_not_global_strategy_gate: bool = False
    strategy_opportunities_available: bool = False
    allowed_advisory_modes: list[str] = Field(default_factory=list)
    allowed_live_modes: list[str] = Field(default_factory=list)
    live_block_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def sync_permission_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            normalized.setdefault("max_gross_exposure_usdt", normalized.get("max_gross_exposure"))
            normalized.setdefault("max_single_order_usdt", normalized.get("max_single_weight"))
            reasons = normalized.get("risk_reason_codes") or normalized.get("reasons") or []
            if isinstance(reasons, str):
                try:
                    import json

                    parsed = json.loads(reasons)
                    reasons = parsed if isinstance(parsed, list) else [str(parsed)]
                except json.JSONDecodeError:
                    reasons = [reasons] if reasons else []
            normalized.setdefault("reason", reasons[0] if reasons else None)
            for list_field in [
                "allowed_advisory_modes",
                "allowed_live_modes",
                "live_block_reasons",
            ]:
                value = normalized.get(list_field)
                if isinstance(value, str):
                    try:
                        import json

                        parsed = json.loads(value)
                        normalized[list_field] = (
                            parsed if isinstance(parsed, list) else [str(parsed)]
                        )
                    except json.JSONDecodeError:
                        normalized[list_field] = [value] if value else []
            return normalized
        return data

    @field_validator(
        "created_at",
        "as_of_ts",
        "source_bundle_ts",
        "expires_at",
        "telemetry_latest_ts",
    )
    @classmethod
    def risk_timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return require_utc(value)
