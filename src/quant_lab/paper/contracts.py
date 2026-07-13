from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PAPER_STRATEGY_CONTRACT_VERSION = "quant_lab.paper_strategy.v1"

SUPPORTED_RULE_OPERATORS = frozenset(
    {
        "all",
        "any",
        "not",
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
    }
)

SUPPORTED_MARKET_FIELDS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "bid",
        "ask",
        "mid",
        "spread_bps",
        "return_1",
        "return_4",
        "return_8",
        "return_24",
        "momentum_4",
        "momentum_8",
        "momentum_24",
        "volatility_8",
        "volatility_24",
        "volume_zscore_24",
        "cross_sectional_rank",
        "cross_sectional_quantile",
        "market_regime",
        "holding_bars",
        "gross_pnl_bps",
        "net_pnl_bps",
        "peak_pnl_bps",
    }
)


class LifecycleState(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    BACKTEST_CANDIDATE = "BACKTEST_CANDIDATE"
    PAPER_PROPOSAL_READY = "PAPER_PROPOSAL_READY"
    PAPER_ACK_PENDING = "PAPER_ACK_PENDING"
    PAPER_TRACKER_ACTIVE = "PAPER_TRACKER_ACTIVE"
    PAPER_EVIDENCE_INSUFFICIENT = "PAPER_EVIDENCE_INSUFFICIENT"
    PAPER_PROMOTION_READY = "PAPER_PROMOTION_READY"
    CANARY_READY = "CANARY_READY"
    LIVE_SCALE_READY = "LIVE_SCALE_READY"
    REJECTED = "REJECTED"
    KILLED = "KILLED"


_LEGAL_TRANSITIONS: dict[LifecycleState, frozenset[LifecycleState]] = {
    LifecycleState.RESEARCH_ONLY: frozenset(
        {LifecycleState.BACKTEST_CANDIDATE, LifecycleState.REJECTED, LifecycleState.KILLED}
    ),
    LifecycleState.BACKTEST_CANDIDATE: frozenset(
        {LifecycleState.PAPER_PROPOSAL_READY, LifecycleState.REJECTED, LifecycleState.KILLED}
    ),
    LifecycleState.PAPER_PROPOSAL_READY: frozenset(
        {LifecycleState.PAPER_ACK_PENDING, LifecycleState.REJECTED, LifecycleState.KILLED}
    ),
    LifecycleState.PAPER_ACK_PENDING: frozenset(
        {LifecycleState.PAPER_TRACKER_ACTIVE, LifecycleState.REJECTED, LifecycleState.KILLED}
    ),
    LifecycleState.PAPER_TRACKER_ACTIVE: frozenset(
        {
            LifecycleState.PAPER_EVIDENCE_INSUFFICIENT,
            LifecycleState.PAPER_PROMOTION_READY,
            LifecycleState.REJECTED,
            LifecycleState.KILLED,
        }
    ),
    LifecycleState.PAPER_EVIDENCE_INSUFFICIENT: frozenset(
        {
            LifecycleState.PAPER_TRACKER_ACTIVE,
            LifecycleState.PAPER_PROMOTION_READY,
            LifecycleState.REJECTED,
            LifecycleState.KILLED,
        }
    ),
    LifecycleState.PAPER_PROMOTION_READY: frozenset(
        {LifecycleState.CANARY_READY, LifecycleState.REJECTED, LifecycleState.KILLED}
    ),
    LifecycleState.CANARY_READY: frozenset(
        {LifecycleState.LIVE_SCALE_READY, LifecycleState.REJECTED, LifecycleState.KILLED}
    ),
    LifecycleState.LIVE_SCALE_READY: frozenset(
        {LifecycleState.CANARY_READY, LifecycleState.REJECTED, LifecycleState.KILLED}
    ),
    LifecycleState.REJECTED: frozenset(),
    LifecycleState.KILLED: frozenset(),
}


def assert_lifecycle_transition(
    current: LifecycleState | str,
    target: LifecycleState | str,
) -> LifecycleState:
    source = LifecycleState(str(current))
    destination = LifecycleState(str(target))
    if source == destination:
        return destination
    if destination not in _LEGAL_TRANSITIONS[source]:
        raise ValueError(f"illegal paper lifecycle transition: {source.value}->{destination.value}")
    return destination


def legacy_lifecycle_state(
    value: Any,
    *,
    accepted: bool | None = None,
    tracker_effective: bool | None = None,
    promotion_ready: bool | None = None,
) -> LifecycleState:
    """Map legacy labels without promoting research evidence into live readiness."""

    text = str(value or "").strip().upper().replace("-", "_")
    if promotion_ready is True:
        return LifecycleState.PAPER_PROMOTION_READY
    if tracker_effective is True:
        return LifecycleState.PAPER_TRACKER_ACTIVE
    if accepted is True:
        return LifecycleState.PAPER_ACK_PENDING
    mapping = {
        "": LifecycleState.RESEARCH_ONLY,
        "RESEARCH": LifecycleState.RESEARCH_ONLY,
        "RESEARCH_ONLY": LifecycleState.RESEARCH_ONLY,
        "KEEP_SHADOW": LifecycleState.RESEARCH_ONLY,
        "SHADOW": LifecycleState.RESEARCH_ONLY,
        "BACKTEST_CANDIDATE": LifecycleState.BACKTEST_CANDIDATE,
        # Historical PAPER_READY meant the proposal threshold, not completed paper evidence.
        "PAPER_READY": LifecycleState.PAPER_PROPOSAL_READY,
        "PAPER_PROPOSAL_READY": LifecycleState.PAPER_PROPOSAL_READY,
        "PROPOSED_AWAITING_ACK": LifecycleState.PAPER_ACK_PENDING,
        "CURRENT_PENDING_ACK": LifecycleState.PAPER_ACK_PENDING,
        "CURRENT_ACKED_TRACKER_MISSING": LifecycleState.PAPER_ACK_PENDING,
        "CURRENT_TRACKER_ACK_MISSING": LifecycleState.PAPER_ACK_PENDING,
        "AWAITING_ACK": LifecycleState.PAPER_ACK_PENDING,
        "ACK_PENDING": LifecycleState.PAPER_ACK_PENDING,
        "PAPER_ACK_PENDING": LifecycleState.PAPER_ACK_PENDING,
        "ACKED": LifecycleState.PAPER_ACK_PENDING,
        "ACKED_TRACKER_MISSING": LifecycleState.PAPER_ACK_PENDING,
        "PAPER_TRACKING": LifecycleState.PAPER_TRACKER_ACTIVE,
        "PAPER_TRACKER_ACTIVE": LifecycleState.PAPER_TRACKER_ACTIVE,
        "PAPER_REVIEW": LifecycleState.PAPER_EVIDENCE_INSUFFICIENT,
        "PAPER_EVIDENCE_INSUFFICIENT": LifecycleState.PAPER_EVIDENCE_INSUFFICIENT,
        "PAPER_PROMOTION_READY": LifecycleState.PAPER_PROMOTION_READY,
        "CANARY_READY": LifecycleState.CANARY_READY,
        "LIVE_SMALL_READY": LifecycleState.CANARY_READY,
        "LIVE_SCALE_READY": LifecycleState.LIVE_SCALE_READY,
        "REJECTED": LifecycleState.REJECTED,
        "REJECTED_BY_V5": LifecycleState.REJECTED,
        "REJECTED_TRACKER_MISSING": LifecycleState.REJECTED,
        "KILL": LifecycleState.KILLED,
        "KILLED": LifecycleState.KILLED,
    }
    return mapping.get(text, LifecycleState.RESEARCH_ONLY)


RuleOperator = Literal[
    "all",
    "any",
    "not",
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


class PaperRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operator: RuleOperator
    field: str | None = None
    reference_field: str | None = None
    value: float | int | str | bool | None = None
    values: list[str | float | int] = Field(default_factory=list, max_length=64)
    window: int | None = Field(default=None, ge=1, le=512)
    periods: int | None = Field(default=None, ge=1, le=512)
    children: list[PaperRule] = Field(default_factory=list, max_length=32)

    @field_validator("field", "reference_field")
    @classmethod
    def validate_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_MARKET_FIELDS:
            raise ValueError(f"unsupported paper rule market field: {value}")
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> PaperRule:
        operator = self.operator
        if operator in {"all", "any"} and not self.children:
            raise ValueError(f"{operator} requires at least one child rule")
        if operator == "not" and len(self.children) != 1:
            raise ValueError("not requires exactly one child rule")
        if operator == "consecutive":
            if len(self.children) != 1 or self.periods is None:
                raise ValueError("consecutive requires one child rule and periods")
        leaf_operators = SUPPORTED_RULE_OPERATORS - {"all", "any", "not", "consecutive"}
        if operator in leaf_operators and self.children:
            raise ValueError(f"{operator} cannot contain child rules")
        field_operators = {
            "gt",
            "gte",
            "lt",
            "lte",
            "crosses_above",
            "crosses_below",
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
        }
        if operator in field_operators and self.field is None:
            raise ValueError(f"{operator} requires field")
        if operator in field_operators and self.reference_field is None and self.value is None:
            raise ValueError(f"{operator} requires value or reference_field")
        if operator == "regime_in" and (self.field != "market_regime" or not self.values):
            raise ValueError("regime_in requires field=market_regime and values")
        if (
            operator
            in {
                "take_profit",
                "stop_loss",
                "trailing_exit",
                "max_holding_bars",
            }
            and self.value is None
        ):
            raise ValueError(f"{operator} requires value")
        return self


class PaperStrategyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    contract_version: Literal[PAPER_STRATEGY_CONTRACT_VERSION] = PAPER_STRATEGY_CONTRACT_VERSION
    proposal_id: str = ""
    proposal_hash: str = ""
    strategy_id: str = Field(min_length=1, max_length=160)
    strategy_version: str = Field(min_length=1, max_length=80)
    strategy_family: str = Field(min_length=1, max_length=120)
    symbol: str
    timeframe: str
    direction: Literal["long", "short"] = "long"
    entry_rule: PaperRule
    exit_rule: PaperRule
    max_holding_bars: int = Field(ge=1, le=10_000)
    min_holding_bars: int = Field(default=0, ge=0, le=10_000)
    cooldown_bars: int = Field(default=0, ge=0, le=10_000)
    signal_confirmation_bars: int = Field(default=1, ge=1, le=1_000)
    cost_quantile: Literal["p50", "p75", "p90", "p95"] = "p75"
    minimum_expected_edge_bps: float = Field(default=0.0, ge=-10_000, le=100_000)
    paper_notional_usdt: float = Field(default=20.0, gt=0, le=1_000_000)
    paper_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    max_live_notional_usdt: Literal[0.0] = 0.0
    created_at: datetime
    expires_at: datetime
    source_pack_sha256: str = ""
    source_dataset_versions: dict[str, str] = Field(default_factory=dict)
    required_market_fields: list[str] = Field(default_factory=list, max_length=64)
    required_cost_trust_level: Literal["BLOCK", "PAPER_ONLY", "CANARY", "SCALE_READY"] = (
        "PAPER_ONLY"
    )
    lifecycle_state: Literal[LifecycleState.PAPER_PROPOSAL_READY] = (
        LifecycleState.PAPER_PROPOSAL_READY
    )
    lifecycle_reason: str = "research_threshold_met_contract_pending_ack"
    blocked_reasons: list[str] = Field(default_factory=lambda: ["v5_ack_required"])
    next_required_actions: list[str] = Field(default_factory=lambda: ["sync_to_v5", "receive_ack"])

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        text = value.strip().upper().replace("_", "/").replace("-", "/")
        parts = [part for part in text.split("/") if part]
        if len(parts) != 2 or not all(re.fullmatch(r"[A-Z0-9]+", part) for part in parts):
            raise ValueError("symbol must use BASE/QUOTE format")
        return f"{parts[0]}/{parts[1]}"

    @field_validator("timeframe")
    @classmethod
    def normalize_timeframe(cls, value: str) -> str:
        text = value.strip().lower()
        match = re.fullmatch(r"([1-9][0-9]*)(m|h|d)", text)
        if not match:
            raise ValueError("timeframe must use positive <n>m, <n>h, or <n>d format")
        return text

    @field_validator("required_market_fields")
    @classmethod
    def validate_required_fields(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw in values:
            value = raw.strip().lower()
            if value not in SUPPORTED_MARKET_FIELDS:
                raise ValueError(f"unsupported required market field: {raw}")
            if value not in normalized:
                normalized.append(value)
        return normalized

    @field_validator("created_at", "expires_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("proposal timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_contract(self) -> PaperStrategyProposal:
        if self.min_holding_bars > self.max_holding_bars:
            raise ValueError("min_holding_bars cannot exceed max_holding_bars")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        expected_hash = paper_proposal_hash(self)
        if self.proposal_hash and self.proposal_hash != expected_hash:
            raise ValueError("proposal_hash does not match canonical proposal content")
        object.__setattr__(self, "proposal_hash", expected_hash)
        if not self.proposal_id:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.strategy_id).strip("_")
            object.__setattr__(
                self,
                "proposal_id",
                f"{safe_id}:{self.strategy_version}:{expected_hash[:12]}",
            )
        return self


class PaperStrategyAck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    proposal_id: str = Field(min_length=1)
    proposal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted: bool
    reject_reason: str = ""
    tracker_id: str = ""
    contract_version: str = PAPER_STRATEGY_CONTRACT_VERSION
    strategy_version: str = Field(min_length=1)
    rules_locked: bool = False
    paper_only: bool = True
    live_order_effect: str = "none"
    accepted_at: datetime
    expires_at: datetime
    source_v5_commit: str = ""
    source_v5_bundle_sha256: str = ""

    @model_validator(mode="after")
    def validate_safety(self) -> PaperStrategyAck:
        if self.accepted:
            if not self.tracker_id:
                raise ValueError("accepted ACK requires tracker_id")
            if not self.rules_locked:
                raise ValueError("accepted ACK requires rules_locked=true")
            if not self.paper_only or self.live_order_effect != "none":
                raise ValueError("accepted ACK must remain paper-only with no live effect")
            if self.reject_reason:
                raise ValueError("accepted ACK cannot contain reject_reason")
        elif not self.reject_reason:
            raise ValueError("rejected ACK requires reject_reason")
        return self


def paper_proposal_hash(value: PaperStrategyProposal | dict[str, Any]) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        normalized = dict(value)
        normalized["proposal_id"] = ""
        normalized["proposal_hash"] = ""
        payload = PaperStrategyProposal.model_validate(normalized).model_dump(mode="json")
    for key in (
        "proposal_id",
        "proposal_hash",
        "created_at",
        "expires_at",
        "source_pack_sha256",
        "lifecycle_reason",
        "blocked_reasons",
        "next_required_actions",
    ):
        payload.pop(key, None)
    material = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
