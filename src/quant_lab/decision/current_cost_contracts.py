"""Current estimates are distinct from the immutable legacy cost observations."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import isclose
from typing import Literal

from pydantic import Field, field_validator, model_validator

from quant_lab.contracts.models import require_utc
from quant_lab.decision.contracts_base import Contract, CostObservation


class FeeRate(Contract):
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    source: Literal["okx_account_trade_fee"] = "okx_account_trade_fee"
    group_id: str = Field(min_length=1, max_length=40)
    # Expense is positive, rebate negative. Zero is a valid reported rate.
    maker_bps: float = Field(ge=-100, le=1000)
    taker_bps: float = Field(ge=-100, le=1000)
    fetched_at: datetime
    exchange_at: datetime

    @field_validator("fetched_at", "exchange_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return require_utc(value)


class SizeCost(Contract):
    notional_usdt: float = Field(gt=0, le=100_000)
    status: Literal["ESTIMATED", "INSUFFICIENT_DEPTH", "BELOW_MINIMUM_SIZE"]
    base_quantity: float = Field(gt=0)
    buy_deviation_bps: float | None = Field(default=None, ge=0)
    sell_deviation_bps: float | None = Field(default=None, ge=0)
    book_roundtrip_bps: float | None = Field(default=None, ge=0)
    fee_roundtrip_bps: float = Field(ge=0)
    uncertainty_bps: float = Field(ge=0)
    roundtrip_bps: float | None = Field(default=None, ge=0, le=10_000)

    @model_validator(mode="after")
    def sum_once(self) -> SizeCost:
        if self.status == "ESTIMATED":
            if any(
                v is None
                for v in (
                    self.buy_deviation_bps,
                    self.sell_deviation_bps,
                    self.book_roundtrip_bps,
                    self.roundtrip_bps,
                )
            ):
                raise ValueError("complete cost requires both book sides")
            if not isclose(
                self.book_roundtrip_bps,
                self.buy_deviation_bps + self.sell_deviation_bps,
                abs_tol=1e-8,
            ) or not isclose(
                self.roundtrip_bps,
                self.book_roundtrip_bps + self.fee_roundtrip_bps + self.uncertainty_bps,
                abs_tol=1e-8,
            ):
                raise ValueError("cost components must reconcile exactly once")
        elif self.roundtrip_bps is not None:
            raise ValueError("incomplete depth cannot have a complete cost")
        return self


class CurrentCostDetail(Contract):
    schema_version: Literal["qlab.current_cost.v1"] = "qlab.current_cost.v1"
    status: Literal["ESTIMATED", "UNAVAILABLE"]
    fee: FeeRate | None
    book_as_of: datetime | None
    valid_until: datetime | None
    execution_assumption: Literal["taker_entry_and_exit_at_current_book"] = (
        "taker_entry_and_exit_at_current_book"
    )
    notional_basis: Literal["base_quantity_at_current_midpoint"] = (
        "base_quantity_at_current_midpoint"
    )
    sizes: list[SizeCost] = Field(max_length=4)
    # Kept verbatim for audit. This is the OLD model's scenario, not today's cost.
    historical_anchor: CostObservation | None
    calibrated: Literal[False] = False

    @field_validator("book_as_of", "valid_until")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None


class CurrentCostObservation(CostObservation):
    # Required field ensures old signed records parse as the unchanged base type.
    current: CurrentCostDetail

    @model_validator(mode="after")
    def consistent_estimate(self) -> CurrentCostObservation:
        if self.trusted_for_paper or self.actual_sample_count:
            raise ValueError("a book estimate is not calibrated fill evidence")
        selected = [s for s in self.current.sizes if s.notional_usdt == self.notional_usdt]
        if len({s.notional_usdt for s in self.current.sizes}) != len(self.current.sizes):
            raise ValueError("duplicate cost sizes")
        if self.current.status == "ESTIMATED":
            if (
                self.current.fee is None
                or self.current.valid_until is None
                or self.current.book_as_of is None
                or len(selected) != 1
                or selected[0].status != "ESTIMATED"
                or self.roundtrip_bps is None
                or self.current.fee.symbol != self.symbol
                or self.as_of != self.current.book_as_of
                or not isclose(self.roundtrip_bps, selected[0].roundtrip_bps)
            ):
                raise ValueError("current cost is not bound to a complete size estimate")
            if not self.as_of < self.current.valid_until <= min(
                self.as_of + timedelta(minutes=15),
                self.current.fee.fetched_at + timedelta(hours=24),
            ):
                raise ValueError("current cost exceeds observation lifetime")
        elif self.roundtrip_bps is not None:
            raise ValueError("unavailable current cost cannot carry a total")
        return self
