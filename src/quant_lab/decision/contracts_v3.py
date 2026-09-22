"""Versioned observation diagnostics; archived v1/v2 identities remain unchanged."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from quant_lab.contracts.models import require_utc
from quant_lab.decision.contracts import SYMBOLS
from quant_lab.decision.contracts_base import Contract
from quant_lab.decision.contracts_v2 import AnalysisResultV2, ScopedForwardSummary


class RegistrationGap(Contract):
    first_entry_at: datetime
    last_entry_at: datetime
    hours: int = Field(gt=0)

    @field_validator("first_entry_at", "last_entry_at")
    @classmethod
    def utc(cls, value):
        return require_utc(value)

    @model_validator(mode="after")
    def valid_interval(self):
        if (self.last_entry_at - self.first_entry_at).total_seconds() != (self.hours - 1) * 3600:
            raise ValueError("gap interval does not match hour count")
        return self


class SymbolRegistration(Contract):
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    expected_hours: int = Field(ge=0)
    registered_hours: int = Field(ge=0)
    missing_hours: int = Field(ge=0)
    partial_horizon_hours: int = Field(ge=0)
    pending_deadline_hours: int = Field(ge=0)
    coverage_fraction: float | None = Field(default=None, ge=0, le=1)
    late_publication_hours: int = Field(ge=0)
    maximum_publication_delay_seconds: float | None = Field(default=None, ge=0)
    latest_registered_entry_at: datetime | None
    gaps: list[RegistrationGap] = Field(max_length=50)
    omitted_gap_ranges: int = Field(ge=0)

    @field_validator("latest_registered_entry_at")
    @classmethod
    def utc(cls, value):
        return require_utc(value) if value is not None else None

    @model_validator(mode="after")
    def counts_reconcile(self):
        if self.registered_hours + self.missing_hours != self.expected_hours:
            raise ValueError("registration hours do not reconcile")
        if self.partial_horizon_hours > self.missing_hours:
            raise ValueError("partial horizon count exceeds missing hours")
        expected = self.registered_hours / self.expected_hours if self.expected_hours else None
        if self.coverage_fraction != expected:
            raise ValueError("registration coverage does not match counts")
        return self


class RegistrationCoverage(Contract):
    denominator: Literal["complete_horizon_pairs_due_since_first_scoped_publication"] = (
        "complete_horizon_pairs_due_since_first_scoped_publication"
    )
    first_entry_at: datetime | None
    due_through_entry_at: datetime
    publication_delay_warning_seconds: Literal[900] = 900
    backfilled_forward_observations: Literal[0] = 0
    status: Literal["NOT_STARTED", "COMPLETE", "GAPS"]
    by_symbol: list[SymbolRegistration] = Field(min_length=4, max_length=4)

    @field_validator("first_entry_at", "due_through_entry_at")
    @classmethod
    def utc(cls, value):
        return require_utc(value) if value is not None else None

    @model_validator(mode="after")
    def consistent_coverage(self):
        if {row.symbol for row in self.by_symbol} != set(SYMBOLS):
            raise ValueError("coverage must contain each symbol exactly once")
        expected_status = (
            "NOT_STARTED"
            if self.first_entry_at is None
            else ("GAPS" if any(row.missing_hours for row in self.by_symbol) else "COMPLETE")
        )
        if self.status != expected_status:
            raise ValueError("coverage status does not match evidence")
        if self.due_through_entry_at != self.due_through_entry_at.replace(
            minute=0, second=0, microsecond=0
        ):
            raise ValueError("coverage deadline must be an hourly boundary")
        expected_hours = (
            max(
                0, int((self.due_through_entry_at - self.first_entry_at).total_seconds() / 3600) + 1
            )
            if self.first_entry_at
            else 0
        )
        if any(row.expected_hours != expected_hours for row in self.by_symbol):
            raise ValueError("coverage denominator does not match interval")
        return self


class MatureNonOverlappingGroup(Contract):
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    horizon_hours: Literal[4, 24]
    action: Literal["NO_VIEW", "DEFER", "KEEP_BASELINE", "REVIEW_ENTRY"]
    selected_observations: int = Field(ge=0)
    matured_observations: int = Field(ge=0)
    waiting_observations: int = Field(ge=0)
    missing_label_observations: int = Field(ge=0)
    net_observations: int = Field(ge=0)
    gross_mean_bps: float | None
    net_mean_bps: float | None

    @model_validator(mode="after")
    def counts_reconcile(self):
        if (
            self.selected_observations
            != (
                self.matured_observations
                + self.waiting_observations
                + self.missing_label_observations
            )
            or self.net_observations > self.matured_observations
        ):
            raise ValueError("non-overlapping observations do not reconcile")
        return self


class ObservedForwardSummary(ScopedForwardSummary):
    registration: RegistrationCoverage
    mature_non_overlapping_by_group: list[MatureNonOverlappingGroup] = Field(max_length=32)
    non_overlapping_selection: Literal["earliest_registered_per_symbol_horizon_before_outcome"] = (
        "earliest_registered_per_symbol_horizon_before_outcome"
    )
    cross_symbol_independence: Literal[False] = False
    legacy_non_overlapping_includes_waiting: Literal[True] = True


class AnalysisResultV3(AnalysisResultV2):
    schema_version: Literal["qlab.decision.result.v3"] = "qlab.decision.result.v3"
    forward: ObservedForwardSummary
