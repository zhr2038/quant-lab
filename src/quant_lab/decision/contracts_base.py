from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quant_lab.contracts.models import require_utc


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CostObservation(Contract):
    """Original signed cost shape; defaults must remain stable for archived identities."""

    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    roundtrip_bps: float | None = Field(default=None, ge=0, le=10_000)
    notional_usdt: float = Field(default=20, gt=0, le=100_000)
    source: str = Field(min_length=1, max_length=100)
    quality: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=200)
    as_of: datetime | None = None
    trusted_for_paper: bool = False
    actual_sample_count: int = Field(default=0, ge=0)
    missing_reasons: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("as_of")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None
