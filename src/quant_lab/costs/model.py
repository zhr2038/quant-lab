import logging
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_lab.contracts.models import CostEstimate, FillEvent

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_COST_BPS = 25.0


class CostBucket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket_id: str | None = None
    symbol: str | None = None
    regime: str | None = None
    min_notional_usdt: float = Field(default=0, ge=0)
    max_notional_usdt: float | None = Field(default=None, gt=0)
    cost_bps: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "CostBucket":
        if self.max_notional_usdt is not None and self.max_notional_usdt < self.min_notional_usdt:
            raise ValueError("max_notional_usdt must be greater than or equal to min_notional_usdt")
        return self

    def includes_notional(self, notional_usdt: float) -> bool:
        if notional_usdt < self.min_notional_usdt:
            return False
        return self.max_notional_usdt is None or notional_usdt <= self.max_notional_usdt


class CostBucketDaily(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    day: str = Field(min_length=10)
    symbol: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    notional_bucket: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    fee_bps_p50: float = Field(ge=0)
    fee_bps_p75: float = Field(ge=0)
    fee_bps_p90: float = Field(ge=0)
    slippage_bps_p50: float = Field(ge=0)
    slippage_bps_p75: float = Field(ge=0)
    slippage_bps_p90: float = Field(ge=0)
    spread_bps_p50: float = Field(ge=0)
    spread_bps_p75: float = Field(ge=0)
    spread_bps_p90: float = Field(ge=0)
    total_cost_bps_p50: float = Field(ge=0)
    total_cost_bps_p75: float = Field(ge=0)
    total_cost_bps_p90: float = Field(ge=0)
    fallback_level: str = Field(min_length=1)
    source: str = Field(min_length=1)


def _normalize_buckets(buckets: Iterable[CostBucket | Mapping[str, Any]]) -> list[CostBucket]:
    return [
        bucket if isinstance(bucket, CostBucket) else CostBucket.model_validate(bucket)
        for bucket in buckets
    ]


def _choose_bucket(
    symbol: str, regime: str, notional_usdt: float, buckets: list[CostBucket]
) -> tuple[CostBucket | None, str]:
    notional_matches = [bucket for bucket in buckets if bucket.includes_notional(notional_usdt)]

    tiers = [
        (
            "NONE",
            [
                bucket
                for bucket in notional_matches
                if bucket.symbol == symbol and bucket.regime == regime
            ],
        ),
        (
            "REGIME_FALLBACK",
            [
                bucket
                for bucket in notional_matches
                if bucket.symbol == symbol and bucket.regime is None
            ],
        ),
        (
            "SYMBOL_FALLBACK",
            [
                bucket
                for bucket in notional_matches
                if bucket.symbol is None and bucket.regime == regime
            ],
        ),
        (
            "GLOBAL_BUCKET_FALLBACK",
            [
                bucket
                for bucket in notional_matches
                if bucket.symbol is None and bucket.regime is None
            ],
        ),
    ]
    for fallback_level, candidates in tiers:
        if candidates:
            return candidates[0], fallback_level
    return None, "DEFAULT_FALLBACK"


def estimate_cost_bps(
    symbol: str,
    regime: str,
    notional_usdt: float,
    buckets: Iterable[CostBucket | Mapping[str, Any]],
) -> CostEstimate:
    if notional_usdt <= 0:
        raise ValueError("notional_usdt must be positive")

    normalized = _normalize_buckets(buckets)
    bucket, fallback_level = _choose_bucket(symbol, regime, notional_usdt, normalized)

    if bucket is None:
        logger.warning(
            "No cost bucket matched; using explicit default fallback",
            extra={
                "symbol": symbol,
                "regime": regime,
                "notional_usdt": notional_usdt,
                "fallback_level": fallback_level,
            },
        )
        return CostEstimate(
            symbol=symbol,
            regime=regime,
            notional_usdt=notional_usdt,
            cost_bps=DEFAULT_FALLBACK_COST_BPS,
            fallback_level=fallback_level,
            bucket_id=None,
        )

    if fallback_level != "NONE":
        logger.warning(
            "Cost bucket fallback used",
            extra={
                "symbol": symbol,
                "regime": regime,
                "notional_usdt": notional_usdt,
                "bucket_id": bucket.bucket_id,
                "fallback_level": fallback_level,
            },
        )

    return CostEstimate(
        symbol=symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        cost_bps=bucket.cost_bps,
        fallback_level=fallback_level,
        bucket_id=bucket.bucket_id,
    )


def build_cost_bucket_daily_inputs(
    fill_events: Iterable[FillEvent | Mapping[str, Any]],
    regime: str = "realized",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_event in fill_events:
        event = (
            raw_event
            if isinstance(raw_event, FillEvent)
            else FillEvent.model_validate(raw_event)
        )
        notional = abs(event.fill_price * event.fill_size)
        if notional <= 0:
            continue
        key = (event.inst_id, event.ts.date().isoformat())
        bucket = grouped.setdefault(
            key,
            {
                "symbol": event.inst_id,
                "cost_day": event.ts.date().isoformat(),
                "regime": regime,
                "notional_usdt": 0.0,
                "fee_abs": 0.0,
                "source": event.source,
            },
        )
        bucket["notional_usdt"] += notional
        bucket["fee_abs"] += abs(event.fee)

    rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        notional = bucket["notional_usdt"]
        fee_abs = bucket["fee_abs"]
        rows.append(
            {
                **bucket,
                "cost_bps": (fee_abs / notional) * 10_000 if notional else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (row["symbol"], row["cost_day"], row["regime"]))


def cost_bucket_daily_to_cost_buckets(
    rows: Iterable[CostBucketDaily | Mapping[str, Any]],
    percentile: str = "p50",
) -> list[CostBucket]:
    cost_column = f"total_cost_bps_{percentile}"
    buckets: list[CostBucket] = []
    for raw_row in rows:
        row = (
            raw_row
            if isinstance(raw_row, CostBucketDaily)
            else CostBucketDaily.model_validate(raw_row)
        )
        if not hasattr(row, cost_column):
            raise ValueError(f"Unsupported cost percentile: {percentile}")
        min_notional, max_notional = _notional_bucket_bounds(row.notional_bucket)
        buckets.append(
            CostBucket(
                bucket_id=(
                    f"{row.day}:{row.symbol}:{row.regime}:"
                    f"{row.event_type}:{row.notional_bucket}"
                ),
                symbol=row.symbol if row.symbol != "GLOBAL" else None,
                regime=row.regime if row.regime != "global_default" else None,
                min_notional_usdt=min_notional,
                max_notional_usdt=max_notional,
                cost_bps=float(getattr(row, cost_column)),
            )
        )
    return buckets


def _notional_bucket_bounds(notional_bucket: str) -> tuple[float, float | None]:
    match notional_bucket:
        case "0-1k":
            return 0.0, 1_000.0
        case "1k-10k":
            return 1_000.0, 10_000.0
        case "10k-100k":
            return 10_000.0, 100_000.0
        case "100k+":
            return 100_000.0, None
        case "all":
            return 0.0, None
        case _:
            return 0.0, None
