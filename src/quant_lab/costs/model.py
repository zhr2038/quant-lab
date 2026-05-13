import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_lab.contracts.models import CostEstimate, FillEvent
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.symbols import normalize_optional_symbol, normalize_symbol

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_COST_BPS = 25.0
SUPPORTED_COST_QUANTILES = {"p50", "p75", "p90"}


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

    @model_validator(mode="before")
    @classmethod
    def normalize_bucket_symbol(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("symbol") is not None:
            normalized = dict(data)
            normalized["symbol"] = normalize_optional_symbol(normalized.get("symbol"))
            return normalized
        return data

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
    cost_model_version: str = Field(default="cost_bucket_daily.v0.1", min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def normalize_daily_symbol(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("symbol") not in {None, "GLOBAL"}:
            normalized = dict(data)
            normalized["symbol"] = normalize_symbol(normalized.get("symbol"))
            return normalized
        return data


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

    requested_symbol = normalize_symbol(symbol)
    normalized = _normalize_buckets(buckets)
    bucket, fallback_level = _choose_bucket(requested_symbol, regime, notional_usdt, normalized)

    if bucket is None:
        logger.warning(
            "No cost bucket matched; using explicit default fallback",
            extra={
                "symbol": requested_symbol,
                "regime": regime,
                "notional_usdt": notional_usdt,
                "fallback_level": fallback_level,
            },
        )
        return CostEstimate(
            symbol=requested_symbol,
            regime=regime,
            notional_usdt=notional_usdt,
            quantile="p75",
            fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            total_cost_bps=DEFAULT_FALLBACK_COST_BPS,
            cost_bps=DEFAULT_FALLBACK_COST_BPS,
            fallback_level=fallback_level,
            source="global_default",
            sample_count=0,
            cost_model_version="legacy_cost_bucket_v0",
            bucket_id=None,
        )

    if fallback_level != "NONE":
        logger.warning(
            "Cost bucket fallback used",
            extra={
                "symbol": requested_symbol,
                "regime": regime,
                "notional_usdt": notional_usdt,
                "bucket_id": bucket.bucket_id,
                "fallback_level": fallback_level,
            },
        )

    return CostEstimate(
        symbol=requested_symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        quantile="p75",
        fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        total_cost_bps=bucket.cost_bps,
        cost_bps=bucket.cost_bps,
        fallback_level=fallback_level,
        source="configured_cost_bucket",
        sample_count=0,
        cost_model_version="legacy_cost_bucket_v0",
        bucket_id=bucket.bucket_id,
    )


def estimate_cost_from_lake(
    lake_root: str | Path,
    symbol: str,
    regime: str,
    notional_usdt: float,
    quantile: str = "p75",
    notional_bucket: str | None = None,
) -> CostEstimate:
    df = read_parquet_dataset(Path(lake_root) / "gold" / "cost_bucket_daily")
    rows = [] if df.is_empty() else df.to_dicts()
    return estimate_cost_from_cost_bucket_daily_rows(
        symbol=symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        quantile=quantile,
        rows=rows,
        notional_bucket=notional_bucket,
    )


def estimate_cost_from_cost_bucket_daily_rows(
    *,
    symbol: str,
    regime: str,
    notional_usdt: float,
    quantile: str,
    rows: Iterable[Mapping[str, Any]],
    notional_bucket: str | None = None,
) -> CostEstimate:
    if notional_usdt <= 0:
        raise ValueError("notional_usdt must be positive")
    if quantile not in SUPPORTED_COST_QUANTILES:
        raise ValueError("quantile must be one of p50, p75, p90")

    requested_symbol = normalize_symbol(symbol)
    normalized_rows = [_normalize_cost_row(row) for row in rows]
    if not normalized_rows:
        return _global_default_estimate(requested_symbol, regime, notional_usdt, quantile)

    tiered = _rank_cost_bucket_rows(
        rows=normalized_rows,
        symbol=requested_symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        notional_bucket=notional_bucket,
    )
    if not tiered:
        return _global_default_estimate(requested_symbol, regime, notional_usdt, quantile)

    row, fallback_level = tiered[0]
    row_fallback_level = str(row.get("fallback_level") or "")
    if row_fallback_level.upper() == "GLOBAL_DEFAULT":
        fallback_level = "GLOBAL_DEFAULT"
    elif row_fallback_level and row_fallback_level not in {
        "NONE",
        "actual_okx_fills_and_bills",
    }:
        fallback_level = (
            row_fallback_level
            if fallback_level == "NONE"
            else f"{fallback_level};{row_fallback_level}"
        )
    fee_bps = _float_value(row, f"fee_bps_{quantile}")
    slippage_bps = _float_value(row, f"slippage_bps_{quantile}")
    spread_bps = _float_value(row, f"spread_bps_{quantile}")
    total_cost_bps = _float_value(row, f"total_cost_bps_{quantile}")
    if total_cost_bps == 0:
        total_cost_bps = fee_bps + slippage_bps + spread_bps

    bucket_id = _cost_bucket_id(row)
    return CostEstimate(
        symbol=requested_symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        quantile=quantile,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        total_cost_bps=total_cost_bps,
        cost_bps=total_cost_bps,
        fallback_level=fallback_level,
        source=str(row.get("source") or "cost_bucket_daily"),
        sample_count=int(row.get("sample_count") or 0),
        cost_model_version=str(
            row.get("cost_model_version") or f"cost_bucket_daily:{row.get('day', 'unknown')}"
        ),
        bucket_id=bucket_id,
        total_cost_bps_p50=_float_value(row, "total_cost_bps_p50"),
        total_cost_bps_p75=_float_value(row, "total_cost_bps_p75"),
        total_cost_bps_p90=_float_value(row, "total_cost_bps_p90"),
        as_of_ts=_row_as_of_ts(row),
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
        symbol = normalize_symbol(event.inst_id)
        key = (symbol, event.ts.date().isoformat())
        bucket = grouped.setdefault(
            key,
            {
                "symbol": symbol,
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


def _rank_cost_bucket_rows(
    *,
    rows: list[dict[str, Any]],
    symbol: str,
    regime: str,
    notional_usdt: float,
    notional_bucket: str | None,
) -> list[tuple[dict[str, Any], str]]:
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for row in rows:
        row_symbol = _row_symbol(row)
        row_regime = str(row.get("regime") or "")
        row_bucket = str(row.get("notional_bucket") or "")
        notional_match = _row_matches_notional(row_bucket, notional_usdt, notional_bucket)

        if row_symbol == symbol and row_regime == regime and notional_match:
            tier, fallback = 0, "NONE"
        elif row_symbol == symbol and row_regime == regime:
            tier, fallback = 1, "NOTIONAL_BUCKET_FALLBACK"
        elif row_symbol == symbol and _is_global_regime(row_regime) and notional_match:
            tier, fallback = 2, "REGIME_FALLBACK"
        elif row_symbol == symbol and notional_match:
            tier, fallback = 3, "REGIME_FALLBACK"
        elif row_symbol == symbol:
            tier, fallback = 4, "REGIME_AND_NOTIONAL_BUCKET_FALLBACK"
        elif _is_global_symbol(row_symbol) and row_regime == regime and notional_match:
            tier, fallback = 5, "SYMBOL_FALLBACK"
        elif _is_global_symbol(row_symbol) and _is_global_regime(row_regime):
            tier, fallback = 6, "GLOBAL_BUCKET_FALLBACK"
        else:
            continue
        ranked.append((tier, fallback, row))

    return [
        (row, fallback)
        for _tier, fallback, row in sorted(
            ranked,
            key=lambda item: (
                item[0],
                _source_priority(str(item[2].get("source") or "")),
                _day_sort_value(item[2]),
                -int(item[2].get("sample_count") or 0),
            ),
        )
    ]


def _row_matches_notional(
    row_bucket: str,
    notional_usdt: float,
    requested_bucket: str | None,
) -> bool:
    if requested_bucket:
        return row_bucket == requested_bucket
    min_notional, max_notional = _notional_bucket_bounds(row_bucket)
    if notional_usdt < min_notional:
        return False
    return max_notional is None or notional_usdt <= max_notional


def _is_global_symbol(symbol: str) -> bool:
    return symbol.upper() in {"", "GLOBAL", "ALL", "*"}


def _is_global_regime(regime: str) -> bool:
    return regime.lower() in {"", "global", "global_default", "all", "*"}


def _float_value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value or 0.0)


def _cost_bucket_id(row: Mapping[str, Any]) -> str:
    return ":".join(
        str(row.get(part) or "unknown")
        for part in ["day", "symbol", "regime", "event_type", "notional_bucket"]
    )


def _day_sort_value(row: Mapping[str, Any]) -> int:
    digits = "".join(character for character in str(row.get("day") or "") if character.isdigit())
    return -int(digits or 0)


def _global_default_estimate(
    symbol: str,
    regime: str,
    notional_usdt: float,
    quantile: str,
) -> CostEstimate:
    return CostEstimate(
        symbol=normalize_symbol(symbol),
        regime=regime,
        notional_usdt=notional_usdt,
        quantile=quantile,
        fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        total_cost_bps=DEFAULT_FALLBACK_COST_BPS,
        cost_bps=DEFAULT_FALLBACK_COST_BPS,
        fallback_level="GLOBAL_DEFAULT",
        source="global_default",
        sample_count=0,
        cost_model_version="global_default_v0",
        bucket_id=None,
        total_cost_bps_p50=DEFAULT_FALLBACK_COST_BPS,
        total_cost_bps_p75=DEFAULT_FALLBACK_COST_BPS,
        total_cost_bps_p90=DEFAULT_FALLBACK_COST_BPS,
        as_of_ts=datetime.now(UTC),
    )


def _normalize_cost_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if not _is_global_symbol(str(normalized.get("symbol") or "")):
        normalized["symbol"] = normalize_symbol(normalized.get("symbol"))
    return normalized


def _row_symbol(row: Mapping[str, Any]) -> str:
    raw = str(row.get("symbol") or "")
    return raw if _is_global_symbol(raw) else normalize_symbol(raw)


def _source_priority(source: str) -> int:
    normalized = source.lower()
    if normalized in {"actual_okx_fills_and_bills", "actual_fills", "mixed_actual_proxy"}:
        return 0
    if normalized == "actual_okx_fills_fee_missing":
        return 1
    if normalized == "public_spread_proxy":
        return 2
    if normalized == "global_default":
        return 3
    return 4


def _row_as_of_ts(row: Mapping[str, Any]) -> datetime | None:
    for key in ("created_at", "as_of_ts"):
        value = row.get(key)
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if value:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed.astimezone(UTC)
    day = row.get("day")
    if day:
        try:
            return datetime.fromisoformat(str(day)).replace(tzinfo=UTC)
        except ValueError:
            return None
    return None
