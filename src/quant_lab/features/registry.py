from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from math import isnan
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quant_lab.contracts.models import FeatureValue, require_utc


class FeatureComputeContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_dataset_version: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        return require_utc(value)


FeatureCompute = Callable[[pl.DataFrame, "FeatureSpec", FeatureComputeContext], list[FeatureValue]]


class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_set: str = Field(min_length=1)
    feature_name: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    lookback_bars: int = Field(gt=0)
    description: str = Field(min_length=1)
    compute: FeatureCompute

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.feature_set, self.feature_name, self.feature_version)


FeatureDefinition = FeatureSpec


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: dict[tuple[str, str, str], FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.key in self._features:
            feature_set, feature_name, feature_version = spec.key
            raise ValueError(
                "feature spec already registered: "
                f"{feature_set}/{feature_name}/{feature_version}"
            )
        self._features[spec.key] = spec

    def get(
        self,
        feature_set: str,
        feature_name: str,
        feature_version: str,
    ) -> FeatureSpec | None:
        return self._features.get((feature_set, feature_name, feature_version))

    def list(self) -> list[FeatureSpec]:
        return [self._features[key] for key in sorted(self._features)]

    def list_names(self) -> list[str]:
        return sorted({spec.feature_name for spec in self._features.values()})


class FeatureTimestampLeakageError(ValueError):
    pass


def close_return_spec(
    lookback_bars: int,
    *,
    feature_set: str = "demo",
    feature_version: str = "v0.1",
    timeframe: str = "1H",
) -> FeatureSpec:
    return FeatureSpec(
        feature_set=feature_set,
        feature_name="close_return_n",
        feature_version=feature_version,
        timeframe=timeframe,
        lookback_bars=lookback_bars,
        description="Close-to-close return using only the current and historical bars.",
        compute=compute_close_return_n,
    )


def rolling_volatility_spec(
    lookback_bars: int,
    *,
    feature_set: str = "demo",
    feature_version: str = "v0.1",
    timeframe: str = "1H",
) -> FeatureSpec:
    return FeatureSpec(
        feature_set=feature_set,
        feature_name="rolling_volatility_n",
        feature_version=feature_version,
        timeframe=timeframe,
        lookback_bars=lookback_bars,
        description="Rolling standard deviation of close returns using historical windows only.",
        compute=compute_rolling_volatility_n,
    )


def default_feature_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    registry.register(close_return_spec(lookback_bars=1))
    registry.register(rolling_volatility_spec(lookback_bars=20))
    return registry


def compute_feature_values(
    spec: FeatureSpec,
    market_bars: pl.DataFrame,
    *,
    input_dataset_version: str,
    input_hash: str,
    code_version: str,
    created_at: datetime,
) -> list[FeatureValue]:
    context = FeatureComputeContext(
        input_dataset_version=input_dataset_version,
        input_hash=input_hash,
        code_version=code_version,
        created_at=created_at,
    )
    return spec.compute(market_bars, spec, context)


def compute_close_return_n(
    market_bars: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    bars = _prepare_market_bars(market_bars)
    computed = bars.with_columns(
        ((pl.col("close") / pl.col("close").shift(spec.lookback_bars).over("symbol")) - 1.0).alias(
            "value"
        )
    )
    return _rows_to_feature_values(computed, spec, context)


def compute_rolling_volatility_n(
    market_bars: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    bars = _prepare_market_bars(market_bars)
    computed = (
        bars.with_columns(
            ((pl.col("close") / pl.col("close").shift(1).over("symbol")) - 1.0).alias(
                "_return"
            )
        )
        .with_columns(
            pl.col("_return")
            .rolling_std(window_size=spec.lookback_bars, min_samples=spec.lookback_bars)
            .over("symbol")
            .alias("value")
        )
        .drop("_return")
    )
    return _rows_to_feature_values(computed, spec, context)


def validate_feature_timestamps(
    feature_values: Sequence[FeatureValue | dict[str, Any]] | pl.DataFrame,
    *,
    decision_delay_bars: int = 1,
) -> None:
    if decision_delay_bars < 0:
        raise ValueError("decision_delay_bars must be non-negative")

    records = _feature_records(feature_values)
    if not records:
        return

    timeline_by_symbol: dict[str, list[datetime]] = {}
    for record in records:
        symbol = str(record["symbol"])
        ts = _coerce_utc_datetime(record["ts"], "ts")
        timeline_by_symbol.setdefault(symbol, []).append(ts)

    timeline_by_symbol = {
        symbol: sorted(set(timestamps)) for symbol, timestamps in timeline_by_symbol.items()
    }

    for record in records:
        symbol = str(record["symbol"])
        ts = _coerce_utc_datetime(record["ts"], "ts")
        created_at = record.get("created_at")
        if created_at is not None and ts > _coerce_utc_datetime(created_at, "created_at"):
            raise FeatureTimestampLeakageError(
                f"feature timestamp {ts.isoformat()} is after created_at"
            )

        decision_ts = record.get("decision_ts")
        if decision_ts is None:
            decision_ts = record.get("label_ts")
        if decision_ts is None:
            continue

        decision_ts = _coerce_utc_datetime(decision_ts, "decision_ts")
        if decision_delay_bars == 0:
            if ts > decision_ts:
                raise FeatureTimestampLeakageError(
                    f"feature timestamp {ts.isoformat()} exceeds decision timestamp "
                    f"{decision_ts.isoformat()}"
                )
            continue

        timeline = timeline_by_symbol[symbol]
        candidate_timestamps = [candidate for candidate in timeline if candidate <= decision_ts]
        allowed_index = len(candidate_timestamps) - 1 - decision_delay_bars
        has_decision_timestamp_in_timeline = decision_ts in timeline
        if has_decision_timestamp_in_timeline and allowed_index >= 0:
            allowed_ts = candidate_timestamps[allowed_index]
            if ts > allowed_ts:
                raise FeatureTimestampLeakageError(
                    f"feature timestamp {ts.isoformat()} exceeds allowed delayed timestamp "
                    f"{allowed_ts.isoformat()} for decision {decision_ts.isoformat()}"
                )
            continue

        if ts >= decision_ts:
            raise FeatureTimestampLeakageError(
                f"feature timestamp {ts.isoformat()} is not delayed before decision timestamp "
                f"{decision_ts.isoformat()}"
            )


def _prepare_market_bars(market_bars: pl.DataFrame) -> pl.DataFrame:
    required_columns = {"symbol", "ts", "close"}
    missing_columns = sorted(required_columns.difference(market_bars.columns))
    if missing_columns:
        raise ValueError(f"market_bars missing required columns: {', '.join(missing_columns)}")
    return market_bars.select("symbol", "ts", "close").sort(["symbol", "ts"])


def _rows_to_feature_values(
    computed: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    values: list[FeatureValue] = []
    for row in computed.select("symbol", "ts", "value").iter_rows(named=True):
        values.append(
            FeatureValue(
                feature_set=spec.feature_set,
                feature_name=spec.feature_name,
                feature_version=spec.feature_version,
                symbol=row["symbol"],
                ts=row["ts"],
                value=_normalize_optional_float(row["value"]),
                lookback_bars=spec.lookback_bars,
                input_dataset_version=context.input_dataset_version,
                input_hash=context.input_hash,
                code_version=context.code_version,
                created_at=context.created_at,
            )
        )
    return values


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if isnan(normalized):
        return None
    return normalized


def _feature_records(
    feature_values: Sequence[FeatureValue | dict[str, Any]] | pl.DataFrame,
) -> list[dict[str, Any]]:
    if isinstance(feature_values, pl.DataFrame):
        return feature_values.to_dicts()
    return [_feature_record(item) for item in feature_values]


def _feature_record(item: FeatureValue | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, FeatureValue):
        return item.model_dump()
    if isinstance(item, dict):
        return dict(item)
    raise TypeError(f"unsupported feature value record type: {type(item)!r}")


def _coerce_utc_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    return require_utc(value)
