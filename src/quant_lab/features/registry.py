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
    required_columns: tuple[str, ...] = ("symbol", "ts", "close")
    output_dtype: str = "float64"

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

    def list_by_feature_set(self, feature_set: str) -> list[FeatureSpec]:
        return [spec for spec in self.list() if spec.feature_set == feature_set]


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


def default_core_registry(
    *,
    feature_version: str = "v0.1",
    timeframe: str = "1H",
) -> FeatureRegistry:
    registry = FeatureRegistry()
    for lookback in [1, 4, 24]:
        registry.register(
            FeatureSpec(
                feature_set="core",
                feature_name=f"close_return_{lookback}",
                feature_version=feature_version,
                timeframe=timeframe,
                lookback_bars=lookback,
                description=f"Close return over {lookback} closed bars.",
                compute=compute_close_return_n,
                required_columns=("symbol", "timeframe", "ts", "close", "is_closed"),
            )
        )
    for lookback in [24, 72]:
        registry.register(
            FeatureSpec(
                feature_set="core",
                feature_name=f"rolling_volatility_{lookback}",
                feature_version=feature_version,
                timeframe=timeframe,
                lookback_bars=lookback,
                description=f"Rolling std of one-bar returns over {lookback} closed bars.",
                compute=compute_rolling_volatility_n,
                required_columns=("symbol", "timeframe", "ts", "close", "is_closed"),
            )
        )
    for feature_name, lookback, description, required_columns, compute in [
        (
            "volume_zscore_24",
            24,
            "Rolling z-score of bar volume over 24 closed bars.",
            ("symbol", "timeframe", "ts", "volume", "is_closed"),
            compute_volume_zscore_n,
        ),
        (
            "range_bps",
            1,
            "Intrabar high-low range divided by close in basis points.",
            ("symbol", "timeframe", "ts", "high", "low", "close", "is_closed"),
            compute_range_bps,
        ),
        (
            "close_position_in_range",
            1,
            "Close location between low and high for the same closed bar.",
            ("symbol", "timeframe", "ts", "high", "low", "close", "is_closed"),
            compute_close_position_in_range,
        ),
        (
            "dollar_volume",
            1,
            "Quote volume or close times base volume for the closed bar.",
            ("symbol", "timeframe", "ts", "close", "volume", "quote_volume", "is_closed"),
            compute_dollar_volume,
        ),
        (
            "liquidity_proxy",
            1,
            "log1p of dollar_volume for the closed bar.",
            ("symbol", "timeframe", "ts", "close", "volume", "quote_volume", "is_closed"),
            compute_liquidity_proxy,
        ),
    ]:
        registry.register(
            FeatureSpec(
                feature_set="core",
                feature_name=feature_name,
                feature_version=feature_version,
                timeframe=timeframe,
                lookback_bars=lookback,
                description=description,
                compute=compute,
                required_columns=required_columns,
            )
        )
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
    group_columns = _group_columns(bars)
    computed = bars.with_columns(
        (
            (pl.col("close") / pl.col("close").shift(spec.lookback_bars).over(group_columns))
            - 1.0
        ).alias("value")
    )
    return _rows_to_feature_values(computed, spec, context)


def compute_rolling_volatility_n(
    market_bars: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    bars = _prepare_market_bars(market_bars)
    group_columns = _group_columns(bars)
    computed = (
        bars.with_columns(
            ((pl.col("close") / pl.col("close").shift(1).over(group_columns)) - 1.0).alias(
                "_return"
            )
        )
        .with_columns(
            pl.col("_return")
            .rolling_std(window_size=spec.lookback_bars, min_samples=spec.lookback_bars)
            .over(group_columns)
            .alias("value")
        )
        .drop("_return")
    )
    return _rows_to_feature_values(computed, spec, context)


def compute_volume_zscore_n(
    market_bars: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    bars = _prepare_market_bars_with_columns(market_bars, ["volume"])
    group_columns = _group_columns(bars)
    rolling_mean = pl.col("volume").rolling_mean(
        window_size=spec.lookback_bars,
        min_samples=spec.lookback_bars,
    ).over(group_columns)
    rolling_std = pl.col("volume").rolling_std(
        window_size=spec.lookback_bars,
        min_samples=spec.lookback_bars,
    ).over(group_columns)
    computed = bars.with_columns(
        [
            rolling_mean.alias("_volume_mean"),
            rolling_std.alias("_volume_std"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("_volume_std") > 0)
            .then((pl.col("volume") - pl.col("_volume_mean")) / pl.col("_volume_std"))
            .otherwise(None)
            .alias("value"),
            pl.when(pl.col("_volume_std").is_null())
            .then(pl.lit("insufficient_lookback"))
            .when(pl.col("_volume_std") <= 0)
            .then(pl.lit("zero_volume_std"))
            .otherwise(None)
            .alias("invalid_reason"),
        ]
    )
    return _rows_to_feature_values(computed, spec, context)


def compute_range_bps(
    market_bars: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    bars = _prepare_market_bars_with_columns(market_bars, ["high", "low", "close"])
    computed = bars.with_columns(
        (((pl.col("high") - pl.col("low")) / pl.col("close")) * 10_000).alias("value")
    )
    return _rows_to_feature_values(computed, spec, context)


def compute_close_position_in_range(
    market_bars: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    bars = _prepare_market_bars_with_columns(market_bars, ["high", "low", "close"])
    range_expr = pl.col("high") - pl.col("low")
    computed = bars.with_columns(
        [
            pl.when(range_expr > 0)
            .then((pl.col("close") - pl.col("low")) / range_expr)
            .otherwise(None)
            .alias("value"),
            pl.when(range_expr <= 0)
            .then(pl.lit("zero_range"))
            .otherwise(None)
            .alias("invalid_reason"),
        ]
    )
    return _rows_to_feature_values(computed, spec, context)


def compute_dollar_volume(
    market_bars: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    bars = _prepare_market_bars_with_columns(market_bars, ["close", "volume", "quote_volume"])
    computed = bars.with_columns(
        pl.coalesce([pl.col("quote_volume"), pl.col("close") * pl.col("volume")]).alias("value")
    )
    return _rows_to_feature_values(computed, spec, context)


def compute_liquidity_proxy(
    market_bars: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    bars = _prepare_market_bars_with_columns(market_bars, ["close", "volume", "quote_volume"])
    dollar_volume = pl.coalesce([pl.col("quote_volume"), pl.col("close") * pl.col("volume")])
    computed = bars.with_columns((dollar_volume + 1.0).log().alias("value"))
    return _rows_to_feature_values(computed, spec, context)


def validate_feature_timestamps(
    feature_values: Sequence[FeatureValue | dict[str, Any]] | pl.DataFrame,
    market_bars: Sequence[dict[str, Any]] | pl.DataFrame | None = None,
    *,
    decision_delay_bars: int = 1,
) -> None:
    if decision_delay_bars < 0:
        raise ValueError("decision_delay_bars must be non-negative")

    records = _feature_records(feature_values)
    if not records:
        return

    if market_bars is not None:
        _validate_against_closed_bars(records, market_bars)

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
    columns = ["symbol", "ts", "close"]
    if "timeframe" in market_bars.columns:
        columns.append("timeframe")
    if "is_closed" in market_bars.columns:
        market_bars = market_bars.filter(pl.col("is_closed"))
    sort_columns = [column for column in ["symbol", "timeframe", "ts"] if column in columns]
    return market_bars.select(columns).sort(sort_columns)


def _prepare_market_bars_with_columns(
    market_bars: pl.DataFrame,
    value_columns: list[str],
) -> pl.DataFrame:
    required_columns = {"symbol", "ts", *value_columns}
    missing_columns = sorted(required_columns.difference(market_bars.columns))
    if missing_columns:
        raise ValueError(f"market_bars missing required columns: {', '.join(missing_columns)}")
    columns = ["symbol", "ts", *value_columns]
    if "timeframe" in market_bars.columns:
        columns.insert(2, "timeframe")
    if "is_closed" in market_bars.columns:
        market_bars = market_bars.filter(pl.col("is_closed"))
    sort_columns = [column for column in ["symbol", "timeframe", "ts"] if column in columns]
    return market_bars.select(columns).sort(sort_columns)


def _rows_to_feature_values(
    computed: pl.DataFrame,
    spec: FeatureSpec,
    context: FeatureComputeContext,
) -> list[FeatureValue]:
    values: list[FeatureValue] = []
    select_columns = ["symbol", "ts", "value"]
    if "invalid_reason" in computed.columns:
        select_columns.append("invalid_reason")
    for row in computed.select(select_columns).iter_rows(named=True):
        value = _normalize_optional_float(row["value"])
        explicit_reason = row.get("invalid_reason")
        invalid_reason = (
            str(explicit_reason)
            if explicit_reason is not None and str(explicit_reason)
            else None
        )
        if value is None and invalid_reason is None:
            invalid_reason = "insufficient_lookback"
        values.append(
            FeatureValue(
                feature_set=spec.feature_set,
                feature_name=spec.feature_name,
                feature_version=spec.feature_version,
                symbol=row["symbol"],
                timeframe=spec.timeframe,
                ts=row["ts"],
                value=value,
                lookback_bars=spec.lookback_bars,
                input_dataset_version=context.input_dataset_version,
                input_hash=context.input_hash,
                code_version=context.code_version,
                created_at=context.created_at,
                source="market_bar",
                is_valid=value is not None and invalid_reason is None,
                invalid_reason=invalid_reason,
            )
        )
    return values


def _group_columns(df: pl.DataFrame) -> list[str]:
    return [column for column in ["symbol", "timeframe"] if column in df.columns]


def _validate_against_closed_bars(
    records: list[dict[str, Any]],
    market_bars: Sequence[dict[str, Any]] | pl.DataFrame,
) -> None:
    bar_records = (
        market_bars.to_dicts() if isinstance(market_bars, pl.DataFrame) else list(market_bars)
    )
    closed_keys: set[tuple[str, str, datetime]] = set()
    for bar in bar_records:
        if bar.get("is_closed", True) is False:
            continue
        symbol = str(bar["symbol"])
        timeframe = str(bar.get("timeframe", "1H"))
        ts = _coerce_utc_datetime(bar["ts"], "market_bar.ts")
        closed_keys.add((symbol, timeframe, ts))

    for record in records:
        symbol = str(record["symbol"])
        timeframe = str(record.get("timeframe", "1H"))
        ts = _coerce_utc_datetime(record["ts"], "ts")
        if (symbol, timeframe, ts) not in closed_keys:
            raise FeatureTimestampLeakageError(
                "feature timestamp is not backed by a closed market_bar: "
                f"{symbol}/{timeframe}/{ts.isoformat()}"
            )


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
