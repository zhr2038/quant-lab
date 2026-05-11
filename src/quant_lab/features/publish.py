import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import FeatureValue
from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset
from quant_lab.features.registry import (
    FeatureSpec,
    compute_close_return_n,
    compute_feature_values,
    compute_rolling_volatility_n,
    validate_feature_timestamps,
)

MARKET_BAR_DATASET = Path("silver") / "market_bar"
FEATURE_VALUE_DATASET = Path("gold") / "feature_value"
FEATURE_CODE_VERSION = "features.core.v0.1"
FEATURE_VALUE_SCHEMA = {
    "feature_set": pl.Utf8,
    "feature_name": pl.Utf8,
    "feature_version": pl.Utf8,
    "symbol": pl.Utf8,
    "ts": pl.Utf8,
    "value": pl.Float64,
    "lookback_bars": pl.Int64,
    "input_dataset_version": pl.Utf8,
    "input_hash": pl.Utf8,
    "code_version": pl.Utf8,
    "created_at": pl.Utf8,
}


class PublishFeatureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    feature_set: str
    timeframe: str
    market_bar_rows: int = Field(ge=0)
    feature_value_rows: int = Field(ge=0)
    published_rows: int = Field(ge=0)
    feature_names: list[str]
    dataset_path: str
    warnings: list[str] = Field(default_factory=list)


def publish_core_features(
    lake_root: str | Path,
    *,
    feature_set: str = "core",
    timeframe: str = "1H",
) -> PublishFeatureResult:
    root = Path(lake_root)
    market_bars = _market_bars_for_timeframe(root, timeframe)
    if market_bars.is_empty():
        return PublishFeatureResult(
            lake_root=str(root),
            feature_set=feature_set,
            timeframe=timeframe,
            market_bar_rows=0,
            feature_value_rows=read_parquet_dataset(root / FEATURE_VALUE_DATASET).height,
            published_rows=0,
            feature_names=[],
            dataset_path=str(root / FEATURE_VALUE_DATASET),
            warnings=["market_bar missing or empty for feature publishing"],
        )

    specs = core_feature_specs(feature_set=feature_set, timeframe=timeframe)
    context = _feature_context(market_bars)
    values: list[FeatureValue] = []
    for spec in specs:
        values.extend(
            compute_feature_values(
                spec,
                market_bars,
                input_dataset_version=context["input_dataset_version"],
                input_hash=context["input_hash"],
                code_version=FEATURE_CODE_VERSION,
                created_at=context["created_at"],
            )
        )

    validate_feature_timestamps(values)
    rows_written = publish_feature_values(root, values)
    return PublishFeatureResult(
        lake_root=str(root),
        feature_set=feature_set,
        timeframe=timeframe,
        market_bar_rows=market_bars.height,
        feature_value_rows=rows_written,
        published_rows=len(values),
        feature_names=[spec.feature_name for spec in specs],
        dataset_path=str(root / FEATURE_VALUE_DATASET),
        warnings=[],
    )


def core_feature_specs(feature_set: str = "core", timeframe: str = "1H") -> list[FeatureSpec]:
    return [
        _close_return_spec(1, feature_set=feature_set, timeframe=timeframe),
        _close_return_spec(6, feature_set=feature_set, timeframe=timeframe),
        _close_return_spec(24, feature_set=feature_set, timeframe=timeframe),
        _rolling_volatility_spec(24, feature_set=feature_set, timeframe=timeframe),
        _rolling_volatility_spec(72, feature_set=feature_set, timeframe=timeframe),
    ]


def publish_feature_values(
    lake_root: str | Path,
    feature_values: Sequence[FeatureValue],
) -> int:
    if not feature_values:
        return read_parquet_dataset(Path(lake_root) / FEATURE_VALUE_DATASET).height
    frame = feature_values_to_frame(feature_values)
    return upsert_parquet_dataset(
        frame,
        Path(lake_root) / FEATURE_VALUE_DATASET,
        key_columns=["feature_set", "feature_name", "feature_version", "symbol", "ts"],
    )


def feature_values_to_frame(feature_values: Sequence[FeatureValue]) -> pl.DataFrame:
    return pl.DataFrame(
        [value.model_dump(mode="json") for value in feature_values],
        schema=FEATURE_VALUE_SCHEMA,
        orient="row",
    )


def _market_bars_for_timeframe(lake_root: Path, timeframe: str) -> pl.DataFrame:
    bars = read_parquet_dataset(lake_root / MARKET_BAR_DATASET)
    if bars.is_empty():
        return bars
    required = {"symbol", "timeframe", "ts", "close", "is_closed"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"market_bar missing required columns: {', '.join(missing)}")
    return (
        bars.filter((pl.col("timeframe") == timeframe) & pl.col("is_closed"))
        .with_columns(
            [
                _datetime_expr(bars, "ts"),
                pl.col("close").cast(pl.Float64),
            ]
        )
        .sort(["symbol", "ts"])
    )


def _feature_context(market_bars: pl.DataFrame) -> dict[str, Any]:
    latest_ts = market_bars.select(pl.col("ts").max()).item()
    if not isinstance(latest_ts, datetime):
        latest_ts = datetime.now(UTC)
    created_at = max(datetime.now(UTC), latest_ts.astimezone(UTC))
    return {
        "input_dataset_version": f"market_bar:{latest_ts.astimezone(UTC).isoformat()}",
        "input_hash": _market_bar_hash(market_bars),
        "created_at": created_at,
    }


def _market_bar_hash(market_bars: pl.DataFrame) -> str:
    rows = []
    for row in market_bars.select("symbol", "timeframe", "ts", "close").to_dicts():
        normalized = dict(row)
        ts = normalized["ts"]
        if isinstance(ts, datetime):
            normalized["ts"] = ts.astimezone(UTC).isoformat()
        rows.append(normalized)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _close_return_spec(lookback_bars: int, *, feature_set: str, timeframe: str) -> FeatureSpec:
    return FeatureSpec(
        feature_set=feature_set,
        feature_name=f"close_return_{lookback_bars}",
        feature_version="v0.1",
        timeframe=timeframe,
        lookback_bars=lookback_bars,
        description=f"Close return over {lookback_bars} closed bars.",
        compute=compute_close_return_n,
    )


def _rolling_volatility_spec(
    lookback_bars: int,
    *,
    feature_set: str,
    timeframe: str,
) -> FeatureSpec:
    return FeatureSpec(
        feature_set=feature_set,
        feature_name=f"rolling_volatility_{lookback_bars}",
        feature_version="v0.1",
        timeframe=timeframe,
        lookback_bars=lookback_bars,
        description=f"Rolling return volatility over {lookback_bars} closed bars.",
        compute=compute_rolling_volatility_n,
    )


def _datetime_expr(df: pl.DataFrame, column: str) -> pl.Expr:
    expression = pl.col(column)
    if df.schema.get(column) == pl.String:
        return expression.str.to_datetime(time_zone="UTC", strict=False).alias(column)
    return expression.cast(pl.Datetime(time_zone="UTC")).alias(column)
