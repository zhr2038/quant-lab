import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset, write_parquet_dataset
from quant_lab.features.registry import (
    FeatureSpec,
    default_core_registry,
    validate_feature_timestamps,
)

MARKET_BAR_DATASET = Path("silver") / "market_bar"
FEATURE_VALUE_DATASET = Path("gold") / "feature_value"
FEATURE_COVERAGE_DATASET = Path("gold") / "feature_coverage_daily"
FEATURE_ANOMALY_DATASET = Path("gold") / "feature_anomaly_daily"
FEATURE_CODE_VERSION_PREFIX = "features.core"

FEATURE_VALUE_SCHEMA = {
    "feature_set": pl.Utf8,
    "feature_name": pl.Utf8,
    "feature_version": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "ts": pl.Datetime(time_zone="UTC"),
    "value": pl.Float64,
    "lookback_bars": pl.Int64,
    "input_dataset_version": pl.Utf8,
    "input_hash": pl.Utf8,
    "code_version": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
    "is_valid": pl.Boolean,
    "invalid_reason": pl.Utf8,
}

FEATURE_COVERAGE_SCHEMA = {
    "day": pl.Utf8,
    "feature_set": pl.Utf8,
    "feature_name": pl.Utf8,
    "feature_version": pl.Utf8,
    "timeframe": pl.Utf8,
    "symbol": pl.Utf8,
    "total_rows": pl.Int64,
    "valid_rows": pl.Int64,
    "null_rows": pl.Int64,
    "coverage": pl.Float64,
    "min_ts": pl.Datetime(time_zone="UTC"),
    "max_ts": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
}

FEATURE_ANOMALY_SCHEMA = {
    "day": pl.Utf8,
    "feature_set": pl.Utf8,
    "feature_name": pl.Utf8,
    "feature_version": pl.Utf8,
    "timeframe": pl.Utf8,
    "symbol": pl.Utf8,
    "anomaly_type": pl.Utf8,
    "anomaly_count": pl.Int64,
    "severity": pl.Utf8,
    "example_ts": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
}


class FeaturePublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    feature_set: str
    feature_version: str
    timeframe: str
    symbols: list[str] | None = None
    feature_count: int = Field(ge=0)
    rows_written: int = Field(ge=0)
    coverage_rows_written: int = Field(ge=0)
    anomaly_rows_written: int = Field(ge=0)
    input_dataset_version: str
    input_hash: str
    code_version: str
    warnings: list[str] = Field(default_factory=list)
    market_bar_rows: int = Field(default=0, ge=0)
    feature_value_rows: int = Field(default=0, ge=0)
    published_rows: int = Field(default=0, ge=0)
    feature_names: list[str] = Field(default_factory=list)
    dataset_path: str


PublishFeatureResult = FeaturePublishResult


class FeatureHealthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    feature_set: str
    date: str | None = None
    coverage_rows: int = Field(ge=0)
    anomaly_rows: int = Field(ge=0)
    low_coverage_count: int = Field(ge=0)
    critical_coverage_count: int = Field(ge=0)
    top_anomalies: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def publish_features(
    lake_root: str | Path,
    *,
    feature_set: str = "core",
    feature_version: str = "v0.1",
    timeframe: str = "1H",
    symbols: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    drop_null: bool = False,
    dry_run: bool = False,
) -> FeaturePublishResult:
    root = Path(lake_root)
    warnings: list[str] = []
    market_bars = market_bars_for_features(
        root,
        timeframe=timeframe,
        symbols=symbols,
        start=start,
        end=end,
    )
    specs = core_feature_specs(
        feature_set=feature_set,
        feature_version=feature_version,
        timeframe=timeframe,
    )
    if market_bars.is_empty():
        warnings.append("market_bar missing or empty for feature publishing")
        existing = read_parquet_dataset(root / FEATURE_VALUE_DATASET).height
        return FeaturePublishResult(
            lake_root=str(root),
            feature_set=feature_set,
            feature_version=feature_version,
            timeframe=timeframe,
            symbols=symbols,
            feature_count=len(specs),
            rows_written=0,
            coverage_rows_written=0,
            anomaly_rows_written=0,
            input_dataset_version="market_bar:none:0",
            input_hash="sha256:empty",
            code_version=_code_version(),
            warnings=warnings,
            market_bar_rows=0,
            feature_value_rows=existing,
            published_rows=0,
            feature_names=[spec.feature_name for spec in specs],
            dataset_path=str(root / FEATURE_VALUE_DATASET),
        )

    context = _feature_context(market_bars)
    feature_frame = _compute_core_features(
        market_bars,
        specs=specs,
        input_dataset_version=context["input_dataset_version"],
        input_hash=context["input_hash"],
        code_version=context["code_version"],
        created_at=context["created_at"],
    )
    if drop_null:
        feature_frame = feature_frame.filter(pl.col("is_valid"))

    validate_feature_timestamps(feature_frame, market_bars)
    coverage = compute_feature_coverage(feature_frame, created_at=context["created_at"])
    anomalies = compute_feature_anomalies(feature_frame, created_at=context["created_at"])

    if dry_run:
        feature_rows = read_parquet_dataset(root / FEATURE_VALUE_DATASET).height
    else:
        feature_rows = _upsert_or_replace_incompatible(
            feature_frame,
            root / FEATURE_VALUE_DATASET,
            key_columns=[
                "feature_set",
                "feature_name",
                "feature_version",
                "symbol",
                "timeframe",
                "ts",
            ],
        )
        _upsert_or_replace_incompatible(
            coverage,
            root / FEATURE_COVERAGE_DATASET,
            key_columns=[
                "day",
                "feature_set",
                "feature_name",
                "feature_version",
                "timeframe",
                "symbol",
            ],
        )
        _upsert_or_replace_incompatible(
            anomalies,
            root / FEATURE_ANOMALY_DATASET,
            key_columns=[
                "day",
                "feature_set",
                "feature_name",
                "feature_version",
                "timeframe",
                "symbol",
                "anomaly_type",
            ],
        )

    return FeaturePublishResult(
        lake_root=str(root),
        feature_set=feature_set,
        feature_version=feature_version,
        timeframe=timeframe,
        symbols=symbols,
        feature_count=len(specs),
        rows_written=0 if dry_run else feature_frame.height,
        coverage_rows_written=0 if dry_run else coverage.height,
        anomaly_rows_written=0 if dry_run else anomalies.height,
        input_dataset_version=context["input_dataset_version"],
        input_hash=context["input_hash"],
        code_version=context["code_version"],
        warnings=warnings,
        market_bar_rows=market_bars.height,
        feature_value_rows=feature_rows,
        published_rows=0 if dry_run else feature_frame.height,
        feature_names=[spec.feature_name for spec in specs],
        dataset_path=str(root / FEATURE_VALUE_DATASET),
    )


def publish_core_features(
    lake_root: str | Path,
    *,
    feature_set: str = "core",
    timeframe: str = "1H",
) -> FeaturePublishResult:
    return publish_features(lake_root=lake_root, feature_set=feature_set, timeframe=timeframe)


def core_feature_specs(
    feature_set: str = "core",
    feature_version: str = "v0.1",
    timeframe: str = "1H",
) -> list[FeatureSpec]:
    registry = default_core_registry(feature_version=feature_version, timeframe=timeframe)
    order = {
        "close_return_1": 0,
        "close_return_4": 1,
        "close_return_24": 2,
        "rolling_volatility_24": 3,
        "rolling_volatility_72": 4,
        "volume_zscore_24": 5,
        "range_bps": 6,
        "close_position_in_range": 7,
        "dollar_volume": 8,
        "liquidity_proxy": 9,
    }
    specs = [
        spec.model_copy(update={"feature_set": feature_set})
        for spec in registry.list_by_feature_set("core")
    ]
    return sorted(specs, key=lambda spec: order.get(spec.feature_name, 999))


def market_bars_for_features(
    lake_root: str | Path,
    *,
    timeframe: str,
    symbols: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    bars = read_parquet_dataset(Path(lake_root) / MARKET_BAR_DATASET)
    if bars.is_empty():
        return bars
    required = {
        "venue",
        "symbol",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_closed",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"market_bar missing required columns: {', '.join(missing)}")
    normalized = bars.with_columns(
        [
            _datetime_expr(bars, "ts"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64)
            if "quote_volume" in bars.columns
            else pl.lit(None, dtype=pl.Float64).alias("quote_volume"),
        ]
    )
    filtered = normalized.filter((pl.col("timeframe") == timeframe) & pl.col("is_closed"))
    if symbols:
        filtered = filtered.filter(pl.col("symbol").is_in(symbols))
    if start is not None:
        filtered = filtered.filter(pl.col("ts") >= _ensure_utc(start))
    if end is not None:
        filtered = filtered.filter(pl.col("ts") <= _ensure_utc(end))
    return filtered.sort(["symbol", "timeframe", "ts"])


def compute_feature_coverage(feature_frame: pl.DataFrame, *, created_at: datetime) -> pl.DataFrame:
    if feature_frame.is_empty():
        return pl.DataFrame(schema=FEATURE_COVERAGE_SCHEMA)
    coverage = (
        feature_frame.with_columns(pl.col("ts").dt.date().cast(pl.Utf8).alias("day"))
        .group_by(
            [
                "day",
                "feature_set",
                "feature_name",
                "feature_version",
                "timeframe",
                "symbol",
            ]
        )
        .agg(
            [
                pl.len().alias("total_rows"),
                pl.col("is_valid").sum().cast(pl.Int64).alias("valid_rows"),
                pl.col("value").is_null().sum().cast(pl.Int64).alias("null_rows"),
                pl.col("ts").min().alias("min_ts"),
                pl.col("ts").max().alias("max_ts"),
            ]
        )
        .with_columns(
            [
                (pl.col("valid_rows") / pl.col("total_rows")).alias("coverage"),
                pl.lit(created_at).alias("created_at"),
            ]
        )
    )
    return coverage.select(list(FEATURE_COVERAGE_SCHEMA))


def compute_feature_anomalies(feature_frame: pl.DataFrame, *, created_at: datetime) -> pl.DataFrame:
    if feature_frame.is_empty():
        return pl.DataFrame(schema=FEATURE_ANOMALY_SCHEMA)
    rows: list[dict[str, Any]] = []
    keys = ["feature_set", "feature_name", "feature_version", "timeframe", "symbol"]
    for group_key, group in feature_frame.with_columns(
        pl.col("ts").dt.date().cast(pl.Utf8).alias("day")
    ).group_by(["day", *keys], maintain_order=True):
        base = dict(zip(["day", *keys], group_key, strict=True))
        valid = group.filter(pl.col("is_valid") & pl.col("value").is_not_null())
        if valid.is_empty():
            rows.append(_anomaly_row(base, "all_null", group.height, "critical", group, created_at))
        elif valid["value"].n_unique() == 1 and valid.height > 1:
            rows.append(
                _anomaly_row(base, "zero_variance", valid.height, "warning", valid, created_at)
            )
        if group.filter(pl.col("value").is_infinite()).height:
            bad = group.filter(pl.col("value").is_infinite())
            rows.append(
                _anomaly_row(base, "infinite_value", bad.height, "critical", bad, created_at)
            )
        if base["feature_name"] == "liquidity_proxy":
            bad = group.filter(pl.col("value") < 0)
            if not bad.is_empty():
                rows.append(
                    _anomaly_row(
                        base,
                        "negative_liquidity_proxy",
                        bad.height,
                        "critical",
                        bad,
                        created_at,
                    )
                )
        extreme = _extreme_zscore_rows(valid)
        if not extreme.is_empty():
            rows.append(
                _anomaly_row(
                    base,
                    "extreme_zscore_abs_gt_10",
                    extreme.height,
                    "warning",
                    extreme,
                    created_at,
                )
            )

    duplicate_keys = [
        "feature_set",
        "feature_name",
        "feature_version",
        "symbol",
        "timeframe",
        "ts",
    ]
    duplicates = (
        feature_frame.group_by(duplicate_keys).len(name="count").filter(pl.col("count") > 1)
    )
    if not duplicates.is_empty():
        for row in duplicates.to_dicts():
            rows.append(
                {
                    "day": row["ts"].date().isoformat(),
                    "feature_set": row["feature_set"],
                    "feature_name": row["feature_name"],
                    "feature_version": row["feature_version"],
                    "timeframe": row["timeframe"],
                    "symbol": row["symbol"],
                    "anomaly_type": "duplicate_feature_key",
                    "anomaly_count": int(row["count"]),
                    "severity": "critical",
                    "example_ts": row["ts"],
                    "created_at": created_at,
                }
            )

    if not rows:
        return pl.DataFrame(schema=FEATURE_ANOMALY_SCHEMA)
    return pl.DataFrame(rows, schema=FEATURE_ANOMALY_SCHEMA, orient="row")


def feature_health(
    lake_root: str | Path,
    *,
    feature_set: str = "core",
    date: str | None = None,
) -> FeatureHealthResult:
    root = Path(lake_root)
    coverage = read_parquet_dataset(root / FEATURE_COVERAGE_DATASET)
    anomalies = read_parquet_dataset(root / FEATURE_ANOMALY_DATASET)
    if not coverage.is_empty():
        coverage = coverage.filter(pl.col("feature_set") == feature_set)
        if date:
            coverage = coverage.filter(pl.col("day") == date)
    if not anomalies.is_empty():
        anomalies = anomalies.filter(pl.col("feature_set") == feature_set)
        if date:
            anomalies = anomalies.filter(pl.col("day") == date)
    warnings: list[str] = []
    low_coverage_count = 0
    critical_coverage_count = 0
    if coverage.is_empty():
        warnings.append("feature_coverage_daily missing or empty")
    else:
        low_coverage_count = coverage.filter(pl.col("coverage") < 0.80).height
        critical_coverage_count = coverage.filter(pl.col("coverage") < 0.50).height
        if critical_coverage_count:
            warnings.append("critical feature coverage below 0.50")
        elif low_coverage_count:
            warnings.append("feature coverage below 0.80")
    top_anomalies = (
        anomalies.group_by(["anomaly_type", "severity"])
        .agg(pl.col("anomaly_count").sum())
        .sort("anomaly_count", descending=True)
        .head(20)
        .to_dicts()
        if not anomalies.is_empty()
        else []
    )
    return FeatureHealthResult(
        lake_root=str(root),
        feature_set=feature_set,
        date=date,
        coverage_rows=coverage.height,
        anomaly_rows=anomalies.height,
        low_coverage_count=low_coverage_count,
        critical_coverage_count=critical_coverage_count,
        top_anomalies=top_anomalies,
        warnings=warnings,
    )


def _upsert_or_replace_incompatible(
    frame: pl.DataFrame,
    dataset_path: Path,
    *,
    key_columns: list[str],
) -> int:
    existing = read_parquet_dataset(dataset_path)
    if not existing.is_empty():
        missing_columns = set(frame.columns).difference(existing.columns)
        if missing_columns:
            write_parquet_dataset(frame, dataset_path)
            return frame.height
    return upsert_parquet_dataset(frame, dataset_path, key_columns=key_columns)


def _compute_core_features(
    market_bars: pl.DataFrame,
    *,
    specs: list[FeatureSpec],
    input_dataset_version: str,
    input_hash: str,
    code_version: str,
    created_at: datetime,
) -> pl.DataFrame:
    group = ["symbol", "timeframe"]
    base = (
        market_bars.with_columns(
            [
                ((pl.col("close") / pl.col("close").shift(1).over(group)) - 1.0).alias(
                    "_return_1"
                ),
                ((pl.col("high") - pl.col("low")) / pl.col("close") * 10_000).alias(
                    "range_bps"
                ),
                pl.when((pl.col("high") - pl.col("low")) == 0)
                .then(None)
                .otherwise((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
                .alias("close_position_in_range"),
                pl.coalesce([pl.col("quote_volume"), pl.col("close") * pl.col("volume")]).alias(
                    "dollar_volume"
                ),
            ]
        )
        .with_columns(
            [
                (pl.col("dollar_volume") + 1.0).log().alias("liquidity_proxy"),
                (
                    (pl.col("volume") - pl.col("volume").rolling_mean(24).over(group))
                    / pl.col("volume").rolling_std(24).over(group)
                ).alias("volume_zscore_24"),
                pl.col("_return_1").rolling_std(24, min_samples=24).over(group).alias(
                    "rolling_volatility_24"
                ),
                pl.col("_return_1").rolling_std(72, min_samples=72).over(group).alias(
                    "rolling_volatility_72"
                ),
                ((pl.col("close") / pl.col("close").shift(4).over(group)) - 1.0).alias(
                    "close_return_4"
                ),
                ((pl.col("close") / pl.col("close").shift(24).over(group)) - 1.0).alias(
                    "close_return_24"
                ),
                pl.col("_return_1").alias("close_return_1"),
            ]
        )
        .with_columns(
            [
                pl.when((pl.col("high") - pl.col("low")) == 0)
                .then(pl.lit("zero_range"))
                .otherwise(None)
                .alias("_close_position_reason"),
                pl.when(pl.col("volume").rolling_std(24).over(group) == 0)
                .then(pl.lit("zero_volume_std"))
                .otherwise(None)
                .alias("_volume_zscore_reason"),
            ]
        )
    )
    frames = [
        _long_feature_frame(
            base,
            spec,
            input_dataset_version=input_dataset_version,
            input_hash=input_hash,
            code_version=code_version,
            created_at=created_at,
        )
        for spec in specs
    ]
    combined = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    if combined.is_empty():
        return pl.DataFrame(schema=FEATURE_VALUE_SCHEMA)
    return combined.select(list(FEATURE_VALUE_SCHEMA)).sort(
        ["symbol", "timeframe", "ts", "feature_name"]
    )


def _long_feature_frame(
    wide: pl.DataFrame,
    spec: FeatureSpec,
    *,
    input_dataset_version: str,
    input_hash: str,
    code_version: str,
    created_at: datetime,
) -> pl.DataFrame:
    value_column = spec.feature_name
    reason_expr = _invalid_reason_expr(spec.feature_name, value_column)
    return wide.select(
        [
            pl.lit(spec.feature_set).alias("feature_set"),
            pl.lit(spec.feature_name).alias("feature_name"),
            pl.lit(spec.feature_version).alias("feature_version"),
            pl.col("symbol"),
            pl.col("timeframe"),
            pl.col("ts"),
            pl.col(value_column).cast(pl.Float64).alias("value"),
            pl.lit(spec.lookback_bars).alias("lookback_bars"),
            pl.lit(input_dataset_version).alias("input_dataset_version"),
            pl.lit(input_hash).alias("input_hash"),
            pl.lit(code_version).alias("code_version"),
            pl.lit(created_at).alias("created_at"),
            pl.lit("market_bar").alias("source"),
            (pl.col(value_column).is_not_null() & pl.col(value_column).is_finite()).alias(
                "is_valid"
            ),
            reason_expr.alias("invalid_reason"),
        ]
    )


def _invalid_reason_expr(feature_name: str, value_column: str) -> pl.Expr:
    base_null_reason = pl.when(pl.col(value_column).is_null()).then(pl.lit("insufficient_lookback"))
    if feature_name == "close_position_in_range":
        return (
            pl.when(pl.col("_close_position_reason").is_not_null())
            .then(pl.col("_close_position_reason"))
            .otherwise(base_null_reason.otherwise(None))
        )
    if feature_name == "volume_zscore_24":
        return (
            pl.when(pl.col("_volume_zscore_reason").is_not_null())
            .then(pl.col("_volume_zscore_reason"))
            .otherwise(base_null_reason.otherwise(None))
        )
    return base_null_reason.otherwise(None)


def _feature_context(market_bars: pl.DataFrame) -> dict[str, Any]:
    latest_ts = market_bars.select(pl.col("ts").max()).item()
    latest_utc = latest_ts.astimezone(UTC) if isinstance(latest_ts, datetime) else datetime.now(UTC)
    row_count = market_bars.height
    return {
        "input_dataset_version": f"market_bar:{latest_utc.isoformat()}:{row_count}",
        "input_hash": _market_bar_hash(market_bars),
        "code_version": _code_version(),
        "created_at": max(datetime.now(UTC), latest_utc),
    }


def _market_bar_hash(market_bars: pl.DataFrame) -> str:
    rows = []
    columns = [
        column
        for column in [
            "venue",
            "symbol",
            "timeframe",
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
        ]
        if column in market_bars.columns
    ]
    for row in market_bars.select(columns).to_dicts():
        normalized = dict(row)
        ts = normalized.get("ts")
        if isinstance(ts, datetime):
            normalized["ts"] = ts.astimezone(UTC).isoformat()
        rows.append(normalized)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _code_version() -> str:
    commit = _git_commit()
    if commit:
        return f"{FEATURE_CODE_VERSION_PREFIX}:{commit}"
    return f"{FEATURE_CODE_VERSION_PREFIX}:{__version__}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _datetime_expr(df: pl.DataFrame, column: str) -> pl.Expr:
    expression = pl.col(column)
    if df.schema.get(column) == pl.String:
        return expression.str.to_datetime(time_zone="UTC", strict=False).alias(column)
    return expression.cast(pl.Datetime(time_zone="UTC")).alias(column)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    return value.astimezone(UTC)


def _extreme_zscore_rows(df: pl.DataFrame) -> pl.DataFrame:
    if df.height < 3:
        return pl.DataFrame()
    std = df["value"].std()
    if std is None or std == 0:
        return pl.DataFrame()
    mean = df["value"].mean()
    return df.with_columns(((pl.col("value") - mean) / std).abs().alias("_z")).filter(
        pl.col("_z") > 10
    )


def _anomaly_row(
    base: dict[str, Any],
    anomaly_type: str,
    count: int,
    severity: str,
    rows: pl.DataFrame,
    created_at: datetime,
) -> dict[str, Any]:
    example_ts = rows.select(pl.col("ts").min()).item()
    return {
        **base,
        "anomaly_type": anomaly_type,
        "anomaly_count": int(count),
        "severity": severity,
        "example_ts": example_ts,
        "created_at": created_at,
    }
