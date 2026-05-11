from datetime import datetime

import polars as pl

LABEL_SCHEMA = {
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "feature_ts": pl.Datetime(time_zone="UTC"),
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "label_ts": pl.Datetime(time_zone="UTC"),
    "forward_return": pl.Float64,
    "horizon_bars": pl.Int64,
    "decision_delay_bars": pl.Int64,
}


def build_forward_return_labels(
    market_bars: pl.DataFrame,
    horizon_bars: int,
    decision_delay_bars: int,
) -> pl.DataFrame:
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if decision_delay_bars < 1:
        raise ValueError("decision_delay_bars must be at least 1")
    if market_bars.is_empty():
        return pl.DataFrame(schema=LABEL_SCHEMA)
    required = {"symbol", "timeframe", "ts", "close"}
    missing = sorted(required.difference(market_bars.columns))
    if missing:
        raise ValueError(f"market_bars missing required columns: {', '.join(missing)}")

    bars = _normalize_bars(market_bars)
    group = ["symbol", "timeframe"]
    future_offset = decision_delay_bars + horizon_bars
    labels = (
        bars.with_columns(
            [
                pl.col("ts").shift(-decision_delay_bars).over(group).alias("decision_ts"),
                pl.col("ts").shift(-future_offset).over(group).alias("label_ts"),
                pl.col("close").shift(-decision_delay_bars).over(group).alias("_decision_close"),
                pl.col("close").shift(-future_offset).over(group).alias("_future_close"),
            ]
        )
        .with_columns(
            ((pl.col("_future_close") / pl.col("_decision_close")) - 1.0).alias(
                "forward_return"
            )
        )
        .rename({"ts": "feature_ts"})
        .with_columns(
            [
                pl.lit(horizon_bars).alias("horizon_bars"),
                pl.lit(decision_delay_bars).alias("decision_delay_bars"),
            ]
        )
        .select(list(LABEL_SCHEMA))
        .filter(
            pl.col("decision_ts").is_not_null()
            & pl.col("label_ts").is_not_null()
            & pl.col("forward_return").is_not_null()
        )
    )
    return labels.sort(["symbol", "timeframe", "feature_ts"])


def validate_no_label_lookahead(labels: pl.DataFrame) -> None:
    if labels.is_empty():
        return
    for row in labels.to_dicts():
        feature_ts = row["feature_ts"]
        decision_ts = row["decision_ts"]
        label_ts = row["label_ts"]
        if not isinstance(feature_ts, datetime) or not isinstance(decision_ts, datetime):
            raise ValueError("label timestamps must be datetimes")
        if decision_ts <= feature_ts:
            raise ValueError("decision_ts must be after feature_ts")
        if label_ts <= decision_ts:
            raise ValueError("label_ts must be after decision_ts")


def _normalize_bars(market_bars: pl.DataFrame) -> pl.DataFrame:
    bars = market_bars
    if "is_closed" in bars.columns:
        bars = bars.filter(pl.col("is_closed"))
    if bars.schema.get("ts") == pl.String:
        bars = bars.with_columns(pl.col("ts").str.to_datetime(time_zone="UTC", strict=False))
    else:
        bars = bars.with_columns(pl.col("ts").cast(pl.Datetime(time_zone="UTC")).alias("ts"))
    return bars.select(["symbol", "timeframe", "ts", "close"]).sort(["symbol", "timeframe", "ts"])
