"""Natural-time funding windows for Alpha Audit v2."""

from __future__ import annotations

import polars as pl


def time_based_funding_observations(
    funding: pl.DataFrame, *, period: str = "20d"
) -> pl.DataFrame:
    """Trailing funding mean over natural time, including this published event.

    The output is on funding publication timestamps.  The later as-of join is
    strict, so an event becomes usable immediately *after* its timestamp, not
    at that exact timestamp and not one settlement cycle later.
    """
    if funding.is_empty():
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "funding_ts": pl.Datetime("us", "UTC"),
                "funding_rate": pl.Float64,
                "observation_count_20d": pl.UInt32,
                "funding_mean_20d": pl.Float64,
            }
        )
    time_col = "funding_ts" if "funding_ts" in funding.columns else "funding_time"
    frame = (
        funding.rename({time_col: "funding_ts"}) if time_col != "funding_ts" else funding
    ).sort(["symbol", "funding_ts"])
    rolled = frame.rolling(
        index_column="funding_ts", period=period, group_by="symbol", closed="right"
    ).agg(
        pl.col("funding_rate").len().alias("observation_count_20d"),
        pl.col("funding_rate").mean().alias("funding_mean_20d"),
        pl.col("funding_ts").min().alias("window_first_ts"),
        pl.col("funding_ts").max().alias("window_last_ts"),
    )
    return frame.join(rolled, on=["symbol", "funding_ts"], how="left")


def funding_fade_time_20d(
    bars_1h_index: pl.DataFrame, funding: pl.DataFrame
) -> pl.DataFrame:
    """Hourly signal grid using only funding published strictly before the bar."""
    schema = {"symbol": pl.Utf8, "feature_ts": pl.Datetime("us", "UTC"), "signal": pl.Float64}
    if funding.is_empty():
        return pl.DataFrame(schema=schema)
    observations = time_based_funding_observations(funding).with_columns(
        (-pl.col("funding_mean_20d")).alias("signal")
    )
    grid = bars_1h_index.select(["symbol", "ts"]).sort(["symbol", "ts"])
    # Each settlement row includes the event at t, but exact matches are
    # forbidden.  Therefore funding_event_ts < decision_ts is the only usable
    # relationship and the event is available from the next instant.
    return grid.join_asof(
        observations.sort(["symbol", "funding_ts"]),
        left_on="ts",
        right_on="funding_ts",
        by="symbol",
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    ).select(["symbol", pl.col("ts").alias("feature_ts"), "signal"])


def funding_frequency_summary(funding: pl.DataFrame) -> pl.DataFrame:
    """Per-symbol count, span, typical frequency, missing rate and max gap."""
    if funding.is_empty():
        return pl.DataFrame()
    time_col = "funding_ts" if "funding_ts" in funding.columns else "funding_time"
    frame = (
        funding.rename({time_col: "funding_ts"}) if time_col != "funding_ts" else funding
    ).sort(["symbol", "funding_ts"])
    rows: list[dict] = []
    for key, group in frame.group_by("symbol", maintain_order=True):
        symbol = key[0] if isinstance(key, tuple) else key
        timestamps = group["funding_ts"].to_list()
        gaps = [
            (right - left).total_seconds() / 3600.0
            for left, right in zip(timestamps, timestamps[1:], strict=False)
        ]
        span_days = (
            (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0
            if len(timestamps) > 1
            else 0.0
        )
        median_hours = float(pl.Series(gaps).median()) if gaps else None
        expected = (
            span_days * 24.0 / median_hours + 1.0
            if median_hours and span_days
            else len(timestamps)
        )
        rows.append(
            {
                "symbol": str(symbol),
                "funding_observation_count": len(timestamps),
                "actual_coverage_days": span_days,
                "median_settlement_frequency_hours": median_hours,
                "missing_rate": max(0.0, 1.0 - len(timestamps) / expected) if expected else 0.0,
                "max_gap_hours": max(gaps) if gaps else None,
            }
        )
    return pl.DataFrame(rows)
