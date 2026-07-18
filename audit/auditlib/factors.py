"""Causal factor engines reconstructed from code, not documentation.

``v5_alpha6_static_signal`` is deliberately a *proxy*: it exactly reproduces
the production Alpha6 base-factor arithmetic and static production weights,
but excludes timestamped sentiment, dynamic IC weights, regime allocations,
TrendFollowing/MeanReversion fusion, and portfolio state. It therefore cannot
be represented as the full current V5 signal.
"""

from __future__ import annotations

import polars as pl

BARS_5D = 120
BARS_20D = 480

V5_LIVE_WEIGHTS = {
    "f1_mom_5d": 0.10,
    "f2_mom_20d": 0.30,
    "f3_vol_adj_ret": 0.35,
    "f4_volume_expansion": 0.15,
    "f5_rsi_trend_confirm": 0.10,
}

_FACTOR_COLS = [
    "f1_mom_5d",
    "f2_mom_20d",
    "f3_vol_adj_ret",
    "f4_volume_expansion",
    "f5_rsi_trend_confirm",
]


def v5_raw_factors(bars: pl.DataFrame) -> pl.DataFrame:
    """Per-symbol raw f1..f5 (no cross-sectional normalization).

    bars: [symbol, ts, close, quote_volume] (any sort; function sorts).
    Returns [symbol, feature_ts, f1..f5] with warmup rows as nulls.
    """
    g = "symbol"
    volume_col = "volume" if "volume" in bars.columns else "quote_volume"
    out = (
        bars.sort([g, "ts"])
        .with_columns(
            # Production indexes close[-120] / close[-480], i.e. 119/479 intervals.
            (pl.col("close") / pl.col("close").shift(BARS_5D - 1).over(g) - 1.0).alias("f1_mom_5d"),
            (pl.col("close") / pl.col("close").shift(BARS_20D - 1).over(g) - 1.0).alias(
                "f2_mom_20d"
            ),
            pl.col("close").pct_change().over(g).alias("_ret_1h"),
            pl.col(volume_col).rolling_mean(24).over(g).alias("_vol_recent"),
            pl.col(volume_col).shift(24).rolling_mean(24 * 7).over(g).alias("_vol_prev7"),
            pl.col("close").diff().over(g).alias("_delta"),
        )
        .with_columns(
            pl.col("_ret_1h").rolling_std(BARS_20D, ddof=0).over(g).alias("_vol_20d"),
            pl.col("_delta").clip(lower_bound=0.0).rolling_mean(14).over(g).alias("_avg_gain"),
            (-pl.col("_delta")).clip(lower_bound=0.0).rolling_mean(14).over(g).alias("_avg_loss"),
        )
        .with_columns(
            (pl.col("f2_mom_20d") / (pl.col("_vol_20d") + 1e-12)).alias("f3_vol_adj_ret"),
            (pl.col("_vol_recent") / (pl.col("_vol_prev7") + 1e-12) - 1.0).alias(
                "f4_volume_expansion"
            ),
            (
                (
                    pl.when(pl.col("_avg_loss") == 0)
                    .then(100.0)
                    .otherwise(100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss")))
                    - 50.0
                )
                / 50.0
            ).alias("f5_rsi_trend_confirm"),
        )
        .rename({"ts": "feature_ts"})
        .select(["symbol", "feature_ts", *_FACTOR_COLS])
    )
    return out


_ALPHA6_TYPICAL_RANGES = {
    "f1_mom_5d": 0.10,
    "f2_mom_20d": 0.20,
    "f3_vol_adj_ret": 2.0,
    "f4_volume_expansion": 0.50,
    "f5_rsi_trend_confirm": 1.0,
}
_ALPHA6_CAPS = {
    "f1_mom_5d": 6.0,
    "f2_mom_20d": 6.0,
    "f3_vol_adj_ret": 4.0,
    "f4_volume_expansion": 3.0,
    "f5_rsi_trend_confirm": 3.0,
}


def v5_alpha6_static_signal(
    raw: pl.DataFrame, weights: dict[str, float] | None = None
) -> pl.DataFrame:
    """Static base-only Alpha6 relative score; never label this as full V5."""
    resolved = weights or V5_LIVE_WEIGHTS
    available = list(resolved)
    expressions = [
        (pl.col(column) / _ALPHA6_TYPICAL_RANGES[column])
        .clip(-_ALPHA6_CAPS[column], _ALPHA6_CAPS[column])
        .alias(f"_z_{column}")
        for column in available
    ]
    scored = raw.drop_nulls(available).with_columns(expressions)
    base_score = sum(pl.col(f"_z_{column}") * float(resolved[column]) for column in available)
    return (
        scored.with_columns(base_score.alias("base_score"))
        .with_columns(
            (pl.col("base_score") - pl.col("base_score").mean().over("feature_ts")).alias("signal")
        )
        .select(["symbol", "feature_ts", "signal", "base_score"])
    )


def _robust_z_expr(col: str, over: str = "feature_ts", winsorize_pct: float = 0.05) -> pl.Expr:
    """Production robust_zscore_cross_section as a vectorized expression.

    winsorize [p5, p95] -> median/MAD*1.4826; MAD~0 -> plain z-score fallback.
    """
    c = pl.col(col)
    lo = c.quantile(winsorize_pct).over(over)
    hi = c.quantile(1 - winsorize_pct).over(over)
    clip = c.clip(lo, hi)
    med = clip.median().over(over)
    mad = (clip - med).abs().median().over(over)
    z = (clip - med) / (mad * 1.4826)
    mu = clip.mean().over(over)
    sd = clip.std().over(over)
    z_fallback = pl.when(sd < 1e-12).then(0.0).otherwise((clip - mu) / sd)
    return pl.when(mad < 1e-12).then(z_fallback).otherwise(z).alias(f"_z_{col}")


def v5_composite_signal(raw: pl.DataFrame, weights: dict | None = None) -> pl.DataFrame:
    """Composite V5 signal: per-factor cross-sectional robust-z, live weights.

    raw: output of v5_raw_factors. Output [symbol, feature_ts, signal].
    Rows where any factor is null (warmup) are excluded.
    """
    w = weights or V5_LIVE_WEIGHTS
    cols = list(w)
    df = raw.drop_nulls(cols)
    df = df.with_columns([_robust_z_expr(c) for c in cols])
    expr = sum(pl.col(f"_z_{c}") * w[c] for c in cols)
    return df.with_columns(expr.alias("signal")).select(["symbol", "feature_ts", "signal"])


def rev_xs_480(bars: pl.DataFrame) -> pl.DataFrame:
    """20d cross-sectional reversal: -(ret_480 - xs_mean(ret_480))."""
    return (
        bars.sort(["symbol", "ts"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(BARS_20D).over("symbol") - 1.0).alias(
                "_ret_480"
            )
        )
        .filter(pl.col("_ret_480").is_not_null())
        .with_columns(
            (-(pl.col("_ret_480") - pl.col("_ret_480").mean().over("ts"))).alias("signal")
        )
        .select(["symbol", pl.col("ts").alias("feature_ts"), "signal"])
    )


def low_vol_480(bars: pl.DataFrame) -> pl.DataFrame:
    """20d low volatility: -rolling_std(ret_1h, 480)."""
    return (
        bars.sort(["symbol", "ts"])
        .with_columns(pl.col("close").pct_change().over("symbol").alias("_r"))
        .with_columns((-pl.col("_r").rolling_std(BARS_20D).over("symbol")).alias("signal"))
        .filter(pl.col("signal").is_not_null())
        .select(["symbol", pl.col("ts").alias("feature_ts"), "signal"])
    )


def funding_fade(bars_1h_index: pl.DataFrame, funding: pl.DataFrame) -> pl.DataFrame:
    """Funding fade: -trailing_mean(funding_rate, 20d = 60 8h-observations).

    Signal at bar t uses only funding rows with funding_ts <= t (backward
    asof, no forward fill beyond publication). Rows with < 60 historical
    observations get null signal (insufficient history).
    """
    schema = {"symbol": pl.Utf8, "feature_ts": pl.Datetime("us", "UTC"), "signal": pl.Float64}
    if funding.is_empty():
        return pl.DataFrame(schema=schema)
    grid = bars_1h_index.select(["symbol", "ts"]).sort(["symbol", "ts"])
    funding_time_col = "funding_ts" if "funding_ts" in funding.columns else "funding_time"
    f = (
        funding.rename({funding_time_col: "funding_ts"})
        if funding_time_col != "funding_ts"
        else funding
    ).sort(["symbol", "funding_ts"])
    f = f.with_columns(
        pl.col("funding_rate").rolling_mean(60).over("symbol").alias("_fmean_20d"),
        pl.col("funding_rate").cum_count().over("symbol").alias("_fcount"),
    ).with_columns(
        pl.when(pl.col("_fcount") >= 60).then(-pl.col("_fmean_20d")).otherwise(None).alias("signal")
    )
    return grid.join_asof(
        f,
        left_on="ts",
        right_on="funding_ts",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    ).select(["symbol", pl.col("ts").alias("feature_ts"), "signal"])
