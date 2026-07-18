"""Diagnose beta, liquidity/large-coin, and crash-rebound factor exposures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.paths import ARTIFACTS, DATA_GOLD, DATA_SILVER  # noqa: E402

FACTOR_FILES = {
    "v5.alpha6_static_proxy": "v5__alpha6_static_proxy.parquet",
    "core.rev_xs_480": "core__rev_xs_480.parquet",
    "core.low_vol_480": "core__low_vol_480.parquet",
    "core.funding_fade_480": "core__funding_fade_480.parquet",
}


def main() -> None:
    bars = pl.read_parquet(DATA_SILVER / "bars_1h.parquet").sort(["symbol", "ts"])
    btc = bars.filter(pl.col("symbol") == "BTC-USDT").select(
        ["ts", pl.col("close").pct_change().alias("btc_return")]
    )
    features = (
        bars.with_columns(
            pl.col("close").pct_change().over("symbol").alias("asset_return"),
            (pl.col("close") / pl.col("close").shift(24).over("symbol") - 1.0).alias("ret_24h"),
            pl.col("quote_volume").rolling_mean(24 * 30).over("symbol").log().alias("log_qv_30d"),
        )
        .join(btc, on="ts", how="left")
        .with_columns(
            (pl.col("asset_return") * pl.col("btc_return")).alias("cross_return"),
            (pl.col("btc_return") * pl.col("btc_return")).alias("btc_sq"),
        )
        .with_columns(
            pl.col("cross_return").rolling_mean(24 * 30).over("symbol").alias("mean_cross"),
            pl.col("asset_return").rolling_mean(24 * 30).over("symbol").alias("mean_asset"),
            pl.col("btc_return").rolling_mean(24 * 30).over("symbol").alias("mean_btc"),
            pl.col("btc_sq").rolling_mean(24 * 30).over("symbol").alias("mean_btc_sq"),
        )
        .with_columns(
            (
                (pl.col("mean_cross") - pl.col("mean_asset") * pl.col("mean_btc"))
                / (pl.col("mean_btc_sq") - pl.col("mean_btc") ** 2 + 1e-12)
            ).alias("beta_30d")
        )
        .filter(pl.col("ts").dt.hour() == 0)
        .select(["symbol", pl.col("ts").alias("feature_ts"), "ret_24h", "log_qv_30d", "beta_30d"])
    )
    universes = pl.read_parquet(DATA_GOLD / "dynamic_universe.parquet")
    locks = json.loads((ARTIFACTS / "oos_lock.json").read_text(encoding="utf-8"))["factors"]
    rows: list[dict] = []
    for factor_id, filename in FACTOR_FILES.items():
        signal = pl.read_parquet(DATA_GOLD / "signals" / filename).filter(
            pl.col("feature_ts").dt.hour() == 0
        )
        universe = str(locks[factor_id]["universe"])
        members = universes.filter(pl.col("universe") == universe).select(["date", "symbol"])
        panel = (
            signal.join(features, on=["symbol", "feature_ts"], how="inner")
            .with_columns((pl.col("feature_ts") + pl.duration(hours=1)).dt.date().alias("date"))
            .join(members, on=["date", "symbol"], how="inner")
        )
        for column, meaning in {
            "ret_24h": "recent momentum; negative correlation indicates crash/rebound exposure",
            "log_qv_30d": "liquidity/large-coin proxy",
            "beta_30d": "trailing 30d BTC beta",
        }.items():
            correlations = (
                panel.filter(pl.col(column).is_finite() & pl.col("signal").is_finite())
                .with_columns(
                    pl.col("signal").rank("average").over("feature_ts").alias("signal_rank"),
                    pl.col(column).rank("average").over("feature_ts").alias("exposure_rank"),
                )
                .group_by("feature_ts")
                .agg(
                    pl.len().alias("cross_count"),
                    pl.corr("signal_rank", "exposure_rank").alias("rank_correlation"),
                )
                .filter((pl.col("cross_count") >= 5) & pl.col("rank_correlation").is_finite())
            )
            rows.append(
                {
                    "factor_id": factor_id,
                    "universe": universe,
                    "exposure": column,
                    "meaning": meaning,
                    "period_count": correlations.height,
                    "mean_rank_correlation": float(correlations["rank_correlation"].mean())
                    if correlations.height
                    else None,
                    "median_rank_correlation": float(correlations["rank_correlation"].median())
                    if correlations.height
                    else None,
                }
            )
    pl.DataFrame(rows).write_csv(ARTIFACTS / "factor_exposure_results.csv")
    print(f"factor exposure rows={len(rows)}")


if __name__ == "__main__":
    main()
