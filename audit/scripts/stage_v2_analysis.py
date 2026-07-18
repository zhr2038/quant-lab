# ruff: noqa: E501
"""Execute the locked Alpha Audit v2 quantitative analyses.

This stage never selects a new factor or parameter.  It adds only the missing
24h rebalance frequency, the natural-time funding definition, predeclared
statistical sensitivity, and null distributions requested by Audit v2.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.evaluate import build_labels, evaluate_factor  # noqa: E402
from audit.auditlib.factors import funding_fade  # noqa: E402
from audit.auditlib.funding_v2 import (  # noqa: E402
    funding_fade_time_20d,
    funding_frequency_summary,
)
from audit.auditlib.ic_stats import ic_triad, plain_tstat  # noqa: E402
from audit.auditlib.multiple_testing import adjust_pvalues  # noqa: E402
from audit.auditlib.portfolio_backtest import (  # noqa: E402
    _capped_weights,
    build_market_matrix,
    performance_metrics,
    simulate_long_only,
)
from audit.auditlib.statistics_v2 import (  # noqa: E402
    annualized_sharpe,
    compounded_return,
    empirical_pvalues,
    hac_bandwidth_sensitivity,
    max_stat_adjusted_pvalue,
    moving_block_bootstrap,
)

V1_CUTOFF = datetime(2026, 7, 17, 23, tzinfo=UTC)
FACTORS = {
    "core.low_vol_480": "core__low_vol_480.parquet",
    "core.rev_xs_480": "core__rev_xs_480.parquet",
}
MAX_WEIGHT = {1: 1.0, 3: 0.50, 5: 0.35}
BLOCK_HOURS = [24, 72, 120, 240]
BOOTSTRAP_COUNT = 1000
PERMUTATION_COUNT = 1000


def _paths() -> tuple[Path, Path, Path]:
    root = Path(os.environ.get("AUDIT_ROOT", "/home/hr/quant-alpha-audit-v2"))
    v1 = Path(os.environ.get("AUDIT_V1_DIR", str(root / "v1")))
    v1_root = Path(os.environ.get("AUDIT_V1_ROOT", "/home/hr/quant-alpha-audit"))
    for directory in (root / "artifacts", root / "reports", root / "manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    return root, v1, v1_root


def _at_or_after(timestamps: list[datetime], target: datetime) -> int:
    for index, timestamp in enumerate(timestamps):
        if timestamp >= target:
            return index
    return len(timestamps)


def _periods(lock: dict, timestamps: list[datetime]) -> dict[str, tuple[int, int]]:
    parsed = {
        name: (datetime.fromisoformat(bounds[0]), datetime.fromisoformat(bounds[1]))
        for name, bounds in lock["data_split"].items()
    }
    return {
        name: (_at_or_after(timestamps, left), _at_or_after(timestamps, right))
        for name, (left, right) in parsed.items()
    }


def _with_universe(labels: pl.DataFrame, universes: pl.DataFrame, name: str) -> pl.DataFrame:
    members = universes.filter(pl.col("universe") == name).select(["date", "symbol"])
    return (
        labels.with_columns(pl.col("decision_ts").dt.date().alias("date"))
        .join(members, on=["date", "symbol"], how="inner")
        .drop("date")
    )


def _period_labels(labels: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return labels.filter(
        (pl.col("feature_ts") >= start)
        & (pl.col("feature_ts") < end)
        & (pl.col("label_ts") < end)
    )


def _ic_frame(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.with_columns(
            pl.col("signal").rank("average").over("decision_ts").alias("_signal_rank"),
            pl.col("forward_return").rank("average").over("decision_ts").alias("_return_rank"),
        )
        .group_by("decision_ts")
        .agg(
            pl.len().alias("cross_count"),
            pl.corr("_signal_rank", "_return_rank").alias("rank_ic"),
            pl.corr("signal", "forward_return").alias("pearson_ic"),
        )
        .filter(
            (pl.col("cross_count") >= 5)
            & pl.col("rank_ic").is_finite()
            & pl.col("pearson_ic").is_finite()
        )
        .sort("decision_ts")
    )


def _run_24h(
    root: Path,
    v1: Path,
    market,
    factor_locks: dict,
    periods: dict[str, tuple[int, int]],
) -> tuple[pl.DataFrame, dict[str, tuple[object, dict]]]:
    rows: list[dict] = []
    contributions: list[dict] = []
    bootstrap_rows: list[dict] = []
    simulations: dict[str, tuple[object, dict]] = {}
    for factor_id in FACTORS:
        universe = str(factor_locks[factor_id]["universe"])
        for top_n in (1, 3, 5):
            for weighting in ("equal", "score"):
                for btc_filter in (False, True):
                    structure_id = (
                        f"{factor_id}|top{top_n}|{weighting}|24h|"
                        f"staggered=0|btc_filter={int(btc_filter)}"
                    )
                    config = {
                        "structure_id": structure_id,
                        "factor_id": factor_id,
                        "universe": universe,
                        "top_n": top_n,
                        "weighting": weighting,
                        "max_single_weight": MAX_WEIGHT[top_n],
                        "rebalance_hours": 24,
                        "staggered": False,
                        "btc_filter": btc_filter,
                        "cost_scenario": "base",
                        "one_way_cost_bps": 15.0,
                        "roundtrip_cost_floor_bps": 30.0,
                    }
                    simulation = simulate_long_only(
                        market,
                        factor_id=factor_id,
                        universe=universe,
                        top_n=top_n,
                        weighting=weighting,
                        max_weight=MAX_WEIGHT[top_n],
                        rebalance_hours=24,
                        staggered=False,
                        btc_filter=btc_filter,
                    )
                    denominator = float(np.abs(simulation.contribution_by_symbol).sum())
                    shares = []
                    for symbol, value in zip(
                        market.symbols, simulation.contribution_by_symbol, strict=True
                    ):
                        if value == 0:
                            continue
                        share = abs(float(value)) / denominator if denominator else 0.0
                        shares.append(share)
                        contributions.append(
                            {
                                "structure_id": structure_id,
                                "factor_id": factor_id,
                                "symbol": symbol,
                                "gross_return_contribution": float(value),
                                "absolute_contribution_share": share,
                            }
                        )
                    concentration = max(shares, default=0.0)
                    full_series: dict | None = None
                    for period, (left, right) in periods.items():
                        metrics, series = performance_metrics(
                            simulation,
                            market,
                            one_way_cost_bps=15.0,
                            fee_bps=10.0,
                            start_index=left,
                            end_index=right,
                            rebalance_hours=24,
                        )
                        row = {
                            **config,
                            "period": period,
                            **metrics,
                            "annualized_return": metrics["cagr"],
                            "hit_rate": metrics["win_rate"],
                            "average_holding_return": metrics["mean_trade_gross_return"],
                            "average_trade_net_return": metrics["mean_trade_net_return"],
                            "hac_t_stat": metrics["daily_hac_tstat"],
                            "max_symbol_contribution_share": concentration,
                        }
                        rows.append(row)
                        if period == "full":
                            full_series = series
                    assert full_series is not None
                    simulations[structure_id] = (simulation, full_series)
                    daily_net = np.asarray(full_series["daily_net"], dtype=float)
                    for block_hours, block_days in zip(
                        BLOCK_HOURS, (1, 3, 5, 10), strict=True
                    ):
                        return_summary = moving_block_bootstrap(
                            daily_net,
                            block_sizes=[block_days],
                            statistic=compounded_return,
                            n_bootstrap=BOOTSTRAP_COUNT,
                            seed=20260718 + block_hours + len(bootstrap_rows),
                        )[0]
                        sharpe_summary = moving_block_bootstrap(
                            daily_net,
                            block_sizes=[block_days],
                            statistic=lambda values: annualized_sharpe(values, 365.25),
                            n_bootstrap=BOOTSTRAP_COUNT,
                            seed=20260718 + block_hours + len(bootstrap_rows) + 100_000,
                        )[0]
                        bootstrap_rows.append(
                            {
                                "object_id": structure_id,
                                "object_type": "portfolio_24h",
                                "metric": "net_return_and_sharpe",
                                "block_size_hours": block_hours,
                                "bootstrap_count": BOOTSTRAP_COUNT,
                                "estimate": return_summary["estimate"],
                                "ci_95_low": return_summary["ci_95_low"],
                                "ci_95_high": return_summary["ci_95_high"],
                                "median": return_summary["median"],
                                "probability_net_return_gt_zero": return_summary[
                                    "probability_greater_than_zero"
                                ],
                                "probability_sharpe_gt_zero": sharpe_summary[
                                    "probability_greater_than_zero"
                                ],
                                "probability_ic_gt_zero": None,
                            }
                        )
    result = pl.DataFrame(rows).sort(
        ["factor_id", "top_n", "weighting", "btc_filter", "period"]
    )
    result.write_csv(root / "artifacts/portfolio_24h_results.csv")
    pl.DataFrame(contributions).write_csv(
        root / "artifacts/portfolio_24h_symbol_contribution.csv"
    )
    pl.DataFrame(bootstrap_rows).write_csv(root / "artifacts/block_bootstrap_summary.csv")
    return result, simulations


def _locked_low_vol_portfolios(
    root: Path,
    v1: Path,
    market,
    periods: dict[str, tuple[int, int]],
) -> pl.DataFrame:
    lock = json.loads((v1 / "artifacts/portfolio_oos_lock.json").read_text())["factors"][
        "core.low_vol_480"
    ]
    rows: list[dict] = []
    for btc_filter, hypothesis_type in (
        (False, "V1_LOCKED_CONFIRMATORY"),
        (True, "POST_HOC_HYPOTHESIS"),
    ):
        simulation = simulate_long_only(
            market,
            factor_id="core.low_vol_480",
            universe=str(lock["universe"]),
            top_n=int(lock["top_n"]),
            weighting=str(lock["weighting"]),
            max_weight=float(lock["max_single_weight"]),
            rebalance_hours=int(lock["rebalance_hours"]),
            staggered=bool(lock["staggered"]),
            btc_filter=btc_filter,
        )
        for period, (left, right) in periods.items():
            metrics, _ = performance_metrics(
                simulation,
                market,
                one_way_cost_bps=15.0,
                fee_bps=10.0,
                start_index=left,
                end_index=right,
                rebalance_hours=120,
            )
            rows.append(
                {
                    "object": "low_vol_locked" if not btc_filter else "low_vol_btc_trend",
                    "hypothesis_type": hypothesis_type,
                    "parameters_locked": True,
                    "factor_id": "core.low_vol_480",
                    "universe": lock["universe"],
                    "top_n": lock["top_n"],
                    "weighting": lock["weighting"],
                    "rebalance_hours": lock["rebalance_hours"],
                    "btc_filter": btc_filter,
                    "btc_trend_lookback_hours": 1440,
                    "period": period,
                    **metrics,
                }
            )
    frame = pl.DataFrame(rows)
    frame.write_csv(root / "artifacts/low_vol_locked_and_posthoc_portfolios.csv")
    return frame


def _signal_analysis(
    root: Path,
    v1: Path,
    bars: pl.DataFrame,
    universes: pl.DataFrame,
    signals: dict[str, pl.DataFrame],
) -> tuple[dict[str, object], dict[str, np.ndarray], pl.DataFrame]:
    lock = json.loads((v1 / "artifacts/oos_lock.json").read_text())
    bounds = {
        name: (datetime.fromisoformat(pair[0]), datetime.fromisoformat(pair[1]))
        for name, pair in lock["data_split"].items()
    }
    labels24 = build_labels(bars, 24, decision_delay_bars=1)
    panels: dict[str, pl.DataFrame] = {}
    eval_series: dict[str, np.ndarray] = {}
    primary: dict[str, object] = {}
    sensitivity_rows: list[dict] = []
    for factor_id, universe in (
        ("core.low_vol_480", "top20"),
        ("core.rev_xs_480", "top50"),
    ):
        labels = _with_universe(labels24, universes, universe)
        panel = labels.join(
            signals[factor_id].select(["symbol", "feature_ts", "signal"]),
            on=["symbol", "feature_ts"],
            how="inner",
        ).filter(pl.col("signal").is_not_null())
        panels[factor_id] = panel
        ic = _ic_frame(panel)
        result = evaluate_factor(
            signals[factor_id],
            labels,
            factor_id=factor_id,
            direction=1,
            horizon_bars=24,
            universe_name=universe,
            period_name="full",
            min_cross=5,
        )
        rank_values = np.asarray(list(result.rank_ic_by_ts.values()), dtype=float)
        eval_series[f"{factor_id}.rank_ic"] = rank_values
        eval_series[f"{factor_id}.pearson_ic"] = ic["pearson_ic"].to_numpy()
        primary[factor_id] = {
            **result.flat_dict(),
            "pearson_ic_mean": float(ic["pearson_ic"].mean()),
            "pearson_hac_tstat": float(
                hac_bandwidth_sensitivity(ic["pearson_ic"].to_numpy(), 24)[1]["t_stat"]
            ),
        }
        for period, (start, end) in bounds.items():
            subset = ic.filter(
                (pl.col("decision_ts") >= start) & (pl.col("decision_ts") < end)
            )
            for metric in ("rank_ic", "pearson_ic"):
                values = subset[metric].to_numpy()
                tri = ic_triad(
                    values,
                    np.asarray(
                        [int(value.timestamp()) for value in subset["decision_ts"].to_list()],
                        dtype=np.int64,
                    ),
                    24,
                )
                sensitivity_rows.append(
                    {
                        "factor_id": factor_id,
                        "slice_type": "oos_period",
                        "slice": period,
                        "metric": metric,
                        "n_observations": tri.n_periods,
                        "estimate": tri.ic_mean,
                        "hac_tstat": tri.hac_tstat,
                        "non_overlap_count": tri.non_overlap_count,
                        "non_overlap_tstat": tri.non_overlap_tstat,
                    }
                )

    low_panel = panels["core.low_vol_480"]
    # Calendar stability from the exact 24h/top20 panel.
    low_ic = _ic_frame(low_panel).with_columns(
        pl.col("decision_ts").dt.year().alias("year"),
        (
            pl.col("decision_ts").dt.year().cast(pl.Utf8)
            + pl.lit("Q")
            + (((pl.col("decision_ts").dt.month() - 1) // 3) + 1).cast(pl.Utf8)
        ).alias("quarter"),
    )
    for column, slice_type in (("year", "calendar_year"), ("quarter", "calendar_quarter")):
        for key, group in low_ic.group_by(column, maintain_order=True):
            value = key[0] if isinstance(key, tuple) else key
            for metric in ("rank_ic", "pearson_ic"):
                values = group[metric].to_numpy()
                sensitivity_rows.append(
                    {
                        "factor_id": "core.low_vol_480",
                        "slice_type": slice_type,
                        "slice": str(value),
                        "metric": metric,
                        "n_observations": len(values),
                        "estimate": float(np.mean(values)),
                        "hac_tstat": hac_bandwidth_sensitivity(values, 24)[1]["t_stat"],
                        "non_overlap_count": int(math.ceil(len(values) / 24)),
                        "non_overlap_tstat": None,
                    }
                )

    # Reuse v1's complete locked horizon/universe grid rather than searching a new one.
    v1_horizons = pl.read_csv(v1 / "artifacts/factor_horizon_results.csv").filter(
        (pl.col("factor_id") == "core.low_vol_480")
        & (pl.col("direction") == 1)
        & (pl.col("period") == "full")
        & (
            ((pl.col("universe") == "top20"))
            | (pl.col("horizon_bars") == 24)
        )
    )
    for row in v1_horizons.iter_rows(named=True):
        sensitivity_rows.append(
            {
                "factor_id": row["factor_id"],
                "slice_type": "v1_locked_grid_reuse",
                "slice": f"h{row['horizon_bars']}_{row['universe']}",
                "metric": "rank_ic",
                "n_observations": row["n_periods"],
                "estimate": row["ic_mean"],
                "hac_tstat": row["hac_tstat"],
                "non_overlap_count": row["non_overlap_count"],
                "non_overlap_tstat": row["non_overlap_tstat"],
            }
        )

    # Exclusion checks: majors and v1's largest selected-return contributor.
    contribution = pl.read_csv(v1 / "artifacts/factor_symbol_contribution.csv").filter(
        (pl.col("factor_id") == "core.low_vol_480")
        & (pl.col("horizon_bars") == 24)
        & (pl.col("universe") == "top20")
    )
    largest_symbol = (
        contribution.sort("absolute_contribution_share", descending=True)["symbol"][0]
        if contribution.height
        else "BTC-USDT"
    )
    for name, excluded in (
        ("exclude_btc_eth_sol", ["BTC-USDT", "ETH-USDT", "SOL-USDT"]),
        ("exclude_largest_contributor", [largest_symbol]),
    ):
        subset = low_panel.filter(~pl.col("symbol").is_in(excluded))
        ic = _ic_frame(subset)
        sensitivity_rows.append(
            {
                "factor_id": "core.low_vol_480",
                "slice_type": "coin_exclusion",
                "slice": f"{name}:{','.join(excluded)}",
                "metric": "rank_ic",
                "n_observations": ic.height,
                "estimate": float(ic["rank_ic"].mean()),
                "hac_tstat": hac_bandwidth_sensitivity(ic["rank_ic"].to_numpy(), 24)[1][
                    "t_stat"
                ],
                "non_overlap_count": int(math.ceil(ic.height / 24)),
                "non_overlap_tstat": None,
            }
        )

    # Daily cross-sectional beta and liquidity residualization.
    btc = bars.filter(pl.col("symbol") == "BTC-USDT").select(
        ["ts", pl.col("close").pct_change().alias("btc_return")]
    )
    exposures = (
        bars.sort(["symbol", "ts"])
        .with_columns(
            pl.col("close").pct_change().over("symbol").alias("asset_return"),
            pl.col("quote_volume").rolling_mean(24 * 30).over("symbol").log().alias("log_qv_30d"),
        )
        .join(btc, on="ts", how="left")
        .with_columns(
            (pl.col("asset_return") * pl.col("btc_return")).alias("cross_return"),
            (pl.col("btc_return") ** 2).alias("btc_sq"),
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
        .select(["symbol", pl.col("ts").alias("feature_ts"), "beta_30d", "log_qv_30d"])
    )
    neutral_panel = low_panel.filter(pl.col("feature_ts").dt.hour() == 0).join(
        exposures, on=["symbol", "feature_ts"], how="inner"
    )
    for exposure in ("beta_30d", "log_qv_30d"):
        residualized = (
            neutral_panel.filter(pl.col(exposure).is_finite())
            .with_columns(
                (pl.col("signal") - pl.col("signal").mean().over("decision_ts")).alias("_s"),
                (pl.col(exposure) - pl.col(exposure).mean().over("decision_ts")).alias("_x"),
            )
            .with_columns(
                (
                    (pl.col("_s") * pl.col("_x")).sum().over("decision_ts")
                    / ((pl.col("_x") ** 2).sum().over("decision_ts") + 1e-12)
                ).alias("_slope")
            )
            .with_columns((pl.col("_s") - pl.col("_slope") * pl.col("_x")).alias("signal"))
        )
        ic = _ic_frame(residualized)
        sensitivity_rows.append(
            {
                "factor_id": "core.low_vol_480",
                "slice_type": "neutralized",
                "slice": exposure,
                "metric": "rank_ic",
                "n_observations": ic.height,
                "estimate": float(ic["rank_ic"].mean()),
                "hac_tstat": hac_bandwidth_sensitivity(ic["rank_ic"].to_numpy(), 1)[1][
                    "t_stat"
                ],
                "non_overlap_count": ic.height,
                "non_overlap_tstat": plain_tstat(ic["rank_ic"].to_numpy()),
            }
        )

    sensitivity = pl.DataFrame(sensitivity_rows)
    sensitivity.write_csv(root / "artifacts/low_vol_signal_sensitivity.csv")
    return primary, eval_series, low_panel


def _funding_analysis(
    root: Path,
    bars: pl.DataFrame,
    universes: pl.DataFrame,
    funding: pl.DataFrame,
) -> tuple[dict, np.ndarray]:
    frequency = funding_frequency_summary(funding)
    frequency.write_csv(root / "artifacts/funding_frequency_summary.csv")
    grid = bars.select(["symbol", "ts"])
    fixed = funding_fade(grid, funding).rename({"signal": "fixed_60_signal"})
    time_based = funding_fade_time_20d(grid, funding).rename({"signal": "time_20d_signal"})
    joined = fixed.join(time_based, on=["symbol", "feature_ts"], how="full", coalesce=True)
    labels = _with_universe(build_labels(bars, 72), universes, "top20")
    evaluations: list[dict] = []
    series = np.asarray([], dtype=float)
    for name, signal_col in (("FIXED_60_OBSERVATIONS", "fixed_60_signal"), ("TIME_BASED_20D", "time_20d_signal")):
        signal = joined.select(
            ["symbol", "feature_ts", pl.col(signal_col).alias("signal")]
        ).filter(pl.col("signal").is_not_null())
        result = evaluate_factor(
            signal,
            labels,
            factor_id=f"funding_fade.{name.lower()}",
            direction=1,
            horizon_bars=72,
            universe_name="top20",
            period_name="full",
            min_cross=5,
        )
        if name == "TIME_BASED_20D":
            series = np.asarray(list(result.rank_ic_by_ts.values()), dtype=float)
        evaluations.append({"window_type": name, **result.flat_dict()})
    paired = joined.filter(
        pl.col("fixed_60_signal").is_not_null() & pl.col("time_20d_signal").is_not_null()
    )
    comparison_summary = {
        "window_type": "PAIRWISE_SIGNAL_DIFFERENCE",
        "factor_id": "funding_fade",
        "direction": 1,
        "horizon_bars": 72,
        "universe": "top20",
        "period": "full",
        "n_periods": paired.height,
        "n_symbol_obs": paired.height,
        "ic_mean": None,
        "naive_tstat": None,
        "non_overlap_count": None,
        "non_overlap_ic_mean": None,
        "non_overlap_tstat": None,
        "hac_tstat": None,
        "hac_lag": None,
        "inflation_ratio": None,
        "paired_signal_correlation": float(
            paired.select(pl.corr("fixed_60_signal", "time_20d_signal")).item()
        )
        if paired.height
        else None,
        "mean_absolute_signal_difference": float(
            paired.select((pl.col("fixed_60_signal") - pl.col("time_20d_signal")).abs().mean()).item()
        )
        if paired.height
        else None,
    }
    frame = pl.concat(
        [pl.DataFrame(evaluations), pl.DataFrame([comparison_summary])], how="diagonal_relaxed"
    )
    frame.write_csv(root / "artifacts/funding_window_comparison.csv")
    return evaluations[1], series


def _daily_permutation_panel(market) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = market.signals["core.low_vol_480"]
    signal_rows: list[np.ndarray] = []
    return_rows: list[np.ndarray] = []
    symbol_rows: list[np.ndarray] = []
    for decision_index in range(1, len(market.timestamps) - 24):
        timestamp = market.timestamps[decision_index]
        if timestamp.hour != 0:
            continue
        eligible = market.universe_members["top20"].get(timestamp.date())
        if eligible is None or eligible.size != 20:
            continue
        values = signal[decision_index - 1, eligible]
        future = np.prod(1.0 + market.returns[decision_index + 1 : decision_index + 25, eligible], axis=0) - 1.0
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(future)):
            continue
        signal_rows.append(values.astype(float))
        return_rows.append(future.astype(float))
        symbol_rows.append(eligible.astype(int))
    return np.vstack(signal_rows), np.vstack(return_rows), np.vstack(symbol_rows)


def _rank_rows(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values, axis=1, kind="mergesort"), axis=1, kind="mergesort").astype(float) + 1.0


def _row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    return np.sum(left_centered * right_centered, axis=1) / denominator


def _permuted_portfolio_returns(
    scores: np.ndarray, realized: np.ndarray, symbols: np.ndarray
) -> np.ndarray:
    output = np.zeros(scores.shape[0], dtype=float)
    prior: dict[int, float] = {}
    for index in range(scores.shape[0]):
        weights = _capped_weights(scores[index], 3, 0.50, "score")
        current = {
            int(symbol): float(weight)
            for symbol, weight in zip(symbols[index], weights, strict=True)
            if weight > 0
        }
        union = set(prior) | set(current)
        traded_fraction = sum(abs(current.get(key, 0.0) - prior.get(key, 0.0)) for key in union)
        gross = float(np.dot(weights, realized[index]))
        output[index] = gross - traded_fraction * 15.0 / 10_000.0
        prior = current
    return output


def _permutations(root: Path, market) -> tuple[pl.DataFrame, dict]:
    signals, realized, symbol_ids = _daily_permutation_panel(market)
    signal_ranks = _rank_rows(signals)
    return_ranks = _rank_rows(realized)
    observed_ics = _row_correlations(signal_ranks, return_ranks)
    observed_net = _permuted_portfolio_returns(signal_ranks, realized, symbol_ids)
    observed = {
        "ic": float(observed_ics.mean()),
        "ic_t_stat": plain_tstat(observed_ics),
        "long_only_net_return": compounded_return(observed_net),
        "long_only_t_stat": plain_tstat(observed_net),
        "independent_daily_samples": int(len(observed_ics)),
    }
    rows: list[dict] = []
    controls = (
        "time_within_factor_permutation",
        "cross_section_rank_permutation",
        "label_permutation",
        "independent_random_noise",
    )
    for control_index, control in enumerate(controls):
        for permutation in range(PERMUTATION_COUNT):
            seed = 20260718 + control_index * 1_000_000 + permutation
            rng = np.random.default_rng(seed)
            permuted_returns = realized
            if control == "time_within_factor_permutation":
                permuted_scores = signal_ranks[rng.permutation(signal_ranks.shape[0])]
            elif control == "cross_section_rank_permutation":
                order = np.argsort(rng.random(signal_ranks.shape), axis=1)
                permuted_scores = np.take_along_axis(signal_ranks, order, axis=1)
            elif control == "label_permutation":
                order = np.argsort(rng.random(return_ranks.shape), axis=1)
                permuted_scores = signal_ranks
                permuted_returns = np.take_along_axis(realized, order, axis=1)
            else:
                permuted_scores = _rank_rows(rng.normal(size=signal_ranks.shape))
            permuted_return_ranks = _rank_rows(permuted_returns)
            ics = _row_correlations(permuted_scores, permuted_return_ranks)
            net = _permuted_portfolio_returns(permuted_scores, permuted_returns, symbol_ids)
            ic_t = plain_tstat(ics)
            net_t = plain_tstat(net)
            rows.append(
                {
                    "control_type": control,
                    "permutation_index": permutation,
                    "seed": seed,
                    "ic": float(ics.mean()),
                    "ic_t_stat": ic_t,
                    "long_only_net_return": compounded_return(net),
                    "long_only_t_stat": net_t,
                    "max_absolute_statistic": max(abs(ic_t), abs(net_t)),
                }
            )
        print(f"completed {PERMUTATION_COUNT} permutations: {control}")
    distribution = pl.DataFrame(rows)
    distribution.write_parquet(root / "artifacts/permutation_distribution.parquet", compression="zstd")
    summary_rows: list[dict] = []
    for control in controls:
        subset = distribution.filter(pl.col("control_type") == control)
        ic_p = empirical_pvalues(
            observed["ic"],
            subset["ic"].to_numpy(),
            subset["max_absolute_statistic"].to_numpy(),
        )
        net_p = empirical_pvalues(
            observed["long_only_net_return"],
            subset["long_only_net_return"].to_numpy(),
            subset["max_absolute_statistic"].to_numpy(),
        )
        max_adjusted = (
            1.0
            + float(
                np.sum(
                    subset["max_absolute_statistic"].to_numpy()
                    >= max(abs(observed["ic_t_stat"]), abs(observed["long_only_t_stat"]))
                )
            )
        ) / (subset.height + 1.0)
        summary_rows.append(
            {
                "control_type": control,
                "control_class": "NULL_CONTROL",
                "permutation_count": subset.height,
                "observed_ic": observed["ic"],
                "observed_ic_t_stat": observed["ic_t_stat"],
                "ic_empirical_p_value": ic_p["empirical_p_value"],
                "ic_two_sided_empirical_p_value": ic_p["two_sided_empirical_p_value"],
                "observed_long_only_net_return": observed["long_only_net_return"],
                "net_return_empirical_p_value": net_p["empirical_p_value"],
                "net_return_two_sided_empirical_p_value": net_p[
                    "two_sided_empirical_p_value"
                ],
                "max_stat_adjusted_p_value": max_adjusted,
                "ic_null_95_low": ic_p["null_95_low"],
                "ic_null_95_high": ic_p["null_95_high"],
                "ic_null_99_low": ic_p["null_99_low"],
                "ic_null_99_high": ic_p["null_99_high"],
            }
        )
    summary = pl.DataFrame(summary_rows)
    summary.write_csv(root / "artifacts/null_control_summary.csv")
    return summary, observed


def _statistical_sensitivity(
    root: Path,
    eval_series: dict[str, np.ndarray],
    funding_series: np.ndarray,
    results24: pl.DataFrame,
    simulations: dict[str, tuple[object, dict]],
    locked: pl.DataFrame,
    v1: Path,
) -> None:
    rows: list[dict] = []
    for object_id, values in eval_series.items():
        for row in hac_bandwidth_sensitivity(values, 24):
            rows.append(
                {
                    "object_id": object_id,
                    "object_type": "signal_ic",
                    "observation_frequency": "hourly_decision",
                    "horizon_hours": 24,
                    **row,
                }
            )
    if funding_series.size:
        for row in hac_bandwidth_sensitivity(funding_series, 72):
            rows.append(
                {
                    "object_id": "funding_fade.time_based_20d.rank_ic",
                    "object_type": "signal_ic",
                    "observation_frequency": "hourly_decision",
                    "horizon_hours": 72,
                    **row,
                }
            )
    for structure_id, (_, series) in simulations.items():
        daily = np.asarray(series["daily_net"], dtype=float)
        for row in hac_bandwidth_sensitivity(daily, 1):
            rows.append(
                {
                    "object_id": structure_id,
                    "object_type": "portfolio_net_return",
                    "observation_frequency": "daily",
                    "horizon_hours": 24,
                    **row,
                }
            )
    # The original locked 120h no-filter portfolio is confirmatory.  Its v1
    # daily series is not persisted per-period, so its published daily HAC
    # result remains the primary point and is accompanied by all v2 rules in
    # the locked portfolio table/report.
    hac = pl.DataFrame(rows)
    hac.write_csv(root / "artifacts/hac_bandwidth_sensitivity.csv")

    robustness = pl.read_csv(v1 / "artifacts/negative_control_results.csv").filter(
        pl.col("control_type").is_in(["wrong_lag_24h", "random_universe", "reverse_factor"])
    ).select(
        "control_type",
        "factor_id",
        "horizon_bars",
        "universe",
        "n_periods",
        "ic_mean",
        "hac_tstat",
    ).with_columns(
        pl.lit("ROBUSTNESS_PERTURBATION_NOT_NULL_CONTROL").alias("classification"),
        pl.lit("inherited from v1; not used as a strict null").alias("interpretation"),
    )
    low_sensitivity = pl.read_csv(root / "artifacts/low_vol_signal_sensitivity.csv").filter(
        pl.col("slice_type").is_in(["coin_exclusion", "v1_locked_grid_reuse"])
    ).select(
        pl.col("slice_type").alias("control_type"),
        "factor_id",
        pl.lit(None).cast(pl.Int64).alias("horizon_bars"),
        pl.col("slice").alias("universe"),
        pl.col("n_observations").alias("n_periods"),
        pl.col("estimate").alias("ic_mean"),
        "hac_tstat",
    ).with_columns(
        pl.lit("ROBUSTNESS_PERTURBATION_NOT_NULL_CONTROL").alias("classification"),
        pl.lit("v2 locked sensitivity; not used as a strict null").alias("interpretation"),
    )
    pl.concat([robustness, low_sensitivity], how="diagonal_relaxed").write_csv(
        root / "artifacts/robustness_perturbation_summary.csv"
    )


def _multiple_testing(
    root: Path,
    results24: pl.DataFrame,
    locked: pl.DataFrame,
    funding_eval: dict,
) -> None:
    tests: list[dict] = []
    for row in results24.filter(pl.col("period") == "full").iter_rows(named=True):
        tests.append(
            {
                "test_id": f"24h|{row['structure_id']}",
                "test_type": "confirmatory",
                "test_family": "24h_portfolio",
                "object": "low_vol_20d" if row["factor_id"] == "core.low_vol_480" else "rev_xs_20d",
                "raw_pvalue": row["raw_pvalue"],
                "observed_statistic": row["daily_hac_tstat"],
                "post_hoc": False,
            }
        )
    lock_row = locked.filter(
        (pl.col("object") == "low_vol_locked") & (pl.col("period") == "full")
    ).to_dicts()[0]
    tests.append(
        {
            "test_id": "locked|low_vol|top3|score|120h|btc_filter=0",
            "test_type": "confirmatory",
            "test_family": "locked_portfolio",
            "object": "low_vol_20d",
            "raw_pvalue": lock_row["raw_pvalue"],
            "observed_statistic": lock_row["daily_hac_tstat"],
            "post_hoc": False,
        }
    )
    tests.append(
        {
            "test_id": "funding|time_based_20d|72h|top20",
            "test_type": "confirmatory",
            "test_family": "funding_window",
            "object": "funding_fade",
            "raw_pvalue": 1.0
            if funding_eval["n_periods"] < 30
            else float(
                2.0
                * __import__("scipy").stats.t.sf(
                    abs(float(funding_eval["hac_tstat"])),
                    df=max(int(funding_eval["n_periods"]) - 1, 1),
                )
            ),
            "observed_statistic": funding_eval["hac_tstat"],
            "post_hoc": False,
        }
    )
    tests.append(
        {
            "test_id": "forward|low_vol_btc_trend|frozen",
            "test_type": "confirmatory",
            "test_family": "forward_paper",
            "object": "low_vol_btc_trend",
            "raw_pvalue": 1.0,
            "observed_statistic": 0.0,
            "post_hoc": False,
        }
    )
    posthoc = locked.filter(
        (pl.col("object") == "low_vol_btc_trend") & (pl.col("period") == "full")
    ).to_dicts()[0]
    tests.append(
        {
            "test_id": "posthoc|low_vol|top3|score|120h|btc_filter=1",
            "test_type": "exploratory",
            "test_family": "v1_grid_post_hoc",
            "object": "low_vol_btc_trend",
            "raw_pvalue": posthoc["raw_pvalue"],
            "observed_statistic": posthoc["daily_hac_tstat"],
            "post_hoc": True,
        }
    )
    frame = pl.DataFrame(tests)
    output: list[pl.DataFrame] = []
    null_distribution = pl.read_parquet(
        root / "artifacts/permutation_distribution.parquet"
    )
    null_by_control = [
        null_distribution.filter(pl.col("control_type") == control)[
            "max_absolute_statistic"
        ].to_numpy()
        for control in null_distribution["control_type"].unique().sort()
    ]
    for test_type in ("confirmatory", "exploratory"):
        subset = frame.filter(pl.col("test_type") == test_type)
        holm, bh = adjust_pvalues(subset["raw_pvalue"].to_numpy())
        max_stat_pvalues = [
            max(
                max_stat_adjusted_pvalue(float(statistic), null_values)
                for null_values in null_by_control
            )
            for statistic in subset["observed_statistic"]
        ]
        output.append(
            subset.with_columns(
                pl.Series("holm_pvalue", holm),
                pl.Series("bh_fdr_pvalue", bh),
                pl.Series("max_stat_permutation_pvalue", max_stat_pvalues),
                pl.lit(subset.height).alias("tests_in_type"),
            )
        )
    pl.concat(output, how="vertical_relaxed").write_csv(
        root / "artifacts/multiple_testing_v2.csv"
    )


def main() -> None:
    root, v1, v1_root = _paths()
    bars = pl.read_parquet(v1_root / "data/silver/bars_1h.parquet")
    funding = pl.read_parquet(v1_root / "data/silver/funding.parquet")
    universes = pl.read_parquet(v1_root / "data/gold/dynamic_universe.parquet")
    signals = {
        factor_id: pl.read_parquet(v1_root / "data/gold/signals" / filename)
        for factor_id, filename in FACTORS.items()
    }
    lock = json.loads((v1 / "artifacts/oos_lock.json").read_text())
    market = build_market_matrix(bars, signals, universes)
    periods = _periods(lock, market.timestamps)
    results24, simulations = _run_24h(
        root, v1, market, lock["factors"], periods
    )
    print(f"24h portfolio rows={results24.height}")
    locked = _locked_low_vol_portfolios(root, v1, market, periods)
    primary, eval_series, low_panel = _signal_analysis(root, v1, bars, universes, signals)
    funding_eval, funding_series = _funding_analysis(root, bars, universes, funding)
    null_summary, observed = _permutations(root, market)
    _statistical_sensitivity(
        root, eval_series, funding_series, results24, simulations, locked, v1
    )
    _multiple_testing(root, results24, locked, funding_eval)

    low_bootstrap = []
    for block_hours, block_observations in zip(BLOCK_HOURS, BLOCK_HOURS, strict=True):
        row = moving_block_bootstrap(
            eval_series["core.low_vol_480.rank_ic"],
            block_sizes=[block_observations],
            statistic=lambda values: float(np.mean(values)),
            n_bootstrap=BOOTSTRAP_COUNT,
            seed=20260718 + block_hours,
        )[0]
        low_bootstrap.append(
            {
                "object_id": "core.low_vol_480.rank_ic.24h.top20",
                "object_type": "signal_ic",
                "metric": "mean_rank_ic",
                "block_size_hours": block_hours,
                "bootstrap_count": BOOTSTRAP_COUNT,
                "estimate": row["estimate"],
                "ci_95_low": row["ci_95_low"],
                "ci_95_high": row["ci_95_high"],
                "median": row["median"],
                "probability_net_return_gt_zero": None,
                "probability_sharpe_gt_zero": None,
                "probability_ic_gt_zero": row["probability_greater_than_zero"],
            }
        )
    bootstrap_path = root / "artifacts/block_bootstrap_summary.csv"
    pl.concat(
        [pl.read_csv(bootstrap_path), pl.DataFrame(low_bootstrap)], how="diagonal_relaxed"
    ).write_csv(bootstrap_path)

    analysis_summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "audit_v1_data_cutoff": V1_CUTOFF.isoformat(),
        "base_cost_one_way_bps": 15.0,
        "roundtrip_cost_floor_bps": 30.0,
        "portfolio_24h_rows": results24.height,
        "permutations_per_null": PERMUTATION_COUNT,
        "bootstrap_count": BOOTSTRAP_COUNT,
        "primary_signal_metrics": primary,
        "funding_time_based_evaluation": funding_eval,
        "permutation_observed": observed,
        "production_mutations": 0,
        "live_orders": 0,
    }
    (root / "artifacts/analysis_summary_v2.json").write_text(
        json.dumps(analysis_summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
