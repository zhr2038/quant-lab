"""Recompute only the Audit v2.1 funding, null and concentration evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.contribution_v21 import (  # noqa: E402
    forward_symbol_contributions,
    partition_symbol_contributions,
)
from audit.auditlib.evaluate import build_labels, evaluate_factor  # noqa: E402
from audit.auditlib.factors import funding_fade  # noqa: E402
from audit.auditlib.funding_v2 import (  # noqa: E402
    funding_fade_time_20d,
    funding_frequency_summary,
)
from audit.auditlib.ic_stats import plain_tstat  # noqa: E402
from audit.auditlib.permutation_v21 import (  # noqa: E402
    classify_control,
    within_symbol_circular_shift,
    within_symbol_signal_permutation,
)
from audit.auditlib.portfolio_backtest import (  # noqa: E402
    build_market_matrix,
    simulate_long_only,
)
from audit.auditlib.statistics_v2 import (  # noqa: E402
    compounded_return,
    empirical_pvalues,
)
from audit.scripts.stage_v2_analysis import (  # noqa: E402
    FACTORS,
    MAX_WEIGHT,
    _daily_permutation_panel,
    _periods,
    _permuted_portfolio_returns,
    _rank_rows,
    _row_correlations,
    _with_universe,
)

PERMUTATION_COUNT = 1000
NEW_STRICT_NULLS = (
    "within_symbol_signal_permutation",
    "within_symbol_circular_shift",
)
REUSED_STRICT_NULLS = (
    "cross_section_rank_permutation",
    "label_permutation",
    "independent_random_noise",
)


def _funding(
    root: Path,
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    universes: pl.DataFrame,
) -> dict:
    frequency = funding_frequency_summary(funding)
    frequency.write_csv(root / "artifacts/funding_frequency_summary_v21.csv")
    grid = bars.select(["symbol", "ts"])
    fixed = funding_fade(grid, funding).rename({"signal": "fixed_60_signal"})
    time_based = funding_fade_time_20d(grid, funding).rename(
        {"signal": "time_20d_signal"}
    )
    joined = fixed.join(
        time_based, on=["symbol", "feature_ts"], how="full", coalesce=True
    )
    labels = _with_universe(build_labels(bars, 72), universes, "top20")
    evaluations: list[dict] = []
    for name, column in (
        ("FIXED_60_OBSERVATIONS", "fixed_60_signal"),
        ("TIME_BASED_20D_STRICT_EVENT_AVAILABILITY", "time_20d_signal"),
    ):
        signal = joined.select(
            ["symbol", "feature_ts", pl.col(column).alias("signal")]
        ).drop_nulls("signal")
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
        evaluations.append(
            {
                "window_type": name,
                "availability_rule": (
                    "funding_event_ts < decision_ts"
                    if name.startswith("TIME_BASED")
                    else "legacy_fixed_observation_count"
                ),
                **result.flat_dict(),
            }
        )
    paired = joined.drop_nulls(["fixed_60_signal", "time_20d_signal"])
    comparison = {
        "window_type": "PAIRWISE_SIGNAL_DIFFERENCE",
        "availability_rule": "same symbol and decision timestamp",
        "n_symbol_obs": paired.height,
        "paired_signal_correlation": (
            float(paired.select(pl.corr("fixed_60_signal", "time_20d_signal")).item())
            if paired.height
            else None
        ),
        "mean_absolute_signal_difference": (
            float(
                paired.select(
                    (pl.col("fixed_60_signal") - pl.col("time_20d_signal"))
                    .abs()
                    .mean()
                ).item()
            )
            if paired.height
            else None
        ),
    }
    pl.concat(
        [pl.DataFrame(evaluations), pl.DataFrame([comparison])],
        how="diagonal_relaxed",
    ).write_csv(root / "artifacts/funding_window_comparison_v21.csv")
    signal_results = pl.DataFrame(evaluations).with_columns(
        pl.lit("INCONCLUSIVE").alias("signal_validity"),
        pl.lit("INCONCLUSIVE").alias("portfolio_validity"),
        pl.lit("INCONCLUSIVE").alias("deployment_readiness"),
        pl.lit("historical coverage remains below preregistered sufficiency").alias(
            "decision_reason"
        ),
    )
    signal_results.write_csv(root / "artifacts/funding_signal_results_v21.csv")
    return signal_results.filter(pl.col("window_type").str.starts_with("TIME_BASED")).to_dicts()[0]


def _null_summary(distribution: pl.DataFrame, observed: dict) -> pl.DataFrame:
    rows: list[dict] = []
    for control in distribution["control_type"].unique().sort().to_list():
        subset = distribution.filter(pl.col("control_type") == control)
        ic_p = empirical_pvalues(
            observed["ic"],
            subset["ic"].to_numpy(),
            subset["paired_metric_max_statistic"].to_numpy(),
        )
        net_p = empirical_pvalues(
            observed["long_only_net_return"],
            subset["long_only_net_return"].to_numpy(),
            subset["paired_metric_max_statistic"].to_numpy(),
        )
        threshold = max(
            abs(float(observed["ic_t_stat"])),
            abs(float(observed["long_only_t_stat"])),
        )
        paired_p = (1.0 + float(
            np.sum(subset["paired_metric_max_statistic"].to_numpy() >= threshold)
        )) / (subset.height + 1.0)
        rows.append(
            {
                "control_type": control,
                "control_class": classify_control(control),
                "permutation_count": subset.height,
                "observed_ic": observed["ic"],
                "observed_ic_t_stat": observed["ic_t_stat"],
                "ic_empirical_p_value": ic_p["empirical_p_value"],
                "ic_two_sided_empirical_p_value": ic_p[
                    "two_sided_empirical_p_value"
                ],
                "observed_long_only_net_return": observed["long_only_net_return"],
                "net_return_empirical_p_value": net_p["empirical_p_value"],
                "net_return_two_sided_empirical_p_value": net_p[
                    "two_sided_empirical_p_value"
                ],
                "paired_metric_max_stat_p_value": paired_p,
                "paired_metric_scope": "IC_T_STAT_AND_PORTFOLIO_T_STAT_ONLY",
                "ic_null_95_low": ic_p["null_95_low"],
                "ic_null_95_high": ic_p["null_95_high"],
                "ic_null_99_low": ic_p["null_99_low"],
                "ic_null_99_high": ic_p["null_99_high"],
            }
        )
    return pl.DataFrame(rows)


def _permutations(root: Path, v2_dir: Path, market) -> tuple[pl.DataFrame, dict]:
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
        "independent_daily_samples": len(observed_ics),
    }
    legacy = pl.read_parquet(v2_dir / "artifacts/permutation_distribution.parquet")
    reused = (
        legacy.filter(pl.col("control_type").is_in(REUSED_STRICT_NULLS))
        .rename({"max_absolute_statistic": "paired_metric_max_statistic"})
        .with_columns(
            pl.lit("REUSED_FROM_IMMUTABLE_V2").alias("v21_execution"),
            pl.lit("IC_T_STAT_AND_PORTFOLIO_T_STAT_ONLY").alias(
                "paired_metric_scope"
            ),
        )
    )
    rows: list[dict] = []
    for control_index, control in enumerate(NEW_STRICT_NULLS):
        for permutation in range(PERMUTATION_COUNT):
            seed = 20260719 + control_index * 1_000_000 + permutation
            rng = np.random.default_rng(seed)
            if control == "within_symbol_signal_permutation":
                null_scores = within_symbol_signal_permutation(
                    signals, symbol_ids, rng
                )
            else:
                null_scores = within_symbol_circular_shift(
                    signals,
                    symbol_ids,
                    rng,
                    minimum_shift_observations=25,
                )
            ranked = _rank_rows(null_scores)
            ics = _row_correlations(ranked, return_ranks)
            net = _permuted_portfolio_returns(ranked, realized, symbol_ids)
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
                    "paired_metric_max_statistic": max(abs(ic_t), abs(net_t)),
                    "v21_execution": "NEW_V21_1000_PERMUTATIONS",
                    "paired_metric_scope": "IC_T_STAT_AND_PORTFOLIO_T_STAT_ONLY",
                }
            )
        print(f"completed {PERMUTATION_COUNT} v2.1 permutations: {control}")
    distribution = pl.concat(
        [reused, pl.DataFrame(rows)], how="diagonal_relaxed"
    ).sort(["control_type", "permutation_index"])
    distribution.write_parquet(
        root / "artifacts/permutation_distribution_v21.parquet", compression="zstd"
    )
    summary = _null_summary(distribution, observed)
    summary.write_csv(root / "artifacts/null_control_summary_v21.csv")

    old_robustness = pl.read_csv(
        v2_dir / "artifacts/robustness_perturbation_summary.csv"
    )
    time_within = legacy.filter(
        pl.col("control_type") == "time_within_factor_permutation"
    )
    time_row = {
        "control_type": "time_within_factor_permutation",
        "factor_id": "core.low_vol_480",
        "horizon_bars": 24,
        "universe": "top20",
        "n_periods": time_within.height,
        "ic_mean": float(time_within["ic"].mean()),
        "hac_tstat": float(time_within["ic_t_stat"].mean()),
        "classification": "ROBUSTNESS_PERTURBATION_NOT_NULL_CONTROL",
        "interpretation": (
            "time shuffling can preserve changing universe/composition effects; "
            "it is not a strict null"
        ),
    }
    robustness = pl.concat(
        [old_robustness, pl.DataFrame([time_row])], how="diagonal_relaxed"
    ).unique(subset=["control_type", "factor_id", "universe"], keep="last")
    robustness.write_csv(root / "artifacts/robustness_perturbation_summary_v21.csv")
    return summary, observed


def _contributions(
    root: Path,
    v2_dir: Path,
    v1_dir: Path,
    market,
    periods: dict[str, tuple[int, int]],
) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for factor_id in FACTORS:
        universe = "top20" if factor_id == "core.low_vol_480" else "top50"
        for top_n in (1, 3, 5):
            for weighting in ("equal", "score"):
                for btc_filter in (False, True):
                    structure_id = (
                        f"{factor_id}|top{top_n}|{weighting}|24h|"
                        f"staggered=0|btc_filter={int(btc_filter)}"
                    )
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
                    frame = partition_symbol_contributions(
                        simulation,
                        market,
                        periods,
                        structure_id=structure_id,
                        factor_id=factor_id,
                        one_way_cost_bps=15.0,
                    )
                    if not frame.is_empty():
                        rows.append(frame)
    trades_path = root / "artifacts/forward_v21_trades.parquet"
    trades = pl.read_parquet(trades_path) if trades_path.exists() else pl.DataFrame()
    rows.append(
        forward_symbol_contributions(
            trades,
            structure_id="core.low_vol_480|top3|score|120h|btc_filter=1",
            factor_id="core.low_vol_480",
        )
    )
    result = pl.concat(rows, how="diagonal_relaxed").sort(
        ["factor_id", "structure_id", "partition", "absolute_contribution_share"],
        descending=[False, False, False, True],
    )
    result.write_csv(root / "artifacts/symbol_contribution_by_partition_v21.csv")

    summary = result.group_by(["structure_id", "partition"]).agg(
        pl.col("top_symbol_share").max().alias("max_symbol_contribution_share_v21"),
        pl.col("top_symbol").first().alias("top_symbol_v21"),
        pl.col("top_3_symbol_share").max().alias("top_3_symbol_share_v21"),
        pl.col("hhi").max().alias("contribution_hhi_v21"),
    )
    old = pl.read_csv(v2_dir / "artifacts/portfolio_24h_results.csv")
    corrected = old.join(
        summary,
        left_on=["structure_id", "period"],
        right_on=["structure_id", "partition"],
        how="left",
    ).rename(
        {"max_symbol_contribution_share": "invalid_v2_full_sample_concentration"}
    )
    corrected.write_csv(root / "artifacts/portfolio_24h_results_v21.csv")
    return result


def _multiple_testing(root: Path, v2_dir: Path) -> None:
    old = pl.read_csv(v2_dir / "artifacts/multiple_testing_v2.csv")
    updated = old.rename(
        {"max_stat_permutation_pvalue": "paired_metric_max_stat_pvalue"}
    ).with_columns(
        pl.lit("HOLM_BONFERRONI").alias("formal_familywise_method"),
        pl.lit("IC_T_STAT_AND_PORTFOLIO_T_STAT_ONLY").alias(
            "paired_metric_scope"
        ),
        pl.lit(False).alias("paired_metric_is_full_familywise_max_stat"),
    )
    updated.write_csv(root / "artifacts/multiple_testing_v21.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--v1-dir", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    v2_dir = args.v2_dir.resolve()
    v1_dir = args.v1_dir.resolve()
    v1_root = args.v1_root.resolve()
    bars = pl.read_parquet(v1_root / "data/silver/bars_1h.parquet")
    funding = pl.read_parquet(v1_root / "data/silver/funding.parquet")
    universes = pl.read_parquet(v1_root / "data/gold/dynamic_universe.parquet")
    signals = {
        factor_id: pl.read_parquet(v1_root / "data/gold/signals" / filename)
        for factor_id, filename in FACTORS.items()
    }
    lock = json.loads((v1_dir / "artifacts/oos_lock.json").read_text())
    market = build_market_matrix(bars, signals, universes)
    periods = _periods(lock, market.timestamps)
    funding_result = _funding(root, bars, funding, universes)
    null_summary, observed = _permutations(root, v2_dir, market)
    contributions = _contributions(root, v2_dir, v1_dir, market, periods)
    _multiple_testing(root, v2_dir)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": [
            "funding_event_availability",
            "new_strict_nulls",
            "partition_symbol_contribution",
            "paired_metric_max_stat_rename",
        ],
        "new_permutations_per_control": PERMUTATION_COUNT,
        "null_control_count": null_summary.height,
        "contribution_rows": contributions.height,
        "funding_time_based_n_periods": funding_result.get("n_periods"),
        "observed": observed,
        "factor_searches": 0,
        "parameter_searches": 0,
        "production_mutations": 0,
        "live_orders": 0,
    }
    (root / "artifacts/analysis_summary_v21.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
