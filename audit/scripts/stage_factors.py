# ruff: noqa: E501
"""Run the predeclared IC, OOS, regime, negative-control, and multiplicity audit."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.evaluate import EvalResult, build_labels, evaluate_factor  # noqa: E402
from audit.auditlib.leakage import permute_labels, shift_signal_values  # noqa: E402
from audit.auditlib.multiple_testing import adjust_pvalues, two_sided_pvalue  # noqa: E402
from audit.auditlib.paths import ARTIFACTS, DATA_GOLD, DATA_SILVER, REPORTS  # noqa: E402

FACTOR_FILES = {
    "v5.alpha6_static_proxy": "v5__alpha6_static_proxy.parquet",
    "core.rev_xs_480": "core__rev_xs_480.parquet",
    "core.low_vol_480": "core__low_vol_480.parquet",
    "core.funding_fade_480": "core__funding_fade_480.parquet",
}
FACTOR_NAMES = {
    "v5.alpha6_static_proxy": "current_v5_alpha6_static_proxy",
    "core.rev_xs_480": "rev_xs_20d",
    "core.low_vol_480": "low_vol_20d",
    "core.funding_fade_480": "funding_fade",
}
HORIZONS = [24, 72, 120, 480]
UNIVERSE_NAMES = ["top10", "top20", "top50"]
PRIMARY_DIRECTION = 1
MIN_CROSS = 5
NEGATIVE_CONTROL_SEEDS = [7, 19, 42, 20260718]


def _row(
    result: EvalResult, *, test_family: str = "factor_ic", slice_type: str = "primary"
) -> dict:
    row = result.flat_dict()
    row["factor_name"] = FACTOR_NAMES.get(result.factor_id, result.factor_id)
    row["test_family"] = test_family
    row["slice_type"] = slice_type
    row["test_id"] = "|".join(
        [
            test_family,
            result.factor_id,
            str(result.direction),
            str(result.horizon_bars),
            result.universe,
            result.period,
        ]
    )
    row["raw_pvalue"] = two_sided_pvalue(result.hac_tstat, max(result.n_periods - 1, 1))
    return row


def _period_filter(labels: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    # Purge any sample whose label exits outside its declared OOS slice.
    return labels.filter(
        (pl.col("feature_ts") >= pl.lit(start))
        & (pl.col("feature_ts") < pl.lit(end))
        & (pl.col("label_ts") < pl.lit(end))
    )


def _with_universe(labels: pl.DataFrame, universes: pl.DataFrame, name: str) -> pl.DataFrame:
    members = universes.filter(pl.col("universe") == name).select(["date", "symbol"])
    return (
        labels.with_columns(pl.col("decision_ts").dt.date().alias("date"))
        .join(members, on=["date", "symbol"], how="inner")
        .drop("date")
    )


def _evaluate(
    signals: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    factor_id: str,
    direction: int,
    horizon: int,
    universe: str,
    period: str,
) -> dict:
    return _row(
        evaluate_factor(
            signals,
            labels,
            factor_id=factor_id,
            direction=direction,
            horizon_bars=horizon,
            universe_name=universe,
            period_name=period,
            min_cross=MIN_CROSS,
        )
    )


def _btc_regimes(bars: pl.DataFrame) -> pl.DataFrame:
    btc = (
        bars.filter(pl.col("symbol") == "BTC-USDT")
        .sort("ts")
        .with_columns(pl.col("close").pct_change().alias("_ret"))
        .with_columns(
            (pl.col("close") / pl.col("close").shift(24 * 60) - 1.0).alias("_ret_60d"),
            pl.col("close").rolling_mean(24 * 60).alias("_ma_60d"),
            pl.col("_ret").rolling_std(24 * 30, ddof=0).alias("_vol_30d"),
        )
        .with_columns(pl.col("_vol_30d").rolling_median(24 * 180).alias("_vol_reference"))
        .with_columns(
            pl.when(pl.col("close") >= pl.col("_ma_60d"))
            .then(pl.lit("btc_trend_up"))
            .otherwise(pl.lit("btc_trend_down"))
            .alias("btc_trend"),
            pl.when(pl.col("_vol_30d") >= pl.col("_vol_reference"))
            .then(pl.lit("high_vol"))
            .otherwise(pl.lit("low_vol"))
            .alias("volatility_regime"),
            pl.when(pl.col("_ret_60d") > 0.10)
            .then(pl.lit("bull"))
            .when(pl.col("_ret_60d") < -0.10)
            .then(pl.lit("bear"))
            .otherwise(pl.lit("sideways"))
            .alias("market_state"),
        )
        .select(
            [pl.col("ts").alias("feature_ts"), "btc_trend", "volatility_regime", "market_state"]
        )
    )
    return btc


def _random_universe(universes: pl.DataFrame, seed: int, top_n: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    top50 = universes.filter(pl.col("universe") == "top50").select(["date", "symbol"])
    for date_key, frame in top50.group_by("date", maintain_order=True):
        date_value = date_key[0] if isinstance(date_key, tuple) else date_key
        symbols = frame["symbol"].to_list()
        chosen = rng.choice(symbols, size=min(top_n, len(symbols)), replace=False)
        rows.extend({"date": date_value, "symbol": str(symbol)} for symbol in chosen)
    return pl.DataFrame(rows)


def main() -> None:
    bars = pl.read_parquet(DATA_SILVER / "bars_1h.parquet")
    universes = pl.read_parquet(DATA_GOLD / "dynamic_universe.parquet")
    signals_by_factor = {
        factor_id: pl.read_parquet(DATA_GOLD / "signals" / filename)
        for factor_id, filename in FACTOR_FILES.items()
    }
    data_start = bars["ts"].min()
    data_end = bars["ts"].max() + timedelta(hours=1)
    duration = data_end - data_start
    split_50 = data_start + duration / 2
    split_75 = data_start + duration * 0.75
    period_bounds = {
        "full": (data_start, data_end),
        "research": (data_start, split_50),
        "validation": (split_50, split_75),
        "blind": (split_75, data_end),
    }

    # Phase A: research-only grid. The lock is persisted before validation or
    # blind metrics are computed.
    research_rows: list[dict] = []
    labels_cache: dict[int, pl.DataFrame] = {}
    universe_labels: dict[tuple[int, str], pl.DataFrame] = {}
    for horizon in HORIZONS:
        labels_cache[horizon] = build_labels(bars, horizon, decision_delay_bars=1)
        for universe in UNIVERSE_NAMES:
            panel = _with_universe(labels_cache[horizon], universes, universe)
            universe_labels[(horizon, universe)] = panel
            research_panel = _period_filter(panel, *period_bounds["research"])
            for factor_id, signals in signals_by_factor.items():
                research_rows.append(
                    _evaluate(
                        signals,
                        research_panel,
                        factor_id=factor_id,
                        direction=PRIMARY_DIRECTION,
                        horizon=horizon,
                        universe=universe,
                        period="research",
                    )
                )

    locks: dict[str, dict] = {}
    research_frame = pl.DataFrame(research_rows)
    for factor_id in FACTOR_FILES:
        candidates = research_frame.filter(
            (pl.col("factor_id") == factor_id) & (pl.col("n_periods") > 0) & (pl.col("ic_mean") > 0)
        )
        if candidates.is_empty():
            candidates = research_frame.filter(
                (pl.col("factor_id") == factor_id) & (pl.col("n_periods") > 0)
            )
        if candidates.is_empty():
            locks[factor_id] = {
                "horizon_bars": 72,
                "universe": "top20",
                "reason": "no research samples; conservative default",
            }
            continue
        selected = (
            candidates.with_columns(
                pl.min_horizontal("hac_tstat", "non_overlap_tstat").alias("selection_score")
            )
            .sort(["selection_score", "ic_mean", "non_overlap_count"], descending=True)
            .head(1)
            .to_dicts()[0]
        )
        locks[factor_id] = {
            "horizon_bars": int(selected["horizon_bars"]),
            "universe": selected["universe"],
            "research_ic_mean": selected["ic_mean"],
            "research_hac_tstat": selected["hac_tstat"],
            "research_non_overlap_tstat": selected["non_overlap_tstat"],
            "selection_rule": "maximize min(HAC t, non-overlap t) among positive-IC research-only cells; all cells remain in multiplicity family",
        }
    lock_payload = {
        "locked_at": datetime.now(UTC).isoformat(),
        "blind_not_read_before_lock": True,
        "data_split": {key: [str(value[0]), str(value[1])] for key, value in period_bounds.items()},
        "factors": locks,
    }
    (ARTIFACTS / "oos_lock.json").write_text(
        json.dumps(lock_payload, indent=2) + "\n", encoding="utf-8"
    )
    print("OOS rules locked before validation/blind evaluation")

    # Phase B: complete declared grid. Reuse the Phase-A positive-direction
    # research cells so each test is counted exactly once.
    rows = list(research_rows)
    for horizon in HORIZONS:
        for universe in UNIVERSE_NAMES:
            panel = universe_labels[(horizon, universe)]
            for period in ("full", "validation", "blind"):
                sliced = _period_filter(panel, *period_bounds[period])
                for factor_id, signals in signals_by_factor.items():
                    rows.append(
                        _evaluate(
                            signals,
                            sliced,
                            factor_id=factor_id,
                            direction=PRIMARY_DIRECTION,
                            horizon=horizon,
                            universe=universe,
                            period=period,
                        )
                    )
            for period, bounds in period_bounds.items():
                sliced = _period_filter(panel, *bounds)
                for factor_id, signals in signals_by_factor.items():
                    rows.append(
                        _evaluate(
                            signals,
                            sliced,
                            factor_id=factor_id,
                            direction=-1,
                            horizon=horizon,
                            universe=universe,
                            period=period,
                        )
                    )
    primary = (
        pl.DataFrame(rows)
        .unique(subset=["test_id"], keep="first")
        .sort(["factor_id", "direction", "horizon_bars", "universe", "period"])
    )

    # Calendar, state, and causal walk-forward slices use only the locked cell.
    regimes = _btc_regimes(bars)
    supplemental_rows: list[dict] = []
    for factor_id, lock in locks.items():
        horizon = int(lock["horizon_bars"])
        universe = str(lock["universe"])
        base_panel = universe_labels[(horizon, universe)]
        signals = signals_by_factor[factor_id]

        # Annual and quarterly slices, purged at each boundary.
        cursor = datetime(data_start.year, 1, 1, tzinfo=UTC)
        while cursor < data_end:
            next_year = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
            start, end = max(cursor, data_start), min(next_year, data_end)
            if end > start:
                result = evaluate_factor(
                    signals,
                    _period_filter(base_panel, start, end),
                    factor_id=factor_id,
                    direction=1,
                    horizon_bars=horizon,
                    universe_name=universe,
                    period_name=f"year_{cursor.year}",
                    min_cross=MIN_CROSS,
                )
                supplemental_rows.append(_row(result, slice_type="calendar_year"))
            cursor = next_year

        cursor = datetime(data_start.year, ((data_start.month - 1) // 3) * 3 + 1, 1, tzinfo=UTC)
        while cursor < data_end:
            month = cursor.month + 3
            year = cursor.year + (month - 1) // 12
            next_quarter = datetime(year, (month - 1) % 12 + 1, 1, tzinfo=UTC)
            start, end = max(cursor, data_start), min(next_quarter, data_end)
            if end > start:
                quarter = (cursor.month - 1) // 3 + 1
                result = evaluate_factor(
                    signals,
                    _period_filter(base_panel, start, end),
                    factor_id=factor_id,
                    direction=1,
                    horizon_bars=horizon,
                    universe_name=universe,
                    period_name=f"quarter_{cursor.year}Q{quarter}",
                    min_cross=MIN_CROSS,
                )
                supplemental_rows.append(_row(result, slice_type="calendar_quarter"))
            cursor = next_quarter

        regime_panel = base_panel.join(regimes, on="feature_ts", how="left")
        for column, values in {
            "btc_trend": ["btc_trend_up", "btc_trend_down"],
            "volatility_regime": ["high_vol", "low_vol"],
            "market_state": ["bull", "bear", "sideways"],
        }.items():
            for value in values:
                result = evaluate_factor(
                    signals,
                    regime_panel.filter(pl.col(column) == value),
                    factor_id=factor_id,
                    direction=1,
                    horizon_bars=horizon,
                    universe_name=universe,
                    period_name=f"regime_{value}",
                    min_cross=MIN_CROSS,
                )
                supplemental_rows.append(_row(result, slice_type="regime"))

        # Four ordered test folds after the research half. Parameters are
        # frozen; expanding/rolling names record the training-window policy.
        fold_span = (data_end - split_50) / 4
        for fold in range(4):
            test_start = split_50 + fold_span * fold
            test_end = split_50 + fold_span * (fold + 1)
            test_panel = _period_filter(base_panel, test_start, test_end)
            for mode in ("expanding", "rolling"):
                result = evaluate_factor(
                    signals,
                    test_panel,
                    factor_id=factor_id,
                    direction=1,
                    horizon_bars=horizon,
                    universe_name=universe,
                    period_name=f"walk_{mode}_fold{fold + 1}",
                    min_cross=MIN_CROSS,
                )
                supplemental_rows.append(_row(result, slice_type=f"walk_{mode}"))

    supplemental = pl.DataFrame(supplemental_rows)
    all_factor_tests = pl.concat([primary, supplemental], how="diagonal_relaxed")

    # Fixed-seed negative controls at the predeclared 72h/top20 cell.
    control_rows: list[dict] = []
    control_horizon, control_universe = 72, "top20"
    control_labels = universe_labels[(control_horizon, control_universe)]
    for seed in NEGATIVE_CONTROL_SEEDS:
        rng = np.random.default_rng(seed)
        template = signals_by_factor["core.rev_xs_480"]
        for control_name, values in {
            "random_factor": rng.normal(size=template.height),
            "random_rank": rng.uniform(size=template.height),
        }.items():
            random_signal = template.select(["symbol", "feature_ts"]).with_columns(
                pl.Series("signal", values)
            )
            result = evaluate_factor(
                random_signal,
                control_labels,
                factor_id=f"negative.{control_name}.seed{seed}",
                direction=1,
                horizon_bars=control_horizon,
                universe_name=control_universe,
                period_name="full",
                min_cross=MIN_CROSS,
            )
            row = _row(result, test_family="negative_control", slice_type=control_name)
            row["control_type"] = control_name
            row["seed"] = seed
            control_rows.append(row)

    random_members = _random_universe(universes, seed=20260718)
    random_panel = (
        labels_cache[control_horizon]
        .with_columns(pl.col("decision_ts").dt.date().alias("date"))
        .join(random_members, on=["date", "symbol"], how="inner")
        .drop("date")
    )
    for factor_id, signals in signals_by_factor.items():
        manipulations = {
            "reverse_factor": signals,
            "wrong_lag_24h": shift_signal_values(signals, 24),
            "random_universe": signals,
        }
        for control_name, manipulated in manipulations.items():
            panel = random_panel if control_name == "random_universe" else control_labels
            direction = -1 if control_name == "reverse_factor" else 1
            result = evaluate_factor(
                manipulated,
                panel,
                factor_id=factor_id,
                direction=direction,
                horizon_bars=control_horizon,
                universe_name="random20" if control_name == "random_universe" else control_universe,
                period_name="full",
                min_cross=MIN_CROSS,
            )
            row = _row(result, test_family="negative_control", slice_type=control_name)
            row["control_type"] = control_name
            row["seed"] = 20260718
            control_rows.append(row)

        permuted = evaluate_factor(
            signals,
            permute_labels(control_labels, 20260718),
            factor_id=factor_id,
            direction=1,
            horizon_bars=control_horizon,
            universe_name=control_universe,
            period_name="full",
            min_cross=MIN_CROSS,
        )
        row = _row(permuted, test_family="negative_control", slice_type="label_permutation")
        row["control_type"] = "label_permutation"
        row["seed"] = 20260718
        control_rows.append(row)

    controls = pl.DataFrame(control_rows)

    # One multiplicity family contains every actual IC test, including
    # negative controls. No failed or inconvenient cell is removed.
    multiplicity = pl.concat(
        [
            all_factor_tests.select(
                [
                    "test_id",
                    "test_family",
                    "factor_id",
                    "direction",
                    "horizon_bars",
                    "universe",
                    "period",
                    "raw_pvalue",
                ]
            ),
            controls.select(
                [
                    "test_id",
                    "test_family",
                    "factor_id",
                    "direction",
                    "horizon_bars",
                    "universe",
                    "period",
                    "raw_pvalue",
                ]
            ),
        ],
        how="vertical_relaxed",
    ).unique(subset=["test_id"], keep="first")
    holm, bh = adjust_pvalues(multiplicity["raw_pvalue"].to_numpy())
    multiplicity = multiplicity.with_columns(
        pl.Series("holm_pvalue", holm),
        pl.Series("bh_fdr_pvalue", bh),
        pl.lit(multiplicity.height).alias("test_count"),
        pl.lit(2).alias("family_count"),
    )
    multiplicity.write_csv(ARTIFACTS / "multiple_testing_summary.csv")

    adjusted_lookup = multiplicity.select(
        ["test_id", "holm_pvalue", "bh_fdr_pvalue", "test_count", "family_count"]
    )
    all_factor_tests = all_factor_tests.join(adjusted_lookup, on="test_id", how="left")
    controls = controls.join(adjusted_lookup, on="test_id", how="left").with_columns(
        (
            (pl.col("ic_mean") > 0.03)
            & ((pl.col("hac_tstat") >= 2.0) | (pl.col("non_overlap_tstat") >= 2.0))
            & (pl.col("holm_pvalue") <= 0.05)
        ).alias("significance_screen_pass"),
        pl.lit(False).alias("paper_ready"),
        pl.lit("negative controls are never promotion candidates").alias("paper_ready_reason"),
    )
    controls.write_csv(ARTIFACTS / "negative_control_results.csv")

    primary_adjusted = all_factor_tests.filter(pl.col("slice_type") == "primary")
    primary_adjusted.write_csv(ARTIFACTS / "ic_statistics.csv")
    primary_adjusted.select(
        [
            "factor_id",
            "direction",
            "horizon_bars",
            "universe",
            "period",
            "n_periods",
            "non_overlap_count",
            "naive_tstat",
            "non_overlap_tstat",
            "hac_tstat",
            "hac_lag",
            "inflation_ratio",
            "raw_pvalue",
            "holm_pvalue",
            "bh_fdr_pvalue",
        ]
    ).write_csv(ARTIFACTS / "hac_statistics.csv")
    primary_adjusted.filter((pl.col("period") == "full") & (pl.col("direction") == 1)).write_csv(
        ARTIFACTS / "factor_horizon_results.csv"
    )
    primary_adjusted.filter((pl.col("period") == "full") & (pl.col("direction") == 1)).write_csv(
        ARTIFACTS / "factor_universe_results.csv"
    )
    all_factor_tests.filter(pl.col("slice_type") != "primary").write_csv(
        ARTIFACTS / "factor_period_results.csv"
    )

    # Gross top-3 contribution diagnostic for each frozen factor cell.
    contribution_rows: list[dict] = []
    for factor_id, lock in locks.items():
        horizon = int(lock["horizon_bars"])
        universe = str(lock["universe"])
        panel = universe_labels[(horizon, universe)].join(
            signals_by_factor[factor_id], on=["symbol", "feature_ts"], how="inner"
        )
        selected = (
            panel.with_columns(
                pl.col("signal")
                .rank("ordinal", descending=True)
                .over("decision_ts")
                .alias("signal_rank")
            )
            .filter(pl.col("signal_rank") <= 3)
            .group_by("symbol")
            .agg(
                pl.len().alias("selected_observations"),
                pl.col("forward_return").mean().alias("mean_gross_forward_return"),
                pl.col("forward_return").sum().alias("gross_return_sum"),
            )
        )
        denominator = float(selected["gross_return_sum"].abs().sum()) if selected.height else 0.0
        for row in selected.iter_rows(named=True):
            row.update(
                {
                    "factor_id": factor_id,
                    "horizon_bars": horizon,
                    "universe": universe,
                    "absolute_contribution_share": abs(float(row["gross_return_sum"])) / denominator
                    if denominator
                    else 0.0,
                }
            )
            contribution_rows.append(row)
    contributions = pl.DataFrame(contribution_rows)
    contributions.write_csv(ARTIFACTS / "factor_symbol_contribution.csv")

    validation_rows: list[dict] = []
    for factor_id, lock in locks.items():
        horizon = int(lock["horizon_bars"])
        universe = str(lock["universe"])
        metrics = {}
        for period in ("full", "research", "validation", "blind"):
            match = primary_adjusted.filter(
                (pl.col("factor_id") == factor_id)
                & (pl.col("direction") == 1)
                & (pl.col("horizon_bars") == horizon)
                & (pl.col("universe") == universe)
                & (pl.col("period") == period)
            )
            metrics[period] = match.to_dicts()[0] if match.height else {}
        full, blind = metrics["full"], metrics["blind"]
        if factor_id == "core.funding_fade_480":
            preliminary, reason = "INCONCLUSIVE", "funding history is about 95 days, not two years"
        elif factor_id == "v5.alpha6_static_proxy":
            preliminary, reason = (
                "INCONCLUSIVE",
                "static Alpha6 proxy omits state required for exact current-V5 replay",
            )
        elif full.get("ic_mean", 0.0) <= 0 or blind.get("ic_mean", 0.0) <= 0:
            preliminary, reason = (
                "FAIL",
                "IC direction is non-positive in full or blind OOS evidence",
            )
        else:
            preliminary, reason = (
                "INCONCLUSIVE",
                "positive pre-cost evidence still requires cost and portfolio validation",
            )
        validation_rows.append(
            {
                "factor_id": factor_id,
                "factor_name": FACTOR_NAMES[factor_id],
                "locked_horizon_bars": horizon,
                "locked_universe": universe,
                "pre_cost_decision": preliminary,
                "pre_cost_reason": reason,
                "full_ic_mean": full.get("ic_mean"),
                "full_hac_tstat": full.get("hac_tstat"),
                "full_non_overlap_tstat": full.get("non_overlap_tstat"),
                "full_holm_pvalue": full.get("holm_pvalue"),
                "research_ic_mean": metrics["research"].get("ic_mean"),
                "validation_ic_mean": metrics["validation"].get("ic_mean"),
                "blind_ic_mean": blind.get("ic_mean"),
                "blind_hac_tstat": blind.get("hac_tstat"),
                "data_start": str(data_start),
                "data_end": str(data_end),
            }
        )
    validations = pl.DataFrame(validation_rows)
    validations.write_csv(ARTIFACTS / "factor_validation_results.csv")

    false_positive_count = int(controls["significance_screen_pass"].sum())
    methodology = [
        "# Statistical Methodology",
        "",
        "- Feature timestamp is a closed 1h bar; decision timestamp is t+1h; label exits at decision+horizon.",
        "- Naive t-stat is retained only to show overlap inflation and is never used for a PASS decision.",
        "- Non-overlap keeps next timestamp only when next >= previous + horizon.",
        "- HAC/Newey-West uses Bartlett weights and lag equal to horizon bars (24/72/120/480).",
        "- OOS split is ordered 50% research / 25% validation / 25% blind; labels crossing a boundary are purged.",
        "- The locked horizon/universe maximizes min(HAC t, non-overlap t) on positive-IC research cells only.",
        "- Holm-Bonferroni and Benjamini-Hochberg include every executed factor, direction, horizon, universe, slice, and negative control.",
        f"- IC multiplicity test_count={multiplicity.height}; family_count=2; no failed cell was deleted.",
        f"- Negative-control corrected significance screens passed={false_positive_count}/{controls.height}; negative controls are never PAPER_READY.",
        "- Formal threshold is predeclared: rank IC > 0.03 and HAC or non-overlap t >= 2, plus OOS/cost/concentration/portfolio requirements.",
        "",
        "## Locked cells",
        "",
        "```json",
        json.dumps(locks, indent=2),
        "```",
    ]
    (REPORTS / "statistical_methodology.md").write_text(
        "\n".join(methodology) + "\n", encoding="utf-8"
    )

    factor_report = [
        "# Factor Validation Report — Pre-cost",
        "",
        f"- generated_at: {datetime.now(UTC).isoformat()}",
        f"- data: {data_start} to {data_end}; 93 backfilled symbols; dynamic top10/top20/top50 use prior-day liquidity.",
        f"- executed IC tests after deduplication: {multiplicity.height}.",
        "- current-listed seed set creates survivorship uncertainty; static V5 proxy is not exact production replay.",
        "",
        "| Factor | Locked cell | Full IC | HAC t | Non-overlap t | Blind IC | Pre-cost decision | Reason |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in validation_rows:
        factor_report.append(
            f"| {row['factor_name']} | {row['locked_universe']}/{row['locked_horizon_bars']}h | "
            f"{(row['full_ic_mean'] or 0):.4f} | {(row['full_hac_tstat'] or 0):.2f} | "
            f"{(row['full_non_overlap_tstat'] or 0):.2f} | {(row['blind_ic_mean'] or 0):.4f} | "
            f"{row['pre_cost_decision']} | {row['pre_cost_reason']} |"
        )
    factor_report.extend(
        [
            "",
            "All PASS decisions are deferred until transaction-cost, long-only portfolio, concentration, and capacity evidence is complete.",
        ]
    )
    (REPORTS / "factor_validation_report.md").write_text(
        "\n".join(factor_report) + "\n", encoding="utf-8"
    )
    print(
        f"factor tests={multiplicity.height}; negative controls={controls.height}; screen false positives={false_positive_count}"
    )


if __name__ == "__main__":
    main()
