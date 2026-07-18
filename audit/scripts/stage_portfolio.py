# ruff: noqa: E501
"""Execute the complete predeclared long-only portfolio grid and OOS lock."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.multiple_testing import adjust_pvalues  # noqa: E402
from audit.auditlib.paths import ARTIFACTS, DATA_GOLD, DATA_SILVER, REPORTS  # noqa: E402
from audit.auditlib.portfolio_backtest import (  # noqa: E402
    Simulation,
    build_market_matrix,
    performance_metrics,
    simulate_long_only,
)

FACTOR_FILES = {
    "v5.alpha6_static_proxy": "v5__alpha6_static_proxy.parquet",
    "core.rev_xs_480": "core__rev_xs_480.parquet",
    "core.low_vol_480": "core__low_vol_480.parquet",
    "core.funding_fade_480": "core__funding_fade_480.parquet",
}
TOP_COUNTS = [1, 3, 5]
WEIGHTINGS = ["equal", "score"]
REBALANCE_HOURS = [72, 120]
STAGGERED = [False, True]
BTC_FILTER = [False, True]
MAX_WEIGHT = {1: 1.0, 3: 0.50, 5: 0.35}


def _index_at_or_after(timestamps: list[datetime], target: datetime) -> int:
    for index, timestamp in enumerate(timestamps):
        if timestamp >= target:
            return index
    return len(timestamps)


def _structure_id(
    factor: str, top_n: int, weighting: str, rebalance: int, staggered: bool, btc_filter: bool
) -> str:
    return f"{factor}|top{top_n}|{weighting}|{rebalance}h|staggered={int(staggered)}|btc_filter={int(btc_filter)}"


def main() -> None:
    bars = pl.read_parquet(DATA_SILVER / "bars_1h.parquet")
    universes = pl.read_parquet(DATA_GOLD / "dynamic_universe.parquet")
    signals = {
        factor_id: pl.read_parquet(DATA_GOLD / "signals" / filename)
        for factor_id, filename in FACTOR_FILES.items()
    }
    factor_lock = json.loads((ARTIFACTS / "oos_lock.json").read_text(encoding="utf-8"))["factors"]
    costs = pl.read_csv(ARTIFACTS / "cost_model_summary.csv")
    cost_rows = {row["scenario"]: row for row in costs.iter_rows(named=True)}
    market = build_market_matrix(bars, signals, universes)

    start = market.timestamps[0]
    end = market.timestamps[-1] + timedelta(hours=1)
    duration = end - start
    split_50 = start + duration / 2
    split_75 = start + duration * 0.75
    bounds = {
        "research": (start, split_50),
        "validation": (split_50, split_75),
        "blind": (split_75, end),
        "full": (start, end),
    }
    indices = {
        name: (
            _index_at_or_after(market.timestamps, left),
            _index_at_or_after(market.timestamps, right),
        )
        for name, (left, right) in bounds.items()
    }

    structures: dict[str, tuple[dict, object]] = {}
    research_rows: list[dict] = []
    for factor_id in FACTOR_FILES:
        universe = str(factor_lock[factor_id]["universe"])
        for top_n in TOP_COUNTS:
            for weighting in WEIGHTINGS:
                for rebalance in REBALANCE_HOURS:
                    for staggered in STAGGERED:
                        for btc_filter in BTC_FILTER:
                            structure_id = _structure_id(
                                factor_id, top_n, weighting, rebalance, staggered, btc_filter
                            )
                            config = {
                                "structure_id": structure_id,
                                "factor_id": factor_id,
                                "universe": universe,
                                "top_n": top_n,
                                "weighting": weighting,
                                "max_single_weight": MAX_WEIGHT[top_n],
                                "rebalance_hours": rebalance,
                                "staggered": staggered,
                                "btc_filter": btc_filter,
                            }
                            simulation = simulate_long_only(
                                market,
                                **{
                                    key: value
                                    for key, value in config.items()
                                    if key not in {"structure_id", "max_single_weight"}
                                },
                                max_weight=MAX_WEIGHT[top_n],
                            )
                            structures[structure_id] = (config, simulation)
                            base_cost = cost_rows["base"]
                            metrics, _ = performance_metrics(
                                simulation,
                                market,
                                one_way_cost_bps=float(base_cost["one_way_cost_bps"]),
                                fee_bps=float(base_cost["fee_bps"] or 10.0),
                                start_index=indices["research"][0],
                                end_index=indices["research"][1],
                                rebalance_hours=rebalance,
                            )
                            research_rows.append(
                                {**config, "cost_scenario": "base", "period": "research", **metrics}
                            )
        print(f"simulated structures for {factor_id}")

    # Lock one portfolio rule per factor using research-only base-cost results.
    research_frame = pl.DataFrame(research_rows)
    portfolio_locks: dict[str, dict] = {}
    for factor_id in FACTOR_FILES:
        candidates = research_frame.filter(pl.col("factor_id") == factor_id).with_columns(
            (pl.col("sharpe") + pl.col("total_return").clip(-1.0, 1.0)).alias("selection_score")
        )
        selected = (
            candidates.sort(
                ["selection_score", "total_return", "max_drawdown", "turnover"],
                descending=[True, True, True, False],
            )
            .head(1)
            .to_dicts()[0]
        )
        portfolio_locks[factor_id] = {
            key: selected[key]
            for key in [
                "structure_id",
                "universe",
                "top_n",
                "weighting",
                "max_single_weight",
                "rebalance_hours",
                "staggered",
                "btc_filter",
                "total_return",
                "sharpe",
            ]
        }
        portfolio_locks[factor_id]["selection_rule"] = (
            "maximize research-only base-cost (Sharpe + clipped total return); all configurations retained"
        )
    (ARTIFACTS / "portfolio_oos_lock.json").write_text(
        json.dumps(
            {
                "locked_at": datetime.now(UTC).isoformat(),
                "blind_not_read_before_lock": True,
                "combination_count": len(structures),
                "factors": portfolio_locks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("portfolio rules locked before validation/blind metrics")

    result_rows: list[dict] = []
    daily_rows: list[dict] = []
    contribution_rows: list[dict] = []
    for structure_id, (config, simulation) in structures.items():
        for scenario, cost in cost_rows.items():
            for period, (left, right) in indices.items():
                metrics, series = performance_metrics(
                    simulation,
                    market,
                    one_way_cost_bps=float(cost["one_way_cost_bps"]),
                    fee_bps=float(cost["fee_bps"] or 10.0),
                    start_index=left,
                    end_index=right,
                    rebalance_hours=int(config["rebalance_hours"]),
                )
                test_id = f"portfolio_return|{structure_id}|{scenario}|{period}"
                result_rows.append(
                    {
                        **config,
                        "cost_scenario": scenario,
                        "one_way_cost_bps": cost["one_way_cost_bps"],
                        "period": period,
                        "test_id": test_id,
                        "test_family": "portfolio_return",
                        **metrics,
                    }
                )
                if scenario == "base" and period == "full":
                    daily_dates = series["daily_dates"]
                    gross_equity = np.cumprod(1.0 + series["daily_gross"])
                    net_equity = np.cumprod(1.0 + series["daily_net"])
                    drawdown = net_equity / np.maximum.accumulate(net_equity) - 1.0
                    for date_value, gross_value, net_value, draw_value in zip(
                        daily_dates,
                        gross_equity,
                        net_equity,
                        drawdown,
                        strict=True,
                    ):
                        daily_rows.append(
                            {
                                "structure_id": structure_id,
                                "factor_id": config["factor_id"],
                                "ts": date_value,
                                "cost_scenario": "base",
                                "gross_equity": float(gross_value),
                                "net_equity": float(net_value),
                                "drawdown": float(draw_value),
                            }
                        )
        denominator = float(np.abs(simulation.contribution_by_symbol).sum())
        for symbol, value in zip(market.symbols, simulation.contribution_by_symbol, strict=True):
            if value == 0:
                continue
            contribution_rows.append(
                {
                    "structure_id": structure_id,
                    "factor_id": config["factor_id"],
                    "symbol": symbol,
                    "gross_contribution": float(value),
                    "absolute_contribution_share": abs(float(value)) / denominator
                    if denominator
                    else 0.0,
                }
            )

    results = pl.DataFrame(result_rows)
    contributions = pl.DataFrame(contribution_rows)
    concentration = (
        contributions.sort(
            ["structure_id", "absolute_contribution_share"], descending=[False, True]
        )
        .group_by("structure_id")
        .agg(
            pl.col("absolute_contribution_share").max().alias("max_symbol_contribution_share"),
            pl.col("absolute_contribution_share").head(3).sum().alias("top3_contribution_share"),
        )
    )
    results = results.join(concentration, on="structure_id", how="left")

    # Recompute global Holm/BH across the prior IC family plus every portfolio
    # combination x cost x period. The prior adjusted values are not trusted.
    prior = pl.read_csv(ARTIFACTS / "multiple_testing_summary.csv").select(
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
    )
    portfolio_tests = (
        results.select(["test_id", "test_family", "factor_id", "universe", "period", "raw_pvalue"])
        .with_columns(
            pl.lit(None).cast(pl.Int64).alias("direction"),
            pl.lit(None).cast(pl.Int64).alias("horizon_bars"),
        )
        .select(prior.columns)
    )
    multiplicity = pl.concat([prior, portfolio_tests], how="vertical_relaxed").unique(
        subset=["test_id"], keep="first"
    )
    holm, bh = adjust_pvalues(multiplicity["raw_pvalue"].to_numpy())
    multiplicity = multiplicity.with_columns(
        pl.Series("holm_pvalue", holm),
        pl.Series("bh_fdr_pvalue", bh),
        pl.lit(multiplicity.height).alias("test_count"),
        pl.lit(3).alias("family_count"),
    )
    multiplicity.write_csv(ARTIFACTS / "multiple_testing_summary.csv")
    results = results.join(
        multiplicity.select(
            ["test_id", "holm_pvalue", "bh_fdr_pvalue", "test_count", "family_count"]
        ),
        on="test_id",
        how="left",
    )
    results.write_csv(ARTIFACTS / "portfolio_backtest_results.csv")
    pl.DataFrame(daily_rows).write_parquet(
        ARTIFACTS / "portfolio_timeseries.parquet", compression="zstd"
    )
    contributions.write_csv(ARTIFACTS / "portfolio_symbol_contribution.csv")

    # Executed long-only benchmarks. The exact current V5 benchmark remains
    # explicitly unavailable rather than being silently replaced by a proxy.
    n_times, n_symbols = market.returns.shape
    benchmark_simulations: dict[str, Simulation] = {}
    for symbol in ("BTC-USDT", "ETH-USDT"):
        gross = np.zeros(n_times)
        turnover = np.zeros(n_times)
        traded = np.zeros(n_times)
        contribution = np.zeros(n_symbols)
        symbol_index = market.symbols.index(symbol)
        gross[2:] = market.returns[2:, symbol_index]
        traded[1] = 1.0
        turnover[1] = 0.5
        contribution[symbol_index] = gross.sum()
        benchmark_simulations[f"{symbol.split('-')[0]} buy-and-hold"] = Simulation(
            gross, turnover, traded, contribution
        )

    dynamic = Simulation(
        np.zeros(n_times), np.zeros(n_times), np.zeros(n_times), np.zeros(n_symbols)
    )
    current_weights = np.zeros(n_symbols)
    last_date = None
    for time_index in range(1, n_times):
        contribution = current_weights * market.returns[time_index]
        dynamic.gross_returns[time_index] = contribution.sum()
        dynamic.contribution_by_symbol += contribution
        date_value = market.timestamps[time_index].date()
        if date_value == last_date:
            continue
        last_date = date_value
        target = np.zeros(n_symbols)
        members = market.universe_members["top20"].get(date_value)
        if members is not None and members.size:
            target[members] = 1.0 / members.size
        delta = np.abs(target - current_weights)
        dynamic.turnover[time_index] = 0.5 * delta.sum()
        dynamic.traded_fraction[time_index] = delta.sum()
        current_weights = target
    benchmark_simulations["dynamic top20 equal-weight"] = dynamic
    benchmark_simulations["cash"] = Simulation(
        np.zeros(n_times), np.zeros(n_times), np.zeros(n_times), np.zeros(n_symbols)
    )

    benchmark_rows = []
    base_cost = cost_rows["base"]
    for benchmark, simulation in benchmark_simulations.items():
        metrics, series = performance_metrics(
            simulation,
            market,
            one_way_cost_bps=float(base_cost["one_way_cost_bps"]),
            fee_bps=float(base_cost["fee_bps"] or 10.0),
            start_index=0,
            end_index=n_times,
            rebalance_hours=24 if benchmark == "dynamic top20 equal-weight" else 24 * 365,
        )
        benchmark_rows.append({"benchmark": benchmark, "evidence_status": "AVAILABLE", **metrics})
        gross_equity = np.cumprod(1.0 + series["daily_gross"])
        net_equity = np.cumprod(1.0 + series["daily_net"])
        drawdown = net_equity / np.maximum.accumulate(net_equity) - 1.0
        for date_value, gross_value, net_value, draw_value in zip(
            series["daily_dates"],
            gross_equity,
            net_equity,
            drawdown,
            strict=True,
        ):
            daily_rows.append(
                {
                    "structure_id": f"benchmark.{benchmark}",
                    "factor_id": "benchmark",
                    "ts": date_value,
                    "cost_scenario": "base",
                    "gross_equity": float(gross_value),
                    "net_equity": float(net_value),
                    "drawdown": float(draw_value),
                }
            )
    benchmark_rows.append(
        {
            "benchmark": "current V5 production strategy exact replay",
            "evidence_status": "INCONCLUSIVE_STATE_HISTORY_MISSING",
            "gross_total_return": None,
            "total_return": None,
        }
    )
    pl.DataFrame(benchmark_rows).write_csv(ARTIFACTS / "portfolio_benchmarks.csv")
    pl.DataFrame(daily_rows).write_parquet(
        ARTIFACTS / "portfolio_timeseries.parquet", compression="zstd"
    )

    locked_rows = []
    for _factor_id, lock in portfolio_locks.items():
        structure_id = lock["structure_id"]
        for period in ("research", "validation", "blind", "full"):
            row = results.filter(
                (pl.col("structure_id") == structure_id)
                & (pl.col("cost_scenario") == "base")
                & (pl.col("period") == period)
            ).to_dicts()[0]
            locked_rows.append(row)
    locked_frame = pl.DataFrame(locked_rows)
    locked_frame.write_csv(ARTIFACTS / "locked_portfolio_results.csv")

    lines = [
        "# Portfolio Validation Report",
        "",
        f"- generated_at: {datetime.now(UTC).isoformat()}",
        f"- structural combinations: {len(structures)}; scenarios x periods result rows: {results.height}.",
        "- Spot long-only only; no short-leg PnL is counted.",
        "- Every feature is delayed one bar. Rebalance occurs at the decision close and affects the next return interval.",
        "- Tested Top1/3/5, equal/score weights, caps, 72h/120h, staggered on/off, BTC 60d trend filter on/off.",
        "- Portfolio locks were selected on research-only base-cost results before validation/blind metrics.",
        "- Exact current V5 production strategy remains an INCONCLUSIVE benchmark because dynamic state history is missing.",
        "",
        "## Locked factor portfolios (base cost)",
        "",
        "| Factor | Period | Rule | Net return | Sharpe | Max DD | Turnover | Edge/cost | Holm p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in locked_rows:
        lines.append(
            f"| {row['factor_id']} | {row['period']} | top{row['top_n']} {row['weighting']} {row['rebalance_hours']}h "
            f"staggered={row['staggered']} filter={row['btc_filter']} | {row['total_return']:.4f} | {row['sharpe']:.2f} | "
            f"{row['max_drawdown']:.4f} | {row['turnover']:.2f} | {row['edge_cost_ratio']:.2f} | {row['holm_pvalue']:.4g} |"
        )
    (REPORTS / "portfolio_validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"portfolio rows={results.height}; daily timeseries rows={len(daily_rows)}; global tests={multiplicity.height}"
    )


if __name__ == "__main__":
    main()
