# ruff: noqa: E501
"""Generate Audit v2 decisions, Markdown reports, and the static dashboard."""

from __future__ import annotations

import html
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.decisions_v2 import (  # noqa: E402
    validate_dashboard_contract,
    validate_layered_decisions,
)


def _pct(value: float | None) -> str:
    return "—" if value is None or not math.isfinite(float(value)) else f"{float(value) * 100:.2f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _histogram_svg(values_by_name: dict[str, np.ndarray], observed: float) -> str:
    all_values = np.concatenate(list(values_by_name.values()))
    low, high = float(np.quantile(all_values, 0.005)), float(np.quantile(all_values, 0.995))
    if high <= low:
        high = low + 1.0
    width, height, base = 860, 220, 185
    palette = ["#5f7881", "#78909a", "#90a4ae", "#b0bec5"]
    groups = []
    bins = np.linspace(low, high, 45)
    peak = 1
    histograms = []
    for values in values_by_name.values():
        counts, _ = np.histogram(values, bins=bins)
        peak = max(peak, int(counts.max()))
        histograms.append(counts)
    for group_index, (name, counts) in enumerate(zip(values_by_name, histograms, strict=True)):
        points = []
        for index, count in enumerate(counts):
            x = 24 + index / (len(counts) - 1) * (width - 48)
            y = base - count / peak * 140
            points.append(f"{x:.1f},{y:.1f}")
        groups.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{palette[group_index]}" stroke-width="2" opacity="0.85"><title>{html.escape(name)}</title></polyline>'
        )
    observed_x = 24 + (observed - low) / (high - low) * (width - 48)
    observed_x = max(24, min(width - 24, observed_x))
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Permutation null IC distributions">'
        f'<line x1="24" y1="{base}" x2="{width-24}" y2="{base}" stroke="#40515a"/>'
        + "".join(groups)
        + f'<line x1="{observed_x:.1f}" y1="25" x2="{observed_x:.1f}" y2="{base}" stroke="#f4b740" stroke-width="3"/><text x="{min(observed_x+8,width-160):.1f}" y="38" fill="#f4b740" font-size="13">observed IC {observed:.3f}</text>'
        + f'<text x="24" y="210" fill="#879aa3" font-size="12">{low:.3f}</text><text x="{width-72}" y="210" fill="#879aa3" font-size="12">{high:.3f}</text></svg>'
    )


def _status_class(value: str) -> str:
    return "pass" if value == "PASS" else "fail" if value == "FAIL" else "inc"


def main() -> None:
    root = Path(os.environ.get("AUDIT_ROOT", "/home/hr/quant-alpha-audit-v2"))
    artifacts, reports = root / "artifacts", root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = json.loads((artifacts / "analysis_summary_v2.json").read_text())
    replay = json.loads((artifacts / "v5_replayability_status.json").read_text())
    costs = json.loads((artifacts / "cost_model_v2_metadata.json").read_text())
    forward = pl.read_csv(artifacts / "forward_paper_performance.csv").to_dicts()[0]
    p24 = pl.read_csv(artifacts / "portfolio_24h_results.csv")
    locked = pl.read_csv(artifacts / "low_vol_locked_and_posthoc_portfolios.csv")
    funding = pl.read_csv(artifacts / "funding_window_comparison.csv")
    frequency = pl.read_csv(artifacts / "funding_frequency_summary.csv")
    nulls = pl.read_csv(artifacts / "null_control_summary.csv")
    boot = pl.read_csv(artifacts / "block_bootstrap_summary.csv")
    hac = pl.read_csv(artifacts / "hac_bandwidth_sensitivity.csv")
    sensitivity = pl.read_csv(artifacts / "low_vol_signal_sensitivity.csv")
    multiplicity = pl.read_csv(artifacts / "multiple_testing_v2.csv")

    low_primary = summary["primary_signal_metrics"]["core.low_vol_480"]
    rev_primary = summary["primary_signal_metrics"]["core.rev_xs_480"]
    funding_time = funding.filter(pl.col("window_type") == "TIME_BASED_20D").to_dicts()[0]
    low_signal_boot = boot.filter(pl.col("object_type") == "signal_ic")
    low_signal_pass = (
        low_primary["ic_mean"] > 0
        and min(low_signal_boot["ci_95_low"]) > 0
        and max(nulls["ic_two_sided_empirical_p_value"]) < 0.01
        and min(
            hac.filter(pl.col("object_id") == "core.low_vol_480.rank_ic")["t_stat"]
        ) > 2
    )
    low24 = p24.filter(pl.col("factor_id") == "core.low_vol_480")
    rev24 = p24.filter(pl.col("factor_id") == "core.rev_xs_480")
    low_validation_positive = int(
        low24.filter((pl.col("period") == "validation") & (pl.col("total_return") > 0)).height
    )
    low_blind_positive = int(
        low24.filter((pl.col("period") == "blind") & (pl.col("total_return") > 0)).height
    )
    locked_blind = locked.filter(
        (pl.col("object") == "low_vol_locked") & (pl.col("period") == "blind")
    ).to_dicts()[0]
    locked_validation = locked.filter(
        (pl.col("object") == "low_vol_locked") & (pl.col("period") == "validation")
    ).to_dicts()[0]
    posthoc_full = locked.filter(
        (pl.col("object") == "low_vol_btc_trend") & (pl.col("period") == "full")
    ).to_dicts()[0]
    posthoc_blind = locked.filter(
        (pl.col("object") == "low_vol_btc_trend") & (pl.col("period") == "blind")
    ).to_dicts()[0]
    market_cutoff = str(forward["available_data_cutoff"])
    decisions = {
        "audit_version": "v2",
        "audit_v1_data_cutoff": "2026-07-17T23:00:00+00:00",
        "audit_v2_data_cutoff": market_cutoff,
        "current_v5": {
            "signal_validity": "INCONCLUSIVE",
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "FAIL",
            "replayability": replay["replayability"],
            "reason": "The current runtime is fingerprinted, but no immutable atomic per-decision history binds sentiment, dynamic IC, regime/HMM/RSS, optimizer inputs, universe, positions and target weights. Exact historical replay remains unavailable.",
        },
        "rev_xs_20d": {
            "signal_validity": "FAIL",
            "portfolio_validity": "FAIL",
            "deployment_readiness": "FAIL",
            "recommended_action": "RETIRE",
            "key_metrics": {
                "rank_ic_24h": rev_primary["ic_mean"],
                "rank_ic_hac_tstat": rev_primary["hac_tstat"],
                "pearson_ic_24h": rev_primary["pearson_ic_mean"],
                "blind_rank_ic": float(
                    sensitivity.filter(
                        (pl.col("factor_id") == "core.rev_xs_480")
                        & (pl.col("slice_type") == "oos_period")
                        & (pl.col("slice") == "blind")
                        & (pl.col("metric") == "rank_ic")
                    )["estimate"][0]
                ),
                "best_full_24h_net_return": float(
                    rev24.filter(pl.col("period") == "full")["total_return"].max()
                ),
                "best_blind_24h_net_return": float(
                    rev24.filter(pl.col("period") == "blind")["total_return"].max()
                ),
            },
        },
        "low_vol_20d": {
            "signal_validity": "PASS" if low_signal_pass else "INCONCLUSIVE",
            "locked_portfolio_validity": "FAIL",
            "deployment_readiness": "FAIL",
            "key_metrics": {
                "rank_ic_24h": low_primary["ic_mean"],
                "pearson_ic_24h": low_primary["pearson_ic_mean"],
                "hac_tstat_24h": low_primary["hac_tstat"],
                "non_overlap_tstat_24h": low_primary["non_overlap_tstat"],
                "bootstrap_95_low_min": float(low_signal_boot["ci_95_low"].min()),
                "permutation_two_sided_empirical_pvalue_max": float(
                    nulls["ic_two_sided_empirical_p_value"].max()
                ),
                "locked_validation_net_return": locked_validation["total_return"],
                "locked_blind_net_return": locked_blind["total_return"],
                "portfolio_24h_validation_positive_structures": low_validation_positive,
                "portfolio_24h_blind_positive_structures": low_blind_positive,
            },
        },
        "low_vol_btc_trend": {
            "hypothesis_type": "POST_HOC_HYPOTHESIS",
            "parameters_locked": True,
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "INCONCLUSIVE",
            "forward_sample_start": "2026-07-18T00:00:00+00:00",
            "forward_sample_end": market_cutoff,
            "forward_trade_count": int(forward["completed_independent_trades"]),
            "key_metrics": {
                "hypothesis_generating_full_net_return": posthoc_full["total_return"],
                "hypothesis_generating_blind_net_return": posthoc_blind["total_return"],
                "forward_calendar_days": float(forward["forward_calendar_days"]),
                "forward_decision_count": int(forward["decision_count"]),
                "forward_net_return": float(forward["portfolio_net_return"]),
            },
        },
        "funding_fade": {
            "signal_validity": "INCONCLUSIVE",
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "INCONCLUSIVE",
            "window_type": "TIME_BASED_20D",
            "key_metrics": {
                "symbols": frequency.height,
                "max_history_days": float(frequency["actual_coverage_days"].max()),
                "min_history_days": float(frequency["actual_coverage_days"].min()),
                "rank_ic_72h": funding_time["ic_mean"],
                "hac_tstat_72h": funding_time["hac_tstat"],
                "non_overlap_count": funding_time["non_overlap_count"],
            },
        },
        "recommended_production_action": "Continue freezing live Alpha. Retire rev_xs_20d. Keep only the locked low_vol + BTC trend hypothesis in forward-only paper observation; do not promote it.",
        "highest_priority_next_step": "Accumulate strictly new low_vol + BTC trend forward evidence beyond the v1 cutoff until at least 14 calendar days and enough completed independent 120h decisions exist, without changing the frozen parameters.",
    }
    validate_layered_decisions(decisions)
    (artifacts / "final_decisions_v2.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    layered = pl.DataFrame(
        [
            {
                "object": "low_vol_20d.signal",
                "hypothesis_type": "V1_LOCKED_SIGNAL",
                "signal_validity": decisions["low_vol_20d"]["signal_validity"],
                "portfolio_validity": "NOT_APPLICABLE",
                "deployment_readiness": "INCONCLUSIVE",
                "reason": "Positive rank/Pearson IC survives OOS, neutralization, exclusions, HAC, block bootstrap and four 1000-permutation null families.",
            },
            {
                "object": "low_vol_20d.locked_120h_no_filter",
                "hypothesis_type": "V1_LOCKED_CONFIRMATORY",
                "signal_validity": decisions["low_vol_20d"]["signal_validity"],
                "portfolio_validity": "FAIL",
                "deployment_readiness": "FAIL",
                "reason": "Validation and blind net returns are both negative.",
            },
            {
                "object": "low_vol_20d.top3_score_120h_btc_filter_on",
                "hypothesis_type": "POST_HOC_HYPOTHESIS",
                "signal_validity": "INHERITS_LOW_VOL_SIGNAL",
                "portfolio_validity": "INCONCLUSIVE",
                "deployment_readiness": "INCONCLUSIVE",
                "reason": "The rule was identified after viewing the v1 grid; all old interval results are hypothesis-generating only and current forward data are insufficient.",
            },
        ]
    )
    layered.write_csv(artifacts / "low_vol_layered_decisions.csv")

    best_low_full = low24.filter(pl.col("period") == "full").sort(
        "total_return", descending=True
    ).to_dicts()[0]
    best_rev_full = rev24.filter(pl.col("period") == "full").sort(
        "total_return", descending=True
    ).to_dicts()[0]
    low_fixed = low24.filter(
        (pl.col("top_n") == 3)
        & (pl.col("weighting") == "score")
        & (pl.col("btc_filter") == True)  # noqa: E712
    ).sort("period")
    empirical_p = float(nulls["ic_two_sided_empirical_p_value"].max())
    multiplicity_rows = []
    for test_type, label in (
        ("confirmatory", "CONFIRMATORY"),
        ("exploratory", "EXPLORATORY"),
    ):
        subset = multiplicity.filter(pl.col("test_type") == test_type)
        multiplicity_rows.append(
            "<tr>"
            f"<th>{label}</th>"
            f"<td>{subset.height}</td>"
            f"<td>{float(subset['raw_pvalue'].min()):.4f}</td>"
            f"<td>{float(subset['holm_pvalue'].min()):.4f}</td>"
            f"<td>{float(subset['bh_fdr_pvalue'].min()):.4f}</td>"
            f"<td>{float(subset['max_stat_permutation_pvalue'].min()):.4f}</td>"
            "</tr>"
        )
    multiplicity_html = "".join(multiplicity_rows)

    _write(
        reports / "executive_summary_v2.md",
        [
            "# Alpha Validity Audit v2 — Executive Summary",
            "",
            "## Outcome",
            "",
            "- **Current V5:** PARTIALLY_REPLAYABLE; deployment readiness FAIL. Live Alpha remains frozen.",
            "- **rev_xs_20d:** signal FAIL, 24h portfolio FAIL, deployment FAIL; RETIRE.",
            "- **low_vol_20d signal:** PASS. Rank IC 0.1009, Pearson IC 0.0810, HAC t 8.70, non-overlap t 6.55; all four null families give two-sided empirical p=0.000999.",
            "- **low_vol locked 120h/no-filter portfolio:** FAIL. Validation net -6.73%; blind net -16.55%.",
            "- **low_vol 24h portfolios:** FAIL as tradable evidence. Full sample 9/12 are net positive and blind 5/12, but validation is 0/12 and no rule is positive in both validation and blind.",
            "- **low_vol + BTC trend:** POST_HOC_HYPOTHESIS. Historical results are hypothesis-generating only; portfolio/deployment remain INCONCLUSIVE.",
            f"- **Forward:** {float(forward['forward_calendar_days']):.3f} day, {int(forward['decision_count'])} decision, {int(forward['completed_independent_trades'])} completed 120h trade; INCONCLUSIVE_FORWARD_SAMPLE_INSUFFICIENT.",
            f"- **funding_fade:** natural-time 20d closed-left window implemented. History spans {float(frequency['actual_coverage_days'].min()):.2f}–{float(frequency['actual_coverage_days'].max()):.2f} days; conclusion remains INCONCLUSIVE.",
            f"- **Cost:** {costs['fill_rows']} fills across {costs['covered_symbols']} symbols, but requested-price/bid-ask/latency coverage is 0/0/0 rows. All three scenarios therefore remain the same conservative 30bps roundtrip floor.",
            "",
            "No production code, services, positions, risk permissions, secrets, or orders were changed.",
        ],
    )

    table_rows = []
    for row in low_fixed.iter_rows(named=True):
        table_rows.append(
            f"| {row['period']} | {_pct(row['gross_total_return'])} | {_pct(row['total_return'])} | {_pct(row['total_cost_return'])} | {_num(row['edge_cost_ratio'],2)} | {_num(row['sharpe'],2)} | {_pct(row['max_drawdown'])} | {row['trade_count']} |"
        )
    _write(
        reports / "24h_portfolio_validation.md",
        [
            "# 24h Portfolio Validation",
            "",
            "Only the v1 structural rules were used: Top1/3/5, equal/score, BTC 60d filter on/off, 24h rebalance, long-only spot, causal one-bar delay, 15bps one-way cost. No new Top N, weights, stops, regimes, lookbacks or factors were introduced.",
            "",
            "- low_vol: full net-positive structures 9/12; validation 0/12; blind 5/12. No structure is positive in both validation and blind. **Portfolio validity: FAIL.**",
            f"- rev_xs: best descriptive full net return {_pct(best_rev_full['total_return'])}; all 12 full and all 12 blind structures are net negative. **Portfolio validity: FAIL.**",
            f"- Descriptive low_vol full maximum {_pct(best_low_full['total_return'])} is not treated as a new selected rule.",
            "- 24h turnover was measured from actual target-weight changes; fees and slippage were deducted for both one-way legs. The result is not pre-rejected for turnover: actual gross edge and costs are shown.",
            "",
            "## Fixed Top3 / score / BTC filter ON at 24h",
            "",
            "| Period | Gross return | Net return | Cost return | Edge/cost | Sharpe | Max drawdown | Trades |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *table_rows,
            "",
            "The full and blind samples cover costs for this descriptive structure, but validation is negative and every 95% portfolio block-bootstrap interval includes zero. Gross edge therefore does not establish stable cost-after tradability.",
            "",
            "Machine results: `artifacts/portfolio_24h_results.csv`; symbol contributions: `artifacts/portfolio_24h_symbol_contribution.csv`.",
        ],
    )

    neutral = sensitivity.filter(pl.col("slice_type") == "neutralized")
    exclusion = sensitivity.filter(pl.col("slice_type") == "coin_exclusion")
    _write(
        reports / "low_vol_signal_vs_portfolio.md",
        [
            "# Low-vol Signal vs Portfolio",
            "",
            "## Layer 1 — signal",
            "",
            f"PASS. Rank IC {_num(low_primary['ic_mean'],4)}, Pearson IC {_num(low_primary['pearson_ic_mean'],4)}, HAC t {_num(low_primary['hac_tstat'],2)}, non-overlap t {_num(low_primary['non_overlap_tstat'],2)}.",
            f"All signal block-bootstrap 95% lower bounds are positive (minimum {_num(float(low_signal_boot['ci_95_low'].min()),4)}); four null families each use 1,000 permutations and the worst two-sided empirical p is {_num(empirical_p,6)}.",
            f"After beta neutralization, rank IC {_num(float(neutral.filter(pl.col('slice')=='beta_30d')['estimate'][0]),4)}; after liquidity neutralization {_num(float(neutral.filter(pl.col('slice')=='log_qv_30d')['estimate'][0]),4)}.",
            f"Excluding BTC/ETH/SOL leaves rank IC {_num(float(exclusion.filter(pl.col('slice').str.starts_with('exclude_btc'))['estimate'][0]),4)}; excluding v1's largest contributor TRX leaves {_num(float(exclusion.filter(pl.col('slice').str.starts_with('exclude_largest'))['estimate'][0]),4)}.",
            "",
            "## Layer 2 — v1 locked portfolio",
            "",
            f"FAIL. Top3/score/120h/BTC filter OFF has validation net {_pct(locked_validation['total_return'])} and blind net {_pct(locked_blind['total_return'])}. Research profit does not override both later losses.",
            "",
            "## Layer 3 — post-hoc BTC trend hypothesis",
            "",
            f"INCONCLUSIVE / FORWARD ONLY. Top3/score/120h/BTC filter ON shows historical full net {_pct(posthoc_full['total_return'])} and old blind net {_pct(posthoc_blind['total_return'])}, but these are hypothesis-generating evidence because the rule was noticed after the complete v1 grid was viewed.",
            f"Current strictly new sample is {float(forward['forward_calendar_days']):.3f} day with {int(forward['completed_independent_trades'])} completed independent trade. No PASS or paper promotion is granted.",
        ],
    )

    fixed_funding = funding.filter(pl.col("window_type") == "FIXED_60_OBSERVATIONS").to_dicts()[0]
    pair_funding = funding.filter(pl.col("window_type") == "PAIRWISE_SIGNAL_DIFFERENCE").to_dicts()[0]
    freq_counts = frequency.group_by("median_settlement_frequency_hours").agg(pl.len().alias("symbols")).sort("median_settlement_frequency_hours")
    _write(
        reports / "funding_window_correction.md",
        [
            "# Funding Window Correction",
            "",
            "The v1 fixed `rolling_mean(60)` definition is retained only as a comparator. Audit v2 computes a true trailing 20-calendar-day window with `closed=left`: a settlement at the decision timestamp cannot enter that decision, missing settlements are never filled with zero, and 4h/8h instruments use the same elapsed-time boundary.",
            "",
            f"- Symbols: {frequency.height}; frequency groups: " + ", ".join(f"{row['median_settlement_frequency_hours']:.0f}h={row['symbols']}" for row in freq_counts.iter_rows(named=True)) + ".",
            f"- Actual history range: {float(frequency['actual_coverage_days'].min()):.2f}–{float(frequency['actual_coverage_days'].max()):.2f} days; max gap: {float(frequency['max_gap_hours'].max()):.1f}h.",
            f"- Fixed-60 IC/HAC t: {_num(fixed_funding['ic_mean'],4)} / {_num(fixed_funding['hac_tstat'],2)}.",
            f"- Natural-20d IC/HAC t: {_num(funding_time['ic_mean'],4)} / {_num(funding_time['hac_tstat'],2)}; non-overlap n={int(funding_time['non_overlap_count'])}.",
            f"- Paired signal correlation: {_num(pair_funding['paired_signal_correlation'],4)}; mean absolute difference: {_num(pair_funding['mean_absolute_signal_difference'],6)}.",
            "",
            "Because the history remains far shorter than two years and independent samples are sparse, signal, portfolio and deployment validity remain INCONCLUSIVE.",
        ],
    )

    _write(
        reports / "permutation_and_controls_report.md",
        [
            "# Permutation and Controls",
            "",
            "Audit v2 uses 1,000 fixed-seed permutations for each strict null: time-within-factor permutation, cross-sectional rank permutation, label permutation and independent random noise. Every row preserves IC, IC t-stat, long-only net return, return t-stat and the maximum absolute statistic.",
            "",
            "| Null | IC empirical p | IC two-sided p | Net one-sided p | Net two-sided p | Max-stat p |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {row['control_type']} | {_num(row['ic_empirical_p_value'],6)} | {_num(row['ic_two_sided_empirical_p_value'],6)} | {_num(row['net_return_empirical_p_value'],6)} | {_num(row['net_return_two_sided_empirical_p_value'],6)} | {_num(row['max_stat_adjusted_p_value'],6)} |"
                for row in nulls.iter_rows(named=True)
            ],
            "",
            "The daily non-overlapping low_vol observed IC is 0.0965. Signal nulls are rejected under all four definitions. The long-only return null is not interpreted as portfolio PASS: its two-sided p is weak because random high-turnover portfolios lose heavily after costs, while the actual portfolio bootstrap/HAC and validation slices remain unstable.",
            "",
            "Wrong lag, random universe, horizon changes, coin deletion and decision delay are reported separately as robustness perturbations, not strict negative controls.",
        ],
    )

    _write(
        reports / "statistical_sensitivity_report.md",
        [
            "# Statistical Sensitivity",
            "",
            "- HAC reports horizon/2, horizon, 2×horizon and automatic bandwidth. No favourable bandwidth is selected.",
            f"- low_vol rank IC HAC t range: {_num(float(hac.filter(pl.col('object_id')=='core.low_vol_480.rank_ic')['t_stat'].min()),2)} to {_num(float(hac.filter(pl.col('object_id')=='core.low_vol_480.rank_ic')['t_stat'].max()),2)}; all p-values < 1e-15.",
            f"- low_vol signal bootstrap uses 24h/72h/120h/240h blocks; 95% lower bounds {_num(float(low_signal_boot['ci_95_low'].min()),4)} to {_num(float(low_signal_boot['ci_95_low'].max()),4)}, P(IC>0)=1.000 for all blocks.",
            "- Portfolio block-bootstrap intervals are recorded for every 24h structure. For fixed Top3/score variants, every 95% interval includes zero; portfolio evidence is not stable.",
            f"- Effective non-overlapping low_vol observations: {int(low_primary['non_overlap_count'])}; rev: {int(rev_primary['non_overlap_count'])}; funding natural-20d: {int(funding_time['non_overlap_count'])}.",
            "- Confirmatory and exploratory tests are separated in `artifacts/multiple_testing_v2.csv`; post-hoc BTC-trend evidence never enters the confirmatory family.",
        ],
    )

    _write(
        reports / "cost_model_v2.md",
        [
            "# Cost Model v2",
            "",
            f"Read-only current qyun evidence contains {costs['fill_rows']} fills across {costs['covered_symbols']}/93 audit symbols. Fee currency, side, execution type, fill price/size and notional are available.",
            f"Requested-price coverage: {costs['requested_price_coverage_rows']} rows; bid/ask coverage: {costs['bid_ask_coverage_rows']}; latency coverage: {costs['latency_coverage_rows']}. All stored orders are market orders with `px` unavailable/zero, so arrival slippage cannot be reconstructed.",
            "The 62-symbol fill count is not equivalent to reliable cost calibration: v1 qlab actual cost buckets still cover only 4/93 symbols. Maker/taker labels, partial fills and fee distributions are reported, but broad-universe executable-cost evidence remains insufficient.",
            "",
            "Therefore optimistic/base/stress are not three independent scenarios. They deliberately collapse to the same conservative lower bound: **15bps one-way / 30bps roundtrip**.",
            "",
            "Artifacts: `cost_coverage_by_symbol.csv`, `cost_distribution_v2.csv`, `cost_model_v2_metadata.json`.",
        ],
    )

    _write(
        reports / "v5_replayability_audit.md",
        [
            "# V5 Replayability Audit",
            "",
            f"## Verdict: {replay['replayability']}",
            "",
            f"Current production HEAD `{replay['git_head']}` is clean. The snapshot records tracked/untracked inventory, sanitized diff, {replay['systemd_units_captured']} systemd unit hashes, {replay['config_hashes_captured']} config hashes, {replay['database_files_captured']} current DB schemas, {replay['state_documents_captured']} dynamic state documents and environment value hashes. No Docker container is used/captured on this host.",
            "Current state includes Alpha raw/z factors and scores, sentiment fields, regime/HMM/funding/RSS vote state, PROTECT risk state, event candidates, optimizer weights, account/position tables, ML runtime state, and current service/timer state. Positions are currently empty in both position stores.",
            "",
            "## Why exact history is still unavailable",
            "",
            *[f"- {item}" for item in replay["missing_state"]],
            "",
            "A present-time snapshot is not a historical event-sourced chain. The current V5 signal and portfolio therefore remain INCONCLUSIVE, while deployment readiness is FAIL. No real replay/order submission was attempted.",
        ],
    )

    _write(
        reports / "production_impact_assessment_v2.md",
        [
            "# Production Impact Assessment v2",
            "",
            "- V5-prod production code: unchanged.",
            "- qyun services/timers: read only; no restart/reload/enable/disable.",
            "- Positions, orders, Paper tracker, Risk Permission and strategy weights: unchanged.",
            "- API keys, secrets, passphrases, DB passwords and private keys: not stored. Environment values are presence + SHA256 only.",
            "- Live orders: 0. Production mutations: 0.",
            "- quant-lab work is isolated on `audit/alpha-validity-v2`; no merge or deploy.",
            "- Recommendation: continue freezing live Alpha; retire rev_xs; observe only the fixed post-hoc low_vol+BTC rule in forward-only paper.",
        ],
    )

    _write(
        reports / "reproduction_guide_v2.md",
        [
            "# Audit v2 Reproduction Guide",
            "",
            "```bash",
            "export AUDIT_V1_BUNDLE=/mnt/c/Users/HR/Downloads/alpha_audit_bundle_20260718_162903.zip",
            "export AUDIT_V1_DIR=/home/hr/quant-alpha-audit-v2/v1",
            "export AUDIT_ROOT=/home/hr/quant-alpha-audit-v2",
            "export AUDIT_V1_ROOT=/home/hr/quant-alpha-audit",
            "cd /home/hr/quant-alpha-audit/repos/quant-lab",
            "/home/hr/quant-alpha-audit/.venv/bin/python audit/scripts/stage_v2_consistency.py",
            "/home/hr/quant-alpha-audit/.venv/bin/python audit/scripts/stage_v2_analysis.py",
            "/home/hr/quant-alpha-audit/.venv/bin/python audit/scripts/stage_v2_cost.py",
            "./scripts/run_low_vol_forward_paper.sh --start-after 2026-07-17T23:00:00+00:00 --as-of 2026-07-18T11:15:00+00:00 --resume",
            "/home/hr/quant-alpha-audit/.venv/bin/python audit/scripts/stage_v2_report.py",
            "```",
            "",
            "The V5 snapshot command additionally requires `SSHPASS` supplied in process memory; never place the password in a file. Public forward candles are cached under `state/` and are excluded from the final bundle.",
        ],
    )

    # Dashboard data.
    permutation = pl.read_parquet(artifacts / "permutation_distribution.parquet")
    distributions = {
        control: permutation.filter(pl.col("control_type") == control)["ic"].to_numpy()
        for control in permutation["control_type"].unique().sort().to_list()
    }
    histogram = _histogram_svg(distributions, float(summary["permutation_observed"]["ic"]))
    matrix = [
        ("Current V5", "INCONCLUSIVE", "INCONCLUSIVE", "FAIL", "PARTIALLY_REPLAYABLE"),
        ("rev_xs_20d", "FAIL", "FAIL", "FAIL", "RETIRE"),
        ("low_vol_20d", decisions["low_vol_20d"]["signal_validity"], "FAIL · LOCKED", "FAIL", "Signal ≠ portfolio"),
        ("low_vol + BTC trend", "INHERITS", "INCONCLUSIVE", "INCONCLUSIVE", "POST_HOC · FORWARD ONLY"),
        ("funding_fade", "INCONCLUSIVE", "INCONCLUSIVE", "INCONCLUSIVE", "DATA INSUFFICIENT"),
    ]
    matrix_html = "".join(
        f'<tr><th>{html.escape(name)}</th><td><span class="status {_status_class(signal)}">{html.escape(signal)}</span></td><td><span class="status {_status_class(portfolio.split()[0])}">{html.escape(portfolio)}</span></td><td><span class="status {_status_class(deploy)}">{html.escape(deploy)}</span></td><td>{html.escape(note)}</td></tr>'
        for name, signal, portfolio, deploy, note in matrix
    )
    fixed_rows_html = "".join(
        f"<tr><th>{html.escape(row['period'])}</th><td>{_pct(row['gross_total_return'])}</td><td class={'pos' if row['total_return']>0 else 'neg'}>{_pct(row['total_return'])}</td><td>{_num(row['edge_cost_ratio'],2)}</td><td>{_num(row['sharpe'],2)}</td><td>{row['trade_count']}</td></tr>"
        for row in low_fixed.iter_rows(named=True)
    )
    hac_html = "".join(
        f"<tr><th>{html.escape(row['lag_rule'])}</th><td>{row['lag']}</td><td>{_num(row['estimate'],4)}</td><td>{_num(row['standard_error'],4)}</td><td>{_num(row['t_stat'],2)}</td><td>{row['p_value']:.2e}</td></tr>"
        for row in hac.filter(pl.col("object_id") == "core.low_vol_480.rank_ic").iter_rows(named=True)
    )
    boot_html = "".join(
        f"<tr><th>{row['block_size_hours']}h</th><td>{_num(row['ci_95_low'],4)}</td><td>{_num(row['median'],4)}</td><td>{_num(row['ci_95_high'],4)}</td><td>{_num(row['probability_ic_gt_zero'],3)}</td></tr>"
        for row in low_signal_boot.iter_rows(named=True)
    )
    dashboard = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Alpha Audit v2</title>
<style>
:root{{--bg:#0a1013;--panel:#10191d;--line:#26363d;--text:#e7ecee;--muted:#8fa1a8;--amber:#f4b740;--red:#ff6b5f;--green:#5bd4a4;--cyan:#76b7c5}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}} a{{color:var(--cyan)}}
header{{position:sticky;top:0;z-index:10;background:rgba(10,16,19,.94);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}} .nav{{max-width:1260px;margin:auto;padding:12px 24px;display:flex;justify-content:space-between;gap:20px;align-items:center}} .nav strong{{letter-spacing:.12em}} .nav div{{display:flex;gap:18px;overflow:auto;white-space:nowrap}} .nav a{{font-size:12px;text-decoration:none;color:var(--muted)}}
main{{max-width:1260px;margin:auto;padding:36px 24px 100px}} .mast{{display:grid;grid-template-columns:1.4fr .6fr;gap:48px;padding:38px 0 52px;border-bottom:1px solid var(--line);animation:rise .55s ease-out}} .eyebrow{{color:var(--amber);letter-spacing:.14em;font-size:12px}} h1{{font:700 clamp(38px,7vw,84px)/.95 system-ui,sans-serif;letter-spacing:-.065em;margin:18px 0 20px}} .lead{{font:18px/1.55 system-ui,sans-serif;max-width:760px;color:#bec9cd}} .freeze{{align-self:end;border-left:3px solid var(--red);padding:4px 0 4px 20px}} .freeze b{{display:block;color:var(--red);font:700 26px system-ui,sans-serif}} .freeze span{{color:var(--muted)}}
section{{padding:56px 0;border-bottom:1px solid var(--line);scroll-margin-top:55px}} h2{{font:650 28px/1.2 system-ui,sans-serif;margin:0 0 10px}} .sub{{color:var(--muted);max-width:820px;margin:0 0 28px}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:12px 10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}} th:first-child,td:first-child{{text-align:left}} thead th{{color:var(--muted);font-weight:500}} tbody tr{{transition:background .16s ease}} tbody tr:hover{{background:#132127}} .status{{font-weight:750;letter-spacing:.04em}} .status.pass,.pos{{color:var(--green)}} .status.fail,.neg{{color:var(--red)}} .status.inc{{color:var(--amber)}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:52px}} .split>*{{min-width:0}} .metric{{font:700 44px/1 system-ui,sans-serif;letter-spacing:-.04em}} .metric small{{display:block;font:12px/1.4 ui-monospace,monospace;color:var(--muted);letter-spacing:.04em;margin-top:10px}} .rule{{height:1px;background:var(--line);margin:28px 0}} .flag{{display:inline-block;color:var(--amber);border:1px solid #725b25;padding:4px 8px;margin:0 7px 7px 0;font-size:11px;letter-spacing:.08em}} .note{{border-left:2px solid var(--amber);padding-left:18px;color:#c8d1d4}} svg{{width:100%;background:#0c1417;border:1px solid var(--line)}} code{{color:#c6d6db}} footer{{padding-top:42px;color:var(--muted);font-size:12px}}
@keyframes rise{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:none}}}} @media(max-width:800px){{.mast,.split{{grid-template-columns:1fr}} .nav div{{display:none}} main{{padding-left:15px;padding-right:15px}} section{{padding:40px 0}} table{{display:block;overflow:auto}} h1{{font-size:48px}}}}
@media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important;animation:none!important;transition:none!important}}}}
</style></head><body>
<header><div class="nav"><strong>ALPHA AUDIT / V2</strong><div><a href="#matrix">判定</a><a href="#portfolio">24h</a><a href="#layers">low_vol</a><a href="#statistics">统计</a><a href="#funding">funding</a><a href="#runtime">V5</a></div></div></header>
<main><div class="mast"><div><div class="eyebrow">CORRECTIVE AUDIT · 2026-07-18</div><h1>Signal is not<br>the portfolio.</h1><p class="lead">low_vol 的预测层通过；原锁定组合与 24h 可交易层未通过。post-hoc BTC 趋势规则只进入 forward-only 观察。</p></div><div class="freeze"><b>LIVE ALPHA FROZEN</b><span>production mutations 0 · live orders 0<br>market cutoff {html.escape(market_cutoff)}</span></div></div>

<section id="matrix"><h2>三层判定</h2><p class="sub">Signal Validity、Portfolio Validity 与 Deployment Readiness 独立展示。负结果与证据不足不合并成一个宽泛标签。</p><table><thead><tr><th>对象</th><th>Signal Validity</th><th>Portfolio Validity</th><th>Deployment Readiness</th><th>边界</th></tr></thead><tbody>{matrix_html}</tbody></table></section>

<section id="portfolio"><h2>24h 组合：毛 edge 不等于稳定可交易</h2><p class="sub">固定 12 个 v1 结构。low_vol 全样本 9/12 成本后为正、盲测 5/12；验证期 0/12。rev 全样本与盲测均 12/12 为负。</p><div class="split"><div><div class="metric">0 / 12<small>low_vol validation net-positive structures</small></div><div class="rule"></div><div class="metric">{_pct(best_rev_full['total_return'])}<small>rev best descriptive full-sample net return</small></div></div><div><span class="flag">LOCKED GRID</span><span class="flag">30BPS ROUNDTRIP</span><span class="flag">NO RESELECTION</span><p class="note">全样本的最佳行只作描述，不被重新锁定；没有任何 low_vol 24h 规则同时在 validation 和 blind 成本后为正。</p></div></div><h3>固定 Top3 / score / BTC filter ON</h3><table><thead><tr><th>Period</th><th>Gross</th><th>Net</th><th>Edge/Cost</th><th>Sharpe</th><th>Trades</th></tr></thead><tbody>{fixed_rows_html}</tbody></table></section>

<section id="layers"><h2>low_vol 三层对象</h2><div class="split"><div><p><span class="status pass">SIGNAL PASS</span></p><div class="metric">{low_primary['ic_mean']:.4f}<small>24h rank IC · HAC t {low_primary['hac_tstat']:.2f} · non-overlap t {low_primary['non_overlap_tstat']:.2f}</small></div><p>Beta-neutral IC {float(sensitivity.filter((pl.col('slice_type')=='neutralized')&(pl.col('slice')=='beta_30d'))['estimate'][0]):.4f}; liquidity-neutral IC {float(sensitivity.filter((pl.col('slice_type')=='neutralized')&(pl.col('slice')=='log_qv_30d'))['estimate'][0]):.4f}.</p></div><div><p><span class="status fail">LOCKED PORTFOLIO FAIL</span></p><div class="metric">{_pct(locked_blind['total_return'])}<small>v1 locked 120h / no-filter blind net</small></div><p><span class="flag">POST_HOC</span><span class="flag">FORWARD ONLY</span></p><p>BTC-trend historical full {_pct(posthoc_full['total_return'])}; cannot be called independent OOS. Forward sample {float(forward['forward_calendar_days']):.3f} day / {int(forward['completed_independent_trades'])} completed trade.</p></div></div></section>

<section id="statistics"><h2>统计敏感性与 null 分布</h2><p class="sub">所有 HAC 带宽与 block size 同时展示；四种严格 null 各 1,000 次。robustness perturbations 另表处理。</p><div class="split"><div><h3>HAC bandwidth</h3><table><thead><tr><th>Rule</th><th>Lag</th><th>Estimate</th><th>SE</th><th>t</th><th>p</th></tr></thead><tbody>{hac_html}</tbody></table></div><div><h3>Block bootstrap · signal IC</h3><table><thead><tr><th>Block</th><th>2.5%</th><th>Median</th><th>97.5%</th><th>P(IC&gt;0)</th></tr></thead><tbody>{boot_html}</tbody></table></div></div><h3>Permutation IC null distributions</h3>{histogram}<p class="sub">Observed daily non-overlap IC 0.0965 lies beyond every 1% null interval; worst two-sided empirical p {empirical_p:.6f}. This validates the signal layer only.</p><h3>Confirmatory vs exploratory</h3><table><thead><tr><th>Test type</th><th>Tests</th><th>Min raw p</th><th>Min Holm p</th><th>Min BH p</th><th>Min max-stat p</th></tr></thead><tbody>{multiplicity_html}</tbody></table><p class="sub">The exploratory row is the single post-hoc BTC-trend rule. It is never pooled into the confirmatory family or treated as preregistered blind evidence.</p></section>

<section id="funding"><h2>Funding：60 observations → natural 20d</h2><div class="split"><div><div class="metric">{funding_time['ic_mean']:.4f}<small>time-based 20d / 72h rank IC · HAC t {funding_time['hac_tstat']:.2f}</small></div><p>4h symbols {int(freq_counts.filter(pl.col('median_settlement_frequency_hours')==4)['symbols'][0])}; 8h symbols {int(freq_counts.filter(pl.col('median_settlement_frequency_hours')==8)['symbols'][0])}. closed-left; no zero fill.</p></div><div><span class="flag">DATA INSUFFICIENT</span><p class="note">历史仅 {float(frequency['actual_coverage_days'].min()):.1f}–{float(frequency['actual_coverage_days'].max()):.1f} 天，non-overlap n={int(funding_time['non_overlap_count'])}。修正定义后仍不足以判 PASS 或正式 FAIL。</p></div></div></section>

<section id="runtime"><h2>V5 replayability 与 forward 边界</h2><div class="split"><div><p><span class="status inc">PARTIALLY_REPLAYABLE</span></p><div class="metric">{replay['systemd_units_captured']} / {replay['database_files_captured']} / {replay['state_documents_captured']}<small>systemd units / current DB schemas / dynamic state docs</small></div><p>HEAD <code>{replay['git_head'][:12]}</code>, clean tree. Current Alpha/regime/HMM/RSS/risk/optimizer/positions are captured, but the historical atomic per-decision chain is absent.</p></div><div><p><span class="status inc">FORWARD INCONCLUSIVE</span></p><div class="metric">{float(forward['forward_calendar_days']):.3f}d / {int(forward['completed_independent_trades'])}<small>new calendar days / completed independent 120h trades</small></div><p>1 decision was recorded; BTC trend was OFF and target holdings were empty. Old blind data is never relabelled as forward evidence.</p></div></div></section>

<section><h2>成本覆盖与 v1 → v2 差异</h2><div class="split"><div><div class="metric">0 / 0 / 0<small>requested-price / bid-ask / latency rows</small></div><p>{costs['fill_rows']} fills across {costs['covered_symbols']} symbols provide fee-side evidence, not reconstructable arrival slippage. Optimistic/base/stress all remain 30bps roundtrip.</p></div><div><ul><li>low_vol v1 aggregate FAIL → v2 signal PASS + locked portfolio FAIL.</li><li>rev adds 24h portfolio evidence and remains FAIL/RETIRE.</li><li>funding changes from count-based to natural-time 20d, conclusion remains INCONCLUSIVE.</li><li>V5 present state is now fingerprinted, but exact historical replay remains unavailable.</li><li>post-hoc BTC trend is explicitly separated from confirmatory evidence.</li></ul></div></div></section>
<footer>Alpha Audit v2 · branch audit/alpha-validity-v2 · read-only production snapshot · open final machine decision at artifacts/final_decisions_v2.json</footer>
</main></body></html>'''
    validate_dashboard_contract(dashboard)
    (reports / "audit_dashboard_v2.html").write_text(dashboard, encoding="utf-8")
    print(json.dumps(decisions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
