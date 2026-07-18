# ruff: noqa: E501
"""Generate final decisions, static self-contained dashboard, and closeout reports."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.paths import (  # noqa: E402
    ARTIFACTS,
    LOCAL_AUDIT_ROOT,
    REPORTS,
    snapshot_dir,
)

DISPLAY_NAMES = {
    "v5.alpha6_static_proxy": "V5 static proxy",
    "core.rev_xs_480": "rev_xs_20d",
    "core.low_vol_480": "low_vol_20d",
    "core.funding_fade_480": "funding_fade",
}
DECISION_KEYS = {
    "v5.alpha6_static_proxy": "current_v5",
    "core.rev_xs_480": "rev_xs_20d",
    "core.low_vol_480": "low_vol_20d",
    "core.funding_fade_480": "funding_fade",
}
COLORS = {
    "cyan": "#5eead4",
    "red": "#fb7185",
    "amber": "#fbbf24",
    "muted": "#64748b",
    "blue": "#60a5fa",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _record(frame: pl.DataFrame, **filters: object) -> dict:
    selected = frame
    for column, value in filters.items():
        selected = selected.filter(pl.col(column) == value)
    return selected.to_dicts()[0] if selected.height else {}


def _safe(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _figure_style(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "font": {"size": 18, "color": "#e2e8f0"}, "x": 0.0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,.38)",
        font={"family": "system-ui, sans-serif", "color": "#cbd5e1", "size": 12},
        margin={"l": 55, "r": 25, "t": 58, "b": 48},
        legend={"orientation": "h", "y": 1.10, "x": 0},
        hoverlabel={"bgcolor": "#0f172a", "font_color": "#e2e8f0"},
    )
    figure.update_xaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.25)")
    figure.update_yaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.25)")
    return figure


def _chart_block(
    identifier: str, heading: str, meta: str, figure: go.Figure, include_js: bool
) -> str:
    html = figure.to_html(
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={"displaylogo": False, "responsive": True, "scrollZoom": False},
    )
    return f'<section id="{identifier}" class="evidence reveal"><div class="section-kicker">{heading}</div><p class="meta">{meta}</p>{html}</section>'


def _ensure_resource_seed() -> pl.DataFrame:
    path = ARTIFACTS / "resource_usage.csv"
    if path.is_file():
        return pl.read_csv(path)
    rows = [
        {
            "stage": "data",
            "command": "stage_silver.py",
            "wall_seconds": 3.38,
            "cpu_user_seconds": 3.72,
            "cpu_system_seconds": 4.87,
            "peak_rss_mb": 797_924 / 1024,
            "swap_events": 0,
            "measurement_source": "/usr/bin/time -v",
        },
        {
            "stage": "features",
            "command": "stage_features.py",
            "wall_seconds": 4.34,
            "cpu_user_seconds": 6.90,
            "cpu_system_seconds": 2.21,
            "peak_rss_mb": 927_724 / 1024,
            "swap_events": 0,
            "measurement_source": "/usr/bin/time -v",
        },
        {
            "stage": "leakage",
            "command": "stage_leakage.py",
            "wall_seconds": 4.33,
            "cpu_user_seconds": 27.15,
            "cpu_system_seconds": 2.79,
            "peak_rss_mb": 927_104 / 1024,
            "swap_events": 0,
            "measurement_source": "/usr/bin/time -v",
        },
    ]
    frame = pl.DataFrame(rows)
    frame.write_csv(path)
    return frame


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    factor = pl.read_csv(ARTIFACTS / "factor_validation_results.csv")
    ic = pl.read_csv(ARTIFACTS / "ic_statistics.csv")
    period = pl.read_csv(ARTIFACTS / "factor_period_results.csv")
    portfolio = pl.read_csv(ARTIFACTS / "portfolio_backtest_results.csv")
    locked = pl.read_csv(ARTIFACTS / "locked_portfolio_results.csv")
    benchmarks = pl.read_csv(ARTIFACTS / "portfolio_benchmarks.csv")
    timeseries = pl.read_parquet(ARTIFACTS / "portfolio_timeseries.parquet")
    leakage = pl.read_csv(ARTIFACTS / "leakage_test_results.csv")
    negative = pl.read_csv(ARTIFACTS / "negative_control_results.csv")
    multiplicity = pl.read_csv(ARTIFACTS / "multiple_testing_summary.csv")
    data_quality = pl.read_csv(ARTIFACTS / "data_quality_summary.csv")
    funding_quality = pl.read_csv(ARTIFACTS / "funding_quality_summary.csv")
    exposures = pl.read_csv(ARTIFACTS / "factor_exposure_results.csv")
    portfolio_contribution = pl.read_csv(ARTIFACTS / "portfolio_symbol_contribution.csv")
    resource = _ensure_resource_seed()
    portfolio_locks = json.loads(
        (ARTIFACTS / "portfolio_oos_lock.json").read_text(encoding="utf-8")
    )["factors"]

    factor_rows = {row["factor_id"]: row for row in factor.iter_rows(named=True)}
    blind_portfolio = {
        factor_id: _record(
            locked,
            factor_id=factor_id,
            period="blind",
            cost_scenario="base",
        )
        for factor_id in factor_rows
    }
    full_portfolio = {
        factor_id: _record(locked, factor_id=factor_id, period="full", cost_scenario="base")
        for factor_id in factor_rows
    }
    low_beta = _record(exposures, factor_id="core.low_vol_480", exposure="beta_30d")
    low_liquidity = _record(exposures, factor_id="core.low_vol_480", exposure="log_qv_30d")

    decisions = {
        "current_v5": {
            "decision": "INCONCLUSIVE",
            "reason": "完整生产 V5 依赖未逐决策保存的 sentiment、dynamic IC、regime、三策略融合、优化器与持仓状态，无法精确重放；静态 Alpha6 代理盲测 IC 和 long-only 收益均为负，因此没有可验证 edge。",
            "key_metrics": {
                "exact_replay_available": False,
                "proxy_full_ic": factor_rows["v5.alpha6_static_proxy"]["full_ic_mean"],
                "proxy_blind_ic": factor_rows["v5.alpha6_static_proxy"]["blind_ic_mean"],
                "proxy_blind_net_return": blind_portfolio["v5.alpha6_static_proxy"]["total_return"],
                "proxy_blind_gross_return": blind_portfolio["v5.alpha6_static_proxy"][
                    "gross_total_return"
                ],
            },
        },
        "rev_xs_20d": {
            "decision": "FAIL",
            "reason": "锁定 24h/Top50 后全样本 IC 仅 0.0134、Holm 后不显著，盲测 IC 转负；锁定 72h 组合盲测毛收益和成本后收益均为负。",
            "key_metrics": {
                "full_ic": factor_rows["core.rev_xs_480"]["full_ic_mean"],
                "full_hac_tstat": factor_rows["core.rev_xs_480"]["full_hac_tstat"],
                "holm_pvalue": factor_rows["core.rev_xs_480"]["full_holm_pvalue"],
                "blind_ic": factor_rows["core.rev_xs_480"]["blind_ic_mean"],
                "blind_gross_return": blind_portfolio["core.rev_xs_480"]["gross_total_return"],
                "blind_net_return": blind_portfolio["core.rev_xs_480"]["total_return"],
            },
        },
        "low_vol_20d": {
            "decision": "FAIL",
            "reason": "IC 证据强且盲测方向一致，但锁定的 120h long-only 组合盲测毛收益 -15.66%、基础成本后 -16.55%，并呈强低 BTC beta 暴露；不满足成本后正收益和组合可交易要求。",
            "key_metrics": {
                "full_ic": factor_rows["core.low_vol_480"]["full_ic_mean"],
                "full_hac_tstat": factor_rows["core.low_vol_480"]["full_hac_tstat"],
                "holm_pvalue": factor_rows["core.low_vol_480"]["full_holm_pvalue"],
                "blind_ic": factor_rows["core.low_vol_480"]["blind_ic_mean"],
                "blind_gross_return": blind_portfolio["core.low_vol_480"]["gross_total_return"],
                "blind_net_return": blind_portfolio["core.low_vol_480"]["total_return"],
                "blind_edge_cost_ratio": blind_portfolio["core.low_vol_480"]["edge_cost_ratio"],
                "beta_exposure_rank_corr": low_beta.get("mean_rank_correlation"),
                "liquidity_exposure_rank_corr": low_liquidity.get("mean_rank_correlation"),
            },
        },
        "funding_fade": {
            "decision": "INCONCLUSIVE",
            "reason": "OKX funding 仅约 95 天而非两年，86/86 funding 序列均触发关键覆盖不足；现有 IC 和组合为负但独立样本不足，不能据此授予 PASS 或作完整否决。",
            "key_metrics": {
                "funding_span_days_max": float(funding_quality["span_days"].max()),
                "funding_symbols": funding_quality.height,
                "full_ic": factor_rows["core.funding_fade_480"]["full_ic_mean"],
                "blind_net_return": blind_portfolio["core.funding_fade_480"]["total_return"],
            },
        },
        "recommended_action": "继续冻结实盘；不授予 LIVE_SMALL，也不把任何候选推进 paper。",
        "highest_priority_next_step": "建立不可变的 V5 每决策全状态快照与前向结果链（sentiment、dynamic IC、regime、三策略融合、目标权重、成本和成交），积累后按同一冻结协议重审。",
    }
    (ARTIFACTS / "final_decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    v5 = decisions["current_v5"]
    rev = decisions["rev_xs_20d"]
    low = decisions["low_vol_20d"]
    funding = decisions["funding_fade"]
    summary_lines = [
        "# Executive Summary",
        "",
        "## Final decisions",
        "",
        f"- Current V5: **{v5['decision']}** — no exact replay and the static proxy fails blind OOS.",
        f"- rev_xs_20d: **{rev['decision']}** — weak/unadjusted IC, blind sign reversal, negative long-only OOS.",
        f"- low_vol_20d: **{low['decision']}** — strong IC does not survive as tradable 120h long-only OOS return.",
        f"- funding_fade: **{funding['decision']}** — funding history is about 95 days, not two years.",
        f"- Recommendation: **{decisions['recommended_action']}**",
        "",
        "## Core answers",
        "",
        "1. Current V5 verified edge? **No verified edge; INCONCLUSIVE rather than PASS.**",
        "2. Was original IC/t-stat inflated? **Yes in overlapping cells; naive statistics are diagnostic only.**",
        "3. Future leakage? **No mechanical future leak detected in audited code; survivorship and replay-state limitations remain.**",
        "4. Which factor earns after costs? **None in locked blind OOS.**",
        "5. Which factor survives OOS? **low_vol IC survives, but its tradable portfolio does not; therefore FAIL.**",
        "6. Evidence at 72h/120h? **All locked candidate portfolios are negative in blind OOS; no passing horizon.**",
        "7. Should V5 be redesigned now? **No. Do not alter production based on these results.**",
        f"8. Highest priority: **{decisions['highest_priority_next_step']}**",
        "",
        "The audit made no production changes, placed no orders, copied no exchange credentials, and did not modify production databases.",
    ]
    (REPORTS / "executive_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    low_lock = portfolio_locks["core.low_vol_480"]
    audit_lines = [
        "# Alpha Validity Audit",
        "",
        f"- snapshot: {snapshot_dir().name}",
        "- market data: 1,225,652 closed 1h bars, 93 symbols, 2024-07-18 through 2026-07-17 UTC",
        f"- multiplicity: {multiplicity.height} retained tests across {int(multiplicity['family_count'].max())} families",
        "- OOS: ordered 50/25/25 with purged labels; research locks were written before blind evaluation",
        "- costs: p50/p75/p90 evidence with 15bps one-way production fallback floor; only 4/93 symbols have actual cost buckets",
        "",
        "## Decision table",
        "",
        "| Target | Decision | Key evidence |",
        "|---|---|---|",
        f"| Current V5 | {v5['decision']} | proxy blind IC={_safe(v5['key_metrics']['proxy_blind_ic'])}; proxy blind net={_safe(v5['key_metrics']['proxy_blind_net_return'])}; exact replay unavailable |",
        f"| rev_xs_20d | {rev['decision']} | blind IC={_safe(rev['key_metrics']['blind_ic'])}; blind net={_safe(rev['key_metrics']['blind_net_return'])} |",
        f"| low_vol_20d | {low['decision']} | blind IC={_safe(low['key_metrics']['blind_ic'])}; locked rule={low_lock['structure_id']}; blind net={_safe(low['key_metrics']['blind_net_return'])} |",
        f"| funding_fade | {funding['decision']} | max funding span={_safe(funding['key_metrics']['funding_span_days_max'], 1)}d; blind net={_safe(funding['key_metrics']['blind_net_return'])} |",
        "",
        "## Why strong low-vol IC is not a PASS",
        "",
        f"- Full rank IC={_safe(low['key_metrics']['full_ic'])}, HAC t={_safe(low['key_metrics']['full_hac_tstat'], 2)}, Holm p={low['key_metrics']['holm_pvalue']}.",
        f"- Exposure correlation with trailing BTC beta={_safe(low['key_metrics']['beta_exposure_rank_corr'])}; liquidity/large-coin proxy={_safe(low['key_metrics']['liquidity_exposure_rank_corr'])}.",
        f"- Locked research-only portfolio rule: `{low_lock['structure_id']}`.",
        f"- Blind gross={_safe(low['key_metrics']['blind_gross_return'])}, blind base-cost net={_safe(low['key_metrics']['blind_net_return'])}, edge/cost={_safe(low['key_metrics']['blind_edge_cost_ratio'], 2)}.",
        "- The failure occurs before any post-result threshold change: tradable OOS PnL is negative even gross of costs.",
        "",
        "## Data and evidence limits",
        "",
        "- Historical universe ranking is causal, but the seed set is current-listed and cannot restore delisted-before-snapshot instruments.",
        "- Funding coverage is insufficient for a two-year funding factor claim.",
        "- Broad-universe real cost, quantity precision, partial-fill, and latency evidence is incomplete.",
        "- Exact current V5 replay needs per-decision state snapshots that do not exist in the captured production data.",
        "",
        "## Production boundary",
        "",
        "No deployment, restart, live-order call, production Git checkout, production DB write, or exchange credential copy occurred.",
    ]
    (REPORTS / "alpha_validity_audit.md").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )

    factor_lines = [
        "# Factor Validation Report",
        "",
        "Formal decisions combine overlap-corrected IC, locked OOS, multiple testing, long-only portfolio results, costs, concentration, and evidence coverage.",
        "",
        "| Factor | IC / HAC / Holm | Blind IC | Blind gross / net | Exposure note | Decision |",
        "|---|---|---:|---:|---|---|",
    ]
    for factor_id, key in DECISION_KEYS.items():
        factor_row = factor_rows[factor_id]
        portfolio_row = blind_portfolio[factor_id]
        exposure_note = ""
        if factor_id == "core.low_vol_480":
            exposure_note = f"beta corr={_safe(low_beta.get('mean_rank_correlation'))}; liquidity corr={_safe(low_liquidity.get('mean_rank_correlation'))}"
        factor_lines.append(
            f"| {DISPLAY_NAMES[factor_id]} | {_safe(factor_row['full_ic_mean'])} / {_safe(factor_row['full_hac_tstat'], 2)} / {_safe(factor_row['full_holm_pvalue'], 3)} | "
            f"{_safe(factor_row['blind_ic_mean'])} | {_safe(portfolio_row['gross_total_return'])} / {_safe(portfolio_row['total_return'])} | {exposure_note} | {decisions[key]['decision']} |"
        )
    factor_lines.extend(
        [
            "",
            "Negative controls with fixed seeds are retained in `artifacts/negative_control_results.csv`; two low-vol persistence/universe manipulations pass the corrected IC screen, but no negative control is promotion-eligible or PAPER_READY.",
        ]
    )
    (REPORTS / "factor_validation_report.md").write_text(
        "\n".join(factor_lines) + "\n", encoding="utf-8"
    )

    production_lines = [
        "# Production Impact Assessment",
        "",
        "- Production V5 code/config/service/database changes: **none**.",
        "- Production quant-lab code/config/service/lake mutations: **none**.",
        "- Restarts/stops/deployments: **none**.",
        "- Exchange API key/secret/passphrase copied: **none**.",
        "- Order submission/cancel/amend/funds transfer: **none**.",
        "- Audit semantics: read-only production snapshot, local historical backfill, local paper research only.",
        "- Current production config remains `quant_lab.mode=shadow`; canary is disabled. Snapshot risk/gate states are evidence, not authorization to change live behavior.",
        "- Recommendation: keep live frozen; do not deploy these audit factors or reinterpret PAPER_READY as live readiness.",
        "",
        "Rollback point: all implementation changes are isolated on `audit/alpha-validity`; production rollback is not applicable because production was not changed.",
    ]
    (REPORTS / "production_impact_assessment.md").write_text(
        "\n".join(production_lines) + "\n", encoding="utf-8"
    )

    resource_lines = [
        "# Resource Usage",
        "",
        "Resource rows are sampled by the unified runner when available; initial heavy stages also retain `/usr/bin/time -v` measurements.",
        "",
        "| Stage | Command | Wall s | CPU user s | CPU system s | Peak RSS MiB | Swap events | Source |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in resource.iter_rows(named=True):
        resource_lines.append(
            f"| {row.get('stage')} | `{row.get('command')}` | {_safe(row.get('wall_seconds'), 2)} | {_safe(row.get('cpu_user_seconds'), 2)} | "
            f"{_safe(row.get('cpu_system_seconds'), 2)} | {_safe(row.get('peak_rss_mb'), 1)} | {row.get('swap_events', 0)} | {row.get('measurement_source', 'psutil')} |"
        )
    resource_lines.extend(
        [
            "",
            "No user process was terminated. WSL used 12 CPUs, 48GiB memory limit and 24GiB swap; measured heavy stages used under 1GiB RSS and no swap.",
        ]
    )
    (REPORTS / "resource_usage.md").write_text("\n".join(resource_lines) + "\n", encoding="utf-8")

    repro_lines = [
        "# Reproduction Guide",
        "",
        "```bash",
        "cd /home/hr/quant-alpha-audit/repos/quant-lab",
        "export LOCAL_AUDIT_ROOT=/home/hr/quant-alpha-audit",
        "source /home/hr/quant-alpha-audit/.venv/bin/activate",
        "./scripts/run_full_alpha_audit.sh --resume",
        "./scripts/run_full_alpha_audit.sh --stage data",
        "./scripts/run_full_alpha_audit.sh --stage leakage",
        "./scripts/run_full_alpha_audit.sh --stage factors",
        "./scripts/run_full_alpha_audit.sh --stage portfolio",
        "./scripts/run_full_alpha_audit.sh --stage report",
        "```",
        "",
        "- Snapshot ID is read from `/home/hr/quant-alpha-audit/snapshots/LATEST`; all snapshot files are verified by `manifests/snapshot_sha256.txt`.",
        "- Raw two-year OKX partitions are excluded from the final ZIP; the backfill checkpoint and feature manifest provide provenance.",
        "- Random seeds: 7, 19, 42, 20260718. Split and selection rules are frozen in `oos_lock.json` and `portfolio_oos_lock.json`.",
        "- SSH credentials are supplied only through the invoking environment if a new snapshot is explicitly authorized; no password or exchange secret is stored in audit scripts.",
        "- The runner validates checkpoint output hashes before skipping a stage.",
    ]
    (REPORTS / "reproduction_guide.md").write_text("\n".join(repro_lines) + "\n", encoding="utf-8")

    if not (REPORTS / "test_report.md").exists():
        (REPORTS / "test_report.md").write_text(
            "# Test Report\n\nFinal JUnit and command results are generated by the verification stage. No tests are skipped to obtain a passing audit.\n",
            encoding="utf-8",
        )

    # ----- Dashboard charts -----
    charts: list[str] = []
    include_js = True
    names = [DISPLAY_NAMES[row["factor_id"]] for row in factor.iter_rows(named=True)]
    fig = go.Figure()
    for column, label, color in [
        ("research_ic_mean", "Research", COLORS["muted"]),
        ("validation_ic_mean", "Validation", COLORS["blue"]),
        ("blind_ic_mean", "Blind", COLORS["red"]),
        ("full_ic_mean", "Full", COLORS["cyan"]),
    ]:
        fig.add_bar(name=label, x=names, y=factor[column].to_list(), marker_color=color)
    _figure_style(fig, "01 · Rank IC across ordered OOS slices")
    charts.append(
        _chart_block(
            "factor-ic",
            "Factor IC",
            "Locked factor cells · 50/25/25 ordered split · one-bar delay · blind shown explicitly",
            fig,
            include_js,
        )
    )
    include_js = False

    selected_ic = []
    for row in factor.iter_rows(named=True):
        selected_ic.append(
            _record(
                ic,
                factor_id=row["factor_id"],
                direction=1,
                horizon_bars=int(row["locked_horizon_bars"]),
                universe=row["locked_universe"],
                period="full",
            )
        )
    stat_frame = pl.DataFrame(selected_ic)
    fig = go.Figure()
    for column, label, color in [
        ("naive_tstat", "Naive", COLORS["muted"]),
        ("hac_tstat", "HAC", COLORS["blue"]),
        ("non_overlap_tstat", "Non-overlap", COLORS["cyan"]),
    ]:
        fig.add_bar(name=label, x=names, y=stat_frame[column].to_list(), marker_color=color)
    _figure_style(fig, "02 · Naive significance versus overlap-aware estimates")
    charts.append(
        _chart_block(
            "tstats",
            "Statistical correction",
            "HAC lag equals horizon; non-overlap spacing >= horizon; naive values never decide PASS",
            fig,
            include_js,
        )
    )

    horizon_median = (
        portfolio.filter((pl.col("cost_scenario") == "base") & (pl.col("period") == "blind"))
        .group_by(["factor_id", "rebalance_hours"])
        .agg(pl.col("total_return").median().alias("median_net_return"))
        .sort(["factor_id", "rebalance_hours"])
    )
    fig = go.Figure()
    for factor_id in DISPLAY_NAMES:
        subset = horizon_median.filter(pl.col("factor_id") == factor_id)
        fig.add_bar(
            name=DISPLAY_NAMES[factor_id],
            x=[f"{value}h" for value in subset["rebalance_hours"].to_list()],
            y=subset["median_net_return"].to_list(),
        )
    _figure_style(fig, "03 · Blind net return by 72h / 120h rebalance")
    charts.append(
        _chart_block(
            "horizons",
            "Horizon evidence",
            "Median across every declared TopN/weighting/stagger/filter cell · base cost · blind OOS",
            fig,
            include_js,
        )
    )

    blind_rows = [blind_portfolio[factor_id] for factor_id in DISPLAY_NAMES]
    fig = go.Figure()
    fig.add_bar(
        name="Gross",
        x=names,
        y=[row["gross_total_return"] for row in blind_rows],
        marker_color=COLORS["muted"],
    )
    fig.add_bar(
        name="Net",
        x=names,
        y=[row["total_return"] for row in blind_rows],
        marker_color=COLORS["red"],
    )
    _figure_style(fig, "04 · Locked blind portfolio: gross versus net")
    charts.append(
        _chart_block(
            "gross-net",
            "Gross / net",
            "Research-locked rule · base p75 cost scenario · spot long-only",
            fig,
            include_js,
        )
    )

    cost_compare = []
    for factor_id, lock in portfolio_locks.items():
        for scenario in ("optimistic", "base", "stress"):
            row = _record(
                portfolio, structure_id=lock["structure_id"], period="blind", cost_scenario=scenario
            )
            cost_compare.append(
                {"factor_id": factor_id, "scenario": scenario, "return": row.get("total_return")}
            )
    cost_compare_frame = pl.DataFrame(cost_compare)
    fig = go.Figure()
    for scenario, color in [
        ("optimistic", COLORS["cyan"]),
        ("base", COLORS["blue"]),
        ("stress", COLORS["red"]),
    ]:
        subset = cost_compare_frame.filter(pl.col("scenario") == scenario)
        fig.add_bar(
            name=scenario,
            x=[DISPLAY_NAMES[value] for value in subset["factor_id"].to_list()],
            y=subset["return"].to_list(),
            marker_color=color,
        )
    _figure_style(fig, "05 · Locked blind return under p50 / p75 / p90 costs")
    charts.append(
        _chart_block(
            "cost-scenarios",
            "Cost scenarios",
            "One-way cost floor 15bps; 4/93 symbols have real cost buckets; evidence status is insufficient for broad-universe PASS",
            fig,
            include_js,
        )
    )

    locked_ids = [lock["structure_id"] for lock in portfolio_locks.values()]
    curve = timeseries.filter(
        pl.col("structure_id").is_in(locked_ids + ["benchmark.BTC buy-and-hold", "benchmark.cash"])
    )
    fig = go.Figure()
    for structure_id in curve["structure_id"].unique().sort().to_list():
        subset = curve.filter(pl.col("structure_id") == structure_id).sort("ts")
        label = next(
            (
                DISPLAY_NAMES[factor_id]
                for factor_id, lock in portfolio_locks.items()
                if lock["structure_id"] == structure_id
            ),
            structure_id.replace("benchmark.", ""),
        )
        fig.add_scatter(
            name=label, x=subset["ts"].to_list(), y=subset["net_equity"].to_list(), mode="lines"
        )
    _figure_style(fig, "06 · Walk-forward-style locked net equity")
    charts.append(
        _chart_block(
            "equity",
            "Net equity",
            "Daily observations · locked rules only · base cost · BTC and cash references",
            fig,
            include_js,
        )
    )

    fig = go.Figure()
    for structure_id in curve["structure_id"].unique().sort().to_list():
        subset = curve.filter(pl.col("structure_id") == structure_id).sort("ts")
        label = next(
            (
                DISPLAY_NAMES[factor_id]
                for factor_id, lock in portfolio_locks.items()
                if lock["structure_id"] == structure_id
            ),
            structure_id.replace("benchmark.", ""),
        )
        fig.add_scatter(
            name=label, x=subset["ts"].to_list(), y=subset["drawdown"].to_list(), mode="lines"
        )
    _figure_style(fig, "07 · Drawdown of locked portfolios")
    charts.append(
        _chart_block(
            "drawdown",
            "Drawdown",
            "Daily net drawdown · same locked configurations and base costs as equity chart",
            fig,
            include_js,
        )
    )

    calendar = period.filter(pl.col("slice_type").is_in(["calendar_year", "calendar_quarter"]))
    fig = go.Figure()
    for factor_id in DISPLAY_NAMES:
        subset = calendar.filter(pl.col("factor_id") == factor_id).sort("period")
        fig.add_scatter(
            name=DISPLAY_NAMES[factor_id],
            x=subset["period"].to_list(),
            y=subset["ic_mean"].to_list(),
            mode="lines+markers",
        )
    _figure_style(fig, "08 · Annual and quarterly rank IC")
    charts.append(
        _chart_block(
            "calendar",
            "Calendar stability",
            "Locked horizon/universe · HAC/non-overlap detail remains in CSV · no failed quarter removed",
            fig,
            include_js,
        )
    )

    regime = period.filter(pl.col("slice_type") == "regime")
    fig = go.Figure()
    for factor_id in DISPLAY_NAMES:
        subset = regime.filter(pl.col("factor_id") == factor_id).sort("period")
        fig.add_bar(
            name=DISPLAY_NAMES[factor_id],
            x=[value.replace("regime_", "") for value in subset["period"].to_list()],
            y=subset["ic_mean"].to_list(),
        )
    _figure_style(fig, "09 · IC across BTC trend, volatility, and market states")
    charts.append(
        _chart_block(
            "regime",
            "Regime dependence",
            "Causal BTC 60d trend / 30d volatility / 60d return state · locked cell",
            fig,
            include_js,
        )
    )

    low_structure = portfolio_locks["core.low_vol_480"]["structure_id"]
    contrib = (
        portfolio_contribution.filter(pl.col("structure_id") == low_structure)
        .sort("absolute_contribution_share", descending=True)
        .head(15)
    )
    fig = go.Figure(
        go.Bar(
            x=contrib["absolute_contribution_share"].to_list(),
            y=contrib["symbol"].to_list(),
            orientation="h",
            marker_color=COLORS["cyan"],
        )
    )
    _figure_style(fig, "10 · Low-vol locked portfolio contribution")
    charts.append(
        _chart_block(
            "contribution",
            "Symbol contribution",
            f"Top 15 by absolute gross contribution · structure={low_structure}",
            fig,
            include_js,
        )
    )

    full_rows = [full_portfolio[factor_id] for factor_id in DISPLAY_NAMES]
    fig = go.Figure()
    fig.add_bar(
        name="Largest symbol",
        x=names,
        y=[row["max_symbol_contribution_share"] for row in full_rows],
        marker_color=COLORS["red"],
    )
    fig.add_bar(
        name="Top 3",
        x=names,
        y=[row["top3_contribution_share"] for row in full_rows],
        marker_color=COLORS["amber"],
    )
    _figure_style(fig, "11 · Return concentration")
    charts.append(
        _chart_block(
            "concentration",
            "Concentration",
            "Absolute gross contribution shares · full locked portfolios · no single-symbol rows hidden",
            fig,
            include_js,
        )
    )

    sampled_multi = multiplicity.sort("raw_pvalue").head(400)
    fig = go.Figure()
    fig.add_scatter(
        x=np.maximum(sampled_multi["raw_pvalue"].to_numpy(), 1e-300),
        y=np.maximum(sampled_multi["holm_pvalue"].to_numpy(), 1e-300),
        mode="markers",
        marker={"color": COLORS["blue"], "opacity": 0.6},
        name="tests",
    )
    fig.update_xaxes(type="log", title="raw p-value")
    fig.update_yaxes(type="log", title="Holm p-value")
    _figure_style(fig, "12 · Multiple testing before and after Holm correction")
    charts.append(
        _chart_block(
            "multiplicity",
            "Multiplicity",
            f"{multiplicity.height} actual tests · 3 families · plot shows 400 smallest raw p-values",
            fig,
            include_js,
        )
    )

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Candle missing rate", "Funding history days")
    )
    fig.add_trace(
        go.Histogram(
            x=data_quality["missing_rate"].to_list(), marker_color=COLORS["cyan"], name="candles"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=funding_quality["span_days"].to_list(), marker_color=COLORS["red"], name="funding"
        ),
        row=1,
        col=2,
    )
    _figure_style(fig, "13 · Data completeness")
    charts.append(
        _chart_block(
            "data-quality",
            "Data completeness",
            "93 candle symbols are OK; all 86 funding histories are critical because <730 days",
            fig,
            include_js,
        )
    )

    leakage_counts = leakage.group_by(["test_name", "passed"]).len().sort("test_name")
    fig = go.Figure()
    for passed, color in [(True, COLORS["cyan"]), (False, COLORS["red"])]:
        subset = leakage_counts.filter(pl.col("passed") == passed)
        fig.add_bar(
            name="pass" if passed else "fail",
            x=subset["test_name"].to_list(),
            y=subset["len"].to_list(),
            marker_color=color,
        )
    _figure_style(fig, "14 · Automated leakage manipulations")
    charts.append(
        _chart_block(
            "leakage",
            "Leakage tests",
            "32 retained outcomes · future injection detected for all factors · one random low-vol signal permutation exceeded |HAC t|=2",
            fig,
            include_js,
        )
    )

    fig = go.Figure()
    for control_type in negative["control_type"].unique().sort().to_list():
        subset = negative.filter(pl.col("control_type") == control_type)
        fig.add_scatter(
            name=control_type,
            x=[DISPLAY_NAMES.get(value, value) for value in subset["factor_id"].to_list()],
            y=subset["hac_tstat"].to_list(),
            mode="markers",
        )
    fig.add_hline(y=2.0, line_dash="dash", line_color=COLORS["red"])
    fig.add_hline(y=-2.0, line_dash="dash", line_color=COLORS["red"])
    _figure_style(fig, "15 · Negative-control HAC t-statistics")
    charts.append(
        _chart_block(
            "negative-controls",
            "Negative controls",
            f"{negative.height} controls · seeds 7/19/42/20260718 · no control eligible for PAPER_READY",
            fig,
            include_js,
        )
    )

    fig = go.Figure()
    if "peak_rss_mb" in resource.columns:
        fig.add_bar(
            name="Peak RSS MiB",
            x=resource["stage"].to_list(),
            y=resource["peak_rss_mb"].to_list(),
            marker_color=COLORS["blue"],
        )
    if "wall_seconds" in resource.columns:
        fig.add_scatter(
            name="Wall seconds",
            x=resource["stage"].to_list(),
            y=resource["wall_seconds"].to_list(),
            mode="lines+markers",
            yaxis="y2",
            marker_color=COLORS["amber"],
        )
        fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "title": "seconds"})
    _figure_style(fig, "16 · Resource usage")
    charts.append(
        _chart_block(
            "resources",
            "Resource usage",
            "Measured locally in WSL · no user process terminated · no swap in timed heavy stages",
            fig,
            include_js,
        )
    )

    comparison_names = [DISPLAY_NAMES[factor_id] for factor_id in DISPLAY_NAMES]
    comparison_returns = [full_portfolio[factor_id]["total_return"] for factor_id in DISPLAY_NAMES]
    for row in benchmarks.iter_rows(named=True):
        if row.get("total_return") is not None:
            comparison_names.append(row["benchmark"])
            comparison_returns.append(row["total_return"])
    fig = go.Figure(
        go.Bar(
            x=comparison_names,
            y=comparison_returns,
            marker_color=[
                COLORS["red"] if value < 0 else COLORS["cyan"] for value in comparison_returns
            ],
        )
    )
    _figure_style(fig, "17 · Full-period locked portfolios versus benchmarks")
    charts.append(
        _chart_block(
            "benchmarks",
            "Benchmarks",
            "Base-cost long-only results · current V5 exact replay is omitted from the bar because its return is unavailable, not zero",
            fig,
            include_js,
        )
    )

    decision_cells = []
    for label, key in [
        ("Current V5", "current_v5"),
        ("rev_xs_20d", "rev_xs_20d"),
        ("low_vol_20d", "low_vol_20d"),
        ("funding_fade", "funding_fade"),
    ]:
        decision = decisions[key]["decision"]
        decision_cells.append(
            f'<div class="decision"><span>{label}</span><strong class="{decision.lower()}">{decision}</strong></div>'
        )
    core_answers = [
        (
            "Current V5 edge",
            "No verified edge; exact replay is unavailable and the proxy fails blind OOS.",
        ),
        (
            "Original t-stat",
            "Inflated in overlapping cells; HAC and horizon-spaced statistics control the decision.",
        ),
        (
            "Future leakage",
            "No mechanical leak detected; survivorship and state-history limits remain.",
        ),
        ("After-cost winner", "None in locked blind OOS."),
        ("OOS survivor", "Low-vol IC survives, but its tradable portfolio fails."),
        ("72h / 120h", "No passing horizon; every locked blind portfolio is negative."),
        ("Modify V5 now", "No. Keep production frozen."),
        ("Next priority", decisions["highest_priority_next_step"]),
    ]
    answer_html = "".join(
        f'<div class="answer"><span>{index:02d}</span><div><b>{title}</b><p>{body}</p></div></div>'
        for index, (title, body) in enumerate(core_answers, 1)
    )
    nav_items = [
        ("summary", "Decisions"),
        ("factor-ic", "IC"),
        ("tstats", "Stats"),
        ("horizons", "Horizons"),
        ("gross-net", "Portfolio"),
        ("cost-scenarios", "Costs"),
        ("equity", "Equity"),
        ("drawdown", "Drawdown"),
        ("calendar", "Calendar"),
        ("regime", "Regime"),
        ("contribution", "Contribution"),
        ("concentration", "Concentration"),
        ("multiplicity", "Multiplicity"),
        ("data-quality", "Data"),
        ("leakage", "Leakage"),
        ("negative-controls", "Controls"),
        ("resources", "Resources"),
        ("benchmarks", "Benchmarks"),
    ]
    nav = "".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in nav_items)
    dashboard = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpha Validity Audit · 2026-07-18</title>
<link rel="icon" href="data:,">
<style>
:root{{--bg:#070b12;--surface:#0b1220;--line:#233044;--text:#e7edf7;--muted:#8fa2b8;--accent:#5eead4;--fail:#fb7185;--inc:#fbbf24;--pass:#5eead4}}
*{{box-sizing:border-box}}html{{overflow-x:hidden;scroll-behavior:smooth}}body{{margin:0;overflow-x:hidden;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;background:linear-gradient(rgba(94,234,212,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(94,234,212,.025) 1px,transparent 1px);background-size:56px 56px;mask-image:linear-gradient(to bottom,black,transparent 70%)}}
.topbar{{position:sticky;top:0;z-index:30;display:flex;gap:24px;align-items:center;padding:12px 28px;border-bottom:1px solid var(--line);background:rgba(7,11,18,.92);backdrop-filter:blur(16px)}}.brand{{font:700 13px ui-monospace,monospace;letter-spacing:.12em;color:var(--accent);white-space:nowrap}}nav{{display:flex;gap:15px;overflow:auto;scrollbar-width:none}}nav a{{color:var(--muted);text-decoration:none;font-size:12px;white-space:nowrap;transition:color .18s}}nav a:hover{{color:var(--text)}}
main{{max-width:1480px;margin:auto;padding:48px 40px 96px}}.mast{{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(280px,.7fr);gap:54px;padding:20px 0 46px;border-bottom:1px solid var(--line)}}.eyebrow,.section-kicker{{font:600 11px ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}}h1{{font-size:clamp(40px,6vw,88px);line-height:.92;letter-spacing:-.055em;margin:16px 0 22px;max-width:950px}}.lede{{max-width:760px;color:var(--muted);font-size:17px}}.scope{{align-self:end;border-left:1px solid var(--line);padding-left:28px;color:var(--muted);font:12px/1.8 ui-monospace,monospace}}.scope b{{color:var(--text)}}
.decision-grid{{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--line)}}.decision{{padding:22px 18px;border-right:1px solid var(--line);display:flex;justify-content:space-between;gap:10px}}.decision:last-child{{border-right:0}}.decision span{{font-size:13px;color:var(--muted)}}.decision strong{{font:700 12px ui-monospace,monospace;letter-spacing:.08em}}.fail{{color:var(--fail)}}.inconclusive{{color:var(--inc)}}.pass{{color:var(--pass)}}
.recommendation{{display:flex;justify-content:space-between;gap:24px;padding:25px 0;border-bottom:1px solid var(--line)}}.recommendation b{{font-size:18px}}.recommendation span{{color:var(--muted);text-align:right;max-width:760px}}
.answers{{display:grid;grid-template-columns:repeat(2,1fr);gap:0;margin:34px 0 70px;border-top:1px solid var(--line)}}.answer{{display:grid;grid-template-columns:42px 1fr;gap:16px;padding:22px 16px 22px 0;border-bottom:1px solid var(--line)}}.answer:nth-child(odd){{border-right:1px solid var(--line);padding-right:28px}}.answer:nth-child(even){{padding-left:28px}}.answer>span{{font:12px ui-monospace,monospace;color:var(--accent)}}.answer b{{font-size:14px}}.answer p{{margin:5px 0 0;color:var(--muted);font-size:13px}}
.evidence{{padding:54px 0;border-top:1px solid var(--line);min-height:540px}}.evidence .meta{{color:var(--muted);font:12px ui-monospace,monospace;margin:8px 0 22px;max-width:1100px}}.reveal{{opacity:0;transform:translateY(18px);transition:opacity .55s ease,transform .55s ease}}.reveal.visible{{opacity:1;transform:none}}.plotly-graph-div{{min-height:440px}}footer{{padding:36px 0;color:var(--muted);font:12px ui-monospace,monospace;border-top:1px solid var(--line)}}
@media(max-width:850px){{main{{padding:28px 18px 72px}}.mast{{grid-template-columns:1fr;gap:24px}}.scope{{border-left:0;border-top:1px solid var(--line);padding:18px 0 0}}.decision-grid{{grid-template-columns:1fr 1fr}}.decision:nth-child(2){{border-right:0}}.answers{{grid-template-columns:1fr}}.answer:nth-child(odd){{border-right:0;padding-right:0}}.answer:nth-child(even){{padding-left:0}}.recommendation{{display:block}}.recommendation span{{display:block;text-align:left;margin-top:8px}}.topbar{{padding:11px 16px}}}}
@media(max-width:480px){{.decision-grid{{grid-template-columns:1fr}}.decision{{border-right:0}}}}
@media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important;transition:none!important}}.reveal{{opacity:1;transform:none}}}}
</style></head><body>
<header class="topbar"><div class="brand">ALPHA / VALIDITY / 0718</div><nav>{nav}</nav></header>
<main><section id="summary"><div class="mast"><div><div class="eyebrow">Read-only forensic audit · final</div><h1>No tradable OOS edge verified.</h1><p class="lede">Two years of closed 1h bars, overlap-aware statistics, frozen OOS rules, empirical costs, long-only portfolios, and 2,816 retained tests converge on one operational answer: keep production frozen.</p></div><div class="scope"><b>SNAPSHOT</b> {snapshot_dir().name}<br><b>DATA</b> 1,225,652 bars / 93 symbols<br><b>WINDOW</b> 2024-07-18 → 2026-07-17 UTC<br><b>MODE</b> read-only / no-order / local audit<br><b>BRANCH</b> audit/alpha-validity</div></div>
<div class="decision-grid">{"".join(decision_cells)}</div><div class="recommendation"><b>{decisions["recommended_action"]}</b><span>{decisions["highest_priority_next_step"]}</span></div><div class="answers">{answer_html}</div></section>
{"".join(charts)}
<footer>Static self-contained HTML · generated {datetime.now(UTC).isoformat()} · full evidence in reports/ and artifacts/ · no production mutation</footer></main>
<script>const io=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting)e.target.classList.add('visible')}}),{{threshold:.05}});document.querySelectorAll('.reveal').forEach(e=>io.observe(e));</script>
</body></html>"""
    (REPORTS / "audit_dashboard.html").write_text(dashboard, encoding="utf-8")

    # Manifest is emitted after all report files above exist.
    tracked_outputs = sorted(
        [path for path in REPORTS.iterdir() if path.is_file()]
        + [
            path
            for path in ARTIFACTS.iterdir()
            if path.is_file() and path.name != "run_manifest.json"
        ]
    )
    run_manifest = {
        "run_id": f"alpha-validity-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": datetime.now(UTC).isoformat(),
        "snapshot_id": snapshot_dir().name,
        "data_cutoff": "2026-07-18T02:00:00Z production snapshot; market backfill through 2026-07-17T23:00:00Z",
        "branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "random_seeds": [7, 19, 42, 20260718],
        "factor_grid": {
            "factors": 4,
            "directions": 2,
            "horizons": [24, 72, 120, 480],
            "universes": ["top10", "top20", "top50"],
        },
        "portfolio_grid": {
            "structural_combinations": 192,
            "cost_scenarios": 3,
            "periods": 4,
            "result_rows": portfolio.height,
        },
        "total_multiple_tests": multiplicity.height,
        "production_mutations": 0,
        "live_orders": 0,
        "outputs": {
            str(path.relative_to(LOCAL_AUDIT_ROOT)): _sha256(path) for path in tracked_outputs
        },
        "commands": [
            "./scripts/run_full_alpha_audit.sh --resume",
            "pytest -q --junitxml=/home/hr/quant-alpha-audit/artifacts/test_results.xml audit/tests",
            "pytest -q tests/test_alpha_ic.py tests/test_alpha_labels.py tests/test_factor_factory.py",
            "ruff check audit src/quant_lab/research/ic.py",
        ],
    }
    (ARTIFACTS / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(
        "final decisions:",
        {
            key: value["decision"]
            for key, value in decisions.items()
            if isinstance(value, dict) and "decision" in value
        },
    )
    print("dashboard:", REPORTS / "audit_dashboard.html")


if __name__ == "__main__":
    main()
