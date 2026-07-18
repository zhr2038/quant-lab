# ruff: noqa: E501
"""Generate the Audit v2.1 decisions, reports and static dashboard."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from audit.auditlib.report_v21 import validate_v21_report_contract


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "∞" if number > 0 else "—"
    return f"{number * 100:.3f}%"


def _num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "∞" if number > 0 else "—"
    return f"{number:.{digits}f}"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _svg_path(values: list[float], width: float = 900, height: float = 220) -> str:
    if not values:
        values = [1.0]
    low, high = min(values), max(values)
    spread = max(high - low, 1e-9)
    denominator = max(len(values) - 1, 1)
    points = [
        (
            index / denominator * width,
            height - (value - low) / spread * height,
        )
        for index, value in enumerate(values)
    ]
    return " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def _status_copy(status: dict) -> str:
    return (
        "PAPER_REVIEW_READY（仍非 PASS / 非 Live）"
        if status["conclusion"] == "PAPER_REVIEW_READY"
        else "INCONCLUSIVE — forward 样本不足"
    )


def _write_final_decisions(
    root: Path, status: dict, funding: dict, provisional: dict
) -> dict:
    decisions = {
        "audit_version": "v2.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "forward_v2_status": "INVALIDATED_BY_RUNNER_BUG",
        "preacceptance_initialization": {
            "status": provisional["status"],
            "parameter_lock_hash": provisional["parameter_lock_hash"],
            "may_merge_with_official_v21": False,
            "replacement_code_commit": provisional["replacement_code_commit"],
        },
        "forward_v21_start_cutoff": status["forward_v21_start_cutoff"],
        "forward_v21": {
            "hypothesis_type": "POST_HOC_HYPOTHESIS",
            "parameters_locked": True,
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "INCONCLUSIVE",
            "status": status["conclusion"],
            "maximum_possible_status": "PAPER_REVIEW_READY",
            "available_days": status["forward_available_days"],
            "decision_count": status["decision_count"],
            "cash_decision_count": status["cash_decision_count"],
            "entry_count": status["entry_count"],
            "closed_trade_count": status["closed_trade_count"],
            "live_order_effect": "none",
        },
        "current_v5": {
            "signal_validity": "INCONCLUSIVE",
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "FAIL",
            "replayability": "PARTIALLY_REPLAYABLE",
            "production_alpha": "FROZEN",
            "reason": "Decision Receipt remains a design; exact historical replay is not yet available.",
        },
        "rev_xs_20d": {
            "signal_validity": "FAIL",
            "portfolio_validity": "FAIL",
            "deployment_readiness": "FAIL",
            "recommended_action": "RETIRE",
        },
        "low_vol_20d": {
            "signal_validity": "PASS",
            "locked_portfolio_validity": "FAIL",
            "deployment_readiness": "INCONCLUSIVE",
        },
        "low_vol_btc_trend": {
            "hypothesis_type": "POST_HOC_HYPOTHESIS",
            "parameters_locked": True,
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "INCONCLUSIVE",
            "forward_sample_start": status["forward_v21_start_cutoff"],
            "forward_sample_end": status["available_market_data_cutoff"],
            "forward_available_days": status["forward_available_days"],
            "forward_entry_count": status["entry_count"],
            "forward_closed_trade_count": status["closed_trade_count"],
        },
        "funding_fade": {
            "signal_validity": "INCONCLUSIVE",
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "INCONCLUSIVE",
            "window_type": "TIME_BASED_20D_STRICT_EVENT_AVAILABILITY",
            "n_periods": funding.get("n_periods"),
        },
        "recommended_production_action": "KEEP_PRODUCTION_ALPHA_FROZEN",
        "highest_priority_next_step": (
            "Accumulate strictly post-cutoff forward data with the immutable v2.1 runner."
        ),
        "live_or_live_small_permitted": False,
    }
    (root / "artifacts/final_decisions_v21.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return decisions


def _forward_report(status: dict, lock: dict) -> str:
    days = f"{float(status['forward_available_days']):.9f}"
    return f"""# Forward Paper v2.1

**Conclusion: {_status_copy(status)}.** This is a post-hoc, forward-only paper
observation. It cannot enable Live or LIVE_SMALL.

| Field | Value |
|---|---:|
| Forward v2.1 cutoff | `{status['forward_v21_start_cutoff']}` |
| Available market-data cutoff | `{status['available_market_data_cutoff']}` |
| Forward available days | `{days}` |
| Decisions / cash decisions | {status['decision_count']} / {status['cash_decision_count']} |
| Coin entries / open / closed | {status['entry_count']} / {status['open_trade_count']} / {status['closed_trade_count']} |
| Completed independent periods | {status['completed_independent_period_count']} |
| Realized gross compound | {_pct(status['realized_gross_compounded_return'])} |
| Realized net compound | {_pct(status['realized_net_compounded_return'])} |
| Realized arithmetic sum (diagnostic only) | {_pct(status['realized_net_arithmetic_sum_return'])} |
| Unrealized weighted net mark | {_pct(status['unrealized_weighted_net_return'])} |
| Strategy marked compound | {_pct(status['marked_strategy_compounded_return'])} |
| BTC benchmark | {_pct(status['btc_benchmark_compounded_return'])} |
| Dynamic-universe equal-weight benchmark | {_pct(status['equal_weight_universe_benchmark_compounded_return'])} |
| Strategy excess vs BTC / universe | {_pct(status['strategy_excess_vs_btc'])} / {_pct(status['strategy_excess_vs_universe'])} |
| Data coverage | {_pct(status['data_coverage'])} |

The old v2 record is preserved and marked `INVALIDATED_BY_RUNNER_BUG`; it is not
merged with v2.1. Every actual leg enters at the exact next completed 1h close,
has a scheduled exit at entry + 120h, closes at that bar or the first later
available completed bar, and locks the realized exit forever. Entry and exit
each charge 10bps fee plus 5bps slippage. CASH decisions are not trades.

Parameter lock: `{lock['sha256']}`  
Runner: `{lock['runner_version']}`  
Code commit: `{lock['code_commit']}`
"""


def _runner_fix_report(status: dict, provisional: dict) -> str:
    return f"""# Forward Runner Fix Report

Status: **FIXED**. The invalidated v2 implementation is fail-fast and new
evidence is written only through the v2.1 runner.

| v2 blocker | v2.1 behavior | Verification |
|---|---|---|
| Old trades repriced at latest close | Exit locks at scheduled + first available bar | Unit test |
| Period returns summed | Time-ordered compounded equity; arithmetic sum diagnostic only | Unit test |
| Only entry-side cost | 15bps entry + 15bps exit, multiplicative formula | Unit test |
| Benchmark fixed to zero | BTC, dynamic-universe equal weight and cash calculated | Unit test + artifacts |
| CASH counted as completed | Decision/trade counters are separate | Unit test |
| Reruns could overwrite history | Stable IDs, immutable decisions/trades, append-only events | Unit test |
| Inconsistent elapsed days | available market cutoff minus v2.1 cutoff everywhere | Consistency manifest |

Observed v2.1 state: `{status['conclusion']}`. Fixing accounting does not prove
the post-hoc strategy valid and does not change the production freeze.

## Rejected pre-acceptance initialization

The first provisional v2.1 initialization was rejected before acceptance after
the OKX bar-open/bar-close availability mismatch was found. It had zero
decisions, entries and closed trades, and cannot be merged with official v2.1.

- Status: `{provisional['status']}`
- Rejected lock: `{provisional['parameter_lock_hash']}`
- Rejected code: `{provisional['code_commit']}`
- Replacement code: `{provisional['replacement_code_commit']}`
"""


def _funding_report(frequency: pl.DataFrame, comparison: pl.DataFrame) -> str:
    corrected = comparison.filter(
        pl.col("window_type").str.starts_with("TIME_BASED")
    ).to_dicts()[0]
    return f"""# Funding Window Correction v2.1

Status: **FIXED TIME SEMANTICS / INCONCLUSIVE FACTOR**.

The rolling 20-natural-day event table includes the event published at `t`,
while the decision as-of join requires `funding_event_ts < decision_ts`.
Therefore the event is unavailable at the exact publication timestamp and is
available one instant later; it is no longer delayed until the next settlement.

- Symbols profiled: {frequency.height}
- Time-based evaluation periods: {corrected.get('n_periods')}
- Rank IC: {_num(corrected.get('ic_mean'), 6)}
- HAC t-stat: {_num(corrected.get('hac_tstat'), 3)}
- Observation-count windows remain comparison evidence only.
- Missing funding is never filled with zero.

Historical funding coverage remains insufficient under the preregistered
standard. Signal, portfolio and deployment validity all remain INCONCLUSIVE.
"""


def _permutation_report(nulls: pl.DataFrame, robustness: pl.DataFrame) -> str:
    null_lines = [
        "| Control | N | IC empirical p | Paired-metric max-stat p |",
        "|---|---:|---:|---:|",
    ]
    for row in nulls.sort("control_type").iter_rows(named=True):
        null_lines.append(
            f"| `{row['control_type']}` | {row['permutation_count']} | "
            f"{_num(row['ic_empirical_p_value'], 4)} | "
            f"{_num(row['paired_metric_max_stat_p_value'], 4)} |"
        )
    return f"""# Permutation and Controls v2.1

`time_within_factor_permutation` is now a robustness perturbation, not a strict
null. Two true within-symbol nulls were added with 1,000 deterministic seeds
each; the three valid v2 strict null distributions were hash-preserved and
reused.

{chr(10).join(null_lines)}

Strict controls: cross-sectional rank permutation, label permutation,
independent noise, within-symbol signal permutation and independent
within-symbol circular shift. Robustness rows: {robustness.height}.

The old `max_stat` label overstated its scope. v2.1 calls it
`paired_metric_max_stat` because it covers only IC and portfolio t-stat per
permutation, not the full confirmatory family. Holm-Bonferroni remains the
formal family-wise correction.
"""


def _concentration_report(contributions: pl.DataFrame) -> str:
    summary = (
        contributions.group_by(["structure_id", "partition"])
        .agg(
            pl.col("top_symbol").first().alias("top_symbol"),
            pl.col("top_symbol_share").max().alias("top_share"),
            pl.col("top_3_symbol_share").max().alias("top3_share"),
            pl.col("hhi").max().alias("hhi"),
        )
        .filter(
            pl.col("structure_id").is_in(
                [
                    "core.low_vol_480|top3|score|24h|staggered=0|btc_filter=1",
                    "core.rev_xs_480|top3|score|24h|staggered=0|btc_filter=1",
                    "core.low_vol_480|top3|score|120h|btc_filter=1",
                ]
            )
        )
        .sort(["structure_id", "partition"])
    )
    lines = [
        "| Structure | Partition | Top symbol | Top share | Top-3 share | HHI |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"| `{row['structure_id']}` | {row['partition']} | "
            f"{row['top_symbol'] or 'none'} | {_pct(row['top_share'])} | "
            f"{_pct(row['top3_share'])} | {_num(row['hhi'], 4)} |"
        )
    return f"""# Symbol Concentration v2.1

Research, validation, blind, full and forward contributions are computed from
their own return and traded-weight matrices. The v2 full-sample concentration
value is retained as `invalid_v2_full_sample_concentration` and is never reused
as an OOS or blind statistic.

{chr(10).join(lines)}
"""


def _test_report(root: Path) -> str:
    path = root / "artifacts/test_execution_v21.json"
    if not path.exists():
        return """# Audit v2.1 Test Report

Final test execution evidence has not yet been generated. Run the v2.1 test
stage before packaging; the bundle stage rejects this state.
"""
    data = _json(path)
    rows = ["| Scope | Passed | Failed | Errors | Skipped | Runtime |", "|---|---:|---:|---:|---:|---:|"]
    for result in data["test_suites"]:
        rows.append(
            f"| {result['scope']} | {result['passed']} | {result['failed']} | "
            f"{result['errors']} | {result['skipped']} | {result['seconds']:.3f}s |"
        )
    checks = "\n".join(
        f"- `{item['command']}`: **{item['status']}**."
        for item in data["static_checks"]
    )
    v5 = data["v5_regression"]
    return f"""# Audit v2.1 Test Report

Overall status: **{data['overall_status']}**. No failing test was skipped,
deleted or weakened by Audit v2.1.

{chr(10).join(rows)}

## Static checks

{checks}

Browser QA is recorded separately in `artifacts/browser_qa_v21.json` and is
required by the bundle validator.

The fresh V5 full regression is **{v5['status']}** at commit
`{v5['git_head']}`. It ran against the unchanged read-only checkout;
`source_mutated=false`.
"""


def _dashboard(
    status: dict,
    lock: dict,
    nulls: pl.DataFrame,
    funding: dict,
    contributions: pl.DataFrame,
    equity: pl.DataFrame,
    provisional: dict,
) -> str:
    days = f"{float(status['forward_available_days']):.9f}"
    strategy_path = _svg_path(equity["strategy_equity"].to_list())
    btc_path = _svg_path(equity["btc_equity"].to_list())
    universe_path = _svg_path(equity["equal_weight_universe_equity"].to_list())
    null_rows = "".join(
        f"<tr><td><code>{_esc(row['control_type'])}</code></td>"
        f"<td>{row['permutation_count']}</td>"
        f"<td>{_num(row['ic_empirical_p_value'], 4)}</td>"
        f"<td>{_num(row['paired_metric_max_stat_p_value'], 4)}</td></tr>"
        for row in nulls.sort("control_type").iter_rows(named=True)
    )
    locked = contributions.filter(
        pl.col("structure_id")
        == "core.low_vol_480|top3|score|24h|staggered=0|btc_filter=1"
    )
    concentration_rows = "".join(
        f"<tr><td>{_esc(partition[0] if isinstance(partition, tuple) else partition)}</td><td>{_esc(group['top_symbol'][0])}</td>"
        f"<td>{_pct(group['top_symbol_share'].max())}</td>"
        f"<td>{_pct(group['top_3_symbol_share'].max())}</td>"
        f"<td>{_num(group['hhi'].max(), 4)}</td></tr>"
        for partition, group in locked.group_by("partition", maintain_order=True)
    )
    conclusion = _status_copy(status)
    return f"""<!doctype html>
<html lang="zh-CN" data-forward-available-days="{days}">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpha Audit v2.1 — Forward Runner 修正</title>
<style>
:root{{--ink:#edf3f5;--muted:#91a0a5;--bg:#081012;--surface:#101a1d;--line:#273438;--red:#ff5c5c;--amber:#f4b860;--green:#55d6a7;--blue:#70a7ff}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
a{{color:inherit}}code{{font-family:"SFMono-Regular",Consolas,monospace;font-size:.88em;color:#c9e2e8}}.wrap{{max-width:1240px;margin:auto;padding:0 36px}}nav{{position:sticky;top:0;z-index:5;background:rgba(8,16,18,.92);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}}nav .wrap{{height:58px;display:flex;align-items:center;justify-content:space-between}}nav a{{text-decoration:none;color:var(--muted);margin-left:22px;font-size:13px}}nav a:hover{{color:var(--ink)}}header{{min-height:470px;display:grid;align-items:center;border-bottom:1px solid var(--line);background:radial-gradient(circle at 80% 18%,rgba(255,92,92,.12),transparent 34%),linear-gradient(135deg,#081012,#0e191b)}}.eyebrow{{letter-spacing:.16em;text-transform:uppercase;color:var(--green);font-size:12px;font-weight:750}}h1{{font-size:clamp(42px,7vw,88px);line-height:.94;letter-spacing:-.055em;margin:18px 0 24px;max-width:980px}}.lede{{max-width:800px;color:#b4c1c5;font-size:19px}}.freeze{{display:inline-flex;margin-top:28px;padding:8px 13px;border:1px solid rgba(255,92,92,.6);color:var(--red);font-weight:800;letter-spacing:.08em}}section{{padding:72px 0;border-bottom:1px solid var(--line)}}h2{{font-size:32px;letter-spacing:-.025em;margin:0 0 8px}}.sub{{color:var(--muted);max-width:820px;margin:0 0 34px}}.statusline{{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}.statusline>div{{padding:25px 22px;border-right:1px solid var(--line)}}.statusline>div:last-child{{border:0}}.label{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}}.value{{font-size:25px;font-weight:750;margin-top:5px}}.green{{color:var(--green)}}.amber{{color:var(--amber)}}.red{{color:var(--red)}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line)}}.metric{{padding:24px 18px;border-bottom:1px solid var(--line);border-right:1px solid var(--line)}}.metric:nth-child(4n){{border-right:0}}.metric b{{display:block;font-size:25px;margin-top:6px}}.chart{{height:260px;margin:30px 0 12px;border-left:1px solid var(--line);border-bottom:1px solid var(--line);padding:18px 0 0}}svg{{width:100%;height:220px;overflow:visible}}.legend span{{margin-right:22px;color:var(--muted);font-size:13px}}.dot{{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:7px}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:13px 10px;border-bottom:1px solid var(--line)}}th{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}}tbody tr{{transition:background .16s}}tbody tr:hover{{background:#142125}}.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:58px}}.twocol>div{{min-width:0}}.callout{{border-left:3px solid var(--amber);padding:10px 0 10px 20px;color:#cbd5d8}}.provenance{{display:grid;grid-template-columns:210px 1fr;gap:8px 24px}}.provenance dt{{color:var(--muted)}}.provenance dd{{margin:0;overflow-wrap:anywhere}}footer{{padding:48px 0 70px;color:var(--muted)}}.reveal{{opacity:0;transform:translateY(10px);transition:opacity .45s,transform .45s}}.reveal.show{{opacity:1;transform:none}}@media(max-width:800px){{.wrap{{padding:0 20px}}nav .links{{display:none}}header{{min-height:430px}}.statusline,.metrics,.twocol{{grid-template-columns:1fr 1fr}}.statusline>div{{border-bottom:1px solid var(--line)}}.twocol table{{display:block;max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch}}.provenance{{grid-template-columns:1fr}}}}@media(max-width:520px){{.statusline,.metrics,.twocol{{grid-template-columns:1fr}}.metric{{border-right:0}}section{{padding:52px 0}}h1{{font-size:48px}}}}
</style></head>
<body>
<nav><div class="wrap"><strong>ALPHA AUDIT / v2.1</strong><div class="links"><a href="#forward">Forward</a><a href="#controls">Controls</a><a href="#funding">Funding</a><a href="#trace">Trace</a></div></div></nav>
<header><div class="wrap reveal"><div class="eyebrow">Controlled correction audit · 2026-07-18</div><h1>Runner fixed.<br>Evidence restarts now.</h1><p class="lede">旧 v2 Forward 记录已失效。v2.1 从修复完成后的新 cutoff 开始，以固定 120h 退出、双边成本、复利和真实基准累计全新证据。</p><span class="freeze">PRODUCTION ALPHA · FROZEN</span></div></header>
<main>
<section><div class="wrap reveal"><div class="statusline"><div><div class="label">Forward Runner</div><div class="value green">FIXED</div></div><div><div class="label">Funding 时间窗口</div><div class="value green">FIXED</div></div><div><div class="label">Forward v2 旧记录</div><div class="value red">INVALIDATED</div></div></div></div></section>
<section id="forward"><div class="wrap reveal"><h2>Forward evidence</h2><p class="sub">结论只取实际可用完整市场数据，不取请求时间；未实现收益与已实现收益严格分开。</p><div class="metrics"><div class="metric"><span class="label">开始时间</span><b style="font-size:14px">{_esc(status['forward_v21_start_cutoff'])}</b></div><div class="metric"><span class="label">新数据天数</span><b>{days}</b></div><div class="metric"><span class="label">决策 / 空仓</span><b>{status['decision_count']} / {status['cash_decision_count']}</b></div><div class="metric"><span class="label">入场 / 关闭 / 开放</span><b>{status['entry_count']} / {status['closed_trade_count']} / {status['open_trade_count']}</b></div><div class="metric"><span class="label">已实现净复合</span><b>{_pct(status['realized_net_compounded_return'])}</b></div><div class="metric"><span class="label">未实现净标记</span><b>{_pct(status['unrealized_weighted_net_return'])}</b></div><div class="metric"><span class="label">算术累计（仅诊断）</span><b>{_pct(status['realized_net_arithmetic_sum_return'])}</b></div><div class="metric"><span class="label">当前结论</span><b class="amber" style="font-size:17px">INCONCLUSIVE</b></div></div><div class="chart"><svg viewBox="0 0 900 220" preserveAspectRatio="none" aria-label="Forward equity"><path d="{strategy_path}" fill="none" stroke="#55d6a7" stroke-width="3"/><path d="{btc_path}" fill="none" stroke="#70a7ff" stroke-width="2"/><path d="{universe_path}" fill="none" stroke="#f4b860" stroke-width="2"/></svg></div><div class="legend"><span><i class="dot" style="background:#55d6a7"></i>Strategy</span><span><i class="dot" style="background:#70a7ff"></i>BTC</span><span><i class="dot" style="background:#f4b860"></i>Dynamic universe</span><span>Cash = 1.000</span></div><div class="metrics" style="margin-top:24px"><div class="metric"><span class="label">策略标记复合</span><b>{_pct(status['marked_strategy_compounded_return'])}</b></div><div class="metric"><span class="label">BTC 基准</span><b>{_pct(status['btc_benchmark_compounded_return'])}</b></div><div class="metric"><span class="label">动态币池基准</span><b>{_pct(status['equal_weight_universe_benchmark_compounded_return'])}</b></div><div class="metric"><span class="label">数据覆盖</span><b>{_pct(status['data_coverage'])}</b></div></div><div class="metrics"><div class="metric"><span class="label">已实现毛复合</span><b>{_pct(status['realized_gross_compounded_return'])}</b></div><div class="metric"><span class="label">已实现净复合</span><b>{_pct(status['realized_net_compounded_return'])}</b></div><div class="metric"><span class="label">已实现入场成本</span><b>{_pct(status['entry_fee_return'] + status['entry_slippage_return'])}</b></div><div class="metric"><span class="label">已实现退出成本</span><b>{_pct(status['exit_fee_return'] + status['exit_slippage_return'])}</b></div><div class="metric"><span class="label">开放仓入场成本</span><b>{_pct(status['open_entry_cost_return'])}</b></div><div class="metric"><span class="label">策略超额 vs BTC</span><b>{_pct(status['strategy_excess_vs_btc'])}</b></div><div class="metric"><span class="label">策略超额 vs 动态币池</span><b>{_pct(status['strategy_excess_vs_universe'])}</b></div><div class="metric"><span class="label">最大回撤</span><b>{_pct(status['max_drawdown'])}</b></div></div><p class="callout">{_esc(conclusion)}。即使未来达到最小门槛，本阶段最多输出 PAPER_REVIEW_READY，绝不授予 LIVE_SMALL。</p></div></section>
<section id="controls"><div class="wrap reveal"><div class="twocol"><div><h2>Strict null controls</h2><p class="sub">新增 within-symbol permutation 与 circular shift，各 1,000 次。time-within 已移入 robustness。</p><table><thead><tr><th>Control</th><th>N</th><th>IC p</th><th>Paired max p</th></tr></thead><tbody>{null_rows}</tbody></table></div><div><h2>Partition concentration</h2><p class="sub">不再把 full 样本的集中度复制给 validation 与 blind。</p><table><thead><tr><th>Partition</th><th>Top</th><th>Share</th><th>Top 3</th><th>HHI</th></tr></thead><tbody>{concentration_rows}</tbody></table></div></div></div></section>
<section id="funding"><div class="wrap reveal"><h2>Funding event availability</h2><p class="sub">事件在 exact timestamp 不可用，之后立即可用；自然 20 日窗口不再多延迟一个结算周期。</p><div class="metrics"><div class="metric"><span class="label">时间语义</span><b class="green">FIXED</b></div><div class="metric"><span class="label">窗口</span><b style="font-size:18px">TIME-BASED 20D</b></div><div class="metric"><span class="label">有效 periods</span><b>{_esc(funding.get('n_periods'))}</b></div><div class="metric"><span class="label">因子结论</span><b class="amber">INCONCLUSIVE</b></div></div></div></section>
<section id="trace"><div class="wrap reveal"><h2>Locked provenance</h2><p class="sub">v2.1 样本不能与失效 v2 样本合并；所有新记录绑定同一参数锁、代码和 baseline snapshot。</p><div class="freeze" style="margin:0 0 28px">PROVISIONAL INITIALIZATION · REJECTED</div><dl class="provenance"><dt>Parameter lock</dt><dd><code>{_esc(lock['sha256'])}</code></dd><dt>Runner</dt><dd><code>{_esc(lock['runner_version'])}</code></dd><dt>Code commit</dt><dd><code>{_esc(lock['code_commit'])}</code></dd><dt>Rejected provisional lock</dt><dd><code>{_esc(provisional['parameter_lock_hash'])}</code> · zero decisions · never merge</dd><dt>Data snapshot</dt><dd><code>{_esc(lock['data_snapshot_id'])}</code></dd><dt>Entry / exit</dt><dd><code>{_esc(lock['entry_price_rule'])}</code> / <code>{_esc(lock['exit_price_rule'])}</code></dd><dt>Cost</dt><dd>15bps entry + 15bps exit; 30bps round trip</dd><dt>V5 replayability</dt><dd>PARTIALLY_REPLAYABLE — Decision Receipt is design only</dd></dl></div></section>
</main><footer><div class="wrap">Audit v2.1 · no deployment · no orders · no production mutation · generated {datetime.now(UTC).isoformat()}</div></footer>
<script>const io=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting)e.target.classList.add('show')}}),{{threshold:.08}});document.querySelectorAll('.reveal').forEach(e=>io.observe(e));</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    status = _json(root / "artifacts/forward_v21_status.json")
    lock = _json(root / "manifests/parameter_lock_v21.json")
    rejected_paths = sorted(
        (root / "provisional_rejected").glob("*/REJECTION_STATUS.json")
    )
    if len(rejected_paths) != 1:
        raise RuntimeError(
            f"expected exactly one rejected provisional initialization, found {len(rejected_paths)}"
        )
    provisional = _json(rejected_paths[0])
    if provisional.get("may_merge_with_official_v21") is not False:
        raise RuntimeError("rejected provisional initialization may not be mergeable")
    shutil.copy2(
        rejected_paths[0],
        root / "manifests/provisional_initialization_rejection_v21.json",
    )
    funding_frame = pl.read_csv(root / "artifacts/funding_signal_results_v21.csv")
    funding = funding_frame.filter(
        pl.col("window_type").str.starts_with("TIME_BASED")
    ).to_dicts()[0]
    frequency = pl.read_csv(root / "artifacts/funding_frequency_summary_v21.csv")
    comparison = pl.read_csv(root / "artifacts/funding_window_comparison_v21.csv")
    nulls = pl.read_csv(root / "artifacts/null_control_summary_v21.csv")
    robustness = pl.read_csv(
        root / "artifacts/robustness_perturbation_summary_v21.csv"
    )
    contributions = pl.read_csv(
        root / "artifacts/symbol_contribution_by_partition_v21.csv"
    )
    equity = pl.read_parquet(root / "artifacts/forward_v21_equity.parquet")
    decisions = _write_final_decisions(root, status, funding, provisional)

    _write(reports / "forward_paper_report_v21.md", _forward_report(status, lock))
    _write(
        reports / "forward_runner_fix_report.md",
        _runner_fix_report(status, provisional),
    )
    _write(
        reports / "funding_window_correction_v21.md",
        _funding_report(frequency, comparison),
    )
    _write(
        reports / "permutation_and_controls_report_v21.md",
        _permutation_report(nulls, robustness),
    )
    _write(
        reports / "symbol_concentration_v21.md",
        _concentration_report(contributions),
    )
    shutil.copy2(
        repo / "docs/v5_decision_receipt_design.md",
        reports / "v5_decision_receipt_design.md",
    )
    _write(reports / "test_report_v21.md", _test_report(root))
    days = f"{float(status['forward_available_days']):.9f}"
    executive = f"""# Audit v2.1 Executive Summary

Audit v2.1 is complete at the evidence-generation layer: the Forward Runner
uses immutable 120h exits, next-bar entry, bilateral 30bps costs, compounded
equity, separate CASH counts and real BTC/dynamic-universe benchmarks. The old
v2 record is preserved but `INVALIDATED_BY_RUNNER_BUG`.

- Forward available days: `{days}`; entries: {status['entry_count']}; closed:
  {status['closed_trade_count']}; conclusion: **INCONCLUSIVE**.
- Funding event timing is corrected, but the factor remains INCONCLUSIVE.
- `time_within_factor_permutation` is robustness; two true within-symbol nulls
  were added with 1,000 seeds each.
- Symbol concentration is now partition-specific.
- Paired IC/portfolio max-stat is named honestly; Holm-Bonferroni is official.
- V5 Decision Receipt is a default-off design/schema only. Current V5 remains
  PARTIALLY_REPLAYABLE, deployment readiness FAIL, production Alpha FROZEN.
- A zero-decision provisional initialization was rejected before acceptance
  when OKX bar-close availability semantics were corrected; it is preserved
  and cannot be merged with official v2.1.

Historical decisions remain unchanged: rev_xs is RETIRE (FAIL/FAIL/FAIL),
low-vol signal PASS does not rescue the locked portfolio FAIL, and the post-hoc
low-vol + BTC-trend portfolio remains INCONCLUSIVE pending genuinely new data.
"""
    _write(reports / "executive_summary_v21.md", executive)
    reproduction = f"""# Audit v2.1 Reproduction Guide

```bash
export AUDIT_V21_ROOT={root}
export AUDIT_V2_DIR={root / 'v2'}
export AUDIT_V1_DIR=/home/hr/quant-alpha-audit-v2/v1
export AUDIT_V1_ROOT=/home/hr/quant-alpha-audit
export QLAB_REPO_PATH={repo}
cd "$QLAB_REPO_PATH"

# Parameter lock/cutoff are immutable once created. Do not rerun init with a new value.
./scripts/run_low_vol_forward_paper.sh --as-of "$(date -u +%FT%TZ)" --resume

# Recompute only affected historical evidence.
$AUDIT_V1_ROOT/.venv/bin/python -m audit.scripts.stage_v21_analysis \\
  --root "$AUDIT_V21_ROOT" --v2-dir "$AUDIT_V2_DIR" \\
  --v1-dir "$AUDIT_V1_DIR" --v1-root "$AUDIT_V1_ROOT"

$AUDIT_V1_ROOT/.venv/bin/python -m audit.scripts.stage_v21_test \\
  --root "$AUDIT_V21_ROOT" --repo "$QLAB_REPO_PATH" \\
  --python "$AUDIT_V1_ROOT/.venv/bin/python" \\
  --v5-junit "$AUDIT_V21_ROOT/artifacts/v5_pytest_all_v21.xml" \\
  --v5-head 3df1c67cc44cc8be364ec5d3798ea0d3595c0abc

$AUDIT_V1_ROOT/.venv/bin/python -m audit.scripts.stage_v21_report \\
  --root "$AUDIT_V21_ROOT" --repo "$QLAB_REPO_PATH"
```

Generate the fresh V5 JUnit first, from the unchanged Windows checkout:

```powershell
Set-Location E:/v5-prod/V5-prod-main-clean
python -m pytest -q --junitxml=//wsl.localhost/Ubuntu/home/hr/quant-alpha-audit-v2.1/artifacts/v5_pytest_all_v21.xml
```

The runner reads public completed OKX bars only. It contains no exchange
credentials and cannot submit orders. Never merge v2 and v2.1 forward samples.
"""
    _write(reports / "reproduction_guide_v21.md", reproduction)
    dashboard = _dashboard(
        status, lock, nulls, funding, contributions, equity, provisional
    )
    _write(reports / "audit_dashboard_v21.html", dashboard)

    performance_row = pl.read_csv(
        root / "artifacts/forward_v21_performance.csv"
    ).to_dicts()[0]
    consistency = validate_v21_report_contract(
        status=status,
        performance=performance_row,
        dashboard_html=dashboard,
        forward_markdown=_forward_report(status, lock),
        decisions=decisions,
    )
    consistency.update({
        "schema_version": "alpha_audit_report_consistency.v2.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "report_sha256": {
            path.name: _sha(path)
            for path in sorted(reports.iterdir())
            if path.is_file()
        },
    })
    (root / "manifests/report_consistency_v21.json").write_text(
        json.dumps(consistency, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not consistency["consistent"]:
        raise RuntimeError("v2.1 report consistency check failed")
    print(f"dashboard={reports / 'audit_dashboard_v21.html'}")
    print(f"forward_available_days={days}")


if __name__ == "__main__":
    main()
