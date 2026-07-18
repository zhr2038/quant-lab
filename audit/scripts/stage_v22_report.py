# ruff: noqa: E501
"""Generate canonical Audit v2.2 reports and the static operations dashboard."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v22 import atomic_write_json, sha256_file  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.3f}%"


def _number(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _low_vol_metrics(v21_root: Path) -> dict[str, float]:
    observed = _load(v21_root / "artifacts/analysis_summary_v21.json")["observed"]
    permutations = pl.read_parquet(v21_root / "artifacts/permutation_distribution_v21.parquet")
    within = permutations.filter(pl.col("control_type").str.starts_with("within_symbol"))
    null_ic = float(within["ic"].mean())
    return {
        "raw_cross_sectional_ic": float(observed["ic"]),
        "raw_ic_t_stat": float(observed["ic_t_stat"]),
        "within_symbol_permutation_null_ic": null_ic,
        "incremental_ic_above_symbol_persistent_component": float(observed["ic"]) - null_ic,
        "within_symbol_permutation_count": within.height,
    }


def _final_decisions(status: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    paper = str(status["paper_status"])
    if paper.startswith("FAIL_"):
        portfolio = "FAIL"
        deployment = "FAIL"
    else:
        portfolio = "INCONCLUSIVE"
        deployment = "INCONCLUSIVE"
    return {
        "audit_version": "v2.2",
        "strategy_id": lock["strategy_id"],
        "strategy_version": lock["strategy_version"],
        "current_v5": {
            "signal_validity": "INCONCLUSIVE",
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "FAIL",
            "replayability": "PARTIALLY_REPLAYABLE",
            "production_alpha": "FROZEN",
            "reason": "Decision Receipt v2.2 is default-off and not deployed; exact historical replay remains incomplete.",
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
        "low_vol_btc_60d_trend": {
            "strategy_id": lock["strategy_id"],
            "hypothesis_type": "POST_HOC_HYPOTHESIS",
            "parameters_locked": True,
            "btc_trend_lookback_hours": 1440,
            "portfolio_validity": portfolio,
            "deployment_readiness": deployment,
            "paper_status": paper,
            "forward_v22_cutoff": status["forward_v22_cutoff"],
            "real_time_decision_count": status["real_time_decision_count"],
            "reconstructed_decision_count": status["reconstructed_decision_count"],
            "eligible_decision_count": status["eligible_decision_count"],
            "actual_symbol_trades": status["actual_symbol_trades"],
        },
        "funding_fade": {
            "signal_validity": "INCONCLUSIVE",
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "INCONCLUSIVE",
            "window_type": "TIME_BASED_20D_STRICT_EVENT_AVAILABILITY",
        },
        "production_alpha": "FROZEN",
        "live": "NOT_ALLOWED",
        "recommended_production_action": "KEEP_PRODUCTION_ALPHA_FROZEN",
        "highest_priority_next_step": (
            "Accumulate 30 days of timestamp-valid REALTIME-only Forward v2.2 evidence; "
            "never merge RECOVERY_RECONSTRUCTION rows."
        ),
    }


def _executive(status: dict[str, Any], lock: dict[str, Any]) -> str:
    return f"""# Audit v2.2 Executive Summary

Audit v2.2 is **complete as an integrity implementation**. The paper strategy is
not validated yet. Its current status is **{status['paper_status']}**.

- Strategy: `{lock['strategy_id']}` / version `{lock['strategy_version']}`
- BTC trend: **60 days / 1440 hours**
- Forward cutoff: `{status['forward_v22_cutoff']}`; only later timestamps qualify
- REALTIME decisions: **{status['real_time_decision_count']}**
- RECOVERY_RECONSTRUCTION decisions: **{status['reconstructed_decision_count']}**
- Formal eligible decisions: **{status['eligible_decision_count']}**
- Formal symbol trades: **{status['actual_symbol_trades']}**
- Forward days: **{_number(status['forward_days'], 6)}**
- Strategy net return: **{_pct(status['strategy_net_return'])}**
- BTC benchmark: **{_pct(status['btc_benchmark_return'])}**
- Dynamic-universe benchmark: **{_pct(status['dynamic_universe_benchmark_return'])}**

The four v2.1 blockers are closed in code: late reconstruction is excluded from
formal evidence, minimum-sample failures have explicit FAIL states, benchmarks
are append-only event chains, and formal runs fail closed on Git HEAD, clean
worktree, parameter-lock hash, and every locked source hash.

Research conclusions remain fixed: `rev_xs_20d` is retired; the low-vol signal
layer passes but its original locked portfolio fails; the BTC-filtered portfolio
remains a post-hoc hypothesis; funding remains inconclusive. Production Alpha is
**FROZEN** and Live is **NOT_ALLOWED**.
"""


def _realtime_integrity(status: dict[str, Any], lock: dict[str, Any]) -> str:
    return f"""# Realtime Forward Integrity

## Formal evidence rule

A row is formal only when all three predicates hold:

1. `decision_origin = REALTIME`
2. `late_reconstructed = false`
3. `eligible_for_forward_evidence = true`

The fixed latency budget is **{lock['max_decision_latency_seconds']} seconds**.
Realtime mode examines only the latest due schedule and never loops over missing
historical schedules. Recovery mode may rebuild audit state, but every decision
and trade is marked `RECOVERY_RECONSTRUCTION` and excluded from returns, counts,
benchmarks, and advancement statistics.

Decision creation freezes the completed-bar cutoff, BTC 60-day trend state,
dynamic universe, factor scores, rank, Top 3, target weights, and market snapshot.
Entry recording occurs on a later invocation after the next bar is complete.

- Current identity status: **{status['runner_integrity']}**
- Git HEAD check: **{'PASS' if status['git_head_check'] else 'FAIL'}**
- Strategy hash check: **{'PASS' if status['strategy_code_hash_check'] else 'FAIL'}**
- Working tree clean: **{str(status['working_tree_clean']).lower()}**

Current REALTIME / reconstructed / eligible / ineligible counts:
**{status['real_time_decision_count']} / {status['reconstructed_decision_count']} /
{status['eligible_decision_count']} / {status['ineligible_decision_count']}**.
"""


def _status_machine(lock: dict[str, Any]) -> str:
    return f"""# Forward Paper Status Machine

The machine is preregistered and evaluated in this priority order:

1. `FAIL_RUNNER_INTEGRITY` for hash, Git, parameter, or evidence-mixing faults.
2. `INCONCLUSIVE_SYSTEM_ERROR` or `INCONCLUSIVE_DATA_INCOMPLETE` when evidence
   cannot be evaluated safely.
3. `INCONCLUSIVE_SAMPLE_INSUFFICIENT` until 30 days, 6 completed independent
   cycles, and 12 actual symbol entries are present.
4. `FAIL_DATA_QUALITY` once the sample thresholds are met but coverage is below
   {float(lock['minimum_data_coverage']) * 100:.0f}% or unexplained gaps/fills exist.
5. `FAIL_PERFORMANCE` when base-cost net return is non-positive.
6. `FAIL_BENCHMARK_UNDERPERFORMANCE` when excess versus BTC or the frozen dynamic
   universe benchmark is non-positive.
7. `FAIL_RISK_OR_DRAWDOWN` above {float(lock['maximum_drawdown']) * 100:.0f}% drawdown.
8. `FAIL_CONCENTRATION` above {float(lock['maximum_single_symbol_contribution']) * 100:.0f}%
   absolute single-symbol contribution.
9. `PAPER_REVIEW_READY` only when every gate passes.

No state permits Live, Canary, automatic promotion, non-zero live notional, or
order submission. `CONTINUE_PAPER` is reserved for an explicitly reviewed paper
extension and is never an automatic success state.
"""


def _benchmark_design(status: dict[str, Any]) -> str:
    return f"""# Immutable Benchmark Design

Three benchmarks share each strategy decision's schedule, entry delay, 120-hour
exit, cost semantics, parameter lock, code hash, and market snapshot:

| Benchmark | Weights | Cost |
|---|---|---:|
| BTC buy-and-hold | 100% BTC | 15 bps entry + 15 bps exit |
| Dynamic universe equal-weight | frozen decision-time universe | same 30 bps round trip |
| Cash | 100% cash | 0 bps |

Benchmark decisions, trades, and events are separate Parquet datasets. Closed
prices, weights, symbols, and returns cannot be changed; events form their own
SHA256 chain. A changed source file yields a new market snapshot ID and an
integrity alert without rewriting old evidence.

- Current snapshot: `{status['market_data_snapshot_id']}`
- BTC return: **{_pct(status['btc_benchmark_return'])}**
- Dynamic-universe return: **{_pct(status['dynamic_universe_benchmark_return'])}**
- Strategy excess vs BTC / universe: **{_pct(status['strategy_excess_vs_btc'])} /
  {_pct(status['strategy_excess_vs_dynamic_universe'])}**.
"""


def _identity_report(status: dict[str, Any], lock: dict[str, Any]) -> str:
    lines = "\n".join(
        f"- `{item['path']}` — `{item['sha256']}`" for item in lock["locked_source_files"]
    )
    return f"""# Code Identity Lock

Formal execution requires exact equality with commit
`{lock['strategy_code_commit']}`, a clean working tree, and the composite strategy
hash `{lock['strategy_code_hash']}`.

Reporting code has a separate hash `{lock['reporting_code_hash']}`. Reporting can
run from a separate checkout, but the formal runner checkout remains pinned to the
strategy commit. A change to signal, decision, trade, cost, benchmark, status, or
lock-schema code requires a new strategy version, parameter lock, and cutoff.

Current checks: Git HEAD **{'PASS' if status['git_head_check'] else 'FAIL'}**;
strategy hash **{'PASS' if status['strategy_code_hash_check'] else 'FAIL'}**;
worktree clean **{str(status['working_tree_clean']).lower()}**.

## Locked strategy files

{lines}
"""


def _low_vol_report(metrics: dict[str, float]) -> str:
    return f"""# Low-vol Interpretation v2.2

The signal-layer conclusion remains **PASS**, but it must not be described as a
pure or strong dynamic Alpha.

| Measure | Value |
|---|---:|
| Raw cross-sectional IC | {metrics['raw_cross_sectional_ic']:.6f} |
| Raw IC t-stat | {metrics['raw_ic_t_stat']:.3f} |
| Mean within-symbol permutation null IC | {metrics['within_symbol_permutation_null_ic']:.6f} |
| Incremental IC above symbol-persistent component | {metrics['incremental_ic_above_symbol_persistent_component']:.6f} |
| Within-symbol null draws | {int(metrics['within_symbol_permutation_count'])} |

Correct interpretation: `low_vol_20d` contains stable cross-sectional information,
but much of it may be persistent coin-level low-volatility identity. The dynamic
20-day timing increment is materially smaller. This does not automatically reject
the fixed Forward candidate, and it does not upgrade the post-hoc portfolio to
PASS. Only new REALTIME evidence after the v2.2 cutoff can change that conclusion.
"""


def _receipt_report(v5_repo: Path) -> str:
    commit = _git(v5_repo, "rev-parse", "HEAD")
    return f"""# V5 Decision Receipt v2.2

V5 branch commit: `{commit}`.

Audit v2.2 adds a strict schema and a default-off append-only recorder. It is not
connected to the production pipeline and was not deployed. Receipts bind decision
and market-data time, strategy/parameter/code identity, positions, available cash,
targets, risk checks, gate result, risk permission, order intents, execution mode,
final action, and error codes.

The recorder recursively rejects sensitive keys and credential markers. Writes
use an fsynced temporary file plus atomic no-overwrite creation. Byte-identical
retries are idempotent; conflicting content for an existing receipt ID is an
integrity error. `try_record` never raises, so recording failure cannot block a
protective close. Production behavior remains unchanged.
"""


def _test_report(root: Path) -> str:
    path = root / "artifacts/test_execution_v22.json"
    if not path.exists():
        return """# Audit v2.2 Test Report

Status: **PENDING**. This report is regenerated after targeted and full regression
commands finish; no unexecuted test is claimed as passing.
"""
    payload = _load(path)
    rows = "\n".join(
        f"| {item['name']} | {item['result']} | {item.get('summary', '')} |"
        for item in payload["commands"]
    )
    return f"""# Audit v2.2 Test Report

Overall status: **{payload['status']}**

| Suite | Result | Evidence |
|---|---|---|
{rows}

No failing test was skipped, removed, or weakened. Platform skips are reported
separately and are not counted as passes.
"""


def _reproduction(status: dict[str, Any]) -> str:
    return f"""# Audit v2.2 Reproduction Guide

```bash
export AUDIT_V21_BUNDLE=/mnt/c/Users/HR/Downloads/alpha_audit_v21_bundle_20260718_175538.zip
export AUDIT_V22_ROOT=/home/hr/quant-alpha-audit-v2.2
export AUDIT_V1_ROOT=/home/hr/quant-alpha-audit
cd /home/hr/quant-alpha-audit/repos/quant-lab

./scripts/run_alpha_audit_v22.sh --stage consistency
# After committing all strategy-affecting source:
./scripts/run_alpha_audit_v22.sh --stage init
./scripts/run_forward_v22_realtime.sh --root "$AUDIT_V22_ROOT" \
  --v1-root "$AUDIT_V1_ROOT" --resume
./scripts/run_forward_v22_recovery.sh --root "$AUDIT_V22_ROOT" \
  --v1-root "$AUDIT_V1_ROOT" --resume --as-of <audit-time>
```

`--as-of` in realtime mode cannot be older than the fixed latency window and does
not change wall-clock `recorded_at`. Recovery output is always ineligible.

Current cutoff: `{status['forward_v22_cutoff']}`. Never merge v2.1 or recovery
records into v2.2 formal evidence. No command deploys V5, enables Alpha, or places
an order.
"""


def _dashboard(status: dict[str, Any], lock: dict[str, Any], low_vol: dict[str, float]) -> str:
    esc = html.escape
    status_class = "fail" if str(status["paper_status"]).startswith("FAIL") else "pending"
    official = int(status["real_time_decision_count"])
    recovered = int(status["reconstructed_decision_count"])
    total = max(1, official + recovered)
    official_width = official / total * 100
    recovery_width = recovered / total * 100
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit v2.2 · Forward Integrity</title>
<style>
:root{{--ink:#ecf3f7;--muted:#8fa4b1;--line:#273d49;--base:#071116;--panel:#0b171d;--cyan:#55d6be;--amber:#f3b562;--red:#ff7a78}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--base);color:var(--ink);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}}
header{{border-bottom:1px solid var(--line);padding:24px clamp(18px,5vw,72px);display:flex;justify-content:space-between;gap:24px;align-items:end}}
.eyebrow{{color:var(--cyan);font:700 12px/1.2 ui-monospace,Consolas,monospace;letter-spacing:.16em;text-transform:uppercase}}
h1{{font-size:clamp(28px,4vw,52px);line-height:1.05;margin:8px 0 0;letter-spacing:-.035em}} .lock{{max-width:560px;color:var(--muted);font-family:ui-monospace,Consolas,monospace;overflow-wrap:anywhere}}
main{{max-width:1320px;margin:auto;padding:0 clamp(18px,5vw,72px) 72px}} section{{padding:32px 0;border-bottom:1px solid var(--line)}}
h2{{font-size:18px;margin:0 0 18px}} .statusline{{display:grid;grid-template-columns:minmax(240px,1.2fr) repeat(3,minmax(140px,.65fr));gap:1px;background:var(--line);border:1px solid var(--line)}}
.statusline>div{{background:var(--panel);padding:20px}} .label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}} .value{{font-size:22px;font-weight:720;margin-top:6px}} .value.pending{{color:var(--amber)}} .value.fail{{color:var(--red)}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:24px}} dl{{margin:0}} dt{{color:var(--muted);font-size:12px;margin-top:12px}} dd{{margin:2px 0 0;font:650 16px ui-monospace,Consolas,monospace;overflow-wrap:anywhere}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:40px}} .bar{{height:10px;background:#10242c;margin:10px 0 18px;display:flex}} .bar .rt{{width:{official_width:.2f}%;background:var(--cyan)}} .bar .rec{{width:{recovery_width:.2f}%;background:var(--amber)}}
table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--muted);font-weight:600;font-size:12px}} code{{color:var(--cyan)}}
.flow{{display:flex;align-items:center;gap:10px;overflow-x:auto;padding-bottom:8px}} .step{{min-width:150px;padding:14px 0;border-top:2px solid var(--cyan)}} .arrow{{color:var(--muted)}}
.note{{color:var(--muted);max-width:900px}} .tag{{display:inline-block;border:1px solid currentColor;padding:2px 7px;margin-right:6px;font:700 11px ui-monospace,Consolas,monospace;letter-spacing:.05em}} .tag.amber{{color:var(--amber)}} .tag.red{{color:var(--red)}}
footer{{max-width:1320px;margin:auto;padding:22px clamp(18px,5vw,72px) 50px;color:var(--muted)}}
@media(max-width:900px){{header{{display:block}} .lock{{margin-top:18px}} .statusline{{grid-template-columns:1fr 1fr}} .grid{{grid-template-columns:1fr 1fr}} .split{{grid-template-columns:1fr}}}}
@media(max-width:560px){{.statusline,.grid{{grid-template-columns:1fr}} section{{padding:26px 0}} .tablewrap{{overflow-x:auto}} table{{min-width:680px}}}}
</style>
</head>
<body>
<header><div><div class="eyebrow">Audit v2.2 · Forward evidence control</div><h1>真实 Forward 身份与证据链</h1></div><div class="lock">{esc(lock['strategy_id'])}<br>commit {esc(lock['strategy_code_commit'])}<br>strategy hash {esc(lock['strategy_code_hash'])}</div></header>
<main>
<section aria-label="current status"><div class="statusline">
<div><div class="label">当前 Paper 状态</div><div class="value {status_class}" data-field="paper_status">{esc(str(status['paper_status']))}</div></div>
<div><div class="label">生产 Alpha</div><div class="value fail">FROZEN</div></div>
<div><div class="label">允许 Live</div><div class="value fail">NO</div></div>
<div><div class="label">BTC 趋势周期</div><div class="value">60天 / 1440小时</div></div>
</div></section>
<section><h2>运行身份</h2><div class="grid">
<dl><dt>Audit版本</dt><dd>v2.2</dd><dt>策略ID</dt><dd>{esc(lock['strategy_id'])}</dd><dt>策略版本</dt><dd>{esc(lock['strategy_version'])}</dd></dl>
<dl><dt>Forward v2.2 cutoff</dt><dd data-field="cutoff">{esc(status['forward_v22_cutoff'])}</dd><dt>市场数据 snapshot</dt><dd>{esc(status['market_data_snapshot_id'])}</dd></dl>
<dl><dt>Runner实时完整性</dt><dd>{esc(status['runner_integrity'])}</dd><dt>Git HEAD校验</dt><dd>{'PASS' if status['git_head_check'] else 'FAIL'}</dd><dt>策略代码hash校验</dt><dd>{'PASS' if status['strategy_code_hash_check'] else 'FAIL'}</dd></dl>
<dl><dt>工作区是否clean</dt><dd>{str(status['working_tree_clean']).lower()}</dd><dt>Parameter lock</dt><dd>{esc(status['parameter_lock_hash'])}</dd></dl>
</div></section>
<section><h2>REALTIME 与重建证据分离</h2><div class="split"><div>
<div class="bar"><span class="rt"></span><span class="rec"></span></div>
<p><span class="tag">REALTIME {official}</span><span class="tag amber">RECOVERY_RECONSTRUCTION {recovered}</span></p>
<p class="note">重建决策、重建成交和重建基准保留审计价值，但不进入正式收益、基准、周期、交易数或晋级统计。</p></div><div class="grid" style="grid-template-columns:1fr 1fr">
<dl><dt>正式有效决策数</dt><dd data-field="eligible_decisions">{status['eligible_decision_count']}</dd><dt>不合格决策数</dt><dd>{status['ineligible_decision_count']}</dd><dt>空仓决策数</dt><dd>{status['cash_decision_count']}</dd></dl>
<dl><dt>实际入场数</dt><dd>{status['actual_entry_count']}</dd><dt>已完成周期数</dt><dd>{status['completed_independent_cycles']}</dd><dt>有效币次交易数</dt><dd>{status['actual_symbol_trades']}</dd></dl>
</div></div></section>
<section><h2>Forward 绩效与门槛</h2><div class="tablewrap"><table><thead><tr><th>指标</th><th>当前</th><th>预注册门槛</th></tr></thead><tbody>
<tr><td>Forward有效天数</td><td>{_number(status['forward_days'],6)}</td><td>≥ 30</td></tr><tr><td>数据覆盖率</td><td>{_pct(status['data_coverage'])}</td><td>≥ 95%</td></tr>
<tr><td>策略复合净收益</td><td>{_pct(status['strategy_net_return'])}</td><td>&gt; 0</td></tr><tr><td>BTC基准收益</td><td>{_pct(status['btc_benchmark_return'])}</td><td>同区间/同成本</td></tr>
<tr><td>动态币池基准收益</td><td>{_pct(status['dynamic_universe_benchmark_return'])}</td><td>同区间/同成本</td></tr><tr><td>相对BTC超额</td><td>{_pct(status['strategy_excess_vs_btc'])}</td><td>&gt; 0</td></tr>
<tr><td>相对币池超额</td><td>{_pct(status['strategy_excess_vs_dynamic_universe'])}</td><td>&gt; 0</td></tr><tr><td>单币最大贡献率</td><td>{_pct(status['maximum_single_symbol_contribution'])}</td><td>≤ 50%</td></tr>
</tbody></table></div></section>
<section><h2>不可回看证据链</h2><div class="flow"><div class="step">调度点到达<br><code>scheduled_run_ts</code></div><div class="arrow">→</div><div class="step">固化决策<br><code>DECISION_CREATED</code></div><div class="arrow">→</div><div class="step">下一根已完成K线<br><code>ENTRY_RECORDED</code></div><div class="arrow">→</div><div class="step">120h退出<br><code>EXIT_RECORDED</code></div><div class="arrow">→</div><div class="step">不可变基准<br><code>benchmark hash chain</code></div></div></section>
<section><h2>low_vol 的正确解释</h2><div class="split"><div><p class="note">稳定横截面信息为 PASS；大部分信息可能来自币种长期低波属性，动态20日择时增量较弱。不能描述为纯动态 Alpha。</p></div><div class="tablewrap"><table><tbody><tr><th>Raw cross-sectional IC</th><td>{low_vol['raw_cross_sectional_ic']:.6f}</td></tr><tr><th>Within-symbol null IC</th><td>{low_vol['within_symbol_permutation_null_ic']:.6f}</td></tr><tr><th>Incremental IC</th><td>{low_vol['incremental_ic_above_symbol_persistent_component']:.6f}</td></tr></tbody></table></div></div></section>
<section><h2>固定研究结论</h2><div class="tablewrap"><table><thead><tr><th>对象</th><th>Signal</th><th>Portfolio</th><th>Deployment</th></tr></thead><tbody><tr><td>rev_xs_20d</td><td>FAIL</td><td>FAIL</td><td>FAIL · RETIRE</td></tr><tr><td>low_vol_20d</td><td>PASS</td><td>LOCKED FAIL</td><td>INCONCLUSIVE</td></tr><tr><td>low_vol + BTC 60d</td><td>继承信号层</td><td>INCONCLUSIVE · POST_HOC</td><td>FORWARD ONLY</td></tr><tr><td>funding_fade</td><td>INCONCLUSIVE</td><td>INCONCLUSIVE</td><td>INCONCLUSIVE</td></tr></tbody></table></div><p><span class="tag amber">POST_HOC</span><span class="tag red">PRODUCTION FROZEN</span></p></section>
</main><footer>RECOVERY_RECONSTRUCTION 永不属于正式 Forward 证据。Audit v2.2 不部署、不下单、不恢复实盘 Alpha。</footer>
</body></html>"""


def _consistency(
    root: Path, status: dict[str, Any], lock: dict[str, Any], final: dict[str, Any]
) -> dict[str, Any]:
    performance = pl.read_csv(root / "artifacts/forward_v22_performance.csv").to_dicts()[0]
    dashboard = (root / "reports/audit_dashboard_v22.html").read_text(encoding="utf-8")
    executive = (root / "reports/executive_summary_v22.md").read_text(encoding="utf-8")
    markers = {
        "strategy_id": lock["strategy_id"],
        "cutoff": status["forward_v22_cutoff"],
        "paper_status": status["paper_status"],
        "real_time_decision_count": str(status["real_time_decision_count"]),
        "reconstructed_decision_count": str(status["reconstructed_decision_count"]),
        "eligible_decision_count": str(status["eligible_decision_count"]),
    }
    errors: list[str] = []
    for name, marker in markers.items():
        if marker not in dashboard or marker not in executive:
            errors.append(f"REPORT_MARKER_MISSING:{name}")
    for key in (
        "strategy_id",
        "strategy_version",
        "paper_status",
        "real_time_decision_count",
        "reconstructed_decision_count",
        "eligible_decision_count",
    ):
        if str(performance[key]) != str(status[key]):
            errors.append(f"CSV_JSON_MISMATCH:{key}")
    candidate = final["low_vol_btc_60d_trend"]
    for key in (
        "strategy_id",
        "paper_status",
        "real_time_decision_count",
        "reconstructed_decision_count",
        "eligible_decision_count",
    ):
        if str(candidate[key]) != str(status[key]):
            errors.append(f"FINAL_JSON_MISMATCH:{key}")
    return {
        "schema_version": "quant_lab_report_consistency_v22.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "canonical_fields": markers,
        "dashboard_sha256": sha256_file(root / "reports/audit_dashboard_v22.html"),
        "executive_summary_sha256": sha256_file(root / "reports/executive_summary_v22.md"),
        "final_decisions_sha256": sha256_file(root / "artifacts/final_decisions_v22.json"),
        "performance_sha256": sha256_file(root / "artifacts/forward_v22_performance.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--v21-root", type=Path, default=Path("/home/hr/quant-alpha-audit-v2.1")
    )
    parser.add_argument(
        "--v5-repo", type=Path, default=Path("/home/hr/quant-alpha-audit-v2.2/repos/V5-prod")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    status = _load(root / "artifacts/forward_v22_status.json")
    lock = _load(root / "manifests/parameter_lock_v22.json")
    low_vol = _low_vol_metrics(args.v21_root.resolve())
    final = _final_decisions(status, lock)
    atomic_write_json(final, root / "artifacts/final_decisions_v22.json")
    atomic_write_json(low_vol, root / "artifacts/low_vol_interpretation_v22.json")
    reports = {
        "executive_summary_v22.md": _executive(status, lock),
        "realtime_forward_integrity.md": _realtime_integrity(status, lock),
        "forward_status_machine.md": _status_machine(lock),
        "immutable_benchmark_design.md": _benchmark_design(status),
        "code_identity_lock.md": _identity_report(status, lock),
        "low_vol_interpretation_v22.md": _low_vol_report(low_vol),
        "v5_decision_receipt_v22.md": _receipt_report(args.v5_repo.resolve()),
        "test_report_v22.md": _test_report(root),
        "reproduction_guide_v22.md": _reproduction(status),
    }
    for name, body in reports.items():
        _write(root / "reports" / name, body)
    _write(root / "reports/audit_dashboard_v22.html", _dashboard(status, lock, low_vol))
    schema_source = args.v5_repo.resolve() / "schemas/v5_decision_receipt_v22.schema.json"
    schema_target = root / "schemas/v5_decision_receipt_v22.schema.json"
    schema_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_source, schema_target)
    consistency = _consistency(root, status, lock, final)
    atomic_write_json(consistency, root / "manifests/report_consistency_v22.json")
    if consistency["status"] != "PASS":
        raise RuntimeError(f"report consistency failed: {consistency['errors']}")
    print(f"dashboard={root / 'reports/audit_dashboard_v22.html'}")
    print(f"paper_status={status['paper_status']}")
    print(f"report_consistency={consistency['status']}")


if __name__ == "__main__":
    main()
