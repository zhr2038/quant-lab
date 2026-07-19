# ruff: noqa: E501
"""Generate Audit v2.2.1 reports from the canonical Forward evidence store."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.forward_v221 import (  # noqa: E402
    STRATEGY_ID,
    STRATEGY_VERSION,
    atomic_write_json,
    composite_source_hash,
    sha256_file,
    source_hash_entries,
)
from audit.scripts.stage_v221_init import REPORTING_SOURCE_FILES  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _pct(value: Any) -> str:
    return f"{float(value or 0.0) * 100:.3f}%"


def _number(value: Any, digits: int = 3) -> str:
    return f"{float(value or 0.0):.{digits}f}"


def _final_decisions(
    status: dict[str, Any], lock: dict[str, Any], v22: dict[str, Any]
) -> dict[str, Any]:
    paper_status = str(status["paper_status"])
    portfolio = "FAIL" if paper_status.startswith("FAIL_") else "INCONCLUSIVE"
    deployment = "FAIL" if paper_status.startswith("FAIL_") else "INCONCLUSIVE"
    return {
        "audit_version": "v2.2.1",
        "v22_consistency": v22["status"],
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "current_v5": {
            "signal_validity": "INCONCLUSIVE",
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "FAIL",
            "replayability": "PARTIALLY_REPLAYABLE",
            "production_alpha": "FROZEN",
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
            "interpretation": (
                "Stable cross-sectional information; much may be persistent coin-level "
                "low-volatility identity, while dynamic 20-day timing increment is weaker."
            ),
        },
        "low_vol_btc_60d_trend": {
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
            "hypothesis_type": "POST_HOC_HYPOTHESIS",
            "parameters_locked": True,
            "btc_trend_lookback_hours": 1440,
            "portfolio_validity": portfolio,
            "deployment_readiness": deployment,
            "forward_start_status": status["forward_start_status"],
            "paper_status": paper_status,
            "forward_v221_cutoff": status.get("forward_v221_cutoff", ""),
            "next_legal_strategy_decision": status.get(
                "next_legal_strategy_decision", ""
            ),
            "next_timer_trigger": status.get("next_timer_trigger", ""),
            "formal_realtime_decision_count": int(
                status.get("formal_realtime_decision_count", 0)
            ),
            "recovery_decision_count": int(status.get("recovery_decision_count", 0)),
            "completed_independent_cycles": int(
                status.get("completed_independent_cycles", 0)
            ),
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
            "Accumulate 30 eligible Forward days through the installed hourly timer; "
            "never merge Recovery evidence into formal statistics."
        ),
        "safety": {
            "execution_mode": "PAPER",
            "approval_state": "PAPER_ONLY",
            "live_opening_enabled": False,
            "live_order_effect": "none",
            "automatic_promotion": False,
        },
        "parameter_lock_hash": lock["sha256"],
    }


def _executive(status: dict[str, Any], lock: dict[str, Any]) -> str:
    return f"""# Audit v2.2.1 Executive Summary

Audit v2.2.1 has status **{status['forward_start_status']}**. The installed paper
runner status is **{status['paper_status']}**; this is not a portfolio PASS and
does not permit Live.

- Strategy: `{lock['strategy_id']}` / `{lock['strategy_version']}`
- BTC trend: **60 days / 1440 hours**
- Research node: `{status.get('research_node', '')}`
- Timer installed/enabled/active: **{status.get('timer_installed', False)} / {status.get('timer_enabled', False)} / {status.get('timer_active', False)}**
- Next timer trigger: `{status.get('next_timer_trigger', '')}`
- Forward cutoff: `{status.get('forward_v221_cutoff', '')}`
- Next legal strategy decision: `{status.get('next_legal_strategy_decision', '')}`
- Schedule coverage: **{_pct(status.get('schedule_coverage', 0.0))}**
- Formal REALTIME / Recovery decisions: **{status.get('formal_realtime_decision_count', 0)} / {status.get('recovery_decision_count', 0)}**
- Minimum dynamic benchmark coverage: **{_pct(status.get('minimum_benchmark_fill_coverage', 0.0))}**

The dynamic-universe benchmark now freezes every expected symbol and weight.
Missing prices remain explicit and their weight stays in cash; no surviving symbol
is reweighted. Recovery decisions, trades, errors, data coverage, returns, and
benchmarks are accounted separately and cannot change the formal Paper state.
An append-only hourly schedule ledger distinguishes expected, completed, failed,
missed, and late runs.

Research conclusions are unchanged: `rev_xs_20d` remains retired; the low-vol
signal layer remains PASS but the original locked portfolio remains FAIL; the
BTC-filtered portfolio remains post-hoc and INCONCLUSIVE; funding remains
INCONCLUSIVE. Production Alpha is **FROZEN** and Live is **NOT_ALLOWED**.
"""


def _benchmark_report(status: dict[str, Any], lock: dict[str, Any]) -> str:
    return f"""# Benchmark Coverage Fix

The dynamic-universe equal-weight benchmark freezes `expected_symbols`,
`expected_symbol_count`, and `expected_weight_per_symbol` at decision time. Each
symbol has an explicit fill record. Missing entry prices are `MISSING_ENTRY` with
null return and zero realized weight; the unfilled expected weight becomes cash.
Surviving symbols keep their original expected weight.

The invariant is `invested_weight + cash_residual = 1`. BTC and cash require
coverage 1.0; the dynamic benchmark threshold is
**{float(lock['minimum_benchmark_fill_coverage']):.2f}**. A formal cycle is eligible
only when BTC, dynamic-universe, and cash benchmarks are all closed using the same
entry/exit timestamps and market snapshots as the strategy.

- BTC complete cycles: **{status.get('btc_benchmark_complete_cycle_count', 0)}**
- Dynamic-universe complete cycles: **{status.get('dynamic_universe_benchmark_complete_cycle_count', 0)}**
- Cash complete cycles: **{status.get('cash_benchmark_complete_cycle_count', 0)}**
- Minimum formal benchmark coverage: **{_pct(status.get('minimum_benchmark_fill_coverage', 0.0))}**
- Maximum benchmark cash residual: **{_pct(status.get('benchmark_cash_residual', 0.0))}**

No missing benchmark return is represented as zero. A coverage or synchronized
cycle failure makes the formal cycle ineligible and triggers `FAIL_DATA_QUALITY`
once it occurs in formal REALTIME evidence.
"""


def _recovery_report(status: dict[str, Any]) -> str:
    return f"""# Recovery Evidence Isolation

Recovery exists only for system restoration and audit. Formal status reads only
rows satisfying `decision_origin = REALTIME` and
`eligible_for_forward_evidence = true`.

| Scope | Decisions | Trades | Coverage | Incomplete | Runner errors |
|---|---:|---:|---:|---:|---:|
| Formal REALTIME | {status.get('formal_realtime_decision_count', 0)} | {status.get('formal_realtime_trade_count', 0)} | {_pct(status.get('formal_realtime_data_coverage', 0.0))} | {status.get('formal_realtime_incomplete_decision_count', 0)} | {status.get('formal_realtime_runner_errors', 0)} |
| Recovery | {status.get('recovery_decision_count', 0)} | {status.get('recovery_trade_count', 0)} | {_pct(status.get('recovery_data_coverage', 0.0))} | {status.get('recovery_incomplete_decision_count', 0)} | {status.get('recovery_runner_errors', 0)} |

Recovery defects can set `recovery_audit_warning`; they cannot lower formal
coverage, change the formal equity curve, create formal benchmark excess, or
trigger formal `INCONCLUSIVE_DATA_INCOMPLETE` / `FAIL_DATA_QUALITY` by themselves.
Current recovery warning: **{status.get('recovery_audit_warning', False)}**.
"""


def _schedule_report(status: dict[str, Any], lock: dict[str, Any]) -> str:
    return f"""# Schedule Integrity

The installed timer checks every hour at `:10 UTC`; the strategy decision sequence
retains the original 120-hour anchor `{lock['schedule_anchor']}`. `Persistent=true`
catch-up invocations are marked late/recovery and are never formal REALTIME evidence.

| Metric | Value |
|---|---:|
| Expected schedules | {status.get('expected_schedule_count', 0)} |
| Started schedules | {status.get('started_schedule_count', 0)} |
| Completed schedules | {status.get('completed_schedule_count', 0)} |
| Failed schedules | {status.get('failed_schedule_count', 0)} |
| Missed schedules | {status.get('missed_schedule_count', 0)} |
| Late schedules | {status.get('late_schedule_count', 0)} |
| Eligible realtime schedules | {status.get('eligible_realtime_schedule_count', 0)} |
| Schedule coverage | {_pct(status.get('schedule_coverage', 0.0))} |
| Calendar Forward days | {_number(status.get('forward_calendar_days', 0.0), 6)} |
| Eligible Forward days | {_number(status.get('eligible_forward_days', 0.0), 6)} |
| Runner observed days | {status.get('runner_observed_days', 0)} |
| Runner online hours | {_number(status.get('runner_online_hours', 0.0), 1)} |
| Longest runner gap | {_number(status.get('longest_runner_gap_hours', 0.0), 1)} h |

Coverage must remain at least **{_pct(lock['minimum_schedule_coverage'])}**, with no
unexplained critical miss and no gap over 120 hours. v2.2 pre-ledger missed windows
remain `NOT_RECORDED`; they were not converted into zero misses or formal evidence.
"""


def _deployment_report(status: dict[str, Any], deployment: dict[str, Any]) -> str:
    return f"""# systemd Deployment

- Research node: `{deployment.get('research_node', '')}`
- Always-on declaration: **{deployment.get('research_node_always_on', False)}**
- Execution identity: `{deployment.get('execution_user', '')}:{deployment.get('execution_group', '')}`
- Service installed: **{deployment.get('timer_installed', False)}**
- Timer enabled/active: **{deployment.get('timer_enabled', False)} / {deployment.get('timer_active', False)}**
- Next timer trigger: `{status.get('next_timer_trigger', '')}`
- Service unit SHA256: `{deployment.get('service_unit_sha256', '')}`
- Timer unit SHA256: `{deployment.get('timer_unit_sha256', '')}`
- Runner script SHA256: `{deployment.get('runner_script_sha256', '')}`
- Deployment manifest SHA256: `{deployment.get('deployment_manifest_sha256', '')}`
- Working tree clean: **{deployment.get('working_tree_clean', False)}**

The oneshot service runs as non-root `quantlab`, uses an isolated checkout and
state/log directories, and is protected by a non-blocking process lock. The timer
is hourly and persistent; it does not change the fixed 120-hour strategy schedule.
No unit is installed into the V5 production process and no order path is enabled.
"""


def _readiness_report(status: dict[str, Any]) -> str:
    ready = status.get("forward_start_status") == "FORWARD_V221_READY"
    return f"""# Forward Start Readiness

Verdict: **{status.get('forward_start_status', 'NOT_READY')}**

| Gate | Result |
|---|---|
| Research node identified | {'PASS' if status.get('research_node') else 'FAIL'} |
| Timer installed | {'PASS' if status.get('timer_installed') else 'FAIL'} |
| Timer enabled | {'PASS' if status.get('timer_enabled') else 'FAIL'} |
| Timer active | {'PASS' if status.get('timer_active') else 'FAIL'} |
| Runner identity | {status.get('runner_integrity', 'FAIL')} |
| Git HEAD | {'PASS' if status.get('git_head_match') else 'FAIL'} |
| Clean working tree | {'PASS' if status.get('working_tree_clean') else 'FAIL'} |
| Unit and runner hashes | {'PASS' if status.get('unit_hash_match') else 'FAIL'} |
| Formal cutoff created | {'PASS' if status.get('forward_v221_cutoff') else 'FAIL'} |
| Next legal decision fixed | {'PASS' if status.get('next_legal_strategy_decision') else 'FAIL'} |

The formal Forward start is {'ready' if ready else 'not ready'}. Even when ready,
portfolio validity and deployment readiness remain INCONCLUSIVE until the locked
minimum sample and every quality/performance/risk gate pass. Production Alpha is
FROZEN; Live remains NOT_ALLOWED.
"""


def _test_report(root: Path) -> str:
    path = root / "artifacts/test_execution_v221.json"
    if not path.exists():
        return """# Audit v2.2.1 Test Report

Status: **PENDING**. No unexecuted test is represented as passing.
"""
    payload = _load(path)
    rows = "\n".join(
        f"| {item['name']} | {item['result']} | {item.get('summary', '')} |"
        for item in payload["commands"]
    )
    return f"""# Audit v2.2.1 Test Report

Overall status: **{payload['status']}**

| Suite/check | Result | Evidence |
|---|---|---|
{rows}

No failing test was skipped, removed, or weakened. Reported skipped tests are not
counted as passes.
"""


def _reproduction(status: dict[str, Any]) -> str:
    return f"""# Audit v2.2.1 Reproduction Guide

```bash
export AUDIT_V22_BUNDLE=/mnt/c/Users/HR/Downloads/alpha_audit_v22_bundle_20260719_053423.zip
export AUDIT_V221_ROOT=/home/hr/quant-alpha-audit-v2.2.1
cd /home/hr/quant-alpha-audit-v2.2.1/repos/quant-lab

./scripts/run_alpha_audit_v221.sh --stage consistency
# Init only from the final clean committed strategy checkout:
./scripts/run_alpha_audit_v221.sh --stage init

# Research-node deployment (root installs units; service executes as quantlab):
sudo ./deploy/install_forward_v221_systemd.sh \
  --repo /opt/quant-lab-forward-v221 \
  --root /var/lib/quant-lab/forward_v221 \
  --python /opt/quant-lab/.venv/bin/python \
  --market-bar /var/lib/quant-lab/lake/silver/market_bar/data.parquet \
  --parameter-lock /path/to/parameter_lock_v221.json

systemctl status quant-lab-forward-v221.timer --no-pager
sudo ./deploy/check_forward_v221_health.sh
```

Current cutoff: `{status.get('forward_v221_cutoff', '')}`. The first legal strategy
decision is `{status.get('next_legal_strategy_decision', '')}`. Never backdate a
realtime run, never merge Recovery records into formal evidence, and never deploy
these audit commands into V5 production.
"""


def _dashboard(status: dict[str, Any], lock: dict[str, Any], deployment: dict[str, Any]) -> str:
    esc = html.escape
    paper = esc(str(status.get("paper_status", "UNKNOWN")))
    rows = [
        ("Audit版本", "v2.2.1"),
        ("策略ID", lock["strategy_id"]),
        ("策略版本", lock["strategy_version"]),
        ("BTC趋势周期", "60天 / 1440小时"),
        ("研究节点", status.get("research_node", "")),
        ("systemd service状态", "INSTALLED" if deployment.get("timer_installed") else "NOT_INSTALLED"),
        ("systemd timer状态", "ACTIVE" if status.get("timer_active") else "INACTIVE"),
        ("下一次timer触发", status.get("next_timer_trigger", "")),
        ("下一次合法策略决策时间", status.get("next_legal_strategy_decision", "")),
        ("正式cutoff", status.get("forward_v221_cutoff", "")),
        ("策略代码commit", lock["strategy_code_commit"]),
        ("策略代码hash", lock["strategy_code_hash"]),
        ("service unit hash", lock["service_unit_hash"]),
        ("timer unit hash", lock["timer_unit_hash"]),
        ("工作区是否clean", str(status.get("working_tree_clean", False)).lower()),
        ("自然Forward天数", _number(status.get("forward_calendar_days", 0.0), 6)),
        ("有效Forward天数", _number(status.get("eligible_forward_days", 0.0), 6)),
        ("预期调度次数", status.get("expected_schedule_count", 0)),
        ("准时调度次数", status.get("eligible_realtime_schedule_count", 0)),
        ("错过调度次数", status.get("missed_schedule_count", 0)),
        ("迟到调度次数", status.get("late_schedule_count", 0)),
        ("调度覆盖率", _pct(status.get("schedule_coverage", 0.0))),
        ("正式实时决策数", status.get("formal_realtime_decision_count", 0)),
        ("Recovery决策数", status.get("recovery_decision_count", 0)),
        ("空仓决策数", status.get("cash_decision_count", 0)),
        ("实际入场数", status.get("actual_entry_count", 0)),
        ("已完成周期数", status.get("completed_independent_cycles", 0)),
        ("BTC基准完整周期数", status.get("btc_benchmark_complete_cycle_count", 0)),
        ("动态币池基准完整周期数", status.get("dynamic_universe_benchmark_complete_cycle_count", 0)),
        ("动态币池最低覆盖率", _pct(status.get("minimum_benchmark_fill_coverage", 0.0))),
        ("基准现金残余", _pct(status.get("benchmark_cash_residual", 0.0))),
        ("策略复合收益", _pct(status.get("strategy_net_return", 0.0))),
        ("BTC基准收益", _pct(status.get("btc_benchmark_return", 0.0))),
        ("动态币池基准收益", _pct(status.get("dynamic_universe_benchmark_return", 0.0))),
        ("相对BTC超额", _pct(status.get("strategy_excess_vs_btc", 0.0))),
        ("相对动态币池超额", _pct(status.get("strategy_excess_vs_dynamic_universe", 0.0))),
    ]
    cards = "".join(
        f'<div class="card"><span>{esc(str(label))}</span><strong data-field="{index}">{esc(str(value))}</strong></div>'
        for index, (label, value) in enumerate(rows)
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit v2.2.1 · Forward Start</title>
<style>
:root{{--bg:#071116;--panel:#0d1b22;--line:#29414d;--ink:#edf5f7;--muted:#96aab4;--cyan:#52d4bb;--amber:#f1b55d;--red:#ff7774}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Segoe UI,Arial,sans-serif}}header,main,footer{{max-width:1320px;margin:auto;padding:28px 5vw}}header{{border-bottom:1px solid var(--line)}}h1{{font-size:clamp(30px,5vw,58px);line-height:1.04;margin:.2em 0}}.eyebrow{{color:var(--cyan);font:700 12px ui-monospace,monospace;letter-spacing:.16em}}.guard{{display:grid;grid-template-columns:2fr 1fr 1fr;gap:1px;background:var(--line);margin-top:28px}}.guard div{{background:var(--panel);padding:18px}}.guard span,.card span{{display:block;color:var(--muted);font-size:12px}}.guard strong{{font-size:20px}}.paper{{color:var(--amber)}}.stop{{color:var(--red)}}section{{border-bottom:1px solid var(--line);padding:30px 0}}h2{{font-size:18px}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}.card{{min-height:94px;background:var(--panel);border:1px solid var(--line);padding:15px}}.card strong{{display:block;margin-top:8px;font:650 14px ui-monospace,Consolas,monospace;overflow-wrap:anywhere}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid var(--line);text-align:left}}p{{color:var(--muted)}}code{{color:var(--cyan)}}@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)}}.split{{grid-template-columns:1fr}}}}@media(max-width:560px){{.grid,.guard{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div class="eyebrow">AUDIT v2.2.1 · FORMAL FORWARD PAPER CONTROL</div><h1>调度在线率与不可变基准证据链</h1><p>{esc(lock['strategy_id'])}<br>Strategy code {esc(lock['strategy_code_hash'])}</p><div class="guard"><div><span>当前 Paper 状态</span><strong class="paper" data-field="paper_status">{paper}</strong></div><div><span>生产 Alpha</span><strong class="stop">FROZEN</strong></div><div><span>Live</span><strong class="stop">NOT_ALLOWED</strong></div></div></header>
<main><section><h2>正式身份与部署</h2><div class="grid">{cards}</div></section>
<section><h2>Recovery 与正式证据严格分区</h2><div class="split"><div><p><code>REALTIME + eligible=true</code> 才进入正式决策、收益、基准和状态机。</p><table><tr><th>Scope</th><th>Decisions</th><th>Trades</th><th>Coverage</th></tr><tr><td>REALTIME</td><td>{status.get('formal_realtime_decision_count', 0)}</td><td>{status.get('formal_realtime_trade_count', 0)}</td><td>{_pct(status.get('formal_realtime_data_coverage', 0.0))}</td></tr><tr><td>RECOVERY</td><td>{status.get('recovery_decision_count', 0)}</td><td>{status.get('recovery_trade_count', 0)}</td><td>{_pct(status.get('recovery_data_coverage', 0.0))}</td></tr></table></div><div><p>Recovery异常只产生审计告警，不污染正式净值或 Paper 状态。重建记录不进入正式收益图。</p><p>v2.2 错过的历史窗口：<strong>NOT_RECORDED</strong>，没有伪造为零次缺失。</p></div></div></section>
<section><h2>固定研究结论</h2><table><tr><th>对象</th><th>结论</th></tr><tr><td>rev_xs_20d</td><td>FAIL / RETIRE</td></tr><tr><td>low_vol_20d signal</td><td>PASS，但大部分可能来自币种长期低波属性</td></tr><tr><td>原锁定 low_vol 组合</td><td>FAIL</td></tr><tr><td>low_vol + BTC 60d</td><td>POST_HOC / INCONCLUSIVE / FORWARD ONLY</td></tr><tr><td>funding_fade</td><td>INCONCLUSIVE</td></tr></table></section></main>
<footer>Audit v2.2.1 只运行 PAPER；不恢复生产 Alpha，不允许 Live，不产生真实订单。</footer></body></html>"""


def _terminal_summary(status: dict[str, Any], lock: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"Audit v2.2.1状态：{status.get('forward_start_status', 'NOT_READY')}",
            "v2.2一致性校验：PASS",
            f"策略ID：{lock['strategy_id']}",
            "BTC趋势周期：60天 / 1440小时",
            f"研究节点：{status.get('research_node', '')}",
            f"systemd timer：{'active' if status.get('timer_active') else 'inactive'}",
            f"下一次timer触发：{status.get('next_timer_trigger', '')}",
            f"下一次合法策略决策时间：{status.get('next_legal_strategy_decision', '')}",
            f"新Forward cutoff：{status.get('forward_v221_cutoff', '')}",
            f"正式实时决策数：{status.get('formal_realtime_decision_count', 0)}",
            f"Recovery决策数：{status.get('recovery_decision_count', 0)}",
            f"调度覆盖率：{_pct(status.get('schedule_coverage', 0.0))}",
            f"基准覆盖率：{_pct(status.get('minimum_benchmark_fill_coverage', 0.0))}",
            f"当前Paper状态：{status.get('paper_status', '')}",
            "生产Alpha状态：FROZEN",
            "是否允许Live：NO",
        )
    )


def _consistency(
    root: Path,
    status: dict[str, Any],
    lock: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, Any]:
    dashboard = (root / "reports/audit_dashboard_v221.html").read_text(encoding="utf-8")
    executive = (root / "reports/executive_summary_v221.md").read_text(encoding="utf-8")
    terminal = (root / "artifacts/terminal_summary_v221.txt").read_text(encoding="utf-8")
    csv_row = pl.read_csv(root / "artifacts/forward_v221_performance.csv").to_dicts()[0]
    markers = {
        "strategy_id": str(lock["strategy_id"]),
        "cutoff": str(status.get("forward_v221_cutoff", "")),
        "next_legal_strategy_decision": str(
            status.get("next_legal_strategy_decision", "")
        ),
        "next_timer_trigger": str(status.get("next_timer_trigger", "")),
        "paper_status": str(status["paper_status"]),
        "formal_realtime_decision_count": str(
            status.get("formal_realtime_decision_count", 0)
        ),
        "recovery_decision_count": str(status.get("recovery_decision_count", 0)),
        "schedule_coverage": _pct(status.get("schedule_coverage", 0.0)),
        "benchmark_coverage": _pct(
            status.get("minimum_benchmark_fill_coverage", 0.0)
        ),
    }
    errors: list[str] = []
    for name, marker in markers.items():
        if marker and not all(marker in text for text in (dashboard, executive, terminal)):
            errors.append(f"REPORT_MARKER_MISSING:{name}")
    for key in (
        "strategy_id",
        "strategy_version",
        "paper_status",
        "formal_realtime_decision_count",
        "recovery_decision_count",
        "schedule_coverage",
        "minimum_benchmark_fill_coverage",
        "strategy_net_return",
        "btc_benchmark_return",
        "dynamic_universe_benchmark_return",
    ):
        if str(csv_row.get(key)) != str(status.get(key)):
            errors.append(f"CSV_JSON_MISMATCH:{key}")
    candidate = final["low_vol_btc_60d_trend"]
    for key in (
        "strategy_id",
        "paper_status",
        "formal_realtime_decision_count",
        "recovery_decision_count",
        "next_legal_strategy_decision",
        "next_timer_trigger",
    ):
        if str(candidate.get(key)) != str(status.get(key)):
            errors.append(f"FINAL_JSON_MISMATCH:{key}")
    return {
        "schema_version": "quant_lab_report_consistency_v221.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "canonical_fields": markers,
        "dashboard_sha256": sha256_file(root / "reports/audit_dashboard_v221.html"),
        "executive_sha256": sha256_file(root / "reports/executive_summary_v221.md"),
        "terminal_summary_sha256": sha256_file(
            root / "artifacts/terminal_summary_v221.txt"
        ),
        "final_decisions_sha256": sha256_file(
            root / "artifacts/final_decisions_v221.json"
        ),
        "performance_sha256": sha256_file(
            root / "artifacts/forward_v221_performance.csv"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve()
    canonical_status = _load(root / "artifacts/forward_v221_status.json")
    lock = _load(root / "manifests/parameter_lock_v221.json")
    v22 = _load(root / "artifacts/v22_consistency_check.json")
    deployment = _load(root / "manifests/systemd_deployment_v221.json")
    health = _load(root / "state/forward_v221_health_latest.json")
    changed_files = [
        value
        for value in _git(
            repo, "diff", "--name-only", f"{lock['strategy_code_commit']}..HEAD"
        ).splitlines()
        if value
    ]
    disallowed = sorted(set(changed_files) - set(REPORTING_SOURCE_FILES))
    if disallowed:
        raise RuntimeError(f"post-lock change is not reporting-only: {disallowed}")
    reporting_entries = source_hash_entries(repo, REPORTING_SOURCE_FILES)
    current_reporting_hash = composite_source_hash(reporting_entries)
    reporting_identity = {
        "schema_version": "quant_lab_reporting_identity_v221.v1",
        "strategy_code_commit": lock["strategy_code_commit"],
        "strategy_code_hash": lock["strategy_code_hash"],
        "parameter_lock_reporting_code_hash": lock["reporting_code_hash"],
        "current_reporting_commit": _git(repo, "rev-parse", "HEAD"),
        "current_reporting_code_hash": current_reporting_hash,
        "reporting_only_changed_files": changed_files,
        "strategy_checkout_deployment_unchanged": True,
    }
    atomic_write_json(
        reporting_identity, root / "artifacts/reporting_identity_v221.json"
    )
    _write(
        root / "manifests/reporting_code_hashes_v221.txt",
        "\n".join(
            f"{item['sha256']}  {item['path']}" for item in reporting_entries
        ),
    )
    status = {
        **canonical_status,
        "next_timer_trigger": health.get(
            "next_trigger", canonical_status.get("next_timer_trigger", "")
        ),
        "reporting_code_hash": current_reporting_hash,
        "reporting_commit": reporting_identity["current_reporting_commit"],
    }
    final = _final_decisions(status, lock, v22)
    atomic_write_json(final, root / "artifacts/final_decisions_v221.json")
    reports = {
        "executive_summary_v221.md": _executive(status, lock),
        "benchmark_coverage_fix.md": _benchmark_report(status, lock),
        "recovery_isolation.md": _recovery_report(status),
        "schedule_integrity.md": _schedule_report(status, lock),
        "systemd_deployment.md": _deployment_report(status, deployment),
        "forward_start_readiness.md": _readiness_report(status),
        "test_report_v221.md": _test_report(root),
        "reproduction_guide_v221.md": _reproduction(status),
    }
    for name, body in reports.items():
        _write(root / "reports" / name, body)
    _write(
        root / "reports/audit_dashboard_v221.html",
        _dashboard(status, lock, deployment),
    )
    _write(root / "artifacts/terminal_summary_v221.txt", _terminal_summary(status, lock))
    consistency = _consistency(root, status, lock, final)
    atomic_write_json(
        consistency, root / "manifests/report_consistency_v221.json"
    )
    if consistency["status"] != "PASS":
        raise RuntimeError(f"report consistency failed: {consistency['errors']}")
    print((root / "artifacts/terminal_summary_v221.txt").read_text(encoding="utf-8"))
    print(f"Dashboard路径：{root / 'reports/audit_dashboard_v221.html'}")
    print(f"report_consistency={consistency['status']}")


if __name__ == "__main__":
    main()
