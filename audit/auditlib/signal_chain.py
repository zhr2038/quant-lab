# ruff: noqa: E501
"""Build the code-and-config grounded current V5 signal-chain evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .paths import ARTIFACTS, REPORTS, snapshot_dir


def _read_dataset(name: str) -> pl.DataFrame:
    root = snapshot_dir() / "qlab" / "gold" / name
    files = sorted(
        file
        for file in root.rglob("*.parquet")
        if "._tmp" not in file.parts and not file.name.startswith(".")
    )
    if not files:
        return pl.DataFrame()
    return pl.scan_parquet([str(file) for file in files], missing_columns="insert").collect()


def _latest_non_bootstrap(frame: pl.DataFrame) -> dict:
    if frame.is_empty():
        return {}
    filtered = frame
    if "version" in frame.columns:
        candidate = frame.filter(pl.col("version").cast(pl.Utf8).str.to_lowercase() != "bootstrap")
        if not candidate.is_empty():
            filtered = candidate
    sort_col = next(
        (
            column
            for column in ("created_at", "as_of_ts", "source_bundle_ts")
            if column in filtered.columns
        ),
        None,
    )
    if sort_col:
        filtered = filtered.sort(sort_col)
    return filtered.tail(1).to_dicts()[0]


def build_current_signal_chain() -> tuple[Path, Path]:
    chain = [
        {
            "step": "raw_data",
            "repo": "V5-prod",
            "file": "src/data/okx_ccxt_provider.py",
            "class": "OKXCcxtProvider",
            "function": "fetch_ohlcv",
            "input": "OKX history candles",
            "output": "closed 1h MarketSeries bars",
            "timestamp_semantics": "confirm=0 and ts+timeframe>end are excluded; UTC closed bar",
            "stateful": False,
            "future_leakage_risk": "LOW after code check",
            "live_effect": "feeds V5 alpha computation",
        },
        {
            "step": "silver",
            "repo": "quant-lab",
            "file": "src/quant_lab/ingestion/market.py and lake writers",
            "class": "market ingestion",
            "function": "publish market_bar",
            "input": "exchange bars",
            "output": "silver/market_bar parquet",
            "timestamp_semantics": "event time and availability time are UTC",
            "stateful": True,
            "future_leakage_risk": "MEDIUM if availability time is ignored",
            "live_effect": "read-only research/lake input",
        },
        {
            "step": "feature",
            "repo": "quant-lab",
            "file": "src/quant_lab/features and src/quant_lab/factors/factory.py",
            "class": "Feature/Factor Factory",
            "function": "build factor values",
            "input": "closed bars and published funding",
            "output": "gold/feature_value and gold/factor_value",
            "timestamp_semantics": "event_time <= available_time; declared availability_lag_bars",
            "stateful": True,
            "future_leakage_risk": "MEDIUM; this audit adds independent shift tests",
            "live_effect": "advisory only",
        },
        {
            "step": "current_v5_factor",
            "repo": "V5-prod",
            "file": "src/strategy/multi_strategy_system.py",
            "class": "Alpha6FactorStrategy",
            "function": "_calculate_factors/_zscore_factors/generate_signals",
            "input": "per-symbol closed 1h OHLCV plus timestamped sentiment cache",
            "output": "cross-section-centered Alpha6 buy/sell signals",
            "timestamp_semantics": "features through bar t; execution is no earlier than next cycle/bar",
            "stateful": True,
            "future_leakage_risk": "LOW for OHLCV arithmetic; replay gap for sentiment and dynamic IC state",
            "live_effect": "55% strategy allocation before multi-strategy fusion",
        },
        {
            "step": "strategy_fusion",
            "repo": "V5-prod",
            "file": "src/alpha/alpha_engine.py and src/strategy/multi_strategy_system.py",
            "class": "AlphaEngine/StrategyOrchestrator",
            "function": "_init_multi_strategy/generate_combined_signals",
            "input": "TrendFollowing 20%, MeanReversion 25%, Alpha6 55%",
            "output": "fused long/exit candidate scores and target notionals",
            "timestamp_semantics": "current closed-bar cycle",
            "stateful": True,
            "future_leakage_risk": "LOW for timing; exact historical replay unavailable without state snapshots",
            "live_effect": "direct input to V5 portfolio construction",
        },
        {
            "step": "candidate_factors",
            "repo": "quant-lab audit branch",
            "file": "audit/auditlib/factors.py",
            "class": "causal factor functions",
            "function": "rev_xs_480/low_vol_480/funding_fade",
            "input": "bars through t; funding published by t",
            "output": "signal at feature_ts",
            "timestamp_semantics": "decision delay=1 bar; backward as-of funding",
            "stateful": False,
            "future_leakage_risk": "LOW after automated causality tests",
            "live_effect": "none; audit-only",
        },
        {
            "step": "label",
            "repo": "quant-lab",
            "file": "src/quant_lab/research/labels.py and audit/auditlib/evaluate.py",
            "class": "forward label builder",
            "function": "build_labels",
            "input": "feature_ts t and closed prices",
            "output": "entry t+1, exit t+1+h, forward_return",
            "timestamp_semantics": "feature_ts < decision_ts < label_ts",
            "stateful": False,
            "future_leakage_risk": "LOW after delay/temporal-order tests",
            "live_effect": "research evidence only",
        },
        {
            "step": "ic",
            "repo": "quant-lab",
            "file": "src/quant_lab/research/ic.py and audit/auditlib/ic_stats.py",
            "class": "IC statistics",
            "function": "compute_rank_ic_nonoverlap/compute_rank_ic_hac_tstat",
            "input": "cross-sectional signal and forward labels",
            "output": "naive, horizon-spaced, and HAC rank-IC significance",
            "timestamp_semantics": "HAC lag=horizon; non-overlap next>=last+horizon",
            "stateful": False,
            "future_leakage_risk": "overlap inflation corrected",
            "live_effect": "research evidence and gate input only",
        },
        {
            "step": "long_short_diagnostic",
            "repo": "quant-lab",
            "file": "src/quant_lab/factors/factory.py",
            "class": "FactorFactory",
            "function": "evaluate_and_publish_factor_evidence",
            "input": "ranked factor/labels",
            "output": "long-short diagnostic and long-only evidence",
            "timestamp_semantics": "forward label window after decision delay",
            "stateful": False,
            "future_leakage_risk": "diagnostic short leg must not be presented as spot-tradable return",
            "live_effect": "none directly",
        },
        {
            "step": "cost",
            "repo": "quant-lab",
            "file": "src/quant_lab/costs/model.py",
            "class": "CostModel",
            "function": "evaluate_live_universe_cost_coverage",
            "input": "actual fills, fees, slippage/spread and proxy buckets",
            "output": "p50/p75/p90 all-in costs and trust tier",
            "timestamp_semantics": "only costs available at each historical decision may be used",
            "stateful": True,
            "future_leakage_risk": "MEDIUM if current cost quantiles are retrofitted; audit labels them scenario assumptions",
            "live_effect": "shadow cost gate in current production mode",
        },
        {
            "step": "paper_ready",
            "repo": "quant-lab",
            "file": "src/quant_lab/factors/factory.py",
            "class": "FactorFactory",
            "function": "candidate decision",
            "input": "factor evidence, costs, coverage and quality",
            "output": "RESEARCH_ONLY/KEEP_SHADOW/PAPER_READY",
            "timestamp_semantics": "evidence as-of publish time",
            "stateful": True,
            "future_leakage_risk": "selection/multiple-testing risk; corrected in this audit",
            "live_effect": "paper readiness is not live readiness",
        },
        {
            "step": "alpha_gate",
            "repo": "quant-lab",
            "file": "src/quant_lab/gates/defaults.py",
            "class": "GateDecision",
            "function": "evaluate_alpha_gate",
            "input": "AlphaEvidence",
            "output": "gate status and reasons",
            "timestamp_semantics": "latest published evidence",
            "stateful": True,
            "future_leakage_risk": "staleness/version mismatch risk",
            "live_effect": "current V5 config consumes it in shadow mode only",
        },
        {
            "step": "risk_permission",
            "repo": "quant-lab/V5-prod",
            "file": "src/quant_lab/risk/publish.py and V5 src/quant_lab_client/guard.py",
            "class": "RiskPermission/QuantLabGuard",
            "function": "publish_risk_permission/evaluate guard",
            "input": "gates, cost trust, telemetry health",
            "output": "ALLOW/SELL_ONLY/ABORT plus ACTIVE_* status",
            "timestamp_semantics": "TTL and freshness checked against V5 telemetry",
            "stateful": True,
            "future_leakage_risk": "not statistical; freshness is operational risk",
            "live_effect": "record-only while mode=shadow; enforce would fail closed for new risk",
        },
        {
            "step": "v5_order",
            "repo": "V5-prod",
            "file": "src/execution/live_execution_engine.py",
            "class": "LiveExecutionEngine",
            "function": "submit gate/order routing",
            "input": "portfolio intents plus local and quant-lab permissions",
            "output": "OPEN_LONG/REBALANCE/CLOSE_LONG decisions",
            "timestamp_semantics": "post-signal live cycle",
            "stateful": True,
            "future_leakage_risk": "none; execution safety boundary",
            "live_effect": "SELL_ONLY blocks buys; ABORT blocks risk, but quant-lab is currently shadow",
        },
    ]

    risk = _latest_non_bootstrap(_read_dataset("risk_permission"))
    gate = _latest_non_bootstrap(_read_dataset("gate_decision"))
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "snapshot_id": snapshot_dir().name,
        "production_config": {
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
            "timeframe": "1h main, 4h auxiliary",
            "decision_delay_bars": 1,
            "multi_strategy": True,
            "strategy_allocations": {
                "TrendFollowing": 0.20,
                "MeanReversion": 0.25,
                "Alpha6Factor": 0.55,
            },
            "alpha6_static_weights": {
                "f1": 0.10,
                "f2": 0.30,
                "f3": 0.35,
                "f4": 0.15,
                "f5": 0.10,
                "f6_sentiment": 0.15,
            },
            "alpha6_direction": "relative_score>0 buy; relative_score<0 sell; threshold 0.10 after tanh",
            "lookbacks": {
                "trend_ma": [20, 60],
                "f1_hours": 119,
                "f2_hours": 479,
                "f3_returns": 480,
                "f4_hours": [24, 168],
                "rsi": 14,
            },
            "horizon": "no single forecast horizon in live signal; audit tests 24/72/120/480h",
            "rebalance_and_hold": "hourly cycle; top-k dropout hold_cycles=2; rank exit min 3h; regime exit min 4h; qualifying swing min 24h",
            "regime": {
                "ensemble": True,
                "HMM": 0.35,
                "funding": 0.40,
                "RSS": 0.25,
                "risk_off_position_multiplier": 0.0,
            },
            "alpha158_overlay": False,
            "dynamic_ic_weighting": True,
            "quant_lab_mode": "shadow",
            "quant_lab_fail_policy": "allow_local_fallback (shadow); fallback forbidden in enforce",
            "canary_enabled": False,
            "execution_mode": "live; dry_run=false (audited only, never invoked by this audit)",
        },
        "snapshot_gate": {
            key: gate.get(key)
            for key in (
                "alpha_id",
                "status",
                "passed",
                "reasons",
                "role",
                "baseline_status",
                "not_live_eligible",
            )
        },
        "snapshot_risk_permission": {
            key: risk.get(key)
            for key in (
                "permission",
                "permission_status",
                "enforceable",
                "reasons",
                "allowed_live_modes",
                "live_block_reasons",
                "system_safety_status",
                "core_alpha_gate_status",
            )
        },
        "replay_limit": "Exact current V5 history is unavailable because sentiment cache, dynamic IC weights, regime allocations, multi-strategy conflict state, optimizer state and portfolio state were not snapshotted per decision. Static Alpha6 results are a proxy, not current V5.",
        "chain": chain,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACTS / "signal_chain.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    config = payload["production_config"]
    lines = [
        "# Current V5 Signal Chain",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- snapshot_id: {payload['snapshot_id']}",
        "- authority: production commit and sanitized live config; documentation is secondary.",
        "- **Exact replay verdict: INCONCLUSIVE.** The audit-only Alpha6 static proxy is not the complete current V5 signal.",
        "",
        "## Current production truth",
        "",
        f"- Universe: {', '.join(config['symbols'])}; dynamic universe disabled for current live allocation.",
        f"- Signal: multi-strategy={config['multi_strategy']}; allocations={config['strategy_allocations']}.",
        f"- Alpha6: weights={config['alpha6_static_weights']}; direction={config['alpha6_direction']}.",
        f"- Lookbacks: {config['lookbacks']}; {config['horizon']}.",
        f"- Rebalance/hold: {config['rebalance_and_hold']}.",
        f"- Regime: {config['regime']}; Alpha158 overlay={config['alpha158_overlay']}; dynamic IC={config['dynamic_ic_weighting']}.",
        f"- Quant-lab: mode={config['quant_lab_mode']}; fail policy={config['quant_lab_fail_policy']}; canary={config['canary_enabled']}.",
        f"- Snapshot gate: {payload['snapshot_gate']}.",
        f"- Snapshot risk permission: {payload['snapshot_risk_permission']}.",
        "- `PAPER_READY` remains paper-only. Snapshot ABORT/SELL_ONLY evidence is advisory while V5 consumes quant-lab in shadow mode.",
        "",
        "## End-to-end map",
        "",
        "| Step | Repo/file | Function | Time semantics | State | Leakage / live impact |",
        "|---|---|---|---|---|---|",
    ]
    for row in chain:
        lines.append(
            f"| {row['step']} | {row['repo']} `{row['file']}` | `{row['class']}.{row['function']}` | "
            f"{row['timestamp_semantics']} | {row['stateful']} | {row['future_leakage_risk']}; {row['live_effect']} |"
        )
    lines.extend(["", "## Replay limitation", "", payload["replay_limit"], ""])
    markdown_path = REPORTS / "current_signal_chain.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, json_path
