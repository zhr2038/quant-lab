from __future__ import annotations

import io
import tarfile
from datetime import UTC, datetime
from pathlib import Path

SECRET_VALUE = "SHOULD_NOT_LEAK_12345"


def make_v5_bundle_fixture(
    path: Path,
    *,
    secret: bool = False,
    top_level_dir: bool = False,
) -> Path:
    files = {
        "raw/recent_runs/run_001/decision_audit.json": (
            '{"quant_lab": {"permission": "ALLOW"}, "decision": "paper"}'
        ),
        "raw/recent_runs/run_001/summary.json": '{"run_id": "run_001", "status": "complete"}',
        "raw/recent_runs/run_001/equity.jsonl": '{"ts": "2026-05-10T01:00:00Z", "equity": 1000}\n',
        "raw/recent_runs/run_001/trades.csv": (
            "ts,symbol,side,qty\n2026-05-10T01:00:00Z,BTC-USDT,buy,1\n"
        ),
        "raw/quant_lab/quant_lab_usage.jsonl": (
            '{"ts": "2026-05-10T01:00:00Z", "endpoint": "/v1/risk/live-permission"}\n'
        ),
        "raw/quant_lab/quant_lab_requests.jsonl": (
            '{"ts": "2026-05-10T01:01:00Z", "method": "GET", '
            '"path": "/v1/costs/estimate", "status_code": 200}\n'
        ),
        "raw/state/kill_switch.json": '{"enabled": false}',
        "raw/state/reconcile_status.json": '{"ok": true}',
        "raw/state/ledger_status.json": '{"ok": true}',
        "raw/state/auto_risk_eval.json": '{"level": "LOW"}',
        "summaries/window_summary.json": (
            '{"run_count": 1, "recent_24h_decision_audit_count": 1, '
            '"latest_24h_trade_count": 1, "last_72h_trade_count": 1, '
            '"last_72h_roundtrip_count": 1, "open_position_count": 1, '
            '"dust_residual_position_count": 0, "high_issue_count": 1, '
            '"medium_issue_count": 0}'
        ),
        "summaries/issues_to_fix.json": (
            '{"issues": [{"severity": "high", '
            '"type": "high_score_blocked_matured_without_label", '
            '"message": "needs labels"}]}'
        ),
        "summaries/quant_lab_compliance.csv": (
            "check,status,violation_count\nrisk_permission,ok,0\n"
        ),
        "summaries/quant_lab_cost_usage.csv": "symbol,cost_source,count\nBTC-USDT,quant_lab,3\n",
        "summaries/quant_lab_fallbacks.csv": (
            "fallback_reason,count\nquant_lab_unavailable_local_fallback,1\n"
        ),
        "summaries/paper_strategy_runs.csv": (
            "as_of_date,proposal_id,strategy_candidate,symbol,recommended_mode,"
            "would_enter,would_exit,would_size,paper_pnl,paper_pnl_bps,"
            "live_block_reason,required_paper_days,required_slippage_coverage\n"
            "2026-05-10,SOL_F4_VOLUME_EXPANSION_PAPER_V1,"
            "v5.f4_volume_expansion_entry,SOL-USDT,paper,true,false,100,0.42,42,"
            "[\"cost_source_not_actual_or_mixed\"],14,0.8\n"
        ),
        "summaries/paper_strategy_daily.csv": (
            "as_of_date,proposal_id,strategy_candidate,symbol,recommended_mode,"
            "paper_days,cumulative_paper_pnl_usdt,required_paper_days,"
            "required_slippage_coverage,live_eligible\n"
            "2026-05-10,SOL_F4_VOLUME_EXPANSION_PAPER_V1,"
            "v5.f4_volume_expansion_entry,SOL-USDT,paper,1,0.42,14,0.8,false\n"
        ),
        "summaries/paper_slippage_coverage.csv": (
            "as_of_date,proposal_id,strategy_candidate,symbol,paper_days,"
            "paper_slippage_coverage,required_slippage_coverage,coverage_status\n"
            "2026-05-10,SOL_F4_VOLUME_EXPANSION_PAPER_V1,"
            "v5.f4_volume_expansion_entry,SOL-USDT,1,0.0,0.8,"
            "insufficient_slippage_observations\n"
        ),
        "summaries/bnb_profit_lock_shadow.csv": (
            "run_id,entry_ts,symbol,entry_px,actual_exit_ts,actual_exit_px,"
            "actual_exit_net_bps,max_unrealized_bps,profit_lock_30bps_exit,"
            "profit_lock_50bps_exit,delayed_exit_6h,delayed_exit_12h,"
            "delayed_exit_24h,best_shadow_exit_policy\n"
            "run-bnb-shadow,2026-05-23T22:00:00Z,BNB/USDT,657.9,"
            "2026-05-24T22:01:00Z,651.3,-120.22,69.9,0.0,20.0,"
            "-40.0,29.28,-12.0,delayed_exit_12h\n"
        ),
        "summaries/bnb_negative_expectancy_attribution.csv": (
            "symbol,cycle_index,entry_ts,exit_ts,exit_reason,exit_priority,net_bps,"
            "attribution,entry_bad,exit_bad,min_hold_violation,gave_back_profit,"
            "trailing_too_early,unknown\n"
            "BNB/USDT,0,2026-05-28T22:00:59Z,2026-05-29T03:00:54Z,"
            "atr_trailing,soft,-142.89,\"[\"\"exit_bad\"\", \"\"min_hold_violation\"\"]\","
            "false,true,true,false,true,false\n"
        ),
        "summaries/negative_expectancy_attribution.csv": (
            "symbol,cycle_index,entry_ts,exit_ts,exit_reason,exit_priority,net_bps,"
            "attribution,entry_bad,exit_bad,min_hold_violation,gave_back_profit,"
            "trailing_too_early,unknown,adjusted_entry_expectancy_bps,raw_would_block,"
            "adjusted_would_block,would_unblock_if_adjusted,block_attribution_conflict\n"
            "BNB/USDT,0,2026-05-28T22:00:59Z,2026-05-29T03:00:54Z,"
            "atr_trailing,soft,-142.89,\"[\"\"exit_bad\"\", \"\"min_hold_violation\"\"]\","
            "false,true,true,false,true,false,0,true,false,true,true\n"
        ),
        "summaries/final_score_vs_alpha6_conflict.csv": (
            "run_id,ts_utc,symbol,final_score,alpha6_score,alpha6_side,"
            "f3_vol_adj_ret,f4_volume_expansion,f5_rsi_trend_confirm,"
            "expected_edge_bps,required_edge_bps,cost_gate_verified,final_decision,"
            "block_reason,no_signal_reason,negative_expectancy_net_bps,"
            "negative_expectancy_fast_fail_net_bps,future_4h_net_bps,"
            "future_8h_net_bps,future_12h_net_bps,future_24h_net_bps,"
            "label_status,missed_profit_flag\n"
            "run_bnb_conflict,2026-05-30T03:00:00Z,BNB/USDT,-0.17,0.994,buy,"
            "12,5.82,0.832,180,45,true,no_order,"
            "negative_expectancy_fast_fail_open_block,final_score_negative,"
            "-151.83,-142.89,500,700,900,1100,complete,true\n"
        ),
        "summaries/bnb_strong_alpha6_bypass_shadow.csv": (
            "run_id,ts_utc,would_bypass,alpha6_score,f3,f4,f5,expected_edge_bps,"
            "required_edge_bps,final_score,final_decision,block_reason,"
            "no_signal_reason,negative_expectancy_blocked,future_4h_net_bps,"
            "future_8h_net_bps,future_12h_net_bps,future_24h_net_bps,"
            "label_status,live_order_effect\n"
            "run_bnb_conflict,2026-05-30T03:00:00Z,true,0.994,12,5.82,0.832,"
            "180,45,-0.17,no_order,negative_expectancy_fast_fail_open_block,"
            "final_score_negative,true,500,700,900,1100,complete,"
            "read_only_no_live_order\n"
        ),
        "summaries/bnb_paper_strategy_runs.csv": (
            "as_of_date,proposal_id,strategy_candidate,symbol,recommended_mode,"
            "would_enter,would_exit,would_size,paper_pnl,paper_pnl_bps,"
            "live_block_reason,required_paper_days,required_slippage_coverage\n"
            "2026-05-10,BNB_F3_DOMINANT_ENTRY_PAPER_V1,"
            "v5.f3_dominant_entry,BNB-USDT,paper,true,false,100,0.42,42,"
            "[\"bnb_paper_only_no_live\"],14,0.8\n"
        ),
        "summaries/bnb_paper_strategy_daily.csv": (
            "as_of_date,proposal_id,strategy_candidate,symbol,recommended_mode,"
            "paper_days,entry_count,cumulative_paper_pnl_usdt,required_paper_days,"
            "required_slippage_coverage,live_eligible\n"
            "2026-05-10,BNB_F3_DOMINANT_ENTRY_PAPER_V1,"
            "v5.f3_dominant_entry,BNB-USDT,paper,1,1,0.42,14,0.8,false\n"
        ),
        "summaries/negative_expectancy_consistency.csv": (
            "symbol,negexp_closed_cycles,negexp_net_expectancy_bps,"
            "adjusted_entry_expectancy_bps,entry_bad_cycles,exit_bad_cycles,"
            "min_hold_violation_cycles,diagnosis\n"
            "BNB/USDT,3,-151.83,0,0,1,1,ok\n"
        ),
        "summaries/router_decisions.csv": "reason,count\nrisk_gate,2\n",
        "summaries/trades_roundtrips.csv": "symbol,pnl\nBTC-USDT,1.2\n",
        "summaries/open_positions.csv": "symbol,size\nBTC-USDT,0\n",
        "summaries/high_score_blocked_targets.csv": "symbol,score\nBTC-USDT,0.9\n",
        "summaries/high_score_blocked_outcomes.csv": (
            "symbol,matured,profitable\nBTC-USDT,true,true\n"
        ),
        "summaries/skipped_candidate_maturity_audit.csv": "symbol,matured\nETH-USDT,true\n",
        "reports/candidate_snapshot.csv": (
            "candidate_id,run_id,ts_utc,symbol,regime_state,risk_level,"
            "btc_trend_state,broad_market_positive_count,funding_state,volatility_bucket,"
            "current_position,"
            "current_weight,target_weight_raw,target_weight_after_risk,final_score,rank,"
            "f1_mom_5d,f2_mom_20d,f3_vol_adj_ret,f4_volume_expansion,"
            "f5_rsi_trend_confirm,alpha6_score,alpha6_side,ml_score,"
            "mean_reversion_score,expected_edge_bps,required_edge_bps,cost_bps,"
            "cost_source,eligible_before_filters,final_decision,block_reason,"
            "strategy_candidate\n"
            ",run_001,2026-05-10T01:00:00Z,BTC-USDT,trend,LOW,"
            "uptrend,3,neutral,medium,0,0,0.05,0.05,"
            "0.91,1,0.12,0.24,0.31,0.42,0.53,0.74,long,0.66,0.11,18,5,3.5,"
            "quant_lab,true,KEEP_SHADOW,risk_gate,v5.btc_leadership_probe_strict\n"
        ),
        "summaries/config_runtime_consumption_audit.csv": (
            "key,status\nalpha_threshold,not_consumed\n"
        ),
        "raw/config_live_prod.yaml": "risk: low\n",
        "raw/reports/effective_live_config.json": '{"risk": "low"}',
        "raw/logs/app.log": "normal log line\n",
    }
    if secret:
        files["raw/config_live_prod.yaml"] = (
            f"api_key: {SECRET_VALUE}\npassphrase: {SECRET_VALUE}\n"
        )
        files["raw/logs/app.log"] = f"token={SECRET_VALUE}\n"
    if top_level_dir:
        files = {
            f"v5_live_followup_bundle_20260510T140249Z/{name}": text
            for name, text in files.items()
        }
    return make_tar(path, files)


def make_tar(path: Path, files: dict[str, str], *, symlink: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(datetime(2026, 5, 10, tzinfo=UTC).timestamp())
            archive.addfile(info, io.BytesIO(data))
        if symlink is not None:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
            archive.addfile(info)
    return path
