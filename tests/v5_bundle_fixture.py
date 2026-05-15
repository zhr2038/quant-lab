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
        "summaries/router_decisions.csv": "reason,count\nrisk_gate,2\n",
        "summaries/trades_roundtrips.csv": "symbol,pnl\nBTC-USDT,1.2\n",
        "summaries/open_positions.csv": "symbol,size\nBTC-USDT,0\n",
        "summaries/high_score_blocked_targets.csv": "symbol,score\nBTC-USDT,0.9\n",
        "summaries/high_score_blocked_outcomes.csv": (
            "symbol,matured,profitable\nBTC-USDT,true,true\n"
        ),
        "summaries/skipped_candidate_maturity_audit.csv": "symbol,matured\nETH-USDT,true\n",
        "reports/candidate_snapshot.csv": (
            "candidate_id,run_id,ts_utc,symbol,regime_state,risk_level,current_position,"
            "current_weight,target_weight_raw,target_weight_after_risk,final_score,rank,"
            "f1_mom_5d,f2_mom_20d,f3_vol_adj_ret,f4_volume_expansion,"
            "f5_rsi_trend_confirm,alpha6_score,alpha6_side,ml_score,"
            "mean_reversion_score,expected_edge_bps,required_edge_bps,cost_bps,"
            "cost_source,eligible_before_filters,final_decision,block_reason,"
            "strategy_candidate\n"
            ",run_001,2026-05-10T01:00:00Z,BTC-USDT,trend,LOW,0,0,0.05,0.05,"
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
