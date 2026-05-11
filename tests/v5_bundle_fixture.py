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
        "raw/state/kill_switch.json": '{"enabled": false}',
        "raw/state/reconcile_status.json": '{"ok": true}',
        "raw/state/ledger_status.json": '{"ok": true}',
        "raw/state/auto_risk_eval.json": '{"level": "LOW"}',
        "summaries/window_summary.json": '{"run_count_72h": 1, "trade_count_72h": 1}',
        "summaries/issues_to_fix.json": (
            '{"issues": [{"severity": "high", '
            '"type": "high_score_blocked_matured_without_label", '
            '"message": "needs labels"}]}'
        ),
        "summaries/router_decisions.csv": "reason,count\nrisk_gate,2\n",
        "summaries/trades_roundtrips.csv": "symbol,pnl\nBTC-USDT,1.2\n",
        "summaries/open_positions.csv": "symbol,size\nBTC-USDT,0\n",
        "summaries/high_score_blocked_targets.csv": "symbol,score\nBTC-USDT,0.9\n",
        "summaries/high_score_blocked_outcomes.csv": (
            "symbol,matured,profitable\nBTC-USDT,true,true\n"
        ),
        "summaries/skipped_candidate_maturity_audit.csv": "symbol,matured\nETH-USDT,true\n",
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
