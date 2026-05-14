from __future__ import annotations

import io
import json
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.costs.calibrate import calibrate_costs_for_day
from quant_lab.costs.model import estimate_cost_from_lake
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.reports.enforce_readiness import write_enforce_readiness_report
from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle


def run_v5_contract_e2e(
    *,
    out_dir: str | Path,
    lake_root: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    root = Path(lake_root) if lake_root is not None else target / "lake"
    bundle = _write_e2e_bundle(
        target / "fixtures" / "v5_live_followup_bundle_20260514T070401Z.tar.gz"
    )

    ingest = ingest_v5_bundle(
        bundle,
        root,
        target / "restricted_archive",
        target / "redacted_archive",
        strategy="v5",
    )
    _write_orderbook(root)
    _write_gate_decision(root)
    _write_risk_permissions(root)
    calibration = calibrate_costs_for_day(root, "2026-05-14", min_sample_count=1)
    cost = estimate_cost_from_lake(
        root,
        symbol="BNB-USDT",
        regime="Trending",
        notional_usdt=1_000.0,
        quantile="p75",
    )
    readiness = write_enforce_readiness_report(root, out_dir=target / "reports")

    health = read_parquet_dataset(root / "gold/strategy_health_daily").to_dicts()[0]
    fallbacks = read_parquet_dataset(root / "silver/v5_quant_lab_fallback")
    requests = read_parquet_dataset(root / "silver/v5_quant_lab_request")
    risk = read_parquet_dataset(root / "gold/risk_permission")
    latest_risk = sorted(risk.to_dicts(), key=lambda row: str(row.get("as_of_ts")))[-1]
    summary = {
        "lake_root": str(root),
        "bundle_path": str(bundle),
        "ingest": ingest.model_dump(mode="json"),
        "request_success_count": int(health.get("request_success_count") or 0),
        "request_error_count": int(health.get("request_error_count") or 0),
        "actual_fallback_count": int(health.get("actual_fallback_count") or 0),
        "unique_request_count": int(health.get("unique_request_count") or 0),
        "unique_actual_fallback_count": int(health.get("unique_actual_fallback_count") or 0),
        "fallback_rows": fallbacks.height,
        "request_rows": requests.height,
        "cost_calibration": calibration.model_dump(mode="json"),
        "bnb_cost_estimate": cost.model_dump(mode="json"),
        "latest_risk_permission": latest_risk,
        "readiness": readiness.model_dump(mode="json"),
        "assertions": {
            "http_200_success_not_fallback": fallbacks.height == 1,
            "timeout_fallback_unique": int(health.get("actual_fallback_count") or 0) == 1,
            "bnb_cost_not_global_default": cost.cost_source != "global_default",
            "risk_latest_as_of": latest_risk.get("as_of_ts"),
            "readiness_not_ready": readiness.readiness_status != "READY",
        },
    }
    _write_reports(target, summary)
    return summary


def _write_e2e_bundle(path: Path) -> Path:
    files = {
        "raw/reports/quant_lab_requests.jsonl": (
            json.dumps(
                {
                    "event_id": "cost-ok-001",
                    "event_type": "request",
                    "ts": "2026-05-14T07:00:00Z",
                    "path": "/v1/costs/estimate",
                    "method": "GET",
                    "status_code": 200,
                    "success": True,
                    "fallback_used": False,
                    "symbol": "BNB-USDT",
                    "strategy": "v5",
                    "version": "5.0.0",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "event_id": "risk-timeout-001",
                    "event_type": "request",
                    "ts": "2026-05-14T07:01:00Z",
                    "path": "/v1/risk/live-permission",
                    "method": "GET",
                    "status_code": 0,
                    "success": False,
                    "fallback_used": True,
                    "error_type": "QuantLabTimeout",
                    "request_id": "risk-timeout-001",
                    "strategy": "v5",
                    "version": "5.0.0",
                }
            )
            + "\n"
        ),
        "raw/reports/quant_lab_usage.jsonl": (
            '{"ts":"2026-05-14T07:00:00Z","mode":"shadow",'
            '"permission_gate_enforced":false,"strategy":"v5","version":"5.0.0"}\n'
        ),
        "summaries/quant_lab_fallbacks.csv": (
            "event_id,event_type,ts,path,status_code,success,fallback_used,error_type,request_id\n"
            "risk-timeout-001,request,2026-05-14T07:01:00Z,/v1/risk/live-permission,"
            "0,false,true,QuantLabTimeout,risk-timeout-001\n"
        ),
        "summaries/quant_lab_cost_usage.csv": (
            "ts,symbol,cost_source,status_code,success,fallback_used\n"
            "2026-05-14T07:00:00Z,BNB-USDT,public_spread_proxy,200,true,false\n"
        ),
        "summaries/quant_lab_compliance.csv": (
            "ts,permission,mode,permission_gate_enforced,actual_violation,hypothetical_violation,"
            "side,intent,new_risk\n"
            "2026-05-14T07:02:00Z,ABORT,shadow,false,false,true,buy,entry,true\n"
        ),
        "raw/recent_runs/run_e2e/summary.json": (
            '{"run_id":"run_e2e","status":"complete","strategy_version":"5.0.0"}'
        ),
        "raw/recent_runs/run_e2e/decision_audit.json": (
            '{"run_id":"run_e2e","quant_lab":{"permission":"ABORT","mode":"shadow",'
            '"remote_permission_as_of_ts":"2026-05-14T07:04:01Z"}}'
        ),
        "raw/recent_runs/run_e2e/trades.csv": (
            "ts_utc,symbol,side,action,qty,price,notional_usdt,fee,fee_ccy,order_id,trade_id,"
            "strategy_id\n"
            "2026-05-14T06:10:00Z,BNB/USDT,buy,entry,0.5,620,310,-0.031,USDT,"
            "bnb-order-buy,bnb-trade-buy,v5\n"
            "2026-05-14T06:30:00Z,BNB-USDT,sell,exit,0.5,622,311,-0.0311,USDT,"
            "bnb-order-sell,bnb-trade-sell,v5\n"
        ),
        "summaries/window_summary.json": (
            '{"run_count":1,"recent_24h_decision_audit_count":1,'
            '"latest_24h_trade_count":2,"last_72h_trade_count":2,'
            '"last_72h_roundtrip_count":0,"open_position_count":0,'
            '"dust_residual_position_count":0,"high_issue_count":0,"medium_issue_count":0}'
        ),
        "raw/state/kill_switch.json": '{"enabled": false}',
        "raw/state/reconcile_status.json": '{"ok": true}',
        "raw/state/ledger_status.json": '{"ok": true}',
        "raw/state/auto_risk_eval.json": '{"level": "LOW"}',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(datetime(2026, 5, 14, 7, 4, 1, tzinfo=UTC).timestamp())
            archive.addfile(info, io.BytesIO(data))
    return path


def _write_orderbook(lake_root: Path) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "venue": "okx",
                    "symbol": "BNB-USDT",
                    "ts": datetime(2026, 5, 14, 6, 0, tzinfo=UTC),
                    "bids_json": "[[620.0, 10.0]]",
                    "asks_json": "[[620.2, 10.0]]",
                    "source": "test",
                    "ingest_ts": datetime.now(UTC),
                }
            ]
        ),
        lake_root / "silver/orderbook_snapshot",
    )


def _write_gate_decision(lake_root: Path) -> None:
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "alpha_id": "v5.core",
                    "version": "v0.1",
                    "gate_version": "gate.v0.1",
                    "status": "DEAD",
                    "passed": False,
                    "reasons": '["test_dead_gate"]',
                    "metrics": "{}",
                    "next_action": "do_not_enforce",
                    "created_at": now.isoformat(),
                }
            ]
        ),
        lake_root / "gold/gate_decision",
    )


def _write_risk_permissions(lake_root: Path) -> None:
    now = datetime.now(UTC)
    old = datetime(2026, 5, 11, 19, 52, 50, tzinfo=UTC)
    rows = [
        _risk_row(old, expires_at=old + timedelta(minutes=30)),
        _risk_row(now, expires_at=now + timedelta(minutes=30)),
    ]
    write_parquet_dataset(pl.DataFrame(rows), lake_root / "gold/risk_permission")


def _risk_row(as_of: datetime, *, expires_at: datetime) -> dict[str, Any]:
    return {
        "strategy": "v5",
        "version": "5.0.0",
        "permission": "ABORT",
        "allowed_modes": "[]",
        "max_gross_exposure": 0.0,
        "max_single_weight": 0.0,
        "max_gross_exposure_usdt": 0.0,
        "max_single_order_usdt": 0.0,
        "cost_model_version": "cost.test",
        "gate_version": "gate.test",
        "reasons": '["required_alpha_gate_dead"]',
        "reason": "required_alpha_gate_dead",
        "risk_reason_codes": '["required_alpha_gate_dead"]',
        "created_at": as_of.isoformat(),
        "as_of_ts": as_of.isoformat(),
        "source_bundle_ts": as_of.isoformat(),
        "telemetry_latest_ts": as_of.isoformat(),
        "expires_at": expires_at.isoformat(),
        "permission_status": "ACTIVE_ABORT",
        "enforceable": True,
        "contract_version": "risk_permission.v0.2",
    }


def _write_reports(out_dir: Path, summary: dict[str, Any]) -> None:
    json_path = out_dir / "e2e_contract_test_report.json"
    md_path = out_dir / "e2e_contract_test_report.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_report(summary), encoding="utf-8")


def _markdown_report(summary: dict[str, Any]) -> str:
    assertions = summary["assertions"]
    lines = [
        "# V5 Quant Lab E2E Contract Test Report",
        "",
        f"- request_success_count: {summary['request_success_count']}",
        f"- request_error_count: {summary['request_error_count']}",
        f"- actual_fallback_count: {summary['actual_fallback_count']}",
        f"- BNB cost source: {summary['bnb_cost_estimate'].get('cost_source')}",
        f"- readiness_status: {summary['readiness'].get('readiness_status')}",
        "",
        "Assertions:",
    ]
    lines.extend(f"- {name}: {value}" for name, value in assertions.items())
    lines.append("")
    return "\n".join(lines)
