from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.reports.enforce_readiness import (
    build_enforce_readiness_report,
    write_enforce_readiness_report,
)


def test_enforce_readiness_blocks_when_cost_global_default_high(tmp_path):
    lake = tmp_path / "lake"
    _write_common_ready_inputs(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="GLOBAL",
                    source="global_default",
                    total=25.0,
                    fallback_level="GLOBAL_DEFAULT",
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    report = build_enforce_readiness_report(lake)

    assert report.readiness_status == "BLOCKED"
    assert "cost_api_global_default_rate" in report.blocked_reasons
    assert report.shadow_only_recommended is True


def test_enforce_readiness_blocks_stale_risk_permission(tmp_path):
    lake = tmp_path / "lake"
    _write_common_ready_inputs(lake)
    _write_cost_rows(lake, source="mixed_actual_proxy")
    _write_risk_permission(lake, status="STALE_ABORT", enforceable=False)

    report = build_enforce_readiness_report(lake)

    assert report.readiness_status == "BLOCKED"
    assert "risk_permission_fresh" in report.blocked_reasons


def test_enforce_readiness_warns_on_high_telemetry_duplicate_rate(tmp_path):
    lake = tmp_path / "lake"
    _write_common_ready_inputs(lake, duplicate_rate=0.2)
    _write_cost_rows(lake, source="mixed_actual_proxy")

    report = build_enforce_readiness_report(lake)

    assert report.readiness_status == "WARN"
    assert "telemetry_duplicate_rate" in report.warning_reasons


def test_enforce_readiness_ready_when_all_checks_pass(tmp_path):
    lake = tmp_path / "lake"
    _write_common_ready_inputs(lake)
    _write_cost_rows(lake, source="mixed_actual_proxy")

    report = build_enforce_readiness_report(lake)

    assert report.readiness_status == "READY"
    assert report.shadow_only_recommended is False


def test_write_enforce_readiness_report_outputs_json_and_csv(tmp_path):
    lake = tmp_path / "lake"
    out = tmp_path / "reports"
    _write_common_ready_inputs(lake)
    _write_cost_rows(lake, source="mixed_actual_proxy")

    report = write_enforce_readiness_report(lake, out_dir=out)

    assert report.readiness_status == "READY"
    payload = json.loads((out / "v5_enforce_readiness.json").read_text())
    assert payload["readiness_status"] == "READY"
    csv_text = (out / "v5_enforce_readiness.csv").read_text()
    assert "readiness_status" in csv_text


def _write_common_ready_inputs(lake, *, duplicate_rate: float = 0.0) -> None:
    now = datetime.now(UTC)
    _write_risk_permission(lake, status="ACTIVE_ABORT", enforceable=True)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now.isoformat(),
                    "decision_audit_count_24h": 1,
                    "duplicate_rate": duplicate_rate,
                    "fallback_rate": 0.0,
                }
            ]
        ),
        lake / "gold/strategy_health_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": "raw/recent_runs/run_001/decision_audit.json",
                    "raw_payload_json": "{}",
                    "ingest_ts": now.isoformat(),
                }
            ]
        ),
        lake / "silver/v5_decision_audit",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "alpha_id": "v5.core",
                    "version": "v0.1",
                    "gate_version": "gate.v0.1",
                    "status": "PAPER_READY",
                    "passed": False,
                    "reasons": "[]",
                    "metrics": "{}",
                    "next_action": "paper",
                    "created_at": now.isoformat(),
                }
            ]
        ),
        lake / "gold/gate_decision",
    )


def _write_risk_permission(lake, *, status: str, enforceable: bool) -> None:
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "ABORT",
                    "allowed_modes": "[]",
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "max_gross_exposure_usdt": 0.0,
                    "max_single_order_usdt": 0.0,
                    "cost_model_version": "cost.v0",
                    "gate_version": "gate.v0",
                    "reasons": '["test"]',
                    "reason": "test",
                    "risk_reason_codes": '["test"]',
                    "created_at": now.isoformat(),
                    "as_of_ts": now.isoformat(),
                    "source_bundle_ts": now.isoformat(),
                    "telemetry_latest_ts": now.isoformat(),
                    "expires_at": (now + timedelta(hours=1)).isoformat()
                    if status.startswith("ACTIVE_")
                    else (now - timedelta(hours=1)).isoformat(),
                    "permission_status": status,
                    "enforceable": enforceable,
                    "contract_version": "risk_permission.v0.2",
                }
            ]
        ),
        lake / "gold/risk_permission",
    )


def _write_cost_rows(lake, *, source: str) -> None:
    write_parquet_dataset(
        pl.DataFrame([_cost_row(symbol=symbol, source=source, total=2.0) for symbol in _symbols()]),
        lake / "gold/cost_bucket_daily",
    )


def _cost_row(
    *,
    symbol: str,
    source: str,
    total: float,
    fallback_level: str = "NONE",
) -> dict:
    now = datetime.now(UTC)
    return {
        "day": now.date().isoformat(),
        "symbol": symbol,
        "regime": "realized" if source != "global_default" else "global_default",
        "event_type": "actual_fill" if source != "global_default" else "default",
        "notional_bucket": "all",
        "sample_count": 4 if source != "global_default" else 0,
        "fee_bps_p50": total,
        "fee_bps_p75": total,
        "fee_bps_p90": total,
        "slippage_bps_p50": 0.0,
        "slippage_bps_p75": 0.0,
        "slippage_bps_p90": 0.0,
        "spread_bps_p50": 0.0,
        "spread_bps_p75": 0.0,
        "spread_bps_p90": 0.0,
        "total_cost_bps_p50": total,
        "total_cost_bps_p75": total,
        "total_cost_bps_p90": total,
        "fallback_level": fallback_level,
        "source": source,
        "cost_source": source,
        "actual_fill_count": 4 if source in {"actual_fills", "mixed_actual_proxy"} else 0,
        "mixed_fill_count": 4 if source == "mixed_actual_proxy" else 0,
        "proxy_sample_count": 0,
        "cost_model_version": "cost_bucket_daily:test",
        "created_at": now.isoformat(),
    }


def _symbols() -> list[str]:
    return ["BNB-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT"]
