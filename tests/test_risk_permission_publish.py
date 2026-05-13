from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.contracts.models import GateDecision, GateStatus
from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.risk.publish import publish_risk_permission


def test_publish_risk_permission_sell_only_when_cost_missing(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_fresh_market_bar(lake)

    result = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert result.permission == "SELL_ONLY"
    assert result.reasons == ["cost_health_missing"]


def test_publish_risk_permission_allows_paper_for_paper_ready_gate(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.PAPER_READY)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)

    publish_risk_permission(lake, strategy="v5", version="5.0.0")
    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]

    assert permission["permission"] == "ALLOW"
    assert permission["allowed_modes"] == '["paper"]'


def test_publish_risk_permission_allows_live_ready_with_healthy_inputs(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)

    result = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert result.permission == "ALLOW"
    assert result.reasons == ["all_required_alpha_gates_live_ready"]


def test_publish_risk_permission_ignores_bootstrap_gate_when_research_gate_exists(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.LIVE_READY)
    _append_bootstrap_gate(lake)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)

    result = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert result.permission == "ALLOW"
    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    assert permission["gate_version"] == "default-v0.1"
    assert "bootstrap" not in permission["gate_version"]


def test_publish_risk_permission_aborts_on_gate_compliance_violation(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": datetime.now(UTC).date().isoformat(),
                    "status": "CRITICAL",
                    "violation_count": 1,
                }
            ]
        ),
        lake / "gold" / "v5_gate_compliance_daily",
    )

    result = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert result.permission == "ABORT"
    assert "v5_gate_compliance_violation" in result.reasons


def test_publish_risk_permission_writes_idempotently(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.QUARANTINE)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)

    first = publish_risk_permission(lake, strategy="v5", version="5.0.0")
    second = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert first.risk_permission_rows == second.risk_permission_rows == 1
    assert read_parquet_dataset(lake / "gold" / "risk_permission").height == 1


def test_publish_risk_permission_marks_active_when_telemetry_within_threshold(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.QUARANTINE)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)
    telemetry_ts = datetime.now(UTC) - timedelta(minutes=5)
    _write_strategy_health(lake, telemetry_ts)

    publish_risk_permission(lake, strategy="v5", version="5.0.0")

    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    assert permission["permission"] == "SELL_ONLY"
    assert permission["permission_status"] == "ACTIVE_SELL_ONLY"
    assert permission["telemetry_latest_ts"].startswith(telemetry_ts.isoformat()[:16])
    assert permission["contract_version"] == "risk_permission.v0.2"


def test_publish_risk_permission_marks_stale_when_telemetry_is_newer_by_48h(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.DEAD)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)
    _write_strategy_health(lake, datetime.now(UTC) + timedelta(hours=48))

    publish_risk_permission(lake, strategy="v5", version="5.0.0")

    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    assert permission["permission"] == "ABORT"
    assert permission["permission_status"] == "STALE_ABORT"
    assert permission["permission_freshness_sec"] >= 47 * 60 * 60


def _write_gate(lake, status: GateStatus) -> None:
    decision = GateDecision(
        alpha_id="v5.core.momentum",
        version="v0.1",
        gate_version="default-v0.1",
        status=status,
        passed=status == GateStatus.LIVE_READY,
        reasons=[],
        metrics={"ic_tstat": 3.0},
        next_action="review",
        created_at=datetime.now(UTC),
    ).model_dump(mode="json")
    write_parquet_dataset(
        pl.DataFrame([decision | {"strategy": "v5"}]),
        lake / "gold" / "gate_decision",
    )


def _append_bootstrap_gate(lake) -> None:
    existing_rows = []
    for row in read_parquet_dataset(lake / "gold" / "gate_decision").to_dicts():
        normalized = dict(row)
        if isinstance(normalized.get("reasons"), list):
            normalized["reasons"] = "[]"
        if isinstance(normalized.get("metrics"), dict):
            normalized["metrics"] = '{"ic_tstat":3.0}'
        existing_rows.append(normalized)
    bootstrap = {
        "strategy": "v5",
        "alpha_id": "v5.core",
        "version": "bootstrap",
        "gate_version": "bootstrap.quarantine.v1",
        "status": "QUARANTINE",
        "passed": False,
        "reasons": '["bootstrap_gate"]',
        "metrics": '{"bootstrap":true}',
        "next_action": "generate_alpha_evidence_before_live",
        "created_at": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        "source": "bootstrap_gold_health",
        "fallback_level": "BOOTSTRAP_CONSERVATIVE",
    }
    write_parquet_dataset(
        pl.DataFrame([*existing_rows, bootstrap]),
        lake / "gold" / "gate_decision",
    )


def _write_fresh_market_bar(lake) -> None:
    ts = datetime.now(UTC) - timedelta(minutes=5)
    write_market_bars(
        lake,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": ts,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1.0,
                "quote_volume": 100.5,
                "source": "test",
                "ingest_ts": datetime.now(UTC),
            }
        ],
    )


def _write_actual_cost(lake) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": datetime.now(UTC).date().isoformat(),
                    "symbol": "BTC-USDT",
                    "regime": "realized",
                    "event_type": "actual_fill",
                    "notional_bucket": "all",
                    "sample_count": 30,
                    "fee_bps_p50": 1.0,
                    "fee_bps_p75": 1.0,
                    "fee_bps_p90": 1.0,
                    "slippage_bps_p50": 0.0,
                    "slippage_bps_p75": 0.0,
                    "slippage_bps_p90": 0.0,
                    "spread_bps_p50": 1.0,
                    "spread_bps_p75": 1.0,
                    "spread_bps_p90": 1.0,
                    "total_cost_bps_p50": 2.0,
                    "total_cost_bps_p75": 2.0,
                    "total_cost_bps_p90": 2.0,
                    "fallback_level": "NONE",
                    "source": "actual_okx_fills_and_bills",
                    "cost_model_version": "costs-test",
                    "created_at": datetime.now(UTC).isoformat(),
                }
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )


def _write_strategy_health(lake, latest_bundle_ts: datetime) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": latest_bundle_ts.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": latest_bundle_ts,
                }
            ]
        ),
        lake / "gold" / "strategy_health_daily",
    )
