from datetime import UTC, datetime, timedelta

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import app
from quant_lab.contracts.models import GateDecision, GateStatus
from quant_lab.data.lake import write_market_bars, write_parquet_dataset


def test_live_permission_api_aborts_without_gate_decision(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_fresh_market_bar(lake)
    _write_cost_bucket(lake)

    response = TestClient(app).get(
        "/v1/risk/live-permission",
        params={"strategy": "v5", "version": "5.0.0"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["permission"] == "ABORT"
    assert payload["reasons"] == ["no_required_gate_decisions"]


def test_live_permission_api_aborts_on_stale_market_data(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_stale_market_bar(lake)

    payload = _get_permission("v5")

    assert payload["permission"] == "ABORT"
    assert payload["reasons"] == ["market_bar_stale"]


def test_live_permission_api_sell_only_when_cost_missing(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_fresh_market_bar(lake)

    payload = _get_permission("v5")

    assert payload["permission"] == "SELL_ONLY"
    assert payload["allowed_modes"] == ["sell_only"]
    assert payload["reasons"] == ["cost_health_missing"]


def test_live_permission_api_allows_when_gate_cost_and_data_are_healthy(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)

    payload = _get_permission("v5")

    assert payload["permission"] == "ALLOW"
    assert payload["allowed_modes"] == ["paper", "live_canary"]
    assert payload["reasons"] == ["all_required_alpha_gates_live_ready"]
    assert payload["cost_model_version"].startswith("cost_bucket_daily:")
    assert payload["gate_version"] == "default-v0.1"


def test_live_permission_api_aborts_on_v5_gate_compliance_violation(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)
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
        lake / "gold/v5_gate_compliance_daily",
    )

    payload = _get_permission("v5")

    assert payload["permission"] == "ABORT"
    assert "v5_gate_compliance_violation" in payload["reasons"]


def test_live_permission_api_reads_published_risk_permission(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "SELL_ONLY",
                    "allowed_modes": '["sell_only"]',
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "costs-test",
                    "gate_version": "default-v0.1",
                    "reasons": '["published_permission"]',
                    "created_at": datetime.now(UTC).isoformat(),
                    "source": "test",
                    "fallback_level": "NONE",
                }
            ]
        ),
        lake / "gold/risk_permission",
    )

    response = TestClient(app).get(
        "/v1/risk/live-permission",
        params={"strategy": "v5", "version": "5.0.0"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["permission"] == "SELL_ONLY"
    assert payload["reasons"] == ["published_permission"]
    assert payload["permission_source"] == "published_cache"


def test_live_permission_api_recomputes_stale_published_allow(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_RISK_PERMISSION_TTL_SECONDS", "300")
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_stale_market_bar(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "ALLOW",
                    "allowed_modes": '["paper","live_canary"]',
                    "max_gross_exposure": 0.25,
                    "max_single_weight": 0.05,
                    "cost_model_version": "costs-test",
                    "gate_version": "default-v0.1",
                    "reasons": '["old_allow"]',
                    "created_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                    "source": "test",
                    "fallback_level": "NONE",
                }
            ]
        ),
        lake / "gold/risk_permission",
    )

    payload = _get_permission("v5")

    assert payload["permission"] == "ABORT"
    assert payload["permission_source"] == "recomputed"
    assert "market_bar_stale" in payload["reasons"]


def _get_permission(strategy: str) -> dict:
    response = TestClient(app).get(
        "/v1/risk/live-permission",
        params={"strategy": strategy, "version": "v1"},
    )
    assert response.status_code == 200
    return response.json()


def _write_gate(lake, status: GateStatus) -> None:
    decision = GateDecision(
        alpha_id="alpha-test",
        version="v1",
        gate_version="default-v0.1",
        status=status,
        passed=status == GateStatus.LIVE_READY,
        reasons=[],
        metrics={"ic_tstat": 3.0},
        next_action="allow canary" if status == GateStatus.LIVE_READY else "review",
        created_at=datetime.now(UTC),
    ).model_dump(mode="json")
    write_parquet_dataset(
        pl.DataFrame([decision | {"strategy": "v5"}]),
        lake / "gold/gate_decision",
    )


def _write_cost_bucket(lake) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": datetime.now(UTC).date().isoformat(),
                    "symbol": "BTC-USDT",
                    "regime": "normal",
                    "event_type": "trade",
                    "notional_bucket": "all",
                    "sample_count": 30,
                    "fee_bps_p50": 1.0,
                    "fee_bps_p75": 1.5,
                    "fee_bps_p90": 2.0,
                    "slippage_bps_p50": 1.0,
                    "slippage_bps_p75": 1.5,
                    "slippage_bps_p90": 2.0,
                    "spread_bps_p50": 0.5,
                    "spread_bps_p75": 0.75,
                    "spread_bps_p90": 1.0,
                    "total_cost_bps_p50": 2.5,
                    "total_cost_bps_p75": 3.75,
                    "total_cost_bps_p90": 5.0,
                    "fallback_level": "actual_okx_fills_and_bills",
                    "source": "actual_okx_fills_and_bills",
                }
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )


def _write_fresh_market_bar(lake) -> None:
    _write_market_bar(lake, datetime.now(UTC) - timedelta(minutes=5))


def _write_stale_market_bar(lake) -> None:
    _write_market_bar(lake, datetime.now(UTC) - timedelta(days=2))


def _write_market_bar(lake, ts: datetime) -> None:
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
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
                "volume": 12.0,
                "quote_volume": 1260.0,
                "source": "test",
                "ingest_ts": datetime.now(UTC),
            }
        ],
    )
