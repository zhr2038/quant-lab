from datetime import UTC, datetime

import polars as pl
import pytest
from fastapi.testclient import TestClient

import quant_lab.api.main as api_main
from quant_lab.api.main import app
from quant_lab.costs.health import build_cost_health_daily, publish_cost_health_daily
from quant_lab.data.lake import write_parquet_dataset


def test_health_is_light_and_does_not_scan_lake(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)

    def fail_lake_health(*_args, **_kwargs):
        raise AssertionError("light health must not scan lake datasets")

    monkeypatch.setattr(api_main, "_lake_data_health", fail_lake_health)
    monkeypatch.setattr(api_main, "_lake_cost_health", fail_lake_health)

    response = TestClient(app).get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "quant-lab", "mode": "read-only"}


def test_health_deep_runs_dataset_checks(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(
        api_main,
        "_lake_data_health",
        lambda *_args, **_kwargs: calls.append("data") or {"status": "ok"},
    )
    monkeypatch.setattr(
        api_main,
        "_lake_cost_health",
        lambda *_args, **_kwargs: calls.append("cost") or {"status": "ok"},
    )
    monkeypatch.setattr(
        api_main,
        "_risk_permission_dependency_meta_health",
        lambda *_args, **_kwargs: {"status": "ok", "warning": ""},
    )

    response = TestClient(app).get("/v1/health/deep")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["data_health"] == {"status": "ok"}
    assert response.json()["cost_health"] == {"status": "ok"}
    assert response.json()["warnings"] == []
    assert calls == ["data", "cost"]


def test_health_deep_status_reflects_nested_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    monkeypatch.setattr(api_main, "_lake_data_health", lambda *_args: {"status": "ok"})
    monkeypatch.setattr(
        api_main,
        "_lake_cost_health",
        lambda *_args: {"status": "warning", "high_fallback": True},
    )
    monkeypatch.setattr(
        api_main,
        "_risk_permission_dependency_meta_health",
        lambda *_args: {"status": "ok", "warning": ""},
    )

    response = TestClient(app).get("/v1/health/deep")

    assert response.status_code == 200
    assert response.json()["status"] == "warning"
    assert response.json()["warnings"] == ["cost_health_warning"]


def test_lake_cost_health_exposes_fallback_breakdown(tmp_path):
    lake = tmp_path / "lake"
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-06-15",
                    "symbol": "BNB-USDT",
                    "source": "actual_fills",
                    "sample_count": 30,
                    "fallback_level": "NONE",
                    "cost_model_version": "costs-v1",
                },
                {
                    "day": "2026-06-15",
                    "symbol": "SOL-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 100,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "cost_model_version": "costs-v1",
                },
                {
                    "day": "2026-06-15",
                    "symbol": "XRP-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 100,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "cost_model_version": "costs-v1",
                },
            ]
        ),
        day="2026-06-15",
        min_sample_count=30,
        expected_symbols=["BNB-USDT", "SOL-USDT", "XRP-USDT", "DOGE-USDT"],
    )
    publish_cost_health_daily(lake, row)
    day = datetime.now(UTC).date().isoformat()
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_bucket_row("BNB-USDT", "actual_fills", day),
                _cost_bucket_row("BTC-USDT", "actual_fills", day),
                _cost_bucket_row("ETH-USDT", "public_spread_proxy", day),
                _cost_bucket_row("SOL-USDT", "public_spread_proxy", day),
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )

    payload = api_main._lake_cost_health(lake)

    assert payload["status"] == "warning"
    assert payload["high_fallback"] is True
    assert payload["fallback_ratio"] == pytest.approx(2 / 3)
    assert payload["hard_fallback_ratio"] == 0
    assert payload["soft_fallback_ratio"] == pytest.approx(2 / 3)
    assert payload["actual_rows"] == 1
    assert payload["mixed_rows"] == 0
    assert payload["proxy_rows"] == 2
    assert payload["global_default_rows"] == 0
    assert payload["proxy_only_count"] == 2
    assert payload["symbols_missing_cost"] == ["DOGE-USDT"]
    assert "soft_fallback_ratio_gt_0.5" in payload["warnings"]
    coverage = payload["live_universe_cost_coverage"]
    assert coverage["coverage_status"] == "PASS"
    assert coverage["coverage_rate"] == pytest.approx(0.5)
    assert coverage["direct_symbols"] == ["BNB-USDT", "BTC-USDT"]
    assert coverage["mixed_proxy_symbols"] == ["ETH-USDT", "SOL-USDT"]


def test_lake_cost_health_warns_when_live_universe_coverage_is_low(tmp_path):
    lake = tmp_path / "lake"
    day = datetime.now(UTC).date().isoformat()
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                _cost_bucket_row("BTC-USDT", "actual_fills", day),
            ]
        ),
        day=day,
        min_sample_count=30,
        expected_symbols=["BTC-USDT"],
    )
    publish_cost_health_daily(lake, row)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_bucket_row("BTC-USDT", "actual_fills", day),
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )

    payload = api_main._lake_cost_health(lake)

    assert payload["status"] == "warning"
    assert "live_universe_cost_coverage_warning" in payload["warnings"]
    coverage = payload["live_universe_cost_coverage"]
    assert coverage["coverage_status"] == "WARNING"
    assert coverage["coverage_rate"] == pytest.approx(0.25)
    assert coverage["covered_symbols"] == ["BTC-USDT"]
    assert coverage["missing_symbols"] == ["BNB-USDT", "ETH-USDT", "SOL-USDT"]


def test_health_deep_status_reflects_nested_critical(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    monkeypatch.setattr(
        api_main,
        "_lake_data_health",
        lambda *_args: {"status": "critical", "is_critical": True},
    )
    monkeypatch.setattr(api_main, "_lake_cost_health", lambda *_args: {"status": "ok"})
    monkeypatch.setattr(
        api_main,
        "_risk_permission_dependency_meta_health",
        lambda *_args: {"status": "warning", "warning": "risk_permission_dependency_meta_missing"},
    )

    response = TestClient(app).get("/v1/health/deep")

    assert response.status_code == 200
    assert response.json()["status"] == "critical"
    assert response.json()["warnings"] == [
        "data_health_critical",
        "risk_permission_dependency_meta_warning",
    ]


def _cost_bucket_row(symbol: str, source: str, day: str) -> dict[str, object]:
    return {
        "day": day,
        "symbol": symbol,
        "regime": "normal",
        "event_type": "trade",
        "notional_bucket": "all",
        "sample_count": 30,
        "actual_fill_count": 30 if source == "actual_fills" else 0,
        "mixed_fill_count": 0,
        "proxy_sample_count": 30 if source == "public_spread_proxy" else 0,
        "fee_bps_p75": 1.5,
        "spread_bps_p75": 0.75,
        "slippage_bps_p75": 1.5,
        "total_cost_bps_p75": 3.75,
        "fallback_level": "PUBLIC_SPREAD_PROXY" if source == "public_spread_proxy" else "NONE",
        "source": source,
    }
