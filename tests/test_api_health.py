import polars as pl
import pytest
from fastapi.testclient import TestClient

import quant_lab.api.main as api_main
from quant_lab.api.main import app
from quant_lab.costs.health import build_cost_health_daily, publish_cost_health_daily


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
