from fastapi.testclient import TestClient

import quant_lab.api.main as api_main
from quant_lab.api.main import app


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
