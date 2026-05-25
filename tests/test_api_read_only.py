from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from quant_lab.api.main import app, create_app


def test_api_has_no_non_get_strategy_routes():
    business_methods = {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
    offending_routes = []

    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/v1/"):
            explicit_business_methods = sorted((route.methods or set()) & business_methods)
            if explicit_business_methods != ["GET"]:
                offending_routes.append((route.path, explicit_business_methods))

    assert offending_routes == []


def test_read_only_example_endpoints_and_catalog():
    client = TestClient(app)

    health = client.get("/v1/health")
    catalog = client.get("/v1/catalog/datasets")
    gate = client.get("/v1/gates/example")
    costs = client.get("/v1/costs/example")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "quant-lab",
        "mode": "read-only",
    }
    assert catalog.status_code == 200
    assert catalog.json()["datasets"] == [
        "market_bar",
        "feature_value",
        "feature_coverage_daily",
        "feature_anomaly_daily",
        "cost_bucket_daily",
        "cost_health_daily",
        "alpha_evidence",
        "gate_decision",
        "risk_permission",
        "api_request_metrics",
        "job_run_history",
        "lake_file_health_daily",
        "v5_quant_lab_mode_daily",
        "v5_quant_lab_enforcement_daily",
    ]
    assert gate.status_code == 200
    assert gate.json()["status"] == "QUARANTINE"
    assert gate.json()["passed"] is False
    assert costs.status_code == 200
    assert costs.json()["fallback_level"] == "NONE"


def test_bearer_token_required_when_configured(monkeypatch):
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "test-token")
    client = TestClient(app)

    missing = client.get("/v1/catalog/datasets")
    wrong = client.get("/v1/catalog/datasets", headers={"Authorization": "Bearer wrong"})
    correct = client.get(
        "/v1/catalog/datasets",
        headers={"Authorization": "Bearer test-token"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert correct.status_code == 200


def test_health_requires_token_when_api_token_is_configured(monkeypatch):
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "test-token")
    client = TestClient(app)

    missing = client.get("/v1/health")
    correct = client.get("/v1/health", headers={"Authorization": "Bearer test-token"})

    assert missing.status_code == 401
    assert correct.status_code == 200


def test_allowed_client_ips_blocks_unlisted_client(monkeypatch):
    monkeypatch.setenv("QUANT_LAB_ALLOWED_CLIENT_IPS", "127.0.0.1")
    client = TestClient(app)

    response = client.get("/v1/catalog/datasets")

    assert response.status_code == 403


def test_docs_can_be_disabled(monkeypatch):
    monkeypatch.setenv("QUANT_LAB_DISABLE_DOCS", "true")
    client = TestClient(create_app())

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_api_metrics_records_request_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    client = TestClient(create_app())

    assert client.get("/v1/health").status_code == 200
    response = client.get("/v1/ops/api-metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_count"] >= 1
    assert payload["by_path"]["/v1/health"] == 1
    assert payload["latency_ms"]["max"] is not None
