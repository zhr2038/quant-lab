from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from quant_lab.api.main import app


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
        "v5_quant_lab_mode_daily",
        "v5_quant_lab_enforcement_daily",
    ]
    assert gate.status_code == 200
    assert gate.json()["status"] == "LIVE_READY"
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


def test_health_requires_token_when_local_bypass_disabled(monkeypatch):
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "test-token")
    monkeypatch.setenv("QUANT_LAB_HEALTH_ALLOW_LOCAL_UNAUTH", "false")
    client = TestClient(app)

    missing = client.get("/v1/health")
    correct = client.get("/v1/health", headers={"Authorization": "Bearer test-token"})

    assert missing.status_code == 401
    assert correct.status_code == 200
