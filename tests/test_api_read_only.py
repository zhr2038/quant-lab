from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from quant_lab.api.main import app, create_app
from quant_lab.ops.api_metrics import api_error_summary, api_metrics_summary


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
    datasets = catalog.json()["datasets"]
    assert "market_bar" in datasets
    assert "cost_bucket_daily" in datasets
    assert "risk_permission" in datasets
    assert "v5_quant_lab_mode_daily" in datasets
    assert "v5_quant_lab_enforcement_daily" in datasets
    assert "v5_candidate_event" in datasets
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


def test_local_health_allows_unauth_when_api_token_is_configured(monkeypatch):
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "test-token")
    monkeypatch.delenv("QUANT_LAB_HEALTH_ALLOW_LOCAL_UNAUTH", raising=False)
    client = TestClient(app, client=("127.0.0.1", 50000))

    missing = client.get("/v1/health")
    deep = client.get("/v1/health/deep")
    catalog = client.get("/v1/catalog/datasets")
    correct = client.get("/v1/health", headers={"Authorization": "Bearer test-token"})

    assert missing.status_code == 200
    assert deep.status_code == 200
    assert catalog.status_code == 401
    assert correct.status_code == 200


def test_health_requires_token_when_local_unauth_disabled(monkeypatch):
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "test-token")
    monkeypatch.setenv("QUANT_LAB_HEALTH_ALLOW_LOCAL_UNAUTH", "false")
    client = TestClient(app, client=("127.0.0.1", 50000))

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


def test_api_metrics_records_auth_result_and_client_context(monkeypatch, tmp_path):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "unit-token")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "1")
    client = TestClient(create_app())

    response = client.get(
        "/v1/catalog/datasets",
        headers={
            "x-quant-lab-client-id": "v5-reader",
            "user-agent": "pytest-agent",
        },
    )

    assert response.status_code == 401
    summary = api_metrics_summary(lake, since_minutes=24 * 60)
    errors = api_error_summary(lake, since_minutes=24 * 60)
    assert summary["by_auth_result"]["missing_bearer_token"] == 1
    assert summary["auth_error_count"] == 1
    row = next(item for item in errors if item["endpoint"] == "/v1/catalog/datasets")
    assert row["status_code"] == 401
    assert row["auth_result"] == "missing_bearer_token"
    assert row["client_id"] == "v5-reader"
    assert row["user_agent"] == "pytest-agent"


def test_api_error_summary_groups_endpoint_status(monkeypatch, tmp_path):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "1")
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    client = TestClient(create_app())

    assert client.get("/v1/not-found-for-metrics").status_code == 404

    rows = api_error_summary(lake, since_minutes=24 * 60)

    row = next(
        item for item in rows if item["endpoint"] == "/v1/not-found-for-metrics"
    )
    assert row["status_code"] == 404
    assert row["auth_result"] == "auth_not_configured"
    assert row["client_id"] == "__unknown__"
    assert row["error_count"] == 1
    assert row["latest_error_ts"]
    assert row["error_rate"] == 1.0
