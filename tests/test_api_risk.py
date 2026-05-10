from fastapi.testclient import TestClient

from quant_lab.api.main import app


def test_live_permission_api_returns_deterministic_demo_permission():
    response = TestClient(app).get(
        "/v1/risk/live-permission",
        params={"strategy": "v5", "version": "v1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "v5"
    assert payload["version"] == "v1"
    assert payload["permission"] == "ALLOW"
    assert payload["allowed_modes"] == ["paper", "live_canary"]
    assert payload["cost_model_version"] == "costs-v1"
    assert payload["gate_version"] == "default-v0.1"
    assert payload["reasons"] == ["all_required_alpha_gates_live_ready"]
    assert payload["created_at"] == "2026-05-10T00:00:00Z"
