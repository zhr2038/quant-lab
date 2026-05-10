from fastapi.testclient import TestClient

from quant_lab.api.main import app


def test_gate_example_returns_live_ready_decision():
    response = TestClient(app).get("/v1/gates/example")

    assert response.status_code == 200
    payload = response.json()
    assert payload["alpha_id"] == "example-alpha"
    assert payload["version"] == "v1"
    assert payload["gate_version"] == "default-v0.1"
    assert payload["status"] == "LIVE_READY"
    assert payload["passed"] is True
    assert payload["reasons"] == ["all_default_gates_passed"]
    assert payload["metrics"]["ic_tstat"] == 3.1


def test_gate_decision_route_returns_deterministic_alpha_decision():
    response = TestClient(app).get("/v1/gates/decision/demo-alpha")

    assert response.status_code == 200
    payload = response.json()
    assert payload["alpha_id"] == "demo-alpha"
    assert payload["status"] == "LIVE_READY"
    assert payload["next_action"] == "eligible_for_strategy_consumer_review"
