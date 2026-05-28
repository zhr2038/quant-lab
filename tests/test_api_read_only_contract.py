from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from quant_lab.api.main import create_app


def test_all_strategy_facing_v1_routes_are_get_only():
    app = create_app()
    mutating_methods = {"POST", "PUT", "PATCH", "DELETE"}
    offending = []
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/v1/"):
            methods = set(route.methods or set())
            if methods & mutating_methods:
                offending.append((route.path, sorted(methods & mutating_methods)))

    assert offending == []


def test_catalog_uses_dataset_registry():
    client = TestClient(create_app())

    response = client.get("/v1/catalog/datasets")

    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert "market_bar" in datasets
    assert "v5_candidate_event" in datasets
    assert "strategy_opportunity_advisory" in datasets
