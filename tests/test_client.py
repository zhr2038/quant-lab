from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from quant_lab.client import (
    QuantLabClient,
    QuantLabPermissionError,
    QuantLabUnavailable,
)
from quant_lab.contracts.models import RiskAction

BASE_URL = "http://quant-lab.test"


def test_client_only_uses_get_requests():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_payload_for_path(request.url.path))

    client = _client(handler)

    assert client.get_health()["status"] == "ok"
    assert client.estimate_cost("BTC-USDT", "normal", 10_000).cost_bps == 4.2
    assert client.get_gate_decision("alpha-1").status == "LIVE_READY"
    assert client.get_live_permission("v5", "v1").permission == RiskAction.ALLOW

    assert [request.method for request in calls] == ["GET", "GET", "GET", "GET"]
    assert [request.url.path for request in calls] == [
        "/v1/health",
        "/v1/costs/estimate",
        "/v1/gates/decision/alpha-1",
        "/v1/risk/live-permission",
    ]
    assert calls[1].url.params["quantile"] == "p75"


def test_timeout_path_raises_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = _client(handler)

    with pytest.raises(QuantLabUnavailable, match="timed out"):
        client.get_health()


def test_unavailable_http_status_raises_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "not ready"})

    client = _client(handler)

    with pytest.raises(QuantLabUnavailable, match="503"):
        client.get_health()


def test_permission_http_status_raises_permission_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "forbidden"})

    client = _client(handler)

    with pytest.raises(QuantLabPermissionError, match="403"):
        client.get_live_permission("v5", "v1")


def test_successful_cost_estimate_parse():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/costs/estimate"
        assert request.url.params["symbol"] == "BTC-USDT"
        assert request.url.params["regime"] == "normal"
        assert request.url.params["notional_usdt"] == "10000"
        assert request.url.params["quantile"] == "p90"
        return httpx.Response(200, json=_cost_payload())

    cost = _client(handler).estimate_cost(
        symbol="BTC-USDT",
        regime="normal",
        notional_usdt=10_000,
        quantile="p90",
    )

    assert cost.symbol == "BTC-USDT"
    assert cost.cost_bps == 4.2
    assert cost.fallback_level == "NONE"


def test_successful_live_permission_parse():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/risk/live-permission"
        assert request.url.params["strategy"] == "v5"
        assert request.url.params["version"] == "v1"
        return httpx.Response(200, json=_permission_payload())

    permission = _client(handler).get_live_permission("v5", "v1")

    assert permission.strategy == "v5"
    assert permission.permission == RiskAction.ALLOW
    assert permission.allowed_modes == ["paper", "live_canary"]
    assert permission.created_at == datetime(2026, 5, 10, tzinfo=UTC)


def _client(handler) -> QuantLabClient:
    http_client = httpx.Client(
        base_url=BASE_URL,
        timeout=0.1,
        transport=httpx.MockTransport(handler),
    )
    return QuantLabClient(base_url=BASE_URL, timeout_seconds=0.1, http_client=http_client)


def _payload_for_path(path: str) -> dict[str, Any]:
    payloads = {
        "/v1/health": {"status": "ok", "service": "quant-lab", "mode": "read-only"},
        "/v1/costs/estimate": _cost_payload(),
        "/v1/gates/decision/alpha-1": _gate_payload(),
        "/v1/risk/live-permission": _permission_payload(),
    }
    return payloads[path]


def _cost_payload() -> dict[str, Any]:
    return {
        "symbol": "BTC-USDT",
        "regime": "normal",
        "notional_usdt": 10_000.0,
        "cost_bps": 4.2,
        "fallback_level": "NONE",
        "bucket_id": "btc-default",
    }


def _gate_payload() -> dict[str, Any]:
    return {
        "alpha_id": "alpha-1",
        "version": "v1",
        "gate_version": "default-v0.1",
        "status": "LIVE_READY",
        "passed": True,
        "reasons": ["all_default_gates_passed"],
        "metrics": {"ic_tstat": 3.1},
        "next_action": "eligible_for_strategy_consumer_review",
        "created_at": "2026-05-10T00:00:00Z",
    }


def _permission_payload() -> dict[str, Any]:
    return {
        "strategy": "v5",
        "version": "v1",
        "permission": "ALLOW",
        "allowed_modes": ["paper", "live_canary"],
        "max_gross_exposure": 0.25,
        "max_single_weight": 0.05,
        "cost_model_version": "costs-v1",
        "gate_version": "default-v0.1",
        "reasons": ["all_required_alpha_gates_live_ready"],
        "created_at": "2026-05-10T00:00:00Z",
    }
