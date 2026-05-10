"""Read-only quant-lab HTTP client for V5/V7 strategy consumers.

This client only issues GET requests. It must not mutate strategy state,
exchange state, orders, balances, or account configuration.
"""

from typing import Any

import httpx
from pydantic import ValidationError

from quant_lab.contracts.models import CostEstimate, GateDecision, RiskPermission


class QuantLabError(RuntimeError):
    pass


class QuantLabUnavailable(QuantLabError):
    pass


class QuantLabValidationError(QuantLabError):
    pass


class QuantLabPermissionError(QuantLabError):
    pass


class QuantLabClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 2.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client = http_client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
        )

    def get_health(self) -> dict[str, Any]:
        payload = self._get("/v1/health")
        if not isinstance(payload, dict):
            raise QuantLabValidationError("quant-lab health response is not a JSON object")
        return payload

    def estimate_cost(
        self,
        symbol: str,
        regime: str,
        notional_usdt: float,
        quantile: str = "p75",
    ) -> CostEstimate:
        payload = self._get(
            "/v1/costs/estimate",
            params={
                "symbol": symbol,
                "regime": regime,
                "notional_usdt": notional_usdt,
                "quantile": quantile,
            },
        )
        return self._validate_model(CostEstimate, payload, "cost estimate")

    def get_gate_decision(self, alpha_id: str) -> GateDecision:
        payload = self._get(f"/v1/gates/decision/{alpha_id}")
        return self._validate_model(GateDecision, payload, "gate decision")

    def get_live_permission(self, strategy: str, version: str) -> RiskPermission:
        payload = self._get(
            "/v1/risk/live-permission",
            params={"strategy": strategy, "version": version},
        )
        return self._validate_model(RiskPermission, payload, "live permission")

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise QuantLabUnavailable(f"quant-lab request timed out: {path}") from exc
        except httpx.TransportError as exc:
            raise QuantLabUnavailable(f"quant-lab request failed: {path}") from exc

        if response.status_code in {401, 403}:
            raise QuantLabPermissionError(
                f"quant-lab rejected read-only request: {response.status_code}"
            )
        if response.status_code in {400, 404, 409, 422}:
            raise QuantLabValidationError(
                f"quant-lab validation error: {response.status_code} {response.text}"
            )
        if response.status_code >= 500:
            raise QuantLabUnavailable(
                f"quant-lab unavailable: {response.status_code} {response.text}"
            )
        if response.status_code >= 300:
            raise QuantLabUnavailable(
                f"quant-lab unexpected response: {response.status_code} {response.text}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise QuantLabValidationError("quant-lab response is not valid JSON") from exc

    def _validate_model(self, model_type, payload: Any, label: str):
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise QuantLabValidationError(f"invalid quant-lab {label} response") from exc
