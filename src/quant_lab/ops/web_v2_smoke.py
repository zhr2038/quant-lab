from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import httpx

DEFAULT_COST_SYMBOLS = ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT")
DEFAULT_API_CONTRACT_ENDPOINTS: tuple[dict[str, Any], ...] = (
    {
        "key": "/v1/web/bigscreen-snapshot",
        "path": "/v1/web/bigscreen-snapshot",
        "expect_no_store": True,
    },
    {"key": "/v1/catalog/datasets", "path": "/v1/catalog/datasets"},
    {
        "key": "/v1/ops/api-metrics",
        "path": "/v1/ops/api-metrics",
        "params": {"since_minutes": "60"},
    },
    {"key": "/v1/gates/example", "path": "/v1/gates/example"},
    {
        "key": "/v1/gates/decision/smoke-missing",
        "path": "/v1/gates/decision/smoke-missing",
    },
    {"key": "/v1/costs/example", "path": "/v1/costs/example"},
    {
        "key": "/v1/strategy-opportunity-advisory/v5-compact",
        "path": "/v1/strategy-opportunity-advisory/v5-compact",
        "params": {"limit": "5"},
    },
    {
        "key": "/v1/risk/live-permission",
        "path": "/v1/risk/live-permission",
        "params": {"strategy": "v5", "version": "5.0.0"},
        "risk_permission_payload": True,
    },
    {
        "key": "/v1/risk/live-permission-detail",
        "path": "/v1/risk/live-permission-detail",
        "params": {"strategy": "v5", "version": "5.0.0"},
        "risk_permission_detail_payload": True,
    },
)


def run_web_v2_smoke(
    *,
    base_url: str = "http://127.0.0.1:8027",
    api_token: str | None = None,
    symbols: tuple[str, ...] = DEFAULT_COST_SYMBOLS,
    timeout_seconds: float = 20.0,
    request_attempts: int = 8,
    retry_delay_seconds: float = 0.75,
    max_snapshot_age_seconds: int = 90,
    allow_live_cost_trust: bool = False,
    allow_live_permission: bool = False,
    include_api_contracts: bool = True,
    now: datetime | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Run read-only Web V2/API smoke checks against a live quant-lab API."""

    checked_at = now or datetime.now(UTC)
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    result: dict[str, Any] = {
        "ok": True,
        "checked_at": _iso(checked_at),
        "base_url": base_url.rstrip("/"),
        "failures": [],
        "warnings": [],
        "endpoints": {},
        "snapshot": {},
        "expert_pack": {},
        "health": {},
        "deep_health": {},
        "api_contracts": [],
        "costs": [],
    }

    with httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=timeout_seconds,
        transport=transport,
        follow_redirects=True,
    ) as client:
        web = _get(
            client,
            "/web-v2",
            attempts=request_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        result["endpoints"]["/web-v2"] = _endpoint_summary(web)
        _require_http_ok(result, "/web-v2", web)

        snapshot_response = _get(
            client,
            "/web-v2/snapshot",
            attempts=request_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        snapshot = _json_payload(snapshot_response)
        result["endpoints"]["/web-v2/snapshot"] = _endpoint_summary(snapshot_response)
        result["snapshot"] = _snapshot_summary(snapshot, snapshot_response, checked_at)
        if _require_http_ok(result, "/web-v2/snapshot", snapshot_response):
            _evaluate_snapshot(
                result,
                snapshot,
                snapshot_response,
                checked_at,
                max_snapshot_age_seconds,
            )

        pack_response = _get(
            client,
            "/web-v2/expert-pack/status",
            attempts=request_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        pack_status = _json_payload(pack_response)
        result["endpoints"]["/web-v2/expert-pack/status"] = _endpoint_summary(pack_response)
        result["expert_pack"] = _expert_pack_summary(pack_status)
        if _require_http_ok(result, "/web-v2/expert-pack/status", pack_response):
            _evaluate_expert_pack(result, pack_status)

        health_response = _get(
            client,
            "/v1/health",
            headers=headers,
            attempts=request_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        health = _json_payload(health_response)
        result["endpoints"]["/v1/health"] = _endpoint_summary(health_response)
        result["health"] = _health_summary(health)
        if _require_http_ok(result, "/v1/health", health_response):
            _evaluate_health(result, health, path="/v1/health")

        deep_response = _get(
            client,
            "/v1/health/deep",
            headers=headers,
            attempts=request_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        deep = _json_payload(deep_response)
        result["endpoints"]["/v1/health/deep"] = _endpoint_summary(deep_response)
        result["deep_health"] = _deep_health_summary(deep)
        if _require_http_ok(result, "/v1/health/deep", deep_response):
            _evaluate_health(result, deep, path="/v1/health/deep")

        if include_api_contracts:
            for spec in DEFAULT_API_CONTRACT_ENDPOINTS:
                response = _get(
                    client,
                    str(spec["path"]),
                    headers=headers,
                    params=spec.get("params"),
                    attempts=request_attempts,
                    retry_delay_seconds=retry_delay_seconds,
                )
                key = str(spec["key"])
                payload = _jsonish_payload(response)
                result["api_contracts"].append(_api_contract_summary(key, payload, response))
                result["endpoints"][key] = _endpoint_summary(response)
                if _require_http_ok(result, key, response):
                    _evaluate_api_contract(
                        result,
                        key,
                        spec,
                        payload,
                        response,
                        allow_live_permission=allow_live_permission,
                    )

        for symbol in symbols:
            params = {
                "symbol": symbol,
                "regime": "normal",
                "notional_usdt": "5",
                "quantile": "p75",
            }
            response = _get(
                client,
                "/v1/costs/estimate",
                headers=headers,
                params=params,
                attempts=request_attempts,
                retry_delay_seconds=retry_delay_seconds,
            )
            payload = _json_payload(response)
            summary = _cost_summary(symbol, payload, response)
            result["costs"].append(summary)
            key = f"/v1/costs/estimate:{symbol}"
            result["endpoints"][key] = _endpoint_summary(response)
            if _require_http_ok(result, key, response):
                _evaluate_cost(
                    result,
                    symbol,
                    payload,
                    allow_live_cost_trust=allow_live_cost_trust,
                )

    if result["failures"]:
        result["ok"] = False
    return result


def _get(
    client: httpx.Client,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    attempts: int = 1,
    retry_delay_seconds: float = 0.0,
) -> httpx.Response | Exception:
    last_error: Exception | None = None
    bounded_attempts = max(1, attempts)
    for attempt in range(bounded_attempts):
        if attempt:
            time.sleep(max(0.0, retry_delay_seconds))
        try:
            response = client.get(path, headers=headers, params=params)
        except Exception as exc:  # pragma: no cover - exercised through integration use.
            last_error = exc
            continue
        if response.status_code >= 500 and attempt + 1 < bounded_attempts:
            continue
        return response
    return last_error or RuntimeError("request_failed_without_response")


def _json_payload(response: httpx.Response | Exception) -> dict[str, Any]:
    if not isinstance(response, httpx.Response):
        return {}
    content_type = response.headers.get("content-type", "")
    if "json" not in content_type.lower():
        return {}
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _jsonish_payload(response: httpx.Response | Exception) -> Any:
    if not isinstance(response, httpx.Response):
        return None
    content_type = response.headers.get("content-type", "")
    if "json" not in content_type.lower():
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _endpoint_summary(response: httpx.Response | Exception) -> dict[str, Any]:
    if isinstance(response, Exception):
        return {"error": type(response).__name__, "message": str(response)}
    return {
        "status_code": response.status_code,
        "cache_control": response.headers.get("cache-control", ""),
        "pragma": response.headers.get("pragma", ""),
        "content_type": response.headers.get("content-type", ""),
        "bigscreen_cache_stale": response.headers.get("x-quant-lab-bigscreen-cache-stale", ""),
    }


def _require_http_ok(
    result: dict[str, Any],
    path: str,
    response: httpx.Response | Exception,
) -> bool:
    if isinstance(response, Exception):
        _failure(result, path, f"request_failed:{type(response).__name__}:{response}")
        return False
    if response.status_code >= 500:
        _failure(result, path, f"http_{response.status_code}")
        return False
    if response.status_code in {401, 403}:
        _failure(result, path, f"auth_failed:http_{response.status_code}")
        return False
    if response.status_code >= 400:
        _failure(result, path, f"http_{response.status_code}")
        return False
    return True


def _evaluate_snapshot(
    result: dict[str, Any],
    snapshot: dict[str, Any],
    response: httpx.Response | Exception,
    now: datetime,
    max_age_seconds: int,
) -> None:
    if not snapshot:
        _failure(result, "/web-v2/snapshot", "missing_json_payload")
        return
    if isinstance(response, httpx.Response):
        cache_control = response.headers.get("cache-control", "")
        pragma = response.headers.get("pragma", "")
        stale_header = response.headers.get("x-quant-lab-bigscreen-cache-stale", "")
        if "no-store" not in cache_control.lower():
            _failure(result, "/web-v2/snapshot", "missing_no_store_cache_header")
        if "no-cache" not in pragma.lower():
            _warning(result, "/web-v2/snapshot", "missing_no_cache_pragma")
        if stale_header.lower() not in {"false", "0", ""}:
            _failure(result, "/web-v2/snapshot", f"stale_cache_header:{stale_header}")

    generated_at = _parse_dt(snapshot.get("generated_at"))
    if generated_at is None:
        _failure(result, "/web-v2/snapshot", "missing_generated_at")
    else:
        age = max(0.0, (now - generated_at).total_seconds())
        result["snapshot"]["age_seconds"] = round(age, 3)
        if age > max_age_seconds:
            _failure(
                result,
                "/web-v2/snapshot",
                f"snapshot_age_seconds_gt_{max_age_seconds}:{age:.1f}",
            )

    status = str(snapshot.get("status") or "").upper()
    if status == "CRITICAL":
        _failure(result, "/web-v2/snapshot", "snapshot_status_critical")
    elif status and status != "OK":
        _warning(result, "/web-v2/snapshot", f"snapshot_status_{status.lower()}")


def _evaluate_expert_pack(result: dict[str, Any], status: dict[str, Any]) -> None:
    if not status:
        _warning(result, "/web-v2/expert-pack/status", "missing_json_payload")
        return
    state = str(status.get("state") or "").lower()
    if state in {"failed", "error"}:
        _failure(result, "/web-v2/expert-pack/status", f"state_{state}")
    elif state in {"running", "starting"}:
        _warning(result, "/web-v2/expert-pack/status", f"state_{state}")
    for key in ("latest_pack_code_lag_status", "latest_pack_v5_lag_status"):
        value = str(status.get(key) or "").upper()
        if value == "WARNING":
            _warning(result, "/web-v2/expert-pack/status", key)


def _evaluate_health(result: dict[str, Any], payload: dict[str, Any], *, path: str) -> None:
    if not payload:
        _failure(result, path, "missing_json_payload")
        return
    status = str(payload.get("status") or "").lower()
    if status in {"critical", "error", "failed"}:
        _failure(result, path, f"status_{status}")
    elif status and status not in {"ok", "warning"}:
        _warning(result, path, f"status_{status}")
    for warning in payload.get("warnings") or []:
        _warning(result, path, str(warning))


def _evaluate_cost(
    result: dict[str, Any],
    symbol: str,
    payload: dict[str, Any],
    *,
    allow_live_cost_trust: bool,
) -> None:
    path = f"/v1/costs/estimate:{symbol}"
    if not payload:
        _failure(result, path, "missing_json_payload")
        return
    if not payload.get("source"):
        _failure(result, path, "missing_cost_source")
    if not payload.get("as_of_ts"):
        _failure(result, path, "missing_as_of_ts")
    if payload.get("cost_trusted_for_live") is True and not allow_live_cost_trust:
        _failure(result, path, "unexpected_live_cost_trust")
    if payload.get("cost_trusted_for_paper") is not True:
        _warning(result, path, "not_trusted_for_paper")
    fallback = str(payload.get("fallback_reason") or "")
    if fallback.lower() == "cost_bucket_stale":
        _failure(result, path, "cost_bucket_stale")


def _evaluate_api_contract(
    result: dict[str, Any],
    key: str,
    spec: dict[str, Any],
    payload: Any,
    response: httpx.Response | Exception,
    *,
    allow_live_permission: bool,
) -> None:
    if isinstance(response, httpx.Response):
        content_type = response.headers.get("content-type", "")
        if "json" not in content_type.lower():
            _failure(result, key, "missing_json_content_type")
        if spec.get("expect_no_store"):
            cache_control = response.headers.get("cache-control", "")
            stale_header = response.headers.get("x-quant-lab-bigscreen-cache-stale", "")
            if "no-store" not in cache_control.lower():
                _failure(result, key, "missing_no_store_cache_header")
            if stale_header.lower() not in {"false", "0", ""}:
                _failure(result, key, f"stale_cache_header:{stale_header}")
    if payload is None:
        _failure(result, key, "missing_json_payload")
        return

    if key == "/v1/catalog/datasets":
        if not isinstance(payload, dict) or not payload.get("datasets"):
            _failure(result, key, "missing_datasets")
    elif key == "/v1/ops/api-metrics":
        if not isinstance(payload, dict) or payload.get("request_count") is None:
            _failure(result, key, "missing_request_count")
    elif key.startswith("/v1/gates/"):
        if not isinstance(payload, dict) or not payload.get("status"):
            _failure(result, key, "missing_gate_status")
    elif key == "/v1/costs/example":
        if not isinstance(payload, dict) or not payload.get("source"):
            _failure(result, key, "missing_cost_source")
    elif key == "/v1/strategy-opportunity-advisory/v5-compact":
        if not isinstance(payload, list):
            _failure(result, key, "expected_json_list")

    if spec.get("risk_permission_payload"):
        _evaluate_risk_permission_payload(
            result,
            key,
            payload if isinstance(payload, dict) else {},
            allow_live_permission=allow_live_permission,
        )
    if spec.get("risk_permission_detail_payload"):
        permission = payload.get("permission") if isinstance(payload, dict) else {}
        _evaluate_risk_permission_payload(
            result,
            key,
            permission if isinstance(permission, dict) else {},
            allow_live_permission=allow_live_permission,
        )


def _evaluate_risk_permission_payload(
    result: dict[str, Any],
    key: str,
    payload: dict[str, Any],
    *,
    allow_live_permission: bool,
) -> None:
    permission = str(payload.get("permission") or "").upper()
    if not permission:
        _failure(result, key, "missing_permission")
        return
    allowed_modes = payload.get("allowed_live_modes")
    if not isinstance(allowed_modes, list):
        allowed_modes = []
    max_order = _float(payload.get("max_single_order_usdt")) or 0.0
    if allow_live_permission:
        return
    if permission not in {"ABORT", "BLOCKED", "DENY"}:
        _failure(result, key, f"unexpected_live_permission:{permission}")
    if allowed_modes:
        _failure(result, key, "unexpected_allowed_live_modes")
    if max_order > 0:
        _failure(result, key, f"unexpected_live_order_notional:{max_order:g}")


def _snapshot_summary(
    snapshot: dict[str, Any],
    response: httpx.Response | Exception,
    now: datetime,
) -> dict[str, Any]:
    generated_at = _parse_dt(snapshot.get("generated_at"))
    age = None if generated_at is None else max(0.0, (now - generated_at).total_seconds())
    return {
        "status": snapshot.get("status"),
        "health_score": snapshot.get("health_score"),
        "generated_at": snapshot.get("generated_at"),
        "age_seconds": None if age is None else round(age, 3),
        "warning_count": len(snapshot.get("warnings") or []),
        "action_count": len(snapshot.get("actions") or []),
        "cache_control": response.headers.get("cache-control", "")
        if isinstance(response, httpx.Response)
        else "",
        "bigscreen_cache_stale": response.headers.get("x-quant-lab-bigscreen-cache-stale", "")
        if isinstance(response, httpx.Response)
        else "",
    }


def _expert_pack_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": payload.get("state"),
        "export_date": payload.get("export_date"),
        "latest_pack_name": payload.get("latest_pack_name"),
        "available_pack_name": payload.get("available_pack_name"),
        "pack_count": payload.get("pack_count"),
        "latest_pack_code_lag_status": payload.get("latest_pack_code_lag_status"),
        "latest_pack_v5_lag_status": payload.get("latest_pack_v5_lag_status"),
    }


def _health_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "service": payload.get("service"),
        "mode": payload.get("mode"),
    }


def _deep_health_summary(payload: dict[str, Any]) -> dict[str, Any]:
    service = (
        payload.get("service_health") if isinstance(payload.get("service_health"), dict) else {}
    )
    readiness = (
        payload.get("live_entry_readiness")
        if isinstance(payload.get("live_entry_readiness"), dict)
        else {}
    )
    cost = payload.get("cost_health") if isinstance(payload.get("cost_health"), dict) else {}
    return {
        "status": payload.get("status"),
        "warnings": payload.get("warnings") or [],
        "service_health": service,
        "readiness_status": readiness.get("status"),
        "veto_status": readiness.get("veto_status"),
        "entry_status": readiness.get("entry_status"),
        "cost_health_status": cost.get("status"),
    }


def _cost_summary(
    symbol: str,
    payload: dict[str, Any],
    response: httpx.Response | Exception,
) -> dict[str, Any]:
    return {
        "symbol": payload.get("symbol") or symbol,
        "status_code": response.status_code if isinstance(response, httpx.Response) else None,
        "source": payload.get("source"),
        "as_of_ts": payload.get("as_of_ts"),
        "fallback_reason": payload.get("fallback_reason"),
        "degraded_reason": payload.get("degraded_reason"),
        "cost_trust_level": payload.get("cost_trust_level"),
        "cost_trusted_for_paper": payload.get("cost_trusted_for_paper"),
        "cost_trusted_for_live": payload.get("cost_trusted_for_live"),
        "roundtrip_all_in_cost_bps": payload.get("roundtrip_all_in_cost_bps"),
    }


def _api_contract_summary(
    key: str,
    payload: Any,
    response: httpx.Response | Exception,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": key,
        "status_code": response.status_code if isinstance(response, httpx.Response) else None,
        "content_type": response.headers.get("content-type", "")
        if isinstance(response, httpx.Response)
        else "",
    }
    if isinstance(payload, dict):
        summary["shape"] = "object"
        for field in ("status", "permission", "request_count"):
            if field in payload:
                summary[field] = payload.get(field)
        if "datasets" in payload and isinstance(payload.get("datasets"), list):
            summary["dataset_count"] = len(payload["datasets"])
        if "permission" in payload and isinstance(payload.get("permission"), dict):
            permission = payload["permission"]
            summary["permission"] = permission.get("permission")
    elif isinstance(payload, list):
        summary["shape"] = "list"
        summary["row_count"] = len(payload)
    else:
        summary["shape"] = "unknown"
    return summary


def _failure(result: dict[str, Any], area: str, reason: str) -> None:
    result["failures"].append({"area": area, "reason": reason})


def _warning(result: dict[str, Any], area: str, reason: str) -> None:
    warning = {"area": area, "reason": reason}
    if warning not in result["warnings"]:
        result["warnings"].append(warning)


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
