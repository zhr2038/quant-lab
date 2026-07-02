from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

DEFAULT_COST_SYMBOLS = ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT")


def run_web_v2_smoke(
    *,
    base_url: str = "http://127.0.0.1:8027",
    api_token: str | None = None,
    symbols: tuple[str, ...] = DEFAULT_COST_SYMBOLS,
    timeout_seconds: float = 20.0,
    max_snapshot_age_seconds: int = 90,
    allow_live_cost_trust: bool = False,
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
        "costs": [],
    }

    with httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=timeout_seconds,
        transport=transport,
        follow_redirects=True,
    ) as client:
        web = _get(client, "/web-v2")
        result["endpoints"]["/web-v2"] = _endpoint_summary(web)
        _require_http_ok(result, "/web-v2", web)

        snapshot_response = _get(client, "/web-v2/snapshot")
        snapshot = _json_payload(snapshot_response)
        result["endpoints"]["/web-v2/snapshot"] = _endpoint_summary(snapshot_response)
        result["snapshot"] = _snapshot_summary(snapshot, snapshot_response, checked_at)
        _require_http_ok(result, "/web-v2/snapshot", snapshot_response)
        _evaluate_snapshot(
            result,
            snapshot,
            snapshot_response,
            checked_at,
            max_snapshot_age_seconds,
        )

        pack_response = _get(client, "/web-v2/expert-pack/status")
        pack_status = _json_payload(pack_response)
        result["endpoints"]["/web-v2/expert-pack/status"] = _endpoint_summary(pack_response)
        result["expert_pack"] = _expert_pack_summary(pack_status)
        _require_http_ok(result, "/web-v2/expert-pack/status", pack_response)
        _evaluate_expert_pack(result, pack_status)

        health_response = _get(client, "/v1/health", headers=headers)
        health = _json_payload(health_response)
        result["endpoints"]["/v1/health"] = _endpoint_summary(health_response)
        result["health"] = _health_summary(health)
        _require_http_ok(result, "/v1/health", health_response)
        _evaluate_health(result, health, path="/v1/health")

        deep_response = _get(client, "/v1/health/deep", headers=headers)
        deep = _json_payload(deep_response)
        result["endpoints"]["/v1/health/deep"] = _endpoint_summary(deep_response)
        result["deep_health"] = _deep_health_summary(deep)
        _require_http_ok(result, "/v1/health/deep", deep_response)
        _evaluate_health(result, deep, path="/v1/health/deep")

        for symbol in symbols:
            params = {
                "symbol": symbol,
                "regime": "normal",
                "notional_usdt": "5",
                "quantile": "p75",
            }
            response = _get(client, "/v1/costs/estimate", headers=headers, params=params)
            payload = _json_payload(response)
            summary = _cost_summary(symbol, payload, response)
            result["costs"].append(summary)
            key = f"/v1/costs/estimate:{symbol}"
            result["endpoints"][key] = _endpoint_summary(response)
            _require_http_ok(result, key, response)
            _evaluate_cost(result, symbol, payload, allow_live_cost_trust=allow_live_cost_trust)

    if result["failures"]:
        result["ok"] = False
    return result


def _get(
    client: httpx.Client,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> httpx.Response | Exception:
    try:
        return client.get(path, headers=headers, params=params)
    except Exception as exc:  # pragma: no cover - exercised through integration use.
        return exc


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
) -> None:
    if isinstance(response, Exception):
        _failure(result, path, f"request_failed:{type(response).__name__}:{response}")
        return
    if response.status_code >= 500:
        _failure(result, path, f"http_{response.status_code}")
        return
    if response.status_code in {401, 403}:
        _failure(result, path, f"auth_failed:http_{response.status_code}")
        return
    if response.status_code >= 400:
        _failure(result, path, f"http_{response.status_code}")


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


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
