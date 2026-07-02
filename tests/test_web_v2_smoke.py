from datetime import UTC, datetime, timedelta

import httpx
from typer.testing import CliRunner

import quant_lab.cli as cli_module
from quant_lab.cli import app
from quant_lab.ops.web_v2_smoke import run_web_v2_smoke

NOW = datetime(2026, 7, 2, 7, 40, tzinfo=UTC)
runner = CliRunner()


def test_web_v2_smoke_allows_read_only_warnings_without_failing():
    transport = httpx.MockTransport(_smoke_handler(snapshot_status="WARNING"))

    result = run_web_v2_smoke(
        base_url="http://testserver",
        api_token="token",
        symbols=("BTC-USDT",),
        now=NOW,
        transport=transport,
    )

    assert result["ok"] is True
    assert result["failures"] == []
    assert {"area": "/web-v2/snapshot", "reason": "snapshot_status_warning"} in result[
        "warnings"
    ]
    assert {
        "area": "/web-v2/expert-pack/status",
        "reason": "latest_pack_v5_lag_status",
    } in result["warnings"]
    assert result["snapshot"]["bigscreen_cache_stale"] == "false"
    assert result["costs"] == [
        {
            "symbol": "BTC-USDT",
            "status_code": 200,
            "source": "public_spread_proxy",
            "as_of_ts": "2026-07-02T07:13:44Z",
            "fallback_reason": "no_matching_regime",
            "degraded_reason": "none",
            "cost_trust_level": "PAPER_ONLY",
            "cost_trusted_for_paper": True,
            "cost_trusted_for_live": False,
            "roundtrip_all_in_cost_bps": 28.0,
        }
    ]
    contract_paths = {row["path"] for row in result["api_contracts"]}
    assert "/v1/catalog/datasets" in contract_paths
    assert "/v1/risk/live-permission" in contract_paths


def test_web_v2_smoke_fails_stale_snapshot_and_stale_cost():
    transport = httpx.MockTransport(
        _smoke_handler(
            generated_at=(NOW - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
            cost_fallback_reason="cost_bucket_stale",
            cost_trusted_for_live=True,
        )
    )

    result = run_web_v2_smoke(
        base_url="http://testserver",
        api_token="token",
        symbols=("BTC-USDT",),
        now=NOW,
        max_snapshot_age_seconds=90,
        transport=transport,
    )

    assert result["ok"] is False
    assert {
        "area": "/web-v2/snapshot",
        "reason": "snapshot_age_seconds_gt_90:600.0",
    } in result["failures"]
    assert {"area": "/v1/costs/estimate:BTC-USDT", "reason": "cost_bucket_stale"} in result[
        "failures"
    ]
    assert {
        "area": "/v1/costs/estimate:BTC-USDT",
        "reason": "unexpected_live_cost_trust",
    } in result["failures"]


def test_web_v2_smoke_fails_unexpected_live_permission():
    transport = httpx.MockTransport(_smoke_handler(live_permission="ALLOW"))

    result = run_web_v2_smoke(
        base_url="http://testserver",
        api_token="token",
        symbols=("BTC-USDT",),
        now=NOW,
        transport=transport,
    )

    assert result["ok"] is False
    assert {
        "area": "/v1/risk/live-permission",
        "reason": "unexpected_live_permission:ALLOW",
    } in result["failures"]
    assert {
        "area": "/v1/risk/live-permission",
        "reason": "unexpected_allowed_live_modes",
    } in result["failures"]
    assert {
        "area": "/v1/risk/live-permission-detail",
        "reason": "unexpected_live_permission:ALLOW",
    } in result["failures"]


def test_web_v2_smoke_cli_writes_output_before_nonzero_exit(monkeypatch, tmp_path):
    output_path = tmp_path / "ops" / "web_v2_smoke" / "latest.json"

    def fake_run_web_v2_smoke(**_: object) -> dict[str, object]:
        return {
            "ok": False,
            "checked_at": "2026-07-02T07:40:00Z",
            "failures": [{"area": "/web-v2/snapshot", "reason": "snapshot_status_critical"}],
            "warnings": [],
        }

    monkeypatch.setattr(cli_module, "run_web_v2_smoke", fake_run_web_v2_smoke)

    result = runner.invoke(app, ["web-v2-smoke", "--output-json", str(output_path)])

    assert result.exit_code == 1
    payload = output_path.read_text(encoding="utf-8")
    assert '"snapshot_status_critical"' in payload


def _smoke_handler(
    *,
    snapshot_status: str = "OK",
    generated_at: str = "2026-07-02T07:39:50Z",
    cost_fallback_reason: str = "no_matching_regime",
    cost_trusted_for_live: bool = False,
    live_permission: str = "ABORT",
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/web-v2":
            return httpx.Response(200, text="<html>ok</html>")
        if request.url.path == "/web-v2/snapshot":
            return httpx.Response(
                200,
                json={
                    "status": snapshot_status,
                    "health_score": 84,
                    "generated_at": generated_at,
                    "warnings": ["data_matrix_attention"] if snapshot_status == "WARNING" else [],
                    "actions": [],
                },
                headers={
                    "cache-control": "no-store, max-age=0",
                    "pragma": "no-cache",
                    "x-quant-lab-bigscreen-cache-stale": "false",
                },
            )
        if request.url.path == "/web-v2/expert-pack/status":
            return httpx.Response(
                200,
                json={
                    "state": "succeeded",
                    "export_date": "2026-07-02",
                    "latest_pack_name": "quant_lab_expert_pack_2026-07-02.zip",
                    "available_pack_name": "quant_lab_expert_pack_2026-07-02.zip",
                    "pack_count": 1,
                    "latest_pack_code_lag_status": "OK",
                    "latest_pack_v5_lag_status": "WARNING",
                },
            )
        if request.url.path == "/v1/health":
            return httpx.Response(200, json={"status": "ok", "service": "quant-lab"})
        if request.url.path == "/v1/health/deep":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "warnings": ["cost_health_warning"],
                    "service_health": {"status": "OK"},
                    "live_entry_readiness": {"status": "ADVISORY_READY"},
                    "cost_health": {"status": "warning"},
                },
            )
        if request.url.path == "/v1/costs/estimate":
            symbol = request.url.params.get("symbol", "BTC-USDT")
            return httpx.Response(
                200,
                json={
                    "symbol": symbol,
                    "source": "public_spread_proxy",
                    "as_of_ts": "2026-07-02T07:13:44Z",
                    "fallback_reason": cost_fallback_reason,
                    "degraded_reason": "none",
                    "cost_trust_level": "PAPER_ONLY",
                    "cost_trusted_for_paper": True,
                    "cost_trusted_for_live": cost_trusted_for_live,
                    "roundtrip_all_in_cost_bps": 28.0,
                },
            )
        if request.url.path == "/v1/web/bigscreen-snapshot":
            return httpx.Response(
                200,
                json={
                    "status": snapshot_status,
                    "health_score": 84,
                    "generated_at": generated_at,
                },
                headers={
                    "cache-control": "no-store, max-age=0",
                    "x-quant-lab-bigscreen-cache-stale": "false",
                },
            )
        if request.url.path == "/v1/catalog/datasets":
            return httpx.Response(200, json={"datasets": ["market_bar", "cost_bucket_daily"]})
        if request.url.path == "/v1/ops/api-metrics":
            return httpx.Response(200, json={"request_count": 12, "by_path": {}})
        if request.url.path == "/v1/gates/example":
            return httpx.Response(200, json={"status": "QUARANTINE"})
        if request.url.path == "/v1/gates/decision/smoke-missing":
            return httpx.Response(200, json={"status": "QUARANTINE"})
        if request.url.path == "/v1/costs/example":
            return httpx.Response(200, json={"source": "example", "as_of_ts": generated_at})
        if request.url.path == "/v1/strategy-opportunity-advisory/v5-compact":
            return httpx.Response(200, json=[])
        if request.url.path == "/v1/risk/live-permission":
            return httpx.Response(200, json=_risk_permission_payload(live_permission))
        if request.url.path == "/v1/risk/live-permission-detail":
            return httpx.Response(
                200,
                json={"permission": _risk_permission_payload(live_permission)},
            )
        return httpx.Response(404, json={"error": "missing route"})

    return handler


def _risk_permission_payload(permission: str) -> dict[str, object]:
    live_allowed = permission.upper() not in {"ABORT", "BLOCKED", "DENY"}
    return {
        "permission": permission,
        "allowed_live_modes": ["canary"] if live_allowed else [],
        "max_single_order_usdt": 5 if live_allowed else 0,
    }
