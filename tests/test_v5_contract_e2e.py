from __future__ import annotations

import json

from fastapi.testclient import TestClient

from quant_lab.api.main import create_app
from quant_lab.e2e import run_v5_contract_e2e


def test_v5_contract_e2e_harness_catches_integration_failures(tmp_path, monkeypatch):
    out_dir = tmp_path / "e2e"
    result = run_v5_contract_e2e(out_dir=out_dir)
    lake_root = result["lake_root"]
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", lake_root)

    client = TestClient(create_app())
    cost_response = client.get(
        "/v1/costs/estimate",
        params={
            "symbol": "BNB-USDT",
            "regime": "Trending",
            "notional_usdt": 1_000,
            "quantile": "p75",
        },
    )
    risk_response = client.get(
        "/v1/risk/live-permission",
        params={"strategy": "v5", "version": "5.0.0"},
    )

    assert cost_response.status_code == 200
    assert risk_response.status_code == 200
    cost = cost_response.json()
    risk = risk_response.json()

    assert result["request_success_count"] == 1
    assert result["request_error_count"] == 1
    assert result["actual_fallback_count"] == 1
    assert result["fallback_rows"] == 1
    assert result["assertions"]["http_200_success_not_fallback"] is True
    assert result["assertions"]["timeout_fallback_unique"] is True
    assert result["bnb_cost_estimate"]["cost_source"] != "global_default"
    assert cost["normalized_symbol"] == "BNB-USDT"
    assert cost["cost_source"] != "global_default"
    assert _canonical_ts(risk["as_of_ts"]) == _canonical_ts(
        result["latest_risk_permission"]["as_of_ts"]
    )
    assert result["readiness"]["readiness_status"] != "READY"

    report_json = out_dir / "e2e_contract_test_report.json"
    report_md = out_dir / "e2e_contract_test_report.md"
    assert report_json.exists()
    assert report_md.exists()
    payload = json.loads(report_json.read_text())
    assert payload["actual_fallback_count"] == 1
    assert "readiness_status" in report_md.read_text()


def _canonical_ts(value: str) -> str:
    return value.replace("Z", "+00:00")
