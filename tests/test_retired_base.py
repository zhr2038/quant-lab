import sys

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from quant_lab.api.main import create_app
from quant_lab.cli import app as cli
from quant_lab.strategy_telemetry.analyze import analyze_v5_telemetry


def test_retired_web_returns_gone_without_jobs_or_data_scans(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("base startup must not scan historical datasets")

    monkeypatch.setattr("quant_lab.api.main._strategy_opportunity_advisory_snapshot", unexpected)
    monkeypatch.setattr("quant_lab.api.main._cost_bucket_snapshot", unexpected)
    with TestClient(create_app()) as client:
        assert client.get("/").status_code == 200
        assert "交易参考" in client.get("/").text
        for path in ("/web-v2", "/web-v2/", "/web-v2/snapshot"):
            response = client.get(path)
            assert response.status_code == 410
            assert response.json()["status"] == "retired"
        assert client.post("/web-v2/expert-pack/generate").status_code == 410
        assert client.get("/v1/health").status_code == 200
    assert "quant_lab.web.bigscreen" not in sys.modules


def test_auth_still_protects_strategy_endpoints(monkeypatch):
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "test-only-retirement-token")
    with TestClient(create_app()) as client:
        assert client.get("/v1/catalog/datasets").status_code == 401
        response = client.get(
            "/v1/catalog/datasets",
            headers={"Authorization": "Bearer test-only-retirement-token"},
        )
        assert response.status_code == 200
        assert response.headers["X-Quant-Lab-Lifecycle"] == "base-with-legacy-read-compat"


def test_retired_producer_command_cannot_be_dispatched():
    result = CliRunner().invoke(cli, ["build-alpha-factory", "--help"])
    assert result.exit_code == 2
    assert "No such command" in result.output
    assert CliRunner().invoke(cli, ["sync-v5-telemetry", "--help"]).exit_code == 0


def test_explicit_retired_research_refresh_fails_before_writes(tmp_path):
    lake = tmp_path / "uncreated-lake"
    with pytest.raises(ValueError, match="retired"):
        analyze_v5_telemetry(lake, refresh_candidate_gold=True)
    assert not lake.exists()
