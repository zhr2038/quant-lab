from __future__ import annotations

from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import create_app
from quant_lab.data.lake import write_parquet_dataset
from quant_lab.web.bigscreen import bigscreen_snapshot, clear_bigscreen_cache


def test_bigscreen_snapshot_empty_lake_is_read_only_and_degraded(tmp_path):
    clear_bigscreen_cache()
    payload = bigscreen_snapshot(tmp_path / "lake")

    assert payload["mode"] == "read-only"
    assert payload["status"] in {"WARNING", "CRITICAL", "OK"}
    assert 0 <= payload["health_score"] <= 100
    assert isinstance(payload["actions"], list)
    assert len(payload["actions"]) <= 8
    assert payload["data_matrix"]["columns"] == [
        "market_bar",
        "ws",
        "spread",
        "trade",
        "cost",
        "evidence",
        "advisory",
    ]


def test_bigscreen_snapshot_redacts_secret_like_fields(tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "permission": "ALLOW",
                    "permission_status": "ALLOW",
                    "api_key": "should-not-leak",
                    "as_of_ts": "2026-06-04T00:00:00Z",
                }
            ]
        ),
        lake / "gold" / "risk_permission",
    )

    payload = bigscreen_snapshot(lake)
    rows = payload["consumers"]["permission_rows"]

    assert rows
    assert rows[0]["api_key"] == "[REDACTED]"
    assert "should-not-leak" not in str(payload)


def test_bigscreen_snapshot_endpoints_return_payload(monkeypatch, tmp_path):
    clear_bigscreen_cache()
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    client = TestClient(create_app())

    protected_response = client.get("/v1/web/bigscreen-snapshot")
    web_response = client.get("/web-v2/snapshot")

    assert protected_response.status_code == 200
    assert web_response.status_code == 200
    assert protected_response.json()["mode"] == "read-only"
    assert web_response.json()["mode"] == "read-only"


def test_bigscreen_static_entry_is_served_if_built():
    static_index = Path("src/quant_lab/web/bigscreen_static/index.html")
    if not static_index.is_file():
        return

    response = TestClient(create_app()).get("/web-v2")

    assert response.status_code == 200
    assert "quant-lab CONTROL CENTER" in response.text
