from __future__ import annotations

import json
import zipfile
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import create_app
from quant_lab.data.lake import write_parquet_dataset
from quant_lab.web.bigscreen import _exports_payload, bigscreen_snapshot, clear_bigscreen_cache


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


def test_web_v2_expert_pack_generate_submits_read_only_job(monkeypatch, tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)

    calls: list[dict[str, str]] = []

    def fake_start_export_job(*, export_date, lake_root, exports_root):
        calls.append(
            {
                "export_date": export_date,
                "lake_root": str(lake_root),
                "exports_root": str(exports_root),
            }
        )
        return {
            "state": "running",
            "export_date": export_date,
            "trigger": "request_file",
            "request_path": str(exports_root / ".quant_lab_web_export_request.json"),
        }

    monkeypatch.setattr(
        "quant_lab.web.pages.expert_exports._start_export_job",
        fake_start_export_job,
    )

    response = TestClient(create_app()).post("/web-v2/expert-pack/generate")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "read_only_export"
    assert payload["live_order_effect"] == "none"
    assert payload["state"] == "running"
    assert calls
    assert calls[0]["lake_root"] == str(lake)
    assert calls[0]["exports_root"] == str(tmp_path / "exports")


def test_web_v2_expert_pack_status_and_download(monkeypatch, tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    exports = tmp_path / "exports"
    pack = exports / "quant_lab_expert_pack_2026-06-05_120000.zip"
    exports.mkdir(parents=True)
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"status": "OK"}))
        archive.writestr("data_quality.json", json.dumps({"status": "OK"}))
        archive.writestr("expert_questions.md", "下一步看什么？\n")

    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    client = TestClient(create_app())

    status_response = client.get("/web-v2/expert-pack/status?export_date=2026-06-05")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["latest_pack_name"] == pack.name
    assert status_payload["latest_download_url"] == f"/web-v2/expert-pack/download/{pack.name}"
    assert status_payload["packs"][0]["download_url"] == f"/web-v2/expert-pack/download/{pack.name}"

    download_response = client.get(f"/web-v2/expert-pack/download/{pack.name}")
    assert download_response.status_code == 200
    assert download_response.headers["content-type"].startswith("application/zip")

    traversal_response = client.get("/web-v2/expert-pack/download/..%5Csecret.zip")
    assert traversal_response.status_code == 404


def test_bigscreen_exports_payload_separates_job_state_from_data_quality():
    pack_path = "/var/lib/quant-lab/exports/quant_lab_expert_pack_2026-06-05_120000.zip"

    payload = _exports_payload(
        {
            "latest_pack": pack_path,
            "packs": pl.DataFrame(
                [
                    {
                        "path": pack_path,
                        "name": "quant_lab_expert_pack_2026-06-05_120000.zip",
                        "size_bytes": 123,
                    }
                ]
            ),
            "manifest_summary": {},
            "data_quality_summary": {"status": "CRITICAL", "warning_count": 7},
            "expert_questions": [],
        }
    )

    assert payload["job_state"] == "pack_available"
    assert payload["data_quality_status"] == "CRITICAL"
    assert payload["data_quality_warning_count"] == 7
