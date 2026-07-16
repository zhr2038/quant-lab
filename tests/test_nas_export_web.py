from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import quant_lab.api.main as api_main


def test_nas_web_generate_only_submits_lightweight_request(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "queue"
    secret = tmp_path / "download.key"
    secret.write_bytes(b"x" * 48)
    monkeypatch.setenv("QUANT_LAB_NAS_EXPORT_ENABLED", "1")
    monkeypatch.setenv("QUANT_LAB_EXPORT_QUEUE_ROOT", str(queue))
    monkeypatch.setenv("QUANT_LAB_NAS_EXPORT_BASE_URL", "http://nas.invalid:8788")
    monkeypatch.setenv("QUANT_LAB_NAS_DOWNLOAD_SIGNING_SECRET_FILE", str(secret))
    monkeypatch.setenv("QUANT_LAB_HEALTH_ALLOW_LOCAL_UNAUTH", "1")
    monkeypatch.setattr(
        "quant_lab.export.daily.export_daily_pack",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local export called")),
    )
    client = TestClient(api_main.create_app())
    response = client.post("/web-v2/expert-pack/generate")
    assert response.status_code == 200
    payload = response.json()
    assert payload["storage_location"] == "nas_only"
    assert payload["cloud_zip_present"] is False
    assert list((queue / "requests" / "pending").glob("*.json"))


def test_nas_web_cloud_download_endpoint_is_gone(tmp_path: Path, monkeypatch) -> None:
    secret = tmp_path / "download.key"
    secret.write_bytes(b"x" * 48)
    monkeypatch.setenv("QUANT_LAB_NAS_EXPORT_ENABLED", "1")
    monkeypatch.setenv("QUANT_LAB_EXPORT_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("QUANT_LAB_NAS_EXPORT_BASE_URL", "http://nas.invalid:8788")
    monkeypatch.setenv("QUANT_LAB_NAS_DOWNLOAD_SIGNING_SECRET_FILE", str(secret))
    monkeypatch.setenv("QUANT_LAB_HEALTH_ALLOW_LOCAL_UNAUTH", "1")
    response = TestClient(api_main.create_app()).get(
        "/web-v2/expert-pack/download/quant_lab_expert_pack_2026-07-16_fake.zip"
    )
    assert response.status_code == 410
