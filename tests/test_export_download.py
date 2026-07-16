from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from quant_lab.export_download.app import app
from quant_lab.export_plane.contracts import ExportPackIndexEntry
from quant_lab.export_plane.signatures import signed_download_token


def _install_pack(tmp_path: Path, monkeypatch) -> tuple[ExportPackIndexEntry, bytes]:
    accepted = tmp_path / "accepted"
    relative = "2026/07/16/expert-pack-test/expert.zip"
    pack = accepted / relative
    pack.parent.mkdir(parents=True)
    pack.write_bytes(b"PK\x03\x04test-pack")
    secret = b"s" * 48
    secret_path = tmp_path / "download.key"
    secret_path.write_bytes(secret)
    row = ExportPackIndexEntry(
        pack_id="expert-pack-test",
        pack_name=pack.name,
        export_date=date(2026, 7, 16),
        generated_at=datetime(2026, 7, 16, tzinfo=UTC),
        accepted_at=datetime(2026, 7, 16, tzinfo=UTC),
        pack_sha256="a" * 64,
        pack_size_bytes=pack.stat().st_size,
        snapshot_id="export-snapshot-test",
        authoritative_input_snapshot=True,
        nas_artifact_validated=True,
        control_plane_receipt_verified=True,
        download_ready=True,
        download_relative_path=relative,
        selected_v5_bundle_sha256="b" * 64,
        acceptance_set_id="acceptance-test",
        worker_id="worker-1",
        worker_commit="c" * 40,
    )
    index = tmp_path / "accepted_index.json"
    index.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "packs": [row.model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NAS_EXPORT_ACCEPTED_ROOT", str(accepted))
    monkeypatch.setenv("NAS_EXPORT_INDEX_PATH", str(index))
    monkeypatch.setenv("NAS_DOWNLOAD_SIGNING_SECRET_FILE", str(secret_path))
    monkeypatch.setenv("NAS_DOWNLOAD_KEY_ID", "nas-download-v1")
    monkeypatch.setenv("NAS_DOWNLOAD_TOKEN_TTL_SECONDS", "1800")
    return row, secret


def _query(row: ExportPackIndexEntry, secret: bytes, expires: int) -> dict[str, object]:
    nonce = "test-nonce-123"
    return {
        "sha256": row.pack_sha256,
        "expires": expires,
        "nonce": nonce,
        "key_id": "nas-download-v1",
        "signature": signed_download_token(
            pack_id=row.pack_id,
            pack_sha256=row.pack_sha256,
            expires_at=expires,
            nonce=nonce,
            key_id="nas-download-v1",
            secret=secret,
        ),
    }


def test_download_returns_nginx_redirect_without_file_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    row, secret = _install_pack(tmp_path, monkeypatch)
    response = TestClient(app).get(
        f"/download/{row.pack_id}",
        params=_query(row, secret, int(time.time()) + 300),
    )
    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["x-accel-redirect"].endswith("/expert.zip")
    assert response.headers["accept-ranges"] == "bytes"


def test_download_rejects_expired_signature(tmp_path: Path, monkeypatch) -> None:
    row, secret = _install_pack(tmp_path, monkeypatch)
    response = TestClient(app).get(
        f"/download/{row.pack_id}",
        params=_query(row, secret, int(time.time()) - 1),
    )
    assert response.status_code == 403


def test_download_rejects_unverified_pack(tmp_path: Path, monkeypatch) -> None:
    row, secret = _install_pack(tmp_path, monkeypatch)
    index = Path(str(tmp_path / "accepted_index.json"))
    unverified = row.model_copy(update={"control_plane_receipt_verified": False})
    index.write_text(json.dumps({"packs": [unverified.model_dump(mode="json")]}), encoding="utf-8")
    response = TestClient(app).get(
        f"/download/{row.pack_id}",
        params=_query(row, secret, int(time.time()) + 300),
    )
    assert response.status_code == 409
