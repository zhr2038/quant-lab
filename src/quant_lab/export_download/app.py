from __future__ import annotations

import html
import json
import os
import secrets
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse

from quant_lab.export_plane.contracts import ExportPackIndexEntry
from quant_lab.export_plane.signatures import signed_download_token, verify_download_token

app = FastAPI(title="quant-lab NAS Expert Pack Center", docs_url=None, redoc_url=None)


def _accepted_root() -> Path:
    return Path(os.getenv("NAS_EXPORT_ACCEPTED_ROOT", "/exports/accepted")).resolve()


def _index_path() -> Path:
    return Path(os.getenv("NAS_EXPORT_INDEX_PATH", "/exports/accepted_index.json"))


def _worker_status_path() -> Path:
    return Path(os.getenv("NAS_EXPORT_WORKER_STATUS_PATH", "/exports/status/worker.json"))


def _worker_status() -> dict[str, Any]:
    try:
        value = json.loads(_worker_status_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _signing_secret() -> bytes:
    path = Path(os.getenv("NAS_DOWNLOAD_SIGNING_SECRET_FILE", "/run/secrets/nas-download.key"))
    value = path.read_bytes().strip()
    if len(value) < 32:
        raise RuntimeError("NAS download signing secret must contain at least 32 bytes")
    return value


def _key_id() -> str:
    return os.getenv("NAS_DOWNLOAD_KEY_ID", "nas-download-v1")


def _token_ttl() -> int:
    return max(60, min(int(os.getenv("NAS_DOWNLOAD_TOKEN_TTL_SECONDS", "1800")), 86_400))


def _load_index() -> tuple[list[ExportPackIndexEntry], dict[str, Any]]:
    try:
        payload = json.loads(_index_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="accepted_pack_index_unavailable") from exc
    rows: list[ExportPackIndexEntry] = []
    for value in payload.get("packs", []):
        try:
            rows.append(ExportPackIndexEntry.model_validate(value))
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda item: item.accepted_at, reverse=True)
    return rows, payload


def _pack(pack_id: str) -> ExportPackIndexEntry:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not pack_id or any(char not in allowed for char in pack_id):
        raise HTTPException(status_code=404, detail="pack_not_found")
    rows, _ = _load_index()
    for row in rows:
        if row.pack_id == pack_id and row.pack_state == "accepted":
            return row
    raise HTTPException(status_code=404, detail="pack_not_found")


def _safe_pack_path(row: ExportPackIndexEntry) -> Path:
    relative = PurePosixPath(row.download_relative_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(status_code=500, detail="unsafe_pack_index_path")
    root = _accepted_root()
    candidate = (root / relative.as_posix()).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="unsafe_pack_index_path") from exc
    if not candidate.is_file() or candidate.stat().st_size != row.pack_size_bytes:
        raise HTTPException(status_code=503, detail="accepted_pack_file_unavailable")
    return candidate


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    try:
        rows, payload = _load_index()
        disk = shutil.disk_usage(_accepted_root())
        return {
            "status": "ok",
            "storage": "nas_local",
            "accepted_pack_count": len(rows),
            "index_updated_at": payload.get("updated_at"),
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "python_streams_pack_bytes": False,
            "worker": _worker_status(),
        }
    except HTTPException as exc:
        return {"status": "degraded", "reason": exc.detail}


@app.get("/api/v1/packs")
def list_packs() -> dict[str, Any]:
    rows, payload = _load_index()
    return {
        "schema_version": "quant_lab_nas_pack_api.v1",
        "updated_at": payload.get("updated_at"),
        "packs": [row.model_dump(mode="json") for row in rows],
        "worker": _worker_status(),
    }


@app.get("/api/v1/packs/{pack_id}")
def pack_detail(pack_id: str) -> dict[str, Any]:
    row = _pack(pack_id)
    return row.model_dump(mode="json")


@app.get("/packs", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def pack_center() -> str:
    rows, _ = _load_index()
    worker = _worker_status()
    body = []
    for row in rows:
        ready = bool(
            row.authoritative_input_snapshot
            and row.nas_artifact_validated
            and row.control_plane_receipt_verified
            and row.download_ready
            and row.pack_state == "accepted"
        )
        link = _local_download_url(row) if ready else ""
        action = f'<a href="{html.escape(link)}">下载</a>' if link else "等待云端验签"
        body.append(
            "<tr>"
            f"<td>{html.escape(row.pack_name)}</td>"
            f"<td>{row.export_date.isoformat()}</td>"
            f"<td><code>{html.escape(row.pack_id)}</code></td>"
            f"<td>{row.pack_size_bytes / 1024 / 1024:.1f} MB</td>"
            f"<td><code>{row.pack_sha256[:16]}...</code></td>"
            f"<td>{'已验证' if ready else '处理中'}</td>"
            f"<td>{action}</td>"
            "</tr>"
        )
    rows_html = "".join(body) or '<tr><td colspan="7">暂无已验收专家包</td></tr>'
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Expert Pack NAS</title><style>
body{{font-family:system-ui,sans-serif;background:#071523;color:#e8f4ff;margin:0;padding:28px}}
main{{max-width:1280px;margin:auto}} h1{{font-size:28px}} p{{color:#9fb6c9}}
table{{width:100%;border-collapse:collapse;background:#0b2134}}
th,td{{padding:12px;border-bottom:1px solid #21435d;text-align:left}}
a{{color:#51e0bd}} code{{font-size:12px}} .note{{padding:12px;border:1px solid #2d6c78}}
</style></head><body><main><h1>Expert Pack NAS</h1>
<p class="note">文件保存在家庭 NAS，下载流量不经过中台。<br>
仅 accepted 且云端回执验签通过的包可下载。</p>
<p>Worker: <code>{html.escape(str(worker.get('worker_commit') or 'unknown')[:12])}</code>
 · 状态 {html.escape(str(worker.get('state') or 'unknown'))}
 · 心跳 {html.escape(str(worker.get('heartbeat_at') or 'not_observable'))}</p>
<table><thead><tr><th>文件</th><th>日期</th><th>Pack ID</th><th>大小</th>
<th>SHA256</th><th>状态</th><th>操作</th></tr></thead>
<tbody>{rows_html}</tbody></table></main></body></html>"""


@app.get("/download/{pack_id}")
def download_pack(
    pack_id: str,
    sha256: str = Query(min_length=64, max_length=64),
    expires: int = Query(gt=0),
    nonce: str = Query(min_length=8, max_length=100),
    key_id: str = Query(min_length=1, max_length=120),
    signature: str = Query(min_length=64, max_length=64),
) -> Response:
    row = _pack(pack_id)
    if not all(
        (
            row.authoritative_input_snapshot,
            row.nas_artifact_validated,
            row.control_plane_receipt_verified,
            row.download_ready,
            row.pack_state == "accepted",
        )
    ):
        raise HTTPException(status_code=409, detail="pack_not_download_ready")
    now = int(time.time())
    if expires < now or expires > now + _token_ttl() + 60:
        raise HTTPException(status_code=403, detail="download_token_expired_or_invalid_ttl")
    if key_id != _key_id() or sha256.lower() != row.pack_sha256:
        raise HTTPException(status_code=403, detail="download_token_binding_mismatch")
    if not verify_download_token(
        token=signature,
        pack_id=row.pack_id,
        pack_sha256=row.pack_sha256,
        expires_at=expires,
        nonce=nonce,
        key_id=key_id,
        secret=_signing_secret(),
    ):
        raise HTTPException(status_code=403, detail="download_token_invalid")
    path = _safe_pack_path(row)
    quoted_relative = quote(row.download_relative_path, safe="/")
    return Response(
        status_code=200,
        headers={
            "X-Accel-Redirect": f"/_protected/{quoted_relative}",
            "Content-Type": "application/zip",
            "Content-Disposition": f'attachment; filename="{path.name}"',
            "Accept-Ranges": "bytes",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _local_download_url(row: ExportPackIndexEntry) -> str:
    expires = int(time.time()) + _token_ttl()
    nonce = secrets.token_urlsafe(12)
    signature = signed_download_token(
        pack_id=row.pack_id,
        pack_sha256=row.pack_sha256,
        expires_at=expires,
        nonce=nonce,
        key_id=_key_id(),
        secret=_signing_secret(),
    )
    return "/download/" + quote(row.pack_id, safe="") + "?" + urlencode(
        {
            "sha256": row.pack_sha256,
            "expires": expires,
            "nonce": nonce,
            "key_id": _key_id(),
            "signature": signature,
        }
    )
