from __future__ import annotations

import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_lab.export.daily import export_daily_pack

WEB_EXPORT_MODE = "authoritative_snapshot"


def run_web_export_request(request_path: Path) -> dict[str, Any]:
    request = _read_json(request_path)
    export_date = str(request.get("export_date") or "").strip()
    lake_root = Path(str(request.get("lake_root") or ""))
    exports_root = Path(str(request.get("exports_root") or ""))
    status_path = Path(
        str(
            request.get("status_path")
            or exports_root / f".quant_lab_web_export_{export_date}.json"
        )
    )
    if not export_date or not str(lake_root) or not str(exports_root):
        raise ValueError("web export request is missing export_date/lake_root/exports_root")

    started_at = datetime.now(UTC).isoformat()
    base_status = {
        "state": "running",
        "mode": WEB_EXPORT_MODE,
        "trigger": "request_file",
        "export_date": export_date,
        "lake_root": str(lake_root),
        "exports_root": str(exports_root),
        "status_path": str(status_path),
        "request_path": str(request_path),
        "worker_pid": os.getpid(),
        "started_at": started_at,
        "requested_at": request.get("requested_at"),
    }
    _write_status(status_path, base_status)
    try:
        result = export_daily_pack(
            export_date=export_date,
            lake_root=lake_root,
            out_dir=exports_root,
            profile="expert",
            command_line=[
                "qlab",
                "export-daily",
                "--date",
                export_date,
                "--lake-root",
                str(lake_root),
                "--out-dir",
                str(exports_root),
                "--no-refresh-risk-permission",
                "--pre-export-v5-refresh",
            ],
            refresh_risk_permission=False,
            risk_strategy="v5",
            risk_version="5.0.0",
            pre_export_v5_refresh=True,
        )
        status = {
            **base_status,
            "state": "succeeded",
            "zip_path": result.zip_path,
            "warnings": list(result.warnings),
            "finished_at": datetime.now(UTC).isoformat(),
        }
        _write_status(status_path, status)
        return status
    except Exception as exc:
        status = {
            **base_status,
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc()[-4000:],
            "finished_at": datetime.now(UTC).isoformat(),
        }
        _write_status(status_path, status)
        raise
    finally:
        try:
            request_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read web export request: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"web export request must be a JSON object: {path}")
    return payload


def _write_status(status_path: Path, status: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(status, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp_path.replace(status_path)
