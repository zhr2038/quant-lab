from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from quant_lab.decision.contracts_v2 import parse_advice, parse_result
from quant_lab.decision.engine import advice_identity
from quant_lab.decision.server_status import server_status
from quant_lab.decision.storage import (
    MAX_RESULT_BYTES,
    effective_snapshot,
    read_json,
    result_identity,
)
from quant_lab.export_plane.signatures import load_public_key, verify_payload

HEADERS = {"Cache-Control": "no-store", "X-Quant-Lab-Decision-Scope": "research_only"}
ASSETS = Path(__file__).parent / "assets"


def is_public_workbench_request(method: str, path: str) -> bool:
    return method == "GET" and (
        path == "/v1/trade-advice/latest"
        or path == "/v1/server-status"
        or re.fullmatch(r"/v1/trade-advice/advice-[a-f0-9]{64}", path) is not None
    )


def install_routes(app: FastAPI, lake_root: Callable[[], Path]) -> None:
    @app.get("/v1/server-status")
    def status_snapshot():
        return JSONResponse(server_status(), headers=HEADERS)

    @app.get("/", include_in_schema=False)
    def workbench():
        return FileResponse(
            ASSETS / "workbench.html",
            headers={
                **HEADERS,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; "
                "style-src 'self'; connect-src 'self'; img-src 'self' data:; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            },
        )

    @app.get("/assets/decision/{name}", include_in_schema=False)
    def asset(name: str):
        if name not in {"workbench.css", "workbench.js"}:
            raise HTTPException(404)
        return FileResponse(ASSETS / name, headers=HEADERS)

    @app.get("/v1/trade-advice/latest")
    def latest():
        now = datetime.now(UTC)
        path = lake_root() / "gold" / "decision_reference" / "publication.json"
        if not path.exists():
            return JSONResponse(
                {"advice": [], "effective_status": "NO_RESULT", "viewed_at": now.isoformat()},
                headers=HEADERS,
            )
        try:
            value = read_json(path, max_bytes=MAX_RESULT_BYTES + 4096)
            raw = value["result"]
            key = load_public_key(
                os.environ.get(
                    "QUANT_LAB_DECISION_WORKER_PUBLIC_KEY", "/etc/quant-lab/decision/worker.pub"
                )
            )
            verify_payload(raw, raw["signature"], key)
            result = parse_result(raw)
            if result.result_id != result_identity(result) or any(
                a.advice_id != advice_identity(a) for a in result.advice
            ):
                raise ValueError("published content identity mismatch")
            if result.generated_at > now:
                raise ValueError("published result is from the future")
            response = effective_snapshot(result, now)
            response["publication"] = value["publication"]
            return JSONResponse(response, headers=HEADERS)
        except (ValueError, KeyError, OSError):
            return JSONResponse(
                {
                    "advice": [],
                    "effective_status": "INTEGRITY_ERROR",
                    "viewed_at": now.isoformat(),
                    "detail": "Published reference failed validation",
                },
                status_code=503,
                headers=HEADERS,
            )

    @app.get("/v1/trade-advice/{advice_id}")
    def detail(advice_id: str):
        if not re.fullmatch(r"advice-[a-f0-9]{64}", advice_id):
            raise HTTPException(404)
        path = lake_root() / "gold" / "decision_reference" / "advice" / (advice_id + ".json")
        if not path.exists():
            raise HTTPException(404, "Reference not in hot storage; consult NAS archive")
        try:
            raw = read_json(path, max_bytes=64 * 1024)
            advice = parse_advice(raw["advice"])
            if advice.advice_id != advice_id or advice_identity(advice) != advice_id:
                raise ValueError("detail identity mismatch")
            value = advice.model_dump(mode="json")
            now = datetime.now(UTC)
            value.update(
                expired=now >= advice.expires_at,
                effective_action="NO_VIEW" if now >= advice.expires_at else advice.action,
                viewed_at=now.isoformat(),
                publication=raw["publication"],
            )
            return JSONResponse(value, headers=HEADERS)
        except (ValueError, KeyError, OSError) as exc:
            raise HTTPException(503, "Reference detail failed validation") from exc
