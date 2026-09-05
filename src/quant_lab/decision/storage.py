from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel

from quant_lab.decision.contracts import AnalysisResult, InputSnapshot
from quant_lab.decision.engine import advice_identity, content_hash
from quant_lab.export_plane.signatures import verify_payload

MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 512 * 1024


def atomic_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("artifact must not be a symlink")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("artifact exceeds byte budget")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("artifact must be a JSON object")
    return value


def input_identity(value: InputSnapshot | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    for name in ("snapshot_id", "generated_at", "signature", "schema_version"):
        payload.pop(name, None)
    return "input-" + content_hash(payload)


def result_identity(value: AnalysisResult | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    payload.pop("result_id", None)
    payload.pop("signature", None)
    return "result-" + content_hash(payload)


def load_input(path: Path, key: Ed25519PublicKey) -> InputSnapshot:
    value = read_json(path, max_bytes=MAX_INPUT_BYTES)
    verify_payload(value, str(value.get("signature", "")), key)
    inputs = InputSnapshot.model_validate(value)
    if inputs.snapshot_id != input_identity(inputs):
        raise ValueError("input content identity mismatch")
    return inputs


def load_result(path: Path, key: Ed25519PublicKey) -> AnalysisResult:
    value = read_json(path, max_bytes=MAX_RESULT_BYTES)
    return validate_result(value, key)


def validate_result(value: dict[str, Any], key: Ed25519PublicKey) -> AnalysisResult:
    verify_payload(value, str(value.get("signature", "")), key)
    result = AnalysisResult.model_validate(value)
    if result.result_id != result_identity(result):
        raise ValueError("result content identity mismatch")
    if any(advice.advice_id != advice_identity(advice) for advice in result.advice):
        raise ValueError("advice content identity mismatch")
    return result


def effective_snapshot(result: AnalysisResult, now: datetime) -> dict[str, Any]:
    """Expiry changes the read view, never the immutable published forecast."""
    value = result.model_dump(mode="json")
    for raw, advice in zip(value["advice"], result.advice, strict=True):
        expired = now >= advice.expires_at
        raw["effective_action"] = "NO_VIEW" if expired else advice.action
        raw["expired"] = expired
        raw["effective_reason_codes"] = advice.reason_codes + (
            ["ADVICE_EXPIRED"] if expired else []
        )
    value["viewed_at"] = now.isoformat()
    value["effective_status"] = (
        "EXPIRED" if all(item["expired"] for item in value["advice"]) else "AVAILABLE"
    )
    return value
