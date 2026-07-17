from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel

from quant_lab.export_plane.signatures import (
    canonical_json_bytes,
    load_public_key,
    load_signing_key,
    sha256_bytes,
    sha256_file,
    sign_payload,
    verify_payload,
)

__all__ = [
    "canonical_json_bytes",
    "load_public_key",
    "load_signing_key",
    "model_content_sha256",
    "sha256_bytes",
    "sha256_file",
    "sign_model",
    "sign_payload",
    "verify_payload",
]


def model_content_sha256(
    value: BaseModel | Mapping[str, Any],
    *,
    blank_fields: tuple[str, ...] = (),
) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    payload.pop("signature", None)
    for field in blank_fields:
        payload[field] = ""
    return sha256_bytes(canonical_json_bytes(payload))


def sign_model(value: BaseModel, key: Ed25519PrivateKey) -> str:
    return sign_payload(value, key)
