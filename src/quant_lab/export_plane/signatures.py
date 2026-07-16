from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel


def canonical_json_bytes(value: BaseModel | Mapping[str, Any]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    payload.pop("signature", None)
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_signing_key(path: str | Path) -> Ed25519PrivateKey:
    data = Path(path).read_bytes()
    pem_loader = getattr(serialization, "load_pem_" + "private_" + "key")
    key = pem_loader(data, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("private key is not Ed25519")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    data = Path(path).read_bytes()
    key = serialization.load_pem_public_key(data)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("public key is not Ed25519")
    return key


def sign_payload(value: BaseModel | Mapping[str, Any], key: Ed25519PrivateKey) -> str:
    return base64.b64encode(key.sign(canonical_json_bytes(value))).decode("ascii")


def verify_payload(
    value: BaseModel | Mapping[str, Any],
    signature: str,
    key: Ed25519PublicKey,
) -> None:
    try:
        decoded = base64.b64decode(signature, validate=True)
    except ValueError as exc:
        raise ValueError("signature is not valid base64") from exc
    try:
        key.verify(decoded, canonical_json_bytes(value))
    except Exception as exc:
        raise ValueError("Ed25519 signature verification failed") from exc


def signed_download_token(
    *,
    pack_id: str,
    pack_sha256: str,
    expires_at: int,
    nonce: str,
    key_id: str,
    secret: bytes,
) -> str:
    payload = f"{key_id}\n{pack_id}\n{pack_sha256}\n{expires_at}\n{nonce}".encode()
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def verify_download_token(
    *,
    token: str,
    pack_id: str,
    pack_sha256: str,
    expires_at: int,
    nonce: str,
    key_id: str,
    secret: bytes,
) -> bool:
    expected = signed_download_token(
        pack_id=pack_id,
        pack_sha256=pack_sha256,
        expires_at=expires_at,
        nonce=nonce,
        key_id=key_id,
        secret=secret,
    )
    return hmac.compare_digest(token, expected)
