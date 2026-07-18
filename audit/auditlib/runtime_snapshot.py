"""Secret-safe primitives for the read-only V5 runtime snapshot."""

from __future__ import annotations

import hashlib
import re

_SENSITIVE = re.compile(
    r"(?:api[_-]?key|secret|pass(?:word|phrase)?|token|private[_-]?key|database[_-]?(?:url|password)|dsn)",
    re.IGNORECASE,
)


def digest_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def environment_presence(environment: dict[str, str]) -> dict[str, dict[str, str | bool]]:
    """Store names, presence, and hashes only; never return raw values."""
    return {
        name: {
            "present": value is not None and value != "",
            "sha256": digest_value(value) if value else "",
            "sensitive_name": bool(_SENSITIVE.search(name)),
        }
        for name, value in sorted(environment.items())
    }


def assert_no_sensitive_values(serialized: str, environment: dict[str, str]) -> None:
    """Fail if any non-empty environment value appears in serialized output."""
    for value in environment.values():
        if value and len(value) >= 4 and value in serialized:
            raise ValueError("runtime snapshot contains a raw environment value")
