from __future__ import annotations

import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

from quant_lab.strategy_telemetry.models import (
    RedactionResult,
    SecretFinding,
    SecretScanResult,
)

REDACTION = "<REDACTED>"
SAFE_REDACTED_VALUES = {"<redacted>", "redacted", "null", "none", "false", "true", "0"}
SAFE_REDACTED_VALUE_NEGATIVE_LOOKAHEAD = (
    r"(?!(?:<REDACTED>|REDACTED|null|none|false|true|0)(?=$|[\s,}\"'}\]]))"
)

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "apiSecret",
    "api" + "_secret",
    "secret" + "_key",
    "passphrase",
    "private" + "_key",
    "ok-access-key",
    "ok-access-sign",
    "ok-access-passphrase",
    "authorization",
    "token",
    "password",
}

SECRET_PATTERNS = [
    (re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"), "high", "private-key-block"),
    (re.compile(r"ssh-rsa\s+[A-Za-z0-9+/=]+"), "high", "ssh-rsa"),
    (re.compile(r"ed25519.*PRIVATE KEY", re.IGNORECASE), "high", "ed25519-private"),
    (re.compile(r"OK-ACCESS-(KEY|SIGN|PASSPHRASE)", re.IGNORECASE), "high", "okx-auth-header"),
    (
        re.compile(
            r"(api[_-]?key|apiSecret)\s*[:=]\s*['\"]?"
            + SAFE_REDACTED_VALUE_NEGATIVE_LOOKAHEAD
            + r"[^'\"\s,}]+",
            re.IGNORECASE,
        ),
        "high",
        "api-key",
    ),
    (
        re.compile(
            r"(secret[_-]?key|api_secret)\s*[:=]\s*['\"]?"
            + SAFE_REDACTED_VALUE_NEGATIVE_LOOKAHEAD
            + r"[^'\"\s,}]+",
            re.IGNORECASE,
        ),
        "high",
        "secret-key",
    ),
    (
        re.compile(
            r"(passphrase|password|token)\s*[:=]\s*['\"]?"
            + SAFE_REDACTED_VALUE_NEGATIVE_LOOKAHEAD
            + r"[^'\"\s,}]+",
            re.IGNORECASE,
        ),
        "high",
        "credential-field",
    ),
]


def scan_for_secrets(path_or_text: str | Path) -> SecretScanResult:
    path = _path_if_existing(path_or_text)
    if path is not None:
        if path.is_dir():
            return _scan_dir(path)
        if path.is_file():
            return _scan_file(path)
    findings = _findings_in_text(str(path_or_text), source_path="<text>")
    return _scan_result(scanned_files=1, findings=findings, warnings=[])


def redact_text(text: str) -> str:
    redacted = text
    for pattern, _severity, _label in SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: _redact_match(match.group(0)), redacted)
    return redacted


def redact_json_like(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: REDACTION if _sensitive_key(str(key)) else redact_json_like(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [redact_json_like(value) for value in obj]
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


def redact_extracted_bundle(extracted_dir: Path, redacted_dir: Path) -> RedactionResult:
    source = Path(extracted_dir)
    target = Path(redacted_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    redacted_files: list[str] = []
    warnings: list[str] = []

    for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = file_path.relative_to(source)
        destination_relative = _redacted_relative(relative)
        destination = target / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(file_path, destination)
            copied.append(str(destination_relative).replace("\\", "/"))
            continue
        safe_text = redact_text(text)
        destination.write_text(safe_text, encoding="utf-8")
        copied.append(str(destination_relative).replace("\\", "/"))
        if safe_text != text:
            redacted_files.append(str(destination_relative).replace("\\", "/"))

    return RedactionResult(
        source_dir=str(source),
        redacted_dir=str(target),
        copied_files=copied,
        redacted_files=redacted_files,
        warnings=warnings,
    )


def _scan_dir(path: Path) -> SecretScanResult:
    findings: list[SecretFinding] = []
    warnings: list[str] = []
    scanned = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        result = _scan_file(file_path)
        scanned += result.scanned_files
        findings.extend(result.findings)
        warnings.extend(result.warnings)
    return _scan_result(scanned_files=scanned, findings=findings, warnings=warnings)


def _scan_file(path: Path) -> SecretScanResult:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _scan_result(scanned_files=1, findings=[], warnings=[f"binary file skipped: {path}"])
    findings = _findings_in_text(text, source_path=str(path))
    return _scan_result(scanned_files=1, findings=findings, warnings=[])


def _findings_in_text(text: str, source_path: str) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern, severity, label in SECRET_PATTERNS:
            match = pattern.search(line)
            if match and not _match_has_safe_redacted_value(match.group(0)):
                findings.append(
                    SecretFinding(
                        source_path=source_path,
                        pattern=label,
                        severity=severity,
                        line_number=line_number,
                    )
                )
    return findings


def _scan_result(
    scanned_files: int,
    findings: list[SecretFinding],
    warnings: list[str],
) -> SecretScanResult:
    high = sum(1 for finding in findings if finding.severity == "high")
    medium = sum(1 for finding in findings if finding.severity == "medium")
    low = sum(1 for finding in findings if finding.severity == "low")
    return SecretScanResult(
        scanned_files=scanned_files,
        findings=findings,
        high_severity_count=high,
        medium_severity_count=medium,
        low_severity_count=low,
        redaction_required=bool(findings),
        warnings=warnings,
    )


def _redact_match(value: str) -> str:
    if ":" in value:
        return value.split(":", 1)[0] + ": " + REDACTION
    if "=" in value:
        return value.split("=", 1)[0] + "=" + REDACTION
    return REDACTION


def _match_has_safe_redacted_value(value: str) -> bool:
    if ":" in value:
        raw_value = value.split(":", 1)[1]
    elif "=" in value:
        raw_value = value.split("=", 1)[1]
    else:
        return False
    cleaned = raw_value.strip().strip("\"' ,;#}]<>").lower()
    return cleaned in SAFE_REDACTED_VALUES


def _redacted_relative(relative: Path) -> Path:
    parts = relative.parts
    if len(parts) > 1 and parts[0].startswith("v5_live_followup_bundle_"):
        return Path(*parts[1:])
    return relative


def _sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return normalized in {item.replace("-", "_").lower() for item in SENSITIVE_KEYS}


def _path_if_existing(path_or_text: str | Path) -> Path | None:
    if isinstance(path_or_text, Path):
        return path_or_text if path_or_text.exists() else None
    text = str(path_or_text)
    if not text.strip():
        return None
    if "\n" in text or "\r" in text or len(text) > 240:
        return None
    try:
        path = Path(text)
        return path if path.exists() else None
    except OSError:
        return None


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(redact_json_like(obj), ensure_ascii=False, sort_keys=True, default=str)
