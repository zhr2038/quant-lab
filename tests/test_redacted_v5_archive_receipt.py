from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy/nas_archive/verify_redacted_v5_archive_receipt.py"
SPEC = importlib.util.spec_from_file_location("verify_redacted_v5_archive_receipt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_receipt_verifies_manifest_and_current_payloads(tmp_path: Path) -> None:
    archive = tmp_path / "2026-09-10"
    _write(archive / "sha/evidence.json", b"evidence")
    manifest = _write_manifest(archive)
    receipt = _write_receipt(archive, manifest)

    result = MODULE.verify_redacted_archive_receipt(
        archive,
        receipt,
        manifest,
        day="2026-09-10",
        source_host="qyun2.hrhome.top",
        source_root="/var/lib/quant-lab/archive/v5/bundles",
    )

    assert result["ok"] is True
    assert result["file_count"] == 1
    assert result["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing"])
def test_receipt_fails_closed_when_nas_payload_no_longer_matches(
    tmp_path: Path,
    mutation: str,
) -> None:
    archive = tmp_path / "2026-09-10"
    evidence = _write(archive / "sha/evidence.json", b"evidence")
    manifest = _write_manifest(archive)
    receipt = _write_receipt(archive, manifest)
    if mutation == "changed":
        evidence.write_bytes(b"changed")
    elif mutation == "extra":
        _write(archive / "sha/untracked.json", b"extra")
    else:
        evidence.unlink()

    with pytest.raises(ValueError):
        MODULE.verify_redacted_archive_receipt(
            archive,
            receipt,
            manifest,
            day="2026-09-10",
            source_host="qyun2.hrhome.top",
            source_root="/var/lib/quant-lab/archive/v5/bundles",
        )


def test_receipt_fails_closed_when_identity_changes(tmp_path: Path) -> None:
    archive = tmp_path / "2026-09-10"
    _write(archive / "sha/evidence.json", b"evidence")
    manifest = _write_manifest(archive)
    receipt = _write_receipt(archive, manifest)

    with pytest.raises(ValueError, match="source mismatch"):
        MODULE.verify_redacted_archive_receipt(
            archive,
            receipt,
            manifest,
            day="2026-09-10",
            source_host="unexpected.example",
            source_root="/var/lib/quant-lab/archive/v5/bundles",
        )


def _write_manifest(archive: Path) -> Path:
    payloads = sorted(
        path
        for path in archive.rglob("*")
        if path.is_file() and not path.name.startswith(".archive_")
    )
    lines = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"./{path.relative_to(archive).as_posix()}\n"
        for path in payloads
    )
    return _write(archive / ".archive_manifest.sha256", lines.encode())


def _write_receipt(archive: Path, manifest: Path) -> Path:
    payload = {
        "schema_version": "quant_lab_nas_redacted_archive_receipt.v1",
        "day": "2026-09-10",
        "source": "qyun2.hrhome.top:/var/lib/quant-lab/archive/v5/bundles",
        "file_count": len(manifest.read_text(encoding="utf-8").splitlines()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    return _write(
        archive / ".archive_receipt.json",
        (json.dumps(payload, sort_keys=True) + "\n").encode(),
    )


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
