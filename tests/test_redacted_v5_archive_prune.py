from __future__ import annotations

import hashlib
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy/nas_archive/prune_verified_redacted_v5_archive.py"
SPEC = importlib.util.spec_from_file_location("prune_verified_redacted_v5_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_source_manifest_matches_sha256sum_manifest_contract(tmp_path: Path) -> None:
    day = tmp_path / "2026-09-10"
    first = _write(day / "a/file.json", b"alpha")
    second = _write(day / "z/file.json", b"omega")

    manifest = MODULE.build_source_manifest(day)

    lines = "".join(
        [
            f"{hashlib.sha256(first.read_bytes()).hexdigest()}  ./a/file.json\n",
            f"{hashlib.sha256(second.read_bytes()).hexdigest()}  ./z/file.json\n",
        ]
    ).encode()
    assert manifest.sha256 == hashlib.sha256(lines).hexdigest()
    assert manifest.file_count == 2


def test_verified_redacted_day_dry_run_and_apply(tmp_path: Path) -> None:
    source = tmp_path / "archive/v5/bundles"
    day = source / "2026-09-10"
    _write(day / "sha-a/redacted_files/evidence.json", b"evidence")
    _write(day / "sha-b/provenance.json", b"provenance")
    manifest = MODULE.build_source_manifest(day)

    preview = MODULE.prune_verified_redacted_day(
        source,
        "2026-09-10",
        expected_manifest_sha256=manifest.sha256,
        expected_file_count=manifest.file_count,
        apply=False,
        today=date(2026, 9, 13),
    )
    assert preview["removed"] is False
    assert preview["eligible_bytes"] == len(b"evidence") + len(b"provenance")
    assert day.exists()

    applied = MODULE.prune_verified_redacted_day(
        source,
        "2026-09-10",
        expected_manifest_sha256=manifest.sha256,
        expected_file_count=manifest.file_count,
        apply=True,
        today=date(2026, 9, 13),
    )
    assert applied["removed"] is True
    assert applied["removed_bytes"] == preview["eligible_bytes"]
    assert not day.exists()


def test_redacted_day_prune_fails_closed_after_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "archive/v5/bundles"
    day = source / "2026-09-10"
    evidence = _write(day / "sha/evidence.json", b"before")
    manifest = MODULE.build_source_manifest(day)
    evidence.write_bytes(b"after")

    with pytest.raises(ValueError, match="source manifest sha256 mismatch"):
        MODULE.prune_verified_redacted_day(
            source,
            "2026-09-10",
            expected_manifest_sha256=manifest.sha256,
            expected_file_count=manifest.file_count,
            apply=True,
            today=date(2026, 9, 13),
        )

    assert day.exists()


def test_redacted_day_prune_rejects_current_day_and_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "archive/v5/bundles"
    current = source / "2026-09-13"
    _write(current / "sha/evidence.json", b"evidence")
    manifest = MODULE.build_source_manifest(current)

    with pytest.raises(ValueError, match="current or future"):
        MODULE.prune_verified_redacted_day(
            source,
            "2026-09-13",
            expected_manifest_sha256=manifest.sha256,
            expected_file_count=manifest.file_count,
            apply=True,
            today=date(2026, 9, 13),
        )

    old = source / "2026-09-10"
    _write(old / "sha/evidence.json", b"evidence")
    try:
        (old / "link").symlink_to(old / "sha/evidence.json")
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="contains a symlink"):
        MODULE.build_source_manifest(old)


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
