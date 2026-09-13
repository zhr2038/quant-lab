from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.strategy_telemetry.retention import (
    prune_v5_telemetry_storage,
    write_v5_telemetry_retention_report,
)


def test_retention_dry_run_reports_only_ingested_raw_and_inbox_copies(tmp_path: Path) -> None:
    ingested_bundle = _write(tmp_path / "inbox/v5/bundles/ingested.tar.gz", b"ingested")
    untracked_bundle = _write(tmp_path / "inbox/v5/bundles/untracked.tar.gz", b"untracked")
    _set_mtime(ingested_bundle, datetime(2026, 9, 8, tzinfo=UTC))
    _set_mtime(untracked_bundle, datetime(2026, 9, 8, tzinfo=UTC))
    ingested_sha = _sha256(ingested_bundle.read_bytes())
    old_raw = _write(
        tmp_path
        / "archive_restricted/v5/bundles/2026-09-05"
        / ingested_sha
        / "raw_bundle.tar.gz",
        b"ingested",
    )
    redacted = _write(
        tmp_path / "archive/v5/bundles/2026-09-05/redacted/evidence.json",
        b"keep coordinated by NAS",
    )
    _write_manifest(tmp_path, [ingested_sha])

    result = prune_v5_telemetry_storage(
        tmp_path,
        keep_restricted_archive_days=7,
        keep_inbox_days=2,
        dry_run=True,
        now=datetime(2026, 9, 13, 12, tzinfo=UTC),
    )

    assert result.ok is True
    assert result.restricted_archive_removed_days == 1
    assert result.inbox_removed_files == 1
    assert result.preserved_uningested_paths == [str(untracked_bundle)]
    assert result.removed_bytes > 0
    assert ingested_bundle.exists()
    assert untracked_bundle.exists()
    assert old_raw.exists()
    assert redacted.exists()


def test_retention_apply_keeps_recent_and_uningested_evidence(tmp_path: Path) -> None:
    old_bundle = _write(tmp_path / "inbox/v5/bundles/old.tar.gz", b"old ingested")
    recent_bundle = _write(tmp_path / "inbox/v5/bundles/recent.tar.gz", b"recent ingested")
    untracked_bundle = _write(tmp_path / "inbox/v5/bundles/untracked.tar.gz", b"untracked")
    _set_mtime(old_bundle, datetime(2026, 9, 8, tzinfo=UTC))
    _set_mtime(recent_bundle, datetime(2026, 9, 13, tzinfo=UTC))
    _set_mtime(untracked_bundle, datetime(2026, 9, 8, tzinfo=UTC))
    old_sha = _sha256(old_bundle.read_bytes())
    recent_sha = _sha256(recent_bundle.read_bytes())
    old_day = (
        tmp_path / "archive_restricted/v5/bundles/2026-09-05" / old_sha
    )
    recent_day = (
        tmp_path / "archive_restricted/v5/bundles/2026-09-12" / recent_sha
    )
    _write(old_day / "raw_bundle.tar.gz", b"old ingested")
    _write(recent_day / "raw_bundle.tar.gz", b"recent ingested")
    _write_manifest(tmp_path, [old_sha, recent_sha])

    result = prune_v5_telemetry_storage(
        tmp_path,
        keep_restricted_archive_days=7,
        keep_inbox_days=2,
        dry_run=False,
        now=datetime(2026, 9, 13, 12, tzinfo=UTC),
    )

    assert result.ok is True
    assert not old_bundle.exists()
    assert recent_bundle.exists()
    assert untracked_bundle.exists()
    assert not old_day.exists()
    assert recent_day.exists()


def test_retention_fails_closed_without_ingest_manifest(tmp_path: Path) -> None:
    old_bundle = _write(tmp_path / "inbox/v5/bundles/old.tar.gz", b"old")
    _set_mtime(old_bundle, datetime(2026, 9, 1, tzinfo=UTC))

    result = prune_v5_telemetry_storage(
        tmp_path,
        dry_run=False,
        now=datetime(2026, 9, 13, tzinfo=UTC),
    )

    assert result.ok is False
    assert any("bundle_manifest_unavailable" in error for error in result.errors)
    assert result.removed_paths == []
    assert old_bundle.exists()


def test_retention_preserves_restricted_day_with_unexpected_entry(tmp_path: Path) -> None:
    bundle_hash = _sha256(b"known")
    old_day = tmp_path / "archive_restricted/v5/bundles/2026-09-05"
    _write(old_day / bundle_hash / "raw_bundle.tar.gz", b"known")
    _write(old_day / "unexpected.txt", b"unknown evidence")
    _write_manifest(tmp_path, [bundle_hash])

    result = prune_v5_telemetry_storage(
        tmp_path,
        dry_run=False,
        now=datetime(2026, 9, 13, tzinfo=UTC),
    )

    assert result.ok is True
    assert result.restricted_archive_removed_days == 0
    assert old_day.exists()
    assert any("restricted_day_unexpected_entry" in warning for warning in result.warnings)


def test_retention_report_is_atomic_and_bounded(tmp_path: Path) -> None:
    bundle_hash = _sha256(b"known")
    _write_manifest(tmp_path, [bundle_hash])
    result = prune_v5_telemetry_storage(
        tmp_path,
        dry_run=True,
        now=datetime(2026, 9, 13, tzinfo=UTC),
    )
    output = tmp_path / "ops/v5_telemetry_retention/latest.json"

    write_v5_telemetry_retention_report(output, result, max_paths_reported=1)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "quant_lab.v5_telemetry_retention.v1"
    assert payload["ok"] is True
    assert not output.with_name(f".{output.name}.tmp").exists()


def _write_manifest(root: Path, hashes: list[str]) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            {
                "bundle_sha256": hashes,
                "bundle_name": [f"bundle-{index}.tar.gz" for index, _ in enumerate(hashes)],
            }
        ),
        root / "lake/bronze/strategy_telemetry/v5/bundle_manifest",
    )


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _set_mtime(path: Path, value: datetime) -> None:
    timestamp = value.timestamp()
    os.utime(path, (timestamp, timestamp))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
