from __future__ import annotations

import hashlib
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "deploy"
    / "nas_archive"
    / "prune_verified_high_frequency_archive.py"
)
SPEC = importlib.util.spec_from_file_location("prune_verified_high_frequency_archive", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _archive_day(tmp_path: Path, dataset: str, day: str) -> Path:
    target = tmp_path / dataset / f"date={day}" / "hour=00"
    target.mkdir(parents=True)
    (target / "a.parquet").write_bytes(b"alpha")
    (target / "b.parquet").write_bytes(b"beta")
    return target.parent


def _manifest_sha(day_root: Path) -> str:
    manifest, _, _ = MODULE.canonical_manifest_bytes(day_root)
    return hashlib.sha256(manifest).hexdigest()


def test_verified_high_frequency_prune_dry_run_preserves_source(tmp_path: Path) -> None:
    dataset = "silver/orderbook_snapshot"
    day = "2026-08-19"
    target = _archive_day(tmp_path, dataset, day)

    result = MODULE.prune_verified_archive_day(
        dataset=dataset,
        day=day,
        expected_manifest_sha256=_manifest_sha(target),
        apply=False,
        archive_root=tmp_path,
        audit_path=tmp_path / "audit.jsonl",
        lock_path=None,
        now=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert result["removed"] is False
    assert result["file_count"] == 2
    assert target.exists()
    assert not (tmp_path / "audit.jsonl").exists()


def test_verified_high_frequency_prune_apply_requires_exact_manifest(tmp_path: Path) -> None:
    dataset = "silver/trade_print"
    day = "2026-08-19"
    target = _archive_day(tmp_path, dataset, day)

    with pytest.raises(MODULE.ArchivePruneError, match="manifest_sha256_mismatch"):
        MODULE.prune_verified_archive_day(
            dataset=dataset,
            day=day,
            expected_manifest_sha256="0" * 64,
            apply=True,
            archive_root=tmp_path,
            audit_path=tmp_path / "audit.jsonl",
            lock_path=None,
            now=datetime(2026, 8, 21, tzinfo=UTC),
        )

    assert target.exists()


def test_verified_high_frequency_prune_requires_writable_audit_before_remove(
    tmp_path: Path,
) -> None:
    dataset = "silver/trade_print"
    day = "2026-08-19"
    target = _archive_day(tmp_path, dataset, day)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block audit parent", encoding="utf-8")

    with pytest.raises(FileExistsError):
        MODULE.prune_verified_archive_day(
            dataset=dataset,
            day=day,
            expected_manifest_sha256=_manifest_sha(target),
            apply=True,
            archive_root=tmp_path,
            audit_path=blocker / "audit.jsonl",
            lock_path=None,
            now=datetime(2026, 8, 21, tzinfo=UTC),
        )

    assert target.exists()


def test_verified_high_frequency_prune_apply_removes_only_verified_day(tmp_path: Path) -> None:
    dataset = "bronze/okx_public_ws"
    day = "2026-08-19"
    target = _archive_day(tmp_path, dataset, day)
    keep = _archive_day(tmp_path, dataset, "2026-08-20")

    result = MODULE.prune_verified_archive_day(
        dataset=dataset,
        day=day,
        expected_manifest_sha256=_manifest_sha(target),
        apply=True,
        archive_root=tmp_path,
        audit_path=tmp_path / "audit.jsonl",
        lock_path=None,
        now=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert result["removed"] is True
    assert not target.exists()
    assert keep.exists()
    assert '"event":"verified_prune_prepared"' in (
        tmp_path / "audit.jsonl"
    ).read_text(encoding="utf-8")
    assert '"event":"source_pruned_after_verified_archive"' in (
        tmp_path / "audit.jsonl"
    ).read_text(encoding="utf-8")
    assert '"removed":true' in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("dataset", "day", "reason"),
    [
        ("silver/not_allowed", "2026-08-19", "dataset_not_allowed"),
        ("silver/trade_print", "../../etc", "invalid_day"),
        ("silver/trade_print", "2026-08-21", "date_not_before_today"),
    ],
)
def test_verified_high_frequency_prune_rejects_unsafe_targets(
    tmp_path: Path,
    dataset: str,
    day: str,
    reason: str,
) -> None:
    with pytest.raises(MODULE.ArchivePruneError, match=reason):
        MODULE.prune_verified_archive_day(
            dataset=dataset,
            day=day,
            expected_manifest_sha256="0" * 64,
            apply=False,
            archive_root=tmp_path,
            audit_path=None,
            lock_path=None,
            now=datetime(2026, 8, 21, tzinfo=UTC),
        )
