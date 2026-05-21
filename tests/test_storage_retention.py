from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from quant_lab.ops.retention import prune_quant_lab_storage


def test_storage_retention_dry_run_preserves_files(tmp_path: Path) -> None:
    old_archive = _write_file(
        tmp_path / "archive" / "v5" / "bundles" / "2026-05-18" / "sha" / "file.txt",
        "old redacted",
    )
    old_inbox = _write_file(tmp_path / "inbox" / "v5" / "bundles" / "old.tar.gz", "old")
    _set_mtime(old_inbox, datetime(2026, 5, 18, tzinfo=UTC))

    result = prune_quant_lab_storage(
        tmp_path,
        keep_redacted_archive_days=3,
        keep_inbox_days=2,
        dry_run=True,
        now=datetime(2026, 5, 22, tzinfo=UTC),
    )

    assert result.redacted_archive_removed_days == 1
    assert result.inbox_removed_files == 1
    assert old_archive.exists()
    assert old_inbox.exists()


def test_storage_retention_apply_deletes_only_regenerable_targets(tmp_path: Path) -> None:
    old_redacted = _write_file(
        tmp_path / "archive" / "v5" / "bundles" / "2026-05-18" / "sha" / "file.txt",
        "old redacted",
    )
    kept_redacted = _write_file(
        tmp_path / "archive" / "v5" / "bundles" / "2026-05-21" / "sha" / "file.txt",
        "kept redacted",
    )
    restricted_raw = _write_file(
        tmp_path / "archive_restricted" / "v5" / "bundles" / "2026-05-18" / "sha" / "raw.tar.gz",
        "restricted raw",
    )
    old_inbox = _write_file(tmp_path / "inbox" / "v5" / "bundles" / "old.tar.gz", "old")
    kept_inbox = _write_file(tmp_path / "inbox" / "v5" / "bundles" / "new.tar.gz", "new")
    _set_mtime(old_inbox, datetime(2026, 5, 18, tzinfo=UTC))
    _set_mtime(kept_inbox, datetime(2026, 5, 21, 12, tzinfo=UTC))
    smoke_dir = tmp_path / "maintenance" / "compaction_smoke_test"
    _write_file(smoke_dir / "tmp.parquet", "smoke")

    exports = []
    for index in range(7):
        pack = _write_file(tmp_path / "exports" / f"pack-{index}.zip", f"pack {index}")
        _set_mtime(pack, datetime(2026, 5, 15, tzinfo=UTC) + timedelta(minutes=index))
        exports.append(pack)

    result = prune_quant_lab_storage(
        tmp_path,
        keep_redacted_archive_days=3,
        keep_inbox_days=2,
        keep_export_packs=5,
        dry_run=False,
        now=datetime(2026, 5, 22, tzinfo=UTC),
    )

    assert result.redacted_archive_removed_days == 1
    assert result.inbox_removed_files == 1
    assert result.export_removed_files == 2
    assert result.maintenance_removed_dirs == 1
    assert not old_redacted.exists()
    assert kept_redacted.exists()
    assert restricted_raw.exists()
    assert not old_inbox.exists()
    assert kept_inbox.exists()
    assert not smoke_dir.exists()
    assert sum(pack.exists() for pack in exports) == 5


def _write_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _set_mtime(path: Path, timestamp: datetime) -> None:
    seconds = timestamp.timestamp()
    os.utime(path, (seconds, seconds))
