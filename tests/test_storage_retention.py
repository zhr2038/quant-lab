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
        keep_restricted_archive_days=7,
        keep_inbox_days=2,
        dry_run=True,
        now=datetime(2026, 5, 22, tzinfo=UTC),
    )

    assert result.redacted_archive_removed_days == 1
    assert result.restricted_archive_removed_days == 0
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
    old_restricted_raw = _write_file(
        tmp_path / "archive_restricted" / "v5" / "bundles" / "2026-05-14" / "sha" / "raw.tar.gz",
        "old restricted raw",
    )
    kept_restricted_raw = _write_file(
        tmp_path / "archive_restricted" / "v5" / "bundles" / "2026-05-18" / "sha" / "raw.tar.gz",
        "kept restricted raw",
    )
    old_new_layout_restricted_raw = _write_file(
        tmp_path / "archive_restricted" / "v5" / "2026-05-14" / "sha" / "raw.tar.gz",
        "old restricted raw new layout",
    )
    old_new_layout_redacted = _write_file(
        tmp_path / "archive" / "v5" / "2026-05-18" / "sha" / "file.txt",
        "old redacted new layout",
    )
    old_high_frequency = _write_file(
        tmp_path
        / "lake"
        / "archive"
        / "high_frequency"
        / "bronze"
        / "okx_public_ws"
        / "date=2026-05-18"
        / "hour=00"
        / "symbol=BTC-USDT"
        / "old.parquet",
        "old raw ws",
    )
    kept_high_frequency = _write_file(
        tmp_path
        / "lake"
        / "archive"
        / "high_frequency"
        / "bronze"
        / "okx_public_ws"
        / "date=2026-05-21"
        / "hour=00"
        / "symbol=BTC-USDT"
        / "kept.parquet",
        "kept raw ws",
    )
    old_silver_trade = _write_file(
        tmp_path
        / "lake"
        / "archive"
        / "high_frequency"
        / "silver"
        / "trade_print"
        / "date=2026-05-18"
        / "hour=00"
        / "old.parquet",
        "old silver trade",
    )
    kept_silver_trade = _write_file(
        tmp_path
        / "lake"
        / "archive"
        / "high_frequency"
        / "silver"
        / "trade_print"
        / "date=2026-05-21"
        / "hour=00"
        / "kept.parquet",
        "kept silver trade",
    )
    old_silver_book = _write_file(
        tmp_path
        / "lake"
        / "archive"
        / "high_frequency"
        / "silver"
        / "orderbook_snapshot"
        / "date=2026-05-18"
        / "hour=00"
        / "old.parquet",
        "old silver book",
    )
    kept_silver_book = _write_file(
        tmp_path
        / "lake"
        / "archive"
        / "high_frequency"
        / "silver"
        / "orderbook_snapshot"
        / "date=2026-05-21"
        / "hour=00"
        / "kept.parquet",
        "kept silver book",
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
        keep_restricted_archive_days=7,
        keep_high_frequency_archive_days=3,
        keep_inbox_days=2,
        keep_export_packs=5,
        dry_run=False,
        now=datetime(2026, 5, 22, tzinfo=UTC),
    )

    assert result.redacted_archive_removed_days == 2
    assert result.restricted_archive_removed_days == 2
    assert result.high_frequency_archive_removed_days == 3
    assert result.inbox_removed_files == 1
    assert result.export_removed_files == 2
    assert result.maintenance_removed_dirs == 1
    assert not old_redacted.exists()
    assert not old_new_layout_redacted.exists()
    assert kept_redacted.exists()
    assert not old_restricted_raw.exists()
    assert kept_restricted_raw.exists()
    assert not old_new_layout_restricted_raw.exists()
    assert not old_high_frequency.exists()
    assert kept_high_frequency.exists()
    assert not old_silver_trade.exists()
    assert kept_silver_trade.exists()
    assert not old_silver_book.exists()
    assert kept_silver_book.exists()
    assert not old_inbox.exists()
    assert kept_inbox.exists()
    assert not smoke_dir.exists()
    assert sum(pack.exists() for pack in exports) == 5


def test_storage_retention_payload_can_truncate_removed_paths(tmp_path: Path) -> None:
    old_one = _write_file(tmp_path / "inbox" / "v5" / "bundles" / "old-1.tar.gz", "old")
    old_two = _write_file(tmp_path / "inbox" / "v5" / "bundles" / "old-2.tar.gz", "old")
    _set_mtime(old_one, datetime(2026, 5, 18, tzinfo=UTC))
    _set_mtime(old_two, datetime(2026, 5, 18, tzinfo=UTC))

    result = prune_quant_lab_storage(
        tmp_path,
        keep_inbox_days=2,
        dry_run=True,
        now=datetime(2026, 5, 22, tzinfo=UTC),
    )
    payload = result.to_dict(max_removed_paths_reported=1)

    assert payload["removed_path_count"] == 2
    assert len(payload["removed_paths"]) == 1
    assert payload["removed_paths_truncated"] is True


def _write_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _set_mtime(path: Path, timestamp: datetime) -> None:
    seconds = timestamp.timestamp()
    os.utime(path, (seconds, seconds))
