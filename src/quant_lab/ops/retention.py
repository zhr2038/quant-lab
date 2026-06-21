from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_BASE_DIR = Path("/var/lib/quant-lab")
DEFAULT_KEEP_REDACTED_ARCHIVE_DAYS = 3
DEFAULT_KEEP_RESTRICTED_ARCHIVE_DAYS = 7
DEFAULT_KEEP_INBOX_DAYS = 2
DEFAULT_KEEP_EXPORT_PACKS = 5


@dataclass
class RetentionPruneResult:
    base_dir: str
    dry_run: bool
    started_at: datetime
    finished_at: datetime | None = None
    removed_paths: list[str] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    removed_bytes: int = 0
    redacted_archive_removed_days: int = 0
    restricted_archive_removed_days: int = 0
    inbox_removed_files: int = 0
    export_removed_files: int = 0
    maintenance_removed_dirs: int = 0

    def to_dict(self, *, max_removed_paths_reported: int | None = None) -> dict[str, Any]:
        removed_paths = self.removed_paths
        removed_paths_truncated = False
        if max_removed_paths_reported is not None:
            limit = max(int(max_removed_paths_reported), 0)
            removed_paths_truncated = len(removed_paths) > limit
            removed_paths = removed_paths[:limit]
        return {
            "base_dir": self.base_dir,
            "dry_run": self.dry_run,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "removed_path_count": len(self.removed_paths),
            "removed_paths": removed_paths,
            "removed_paths_truncated": removed_paths_truncated,
            "skipped_paths": self.skipped_paths,
            "warnings": self.warnings,
            "removed_bytes": self.removed_bytes,
            "redacted_archive_removed_days": self.redacted_archive_removed_days,
            "restricted_archive_removed_days": self.restricted_archive_removed_days,
            "inbox_removed_files": self.inbox_removed_files,
            "export_removed_files": self.export_removed_files,
            "maintenance_removed_dirs": self.maintenance_removed_dirs,
        }


def prune_quant_lab_storage(
    base_dir: str | Path = DEFAULT_BASE_DIR,
    *,
    keep_redacted_archive_days: int = DEFAULT_KEEP_REDACTED_ARCHIVE_DAYS,
    keep_restricted_archive_days: int = DEFAULT_KEEP_RESTRICTED_ARCHIVE_DAYS,
    keep_inbox_days: int = DEFAULT_KEEP_INBOX_DAYS,
    keep_export_packs: int = DEFAULT_KEEP_EXPORT_PACKS,
    dry_run: bool = True,
    now: datetime | None = None,
) -> RetentionPruneResult:
    """Prune regenerable quant-lab storage while preserving lake and bounded raw audit."""
    current = _utc_now(now)
    root = Path(base_dir)
    result = RetentionPruneResult(
        base_dir=str(root),
        dry_run=dry_run,
        started_at=current,
    )
    if not root.exists():
        result.warnings.append(f"base_dir_missing:{root}")
        result.finished_at = _utc_now(now)
        return result

    _prune_redacted_archive_days(
        root,
        keep_days=keep_redacted_archive_days,
        dry_run=dry_run,
        now=current,
        result=result,
    )
    _prune_restricted_archive_days(
        root,
        keep_days=keep_restricted_archive_days,
        dry_run=dry_run,
        now=current,
        result=result,
    )
    _prune_inbox_bundles(
        root,
        keep_days=keep_inbox_days,
        dry_run=dry_run,
        now=current,
        result=result,
    )
    _prune_expert_exports(
        root,
        keep_count=keep_export_packs,
        dry_run=dry_run,
        result=result,
    )
    _prune_maintenance_smoke_dirs(root, dry_run=dry_run, result=result)
    result.finished_at = datetime.now(UTC)
    return result


def _prune_redacted_archive_days(
    root: Path,
    *,
    keep_days: int,
    dry_run: bool,
    now: datetime,
    result: RetentionPruneResult,
) -> None:
    if keep_days < 1:
        result.warnings.append("keep_redacted_archive_days_must_be_positive")
        return
    for archive_root in (root / "archive" / "v5" / "bundles", root / "archive" / "v5"):
        result.redacted_archive_removed_days += _prune_archive_day_dirs(
            archive_root,
            root,
            keep_days=keep_days,
            dry_run=dry_run,
            now=now,
            result=result,
        )


def _prune_restricted_archive_days(
    root: Path,
    *,
    keep_days: int,
    dry_run: bool,
    now: datetime,
    result: RetentionPruneResult,
) -> None:
    if keep_days < 1:
        result.warnings.append("keep_restricted_archive_days_must_be_positive")
        return
    for archive_root in (
        root / "archive_restricted" / "v5" / "bundles",
        root / "archive_restricted" / "v5",
    ):
        result.restricted_archive_removed_days += _prune_archive_day_dirs(
            archive_root,
            root,
            keep_days=keep_days,
            dry_run=dry_run,
            now=now,
            result=result,
        )


def _prune_archive_day_dirs(
    archive_root: Path,
    root: Path,
    *,
    keep_days: int,
    dry_run: bool,
    now: datetime,
    result: RetentionPruneResult,
) -> int:
    removed = 0
    cutoff = now.date() - timedelta(days=keep_days - 1)
    for day_dir in _safe_children(archive_root, result):
        parsed = _parse_day_dir(day_dir)
        if parsed is None:
            if day_dir.name != "bundles":
                result.skipped_paths.append(str(day_dir))
            continue
        if parsed < cutoff and _remove_path(day_dir, root, dry_run=dry_run, result=result):
            removed += 1
    return removed


def _prune_inbox_bundles(
    root: Path,
    *,
    keep_days: int,
    dry_run: bool,
    now: datetime,
    result: RetentionPruneResult,
) -> None:
    if keep_days < 1:
        result.warnings.append("keep_inbox_days_must_be_positive")
        return
    inbox_root = root / "inbox" / "v5" / "bundles"
    cutoff = now - timedelta(days=keep_days)
    for bundle in sorted(inbox_root.glob("*.tar.gz")) if inbox_root.exists() else []:
        try:
            mtime = datetime.fromtimestamp(bundle.stat().st_mtime, UTC)
        except OSError as exc:
            result.warnings.append(f"stat_failed:{bundle}:{exc}")
            continue
        if mtime < cutoff and _remove_path(bundle, root, dry_run=dry_run, result=result):
            result.inbox_removed_files += 1


def _prune_expert_exports(
    root: Path,
    *,
    keep_count: int,
    dry_run: bool,
    result: RetentionPruneResult,
) -> None:
    if keep_count < 1:
        result.warnings.append("keep_export_packs_must_be_positive")
        return
    exports_root = root / "exports"
    packs = (
        [item for item in exports_root.glob("*.zip") if item.is_file()]
        if exports_root.exists()
        else []
    )
    packs.sort(key=lambda path: _mtime(path), reverse=True)
    for pack in packs[keep_count:]:
        if _remove_path(pack, root, dry_run=dry_run, result=result):
            result.export_removed_files += 1


def _prune_maintenance_smoke_dirs(
    root: Path,
    *,
    dry_run: bool,
    result: RetentionPruneResult,
) -> None:
    maintenance_root = root / "maintenance"
    if not maintenance_root.exists():
        return
    for item in maintenance_root.glob("compaction_smoke*"):
        if item.is_dir() and _remove_path(item, root, dry_run=dry_run, result=result):
            result.maintenance_removed_dirs += 1


def _safe_children(path: Path, result: RetentionPruneResult) -> list[Path]:
    if not path.exists():
        return []
    try:
        return sorted(item for item in path.iterdir() if item.is_dir())
    except OSError as exc:
        result.warnings.append(f"list_failed:{path}:{exc}")
        return []


def _parse_day_dir(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.name)
    except ValueError:
        return None


def _remove_path(
    path: Path,
    root: Path,
    *,
    dry_run: bool,
    result: RetentionPruneResult,
) -> bool:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not _is_relative_to(resolved_path, resolved_root):
        result.warnings.append(f"refused_outside_base_dir:{path}")
        return False
    size = _path_size(path)
    result.removed_paths.append(str(path))
    result.removed_bytes += size
    if dry_run:
        return True
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        result.warnings.append(f"remove_failed:{path}:{exc}")
        return False
    return True


def _path_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    if path.is_dir():
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
    return total


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)
