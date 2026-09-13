from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from quant_lab.data.lake import read_parquet_dataset

DEFAULT_BASE_DIR = Path("/var/lib/quant-lab")
DEFAULT_KEEP_RESTRICTED_ARCHIVE_DAYS = 7
DEFAULT_KEEP_INBOX_DAYS = 2
_BUNDLE_MANIFEST = Path("lake/bronze/strategy_telemetry/v5/bundle_manifest")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class V5TelemetryRetentionResult:
    base_dir: str
    dry_run: bool
    started_at: datetime
    finished_at: datetime | None = None
    ingested_bundle_count: int = 0
    removed_paths: list[str] = field(default_factory=list)
    removed_bytes: int = 0
    restricted_archive_removed_days: int = 0
    inbox_removed_files: int = 0
    preserved_uningested_paths: list[str] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self, *, max_paths_reported: int = 50) -> dict[str, Any]:
        limit = max(int(max_paths_reported), 0)
        return {
            "schema_version": "quant_lab.v5_telemetry_retention.v1",
            "ok": self.ok,
            "base_dir": self.base_dir,
            "dry_run": self.dry_run,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "ingested_bundle_count": self.ingested_bundle_count,
            "removed_path_count": len(self.removed_paths),
            "removed_paths": self.removed_paths[:limit],
            "removed_paths_truncated": len(self.removed_paths) > limit,
            "removed_bytes": self.removed_bytes,
            "restricted_archive_removed_days": self.restricted_archive_removed_days,
            "inbox_removed_files": self.inbox_removed_files,
            "preserved_uningested_path_count": len(self.preserved_uningested_paths),
            "preserved_uningested_paths": self.preserved_uningested_paths[:limit],
            "preserved_uningested_paths_truncated": (
                len(self.preserved_uningested_paths) > limit
            ),
            "skipped_path_count": len(self.skipped_paths),
            "skipped_paths": self.skipped_paths[:limit],
            "skipped_paths_truncated": len(self.skipped_paths) > limit,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def prune_v5_telemetry_storage(
    base_dir: str | Path = DEFAULT_BASE_DIR,
    *,
    keep_restricted_archive_days: int = DEFAULT_KEEP_RESTRICTED_ARCHIVE_DAYS,
    keep_inbox_days: int = DEFAULT_KEEP_INBOX_DAYS,
    dry_run: bool = True,
    now: datetime | None = None,
) -> V5TelemetryRetentionResult:
    """Bound V5 raw/inbox copies after proving each bundle was ingested.

    The redacted archive is deliberately excluded. Its removal is coordinated by
    the NAS archive job only after a checksum-verified long-term copy exists.
    """
    current = _utc_now(now)
    root = Path(base_dir)
    result = V5TelemetryRetentionResult(
        base_dir=str(root),
        dry_run=dry_run,
        started_at=current,
    )
    if keep_restricted_archive_days < 1:
        result.errors.append("keep_restricted_archive_days_must_be_positive")
    if keep_inbox_days < 1:
        result.errors.append("keep_inbox_days_must_be_positive")
    if not root.is_dir():
        result.errors.append(f"base_dir_missing:{root}")
    if result.errors:
        result.finished_at = datetime.now(UTC)
        return result

    ingested = _load_ingested_bundle_sha256s(root, result)
    if ingested is None:
        result.finished_at = datetime.now(UTC)
        return result
    result.ingested_bundle_count = len(ingested)

    _prune_ingested_inbox_files(
        root,
        ingested=ingested,
        keep_days=keep_inbox_days,
        dry_run=dry_run,
        now=current,
        result=result,
    )
    _prune_ingested_restricted_days(
        root,
        ingested=ingested,
        keep_days=keep_restricted_archive_days,
        dry_run=dry_run,
        now=current,
        result=result,
    )
    result.finished_at = datetime.now(UTC)
    return result


def write_v5_telemetry_retention_report(
    path: str | Path,
    result: V5TelemetryRetentionResult,
    *,
    max_paths_reported: int = 50,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(
            result.to_dict(max_paths_reported=max_paths_reported),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def prepare_v5_telemetry_retention_report(path: str | Path) -> None:
    """Fail before pruning when the configured report destination is not writable."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".preflight",
    ) as probe:
        probe.write("{}\n")
        probe.flush()


def _load_ingested_bundle_sha256s(
    root: Path,
    result: V5TelemetryRetentionResult,
) -> set[str] | None:
    manifest = root / _BUNDLE_MANIFEST
    try:
        frame = read_parquet_dataset(manifest)
    except Exception as exc:
        result.errors.append(f"bundle_manifest_read_failed:{type(exc).__name__}:{exc}")
        return None
    if frame.is_empty() or "bundle_sha256" not in frame.columns:
        result.errors.append(f"bundle_manifest_unavailable:{manifest}")
        return None
    hashes = {
        str(value).lower()
        for value in frame["bundle_sha256"].to_list()
        if value and _SHA256_RE.fullmatch(str(value).lower())
    }
    if not hashes:
        result.errors.append(f"bundle_manifest_has_no_valid_sha256:{manifest}")
        return None
    return hashes


def _prune_ingested_inbox_files(
    root: Path,
    *,
    ingested: set[str],
    keep_days: int,
    dry_run: bool,
    now: datetime,
    result: V5TelemetryRetentionResult,
) -> None:
    inbox = root / "inbox" / "v5" / "bundles"
    cutoff = now - timedelta(days=keep_days)
    for bundle in sorted(inbox.glob("*.tar.gz")) if inbox.is_dir() else []:
        if bundle.is_symlink() or not bundle.is_file():
            result.skipped_paths.append(str(bundle))
            result.warnings.append(f"inbox_path_not_regular_file:{bundle}")
            continue
        try:
            modified = datetime.fromtimestamp(bundle.stat().st_mtime, UTC)
        except OSError as exc:
            result.warnings.append(f"inbox_stat_failed:{bundle}:{type(exc).__name__}:{exc}")
            continue
        if modified >= cutoff:
            continue
        try:
            bundle_sha256 = _file_sha256(bundle)
        except OSError as exc:
            result.warnings.append(f"inbox_hash_failed:{bundle}:{type(exc).__name__}:{exc}")
            continue
        if bundle_sha256 not in ingested:
            result.preserved_uningested_paths.append(str(bundle))
            result.warnings.append(f"preserved_uningested_inbox_bundle:{bundle.name}")
            continue
        if _remove_path(bundle, root, dry_run=dry_run, result=result):
            result.inbox_removed_files += 1


def _prune_ingested_restricted_days(
    root: Path,
    *,
    ingested: set[str],
    keep_days: int,
    dry_run: bool,
    now: datetime,
    result: V5TelemetryRetentionResult,
) -> None:
    cutoff = now.date() - timedelta(days=keep_days - 1)
    for archive_root in (
        root / "archive_restricted" / "v5" / "bundles",
        root / "archive_restricted" / "v5",
    ):
        for day_dir in _safe_directories(archive_root, result):
            day = _parse_iso_day(day_dir.name)
            if day is None:
                if day_dir.name != "bundles":
                    result.skipped_paths.append(str(day_dir))
                continue
            if day >= cutoff:
                continue
            if not _restricted_day_is_fully_ingested(day_dir, ingested, result):
                continue
            if _remove_path(day_dir, root, dry_run=dry_run, result=result):
                result.restricted_archive_removed_days += 1


def _restricted_day_is_fully_ingested(
    day_dir: Path,
    ingested: set[str],
    result: V5TelemetryRetentionResult,
) -> bool:
    try:
        entries = sorted(day_dir.iterdir())
    except OSError as exc:
        result.warnings.append(f"restricted_day_list_failed:{day_dir}:{type(exc).__name__}:{exc}")
        return False
    if not entries:
        result.skipped_paths.append(str(day_dir))
        result.warnings.append(f"restricted_day_empty:{day_dir}")
        return False
    for bundle_dir in entries:
        if (
            bundle_dir.is_symlink()
            or not bundle_dir.is_dir()
            or not _SHA256_RE.fullmatch(bundle_dir.name)
        ):
            result.skipped_paths.append(str(bundle_dir))
            result.warnings.append(f"restricted_day_unexpected_entry:{bundle_dir}")
            return False
        raw_bundle = bundle_dir / "raw_bundle.tar.gz"
        if raw_bundle.is_symlink() or not raw_bundle.is_file():
            result.skipped_paths.append(str(bundle_dir))
            result.warnings.append(f"restricted_bundle_missing_raw_copy:{bundle_dir}")
            return False
        if bundle_dir.name not in ingested:
            result.preserved_uningested_paths.append(str(bundle_dir))
            result.warnings.append(f"preserved_uningested_restricted_bundle:{bundle_dir.name}")
            return False
    return True


def _safe_directories(path: Path, result: V5TelemetryRetentionResult) -> list[Path]:
    if not path.is_dir():
        return []
    try:
        return sorted(item for item in path.iterdir() if item.is_dir())
    except OSError as exc:
        result.warnings.append(f"directory_list_failed:{path}:{type(exc).__name__}:{exc}")
        return []


def _remove_path(
    path: Path,
    root: Path,
    *,
    dry_run: bool,
    result: V5TelemetryRetentionResult,
) -> bool:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root) or resolved_path == resolved_root:
        result.errors.append(f"refused_outside_base_dir:{path}")
        return False
    size = _path_size(path)
    if not dry_run:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            result.warnings.append(f"remove_failed:{path}:{type(exc).__name__}:{exc}")
            return False
    result.removed_paths.append(str(path))
    result.removed_bytes += size
    return True


def _path_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*") if path.is_dir() else []:
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_iso_day(value: str) -> date | None:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _utc_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
