#!/usr/bin/env python3
"""Remove one cloud high-frequency archive day after an exact NAS copy.

The command is intentionally narrow.  It can only remove a completed UTC day
from one of the three high-frequency archive datasets, and only when the
caller's manifest hash exactly matches the files that are still on qyun2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

ARCHIVE_ROOT = Path("/var/lib/quant-lab/lake/archive/high_frequency")
AUDIT_PATH = ARCHIVE_ROOT / ".nas_high_frequency_prune.jsonl"
HEAVY_LOCK_PATH = Path("/var/lock/quant-lab-heavy.lock")
ALLOWED_DATASETS = {
    "bronze/okx_public_ws",
    "silver/orderbook_snapshot",
    "silver/trade_print",
}
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArchivePruneError(RuntimeError):
    """Fail-closed archive pruning error."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_bytes(day_root: Path) -> tuple[bytes, int, int]:
    for path in day_root.rglob("*"):
        if path.is_symlink():
            raise ArchivePruneError(f"archive_path_symlink:{path}")
    files = sorted(
        (path for path in day_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(day_root).as_posix(),
    )
    if not files:
        raise ArchivePruneError("archive_day_empty")

    lines: list[str] = []
    total_bytes = 0
    for path in files:
        relative = path.relative_to(day_root).as_posix()
        lines.append(f"{_sha256_file(path)}  ./{relative}\n")
        total_bytes += path.stat().st_size
    return "".join(lines).encode("utf-8"), len(files), total_bytes


@contextmanager
def _exclusive_lock(lock_path: Path | None, timeout_seconds: int) -> Iterator[None]:
    if lock_path is None:
        yield
        return
    if os.name != "posix":
        raise ArchivePruneError("posix_lock_required")

    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        deadline = time.monotonic() + max(0, int(timeout_seconds))
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise ArchivePruneError("heavy_lock_timeout") from exc
                time.sleep(0.25)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validated_target(
    archive_root: Path,
    dataset: str,
    day: str,
    *,
    now: datetime,
) -> Path:
    if dataset not in ALLOWED_DATASETS:
        raise ArchivePruneError(f"dataset_not_allowed:{dataset}")
    if not DAY_RE.fullmatch(day):
        raise ArchivePruneError(f"invalid_day:{day}")
    try:
        parsed_day = date.fromisoformat(day)
    except ValueError as exc:
        raise ArchivePruneError(f"invalid_day:{day}") from exc
    if parsed_day >= now.astimezone(UTC).date():
        raise ArchivePruneError(f"date_not_before_today:{day}")

    root = archive_root.resolve(strict=True)
    lexical_target = archive_root / Path(dataset) / f"date={day}"
    if lexical_target.is_symlink():
        raise ArchivePruneError(f"archive_day_symlink:{lexical_target}")
    target = lexical_target.resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ArchivePruneError(f"unsafe_archive_target:{target}") from exc
    if not target.is_dir():
        raise ArchivePruneError(f"archive_day_not_directory:{target}")
    return target


def prune_verified_archive_day(
    *,
    dataset: str,
    day: str,
    expected_manifest_sha256: str,
    apply: bool,
    archive_root: Path = ARCHIVE_ROOT,
    audit_path: Path | None = AUDIT_PATH,
    lock_path: Path | None = HEAVY_LOCK_PATH,
    lock_timeout_seconds: int = 600,
    now: datetime | None = None,
) -> dict[str, object]:
    expected = str(expected_manifest_sha256 or "").strip().lower()
    if not SHA256_RE.fullmatch(expected):
        raise ArchivePruneError("invalid_manifest_sha256")
    current = now or datetime.now(UTC)

    with _exclusive_lock(lock_path, lock_timeout_seconds):
        target = _validated_target(archive_root, dataset, day, now=current)
        manifest, file_count, byte_count = canonical_manifest_bytes(target)
        actual = hashlib.sha256(manifest).hexdigest()
        if actual != expected:
            raise ArchivePruneError(
                f"manifest_sha256_mismatch:expected={expected}:actual={actual}"
            )

        result: dict[str, object] = {
            "schema_version": "quant_lab_verified_high_frequency_prune.v1",
            "dataset": dataset,
            "day": day,
            "manifest_sha256": actual,
            "file_count": file_count,
            "byte_count": byte_count,
            "apply": bool(apply),
            "removed": False,
            "target": str(target),
        }
        if not apply:
            return result

        tombstone = target.with_name(
            f".{target.name}.pruning-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}"
        )
        if tombstone.exists():
            raise ArchivePruneError(f"tombstone_exists:{tombstone}")
        audit_handle = None
        try:
            if audit_path is not None:
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_handle = audit_path.open("a", encoding="utf-8")
                prepared = {
                    **result,
                    "event": "verified_prune_prepared",
                    "prepared_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                }
                audit_handle.write(
                    json.dumps(prepared, sort_keys=True, separators=(",", ":")) + "\n"
                )
                audit_handle.flush()
                os.fsync(audit_handle.fileno())

            target.rename(tombstone)
            try:
                shutil.rmtree(tombstone)
            except Exception:
                if tombstone.exists() and not target.exists():
                    tombstone.rename(target)
                raise

            result["event"] = "source_pruned_after_verified_archive"
            result["removed"] = True
            result["removed_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            if audit_handle is not None:
                audit_handle.write(
                    json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
                )
                audit_handle.flush()
                os.fsync(audit_handle.fileno())
        finally:
            if audit_handle is not None:
                audit_handle.close()
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(ALLOWED_DATASETS))
    parser.add_argument("day")
    parser.add_argument("manifest_sha256")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--lock-timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    try:
        result = prune_verified_archive_day(
            dataset=args.dataset,
            day=args.day,
            expected_manifest_sha256=args.manifest_sha256,
            apply=args.apply,
            lock_timeout_seconds=args.lock_timeout_seconds,
        )
    except (ArchivePruneError, FileNotFoundError, OSError) as exc:
        print(
            json.dumps(
                {"status": "error", "reason": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
