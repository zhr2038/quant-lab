#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

PRODUCTION_SOURCE_ROOT = Path("/var/lib/quant-lab/archive/v5/bundles")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceManifest:
    sha256: str
    file_count: int
    file_bytes: int


def build_source_manifest(day_root: Path) -> SourceManifest:
    root = Path(day_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"source day is not a regular directory: {root}")
    files: list[tuple[str, Path]] = []
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"source archive contains a symlink: {item}")
        if item.is_file():
            relative = item.relative_to(root).as_posix()
            files.append((relative, item))
    files.sort(key=lambda pair: pair[0].encode("utf-8"))
    if not files:
        raise ValueError(f"source day contains no files: {root}")

    manifest_digest = hashlib.sha256()
    file_bytes = 0
    for relative, item in files:
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        file_bytes += item.stat().st_size
        manifest_digest.update(f"{digest.hexdigest()}  ./{relative}\n".encode())
    return SourceManifest(
        sha256=manifest_digest.hexdigest(),
        file_count=len(files),
        file_bytes=file_bytes,
    )


def prune_verified_redacted_day(
    source_root: Path,
    day: str,
    *,
    expected_manifest_sha256: str,
    expected_file_count: int,
    apply: bool = False,
    today: date | None = None,
) -> dict[str, object]:
    if not _DAY_RE.fullmatch(day):
        raise ValueError("day must use YYYY-MM-DD")
    parsed_day = date.fromisoformat(day)
    current_day = today or datetime.now(UTC).date()
    if parsed_day >= current_day:
        raise ValueError("current or future archive day cannot be pruned")
    expected_hash = expected_manifest_sha256.lower()
    if not _SHA256_RE.fullmatch(expected_hash):
        raise ValueError("expected manifest sha256 is invalid")
    if expected_file_count < 1:
        raise ValueError("expected file count must be positive")

    root = Path(source_root).resolve()
    day_root = (root / day).resolve()
    if not day_root.is_relative_to(root) or day_root == root:
        raise ValueError("source day resolves outside the configured archive root")
    observed = build_source_manifest(day_root)
    if observed.sha256 != expected_hash:
        raise ValueError(
            f"source manifest sha256 mismatch: expected={expected_hash} observed={observed.sha256}"
        )
    if observed.file_count != expected_file_count:
        raise ValueError(
            "source file count mismatch: "
            f"expected={expected_file_count} observed={observed.file_count}"
        )

    if apply:
        shutil.rmtree(day_root)
        if day_root.exists():
            raise OSError(f"source archive day still exists after prune: {day_root}")
    return {
        "schema_version": "quant_lab.redacted_v5_source_prune.v1",
        "ok": True,
        "dry_run": not apply,
        "source_root": str(root),
        "day": day,
        "verified_manifest_sha256": observed.sha256,
        "verified_file_count": observed.file_count,
        "removed_bytes": observed.file_bytes if apply else 0,
        "eligible_bytes": observed.file_bytes,
        "removed": apply,
        "finished_at": datetime.now(UTC).isoformat(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune one redacted V5 source day after an exact NAS manifest match."
    )
    parser.add_argument("--source-root", default=str(PRODUCTION_SOURCE_ROOT))
    parser.add_argument("--day", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-file-count", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    source_root = Path(args.source_root).resolve()
    if source_root != PRODUCTION_SOURCE_ROOT.resolve():
        raise SystemExit(f"refusing non-production source root: {source_root}")
    result = prune_verified_redacted_day(
        source_root,
        args.day,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_file_count=args.expected_file_count,
        apply=args.apply,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
