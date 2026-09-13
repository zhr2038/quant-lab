#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

_MANIFEST_LINE_RE = re.compile(r"([0-9a-f]{64})  \./(.+)")
_RECEIPT_SCHEMA = "quant_lab_nas_redacted_archive_receipt.v1"
_RECEIPT_FILES = {".archive_manifest.sha256", ".archive_receipt.json"}


def verify_redacted_archive_receipt(
    archive_root: Path,
    receipt_path: Path,
    manifest_path: Path,
    *,
    day: str,
    source_host: str,
    source_root: str,
) -> dict[str, object]:
    root = Path(archive_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("redacted archive root is not a regular directory")
    resolved_root = root.resolve()
    receipt_file = _regular_file_within(receipt_path, resolved_root)
    manifest_file = _regular_file_within(manifest_path, resolved_root)
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    manifest_bytes = manifest_file.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    manifest_entries: dict[str, str] = {}
    for raw_line in manifest_bytes.decode("utf-8").splitlines():
        match = _MANIFEST_LINE_RE.fullmatch(raw_line)
        if match is None:
            raise ValueError("invalid redacted archive manifest line")
        expected_sha256, relative = match.groups()
        if relative in manifest_entries:
            raise ValueError("duplicate redacted archive manifest path")
        target = _regular_file_within(root / relative, resolved_root)
        if _file_sha256(target) != expected_sha256:
            raise ValueError(f"redacted archive payload checksum mismatch: {relative}")
        manifest_entries[relative] = expected_sha256
    if not manifest_entries:
        raise ValueError("redacted archive manifest is empty")

    actual_payloads: set[str] = set()
    for target in root.rglob("*"):
        if target.is_symlink():
            raise ValueError(f"redacted archive payload contains a symlink: {target}")
        if not target.is_file():
            continue
        relative = target.relative_to(root).as_posix()
        if relative not in _RECEIPT_FILES:
            actual_payloads.add(relative)
    if actual_payloads != set(manifest_entries):
        raise ValueError("redacted archive payload set differs from manifest")

    expected_source = f"{source_host}:{source_root}"
    if receipt.get("schema_version") != _RECEIPT_SCHEMA:
        raise ValueError("unsupported redacted archive receipt schema")
    if receipt.get("day") != day:
        raise ValueError("redacted archive receipt day mismatch")
    if receipt.get("source") != expected_source:
        raise ValueError("redacted archive receipt source mismatch")
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise ValueError("redacted archive receipt manifest mismatch")
    if receipt.get("file_count") != len(manifest_entries):
        raise ValueError("redacted archive receipt file count mismatch")
    return {
        "schema_version": "quant_lab.redacted_v5_nas_verification.v1",
        "ok": True,
        "archive_root": str(resolved_root),
        "day": day,
        "source": expected_source,
        "manifest_sha256": manifest_sha256,
        "file_count": len(manifest_entries),
    }


def _regular_file_within(path: Path, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"archive path is not a regular file: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"archive path resolves outside archive root: {candidate}") from exc
    if resolved == root:
        raise ValueError(f"archive path resolves outside archive root: {candidate}")
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a NAS redacted V5 archive receipt against every payload file."
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--line-output", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = verify_redacted_archive_receipt(
        args.archive_root,
        args.receipt,
        args.manifest,
        day=args.day,
        source_host=args.source_host,
        source_root=args.source_root,
    )
    if args.line_output:
        print(result["manifest_sha256"])
        print(result["file_count"])
    else:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
