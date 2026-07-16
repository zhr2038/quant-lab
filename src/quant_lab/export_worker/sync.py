from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from quant_lab.export_plane.contracts import ExportSnapshotManifest
from quant_lab.export_plane.signatures import sha256_file


@dataclass(frozen=True)
class SnapshotSyncResult:
    snapshot_root: Path
    cache_hits: int
    downloaded_files: int
    downloaded_bytes: int


BlobFetcher = Callable[[str, Path], None]


def sync_snapshot_blobs(
    manifest: ExportSnapshotManifest,
    *,
    data_root: str | Path,
    fetch_blob: BlobFetcher,
    min_free_disk_bytes: int,
    max_snapshot_bytes: int,
) -> SnapshotSyncResult:
    root = Path(data_root)
    if manifest.total_input_bytes > max_snapshot_bytes:
        raise RuntimeError("snapshot_input_limit_exceeded")
    blobs_root = root / "blobs" / "sha256"
    snapshots_root = root / "snapshots"
    blobs_root.mkdir(parents=True, exist_ok=True)
    snapshots_root.mkdir(parents=True, exist_ok=True)
    final_snapshot = snapshots_root / manifest.snapshot_id
    if (final_snapshot / "SEALED").is_file():
        _verify_local_snapshot(final_snapshot, manifest)
        return SnapshotSyncResult(
            snapshot_root=final_snapshot,
            cache_hits=len(manifest.files),
            downloaded_files=0,
            downloaded_bytes=0,
        )

    missing_bytes = sum(
        reference.size_bytes
        for reference in manifest.files
        if not _valid_blob(_blob_path(blobs_root, reference.sha256), reference.sha256)
    )
    free_bytes = shutil.disk_usage(root).free
    if free_bytes - missing_bytes < min_free_disk_bytes:
        raise RuntimeError("insufficient_nas_disk_space")

    cache_hits = 0
    downloaded_files = 0
    downloaded_bytes = 0
    for reference in manifest.files:
        blob = _blob_path(blobs_root, reference.sha256)
        if _valid_blob(blob, reference.sha256, expected_size=reference.size_bytes):
            cache_hits += 1
            continue
        blob.parent.mkdir(parents=True, exist_ok=True)
        partial = blob.with_name(f".{blob.name}.partial")
        partial.unlink(missing_ok=True)
        fetch_blob(reference.relative_path, partial)
        if partial.stat().st_size != reference.size_bytes:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"blob_size_mismatch:{reference.relative_path}")
        if sha256_file(partial) != reference.sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"blob_sha256_mismatch:{reference.relative_path}")
        os.replace(partial, blob)
        blob.chmod(0o440)
        downloaded_files += 1
        downloaded_bytes += reference.size_bytes

    temporary = snapshots_root / f".{manifest.snapshot_id}.partial"
    shutil.rmtree(temporary, ignore_errors=True)
    files_root = temporary / "files"
    for reference in manifest.files:
        blob = _blob_path(blobs_root, reference.sha256)
        destination = files_root / reference.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(blob, destination)
        except OSError:
            shutil.copy2(blob, destination)
        destination.chmod(0o440)
    (temporary / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (temporary / "SEALED").write_text(manifest.manifest_sha256 + "\n", encoding="ascii")
    if final_snapshot.exists():
        _verify_local_snapshot(final_snapshot, manifest)
        shutil.rmtree(temporary, ignore_errors=True)
    else:
        os.replace(temporary, final_snapshot)
    return SnapshotSyncResult(
        snapshot_root=final_snapshot,
        cache_hits=cache_hits,
        downloaded_files=downloaded_files,
        downloaded_bytes=downloaded_bytes,
    )


def _blob_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / digest


def _valid_blob(path: Path, digest: str, *, expected_size: int | None = None) -> bool:
    try:
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        return path.is_file() and sha256_file(path) == digest
    except OSError:
        return False


def _verify_local_snapshot(path: Path, manifest: ExportSnapshotManifest) -> None:
    seal = (path / "SEALED").read_text(encoding="ascii").strip()
    if seal != manifest.manifest_sha256:
        raise RuntimeError("local_snapshot_manifest_mismatch")
    for reference in manifest.files:
        local = path / "files" / reference.relative_path
        if not _valid_blob(local, reference.sha256, expected_size=reference.size_bytes):
            raise RuntimeError(f"local_snapshot_blob_invalid:{reference.relative_path}")
