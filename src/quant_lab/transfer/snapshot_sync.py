from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from quant_lab.export_plane.signatures import sha256_file


class SnapshotFileReference(Protocol):
    relative_path: str
    sha256: str
    size_bytes: int


class SnapshotManifest(Protocol):
    snapshot_id: str
    manifest_sha256: str
    total_input_bytes: int
    files: Sequence[SnapshotFileReference]

    def model_dump_json(self, *, indent: int | None = None) -> str: ...


ReferenceT = TypeVar("ReferenceT", bound=SnapshotFileReference)


@dataclass(frozen=True)
class SnapshotSyncResult:
    snapshot_root: Path
    cache_hits: int
    downloaded_files: int
    downloaded_bytes: int


BlobFetcher = Callable[[str, Path], None]
BlobBatchFetcher = Callable[[list[SnapshotFileReference], Path], None]


def sync_snapshot_blobs(
    manifest: SnapshotManifest | BaseModel,
    *,
    data_root: str | Path,
    fetch_blob: BlobFetcher,
    fetch_blobs: BlobBatchFetcher | None = None,
    batch_fetch_workers: int = 1,
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

    missing_by_sha: dict[str, SnapshotFileReference] = {}
    for reference in manifest.files:
        if not _valid_blob(
            _blob_path(blobs_root, reference.sha256),
            reference.sha256,
            expected_size=reference.size_bytes,
        ):
            existing = missing_by_sha.setdefault(reference.sha256, reference)
            if existing.size_bytes != reference.size_bytes:
                raise RuntimeError("snapshot_duplicate_sha_size_mismatch")
    missing = list(missing_by_sha.values())
    missing_bytes = sum(reference.size_bytes for reference in missing)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes - missing_bytes < min_free_disk_bytes:
        raise RuntimeError("insufficient_nas_disk_space")

    cache_hits = len(manifest.files) - len(missing)
    if fetch_blobs is not None and missing:
        incoming = root / "incoming" / f".{manifest.snapshot_id}.{os.getpid()}.partial"
        shutil.rmtree(incoming, ignore_errors=True)
        incoming.mkdir(parents=True, exist_ok=False)
        try:
            batches = _balanced_batches(missing, batch_fetch_workers)
            with ThreadPoolExecutor(max_workers=len(batches)) as executor:
                futures = [
                    executor.submit(
                        _fetch_and_install_batch,
                        batch,
                        incoming / f"batch-{index:02d}",
                        fetch_blobs,
                        blobs_root,
                    )
                    for index, batch in enumerate(batches)
                ]
                for future in futures:
                    future.result()
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
    else:
        for reference in missing:
            blob = _blob_path(blobs_root, reference.sha256)
            blob.parent.mkdir(parents=True, exist_ok=True)
            partial = blob.with_name(f".{blob.name}.partial")
            partial.unlink(missing_ok=True)
            fetch_blob(reference.relative_path, partial)
            _install_blob(partial, blob, reference)

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
        downloaded_files=len(missing),
        downloaded_bytes=missing_bytes,
    )


def _balanced_batches(
    references: list[ReferenceT],
    worker_count: int,
) -> list[list[ReferenceT]]:
    count = min(max(1, worker_count), len(references))
    batches: list[list[ReferenceT]] = [[] for _ in range(count)]
    batch_bytes = [0] * count
    for reference in sorted(
        references,
        key=lambda item: (-item.size_bytes, item.relative_path),
    ):
        index = min(range(count), key=lambda value: (batch_bytes[value], value))
        batches[index].append(reference)
        batch_bytes[index] += reference.size_bytes
    return [batch for batch in batches if batch]


def _fetch_and_install_batch(
    references: list[SnapshotFileReference],
    incoming: Path,
    fetch_blobs: BlobBatchFetcher,
    blobs_root: Path,
) -> None:
    incoming.mkdir(parents=True, exist_ok=False)
    try:
        fetch_blobs(references, incoming)
        for reference in references:
            _install_blob(
                incoming / reference.relative_path,
                _blob_path(blobs_root, reference.sha256),
                reference,
            )
    finally:
        shutil.rmtree(incoming, ignore_errors=True)


def _install_blob(
    source: Path,
    blob: Path,
    reference: SnapshotFileReference,
) -> None:
    try:
        if source.is_symlink():
            raise RuntimeError(f"blob_symlink_forbidden:{reference.relative_path}")
        if source.stat().st_size != reference.size_bytes:
            raise RuntimeError(f"blob_size_mismatch:{reference.relative_path}")
        if sha256_file(source) != reference.sha256:
            raise RuntimeError(f"blob_sha256_mismatch:{reference.relative_path}")
        blob.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, blob)
        blob.chmod(0o440)
    except FileNotFoundError as exc:
        raise RuntimeError(f"blob_missing:{reference.relative_path}") from exc
    except Exception:
        source.unlink(missing_ok=True)
        raise


def _blob_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / digest


def _valid_blob(path: Path, digest: str, *, expected_size: int | None = None) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        return sha256_file(path) == digest
    except OSError:
        return False


def _verify_local_snapshot(path: Path, manifest: SnapshotManifest | BaseModel) -> None:
    root = path.resolve(strict=True)
    seal = (root / "SEALED").read_text(encoding="ascii").strip()
    if seal != manifest.manifest_sha256:
        raise RuntimeError("local_snapshot_manifest_mismatch")
    for reference in manifest.files:
        candidate = root / "files" / reference.relative_path
        if candidate.is_symlink():
            raise RuntimeError(f"local_snapshot_path_escape:{reference.relative_path}")
        local = candidate.resolve(strict=True)
        if root not in local.parents:
            raise RuntimeError(f"local_snapshot_path_escape:{reference.relative_path}")
        if not _valid_blob(local, reference.sha256, expected_size=reference.size_bytes):
            raise RuntimeError(f"local_snapshot_blob_invalid:{reference.relative_path}")
