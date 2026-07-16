from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from quant_lab import __version__
from quant_lab.export import daily as daily_export
from quant_lab.export_plane.contracts import ExportDatasetReference, ExportSnapshotManifest
from quant_lab.export_plane.signatures import (
    canonical_json_bytes,
    load_signing_key,
    sha256_bytes,
    sha256_file,
    sign_payload,
)
from quant_lab.export_plane.status import atomic_write_json, ensure_queue_layout
from quant_lab.web import readers


def seal_export_snapshot(
    *,
    export_date: date,
    lake_root: str | Path,
    queue_root: str | Path,
    signing_key_path: str | Path,
    signature_key_id: str,
    acceptance_set_id: str | None = None,
    rehydrate_released: bool = False,
) -> tuple[ExportSnapshotManifest, Path]:
    root = Path(lake_root).resolve()
    queue = ensure_queue_layout(queue_root)
    if not root.is_dir():
        raise FileNotFoundError(f"lake root does not exist: {root}")
    quant_commit = _git_commit()
    v5_context = daily_export._observe_v5_before_export(root)  # noqa: SLF001
    selected_v5 = Path(str(v5_context.get("selected_v5_bundle_path") or ""))
    selected_sha = str(v5_context.get("selected_v5_bundle_sha256") or "").lower()
    if not selected_v5.is_file() or len(selected_sha) != 64:
        raise RuntimeError("authoritative V5 bundle is not observable for snapshot sealing")
    v5_commit = daily_export._selected_v5_bundle_git_commit(v5_context)  # noqa: SLF001
    if len(v5_commit) != 40:
        raise RuntimeError("selected V5 bundle does not expose a full git commit")

    sources = list(_export_source_files(root))
    sources.append(("v5_bundle", selected_v5, _relative_source_path(root, selected_v5)))
    temporary = Path(tempfile.mkdtemp(prefix=".sealing.", dir=queue / "snapshots"))
    try:
        files_root = temporary / "files"
        references: list[ExportDatasetReference] = []
        for dataset, source, relative_path in sources:
            destination = files_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            references.append(
                _copy_stable_reference(dataset, source, destination, relative_path)
            )
        references.sort(key=lambda item: item.relative_path)
        v5_reference = next(item for item in references if item.dataset == "v5_bundle")
        if v5_reference.sha256 != selected_sha:
            raise RuntimeError("selected V5 bundle changed while sealing")

        source_digest = sha256_bytes(
            json.dumps(
                [item.model_dump(mode="json") for item in references],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        snapshot_id = f"export-snapshot-{source_digest[:24]}"
        acceptance_id = acceptance_set_id or (
            f"nas-export-{export_date:%Y%m%d}-{source_digest[:16]}"
        )
        final_dir = queue / "snapshots" / snapshot_id
        if final_dir.exists():
            existing = ExportSnapshotManifest.model_validate_json(
                (final_dir / "manifest.json").read_text(encoding="utf-8")
            )
            verify_snapshot_manifest_digest(existing)
            if existing.files != references:
                raise RuntimeError("snapshot_id already exists with different file references")
            if existing.acceptance_set_id != acceptance_id:
                raise RuntimeError("snapshot_id already exists with a different acceptance set")
            if rehydrate_released and not (final_dir / "files").is_dir():
                _set_snapshot_permissions(files_root)
                os.replace(files_root, final_dir / "files")
                (final_dir / "RELEASED.json").unlink(missing_ok=True)
            return existing, final_dir

        created_at = datetime.now(UTC)
        unsigned = {
            "schema_version": "quant_lab_export_snapshot.v1",
            "snapshot_id": snapshot_id,
            "export_date": export_date,
            "created_at": created_at,
            "quant_lab_commit": quant_commit,
            "quant_lab_version": __version__,
            "v5_commit": v5_commit,
            "selected_v5_bundle_name": selected_v5.name,
            "selected_v5_bundle_sha256": selected_sha,
            "acceptance_set_id": acceptance_id,
            "risk_permission_identity": _dataset_identity(root, "risk_permission"),
            "paper_lifecycle_identity": _dataset_identity(
                root,
                "paper_strategy_proposal_snapshot",
            ),
            "environment_fingerprint": _environment_fingerprint(),
            "schema_fingerprint": _schema_fingerprint(),
            "files": references,
            "total_input_bytes": sum(item.size_bytes for item in references),
            "authoritative_input_snapshot": True,
            "manifest_sha256": "0" * 64,
            "signature_key_id": signature_key_id,
            "signature_algorithm": "ed25519",
            "signature": "A" * 88,
        }
        manifest = _finalize_snapshot_manifest(unsigned, signing_key_path)
        atomic_write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
        (temporary / "SEALED").write_text(manifest.manifest_sha256 + "\n", encoding="ascii")
        _set_snapshot_permissions(temporary)
        os.replace(temporary, final_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return manifest, final_dir


def _finalize_snapshot_manifest(
    unsigned: dict[str, object],
    signing_key_path: str | Path,
) -> ExportSnapshotManifest:
    provisional = ExportSnapshotManifest.model_validate(unsigned)
    digest_payload = provisional.model_dump(mode="json")
    digest_payload.pop("signature")
    digest_payload.pop("manifest_sha256")
    with_digest = ExportSnapshotManifest.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "manifest_sha256": sha256_bytes(canonical_json_bytes(digest_payload)),
        }
    )
    manifest = ExportSnapshotManifest.model_validate(
        {
            **with_digest.model_dump(mode="json"),
            "signature": sign_payload(with_digest, load_signing_key(signing_key_path)),
        }
    )
    verify_snapshot_manifest_digest(manifest)
    return manifest


def verify_snapshot_manifest_digest(manifest: ExportSnapshotManifest) -> None:
    payload = manifest.model_dump(mode="json")
    expected = payload.pop("manifest_sha256")
    payload.pop("signature")
    actual = sha256_bytes(canonical_json_bytes(payload))
    if actual != expected:
        raise ValueError("snapshot manifest SHA256 mismatch")


def _export_source_files(lake_root: Path) -> Iterable[tuple[str, Path, str]]:
    base = lake_root.parent
    seen: set[Path] = set()
    for dataset in sorted(readers.DATASET_PATHS):
        path = readers.dataset_path_for(lake_root, dataset)
        files = _dataset_files(path, dataset)
        for item in files:
            resolved = item.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield dataset, resolved, _relative_to_base(base, resolved)


def _dataset_files(path: Path, dataset: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    if dataset in daily_export.HEAVY_EXPORT_DATASET_LIMITS:
        limit = daily_export.HEAVY_EXPORT_RECENT_FILE_LIMITS.get(
            dataset,
            daily_export.DEFAULT_EXPORT_RECENT_FILE_LIMIT,
        )
        return daily_export._recent_heavy_dataset_files(path, max_files=limit)  # noqa: SLF001
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in {".parquet", ".json", ".csv", ".yaml", ".yml"}
        and not item.name.startswith(".")
    )


def _copy_stable_reference(
    dataset: str,
    source: Path,
    destination: Path,
    relative_path: str,
    *,
    max_attempts: int = 5,
) -> ExportDatasetReference:
    last_error: Exception | None = None
    for attempt in range(max(1, max_attempts)):
        destination.unlink(missing_ok=True)
        try:
            captured = _copy_snapshot_input(source, destination)
            return ExportDatasetReference(
                relative_path=relative_path,
                sha256=sha256_file(destination),
                size_bytes=captured.st_size,
                mtime_ns=captured.st_mtime_ns,
                row_count=(
                    _parquet_rows(destination)
                    if source.suffix.lower() == ".parquet"
                    else None
                ),
                dataset=dataset,
                media_type=_media_type(source, dataset),
            )
        except (FileNotFoundError, RuntimeError) as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt + 1 < max_attempts:
                time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"snapshot input did not stabilize: {relative_path}") from last_error


def _parquet_rows(path: Path) -> int | None:
    try:
        return int(pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item())
    except Exception:
        return None


def _relative_to_base(base: Path, path: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError as exc:
        raise ValueError(f"export input is outside the quant-lab data root: {path}") from exc


def _relative_source_path(lake_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(lake_root.parent).as_posix()
    except ValueError:
        return f"inputs/v5/{path.name}"


def _copy_snapshot_input(source: Path, destination: Path) -> os.stat_result:
    with source.open("rb") as source_handle, destination.open("xb") as target_handle:
        before = os.fstat(source_handle.fileno())
        shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        target_handle.flush()
        os.fsync(target_handle.fileno())
        after = os.fstat(source_handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"snapshot input changed while copying: {source}")
    if destination.stat().st_size != after.st_size:
        raise RuntimeError(f"snapshot input size changed while copying: {source}")
    os.utime(destination, ns=(after.st_atime_ns, after.st_mtime_ns))
    return after


def _set_snapshot_permissions(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o2750 if path.is_dir() else 0o440)
    root.chmod(0o2750)


def _dataset_identity(lake_root: Path, dataset: str) -> str:
    try:
        frame = readers.read_dataset(lake_root, dataset)
    except Exception:
        return "not_observable"
    if frame.is_empty():
        return "empty"
    row = frame.tail(1).to_dicts()[0]
    return sha256_bytes(canonical_json_bytes(row))


def _git_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
    ).strip().lower()
    if len(value) != 40:
        raise RuntimeError("quant-lab git commit is not a full SHA")
    return value


def _environment_fingerprint() -> str:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "polars": pl.__version__,
        "env_keys": sorted(
            key
            for key in os.environ
            if key.startswith(("QUANT_LAB_", "V5_", "POLARS_"))
            and not any(token in key for token in ("TOKEN", "SECRET", "KEY", "PASSWORD"))
        ),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _schema_fingerprint() -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "datasets": sorted(readers.DATASET_PATHS),
                "report_schema": "quant_lab.expert_pack.v1",
                "package_version": __version__,
            }
        )
    )


def _media_type(path: Path, dataset: str) -> str:
    if dataset == "v5_bundle":
        return "bundle"
    suffix = path.suffix.lower().lstrip(".")
    return suffix if suffix in {"parquet", "json", "csv", "yaml"} else "other"
