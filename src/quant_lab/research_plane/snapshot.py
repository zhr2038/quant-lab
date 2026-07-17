from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.data.file_index import LAKE_FILE_INDEX, build_lake_file_index
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.entry_quality import (
    COST_BUCKET_DAILY_DATASET,
    ENTRY_QUALITY_SCHEMA_VERSION,
    MARKET_BAR_DATASET,
    PULLBACK_HORIZON_HOURS,
    V5_CANDIDATE_EVENT_DATASET,
    V5_CANDIDATE_LABEL_DATASET,
    V5_ORDER_LIFECYCLE_DATASET,
    V5_TRADE_EVENT_DATASET,
)
from quant_lab.research_plane.contracts import (
    RESEARCH_SNAPSHOT_SCHEMA,
    ResearchDatasetReference,
    ResearchSnapshotManifest,
)
from quant_lab.research_plane.signatures import model_content_sha256, sha256_file, sign_model
from quant_lab.research_plane.status import ensure_research_queue_layout

ENTRY_QUALITY_INPUT_DATASETS = (
    V5_TRADE_EVENT_DATASET,
    V5_ORDER_LIFECYCLE_DATASET,
    MARKET_BAR_DATASET,
    V5_CANDIDATE_EVENT_DATASET,
    V5_CANDIDATE_LABEL_DATASET,
    COST_BUCKET_DAILY_DATASET,
)

SNAPSHOT_TIME_COLUMNS = {
    str(V5_TRADE_EVENT_DATASET).replace("\\", "/"): ("ts_utc", "ts", "entry_ts"),
    str(V5_ORDER_LIFECYCLE_DATASET).replace("\\", "/"): (
        "ts_utc",
        "last_fill_ts",
        "submit_ts",
        "decision_ts",
    ),
    str(MARKET_BAR_DATASET).replace("\\", "/"): ("ts",),
    str(V5_CANDIDATE_EVENT_DATASET).replace("\\", "/"): ("ts_utc", "ts"),
    str(V5_CANDIDATE_LABEL_DATASET).replace("\\", "/"): (
        "decision_ts",
        "ts_utc",
    ),
    str(COST_BUCKET_DAILY_DATASET).replace("\\", "/"): ("as_of_date",),
}


def seal_entry_quality_history_snapshot(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    start_date: date,
    end_date: date,
    selected_v5_bundle_id: str,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str | None = None,
) -> ResearchSnapshotManifest:
    root = Path(lake_root).resolve()
    queue = ensure_research_queue_layout(queue_root)
    commit = quant_lab_commit or _git_commit()
    start_dt = datetime.combine(start_date, time.min, tzinfo=UTC)
    end_dt = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=UTC)
    windows = {
        str(V5_TRADE_EVENT_DATASET).replace("\\", "/"): (start_dt, end_dt),
        str(V5_ORDER_LIFECYCLE_DATASET).replace("\\", "/"): (start_dt, end_dt),
        str(V5_CANDIDATE_EVENT_DATASET).replace("\\", "/"): (start_dt, end_dt),
        str(MARKET_BAR_DATASET).replace("\\", "/"): (
            start_dt - timedelta(hours=24),
            end_dt + timedelta(hours=max(PULLBACK_HORIZON_HOURS)),
        ),
        str(V5_CANDIDATE_LABEL_DATASET).replace("\\", "/"): (
            start_dt,
            end_dt + timedelta(hours=max(PULLBACK_HORIZON_HOURS)),
        ),
        str(COST_BUCKET_DAILY_DATASET).replace("\\", "/"): (start_dt, end_dt),
    }
    index = _load_required_file_index(root)
    selected = _select_indexed_files(root, index, windows)
    temporary = queue / "snapshots" / f".sealing.{uuid.uuid4().hex}.partial"
    temporary.mkdir(parents=True, exist_ok=False)
    references: list[ResearchDatasetReference] = []
    try:
        for dataset, indexed_source, min_ts, max_ts in selected:
            if indexed_source.is_symlink():
                raise ValueError("snapshot_source_path_escape")
            source = indexed_source.resolve(strict=True)
            if root not in source.parents:
                raise ValueError("snapshot_source_path_escape")
            source_relative = str(source.relative_to(root)).replace("\\", "/")
            relative_path = source_relative
            destination = temporary / "files" / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            before = source.stat()
            shutil.copy2(source, destination)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("snapshot_source_changed_while_sealing")
            destination.chmod(0o440)
            stat = destination.stat()
            references.append(
                ResearchDatasetReference(
                    dataset_name=dataset,
                    source_relative_path=source_relative,
                    relative_path=relative_path,
                    sha256=sha256_file(destination),
                    size_bytes=stat.st_size,
                    row_count=_parquet_row_count(destination),
                    mtime_ns=stat.st_mtime_ns,
                    min_ts=min_ts,
                    max_ts=max_ts,
                )
            )
        references.sort(key=lambda item: (item.dataset_name, item.relative_path))
        snapshot_seed = model_content_sha256(
            {
                "schema_version": RESEARCH_SNAPSHOT_SCHEMA,
                "commit": commit,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "selected_v5_bundle_id": selected_v5_bundle_id,
                "entry_quality_schema_version": ENTRY_QUALITY_SCHEMA_VERSION,
                "files": [item.model_dump(mode="json") for item in references],
            }
        )[:24]
        snapshot_id = f"entry-quality-history-{snapshot_seed}"
        final_root = queue / "snapshots" / snapshot_id
        if (final_root / "SEALED").is_file():
            manifest = ResearchSnapshotManifest.model_validate_json(
                (final_root / "manifest.json").read_text(encoding="utf-8")
            )
            verify_snapshot_manifest(manifest, final_root=final_root)
            shutil.rmtree(temporary, ignore_errors=True)
            return manifest
        if final_root.exists():
            raise RuntimeError("research_snapshot_destination_incomplete")
        provisional = ResearchSnapshotManifest(
            schema_version=RESEARCH_SNAPSHOT_SCHEMA,
            snapshot_id=snapshot_id,
            generated_at=datetime.now(UTC),
            quant_lab_commit=commit,
            selected_v5_bundle_id=selected_v5_bundle_id,
            entry_quality_schema_version=ENTRY_QUALITY_SCHEMA_VERSION,
            datasets=[str(path).replace("\\", "/") for path in ENTRY_QUALITY_INPUT_DATASETS],
            files=references,
            total_input_bytes=sum(item.size_bytes for item in references),
            total_input_rows=sum(item.row_count for item in references),
            manifest_sha256="0" * 64,
            signature_key_id=signature_key_id,
            signature="pending",
        )
        digest = model_content_sha256(provisional, blank_fields=("manifest_sha256",))
        unsigned = provisional.model_copy(update={"manifest_sha256": digest})
        manifest = unsigned.model_copy(update={"signature": sign_model(unsigned, signing_key)})
        (temporary / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        (temporary / "SEALED").write_text(digest + "\n", encoding="ascii")
        _make_snapshot_read_only(temporary)
        os.replace(temporary, final_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_snapshot_manifest(
    manifest: ResearchSnapshotManifest,
    *,
    final_root: Path | None = None,
) -> None:
    expected = model_content_sha256(manifest, blank_fields=("manifest_sha256",))
    if expected != manifest.manifest_sha256:
        raise ValueError("research_snapshot_manifest_digest_mismatch")
    expected_datasets = {str(path).replace("\\", "/") for path in ENTRY_QUALITY_INPUT_DATASETS}
    if set(manifest.datasets) != expected_datasets:
        raise ValueError("research_snapshot_dataset_set_mismatch")
    if final_root is not None:
        root = final_root.resolve(strict=True)
        seal = (root / "SEALED").read_text(encoding="ascii").strip()
        if seal != manifest.manifest_sha256:
            raise ValueError("research_snapshot_seal_mismatch")
        for reference in manifest.files:
            unresolved = root / "files" / reference.relative_path
            if _path_has_symlink(root, unresolved):
                raise ValueError("research_snapshot_path_escape")
            candidate = unresolved.resolve(strict=True)
            if root not in candidate.parents:
                raise ValueError("research_snapshot_path_escape")
            if candidate.stat().st_size != reference.size_bytes:
                raise ValueError("research_snapshot_size_mismatch")
            if sha256_file(candidate) != reference.sha256:
                raise ValueError("research_snapshot_sha256_mismatch")


def _load_required_file_index(root: Path) -> pl.DataFrame:
    required_names = {str(path).replace("\\", "/") for path in ENTRY_QUALITY_INPUT_DATASETS}
    # Refresh file membership every request so a previously indexed dataset cannot
    # hide newly arrived files. The index reuses unchanged file metadata.
    build_lake_file_index(root, sorted(required_names))
    index = read_parquet_dataset(root / LAKE_FILE_INDEX)
    if index.is_empty():
        return pl.DataFrame(
            schema={
                "dataset": pl.Utf8,
                "path": pl.Utf8,
                "min_ts": pl.Utf8,
                "max_ts": pl.Utf8,
            }
        )
    if not {"dataset", "path", "min_ts", "max_ts"}.issubset(set(index.columns)):
        raise RuntimeError("lake_file_index_missing_or_invalid")
    return index


def _select_indexed_files(
    root: Path,
    index: pl.DataFrame,
    windows: dict[str, tuple[datetime, datetime]],
) -> list[tuple[str, Path, datetime | None, datetime | None]]:
    selected: list[tuple[str, Path, datetime | None, datetime | None]] = []
    for row in index.to_dicts():
        dataset = str(row.get("dataset") or "")
        if dataset not in windows:
            continue
        min_ts = _as_utc(row.get("min_ts"))
        max_ts = _as_utc(row.get("max_ts"))
        since, before = windows[dataset]
        relative = str(row.get("path") or "")
        if not relative:
            continue
        indexed = root / relative
        if _path_has_symlink(root, indexed):
            raise ValueError("lake_file_index_path_escape")
        candidate = indexed.resolve(strict=True)
        dataset_root = (root / dataset).resolve(strict=True)
        if (
            root not in dataset_root.parents
            or dataset_root not in candidate.parents
            or candidate.suffix != ".parquet"
        ):
            raise ValueError("lake_file_index_path_escape")
        if min_ts is None or max_ts is None:
            min_ts, max_ts = _dataset_file_time_bounds(candidate, dataset)
        if min_ts is not None and max_ts is not None and not (max_ts >= since and min_ts < before):
            continue
        selected.append((dataset, candidate, min_ts, max_ts))
    selected.sort(key=lambda item: (item[0], str(item[1])))
    return selected


def _parquet_row_count(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item())


def _dataset_file_time_bounds(
    path: Path,
    dataset: str,
) -> tuple[datetime | None, datetime | None]:
    schema = pl.read_parquet_schema(path)
    column = next(
        (name for name in SNAPSHOT_TIME_COLUMNS.get(dataset, ()) if name in schema),
        None,
    )
    if column is None:
        return None, None
    expression = pl.coalesce(
        [
            pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(time_zone="UTC", strict=False),
        ]
    )
    bounds = (
        pl.scan_parquet(path)
        .select(expression.min().alias("min_ts"), expression.max().alias("max_ts"))
        .collect(engine="streaming")
    )
    return _as_utc(bounds.item(0, "min_ts")), _as_utc(bounds.item(0, "max_ts"))


def _as_utc(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _make_snapshot_read_only(path: Path) -> None:
    for candidate in sorted(path.rglob("*"), reverse=True):
        try:
            candidate.chmod(0o440 if candidate.is_file() else 0o550)
        except OSError:
            pass
    try:
        path.chmod(0o550)
    except OSError:
        pass


def _path_has_symlink(root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _git_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if len(value) != 40:
        raise RuntimeError("quant_lab_commit_unavailable")
    return value
