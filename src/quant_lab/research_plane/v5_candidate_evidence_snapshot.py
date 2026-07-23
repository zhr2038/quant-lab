from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from quant_lab.data.file_index import LAKE_FILE_INDEX, build_lake_file_index
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.research.candidate_labels import (
    CANDIDATE_EVENT_DATASET,
    LABEL_SCHEMA_VERSION,
    MARKET_BAR_DATASET,
    RUN_SUMMARY_DATASET,
    candidate_event_symbol_expr,
    run_summary_run_ids,
)
from quant_lab.research.strategy_evidence import EVIDENCE_VERSION
from quant_lab.research_plane.signatures import (
    model_content_sha256,
    sha256_file,
    sign_model,
    verify_payload,
)
from quant_lab.research_plane.snapshot_lock import snapshot_payload_lock
from quant_lab.research_plane.status import ensure_research_queue_layout
from quant_lab.research_plane.v5_candidate_evidence_contracts import (
    V5_CANDIDATE_EVIDENCE_HORIZONS,
    V5CandidateEvidenceInputFingerprint,
    V5CandidateEvidenceSnapshotFile,
    V5CandidateEvidenceSnapshotManifest,
    V5CandidateEvidenceSourceFileIdentity,
    V5CandidateEvidenceTaskPayload,
)
from quant_lab.symbols import normalize_symbol

V5_CANDIDATE_EVIDENCE_INPUT_DATASETS = (
    CANDIDATE_EVENT_DATASET,
    MARKET_BAR_DATASET,
    RUN_SUMMARY_DATASET,
)
V5_CANDIDATE_EVIDENCE_REHYDRATE_PARTIAL_STALE_SECONDS = 6 * 60 * 60
V5_CANDIDATE_EVIDENCE_DATASET_NAMES = tuple(
    str(path).replace("\\", "/") for path in V5_CANDIDATE_EVIDENCE_INPUT_DATASETS
)

_EVENT_MINIMAL_SCHEMA = {
    "strategy": pl.Utf8,
    "candidate_id": pl.Utf8,
    "run_id": pl.Utf8,
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "normalized_symbol": pl.Utf8,
    "inst_id": pl.Utf8,
    "instId": pl.Utf8,
    "raw_payload_json": pl.Utf8,
}
_MARKET_COLUMNS = ("symbol", "timeframe", "ts", "close", "high", "low", "is_closed")
_MARKET_MINIMAL_SCHEMA = {
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "ts": pl.Datetime(time_zone="UTC"),
    "close": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "is_closed": pl.Boolean,
}
_RUN_SUMMARY_MINIMAL_SCHEMA = {
    "run_id": pl.Utf8,
    "bundle_ts": pl.Datetime(time_zone="UTC"),
}


@dataclass(frozen=True)
class V5CandidateEvidenceSnapshotPreflight:
    lake_root: Path
    queue_root: Path
    parameters: V5CandidateEvidenceTaskPayload
    quant_lab_commit: str
    event_window_start: datetime
    event_window_end: datetime
    market_window_start: datetime
    market_window_end: datetime
    candidate_symbols: tuple[str, ...]
    candidate_run_ids: tuple[str, ...]
    run_summary_run_ids: tuple[str, ...]
    selected_timeframes: tuple[tuple[str, str], ...]
    source_files: tuple[V5CandidateEvidenceSourceFileIdentity, ...]
    input_fingerprint: V5CandidateEvidenceInputFingerprint
    identity_payload: dict[str, Any]
    snapshot_id: str
    min_event_ts: datetime | None
    max_event_ts: datetime | None


@dataclass(frozen=True)
class V5CandidateEvidenceSnapshotMaterialization:
    manifest: V5CandidateEvidenceSnapshotManifest
    snapshot_materialized: bool
    snapshot_rehydrated: bool


def preflight_v5_candidate_evidence_snapshot(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    quant_lab_commit: str,
    mode: str = "incremental",
    lookback_days: int = 8,
    horizon_hours: tuple[int, ...] = V5_CANDIDATE_EVIDENCE_HORIZONS,
    include_historical_outcomes: bool = False,
) -> V5CandidateEvidenceSnapshotPreflight:
    """Resolve filtered Candidate Evidence identity before Snapshot materialization."""

    parameters = V5CandidateEvidenceTaskPayload(
        as_of_date=as_of_date,
        mode=mode,
        lookback_days=lookback_days,
        horizon_hours=horizon_hours,
        include_historical_outcomes=include_historical_outcomes,
    )
    root = Path(lake_root).resolve(strict=True)
    queue = ensure_research_queue_layout(queue_root)
    event_start = datetime.combine(
        as_of_date - timedelta(days=parameters.lookback_days),
        time.min,
        tzinfo=UTC,
    )
    event_end = datetime.combine(as_of_date + timedelta(days=1), time.min, tzinfo=UTC)
    market_end = datetime.combine(as_of_date + timedelta(days=6), time.min, tzinfo=UTC)

    indexed = _load_file_index(root)
    source_files = tuple(
        identity
        for dataset in V5_CANDIDATE_EVIDENCE_INPUT_DATASETS
        for identity in _indexed_source_identities(root, indexed, dataset)
    )
    event_sources = _source_paths(root, source_files, CANDIDATE_EVENT_DATASET)
    market_sources = _source_paths(root, source_files, MARKET_BAR_DATASET)
    run_sources = _source_paths(root, source_files, RUN_SUMMARY_DATASET)
    event_lazy = _event_projection(event_sources, event_start, event_end)
    event_digest, event_rows, min_event_ts, max_event_ts = _projection_digest(
        event_lazy,
        timestamp_column="ts_utc",
    )
    candidate_symbols = _candidate_symbols(event_lazy)
    candidate_run_ids = _candidate_run_ids(event_lazy)
    market_start = min_event_ts or event_start
    selected_timeframes = _selected_market_timeframes(
        market_sources,
        symbols=candidate_symbols,
        start=market_start,
        end=market_end,
    )
    market_lazy = _market_projection(
        market_sources,
        symbols=candidate_symbols,
        selected_timeframes=selected_timeframes,
        start=market_start,
        end=market_end,
    )
    run_lazy = _run_summary_projection(
        run_sources,
        event_start,
        event_end,
    )
    summary_run_ids = _run_summary_run_ids(run_lazy)
    market_digest, market_rows, _, _ = _projection_digest(
        market_lazy,
        timestamp_column="ts",
    )
    run_digest, run_rows, _, _ = _projection_digest(
        run_lazy,
        timestamp_column=_first_column(
            run_lazy,
            ("bundle_ts", "ingest_ts", "created_at"),
        ),
    )
    source_by_dataset = {
        name: [item for item in source_files if item.dataset_name == name]
        for name in V5_CANDIDATE_EVIDENCE_DATASET_NAMES
    }
    fingerprint_identity = {
        "schema_version": "v5_candidate_evidence_input_identity.v2",
        "quant_lab_commit": quant_lab_commit,
        "as_of_date": as_of_date,
        "mode": parameters.mode,
        "lookback_days": parameters.lookback_days,
        "horizon_hours": list(parameters.horizon_hours),
        "include_historical_outcomes": parameters.include_historical_outcomes,
        "candidate_label_schema_version": LABEL_SCHEMA_VERSION,
        "strategy_evidence_version": EVIDENCE_VERSION,
        "projection_version": parameters.projection_version,
        "event_window_start": event_start,
        "event_window_end": event_end,
        "market_window_start": market_start,
        "market_window_end": market_end,
        "candidate_symbols": list(candidate_symbols),
        "candidate_run_ids": list(candidate_run_ids),
        "run_summary_run_ids": list(summary_run_ids),
        "selected_timeframes": [list(item) for item in selected_timeframes],
        "candidate_event_digest": event_digest,
        "market_bar_digest": market_digest,
        "run_summary_digest": run_digest,
    }
    fingerprint_digest = model_content_sha256(fingerprint_identity)
    input_fingerprint = V5CandidateEvidenceInputFingerprint(
        quant_lab_commit=quant_lab_commit,
        as_of_date=as_of_date,
        mode=parameters.mode,
        lookback_days=parameters.lookback_days,
        horizon_hours=parameters.horizon_hours,
        candidate_label_schema_version=LABEL_SCHEMA_VERSION,
        strategy_evidence_version=EVIDENCE_VERSION,
        projection_version=parameters.projection_version,
        event_window_start=event_start,
        event_window_end=event_end,
        market_window_start=market_start,
        market_window_end=market_end,
        candidate_event_digest=event_digest,
        market_bar_digest=market_digest,
        run_summary_digest=run_digest,
        input_fingerprint_digest=fingerprint_digest,
        candidate_event_file_count=len(source_by_dataset["silver/v5_candidate_event"]),
        market_bar_file_count=len(source_by_dataset["silver/market_bar"]),
        run_summary_file_count=len(source_by_dataset["silver/v5_run_summary"]),
        candidate_event_row_count=event_rows,
        market_bar_row_count=market_rows,
        run_summary_row_count=run_rows,
        candidate_event_bytes=sum(
            item.size_bytes for item in source_by_dataset["silver/v5_candidate_event"]
        ),
        market_bar_bytes=sum(item.size_bytes for item in source_by_dataset["silver/market_bar"]),
        run_summary_bytes=sum(
            item.size_bytes for item in source_by_dataset["silver/v5_run_summary"]
        ),
        estimated_uncompressed_bytes=sum(item.uncompressed_bytes for item in source_files),
        observed_at=datetime.now(UTC),
    )
    identity_payload = {
        "schema_version": "quant_lab_v5_candidate_evidence_snapshot_identity.v2",
        **fingerprint_identity,
        "input_fingerprint_digest": fingerprint_digest,
    }
    snapshot_id = f"v5-candidate-evidence-{model_content_sha256(identity_payload)[:24]}"
    _assert_source_identities(root, source_files)
    return V5CandidateEvidenceSnapshotPreflight(
        lake_root=root,
        queue_root=queue,
        parameters=parameters,
        quant_lab_commit=quant_lab_commit,
        event_window_start=event_start,
        event_window_end=event_end,
        market_window_start=market_start,
        market_window_end=market_end,
        candidate_symbols=candidate_symbols,
        candidate_run_ids=candidate_run_ids,
        run_summary_run_ids=summary_run_ids,
        selected_timeframes=selected_timeframes,
        source_files=source_files,
        input_fingerprint=input_fingerprint,
        identity_payload=identity_payload,
        snapshot_id=snapshot_id,
        min_event_ts=min_event_ts,
        max_event_ts=max_event_ts,
    )


def materialize_v5_candidate_evidence_snapshot(
    preflight: V5CandidateEvidenceSnapshotPreflight,
    *,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    max_input_bytes: int,
    max_input_uncompressed_bytes: int,
    max_input_rows: int,
) -> V5CandidateEvidenceSnapshotMaterialization:
    """Materialize, reuse, or rehydrate the immutable filtered Snapshot."""

    queue = preflight.queue_root
    final_root = queue / "snapshots" / preflight.snapshot_id
    with snapshot_payload_lock(queue, preflight.snapshot_id, timeout_seconds=120):
        if final_root.is_dir() and (final_root / "files").is_dir():
            manifest = V5CandidateEvidenceSnapshotManifest.model_validate_json(
                (final_root / "manifest.json").read_text("utf-8")
            )
            verify_v5_candidate_evidence_snapshot_manifest(
                manifest,
                final_root=final_root,
                public_key=signing_key.public_key(),
            )
            if manifest.input_fingerprint_digest != (
                preflight.input_fingerprint.input_fingerprint_digest
            ):
                raise RuntimeError("v5_candidate_evidence_snapshot_identity_mismatch")
            return V5CandidateEvidenceSnapshotMaterialization(manifest, False, False)

        rehydrating = final_root.is_dir() and (final_root / "FILES_RELEASED.json").is_file()
        retained_manifest: V5CandidateEvidenceSnapshotManifest | None = None
        if rehydrating:
            retained_manifest = V5CandidateEvidenceSnapshotManifest.model_validate_json(
                (final_root / "manifest.json").read_text("utf-8")
            )
            verify_v5_candidate_evidence_snapshot_manifest(
                retained_manifest,
                final_root=final_root,
                public_key=signing_key.public_key(),
                require_payload=False,
            )
            if (
                retained_manifest.snapshot_id != preflight.snapshot_id
                or retained_manifest.input_fingerprint_digest
                != preflight.input_fingerprint.input_fingerprint_digest
            ):
                raise RuntimeError("v5_candidate_evidence_snapshot_rehydrate_identity_mismatch")
        temporary = (
            queue
            / "snapshots"
            / (
                f".rehydrate.{preflight.snapshot_id}.{uuid.uuid4().hex}.partial"
                if rehydrating
                else f".sealing.{preflight.snapshot_id}.{uuid.uuid4().hex}.partial"
            )
        )
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            if rehydrating and retained_manifest is not None:
                atomic_write_json(
                    temporary / "REHYDRATE.json",
                    {
                        "schema_version": ("quant_lab_v5_candidate_evidence_snapshot_rehydrate.v1"),
                        "snapshot_id": retained_manifest.snapshot_id,
                        "manifest_sha256": retained_manifest.manifest_sha256,
                        "started_at": datetime.now(UTC).isoformat(),
                    },
                )
            references = _materialize_snapshot_files(preflight, temporary)
            total_bytes = sum(item.size_bytes for item in references)
            total_rows = sum(item.row_count for item in references)
            total_uncompressed = sum(item.uncompressed_bytes for item in references)
            if total_bytes > max_input_bytes:
                raise RuntimeError("v5_candidate_evidence_snapshot_input_size_limit_exceeded")
            if total_rows > max_input_rows:
                raise RuntimeError("v5_candidate_evidence_snapshot_input_row_limit_exceeded")
            if total_uncompressed > max_input_uncompressed_bytes:
                raise RuntimeError(
                    "v5_candidate_evidence_snapshot_input_uncompressed_limit_exceeded"
                )
            _assert_source_identities(preflight.lake_root, preflight.source_files)
            if retained_manifest is not None:
                _verify_rehydrated_snapshot_files(retained_manifest, references)
                (temporary / "manifest.json").write_bytes(
                    (final_root / "manifest.json").read_bytes()
                )
                (temporary / "SEALED").write_bytes((final_root / "SEALED").read_bytes())
                manifest = retained_manifest
            else:
                manifest = _build_signed_snapshot_manifest(
                    preflight,
                    references,
                    total_bytes=total_bytes,
                    total_rows=total_rows,
                    total_uncompressed=total_uncompressed,
                    signing_key=signing_key,
                    signature_key_id=signature_key_id,
                )
                (temporary / "manifest.json").write_text(
                    manifest.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                (temporary / "SEALED").write_text(
                    manifest.manifest_sha256 + "\n",
                    encoding="ascii",
                )
            (temporary / "REHYDRATE.json").unlink(missing_ok=True)
            _make_tree_read_only(temporary)
            if final_root.exists():
                _make_tree_writable(final_root)
                backup = queue / "snapshots" / f".__v5_candidate_evidence_old_{uuid.uuid4().hex}"
                os.replace(final_root, backup)
                try:
                    os.replace(temporary, final_root)
                except Exception:
                    os.replace(backup, final_root)
                    raise
                shutil.rmtree(backup, ignore_errors=True)
            else:
                os.replace(temporary, final_root)
            return V5CandidateEvidenceSnapshotMaterialization(
                manifest,
                True,
                rehydrating,
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def _build_signed_snapshot_manifest(
    preflight: V5CandidateEvidenceSnapshotPreflight,
    references: list[V5CandidateEvidenceSnapshotFile],
    *,
    total_bytes: int,
    total_rows: int,
    total_uncompressed: int,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
) -> V5CandidateEvidenceSnapshotManifest:
    parameters = preflight.parameters
    provisional = V5CandidateEvidenceSnapshotManifest(
        snapshot_id=preflight.snapshot_id,
        generated_at=datetime.now(UTC),
        quant_lab_commit=preflight.quant_lab_commit,
        as_of_date=parameters.as_of_date,
        mode=parameters.mode,
        lookback_days=parameters.lookback_days,
        horizon_hours=parameters.horizon_hours,
        include_historical_outcomes=parameters.include_historical_outcomes,
        candidate_label_schema_version=parameters.candidate_label_schema_version,
        strategy_evidence_version=parameters.strategy_evidence_version,
        projection_version=parameters.projection_version,
        candidate_event_digest=preflight.input_fingerprint.candidate_event_digest,
        market_bar_digest=preflight.input_fingerprint.market_bar_digest,
        run_summary_digest=preflight.input_fingerprint.run_summary_digest,
        input_fingerprint_digest=preflight.input_fingerprint.input_fingerprint_digest,
        event_window_start=preflight.event_window_start,
        event_window_end=preflight.event_window_end,
        market_window_start=preflight.market_window_start,
        market_window_end=preflight.market_window_end,
        min_event_ts=preflight.min_event_ts,
        max_event_ts=preflight.max_event_ts,
        candidate_symbols=preflight.candidate_symbols,
        candidate_run_ids=preflight.candidate_run_ids,
        run_summary_run_ids=preflight.run_summary_run_ids,
        selected_timeframes=preflight.selected_timeframes,
        candidate_event_row_count=preflight.input_fingerprint.candidate_event_row_count,
        candidate_event_file_count=preflight.input_fingerprint.candidate_event_file_count,
        market_bar_row_count=preflight.input_fingerprint.market_bar_row_count,
        market_bar_file_count=preflight.input_fingerprint.market_bar_file_count,
        run_summary_row_count=preflight.input_fingerprint.run_summary_row_count,
        run_summary_file_count=preflight.input_fingerprint.run_summary_file_count,
        datasets=V5_CANDIDATE_EVIDENCE_DATASET_NAMES,
        files=tuple(references),
        total_input_bytes=total_bytes,
        total_input_rows=total_rows,
        estimated_uncompressed_bytes=total_uncompressed,
        manifest_sha256="0" * 64,
        signature_key_id=signature_key_id,
        signature="pending",
    )
    digest = model_content_sha256(provisional, blank_fields=("manifest_sha256",))
    unsigned = provisional.model_copy(update={"manifest_sha256": digest})
    return unsigned.model_copy(update={"signature": sign_model(unsigned, signing_key)})


def _verify_rehydrated_snapshot_files(
    retained: V5CandidateEvidenceSnapshotManifest,
    references: list[V5CandidateEvidenceSnapshotFile],
) -> None:
    def identity(item: V5CandidateEvidenceSnapshotFile) -> tuple[object, ...]:
        return (
            item.dataset_name,
            item.source_relative_path,
            item.relative_path,
            item.sha256,
            item.size_bytes,
            item.row_count,
            item.min_ts,
            item.max_ts,
            item.schema_fingerprint,
            item.uncompressed_bytes,
            item.media_type,
        )

    expected = sorted((identity(item) for item in retained.files), key=lambda item: item[2])
    actual = sorted((identity(item) for item in references), key=lambda item: item[2])
    if expected != actual:
        raise RuntimeError("v5_candidate_evidence_snapshot_rehydrate_identity_mismatch")
    if retained.total_input_bytes != sum(item.size_bytes for item in references):
        raise RuntimeError("v5_candidate_evidence_snapshot_rehydrate_identity_mismatch")
    if retained.total_input_rows != sum(item.row_count for item in references):
        raise RuntimeError("v5_candidate_evidence_snapshot_rehydrate_identity_mismatch")
    if retained.estimated_uncompressed_bytes != sum(item.uncompressed_bytes for item in references):
        raise RuntimeError("v5_candidate_evidence_snapshot_rehydrate_identity_mismatch")


def rehydrate_v5_candidate_evidence_snapshot_payload(
    lake_root: str | Path,
    queue_root: str | Path,
    snapshot_id: str,
    *,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    max_input_bytes: int,
    max_input_uncompressed_bytes: int,
    max_input_rows: int,
) -> V5CandidateEvidenceSnapshotMaterialization:
    root = Path(queue_root) / "snapshots" / snapshot_id
    manifest = V5CandidateEvidenceSnapshotManifest.model_validate_json(
        (root / "manifest.json").read_text("utf-8")
    )
    preflight = preflight_v5_candidate_evidence_snapshot(
        lake_root,
        queue_root,
        as_of_date=manifest.as_of_date,
        quant_lab_commit=manifest.quant_lab_commit,
        mode=manifest.mode,
        lookback_days=manifest.lookback_days,
        horizon_hours=manifest.horizon_hours,
        include_historical_outcomes=manifest.include_historical_outcomes,
    )
    if preflight.snapshot_id != snapshot_id:
        raise RuntimeError("v5_candidate_evidence_snapshot_rehydrate_identity_mismatch")
    return materialize_v5_candidate_evidence_snapshot(
        preflight,
        signing_key=signing_key,
        signature_key_id=signature_key_id,
        max_input_bytes=max_input_bytes,
        max_input_uncompressed_bytes=max_input_uncompressed_bytes,
        max_input_rows=max_input_rows,
    )


def cleanup_stale_v5_candidate_evidence_rehydrate_partials(
    queue_root: str | Path,
    *,
    stale_after_seconds: int = V5_CANDIDATE_EVIDENCE_REHYDRATE_PARTIAL_STALE_SECONDS,
    now: datetime | None = None,
) -> tuple[str, ...]:
    """Remove abandoned Candidate Evidence rehydrates only while their lock is free."""

    queue = ensure_research_queue_layout(queue_root)
    observed_at = now or datetime.now(UTC)
    removed: list[str] = []
    for partial in sorted((queue / "snapshots").glob(".rehydrate.*.partial")):
        marker_path = partial / "REHYDRATE.json"
        try:
            marker = json.loads(marker_path.read_text("utf-8"))
            if marker.get("schema_version") != (
                "quant_lab_v5_candidate_evidence_snapshot_rehydrate.v1"
            ):
                continue
            snapshot_id = str(marker["snapshot_id"])
            age = observed_at.timestamp() - partial.stat().st_mtime
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if age < stale_after_seconds:
            continue
        try:
            with snapshot_payload_lock(queue, snapshot_id, timeout_seconds=0):
                shutil.rmtree(partial, ignore_errors=False)
        except (TimeoutError, OSError):
            continue
        removed.append(partial.name)
        audit_path = queue / "audit" / "v5_candidate_evidence_snapshot.jsonl"
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "action": "snapshot_rehydrate_partial_cleaned",
                        "observed_at": observed_at.isoformat(),
                        "snapshot_id": snapshot_id,
                        "partial_name": partial.name,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
    return tuple(removed)


def verify_v5_candidate_evidence_snapshot_manifest(
    manifest: V5CandidateEvidenceSnapshotManifest,
    *,
    final_root: Path,
    public_key: Ed25519PublicKey,
    require_payload: bool = True,
) -> None:
    expected = model_content_sha256(manifest, blank_fields=("manifest_sha256",))
    if manifest.manifest_sha256 != expected:
        raise ValueError("v5_candidate_evidence_snapshot_manifest_digest_mismatch")
    verify_payload(manifest, manifest.signature, public_key)
    if (final_root / "SEALED").read_text("ascii").strip() != manifest.manifest_sha256:
        raise ValueError("v5_candidate_evidence_snapshot_seal_mismatch")
    if not require_payload:
        return
    expected_paths = {item.relative_path for item in manifest.files}
    actual_paths = {
        str(path.relative_to(final_root / "files")).replace("\\", "/")
        for path in (final_root / "files").rglob("*.parquet")
        if path.is_file()
    }
    if expected_paths != actual_paths:
        raise ValueError("v5_candidate_evidence_snapshot_file_set_mismatch")
    observed_candidate_symbols: set[str] = set()
    observed_candidate_run_ids: set[str] = set()
    observed_summary_run_ids: set[str] = set()
    for item in manifest.files:
        path = final_root / "files" / item.relative_path
        resolved = path.resolve(strict=True)
        files_root = (final_root / "files").resolve(strict=True)
        if files_root not in resolved.parents:
            raise ValueError("v5_candidate_evidence_snapshot_path_escape")
        if path.stat().st_size != item.size_bytes:
            raise ValueError("v5_candidate_evidence_snapshot_size_mismatch")
        if sha256_file(path) != item.sha256:
            raise ValueError("v5_candidate_evidence_snapshot_sha256_mismatch")
        if _schema_fingerprint(path) != item.schema_fingerprint:
            raise ValueError("v5_candidate_evidence_snapshot_schema_mismatch")
        if _parquet_uncompressed_bytes(path) != item.uncompressed_bytes:
            raise ValueError("v5_candidate_evidence_snapshot_uncompressed_mismatch")
        digest, rows, lower, upper = _projection_digest(
            pl.scan_parquet(path),
            timestamp_column=_timestamp_column_for_dataset(item.dataset_name, path),
        )
        if rows != item.row_count or lower != item.min_ts or upper != item.max_ts:
            raise ValueError("v5_candidate_evidence_snapshot_metadata_mismatch")
        expected_digest = {
            "silver/v5_candidate_event": manifest.candidate_event_digest,
            "silver/market_bar": manifest.market_bar_digest,
            "silver/v5_run_summary": manifest.run_summary_digest,
        }[item.dataset_name]
        if digest != expected_digest:
            raise ValueError("v5_candidate_evidence_snapshot_projection_digest_mismatch")
        lazy = pl.scan_parquet(path)
        if item.dataset_name == "silver/v5_candidate_event":
            observed_candidate_symbols.update(_candidate_symbols(lazy))
            observed_candidate_run_ids.update(_candidate_run_ids(lazy))
        elif (
            item.dataset_name == "silver/v5_run_summary"
            and "run_id" in lazy.collect_schema().names()
        ):
            observed_summary_run_ids.update(
                run_summary_run_ids(lazy.select("run_id").collect(engine="streaming"))
            )
    if tuple(sorted(observed_candidate_symbols)) != manifest.candidate_symbols:
        raise ValueError("v5_candidate_evidence_snapshot_symbol_identity_mismatch")
    if tuple(sorted(observed_candidate_run_ids)) != manifest.candidate_run_ids:
        raise ValueError("v5_candidate_evidence_snapshot_candidate_run_identity_mismatch")
    if tuple(sorted(observed_summary_run_ids)) != manifest.run_summary_run_ids:
        raise ValueError("v5_candidate_evidence_snapshot_summary_run_identity_mismatch")


def load_v5_candidate_evidence_generation_binding(
    lake_root: str | Path,
) -> tuple[str | None, str | None, dict[str, Any]]:
    pointer = Path(lake_root) / "gold" / "v5_candidate_evidence_generation.json"
    if not pointer.is_file():
        return None, None, {}
    try:
        payload = json.loads(pointer.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("v5_candidate_evidence_generation_pointer_invalid") from exc
    generation_id = str(payload.get("generation_id") or "") or None
    generation_digest = str(payload.get("generation_digest") or "") or None
    if (generation_id is None) != (generation_digest is None):
        raise RuntimeError("v5_candidate_evidence_generation_pointer_incomplete")
    return generation_id, generation_digest, payload


def _materialize_snapshot_files(
    preflight: V5CandidateEvidenceSnapshotPreflight,
    temporary: Path,
) -> list[V5CandidateEvidenceSnapshotFile]:
    source_files = preflight.source_files
    event_sources = _source_paths(preflight.lake_root, source_files, CANDIDATE_EVENT_DATASET)
    market_sources = _source_paths(preflight.lake_root, source_files, MARKET_BAR_DATASET)
    run_sources = _source_paths(preflight.lake_root, source_files, RUN_SUMMARY_DATASET)
    projections = (
        (
            "silver/v5_candidate_event",
            _event_projection(
                event_sources,
                preflight.event_window_start,
                preflight.event_window_end,
            ),
            "ts_utc",
        ),
        (
            "silver/market_bar",
            _market_projection(
                market_sources,
                symbols=preflight.candidate_symbols,
                selected_timeframes=preflight.selected_timeframes,
                start=preflight.market_window_start,
                end=preflight.market_window_end,
            ),
            "ts",
        ),
        (
            "silver/v5_run_summary",
            _run_summary_projection(
                run_sources,
                preflight.event_window_start,
                preflight.event_window_end,
            ),
            None,
        ),
    )
    references: list[V5CandidateEvidenceSnapshotFile] = []
    expected_digests = {
        "silver/v5_candidate_event": preflight.input_fingerprint.candidate_event_digest,
        "silver/market_bar": preflight.input_fingerprint.market_bar_digest,
        "silver/v5_run_summary": preflight.input_fingerprint.run_summary_digest,
    }
    for dataset_name, lazy, timestamp_column in projections:
        relative_path = f"{dataset_name}/data.parquet"
        path = temporary / "files" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        lazy.sink_parquet(path, compression="zstd", maintain_order=True)
        actual_ts_column = timestamp_column or _timestamp_column_for_dataset(dataset_name, path)
        digest, rows, lower, upper = _projection_digest(
            pl.scan_parquet(path),
            timestamp_column=actual_ts_column,
        )
        if digest != expected_digests[dataset_name]:
            raise RuntimeError("snapshot_source_changed_while_sealing")
        stat = path.stat()
        references.append(
            V5CandidateEvidenceSnapshotFile(
                dataset_name=dataset_name,
                source_relative_path=dataset_name,
                relative_path=relative_path,
                sha256=sha256_file(path),
                size_bytes=stat.st_size,
                row_count=rows,
                mtime_ns=stat.st_mtime_ns,
                min_ts=lower,
                max_ts=upper,
                schema_fingerprint=_schema_fingerprint(path),
                uncompressed_bytes=_parquet_uncompressed_bytes(path),
            )
        )
    return references


def _event_projection(
    sources: tuple[Path, ...],
    start: datetime,
    end: datetime,
) -> pl.LazyFrame:
    lazy = _scan_union_sources(sources, _EVENT_MINIMAL_SCHEMA)
    column = _first_column(lazy, ("ts_utc", "bundle_ts", "ingest_ts"))
    if column is None:
        return lazy
    return lazy.filter(_utc_expr(column).is_between(start, end, closed="left"))


def _market_projection(
    sources: tuple[Path, ...],
    *,
    symbols: tuple[str, ...],
    selected_timeframes: tuple[tuple[str, str], ...],
    start: datetime,
    end: datetime,
) -> pl.LazyFrame:
    lazy = _scan_sources(sources, _MARKET_MINIMAL_SCHEMA)
    columns = lazy.collect_schema().names()
    required = {"symbol", "ts", "close", "high", "low"}
    if not symbols or not required.issubset(columns):
        return pl.DataFrame(schema=_MARKET_MINIMAL_SCHEMA).lazy()
    if "is_closed" in columns:
        lazy = lazy.filter(pl.col("is_closed").cast(pl.Boolean, strict=False).fill_null(True))
    lazy = lazy.with_columns(
        pl.col("symbol").map_elements(normalize_symbol, return_dtype=pl.Utf8).alias("symbol")
    )
    lazy = lazy.filter(_utc_expr("ts").is_between(start, end, closed="left"))
    if symbols:
        lazy = lazy.filter(pl.col("symbol").cast(pl.Utf8).is_in(symbols))
    if selected_timeframes and "timeframe" in columns:
        predicate: pl.Expr | None = None
        for symbol, timeframe in selected_timeframes:
            item = (pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe)
            predicate = item if predicate is None else predicate | item
        if predicate is not None:
            lazy = lazy.filter(predicate)
    selected = [column for column in _MARKET_COLUMNS if column in columns]
    return lazy.select(selected)


def _run_summary_projection(
    sources: tuple[Path, ...],
    start: datetime,
    end: datetime,
) -> pl.LazyFrame:
    lazy = _scan_sources(sources, _RUN_SUMMARY_MINIMAL_SCHEMA)
    column = _first_column(lazy, ("bundle_ts", "ingest_ts", "created_at"))
    if column is None:
        return lazy
    lazy = lazy.filter(_utc_expr(column).is_between(start, end, closed="left"))
    return lazy


def _candidate_symbols(events: pl.LazyFrame) -> tuple[str, ...]:
    columns = events.collect_schema().names()
    values = (
        events.select(candidate_event_symbol_expr(columns).alias("_candidate_symbol"))
        .filter(pl.col("_candidate_symbol") != "")
        .unique()
        .collect(engine="streaming")
        .get_column("_candidate_symbol")
        .to_list()
    )
    return tuple(sorted({str(value) for value in values if str(value)}))


def _candidate_run_ids(events: pl.LazyFrame) -> tuple[str, ...]:
    if "run_id" not in events.collect_schema().names():
        return ()
    values = (
        events.select(pl.col("run_id").cast(pl.Utf8).drop_nulls().unique())
        .collect(engine="streaming")
        .get_column("run_id")
        .to_list()
    )
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _run_summary_run_ids(run_summary: pl.LazyFrame) -> tuple[str, ...]:
    frame = run_summary.collect(engine="streaming")
    return tuple(sorted(run_summary_run_ids(frame)))


def _selected_market_timeframes(
    sources: tuple[Path, ...],
    *,
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> tuple[tuple[str, str], ...]:
    if not sources or not symbols:
        return ()
    lazy = _scan_sources(sources, _MARKET_MINIMAL_SCHEMA)
    columns = lazy.collect_schema().names()
    if not {"symbol", "timeframe", "ts"}.issubset(columns):
        return ()
    if "is_closed" in columns:
        lazy = lazy.filter(pl.col("is_closed").cast(pl.Boolean, strict=False).fill_null(True))
    lazy = lazy.with_columns(
        pl.col("symbol").map_elements(normalize_symbol, return_dtype=pl.Utf8).alias("symbol")
    )
    available = (
        lazy.filter(
            pl.col("symbol").cast(pl.Utf8).is_in(symbols)
            & _utc_expr("ts").is_between(start, end, closed="left")
        )
        .group_by(["symbol", "timeframe"])
        .agg(pl.len().alias("row_count"))
        .collect(engine="streaming")
    )
    selected: list[tuple[str, str]] = []
    for symbol in symbols:
        rows = available.filter(pl.col("symbol") == symbol).to_dicts()
        if not rows:
            continue
        chosen = min(
            rows,
            key=lambda row: _timeframe_rank(
                str(row.get("timeframe") or ""),
                int(row.get("row_count") or 0),
            ),
        )
        selected.append((symbol, str(chosen["timeframe"])))
    return tuple(selected)


def _timeframe_rank(value: str, row_count: int) -> tuple[int, int, int, int]:
    minutes = _timeframe_minutes(value)
    return (
        0 if minutes == 60 else 1,
        0 if minutes is not None and minutes <= 60 else 1,
        minutes or 1_000_000,
        -row_count,
    )


def _timeframe_minutes(value: str) -> int | None:
    text = value.strip()
    if len(text) < 2:
        return None
    try:
        amount = int(float(text[:-1]))
    except ValueError:
        return None
    unit = text[-1].lower()
    return {"m": amount, "h": amount * 60, "d": amount * 1440}.get(unit)


def _projection_digest(
    lazy: pl.LazyFrame,
    *,
    timestamp_column: str | None,
) -> tuple[str, int, datetime | None, datetime | None]:
    schema = lazy.collect_schema()
    columns = sorted(schema.names())
    schema_payload = [(column, str(schema[column])) for column in columns]
    expressions: list[pl.Expr] = [pl.len().alias("row_count")]
    if columns:
        row = pl.struct(columns)
        expressions.extend(
            [
                row.hash(seed=0).sum().alias("hash_sum_0"),
                row.hash(seed=1).sum().alias("hash_sum_1"),
                row.hash(seed=2).min().alias("hash_min"),
                row.hash(seed=3).max().alias("hash_max"),
            ]
        )
    if timestamp_column and timestamp_column in columns:
        expressions.extend(
            [
                _utc_expr(timestamp_column).min().alias("min_ts"),
                _utc_expr(timestamp_column).max().alias("max_ts"),
            ]
        )
    frame = lazy.select(expressions).collect(engine="streaming")
    payload = {
        "schema_version": "v5_candidate_evidence_projection_digest.v2",
        "schema": schema_payload,
        "row_count": int(frame.item(0, "row_count") or 0),
        "hash_sum_0": int(frame.item(0, "hash_sum_0") or 0) if columns else 0,
        "hash_sum_1": int(frame.item(0, "hash_sum_1") or 0) if columns else 0,
        "hash_min": int(frame.item(0, "hash_min") or 0) if columns else 0,
        "hash_max": int(frame.item(0, "hash_max") or 0) if columns else 0,
    }
    lower = _as_utc(frame.item(0, "min_ts")) if "min_ts" in frame.columns else None
    upper = _as_utc(frame.item(0, "max_ts")) if "max_ts" in frame.columns else None
    return model_content_sha256(payload), payload["row_count"], lower, upper


def _load_file_index(root: Path) -> pl.DataFrame:
    build_lake_file_index(root, V5_CANDIDATE_EVIDENCE_DATASET_NAMES)
    index = read_parquet_dataset(root / LAKE_FILE_INDEX)
    required = {
        "dataset",
        "path",
        "row_count",
        "size_bytes",
        "mtime_ns",
        "sha256",
        "schema_fingerprint",
        "uncompressed_bytes",
    }
    if index.is_empty():
        return pl.DataFrame(schema={column: pl.Utf8 for column in sorted(required)})
    if not required.issubset(index.columns):
        raise RuntimeError("lake_file_index_missing_or_invalid")
    return index


def _indexed_source_identities(
    root: Path,
    index: pl.DataFrame,
    dataset: Path,
) -> tuple[V5CandidateEvidenceSourceFileIdentity, ...]:
    dataset_name = str(dataset).replace("\\", "/")
    if index.is_empty():
        return ()
    identities: list[V5CandidateEvidenceSourceFileIdentity] = []
    for row in index.filter(pl.col("dataset") == dataset_name).sort("path").to_dicts():
        relative = str(row.get("path") or "")
        candidate = (root / relative).resolve(strict=True)
        dataset_root = (root / dataset).resolve(strict=True)
        if dataset_root not in candidate.parents:
            raise ValueError("v5_candidate_evidence_source_path_escape")
        stat = candidate.stat()
        if stat.st_size != int(row["size_bytes"]) or stat.st_mtime_ns != int(row["mtime_ns"]):
            raise RuntimeError(f"lake_file_index_source_changed:{relative}")
        identities.append(
            V5CandidateEvidenceSourceFileIdentity(
                dataset_name=dataset_name,
                relative_path=relative,
                sha256=str(row["sha256"]),
                size_bytes=int(row["size_bytes"]),
                mtime_ns=int(row["mtime_ns"]),
                row_count=int(row["row_count"]),
                min_ts=_as_utc(row.get("min_ts")),
                max_ts=_as_utc(row.get("max_ts")),
                schema_fingerprint=str(row["schema_fingerprint"]),
                uncompressed_bytes=int(row["uncompressed_bytes"]),
            )
        )
    return tuple(identities)


def _assert_source_identities(
    root: Path,
    identities: tuple[V5CandidateEvidenceSourceFileIdentity, ...],
) -> None:
    for item in identities:
        path = root / item.relative_path
        stat = path.stat()
        if stat.st_size != item.size_bytes or stat.st_mtime_ns != item.mtime_ns:
            raise RuntimeError("snapshot_source_changed_while_sealing")
        if sha256_file(path) != item.sha256:
            raise RuntimeError("snapshot_source_changed_while_sealing")


def _source_paths(
    root: Path,
    identities: tuple[V5CandidateEvidenceSourceFileIdentity, ...],
    dataset: Path,
) -> tuple[Path, ...]:
    name = str(dataset).replace("\\", "/")
    return tuple(root / item.relative_path for item in identities if item.dataset_name == name)


def _scan_sources(sources: tuple[Path, ...], empty_schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    if not sources:
        return pl.DataFrame(schema=empty_schema).lazy()
    return pl.scan_parquet(
        [str(path) for path in sources],
        missing_columns="insert",
        extra_columns="ignore",
    )


def _scan_union_sources(
    sources: tuple[Path, ...],
    empty_schema: dict[str, pl.DataType],
) -> pl.LazyFrame:
    if not sources:
        return pl.DataFrame(schema=empty_schema).lazy()
    return pl.concat(
        [pl.scan_parquet(str(path)) for path in sources],
        how="diagonal_relaxed",
    )


def _first_column(lazy: pl.LazyFrame, candidates: tuple[str, ...]) -> str | None:
    columns = lazy.collect_schema().names()
    return next((column for column in candidates if column in columns), None)


def _timestamp_column_for_dataset(dataset_name: str, path: Path) -> str | None:
    schema = pl.read_parquet_schema(path)
    candidates = {
        "silver/v5_candidate_event": ("ts_utc", "bundle_ts", "ingest_ts"),
        "silver/market_bar": ("ts",),
        "silver/v5_run_summary": ("bundle_ts", "ingest_ts", "created_at"),
    }[dataset_name]
    return next((column for column in candidates if column in schema), None)


def _utc_expr(column: str) -> pl.Expr:
    return pl.coalesce(
        [
            pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(time_zone="UTC", strict=False),
        ]
    )


def _as_utc(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value in (None, ""):
        return None
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _schema_fingerprint(path: Path) -> str:
    schema = pl.read_parquet_schema(path)
    payload = json.dumps(
        [(str(name), str(dtype)) for name, dtype in schema.items()],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parquet_uncompressed_bytes(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(group).column(column).total_uncompressed_size
        for group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _make_tree_read_only(path: Path) -> None:
    for candidate in sorted(path.rglob("*"), reverse=True):
        candidate.chmod(0o440 if candidate.is_file() else 0o550)
    path.chmod(0o550)


def _make_tree_writable(path: Path) -> None:
    path.chmod(0o750)
    for candidate in path.rglob("*"):
        candidate.chmod(0o640 if candidate.is_file() else 0o750)
