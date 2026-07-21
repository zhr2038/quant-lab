from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
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
from quant_lab.factors.factory import (
    COST_BUCKET_DAILY_DATASET,
    FEATURE_VALUE_DATASET,
    MARKET_BAR_DATASET,
)
from quant_lab.factors.plan import build_effective_factor_plan
from quant_lab.research_plane.contracts import (
    DEFAULT_FACTOR_FACTORY_MAX_SNAPSHOT_BYTES,
    FACTOR_FACTORY_SNAPSHOT_SCHEMA,
    FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1,
    FactorFactoryCostSnapshotRecord,
    FactorFactoryPreviousGeneration,
    FactorFactorySnapshotManifest,
    FactorFactorySourceFileIdentity,
    ResearchDatasetReference,
)
from quant_lab.research_plane.signatures import (
    model_content_sha256,
    sha256_file,
    sign_model,
    verify_payload,
)
from quant_lab.research_plane.snapshot_lock import snapshot_payload_lock
from quant_lab.research_plane.status import ensure_research_queue_layout

FACTOR_FACTORY_INPUT_DATASETS = (
    FEATURE_VALUE_DATASET,
    MARKET_BAR_DATASET,
    COST_BUCKET_DAILY_DATASET,
)

FEATURE_VALUE_COLUMNS = (
    "feature_set",
    "feature_name",
    "feature_version",
    "symbol",
    "timeframe",
    "ts",
    "value",
    "lookback_bars",
    "input_dataset_version",
    "input_hash",
    "code_version",
    "created_at",
    "source",
    "is_valid",
    "invalid_reason",
)
MARKET_BAR_COLUMNS = (
    "symbol",
    "timeframe",
    "ts",
    "close",
    "is_closed",
)
COST_COLUMNS = (
    "as_of_date",
    "day",
    "symbol",
    "total_cost_bps_p50",
    "total_cost_bps_p75",
    "total_cost_bps_p90",
    "total_cost_bps_p95",
    "cost_model_version",
    "cost_source",
    "source",
)

FACTOR_FACTORY_PROJECTION_VERSION = "factor_factory_snapshot_projection.v2"
REHYDRATE_PARTIAL_STALE_SECONDS = 60 * 60


@dataclass(frozen=True)
class FactorFactorySnapshotPreflight:
    lake_root: Path
    queue_root: Path
    as_of_date: date
    feature_set: str
    feature_version: str
    factor_version: str
    timeframe: str
    horizon_bars: tuple[int, ...]
    decision_delay_bars: int
    max_factors: int
    min_samples: int
    top_quantile: float
    cost_quantile: str
    quant_lab_commit: str
    factor_plan: Any
    feature_sources: tuple[Path, ...]
    market_sources: tuple[Path, ...]
    cost_sources: tuple[Path, ...]
    source_files: tuple[FactorFactorySourceFileIdentity, ...]
    feature_min_ts: datetime | None
    feature_max_ts: datetime | None
    market_before: datetime | None
    cost_frame: pl.DataFrame
    cost_snapshot: tuple[FactorFactoryCostSnapshotRecord, ...]
    source_input_digest: str
    cost_input_digest: str
    identity_payload: dict[str, Any]
    snapshot_id: str
    feature_estimated_uncompressed_bytes: int
    market_estimated_uncompressed_bytes: int
    cost_estimated_uncompressed_bytes: int
    estimated_uncompressed_bytes: int


@dataclass(frozen=True)
class FactorFactorySnapshotMaterialization:
    manifest: FactorFactorySnapshotManifest
    snapshot_materialized: bool
    snapshot_rehydrated: bool


def seal_factor_factory_snapshot(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    feature_set: str = "core",
    feature_version: str = "v0.1",
    factor_version: str = "v0.1",
    timeframe: str = "1H",
    horizon_bars: tuple[int, ...] = (4, 8, 24, 72),
    decision_delay_bars: int = 1,
    max_factors: int = 200,
    min_samples: int = 100,
    top_quantile: float = 0.2,
    cost_quantile: str = "p75",
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str | None = None,
    max_input_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_SNAPSHOT_BYTES,
    max_input_rows: int = 150_000_000,
) -> FactorFactorySnapshotManifest:
    """Compatibility wrapper for preflight plus conditional materialization."""

    preflight = preflight_factor_factory_snapshot(
        lake_root,
        queue_root,
        as_of_date=as_of_date,
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        horizon_bars=horizon_bars,
        decision_delay_bars=decision_delay_bars,
        max_factors=max_factors,
        min_samples=min_samples,
        top_quantile=top_quantile,
        cost_quantile=cost_quantile,
        quant_lab_commit=quant_lab_commit,
    )
    return materialize_factor_factory_snapshot(
        preflight,
        signing_key=signing_key,
        signature_key_id=signature_key_id,
        max_input_bytes=max_input_bytes,
        max_input_rows=max_input_rows,
    ).manifest


def preflight_factor_factory_snapshot(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    feature_set: str = "core",
    feature_version: str = "v0.1",
    factor_version: str = "v0.1",
    timeframe: str = "1H",
    horizon_bars: tuple[int, ...] = (4, 8, 24, 72),
    decision_delay_bars: int = 1,
    max_factors: int = 200,
    min_samples: int = 100,
    top_quantile: float = 0.2,
    cost_quantile: str = "p75",
    quant_lab_commit: str | None = None,
) -> FactorFactorySnapshotPreflight:
    """Resolve stable input identity without writing any Snapshot payload."""

    root = Path(lake_root).resolve(strict=True)
    queue = ensure_research_queue_layout(queue_root)
    commit = quant_lab_commit or _git_commit()
    horizons = tuple(sorted({int(value) for value in horizon_bars if int(value) > 0}))
    if not horizons:
        raise ValueError("factor_factory_horizon_bars_required")
    if decision_delay_bars < 1:
        raise ValueError("factor_factory_decision_delay_invalid")
    indexed = _load_factor_factory_file_index(root)
    feature_identities = _indexed_source_identities(root, indexed, FEATURE_VALUE_DATASET)
    market_identities = _indexed_source_identities(root, indexed, MARKET_BAR_DATASET)
    cost_identities = _indexed_source_identities(root, indexed, COST_BUCKET_DAILY_DATASET)
    feature_sources = tuple(root / item.relative_path for item in feature_identities)
    cost_sources = tuple(root / item.relative_path for item in cost_identities)
    feature_names, feature_min_ts, feature_max_ts = _feature_identity(
        list(feature_sources),
        feature_set=feature_set,
        feature_version=feature_version,
        timeframe=timeframe,
    )
    plan = build_effective_factor_plan(
        feature_names,
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        max_factors=max_factors,
        quant_lab_commit=commit,
        created_at=datetime.combine(as_of_date, time.min, tzinfo=UTC),
    )
    market_before: datetime | None = None
    cost_frame = pl.DataFrame()
    cost_snapshot: tuple[FactorFactoryCostSnapshotRecord, ...] = ()
    selected_market_identities: tuple[FactorFactorySourceFileIdentity, ...] = ()
    selected_cost_identities: tuple[FactorFactorySourceFileIdentity, ...] = ()
    if feature_min_ts is not None:
        if feature_max_ts is None:
            raise RuntimeError("factor_factory_feature_bounds_incomplete")
        market_before = feature_max_ts + _timeframe_delta(timeframe) * (
            decision_delay_bars + max(horizons) + 1
        )
        selected_market_identities = market_identities
        selected_cost_identities = cost_identities
        cost_frame, cost_snapshot = _select_cost_frame(
            list(cost_sources), cost_quantile=cost_quantile
        )
    source_files = tuple(
        sorted(
            (*feature_identities, *selected_market_identities, *selected_cost_identities),
            key=lambda item: (item.dataset_name, item.relative_path),
        )
    )
    source_digest = _projected_source_digest(
        source_files,
        feature_set=feature_set,
        feature_version=feature_version,
        timeframe=timeframe,
        feature_min_ts=feature_min_ts,
        feature_max_ts=feature_max_ts,
        market_before=market_before,
    )
    cost_digest = _projected_cost_digest(
        source_files,
        cost_quantile=cost_quantile,
        cost_snapshot=cost_snapshot,
    )
    identity_payload = {
        "schema_version": "quant_lab_factor_factory_snapshot_identity.v2",
        "quant_lab_commit": commit,
        "factor_plan_digest": plan.plan_digest,
        "source_input_digest": source_digest,
        "cost_input_digest": cost_digest,
        "feature_set": feature_set,
        "feature_version": feature_version,
        "factor_version": factor_version,
        "timeframe": timeframe,
        "horizon_bars": list(horizons),
        "decision_delay_bars": decision_delay_bars,
        "max_factors": max_factors,
        "min_samples": min_samples,
        "top_quantile": top_quantile,
        "cost_quantile": cost_quantile,
        "result_mode": "PARITY_FULL",
        "history_mode": "bootstrap_full",
    }
    snapshot_id = f"factor-factory-{model_content_sha256(identity_payload)[:24]}"
    estimates = {
        dataset: sum(
            item.uncompressed_bytes for item in source_files if item.dataset_name == dataset
        )
        for dataset in (_dataset_name(path) for path in FACTOR_FACTORY_INPUT_DATASETS)
    }
    _assert_source_identities(root, source_files)
    return FactorFactorySnapshotPreflight(
        lake_root=root,
        queue_root=queue,
        as_of_date=as_of_date,
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        horizon_bars=horizons,
        decision_delay_bars=decision_delay_bars,
        max_factors=max_factors,
        min_samples=min_samples,
        top_quantile=top_quantile,
        cost_quantile=cost_quantile,
        quant_lab_commit=commit,
        factor_plan=plan,
        feature_sources=feature_sources,
        market_sources=tuple(root / item.relative_path for item in selected_market_identities),
        cost_sources=tuple(root / item.relative_path for item in selected_cost_identities),
        source_files=source_files,
        feature_min_ts=feature_min_ts,
        feature_max_ts=feature_max_ts,
        market_before=market_before,
        cost_frame=cost_frame,
        cost_snapshot=cost_snapshot,
        source_input_digest=source_digest,
        cost_input_digest=cost_digest,
        identity_payload=identity_payload,
        snapshot_id=snapshot_id,
        feature_estimated_uncompressed_bytes=estimates.get("gold/feature_value", 0),
        market_estimated_uncompressed_bytes=estimates.get("silver/market_bar", 0),
        cost_estimated_uncompressed_bytes=estimates.get("gold/cost_bucket_daily", 0),
        estimated_uncompressed_bytes=sum(estimates.values()),
    )


def materialize_factor_factory_snapshot(
    preflight: FactorFactorySnapshotPreflight,
    *,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    max_input_bytes: int,
    max_input_rows: int,
) -> FactorFactorySnapshotMaterialization:
    """Create, reuse, or strictly rehydrate the preflight-selected Snapshot."""

    queue = preflight.queue_root
    final_root = queue / "snapshots" / preflight.snapshot_id
    with snapshot_payload_lock(queue, preflight.snapshot_id, timeout_seconds=120):
        if final_root.exists():
            manifest = FactorFactorySnapshotManifest.model_validate_json(
                (final_root / "manifest.json").read_text(encoding="utf-8")
            )
            _require_preflight_manifest_identity(preflight, manifest)
            _require_snapshot_capacity(
                manifest,
                max_input_bytes=max_input_bytes,
                max_input_rows=max_input_rows,
            )
            if (final_root / "files").is_dir():
                verify_factor_factory_snapshot_manifest(
                    manifest,
                    final_root=final_root,
                    public_key=signing_key.public_key(),
                )
                _clear_stale_release_marker(final_root)
                _make_read_only(final_root)
                return FactorFactorySnapshotMaterialization(manifest, False, False)
            if not (final_root / "FILES_RELEASED.json").is_file():
                raise RuntimeError("factor_factory_snapshot_payload_missing_without_release")
            _rehydrate_snapshot_payload_locked(
                preflight,
                manifest,
                final_root=final_root,
                public_key=signing_key.public_key(),
                signature_key_id=signature_key_id,
            )
            return FactorFactorySnapshotMaterialization(manifest, True, True)

        temporary = (
            queue / "snapshots" / f".sealing.{preflight.snapshot_id}.{uuid.uuid4().hex}.partial"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        (temporary / "files").mkdir()
        try:
            references, market_min_ts, market_max_ts = _materialize_preflight(preflight, temporary)
            total_bytes = sum(item.size_bytes for item in references)
            total_rows = sum(item.row_count for item in references)
            if total_bytes > max_input_bytes:
                raise RuntimeError("factor_factory_snapshot_input_size_limit_exceeded")
            if total_rows > max_input_rows:
                raise RuntimeError("factor_factory_snapshot_input_row_limit_exceeded")
            provisional = FactorFactorySnapshotManifest(
                snapshot_id=preflight.snapshot_id,
                generated_at=datetime.now(UTC),
                quant_lab_commit=preflight.quant_lab_commit,
                as_of_date=preflight.as_of_date,
                feature_set=preflight.feature_set,
                feature_version=preflight.feature_version,
                factor_version=preflight.factor_version,
                timeframe=preflight.timeframe,
                horizon_bars=preflight.horizon_bars,
                decision_delay_bars=preflight.decision_delay_bars,
                max_factors=preflight.max_factors,
                min_samples=preflight.min_samples,
                top_quantile=preflight.top_quantile,
                cost_quantile=preflight.cost_quantile,
                factor_plan=preflight.factor_plan,
                factor_plan_digest=preflight.factor_plan.plan_digest,
                source_input_digest=preflight.source_input_digest,
                cost_input_digest=preflight.cost_input_digest,
                cost_snapshot=preflight.cost_snapshot,
                feature_min_ts=preflight.feature_min_ts,
                feature_max_ts=preflight.feature_max_ts,
                market_min_ts=market_min_ts,
                market_max_ts=market_max_ts,
                source_files=preflight.source_files,
                feature_estimated_uncompressed_bytes=(
                    preflight.feature_estimated_uncompressed_bytes
                ),
                market_estimated_uncompressed_bytes=(preflight.market_estimated_uncompressed_bytes),
                cost_estimated_uncompressed_bytes=(preflight.cost_estimated_uncompressed_bytes),
                estimated_uncompressed_bytes=preflight.estimated_uncompressed_bytes,
                datasets=[_dataset_name(path) for path in FACTOR_FACTORY_INPUT_DATASETS],
                files=references,
                total_input_bytes=total_bytes,
                total_input_rows=total_rows,
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
            _make_read_only(temporary)
            os.replace(temporary, final_root)
            return FactorFactorySnapshotMaterialization(manifest, True, False)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def rehydrate_factor_factory_snapshot_payload(
    lake_root: str | Path,
    queue_root: str | Path,
    snapshot_id: str,
    *,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    max_input_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_SNAPSHOT_BYTES,
    max_input_rows: int = 150_000_000,
) -> FactorFactorySnapshotManifest:
    """Strictly restore one released v2 Snapshot without changing its manifest."""

    queue = ensure_research_queue_layout(queue_root)
    final_root = queue / "snapshots" / snapshot_id
    manifest = FactorFactorySnapshotManifest.model_validate_json(
        (final_root / "manifest.json").read_text(encoding="utf-8")
    )
    _require_snapshot_capacity(
        manifest,
        max_input_bytes=max_input_bytes,
        max_input_rows=max_input_rows,
    )
    preflight = preflight_factor_factory_snapshot(
        lake_root,
        queue,
        as_of_date=manifest.as_of_date,
        feature_set=manifest.feature_set,
        feature_version=manifest.feature_version,
        factor_version=manifest.factor_version,
        timeframe=manifest.timeframe,
        horizon_bars=manifest.horizon_bars,
        decision_delay_bars=manifest.decision_delay_bars,
        max_factors=manifest.max_factors,
        min_samples=manifest.min_samples,
        top_quantile=manifest.top_quantile,
        cost_quantile=manifest.cost_quantile,
        quant_lab_commit=manifest.quant_lab_commit,
    )
    with snapshot_payload_lock(queue, snapshot_id, timeout_seconds=120):
        if (final_root / "files").is_dir():
            verify_factor_factory_snapshot_manifest(
                manifest,
                final_root=final_root,
                public_key=signing_key.public_key(),
            )
            _clear_stale_release_marker(final_root)
            _make_read_only(final_root)
            return manifest
        _require_preflight_manifest_identity(preflight, manifest)
        _rehydrate_snapshot_payload_locked(
            preflight,
            manifest,
            final_root=final_root,
            public_key=signing_key.public_key(),
            signature_key_id=signature_key_id,
        )
    return manifest


def cleanup_stale_factor_factory_rehydrate_partials(
    queue_root: str | Path,
    *,
    stale_after_seconds: int = REHYDRATE_PARTIAL_STALE_SECONDS,
    now: datetime | None = None,
) -> tuple[str, ...]:
    """Remove abandoned rehydrate work only when its snapshot lock is available."""

    queue = ensure_research_queue_layout(queue_root)
    observed_at = now or datetime.now(UTC)
    removed: list[str] = []
    for partial in sorted((queue / "snapshots").glob(".rehydrate.*.partial")):
        marker_path = partial / "REHYDRATE.json"
        try:
            marker = json.loads(marker_path.read_text("utf-8"))
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
    return tuple(removed)


def _materialize_preflight(
    preflight: FactorFactorySnapshotPreflight,
    temporary: Path,
) -> tuple[list[ResearchDatasetReference], datetime | None, datetime | None]:
    source_sha_by_path = {item.relative_path: item.sha256 for item in preflight.source_files}
    references = _materialize_feature_files(
        preflight.lake_root,
        temporary,
        list(preflight.feature_sources),
        feature_set=preflight.feature_set,
        feature_version=preflight.feature_version,
        timeframe=preflight.timeframe,
        source_sha_by_path=source_sha_by_path,
    )
    market_references: list[ResearchDatasetReference] = []
    if preflight.feature_min_ts is not None:
        if preflight.market_before is None:
            raise RuntimeError("factor_factory_market_boundary_missing")
        market_references = _materialize_market_files(
            preflight.lake_root,
            temporary,
            list(preflight.market_sources),
            since=preflight.feature_min_ts,
            before=preflight.market_before,
            timeframe=preflight.timeframe,
            source_sha_by_path=source_sha_by_path,
        )
        cost_references, observed_cost = _materialize_cost_selection(
            preflight.lake_root,
            temporary,
            list(preflight.cost_sources),
            cost_quantile=preflight.cost_quantile,
            selected_frame=preflight.cost_frame,
        )
        if observed_cost != preflight.cost_snapshot:
            raise RuntimeError("snapshot_source_changed_while_sealing")
        references.extend(market_references)
        references.extend(cost_references)
    _assert_source_identities(preflight.lake_root, preflight.source_files)
    references.sort(key=lambda item: (item.dataset_name, item.relative_path))
    market_min_ts, market_max_ts = _reference_bounds(market_references)
    return references, market_min_ts, market_max_ts


def _rehydrate_snapshot_payload_locked(
    preflight: FactorFactorySnapshotPreflight,
    manifest: FactorFactorySnapshotManifest,
    *,
    final_root: Path,
    public_key: Ed25519PublicKey,
    signature_key_id: str,
) -> None:
    if manifest.schema_version == FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1:
        raise RuntimeError("snapshot_rehydrate_identity_mismatch")
    if manifest.signature_key_id != signature_key_id:
        raise RuntimeError("snapshot_rehydrate_identity_mismatch")
    verify_factor_factory_snapshot_manifest(
        manifest,
        final_root=final_root,
        public_key=public_key,
        require_payload=False,
    )
    _require_preflight_manifest_identity(preflight, manifest)
    for abandoned in preflight.queue_root.joinpath("snapshots").glob(
        f".rehydrate.{manifest.snapshot_id}.*.partial"
    ):
        shutil.rmtree(abandoned, ignore_errors=True)
    partial = (
        preflight.queue_root
        / "snapshots"
        / f".rehydrate.{manifest.snapshot_id}.{uuid.uuid4().hex}.partial"
    )
    partial.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        partial / "REHYDRATE.json",
        {
            "schema_version": "quant_lab_factor_factory_snapshot_rehydrate.v1",
            "snapshot_id": manifest.snapshot_id,
            "manifest_sha256": manifest.manifest_sha256,
            "started_at": datetime.now(UTC).isoformat(),
        },
    )
    (partial / "files").mkdir()
    try:
        references, market_min_ts, market_max_ts = _materialize_preflight(preflight, partial)
        if [item.model_dump(mode="json") for item in references] != [
            item.model_dump(mode="json") for item in manifest.files
        ]:
            raise RuntimeError("snapshot_rehydrate_identity_mismatch")
        if (market_min_ts, market_max_ts) != (manifest.market_min_ts, manifest.market_max_ts):
            raise RuntimeError("snapshot_rehydrate_identity_mismatch")
        _make_tree_writable(final_root)
        os.replace(partial / "files", final_root / "files")
        (final_root / "FILES_RELEASED.json").unlink(missing_ok=True)
        atomic_write_json(
            final_root / "FILES_REHYDRATED.json",
            {
                "schema_version": "quant_lab_factor_factory_snapshot_rehydrated.v1",
                "snapshot_id": manifest.snapshot_id,
                "manifest_sha256": manifest.manifest_sha256,
                "rehydrated_at": datetime.now(UTC).isoformat(),
                "state": "rehydrated",
            },
        )
        verify_factor_factory_snapshot_manifest(
            manifest,
            final_root=final_root,
            public_key=public_key,
        )
        _make_read_only(final_root)
        _append_snapshot_audit(
            preflight.queue_root,
            {
                "action": "snapshot_rehydrated",
                "snapshot_id": manifest.snapshot_id,
                "manifest_sha256": manifest.manifest_sha256,
            },
        )
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc) == "snapshot_rehydrate_identity_mismatch":
            raise
        raise RuntimeError("snapshot_rehydrate_identity_mismatch") from exc
    finally:
        shutil.rmtree(partial, ignore_errors=True)


def _clear_stale_release_marker(final_root: Path) -> None:
    """Repair a crash between payload install and release-marker removal."""

    marker = final_root / "FILES_RELEASED.json"
    if not marker.exists():
        return
    _make_tree_writable(final_root)
    marker.unlink()
    _make_read_only(final_root)


def _require_snapshot_capacity(
    manifest: FactorFactorySnapshotManifest,
    *,
    max_input_bytes: int,
    max_input_rows: int,
) -> None:
    if manifest.total_input_bytes > max_input_bytes:
        raise RuntimeError("factor_factory_snapshot_input_size_limit_exceeded")
    if manifest.total_input_rows > max_input_rows:
        raise RuntimeError("factor_factory_snapshot_input_row_limit_exceeded")


def _require_preflight_manifest_identity(
    preflight: FactorFactorySnapshotPreflight,
    manifest: FactorFactorySnapshotManifest,
) -> None:
    observed = {
        "schema_version": "quant_lab_factor_factory_snapshot_identity.v2",
        "quant_lab_commit": manifest.quant_lab_commit,
        "factor_plan_digest": manifest.factor_plan_digest,
        "source_input_digest": manifest.source_input_digest,
        "cost_input_digest": manifest.cost_input_digest,
        "feature_set": manifest.feature_set,
        "feature_version": manifest.feature_version,
        "factor_version": manifest.factor_version,
        "timeframe": manifest.timeframe,
        "horizon_bars": list(manifest.horizon_bars),
        "decision_delay_bars": manifest.decision_delay_bars,
        "max_factors": manifest.max_factors,
        "min_samples": manifest.min_samples,
        "top_quantile": manifest.top_quantile,
        "cost_quantile": manifest.cost_quantile,
        "result_mode": manifest.result_mode,
        "history_mode": manifest.history_mode,
    }
    if (
        manifest.schema_version != FACTOR_FACTORY_SNAPSHOT_SCHEMA
        or manifest.snapshot_id != preflight.snapshot_id
        or observed != preflight.identity_payload
        or [_identity_payload(item) for item in manifest.source_files]
        != [_identity_payload(item) for item in preflight.source_files]
        or manifest.cost_snapshot != preflight.cost_snapshot
        or manifest.estimated_uncompressed_bytes != preflight.estimated_uncompressed_bytes
    ):
        raise RuntimeError("snapshot_rehydrate_identity_mismatch")


def _append_snapshot_audit(queue: Path, payload: dict[str, Any]) -> None:
    path = queue / "audit" / "factor_factory_snapshot.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"observed_at": datetime.now(UTC).isoformat(), **payload},
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n"
        )


def verify_factor_factory_snapshot_manifest(
    manifest: FactorFactorySnapshotManifest,
    *,
    final_root: Path | None = None,
    public_key: Ed25519PublicKey | None = None,
    require_payload: bool = True,
) -> None:
    expected = _factor_factory_manifest_content_sha256(manifest)
    if expected != manifest.manifest_sha256:
        raise ValueError("factor_factory_snapshot_manifest_digest_mismatch")
    if public_key is not None:
        if manifest.schema_version == FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1:
            verify_payload(
                _legacy_factor_factory_manifest_payload(manifest),
                manifest.signature,
                public_key,
            )
        else:
            verify_payload(manifest, manifest.signature, public_key)
    expected_datasets = {_dataset_name(path) for path in FACTOR_FACTORY_INPUT_DATASETS}
    if set(manifest.datasets) != expected_datasets:
        raise ValueError("factor_factory_snapshot_dataset_set_mismatch")
    if final_root is None:
        return
    root = final_root.resolve(strict=True)
    if (root / "SEALED").read_text(encoding="ascii").strip() != manifest.manifest_sha256:
        raise ValueError("factor_factory_snapshot_seal_mismatch")
    files_root = root / "files"
    if not files_root.is_dir():
        if not require_payload and (root / "FILES_RELEASED.json").is_file():
            return
        raise FileNotFoundError(f"factor factory snapshot payload missing: {manifest.snapshot_id}")
    for reference in manifest.files:
        unresolved = files_root / reference.relative_path
        if _path_has_symlink(root, unresolved):
            raise ValueError("factor_factory_snapshot_path_escape")
        candidate = unresolved.resolve(strict=True)
        if root not in candidate.parents:
            raise ValueError("factor_factory_snapshot_path_escape")
        if candidate.stat().st_size != reference.size_bytes:
            raise ValueError("factor_factory_snapshot_size_mismatch")
        if sha256_file(candidate) != reference.sha256:
            raise ValueError("factor_factory_snapshot_sha256_mismatch")
        if reference.schema_fingerprint is not None:
            if _schema_fingerprint(pl.read_parquet_schema(candidate)) != (
                reference.schema_fingerprint
            ):
                raise ValueError("factor_factory_snapshot_schema_mismatch")
        if reference.uncompressed_bytes is not None:
            if _parquet_uncompressed_bytes(candidate) != reference.uncompressed_bytes:
                raise ValueError("factor_factory_snapshot_uncompressed_size_mismatch")
        if _parquet_row_count(candidate) != reference.row_count:
            raise ValueError("factor_factory_snapshot_row_count_mismatch")
        if _parquet_bounds(candidate, "ts") != (reference.min_ts, reference.max_ts):
            raise ValueError("factor_factory_snapshot_bounds_mismatch")


def _factor_factory_manifest_content_sha256(
    manifest: FactorFactorySnapshotManifest,
) -> str:
    if manifest.schema_version == FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1:
        return model_content_sha256(
            _legacy_factor_factory_manifest_payload(manifest),
            blank_fields=("manifest_sha256",),
        )
    return model_content_sha256(manifest, blank_fields=("manifest_sha256",))


def _legacy_factor_factory_manifest_payload(
    manifest: FactorFactorySnapshotManifest,
) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    for name in (
        "source_files",
        "feature_estimated_uncompressed_bytes",
        "market_estimated_uncompressed_bytes",
        "cost_estimated_uncompressed_bytes",
        "estimated_uncompressed_bytes",
    ):
        payload.pop(name, None)
    for item in payload.get("files", []):
        item.pop("schema_fingerprint", None)
        item.pop("uncompressed_bytes", None)
    return payload


def load_factor_factory_generation_binding(
    root: Path,
) -> tuple[str | None, str | None, FactorFactoryPreviousGeneration | None]:
    pointer = root / "gold" / "factor_factory_generation.json"
    if not pointer.exists():
        return None, None, None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("factor_factory_generation_pointer_invalid") from exc
    generation_id = str(payload.get("generation_id") or "").strip()
    generation_digest = str(payload.get("generation_digest") or "").strip()
    if not generation_id or len(generation_digest) != 64:
        raise RuntimeError("factor_factory_generation_pointer_incomplete")
    fields = {
        "schema_version",
        "generation_id",
        "generation_digest",
        "task_id",
        "snapshot_id",
        "quant_lab_commit",
        "factor_plan_digest",
        "source_input_digest",
        "cost_input_digest",
        "feature_set",
        "feature_version",
        "factor_version",
        "timeframe",
        "as_of_date",
        "row_counts",
        "dataset_hashes",
        "published_at",
        "diagnostic_only",
        "research_only",
        "live_order_effect",
        "automatic_promotion",
        "max_live_notional_usdt",
    }
    try:
        manifest = FactorFactoryPreviousGeneration.model_validate(
            {field: payload.get(field) for field in fields}
        )
    except ValueError as exc:
        raise RuntimeError("factor_factory_generation_pointer_incomplete") from exc
    return generation_id, generation_digest, manifest


def _load_factor_factory_file_index(root: Path) -> pl.DataFrame:
    names = [_dataset_name(path) for path in FACTOR_FACTORY_INPUT_DATASETS]
    build_lake_file_index(root, names)
    index = read_parquet_dataset(root / LAKE_FILE_INDEX)
    required = {
        "dataset",
        "path",
        "min_ts",
        "max_ts",
        "row_count",
        "size_bytes",
        "mtime_ns",
        "sha256",
        "schema_fingerprint",
        "uncompressed_bytes",
    }
    if index.is_empty():
        return pl.DataFrame(schema={name: pl.Utf8 for name in sorted(required)})
    if not required.issubset(index.columns):
        raise RuntimeError("lake_file_index_missing_or_invalid")
    return index


def _indexed_source_identities(
    root: Path,
    index: pl.DataFrame,
    dataset: Path,
) -> tuple[FactorFactorySourceFileIdentity, ...]:
    name = _dataset_name(dataset)
    if index.is_empty():
        return ()
    identities: list[FactorFactorySourceFileIdentity] = []
    dataset_root = root / dataset
    if not dataset_root.exists():
        return ()
    resolved_dataset_root = dataset_root.resolve(strict=True)
    for row in index.filter(pl.col("dataset") == name).sort("path").to_dicts():
        relative = str(row.get("path") or "")
        unresolved = root / relative
        if not relative or _path_has_symlink(root, unresolved):
            raise ValueError("lake_file_index_path_escape")
        candidate = unresolved.resolve(strict=True)
        if (
            root not in resolved_dataset_root.parents
            or resolved_dataset_root not in candidate.parents
        ):
            raise ValueError("lake_file_index_path_escape")
        stat = candidate.stat()
        if stat.st_size != int(row["size_bytes"]) or stat.st_mtime_ns != int(row["mtime_ns"]):
            raise RuntimeError(f"lake_file_index_source_changed:{relative}")
        identities.append(
            FactorFactorySourceFileIdentity(
                dataset_name=name,
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


def _identity_payload(item: FactorFactorySourceFileIdentity) -> dict[str, Any]:
    return item.model_dump(
        mode="json",
        exclude={"mtime_ns"},
    )


def _projected_source_digest(
    source_files: tuple[FactorFactorySourceFileIdentity, ...],
    *,
    feature_set: str,
    feature_version: str,
    timeframe: str,
    feature_min_ts: datetime | None,
    feature_max_ts: datetime | None,
    market_before: datetime | None,
) -> str:
    allowed = {_dataset_name(FEATURE_VALUE_DATASET), _dataset_name(MARKET_BAR_DATASET)}
    return model_content_sha256(
        {
            "schema_version": "quant_lab_factor_factory_projected_source_identity.v2",
            "projection_version": FACTOR_FACTORY_PROJECTION_VERSION,
            "filters": {
                "feature_set": feature_set,
                "feature_version": feature_version,
                "timeframe": timeframe,
                "feature_is_valid": True,
                "market_is_closed": True,
                "feature_min_ts": feature_min_ts,
                "feature_max_ts": feature_max_ts,
                "market_before": market_before,
            },
            "projected_columns": {
                "feature": list(FEATURE_VALUE_COLUMNS),
                "market": list(MARKET_BAR_COLUMNS),
            },
            "source_files": [
                _identity_payload(item) for item in source_files if item.dataset_name in allowed
            ],
        }
    )


def _projected_cost_digest(
    source_files: tuple[FactorFactorySourceFileIdentity, ...],
    *,
    cost_quantile: str,
    cost_snapshot: tuple[FactorFactoryCostSnapshotRecord, ...],
) -> str:
    dataset = _dataset_name(COST_BUCKET_DAILY_DATASET)
    return model_content_sha256(
        {
            "schema_version": "quant_lab_factor_factory_projected_cost_identity.v2",
            "projection_version": FACTOR_FACTORY_PROJECTION_VERSION,
            "filter": {
                "cost_quantile": cost_quantile,
                "selection": "latest_per_symbol",
            },
            "projected_columns": list(COST_COLUMNS),
            "source_files": [
                _identity_payload(item) for item in source_files if item.dataset_name == dataset
            ],
            "selected_cost_rows": [item.model_dump(mode="json") for item in cost_snapshot],
        }
    )


def _assert_source_identities(
    root: Path,
    source_files: tuple[FactorFactorySourceFileIdentity, ...],
) -> None:
    for identity in source_files:
        path = root / identity.relative_path
        try:
            stat = path.stat()
        except OSError as exc:
            raise RuntimeError("snapshot_source_changed_while_sealing") from exc
        if (stat.st_size, stat.st_mtime_ns) != (identity.size_bytes, identity.mtime_ns):
            raise RuntimeError("snapshot_source_changed_while_sealing")


def _indexed_dataset_files(root: Path, index: pl.DataFrame, dataset: Path) -> list[Path]:
    name = _dataset_name(dataset)
    files: list[Path] = []
    if index.is_empty():
        return files
    for value in index.filter(pl.col("dataset") == name).get_column("path").to_list():
        relative = str(value or "")
        if not relative:
            continue
        unresolved = root / relative
        if _path_has_symlink(root, unresolved):
            raise ValueError("lake_file_index_path_escape")
        candidate = unresolved.resolve(strict=True)
        dataset_root = (root / dataset).resolve(strict=True)
        if root not in dataset_root.parents or dataset_root not in candidate.parents:
            raise ValueError("lake_file_index_path_escape")
        if candidate.suffix != ".parquet":
            raise ValueError("lake_file_index_non_parquet")
        files.append(candidate)
    return sorted(set(files))


def _feature_identity(
    sources: list[Path],
    *,
    feature_set: str,
    feature_version: str,
    timeframe: str,
) -> tuple[list[str], datetime | None, datetime | None]:
    if not sources:
        return [], None, None
    _require_columns(
        sources,
        {
            "feature_set",
            "feature_name",
            "feature_version",
            "timeframe",
            "symbol",
            "ts",
            "value",
            "is_valid",
        },
        "feature_value",
    )
    lazy = _scan_sources(sources).filter(
        (pl.col("feature_set") == feature_set)
        & (pl.col("feature_version") == feature_version)
        & (pl.col("timeframe") == timeframe)
    )
    lazy = lazy.filter(pl.col("is_valid").cast(pl.Boolean, strict=False).fill_null(False))
    identity = lazy.select(
        _utc_expr("ts").min().alias("min_ts"),
        _utc_expr("ts").max().alias("max_ts"),
    ).collect(engine="streaming")
    names = (
        lazy.select(pl.col("feature_name").cast(pl.Utf8))
        .unique()
        .sort("feature_name")
        .collect(engine="streaming")
        .get_column("feature_name")
        .drop_nulls()
        .to_list()
    )
    return names, _as_utc(identity.item(0, "min_ts")), _as_utc(identity.item(0, "max_ts"))


def _materialize_feature_files(
    root: Path,
    temporary: Path,
    sources: list[Path],
    *,
    feature_set: str,
    feature_version: str,
    timeframe: str,
    source_sha_by_path: dict[str, str] | None = None,
) -> list[ResearchDatasetReference]:
    references: list[ResearchDatasetReference] = []
    for ordinal, source in enumerate(sources):
        schema = pl.read_parquet_schema(source)
        columns = [column for column in FEATURE_VALUE_COLUMNS if column in schema]
        lazy = pl.scan_parquet(source).filter(
            (pl.col("feature_set") == feature_set)
            & (pl.col("feature_version") == feature_version)
            & (pl.col("timeframe") == timeframe)
        )
        if "is_valid" in schema:
            lazy = lazy.filter(pl.col("is_valid").cast(pl.Boolean, strict=False).fill_null(False))
        reference = _materialize_lazy_source(
            root,
            temporary,
            source,
            dataset=FEATURE_VALUE_DATASET,
            ordinal=ordinal,
            lazy=lazy.select(columns),
            source_sha=(source_sha_by_path or {}).get(
                str(source.relative_to(root)).replace("\\", "/")
            ),
        )
        if reference is not None:
            references.append(reference)
    return references


def _materialize_market_files(
    root: Path,
    temporary: Path,
    sources: list[Path],
    *,
    since: datetime,
    before: datetime,
    timeframe: str,
    source_sha_by_path: dict[str, str] | None = None,
) -> list[ResearchDatasetReference]:
    if not sources:
        return []
    _require_columns(
        sources,
        {"symbol", "timeframe", "ts", "close", "is_closed"},
        "market_bar",
    )
    references: list[ResearchDatasetReference] = []
    for ordinal, source in enumerate(sources):
        schema = pl.read_parquet_schema(source)
        columns = [column for column in MARKET_BAR_COLUMNS if column in schema]
        lazy = pl.scan_parquet(source).filter(
            (pl.col("timeframe") == timeframe)
            & (_utc_expr("ts") >= since)
            & (_utc_expr("ts") < before)
        )
        lazy = lazy.filter(pl.col("is_closed").cast(pl.Boolean, strict=False).fill_null(False))
        reference = _materialize_lazy_source(
            root,
            temporary,
            source,
            dataset=MARKET_BAR_DATASET,
            ordinal=ordinal,
            lazy=lazy.select(columns),
            source_sha=(source_sha_by_path or {}).get(
                str(source.relative_to(root)).replace("\\", "/")
            ),
        )
        if reference is not None:
            references.append(reference)
    return references


def _materialize_cost_selection(
    root: Path,
    temporary: Path,
    sources: list[Path],
    *,
    cost_quantile: str,
    selected_frame: pl.DataFrame | None = None,
) -> tuple[list[ResearchDatasetReference], tuple[FactorFactoryCostSnapshotRecord, ...]]:
    if not sources or (selected_frame is not None and selected_frame.is_empty()):
        return [], ()
    source_stats = {source: source.stat() for source in sources}
    if selected_frame is None:
        selected_frame, records = _select_cost_frame(sources, cost_quantile=cost_quantile)
    else:
        records = _cost_snapshot_records(selected_frame, cost_quantile=cost_quantile)
    destination = temporary / "files" / COST_BUCKET_DAILY_DATASET / "part-selected.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected_frame.lazy().sink_parquet(destination, compression="zstd")
    for source, before in source_stats.items():
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("snapshot_source_changed_while_sealing")
    rows = _parquet_row_count(destination)
    if rows == 0:
        destination.unlink()
        return [], ()
    destination.chmod(0o440)
    stat = destination.stat()
    reference = ResearchDatasetReference(
        dataset_name=_dataset_name(COST_BUCKET_DAILY_DATASET),
        source_relative_path=_dataset_name(COST_BUCKET_DAILY_DATASET),
        relative_path=f"{_dataset_name(COST_BUCKET_DAILY_DATASET)}/part-selected.parquet",
        sha256=sha256_file(destination),
        size_bytes=stat.st_size,
        row_count=rows,
        mtime_ns=0,
        schema_fingerprint=_schema_fingerprint(pl.read_parquet_schema(destination)),
        uncompressed_bytes=_parquet_uncompressed_bytes(destination),
    )
    return [reference], records


def _select_cost_frame(
    sources: list[Path],
    *,
    cost_quantile: str,
) -> tuple[pl.DataFrame, tuple[FactorFactoryCostSnapshotRecord, ...]]:
    if not sources:
        return pl.DataFrame(), ()
    cost_column = f"total_cost_bps_{cost_quantile}"
    _require_columns(sources, {"symbol", cost_column}, "cost_bucket_daily")
    schemas = [pl.read_parquet_schema(source) for source in sources]
    available = set.intersection(*(set(schema) for schema in schemas))
    if "day" not in available and "as_of_date" not in available:
        raise ValueError("factor_factory_cost_bucket_daily_columns_missing:day_or_as_of_date")
    columns = [column for column in COST_COLUMNS if column in available]
    day_column = "day" if "day" in available else "as_of_date"
    selected = (
        _scan_sources(sources)
        .sort(day_column)
        .select(columns)
        .unique(subset=["symbol"], keep="last")
        .sort("symbol")
        .collect(engine="streaming")
    )
    return selected, _cost_snapshot_records(selected, cost_quantile=cost_quantile)


def _cost_snapshot_records(
    selected_frame: pl.DataFrame,
    *,
    cost_quantile: str,
) -> tuple[FactorFactoryCostSnapshotRecord, ...]:
    if selected_frame.is_empty():
        return ()
    cost_column = f"total_cost_bps_{cost_quantile}"
    date_column = "day" if "day" in selected_frame.columns else "as_of_date"
    source_column = "cost_source" if "cost_source" in selected_frame.columns else "source"
    return tuple(
        FactorFactoryCostSnapshotRecord(
            symbol=str(row["symbol"]),
            cost_date=(str(row[date_column]) if row.get(date_column) is not None else None),
            cost_model_version=str(row.get("cost_model_version") or "unknown"),
            cost_source=str(row.get(source_column) or "unknown"),
            cost_quantile=cost_quantile,
            cost_bps=float(row[cost_column]),
        )
        for row in selected_frame.sort("symbol").to_dicts()
    )


def _materialize_lazy_source(
    root: Path,
    temporary: Path,
    source: Path,
    *,
    dataset: Path,
    ordinal: int,
    lazy: pl.LazyFrame,
    source_sha: str | None = None,
) -> ResearchDatasetReference | None:
    before = source.stat()
    source_relative = str(source.relative_to(root)).replace("\\", "/")
    part_id = model_content_sha256(
        {
            "source": source_relative,
            "ordinal": ordinal,
            "source_sha256": source_sha or sha256_file(source),
        }
    )[:16]
    relative_path = f"{_dataset_name(dataset)}/part-{part_id}.parquet"
    destination = temporary / "files" / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    lazy.sink_parquet(destination, compression="zstd")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("snapshot_source_changed_while_sealing")
    row_count = _parquet_row_count(destination)
    if row_count == 0:
        destination.unlink()
        return None
    destination.chmod(0o440)
    min_ts, max_ts = _parquet_bounds(destination, "ts")
    stat = destination.stat()
    return ResearchDatasetReference(
        dataset_name=_dataset_name(dataset),
        source_relative_path=source_relative,
        relative_path=relative_path,
        sha256=sha256_file(destination),
        size_bytes=stat.st_size,
        row_count=row_count,
        mtime_ns=0,
        min_ts=min_ts,
        max_ts=max_ts,
        schema_fingerprint=_schema_fingerprint(pl.read_parquet_schema(destination)),
        uncompressed_bytes=_parquet_uncompressed_bytes(destination),
    )


def _references_digest(
    references: list[ResearchDatasetReference],
    *,
    datasets: set[str],
) -> str:
    return model_content_sha256(
        {
            "schema_version": "quant_lab.factor_factory_input_identity.v1",
            "files": [
                item.model_dump(mode="json") for item in references if item.dataset_name in datasets
            ],
        }
    )


def _reference_bounds(
    references: list[ResearchDatasetReference],
) -> tuple[datetime | None, datetime | None]:
    minima = [item.min_ts for item in references if item.min_ts is not None]
    maxima = [item.max_ts for item in references if item.max_ts is not None]
    return (min(minima) if minima else None, max(maxima) if maxima else None)


def _require_columns(sources: list[Path], required: set[str], dataset: str) -> None:
    for source in sources:
        missing = sorted(required - set(pl.read_parquet_schema(source)))
        if missing:
            raise ValueError(f"factor_factory_{dataset}_columns_missing:{','.join(missing)}")


def _scan_sources(sources: list[Path]) -> pl.LazyFrame:
    return pl.scan_parquet([str(path) for path in sources])


def _parquet_row_count(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item())


def _parquet_bounds(path: Path, column: str) -> tuple[datetime | None, datetime | None]:
    schema = pl.read_parquet_schema(path)
    if column not in schema:
        return None, None
    frame = (
        pl.scan_parquet(path)
        .select(_utc_expr(column).min().alias("minimum"), _utc_expr(column).max().alias("maximum"))
        .collect(engine="streaming")
    )
    return _as_utc(frame.item(0, "minimum")), _as_utc(frame.item(0, "maximum"))


def _utc_expr(column: str) -> pl.Expr:
    return pl.coalesce(
        pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
        pl.col(column).cast(pl.Utf8, strict=False).str.to_datetime(time_zone="UTC", strict=False),
    )


def _as_utc(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _timeframe_delta(timeframe: str) -> timedelta:
    value = timeframe.strip().lower()
    if len(value) < 2 or not value[:-1].isdigit():
        raise ValueError(f"unsupported factor factory timeframe: {timeframe}")
    amount = int(value[:-1])
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    unit = value[-1]
    if amount <= 0 or unit not in multipliers:
        raise ValueError(f"unsupported factor factory timeframe: {timeframe}")
    return timedelta(seconds=amount * multipliers[unit])


def _dataset_name(path: Path) -> str:
    return str(path).replace("\\", "/")


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


def _make_read_only(path: Path) -> None:
    for candidate in sorted(path.rglob("*"), reverse=True):
        try:
            candidate.chmod(0o440 if candidate.is_file() else 0o550)
        except OSError:
            pass
    try:
        path.chmod(0o550)
    except OSError:
        pass


def _make_tree_writable(path: Path) -> None:
    path.chmod(0o750)
    for candidate in path.rglob("*"):
        candidate.chmod(0o640 if candidate.is_file() else 0o750)


def _parquet_uncompressed_bytes(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(row_group).column(column).total_uncompressed_size
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _schema_fingerprint(schema: Any) -> str:
    payload = json.dumps(
        [(str(name), str(dtype)) for name, dtype in schema.items()],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
