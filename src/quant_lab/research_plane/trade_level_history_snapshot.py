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

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.research.candidate_labels import (
    CANDIDATE_LABEL_DATASET,
    LABEL_SCHEMA,
)
from quant_lab.research.candidate_labels import (
    SOURCE_NAME as CANDIDATE_LABEL_SOURCE,
)
from quant_lab.research_plane.signatures import (
    model_content_sha256,
    sha256_file,
    sign_model,
    verify_payload,
)
from quant_lab.research_plane.snapshot_lock import snapshot_payload_lock
from quant_lab.research_plane.status import ensure_research_queue_layout
from quant_lab.research_plane.trade_level_history_contracts import (
    TRADE_LEVEL_HISTORY_INPUT_DATASETS,
    TRADE_LEVEL_HISTORY_RISK_JOIN_VERSION,
    TradeLevelHistoryInputFingerprint,
    TradeLevelHistorySnapshotFile,
    TradeLevelHistorySnapshotManifest,
    TradeLevelHistoryTaskPayload,
)
from quant_lab.research_plane.v5_candidate_evidence_publish import (
    V5_CANDIDATE_EVIDENCE_GENERATION_POINTER,
    verify_v5_candidate_evidence_generation_fast,
)
from quant_lab.symbols import normalize_symbol
from quant_lab.trade_level.judgment import (
    RISK_PERMISSION_DATASET,
    TRADE_OPPORTUNITY_EVENT_SCHEMA,
    V5_CANDIDATE_EVENT_DATASET,
    V5_ORDER_LIFECYCLE_DATASET,
    V5_TRADE_EVENT_DATASET,
    build_trade_opportunity_events,
    event_id_for_row,
)

TRADE_LEVEL_HISTORY_REHYDRATE_PARTIAL_STALE_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class TradeLevelHistorySnapshotPreflight:
    lake_root: Path
    queue_root: Path
    parameters: TradeLevelHistoryTaskPayload
    quant_lab_commit: str
    events: pl.DataFrame
    candidate_labels: pl.DataFrame
    candidate_event_digest: str
    risk_permission_digest: str
    v5_trade_event_digest: str
    order_lifecycle_digest: str
    input_fingerprint: TradeLevelHistoryInputFingerprint
    identity_payload: dict[str, Any]
    snapshot_id: str


@dataclass(frozen=True)
class TradeLevelHistorySnapshotMaterialization:
    manifest: TradeLevelHistorySnapshotManifest
    snapshot_materialized: bool
    snapshot_rehydrated: bool


def preflight_trade_level_history_snapshot(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    quant_lab_commit: str,
) -> TradeLevelHistorySnapshotPreflight:
    """Derive and validate immutable full-history inputs before materialization."""

    root = Path(lake_root).resolve(strict=True)
    queue = ensure_research_queue_layout(queue_root)
    pointer = _load_candidate_evidence_pointer(root)
    candidate_generation_id = _required_text(pointer, "generation_id")
    candidate_generation_digest = _required_sha(pointer, "generation_digest")
    candidate_input_fingerprint = _required_sha(pointer, "input_fingerprint_digest")
    verify_v5_candidate_evidence_generation_fast(
        root,
        candidate_generation_id,
        expected_input_fingerprint=candidate_input_fingerprint,
    )
    parameters = TradeLevelHistoryTaskPayload(
        as_of_date=as_of_date,
        candidate_evidence_generation_id=candidate_generation_id,
        candidate_evidence_generation_digest=candidate_generation_digest,
        candidate_evidence_input_fingerprint=candidate_input_fingerprint,
    )

    source_paths = {
        "candidate_event": root / V5_CANDIDATE_EVENT_DATASET,
        "risk_permission": root / RISK_PERMISSION_DATASET,
        "v5_trade_event": root / V5_TRADE_EVENT_DATASET,
        "order_lifecycle": root / V5_ORDER_LIFECYCLE_DATASET,
        "candidate_label": root / CANDIDATE_LABEL_DATASET,
    }
    for dataset_name, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"trade_level_history_source_missing:{dataset_name}")

    candidate_events = read_parquet_dataset(source_paths["candidate_event"])
    risk_permissions = read_parquet_dataset(source_paths["risk_permission"])
    v5_trades = read_parquet_dataset(source_paths["v5_trade_event"])
    order_lifecycles = read_parquet_dataset(source_paths["order_lifecycle"])
    candidate_labels = _candidate_generation_labels(
        read_parquet_dataset(source_paths["candidate_label"]),
        pointer,
    )
    events = build_trade_opportunity_events(
        candidate_events,
        risk_permissions=risk_permissions,
        v5_trades=v5_trades,
        order_lifecycles=order_lifecycles,
        created_at=datetime.combine(as_of_date, time.min, tzinfo=UTC),
    )
    _validate_trade_opportunity_events(
        events,
        as_of_date=as_of_date,
    )

    candidate_event_digest = _frame_digest("silver/v5_candidate_event", candidate_events)
    risk_permission_digest = _frame_digest("gold/risk_permission", risk_permissions)
    v5_trade_event_digest = _frame_digest("silver/v5_trade_event", v5_trades)
    order_lifecycle_digest = _frame_digest(
        "silver/v5_order_lifecycle",
        order_lifecycles,
    )
    derived_event_digest = _frame_digest("cloud/trade_opportunity_event", events)
    candidate_label_dataset_hash = _candidate_label_dataset_hash(pointer)
    event_min_ts, event_max_ts = _frame_time_bounds(events, ("decision_ts",))
    candidate_label_min_ts, candidate_label_max_ts = _frame_time_bounds(
        candidate_labels,
        ("decision_ts", "ts_utc", "label_ts"),
    )
    fingerprint_identity = {
        "schema_version": "trade_level_history_input_identity.v1",
        "quant_lab_commit": quant_lab_commit,
        "as_of_date": as_of_date,
        "history_mode": parameters.history_mode,
        "trade_event_schema_version": parameters.trade_event_schema_version,
        "trade_label_schema_version": parameters.trade_label_schema_version,
        "similarity_schema_version": parameters.similarity_schema_version,
        "similarity_availability_policy": parameters.similarity_availability_policy,
        "derived_event_digest": derived_event_digest,
        "candidate_label_dataset_hash": candidate_label_dataset_hash,
        "candidate_evidence_generation_id": candidate_generation_id,
        "candidate_evidence_generation_digest": candidate_generation_digest,
        "candidate_evidence_input_fingerprint": candidate_input_fingerprint,
        "event_row_count": events.height,
        "candidate_label_row_count": candidate_labels.height,
        "event_min_ts": event_min_ts,
        "event_max_ts": event_max_ts,
        "candidate_label_min_ts": candidate_label_min_ts,
        "candidate_label_max_ts": candidate_label_max_ts,
    }
    fingerprint_digest = model_content_sha256(fingerprint_identity)
    fingerprint = TradeLevelHistoryInputFingerprint(
        **parameters.model_dump(),
        quant_lab_commit=quant_lab_commit,
        derived_event_digest=derived_event_digest,
        candidate_label_dataset_hash=candidate_label_dataset_hash,
        event_row_count=events.height,
        candidate_label_row_count=candidate_labels.height,
        event_min_ts=event_min_ts,
        event_max_ts=event_max_ts,
        candidate_label_min_ts=candidate_label_min_ts,
        candidate_label_max_ts=candidate_label_max_ts,
        input_fingerprint_digest=fingerprint_digest,
        observed_at=datetime.now(UTC),
    )
    snapshot_identity = {
        "schema_version": "trade_level_history_snapshot_identity.v1",
        "derived_event_digest": derived_event_digest,
        "candidate_label_dataset_hash": candidate_label_dataset_hash,
        "candidate_evidence_generation_id": candidate_generation_id,
        "candidate_evidence_generation_digest": candidate_generation_digest,
        "candidate_evidence_input_fingerprint": candidate_input_fingerprint,
        "trade_event_schema_version": parameters.trade_event_schema_version,
        "trade_label_schema_version": parameters.trade_label_schema_version,
        "similarity_schema_version": parameters.similarity_schema_version,
        "similarity_availability_policy": parameters.similarity_availability_policy,
        "quant_lab_commit": quant_lab_commit,
        "history_mode": parameters.history_mode,
    }
    snapshot_id = "trade-level-history-" + model_content_sha256(snapshot_identity)[:24]
    return TradeLevelHistorySnapshotPreflight(
        lake_root=root,
        queue_root=queue,
        parameters=parameters,
        quant_lab_commit=quant_lab_commit,
        events=events,
        candidate_labels=candidate_labels,
        candidate_event_digest=candidate_event_digest,
        risk_permission_digest=risk_permission_digest,
        v5_trade_event_digest=v5_trade_event_digest,
        order_lifecycle_digest=order_lifecycle_digest,
        input_fingerprint=fingerprint,
        identity_payload=snapshot_identity,
        snapshot_id=snapshot_id,
    )


def materialize_trade_level_history_snapshot(
    preflight: TradeLevelHistorySnapshotPreflight,
    *,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    max_input_bytes: int,
    max_input_uncompressed_bytes: int,
    max_input_rows: int,
) -> TradeLevelHistorySnapshotMaterialization:
    queue = preflight.queue_root
    final_root = queue / "snapshots" / preflight.snapshot_id
    with snapshot_payload_lock(queue, preflight.snapshot_id, timeout_seconds=120):
        if final_root.is_dir() and (final_root / "files").is_dir():
            manifest = TradeLevelHistorySnapshotManifest.model_validate_json(
                (final_root / "manifest.json").read_text("utf-8")
            )
            verify_trade_level_history_snapshot_manifest(
                manifest,
                final_root=final_root,
                public_key=signing_key.public_key(),
            )
            if (
                manifest.input_fingerprint_digest
                != preflight.input_fingerprint.input_fingerprint_digest
            ):
                raise RuntimeError("trade_level_history_snapshot_identity_mismatch")
            return TradeLevelHistorySnapshotMaterialization(manifest, False, False)

        rehydrating = final_root.is_dir() and (final_root / "FILES_RELEASED.json").is_file()
        retained: TradeLevelHistorySnapshotManifest | None = None
        if rehydrating:
            retained = TradeLevelHistorySnapshotManifest.model_validate_json(
                (final_root / "manifest.json").read_text("utf-8")
            )
            verify_trade_level_history_snapshot_manifest(
                retained,
                final_root=final_root,
                public_key=signing_key.public_key(),
                require_payload=False,
            )
            if (
                retained.snapshot_id != preflight.snapshot_id
                or retained.input_fingerprint_digest
                != preflight.input_fingerprint.input_fingerprint_digest
            ):
                raise RuntimeError("trade_level_history_snapshot_rehydrate_identity_mismatch")

        token = uuid.uuid4().hex
        prefix = "rehydrate" if rehydrating else "sealing"
        temporary = queue / "snapshots" / f".{prefix}.{preflight.snapshot_id}.{token}.partial"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            if retained is not None:
                atomic_write_json(
                    temporary / "REHYDRATE.json",
                    {
                        "schema_version": "quant_lab_trade_level_history_snapshot_rehydrate.v1",
                        "snapshot_id": retained.snapshot_id,
                        "manifest_sha256": retained.manifest_sha256,
                        "started_at": datetime.now(UTC).isoformat(),
                    },
                )
            references = _materialize_snapshot_files(preflight, temporary)
            total_bytes = sum(item.size_bytes for item in references)
            total_rows = sum(item.row_count for item in references)
            total_uncompressed = sum(item.uncompressed_bytes for item in references)
            if total_bytes > max_input_bytes:
                raise RuntimeError("trade_level_history_snapshot_input_size_limit_exceeded")
            if total_uncompressed > max_input_uncompressed_bytes:
                raise RuntimeError(
                    "trade_level_history_snapshot_input_uncompressed_limit_exceeded"
                )
            if total_rows > max_input_rows:
                raise RuntimeError("trade_level_history_snapshot_input_row_limit_exceeded")
            if retained is not None:
                _verify_rehydrated_snapshot_files(retained, references)
                (temporary / "manifest.json").write_bytes(
                    (final_root / "manifest.json").read_bytes()
                )
                (temporary / "SEALED").write_bytes((final_root / "SEALED").read_bytes())
                manifest = retained
            else:
                manifest = _build_signed_manifest(
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
                backup = queue / "snapshots" / f".__trade_level_history_old_{token}"
                os.replace(final_root, backup)
                try:
                    os.replace(temporary, final_root)
                except Exception:
                    os.replace(backup, final_root)
                    raise
                shutil.rmtree(backup, ignore_errors=True)
            else:
                os.replace(temporary, final_root)
            return TradeLevelHistorySnapshotMaterialization(
                manifest,
                True,
                rehydrating,
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def rehydrate_trade_level_history_snapshot_payload(
    lake_root: str | Path,
    queue_root: str | Path,
    snapshot_id: str,
    *,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    max_input_bytes: int,
    max_input_uncompressed_bytes: int,
    max_input_rows: int,
) -> TradeLevelHistorySnapshotMaterialization:
    root = Path(queue_root) / "snapshots" / snapshot_id
    manifest = TradeLevelHistorySnapshotManifest.model_validate_json(
        (root / "manifest.json").read_text("utf-8")
    )
    preflight = preflight_trade_level_history_snapshot(
        lake_root,
        queue_root,
        as_of_date=manifest.as_of_date,
        quant_lab_commit=manifest.quant_lab_commit,
    )
    if preflight.snapshot_id != snapshot_id:
        raise RuntimeError("trade_level_history_snapshot_rehydrate_identity_mismatch")
    return materialize_trade_level_history_snapshot(
        preflight,
        signing_key=signing_key,
        signature_key_id=signature_key_id,
        max_input_bytes=max_input_bytes,
        max_input_uncompressed_bytes=max_input_uncompressed_bytes,
        max_input_rows=max_input_rows,
    )


def cleanup_stale_trade_level_history_rehydrate_partials(
    queue_root: str | Path,
    *,
    stale_after_seconds: int = (
        TRADE_LEVEL_HISTORY_REHYDRATE_PARTIAL_STALE_SECONDS
    ),
    now: datetime | None = None,
) -> tuple[str, ...]:
    """Remove abandoned rehydrates only while the snapshot lock is free."""

    queue = ensure_research_queue_layout(queue_root)
    observed_at = now or datetime.now(UTC)
    removed: list[str] = []
    for partial in sorted(
        (queue / "snapshots").glob(".rehydrate.*.partial")
    ):
        marker_path = partial / "REHYDRATE.json"
        try:
            marker = json.loads(marker_path.read_text("utf-8"))
            if marker.get("schema_version") != (
                "quant_lab_trade_level_history_snapshot_rehydrate.v1"
            ):
                continue
            snapshot_id = str(marker["snapshot_id"])
            age = observed_at.timestamp() - partial.stat().st_mtime
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if age < stale_after_seconds:
            continue
        try:
            with snapshot_payload_lock(
                queue,
                snapshot_id,
                timeout_seconds=0,
            ):
                shutil.rmtree(partial, ignore_errors=False)
        except (TimeoutError, OSError):
            continue
        removed.append(partial.name)
        audit_path = (
            queue / "audit" / "trade_level_history_snapshot.jsonl"
        )
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "action": (
                            "snapshot_rehydrate_partial_cleaned"
                        ),
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


def verify_trade_level_history_snapshot_manifest(
    manifest: TradeLevelHistorySnapshotManifest,
    *,
    final_root: Path,
    public_key: Ed25519PublicKey,
    require_payload: bool = True,
) -> None:
    expected = model_content_sha256(manifest, blank_fields=("manifest_sha256",))
    if manifest.manifest_sha256 != expected:
        raise ValueError("trade_level_history_snapshot_manifest_digest_mismatch")
    verify_payload(manifest, manifest.signature, public_key)
    if (final_root / "SEALED").read_text("ascii").strip() != manifest.manifest_sha256:
        raise ValueError("trade_level_history_snapshot_seal_mismatch")
    if not require_payload:
        return
    expected_paths = {item.relative_path for item in manifest.files}
    actual_paths = {
        str(path.relative_to(final_root / "files")).replace("\\", "/")
        for path in (final_root / "files").rglob("*.parquet")
        if path.is_file()
    }
    if expected_paths != actual_paths:
        raise ValueError("trade_level_history_snapshot_file_set_mismatch")
    for item in manifest.files:
        path = final_root / "files" / item.relative_path
        resolved = path.resolve(strict=True)
        files_root = (final_root / "files").resolve(strict=True)
        if files_root not in resolved.parents:
            raise ValueError("trade_level_history_snapshot_path_escape")
        if path.stat().st_size != item.size_bytes:
            raise ValueError("trade_level_history_snapshot_size_mismatch")
        if sha256_file(path) != item.sha256:
            raise ValueError("trade_level_history_snapshot_sha256_mismatch")
        if _schema_fingerprint(path) != item.schema_fingerprint:
            raise ValueError("trade_level_history_snapshot_schema_mismatch")
        if _parquet_uncompressed_bytes(path) != item.uncompressed_bytes:
            raise ValueError("trade_level_history_snapshot_uncompressed_mismatch")
        frame = pl.read_parquet(path)
        if frame.height != item.row_count:
            raise ValueError("trade_level_history_snapshot_row_count_mismatch")
        time_columns = (
            ("decision_ts",)
            if item.dataset_name == "cloud/trade_opportunity_event"
            else ("decision_ts", "ts_utc", "label_ts")
        )
        lower, upper = _frame_time_bounds(frame, time_columns)
        if lower != item.min_ts or upper != item.max_ts:
            raise ValueError("trade_level_history_snapshot_time_bounds_mismatch")
        if item.dataset_name == "cloud/trade_opportunity_event":
            _validate_trade_opportunity_events(
                frame,
                as_of_date=manifest.as_of_date,
            )
            if (
                _frame_digest("cloud/trade_opportunity_event", frame)
                != manifest.derived_trade_opportunity_event_digest
            ):
                raise ValueError("trade_level_history_snapshot_event_digest_mismatch")


def load_trade_level_history_generation_binding(
    lake_root: str | Path,
) -> tuple[str | None, str | None, dict[str, Any]]:
    pointer = Path(lake_root) / "gold" / "trade_level_history_generation.json"
    if not pointer.is_file():
        return None, None, {}
    try:
        payload = json.loads(pointer.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("trade_level_history_generation_pointer_invalid") from exc
    generation_id = str(payload.get("generation_id") or "") or None
    generation_digest = str(payload.get("generation_digest") or "") or None
    if (generation_id is None) != (generation_digest is None):
        raise RuntimeError("trade_level_history_generation_pointer_incomplete")
    return generation_id, generation_digest, payload


def _candidate_generation_labels(
    frame: pl.DataFrame,
    pointer: dict[str, Any],
) -> pl.DataFrame:
    managed_columns = tuple(
        str(column)
        for column in dict(pointer.get("managed_columns") or {}).get(
            "v5_candidate_label",
            (),
        )
    )
    if not managed_columns:
        raise RuntimeError("trade_level_history_candidate_label_columns_missing")
    missing = sorted(set(managed_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(
            "trade_level_history_candidate_label_columns_missing:" + ",".join(missing)
        )
    filtered = frame
    if {"strategy", "source"}.issubset(filtered.columns):
        filtered = filtered.filter(
            (pl.col("strategy") == "v5")
            & (pl.col("source") == CANDIDATE_LABEL_SOURCE)
        )
    filtered = filtered.select(managed_columns)
    expected_rows = int(
        dict(pointer.get("row_counts") or {}).get("v5_candidate_label", -1)
    )
    if filtered.height != expected_rows:
        raise RuntimeError("trade_level_history_candidate_label_row_count_mismatch")
    required = set(LABEL_SCHEMA)
    if not required.issubset(filtered.columns):
        raise RuntimeError("trade_level_history_candidate_label_schema_incomplete")
    return filtered


def _load_candidate_evidence_pointer(root: Path) -> dict[str, Any]:
    path = root / V5_CANDIDATE_EVIDENCE_GENERATION_POINTER
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("trade_level_history_candidate_generation_invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("trade_level_history_candidate_generation_invalid")
    return payload


def _candidate_label_dataset_hash(pointer: dict[str, Any]) -> str:
    value = str(
        dict(pointer.get("dataset_hashes") or {}).get("v5_candidate_label") or ""
    )
    if len(value) != 64:
        raise RuntimeError("trade_level_history_candidate_label_hash_missing")
    return value


def _materialize_snapshot_files(
    preflight: TradeLevelHistorySnapshotPreflight,
    temporary: Path,
) -> list[TradeLevelHistorySnapshotFile]:
    frames = {
        "cloud/trade_opportunity_event": preflight.events,
        "gold/v5_candidate_label": preflight.candidate_labels,
    }
    references: list[TradeLevelHistorySnapshotFile] = []
    for dataset_name in TRADE_LEVEL_HISTORY_INPUT_DATASETS:
        relative_path = f"{dataset_name}/data.parquet"
        path = temporary / "files" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[dataset_name].write_parquet(path, compression="zstd")
        time_columns = (
            ("decision_ts",)
            if dataset_name == "cloud/trade_opportunity_event"
            else ("decision_ts", "ts_utc", "label_ts")
        )
        lower, upper = _frame_time_bounds(frames[dataset_name], time_columns)
        stat = path.stat()
        references.append(
            TradeLevelHistorySnapshotFile(
                dataset_name=dataset_name,
                relative_path=relative_path,
                sha256=sha256_file(path),
                size_bytes=stat.st_size,
                row_count=frames[dataset_name].height,
                min_ts=lower,
                max_ts=upper,
                schema_fingerprint=_schema_fingerprint(path),
                uncompressed_bytes=_parquet_uncompressed_bytes(path),
            )
        )
    return references


def _build_signed_manifest(
    preflight: TradeLevelHistorySnapshotPreflight,
    references: list[TradeLevelHistorySnapshotFile],
    *,
    total_bytes: int,
    total_rows: int,
    total_uncompressed: int,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
) -> TradeLevelHistorySnapshotManifest:
    parameters = preflight.parameters
    fingerprint = preflight.input_fingerprint
    provisional = TradeLevelHistorySnapshotManifest(
        **parameters.model_dump(),
        snapshot_id=preflight.snapshot_id,
        generated_at=datetime.now(UTC),
        quant_lab_commit=preflight.quant_lab_commit,
        input_fingerprint_digest=fingerprint.input_fingerprint_digest,
        candidate_event_digest=preflight.candidate_event_digest,
        risk_permission_digest=preflight.risk_permission_digest,
        v5_trade_event_digest=preflight.v5_trade_event_digest,
        order_lifecycle_digest=preflight.order_lifecycle_digest,
        derived_trade_opportunity_event_digest=fingerprint.derived_event_digest,
        risk_permission_join_version=TRADE_LEVEL_HISTORY_RISK_JOIN_VERSION,
        candidate_label_dataset_hash=fingerprint.candidate_label_dataset_hash,
        candidate_label_row_count=fingerprint.candidate_label_row_count,
        candidate_label_schema="v5.candidate_label.v1",
        event_row_count=fingerprint.event_row_count,
        event_min_ts=fingerprint.event_min_ts,
        event_max_ts=fingerprint.event_max_ts,
        candidate_label_min_ts=fingerprint.candidate_label_min_ts,
        candidate_label_max_ts=fingerprint.candidate_label_max_ts,
        datasets=TRADE_LEVEL_HISTORY_INPUT_DATASETS,
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
    retained: TradeLevelHistorySnapshotManifest,
    references: list[TradeLevelHistorySnapshotFile],
) -> None:
    expected = sorted(
        (item.model_dump(mode="json") for item in retained.files),
        key=lambda item: str(item["relative_path"]),
    )
    actual = sorted(
        (item.model_dump(mode="json") for item in references),
        key=lambda item: str(item["relative_path"]),
    )
    if expected != actual:
        raise RuntimeError("trade_level_history_snapshot_rehydrate_identity_mismatch")


def _validate_trade_opportunity_events(
    frame: pl.DataFrame,
    *,
    as_of_date: date,
) -> None:
    required = set(TRADE_OPPORTUNITY_EVENT_SCHEMA)
    if not required.issubset(frame.columns):
        raise ValueError("trade_level_history_event_schema_incomplete")
    if frame.is_empty():
        return
    if frame["event_id"].null_count() or frame["event_id"].n_unique() != frame.height:
        raise ValueError("trade_level_history_event_primary_key_invalid")
    if frame["decision_ts"].null_count():
        raise ValueError("trade_level_history_event_decision_ts_missing")
    allowed_sources = {
        "candidate_event_signed_context",
        "risk_permission_asof_join",
        "missing",
    }
    as_of_exclusive = datetime.combine(
        as_of_date + timedelta(days=1),
        time.min,
        tzinfo=UTC,
    )
    for row in frame.to_dicts():
        if event_id_for_row(row) != str(row.get("event_id") or ""):
            raise ValueError("trade_level_history_event_id_mismatch")
        symbol = str(row.get("symbol") or "")
        if not symbol or normalize_symbol(symbol) != symbol:
            raise ValueError("trade_level_history_event_symbol_invalid")
        source = str(row.get("risk_permission_source") or "")
        if source not in allowed_sources:
            raise ValueError("trade_level_history_event_risk_permission_source_invalid")
        decision_ts = _as_utc(row.get("decision_ts"))
        if decision_ts is None or decision_ts >= as_of_exclusive:
            raise ValueError(
                "trade_level_history_event_decision_ts_invalid"
            )
        permission_ts = _as_utc(row.get("risk_permission_as_of_ts"))
        if (
            decision_ts is not None
            and permission_ts is not None
            and permission_ts > decision_ts
        ):
            raise ValueError("trade_level_history_event_future_risk_permission")
        if source == "missing" and (
            str(row.get("quant_lab_permission") or "").upper() != "UNKNOWN"
            or str(row.get("risk_permission_status_at_decision") or "").upper()
            != "MISSING"
        ):
            raise ValueError("trade_level_history_event_missing_permission_not_closed")


def _frame_digest(dataset_name: str, frame: pl.DataFrame) -> str:
    columns = sorted(frame.columns)
    schema = [(column, str(frame.schema[column])) for column in columns]
    if not columns:
        aggregates = {
            "row_count": frame.height,
            "hash_sum_0": 0,
            "hash_sum_1": 0,
            "hash_min": 0,
            "hash_max": 0,
        }
    else:
        row = pl.struct(columns)
        values = frame.select(
            [
                pl.len().alias("row_count"),
                row.hash(seed=0).sum().alias("hash_sum_0"),
                row.hash(seed=1).sum().alias("hash_sum_1"),
                row.hash(seed=2).min().alias("hash_min"),
                row.hash(seed=3).max().alias("hash_max"),
            ]
        ).row(0, named=True)
        aggregates = {name: int(value or 0) for name, value in values.items()}
    return model_content_sha256(
        {
            "schema_version": "trade_level_history_frame_digest.v1",
            "dataset_name": dataset_name,
            "schema": schema,
            **aggregates,
        }
    )


def _frame_time_bounds(
    frame: pl.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[datetime | None, datetime | None]:
    columns = [column for column in candidates if column in frame.columns]
    if frame.is_empty() or not columns:
        return None, None
    observed_at = pl.coalesce([_utc_expr(column) for column in columns])
    bounds = frame.select(
        observed_at.min().alias("minimum"),
        observed_at.max().alias("maximum"),
    ).row(0, named=True)
    return _as_utc(bounds["minimum"]), _as_utc(bounds["maximum"])


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


def _required_text(payload: dict[str, Any], field_name: str) -> str:
    value = str(payload.get(field_name) or "")
    if not value:
        raise RuntimeError(f"trade_level_history_candidate_{field_name}_missing")
    return value


def _required_sha(payload: dict[str, Any], field_name: str) -> str:
    value = _required_text(payload, field_name)
    if len(value) != 64:
        raise RuntimeError(f"trade_level_history_candidate_{field_name}_invalid")
    return value


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
