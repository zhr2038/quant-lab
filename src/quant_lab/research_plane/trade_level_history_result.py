from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from quant_lab.research_plane.result import schema_fingerprint
from quant_lab.research_plane.signatures import (
    model_content_sha256,
    sha256_file,
    verify_payload,
)
from quant_lab.research_plane.trade_level_history_contracts import (
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_FILE_COUNT,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_UNCOMPRESSED_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_UNCOMPRESSED_BYTES,
    TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS,
    TRADE_LEVEL_HISTORY_OUTPUT_DATASETS,
    TradeLevelHistoryOutputDataset,
    TradeLevelHistoryResultManifest,
    TradeLevelHistorySnapshotManifest,
    TradeLevelHistoryTask,
    TradeLevelHistoryWorkerReceipt,
)
from quant_lab.trade_level.labels import (
    TRADE_OPPORTUNITY_LABEL_SCHEMA,
    build_trade_opportunity_labels,
)
from quant_lab.trade_level.similarity import (
    TRADE_LEVEL_SIMILARITY_SCHEMA,
    build_trade_level_similarity_outcome,
)

TRADE_LEVEL_HISTORY_REQUIRED_REPORTS = frozenset(
    {
        "reports/trade_level_history_worker_report.json",
        "reports/trade_level_history_anti_leakage.json",
    }
)
TRADE_LEVEL_HISTORY_OUTPUT_SCHEMAS = {
    "trade_opportunity_label": TRADE_OPPORTUNITY_LABEL_SCHEMA,
    "trade_level_similarity_outcome": TRADE_LEVEL_SIMILARITY_SCHEMA,
}
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "trade_level_decision",
        "bucket_policy_action",
        "trade_level_bucket_policy",
        "opportunity_queue",
        "trade_level_opportunity_queue",
        "max_single_order_usdt",
        "daily_trade_limit",
        "risk_permission",
        "paper_status",
        "live_status",
    }
)


@dataclass(frozen=True)
class ValidatedTradeLevelHistoryResult:
    manifest: TradeLevelHistoryResultManifest
    receipt: TradeLevelHistoryWorkerReceipt
    snapshot: TradeLevelHistorySnapshotManifest
    label_paths: tuple[Path, ...]
    similarity_paths: tuple[Path, ...]
    reports: dict[str, dict[str, Any]]


def validate_trade_level_history_result_bundle(
    bundle_root: str | Path,
    *,
    manifest: TradeLevelHistoryResultManifest,
    receipt: TradeLevelHistoryWorkerReceipt,
    task: TradeLevelHistoryTask,
    snapshot: TradeLevelHistorySnapshotManifest,
    snapshot_root: str | Path,
    worker_public_key: Ed25519PublicKey,
    expected_worker_key_id: str,
    max_result_bytes: int = DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_BYTES,
    max_result_uncompressed_bytes: int = (
        DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_UNCOMPRESSED_BYTES
    ),
    max_partition_bytes: int = DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_BYTES,
    max_partition_uncompressed_bytes: int = (
        DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_UNCOMPRESSED_BYTES
    ),
    max_file_count: int = DEFAULT_TRADE_LEVEL_HISTORY_MAX_FILE_COUNT,
) -> ValidatedTradeLevelHistoryResult:
    """Strictly validate a NAS result and causally recompute it by symbol."""

    root = Path(bundle_root).resolve(strict=True)
    signed_snapshot_root = Path(snapshot_root).resolve(strict=True)
    if manifest.worker_key_id != expected_worker_key_id:
        raise ValueError("trade_level_history_result_unknown_worker_key")
    if receipt.worker_key_id != expected_worker_key_id:
        raise ValueError("trade_level_history_receipt_unknown_worker_key")
    verify_payload(manifest, manifest.signature, worker_public_key)
    verify_payload(receipt, receipt.signature, worker_public_key)
    _validate_binding(manifest, receipt, task, snapshot)

    manifest_path = _safe_bundle_path(root, "manifest.json")
    if receipt.result_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError(
            "trade_level_history_receipt_manifest_sha256_mismatch"
        )
    declared_file_count = 2 + len(manifest.outputs) + len(manifest.reports)
    if declared_file_count > max_file_count:
        raise ValueError("trade_level_history_result_file_count_limit_exceeded")
    if manifest.output_bytes > max_result_bytes:
        raise ValueError("trade_level_history_result_size_limit_exceeded")
    if any(item.size_bytes > max_partition_bytes for item in manifest.outputs):
        raise ValueError("trade_level_history_partition_size_limit_exceeded")
    if {item.relative_path for item in manifest.reports} != (
        TRADE_LEVEL_HISTORY_REQUIRED_REPORTS
    ):
        raise ValueError("trade_level_history_result_report_set_mismatch")

    actual_uncompressed: dict[str, int] = {}
    for reference in (*manifest.outputs, *manifest.reports):
        path = _safe_bundle_path(root, reference.relative_path)
        if (
            path.stat().st_size != reference.size_bytes
            or sha256_file(path) != reference.sha256
        ):
            raise ValueError(
                f"trade_level_history_result_file_integrity_mismatch:"
                f"{reference.relative_path}"
            )
        if isinstance(reference, TradeLevelHistoryOutputDataset):
            uncompressed = _parquet_uncompressed_bytes(path)
            if uncompressed != reference.uncompressed_bytes:
                raise ValueError(
                    "trade_level_history_partition_uncompressed_size_mismatch"
                )
            if uncompressed > max_partition_uncompressed_bytes:
                raise ValueError(
                    "trade_level_history_partition_uncompressed_size_limit_exceeded"
                )
            actual_uncompressed[reference.relative_path] = uncompressed
        else:
            actual_uncompressed[reference.relative_path] = path.stat().st_size
    _validate_declared_file_set(root, manifest)
    total_uncompressed = sum(actual_uncompressed.values())
    if total_uncompressed != manifest.output_uncompressed_bytes:
        raise ValueError(
            "trade_level_history_result_uncompressed_size_mismatch"
        )
    if total_uncompressed > max_result_uncompressed_bytes:
        raise ValueError(
            "trade_level_history_result_uncompressed_size_limit_exceeded"
        )

    paths_by_dataset: dict[str, list[Path]] = {
        name: [] for name in TRADE_LEVEL_HISTORY_OUTPUT_DATASETS
    }
    for output in manifest.outputs:
        path = _safe_bundle_path(root, output.relative_path)
        expected_schema = TRADE_LEVEL_HISTORY_OUTPUT_SCHEMAS[
            output.dataset_name
        ]
        actual_schema = pl.read_parquet_schema(path)
        if list(actual_schema.items()) != list(expected_schema.items()):
            raise ValueError(
                f"trade_level_history_result_schema_mismatch:"
                f"{output.dataset_name}"
            )
        if output.schema_fingerprint != schema_fingerprint(actual_schema):
            raise ValueError(
                f"trade_level_history_result_schema_fingerprint_mismatch:"
                f"{output.dataset_name}"
            )
        if output.partition_identity != _partition_identity(output):
            raise ValueError(
                "trade_level_history_partition_identity_mismatch"
            )
        lazy = pl.scan_parquet(path)
        _validate_row_count(lazy, output.row_count, output.relative_path)
        _validate_unique_keys(
            lazy,
            list(output.primary_keys),
            output.relative_path,
        )
        lower, upper = _timestamp_bounds(lazy)
        if (lower, upper) != (output.min_ts, output.max_ts):
            raise ValueError(
                "trade_level_history_partition_timestamp_mismatch"
            )
        _validate_partition_symbol(lazy, output)
        paths_by_dataset[output.dataset_name].append(path)

    label_paths = tuple(paths_by_dataset["trade_opportunity_label"])
    similarity_paths = tuple(
        paths_by_dataset["trade_level_similarity_outcome"]
    )
    labels = _scan_paths(label_paths, TRADE_OPPORTUNITY_LABEL_SCHEMA)
    similarity = _scan_paths(
        similarity_paths,
        TRADE_LEVEL_SIMILARITY_SCHEMA,
    )
    _validate_unique_keys(
        labels,
        ["event_id"],
        "trade_opportunity_label_all_partitions",
    )
    _validate_unique_keys(
        similarity,
        ["event_id"],
        "trade_level_similarity_outcome_all_partitions",
    )
    reports = _validate_reports(root, manifest)
    if receipt.output_rows != sum(item.row_count for item in manifest.outputs):
        raise ValueError("trade_level_history_receipt_output_rows_mismatch")
    _validate_causal_outputs(
        signed_snapshot_root,
        snapshot,
        manifest,
        label_paths=label_paths,
        similarity_paths=similarity_paths,
    )
    return ValidatedTradeLevelHistoryResult(
        manifest=manifest,
        receipt=receipt,
        snapshot=snapshot,
        label_paths=label_paths,
        similarity_paths=similarity_paths,
        reports=reports,
    )


def _validate_binding(
    manifest: TradeLevelHistoryResultManifest,
    receipt: TradeLevelHistoryWorkerReceipt,
    task: TradeLevelHistoryTask,
    snapshot: TradeLevelHistorySnapshotManifest,
) -> None:
    task_fields = (
        "as_of_date",
        "history_mode",
        "candidate_evidence_generation_id",
        "candidate_evidence_generation_digest",
        "candidate_evidence_input_fingerprint",
        "trade_event_schema_version",
        "trade_label_schema_version",
        "similarity_schema_version",
        "similarity_availability_policy",
    )
    expected = (
        task.task_id,
        task.snapshot_id,
        task.snapshot_manifest_sha256,
        task.quant_lab_commit,
        task.input_fingerprint_digest,
        snapshot.derived_trade_opportunity_event_digest,
        snapshot.candidate_label_dataset_hash,
        task.previous_generation_id,
        task.previous_generation_digest,
        *(getattr(task, name) for name in task_fields),
    )
    actual = (
        manifest.task_id,
        manifest.snapshot_id,
        manifest.snapshot_manifest_sha256,
        manifest.quant_lab_commit,
        manifest.input_fingerprint_digest,
        manifest.derived_event_digest,
        manifest.candidate_label_dataset_hash,
        manifest.previous_generation_id,
        manifest.previous_generation_digest,
        *(getattr(manifest, name) for name in task_fields),
    )
    if expected != actual:
        raise ValueError("trade_level_history_result_task_binding_mismatch")
    snapshot_binding = (
        snapshot.snapshot_id,
        snapshot.manifest_sha256,
        snapshot.quant_lab_commit,
        snapshot.input_fingerprint_digest,
        snapshot.candidate_evidence_generation_id,
        snapshot.candidate_evidence_generation_digest,
        snapshot.candidate_evidence_input_fingerprint,
    )
    task_binding = (
        task.snapshot_id,
        task.snapshot_manifest_sha256,
        task.quant_lab_commit,
        task.input_fingerprint_digest,
        task.candidate_evidence_generation_id,
        task.candidate_evidence_generation_digest,
        task.candidate_evidence_input_fingerprint,
    )
    if snapshot_binding != task_binding:
        raise ValueError(
            "trade_level_history_result_snapshot_binding_mismatch"
        )
    if manifest.worker_commit != task.quant_lab_commit:
        raise ValueError("trade_level_history_result_worker_commit_mismatch")
    receipt_expected = (
        manifest.task_id,
        manifest.snapshot_id,
        manifest.worker_commit,
        manifest.completed_at,
        manifest.anti_leakage_status,
        manifest.anti_leakage_violation_count,
    )
    receipt_actual = (
        receipt.task_id,
        receipt.snapshot_id,
        receipt.worker_commit,
        receipt.completed_at,
        receipt.anti_leakage_status,
        receipt.anti_leakage_violation_count,
    )
    if receipt_expected != receipt_actual or receipt.state != "completed":
        raise ValueError("trade_level_history_result_receipt_binding_mismatch")
    if (
        receipt.input_bytes != manifest.input_bytes
        or receipt.downloaded_bytes != manifest.downloaded_bytes
        or receipt.cache_hit_bytes != manifest.cache_hit_bytes
    ):
        raise ValueError("trade_level_history_receipt_capacity_mismatch")


def _validate_reports(
    root: Path,
    manifest: TradeLevelHistoryResultManifest,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for reference in manifest.reports:
        path = _safe_bundle_path(root, reference.relative_path)
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "trade_level_history_report_invalid_json"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("trade_level_history_report_invalid_shape")
        reports[reference.relative_path] = payload
    anti = reports[
        "reports/trade_level_history_anti_leakage.json"
    ]
    if anti.get("status") != "PASS" or anti.get("violation_count") != 0:
        raise ValueError("trade_level_history_anti_leakage_not_pass")
    checks = anti.get("checks")
    if not isinstance(checks, list):
        raise ValueError(
            "trade_level_history_anti_leakage_checks_invalid"
        )
    names = tuple(str(item.get("check_name") or "") for item in checks)
    if names != TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS:
        raise ValueError(
            "trade_level_history_anti_leakage_check_set_mismatch"
        )
    if any(
        not isinstance(item, dict)
        or item.get("status") != "PASS"
        or item.get("violation_count") != 0
        for item in checks
    ):
        raise ValueError("trade_level_history_anti_leakage_check_failed")
    manifest_checks = tuple(
        item.model_dump(mode="json")
        for item in manifest.anti_leakage_checks
    )
    normalized_checks = tuple(
        {
            "check_name": item.get("check_name"),
            "status": item.get("status"),
            "violation_count": item.get("violation_count"),
            "detail": item.get("detail"),
        }
        for item in checks
    )
    if manifest_checks != normalized_checks:
        raise ValueError(
            "trade_level_history_manifest_report_check_mismatch"
        )
    for payload in reports.values():
        if payload.get("automatic_promotion") is not False:
            raise ValueError(
                "trade_level_history_report_automatic_promotion_forbidden"
            )
        if payload.get("max_live_notional_usdt") != 0:
            raise ValueError(
                "trade_level_history_report_live_notional_forbidden"
            )
        if payload.get("live_order_effect") != (
            "none_read_only_research"
        ):
            raise ValueError(
                "trade_level_history_report_live_effect_forbidden"
            )
        if _contains_forbidden_report_key(payload):
            raise ValueError(
                "trade_level_history_report_control_surface_forbidden"
            )
    worker = reports[
        "reports/trade_level_history_worker_report.json"
    ]
    if worker.get("warnings") not in ([], ()):
        raise ValueError("trade_level_history_worker_warnings_forbidden")
    return reports


def _validate_causal_outputs(
    snapshot_root: Path,
    snapshot: TradeLevelHistorySnapshotManifest,
    manifest: TradeLevelHistoryResultManifest,
    *,
    label_paths: tuple[Path, ...],
    similarity_paths: tuple[Path, ...],
) -> None:
    files_root = snapshot_root / "files"
    event_paths = [
        files_root / item.relative_path
        for item in snapshot.files
        if item.dataset_name == "cloud/trade_opportunity_event"
    ]
    candidate_paths = [
        files_root / item.relative_path
        for item in snapshot.files
        if item.dataset_name == "gold/v5_candidate_label"
    ]
    if len(event_paths) != 1 or len(candidate_paths) != 1:
        raise ValueError(
            "trade_level_history_validation_snapshot_file_set_mismatch"
        )
    events = pl.scan_parquet(event_paths)
    source_labels = pl.scan_parquet(candidate_paths)
    actual_labels = _scan_paths(
        label_paths,
        TRADE_OPPORTUNITY_LABEL_SCHEMA,
    )
    actual_similarity = _scan_paths(
        similarity_paths,
        TRADE_LEVEL_SIMILARITY_SCHEMA,
    )
    event_rows = int(
        events.select(pl.len()).collect(engine="streaming").item()
    )
    label_rows = int(
        actual_labels.select(pl.len()).collect(engine="streaming").item()
    )
    similarity_rows = int(
        actual_similarity.select(pl.len()).collect(engine="streaming").item()
    )
    if event_rows != label_rows or event_rows != similarity_rows:
        raise ValueError(
            "trade_level_history_result_event_cardinality_mismatch"
        )
    symbols = (
        events.select(pl.col("symbol").unique().sort())
        .collect(engine="streaming")
        .get_column("symbol")
        .drop_nulls()
        .to_list()
    )
    for raw_symbol in symbols:
        symbol = str(raw_symbol)
        symbol_events = (
            events.filter(pl.col("symbol") == symbol)
            .sort(["decision_ts", "event_id"])
            .collect(engine="streaming")
        )
        symbol_source_labels = _collect_candidate_labels_for_events(
            source_labels,
            symbol_events,
            symbol=symbol,
        )
        expected_labels = build_trade_opportunity_labels(
            symbol_events,
            symbol_source_labels,
            created_at=manifest.generated_at,
        ).sort("event_id")
        observed_labels = (
            actual_labels.filter(pl.col("symbol") == symbol)
            .sort("event_id")
            .collect(engine="streaming")
        )
        if not observed_labels.equals(expected_labels, null_equal=True):
            raise ValueError(
                f"trade_level_history_label_recomputation_mismatch:{symbol}"
            )
        expected_similarity = build_trade_level_similarity_outcome(
            symbol_events,
            expected_labels,
            created_at=manifest.generated_at,
        ).sort("event_id")
        observed_similarity = (
            actual_similarity.filter(pl.col("symbol") == symbol)
            .sort("event_id")
            .collect(engine="streaming")
        )
        if not observed_similarity.equals(
            expected_similarity,
            null_equal=True,
        ):
            raise ValueError(
                f"trade_level_history_similarity_recomputation_mismatch:"
                f"{symbol}"
            )


def _collect_candidate_labels_for_events(
    labels: pl.LazyFrame,
    events: pl.DataFrame,
    *,
    symbol: str,
) -> pl.DataFrame:
    candidate_ids = [
        str(value)
        for value in events.get_column("candidate_id")
        .drop_nulls()
        .unique()
        .to_list()
        if str(value)
    ]
    run_ids = [
        str(value)
        for value in events.get_column("run_id")
        .drop_nulls()
        .unique()
        .to_list()
        if str(value)
    ]
    identity_predicates: list[pl.Expr] = []
    if candidate_ids:
        identity_predicates.append(
            pl.col("candidate_id").is_in(candidate_ids)
        )
    if run_ids:
        identity_predicates.append(pl.col("run_id").is_in(run_ids))
    predicate = pl.col("symbol") == symbol
    if identity_predicates:
        predicate = predicate & pl.any_horizontal(
            identity_predicates
        )
    return labels.filter(predicate).collect(engine="streaming")


def _partition_identity(
    partition: TradeLevelHistoryOutputDataset,
) -> str:
    return model_content_sha256(
        {
            "schema_version": (
                "trade_level_history_partition_identity.v1"
            ),
            "dataset_name": partition.dataset_name,
            "partition_symbol": partition.partition_symbol,
            "part_number": partition.part_number,
            "sha256": partition.sha256,
            "row_count": partition.row_count,
            "schema_fingerprint": partition.schema_fingerprint,
            "min_ts": partition.min_ts,
            "max_ts": partition.max_ts,
        }
    )


def _validate_partition_symbol(
    lazy: pl.LazyFrame,
    output: TradeLevelHistoryOutputDataset,
) -> None:
    if output.row_count == 0:
        if output.partition_symbol != "EMPTY":
            raise ValueError(
                "trade_level_history_empty_partition_symbol_mismatch"
            )
        return
    symbols = (
        lazy.select(pl.col("symbol").unique())
        .collect(engine="streaming")
        .get_column("symbol")
        .drop_nulls()
        .to_list()
    )
    if symbols != [output.partition_symbol]:
        raise ValueError(
            "trade_level_history_partition_symbol_mismatch"
        )


def _validate_unique_keys(
    lazy: pl.LazyFrame,
    keys: list[str],
    label: str,
) -> None:
    columns = lazy.collect_schema().names()
    if not set(keys).issubset(columns):
        raise ValueError(f"trade_level_history_result_key_missing:{label}")
    duplicate = (
        lazy.group_by(keys)
        .agg(pl.len().alias("_count"))
        .filter(pl.col("_count") > 1)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    nulls = (
        lazy.filter(pl.any_horizontal([pl.col(key).is_null() for key in keys]))
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if duplicate or nulls:
        raise ValueError(
            f"trade_level_history_result_primary_key_invalid:{label}"
        )


def _validate_row_count(
    lazy: pl.LazyFrame,
    expected: int,
    label: str,
) -> None:
    actual = int(
        lazy.select(pl.len().alias("rows")).collect().item(0, "rows") or 0
    )
    if actual != expected:
        raise ValueError(
            f"trade_level_history_result_row_count_mismatch:{label}"
        )


def _timestamp_bounds(
    lazy: pl.LazyFrame,
) -> tuple[datetime | None, datetime | None]:
    frame = lazy.select(
        pl.col("decision_ts").min().alias("lower"),
        pl.col("decision_ts").max().alias("upper"),
    ).collect(engine="streaming")
    return _as_utc(frame.item(0, "lower")), _as_utc(
        frame.item(0, "upper")
    )


def _scan_paths(
    paths: tuple[Path, ...],
    schema: dict[str, Any],
) -> pl.LazyFrame:
    if not paths:
        return pl.DataFrame(schema=schema).lazy()
    return pl.scan_parquet(
        [str(path) for path in paths],
        missing_columns="insert",
        extra_columns="ignore",
    ).select(list(schema))


def _validate_declared_file_set(
    root: Path,
    manifest: TradeLevelHistoryResultManifest,
) -> None:
    declared = {
        "manifest.json",
        "receipt.json",
        *(item.relative_path for item in manifest.outputs),
        *(item.relative_path for item in manifest.reports),
    }
    handoff_marker = root / ".HANDOFF_READY"
    if handoff_marker.is_symlink() or (
        handoff_marker.exists()
        and (
            not handoff_marker.is_file()
            or handoff_marker.stat().st_size != 0
        )
    ):
        raise ValueError("trade_level_history_result_handoff_marker_invalid")
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("trade_level_history_result_symlink_forbidden")
        if path.is_file():
            actual.add(str(path.relative_to(root)).replace("\\", "/"))
    actual.discard(".HANDOFF_READY")
    if actual != declared:
        raise ValueError("trade_level_history_result_file_set_mismatch")


def _safe_bundle_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("trade_level_history_result_path_escape")
    path = root.joinpath(*pure.parts)
    resolved = path.resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise ValueError("trade_level_history_result_path_escape")
    if resolved.is_symlink():
        raise ValueError("trade_level_history_result_symlink_forbidden")
    return resolved


def _contains_forbidden_report_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_REPORT_KEYS:
                return True
            if _contains_forbidden_report_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_report_key(item) for item in value)
    return False


def _parquet_uncompressed_bytes(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(group).column(column).total_uncompressed_size
        for group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _as_utc(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value in (None, ""):
        return None
    else:
        try:
            parsed = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
