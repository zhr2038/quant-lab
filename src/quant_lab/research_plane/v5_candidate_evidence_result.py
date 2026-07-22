from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from quant_lab.research.candidate_labels import LABEL_SCHEMA
from quant_lab.research.strategy_evidence import EVIDENCE_VERSION, SAMPLE_SCHEMA, SOURCE_NAME
from quant_lab.research_plane.result import schema_fingerprint
from quant_lab.research_plane.signatures import model_content_sha256, sha256_file, verify_payload
from quant_lab.research_plane.v5_candidate_evidence_contracts import (
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_FILE_COUNT,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_UNCOMPRESSED_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_UNCOMPRESSED_BYTES,
    V5_CANDIDATE_LABEL_DELTA_PRIMARY_KEYS,
    V5_STRATEGY_EVIDENCE_SAMPLE_DELTA_PRIMARY_KEYS,
    V5CandidateEvidenceOutputDataset,
    V5CandidateEvidenceResultManifest,
    V5CandidateEvidenceSnapshotManifest,
    V5CandidateEvidenceTask,
    V5CandidateEvidenceWorkerReceipt,
)
from quant_lab.research_worker.v5_candidate_evidence import (
    V5_CANDIDATE_EVIDENCE_ANTI_LEAKAGE_CHECKS,
)

V5_CANDIDATE_EVIDENCE_REQUIRED_REPORTS = frozenset(
    {
        "reports/v5_candidate_evidence_worker_report.json",
        "reports/v5_candidate_evidence_anti_leakage.json",
    }
)
V5_CANDIDATE_EVIDENCE_OUTPUT_SCHEMAS = {
    "v5_candidate_label_delta": LABEL_SCHEMA,
    "strategy_evidence_sample_delta": SAMPLE_SCHEMA,
}


@dataclass(frozen=True)
class ValidatedV5CandidateEvidenceResult:
    manifest: V5CandidateEvidenceResultManifest
    receipt: V5CandidateEvidenceWorkerReceipt
    snapshot: V5CandidateEvidenceSnapshotManifest
    label_paths: tuple[Path, ...]
    sample_paths: tuple[Path, ...]
    reports: dict[str, dict[str, Any]]


def validate_v5_candidate_evidence_result_bundle(
    bundle_root: str | Path,
    *,
    manifest: V5CandidateEvidenceResultManifest,
    receipt: V5CandidateEvidenceWorkerReceipt,
    task: V5CandidateEvidenceTask,
    snapshot: V5CandidateEvidenceSnapshotManifest,
    worker_public_key: Ed25519PublicKey,
    expected_worker_key_id: str,
    max_result_bytes: int = DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_BYTES,
    max_result_uncompressed_bytes: int = (
        DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_UNCOMPRESSED_BYTES
    ),
    max_partition_bytes: int = DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_BYTES,
    max_partition_uncompressed_bytes: int = (
        DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_UNCOMPRESSED_BYTES
    ),
    max_file_count: int = DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_FILE_COUNT,
) -> ValidatedV5CandidateEvidenceResult:
    """Validate capacity and immutable bindings before any global result scan."""

    root = Path(bundle_root).resolve(strict=True)
    if manifest.worker_key_id != expected_worker_key_id:
        raise ValueError("v5_candidate_evidence_result_unknown_worker_key")
    if receipt.worker_key_id != expected_worker_key_id:
        raise ValueError("v5_candidate_evidence_receipt_unknown_worker_key")
    verify_payload(manifest, manifest.signature, worker_public_key)
    verify_payload(receipt, receipt.signature, worker_public_key)
    _validate_binding(manifest, receipt, task, snapshot)

    manifest_path = _safe_bundle_path(root, "manifest.json")
    if receipt.result_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("v5_candidate_evidence_receipt_manifest_sha256_mismatch")
    declared_file_count = 2 + len(manifest.outputs) + len(manifest.reports)
    if declared_file_count > max_file_count:
        raise ValueError("v5_candidate_evidence_result_file_count_limit_exceeded")
    if manifest.output_bytes > max_result_bytes:
        raise ValueError("v5_candidate_evidence_result_size_limit_exceeded")
    if any(item.size_bytes > max_partition_bytes for item in manifest.outputs):
        raise ValueError("v5_candidate_evidence_partition_size_limit_exceeded")
    report_paths = {item.relative_path for item in manifest.reports}
    if report_paths != V5_CANDIDATE_EVIDENCE_REQUIRED_REPORTS:
        raise ValueError("v5_candidate_evidence_result_report_set_mismatch")

    # File-set, compressed and Parquet-metadata gates precede Unique/GroupBy/Join scans.
    declared_paths: list[Path] = []
    actual_uncompressed: dict[str, int] = {}
    for reference in (*manifest.outputs, *manifest.reports):
        path = _safe_bundle_path(root, reference.relative_path)
        _validate_file_integrity(
            path,
            size_bytes=reference.size_bytes,
            sha256=reference.sha256,
            label=reference.relative_path,
        )
        declared_paths.append(path)
        if isinstance(reference, V5CandidateEvidenceOutputDataset):
            uncompressed = _parquet_uncompressed_bytes(path)
            if reference.uncompressed_bytes != uncompressed:
                raise ValueError(
                    "v5_candidate_evidence_partition_uncompressed_size_mismatch"
                )
            if uncompressed > max_partition_uncompressed_bytes:
                raise ValueError(
                    "v5_candidate_evidence_partition_uncompressed_size_limit_exceeded"
                )
            actual_uncompressed[reference.relative_path] = uncompressed
        else:
            actual_uncompressed[reference.relative_path] = path.stat().st_size
    _validate_declared_file_set(root, manifest)
    total_uncompressed = sum(actual_uncompressed.values())
    if manifest.output_uncompressed_bytes != total_uncompressed:
        raise ValueError("v5_candidate_evidence_result_uncompressed_size_mismatch")
    if total_uncompressed > max_result_uncompressed_bytes:
        raise ValueError("v5_candidate_evidence_result_uncompressed_size_limit_exceeded")

    paths_by_dataset: dict[str, list[Path]] = {
        "v5_candidate_label_delta": [],
        "strategy_evidence_sample_delta": [],
    }
    for output in manifest.outputs:
        path = _safe_bundle_path(root, output.relative_path)
        expected_schema = V5_CANDIDATE_EVIDENCE_OUTPUT_SCHEMAS[output.dataset_name]
        schema = pl.read_parquet_schema(path)
        if list(schema.items()) != list(expected_schema.items()):
            raise ValueError(
                f"v5_candidate_evidence_result_schema_mismatch:{output.dataset_name}"
            )
        if output.schema_fingerprint != schema_fingerprint(schema):
            raise ValueError(
                "v5_candidate_evidence_result_schema_fingerprint_mismatch:"
                f"{output.dataset_name}"
            )
        if output.partition_identity != _partition_identity(output):
            raise ValueError("v5_candidate_evidence_partition_identity_mismatch")
        lazy = pl.scan_parquet(path)
        _validate_row_count(lazy, output.row_count, output.relative_path)
        _validate_unique_keys(lazy, list(output.primary_keys), output.relative_path)
        lower, upper = _timestamp_bounds(lazy, "ts_utc")
        if (lower, upper) != (output.min_ts, output.max_ts):
            raise ValueError("v5_candidate_evidence_partition_timestamp_mismatch")
        paths_by_dataset[output.dataset_name].append(path)

    label_paths = tuple(paths_by_dataset["v5_candidate_label_delta"])
    sample_paths = tuple(paths_by_dataset["strategy_evidence_sample_delta"])
    labels = _scan_or_empty(label_paths, LABEL_SCHEMA)
    samples = _scan_or_empty(sample_paths, SAMPLE_SCHEMA)
    _validate_unique_keys(
        labels,
        list(V5_CANDIDATE_LABEL_DELTA_PRIMARY_KEYS),
        "v5_candidate_label_delta_all_partitions",
    )
    _validate_unique_keys(
        samples,
        list(V5_STRATEGY_EVIDENCE_SAMPLE_DELTA_PRIMARY_KEYS),
        "strategy_evidence_sample_delta_all_partitions",
    )
    _validate_output_scope(labels, samples, manifest)
    reports = _validate_reports(root, manifest)
    if receipt.output_rows != sum(item.row_count for item in manifest.outputs):
        raise ValueError("v5_candidate_evidence_receipt_output_rows_mismatch")
    return ValidatedV5CandidateEvidenceResult(
        manifest=manifest,
        receipt=receipt,
        snapshot=snapshot,
        label_paths=label_paths,
        sample_paths=sample_paths,
        reports=reports,
    )


def _validate_binding(
    manifest: V5CandidateEvidenceResultManifest,
    receipt: V5CandidateEvidenceWorkerReceipt,
    task: V5CandidateEvidenceTask,
    snapshot: V5CandidateEvidenceSnapshotManifest,
) -> None:
    expected = (
        task.task_id,
        task.snapshot_id,
        task.snapshot_manifest_sha256,
        task.quant_lab_commit,
        task.input_fingerprint_digest,
        snapshot.candidate_event_digest,
        snapshot.market_bar_digest,
        snapshot.run_summary_digest,
        task.previous_generation_id,
        task.previous_generation_digest,
        task.as_of_date,
        task.mode,
        task.lookback_days,
        task.horizon_hours,
        task.include_historical_outcomes,
        task.candidate_label_schema_version,
        task.strategy_evidence_version,
    )
    actual = (
        manifest.task_id,
        manifest.snapshot_id,
        manifest.snapshot_manifest_sha256,
        manifest.quant_lab_commit,
        manifest.input_fingerprint_digest,
        manifest.candidate_event_digest,
        manifest.market_bar_digest,
        manifest.run_summary_digest,
        manifest.previous_generation_id,
        manifest.previous_generation_digest,
        manifest.as_of_date,
        manifest.mode,
        manifest.lookback_days,
        manifest.horizon_hours,
        manifest.include_historical_outcomes,
        manifest.candidate_label_schema_version,
        manifest.strategy_evidence_version,
    )
    if expected != actual:
        raise ValueError("v5_candidate_evidence_result_task_binding_mismatch")
    if manifest.worker_commit != task.quant_lab_commit:
        raise ValueError("v5_candidate_evidence_result_worker_commit_mismatch")
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
        raise ValueError("v5_candidate_evidence_result_receipt_binding_mismatch")


def _validate_output_scope(
    labels: pl.LazyFrame,
    samples: pl.LazyFrame,
    manifest: V5CandidateEvidenceResultManifest,
) -> None:
    invalid_horizons = labels.filter(
        ~pl.col("horizon_hours").is_in(manifest.horizon_hours)
    ).select(pl.len()).collect(engine="streaming").item()
    if invalid_horizons:
        raise ValueError("v5_candidate_evidence_result_horizon_scope_mismatch")
    label_rows = labels.select(
        ["strategy", "candidate_id", "horizon_hours", "net_bps_after_cost"]
    )
    sample_rows = samples.select(
        [
            "strategy",
            "candidate_id",
            "horizon_hours",
            "net_bps_after_cost",
            "source_type",
            "evidence_version",
            "source",
        ]
    )
    missing_labels = (
        sample_rows.join(
            label_rows,
            on=["strategy", "candidate_id", "horizon_hours"],
            how="anti",
        )
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if missing_labels:
        raise ValueError("v5_candidate_evidence_sample_label_binding_mismatch")
    invalid_samples = (
        sample_rows.filter(
            (pl.col("source_type") != "candidate_event_label")
            | (pl.col("evidence_version") != EVIDENCE_VERSION)
            | (pl.col("source") != SOURCE_NAME)
        )
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if invalid_samples:
        raise ValueError("v5_candidate_evidence_sample_scope_mismatch")
    net_mismatches = (
        sample_rows.join(
            label_rows,
            on=["strategy", "candidate_id", "horizon_hours"],
            how="inner",
            suffix="_label",
        )
        .filter(
            ~(
                pl.col("net_bps_after_cost").eq_missing(
                    pl.col("net_bps_after_cost_label")
                )
            )
        )
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if net_mismatches:
        raise ValueError("v5_candidate_evidence_sample_net_bps_mismatch")


def _validate_reports(
    root: Path,
    manifest: V5CandidateEvidenceResultManifest,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for reference in manifest.reports:
        path = _safe_bundle_path(root, reference.relative_path)
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("v5_candidate_evidence_report_invalid_json") from exc
        if not isinstance(payload, dict):
            raise ValueError("v5_candidate_evidence_report_invalid_shape")
        reports[reference.relative_path] = payload
    anti = reports["reports/v5_candidate_evidence_anti_leakage.json"]
    if anti.get("status") != "PASS" or anti.get("violation_count") != 0:
        raise ValueError("v5_candidate_evidence_anti_leakage_not_pass")
    checks = anti.get("checks")
    if not isinstance(checks, list):
        raise ValueError("v5_candidate_evidence_anti_leakage_checks_invalid")
    names = tuple(str(item.get("check_name") or "") for item in checks)
    if names != V5_CANDIDATE_EVIDENCE_ANTI_LEAKAGE_CHECKS:
        raise ValueError("v5_candidate_evidence_anti_leakage_check_set_mismatch")
    if any(item.get("status") != "PASS" or item.get("violation_count") != 0 for item in checks):
        raise ValueError("v5_candidate_evidence_anti_leakage_check_failed")
    manifest_checks = tuple(item.model_dump(mode="json") for item in manifest.anti_leakage_checks)
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
        raise ValueError("v5_candidate_evidence_manifest_report_check_mismatch")
    for payload in reports.values():
        if payload.get("automatic_promotion") is not False:
            raise ValueError("v5_candidate_evidence_report_automatic_promotion_forbidden")
        if payload.get("max_live_notional_usdt") != 0:
            raise ValueError("v5_candidate_evidence_report_live_notional_forbidden")
        if payload.get("live_order_effect") != "none_read_only_research":
            raise ValueError("v5_candidate_evidence_report_live_effect_forbidden")
    return reports


def _partition_identity(partition: V5CandidateEvidenceOutputDataset) -> str:
    return model_content_sha256(
        {
            "schema_version": "v5_candidate_evidence_partition_identity.v1",
            "dataset_name": partition.dataset_name,
            "partition_date": partition.partition_date,
            "part_number": partition.part_number,
            "sha256": partition.sha256,
            "row_count": partition.row_count,
            "schema_fingerprint": partition.schema_fingerprint,
            "min_ts": partition.min_ts,
            "max_ts": partition.max_ts,
        }
    )


def _validate_unique_keys(lazy: pl.LazyFrame, keys: list[str], label: str) -> None:
    columns = lazy.collect_schema().names()
    if not set(keys).issubset(columns):
        raise ValueError(f"v5_candidate_evidence_result_key_missing:{label}")
    duplicate = (
        lazy.group_by(keys)
        .agg(pl.len().alias("_count"))
        .filter(pl.col("_count") > 1)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if duplicate:
        raise ValueError(f"v5_candidate_evidence_result_duplicate_key:{label}")


def _validate_row_count(lazy: pl.LazyFrame, expected: int, label: str) -> None:
    actual = int(lazy.select(pl.len().alias("rows")).collect().item(0, "rows") or 0)
    if actual != expected:
        raise ValueError(f"v5_candidate_evidence_result_row_count_mismatch:{label}")


def _timestamp_bounds(
    lazy: pl.LazyFrame,
    column: str,
) -> tuple[datetime | None, datetime | None]:
    if column not in lazy.collect_schema().names():
        return None, None
    frame = lazy.select(
        [
            pl.col(column).min().alias("lower"),
            pl.col(column).max().alias("upper"),
        ]
    ).collect()
    return _as_utc(frame.item(0, "lower")), _as_utc(frame.item(0, "upper"))


def _scan_or_empty(paths: tuple[Path, ...], schema: dict[str, Any]) -> pl.LazyFrame:
    if not paths:
        return pl.DataFrame(schema=schema).lazy()
    return pl.scan_parquet([str(path) for path in paths], extra_columns="ignore")


def _validate_declared_file_set(
    root: Path,
    manifest: V5CandidateEvidenceResultManifest,
) -> None:
    expected = {
        "manifest.json",
        "receipt.json",
        *(item.relative_path for item in manifest.outputs),
        *(item.relative_path for item in manifest.reports),
    }
    actual = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError("v5_candidate_evidence_result_file_set_mismatch")


def _safe_bundle_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative_path:
        raise ValueError("v5_candidate_evidence_result_unsafe_path")
    path = root.joinpath(*pure.parts)
    resolved = path.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError("v5_candidate_evidence_result_path_escape")
    return resolved


def _validate_file_integrity(
    path: Path,
    *,
    size_bytes: int,
    sha256: str,
    label: str,
) -> None:
    if path.stat().st_size != size_bytes:
        raise ValueError(f"v5_candidate_evidence_result_size_mismatch:{label}")
    if sha256_file(path) != sha256:
        raise ValueError(f"v5_candidate_evidence_result_sha256_mismatch:{label}")


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
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
