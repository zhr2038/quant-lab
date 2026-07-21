from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from quant_lab.factors.factory import (
    FACTOR_CORRELATION_SCHEMA,
    FACTOR_DEFINITION_SCHEMA,
    FACTOR_EVIDENCE_SCHEMA,
    FACTOR_VALUE_SCHEMA,
    build_factor_definition_frame,
)
from quant_lab.research_plane.contracts import (
    FACTOR_FACTORY_RECEIPT_SCHEMA,
    FACTOR_FACTORY_RESULT_SCHEMA,
    FactorFactoryResultManifest,
    FactorFactorySnapshotManifest,
    FactorFactoryTask,
    FactorFactoryWorkerReceipt,
)
from quant_lab.research_plane.result import schema_fingerprint
from quant_lab.research_plane.signatures import sha256_file, verify_payload
from quant_lab.research_worker.factor_factory import (
    FACTOR_FACTORY_ALLOWED_DECISIONS,
    FACTOR_FACTORY_ANTI_LEAKAGE_CHECKS,
    manifest_parameters,
)

FACTOR_FACTORY_REQUIRED_REPORTS = frozenset(
    {
        "reports/factor_factory_worker_report.json",
        "reports/factor_factory_anti_leakage.json",
    }
)
FACTOR_FACTORY_OUTPUT_SCHEMAS = {
    "factor_definition_preview": FACTOR_DEFINITION_SCHEMA,
    "factor_evidence": FACTOR_EVIDENCE_SCHEMA,
    "factor_correlation_daily": FACTOR_CORRELATION_SCHEMA,
}
FACTOR_FACTORY_FORBIDDEN_STATES = frozenset(
    {"LIVE_SMALL_READY", "LIVE", "CANARY", "ENFORCE", "AUTO_PROMOTE"}
)


@dataclass(frozen=True)
class ValidatedFactorFactoryResult:
    manifest: FactorFactoryResultManifest
    receipt: FactorFactoryWorkerReceipt
    snapshot: FactorFactorySnapshotManifest
    output_paths: dict[str, Path]
    value_partition_paths: tuple[Path, ...]
    reports: dict[str, bytes]


def validate_factor_factory_result_bundle(
    bundle_root: str | Path,
    *,
    manifest: FactorFactoryResultManifest,
    receipt: FactorFactoryWorkerReceipt,
    task: FactorFactoryTask,
    snapshot: FactorFactorySnapshotManifest,
    worker_public_key: Ed25519PublicKey,
    expected_worker_key_id: str,
    max_result_bytes: int,
    max_value_partition_bytes: int = 256 * 1024**2,
    max_file_count: int = 20_000,
    max_uncompressed_bytes: int = 16 * 1024**3,
) -> ValidatedFactorFactoryResult:
    root = Path(bundle_root).resolve(strict=True)
    if manifest.schema_version != FACTOR_FACTORY_RESULT_SCHEMA:
        raise ValueError("factor_factory_result_schema_version_mismatch")
    if receipt.schema_version != FACTOR_FACTORY_RECEIPT_SCHEMA:
        raise ValueError("factor_factory_receipt_schema_version_mismatch")
    if manifest.worker_key_id != expected_worker_key_id:
        raise ValueError("factor_factory_result_unknown_worker_key")
    if receipt.worker_key_id != expected_worker_key_id:
        raise ValueError("factor_factory_receipt_unknown_worker_key")
    verify_payload(manifest, manifest.signature, worker_public_key)
    verify_payload(receipt, receipt.signature, worker_public_key)
    _validate_binding(manifest, receipt, task, snapshot)
    manifest_path = _safe_bundle_path(root, "manifest.json")
    if receipt.result_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("factor_factory_receipt_manifest_sha256_mismatch")
    if manifest.output_bytes > max_result_bytes:
        raise ValueError("factor_factory_result_size_limit_exceeded")
    declared_file_count = 2 + len(manifest.outputs) + len(manifest.value_partitions) + len(
        manifest.reports
    )
    if declared_file_count > max_file_count:
        raise ValueError("factor_factory_result_file_count_limit_exceeded")
    report_paths = {item.relative_path for item in manifest.reports}
    if report_paths != FACTOR_FACTORY_REQUIRED_REPORTS or len(manifest.reports) != len(
        FACTOR_FACTORY_REQUIRED_REPORTS
    ):
        raise ValueError("factor_factory_result_report_set_mismatch")

    output_paths: dict[str, Path] = {}
    for output in manifest.outputs:
        expected_schema = FACTOR_FACTORY_OUTPUT_SCHEMAS[output.dataset_name]
        path = _safe_bundle_path(root, output.relative_path)
        _validate_file_integrity(
            path,
            size_bytes=output.size_bytes,
            sha256=output.sha256,
            label=output.dataset_name,
        )
        schema = pl.read_parquet_schema(path)
        if list(schema.items()) != list(expected_schema.items()):
            raise ValueError(f"factor_factory_result_schema_mismatch:{output.dataset_name}")
        if schema_fingerprint(schema) != output.schema_fingerprint:
            raise ValueError(
                f"factor_factory_result_schema_fingerprint_mismatch:{output.dataset_name}"
            )
        lazy = pl.scan_parquet(path)
        _validate_row_count(lazy, output.row_count, output.dataset_name)
        _validate_unique_keys(lazy, list(output.primary_keys), output.dataset_name)
        output_paths[output.dataset_name] = path

    value_paths: list[Path] = []
    for partition in manifest.value_partitions:
        if partition.size_bytes > max_value_partition_bytes:
            raise ValueError("factor_factory_value_partition_size_limit_exceeded")
        path = _safe_bundle_path(root, partition.relative_path)
        _validate_file_integrity(
            path,
            size_bytes=partition.size_bytes,
            sha256=partition.sha256,
            label=partition.relative_path,
        )
        schema = pl.read_parquet_schema(path)
        if list(schema.items()) != list(FACTOR_VALUE_SCHEMA.items()):
            raise ValueError("factor_factory_result_schema_mismatch:factor_value")
        if schema_fingerprint(schema) != partition.schema_fingerprint:
            raise ValueError("factor_factory_result_schema_fingerprint_mismatch:factor_value")
        lazy = pl.scan_parquet(path)
        _validate_row_count(lazy, partition.row_count, partition.relative_path)
        _validate_unique_keys(lazy, list(partition.primary_keys), partition.relative_path)
        _validate_value_partition_scope(lazy, partition, task)
        value_paths.append(path)
    if value_paths:
        _validate_unique_keys(
            pl.scan_parquet([str(path) for path in value_paths]),
            ["factor_id", "factor_version", "symbol", "timeframe", "ts"],
            "factor_value_all_partitions",
        )

    _validate_output_scope(output_paths, value_paths, manifest, task, snapshot)
    reports = _validate_reports(root, manifest, task, snapshot)
    _validate_declared_file_set(root, manifest)
    uncompressed_bytes = sum(
        _file_uncompressed_bytes(path)
        for path in (
            manifest_path,
            _safe_bundle_path(root, "receipt.json"),
            *output_paths.values(),
            *value_paths,
            *(_safe_bundle_path(root, item.relative_path) for item in manifest.reports),
        )
    )
    if uncompressed_bytes > max_uncompressed_bytes:
        raise ValueError("factor_factory_result_uncompressed_size_limit_exceeded")
    output_rows = sum(item.row_count for item in manifest.outputs) + sum(
        item.row_count for item in manifest.value_partitions
    )
    if receipt.output_rows != output_rows:
        raise ValueError("factor_factory_receipt_output_rows_mismatch")
    return ValidatedFactorFactoryResult(
        manifest=manifest,
        receipt=receipt,
        snapshot=snapshot,
        output_paths=output_paths,
        value_partition_paths=tuple(value_paths),
        reports=reports,
    )


def _validate_binding(
    manifest: FactorFactoryResultManifest,
    receipt: FactorFactoryWorkerReceipt,
    task: FactorFactoryTask,
    snapshot: FactorFactorySnapshotManifest,
) -> None:
    expected = (
        task.task_id,
        task.snapshot_id,
        task.snapshot_manifest_sha256,
        task.quant_lab_commit,
        task.factor_plan_digest,
        task.source_input_digest,
        task.cost_input_digest,
        task.previous_generation_id,
        task.previous_generation_digest,
        task.as_of_date,
        task.feature_set,
        task.feature_version,
        task.factor_version,
        task.timeframe,
        task.horizon_bars,
        task.decision_delay_bars,
        task.min_samples,
        task.top_quantile,
        task.cost_quantile,
        task.result_mode,
        task.history_mode,
    )
    observed = (
        manifest.task_id,
        manifest.snapshot_id,
        manifest.snapshot_manifest_sha256,
        manifest.quant_lab_commit,
        manifest.factor_plan_digest,
        manifest.source_input_digest,
        manifest.cost_input_digest,
        manifest.previous_generation_id,
        manifest.previous_generation_digest,
        manifest.as_of_date,
        manifest.feature_set,
        manifest.feature_version,
        manifest.factor_version,
        manifest.timeframe,
        manifest.horizon_bars,
        manifest.decision_delay_bars,
        manifest.min_samples,
        manifest.top_quantile,
        manifest.cost_quantile,
        manifest.result_mode,
        manifest.history_mode,
    )
    if observed != expected:
        raise ValueError("factor_factory_result_task_binding_mismatch")
    if receipt.task_id != task.task_id or receipt.snapshot_id != task.snapshot_id:
        raise ValueError("factor_factory_receipt_task_binding_mismatch")
    if (
        task.snapshot_id != snapshot.snapshot_id
        or task.parameters.model_dump() != manifest_parameters(snapshot)
    ):
        raise ValueError("factor_factory_result_snapshot_binding_mismatch")
    if (
        manifest.worker_commit != task.quant_lab_commit
        or receipt.worker_commit != task.quant_lab_commit
    ):
        raise ValueError("factor_factory_result_worker_commit_mismatch")
    expected_factor_ids = tuple(
        sorted(item.factor_id for item in snapshot.factor_plan.factor_specs)
    )
    if manifest.factor_ids != expected_factor_ids:
        raise ValueError("factor_factory_result_factor_membership_mismatch")


def _validate_output_scope(
    output_paths: dict[str, Path],
    value_paths: list[Path],
    manifest: FactorFactoryResultManifest,
    task: FactorFactoryTask,
    snapshot: FactorFactorySnapshotManifest,
) -> None:
    if manifest.completed_no_update:
        if output_paths or value_paths:
            raise ValueError("factor_factory_no_update_contains_outputs")
        return
    definitions = pl.scan_parquet(output_paths["factor_definition_preview"])
    observed_definitions = definitions.collect(engine="streaming").sort(
        ["factor_id", "factor_version"]
    )
    expected_definitions = build_factor_definition_frame(
        snapshot.factor_plan.factor_spec_models(),
        created_at=manifest.generated_at,
    ).sort(["factor_id", "factor_version"])
    if not observed_definitions.equals(expected_definitions, null_equal=True):
        raise ValueError("factor_factory_definition_preview_plan_mismatch")
    observed_ids = tuple(
        definitions.select("factor_id")
        .unique()
        .sort("factor_id")
        .collect(engine="streaming")
        .get_column("factor_id")
        .to_list()
    )
    if observed_ids != manifest.factor_ids:
        raise ValueError("factor_factory_definition_factor_membership_mismatch")
    plan_identity = {
        item.factor_id: (
            item.expression_hash,
            item.factor_formula_hash,
            item.operator_graph_hash,
        )
        for item in snapshot.factor_plan.factor_specs
    }
    definition_identity = {
        str(row["factor_id"]): (
            str(row["expression_hash"]),
            str(row["factor_formula_hash"]),
            str(row["operator_graph_hash"]),
        )
        for row in definitions.select(
            "factor_id",
            "expression_hash",
            "factor_formula_hash",
            "operator_graph_hash",
        )
        .collect(engine="streaming")
        .to_dicts()
    }
    if definition_identity != plan_identity:
        raise ValueError("factor_factory_definition_plan_identity_mismatch")
    values = pl.scan_parquet([str(path) for path in value_paths])
    _require_lazy_scope(
        values,
        {
            "factor_version": task.factor_version,
            "timeframe": task.timeframe,
        },
        "factor_value",
    )
    invalid_availability = values.filter(
        pl.col("available_time").is_null()
        | pl.col("event_time").is_null()
        | (pl.col("available_time") < pl.col("event_time"))
        | (pl.col("event_time") != pl.col("ts"))
    ).limit(1)
    if not invalid_availability.collect(engine="streaming").is_empty():
        raise ValueError("factor_factory_value_availability_invalid")
    lag_rows = pl.DataFrame(
        [
            {
                "factor_id": item.factor_id,
                "_expected_lag_seconds": (
                    _timeframe_seconds(item.timeframe) * item.availability_lag_bars
                ),
            }
            for item in snapshot.factor_plan.factor_specs
        ]
    )
    lagged = values.join(lag_rows.lazy(), on="factor_id", how="left")
    actual_lag = (pl.col("available_time") - pl.col("event_time")).dt.total_seconds()
    invalid_lag = lagged.filter(
        pl.col("_expected_lag_seconds").is_null()
        | actual_lag.is_null()
        | (actual_lag != pl.col("_expected_lag_seconds"))
    ).limit(1)
    if not invalid_lag.collect(engine="streaming").is_empty():
        raise ValueError("factor_factory_value_availability_lag_mismatch")
    _validate_value_metadata(values, observed_definitions)
    evidence = pl.scan_parquet(output_paths["factor_evidence"])
    _require_lazy_scope(
        evidence,
        {
            "as_of_date": task.as_of_date.isoformat(),
            "factor_version": task.factor_version,
            "timeframe": task.timeframe,
            "decision_delay_bars": task.decision_delay_bars,
        },
        "factor_evidence",
    )
    invalid_evidence = evidence.filter(
        ~pl.col("factor_id").is_in(list(manifest.factor_ids))
        | ~pl.col("horizon_bars").is_in(list(task.horizon_bars))
        | ~pl.col("decision").is_in(sorted(FACTOR_FACTORY_ALLOWED_DECISIONS))
    ).limit(1)
    if not invalid_evidence.collect(engine="streaming").is_empty():
        raise ValueError("factor_factory_evidence_scope_invalid")
    _validate_evidence_metadata(evidence, observed_definitions)
    correlations = pl.scan_parquet(output_paths["factor_correlation_daily"])
    _require_lazy_scope(
        correlations,
        {
            "as_of_date": task.as_of_date.isoformat(),
            "factor_version": task.factor_version,
            "timeframe": task.timeframe,
        },
        "factor_correlation_daily",
    )
    invalid_correlation = correlations.filter(
        ~pl.col("factor_id_left").is_in(list(manifest.factor_ids))
        | ~pl.col("factor_id_right").is_in(list(manifest.factor_ids))
    ).limit(1)
    if not invalid_correlation.collect(engine="streaming").is_empty():
        raise ValueError("factor_factory_correlation_scope_invalid")


def _validate_value_partition_scope(
    lazy: pl.LazyFrame, partition: object, task: FactorFactoryTask
) -> None:
    invalid = lazy.filter(
        (pl.col("factor_version") != task.factor_version)
        | (pl.col("timeframe") != task.timeframe)
        | (pl.col("ts").dt.date() != partition.partition_date)
    ).limit(1)
    if not invalid.collect(engine="streaming").is_empty():
        raise ValueError("factor_factory_value_partition_scope_mismatch")
    bounds = lazy.select(
        pl.col("ts").min().alias("min_ts"), pl.col("ts").max().alias("max_ts")
    ).collect(engine="streaming")
    if bounds.item(0, "min_ts") != partition.min_ts or bounds.item(0, "max_ts") != partition.max_ts:
        raise ValueError("factor_factory_value_partition_bounds_mismatch")


def _validate_value_metadata(values: pl.LazyFrame, definitions: pl.DataFrame) -> None:
    definition_columns = [
        "factor_id",
        "factor_name",
        "factor_family",
        "factor_version",
        "timeframe",
        "expression_hash",
        "factor_hash",
        "canonical_factor_id",
        "factor_formula_hash",
        "formula_hash",
        "operator_graph_hash",
        "correlation_cluster_id",
        "effective_independence_weight",
        "independence_weight",
        "input_features_json",
        "source",
        "status",
    ]
    value_columns = [*definition_columns[:-1], "factor_status"]
    observed = values.select(value_columns).unique().collect(engine="streaming").sort(
        "factor_id"
    )
    expected = (
        definitions.select(definition_columns)
        .rename({"status": "factor_status"})
        .sort("factor_id")
    )
    if not observed.equals(expected, null_equal=True):
        raise ValueError("factor_factory_value_definition_metadata_mismatch")


def _validate_evidence_metadata(evidence: pl.LazyFrame, definitions: pl.DataFrame) -> None:
    columns = [
        "factor_id",
        "factor_name",
        "factor_family",
        "factor_version",
        "timeframe",
        "factor_hash",
        "canonical_factor_id",
        "formula_hash",
        "independence_weight",
        "source",
    ]
    observed = evidence.select(columns).unique().collect(engine="streaming").sort("factor_id")
    observed_ids = observed.get_column("factor_id").to_list()
    expected = (
        definitions.filter(pl.col("factor_id").is_in(observed_ids))
        .select(columns)
        .sort("factor_id")
    )
    if not observed.equals(expected, null_equal=True):
        raise ValueError("factor_factory_evidence_definition_metadata_mismatch")


def _timeframe_seconds(timeframe: str) -> int:
    value = timeframe.strip().lower()
    if len(value) < 2 or not value[:-1].isdigit():
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    amount = int(value[:-1])
    unit = value[-1]
    if amount <= 0 or unit not in multipliers:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return amount * multipliers[unit]


def _validate_reports(
    root: Path,
    manifest: FactorFactoryResultManifest,
    task: FactorFactoryTask,
    snapshot: FactorFactorySnapshotManifest,
) -> dict[str, bytes]:
    reports: dict[str, bytes] = {}
    for report in manifest.reports:
        path = _safe_bundle_path(root, report.relative_path)
        _validate_file_integrity(
            path,
            size_bytes=report.size_bytes,
            sha256=report.sha256,
            label=report.relative_path,
        )
        payload = path.read_bytes()
        if any(value.encode("ascii") in payload for value in FACTOR_FACTORY_FORBIDDEN_STATES):
            raise ValueError(f"factor_factory_result_live_state_forbidden:{report.relative_path}")
        reports[Path(report.relative_path).name] = payload
    anti = _load_json(reports["factor_factory_anti_leakage.json"], "anti_leakage")
    if (
        anti.get("task_id") != task.task_id
        or anti.get("snapshot_id") != snapshot.snapshot_id
        or anti.get("factor_plan_digest") != task.factor_plan_digest
    ):
        raise ValueError("factor_factory_anti_leakage_binding_mismatch")
    checks = anti.get("checks")
    if not isinstance(checks, list) or len(checks) != len(FACTOR_FACTORY_ANTI_LEAKAGE_CHECKS):
        raise ValueError("factor_factory_anti_leakage_incomplete")
    if tuple(str(item.get("check_name") or "") for item in checks) != (
        FACTOR_FACTORY_ANTI_LEAKAGE_CHECKS
    ):
        raise ValueError("factor_factory_anti_leakage_check_set_mismatch")
    if (
        anti.get("status") != "PASS"
        or anti.get("violation_count") != 0
        or any(item.get("status") != "PASS" or item.get("violation_count") != 0 for item in checks)
    ):
        raise ValueError("factor_factory_anti_leakage_failed")
    _require_safety_fields(anti, "factor_factory_anti_leakage")
    worker = _load_json(reports["factor_factory_worker_report.json"], "worker_report")
    expected_worker = {
        "task_id": task.task_id,
        "snapshot_id": snapshot.snapshot_id,
        "factor_plan_digest": task.factor_plan_digest,
        "source_input_digest": task.source_input_digest,
        "cost_input_digest": task.cost_input_digest,
        "previous_generation_id": task.previous_generation_id,
        "previous_generation_digest": task.previous_generation_digest,
        "result_mode": task.result_mode,
        "history_mode": task.history_mode,
        "min_samples": task.min_samples,
        "top_quantile": task.top_quantile,
        "cost_quantile": task.cost_quantile,
        "factor_count": manifest.factor_count,
        "factor_ids": list(manifest.factor_ids),
        "completed_no_update": manifest.completed_no_update,
        "no_update_reason": manifest.no_update_reason,
    }
    for field, expected in expected_worker.items():
        if worker.get(field) != expected:
            raise ValueError(f"factor_factory_worker_report_mismatch:{field}")
    _require_safety_fields(worker, "factor_factory_worker_report")
    return reports


def _require_safety_fields(payload: dict[str, object], label: str) -> None:
    expected = {
        "diagnostic_only": True,
        "research_only": True,
        "live_order_effect": "none_read_only_research",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise ValueError(f"{label}_safety_mismatch")


def _validate_declared_file_set(root: Path, manifest: FactorFactoryResultManifest) -> None:
    expected = {"manifest.json", "receipt.json"}
    expected.update(item.relative_path for item in manifest.outputs)
    expected.update(item.relative_path for item in manifest.value_partitions)
    expected.update(item.relative_path for item in manifest.reports)
    handoff_marker = root / ".HANDOFF_READY"
    if handoff_marker.exists() and (
        not handoff_marker.is_file() or handoff_marker.stat().st_size != 0
    ):
        raise ValueError("factor_factory_result_handoff_marker_invalid")
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("factor_factory_result_symlink_forbidden")
        if path.is_file():
            observed.add(str(path.relative_to(root)).replace("\\", "/"))
    observed.discard(".HANDOFF_READY")
    if observed != expected:
        raise ValueError("factor_factory_result_undeclared_file_set")


def _safe_bundle_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("factor_factory_result_unsafe_path")
    unresolved = root.joinpath(*relative.parts)
    if unresolved.is_symlink():
        raise ValueError("factor_factory_result_symlink_forbidden")
    path = unresolved.resolve(strict=True)
    if root not in path.parents:
        raise ValueError("factor_factory_result_path_escape")
    return path


def _validate_file_integrity(
    path: Path,
    *,
    size_bytes: int,
    sha256: str,
    label: str,
) -> None:
    if path.stat().st_size != size_bytes or sha256_file(path) != sha256:
        raise ValueError(f"factor_factory_result_file_integrity_mismatch:{label}")


def _file_uncompressed_bytes(path: Path) -> int:
    if path.suffix != ".parquet":
        return path.stat().st_size
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(row_group).column(column).total_uncompressed_size
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _validate_row_count(lazy: pl.LazyFrame, expected: int, label: str) -> None:
    actual = int(lazy.select(pl.len()).collect(engine="streaming").item())
    if actual != expected:
        raise ValueError(f"factor_factory_result_row_count_mismatch:{label}")


def _validate_unique_keys(lazy: pl.LazyFrame, keys: list[str], label: str) -> None:
    duplicate = lazy.group_by(keys).len().filter(pl.col("len") > 1).limit(1)
    if not duplicate.collect(engine="streaming").is_empty():
        raise ValueError(f"factor_factory_result_duplicate_key:{label}")


def _require_lazy_scope(
    lazy: pl.LazyFrame,
    expected: dict[str, object],
    label: str,
) -> None:
    for column, value in expected.items():
        invalid = lazy.filter(pl.col(column).is_null() | (pl.col(column) != value)).limit(1)
        if not invalid.collect(engine="streaming").is_empty():
            raise ValueError(f"factor_factory_result_scope_mismatch:{label}:{column}")


def _load_json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"factor_factory_{label}_invalid_json") from exc
    if not isinstance(value, dict):
        raise ValueError(f"factor_factory_{label}_invalid_json")
    return value
