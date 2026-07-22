from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.factors.factory import (
    FACTOR_CORRELATION_SCHEMA,
    FACTOR_DEFINITION_SCHEMA,
    FACTOR_EVIDENCE_SCHEMA,
    FACTOR_VALUE_SCHEMA,
)
from quant_lab.research.alpha_factory.factory import (
    ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS,
)
from quant_lab.research.entry_quality import (
    ENTRY_QUALITY_HISTORY_OUTPUT_SPECS,
    EntryQualityHistoryArtifacts,
    EntryQualityHistoryOutputSpec,
)
from quant_lab.research.factor_research.outputs import FACTOR_RESEARCH_OUTPUT_SPECS
from quant_lab.research_plane.contracts import (
    ALPHA_FACTORY_RECEIPT_SCHEMA,
    ALPHA_FACTORY_RESULT_SCHEMA,
    DEFAULT_FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES,
    DEFAULT_FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES,
    FACTOR_FACTORY_RECEIPT_SCHEMA,
    FACTOR_FACTORY_RESULT_SCHEMA,
    FACTOR_RESEARCH_RECEIPT_SCHEMA,
    FACTOR_RESEARCH_RESULT_SCHEMA,
    RESEARCH_RECEIPT_SCHEMA,
    RESEARCH_RESULT_SCHEMA,
    AlphaFactoryResultManifest,
    AlphaFactorySnapshotManifest,
    AlphaFactoryTask,
    AlphaFactoryWorkerReceipt,
    FactorFactoryAntiLeakageCheck,
    FactorFactoryOutputDataset,
    FactorFactoryPartitionReference,
    FactorFactoryResultManifest,
    FactorFactorySnapshotManifest,
    FactorFactoryTask,
    FactorFactoryWorkerReceipt,
    FactorResearchResultManifest,
    FactorResearchSnapshotManifest,
    FactorResearchTask,
    FactorResearchWorkerReceipt,
    ResearchOutputDataset,
    ResearchOutputFile,
    ResearchResultManifest,
    ResearchSnapshotManifest,
    ResearchTask,
    ResearchWorkerReceipt,
)
from quant_lab.research_plane.factor_factory_result import (
    validate_factor_factory_result_bundle,
)
from quant_lab.research_plane.result import (
    schema_fingerprint,
    validate_alpha_factory_result_bundle,
    validate_entry_quality_history_result_bundle,
    validate_factor_research_result_bundle,
)
from quant_lab.research_plane.signatures import (
    model_content_sha256,
    sha256_file,
    sign_model,
)
from quant_lab.research_worker.alpha_factory import AlphaFactoryWorkerComputeResult
from quant_lab.research_worker.factor_factory import (
    FactorFactoryComputeArtifacts,
    StagedFactorValueSet,
)
from quant_lab.research_worker.factor_research import FactorResearchComputeArtifacts

FACTOR_FACTORY_CONTROL_OUTPUTS = (
    (
        "factor_definition_preview",
        FACTOR_DEFINITION_SCHEMA,
        ("factor_id", "factor_version"),
    ),
    (
        "factor_evidence",
        FACTOR_EVIDENCE_SCHEMA,
        (
            "as_of_date",
            "factor_id",
            "factor_version",
            "timeframe",
            "horizon_bars",
            "decision_delay_bars",
        ),
    ),
    (
        "factor_correlation_daily",
        FACTOR_CORRELATION_SCHEMA,
        (
            "as_of_date",
            "factor_id_left",
            "factor_id_right",
            "factor_version",
            "timeframe",
        ),
    ),
)


def write_factor_factory_result_bundle(
    destination_root: str | Path,
    *,
    task: FactorFactoryTask,
    snapshot: FactorFactorySnapshotManifest,
    compute: FactorFactoryComputeArtifacts,
    worker_id: str,
    worker_commit: str,
    worker_key_id: str,
    worker_signing_key: Ed25519PrivateKey,
    claimed_at: datetime,
    input_bytes: int,
    cache_hit_bytes: int,
    downloaded_bytes: int,
    peak_rss_bytes: int,
    compute_duration_seconds: float,
    max_result_bytes: int,
    max_value_partition_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES,
    max_value_partition_rows: int = 1_000_000,
    max_file_count: int = 20_000,
    max_uncompressed_bytes: int = DEFAULT_FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES,
) -> tuple[Path, FactorFactoryResultManifest, FactorFactoryWorkerReceipt]:
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / task.task_id
    if final.exists():
        manifest = FactorFactoryResultManifest.model_validate_json(
            (final / "manifest.json").read_text("utf-8")
        )
        receipt = FactorFactoryWorkerReceipt.model_validate_json(
            (final / "receipt.json").read_text("utf-8")
        )
        validate_factor_factory_result_bundle(
            final,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
            max_value_partition_bytes=max_value_partition_bytes,
            max_file_count=max_file_count,
            max_uncompressed_bytes=max_uncompressed_bytes,
        )
        return final, manifest, receipt
    temporary = root / f".ff-{task.task_id[-12:]}-{uuid.uuid4().hex[:8]}.partial"
    outputs_root = temporary / "outputs"
    reports_root = temporary / "reports"
    outputs_root.mkdir(parents=True, exist_ok=False)
    reports_root.mkdir(parents=True, exist_ok=False)
    outputs: list[FactorFactoryOutputDataset] = []
    partitions: list[FactorFactoryPartitionReference] = []
    reports: list[ResearchOutputFile] = []
    try:
        if compute.no_update_reason is None:
            frames = {
                "factor_definition_preview": compute.definitions,
                "factor_evidence": compute.evidence,
                "factor_correlation_daily": compute.correlations,
            }
            for dataset_name, schema, primary_keys in FACTOR_FACTORY_CONTROL_OUTPUTS:
                frame = frames[dataset_name]
                if set(frame.columns) != set(schema):
                    raise ValueError(f"factor_factory_output_schema_mismatch:{dataset_name}")
                normalized = frame.select(list(schema)).cast(schema, strict=True)
                path = outputs_root / f"{dataset_name}.parquet"
                normalized.write_parquet(path, compression="zstd")
                outputs.append(
                    FactorFactoryOutputDataset(
                        dataset_name=dataset_name,
                        relative_path=f"outputs/{dataset_name}.parquet",
                        schema_fingerprint=schema_fingerprint(normalized.schema),
                        sha256=sha256_file(path),
                        row_count=normalized.height,
                        size_bytes=path.stat().st_size,
                        primary_keys=primary_keys,
                    )
                )
            partitions = _write_factor_value_partitions(
                outputs_root,
                compute.values,
                max_partition_bytes=max_value_partition_bytes,
                max_partition_rows=max_value_partition_rows,
            )
        for name, payload in (
            ("factor_factory_worker_report.json", compute.worker_report),
            ("factor_factory_anti_leakage.json", compute.anti_leakage),
        ):
            path = reports_root / name
            path.write_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            reports.append(
                ResearchOutputFile(
                    relative_path=f"reports/{name}",
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
        completed_at = datetime.now(UTC)
        output_bytes = (
            sum(item.size_bytes for item in outputs)
            + sum(item.size_bytes for item in partitions)
            + sum(item.size_bytes for item in reports)
        )
        if output_bytes > max_result_bytes:
            raise RuntimeError("factor_factory_result_size_limit_exceeded")
        provisional = FactorFactoryResultManifest(
            schema_version=FACTOR_FACTORY_RESULT_SCHEMA,
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_manifest_sha256=snapshot.manifest_sha256,
            quant_lab_commit=task.quant_lab_commit,
            worker_commit=worker_commit,
            factor_plan_digest=task.factor_plan_digest,
            source_input_digest=task.source_input_digest,
            cost_input_digest=task.cost_input_digest,
            previous_generation_id=task.previous_generation_id,
            previous_generation_digest=task.previous_generation_digest,
            as_of_date=task.as_of_date,
            feature_set=task.feature_set,
            feature_version=task.feature_version,
            factor_version=task.factor_version,
            timeframe=task.timeframe,
            horizon_bars=task.horizon_bars,
            decision_delay_bars=task.decision_delay_bars,
            min_samples=task.min_samples,
            top_quantile=task.top_quantile,
            cost_quantile=task.cost_quantile,
            generation_id=f"factor-factory-{task.task_id.rsplit('-', 1)[-1]}",
            generated_at=compute.generated_at,
            completed_at=completed_at,
            factor_ids=compute.factor_ids,
            factor_count=len(compute.factor_ids),
            value_partitions=tuple(partitions),
            outputs=tuple(outputs),
            reports=tuple(reports),
            anti_leakage_checks=tuple(
                FactorFactoryAntiLeakageCheck.model_validate(item)
                for item in compute.anti_leakage["checks"]
            ),
            completed_no_update=compute.no_update_reason is not None,
            no_update_reason=compute.no_update_reason,
            warnings=compute.warnings,
            input_bytes=input_bytes,
            cache_hit_bytes=cache_hit_bytes,
            downloaded_bytes=downloaded_bytes,
            output_bytes=output_bytes,
            peak_rss_bytes=peak_rss_bytes,
            compute_duration_seconds=compute_duration_seconds,
            worker_key_id=worker_key_id,
            signature="pending",
        )
        manifest = provisional.model_copy(
            update={"signature": sign_model(provisional, worker_signing_key)}
        )
        (temporary / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        receipt_provisional = FactorFactoryWorkerReceipt(
            schema_version=FACTOR_FACTORY_RECEIPT_SCHEMA,
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            worker_id=worker_id,
            worker_commit=worker_commit,
            state="completed",
            claimed_at=claimed_at,
            completed_at=completed_at,
            result_manifest_sha256=sha256_file(temporary / "manifest.json"),
            output_rows=sum(item.row_count for item in outputs)
            + sum(item.row_count for item in partitions),
            input_bytes=input_bytes,
            downloaded_bytes=downloaded_bytes,
            cache_hit_bytes=cache_hit_bytes,
            anti_leakage_status="PASS",
            worker_key_id=worker_key_id,
            signature="pending",
        )
        receipt = receipt_provisional.model_copy(
            update={"signature": sign_model(receipt_provisional, worker_signing_key)}
        )
        (temporary / "receipt.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        validate_factor_factory_result_bundle(
            temporary,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
            max_value_partition_bytes=max_value_partition_bytes,
            max_file_count=max_file_count,
            max_uncompressed_bytes=max_uncompressed_bytes,
        )
        os.replace(temporary, final)
        return final, manifest, receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_factor_value_partitions(
    outputs_root: Path,
    values: pl.DataFrame | StagedFactorValueSet,
    *,
    max_partition_bytes: int,
    max_partition_rows: int,
) -> list[FactorFactoryPartitionReference]:
    if isinstance(values, StagedFactorValueSet):
        references: list[FactorFactoryPartitionReference] = []
        observed_rows = 0
        for staged_path in values.paths:
            frame = pl.read_parquet(staged_path)
            observed_rows += frame.height
            references.extend(
                _write_factor_value_partitions(
                    outputs_root,
                    frame,
                    max_partition_bytes=max_partition_bytes,
                    max_partition_rows=max_partition_rows,
                )
            )
            del frame
        if observed_rows != values.row_count:
            raise RuntimeError("factor_factory_staged_value_row_count_mismatch")
        return references
    if values.is_empty():
        return []
    if set(values.columns) != set(FACTOR_VALUE_SCHEMA):
        raise ValueError("factor_factory_output_schema_mismatch:factor_value")
    normalized = values.select(list(FACTOR_VALUE_SCHEMA)).cast(FACTOR_VALUE_SCHEMA, strict=True)
    version_values = normalized.get_column("factor_version").unique().to_list()
    timeframe_values = normalized.get_column("timeframe").unique().to_list()
    for value in [*version_values, *timeframe_values]:
        _require_safe_partition_segment(str(value))
    partition_keys = ["factor_version", "timeframe", "_partition_date"]
    with_dates = normalized.with_columns(pl.col("ts").dt.date().alias("_partition_date")).sort(
        [*partition_keys, "factor_id", "symbol", "ts"]
    )
    references: list[FactorFactoryPartitionReference] = []
    group_counts = with_dates.group_by(partition_keys, maintain_order=True).len()
    group_offset = 0
    for group in group_counts.iter_rows(named=True):
        factor_version = group["factor_version"]
        timeframe = group["timeframe"]
        partition_date = group["_partition_date"]
        group_rows = int(group["len"])
        frame = with_dates.slice(group_offset, group_rows).drop("_partition_date")
        group_offset += group_rows
        part_number = 0
        for offset in range(0, frame.height, max_partition_rows):
            chunk = frame.slice(offset, max_partition_rows)
            written = _write_bounded_factor_value_chunk(
                outputs_root,
                chunk,
                factor_version=str(factor_version),
                timeframe=str(timeframe),
                partition_date=partition_date,
                first_part_number=part_number,
                max_partition_bytes=max_partition_bytes,
            )
            references.extend(written)
            part_number += len(written)
    return references


def _write_bounded_factor_value_chunk(
    outputs_root: Path,
    frame: pl.DataFrame,
    *,
    factor_version: str,
    timeframe: str,
    partition_date: object,
    first_part_number: int,
    max_partition_bytes: int,
) -> list[FactorFactoryPartitionReference]:
    relative_parent = (
        Path("factor_value")
        / f"factor_version={factor_version}"
        / (f"timeframe={timeframe}")
        / f"date={partition_date.isoformat()}"
    )
    path = outputs_root / relative_parent / f"part-{first_part_number:05d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    if path.stat().st_size > max_partition_bytes:
        path.unlink()
        if frame.height <= 1:
            raise RuntimeError("factor_factory_value_partition_size_limit_exceeded")
        split = max(1, frame.height // 2)
        left = _write_bounded_factor_value_chunk(
            outputs_root,
            frame.slice(0, split),
            factor_version=factor_version,
            timeframe=timeframe,
            partition_date=partition_date,
            first_part_number=first_part_number,
            max_partition_bytes=max_partition_bytes,
        )
        right = _write_bounded_factor_value_chunk(
            outputs_root,
            frame.slice(split),
            factor_version=factor_version,
            timeframe=timeframe,
            partition_date=partition_date,
            first_part_number=first_part_number + len(left),
            max_partition_bytes=max_partition_bytes,
        )
        return [*left, *right]
    relative = str(path.relative_to(outputs_root.parent)).replace("\\", "/")
    file_sha256 = sha256_file(path)
    uncompressed_bytes = _parquet_uncompressed_bytes(path)
    min_ts = frame.get_column("ts").min()
    max_ts = frame.get_column("ts").max()
    partition_identity = model_content_sha256(
        {
            "schema_version": "quant_lab_factor_factory_partition_identity.v1",
            "factor_version": factor_version,
            "timeframe": timeframe,
            "partition_date": partition_date.isoformat(),
            "part_number": first_part_number,
            "sha256": file_sha256,
            "row_count": frame.height,
            "schema_fingerprint": schema_fingerprint(frame.schema),
            "min_ts": min_ts,
            "max_ts": max_ts,
        }
    )
    return [
        FactorFactoryPartitionReference(
            factor_version=factor_version,
            timeframe=timeframe,
            partition_date=partition_date,
            part_number=first_part_number,
            relative_path=relative,
            schema_fingerprint=schema_fingerprint(frame.schema),
            sha256=file_sha256,
            row_count=frame.height,
            size_bytes=path.stat().st_size,
            uncompressed_bytes=uncompressed_bytes,
            partition_identity=partition_identity,
            min_ts=min_ts,
            max_ts=max_ts,
        )
    ]


def _parquet_uncompressed_bytes(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(row_group).column(column).total_uncompressed_size
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _require_safe_partition_segment(value: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if not value or any(character not in allowed for character in value):
        raise ValueError("factor_factory_unsafe_partition_identity")


def write_factor_research_result_bundle(
    destination_root: str | Path,
    *,
    task: FactorResearchTask,
    snapshot: FactorResearchSnapshotManifest,
    compute: FactorResearchComputeArtifacts,
    worker_id: str,
    worker_commit: str,
    worker_key_id: str,
    worker_signing_key: Ed25519PrivateKey,
    claimed_at: datetime,
    input_bytes: int,
    cache_hit_bytes: int,
    downloaded_bytes: int,
    peak_rss_bytes: int,
    compute_duration_seconds: float,
    max_result_bytes: int,
) -> tuple[Path, FactorResearchResultManifest, FactorResearchWorkerReceipt]:
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / task.task_id
    if final.exists():
        manifest = FactorResearchResultManifest.model_validate_json(
            (final / "manifest.json").read_text("utf-8")
        )
        receipt = FactorResearchWorkerReceipt.model_validate_json(
            (final / "receipt.json").read_text("utf-8")
        )
        validate_factor_research_result_bundle(
            final,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        return final, manifest, receipt

    temporary = root / f".{task.task_id}.{uuid.uuid4().hex}.partial"
    outputs_root = temporary / "outputs"
    reports_root = temporary / "reports"
    outputs_root.mkdir(parents=True, exist_ok=False)
    reports_root.mkdir(parents=True, exist_ok=False)
    output_rows: list[ResearchOutputDataset] = []
    report_rows: list[ResearchOutputFile] = []
    try:
        frames = compute.frames_by_dataset()
        for spec in FACTOR_RESEARCH_OUTPUT_SPECS:
            frame = frames[spec.dataset_name]
            if set(frame.columns) != set(spec.schema):
                raise ValueError(
                    f"factor_research_output_schema_columns_mismatch:{spec.dataset_name}"
                )
            normalized = frame.select(list(spec.schema)).cast(spec.schema, strict=True)
            path = outputs_root / f"{spec.dataset_name}.parquet"
            normalized.write_parquet(path, compression="zstd")
            output_rows.append(
                ResearchOutputDataset(
                    dataset_name=spec.dataset_name,
                    relative_path=f"outputs/{path.name}",
                    schema_fingerprint=schema_fingerprint(normalized.schema),
                    sha256=sha256_file(path),
                    row_count=normalized.height,
                    size_bytes=path.stat().st_size,
                    publish_mode=spec.publish_mode,
                    primary_keys=list(spec.primary_keys),
                    window_keys=list(spec.window_keys),
                    empty_result_semantics=spec.empty_result_semantics,
                )
            )
        report_payloads = {
            "factor_research_worker_report.json": compute.worker_report,
            "factor_research_anti_leakage.json": compute.anti_leakage,
        }
        for name, payload in report_payloads.items():
            path = reports_root / name
            path.write_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            report_rows.append(
                ResearchOutputFile(
                    relative_path=f"reports/{name}",
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
        completed_at = datetime.now(UTC)
        output_bytes = sum(item.size_bytes for item in output_rows) + sum(
            item.size_bytes for item in report_rows
        )
        if output_bytes > max_result_bytes:
            raise RuntimeError("factor_research_result_size_limit_exceeded")
        provisional = FactorResearchResultManifest(
            schema_version=FACTOR_RESEARCH_RESULT_SCHEMA,
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_manifest_sha256=snapshot.manifest_sha256,
            selected_v5_bundle_id=task.selected_v5_bundle_id,
            quant_lab_commit=task.quant_lab_commit,
            worker_commit=worker_commit,
            factor_research_schema_version=task.factor_research_schema_version,
            hypothesis_registry_digest=task.hypothesis_registry_digest,
            trial_ledger_digest=task.trial_ledger_digest,
            source_input_digest=task.source_input_digest,
            as_of_date=task.as_of_date,
            start_date=task.start_date,
            end_date=task.end_date,
            max_history_days=task.max_history_days,
            hypothesis_ids=task.hypothesis_ids,
            trial_ids=task.trial_ids,
            test_count=task.test_count,
            multiple_testing_family="factor_research_v2.global_confirmatory",
            generation_id=f"factor-research-{task.task_id.rsplit('-', 1)[-1]}",
            generated_at=compute.generated_at,
            completed_at=completed_at,
            outputs=output_rows,
            reports=report_rows,
            anti_leakage_status="PASS",
            anti_leakage_violation_count=0,
            warnings=list(compute.warnings),
            input_bytes=input_bytes,
            cache_hit_bytes=cache_hit_bytes,
            downloaded_bytes=downloaded_bytes,
            output_bytes=output_bytes,
            peak_rss_bytes=peak_rss_bytes,
            compute_duration_seconds=compute_duration_seconds,
            worker_key_id=worker_key_id,
            signature="pending",
        )
        manifest = provisional.model_copy(
            update={"signature": sign_model(provisional, worker_signing_key)}
        )
        (temporary / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        receipt_provisional = FactorResearchWorkerReceipt(
            schema_version=FACTOR_RESEARCH_RECEIPT_SCHEMA,
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            worker_id=worker_id,
            worker_commit=worker_commit,
            state="completed",
            claimed_at=claimed_at,
            completed_at=completed_at,
            result_manifest_sha256=sha256_file(temporary / "manifest.json"),
            output_rows=sum(item.row_count for item in output_rows),
            input_bytes=input_bytes,
            downloaded_bytes=downloaded_bytes,
            cache_hit_bytes=cache_hit_bytes,
            anti_leakage_status="PASS",
            anti_leakage_violation_count=0,
            worker_key_id=worker_key_id,
            signature="pending",
        )
        receipt = receipt_provisional.model_copy(
            update={"signature": sign_model(receipt_provisional, worker_signing_key)}
        )
        (temporary / "receipt.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        validate_factor_research_result_bundle(
            temporary,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        os.replace(temporary, final)
        return final, manifest, receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_alpha_factory_result_bundle(
    destination_root: str | Path,
    *,
    task: AlphaFactoryTask,
    snapshot: AlphaFactorySnapshotManifest,
    compute: AlphaFactoryWorkerComputeResult,
    worker_id: str,
    worker_commit: str,
    worker_key_id: str,
    worker_signing_key: Ed25519PrivateKey,
    claimed_at: datetime,
    input_bytes: int,
    cache_hit_bytes: int,
    downloaded_bytes: int,
    peak_rss_bytes: int,
    compute_duration_seconds: float,
    max_result_bytes: int,
) -> tuple[Path, AlphaFactoryResultManifest, AlphaFactoryWorkerReceipt]:
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / task.task_id
    if final.exists():
        manifest = AlphaFactoryResultManifest.model_validate_json(
            (final / "manifest.json").read_text("utf-8")
        )
        receipt = AlphaFactoryWorkerReceipt.model_validate_json(
            (final / "receipt.json").read_text("utf-8")
        )
        validate_alpha_factory_result_bundle(
            final,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        return final, manifest, receipt
    temporary = root / f".{task.task_id}.{uuid.uuid4().hex}.partial"
    outputs_root = temporary / "outputs"
    reports_root = temporary / "reports"
    outputs_root.mkdir(parents=True, exist_ok=False)
    reports_root.mkdir(parents=True, exist_ok=False)
    output_rows: list[ResearchOutputDataset] = []
    report_rows: list[ResearchOutputFile] = []
    try:
        frames = compute.artifacts.frames_by_dataset()
        for spec in ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS:
            frame = frames[spec.dataset_name]
            if set(frame.columns) != set(spec.schema):
                raise ValueError(
                    f"alpha_factory_output_schema_columns_mismatch:{spec.dataset_name}"
                )
            normalized = frame.select(list(spec.schema)).cast(spec.schema, strict=True)
            path = outputs_root / f"{spec.dataset_name}.parquet"
            normalized.write_parquet(path, compression="zstd")
            output_rows.append(
                ResearchOutputDataset(
                    dataset_name=spec.dataset_name,
                    relative_path=f"outputs/{path.name}",
                    schema_fingerprint=schema_fingerprint(normalized.schema),
                    sha256=sha256_file(path),
                    row_count=normalized.height,
                    size_bytes=path.stat().st_size,
                    publish_mode=spec.publish_mode,
                    primary_keys=list(spec.primary_keys),
                    window_keys=list(spec.window_keys),
                    empty_result_semantics=spec.empty_result_semantics,
                )
            )
        factor_report = reports_root / "factor_strategy_bridge_candidates.csv"
        compute.artifacts.factor_strategy_bridge_candidates.write_csv(factor_report)
        worker_report = reports_root / "alpha_factory_worker_report.json"
        worker_report.write_text(
            json.dumps(compute.worker_report, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        anti_leakage = reports_root / "alpha_factory_anti_leakage.json"
        anti_leakage.write_text(
            json.dumps(compute.anti_leakage, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        for path in (factor_report, worker_report, anti_leakage):
            report_rows.append(
                ResearchOutputFile(
                    relative_path=f"reports/{path.name}",
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
        completed_at = datetime.now(UTC)
        generation_id = f"alpha-factory-{task.task_id.rsplit('-', 1)[-1]}"
        output_bytes = sum(item.size_bytes for item in output_rows) + sum(
            item.size_bytes for item in report_rows
        )
        if output_bytes > max_result_bytes:
            raise RuntimeError("alpha_factory_result_size_limit_exceeded")
        provisional = AlphaFactoryResultManifest(
            schema_version=ALPHA_FACTORY_RESULT_SCHEMA,
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_manifest_sha256=snapshot.manifest_sha256,
            selected_v5_bundle_id=task.selected_v5_bundle_id,
            quant_lab_commit=task.quant_lab_commit,
            worker_commit=worker_commit,
            alpha_factory_schema_version=task.alpha_factory_schema_version,
            second_stage_schema_version=task.second_stage_schema_version,
            template_registry_digest=task.template_registry_digest,
            factor_generation_id=task.factor_generation_id,
            factor_generation_digest=task.factor_generation_digest,
            factor_generation_as_of_date=task.factor_generation_as_of_date,
            factor_generation_published_at=task.factor_generation_published_at,
            hypothesis_registry_digest=task.hypothesis_registry_digest,
            trial_ledger_digest=task.trial_ledger_digest,
            factor_generation_fresh=task.factor_generation_fresh,
            factor_generation_hypothesis_ids=task.factor_generation_hypothesis_ids,
            as_of_date=task.as_of_date,
            lookback_days=task.lookback_days,
            max_candidates=task.max_candidates,
            generation_id=generation_id,
            generated_at=compute.artifacts.generated_at,
            completed_at=completed_at,
            outputs=output_rows,
            reports=report_rows,
            anti_leakage_status="PASS",
            anti_leakage_violation_count=0,
            warnings=list(compute.artifacts.warnings),
            input_bytes=input_bytes,
            cache_hit_bytes=cache_hit_bytes,
            downloaded_bytes=downloaded_bytes,
            output_bytes=output_bytes,
            peak_rss_bytes=peak_rss_bytes,
            compute_duration_seconds=compute_duration_seconds,
            worker_key_id=worker_key_id,
            signature="pending",
        )
        manifest = provisional.model_copy(
            update={"signature": sign_model(provisional, worker_signing_key)}
        )
        (temporary / "manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        receipt_provisional = AlphaFactoryWorkerReceipt(
            schema_version=ALPHA_FACTORY_RECEIPT_SCHEMA,
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            worker_id=worker_id,
            worker_commit=worker_commit,
            state="completed",
            claimed_at=claimed_at,
            completed_at=completed_at,
            result_manifest_sha256=sha256_file(temporary / "manifest.json"),
            output_rows=sum(item.row_count for item in output_rows),
            input_bytes=input_bytes,
            downloaded_bytes=downloaded_bytes,
            cache_hit_bytes=cache_hit_bytes,
            anti_leakage_status="PASS",
            anti_leakage_violation_count=0,
            worker_key_id=worker_key_id,
            signature="pending",
        )
        receipt = receipt_provisional.model_copy(
            update={"signature": sign_model(receipt_provisional, worker_signing_key)}
        )
        (temporary / "receipt.json").write_text(
            receipt.model_dump_json(indent=2),
            encoding="utf-8",
        )
        validate_alpha_factory_result_bundle(
            temporary,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        os.replace(temporary, final)
        return final, manifest, receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_entry_quality_history_result_bundle(
    destination_root: str | Path,
    *,
    task: ResearchTask,
    snapshot: ResearchSnapshotManifest,
    artifacts: EntryQualityHistoryArtifacts,
    worker_id: str,
    worker_commit: str,
    worker_key_id: str,
    worker_signing_key: Ed25519PrivateKey,
    claimed_at: datetime,
    input_bytes: int,
    cache_hit_bytes: int,
    downloaded_bytes: int,
    peak_rss_bytes: int,
    compute_duration_seconds: float,
    max_result_bytes: int,
) -> tuple[Path, ResearchResultManifest, ResearchWorkerReceipt]:
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / task.task_id
    if final.exists():
        manifest = ResearchResultManifest.model_validate_json(
            (final / "manifest.json").read_text("utf-8")
        )
        receipt = ResearchWorkerReceipt.model_validate_json(
            (final / "receipt.json").read_text("utf-8")
        )
        validate_entry_quality_history_result_bundle(
            final,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        return final, manifest, receipt
    temporary = root / f".{task.task_id}.{uuid.uuid4().hex}.partial"
    outputs_root = temporary / "outputs"
    reports_root = temporary / "reports"
    outputs_root.mkdir(parents=True, exist_ok=False)
    reports_root.mkdir(parents=True, exist_ok=False)
    output_rows: list[ResearchOutputDataset] = []
    report_rows: list[ResearchOutputFile] = []
    try:
        frames = artifacts.frames_by_dataset()
        for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS:
            frame = _normalize_output_frame(frames[spec.dataset_name], spec)
            path = outputs_root / f"{spec.dataset_name}.parquet"
            frame.write_parquet(path)
            output_rows.append(
                ResearchOutputDataset(
                    dataset_name=spec.dataset_name,
                    relative_path=f"outputs/{path.name}",
                    schema_fingerprint=schema_fingerprint(frame.schema),
                    sha256=sha256_file(path),
                    row_count=frame.height,
                    size_bytes=path.stat().st_size,
                    publish_mode=spec.publish_mode,
                    primary_keys=list(spec.primary_keys),
                    window_keys=list(spec.window_keys),
                    empty_result_semantics=spec.empty_result_semantics,
                )
            )
        for name, payload in sorted(artifacts.reports.items()):
            path = reports_root / name
            path.write_bytes(payload)
            report_rows.append(
                ResearchOutputFile(
                    relative_path=f"reports/{name}",
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
        completed_at = datetime.now(UTC)
        generation_id = f"entry-quality-history-{task.task_id.rsplit('-', 1)[-1]}"
        output_bytes = sum(item.size_bytes for item in output_rows) + sum(
            item.size_bytes for item in report_rows
        )
        provisional = ResearchResultManifest(
            schema_version=RESEARCH_RESULT_SCHEMA,
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_manifest_sha256=snapshot.manifest_sha256,
            selected_v5_bundle_id=task.selected_v5_bundle_id,
            quant_lab_commit=task.quant_lab_commit,
            worker_commit=worker_commit,
            entry_quality_schema_version=task.entry_quality_schema_version,
            start_date=task.parameters.start_date,
            end_date=task.parameters.end_date,
            mode=task.parameters.mode,
            cost_mode=task.parameters.cost_mode,
            window_hours=task.parameters.window_hours,
            generation_id=generation_id,
            generated_at=artifacts.generated_at,
            completed_at=completed_at,
            outputs=output_rows,
            reports=report_rows,
            anti_leakage_status=_anti_leakage_status(artifacts.anti_leakage_check),
            warnings=list(artifacts.warnings),
            input_bytes=input_bytes,
            cache_hit_bytes=cache_hit_bytes,
            downloaded_bytes=downloaded_bytes,
            output_bytes=output_bytes,
            peak_rss_bytes=peak_rss_bytes,
            compute_duration_seconds=compute_duration_seconds,
            worker_key_id=worker_key_id,
            signature="pending",
        )
        manifest = provisional.model_copy(
            update={"signature": sign_model(provisional, worker_signing_key)}
        )
        (temporary / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        manifest_file_sha = sha256_file(temporary / "manifest.json")
        receipt_provisional = ResearchWorkerReceipt(
            schema_version=RESEARCH_RECEIPT_SCHEMA,
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            worker_id=worker_id,
            worker_commit=worker_commit,
            state="completed",
            claimed_at=claimed_at,
            completed_at=completed_at,
            result_manifest_sha256=manifest_file_sha,
            output_rows=sum(item.row_count for item in output_rows),
            input_bytes=input_bytes,
            downloaded_bytes=downloaded_bytes,
            cache_hit_bytes=cache_hit_bytes,
            anti_leakage_status=manifest.anti_leakage_status,
            worker_key_id=worker_key_id,
            signature="pending",
        )
        receipt = receipt_provisional.model_copy(
            update={"signature": sign_model(receipt_provisional, worker_signing_key)}
        )
        (temporary / "receipt.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        validate_entry_quality_history_result_bundle(
            temporary,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
        )
        os.replace(temporary, final)
        return final, manifest, receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _normalize_output_frame(
    frame: pl.DataFrame,
    spec: EntryQualityHistoryOutputSpec,
) -> pl.DataFrame:
    schema = spec.schema
    if set(frame.columns) != set(schema):
        raise ValueError(f"entry_quality_output_schema_columns_mismatch:{spec.dataset_name}")
    return frame.select(list(schema)).cast(schema, strict=True)


def _anti_leakage_status(frame: pl.DataFrame) -> str:
    if frame.is_empty() or "status" not in frame.columns or "violation_count" not in frame.columns:
        raise ValueError("anti_leakage_missing")
    statuses = {str(value).upper() for value in frame.get_column("status").to_list()}
    violations = sum(int(value or 0) for value in frame.get_column("violation_count").to_list())
    if statuses != {"PASS"} or violations != 0:
        raise ValueError("anti_leakage_not_pass")
    return "PASS"
