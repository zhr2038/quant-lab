from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.research.candidate_labels import LABEL_SCHEMA
from quant_lab.research.strategy_evidence import SAMPLE_SCHEMA
from quant_lab.research_plane.result import schema_fingerprint
from quant_lab.research_plane.signatures import model_content_sha256, sha256_file, sign_model
from quant_lab.research_plane.v5_candidate_evidence_contracts import (
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_FILE_COUNT,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_UNCOMPRESSED_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_UNCOMPRESSED_BYTES,
    V5_CANDIDATE_LABEL_DELTA_PRIMARY_KEYS,
    V5_STRATEGY_EVIDENCE_SAMPLE_DELTA_PRIMARY_KEYS,
    V5CandidateEvidenceAntiLeakageCheck,
    V5CandidateEvidenceOutputDataset,
    V5CandidateEvidenceReportFile,
    V5CandidateEvidenceResultManifest,
    V5CandidateEvidenceSnapshotManifest,
    V5CandidateEvidenceTask,
    V5CandidateEvidenceWorkerReceipt,
)
from quant_lab.research_plane.v5_candidate_evidence_result import (
    validate_v5_candidate_evidence_result_bundle,
)
from quant_lab.research_worker.v5_candidate_evidence import (
    StagedV5CandidateEvidenceDataset,
    V5CandidateEvidenceComputeArtifacts,
)


def write_v5_candidate_evidence_result_bundle(
    destination_root: str | Path,
    *,
    task: V5CandidateEvidenceTask,
    snapshot: V5CandidateEvidenceSnapshotManifest,
    compute: V5CandidateEvidenceComputeArtifacts,
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
    max_result_bytes: int = DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_BYTES,
    max_result_uncompressed_bytes: int = (
        DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_UNCOMPRESSED_BYTES
    ),
    max_partition_bytes: int = DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_BYTES,
    max_partition_uncompressed_bytes: int = (
        DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_UNCOMPRESSED_BYTES
    ),
    max_file_count: int = DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_FILE_COUNT,
    max_partition_rows: int = 250_000,
) -> tuple[Path, V5CandidateEvidenceResultManifest, V5CandidateEvidenceWorkerReceipt]:
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / task.task_id
    if final.exists():
        manifest = V5CandidateEvidenceResultManifest.model_validate_json(
            (final / "manifest.json").read_text("utf-8")
        )
        receipt = V5CandidateEvidenceWorkerReceipt.model_validate_json(
            (final / "receipt.json").read_text("utf-8")
        )
        validate_v5_candidate_evidence_result_bundle(
            final,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
            max_result_uncompressed_bytes=max_result_uncompressed_bytes,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=max_partition_uncompressed_bytes,
            max_file_count=max_file_count,
        )
        return final, manifest, receipt

    temporary = root / f".v5ce-{task.task_id[-12:]}-{uuid.uuid4().hex[:8]}.partial"
    outputs_root = temporary / "outputs"
    reports_root = temporary / "reports"
    outputs_root.mkdir(parents=True, exist_ok=False)
    reports_root.mkdir(parents=True, exist_ok=False)
    try:
        outputs: list[V5CandidateEvidenceOutputDataset] = []
        outputs.extend(
            _write_dataset_partitions(
                outputs_root,
                compute.labels,
                schema=LABEL_SCHEMA,
                primary_keys=V5_CANDIDATE_LABEL_DELTA_PRIMARY_KEYS,
                empty_partition_date=task.as_of_date,
                max_partition_bytes=max_partition_bytes,
                max_partition_uncompressed_bytes=max_partition_uncompressed_bytes,
                max_partition_rows=max_partition_rows,
            )
        )
        outputs.extend(
            _write_dataset_partitions(
                outputs_root,
                compute.samples,
                schema=SAMPLE_SCHEMA,
                primary_keys=V5_STRATEGY_EVIDENCE_SAMPLE_DELTA_PRIMARY_KEYS,
                empty_partition_date=task.as_of_date,
                max_partition_bytes=max_partition_bytes,
                max_partition_uncompressed_bytes=max_partition_uncompressed_bytes,
                max_partition_rows=max_partition_rows,
            )
        )

        reports: list[V5CandidateEvidenceReportFile] = []
        for relative_path, payload in (
            (
                "reports/v5_candidate_evidence_worker_report.json",
                compute.worker_report,
            ),
            (
                "reports/v5_candidate_evidence_anti_leakage.json",
                compute.anti_leakage,
            ),
        ):
            path = temporary / relative_path
            path.write_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            reports.append(
                V5CandidateEvidenceReportFile(
                    relative_path=relative_path,
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
        if 2 + len(outputs) + len(reports) > max_file_count:
            raise RuntimeError("v5_candidate_evidence_result_file_count_limit_exceeded")
        output_bytes = sum(item.size_bytes for item in outputs) + sum(
            item.size_bytes for item in reports
        )
        output_uncompressed_bytes = sum(item.uncompressed_bytes for item in outputs) + sum(
            item.size_bytes for item in reports
        )
        if output_bytes > max_result_bytes:
            raise RuntimeError("v5_candidate_evidence_result_size_limit_exceeded")
        if output_uncompressed_bytes > max_result_uncompressed_bytes:
            raise RuntimeError("v5_candidate_evidence_result_uncompressed_size_limit_exceeded")

        completed_at = datetime.now(UTC)
        generation_id = (
            "v5-candidate-evidence-"
            + model_content_sha256(
                {
                    "task_id": task.task_id,
                    "snapshot_id": task.snapshot_id,
                    "previous_generation_id": task.previous_generation_id,
                    "previous_generation_digest": task.previous_generation_digest,
                }
            )[:24]
        )
        provisional = V5CandidateEvidenceResultManifest(
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_manifest_sha256=snapshot.manifest_sha256,
            quant_lab_commit=task.quant_lab_commit,
            worker_commit=worker_commit,
            input_fingerprint_digest=task.input_fingerprint_digest,
            candidate_event_digest=snapshot.candidate_event_digest,
            market_bar_digest=snapshot.market_bar_digest,
            run_summary_digest=snapshot.run_summary_digest,
            previous_generation_id=task.previous_generation_id,
            previous_generation_digest=task.previous_generation_digest,
            as_of_date=task.as_of_date,
            mode=task.mode,
            lookback_days=task.lookback_days,
            horizon_hours=task.horizon_hours,
            include_historical_outcomes=task.include_historical_outcomes,
            candidate_label_schema_version=task.candidate_label_schema_version,
            strategy_evidence_version=task.strategy_evidence_version,
            generation_id=generation_id,
            generated_at=compute.generated_at,
            completed_at=completed_at,
            outputs=tuple(outputs),
            reports=tuple(reports),
            anti_leakage_checks=tuple(
                V5CandidateEvidenceAntiLeakageCheck.model_validate(item)
                for item in compute.anti_leakage["checks"]
            ),
            warnings=compute.warnings,
            input_bytes=input_bytes,
            input_uncompressed_bytes=snapshot.estimated_uncompressed_bytes,
            cache_hit_bytes=cache_hit_bytes,
            downloaded_bytes=downloaded_bytes,
            output_bytes=output_bytes,
            output_uncompressed_bytes=output_uncompressed_bytes,
            peak_rss_bytes=max(peak_rss_bytes, compute.peak_rss_bytes),
            temporary_disk_peak_bytes=compute.temporary_disk_peak_bytes,
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
        receipt_provisional = V5CandidateEvidenceWorkerReceipt(
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            worker_id=worker_id,
            worker_commit=worker_commit,
            state="completed",
            claimed_at=claimed_at,
            completed_at=completed_at,
            result_manifest_sha256=sha256_file(temporary / "manifest.json"),
            output_rows=sum(item.row_count for item in outputs),
            input_bytes=input_bytes,
            downloaded_bytes=downloaded_bytes,
            cache_hit_bytes=cache_hit_bytes,
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
        validate_v5_candidate_evidence_result_bundle(
            temporary,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_signing_key.public_key(),
            expected_worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
            max_result_uncompressed_bytes=max_result_uncompressed_bytes,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=max_partition_uncompressed_bytes,
            max_file_count=max_file_count,
        )
        os.replace(temporary, final)
        return final, manifest, receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_dataset_partitions(
    outputs_root: Path,
    staged: StagedV5CandidateEvidenceDataset,
    *,
    schema: dict[str, Any],
    primary_keys: tuple[str, ...],
    empty_partition_date: date,
    max_partition_bytes: int,
    max_partition_uncompressed_bytes: int,
    max_partition_rows: int,
) -> list[V5CandidateEvidenceOutputDataset]:
    references: list[V5CandidateEvidenceOutputDataset] = []
    frames: list[tuple[date, pl.DataFrame]] = []
    for source in sorted(staged.paths):
        frame = pl.read_parquet(source).select(list(schema)).cast(schema, strict=True)
        partition_date = _partition_date_from_path(source, frame, empty_partition_date)
        for offset in range(0, max(frame.height, 1), max_partition_rows):
            frames.append((partition_date, frame.slice(offset, max_partition_rows)))
    if not frames:
        frames.append((empty_partition_date, pl.DataFrame(schema=schema)))
    for partition_date, frame in frames:
        _write_bounded_frame(
            outputs_root,
            staged.dataset_name,
            partition_date,
            frame,
            schema=schema,
            primary_keys=primary_keys,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=max_partition_uncompressed_bytes,
            references=references,
        )
    return references


def _write_bounded_frame(
    outputs_root: Path,
    dataset_name: str,
    partition_date: date,
    frame: pl.DataFrame,
    *,
    schema: dict[str, Any],
    primary_keys: tuple[str, ...],
    max_partition_bytes: int,
    max_partition_uncompressed_bytes: int,
    references: list[V5CandidateEvidenceOutputDataset],
) -> None:
    normalized = frame.select(list(schema)).cast(schema, strict=True)
    part_number = len(references)
    relative_path = (
        f"outputs/{dataset_name}/date={partition_date.isoformat()}/"
        f"part-{part_number:05d}.parquet"
    )
    path = outputs_root.parent / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_parquet(path, compression="zstd")
    uncompressed = _parquet_uncompressed_bytes(path)
    if (
        path.stat().st_size > max_partition_bytes
        or uncompressed > max_partition_uncompressed_bytes
    ):
        path.unlink()
        if normalized.height <= 1:
            raise RuntimeError("v5_candidate_evidence_partition_size_limit_exceeded")
        midpoint = normalized.height // 2
        _write_bounded_frame(
            outputs_root,
            dataset_name,
            partition_date,
            normalized.slice(0, midpoint),
            schema=schema,
            primary_keys=primary_keys,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=max_partition_uncompressed_bytes,
            references=references,
        )
        _write_bounded_frame(
            outputs_root,
            dataset_name,
            partition_date,
            normalized.slice(midpoint),
            schema=schema,
            primary_keys=primary_keys,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=max_partition_uncompressed_bytes,
            references=references,
        )
        return
    lower, upper = _timestamp_bounds(normalized, "ts_utc")
    sha = sha256_file(path)
    fingerprint = schema_fingerprint(normalized.schema)
    identity = model_content_sha256(
        {
            "schema_version": "v5_candidate_evidence_partition_identity.v1",
            "dataset_name": dataset_name,
            "partition_date": partition_date,
            "part_number": part_number,
            "sha256": sha,
            "row_count": normalized.height,
            "schema_fingerprint": fingerprint,
            "min_ts": lower,
            "max_ts": upper,
        }
    )
    references.append(
        V5CandidateEvidenceOutputDataset(
            dataset_name=dataset_name,
            partition_date=partition_date,
            part_number=part_number,
            relative_path=relative_path,
            sha256=sha,
            size_bytes=path.stat().st_size,
            uncompressed_bytes=uncompressed,
            row_count=normalized.height,
            schema_fingerprint=fingerprint,
            min_ts=lower,
            max_ts=upper,
            partition_identity=identity,
            primary_keys=primary_keys,
        )
    )


def _partition_date_from_path(
    path: Path,
    frame: pl.DataFrame,
    fallback: date,
) -> date:
    for part in path.parts:
        if part.startswith("date="):
            try:
                return date.fromisoformat(part.removeprefix("date="))
            except ValueError:
                break
    if not frame.is_empty() and "ts_utc" in frame.columns:
        value = frame.select(pl.col("ts_utc").min()).item()
        if isinstance(value, datetime):
            return value.date()
    return fallback


def _timestamp_bounds(
    frame: pl.DataFrame,
    column: str,
) -> tuple[datetime | None, datetime | None]:
    if frame.is_empty() or column not in frame.columns:
        return None, None
    bounds = frame.select(
        [pl.col(column).min().alias("lower"), pl.col(column).max().alias("upper")]
    )
    return bounds.item(0, "lower"), bounds.item(0, "upper")


def _parquet_uncompressed_bytes(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(group).column(column).total_uncompressed_size
        for group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )
