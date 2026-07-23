from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.research_plane.result import schema_fingerprint
from quant_lab.research_plane.signatures import (
    model_content_sha256,
    sha256_file,
    sign_model,
)
from quant_lab.research_plane.trade_level_history_contracts import (
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_FILE_COUNT,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_UNCOMPRESSED_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_UNCOMPRESSED_BYTES,
    TRADE_LEVEL_HISTORY_PRIMARY_KEYS,
    TradeLevelHistoryAntiLeakageCheck,
    TradeLevelHistoryOutputDataset,
    TradeLevelHistoryReportFile,
    TradeLevelHistoryResultManifest,
    TradeLevelHistorySnapshotManifest,
    TradeLevelHistoryTask,
    TradeLevelHistoryWorkerReceipt,
)
from quant_lab.trade_level.labels import TRADE_OPPORTUNITY_LABEL_SCHEMA
from quant_lab.trade_level.similarity import TRADE_LEVEL_SIMILARITY_SCHEMA

from .trade_level_history import (
    StagedTradeLevelHistoryDataset,
    TradeLevelHistoryComputeArtifacts,
)

_OUTPUT_SCHEMAS = {
    "trade_opportunity_label": TRADE_OPPORTUNITY_LABEL_SCHEMA,
    "trade_level_similarity_outcome": TRADE_LEVEL_SIMILARITY_SCHEMA,
}


def write_trade_level_history_result_bundle(
    destination_root: str | Path,
    *,
    task: TradeLevelHistoryTask,
    snapshot: TradeLevelHistorySnapshotManifest,
    snapshot_root: str | Path,
    compute: TradeLevelHistoryComputeArtifacts,
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
    max_result_bytes: int = DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_BYTES,
    max_result_uncompressed_bytes: int = (
        DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_UNCOMPRESSED_BYTES
    ),
    max_partition_bytes: int = DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_BYTES,
    max_partition_uncompressed_bytes: int = (
        DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_UNCOMPRESSED_BYTES
    ),
    max_file_count: int = DEFAULT_TRADE_LEVEL_HISTORY_MAX_FILE_COUNT,
    max_partition_rows: int = 250_000,
) -> tuple[
    Path,
    TradeLevelHistoryResultManifest,
    TradeLevelHistoryWorkerReceipt,
]:
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / task.task_id
    if final.exists():
        manifest = TradeLevelHistoryResultManifest.model_validate_json(
            (final / "manifest.json").read_text("utf-8")
        )
        receipt = TradeLevelHistoryWorkerReceipt.model_validate_json(
            (final / "receipt.json").read_text("utf-8")
        )
        _validate_existing_bundle(
            final,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            snapshot_root=snapshot_root,
            worker_signing_key=worker_signing_key,
            worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
            max_result_uncompressed_bytes=max_result_uncompressed_bytes,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=max_partition_uncompressed_bytes,
            max_file_count=max_file_count,
        )
        return final, manifest, receipt

    temporary = (
        root
        / f".trade-level-{task.task_id[-12:]}-{uuid.uuid4().hex[:8]}.partial"
    )
    (temporary / "outputs").mkdir(parents=True, exist_ok=False)
    (temporary / "reports").mkdir(parents=True, exist_ok=False)
    try:
        outputs: list[TradeLevelHistoryOutputDataset] = []
        for staged in (compute.labels, compute.similarity):
            outputs.extend(
                _write_dataset_partitions(
                    temporary,
                    staged,
                    schema=_OUTPUT_SCHEMAS[staged.dataset_name],
                    primary_keys=TRADE_LEVEL_HISTORY_PRIMARY_KEYS[
                        staged.dataset_name
                    ],
                    max_partition_bytes=max_partition_bytes,
                    max_partition_uncompressed_bytes=(
                        max_partition_uncompressed_bytes
                    ),
                    max_partition_rows=max_partition_rows,
                )
            )

        reports: list[TradeLevelHistoryReportFile] = []
        for relative_path, payload in (
            (
                "reports/trade_level_history_worker_report.json",
                compute.worker_report,
            ),
            (
                "reports/trade_level_history_anti_leakage.json",
                compute.anti_leakage,
            ),
        ):
            path = temporary / relative_path
            path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            reports.append(
                TradeLevelHistoryReportFile(
                    relative_path=relative_path,
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
        if 2 + len(outputs) + len(reports) > max_file_count:
            raise RuntimeError(
                "trade_level_history_result_file_count_limit_exceeded"
            )
        output_bytes = sum(item.size_bytes for item in outputs) + sum(
            item.size_bytes for item in reports
        )
        output_uncompressed_bytes = sum(
            item.uncompressed_bytes for item in outputs
        ) + sum(item.size_bytes for item in reports)
        if output_bytes > max_result_bytes:
            raise RuntimeError("trade_level_history_result_size_limit_exceeded")
        if output_uncompressed_bytes > max_result_uncompressed_bytes:
            raise RuntimeError(
                "trade_level_history_result_uncompressed_size_limit_exceeded"
            )

        completed_at = datetime.now(UTC)
        generation_id = "trade-level-history-" + model_content_sha256(
            {
                "schema_version": "trade_level_history_result_identity.v1",
                "task_id": task.task_id,
                "snapshot_id": task.snapshot_id,
                "previous_generation_id": task.previous_generation_id,
                "previous_generation_digest": task.previous_generation_digest,
            }
        )[:24]
        provisional = TradeLevelHistoryResultManifest(
            **task.parameters.model_dump(),
            task_id=task.task_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_manifest_sha256=snapshot.manifest_sha256,
            quant_lab_commit=task.quant_lab_commit,
            worker_commit=worker_commit,
            input_fingerprint_digest=task.input_fingerprint_digest,
            derived_event_digest=(
                snapshot.derived_trade_opportunity_event_digest
            ),
            candidate_label_dataset_hash=(
                snapshot.candidate_label_dataset_hash
            ),
            previous_generation_id=task.previous_generation_id,
            previous_generation_digest=task.previous_generation_digest,
            generation_id=generation_id,
            generated_at=compute.generated_at,
            completed_at=completed_at,
            outputs=tuple(outputs),
            reports=tuple(reports),
            anti_leakage_checks=tuple(
                TradeLevelHistoryAntiLeakageCheck.model_validate(item)
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
        receipt_provisional = TradeLevelHistoryWorkerReceipt(
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
            anti_leakage_status="PASS",
            anti_leakage_violation_count=0,
            worker_key_id=worker_key_id,
            signature="pending",
        )
        receipt = receipt_provisional.model_copy(
            update={
                "signature": sign_model(
                    receipt_provisional,
                    worker_signing_key,
                )
            }
        )
        (temporary / "receipt.json").write_text(
            receipt.model_dump_json(indent=2),
            encoding="utf-8",
        )
        _validate_existing_bundle(
            temporary,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            snapshot_root=snapshot_root,
            worker_signing_key=worker_signing_key,
            worker_key_id=worker_key_id,
            max_result_bytes=max_result_bytes,
            max_result_uncompressed_bytes=max_result_uncompressed_bytes,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=(
                max_partition_uncompressed_bytes
            ),
            max_file_count=max_file_count,
        )
        os.replace(temporary, final)
        return final, manifest, receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_existing_bundle(
    root: Path,
    *,
    manifest: TradeLevelHistoryResultManifest,
    receipt: TradeLevelHistoryWorkerReceipt,
    task: TradeLevelHistoryTask,
    snapshot: TradeLevelHistorySnapshotManifest,
    snapshot_root: str | Path,
    worker_signing_key: Ed25519PrivateKey,
    worker_key_id: str,
    max_result_bytes: int,
    max_result_uncompressed_bytes: int,
    max_partition_bytes: int,
    max_partition_uncompressed_bytes: int,
    max_file_count: int,
) -> None:
    from quant_lab.research_plane.trade_level_history_result import (  # noqa: PLC0415
        validate_trade_level_history_result_bundle,
    )

    validate_trade_level_history_result_bundle(
        root,
        manifest=manifest,
        receipt=receipt,
        task=task,
        snapshot=snapshot,
        snapshot_root=snapshot_root,
        worker_public_key=worker_signing_key.public_key(),
        expected_worker_key_id=worker_key_id,
        max_result_bytes=max_result_bytes,
        max_result_uncompressed_bytes=max_result_uncompressed_bytes,
        max_partition_bytes=max_partition_bytes,
        max_partition_uncompressed_bytes=(
            max_partition_uncompressed_bytes
        ),
        max_file_count=max_file_count,
    )


def _write_dataset_partitions(
    bundle_root: Path,
    staged: StagedTradeLevelHistoryDataset,
    *,
    schema: dict[str, Any],
    primary_keys: tuple[str, ...],
    max_partition_bytes: int,
    max_partition_uncompressed_bytes: int,
    max_partition_rows: int,
) -> list[TradeLevelHistoryOutputDataset]:
    references: list[TradeLevelHistoryOutputDataset] = []
    wrote_partition = False
    for source in sorted(staged.paths):
        frame = (
            pl.read_parquet(source)
            .select(list(schema))
            .cast(schema, strict=True)
        )
        symbols = frame.get_column("symbol").drop_nulls().unique().to_list()
        if len(symbols) != 1:
            raise RuntimeError(
                "trade_level_history_stage_symbol_partition_mismatch"
            )
        symbol = str(symbols[0])
        for offset in range(0, max(frame.height, 1), max_partition_rows):
            _write_bounded_frame(
                bundle_root,
                staged.dataset_name,
                symbol,
                frame.slice(offset, max_partition_rows),
                schema=schema,
                primary_keys=primary_keys,
                max_partition_bytes=max_partition_bytes,
                max_partition_uncompressed_bytes=(
                    max_partition_uncompressed_bytes
                ),
                references=references,
            )
            wrote_partition = True
        del frame
    if not wrote_partition:
        _write_bounded_frame(
            bundle_root,
            staged.dataset_name,
            "EMPTY",
            pl.DataFrame(schema=schema),
            schema=schema,
            primary_keys=primary_keys,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=(
                max_partition_uncompressed_bytes
            ),
            references=references,
        )
    return references


def _write_bounded_frame(
    bundle_root: Path,
    dataset_name: str,
    symbol: str,
    frame: pl.DataFrame,
    *,
    schema: dict[str, Any],
    primary_keys: tuple[str, ...],
    max_partition_bytes: int,
    max_partition_uncompressed_bytes: int,
    references: list[TradeLevelHistoryOutputDataset],
) -> None:
    normalized = frame.select(list(schema)).cast(schema, strict=True)
    part_number = len(references)
    token = _safe_symbol_token(symbol)
    relative_path = (
        f"outputs/{dataset_name}/symbol={token}/part-{part_number:05d}.parquet"
    )
    path = bundle_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_parquet(path, compression="zstd")
    uncompressed = _parquet_uncompressed_bytes(path)
    if (
        path.stat().st_size > max_partition_bytes
        or uncompressed > max_partition_uncompressed_bytes
    ):
        path.unlink()
        if normalized.height <= 1:
            raise RuntimeError(
                "trade_level_history_partition_size_limit_exceeded"
            )
        midpoint = normalized.height // 2
        _write_bounded_frame(
            bundle_root,
            dataset_name,
            symbol,
            normalized.slice(0, midpoint),
            schema=schema,
            primary_keys=primary_keys,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=(
                max_partition_uncompressed_bytes
            ),
            references=references,
        )
        _write_bounded_frame(
            bundle_root,
            dataset_name,
            symbol,
            normalized.slice(midpoint),
            schema=schema,
            primary_keys=primary_keys,
            max_partition_bytes=max_partition_bytes,
            max_partition_uncompressed_bytes=(
                max_partition_uncompressed_bytes
            ),
            references=references,
        )
        return
    lower, upper = _timestamp_bounds(normalized)
    sha = sha256_file(path)
    fingerprint = schema_fingerprint(normalized.schema)
    identity = model_content_sha256(
        {
            "schema_version": (
                "trade_level_history_partition_identity.v1"
            ),
            "dataset_name": dataset_name,
            "partition_symbol": symbol,
            "part_number": part_number,
            "sha256": sha,
            "row_count": normalized.height,
            "schema_fingerprint": fingerprint,
            "min_ts": lower,
            "max_ts": upper,
        }
    )
    references.append(
        TradeLevelHistoryOutputDataset(
            dataset_name=dataset_name,
            partition_symbol=symbol,
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


def _timestamp_bounds(
    frame: pl.DataFrame,
) -> tuple[datetime | None, datetime | None]:
    if frame.is_empty():
        return None, None
    bounds = frame.select(
        pl.col("decision_ts").min().alias("lower"),
        pl.col("decision_ts").max().alias("upper"),
    )
    return bounds.item(0, "lower"), bounds.item(0, "upper")


def _safe_symbol_token(symbol: str) -> str:
    readable = "".join(
        character.lower() if character.isalnum() else "-"
        for character in symbol
    )
    readable = "-".join(part for part in readable.split("-") if part) or "empty"
    digest = hashlib.sha256(symbol.encode()).hexdigest()[:8]
    return f"{readable[:48]}-{digest}"


def _parquet_uncompressed_bytes(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(group).column(column).total_uncompressed_size
        for group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )
