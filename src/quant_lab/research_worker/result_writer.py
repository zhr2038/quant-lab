from __future__ import annotations

import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.research.entry_quality import (
    ENTRY_QUALITY_HISTORY_OUTPUT_SPECS,
    EntryQualityHistoryArtifacts,
    EntryQualityHistoryOutputSpec,
)
from quant_lab.research_plane.contracts import (
    RESEARCH_RECEIPT_SCHEMA,
    RESEARCH_RESULT_SCHEMA,
    ResearchOutputDataset,
    ResearchOutputFile,
    ResearchResultManifest,
    ResearchSnapshotManifest,
    ResearchTask,
    ResearchWorkerReceipt,
)
from quant_lab.research_plane.result import (
    schema_fingerprint,
    validate_entry_quality_history_result_bundle,
)
from quant_lab.research_plane.signatures import (
    sha256_file,
    sign_model,
)


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
