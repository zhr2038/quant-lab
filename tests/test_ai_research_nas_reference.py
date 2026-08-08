from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

from quant_lab.ai_research.contracts import compute_task_packet_sha256
from quant_lab.ai_research.packet import (
    ai_task_retry_status,
    build_task_from_nas_pack_reference,
    hydrate_nas_ai_research_task,
)
from quant_lab.export_plane.contracts import ExportPackIndexEntry
from tools.quant_ai_queue import main as quant_ai_queue_main


def test_ai_queue_reports_missing_accepted_source_pack(tmp_path, capsys) -> None:
    exit_code = quant_ai_queue_main(
        [
            "build-task",
            "--export-queue-root",
            str(tmp_path / "export-queue"),
            "--queue-root",
            str(tmp_path / "ai-queue"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["created"] is False
    assert payload["no_task_reason"] == "NO_ACCEPTED_SOURCE_PACK"


def test_nas_ai_task_sends_reference_then_hydrates_locally(tmp_path) -> None:
    pack_path = tmp_path / "expert.zip"
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps({"authoritative_snapshot": True}))
        archive.writestr("provenance.json", json.dumps({"source": "snapshot"}))
        archive.writestr("data_quality.json", json.dumps({"status": "PASS"}))
        archive.writestr(
            "reports/factor_evidence.csv",
            "factor_id,rank_ic_mean\nfactor-1,0.1\n",
        )
    digest = hashlib.sha256(pack_path.read_bytes()).hexdigest()
    row = ExportPackIndexEntry(
        pack_id="expert-pack-test",
        pack_name=pack_path.name,
        export_date=date(2026, 7, 16),
        generated_at=datetime(2026, 7, 16, tzinfo=UTC),
        accepted_at=datetime(2026, 7, 16, tzinfo=UTC),
        pack_sha256=digest,
        pack_size_bytes=pack_path.stat().st_size,
        snapshot_id="export-snapshot-test",
        authoritative_input_snapshot=True,
        nas_artifact_validated=True,
        control_plane_receipt_verified=True,
        download_ready=True,
        download_relative_path="2026/07/16/expert-pack-test/expert.zip",
        selected_v5_bundle_sha256="b" * 64,
        acceptance_set_id="acceptance-test",
        worker_id="worker-1",
        worker_commit="c" * 40,
    )
    task, task_path = build_task_from_nas_pack_reference(
        row,
        queue_root=tmp_path / "cloud-queue",
        now=datetime(2026, 7, 16, 1, tzinfo=UTC),
    )
    assert task is not None and task_path is not None
    assert task.source_location == "nas_accepted"
    assert task.sections == {}
    assert compute_task_packet_sha256(task) == task.packet_sha256
    cloud_payload = task_path.read_text(encoding="utf-8")
    assert "factor-1" not in cloud_payload
    assert len(cloud_payload.encode("utf-8")) < 20_000

    hydrated = hydrate_nas_ai_research_task(
        task,
        pack_path=pack_path,
        work_root=tmp_path / "nas-work",
    )
    assert "core_state" in hydrated.sections
    assert "factor_research" in hydrated.sections
    assert hydrated.packet_sha256 == task.packet_sha256
    assert hydrated.task_id == task.task_id


def test_nas_ai_task_automatically_retries_transient_model_failure(tmp_path) -> None:
    queue = tmp_path / "cloud-queue"
    row = _accepted_pack_row()
    first_at = datetime(2026, 7, 29, 12, 48, tzinfo=UTC)
    first, _ = build_task_from_nas_pack_reference(row, queue_root=queue, now=first_at)
    assert first is not None
    _fail_task(
        queue,
        first.task_id,
        failed_at=datetime(2026, 7, 29, 12, 59, tzinfo=UTC),
        message="stage1 model call failed after retries: Responses API HTTP 503",
    )

    cooling, _ = build_task_from_nas_pack_reference(
        row,
        queue_root=queue,
        now=datetime(2026, 7, 29, 13, 10, tzinfo=UTC),
    )
    assert cooling is None

    retry, retry_path = build_task_from_nas_pack_reference(
        row,
        queue_root=queue,
        now=datetime(2026, 7, 29, 13, 30, tzinfo=UTC),
    )
    assert retry is not None and retry_path is not None
    assert retry.task_id != first.task_id
    assert "automatic_retry_of_transient_model_provider_failure" in retry.warnings
    state = json.loads((queue / "state" / "last_task.json").read_text(encoding="utf-8"))
    assert state["source_attempt"] == 2
    assert state["retry_of_task_id"] == first.task_id
    assert state["retry_failure_class"] == "TRANSIENT_MODEL_PROVIDER_FAILURE"

    duplicate, _ = build_task_from_nas_pack_reference(
        row,
        queue_root=queue,
        now=datetime(2026, 7, 29, 13, 31, tzinfo=UTC),
    )
    assert duplicate is None

    _fail_task(
        queue,
        retry.task_id,
        failed_at=datetime(2026, 7, 29, 13, 32, tzinfo=UTC),
        message="stage1 model call failed after retries: Responses API HTTP 502",
    )
    final_retry, _ = build_task_from_nas_pack_reference(
        row,
        queue_root=queue,
        now=datetime(2026, 7, 29, 14, 3, tzinfo=UTC),
    )
    assert final_retry is not None
    _fail_task(
        queue,
        final_retry.task_id,
        failed_at=datetime(2026, 7, 29, 14, 4, tzinfo=UTC),
        message="stage1 model call failed after retries: Responses API HTTP 503",
    )
    exhausted = ai_task_retry_status(
        queue,
        task_id=final_retry.task_id,
        source_pack_sha256=row.pack_sha256,
        now=datetime(2026, 7, 29, 15, tzinfo=UTC),
    )
    assert exhausted["retry_state"] == "EXHAUSTED"
    assert exhausted["attempt_count"] == 3
    no_fourth_attempt, _ = build_task_from_nas_pack_reference(
        row,
        queue_root=queue,
        now=datetime(2026, 7, 29, 15, tzinfo=UTC),
    )
    assert no_fourth_attempt is None


def test_nas_ai_task_does_not_retry_deterministic_worker_failure(tmp_path) -> None:
    queue = tmp_path / "cloud-queue"
    row = _accepted_pack_row()
    first_at = datetime(2026, 7, 29, 12, 48, tzinfo=UTC)
    first, _ = build_task_from_nas_pack_reference(row, queue_root=queue, now=first_at)
    assert first is not None
    _fail_task(
        queue,
        first.task_id,
        failed_at=datetime(2026, 7, 29, 12, 49, tzinfo=UTC),
        message="AI research task prompt version is not executable by this worker",
    )

    retry, _ = build_task_from_nas_pack_reference(
        row,
        queue_root=queue,
        now=datetime(2026, 7, 29, 18, tzinfo=UTC),
    )
    assert retry is None
    status = ai_task_retry_status(
        queue,
        task_id=first.task_id,
        source_pack_sha256=row.pack_sha256,
        now=datetime(2026, 7, 29, 18, tzinfo=UTC),
    )
    assert status["retry_state"] == "NON_RETRYABLE"
    assert status["retryable"] is False


def _accepted_pack_row() -> ExportPackIndexEntry:
    return ExportPackIndexEntry(
        pack_id="expert-pack-retry",
        pack_name="quant_lab_expert_pack_2026-07-29_retry.zip",
        export_date=date(2026, 7, 29),
        generated_at=datetime(2026, 7, 29, 12, 45, tzinfo=UTC),
        accepted_at=datetime(2026, 7, 29, 12, 45, tzinfo=UTC),
        pack_sha256="a" * 64,
        pack_size_bytes=1024,
        snapshot_id="export-snapshot-retry",
        authoritative_input_snapshot=True,
        nas_artifact_validated=True,
        control_plane_receipt_verified=True,
        download_ready=True,
        download_relative_path="2026/07/29/expert-pack-retry/expert.zip",
        selected_v5_bundle_sha256="b" * 64,
        acceptance_set_id="acceptance-retry",
        worker_id="worker-1",
        worker_commit="c" * 40,
    )


def _fail_task(
    queue: Path,
    task_id: str,
    *,
    failed_at: datetime,
    message: str,
) -> None:
    failed = queue / "failed" / task_id
    failed.parent.mkdir(parents=True, exist_ok=True)
    os.replace(queue / "pending" / task_id, failed)
    rejected = queue / "results" / "rejected" / task_id
    rejected.mkdir(parents=True, exist_ok=True)
    (rejected / "worker_error.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "worker_id": "quant-ai-nas-01",
                "failed_at": failed_at.isoformat(),
                "error_type": "RuntimeError",
                "message": message,
            }
        ),
        encoding="utf-8",
    )
