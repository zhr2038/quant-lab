from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, date, datetime

from quant_lab.ai_research.contracts import compute_task_packet_sha256
from quant_lab.ai_research.packet import (
    build_task_from_nas_pack_reference,
    hydrate_nas_ai_research_task,
)
from quant_lab.export_plane.contracts import ExportPackIndexEntry


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
