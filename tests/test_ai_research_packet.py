from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from datetime import UTC, datetime

from quant_lab.ai_research.contracts import compute_task_packet_sha256
from quant_lab.ai_research.packet import (
    build_ai_research_task,
    find_latest_expert_pack,
    queue_status,
)


def test_packet_uses_existing_expert_pack_and_deduplicates(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_2026-07-14.zip"
    full_csv = "factor_id,rank_ic_mean,long_short_mean_bps\n" + "\n".join(
        f"factor-{index},{index / 1000:.4f},{index / 10:.2f}" for index in range(500)
    )
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps({"fresh": True, "pack": "test"}))
        archive.writestr("data_quality.json", json.dumps({"status": "PASS"}))
        archive.writestr("reports/factor_evidence.csv", full_csv)
        archive.writestr(
            "reports/v5_trade_outcome_attribution.csv",
            "category,count\nentry_bad,2\nexit_bad,7\n",
        )
        archive.writestr("restricted/secret.txt", "must never be selected")

    queue = tmp_path / "queue"
    created_at = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)
    task, task_path = build_ai_research_task(
        pack,
        queue_root=queue,
        max_member_bytes=256,
        max_document_chars=10_000,
        max_total_chars=100_000,
        max_csv_rows=3,
        max_docs_per_section=4,
        now=created_at,
    )

    assert task is not None
    assert task_path is not None and task_path.is_file()
    assert compute_task_packet_sha256(task) == task.packet_sha256
    assert "core_state" in task.sections
    assert "factor_research" in task.sections
    assert "trade_learning" in task.sections
    assert all(
        "restricted" not in doc.source_member
        for docs in task.sections.values()
        for doc in docs
    )

    factor_doc = next(
        doc
        for doc in task.sections["factor_research"]
        if doc.source_member.endswith("factor_evidence.csv")
    )
    assert factor_doc.truncated is True
    assert factor_doc.content_sha256 == hashlib.sha256(
        full_csv.encode("utf-8")[:256]
    ).hexdigest()
    assert factor_doc.source_size_bytes == len(full_csv.encode("utf-8"))
    assert len(factor_doc.content["rows"]) == 3

    duplicate, duplicate_path = build_ai_research_task(
        pack,
        queue_root=queue,
        max_member_bytes=256,
        max_document_chars=10_000,
        max_total_chars=100_000,
        max_csv_rows=3,
        max_docs_per_section=4,
        now=created_at,
    )
    assert duplicate is None
    assert duplicate_path is None
    assert queue_status(queue)["counts"]["pending"] == 1
    assert (task_path.parent / "task_manifest.json").is_file()
    assert not list((queue / ".staging").iterdir())
    if os.name != "nt":
        assert stat.S_IMODE(task_path.parent.stat().st_mode) == 0o2770
        assert stat.S_IMODE(task_path.stat().st_mode) == 0o660
        assert stat.S_IMODE((task_path.parent / "task_manifest.json").stat().st_mode) == 0o660


def test_latest_expert_pack_skips_newer_partial_zip(tmp_path) -> None:
    valid = tmp_path / "quant_lab_expert_pack_2026-07-14_valid.zip"
    with zipfile.ZipFile(valid, "w") as archive:
        archive.writestr("manifest.json", "{}")
    partial = tmp_path / "quant_lab_expert_pack_2026-07-14_partial.zip"
    partial.write_bytes(b"PK\x03\x04unfinished")
    valid.touch()
    partial.touch()

    assert find_latest_expert_pack(tmp_path) == valid
