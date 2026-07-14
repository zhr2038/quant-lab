from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from datetime import UTC, datetime

from quant_lab.ai_research.contracts import canonical_json, compute_task_packet_sha256
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
    assert task.preflight is not None
    assert task.preflight.status == "BLOCK"
    assert task.preflight.missing_core_members == ["provenance.json"]
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


def test_default_packet_stays_within_model_input_budget(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_2026-07-14_large.zip"
    large_csv = "factor_id,rank_ic_mean\n" + "\n".join(
        f"factor-{index},{index / 1000:.4f}" for index in range(20_000)
    )
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps({"fresh": True}))
        for index in range(8):
            archive.writestr(f"reports/factor_evidence_{index}.csv", large_csv)

    task, _ = build_ai_research_task(pack, queue_root=tmp_path / "queue")

    assert task is not None
    assert len(canonical_json(task.model_dump(mode="json")).encode("utf-8")) < 400_000


def test_packet_preflight_passes_with_complete_core_identity(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_complete.zip"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("provenance.json", "{}")
        archive.writestr("data_quality.json", "{}")

    task, _ = build_ai_research_task(pack, queue_root=tmp_path / "queue")

    assert task is not None and task.preflight is not None
    assert task.preflight.status == "PASS"
    assert task.preflight.blockers == []


def test_packet_preflight_warns_but_does_not_block_truncated_core(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_large_core.zip"
    large = json.dumps({"rows": [{"value": "x" * (2 * 1024 * 1024 + 1_024)}]})
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", large)
        archive.writestr("provenance.json", "{}")
        archive.writestr("data_quality.json", "{}")

    task, _ = build_ai_research_task(
        pack,
        queue_root=tmp_path / "queue",
        max_document_chars=500,
    )

    assert task is not None and task.preflight is not None
    assert task.preflight.status == "WARN"
    assert task.preflight.blockers == []
    assert any(item.startswith("truncated_core_member:") for item in task.preflight.warnings)


def test_large_core_json_uses_complete_deterministic_summary(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_large_core_summary.zip"
    manifest = {
        "generated_at": "2026-07-14T00:00:00Z",
        "files": [{"path": f"reports/{index}.csv"} for index in range(2_000)],
        "row_counts": {f"factor_{index}": index for index in range(200)},
        "dataset_freshness": {
            f"factor_{index}": {"status": "OK", "latest_ts": "2026-07-14T00:00:00Z"}
            for index in range(200)
        },
    }
    data_quality = {
        "status": "WARN",
        "checks": [
            {"dataset": f"factor_{index}", "status": "PASS", "detail": "x" * 500}
            for index in range(500)
        ],
        "warnings": ["bounded warning"],
        "dataset_governance": {
            "status": "WARN",
            "checks": [
                {"dataset": f"factor_{index}", "status": "PASS", "detail": "y" * 500}
                for index in range(500)
            ],
        },
    }
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("provenance.json", "{}")
        archive.writestr("data_quality.json", json.dumps(data_quality))

    task, _ = build_ai_research_task(
        pack,
        queue_root=tmp_path / "queue",
        max_member_bytes=256,
        max_document_chars=40_000,
    )

    assert task is not None and task.preflight is not None
    core = {item.source_member: item for item in task.sections["core_state"]}
    assert task.preflight.status == "PASS"
    assert core["manifest.json"].representation == "deterministic_summary"
    assert core["data_quality.json"].representation == "deterministic_summary"
    assert core["manifest.json"].truncated is False
    assert core["data_quality.json"].truncated is False
    assert core["manifest.json"].content["_representation"]["file_count"] == 2_000
    assert core["data_quality.json"].content["checks"]["count"] == 500
