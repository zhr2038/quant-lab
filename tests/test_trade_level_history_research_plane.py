from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from quant_lab.research.candidate_labels import (
    LABEL_SCHEMA,
)
from quant_lab.research.candidate_labels import (
    SOURCE_NAME as CANDIDATE_LABEL_SOURCE,
)
from quant_lab.research_plane.importer import (
    import_entry_quality_history_result,
)
from quant_lab.research_plane.queue import (
    create_trade_level_history_task,
)
from quant_lab.research_plane.snapshot_gc import (
    release_snapshot_payload,
)
from quant_lab.research_plane.status import research_plane_status
from quant_lab.research_plane.trade_level_history_contracts import (
    TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS,
    TradeLevelHistorySnapshotFile,
    TradeLevelHistorySnapshotManifest,
    TradeLevelHistoryTask,
)
from quant_lab.research_plane.trade_level_history_publish import (
    TRADE_LEVEL_HISTORY_DATASETS,
    publish_trade_level_history_generation,
    verify_trade_level_history_generation_fast,
)
from quant_lab.research_plane.trade_level_history_result import (
    validate_trade_level_history_result_bundle,
)
from quant_lab.research_worker.runner import (
    Config as ResearchWorkerConfig,
)
from quant_lab.research_worker.runner import _validate_config
from quant_lab.research_worker.trade_level_history import (
    compute_trade_level_history_result,
)
from quant_lab.research_worker.trade_level_history_result_writer import (
    write_trade_level_history_result_bundle,
)
from quant_lab.trade_level.judgment import (
    build_and_publish_trade_level_control,
    build_trade_opportunity_events,
)
from quant_lab.trade_level.shadow import (
    build_and_publish_trade_level_legacy_control_shadow,
)

COMMIT = "a" * 40
SHA = "b" * 64
CANDIDATE_GENERATION_ID = "candidate-generation-test"
CANDIDATE_GENERATION_DIGEST = "c" * 64
CANDIDATE_FINGERPRINT = "d" * 64
CANDIDATE_LABEL_HASH = "e" * 64


def test_trade_level_history_worker_result_and_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root, snapshot, task = _signed_snapshot_fixture(tmp_path)
    compute = compute_trade_level_history_result(
        snapshot_root,
        snapshot,
        task,
        min_free_disk_bytes=0,
        work_dir=tmp_path / "work",
    )
    assert tuple(
        item["check_name"] for item in compute.anti_leakage["checks"]
    ) == TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS
    assert compute.anti_leakage["status"] == "PASS"

    worker_key = Ed25519PrivateKey.generate()
    bundle, manifest, receipt = write_trade_level_history_result_bundle(
        tmp_path / "results",
        task=task,
        snapshot=snapshot,
        snapshot_root=snapshot_root,
        compute=compute,
        worker_id="nas-worker-test",
        worker_commit=COMMIT,
        worker_key_id="worker-key-test",
        worker_signing_key=worker_key,
        claimed_at=task.requested_at,
        input_bytes=snapshot.total_input_bytes,
        cache_hit_bytes=snapshot.total_input_bytes,
        downloaded_bytes=0,
        peak_rss_bytes=0,
        compute_duration_seconds=1.0,
    )
    handoff_marker = bundle / ".HANDOFF_READY"
    handoff_marker.touch()
    validated = validate_trade_level_history_result_bundle(
        bundle,
        manifest=manifest,
        receipt=receipt,
        task=task,
        snapshot=snapshot,
        snapshot_root=snapshot_root,
        worker_public_key=worker_key.public_key(),
        expected_worker_key_id="worker-key-test",
    )
    handoff_marker.write_text("not-empty", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="trade_level_history_result_handoff_marker_invalid",
    ):
        validate_trade_level_history_result_bundle(
            bundle,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            snapshot_root=snapshot_root,
            worker_public_key=worker_key.public_key(),
            expected_worker_key_id="worker-key-test",
        )
    handoff_marker.unlink()
    handoff_marker.mkdir()
    with pytest.raises(
        ValueError,
        match="trade_level_history_result_handoff_marker_invalid",
    ):
        validate_trade_level_history_result_bundle(
            bundle,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            snapshot_root=snapshot_root,
            worker_public_key=worker_key.public_key(),
            expected_worker_key_id="worker-key-test",
        )
    handoff_marker.rmdir()
    handoff_marker.write_bytes(b"")
    assert {item.dataset_name for item in manifest.outputs} == {
        "trade_opportunity_label",
        "trade_level_similarity_outcome",
    }
    similarity = pl.read_parquet(validated.similarity_paths)
    assert (
        similarity.sort("decision_ts")
        .get_column("similar_sample_count")
        .to_list()
        == [0, 1, 2, 3]
    )

    lake = tmp_path / "lake"
    (lake / "gold").mkdir(parents=True)
    _write_candidate_pointer(lake)
    monkeypatch.setattr(
        "quant_lab.research_plane.trade_level_history_publish."
        "verify_v5_candidate_evidence_generation_fast",
        lambda *_args, **_kwargs: {"v5_candidate_label": 12},
    )
    published = publish_trade_level_history_generation(
        lake,
        validated,
        snapshot_root=snapshot_root,
    )
    assert published["published"] is True
    assert published["row_counts"] == {
        "trade_opportunity_event": 4,
        "trade_opportunity_label": 4,
        "trade_level_similarity_outcome": 4,
    }
    for relative in TRADE_LEVEL_HISTORY_DATASETS.values():
        assert (lake / relative / "data.parquet").is_file()
    assert verify_trade_level_history_generation_fast(
        lake,
        manifest.generation_id,
        expected_input_fingerprint=task.input_fingerprint_digest,
        expected_candidate_generation_id=CANDIDATE_GENERATION_ID,
        expected_candidate_generation_digest=CANDIDATE_GENERATION_DIGEST,
    ) == published["row_counts"]

    repeated = publish_trade_level_history_generation(
        lake,
        validated,
        snapshot_root=snapshot_root,
    )
    assert repeated["published"] is False
    assert repeated["idempotent"] is True

    label_path = lake / TRADE_LEVEL_HISTORY_DATASETS[
        "trade_opportunity_label"
    ] / "data.parquet"
    label_path.write_bytes(label_path.read_bytes() + b"corrupt")
    with pytest.raises(
        RuntimeError,
        match="dataset_hash_mismatch|dataset_integrity_failed",
    ):
        verify_trade_level_history_generation_fast(
            lake,
            manifest.generation_id,
        )


def test_trade_level_history_contract_rejects_extra_field(
    tmp_path: Path,
) -> None:
    _root, _snapshot, task = _signed_snapshot_fixture(tmp_path)
    payload = task.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValueError):
        TradeLevelHistoryTask.model_validate(payload)


@pytest.mark.parametrize(
    "mismatched_field",
    (
        "worker_commit",
        "build_git_commit",
        "runtime_worker_commit",
        "repository_commit",
    ),
)
def test_worker_provenance_mismatch_is_fail_closed(
    tmp_path: Path,
    mismatched_field: str,
) -> None:
    secret_paths = [
        tmp_path / name
        for name in (
            "ssh",
            "known-hosts",
            "task-public",
            "worker-private",
        )
    ]
    for path in secret_paths:
        path.write_text("test", encoding="utf-8")
    provenance = {
        "worker_commit": COMMIT,
        "build_git_commit": COMMIT,
        "runtime_worker_commit": COMMIT,
        "repository_commit": COMMIT,
    }
    provenance[mismatched_field] = "f" * 40
    config = ResearchWorkerConfig(
        cloud_host="qyun2.example",
        cloud_user="quant-research",
        cloud_port=22,
        ssh_key_path=secret_paths[0],
        known_hosts_path=secret_paths[1],
        cloud_queue_root="/queue",
        data_root=tmp_path / "data",
        task_public_key_path=secret_paths[2],
        task_key_id="task-key",
        worker_signing_key_path=secret_paths[3],
        worker_key_id="worker-key",
        worker_id="worker-1",
        worker_commit=provenance["worker_commit"],
        run_once=True,
        poll_seconds=30,
        heartbeat_seconds=30,
        min_free_disk_bytes=0,
        max_snapshot_bytes=1,
        max_result_bytes=1,
        heavy_job_lock=tmp_path / "heavy.lock",
        batch_fetch_workers=1,
        build_git_commit=provenance["build_git_commit"],
        runtime_worker_commit=provenance["runtime_worker_commit"],
        repository_commit=provenance["repository_commit"],
    )
    with pytest.raises(
        ValueError,
        match="research_worker_provenance_mismatch",
    ):
        _validate_config(config)
    _validate_config(
        config.__class__(
            **{
                **config.__dict__,
                "worker_commit": COMMIT,
                "build_git_commit": COMMIT,
                "runtime_worker_commit": COMMIT,
                "repository_commit": COMMIT,
            }
        )
    )


def test_trade_level_history_queue_import_and_no_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candidate_rows = [
        {
            "decision_ts": base + timedelta(hours=hour),
            "run_id": f"run-{index}",
            "candidate_id": f"candidate-{index}",
            "symbol": "BTC-USDT",
            "strategy_candidate": "trend-long",
            "risk_level": "LOW",
            "regime": "TREND",
            "alpha6_score": 0.9,
            "rank": 1,
            "cost_bps": 10.0,
        }
        for index, hour in enumerate((0, 5, 9, 25))
    ]
    candidate_labels = _candidate_labels(candidate_rows)
    _write_frame(
        lake / "silver" / "v5_candidate_event",
        pl.DataFrame(candidate_rows),
    )
    _write_frame(
        lake / "gold" / "v5_candidate_label",
        candidate_labels,
    )
    _write_frame(
        lake / "gold" / "risk_permission",
        pl.DataFrame(
            [
                {
                    "as_of_ts": base - timedelta(hours=1),
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "live_block_reasons": "[]",
                    "allowed_live_modes": "[]",
                }
            ]
        ),
    )
    _write_frame(
        lake / "silver" / "v5_trade_event",
        pl.DataFrame(
            schema={"run_id": pl.Utf8, "symbol": pl.Utf8}
        ),
    )
    _write_frame(
        lake / "silver" / "v5_order_lifecycle",
        pl.DataFrame(
            schema={"run_id": pl.Utf8, "symbol": pl.Utf8}
        ),
    )
    _write_candidate_pointer(
        lake,
        candidate_label_rows=candidate_labels.height,
        managed_columns=list(LABEL_SCHEMA),
    )
    monkeypatch.setattr(
        "quant_lab.research_plane.trade_level_history_snapshot."
        "verify_v5_candidate_evidence_generation_fast",
        lambda *_args, **_kwargs: {
            "v5_candidate_label": candidate_labels.height
        },
    )
    monkeypatch.setattr(
        "quant_lab.research_plane.trade_level_history_publish."
        "verify_v5_candidate_evidence_generation_fast",
        lambda *_args, **_kwargs: {
            "v5_candidate_label": candidate_labels.height
        },
    )

    request = create_trade_level_history_task(
        lake,
        queue,
        as_of_date=date(2026, 1, 2),
        signing_key=task_key,
        signature_key_id="task-key-test",
        quant_lab_commit=COMMIT,
    )
    assert request.state == "task_created"
    assert request.snapshot_materialized is True
    assert request.task is not None
    task = request.task
    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = TradeLevelHistorySnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    os.replace(
        queue / "pending" / task.task_id,
        queue / "running" / task.task_id,
    )
    compute = compute_trade_level_history_result(
        snapshot_root,
        snapshot,
        task,
        min_free_disk_bytes=0,
        work_dir=tmp_path / "compute",
    )
    write_trade_level_history_result_bundle(
        queue / "results" / "inbox",
        task=task,
        snapshot=snapshot,
        snapshot_root=snapshot_root,
        compute=compute,
        worker_id="nas-worker-test",
        worker_commit=COMMIT,
        worker_key_id="worker-key-test",
        worker_signing_key=worker_key,
        claimed_at=task.requested_at,
        input_bytes=snapshot.total_input_bytes,
        cache_hit_bytes=0,
        downloaded_bytes=snapshot.total_input_bytes,
        peak_rss_bytes=1,
        compute_duration_seconds=1.0,
    )
    imported = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        task_public_key=task_key.public_key(),
        worker_public_key=worker_key.public_key(),
        expected_task_key_id="task-key-test",
        expected_worker_key_id="worker-key-test",
        expected_quant_lab_commit=COMMIT,
    )
    assert imported.state == "completed"
    assert imported.idempotent is False
    assert (
        queue / "results" / "imported" / task.task_id
    ).is_dir()
    assert (queue / "completed" / task.task_id).is_dir()
    assert (snapshot_root / "FILES_RELEASED.json").is_file()
    assert not (snapshot_root / "files").exists()
    plane_status = research_plane_status(queue)
    assert (
        plane_status["tasks"]["trade_level_history"]["state"]
        == "completed"
    )
    assert (
        plane_status["tasks"]["trade_level_history"][
            "trade_level_history_generation_id"
        ]
        == imported.generation_id
    )

    repeated = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        task_public_key=task_key.public_key(),
        worker_public_key=worker_key.public_key(),
        expected_task_key_id="task-key-test",
        expected_worker_key_id="worker-key-test",
        expected_quant_lab_commit=COMMIT,
    )
    assert repeated.idempotent is True

    no_change = create_trade_level_history_task(
        lake,
        queue,
        as_of_date=date(2026, 1, 2),
        signing_key=task_key,
        signature_key_id="task-key-test",
        quant_lab_commit=COMMIT,
    )
    assert no_change.state == "already_current"
    assert no_change.snapshot_materialized is False
    assert no_change.task_created is False

    shadow = build_and_publish_trade_level_legacy_control_shadow(
        lake,
        tmp_path / "shadow-reports",
        as_of_date=date(2026, 1, 2),
    )
    assert shadow.status == "PASS"
    assert shadow.similarity_mismatch_count == 0
    assert shadow.published_new_micro_canary_allow_count == 0
    assert shadow.risk_permission_unchanged is True
    assert Path(shadow.report_path).is_file()
    shadow_report = json.loads(
        Path(shadow.report_path).read_text("utf-8")
    )
    assert shadow_report[
        "judgment_order_limit_increase_event_ids"
    ] == []
    assert shadow_report[
        "policy_order_limit_increase_bucket_keys"
    ] == []

    current_control = build_and_publish_trade_level_control(
        lake,
        as_of_date=date(2026, 1, 2),
    )
    assert current_control.history_status == "CURRENT"

    history_pointer_path = (
        lake / "gold" / "trade_level_history_generation.json"
    )
    history_pointer = json.loads(
        history_pointer_path.read_text("utf-8")
    )
    history_pointer["published_at"] = (
        base - timedelta(days=10)
    ).isoformat()
    history_pointer_path.write_text(
        json.dumps(history_pointer),
        encoding="utf-8",
    )
    stale_control = build_and_publish_trade_level_control(
        lake,
        as_of_date=date(2026, 1, 2),
        max_generation_age_hours=1,
    )
    assert stale_control.history_status == "STALE"
    assert stale_control.micro_canary_allow_count == 0
    assert stale_control.max_live_notional_usdt == 0.0

    similarity_path = (
        lake
        / "gold"
        / "trade_level_similarity_outcome"
        / "data.parquet"
    )
    similarity_path.write_bytes(
        similarity_path.read_bytes() + b"corrupt"
    )
    invalid_control = build_and_publish_trade_level_control(
        lake,
        as_of_date=date(2026, 1, 2),
    )
    assert invalid_control.history_status == "INVALID"
    assert invalid_control.trade_level_judgment_rows == 0
    assert invalid_control.micro_canary_allow_count == 0
    assert invalid_control.max_live_notional_usdt == 0.0
    integrity_failed = create_trade_level_history_task(
        lake,
        queue,
        as_of_date=date(2026, 1, 2),
        signing_key=task_key,
        signature_key_id="task-key-test",
        quant_lab_commit=COMMIT,
    )
    assert integrity_failed.state == "generation_integrity_failed"
    assert integrity_failed.snapshot_materialized is False
    assert integrity_failed.task_created is False


def test_trade_level_history_queue_coalesces_supersedes_and_rehydrates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candidate_rows, candidate_labels = _prepare_queue_lake(
        lake,
        base,
    )
    monkeypatch.setattr(
        "quant_lab.research_plane.trade_level_history_snapshot."
        "verify_v5_candidate_evidence_generation_fast",
        lambda *_args, **_kwargs: {
            "v5_candidate_label": candidate_labels.height
        },
    )
    signing_key = Ed25519PrivateKey.generate()
    first = create_trade_level_history_task(
        lake,
        queue,
        as_of_date=date(2026, 1, 2),
        signing_key=signing_key,
        signature_key_id="task-key-test",
        quant_lab_commit=COMMIT,
    )
    assert first.task is not None
    repeated = create_trade_level_history_task(
        lake,
        queue,
        as_of_date=date(2026, 1, 2),
        signing_key=signing_key,
        signature_key_id="task-key-test",
        quant_lab_commit=COMMIT,
    )
    assert repeated.state == "coalesced"
    assert repeated.task_created is False
    os.replace(
        queue / "pending" / first.task.task_id,
        queue / "running" / first.task.task_id,
    )

    candidate_rows.append(
        {
            **candidate_rows[-1],
            "decision_ts": base + timedelta(hours=30),
            "run_id": "run-successor-1",
            "candidate_id": "candidate-successor-1",
        }
    )
    _write_frame(
        lake / "silver" / "v5_candidate_event",
        pl.DataFrame(candidate_rows),
    )
    successor = create_trade_level_history_task(
        lake,
        queue,
        as_of_date=date(2026, 1, 2),
        signing_key=signing_key,
        signature_key_id="task-key-test",
        quant_lab_commit=COMMIT,
    )
    assert successor.state == "task_created"
    assert successor.task is not None
    assert (queue / "running" / first.task.task_id).is_dir()

    candidate_rows.append(
        {
            **candidate_rows[-1],
            "decision_ts": base + timedelta(hours=31),
            "run_id": "run-successor-2",
            "candidate_id": "candidate-successor-2",
        }
    )
    _write_frame(
        lake / "silver" / "v5_candidate_event",
        pl.DataFrame(candidate_rows),
    )
    latest = create_trade_level_history_task(
        lake,
        queue,
        as_of_date=date(2026, 1, 2),
        signing_key=signing_key,
        signature_key_id="task-key-test",
        quant_lab_commit=COMMIT,
    )
    assert latest.state == "task_created"
    assert latest.task is not None
    assert (
        queue / "cancelled" / successor.task.task_id
    ).is_dir()
    assert (queue / "running" / first.task.task_id).is_dir()
    pending_tasks = [
        path
        for path in (queue / "pending").iterdir()
        if path.is_dir()
    ]
    assert [path.name for path in pending_tasks] == [
        latest.task.task_id
    ]

    os.replace(
        queue / "pending" / latest.task.task_id,
        queue / "cancelled" / latest.task.task_id,
    )
    assert release_snapshot_payload(
        queue,
        latest.snapshot_id,
        reason="test_rehydrate",
    )
    (lake / "gold" / "trade_level_history_generation.json").write_text(
        json.dumps(
            {
                "generation_id": "previous-generation-test",
                "generation_digest": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    rehydrated = create_trade_level_history_task(
        lake,
        queue,
        as_of_date=date(2026, 1, 2),
        signing_key=signing_key,
        signature_key_id="task-key-test",
        quant_lab_commit=COMMIT,
    )
    assert rehydrated.state == "task_created"
    assert rehydrated.snapshot_id == latest.snapshot_id
    assert rehydrated.snapshot_rehydrated is True
    assert rehydrated.snapshot_materialized is True
    assert (
        queue
        / "snapshots"
        / rehydrated.snapshot_id
        / "files"
    ).is_dir()


def _signed_snapshot_fixture(
    tmp_path: Path,
) -> tuple[Path, TradeLevelHistorySnapshotManifest, TradeLevelHistoryTask]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    decision_hours = (0, 5, 9, 25)
    candidate_rows = [
        {
            "decision_ts": base + timedelta(hours=hour),
            "run_id": f"run-{index}",
            "candidate_id": f"candidate-{index}",
            "symbol": "BTC-USDT",
            "strategy_candidate": "trend-long",
            "risk_level": "LOW",
            "regime": "TREND",
            "alpha6_score": 0.9,
            "rank": 1,
            "cost_bps": 10.0,
        }
        for index, hour in enumerate(decision_hours)
    ]
    risk_permissions = pl.DataFrame(
        [
            {
                "as_of_ts": base - timedelta(hours=1),
                "permission": "ABORT",
                "permission_status": "ACTIVE_ABORT",
                "live_block_reasons": "[]",
                "allowed_live_modes": "[]",
            }
        ]
    )
    events = build_trade_opportunity_events(
        pl.DataFrame(candidate_rows),
        risk_permissions=risk_permissions,
        created_at=base,
    )
    candidate_labels = _candidate_labels(candidate_rows)

    snapshot_root = tmp_path / "snapshot"
    event_path = (
        snapshot_root
        / "files"
        / "cloud"
        / "trade_opportunity_event"
        / "data.parquet"
    )
    candidate_path = (
        snapshot_root
        / "files"
        / "gold"
        / "v5_candidate_label"
        / "data.parquet"
    )
    event_path.parent.mkdir(parents=True)
    candidate_path.parent.mkdir(parents=True)
    events.write_parquet(event_path)
    candidate_labels.write_parquet(candidate_path)
    files = (
        TradeLevelHistorySnapshotFile(
            dataset_name="cloud/trade_opportunity_event",
            relative_path="cloud/trade_opportunity_event/data.parquet",
            sha256=SHA,
            size_bytes=event_path.stat().st_size,
            row_count=events.height,
            min_ts=events["decision_ts"].min(),
            max_ts=events["decision_ts"].max(),
            schema_fingerprint=SHA,
            uncompressed_bytes=0,
        ),
        TradeLevelHistorySnapshotFile(
            dataset_name="gold/v5_candidate_label",
            relative_path="gold/v5_candidate_label/data.parquet",
            sha256=SHA,
            size_bytes=candidate_path.stat().st_size,
            row_count=candidate_labels.height,
            min_ts=candidate_labels["decision_ts"].min(),
            max_ts=candidate_labels["decision_ts"].max(),
            schema_fingerprint=SHA,
            uncompressed_bytes=0,
        ),
    )
    snapshot = TradeLevelHistorySnapshotManifest(
        as_of_date=date(2026, 1, 2),
        candidate_evidence_generation_id=CANDIDATE_GENERATION_ID,
        candidate_evidence_generation_digest=CANDIDATE_GENERATION_DIGEST,
        candidate_evidence_input_fingerprint=CANDIDATE_FINGERPRINT,
        snapshot_id="trade-level-history-snapshot-test",
        generated_at=base,
        quant_lab_commit=COMMIT,
        input_fingerprint_digest=SHA,
        candidate_event_digest=SHA,
        risk_permission_digest=SHA,
        v5_trade_event_digest=SHA,
        order_lifecycle_digest=SHA,
        derived_trade_opportunity_event_digest=SHA,
        candidate_label_dataset_hash=CANDIDATE_LABEL_HASH,
        candidate_label_row_count=candidate_labels.height,
        candidate_label_schema="v5.candidate_label.v1",
        event_row_count=events.height,
        event_min_ts=events["decision_ts"].min(),
        event_max_ts=events["decision_ts"].max(),
        candidate_label_min_ts=candidate_labels["decision_ts"].min(),
        candidate_label_max_ts=candidate_labels["decision_ts"].max(),
        datasets=(
            "cloud/trade_opportunity_event",
            "gold/v5_candidate_label",
        ),
        files=files,
        total_input_bytes=sum(item.size_bytes for item in files),
        total_input_rows=sum(item.row_count for item in files),
        estimated_uncompressed_bytes=0,
        manifest_sha256=SHA,
        signature_key_id="task-key-test",
        signature="snapshot-signature",
    )
    task = TradeLevelHistoryTask(
        as_of_date=snapshot.as_of_date,
        candidate_evidence_generation_id=CANDIDATE_GENERATION_ID,
        candidate_evidence_generation_digest=CANDIDATE_GENERATION_DIGEST,
        candidate_evidence_input_fingerprint=CANDIDATE_FINGERPRINT,
        task_id="trade-level-history-task-test",
        snapshot_id=snapshot.snapshot_id,
        input_fingerprint_digest=snapshot.input_fingerprint_digest,
        quant_lab_commit=COMMIT,
        snapshot_manifest_sha256=snapshot.manifest_sha256,
        requested_at=base + timedelta(days=2),
        signature_key_id="task-key-test",
        signature="task-signature",
    )
    return snapshot_root, snapshot, task


def _candidate_labels(
    candidate_rows: list[dict[str, object]],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidate_rows):
        decision_ts = candidate["decision_ts"]
        assert isinstance(decision_ts, datetime)
        for horizon in (4, 8, 24):
            row = {name: None for name in LABEL_SCHEMA}
            row.update(
                {
                    "strategy": "v5",
                    "candidate_label_schema_version": (
                        "v5.candidate_label.v1"
                    ),
                    "candidate_id": candidate["candidate_id"],
                    "run_id": candidate["run_id"],
                    "ts_utc": decision_ts,
                    "symbol": candidate["symbol"],
                    "strategy_candidate": candidate[
                        "strategy_candidate"
                    ],
                    "block_reason": "",
                    "final_decision": "PAPER",
                    "horizon_hours": horizon,
                    "decision_ts": decision_ts,
                    "label_ts": decision_ts + timedelta(hours=horizon),
                    "entry_close": 100.0,
                    "label_close": 101.0,
                    "gross_bps": 100.0,
                    "net_bps_after_cost": float(
                        10 + index + horizon
                    ),
                    "mfe_bps": 120.0,
                    "mae_bps": float(-10 - index - horizon),
                    "win": True,
                    "label_status": "complete",
                    "label_reason": "ok",
                    "cost_bps": 10.0,
                    "cost_source": "signed",
                    "alpha6_side": "buy",
                    "regime_state": "TREND",
                    "risk_level": "LOW",
                    "btc_trend_state": "UP",
                    "broad_market_positive_count": 3,
                    "funding_state": "NORMAL",
                    "volatility_bucket": "MEDIUM",
                    "protect_level": "NONE",
                    "final_score": 0.9,
                    "expected_edge_bps": 50.0,
                    "required_edge_bps": 10.0,
                    "source_event_bundle_sha256": SHA,
                    "source_path_inside_bundle": "run.json",
                    "created_at": decision_ts,
                    "source": CANDIDATE_LABEL_SOURCE,
                }
            )
            rows.append(row)
    return pl.DataFrame(rows, schema=LABEL_SCHEMA, orient="row")


def _write_candidate_pointer(
    lake: Path,
    *,
    candidate_label_rows: int | None = None,
    managed_columns: list[str] | None = None,
) -> None:
    payload = {
        "generation_id": CANDIDATE_GENERATION_ID,
        "generation_digest": CANDIDATE_GENERATION_DIGEST,
        "input_fingerprint_digest": CANDIDATE_FINGERPRINT,
        "dataset_hashes": {
            "v5_candidate_label": CANDIDATE_LABEL_HASH,
        },
        "row_counts": {
            "v5_candidate_label": (
                candidate_label_rows
                if candidate_label_rows is not None
                else 12
            ),
        },
        "managed_columns": {
            "v5_candidate_label": (
                managed_columns
                if managed_columns is not None
                else list(LABEL_SCHEMA)
            ),
        },
    }
    (lake / "gold" / "v5_candidate_evidence_generation.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pl.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path / "data.parquet")


def _prepare_queue_lake(
    lake: Path,
    base: datetime,
) -> tuple[list[dict[str, object]], pl.DataFrame]:
    candidate_rows: list[dict[str, object]] = [
        {
            "decision_ts": base + timedelta(hours=hour),
            "run_id": f"run-{index}",
            "candidate_id": f"candidate-{index}",
            "symbol": "BTC-USDT",
            "strategy_candidate": "trend-long",
            "risk_level": "LOW",
            "regime": "TREND",
            "alpha6_score": 0.9,
            "rank": 1,
            "cost_bps": 10.0,
        }
        for index, hour in enumerate((0, 5, 9, 25))
    ]
    candidate_labels = _candidate_labels(candidate_rows)
    _write_frame(
        lake / "silver" / "v5_candidate_event",
        pl.DataFrame(candidate_rows),
    )
    _write_frame(
        lake / "gold" / "v5_candidate_label",
        candidate_labels,
    )
    _write_frame(
        lake / "gold" / "risk_permission",
        pl.DataFrame(
            [
                {
                    "as_of_ts": base - timedelta(hours=1),
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "live_block_reasons": "[]",
                    "allowed_live_modes": "[]",
                }
            ]
        ),
    )
    for relative in (
        Path("silver") / "v5_trade_event",
        Path("silver") / "v5_order_lifecycle",
    ):
        _write_frame(
            lake / relative,
            pl.DataFrame(
                schema={"run_id": pl.Utf8, "symbol": pl.Utf8}
            ),
        )
    _write_candidate_pointer(
        lake,
        candidate_label_rows=candidate_labels.height,
        managed_columns=list(LABEL_SCHEMA),
    )
    return candidate_rows, candidate_labels
