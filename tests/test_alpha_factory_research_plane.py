from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

import quant_lab.research_plane.importer as importer_module
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.alpha_factory.factory import (
    ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS,
    ALPHA_FACTORY_PROMOTION_QUEUE_DATASET,
    ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET,
    alpha_factory_template_registry_digest,
    build_default_template_registry,
    merge_alpha_factory_managed_evidence,
    prepare_alpha_factory_control_state,
)
from quant_lab.research.second_stage_alpha_factory import EXPANDED_QUALITY_DATASET
from quant_lab.research.strategy_evidence import SAMPLE_SCHEMA, SUMMARY_SCHEMA
from quant_lab.research_plane.alpha_factory_publish import (
    ALPHA_FACTORY_GENERATION_POINTER,
    verify_alpha_factory_generation,
)
from quant_lab.research_plane.atomic_publish import (
    AtomicPublishItem,
    commit_atomic_research_generation,
)
from quant_lab.research_plane.contracts import (
    RESEARCH_TASK_ADAPTER,
    AlphaFactoryTask,
    ResearchTask,
)
from quant_lab.research_plane.importer import import_entry_quality_history_result
from quant_lab.research_plane.queue import create_alpha_factory_task
from quant_lab.research_plane.result import (
    _validate_alpha_frame_safety,
    _validate_alpha_frame_scope,
    _validate_factor_bridge_report,
    _validate_lazy_unique_keys,
)
from quant_lab.research_plane.snapshot import (
    seal_alpha_factory_snapshot,
    verify_alpha_factory_snapshot_manifest,
)
from quant_lab.research_plane.snapshot_gc import release_snapshot_payload
from quant_lab.research_plane.status import research_plane_status
from quant_lab.research_worker.alpha_factory import (
    ALPHA_FACTORY_ANTI_LEAKAGE_CHECKS,
    compute_alpha_factory_from_snapshot,
)
from quant_lab.research_worker.result_writer import write_alpha_factory_result_bundle

COMMIT = "c" * 40
TASK_KEY_ID = "cloud-research-v1"
WORKER_KEY_ID = "nas-research-v1"
BUNDLE_ID = "v5-bundle-sha256:" + "d" * 64


def test_research_task_discriminated_union_keeps_entry_v1_strict() -> None:
    payload = {
        "schema_version": "quant_lab_research_task.v1",
        "task_type": "entry_quality_history",
        "task_id": "entry-task",
        "snapshot_id": "entry-snapshot",
        "start_date": "2026-07-01",
        "end_date": "2026-07-18",
        "mode": "recent_30d",
        "cost_mode": "conservative",
        "window_hours": 24,
        "quant_lab_commit": COMMIT,
        "entry_quality_schema_version": "entry_quality.v0.1",
        "selected_v5_bundle_id": BUNDLE_ID,
        "snapshot_manifest_sha256": "e" * 64,
        "requested_at": "2026-07-18T00:00:00Z",
        "lease_seconds": 3600,
        "max_attempts": 3,
        "signature_key_id": TASK_KEY_ID,
        "research_only": True,
        "live_order_effect": "none",
        "signature": "test",
    }
    assert isinstance(RESEARCH_TASK_ADAPTER.validate_python(payload), ResearchTask)
    with pytest.raises(ValidationError):
        RESEARCH_TASK_ADAPTER.validate_python(payload | {"lookback_days": 30})


def test_alpha_factory_snapshot_is_content_addressed_and_reused(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    lake.mkdir()
    registry = build_default_template_registry(
        datetime(2026, 7, 18, tzinfo=UTC)
    )
    write_parquet_dataset(registry, lake / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET)
    key = Ed25519PrivateKey.generate()
    first = seal_alpha_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 18),
        lookback_days=30,
        max_candidates=200,
        selected_v5_bundle_id=BUNDLE_ID,
        effective_registry=registry,
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    second = seal_alpha_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 18),
        lookback_days=30,
        max_candidates=200,
        selected_v5_bundle_id=BUNDLE_ID,
        effective_registry=registry,
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert first.snapshot_id == second.snapshot_id
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.template_registry_digest == alpha_factory_template_registry_digest(registry)
    verify_alpha_factory_snapshot_manifest(
        first,
        final_root=queue / "snapshots" / first.snapshot_id,
    )


def test_alpha_factory_snapshot_seals_only_latest_expanded_quality_day(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    lake.mkdir()
    registry = build_default_template_registry(datetime(2026, 7, 18, tzinfo=UTC))
    write_parquet_dataset(registry, lake / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET)
    quality_root = lake / EXPANDED_QUALITY_DATASET
    quality_root.mkdir(parents=True)
    pl.DataFrame(
        {
            "as_of_date": [date(2026, 7, 16)],
            "symbol": ["ETH-USDT"],
            "quality_score": [0.5],
        }
    ).write_parquet(quality_root / "part-old.parquet")
    pl.DataFrame(
        {
            "as_of_date": [date(2026, 7, 17), date(2026, 7, 18)],
            "symbol": ["SOL-USDT", "BNB-USDT"],
            "quality_score": [0.7, 0.9],
        }
    ).write_parquet(quality_root / "part-current.parquet")

    manifest = seal_alpha_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 18),
        lookback_days=30,
        max_candidates=200,
        selected_v5_bundle_id=BUNDLE_ID,
        effective_registry=registry,
        signing_key=Ed25519PrivateKey.generate(),
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )

    quality_references = [
        reference
        for reference in manifest.files
        if reference.dataset_name == "gold/expanded_universe_quality"
    ]
    assert len(quality_references) == 1
    snapshot_root = queue / "snapshots" / manifest.snapshot_id / "files"
    quality = pl.read_parquet(snapshot_root / quality_references[0].relative_path)
    assert quality.get_column("as_of_date").to_list() == [date(2026, 7, 18)]
    assert quality.get_column("symbol").to_list() == ["BNB-USDT"]


def test_alpha_factory_empty_result_cloud_derivation_and_import(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    lake.mkdir()
    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    task, _ = create_alpha_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 18),
        lookback_days=30,
        max_candidates=200,
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )
    assert isinstance(task, AlphaFactoryTask)
    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = json.loads((snapshot_root / "manifest.json").read_text("utf-8"))
    from quant_lab.research_plane.contracts import (  # noqa: PLC0415
        AlphaFactorySnapshotManifest,
    )

    manifest = AlphaFactorySnapshotManifest.model_validate(snapshot)
    compute = compute_alpha_factory_from_snapshot(snapshot_root, manifest, task)
    assert compute.anti_leakage["status"] == "PASS"
    result_root, result_manifest, _ = write_alpha_factory_result_bundle(
        tmp_path / "worker-results",
        task=task,
        snapshot=manifest,
        compute=compute,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        worker_key_id=WORKER_KEY_ID,
        worker_signing_key=worker_key,
        claimed_at=datetime(2026, 7, 18, 1, 0, tzinfo=UTC),
        input_bytes=manifest.total_input_bytes,
        cache_hit_bytes=manifest.total_input_bytes,
        downloaded_bytes=0,
        peak_rss_bytes=256 * 1024**2,
        compute_duration_seconds=1.0,
        max_result_bytes=256 * 1024**2,
    )
    assert result_manifest.automatic_promotion is False
    os.replace(queue / "pending" / task.task_id, queue / "running" / task.task_id)
    shutil.copytree(result_root, queue / "results" / "inbox" / task.task_id)
    imported = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        task_public_key=task_key.public_key(),
        worker_public_key=worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    assert imported.state == "completed"
    completed_status = json.loads(
        (queue / "status" / f"{task.task_id}.json").read_text("utf-8")
    )
    assert completed_status["output_bytes"] == result_manifest.output_bytes
    assert completed_status["peak_rss_bytes"] == result_manifest.peak_rss_bytes
    assert (
        completed_status["compute_duration_seconds"]
        == result_manifest.compute_duration_seconds
    )
    validation_events = [
        json.loads(line)
        for line in (queue / "validation" / f"{task.task_id}.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    strict_event = next(
        event
        for event in validation_events
        if event["check_name"] == "strict_alpha_factory_result_validation"
    )
    assert strict_event["detail"].startswith(
        f"{len(ALPHA_FACTORY_ANTI_LEAKAGE_CHECKS)} checks passed"
    )
    pointer = json.loads((lake / ALPHA_FACTORY_GENERATION_POINTER).read_text("utf-8"))
    assert pointer["research_only"] is True
    assert pointer["live_order_effect"] == "none"
    assert pointer["automatic_promotion"] is False
    for spec in ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS:
        assert read_parquet_dataset(lake / spec.relative_path).is_empty()
    assert read_parquet_dataset(lake / ALPHA_FACTORY_PROMOTION_QUEUE_DATASET).is_empty()
    assert research_plane_status(queue)["tasks"]["alpha_factory"]["state"] == "completed"
    repeated = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        task_public_key=task_key.public_key(),
        worker_public_key=worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    assert repeated.idempotent is True
    assert repeated.published_rows == imported.published_rows

    pointer_path = lake / ALPHA_FACTORY_GENERATION_POINTER
    first_dataset = next(iter(pointer["row_counts"]))
    pointer_path.write_text(
        json.dumps(
            pointer
            | {
                "row_counts": pointer["row_counts"]
                | {first_dataset: pointer["row_counts"][first_dataset] + 1}
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="dataset_row_count_mismatch"):
        verify_alpha_factory_generation(lake, pointer["generation_id"])
    pointer_path.write_text(
        json.dumps(pointer | {"live_order_effect": "orders"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="generation_safety_mismatch"):
        verify_alpha_factory_generation(lake, pointer["generation_id"])


def test_alpha_factory_snapshot_payload_release_keeps_manifest(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    lake.mkdir()
    registry = build_default_template_registry()
    write_parquet_dataset(registry, lake / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET)
    task_key = Ed25519PrivateKey.generate()
    task, _ = create_alpha_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 18),
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )
    os.replace(queue / "pending" / task.task_id, queue / "completed" / task.task_id)
    assert release_snapshot_payload(queue, task.snapshot_id, reason="test") is True
    snapshot = queue / "snapshots" / task.snapshot_id
    assert (snapshot / "manifest.json").is_file()
    assert (snapshot / "FILES_RELEASED.json").is_file()
    assert not (snapshot / "files").exists()


def test_alpha_factory_import_rejects_result_after_registry_change(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    lake.mkdir()
    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    task, _ = create_alpha_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 18),
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )
    snapshot_root = queue / "snapshots" / task.snapshot_id
    from quant_lab.research_plane.contracts import (  # noqa: PLC0415
        AlphaFactorySnapshotManifest,
    )

    snapshot = AlphaFactorySnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    compute = compute_alpha_factory_from_snapshot(snapshot_root, snapshot, task)
    result_root, _, _ = write_alpha_factory_result_bundle(
        tmp_path / "worker-results",
        task=task,
        snapshot=snapshot,
        compute=compute,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        worker_key_id=WORKER_KEY_ID,
        worker_signing_key=worker_key,
        claimed_at=datetime(2026, 7, 18, 1, 0, tzinfo=UTC),
        input_bytes=snapshot.total_input_bytes,
        cache_hit_bytes=snapshot.total_input_bytes,
        downloaded_bytes=0,
        peak_rss_bytes=256 * 1024**2,
        compute_duration_seconds=1.0,
        max_result_bytes=256 * 1024**2,
    )
    os.replace(queue / "pending" / task.task_id, queue / "running" / task.task_id)
    shutil.copytree(result_root, queue / "results" / "inbox" / task.task_id)

    changed_registry = read_parquet_dataset(
        snapshot_root / "files" / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET
    ).with_columns(
        pl.when(pl.col("template_id") == "expanded_relative_strength_v1")
        .then(pl.lit(False))
        .otherwise(pl.col("enabled"))
        .alias("enabled")
    )
    write_parquet_dataset(
        changed_registry,
        lake / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET,
    )

    with pytest.raises(
        ValueError,
        match="alpha_factory_result_superseded_by_registry_change",
    ):
        import_entry_quality_history_result(
            lake,
            queue,
            task.task_id,
            task_public_key=task_key.public_key(),
            worker_public_key=worker_key.public_key(),
            expected_task_key_id=TASK_KEY_ID,
            expected_worker_key_id=WORKER_KEY_ID,
            expected_quant_lab_commit=COMMIT,
        )
    assert (queue / "results" / "rejected" / task.task_id).is_dir()
    assert (queue / "failed" / task.task_id).is_dir()


def test_alpha_factory_publish_failure_keeps_valid_result_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake, queue, task, task_key, worker_key = _stage_empty_alpha_result(tmp_path)
    original_publish = importer_module.publish_alpha_factory_generation

    def fail_publish(*_args, **_kwargs):
        raise OSError("temporary disk pressure")

    monkeypatch.setattr(
        importer_module,
        "publish_alpha_factory_generation",
        fail_publish,
    )
    pending = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        task_public_key=task_key.public_key(),
        worker_public_key=worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    assert pending.state == "publish_retry_pending"
    assert (queue / "results" / "inbox" / task.task_id).is_dir()
    assert (queue / "running" / task.task_id).is_dir()
    assert not (queue / "results" / "rejected" / task.task_id).exists()
    status = json.loads((queue / "status" / f"{task.task_id}.json").read_text("utf-8"))
    assert status["import_status"] == "publish_retry_pending"

    monkeypatch.setattr(
        importer_module,
        "publish_alpha_factory_generation",
        original_publish,
    )
    completed = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        task_public_key=task_key.public_key(),
        worker_public_key=worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    assert completed.state == "completed"


def test_atomic_publish_rolls_back_datasets_and_pointer_when_verification_fails(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    target = lake / "gold" / "alpha_factory_result"
    staged = lake / "gold" / ".stage" / "alpha_factory_result"
    pointer = lake / "gold" / "alpha_factory_generation.json"
    target.mkdir(parents=True)
    staged.mkdir(parents=True)
    (target / "value.txt").write_text("old", encoding="utf-8")
    (staged / "value.txt").write_text("new", encoding="utf-8")
    pointer.write_text(
        json.dumps(
            {
                "generation_id": "old-generation",
                "snapshot_id": "old-snapshot",
                "task_id": "old-task",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="verification_failed"):
        commit_atomic_research_generation(
            lake,
            transaction_name="alpha_factory_test",
            generation_payload={
                "generation_id": "new-generation",
                "snapshot_id": "new-snapshot",
                "task_id": "new-task",
            },
            pointer_path=Path("gold/alpha_factory_generation.json"),
            items=[
                AtomicPublishItem(
                    target=Path("gold/alpha_factory_result"),
                    staged=Path("gold/.stage/alpha_factory_result"),
                )
            ],
            post_commit_validate=lambda: (_ for _ in ()).throw(
                RuntimeError("verification_failed")
            ),
        )

    assert (target / "value.txt").read_text("utf-8") == "old"
    assert json.loads(pointer.read_text("utf-8"))["generation_id"] == "old-generation"
    assert not (lake / "gold/.alpha_factory_test_publish_transaction.json").exists()


def test_alpha_factory_control_state_rejects_nonzero_live_notional() -> None:
    registry = build_default_template_registry(datetime(2026, 7, 18, tzinfo=UTC))
    unsafe = registry.with_columns(
        pl.when(pl.col("template_id") == "expanded_relative_strength_v1")
        .then(pl.lit('{"max_live_notional_usdt":[1]}'))
        .otherwise(pl.col("parameter_space_json"))
        .alias("parameter_space_json")
    )

    with pytest.raises(
        ValueError,
        match="alpha_factory_template_registry_nonzero_live_notional",
    ):
        prepare_alpha_factory_control_state(unsafe)


@pytest.mark.parametrize(
    ("row", "error"),
    [
        (
            {"max_live_notional_usdt": 1.0},
            "alpha_factory_result_nonzero_live_notional",
        ),
        (
            {"safety_mode": "live"},
            "alpha_factory_result_unsafe_safety_mode",
        ),
        (
            {"candidate_state": "PAPER_READY"},
            "alpha_factory_result_unknown_candidate_state",
        ),
        (
            {"decision": "LIVE_SMALL_READY"},
            "alpha_factory_result_live_state_forbidden",
        ),
        (
            {"decision": "UNKNOWN"},
            "alpha_factory_result_unknown_decision",
        ),
        (
            {
                "strategy_candidate": "v5.futures_risk_off_hedge_proxy_shadow",
                "decision": "PAPER_READY",
            },
            "alpha_factory_result_futures_proxy_paper_ready",
        ),
        (
            {
                "template_name": "factor_strategy_bridge",
                "decision": "PAPER_READY",
            },
            "alpha_factory_result_factor_bridge_not_research",
        ),
        (
            {
                "template_name": "factor_strategy_bridge",
                "parameter_json": "{}",
            },
            "alpha_factory_result_factor_bridge_not_strategy_review_only",
        ),
        (
            {
                "strategy_candidate": "v5.futures_risk_off_hedge_proxy_shadow",
                "futures_data_available": True,
            },
            "alpha_factory_result_futures_data_claim_forbidden",
        ),
        (
            {
                "strategy_candidate": "v5.futures_risk_off_hedge_proxy_shadow",
                "funding_available": True,
            },
            "alpha_factory_result_funding_claim_forbidden",
        ),
    ],
)
def test_alpha_factory_result_safety_validator_fails_closed(
    row: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _validate_alpha_frame_safety(pl.DataFrame([row]).lazy(), "test_output")


def test_alpha_factory_result_scope_and_primary_keys_reject_nulls() -> None:
    task = AlphaFactoryTask(
        task_id="alpha-task",
        snapshot_id="alpha-snapshot",
        as_of_date=date(2026, 7, 18),
        quant_lab_commit=COMMIT,
        alpha_factory_schema_version="alpha_factory.v0.1",
        second_stage_schema_version="second_stage_alpha_factory.v0.1",
        template_registry_digest="a" * 64,
        selected_v5_bundle_id=BUNDLE_ID,
        snapshot_manifest_sha256="b" * 64,
        requested_at=datetime(2026, 7, 18, tzinfo=UTC),
        signature_key_id=TASK_KEY_ID,
        signature="test",
    )
    null_scope = pl.DataFrame(
        {"as_of_date": [None], "candidate_id": ["candidate"]},
        schema={"as_of_date": pl.Date, "candidate_id": pl.Utf8},
    ).lazy()
    with pytest.raises(ValueError, match="scope_null.*as_of_date"):
        _validate_alpha_frame_scope(
            null_scope,
            dataset_name="alpha_factory_result",
            task=task,
        )

    null_key = pl.DataFrame(
        {"as_of_date": [date(2026, 7, 18)], "candidate_id": [None]},
        schema={"as_of_date": pl.Date, "candidate_id": pl.Utf8},
    ).lazy()
    with pytest.raises(ValueError, match="primary_key_null"):
        _validate_lazy_unique_keys(
            null_key,
            ("as_of_date", "candidate_id"),
            "alpha_factory_result",
        )


def test_factor_bridge_report_accepts_only_read_only_effects(tmp_path: Path) -> None:
    report = tmp_path / "factor_strategy_bridge_candidates.csv"
    pl.DataFrame(
        [
            {
                "factor_id": "core.test",
                "live_order_effect": "none_read_only_research",
                "eligible_for_alpha_factory": "strategy_review_pending",
            }
        ]
    ).write_csv(report)
    _validate_factor_bridge_report(report)

    pl.DataFrame(
        [
            {
                "factor_id": "core.test",
                "live_order_effect": "submit_order",
                "eligible_for_alpha_factory": "strategy_review_pending",
            }
        ]
    ).write_csv(report)
    with pytest.raises(ValueError, match="live_effect_forbidden"):
        _validate_factor_bridge_report(report)


def test_alpha_factory_evidence_merge_preserves_other_producers_and_dates() -> None:
    day = date(2026, 7, 18)
    existing_summary = _typed_rows(
        SUMMARY_SCHEMA,
        [
            {
                "as_of_date": day.isoformat(),
                "strategy_candidate": "manual.research_candidate",
                "decision": "RESEARCH",
            },
            {
                "as_of_date": day.isoformat(),
                "strategy_candidate": "v5.alt_impulse_shadow",
                "decision": "KEEP_SHADOW",
                "sample_count": 10,
            },
            {
                "as_of_date": "2026-07-17",
                "strategy_candidate": "v5.alt_impulse_shadow",
                "decision": "KEEP_SHADOW",
                "sample_count": 8,
            },
        ],
    )
    delta_summary = _typed_rows(
        SUMMARY_SCHEMA,
        [
            {
                "as_of_date": day.isoformat(),
                "strategy_candidate": "v5.alt_impulse_shadow",
                "decision": "RESEARCH",
                "sample_count": 12,
            }
        ],
    )
    merged = merge_alpha_factory_managed_evidence(
        existing_summary,
        delta_summary,
        as_of_date=day,
        sample=False,
    )

    assert merged.filter(
        (pl.col("as_of_date") == day.isoformat())
        & (pl.col("strategy_candidate") == "manual.research_candidate")
    ).height == 1
    assert merged.filter(
        (pl.col("as_of_date") == "2026-07-17")
        & (pl.col("strategy_candidate") == "v5.alt_impulse_shadow")
    ).height == 1
    current = merged.filter(
        (pl.col("as_of_date") == day.isoformat())
        & (pl.col("strategy_candidate") == "v5.alt_impulse_shadow")
    )
    assert current.height == 1
    assert current["sample_count"][0] == 12

    existing_samples = _typed_rows(
        SAMPLE_SCHEMA,
        [
            {
                "as_of_date": day.isoformat(),
                "strategy_candidate": "manual.research_candidate",
                "candidate_id": "manual-1",
            },
            {
                "as_of_date": day.isoformat(),
                "strategy_candidate": "v5.alt_impulse_shadow",
                "candidate_id": "stale-alpha-1",
            },
        ],
    )
    cleared = merge_alpha_factory_managed_evidence(
        existing_samples,
        pl.DataFrame(schema=SAMPLE_SCHEMA),
        as_of_date=day,
        sample=True,
    )
    assert cleared.filter(
        pl.col("strategy_candidate") == "manual.research_candidate"
    ).height == 1
    assert cleared.filter(
        pl.col("strategy_candidate") == "v5.alt_impulse_shadow"
    ).is_empty()


def _typed_rows(
    schema: dict[str, pl.DataType],
    rows: list[dict[str, object]],
) -> pl.DataFrame:
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return frame.select(
        [
            (
                pl.col(column).cast(dtype, strict=False)
                if column in frame.columns
                else pl.lit(None).cast(dtype).alias(column)
            )
            for column, dtype in schema.items()
        ]
    )


def _stage_empty_alpha_result(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    AlphaFactoryTask,
    Ed25519PrivateKey,
    Ed25519PrivateKey,
]:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    lake.mkdir()
    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    task, _ = create_alpha_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 18),
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )
    from quant_lab.research_plane.contracts import (  # noqa: PLC0415
        AlphaFactorySnapshotManifest,
    )

    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = AlphaFactorySnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    compute = compute_alpha_factory_from_snapshot(snapshot_root, snapshot, task)
    result_root, _, _ = write_alpha_factory_result_bundle(
        tmp_path / "worker-results",
        task=task,
        snapshot=snapshot,
        compute=compute,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        worker_key_id=WORKER_KEY_ID,
        worker_signing_key=worker_key,
        claimed_at=datetime(2026, 7, 18, 1, 0, tzinfo=UTC),
        input_bytes=snapshot.total_input_bytes,
        cache_hit_bytes=snapshot.total_input_bytes,
        downloaded_bytes=0,
        peak_rss_bytes=256 * 1024**2,
        compute_duration_seconds=1.0,
        max_result_bytes=256 * 1024**2,
    )
    os.replace(queue / "pending" / task.task_id, queue / "running" / task.task_id)
    shutil.copytree(result_root, queue / "results" / "inbox" / task.task_id)
    return lake, queue, task, task_key, worker_key
