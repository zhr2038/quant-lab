from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from polars.testing import assert_frame_equal
from pydantic import ValidationError
from typer.testing import CliRunner

from quant_lab.cli import app
from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.factors.factory import (
    LEGACY_MAIN_DECISION_POLICY,
    build_and_publish_factor_factory,
)
from quant_lab.factors.plan import build_effective_factor_plan
from quant_lab.features.publish import publish_features
from quant_lab.research_plane import factor_factory_result as factor_factory_result_module
from quant_lab.research_plane import factor_factory_snapshot as factor_factory_snapshot_module
from quant_lab.research_plane.contracts import FactorFactorySnapshotManifest, FactorFactoryTask
from quant_lab.research_plane.factor_factory_publish import (
    FACTOR_FACTORY_DATASETS,
    FACTOR_FACTORY_GENERATION_POINTER,
    FACTOR_FACTORY_NO_UPDATE_POINTER,
    FACTOR_FACTORY_PRIMARY_KEYS,
    _configure_duckdb_for_bounded_scan,
    _dataset_digest,
    _require_writable_spill_directory,
    _validate_published_candidates,
    publish_factor_factory_generation,
    verify_factor_factory_generation,
)
from quant_lab.research_plane.factor_factory_result import (
    validate_factor_factory_result_bundle,
)
from quant_lab.research_plane.factor_factory_snapshot import (
    cleanup_stale_factor_factory_rehydrate_partials,
    preflight_factor_factory_snapshot,
    rehydrate_factor_factory_snapshot_payload,
    verify_factor_factory_snapshot_manifest,
)
from quant_lab.research_plane.factor_research_publish import (
    FACTOR_RESEARCH_GENERATION_POINTER,
    verify_factor_research_generation,
)
from quant_lab.research_plane.importer import import_entry_quality_history_result
from quant_lab.research_plane.queue import create_factor_factory_task
from quant_lab.research_plane.result import validate_research_task_snapshot
from quant_lab.research_plane.signatures import verify_payload
from quant_lab.research_plane.snapshot_gc import release_snapshot_payload
from quant_lab.research_plane.status import research_plane_status
from quant_lab.research_worker import factor_factory as factor_factory_worker_module
from quant_lab.research_worker import runner as runner_module
from quant_lab.research_worker.factor_factory import (
    FACTOR_FACTORY_ANTI_LEAKAGE_CHECKS,
    compute_factor_factory_result,
)
from quant_lab.research_worker.result_writer import write_factor_factory_result_bundle
from tests.helpers.factor_research import seed_verified_factor_generation

COMMIT = "a" * 40
TASK_KEY_ID = "cloud-research-v1"
WORKER_KEY_ID = "nas-worker-v1"
CLI_RUNNER = CliRunner()


def test_factor_factory_candidate_limit_is_scoped_to_current_as_of_date() -> None:
    historical = pl.DataFrame(
        {
            "as_of_date": ["2026-05-19"] * 250,
            "candidate_state": ["KEEP_SHADOW"] * 250,
            "manual_review_required": [True] * 250,
            "source": ["factors.factory.v0.1"] * 250,
        }
    )
    current = historical.head(22).with_columns(pl.lit("2026-05-20").alias("as_of_date"))
    _validate_published_candidates(
        pl.concat([historical, current]),
        as_of_date="2026-05-20",
    )

    too_many_current = historical.with_columns(pl.lit("2026-05-20").alias("as_of_date"))
    with pytest.raises(ValueError, match="candidate_limit_exceeded"):
        _validate_published_candidates(too_many_current, as_of_date="2026-05-20")


def test_factor_factory_duckdb_spill_directory_is_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spill = tmp_path / "spill"
    spill.mkdir()
    _require_writable_spill_directory(spill)
    assert not list(spill.iterdir())

    statements: list[str] = []
    connection = SimpleNamespace(execute=statements.append)
    _configure_duckdb_for_bounded_scan(connection, spill)
    assert "SET threads = 2" in statements
    assert "SET memory_limit = '768MB'" in statements
    assert any(statement.startswith("SET temp_directory = ") for statement in statements)

    original_open = Path.open

    def reject_probe(path: Path, *args: object, **kwargs: object):
        if path.parent == spill:
            raise PermissionError("read-only spill")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_probe)
    with pytest.raises(RuntimeError, match="duckdb_spill_directory_not_writable"):
        _require_writable_spill_directory(spill)


def test_factor_factory_signed_full_history_round_trip(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_bars(lake, count=180)
    _write_costs(lake)
    publish_features(lake)
    factor_research_generation_id = "factor-research-before-factor-factory"
    seed_verified_factor_generation(
        lake,
        as_of_date=date(2026, 5, 20),
        generation_id=factor_research_generation_id,
    )
    factor_research_rows_before = verify_factor_research_generation(
        lake, factor_research_generation_id
    )
    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()

    task, status = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        horizon_bars=(4, 8),
        min_samples=20,
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = FactorFactorySnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    verify_payload(task, task.signature, task_key.public_key())
    validate_research_task_snapshot(
        task,
        snapshot,
        task_public_key=task_key.public_key(),
        expected_key_id=TASK_KEY_ID,
        expected_quant_lab_commit=COMMIT,
        snapshot_root=snapshot_root,
    )
    assert task.result_mode == "PARITY_FULL"
    assert task.history_mode == "bootstrap_full"
    assert snapshot.factor_plan.include_legacy_enumeration is True
    assert status.task_type == "factor_factory"

    compute = compute_factor_factory_result(snapshot_root, snapshot, task)
    assert compute.no_update_reason is None
    assert compute.values.height > 0
    assert len(compute.anti_leakage["checks"]) == len(FACTOR_FACTORY_ANTI_LEAKAGE_CHECKS)
    assert compute.anti_leakage["status"] == "PASS"
    assert compute.anti_leakage["violation_count"] == 0
    assert compute.worker_report["streaming_enabled"] is True
    assert compute.worker_report["factor_value_stage_partition_count"] > 0
    assert compute.worker_report["peak_rss_bytes_observed"] > 0
    assert compute.worker_report["temporary_disk_peak_bytes"] > 0
    assert "feature_scan_released" in compute.worker_report["stage_release_events"]

    results_root = tmp_path / "worker-results"
    result_root, manifest, receipt = write_factor_factory_result_bundle(
        results_root,
        task=task,
        snapshot=snapshot,
        compute=compute,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        worker_key_id=WORKER_KEY_ID,
        worker_signing_key=worker_key,
        claimed_at=datetime.now(UTC),
        input_bytes=snapshot.total_input_bytes,
        cache_hit_bytes=snapshot.total_input_bytes,
        downloaded_bytes=0,
        peak_rss_bytes=1,
        compute_duration_seconds=1.0,
        max_result_bytes=2 * 1024**3,
        max_value_partition_bytes=32 * 1024**2,
        max_value_partition_rows=500,
    )
    assert manifest.value_partitions
    assert all("date=" in item.relative_path for item in manifest.value_partitions)
    assert receipt.output_rows > 0

    os.replace(queue / "pending" / task.task_id, queue / "running" / task.task_id)
    inbox = queue / "results" / "inbox" / task.task_id
    inbox.parent.mkdir(parents=True, exist_ok=True)
    os.replace(result_root, inbox)
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
    assert set(imported.published_rows) == set(FACTOR_FACTORY_DATASETS)
    assert verify_factor_factory_generation(lake, manifest.generation_id) == (
        imported.published_rows
    )
    factor_research_rows_after = verify_factor_research_generation(
        lake, factor_research_generation_id
    )
    assert factor_research_rows_after["factor_value"] >= factor_research_rows_before["factor_value"]
    factor_research_pointer = (lake / FACTOR_RESEARCH_GENERATION_POINTER).read_text("utf-8")
    assert manifest.generation_id in factor_research_pointer
    pointer = (lake / FACTOR_FACTORY_GENERATION_POINTER).read_text("utf-8")
    assert "none_read_only_research" in pointer
    candidates = read_parquet_dataset(lake / "gold" / "factor_candidate")
    assert candidates.get_column("manual_review_required").all()
    assert not candidates.get_column("candidate_state").is_in(["LIVE", "CANARY", "ENFORCE"]).any()
    factor_status = research_plane_status(queue)["tasks"]["factor_factory"]
    assert factor_status["state"] == "completed"
    assert factor_status["task"]["factor_plan_digest"] == task.factor_plan_digest
    assert factor_status["task"]["factor_count"] == manifest.factor_count
    assert factor_status["task"]["value_rows"] == sum(
        item.row_count for item in manifest.value_partitions
    )
    successor = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        horizon_bars=(4, 8),
        min_samples=20,
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert successor.state == "already_current"
    assert successor.task_created is False
    assert successor.snapshot_materialized is False
    assert successor.current_generation_id == manifest.generation_id
    assert research_plane_status(queue)["tasks"]["factor_factory"]["state"] == "up_to_date"


def test_factor_factory_empty_input_completes_without_gold_update(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    lake.mkdir()
    queue = tmp_path / "queue"
    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    task, _ = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    snapshot_root = queue / "snapshots" / task.snapshot_id
    snapshot = FactorFactorySnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    compute = compute_factor_factory_result(snapshot_root, snapshot, task)
    assert compute.no_update_reason == "feature_value_missing_or_empty"
    result_root, manifest, _ = write_factor_factory_result_bundle(
        tmp_path / "worker-results",
        task=task,
        snapshot=snapshot,
        compute=compute,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        worker_key_id=WORKER_KEY_ID,
        worker_signing_key=worker_key,
        claimed_at=datetime.now(UTC),
        input_bytes=0,
        cache_hit_bytes=0,
        downloaded_bytes=0,
        peak_rss_bytes=1,
        compute_duration_seconds=0.1,
        max_result_bytes=2 * 1024**3,
    )
    assert manifest.completed_no_update is True
    assert not manifest.outputs
    assert not manifest.value_partitions
    os.replace(queue / "pending" / task.task_id, queue / "running" / task.task_id)
    inbox = queue / "results" / "inbox" / task.task_id
    inbox.parent.mkdir(parents=True, exist_ok=True)
    os.replace(result_root, inbox)
    result = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        task_public_key=task_key.public_key(),
        worker_public_key=worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    assert result.state == "completed"
    assert not (lake / FACTOR_FACTORY_GENERATION_POINTER).exists()
    assert (lake / FACTOR_FACTORY_NO_UPDATE_POINTER).is_file()
    no_update_pointer = json.loads(
        (lake / FACTOR_FACTORY_NO_UPDATE_POINTER).read_text("utf-8")
    )
    assert {
        "schema_version",
        "quant_lab_commit",
        "factor_plan_digest",
        "feature_input_digest",
        "market_input_digest",
        "cost_input_digest",
        "combined_input_digest",
        "reason",
        "observed_at",
        "research_only",
        "live_order_effect",
    }.issubset(no_update_pointer)
    assert no_update_pointer["reason"] == "factor_factory_input_still_empty"
    assert all(not (lake / path).exists() for path in FACTOR_FACTORY_DATASETS.values())
    assert (snapshot_root / "FILES_RELEASED.json").is_file()
    assert not (snapshot_root / "files").exists()

    repeated = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert repeated.state == "already_current_no_update"
    assert repeated.task_created is False
    assert repeated.snapshot_materialized is False
    repeated_status = research_plane_status(queue)["tasks"]["factor_factory"]
    assert repeated_status["request_outcome"] == "already_current_no_update"
    assert repeated_status["no_update_reason"] == "factor_factory_input_still_empty"

    _write_bars(lake, count=12)
    publish_features(lake)
    changed = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert changed.state == "task_created"
    assert changed.task_created is True


def test_factor_factory_nas_round_trip_matches_legacy_full_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_lake = tmp_path / "legacy-lake"
    nas_lake = tmp_path / "nas-lake"
    _write_bars(legacy_lake, count=180)
    _write_costs(legacy_lake)
    publish_features(legacy_lake)
    shutil.copytree(legacy_lake, nas_lake)
    build_and_publish_factor_factory(
        legacy_lake,
        as_of_date=date(2026, 5, 20),
        horizon_bars=(4, 8),
        min_samples=20,
        legacy_enumeration=True,
        decision_policy=LEGACY_MAIN_DECISION_POLICY,
    )

    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    task, _ = create_factor_factory_task(
        nas_lake,
        tmp_path / "queue",
        as_of_date=date(2026, 5, 20),
        horizon_bars=(4, 8),
        min_samples=20,
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    snapshot_root = tmp_path / "queue" / "snapshots" / task.snapshot_id
    snapshot = FactorFactorySnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    compute = compute_factor_factory_result(snapshot_root, snapshot, task)
    result_root, manifest, receipt = write_factor_factory_result_bundle(
        tmp_path / "results",
        task=task,
        snapshot=snapshot,
        compute=compute,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        worker_key_id=WORKER_KEY_ID,
        worker_signing_key=worker_key,
        claimed_at=datetime.now(UTC),
        input_bytes=snapshot.total_input_bytes,
        cache_hit_bytes=0,
        downloaded_bytes=snapshot.total_input_bytes,
        peak_rss_bytes=1,
        compute_duration_seconds=1.0,
        max_result_bytes=2 * 1024**3,
    )
    handoff_marker = result_root / ".HANDOFF_READY"
    handoff_marker.touch()
    with pytest.raises(ValueError, match="file_count_limit"):
        validate_factor_factory_result_bundle(
            result_root,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_key.public_key(),
            expected_worker_key_id=WORKER_KEY_ID,
            max_result_bytes=2 * 1024**3,
            max_file_count=1,
        )
    handoff_marker.write_text("not-empty", encoding="utf-8")
    with pytest.raises(ValueError, match="handoff_marker_invalid"):
        validate_factor_factory_result_bundle(
            result_root,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_key.public_key(),
            expected_worker_key_id=WORKER_KEY_ID,
            max_result_bytes=2 * 1024**3,
        )
    handoff_marker.write_bytes(b"")
    original_unique = factor_factory_result_module._validate_unique_keys

    def unexpected_global_scan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("uncompressed gate must precede global scans")

    monkeypatch.setattr(
        factor_factory_result_module,
        "_validate_unique_keys",
        unexpected_global_scan,
    )
    with pytest.raises(ValueError, match="partition_uncompressed_size_limit_exceeded"):
        validate_factor_factory_result_bundle(
            result_root,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_key.public_key(),
            expected_worker_key_id=WORKER_KEY_ID,
            max_result_bytes=2 * 1024**3,
            max_value_partition_uncompressed_bytes=1,
        )
    with pytest.raises(ValueError, match="uncompressed_size_limit_exceeded"):
        validate_factor_factory_result_bundle(
            result_root,
            manifest=manifest,
            receipt=receipt,
            task=task,
            snapshot=snapshot,
            worker_public_key=worker_key.public_key(),
            expected_worker_key_id=WORKER_KEY_ID,
            max_result_bytes=2 * 1024**3,
            max_uncompressed_bytes=1,
        )
    monkeypatch.setattr(
        factor_factory_result_module,
        "_validate_unique_keys",
        original_unique,
    )
    validated = validate_factor_factory_result_bundle(
        result_root,
        manifest=manifest,
        receipt=receipt,
        task=task,
        snapshot=snapshot,
        worker_public_key=worker_key.public_key(),
        expected_worker_key_id=WORKER_KEY_ID,
        max_result_bytes=2 * 1024**3,
    )
    publish_factor_factory_generation(nas_lake, validated)

    volatile = {"created_at", "calculated_at"}
    for dataset_name, relative_path in FACTOR_FACTORY_DATASETS.items():
        legacy = read_parquet_dataset(legacy_lake / relative_path)
        nas = read_parquet_dataset(nas_lake / relative_path)
        comparable_columns = [column for column in legacy.columns if column not in volatile]
        keys = list(FACTOR_FACTORY_PRIMARY_KEYS[dataset_name])
        assert_frame_equal(
            legacy.select(comparable_columns).sort(keys),
            nas.select(comparable_columns).sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_factor_factory_importer_detects_duplicate_keys_across_partitions(
    tmp_path: Path,
) -> None:
    partitions = tmp_path / "factor-value"
    partitions.mkdir()
    row = {
        "factor_id": "factor-one",
        "factor_version": "v0.1",
        "symbol": "BTC-USDT",
        "timeframe": "1H",
        "ts": datetime(2026, 5, 20, tzinfo=UTC),
    }
    pl.DataFrame([row]).write_parquet(partitions / "part-000.parquet")
    pl.DataFrame([row]).write_parquet(partitions / "part-001.parquet")

    lazy = pl.scan_parquet(str(partitions / "*.parquet"))
    with pytest.raises(ValueError, match="factor_factory_result_duplicate_key:factor_value"):
        factor_factory_result_module._validate_unique_keys(
            lazy,
            list(FACTOR_FACTORY_PRIMARY_KEYS["factor_value"]),
            "factor_value",
        )


def test_factor_factory_requests_coalesce_unclaimed_successor(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    lake.mkdir()
    queue = tmp_path / "queue"
    task_key = Ed25519PrivateKey.generate()
    first, _ = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 19),
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    second, _ = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert first.task_id != second.task_id
    assert (queue / "cancelled" / first.task_id).is_dir()
    assert (queue / "pending" / second.task_id).is_dir()
    pending = [path for path in (queue / "pending").iterdir() if path.is_dir()]
    assert pending == [queue / "pending" / second.task_id]


def test_factor_factory_contract_and_plan_are_strict_and_replayable(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    lake.mkdir()
    task_key = Ed25519PrivateKey.generate()
    task, _ = create_factor_factory_task(
        lake,
        tmp_path / "queue",
        as_of_date=date(2026, 5, 20),
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    payload = task.model_dump(mode="json")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FactorFactoryTask.model_validate(payload | {"factor_candidate": "worker-owned"})
    with pytest.raises(ValidationError):
        FactorFactoryTask.model_validate(payload | {"cost_quantile": "p95"})

    created_at = datetime(2026, 5, 20, tzinfo=UTC)
    first = build_effective_factor_plan(
        ["zeta", "alpha", "alpha"],
        feature_set="core",
        feature_version="v0.1",
        factor_version="v0.1",
        timeframe="1H",
        max_factors=200,
        quant_lab_commit=COMMIT,
        created_at=created_at,
    )
    replay = build_effective_factor_plan(
        ["alpha", "zeta"],
        feature_set="core",
        feature_version="v0.1",
        factor_version="v0.1",
        timeframe="1H",
        max_factors=200,
        quant_lab_commit=COMMIT,
        created_at=created_at,
    )
    changed = build_effective_factor_plan(
        ["alpha", "beta", "zeta"],
        feature_set="core",
        feature_version="v0.1",
        factor_version="v0.1",
        timeframe="1H",
        max_factors=200,
        quant_lab_commit=COMMIT,
        created_at=created_at,
    )
    assert replay == first
    assert changed.plan_digest != first.plan_digest
    assert len({item.factor_id for item in first.factor_specs}) == first.factor_count
    assert all(item.causal for item in first.factor_specs)


def test_factor_factory_snapshot_preserves_legacy_full_history_and_latest_cost(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    _write_bars(lake, count=180)
    _write_costs(lake, days=("2026-05-10", "2026-05-25"))
    publish_features(lake)
    task, _ = create_factor_factory_task(
        lake,
        tmp_path / "queue",
        as_of_date=date(2026, 5, 12),
        horizon_bars=(4, 8),
        min_samples=20,
        signing_key=Ed25519PrivateKey.generate(),
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    snapshot_root = tmp_path / "queue" / "snapshots" / task.snapshot_id
    snapshot = FactorFactorySnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    assert set(snapshot.datasets) == {
        "gold/feature_value",
        "silver/market_bar",
        "gold/cost_bucket_daily",
    }
    assert snapshot.feature_max_ts is not None
    assert snapshot.feature_max_ts.date() > task.as_of_date
    assert {item.cost_date for item in snapshot.cost_snapshot} == {"2026-05-25"}
    compute = compute_factor_factory_result(snapshot_root, snapshot, task)
    assert compute.anti_leakage["status"] == "PASS"


def test_factor_factory_snapshot_identity_is_content_addressed_before_materialization(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_bars(lake, count=24)
    _write_costs(lake)
    publish_features(lake)

    original = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        quant_lab_commit=COMMIT,
    )
    next_day = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        quant_lab_commit=COMMIT,
    )
    assert next_day.snapshot_id == original.snapshot_id
    assert next_day.factor_plan.plan_digest == original.factor_plan.plan_digest

    parameter_change = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        min_samples=101,
        quant_lab_commit=COMMIT,
    )
    assert parameter_change.snapshot_id != original.snapshot_id

    plan_change = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        factor_version="v0.2",
        quant_lab_commit=COMMIT,
    )
    assert plan_change.factor_plan.plan_digest != original.factor_plan.plan_digest
    assert plan_change.snapshot_id != original.snapshot_id

    _write_costs(lake, days=("2026-05-19", "2026-05-21"))
    cost_change = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        quant_lab_commit=COMMIT,
    )
    assert cost_change.source_input_digest == original.source_input_digest
    assert cost_change.cost_input_digest != original.cost_input_digest
    assert cost_change.snapshot_id != original.snapshot_id

    _write_bars(lake, count=25)
    publish_features(lake)
    source_change = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        quant_lab_commit=COMMIT,
    )
    assert source_change.source_input_digest != cost_change.source_input_digest
    assert source_change.snapshot_id != cost_change.snapshot_id

    commit_change = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        quant_lab_commit="b" * 40,
    )
    assert commit_change.snapshot_id != source_change.snapshot_id


def test_factor_factory_fingerprint_ignores_unrelated_feature_scopes(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_bars(lake, count=24)
    _write_costs(lake)
    publish_features(lake)
    original = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        quant_lab_commit=COMMIT,
    )
    feature_root = lake / "gold" / "feature_value"
    relevant = pl.read_parquet(feature_root / "data.parquet")
    relevant.head(8).with_columns(pl.lit("4H").alias("timeframe")).write_parquet(
        feature_root / "unrelated-timeframe.parquet"
    )
    relevant.head(8).with_columns(pl.lit("experimental").alias("feature_set")).write_parquet(
        feature_root / "unrelated-feature-set.parquet"
    )
    with_unrelated = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        quant_lab_commit=COMMIT,
    )
    assert with_unrelated.input_fingerprint.combined_input_digest == (
        original.input_fingerprint.combined_input_digest
    )
    assert with_unrelated.snapshot_id == original.snapshot_id

    unrelated = pl.read_parquet(feature_root / "unrelated-timeframe.parquet")
    unrelated.with_columns((pl.col("value") + 123.0).alias("value")).write_parquet(
        feature_root / "unrelated-timeframe.parquet"
    )
    changed_unrelated = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        quant_lab_commit=COMMIT,
    )
    assert changed_unrelated.input_fingerprint == with_unrelated.input_fingerprint.model_copy(
        update={"observed_at": changed_unrelated.input_fingerprint.observed_at}
    )
    assert changed_unrelated.snapshot_id == original.snapshot_id


def test_previous_generation_changes_task_not_snapshot_identity(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    lake.mkdir()
    queue = tmp_path / "queue"
    key = Ed25519PrivateKey.generate()
    _write_factor_factory_binding(lake, generation_id="generation-one", digest_character="1")
    first = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert first.task is not None
    assert first.task.previous_generation_id == "generation-one"
    snapshot = FactorFactorySnapshotManifest.model_validate_json(
        (queue / "snapshots" / first.snapshot_id / "manifest.json").read_text("utf-8")
    )
    assert snapshot.previous_generation_id is None
    assert snapshot.previous_generation_digest is None
    assert snapshot.previous_generation_manifest is None

    _write_factor_factory_binding(lake, generation_id="generation-two", digest_character="2")
    second = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert second.task is not None
    assert second.snapshot_id == first.snapshot_id
    assert second.task.task_id != first.task.task_id
    assert second.task.previous_generation_id == "generation-two"


def test_factor_factory_no_change_fast_path_precedes_all_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_bars(lake, count=24)
    _write_costs(lake)
    publish_features(lake)
    preflight = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        quant_lab_commit=COMMIT,
    )
    _write_verified_factor_factory_generation(
        lake,
        identity_payload=preflight.identity_payload,
        snapshot_id=preflight.snapshot_id,
        generation_id="already-current-generation",
    )
    pointer_path = lake / FACTOR_FACTORY_GENERATION_POINTER

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("No-Change must return before Snapshot materialization")

    monkeypatch.setattr(factor_factory_snapshot_module, "_materialize_feature_files", unexpected)
    monkeypatch.setattr(factor_factory_snapshot_module, "_materialize_market_files", unexpected)
    monkeypatch.setattr(factor_factory_snapshot_module, "_materialize_cost_selection", unexpected)
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", unexpected)
    result = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        signing_key=Ed25519PrivateKey.generate(),
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert result.state == "already_current"
    assert result.task_created is False
    assert result.snapshot_materialized is False
    assert not list((queue / "snapshots").iterdir())
    status = research_plane_status(queue)["tasks"]["factor_factory"]
    assert status["request_outcome"] == "already_current"
    assert status["fingerprint_matches_generation"] is True
    assert status["snapshot_materialized"] is False
    assert status["already_current_at"] is not None

    pointer = json.loads(pointer_path.read_text("utf-8"))
    pointer["source_input_digest"] = "f" * 64
    pointer["published_at"] = datetime.now(UTC).isoformat()
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    deferred = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        signing_key=Ed25519PrivateKey.generate(),
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        min_recompute_interval_seconds=6 * 60 * 60,
    )
    assert deferred.state == "recompute_deferred"
    assert deferred.task_created is False
    assert deferred.snapshot_materialized is False


def test_factor_factory_no_change_rejects_damaged_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_bars(lake, count=24)
    _write_costs(lake)
    publish_features(lake)
    preflight = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        quant_lab_commit=COMMIT,
    )
    _write_verified_factor_factory_generation(
        lake,
        identity_payload=preflight.identity_payload,
        snapshot_id=preflight.snapshot_id,
        generation_id="damaged-generation",
    )
    (lake / FACTOR_FACTORY_DATASETS["factor_evidence"] / "data.parquet").unlink()

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("integrity failure must not materialize a Snapshot")

    monkeypatch.setattr(factor_factory_snapshot_module, "_materialize_feature_files", unexpected)
    monkeypatch.setattr(factor_factory_snapshot_module, "_materialize_market_files", unexpected)
    monkeypatch.setattr(factor_factory_snapshot_module, "_materialize_cost_selection", unexpected)
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", unexpected)
    result = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        signing_key=Ed25519PrivateKey.generate(),
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )

    assert result.state == "generation_integrity_failed"
    assert result.task_created is False
    assert result.snapshot_materialized is False
    assert result.fingerprint_matches_generation is True
    assert "factor_factory_dataset_row_count_mismatch:factor_evidence" in result.reason
    assert not list((queue / "snapshots").iterdir())
    status = research_plane_status(queue)["tasks"]["factor_factory"]
    assert status["state"] == "generation_integrity_failed"
    assert status["health_state"] == "generation_integrity_failed"
    assert status["request_outcome"] == "generation_integrity_failed"
    assert status["already_current_at"] is None


def test_factor_factory_status_reports_snapshot_transient_states(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    sealing = queue / "snapshots" / ".sealing.snapshot-one.test.partial"
    sealing.mkdir(parents=True)
    sealing_status = research_plane_status(queue)["tasks"]["factor_factory"]
    assert sealing_status["state"] == "snapshot_sealing"
    assert sealing_status["snapshot_payload_state"] == "sealing"

    sealing.rmdir()
    rehydrating = queue / "snapshots" / ".rehydrate.snapshot-one.test.partial"
    rehydrating.mkdir()
    rehydrate_status = research_plane_status(queue)["tasks"]["factor_factory"]
    assert rehydrate_status["state"] == "snapshot_rehydrating"
    assert rehydrate_status["snapshot_payload_state"] == "rehydrating"


def test_factor_factory_cli_treats_no_change_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    lake.mkdir()
    preflight = preflight_factor_factory_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        quant_lab_commit=COMMIT,
    )
    _write_verified_factor_factory_generation(
        lake,
        identity_payload=preflight.identity_payload,
        snapshot_id=preflight.snapshot_id,
        generation_id="cli-current-generation",
    )
    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "task-key.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    monkeypatch.setenv("QUANT_LAB_NAS_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("QUANT_LAB_NAS_FACTOR_FACTORY_ENABLED", "1")
    result = CLI_RUNNER.invoke(
        app,
        [
            "request-factor-factory",
            "--lake-root",
            str(lake),
            "--queue-root",
            str(queue),
            "--signing-key-path",
            str(key_path),
            "--key-id",
            TASK_KEY_ID,
            "--quant-lab-commit",
            COMMIT,
            "--date",
            "2026-05-21",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "FACTOR_FACTORY_ALREADY_CURRENT" in result.output
    assert '"state": "already_current"' in result.output

    (lake / FACTOR_FACTORY_DATASETS["factor_value"] / "data.parquet").unlink()
    damaged = CLI_RUNNER.invoke(
        app,
        [
            "request-factor-factory",
            "--lake-root",
            str(lake),
            "--queue-root",
            str(queue),
            "--signing-key-path",
            str(key_path),
            "--key-id",
            TASK_KEY_ID,
            "--quant-lab-commit",
            COMMIT,
            "--date",
            "2026-05-21",
        ],
    )
    assert damaged.exit_code == 0, damaged.output
    assert "FACTOR_FACTORY_GENERATION_INTEGRITY_FAILED" in damaged.output
    assert '"state": "generation_integrity_failed"' in damaged.output


def test_released_factor_factory_snapshot_rehydrates_once_and_preserves_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_bars(lake, count=24)
    _write_costs(lake)
    publish_features(lake)
    key = Ed25519PrivateKey.generate()
    created = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert created.task is not None
    snapshot_root = queue / "snapshots" / created.snapshot_id
    manifest_text = (snapshot_root / "manifest.json").read_text("utf-8")
    manifest = FactorFactorySnapshotManifest.model_validate_json(manifest_text)
    signature = manifest.signature
    os.replace(
        queue / "pending" / created.task.task_id,
        queue / "completed" / created.task.task_id,
    )
    assert release_snapshot_payload(queue, created.snapshot_id, reason="rehydrate_test")
    assert (snapshot_root / "FILES_RELEASED.json").is_file()
    assert not (snapshot_root / "files").exists()

    crash_partial = queue / "snapshots" / f".rehydrate.{created.snapshot_id}.crash.partial"
    crash_partial.mkdir()
    (crash_partial / "REHYDRATE.json").write_text(
        json.dumps({"snapshot_id": created.snapshot_id}), encoding="utf-8"
    )
    (crash_partial / "orphan").write_text("partial", encoding="utf-8")
    restored = rehydrate_factor_factory_snapshot_payload(
        lake,
        queue,
        created.snapshot_id,
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
    )
    assert restored.manifest_sha256 == manifest.manifest_sha256
    assert restored.signature == signature
    assert not (snapshot_root / "FILES_RELEASED.json").exists()
    assert (snapshot_root / "manifest.json").read_text("utf-8") == manifest_text
    assert not crash_partial.exists()
    verify_factor_factory_snapshot_manifest(
        restored,
        final_root=snapshot_root,
        public_key=key.public_key(),
    )
    audit = (queue / "audit" / "factor_factory_snapshot.jsonl").read_text("utf-8")
    assert "snapshot_rehydrate_started" in audit
    assert "snapshot_rehydrate_completed" in audit

    assert release_snapshot_payload(queue, created.snapshot_id, reason="concurrent_rehydrate_test")
    original_materialize = factor_factory_snapshot_module._materialize_preflight
    materialize_count = 0
    count_lock = threading.Lock()
    materialize_started = threading.Event()
    allow_materialize = threading.Event()

    def counted_materialize(*args: object, **kwargs: object) -> object:
        nonlocal materialize_count
        with count_lock:
            materialize_count += 1
            current_count = materialize_count
        if current_count == 1:
            materialize_started.set()
            assert allow_materialize.wait(timeout=30)
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(
        factor_factory_snapshot_module,
        "_materialize_preflight",
        counted_materialize,
    )
    errors: list[Exception] = []

    def run_rehydrate() -> None:
        try:
            rehydrate_factor_factory_snapshot_payload(
                lake,
                queue,
                created.snapshot_id,
                signing_key=key,
                signature_key_id=TASK_KEY_ID,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=run_rehydrate) for _ in range(2)]
    threads[0].start()
    assert materialize_started.wait(timeout=30)
    assert not release_snapshot_payload(
        queue, created.snapshot_id, reason="must_not_release_during_rehydrate"
    )
    threads[1].start()
    allow_materialize.set()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors
    assert materialize_count == 1
    assert not list((queue / "snapshots").glob(".rehydrate.*.partial"))


def test_rehydrate_rejects_changed_source_and_stale_partial_cleanup(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_bars(lake, count=24)
    _write_costs(lake)
    publish_features(lake)
    key = Ed25519PrivateKey.generate()
    created = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert created.task is not None
    os.replace(
        queue / "pending" / created.task.task_id,
        queue / "completed" / created.task.task_id,
    )
    assert release_snapshot_payload(queue, created.snapshot_id, reason="source_change_test")
    _write_costs(lake, days=("2026-05-19", "2026-05-21"))
    with pytest.raises(RuntimeError, match="snapshot_rehydrate_identity_mismatch"):
        rehydrate_factor_factory_snapshot_payload(
            lake,
            queue,
            created.snapshot_id,
            signing_key=key,
            signature_key_id=TASK_KEY_ID,
        )
    audit_path = queue / "audit" / "factor_factory_snapshot.jsonl"
    assert "snapshot_rehydrate_failed" in audit_path.read_text("utf-8")

    stale = queue / "snapshots" / f".rehydrate.{created.snapshot_id}.stale.partial"
    stale.mkdir()
    (stale / "REHYDRATE.json").write_text(
        json.dumps({"snapshot_id": created.snapshot_id}), encoding="utf-8"
    )
    removed = cleanup_stale_factor_factory_rehydrate_partials(
        queue,
        stale_after_seconds=0,
    )
    assert stale.name in removed
    assert not stale.exists()
    assert "snapshot_rehydrate_partial_cleaned" in (
        queue / "audit" / "factor_factory_snapshot.jsonl"
    ).read_text("utf-8")


def test_worker_rejects_uncompressed_input_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_bars(lake, count=24)
    _write_costs(lake)
    publish_features(lake)
    created = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 20),
        signing_key=Ed25519PrivateKey.generate(),
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert created.task is not None
    manifest = FactorFactorySnapshotManifest.model_validate_json(
        (queue / "snapshots" / created.snapshot_id / "manifest.json").read_text("utf-8")
    )

    def unexpected_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("input gate must run before reading Parquet")

    monkeypatch.setattr(
        factor_factory_worker_module,
        "read_parquet_dataset",
        unexpected_read,
    )
    with pytest.raises(ValueError, match="input_uncompressed_size_limit_exceeded"):
        compute_factor_factory_result(
            queue / "snapshots" / created.snapshot_id,
            manifest,
            created.task,
            max_input_uncompressed_bytes=1,
        )

    legacy = manifest.model_copy(
        update={
            "schema_version": "quant_lab_factor_factory_snapshot.v1",
            "estimated_uncompressed_bytes": 0,
        }
    )
    monkeypatch.setattr(
        factor_factory_worker_module,
        "_parquet_uncompressed_bytes",
        lambda _path: 2,
    )
    with pytest.raises(ValueError, match="input_uncompressed_size_limit_exceeded"):
        compute_factor_factory_result(
            queue / "snapshots" / created.snapshot_id,
            legacy,
            created.task,
            max_input_uncompressed_bytes=1,
        )


def test_worker_gate_skips_factor_factory_before_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[str] = []

    def fake_ssh(_config: object, command: str, *, check: bool = True) -> object:
        commands.append(command)
        return subprocess.CompletedProcess([], 44, "", "")

    monkeypatch.setattr(runner_module, "_ssh", fake_ssh)
    config = SimpleNamespace(
        cloud_queue_root="/queue",
        worker_id="nas-research-worker-01",
        factor_factory_enabled=False,
    )
    assert runner_module.claim_next_task(config) is None
    assert "allow_factor_factory=0" in commands[0]
    assert '"factor_factory"' in commands[0]


def test_legacy_local_factor_factory_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QUANT_LAB_LOCAL_FACTOR_FACTORY_ENABLED", raising=False)
    result = CLI_RUNNER.invoke(
        app,
        ["build-factor-factory", "--lake-root", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code != 0
    assert "local Factor Factory fallback disabled" in result.output


def _write_bars(lake: Path, *, count: int) -> None:
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]):
        for index in range(count):
            close = 100.0 + index * (0.05 + symbol_index * 0.02) + symbol_index * 10.0
            rows.append(
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": start + timedelta(hours=index),
                    "open": close - 0.1,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 10.0 + index,
                    "quote_volume": close * (10.0 + index),
                    "source": "test",
                    "ingest_ts": start + timedelta(hours=index, minutes=1),
                    "is_closed": True,
                }
            )
    write_market_bars(lake, rows)


def _write_factor_factory_binding(
    lake: Path,
    *,
    generation_id: str,
    digest_character: str,
) -> None:
    datasets = {
        "factor_definition",
        "factor_value",
        "factor_evidence",
        "factor_candidate",
        "factor_correlation_daily",
    }
    pointer = {
        "schema_version": "factor_factory_generation.v1",
        "generation_id": generation_id,
        "generation_digest": digest_character * 64,
        "task_id": f"task-{generation_id}",
        "snapshot_id": f"snapshot-{generation_id}",
        "quant_lab_commit": COMMIT,
        "factor_plan_digest": "a" * 64,
        "source_input_digest": "b" * 64,
        "cost_input_digest": "c" * 64,
        "feature_set": "other",
        "feature_version": "v0.1",
        "factor_version": "v0.1",
        "timeframe": "1H",
        "as_of_date": "2026-05-19",
        "row_counts": {name: 0 for name in datasets},
        "dataset_hashes": {name: "d" * 64 for name in datasets},
        "published_at": datetime(2026, 5, 19, tzinfo=UTC).isoformat(),
        "diagnostic_only": True,
        "research_only": True,
        "live_order_effect": "none_read_only_research",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
    }
    path = lake / FACTOR_FACTORY_GENERATION_POINTER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pointer), encoding="utf-8")


def _write_verified_factor_factory_generation(
    lake: Path,
    *,
    identity_payload: dict[str, object],
    snapshot_id: str,
    generation_id: str,
) -> None:
    rows_by_dataset = {
        "factor_definition": {"factor_id": "factor-one", "factor_version": "v0.1"},
        "factor_value": {
            "factor_id": "factor-one",
            "factor_version": "v0.1",
            "symbol": "BTC-USDT",
            "timeframe": "1H",
            "ts": "2026-05-20T00:00:00+00:00",
        },
        "factor_evidence": {
            "as_of_date": "2026-05-20",
            "factor_id": "factor-one",
            "factor_version": "v0.1",
            "timeframe": "1H",
            "horizon_bars": 4,
            "decision_delay_bars": 1,
        },
        "factor_candidate": {
            "as_of_date": "2026-05-20",
            "factor_id": "factor-one",
            "factor_version": "v0.1",
            "timeframe": "1H",
            "candidate_state": "KEEP_SHADOW",
            "manual_review_required": True,
            "source": "factors.factory.v0.1",
        },
        "factor_correlation_daily": {
            "as_of_date": "2026-05-20",
            "factor_id_left": "factor-one",
            "factor_id_right": "factor-one",
            "factor_version": "v0.1",
            "timeframe": "1H",
        },
    }
    for dataset_name, target in FACTOR_FACTORY_DATASETS.items():
        dataset_root = lake / target
        dataset_root.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([rows_by_dataset[dataset_name]]).write_parquet(
            dataset_root / "data.parquet"
        )

    generation_digest = "e" * 64
    pointer = {
        **identity_payload,
        "schema_version": "factor_factory_generation.v1",
        "generation_id": generation_id,
        "generation_digest": generation_digest,
        "snapshot_id": snapshot_id,
        "as_of_date": "2026-05-20",
        "row_counts": {name: 1 for name in FACTOR_FACTORY_DATASETS},
        "dataset_hashes": {
            name: _dataset_digest(lake / target)
            for name, target in FACTOR_FACTORY_DATASETS.items()
        },
        "diagnostic_only": True,
        "research_only": True,
        "live_order_effect": "none_read_only_research",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "manual_review_required": True,
        "published_at": datetime.now(UTC).isoformat(),
    }
    pointer_path = lake / FACTOR_FACTORY_GENERATION_POINTER
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    sidecar = {
        "generation_id": generation_id,
        "generation_digest": generation_digest,
    }
    for target in FACTOR_FACTORY_DATASETS.values():
        (lake / target / "_factor_factory_generation.json").write_text(
            json.dumps(sidecar),
            encoding="utf-8",
        )


def _write_costs(lake: Path, *, days: tuple[str, ...] = ("2026-05-10",)) -> None:
    rows = [
        {
            "day": day,
            "symbol": symbol,
            "total_cost_bps_p50": float(day_index + 1),
            "total_cost_bps_p75": float(day_index + 1),
            "total_cost_bps_p90": float(day_index + 1),
            "cost_model_version": f"costs-test-{day_index}",
            "source": "public_spread_proxy",
        }
        for day_index, day in enumerate(days)
        for symbol in ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]
    ]
    write_parquet_dataset(pl.DataFrame(rows), lake / "gold" / "cost_bucket_daily")
