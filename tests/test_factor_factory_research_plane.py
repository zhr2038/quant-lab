from __future__ import annotations

import os
import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
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
from quant_lab.research_plane.contracts import FactorFactorySnapshotManifest, FactorFactoryTask
from quant_lab.research_plane.factor_factory_publish import (
    FACTOR_FACTORY_DATASETS,
    FACTOR_FACTORY_GENERATION_POINTER,
    FACTOR_FACTORY_PRIMARY_KEYS,
    publish_factor_factory_generation,
    verify_factor_factory_generation,
)
from quant_lab.research_plane.factor_factory_result import (
    validate_factor_factory_result_bundle,
)
from quant_lab.research_plane.factor_research_publish import (
    FACTOR_RESEARCH_GENERATION_POINTER,
    verify_factor_research_generation,
)
from quant_lab.research_plane.importer import import_entry_quality_history_result
from quant_lab.research_plane.queue import create_factor_factory_task
from quant_lab.research_plane.result import validate_research_task_snapshot
from quant_lab.research_plane.signatures import verify_payload
from quant_lab.research_plane.status import research_plane_status
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
    successor, _ = create_factor_factory_task(
        lake,
        queue,
        as_of_date=date(2026, 5, 21),
        horizon_bars=(4, 8),
        min_samples=20,
        signing_key=task_key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    successor_snapshot = FactorFactorySnapshotManifest.model_validate_json(
        (
            queue / "snapshots" / successor.snapshot_id / "manifest.json"
        ).read_text("utf-8")
    )
    assert successor_snapshot.previous_generation_manifest is not None
    assert successor_snapshot.previous_generation_manifest.generation_id == manifest.generation_id


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
    assert all(not (lake / path).exists() for path in FACTOR_FACTORY_DATASETS.values())


def test_factor_factory_nas_round_trip_matches_legacy_full_fixture(tmp_path: Path) -> None:
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
