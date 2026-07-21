from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.data.file_index import build_lake_file_index
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.factor_research.registry import (
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
    default_hypothesis_registry,
)
from quant_lab.research_plane.contracts import FactorResearchSnapshotManifest
from quant_lab.research_plane.factor_research_publish import (
    FACTOR_RESEARCH_GENERATION_POINTER,
    FACTOR_RESEARCH_GENERATION_SCHEMA,
    _merge_managed_factor_rows,
    current_factor_research_generation_binding,
)
from quant_lab.research_plane.importer import import_entry_quality_history_result
from quant_lab.research_plane.queue import create_factor_research_task
from quant_lab.research_plane.result import validate_factor_research_result_bundle
from quant_lab.research_plane.signatures import verify_payload
from quant_lab.research_plane.snapshot import (
    _select_factor_research_inputs,
    verify_factor_research_snapshot_manifest,
)
from quant_lab.research_plane.status import research_plane_status
from quant_lab.research_worker.factor_research import (
    _point_in_time_market_universe,
    compute_factor_research_result,
)
from quant_lab.research_worker.result_writer import write_factor_research_result_bundle

COMMIT = "a" * 40
NEXT_COMMIT = "c" * 40
TASK_KEY_ID = "cloud-research-v1"
BUNDLE_ID = "v5-bundle-sha256:" + "b" * 64


def test_factor_research_correlation_replaces_same_day_research_rows() -> None:
    schema = {
        "as_of_date": pl.Utf8,
        "factor_id_left": pl.Utf8,
        "factor_id_right": pl.Utf8,
        "research_only": pl.Boolean,
        "correlation": pl.Float64,
    }
    existing = pl.DataFrame(
        [
            {
                "as_of_date": "2026-07-19",
                "factor_id_left": "legacy-a",
                "factor_id_right": "legacy-b",
                "research_only": None,
                "correlation": 0.1,
            },
            {
                "as_of_date": "2026-07-19",
                "factor_id_left": "research-a",
                "factor_id_right": "research-b",
                "research_only": True,
                "correlation": 0.2,
            },
        ]
    )
    incoming = pl.DataFrame(
        [
            {
                "as_of_date": "2026-07-19",
                "factor_id_left": "research-a",
                "factor_id_right": "research-b",
                "research_only": True,
                "correlation": 0.3,
            }
        ]
    )

    merged = _merge_managed_factor_rows(
        existing,
        incoming,
        schema=schema,
        primary_keys=("as_of_date", "factor_id_left", "factor_id_right"),
        hypothesis_ids={"unused-for-correlation"},
        as_of_date="2026-07-19",
    )

    assert merged.height == 2
    assert merged.filter(pl.col("research_only").fill_null(False))["correlation"].item() == 0.3


def _write_source_data(root: Path, *, include_regime: bool = True) -> None:
    market = root / "silver" / "market_bar"
    market.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["SOL-USDT", "SOL-USDT", "SOL-USDT", "SOL-USDT"],
            "timeframe": ["1h", "4h", "1h", "1h"],
            "ts": [
                datetime(2024, 7, 17, 23, tzinfo=UTC),
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2026, 7, 16, 23, tzinfo=UTC),
                datetime(2026, 7, 17, 0, tzinfo=UTC),
            ],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
            "quote_volume": [100_000.0, 100_000.0, 100_000.0, 100_000.0],
            "is_closed": [True, True, True, False],
        }
    ).write_parquet(market / "part-market.parquet")

    quality = root / "gold" / "expanded_universe_quality"
    quality.mkdir(parents=True)
    pl.DataFrame(
        {
            "as_of_date": [date(2024, 7, 18), date(2026, 7, 16)],
            "symbol": ["SOL-USDT", "SOL-USDT"],
            "quality_score": [0.8, 0.9],
        }
    ).write_parquet(quality / "part-quality.parquet")

    costs = root / "gold" / "cost_bucket_daily"
    costs.mkdir(parents=True)
    pl.DataFrame(
        {
            "day": [date(2024, 7, 18), date(2026, 7, 15)],
            "symbol": ["SOL-USDT", "SOL-USDT"],
            "total_cost_bps_p75": [12.0, 10.0],
            "cost_source": ["actual", "actual"],
        }
    ).write_parquet(costs / "part-cost.parquet")

    spreads = root / "silver" / "orderbook_spread_1m"
    spreads.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["SOL-USDT"],
            "channel": ["books5"],
            "minute_ts": [datetime(2026, 7, 15, 23, 0, tzinfo=UTC)],
            "spread_bps": [8.0],
        }
    ).write_parquet(spreads / "part-spread.parquet")

    if include_regime:
        regime = root / "gold" / "market_regime_daily"
        regime.mkdir(parents=True)
        pl.DataFrame(
            {
                "as_of_date": [date(2024, 7, 18), date(2026, 7, 16)],
                "current_regime": ["SIDEWAYS", "TREND_UP"],
            }
        ).write_parquet(regime / "part-regime.parquet")


def test_factor_research_ignores_stale_index_rows_outside_executable_requirements(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    _write_source_data(lake)
    build_lake_file_index(lake, ["gold/market_regime_daily"])
    (lake / "gold/market_regime_daily/part-regime.parquet").unlink()

    selected, _windows, _symbols = _select_factor_research_inputs(
        lake,
        start_date=date(2024, 7, 18),
        end_date=date(2026, 7, 17),
        hypotheses=[default_hypothesis_registry()[0]],
    )

    datasets = {dataset for dataset, _path, _min_ts, _max_ts in selected}
    assert "gold/market_regime_daily" not in datasets
    assert datasets == {
        "gold/cost_bucket_daily",
        "silver/market_bar",
        "silver/orderbook_spread_1m",
    }


def test_factor_research_task_is_content_addressed_signed_and_idempotent(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_source_data(lake)
    key = Ed25519PrivateKey.generate()
    first, first_status = create_factor_research_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 19),
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )
    second, second_status = create_factor_research_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 19),
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )
    assert first.task_id == second.task_id
    assert first.snapshot_id == second.snapshot_id
    assert first.end_date == date(2026, 7, 15)
    assert first_status.task_id == second_status.task_id
    assert first.test_count == 8
    assert len(list((queue / "pending").iterdir())) == 1
    verify_payload(first, first.signature, key.public_key())
    assert research_plane_status(queue)["tasks"]["factor_research"]["state"] == "pending"

    snapshot_root = queue / "snapshots" / first.snapshot_id
    manifest = FactorResearchSnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    verify_factor_research_snapshot_manifest(manifest, final_root=snapshot_root)
    assert manifest.hypothesis_ids == (
        "defensive.low_vol_decomposition",
        "timing.market_breadth",
    )
    assert manifest.trial_ids == first.trial_ids
    assert manifest.source_input_digest == first.source_input_digest
    assert "silver/orderbook_spread_1m" in set(manifest.datasets)

    snapshot_costs = read_parquet_dataset(
        snapshot_root / "files" / "gold" / "cost_bucket_daily"
    )
    assert snapshot_costs.get_column("day").to_list() == [date(2026, 7, 15)]
    snapshot_spreads = read_parquet_dataset(
        snapshot_root / "files" / "silver" / "orderbook_spread_1m"
    )
    assert snapshot_spreads.height == 1

    assert read_parquet_dataset(lake / RESEARCH_HYPOTHESIS_REGISTRY_DATASET).is_empty()
    assert read_parquet_dataset(lake / RESEARCH_TRIAL_LEDGER_DATASET).is_empty()

    compute = compute_factor_research_result(snapshot_root, manifest, first)
    assert compute.definitions.height == 4
    assert compute.evidence.height == 8
    assert "research_only" in compute.correlations.columns
    assert "live_order_effect" in compute.correlations.columns
    assert compute.candidates.height == 4
    assert "PAPER_CANDIDATE" not in set(compute.candidates.get_column("candidate_state").to_list())
    assert compute.anti_leakage["status"] == "PASS"

    worker_key = Ed25519PrivateKey.generate()
    result_root, result_manifest, receipt = write_factor_research_result_bundle(
        tmp_path / "worker-results",
        task=first,
        snapshot=manifest,
        compute=compute,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        worker_key_id="nas-research-v1",
        worker_signing_key=worker_key,
        claimed_at=datetime(2026, 7, 19, 1, 0, tzinfo=UTC),
        input_bytes=manifest.total_input_bytes,
        cache_hit_bytes=manifest.total_input_bytes,
        downloaded_bytes=0,
        peak_rss_bytes=256 * 1024**2,
        compute_duration_seconds=1.0,
        max_result_bytes=256 * 1024**2,
    )
    validated = validate_factor_research_result_bundle(
        result_root,
        manifest=result_manifest,
        receipt=receipt,
        task=first,
        snapshot=manifest,
        worker_public_key=worker_key.public_key(),
        expected_worker_key_id="nas-research-v1",
        max_result_bytes=256 * 1024**2,
        snapshot_root=snapshot_root,
    )
    assert set(validated.output_paths) == {
        "factor_definition",
        "factor_value",
        "factor_evidence",
        "factor_attribution",
        "factor_portfolio_validation",
        "factor_correlation_daily",
        "factor_candidate",
    }
    assert result_manifest.research_only is True
    assert result_manifest.live_order_effect == "none"
    assert receipt.output_rows == sum(item.row_count for item in result_manifest.outputs)

    # Old bootstrap generations could leave one all-null schema placeholder.
    # A real generation must remove it instead of carrying it forward forever.
    for dataset_name in ("factor_attribution", "factor_portfolio_validation"):
        dataset_root = lake / "gold" / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "as_of_date": [None],
                "trial_id": [None],
                "factor_id": [None],
            }
        ).write_parquet(dataset_root / "legacy-null-placeholder.parquet")

    os.replace(queue / "pending" / first.task_id, queue / "running" / first.task_id)
    shutil.copytree(result_root, queue / "results" / "inbox" / first.task_id)
    imported = import_entry_quality_history_result(
        lake,
        queue,
        first.task_id,
        task_public_key=key.public_key(),
        worker_public_key=worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id="nas-research-v1",
        expected_quant_lab_commit=COMMIT,
    )
    assert imported.state == "completed"
    assert imported.idempotent is False
    pointer = json.loads((lake / FACTOR_RESEARCH_GENERATION_POINTER).read_text("utf-8"))
    assert pointer["schema_version"] == FACTOR_RESEARCH_GENERATION_SCHEMA
    assert pointer["generation_id"] == result_manifest.generation_id
    assert pointer["research_only"] is True
    assert pointer["live_order_effect"] == "none"
    assert pointer["automatic_promotion"] is False
    assert pointer["max_live_notional_usdt"] == 0
    for dataset_name in ("factor_attribution", "factor_portfolio_validation"):
        published = read_parquet_dataset(lake / "gold" / dataset_name)
        assert published.filter(
            pl.col("as_of_date").is_null() | pl.col("trial_id").is_null()
        ).is_empty()
    completed_ledger = read_parquet_dataset(lake / RESEARCH_TRIAL_LEDGER_DATASET)
    assert set(completed_ledger.get_column("status").to_list()) == {"COMPLETED"}
    completed_registry = read_parquet_dataset(lake / RESEARCH_HYPOTHESIS_REGISTRY_DATASET)
    evaluated = completed_registry.filter(
        pl.col("hypothesis_id").is_in(first.hypothesis_ids)
    )
    assert set(evaluated.get_column("status").to_list()) == {"APPROVED_FOR_RESEARCH"}
    assert (queue / "completed" / first.task_id).is_dir()
    assert (queue / "results" / "imported" / first.task_id).is_dir()
    completed_status = research_plane_status(queue)
    assert completed_status["tasks"]["factor_research"]["state"] == "completed"
    assert completed_status["tasks"]["factor_research"]["task"]["task_id"] == first.task_id

    next_task, _ = create_factor_research_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 19),
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=NEXT_COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )
    assert next_task.task_id != first.task_id
    historical_ledger = read_parquet_dataset(lake / RESEARCH_TRIAL_LEDGER_DATASET)
    assert historical_ledger.height == first.test_count
    pointer_before_next_import = (
        lake / FACTOR_RESEARCH_GENERATION_POINTER
    ).read_bytes()
    assert current_factor_research_generation_binding(
        lake,
        alpha_as_of_date=date(2026, 7, 19),
    )["factor_generation_id"] == result_manifest.generation_id
    next_snapshot_root = queue / "snapshots" / next_task.snapshot_id
    next_manifest = FactorResearchSnapshotManifest.model_validate_json(
        (next_snapshot_root / "manifest.json").read_text("utf-8")
    )
    next_compute = compute_factor_research_result(
        next_snapshot_root, next_manifest, next_task
    )
    next_result_root, _, _ = write_factor_research_result_bundle(
        tmp_path / "worker-results-next",
        task=next_task,
        snapshot=next_manifest,
        compute=next_compute,
        worker_id="nas-research-worker-01",
        worker_commit=NEXT_COMMIT,
        worker_key_id="nas-research-v1",
        worker_signing_key=worker_key,
        claimed_at=datetime(2026, 7, 19, 2, 0, tzinfo=UTC),
        input_bytes=next_manifest.total_input_bytes,
        cache_hit_bytes=next_manifest.total_input_bytes,
        downloaded_bytes=0,
        peak_rss_bytes=256 * 1024**2,
        compute_duration_seconds=1.0,
        max_result_bytes=256 * 1024**2,
    )
    os.replace(queue / "pending" / next_task.task_id, queue / "running" / next_task.task_id)
    shutil.copytree(next_result_root, queue / "results" / "inbox" / next_task.task_id)
    next_imported = import_entry_quality_history_result(
        lake,
        queue,
        next_task.task_id,
        task_public_key=key.public_key(),
        worker_public_key=worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id="nas-research-v1",
        expected_quant_lab_commit=NEXT_COMMIT,
    )
    assert next_imported.state == "completed"
    assert next_imported.idempotent is False
    assert (lake / FACTOR_RESEARCH_GENERATION_POINTER).read_bytes() != pointer_before_next_import
    final_ledger = read_parquet_dataset(lake / RESEARCH_TRIAL_LEDGER_DATASET)
    assert final_ledger.height == first.test_count + next_task.test_count
    assert set(final_ledger.get_column("status").to_list()) == {"COMPLETED"}
    final_registry = read_parquet_dataset(lake / RESEARCH_HYPOTHESIS_REGISTRY_DATASET)
    final_evaluated = final_registry.filter(
        pl.col("hypothesis_id").is_in(next_task.hypothesis_ids)
    )
    assert set(final_evaluated.get_column("status").to_list()) == {
        "APPROVED_FOR_RESEARCH"
    }
    status = research_plane_status(queue)
    assert status["tasks"]["factor_research"]["state"] == "completed"
    assert status["tasks"]["factor_research"]["task"]["task_id"] == next_task.task_id


def test_factor_research_snapshot_projects_only_closed_one_hour_bars(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_source_data(lake)
    task, _ = create_factor_research_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 19),
        signing_key=Ed25519PrivateKey.generate(),
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )
    market_root = queue / "snapshots" / task.snapshot_id / "files" / "silver" / "market_bar"
    market = pl.concat([pl.read_parquet(path) for path in market_root.glob("*.parquet")])
    assert market.get_column("timeframe").unique().to_list() == ["1h"]
    assert market.get_column("is_closed").unique().to_list() == [True]
    assert market.height == 2


def test_factor_research_request_fails_closed_when_required_source_is_missing(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    _write_source_data(lake, include_regime=False)
    with pytest.raises(ValueError, match="required_dataset_empty:gold/market_regime_daily"):
        create_factor_research_task(
            lake,
            tmp_path / "queue",
            as_of_date=date(2026, 7, 19),
            signing_key=Ed25519PrivateKey.generate(),
            signature_key_id=TASK_KEY_ID,
            quant_lab_commit=COMMIT,
            selected_v5_bundle_id=BUNDLE_ID,
        )


def test_factor_research_request_requires_point_in_time_spread_history(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    _write_source_data(lake)
    shutil.rmtree(lake / "silver" / "orderbook_spread_1m")

    with pytest.raises(
        ValueError,
        match="required_dataset_empty:silver/orderbook_spread_1m",
    ):
        create_factor_research_task(
            lake,
            tmp_path / "queue",
            as_of_date=date(2026, 7, 19),
            signing_key=Ed25519PrivateKey.generate(),
            signature_key_id=TASK_KEY_ID,
            quant_lab_commit=COMMIT,
            selected_v5_bundle_id=BUNDLE_ID,
        )


def test_factor_research_does_not_use_current_quality_snapshot_for_historical_universe(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _write_source_data(lake)
    quality_path = lake / "gold" / "expanded_universe_quality" / "part-quality.parquet"
    pl.DataFrame(
        {
            "as_of_date": [date(2026, 7, 19)],
            "symbol": ["SOL-USDT"],
            "quality_score": [0.9],
        }
    ).write_parquet(quality_path)

    task, _ = create_factor_research_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 19),
        signing_key=Ed25519PrivateKey.generate(),
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )

    manifest = FactorResearchSnapshotManifest.model_validate_json(
        (queue / "snapshots" / task.snapshot_id / "manifest.json").read_text("utf-8")
    )
    assert all(item.dataset_name != "gold/expanded_universe_quality" for item in manifest.files)
    registry = read_parquet_dataset(
        queue
        / "snapshots"
        / task.snapshot_id
        / "files"
        / RESEARCH_HYPOTHESIS_REGISTRY_DATASET
    )
    active = registry.filter(pl.col("status") == "APPROVED_FOR_RESEARCH")
    assert set(active.get_column("hypothesis_version").to_list()) == {2}
    assert all(
        '"dynamic_source_dataset":"silver/market_bar"' in value
        for value in active.get_column("universe_definition_json").to_list()
    )


def test_point_in_time_market_universe_uses_only_previous_complete_day() -> None:
    bars = pl.DataFrame(
        {
            "symbol": ["SOL-USDT"] * 24 + ["ETH-USDT"] * 17,
            "timeframe": ["1h"] * 41,
            "ts": [
                datetime(2026, 7, 15, hour, tzinfo=UTC) for hour in range(24)
            ]
            + [datetime(2026, 7, 15, hour, tzinfo=UTC) for hour in range(17)],
            "close": [100.0] * 41,
            "quote_volume": [1_000.0] * 41,
            "is_closed": [True] * 41,
        }
    )

    universe = _point_in_time_market_universe(bars)

    assert universe.height == 1
    assert universe.item(0, "symbol") == "SOL-USDT"
    assert universe.item(0, "source_day") == date(2026, 7, 16)
