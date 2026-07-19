from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.factor_research.registry import (
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
)
from quant_lab.research_plane.contracts import FactorResearchSnapshotManifest
from quant_lab.research_plane.queue import create_factor_research_task
from quant_lab.research_plane.signatures import verify_payload
from quant_lab.research_plane.snapshot import verify_factor_research_snapshot_manifest

COMMIT = "a" * 40
TASK_KEY_ID = "cloud-research-v1"
BUNDLE_ID = "v5-bundle-sha256:" + "b" * 64


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
            "day": [date(2024, 7, 18), date(2026, 7, 16)],
            "symbol": ["SOL-USDT", "SOL-USDT"],
            "total_cost_bps_p75": [12.0, 10.0],
            "cost_source": ["actual", "actual"],
        }
    ).write_parquet(costs / "part-cost.parquet")

    if include_regime:
        regime = root / "gold" / "market_regime_daily"
        regime.mkdir(parents=True)
        pl.DataFrame(
            {
                "day": [date(2024, 7, 18), date(2026, 7, 16)],
                "symbol": ["SOL-USDT", "SOL-USDT"],
                "current_regime": ["SIDEWAYS", "TREND_UP"],
            }
        ).write_parquet(regime / "part-regime.parquet")


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
    assert first_status.task_id == second_status.task_id
    assert first.test_count == 8
    assert len(list((queue / "pending").iterdir())) == 1
    verify_payload(first, first.signature, key.public_key())

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

    registry = read_parquet_dataset(lake / RESEARCH_HYPOTHESIS_REGISTRY_DATASET)
    ledger = read_parquet_dataset(lake / RESEARCH_TRIAL_LEDGER_DATASET)
    assert registry.height == 4
    assert ledger.height == 8
    assert ledger.get_column("nas_task_id").unique().to_list() == [first.task_id]


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
