from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.paper_promotion import (
    build_and_publish_paper_strategy_pipeline,
    build_paper_strategy_pipeline_frames,
)


def test_paper_strategy_pipeline_blocks_unacked_proposal() -> None:
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame(
            [
                {
                    "proposal_id": "SOL_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "SOL-USDT",
                    "recommended_mode": "paper",
                    "created_at": "2026-06-30T00:00:00Z",
                }
            ]
        ),
        proposal_ack=pl.DataFrame(),
        runs=pl.DataFrame(),
        daily=pl.DataFrame(),
        created_at=datetime(2026, 6, 30, tzinfo=UTC),
    )

    registry = frames["paper_strategy_registry"].to_dicts()[0]
    gate = frames["paper_strategy_promotion_gate"].to_dicts()[0]

    assert registry["status"] == "PROPOSED_AWAITING_ACK"
    assert registry["rules_locked"] is False
    assert gate["paper_ready"] is False
    assert gate["paper_tracker_created"] is False
    assert gate["paper_tracker_effective"] is False
    assert gate["paper_tracker_status"] == "MISSING"
    assert "proposal_not_acked" in json.loads(gate["block_reason"])


def test_paper_strategy_pipeline_marks_unacked_tracker_evidence_not_effective() -> None:
    proposal_id = "BNB_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
    tracker_id = "BNB_F3_DOMINANT_ENTRY_PAPER_V1"
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame(),
        proposal_ack=pl.DataFrame(),
        runs=pl.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
                    "paper_tracker_id": tracker_id,
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "BNB-USDT",
                    "as_of_date": "2026-06-30",
                    "paper_pnl_bps": 12.0,
                    "would_enter": True,
                    "would_exit": True,
                    "arrival_mid": 600.0,
                }
            ]
        ),
        daily=pl.DataFrame(),
        created_at=datetime(2026, 6, 30, tzinfo=UTC),
    )

    registry = frames["paper_strategy_registry"].to_dicts()[0]
    gate = frames["paper_strategy_promotion_gate"].to_dicts()[0]
    block_reasons = json.loads(gate["block_reason"])

    assert registry["status"] == "PROPOSED_AWAITING_ACK"
    assert gate["paper_tracker_created"] is True
    assert gate["paper_tracker_effective"] is False
    assert gate["paper_tracker_status"] == "AWAITING_ACK"
    assert "proposal_not_acked" in block_reasons
    assert "paper_tracker_not_effective_without_ack" in block_reasons


def test_paper_strategy_pipeline_marks_ready_only_after_ack_and_future_paper_evidence() -> None:
    proposal_id = "SOL_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
    tracker_id = proposal_id
    run_rows = []
    start = date(2026, 6, 1)
    for index in range(20):
        run_rows.append(
            {
                "proposal_id": proposal_id,
                "paper_tracker_id": tracker_id,
                "strategy_candidate": "v5.f3_dominant_entry",
                "symbol": "SOL-USDT",
                "as_of_date": (start + timedelta(days=index % 14)).isoformat(),
                "paper_pnl_bps": 12.0 + index,
                "would_enter": True,
                "would_exit": True,
                "arrival_mid": 70.0,
            }
        )

    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "SOL-USDT",
                    "recommended_mode": "paper",
                    "created_at": "2026-06-01T00:00:00Z",
                }
            ]
        ),
        proposal_ack=pl.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
                    "paper_tracker_id": tracker_id,
                    "accepted": "true",
                    "recommended_mode": "paper",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "live_order_effect": "paper_only_no_live_order",
                    "bundle_sha256": "abc123",
                    "ingest_ts": "2026-06-01T00:01:00Z",
                }
            ]
        ),
        runs=pl.DataFrame(run_rows),
        daily=pl.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
                    "paper_tracker_id": tracker_id,
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "SOL-USDT",
                    "paper_days": 14,
                    "entry_day_count": 7,
                    "paper_pnl_observed_count": 20,
                    "avg_paper_pnl_bps": 21.5,
                    "arrival_mid_coverage": 0.95,
                    "spread_observation_coverage": 0.95,
                    "cost_source_mix": '[{"cost_source":"mixed_actual_proxy","count":20}]',
                    "live_block_reason": "[]",
                    "live_eligible": True,
                    "created_at": "2026-06-14T00:00:00Z",
                }
            ]
        ),
        created_at=datetime(2026, 6, 30, tzinfo=UTC),
    )

    registry = frames["paper_strategy_registry"].to_dicts()[0]
    gate = frames["paper_strategy_promotion_gate"].to_dicts()[0]

    assert registry["status"] == "PAPER_REVIEW"
    assert registry["rules_locked"] is True
    assert gate["paper_ready"] is True
    assert gate["lifecycle_state"] == "PAPER_READY"
    assert gate["paper_tracker_created"] is True
    assert gate["paper_tracker_effective"] is True
    assert gate["paper_tracker_status"] == "EFFECTIVE"
    assert json.loads(gate["block_reason"]) == []


def test_build_and_publish_paper_strategy_pipeline_writes_gold_outputs(tmp_path) -> None:
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": "BNB_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "paper_tracker_id": "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
                    "accepted": "true",
                    "recommended_mode": "paper",
                    "symbol": "BNB-USDT",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "live_order_effect": "paper_only_no_live_order",
                    "bundle_sha256": "sha",
                    "ingest_ts": "2026-06-30T00:00:00Z",
                }
            ]
        ),
        lake / "silver" / "v5_paper_strategy_proposal_ack",
    )
    write_parquet_dataset(pl.DataFrame(), lake / "gold" / "strategy_opportunity_advisory")
    write_parquet_dataset(pl.DataFrame(), lake / "gold" / "paper_strategy_runs")
    write_parquet_dataset(pl.DataFrame(), lake / "gold" / "paper_strategy_daily")

    result = build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-06-30")

    assert result.paper_strategy_registry == 1
    registry = read_parquet_dataset(lake / "gold" / "paper_strategy_registry")
    gate = read_parquet_dataset(lake / "gold" / "paper_strategy_promotion_gate")
    assert registry.to_dicts()[0]["proposal_id"] == "BNB_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
    assert gate.to_dicts()[0]["paper_ready"] is False
    assert gate.to_dicts()[0]["paper_tracker_effective"] is True
    assert gate.to_dicts()[0]["paper_tracker_status"] == "EFFECTIVE"
