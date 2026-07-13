from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.paper_promotion import (
    build_and_publish_paper_strategy_pipeline,
    build_paper_strategy_pipeline_frames,
)


def _current_tracker(
    proposal_id: str,
    *,
    proposal_hash: str = "",
    strategy_id: str = "",
    symbol: str = "",
    created_at: str = "2026-07-11T01:05:00Z",
) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "proposal_hash": proposal_hash,
        "tracker_id": f"paper:{proposal_id}",
        "strategy_id": strategy_id or proposal_id,
        "symbol": symbol,
        "state": "WAITING_SIGNAL",
        "rules_locked": True,
        "paper_only": True,
        "live_order_effect": "none",
        "created_at": created_at,
        "updated_at": created_at,
    }


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

    assert registry["status"] == "CURRENT_PENDING_ACK"
    assert registry["rules_locked"] is False
    assert gate["paper_ready"] is False
    assert gate["paper_tracker_created"] is False
    assert gate["paper_tracker_effective"] is False
    assert gate["paper_tracker_status"] == "MISSING"
    assert "proposal_not_acked" in json.loads(gate["block_reason"])


def test_pipeline_uses_only_current_structured_proposal_and_clears_old_reject() -> None:
    current_id = "TAO_F3_F4_DEDUP_8H_PAPER:1.0.0:bbbbbbbbbbbb"
    old_id = "TAO_F3_F4_DEDUP_8H_PAPER:1.0.0:aaaaaaaaaaaa"
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame(
            [
                {
                    "proposal_id": current_id,
                    "proposal_hash": "b" * 64,
                    "strategy_id": "TAO_F3_F4_DEDUP_8H_PAPER",
                    "strategy_version": "1.0.0",
                    "strategy_candidate": "f3_f4_deduplicated_entry",
                    "symbol": "TAO-USDT",
                    "recommended_mode": "paper",
                    "entry_rule": '{"operator":"gt","field":"close","value":0}',
                    "exit_rule": '{"operator":"max_holding_bars","value":8}',
                }
            ]
        ),
        proposal_ack=pl.DataFrame(
            [
                {
                    "proposal_id": old_id,
                    "proposal_hash": "a" * 64,
                    "accepted": False,
                    "reject_reason": "no_supported_paper_tracker",
                    "symbol": "TAO-USDT",
                },
                {
                    "proposal_id": current_id,
                    "proposal_hash": "b" * 64,
                    "paper_tracker_id": f"paper:{current_id}",
                    "accepted": True,
                    "reject_reason": "",
                    "rules_locked": True,
                    "paper_only": True,
                    "live_order_effect": "none",
                    "strategy_version": "1.0.0",
                    "strategy_candidate": "f3_f4_deduplicated_entry",
                    "symbol": "TAO-USDT",
                    "accepted_at": "2026-07-11T01:00:00Z",
                },
            ]
        ),
        trackers_current=pl.DataFrame(
            [
                _current_tracker(
                    current_id,
                    proposal_hash="b" * 64,
                    strategy_id="TAO_F3_F4_DEDUP_8H_PAPER",
                    symbol="TAO-USDT",
                )
            ]
        ),
        daily=pl.DataFrame(
            [
                {
                    "proposal_id": old_id,
                    "paper_tracker_id": f"paper:{old_id}",
                    "paper_days": 1,
                },
                {
                    "proposal_id": current_id,
                    "paper_tracker_id": f"paper:{current_id}",
                    "paper_days": 1,
                    "arrival_mid_coverage": 1.0,
                },
            ]
        ),
        created_at=datetime(2026, 7, 11, 2, tzinfo=UTC),
    )

    registry = frames["paper_strategy_registry"].to_dicts()
    gate = frames["paper_strategy_promotion_gate"].to_dicts()
    assert len(registry) == 1
    assert len(gate) == 1
    assert registry[0]["proposal_id"] == current_id
    assert registry[0]["accepted"] is True
    assert registry[0]["reject_reason"] == ""
    assert registry[0]["lifecycle_state"] == "PAPER_TRACKER_ACTIVE"
    assert "v5_rejected" not in gate[0]["promotion_block_reasons"]


def test_pipeline_drops_same_id_ack_with_wrong_contract_hash() -> None:
    proposal_id = "TAO_F3_F4_DEDUP_8H_PAPER:1.0.0:bbbbbbbbbbbb"
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame(
            [
                {
                    "contract_version": "quant_lab.paper_strategy.v1",
                    "proposal_id": proposal_id,
                    "proposal_hash": "b" * 64,
                    "strategy_id": "TAO_F3_F4_DEDUP_8H_PAPER",
                    "strategy_version": "1.0.0",
                    "strategy_candidate": "f3_f4_deduplicated_entry",
                    "symbol": "TAO-USDT",
                    "entry_rule": '{"operator":"gt","field":"close","value":0}',
                    "exit_rule": '{"operator":"max_holding_bars","value":8}',
                }
            ]
        ),
        proposal_ack=pl.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
                    "proposal_hash": "a" * 64,
                    "accepted": False,
                    "reject_reason": "no_supported_paper_tracker",
                },
                {
                    "proposal_id": proposal_id,
                    "proposal_hash": "b" * 64,
                    "paper_tracker_id": f"paper:{proposal_id}",
                    "accepted": True,
                    "rules_locked": True,
                    "paper_only": True,
                    "live_order_effect": "none",
                },
            ]
        ),
        trackers_current=pl.DataFrame(
            [
                _current_tracker(
                    proposal_id,
                    proposal_hash="b" * 64,
                    strategy_id="TAO_F3_F4_DEDUP_8H_PAPER",
                    symbol="TAO-USDT",
                )
            ]
        ),
        created_at=datetime(2026, 7, 11, 2, tzinfo=UTC),
    )

    registry = frames["paper_strategy_registry"].to_dicts()
    assert len(registry) == 1
    assert registry[0]["accepted"] is True
    assert registry[0]["reject_reason"] == ""


def test_accepted_tracker_is_active_before_first_daily_observation() -> None:
    proposal_id = "TAO_F3_F4_DEDUP_8H_PAPER:1.0.0:cccccccccccc"
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
                    "proposal_hash": "c" * 64,
                    "strategy_id": "TAO_F3_F4_DEDUP_8H_PAPER",
                    "strategy_version": "1.0.0",
                    "strategy_candidate": "f3_f4_deduplicated_entry",
                    "symbol": "TAO-USDT",
                    "recommended_mode": "paper",
                }
            ]
        ),
        proposal_ack=pl.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
                    "proposal_hash": "c" * 64,
                    "paper_tracker_id": f"paper:{proposal_id}",
                    "accepted": True,
                    "rules_locked": True,
                    "paper_only": True,
                    "live_order_effect": "none",
                    "strategy_version": "1.0.0",
                    "strategy_candidate": "f3_f4_deduplicated_entry",
                    "symbol": "TAO-USDT",
                }
            ]
        ),
        trackers_current=pl.DataFrame(
            [
                _current_tracker(
                    proposal_id,
                    proposal_hash="c" * 64,
                    strategy_id="TAO_F3_F4_DEDUP_8H_PAPER",
                    symbol="TAO-USDT",
                )
            ]
        ),
        created_at=datetime(2026, 7, 11, 2, tzinfo=UTC),
    )

    registry = frames["paper_strategy_registry"].to_dicts()[0]
    gate = frames["paper_strategy_promotion_gate"].to_dicts()[0]
    assert registry["status"] == "CURRENT_ACTIVE"
    assert registry["lifecycle_state"] == "PAPER_TRACKER_ACTIVE"
    assert gate["paper_tracker_effective"] is True
    assert gate["lifecycle_state"] == "PAPER_EVIDENCE_INSUFFICIENT"


def test_paper_strategy_pipeline_marks_unacked_tracker_evidence_not_effective() -> None:
    proposal_id = "BNB_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
    tracker_id = "BNB_F3_DOMINANT_ENTRY_PAPER_V1"
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
                    "strategy_id": proposal_id,
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "BNB-USDT",
                    "recommended_mode": "paper",
                }
            ]
        ),
        proposal_ack=pl.DataFrame(),
        trackers_current=pl.DataFrame(
            [
                {
                    **_current_tracker(
                        proposal_id,
                        strategy_id=proposal_id,
                        symbol="BNB-USDT",
                    ),
                    "tracker_id": tracker_id,
                }
            ]
        ),
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

    assert registry["status"] == "CURRENT_TRACKER_ACK_MISSING"
    assert gate["paper_tracker_created"] is True
    assert gate["paper_tracker_effective"] is True
    assert gate["current_runtime_eligible"] is False
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
                "market_regime": "TREND_UP" if index % 2 else "SIDEWAYS",
                "cost_source": "mixed_actual_proxy",
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
        trackers_current=pl.DataFrame(
            [
                _current_tracker(
                    proposal_id,
                    strategy_id=proposal_id,
                    symbol="SOL-USDT",
                    created_at="2026-06-01T00:02:00Z",
                )
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
        strategy_cost_trust=pl.DataFrame(
            [
                {
                    "strategy_id": proposal_id,
                    "proposal_id": proposal_id,
                    "strategy_version": "V1",
                    "cost_trust_level": "CANARY",
                    "paper_cost_usable": True,
                    "canary_cost_usable": True,
                    "live_cost_usable": False,
                    "created_at": "2026-06-14T00:00:00Z",
                }
            ]
        ),
        created_at=datetime(2026, 6, 30, tzinfo=UTC),
    )

    registry = frames["paper_strategy_registry"].to_dicts()[0]
    gate = frames["paper_strategy_promotion_gate"].to_dicts()[0]

    assert registry["status"] == "CURRENT_ACTIVE"
    assert registry["rules_locked"] is True
    assert gate["paper_ready"] is True
    assert gate["lifecycle_state"] == "PAPER_PROMOTION_READY"
    assert gate["paper_tracker_created"] is True
    assert gate["paper_tracker_effective"] is True
    assert gate["paper_tracker_status"] == "EFFECTIVE"
    assert gate["strategy_cost_trust_level"] == "CANARY"
    assert gate["required_cost_trust_level"] == "PAPER_ONLY"
    assert json.loads(gate["block_reason"]) == []


def test_strategy_dimensional_cost_trust_separates_paper_from_canary() -> None:
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame(
            [
                {
                    "proposal_id": "DIMENSIONAL_COST_PAPER",
                    "strategy_id": "DIMENSIONAL_COST_PAPER",
                    "strategy_version": "1.0.0",
                    "strategy_candidate": "v5.test",
                    "symbol": "TRX-USDT",
                    "recommended_mode": "paper",
                    "required_cost_trust_level": "CANARY",
                    "created_at": "2026-06-01T00:00:00Z",
                }
            ]
        ),
        strategy_cost_trust=pl.DataFrame(
            [
                {
                    "proposal_id": "DIMENSIONAL_COST_PAPER",
                    "strategy_id": "DIMENSIONAL_COST_PAPER",
                    "strategy_version": "1.0.0",
                    "cost_trust_level": "PAPER_ONLY",
                    "paper_cost_usable": True,
                    "canary_cost_usable": False,
                    "live_cost_usable": False,
                    "created_at": "2026-06-14T00:00:00Z",
                }
            ]
        ),
        created_at=datetime(2026, 6, 30, tzinfo=UTC),
    )

    gate = frames["paper_strategy_promotion_gate"].to_dicts()[0]
    reasons = json.loads(gate["promotion_block_reasons"])
    assert gate["paper_ready"] is False
    assert gate["strategy_cost_trust_level"] == "PAPER_ONLY"
    assert gate["required_cost_trust_level"] == "CANARY"
    assert gate["dimensional_cost_trust_matched"] is True
    assert gate["cost_trusted_for_canary"] is False
    assert "cost_trust_row_missing" not in reasons


def test_build_and_publish_paper_strategy_pipeline_writes_gold_outputs(tmp_path) -> None:
    lake = tmp_path / "lake"
    proposal_id = "BNB_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
    proposal = {
        "proposal_id": proposal_id,
        "strategy_id": proposal_id,
        "strategy_candidate": "v5.f3_dominant_entry",
        "symbol": "BNB-USDT",
        "recommended_mode": "paper",
    }
    write_parquet_dataset(
        pl.DataFrame([proposal]), lake / "gold" / "paper_strategy_proposal"
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
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
        lake / "silver" / "v5_paper_strategy_proposal_ack_current",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    **_current_tracker(
                        proposal_id,
                        strategy_id=proposal_id,
                        symbol="BNB-USDT",
                    ),
                    "tracker_id": "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
                }
            ]
        ),
        lake / "silver" / "v5_paper_strategy_trackers_current",
    )
    write_parquet_dataset(pl.DataFrame(), lake / "gold" / "strategy_opportunity_advisory")
    write_parquet_dataset(pl.DataFrame(), lake / "gold" / "paper_strategy_runs")
    write_parquet_dataset(pl.DataFrame(), lake / "gold" / "paper_strategy_daily")

    result = build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-06-30")

    assert result.paper_strategy_registry == 1
    registry = read_parquet_dataset(lake / "gold" / "paper_strategy_registry")
    gate = read_parquet_dataset(lake / "gold" / "paper_strategy_promotion_gate")
    assert registry.to_dicts()[0]["proposal_id"] == proposal_id
    assert gate.to_dicts()[0]["paper_ready"] is False
    assert gate.to_dicts()[0]["paper_tracker_effective"] is True
    assert gate.to_dicts()[0]["paper_tracker_status"] == "EFFECTIVE"


def test_build_and_publish_pipeline_is_idempotent_and_audits_legacy_rows(tmp_path) -> None:
    lake = tmp_path / "lake"
    advisory = pl.DataFrame(
        [
            {
                "strategy_candidate": "v5.alt_impulse_shadow",
                "symbol": "TRX-USDT",
                "horizon_hours": 48,
                "decision": "PAPER_READY",
                "complete_sample_count": 30,
            },
            {
                "strategy_candidate": "legacy.unselected",
                "symbol": "ETH-USDT",
                "horizon_hours": 24,
                "decision": "PAPER_READY",
                "complete_sample_count": 30,
            },
        ]
    )
    write_parquet_dataset(advisory, lake / "gold" / "strategy_opportunity_advisory")

    first = build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-10")
    first_audit = read_parquet_dataset(lake / "gold" / "paper_strategy_migration_audit")
    second = build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-10")
    second_audit = read_parquet_dataset(lake / "gold" / "paper_strategy_migration_audit")
    proposals = read_parquet_dataset(lake / "gold" / "paper_strategy_proposal")

    assert first.paper_strategy_migration_audit == 2
    assert second.paper_strategy_migration_audit == 2
    assert first_audit["legacy_row_id"].sort().to_list() == (
        second_audit["legacy_row_id"].sort().to_list()
    )
    assert proposals.height == 1
    assert proposals["proposal_id"].n_unique() == 1
    registry = read_parquet_dataset(lake / "gold" / "paper_strategy_registry").to_dicts()[0]
    proposal = proposals.to_dicts()[0]
    assert registry["paper_tracker_id"] == ""
    assert registry["proposal_hash"] == proposal["proposal_hash"]
    assert registry["strategy_id"] == proposal["strategy_id"]
    assert registry["strategy_version"] == proposal["strategy_version"]
    assert registry["lifecycle_state"] == "PAPER_ACK_PENDING"


def test_pipeline_reuses_persisted_proposal_when_current_advisory_is_empty(tmp_path) -> None:
    lake = tmp_path / "lake"
    advisory = pl.DataFrame(
        [
            {
                "strategy_candidate": "v5.alt_impulse_shadow",
                "symbol": "TRX-USDT",
                "horizon_hours": 48,
                "decision": "PAPER_READY",
                "complete_sample_count": 30,
            }
        ]
    )
    write_parquet_dataset(advisory, lake / "gold" / "strategy_opportunity_advisory")
    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-10")
    proposal = read_parquet_dataset(lake / "gold" / "paper_strategy_proposal").to_dicts()[0]

    write_parquet_dataset(pl.DataFrame(), lake / "gold" / "strategy_opportunity_advisory")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": proposal["proposal_id"],
                    "proposal_hash": proposal["proposal_hash"],
                    "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                    "accepted": True,
                    "rules_locked": True,
                    "paper_only": True,
                    "live_order_effect": "none",
                    "symbol": "TRX/USDT",
                    "strategy_version": proposal["strategy_version"],
                    "contract_version": proposal["contract_version"],
                    "accepted_at": "2026-07-11T00:00:00Z",
                }
            ]
        ),
        lake / "silver" / "v5_paper_strategy_proposal_ack_current",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                _current_tracker(
                    str(proposal["proposal_id"]),
                    proposal_hash=str(proposal["proposal_hash"]),
                    strategy_id=str(proposal["strategy_id"]),
                    symbol="TRX/USDT",
                    created_at="2026-07-11T00:05:00Z",
                )
            ]
        ),
        lake / "silver" / "v5_paper_strategy_trackers_current",
    )

    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-11")

    registry = read_parquet_dataset(lake / "gold" / "paper_strategy_registry")
    cost_trust = read_parquet_dataset(lake / "gold" / "strategy_cost_trust")
    row = registry.filter(pl.col("proposal_id") == proposal["proposal_id"]).to_dicts()[0]
    assert row["accepted"] is True
    assert row["proposal_hash"] == proposal["proposal_hash"]
    assert row["contract_version"] == "quant_lab.paper_strategy.v1"
    assert cost_trust.filter(pl.col("strategy_id") == proposal["strategy_id"]).height == 1
