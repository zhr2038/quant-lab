from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.paper_promotion import (
    build_and_publish_paper_strategy_pipeline,
    build_paper_strategy_pipeline_frames,
    parse_cost_source_mix,
)

NOW = datetime(2026, 7, 13, 0, 0, tzinfo=UTC)


def _proposal(
    strategy_id: str,
    *,
    proposal_hash: str,
    horizon: int,
) -> dict[str, object]:
    return {
        "contract_version": "quant_lab.paper_strategy.v1",
        "proposal_id": f"{strategy_id}:1.0.0:{proposal_hash[:12]}",
        "proposal_hash": proposal_hash,
        "strategy_id": strategy_id,
        "strategy_version": "1.0.0",
        "strategy_family": "f3_f4_deduplicated_entry",
        "strategy_candidate": "f3_f4_deduplicated_entry",
        "symbol": "TRX/USDT",
        "timeframe": "1h",
        "max_holding_bars": horizon,
        "recommended_mode": "paper",
        "required_cost_trust_level": "PAPER_ONLY",
        "entry_rule": '{"operator":"gt","field":"close","value":0}',
        "exit_rule": json.dumps({"operator": "max_holding_bars", "value": horizon}),
        "created_at": "2026-07-12T00:00:00Z",
    }


def _ack(proposal: dict[str, object]) -> dict[str, object]:
    proposal_id = str(proposal["proposal_id"])
    return {
        "proposal_id": proposal_id,
        "proposal_hash": proposal["proposal_hash"],
        "paper_tracker_id": f"paper:{proposal_id}",
        "tracker_id": f"paper:{proposal_id}",
        "accepted": True,
        "rules_locked": True,
        "paper_only": True,
        "live_order_effect": "none",
        "strategy_version": "1.0.0",
        "symbol": "TRX/USDT",
        "accepted_at": "2026-07-12T01:00:00Z",
    }


def _tracker(
    proposal: dict[str, object],
    *,
    created_at: str = "2026-07-12T01:05:00Z",
) -> dict[str, object]:
    proposal_id = str(proposal["proposal_id"])
    return {
        "proposal_id": proposal_id,
        "proposal_hash": proposal["proposal_hash"],
        "tracker_id": f"paper:{proposal_id}",
        "strategy_id": proposal["strategy_id"],
        "strategy_version": proposal["strategy_version"],
        "strategy_family": proposal["strategy_family"],
        "symbol": proposal["symbol"],
        "timeframe": proposal["timeframe"],
        "state": "WAITING_SIGNAL",
        "rules_locked": True,
        "paper_only": True,
        "live_order_effect": "none",
        "created_at": created_at,
        "updated_at": created_at,
        "current_proposal_member": True,
        "current_cohort_member": True,
        "supersession_status": "CURRENT_ACTIVE",
        "new_entry_allowed": True,
        "exit_allowed": True,
    }


def _cost_trust(proposal: dict[str, object]) -> dict[str, object]:
    return {
        "proposal_id": proposal["proposal_id"],
        "proposal_hash": proposal["proposal_hash"],
        "strategy_id": proposal["strategy_id"],
        "strategy_version": proposal["strategy_version"],
        "symbol": proposal["symbol"],
        "horizon_hours": proposal["max_holding_bars"],
        "cost_trust_level": "PAPER_ONLY",
        "paper_cost_usable": True,
        "canary_cost_usable": False,
        "live_cost_usable": False,
        "created_at": "2026-07-13T00:00:00Z",
    }


def test_cost_source_mix_mapping_and_quality_errors() -> None:
    assert parse_cost_source_mix('{"configured_conservative_paper":1}') == (
        {"configured_conservative_paper"},
        [],
    )
    assert parse_cost_source_mix(
        '{"actual_fills":2,"configured_conservative_paper":1}'
    ) == ({"actual_fills", "configured_conservative_paper"}, [])
    assert parse_cost_source_mix('{"public_spread_proxy":3}') == (
        {"public_spread_proxy"},
        [],
    )
    assert parse_cost_source_mix("[]") == (set(), [])
    assert parse_cost_source_mix("{}") == (set(), [])
    assert parse_cost_source_mix('{"actual_fills":0}') == (set(), [])
    assert parse_cost_source_mix('{"actual_fills":-1}')[0] == set()
    assert "negative_count:actual_fills" in parse_cost_source_mix(
        '{"actual_fills":-1}'
    )[1]
    assert parse_cost_source_mix("{invalid")[1] == ["invalid_json"]


def test_missing_cost_trust_is_fail_closed() -> None:
    proposal = _proposal("TRX_F3_F4_DEDUP_12H_PAPER", proposal_hash="1" * 64, horizon=12)
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame([proposal]),
        proposal_ack=pl.DataFrame([_ack(proposal)]),
        trackers_current=pl.DataFrame([_tracker(proposal)]),
        runs=pl.DataFrame(
            [
                {
                    "proposal_id": proposal["proposal_id"],
                    "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                    "paper_pnl_bps": 10.0,
                    "cost_source": "configured_conservative_paper",
                    "entry_signal_ts": "2026-07-12T02:00:00Z",
                }
            ]
        ),
        daily=pl.DataFrame(
            [
                {
                    "proposal_id": proposal["proposal_id"],
                    "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                    "closed_entries": 1,
                    "cost_source_mix": '{"configured_conservative_paper":1}',
                }
            ]
        ),
        strategy_cost_trust=pl.DataFrame(),
        created_at=NOW,
    )
    gate = frames["paper_strategy_promotion_gate"].to_dicts()[0]
    assert gate["dimensional_cost_trust_level"] == "NOT_EVALUATED"
    assert gate["dimensional_cost_trust_matched"] is False
    assert gate["cost_evidence_status"] == "COST_TRUST_ROW_MISSING"
    assert gate["cost_trusted_for_paper"] is False
    assert gate["paper_ready"] is False


def test_registry_identity_is_canonical_and_cost_match_is_exact() -> None:
    proposal = _proposal("TRX_F3_F4_DEDUP_12H_PAPER", proposal_hash="2" * 64, horizon=12)
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame([proposal]),
        proposal_ack=pl.DataFrame([_ack(proposal)]),
        trackers_current=pl.DataFrame([_tracker(proposal)]),
        runs=pl.DataFrame(
            [
                {
                    "proposal_id": proposal["proposal_id"],
                    "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                    "strategy_id": "TRX_F3_F4_DEDUP_4H_PAPER",
                    "paper_pnl_bps": 12.0,
                    "cost_source": "configured_conservative_paper",
                    "entry_signal_ts": "2026-07-12T02:00:00Z",
                }
            ]
        ),
        daily=pl.DataFrame(
            [
                {
                    "proposal_id": proposal["proposal_id"],
                    "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                    "strategy_id": "TRX_F3_F4_DEDUP_4H_PAPER",
                    "closed_entries": 1,
                    "cost_source_mix": '{"configured_conservative_paper":1}',
                }
            ]
        ),
        strategy_cost_trust=pl.DataFrame([_cost_trust(proposal)]),
        created_at=NOW,
    )
    registry = frames["paper_strategy_registry"].to_dicts()[0]
    gate = frames["paper_strategy_promotion_gate"].to_dicts()[0]
    assert registry["strategy_id"] == "TRX_F3_F4_DEDUP_12H_PAPER"
    assert registry["max_holding_bars"] == 12
    assert gate["dimensional_cost_trust_matched"] is True
    assert gate["paper_cost_model_usable"] is True
    assert gate["closed_trade_cost_coverage"] == 1.0
    assert gate["cost_trusted_for_paper"] is True
    assert gate["cost_trusted_for_canary"] is False


def test_shared_entry_event_counts_once_across_horizons() -> None:
    proposal_8h = _proposal("TRX_F3_F4_DEDUP_8H_PAPER", proposal_hash="3" * 64, horizon=8)
    proposal_12h = _proposal("TRX_F3_F4_DEDUP_12H_PAPER", proposal_hash="4" * 64, horizon=12)
    runs = []
    daily = []
    trusts = []
    for proposal in (proposal_8h, proposal_12h):
        runs.append(
            {
                "proposal_id": proposal["proposal_id"],
                "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                "paper_pnl_bps": 10.0,
                "cost_source": "configured_conservative_paper",
                "entry_signal_ts": "2026-07-12T02:00:00Z",
                "entry_decision_ts": "2026-07-12T02:00:01Z",
            }
        )
        daily.append(
            {
                "proposal_id": proposal["proposal_id"],
                "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                "closed_entries": 1,
                "cost_source_mix": '{"configured_conservative_paper":1}',
            }
        )
        trusts.append(_cost_trust(proposal))
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame([proposal_8h, proposal_12h]),
        proposal_ack=pl.DataFrame([_ack(proposal_8h), _ack(proposal_12h)]),
        trackers_current=pl.DataFrame([_tracker(proposal_8h), _tracker(proposal_12h)]),
        runs=pl.DataFrame(runs),
        daily=pl.DataFrame(daily),
        strategy_cost_trust=pl.DataFrame(trusts),
        created_at=NOW,
    )
    gates = frames["paper_strategy_promotion_gate"].to_dicts()
    assert sum(row["raw_closed_trade_count"] for row in gates) == 2
    assert sum(row["evidence_independence_weight"] for row in gates) == 1.0
    assert {row["horizon_variant_count"] for row in gates} == {2}


def test_structured_rows_do_not_cross_match_same_candidate_symbol() -> None:
    proposal_8h = _proposal("TRX_F3_F4_DEDUP_8H_PAPER", proposal_hash="8" * 64, horizon=8)
    proposal_12h = _proposal(
        "TRX_F3_F4_DEDUP_12H_PAPER", proposal_hash="9" * 64, horizon=12
    )
    runs = []
    daily = []
    trusts = []
    for proposal in (proposal_8h, proposal_12h):
        runs.append(
            {
                "proposal_id": proposal["proposal_id"],
                "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                "strategy_id": proposal["strategy_id"],
                "strategy_candidate": "f3_f4_deduplicated_entry",
                "symbol": "TRX/USDT",
                "paper_pnl_bps": 10.0,
                "cost_source": "configured_conservative_paper",
                "entry_signal_ts": "2026-07-12T02:00:00Z",
            }
        )
        daily.append(
            {
                "proposal_id": proposal["proposal_id"],
                "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                "strategy_id": proposal["strategy_id"],
                "strategy_candidate": "f3_f4_deduplicated_entry",
                "symbol": "TRX/USDT",
                "closed_entries": 1,
                "cost_source_mix": '{"configured_conservative_paper":1}',
            }
        )
        trusts.append(_cost_trust(proposal))

    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame([proposal_8h, proposal_12h]),
        proposal_ack=pl.DataFrame([_ack(proposal_8h), _ack(proposal_12h)]),
        trackers_current=pl.DataFrame([_tracker(proposal_8h), _tracker(proposal_12h)]),
        runs=pl.DataFrame(runs),
        daily=pl.DataFrame(daily),
        strategy_cost_trust=pl.DataFrame(trusts),
        created_at=NOW,
    )

    gates = frames["paper_strategy_promotion_gate"].to_dicts()
    assert {row["raw_closed_trade_count"] for row in gates} == {1}
    assert {row["closed_trade_cost_observation_count"] for row in gates} == {1}
    assert {row["closed_trade_cost_coverage"] for row in gates} == {1.0}
    assert sum(row["evidence_independence_weight"] for row in gates) == 1.0


def test_published_gate_replaces_stale_derived_evidence(tmp_path) -> None:
    lake = tmp_path / "lake"
    proposal = _proposal(
        "TRX_F3_F4_DEDUP_8H_PAPER", proposal_hash="a" * 64, horizon=8
    )
    write_parquet_dataset(
        pl.DataFrame([proposal]), lake / "gold/paper_strategy_proposal"
    )
    write_parquet_dataset(
        pl.DataFrame([_ack(proposal)]),
        lake / "silver/v5_paper_strategy_proposal_ack_current",
    )
    write_parquet_dataset(
        pl.DataFrame([_tracker(proposal)]),
        lake / "silver/v5_paper_strategy_trackers_current",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": proposal["proposal_id"],
                    "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                    "strategy_id": proposal["strategy_id"],
                    "strategy_candidate": "f3_f4_deduplicated_entry",
                    "symbol": "TRX/USDT",
                    "paper_pnl_bps": 10.0,
                    "cost_source": "configured_conservative_paper",
                    "entry_signal_ts": "2026-07-12T02:00:00Z",
                }
            ]
        ),
        lake / "gold/paper_strategy_runs",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": proposal["proposal_id"],
                    "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                    "strategy_id": proposal["strategy_id"],
                    "strategy_candidate": "f3_f4_deduplicated_entry",
                    "symbol": "TRX/USDT",
                    "paper_pnl_observed_count": 1,
                    "cost_source_mix": '{"configured_conservative_paper":1}',
                }
            ]
        ),
        lake / "gold/paper_strategy_daily",
    )

    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-13")
    stale = read_parquet_dataset(lake / "gold/paper_strategy_promotion_gate").with_columns(
        pl.lit(99).alias("paper_runs"),
        pl.lit(99).alias("closed_entries"),
        pl.lit(99).alias("raw_closed_trade_count"),
        pl.lit(99).alias("closed_trade_cost_observation_count"),
        pl.lit(99.0).alias("closed_trade_cost_coverage"),
    )
    write_parquet_dataset(stale, lake / "gold/paper_strategy_promotion_gate")

    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-13")

    gate = read_parquet_dataset(lake / "gold/paper_strategy_promotion_gate").to_dicts()[0]
    assert gate["closed_entries"] == 1
    assert gate["raw_closed_trade_count"] == 1
    assert gate["closed_trade_cost_observation_count"] == 1
    assert gate["closed_trade_cost_coverage"] == 1.0


def test_tracker_id_must_belong_to_its_proposal(tmp_path) -> None:
    lake = tmp_path / "lake"
    proposal = _proposal(
        "TRX_F3_F4_DEDUP_8H_PAPER",
        proposal_hash="7" * 64,
        horizon=8,
    )
    wrong_ack = _ack(proposal)
    wrong_ack["paper_tracker_id"] = "paper:another-proposal"
    wrong_ack["tracker_id"] = "paper:another-proposal"
    write_parquet_dataset(
        pl.DataFrame([proposal]), lake / "gold/paper_strategy_proposal"
    )
    write_parquet_dataset(
        pl.DataFrame([wrong_ack]),
        lake / "silver/v5_paper_strategy_proposal_ack_current",
    )

    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-13")

    conflicts = read_parquet_dataset(lake / "gold/paper_strategy_identity_conflict")
    tracker_conflicts = conflicts.filter(
        pl.col("conflict_field") == "tracker_id_to_proposal_id"
    )
    assert tracker_conflicts.height == 1
    assert tracker_conflicts.to_dicts()[0]["active_conflict"] is True
    registry = read_parquet_dataset(lake / "gold/paper_strategy_registry_current")
    assert registry.to_dicts()[0]["lifecycle_state"] == "IDENTITY_CONFLICT"
    gate = read_parquet_dataset(lake / "gold/paper_strategy_promotion_gate")
    assert gate.to_dicts()[0]["paper_ready"] is False
    assert gate.to_dicts()[0]["cost_trusted_for_paper"] is False


def test_publish_separates_current_history_and_freezes_cohort(tmp_path) -> None:
    lake = tmp_path / "lake"
    first = _proposal("TRX_F3_F4_DEDUP_8H_PAPER", proposal_hash="5" * 64, horizon=8)
    write_parquet_dataset(pl.DataFrame([first]), lake / "gold/paper_strategy_proposal")
    write_parquet_dataset(
        pl.DataFrame([_ack(first)]),
        lake / "silver/v5_paper_strategy_proposal_ack_current",
    )
    write_parquet_dataset(
        pl.DataFrame([_tracker(first)]),
        lake / "silver/v5_paper_strategy_trackers_current",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    **{key: value for key, value in _ack(first).items()},
                    "strategy_id": "TRX_F3_F4_DEDUP_4H_PAPER",
                    "status": "PAPER_TRACKING",
                    "paper_start_at": "2026-07-12T01:00:00Z",
                }
            ]
        ),
        lake / "gold/paper_strategy_registry",
    )
    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-13")

    current = read_parquet_dataset(lake / "gold/paper_strategy_registry_current")
    history = read_parquet_dataset(lake / "gold/paper_strategy_registry_history")
    conflicts = read_parquet_dataset(lake / "gold/paper_strategy_identity_conflict")
    cohort_v1 = read_parquet_dataset(lake / "gold/paper_cohort_manifest")
    assert current.to_dicts()[0]["strategy_id"] == first["strategy_id"]
    assert history.height >= 1
    assert conflicts.filter(pl.col("conflict_field") == "strategy_id").height == 1
    assert cohort_v1.height == 1
    assert cohort_v1.to_dicts()[0]["status"] == "OBSERVING"
    assert cohort_v1.to_dicts()[0]["all_members_admitted"] is True
    original_members = cohort_v1.to_dicts()[0]["proposal_ids"]

    write_parquet_dataset(
        pl.DataFrame(
            [{
                "proposal_id": first["proposal_id"],
                "paper_tracker_id": f"paper:{first['proposal_id']}",
                "entry_signal_ts": "2026-07-13T02:00:00Z",
            }]
        ),
        lake / "gold/paper_strategy_runs",
    )
    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-14")
    refreshed = read_parquet_dataset(lake / "gold/paper_cohort_manifest")
    assert refreshed.height == 1
    assert refreshed.to_dicts()[0]["cohort_version"] == 1
    assert refreshed.to_dicts()[0]["raw_closed_trade_count"] == 1
    assert refreshed.to_dicts()[0]["independent_closed_trade_count"] == 1

    second = _proposal("TRX_F3_F4_DEDUP_12H_PAPER", proposal_hash="6" * 64, horizon=12)
    write_parquet_dataset(pl.DataFrame([first, second]), lake / "gold/paper_strategy_proposal")
    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-15")
    cohorts = read_parquet_dataset(lake / "gold/paper_cohort_manifest").sort("cohort_version")
    assert cohorts.height == 2
    assert cohorts.to_dicts()[0]["proposal_ids"] == original_members
    assert cohorts.to_dicts()[0]["status"] == "FROZEN"
    assert json.loads(cohorts.to_dicts()[1]["proposal_ids"]) == sorted(
        [first["proposal_id"], second["proposal_id"]]
    )
    assert cohorts.to_dicts()[1]["status"] == "FORMING"
    assert cohorts.to_dicts()[1]["observation_start_at"] is None

    second_ack = _ack(second)
    second_ack["accepted_at"] = "2026-07-15T03:00:00Z"
    write_parquet_dataset(
        pl.DataFrame([_ack(first), second_ack]),
        lake / "silver/v5_paper_strategy_proposal_ack_current",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                _tracker(first),
                _tracker(second, created_at="2026-07-15T03:05:00Z"),
            ]
        ),
        lake / "silver/v5_paper_strategy_trackers_current",
    )
    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-15")
    admitted = read_parquet_dataset(lake / "gold/paper_cohort_manifest").sort(
        "cohort_version"
    )
    assert admitted.to_dicts()[1]["status"] == "OBSERVING"
    assert admitted.to_dicts()[1]["formal_observation_eligible"] is True
    assert admitted.to_dicts()[1]["observation_start_at"] == "2026-07-15T03:05:00Z"


def test_current_lifecycle_does_not_inherit_history() -> None:
    proposal = _proposal(
        "ARB_F3_F4_DEDUP_8H_PAPER", proposal_hash="b" * 64, horizon=8
    )
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame([proposal]),
        proposal_ack=pl.DataFrame(),
        trackers_current=pl.DataFrame(),
        runs=pl.DataFrame(
            [
                {
                    "proposal_id": proposal["proposal_id"],
                    "proposal_hash": proposal["proposal_hash"],
                    "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                    "paper_pnl_bps": 12.0,
                    "cost_source": "configured_conservative_paper",
                    "entry_signal_ts": "2026-07-01T00:00:00Z",
                }
            ]
        ),
        daily=pl.DataFrame(),
        strategy_cost_trust=pl.DataFrame([_cost_trust(proposal)]),
        created_at=NOW,
    )
    registry = frames["paper_strategy_registry"].to_dicts()[0]
    gate = frames["paper_strategy_promotion_gate"].to_dicts()[0]
    assert registry["accepted"] is False
    assert registry["current_ack_present"] is False
    assert registry["current_tracker_present"] is False
    assert registry["paper_tracker_effective"] is False
    assert registry["current_runtime_eligible"] is False
    assert registry["evidence_from_history"] is True
    assert registry["supersession_status"] == "CURRENT_PENDING_ACK"
    assert gate["historical_evidence_present"] is True
    assert gate["historical_closed_trade_count"] == 1
    assert gate["current_contract_closed_trade_count"] == 1
    assert gate["current_runtime_eligible"] is False
    assert gate["paper_ready"] is False


def test_published_current_drops_historical_ack_and_tracker_but_history_retains_them(
    tmp_path,
) -> None:
    lake = tmp_path / "lake"
    proposal = _proposal(
        "ARB_F3_F4_DEDUP_8H_PAPER", proposal_hash="e" * 64, horizon=8
    )
    historical = {
        **proposal,
        **_ack(proposal),
        "status": "CURRENT_ACTIVE",
        "lifecycle_state": "PAPER_TRACKER_ACTIVE",
        "paper_tracker_effective": True,
        "current_tracker_effective": True,
        "current_runtime_eligible": True,
        "current_ack_present": True,
        "current_tracker_present": True,
        "rules_locked": True,
        "new_entry_allowed": True,
        "exit_allowed": True,
    }
    write_parquet_dataset(
        pl.DataFrame([proposal]), lake / "gold/paper_strategy_proposal"
    )
    write_parquet_dataset(
        pl.DataFrame([historical]), lake / "gold/paper_strategy_registry_current"
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": proposal["proposal_id"],
                    "proposal_hash": proposal["proposal_hash"],
                    "paper_tracker_id": f"paper:{proposal['proposal_id']}",
                    "paper_pnl_bps": 15.0,
                    "cost_source": "configured_conservative_paper",
                    "entry_signal_ts": "2026-07-01T00:00:00Z",
                }
            ]
        ),
        lake / "gold/paper_strategy_runs",
    )

    build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-13")

    current = read_parquet_dataset(
        lake / "gold/paper_strategy_registry_current"
    ).to_dicts()[0]
    history = read_parquet_dataset(
        lake / "gold/paper_strategy_registry_history"
    ).to_dicts()[0]
    assert current["accepted"] is False
    assert current["paper_tracker_effective"] is False
    assert current["current_runtime_eligible"] is False
    assert current["evidence_from_history"] is True
    assert current["supersession_status"] == "CURRENT_PENDING_ACK"
    assert history["accepted"] is True
    assert history["evidence_from_history"] is True
    assert history["current_tracker_effective"] is False
    assert history["supersession_status"] == "HISTORY_ONLY"


def test_current_ack_without_tracker_is_fail_closed() -> None:
    proposal = _proposal(
        "BNB_F3_F4_DEDUP_8H_PAPER", proposal_hash="c" * 64, horizon=8
    )
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame([proposal]),
        proposal_ack=pl.DataFrame([_ack(proposal)]),
        trackers_current=pl.DataFrame(),
        created_at=NOW,
    )
    registry = frames["paper_strategy_registry"].to_dicts()[0]
    assert registry["accepted"] is True
    assert registry["current_ack_present"] is True
    assert registry["current_tracker_present"] is False
    assert registry["paper_tracker_effective"] is False
    assert registry["supersession_status"] == "CURRENT_ACKED_TRACKER_MISSING"
    assert registry["new_entry_allowed"] is False


def test_current_tracker_without_ack_is_fail_closed() -> None:
    proposal = _proposal(
        "TAO_F3_F4_DEDUP_8H_PAPER", proposal_hash="d" * 64, horizon=8
    )
    frames = build_paper_strategy_pipeline_frames(
        proposals=pl.DataFrame([proposal]),
        proposal_ack=pl.DataFrame(),
        trackers_current=pl.DataFrame([_tracker(proposal)]),
        created_at=NOW,
    )
    registry = frames["paper_strategy_registry"].to_dicts()[0]
    assert registry["accepted"] is False
    assert registry["current_ack_present"] is False
    assert registry["current_tracker_present"] is True
    assert registry["paper_tracker_effective"] is True
    assert registry["current_runtime_eligible"] is False
    assert registry["supersession_status"] == "CURRENT_TRACKER_ACK_MISSING"
    assert registry["new_entry_allowed"] is False
