from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.factors.registry import discover_factor_specs
from quant_lab.paper.canonical import (
    annotate_shared_events,
    canonical_market_event_id,
    strategy_evaluation_id,
)
from quant_lab.paper.contracts import (
    LifecycleState,
    PaperStrategyProposal,
    assert_lifecycle_transition,
    legacy_lifecycle_state,
    paper_proposal_hash,
)
from quant_lab.paper.cost_trust import evaluate_strategy_cost_trust
from quant_lab.paper.proposals import (
    build_configured_proposals,
    build_legacy_proposal_migration_audit,
    load_proposal_templates,
)


def _proposal(**updates):
    payload = {
        "strategy_id": "TEST_PAPER",
        "strategy_version": "1.0.0",
        "strategy_family": "test",
        "symbol": "TRX/USDT",
        "timeframe": "1h",
        "entry_rule": {"operator": "momentum_gt", "field": "momentum_24", "value": 0},
        "exit_rule": {"operator": "max_holding_bars", "value": 48},
        "max_holding_bars": 48,
        "created_at": datetime(2026, 7, 10, tzinfo=UTC),
        "expires_at": datetime(2026, 8, 10, tzinfo=UTC),
        "required_market_fields": ["bid", "ask", "mid", "momentum_24"],
        "required_cost_trust_level": "PAPER_ONLY",
    }
    payload.update(updates)
    return PaperStrategyProposal(**payload)


def test_lifecycle_state_machine_and_legacy_paper_ready_mapping():
    assert (
        assert_lifecycle_transition("BACKTEST_CANDIDATE", "PAPER_PROPOSAL_READY")
        == LifecycleState.PAPER_PROPOSAL_READY
    )
    with pytest.raises(ValueError, match="illegal paper lifecycle transition"):
        assert_lifecycle_transition("PAPER_PROPOSAL_READY", "CANARY_READY")
    assert legacy_lifecycle_state("PAPER_READY") == LifecycleState.PAPER_PROPOSAL_READY
    assert legacy_lifecycle_state("PROPOSED_AWAITING_ACK") == LifecycleState.PAPER_ACK_PENDING


def test_proposal_hash_is_idempotent_and_rule_change_requires_new_hash():
    first = _proposal()
    duplicate = _proposal(created_at=first.created_at + timedelta(hours=1))
    changed = _proposal(
        strategy_version="1.1.0",
        entry_rule={"operator": "momentum_gt", "field": "momentum_24", "value": 0.01},
    )

    assert first.proposal_hash == duplicate.proposal_hash
    assert first.proposal_id == duplicate.proposal_id
    assert changed.proposal_hash != first.proposal_hash
    assert "entry_rule" in PaperStrategyProposal.model_json_schema()["properties"]


def test_raw_contract_hash_matches_v5_canonical_vector():
    payload = {
        "contract_version": "quant_lab.paper_strategy.v1",
        "strategy_id": "CONTRACT_TEST",
        "strategy_version": "1.0.0",
        "strategy_family": "contract",
        "symbol": "TRX/USDT",
        "timeframe": "1h",
        "direction": "long",
        "entry_rule": {"operator": "momentum_gt", "field": "momentum_24", "value": 0},
        "exit_rule": {"operator": "max_holding_bars", "value": 48},
        "max_holding_bars": 48,
        "min_holding_bars": 1,
        "cooldown_bars": 2,
        "signal_confirmation_bars": 1,
        "cost_quantile": "p75",
        "minimum_expected_edge_bps": 10.0,
        "paper_notional_usdt": 20.0,
        "paper_only": True,
        "live_order_effect": "none",
        "max_live_notional_usdt": 0.0,
        "created_at": "2026-07-10T00:00:00Z",
        "expires_at": "2026-08-10T00:00:00Z",
        "source_pack_sha256": "",
        "source_dataset_versions": {"alpha_discovery_board": "v1"},
        "required_market_fields": ["bid", "ask", "mid", "momentum_24"],
        "required_cost_trust_level": "PAPER_ONLY",
        "lifecycle_state": "PAPER_PROPOSAL_READY",
        "lifecycle_reason": "ignored",
        "blocked_reasons": ["ignored"],
        "next_required_actions": ["ignored"],
    }

    assert paper_proposal_hash(payload) == (
        "6d922297dfdd33019d720d5491e276382d49c710e0823997f78e44a21dd29acb"
    )


def test_first_batch_templates_build_only_three_generic_proposals():
    templates = load_proposal_templates()
    rows = [
        {
            "strategy_candidate": "v5.alt_impulse_shadow",
            "symbol": "TRX-USDT",
            "horizon_hours": 48,
            "sample_count": 40,
            "complete_sample_count": 30,
            "avg_net_bps": 40.0,
            "p25_net_bps": 1.0,
            "win_rate": 0.6,
        },
        {
            "strategy_candidate": "v5.f3_dominant_entry",
            "symbol": "BCH-USDT",
            "horizon_hours": 72,
            "sample_count": 40,
            "complete_sample_count": 30,
        },
        {
            "strategy_candidate": "v5.f4_volume_expansion_entry",
            "symbol": "BCH-USDT",
            "horizon_hours": 72,
            "sample_count": 40,
            "complete_sample_count": 30,
        },
        {
            "strategy_candidate": "v5.f4_volume_expansion_entry",
            "symbol": "TAO-USDT",
            "horizon_hours": 8,
            "sample_count": 40,
            "complete_sample_count": 30,
        },
    ]
    proposals = build_configured_proposals(
        pl.DataFrame(rows), created_at=datetime(2026, 7, 10, tzinfo=UTC)
    )

    assert len(templates) == 3
    assert {proposal.strategy_id for proposal, _ in proposals} == {
        "TRX_ALT_IMPULSE_48H_PAPER",
        "BCH_F3_F4_DEDUP_72H_PAPER",
        "TAO_F3_F4_DEDUP_8H_PAPER",
    }
    assert all(proposal.paper_only for proposal, _ in proposals)
    assert all(proposal.live_order_effect == "none" for proposal, _ in proposals)


def test_legacy_paper_rows_have_explicit_migration_outcomes():
    rows = [
        {
            "strategy_candidate": "v5.alt_impulse_shadow",
            "symbol": "TRX-USDT",
            "horizon_hours": 48,
            "decision": "PAPER_READY",
            "complete_sample_count": 30,
        },
        {
            "strategy_candidate": "v5.f3_dominant_entry",
            "symbol": "BCH-USDT",
            "horizon_hours": 72,
            "decision": "PAPER_READY",
            "complete_sample_count": 31,
        },
        {
            "strategy_candidate": "v5.f4_volume_expansion_entry",
            "symbol": "BCH-USDT",
            "horizon_hours": 72,
            "decision": "PAPER_READY",
            "complete_sample_count": 30,
        },
        {
            "strategy_candidate": "legacy.eth.paper",
            "symbol": "ETH-USDT",
            "horizon_hours": 24,
            "decision": "PAPER_READY",
        },
        {
            "strategy_candidate": "legacy.invalid",
            "symbol": "",
            "horizon_hours": 0,
            "recommended_mode": "paper",
        },
    ]
    evidence = pl.DataFrame(rows)
    configured = build_configured_proposals(
        evidence,
        created_at=datetime(2026, 7, 10, tzinfo=UTC),
    )

    audit = build_legacy_proposal_migration_audit(
        evidence,
        configured,
        created_at=datetime(2026, 7, 10, tzinfo=UTC),
    )
    outcomes = {row["strategy_candidate"]: row["migration_status"] for row in audit.to_dicts()}

    assert outcomes["v5.alt_impulse_shadow"] == "MIGRATED_TO_V1_CONTRACT"
    assert outcomes["v5.f3_dominant_entry"] == "MIGRATED_TO_V1_CONTRACT"
    assert outcomes["v5.f4_volume_expansion_entry"] == "DEDUPED_TO_CANONICAL_PROPOSAL"
    assert outcomes["legacy.eth.paper"] == "NOT_SELECTED_FIRST_BATCH"
    assert outcomes["legacy.invalid"] == "INVALID_LEGACY_ROW"
    assert audit["legacy_row_id"].n_unique() == audit.height


def test_f3_f4_market_events_share_canonical_identity_but_keep_evaluations():
    common = {
        "symbol": "BCH-USDT",
        "timeframe": "1h",
        "decision_ts": "2026-07-10T00:01:00Z",
        "source_dataset_version": "v1",
        "rank": 1,
    }
    f3 = canonical_market_event_id({**common, "strategy_candidate": "v5.f3_dominant_entry"})
    f4 = canonical_market_event_id(
        {
            **common,
            "decision_ts": "2026-07-10T00:59:00Z",
            "candidate_id": "strategy-specific-f4-id",
            "strategy_candidate": "v5.f4_volume_expansion",
        }
    )
    annotated = annotate_shared_events(
        [
            {"canonical_event_id": f3, "strategy_id": "F3", "strategy_version": "1"},
            {"canonical_event_id": f4, "strategy_id": "F4", "strategy_version": "1"},
            {"canonical_event_id": f4, "strategy_id": "F4", "strategy_version": "1"},
        ]
    )

    assert f3 == f4
    assert annotated[0]["shared_event_strategy_count"] == 2
    assert annotated[0]["event_independence_weight"] == 0.5
    assert len(annotated) == 2
    assert strategy_evaluation_id(event_id=f3, strategy_id="F3", strategy_version="1") != (
        strategy_evaluation_id(event_id=f3, strategy_id="F4", strategy_version="1")
    )


def test_strategy_evaluation_dedupes_reimport_but_keeps_distinct_source_events():
    first = strategy_evaluation_id(
        event_id="evt:shared",
        strategy_id="F3",
        strategy_version="1",
        source_event_id="candidate-1",
    )
    repeated = strategy_evaluation_id(
        event_id="evt:shared",
        strategy_id="F3",
        strategy_version="1",
        source_event_id="candidate-1",
    )
    second = strategy_evaluation_id(
        event_id="evt:shared",
        strategy_id="F3",
        strategy_version="1",
        source_event_id="candidate-2",
    )

    assert first == repeated
    assert first != second


def test_factor_semantic_duplicates_keep_lineage_and_share_weight():
    specs = discover_factor_specs(["close_return_24"])
    by_id = {spec.factor_id: spec for spec in specs}

    assert by_id["core.close_return_24"].canonical_factor_id == (
        by_id["auto.single.close_return_24"].canonical_factor_id
    )
    assert by_id["auto.single.close_return_24"].duplicate_of == "core.close_return_24"
    assert by_id["core.close_return_24"].effective_independence_weight == 0.5


def test_strategy_dimensional_cost_trust_blocks_missing_conditions():
    result = evaluate_strategy_cost_trust(
        strategy_id="TEST",
        required_conditions=[
            {
                "symbol": "TRX-USDT",
                "notional_bucket": "0_20",
                "market_regime": "normal",
                "liquidity_role": "taker",
                "order_leg": "entry",
                "spread_bucket": "tight",
                "volatility_bucket": "normal",
            }
        ],
        observations=[],
    )

    assert result.cost_trust_level == "BLOCK"
    assert "symbol" in result.missing_dimensions


def test_cost_trust_requires_fresh_entry_and_exit_and_never_promotes_bootstrap():
    now = datetime(2026, 7, 10, tzinfo=UTC)
    required = [
        {
            "symbol": "TRX-USDT",
            "notional_bucket": "0_20",
            "market_regime": "normal",
            "liquidity_role": "taker",
            "order_leg": leg,
            "spread_bucket": "tight",
            "volatility_bucket": "normal",
        }
        for leg in ("entry", "exit")
    ]
    actual = [
        {
            **condition,
            "cost_source": "actual_fills_bills",
            "sample_count": 10,
            "observed_at": "2026-07-10T00:00:00Z",
        }
        for condition in required
    ]
    bootstrap = [
        {
            **condition,
            "cost_source": "bootstrap_cost_probe",
            "sample_count": 30,
            "observed_at": "2026-07-10T00:00:00Z",
        }
        for condition in required
    ]
    undated = [
        {
            **condition,
            "cost_source": "actual_fills_bills",
            "sample_count": 30,
        }
        for condition in required
    ]

    assert (
        evaluate_strategy_cost_trust(
            strategy_id="TEST",
            required_conditions=required,
            observations=actual,
            now=now,
        ).cost_trust_level
        == "CANARY"
    )
    assert (
        evaluate_strategy_cost_trust(
            strategy_id="TEST",
            required_conditions=required,
            observations=bootstrap,
            now=now,
        ).cost_trust_level
        == "PAPER_ONLY"
    )
    assert (
        evaluate_strategy_cost_trust(
            strategy_id="TEST",
            required_conditions=required,
            observations=undated,
            now=now,
        ).cost_trust_level
        == "PAPER_ONLY"
    )
