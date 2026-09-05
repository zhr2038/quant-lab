from datetime import UTC, datetime, timedelta

import pytest

from quant_lab.paper.contracts import (
    LifecycleState,
    PaperStrategyProposal,
    assert_lifecycle_transition,
    legacy_lifecycle_state,
    paper_proposal_hash,
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
