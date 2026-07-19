from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.factor_research.contracts import (
    DiscoverySource,
    FactorResearchDecision,
    HypothesisStatus,
    ResearchTrial,
    TrialKind,
    TrialStatus,
)
from quant_lab.research.factor_research.registry import (
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
    default_hypothesis_registry,
    publish_hypothesis_registry,
    publish_trial_ledger,
    validate_hypothesis_budget,
)


def test_default_hypotheses_are_bounded_and_ai_drafts_do_not_execute() -> None:
    hypotheses = default_hypothesis_registry()
    validate_hypothesis_budget(hypotheses)
    active = [item for item in hypotheses if item.status == HypothesisStatus.APPROVED_FOR_RESEARCH]
    assert len(active) == 2
    assert all(len(item.allowed_variants) <= 3 for item in hypotheses)
    assert all(len(item.expected_horizons) <= 3 for item in hypotheses)
    assert all(item.discovery_source != DiscoverySource.AI_DRAFT for item in active)
    blocked = [item for item in hypotheses if item.status == HypothesisStatus.DATA_BLOCKED]
    assert {item.factor_family.value for item in blocked} == {
        "DERIVATIVES_CROWDING",
        "LIQUIDITY_MICROSTRUCTURE",
    }


def test_hypothesis_budget_rejects_more_than_two_active_per_family() -> None:
    source = default_hypothesis_registry()[0]
    hypotheses = [
        source.model_copy(
            update={
                "hypothesis_id": f"defensive.extra.{index}",
                "research_thread_id": f"factor-v2.defensive.extra.{index}",
            }
        )
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="RESEARCH_BUDGET_EXCEEDED"):
        validate_hypothesis_budget(hypotheses)


def test_hypothesis_definition_change_requires_new_version(tmp_path) -> None:
    root = tmp_path / "lake"
    original = default_hypothesis_registry()[0]
    publish_hypothesis_registry(root, [original])
    changed = original.model_copy(update={"expected_horizons": (24,)})
    with pytest.raises(ValueError, match="new version"):
        publish_hypothesis_registry(root, [changed])
    version_two = changed.model_copy(update={"hypothesis_version": 2})
    assert publish_hypothesis_registry(root, [version_two]) == 2
    frame = read_parquet_dataset(root / RESEARCH_HYPOTHESIS_REGISTRY_DATASET)
    assert frame.height == 2


def test_trial_ledger_keeps_failures_and_rejects_identity_mutation(tmp_path) -> None:
    now = datetime(2026, 7, 19, 4, 10, tzinfo=UTC)
    digest = hashlib.sha256(b"formula").hexdigest()
    recipe = hashlib.sha256(b"recipe").hexdigest()
    trial = ResearchTrial(
        trial_id="factor-trial-test-001",
        hypothesis_id="defensive.low_vol_decomposition",
        hypothesis_version=1,
        test_family_id="defensive.low-vol.v1",
        factor_formula_hash=digest,
        feature_recipe_hash=recipe,
        direction=-1,
        lookback=480,
        horizon=24,
        universe_id="spot-dynamic-quality-v1",
        neutralization_id="xs-core-controls-v1",
        cost_model_id="research-point-in-time-p75-v1",
        portfolio_rule_id="top3-equal-weight-v1",
        split_definition="chronological:research=50,validation=25,blind=25",
        blind_period_id="blind-2026h2-v1",
        random_seed=20260719,
        code_commit="a" * 40,
        data_snapshot_id="factor-snapshot-test-001",
        nas_task_id="factor-research-test-001",
        trial_kind=TrialKind.CONFIRMATORY,
        parameter_locked_at=now,
        submitted_at=now,
        finished_at=now,
        status=TrialStatus.FAILED,
        decision=FactorResearchDecision.INCONCLUSIVE,
        failure_reason="fixture_failure",
    )
    root = tmp_path / "lake"
    assert publish_trial_ledger(root, [trial]) == 1
    frame = read_parquet_dataset(root / RESEARCH_TRIAL_LEDGER_DATASET)
    assert frame.row(0, named=True)["counts_toward_multiple_testing"] is True
    assert frame.row(0, named=True)["status"] == "FAILED"

    changed = trial.model_copy(update={"horizon": 72})
    with pytest.raises(ValueError, match="identity mutation"):
        publish_trial_ledger(root, [changed])


def test_post_hoc_confirmatory_change_invalidates_blind() -> None:
    now = datetime(2026, 7, 19, 4, 10, tzinfo=UTC)
    fields = {
        "trial_id": "factor-trial-test-002",
        "hypothesis_id": "defensive.low_vol_decomposition",
        "hypothesis_version": 1,
        "test_family_id": "defensive.low-vol.v1",
        "factor_formula_hash": "a" * 64,
        "feature_recipe_hash": "b" * 64,
        "direction": -1,
        "lookback": 480,
        "horizon": 24,
        "universe_id": "spot-dynamic-quality-v1",
        "neutralization_id": "xs-core-controls-v1",
        "cost_model_id": "research-point-in-time-p75-v1",
        "portfolio_rule_id": "top3-equal-weight-v1",
        "split_definition": "chronological",
        "blind_period_id": "blind-test-v1",
        "random_seed": 7,
        "code_commit": "a" * 40,
        "data_snapshot_id": "factor-snapshot-test-002",
        "nas_task_id": "factor-research-test-002",
        "trial_kind": TrialKind.CONFIRMATORY,
        "parameter_locked_at": now,
        "blind_opened_at": now,
        "blind_invalidated": True,
        "submitted_at": now,
        "finished_at": now,
        "status": TrialStatus.COMPLETED,
        "decision": FactorResearchDecision.SIGNAL_VALID,
    }
    with pytest.raises(ValueError, match="INVALIDATED"):
        ResearchTrial(**fields)
