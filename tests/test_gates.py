from datetime import UTC, datetime

import pytest

from quant_lab.contracts.models import AlphaEvidence, GateStatus
from quant_lab.gates.defaults import DEFAULT_GATE_VERSION, evaluate_alpha_gate


def evidence(**overrides):
    values = {
        "alpha_id": "alpha-test",
        "version": "v1",
        "data_version": "data-v1",
        "feature_version": "feature-v1",
        "cost_model_version": "cost-v1",
        "universe_id": "okx-spot-major",
        "start_ts": datetime(2026, 1, 1, tzinfo=UTC),
        "end_ts": datetime(2026, 5, 10, tzinfo=UTC),
        "coverage": 0.99,
        "ic_mean": 0.03,
        "ic_tstat": 2.5,
        "rank_ic_mean": 0.04,
        "rank_ic_tstat": 2.8,
        "edge_cost_ratio": 2.0,
        "oos_sharpe": 1.0,
        "oos_sortino": 1.2,
        "oos_cagr": 0.18,
        "oos_max_drawdown": 0.12,
        "profit_factor": 1.3,
        "turnover": 0.4,
        "cost_ratio": 0.35,
        "profitable_folds_ratio": 0.75,
        "train_oos_decay": 0.25,
        "pbo_score": 0.2,
        "paper_days": 20,
        "paper_slippage_coverage": 0.9,
        "created_at": datetime(2026, 5, 10, tzinfo=UTC),
    }
    values.update(overrides)
    return AlphaEvidence(**values)


@pytest.mark.parametrize(
    ("overrides", "expected_status", "expected_reason"),
    [
        ({"coverage": 0.94}, GateStatus.DEAD, "insufficient_coverage"),
        ({"ic_mean": -0.01}, GateStatus.DEAD, "non_positive_ic"),
        ({"ic_tstat": 1.99}, GateStatus.DEAD, "weak_ic_tstat"),
        ({"edge_cost_ratio": 1.49}, GateStatus.DEAD, "cost_exceeds_edge"),
        ({"oos_sharpe": 0.79}, GateStatus.QUARANTINE, "weak_oos_sharpe"),
        ({"oos_max_drawdown": 0.21}, GateStatus.QUARANTINE, "excessive_drawdown"),
        ({"profitable_folds_ratio": 0.59}, GateStatus.QUARANTINE, "unstable_folds"),
        ({"train_oos_decay": 0.51}, GateStatus.QUARANTINE, "train_oos_decay"),
        ({"paper_days": 13}, GateStatus.PAPER_READY, "needs_paper_observation"),
        (
            {"paper_slippage_coverage": 0.79},
            GateStatus.PAPER_READY,
            "insufficient_paper_slippage_coverage",
        ),
    ],
)
def test_default_gate_decisions(overrides, expected_status, expected_reason):
    decision = evaluate_alpha_gate(evidence(**overrides))

    assert decision.status == expected_status
    assert decision.passed is False
    assert expected_reason in decision.reasons
    assert decision.alpha_id == "alpha-test"
    assert decision.version == "v1"
    assert decision.gate_version == DEFAULT_GATE_VERSION
    assert decision.created_at == datetime(2026, 5, 10, tzinfo=UTC)
    assert decision.metrics["coverage"] == overrides.get("coverage", 0.99)


def test_strong_evidence_is_live_ready():
    decision = evaluate_alpha_gate(evidence())

    assert decision.status == GateStatus.LIVE_READY
    assert decision.passed is True
    assert decision.reasons == ["all_default_gates_passed"]
    assert decision.next_action == "eligible_for_strategy_consumer_review"


def test_insufficient_samples_quarantines_without_faking_negative_alpha():
    decision = evaluate_alpha_gate(
        evidence(evidence_status="insufficient_samples", ic_mean=0.08, ic_tstat=5.0)
    )

    assert decision.status == GateStatus.QUARANTINE
    assert decision.reasons == ["insufficient_samples"]
    assert decision.metrics["ic_mean"] == 0.08
    assert decision.metrics["ic_tstat"] == 5.0
