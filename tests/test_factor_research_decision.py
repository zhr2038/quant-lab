from quant_lab.research.factor_research.contracts import FactorResearchDecision
from quant_lab.research.factor_research.decision import (
    FactorDecisionEvidence,
    decide_factor_research,
)


def _evidence(**updates: object) -> FactorDecisionEvidence:
    values: dict[str, object] = {
        "data_available": True,
        "data_quality_pass": True,
        "leakage_pass": True,
        "duplicate_rejected": False,
        "coverage": 0.95,
        "development_rank_ic": 0.05,
        "development_hac_tstat": 2.5,
        "development_halves_same_sign": True,
        "blind_rank_ic": 0.05,
        "confirmatory_hac_tstat": 3.5,
        "holm_adjusted_pvalue": 0.01,
        "non_overlapping_direction_consistent": True,
        "bootstrap_supports_direction": True,
        "permutation_empirical_pvalue": 0.01,
        "major_periods_same_sign_count": 2,
        "max_symbol_contribution_share": 0.30,
        "portfolio_validity": "PASS",
        "edge_cost_ratio": 2.0,
        "validation_net_return": 0.05,
        "blind_net_return": 0.03,
        "benchmark_not_worse": True,
        "max_drawdown_pass": True,
        "overfit_status": "PASS",
        "pbo": 0.10,
        "dsr_probability": 0.97,
    }
    values.update(updates)
    return FactorDecisionEvidence(**values)  # type: ignore[arg-type]


def test_development_evidence_never_skips_confirmatory_stage() -> None:
    result = decide_factor_research(
        _evidence(blind_rank_ic=None, confirmatory_hac_tstat=None)
    )

    assert result.decision == FactorResearchDecision.SIGNAL_CANDIDATE
    assert result.deployment_readiness == "BLOCKED_CONFIRMATORY_REQUIRED"


def test_signal_pass_and_long_only_failure_is_portfolio_fail() -> None:
    result = decide_factor_research(_evidence(portfolio_validity="FAIL"))

    assert result.decision == FactorResearchDecision.PORTFOLIO_FAIL
    assert result.signal_validity == "PASS"
    assert result.portfolio_validity == "FAIL"


def test_inconclusive_overfit_diagnostics_cannot_be_paper_candidate() -> None:
    result = decide_factor_research(
        _evidence(
            overfit_status="INCONCLUSIVE_OVERFIT_DIAGNOSTICS",
            pbo=None,
            dsr_probability=None,
        )
    )

    assert result.decision == FactorResearchDecision.INCONCLUSIVE_OVERFIT_DIAGNOSTICS


def test_only_complete_evidence_reaches_manual_paper_candidate() -> None:
    result = decide_factor_research(_evidence())

    assert result.decision == FactorResearchDecision.PAPER_CANDIDATE
    assert result.deployment_readiness == "PAPER_REVIEW_ONLY"
    assert "manual_paper_review_required" in result.blockers
