from __future__ import annotations

from dataclasses import dataclass

from quant_lab.research.factor_research.contracts import FactorResearchDecision


@dataclass(frozen=True)
class FactorDecisionEvidence:
    data_available: bool
    data_quality_pass: bool
    leakage_pass: bool
    duplicate_rejected: bool
    coverage: float
    development_rank_ic: float
    development_hac_tstat: float
    development_halves_same_sign: bool
    blind_rank_ic: float | None
    confirmatory_hac_tstat: float | None
    holm_adjusted_pvalue: float | None
    non_overlapping_direction_consistent: bool
    bootstrap_supports_direction: bool
    permutation_empirical_pvalue: float | None
    major_periods_same_sign_count: int
    max_symbol_contribution_share: float
    portfolio_validity: str
    edge_cost_ratio: float | None
    validation_net_return: float | None
    blind_net_return: float | None
    benchmark_not_worse: bool
    max_drawdown_pass: bool
    overfit_status: str
    pbo: float | None
    dsr_probability: float | None


@dataclass(frozen=True)
class FactorDecisionResult:
    decision: FactorResearchDecision
    signal_validity: str
    portfolio_validity: str
    deployment_readiness: str
    blockers: tuple[str, ...]


def decide_factor_research(evidence: FactorDecisionEvidence) -> FactorDecisionResult:
    if not evidence.data_available:
        return _result(FactorResearchDecision.DATA_BLOCKED, "UNKNOWN", "UNKNOWN")
    if not evidence.data_quality_pass:
        return _result(FactorResearchDecision.REJECTED_DATA_QUALITY, "FAIL", "UNKNOWN")
    if not evidence.leakage_pass:
        return _result(FactorResearchDecision.REJECTED_LEAKAGE, "FAIL", "UNKNOWN")
    if evidence.duplicate_rejected:
        return _result(FactorResearchDecision.REJECTED_DUPLICATE, "FAIL", "UNKNOWN")
    development_blockers = _development_blockers(evidence)
    if development_blockers:
        return FactorDecisionResult(
            decision=FactorResearchDecision.REJECTED_NO_SIGNAL,
            signal_validity="FAIL",
            portfolio_validity="UNKNOWN",
            deployment_readiness="BLOCKED",
            blockers=tuple(development_blockers),
        )
    if evidence.blind_rank_ic is None or evidence.confirmatory_hac_tstat is None:
        return FactorDecisionResult(
            decision=FactorResearchDecision.SIGNAL_CANDIDATE,
            signal_validity="CANDIDATE",
            portfolio_validity="UNKNOWN",
            deployment_readiness="BLOCKED_CONFIRMATORY_REQUIRED",
            blockers=("blind_confirmatory_evidence_missing",),
        )
    confirmatory_blockers = _confirmatory_blockers(evidence)
    if confirmatory_blockers:
        decision = (
            FactorResearchDecision.REJECTED_MULTIPLE_TESTING
            if "confirmatory_significance_failed" in confirmatory_blockers
            else FactorResearchDecision.REJECTED_NO_SIGNAL
        )
        return FactorDecisionResult(
            decision=decision,
            signal_validity="FAIL",
            portfolio_validity="UNKNOWN",
            deployment_readiness="BLOCKED",
            blockers=tuple(confirmatory_blockers),
        )
    if evidence.portfolio_validity != "PASS":
        return FactorDecisionResult(
            decision=FactorResearchDecision.PORTFOLIO_FAIL,
            signal_validity="PASS",
            portfolio_validity="FAIL",
            deployment_readiness="BLOCKED",
            blockers=("pre_registered_long_only_portfolio_failed",),
        )
    if evidence.overfit_status == "INCONCLUSIVE_OVERFIT_DIAGNOSTICS":
        return FactorDecisionResult(
            decision=FactorResearchDecision.INCONCLUSIVE_OVERFIT_DIAGNOSTICS,
            signal_validity="PASS",
            portfolio_validity="PASS",
            deployment_readiness="BLOCKED",
            blockers=("overfit_diagnostics_inconclusive",),
        )
    if evidence.overfit_status != "PASS":
        return FactorDecisionResult(
            decision=FactorResearchDecision.REJECTED_OVERFIT,
            signal_validity="PASS",
            portfolio_validity="PASS",
            deployment_readiness="BLOCKED",
            blockers=("pbo_or_dsr_failed",),
        )
    portfolio_blockers = _paper_blockers(evidence)
    if portfolio_blockers:
        return FactorDecisionResult(
            decision=FactorResearchDecision.PORTFOLIO_FAIL,
            signal_validity="PASS",
            portfolio_validity="FAIL",
            deployment_readiness="BLOCKED",
            blockers=tuple(portfolio_blockers),
        )
    return FactorDecisionResult(
        decision=FactorResearchDecision.PAPER_CANDIDATE,
        signal_validity="PASS",
        portfolio_validity="PASS",
        deployment_readiness="PAPER_REVIEW_ONLY",
        blockers=("manual_paper_review_required",),
    )


def _development_blockers(evidence: FactorDecisionEvidence) -> list[str]:
    blockers: list[str] = []
    if evidence.coverage < 0.80:
        blockers.append("coverage_below_0_80")
    if evidence.development_rank_ic <= 0.03:
        blockers.append("development_rank_ic_not_above_0_03")
    if evidence.development_hac_tstat < 2.0:
        blockers.append("development_hac_tstat_below_2")
    if not evidence.development_halves_same_sign:
        blockers.append("development_halves_sign_mismatch")
    return blockers


def _confirmatory_blockers(evidence: FactorDecisionEvidence) -> list[str]:
    blockers: list[str] = []
    if (evidence.blind_rank_ic or 0.0) <= 0.03:
        blockers.append("blind_rank_ic_not_above_0_03")
    hac_pass = (evidence.confirmatory_hac_tstat or 0.0) >= 3.0
    holm_pass = (evidence.holm_adjusted_pvalue or 1.0) <= 0.05
    if not (hac_pass or holm_pass):
        blockers.append("confirmatory_significance_failed")
    if not evidence.non_overlapping_direction_consistent:
        blockers.append("non_overlapping_direction_mismatch")
    if not evidence.bootstrap_supports_direction:
        blockers.append("bootstrap_interval_does_not_support_direction")
    if (evidence.permutation_empirical_pvalue or 1.0) > 0.05:
        blockers.append("permutation_empirical_pvalue_failed")
    if evidence.major_periods_same_sign_count < 2:
        blockers.append("fewer_than_two_major_periods_same_sign")
    if evidence.max_symbol_contribution_share > 0.50:
        blockers.append("single_symbol_driven")
    return blockers


def _paper_blockers(evidence: FactorDecisionEvidence) -> list[str]:
    blockers: list[str] = []
    if (evidence.edge_cost_ratio or 0.0) <= 1.5:
        blockers.append("edge_cost_ratio_not_above_1_5")
    if (evidence.validation_net_return or 0.0) <= 0:
        blockers.append("validation_portfolio_not_positive")
    if (evidence.blind_net_return or 0.0) <= 0:
        blockers.append("blind_portfolio_not_positive")
    if not evidence.benchmark_not_worse:
        blockers.append("portfolio_underperforms_benchmark")
    if not evidence.max_drawdown_pass:
        blockers.append("max_drawdown_failed")
    if (evidence.pbo if evidence.pbo is not None else 1.0) > 0.20:
        blockers.append("pbo_above_0_20")
    if (evidence.dsr_probability or 0.0) < 0.95:
        blockers.append("dsr_probability_below_0_95")
    return blockers


def _result(
    decision: FactorResearchDecision,
    signal_validity: str,
    portfolio_validity: str,
) -> FactorDecisionResult:
    return FactorDecisionResult(
        decision=decision,
        signal_validity=signal_validity,
        portfolio_validity=portfolio_validity,
        deployment_readiness="BLOCKED",
        blockers=(decision.value.lower(),),
    )
