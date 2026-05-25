import os
from datetime import UTC, datetime
from typing import Any

from quant_lab.contracts.models import AlphaEvidence, GateDecision, GateStatus

DEFAULT_GATE_VERSION = "default-v0.1"
EXAMPLE_GATE_VERSION = "example-conservative-v0.1"


def conservative_example_gate_decision() -> GateDecision:
    """Return a fail-closed example response for docs and smoke checks."""
    return GateDecision(
        alpha_id="example-alpha",
        version="example",
        gate_version=EXAMPLE_GATE_VERSION,
        status=GateStatus.QUARANTINE,
        passed=False,
        reasons=["example_not_live_ready_evidence"],
        metrics={},
        next_action="replace_example_with_real_alpha_evidence",
        created_at=datetime.now(UTC),
    )


def evaluate_alpha_gate(evidence: AlphaEvidence) -> GateDecision:
    if evidence.evidence_status in {"insufficient_data", "insufficient_samples", "stale"}:
        status = _insufficient_samples_status()
        reason = evidence.evidence_status
        return _decision(
            evidence,
            status=status,
            reasons=[reason],
            next_action=(
                "collect_more_research_samples"
                if status == GateStatus.QUARANTINE
                else "retire_alpha_or_expand_universe"
            ),
        )

    dead_reasons = _dead_reasons(evidence)
    if dead_reasons:
        return _decision(
            evidence,
            status=GateStatus.DEAD,
            reasons=dead_reasons,
            next_action="retire_alpha_or_research_new_hypothesis",
        )

    quarantine_reasons = _quarantine_reasons(evidence)
    if quarantine_reasons:
        return _decision(
            evidence,
            status=GateStatus.QUARANTINE,
            reasons=quarantine_reasons,
            next_action="keep_out_of_paper_and_collect_more_oos_evidence",
        )

    paper_reasons = _paper_ready_reasons(evidence)
    if paper_reasons:
        return _decision(
            evidence,
            status=GateStatus.PAPER_READY,
            reasons=paper_reasons,
            next_action="run_or_continue_paper_observation",
        )

    return _decision(
        evidence,
        status=GateStatus.LIVE_READY,
        reasons=["all_default_gates_passed"],
        next_action="eligible_for_strategy_consumer_review",
    )


def _dead_reasons(evidence: AlphaEvidence) -> list[str]:
    reasons: list[str] = []
    if evidence.coverage < 0.95:
        reasons.append("insufficient_coverage")
    if evidence.ic_mean <= 0:
        reasons.append("non_positive_ic")
    if evidence.ic_tstat < 2.0:
        reasons.append("weak_ic_tstat")
    if evidence.edge_cost_ratio < 1.5:
        reasons.append("cost_exceeds_edge")
    return reasons


def _quarantine_reasons(evidence: AlphaEvidence) -> list[str]:
    reasons: list[str] = []
    if evidence.oos_sharpe < 0.8:
        reasons.append("weak_oos_sharpe")
    if evidence.oos_max_drawdown > 0.20:
        reasons.append("excessive_drawdown")
    if evidence.profitable_folds_ratio < 0.60:
        reasons.append("unstable_folds")
    if evidence.train_oos_decay > 0.50:
        reasons.append("train_oos_decay")
    return reasons


def _paper_ready_reasons(evidence: AlphaEvidence) -> list[str]:
    reasons: list[str] = []
    if evidence.paper_days < 14:
        reasons.append("needs_paper_observation")
    if evidence.paper_slippage_coverage < 0.80:
        reasons.append("insufficient_paper_slippage_coverage")
    return reasons


def _insufficient_samples_status() -> GateStatus:
    configured = os.environ.get(
        "QUANT_LAB_INSUFFICIENT_SAMPLES_GATE_STATUS",
        GateStatus.QUARANTINE.value,
    ).strip().upper()
    if configured == GateStatus.DEAD.value:
        return GateStatus.DEAD
    return GateStatus.QUARANTINE


def _decision(
    evidence: AlphaEvidence,
    status: GateStatus,
    reasons: list[str],
    next_action: str,
) -> GateDecision:
    return GateDecision(
        alpha_id=evidence.alpha_id,
        version=evidence.version,
        gate_version=DEFAULT_GATE_VERSION,
        status=status,
        passed=status == GateStatus.LIVE_READY,
        reasons=reasons,
        metrics=_metrics(evidence),
        next_action=next_action,
        created_at=evidence.created_at,
    )


def _metrics(evidence: AlphaEvidence) -> dict[str, Any]:
    return {
        "coverage": evidence.coverage,
        "ic_mean": evidence.ic_mean,
        "ic_tstat": evidence.ic_tstat,
        "rank_ic_mean": evidence.rank_ic_mean,
        "rank_ic_tstat": evidence.rank_ic_tstat,
        "edge_cost_ratio": evidence.edge_cost_ratio,
        "oos_sharpe": evidence.oos_sharpe,
        "oos_sortino": evidence.oos_sortino,
        "oos_cagr": evidence.oos_cagr,
        "oos_max_drawdown": evidence.oos_max_drawdown,
        "profit_factor": evidence.profit_factor,
        "turnover": evidence.turnover,
        "cost_ratio": evidence.cost_ratio,
        "profitable_folds_ratio": evidence.profitable_folds_ratio,
        "train_oos_decay": evidence.train_oos_decay,
        "pbo_score": evidence.pbo_score,
        "paper_days": evidence.paper_days,
        "paper_slippage_coverage": evidence.paper_slippage_coverage,
        "evidence_status": evidence.evidence_status,
    }
