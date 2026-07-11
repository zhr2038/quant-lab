"""Versioned, paper-only strategy contracts and research evidence helpers."""

from quant_lab.paper.contracts import (
    PAPER_STRATEGY_CONTRACT_VERSION,
    LifecycleState,
    PaperRule,
    PaperStrategyAck,
    PaperStrategyProposal,
    assert_lifecycle_transition,
    legacy_lifecycle_state,
)

__all__ = [
    "PAPER_STRATEGY_CONTRACT_VERSION",
    "LifecycleState",
    "PaperRule",
    "PaperStrategyAck",
    "PaperStrategyProposal",
    "assert_lifecycle_transition",
    "legacy_lifecycle_state",
]
