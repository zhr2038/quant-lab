CORE_MOMENTUM_ALPHA_ID = "v5.core.momentum"
RESEARCH_BASELINE_ROLE = "research_baseline"
STRATEGY_ALPHA_ROLE = "strategy_alpha"


def alpha_role(alpha_id: object) -> str:
    return (
        RESEARCH_BASELINE_ROLE
        if str(alpha_id or "") == CORE_MOMENTUM_ALPHA_ID
        else STRATEGY_ALPHA_ROLE
    )


def is_research_baseline_alpha(alpha_id: object) -> bool:
    return alpha_role(alpha_id) == RESEARCH_BASELINE_ROLE
