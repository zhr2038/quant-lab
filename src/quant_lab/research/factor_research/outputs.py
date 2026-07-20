from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.factors.factory import (
    FACTOR_CANDIDATE_DATASET,
    FACTOR_DEFINITION_DATASET,
    FACTOR_EVIDENCE_DATASET,
    FACTOR_VALUE_DATASET,
)
from quant_lab.factors.factory import (
    FACTOR_CANDIDATE_SCHEMA as LEGACY_FACTOR_CANDIDATE_SCHEMA,
)
from quant_lab.factors.factory import (
    FACTOR_DEFINITION_SCHEMA as LEGACY_FACTOR_DEFINITION_SCHEMA,
)
from quant_lab.factors.factory import (
    FACTOR_EVIDENCE_SCHEMA as LEGACY_FACTOR_EVIDENCE_SCHEMA,
)
from quant_lab.factors.factory import (
    FACTOR_VALUE_SCHEMA as LEGACY_FACTOR_VALUE_SCHEMA,
)

FACTOR_ATTRIBUTION_DATASET = Path("gold") / "factor_attribution"
FACTOR_PORTFOLIO_VALIDATION_DATASET = Path("gold") / "factor_portfolio_validation"

FACTOR_RESEARCH_DEFINITION_SCHEMA: dict[str, Any] = {
    "hypothesis_id": pl.Utf8,
    "hypothesis_version": pl.Int64,
    "research_thread_id": pl.Utf8,
    "feature_recipe_id": pl.Utf8,
    **LEGACY_FACTOR_DEFINITION_SCHEMA,
    "research_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "automatic_promotion": pl.Boolean,
    "max_live_notional_usdt": pl.Float64,
}

FACTOR_RESEARCH_VALUE_SCHEMA: dict[str, Any] = {
    "hypothesis_id": pl.Utf8,
    "hypothesis_version": pl.Int64,
    "feature_recipe_id": pl.Utf8,
    "data_snapshot_id": pl.Utf8,
    **LEGACY_FACTOR_VALUE_SCHEMA,
    "research_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
}

FACTOR_RESEARCH_EVIDENCE_SCHEMA: dict[str, Any] = {
    "hypothesis_id": pl.Utf8,
    "hypothesis_version": pl.Int64,
    "trial_id": pl.Utf8,
    "test_family_id": pl.Utf8,
    "data_snapshot_id": pl.Utf8,
    **LEGACY_FACTOR_EVIDENCE_SCHEMA,
    "signal_validity": pl.Utf8,
    "portfolio_validity": pl.Utf8,
    "deployment_readiness": pl.Utf8,
    "attribution_type": pl.Utf8,
    "joint_residual_rank_ic": pl.Float64,
    "max_symbol_contribution_share": pl.Float64,
    "cost_evaluation_window_days": pl.Int64,
    "cost_evaluation_sample_count": pl.Int64,
    "cost_coverage": pl.Float64,
    "trusted_cost_coverage": pl.Float64,
    "public_proxy_cost_coverage": pl.Float64,
    "reconstructed_proxy_cost_coverage": pl.Float64,
    "bootstrap_cost_coverage": pl.Float64,
    "stale_cost_rate": pl.Float64,
    "pbo": pl.Float64,
    "dsr_probability": pl.Float64,
    "overfit_status": pl.Utf8,
    "research_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "automatic_promotion": pl.Boolean,
    "max_live_notional_usdt": pl.Float64,
}

FACTOR_RESEARCH_CANDIDATE_SCHEMA: dict[str, Any] = {
    "hypothesis_id": pl.Utf8,
    "hypothesis_version": pl.Int64,
    "data_snapshot_id": pl.Utf8,
    **LEGACY_FACTOR_CANDIDATE_SCHEMA,
    "signal_validity": pl.Utf8,
    "portfolio_validity": pl.Utf8,
    "deployment_readiness": pl.Utf8,
    "research_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "automatic_promotion": pl.Boolean,
    "max_live_notional_usdt": pl.Float64,
}

FACTOR_ATTRIBUTION_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "hypothesis_id": pl.Utf8,
    "hypothesis_version": pl.Int64,
    "trial_id": pl.Utf8,
    "factor_id": pl.Utf8,
    "horizon_bars": pl.Int64,
    "raw_rank_ic": pl.Float64,
    "symbol_fixed_effect_null_ic": pl.Float64,
    "beta_neutral_rank_ic": pl.Float64,
    "liquidity_neutral_rank_ic": pl.Float64,
    "momentum_neutral_rank_ic": pl.Float64,
    "joint_residual_rank_ic": pl.Float64,
    "incremental_ic": pl.Float64,
    "attribution_type": pl.Utf8,
    "max_symbol_contribution_share": pl.Float64,
    "dominant_symbol": pl.Utf8,
    "row_count": pl.Int64,
    "period_count": pl.Int64,
    "controls_available_json": pl.Utf8,
    "warnings_json": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
    "research_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
}

FACTOR_PORTFOLIO_VALIDATION_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "hypothesis_id": pl.Utf8,
    "hypothesis_version": pl.Int64,
    "trial_id": pl.Utf8,
    "factor_id": pl.Utf8,
    "horizon_bars": pl.Int64,
    "portfolio_rule_id": pl.Utf8,
    "period_count": pl.Int64,
    "gross_return": pl.Float64,
    "net_return": pl.Float64,
    "turnover": pl.Float64,
    "fees": pl.Float64,
    "slippage": pl.Float64,
    "edge_cost_ratio": pl.Float64,
    "sharpe": pl.Float64,
    "sortino": pl.Float64,
    "max_drawdown": pl.Float64,
    "calmar": pl.Float64,
    "beta": pl.Float64,
    "excess_vs_btc": pl.Float64,
    "excess_vs_universe": pl.Float64,
    "symbol_contribution_json": pl.Utf8,
    "concentration_hhi": pl.Float64,
    "max_symbol_contribution_share": pl.Float64,
    "average_cash_residual": pl.Float64,
    "cost_evaluation_window_days": pl.Int64,
    "cost_evaluation_sample_count": pl.Int64,
    "cost_coverage": pl.Float64,
    "trusted_cost_coverage": pl.Float64,
    "public_proxy_cost_coverage": pl.Float64,
    "reconstructed_proxy_cost_coverage": pl.Float64,
    "bootstrap_cost_coverage": pl.Float64,
    "stale_cost_rate": pl.Float64,
    "validation_net_return": pl.Float64,
    "blind_net_return": pl.Float64,
    "portfolio_validity": pl.Utf8,
    "deployment_readiness": pl.Utf8,
    "decision": pl.Utf8,
    "blockers_json": pl.Utf8,
    "pbo": pl.Float64,
    "dsr_probability": pl.Float64,
    "selection_degradation": pl.Float64,
    "overfit_status": pl.Utf8,
    "overfit_blockers_json": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
    "research_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "automatic_promotion": pl.Boolean,
    "max_live_notional_usdt": pl.Float64,
}


@dataclass(frozen=True)
class FactorResearchOutputSpec:
    dataset_name: str
    relative_path: Path
    schema: dict[str, pl.DataType]
    primary_keys: tuple[str, ...]
    window_keys: tuple[str, ...]
    publish_mode: str = "window_replace"
    empty_result_semantics: str = "clear_window"


FACTOR_RESEARCH_OUTPUT_SPECS = (
    FactorResearchOutputSpec(
        "factor_definition",
        FACTOR_DEFINITION_DATASET,
        FACTOR_RESEARCH_DEFINITION_SCHEMA,
        ("hypothesis_id", "hypothesis_version", "feature_recipe_id"),
        ("source",),
    ),
    FactorResearchOutputSpec(
        "factor_value",
        FACTOR_VALUE_DATASET,
        FACTOR_RESEARCH_VALUE_SCHEMA,
        ("factor_id", "symbol", "ts"),
        ("data_snapshot_id",),
    ),
    FactorResearchOutputSpec(
        "factor_evidence",
        FACTOR_EVIDENCE_DATASET,
        FACTOR_RESEARCH_EVIDENCE_SCHEMA,
        ("as_of_date", "trial_id"),
        ("as_of_date",),
    ),
    FactorResearchOutputSpec(
        "factor_attribution",
        FACTOR_ATTRIBUTION_DATASET,
        FACTOR_ATTRIBUTION_SCHEMA,
        ("as_of_date", "trial_id"),
        ("as_of_date",),
    ),
    FactorResearchOutputSpec(
        "factor_portfolio_validation",
        FACTOR_PORTFOLIO_VALIDATION_DATASET,
        FACTOR_PORTFOLIO_VALIDATION_SCHEMA,
        ("as_of_date", "trial_id"),
        ("as_of_date",),
    ),
    FactorResearchOutputSpec(
        "factor_candidate",
        FACTOR_CANDIDATE_DATASET,
        FACTOR_RESEARCH_CANDIDATE_SCHEMA,
        ("as_of_date", "factor_id"),
        ("as_of_date",),
    ),
)

FACTOR_RESEARCH_OUTPUT_SPEC_BY_NAME = {
    spec.dataset_name: spec for spec in FACTOR_RESEARCH_OUTPUT_SPECS
}
