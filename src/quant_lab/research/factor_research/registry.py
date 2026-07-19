from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset
from quant_lab.research.factor_research.contracts import (
    BenchmarkPlan,
    CostModelPlan,
    DataRequirement,
    DiscoverySource,
    FactorResearchDecision,
    FeatureRecipe,
    HypothesisStatus,
    NeutralizationPlan,
    RecipeParameter,
    ResearchFamily,
    ResearchHypothesis,
    ResearchTrial,
    StrategyFit,
    TrialKind,
    TrialStatus,
    UniverseDefinition,
    build_trial_id,
    canonical_json,
    utc_midnight,
)

RESEARCH_HYPOTHESIS_REGISTRY_DATASET = Path("gold") / "research_hypothesis_registry"
RESEARCH_TRIAL_LEDGER_DATASET = Path("gold") / "research_trial_ledger"
FACTOR_RETIREMENT_DATASET = Path("gold") / "factor_retirement"

MAX_ACTIVE_HYPOTHESES_PER_FAMILY = 2
MAX_ACTIVE_HYPOTHESES = 6
MAX_TRIALS_PER_HYPOTHESIS = 9
MAX_RESEARCH_TRIALS = 54

EXECUTABLE_HYPOTHESIS_STATUSES = frozenset(
    {
        HypothesisStatus.APPROVED_FOR_RESEARCH,
        HypothesisStatus.RUNNING,
        HypothesisStatus.SIGNAL_VALID,
        HypothesisStatus.PORTFOLIO_FAIL,
        HypothesisStatus.PAPER_CANDIDATE,
    }
)

HYPOTHESIS_REGISTRY_SCHEMA: dict[str, Any] = {
    "hypothesis_id": pl.Utf8,
    "hypothesis_version": pl.Int64,
    "research_thread_id": pl.Utf8,
    "title": pl.Utf8,
    "factor_family": pl.Utf8,
    "economic_mechanism": pl.Utf8,
    "who_pays_the_edge": pl.Utf8,
    "why_edge_may_persist": pl.Utf8,
    "why_edge_may_decay": pl.Utf8,
    "strategy_fit_json": pl.Utf8,
    "data_requirements_json": pl.Utf8,
    "available_data_confirmed": pl.Boolean,
    "blocked_missing_data_json": pl.Utf8,
    "feature_recipes_json": pl.Utf8,
    "expected_direction": pl.Int64,
    "expected_horizons_json": pl.Utf8,
    "allowed_variants_json": pl.Utf8,
    "variant_budget": pl.Int64,
    "universe_definition_json": pl.Utf8,
    "neutralization_plan_json": pl.Utf8,
    "benchmark_plan_json": pl.Utf8,
    "cost_model_json": pl.Utf8,
    "falsification_conditions_json": pl.Utf8,
    "stopping_conditions_json": pl.Utf8,
    "success_conditions_json": pl.Utf8,
    "known_overlap_json": pl.Utf8,
    "related_hypotheses_json": pl.Utf8,
    "related_existing_factors_json": pl.Utf8,
    "discovery_source": pl.Utf8,
    "status": pl.Utf8,
    "approved_by": pl.Utf8,
    "approved_at": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
    "updated_at": pl.Datetime(time_zone="UTC"),
    "definition_digest": pl.Utf8,
    "research_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "automatic_promotion": pl.Boolean,
    "max_live_notional_usdt": pl.Float64,
}

TRIAL_LEDGER_SCHEMA: dict[str, Any] = {
    "trial_id": pl.Utf8,
    "hypothesis_id": pl.Utf8,
    "hypothesis_version": pl.Int64,
    "test_family_id": pl.Utf8,
    "factor_formula_hash": pl.Utf8,
    "feature_recipe_hash": pl.Utf8,
    "direction": pl.Int64,
    "lookback": pl.Int64,
    "horizon": pl.Int64,
    "universe_id": pl.Utf8,
    "neutralization_id": pl.Utf8,
    "cost_model_id": pl.Utf8,
    "portfolio_rule_id": pl.Utf8,
    "split_definition": pl.Utf8,
    "blind_period_id": pl.Utf8,
    "random_seed": pl.Int64,
    "code_commit": pl.Utf8,
    "data_snapshot_id": pl.Utf8,
    "nas_task_id": pl.Utf8,
    "trial_kind": pl.Utf8,
    "parameter_locked_at": pl.Datetime(time_zone="UTC"),
    "blind_opened_at": pl.Datetime(time_zone="UTC"),
    "blind_invalidated": pl.Boolean,
    "submitted_at": pl.Datetime(time_zone="UTC"),
    "started_at": pl.Datetime(time_zone="UTC"),
    "finished_at": pl.Datetime(time_zone="UTC"),
    "status": pl.Utf8,
    "decision": pl.Utf8,
    "failure_reason": pl.Utf8,
    "counts_toward_multiple_testing": pl.Boolean,
    "identity_digest": pl.Utf8,
    "research_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
    "automatic_promotion": pl.Boolean,
    "max_live_notional_usdt": pl.Float64,
}

FACTOR_RETIREMENT_SCHEMA: dict[str, Any] = {
    "factor_id": pl.Utf8,
    "factor_family": pl.Utf8,
    "status": pl.Utf8,
    "enabled": pl.Boolean,
    "retirement_reason": pl.Utf8,
    "signal_validity": pl.Utf8,
    "portfolio_validity": pl.Utf8,
    "source_audit": pl.Utf8,
    "recorded_at": pl.Datetime(time_zone="UTC"),
    "research_only": pl.Boolean,
    "live_order_effect": pl.Utf8,
}


def default_hypothesis_registry() -> list[ResearchHypothesis]:
    recorded_at = datetime(2026, 7, 19, 4, 2, tzinfo=UTC)
    market_universe = UniverseDefinition(
        universe_id="spot-dynamic-quality-v1",
        dynamic_source_dataset="gold/expanded_universe_quality",
        inclusion_rule="quality-qualified spot symbols with closed 1H market bars",
        exclusion_rule="exclude stale, untradeable, or insufficient-history symbols",
    )
    common_neutralization = NeutralizationPlan(
        neutralization_id="xs-core-controls-v1",
        controls=(
            "symbol_fixed_effect",
            "market_beta",
            "liquidity",
            "long_run_volatility",
            "momentum",
            "regime",
        ),
    )
    benchmarks = BenchmarkPlan(
        benchmark_id="spot-btc-and-dynamic-ew-v1",
        benchmarks=("BTC", "DYNAMIC_UNIVERSE_EQUAL_WEIGHT"),
    )
    costs = CostModelPlan(cost_model_id="research-point-in-time-p75-v1")
    common_conditions = {
        "falsification_conditions": (
            "blind rank IC is not positive after multiple-testing adjustment",
            "joint residual rank IC loses sign",
            "performance is driven by one symbol",
        ),
        "stopping_conditions": (
            "research budget is exhausted",
            "required point-in-time data becomes unavailable",
            "leakage or provenance validation fails",
        ),
        "success_conditions": (
            "confirmatory HAC t-stat is at least 3 or Holm-adjusted p is at most 0.05",
            "non-overlapping and bootstrap evidence support the same sign",
            "pre-registered long-only portfolio is positive after cost",
        ),
    }
    return [
        ResearchHypothesis(
            hypothesis_id="defensive.low_vol_decomposition",
            hypothesis_version=1,
            research_thread_id="factor-v2.defensive.low-vol",
            title="Structural versus dynamic low-volatility effect",
            factor_family=ResearchFamily.DEFENSIVE_QUALITY,
            economic_mechanism=(
                "Separate persistent low-risk asset identity from a recent decline in each asset's "
                "own volatility before evaluating either as a return predictor."
            ),
            who_pays_the_edge="risk-seeking participants who overpay for volatile crypto exposure",
            why_edge_may_persist="lottery preference and leverage constraints can be persistent",
            why_edge_may_decay=(
                "regime shifts and exchange-universe selection can erase the premium"
            ),
            strategy_fit=(StrategyFit.CROSS_SECTIONAL_RANK, StrategyFit.LONG_ONLY_SPOT),
            data_requirements=(
                DataRequirement(
                    dataset_name="silver/market_bar",
                    required_columns=("symbol", "timeframe", "ts", "close", "is_closed"),
                    min_history_days=730,
                ),
                DataRequirement(
                    dataset_name="gold/expanded_universe_quality",
                    required_columns=("symbol", "as_of_date"),
                    min_history_days=30,
                ),
                DataRequirement(
                    dataset_name="gold/cost_bucket_daily",
                    required_columns=("symbol", "day", "total_cost_bps_p75"),
                    min_history_days=30,
                ),
            ),
            available_data_confirmed=True,
            feature_recipes=(
                FeatureRecipe(
                    recipe_id="structural_low_vol_20d",
                    operation="rolling_volatility",
                    source_fields=("close",),
                    parameters=(RecipeParameter(name="lookback_bars", value="480"),),
                ),
                FeatureRecipe(
                    recipe_id="dynamic_low_vol_change_20d",
                    operation="volatility_change",
                    source_fields=("close",),
                    parameters=(
                        RecipeParameter(name="current_lookback_bars", value="480"),
                        RecipeParameter(name="baseline_lookback_bars", value="480"),
                    ),
                ),
            ),
            expected_direction=-1,
            expected_horizons=(24, 72),
            allowed_variants=("structural_low_vol_20d", "dynamic_low_vol_change_20d"),
            variant_budget=2,
            universe_definition=market_universe,
            neutralization_plan=common_neutralization,
            benchmark_plan=benchmarks,
            cost_model=costs,
            known_overlap=("low_vol_20d", "beta", "liquidity", "symbol_fixed_effect"),
            related_existing_factors=("low_vol_20d",),
            discovery_source=DiscoverySource.AUDIT_IMPORT,
            status=HypothesisStatus.APPROVED_FOR_RESEARCH,
            approved_by="migration_audit_20260719",
            approved_at=recorded_at,
            created_at=recorded_at,
            updated_at=recorded_at,
            **common_conditions,
        ),
        ResearchHypothesis(
            hypothesis_id="timing.market_breadth",
            hypothesis_version=1,
            research_thread_id="factor-v2.timing.market-breadth",
            title="Market breadth as absolute spot timing",
            factor_family=ResearchFamily.ABSOLUTE_TIMING_BREADTH,
            economic_mechanism=(
                "Broad participation may identify when long-only spot exposure is compensated, "
                "independently of cross-sectional coin ranking."
            ),
            who_pays_the_edge="late trend participants entering after breadth has already improved",
            why_edge_may_persist="slow diffusion across a fragmented crypto universe",
            why_edge_may_decay="rapid common-factor repricing can make breadth contemporaneous",
            strategy_fit=(StrategyFit.ABSOLUTE_TIMING, StrategyFit.LONG_ONLY_SPOT),
            data_requirements=(
                DataRequirement(
                    dataset_name="silver/market_bar",
                    required_columns=("symbol", "timeframe", "ts", "close", "is_closed"),
                    min_history_days=730,
                ),
                DataRequirement(
                    dataset_name="gold/market_regime_daily",
                    required_columns=("symbol", "day"),
                    min_history_days=365,
                ),
            ),
            available_data_confirmed=True,
            feature_recipes=(
                FeatureRecipe(
                    recipe_id="advance_ratio_24h",
                    operation="market_breadth",
                    source_fields=("close",),
                    parameters=(RecipeParameter(name="return_lookback_bars", value="24"),),
                ),
                FeatureRecipe(
                    recipe_id="cross_sectional_dispersion_24h",
                    operation="market_dispersion",
                    source_fields=("close",),
                    parameters=(RecipeParameter(name="return_lookback_bars", value="24"),),
                ),
            ),
            expected_direction=1,
            expected_horizons=(24, 72),
            allowed_variants=("advance_ratio_24h", "cross_sectional_dispersion_24h"),
            variant_budget=2,
            universe_definition=market_universe,
            neutralization_plan=common_neutralization,
            benchmark_plan=benchmarks,
            cost_model=costs,
            known_overlap=("BTC trend", "market beta", "market regime"),
            related_hypotheses=("defensive.low_vol_decomposition",),
            discovery_source=DiscoverySource.HUMAN,
            status=HypothesisStatus.APPROVED_FOR_RESEARCH,
            approved_by="migration_audit_20260719",
            approved_at=recorded_at,
            created_at=recorded_at,
            updated_at=recorded_at,
            **common_conditions,
        ),
        _data_blocked_hypothesis(
            hypothesis_id="derivatives.funding_crowding",
            research_thread_id="factor-v2.derivatives.funding",
            title="Derivatives crowding and price-funding divergence",
            family=ResearchFamily.DERIVATIVES_CROWDING,
            operation="funding_crowding",
            required_dataset="silver/funding_rate",
            required_columns=("symbol", "ts", "funding_rate"),
            missing=("audited multi-year funding history", "basis", "open interest"),
            recorded_at=recorded_at,
        ),
        _data_blocked_hypothesis(
            hypothesis_id="microstructure.real_liquidity",
            research_thread_id="factor-v2.microstructure.liquidity",
            title="Real spread, depth and aggressive-flow liquidity",
            family=ResearchFamily.LIQUIDITY_MICROSTRUCTURE,
            operation="microstructure_liquidity",
            required_dataset="silver/orderbook_snapshot",
            required_columns=("symbol", "ts", "bid", "ask"),
            missing=("audited depth history", "order-book imbalance", "aggressive flow"),
            recorded_at=recorded_at,
        ),
    ]


def validate_hypothesis_budget(hypotheses: Iterable[ResearchHypothesis]) -> None:
    materialized = list(hypotheses)
    keys = [(item.hypothesis_id, item.hypothesis_version) for item in materialized]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate hypothesis identity")
    active = [item for item in materialized if item.status in EXECUTABLE_HYPOTHESIS_STATUSES]
    if len(active) > MAX_ACTIVE_HYPOTHESES:
        raise ValueError("RESEARCH_BUDGET_EXCEEDED:active_hypotheses")
    family_counts: dict[ResearchFamily, int] = {}
    for item in active:
        family_counts[item.factor_family] = family_counts.get(item.factor_family, 0) + 1
    exceeded = [family.value for family, count in family_counts.items() if count > 2]
    if exceeded:
        raise ValueError(f"RESEARCH_BUDGET_EXCEEDED:families:{','.join(sorted(exceeded))}")


def hypothesis_registry_frame(hypotheses: Iterable[ResearchHypothesis]) -> pl.DataFrame:
    """Return the canonical, schema-locked representation of the control registry."""
    materialized = list(hypotheses)
    validate_hypothesis_budget(materialized)
    return _schema_frame(
        [_hypothesis_row(item) for item in materialized],
        HYPOTHESIS_REGISTRY_SCHEMA,
    ).sort(["hypothesis_id", "hypothesis_version"])


def hypotheses_from_registry(frame: pl.DataFrame) -> list[ResearchHypothesis]:
    """Parse the authoritative Gold registry without inventing missing control fields."""
    if frame.is_empty():
        return []
    missing = sorted(set(HYPOTHESIS_REGISTRY_SCHEMA) - set(frame.columns))
    if missing:
        raise ValueError(f"hypothesis registry missing columns: {','.join(missing)}")
    hypotheses: list[ResearchHypothesis] = []
    for row in frame.select(list(HYPOTHESIS_REGISTRY_SCHEMA)).to_dicts():
        if (
            row.get("research_only") is not True
            or row.get("live_order_effect") != "none"
            or row.get("automatic_promotion") is not False
            or float(row.get("max_live_notional_usdt") or 0.0) != 0.0
        ):
            raise ValueError("hypothesis registry violates research-only safety boundary")
        hypothesis = ResearchHypothesis(
            hypothesis_id=row["hypothesis_id"],
            hypothesis_version=row["hypothesis_version"],
            research_thread_id=row["research_thread_id"],
            title=row["title"],
            factor_family=row["factor_family"],
            economic_mechanism=row["economic_mechanism"],
            who_pays_the_edge=row["who_pays_the_edge"],
            why_edge_may_persist=row["why_edge_may_persist"],
            why_edge_may_decay=row["why_edge_may_decay"],
            strategy_fit=_json_value(row, "strategy_fit_json"),
            data_requirements=_json_value(row, "data_requirements_json"),
            available_data_confirmed=row["available_data_confirmed"],
            blocked_missing_data=_json_value(row, "blocked_missing_data_json"),
            feature_recipes=_json_value(row, "feature_recipes_json"),
            expected_direction=row["expected_direction"],
            expected_horizons=_json_value(row, "expected_horizons_json"),
            allowed_variants=_json_value(row, "allowed_variants_json"),
            variant_budget=row["variant_budget"],
            universe_definition=_json_value(row, "universe_definition_json"),
            neutralization_plan=_json_value(row, "neutralization_plan_json"),
            benchmark_plan=_json_value(row, "benchmark_plan_json"),
            cost_model=_json_value(row, "cost_model_json"),
            falsification_conditions=_json_value(row, "falsification_conditions_json"),
            stopping_conditions=_json_value(row, "stopping_conditions_json"),
            success_conditions=_json_value(row, "success_conditions_json"),
            known_overlap=_json_value(row, "known_overlap_json"),
            related_hypotheses=_json_value(row, "related_hypotheses_json"),
            related_existing_factors=_json_value(row, "related_existing_factors_json"),
            discovery_source=row["discovery_source"],
            status=row["status"],
            approved_by=row.get("approved_by"),
            approved_at=row.get("approved_at"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        if hypothesis.definition_digest != row["definition_digest"]:
            raise ValueError("hypothesis registry definition digest mismatch")
        hypotheses.append(hypothesis)
    validate_hypothesis_budget(hypotheses)
    return hypotheses


def prepare_factor_research_control_state(existing: pl.DataFrame) -> pl.DataFrame:
    """Seed defaults only on an empty lake; otherwise preserve operator control state."""
    if existing.is_empty():
        return hypothesis_registry_frame(default_hypothesis_registry())
    hypotheses = hypotheses_from_registry(existing)
    active_versions: dict[str, int] = {}
    for item in hypotheses:
        if item.status not in EXECUTABLE_HYPOTHESIS_STATUSES:
            continue
        if item.hypothesis_id in active_versions:
            raise ValueError("multiple active versions for one research hypothesis")
        active_versions[item.hypothesis_id] = item.hypothesis_version
    return hypothesis_registry_frame(hypotheses)


def hypothesis_registry_digest(frame: pl.DataFrame) -> str:
    canonical = prepare_factor_research_control_state(frame)
    return _frame_digest(canonical, sort_by=["hypothesis_id", "hypothesis_version"])


def trial_ledger_frame(trials: Iterable[ResearchTrial]) -> pl.DataFrame:
    materialized = list(trials)
    ids = [item.trial_id for item in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trial_id")
    if len(materialized) > MAX_RESEARCH_TRIALS:
        raise ValueError("RESEARCH_BUDGET_EXCEEDED:trials")
    return _schema_frame([_trial_row(item) for item in materialized], TRIAL_LEDGER_SCHEMA).sort(
        "trial_id"
    )


def trial_ledger_digest(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        raise ValueError("factor research trial ledger must not be empty")
    missing = sorted(set(TRIAL_LEDGER_SCHEMA) - set(frame.columns))
    if missing:
        raise ValueError(f"trial ledger missing columns: {','.join(missing)}")
    return _frame_digest(frame.select(list(TRIAL_LEDGER_SCHEMA)), sort_by=["trial_id"])


def trials_from_ledger(frame: pl.DataFrame) -> list[ResearchTrial]:
    if frame.is_empty():
        return []
    missing = sorted(set(TRIAL_LEDGER_SCHEMA) - set(frame.columns))
    if missing:
        raise ValueError(f"trial ledger missing columns: {','.join(missing)}")
    model_fields = set(ResearchTrial.model_fields)
    trials: list[ResearchTrial] = []
    for row in frame.select(list(TRIAL_LEDGER_SCHEMA)).to_dicts():
        trial = ResearchTrial.model_validate(
            {key: value for key, value in row.items() if key in model_fields}
        )
        if trial.identity_digest != row["identity_digest"]:
            raise ValueError("trial ledger identity digest mismatch")
        trials.append(trial)
    return sorted(trials, key=lambda item: item.trial_id)


def plan_factor_research_trials(
    hypotheses: Iterable[ResearchHypothesis],
    *,
    start_date: date,
    end_date: date,
    code_commit: str,
    data_snapshot_id: str,
    nas_task_id: str,
) -> list[ResearchTrial]:
    """Pre-register the bounded variant/horizon grid before NAS sees any outcomes."""
    materialized = list(hypotheses)
    validate_hypothesis_budget(materialized)
    executable = [item for item in materialized if item.status in EXECUTABLE_HYPOTHESIS_STATUSES]
    if not executable:
        raise ValueError("no approved factor research hypotheses")
    locked_at = utc_midnight(end_date)
    split_definition = (
        f"chronological_v1:start={start_date.isoformat()};end={end_date.isoformat()};"
        "research=60%;validation=20%;blind=20%;embargo=max_horizon"
    )
    blind_period_id = f"blind-{start_date.isoformat()}-{end_date.isoformat()}"
    trials: list[ResearchTrial] = []
    for hypothesis in sorted(
        executable, key=lambda item: (item.hypothesis_id, item.hypothesis_version)
    ):
        if not hypothesis.available_data_confirmed or hypothesis.blocked_missing_data:
            raise ValueError(f"hypothesis data blocked: {hypothesis.hypothesis_id}")
        if hypothesis.discovery_source == DiscoverySource.AI_DRAFT:
            raise ValueError("AI draft cannot be scheduled as factor research")
        recipes = {item.recipe_id: item for item in hypothesis.feature_recipes}
        selected_recipes = [recipes[item] for item in hypothesis.allowed_variants]
        if len(selected_recipes) * len(hypothesis.expected_horizons) > MAX_TRIALS_PER_HYPOTHESIS:
            raise ValueError("RESEARCH_BUDGET_EXCEEDED:trials_per_hypothesis")
        for recipe in selected_recipes:
            feature_recipe_hash = _json_digest(recipe.model_dump(mode="json"))
            lookback = max(
                [int(item.value) for item in recipe.parameters if item.value.isdigit()] or [1]
            )
            portfolio_rule_id = (
                "long_only_market_timing_equal_weight_v1"
                if StrategyFit.ABSOLUTE_TIMING in hypothesis.strategy_fit
                else "long_only_top3_equal_weight_v1"
            )
            for horizon in hypothesis.expected_horizons:
                formula_payload = {
                    "hypothesis_definition_digest": hypothesis.definition_digest,
                    "recipe": recipe.model_dump(mode="json"),
                    "direction": hypothesis.expected_direction,
                }
                factor_formula_hash = _json_digest(formula_payload)
                test_family_id = (
                    f"{hypothesis.factor_family.value.lower()}."
                    f"{hypothesis.hypothesis_id}.v{hypothesis.hypothesis_version}"
                )
                seed_payload = {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "recipe_id": recipe.recipe_id,
                    "horizon": horizon,
                    "data_snapshot_id": data_snapshot_id,
                }
                random_seed = int(_json_digest(seed_payload)[:8], 16) % (2**31)
                identity = {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "hypothesis_version": hypothesis.hypothesis_version,
                    "test_family_id": test_family_id,
                    "factor_formula_hash": factor_formula_hash,
                    "feature_recipe_hash": feature_recipe_hash,
                    "direction": hypothesis.expected_direction,
                    "lookback": lookback,
                    "horizon": horizon,
                    "universe_id": hypothesis.universe_definition.universe_id,
                    "neutralization_id": hypothesis.neutralization_plan.neutralization_id,
                    "cost_model_id": hypothesis.cost_model.cost_model_id,
                    "portfolio_rule_id": portfolio_rule_id,
                    "split_definition": split_definition,
                    "blind_period_id": blind_period_id,
                    "random_seed": random_seed,
                    "code_commit": code_commit,
                    "data_snapshot_id": data_snapshot_id,
                }
                trial_id = build_trial_id(**identity)
                trials.append(
                    ResearchTrial(
                        trial_id=trial_id,
                        **identity,
                        nas_task_id=nas_task_id,
                        trial_kind=TrialKind.CONFIRMATORY,
                        parameter_locked_at=locked_at,
                        submitted_at=locked_at,
                        status=TrialStatus.SUBMITTED,
                        decision=FactorResearchDecision.INCONCLUSIVE,
                    )
                )
    if len(trials) > MAX_RESEARCH_TRIALS:
        raise ValueError("RESEARCH_BUDGET_EXCEEDED:trials")
    return trials


def publish_hypothesis_registry(
    lake_root: str | Path,
    hypotheses: Iterable[ResearchHypothesis],
    *,
    dry_run: bool = False,
) -> int:
    materialized = list(hypotheses)
    validate_hypothesis_budget(materialized)
    rows = [_hypothesis_row(item) for item in materialized]
    frame = _schema_frame(rows, HYPOTHESIS_REGISTRY_SCHEMA)
    root = Path(lake_root)
    existing = read_parquet_dataset(root / RESEARCH_HYPOTHESIS_REGISTRY_DATASET)
    _require_hypothesis_versions_immutable(existing, frame)
    if dry_run:
        return existing.height
    return upsert_parquet_dataset(
        frame,
        root / RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
        key_columns=["hypothesis_id", "hypothesis_version"],
    )


def publish_trial_ledger(
    lake_root: str | Path,
    trials: Iterable[ResearchTrial],
    *,
    dry_run: bool = False,
) -> int:
    materialized = list(trials)
    ids = [item.trial_id for item in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trial_id in publish batch")
    rows = [_trial_row(item) for item in materialized]
    frame = _schema_frame(rows, TRIAL_LEDGER_SCHEMA)
    root = Path(lake_root)
    existing = read_parquet_dataset(root / RESEARCH_TRIAL_LEDGER_DATASET)
    _require_trial_identities_immutable(existing, frame)
    if dry_run:
        return existing.height
    return upsert_parquet_dataset(
        frame,
        root / RESEARCH_TRIAL_LEDGER_DATASET,
        key_columns=["trial_id"],
    )


def factor_retirement_registry_frame(*, recorded_at: datetime | None = None) -> pl.DataFrame:
    observed_at = recorded_at or datetime.now(UTC)
    rows = [
        _retirement("rev_xs_20d", "reversal", "RETIRED", "audit_signal_invalid", "FAIL"),
        _retirement(
            "production_20d_momentum", "momentum", "RETIRED", "audit_signal_invalid", "FAIL"
        ),
        _retirement("vol_adjusted_momentum", "momentum", "RETIRED", "audit_signal_invalid", "FAIL"),
        _retirement("volume_dry_reversal", "reversal", "RETIRED", "audit_signal_invalid", "FAIL"),
        _retirement("pullback_uptrend", "timing", "RETIRED", "audit_signal_invalid", "FAIL"),
        _retirement(
            "failed_alpha158_overlays",
            "legacy_overlay",
            "RETIRED",
            "audit_multiple_testing_failure",
            "FAIL",
        ),
        _retirement(
            "funding_fade",
            "derivatives_crowding",
            "DATA_PENDING",
            "real_funding_history_missing",
            "UNKNOWN",
        ),
        {
            **_retirement(
                "low_vol_20d",
                "defensive_quality",
                "REFERENCE_SIGNAL",
                "requires_structural_dynamic_decomposition",
                "PASS",
            ),
            "portfolio_validity": "FAIL",
        },
    ]
    return _schema_frame(
        [{**row, "recorded_at": observed_at} for row in rows],
        FACTOR_RETIREMENT_SCHEMA,
    )


def _data_blocked_hypothesis(
    *,
    hypothesis_id: str,
    research_thread_id: str,
    title: str,
    family: ResearchFamily,
    operation: str,
    required_dataset: str,
    required_columns: tuple[str, ...],
    missing: tuple[str, ...],
    recorded_at: datetime,
) -> ResearchHypothesis:
    return ResearchHypothesis(
        hypothesis_id=hypothesis_id,
        hypothesis_version=1,
        research_thread_id=research_thread_id,
        title=title,
        factor_family=family,
        economic_mechanism=(
            "Crowding or execution pressure may predict subsequent relative returns."
        ),
        who_pays_the_edge="overcrowded leveraged participants",
        why_edge_may_persist="positioning constraints can unwind slowly",
        why_edge_may_decay="exchange structure and participant mix change",
        strategy_fit=(StrategyFit.CROSS_SECTIONAL_RANK,),
        data_requirements=(
            DataRequirement(
                dataset_name=required_dataset,
                required_columns=required_columns,
                min_history_days=730,
            ),
        ),
        available_data_confirmed=False,
        blocked_missing_data=missing,
        feature_recipes=(
            FeatureRecipe(
                recipe_id=f"{hypothesis_id}.v1",
                operation=operation,
                source_fields=required_columns[2:] or required_columns,
            ),
        ),
        expected_direction=-1,
        expected_horizons=(24,),
        allowed_variants=(f"{hypothesis_id}.v1",),
        variant_budget=1,
        universe_definition=UniverseDefinition(
            universe_id="spot-dynamic-quality-v1",
            dynamic_source_dataset="gold/expanded_universe_quality",
            inclusion_rule="quality-qualified symbols with audited source data",
            exclusion_rule="exclude proxy-only or incomplete history",
        ),
        neutralization_plan=NeutralizationPlan(
            neutralization_id="xs-core-controls-v1",
            controls=("symbol_fixed_effect", "market_beta", "liquidity", "momentum"),
        ),
        benchmark_plan=BenchmarkPlan(
            benchmark_id="spot-btc-and-dynamic-ew-v1",
            benchmarks=("BTC", "DYNAMIC_UNIVERSE_EQUAL_WEIGHT"),
        ),
        cost_model=CostModelPlan(cost_model_id="research-point-in-time-p75-v1"),
        falsification_conditions=("no corrected out-of-sample signal",),
        stopping_conditions=("required real dataset remains unavailable",),
        success_conditions=("audited real source history reaches two years",),
        discovery_source=DiscoverySource.AUDIT_IMPORT,
        status=HypothesisStatus.DATA_BLOCKED,
        created_at=recorded_at,
        updated_at=recorded_at,
    )


def _hypothesis_row(item: ResearchHypothesis) -> dict[str, Any]:
    payload = item.model_dump(mode="json")
    return {
        "hypothesis_id": item.hypothesis_id,
        "hypothesis_version": item.hypothesis_version,
        "research_thread_id": item.research_thread_id,
        "title": item.title,
        "factor_family": item.factor_family.value,
        "economic_mechanism": item.economic_mechanism,
        "who_pays_the_edge": item.who_pays_the_edge,
        "why_edge_may_persist": item.why_edge_may_persist,
        "why_edge_may_decay": item.why_edge_may_decay,
        "strategy_fit_json": canonical_json(payload["strategy_fit"]),
        "data_requirements_json": canonical_json(payload["data_requirements"]),
        "available_data_confirmed": item.available_data_confirmed,
        "blocked_missing_data_json": canonical_json(payload["blocked_missing_data"]),
        "feature_recipes_json": canonical_json(payload["feature_recipes"]),
        "expected_direction": item.expected_direction,
        "expected_horizons_json": canonical_json(payload["expected_horizons"]),
        "allowed_variants_json": canonical_json(payload["allowed_variants"]),
        "variant_budget": item.variant_budget,
        "universe_definition_json": canonical_json(payload["universe_definition"]),
        "neutralization_plan_json": canonical_json(payload["neutralization_plan"]),
        "benchmark_plan_json": canonical_json(payload["benchmark_plan"]),
        "cost_model_json": canonical_json(payload["cost_model"]),
        "falsification_conditions_json": canonical_json(payload["falsification_conditions"]),
        "stopping_conditions_json": canonical_json(payload["stopping_conditions"]),
        "success_conditions_json": canonical_json(payload["success_conditions"]),
        "known_overlap_json": canonical_json(payload["known_overlap"]),
        "related_hypotheses_json": canonical_json(payload["related_hypotheses"]),
        "related_existing_factors_json": canonical_json(payload["related_existing_factors"]),
        "discovery_source": item.discovery_source.value,
        "status": item.status.value,
        "approved_by": item.approved_by,
        "approved_at": item.approved_at,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "definition_digest": item.definition_digest,
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0.0,
    }


def _trial_row(item: ResearchTrial) -> dict[str, Any]:
    payload = item.model_dump(mode="python")
    payload.update(
        {
            "trial_kind": item.trial_kind.value,
            "status": item.status.value,
            "decision": item.decision.value,
            "identity_digest": item.identity_digest,
            "max_live_notional_usdt": 0.0,
        }
    )
    return payload


def _retirement(
    factor_id: str,
    family: str,
    status: str,
    reason: str,
    signal_validity: str,
) -> dict[str, Any]:
    return {
        "factor_id": factor_id,
        "factor_family": family,
        "status": status,
        "enabled": False,
        "retirement_reason": reason,
        "signal_validity": signal_validity,
        "portfolio_validity": "UNKNOWN",
        "source_audit": "factor_discovery_v2_migration",
        "research_only": True,
        "live_order_effect": "none",
    }


def _require_hypothesis_versions_immutable(existing: pl.DataFrame, incoming: pl.DataFrame) -> None:
    if existing.is_empty() or incoming.is_empty():
        return
    required = ["hypothesis_id", "hypothesis_version", "definition_digest"]
    if not set(required).issubset(existing.columns):
        raise ValueError("existing hypothesis registry lacks immutable identity fields")
    conflicts = (
        incoming.select(required)
        .join(
            existing.select(required),
            on=["hypothesis_id", "hypothesis_version"],
            how="inner",
            suffix="_existing",
        )
        .filter(pl.col("definition_digest") != pl.col("definition_digest_existing"))
    )
    if conflicts.height:
        raise ValueError("hypothesis definition changed without a new version")


def _require_trial_identities_immutable(existing: pl.DataFrame, incoming: pl.DataFrame) -> None:
    if existing.is_empty() or incoming.is_empty():
        return
    required = ["trial_id", "identity_digest"]
    if not set(required).issubset(existing.columns):
        raise ValueError("existing trial ledger lacks immutable identity fields")
    conflicts = (
        incoming.select(required)
        .join(
            existing.select(required),
            on=["trial_id"],
            how="inner",
            suffix="_existing",
        )
        .filter(pl.col("identity_digest") != pl.col("identity_digest_existing"))
    )
    if conflicts.height:
        raise ValueError("trial identity mutation is forbidden")


def _schema_frame(rows: list[dict[str, Any]], schema: dict[str, Any]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    frame = pl.DataFrame(rows)
    for column, dtype in schema.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return frame.select(list(schema)).cast(schema, strict=False)


def _json_value(row: dict[str, Any], column: str) -> Any:
    value = row.get(column)
    if not isinstance(value, str):
        raise ValueError(f"hypothesis registry {column} must contain canonical JSON")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"hypothesis registry {column} is invalid JSON") from exc


def _json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _frame_digest(frame: pl.DataFrame, *, sort_by: list[str]) -> str:
    ordered = frame.sort(sort_by)
    rows = ordered.to_dicts()
    return _json_digest(rows)
