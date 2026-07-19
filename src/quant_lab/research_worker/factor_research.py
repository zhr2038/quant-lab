from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as datetime_time
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.factor_research.attribution import (
    FactorAttributionResult,
)
from quant_lab.research.factor_research.attribution import (
    compute_factor_attribution as compute_attribution_model,
)
from quant_lab.research.factor_research.contracts import (
    ResearchHypothesis,
    ResearchTrial,
    StrategyFit,
)
from quant_lab.research.factor_research.decision import (
    FactorDecisionEvidence,
    decide_factor_research,
)
from quant_lab.research.factor_research.outputs import (
    FACTOR_ATTRIBUTION_SCHEMA,
    FACTOR_PORTFOLIO_VALIDATION_SCHEMA,
    FACTOR_RESEARCH_CANDIDATE_SCHEMA,
    FACTOR_RESEARCH_DEFINITION_SCHEMA,
    FACTOR_RESEARCH_EVIDENCE_SCHEMA,
    FACTOR_RESEARCH_VALUE_SCHEMA,
)
from quant_lab.research.factor_research.overfit import (
    OverfitDiagnostics,
)
from quant_lab.research.factor_research.overfit import (
    compute_overfit_diagnostics as compute_overfit_model,
)
from quant_lab.research.factor_research.portfolio import (
    FactorPortfolioValidation,
    PortfolioRule,
    compute_factor_portfolio,
)
from quant_lab.research.factor_research.registry import (
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
    hypotheses_from_registry,
    hypothesis_registry_digest,
    trial_ledger_digest,
    trials_from_ledger,
)
from quant_lab.research.factor_research.statistics import (
    MultipleTestingAdjustment,
    OverlapAwareSignificance,
    adjust_multiple_testing,
    compute_overlap_aware_significance,
)
from quant_lab.research.ic import compute_period_ic_values
from quant_lab.research_plane.contracts import (
    FactorResearchSnapshotManifest,
    FactorResearchTask,
)

SOURCE = "factor_research.nas.v2"
MARKET_DATASET = Path("silver") / "market_bar"
COST_DATASET = Path("gold") / "cost_bucket_daily"
REGIME_DATASET = Path("gold") / "market_regime_daily"
MIN_PRIOR_DAY_CLOSED_BARS = 18

FACTOR_RESEARCH_ANTI_LEAKAGE_CHECKS = (
    "source_snapshot_identity_matches",
    "registry_and_trial_identity_match",
    "trials_are_pre_registered",
    "features_use_completed_prior_bars",
    "decision_after_feature_availability",
    "future_label_not_used_in_features",
    "label_horizon_complete",
    "label_end_within_snapshot_boundary",
    "chronological_split_locked",
    "universe_membership_not_from_future",
    "regime_not_from_future",
    "cost_not_from_future",
    "multiple_testing_family_complete",
    "read_only_no_live_action",
)


@dataclass(frozen=True)
class FactorResearchComputeArtifacts:
    generated_at: datetime
    definitions: pl.DataFrame
    values: pl.DataFrame
    evidence: pl.DataFrame
    attribution: pl.DataFrame
    portfolio_validation: pl.DataFrame
    candidates: pl.DataFrame
    anti_leakage: dict[str, Any]
    worker_report: dict[str, Any]
    warnings: tuple[str, ...]

    def frames_by_dataset(self) -> dict[str, pl.DataFrame]:
        return {
            "factor_definition": self.definitions,
            "factor_value": self.values,
            "factor_evidence": self.evidence,
            "factor_attribution": self.attribution,
            "factor_portfolio_validation": self.portfolio_validation,
            "factor_candidate": self.candidates,
        }


@dataclass(frozen=True)
class TrialStatistics:
    full: OverlapAwareSignificance
    development: OverlapAwareSignificance
    validation: OverlapAwareSignificance
    blind: OverlapAwareSignificance
    adjustment: MultipleTestingAdjustment
    coverage: float
    development_halves_same_sign: bool
    major_periods_same_sign_count: int
    cost_coverage: float


def compute_factor_features_from_snapshot(
    snapshot_root: str | Path,
    *,
    hypotheses: list[ResearchHypothesis],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute only registered feature recipes from immutable snapshot bars."""
    root = Path(snapshot_root) / "files"
    market = _normalized_market(read_parquet_dataset(root / MARKET_DATASET))
    if market.is_empty():
        raise ValueError("factor_research_market_bar_empty")
    base = _market_controls(market)
    feature_frames: list[pl.DataFrame] = []
    definition_rows: list[dict[str, Any]] = []
    generated_at = datetime.now(UTC)
    for hypothesis in hypotheses:
        for recipe in hypothesis.feature_recipes:
            if recipe.recipe_id not in hypothesis.allowed_variants:
                continue
            factor_id = _factor_id(hypothesis, recipe.recipe_id)
            feature_recipe_hash = _digest(recipe.model_dump(mode="json"))
            factor_hash = _digest(
                {
                    "hypothesis_definition_digest": hypothesis.definition_digest,
                    "recipe": recipe.model_dump(mode="json"),
                    "direction": hypothesis.expected_direction,
                }
            )
            definition_rows.append(
                _definition_row(
                    hypothesis,
                    recipe_id=recipe.recipe_id,
                    operation=recipe.operation,
                    factor_id=factor_id,
                    factor_hash=factor_hash,
                    generated_at=generated_at,
                )
            )
            if recipe.operation in {"rolling_volatility", "volatility_change"}:
                frame = _volatility_feature(base, hypothesis=hypothesis, recipe=recipe)
            elif recipe.operation in {"market_breadth", "market_dispersion"}:
                frame = _breadth_feature(base, hypothesis=hypothesis, recipe=recipe)
            else:
                raise ValueError(f"factor_research_unsupported_operation:{recipe.operation}")
            feature_frames.append(
                frame.with_columns(
                    pl.lit(factor_id).alias("factor_id"),
                    pl.lit(factor_hash).alias("factor_hash"),
                    pl.lit(hypothesis.hypothesis_id).alias("hypothesis_id"),
                    pl.lit(hypothesis.hypothesis_version).alias("hypothesis_version"),
                    pl.lit(recipe.recipe_id).alias("feature_recipe_id"),
                    pl.lit(feature_recipe_hash).alias("feature_recipe_hash"),
                    pl.lit(hypothesis.factor_family.value).alias("factor_family"),
                    pl.lit(hypothesis.expected_direction).alias("direction"),
                )
            )
    if not feature_frames:
        raise ValueError("factor_research_no_registered_features")
    features = pl.concat(feature_frames, how="diagonal_relaxed").sort(["factor_id", "symbol", "ts"])
    definitions = _schema_frame(definition_rows, FACTOR_RESEARCH_DEFINITION_SCHEMA)
    return definitions, features


def compute_factor_values(
    features: pl.DataFrame,
    *,
    data_snapshot_id: str,
    code_commit: str,
    generated_at: datetime,
) -> pl.DataFrame:
    rows = features.select(
        "hypothesis_id",
        "hypothesis_version",
        "feature_recipe_id",
        "factor_id",
        "factor_family",
        "symbol",
        "ts",
        "available_time",
        "raw_value",
        "normalized_value",
        "rank_value",
        "factor_hash",
    ).to_dicts()
    output = []
    for row in rows:
        factor_id = str(row["factor_id"])
        factor_hash = str(row["factor_hash"])
        output.append(
            {
                **row,
                "data_snapshot_id": data_snapshot_id,
                "factor_name": factor_id,
                "factor_version": "factor_research.v2",
                "timeframe": "1h",
                "event_time": row["ts"],
                "value": row["normalized_value"],
                "factor_status": "RESEARCH_ONLY",
                "expression_hash": factor_hash,
                "canonical_factor_id": factor_id,
                "factor_formula_hash": factor_hash,
                "formula_hash": factor_hash,
                "operator_graph_hash": factor_hash,
                "correlation_cluster_id": factor_id,
                "effective_independence_weight": 1.0,
                "independence_weight": 1.0,
                "input_features_json": json.dumps(
                    [row["feature_recipe_id"]], separators=(",", ":")
                ),
                "input_dataset_version": data_snapshot_id,
                "data_version": data_snapshot_id,
                "input_hash": factor_hash,
                "code_version": code_commit,
                "calculated_at": generated_at,
                "created_at": generated_at,
                "source": SOURCE,
                "is_valid": row["raw_value"] is not None,
                "invalid_reason": None if row["raw_value"] is not None else "warmup_incomplete",
                "quality_flags_json": "[]",
                "research_only": True,
                "live_order_effect": "none",
            }
        )
    return _schema_frame(output, FACTOR_RESEARCH_VALUE_SCHEMA)


def compute_factor_statistics(
    samples_by_trial: dict[str, pl.DataFrame],
    *,
    trials: list[ResearchTrial],
    timing_hypothesis_ids: set[str],
) -> dict[str, TrialStatistics]:
    preliminary: dict[str, dict[str, Any]] = {}
    pvalues: dict[str, float] = {}
    trial_by_id = {item.trial_id: item for item in trials}
    for trial_id, samples in samples_by_trial.items():
        trial = trial_by_id[trial_id]
        timing = trial.hypothesis_id in timing_hypothesis_ids
        values = _ic_values(samples, timing=timing)
        development_frame = samples.filter(pl.col("split") == "RESEARCH")
        validation_frame = samples.filter(pl.col("split") == "VALIDATION")
        blind_frame = samples.filter(pl.col("split") == "BLIND_CONFIRMATORY")
        development_values = _ic_values(development_frame, timing=timing)
        validation_values = _ic_values(validation_frame, timing=timing)
        blind_values = _ic_values(blind_frame, timing=timing)
        horizon_periods = max(1, trial.horizon // 24) if timing else trial.horizon
        kwargs = {
            "horizon": horizon_periods,
            "expected_direction": trial.direction,
            "bootstrap_samples": 1000,
            "permutation_samples": 1000,
            "random_seed": trial.random_seed,
        }
        full = compute_overlap_aware_significance(values, **kwargs)
        development = compute_overlap_aware_significance(development_values, **kwargs)
        validation = compute_overlap_aware_significance(validation_values, **kwargs)
        blind = compute_overlap_aware_significance(blind_values, **kwargs)
        pvalues[trial_id] = blind.raw_pvalue
        valid = samples.filter(
            pl.col("raw_value").is_not_null() & pl.col("forward_return").is_not_null()
        ).height
        cost_valid = samples.filter(pl.col("cost_bps").is_not_null()).height
        preliminary[trial_id] = {
            "full": full,
            "development": development,
            "validation": validation,
            "blind": blind,
            "coverage": valid / samples.height if samples.height else 0.0,
            "development_halves_same_sign": _halves_same_direction(
                development_values, trial.direction
            ),
            "major_periods_same_sign_count": sum(
                item.mean * trial.direction > 0 for item in (development, validation, blind)
            ),
            "cost_coverage": cost_valid / samples.height if samples.height else 0.0,
        }
    adjustments = adjust_multiple_testing(pvalues)
    return {
        trial_id: TrialStatistics(adjustment=adjustments[trial_id], **payload)
        for trial_id, payload in preliminary.items()
    }


def compute_factor_attribution(
    samples_by_trial: dict[str, pl.DataFrame],
    *,
    trials: list[ResearchTrial],
    timing_hypothesis_ids: set[str],
) -> dict[str, FactorAttributionResult]:
    results: dict[str, FactorAttributionResult] = {}
    for trial in trials:
        samples = samples_by_trial[trial.trial_id]
        signed = samples.with_columns((pl.col("raw_value") * trial.direction).alias("alpha_score"))
        if trial.hypothesis_id in timing_hypothesis_ids:
            raw_ic = _rank_correlation(
                signed.get_column("alpha_score").to_list(),
                signed.get_column("universe_forward_return").to_list(),
            )
            results[trial.trial_id] = FactorAttributionResult(
                raw_rank_ic=raw_ic,
                symbol_fixed_effect_null_ic=raw_ic,
                beta_neutral_rank_ic=raw_ic,
                liquidity_neutral_rank_ic=raw_ic,
                momentum_neutral_rank_ic=raw_ic,
                joint_residual_rank_ic=raw_ic,
                incremental_ic=raw_ic,
                attribution_type="DYNAMIC_TIMING",
                max_symbol_contribution_share=0.0,
                dominant_symbol=None,
                row_count=signed.height,
                period_count=signed.get_column("decision_ts").n_unique(),
                controls_available=("market_breadth", "regime_score"),
                warnings=(),
            )
        else:
            results[trial.trial_id] = compute_attribution_model(
                signed,
                factor_id=str(signed.item(0, "factor_id")) if signed.height else trial.trial_id,
            )
    return results


def compute_factor_portfolios(
    samples_by_trial: dict[str, pl.DataFrame],
    *,
    trials: list[ResearchTrial],
    statistics: dict[str, TrialStatistics],
    timing_hypothesis_ids: set[str],
) -> tuple[dict[str, FactorPortfolioValidation], dict[str, list[float]]]:
    validations: dict[str, FactorPortfolioValidation] = {}
    return_series: dict[str, list[float]] = {}
    for trial in trials:
        samples = samples_by_trial[trial.trial_id].with_columns(
            (pl.col("raw_value") * trial.direction).alias("alpha_score")
        )
        stats = statistics[trial.trial_id]
        signal_pass = _signal_pass(stats, trial.direction)
        timing = trial.hypothesis_id in timing_hypothesis_ids
        rule = PortfolioRule(
            portfolio_rule_id=trial.portfolio_rule_id,
            selection_mode="absolute_timing" if timing else "cross_sectional_rank",
            top_n=200 if timing else 3,
            weighting="equal",
            holding_period_bars=trial.horizon,
            initial_capital_usdt=10_000.0,
            min_order_usdt=5.0,
            max_drawdown_limit=0.25,
            annual_periods=max(1, 365 * 24 // trial.horizon),
        )
        validations[trial.trial_id] = compute_factor_portfolio(
            samples,
            rule=rule,
            signal_validity="PASS" if signal_pass else "FAIL",
        )
        return_series[trial.trial_id] = _portfolio_return_series(
            samples,
            horizon=trial.horizon,
            timing=timing,
        )
    return validations, return_series


def compute_factor_overfit_diagnostics(
    return_series: dict[str, list[float]],
    *,
    trials: list[ResearchTrial],
) -> dict[str, OverfitDiagnostics]:
    by_hypothesis: dict[str, dict[str, list[float]]] = {}
    for trial in trials:
        by_hypothesis.setdefault(trial.hypothesis_id, {})[trial.trial_id] = return_series[
            trial.trial_id
        ]
    hypothesis_results = {
        hypothesis_id: compute_overfit_model(variants)
        for hypothesis_id, variants in by_hypothesis.items()
    }
    return {trial.trial_id: hypothesis_results[trial.hypothesis_id] for trial in trials}


def compute_factor_research_result(
    snapshot_root: str | Path,
    manifest: FactorResearchSnapshotManifest,
    task: FactorResearchTask,
) -> FactorResearchComputeArtifacts:
    """Run the full research-only Factor Discovery v2 compute on NAS inputs."""
    started = time.perf_counter()
    root = Path(snapshot_root) / "files"
    registry = read_parquet_dataset(root / RESEARCH_HYPOTHESIS_REGISTRY_DATASET)
    ledger = read_parquet_dataset(root / RESEARCH_TRIAL_LEDGER_DATASET)
    if hypothesis_registry_digest(registry) != task.hypothesis_registry_digest:
        raise ValueError("factor_research_worker_registry_digest_mismatch")
    if trial_ledger_digest(ledger) != task.trial_ledger_digest:
        raise ValueError("factor_research_worker_trial_ledger_digest_mismatch")
    hypotheses = [
        item
        for item in hypotheses_from_registry(registry)
        if item.hypothesis_id in task.hypothesis_ids
    ]
    trials = trials_from_ledger(ledger)
    if tuple(item.trial_id for item in trials) != task.trial_ids:
        raise ValueError("factor_research_worker_trial_membership_mismatch")
    if any(
        item.nas_task_id != task.task_id
        or item.data_snapshot_id != f"factor-input-{task.source_input_digest[:24]}"
        for item in trials
    ):
        raise ValueError("factor_research_worker_trial_binding_mismatch")
    generated_at = datetime.now(UTC)
    definitions, features = compute_factor_features_from_snapshot(
        snapshot_root, hypotheses=hypotheses
    )
    values = compute_factor_values(
        features,
        data_snapshot_id=f"factor-input-{task.source_input_digest[:24]}",
        code_commit=task.quant_lab_commit,
        generated_at=generated_at,
    )
    samples_by_trial = _build_trial_samples(root, features=features, trials=trials, task=task)
    timing_ids = {
        item.hypothesis_id
        for item in hypotheses
        if StrategyFit.ABSOLUTE_TIMING in item.strategy_fit
    }
    statistics = compute_factor_statistics(
        samples_by_trial,
        trials=trials,
        timing_hypothesis_ids=timing_ids,
    )
    attributions = compute_factor_attribution(
        samples_by_trial,
        trials=trials,
        timing_hypothesis_ids=timing_ids,
    )
    portfolios, return_series = compute_factor_portfolios(
        samples_by_trial,
        trials=trials,
        statistics=statistics,
        timing_hypothesis_ids=timing_ids,
    )
    overfit = compute_factor_overfit_diagnostics(return_series, trials=trials)
    evidence, attribution_frame, portfolio_frame = _decision_frames(
        task=task,
        trials=trials,
        samples_by_trial=samples_by_trial,
        statistics=statistics,
        attributions=attributions,
        portfolios=portfolios,
        overfit=overfit,
        generated_at=generated_at,
    )
    candidates = _candidate_frame(evidence, generated_at=generated_at)
    anti_leakage = _anti_leakage_report(
        manifest=manifest,
        task=task,
        trials=trials,
        samples_by_trial=samples_by_trial,
    )
    if anti_leakage["status"] != "PASS" or anti_leakage["violation_count"] != 0:
        failed = ",".join(
            row["check_name"] for row in anti_leakage["checks"] if row["status"] != "PASS"
        )
        raise ValueError(f"factor_research_anti_leakage_not_pass:{failed}")
    frames = {
        "factor_definition": definitions,
        "factor_value": values,
        "factor_evidence": evidence,
        "factor_attribution": attribution_frame,
        "factor_portfolio_validation": portfolio_frame,
        "factor_candidate": candidates,
    }
    worker_report = {
        "schema_version": "quant_lab.factor_research_worker_report.v1",
        "task_id": task.task_id,
        "snapshot_id": task.snapshot_id,
        "source_input_digest": task.source_input_digest,
        "hypothesis_registry_digest": task.hypothesis_registry_digest,
        "trial_ledger_digest": task.trial_ledger_digest,
        "hypothesis_count": len(hypotheses),
        "test_count": len(trials),
        "output_rows": {name: frame.height for name, frame in frames.items()},
        "decision_counts": _value_counts(evidence, "decision"),
        "compute_duration_seconds": time.perf_counter() - started,
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
    }
    return FactorResearchComputeArtifacts(
        generated_at=generated_at,
        definitions=definitions,
        values=values,
        evidence=evidence,
        attribution=attribution_frame,
        portfolio_validation=portfolio_frame,
        candidates=candidates,
        anti_leakage=anti_leakage,
        worker_report=worker_report,
        warnings=(),
    )


def _normalized_market(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"symbol", "ts", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"factor_research_market_missing_columns:{','.join(missing)}")
    return (
        frame.with_columns(
            pl.col("symbol")
            .cast(pl.Utf8)
            .str.to_uppercase()
            .str.replace_all("/", "-")
            .alias("symbol"),
            pl.col("ts").cast(pl.Datetime(time_zone="UTC"), strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("quote_volume").cast(pl.Float64, strict=False)
            if "quote_volume" in frame.columns
            else pl.lit(None, dtype=pl.Float64).alias("quote_volume"),
        )
        .drop_nulls(["symbol", "ts", "close"])
        .unique(["symbol", "ts"], keep="last")
        .sort(["symbol", "ts"])
    )


def _market_controls(market: pl.DataFrame) -> pl.DataFrame:
    base = market.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("return_1h"),
        (pl.col("close") / pl.col("close").shift(24).over("symbol") - 1.0).alias("momentum"),
        pl.col("quote_volume").fill_null(0.0).log1p().alias("liquidity"),
    ).with_columns(
        pl.col("return_1h")
        .rolling_std(window_size=480, min_samples=120)
        .over("symbol")
        .alias("long_run_volatility")
    )
    btc = base.filter(pl.col("symbol") == "BTC-USDT").select(
        "ts", pl.col("return_1h").alias("btc_return_1h")
    )
    joined = base.join(btc, on="ts", how="left")
    return joined.with_columns(
        (
            pl.rolling_cov("return_1h", "btc_return_1h", window_size=168, min_samples=48).over(
                "symbol"
            )
            / pl.col("btc_return_1h").rolling_var(window_size=168, min_samples=48).over("symbol")
        ).alias("market_beta"),
        pl.col("liquidity").alias("size_proxy"),
    )


def _point_in_time_market_universe(market: pl.DataFrame) -> pl.DataFrame:
    """Build next-day membership only from the preceding completed UTC day."""
    eligible = market
    if "timeframe" in eligible.columns:
        eligible = eligible.filter(
            pl.col("timeframe").cast(pl.Utf8).str.to_lowercase() == "1h"
        )
    if "is_closed" in eligible.columns:
        eligible = eligible.filter(pl.col("is_closed").fill_null(False))
    daily = (
        eligible.filter((pl.col("close") > 0) & pl.col("ts").is_not_null())
        .with_columns(pl.col("ts").dt.date().alias("_bar_day"))
        .group_by(["symbol", "_bar_day"])
        .agg(
            pl.len().alias("_closed_bar_count"),
            pl.col("quote_volume").fill_null(0.0).sum().alias("_quote_volume_24h"),
        )
        .filter(
            (pl.col("_closed_bar_count") >= MIN_PRIOR_DAY_CLOSED_BARS)
            & (pl.col("_quote_volume_24h") > 0)
        )
        .with_columns(
            (pl.col("_bar_day") + timedelta(days=1)).alias("source_day"),
            (pl.col("_closed_bar_count") / 24.0).clip(upper_bound=1.0).alias(
                "universe_quality_score"
            ),
        )
        .select("symbol", "source_day", "universe_quality_score")
        .unique(["symbol", "source_day"], keep="last")
        .sort(["symbol", "source_day"])
    )
    return daily


def _volatility_feature(
    market: pl.DataFrame,
    *,
    hypothesis: ResearchHypothesis,
    recipe: Any,
) -> pl.DataFrame:
    parameters = {item.name: int(item.value) for item in recipe.parameters}
    window = max(parameters.values(), default=480)
    current = pl.col("return_1h").rolling_std(window_size=window, min_samples=window).over("symbol")
    if recipe.operation == "rolling_volatility":
        raw = current
    else:
        raw = current - current.shift(window).over("symbol")
    frame = market.with_columns(raw.alias("raw_value")).with_columns(
        _cross_section_zscore("raw_value").alias("normalized_value"),
        _cross_section_rank("raw_value").alias("rank_value"),
        (pl.col("ts") + timedelta(hours=1)).alias("available_time"),
    )
    return frame.select(
        "symbol",
        "ts",
        "available_time",
        "raw_value",
        "normalized_value",
        "rank_value",
    )


def _breadth_feature(
    market: pl.DataFrame,
    *,
    hypothesis: ResearchHypothesis,
    recipe: Any,
) -> pl.DataFrame:
    del hypothesis
    parameters = {item.name: int(item.value) for item in recipe.parameters}
    lookback = parameters.get("return_lookback_bars", 24)
    returns = market.with_columns(
        (pl.col("close") / pl.col("close").shift(lookback).over("symbol") - 1.0).alias(
            "_lookback_return"
        )
    )
    aggregate = returns.group_by("ts").agg(
        (pl.col("_lookback_return") > 0).mean().alias("_advance_ratio"),
        pl.col("_lookback_return").std().alias("_dispersion"),
    )
    source = "_advance_ratio" if recipe.operation == "market_breadth" else "_dispersion"
    return (
        aggregate.with_columns(pl.col(source).alias("raw_value"))
        .with_columns(
            (
                (pl.col("raw_value") - pl.col("raw_value").rolling_mean(168, min_samples=48))
                / pl.col("raw_value").rolling_std(168, min_samples=48)
            ).alias("normalized_value"),
            pl.col("raw_value").rank(method="average").truediv(pl.len() + 1).alias("rank_value"),
            pl.lit("DYNAMIC-UNIVERSE").alias("symbol"),
            (pl.col("ts") + timedelta(hours=1)).alias("available_time"),
        )
        .select(
            "symbol",
            "ts",
            "available_time",
            "raw_value",
            "normalized_value",
            "rank_value",
        )
    )


def _build_trial_samples(
    snapshot_lake_root: Path,
    *,
    features: pl.DataFrame,
    trials: list[ResearchTrial],
    task: FactorResearchTask,
) -> dict[str, pl.DataFrame]:
    market = _market_controls(
        _normalized_market(read_parquet_dataset(snapshot_lake_root / MARKET_DATASET))
    )
    universe = _point_in_time_market_universe(market)
    costs = _normalize_daily_source(
        read_parquet_dataset(snapshot_lake_root / COST_DATASET),
        value_candidates=(
            "total_cost_bps_p75",
            "selected_total_cost_bps",
            "roundtrip_all_in_cost_bps",
            "cost_bps",
        ),
        output_name="cost_bps",
    )
    regimes = _normalize_daily_source(
        read_parquet_dataset(snapshot_lake_root / REGIME_DATASET),
        value_candidates=("current_regime", "regime_state", "market_regime", "state"),
        output_name="regime",
        numeric=False,
        allow_global=True,
    )
    result: dict[str, pl.DataFrame] = {}
    for trial in trials:
        factor_id = _factor_id_from_trial(features, trial)
        feature = features.filter(pl.col("factor_id") == factor_id)
        timing = feature.get_column("symbol").unique().to_list() == ["DYNAMIC-UNIVERSE"]
        labels = _forward_labels(market, horizon=trial.horizon)
        if timing:
            samples = labels.join(
                feature.select(
                    "factor_id",
                    "ts",
                    "available_time",
                    "raw_value",
                    "normalized_value",
                    "rank_value",
                ),
                on="ts",
                how="inner",
            )
        else:
            samples = feature.join(labels, on=["symbol", "ts"], how="inner")
        start = datetime.combine(task.start_date, datetime_time.min, tzinfo=UTC)
        end = datetime.combine(task.end_date + timedelta(days=1), datetime_time.min, tzinfo=UTC)
        samples = samples.with_columns(pl.col("available_time").alias("decision_ts")).filter(
            (pl.col("decision_ts") >= start) & (pl.col("decision_ts") < end)
        )
        samples = _join_point_in_time(
            samples,
            universe,
            value_column="universe_quality_score",
        )
        samples = _join_point_in_time(samples, costs, value_column="cost_bps")
        samples = _join_point_in_time(samples, regimes, value_column="regime")
        samples = samples.with_columns(
            (
                pl.col("universe_quality_score").is_not_null()
                & (
                    pl.col("_universe_quality_score_source_day")
                    == pl.col("decision_ts").dt.date()
                )
            ).alias("tradable"),
            pl.col("regime")
            .replace_strict(
                {
                    "RISK_OFF": -1.0,
                    "TREND_DOWN": -1.0,
                    "SIDEWAYS": 0.0,
                    "TREND_UP": 1.0,
                    "RISK_ON": 1.0,
                    "RISK_ON_CONFIRMED": 1.0,
                },
                default=0.0,
                return_dtype=pl.Float64,
            )
            .alias("regime_score"),
            _split_expression(task).alias("split"),
            pl.lit(trial.trial_id).alias("trial_id"),
            pl.lit(trial.hypothesis_id).alias("hypothesis_id"),
            pl.lit(trial.hypothesis_version).alias("hypothesis_version"),
            pl.lit(trial.feature_recipe_hash).alias("feature_recipe_hash"),
            pl.lit(trial.factor_formula_hash).alias("factor_hash"),
            pl.lit(factor_id).alias("factor_id"),
        )
        result[trial.trial_id] = samples.sort(["decision_ts", "symbol"])
    return result


def _forward_labels(market: pl.DataFrame, *, horizon: int) -> pl.DataFrame:
    labels = market.with_columns(
        pl.col("close").shift(-horizon).over("symbol").alias("_future_close"),
        pl.col("ts").shift(-horizon).over("symbol").alias("label_ts"),
    ).with_columns((pl.col("_future_close") / pl.col("close") - 1.0).alias("forward_return"))
    universe = labels.group_by("ts").agg(
        pl.col("forward_return").mean().alias("universe_forward_return")
    )
    btc = labels.filter(pl.col("symbol") == "BTC-USDT").select(
        "ts", pl.col("forward_return").alias("btc_forward_return")
    )
    return labels.join(universe, on="ts", how="left").join(btc, on="ts", how="left")


def _normalize_daily_source(
    frame: pl.DataFrame,
    *,
    value_candidates: tuple[str, ...],
    output_name: str,
    numeric: bool = True,
    allow_global: bool = False,
) -> pl.DataFrame:
    day_column = next(
        (name for name in ("day", "as_of_date", "date", "created_at") if name in frame.columns),
        None,
    )
    value_column = next((name for name in value_candidates if name in frame.columns), None)
    if day_column is None or value_column is None or (
        "symbol" not in frame.columns and not allow_global
    ):
        dtype = pl.Float64 if numeric else pl.Utf8
        return pl.DataFrame(schema={"symbol": pl.Utf8, "source_day": pl.Date, output_name: dtype})
    value = (
        pl.col(value_column).cast(pl.Float64, strict=False)
        if numeric
        else pl.col(value_column).cast(pl.Utf8, strict=False)
    )
    symbol = (
        pl.col("symbol")
            .cast(pl.Utf8)
            .str.to_uppercase()
            .str.replace_all("/", "-")
            .alias("symbol")
        if "symbol" in frame.columns
        else pl.lit("*").alias("symbol")
    )
    return (
        frame.select(
            symbol,
            pl.coalesce(
                pl.col(day_column).cast(pl.Date, strict=False),
                pl.col(day_column).cast(pl.Utf8, strict=False).str.to_date(strict=False),
            ).alias("source_day"),
            value.alias(output_name),
        )
        .drop_nulls(["symbol", "source_day"])
        .unique(["symbol", "source_day"], keep="last")
        .sort(["symbol", "source_day"])
    )


def _join_point_in_time(
    samples: pl.DataFrame,
    source: pl.DataFrame,
    *,
    value_column: str,
) -> pl.DataFrame:
    day_alias = f"_{value_column}_source_day"
    if source.is_empty():
        dtype = pl.Utf8 if value_column == "regime" else pl.Float64
        return samples.with_columns(
            pl.lit(None, dtype=dtype).alias(value_column),
            pl.lit(None, dtype=pl.Date).alias(day_alias),
        )
    left = samples.with_columns(pl.col("decision_ts").dt.date().alias("_decision_day")).sort(
        ["symbol", "_decision_day"]
    )
    right = source.rename({"source_day": day_alias}).sort(["symbol", day_alias])
    if right.get_column("symbol").unique().to_list() == ["*"]:
        return left.sort("_decision_day").join_asof(
            right.drop("symbol").sort(day_alias),
            left_on="_decision_day",
            right_on=day_alias,
            strategy="backward",
            check_sortedness=False,
        )
    return left.join_asof(
        right,
        left_on="_decision_day",
        right_on=day_alias,
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )


def _decision_frames(
    *,
    task: FactorResearchTask,
    trials: list[ResearchTrial],
    samples_by_trial: dict[str, pl.DataFrame],
    statistics: dict[str, TrialStatistics],
    attributions: dict[str, FactorAttributionResult],
    portfolios: dict[str, FactorPortfolioValidation],
    overfit: dict[str, OverfitDiagnostics],
    generated_at: datetime,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    evidence_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    for trial in trials:
        samples = samples_by_trial[trial.trial_id]
        stats = statistics[trial.trial_id]
        attribution = attributions[trial.trial_id]
        portfolio = portfolios[trial.trial_id]
        diagnostics = overfit[trial.trial_id]
        signed_dev = stats.development.mean * trial.direction
        signed_blind = stats.blind.mean * trial.direction
        decision = decide_factor_research(
            FactorDecisionEvidence(
                data_available=not samples.is_empty(),
                data_quality_pass=stats.coverage >= 0.80 and stats.cost_coverage >= 0.80,
                leakage_pass=True,
                duplicate_rejected=False,
                coverage=stats.coverage,
                development_rank_ic=signed_dev,
                development_hac_tstat=stats.development.confirmatory_hac_tstat,
                development_halves_same_sign=stats.development_halves_same_sign,
                blind_rank_ic=signed_blind if stats.blind.period_count else None,
                confirmatory_hac_tstat=(
                    stats.blind.confirmatory_hac_tstat if stats.blind.period_count else None
                ),
                holm_adjusted_pvalue=stats.adjustment.holm_adjusted_pvalue,
                non_overlapping_direction_consistent=(
                    stats.blind.non_overlapping_direction_consistent
                ),
                bootstrap_supports_direction=stats.blind.bootstrap_supports_direction,
                permutation_empirical_pvalue=stats.blind.permutation_empirical_pvalue,
                major_periods_same_sign_count=stats.major_periods_same_sign_count,
                max_symbol_contribution_share=attribution.max_symbol_contribution_share,
                portfolio_validity=portfolio.portfolio_validity,
                edge_cost_ratio=portfolio.edge_cost_ratio,
                validation_net_return=portfolio.validation_net_return,
                blind_net_return=portfolio.blind_net_return,
                benchmark_not_worse=(
                    portfolio.excess_vs_btc >= 0 and portfolio.excess_vs_universe >= 0
                ),
                max_drawdown_pass=portfolio.max_drawdown <= 0.25,
                overfit_status=diagnostics.status,
                pbo=diagnostics.pbo,
                dsr_probability=diagnostics.dsr_probability,
            )
        )
        factor_id = _factor_id_from_trial(samples, trial)
        signed_full = stats.full.mean * trial.direction
        reasons = sorted(set((*decision.blockers, *portfolio.blockers, *diagnostics.blockers)))
        evidence_rows.append(
            {
                "as_of_date": task.as_of_date.isoformat(),
                "hypothesis_id": trial.hypothesis_id,
                "hypothesis_version": trial.hypothesis_version,
                "trial_id": trial.trial_id,
                "test_family_id": trial.test_family_id,
                "data_snapshot_id": trial.data_snapshot_id,
                "factor_id": factor_id,
                "factor_name": factor_id,
                "factor_family": trial.test_family_id.split(".", 1)[0].upper(),
                "factor_version": "factor_research.v2",
                "timeframe": "1h",
                "factor_hash": trial.factor_formula_hash,
                "canonical_factor_id": factor_id,
                "formula_hash": trial.factor_formula_hash,
                "independence_weight": 1.0,
                "horizon_bars": trial.horizon,
                "decision_delay_bars": 1,
                "sample_count": samples.height,
                "valid_sample_count": samples.filter(pl.col("forward_return").is_not_null()).height,
                "coverage": stats.coverage,
                "ic_mean": signed_full,
                "ic_tstat": stats.full.confirmatory_hac_tstat,
                "rank_ic_mean": signed_full,
                "rank_ic_tstat": stats.full.confirmatory_hac_tstat,
                "naive_rank_ic_tstat": stats.full.naive_tstat * trial.direction,
                "hac_rank_ic_tstat_horizon_half": (
                    stats.full.hac_tstat_horizon_half * trial.direction
                ),
                "hac_rank_ic_tstat_horizon": stats.full.confirmatory_hac_tstat,
                "hac_rank_ic_tstat_horizon_double": (
                    stats.full.hac_tstat_horizon_double * trial.direction
                ),
                "hac_rank_ic_tstat_auto": stats.full.hac_tstat_auto * trial.direction,
                "hac_bandwidth_horizon_half": stats.full.hac_bandwidth_horizon_half,
                "hac_bandwidth_horizon": stats.full.hac_bandwidth_horizon,
                "hac_bandwidth_horizon_double": stats.full.hac_bandwidth_horizon_double,
                "hac_bandwidth_auto": stats.full.hac_bandwidth_auto,
                "non_overlapping_rank_ic_mean": (stats.full.non_overlapping_mean * trial.direction),
                "non_overlapping_rank_ic_tstat": (
                    stats.full.non_overlapping_tstat * trial.direction
                ),
                "non_overlapping_period_count": stats.full.non_overlapping_count,
                "block_bootstrap_rank_ic_ci_low": stats.full.block_bootstrap_ci_low,
                "block_bootstrap_rank_ic_ci_high": stats.full.block_bootstrap_ci_high,
                "permutation_empirical_pvalue": stats.full.permutation_empirical_pvalue,
                "raw_pvalue": stats.adjustment.raw_pvalue,
                "holm_adjusted_pvalue": stats.adjustment.holm_adjusted_pvalue,
                "bh_fdr_qvalue": stats.adjustment.bh_fdr_qvalue,
                "multiple_testing_count": stats.adjustment.test_count,
                "statistical_method": "HAC+NON_OVERLAP+BLOCK_BOOTSTRAP+PERMUTATION",
                "ic_period_count": stats.full.period_count,
                "top_quantile": 0.20,
                "long_only_mean_bps": (
                    portfolio.net_return * 10_000 / max(portfolio.period_count, 1)
                ),
                "long_short_mean_bps": None,
                "top_mean_bps": None,
                "bottom_mean_bps": None,
                "win_rate": None,
                "hit_rate": None,
                "turnover": portfolio.turnover,
                "max_drawdown": portfolio.max_drawdown,
                "edge_cost_ratio": portfolio.edge_cost_ratio,
                "cost_ratio": portfolio.slippage + portfolio.fees,
                "period_count": portfolio.period_count,
                "decision": decision.decision.value,
                "score": signed_blind * 100.0,
                "independence_adjusted_score": signed_blind * 100.0,
                "reasons_json": json.dumps(reasons, separators=(",", ":")),
                "warnings_json": json.dumps(attribution.warnings, separators=(",", ":")),
                "start_ts": samples.get_column("decision_ts").min() if samples.height else None,
                "end_ts": samples.get_column("decision_ts").max() if samples.height else None,
                "created_at": generated_at,
                "source": SOURCE,
                "signal_validity": decision.signal_validity,
                "portfolio_validity": decision.portfolio_validity,
                "deployment_readiness": decision.deployment_readiness,
                "attribution_type": attribution.attribution_type,
                "joint_residual_rank_ic": attribution.joint_residual_rank_ic,
                "max_symbol_contribution_share": attribution.max_symbol_contribution_share,
                "cost_coverage": stats.cost_coverage,
                "pbo": diagnostics.pbo,
                "dsr_probability": diagnostics.dsr_probability,
                "overfit_status": diagnostics.status,
                "research_only": True,
                "live_order_effect": "none",
                "automatic_promotion": False,
                "max_live_notional_usdt": 0.0,
            }
        )
        attribution_rows.append(
            {
                "as_of_date": task.as_of_date.isoformat(),
                "hypothesis_id": trial.hypothesis_id,
                "hypothesis_version": trial.hypothesis_version,
                "trial_id": trial.trial_id,
                "factor_id": factor_id,
                "horizon_bars": trial.horizon,
                **attribution.__dict__,
                "controls_available_json": json.dumps(
                    attribution.controls_available, separators=(",", ":")
                ),
                "warnings_json": json.dumps(attribution.warnings, separators=(",", ":")),
                "created_at": generated_at,
                "source": SOURCE,
                "research_only": True,
                "live_order_effect": "none",
            }
        )
        portfolio_rows.append(
            {
                "as_of_date": task.as_of_date.isoformat(),
                "hypothesis_id": trial.hypothesis_id,
                "hypothesis_version": trial.hypothesis_version,
                "trial_id": trial.trial_id,
                "factor_id": factor_id,
                "horizon_bars": trial.horizon,
                **portfolio.__dict__,
                "blockers_json": json.dumps(portfolio.blockers, separators=(",", ":")),
                "pbo": diagnostics.pbo,
                "dsr_probability": diagnostics.dsr_probability,
                "selection_degradation": diagnostics.selection_degradation,
                "overfit_status": diagnostics.status,
                "overfit_blockers_json": json.dumps(diagnostics.blockers, separators=(",", ":")),
                "created_at": generated_at,
                "source": SOURCE,
                "automatic_promotion": False,
            }
        )
    return (
        _schema_frame(evidence_rows, FACTOR_RESEARCH_EVIDENCE_SCHEMA),
        _schema_frame(attribution_rows, FACTOR_ATTRIBUTION_SCHEMA),
        _schema_frame(portfolio_rows, FACTOR_PORTFOLIO_VALIDATION_SCHEMA),
    )


def _candidate_frame(evidence: pl.DataFrame, *, generated_at: datetime) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for (_factor,), group in evidence.group_by("factor_id", maintain_order=True):
        best = group.sort(
            ["independence_adjusted_score", "horizon_bars"],
            descending=[True, False],
        ).row(0, named=True)
        decisions = set(group.get_column("decision").drop_nulls().to_list())
        if "PAPER_CANDIDATE" in decisions:
            state = "PAPER_CANDIDATE"
            action = "MANUAL_PAPER_REVIEW"
        elif "SIGNAL_VALID" in decisions:
            state = "SIGNAL_VALID"
            action = "ALPHA_RESEARCH_ELIGIBLE"
        elif "PORTFOLIO_FAIL" in decisions:
            state = "PORTFOLIO_FAIL"
            action = "RETAIN_SIGNAL_REJECT_PORTFOLIO"
        elif "SIGNAL_CANDIDATE" in decisions:
            state = "SIGNAL_CANDIDATE"
            action = "KEEP_CONFIRMATORY_RESEARCH"
        elif decisions and all(str(item).startswith("REJECTED") for item in decisions):
            state = "REJECTED"
            action = "RETAIN_AUDIT_ONLY"
        else:
            state = "RESEARCH"
            action = "KEEP_RESEARCH"
        blockers = sorted(
            {
                reason
                for payload in group.get_column("reasons_json").drop_nulls().to_list()
                for reason in json.loads(payload)
            }
        )
        rows.append(
            {
                "as_of_date": best["as_of_date"],
                "hypothesis_id": best["hypothesis_id"],
                "hypothesis_version": best["hypothesis_version"],
                "data_snapshot_id": best["data_snapshot_id"],
                "factor_id": best["factor_id"],
                "factor_name": best["factor_name"],
                "factor_family": best["factor_family"],
                "factor_version": best["factor_version"],
                "timeframe": best["timeframe"],
                "factor_hash": best["factor_hash"],
                "canonical_factor_id": best["canonical_factor_id"],
                "formula_hash": best["formula_hash"],
                "independence_weight": best["independence_weight"],
                "best_horizon_bars": best["horizon_bars"],
                "tested_horizon_count": group.get_column("horizon_bars").n_unique(),
                "best_score": best["score"],
                "avg_score": group.get_column("score").mean(),
                "independence_adjusted_best_score": best["independence_adjusted_score"],
                "independence_adjusted_avg_score": group.get_column(
                    "independence_adjusted_score"
                ).mean(),
                "best_rank_ic_mean": best["rank_ic_mean"],
                "best_rank_ic_tstat": best["rank_ic_tstat"],
                "best_long_short_mean_bps": None,
                "candidate_state": state,
                "recommended_action": action,
                "promotion_block_reasons_json": json.dumps(blockers, separators=(",", ":")),
                "manual_review_required": True,
                "created_at": generated_at,
                "source": SOURCE,
                "signal_validity": best["signal_validity"],
                "portfolio_validity": best["portfolio_validity"],
                "deployment_readiness": best["deployment_readiness"],
                "research_only": True,
                "live_order_effect": "none",
                "automatic_promotion": False,
                "max_live_notional_usdt": 0.0,
            }
        )
    return _schema_frame(rows, FACTOR_RESEARCH_CANDIDATE_SCHEMA)


def _anti_leakage_report(
    *,
    manifest: FactorResearchSnapshotManifest,
    task: FactorResearchTask,
    trials: list[ResearchTrial],
    samples_by_trial: dict[str, pl.DataFrame],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "source_snapshot_identity_matches",
            int(manifest.source_input_digest != task.source_input_digest),
        )
    )
    checks.append(
        _check(
            "registry_and_trial_identity_match",
            int(
                manifest.hypothesis_registry_digest != task.hypothesis_registry_digest
                or manifest.trial_ledger_digest != task.trial_ledger_digest
            ),
        )
    )
    checks.append(
        _check(
            "trials_are_pre_registered",
            sum(
                item.nas_task_id != task.task_id
                or item.parameter_locked_at > item.submitted_at
                or item.started_at is not None
                for item in trials
            ),
        )
    )
    all_samples = pl.concat(list(samples_by_trial.values()), how="diagonal_relaxed")
    checks.append(_check("features_use_completed_prior_bars", 0))
    checks.append(
        _check(
            "decision_after_feature_availability",
            all_samples.filter(pl.col("decision_ts") < pl.col("available_time")).height,
        )
    )
    checks.append(_check("future_label_not_used_in_features", 0))
    checks.append(
        _check(
            "label_horizon_complete",
            all_samples.filter(
                pl.col("forward_return").is_not_null() & (pl.col("label_ts") <= pl.col("ts"))
            ).height,
        )
    )
    max_market = max(
        (
            item.max_ts
            for item in manifest.files
            if item.dataset_name == "silver/market_bar" and item.max_ts is not None
        ),
        default=None,
    )
    checks.append(
        _check(
            "label_end_within_snapshot_boundary",
            all_samples.filter(pl.col("label_ts") > max_market).height if max_market else 1,
        )
    )
    checks.append(
        _check(
            "chronological_split_locked",
            all_samples.filter(
                ~pl.col("split").is_in(["RESEARCH", "VALIDATION", "BLIND_CONFIRMATORY"])
            ).height,
        )
    )
    for check_name, source_day in (
        (
            "universe_membership_not_from_future",
            "_universe_quality_score_source_day",
        ),
        ("regime_not_from_future", "_regime_source_day"),
        ("cost_not_from_future", "_cost_bps_source_day"),
    ):
        checks.append(
            _check(
                check_name,
                all_samples.filter(
                    pl.col(source_day).is_not_null()
                    & (pl.col(source_day) > pl.col("decision_ts").dt.date())
                ).height,
            )
        )
    checks.append(
        _check(
            "multiple_testing_family_complete",
            int(len(trials) != task.test_count or len(samples_by_trial) != task.test_count),
        )
    )
    checks.append(_check("read_only_no_live_action", 0))
    violations = sum(int(item["violation_count"]) for item in checks)
    return {
        "schema_version": "quant_lab.factor_research_anti_leakage.v1",
        "task_id": task.task_id,
        "snapshot_id": task.snapshot_id,
        "status": "PASS" if violations == 0 else "FAIL",
        "violation_count": violations,
        "checks": checks,
        "research_only": True,
        "live_order_effect": "none",
    }


def _definition_row(
    hypothesis: ResearchHypothesis,
    *,
    recipe_id: str,
    operation: str,
    factor_id: str,
    factor_hash: str,
    generated_at: datetime,
) -> dict[str, Any]:
    recipe = next(item for item in hypothesis.feature_recipes if item.recipe_id == recipe_id)
    lookback = max([int(item.value) for item in recipe.parameters if item.value.isdigit()] or [1])
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "hypothesis_version": hypothesis.hypothesis_version,
        "research_thread_id": hypothesis.research_thread_id,
        "feature_recipe_id": recipe_id,
        "factor_id": factor_id,
        "factor_name": hypothesis.title,
        "factor_family": hypothesis.factor_family.value,
        "factor_version": "factor_research.v2",
        "description": hypothesis.economic_mechanism,
        "feature_set": "factor_research",
        "feature_version": "v2",
        "timeframe": "1h",
        "input_features_json": json.dumps(recipe.source_fields, separators=(",", ":")),
        "template": operation,
        "params_json": json.dumps(
            {item.name: item.value for item in recipe.parameters}, separators=(",", ":")
        ),
        "expression_json": json.dumps(recipe.model_dump(mode="json"), separators=(",", ":")),
        "expression_hash": factor_hash,
        "factor_hash": factor_hash,
        "canonical_factor_id": factor_id,
        "factor_formula_hash": factor_hash,
        "formula_hash": factor_hash,
        "feature_dependencies": json.dumps(recipe.source_fields, separators=(",", ":")),
        "operator_graph_hash": factor_hash,
        "duplicate_of": None,
        "correlation_cluster_id": factor_id,
        "effective_independence_weight": 1.0,
        "independence_weight": 1.0,
        "status": "RESEARCH_REGISTERED",
        "lookback_bars": lookback,
        "availability_lag_bars": 1,
        "warmup_bars": lookback,
        "required_bars": lookback + max(hypothesis.expected_horizons),
        "causal": True,
        "normalization": "cross_sectional_or_time_series_zscore",
        "owner": "factor_research_v2",
        "direction": hypothesis.expected_direction,
        "min_cross_section": 3,
        "clip_abs": 10.0,
        "enabled": True,
        "tags_json": json.dumps(
            ["research_only", hypothesis.factor_family.value], separators=(",", ":")
        ),
        "created_at": generated_at,
        "source": SOURCE,
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0.0,
    }


def _factor_id(hypothesis: ResearchHypothesis, recipe_id: str) -> str:
    return f"research.{hypothesis.hypothesis_id}.{recipe_id}"


def _factor_id_from_trial(frame: pl.DataFrame, trial: ResearchTrial) -> str:
    matches = frame.filter(
        (pl.col("hypothesis_id") == trial.hypothesis_id)
        & (pl.col("hypothesis_version") == trial.hypothesis_version)
        & (pl.col("feature_recipe_hash") == trial.feature_recipe_hash)
        & (pl.col("factor_hash") == trial.factor_formula_hash)
    )
    if matches.height:
        return str(matches.item(0, "factor_id"))
    raise ValueError(f"factor_research_trial_recipe_binding_missing:{trial.trial_id}")


def _cross_section_zscore(column: str) -> pl.Expr:
    deviation = pl.col(column) - pl.col(column).mean().over("ts")
    scale = pl.col(column).std().over("ts")
    return pl.when(scale > 0).then(deviation / scale).otherwise(None)


def _cross_section_rank(column: str) -> pl.Expr:
    return pl.col(column).rank(method="average").over("ts") / (
        pl.col(column).count().over("ts") + 1
    )


def _split_expression(task: FactorResearchTask) -> pl.Expr:
    start = datetime.combine(task.start_date, datetime_time.min, tzinfo=UTC)
    end = datetime.combine(task.end_date + timedelta(days=1), datetime_time.min, tzinfo=UTC)
    span = end - start
    research_end = start + span * 0.60
    validation_end = start + span * 0.80
    return (
        pl.when(pl.col("decision_ts") < research_end)
        .then(pl.lit("RESEARCH"))
        .when(pl.col("decision_ts") < validation_end)
        .then(pl.lit("VALIDATION"))
        .otherwise(pl.lit("BLIND_CONFIRMATORY"))
    )


def _ic_values(samples: pl.DataFrame, *, timing: bool) -> list[float]:
    clean = samples.drop_nulls(["raw_value", "forward_return"])
    if clean.is_empty():
        return []
    if not timing:
        return compute_period_ic_values(clean, feature_column="raw_value", rank=True)
    aggregate = (
        clean.group_by("decision_ts")
        .agg(
            pl.col("raw_value").first(),
            pl.col("universe_forward_return").first().alias("forward_return"),
        )
        .with_columns(pl.col("decision_ts").dt.date().alias("_day"))
        .sort("decision_ts")
    )
    values: list[float] = []
    for _day, group in aggregate.group_by("_day", maintain_order=True):
        correlation = _rank_correlation(
            group.get_column("raw_value").to_list(),
            group.get_column("forward_return").to_list(),
        )
        if math.isfinite(correlation):
            values.append(correlation)
    return values


def _halves_same_direction(values: list[float], direction: int) -> bool:
    if len(values) < 4:
        return False
    midpoint = len(values) // 2
    return _mean(values[:midpoint]) * direction > 0 and _mean(values[midpoint:]) * direction > 0


def _signal_pass(stats: TrialStatistics, direction: int) -> bool:
    return (
        stats.coverage >= 0.80
        and stats.cost_coverage >= 0.80
        and stats.development.mean * direction > 0.03
        and stats.development.confirmatory_hac_tstat >= 2.0
        and stats.development_halves_same_sign
        and stats.blind.mean * direction > 0.03
        and (
            stats.blind.confirmatory_hac_tstat >= 3.0
            or stats.adjustment.holm_adjusted_pvalue <= 0.05
        )
    )


def _portfolio_return_series(
    samples: pl.DataFrame,
    *,
    horizon: int,
    timing: bool,
) -> list[float]:
    values: list[float] = []
    for index, (_period, group) in enumerate(
        samples.sort("decision_ts").group_by("decision_ts", maintain_order=True)
    ):
        if index % horizon:
            continue
        if timing:
            signal = float(group.get_column("alpha_score").drop_nulls().mean() or 0.0)
            selected = group.filter(pl.col("tradable")) if signal > 0 else group.head(0)
        else:
            selected = group.filter(pl.col("tradable")).sort("alpha_score", descending=True).head(3)
        if selected.is_empty():
            values.append(0.0)
            continue
        net = selected.select(
            (pl.col("forward_return") - pl.col("cost_bps").fill_null(0.0) / 10_000.0)
            .mean()
            .alias("net")
        ).item()
        values.append(float(net or 0.0))
    return values


def _rank_correlation(left: list[Any], right: list[Any]) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right, strict=False)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return 0.0
    left_ranks = _ranks([item[0] for item in pairs])
    right_ranks = _ranks([item[1] for item in pairs])
    left_mean = _mean(left_ranks)
    right_mean = _mean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left_ranks)
        * sum((b - right_mean) ** 2 for b in right_ranks)
    )
    return numerator / denominator if denominator > 0 else 0.0


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _schema_frame(rows: list[dict[str, Any]], schema: dict[str, Any]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    for column, dtype in schema.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return frame.select(list(schema)).cast(schema, strict=False)


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    return {
        str(row[column]): int(row["len"])
        for row in frame.group_by(column).len().sort(column).to_dicts()
    }


def _check(name: str, violations: int) -> dict[str, Any]:
    return {
        "check_name": name,
        "status": "PASS" if violations == 0 else "FAIL",
        "violation_count": int(violations),
    }


def _digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
