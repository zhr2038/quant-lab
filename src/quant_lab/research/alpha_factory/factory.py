from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.backtest.reports import build_factor_forward_validation
from quant_lab.data.lake import count_parquet_rows, read_parquet_dataset, write_parquet_dataset
from quant_lab.factors.composite_factory import build_factor_strategy_bridge_candidates
from quant_lab.research.second_stage_alpha_factory import (
    SECOND_STAGE_CANDIDATES,
    SECOND_STAGE_SAMPLE_DATASET,
    SECOND_STAGE_SUMMARY_DATASET,
    build_and_publish_second_stage_alpha_factory,
)
from quant_lab.research.strategy_evidence import (
    STRATEGY_EVIDENCE_SAMPLE_DATASET,
    SUMMARY_SCHEMA,
    normalize_strategy_evidence_decisions,
)
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

SOURCE_NAME = "research.alpha_factory.v0.1"
SCHEMA_VERSION = "alpha_factory.v0.1"
MAX_DAILY_CANDIDATES = 200
FACTOR_BRIDGE_TEMPLATE_FAMILY = "factor_strategy_bridge"
FACTOR_BRIDGE_CANDIDATE_PREFIX = "v5.factor_bridge."
FAST_MICROSTRUCTURE_BRIDGE_CANDIDATE_PREFIX = "v5.fast_microstructure_bridge."
FACTOR_BRIDGE_CANDIDATE_PREFIXES = (
    FACTOR_BRIDGE_CANDIDATE_PREFIX,
    FAST_MICROSTRUCTURE_BRIDGE_CANDIDATE_PREFIX,
)
FACTOR_BRIDGE_TEMPLATE_PATTERN = f"{FACTOR_BRIDGE_CANDIDATE_PREFIX}*"
FAST_MICROSTRUCTURE_BRIDGE_TEMPLATE_PATTERN = (
    f"{FAST_MICROSTRUCTURE_BRIDGE_CANDIDATE_PREFIX}*"
)
FACTOR_BRIDGE_TEMPLATE_PATTERNS = (
    FACTOR_BRIDGE_TEMPLATE_PATTERN,
    FAST_MICROSTRUCTURE_BRIDGE_TEMPLATE_PATTERN,
)
FACTOR_BRIDGE_RECOMPUTE_MAX_FACTOR_VALUE_ROWS = 50_000
FACTOR_BRIDGE_RECOMPUTE_MAX_MARKET_BAR_ROWS = 500_000

ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET = (
    Path("gold") / "alpha_factory_template_registry"
)
ALPHA_FACTORY_CANDIDATE_DATASET = Path("gold") / "alpha_factory_candidate"
ALPHA_FACTORY_RESULT_DATASET = Path("gold") / "alpha_factory_result"
ALPHA_FACTORY_PROMOTION_QUEUE_DATASET = Path("gold") / "alpha_factory_promotion_queue"
STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"
FACTOR_CANDIDATE_DATASET = Path("gold") / "factor_candidate"
FACTOR_VALUE_DATASET = Path("gold") / "factor_value"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
MARKET_REGIME_DATASET = Path("gold") / "market_regime_daily"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
FACTOR_BRIDGE_REPORT_MEMBER = "reports/factor_strategy_bridge_candidates.csv"

ALPHA_FACTORY_CANDIDATES = frozenset([*SECOND_STAGE_CANDIDATES, "v5.alt_impulse_shadow"])

TEMPLATE_REGISTRY_SCHEMA: dict[str, Any] = {
    "template_id": pl.Utf8,
    "template_family": pl.Utf8,
    "enabled": pl.Boolean,
    "description": pl.Utf8,
    "universe_scope": pl.Utf8,
    "allowed_regimes": pl.Utf8,
    "parameter_space_json": pl.Utf8,
    "max_candidates_per_day": pl.Int64,
    "safety_mode": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

CANDIDATE_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "template_name": pl.Utf8,
    "candidate_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "regime_state": pl.Utf8,
    "horizon_hours": pl.Int64,
    "parameter_json": pl.Utf8,
    "whitelist_json": pl.Utf8,
    "blacklist_json": pl.Utf8,
    "source_dataset": pl.Utf8,
    "candidate_state": pl.Utf8,
    "max_live_notional_usdt": pl.Float64,
    "safety_mode": pl.Utf8,
    "source": pl.Utf8,
}

DEFAULT_TEMPLATE_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "template_id": "expanded_relative_strength_v1",
        "template_family": "expanded_relative_strength",
        "enabled": True,
        "description": "Batch scan expanded spot symbols for relative strength candidates.",
        "universe_scope": "expanded_crypto_universe",
        "allowed_regimes": ["TREND_UP", "ALT_IMPULSE", "SOL_STRENGTH", "LOW_VOL"],
        "max_candidates_per_day": 200,
        "parameter_space": {
            "lookback_hours": [4, 8, 12, 24],
            "top_k": [1, 3, 5],
            "hold_hours": [4, 8, 12, 24, 48],
            "max_spread_bps": [1, 2, 5],
            "btc_corr_max": [0.3, 0.5, 0.7],
            "min_quality_score": [75, 80],
            "min_24h_volume_usdt": [5_000_000, 10_000_000],
        },
        "candidate_patterns": [
            "v5.expanded_relative_strength_top1_shadow",
            "v5.expanded_relative_strength_top3_shadow",
            "v5.expanded_relative_strength_rotation_shadow",
        ],
    },
    {
        "template_id": "exit_policy_review_v1",
        "template_family": "exit_policy_review",
        "enabled": True,
        "description": "Compare actual exits against fixed-hold exit policies in paper research.",
        "universe_scope": "v5_current_universe",
        "allowed_regimes": ["TREND_UP", "TREND_DOWN", "SIDEWAYS", "HIGH_VOL", "LOW_VOL"],
        "max_candidates_per_day": 80,
        "parameter_space": {
            "target_strategy": [
                "btc_strict_probe",
                "eth_f3_dominant",
                "sol_f4_volume_expansion",
                "sol_protect_alpha6_low",
            ],
            "exit_policy": [
                "actual_exit",
                "fixed_hold_4h",
                "fixed_hold_8h",
                "fixed_hold_12h",
                "fixed_hold_24h",
                "fixed_hold_48h",
            ],
        },
        "candidate_patterns": [
            "v5.btc_strict_probe_exit_policy_review",
            "v5.eth_f3_exit_policy_review",
            "v5.sol_paper_exit_policy_review",
        ],
    },
    {
        "template_id": "futures_hedge_shadow_v1",
        "template_family": "futures_hedge_shadow",
        "enabled": True,
        "description": (
            "Spot inverse proxy for future futures hedge and downtrend short research. "
            "No perp mark, funding, or open-interest data is observed in v0.1."
        ),
        "universe_scope": "okx_spot_inverse_proxy_for_perpetual_shadow",
        "allowed_regimes": ["RISK_OFF", "TREND_DOWN", "HIGH_VOL"],
        "max_candidates_per_day": 60,
        "parameter_space": {
            "hedge_symbol": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
            "hedge_ratio": [0.25, 0.5, 0.75],
            "hold_hours": [4, 8, 12, 24],
            "max_funding_cost_bps": [5, 10, 20],
        },
        "candidate_patterns": [
            "v5.futures_risk_off_hedge_proxy_shadow",
            "v5.futures_downtrend_short_proxy_shadow",
        ],
    },
    {
        "template_id": "pair_market_neutral_shadow_v1",
        "template_family": "pair_market_neutral_shadow",
        "enabled": True,
        "description": "Paper-only pair and basket relative-value research.",
        "universe_scope": "v5_current_and_expanded_universe",
        "allowed_regimes": ["SIDEWAYS", "LOW_VOL", "LIQUIDITY_THIN"],
        "max_candidates_per_day": 80,
        "parameter_space": {
            "pair": ["ETH/BTC", "SOL/ETH", "ALT_BASKET/BTC"],
            "zscore_lookback_hours": [24, 48, 72],
            "entry_zscore": [1.0, 1.5, 2.0],
            "hold_hours": [8, 12, 24, 48],
        },
        "candidate_patterns": [
            "v5.pair_trade_eth_btc_shadow",
            "v5.pair_trade_sol_eth_shadow",
            "v5.alt_basket_vs_btc_shadow",
        ],
    },
    {
        "template_id": "alt_impulse_regime_v1",
        "template_family": "alt_impulse_regime",
        "enabled": True,
        "description": "Regime-aware alt impulse shadow research.",
        "universe_scope": "v5_and_expanded_alt_universe",
        "allowed_regimes": ["ALT_IMPULSE", "TREND_UP"],
        "max_candidates_per_day": 80,
        "parameter_space": {
            "regime_state": ["ALT_IMPULSE", "TREND_UP"],
            "hold_hours": [4, 8, 12, 24, 48, 72, 120],
            "min_breadth_count": [3, 5, 8],
            "min_alpha6_score": [0.4, 0.6],
        },
        "candidate_patterns": ["v5.alt_impulse_shadow"],
    },
    {
        "template_id": "factor_strategy_bridge_v1",
        "template_family": FACTOR_BRIDGE_TEMPLATE_FAMILY,
        "enabled": True,
        "description": (
            "Route Factor Factory forward-validation pass rows into Alpha Factory "
            "strategy-review candidates."
        ),
        "universe_scope": "factor_factory_forward_validation_pass",
        "allowed_regimes": ["RISK_ON_CONFIRMED", "TREND_UP", "SIDEWAYS", "RISK_OFF"],
        "max_candidates_per_day": 60,
        "parameter_space": {
            "bridge_source": ["factor_strategy_bridge_candidates"],
            "review_mode": ["strategy_review_only"],
            "dry_run_first": [True],
        },
        "candidate_patterns": list(FACTOR_BRIDGE_TEMPLATE_PATTERNS),
    },
)

RESULT_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "template_name": pl.Utf8,
    "candidate_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "regime_state": pl.Utf8,
    "horizon_hours": pl.Int64,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "median_net_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "avg_mae_bps": pl.Float64,
    "avg_mfe_bps": pl.Float64,
    "cost_quality_score": pl.Float64,
    "recent_sample_sufficient": pl.Boolean,
    "paper_ready_block_reasons": pl.Utf8,
    "train_metrics_json": pl.Utf8,
    "validation_metrics_json": pl.Utf8,
    "recent_7d_metrics_json": pl.Utf8,
    "stability_score": pl.Float64,
    "tail_loss_penalty": pl.Float64,
    "recent_degradation_penalty": pl.Float64,
    "validation_failure_penalty": pl.Float64,
    "alpha_factory_score": pl.Float64,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "start_ts": pl.Datetime(time_zone="UTC"),
    "end_ts": pl.Datetime(time_zone="UTC"),
    "max_live_notional_usdt": pl.Float64,
    "source": pl.Utf8,
}

PROMOTION_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "template_name": pl.Utf8,
    "candidate_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon_hours": pl.Int64,
    "promotion_state": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "action": pl.Utf8,
    "reasons": pl.Utf8,
    "max_live_notional_usdt": pl.Float64,
    "manual_live_approval_required": pl.Boolean,
    "source": pl.Utf8,
}


class AlphaFactoryBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    candidate_rows: int = Field(ge=0)
    result_rows: int = Field(ge=0)
    promotion_rows: int = Field(ge=0)
    template_registry_rows: int = Field(ge=0)
    strategy_evidence_rows: int = Field(ge=0)
    strategy_evidence_sample_rows: int = Field(ge=0)
    decision_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_alpha_factory(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
    lookback_days: int = 30,
    max_candidates: int = MAX_DAILY_CANDIDATES,
) -> AlphaFactoryBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    generated_at = datetime.now(UTC)
    registry = publish_alpha_factory_template_registry(root, generated_at=generated_at)
    registry_lookup = template_registry_lookup(registry)

    second_stage = build_and_publish_second_stage_alpha_factory(
        root,
        as_of_date=day,
        lookback_days=lookback_days,
    )
    summary = _alpha_factory_source_summary(root, day)
    samples = _alpha_factory_source_samples(root, day)
    candidates = build_alpha_factory_candidates(
        summary,
        as_of_date=day,
        generated_at=generated_at,
        max_candidates=max_candidates,
        registry_lookup=registry_lookup,
    )
    results = build_alpha_factory_results(
        summary,
        samples=samples,
        as_of_date=day,
        generated_at=generated_at,
        max_candidates=max_candidates,
        registry_lookup=registry_lookup,
    )
    promotion = build_alpha_factory_promotion_queue(results, generated_at=generated_at)

    write_parquet_dataset(candidates, root / ALPHA_FACTORY_CANDIDATE_DATASET)
    write_parquet_dataset(results, root / ALPHA_FACTORY_RESULT_DATASET)
    write_parquet_dataset(promotion, root / ALPHA_FACTORY_PROMOTION_QUEUE_DATASET)
    evidence_rows = _publish_alpha_factory_results_to_strategy_evidence(root, results, day)

    return AlphaFactoryBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        candidate_rows=candidates.height,
        result_rows=results.height,
        promotion_rows=promotion.height,
        template_registry_rows=registry.height,
        strategy_evidence_rows=evidence_rows,
        strategy_evidence_sample_rows=read_parquet_dataset(
            root / "gold" / "strategy_evidence_sample"
        ).height,
        decision_counts=_decision_counts(results),
        warnings=list(second_stage.warnings)
        + ([] if candidates.height <= max_candidates else ["alpha_factory_candidate_cap_applied"]),
    )


def build_alpha_factory_candidates(
    summary: pl.DataFrame,
    *,
    as_of_date: date,
    generated_at: datetime | None = None,
    max_candidates: int = MAX_DAILY_CANDIDATES,
    registry_lookup: dict[str, dict[str, Any]] | None = None,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    registry = registry_lookup or template_registry_lookup(
        build_default_template_registry(generated)
    )
    rows = []
    for row in _ranked_summary_rows(
        summary,
        as_of_date=as_of_date,
        max_candidates=max_candidates,
        registry_lookup=registry,
    ):
        candidate = str(row.get("strategy_candidate") or "")
        template = _template_for_candidate(candidate, registry)
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": SCHEMA_VERSION,
                "template_name": template,
                "candidate_id": _candidate_id(row),
                "strategy_candidate": candidate,
                "symbol": normalize_symbol(row.get("symbol")) or "UNKNOWN",
                "regime_state": str(row.get("regime_state") or "UNKNOWN"),
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "parameter_json": safe_json_dumps(
                    _parameter_payload(row, template, registry)
                ),
                "whitelist_json": safe_json_dumps([normalize_symbol(row.get("symbol"))]),
                "blacklist_json": safe_json_dumps([]),
                "source_dataset": str(row.get("source_dataset") or "gold/strategy_evidence"),
                "candidate_state": "RESEARCH",
                "max_live_notional_usdt": 0.0,
                "safety_mode": "paper_shadow_only",
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=CANDIDATE_SCHEMA, orient="row") if rows else pl.DataFrame(
        schema=CANDIDATE_SCHEMA
    )


def build_alpha_factory_results(
    summary: pl.DataFrame,
    *,
    samples: pl.DataFrame | None = None,
    as_of_date: date,
    generated_at: datetime | None = None,
    max_candidates: int = MAX_DAILY_CANDIDATES,
    registry_lookup: dict[str, dict[str, Any]] | None = None,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    registry = registry_lookup or template_registry_lookup(
        build_default_template_registry(generated)
    )
    sample_groups = _sample_groups(samples if samples is not None else pl.DataFrame(), as_of_date)
    rows = []
    for row in _ranked_summary_rows(
        summary,
        as_of_date=as_of_date,
        max_candidates=max_candidates,
        registry_lookup=registry,
    ):
        split_metrics = _split_sample_metrics(
            sample_groups.get(_sample_group_key(row), []),
            row,
        )
        avg_mae_bps = _float(split_metrics["all"].get("avg_mae_bps"))
        avg_mfe_bps = _float(split_metrics["all"].get("avg_mfe_bps"))
        cost_source_mix = str(row.get("cost_source_mix") or "{}")
        cost_quality_score = _cost_quality_score(cost_source_mix)
        recent_complete = _int(split_metrics["recent_7d"].get("complete_sample_count")) or 0
        recent_sample_sufficient = recent_complete >= 5
        is_futures_proxy = _is_futures_proxy_candidate(row.get("strategy_candidate"))
        paper_ready_block_reasons = _paper_ready_block_reasons(
            complete_sample_count=_int(row.get("complete_sample_count")) or 0,
            p25_net_bps=_float(row.get("p25_net_bps")),
            win_rate=_float(row.get("win_rate")),
            validation_metrics=split_metrics["validation"],
            recent_7d_metrics=split_metrics["recent_7d"],
            avg_mae_bps=avg_mae_bps,
            cost_quality_score=cost_quality_score,
            cost_source_mix=cost_source_mix,
            is_futures_proxy=is_futures_proxy,
        )
        scores = _alpha_factory_scores(
            avg_net_bps=_float(row.get("avg_net_bps")),
            p25_net_bps=_float(row.get("p25_net_bps")),
            train_metrics=split_metrics["train"],
            validation_metrics=split_metrics["validation"],
            recent_7d_metrics=split_metrics["recent_7d"],
        )
        decision, reasons = alpha_factory_decision(
            sample_count=_int(row.get("sample_count")) or 0,
            complete_sample_count=_int(row.get("complete_sample_count")) or 0,
            avg_net_bps=_float(row.get("avg_net_bps")),
            p25_net_bps=_float(row.get("p25_net_bps")),
            win_rate=_float(row.get("win_rate")),
            validation_metrics=split_metrics["validation"],
            recent_7d_metrics=split_metrics["recent_7d"],
            avg_mae_bps=avg_mae_bps,
            cost_quality_score=cost_quality_score,
            paper_ready_block_reasons=paper_ready_block_reasons,
        )
        if is_futures_proxy:
            if decision == "PAPER_READY":
                decision = "KEEP_SHADOW"
            reasons = _dedupe_text(
                [
                    *reasons,
                    "spot_inverse_proxy_only",
                    "futures_data_missing",
                    "funding_not_observable",
                ]
            )
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": SCHEMA_VERSION,
                "template_name": _template_for_candidate(
                    row.get("strategy_candidate"),
                    registry,
                ),
                "candidate_id": _candidate_id(row),
                "strategy_candidate": str(row.get("strategy_candidate") or ""),
                "symbol": normalize_symbol(row.get("symbol")) or "UNKNOWN",
                "regime_state": str(row.get("regime_state") or "UNKNOWN"),
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "sample_count": _int(row.get("sample_count")) or 0,
                "complete_sample_count": _int(row.get("complete_sample_count")) or 0,
                "avg_net_bps": _float(row.get("avg_net_bps")),
                "median_net_bps": _float(row.get("median_net_bps")),
                "p25_net_bps": _float(row.get("p25_net_bps")),
                "win_rate": _float(row.get("win_rate")),
                "cost_source_mix": cost_source_mix,
                "avg_mae_bps": avg_mae_bps,
                "avg_mfe_bps": avg_mfe_bps,
                "cost_quality_score": cost_quality_score,
                "recent_sample_sufficient": recent_sample_sufficient,
                "paper_ready_block_reasons": safe_json_dumps(paper_ready_block_reasons),
                "train_metrics_json": safe_json_dumps(split_metrics["train"]),
                "validation_metrics_json": safe_json_dumps(split_metrics["validation"]),
                "recent_7d_metrics_json": safe_json_dumps(split_metrics["recent_7d"]),
                "stability_score": scores["stability_score"],
                "tail_loss_penalty": scores["tail_loss_penalty"],
                "recent_degradation_penalty": scores["recent_degradation_penalty"],
                "validation_failure_penalty": scores["validation_failure_penalty"],
                "alpha_factory_score": scores["alpha_factory_score"],
                "decision": decision,
                "decision_reasons": safe_json_dumps(
                    _dedupe_text([*reasons, *scores["score_reasons"]])
                ),
                "recommended_mode": _recommended_mode(decision),
                "start_ts": _parse_dt(row.get("start_ts")),
                "end_ts": _parse_dt(row.get("end_ts")),
                "max_live_notional_usdt": 0.0,
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=RESULT_SCHEMA, orient="row") if rows else pl.DataFrame(
        schema=RESULT_SCHEMA
    )


def build_alpha_factory_promotion_queue(
    results: pl.DataFrame,
    *,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    rows = []
    for row in results.to_dicts() if not results.is_empty() else []:
        decision = str(row.get("decision") or "RESEARCH").upper()
        reasons = _json_list(row.get("decision_reasons"))
        rows.append(
            {
                "as_of_date": str(row.get("as_of_date") or ""),
                "generated_at": generated,
                "schema_version": SCHEMA_VERSION,
                "template_name": str(row.get("template_name") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "strategy_candidate": str(row.get("strategy_candidate") or ""),
                "symbol": normalize_symbol(row.get("symbol")) or "UNKNOWN",
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "promotion_state": decision,
                "recommended_mode": _recommended_mode(decision),
                "action": _promotion_action(decision),
                "reasons": safe_json_dumps(
                    _dedupe_text([*reasons, "alpha_factory_live_disabled"])
                ),
                "max_live_notional_usdt": 0.0,
                "manual_live_approval_required": True,
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=PROMOTION_SCHEMA, orient="row") if rows else pl.DataFrame(
        schema=PROMOTION_SCHEMA
    )


def alpha_factory_decision(
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    p25_net_bps: float | None,
    win_rate: float | None,
    validation_metrics: dict[str, Any] | None = None,
    recent_7d_metrics: dict[str, Any] | None = None,
    avg_mae_bps: float | None = None,
    cost_quality_score: float | None = None,
    cost_source_mix: Any = None,
    paper_ready_block_reasons: list[str] | None = None,
    is_futures_proxy: bool = False,
) -> tuple[str, list[str]]:
    if complete_sample_count < 10:
        return "RESEARCH", ["insufficient_complete_samples"]
    if sample_count >= 30 and (avg_net_bps or 0.0) < 0.0 and (win_rate or 0.0) < 0.45:
        return "KILL", ["negative_after_cost_edge", "win_rate_below_threshold"]

    reasons: list[str] = []
    validation_avg = _float((validation_metrics or {}).get("avg_net_bps"))
    validation_complete = _int((validation_metrics or {}).get("complete_sample_count")) or 0
    if validation_complete <= 0:
        reasons.append("insufficient_validation_samples")
    elif validation_avg is None:
        reasons.append("validation_avg_net_bps_missing")
    elif validation_avg is not None and validation_avg <= 0.0:
        reasons.append("validation_avg_net_bps_non_positive")

    recent_avg = _float((recent_7d_metrics or {}).get("avg_net_bps"))
    recent_complete = _int((recent_7d_metrics or {}).get("complete_sample_count")) or 0
    recent_insufficient = recent_complete < 5
    recent_negative = recent_avg is not None and recent_avg < 0.0
    if recent_insufficient:
        reasons.append("insufficient_recent_samples")
    elif recent_negative:
        reasons.append("recent_7d_avg_net_bps_negative")

    paper_block_reasons = paper_ready_block_reasons
    if paper_block_reasons is None:
        paper_block_reasons = _paper_ready_block_reasons(
            complete_sample_count=complete_sample_count,
            p25_net_bps=p25_net_bps,
            win_rate=win_rate,
            validation_metrics=validation_metrics or {},
            recent_7d_metrics=recent_7d_metrics or {},
            avg_mae_bps=avg_mae_bps,
            cost_quality_score=cost_quality_score,
            cost_source_mix=cost_source_mix,
            is_futures_proxy=is_futures_proxy,
        )

    if not paper_block_reasons:
        return "PAPER_READY", _dedupe_text(
            [*reasons, "paper_ready_thresholds_met", "live_disabled"]
        )
    if complete_sample_count >= 10 and avg_net_bps is not None and avg_net_bps > 0.0:
        return "KEEP_SHADOW", _dedupe_text(
            [
                *reasons,
                *paper_block_reasons,
                "positive_after_cost_edge",
                "collect_more_samples",
            ]
        )
    return "RESEARCH", _dedupe_text([*reasons, *paper_block_reasons, "edge_not_confirmed"])


def alpha_factory_daily_md(
    *,
    candidates: pl.DataFrame,
    results: pl.DataFrame,
    promotion_queue: pl.DataFrame,
    as_of_date: str | date | None = None,
) -> str:
    day = str(as_of_date or _latest_as_of_date(results) or _latest_as_of_date(candidates) or "auto")
    lines = [
        f"# Alpha Factory Daily - {day}",
        "",
        "Alpha Factory is read-only. It can only produce research, shadow, or paper candidates.",
        "It never changes V5 live symbols and never creates live orders.",
        "",
        f"- candidates: {candidates.height}",
        f"- results: {results.height}",
        f"- promotion queue: {promotion_queue.height}",
        "- max_live_notional_usdt: 0",
        "",
        "## Decision Counts",
        "",
    ]
    counts = _decision_counts(results)
    if counts:
        for decision, count in sorted(counts.items()):
            lines.append(f"- {decision}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Templates", ""])
    default_templates = build_default_template_registry(datetime.now(UTC))
    families = _template_families_from_results(results, default_templates)
    for template in families:
        frame = _filter_template(results, template)
        lines.append(f"- {template}: {frame.height} rows")
    return "\n".join(lines).rstrip() + "\n"


def build_default_template_registry(generated_at: datetime | None = None) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    rows = []
    for template in DEFAULT_TEMPLATE_REGISTRY:
        parameter_space = {
            **template["parameter_space"],
            "max_live_notional_usdt": [0],
        }
        rows.append(
            {
                "template_id": template["template_id"],
                "template_family": template["template_family"],
                "enabled": bool(template["enabled"]),
                "description": template["description"],
                "universe_scope": template["universe_scope"],
                "allowed_regimes": safe_json_dumps(template["allowed_regimes"]),
                "parameter_space_json": safe_json_dumps(parameter_space),
                "max_candidates_per_day": int(template["max_candidates_per_day"]),
                "safety_mode": "paper_shadow_only",
                "created_at": generated,
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=TEMPLATE_REGISTRY_SCHEMA, orient="row")


def publish_alpha_factory_template_registry(
    lake_root: str | Path,
    *,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    root = Path(lake_root)
    published_at = generated_at or datetime.now(UTC)
    existing = read_parquet_dataset(root / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET)
    defaults = build_default_template_registry(published_at)
    registry = _merge_template_registry(existing, defaults)
    if not registry.is_empty():
        registry = registry.with_columns(pl.lit(published_at).alias("created_at"))
    write_parquet_dataset(registry, root / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET)
    return registry


def template_registry_lookup(registry: pl.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if registry.is_empty():
        return lookup
    for row in registry.to_dicts():
        if not bool(row.get("enabled")):
            continue
        if str(row.get("safety_mode") or "") != "paper_shadow_only":
            continue
        family = str(row.get("template_family") or "")
        if not family:
            continue
        patterns = _candidate_patterns_for_family(family)
        payload = {
            **row,
            "allowed_regimes_list": _json_list(row.get("allowed_regimes")),
            "parameter_space": _json_dict(row.get("parameter_space_json")),
        }
        for candidate in patterns:
            lookup[candidate] = payload
        if family == FACTOR_BRIDGE_TEMPLATE_FAMILY:
            for pattern in FACTOR_BRIDGE_TEMPLATE_PATTERNS:
                lookup[pattern] = payload
    return lookup


def _publish_alpha_factory_results_to_strategy_evidence(
    root: Path,
    results: pl.DataFrame,
    day: date,
) -> int:
    dataset = root / STRATEGY_EVIDENCE_DATASET
    existing = read_parquet_dataset(dataset)
    summary = _results_to_strategy_evidence(results)
    if existing.is_empty():
        write_parquet_dataset(summary, dataset)
        return summary.height
    retained = existing
    if "as_of_date" in retained.columns and "strategy_candidate" in retained.columns:
        retained = retained.filter(
            ~(
                (pl.col("as_of_date").cast(pl.Utf8) == day.isoformat())
                & _alpha_factory_managed_candidate_expr(pl.col("strategy_candidate"))
            )
        )
    combined = pl.concat([retained, summary], how="diagonal_relaxed")
    combined = normalize_strategy_evidence_decisions(combined)
    write_parquet_dataset(combined, dataset)
    return combined.height


def _results_to_strategy_evidence(results: pl.DataFrame) -> pl.DataFrame:
    if results.is_empty():
        return pl.DataFrame(schema=SUMMARY_SCHEMA)
    rows = []
    for row in results.to_dicts():
        rows.append(
            {
                "strategy": "v5",
                "evidence_version": SCHEMA_VERSION,
                "as_of_date": str(row.get("as_of_date") or ""),
                "strategy_candidate": str(row.get("strategy_candidate") or ""),
                "candidate_name": str(row.get("strategy_candidate") or ""),
                "symbol": normalize_symbol(row.get("symbol")) or "UNKNOWN",
                "regime_state": str(row.get("regime_state") or "UNKNOWN"),
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "sample_count": _int(row.get("sample_count")) or 0,
                "complete_sample_count": _int(row.get("complete_sample_count")) or 0,
                "avg_net_bps": _float(row.get("avg_net_bps")),
                "median_net_bps": _float(row.get("median_net_bps")),
                "p25_net_bps": _float(row.get("p25_net_bps")),
                "win_rate": _float(row.get("win_rate")),
                "cost_source_mix": str(row.get("cost_source_mix") or "{}"),
                "decision": _strategy_evidence_decision(str(row.get("decision") or "")),
                "decision_reasons": safe_json_dumps(
                    _dedupe_text(
                        [
                            *_json_list(row.get("decision_reasons")),
                            "alpha_factory_live_disabled",
                            "max_live_notional_zero",
                        ]
                    )
                ),
                "start_ts": _parse_dt(row.get("start_ts")),
                "end_ts": _parse_dt(row.get("end_ts")),
                "created_at": _parse_dt(row.get("generated_at")) or datetime.now(UTC),
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=SUMMARY_SCHEMA, orient="row")


def _ranked_summary_rows(
    summary: pl.DataFrame,
    *,
    as_of_date: date,
    max_candidates: int,
    registry_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if summary.is_empty():
        return []
    rows = [
        row
        for row in summary.to_dicts()
        if str(row.get("as_of_date") or "")[:10] == as_of_date.isoformat()
        and _candidate_enabled_for_registry(
            row.get("strategy_candidate"),
            registry_lookup,
        )
    ]
    rows.sort(
        key=lambda row: (
            _int(row.get("complete_sample_count")) or 0,
            _float(row.get("avg_net_bps")) or -1e9,
            _float(row.get("win_rate")) or 0.0,
            str(row.get("strategy_candidate") or ""),
            str(row.get("symbol") or ""),
        ),
        reverse=True,
    )
    output: list[dict[str, Any]] = []
    by_family: dict[str, int] = {}
    for row in rows:
        candidate = str(row.get("strategy_candidate") or "")
        template = _template_for_candidate(candidate, registry_lookup or {})
        template_config = (registry_lookup or {}).get(candidate, {})
        family_cap = _int(template_config.get("max_candidates_per_day")) or max_candidates
        if by_family.get(template, 0) >= family_cap:
            continue
        output.append(row)
        by_family[template] = by_family.get(template, 0) + 1
        if len(output) >= max(max_candidates, 1):
            break
    return output


def _candidate_id(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("as_of_date") or ""),
            str(row.get("strategy_candidate") or ""),
            normalize_symbol(row.get("symbol")) or "UNKNOWN",
            str(row.get("regime_state") or "UNKNOWN"),
            str(_int(row.get("horizon_hours")) or 0),
        ]
    )


def _sample_group_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("strategy_candidate") or ""),
        normalize_symbol(row.get("symbol")) or "UNKNOWN",
        str(row.get("regime_state") or "UNKNOWN"),
        _int(row.get("horizon_hours")) or 0,
    )


def _sample_groups(
    samples: pl.DataFrame,
    as_of_date: date,
) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    if samples.is_empty():
        return groups
    for row in samples.to_dicts():
        if str(row.get("as_of_date") or "")[:10] != as_of_date.isoformat():
            continue
        key = _sample_group_key(row)
        groups.setdefault(key, []).append(row)
    return groups


def _split_sample_metrics(
    sample_rows: list[dict[str, Any]],
    summary_row: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not sample_rows:
        empty = _empty_window_metrics()
        return {
            "all": _summary_window_metrics(summary_row),
            "train": _summary_window_metrics(summary_row),
            "validation": empty.copy(),
            "recent_7d": empty.copy(),
        }

    rows = [
        (timestamp, row)
        for row in sample_rows
        if (timestamp := _sample_timestamp(row)) is not None
    ]
    if not rows:
        empty = _empty_window_metrics()
        return {
            "all": _summary_window_metrics(summary_row),
            "train": _summary_window_metrics(summary_row),
            "validation": empty.copy(),
            "recent_7d": empty.copy(),
        }
    rows.sort(key=lambda item: item[0])
    start_ts = rows[0][0]
    end_ts = rows[-1][0]
    total_seconds = max((end_ts - start_ts).total_seconds(), 0.0)
    cut_ts = start_ts + timedelta(seconds=total_seconds * 0.7)
    recent_start = end_ts - timedelta(days=7)
    train_rows = [row for timestamp, row in rows if timestamp <= cut_ts]
    validation_rows = [row for timestamp, row in rows if timestamp > cut_ts]
    recent_rows = [row for timestamp, row in rows if timestamp >= recent_start]
    return {
        "all": _window_metrics([row for _, row in rows]),
        "train": _window_metrics(train_rows),
        "validation": _window_metrics(validation_rows),
        "recent_7d": _window_metrics(recent_rows),
    }


def _sample_timestamp(row: dict[str, Any]) -> datetime | None:
    for column in ("decision_ts", "ts_utc", "label_ts", "created_at"):
        parsed = _parse_dt(row.get(column))
        if parsed is not None:
            return parsed
    return None


def _summary_window_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": _int(row.get("sample_count")) or 0,
        "complete_sample_count": _int(row.get("complete_sample_count")) or 0,
        "avg_net_bps": _float(row.get("avg_net_bps")),
        "p25_net_bps": _float(row.get("p25_net_bps")),
        "win_rate": _float(row.get("win_rate")),
        "avg_mae_bps": _float(row.get("avg_mae_bps")),
        "avg_mfe_bps": _float(row.get("avg_mfe_bps")),
    }


def _window_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete_rows = [
        row
        for row in rows
        if str(row.get("label_status") or "").lower() in {"complete", "completed", ""}
        and _float(row.get("net_bps_after_cost")) is not None
    ]
    net_values = [
        value
        for row in complete_rows
        if (value := _float(row.get("net_bps_after_cost"))) is not None
    ]
    wins = [_sample_win(row, value) for row, value in zip(complete_rows, net_values, strict=False)]
    mae_values = [
        value for row in complete_rows if (value := _float(row.get("mae_bps"))) is not None
    ]
    mfe_values = [
        value for row in complete_rows if (value := _float(row.get("mfe_bps"))) is not None
    ]
    return {
        "sample_count": len(rows),
        "complete_sample_count": len(net_values),
        "avg_net_bps": _mean(net_values),
        "p25_net_bps": _percentile(net_values, 0.25),
        "win_rate": _mean([1.0 if win else 0.0 for win in wins]) if wins else None,
        "avg_mae_bps": _mean(mae_values),
        "avg_mfe_bps": _mean(mfe_values),
    }


def _sample_win(row: dict[str, Any], net_bps: float) -> bool:
    value = row.get("win")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "win"}:
        return True
    if text in {"false", "0", "no", "loss"}:
        return False
    return net_bps > 0.0


def _empty_window_metrics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "complete_sample_count": 0,
        "avg_net_bps": None,
        "p25_net_bps": None,
        "win_rate": None,
        "avg_mae_bps": None,
        "avg_mfe_bps": None,
    }


def _alpha_factory_scores(
    *,
    avg_net_bps: float | None,
    p25_net_bps: float | None,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    recent_7d_metrics: dict[str, Any],
) -> dict[str, Any]:
    validation_avg = _float(validation_metrics.get("avg_net_bps"))
    validation_complete = _int(validation_metrics.get("complete_sample_count")) or 0
    recent_avg = _float(recent_7d_metrics.get("avg_net_bps"))
    recent_complete = _int(recent_7d_metrics.get("complete_sample_count")) or 0
    train_avg = _float(train_metrics.get("avg_net_bps"))

    tail_loss_penalty = _clip(
        ((abs(p25_net_bps) - 50.0) / 150.0)
        if p25_net_bps is not None and p25_net_bps < -50.0
        else 0.0
    )
    validation_failure_penalty = (
        1.0
        if validation_complete <= 0 or validation_avg is None or validation_avg <= 0.0
        else 0.0
    )
    recent_degradation_penalty = 0.0
    if recent_complete >= 5 and recent_avg is not None:
        reference = validation_avg if validation_avg is not None else train_avg
        if recent_avg < 0.0:
            recent_degradation_penalty = 1.0
        elif reference is not None and recent_avg < reference:
            recent_degradation_penalty = _clip((reference - recent_avg) / 100.0)
    stability_score = _clip(
        1.0
        - tail_loss_penalty * 0.25
        - recent_degradation_penalty * 0.35
        - validation_failure_penalty * 0.40
    )
    alpha_factory_score = (avg_net_bps or 0.0) * stability_score - (
        tail_loss_penalty + recent_degradation_penalty + validation_failure_penalty
    ) * 25.0
    score_reasons: list[str] = []
    if tail_loss_penalty > 0.0:
        score_reasons.append("tail_loss_penalty_applied")
    if recent_degradation_penalty > 0.0:
        score_reasons.append("recent_degradation_penalty_applied")
    if validation_failure_penalty > 0.0:
        score_reasons.append("validation_failure_penalty_applied")
    return {
        "stability_score": stability_score,
        "tail_loss_penalty": tail_loss_penalty,
        "recent_degradation_penalty": recent_degradation_penalty,
        "validation_failure_penalty": validation_failure_penalty,
        "alpha_factory_score": alpha_factory_score,
        "score_reasons": score_reasons,
    }


def _paper_ready_block_reasons(
    *,
    complete_sample_count: int,
    p25_net_bps: float | None,
    win_rate: float | None,
    validation_metrics: dict[str, Any],
    recent_7d_metrics: dict[str, Any],
    avg_mae_bps: float | None,
    cost_quality_score: float | None,
    cost_source_mix: Any,
    is_futures_proxy: bool,
) -> list[str]:
    reasons: list[str] = []
    if complete_sample_count < 30:
        reasons.append("insufficient_complete_samples_for_paper")
    if win_rate is None or win_rate <= 0.55:
        reasons.append("win_rate_below_paper_threshold")
    if p25_net_bps is None or p25_net_bps <= -50.0:
        reasons.append("tail_loss_p25_below_threshold")

    validation_avg = _float(validation_metrics.get("avg_net_bps"))
    validation_complete = _int(validation_metrics.get("complete_sample_count")) or 0
    if validation_complete <= 0:
        reasons.append("insufficient_validation_samples")
    elif validation_avg is None:
        reasons.append("validation_avg_net_bps_missing")
    elif validation_avg <= 0.0:
        reasons.append("validation_avg_net_bps_non_positive")

    recent_avg = _float(recent_7d_metrics.get("avg_net_bps"))
    recent_complete = _int(recent_7d_metrics.get("complete_sample_count")) or 0
    if recent_complete < 5:
        reasons.append("insufficient_recent_samples")
    elif recent_avg is None:
        reasons.append("recent_7d_avg_net_bps_missing")
    elif recent_avg < 0.0:
        reasons.append("recent_7d_avg_net_bps_negative")

    if avg_mae_bps is not None and avg_mae_bps <= -150.0:
        reasons.append("mae_too_deep_for_paper")

    score = cost_quality_score
    if score is None and cost_source_mix is not None:
        score = _cost_quality_score(cost_source_mix)
    if score is not None and score < 0.5:
        reasons.append("cost_quality_not_paper_ready")
    if _cost_source_mix_is_local_or_degraded(cost_source_mix):
        reasons.append("cost_quality_not_paper_ready")

    if is_futures_proxy:
        reasons.append("futures_proxy_not_paper_ready")
    return _dedupe_text(reasons)


def _cost_quality_score(cost_source_mix: Any) -> float:
    counts = _cost_source_counts(cost_source_mix)
    if not counts:
        return 0.0
    weights = {
        "actual_fills": 1.0,
        "actual_okx_fills_and_bills": 1.0,
        "quant_lab_actual": 1.0,
        "mixed_actual_proxy": 0.9,
        "paper_slippage_observed": 0.8,
        "public_spread_proxy": 0.6,
        "local_estimate": 0.25,
        "degraded": 0.0,
        "global_default": 0.0,
        "missing": 0.0,
        "cost_not_requested_no_order": 0.0,
    }
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return sum(weights.get(source, 0.4) * count for source, count in counts.items()) / total


def _cost_source_mix_is_local_or_degraded(cost_source_mix: Any) -> bool:
    counts = _cost_source_counts(cost_source_mix)
    if not counts:
        return False
    weak_sources = {"local_estimate", "degraded"}
    weak_count = sum(count for source, count in counts.items() if source in weak_sources)
    return weak_count / max(sum(counts.values()), 1) >= 0.5


def _cost_source_counts(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        output: dict[str, int] = {}
        for source, count in value.items():
            name = str(source or "").strip().lower()
            if name:
                output[name] = output.get(name, 0) + max(_int(count) or 0, 1)
        return output
    if isinstance(value, list):
        output: dict[str, int] = {}
        for item in value:
            if isinstance(item, dict):
                source = str(
                    item.get("cost_source") or item.get("source") or item.get("name") or ""
                ).strip().lower()
                count = max(_int(item.get("count")) or 0, 1)
                if source:
                    output[source] = output.get(source, 0) + count
            elif isinstance(item, str) and item.strip():
                source = item.strip().lower()
                output[source] = output.get(source, 0) + 1
        return output
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {text.lower(): 1}
        return _cost_source_counts(parsed)
    text = str(value).strip().lower()
    return {text: 1} if text else {}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _template_for_candidate(
    candidate: Any,
    registry_lookup: dict[str, dict[str, Any]],
) -> str:
    template = _registry_payload_for_candidate(candidate, registry_lookup) or {}
    return str(template.get("template_family") or _fallback_template_family(candidate))


def _fallback_template_family(candidate: Any) -> str:
    text = str(candidate or "")
    if _is_factor_bridge_candidate(text):
        return FACTOR_BRIDGE_TEMPLATE_FAMILY
    if text.startswith("v5.expanded_relative_strength"):
        return "expanded_relative_strength"
    if text in {
        "v5.btc_strict_probe_exit_policy_review",
        "v5.eth_f3_exit_policy_review",
        "v5.sol_paper_exit_policy_review",
    }:
        return "exit_policy_review"
    if text.startswith("v5.futures_"):
        return "futures_hedge_shadow"
    if text.startswith("v5.pair_") or text == "v5.alt_basket_vs_btc_shadow":
        return "pair_market_neutral_shadow"
    if text == "v5.alt_impulse_shadow":
        return "alt_impulse_regime"
    return "alpha_factory"


def _is_futures_proxy_candidate(candidate: Any) -> bool:
    return str(candidate or "") in {
        "v5.futures_risk_off_hedge_proxy_shadow",
        "v5.futures_downtrend_short_proxy_shadow",
    }


def _candidate_enabled_for_registry(
    candidate: Any,
    registry_lookup: dict[str, dict[str, Any]] | None,
) -> bool:
    text = str(candidate or "")
    if not text:
        return False
    if not registry_lookup:
        return text in ALPHA_FACTORY_CANDIDATES or _is_factor_bridge_candidate(text)
    return _registry_payload_for_candidate(text, registry_lookup) is not None


def _registry_payload_for_candidate(
    candidate: Any,
    registry_lookup: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not registry_lookup:
        return None
    text = str(candidate or "")
    if text in registry_lookup:
        return registry_lookup[text]
    if _is_factor_bridge_candidate(text):
        for prefix, pattern in zip(
            FACTOR_BRIDGE_CANDIDATE_PREFIXES,
            FACTOR_BRIDGE_TEMPLATE_PATTERNS,
            strict=True,
        ):
            if text.startswith(prefix) and pattern in registry_lookup:
                return registry_lookup[pattern]
        return registry_lookup.get(FACTOR_BRIDGE_TEMPLATE_PATTERN)
    return None


def _is_factor_bridge_candidate(candidate: Any) -> bool:
    text = str(candidate or "")
    return text.startswith(FACTOR_BRIDGE_CANDIDATE_PREFIXES)


def _alpha_factory_managed_candidate_expr(candidate: pl.Expr) -> pl.Expr:
    candidate_text = candidate.cast(pl.Utf8, strict=False).fill_null("")
    bridge_expr = candidate_text.str.starts_with(FACTOR_BRIDGE_CANDIDATE_PREFIX)
    for prefix in FACTOR_BRIDGE_CANDIDATE_PREFIXES[1:]:
        bridge_expr = bridge_expr | candidate_text.str.starts_with(prefix)
    return candidate_text.is_in(sorted(ALPHA_FACTORY_CANDIDATES)) | bridge_expr


def _alpha_factory_source_summary(root: Path, day: date) -> pl.DataFrame:
    second_stage = read_parquet_dataset(root / SECOND_STAGE_SUMMARY_DATASET)
    second_stage = _with_source_dataset(
        second_stage,
        "gold/second_stage_alpha_factory_summary",
    )
    strategy_evidence = read_parquet_dataset(root / STRATEGY_EVIDENCE_DATASET)
    alt_impulse = pl.DataFrame()
    if not strategy_evidence.is_empty() and "strategy_candidate" in strategy_evidence.columns:
        alt_impulse = strategy_evidence.filter(
            (pl.col("strategy_candidate").cast(pl.Utf8) == "v5.alt_impulse_shadow")
            & (pl.col("as_of_date").cast(pl.Utf8).str.slice(0, 10) == day.isoformat())
        )
        alt_impulse = _with_source_dataset(alt_impulse, "gold/strategy_evidence")
    factor_bridge = _factor_bridge_source_summary(root, day)
    frames = [
        frame
        for frame in [second_stage, alt_impulse, factor_bridge]
        if not frame.is_empty()
    ]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _factor_bridge_source_summary(root: Path, day: date) -> pl.DataFrame:
    input_mtime = _factor_bridge_input_latest_mtime(root)
    from_latest_pack = _latest_factor_bridge_report_summary(
        root,
        day,
        min_pack_mtime=input_mtime,
    )
    if not from_latest_pack.is_empty():
        return from_latest_pack
    if not _factor_bridge_lake_recompute_allowed(root):
        return pl.DataFrame()

    factor_candidates = read_parquet_dataset(root / FACTOR_CANDIDATE_DATASET)
    factor_values = read_parquet_dataset(root / FACTOR_VALUE_DATASET)
    market_bars = read_parquet_dataset(root / MARKET_BAR_DATASET)
    if (
        not factor_candidates.is_empty()
        and not factor_values.is_empty()
        and not market_bars.is_empty()
    ):
        factor_forward = build_factor_forward_validation(
            factor_candidates=factor_candidates,
            factor_values=factor_values,
            market_bars=market_bars,
            market_regime=read_parquet_dataset(root / MARKET_REGIME_DATASET),
            cost_bucket_daily=read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET),
        )
        bridge = build_factor_strategy_bridge_candidates(
            paper_queue=pl.DataFrame(),
            factor_forward_validation=factor_forward,
        )
        current = _factor_bridge_rows_to_source_summary(
            bridge.to_dicts() if not bridge.is_empty() else [],
            day,
            source_dataset=FACTOR_BRIDGE_REPORT_MEMBER,
        )
        if not current.is_empty():
            return current

    if input_mtime is None:
        return _latest_factor_bridge_report_summary(root, day)
    return pl.DataFrame()


def _latest_factor_bridge_report_summary(
    root: Path,
    day: date,
    *,
    min_pack_mtime: datetime | None = None,
) -> pl.DataFrame:
    exports_root = _default_exports_root(root)
    if not exports_root.exists():
        return pl.DataFrame()
    packs = sorted(
        [
            path
            for path in exports_root.glob(f"quant_lab_expert_pack_{day.isoformat()}_*.zip")
            if path.is_file()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    packs.extend(
        sorted(
            [
                path
                for path in exports_root.glob("quant_lab_expert_pack_*.zip")
                if path.is_file() and path not in packs
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    for pack in packs[:5]:
        pack_mtime = _expert_pack_content_time(pack) or _path_mtime(pack)
        if (
            min_pack_mtime is not None
            and pack_mtime is not None
            and pack_mtime + timedelta(seconds=1) < min_pack_mtime
        ):
            continue
        try:
            with zipfile.ZipFile(pack) as archive:
                if FACTOR_BRIDGE_REPORT_MEMBER not in archive.namelist():
                    continue
                rows = list(
                    csv.DictReader(
                        io.StringIO(
                            archive.read(FACTOR_BRIDGE_REPORT_MEMBER).decode("utf-8")
                        )
                    )
                )
        except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError):
            continue
        summary = _factor_bridge_rows_to_source_summary(
            rows,
            day,
            source_dataset=FACTOR_BRIDGE_REPORT_MEMBER,
        )
        if not summary.is_empty():
            return summary
    return pl.DataFrame()


def _expert_pack_content_time(path: Path) -> datetime | None:
    try:
        with zipfile.ZipFile(path) as archive:
            if "manifest.json" not in archive.namelist():
                return None
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    for field in (
        "export_finished_at",
        "export_generated_at",
        "generated_at",
        "created_at",
    ):
        parsed = _parse_dt(manifest.get(field))
        if parsed is not None:
            return parsed
    return None


def _factor_bridge_lake_recompute_allowed(root: Path) -> bool:
    factor_limit = _int_env(
        "QUANT_LAB_ALPHA_FACTORY_FACTOR_BRIDGE_RECOMPUTE_MAX_FACTOR_VALUE_ROWS",
        FACTOR_BRIDGE_RECOMPUTE_MAX_FACTOR_VALUE_ROWS,
    )
    market_limit = _int_env(
        "QUANT_LAB_ALPHA_FACTORY_FACTOR_BRIDGE_RECOMPUTE_MAX_MARKET_BAR_ROWS",
        FACTOR_BRIDGE_RECOMPUTE_MAX_MARKET_BAR_ROWS,
    )
    if factor_limit <= 0 or market_limit <= 0:
        return False
    try:
        return (
            count_parquet_rows(root / FACTOR_VALUE_DATASET) <= factor_limit
            and count_parquet_rows(root / MARKET_BAR_DATASET) <= market_limit
        )
    except Exception:
        return False


def _factor_bridge_input_latest_mtime(root: Path) -> datetime | None:
    return _max_dt(
        *[
            _dataset_latest_mtime(root / dataset)
            for dataset in (
                FACTOR_CANDIDATE_DATASET,
                FACTOR_VALUE_DATASET,
                MARKET_BAR_DATASET,
                MARKET_REGIME_DATASET,
                COST_BUCKET_DAILY_DATASET,
            )
        ]
    )


def _dataset_latest_mtime(path: Path) -> datetime | None:
    mtimes = [_path_mtime(item) for item in path.rglob("*.parquet")] if path.exists() else []
    present = [item for item in mtimes if item is not None]
    return max(present) if present else None


def _path_mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None


def _max_dt(*values: datetime | None) -> datetime | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_exports_root(root: Path) -> Path:
    return root.parent / "exports" if root.name == "lake" else root / "exports"


def _factor_bridge_rows_to_source_summary(
    bridge_rows: list[dict[str, Any]],
    day: date,
    *,
    source_dataset: str,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in bridge_rows:
        if str(row.get("recommended_action") or "") != "REVIEW_FOR_ALPHA_FACTORY_STRATEGY":
            continue
        if str(row.get("eligible_for_alpha_factory") or "") != "strategy_review_pending":
            continue
        sample_count = _int(row.get("forward_sample_count")) or 0
        if sample_count <= 0:
            continue
        candidate = str(row.get("bridge_candidate_id") or "")
        if not _is_factor_bridge_candidate(candidate):
            continue
        symbol = normalize_symbol(_first_csv_value(row.get("symbol"))) or "UNKNOWN"
        source_label = (
            "fast_microstructure_forward_validation_read_only"
            if candidate.startswith(FAST_MICROSTRUCTURE_BRIDGE_CANDIDATE_PREFIX)
            else "factor_forward_validation_read_only"
        )
        pass_reason = (
            "fast_microstructure_forward_validation_pass"
            if candidate.startswith(FAST_MICROSTRUCTURE_BRIDGE_CANDIDATE_PREFIX)
            else "factor_forward_validation_pass"
        )
        rows.append(
            {
                "as_of_date": day.isoformat(),
                "source_as_of_date": str(row.get("as_of_date") or ""),
                "strategy_candidate": candidate,
                "candidate_name": candidate,
                "symbol": symbol,
                "regime_state": _first_csv_value(row.get("regime")) or "UNKNOWN",
                "horizon_hours": _first_int_csv_value(row.get("horizon_hours")) or 0,
                "sample_count": sample_count,
                "complete_sample_count": sample_count,
                "avg_net_bps": None,
                "median_net_bps": None,
                "p25_net_bps": None,
                "win_rate": None,
                "cost_source_mix": safe_json_dumps({source_label: sample_count}),
                "decision": "RESEARCH_ONLY",
                "decision_reasons": safe_json_dumps(
                    [
                        pass_reason,
                        "alpha_factory_strategy_review_required",
                        "strategy_review_only_no_live_order",
                    ]
                ),
                "start_ts": None,
                "end_ts": None,
                "factor_id": row.get("factor_id"),
                "factor_family": row.get("factor_family"),
                "bridge_candidate_id": candidate,
                "forward_regime": row.get("regime"),
                "forward_horizon_hours": row.get("horizon_hours"),
                "forward_sample_count": sample_count,
                "forward_cost_adjusted_score": _float(row.get("forward_cost_adjusted_score")),
                "blocking_reasons": row.get("blocking_reasons"),
                "created_at": datetime.now(UTC),
                "source": SOURCE_NAME,
                "source_dataset": source_dataset,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def _alpha_factory_source_samples(root: Path, day: date) -> pl.DataFrame:
    second_stage = read_parquet_dataset(root / SECOND_STAGE_SAMPLE_DATASET)
    second_stage = _filter_samples_for_day(
        second_stage,
        day,
        source_dataset="gold/second_stage_alpha_factory_sample",
    )
    strategy_samples = read_parquet_dataset(root / STRATEGY_EVIDENCE_SAMPLE_DATASET)
    alt_impulse = pl.DataFrame()
    if not strategy_samples.is_empty() and "strategy_candidate" in strategy_samples.columns:
        alt_impulse = strategy_samples.filter(
            (pl.col("strategy_candidate").cast(pl.Utf8) == "v5.alt_impulse_shadow")
            & (pl.col("as_of_date").cast(pl.Utf8).str.slice(0, 10) == day.isoformat())
        )
        alt_impulse = _with_source_dataset(
            alt_impulse,
            "gold/strategy_evidence_sample",
        )
    frames = [frame for frame in [second_stage, alt_impulse] if not frame.is_empty()]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _filter_samples_for_day(
    frame: pl.DataFrame,
    day: date,
    *,
    source_dataset: str,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "as_of_date" not in frame.columns:
        return _with_source_dataset(frame, source_dataset)
    filtered = frame.filter(pl.col("as_of_date").cast(pl.Utf8).str.slice(0, 10) == day.isoformat())
    return _with_source_dataset(filtered, source_dataset)


def _with_source_dataset(frame: pl.DataFrame, source_dataset: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "source_dataset" in frame.columns:
        return frame.with_columns(pl.lit(source_dataset).alias("source_dataset"))
    return frame.with_columns(pl.lit(source_dataset).alias("source_dataset"))


def _parameter_payload(
    row: dict[str, Any],
    template: str,
    registry_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate = str(row.get("strategy_candidate") or "")
    config = _registry_payload_for_candidate(candidate, registry_lookup) or {}
    payload = {
        "template": template,
        "template_id": str(config.get("template_id") or ""),
        "parameter_space": config.get("parameter_space") or {},
        "horizon_hours": _int(row.get("horizon_hours")) or 0,
        "regime_state": str(row.get("regime_state") or "UNKNOWN"),
        "dry_run_first": True,
        "batch_scan": True,
        "paper_only_by_default": True,
    }
    if _is_factor_bridge_candidate(candidate):
        payload.update(
            {
                "factor_id": str(row.get("factor_id") or ""),
                "factor_family": str(row.get("factor_family") or ""),
                "bridge_candidate_id": str(row.get("bridge_candidate_id") or candidate),
                "source_as_of_date": str(row.get("source_as_of_date") or ""),
                "forward_regime": str(row.get("forward_regime") or ""),
                "forward_horizon_hours": str(row.get("forward_horizon_hours") or ""),
                "forward_sample_count": _int(row.get("forward_sample_count")) or 0,
                "forward_cost_adjusted_score": _float(
                    row.get("forward_cost_adjusted_score")
                ),
                "blocking_reasons": _json_list(row.get("blocking_reasons")),
                "strategy_review_only": True,
            }
        )
    return payload


def _recommended_mode(decision: str) -> str:
    if decision == "KILL":
        return "none"
    if decision == "PAPER_READY":
        return "paper"
    if decision == "KEEP_SHADOW":
        return "shadow"
    return "research"


def _promotion_action(decision: str) -> str:
    if decision == "KILL":
        return "CLOSE_RESEARCH"
    if decision == "PAPER_READY":
        return "QUEUE_FOR_PAPER_REVIEW"
    if decision == "KEEP_SHADOW":
        return "CONTINUE_SHADOW"
    return "COLLECT_MORE_SAMPLES"


def _strategy_evidence_decision(factory_decision: str) -> str:
    if factory_decision == "RESEARCH":
        return "RESEARCH_ONLY"
    return factory_decision


def _filter_template(frame: pl.DataFrame, template: str) -> pl.DataFrame:
    if frame.is_empty() or "template_name" not in frame.columns:
        return pl.DataFrame()
    return frame.filter(pl.col("template_name") == template)


def _latest_as_of_date(frame: pl.DataFrame) -> str | None:
    if frame.is_empty() or "as_of_date" not in frame.columns:
        return None
    values = [
        str(value)[:10]
        for value in frame.get_column("as_of_date").drop_nulls().to_list()
        if str(value).strip()
    ]
    return max(values) if values else None


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value or "auto").strip().lower()
    if not text or text == "auto":
        return datetime.now(UTC).date()
    return date.fromisoformat(text[:10])


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _int(value: Any) -> int | None:
    parsed = _float(value)
    return int(parsed) if parsed is not None else None


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if not isinstance(value, str):
        return [str(value)]
    try:
        import json

        parsed = json.loads(value)
    except Exception:
        return [value] if value else []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [str(parsed)] if parsed is not None else []


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        import json

        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_csv_value(value: Any) -> str:
    for item in str(value or "").split(","):
        text = item.strip()
        if text:
            return text
    return ""


def _first_int_csv_value(value: Any) -> int | None:
    text = _first_csv_value(value)
    return _int(text) if text else None


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _decision_counts(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty() or "decision" not in frame.columns:
        return {}
    return {
        str(row["decision"]): int(row["count"])
        for row in frame.group_by("decision").len(name="count").to_dicts()
    }


def _merge_template_registry(existing: pl.DataFrame, defaults: pl.DataFrame) -> pl.DataFrame:
    if existing.is_empty():
        return defaults
    existing = _normalize_template_registry(existing)
    if existing.is_empty():
        return defaults
    configured_ids = set(existing.get_column("template_id").drop_nulls().to_list())
    missing_defaults = defaults.filter(~pl.col("template_id").is_in(configured_ids))
    merged = pl.concat([existing, missing_defaults], how="diagonal_relaxed")
    return _normalize_template_registry(merged)


def _normalize_template_registry(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=TEMPLATE_REGISTRY_SCHEMA)
    normalized = frame
    for column, dtype in TEMPLATE_REGISTRY_SCHEMA.items():
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=dtype).alias(column))
    normalized = normalized.with_columns(
        pl.when(pl.col("safety_mode").is_null() | (pl.col("safety_mode").cast(pl.Utf8) == ""))
        .then(pl.lit("paper_shadow_only"))
        .otherwise(pl.col("safety_mode").cast(pl.Utf8))
        .alias("safety_mode"),
        pl.when(pl.col("enabled").is_null())
        .then(pl.lit(True))
        .otherwise(pl.col("enabled").cast(pl.Boolean))
        .alias("enabled"),
    )
    return normalized.select(list(TEMPLATE_REGISTRY_SCHEMA))


def _candidate_patterns_for_family(template_family: str) -> list[str]:
    for template in DEFAULT_TEMPLATE_REGISTRY:
        if template["template_family"] == template_family:
            return [str(value) for value in template["candidate_patterns"]]
    return []


def _template_families_from_results(results: pl.DataFrame, registry: pl.DataFrame) -> list[str]:
    families = []
    if not registry.is_empty():
        families.extend(
            str(row["template_family"])
            for row in registry.to_dicts()
            if bool(row.get("enabled")) and str(row.get("template_family") or "")
        )
    if not results.is_empty() and "template_name" in results.columns:
        families.extend(str(value) for value in results["template_name"].drop_nulls().to_list())
    return _dedupe_text(families)
