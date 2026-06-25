from __future__ import annotations

import math
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset, write_parquet_dataset
from quant_lab.factors.operators import numeric, safe_divide, winsorize_expr
from quant_lab.factors.registry import FactorSpec, discover_factor_specs
from quant_lab.research.evidence import DEFAULT_RESEARCH_COST_BPS
from quant_lab.research.ic import compute_ic, compute_rank_ic
from quant_lab.research.labels import build_forward_return_labels, validate_no_label_lookahead
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps

SOURCE_NAME = "factors.factory.v0.1"
CODE_VERSION_PREFIX = "factors.factory"

FEATURE_VALUE_DATASET = Path("gold") / "feature_value"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"

FACTOR_DEFINITION_DATASET = Path("gold") / "factor_definition"
FACTOR_VALUE_DATASET = Path("gold") / "factor_value"
FACTOR_EVIDENCE_DATASET = Path("gold") / "factor_evidence"
FACTOR_CANDIDATE_DATASET = Path("gold") / "factor_candidate"
FACTOR_CORRELATION_DAILY_DATASET = Path("gold") / "factor_correlation_daily"

FACTOR_DEFINITION_SCHEMA: dict[str, Any] = {
    "factor_id": pl.Utf8,
    "factor_name": pl.Utf8,
    "factor_family": pl.Utf8,
    "factor_version": pl.Utf8,
    "description": pl.Utf8,
    "feature_set": pl.Utf8,
    "feature_version": pl.Utf8,
    "timeframe": pl.Utf8,
    "input_features_json": pl.Utf8,
    "template": pl.Utf8,
    "params_json": pl.Utf8,
    "expression_json": pl.Utf8,
    "expression_hash": pl.Utf8,
    "status": pl.Utf8,
    "lookback_bars": pl.Int64,
    "availability_lag_bars": pl.Int64,
    "warmup_bars": pl.Int64,
    "required_bars": pl.Int64,
    "causal": pl.Boolean,
    "normalization": pl.Utf8,
    "owner": pl.Utf8,
    "direction": pl.Int64,
    "min_cross_section": pl.Int64,
    "clip_abs": pl.Float64,
    "enabled": pl.Boolean,
    "tags_json": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

FACTOR_VALUE_SCHEMA: dict[str, Any] = {
    "factor_id": pl.Utf8,
    "factor_name": pl.Utf8,
    "factor_family": pl.Utf8,
    "factor_version": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "ts": pl.Datetime(time_zone="UTC"),
    "event_time": pl.Datetime(time_zone="UTC"),
    "available_time": pl.Datetime(time_zone="UTC"),
    "raw_value": pl.Float64,
    "normalized_value": pl.Float64,
    "rank_value": pl.Float64,
    "value": pl.Float64,
    "factor_status": pl.Utf8,
    "expression_hash": pl.Utf8,
    "input_features_json": pl.Utf8,
    "input_dataset_version": pl.Utf8,
    "data_version": pl.Utf8,
    "input_hash": pl.Utf8,
    "code_version": pl.Utf8,
    "calculated_at": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
    "is_valid": pl.Boolean,
    "invalid_reason": pl.Utf8,
    "quality_flags_json": pl.Utf8,
}

FACTOR_EVIDENCE_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "factor_id": pl.Utf8,
    "factor_name": pl.Utf8,
    "factor_family": pl.Utf8,
    "factor_version": pl.Utf8,
    "timeframe": pl.Utf8,
    "horizon_bars": pl.Int64,
    "decision_delay_bars": pl.Int64,
    "sample_count": pl.Int64,
    "valid_sample_count": pl.Int64,
    "coverage": pl.Float64,
    "ic_mean": pl.Float64,
    "ic_tstat": pl.Float64,
    "rank_ic_mean": pl.Float64,
    "rank_ic_tstat": pl.Float64,
    "ic_period_count": pl.Int64,
    "top_quantile": pl.Float64,
    "long_only_mean_bps": pl.Float64,
    "long_short_mean_bps": pl.Float64,
    "top_mean_bps": pl.Float64,
    "bottom_mean_bps": pl.Float64,
    "win_rate": pl.Float64,
    "hit_rate": pl.Float64,
    "turnover": pl.Float64,
    "max_drawdown": pl.Float64,
    "edge_cost_ratio": pl.Float64,
    "cost_ratio": pl.Float64,
    "period_count": pl.Int64,
    "decision": pl.Utf8,
    "score": pl.Float64,
    "reasons_json": pl.Utf8,
    "warnings_json": pl.Utf8,
    "start_ts": pl.Datetime(time_zone="UTC"),
    "end_ts": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

FACTOR_CANDIDATE_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "factor_id": pl.Utf8,
    "factor_name": pl.Utf8,
    "factor_family": pl.Utf8,
    "factor_version": pl.Utf8,
    "timeframe": pl.Utf8,
    "best_horizon_bars": pl.Int64,
    "tested_horizon_count": pl.Int64,
    "best_score": pl.Float64,
    "avg_score": pl.Float64,
    "best_rank_ic_mean": pl.Float64,
    "best_rank_ic_tstat": pl.Float64,
    "best_long_short_mean_bps": pl.Float64,
    "candidate_state": pl.Utf8,
    "recommended_action": pl.Utf8,
    "promotion_block_reasons_json": pl.Utf8,
    "manual_review_required": pl.Boolean,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

FACTOR_CORRELATION_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "factor_id_left": pl.Utf8,
    "factor_id_right": pl.Utf8,
    "factor_version": pl.Utf8,
    "timeframe": pl.Utf8,
    "sample_count": pl.Int64,
    "correlation": pl.Float64,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


class FactorPublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    feature_set: str
    feature_version: str
    factor_version: str
    timeframe: str
    factor_count: int = Field(ge=0)
    definition_rows: int = Field(ge=0)
    value_rows: int = Field(ge=0)
    feature_rows: int = Field(ge=0)
    published_value_rows: int = Field(ge=0)
    factor_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FactorEvidenceBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    evidence_rows: int = Field(ge=0)
    candidate_rows: int = Field(ge=0)
    correlation_rows: int = Field(ge=0)
    decision_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class FactorFactoryBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    factor_count: int = Field(ge=0)
    definition_rows: int = Field(ge=0)
    value_rows: int = Field(ge=0)
    evidence_rows: int = Field(ge=0)
    candidate_rows: int = Field(ge=0)
    correlation_rows: int = Field(ge=0)
    decision_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    diagnostic_only: bool = True
    live_order_effect: str = "none_read_only_research"


class FactorHealthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    definition_rows: int = Field(ge=0)
    value_rows: int = Field(ge=0)
    evidence_rows: int = Field(ge=0)
    candidate_rows: int = Field(ge=0)
    correlation_rows: int = Field(ge=0)
    latest_value_ts: str | None = None
    latest_evidence_created_at: str | None = None
    latest_candidate_created_at: str | None = None
    decision_counts: dict[str, int] = Field(default_factory=dict)
    paper_ready_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    diagnostic_only: bool = True
    live_order_effect: str = "none_read_only_research"


def build_and_publish_factor_factory(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = "auto",
    feature_set: str = "core",
    feature_version: str = "v0.1",
    factor_version: str = "v0.1",
    timeframe: str = "1H",
    horizon_bars: tuple[int, ...] = (4, 8, 24, 72),
    decision_delay_bars: int = 1,
    max_factors: int = 200,
    min_samples: int = 100,
    top_quantile: float = 0.2,
    cost_quantile: str = "p75",
    dry_run: bool = False,
) -> FactorFactoryBuildResult:
    if dry_run:
        return _build_factor_factory_dry_run(
            lake_root,
            as_of_date=as_of_date,
            feature_set=feature_set,
            feature_version=feature_version,
            factor_version=factor_version,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            decision_delay_bars=decision_delay_bars,
            max_factors=max_factors,
            min_samples=min_samples,
            top_quantile=top_quantile,
            cost_quantile=cost_quantile,
        )
    published = publish_factor_values(
        lake_root,
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        max_factors=max_factors,
        dry_run=dry_run,
    )
    evidence = evaluate_and_publish_factor_evidence(
        lake_root,
        as_of_date=as_of_date,
        factor_version=factor_version,
        timeframe=timeframe,
        horizon_bars=horizon_bars,
        decision_delay_bars=decision_delay_bars,
        min_samples=min_samples,
        top_quantile=top_quantile,
        cost_quantile=cost_quantile,
        dry_run=dry_run,
    )
    return FactorFactoryBuildResult(
        lake_root=str(Path(lake_root)),
        as_of_date=evidence.as_of_date,
        factor_count=published.factor_count,
        definition_rows=published.definition_rows,
        value_rows=published.value_rows,
        evidence_rows=evidence.evidence_rows,
        candidate_rows=evidence.candidate_rows,
        correlation_rows=evidence.correlation_rows,
        decision_counts=evidence.decision_counts,
        warnings=_dedupe([*published.warnings, *evidence.warnings]),
    )


def _build_factor_factory_dry_run(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None,
    feature_set: str,
    feature_version: str,
    factor_version: str,
    timeframe: str,
    horizon_bars: tuple[int, ...],
    decision_delay_bars: int,
    max_factors: int,
    min_samples: int,
    top_quantile: float,
    cost_quantile: str,
) -> FactorFactoryBuildResult:
    source_root = Path(lake_root)
    with tempfile.TemporaryDirectory(prefix="quant_lab_factor_factory_") as tmp:
        temp_root = Path(tmp) / "lake"
        for relative in [FEATURE_VALUE_DATASET, MARKET_BAR_DATASET, COST_BUCKET_DAILY_DATASET]:
            frame = read_parquet_dataset(source_root / relative)
            if not frame.is_empty():
                write_parquet_dataset(frame, temp_root / relative)
        result = build_and_publish_factor_factory(
            temp_root,
            as_of_date=as_of_date,
            feature_set=feature_set,
            feature_version=feature_version,
            factor_version=factor_version,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            decision_delay_bars=decision_delay_bars,
            max_factors=max_factors,
            min_samples=min_samples,
            top_quantile=top_quantile,
            cost_quantile=cost_quantile,
            dry_run=False,
        )
    return result.model_copy(
        update={
            "lake_root": str(source_root),
            "warnings": _dedupe([*result.warnings, "dry_run_no_production_write"]),
        }
    )


def publish_factor_definitions(
    lake_root: str | Path,
    specs: list[FactorSpec],
    *,
    created_at: datetime | None = None,
    dry_run: bool = False,
) -> int:
    root = Path(lake_root)
    now = created_at or datetime.now(UTC)
    frame = _schema_frame(
        [spec.definition_row(created_at=now, source=SOURCE_NAME) for spec in specs],
        FACTOR_DEFINITION_SCHEMA,
    )
    if dry_run:
        return read_parquet_dataset(root / FACTOR_DEFINITION_DATASET).height
    return upsert_parquet_dataset(
        frame,
        root / FACTOR_DEFINITION_DATASET,
        key_columns=["factor_id", "factor_version"],
    )


def publish_factor_values(
    lake_root: str | Path,
    *,
    feature_set: str = "core",
    feature_version: str = "v0.1",
    factor_version: str = "v0.1",
    timeframe: str = "1H",
    max_factors: int = 200,
    dry_run: bool = False,
) -> FactorPublishResult:
    root = Path(lake_root)
    warnings: list[str] = []
    features = _load_feature_values(
        root,
        feature_set=feature_set,
        feature_version=feature_version,
        timeframe=timeframe,
        warnings=warnings,
    )
    if features.is_empty():
        existing_definitions = read_parquet_dataset(root / FACTOR_DEFINITION_DATASET).height
        existing_values = read_parquet_dataset(root / FACTOR_VALUE_DATASET).height
        warnings.append("feature_value missing or empty for factor factory")
        return FactorPublishResult(
            lake_root=str(root),
            feature_set=feature_set,
            feature_version=feature_version,
            factor_version=factor_version,
            timeframe=timeframe,
            factor_count=0,
            definition_rows=existing_definitions,
            value_rows=existing_values,
            feature_rows=0,
            published_value_rows=0,
            warnings=_dedupe(warnings),
        )

    available_features = sorted(features["feature_name"].drop_nulls().unique().to_list())
    specs = discover_factor_specs(
        available_features,
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        max_factors=max_factors,
    )
    now = datetime.now(UTC)
    definitions_rows = publish_factor_definitions(root, specs, created_at=now, dry_run=dry_run)
    values = _build_factor_value_frame(features, specs, created_at=now)
    if values.is_empty():
        warnings.append("no factor values computed")
        existing_values = read_parquet_dataset(root / FACTOR_VALUE_DATASET).height
        return FactorPublishResult(
            lake_root=str(root),
            feature_set=feature_set,
            feature_version=feature_version,
            factor_version=factor_version,
            timeframe=timeframe,
            factor_count=len(specs),
            definition_rows=definitions_rows,
            value_rows=existing_values,
            feature_rows=features.height,
            published_value_rows=0,
            factor_ids=[spec.factor_id for spec in specs],
            warnings=_dedupe(warnings),
        )

    if dry_run:
        value_rows = read_parquet_dataset(root / FACTOR_VALUE_DATASET).height
    else:
        value_rows = upsert_parquet_dataset(
            values,
            root / FACTOR_VALUE_DATASET,
            key_columns=["factor_id", "factor_version", "symbol", "timeframe", "ts"],
        )

    return FactorPublishResult(
        lake_root=str(root),
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        factor_count=len(specs),
        definition_rows=definitions_rows,
        value_rows=value_rows,
        feature_rows=features.height,
        published_value_rows=0 if dry_run else values.height,
        factor_ids=[spec.factor_id for spec in specs],
        warnings=_dedupe(warnings),
    )


def evaluate_and_publish_factor_evidence(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = "auto",
    factor_version: str = "v0.1",
    timeframe: str = "1H",
    horizon_bars: tuple[int, ...] = (4, 8, 24, 72),
    decision_delay_bars: int = 1,
    min_samples: int = 100,
    top_quantile: float = 0.2,
    cost_quantile: str = "p75",
    dry_run: bool = False,
) -> FactorEvidenceBuildResult:
    if decision_delay_bars < 1:
        raise ValueError("decision_delay_bars must be at least 1")
    if not horizon_bars:
        raise ValueError("horizon_bars must not be empty")
    root = Path(lake_root)
    day = _parse_as_of_date(as_of_date)
    warnings: list[str] = []
    values = _load_factor_values(
        root,
        factor_version=factor_version,
        timeframe=timeframe,
        warnings=warnings,
    )
    market_bars = _load_market_bars(root, timeframe=timeframe, warnings=warnings)
    if values.is_empty() or market_bars.is_empty():
        warnings.append("factor_value or market_bar missing for factor evidence")
        return FactorEvidenceBuildResult(
            lake_root=str(root),
            as_of_date=day.isoformat(),
            evidence_rows=read_parquet_dataset(root / FACTOR_EVIDENCE_DATASET).height,
            candidate_rows=read_parquet_dataset(root / FACTOR_CANDIDATE_DATASET).height,
            correlation_rows=read_parquet_dataset(root / FACTOR_CORRELATION_DAILY_DATASET).height,
            warnings=_dedupe(warnings),
        )

    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    factor_keys = [
        "factor_id",
        "factor_name",
        "factor_family",
        "factor_version",
        "timeframe",
    ]
    for horizon in sorted({int(item) for item in horizon_bars if int(item) > 0}):
        labels = build_forward_return_labels(
            market_bars,
            horizon_bars=horizon,
            decision_delay_bars=decision_delay_bars,
        )
        validate_no_label_lookahead(labels)
        if labels.is_empty():
            warnings.append(f"horizon_{horizon}_labels_empty")
            continue
        evidence_dataset = values.rename({"ts": "feature_ts"}).join(
            labels,
            on=["symbol", "timeframe", "feature_ts"],
            how="inner",
        )
        evidence_dataset = _attach_symbol_costs(
            evidence_dataset,
            root,
            cost_quantile=cost_quantile,
            warnings=warnings,
        )
        for factor_key, group in evidence_dataset.group_by(factor_keys, maintain_order=True):
            factor_meta = dict(
                zip(factor_keys, _as_tuple(factor_key, len(factor_keys)), strict=True)
            )
            rows.append(
                _factor_evidence_row(
                    group,
                    factor_meta=factor_meta,
                    as_of_date=day,
                    horizon_bars=horizon,
                    decision_delay_bars=decision_delay_bars,
                    min_samples=min_samples,
                    top_quantile=top_quantile,
                    created_at=now,
                )
            )

    evidence = _schema_frame(rows, FACTOR_EVIDENCE_SCHEMA)
    candidates = _candidate_frame_from_evidence(evidence, as_of_date=day, created_at=now)
    correlations = _factor_correlation_frame(
        values,
        as_of_date=day,
        factor_version=factor_version,
        timeframe=timeframe,
        created_at=now,
    )
    if dry_run:
        evidence_rows = read_parquet_dataset(root / FACTOR_EVIDENCE_DATASET).height
        candidate_rows = read_parquet_dataset(root / FACTOR_CANDIDATE_DATASET).height
        correlation_rows = read_parquet_dataset(root / FACTOR_CORRELATION_DAILY_DATASET).height
    else:
        evidence_rows = upsert_parquet_dataset(
            evidence,
            root / FACTOR_EVIDENCE_DATASET,
            key_columns=[
                "as_of_date",
                "factor_id",
                "factor_version",
                "timeframe",
                "horizon_bars",
                "decision_delay_bars",
            ],
        )
        candidate_rows = upsert_parquet_dataset(
            candidates,
            root / FACTOR_CANDIDATE_DATASET,
            key_columns=["as_of_date", "factor_id", "factor_version", "timeframe"],
        )
        correlation_rows = upsert_parquet_dataset(
            correlations,
            root / FACTOR_CORRELATION_DAILY_DATASET,
            key_columns=[
                "as_of_date",
                "factor_id_left",
                "factor_id_right",
                "factor_version",
                "timeframe",
            ],
        )

    return FactorEvidenceBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        evidence_rows=evidence_rows,
        candidate_rows=candidate_rows,
        correlation_rows=correlation_rows,
        decision_counts=_decision_counts(evidence),
        warnings=_dedupe(warnings),
    )


def build_and_publish_factor_candidates(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = "auto",
    dry_run: bool = False,
) -> FactorEvidenceBuildResult:
    root = Path(lake_root)
    day = _parse_as_of_date(as_of_date)
    evidence = read_parquet_dataset(root / FACTOR_EVIDENCE_DATASET)
    if not evidence.is_empty() and "as_of_date" in evidence.columns:
        evidence = evidence.filter(pl.col("as_of_date") == day.isoformat())
    now = datetime.now(UTC)
    candidates = _candidate_frame_from_evidence(evidence, as_of_date=day, created_at=now)
    if dry_run:
        candidate_rows = read_parquet_dataset(root / FACTOR_CANDIDATE_DATASET).height
    else:
        candidate_rows = upsert_parquet_dataset(
            candidates,
            root / FACTOR_CANDIDATE_DATASET,
            key_columns=["as_of_date", "factor_id", "factor_version", "timeframe"],
        )
    return FactorEvidenceBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        evidence_rows=evidence.height,
        candidate_rows=candidate_rows,
        correlation_rows=read_parquet_dataset(root / FACTOR_CORRELATION_DAILY_DATASET).height,
        decision_counts=_decision_counts(evidence),
        warnings=[] if not evidence.is_empty() else ["factor_evidence_missing_or_empty"],
    )


def factor_factory_health(lake_root: str | Path) -> FactorHealthResult:
    root = Path(lake_root)
    definitions = read_parquet_dataset(root / FACTOR_DEFINITION_DATASET)
    values = read_parquet_dataset(root / FACTOR_VALUE_DATASET)
    evidence = read_parquet_dataset(root / FACTOR_EVIDENCE_DATASET)
    candidates = read_parquet_dataset(root / FACTOR_CANDIDATE_DATASET)
    correlations = read_parquet_dataset(root / FACTOR_CORRELATION_DAILY_DATASET)
    warnings: list[str] = []
    if definitions.is_empty():
        warnings.append("factor_definition_missing_or_empty")
    if values.is_empty():
        warnings.append("factor_value_missing_or_empty")
    if evidence.is_empty():
        warnings.append("factor_evidence_missing_or_empty")
    if candidates.is_empty():
        warnings.append("factor_candidate_missing_or_empty")
    return FactorHealthResult(
        lake_root=str(root),
        definition_rows=definitions.height,
        value_rows=values.height,
        evidence_rows=evidence.height,
        candidate_rows=candidates.height,
        correlation_rows=correlations.height,
        latest_value_ts=_frame_datetime_iso(values, "ts", "max"),
        latest_evidence_created_at=_frame_datetime_iso(evidence, "created_at", "max"),
        latest_candidate_created_at=_frame_datetime_iso(candidates, "created_at", "max"),
        decision_counts=_decision_counts(evidence),
        paper_ready_count=_count_value(candidates, "candidate_state", "PAPER_READY"),
        warnings=warnings,
    )


def _load_feature_values(
    root: Path,
    *,
    feature_set: str,
    feature_version: str,
    timeframe: str,
    warnings: list[str],
) -> pl.DataFrame:
    df = read_parquet_dataset(root / FEATURE_VALUE_DATASET)
    if df.is_empty():
        return df
    required = {"feature_set", "feature_name", "feature_version", "timeframe", "symbol", "ts"}
    missing = sorted(required.difference(df.columns))
    if missing:
        warnings.append(f"feature_value missing columns: {','.join(missing)}")
        return pl.DataFrame()
    filtered = _normalize_datetime(df, "ts").filter(
        (pl.col("feature_set") == feature_set)
        & (pl.col("feature_version") == feature_version)
        & (pl.col("timeframe") == timeframe)
    )
    if "is_valid" in filtered.columns:
        filtered = filtered.filter(pl.col("is_valid"))
    return filtered


def _load_factor_values(
    root: Path,
    *,
    factor_version: str,
    timeframe: str,
    warnings: list[str],
) -> pl.DataFrame:
    df = read_parquet_dataset(root / FACTOR_VALUE_DATASET)
    if df.is_empty():
        return df
    required = {"factor_id", "factor_version", "symbol", "timeframe", "ts", "value"}
    missing = sorted(required.difference(df.columns))
    if missing:
        warnings.append(f"factor_value missing columns: {','.join(missing)}")
        return pl.DataFrame()
    return _normalize_datetime(df, "ts").filter(
        (pl.col("factor_version") == factor_version) & (pl.col("timeframe") == timeframe)
    )


def _load_market_bars(root: Path, *, timeframe: str, warnings: list[str]) -> pl.DataFrame:
    df = read_parquet_dataset(root / MARKET_BAR_DATASET)
    if df.is_empty():
        return df
    required = {"symbol", "timeframe", "ts", "close"}
    missing = sorted(required.difference(df.columns))
    if missing:
        warnings.append(f"market_bar missing columns: {','.join(missing)}")
        return pl.DataFrame()
    normalized = _normalize_datetime(df, "ts")
    if "is_closed" in normalized.columns:
        normalized = normalized.filter(pl.col("is_closed"))
    return normalized.filter(pl.col("timeframe") == timeframe).sort(["symbol", "timeframe", "ts"])


def _build_factor_value_frame(
    features: pl.DataFrame,
    specs: list[FactorSpec],
    *,
    created_at: datetime,
) -> pl.DataFrame:
    if features.is_empty() or not specs:
        return pl.DataFrame(schema=FACTOR_VALUE_SCHEMA)
    wide = _pivot_feature_values(features)
    input_dataset_version = _text_mode(features, "input_dataset_version", "feature_value:unknown")
    input_hash = _text_mode(features, "input_hash", "sha256:unknown")
    code_version = _code_version()
    frames: list[pl.DataFrame] = []
    for spec in specs:
        if not spec.causal:
            raise ValueError(f"factor {spec.factor_id} is marked non-causal")
        if not set(spec.input_features).issubset(set(wide.columns)):
            continue
        raw = _factor_raw_expr(spec)
        if spec.clip_abs is not None:
            raw = winsorize_expr(raw, lower=-spec.clip_abs, upper=spec.clip_abs)
        availability_offset = timedelta(
            seconds=_timeframe_seconds(spec.timeframe) * spec.availability_lag_bars
        )
        frame = wide.select(
            [
                pl.lit(spec.factor_id).alias("factor_id"),
                pl.lit(spec.factor_name).alias("factor_name"),
                pl.lit(spec.factor_family).alias("factor_family"),
                pl.lit(spec.factor_version).alias("factor_version"),
                pl.col("symbol"),
                pl.col("timeframe"),
                pl.col("ts"),
                pl.col("ts").alias("event_time"),
                (pl.col("ts") + pl.lit(availability_offset)).alias("available_time"),
                raw.cast(pl.Float64, strict=False).alias("raw_value"),
                pl.lit(spec.status.value).alias("factor_status"),
                pl.lit(spec.expression_hash).alias("expression_hash"),
                pl.lit(safe_json_dumps(list(spec.input_features))).alias("input_features_json"),
                pl.lit(input_dataset_version).alias("input_dataset_version"),
                pl.lit(input_dataset_version).alias("data_version"),
                pl.lit(input_hash).alias("input_hash"),
                pl.lit(code_version).alias("code_version"),
                pl.lit(created_at).alias("calculated_at"),
                pl.lit(created_at).alias("created_at"),
                pl.lit(SOURCE_NAME).alias("source"),
                pl.lit(spec.direction).alias("_direction"),
                pl.lit(spec.min_cross_section).alias("_min_cross_section"),
            ]
        )
        frames.append(frame)
    if not frames:
        return pl.DataFrame(schema=FACTOR_VALUE_SCHEMA)
    raw_values = pl.concat(frames, how="vertical_relaxed")
    group = ["factor_id", "factor_version", "timeframe", "ts"]
    clean = raw_values.with_columns(
        pl.when(pl.col("raw_value").is_finite())
        .then(pl.col("raw_value"))
        .otherwise(None)
        .alias("_clean_raw_value")
    ).with_columns(
        [
            pl.col("_clean_raw_value").count().over(group).alias("_valid_count"),
            pl.col("_clean_raw_value").mean().over(group).alias("_group_mean"),
            pl.col("_clean_raw_value").std().over(group).alias("_group_std"),
            pl.col("_clean_raw_value").rank("average").over(group).alias("_rank"),
        ]
    )
    normalized = clean.with_columns(
        [
            (
                pl.col("_clean_raw_value").is_not_null()
                & (pl.col("_valid_count") >= pl.col("_min_cross_section"))
            ).alias("is_valid"),
            pl.when(pl.col("_clean_raw_value").is_null())
            .then(pl.lit("invalid_or_insufficient_input"))
            .when(pl.col("_valid_count") < pl.col("_min_cross_section"))
            .then(pl.lit("insufficient_cross_section"))
            .otherwise(None)
            .alias("invalid_reason"),
            pl.when((pl.col("_group_std") > 0) & (pl.col("_valid_count") > 1))
            .then((pl.col("_clean_raw_value") - pl.col("_group_mean")) / pl.col("_group_std"))
            .otherwise(None)
            .alias("normalized_value"),
            pl.when(pl.col("_valid_count") > 1)
            .then((pl.col("_rank") - 1.0) / (pl.col("_valid_count") - 1.0))
            .otherwise(0.5)
            .alias("rank_value"),
        ]
    ).with_columns(
        pl.when(pl.col("is_valid"))
        .then(pl.col("normalized_value") * pl.col("_direction"))
        .otherwise(None)
        .alias("value")
    ).with_columns(
        pl.when(pl.col("is_valid"))
        .then(pl.lit("[]"))
        .otherwise(pl.format('["{}"]', pl.col("invalid_reason").fill_null("invalid_factor_value")))
        .alias("quality_flags_json")
    )
    return normalized.select(list(FACTOR_VALUE_SCHEMA)).sort(
        ["factor_id", "symbol", "timeframe", "ts"]
    )


def _pivot_feature_values(features: pl.DataFrame) -> pl.DataFrame:
    selected = (
        features.select(["symbol", "timeframe", "ts", "feature_name", "value"])
        .unique(subset=["symbol", "timeframe", "ts", "feature_name"], keep="last")
        .sort(["symbol", "timeframe", "ts", "feature_name"])
    )
    return selected.pivot(
        on="feature_name",
        index=["symbol", "timeframe", "ts"],
        values="value",
        aggregate_function="last",
    )


def _factor_raw_expr(spec: FactorSpec) -> pl.Expr:
    params = spec.params
    if spec.template == "feature":
        return numeric(str(params.get("feature") or spec.input_features[0]))
    if spec.template == "neg_feature":
        return -numeric(str(params.get("feature") or spec.input_features[0]))
    if spec.template == "product":
        return numeric(str(params["left"])) * numeric(str(params["right"]))
    if spec.template == "difference":
        return numeric(str(params["left"])) - numeric(str(params["right"]))
    if spec.template == "safe_divide":
        return safe_divide(numeric(str(params["numerator"])), numeric(str(params["denominator"])))
    if spec.template == "vol_adjusted":
        return safe_divide(
            numeric(str(params["return_feature"])),
            numeric(str(params["vol_feature"])),
        )
    if spec.template == "range_vol_ratio":
        return safe_divide(
            numeric(str(params["range_feature"])),
            numeric(str(params["vol_feature"])),
        )
    if spec.template == "range_location":
        return numeric(str(params["range_feature"])) * (
            numeric(str(params["location_feature"])) - 0.5
        )
    if spec.template == "liquidity_adjusted":
        return safe_divide(
            numeric(str(params["return_feature"])),
            numeric(str(params["liquidity_feature"])),
        )
    raise ValueError(f"unsupported factor template: {spec.template}")


def _timeframe_seconds(timeframe: str) -> int:
    text = str(timeframe or "").strip()
    if len(text) < 2:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    amount_text, unit = text[:-1], text[-1].lower()
    if not amount_text.isdigit():
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    amount = int(amount_text)
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    if amount <= 0 or unit not in multipliers:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return amount * multipliers[unit]


def _attach_symbol_costs(
    dataset: pl.DataFrame,
    root: Path,
    *,
    cost_quantile: str,
    warnings: list[str],
) -> pl.DataFrame:
    cost_frame = _latest_cost_frame(root, cost_quantile=cost_quantile, warnings=warnings)
    if cost_frame.is_empty():
        return dataset.with_columns(
            [
                pl.lit(DEFAULT_RESEARCH_COST_BPS).alias("cost_bps"),
                pl.lit("global_default_v0").alias("cost_model_version"),
                pl.lit("global_default").alias("cost_source"),
            ]
        )
    joined = dataset.join(cost_frame, on="symbol", how="left")
    return joined.with_columns(
        [
            pl.col("cost_bps").fill_null(DEFAULT_RESEARCH_COST_BPS),
            pl.col("cost_model_version").fill_null("global_default_v0"),
            pl.col("cost_source").fill_null("global_default"),
        ]
    )


def _latest_cost_frame(root: Path, *, cost_quantile: str, warnings: list[str]) -> pl.DataFrame:
    costs = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET)
    cost_column = f"total_cost_bps_{cost_quantile}"
    if costs.is_empty() or cost_column not in costs.columns:
        warnings.append("cost_bucket_daily missing; using research global default cost")
        return pl.DataFrame()
    latest = costs
    if "day" in latest.columns:
        latest = latest.sort("day")
    latest = latest.unique(subset=["symbol"], keep="last")
    columns = ["symbol", cost_column]
    if "cost_model_version" in latest.columns:
        columns.append("cost_model_version")
    if "cost_source" in latest.columns:
        columns.append("cost_source")
    elif "source" in latest.columns:
        latest = latest.with_columns(pl.col("source").alias("cost_source"))
        columns.append("cost_source")
    selected = latest.select([column for column in columns if column in latest.columns]).rename(
        {cost_column: "cost_bps"}
    )
    if "cost_model_version" not in selected.columns:
        selected = selected.with_columns(pl.lit("unknown").alias("cost_model_version"))
    if "cost_source" not in selected.columns:
        selected = selected.with_columns(pl.lit("unknown").alias("cost_source"))
    return selected


def _factor_evidence_row(
    dataset: pl.DataFrame,
    *,
    factor_meta: dict[str, Any],
    as_of_date: date,
    horizon_bars: int,
    decision_delay_bars: int,
    min_samples: int,
    top_quantile: float,
    created_at: datetime,
) -> dict[str, Any]:
    total_rows = dataset.height
    valid = dataset.filter(
        pl.col("is_valid")
        & pl.col("value").is_not_null()
        & pl.col("value").is_finite()
        & pl.col("forward_return").is_not_null()
    )
    valid_rows = valid.height
    coverage = valid_rows / total_rows if total_rows else 0.0
    stats_input = valid.with_columns(pl.col("value").alias("alpha_score"))
    ic_stats = compute_ic(stats_input)
    rank_ic_stats = compute_rank_ic(stats_input)
    portfolio = _portfolio_stats(valid, top_quantile=top_quantile)
    decision, score, reasons, decision_warnings = _factor_decision(
        valid_sample_count=valid_rows,
        min_samples=min_samples,
        coverage=coverage,
        rank_ic_mean=rank_ic_stats.mean,
        rank_ic_tstat=rank_ic_stats.tstat,
        long_short_mean_bps=portfolio["long_short_mean_bps"],
        edge_cost_ratio=portfolio["edge_cost_ratio"],
    )
    warnings = []
    if ic_stats.status != "ok":
        warnings.append(ic_stats.status)
    if rank_ic_stats.status != "ok":
        warnings.append(f"rank_{rank_ic_stats.status}")
    warnings.extend(decision_warnings)
    start_ts = _frame_datetime(valid, "feature_ts", "min")
    end_ts = _frame_datetime(valid, "feature_ts", "max")
    return {
        "as_of_date": as_of_date.isoformat(),
        **factor_meta,
        "horizon_bars": horizon_bars,
        "decision_delay_bars": decision_delay_bars,
        "sample_count": total_rows,
        "valid_sample_count": valid_rows,
        "coverage": coverage,
        "ic_mean": ic_stats.mean,
        "ic_tstat": ic_stats.tstat,
        "rank_ic_mean": rank_ic_stats.mean,
        "rank_ic_tstat": rank_ic_stats.tstat,
        "ic_period_count": max(ic_stats.period_count, rank_ic_stats.period_count),
        "top_quantile": top_quantile,
        **portfolio,
        "decision": decision,
        "score": score,
        "reasons_json": safe_json_dumps(reasons),
        "warnings_json": safe_json_dumps(_dedupe(warnings)),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "created_at": created_at,
        "source": SOURCE_NAME,
    }


def _portfolio_stats(valid: pl.DataFrame, *, top_quantile: float) -> dict[str, Any]:
    defaults = {
        "long_only_mean_bps": 0.0,
        "long_short_mean_bps": 0.0,
        "top_mean_bps": 0.0,
        "bottom_mean_bps": 0.0,
        "win_rate": 0.0,
        "hit_rate": 0.0,
        "turnover": 0.0,
        "max_drawdown": 0.0,
        "edge_cost_ratio": 0.0,
        "cost_ratio": 0.0,
        "period_count": 0,
    }
    if valid.is_empty():
        return defaults
    ranked = valid.with_columns(
        [
            pl.col("value").rank("average").over("decision_ts").alias("_rank"),
            pl.col("value").count().over("decision_ts").alias("_count"),
            (pl.col("forward_return") * 10_000.0).alias("_gross_bps"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("_count") > 1)
            .then((pl.col("_rank") - 1.0) / (pl.col("_count") - 1.0))
            .otherwise(0.5)
            .alias("_rank_pct"),
            (pl.col("_gross_bps") - pl.col("cost_bps")).alias("_after_cost_bps"),
        ]
    )
    top = ranked.filter(pl.col("_rank_pct") >= 1.0 - top_quantile)
    bottom = ranked.filter(pl.col("_rank_pct") <= top_quantile)
    if top.is_empty():
        return defaults
    top_gross = _mean(_float_values(top, "_gross_bps"))
    top_after = _mean(_float_values(top, "_after_cost_bps"))
    bottom_gross = _mean(_float_values(bottom, "_gross_bps")) if not bottom.is_empty() else 0.0
    mean_cost = _mean(_float_values(ranked, "cost_bps"))
    long_short = top_gross - bottom_gross - (2.0 * mean_cost)
    period_returns = (
        top.group_by("decision_ts", maintain_order=True)
        .agg(pl.col("_after_cost_bps").mean().alias("_period_after_cost_bps"))
        .sort("decision_ts")
    )
    period_values = [
        value / 10_000.0
        for value in _float_values(period_returns, "_period_after_cost_bps")
    ]
    return {
        "long_only_mean_bps": top_after,
        "long_short_mean_bps": long_short,
        "top_mean_bps": top_gross,
        "bottom_mean_bps": bottom_gross,
        "win_rate": _positive_rate(_float_values(top, "_after_cost_bps")),
        "hit_rate": 1.0 if long_short > 0 else 0.0,
        "turnover": _turnover_proxy(top),
        "max_drawdown": _max_drawdown(period_values),
        "edge_cost_ratio": long_short / mean_cost if mean_cost > 0 else 0.0,
        "cost_ratio": mean_cost / abs(long_short) if long_short else 0.0,
        "period_count": period_returns.height,
    }


def _factor_decision(
    *,
    valid_sample_count: int,
    min_samples: int,
    coverage: float,
    rank_ic_mean: float,
    rank_ic_tstat: float,
    long_short_mean_bps: float,
    edge_cost_ratio: float,
) -> tuple[str, float, list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    score = (
        rank_ic_mean * 100.0
        + rank_ic_tstat
        + min(edge_cost_ratio, 5.0)
        + long_short_mean_bps / 100.0
    )
    if valid_sample_count < min_samples:
        reasons.append("sample_count_below_min")
        warnings.append("insufficient_samples")
        return "RESEARCH", score, reasons, warnings
    if coverage < 0.50:
        reasons.append("coverage_below_0_50")
        return "KILL", score, reasons, warnings
    if rank_ic_mean < -0.01 and long_short_mean_bps < 0:
        reasons.append("negative_rank_ic_and_spread")
        return "KILL", score, reasons, warnings
    if (
        coverage >= 0.80
        and rank_ic_mean > 0.0
        and rank_ic_tstat >= 1.0
        and long_short_mean_bps > 0
        and edge_cost_ratio > 1.0
    ):
        reasons.append("positive_rank_ic_after_cost_spread")
        return "PAPER_READY", score, reasons, warnings
    if rank_ic_mean > 0.0 or long_short_mean_bps > 0:
        reasons.append("positive_but_not_paper_ready")
        return "KEEP_SHADOW", score, reasons, warnings
    reasons.append("weak_or_neutral_evidence")
    return "RESEARCH", score, reasons, warnings


def _candidate_frame_from_evidence(
    evidence: pl.DataFrame,
    *,
    as_of_date: date,
    created_at: datetime,
) -> pl.DataFrame:
    if evidence.is_empty():
        return pl.DataFrame(schema=FACTOR_CANDIDATE_SCHEMA)
    rows: list[dict[str, Any]] = []
    group_columns = ["factor_id", "factor_name", "factor_family", "factor_version", "timeframe"]
    for key, group in evidence.group_by(group_columns, maintain_order=True):
        meta = dict(zip(group_columns, _as_tuple(key, len(group_columns)), strict=True))
        sorted_group = group.sort("score", descending=True)
        best = sorted_group.to_dicts()[0]
        decision = str(best.get("decision") or "RESEARCH")
        scores = _float_values(group, "score")
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                **meta,
                "best_horizon_bars": int(best.get("horizon_bars") or 0),
                "tested_horizon_count": group["horizon_bars"].n_unique(),
                "best_score": _float(best.get("score")) or 0.0,
                "avg_score": _mean(scores),
                "best_rank_ic_mean": _float(best.get("rank_ic_mean")) or 0.0,
                "best_rank_ic_tstat": _float(best.get("rank_ic_tstat")) or 0.0,
                "best_long_short_mean_bps": _float(best.get("long_short_mean_bps")) or 0.0,
                "candidate_state": decision,
                "recommended_action": _recommended_action(decision),
                "promotion_block_reasons_json": (
                    "[]"
                    if decision == "PAPER_READY"
                    else str(best.get("reasons_json") or "[]")
                ),
                "manual_review_required": True,
                "created_at": created_at,
                "source": SOURCE_NAME,
            }
        )
    return _schema_frame(rows, FACTOR_CANDIDATE_SCHEMA).sort(
        ["candidate_state", "best_score"],
        descending=[False, True],
    )


def _factor_correlation_frame(
    values: pl.DataFrame,
    *,
    as_of_date: date,
    factor_version: str,
    timeframe: str,
    created_at: datetime,
) -> pl.DataFrame:
    if values.is_empty():
        return pl.DataFrame(schema=FACTOR_CORRELATION_SCHEMA)
    valid = values.filter(pl.col("is_valid") & pl.col("value").is_not_null())
    if valid.height > 250_000:
        valid = valid.sort("ts", descending=True).head(250_000)
    pivot = (
        valid.select(["symbol", "timeframe", "ts", "factor_id", "value"])
        .unique(subset=["symbol", "timeframe", "ts", "factor_id"], keep="last")
        .pivot(
            on="factor_id",
            index=["symbol", "timeframe", "ts"],
            values="value",
            aggregate_function="last",
        )
    )
    factor_ids = sorted(
        column for column in pivot.columns if column not in {"symbol", "timeframe", "ts"}
    )
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(factor_ids):
        for right in factor_ids[left_index + 1 :]:
            pairs = pivot.select([left, right]).drop_nulls()
            rows.append(
                {
                    "as_of_date": as_of_date.isoformat(),
                    "factor_id_left": left,
                    "factor_id_right": right,
                    "factor_version": factor_version,
                    "timeframe": timeframe,
                    "sample_count": pairs.height,
                    "correlation": _pearson(
                        pairs[left].to_list() if left in pairs.columns else [],
                        pairs[right].to_list() if right in pairs.columns else [],
                    ),
                    "created_at": created_at,
                    "source": SOURCE_NAME,
                }
            )
    return _schema_frame(rows, FACTOR_CORRELATION_SCHEMA)


def _recommended_action(decision: str) -> str:
    return {
        "KILL": "drop_or_quarantine",
        "RESEARCH": "research_only",
        "KEEP_SHADOW": "keep_shadow",
        "PAPER_READY": "paper_review_only",
    }.get(decision, "research_only")


def _turnover_proxy(top: pl.DataFrame) -> float:
    if top.is_empty():
        return 0.0
    selected_sets: list[set[str]] = []
    for _period, group in top.group_by("decision_ts", maintain_order=True):
        selected_sets.append(set(str(value) for value in group["symbol"].to_list()))
    if len(selected_sets) < 2:
        return 0.0
    changes = 0
    for previous, current in zip(selected_sets, selected_sets[1:], strict=False):
        if previous != current:
            changes += 1
    return changes / (len(selected_sets) - 1)


def _schema_frame(rows: list[dict[str, Any]], schema: dict[str, Any]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    frame = pl.DataFrame(rows, schema=schema, orient="row")
    return frame.select(list(schema))


def _normalize_datetime(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in df.columns:
        return df
    if df.schema.get(column) == pl.String:
        return df.with_columns(pl.col(column).str.to_datetime(time_zone="UTC", strict=False))
    return df.with_columns(pl.col(column).cast(pl.Datetime(time_zone="UTC")).alias(column))


def _frame_datetime(df: pl.DataFrame, column: str, op: str) -> datetime | None:
    if df.is_empty() or column not in df.columns:
        return None
    try:
        value = df.select(getattr(pl.col(column), op)()).item()
    except Exception:
        return None
    return value.astimezone(UTC) if isinstance(value, datetime) else None


def _frame_datetime_iso(df: pl.DataFrame, column: str, op: str) -> str | None:
    value = _frame_datetime(df, column, op)
    return value.isoformat() if value else None


def _parse_as_of_date(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "auto").strip()
    if text.lower() in {"", "auto", "today"}:
        return datetime.now(UTC).date()
    return date.fromisoformat(text)


def _decision_counts(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty() or "decision" not in frame.columns:
        return {}
    return {
        str(row["decision"]): int(row["len"])
        for row in frame.group_by("decision").len().sort("decision").to_dicts()
    }


def _count_value(frame: pl.DataFrame, column: str, expected: str) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    return frame.filter(pl.col(column) == expected).height


def _text_mode(df: pl.DataFrame, column: str, fallback: str) -> str:
    if df.is_empty() or column not in df.columns:
        return fallback
    values = [str(value) for value in df[column].drop_nulls().to_list() if str(value).strip()]
    if not values:
        return fallback
    return Counter(values).most_common(1)[0][0]


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _float_values(frame: pl.DataFrame, column: str) -> list[float]:
    if frame.is_empty() or column not in frame.columns:
        return []
    values: list[float] = []
    for value in frame[column].to_list():
        normalized = _float(value)
        if normalized is not None:
            values.append(normalized)
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _positive_rate(values: list[float]) -> float:
    return sum(1 for value in values if value > 0) / len(values) if values else 0.0


def _max_drawdown(values: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def _pearson(left: list[Any], right: list[Any]) -> float:
    pairs = []
    for x_raw, y_raw in zip(left, right, strict=False):
        x = _float(x_raw)
        y = _float(y_raw)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 2:
        return 0.0
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator > 0 else 0.0


def _as_tuple(value: Any, size: int) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return tuple([value, *([None] * (size - 1))])[:size]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _code_version() -> str:
    commit = _git_commit()
    if commit:
        return f"{CODE_VERSION_PREFIX}:{commit}"
    return f"{CODE_VERSION_PREFIX}:{__version__}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None
