from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import (
    read_parquet_dataset,
    read_parquet_lazy,
    upsert_parquet_dataset,
    write_parquet_dataset,
)
from quant_lab.research.evidence import DEFAULT_RESEARCH_COST_BPS
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"
STRATEGY_EVIDENCE_SAMPLE_DATASET = Path("gold") / "strategy_evidence_sample"
STRATEGY_EVIDENCE_QUALITY_DATASET = Path("gold") / "strategy_evidence_quality"
EVIDENCE_VERSION = "strategy-evidence-v0.1"
SOURCE_NAME = "research.strategy_evidence.v0.1"
MIN_LIVE_SMALL_READY_SAMPLES = 30
MIN_LIVE_SMALL_READY_ENTRY_DAYS = 3
ALT_IMPULSE_SHADOW_CANDIDATE = "v5.alt_impulse_shadow"
ALT_IMPULSE_REGIME_SHADOW_MIN_COMPLETE_SAMPLES = 5
ALT_IMPULSE_REGIME_SHADOW_MIN_WIN_RATE = 0.50
HORIZON_HOURS = (4, 8, 12, 24, 48, 72, 120)
DEFAULT_INCREMENTAL_LOOKBACK_DAYS = 8
LIVE_READY_COST_SOURCES = {"mixed_actual_proxy", "actual_fills"}
LIVE_BLOCKING_COST_SOURCES = {"global_default", "cost_not_requested_no_order"}

STRATEGY_CANDIDATES = (
    "v5.btc_leadership_probe_strict",
    "v5.btc_leadership_blocked_relaxed",
    "v5.btc_leadership_alpha6_low_blocked",
    "v5.btc_leadership_f5_low_blocked",
    "v5.btc_leadership_no_breakout_blocked",
    "v5.sol_protect_exception",
    "v5.sol_protect_alpha6_low_exception",
    "v5.sol_protect_rsi_weak_exception",
    "v5.alt_impulse_shadow",
    "v5.multi_position_k1",
    "v5.multi_position_k2",
    "v5.multi_position_k3",
    "v5.swing_f4_f5_alpha6",
    "v5.f3_dominant_entry",
    "v5.f4_volume_expansion_entry",
    "v5.mean_reversion_sideways",
)

LABEL_DATASET = Path("gold") / "v5_candidate_label"
EVENT_DATASET = Path("silver") / "v5_candidate_event"
OUTCOME_DATASETS = {
    "v5_high_score_blocked_outcome": Path("silver") / "v5_high_score_blocked_outcome",
    "v5_shadow_outcome": Path("silver") / "v5_shadow_outcome",
}

SAMPLE_SCHEMA: dict[str, Any] = {
    "strategy": pl.Utf8,
    "evidence_version": pl.Utf8,
    "as_of_date": pl.Utf8,
    "candidate_id": pl.Utf8,
    "run_id": pl.Utf8,
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "candidate_name": pl.Utf8,
    "source_type": pl.Utf8,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "regime_state": pl.Utf8,
    "horizon_hours": pl.Int64,
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "label_ts": pl.Datetime(time_zone="UTC"),
    "entry_close": pl.Float64,
    "label_close": pl.Float64,
    "gross_bps": pl.Float64,
    "net_bps_after_cost": pl.Float64,
    "mfe_bps": pl.Float64,
    "mae_bps": pl.Float64,
    "win": pl.Boolean,
    "label_status": pl.Utf8,
    "label_reason": pl.Utf8,
    "cost_bps": pl.Float64,
    "cost_source": pl.Utf8,
    "block_reason": pl.Utf8,
    "final_decision": pl.Utf8,
    "final_score": pl.Float64,
    "rank_score_bps": pl.Float64,
    "rank_lookback_hours": pl.Int64,
    "selected_rank": pl.Int64,
    "top_k": pl.Int64,
    "selection_reason": pl.Utf8,
    "anti_leakage_check": pl.Utf8,
    "futures_data_available": pl.Boolean,
    "funding_available": pl.Boolean,
    "funding_cost_bps": pl.Float64,
    "mark_price_source": pl.Utf8,
    "liquidation_buffer_pct": pl.Float64,
    "proxy_warning": pl.Utf8,
    "expected_edge_bps": pl.Float64,
    "required_edge_bps": pl.Float64,
    "alpha6_score": pl.Float64,
    "alpha6_side": pl.Utf8,
    "protect_level": pl.Utf8,
    "risk_level": pl.Utf8,
    "btc_trend_state": pl.Utf8,
    "broad_market_positive_count": pl.Int64,
    "funding_state": pl.Utf8,
    "volatility_bucket": pl.Utf8,
    "source_path_inside_bundle": pl.Utf8,
    "source_event_key": pl.Utf8,
    "source_bundle_ts": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

SUMMARY_SCHEMA: dict[str, Any] = {
    "strategy": pl.Utf8,
    "evidence_version": pl.Utf8,
    "as_of_date": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "candidate_name": pl.Utf8,
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
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "start_ts": pl.Datetime(time_zone="UTC"),
    "end_ts": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

QUALITY_SCHEMA: dict[str, Any] = {
    "strategy": pl.Utf8,
    "evidence_version": pl.Utf8,
    "as_of_date": pl.Utf8,
    "severity": pl.Utf8,
    "warning_type": pl.Utf8,
    "warning_count": pl.Int64,
    "detail": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


class StrategyEvidenceBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    sample_rows: int = Field(ge=0)
    strategy_evidence_rows: int = Field(ge=0)
    extracted_sample_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    mode: str = "full"
    lookback_days: int | None = None
    decision_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_strategy_evidence(
    lake_root: str | Path,
    *,
    as_of_date: str | None = None,
    min_live_samples: int = MIN_LIVE_SMALL_READY_SAMPLES,
    mode: str = "full",
    lookback_days: int = DEFAULT_INCREMENTAL_LOOKBACK_DAYS,
    include_historical_outcomes: bool | None = None,
) -> StrategyEvidenceBuildResult:
    root = Path(lake_root)
    day = _as_of_date(as_of_date)
    normalized_mode = _normalize_build_mode(mode)
    samples, warnings = build_strategy_evidence_samples(
        root,
        as_of_date=day.isoformat(),
        mode=normalized_mode,
        lookback_days=lookback_days,
        include_historical_outcomes=(
            normalized_mode == "full"
            if include_historical_outcomes is None
            else include_historical_outcomes
        ),
    )
    summaries = summarize_strategy_evidence(
        _samples_for_summary(
            root,
            samples,
            normalized_mode,
            as_of_date=day,
            lookback_days=lookback_days,
        ),
        as_of_date=day,
        min_live_samples=min_live_samples,
    )
    sample_rows = publish_strategy_evidence_samples(
        root,
        samples,
        replace_as_of_dates=normalized_mode == "full",
    )
    summary_rows = publish_strategy_evidence_summary(root, summaries)
    publish_strategy_evidence_quality(root, day, warnings)
    return StrategyEvidenceBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        sample_rows=sample_rows,
        strategy_evidence_rows=summary_rows,
        extracted_sample_count=samples.height,
        candidate_count=len(summaries),
        mode=normalized_mode,
        lookback_days=lookback_days if normalized_mode == "incremental" else None,
        decision_counts=_decision_counts(summaries),
        warnings=warnings,
    )


def build_strategy_evidence_samples(
    lake_root: str | Path,
    *,
    as_of_date: str | None = None,
    mode: str = "full",
    lookback_days: int = DEFAULT_INCREMENTAL_LOOKBACK_DAYS,
    include_historical_outcomes: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    root = Path(lake_root)
    warnings: list[str] = []
    day = _as_of_date(as_of_date)
    normalized_mode = _normalize_build_mode(mode)
    if normalized_mode == "incremental":
        labels = _read_recent_dataset(
            root / LABEL_DATASET,
            day=day,
            lookback_days=lookback_days,
            timestamp_columns=("ts_utc", "source_bundle_ts", "created_at"),
        )
        events = _read_recent_dataset(
            root / EVENT_DATASET,
            day=day,
            lookback_days=lookback_days,
            timestamp_columns=("ts_utc", "bundle_ts", "ingest_ts"),
        )
    else:
        labels = read_parquet_dataset(root / LABEL_DATASET)
        events = read_parquet_dataset(root / EVENT_DATASET)
    cost_context = _CostContext(root)
    if labels.is_empty():
        warnings.append("v5_candidate_label_empty")

    event_context = _event_context_by_candidate_id(events)
    rows: list[dict[str, Any]] = []
    for label in labels.to_dicts():
        sample = _sample_from_candidate_label(label, event_context)
        if sample is not None:
            rows.append(sample)
    for dataset_name, relative_path in (
        OUTCOME_DATASETS.items() if include_historical_outcomes else []
    ):
        frame = (
            _read_recent_dataset(
                root / relative_path,
                day=day,
                lookback_days=lookback_days,
                timestamp_columns=("source_bundle_ts", "bundle_ts", "ingest_ts", "ts_utc", "ts"),
            )
            if normalized_mode == "incremental"
            else read_parquet_dataset(root / relative_path)
        )
        if frame.is_empty():
            continue
        for outcome in _dedupe_outcome_source_rows(dataset_name, frame.to_dicts()):
            rows.extend(_samples_from_outcome_row(dataset_name, outcome, cost_context))

    rows = _dedupe_formal_sample_rows(_filter_samples_as_of(rows, as_of_date))
    skipped_unknown = _unknown_symbol_count(rows)
    rows = _drop_unknown_symbol_samples(rows)
    if skipped_unknown:
        warnings.append(f"strategy_evidence_unknown_symbol_samples_skipped:{skipped_unknown}")
    if not rows:
        return pl.DataFrame(schema=SAMPLE_SCHEMA), warnings
    return normalize_strategy_evidence_samples(_formal_samples_frame(rows)), warnings


def summarize_strategy_evidence(
    samples: pl.DataFrame,
    *,
    as_of_date: date,
    min_live_samples: int = MIN_LIVE_SMALL_READY_SAMPLES,
) -> list[dict[str, Any]]:
    created_at = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    sample_rows = samples.to_dicts() if not samples.is_empty() else []
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in sample_rows:
        key = (
            str(row.get("strategy_candidate") or row.get("candidate_name") or "UNKNOWN"),
            normalize_symbol(row.get("symbol")),
            str(row.get("regime_state") or "UNKNOWN"),
            int(row.get("horizon_hours") or 0),
        )
        groups.setdefault(key, []).append(row)
    for (candidate, symbol, regime, horizon), candidate_rows in sorted(groups.items()):
        rows.append(
            _formal_summary_row(
                strategy_candidate=candidate,
                symbol=symbol,
                regime_state=regime,
                horizon_hours=horizon,
                rows=candidate_rows,
                as_of_date=as_of_date,
                created_at=created_at,
                min_live_samples=min_live_samples,
            )
        )
    return rows


def publish_strategy_evidence_samples(
    lake_root: str | Path,
    samples: pl.DataFrame,
    *,
    replace_as_of_dates: bool = True,
) -> int:
    dataset_path = Path(lake_root) / STRATEGY_EVIDENCE_SAMPLE_DATASET
    if samples.is_empty():
        return read_parquet_dataset(dataset_path).height
    samples = normalize_strategy_evidence_samples(samples)
    if not replace_as_of_dates:
        return upsert_parquet_dataset(
            samples,
            dataset_path,
            key_columns=[
                "strategy",
                "source_type",
                "candidate_id",
                "symbol",
                "strategy_candidate",
                "horizon_hours",
                "source_event_key",
            ],
        )
    existing = read_parquet_dataset(dataset_path)
    if _needs_formal_schema_replace(existing, ["strategy", "candidate_id", "horizon_hours"]):
        write_parquet_dataset(samples, dataset_path)
        return samples.height
    combined = _replace_matching_as_of_dates(existing, samples)
    normalized = _drop_unknown_symbol_rows(normalize_strategy_evidence_samples(combined))
    write_parquet_dataset(normalized, dataset_path)
    return normalized.height


def publish_strategy_evidence_quality(
    lake_root: str | Path,
    as_of_date: date,
    warnings: list[str],
) -> int:
    dataset_path = Path(lake_root) / STRATEGY_EVIDENCE_QUALITY_DATASET
    created_at = datetime.now(UTC)
    rows = _quality_rows_for_warnings(as_of_date, warnings, created_at)
    frame = pl.DataFrame(rows, schema=QUALITY_SCHEMA, orient="row")
    existing = read_parquet_dataset(dataset_path)
    combined = _replace_matching_as_of_dates(existing, frame)
    write_parquet_dataset(combined, dataset_path)
    return combined.height


def _quality_rows_for_warnings(
    as_of_date: date,
    warnings: list[str],
    created_at: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for warning in warnings:
        warning_type, _, raw_count = warning.partition(":")
        count = _int_or_none(raw_count) if raw_count else None
        rows.append(
            {
                "strategy": "v5",
                "evidence_version": EVIDENCE_VERSION,
                "as_of_date": as_of_date.isoformat(),
                "severity": "WARN",
                "warning_type": warning_type,
                "warning_count": int(count if count is not None else 1),
                "detail": warning,
                "created_at": created_at,
                "source": SOURCE_NAME,
            }
        )
    if rows:
        return rows
    return [
        {
            "strategy": "v5",
            "evidence_version": EVIDENCE_VERSION,
            "as_of_date": as_of_date.isoformat(),
            "severity": "PASS",
            "warning_type": "none",
            "warning_count": 0,
            "detail": "",
            "created_at": created_at,
            "source": SOURCE_NAME,
        }
    ]


def publish_strategy_evidence_summary(
    lake_root: str | Path,
    rows: list[dict[str, Any]],
) -> int:
    dataset_path = Path(lake_root) / STRATEGY_EVIDENCE_DATASET
    if not rows:
        return read_parquet_dataset(dataset_path).height
    frame = pl.DataFrame(rows, schema=SUMMARY_SCHEMA, orient="row")
    existing = read_parquet_dataset(dataset_path)
    if _needs_formal_schema_replace(
        existing,
        ["strategy", "strategy_candidate", "symbol", "regime_state", "horizon_hours"],
    ):
        write_parquet_dataset(normalize_strategy_evidence_decisions(frame), dataset_path)
        return frame.height
    combined = _replace_matching_as_of_dates(existing, normalize_strategy_evidence_decisions(frame))
    normalized = _drop_unknown_symbol_rows(normalize_strategy_evidence_decisions(combined))
    write_parquet_dataset(normalized, dataset_path)
    return normalized.height


def _replace_matching_as_of_dates(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    """Replace complete daily evidence snapshots instead of preserving stale daily rows."""
    if existing.is_empty():
        return incoming
    if incoming.is_empty():
        return existing
    if "as_of_date" not in existing.columns or "as_of_date" not in incoming.columns:
        return pl.concat([existing, incoming], how="diagonal_relaxed")
    dates = {
        str(value)
        for value in incoming["as_of_date"].drop_nulls().cast(pl.Utf8).unique().to_list()
        if str(value).strip()
    }
    if not dates:
        return pl.concat([existing, incoming], how="diagonal_relaxed")
    retained = existing.filter(~pl.col("as_of_date").cast(pl.Utf8).is_in(sorted(dates)))
    if retained.is_empty():
        return incoming
    return pl.concat([retained, incoming], how="diagonal_relaxed")


def _normalize_build_mode(mode: str) -> str:
    normalized = str(mode or "full").strip().lower()
    if normalized not in {"full", "incremental"}:
        raise ValueError("mode must be either 'full' or 'incremental'")
    return normalized


def _samples_for_summary(
    lake_root: Path,
    incoming_samples: pl.DataFrame,
    mode: str,
    *,
    as_of_date: date,
    lookback_days: int,
) -> pl.DataFrame:
    if mode == "full":
        return incoming_samples
    existing = _read_recent_dataset(
        lake_root / STRATEGY_EVIDENCE_SAMPLE_DATASET,
        day=as_of_date,
        lookback_days=lookback_days,
        timestamp_columns=("ts_utc", "decision_ts", "label_ts"),
    )
    frames = [frame for frame in [existing, incoming_samples] if not frame.is_empty()]
    if not frames:
        return pl.DataFrame(schema=SAMPLE_SCHEMA)
    combined = pl.concat(frames, how="diagonal_relaxed")
    return normalize_strategy_evidence_samples(combined).unique(
        subset=[
            "strategy",
            "source_type",
            "candidate_id",
            "symbol",
            "strategy_candidate",
            "horizon_hours",
            "source_event_key",
        ],
        keep="last",
        maintain_order=True,
    )


def _read_recent_dataset(
    dataset_path: Path,
    *,
    day: date,
    lookback_days: int,
    timestamp_columns: tuple[str, ...],
) -> pl.DataFrame:
    try:
        lazy = read_parquet_lazy(dataset_path)
        columns = lazy.collect_schema().names()
    except Exception:
        return pl.DataFrame()
    ts_column = next((column for column in timestamp_columns if column in columns), None)
    if ts_column is None:
        return lazy.collect()
    start = datetime.combine(day - timedelta(days=max(lookback_days, 0)), time.min, tzinfo=UTC)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=UTC)
    try:
        ts_expr = _utc_datetime_expr(ts_column)
        return lazy.filter(ts_expr.is_between(start, end, closed="left")).collect()
    except Exception:
        frame = read_parquet_dataset(dataset_path)
        if frame.is_empty() or ts_column not in frame.columns:
            return frame
        ts_expr = _utc_datetime_expr(ts_column)
        return frame.filter(ts_expr.is_between(start, end, closed="left"))


def _utc_datetime_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False)


def normalize_strategy_evidence_decisions(evidence: pl.DataFrame) -> pl.DataFrame:
    if evidence.is_empty() or "decision" not in evidence.columns:
        return evidence
    rows: list[dict[str, Any]] = []
    for row in evidence.to_dicts():
        candidate = _canonical_candidate_name(
            row.get("strategy_candidate") or row.get("candidate_name"),
            dataset_name="strategy_evidence",
        )
        if candidate:
            row["strategy_candidate"] = candidate
            row["candidate_name"] = candidate
        decision, reasons = strategy_evidence_decision_ladder(
            sample_count=int(_finite_float(row.get("sample_count")) or 0),
            complete_sample_count=int(_finite_float(row.get("complete_sample_count")) or 0),
            avg_net_bps=_finite_float(row.get("avg_net_bps")),
            p25_net_bps=_finite_float(row.get("p25_net_bps")),
            win_rate=_finite_float(row.get("win_rate")),
            paper_days=int(_finite_float(row.get("paper_days")) or 0),
            entry_day_count=int(_finite_float(row.get("entry_day_count")) or 0),
            paper_pnl_observed_count=int(
                _finite_float(row.get("paper_pnl_observed_count")) or 0
            ),
            paper_pnl_day_count=int(_finite_float(row.get("paper_pnl_day_count")) or 0),
            arrival_mid_coverage=_finite_float(row.get("arrival_mid_coverage")) or 0.0,
            paper_slippage_coverage=_finite_float(row.get("paper_slippage_coverage")) or 0.0,
            cost_source_mix=row.get("cost_source_mix"),
            candidate_name=candidate,
        )
        row["decision"] = decision
        row["decision_reasons"] = _json(reasons)
        rows.append(row)
    normalized = pl.DataFrame(rows, schema=evidence.schema, orient="row")
    keys = [
        column
        for column in [
            "strategy",
            "evidence_version",
            "as_of_date",
            "strategy_candidate",
            "symbol",
            "regime_state",
            "horizon_hours",
        ]
        if column in normalized.columns
    ]
    return normalized.unique(subset=keys, keep="last", maintain_order=True) if keys else normalized


def normalize_strategy_evidence_samples(samples: pl.DataFrame) -> pl.DataFrame:
    if samples.is_empty():
        return pl.DataFrame(schema=SAMPLE_SCHEMA)
    rows: list[dict[str, Any]] = []
    for row in samples.to_dicts():
        payload = _payload(row)
        candidate = _canonical_candidate_name(
            row.get("strategy_candidate") or row.get("candidate_name"),
            dataset_name="strategy_evidence_sample",
        )
        if candidate:
            row["strategy_candidate"] = candidate
            row["candidate_name"] = candidate
        if not str(row.get("source_type") or "").strip():
            row["source_type"] = _source_type(
                str(row.get("source_dataset") or "strategy_evidence_sample"),
                row,
                payload,
            )
        rows.append({column: row.get(column) for column in SAMPLE_SCHEMA})
    return pl.DataFrame(rows, schema=SAMPLE_SCHEMA, orient="row")


def _drop_unknown_symbol_rows(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "symbol" not in frame.columns:
        return frame
    symbol = pl.col("symbol").fill_null("").cast(pl.Utf8).str.to_uppercase()
    return frame.filter(symbol != "UNKNOWN")


def _needs_formal_schema_replace(existing: pl.DataFrame, required_columns: list[str]) -> bool:
    return not existing.is_empty() and not set(required_columns).issubset(existing.columns)


def _event_context_by_candidate_id(events: pl.DataFrame) -> dict[str, dict[str, Any]]:
    if events.is_empty() or "candidate_id" not in events.columns:
        return {}
    context: dict[str, dict[str, Any]] = {}
    for row in events.to_dicts():
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            context[candidate_id] = row
    return context


def _sample_from_candidate_label(
    label: dict[str, Any],
    event_context: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    candidate_id = str(label.get("candidate_id") or "").strip()
    strategy_candidate = _canonical_candidate_name(
        label.get("strategy_candidate"),
        dataset_name="v5_candidate_label",
    )
    ts_utc = _parse_timestamp(label.get("ts_utc"))
    horizon_hours = _int_or_none(label.get("horizon_hours"))
    if not candidate_id or not strategy_candidate or ts_utc is None or horizon_hours is None:
        return None

    event = event_context.get(candidate_id, {})
    symbol = normalize_symbol(label.get("symbol") or event.get("symbol"))
    regime_state = str(label.get("regime_state") or event.get("regime_state") or "UNKNOWN")
    created_at = _parse_timestamp(label.get("created_at")) or datetime.now(UTC)
    return {
        "strategy": str(label.get("strategy") or "v5"),
        "evidence_version": EVIDENCE_VERSION,
        "as_of_date": ts_utc.date().isoformat(),
        "candidate_id": candidate_id,
        "run_id": str(label.get("run_id") or event.get("run_id") or ""),
        "ts_utc": ts_utc,
        "symbol": symbol,
        "strategy_candidate": strategy_candidate,
        "candidate_name": strategy_candidate,
        "source_type": "candidate_event_label",
        "sample_count": 1,
        "complete_sample_count": 1 if str(label.get("label_status") or "") == "complete" else 0,
        "regime_state": regime_state,
        "horizon_hours": horizon_hours,
        "decision_ts": _parse_timestamp(label.get("decision_ts")),
        "label_ts": _parse_timestamp(label.get("label_ts")),
        "entry_close": _finite_float(label.get("entry_close")),
        "label_close": _finite_float(label.get("label_close")),
        "gross_bps": _finite_float(label.get("gross_bps")),
        "net_bps_after_cost": _finite_float(label.get("net_bps_after_cost")),
        "mfe_bps": _finite_float(label.get("mfe_bps")),
        "mae_bps": _finite_float(label.get("mae_bps")),
        "win": label.get("win"),
        "label_status": str(label.get("label_status") or ""),
        "label_reason": str(label.get("label_reason") or ""),
        "cost_bps": _finite_float(label.get("cost_bps")),
        "cost_source": str(label.get("cost_source") or "MISSING"),
        "block_reason": str(label.get("block_reason") or event.get("block_reason") or ""),
        "final_decision": str(label.get("final_decision") or event.get("final_decision") or ""),
        "final_score": _finite_float(label.get("final_score") or event.get("final_score")),
        "expected_edge_bps": _finite_float(
            label.get("expected_edge_bps") or event.get("expected_edge_bps")
        ),
        "required_edge_bps": _finite_float(
            label.get("required_edge_bps") or event.get("required_edge_bps")
        ),
        "alpha6_score": _finite_float(label.get("alpha6_score") or event.get("alpha6_score")),
        "alpha6_side": str(label.get("alpha6_side") or event.get("alpha6_side") or ""),
        "protect_level": str(label.get("protect_level") or event.get("protect_level") or ""),
        "risk_level": str(label.get("risk_level") or event.get("risk_level") or ""),
        "btc_trend_state": str(
            label.get("btc_trend_state") or event.get("btc_trend_state") or ""
        ),
        "broad_market_positive_count": _int_or_none(
            label.get("broad_market_positive_count")
            or event.get("broad_market_positive_count")
        ),
        "funding_state": str(label.get("funding_state") or event.get("funding_state") or ""),
        "volatility_bucket": str(
            label.get("volatility_bucket") or event.get("volatility_bucket") or ""
        ),
        "source_path_inside_bundle": str(
            label.get("source_path_inside_bundle")
            or event.get("source_path_inside_bundle")
            or ""
        ),
        "source_event_key": candidate_id,
        "source_bundle_ts": _parse_timestamp(event.get("bundle_ts") or event.get("ingest_ts")),
        "created_at": created_at,
        "source": SOURCE_NAME,
    }


def _samples_from_outcome_row(
    dataset_name: str,
    row: dict[str, Any],
    cost_context: _CostContext,
) -> list[dict[str, Any]]:
    payload = _payload(row)
    horizons = _outcome_horizons(row, payload)
    samples: list[dict[str, Any]] = []
    for horizon_hours in horizons:
        sample = _sample_from_outcome_row(
            dataset_name,
            row,
            cost_context,
            horizon_hours=horizon_hours,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def _sample_from_outcome_row(
    dataset_name: str,
    row: dict[str, Any],
    cost_context: _CostContext,
    *,
    horizon_hours: int | None = None,
) -> dict[str, Any] | None:
    payload = _payload(row)
    strategy_candidate = _candidate_name(dataset_name, row, payload)
    ts_utc = _first_timestamp(
        row,
        payload,
        [
            "ts_utc",
            "decision_ts",
            "event_ts",
            "label_ts",
            "created_at",
            "timestamp",
            "time",
            "ts",
            "bundle_ts",
            "ingest_ts",
        ],
    )
    horizon_hours = horizon_hours or _horizon_hours(row, payload)
    if strategy_candidate is None or ts_utc is None or horizon_hours is None:
        return None

    symbol = _symbol(row, payload, strategy_candidate)
    cost = cost_context.for_sample(symbol=symbol, ts_utc=ts_utc, row=row, payload=payload)
    net_bps = _outcome_net_bps(row, payload, horizon_hours=horizon_hours)
    if net_bps is None:
        return None
    gross_bps = _first_numeric(
        row,
        payload,
        _horizon_keys(
            horizon_hours,
            ["gross_bps", "gross_return_bps", "return_bps", "forward_return_bps"],
        ),
    )
    if gross_bps is None and net_bps is not None:
        gross_bps = net_bps + float(cost["cost_bps"] or 0.0)
    win = _first_bool(
        row,
        payload,
        _horizon_keys(
            horizon_hours,
            ["win", "profitable", "is_profitable", "success", "label_win", "outcome_win"],
        ),
    )
    if win is None and net_bps is not None:
        win_rate_value = _first_numeric(row, payload, _horizon_keys(horizon_hours, ["win_rate"]))
        win = (win_rate_value > 0.5) if win_rate_value is not None else net_bps > 0.0
    label_status = _outcome_label_status(row, payload, horizon_hours, net_bps)
    source_type = _source_type(dataset_name, row, payload)
    source_count = _source_sample_count(row, payload, source_type=source_type)
    source_event_key = _source_event_key(dataset_name, row, payload)
    run_id = str(row.get("run_id") or _first_text(row, payload, ["run_id"]) or "")
    candidate_id = _outcome_candidate_id(
        dataset_name=dataset_name,
        row=row,
        payload=payload,
        source_type=source_type,
        strategy_candidate=strategy_candidate,
        symbol=symbol,
        ts_utc=ts_utc,
    )
    return {
        "strategy": str(row.get("strategy") or payload.get("strategy") or "v5"),
        "evidence_version": EVIDENCE_VERSION,
        "as_of_date": ts_utc.date().isoformat(),
        "candidate_id": candidate_id,
        "run_id": run_id,
        "ts_utc": ts_utc,
        "symbol": symbol,
        "strategy_candidate": strategy_candidate,
        "candidate_name": strategy_candidate,
        "source_type": source_type,
        "sample_count": source_count,
        "complete_sample_count": _source_complete_sample_count(
            row,
            payload,
            source_count,
            label_status,
            source_type=source_type,
            horizon_hours=horizon_hours,
        ),
        "regime_state": _first_text(
            row,
            payload,
            ["regime_state", "regime", "market_regime", "state"],
        )
        or _default_regime_state(strategy_candidate),
        "horizon_hours": horizon_hours,
        "decision_ts": _first_timestamp(row, payload, ["decision_ts", "ts_utc", "ts"]),
        "label_ts": _first_timestamp(row, payload, ["label_ts", "outcome_ts", "matured_ts"]),
        "entry_close": _first_numeric(row, payload, ["entry_close", "entry_price", "price"]),
        "label_close": _first_numeric(row, payload, ["label_close", "exit_price", "future_close"]),
        "gross_bps": gross_bps,
        "net_bps_after_cost": net_bps,
        "mfe_bps": _first_numeric(
            row,
            payload,
            _horizon_keys(horizon_hours, ["mfe_bps", "max_favorable_bps"]),
        ),
        "mae_bps": _first_numeric(
            row,
            payload,
            _horizon_keys(horizon_hours, ["mae_bps", "max_adverse_bps", "drawdown_bps"]),
        ),
        "win": win,
        "label_status": label_status,
        "label_reason": _first_text(row, payload, ["label_reason", "outcome_reason", "reason"]),
        "cost_bps": float(cost["cost_bps"] or 0.0),
        "cost_source": str(cost["cost_source"] or "MISSING"),
        "block_reason": _first_text(
            row,
            payload,
            ["block_reason", "blocked_reason", "router_reason", "reason", "skip_reason"],
        ),
        "final_decision": _first_text(
            row,
            payload,
            ["final_decision", "decision", "outcome_decision", "action"],
        ),
        "final_score": _first_numeric(
            row,
            payload,
            ["final_score", "score", "candidate_score", "total_score", "composite_score"],
        ),
        "expected_edge_bps": _first_numeric(
            row,
            payload,
            ["expected_edge_bps", "expected_net_edge_bps", "edge_bps", "gross_edge_bps"],
        ),
        "required_edge_bps": _first_numeric(
            row,
            payload,
            ["required_edge_bps", "min_required_edge_bps", "edge_required_bps"],
        ),
        "alpha6_score": _first_numeric(row, payload, ["alpha6_score", "alpha6"]),
        "alpha6_side": _first_text(row, payload, ["alpha6_side", "side", "direction"]),
        "protect_level": _first_text(
            row,
            payload,
            ["protect_level", "protection_level", "auto_risk_level", "risk_level"],
        ),
        "risk_level": _first_text(row, payload, ["risk_level", "auto_risk_level"]),
        "btc_trend_state": _first_text(
            row,
            payload,
            ["btc_trend_state", "btc_state", "btc_regime"],
        ),
        "broad_market_positive_count": _first_int(
            row,
            payload,
            ["broad_market_positive_count", "positive_count", "breadth_positive_count"],
        ),
        "funding_state": _first_text(row, payload, ["funding_state"]),
        "volatility_bucket": _first_text(row, payload, ["volatility_bucket", "vol_bucket"]),
        "source_path_inside_bundle": str(row.get("source_path_inside_bundle") or ""),
        "source_event_key": source_event_key,
        "source_bundle_ts": _first_timestamp(row, payload, ["bundle_ts", "ingest_ts"]),
        "created_at": datetime.now(UTC),
        "source": SOURCE_NAME,
    }


def _formal_summary_row(
    *,
    strategy_candidate: str,
    symbol: str,
    regime_state: str,
    horizon_hours: int,
    rows: list[dict[str, Any]],
    as_of_date: date,
    created_at: datetime,
    min_live_samples: int,
) -> dict[str, Any]:
    complete = [row for row in rows if str(row.get("label_status") or "") == "complete"]
    sample_count = sum(_sample_weight(row) for row in rows)
    complete_sample_count = sum(_complete_weight(row) for row in complete)
    weighted_net_values = _weighted_values(complete, "net_bps_after_cost")
    wins = [
        (bool(row.get("win")), _complete_weight(row))
        for row in complete
        if row.get("win") is not None and _complete_weight(row) > 0
    ]
    cost_mix = _cost_source_mix(rows)
    avg_net = _weighted_mean(weighted_net_values)
    median_net = _weighted_quantile(weighted_net_values, 0.5)
    p25_net = _weighted_quantile(weighted_net_values, 0.25)
    win_rate = (
        sum(weight for won, weight in wins if won) / sum(weight for _, weight in wins)
        if wins
        else None
    )
    decision, reasons = _formal_decision(
        candidate_name=strategy_candidate,
        sample_count=sample_count,
        complete_sample_count=complete_sample_count,
        avg_net_bps=avg_net,
        p25_net_bps=p25_net,
        win_rate=win_rate,
        min_live_samples=min_live_samples,
        cost_source_mix=cost_mix,
    )
    timestamps = [
        value
        for value in (_parse_timestamp(row.get("ts_utc")) for row in rows)
        if value is not None
    ]
    return {
        "strategy": "v5",
        "evidence_version": EVIDENCE_VERSION,
        "as_of_date": as_of_date.isoformat(),
        "strategy_candidate": strategy_candidate,
        "candidate_name": strategy_candidate,
        "symbol": symbol,
        "regime_state": regime_state,
        "horizon_hours": horizon_hours,
        "sample_count": sample_count,
        "complete_sample_count": complete_sample_count,
        "avg_net_bps": avg_net,
        "median_net_bps": median_net,
        "p25_net_bps": p25_net,
        "win_rate": win_rate,
        "cost_source_mix": _json(cost_mix),
        "decision": decision,
        "decision_reasons": _json(reasons),
        "start_ts": min(timestamps) if timestamps else None,
        "end_ts": max(timestamps) if timestamps else None,
        "created_at": created_at,
        "source": SOURCE_NAME,
    }


def _formal_decision(
    *,
    candidate_name: str,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    p25_net_bps: float | None,
    win_rate: float | None,
    min_live_samples: int,
    paper_days: int = 0,
    entry_day_count: int = 0,
    paper_pnl_observed_count: int = 0,
    paper_pnl_day_count: int = 0,
    arrival_mid_coverage: float = 0.0,
    cost_source_mix: Any = None,
) -> tuple[str, list[str]]:
    return strategy_evidence_decision_ladder(
        sample_count=sample_count,
        complete_sample_count=complete_sample_count,
        avg_net_bps=avg_net_bps,
        p25_net_bps=p25_net_bps,
        win_rate=win_rate,
        paper_days=paper_days,
        entry_day_count=entry_day_count,
        paper_pnl_observed_count=paper_pnl_observed_count,
        paper_pnl_day_count=paper_pnl_day_count,
        arrival_mid_coverage=arrival_mid_coverage,
        cost_source_mix=cost_source_mix,
        paper_ready_sample_count=min_live_samples,
        candidate_name=candidate_name,
    )


def strategy_evidence_decision_ladder(
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    p25_net_bps: float | None,
    win_rate: float | None,
    paper_days: int = 0,
    entry_day_count: int = 0,
    paper_pnl_observed_count: int = 0,
    paper_pnl_day_count: int = 0,
    required_entry_day_count: int = MIN_LIVE_SMALL_READY_ENTRY_DAYS,
    arrival_mid_coverage: float = 0.0,
    paper_slippage_coverage: float = 0.0,
    cost_source_mix: Any = None,
    paper_ready_sample_count: int = MIN_LIVE_SMALL_READY_SAMPLES,
    candidate_name: str | None = None,
) -> tuple[str, list[str]]:
    if _is_alt_impulse_shadow_candidate(candidate_name):
        return _alt_impulse_regime_shadow_decision(
            sample_count=sample_count,
            complete_sample_count=complete_sample_count,
            avg_net_bps=avg_net_bps,
            win_rate=win_rate,
        )

    reasons: list[str] = []
    if sample_count < 10:
        reasons.append("insufficient_total_samples")
    if complete_sample_count < 5:
        reasons.append("insufficient_complete_samples")
    if reasons:
        return "RESEARCH_ONLY", reasons

    if complete_sample_count >= 10 and _is_negative_after_cost_edge(avg_net_bps):
        if win_rate is not None and win_rate < 0.45:
            return "KILL", ["non_positive_after_cost_edge", "win_rate_below_threshold"]

    paper_ready = (
        complete_sample_count >= paper_ready_sample_count
        and win_rate is not None
        and win_rate > 0.55
        and p25_net_bps is not None
        and p25_net_bps > -50.0
    )
    has_global_default_cost = _cost_source_mix_contains_global_default(cost_source_mix)
    has_live_blocking_cost = _cost_source_mix_contains_any(
        cost_source_mix, LIVE_BLOCKING_COST_SOURCES
    )
    has_live_ready_cost = _cost_source_mix_contains_any(cost_source_mix, LIVE_READY_COST_SOURCES)
    if (
        paper_ready
        and complete_sample_count >= 60
        and entry_day_count >= required_entry_day_count
        and paper_pnl_observed_count > 0
        and paper_pnl_day_count >= 14
        and arrival_mid_coverage >= 0.8
        and paper_slippage_coverage >= 0.8
        and has_live_ready_cost
        and not has_live_blocking_cost
    ):
        return "LIVE_SMALL_READY", ["live_small_ready_thresholds_met"]
    if paper_ready and has_global_default_cost:
        return "KEEP_SHADOW", ["paper_ready_thresholds_met", "cost_source_not_trusted"]
    if paper_ready:
        reasons = ["paper_ready_thresholds_met"]
        if has_live_blocking_cost or not has_live_ready_cost:
            reasons.append("cost_source_not_trusted")
        if paper_days < 14:
            reasons.append("heartbeat_days_not_effective_paper_pnl")
        if entry_day_count < required_entry_day_count:
            reasons.append("insufficient_entry_days")
        if paper_pnl_observed_count <= 0:
            reasons.append("no_paper_pnl_observations")
        if paper_pnl_day_count < 14:
            reasons.append("insufficient_paper_pnl_days")
        if arrival_mid_coverage < 0.8:
            reasons.append("insufficient_arrival_mid_coverage")
        if paper_slippage_coverage < 0.8:
            reasons.append("no_live_slippage_coverage")
        return "PAPER_READY", reasons
    if complete_sample_count >= 10 and avg_net_bps is not None and avg_net_bps > 0:
        return "KEEP_SHADOW", ["positive_avg_net_bps_needs_more_evidence"]

    if _is_negative_after_cost_edge(avg_net_bps):
        reasons.append("non_positive_after_cost_edge")
    if win_rate is None or win_rate < 0.45:
        reasons.append("win_rate_below_threshold")
    return "RESEARCH_ONLY", reasons or ["evidence_inconclusive"]


def _is_alt_impulse_shadow_candidate(candidate_name: str | None) -> bool:
    return (
        _canonical_candidate_name(candidate_name, dataset_name="strategy_evidence")
        == ALT_IMPULSE_SHADOW_CANDIDATE
    )


def _alt_impulse_regime_shadow_decision(
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    win_rate: float | None,
) -> tuple[str, list[str]]:
    if complete_sample_count >= 10 and _is_negative_after_cost_edge(avg_net_bps):
        if win_rate is not None and win_rate < 0.45:
            return "KEEP_SHADOW", [
                "alt_impulse_regime_shadow_only",
                "live_disabled",
                "negative_regime_net_edge",
                "win_rate_below_threshold",
            ]

    if (
        complete_sample_count >= ALT_IMPULSE_REGIME_SHADOW_MIN_COMPLETE_SAMPLES
        and avg_net_bps is not None
        and avg_net_bps > 0.0
        and (
            win_rate is None
            or win_rate >= ALT_IMPULSE_REGIME_SHADOW_MIN_WIN_RATE
        )
    ):
        return "REGIME_SHADOW", [
            "alt_impulse_regime_shadow_only",
            "positive_regime_net_edge",
            "live_disabled",
        ]

    reasons = ["alt_impulse_regime_shadow_only", "live_disabled"]
    if sample_count < 10:
        reasons.append("insufficient_total_samples")
    if complete_sample_count < ALT_IMPULSE_REGIME_SHADOW_MIN_COMPLETE_SAMPLES:
        reasons.append("insufficient_regime_complete_samples")
    if avg_net_bps is None:
        reasons.append("avg_net_bps_not_observable")
    elif avg_net_bps <= 0.0:
        reasons.append("regime_net_edge_not_positive")
    if win_rate is not None and win_rate < ALT_IMPULSE_REGIME_SHADOW_MIN_WIN_RATE:
        reasons.append("weak_regime_win_rate")
    return "KEEP_SHADOW", reasons


def _is_negative_after_cost_edge(avg_net_bps: float | None) -> bool:
    return avg_net_bps is not None and avg_net_bps < 0.0


def _cost_source_mix_contains_global_default(cost_source_mix: Any) -> bool:
    return _cost_source_mix_contains_any(cost_source_mix, {"global_default"})


def _cost_source_mix_contains_any(cost_source_mix: Any, blocked_sources: set[str]) -> bool:
    return any(source in blocked_sources for source in _cost_source_mix_sources(cost_source_mix))


def _cost_source_mix_sources(cost_source_mix: Any) -> set[str]:
    if cost_source_mix is None:
        return set()
    if isinstance(cost_source_mix, dict):
        return {_normalize_cost_source(source) for source in cost_source_mix}
    if isinstance(cost_source_mix, list):
        sources: set[str] = set()
        for item in cost_source_mix:
            if isinstance(item, dict):
                source = item.get("cost_source") or item.get("source")
            else:
                source = item
            sources.add(_normalize_cost_source(source))
        return sources
    if isinstance(cost_source_mix, str):
        try:
            parsed = json.loads(cost_source_mix)
        except json.JSONDecodeError:
            return {_normalize_cost_source(cost_source_mix)}
        return _cost_source_mix_sources(parsed)
    return {_normalize_cost_source(cost_source_mix)}


def _normalize_cost_source(source: Any) -> str:
    return str(source or "").strip().lower()


def _cost_source_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("cost_source") or "MISSING")
        counts[key] = counts.get(key, 0) + _sample_weight(row)
    return counts


def _sample_weight(row: dict[str, Any]) -> int:
    return max(int(_finite_float(row.get("sample_count")) or 1), 1)


def _complete_weight(row: dict[str, Any]) -> int:
    if str(row.get("label_status") or "") != "complete":
        return 0
    return max(int(_finite_float(row.get("complete_sample_count")) or 1), 1)


def _weighted_values(rows: list[dict[str, Any]], column: str) -> list[tuple[float, int]]:
    values: list[tuple[float, int]] = []
    for row in rows:
        value = _finite_float(row.get(column))
        weight = _complete_weight(row)
        if value is not None and weight > 0:
            values.append((value, weight))
    return values


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    total_weight = sum(weight for _, weight in values)
    if not total_weight:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def _weighted_quantile(values: list[tuple[float, int]], q: float) -> float | None:
    total_weight = sum(weight for _, weight in values)
    if not total_weight:
        return None
    threshold = total_weight * q
    cumulative = 0
    for value, weight in sorted(values, key=lambda item: item[0]):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return values[-1][0] if values else None


def _formal_samples_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=SAMPLE_SCHEMA)
    return pl.DataFrame(rows, schema=SAMPLE_SCHEMA, orient="row")


def _dedupe_formal_sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            candidate_id = "|".join(
                [
                    str(row.get("ts_utc") or ""),
                    str(row.get("symbol") or ""),
                    str(row.get("strategy_candidate") or ""),
                    str(row.get("source_event_key") or ""),
                ]
            )
        key = (candidate_id, int(row.get("horizon_hours") or 0))
        keyed[key] = row
    return list(keyed.values())


def _dedupe_outcome_source_rows(
    dataset_name: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        payload = _payload(row)
        key = _outcome_source_key(dataset_name, row, payload)
        current = keyed.get(key)
        if current is None or _source_row_seen_time(row) >= _source_row_seen_time(current):
            keyed[key] = row
    return list(keyed.values())


def _outcome_source_key(
    dataset_name: str,
    row: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[str, ...]:
    explicit = _first_text(row, payload, ["event_id", "event_key", "candidate_id", "id"])
    path = str(row.get("source_path_inside_bundle") or "")
    candidate = _candidate_name(dataset_name, row, payload) or ""
    symbol = normalize_symbol(
        _first_text(
            row,
            payload,
            [
                "symbol",
                "normalized_symbol",
                "candidate_symbol",
                "target_symbol",
                "base_symbol",
                "inst_id",
            ],
        )
    )
    if explicit:
        return (dataset_name, explicit, candidate, symbol)
    ts = _first_timestamp(
        row,
        payload,
        ["ts_utc", "decision_ts", "event_ts", "label_ts", "created_at", "timestamp", "time", "ts"],
    )
    row_index = str(row.get("row_index") or "")
    run_id = str(row.get("run_id") or "")
    return (
        dataset_name,
        path,
        run_id,
        _iso(ts),
        symbol,
        candidate,
        row_index,
    )


def _source_row_seen_time(row: dict[str, Any]) -> datetime:
    for key in ["bundle_ts", "ingest_ts", "source_bundle_ts", "ts_utc", "created_at"]:
        value = _parse_timestamp(row.get(key))
        if value is not None:
            return value
    return datetime.min.replace(tzinfo=UTC)


def _unknown_symbol_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if _is_unknown_symbol(row.get("symbol")))


def _drop_unknown_symbol_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not _is_unknown_symbol(row.get("symbol"))]


def _is_unknown_symbol(value: Any) -> bool:
    symbol = normalize_symbol(value)
    return not symbol or symbol == "UNKNOWN"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sample_from_telemetry_row(
    dataset_name: str,
    row: dict[str, Any],
    cost_context: _CostContext,
) -> dict[str, Any] | None:
    payload = _payload(row)
    candidate_name = _candidate_name(dataset_name, row, payload)
    if candidate_name is None:
        return None

    ts_utc = _first_timestamp(
        row,
        payload,
        [
            "ts_utc",
            "decision_ts",
            "event_ts",
            "created_at",
            "timestamp",
            "time",
            "ts",
            "bundle_ts",
            "ingest_ts",
        ],
    )
    if ts_utc is None:
        return None

    symbol = _symbol(row, payload, candidate_name)
    entry_side = _first_text(
        row,
        payload,
        ["entry_condition_side", "side", "alpha6_side", "direction", "intent", "action"],
    )
    cost = cost_context.for_sample(
        symbol=symbol,
        ts_utc=ts_utc,
        row=row,
        payload=payload,
    )
    expected_edge_bps = _first_numeric(
        row,
        payload,
        ["expected_edge_bps", "expected_net_edge_bps", "edge_bps", "gross_edge_bps"],
    )
    required_edge_bps = _first_numeric(
        row,
        payload,
        ["required_edge_bps", "min_required_edge_bps", "edge_required_bps"],
    )
    if required_edge_bps is None:
        required_edge_bps = cost["cost_bps"]

    return _schema_row(
        {
            "ts_utc": ts_utc,
            "symbol": symbol,
            "candidate_name": candidate_name,
            "entry_condition_name": _first_text(
                row,
                payload,
                ["entry_condition_name", "entry_condition", "condition_name"],
            ),
            "entry_condition_side": entry_side,
            "entry_condition_signal": _first_text(
                row,
                payload,
                ["entry_condition_signal", "signal", "router_intent", "intent", "action"],
            ),
            "entry_condition_passed": _first_bool(
                row,
                payload,
                [
                    "entry_condition_passed",
                    "entry_passed",
                    "candidate_passed",
                    "allowed",
                    "selected",
                ],
            ),
            "entry_conditions_json": safe_json_dumps(_entry_conditions(row, payload)),
            "block_reason": _first_text(
                row,
                payload,
                [
                    "block_reason",
                    "blocked_reason",
                    "router_reason",
                    "decision_reason",
                    "reason",
                    "skip_reason",
                    "protect_reason",
                ],
            ),
            "final_score": _first_numeric(
                row,
                payload,
                ["final_score", "score", "candidate_score", "total_score", "composite_score"],
            ),
            "f1": _first_numeric(row, payload, ["f1", "f1_score"]),
            "f2": _first_numeric(row, payload, ["f2", "f2_score"]),
            "f3": _first_numeric(row, payload, ["f3", "f3_score"]),
            "f4": _first_numeric(row, payload, ["f4", "f4_score"]),
            "f5": _first_numeric(row, payload, ["f5", "f5_score"]),
            "alpha6_score": _first_numeric(row, payload, ["alpha6_score", "alpha6"]),
            "alpha6_side": _first_text(row, payload, ["alpha6_side", "side", "direction"]),
            "regime_state": _first_text(
                row,
                payload,
                ["regime_state", "regime", "market_regime", "state"],
            ),
            "protect_level": _first_text(
                row,
                payload,
                ["protect_level", "protection_level", "auto_risk_level", "risk_level"],
            ),
            "expected_edge_bps": expected_edge_bps,
            "required_edge_bps": required_edge_bps,
            "cost_source": cost["cost_source"],
            "cost_bps": cost["cost_bps"],
            "source_dataset": dataset_name,
            "source_path_inside_bundle": str(row.get("source_path_inside_bundle") or ""),
            "source_event_key": _source_event_key(dataset_name, row, payload),
            "source_bundle_ts": _first_timestamp(row, payload, ["bundle_ts", "ingest_ts"]),
            "created_at": datetime.now(UTC),
            "source": SOURCE_NAME,
        }
    )


def _candidate_name(
    dataset_name: str,
    row: dict[str, Any],
    payload: dict[str, Any],
) -> str | None:
    explicit = _first_text(
        row,
        payload,
        [
            "candidate_name",
            "strategy_candidate",
            "candidate",
            "strategy_id",
            "strategy_name",
            "alpha_id",
            "probe_name",
            "probe_type",
            "entry_rule",
        ],
    )
    if explicit:
        explicit_candidate = _normalize_candidate_text(explicit, dataset_name=dataset_name)
        if explicit_candidate is not None:
            return explicit_candidate

    path = str(row.get("source_path_inside_bundle") or "").lower()
    if "multi_position_swing" in path:
        k_value = _first_text(row, payload, ["k", "position_k", "slot", "position_slot"])
        if k_value in {"1", "k1"}:
            return "v5.multi_position_k1"
        if k_value in {"2", "k2"}:
            return "v5.multi_position_k2"
        if k_value in {"3", "k3"}:
            return "v5.multi_position_k3"

    text = _row_search_text(row, payload)
    return _normalize_candidate_text(text, dataset_name=dataset_name)


def _canonical_candidate_name(value: Any, *, dataset_name: str) -> str:
    raw = str(value or "").strip()
    return _normalize_candidate_text(raw, dataset_name=dataset_name) or raw


def _normalize_candidate_text(value: str | None, *, dataset_name: str) -> str | None:
    text = (value or "").strip().lower()
    if not text and dataset_name == "v5_shadow_outcome":
        return "v5.alt_impulse_shadow"
    normalized = (
        text.replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )
    direct = {
        "v5_btc_leadership_probe_strict": "v5.btc_leadership_probe_strict",
        "btc_leadership_probe_strict": "v5.btc_leadership_probe_strict",
        "btc_strict_probe": "v5.btc_leadership_probe_strict",
        "strict_btc_leadership_probe": "v5.btc_leadership_probe_strict",
        "btc_leadership_probe_blocked_outcomes": "v5.btc_leadership_probe_strict",
        "btc_leadership_blocked_relaxed": "v5.btc_leadership_blocked_relaxed",
        "btc_leadership_relaxed_blocker": "v5.btc_leadership_blocked_relaxed",
        "btc_relaxed_blocker": "v5.btc_leadership_blocked_relaxed",
        "btc_leadership_alpha6_low_blocked": "v5.btc_leadership_alpha6_low_blocked",
        "btc_leadership_f5_low_blocked": "v5.btc_leadership_f5_low_blocked",
        "btc_leadership_no_breakout_blocked": "v5.btc_leadership_no_breakout_blocked",
        "v5_sol_protect_exception": "v5.sol_protect_exception",
        "sol_protect_exception": "v5.sol_protect_exception",
        "sol_protect_alpha6_low_exception": "v5.sol_protect_alpha6_low_exception",
        "sol_protect_rsi_weak_exception": "v5.sol_protect_rsi_weak_exception",
        "v5_alt_impulse_shadow": "v5.alt_impulse_shadow",
        "alt_impulse_shadow": "v5.alt_impulse_shadow",
        "alt_impulse_shadow_outcomes": "v5.alt_impulse_shadow",
        "alt_impulse": "v5.alt_impulse_shadow",
        "multi_position_k1": "v5.multi_position_k1",
        "multi_position_k2": "v5.multi_position_k2",
        "multi_position_k3": "v5.multi_position_k3",
        "multi_position_swing_k1": "v5.multi_position_k1",
        "multi_position_swing_k2": "v5.multi_position_k2",
        "multi_position_swing_k3": "v5.multi_position_k3",
        "v5_swing_f4_f5_alpha6": "v5.swing_f4_f5_alpha6",
        "swing_f4_f5_alpha6": "v5.swing_f4_f5_alpha6",
        "v5_f3_dominant_entry": "v5.f3_dominant_entry",
        "f3_dominant_entry": "v5.f3_dominant_entry",
        "f3_dominant": "v5.f3_dominant_entry",
        "v5_f4_volume_expansion_entry": "v5.f4_volume_expansion_entry",
        "f4_volume_expansion_entry": "v5.f4_volume_expansion_entry",
        "f4_volume_expansion": "v5.f4_volume_expansion_entry",
        "v5_mean_reversion_sideways": "v5.mean_reversion_sideways",
        "mean_reversion_sideways": "v5.mean_reversion_sideways",
    }
    if normalized in direct:
        return direct[normalized]
    if dataset_name == "v5_shadow_outcome" and "alt_impulse" in normalized:
        return "v5.alt_impulse_shadow"
    if "btc" in normalized and "leadership" in normalized:
        if "alpha6" in normalized and "low" in normalized:
            return "v5.btc_leadership_alpha6_low_blocked"
        if "f5" in normalized and "low" in normalized:
            return "v5.btc_leadership_f5_low_blocked"
        if "no_breakout" in normalized or ("no" in normalized and "breakout" in normalized):
            return "v5.btc_leadership_no_breakout_blocked"
        if "probe" not in normalized and "block" not in normalized:
            return None
        if "strict" in normalized:
            return "v5.btc_leadership_probe_strict"
        if "relaxed" in normalized or "blocker" in normalized:
            return "v5.btc_leadership_blocked_relaxed"
        return None
    if "sol" in normalized and "protect" in normalized and "exception" in normalized:
        if "alpha6" in normalized and "low" in normalized:
            return "v5.sol_protect_alpha6_low_exception"
        if "rsi" in normalized and "weak" in normalized:
            return "v5.sol_protect_rsi_weak_exception"
        return "v5.sol_protect_exception"
    if "alt" in normalized and "impulse" in normalized:
        return "v5.alt_impulse_shadow"
    if "swing" in normalized and "f4" in normalized and "f5" in normalized:
        return "v5.swing_f4_f5_alpha6"
    if "multi" in normalized and "position" in normalized:
        if "k3" in normalized or re.search(r"(?:^|_)k_+3(?:_|$)", normalized):
            return "v5.multi_position_k3"
        if "k2" in normalized or re.search(r"(?:^|_)k_+2(?:_|$)", normalized):
            return "v5.multi_position_k2"
        return "v5.multi_position_k1"
    if "f3" in normalized and "dominant" in normalized:
        return "v5.f3_dominant_entry"
    if "f4" in normalized and ("volume" in normalized or "expansion" in normalized):
        return "v5.f4_volume_expansion_entry"
    if "mean" in normalized and "reversion" in normalized and "sideways" in normalized:
        return "v5.mean_reversion_sideways"
    return None


class _CostContext:
    def __init__(self, lake_root: Path) -> None:
        self.usage_rows = _cost_usage_rows(
            read_parquet_dataset(lake_root / "silver" / "v5_quant_lab_cost_usage")
        )
        self.bucket_rows = _cost_bucket_rows(
            read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily")
        )

    def for_sample(
        self,
        *,
        symbol: str,
        ts_utc: datetime,
        row: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        row_cost_bps = _first_numeric(
            row,
            payload,
            [
                "cost_bps",
                "total_cost_bps",
                "selected_total_cost_bps",
                "estimated_cost_bps",
                "required_edge_bps",
            ],
        )
        row_cost_source = _first_text(
            row,
            payload,
            ["cost_source", "source", "cost_model_source", "fallback_level"],
        )
        if row_cost_bps is not None:
            return {
                "cost_bps": max(row_cost_bps, 0.0),
                "cost_source": row_cost_source or "telemetry_row",
            }

        usage = _latest_cost_row(self.usage_rows.get(symbol, []), ts_utc)
        if usage is not None:
            return usage
        bucket = self.bucket_rows.get(symbol) or self.bucket_rows.get("GLOBAL")
        if bucket is not None:
            return bucket
        return {"cost_bps": DEFAULT_RESEARCH_COST_BPS, "cost_source": "global_default"}


def _cost_usage_rows(frame: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    if frame.is_empty():
        return rows
    for row in frame.to_dicts():
        payload = _payload(row)
        symbol = normalize_symbol(
            _first_text(row, payload, ["symbol", "normalized_symbol", "inst_id"])
        )
        if not symbol:
            continue
        cost_bps = _first_numeric(
            row,
            payload,
            ["cost_bps", "total_cost_bps", "selected_total_cost_bps", "estimated_cost_bps"],
        )
        if cost_bps is None:
            continue
        ts = _first_timestamp(row, payload, ["ts_utc", "ts", "created_at", "bundle_ts"])
        rows.setdefault(symbol, []).append(
            {
                "ts_utc": ts or datetime.min.replace(tzinfo=UTC),
                "cost_bps": max(cost_bps, 0.0),
                "cost_source": _first_text(row, payload, ["cost_source", "source"])
                or "v5_quant_lab_cost_usage",
            }
        )
    for symbol_rows in rows.values():
        symbol_rows.sort(key=lambda item: item["ts_utc"])
    return rows


def _cost_bucket_rows(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if frame.is_empty():
        return rows
    sort_column = "day" if "day" in frame.columns else frame.columns[0]
    for row in frame.sort(sort_column).to_dicts():
        symbol = normalize_symbol(row.get("symbol")) or "GLOBAL"
        cost_bps = _first_existing_numeric(
            row,
            ["total_cost_bps_p75", "total_cost_bps", "cost_bps", "selected_total_cost_bps"],
        )
        if cost_bps is None:
            continue
        rows[symbol] = {
            "cost_bps": max(cost_bps, 0.0),
            "cost_source": str(row.get("cost_source") or row.get("source") or "cost_bucket_daily"),
        }
    return rows


def _latest_cost_row(rows: list[dict[str, Any]], ts_utc: datetime) -> dict[str, Any] | None:
    if not rows:
        return None
    selected = rows[0]
    for row in rows:
        if row["ts_utc"] <= ts_utc:
            selected = row
        else:
            break
    return {"cost_bps": selected["cost_bps"], "cost_source": selected["cost_source"]}


def _attach_forward_labels(samples: pl.DataFrame, market_bars: pl.DataFrame) -> pl.DataFrame:
    bars_by_symbol = _bars_by_symbol(market_bars)
    rows: list[dict[str, Any]] = []
    for row in samples.to_dicts():
        labeled = dict(row)
        bars = bars_by_symbol.get(str(row.get("symbol") or ""))
        if not bars:
            rows.append(_with_empty_labels(labeled))
            continue
        rows.append(_label_sample(labeled, bars))
    return _samples_frame(rows)


def _bars_by_symbol(market_bars: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if market_bars.is_empty():
        return {}
    required = {"symbol", "ts", "close", "high", "low"}
    if not required.issubset(market_bars.columns):
        return {}
    bars = market_bars
    if "is_closed" in bars.columns:
        bars = bars.filter(pl.col("is_closed"))
    bars = bars.with_columns(
        [
            pl.col("symbol").map_elements(normalize_symbol, return_dtype=pl.Utf8),
            _datetime_expr(bars, "ts"),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
        ]
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for symbol, group in bars.sort(["symbol", "ts"]).group_by("symbol", maintain_order=True):
        key = str(symbol[0] if isinstance(symbol, tuple) else symbol)
        grouped[key] = group.to_dicts()
    return grouped


def _label_sample(row: dict[str, Any], bars: list[dict[str, Any]]) -> dict[str, Any]:
    ts_values = [bar["ts"] for bar in bars]
    signal_ts = _ensure_utc(row["ts_utc"])
    decision_index = bisect.bisect_right(ts_values, signal_ts)
    if decision_index >= len(bars):
        return _with_empty_labels(row)

    decision_bar = bars[decision_index]
    decision_ts = _ensure_utc(decision_bar["ts"])
    decision_close = _finite_float(decision_bar.get("close"))
    if decision_close is None or decision_close <= 0:
        return _with_empty_labels(row)

    row["decision_ts"] = decision_ts
    complete = True
    for hours in HORIZON_HOURS:
        target_ts = decision_ts + timedelta(hours=hours)
        future_index = bisect.bisect_left(ts_values, target_ts)
        if future_index >= len(bars):
            complete = False
            _set_empty_horizon(row, hours)
            continue
        future_bar = bars[future_index]
        future_close = _finite_float(future_bar.get("close"))
        if future_close is None or future_close <= 0:
            complete = False
            _set_empty_horizon(row, hours)
            continue
        gross_bps = _directional_return_bps(row, decision_close, future_close)
        cost_bps = _finite_float(row.get("cost_bps")) or 0.0
        net_bps = gross_bps - cost_bps
        row[f"gross_bps_{hours}h"] = gross_bps
        row[f"net_bps_after_cost_{hours}h"] = net_bps
        row[f"win_{hours}h"] = net_bps > 0.0
        row[f"drawdown_proxy_bps_{hours}h"] = _drawdown_proxy_bps(
            row,
            bars[decision_index : future_index + 1],
            decision_close,
            cost_bps,
        )
    row["label_status"] = "complete" if complete else "partial"
    return row


def _directional_return_bps(
    row: dict[str, Any],
    decision_close: float,
    future_close: float,
) -> float:
    side = str(
        row.get("alpha6_side")
        or row.get("entry_condition_side")
        or row.get("entry_condition_signal")
        or ""
    ).lower()
    if side in {"short", "sell", "down", "bear", "negative"}:
        return ((decision_close / future_close) - 1.0) * 10_000.0
    return ((future_close / decision_close) - 1.0) * 10_000.0


def _drawdown_proxy_bps(
    row: dict[str, Any],
    path: list[dict[str, Any]],
    decision_close: float,
    cost_bps: float,
) -> float:
    side = str(
        row.get("alpha6_side")
        or row.get("entry_condition_side")
        or row.get("entry_condition_signal")
        or ""
    ).lower()
    if side in {"short", "sell", "down", "bear", "negative"}:
        highs = [_finite_float(bar.get("high")) for bar in path]
        high = max(value for value in highs if value is not None)
        return max(((high / decision_close) - 1.0) * 10_000.0, 0.0) + cost_bps
    lows = [_finite_float(bar.get("low")) for bar in path]
    low = min(value for value in lows if value is not None)
    return max(((decision_close - low) / decision_close) * 10_000.0, 0.0) + cost_bps


def _summary_row(
    candidate_name: str,
    rows: list[dict[str, Any]],
    *,
    as_of_date: date,
    created_at: datetime,
    min_live_samples: int,
) -> dict[str, Any]:
    sample_count = len(rows)
    complete_rows = [row for row in rows if row.get("label_status") == "complete"]
    complete_sample_count = len(complete_rows)
    avg_net = _horizon_stat(rows, "net_bps_after_cost", _mean)
    median_net = _horizon_stat(rows, "net_bps_after_cost", _median)
    win_rates = _horizon_win_rate(rows)
    downside = _horizon_stat(rows, "net_bps_after_cost", _p25)
    max_drawdown = _max_drawdown(rows)
    decision, reasons = _candidate_decision(
        candidate_name,
        rows,
        sample_count=sample_count,
        complete_sample_count=complete_sample_count,
        avg_net=avg_net,
        win_rates=win_rates,
        downside=downside,
        max_drawdown=max_drawdown,
        min_live_samples=min_live_samples,
    )
    ts_values = [
        _ensure_utc(row["ts_utc"]) for row in rows if isinstance(row.get("ts_utc"), datetime)
    ]
    return {
        "candidate_name": candidate_name,
        "evidence_version": EVIDENCE_VERSION,
        "as_of_date": as_of_date.isoformat(),
        "sample_count": sample_count,
        "complete_sample_count": complete_sample_count,
        "avg_net_bps_by_horizon": _json(avg_net),
        "median_net_bps_by_horizon": _json(median_net),
        "win_rate_by_horizon": _json(win_rates),
        "downside_p25_by_horizon": _json(downside),
        "max_drawdown_proxy": max_drawdown,
        "cost_sensitivity": _json(_cost_sensitivity(rows)),
        "symbol_breakdown": _json(_breakdown(rows, "symbol")),
        "regime_breakdown": _json(_breakdown(rows, "regime_state")),
        "decision": decision,
        "decision_reasons": _json(reasons),
        "start_ts": min(ts_values) if ts_values else None,
        "end_ts": max(ts_values) if ts_values else None,
        "created_at": created_at,
        "source": SOURCE_NAME,
    }


def _candidate_decision(
    candidate_name: str,
    rows: list[dict[str, Any]],
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net: dict[str, float | None],
    win_rates: dict[str, float | None],
    downside: dict[str, float | None],
    max_drawdown: float | None,
    min_live_samples: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if sample_count == 0:
        return "KEEP_SHADOW", ["no_candidate_samples"]
    if sample_count < min_live_samples:
        reasons.append(f"sample_count_below_{min_live_samples}")

    avg_24h = avg_net.get("24h")
    avg_72h = avg_net.get("72h")
    win_24h = win_rates.get("24h")
    downside_24h = downside.get("24h")
    edge_cost_ratio = _cost_sensitivity(rows).get("expected_edge_to_cost_ratio")
    cost_source_mix = _cost_source_mix(rows)
    has_global_default_cost = _cost_source_mix_contains_global_default(cost_source_mix)
    has_live_blocking_cost = _cost_source_mix_contains_any(
        cost_source_mix, LIVE_BLOCKING_COST_SOURCES
    )
    has_live_ready_cost = _cost_source_mix_contains_any(cost_source_mix, LIVE_READY_COST_SOURCES)
    bad_24h = avg_24h is not None and avg_24h < -5.0
    bad_72h = avg_72h is not None and avg_72h < -5.0
    weak_win = win_24h is not None and win_24h < 0.42
    if complete_sample_count >= 10 and (bad_24h or bad_72h or weak_win):
        if bad_24h:
            reasons.append("negative_24h_net_edge")
        if bad_72h:
            reasons.append("negative_72h_net_edge")
        if weak_win:
            reasons.append("weak_24h_win_rate")
        return "KILL", reasons

    if candidate_name == ALT_IMPULSE_SHADOW_CANDIDATE:
        return _alt_impulse_regime_shadow_decision(
            sample_count=sample_count,
            complete_sample_count=complete_sample_count,
            avg_net_bps=avg_24h,
            win_rate=win_24h,
        )

    if candidate_name in {
        "v5.sol_protect_exception",
        "v5.sol_protect_alpha6_low_exception",
        "v5.sol_protect_rsi_weak_exception",
    }:
        reasons.append("candidate_is_shadow_or_protect_exception")
        return "KEEP_SHADOW", reasons

    if sample_count < min_live_samples or complete_sample_count < min_live_samples:
        if complete_sample_count < min_live_samples:
            reasons.append(f"complete_sample_count_below_{min_live_samples}")
        return "KEEP_SHADOW", reasons

    live_ready = (
        (avg_24h is not None and avg_24h > 0.0)
        and (avg_72h is not None and avg_72h > 0.0)
        and (win_24h is not None and win_24h >= 0.55)
        and (downside_24h is not None and downside_24h > -50.0)
        and (max_drawdown is None or max_drawdown <= 250.0)
        and (edge_cost_ratio is None or edge_cost_ratio >= 1.5)
    )
    if live_ready:
        if has_global_default_cost:
            return "KEEP_SHADOW", [
                "positive_net_edge_and_sample_floor_met",
                "cost_source_not_trusted",
            ]
        if has_live_blocking_cost or not has_live_ready_cost:
            return "PAPER_READY", [
                "positive_24h_net_edge_needs_paper_confirmation",
                "cost_source_not_trusted",
            ]
        return "PAPER_READY", [
            "positive_24h_net_edge_needs_paper_confirmation",
            "no_live_slippage_coverage",
        ]

    paper_ready = (
        (avg_24h is not None and avg_24h > 0.0)
        and (win_24h is not None and win_24h >= 0.50)
        and (downside_24h is None or downside_24h > -100.0)
    )
    if paper_ready:
        if has_global_default_cost:
            return "KEEP_SHADOW", [
                "positive_24h_net_edge_needs_paper_confirmation",
                "cost_source_not_trusted",
            ]
        return "PAPER_READY", ["positive_24h_net_edge_needs_paper_confirmation"]

    reasons.append("evidence_not_strong_enough_for_paper")
    return "KEEP_SHADOW", reasons


def _cost_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "avg_cost_bps": None,
            "avg_expected_edge_bps": None,
            "expected_edge_to_cost_ratio": None,
            "avg_net_bps_24h_double_cost": None,
            "proxy_or_default_cost_ratio": None,
        }
    cost_values = _values(rows, "cost_bps")
    expected_values = _values(rows, "expected_edge_bps")
    net_24h = _values(rows, "net_bps_after_cost_24h")
    avg_cost = _mean(cost_values)
    avg_expected = _mean(expected_values)
    double_cost_net = None
    if net_24h and cost_values:
        double_cost_net = _mean(
            [
                net - cost
                for net, cost in zip(net_24h, cost_values, strict=False)
                if net is not None and cost is not None
            ]
        )
    proxy_rows = [
        row
        for row in rows
        if str(row.get("cost_source") or "").lower()
        in {"global_default", "public_spread_proxy", "proxy", "fallback"}
    ]
    return {
        "avg_cost_bps": avg_cost,
        "avg_expected_edge_bps": avg_expected,
        "expected_edge_to_cost_ratio": (
            None
            if avg_expected is None or avg_cost in {None, 0.0}
            else avg_expected / avg_cost
        ),
        "avg_net_bps_24h_double_cost": double_cost_net,
        "proxy_or_default_cost_ratio": len(proxy_rows) / len(rows),
    }


def _breakdown(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(column) or "unknown")
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        net_24h = _values(group, "net_bps_after_cost_24h")
        wins = [row.get("win_24h") for row in group if row.get("win_24h") is not None]
        output.append(
            {
                column: key,
                "sample_count": len(group),
                "avg_net_bps_24h": _mean(net_24h),
                "win_rate_24h": (
                    None if not wins else sum(1 for value in wins if value) / len(wins)
                ),
            }
        )
    return output


def _horizon_stat(
    rows: list[dict[str, Any]],
    prefix: str,
    fn: Any,
) -> dict[str, float | None]:
    return {
        f"{hours}h": fn(_values(rows, f"{prefix}_{hours}h"))
        for hours in HORIZON_HOURS
    }


def _horizon_win_rate(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    rates: dict[str, float | None] = {}
    for hours in HORIZON_HOURS:
        values = [
            row.get(f"win_{hours}h")
            for row in rows
            if row.get(f"win_{hours}h") is not None
        ]
        rates[f"{hours}h"] = (
            None if not values else sum(1 for value in values if value) / len(values)
        )
    return rates


def _max_drawdown(rows: list[dict[str, Any]]) -> float | None:
    values = [
        value
        for hours in HORIZON_HOURS
        for value in _values(rows, f"drawdown_proxy_bps_{hours}h")
    ]
    return None if not values else max(values)


def _samples_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=SAMPLE_SCHEMA)
    return pl.DataFrame([_schema_row(row) for row in rows], schema=SAMPLE_SCHEMA, orient="row")


def _schema_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for column in SAMPLE_SCHEMA:
        normalized[column] = row.get(column)
    normalized.setdefault("label_status", "unlabeled")
    return normalized


def _with_empty_labels(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["decision_ts"] = row.get("decision_ts")
    row["label_status"] = "unlabeled"
    for hours in HORIZON_HOURS:
        _set_empty_horizon(row, hours)
    return _schema_row(row)


def _set_empty_horizon(row: dict[str, Any], hours: int) -> None:
    row[f"gross_bps_{hours}h"] = None
    row[f"net_bps_after_cost_{hours}h"] = None
    row[f"win_{hours}h"] = None
    row[f"drawdown_proxy_bps_{hours}h"] = None


def _dedupe_sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("candidate_name") or ""),
            str(row.get("symbol") or ""),
            _iso(row.get("ts_utc")),
            str(row.get("source_dataset") or ""),
            str(row.get("source_event_key") or ""),
        )
        current = selected.get(key)
        if current is None or _seen_time(row) >= _seen_time(current):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda item: (
            str(item.get("candidate_name") or ""),
            str(item.get("symbol") or ""),
            _seen_time(item),
        ),
    )


def _filter_samples_as_of(
    rows: list[dict[str, Any]],
    as_of_date: str | None,
) -> list[dict[str, Any]]:
    if as_of_date is None:
        return rows
    day = _as_of_date(as_of_date)
    cutoff = datetime.combine(day + timedelta(days=1), time.min, tzinfo=UTC)
    return [
        row
        for row in rows
        if isinstance(row.get("ts_utc"), datetime) and _ensure_utc(row["ts_utc"]) < cutoff
    ]


def _entry_conditions(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    for container_name in ["entry_conditions", "entry_condition", "conditions"]:
        value = _nested_value(payload, container_name)
        if isinstance(value, dict):
            collected.update(_safe_dict(value))
    for key in [
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "alpha6_score",
        "alpha6_side",
        "regime_state",
        "protect_level",
        "expected_edge_bps",
        "required_edge_bps",
        "entry_condition_passed",
        "router_reason",
        "block_reason",
    ]:
        value = _first_value(row, payload, [key])
        if value is not None:
            collected[key] = value
    return _safe_dict(collected)


def _safe_dict(values: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in values.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ["secret", "passphrase", "private_key", "token"]):
            continue
        if isinstance(value, (dict, list)):
            safe[key] = safe_json_dumps(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _symbol(row: dict[str, Any], payload: dict[str, Any], candidate_name: str) -> str:
    raw = _first_text(
        row,
        payload,
        [
            "symbol",
            "normalized_symbol",
            "candidate_symbol",
            "target_symbol",
            "base_symbol",
            "coin",
            "asset",
            "inst_id",
            "instId",
            "instrument",
            "instrument_id",
            "coin_symbol",
            "asset_symbol",
            "base",
            "pair",
            "ticker",
        ],
    )
    normalized = normalize_symbol(raw)
    if normalized:
        return normalized
    if "btc" in candidate_name:
        return "BTC-USDT"
    if "sol" in candidate_name:
        return "SOL-USDT"
    if "multi_position" in candidate_name:
        return "PORTFOLIO"
    return "UNKNOWN"


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_payload_json")
    if not isinstance(raw, str) or not raw.strip():
        raw = row.get("raw_json")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_value(row: dict[str, Any], payload: dict[str, Any], keys: list[str]) -> Any:
    raw_payload: dict[str, Any] | None = None
    for key in keys:
        value = row.get(key)
        if _present(value):
            return value
        value = _nested_value(payload, key)
        if _present(value):
            return value
        if raw_payload is None:
            raw_payload = _raw_payload(payload)
        value = _nested_value(raw_payload, key)
        if _present(value):
            return value
    return None


def _nested_value(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _raw_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("raw_json")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_text(row: dict[str, Any], payload: dict[str, Any], keys: list[str]) -> str:
    value = _first_value(row, payload, keys)
    if value is None:
        return ""
    rendered = str(value).strip()
    return "" if rendered.lower() in {"none", "null", "nan", "unknown", "n/a"} else rendered


def _first_numeric(row: dict[str, Any], payload: dict[str, Any], keys: list[str]) -> float | None:
    return _finite_float(_first_value(row, payload, keys))


def _first_int(row: dict[str, Any], payload: dict[str, Any], keys: list[str]) -> int | None:
    return _int_or_none(_first_value(row, payload, keys))


def _first_existing_numeric(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _first_bool(row: dict[str, Any], payload: dict[str, Any], keys: list[str]) -> bool | None:
    value = _first_value(row, payload, keys)
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "pass", "passed"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "fail", "failed"}:
        return False
    return None


def _first_timestamp(
    row: dict[str, Any],
    payload: dict[str, Any],
    keys: list[str],
) -> datetime | None:
    for key in keys:
        parsed = _parse_timestamp(_first_value(row, payload, [key]))
        if parsed is not None:
            return parsed
    return None


def _horizon_hours(row: dict[str, Any], payload: dict[str, Any]) -> int | None:
    value = _first_value(
        row,
        payload,
        ["horizon_hours", "horizon_hour", "horizon_h", "horizon", "label_horizon_hours"],
    )
    if value is None:
        return 24
    text = str(value).strip().lower().replace("hours", "h").replace("hour", "h")
    if text.endswith("h"):
        text = text[:-1]
    return _int_or_none(text)


def _outcome_horizons(row: dict[str, Any], payload: dict[str, Any]) -> list[int]:
    keys: set[str] = set(row)
    keys.update(_payload_keys(payload))
    keys.update(_payload_keys(_raw_payload(payload)))
    horizons: set[int] = set()
    for key in keys:
        text = str(key)
        for match in re.finditer(r"(?:^|_)(\d{1,3})h(?:_|$)", text.lower()):
            hours = _int_or_none(match.group(1))
            if hours in HORIZON_HOURS:
                horizons.add(hours)
    if horizons:
        return sorted(horizons)
    direct = _horizon_hours(row, payload)
    return [direct or 24]


def _payload_keys(payload: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                keys.add(str(key))
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(item for item in current if isinstance(item, (dict, list)))
    return keys


def _horizon_keys(horizon_hours: int, base_keys: list[str]) -> list[str]:
    prefixed = []
    for key in base_keys:
        prefixed.extend(
            [
                f"label_{horizon_hours}h_{key}",
                f"{key}_{horizon_hours}h",
                f"avg_{horizon_hours}h_{key}",
                f"{horizon_hours}h_{key}",
            ]
        )
    return prefixed + base_keys


def _outcome_net_bps(
    row: dict[str, Any],
    payload: dict[str, Any],
    *,
    horizon_hours: int | None = None,
) -> float | None:
    keys = [
        "net_bps_after_cost",
        "net_bps",
        "avg_net_bps",
        "net_return_bps",
        "net_pnl_bps",
        "outcome_net_bps",
        "realized_net_bps",
        "after_cost_bps",
        "net_after_cost_bps",
        "profit_bps",
        "pnl_bps",
    ]
    if horizon_hours is not None:
        keys = _horizon_keys(horizon_hours, keys)
    direct = _first_numeric(
        row,
        payload,
        keys,
    )
    if direct is not None:
        return direct
    pct = _first_numeric(
        row,
        payload,
        _horizon_keys(
            horizon_hours,
            ["net_return_pct", "return_pct", "pnl_pct", "profit_pct"],
        )
        if horizon_hours is not None
        else ["net_return_pct", "return_pct", "pnl_pct", "profit_pct"],
    )
    if pct is not None:
        return pct * 10_000.0 if abs(pct) <= 1 else pct * 100.0
    return None


OUTCOME_EVENT_SOURCE_TYPES = {
    "high_score_blocked_outcome",
    "btc_leadership_blocked_outcome",
    "alt_impulse_shadow_outcome",
    "multi_position_swing_shadow_outcome",
    "factor_contribution_outcome",
    "protect_sol_exception_shadow_outcome",
}


def _source_sample_count(
    row: dict[str, Any],
    payload: dict[str, Any],
    *,
    source_type: str,
) -> int:
    if source_type in OUTCOME_EVENT_SOURCE_TYPES:
        return 1
    value = _first_numeric(
        row,
        payload,
        ["sample_count", "count", "rows", "label_count", "candidate_count"],
    )
    return max(int(value or 1), 1)


def _source_complete_sample_count(
    row: dict[str, Any],
    payload: dict[str, Any],
    sample_count: int,
    label_status: str,
    *,
    source_type: str,
    horizon_hours: int | None = None,
) -> int:
    if source_type in OUTCOME_EVENT_SOURCE_TYPES:
        horizon_status = _outcome_completion_status(row, payload, horizon_hours)
        if horizon_status:
            return 1 if horizon_status == "complete" else 0
        return 1 if label_status == "complete" else 0
    value = _first_numeric(
        row,
        payload,
        ["complete_sample_count", "complete_count", "matured_count", "labeled_count"],
    )
    if value is not None:
        return max(int(value), 0)
    return sample_count if label_status == "complete" else 0


def _outcome_completion_status(
    row: dict[str, Any],
    payload: dict[str, Any],
    horizon_hours: int | None,
) -> str:
    if horizon_hours is None:
        return ""
    status = _first_text(
        row,
        payload,
        _horizon_keys(
            horizon_hours,
            [
                "label_status",
                "status",
                "outcome_status",
                "complete_status",
                "completion_status",
            ],
        )
        + [
            f"label_{horizon_hours}h_complete",
            f"{horizon_hours}h_complete",
            f"is_{horizon_hours}h_complete",
            f"complete_{horizon_hours}h",
        ],
    ).lower()
    if status in {"complete", "completed", "matured", "labeled", "true", "1", "yes"}:
        return "complete"
    if status in {"partial", "pending", "incomplete", "unlabeled", "false", "0", "no"}:
        return "incomplete"
    return ""


def _outcome_label_status(
    row: dict[str, Any],
    payload: dict[str, Any],
    horizon_hours: int,
    net_bps: float | None,
) -> str:
    status = _first_text(
        row,
        payload,
        _horizon_keys(horizon_hours, ["label_status", "status", "outcome_status"]),
    ).lower()
    if status in {"complete", "completed", "matured", "labeled"}:
        return "complete"
    if status in {"partial", "pending", "incomplete", "unlabeled"}:
        return status
    return "complete" if net_bps is not None else "unlabeled"


def _outcome_candidate_id(
    *,
    dataset_name: str,
    row: dict[str, Any],
    payload: dict[str, Any],
    source_type: str,
    strategy_candidate: str,
    symbol: str,
    ts_utc: datetime,
) -> str:
    explicit = _first_text(
        row,
        payload,
        ["candidate_id", "event_id", "event_key", "request_id", "id"],
    )
    if explicit:
        return f"{source_type}:{explicit}"
    return "|".join(
        [
            source_type or dataset_name,
            _iso(ts_utc),
            normalize_symbol(symbol),
            strategy_candidate,
        ]
    )


def _source_type(dataset_name: str, row: dict[str, Any], payload: dict[str, Any]) -> str:
    path = str(row.get("source_path_inside_bundle") or "").lower()
    if "high_score_blocked_outcomes" in path:
        return "high_score_blocked_outcome"
    if "btc_leadership_probe_blocked_outcomes" in path:
        return "btc_leadership_blocked_outcome"
    if "alt_impulse_shadow" in path:
        return "alt_impulse_shadow_outcome"
    if "multi_position_swing_shadow" in path:
        return "multi_position_swing_shadow_outcome"
    if "factor_contribution_outcomes_by_factor" in path:
        return "factor_contribution_outcome"
    if "protect_sol_exception_shadow_outcomes" in path:
        return "protect_sol_exception_shadow_outcome"
    if dataset_name == "v5_high_score_blocked_outcome":
        return "high_score_blocked_outcome"
    if dataset_name == "v5_shadow_outcome":
        candidate = _candidate_name(dataset_name, row, payload) or ""
        lowered = candidate.lower()
        if "alt_impulse" in lowered:
            return "alt_impulse_shadow_outcome"
        if "multi_position" in lowered:
            return "multi_position_swing_shadow_outcome"
        if "btc_leadership" in lowered:
            return "btc_leadership_blocked_outcome"
        if "sol_protect" in lowered:
            return "protect_sol_exception_shadow_outcome"
        if "f3_" in lowered or "f4_" in lowered:
            return "factor_contribution_outcome"
        return "shadow_outcome"
    explicit = _first_text(row, payload, ["source_type", "outcome_source_type"])
    return explicit or dataset_name


def _default_regime_state(strategy_candidate: str) -> str:
    lowered = strategy_candidate.lower()
    if "alt_impulse" in lowered:
        return "impulse"
    if "sol_protect" in lowered:
        return "protect"
    if (
        "multi_position" in lowered
        or "btc_leadership" in lowered
        or "f3_" in lowered
        or "f4_" in lowered
    ):
        return "trend"
    return "UNKNOWN"


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=UTC)
    text = str(value).strip()
    if not text:
        return None
    try:
        if len(text) == 10:
            return datetime.combine(date.fromisoformat(text), time.min, tzinfo=UTC)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _ensure_utc(parsed)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _row_search_text(row: dict[str, Any], payload: dict[str, Any]) -> str:
    values: list[str] = []
    for container in [row, payload, _raw_payload(payload)]:
        for value in container.values():
            if isinstance(value, (str, int, float, bool)):
                values.append(str(value))
    return " ".join(values).lower()


def _source_event_key(dataset_name: str, row: dict[str, Any], payload: dict[str, Any]) -> str:
    explicit = _first_text(row, payload, ["event_key", "event_id", "source_event_id", "id"])
    if explicit:
        return explicit
    stable = {
        "dataset": dataset_name,
        "source_path": row.get("source_path_inside_bundle"),
        "run_id": row.get("run_id"),
        "row_index": row.get("row_index"),
        "ts": _iso(_first_timestamp(row, payload, ["ts_utc", "ts", "bundle_ts", "ingest_ts"])),
        "payload": payload or {
            key: value
            for key, value in row.items()
            if key not in {"ingest_ts", "bundle_ts", "source_count"}
        },
    }
    rendered = json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _seen_time(row: dict[str, Any]) -> datetime:
    for key in ["source_bundle_ts", "ts_utc", "created_at"]:
        value = row.get(key)
        if isinstance(value, datetime):
            return _ensure_utc(value)
    return datetime.min.replace(tzinfo=UTC)


def _datetime_expr(df: pl.DataFrame, column: str) -> pl.Expr:
    if df.schema.get(column) == pl.String:
        return pl.col(column).str.to_datetime(time_zone="UTC", strict=False).alias(column)
    return pl.col(column).cast(pl.Datetime(time_zone="UTC")).alias(column)


def _values(rows: list[dict[str, Any]], column: str) -> list[float]:
    return [
        value
        for value in (_finite_float(row.get(column)) for row in rows)
        if value is not None
    ]


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[index]
    return (ordered[index - 1] + ordered[index]) / 2.0


def _p25(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(int((len(ordered) - 1) * 0.25), 0)
    return ordered[index]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return _ensure_utc(value).isoformat()
    return str(value or "")


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip() == "")


def _as_of_date(value: str | None) -> date:
    if value is None or value == "auto":
        return datetime.now(UTC).date()
    return date.fromisoformat(value)


def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision") or "UNKNOWN")
        counts[decision] = counts.get(decision, 0) + 1
    return counts
