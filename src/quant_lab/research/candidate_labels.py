from __future__ import annotations

import bisect
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, read_parquet_lazy, upsert_parquet_dataset
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

SOURCE_NAME = "research.v5_candidate_labels.v0.1"
EVENT_SCHEMA_VERSION = "v5.candidate_snapshot.v1"
LABEL_SCHEMA_VERSION = "v5.candidate_label.v1"
QUALITY_SCHEMA_VERSION = "v5.candidate_quality.v1"
SUMMARY_SCHEMA_VERSION = "v5.candidate_outcome_summary.v1"
HORIZON_HOURS = (4, 8, 12, 24, 48, 72, 120)
DEFAULT_INCREMENTAL_LOOKBACK_DAYS = 8
CANDIDATE_EVENT_SYMBOL_FIELDS = ("symbol", "normalized_symbol", "inst_id", "instId")

CANDIDATE_EVENT_DATASET = Path("silver") / "v5_candidate_event"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
RUN_SUMMARY_DATASET = Path("silver") / "v5_run_summary"
CANDIDATE_LABEL_DATASET = Path("gold") / "v5_candidate_label"
CANDIDATE_QUALITY_DATASET = Path("gold") / "v5_candidate_quality_daily"
CANDIDATE_OUTCOME_SUMMARY_DATASET = Path("gold") / "v5_candidate_outcome_summary"

CANDIDATE_FEATURE_FIELDS = (
    "final_score",
    "f1_mom_5d",
    "f2_mom_20d",
    "f3_vol_adj_ret",
    "f4_volume_expansion",
    "f5_rsi_trend_confirm",
    "alpha6_score",
    "ml_score",
    "mean_reversion_score",
    "expected_edge_bps",
    "required_edge_bps",
)
CANDIDATE_REQUIRED_FEATURE_FIELDS = (
    "final_score",
    "expected_edge_bps",
    "required_edge_bps",
)

LABEL_SCHEMA = {
    "strategy": pl.Utf8,
    "candidate_label_schema_version": pl.Utf8,
    "candidate_id": pl.Utf8,
    "run_id": pl.Utf8,
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "block_reason": pl.Utf8,
    "final_decision": pl.Utf8,
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
    "alpha6_side": pl.Utf8,
    "regime_state": pl.Utf8,
    "risk_level": pl.Utf8,
    "btc_trend_state": pl.Utf8,
    "broad_market_positive_count": pl.Int64,
    "funding_state": pl.Utf8,
    "volatility_bucket": pl.Utf8,
    "protect_level": pl.Utf8,
    "final_score": pl.Float64,
    "expected_edge_bps": pl.Float64,
    "required_edge_bps": pl.Float64,
    "source_event_bundle_sha256": pl.Utf8,
    "source_path_inside_bundle": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


class CandidateLabelBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of_date: str
    candidate_event_rows: int = Field(ge=0)
    candidate_label_rows: int = Field(ge=0)
    candidate_quality_rows: int = Field(ge=0)
    candidate_outcome_summary_rows: int = Field(ge=0)
    complete_label_rows: int = Field(ge=0)
    mode: str = "full"
    lookback_days: int | None = None
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_candidate_labels(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
    mode: str = "full",
    lookback_days: int = DEFAULT_INCREMENTAL_LOOKBACK_DAYS,
) -> CandidateLabelBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    created_at = datetime.now(UTC)
    normalized_mode = _normalize_build_mode(mode)

    if normalized_mode == "incremental":
        events = _read_recent_dataset(
            root / CANDIDATE_EVENT_DATASET,
            day=day,
            lookback_days=lookback_days,
            timestamp_columns=("ts_utc", "bundle_ts", "ingest_ts"),
        )
        market_bars = _read_recent_dataset(
            root / MARKET_BAR_DATASET,
            day=day,
            lookback_days=lookback_days + 1,
            timestamp_columns=("ts", "ingest_ts"),
            future_days=6,
        )
        run_summary = _read_recent_dataset(
            root / RUN_SUMMARY_DATASET,
            day=day,
            lookback_days=lookback_days,
            timestamp_columns=("bundle_ts", "ingest_ts"),
        )
    else:
        events = read_parquet_dataset(root / CANDIDATE_EVENT_DATASET)
        market_bars = read_parquet_dataset(root / MARKET_BAR_DATASET)
        run_summary = read_parquet_dataset(root / RUN_SUMMARY_DATASET)

    labels = build_candidate_labels(events, market_bars, created_at=created_at)
    quality = build_candidate_quality(
        events,
        labels,
        run_summary,
        as_of_date=day,
        created_at=created_at,
    )
    summary = build_candidate_outcome_summary(labels, as_of_date=day, created_at=created_at)

    label_rows = _upsert_if_not_empty(
        labels,
        root / CANDIDATE_LABEL_DATASET,
        ["strategy", "candidate_id", "horizon_hours"],
    )
    quality_rows = upsert_parquet_dataset(
        quality,
        root / CANDIDATE_QUALITY_DATASET,
        key_columns=["strategy", "date"],
    )
    summary_rows = _upsert_if_not_empty(
        summary,
        root / CANDIDATE_OUTCOME_SUMMARY_DATASET,
        ["strategy", "date", "block_reason", "strategy_candidate", "symbol", "horizon_hours"],
    )
    complete_label_rows = _complete_label_count(labels)
    warnings = _quality_warnings(quality)
    return CandidateLabelBuildResult(
        as_of_date=day.isoformat(),
        candidate_event_rows=events.height,
        candidate_label_rows=label_rows,
        candidate_quality_rows=quality_rows,
        candidate_outcome_summary_rows=summary_rows,
        complete_label_rows=complete_label_rows,
        mode=normalized_mode,
        lookback_days=lookback_days if normalized_mode == "incremental" else None,
        warnings=warnings,
    )


def build_candidate_labels(
    events: pl.DataFrame,
    market_bars: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    return compute_v5_candidate_labels(events, market_bars, created_at=created_at)


def compute_v5_candidate_labels(
    events: pl.DataFrame,
    market_bars: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    """Compute Candidate Label rows without reading or writing a Lake."""

    if events.is_empty():
        return pl.DataFrame(schema=LABEL_SCHEMA)
    created = created_at or datetime.now(UTC)
    bars_by_symbol = candidate_label_bars_by_symbol(market_bars)
    rows: list[dict[str, Any]] = []
    for event in events.to_dicts():
        rows.extend(_label_event(event, bars_by_symbol, created_at=created))
    return _label_frame(rows)


def build_candidate_quality(
    events: pl.DataFrame,
    labels: pl.DataFrame,
    run_summary: pl.DataFrame,
    *,
    as_of_date: date,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    return derive_candidate_quality(
        events,
        labels,
        run_summary,
        as_of_date=as_of_date,
        created_at=created_at,
    )


def derive_candidate_quality(
    events: pl.DataFrame,
    labels: pl.DataFrame,
    run_summary: pl.DataFrame,
    *,
    as_of_date: date,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    """Derive the cloud-owned Candidate Quality row from accepted inputs."""

    created = created_at or datetime.now(UTC)
    event_rows = events.to_dicts() if not events.is_empty() else []
    label_rows = labels.to_dicts() if not labels.is_empty() else []
    summary_runs = run_summary_run_ids(run_summary)
    event_run_ids = {
        _clean_text(row.get("run_id")) for row in event_rows if _clean_text(row.get("run_id"))
    }
    rows_by_run: Counter[str] = Counter()
    symbol_rows_by_run: dict[str, Counter[str]] = defaultdict(Counter)
    for row in event_rows:
        run_id = _clean_text(row.get("run_id")) or "UNKNOWN_RUN"
        symbol = _clean_text(row.get("symbol")) or "UNKNOWN_SYMBOL"
        rows_by_run[run_id] += 1
        symbol_rows_by_run[run_id][symbol] += 1

    summary_runs_in_candidate_window = _summary_runs_in_candidate_window(
        summary_runs,
        event_run_ids,
    )
    missing_runs = sorted(summary_runs_in_candidate_window - event_run_ids)
    feature_rows, no_signal_context_rows = _candidate_feature_denominator_rows(event_rows)
    feature_by_field = _feature_completeness_by_field(feature_rows)
    feature_completeness = statistics.fmean(feature_by_field.values()) if feature_by_field else 0.0
    required_feature_by_field = _feature_completeness_by_field(
        feature_rows,
        fields=CANDIDATE_REQUIRED_FEATURE_FIELDS,
    )
    required_feature_completeness = (
        statistics.fmean(required_feature_by_field.values()) if required_feature_by_field else 0.0
    )
    expected_labels = len(event_rows) * len(HORIZON_HOURS)
    complete_labels = sum(1 for row in label_rows if row.get("label_status") == "complete")
    future_pending_labels = sum(
        1 for row in label_rows if row.get("label_reason") == "future_bar_unavailable"
    )
    eligible_labels = max(len(label_rows) - future_pending_labels, 0)
    label_completeness = complete_labels / eligible_labels if eligible_labels else 0.0
    raw_label_completeness = complete_labels / expected_labels if expected_labels else 0.0
    cost_sources = Counter(_candidate_cost_source(row.get("cost_source")) for row in event_rows)
    cost_source_covered = sum(
        count for source, count in cost_sources.items() if source != "MISSING"
    )
    cost_source_coverage = cost_source_covered / len(event_rows) if event_rows else 0.0
    cost_source_quality_counts = {
        "covered": cost_source_covered,
        "missing": cost_sources.get("MISSING", 0),
        "by_source": dict(sorted(cost_sources.items())),
    }
    run_symbol_min_rows = min(
        (count for symbols in symbol_rows_by_run.values() for count in symbols.values()),
        default=0,
    )

    warnings: list[str] = []
    if not event_rows:
        warnings.append("v5_candidate_event_empty")
    if missing_runs:
        warnings.append("runs_without_candidate_event")
    if required_feature_completeness < 0.8 and event_rows:
        warnings.append("candidate_required_feature_completeness_below_80pct")
    if label_completeness < 0.8 and event_rows:
        warnings.append("candidate_label_completeness_below_80pct")
    if cost_source_coverage < 0.8 and event_rows:
        warnings.append("candidate_cost_source_coverage_below_80pct")

    status = "PASS" if not warnings else "WARN"
    return pl.DataFrame(
        [
            {
                "strategy": "v5",
                "date": as_of_date.isoformat(),
                "schema_version": QUALITY_SCHEMA_VERSION,
                "status": status,
                "candidate_event_rows": len(event_rows),
                "run_count": len(summary_runs_in_candidate_window or event_run_ids),
                "runs_with_candidate_event": len(event_run_ids),
                "runs_without_candidate_event": len(missing_runs),
                "runs_without_candidate_event_json": safe_json_dumps(missing_runs),
                "candidate_rows_by_run_json": safe_json_dumps(dict(sorted(rows_by_run.items()))),
                "candidate_symbol_rows_by_run_json": safe_json_dumps(
                    {
                        run_id: dict(sorted(symbols.items()))
                        for run_id, symbols in sorted(symbol_rows_by_run.items())
                    }
                ),
                "run_symbol_min_rows": run_symbol_min_rows,
                "feature_denominator_rows": len(feature_rows),
                "no_signal_context_rows": no_signal_context_rows,
                "feature_completeness": feature_completeness,
                "feature_completeness_by_field_json": safe_json_dumps(feature_by_field),
                "required_feature_completeness": required_feature_completeness,
                "required_feature_completeness_by_field_json": safe_json_dumps(
                    required_feature_by_field
                ),
                "expected_label_rows": expected_labels,
                "label_rows": len(label_rows),
                "eligible_label_rows": eligible_labels,
                "future_pending_label_rows": future_pending_labels,
                "complete_label_rows": complete_labels,
                "label_completeness": label_completeness,
                "raw_label_completeness": raw_label_completeness,
                "cost_source_coverage": cost_source_coverage,
                "cost_source_counts_json": safe_json_dumps(dict(sorted(cost_sources.items()))),
                "cost_source_quality_counts": safe_json_dumps(cost_source_quality_counts),
                "warnings_json": safe_json_dumps(sorted(warnings)),
                "created_at": created,
                "source": SOURCE_NAME,
            }
        ],
        orient="row",
    )


def build_candidate_outcome_summary(
    labels: pl.DataFrame,
    *,
    as_of_date: date,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    return derive_candidate_outcome_summary(
        labels,
        as_of_date=as_of_date,
        created_at=created_at,
    )


def derive_candidate_outcome_summary(
    labels: pl.DataFrame,
    *,
    as_of_date: date,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    """Derive the cloud-owned Candidate Outcome Summary without Lake I/O."""

    created = created_at or datetime.now(UTC)
    if labels.is_empty():
        return _summary_frame([])
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in labels.to_dicts():
        key = (
            _clean_text(row.get("block_reason")) or "NONE",
            _clean_text(row.get("strategy_candidate")) or "UNKNOWN",
            _clean_text(row.get("symbol")) or "UNKNOWN",
            int(row.get("horizon_hours") or 0),
        )
        grouped[key].append(row)

    rows: list[dict[str, Any]] = []
    for (block_reason, strategy_candidate, symbol, horizon_hours), group_rows in sorted(
        grouped.items()
    ):
        complete = [row for row in group_rows if row.get("label_status") == "complete"]
        net_values = _float_values(complete, "net_bps_after_cost")
        gross_values = _float_values(complete, "gross_bps")
        mfe_values = _float_values(complete, "mfe_bps")
        mae_values = _float_values(complete, "mae_bps")
        wins = [bool(row.get("win")) for row in complete if row.get("win") is not None]
        rows.append(
            {
                "strategy": "v5",
                "date": as_of_date.isoformat(),
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "block_reason": block_reason,
                "strategy_candidate": strategy_candidate,
                "symbol": symbol,
                "horizon_hours": horizon_hours,
                "sample_count": len(group_rows),
                "complete_sample_count": len(complete),
                "avg_gross_bps": _mean(gross_values),
                "avg_net_bps": _mean(net_values),
                "median_net_bps": _median(net_values),
                "win_rate": (sum(wins) / len(wins)) if wins else None,
                "downside_p25_bps": _quantile(net_values, 0.25),
                "avg_mfe_bps": _mean(mfe_values),
                "avg_mae_bps": _mean(mae_values),
                "label_status_counts_json": safe_json_dumps(
                    dict(
                        sorted(
                            Counter(
                                str(row.get("label_status") or "") for row in group_rows
                            ).items()
                        )
                    )
                ),
                "created_at": created,
                "source": SOURCE_NAME,
            }
        )
    return _summary_frame(rows)


def _label_event(
    event: dict[str, Any],
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    created_at: datetime,
) -> list[dict[str, Any]]:
    payload = _payload(event)
    symbol = _symbol(event, payload)
    ts_utc = _coerce_timestamp(_first_value(event, payload, ["ts_utc", "ts", "timestamp"]))
    base = _base_label_row(event, payload, symbol, ts_utc, created_at)
    if not symbol:
        return [_empty_horizon_row(base, hours, "missing_symbol") for hours in HORIZON_HOURS]
    if ts_utc is None:
        return [_empty_horizon_row(base, hours, "missing_event_ts") for hours in HORIZON_HOURS]
    bars = bars_by_symbol.get(symbol, [])
    if not bars:
        return [_empty_horizon_row(base, hours, "missing_market_bar") for hours in HORIZON_HOURS]

    decision = next_candidate_decision_bar(bars, ts_utc)
    if decision is None:
        return [_empty_horizon_row(base, hours, "missing_decision_bar") for hours in HORIZON_HOURS]

    decision_index, decision_bar = decision
    ts_values = [bar["ts"] for bar in bars]
    decision_ts = _coerce_timestamp(decision_bar.get("ts"))
    decision_close = _finite_float(decision_bar.get("close"))
    if decision_ts is None or decision_close is None or decision_close <= 0:
        return [_empty_horizon_row(base, hours, "bad_decision_bar") for hours in HORIZON_HOURS]

    rows = []
    direction = _direction_multiplier(event, payload)
    cost_bps = abs(_finite_float(_first_value(event, payload, ["cost_bps", "cost"])) or 0.0)
    for hours in HORIZON_HOURS:
        target_ts = decision_ts + timedelta(hours=hours)
        future_index = bisect.bisect_left(ts_values, target_ts)
        if future_index >= len(bars):
            rows.append(
                _empty_horizon_row(
                    base | {"decision_ts": decision_ts, "entry_close": decision_close},
                    hours,
                    "future_bar_unavailable",
                )
            )
            continue
        future_bar = bars[future_index]
        label_ts = _coerce_timestamp(future_bar.get("ts"))
        label_close = _finite_float(future_bar.get("close"))
        if label_ts is None or label_close is None or label_close <= 0:
            rows.append(
                _empty_horizon_row(
                    base | {"decision_ts": decision_ts, "entry_close": decision_close},
                    hours,
                    "bad_future_bar",
                )
            )
            continue
        path = bars[decision_index : future_index + 1]
        gross_bps = ((label_close / decision_close) - 1.0) * 10_000.0 * direction
        net_bps = gross_bps - cost_bps
        mfe_bps, mae_bps = _mfe_mae_bps(path, decision_close, direction)
        rows.append(
            base
            | {
                "horizon_hours": hours,
                "decision_ts": decision_ts,
                "label_ts": label_ts,
                "entry_close": decision_close,
                "label_close": label_close,
                "gross_bps": gross_bps,
                "net_bps_after_cost": net_bps,
                "mfe_bps": mfe_bps,
                "mae_bps": mae_bps,
                "win": net_bps > 0.0,
                "label_status": "complete",
                "label_reason": "ok",
            }
        )
    return rows


def _base_label_row(
    event: dict[str, Any],
    payload: dict[str, Any],
    symbol: str,
    ts_utc: datetime | None,
    created_at: datetime,
) -> dict[str, Any]:
    return {
        "strategy": _clean_text(event.get("strategy")) or "v5",
        "candidate_label_schema_version": LABEL_SCHEMA_VERSION,
        "candidate_id": _clean_text(_first_value(event, payload, ["candidate_id"])),
        "run_id": _clean_text(_first_value(event, payload, ["run_id"])),
        "ts_utc": ts_utc,
        "symbol": symbol,
        "strategy_candidate": _clean_text(
            _first_value(event, payload, ["strategy_candidate", "candidate_name", "candidate"])
        ),
        "block_reason": _clean_text(_first_value(event, payload, ["block_reason", "reason"])),
        "final_decision": _clean_text(_first_value(event, payload, ["final_decision", "decision"])),
        "cost_bps": abs(_finite_float(_first_value(event, payload, ["cost_bps", "cost"])) or 0.0),
        "cost_source": _clean_text(_first_value(event, payload, ["cost_source"])),
        "alpha6_side": _clean_text(_first_value(event, payload, ["alpha6_side"])),
        "regime_state": _clean_text(_first_value(event, payload, ["regime_state", "regime"])),
        "risk_level": _clean_text(_first_value(event, payload, ["risk_level"])),
        "btc_trend_state": _clean_text(
            _first_value(event, payload, ["btc_trend_state", "btc_state", "btc_regime"])
        ),
        "broad_market_positive_count": _finite_int(
            _first_value(
                event,
                payload,
                ["broad_market_positive_count", "positive_count", "breadth_positive_count"],
            )
        ),
        "funding_state": _clean_text(_first_value(event, payload, ["funding_state"])),
        "volatility_bucket": _clean_text(
            _first_value(event, payload, ["volatility_bucket", "vol_bucket"])
        ),
        "protect_level": _clean_text(_first_value(event, payload, ["protect_level"])),
        "final_score": _finite_float(_first_value(event, payload, ["final_score", "score"])),
        "expected_edge_bps": _finite_float(_first_value(event, payload, ["expected_edge_bps"])),
        "required_edge_bps": _finite_float(_first_value(event, payload, ["required_edge_bps"])),
        "source_event_bundle_sha256": _clean_text(event.get("bundle_sha256")),
        "source_path_inside_bundle": _clean_text(event.get("source_path_inside_bundle")),
        "created_at": created_at,
        "source": SOURCE_NAME,
    }


def _empty_horizon_row(base: dict[str, Any], hours: int, reason: str) -> dict[str, Any]:
    return base | {
        "horizon_hours": hours,
        "decision_ts": base.get("decision_ts"),
        "label_ts": None,
        "entry_close": base.get("entry_close"),
        "label_close": None,
        "gross_bps": None,
        "net_bps_after_cost": None,
        "mfe_bps": None,
        "mae_bps": None,
        "win": None,
        "label_status": "partial",
        "label_reason": reason,
    }


def _mfe_mae_bps(
    path: list[dict[str, Any]],
    decision_close: float,
    direction: int,
) -> tuple[float | None, float | None]:
    highs = [_finite_float(bar.get("high")) for bar in path]
    lows = [_finite_float(bar.get("low")) for bar in path]
    highs = [value for value in highs if value is not None and value > 0]
    lows = [value for value in lows if value is not None and value > 0]
    if not highs or not lows:
        return None, None
    if direction < 0:
        mfe = ((decision_close / min(lows)) - 1.0) * 10_000.0
        mae = ((decision_close / max(highs)) - 1.0) * 10_000.0
    else:
        mfe = ((max(highs) / decision_close) - 1.0) * 10_000.0
        mae = ((min(lows) / decision_close) - 1.0) * 10_000.0
    return mfe, mae


def candidate_label_bars_by_symbol(
    market_bars: pl.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Return the exact closed-bar series used by the candidate label builder."""
    if market_bars.is_empty():
        return {}
    required = {"symbol", "ts", "close", "high", "low"}
    if not required.issubset(market_bars.columns):
        return {}
    bars = market_bars
    if "is_closed" in bars.columns:
        bars = bars.filter(pl.col("is_closed").fill_null(True))
    bars = bars.with_columns(
        [
            pl.col("symbol").map_elements(normalize_symbol, return_dtype=pl.Utf8),
            _datetime_expr(bars, "ts"),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
        ]
    )
    selected_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for symbol, symbol_bars in bars.sort(["symbol", "ts"]).group_by("symbol", maintain_order=True):
        symbol_key = str(symbol[0] if isinstance(symbol, tuple) else symbol)
        selected = _select_timeframe(symbol_bars)
        selected_by_symbol[symbol_key] = selected.sort("ts").to_dicts()
    return selected_by_symbol


def next_candidate_decision_bar(
    bars: list[dict[str, Any]],
    candidate_ts: datetime,
) -> tuple[int, dict[str, Any]] | None:
    """Select the first actual market bar strictly after a candidate event."""
    ts_values = [_coerce_timestamp(bar.get("ts")) for bar in bars]
    if any(value is None for value in ts_values):
        return None
    decision_index = bisect.bisect_right(ts_values, candidate_ts)  # type: ignore[arg-type]
    if decision_index >= len(bars):
        return None
    return decision_index, bars[decision_index]


def _select_timeframe(symbol_bars: pl.DataFrame) -> pl.DataFrame:
    if "timeframe" not in symbol_bars.columns:
        return symbol_bars
    choices = []
    for timeframe, group in symbol_bars.group_by("timeframe", maintain_order=True):
        value = str(timeframe[0] if isinstance(timeframe, tuple) else timeframe)
        minutes = _timeframe_minutes(value)
        exact_one_hour = 0 if minutes == 60 else 1
        small_frame_rank = 0 if minutes is not None and minutes <= 60 else 1
        choices.append(
            (exact_one_hour, small_frame_rank, minutes or 1_000_000, -group.height, group)
        )
    if not choices:
        return symbol_bars
    return sorted(choices, key=lambda item: item[:4])[0][4]


def _timeframe_minutes(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    unit = text[-1].lower()
    try:
        amount = int(float(text[:-1]))
    except ValueError:
        return None
    if unit == "m":
        return amount
    if unit == "h":
        return amount * 60
    if unit == "d":
        return amount * 60 * 24
    return None


def run_summary_run_ids(run_summary: pl.DataFrame) -> set[str]:
    if run_summary.is_empty():
        return set()
    run_ids: set[str] = set()
    for row in run_summary.to_dicts():
        source_path = str(row.get("source_path_inside_bundle") or "")
        if "window_summary.json" in source_path:
            continue
        payload = _payload(row)
        run_id = _clean_text(_first_value(row, payload, ["run_id", "runId", "run"]))
        if run_id:
            run_ids.add(run_id)
    return run_ids


def _summary_runs_in_candidate_window(
    summary_runs: set[str],
    event_run_ids: set[str],
) -> set[str]:
    if not summary_runs or not event_run_ids:
        return summary_runs
    first_candidate_run = min(event_run_ids, key=_run_id_sort_key)
    first_candidate_key = _run_id_sort_key(first_candidate_run)
    return {run_id for run_id in summary_runs if _run_id_sort_key(run_id) >= first_candidate_key}


def _run_id_sort_key(run_id: str) -> tuple[int, str, str]:
    text = _clean_text(run_id)
    if len(text) >= 11 and text[8] == "_":
        day = text[:8]
        hour = text[9:11]
        if day.isdigit() and hour.isdigit():
            return (0, day, hour)
    return (1, text, "")


def _feature_completeness_by_field(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...] = CANDIDATE_FEATURE_FIELDS,
) -> dict[str, float]:
    if not rows:
        return {field: 0.0 for field in fields}
    result: dict[str, float] = {}
    for field in fields:
        complete = 0
        for row in rows:
            payload = _payload(row)
            if _clean_text(_first_value(row, payload, [field])):
                complete += 1
        result[field] = complete / len(rows)
    return result


def _candidate_feature_denominator_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if not rows:
        return [], 0
    candidate_rows = [row for row in rows if _candidate_row_needs_feature_coverage(row)]
    if candidate_rows:
        return candidate_rows, len(rows) - len(candidate_rows)
    return rows, 0


def _candidate_row_needs_feature_coverage(row: dict[str, Any]) -> bool:
    payload = _payload(row)
    eligible = _clean_text(_first_value(row, payload, ["eligible_before_filters"])).lower()
    if eligible in {"true", "1", "yes", "y"}:
        return True
    if eligible in {"false", "0", "no", "n"}:
        if _clean_text(_first_value(row, payload, ["block_reason"])):
            return True
        decision = _clean_text(_first_value(row, payload, ["final_decision"])).lower()
        return decision not in {"", "no_order"}
    if _clean_text(_first_value(row, payload, ["block_reason"])):
        return True
    decision = _clean_text(_first_value(row, payload, ["final_decision"])).lower()
    if decision and decision != "no_order":
        return True
    for field in ["target_weight_raw", "target_weight_after_risk", "current_weight"]:
        value = _finite_float(_first_value(row, payload, [field]))
        if value is not None and abs(value) > 0.0:
            return True
    return bool(_clean_text(_first_value(row, payload, ["final_score", "rank"])))


def _symbol(row: dict[str, Any], payload: dict[str, Any]) -> str:
    return candidate_event_symbol(row, payload)


def candidate_event_symbol(
    row: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Resolve one Candidate Event symbol with the signed, legacy-compatible order."""

    resolved_payload: Mapping[str, Any] = payload if payload is not None else _payload(dict(row))
    for field in CANDIDATE_EVENT_SYMBOL_FIELDS:
        for value in (row.get(field), _nested_value(dict(resolved_payload), field)):
            cleaned = _clean_text(value)
            if not cleaned:
                continue
            normalized = normalize_symbol(cleaned)
            if normalized:
                return normalized
    return ""


def candidate_event_symbol_expr(columns: Collection[str]) -> pl.Expr:
    """Return the same Candidate Event symbol resolver for lazy Snapshot/Worker plans."""

    available = [
        field for field in (*CANDIDATE_EVENT_SYMBOL_FIELDS, "raw_payload_json") if field in columns
    ]
    if not available:
        return pl.lit("", dtype=pl.Utf8)
    return pl.struct(available).map_elements(
        candidate_event_symbol,
        return_dtype=pl.Utf8,
    )


def _direction_multiplier(row: dict[str, Any], payload: dict[str, Any]) -> int:
    for field in ["alpha6_side", "side", "entry_side", "direction", "final_decision"]:
        value = _clean_text(_first_value(row, payload, [field])).lower()
        if value in {"short", "sell", "down", "bear", "negative"}:
            return -1
        if value in {"long", "buy", "up", "bull", "positive"}:
            return 1
    current = _finite_float(_first_value(row, payload, ["current_weight"]))
    target = _finite_float(
        _first_value(row, payload, ["target_weight_after_risk", "target_weight_raw"])
    )
    if current is not None and target is not None and target < current:
        return -1
    return 1


def _label_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=LABEL_SCHEMA)
    normalized = [{column: row.get(column) for column in LABEL_SCHEMA} for row in rows]
    return pl.DataFrame(normalized, schema=LABEL_SCHEMA, orient="row")


def _summary_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = {
        "strategy": pl.Utf8,
        "date": pl.Utf8,
        "schema_version": pl.Utf8,
        "block_reason": pl.Utf8,
        "strategy_candidate": pl.Utf8,
        "symbol": pl.Utf8,
        "horizon_hours": pl.Int64,
        "sample_count": pl.Int64,
        "complete_sample_count": pl.Int64,
        "avg_gross_bps": pl.Float64,
        "avg_net_bps": pl.Float64,
        "median_net_bps": pl.Float64,
        "win_rate": pl.Float64,
        "downside_p25_bps": pl.Float64,
        "avg_mfe_bps": pl.Float64,
        "avg_mae_bps": pl.Float64,
        "label_status_counts_json": pl.Utf8,
        "created_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row")


def _upsert_if_not_empty(df: pl.DataFrame, dataset_path: Path, keys: list[str]) -> int:
    if df.is_empty():
        return read_parquet_dataset(dataset_path).height
    return upsert_parquet_dataset(df, dataset_path, key_columns=keys)


def _complete_label_count(labels: pl.DataFrame) -> int:
    if labels.is_empty() or "label_status" not in labels.columns:
        return 0
    return labels.filter(pl.col("label_status") == "complete").height


def _normalize_build_mode(mode: str) -> str:
    normalized = str(mode or "full").strip().lower()
    if normalized not in {"full", "incremental"}:
        raise ValueError("mode must be either 'full' or 'incremental'")
    return normalized


def _read_recent_dataset(
    dataset_path: Path,
    *,
    day: date,
    lookback_days: int,
    timestamp_columns: tuple[str, ...],
    future_days: int = 1,
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
    end = datetime.combine(day + timedelta(days=max(future_days, 0)), time.min, tzinfo=UTC)
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


def _quality_warnings(quality: pl.DataFrame) -> list[str]:
    if quality.is_empty() or "warnings_json" not in quality.columns:
        return []
    value = quality["warnings_json"][0]
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in payload] if isinstance(payload, list) else []


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_payload_json")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_value(row: dict[str, Any], payload: dict[str, Any], fields: list[str]) -> Any:
    for field in fields:
        value = row.get(field)
        if _empty(value):
            value = _nested_value(payload, field)
        if not _empty(value):
            return value
    return None


def _nested_value(payload: dict[str, Any], field: str) -> Any:
    value: Any = payload
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    rendered = str(value).strip()
    return "" if rendered.lower() in {"", "none", "null", "nan", "n/a", "na"} else rendered


def _candidate_cost_source(value: Any) -> str:
    if value is None:
        return "MISSING"
    rendered = str(value).strip()
    if not rendered or rendered.lower() in {"none", "null", "nan", "n/a", "na", "missing"}:
        return "MISSING"
    return rendered


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_int(value: Any) -> int | None:
    number = _finite_float(value)
    return int(number) if number is not None else None


def _float_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _finite_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(math.ceil(q * len(ordered)) - 1, 0), len(ordered) - 1)
    return ordered[index]


def _coerce_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _datetime_expr(df: pl.DataFrame, column: str) -> pl.Expr:
    expression = pl.col(column)
    if df.schema.get(column) == pl.String:
        return expression.str.to_datetime(time_zone="UTC", strict=False).alias(column)
    return expression.cast(pl.Datetime(time_zone="UTC"), strict=False).alias(column)


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value and str(value).strip().lower() != "auto":
        return date.fromisoformat(str(value))
    return datetime.now(UTC).date()
