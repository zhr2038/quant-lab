from __future__ import annotations

import bisect
import hashlib
import json
import math
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset
from quant_lab.research.evidence import DEFAULT_RESEARCH_COST_BPS
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"
STRATEGY_EVIDENCE_SAMPLE_DATASET = Path("gold") / "strategy_evidence_sample"
EVIDENCE_VERSION = "strategy-evidence-v0.1"
SOURCE_NAME = "research.strategy_evidence.v0.1"
MIN_LIVE_SMALL_READY_SAMPLES = 30
HORIZON_HOURS = (4, 8, 12, 24, 48, 72, 120)

STRATEGY_CANDIDATES = (
    "v5.btc_leadership_probe_strict",
    "v5.sol_protect_exception",
    "v5.alt_impulse_shadow",
    "v5.swing_f4_f5_alpha6",
    "v5.f3_dominant_entry",
    "v5.mean_reversion_sideways",
)

SOURCE_DATASETS = {
    "v5_decision_audit": Path("silver") / "v5_decision_audit",
    "v5_shadow_outcome": Path("silver") / "v5_shadow_outcome",
    "v5_high_score_blocked_target": Path("silver") / "v5_high_score_blocked_target",
    "v5_high_score_blocked_outcome": Path("silver") / "v5_high_score_blocked_outcome",
    "v5_skipped_candidate_outcome": Path("silver") / "v5_skipped_candidate_outcome",
    "v5_router_decision": Path("silver") / "v5_router_decision",
    "v5_probe_diagnostic": Path("silver") / "v5_probe_diagnostic",
    "v5_quant_lab_cost_usage": Path("silver") / "v5_quant_lab_cost_usage",
}

SAMPLE_SCHEMA: dict[str, Any] = {
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "candidate_name": pl.Utf8,
    "entry_condition_name": pl.Utf8,
    "entry_condition_side": pl.Utf8,
    "entry_condition_signal": pl.Utf8,
    "entry_condition_passed": pl.Boolean,
    "entry_conditions_json": pl.Utf8,
    "block_reason": pl.Utf8,
    "final_score": pl.Float64,
    "f1": pl.Float64,
    "f2": pl.Float64,
    "f3": pl.Float64,
    "f4": pl.Float64,
    "f5": pl.Float64,
    "alpha6_score": pl.Float64,
    "alpha6_side": pl.Utf8,
    "regime_state": pl.Utf8,
    "protect_level": pl.Utf8,
    "expected_edge_bps": pl.Float64,
    "required_edge_bps": pl.Float64,
    "cost_source": pl.Utf8,
    "cost_bps": pl.Float64,
    "source_dataset": pl.Utf8,
    "source_path_inside_bundle": pl.Utf8,
    "source_event_key": pl.Utf8,
    "source_bundle_ts": pl.Datetime(time_zone="UTC"),
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "label_status": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}
for _hours in HORIZON_HOURS:
    SAMPLE_SCHEMA[f"gross_bps_{_hours}h"] = pl.Float64
    SAMPLE_SCHEMA[f"net_bps_after_cost_{_hours}h"] = pl.Float64
    SAMPLE_SCHEMA[f"win_{_hours}h"] = pl.Boolean
    SAMPLE_SCHEMA[f"drawdown_proxy_bps_{_hours}h"] = pl.Float64

SUMMARY_SCHEMA: dict[str, Any] = {
    "candidate_name": pl.Utf8,
    "evidence_version": pl.Utf8,
    "as_of_date": pl.Utf8,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps_by_horizon": pl.Utf8,
    "median_net_bps_by_horizon": pl.Utf8,
    "win_rate_by_horizon": pl.Utf8,
    "downside_p25_by_horizon": pl.Utf8,
    "max_drawdown_proxy": pl.Float64,
    "cost_sensitivity": pl.Utf8,
    "symbol_breakdown": pl.Utf8,
    "regime_breakdown": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "start_ts": pl.Datetime(time_zone="UTC"),
    "end_ts": pl.Datetime(time_zone="UTC"),
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
    decision_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_strategy_evidence(
    lake_root: str | Path,
    *,
    as_of_date: str | None = None,
    min_live_samples: int = MIN_LIVE_SMALL_READY_SAMPLES,
) -> StrategyEvidenceBuildResult:
    root = Path(lake_root)
    day = _as_of_date(as_of_date)
    samples, warnings = build_strategy_evidence_samples(root, as_of_date=day.isoformat())
    summaries = summarize_strategy_evidence(
        samples,
        as_of_date=day,
        min_live_samples=min_live_samples,
    )
    sample_rows = publish_strategy_evidence_samples(root, samples)
    summary_rows = publish_strategy_evidence_summary(root, summaries)
    return StrategyEvidenceBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        sample_rows=sample_rows,
        strategy_evidence_rows=summary_rows,
        extracted_sample_count=samples.height,
        candidate_count=len(summaries),
        decision_counts=_decision_counts(summaries),
        warnings=warnings,
    )


def build_strategy_evidence_samples(
    lake_root: str | Path,
    *,
    as_of_date: str | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    root = Path(lake_root)
    warnings: list[str] = []
    cost_context = _CostContext(root)
    rows: list[dict[str, Any]] = []
    for dataset_name, relative_path in SOURCE_DATASETS.items():
        frame = read_parquet_dataset(root / relative_path)
        if frame.is_empty():
            continue
        for row in frame.to_dicts():
            sample = _sample_from_telemetry_row(dataset_name, row, cost_context)
            if sample is not None:
                rows.append(sample)

    rows = _dedupe_sample_rows(_filter_samples_as_of(rows, as_of_date))
    if not rows:
        return pl.DataFrame(schema=SAMPLE_SCHEMA), warnings

    samples = _samples_frame(rows)
    market_bars = read_parquet_dataset(root / "silver" / "market_bar")
    if market_bars.is_empty():
        warnings.append("market_bar missing; strategy evidence samples have no forward labels")
        return _samples_frame([_with_empty_labels(row) for row in rows]), warnings
    return _attach_forward_labels(samples, market_bars), warnings


def summarize_strategy_evidence(
    samples: pl.DataFrame,
    *,
    as_of_date: date,
    min_live_samples: int = MIN_LIVE_SMALL_READY_SAMPLES,
) -> list[dict[str, Any]]:
    created_at = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    sample_rows = samples.to_dicts() if not samples.is_empty() else []
    for candidate_name in STRATEGY_CANDIDATES:
        candidate_rows = [
            row for row in sample_rows if row.get("candidate_name") == candidate_name
        ]
        rows.append(
            _summary_row(
                candidate_name,
                candidate_rows,
                as_of_date=as_of_date,
                created_at=created_at,
                min_live_samples=min_live_samples,
            )
        )
    return rows


def publish_strategy_evidence_samples(lake_root: str | Path, samples: pl.DataFrame) -> int:
    dataset_path = Path(lake_root) / STRATEGY_EVIDENCE_SAMPLE_DATASET
    if samples.is_empty():
        return read_parquet_dataset(dataset_path).height
    return upsert_parquet_dataset(
        samples,
        dataset_path,
        key_columns=["candidate_name", "symbol", "ts_utc", "source_dataset", "source_event_key"],
    )


def publish_strategy_evidence_summary(
    lake_root: str | Path,
    rows: list[dict[str, Any]],
) -> int:
    dataset_path = Path(lake_root) / STRATEGY_EVIDENCE_DATASET
    if not rows:
        return read_parquet_dataset(dataset_path).height
    frame = pl.DataFrame(rows, schema=SUMMARY_SCHEMA, orient="row")
    return upsert_parquet_dataset(
        frame,
        dataset_path,
        key_columns=["candidate_name", "evidence_version", "as_of_date"],
    )


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
    explicit_candidate = _normalize_candidate_text(explicit, dataset_name=dataset_name)
    if explicit_candidate is not None:
        return explicit_candidate

    text = _row_search_text(row, payload)
    return _normalize_candidate_text(text, dataset_name=dataset_name)


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
        "v5_sol_protect_exception": "v5.sol_protect_exception",
        "sol_protect_exception": "v5.sol_protect_exception",
        "v5_alt_impulse_shadow": "v5.alt_impulse_shadow",
        "alt_impulse_shadow": "v5.alt_impulse_shadow",
        "alt_impulse": "v5.alt_impulse_shadow",
        "v5_swing_f4_f5_alpha6": "v5.swing_f4_f5_alpha6",
        "swing_f4_f5_alpha6": "v5.swing_f4_f5_alpha6",
        "v5_f3_dominant_entry": "v5.f3_dominant_entry",
        "f3_dominant_entry": "v5.f3_dominant_entry",
        "v5_mean_reversion_sideways": "v5.mean_reversion_sideways",
        "mean_reversion_sideways": "v5.mean_reversion_sideways",
    }
    if normalized in direct:
        return direct[normalized]
    if dataset_name == "v5_shadow_outcome" and "alt_impulse" in normalized:
        return "v5.alt_impulse_shadow"
    if "btc" in normalized and "leadership" in normalized and "probe" in normalized:
        if "strict" in normalized:
            return "v5.btc_leadership_probe_strict"
        return None
    if "sol" in normalized and "protect" in normalized and "exception" in normalized:
        return "v5.sol_protect_exception"
    if "alt" in normalized and "impulse" in normalized:
        return "v5.alt_impulse_shadow"
    if "swing" in normalized and "f4" in normalized and "f5" in normalized:
        return "v5.swing_f4_f5_alpha6"
    if "f3" in normalized and "dominant" in normalized:
        return "v5.f3_dominant_entry"
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

    if candidate_name in {"v5.sol_protect_exception", "v5.alt_impulse_shadow"}:
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
        return "LIVE_SMALL_READY", ["positive_net_edge_and_sample_floor_met"]

    paper_ready = (
        (avg_24h is not None and avg_24h > 0.0)
        and (win_24h is not None and win_24h >= 0.50)
        and (downside_24h is None or downside_24h > -100.0)
    )
    if paper_ready:
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
        ["symbol", "normalized_symbol", "inst_id", "instId", "instrument", "pair"],
    )
    normalized = normalize_symbol(raw)
    if normalized:
        return normalized
    if "btc" in candidate_name:
        return "BTC-USDT"
    if "sol" in candidate_name:
        return "SOL-USDT"
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
