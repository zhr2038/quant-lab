from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl

EFFECTIVENESS_SCHEMA = {
    "evaluation_level": pl.Utf8,
    "total_denominator": pl.Int64,
    "deduplicated_denominator": pl.Int64,
    "missed_profitable_count": pl.Int64,
    "avoided_loss_count": pl.Int64,
    "net_effect_bps": pl.Float64,
    "mean_effect_bps": pl.Float64,
    "confidence_interval_low_bps": pl.Float64,
    "confidence_interval_high_bps": pl.Float64,
    "outcome_coverage": pl.Float64,
    "coverage_scope": pl.Utf8,
    "production_decision_eligible": pl.Boolean,
    "production_ineligible_reason": pl.Utf8,
    "schema_version": pl.Utf8,
}


def build_gate_effectiveness_report(
    rows: pl.DataFrame,
    *,
    evaluation_level: str,
    dedupe_field: str,
    production_min_rows: int = 30,
    production_min_coverage: float = 0.8,
) -> pl.DataFrame:
    materialized = rows.to_dicts() if not rows.is_empty() else []
    deduplicated = _dedupe(materialized, dedupe_field)
    observed = [row for row in deduplicated if _outcome(row) is not None]
    effects = [_effect(row) for row in observed]
    missed = sum(_bool(row.get("false_block")) for row in observed)
    avoided = sum(_bool(row.get("loss_saved")) for row in observed)
    coverage = len(observed) / len(deduplicated) if deduplicated else 0.0
    low, high = _confidence_interval(effects)
    eligible = len(observed) >= production_min_rows and coverage >= production_min_coverage
    row = {
        "evaluation_level": evaluation_level,
        "total_denominator": len(materialized),
        "deduplicated_denominator": len(deduplicated),
        "missed_profitable_count": missed,
        "avoided_loss_count": avoided,
        "net_effect_bps": sum(effects),
        "mean_effect_bps": statistics.fmean(effects) if effects else 0.0,
        "confidence_interval_low_bps": low,
        "confidence_interval_high_bps": high,
        "outcome_coverage": coverage,
        "coverage_scope": dedupe_field,
        "production_decision_eligible": eligible,
        "production_ineligible_reason": "" if eligible else "insufficient_deduplicated_evidence",
        "schema_version": "gate_effectiveness.v1",
    }
    return (
        pl.DataFrame([row])
        .cast(EFFECTIVENESS_SCHEMA, strict=False)
        .select(list(EFFECTIVENESS_SCHEMA))
    )


def _dedupe(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = str(row.get(field) or row.get("event_id") or row.get("paper_trade_id") or index)
        current = output.get(key)
        if current is None or (_outcome(current) is None and _outcome(row) is not None):
            output[key] = row
    return list(output.values())


def _outcome(row: Mapping[str, Any]) -> float | None:
    for key in ("after_cost_bps", "net_pnl_bps", "net_bps", "paper_pnl_bps"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _effect(row: Mapping[str, Any]) -> float:
    avoided = _float(row.get("loss_saved_bps")) or 0.0
    missed = _float(row.get("missed_profit_bps")) or 0.0
    if avoided or missed:
        return avoided - missed
    return _outcome(row) or 0.0


def _confidence_interval(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    if len(materialized) < 2:
        mean = materialized[0] if materialized else 0.0
        return mean, mean
    mean = statistics.fmean(materialized)
    error = 1.96 * statistics.stdev(materialized) / math.sqrt(len(materialized))
    return mean - error, mean + error


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool:
    return value is True or str(value or "").strip().lower() in {"1", "true", "yes"}
