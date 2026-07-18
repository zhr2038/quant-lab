"""Frozen low-volatility forward-paper contract."""

from __future__ import annotations

from datetime import datetime

import polars as pl

FORWARD_CONFIG = {
    "factor": "low_vol_20d",
    "universe": "v1_dynamic_top20",
    "top_n": 3,
    "weighting": "score",
    "rebalance_hours": 120,
    "btc_trend_filter": True,
    "btc_trend_lookback_hours": 1440,
    "one_way_cost_bps": 15.0,
    "hypothesis_type": "POST_HOC_HYPOTHESIS",
}


def validate_forward_timestamps(timestamps: pl.Series, cutoff: datetime) -> None:
    if timestamps.len() and timestamps.min() <= cutoff:
        raise ValueError("forward paper includes timestamp at or before v1 cutoff")


def upsert_by_decision_timestamp(existing: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
    """Idempotent resume: new rows replace the same decision timestamp."""
    if existing.is_empty():
        return new.sort("decision_timestamp")
    if new.is_empty():
        return existing.sort("decision_timestamp")
    return pl.concat([existing, new], how="diagonal_relaxed").unique(
        subset=["decision_timestamp"], keep="last", maintain_order=True
    ).sort("decision_timestamp")
