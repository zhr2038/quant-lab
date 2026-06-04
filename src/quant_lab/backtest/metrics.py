from __future__ import annotations

from statistics import median
from typing import Any

import polars as pl

from quant_lab.backtest.datasets import float_or_none


def quantile(values: list[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_net_bps(values: list[Any]) -> dict[str, Any]:
    numeric = [value for value in (float_or_none(item) for item in values) if value is not None]
    if not numeric:
        return {
            "complete_sample_count": 0,
            "avg_net_bps": None,
            "median_net_bps": None,
            "p25_net_bps": None,
            "p10_net_bps": None,
            "win_rate": None,
            "max_loss_bps": None,
        }
    return {
        "complete_sample_count": len(numeric),
        "avg_net_bps": sum(numeric) / len(numeric),
        "median_net_bps": median(numeric),
        "p25_net_bps": quantile(numeric, 0.25),
        "p10_net_bps": quantile(numeric, 0.10),
        "win_rate": sum(1 for item in numeric if item > 0) / len(numeric),
        "max_loss_bps": min(numeric),
    }


def recent_7d_avg_net_bps(samples: list[dict[str, Any]]) -> float | None:
    dated = [row for row in samples if row.get("_decision_ts") is not None]
    if not dated:
        return None
    latest = max(row["_decision_ts"] for row in dated)
    cutoff = latest.timestamp() - 7 * 24 * 3600
    values = [
        float_or_none(row.get("net_bps"))
        for row in dated
        if row["_decision_ts"].timestamp() >= cutoff
    ]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def max_drawdown_bps(values: list[Any]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for raw in values:
        bps = float_or_none(raw)
        if bps is None:
            continue
        equity *= 1.0 + bps / 10_000.0
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd * 10_000.0


def frame_with_schema(rows: list[dict[str, Any]], fields: list[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={field: pl.Utf8 for field in fields})
    frame = pl.DataFrame(rows, infer_schema_length=None)
    for field in fields:
        if field not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(field))
    return frame.select(fields)
