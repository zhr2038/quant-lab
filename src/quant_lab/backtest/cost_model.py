from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from quant_lab.backtest.datasets import float_or_none, normalize_strategy_symbol, rows


@dataclass(frozen=True)
class BacktestCost:
    cost_bps: float
    cost_model: str


def conservative_cost_for_symbol(
    cost_bucket_daily: pl.DataFrame | None,
    *,
    symbol: str | None = None,
    default_bps: float = 30.0,
) -> BacktestCost:
    symbol_norm = normalize_strategy_symbol(symbol) if symbol else ""
    values: list[float] = []
    sources: set[str] = set()
    for row in rows(cost_bucket_daily):
        if symbol_norm:
            row_symbol = normalize_strategy_symbol(row.get("symbol"))
            if row_symbol != symbol_norm:
                continue
        value = _row_cost_bps(row)
        if value is None:
            continue
        values.append(value)
        source = str(row.get("source") or row.get("cost_source") or row.get("fallback_level") or "").strip()
        if source:
            sources.add(source)
    if values:
        values.sort()
        index = min(max(int(round((len(values) - 1) * 0.75)), 0), len(values) - 1)
        source_text = "+".join(sorted(sources)) if sources else "cost_bucket_daily"
        return BacktestCost(cost_bps=float(values[index]), cost_model=f"conservative_p75:{source_text}")
    return BacktestCost(cost_bps=float(default_bps), cost_model="conservative_default_30bps")


def _row_cost_bps(row: dict[str, Any]) -> float | None:
    for name in (
        "roundtrip_all_in_cost_bps",
        "selected_total_cost_bps",
        "total_cost_bps_p75",
        "cost_bps",
        "selected_entry_gate_cost_bps",
    ):
        value = float_or_none(row.get(name))
        if value is not None and value >= 0:
            return value
    fee = float_or_none(row.get("fee_bps_p75")) or 0.0
    slippage = float_or_none(row.get("slippage_bps_p75")) or 0.0
    spread = float_or_none(row.get("spread_bps_p75")) or 0.0
    total = fee + slippage + spread
    return total if total > 0 else None
