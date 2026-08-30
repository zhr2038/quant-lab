from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import polars as pl

from quant_lab.backtest.datasets import float_or_none, normalize_strategy_symbol, rows


@dataclass(frozen=True)
class BacktestCost:
    cost_bps: float
    cost_model: str


@dataclass(frozen=True)
class BacktestCostLookup:
    by_symbol: dict[str, BacktestCost]
    all_symbols: BacktestCost
    default: BacktestCost

    def for_symbol(self, symbol: str | None = None) -> BacktestCost:
        if not symbol:
            return self.all_symbols
        return self.by_symbol.get(normalize_strategy_symbol(symbol), self.default)


def build_conservative_cost_lookup(
    cost_bucket_daily: pl.DataFrame | None,
    *,
    default_bps: float = 30.0,
) -> BacktestCostLookup:
    """Materialize the cost table once for repeated symbol lookups."""

    values_by_symbol: dict[str, list[float]] = defaultdict(list)
    sources_by_symbol: dict[str, set[str]] = defaultdict(set)
    all_values: list[float] = []
    all_sources: set[str] = set()
    for row in rows(cost_bucket_daily):
        value = _row_cost_bps(row)
        if value is None:
            continue
        symbol = normalize_strategy_symbol(row.get("symbol"))
        source = str(
            row.get("source") or row.get("cost_source") or row.get("fallback_level") or ""
        ).strip()
        values_by_symbol[symbol].append(value)
        all_values.append(value)
        if source:
            sources_by_symbol[symbol].add(source)
            all_sources.add(source)

    default = BacktestCost(
        cost_bps=float(default_bps),
        cost_model="conservative_default_30bps",
    )
    return BacktestCostLookup(
        by_symbol={
            symbol: _cost_from_values(values, sources_by_symbol[symbol], default=default)
            for symbol, values in values_by_symbol.items()
        },
        all_symbols=_cost_from_values(all_values, all_sources, default=default),
        default=default,
    )


def conservative_cost_for_symbol(
    cost_bucket_daily: pl.DataFrame | None,
    *,
    symbol: str | None = None,
    default_bps: float = 30.0,
) -> BacktestCost:
    return build_conservative_cost_lookup(
        cost_bucket_daily,
        default_bps=default_bps,
    ).for_symbol(symbol)


def _cost_from_values(
    values: list[float],
    sources: set[str],
    *,
    default: BacktestCost,
) -> BacktestCost:
    if not values:
        return default
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * 0.75)), 0), len(ordered) - 1)
    source_text = "+".join(sorted(sources)) if sources else "cost_bucket_daily"
    return BacktestCost(
        cost_bps=float(ordered[index]),
        cost_model=f"conservative_p75:{source_text}",
    )


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
