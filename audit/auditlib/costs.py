# ruff: noqa: E501
"""Explicit transaction-cost arithmetic used by portfolio validation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import polars as pl

from .paths import snapshot_dir


@dataclass(frozen=True)
class CostResult:
    gross_return: float
    turnover: float
    one_way_cost_bps: float
    cost_return: float
    net_return: float
    edge_cost_ratio: float


def apply_costs(*, gross_return: float, turnover: float, one_way_cost_bps: float) -> CostResult:
    """Deduct two one-way legs for each unit of portfolio turnover."""
    if turnover < 0 or one_way_cost_bps < 0:
        raise ValueError("turnover and one_way_cost_bps must be non-negative")
    cost_return = float(turnover) * 2.0 * float(one_way_cost_bps) / 10_000.0
    net_return = float(gross_return) - cost_return
    ratio = float(gross_return) / cost_return if cost_return > 0 else float("inf")
    return CostResult(
        gross_return=float(gross_return),
        turnover=float(turnover),
        one_way_cost_bps=float(one_way_cost_bps),
        cost_return=cost_return,
        net_return=net_return,
        edge_cost_ratio=ratio,
    )


def _read_gold_dataset(name: str) -> pl.DataFrame:
    root = snapshot_dir() / "qlab" / "gold" / name
    files = sorted(
        file
        for file in root.rglob("*.parquet")
        if "._tmp" not in file.parts and not file.name.startswith(".")
    )
    if not files:
        return pl.DataFrame()
    return pl.scan_parquet([str(file) for file in files], missing_columns="insert").collect()


def _weighted_quantile(values: pl.Series, weights: pl.Series, quantile: float) -> float | None:
    value_array = values.cast(pl.Float64, strict=False).to_numpy()
    weight_array = weights.cast(pl.Float64, strict=False).fill_null(0).to_numpy()
    mask = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not mask.any():
        return None
    value_array, weight_array = value_array[mask], weight_array[mask]
    order = np.argsort(value_array, kind="mergesort")
    value_array, weight_array = value_array[order], weight_array[order]
    cumulative = np.cumsum(weight_array)
    index = int(np.searchsorted(cumulative, float(quantile) * cumulative[-1], side="left"))
    return float(value_array[min(index, value_array.size - 1)])


def extract_real_cost_evidence() -> tuple[pl.DataFrame, dict]:
    """Extract fee evidence from read-only V5 fills and cost quantiles from qlab."""
    db_path = snapshot_dir() / "v5" / "db" / "fills.sqlite"
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        fills = pl.read_database(
            "SELECT inst_id, trade_id, ts_ms, cl_ord_id, side, exec_type, fill_px, fill_sz, fee, fee_ccy, source FROM fills",
            connection,
        )
    finally:
        connection.close()
    fills = fills.with_columns(
        pl.col("fill_px").cast(pl.Float64, strict=False),
        pl.col("fill_sz").cast(pl.Float64, strict=False),
        pl.col("fee").cast(pl.Float64, strict=False),
        pl.col("inst_id").str.split("-").list.first().alias("base_ccy"),
    ).with_columns((pl.col("fill_px") * pl.col("fill_sz")).alias("fill_notional_usdt"))
    fills = fills.with_columns(
        pl.when(pl.col("fee_ccy").str.to_uppercase().is_in(["USDT", "USD"]))
        .then(pl.col("fee").abs())
        .when(pl.col("fee_ccy").str.to_uppercase() == pl.col("base_ccy").str.to_uppercase())
        .then(pl.col("fee").abs() * pl.col("fill_px"))
        .otherwise(None)
        .alias("fee_usdt")
    ).with_columns((pl.col("fee_usdt") / pl.col("fill_notional_usdt") * 10_000).alias("fee_bps"))

    bucket = _read_gold_dataset("cost_bucket_daily")
    actual = (
        bucket.filter(pl.col("actual_fill_count").fill_null(0) > 0)
        if not bucket.is_empty()
        else pl.DataFrame()
    )
    symbols = actual["symbol"].unique().sort().to_list() if actual.height else []
    readiness = _read_gold_dataset("cost_bootstrap_readiness")

    fee_values = fills["fee_bps"].drop_nulls()
    fill_fee = {
        "fill_rows": fills.height,
        "fee_converted_rows": fee_values.len(),
        "fee_unconvertible_rows": fills.height - fee_values.len(),
        "fee_bps_p50": float(fee_values.quantile(0.50)) if fee_values.len() else None,
        "fee_bps_p75": float(fee_values.quantile(0.75)) if fee_values.len() else None,
        "fee_bps_p90": float(fee_values.quantile(0.90)) if fee_values.len() else None,
        "first_fill_ts_ms": int(fills["ts_ms"].cast(pl.Int64).min()) if fills.height else None,
        "last_fill_ts_ms": int(fills["ts_ms"].cast(pl.Int64).max()) if fills.height else None,
    }

    empirical: dict[str, float | None] = {}
    if actual.height:
        empirical = {
            "p50": _weighted_quantile(
                actual["total_cost_bps_p50"], actual["actual_fill_count"], 0.50
            ),
            "p75": _weighted_quantile(
                actual["total_cost_bps_p75"], actual["actual_fill_count"], 0.75
            ),
            "p90": _weighted_quantile(
                actual["total_cost_bps_p90"], actual["actual_fill_count"], 0.90
            ),
            "slippage_p50": _weighted_quantile(
                actual["slippage_bps_p50"], actual["actual_fill_count"], 0.50
            ),
            "slippage_p75": _weighted_quantile(
                actual["slippage_bps_p75"], actual["actual_fill_count"], 0.75
            ),
            "slippage_p90": _weighted_quantile(
                actual["slippage_bps_p90"], actual["actual_fill_count"], 0.90
            ),
            "spread_p50": _weighted_quantile(
                actual["spread_bps_p50"], actual["actual_fill_count"], 0.50
            ),
            "spread_p75": _weighted_quantile(
                actual["spread_bps_p75"], actual["actual_fill_count"], 0.75
            ),
            "spread_p90": _weighted_quantile(
                actual["spread_bps_p90"], actual["actual_fill_count"], 0.90
            ),
        }

    # Production config fallback is 10bps fee + 5bps slippage one-way. Never
    # let a sparse/zero field lower the audit below that observed production
    # fallback. This rule is fixed independent of factor results.
    floor_one_way_bps = 15.0
    scenario_values = {
        "optimistic": max(floor_one_way_bps, float(empirical.get("p50") or 0.0)),
        "base": max(floor_one_way_bps, float(empirical.get("p75") or 0.0)),
        "stress": max(floor_one_way_bps, float(empirical.get("p90") or 0.0)),
    }
    # Enforce monotonic scenarios without changing any significance threshold.
    scenario_values["base"] = max(scenario_values["base"], scenario_values["optimistic"])
    scenario_values["stress"] = max(scenario_values["stress"], scenario_values["base"])
    rows = []
    for scenario in ("optimistic", "base", "stress"):
        suffix = {"optimistic": "p50", "base": "p75", "stress": "p90"}[scenario]
        one_way = scenario_values[scenario]
        rows.append(
            {
                "scenario": scenario,
                "quantile": suffix,
                "one_way_cost_bps": one_way,
                "roundtrip_cost_bps": 2.0 * one_way,
                "empirical_total_cost_bps": empirical.get(suffix),
                "empirical_slippage_bps": empirical.get(f"slippage_{suffix}"),
                "empirical_spread_bps": empirical.get(f"spread_{suffix}"),
                "fee_bps": fill_fee.get(f"fee_bps_{suffix}"),
                "production_fallback_floor_bps": floor_one_way_bps,
                "actual_cost_bucket_rows": actual.height,
                "v5_fill_rows": fills.height,
                "actual_cost_symbols": len(symbols),
                "audit_symbols": 93,
                "symbol_coverage": len(symbols) / 93.0,
                "evidence_status": "INSUFFICIENT_FOR_BROAD_UNIVERSE"
                if len(symbols) < 20
                else "SUFFICIENT",
                "source": "V5 fills.sqlite + quant-lab cost_bucket_daily actual_fills",
            }
        )
    metadata = {
        "fill_fee": fill_fee,
        "empirical": empirical,
        "actual_symbols": symbols,
        "actual_bucket_rows": actual.height,
        "readiness_rows": readiness.to_dicts(),
        "direct_market_order_arrival_slippage_matches": 0,
        "limitation": "All V5 orders in the snapshot are market orders with no stored order price; direct fill-vs-arrival slippage cannot be reconstructed from orders.sqlite. qlab cost buckets supply the available slippage/spread evidence.",
    }
    return pl.DataFrame(rows), metadata
