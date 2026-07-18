"""Partition-scoped symbol concentration for Alpha Audit v2.1."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl

from .portfolio_backtest import MarketMatrix, Simulation


def _rows_for_partition(
    *,
    structure_id: str,
    factor_id: str,
    partition: str,
    symbols: list[str],
    gross: np.ndarray,
    net: np.ndarray,
) -> list[dict]:
    active = [
        (symbol, float(gross_value), float(net_value))
        for symbol, gross_value, net_value in zip(symbols, gross, net, strict=True)
        if abs(float(gross_value)) > 1e-18 or abs(float(net_value)) > 1e-18
    ]
    if not active:
        return []
    absolute_denominator = sum(abs(item[2]) for item in active)
    signed_denominator = sum(item[2] for item in active)
    shares = [
        abs(item[2]) / absolute_denominator if absolute_denominator else 0.0
        for item in active
    ]
    order = np.argsort(-np.asarray(shares, dtype=float), kind="mergesort")
    top_symbol = active[int(order[0])][0]
    top_symbol_share = shares[int(order[0])]
    top_3_symbol_share = float(sum(shares[int(index)] for index in order[:3]))
    hhi = float(sum(share**2 for share in shares))
    rows: list[dict] = []
    for (symbol, gross_value, net_value), absolute_share in zip(
        active, shares, strict=True
    ):
        rows.append(
            {
                "structure_id": structure_id,
                "factor_id": factor_id,
                "partition": partition,
                "symbol": symbol,
                "symbol_gross_contribution": gross_value,
                "symbol_net_contribution": net_value,
                "absolute_contribution_share": absolute_share,
                "signed_contribution_share": (
                    net_value / signed_denominator if signed_denominator else None
                ),
                "top_symbol": top_symbol,
                "top_symbol_share": top_symbol_share,
                "top_3_symbol_share": top_3_symbol_share,
                "hhi": hhi,
            }
        )
    return rows


def partition_symbol_contributions(
    simulation: Simulation,
    market: MarketMatrix,
    periods: Mapping[str, tuple[int, int]],
    *,
    structure_id: str,
    factor_id: str,
    one_way_cost_bps: float,
) -> pl.DataFrame:
    if simulation.contribution_matrix is None or simulation.traded_by_symbol is None:
        raise ValueError("simulation does not contain partition attribution matrices")
    rows: list[dict] = []
    for partition, (left, right) in periods.items():
        gross = simulation.contribution_matrix[left:right].sum(axis=0)
        costs = (
            simulation.traded_by_symbol[left:right].sum(axis=0)
            * float(one_way_cost_bps)
            / 10_000.0
        )
        rows.extend(
            _rows_for_partition(
                structure_id=structure_id,
                factor_id=factor_id,
                partition=partition,
                symbols=market.symbols,
                gross=gross,
                net=gross - costs,
            )
        )
    return pl.DataFrame(rows)


def forward_symbol_contributions(
    trades: pl.DataFrame, *, structure_id: str, factor_id: str
) -> pl.DataFrame:
    if trades.is_empty():
        return pl.DataFrame(
            [
                {
                    "structure_id": structure_id,
                    "factor_id": factor_id,
                    "partition": "forward",
                    "symbol": "",
                    "symbol_gross_contribution": 0.0,
                    "symbol_net_contribution": 0.0,
                    "absolute_contribution_share": 0.0,
                    "signed_contribution_share": None,
                    "top_symbol": "",
                    "top_symbol_share": 0.0,
                    "top_3_symbol_share": 0.0,
                    "hhi": 0.0,
                }
            ]
        )
    grouped = (
        trades.filter(pl.col("status") == "CLOSED")
        .group_by("symbol")
        .agg(
            pl.col("weighted_gross_contribution")
            .sum()
            .alias("symbol_gross_contribution"),
            pl.col("weighted_net_contribution")
            .sum()
            .alias("symbol_net_contribution"),
        )
        .sort("symbol")
    )
    if grouped.is_empty():
        return forward_symbol_contributions(
            pl.DataFrame(), structure_id=structure_id, factor_id=factor_id
        )
    return pl.DataFrame(
        _rows_for_partition(
            structure_id=structure_id,
            factor_id=factor_id,
            partition="forward",
            symbols=grouped["symbol"].to_list(),
            gross=grouped["symbol_gross_contribution"].to_numpy(),
            net=grouped["symbol_net_contribution"].to_numpy(),
        )
    )
