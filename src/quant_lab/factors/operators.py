from __future__ import annotations

from collections.abc import Sequence

import polars as pl

EPSILON = 1e-12


def numeric(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False)


def safe_divide(numerator: pl.Expr, denominator: pl.Expr, *, epsilon: float = EPSILON) -> pl.Expr:
    return pl.when(denominator.abs() > epsilon).then(numerator / denominator).otherwise(None)


def winsorize_expr(value: pl.Expr, *, lower: float, upper: float) -> pl.Expr:
    return pl.when(value < lower).then(lower).when(value > upper).then(upper).otherwise(value)


def cross_sectional_zscore(value_column: str, group_columns: Sequence[str]) -> pl.Expr:
    value = numeric(value_column)
    mean = value.mean().over(list(group_columns))
    std = value.std().over(list(group_columns))
    return pl.when(std > 0).then((value - mean) / std).otherwise(None)


def cross_sectional_rank_pct(value_column: str, group_columns: Sequence[str]) -> pl.Expr:
    value = numeric(value_column)
    rank = value.rank("average").over(list(group_columns))
    count = value.count().over(list(group_columns))
    return pl.when(count > 1).then((rank - 1.0) / (count - 1.0)).otherwise(0.5)
