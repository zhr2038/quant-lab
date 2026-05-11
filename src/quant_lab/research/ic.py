import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class ICStats:
    mean: float
    tstat: float
    period_count: int
    status: str = "ok"


def compute_ic(df: pl.DataFrame, feature_column: str = "alpha_score") -> ICStats:
    return _compute_period_stats(df, feature_column, rank=False)


def compute_rank_ic(df: pl.DataFrame, feature_column: str = "alpha_score") -> ICStats:
    return _compute_period_stats(df, feature_column, rank=True)


def compute_ic_tstat(values: list[float]) -> float:
    return _tstat(values)


def compute_rank_ic_tstat(values: list[float]) -> float:
    return _tstat(values)


def compute_by_symbol_ic(df: pl.DataFrame, feature_column: str = "alpha_score") -> dict[str, float]:
    if df.is_empty() or feature_column not in df.columns:
        return {}
    results: dict[str, float] = {}
    for symbol_key, group in df.group_by("symbol", maintain_order=True):
        symbol = symbol_key[0] if isinstance(symbol_key, tuple) else symbol_key
        pairs = _valid_pairs(group.to_dicts(), feature_column)
        results[str(symbol)] = _pearson([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    return results


def _compute_period_stats(df: pl.DataFrame, feature_column: str, *, rank: bool) -> ICStats:
    if df.is_empty() or feature_column not in df.columns:
        return ICStats(mean=0.0, tstat=0.0, period_count=0, status="insufficient_samples")
    values: list[float] = []
    for _period, group in df.group_by("decision_ts", maintain_order=True):
        pairs = _valid_pairs(group.to_dicts(), feature_column)
        if len(pairs) < 3:
            continue
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]
        if rank:
            x_values = _ranks(x_values)
            y_values = _ranks(y_values)
        corr = _pearson(x_values, y_values)
        if math.isfinite(corr):
            values.append(corr)
    if not values:
        return ICStats(mean=0.0, tstat=0.0, period_count=0, status="insufficient_samples")
    return ICStats(mean=sum(values) / len(values), tstat=_tstat(values), period_count=len(values))


def _valid_pairs(rows: list[dict[str, Any]], feature_column: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        x_value = row.get(feature_column)
        y_value = row.get("forward_return")
        if x_value is None or y_value is None:
            continue
        x = float(x_value)
        y = float(y_value)
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    return pairs


def _tstat(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(len(values))


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True))
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _ranks(values: list[float]) -> list[float]:
    grouped: dict[float, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[value].append(index)
    ranks = [0.0] * len(values)
    current = 1
    for value in sorted(grouped):
        indexes = grouped[value]
        average_rank = (current + current + len(indexes) - 1) / 2.0
        for index in indexes:
            ranks[index] = average_rank
        current += len(indexes)
    return ranks
