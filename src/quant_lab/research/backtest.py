import math
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class OOSResearchStats:
    oos_sharpe: float
    oos_sortino: float | None
    oos_cagr: float | None
    oos_max_drawdown: float
    profit_factor: float | None
    turnover: float
    edge_cost_ratio: float
    cost_ratio: float | None
    profitable_folds_ratio: float
    train_oos_decay: float
    period_count: int
    warnings: list[str]


def simulate_long_only_oos(
    dataset: pl.DataFrame,
    *,
    top_quantile: float,
    cost_quantile: str,
) -> OOSResearchStats:
    if dataset.is_empty() or "alpha_score" not in dataset.columns:
        return _empty_stats(["insufficient_samples"])
    period_returns: list[float] = []
    gross_returns: list[float] = []
    cost_returns: list[float] = []
    previous_symbols: set[str] | None = None
    turnovers: list[float] = []

    for _period, group in dataset.group_by("decision_ts", maintain_order=True):
        valid = group.filter(
            pl.col("alpha_score").is_not_null() & pl.col("forward_return").is_not_null()
        )
        if valid.is_empty():
            continue
        selected = valid.sort("alpha_score", descending=True).head(
            max(1, math.ceil(valid.height * top_quantile))
        )
        symbols = {str(symbol) for symbol in selected["symbol"].to_list()}
        gross = float(selected["forward_return"].mean() or 0.0)
        cost_bps = _cost_bps(selected, cost_quantile)
        after_cost = gross - cost_bps / 10_000.0
        gross_returns.append(gross)
        cost_returns.append(cost_bps / 10_000.0)
        period_returns.append(after_cost)
        if previous_symbols is not None:
            changed = len(symbols.symmetric_difference(previous_symbols))
            denominator = max(len(symbols.union(previous_symbols)), 1)
            turnovers.append(changed / denominator)
        previous_symbols = symbols

    if not period_returns:
        return _empty_stats(["insufficient_samples"])

    sharpe = _sharpe(period_returns)
    sortino = _sortino(period_returns)
    max_drawdown = _max_drawdown(period_returns)
    profit_factor = _profit_factor(period_returns)
    mean_gross_bps = abs(_mean(gross_returns)) * 10_000.0
    mean_cost_bps = _mean(cost_returns) * 10_000.0
    edge_cost_ratio = mean_gross_bps / mean_cost_bps if mean_cost_bps > 0 else mean_gross_bps
    cost_ratio = mean_cost_bps / mean_gross_bps if mean_gross_bps > 0 else None
    fold_stats = _fold_stats(period_returns)
    return OOSResearchStats(
        oos_sharpe=sharpe,
        oos_sortino=sortino,
        oos_cagr=None,
        oos_max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        turnover=_mean(turnovers) if turnovers else 0.0,
        edge_cost_ratio=edge_cost_ratio,
        cost_ratio=cost_ratio,
        profitable_folds_ratio=fold_stats["profitable_folds_ratio"],
        train_oos_decay=fold_stats["train_oos_decay"],
        period_count=len(period_returns),
        warnings=fold_stats["warnings"],
    )


def _cost_bps(df: pl.DataFrame, quantile: str) -> float:
    column = f"total_cost_bps_{quantile}"
    if column in df.columns:
        return float(df[column].mean() or 0.0)
    if "cost_bps" in df.columns:
        return float(df["cost_bps"].mean() or 0.0)
    return 0.0


def _empty_stats(warnings: list[str]) -> OOSResearchStats:
    return OOSResearchStats(
        oos_sharpe=0.0,
        oos_sortino=None,
        oos_cagr=None,
        oos_max_drawdown=0.0,
        profit_factor=None,
        turnover=0.0,
        edge_cost_ratio=0.0,
        cost_ratio=None,
        profitable_folds_ratio=0.0,
        train_oos_decay=1.0,
        period_count=0,
        warnings=warnings,
    )


def _sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    std = _std(values)
    if std == 0:
        return 0.0
    return _mean(values) / std * math.sqrt(len(values))


def _sortino(values: list[float]) -> float | None:
    downside = [value for value in values if value < 0]
    if len(downside) < 2:
        return None
    std = _std(downside)
    if std == 0:
        return None
    return _mean(values) / std * math.sqrt(len(values))


def _max_drawdown(values: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def _profit_factor(values: list[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value < 0))
    if losses == 0:
        return None
    return gains / losses


def _fold_stats(values: list[float]) -> dict[str, Any]:
    if len(values) < 10:
        return {
            "profitable_folds_ratio": 0.0,
            "train_oos_decay": 1.0,
            "warnings": ["insufficient_folds"],
        }
    fold_count = min(5, max(3, len(values) // 10))
    size = max(1, len(values) // fold_count)
    profitable = 0
    decays: list[float] = []
    for index in range(fold_count):
        fold = values[index * size : (index + 1) * size]
        if len(fold) < 3:
            continue
        split = max(1, int(len(fold) * 0.7))
        train = fold[:split]
        oos = fold[split:]
        if not oos:
            continue
        train_mean = _mean(train)
        oos_mean = _mean(oos)
        if oos_mean > 0:
            profitable += 1
        if train_mean != 0:
            decays.append(max((train_mean - oos_mean) / abs(train_mean), 0.0))
    denominator = max(fold_count, 1)
    return {
        "profitable_folds_ratio": profitable / denominator,
        "train_oos_decay": min(_mean(decays), 1.0) if decays else 1.0,
        "warnings": [],
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
