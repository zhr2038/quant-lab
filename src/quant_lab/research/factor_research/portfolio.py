from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

import polars as pl
from pydantic import Field

from quant_lab.research.factor_research.contracts import StrictModel


class PortfolioRule(StrictModel):
    portfolio_rule_id: str = Field(min_length=1, max_length=180)
    selection_mode: Literal["cross_sectional_rank", "absolute_timing"] = "cross_sectional_rank"
    top_n: int = Field(ge=1, le=200)
    weighting: Literal["equal", "score"]
    holding_period_bars: int = Field(ge=1, le=1095 * 24)
    initial_capital_usdt: float = Field(gt=0, le=1_000_000)
    min_order_usdt: float = Field(gt=0, le=10_000)
    max_drawdown_limit: float = Field(gt=0, le=1)
    annual_periods: int = Field(ge=1, le=365 * 24 * 60)


@dataclass(frozen=True)
class FactorPortfolioValidation:
    portfolio_rule_id: str
    period_count: int
    gross_return: float
    net_return: float
    turnover: float
    fees: float
    slippage: float
    edge_cost_ratio: float
    sharpe: float
    sortino: float | None
    max_drawdown: float
    calmar: float | None
    beta: float
    excess_vs_btc: float
    excess_vs_universe: float
    symbol_contribution_json: str
    concentration_hhi: float
    max_symbol_contribution_share: float
    average_cash_residual: float
    validation_net_return: float | None
    blind_net_return: float | None
    portfolio_validity: str
    deployment_readiness: str
    decision: str
    blockers: tuple[str, ...]
    research_only: bool = True
    live_order_effect: str = "none"
    max_live_notional_usdt: float = 0.0


def compute_factor_portfolio(
    dataset: pl.DataFrame,
    *,
    rule: PortfolioRule,
    signal_validity: Literal["PASS", "FAIL", "INCONCLUSIVE"],
) -> FactorPortfolioValidation:
    required = {"decision_ts", "symbol", "alpha_score", "forward_return"}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"factor portfolio missing columns: {','.join(missing)}")
    period_results: list[dict[str, object]] = []
    ordered = dataset.sort(["decision_ts", "alpha_score"], descending=[False, True])
    for period_index, (_period, group) in enumerate(
        ordered.group_by("decision_ts", maintain_order=True)
    ):
        if period_index % rule.holding_period_bars != 0:
            continue
        period_results.append(_portfolio_period(group, rule=rule))
    net_returns = [float(row["net_return"]) for row in period_results]
    gross_returns = [float(row["gross_return"]) for row in period_results]
    btc_returns = [float(row["btc_return"]) for row in period_results]
    universe_returns = [float(row["universe_return"]) for row in period_results]
    fees = sum(float(row["fees"]) for row in period_results)
    slippage = sum(float(row["slippage"]) for row in period_results)
    total_cost = fees + slippage
    gross_return = _compound(gross_returns)
    net_return = _compound(net_returns)
    btc_return = _compound(btc_returns)
    universe_return = _compound(universe_returns)
    contributions = _aggregate_contributions(period_results)
    contribution_abs = sum(abs(value) for value in contributions.values())
    shares = {
        symbol: abs(value) / contribution_abs if contribution_abs > 0 else 0.0
        for symbol, value in contributions.items()
    }
    max_share = max(shares.values(), default=0.0)
    validation_net = _split_return(period_results, "VALIDATION")
    blind_net = _split_return(period_results, "BLIND_CONFIRMATORY")
    max_drawdown = _max_drawdown(net_returns)
    blockers: list[str] = []
    if signal_validity != "PASS":
        blockers.append("signal_not_valid")
    if net_return <= 0:
        blockers.append("long_only_net_return_not_positive")
    edge_cost_ratio = abs(gross_return) / total_cost if total_cost > 0 else 0.0
    if edge_cost_ratio <= 1.5:
        blockers.append("edge_cost_ratio_not_above_1_5")
    if validation_net is None or blind_net is None:
        blockers.append("validation_or_blind_split_missing")
    elif validation_net < 0 or blind_net < 0:
        blockers.append("validation_or_blind_loss")
    if net_return < btc_return:
        blockers.append("underperforms_btc")
    if net_return < universe_return:
        blockers.append("underperforms_dynamic_universe")
    if max_drawdown > rule.max_drawdown_limit:
        blockers.append("max_drawdown_limit_exceeded")
    if max_share > 0.50:
        blockers.append("single_symbol_contribution_above_50pct")
    if signal_validity == "PASS" and blockers:
        portfolio_validity = "FAIL"
        decision = "PORTFOLIO_FAIL"
    elif not blockers:
        portfolio_validity = "PASS"
        decision = "SIGNAL_VALID"
    else:
        portfolio_validity = "INCONCLUSIVE"
        decision = "INCONCLUSIVE"
    return FactorPortfolioValidation(
        portfolio_rule_id=rule.portfolio_rule_id,
        period_count=len(period_results),
        gross_return=gross_return,
        net_return=net_return,
        turnover=sum(float(row["turnover"]) for row in period_results),
        fees=fees,
        slippage=slippage,
        edge_cost_ratio=edge_cost_ratio,
        sharpe=_sharpe(net_returns, annual_periods=rule.annual_periods),
        sortino=_sortino(net_returns, annual_periods=rule.annual_periods),
        max_drawdown=max_drawdown,
        calmar=_calmar(net_returns, max_drawdown, annual_periods=rule.annual_periods),
        beta=_beta(net_returns, btc_returns),
        excess_vs_btc=net_return - btc_return,
        excess_vs_universe=net_return - universe_return,
        symbol_contribution_json=json.dumps(contributions, sort_keys=True, separators=(",", ":")),
        concentration_hhi=sum(share * share for share in shares.values()),
        max_symbol_contribution_share=max_share,
        average_cash_residual=_mean([float(row["cash_residual"]) for row in period_results]),
        validation_net_return=validation_net,
        blind_net_return=blind_net,
        portfolio_validity=portfolio_validity,
        deployment_readiness="BLOCKED_REQUIRES_OVERFIT_AND_FORWARD_REVIEW",
        decision=decision,
        blockers=tuple(sorted(set(blockers))),
    )


def _portfolio_period(group: pl.DataFrame, *, rule: PortfolioRule) -> dict[str, object]:
    if rule.selection_mode == "absolute_timing":
        signal = _float(group.get_column("alpha_score").drop_nulls().mean()) or 0.0
        ranked = group if signal > 0 else group.head(0)
    else:
        ranked = group.sort("alpha_score", descending=True).head(rule.top_n)
    rows = ranked.to_dicts()
    raw_weights = _raw_weights(rows, rule=rule)
    contributions: dict[str, float] = {}
    gross = 0.0
    fees = 0.0
    slippage = 0.0
    invested = 0.0
    for row, target_weight in zip(rows, raw_weights, strict=True):
        tradable = bool(row.get("tradable", True))
        if not tradable or target_weight * rule.initial_capital_usdt < rule.min_order_usdt:
            continue
        forward_return = _float(row.get("forward_return")) or 0.0
        fee_bps = (_float(row.get("entry_fee_bps")) or 0.0) + (
            _float(row.get("exit_fee_bps")) or 0.0
        )
        slippage_bps = (_float(row.get("entry_slippage_bps")) or 0.0) + (
            _float(row.get("exit_slippage_bps")) or 0.0
        )
        if fee_bps == 0.0 and slippage_bps == 0.0:
            slippage_bps = _float(row.get("cost_bps")) or 0.0
        gross_contribution = target_weight * forward_return
        fee = target_weight * fee_bps / 10_000.0
        slip = target_weight * slippage_bps / 10_000.0
        net_contribution = gross_contribution - fee - slip
        symbol = str(row.get("symbol") or "UNKNOWN")
        contributions[symbol] = contributions.get(symbol, 0.0) + net_contribution
        gross += gross_contribution
        fees += fee
        slippage += slip
        invested += target_weight
    first = rows[0] if rows else {}
    return {
        "gross_return": gross,
        "net_return": gross - fees - slippage,
        "fees": fees,
        "slippage": slippage,
        "turnover": invested * 2.0,
        "cash_residual": max(0.0, 1.0 - invested),
        "btc_return": _float(first.get("btc_forward_return")) or 0.0,
        "universe_return": _float(first.get("universe_forward_return")) or 0.0,
        "split": str(first.get("split") or "UNKNOWN").upper(),
        "contributions": contributions,
    }


def _raw_weights(rows: list[dict[str, object]], *, rule: PortfolioRule) -> list[float]:
    if not rows:
        return []
    if rule.selection_mode == "absolute_timing":
        return [1.0 / len(rows) for _ in rows]
    if rule.weighting == "equal":
        return [1.0 / rule.top_n for _ in rows]
    positive_scores = [max(_float(row.get("alpha_score")) or 0.0, 0.0) for row in rows]
    denominator = sum(positive_scores)
    if denominator <= 0:
        return [1.0 / rule.top_n for _ in rows]
    slot_scale = len(rows) / rule.top_n
    return [score / denominator * slot_scale for score in positive_scores]


def _aggregate_contributions(period_results: list[dict[str, object]]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for result in period_results:
        contributions = result.get("contributions")
        if not isinstance(contributions, dict):
            continue
        for symbol, value in contributions.items():
            combined[str(symbol)] = combined.get(str(symbol), 0.0) + float(value)
    return dict(sorted(combined.items()))


def _split_return(period_results: list[dict[str, object]], split: str) -> float | None:
    values = [
        float(row["net_return"])
        for row in period_results
        if str(row.get("split") or "").upper() == split
    ]
    return _compound(values) if values else None


def _compound(values: list[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + value
    return equity - 1.0


def _sharpe(values: list[float], *, annual_periods: int) -> float:
    standard_deviation = _std(values)
    if standard_deviation <= 0:
        return 0.0
    return _mean(values) / standard_deviation * math.sqrt(annual_periods)


def _sortino(values: list[float], *, annual_periods: int) -> float | None:
    downside = [min(value, 0.0) for value in values]
    downside_deviation = math.sqrt(_mean([value * value for value in downside]))
    if downside_deviation <= 0:
        return None
    return _mean(values) / downside_deviation * math.sqrt(annual_periods)


def _max_drawdown(values: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0:
            drawdown = max(drawdown, (peak - equity) / peak)
    return drawdown


def _calmar(values: list[float], drawdown: float, *, annual_periods: int) -> float | None:
    if not values or drawdown <= 0:
        return None
    annualized = (1.0 + _compound(values)) ** (annual_periods / len(values)) - 1.0
    return annualized / drawdown


def _beta(values: list[float], benchmark: list[float]) -> float:
    pairs = list(zip(values, benchmark, strict=False))
    if len(pairs) < 2:
        return 0.0
    benchmark_mean = _mean([item[1] for item in pairs])
    value_mean = _mean([item[0] for item in pairs])
    variance = sum((item[1] - benchmark_mean) ** 2 for item in pairs)
    if variance <= 0:
        return 0.0
    covariance = sum(
        (value - value_mean) * (reference - benchmark_mean) for value, reference in pairs
    )
    return covariance / variance


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _float(value: object) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None
