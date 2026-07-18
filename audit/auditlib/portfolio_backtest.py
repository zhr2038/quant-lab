"""Causal long-only spot portfolio simulation for the alpha audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from .ic_stats import newey_west_tstat
from .multiple_testing import two_sided_pvalue


@dataclass(frozen=True)
class MarketMatrix:
    timestamps: list[datetime]
    symbols: list[str]
    returns: np.ndarray
    signals: dict[str, np.ndarray]
    universe_members: dict[str, dict[object, np.ndarray]]
    btc_up: np.ndarray
    btc_returns: np.ndarray


@dataclass
class Simulation:
    gross_returns: np.ndarray
    turnover: np.ndarray
    traded_fraction: np.ndarray
    contribution_by_symbol: np.ndarray
    contribution_matrix: np.ndarray | None = None
    traded_by_symbol: np.ndarray | None = None


def build_market_matrix(
    bars: pl.DataFrame,
    signals_by_factor: dict[str, pl.DataFrame],
    universes: pl.DataFrame,
) -> MarketMatrix:
    close_wide = bars.pivot(
        on="symbol", index="ts", values="close", aggregate_function="last"
    ).sort("ts")
    timestamps = close_wide["ts"].to_list()
    symbols = sorted(column for column in close_wide.columns if column != "ts")
    close = close_wide.select(symbols).to_numpy()
    returns = np.zeros_like(close, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[1:] = close[1:] / close[:-1] - 1.0
    returns[~np.isfinite(returns)] = 0.0

    timeline = pl.DataFrame({"feature_ts": timestamps})
    signal_matrices: dict[str, np.ndarray] = {}
    for factor_id, signal in signals_by_factor.items():
        wide = signal.pivot(
            on="symbol", index="feature_ts", values="signal", aggregate_function="last"
        )
        aligned = timeline.join(wide, on="feature_ts", how="left")
        for symbol in symbols:
            if symbol not in aligned.columns:
                aligned = aligned.with_columns(pl.lit(None, dtype=pl.Float64).alias(symbol))
        signal_matrices[factor_id] = aligned.select(symbols).to_numpy()

    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    members: dict[str, dict[object, np.ndarray]] = {}
    for universe_name in universes["universe"].unique().sort().to_list():
        by_date: dict[object, np.ndarray] = {}
        subset = universes.filter(pl.col("universe") == universe_name)
        for date_key, frame in subset.group_by("date", maintain_order=True):
            date_value = date_key[0] if isinstance(date_key, tuple) else date_key
            indices = [
                symbol_index[symbol]
                for symbol in frame["symbol"].to_list()
                if symbol in symbol_index
            ]
            by_date[date_value] = np.asarray(indices, dtype=int)
        members[str(universe_name)] = by_date

    btc_index = symbol_index["BTC-USDT"]
    btc_close = close[:, btc_index]
    window = 24 * 60
    btc_up = np.zeros(len(timestamps), dtype=bool)
    valid_btc = np.isfinite(btc_close)
    cumulative = np.cumsum(np.where(valid_btc, btc_close, 0.0))
    for index in range(window - 1, len(timestamps)):
        total = cumulative[index] - (cumulative[index - window] if index >= window else 0.0)
        mean = total / window
        btc_up[index] = valid_btc[index] and btc_close[index] >= mean
    return MarketMatrix(
        timestamps=timestamps,
        symbols=symbols,
        returns=returns,
        signals=signal_matrices,
        universe_members=members,
        btc_up=btc_up,
        btc_returns=returns[:, btc_index],
    )


def _capped_weights(scores: np.ndarray, top_n: int, max_weight: float, method: str) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")[:top_n]
    if order.size == 0:
        return np.empty(0)
    if method == "equal":
        weights = np.full(order.size, 1.0 / order.size)
    elif method == "score":
        selected = scores[order]
        raw = selected - np.min(selected) + 1e-12
        weights = raw / raw.sum()
    else:
        raise ValueError(f"unsupported weighting method: {method}")
    cap = max(float(max_weight), 1.0 / order.size)
    fixed = np.zeros(order.size, dtype=bool)
    for _ in range(order.size + 1):
        over = (~fixed) & (weights > cap + 1e-14)
        if not over.any():
            break
        weights[over] = cap
        fixed |= over
        free = ~fixed
        remaining = 1.0 - weights[fixed].sum()
        if not free.any():
            break
        free_total = weights[free].sum()
        weights[free] = (
            remaining / free.sum() if free_total <= 0 else remaining * weights[free] / free_total
        )
    out = np.zeros(scores.size, dtype=float)
    out[order] = weights
    return out


def simulate_long_only(
    market: MarketMatrix,
    *,
    factor_id: str,
    universe: str,
    top_n: int,
    weighting: str,
    max_weight: float,
    rebalance_hours: int,
    staggered: bool,
    btc_filter: bool,
) -> Simulation:
    signal_matrix = market.signals[factor_id]
    n_times, n_symbols = signal_matrix.shape
    sleeve_count = 3 if staggered else 1
    sleeves = np.zeros((sleeve_count, n_symbols), dtype=float)
    gross = np.zeros(n_times, dtype=float)
    turnover = np.zeros(n_times, dtype=float)
    traded = np.zeros(n_times, dtype=float)
    contributions = np.zeros(n_symbols, dtype=float)
    contribution_matrix = np.zeros((n_times, n_symbols), dtype=float)
    traded_by_symbol = np.zeros((n_times, n_symbols), dtype=float)

    # First event requires both warmup-complete signals and an available causal universe.
    anchor = None
    for decision_index in range(1, n_times):
        eligible = market.universe_members[universe].get(market.timestamps[decision_index].date())
        if eligible is None or eligible.size < top_n:
            continue
        values = signal_matrix[decision_index - 1, eligible]
        if np.isfinite(values).sum() >= top_n:
            anchor = decision_index
            break
    if anchor is None:
        return Simulation(
            gross,
            turnover,
            traded,
            contributions,
            contribution_matrix,
            traded_by_symbol,
        )

    offsets = [int(round(index * rebalance_hours / sleeve_count)) for index in range(sleeve_count)]
    for time_index in range(1, n_times):
        current = sleeves.mean(axis=0)
        contribution = current * market.returns[time_index]
        gross[time_index] = contribution.sum()
        contributions += contribution
        contribution_matrix[time_index] = contribution

        due_sleeves = [
            sleeve
            for sleeve, offset in enumerate(offsets)
            if time_index >= anchor + offset
            and (time_index - anchor - offset) % rebalance_hours == 0
        ]
        if not due_sleeves:
            continue
        before = current.copy()
        feature_index = time_index - 1
        for sleeve in due_sleeves:
            target = np.zeros(n_symbols, dtype=float)
            filter_allows = (not btc_filter) or bool(market.btc_up[feature_index])
            eligible = market.universe_members[universe].get(market.timestamps[time_index].date())
            if filter_allows and eligible is not None:
                values = signal_matrix[feature_index, eligible]
                valid = np.isfinite(values)
                if valid.sum() >= top_n:
                    valid_indices = eligible[valid]
                    local = _capped_weights(values[valid], top_n, max_weight, weighting)
                    target[valid_indices] = local
            sleeves[sleeve] = target
        after = sleeves.mean(axis=0)
        delta = np.abs(after - before)
        turnover[time_index] = 0.5 * delta.sum()
        traded[time_index] = delta.sum()
        traded_by_symbol[time_index] = delta
    return Simulation(
        gross,
        turnover,
        traded,
        contributions,
        contribution_matrix,
        traded_by_symbol,
    )


def _daily_compound(
    values: np.ndarray, timestamps: list[datetime]
) -> tuple[list[datetime], np.ndarray]:
    dates: list[datetime] = []
    compounded: list[float] = []
    start = 0
    while start < len(timestamps):
        day = timestamps[start].date()
        end = start + 1
        while end < len(timestamps) and timestamps[end].date() == day:
            end += 1
        dates.append(timestamps[end - 1])
        compounded.append(float(np.prod(1.0 + values[start:end]) - 1.0))
        start = end
    return dates, np.asarray(compounded)


def performance_metrics(
    simulation: Simulation,
    market: MarketMatrix,
    *,
    one_way_cost_bps: float,
    fee_bps: float,
    start_index: int,
    end_index: int,
    rebalance_hours: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    gross = simulation.gross_returns[start_index:end_index]
    traded = simulation.traded_fraction[start_index:end_index]
    turnover = simulation.turnover[start_index:end_index]
    timestamps = market.timestamps[start_index:end_index]
    btc = market.btc_returns[start_index:end_index]
    total_cost = traded * one_way_cost_bps / 10_000.0
    fee_cost = traded * min(fee_bps, one_way_cost_bps) / 10_000.0
    slippage_cost = total_cost - fee_cost
    net = gross - total_cost
    gross_equity = np.cumprod(1.0 + gross)
    net_equity = np.cumprod(1.0 + net)
    running_max = np.maximum.accumulate(net_equity) if net_equity.size else np.array([])
    drawdown = net_equity / running_max - 1.0 if net_equity.size else np.array([])
    hours = max(len(net), 1)
    years = hours / (365.25 * 24)
    total_return = float(net_equity[-1] - 1.0) if net_equity.size else 0.0
    gross_total_return = float(gross_equity[-1] - 1.0) if gross_equity.size else 0.0
    cagr = (
        float(net_equity[-1] ** (1.0 / years) - 1.0)
        if years > 0 and net_equity.size and net_equity[-1] > 0
        else -1.0
    )
    annual_vol = float(np.std(net, ddof=1) * math.sqrt(365.25 * 24)) if len(net) > 1 else 0.0
    sharpe = (
        float(np.mean(net) / np.std(net, ddof=1) * math.sqrt(365.25 * 24))
        if len(net) > 1 and np.std(net, ddof=1) > 0
        else 0.0
    )
    downside = net[net < 0]
    sortino = (
        float(np.mean(net) / np.std(downside, ddof=1) * math.sqrt(365.25 * 24))
        if downside.size > 1 and np.std(downside, ddof=1) > 0
        else 0.0
    )
    max_drawdown = float(drawdown.min()) if drawdown.size else 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    daily_dates, daily_net = _daily_compound(net, timestamps)
    _, daily_gross = _daily_compound(gross, timestamps)
    win_rate = float(np.mean(daily_net > 0)) if daily_net.size else 0.0
    profit_factor = (
        float(daily_net[daily_net > 0].sum() / abs(daily_net[daily_net < 0].sum()))
        if np.any(daily_net < 0)
        else float("inf")
    )

    event_indices = np.flatnonzero(traded > 1e-12)
    trade_gross: list[float] = []
    trade_net: list[float] = []
    for position, event in enumerate(event_indices):
        next_event = event_indices[position + 1] if position + 1 < len(event_indices) else len(net)
        trade_gross.append(float(np.prod(1.0 + gross[event:next_event]) - 1.0))
        trade_net.append(float(np.prod(1.0 + net[event:next_event]) - 1.0))

    if np.var(btc) > 0:
        beta = float(np.cov(net, btc, ddof=1)[0, 1] / np.var(btc, ddof=1))
        alpha = float(np.mean(net - beta * btc) * 365.25 * 24)
    else:
        beta = alpha = 0.0
    btc_total = float(np.prod(1.0 + btc) - 1.0)

    quarter_returns: dict[str, list[float]] = {}
    for date_value, value in zip(daily_dates, daily_net, strict=True):
        key = f"{date_value.year}Q{(date_value.month - 1) // 3 + 1}"
        quarter_returns.setdefault(key, []).append(float(value))
    quarter_compounded = {
        key: float(np.prod(1.0 + np.asarray(values)) - 1.0)
        for key, values in quarter_returns.items()
    }
    worst_quarter = (
        min(quarter_compounded, key=quarter_compounded.get) if quarter_compounded else ""
    )
    worst_quarter_return = quarter_compounded.get(worst_quarter, 0.0)
    max_consecutive_losses = current_losses = 0
    for value in daily_net:
        current_losses = current_losses + 1 if value < 0 else 0
        max_consecutive_losses = max(max_consecutive_losses, current_losses)

    total_cost_return = float(total_cost.sum())
    edge_cost_ratio = (
        gross_total_return / total_cost_return if total_cost_return > 0 else float("inf")
    )
    hac_t = (
        newey_west_tstat(daily_net, lag=max(1, math.ceil(rebalance_hours / 24)))
        if daily_net.size
        else 0.0
    )
    raw_p = two_sided_pvalue(hac_t, max(len(daily_net) - 1, 1))
    metrics = {
        "gross_total_return": gross_total_return,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "turnover": float(turnover.sum()),
        "traded_fraction": float(traded.sum()),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "mean_trade_gross_return": float(np.mean(trade_gross)) if trade_gross else 0.0,
        "mean_trade_net_return": float(np.mean(trade_net)) if trade_net else 0.0,
        "trade_count": len(trade_net),
        "total_fees_return": float(fee_cost.sum()),
        "total_slippage_return": float(slippage_cost.sum()),
        "total_cost_return": total_cost_return,
        "cost_to_abs_gross_ratio": total_cost_return / abs(gross_total_return)
        if gross_total_return
        else float("inf"),
        "edge_cost_ratio": edge_cost_ratio,
        "market_beta": beta,
        "annualized_alpha": alpha,
        "worst_quarter": worst_quarter,
        "worst_quarter_return": worst_quarter_return,
        "max_consecutive_losing_days": max_consecutive_losses,
        "btc_total_return": btc_total,
        "excess_vs_btc": total_return - btc_total,
        "daily_hac_tstat": hac_t,
        "raw_pvalue": raw_p,
        "daily_observations": len(daily_net),
    }
    series = {
        "timestamps": np.asarray(timestamps, dtype=object),
        "gross": gross,
        "net": net,
        "gross_equity": gross_equity,
        "net_equity": net_equity,
        "drawdown": drawdown,
        "daily_dates": np.asarray(daily_dates, dtype=object),
        "daily_net": daily_net,
        "daily_gross": daily_gross,
    }
    return metrics, series
