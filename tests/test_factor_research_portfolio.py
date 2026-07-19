from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.research.factor_research.portfolio import (
    PortfolioRule,
    compute_factor_portfolio,
)


def _rule() -> PortfolioRule:
    return PortfolioRule(
        portfolio_rule_id="top3-equal-1bar-v1",
        top_n=3,
        weighting="equal",
        holding_period_bars=1,
        initial_capital_usdt=100.0,
        min_order_usdt=5.0,
        max_drawdown_limit=0.30,
        annual_periods=365,
    )


def test_ic_pass_but_long_only_loss_is_portfolio_fail() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for period in range(12):
        split = "VALIDATION" if period < 6 else "BLIND_CONFIRMATORY"
        for rank, symbol in enumerate(["A", "B", "C", "D"]):
            rows.append(
                {
                    "decision_ts": start + timedelta(days=period),
                    "symbol": symbol,
                    "alpha_score": float(4 - rank),
                    "forward_return": -0.01 - rank * 0.001,
                    "cost_bps": 20.0,
                    "tradable": True,
                    "btc_forward_return": -0.005,
                    "universe_forward_return": -0.006,
                    "split": split,
                }
            )

    result = compute_factor_portfolio(
        pl.DataFrame(rows),
        rule=_rule(),
        signal_validity="PASS",
    )

    assert result.net_return < 0
    assert result.portfolio_validity == "FAIL"
    assert result.decision == "PORTFOLIO_FAIL"
    assert "long_only_net_return_not_positive" in result.blockers


def test_portfolio_applies_two_sided_costs_and_cash_residual_without_replacement() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        [
            {
                "decision_ts": now,
                "symbol": "A",
                "alpha_score": 4.0,
                "forward_return": 0.04,
                "tradable": False,
                "entry_fee_bps": 5.0,
                "exit_fee_bps": 5.0,
                "entry_slippage_bps": 5.0,
                "exit_slippage_bps": 5.0,
                "btc_forward_return": 0.0,
                "universe_forward_return": 0.0,
                "split": "VALIDATION",
            },
            {
                "decision_ts": now,
                "symbol": "B",
                "alpha_score": 3.0,
                "forward_return": 0.03,
                "tradable": True,
                "entry_fee_bps": 5.0,
                "exit_fee_bps": 5.0,
                "entry_slippage_bps": 5.0,
                "exit_slippage_bps": 5.0,
                "btc_forward_return": 0.0,
                "universe_forward_return": 0.0,
                "split": "VALIDATION",
            },
            {
                "decision_ts": now,
                "symbol": "C",
                "alpha_score": 2.0,
                "forward_return": 0.02,
                "tradable": True,
                "entry_fee_bps": 5.0,
                "exit_fee_bps": 5.0,
                "entry_slippage_bps": 5.0,
                "exit_slippage_bps": 5.0,
                "btc_forward_return": 0.0,
                "universe_forward_return": 0.0,
                "split": "VALIDATION",
            },
            {
                "decision_ts": now,
                "symbol": "D",
                "alpha_score": 1.0,
                "forward_return": 0.50,
                "tradable": True,
                "entry_fee_bps": 0.0,
                "exit_fee_bps": 0.0,
                "entry_slippage_bps": 0.0,
                "exit_slippage_bps": 0.0,
                "btc_forward_return": 0.0,
                "universe_forward_return": 0.0,
                "split": "VALIDATION",
            },
        ]
    )

    result = compute_factor_portfolio(frame, rule=_rule(), signal_validity="PASS")

    assert result.average_cash_residual == pytest.approx(1 / 3)
    assert result.turnover == pytest.approx(4 / 3)
    assert result.fees > 0
    assert result.slippage > 0
    assert "D" not in result.symbol_contribution_json


def test_profitable_diversified_portfolio_reports_benchmarks_and_concentration() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for period in range(20):
        split = "VALIDATION" if period < 10 else "BLIND_CONFIRMATORY"
        for rank, symbol in enumerate(["A", "B", "C", "D", "E"]):
            rows.append(
                {
                    "decision_ts": start + timedelta(days=period),
                    "symbol": symbol,
                    "alpha_score": float(5 - rank),
                    "forward_return": 0.02 - rank * 0.002,
                    "cost_bps": 4.0,
                    "tradable": True,
                    "btc_forward_return": 0.004,
                    "universe_forward_return": 0.005,
                    "split": split,
                }
            )

    result = compute_factor_portfolio(
        pl.DataFrame(rows),
        rule=_rule(),
        signal_validity="PASS",
    )

    assert result.net_return > 0
    assert result.excess_vs_btc > 0
    assert result.excess_vs_universe > 0
    assert result.concentration_hhi <= 0.5
    assert result.validation_net_return and result.validation_net_return > 0
    assert result.blind_net_return and result.blind_net_return > 0
