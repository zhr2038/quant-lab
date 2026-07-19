from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.research.factor_research.attribution import (
    compute_factor_attribution,
    decompose_low_volatility,
)


def test_symbol_fixed_effect_and_joint_residual_remove_static_coin_identity() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    symbols = ["A-USDT", "B-USDT", "C-USDT", "D-USDT"]
    rows = []
    for period in range(40):
        for index, symbol in enumerate(symbols):
            static = float(index)
            rows.append(
                {
                    "symbol": symbol,
                    "decision_ts": start + timedelta(hours=period),
                    "alpha_score": static,
                    "forward_return": static * 0.01,
                    "market_beta": static,
                    "liquidity": static * 2,
                    "momentum": 0.0,
                    "long_run_volatility": static,
                    "size_proxy": static,
                    "regime_score": 0.0,
                }
            )

    result = compute_factor_attribution(
        pl.DataFrame(rows),
        factor_id="legacy.static_coin_identity",
    )

    assert result.raw_rank_ic > 0.9
    assert abs(result.symbol_fixed_effect_null_ic) < 0.01
    assert abs(result.joint_residual_rank_ic) < 0.01
    assert result.attribution_type == "NO_INCREMENTAL_EDGE"


def test_beta_and_liquidity_residualization_preserve_independent_component() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for period in range(50):
        for index, symbol in enumerate(["A", "B", "C", "D", "E", "F"]):
            beta = float(index % 3)
            liquidity = float(index // 2)
            independent = float((index + period) % 6)
            rows.append(
                {
                    "symbol": symbol,
                    "decision_ts": start + timedelta(hours=period),
                    "alpha_score": 8.0 * beta + 5.0 * liquidity + independent,
                    "forward_return": independent * 0.002,
                    "market_beta": beta,
                    "liquidity": liquidity,
                    "momentum": 0.0,
                    "long_run_volatility": 0.0,
                    "size_proxy": 0.0,
                    "regime_score": 0.0,
                }
            )

    result = compute_factor_attribution(pl.DataFrame(rows), factor_id="test.independent")

    assert result.beta_neutral_rank_ic > 0
    assert result.liquidity_neutral_rank_ic > 0
    assert result.joint_residual_rank_ic > 0.5
    assert result.incremental_ic == result.joint_residual_rank_ic


def test_low_vol_decomposition_separates_structural_and_dynamic_change() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for symbol, first_step, second_step in (
        ("STABLE", 0.001, 0.001),
        ("CALMING", 0.03, 0.002),
    ):
        close = 100.0
        for index in range(60):
            step = first_step if index < 40 else second_step
            close *= 1.0 + step * (1 if index % 2 == 0 else -1)
            rows.append(
                {"symbol": symbol, "ts": start + timedelta(hours=index), "close": close}
            )

    result = decompose_low_volatility(
        pl.DataFrame(rows),
        recent_window=20,
        structural_window=40,
    )
    latest = {row["symbol"]: row for row in result.group_by("symbol").tail(1).to_dicts()}

    assert latest["STABLE"]["structural_low_vol"] > latest["CALMING"]["structural_low_vol"]
    assert latest["CALMING"]["dynamic_low_vol"] > latest["STABLE"]["dynamic_low_vol"]
