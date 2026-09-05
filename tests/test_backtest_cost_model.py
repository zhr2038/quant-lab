from __future__ import annotations

import polars as pl

from quant_lab.backtest import cost_model


def test_conservative_cost_lookup_preserves_symbol_and_default_semantics(monkeypatch):
    frame = pl.DataFrame(
        [
            {"symbol": "BTC-USDT", "cost_bps": 10.0, "source": "a"},
            {"symbol": "BTC-USDT", "cost_bps": 20.0, "source": "b"},
            {"symbol": "ETH-USDT", "cost_bps": 40.0, "source": "c"},
        ]
    )
    calls = 0
    original_rows = cost_model.rows

    def counting_rows(value):
        nonlocal calls
        calls += 1
        return original_rows(value)

    monkeypatch.setattr(cost_model, "rows", counting_rows)

    lookup = cost_model.build_conservative_cost_lookup(frame)

    assert lookup.for_symbol("BTC/USDT") == cost_model.BacktestCost(
        cost_bps=20.0,
        cost_model="conservative_p75:a+b",
    )
    assert lookup.for_symbol("ETH-USDT").cost_bps == 40.0
    assert lookup.for_symbol("SOL-USDT") == cost_model.BacktestCost(
        cost_bps=30.0,
        cost_model="conservative_default_30bps",
    )
    assert lookup.for_symbol().cost_bps == 40.0
    assert calls == 1
