from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.backtest import cost_model
from quant_lab.backtest.reports import build_factor_forward_validation


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


def test_factor_forward_validation_materializes_cost_table_once(monkeypatch):
    base = datetime(2026, 8, 30, tzinfo=UTC)
    factor_values = pl.DataFrame(
        [
            {
                "factor_id": "factor:test",
                "symbol": "BTC-USDT",
                "available_time": base + timedelta(hours=hour),
                "value": float(hour),
                "is_valid": True,
            }
            for hour in range(3)
        ]
    )
    market_bars = pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "ts": base + timedelta(hours=hour),
                "close": 100.0 + hour,
            }
            for hour in range(12)
        ]
    )
    costs = pl.DataFrame(
        [{"symbol": "BTC-USDT", "cost_bps": float(value)} for value in range(100)]
    )
    calls = 0
    original_rows = cost_model.rows

    def counting_rows(value):
        nonlocal calls
        calls += 1
        return original_rows(value)

    monkeypatch.setattr(cost_model, "rows", counting_rows)

    result = build_factor_forward_validation(
        factor_candidates=pl.DataFrame(
            [
                {
                    "factor_id": "factor:test",
                    "factor_family": "test",
                    "candidate_state": "KEEP_SHADOW",
                }
            ]
        ),
        factor_values=factor_values,
        market_bars=market_bars,
        cost_bucket_daily=costs,
        horizon_hours=(1,),
    )

    assert not result.is_empty()
    assert calls == 1
