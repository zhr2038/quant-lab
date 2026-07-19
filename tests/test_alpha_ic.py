from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.research.ic import (
    compute_by_symbol_ic,
    compute_ic,
    compute_ic_tstat,
    compute_period_ic_values,
    compute_rank_ic,
)


def test_compute_ic_and_rank_ic_cross_sectional():
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rows = []
    for period in range(3):
        for index, symbol in enumerate(["BTC-USDT", "ETH-USDT", "SOL-USDT"]):
            rows.append(
                {
                    "symbol": symbol,
                    "decision_ts": start + timedelta(hours=period),
                    "alpha_score": float(index + period * 0.1),
                    "forward_return": float(index + period * 0.2),
                }
            )
    df = pl.DataFrame(rows)

    ic = compute_ic(df)
    rank_ic = compute_rank_ic(df)

    assert ic.mean == pytest.approx(1.0)
    assert rank_ic.mean == pytest.approx(1.0)
    assert ic.period_count == 3
    assert compute_period_ic_values(df, rank=True) == pytest.approx([1.0, 1.0, 1.0])


def test_ic_skips_periods_with_too_few_symbols():
    df = pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "decision_ts": datetime(2026, 5, 10, tzinfo=UTC),
                "alpha_score": 1.0,
                "forward_return": 0.01,
            },
            {
                "symbol": "ETH-USDT",
                "decision_ts": datetime(2026, 5, 10, tzinfo=UTC),
                "alpha_score": 2.0,
                "forward_return": 0.02,
            },
        ]
    )

    stats = compute_ic(df)

    assert stats.status == "insufficient_samples"
    assert stats.period_count == 0


def test_ic_tstat_and_by_symbol_ic():
    df = pl.DataFrame(
        [
            {"symbol": "BTC-USDT", "alpha_score": 1.0, "forward_return": 0.01},
            {"symbol": "BTC-USDT", "alpha_score": 2.0, "forward_return": 0.02},
            {"symbol": "ETH-USDT", "alpha_score": 1.0, "forward_return": 0.03},
            {"symbol": "ETH-USDT", "alpha_score": 2.0, "forward_return": 0.01},
        ]
    )

    assert compute_ic_tstat([0.1, 0.2, 0.3]) > 0
    by_symbol = compute_by_symbol_ic(df)
    assert by_symbol["BTC-USDT"] == pytest.approx(1.0)
    assert by_symbol["ETH-USDT"] == pytest.approx(-1.0)
