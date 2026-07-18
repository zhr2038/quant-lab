from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.factors import (  # noqa: E402
    funding_fade,
    v5_alpha6_static_signal,
    v5_raw_factors,
)
from audit.auditlib.universe import UniverseSpec, build_daily_universe  # noqa: E402


def _hourly_liquidity_panel(days: int = 42) -> pl.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for day in range(days):
        for hour in range(24):
            ts = start + timedelta(days=day, hours=hour)
            rows.append({"symbol": "A", "ts": ts, "quote_volume": 100.0, "close": 10.0})
            b_volume = 50.0 if day != 40 else 100_000.0
            rows.append({"symbol": "B", "ts": ts, "quote_volume": b_volume, "close": 10.0})
    return pl.DataFrame(rows)


def test_dynamic_universe_uses_only_prior_days_liquidity() -> None:
    bars = _hourly_liquidity_panel()
    spec = UniverseSpec(
        "top1",
        1,
        min_age_days=3,
        min_daily_quote_volume=0.0,
        lookback_days=3,
        max_missing_rate=0.01,
    )
    universe = build_daily_universe(bars, spec)
    day_40 = datetime(2026, 2, 10).date()
    day_41 = datetime(2026, 2, 11).date()
    assert universe.filter(pl.col("date") == day_40)["symbol"].to_list() == ["A"]
    assert universe.filter(pl.col("date") == day_41)["symbol"].to_list() == ["B"]


def test_v5_raw_factors_match_production_indexing() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    n = 500
    close = np.linspace(100.0, 200.0, n)
    volume = np.linspace(1_000.0, 2_000.0, n)
    bars = pl.DataFrame(
        {
            "symbol": ["A"] * n,
            "ts": [start + timedelta(hours=i) for i in range(n)],
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
        }
    )
    latest = v5_raw_factors(bars).sort("feature_ts").tail(1).row(0, named=True)
    assert latest["f1_mom_5d"] == pytest.approx(close[-1] / close[-120] - 1.0)
    assert latest["f2_mom_20d"] == pytest.approx(close[-1] / close[-480] - 1.0)
    expected_vol = np.std(np.diff(close[-481:]) / close[-481:-1], ddof=0)
    assert latest["f3_vol_adj_ret"] == pytest.approx(latest["f2_mom_20d"] / (expected_vol + 1e-12))
    expected_f4 = volume[-24:].mean() / volume[-192:-24].mean() - 1.0
    assert latest["f4_volume_expansion"] == pytest.approx(expected_f4)


def test_v5_static_proxy_is_cross_section_centered() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for symbol, drift in [("A", 0.001), ("B", -0.0002), ("C", 0.0005)]:
        for i in range(520):
            close = 100.0 * np.exp(drift * i)
            rows.append(
                {
                    "symbol": symbol,
                    "ts": start + timedelta(hours=i),
                    "close": close,
                    "volume": 1_000.0 + i,
                    "quote_volume": (1_000.0 + i) * close,
                }
            )
    raw = v5_raw_factors(pl.DataFrame(rows))
    signals = v5_alpha6_static_signal(raw)
    latest = signals.filter(pl.col("feature_ts") == signals["feature_ts"].max())
    assert latest["signal"].mean() == pytest.approx(0.0, abs=1e-12)


def test_funding_fade_counts_publications_not_hourly_asof_repeats() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = pl.DataFrame(
        {
            "symbol": ["A"] * 100,
            "ts": [start + timedelta(hours=i) for i in range(100)],
        }
    )
    funding = pl.DataFrame(
        {
            "symbol": ["A"] * 10,
            "funding_ts": [start + timedelta(hours=8 * i) for i in range(10)],
            "funding_rate": [0.001] * 10,
        }
    )
    assert funding_fade(bars, funding)["signal"].drop_nulls().is_empty()
