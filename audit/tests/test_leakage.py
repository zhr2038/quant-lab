"""Leakage-harness self tests: the battery must detect artificial leakage and
must not produce false positives on honest signals."""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit.auditlib.evaluate import build_labels  # noqa: E402
from audit.auditlib.leakage import run_leakage_battery, validate_temporal_order  # noqa: E402


def _synthetic_market(n_symbols=20, n_bars=24 * 120, seed=3):
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for s in range(n_symbols):
        rets = rng.normal(0.0003, 0.01, n_bars)
        close = 100 * np.exp(np.cumsum(rets))
        for t in range(n_bars):
            rows.append({
                "symbol": f"S{s:02d}",
                "ts": base + timedelta(hours=t),
                "open": close[t] * 0.999,
                "high": close[t] * 1.002,
                "low": close[t] * 0.998,
                "close": close[t],
                "volume": 1000.0,
                "quote_volume": 1000.0 * close[t],
            })
    return pl.DataFrame(rows)


def _momentum_signal(bars: pl.DataFrame, lookback: int = 24) -> pl.DataFrame:
    return (
        bars.sort(["symbol", "ts"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(lookback).over("symbol") - 1.0).alias("signal")
        )
        .select(["symbol", pl.col("ts").alias("feature_ts"), "signal"])
        .filter(pl.col("signal").is_not_null())
    )


def test_battery_detects_injected_leakage_and_passes_honest():
    bars = _synthetic_market()
    signals = _momentum_signal(bars)
    results = run_leakage_battery(signals, bars, factor_id="synthetic_mom", horizon_bars=24)
    by_name = {r.test_name: r for r in results}
    assert by_name["label_permutation"].passed, by_name["label_permutation"].detail
    assert by_name["signal_permutation"].passed, by_name["signal_permutation"].detail
    assert by_name["future_injection_detected"].passed, by_name["future_injection_detected"].detail
    assert by_name["decision_delay_zero_rejected"].passed


def test_label_builder_temporal_order():
    bars = _synthetic_market(n_symbols=3, n_bars=200)
    labels = build_labels(bars, horizon_bars=24, decision_delay_bars=1)
    validate_temporal_order(labels)  # must not raise


def test_label_builder_delay_values():
    bars = _synthetic_market(n_symbols=2, n_bars=100)
    labels = build_labels(bars, horizon_bars=10, decision_delay_bars=1)
    row = labels.sort(["symbol", "feature_ts"]).row(0, named=True)
    delta_dec = (row["decision_ts"] - row["feature_ts"]).total_seconds() / 3600
    delta_lab = (row["label_ts"] - row["decision_ts"]).total_seconds() / 3600
    assert delta_dec == 1
    assert delta_lab == 10