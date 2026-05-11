from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.research.labels import build_forward_return_labels, validate_no_label_lookahead


def test_forward_return_labels_use_one_bar_decision_delay():
    start = datetime(2026, 5, 10, tzinfo=UTC)
    bars = pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "timeframe": "1H",
                "ts": start + timedelta(hours=index),
                "close": 100.0 + index,
            }
            for index in range(6)
        ]
    )

    labels = build_forward_return_labels(
        bars,
        horizon_bars=2,
        decision_delay_bars=1,
    )
    first = labels.to_dicts()[0]

    assert first["feature_ts"] == start
    assert first["decision_ts"] == start + timedelta(hours=1)
    assert first["label_ts"] == start + timedelta(hours=3)
    assert first["forward_return"] == pytest.approx(103.0 / 101.0 - 1.0)


def test_forward_return_labels_do_not_cross_symbols():
    start = datetime(2026, 5, 10, tzinfo=UTC)
    bars = pl.DataFrame(
        [
            {
                "symbol": symbol,
                "timeframe": "1H",
                "ts": start + timedelta(hours=index),
                "close": base + index,
            }
            for symbol, base in [("BTC-USDT", 100.0), ("ETH-USDT", 200.0)]
            for index in range(5)
        ]
    )

    labels = build_forward_return_labels(bars, horizon_bars=1, decision_delay_bars=1)

    assert labels.filter(pl.col("symbol") == "BTC-USDT").height == 3
    assert labels.filter(pl.col("symbol") == "ETH-USDT").height == 3


def test_validate_no_label_lookahead_rejects_bad_fixture():
    ts = datetime(2026, 5, 10, tzinfo=UTC)
    labels = pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "timeframe": "1H",
                "feature_ts": ts,
                "decision_ts": ts,
                "label_ts": ts + timedelta(hours=1),
                "forward_return": 0.01,
                "horizon_bars": 1,
                "decision_delay_bars": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="decision_ts must be after feature_ts"):
        validate_no_label_lookahead(labels)
