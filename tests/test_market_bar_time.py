from datetime import UTC, datetime, timedelta

from quant_lab.data.market_bar_time import (
    market_bar_close_ts,
    market_bar_freshness_seconds,
    market_bar_interval_seconds,
)


def test_market_bar_interval_seconds_parses_common_okx_timeframes():
    assert market_bar_interval_seconds("1H") == 60 * 60
    assert market_bar_interval_seconds("15m") == 15 * 60
    assert market_bar_interval_seconds("4h") == 4 * 60 * 60
    assert market_bar_interval_seconds("1D") == 24 * 60 * 60


def test_market_bar_freshness_uses_close_time_not_open_time():
    now = datetime(2026, 6, 27, 15, 55, tzinfo=UTC)
    latest_open = datetime(2026, 6, 27, 14, 0, tzinfo=UTC)

    assert market_bar_close_ts(latest_open, "1H") == datetime(
        2026,
        6,
        27,
        15,
        0,
        tzinfo=UTC,
    )
    assert market_bar_freshness_seconds(latest_open, timeframe="1H", now=now) == int(
        timedelta(minutes=55).total_seconds()
    )
