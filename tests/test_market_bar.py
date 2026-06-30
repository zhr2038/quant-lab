from datetime import UTC, datetime, timedelta, timezone

import pytest

from quant_lab.contracts.models import MarketBar
from quant_lab.data.lake import (
    market_bars_to_polars,
    read_market_bars,
    validate_market_bars,
    write_market_bars,
)


def bar(**overrides) -> dict:
    values = {
        "venue": "okx",
        "symbol": "BTC-USDT",
        "market_type": "SPOT",
        "timeframe": "1H",
        "ts": datetime(2026, 5, 10, 1, tzinfo=UTC),
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "volume": 12.0,
        "quote_volume": 1260.0,
        "source": "test_fixture",
        "ingest_ts": datetime(2026, 5, 10, 2, tzinfo=UTC),
    }
    values.update(overrides)
    return values


def test_valid_market_bars_pass_and_convert_to_polars():
    records = validate_market_bars([bar()])
    df = market_bars_to_polars(records)

    assert len(records) == 1
    assert isinstance(records[0], MarketBar)
    assert df.height == 1
    assert df["symbol"][0] == "BTC-USDT"
    assert df["quote_volume"][0] == 1260.0


def test_quote_volume_is_optional():
    records = validate_market_bars([bar(quote_volume=None)])

    assert records[0].quote_volume is None


def test_venue_is_normalized_and_blank_rejected():
    records = validate_market_bars([bar(venue=" OKX ")])

    assert records[0].venue == "okx"
    with pytest.raises(ValueError, match="venue must not be blank"):
        validate_market_bars([bar(venue="   ")])


def test_invalid_high_low_fails():
    with pytest.raises(ValueError, match="high"):
        validate_market_bars([bar(high=89.0, low=90.0)])


def test_duplicate_primary_key_fails():
    duplicate = bar()

    with pytest.raises(ValueError, match="duplicate market_bar primary key"):
        validate_market_bars([duplicate, duplicate])


def test_timestamp_is_normalized_to_utc():
    shanghai_tz = timezone(timedelta(hours=8))
    records = validate_market_bars(
        [
            bar(
                ts=datetime(2026, 5, 10, 9, tzinfo=shanghai_tz),
                ingest_ts=datetime(2026, 5, 10, 10, tzinfo=shanghai_tz),
            )
        ]
    )

    assert records[0].ts == datetime(2026, 5, 10, 1, tzinfo=UTC)
    assert records[0].ingest_ts == datetime(2026, 5, 10, 2, tzinfo=UTC)


def test_write_and_read_market_bars(tmp_path):
    lake_root = tmp_path / "lake"
    records = [
        bar(ts=datetime(2026, 5, 10, 1, tzinfo=UTC), close=101.0),
        bar(ts=datetime(2026, 5, 10, 2, tzinfo=UTC), close=102.0),
        bar(
            symbol="ETH-USDT",
            ts=datetime(2026, 5, 10, 1, tzinfo=UTC),
            open=200.0,
            high=220.0,
            low=190.0,
            close=201.0,
        ),
    ]

    rows_after_first_write = write_market_bars(lake_root, records)
    rows_after_second_write = write_market_bars(lake_root, records)
    loaded = read_market_bars(
        lake_root,
        venue="okx",
        symbol="BTC-USDT",
        timeframe="1H",
        start=datetime(2026, 5, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 10, 2, tzinfo=UTC),
    )

    assert rows_after_first_write == 3
    assert rows_after_second_write == 3
    assert [record.close for record in loaded] == [101.0, 102.0]


def test_write_empty_market_bars_does_not_delete_existing_dataset(tmp_path):
    lake_root = tmp_path / "lake"
    records = [
        bar(ts=datetime(2026, 5, 10, 1, tzinfo=UTC), close=101.0),
        bar(ts=datetime(2026, 5, 10, 2, tzinfo=UTC), close=102.0),
    ]

    assert write_market_bars(lake_root, records) == 2
    assert write_market_bars(lake_root, []) == 2

    loaded = read_market_bars(
        lake_root,
        venue="okx",
        symbol="BTC-USDT",
        timeframe="1H",
        start=datetime(2026, 5, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 10, 3, tzinfo=UTC),
    )

    assert [record.close for record in loaded] == [101.0, 102.0]


def test_market_bar_hot_paths_do_not_eager_read_full_dataset(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    records = [
        bar(ts=datetime(2026, 5, 10, 1, tzinfo=UTC), close=101.0),
        bar(ts=datetime(2026, 5, 10, 2, tzinfo=UTC), close=102.0),
    ]
    write_market_bars(lake_root, records)

    def fail_full_read(*_args, **_kwargs):
        raise AssertionError("market_bar hot path should not eager-read full dataset")

    monkeypatch.setattr("quant_lab.data.lake.read_parquet_dataset", fail_full_read)

    assert write_market_bars(lake_root, []) == 2
    loaded = read_market_bars(
        lake_root,
        venue="okx",
        symbol="BTC-USDT",
        timeframe="1H",
        start=datetime(2026, 5, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 10, 3, tzinfo=UTC),
    )

    assert [record.close for record in loaded] == [101.0, 102.0]
