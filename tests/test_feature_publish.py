from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.features.publish import feature_health, publish_features
from quant_lab.features.registry import FeatureTimestampLeakageError, validate_feature_timestamps


def test_publish_features_writes_values_coverage_and_anomalies(tmp_path):
    lake = tmp_path / "lake"
    _write_bars(lake, symbols=["BTC-USDT", "ETH-USDT"], count=80)

    result = publish_features(lake, symbols=["BTC-USDT", "ETH-USDT"])
    second = publish_features(lake, symbols=["BTC-USDT", "ETH-USDT"])

    features = read_parquet_dataset(lake / "gold" / "feature_value")
    coverage = read_parquet_dataset(lake / "gold" / "feature_coverage_daily")
    anomalies = read_parquet_dataset(lake / "gold" / "feature_anomaly_daily")

    assert result.feature_count == 10
    assert result.rows_written == 1600
    assert second.feature_value_rows == result.feature_value_rows
    assert features.height == 1600
    assert coverage.height > 0
    assert anomalies.filter(pl.col("anomaly_type") == "zero_variance").height > 0


def test_publish_features_rejects_incompatible_legacy_feature_schema_by_default(tmp_path):
    lake = tmp_path / "lake"
    _write_bars(lake, symbols=["BTC-USDT"], count=30)

    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "feature_set": "core",
                    "feature_name": "close_return_1",
                    "feature_version": "v0.1",
                    "symbol": "BTC-USDT",
                    "ts": datetime(2026, 5, 10, tzinfo=UTC),
                    "value": 0.0,
                }
            ]
        ),
        lake / "gold" / "feature_value",
    )
    original = read_parquet_dataset(lake / "gold" / "feature_value")

    with pytest.raises(ValueError, match="schema incompatible"):
        publish_features(lake)

    after = read_parquet_dataset(lake / "gold" / "feature_value")

    assert after.to_dicts() == original.to_dicts()


def test_publish_features_can_replace_incompatible_legacy_feature_schema_when_allowed(tmp_path):
    lake = tmp_path / "lake"
    _write_bars(lake, symbols=["BTC-USDT"], count=30)

    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "feature_set": "core",
                    "feature_name": "close_return_1",
                    "feature_version": "v0.1",
                    "symbol": "BTC-USDT",
                    "ts": datetime(2026, 5, 10, tzinfo=UTC),
                    "value": 0.0,
                }
            ]
        ),
        lake / "gold" / "feature_value",
    )

    result = publish_features(lake, allow_schema_replace=True)
    features = read_parquet_dataset(lake / "gold" / "feature_value")

    assert result.feature_value_rows == 300
    assert features.height == 300
    assert "timeframe" in features.columns
    assert "is_valid" in features.columns


def test_core_feature_values_are_correct_and_symbol_isolated(tmp_path):
    lake = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    _write_bars(lake, symbols=["BTC-USDT", "ETH-USDT"], count=80, start=start)

    publish_features(lake)
    features = read_parquet_dataset(lake / "gold" / "feature_value")
    btc = features.filter(
        (pl.col("symbol") == "BTC-USDT") & (pl.col("feature_name") == "close_return_4")
    ).sort("ts")
    eth_first = features.filter(
        (pl.col("symbol") == "ETH-USDT") & (pl.col("feature_name") == "close_return_1")
    ).sort("ts")
    range_bps = features.filter(
        (pl.col("symbol") == "BTC-USDT")
        & (pl.col("feature_name") == "range_bps")
        & (pl.col("ts") == start + timedelta(hours=10))
    )
    dollar_volume = features.filter(
        (pl.col("symbol") == "BTC-USDT")
        & (pl.col("feature_name") == "dollar_volume")
        & (pl.col("ts") == start + timedelta(hours=10))
    )

    assert btc["value"][4] == pytest.approx(104.0 / 100.0 - 1.0)
    assert eth_first["value"][0] is None
    assert range_bps["value"][0] == pytest.approx((112.0 - 108.0) / 110.0 * 10_000)
    assert dollar_volume["value"][0] == pytest.approx(1100.0)


def test_close_position_in_zero_range_is_invalid(tmp_path):
    lake = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rows = [
        _bar("BTC-USDT", start + timedelta(hours=index), close=100.0 + index)
        for index in range(30)
    ]
    rows[10] = _bar("BTC-USDT", start + timedelta(hours=10), close=110.0, zero_range=True)
    write_market_bars(lake, rows)

    publish_features(lake)
    features = read_parquet_dataset(lake / "gold" / "feature_value")
    zero_range = features.filter(
        (pl.col("feature_name") == "close_position_in_range")
        & (pl.col("ts") == start + timedelta(hours=10))
    )

    assert zero_range["value"][0] is None
    assert zero_range["is_valid"][0] is False
    assert zero_range["invalid_reason"][0] == "zero_range"


def test_anti_leakage_requires_closed_market_bar():
    ts = datetime(2026, 5, 10, tzinfo=UTC)
    feature = {
        "feature_set": "core",
        "feature_name": "close_return_1",
        "feature_version": "v0.1",
        "symbol": "BTC-USDT",
        "timeframe": "1H",
        "ts": ts,
        "value": 0.01,
        "created_at": ts + timedelta(hours=1),
    }
    market = pl.DataFrame(
        [{"symbol": "BTC-USDT", "timeframe": "1H", "ts": ts, "is_closed": False}]
    )

    with pytest.raises(FeatureTimestampLeakageError):
        validate_feature_timestamps([feature], market)


def test_feature_health_reports_coverage_and_anomalies(tmp_path):
    lake = tmp_path / "lake"
    _write_bars(lake, symbols=["BTC-USDT"], count=40)
    publish_features(lake)

    health = feature_health(lake, feature_set="core", date="2026-05-10")

    assert health.coverage_rows > 0
    assert health.anomaly_rows > 0
    assert health.top_anomalies


def _write_bars(
    lake,
    *,
    symbols: list[str],
    count: int,
    start: datetime = datetime(2026, 5, 10, tzinfo=UTC),
) -> None:
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for index in range(count):
            close = 100.0 + index + symbol_index * 10
            rows.append(_bar(symbol, start + timedelta(hours=index), close=close))
    write_market_bars(lake, rows)


def _bar(symbol: str, ts: datetime, *, close: float, zero_range: bool = False) -> dict:
    high = close if zero_range else close + 2.0
    low = close if zero_range else close - 2.0
    return {
        "venue": "okx",
        "symbol": symbol,
        "market_type": "SPOT",
        "timeframe": "1H",
        "ts": ts,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 10.0,
        "quote_volume": close * 10.0,
        "source": "test",
        "ingest_ts": ts + timedelta(minutes=1),
    }
