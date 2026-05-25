from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.contracts.models import FeatureValue
from quant_lab.data.lake import read_parquet_dataset, write_market_bars
from quant_lab.features.publish import publish_core_features
from quant_lab.features.registry import (
    FeatureRegistry,
    FeatureTimestampLeakageError,
    close_return_spec,
    compute_close_position_in_range,
    compute_dollar_volume,
    compute_feature_values,
    compute_liquidity_proxy,
    compute_range_bps,
    compute_volume_zscore_n,
    default_core_registry,
    rolling_volatility_spec,
    validate_feature_timestamps,
)
from quant_lab.web import readers

CREATED_AT = datetime(2026, 5, 10, tzinfo=UTC)


def market_bars() -> pl.DataFrame:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": ["BTC-USDT"] * 5,
            "ts": [start + timedelta(hours=index) for index in range(5)],
            "close": [100.0, 101.0, 103.0, 102.0, 106.0],
        }
    )


def test_registry_register_get_list():
    registry = FeatureRegistry()
    spec = close_return_spec(lookback_bars=2, feature_version="v1")

    registry.register(spec)

    assert registry.get("demo", "close_return_n", "v1") == spec
    assert registry.list() == [spec]
    assert registry.list_names() == ["close_return_n"]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_core_registry_uses_specific_compute_functions_for_non_return_features():
    registry = default_core_registry()

    assert registry.get("core", "volume_zscore_24", "v0.1").compute is compute_volume_zscore_n
    assert registry.get("core", "range_bps", "v0.1").compute is compute_range_bps
    assert (
        registry.get("core", "close_position_in_range", "v0.1").compute
        is compute_close_position_in_range
    )
    assert registry.get("core", "dollar_volume", "v0.1").compute is compute_dollar_volume
    assert registry.get("core", "liquidity_proxy", "v0.1").compute is compute_liquidity_proxy


def test_non_return_feature_compute_does_not_calculate_close_return():
    registry = default_core_registry()
    bars = pl.DataFrame(
        {
            "symbol": ["BTC-USDT", "BTC-USDT"],
            "timeframe": ["1H", "1H"],
            "ts": [
                datetime(2026, 5, 1, tzinfo=UTC),
                datetime(2026, 5, 1, 1, tzinfo=UTC),
            ],
            "open": [100.0, 200.0],
            "high": [110.0, 220.0],
            "low": [90.0, 180.0],
            "close": [100.0, 200.0],
            "volume": [10.0, 20.0],
            "quote_volume": [1000.0, 4000.0],
            "is_closed": [True, True],
        }
    )
    context = {
        "input_dataset_version": "market-bar-v1",
        "input_hash": "sha256:test-bars",
        "code_version": "features-code-v1",
        "created_at": CREATED_AT,
    }

    range_value = compute_feature_values(
        registry.get("core", "range_bps", "v0.1"),
        bars,
        **context,
    )[1]
    dollar_value = compute_feature_values(
        registry.get("core", "dollar_volume", "v0.1"),
        bars,
        **context,
    )[1]

    assert range_value.value == pytest.approx((220.0 - 180.0) / 200.0 * 10_000)
    assert dollar_value.value == pytest.approx(4000.0)
    assert range_value.value != pytest.approx(200.0 / 100.0 - 1.0)


def test_close_return_feature_computation_output_shape():
    spec = close_return_spec(lookback_bars=2, feature_version="v1")

    values = compute_feature_values(
        spec,
        market_bars(),
        input_dataset_version="market-bar-v1",
        input_hash="sha256:test-bars",
        code_version="features-code-v1",
        created_at=CREATED_AT,
    )

    assert len(values) == 5
    assert all(value.feature_name == "close_return_n" for value in values)
    assert all(value.lookback_bars == 2 for value in values)
    assert [value.value for value in values[:2]] == [None, None]
    assert values[2].value == pytest.approx(0.03)
    assert values[4].value == pytest.approx(106.0 / 103.0 - 1.0)


def test_rolling_volatility_null_handling_is_explicit():
    spec = rolling_volatility_spec(lookback_bars=3, feature_version="v1")

    values = compute_feature_values(
        spec,
        market_bars(),
        input_dataset_version="market-bar-v1",
        input_hash="sha256:test-bars",
        code_version="features-code-v1",
        created_at=CREATED_AT,
    )

    assert len(values) == 5
    assert any(value.value is None for value in values)
    assert all(value.value is None or isinstance(value.value, float) for value in values)

    with pytest.raises(ValueError, match="finite or null"):
        FeatureValue(
            feature_set="demo",
            feature_name="bad_nan",
            feature_version="v1",
            symbol="BTC-USDT",
            ts=datetime(2026, 5, 1, tzinfo=UTC),
            value=float("nan"),
            lookback_bars=1,
            input_dataset_version="market-bar-v1",
            input_hash="sha256:test-bars",
            code_version="features-code-v1",
            created_at=CREATED_AT,
        )


def test_feature_timestamps_pass_with_one_bar_decision_delay():
    bars = market_bars()
    spec = close_return_spec(lookback_bars=1, feature_version="v1")
    values = compute_feature_values(
        spec,
        bars,
        input_dataset_version="market-bar-v1",
        input_hash="sha256:test-bars",
        code_version="features-code-v1",
        created_at=CREATED_AT,
    )
    joined_records = [
        values[0].model_dump() | {"decision_ts": bars["ts"][1]},
        values[1].model_dump() | {"decision_ts": bars["ts"][2]},
        values[2].model_dump() | {"decision_ts": bars["ts"][3]},
    ]

    validate_feature_timestamps(values)
    validate_feature_timestamps(joined_records, decision_delay_bars=1)


def test_deliberate_look_ahead_leak_fails():
    bars = market_bars()
    spec = close_return_spec(lookback_bars=1, feature_version="v1")
    values = compute_feature_values(
        spec,
        bars,
        input_dataset_version="market-bar-v1",
        input_hash="sha256:test-bars",
        code_version="features-code-v1",
        created_at=CREATED_AT,
    )
    leaking_records = [value.model_dump() | {"decision_ts": value.ts} for value in values]

    with pytest.raises(FeatureTimestampLeakageError):
        validate_feature_timestamps(leaking_records, decision_delay_bars=1)


def test_feature_created_before_bar_timestamp_fails():
    spec = close_return_spec(lookback_bars=1, feature_version="v1")

    with pytest.raises(ValueError, match="created_at must not be earlier"):
        compute_feature_values(
            spec,
            market_bars(),
            input_dataset_version="market-bar-v1",
            input_hash="sha256:test-bars",
            code_version="features-code-v1",
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
        )


def test_publish_core_features_writes_feature_value_dataset(tmp_path):
    lake = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    write_market_bars(
        lake,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": start + timedelta(hours=index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.0 + index,
                "volume": 10.0,
                "quote_volume": 1000.0,
                "source": "test",
                "ingest_ts": start + timedelta(hours=index, minutes=1),
            }
            for index in range(80)
        ],
    )

    result = publish_core_features(lake, feature_set="core", timeframe="1H")
    second = publish_core_features(lake, feature_set="core", timeframe="1H")
    features = read_parquet_dataset(lake / "gold" / "feature_value")

    assert result.market_bar_rows == 80
    assert result.published_rows == 800
    assert result.feature_names == [
        "close_return_1",
        "close_return_4",
        "close_return_24",
        "rolling_volatility_24",
        "rolling_volatility_72",
        "volume_zscore_24",
        "range_bps",
        "close_position_in_range",
        "dollar_volume",
        "liquidity_proxy",
    ]
    assert second.feature_value_rows == result.feature_value_rows
    assert features.height == 800
    assert set(features["feature_name"].unique().to_list()) == set(result.feature_names)
    assert features.filter(pl.col("feature_name") == "close_return_1")["value"][1] == pytest.approx(
        101.0 / 100.0 - 1.0
    )


def test_publish_core_features_empty_market_bar_is_noop(tmp_path):
    result = publish_core_features(tmp_path / "lake", feature_set="core", timeframe="1H")

    assert result.market_bar_rows == 0
    assert result.published_rows == 0
    assert result.feature_value_rows == 0
    assert result.warnings == ["market_bar missing or empty for feature publishing"]


def test_data_health_labels_feature_and_alpha_gaps_as_not_yet_generated(tmp_path):
    status_rows = readers.data_health_summary(tmp_path / "lake")["stale_datasets"].to_dicts()
    status_by_dataset = {row["dataset"]: row["status"] for row in status_rows}

    assert status_by_dataset["feature_value"] == "特征尚未发布"
    assert status_by_dataset["alpha_evidence"] == "研究证据尚未生成"
    feature_row = next(row for row in status_rows if row["dataset"] == "feature_value")
    alpha_row = next(row for row in status_rows if row["dataset"] == "alpha_evidence")
    assert feature_row["takeaway"].startswith("特征值 暂无数据")
    assert "qlab publish-features" in feature_row["next_action"]
    assert alpha_row["severity"] == "WARNING"
    assert "qlab build-alpha-evidence" in alpha_row["next_action"]
