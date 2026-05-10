from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.contracts.models import FeatureValue
from quant_lab.features.registry import (
    FeatureRegistry,
    FeatureTimestampLeakageError,
    close_return_spec,
    compute_feature_values,
    rolling_volatility_spec,
    validate_feature_timestamps,
)

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
