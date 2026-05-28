from datetime import UTC, datetime

import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.ops.data_quality import run_data_quality


def test_feature_value_quality_checks_required_schema_and_key(tmp_path):
    lake = tmp_path / "lake"
    created_at = datetime(2026, 5, 28, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        [
            {
                "feature_set": "core",
                "feature_name": "close_return_24h",
                "feature_version": "v0.1",
                "symbol": "BTC-USDT",
                "timeframe": "1H",
                "ts": datetime(2026, 5, 28, 1, tzinfo=UTC),
                "value": 0.01,
                "created_at": created_at,
                "source": "market_bar",
                "is_valid": True,
            },
            {
                "feature_set": "core",
                "feature_name": "close_return_24h",
                "feature_version": "v0.1",
                "symbol": "BTC-USDT",
                "timeframe": "1H",
                "ts": datetime(2026, 5, 28, 1, tzinfo=UTC),
                "value": 0.02,
                "created_at": created_at,
                "source": "market_bar",
                "is_valid": True,
            },
        ]
    )
    write_parquet_dataset(frame, lake / "gold" / "feature_value")

    result = run_data_quality(
        lake,
        dataset_names=["feature_value"],
        reference_at=datetime(2026, 5, 28, 3, tzinfo=UTC),
    ).to_dict()

    checks = {check["rule"]: check for check in result["checks"]}
    assert checks["schema_required_columns"]["status"] == "PASS"
    assert checks["primary_key_unique"]["status"] == "FAIL"


def test_feature_coverage_quality_flags_missing_required_columns(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame([{"day": "2026-05-28", "feature_name": "close_return_24h"}]),
        lake / "gold" / "feature_coverage_daily",
    )

    result = run_data_quality(
        lake,
        dataset_names=["feature_coverage_daily"],
        reference_at=datetime(2026, 5, 28, 3, tzinfo=UTC),
    ).to_dict()

    schema = next(check for check in result["checks"] if check["rule"] == "schema_required_columns")
    assert schema["status"] == "FAIL"
    assert "missing_columns" in schema["detail"]


def test_feature_value_quality_flags_infinite_and_invalid_consistency(tmp_path):
    lake = tmp_path / "lake"
    created_at = datetime(2026, 5, 28, 2, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "feature_set": "core",
                    "feature_name": "close_return_24h",
                    "feature_version": "v0.1",
                    "symbol": "BTC-USDT",
                    "timeframe": "1H",
                    "ts": datetime(2026, 5, 28, 1, tzinfo=UTC),
                    "value": float("inf"),
                    "created_at": created_at,
                    "source": "market_bar",
                    "is_valid": True,
                    "invalid_reason": None,
                },
                {
                    "feature_set": "core",
                    "feature_name": "volatility_24h",
                    "feature_version": "v0.1",
                    "symbol": "BTC-USDT",
                    "timeframe": "1H",
                    "ts": datetime(2026, 5, 28, 1, tzinfo=UTC),
                    "value": None,
                    "created_at": created_at,
                    "source": "market_bar",
                    "is_valid": True,
                    "invalid_reason": None,
                },
                {
                    "feature_set": "core",
                    "feature_name": "rsi_14",
                    "feature_version": "v0.1",
                    "symbol": "BTC-USDT",
                    "timeframe": "1H",
                    "ts": datetime(2026, 5, 28, 1, tzinfo=UTC),
                    "value": 42.0,
                    "created_at": created_at,
                    "source": "market_bar",
                    "is_valid": True,
                    "invalid_reason": "stale_input",
                },
            ]
        ),
        lake / "gold" / "feature_value",
    )

    result = run_data_quality(
        lake,
        dataset_names=["feature_value"],
        reference_at=datetime(2026, 5, 28, 3, tzinfo=UTC),
    ).to_dict()

    checks = {check["rule"]: check for check in result["checks"]}
    assert result["status"] == "FAIL"
    assert checks["feature_value_no_infinite"]["status"] == "FAIL"
    assert checks["feature_value_null_valid_consistency"]["status"] == "FAIL"
    assert checks["feature_value_invalid_reason_consistency"]["status"] == "FAIL"
