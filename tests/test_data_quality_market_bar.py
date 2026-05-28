from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.ops.data_quality import run_data_quality


def _market_bar_row(**updates):
    row = {
        "venue": "okx",
        "symbol": "BTC-USDT",
        "market_type": "SPOT",
        "timeframe": "1H",
        "ts": datetime(2026, 5, 28, 1, tzinfo=UTC),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1.0,
        "quote_volume": 100.5,
        "source": "okx_public_rest",
        "ingest_ts": datetime(2026, 5, 28, 2, tzinfo=UTC),
        "is_closed": True,
    }
    row.update(updates)
    return row


def test_market_bar_quality_passes_for_closed_unique_utc_bars(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame([_market_bar_row()]),
        lake / "silver" / "market_bar",
    )

    result = run_data_quality(
        lake,
        dataset_names=["market_bar"],
        reference_at=datetime(2026, 5, 28, 2, 30, tzinfo=UTC),
    ).to_dict()

    checks = {(check["dataset"], check["rule"]): check for check in result["checks"]}
    assert result["status"] == "PASS"
    assert checks[("market_bar", "closed_bar_only")]["status"] == "PASS"
    assert checks[("market_bar", "primary_key_unique")]["status"] == "PASS"


def test_market_bar_quality_fails_for_unclosed_or_duplicate_bars(tmp_path):
    lake = tmp_path / "lake"
    duplicate_ts = datetime(2026, 5, 28, 1, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _market_bar_row(ts=duplicate_ts, is_closed=False),
                _market_bar_row(ts=duplicate_ts, close=100.8),
            ]
        ),
        lake / "silver" / "market_bar",
    )

    result = run_data_quality(
        lake,
        dataset_names=["market_bar"],
        reference_at=datetime(2026, 5, 28, 2, 30, tzinfo=UTC),
    ).to_dict()

    checks = {(check["dataset"], check["rule"]): check for check in result["checks"]}
    assert result["status"] == "FAIL"
    assert checks[("market_bar", "closed_bar_only")]["status"] == "FAIL"
    assert checks[("market_bar", "primary_key_unique")]["status"] == "FAIL"


def test_market_bar_quality_fails_for_stale_dataset(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame([_market_bar_row(ts=datetime(2026, 5, 27, 1, tzinfo=UTC))]),
        lake / "silver" / "market_bar",
    )

    result = run_data_quality(
        lake,
        dataset_names=["market_bar"],
        reference_at=datetime(2026, 5, 28, 6, tzinfo=UTC),
    ).to_dict()

    freshness = next(check for check in result["checks"] if check["rule"] == "freshness")
    assert freshness["status"] == "FAIL"
    assert int(freshness["observed_value"]) > int(freshness["expected_value"])


def test_market_bar_quality_fails_for_non_utc_datetime(tmp_path):
    lake = tmp_path / "lake"
    row = _market_bar_row(
        ts=datetime(2026, 5, 28, 1),
        ingest_ts=datetime(2026, 5, 28, 2),
    )
    write_parquet_dataset(pl.DataFrame([row]), lake / "silver" / "market_bar")

    result = run_data_quality(
        lake,
        dataset_names=["market_bar"],
        reference_at=datetime(2026, 5, 28, 2, tzinfo=UTC) + timedelta(minutes=1),
    ).to_dict()

    utc_checks = [check for check in result["checks"] if check["rule"].startswith("utc_timestamp")]
    assert any(check["status"] == "FAIL" for check in utc_checks)
