import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.ops.lake_health import lake_file_health_summary, write_lake_file_health_daily


def test_lake_file_health_summary_is_read_only(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame([{"symbol": "BTC-USDT", "value": 1.0}]),
        lake / "silver" / "market_bar",
    )

    summary = lake_file_health_summary(lake)

    assert summary["dataset_count"] > 0
    assert summary["total_parquet_files"] >= 1
    assert not (lake / "gold" / "lake_file_health_daily").exists()


def test_write_lake_file_health_daily_persists_snapshot(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame([{"symbol": "BTC-USDT", "value": 1.0}]),
        lake / "silver" / "market_bar",
    )

    summary = write_lake_file_health_daily(lake)

    assert summary["dataset_count"] > 0
    assert (lake / "gold" / "lake_file_health_daily").exists()
