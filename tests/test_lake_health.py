import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.ops.lake_health import (
    lake_dataset_quality_summary,
    lake_file_health_summary,
    write_lake_file_health_daily,
)


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


def test_lake_file_health_ignores_internal_temp_and_backup_parquet(tmp_path):
    lake = tmp_path / "lake"
    dataset = lake / "silver" / "market_bar"
    write_parquet_dataset(
        pl.DataFrame([{"symbol": "BTC-USDT", "value": 1.0}]),
        dataset,
    )
    internal_tmp = dataset / "._tmp"
    internal_tmp.mkdir()
    pl.DataFrame([{"symbol": "ETH-USDT", "value": 2.0}]).write_parquet(
        internal_tmp / "temp.parquet"
    )
    internal_backup = lake / "silver" / "__market_bar_backup_abc"
    internal_backup.mkdir()
    pl.DataFrame([{"symbol": "SOL-USDT", "value": 3.0}]).write_parquet(
        internal_backup / "data.parquet"
    )

    summary = lake_file_health_summary(lake)
    market_bar = next(row for row in summary["rows"] if row["dataset"] == "market_bar")

    assert market_bar["parquet_file_count"] == 1


def test_lake_dataset_quality_summary_is_read_only(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame([{"symbol": "BTC-USDT", "value": 1.0}]),
        lake / "silver" / "market_bar",
    )

    summary = lake_dataset_quality_summary(lake, dataset_names=["market_bar"])

    assert summary["dataset_count"] == 1
    assert summary["check_count"] > 0
    assert not (lake / "gold" / "lake_file_health_daily").exists()
