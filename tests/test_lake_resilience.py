from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pytest

from quant_lab.data.lake import (
    _lock_is_stale,
    invalid_parquet_files,
    read_parquet_dataset,
    upsert_parquet_dataset,
    write_parquet_dataset,
)


def test_read_parquet_dataset_ignores_invalid_parquet_file(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "orderbook_snapshot"
    write_parquet_dataset(pl.DataFrame([{"symbol": "BTC-USDT", "spread_bps": 1.25}]), dataset)
    bad_file = dataset / "partial.parquet"
    bad_file.write_bytes(b"bad")

    df = read_parquet_dataset(dataset)

    assert df.height == 1
    assert df.to_dicts()[0]["symbol"] == "BTC-USDT"
    assert invalid_parquet_files(dataset) == [bad_file]


def test_read_parquet_dataset_returns_empty_when_only_invalid_files_exist(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "orderbook_snapshot"
    dataset.mkdir(parents=True)
    (dataset / "data.parquet").write_bytes(b"bad")

    df = read_parquet_dataset(dataset)

    assert df.is_empty()
    assert invalid_parquet_files(dataset) == [dataset / "data.parquet"]


def test_write_parquet_dataset_keeps_previous_data_when_rewrite_fails(tmp_path, monkeypatch):
    dataset = tmp_path / "lake" / "silver" / "orderbook_snapshot"
    write_parquet_dataset(pl.DataFrame([{"symbol": "BTC-USDT", "spread_bps": 1.25}]), dataset)
    original = read_parquet_dataset(dataset)
    original_write_parquet = pl.DataFrame.write_parquet

    def failing_write_parquet(self, file, *args, **kwargs):
        file.write_bytes(b"partial")
        raise OSError("simulated interrupted parquet write")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", failing_write_parquet)

    with pytest.raises(OSError, match="simulated interrupted parquet write"):
        write_parquet_dataset(pl.DataFrame([{"symbol": "ETH-USDT", "spread_bps": 2.0}]), dataset)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", original_write_parquet)
    after = read_parquet_dataset(dataset)

    assert after.to_dicts() == original.to_dicts()
    assert not invalid_parquet_files(dataset)


def test_concurrent_upserts_do_not_corrupt_parquet(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "concurrent"

    def write_row(index: int) -> int:
        return upsert_parquet_dataset(
            pl.DataFrame([{"id": index, "value": f"row-{index}"}]),
            dataset,
            key_columns=["id"],
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write_row, range(20)))

    result = read_parquet_dataset(dataset)

    assert result.height == 20
    assert result["id"].sort().to_list() == list(range(20))
    assert not invalid_parquet_files(dataset)


def test_concurrent_writes_do_not_create_invalid_parquet(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "concurrent_replace"

    def write_row(index: int) -> None:
        write_parquet_dataset(
            pl.DataFrame([{"id": index, "value": f"row-{index}"}]),
            dataset,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write_row, range(20)))

    result = read_parquet_dataset(dataset)

    assert result.height == 1
    assert result["id"][0] in set(range(20))
    assert not invalid_parquet_files(dataset)


def test_empty_dataset_lock_becomes_stale(tmp_path, monkeypatch):
    lock = tmp_path / ".dataset.lock"
    lock.write_text("", encoding="ascii")

    monkeypatch.setattr("quant_lab.data.lake.time.time", lambda: lock.stat().st_mtime + 6)

    assert _lock_is_stale(lock)


def test_dead_pid_dataset_lock_is_stale(tmp_path):
    lock = tmp_path / ".dataset.lock"
    lock.write_text("999999999", encoding="ascii")

    assert _lock_is_stale(lock)
