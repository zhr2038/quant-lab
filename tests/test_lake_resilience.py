from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pytest

from quant_lab.data.lake import (
    _lock_is_stale,
    append_parquet_dataset,
    compact_parquet_dataset,
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


def test_read_parquet_dataset_ignores_internal_tmp_dir(tmp_path):
    dataset = tmp_path / "lake" / "bronze" / "api_request_metrics"
    write_parquet_dataset(pl.DataFrame([{"path": "/v1/health", "count": 1}]), dataset)
    tmp_dir = dataset / "._tmp"
    tmp_dir.mkdir()
    pl.DataFrame([{"path": "/v1/should_not_read", "count": 99}]).write_parquet(
        tmp_dir / "leftover.tmp.parquet"
    )

    df = read_parquet_dataset(dataset)

    assert df.height == 1
    assert df["path"].to_list() == ["/v1/health"]


def test_read_parquet_dataset_inserts_missing_columns_across_schema_versions(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "v5_paper_strategy_run"
    dataset.mkdir(parents=True)
    pl.DataFrame([{"strategy_id": "SOL_PAPER", "symbol": "SOL-USDT"}]).write_parquet(
        dataset / "part-old.parquet"
    )
    pl.DataFrame(
        [
            {
                "strategy_id": "SOL_PAPER",
                "symbol": "SOL-USDT",
                "no_sample_reason": "heartbeat_no_candidate",
            }
        ]
    ).write_parquet(dataset / "part-new.parquet")

    df = read_parquet_dataset(dataset)

    assert df.height == 2
    assert "no_sample_reason" in df.columns
    assert df.filter(pl.col("no_sample_reason").is_null()).height == 1


def test_read_parquet_dataset_ignores_extra_columns_across_schema_versions(tmp_path):
    dataset = tmp_path / "lake" / "bronze" / "okx_public_ws"
    dataset.mkdir(parents=True)
    pl.DataFrame(
        [{"channel": "trades", "inst_id": "BTC-USDT", "raw_json": "{}"}]
    ).write_parquet(dataset / "part-base.parquet")
    pl.DataFrame(
        [
            {
                "channel": "trades",
                "inst_id": "BTC-USDT",
                "raw_json": "{}",
                "day": "2026-05-18",
            }
        ]
    ).write_parquet(dataset / "part-with-partition-column.parquet")

    df = read_parquet_dataset(dataset)

    assert df.height == 2
    assert "day" not in df.columns
    assert set(df["inst_id"].to_list()) == {"BTC-USDT"}


def test_read_parquet_dataset_relaxes_null_string_schema_versions(tmp_path):
    dataset = tmp_path / "lake" / "gold" / "job_run_history"
    dataset.mkdir(parents=True)
    pl.DataFrame([{"job_name": "ok", "error_type": None}]).write_parquet(
        dataset / "part-null.parquet"
    )
    pl.DataFrame([{"job_name": "failed", "error_type": "SchemaError"}]).write_parquet(
        dataset / "part-string.parquet"
    )

    df = read_parquet_dataset(dataset)

    assert df.height == 2
    assert df.schema["error_type"] == pl.String
    assert set(df["job_name"].to_list()) == {"ok", "failed"}


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


def test_append_parquet_dataset_writes_partitioned_batches(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "trade_print"
    result = append_parquet_dataset(
        pl.DataFrame(
            [
                {"day": "2026-05-18", "symbol": "BTC-USDT", "value": 1},
                {"day": "2026-05-18", "symbol": "ETH-USDT", "value": 2},
            ]
        ),
        dataset,
        partition_by=["day", "symbol"],
    )

    files = list(dataset.rglob("*.parquet"))
    read_back = read_parquet_dataset(dataset)

    assert result.rows_written == 2
    assert result.file_count == 2
    assert len(files) == 2
    assert (dataset / "day=2026-05-18" / "symbol=BTC-USDT").exists()
    assert read_back.height == 2
    assert set(read_back["symbol"].to_list()) == {"BTC-USDT", "ETH-USDT"}


def test_append_parquet_dataset_does_not_leave_parquet_visible_on_failed_write(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "lake" / "silver" / "trade_print"
    original_write_parquet = pl.DataFrame.write_parquet

    def failing_write_parquet(self, file, *args, **kwargs):
        file.write_bytes(b"partial")
        raise OSError("simulated append failure")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", failing_write_parquet)

    with pytest.raises(OSError, match="simulated append failure"):
        append_parquet_dataset(
            pl.DataFrame([{"day": "2026-05-18", "symbol": "BTC-USDT", "value": 1}]),
            dataset,
            partition_by=["day", "symbol"],
        )

    monkeypatch.setattr(pl.DataFrame, "write_parquet", original_write_parquet)

    assert list(dataset.rglob("*.parquet")) == []
    assert read_parquet_dataset(dataset).is_empty()


def test_compact_parquet_dataset_preserves_rows_and_reduces_files(tmp_path):
    dataset = tmp_path / "lake" / "bronze" / "okx_public_ws"
    for index in range(5):
        append_parquet_dataset(
            pl.DataFrame(
                [
                    {
                        "day": "2026-05-18",
                        "channel": "trades",
                        "inst_id": "BTC-USDT",
                        "value": index,
                    }
                ]
            ),
            dataset,
            partition_by=["day", "channel", "inst_id"],
            target_rows_per_file=1,
        )

    before = len(list(dataset.rglob("*.parquet")))
    result = compact_parquet_dataset(
        dataset,
        partition_by=["day", "channel", "inst_id"],
        target_rows_per_file=10,
    )
    after = len(list(dataset.rglob("*.parquet")))
    read_back = read_parquet_dataset(dataset)

    assert before == 5
    assert after == 1
    assert result.source_file_count == 5
    assert result.output_file_count == 1
    assert read_back.height == 5
    assert sorted(read_back["value"].to_list()) == list(range(5))
