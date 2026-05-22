from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.data.lake import (
    _lock_is_stale,
    append_parquet_dataset,
    compact_parquet_dataset,
    compact_parquet_directory_files,
    invalid_parquet_files,
    read_parquet_dataset,
    upsert_parquet_dataset,
    write_parquet_dataset,
)
from quant_lab.ops.metrics import record_job_run


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
    pl.DataFrame([{"channel": "trades", "inst_id": "BTC-USDT", "raw_json": "{}"}]).write_parquet(
        dataset / "part-base.parquet"
    )
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


def test_record_job_run_rewrites_history_instead_of_appending_small_files(tmp_path):
    lake_root = tmp_path / "lake"
    for minute in range(3):
        started = datetime(2026, 5, 19, 1, minute, tzinfo=UTC)
        record_job_run(
            lake_root=lake_root,
            job_name="sync-v5-telemetry",
            status="succeeded",
            started_at=started,
            finished_at=started + timedelta(seconds=2),
        )

    files = list((lake_root / "gold" / "job_run_history").rglob("*.parquet"))
    rows = read_parquet_dataset(lake_root / "gold" / "job_run_history")

    assert len(files) == 1
    assert rows.height == 3


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


def test_append_parquet_dataset_auto_compacts_direct_small_files(tmp_path, monkeypatch):
    dataset = tmp_path / "lake" / "bronze" / "okx_public_ws"
    monkeypatch.setenv("QUANT_LAB_APPEND_AUTO_COMPACT_FILES", "3")
    monkeypatch.setenv("QUANT_LAB_APPEND_AUTO_COMPACT_TARGET_ROWS", "100")

    for index in range(5):
        append_parquet_dataset(
            pl.DataFrame([{"channel": "trades", "inst_id": "BTC-USDT", "value": index}]),
            dataset,
            target_rows_per_file=1,
        )

    files = list(dataset.glob("*.parquet"))
    read_back = read_parquet_dataset(dataset)

    assert len(files) <= 3
    assert read_back.height == 5
    assert sorted(read_back["value"].to_list()) == list(range(5))
    assert not invalid_parquet_files(dataset)


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


def test_compact_parquet_directory_files_preserves_partition_dirs(tmp_path):
    dataset = tmp_path / "lake" / "bronze" / "okx_public_ws"
    dataset.mkdir(parents=True)
    for index in range(5):
        pl.DataFrame([{"channel": "trades", "inst_id": "BTC-USDT", "value": index}]).write_parquet(
            dataset / f"batch-{index}.parquet"
        )
    partition_dir = dataset / "day=2026-05-18" / "channel=trades" / "inst_id=ETH-USDT"
    partition_dir.mkdir(parents=True)
    pl.DataFrame([{"channel": "trades", "inst_id": "ETH-USDT", "value": 99}]).write_parquet(
        partition_dir / "historical.parquet"
    )

    result = compact_parquet_directory_files(dataset, target_rows_per_file=10)
    direct_files = list(dataset.glob("*.parquet"))
    read_back = read_parquet_dataset(dataset)

    assert result.source_file_count == 5
    assert result.output_file_count == 1
    assert len(direct_files) == 1
    assert (partition_dir / "historical.parquet").exists()
    assert read_back.height == 6
    assert sorted(read_back["value"].to_list()) == [0, 1, 2, 3, 4, 99]


def test_read_parquet_dataset_ignores_internal_compaction_and_temp_files(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "v5_config_audit"
    dataset.mkdir(parents=True)
    pl.DataFrame([{"value": "live"}]).write_parquet(dataset / "data.parquet")

    backup = dataset.parent / "__v5_config_audit_backup_deadbeef"
    backup.mkdir()
    pl.DataFrame([{"value": "backup"}]).write_parquet(backup / "data.parquet")
    pl.DataFrame([{"value": "temp"}]).write_parquet(
        dataset.parent / ".v5_config_audit.orphan.tmp.parquet"
    )
    temp_dir = dataset / "._tmp"
    temp_dir.mkdir()
    pl.DataFrame([{"value": "hidden_tmp"}]).write_parquet(temp_dir / "staged.tmp.parquet")

    read_back = read_parquet_dataset(dataset)

    assert read_back["value"].to_list() == ["live"]
