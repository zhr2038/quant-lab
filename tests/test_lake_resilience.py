import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_lab.data.lake import (
    _lock_is_stale,
    _parquet_file_batches,
    _remove_internal_path,
    append_parquet_dataset,
    compact_parquet_dataset,
    compact_parquet_directory_files,
    invalid_parquet_files,
    read_parquet_dataset,
    repair_parquet_partition_values,
    upsert_parquet_dataset,
    write_parquet_dataset,
    write_snapshot_meta,
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode repair is not meaningful on Windows")
def test_remove_internal_path_repairs_unwritable_internal_directory(tmp_path):
    dataset = tmp_path / "lake" / "gold" / "lake_file_health_daily"
    internal = dataset.parent / "__lake_file_health_daily_backup_test"
    internal.mkdir(parents=True)
    data_file = internal / "data.parquet"
    data_file.write_bytes(b"PAR1xxxxPAR1")
    data_file.chmod(0o444)
    internal.chmod(0o555)

    _remove_internal_path(internal)

    assert not internal.exists()


def test_write_parquet_dataset_falls_back_when_dataset_tmp_path_is_unusable(tmp_path):
    dataset = tmp_path / "lake" / "gold" / "research_portfolio_status"
    dataset.mkdir(parents=True)
    (dataset / "._tmp").write_text("not a directory", encoding="utf-8")

    write_parquet_dataset(pl.DataFrame([{"research_id": "eth_f3", "status": "PAPER"}]), dataset)

    df = read_parquet_dataset(dataset)

    assert df.to_dicts() == [{"research_id": "eth_f3", "status": "PAPER"}]
    assert not (dataset / "._tmp").exists()
    assert not list((dataset.parent / ".research_portfolio_status._tmp").glob("*.parquet"))


def test_write_parquet_dataset_replaces_directory_when_inner_file_replace_is_unusable(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "lake" / "gold" / "v5_risk_on_multi_buy_shadow"
    write_parquet_dataset(pl.DataFrame([{"symbol": "BTC-USDT", "value": 1}]), dataset)

    import quant_lab.data.lake as lake_module

    original_replace_path = lake_module._replace_path

    def guarded_replace_path(source: Path, target: Path) -> None:
        if Path(target) == dataset / "data.parquet":
            raise PermissionError("simulated dataset directory permission drift")
        original_replace_path(source, target)

    monkeypatch.setattr(lake_module, "_replace_path", guarded_replace_path)

    write_parquet_dataset(pl.DataFrame([{"symbol": "SOL-USDT", "value": 2}]), dataset)

    read_back = read_parquet_dataset(dataset)

    assert read_back.to_dicts() == [{"symbol": "SOL-USDT", "value": 2}]
    assert not list(dataset.parent.glob("__v5_risk_on_multi_buy_shadow_backup_*"))
    assert not list(dataset.parent.glob("__v5_risk_on_multi_buy_shadow_write_*"))


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


def test_record_job_run_bounds_history_rows(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_JOB_RUN_HISTORY_MAX_ROWS", "3")
    started = datetime(2026, 5, 19, 1, 0, tzinfo=UTC)
    for index in range(6):
        record_job_run(
            lake_root=lake_root,
            job_name=f"job-{index}",
            status="succeeded",
            started_at=started + timedelta(minutes=index),
            finished_at=started + timedelta(minutes=index, seconds=1),
        )

    rows = read_parquet_dataset(lake_root / "gold" / "job_run_history")

    assert rows.height == 3
    assert set(rows["job_name"].to_list()) == {"job-3", "job-4", "job-5"}
    assert len(list((lake_root / "gold" / "job_run_history").rglob("*.parquet"))) == 1


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


def test_partitioned_write_keeps_previous_data_when_rewrite_fails(tmp_path, monkeypatch):
    dataset = tmp_path / "lake" / "silver" / "market_bar_partitioned"
    write_parquet_dataset(
        pl.DataFrame([{"day": "2026-05-25", "symbol": "BTC-USDT", "close": 100.0}]),
        dataset,
        partition_by="day",
    )
    original = read_parquet_dataset(dataset)
    original_write_parquet = pl.DataFrame.write_parquet

    def failing_write_parquet(self, file, *args, **kwargs):
        if kwargs.get("partition_by") == "day":
            raise OSError("simulated interrupted partitioned parquet write")
        return original_write_parquet(self, file, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", failing_write_parquet)

    with pytest.raises(OSError, match="simulated interrupted partitioned parquet write"):
        write_parquet_dataset(
            pl.DataFrame([{"day": "2026-05-26", "symbol": "ETH-USDT", "close": 200.0}]),
            dataset,
            partition_by="day",
        )

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

    results = []
    for index in range(5):
        results.append(
            append_parquet_dataset(
                pl.DataFrame([{"channel": "trades", "inst_id": "BTC-USDT", "value": index}]),
                dataset,
                target_rows_per_file=1,
            )
        )

    files = list(dataset.glob("*.parquet"))
    read_back = read_parquet_dataset(dataset)

    compacted = [result for result in results if result.auto_compact_triggered]
    assert compacted
    assert compacted[0].compact_source_file_count > compacted[0].compact_output_file_count
    assert len(files) <= 3
    assert read_back.height == 5
    assert sorted(read_back["value"].to_list()) == list(range(5))
    assert not invalid_parquet_files(dataset)


def test_append_auto_compaction_respects_min_total_bytes(tmp_path, monkeypatch):
    dataset = tmp_path / "lake" / "bronze" / "api_request_metrics"
    monkeypatch.setenv("QUANT_LAB_APPEND_AUTO_COMPACT_FILES", "1")
    monkeypatch.setenv("QUANT_LAB_APPEND_AUTO_COMPACT_TARGET_ROWS", "100")
    monkeypatch.setenv("QUANT_LAB_APPEND_AUTO_COMPACT_MIN_TOTAL_BYTES", "100000000")

    results = [
        append_parquet_dataset(pl.DataFrame([{"value": index}]), dataset, target_rows_per_file=1)
        for index in range(3)
    ]

    assert not any(result.auto_compact_triggered for result in results)
    assert len(list(dataset.glob("*.parquet"))) == 3


def test_append_auto_compaction_skips_existing_compact_outputs(tmp_path, monkeypatch):
    dataset = tmp_path / "lake" / "bronze" / "okx_public_ws"
    dataset.mkdir(parents=True)
    pl.DataFrame([{"value": 100}]).write_parquet(dataset / "compact_existing.parquet")
    monkeypatch.setenv("QUANT_LAB_APPEND_AUTO_COMPACT_FILES", "2")
    monkeypatch.setenv("QUANT_LAB_APPEND_AUTO_COMPACT_TARGET_ROWS", "100")

    for index in range(3):
        append_parquet_dataset(
            pl.DataFrame([{"value": index}]),
            dataset,
            target_rows_per_file=1,
        )

    files = sorted(path.name for path in dataset.glob("*.parquet"))
    read_back = read_parquet_dataset(dataset)

    assert "compact_existing.parquet" in files
    assert sum(1 for name in files if name.startswith("compact_")) == 2
    assert sorted(read_back["value"].to_list()) == [0, 1, 2, 100]
    assert not invalid_parquet_files(dataset)


def test_append_parquet_dataset_auto_compacts_partition_leaf_files(tmp_path, monkeypatch):
    dataset = tmp_path / "lake" / "silver" / "trade_print"
    monkeypatch.setenv("QUANT_LAB_APPEND_AUTO_COMPACT_FILES", "3")
    monkeypatch.setenv("QUANT_LAB_APPEND_AUTO_COMPACT_TARGET_ROWS", "100")

    for index in range(5):
        append_parquet_dataset(
            pl.DataFrame(
                [
                    {
                        "day": "2026-05-23",
                        "symbol": "BTC-USDT",
                        "value": index,
                    }
                ]
            ),
            dataset,
            partition_by=["day", "symbol"],
            target_rows_per_file=1,
        )

    leaf = dataset / "day=2026-05-23" / "symbol=BTC-USDT"
    files = list(leaf.glob("*.parquet"))
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


def test_parquet_file_batches_respect_byte_limit(tmp_path):
    files = []
    for index, size in enumerate([10, 40, 40, 10]):
        path = tmp_path / f"file_{index}.parquet"
        path.write_bytes(b"x" * size)
        files.append(path)

    batches = _parquet_file_batches(
        files,
        max_source_files_per_batch=10,
        max_source_batch_bytes=50,
    )

    assert [[path.name for path in batch] for batch in batches] == [
        ["file_0.parquet", "file_1.parquet"],
        ["file_2.parquet", "file_3.parquet"],
    ]


def test_repair_parquet_partition_values_moves_null_day_partition(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "trade_print"
    bad_leaf = dataset / "day=__null__" / "symbol=BTC-USDT"
    bad_leaf.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "day": None,
                "trade_id": "1",
                "ts": None,
                "ingest_ts": "2026-05-23T01:02:03Z",
                "price": 100.0,
            }
        ]
    ).write_parquet(bad_leaf / "bad.parquet")

    result = repair_parquet_partition_values(
        dataset,
        partition_by=["day", "symbol"],
        target_rows_per_file=10,
    )
    read_back = read_parquet_dataset(dataset)

    assert result.bad_file_count == 1
    assert result.repaired_rows == 1
    assert result.removed_bad_file_count == 1
    assert not list(dataset.rglob("*__null__*"))
    assert list((dataset / "day=2026-05-23" / "symbol=BTC-USDT").glob("*.parquet"))
    assert read_back.height == 1
    assert read_back["day"][0] == "2026-05-23"
    assert read_back["ts"][0] == "2026-05-23T01:02:03Z"


def test_repair_parquet_partition_values_fills_raw_unknown_keys(tmp_path):
    dataset = tmp_path / "lake" / "bronze" / "okx_public_ws"
    bad_leaf = dataset / "day=2026-05-23" / "channel=__null__" / "inst_id=__null__"
    bad_leaf.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "day": "2026-05-23",
                "channel": None,
                "inst_id": None,
                "received_at": "2026-05-23T01:02:03Z",
                "raw_json": "{}",
            }
        ]
    ).write_parquet(bad_leaf / "bad.parquet")

    result = repair_parquet_partition_values(
        dataset,
        partition_by=["day", "channel", "inst_id"],
        target_rows_per_file=10,
    )
    read_back = read_parquet_dataset(dataset)

    assert result.bad_file_count == 1
    assert result.repaired_rows == 1
    assert not list(dataset.rglob("*__null__*"))
    assert list(
        (dataset / "day=2026-05-23" / "channel=unknown" / "inst_id=unknown").glob("*.parquet")
    )
    assert read_back["channel"][0] == "unknown"
    assert read_back["inst_id"][0] == "unknown"


def test_repair_parquet_partition_values_failure_does_not_write_visible_repairs(
    tmp_path,
    monkeypatch,
):
    dataset = tmp_path / "lake" / "silver" / "trade_print"
    bad_leaf = dataset / "day=__null__" / "symbol=BTC-USDT"
    bad_leaf.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "day": None,
                "trade_id": "1",
                "ts": None,
                "ingest_ts": "2026-05-23T01:02:03Z",
            }
        ]
    ).write_parquet(bad_leaf / "bad.parquet")

    import quant_lab.data.lake as lake_module

    original_move = lake_module._move_repaired_staging_files

    def fail_after_staging(staging: Path, dataset_path: Path) -> int:
        assert list(staging.rglob("*.parquet"))
        raise RuntimeError("simulated move failure")

    monkeypatch.setattr(lake_module, "_move_repaired_staging_files", fail_after_staging)

    with pytest.raises(RuntimeError, match="simulated move failure"):
        repair_parquet_partition_values(
            dataset,
            partition_by=["day", "symbol"],
            target_rows_per_file=10,
        )

    assert original_move
    assert list(bad_leaf.glob("*.parquet"))
    assert not list((dataset / "day=2026-05-23").rglob("*.parquet"))
    assert not list(dataset.parent.glob("__trade_print_repair_*"))


def test_repair_parquet_partition_values_removes_empty_bad_partition_dirs(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "trade_print"
    bad_leaf = dataset / "day=__null__" / "symbol=BTC-USDT"
    bad_leaf.mkdir(parents=True)

    result = repair_parquet_partition_values(
        dataset,
        partition_by=["day", "symbol"],
        target_rows_per_file=10,
    )

    assert result.bad_file_count == 0
    assert not list(dataset.rglob("*__null__*"))


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


def test_compact_parquet_directory_files_skips_existing_compact_outputs(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "orderbook_snapshot"
    dataset.mkdir(parents=True)
    pl.DataFrame([{"value": 100}]).write_parquet(dataset / "compact_existing.parquet")
    for index in range(3):
        pl.DataFrame([{"value": index}]).write_parquet(dataset / f"part-{index}.parquet")

    result = compact_parquet_directory_files(dataset, target_rows_per_file=10)
    direct_files = sorted(path.name for path in dataset.glob("*.parquet"))
    read_back = read_parquet_dataset(dataset)

    assert result.source_file_count == 3
    assert result.output_file_count == 1
    assert "compact_existing.parquet" in direct_files
    assert not any(name.startswith("part-") for name in direct_files)
    assert len(direct_files) == 2
    assert sorted(read_back["value"].to_list()) == [0, 1, 2, 100]


def test_compact_parquet_directory_files_preserves_schema_evolution_columns(tmp_path):
    dataset = tmp_path / "lake" / "bronze" / "api_request_metrics"
    dataset.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "day": "2026-06-18",
                "path": "/v1/health",
                "status_code": 200,
                "user_agent": "old-client",
            }
        ]
    ).write_parquet(dataset / "old-schema.parquet")
    pl.DataFrame(
        [
            {
                "day": "2026-06-18",
                "path": "/v1/health/deep",
                "status_code": 401,
                "user_agent": "new-client",
                "client_id": "v5.dashboard_proxy",
                "auth_result": "missing_bearer_token",
            }
        ]
    ).write_parquet(dataset / "new-schema.parquet")

    result = compact_parquet_directory_files(dataset, target_rows_per_file=10)
    read_back = read_parquet_dataset(dataset)

    assert result.source_file_count == 2
    assert {"client_id", "auth_result"} <= set(read_back.columns)
    by_path = {row["path"]: row for row in read_back.to_dicts()}
    assert by_path["/v1/health"]["client_id"] is None
    assert by_path["/v1/health/deep"]["client_id"] == "v5.dashboard_proxy"
    assert by_path["/v1/health/deep"]["auth_result"] == "missing_bearer_token"


def test_compact_parquet_directory_files_noops_when_only_compact_outputs_exist(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "trade_print"
    dataset.mkdir(parents=True)
    pl.DataFrame([{"value": 1}]).write_parquet(dataset / "compact_existing.parquet")

    result = compact_parquet_directory_files(dataset, target_rows_per_file=10)

    assert result.source_file_count == 0
    assert result.output_file_count == 0
    assert (dataset / "compact_existing.parquet").exists()
    assert read_parquet_dataset(dataset)["value"].to_list() == [1]


def test_compact_parquet_directory_files_can_consolidate_existing_compact_outputs(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "trade_print"
    dataset.mkdir(parents=True)
    for index in range(5):
        pl.DataFrame([{"value": index}]).write_parquet(dataset / f"compact_{index}.parquet")

    result = compact_parquet_directory_files(
        dataset,
        target_rows_per_file=10,
        include_existing_compact_files=True,
    )

    direct_files = sorted(path.name for path in dataset.glob("*.parquet"))
    read_back = read_parquet_dataset(dataset)

    assert result.source_file_count == 5
    assert result.output_file_count == 1
    assert len(direct_files) == 1
    assert direct_files[0].startswith("compact_")
    assert sorted(read_back["value"].to_list()) == [0, 1, 2, 3, 4]


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


def test_snapshot_source_sha_is_order_independent_and_content_sensitive(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "wide_evidence"
    first = pl.DataFrame(
        [
            {"id": 1, "value": "alpha", "generated_at": "2026-07-10T00:00:00Z"},
            {"id": 2, "value": "beta", "generated_at": "2026-07-10T00:01:00Z"},
        ]
    )
    write_snapshot_meta(dataset, dataset_name="wide_evidence", frame=first)
    first_meta = json.loads((dataset / "_snapshot_meta.json").read_text(encoding="utf-8"))

    write_snapshot_meta(dataset, dataset_name="wide_evidence", frame=first.reverse())
    reordered_meta = json.loads((dataset / "_snapshot_meta.json").read_text(encoding="utf-8"))

    changed = first.with_columns(
        pl.when(pl.col("id") == 2)
        .then(pl.lit("changed"))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    write_snapshot_meta(dataset, dataset_name="wide_evidence", frame=changed)
    changed_meta = json.loads((dataset / "_snapshot_meta.json").read_text(encoding="utf-8"))

    assert first_meta["source_sha"] == reordered_meta["source_sha"]
    assert first_meta["source_sha"] != changed_meta["source_sha"]
