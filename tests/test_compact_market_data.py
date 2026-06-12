import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

import quant_lab.data.file_index as file_index_module
import quant_lab.jobs.compact_market_data as compact_market_data_module
from quant_lab.data.file_index import build_lake_file_index, old_files_for_dataset
from quant_lab.data.lake import (
    append_parquet_dataset,
    count_parquet_rows,
    read_parquet_dataset,
    write_parquet_dataset,
)
from quant_lab.jobs.compact_market_data import (
    build_market_data_1m_rollups,
    build_orderbook_spread_1m_rollup,
    build_trade_activity_1m_rollup,
    compact_market_data,
)
from quant_lab.jobs.small_file_maintenance import (
    lake_small_file_maintenance,
    small_file_groups,
)


def test_market_data_rollups_generate_1m_tables(tmp_path):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 31, 10, 0, 15, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"symbol": "BNB-USDT", "ts": ts, "size": 1.0, "side": "buy"},
                {
                    "symbol": "BNB-USDT",
                    "ts": ts + timedelta(seconds=10),
                    "size": 2.0,
                    "side": "sell",
                },
            ]
        ),
        lake / "silver/trade_print",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "channel": "books5",
                    "ts": ts,
                    "asks_json": '[["101", "1"]]',
                    "bids_json": '[["100", "3"]]',
                }
            ]
        ),
        lake / "silver/orderbook_snapshot",
    )

    trades = build_trade_activity_1m_rollup(lake)
    spreads = build_orderbook_spread_1m_rollup(lake)

    assert trades["trade_count"][0] == 2
    assert trades["size_sum"][0] == 3.0
    assert trades["taker_buy_size_sum"][0] == 1.0
    assert trades["taker_sell_size_sum"][0] == 2.0
    assert spreads["spread_bps"][0] > 0
    assert spreads["orderbook_imbalance"][0] == 0.5


def test_market_data_rollups_are_written_idempotently(tmp_path):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 31, 10, 0, 15, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"symbol": "BNB-USDT", "ts": ts, "size": 1.0, "side": "buy"},
                {
                    "symbol": "BNB-USDT",
                    "ts": ts + timedelta(seconds=10),
                    "size": 2.0,
                    "side": "sell",
                },
            ]
        ),
        lake / "silver/trade_print",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "channel": "books5",
                    "ts": ts,
                    "asks_json": '[["101", "1"]]',
                    "bids_json": '[["100", "1"]]',
                }
            ]
        ),
        lake / "silver/orderbook_snapshot",
    )

    first = build_market_data_1m_rollups(lake, dry_run=False)
    second = build_market_data_1m_rollups(lake, dry_run=False)

    assert (
        first.rollup_rows
        == second.rollup_rows
        == {
            "trade_activity_1m": 1,
            "orderbook_spread_1m": 1,
        }
    )
    assert count_parquet_rows(lake / "silver/trade_activity_1m") == 1
    assert count_parquet_rows(lake / "silver/orderbook_spread_1m") == 1
    trade_row = read_parquet_dataset(lake / "silver/trade_activity_1m").to_dicts()[0]
    spread_row = read_parquet_dataset(lake / "silver/orderbook_spread_1m").to_dicts()[0]
    assert trade_row["trade_count"] == 2
    assert trade_row["taker_buy_size_sum"] == 1.0
    assert trade_row["taker_sell_size_sum"] == 2.0
    assert spread_row["spread_bps"] > 0
    assert "orderbook_imbalance" in spread_row


def test_market_data_rollups_parse_iso_string_timestamps(tmp_path):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 31, 10, 0, 15, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"symbol": "BNB-USDT", "ts": ts.isoformat().replace("+00:00", "Z"), "size": 1.0},
                {
                    "symbol": "BNB-USDT",
                    "ts": (ts + timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
                    "size": 2.0,
                },
            ]
        ),
        lake / "silver/trade_print",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "channel": "books5",
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "asks_json": '[["101", "1"]]',
                    "bids_json": '[["100", "1"]]',
                }
            ]
        ),
        lake / "silver/orderbook_snapshot",
    )

    result = build_market_data_1m_rollups(lake, dry_run=False)

    assert result.rollup_rows == {"trade_activity_1m": 1, "orderbook_spread_1m": 1}
    assert read_parquet_dataset(lake / "silver/trade_activity_1m")["trade_count"][0] == 2


def test_market_data_rollup_lookback_skips_old_source_files(tmp_path):
    lake = tmp_path / "lake"
    source = lake / "silver/trade_print"
    source.mkdir(parents=True)
    old_ts = datetime(2026, 5, 30, 10, 0, tzinfo=UTC)
    new_ts = datetime(2026, 5, 31, 10, 0, tzinfo=UTC)
    old_file = source / "old.parquet"
    new_file = source / "new.parquet"
    pl.DataFrame([{"symbol": "BNB-USDT", "ts": old_ts, "size": 100.0}]).write_parquet(old_file)
    pl.DataFrame([{"symbol": "BNB-USDT", "ts": new_ts, "size": 2.0}]).write_parquet(new_file)
    old_mtime = old_ts.timestamp()
    new_mtime = new_ts.timestamp()
    os.utime(old_file, (old_mtime, old_mtime))
    os.utime(new_file, (new_mtime, new_mtime))

    result = build_market_data_1m_rollups(
        lake,
        dry_run=False,
        now=new_ts + timedelta(hours=1),
        lookback_hours=2,
    )

    assert result.rollup_rows["trade_activity_1m"] == 1
    trade_row = read_parquet_dataset(lake / "silver/trade_activity_1m").to_dicts()[0]
    assert trade_row["size_sum"] == 2.0


def test_market_data_rollup_merges_unindexed_recent_files_without_rebuilding_index(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    source = lake / "silver/trade_print"
    source.mkdir(parents=True)
    indexed_ts = datetime(2026, 5, 31, 8, 0, tzinfo=UTC)
    new_ts = datetime(2026, 5, 31, 10, 0, tzinfo=UTC)
    indexed_file = source / "indexed.parquet"
    new_file = source / "not-yet-indexed.parquet"
    pl.DataFrame([{"symbol": "BNB-USDT", "ts": indexed_ts, "size": 1.0}]).write_parquet(
        indexed_file
    )
    os.utime(indexed_file, (indexed_ts.timestamp(), indexed_ts.timestamp()))
    build_lake_file_index(lake, ["silver/trade_print"])

    def fail_full_file_time_scan(_file_path):
        raise AssertionError("rollup should not rebuild the full source file index")

    monkeypatch.setattr(file_index_module, "_file_time_bounds", fail_full_file_time_scan)
    pl.DataFrame([{"symbol": "BNB-USDT", "ts": new_ts, "size": 7.0}]).write_parquet(new_file)
    os.utime(new_file, (new_ts.timestamp(), new_ts.timestamp()))

    result = build_market_data_1m_rollups(
        lake,
        dry_run=False,
        now=new_ts + timedelta(hours=1),
        lookback_hours=2,
    )

    assert result.rollup_rows["trade_activity_1m"] == 1
    trade_row = read_parquet_dataset(lake / "silver/trade_activity_1m").to_dicts()[0]
    assert trade_row["size_sum"] == 7.0
    assert any(
        item.startswith("file_index_stale_merged_recent_mtime_files:")
        for item in result.warnings
    )


def test_orderbook_spread_rollup_prefers_spread_bps_without_json_udf(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 31, 10, 0, 15, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "channel": "books5",
                    "ts": ts,
                    "spread_bps": 12.5,
                    "asks_json": '[["999", "1"]]',
                    "bids_json": '[["1", "3"]]',
                }
            ]
        ),
        lake / "silver/orderbook_snapshot",
    )

    def fail_json_udf(_row):
        raise AssertionError("spread_bps fast path should not parse orderbook JSON")

    monkeypatch.setattr(compact_market_data_module, "_spread_bps", fail_json_udf)

    spreads = build_orderbook_spread_1m_rollup(lake)

    assert spreads.height == 1
    assert spreads["spread_bps"][0] == 12.5
    assert spreads["orderbook_imbalance"][0] == 0.5


def test_orderbook_spread_rollup_uses_vectorized_json_fast_path(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 31, 10, 0, 15, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "channel": "books5",
                    "ts": ts,
                    "asks_json": '[["101", "1"]]',
                    "bids_json": '[["100", "1"]]',
                }
            ]
        ),
        lake / "silver/orderbook_snapshot",
    )
    warnings: list[str] = []

    def fail_json_udf(_row):
        raise AssertionError("orderbook JSON rollup should use the vectorized fast path")

    monkeypatch.setattr(compact_market_data_module, "_spread_bps", fail_json_udf)
    monkeypatch.setattr(compact_market_data_module, "_book_imbalance", fail_json_udf)

    spreads = build_orderbook_spread_1m_rollup(lake, warnings=warnings)

    assert spreads.height == 1
    assert spreads["spread_bps"][0] > 0
    assert spreads["orderbook_imbalance"][0] == 0.0
    assert "orderbook_rollup_python_udf_fallback" not in warnings


def test_recent_file_selection_uses_index_max_ts_not_mtime(tmp_path):
    lake = tmp_path / "lake"
    source = lake / "silver/trade_print"
    source.mkdir(parents=True)
    old_mtime_ts = datetime(2026, 5, 30, 10, 0, tzinfo=UTC)
    recent_data_ts = datetime(2026, 5, 31, 10, 0, tzinfo=UTC)
    file_path = source / "recent-data-old-mtime.parquet"
    pl.DataFrame([{"symbol": "BNB-USDT", "ts": recent_data_ts, "size": 2.0}]).write_parquet(
        file_path
    )
    old_mtime = old_mtime_ts.timestamp()
    os.utime(file_path, (old_mtime, old_mtime))
    build_lake_file_index(lake, ["silver/trade_print"])

    trades = build_trade_activity_1m_rollup(
        lake,
        since=recent_data_ts - timedelta(hours=1),
    )

    assert trades.height == 1
    assert trades["size_sum"][0] == 2.0


def test_old_file_selection_uses_index_max_ts_before_cutoff(tmp_path):
    lake = tmp_path / "lake"
    source = lake / "bronze/okx_public_ws"
    source.mkdir(parents=True)
    old_ts = datetime(2026, 5, 30, 10, 0, tzinfo=UTC)
    hot_ts = datetime(2026, 5, 31, 10, 0, tzinfo=UTC)
    old = source / "old.parquet"
    hot = source / "hot.parquet"
    pl.DataFrame([{"symbol": "BNB-USDT", "received_at": old_ts}]).write_parquet(old)
    pl.DataFrame([{"symbol": "BNB-USDT", "received_at": hot_ts}]).write_parquet(hot)
    build_lake_file_index(lake, ["bronze/okx_public_ws"])

    files = old_files_for_dataset(source, before=datetime(2026, 5, 31, tzinfo=UTC))

    assert files == [old]


def test_lake_file_index_reuses_unchanged_rows_and_scans_only_new_files(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    source = lake / "silver/trade_print"
    source.mkdir(parents=True)
    first = source / "first.parquet"
    second = source / "second.parquet"
    pl.DataFrame(
        [{"symbol": "BNB-USDT", "ts": datetime(2026, 5, 31, 9, tzinfo=UTC), "size": 1.0}]
    ).write_parquet(first)
    pl.DataFrame(
        [{"symbol": "BNB-USDT", "ts": datetime(2026, 5, 31, 10, tzinfo=UTC), "size": 2.0}]
    ).write_parquet(second)
    original_bounds = file_index_module._file_time_bounds
    build_lake_file_index(lake, ["silver/trade_print"])

    def fail_scan(_path):
        raise AssertionError("unchanged file should reuse indexed bounds")

    monkeypatch.setattr(file_index_module, "_file_time_bounds", fail_scan)
    reused = build_lake_file_index(lake, ["silver/trade_print"])
    assert reused.height == 2
    assert "indexed_at" in reused.columns
    assert "index_version" in reused.columns
    assert reused.get_column("reused_from_previous_index").to_list() == [True, True]

    third = source / "third.parquet"
    pl.DataFrame(
        [{"symbol": "BNB-USDT", "ts": datetime(2026, 5, 31, 11, tzinfo=UTC), "size": 3.0}]
    ).write_parquet(third)
    scanned: list[str] = []

    def count_new_file_scan(path):
        if Path(path).name != "third.parquet":
            raise AssertionError("only newly added files should be scanned")
        scanned.append(Path(path).name)
        return original_bounds(path)

    monkeypatch.setattr(file_index_module, "_file_time_bounds", count_new_file_scan)
    updated = build_lake_file_index(lake, ["silver/trade_print"])

    assert updated.height == 3
    assert scanned == ["third.parquet"]
    reused_flags = updated.sort("path").get_column("reused_from_previous_index").to_list()
    assert reused_flags.count(True) == 2
    assert reused_flags.count(False) == 1


def test_small_file_maintenance_compacts_priority_partition_groups(tmp_path):
    lake = tmp_path / "lake"
    dataset = lake / "silver" / "v5_quant_lab_request"
    for index in range(18):
        append_parquet_dataset(
            pl.DataFrame(
                [
                    {
                        "event_key": f"req-{index}",
                        "ts_utc": datetime(2026, 5, 31, 10, index % 60, tzinfo=UTC),
                        "run_id": "20260531_10",
                    }
                ]
            ),
            dataset,
            target_rows_per_file=1,
        )
    build_lake_file_index(lake, ["silver/v5_quant_lab_request"])

    groups = small_file_groups(lake, min_files=16, max_avg_file_size_mb=8)
    assert groups
    assert groups[0].dataset == "silver/v5_quant_lab_request"

    result = lake_small_file_maintenance(
        lake,
        min_files=16,
        max_groups=5,
        target_rows_per_file=250_000,
        max_source_files_per_batch=64,
        dry_run=False,
    )

    assert result.compacted_group_count == 1
    assert result.before_file_count == 18
    assert result.after_file_count == 1
    assert result.source_file_count == 18
    assert result.output_file_count == 1
    assert count_parquet_rows(dataset) == 18
    assert len(list(dataset.glob("*.parquet"))) == 1


def test_small_file_maintenance_can_consolidate_existing_compact_outputs(tmp_path):
    lake = tmp_path / "lake"
    dataset = lake / "silver" / "trade_print"
    dataset.mkdir(parents=True)
    for index in range(18):
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "ts": datetime(2026, 5, 31, 10, index % 60, tzinfo=UTC),
                    "size": float(index + 1),
                }
            ]
        ).write_parquet(dataset / f"compact_existing_{index}.parquet")
    build_lake_file_index(lake, ["silver/trade_print"])

    groups = small_file_groups(lake, min_files=16, max_avg_file_size_mb=8)
    group = next(item for item in groups if item.dataset == "silver/trade_print")
    assert group.compact_file_count == 18
    assert group.include_existing_compact_files is True

    result = lake_small_file_maintenance(
        lake,
        min_files=16,
        max_groups=5,
        target_rows_per_file=250_000,
        max_source_files_per_batch=64,
        dry_run=False,
    )

    assert result.compacted_group_count == 1
    assert result.source_file_count == 18
    assert result.output_file_count == 1
    assert count_parquet_rows(dataset) == 18
    assert len(list(dataset.glob("*.parquet"))) == 1


def test_small_file_maintenance_limits_source_files_per_group(tmp_path):
    lake = tmp_path / "lake"
    dataset = lake / "bronze" / "okx_public_ws"
    dataset.mkdir(parents=True)
    for index in range(20):
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "ts": datetime(2026, 5, 31, 10, index % 60, tzinfo=UTC),
                    "payload": f"event-{index}",
                }
            ]
        ).write_parquet(dataset / f"compact_existing_{index}.parquet")
    build_lake_file_index(lake, ["bronze/okx_public_ws"])

    result = lake_small_file_maintenance(
        lake,
        min_files=16,
        max_groups=5,
        target_rows_per_file=250_000,
        max_source_files_per_batch=8,
        max_source_files_per_group=8,
        dry_run=False,
    )

    assert result.compacted_group_count == 1
    assert result.source_file_count == 8
    assert result.output_file_count == 1
    assert count_parquet_rows(dataset) == 20
    assert len(list(dataset.glob("*.parquet"))) == 13


def test_rollup_records_warning_when_file_index_missing(tmp_path):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 31, 10, 0, 15, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame([{"symbol": "BNB-USDT", "ts": ts, "size": 2.0}]),
        lake / "silver/trade_print",
    )
    warnings: list[str] = []

    trades = build_trade_activity_1m_rollup(
        lake,
        since=ts - timedelta(hours=1),
        warnings=warnings,
    )

    assert trades.height == 1
    assert any(item.startswith("file_index_missing_fallback_rglob:") for item in warnings)


def test_compact_market_data_archives_only_old_ws_files_when_applied(tmp_path):
    lake = tmp_path / "lake"
    hot = lake / "bronze/okx_public_ws/hot.parquet"
    old = lake / "bronze/okx_public_ws/old.parquet"
    hot.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{"symbol": "BNB-USDT", "received_at": datetime(2026, 5, 31, tzinfo=UTC)}]
    ).write_parquet(hot)
    pl.DataFrame(
        [{"symbol": "BNB-USDT", "received_at": datetime(2026, 5, 30, tzinfo=UTC)}]
    ).write_parquet(old)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    old_mtime = (now - timedelta(hours=30)).timestamp()
    hot_mtime = (now - timedelta(hours=1)).timestamp()
    os.utime(old, (old_mtime, old_mtime))
    os.utime(hot, (hot_mtime, hot_mtime))
    build_lake_file_index(lake, ["bronze/okx_public_ws"])

    result = compact_market_data(lake, dry_run=False, now=now)

    assert str(old) in result.archived_files
    assert hot.exists()
    assert not old.exists()
    assert not any(item.startswith("archive_fallback_rglob:") for item in result.warnings)


def test_compact_market_data_warns_when_archive_file_index_missing(
    tmp_path,
):
    lake = tmp_path / "lake"
    old = lake / "bronze/okx_public_ws/old.parquet"
    old.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{"symbol": "BNB-USDT", "received_at": datetime(2026, 5, 30, tzinfo=UTC)}]
    ).write_parquet(old)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    old_mtime = (now - timedelta(hours=30)).timestamp()
    os.utime(old, (old_mtime, old_mtime))

    result = compact_market_data(lake, dry_run=False, now=now)

    assert str(old) in result.archived_files
    assert any(item.startswith("archive_fallback_rglob:") for item in result.warnings)
