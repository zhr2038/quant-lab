import os
from datetime import UTC, datetime, timedelta

import polars as pl

import quant_lab.jobs.compact_market_data as compact_market_data_module
from quant_lab.data.file_index import build_lake_file_index
from quant_lab.data.lake import (
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


def test_market_data_rollups_generate_1m_tables(tmp_path):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 31, 10, 0, 15, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"symbol": "BNB-USDT", "ts": ts, "size": 1.0},
                {"symbol": "BNB-USDT", "ts": ts + timedelta(seconds=10), "size": 2.0},
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
                    "asks_json": "[[\"101\", \"1\"]]",
                    "bids_json": "[[\"100\", \"1\"]]",
                }
            ]
        ),
        lake / "silver/orderbook_snapshot",
    )

    trades = build_trade_activity_1m_rollup(lake)
    spreads = build_orderbook_spread_1m_rollup(lake)

    assert trades["trade_count"][0] == 2
    assert trades["size_sum"][0] == 3.0
    assert spreads["spread_bps"][0] > 0


def test_market_data_rollups_are_written_idempotently(tmp_path):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 31, 10, 0, 15, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"symbol": "BNB-USDT", "ts": ts, "size": 1.0},
                {"symbol": "BNB-USDT", "ts": ts + timedelta(seconds=10), "size": 2.0},
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
                    "asks_json": "[[\"101\", \"1\"]]",
                    "bids_json": "[[\"100\", \"1\"]]",
                }
            ]
        ),
        lake / "silver/orderbook_snapshot",
    )

    first = build_market_data_1m_rollups(lake, dry_run=False)
    second = build_market_data_1m_rollups(lake, dry_run=False)

    assert first.rollup_rows == second.rollup_rows == {
        "trade_activity_1m": 1,
        "orderbook_spread_1m": 1,
    }
    assert count_parquet_rows(lake / "silver/trade_activity_1m") == 1
    assert count_parquet_rows(lake / "silver/orderbook_spread_1m") == 1
    trade_row = read_parquet_dataset(lake / "silver/trade_activity_1m").to_dicts()[0]
    spread_row = read_parquet_dataset(lake / "silver/orderbook_spread_1m").to_dicts()[0]
    assert trade_row["trade_count"] == 2
    assert spread_row["spread_bps"] > 0


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
                    "asks_json": "[[\"101\", \"1\"]]",
                    "bids_json": "[[\"100\", \"1\"]]",
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
    pl.DataFrame([{"symbol": "BNB-USDT", "ts": old_ts, "size": 100.0}]).write_parquet(
        old_file
    )
    pl.DataFrame([{"symbol": "BNB-USDT", "ts": new_ts, "size": 2.0}]).write_parquet(
        new_file
    )
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
                    "asks_json": "[[\"999\", \"1\"]]",
                    "bids_json": "[[\"1\", \"1\"]]",
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


def test_orderbook_spread_rollup_warns_on_legacy_json_udf_fallback(tmp_path):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 31, 10, 0, 15, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "channel": "books5",
                    "ts": ts,
                    "asks_json": "[[\"101\", \"1\"]]",
                    "bids_json": "[[\"100\", \"1\"]]",
                }
            ]
        ),
        lake / "silver/orderbook_snapshot",
    )
    warnings: list[str] = []

    spreads = build_orderbook_spread_1m_rollup(lake, warnings=warnings)

    assert spreads.height == 1
    assert "orderbook_rollup_python_udf_fallback" in warnings


def test_recent_file_selection_uses_index_max_ts_not_mtime(tmp_path):
    lake = tmp_path / "lake"
    source = lake / "silver/trade_print"
    source.mkdir(parents=True)
    old_mtime_ts = datetime(2026, 5, 30, 10, 0, tzinfo=UTC)
    recent_data_ts = datetime(2026, 5, 31, 10, 0, tzinfo=UTC)
    file_path = source / "recent-data-old-mtime.parquet"
    pl.DataFrame(
        [{"symbol": "BNB-USDT", "ts": recent_data_ts, "size": 2.0}]
    ).write_parquet(file_path)
    old_mtime = old_mtime_ts.timestamp()
    os.utime(file_path, (old_mtime, old_mtime))
    build_lake_file_index(lake, ["silver/trade_print"])

    trades = build_trade_activity_1m_rollup(
        lake,
        since=recent_data_ts - timedelta(hours=1),
    )

    assert trades.height == 1
    assert trades["size_sum"][0] == 2.0


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

    result = compact_market_data(lake, dry_run=False, now=now)

    assert str(old) in result.archived_files
    assert hot.exists()
    assert not old.exists()
