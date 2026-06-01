import os
from datetime import UTC, datetime, timedelta

import polars as pl

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
