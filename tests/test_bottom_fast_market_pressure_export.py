from __future__ import annotations

import zipfile
from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.features.fast_microstructure import build_fast_microstructure_features
from quant_lab.research.bottom_zone_reversal import build_bottom_zone_reversal_shadow
from quant_lab.research.market_pressure import build_market_pressure_score


def _market_rows(symbol: str, start: datetime, closes: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, close in enumerate(closes):
        ts = start + timedelta(hours=index)
        rows.append(
            {
                "venue": "okx",
                "symbol": symbol,
                "market_type": "spot",
                "timeframe": "1h",
                "ts": ts,
                "open": close * 1.01,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 1000.0 + index,
                "quote_volume": close * (1000.0 + index),
                "source": "test",
                "ingest_ts": ts + timedelta(minutes=1),
                "is_closed": True,
            }
        )
    return rows


def test_bottom_fast_microstructure_and_market_pressure_reports_export(tmp_path):
    lake = tmp_path / "lake"
    start = datetime(2026, 6, 1, 0, tzinfo=UTC)
    bnb_closes = [100.0] * 48 + [97.0, 94.0, 91.0, 90.5, 91.0, 92.0, 93.5, 94.5]
    rows = _market_rows("BNB-USDT", start, bnb_closes)
    rows.extend(_market_rows("BTC-USDT", start, [100.0] * 56))
    rows.extend(_market_rows("ETH-USDT", start, [100.0] * 56))
    rows.extend(_market_rows("SOL-USDT", start, [100.0] * 56))
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "market_bar")
    latest = start + timedelta(hours=len(bnb_closes) - 1)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "channel": "books5",
                    "minute_ts": latest - timedelta(minutes=minute),
                    "ts": latest - timedelta(minutes=minute),
                    "spread_bps": 8.0,
                }
                for minute in range(60)
            ]
        ),
        lake / "silver" / "orderbook_spread_1m",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "minute_ts": latest - timedelta(minutes=minute),
                    "latest_trade_ts": latest - timedelta(minutes=minute),
                    "trade_count": 3,
                    "size_sum": 10.0,
                }
                for minute in range(60)
            ]
        ),
        lake / "silver" / "trade_activity_1m",
    )

    market = pl.DataFrame(rows)
    spreads = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "channel": "books5",
                "minute_ts": latest - timedelta(minutes=minute),
                "ts": latest - timedelta(minutes=minute),
                "spread_bps": 8.0,
            }
            for minute in range(60)
        ]
    )
    trades = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": latest - timedelta(minutes=minute),
                "latest_trade_ts": latest - timedelta(minutes=minute),
                "trade_count": 3,
                "size_sum": 10.0,
            }
            for minute in range(60)
        ]
    )
    bottom = build_bottom_zone_reversal_shadow(
        market_bars=market,
        orderbook_spread_1m=spreads,
        trade_activity_1m=trades,
        generated_at=latest,
    )
    fast = build_fast_microstructure_features(
        market_bars=market,
        orderbook_spread_1m=spreads,
        trade_activity_1m=trades,
        generated_at=latest,
    )
    pressure = build_market_pressure_score(
        bottom_zone_reversal_shadow=bottom,
        fast_microstructure_features=fast,
        generated_at=latest,
    )

    assert bottom.filter(pl.col("symbol") == "BNB-USDT").height == 1
    assert "live_order_effect" in bottom.columns
    assert fast.filter(pl.col("symbol") == "BNB-USDT")["trade_count_60m"][0] == 180.0
    assert pressure["market_pressure_state"][0] in {
        "BOTTOM_PROBE_ALLOWED",
        "CAPITULATION_WATCH",
        "RISK_OFF_NO_CATCH",
        "RISK_ON_CONFIRMED",
    }

    result = export_daily_pack(
        export_date="2026-06-01",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/bottom_zone_reversal_shadow.csv" in names
        assert "reports/bottom_zone_reversal_summary.md" in names
        assert "reports/fast_microstructure_features.csv" in names
        assert "reports/market_pressure_score.csv" in names
        assert "reports/market_pressure_summary.md" in names
        assert "reports/late_breakout_failure_protect_shadow.csv" in names
        bottom_csv = archive.read("reports/bottom_zone_reversal_shadow.csv").decode()
        assert "read_only_no_live_order" in bottom_csv
