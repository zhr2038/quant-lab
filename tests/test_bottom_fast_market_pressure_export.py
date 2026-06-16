from __future__ import annotations

import zipfile
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.features.fast_microstructure import (
    FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS,
    _forward_lookback_bars,
    _forward_recommendation,
    build_fast_microstructure_features,
    build_fast_microstructure_forward_test,
    build_fast_microstructure_strategy_candidates,
    fast_microstructure_forward_summary_md,
)
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


def test_fast_microstructure_forward_defaults_and_sample_gate(monkeypatch):
    monkeypatch.delenv("FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS", raising=False)
    monkeypatch.delenv("QUANT_LAB_FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS", raising=False)
    assert FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS == 2000
    assert _forward_lookback_bars() == 2000
    monkeypatch.setenv("FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS", "2400")
    assert _forward_lookback_bars() == 2400
    monkeypatch.setenv("FAST_MICROSTRUCTURE_FORWARD_LOOKBACK_BARS", "0")
    with pytest.raises(ValueError, match="must be >= 1"):
        _forward_lookback_bars()
    assert (
        _forward_recommendation(
            sample_count=29,
            rank_ic=0.20,
            long_short_bps=12.0,
            p25_net_bps=1.0,
            hit_rate=0.70,
        )
        == "NEEDS_MORE_FORWARD_SAMPLES"
    )
    assert (
        _forward_recommendation(
            sample_count=30,
            rank_ic=0.20,
            long_short_bps=12.0,
            p25_net_bps=1.0,
            hit_rate=0.70,
        )
        == "FORWARD_VALIDATION_PASS"
    )


def test_fast_microstructure_forward_adds_all_regimes_sample_pool():
    start = datetime(2026, 6, 1, tzinfo=UTC)
    market_rows = []
    spread_rows = []
    trade_rows = []
    for index in range(70):
        ts = start + timedelta(hours=index)
        close = 100.0 + 0.02 * (index**2)
        market_rows.append(
            {
                "venue": "okx",
                "symbol": "SOL-USDT",
                "market_type": "spot",
                "timeframe": "1h",
                "ts": ts,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000.0 + index,
                "quote_volume": close * (1000.0 + index),
                "source": "test",
                "ingest_ts": ts + timedelta(minutes=1),
                "is_closed": True,
                "regime": "SIDEWAYS" if index < 35 else "TREND_UP",
            }
        )
        spread_rows.append(
            {
                "symbol": "SOL-USDT",
                "minute_ts": ts,
                "ts": ts,
                "spread_bps": 1.0,
                "orderbook_imbalance": float(index),
            }
        )
        trade_rows.append(
            {
                "symbol": "SOL-USDT",
                "minute_ts": ts,
                "latest_trade_ts": ts,
                "trade_count": 10 + index,
                "size_sum": 100.0 + index,
                "taker_buy_size_sum": 60.0 + index,
                "taker_sell_size_sum": 40.0,
            }
        )

    forward = build_fast_microstructure_forward_test(
        market_bars=pl.DataFrame(market_rows),
        orderbook_spread_1m=pl.DataFrame(spread_rows),
        trade_activity_1m=pl.DataFrame(trade_rows),
        generated_at=start + timedelta(hours=70),
    )
    rows = {
        (row["regime"], row["horizon_hours"]): row
        for row in forward.filter(
            (pl.col("symbol") == "SOL-USDT")
            & (pl.col("feature_name") == "orderbook_imbalance_1m")
        ).to_dicts()
    }

    aggregate_8h = rows[("ALL_REGIMES", 8)]
    trend_8h = rows[("TREND_UP", 8)]
    assert aggregate_8h["sample_count"] == 62
    assert aggregate_8h["recommendation"] == "FORWARD_VALIDATION_PASS"
    assert trend_8h["sample_count"] == 27
    assert trend_8h["recommendation"] == "NEEDS_MORE_FORWARD_SAMPLES"
    assert aggregate_8h["live_order_effect"] == "read_only_no_live_order"


def test_fast_microstructure_strategy_candidates_only_use_forward_pass_rows():
    forward = pl.DataFrame(
        [
            {
                "generated_at": "2026-06-16T00:00:00Z",
                "feature_name": "orderbook_imbalance_1m",
                "symbol": "SOL-USDT",
                "regime": "SIDEWAYS",
                "horizon_hours": 8,
                "sample_count": 120,
                "rank_ic": 0.25,
                "long_short_bps": 44.0,
                "p25_net_bps": -10.0,
                "hit_rate": 0.61,
                "recent_7d_score": 0.5,
                "lookback_bars": 2000,
                "recommendation": "FORWARD_VALIDATION_PASS",
                "data_leakage_check": "future_price_used_only_as_label",
                "live_order_effect": "read_only_no_live_order",
            },
            {
                "generated_at": "2026-06-16T00:00:00Z",
                "feature_name": "orderbook_imbalance_5m",
                "symbol": "SOL-USDT",
                "regime": "ALL_REGIMES",
                "horizon_hours": 8,
                "sample_count": 120,
                "rank_ic": 0.25,
                "long_short_bps": 44.0,
                "p25_net_bps": -10.0,
                "hit_rate": 0.61,
                "recent_7d_score": 0.5,
                "lookback_bars": 2000,
                "recommendation": "FORWARD_VALIDATION_PASS",
                "data_leakage_check": "future_price_used_only_as_label",
                "live_order_effect": "read_only_no_live_order",
            },
            {
                "generated_at": "2026-06-16T00:00:00Z",
                "feature_name": "cvd_5m",
                "symbol": "SOL-USDT",
                "regime": "ALL_REGIMES",
                "horizon_hours": 8,
                "sample_count": 20,
                "rank_ic": 0.01,
                "long_short_bps": 4.0,
                "p25_net_bps": -80.0,
                "hit_rate": 0.48,
                "recent_7d_score": 0.1,
                "lookback_bars": 2000,
                "recommendation": "NEEDS_MORE_FORWARD_SAMPLES",
                "data_leakage_check": "future_price_used_only_as_label",
                "live_order_effect": "read_only_no_live_order",
            },
        ]
    )

    candidates = build_fast_microstructure_strategy_candidates(forward)
    rows = candidates.to_dicts()

    assert len(rows) == 1
    assert rows[0]["feature_name"] == "orderbook_imbalance_1m"
    assert rows[0]["generated_at"] == "2026-06-16T00:00:00Z"
    assert rows[0]["forward_sample_count"] == 120
    assert rows[0]["recent_7d_score"] == 0.5
    assert rows[0]["lookback_bars"] == 2000
    assert rows[0]["recommended_stage"] == "SHADOW_REVIEW"
    assert "needs_paper_tracking" in rows[0]["review_blocking_reasons"]
    assert rows[0]["data_leakage_check"] == "future_price_used_only_as_label"
    assert rows[0]["live_order_effect"] == "read_only_no_live_order"
    assert "sol_usdt" in rows[0]["candidate_strategy_id"]


def test_fast_microstructure_summary_explains_aggregate_only_passes():
    forward = pl.DataFrame(
        [
            {
                "generated_at": "2026-06-16T00:00:00Z",
                "feature_name": "orderbook_imbalance_5m",
                "symbol": "SOL-USDT",
                "regime": "ALL_REGIMES",
                "horizon_hours": 8,
                "sample_count": 120,
                "rank_ic": 0.25,
                "long_short_bps": 44.0,
                "p25_net_bps": -10.0,
                "hit_rate": 0.61,
                "recent_7d_score": 0.5,
                "lookback_bars": 2000,
                "recommendation": "FORWARD_VALIDATION_PASS",
                "data_leakage_check": "future_price_used_only_as_label",
                "live_order_effect": "read_only_no_live_order",
            }
        ]
    )

    summary = fast_microstructure_forward_summary_md(forward)

    assert "- pass_rows: 1" in summary
    assert "- aggregate_pass_rows: 1" in summary
    assert "- strategy_candidate_eligible_pass_rows: 0" in summary
    assert "aggregate ALL_REGIMES passes stay validation-only" in summary


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
                    "spread_bps": 8.0 + (0.01 * minute),
                    "orderbook_imbalance": 0.25,
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
                    "taker_buy_size_sum": 7.0,
                    "taker_sell_size_sum": 3.0,
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
                "spread_bps": 8.0 + (0.01 * minute),
                "orderbook_imbalance": 0.25,
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
                "taker_buy_size_sum": 7.0,
                "taker_sell_size_sum": 3.0,
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
    for field in (
        "support_zone_low",
        "support_zone_high",
        "distance_to_support_bps",
        "orderbook_imbalance_1m",
        "taker_buy_sell_imbalance_5m",
        "cvd_5m",
        "vwap_reclaim_15m",
        "volatility_climax_score",
        "bottom_zone_score",
        "bounce_probability_4h",
        "invalid_below_px",
        "no_trigger_reasons",
    ):
        assert field in bottom.columns
    assert fast.filter(pl.col("symbol") == "BNB-USDT")["trade_count_60m"][0] == 180.0
    fast_row = fast.filter(pl.col("symbol") == "BNB-USDT").to_dicts()[0]
    assert fast_row["orderbook_imbalance_1m"] == 0.25
    assert fast_row["orderbook_imbalance_5m"] == 0.25
    assert fast_row["taker_buy_sell_imbalance_5m"] > 0
    assert fast_row["cvd_5m"] > 0
    assert fast_row["cvd_divergence"] is not None
    assert fast_row["spread_bps_change_5m"] < 0
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
        assert "reports/backtest_label_summary.csv" in names
        assert "reports/v5_decision_replay_summary.md" in names
        assert "reports/bottom_zone_backtest.csv" in names
        assert "reports/backtest_regime_breakdown.csv" in names
        assert "reports/research_promotion_decision.csv" in names
        bottom_csv = archive.read("reports/bottom_zone_reversal_shadow.csv").decode()
        assert "support_zone_low" in bottom_csv
        assert "bottom_zone_score" in bottom_csv
        assert "read_only_no_live_order" in bottom_csv


def test_fast_microstructure_uses_latest_rollup_time_when_market_bar_lags():
    market_ts = datetime(2026, 6, 6, 3, tzinfo=UTC)
    rollup_ts = datetime(2026, 6, 6, 9, tzinfo=UTC)
    market = pl.DataFrame(
        _market_rows("BNB-USDT", market_ts - timedelta(hours=4), [100, 101, 102, 103, 104])
    )
    spreads = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": rollup_ts - timedelta(minutes=minute),
                "ts": rollup_ts - timedelta(minutes=minute),
                "spread_bps": 2.0,
                "orderbook_imbalance": 0.3,
            }
            for minute in range(60)
        ]
    )
    trades = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": rollup_ts - timedelta(minutes=minute),
                "latest_trade_ts": rollup_ts - timedelta(minutes=minute),
                "trade_count": 4,
                "size_sum": 8.0,
                "taker_buy_size_sum": 5.0,
                "taker_sell_size_sum": 3.0,
            }
            for minute in range(60)
        ]
    )

    fast = build_fast_microstructure_features(
        market_bars=market,
        orderbook_spread_1m=spreads,
        trade_activity_1m=trades,
        generated_at=rollup_ts,
    )

    row = fast.filter(pl.col("symbol") == "BNB-USDT").to_dicts()[0]
    assert row["symbol"] == "BNB-USDT"
    assert row["ts_utc"] == "2026-06-06T09:00:00Z"
    assert row["latest_spread_bps"] == 2.0
    assert row["orderbook_imbalance_1m"] == 0.3
    assert row["trade_count_60m"] == 240.0
    assert row["taker_buy_sell_imbalance_5m"] > 0


def test_fast_microstructure_anchors_to_rollup_when_market_bar_is_newer():
    market_ts = datetime(2026, 6, 6, 12, tzinfo=UTC)
    rollup_ts = datetime(2026, 6, 6, 9, tzinfo=UTC)
    market = pl.DataFrame(
        _market_rows(
            "BNB-USDT",
            market_ts - timedelta(hours=6),
            [100, 101, 102, 103, 104, 105, 106],
        )
    )
    spreads = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": rollup_ts - timedelta(minutes=minute),
                "ts": rollup_ts - timedelta(minutes=minute),
                "spread_bps": 2.0,
                "orderbook_imbalance": 0.3,
            }
            for minute in range(60)
        ]
    )
    trades = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": rollup_ts - timedelta(minutes=minute),
                "latest_trade_ts": rollup_ts - timedelta(minutes=minute),
                "trade_count": 4,
                "size_sum": 8.0,
                "taker_buy_size_sum": 5.0,
                "taker_sell_size_sum": 3.0,
            }
            for minute in range(60)
        ]
    )

    fast = build_fast_microstructure_features(
        market_bars=market,
        orderbook_spread_1m=spreads,
        trade_activity_1m=trades,
        generated_at=market_ts,
    )

    row = fast.filter(pl.col("symbol") == "BNB-USDT").to_dicts()[0]
    assert row["ts_utc"] == "2026-06-06T09:00:00Z"
    assert row["avg_spread_bps_5m"] == 2.0
    assert row["trade_count_5m"] == 24.0
    assert row["taker_buy_sell_imbalance_5m"] > 0


def test_fast_microstructure_infers_trade_side_from_price_vs_mid():
    market_ts = datetime(2026, 6, 6, 9, tzinfo=UTC)
    market = pl.DataFrame(
        _market_rows("BNB-USDT", market_ts - timedelta(hours=4), [100, 101, 102, 103, 104])
    )
    spreads = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": market_ts - timedelta(minutes=minute),
                "ts": market_ts - timedelta(minutes=minute),
                "bid": 103.90,
                "ask": 104.10,
                "mid": 104.0,
                "bid_size": 12.0,
                "ask_size": 8.0,
            }
            for minute in range(10)
        ]
    )
    trades = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": market_ts - timedelta(minutes=minute),
                "latest_trade_ts": market_ts - timedelta(minutes=minute),
                "trade_count": 1,
                "size_sum": 5.0,
                "price": 104.08 if minute % 2 == 0 else 103.95,
            }
            for minute in range(10)
        ]
    )

    fast = build_fast_microstructure_features(
        market_bars=market,
        orderbook_spread_1m=spreads,
        trade_activity_1m=trades,
        generated_at=market_ts,
    )

    row = fast.filter(pl.col("symbol") == "BNB-USDT").to_dicts()[0]
    assert row["latest_spread_bps"] == pytest.approx((104.10 - 103.90) / 104.0 * 10000.0)
    assert row["orderbook_imbalance_1m"] == pytest.approx((12.0 - 8.0) / 20.0)
    assert row["side_inferred"] is True
    assert row["taker_buy_size_sum_15m"] > 0
    assert row["taker_sell_size_sum_15m"] > 0
    assert row["taker_buy_sell_imbalance_5m"] is not None


def test_fast_microstructure_keeps_target_symbols_when_rollups_are_sparse():
    latest = datetime(2026, 6, 6, 9, tzinfo=UTC)
    market = pl.DataFrame(
        [
            row
            for symbol in (
                "BTC-USDT",
                "ETH-USDT",
                "SOL-USDT",
                "BNB-USDT",
                "WLD-USDT",
                "HYPE-USDT",
                "XRP-USDT",
                "ZEC-USDT",
            )
            for row in _market_rows(
                symbol,
                latest - timedelta(hours=4),
                [100, 101, 102, 103, 104],
            )
        ]
    )
    spreads = pl.DataFrame(
        [
            {
                "symbol": symbol,
                "minute_ts": latest - timedelta(minutes=minute),
                "ts": latest - timedelta(minutes=minute),
                "spread_bps": 2.0,
                "bid_size": 10.0 + minute,
                "ask_size": 8.0,
            }
            for symbol in ("XRP-USDT", "ZEC-USDT")
            for minute in range(60)
        ]
    )
    trades = pl.DataFrame(
        [
            {
                "symbol": symbol,
                "minute_ts": latest - timedelta(minutes=minute),
                "latest_trade_ts": latest - timedelta(minutes=minute),
                "trade_count": 4,
                "size_sum": 8.0,
                "taker_buy_size_sum": 5.0,
                "taker_sell_size_sum": 3.0,
            }
            for symbol in ("XRP-USDT", "ZEC-USDT")
            for minute in range(60)
        ]
    )

    fast = build_fast_microstructure_features(
        market_bars=market,
        orderbook_spread_1m=spreads,
        trade_activity_1m=trades,
        generated_at=latest,
    )

    assert set(fast["symbol"].to_list()) == {
        "BTC-USDT",
        "ETH-USDT",
        "SOL-USDT",
        "BNB-USDT",
        "WLD-USDT",
        "HYPE-USDT",
        "XRP-USDT",
        "ZEC-USDT",
    }
    assert "bid_depth_recovery" in fast.columns
    assert "spread_normalization" in fast.columns
    assert "missing_reason" in fast.columns
    xrp = fast.filter(pl.col("symbol") == "XRP-USDT").to_dicts()[0]
    assert xrp["taker_buy_sell_imbalance_5m"] > 0
    assert xrp["spread_normalization"] == 0.0
    assert xrp["missing_reason"] == "none"
    btc = fast.filter(pl.col("symbol") == "BTC-USDT").to_dicts()[0]
    assert btc["return_1h_bps"] is not None
    assert btc["latest_spread_bps"] is None
    assert "missing_orderbook_rollup" in btc["missing_reason"]
    assert "missing_trade_rollup" in btc["missing_reason"]


def test_fast_microstructure_outputs_target_symbol_missing_reasons():
    latest = datetime(2026, 6, 6, 9, tzinfo=UTC)
    market = pl.DataFrame(
        _market_rows("BTC-USDT", latest - timedelta(hours=4), [100, 101, 102, 103, 104])
    )

    fast = build_fast_microstructure_features(
        market_bars=market,
        orderbook_spread_1m=pl.DataFrame(),
        trade_activity_1m=pl.DataFrame(),
        generated_at=latest,
    )

    assert set(fast["symbol"].to_list()) == {
        "BTC-USDT",
        "ETH-USDT",
        "SOL-USDT",
        "BNB-USDT",
        "WLD-USDT",
        "HYPE-USDT",
        "XRP-USDT",
        "ZEC-USDT",
    }
    btc = fast.filter(pl.col("symbol") == "BTC-USDT").to_dicts()[0]
    assert "missing_orderbook_rollup" in btc["missing_reason"]
    zec = fast.filter(pl.col("symbol") == "ZEC-USDT").to_dicts()[0]
    assert zec["missing_reason"] == "missing_market_bar"


def test_fast_microstructure_keeps_core_rollup_fields_without_market_bar():
    latest = datetime(2026, 6, 6, 9, tzinfo=UTC)
    spreads = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": latest - timedelta(minutes=minute),
                "ts": latest - timedelta(minutes=minute),
                "spread_bps": 2.5,
                "orderbook_imbalance": 0.25,
            }
            for minute in range(15)
        ]
    )
    trades = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": latest - timedelta(minutes=minute),
                "latest_trade_ts": latest - timedelta(minutes=minute),
                "trade_count": 3,
                "size_sum": 6.0,
                "taker_buy_size_sum": 4.0,
                "taker_sell_size_sum": 2.0,
            }
            for minute in range(15)
        ]
    )

    fast = build_fast_microstructure_features(
        market_bars=pl.DataFrame(
            [
                {
                    "symbol": "RENDER-USDT",
                    "ts": latest,
                    "close": 2.0,
                }
            ]
        ),
        orderbook_spread_1m=spreads,
        trade_activity_1m=trades,
        generated_at=latest,
    )

    assert "RENDER-USDT" not in set(fast["symbol"].to_list())
    bnb = fast.filter(pl.col("symbol") == "BNB-USDT").to_dicts()[0]
    assert bnb["latest_spread_bps"] == 2.5
    assert bnb["avg_spread_bps_5m"] == 2.5
    assert bnb["orderbook_imbalance_1m"] == 0.25
    assert bnb["orderbook_imbalance_5m"] == 0.25
    assert bnb["taker_buy_sell_imbalance_5m"] > 0
    assert bnb["cvd_5m"] > 0
    assert bnb["spread_bps_change_5m"] == 0.0
    assert "missing_market_bar" in bnb["missing_reason"]
