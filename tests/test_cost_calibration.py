import json
from pathlib import Path

import polars as pl

from quant_lab.costs.calibrate import calibrate_costs_for_day
from quant_lab.costs.model import cost_bucket_daily_to_cost_buckets, estimate_cost_bps
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset


def test_actual_fills_and_bills_generate_actual_cost_bucket(tmp_path):
    lake_root = tmp_path / "lake"
    _write_fills(lake_root)
    _write_bills(lake_root)
    _write_orderbooks(lake_root)

    result = calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=1)

    assert result.rows_written == 2
    assert result.health_rows_written == 1
    assert result.sources == ["actual_okx_fills_and_bills"]
    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    assert len(rows) == 2
    row = [item for item in rows if item["notional_bucket"] == "0-1k"][0]
    assert row["day"] == "2026-05-10"
    assert row["symbol"] == "BTC-USDT"
    assert row["regime"] == "realized"
    assert row["event_type"] == "actual_fill"
    assert row["notional_bucket"] == "0-1k"
    assert row["sample_count"] == 1
    assert row["fee_bps_p50"] == 5.0
    assert row["spread_bps_p50"] == 200.0
    assert row["slippage_bps_p50"] == 0.0
    assert row["total_cost_bps_p50"] == 205.0
    assert row["fallback_level"] == "SLIPPAGE_UNKNOWN;SPREAD_PROXY"
    assert row["source"] == "actual_okx_fills_and_bills"
    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    assert health["actual_rows"] == 2


def test_missing_bills_fallback_is_explicit(tmp_path):
    lake_root = tmp_path / "lake"
    _write_fills(lake_root)
    _write_orderbooks(lake_root)

    calibrate_costs_for_day(lake_root, "2026-05-10")

    row = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()[0]
    assert row["source"] == "actual_okx_fills_fee_missing"
    assert "BILLS_MISSING" in row["fallback_level"]
    assert "SLIPPAGE_UNKNOWN" in row["fallback_level"]
    assert row["fee_bps_p50"] == 5.0


def test_sample_too_small_is_explicit_and_not_fully_actual(tmp_path):
    lake_root = tmp_path / "lake"
    _write_fills(lake_root)
    _write_bills(lake_root)
    _write_orderbooks(lake_root)

    calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=30)

    row = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()[0]
    assert row["source"] == "actual_okx_fills_fee_missing"
    assert "SAMPLE_TOO_SMALL" in row["fallback_level"]


def test_without_fills_uses_public_spread_proxy_only(tmp_path):
    lake_root = tmp_path / "lake"
    _write_orderbooks(lake_root)

    calibrate_costs_for_day(lake_root, "2026-05-10")

    row = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()[0]
    assert row["source"] == "public_spread_proxy"
    assert row["event_type"] == "spread_proxy"
    assert row["regime"] == "public_proxy"
    assert row["fee_bps_p50"] == 0.0
    assert row["slippage_bps_p50"] == 0.0
    assert row["spread_bps_p50"] == 200.0
    assert row["total_cost_bps_p50"] == 200.0
    assert row["fallback_level"] == "FEE_MISSING;SLIPPAGE_UNKNOWN;PUBLIC_SPREAD_PROXY"
    assert "actual" not in row["source"]


def test_global_default_when_no_cost_inputs_exist(tmp_path):
    lake_root = tmp_path / "lake"

    calibrate_costs_for_day(lake_root, "2026-05-10")

    row = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()[0]
    assert row["symbol"] == "GLOBAL"
    assert row["source"] == "global_default"
    assert row["fallback_level"] == "GLOBAL_DEFAULT"
    assert row["total_cost_bps_p50"] == 25.0


def test_calibrated_output_can_feed_cost_estimate(tmp_path):
    lake_root = tmp_path / "lake"
    _write_fills(lake_root)
    _write_bills(lake_root)
    _write_orderbooks(lake_root)
    calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=1)

    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    buckets = cost_bucket_daily_to_cost_buckets(rows)
    estimate = estimate_cost_bps("BTC-USDT", "realized", 200, buckets)

    assert estimate.cost_bps == 205.0
    assert estimate.fallback_level == "NONE"
    assert estimate.bucket_id == "2026-05-10:BTC-USDT:realized:actual_fill:0-1k"


def _write_fills(lake_root: Path) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "venue": "okx",
                    "inst_type": "SPOT",
                    "inst_id": "BTC-USDT",
                    "trade_id": "trade-1",
                    "order_id": "order-1",
                    "side": "buy",
                    "fill_price": 100.0,
                    "fill_size": 2.0,
                    "fee": -0.1,
                    "fee_currency": "USDT",
                    "liquidity": "T",
                    "ts": "2026-05-10T00:00:00Z",
                    "source": "okx_readonly_private",
                }
            ]
        ),
        lake_root / "silver" / "fill_event",
    )


def _write_bills(lake_root: Path) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "venue": "okx",
                    "bill_id": "bill-1",
                    "ccy": "USDT",
                    "amount": -0.1,
                    "balance": 999.9,
                    "bill_type": "2",
                    "sub_type": "1",
                    "ts": "2026-05-10T00:00:01Z",
                    "source": "okx_readonly_private",
                }
            ]
        ),
        lake_root / "silver" / "account_bill",
    )


def _write_orderbooks(lake_root: Path) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "venue": "okx",
                    "symbol": "BTC-USDT",
                    "channel": "books5",
                    "ts": "2026-05-10T00:00:00Z",
                    "asks_json": json.dumps([["101", "1"]]),
                    "bids_json": json.dumps([["99", "1"]]),
                    "checksum": 42,
                    "source": "okx_public_ws",
                    "ingest_ts": "2026-05-10T00:00:00Z",
                    "raw_json": "{}",
                }
            ]
        ),
        lake_root / "silver" / "orderbook_snapshot",
    )
