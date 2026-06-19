import json
from pathlib import Path

import polars as pl

from quant_lab.costs.calibrate import (
    _read_day_dataset,
    _v5_order_lifecycle_fill_samples,
    calibrate_costs_for_day,
)
from quant_lab.costs.model import (
    cost_bucket_daily_to_cost_buckets,
    estimate_cost_bps,
    estimate_cost_from_cost_bucket_daily_rows,
)
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.ingest.okx_readonly_private import (
    BRONZE_BILLS_DATASET,
    BRONZE_FILLS_DATASET,
)


def test_actual_fills_and_bills_generate_actual_cost_bucket(tmp_path):
    lake_root = tmp_path / "lake"
    _write_fills(lake_root)
    _write_bills(lake_root)
    _write_orderbooks(lake_root)

    result = calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=1)

    assert result.rows_written == 2
    assert result.health_rows_written == 1
    assert result.sources == ["mixed_actual_proxy"]
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
    assert row["fallback_level"] == "SLIPPAGE_UNKNOWN"
    assert row["spread_source"] == "fresh_public_orderbook_p75"
    assert row["source"] == "mixed_actual_proxy"
    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    assert health["actual_rows"] == 0
    assert health["mixed_rows"] == 2
    assert "BTC-USDT" in json.loads(health["symbols_with_mixed_cost"])
    cost_meta = json.loads(
        (lake_root / "gold" / "cost_bucket_daily" / "_snapshot_meta.json").read_text(
            encoding="utf-8"
        )
    )
    health_meta = json.loads(
        (lake_root / "gold" / "cost_health_daily" / "_snapshot_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert cost_meta["dataset"] == "cost_bucket_daily"
    assert cost_meta["row_count"] == 2
    assert cost_meta["source_sha"]
    assert health_meta["dataset"] == "cost_health_daily"
    assert health_meta["row_count"] == 1
    assert health_meta["source_sha"]


def test_missing_bills_fallback_is_explicit(tmp_path):
    lake_root = tmp_path / "lake"
    _write_fills(lake_root)
    _write_orderbooks(lake_root)

    calibrate_costs_for_day(lake_root, "2026-05-10")

    row = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()[0]
    assert row["source"] == "mixed_actual_proxy"
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
    assert row["source"] == "mixed_actual_proxy"
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
    assert row["spread_source"] == "fresh_public_orderbook_p75"
    assert "actual" not in row["source"]


def test_cost_calibration_writes_api_usage_counts_to_cost_health(tmp_path):
    lake_root = tmp_path / "lake"
    _write_orderbooks(lake_root, symbol="BNB-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "bundle_ts": "2026-05-10T01:00:00Z",
                    "symbol": "UNKNOWN-USDT",
                    "cost_source": "global_default",
                    "fallback_level": "GLOBAL_DEFAULT",
                    "degraded_cost_model": "true",
                },
                {
                    "bundle_ts": "2026-05-10T01:01:00Z",
                    "symbol": "BNB-USDT",
                    "cost_source": "public_spread_proxy",
                    "fallback_level": "NONE",
                    "degraded_cost_model": "false",
                },
                {
                    "bundle_ts": "2026-05-10T01:02:00Z",
                    "raw_payload_json": (
                        '{"response": {"cost_source": "public_spread_proxy", '
                        '"fallback_level": "REGIME_FALLBACK", '
                        '"degraded_cost_model": true}}'
                    ),
                },
            ]
        ),
        lake_root / "silver" / "v5_quant_lab_cost_usage",
    )

    calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=1)

    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    assert health["api_cost_usage_rows"] == 3
    assert health["api_global_default_count"] == 1
    assert health["api_symbol_proxy_hit_count"] == 2
    assert health["api_regime_fallback_count"] == 1
    assert health["api_degraded_cost_count"] == 2


def test_read_day_dataset_uses_day_named_files_for_hot_append_dataset(tmp_path):
    lake_root = tmp_path / "lake"
    dataset = lake_root / "silver" / "orderbook_snapshot"
    dataset.mkdir(parents=True)
    _orderbook_frame("BTC-USDT", "2026-05-10").write_parquet(
        dataset / "batch_20260510T010000Z.parquet"
    )
    _orderbook_frame("ETH-USDT", "2026-05-11").write_parquet(
        dataset / "batch_20260511T010000Z.parquet"
    )

    frame = _read_day_dataset(
        lake_root,
        Path("silver") / "orderbook_snapshot",
        "2026-05-10",
        max_files=10,
    )

    assert frame.height == 1
    assert frame["symbol"].to_list() == ["BTC-USDT"]


def test_read_day_dataset_does_not_full_scan_hot_dataset_without_day_files(tmp_path):
    lake_root = tmp_path / "lake"
    dataset = lake_root / "silver" / "orderbook_snapshot"
    dataset.mkdir(parents=True)
    for index in range(3):
        _orderbook_frame("BTC-USDT", "2026-05-10").write_parquet(
            dataset / f"data_{index}.parquet"
        )

    frame = _read_day_dataset(
        lake_root,
        Path("silver") / "orderbook_snapshot",
        "2026-05-10",
        max_files=2,
    )

    assert frame.is_empty()


def test_read_day_dataset_uses_day_column_and_projects_hot_dataset_columns(tmp_path):
    lake_root = tmp_path / "lake"
    dataset = lake_root / "silver" / "orderbook_snapshot"
    dataset.mkdir(parents=True)
    for index, day in enumerate(["2026-05-09", "2026-05-10", "2026-05-11"]):
        _orderbook_frame("BTC-USDT", day).with_columns(
            pl.lit(day).alias("day"),
            pl.lit("large-raw-payload").alias("raw_json"),
            pl.lit("ignored").alias("source"),
        ).write_parquet(dataset / f"compact_{index}.parquet")

    frame = _read_day_dataset(
        lake_root,
        Path("silver") / "orderbook_snapshot",
        "2026-05-10",
        max_files=10,
        columns=["symbol", "day", "ts", "asks_json", "bids_json"],
    )

    assert frame.height == 1
    assert frame["day"].to_list() == ["2026-05-10"]
    assert frame["symbol"].to_list() == ["BTC-USDT"]
    assert "raw_json" not in frame.columns
    assert "source" not in frame.columns


def test_read_day_dataset_caps_hot_dataset_rows_per_symbol(tmp_path):
    lake_root = tmp_path / "lake"
    dataset = lake_root / "silver" / "orderbook_snapshot"
    dataset.mkdir(parents=True)
    rows = []
    for symbol in ["BTC-USDT", "ETH-USDT"]:
        for minute in range(3):
            rows.append(
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "day": "2026-05-10",
                    "channel": "books5",
                    "ts": f"2026-05-10T00:0{minute}:00Z",
                    "asks_json": json.dumps([["101", "1"]]),
                    "bids_json": json.dumps([["99", "1"]]),
                    "raw_json": "large-raw-payload",
                }
            )
    pl.DataFrame(rows).write_parquet(dataset / "compact_0.parquet")

    frame = _read_day_dataset(
        lake_root,
        Path("silver") / "orderbook_snapshot",
        "2026-05-10",
        max_files=10,
        columns=["symbol", "day", "ts", "asks_json", "bids_json"],
        max_rows_per_symbol=2,
    )

    assert frame.height == 4
    assert frame.group_by("symbol").len()["len"].to_list() == [2, 2]
    assert "raw_json" not in frame.columns


def test_v5_trade_events_generate_actual_fill_bucket_before_spread_proxy(tmp_path):
    lake_root = tmp_path / "lake"
    _write_v5_trades(lake_root)
    _write_orderbooks(lake_root, symbol="BNB-USDT")

    result = calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=1)

    assert result.sources == ["mixed_actual_proxy"]
    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    all_row = [
        row for row in rows if row["symbol"] == "BNB-USDT" and row["notional_bucket"] == "all"
    ][0]
    assert all_row["source"] == "mixed_actual_proxy"
    assert all_row["event_type"] == "actual_fill"
    assert all_row["sample_count"] == 2
    assert all_row["fee_bps_p50"] > 0
    assert "SPREAD_PROXY" not in all_row["fallback_level"]
    assert all_row["spread_source"] == "fresh_public_orderbook_p75"
    assert "SLIPPAGE_UNKNOWN" in all_row["fallback_level"]

    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    assert health["actual_rows"] == 0
    assert health["mixed_rows"] == len(rows)


def test_v5_order_lifecycle_generates_actual_fills_bucket(tmp_path):
    lake_root = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": (
                        "raw/recent_runs/run_lifecycle/order_lifecycle.csv"
                    ),
                    "run_id": "run_lifecycle",
                    "ts_utc": "2026-05-15T01:00:04Z",
                    "symbol": "BNB-USDT",
                    "normalized_symbol": "BNB-USDT",
                    "side": "buy",
                    "intent": "OPEN_LONG",
                    "signal_price": "600",
                    "arrival_mid": "600",
                    "spread_cost_bps": "16.6666666667",
                    "arrival_slippage_bps": "33.3333333333",
                    "delay_cost_bps": "0",
                    "avg_fill_px": "602",
                    "filled_qty": "0.2",
                    "fee": "-0.1204",
                    "fee_ccy": "USDT",
                    "fee_usdt": "0.1204",
                    "notional_usdt": "120.4",
                    "exchange_order_id": "okx-1",
                    "trade_ids": "trade-1",
                    "last_fill_ts": "2026-05-15T01:00:04Z",
                }
            ]
        ),
        lake_root / "silver" / "v5_order_lifecycle",
    )

    result = calibrate_costs_for_day(lake_root, "2026-05-15")

    assert result.sources == ["actual_fills"]
    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    all_row = [
        row for row in rows if row["symbol"] == "BNB-USDT" and row["notional_bucket"] == "all"
    ][0]
    assert all_row["source"] == "actual_fills"
    assert all_row["actual_fill_count"] == 1
    assert all_row["mixed_fill_count"] == 0
    assert all_row["fee_bps_p50"] > 0
    assert all_row["slippage_bps_p50"] > 0
    assert all_row["spread_bps_p50"] > 0
    assert all_row["spread_source"] == "actual_arrival_book"
    assert "SAMPLE_TOO_SMALL" in all_row["fallback_level"]
    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    checks = json.loads(health["data_quality_checks_json"])
    assert health["actual_rows"] == len(rows)
    assert health["mixed_rows"] == 0
    assert checks["lifecycle_present_but_not_in_actual_cost"] is True
    assert checks["filled_order_missing_lifecycle_cost"] is True
    assert checks["fill_count_zero_for_filled_order"] is True


def test_actual_lifecycle_cost_calibration_reaches_canary_with_arrival_spread(tmp_path):
    lake_root = tmp_path / "lake"
    _write_lifecycle_cost_samples(lake_root, day="2026-05-15", sample_count=30)
    _write_bills_for_day(lake_root, "2026-05-15")

    calibrate_costs_for_day(lake_root, "2026-05-15", min_sample_count=30)

    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    row = _cost_bucket_row(rows, symbol="BTC-USDT", notional_bucket="all")
    assert row["source"] == "actual_fills"
    assert row["fallback_level"] == "NONE"
    assert row["spread_source"] == "actual_arrival_book"
    assert row["sample_count"] == 30

    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="realized",
        notional_usdt=120.0,
        quantile="p75",
        rows=rows,
    )

    assert estimate.cost_trust_level == "CANARY"
    assert estimate.cost_trusted_for_live_canary is True
    assert estimate.cost_trusted_for_live_scale is False
    assert estimate.spread_source == "actual_arrival_book"


def test_actual_lifecycle_cost_calibration_reaches_scale_ready_with_complete_samples(tmp_path):
    lake_root = tmp_path / "lake"
    _write_lifecycle_cost_samples(lake_root, day="2026-05-15", sample_count=100)
    _write_bills_for_day(lake_root, "2026-05-15")

    calibrate_costs_for_day(lake_root, "2026-05-15", min_sample_count=30)

    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    row = _cost_bucket_row(rows, symbol="BTC-USDT", notional_bucket="all")
    assert row["source"] == "actual_fills"
    assert row["fallback_level"] == "NONE"
    assert row["spread_source"] == "actual_arrival_book"
    assert row["sample_count"] == 100

    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="realized",
        notional_usdt=120.0,
        quantile="p75",
        rows=rows,
    )

    assert estimate.cost_trust_level == "SCALE_READY"
    assert estimate.cost_trusted_for_live_canary is True
    assert estimate.cost_trusted_for_live_scale is True
    assert estimate.spread_source == "actual_arrival_book"


def test_cost_probe_lifecycle_requires_explicit_cost_model_eligibility():
    rows = pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "order_state": "FILLED",
                "fill_count": 1,
                "avg_fill_px": "70000",
                "filled_qty": "0.0001",
                "notional_usdt": "7",
                "fee_bps": "1.0",
                "arrival_slippage_bps": "0.5",
                "arrival_spread_bps": "0.2",
                "execution_purpose": "cost_probe",
                "eligible_for_alpha_pnl": "false",
                "last_fill_ts": "2026-05-15T01:00:04Z",
            },
            {
                "symbol": "ETH-USDT",
                "order_state": "FILLED",
                "fill_count": 1,
                "avg_fill_px": "3500",
                "filled_qty": "0.002",
                "notional_usdt": "7",
                "fee_bps": "1.0",
                "arrival_slippage_bps": "0.5",
                "arrival_spread_bps": "0.2",
                "execution_purpose": "cost_probe",
                "eligible_for_cost_model": "true",
                "eligible_for_alpha_pnl": "false",
                "last_fill_ts": "2026-05-15T01:00:05Z",
            },
        ]
    )

    samples = _v5_order_lifecycle_fill_samples(rows)

    assert [sample["symbol"] for sample in samples] == ["ETH-USDT"]
    assert samples[0]["sample_origin"] == "cost_probe"
    assert samples[0]["eligible_for_cost_model"] is True
    assert samples[0]["eligible_for_alpha_pnl"] is False


def test_cost_probe_lifecycle_calibrates_as_bootstrap_not_actual(tmp_path):
    lake_root = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": (
                        "raw/recent_runs/run_lifecycle/order_lifecycle.csv"
                    ),
                    "run_id": "run_lifecycle",
                    "ts_utc": "2026-05-15T01:00:04Z",
                    "symbol": "BTC-USDT",
                    "normalized_symbol": "BTC-USDT",
                    "side": "buy",
                    "intent": "OPEN_LONG",
                    "order_state": "FILLED",
                    "avg_fill_px": "70000",
                    "filled_qty": "0.0001",
                    "notional_usdt": "7",
                    "fee_bps": "1.0",
                    "arrival_slippage_bps": "0.5",
                    "arrival_spread_bps": "0.2",
                    "execution_purpose": "cost_probe",
                    "eligible_for_cost_model": "true",
                    "eligible_for_alpha_pnl": "false",
                    "fill_count": "1",
                    "exchange_order_id": "probe-order-1",
                    "trade_ids": "probe-trade-1",
                    "last_fill_ts": "2026-05-15T01:00:04Z",
                }
            ]
        ),
        lake_root / "silver" / "v5_order_lifecycle",
    )

    result = calibrate_costs_for_day(lake_root, "2026-05-15", min_sample_count=1)

    assert result.sources == ["bootstrap_cost_probe"]
    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    all_row = [
        row for row in rows if row["symbol"] == "BTC-USDT" and row["notional_bucket"] == "all"
    ][0]
    assert all_row["source"] == "bootstrap_cost_probe"
    assert all_row["actual_fill_count"] == 0
    assert all_row["mixed_fill_count"] == 0
    assert all_row["cost_probe_fill_count"] == 1
    assert all_row["strategy_live_fill_count"] == 0
    assert all_row["private_fill_count"] == 0
    assert all_row["sample_origin_mix"] == "cost_probe_only"
    assert all_row["eligible_for_live_cost_coverage"] is False
    assert "COST_PROBE_ONLY" in all_row["fallback_level"]
    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    checks = json.loads(health["data_quality_checks_json"])
    assert health["actual_rows"] == 0
    assert health["mixed_rows"] == 0
    assert health["status"] == "WARNING"
    assert checks["lifecycle_present_but_not_in_actual_cost"] is True


def test_v5_order_lifecycle_stays_actual_when_trade_csv_also_exists(tmp_path):
    lake_root = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": (
                        "raw/recent_runs/run_lifecycle/order_lifecycle.csv"
                    ),
                    "run_id": "run_lifecycle",
                    "ts_utc": "2026-05-15T01:00:04Z",
                    "symbol": "BTC-USDT",
                    "normalized_symbol": "BTC-USDT",
                    "side": "buy",
                    "intent": "OPEN_LONG",
                    "arrival_mid": "60000",
                    "spread_bps_at_decision": "2.5",
                    "arrival_slippage_bps": "-1.6666666667",
                    "delay_cost_bps": "0.5",
                    "avg_fill_px": "59990",
                    "filled_qty": "0.01",
                    "fee_usdt": "0.29995",
                    "notional_usdt": "599.9",
                    "fill_count": "1",
                    "exchange_order_id": "btc-order-1",
                    "trade_ids": "btc-trade-1",
                    "last_fill_ts": "2026-05-15T01:00:04Z",
                }
            ]
        ),
        lake_root / "silver" / "v5_order_lifecycle",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": "raw/recent_runs/run_lifecycle/trades.csv",
                    "run_id": "run_lifecycle",
                    "ts_utc": "2026-05-15T01:00:04Z",
                    "symbol": "BTC-USDT",
                    "side": "buy",
                    "qty": "0.01",
                    "price": "59990",
                    "fee": "-0.29995",
                    "fee_ccy": "USDT",
                    "order_id": "btc-order-1",
                    "trade_id": "btc-trade-1",
                }
            ]
        ),
        lake_root / "silver" / "v5_trade_event",
    )

    result = calibrate_costs_for_day(lake_root, "2026-05-15")

    assert result.sources == ["actual_fills"]
    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    all_row = [
        row for row in rows if row["symbol"] == "BTC-USDT" and row["notional_bucket"] == "all"
    ][0]
    assert all_row["source"] == "actual_fills"
    assert all_row["sample_count"] == 1
    assert all_row["actual_fill_count"] == 1
    assert all_row["mixed_fill_count"] == 0
    assert all_row["fee_bps_p50"] > 0
    assert all_row["slippage_bps_p50"] == 0.0
    assert all_row["spread_source"] == "actual_arrival_book"

    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    checks = json.loads(health["data_quality_checks_json"])
    assert health["actual_rows"] == len(rows)
    assert health["mixed_rows"] == 0
    assert checks["lifecycle_present_but_not_in_actual_cost"] is True
    assert checks["filled_order_missing_lifecycle_cost"] is True


def test_v5_order_lifecycle_fill_ts_rows_enter_btc_actual_cost_bucket(tmp_path):
    lake_root = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": (
                        "raw/recent_runs/run_lifecycle/order_lifecycle.csv"
                    ),
                    "run_id": "run_lifecycle",
                    "fill_ts": "2026-05-15T02:00:04Z",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "intent": "OPEN_LONG",
                    "order_state": "FILLED",
                    "signal_price": "60000",
                    "arrival_mid": "60005",
                    "spread": "2.5",
                    "fill_px": "60020",
                    "fill_qty": "0.01",
                    "fee_usdt": "0.3001",
                    "notional_usdt": "600.2",
                    "fill_count": "1",
                    "exchange_order_id": "btc-open-1",
                    "trade_id": "btc-trade-open-1",
                },
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": (
                        "raw/recent_runs/run_lifecycle/order_lifecycle.csv"
                    ),
                    "run_id": "run_lifecycle",
                    "fill_ts": "2026-05-15T03:00:04Z",
                    "symbol": "BTC-USDT",
                    "side": "sell",
                    "intent": "CLOSE_LONG",
                    "order_state": "FILLED",
                    "signal_price": "60100",
                    "arrival_mid": "60095",
                    "spread": "2.0",
                    "fill_px": "60085",
                    "fill_qty": "0.01",
                    "fee_usdt": "0.300425",
                    "notional_usdt": "600.85",
                    "fill_count": "1",
                    "exchange_order_id": "btc-close-1",
                    "trade_id": "btc-trade-close-1",
                },
            ]
        ),
        lake_root / "silver" / "v5_order_lifecycle",
    )

    result = calibrate_costs_for_day(lake_root, "2026-05-15", min_sample_count=2)

    assert result.sources == ["actual_fills"]
    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    all_row = [
        row for row in rows if row["symbol"] == "BTC-USDT" and row["notional_bucket"] == "all"
    ][0]
    assert all_row["source"] == "actual_fills"
    assert all_row["actual_fill_count"] == 2
    assert all_row["mixed_fill_count"] == 0
    assert all_row["sample_count"] == 2
    assert all_row["fee_bps_p50"] > 0
    assert all_row["slippage_bps_p50"] > 0
    assert all_row["spread_bps_p50"] == 2.25
    assert all_row["spread_source"] == "actual_arrival_book"
    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    checks = json.loads(health["data_quality_checks_json"])
    assert health["actual_rows"] == len(rows)
    assert checks["lifecycle_present_but_not_in_actual_cost"] is True
    assert checks["filled_order_missing_lifecycle_cost"] is True


def test_v5_order_lifecycle_zero_fill_is_not_used_as_actual_cost(tmp_path):
    lake_root = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": (
                        "raw/recent_runs/run_lifecycle/order_lifecycle.csv"
                    ),
                    "run_id": "run_lifecycle",
                    "ts_utc": "2026-05-15T01:00:04Z",
                    "symbol": "BTC-USDT",
                    "normalized_symbol": "BTC-USDT",
                    "side": "buy",
                    "intent": "OPEN_LONG",
                    "order_state": "FILLED",
                    "signal_price": "60000",
                    "arrival_mid": "60000",
                    "spread_bps_at_decision": "2.5",
                    "avg_fill_px": "60010",
                    "filled_qty": "0",
                    "fee_usdt": "0",
                    "notional_usdt": "0",
                    "requested_notional_usdt": "120",
                    "fill_count": "0",
                }
            ]
        ),
        lake_root / "silver" / "v5_order_lifecycle",
    )

    result = calibrate_costs_for_day(lake_root, "2026-05-15", min_sample_count=1)

    assert result.sources == ["global_default"]
    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    assert rows[0]["source"] == "global_default"
    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    checks = json.loads(health["data_quality_checks_json"])
    assert health["status"] == "CRITICAL"
    assert checks["lifecycle_present_but_not_in_actual_cost"] is False
    assert checks["filled_order_missing_lifecycle_cost"] is False
    assert checks["fill_count_zero_for_filled_order"] is False
    warnings = json.loads(health["warnings_json"])
    assert "lifecycle_present_but_not_in_actual_cost" in warnings
    assert "filled_order_missing_lifecycle_cost" in warnings
    assert "fill_count_zero_for_filled_order" in warnings


def test_recent_v5_trades_feed_later_day_mixed_actual_cost(tmp_path):
    lake_root = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": "raw/recent_runs/run_20260512_06/trades.csv",
                    "run_id": "run_20260512_06",
                    "ts_utc": "2026-05-12T06:01:00Z",
                    "symbol": "BNB/USDT",
                    "normalized_symbol": "BNB-USDT",
                    "side": "buy",
                    "action": "entry",
                    "qty": "0.5",
                    "price": "620",
                    "fee": "-0.031",
                    "fee_ccy": "USDT",
                    "order_id": "bnb-order-buy",
                    "trade_id": "bnb-trade-buy",
                },
                {
                    "strategy": "v5",
                    "source_path_inside_bundle": "raw/recent_runs/run_20260512_11/trades.csv",
                    "run_id": "run_20260512_11",
                    "ts_utc": "2026-05-12T11:01:00Z",
                    "symbol": "BNB-USDT",
                    "side": "sell",
                    "qty": "0.5",
                    "price": "622",
                    "fee": "-0.0311",
                    "fee_ccy": "USDT",
                    "order_id": "bnb-order-sell",
                    "trade_id": "bnb-trade-sell",
                },
            ]
        ),
        lake_root / "silver" / "v5_trade_event",
    )
    _write_orderbooks_for_day(lake_root, symbol="BNB-USDT", day="2026-05-14")

    result = calibrate_costs_for_day(lake_root, "2026-05-14", min_sample_count=1)

    assert result.sources == ["mixed_actual_proxy"]
    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    all_row = [
        row for row in rows if row["symbol"] == "BNB-USDT" and row["notional_bucket"] == "all"
    ][0]
    assert all_row["source"] == "mixed_actual_proxy"
    assert all_row["cost_source"] == "mixed_actual_proxy"
    assert all_row["sample_count"] == 2
    assert all_row["mixed_fill_count"] == 2
    assert all_row["proxy_sample_count"] == 1
    assert all_row["fee_bps_p50"] > 0
    assert "PRIVATE_FILL_LOOKBACK" in all_row["fallback_level"]

    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    checks = json.loads(health["data_quality_checks_json"])
    assert health["actual_rows"] == 0
    assert health["mixed_rows"] == len(rows)
    assert json.loads(health["symbols_with_mixed_cost"]) == ["BNB-USDT"]
    assert json.loads(health["actual_sample_count_by_symbol"]) == {"BNB-USDT": 2}
    assert checks["trades_present_but_not_in_cost_model"] is True
    assert checks["fee_missing_rate"] == "0/2"


def test_okx_private_bronze_fills_and_bills_feed_mixed_actual_cost(tmp_path):
    lake_root = tmp_path / "lake"
    _write_bronze_okx_private_fills(
        lake_root,
        [
            _raw_bnb_fill("fill-1", "order-1", "300", "10", "-0.001", fee_ccy="BNB"),
            _raw_bnb_fill("fill-2", "order-2", "310", "5", "-0.155"),
        ],
    )
    _write_bronze_okx_private_bills(lake_root, [_raw_bnb_bill()])
    _write_orderbooks(lake_root, symbol="BNB-USDT")

    result = calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=1)

    assert result.sources == ["mixed_actual_proxy"]
    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    all_row = [
        row for row in rows if row["symbol"] == "BNB-USDT" and row["notional_bucket"] == "all"
    ][0]
    assert all_row["source"] == "mixed_actual_proxy"
    assert all_row["sample_count"] == 2
    assert all_row["fee_bps_p50"] > 0
    assert "SLIPPAGE_UNKNOWN" in all_row["fallback_level"]

    health = read_parquet_dataset(lake_root / "gold" / "cost_health_daily").to_dicts()[0]
    assert health["actual_rows"] == 0
    assert health["mixed_rows"] == len(rows)
    assert json.loads(health["symbols_with_mixed_cost"]) == ["BNB-USDT"]
    assert json.loads(health["actual_sample_count_by_symbol"]) == {"BNB-USDT": 2}


def test_actual_fills_do_not_hide_proxy_rows_for_other_symbols(tmp_path):
    lake_root = tmp_path / "lake"
    _write_fills(lake_root)
    _write_bills(lake_root)
    _write_orderbooks_multi(lake_root, ["BTC-USDT", "ETH-USDT"])

    calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=1)

    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    sources_by_symbol = {
        row["symbol"]: row["source"] for row in rows if row["notional_bucket"] == "all"
    }
    assert sources_by_symbol["BTC-USDT"] == "mixed_actual_proxy"
    assert sources_by_symbol["ETH-USDT"] == "public_spread_proxy"


def test_recalibration_replaces_same_day_obsolete_proxy_rows(tmp_path):
    lake_root = tmp_path / "lake"
    _write_orderbooks(lake_root)
    calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=1)

    _write_fills(lake_root)
    _write_bills(lake_root)
    _write_orderbooks(lake_root)
    calibrate_costs_for_day(lake_root, "2026-05-10", min_sample_count=1)

    rows = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
    btc_rows = [row for row in rows if row["symbol"] == "BTC-USDT"]
    assert {row["source"] for row in btc_rows} == {"mixed_actual_proxy"}


def test_recent_private_fills_can_calibrate_later_cost_day_with_explicit_lookback(tmp_path):
    lake_root = tmp_path / "lake"
    _write_fills(lake_root)
    _write_bills(lake_root)
    _write_orderbooks_for_day(lake_root, symbol="BTC-USDT", day="2026-05-14")

    calibrate_costs_for_day(lake_root, "2026-05-14", min_sample_count=1)

    row = [
        item
        for item in read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").to_dicts()
        if item["symbol"] == "BTC-USDT" and item["notional_bucket"] == "all"
    ][0]
    assert row["source"] == "mixed_actual_proxy"
    assert "PRIVATE_FILL_LOOKBACK" in row["fallback_level"]


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


def _cost_bucket_row(rows: list[dict], *, symbol: str, notional_bucket: str) -> dict:
    return [
        row
        for row in rows
        if row["symbol"] == symbol and row["notional_bucket"] == notional_bucket
    ][0]


def _write_lifecycle_cost_samples(lake_root: Path, *, day: str, sample_count: int) -> None:
    rows = []
    for index in range(sample_count):
        second = index % 60
        rows.append(
            {
                "strategy": "v5",
                "source_path_inside_bundle": "raw/recent_runs/run_lifecycle/order_lifecycle.csv",
                "run_id": "run_lifecycle",
                "ts_utc": f"{day}T01:00:{second:02d}Z",
                "symbol": "BTC-USDT",
                "normalized_symbol": "BTC-USDT",
                "side": "buy" if index % 2 == 0 else "sell",
                "intent": "OPEN_LONG" if index % 2 == 0 else "CLOSE_LONG",
                "order_state": "FILLED",
                "arrival_mid": "60000",
                "arrival_bid": "59994",
                "arrival_ask": "60006",
                "arrival_slippage_bps": "1.5",
                "delay_cost_bps": "0",
                "avg_fill_px": "60009" if index % 2 == 0 else "59991",
                "filled_qty": "0.002",
                "fee_bps": "1.0",
                "fee_usdt": "0.012",
                "notional_usdt": "120",
                "fill_count": "1",
                "execution_purpose": "strategy_live",
                "exchange_order_id": f"okx-{index}",
                "trade_ids": f"trade-{index}",
                "last_fill_ts": f"{day}T01:00:{second:02d}Z",
            }
        )
    write_parquet_dataset(
        pl.DataFrame(rows),
        lake_root / "silver" / "v5_order_lifecycle",
    )


def _write_bills_for_day(lake_root: Path, day: str) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "venue": "okx",
                    "bill_id": f"bill-{day}",
                    "ccy": "USDT",
                    "amount": -0.012,
                    "balance": 999.9,
                    "bill_type": "2",
                    "sub_type": "1",
                    "ts": f"{day}T01:00:00Z",
                    "source": "okx_readonly_private",
                }
            ]
        ),
        lake_root / "silver" / "account_bill",
    )


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


def _write_orderbooks(lake_root: Path, symbol: str = "BTC-USDT") -> None:
    _write_orderbooks_for_day(lake_root, symbol=symbol, day="2026-05-10")


def _write_orderbooks_for_day(lake_root: Path, symbol: str, day: str) -> None:
    write_parquet_dataset(
        _orderbook_frame(symbol, day),
        lake_root / "silver" / "orderbook_snapshot",
    )


def _orderbook_frame(symbol: str, day: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "venue": "okx",
                "symbol": symbol,
                "channel": "books5",
                "ts": f"{day}T00:00:00Z",
                "asks_json": json.dumps([["101", "1"]]),
                "bids_json": json.dumps([["99", "1"]]),
                "checksum": 42,
                "source": "okx_public_ws",
                "ingest_ts": f"{day}T00:00:00Z",
                "raw_json": "{}",
            }
        ]
    )


def _write_orderbooks_multi(lake_root: Path, symbols: list[str]) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "channel": "books5",
                    "ts": "2026-05-10T00:00:00Z",
                    "asks_json": json.dumps([["101", "1"]]),
                    "bids_json": json.dumps([["99", "1"]]),
                    "checksum": 42,
                    "source": "okx_public_ws",
                    "ingest_ts": "2026-05-10T00:00:00Z",
                    "raw_json": "{}",
                }
                for symbol in symbols
            ]
        ),
        lake_root / "silver" / "orderbook_snapshot",
    )


def _write_v5_trades(lake_root: Path) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "sha",
                    "bundle_name": "fixture.tar.gz",
                    "source_path_inside_bundle": "raw/recent_runs/run_1/trades.csv",
                    "run_id": "run_1",
                    "row_index": 0,
                    "symbol": "BNB/USDT",
                    "side": "buy",
                    "qty": "10",
                    "price": "300",
                    "fee": "0.3",
                    "notional": "3000",
                    "ts": "2026-05-10T00:00:00Z",
                    "ingest_ts": "2026-05-10T00:01:00Z",
                },
                {
                    "strategy": "v5",
                    "bundle_sha256": "sha",
                    "bundle_name": "fixture.tar.gz",
                    "source_path_inside_bundle": "raw/recent_runs/run_1/trades.csv",
                    "run_id": "run_1",
                    "row_index": 1,
                    "symbol": "OKX:BNB-USDT",
                    "side": "sell",
                    "qty": "5",
                    "price": "310",
                    "fee": "0.155",
                    "notional": "1550",
                    "ts": "2026-05-10T00:05:00Z",
                    "ingest_ts": "2026-05-10T00:06:00Z",
                },
            ]
        ),
        lake_root / "silver" / "v5_trade_event",
    )


def _write_bronze_okx_private_fills(lake_root: Path, raw_items: list[dict[str, str]]) -> None:
    _write_bronze_okx_private(lake_root / BRONZE_FILLS_DATASET, raw_items)


def _write_bronze_okx_private_bills(lake_root: Path, raw_items: list[dict[str, str]]) -> None:
    _write_bronze_okx_private(lake_root / BRONZE_BILLS_DATASET, raw_items)


def _write_bronze_okx_private(dataset_path: Path, raw_items: list[dict[str, str]]) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "endpoint": "/api/v5/read-only-fixture",
                    "ingest_ts": "2026-05-10T00:01:00Z",
                    "raw_json": json.dumps(raw_item, sort_keys=True),
                }
                for raw_item in raw_items
            ]
        ),
        dataset_path,
    )


def _raw_bnb_fill(
    trade_id: str,
    order_id: str,
    fill_px: str,
    fill_sz: str,
    fee: str,
    fee_ccy: str = "USDT",
) -> dict[str, str]:
    return {
        "instType": "SPOT",
        "instId": "BNB-USDT",
        "tradeId": trade_id,
        "ordId": order_id,
        "side": "buy",
        "fillPx": fill_px,
        "fillSz": fill_sz,
        "fee": fee,
        "feeCcy": fee_ccy,
        "execType": "T",
        "ts": "1778371200000",
    }


def _raw_bnb_bill() -> dict[str, str]:
    return {
        "billId": "bill-bnb-1",
        "ccy": "USDT",
        "balChg": "-0.455",
        "bal": "999.545",
        "type": "2",
        "subType": "1",
        "ts": "1778371201000",
    }
