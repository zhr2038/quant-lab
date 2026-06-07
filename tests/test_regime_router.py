from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.regime_router import build_and_publish_regime_router


def test_regime_router_allows_alt_impulse_only_in_valid_regime(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(
        lake,
        {
            "BTC-USDT": (100.0, 101.0),
            "ETH-USDT": (100.0, 103.0),
            "SOL-USDT": (100.0, 104.0),
            "BNB-USDT": (100.0, 102.8),
        },
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-22",
                    "strategy_candidate": "v5.alt_impulse_shadow",
                    "candidate_name": "v5.alt_impulse_shadow",
                    "symbol": "SOL-USDT",
                    "regime_state": "impulse",
                    "horizon_hours": 24,
                    "sample_count": 16,
                    "complete_sample_count": 16,
                    "avg_net_bps": 88.0,
                    "median_net_bps": 60.0,
                    "p25_net_bps": -20.0,
                    "win_rate": 0.70,
                    "cost_source_mix": '{"mixed_actual_proxy":16}',
                    "decision": "PAPER_READY",
                    "decision_reasons": '["paper_ready_thresholds_met"]',
                    "created_at": datetime(2026, 5, 22, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "strategy_evidence",
    )

    result = build_and_publish_regime_router(lake, as_of_date="2026-05-22")

    assert result.market_regime_rows == 1
    regime = read_parquet_dataset(lake / "gold" / "market_regime_daily").to_dicts()[0]
    assert regime["current_regime"] == "ALT_IMPULSE"
    matrix = read_parquet_dataset(lake / "gold" / "strategy_regime_matrix").to_dicts()
    assert matrix[0]["strategy_candidate"] == "ALT_IMPULSE_REGIME_SHADOW_V1"
    assert matrix[0]["decision"] == "PAPER_READY"
    advisory = read_parquet_dataset(lake / "gold" / "regime_strategy_advisory").to_dicts()[0]
    assert "ALT_IMPULSE_REGIME_SHADOW_V1" in json.loads(
        advisory["allowed_strategy_candidates"]
    )
    assert advisory["recommended_mode"] == "paper"
    assert "no_live_small_from_regime_router" in advisory["live_block_reasons"]


def test_regime_router_blocks_pullback_v2_in_risk_off(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(
        lake,
        {
            "BTC-USDT": (100.0, 95.0),
            "ETH-USDT": (100.0, 96.0),
            "SOL-USDT": (100.0, 95.5),
            "BNB-USDT": (100.0, 97.0),
        },
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-22",
                    "strategy_candidate": "v5.pullback_reversal_shadow_btc",
                    "rule_version": "confirmed_reversal_v0.2",
                    "symbol": "BTC-USDT",
                    "regime_state": "risk_off",
                    "horizon_hours": 24,
                    "net_bps_after_cost": 55.0,
                    "mae_bps": -30.0,
                    "label_status": "complete",
                    "cost_quality": "mixed_actual_proxy",
                    "generated_at_utc": datetime(2026, 5, 22, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "v5_pullback_reversal_shadow",
    )

    build_and_publish_regime_router(lake, as_of_date="2026-05-22")

    advisory = read_parquet_dataset(lake / "gold" / "regime_strategy_advisory").to_dicts()[0]
    blocked = json.loads(advisory["blocked_strategy_candidates"])
    assert "pullback_reversal_v2" in blocked
    assert "pullback_reversal_blocked_in_down_or_risk_off" in advisory["live_block_reasons"]


def test_regime_router_uses_orderbook_spread_rollup_for_market_regime(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(
        lake,
        {
            "BTC-USDT": (100.0, 101.0),
            "ETH-USDT": (100.0, 101.0),
            "SOL-USDT": (100.0, 101.0),
            "BNB-USDT": (100.0, 101.0),
        },
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "channel": "books5",
                    "minute_ts": datetime(2026, 5, 22, 12, 0, tzinfo=UTC),
                    "ts": datetime(2026, 5, 22, 12, 0, tzinfo=UTC),
                    "spread_bps": 2.0,
                },
                {
                    "symbol": "BNB-USDT",
                    "channel": "books5",
                    "minute_ts": datetime(2026, 5, 22, 12, 1, tzinfo=UTC),
                    "ts": datetime(2026, 5, 22, 12, 1, tzinfo=UTC),
                    "spread_bps": 4.0,
                },
            ]
        ),
        lake / "silver" / "orderbook_spread_1m",
    )

    build_and_publish_regime_router(lake, as_of_date="2026-05-22")

    regime = read_parquet_dataset(lake / "gold" / "market_regime_daily").to_dicts()[0]
    assert regime["avg_spread_bps"] == 3.0
    assert regime["liquidity_thin"] is False


def test_regime_router_replaces_same_day_partition_without_duplicate_rows(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(
        lake,
        {
            "BTC-USDT": (100.0, 101.0),
            "ETH-USDT": (100.0, 103.0),
            "SOL-USDT": (100.0, 104.0),
            "BNB-USDT": (100.0, 102.8),
        },
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-22",
                    "strategy_candidate": "v5.alt_impulse_shadow",
                    "candidate_name": "v5.alt_impulse_shadow",
                    "symbol": "SOL-USDT",
                    "regime_state": "impulse",
                    "horizon_hours": 24,
                    "sample_count": 16,
                    "complete_sample_count": 16,
                    "avg_net_bps": 88.0,
                    "median_net_bps": 60.0,
                    "p25_net_bps": -20.0,
                    "win_rate": 0.70,
                    "cost_source_mix": '{"mixed_actual_proxy":16}',
                    "decision": "PAPER_READY",
                    "decision_reasons": '["paper_ready_thresholds_met"]',
                    "created_at": datetime(2026, 5, 22, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "strategy_evidence",
    )

    first = build_and_publish_regime_router(lake, as_of_date="2026-05-22")
    second = build_and_publish_regime_router(lake, as_of_date="2026-05-22")

    matrix = read_parquet_dataset(lake / "gold" / "strategy_regime_matrix")
    assert first.strategy_regime_matrix_rows == second.strategy_regime_matrix_rows
    assert matrix.height == second.strategy_regime_matrix_rows
    assert (lake / "gold" / "strategy_regime_matrix" / "as_of_date=2026-05-22").exists()
    assert not list((lake / "gold" / "strategy_regime_matrix").glob("*.parquet"))


def test_daily_export_contains_regime_reports_and_advisory_rows(tmp_path):
    lake = tmp_path / "lake"
    out = tmp_path / "exports"
    _write_market_bars(
        lake,
        {
            "BTC-USDT": (100.0, 101.0),
            "ETH-USDT": (100.0, 103.0),
            "SOL-USDT": (100.0, 104.0),
            "BNB-USDT": (100.0, 102.8),
        },
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-22",
                    "strategy_candidate": "v5.alt_impulse_shadow",
                    "symbol": "SOL-USDT",
                    "regime_state": "impulse",
                    "horizon_hours": 24,
                    "sample_count": 16,
                    "complete_sample_count": 16,
                    "avg_net_bps": 88.0,
                    "p25_net_bps": -20.0,
                    "win_rate": 0.70,
                    "cost_source_mix": '{"mixed_actual_proxy":16}',
                    "decision": "PAPER_READY",
                    "decision_reasons": '["paper_ready_thresholds_met"]',
                    "created_at": datetime(2026, 5, 22, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "strategy_evidence",
    )
    build_and_publish_regime_router(lake, as_of_date="2026-05-22")

    result = export_daily_pack(
        export_date="2026-05-22",
        lake_root=lake,
        out_dir=out,
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
        refresh_risk_permission=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/market_regime_daily.csv" in names
        assert "reports/strategy_regime_matrix.csv" in names
        assert "reports/regime_strategy_advisory.csv" in names
        advisory = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
                )
            )
        )
    regime_rows = [
        row for row in advisory if row["strategy_candidate"].startswith("regime_router:")
    ]
    assert regime_rows
    assert all(float(row["max_live_notional_usdt"]) == 0.0 for row in regime_rows)


def _write_market_bars(lake, closes: dict[str, tuple[float, float]]) -> None:
    start = datetime(2026, 5, 21, tzinfo=UTC)
    rows = []
    for symbol, (first_close, last_close) in closes.items():
        for hour in range(25):
            ratio = hour / 24
            close = first_close + (last_close - first_close) * ratio
            rows.append(
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": start + timedelta(hours=hour),
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000.0,
                    "quote_volume": 100000.0,
                    "source": "test",
                    "ingest_ts": start,
                    "is_closed": True,
                }
            )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "market_bar")
