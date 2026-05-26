from __future__ import annotations

import csv
import io
import zipfile
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.bnb_swing_exit_policy import (
    build_and_publish_bnb_swing_exit_policy_review,
)


def test_bnb_swing_exit_policy_reviews_giveback_after_unrealized_profit(tmp_path):
    lake = tmp_path / "lake"
    entry_ts = datetime(2026, 5, 23, 22, tzinfo=UTC)
    _write_bnb_swing_inputs(lake, entry_ts=entry_ts)

    result = build_and_publish_bnb_swing_exit_policy_review(
        lake,
        as_of_date="2026-05-24",
    )

    assert result.review_rows == 1
    review = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_review")
    row = review.to_dicts()[0]
    assert row["entry_px"] == pytest.approx(657.9)
    assert row["actual_exit_px"] == pytest.approx(651.3)
    assert row["actual_exit_net_bps"] == pytest.approx(-120.0, abs=0.5)
    assert row["highest_px_after_entry"] == pytest.approx(665.0)
    assert row["max_unrealized_bps"] > 100
    assert row["profit_lock_50bps_exit"] > row["actual_exit_net_bps"]
    assert row["delayed_exit_12h_net_bps"] > row["actual_exit_net_bps"]
    assert row["best_shadow_exit_policy"] == row["best_exit_policy"]
    assert row["best_exit_policy"] in {
        "profit_lock_50bps",
        "fixed_hold_4h",
        "delayed_exit_12h",
        "trailing_atr",
    }
    assert row["delta_vs_actual_bps"] > 0
    assert row["diagnosis"] in {
        "profit_lock_too_late",
        "gave_back_unrealized_profit",
        "trailing_variant_may_improve",
    }
    summary = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_summary").to_dicts()[0]
    assert summary["sample_count"] == 1
    assert summary["min_sample_count_for_exit_change"] == 10
    assert summary["decision"] == "RESEARCH_ONLY"
    assert "insufficient_sample_count_for_exit_change" in summary["decision_reasons"]


def test_bnb_swing_exit_policy_reads_v5_profit_lock_shadow(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-bnb-shadow",
                    "entry_ts": "2026-05-23T22:00:00Z",
                    "symbol": "BNB/USDT",
                    "entry_px": "657.9",
                    "actual_exit_ts": "2026-05-24T22:01:00Z",
                    "actual_exit_px": "651.3",
                    "actual_exit_net_bps": "-120.22",
                    "max_unrealized_bps": "69.9",
                    "profit_lock_30bps_exit": "0.0",
                    "profit_lock_50bps_exit": "20.0",
                    "delayed_exit_6h": "-40.0",
                    "delayed_exit_12h": "29.28",
                    "delayed_exit_24h": "-12.0",
                    "best_shadow_exit_policy": "delayed_exit_12h",
                    "exit_reason": "atr_trailing/exit_signal_priority",
                    "source_entry_id": "bnb-shadow-entry",
                }
            ]
        ),
        lake / "silver" / "v5_bnb_profit_lock_shadow",
    )

    result = build_and_publish_bnb_swing_exit_policy_review(lake, as_of_date="2026-05-24")

    assert result.review_rows == 1
    row = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_review").to_dicts()[0]
    assert row["symbol"] == "BNB-USDT"
    assert row["actual_exit_net_bps"] == pytest.approx(-120.22)
    assert row["delayed_exit_12h_net_bps"] == pytest.approx(29.28)
    assert row["best_shadow_exit_policy"] == "delayed_exit_12h"
    summary = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_summary").to_dicts()[0]
    assert summary["sample_count"] == 1
    assert summary["decision"] == "RESEARCH_ONLY"
    assert summary["delayed_exit_better_count"] == 1
    assert "delayed_exit_would_improve_exit" in summary["decision_reasons"]


def test_daily_export_contains_bnb_swing_exit_policy_review(tmp_path):
    lake = tmp_path / "lake"
    entry_ts = datetime(2026, 5, 23, 22, tzinfo=UTC)
    _write_bnb_swing_inputs(lake, entry_ts=entry_ts)
    build_and_publish_bnb_swing_exit_policy_review(lake, as_of_date="2026-05-24")

    result = export_daily_pack(
        export_date="2026-05-24",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/bnb_swing_exit_policy_review.csv" in names
        assert "reports/bnb_swing_exit_policy_summary.md" in names
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/bnb_swing_exit_policy_review.csv").decode("utf-8")
                )
            )
        )
        summary = archive.read("reports/bnb_swing_exit_policy_summary.md").decode("utf-8")

    assert len(rows) == 1
    assert rows[0]["symbol"] == "BNB-USDT"
    assert float(rows[0]["actual_exit_net_bps"]) == pytest.approx(-120.0, abs=0.5)
    assert "BNB Swing Exit Policy Review" in summary
    assert "read-only research" in summary


def _write_bnb_swing_inputs(lake, *, entry_ts: datetime) -> None:
    exit_ts = entry_ts + timedelta(hours=24, minutes=1)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-bnb-swing",
                    "ts_utc": entry_ts.isoformat(),
                    "symbol": "BNB-USDT",
                    "normalized_symbol": "BNB-USDT",
                    "side": "buy",
                    "action": "entry",
                    "qty": "1",
                    "price": "657.9",
                    "notional_usdt": "657.9",
                    "fee_usdt": "0.647",
                    "strategy_id": "BNB_SWING_F3",
                    "trade_id": "bnb-entry-1",
                },
                {
                    "run_id": "run-bnb-swing",
                    "ts_utc": exit_ts.isoformat(),
                    "symbol": "BNB-USDT",
                    "normalized_symbol": "BNB-USDT",
                    "side": "sell",
                    "action": "exit",
                    "qty": "1",
                    "price": "651.3",
                    "notional_usdt": "651.3",
                    "fee_usdt": "0.647",
                    "strategy_id": "BNB_SWING_F3",
                    "exit_reason": "atr_trailing/exit_signal_priority",
                    "trade_id": "bnb-exit-1",
                },
            ]
        ),
        lake / "silver" / "v5_trade_event",
    )
    bars = [
        _bar(entry_ts, close=657.9, high=658.0, low=657.0),
        _bar(entry_ts + timedelta(hours=1), close=662.0, high=665.0, low=661.0),
        _bar(entry_ts + timedelta(hours=4), close=660.0, high=661.0, low=659.0),
        _bar(entry_ts + timedelta(hours=8), close=655.0, high=657.0, low=654.0),
        _bar(entry_ts + timedelta(hours=12), close=652.0, high=653.0, low=651.0),
        _bar(entry_ts + timedelta(hours=24), close=651.3, high=652.0, low=650.0),
        _bar(entry_ts + timedelta(hours=37), close=660.0, high=661.0, low=659.0),
    ]
    write_parquet_dataset(pl.DataFrame(bars), lake / "silver" / "market_bar")


def _bar(ts: datetime, *, close: float, high: float, low: float) -> dict:
    return {
        "venue": "okx",
        "symbol": "BNB-USDT",
        "market_type": "SPOT",
        "timeframe": "1H",
        "ts": ts,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100.0,
        "quote_volume": close * 100.0,
        "source": "test_fixture",
        "ingest_ts": ts + timedelta(minutes=1),
    }
