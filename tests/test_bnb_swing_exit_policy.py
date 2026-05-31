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
    bnb_swing_exit_policy_summary_md,
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
    assert row["fixed_hold_12h_from_entry_net_bps"] == row["fixed_hold_12h_net_bps"]
    assert row["delayed_exit_12h_from_actual_exit_net_bps"] == row["delayed_exit_12h_net_bps"]
    assert row["best_shadow_exit_policy"] == row["best_exit_policy"]
    assert row["best_exit_policy"] in {
        "profit_lock_50bps",
        "fixed_hold_4h_from_entry",
        "fixed_hold_12h_from_entry",
        "delayed_exit_12h_from_actual_exit",
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
    assert summary["recommendation"] == "collect_more_samples"
    assert summary["review_reason"] == "sample_count_lt_10"
    assert summary["sample_count_gate_met_for_exit_change_review"] is False
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
    assert row["delayed_exit_12h_from_actual_exit_net_bps"] == pytest.approx(29.28)
    assert row["best_shadow_exit_policy"] == "delayed_exit_12h_from_actual_exit"
    consistency = read_parquet_dataset(
        lake / "gold" / "bnb_exit_policy_v5_vs_quant_lab_consistency"
    ).to_dicts()[0]
    assert consistency["consistency_status"] == "V5_ONLY"
    summary = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_summary").to_dicts()[0]
    assert summary["sample_count"] == 1
    assert summary["decision"] == "RESEARCH_ONLY"
    assert summary["recommendation"] == "collect_more_samples"
    assert summary["review_reason"] == "sample_count_lt_10"
    assert summary["delayed_exit_better_count"] == 1
    assert "delayed_exit_would_improve_exit" in summary["decision_reasons"]


def test_bnb_swing_exit_policy_collects_more_samples_below_gate(tmp_path):
    lake = tmp_path / "lake"
    _write_bnb_profit_lock_shadow_rows(lake, count=2, delayed_exit_12h=29.28)

    build_and_publish_bnb_swing_exit_policy_review(lake, as_of_date="2026-05-24")

    summary = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_summary").to_dicts()[0]
    assert summary["sample_count"] == 2
    assert summary["decision"] == "RESEARCH_ONLY"
    assert summary["recommendation"] == "collect_more_samples"
    assert summary["review_reason"] == "sample_count_lt_10"
    assert summary["sample_count_gate_met_for_exit_change_review"] is False


def test_bnb_swing_exit_policy_reviews_only_after_sample_gate_and_clear_advantage(tmp_path):
    lake = tmp_path / "lake"
    _write_bnb_profit_lock_shadow_rows(lake, count=10, delayed_exit_12h=29.28)

    build_and_publish_bnb_swing_exit_policy_review(lake, as_of_date="2026-05-24")

    summary = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_summary").to_dicts()[0]
    assert summary["sample_count"] == 10
    assert summary["sample_count_gate_met_for_exit_change_review"] is True
    assert summary["shadow_help_rate"] == pytest.approx(1.0)
    assert summary["avg_best_shadow_improvement_bps"] > 50.0
    assert summary["decision"] == "REVIEW_EXIT_POLICY"
    assert summary["recommendation"] == "REVIEW_EXIT_POLICY"
    assert summary["review_reason"] == "sample_gate_met_shadow_exit_outperforms_actual"


def test_bnb_swing_exit_policy_does_not_review_when_sample_gate_met_but_advantage_weak(tmp_path):
    lake = tmp_path / "lake"
    _write_bnb_profit_lock_shadow_rows(
        lake,
        count=10,
        delayed_exit_6h=-95.0,
        delayed_exit_12h=-90.0,
        delayed_exit_24h=-95.0,
        profit_lock_30bps_exit=-95.0,
        profit_lock_50bps_exit=-95.0,
    )

    build_and_publish_bnb_swing_exit_policy_review(lake, as_of_date="2026-05-24")

    summary = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_summary").to_dicts()[0]
    assert summary["sample_count"] == 10
    assert summary["sample_count_gate_met_for_exit_change_review"] is True
    assert summary["recommendation"] == "no_change_recommended"
    assert summary["review_reason"] == "sample_gate_met_no_clear_shadow_advantage"
    assert summary["decision"] == "RESEARCH_ONLY"


def test_bnb_swing_exit_policy_dedupes_shadow_and_trade_rows_for_latest_summary(tmp_path):
    lake = tmp_path / "lake"
    entry_ts = datetime(2026, 5, 24, 6, tzinfo=UTC)
    exit_ts = entry_ts + timedelta(hours=24, minutes=1)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "entry_ts": entry_ts.isoformat(),
                    "symbol": "BNB/USDT",
                    "entry_px": "657.9",
                    "actual_exit_ts": exit_ts.isoformat(),
                    "actual_exit_px": "651.3",
                    "actual_exit_net_bps": "-120.22",
                    "max_unrealized_bps": "-30.0",
                    "delayed_exit_12h": "29.28",
                    "best_shadow_exit_policy": "delayed_exit_12h",
                    "exit_reason": "atr_trailing/exit_signal_priority",
                }
            ]
        ),
        lake / "silver" / "v5_bnb_profit_lock_shadow",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "20260524_06",
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
                    "trade_id": "46678654",
                },
                {
                    "run_id": "20260524_06",
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
                    "trade_id": "bnb-exit-46678654",
                },
            ]
        ),
        lake / "silver" / "v5_trade_event",
    )
    bars = [
        _bar(entry_ts, close=657.9, high=658.0, low=657.0),
        _bar(entry_ts + timedelta(hours=1), close=662.0, high=662.5, low=661.0),
        _bar(entry_ts + timedelta(hours=4), close=660.0, high=661.0, low=659.0),
        _bar(entry_ts + timedelta(hours=8), close=655.0, high=657.0, low=654.0),
        _bar(entry_ts + timedelta(hours=12), close=652.0, high=653.0, low=651.0),
        _bar(entry_ts + timedelta(hours=24), close=651.3, high=652.0, low=650.0),
        _bar(exit_ts + timedelta(hours=24), close=664.0, high=664.5, low=663.0),
    ]
    write_parquet_dataset(pl.DataFrame(bars), lake / "silver" / "market_bar")

    result = build_and_publish_bnb_swing_exit_policy_review(lake, as_of_date="2026-05-25")

    assert result.review_rows == 1
    review = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_review")
    row = review.to_dicts()[0]
    assert row["run_id"] == "20260524_06"
    assert row["source_entry_id"] == "46678654"
    assert row["duplicate_row_count"] == 2
    assert row["selected_for_summary"] is True
    assert row["summary_eligible"] is True
    assert row["v5_vs_quant_lab_consistency_status"] != "MISMATCH"
    assert row["max_unrealized_bps"] == pytest.approx(69.92, abs=0.2)
    assert row["max_unrealized_bps"] > 0

    summary = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_summary")
    markdown = bnb_swing_exit_policy_summary_md(summary, review)
    assert "source_entry_id: 46678654" in markdown
    assert "duplicate_row_count: 2" in markdown
    assert "max_unrealized_bps: 69." in markdown
    assert "max_unrealized_bps: -30" not in markdown


def test_bnb_swing_exit_policy_excludes_v5_quant_lab_mismatch_from_summary(tmp_path):
    lake = tmp_path / "lake"
    entry_ts = datetime(2026, 5, 24, 6, tzinfo=UTC)
    exit_ts = entry_ts + timedelta(hours=24, minutes=1)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "entry_ts": entry_ts.isoformat(),
                    "symbol": "BNB/USDT",
                    "entry_px": "657.9",
                    "actual_exit_ts": exit_ts.isoformat(),
                    "actual_exit_px": "651.3",
                    "actual_exit_net_bps": "-120.22",
                    "max_unrealized_bps": "69.9",
                    "delayed_exit_12h": "29.28",
                    "best_shadow_exit_policy": "delayed_exit_12h",
                    "source_entry_id": "46678654",
                }
            ]
        ),
        lake / "silver" / "v5_bnb_profit_lock_shadow",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "20260524_06",
                    "ts_utc": entry_ts.isoformat(),
                    "symbol": "BNB-USDT",
                    "normalized_symbol": "BNB-USDT",
                    "side": "buy",
                    "action": "entry",
                    "qty": "1",
                    "price": "657.9",
                    "strategy_id": "BNB_SWING_F3",
                    "trade_id": "46678654",
                },
                {
                    "run_id": "20260524_06",
                    "ts_utc": exit_ts.isoformat(),
                    "symbol": "BNB-USDT",
                    "normalized_symbol": "BNB-USDT",
                    "side": "sell",
                    "action": "exit",
                    "qty": "1",
                    "price": "651.3",
                    "strategy_id": "BNB_SWING_F3",
                    "trade_id": "bnb-exit-46678654",
                },
            ]
        ),
        lake / "silver" / "v5_trade_event",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                _bar(entry_ts, close=657.9, high=658.0, low=657.0),
                _bar(exit_ts, close=651.3, high=652.0, low=650.0),
                _bar(exit_ts + timedelta(hours=12), close=653.0, high=653.5, low=652.0),
            ]
        ),
        lake / "silver" / "market_bar",
    )

    result = build_and_publish_bnb_swing_exit_policy_review(lake, as_of_date="2026-05-25")

    assert result.review_rows == 1
    assert "bnb_exit_policy_v5_quant_lab_mismatch" in result.warnings
    review = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_review").to_dicts()[0]
    assert review["selected_for_summary"] is False
    assert review["summary_eligible"] is False
    assert review["v5_vs_quant_lab_consistency_status"] == "MISMATCH"
    assert (
        "delayed_exit_12h_from_actual_exit_net_bps_mismatch"
        in review["v5_vs_quant_lab_mismatch_reason"]
    )
    consistency = read_parquet_dataset(
        lake / "gold" / "bnb_exit_policy_v5_vs_quant_lab_consistency"
    ).to_dicts()[0]
    assert consistency["consistency_status"] == "MISMATCH"
    assert consistency["selected_for_summary_allowed"] is False
    summary = read_parquet_dataset(lake / "gold" / "bnb_swing_exit_policy_summary").to_dicts()[0]
    assert summary["sample_count"] == 0
    assert "bnb_exit_policy_v5_quant_lab_mismatch_excluded" in summary["decision_reasons"]


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
        assert "reports/bnb_exit_policy_v5_vs_quant_lab_consistency.csv" in names
        assert "reports/bnb_swing_exit_policy_summary.md" in names
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/bnb_swing_exit_policy_review.csv").decode("utf-8")
                )
            )
        )
        summary = archive.read("reports/bnb_swing_exit_policy_summary.md").decode("utf-8")
        consistency_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(
                        "reports/bnb_exit_policy_v5_vs_quant_lab_consistency.csv"
                    ).decode("utf-8")
                )
            )
        )

    assert len(rows) == 1
    assert len(consistency_rows) == 1
    assert rows[0]["symbol"] == "BNB-USDT"
    assert float(rows[0]["actual_exit_net_bps"]) == pytest.approx(-120.0, abs=0.5)
    assert "delayed_exit_12h_from_actual_exit_net_bps" in rows[0]
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


def _write_bnb_profit_lock_shadow_rows(
    lake,
    *,
    count: int,
    delayed_exit_6h: float = -40.0,
    delayed_exit_12h: float,
    delayed_exit_24h: float = -12.0,
    profit_lock_30bps_exit: float = 0.0,
    profit_lock_50bps_exit: float = 20.0,
) -> None:
    entry_start = datetime(2026, 5, 23, 22, tzinfo=UTC)
    rows = []
    for index in range(count):
        entry_ts = entry_start + timedelta(hours=index)
        rows.append(
            {
                "run_id": f"run-bnb-shadow-{index}",
                "entry_ts": entry_ts.isoformat(),
                "symbol": "BNB/USDT",
                "entry_px": "657.9",
                "actual_exit_ts": (entry_ts + timedelta(hours=24, minutes=1)).isoformat(),
                "actual_exit_px": "651.3",
                "actual_exit_net_bps": "-120.22",
                "max_unrealized_bps": "69.9",
                "profit_lock_30bps_exit": str(profit_lock_30bps_exit),
                "profit_lock_50bps_exit": str(profit_lock_50bps_exit),
                "delayed_exit_6h": str(delayed_exit_6h),
                "delayed_exit_12h": str(delayed_exit_12h),
                "delayed_exit_24h": str(delayed_exit_24h),
                "best_shadow_exit_policy": "delayed_exit_12h",
                "exit_reason": "atr_trailing/exit_signal_priority",
                "source_entry_id": f"bnb-shadow-entry-{index}",
            }
        )
    write_parquet_dataset(
        pl.DataFrame(rows),
        lake / "silver" / "v5_bnb_profit_lock_shadow",
    )


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
