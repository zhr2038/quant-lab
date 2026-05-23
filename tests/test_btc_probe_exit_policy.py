from __future__ import annotations

import csv
import io
import zipfile
from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.btc_probe_exit_policy import (
    build_and_publish_btc_probe_exit_policy_review,
)


def test_btc_strict_probe_stop_loss_compares_actual_vs_hold_labels(tmp_path):
    lake = tmp_path / "lake"
    entry_ts = datetime(2026, 5, 20, 0, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-btc-1",
                    "roundtrip_id": "rt-btc-1",
                    "symbol": "BTC-USDT",
                    "strategy_candidate": "v5.btc_leadership_probe_strict",
                    "entry_ts": entry_ts,
                    "exit_ts": entry_ts + timedelta(hours=7),
                    "entry_px": 100_000.0,
                    "exit_px": 99_500.0,
                    "actual_exit_net_bps": -37.70,
                    "would_have_held_8h_net_bps": -10.0,
                    "would_have_held_12h_net_bps": 12.0,
                    "would_have_held_24h_net_bps": 45.95,
                    "exit_reason": "probe_stop_loss",
                    "bundle_ts": datetime(2026, 5, 20, 8, tzinfo=UTC),
                }
            ]
        ),
        lake / "silver" / "v5_roundtrip",
    )

    result = build_and_publish_btc_probe_exit_policy_review(
        lake,
        as_of_date="2026-05-20",
        min_sample_count=10,
    )

    assert result.review_rows == 1
    assert result.status == "RESEARCH_ONLY"
    review = read_parquet_dataset(lake / "gold" / "btc_probe_exit_policy_review").to_dicts()[0]
    assert review["actual_exit_net_bps"] == -37.70
    assert review["would_hold_24h_net_bps"] == 45.95
    assert review["exit_policy_signal"] == "possible_stop_loss_too_early"
    summary = read_parquet_dataset(lake / "gold" / "btc_probe_exit_policy_summary").to_dicts()[0]
    assert summary["sample_count"] == 1
    assert summary["premature_stop_loss_count"] == 1
    assert summary["status"] == "RESEARCH_ONLY"
    assert "probe_stop_loss_may_be_too_early" in summary["decision_reasons"]


def test_btc_leadership_probe_roundtrip_is_deduped_across_bundles(tmp_path):
    lake = tmp_path / "lake"
    entry_ts = datetime(2026, 5, 20, 0, tzinfo=UTC)
    duplicated = {
        "run_id": "20260520_22",
        "symbol": "BTC-USDT",
        "probe_type": "btc_leadership_probe",
        "entry_ts": entry_ts,
        "exit_ts": entry_ts + timedelta(hours=7),
        "entry_px": 77_383.7,
        "exit_px": 77_246.6,
        "net_bps": -37.6992,
        "would_have_held_24h_net_bps": 45.9489,
        "exit_reason": "probe_stop_loss",
    }
    write_parquet_dataset(
        pl.DataFrame(
            [
                duplicated | {"bundle_name": "v5_live_followup_bundle_1.tar.gz"},
                duplicated | {"bundle_name": "v5_live_followup_bundle_2.tar.gz"},
            ]
        ),
        lake / "silver" / "v5_roundtrip",
    )

    build_and_publish_btc_probe_exit_policy_review(lake, as_of_date="2026-05-20")

    review = read_parquet_dataset(lake / "gold" / "btc_probe_exit_policy_review")
    assert review.height == 1
    row = review.to_dicts()[0]
    assert row["actual_exit_net_bps"] == -37.6992
    assert row["would_hold_24h_net_bps"] == 45.9489


def test_btc_probe_exit_policy_computes_hold_labels_from_market_bar(tmp_path):
    lake = tmp_path / "lake"
    entry_ts = datetime(2026, 5, 20, 0, tzinfo=UTC)
    market_rows = [
        {
            "venue": "okx",
            "symbol": "BTC-USDT",
            "timeframe": "1H",
            "ts": entry_ts + timedelta(hours=hour),
            "open": 100_000.0,
            "high": 101_000.0,
            "low": 99_000.0,
            "close": 100_000.0 + hour * 100.0,
            "volume": 1.0,
        }
        for hour in range(0, 30)
    ]
    write_parquet_dataset(pl.DataFrame(market_rows), lake / "silver" / "market_bar")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-btc-1",
                    "roundtrip_id": "rt-btc-1",
                    "symbol": "BTC/USDT",
                    "probe_name": "btc strict probe",
                    "entry_ts": entry_ts,
                    "exit_ts": entry_ts + timedelta(hours=7),
                    "entry_px": 100_000.0,
                    "actual_exit_net_bps": -37.70,
                    "exit_reason": "probe_stop_loss",
                    "selected_roundtrip_cost_bps": 30.0,
                }
            ]
        ),
        lake / "silver" / "v5_roundtrip",
    )

    build_and_publish_btc_probe_exit_policy_review(lake, as_of_date="2026-05-20")

    row = read_parquet_dataset(lake / "gold" / "btc_probe_exit_policy_review").to_dicts()[0]
    assert round(row["would_hold_24h_net_bps"], 4) == 210.0
    assert row["label_source"] == "market_bar_conservative_cost"


def test_daily_export_contains_btc_probe_exit_policy_reports(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-btc-1",
                    "roundtrip_id": "rt-btc-1",
                    "symbol": "BTC-USDT",
                    "strategy_candidate": "v5.btc_leadership_probe_strict",
                    "entry_ts": datetime(2026, 5, 20, tzinfo=UTC),
                    "actual_exit_net_bps": -37.70,
                    "would_have_held_24h_net_bps": 45.95,
                    "exit_reason": "probe_stop_loss",
                }
            ]
        ),
        lake / "silver" / "v5_roundtrip",
    )
    build_and_publish_btc_probe_exit_policy_review(lake, as_of_date="2026-05-20")

    result = export_daily_pack(
        export_date="2026-05-20",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        pre_export_v5_refresh=False,
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/btc_probe_exit_policy_review.csv" in names
        assert "reports/btc_probe_exit_policy_summary.md" in names
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/btc_probe_exit_policy_review.csv").decode("utf-8")
                )
            )
        )
    assert rows
    assert rows[0]["status"] == "RESEARCH_ONLY"
