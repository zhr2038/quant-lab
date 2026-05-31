from __future__ import annotations

import csv
import io
import zipfile

import polars as pl
import pytest

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.export.daily import (
    _bnb_missed_opportunity_samples_for_export,
    export_daily_pack,
)


def test_bnb_missed_opportunity_filters_bnb_alpha6_buy_no_order_after_may_30():
    candidates = pl.DataFrame(
        [
            {
                "run_id": "20260530_01",
                "ts_utc": "2026-05-30T01:00:00Z",
                "symbol": "BNB/USDT",
                "entry_close": 600.0,
                "alpha6_side": "buy",
                "alpha6_score": 0.88,
                "f3_vol_adj_ret": 0.12,
                "f4_volume_expansion": 0.25,
                "f5_rsi_trend_confirm": 0.31,
                "final_score": 0.92,
                "final_decision": "no_order",
                "expected_edge_bps": 80.0,
                "required_edge_bps": 30.0,
            },
            {
                "run_id": "20260530_02",
                "ts_utc": "2026-05-30T02:00:00Z",
                "symbol": "BNB-USDT",
                "entry_close": 601.0,
                "alpha6_side": "buy",
                "alpha6_score": 0.81,
                "f3_vol_adj_ret": 0.07,
                "f4_volume_expansion": 0.11,
                "f5_rsi_trend_confirm": 0.20,
                "final_score": 0.86,
                "final_decision": "blocked",
                "expected_edge_bps": 65.0,
                "required_edge_bps": 40.0,
            },
            {
                "run_id": "sell_side_excluded",
                "ts_utc": "2026-05-30T03:00:00Z",
                "symbol": "BNB-USDT",
                "entry_close": 602.0,
                "alpha6_side": "sell",
                "final_score": 0.99,
                "final_decision": "no_order",
                "expected_edge_bps": 90.0,
                "required_edge_bps": 30.0,
            },
            {
                "run_id": "opened_excluded",
                "ts_utc": "2026-05-30T04:00:00Z",
                "symbol": "BNB-USDT",
                "entry_close": 603.0,
                "alpha6_side": "buy",
                "final_score": 0.99,
                "final_decision": "OPEN_LONG",
                "expected_edge_bps": 90.0,
                "required_edge_bps": 30.0,
            },
            {
                "run_id": "old_excluded",
                "ts_utc": "2026-05-29T23:00:00Z",
                "symbol": "BNB-USDT",
                "entry_close": 590.0,
                "alpha6_side": "buy",
                "final_score": 0.99,
                "final_decision": "no_order",
                "expected_edge_bps": 90.0,
                "required_edge_bps": 30.0,
            },
            {
                "run_id": "weak_edge_excluded",
                "ts_utc": "2026-05-30T05:00:00Z",
                "symbol": "BNB-USDT",
                "entry_close": 604.0,
                "alpha6_side": "buy",
                "final_score": 0.70,
                "final_decision": "no_order",
                "expected_edge_bps": 20.0,
                "required_edge_bps": 30.0,
            },
        ]
    )
    market = _bnb_market_frame()

    samples = _bnb_missed_opportunity_samples_for_export(
        candidate_events=candidates,
        market_bars=market,
    )

    assert samples["run_id"].to_list() == ["20260530_01", "20260530_02"]
    first = samples.to_dicts()[0]
    assert first["entry_close"] == 600.0
    assert first["f3"] == 0.12
    assert first["f4"] == 0.25
    assert first["f5"] == 0.31
    assert first["future_4h_net_bps"] == pytest.approx(170.0)
    assert first["missed_profit_flag"] is True


def test_daily_export_contains_bnb_missed_opportunity_reports(tmp_path):
    lake = tmp_path / "lake"
    candidates = pl.DataFrame(
        [
            {
                "run_id": "20260530_01",
                "ts_utc": "2026-05-30T01:00:00Z",
                "symbol": "BNB-USDT",
                "entry_close": 600.0,
                "alpha6_side": "buy",
                "alpha6_score": 0.88,
                "f3_vol_adj_ret": 0.12,
                "f4_volume_expansion": 0.25,
                "f5_rsi_trend_confirm": 0.31,
                "final_score": 0.92,
                "final_decision": "no_order",
                "expected_edge_bps": 80.0,
                "required_edge_bps": 30.0,
            }
        ]
    )
    write_parquet_dataset(candidates, lake / "silver" / "v5_candidate_event")
    write_parquet_dataset(_bnb_market_frame(), lake / "silver" / "market_bar")

    export = export_daily_pack(
        export_date="2026-05-31",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(export.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/bnb_missed_opportunity_samples.csv" in names
        assert "reports/bnb_missed_opportunity_summary.md" in names
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(
                        "reports/bnb_missed_opportunity_samples.csv"
                    ).decode("utf-8")
                )
            )
        )
        summary = archive.read("reports/bnb_missed_opportunity_summary.md").decode("utf-8")

    assert len(rows) == 1
    assert rows[0]["run_id"] == "20260530_01"
    assert rows[0]["missed_profit_flag"] == "True"
    assert "qualifying_samples: 1" in summary
    assert "missed_profit_samples: 1" in summary


def _bnb_market_frame() -> pl.DataFrame:
    rows = []
    for hour, close in [
        (1, 600.0),
        (5, 612.0),
        (9, 618.0),
        (13, 630.0),
        (25, 660.0),
    ]:
        rows.append(
            {
                "venue": "okx",
                "symbol": "BNB-USDT",
                "timeframe": "1h",
                "ts": f"2026-05-30T{hour:02d}:00:00Z"
                if hour < 24
                else "2026-05-31T01:00:00Z",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1.0,
                "is_closed": True,
            }
        )
    return pl.DataFrame(rows)
