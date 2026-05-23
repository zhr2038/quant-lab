import csv
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.sol_protect_paper_loss import (
    build_and_publish_sol_protect_paper_loss_attribution,
    sol_protect_paper_loss_summary_md,
)


def test_sol_protect_paper_loss_attribution_builds_entry_rows_and_summary(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(lake)
    _write_sol_paper_runs(lake)

    result = build_and_publish_sol_protect_paper_loss_attribution(
        lake,
        as_of_date="2026-05-22",
    )

    assert result.attribution_rows == 2
    attribution = read_parquet_dataset(
        lake / "gold" / "sol_protect_paper_loss_attribution"
    )
    rows = attribution.to_dicts()
    loss = next(row for row in rows if row["run_id"] == "loss-run")
    assert loss["paper_pnl_24h"] == -333.0
    assert loss["entry_position_in_24h_range"] is not None
    tags = set(json.loads(loss["attribution_tags"]))
    assert "paper_loss_observed" in tags
    assert "weak_f4_volume_confirmation" in tags
    assert "weak_f5_rsi_confirmation" in tags
    assert "btc_trend_unfavorable" in tags

    summary = read_parquet_dataset(lake / "gold" / "sol_protect_paper_loss_summary")
    summary_row = summary.to_dicts()[0]
    assert summary_row["entry_count"] == 2
    assert summary_row["loss_entry_count"] == 1
    assert "weak_f4_volume_confirmation" in summary_row["common_loss_tags"]

    md = sol_protect_paper_loss_summary_md(summary, attribution)
    assert "SOL Protect Paper Loss Attribution" in md
    assert "loss-run" in md
    assert "weak_f4_volume_confirmation" in md


def test_daily_export_contains_sol_protect_paper_loss_reports(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(lake)
    _write_sol_paper_runs(lake)
    build_and_publish_sol_protect_paper_loss_attribution(lake, as_of_date="2026-05-22")

    export = export_daily_pack(
        export_date="2026-05-22",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        pre_export_v5_refresh=False,
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(export.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/sol_protect_paper_loss_attribution.csv" in names
        assert "reports/sol_protect_paper_loss_summary.md" in names
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(
                        "reports/sol_protect_paper_loss_attribution.csv"
                    ).decode("utf-8")
                )
            )
        )
        summary = archive.read("reports/sol_protect_paper_loss_summary.md").decode("utf-8")

    assert rows
    assert rows[0]["contract_version"] == "v5.quant_lab.telemetry.v2"
    assert "SOL Protect Paper Loss Attribution" in summary


def _write_sol_paper_runs(lake):
    rows = [
        {
            "as_of_date": "2026-05-22",
            "bundle_ts": "2026-05-22T08:00:00Z",
            "bundle_name": "v5_bundle_20260522_0800",
            "strategy_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            "proposal_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
            "run_id": "loss-run",
            "ts_utc": "2026-05-22T08:00:00Z",
            "symbol": "SOL-USDT",
            "would_enter": True,
            "estimated_fill_px": 110.0,
            "alpha6_score": 0.92,
            "alpha6_side": "long",
            "f4_volume_expansion": 0.20,
            "f5_rsi_trend_confirm": 0.30,
            "risk_level": "PROTECT",
            "btc_trend_state": "trend_down",
            "market_regime": "RISK_OFF",
            "paper_pnl_bps_4h": -50.0,
            "paper_pnl_bps_8h": -75.0,
            "paper_pnl_bps_12h": -110.0,
            "paper_pnl_bps_24h": -333.0,
            "mae_bps_24h": -180.0,
            "mfe_bps_24h": 12.0,
            "raw_payload_json": "{}",
        },
        {
            "as_of_date": "2026-05-22",
            "bundle_ts": "2026-05-22T08:00:00Z",
            "bundle_name": "v5_bundle_20260522_0800",
            "strategy_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            "proposal_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
            "run_id": "profit-run",
            "ts_utc": "2026-05-22T09:00:00Z",
            "symbol": "SOL-USDT",
            "would_enter": True,
            "estimated_fill_px": 108.0,
            "alpha6_score": 0.88,
            "alpha6_side": "long",
            "f4_volume_expansion": 0.70,
            "f5_rsi_trend_confirm": 0.65,
            "risk_level": "NORMAL",
            "btc_trend_state": "trend_up",
            "market_regime": "TREND_UP",
            "paper_pnl_bps_4h": 12.0,
            "paper_pnl_bps_8h": 20.0,
            "paper_pnl_bps_12h": 31.0,
            "paper_pnl_bps_24h": 44.0,
            "mae_bps_24h": -30.0,
            "mfe_bps_24h": 80.0,
            "raw_payload_json": "{}",
        },
    ]
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "v5_paper_strategy_run")


def _write_market_bars(lake):
    start = datetime(2026, 5, 21, 8, tzinfo=UTC)
    rows = []
    for idx in range(26):
        ts = start + timedelta(hours=idx)
        rows.append(
            {
                "venue": "okx",
                "symbol": "SOL-USDT",
                "timeframe": "1H",
                "ts": ts.isoformat(),
                "open": 105.0,
                "high": 120.0,
                "low": 100.0,
                "close": 109.0,
                "volume": 1000.0,
                "quote_volume": 109000.0,
            }
        )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "market_bar")
