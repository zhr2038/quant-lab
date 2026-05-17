import csv
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board


def test_alpha_discovery_board_decisions_are_candidate_symbol_regime_horizon(tmp_path):
    lake = tmp_path / "lake"
    _write_candidate_labels(lake)
    _write_candidate_events(lake)

    result = build_and_publish_alpha_discovery_board(lake, as_of_date="2026-05-10")

    board = read_parquet_dataset(lake / "gold" / "alpha_discovery_board")
    rows = {
        (row["strategy_candidate"], row["symbol"], row["regime_state"], row["horizon_hours"]): row
        for row in board.to_dicts()
    }

    assert result.alpha_discovery_board_rows == board.height
    assert rows[("v5.alt_impulse_shadow", "ETH-USDT", "impulse", 24)]["decision"] == (
        "LIVE_SMALL_READY"
    )
    assert rows[("v5.sol_protect_exception", "SOL-USDT", "protect", 24)]["decision"] == (
        "KILL"
    )
    assert rows[("v5.swing_f4_f5_alpha6", "BTC-USDT", "trend", 24)]["decision"] == (
        "PAPER_READY"
    )
    bnb_global = rows[("v5.swing_f4_f5_alpha6", "BNB-USDT", "trend", 24)]
    assert bnb_global["decision"] == "KEEP_SHADOW"
    assert "cost_source_not_trusted" in json.loads(bnb_global["decision_reasons"])
    assert rows[("v5.f4_volume_expansion_entry", "BNB-USDT", "trend", 24)][
        "decision"
    ] == "PAPER_READY"
    assert rows[("v5.btc_leadership_probe_strict", "BTC-USDT", "trend", 24)][
        "sample_count"
    ] == 12
    assert rows[("v5.f3_dominant_entry", "BNB-USDT", "trend", 24)]["decision"] == (
        "KEEP_SHADOW"
    )
    assert rows[("v5.mean_reversion_sideways", "XRP-USDT", "sideways", 24)]["decision"] == (
        "RESEARCH_ONLY"
    )
    cost_mix = json.loads(
        rows[("v5.alt_impulse_shadow", "ETH-USDT", "impulse", 24)]["cost_source_mix"]
    )
    assert cost_mix == [{"cost_source": "quant_lab_actual", "count": 60, "ratio": 1.0}]
    assert json.loads(
        rows[("v5.swing_f4_f5_alpha6", "BTC-USDT", "trend", 24)]["stability_by_day"]
    )


def test_daily_export_uses_alpha_discovery_board_lists(tmp_path):
    lake = tmp_path / "lake"
    _write_candidate_labels(lake)
    _write_candidate_events(lake)
    build_and_publish_alpha_discovery_board(lake, as_of_date="2026-05-10")

    result = export_daily_pack(
        export_date="2026-05-10",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/alpha_discovery_board.csv" in names
        assert "reports/candidate_kill_list.csv" in names
        assert "reports/candidate_shadow_watchlist.csv" in names
        assert "reports/candidate_paper_ready.csv" in names
        board = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/alpha_discovery_board.csv").decode("utf-8"))
            )
        )
        watch = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/candidate_shadow_watchlist.csv").decode("utf-8")
                )
            )
        )
        paper = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/candidate_paper_ready.csv").decode("utf-8"))
            )
        )
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        summary = archive.read("reports/strategy_evidence_summary.md").decode("utf-8")

    assert {row["strategy_candidate"] for row in board} >= {
        "v5.alt_impulse_shadow",
        "v5.sol_protect_exception",
        "v5.btc_leadership_probe_strict",
        "v5.f3_dominant_entry",
    }
    assert any(row["strategy_candidate"] == "v5.f3_dominant_entry" for row in watch)
    assert any(row["strategy_candidate"] == "v5.swing_f4_f5_alpha6" for row in paper)
    assert "v5.f3_dominant_entry" in summary
    assert not any(
        str(warning).startswith("strategy_evidence_present")
        for warning in data_quality["warnings"]
    )


def _write_candidate_labels(lake: Path) -> None:
    start = datetime(2026, 4, 1, tzinfo=UTC)
    rows: list[dict] = []
    _add_labels(
        rows,
        start=start,
        candidate="v5.alt_impulse_shadow",
        symbol="ETH-USDT",
        regime="impulse",
        net_values=[25.0] * 60,
    )
    _add_labels(
        rows,
        start=start,
        candidate="v5.sol_protect_exception",
        symbol="SOL-USDT",
        regime="protect",
        net_values=[-20.0] * 35,
    )
    _add_labels(
        rows,
        start=start,
        candidate="v5.swing_f4_f5_alpha6",
        symbol="BTC-USDT",
        regime="trend",
        net_values=[30.0] * 21 + [-20.0] * 14,
    )
    _add_labels(
        rows,
        start=start,
        candidate="v5.btc_leadership_probe_strict",
        symbol="BTC-USDT",
        regime="trend",
        net_values=[12.0] * 12,
    )
    _add_labels(
        rows,
        start=start,
        candidate="v5.f3_dominant_entry",
        symbol="BNB-USDT",
        regime="trend",
        net_values=[18.0] * 12,
    )
    _add_labels(
        rows,
        start=start,
        candidate="v5.swing_f4_f5_alpha6",
        symbol="BNB-USDT",
        regime="trend",
        net_values=[28.0] * 72,
        cost_source="global_default",
    )
    _add_labels(
        rows,
        start=start,
        candidate="v5.f4_volume_expansion_entry",
        symbol="BNB-USDT",
        regime="trend",
        net_values=[28.0] * 72,
        cost_source="mixed_actual_proxy",
    )
    _add_labels(
        rows,
        start=start,
        candidate="v5.mean_reversion_sideways",
        symbol="XRP-USDT",
        regime="sideways",
        net_values=[8.0] * 8,
    )
    write_parquet_dataset(pl.DataFrame(rows), lake / "gold" / "v5_candidate_label")


def _add_labels(
    rows: list[dict],
    *,
    start: datetime,
    candidate: str,
    symbol: str,
    regime: str,
    net_values: list[float],
    cost_source: str = "quant_lab_actual",
) -> None:
    for index, net in enumerate(net_values):
        ts = start + timedelta(hours=index)
        rows.append(
            {
                "strategy": "v5",
                "candidate_id": f"{candidate}-{symbol}-{index}",
                "run_id": f"run-{candidate}-{index}",
                "ts_utc": ts,
                "symbol": symbol,
                "strategy_candidate": candidate,
                "block_reason": "",
                "final_decision": "SHADOW",
                "horizon_hours": 24,
                "gross_bps": net + 4.0,
                "net_bps_after_cost": net,
                "mfe_bps": max(net, 0.0) + 5.0,
                "mae_bps": min(net, 0.0) - 5.0,
                "win": net > 0.0,
                "label_status": "complete",
                "cost_bps": 4.0,
                "cost_source": cost_source,
                "alpha6_side": "long",
                "regime_state": regime,
                "created_at": ts + timedelta(minutes=1),
            }
        )


def _write_candidate_events(lake: Path) -> None:
    start = datetime(2026, 4, 1, tzinfo=UTC)
    rows = [
        {
            "strategy": "v5",
            "candidate_id": f"paper-alt-{index}",
            "run_id": f"paper-run-{index}",
            "ts_utc": start + timedelta(days=index),
            "symbol": "ETH-USDT",
            "strategy_candidate": "v5.alt_impulse_shadow",
            "regime_state": "impulse",
            "final_decision": "PAPER_SHADOW",
        }
        for index in range(20)
    ]
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "v5_candidate_event")
