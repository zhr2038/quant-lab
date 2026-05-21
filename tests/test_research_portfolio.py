import csv
import io
import zipfile

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.portfolio import build_and_publish_research_portfolio_status


def test_research_portfolio_status_prunes_and_preserves_paper_items(tmp_path):
    lake = tmp_path / "lake"
    _write_strategy_evidence(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": "ETH_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "paper_days": 3,
                    "entry_day_count": 2,
                    "created_at": "2026-05-20T00:00:00Z",
                }
            ]
        ),
        lake / "gold" / "paper_strategy_daily",
    )

    result = build_and_publish_research_portfolio_status(lake, as_of_date="2026-05-20")

    assert result.rows_written >= 10
    rows = {
        row["research_id"]: row
        for row in read_parquet_dataset(lake / "gold" / "research_portfolio_status").to_dicts()
    }
    assert rows["v5.core.momentum"]["status"] == "BASELINE_ONLY"
    assert rows["v5.multi_position_k2"]["status"] == "KILL"
    assert rows["ETH_F3_DOMINANT_ENTRY_PAPER_V1"]["status"] == "PAPER"
    assert rows["ETH_F3_DOMINANT_ENTRY_PAPER_V1"]["paper_days"] == 3
    assert rows["v5.alt_impulse_shadow"]["status"] == "SHADOW"
    assert rows["v5.alt_impulse_shadow"]["action"] == "REGIME_SHADOW"
    assert rows["v5.late_entry_chase_guard_shadow"]["status"] == "SHADOW"
    assert rows["v5.pullback_reversal_v1"]["status"] == "KILL"
    assert rows["v5.multi_position_k2"]["killed_research_count"] >= 1
    assert rows["v5.multi_position_k2"]["freed_research_slots"] >= 1


def test_daily_export_contains_research_portfolio_status(tmp_path):
    lake = tmp_path / "lake"
    _write_strategy_evidence(lake)

    result = export_daily_pack(
        export_date="2026-05-20",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        pre_export_v5_refresh=False,
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/research_portfolio_status.csv").decode("utf-8")
                )
            )
        )

    assert rows
    by_id = {row["research_id"]: row for row in rows}
    assert by_id["v5.core.momentum"]["status"] == "BASELINE_ONLY"
    assert by_id["v5.multi_position_k2"]["status"] == "KILL"
    assert by_id["SOL_F4_VOLUME_EXPANSION_PAPER_V1"]["status"] == "PAPER"
    assert "freed_research_slots" in rows[0]
    assert "active_research_count" in rows[0]


def _write_strategy_evidence(lake):
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-20",
                    "strategy_candidate": "v5.multi_position_k2",
                    "symbol": "BNB-USDT",
                    "horizon_hours": 24,
                    "sample_count": 42,
                    "complete_sample_count": 35,
                    "avg_net_bps": -80.0,
                    "p25_net_bps": -120.0,
                    "win_rate": 0.30,
                    "cost_source_mix": '{"mixed_actual_proxy":35}',
                    "decision": "KILL",
                    "created_at": "2026-05-20T00:00:00Z",
                },
                {
                    "as_of_date": "2026-05-20",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "horizon_hours": 48,
                    "sample_count": 32,
                    "complete_sample_count": 18,
                    "avg_net_bps": 18.0,
                    "p25_net_bps": -20.0,
                    "win_rate": 0.58,
                    "cost_source_mix": '{"mixed_actual_proxy":18}',
                    "decision": "PAPER_READY",
                    "created_at": "2026-05-20T00:00:00Z",
                },
                {
                    "as_of_date": "2026-05-20",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "horizon_hours": 48,
                    "sample_count": 40,
                    "complete_sample_count": 30,
                    "avg_net_bps": 45.0,
                    "p25_net_bps": -10.0,
                    "win_rate": 0.64,
                    "cost_source_mix": '{"public_spread_proxy":30}',
                    "decision": "PAPER_READY",
                    "created_at": "2026-05-20T00:00:00Z",
                },
                {
                    "as_of_date": "2026-05-20",
                    "strategy_candidate": "v5.alt_impulse_shadow",
                    "symbol": "SOL-USDT",
                    "horizon_hours": 24,
                    "sample_count": 60,
                    "complete_sample_count": 45,
                    "avg_net_bps": 50.0,
                    "p25_net_bps": -30.0,
                    "win_rate": 0.60,
                    "cost_source_mix": '{"public_spread_proxy":45}',
                    "decision": "REGIME_SHADOW",
                    "created_at": "2026-05-20T00:00:00Z",
                },
            ]
        ),
        lake / "gold" / "strategy_evidence",
    )
