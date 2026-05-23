import csv
import io
import zipfile

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.portfolio import (
    build_and_publish_research_portfolio_status,
    dedupe_research_portfolio_status,
    research_portfolio_summary_md,
)


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
    assert rows["v5.multi_position_k1"]["status"] == "KILL"
    assert rows["v5.multi_position_k2"]["status"] == "KILL"
    assert rows["v5.btc_leadership_f5_low"]["status"] == "KILL"
    assert rows["v5.btc_leadership_no_breakout"]["status"] == "KILL"
    assert rows["v5.portfolio_trend_following"]["status"] == "KILL"
    assert rows["ETH_F3_DOMINANT_ENTRY_PAPER_V1"]["status"] == "PAPER"
    assert rows["ETH_F3_DOMINANT_ENTRY_PAPER_V1"]["paper_days"] == 3
    assert rows["v5.alt_impulse_shadow"]["status"] == "SHADOW"
    assert rows["v5.alt_impulse_shadow"]["action"] == "REGIME_SHADOW"
    assert rows["v5.late_entry_chase_guard_shadow"]["status"] == "SHADOW"
    assert rows["v5.pullback_reversal_v1"]["status"] == "KILL"
    assert rows["v5.multi_position_k2"]["killed_research_count"] >= 1
    assert rows["v5.multi_position_k2"]["freed_research_slots"] >= 1


def test_research_portfolio_downgrades_eth_f3_when_48h_paper_is_negative(tmp_path):
    lake = tmp_path / "lake"
    _write_strategy_evidence(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": "ETH_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "latest_board_decision": "KEEP_SHADOW",
                    "paper_days": 3,
                    "entry_day_count": 2,
                    "live_block_reason": '["eth_f3_48h_paper_pnl_negative"]',
                    "created_at": "2026-05-20T00:00:00Z",
                }
            ]
        ),
        lake / "gold" / "paper_strategy_daily",
    )

    build_and_publish_research_portfolio_status(lake, as_of_date="2026-05-20")

    rows = {
        row["research_id"]: row
        for row in read_parquet_dataset(lake / "gold" / "research_portfolio_status").to_dicts()
    }
    eth = rows["ETH_F3_DOMINANT_ENTRY_PAPER_V1"]
    assert eth["status"] == "SHADOW"
    assert eth["action"] == "KEEP_SHADOW"
    assert eth["reason"] == "eth_f3_negative_paper_streak_keep_shadow_no_live"


def test_research_portfolio_downgrades_negative_paper_streaks(tmp_path):
    lake = tmp_path / "lake"
    _write_strategy_evidence(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
                    "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
                    "symbol": "SOL-USDT",
                    "latest_board_decision": "KEEP_SHADOW",
                    "paper_days": 2,
                    "entry_day_count": 2,
                    "paper_negative_streak": 2,
                    "latest_paper_trend": "negative_24h_or_48h_streak",
                    "live_block_reason": '["paper_negative_24h_or_48h_streak"]',
                    "created_at": "2026-05-22T00:00:00Z",
                }
            ]
        ),
        lake / "gold" / "paper_strategy_daily",
    )

    build_and_publish_research_portfolio_status(lake, as_of_date="2026-05-22")

    rows = {
        row["research_id"]: row
        for row in read_parquet_dataset(lake / "gold" / "research_portfolio_status").to_dicts()
    }
    sol = rows["SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1"]
    assert sol["status"] == "SHADOW"
    assert sol["action"] == "KEEP_SHADOW"
    assert sol["paper_negative_streak"] == 2
    assert sol["downgrade_reason"] == "paper_negative_24h_or_48h_streak"
    assert sol["reason"] == "sol_protect_negative_paper_streak_keep_shadow_no_live"


def test_research_portfolio_uses_latest_as_of_date_for_paper_downgrade(tmp_path):
    lake = tmp_path / "lake"
    _write_strategy_evidence(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-22",
                    "proposal_id": "ETH_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "latest_board_decision": "PAPER_READY",
                    "paper_days": 4,
                    "entry_day_count": 2,
                    "paper_negative_streak": 0,
                    "latest_paper_trend": "waiting_for_24h_48h_labels",
                    "live_block_reason": "[]",
                    "created_at": "2026-05-23T23:00:00Z",
                },
                {
                    "as_of_date": "2026-05-23",
                    "proposal_id": "ETH_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "latest_board_decision": "KEEP_SHADOW",
                    "paper_days": 4,
                    "entry_day_count": 2,
                    "paper_negative_streak": 2,
                    "latest_paper_trend": "negative_24h_or_48h_streak",
                    "live_block_reason": '["paper_negative_24h_or_48h_streak"]',
                    "created_at": "2026-05-23T00:00:00Z",
                },
            ]
        ),
        lake / "gold" / "paper_strategy_daily",
    )

    build_and_publish_research_portfolio_status(lake, as_of_date="2026-05-23")

    rows = {
        row["research_id"]: row
        for row in read_parquet_dataset(lake / "gold" / "research_portfolio_status").to_dicts()
    }
    eth = rows["ETH_F3_DOMINANT_ENTRY_PAPER_V1"]
    assert eth["status"] == "SHADOW"
    assert eth["action"] == "KEEP_SHADOW"
    assert eth["paper_negative_streak"] == 2
    assert eth["downgrade_reason"] == "paper_negative_24h_or_48h_streak"


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
    assert by_id["v5.multi_position_k1"]["status"] == "KILL"
    assert by_id["v5.portfolio_trend_following"]["status"] == "KILL"
    assert by_id["SOL_F4_VOLUME_EXPANSION_PAPER_V1"]["status"] == "PAPER"
    assert "freed_research_slots" in rows[0]
    assert "active_research_count" in rows[0]

    with zipfile.ZipFile(result.zip_path) as archive:
        summary = archive.read("reports/research_portfolio_summary.md").decode("utf-8")

    assert "## CLOSE_RESEARCH" in summary
    assert "## CONTINUE_PAPER" in summary
    assert "## CONTINUE_SHADOW" in summary
    assert "v5.multi_position_k2" in summary
    assert "v5.portfolio_trend_following" in summary
    assert "avg_net_bps" in summary


def test_research_portfolio_dedupes_by_as_of_date_research_id_latest_created_at():
    frame = pl.DataFrame(
        [
            {
                "schema_version": "research_portfolio_status.v0.1",
                "as_of_date": "2026-05-20",
                "research_id": "same",
                "module": "old",
                "strategy_candidate": "candidate",
                "status": "PAUSED",
                "action": "OLD",
                "reason": "old_reason",
                "sample_count": 1,
                "complete_sample_count": 1,
                "avg_net_bps": -1.0,
                "win_rate": 0.1,
                "p25_net_bps": -10.0,
                "paper_days": 0,
                "entry_day_count": 0,
                "cost_source_mix": "{}",
                "last_review_date": "2026-05-20",
                "next_review_date": "2026-05-21",
                "recommended_new_research_slots": 0,
                "freed_research_slots": 0,
                "active_research_count": 0,
                "killed_research_count": 0,
                "created_at": "2026-05-20T00:00:00Z",
                "source": "test",
            },
            {
                "schema_version": "research_portfolio_status.v0.1",
                "as_of_date": "2026-05-20",
                "research_id": "same",
                "module": "new",
                "strategy_candidate": "candidate",
                "status": "KILL",
                "action": "CLOSE_RESEARCH",
                "reason": "new_reason",
                "sample_count": 30,
                "complete_sample_count": 30,
                "avg_net_bps": -80.0,
                "win_rate": 0.2,
                "p25_net_bps": -120.0,
                "paper_days": 0,
                "entry_day_count": 0,
                "cost_source_mix": '{"mixed_actual_proxy":30}',
                "last_review_date": "2026-05-20",
                "next_review_date": "2026-05-21",
                "recommended_new_research_slots": 1,
                "freed_research_slots": 1,
                "active_research_count": 0,
                "killed_research_count": 1,
                "created_at": "2026-05-20T01:00:00Z",
                "source": "test",
            },
        ]
    )

    deduped = dedupe_research_portfolio_status(frame)

    assert deduped.height == 1
    row = deduped.to_dicts()[0]
    assert row["research_id"] == "same"
    assert row["status"] == "KILL"
    assert row["reason"] == "new_reason"

    summary = research_portfolio_summary_md(deduped, as_of_date="2026-05-20")
    assert "## CLOSE_RESEARCH" in summary
    assert "new_reason" in summary
    assert "sample=30" in summary


def test_research_portfolio_publish_replaces_same_day_obsolete_rows(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "schema_version": "research_portfolio_status.v0.1",
                    "as_of_date": "2026-05-20",
                    "research_id": "obsolete_same_day",
                    "module": "old",
                    "strategy_candidate": "old",
                    "status": "SHADOW",
                    "action": "OLD",
                    "reason": "old",
                    "sample_count": 1,
                    "complete_sample_count": 1,
                    "avg_net_bps": 1.0,
                    "win_rate": 1.0,
                    "p25_net_bps": 1.0,
                    "paper_days": 0,
                    "entry_day_count": 0,
                    "cost_source_mix": "{}",
                    "last_review_date": "2026-05-20",
                    "next_review_date": "2026-05-21",
                    "recommended_new_research_slots": 0,
                    "freed_research_slots": 0,
                    "active_research_count": 0,
                    "killed_research_count": 0,
                    "created_at": "2026-05-20T00:00:00Z",
                    "source": "test",
                }
            ]
        ),
        lake / "gold" / "research_portfolio_status",
    )

    build_and_publish_research_portfolio_status(lake, as_of_date="2026-05-20")

    rows = read_parquet_dataset(lake / "gold" / "research_portfolio_status").to_dicts()
    assert "obsolete_same_day" not in {row["research_id"] for row in rows}


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
