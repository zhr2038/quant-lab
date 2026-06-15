import csv
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import (
    STRATEGY_OPPORTUNITY_ADVISORY_TTL_SECONDS,
    _final_score_vs_alpha6_conflict_for_export,
    _final_score_vs_alpha6_conflict_summary_md,
    _late_breakout_failure_shadow_for_export,
    _paper_strategy_proposals_for_export,
    _post_impulse_overextension_shadow_for_export,
    _strategy_opportunity_advisory_for_export,
    _v5_quant_lab_consistency_dashboard_md,
    export_daily_pack,
)
from quant_lab.research.alpha_discovery import (
    build_and_publish_alpha_discovery_board,
    normalize_alpha_discovery_board_decisions,
)
from quant_lab.research.paper_tracking import (
    build_and_publish_paper_strategy_tracking,
    build_paper_strategy_daily_from_v5,
    build_paper_strategy_runs_from_v5,
    build_paper_strategy_runs_report_from_v5,
    enrich_paper_strategy_daily_from_runs,
    paper_strategy_summary_md,
)


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
        "REGIME_SHADOW"
    )
    assert "live_disabled" in json.loads(
        rows[("v5.alt_impulse_shadow", "ETH-USDT", "impulse", 24)][
            "decision_reasons"
        ]
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
    sol_proxy = rows[("v5.f4_volume_expansion_entry", "SOL-USDT", "trend", 24)]
    assert sol_proxy["decision"] == "PAPER_READY"
    assert "cost_source_not_trusted" in json.loads(sol_proxy["decision_reasons"])
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


def test_strategy_opportunity_export_emits_hype_wld_expanded_paper_rows():
    generated_at = datetime(2026, 5, 26, 8, tzinfo=UTC)
    maturity = pl.DataFrame(
        [
            {
                "symbol": "HYPE-USDT",
                "expanded_universe_maturity_state": "PAPER_READY",
                "generated_at": generated_at,
                "sample_count": 31,
                "complete_sample_count": 24,
                "avg_net_bps": 88.0,
                "p25_net_bps": -18.0,
                "win_rate": 0.64,
                "max_paper_notional_usdt": 75.0,
                "cost_source_mix": '{"public_spread_proxy":31}',
                "cost_quality": "public_proxy",
            },
            {
                "symbol": "WLD-USDT",
                "maturity_state": "PAPER_READY",
                "generated_at": generated_at,
                "sample_count": 34,
                "complete_sample_count": 27,
                "avg_net_bps": 93.0,
                "p25_net_bps": -14.0,
                "win_rate": 0.67,
                "max_paper_notional_usdt": 80.0,
                "cost_source": "mixed_actual_proxy",
                "cost_source_quality": "mixed",
            },
        ]
    )

    advisory = _strategy_opportunity_advisory_for_export(
        alpha_discovery_board=pl.DataFrame(),
        strategy_evidence=pl.DataFrame(),
        paper_proposals=pl.DataFrame(),
        risk_permissions=pl.DataFrame(),
        cost_health=pl.DataFrame(),
        paper_daily=pl.DataFrame(),
        paper_slippage=pl.DataFrame(),
        expanded_universe_maturity=maturity,
    )

    rows = {row["strategy_id"]: row for row in advisory.to_dicts()}
    hype = rows["HYPE_EXPANDED_UNIVERSE_PAPER_V1"]
    wld = rows["WLD_EXPANDED_UNIVERSE_PAPER_V1"]
    assert hype["strategy_candidate"] == "v5.expanded_universe_hype_paper"
    assert wld["strategy_candidate"] == "v5.expanded_universe_wld_paper"
    assert hype["universe_type"] == "expanded_paper"
    assert wld["universe_type"] == "expanded_paper"
    assert hype["decision"] == "PAPER_READY"
    assert wld["recommended_mode"] == "paper"
    assert hype["max_live_notional_usdt"] == 0.0
    assert wld["max_live_notional_usdt"] == 0.0
    assert hype["max_paper_notional_usdt"] == 75.0
    assert wld["max_paper_notional_usdt"] == 80.0
    assert "expanded_universe_not_live_approved" in hype["live_block_reasons"]


def test_alpha_discovery_board_normalization_dedupes_by_source_type_and_cost_rules():
    rows = [
        _board_row(
            strategy_candidate="v5.f4_volume_expansion_entry",
            symbol="SOL-USDT",
            source_type="candidate_event_label",
            avg_net_bps=20.0,
            decision="LIVE_SMALL_READY",
            cost_source_mix='[{"cost_source":"public_spread_proxy","count":72}]',
        ),
        _board_row(
            strategy_candidate="v5.f4_volume_expansion_entry",
            symbol="SOL-USDT",
            source_type="candidate_event_label",
            avg_net_bps=30.0,
            decision="LIVE_SMALL_READY",
            cost_source_mix='[{"cost_source":"public_spread_proxy","count":72}]',
        ),
        _board_row(
            strategy_candidate="v5.swing_f4_f5_alpha6",
            symbol="BNB-USDT",
            source_type="candidate_event_label",
            avg_net_bps=30.0,
            decision="PAPER_READY",
            cost_source_mix='[{"cost_source":"global_default","count":72}]',
        ),
        _board_row(
            strategy_candidate="v5.multi_position_k2",
            symbol="BNB-USDT",
            source_type="multi_position_swing_shadow_outcome",
            avg_net_bps=120.0,
            decision="PAPER_READY",
            cost_source_mix='[{"cost_source":"mixed_actual_proxy","count":72}]',
        ),
    ]

    normalized = normalize_alpha_discovery_board_decisions(pl.DataFrame(rows))
    output = {
        (row["strategy_candidate"], row["symbol"], row["source_type"]): row
        for row in normalized.to_dicts()
    }

    assert normalized.height == 3
    sol = output[("v5.f4_volume_expansion_entry", "SOL-USDT", "candidate_event_label")]
    assert sol["avg_net_bps"] == 30.0
    assert sol["decision"] == "PAPER_READY"
    bnb = output[("v5.swing_f4_f5_alpha6", "BNB-USDT", "candidate_event_label")]
    assert bnb["decision"] == "KEEP_SHADOW"
    closed = output[("v5.multi_position_k2", "BNB-USDT", "multi_position_swing_shadow_outcome")]
    assert closed["decision"] == "KILL"
    assert "closed_research_not_in_promotion_queue" in json.loads(
        closed["decision_reasons"]
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
        assert "research/alt_impulse_shadow_by_regime.csv" in names
        assert "research/alt_impulse_shadow_by_symbol_regime_horizon.csv" in names
        assert "reports/candidate_kill_list.csv" in names
        assert "reports/candidate_shadow_watchlist.csv" in names
        assert "reports/candidate_paper_ready.csv" in names
        assert "reports/paper_strategy_proposals.csv" in names
        assert "reports/strategy_opportunity_advisory.csv" in names
        board = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/alpha_discovery_board.csv").decode("utf-8"))
            )
        )
        by_regime = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("research/alt_impulse_shadow_by_regime.csv").decode("utf-8")
                )
            )
        )
        by_symbol_regime = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(
                        "research/alt_impulse_shadow_by_symbol_regime_horizon.csv"
                    ).decode("utf-8")
                )
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
        proposals = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/paper_strategy_proposals.csv").decode("utf-8")
                )
            )
        )
        advisory = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
                )
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
    assert by_regime == []
    assert by_symbol_regime == []
    assert any(row["strategy_candidate"] == "v5.f3_dominant_entry" for row in watch)
    assert any(row["strategy_candidate"] == "v5.swing_f4_f5_alpha6" for row in paper)
    proposal_ids = {row["proposal_id"] for row in proposals}
    assert "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1" in proposal_ids
    assert "SOL_F4_VOLUME_EXPANSION_PAPER_V1" in proposal_ids
    sol_proposals = [row for row in proposals if row["symbol"] == "SOL-USDT"]
    assert sol_proposals
    assert {row["recommended_mode"] for row in sol_proposals} == {"paper"}
    assert {row["required_paper_days"] for row in sol_proposals} == {"14"}
    assert {row["required_slippage_coverage"] for row in sol_proposals} == {"0.8"}
    assert not any("LIVE_SMALL_READY" in json.dumps(row) for row in proposals)
    assert all(
        "cost_source_not_actual_or_mixed" in row["live_block_reason"]
        for row in sol_proposals
    )
    assert all(row["complete_sample_count"] for row in sol_proposals)
    assert all(
        json.loads(row["entry_conditions"])["board_decision"] == "PAPER_READY"
        for row in sol_proposals
    )
    assert advisory
    paper_advisory = [row for row in advisory if row["decision"] == "PAPER_READY"]
    assert paper_advisory
    assert {row["recommended_mode"] for row in paper_advisory} == {"paper"}
    assert all(float(row["max_live_notional_usdt"] or 0) == 0.0 for row in paper_advisory)
    paused_advisory = [
        row
        for row in advisory
        if row["strategy_candidate"] == "v5.sol_protect_exception"
    ]
    assert paused_advisory
    assert {row["recommended_mode"] for row in paused_advisory} == {"research"}
    assert all("research_paused" in row["live_block_reasons"] for row in paused_advisory)
    assert "v5.f3_dominant_entry" in summary
    assert not any(
        str(warning).startswith("strategy_evidence_present")
        for warning in data_quality["warnings"]
    )


def test_paper_strategy_proposals_use_latest_board_date(tmp_path):
    lake = tmp_path / "lake"
    stale = _board_row(
        strategy_candidate="v5.sol_protect_alpha6_low_exception",
        symbol="SOL-USDT",
        source_type="protect_sol_exception_shadow_outcome",
        avg_net_bps=250.0,
        decision="PAPER_READY",
        cost_source_mix='[{"cost_source":"public_spread_proxy","count":72}]',
    ) | {"as_of_date": "2026-05-16", "horizon_hours": 120}
    latest_protect = _board_row(
        strategy_candidate="v5.sol_protect_alpha6_low_exception",
        symbol="SOL-USDT",
        source_type="protect_sol_exception_shadow_outcome",
        avg_net_bps=45.0,
        decision="PAPER_READY",
        cost_source_mix='[{"cost_source":"public_spread_proxy","count":72}]',
    ) | {"as_of_date": "2026-05-17", "horizon_hours": 72}
    latest_f4 = _board_row(
        strategy_candidate="v5.f4_volume_expansion_entry",
        symbol="SOL-USDT",
        source_type="candidate_event_label",
        avg_net_bps=55.0,
        decision="PAPER_READY",
        cost_source_mix='[{"cost_source":"public_spread_proxy","count":72}]',
    ) | {"as_of_date": "2026-05-17", "horizon_hours": 48}
    write_parquet_dataset(
        pl.DataFrame([stale, latest_protect, latest_f4]),
        lake / "gold" / "alpha_discovery_board",
    )

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        proposals = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/paper_strategy_proposals.csv").decode("utf-8")
                )
            )
        )

    assert {row["proposal_id"] for row in proposals} == {
        "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
        "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
    }
    assert {row["as_of_date"] for row in proposals} == {"2026-05-17"}
    assert {row["recommended_mode"] for row in proposals} == {"paper"}
    assert not any("LIVE_SMALL_READY" in json.dumps(row) for row in proposals)
    assert all(
        "cost_source_not_actual_or_mixed" in row["live_block_reason"]
        for row in proposals
    )


def test_strategy_opportunity_advisory_handles_mixed_numeric_columns(tmp_path):
    lake = tmp_path / "lake"
    board_rows = [
        _board_row(
            strategy_candidate="v5.stale_candidate",
            symbol="ETH-USDT",
            source_type="candidate_event_label",
            avg_net_bps=99.0,
            decision="PAPER_READY",
            cost_source_mix='[{"cost_source":"public_spread_proxy","count":72}]',
        )
        | {"as_of_date": "2026-05-09"},
        _board_row(
            strategy_candidate="v5.multi_position_k2",
            symbol="BTC-USDT",
            source_type="multi_position_swing_shadow_outcome",
            avg_net_bps=-25.0,
            decision="KILL",
            cost_source_mix='[{"cost_source":"mixed_actual_proxy","count":12}]',
        ),
        _board_row(
            strategy_candidate="v5.f4_volume_expansion_entry",
            symbol="SOL-USDT",
            source_type="factor_contribution_outcome",
            avg_net_bps=42.0,
            decision="PAPER_READY",
            cost_source_mix='[{"cost_source":"public_spread_proxy","count":72}]',
        ),
    ]
    write_parquet_dataset(pl.DataFrame(board_rows), lake / "gold" / "alpha_discovery_board")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-10",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "paper_days": 2,
                    "paper_slippage_coverage": 0.727273,
                    "arrival_mid_coverage": 0.9,
                }
            ]
        ),
        lake / "gold" / "paper_slippage_coverage",
    )

    result = export_daily_pack(
        export_date="2026-05-10",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
                )
            )
        )

    assert {row["strategy_candidate"] for row in rows} == {
        "v5.f4_volume_expansion_entry",
        "v5.multi_position_k2",
    }
    sol = next(row for row in rows if row["symbol"] == "SOL-USDT")
    assert sol["recommended_mode"] == "paper"
    assert float(sol["max_live_notional_usdt"]) == 0.0
    assert sol["slippage_coverage"] == "0.727273"


def test_sol_paper_strategy_tracking_waits_for_v5_telemetry(tmp_path):
    lake = tmp_path / "lake"
    rows = [
        _board_row(
            strategy_candidate="v5.sol_protect_alpha6_low_exception",
            symbol="SOL-USDT",
            source_type="protect_sol_exception_shadow_outcome",
            avg_net_bps=42.0,
            decision="PAPER_READY",
            cost_source_mix='[{"cost_source":"public_spread_proxy","count":72}]',
        )
        | {"as_of_date": "2026-05-17", "horizon_hours": 72},
        _board_row(
            strategy_candidate="v5.f4_volume_expansion_entry",
            symbol="SOL-USDT",
            source_type="candidate_event_label",
            avg_net_bps=35.0,
            decision="PAPER_READY",
            cost_source_mix='[{"cost_source":"public_spread_proxy","count":72}]',
        )
        | {"as_of_date": "2026-05-17", "horizon_hours": 48},
    ]
    write_parquet_dataset(
        pl.DataFrame(rows),
        lake / "gold" / "alpha_discovery_board",
    )

    result = build_and_publish_paper_strategy_tracking(lake, as_of_date="2026-05-17")

    runs = read_parquet_dataset(lake / "gold" / "paper_strategy_runs")
    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily")
    slippage = read_parquet_dataset(lake / "gold" / "paper_slippage_coverage")
    assert result.paper_strategy_runs == 2
    assert set(runs["proposal_id"].to_list()) == {
        "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
        "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
    }
    assert set(runs["recommended_mode"].to_list()) == {"paper"}
    assert set(runs["paper_tracking_status"].to_list()) == {
        "waiting_for_v5_paper_telemetry"
    }
    assert set(runs["tracking_stage"].to_list()) == {"proposed_paper_strategy"}
    assert set(runs["would_enter"].to_list()) == {False}
    assert set(runs["would_exit"].to_list()) == {False}
    assert set(daily["paper_days"].to_list()) == {0}
    assert set(daily["paper_tracking_status"].to_list()) == {
        "waiting_for_v5_paper_telemetry"
    }
    assert set(daily["live_eligible"].to_list()) == {False}
    assert set(slippage["paper_slippage_coverage"].to_list()) == {0.0}
    assert set(slippage["coverage_status"].to_list()) == {
        "waiting_for_v5_paper_telemetry"
    }

    export = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )
    with zipfile.ZipFile(export.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/paper_strategy_runs.csv" in names
        assert "reports/paper_strategy_daily.csv" in names
        assert "reports/paper_slippage_coverage.csv" in names
        exported_runs = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/paper_strategy_runs.csv").decode("utf-8"))
            )
        )
    assert {row["recommended_mode"] for row in exported_runs} == {"paper"}
    assert {row["paper_tracking_status"] for row in exported_runs} == {
        "waiting_for_v5_paper_telemetry"
    }
    assert not any("LIVE_SMALL_READY" in json.dumps(row) for row in exported_runs)


def test_paper_strategy_tracking_uses_v5_telemetry_when_present(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-18",
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "recommended_mode": "paper",
                    "would_enter": "true",
                    "would_exit": "false",
                    "would_size": "100",
                    "paper_pnl": "0.42",
                    "paper_pnl_bps": "42",
                    "arrival_bid": "172.10",
                    "arrival_ask": "172.14",
                    "arrival_mid": "172.12",
                    "estimated_spread_bps": "2.324",
                    "expected_order_type": "post_only_limit",
                    "estimated_fill_px": "172.13",
                    "cost_source_mix": '{"public_spread_proxy":1}',
                    "live_block_reason": '["cost_source_not_actual_or_mixed"]',
                    "required_paper_days": "14",
                    "required_slippage_coverage": "0.8",
                    "raw_payload_json": "{}",
                }
            ]
        ),
        lake / "silver" / "v5_paper_strategy_run",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-18",
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "recommended_mode": "paper",
                    "paper_days": "1",
                    "cumulative_paper_pnl_usdt": "0.42",
                    "required_paper_days": "14",
                    "required_slippage_coverage": "0.8",
                    "live_eligible": "false",
                    "raw_payload_json": "{}",
                }
            ]
        ),
        lake / "silver" / "v5_paper_strategy_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-18",
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "paper_days": "1",
                    "paper_slippage_coverage": "0.0",
                    "required_slippage_coverage": "0.8",
                    "coverage_status": "insufficient_slippage_observations",
                    "raw_payload_json": "{}",
                }
            ]
        ),
        lake / "silver" / "v5_paper_slippage_coverage",
    )

    result = build_and_publish_paper_strategy_tracking(lake, as_of_date="2026-05-18")

    runs = read_parquet_dataset(lake / "gold" / "paper_strategy_runs")
    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily")
    slippage = read_parquet_dataset(lake / "gold" / "paper_slippage_coverage")
    assert result.paper_strategy_runs == 1
    assert set(runs["paper_tracking_status"].to_list()) == {"active"}
    assert set(runs["tracking_stage"].to_list()) == {"active_paper_strategy"}
    assert set(runs["would_enter"].to_list()) == {True}
    assert runs["arrival_mid"].to_list() == [172.12]
    assert runs["expected_order_type"].to_list() == ["post_only_limit"]
    assert set(daily["paper_days"].to_list()) == {0}
    assert set(daily["heartbeat_days"].to_list()) == {0}
    assert set(daily["heartbeat_day_count"].to_list()) == {0}
    assert set(daily["entry_day_count"].to_list()) == {1}
    assert set(daily["would_enter_count"].to_list()) == {1}
    assert set(daily["paper_pnl_observed_count"].to_list()) == {1}
    assert set(daily["paper_pnl_day_count"].to_list()) == {1}
    assert set(daily["paper_tracking_status"].to_list()) == {"active"}
    assert set(daily["arrival_mid_coverage"].to_list()) == {1.0}
    assert set(daily["spread_observation_coverage"].to_list()) == {1.0}
    assert set(daily["live_eligible"].to_list()) == {False}
    assert set(slippage["paper_tracking_status"].to_list()) == {
        "active"
    }


def test_sol_f4_factor_condition_candidate_event_generates_paper_entry(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-24",
                    "candidate_id": "sol_strong_001",
                    "run_id": "run_20260524_10",
                    "ts_utc": "2026-05-24T10:00:00Z",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "final_decision": "no_order",
                    "alpha6_side": "buy",
                    "f4_volume_expansion": "0.25",
                    "expected_edge_bps": "90",
                    "required_edge_bps": "30",
                    "cost_gate_verified": "true",
                    "current_regime": "ALT_IMPULSE",
                    "final_score": "0.93",
                    "cost_source": "mixed_actual_proxy",
                    "raw_payload_json": "{}",
                    "bundle_ts": datetime(2026, 5, 24, 10, tzinfo=UTC),
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )

    build_and_publish_paper_strategy_tracking(lake, as_of_date="2026-05-24")

    run = read_parquet_dataset(lake / "gold" / "paper_strategy_runs").to_dicts()[0]
    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily").to_dicts()[0]
    assert run["proposal_id"] == "SOL_F4_VOLUME_EXPANSION_PAPER_V1"
    assert run["strategy_candidate"] == "v5.f4_volume_expansion_entry"
    assert run["symbol"] == "SOL-USDT"
    assert run["would_enter"] is True
    assert run["paper_trigger_type"] == "factor_condition_match"
    assert "expected_edge_gt_required_edge" in run["paper_trigger_reason"]
    assert daily["would_enter_count"] == 1
    assert daily["daily_would_enter_count"] == 1
    assert daily["cumulative_would_enter_count"] == 1
    assert daily["count_scope"] == "cumulative"
    assert daily["entry_day_count"] == 1


def test_paper_strategy_daily_keeps_v5_entry_count_daily_when_runs_are_cumulative():
    as_of = "2026-05-24"
    v5_daily = build_paper_strategy_daily_from_v5(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-25",
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "entry_count": "0",
                    "complete_count": "0",
                    "paper_days": "3",
                    "raw_payload_json": "{}",
                },
                {
                    "as_of_date": as_of,
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "entry_count": "0",
                    "complete_count": "0",
                    "paper_days": "3",
                    "raw_payload_json": "{}",
                }
            ]
        )
    )
    v5_runs = build_paper_strategy_runs_from_v5(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-23",
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "would_enter": "true",
                    "paper_pnl_bps_24h": "12.0",
                    "raw_payload_json": "{}",
                },
                {
                    "as_of_date": "2026-05-23",
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "would_enter": "true",
                    "paper_pnl_bps_24h": "-8.0",
                    "raw_payload_json": "{}",
                },
            ]
        )
    )

    enriched = enrich_paper_strategy_daily_from_runs(
        v5_daily,
        v5_runs,
        as_of_date=datetime(2026, 5, 24, tzinfo=UTC).date(),
    ).to_dicts()[0]

    assert enriched["daily_would_enter_count"] == 0
    assert enriched["cumulative_would_enter_count"] == 2
    assert enriched["would_enter_count"] == 2
    assert enriched["daily_paper_pnl_observed_count"] == 0
    assert enriched["cumulative_paper_pnl_observed_count"] == 2
    assert enriched["paper_pnl_observed_count"] == 2
    assert enriched["count_scope"] == "cumulative"
    summary = paper_strategy_summary_md(pl.DataFrame([enriched]))
    assert "今日新 entry 数: 0" in summary
    assert "累计 entry 数: 2" in summary


def test_paper_strategy_daily_splits_v5_entries_from_synthetic_would_enter():
    as_of = "2026-05-26"
    v5_daily = build_paper_strategy_daily_from_v5(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-25",
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "entry_count": "0",
                    "complete_count": "0",
                    "paper_days": "3",
                    "raw_payload_json": "{}",
                },
                {
                    "as_of_date": as_of,
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "entry_count": "0",
                    "complete_count": "0",
                    "paper_days": "4",
                    "raw_payload_json": "{}",
                }
            ]
        )
    )
    v5_runs = pl.DataFrame(
        [
            {
                "as_of_date": as_of,
                "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                "strategy_candidate": "v5.f4_volume_expansion_entry",
                "symbol": "SOL-USDT",
                "would_enter": "false",
                "final_decision": "heartbeat",
                "alpha6_side": "sell",
                "raw_payload_json": "{}",
            }
        ]
    )
    candidate_events = pl.DataFrame(
        [
            {
                "as_of_date": as_of,
                "candidate_id": f"sol_synthetic_{index}",
                "run_id": f"run_20260526_{index}",
                "ts_utc": f"2026-05-26T{index:02d}:00:00Z",
                "symbol": "SOL-USDT",
                "strategy_candidate": "v5.f3_dominant_entry",
                "final_decision": "no_order",
                "alpha6_side": "buy",
                "f4_volume_expansion": "0.25",
                "expected_edge_bps": "90",
                "required_edge_bps": "30",
                "cost_gate_verified": "true",
                "current_regime": "ALT_IMPULSE",
                "final_score": "0.93",
                "cost_source": "mixed_actual_proxy",
                "raw_payload_json": "{}",
            }
            for index in range(9)
        ]
    )
    runs = build_paper_strategy_runs_from_v5(v5_runs, candidate_events=candidate_events)

    enriched = enrich_paper_strategy_daily_from_runs(
        v5_daily,
        runs,
        as_of_date=datetime(2026, 5, 26, tzinfo=UTC).date(),
    )
    enriched_by_date = {row["as_of_date"]: row for row in enriched.to_dicts()}
    enriched_latest = enriched_by_date[as_of]
    enriched_previous = enriched_by_date["2026-05-25"]

    run_rows = runs.to_dicts()
    assert sum(row["paper_source"] == "v5_telemetry" for row in run_rows) == 1
    assert sum(row["paper_source"] == "quant_lab_synthetic" for row in run_rows) == 9
    assert enriched_previous["daily_synthetic_would_enter_count"] == 0
    assert enriched_latest["daily_v5_entry_count"] == 0
    assert enriched_latest["daily_synthetic_would_enter_count"] == 9
    assert enriched_latest["cumulative_v5_entry_count"] == 0
    assert enriched_latest["cumulative_synthetic_would_enter_count"] == 9
    assert enriched_latest["daily_would_enter_count"] == 0
    assert enriched_latest["cumulative_would_enter_count"] == 9
    assert enriched_latest["would_enter_count"] == 9
    summary = paper_strategy_summary_md(enriched)
    assert "今日 V5 实际 paper entry: 0" in summary
    assert "今日中台 synthetic would_enter: 9" in summary


def test_sol_f4_factor_condition_can_override_no_enter_v5_paper_row():
    rows = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-24",
                "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                "strategy_candidate": "v5.f4_volume_expansion_entry",
                "source_strategy_candidate": "v5.f3_dominant_entry",
                "symbol": "SOL-USDT",
                "would_enter": "false",
                "final_decision": "no_order",
                "alpha6_side": "buy",
                "f4_volume_expansion": "0.10",
                "expected_edge_bps": "80",
                "required_edge_bps": "25",
                "cost_gate_verified": "true",
                "current_regime": "TREND_UP",
                "raw_payload_json": "{}",
            }
        ]
    )

    run = build_paper_strategy_runs_from_v5(rows).to_dicts()[0]

    assert run["would_enter"] is True
    assert run["paper_trigger_type"] == "factor_condition_match"
    assert "cost_gate_verified" in run["paper_trigger_reason"]


def test_bnb_paper_synthetic_tracks_alpha6_buy_no_order():
    candidate_events = pl.DataFrame(
        [
            {
                "candidate_id": "bnb-20260530-03",
                "run_id": "run_20260530_03",
                "ts_utc": "2026-05-30T03:00:00Z",
                "symbol": "BNB-USDT",
                "strategy_candidate": "v5.f3_dominant_entry",
                "final_decision": "no_order",
                "alpha6_score": "0.994",
                "alpha6_side": "buy",
                "f4_volume_expansion": "5.82",
                "f5_rsi_trend_confirm": "0.832",
                "expected_edge_bps": "180",
                "required_edge_bps": "45",
                "cost_gate_verified": "true",
                "current_regime": "TREND_UP",
                "raw_payload_json": "{}",
            }
        ]
    )

    runs = build_paper_strategy_runs_from_v5(pl.DataFrame(), candidate_events=candidate_events)
    report_rows = build_paper_strategy_runs_report_from_v5(
        pl.DataFrame(),
        candidate_events=candidate_events,
    )

    by_proposal = {row["proposal_id"]: row for row in runs.to_dicts()}
    assert set(by_proposal) == {
        "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
        "BNB_RISK_ON_BUY_PAPER_V1",
    }
    assert {row["would_enter"] for row in by_proposal.values()} == {True}
    assert all(row["paper_source"] == "quant_lab_synthetic" for row in by_proposal.values())
    assert all(row["would_size_usdt"] == 100.0 for row in by_proposal.values())
    assert all(
        "bnb_factor_condition_match" in row["paper_trigger_reason"]
        for row in by_proposal.values()
    )

    reports_by_proposal = {row["proposal_id"]: row for row in report_rows.to_dicts()}
    assert reports_by_proposal["BNB_F3_DOMINANT_ENTRY_PAPER_V1"]["alpha6_score"] == 0.994
    assert reports_by_proposal["BNB_RISK_ON_BUY_PAPER_V1"]["symbol"] == "BNB-USDT"


def test_bnb_paper_synthetic_blocks_when_edge_not_verified():
    candidate_events = pl.DataFrame(
        [
            {
                "candidate_id": "bnb-edge-fail",
                "run_id": "run_20260530_10",
                "ts_utc": "2026-05-30T10:00:00Z",
                "symbol": "BNB-USDT",
                "strategy_candidate": "v5.f3_dominant_entry",
                "final_decision": "no_order",
                "alpha6_score": "0.95",
                "alpha6_side": "buy",
                "expected_edge_bps": "20",
                "required_edge_bps": "45",
                "cost_gate_verified": "true",
                "current_regime": "ALT_IMPULSE",
                "raw_payload_json": "{}",
            }
        ]
    )

    rows = build_paper_strategy_runs_from_v5(
        pl.DataFrame(),
        candidate_events=candidate_events,
    ).to_dicts()

    assert len(rows) == 2
    assert {row["would_enter"] for row in rows} == {False}
    assert {row["no_sample_reason"] for row in rows} == {"edge_not_above_required"}


def test_export_daily_pack_includes_bnb_paper_reports(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "bnb-20260530-16",
                    "run_id": "run_20260530_16",
                    "ts_utc": "2026-05-30T16:00:00Z",
                    "symbol": "BNB-USDT",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "final_decision": "no_order",
                    "alpha6_score": "0.994",
                    "alpha6_side": "buy",
                    "f4_volume_expansion": "5.82",
                    "f5_rsi_trend_confirm": "0.832",
                    "expected_edge_bps": "180",
                    "required_edge_bps": "45",
                    "cost_gate_verified": "true",
                    "current_regime": "ALT_IMPULSE",
                    "raw_payload_json": "{}",
                    "bundle_ts": datetime(2026, 5, 30, 16, tzinfo=UTC),
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )

    export = export_daily_pack(
        export_date="2026-05-30",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(export.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/bnb_paper_strategy_runs.csv" in names
        assert "reports/bnb_paper_strategy_daily.csv" in names
        rows = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/bnb_paper_strategy_runs.csv").decode("utf-8"))
            )
        )
        daily = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/bnb_paper_strategy_daily.csv").decode("utf-8"))
            )
        )
    assert {row["proposal_id"] for row in rows} == {
        "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
        "BNB_RISK_ON_BUY_PAPER_V1",
    }
    assert {row["would_enter"] for row in rows} == {"True"}
    assert {row["live_eligible"].lower() for row in daily} == {"false"}


def test_final_score_vs_alpha6_conflict_quantifies_bnb_no_order():
    ts = datetime(2026, 5, 30, 3, tzinfo=UTC)
    candidate_events = pl.DataFrame(
        [
            {
                "candidate_id": "bnb-conflict",
                "run_id": "run_20260530_03",
                "ts_utc": ts,
                "symbol": "BNB-USDT",
                "strategy_candidate": "v5.f3_dominant_entry",
                "final_score": "-0.17",
                "final_decision": "no_order",
                "no_signal_reason": "final_score_negative",
                "block_reason": "negative_expectancy_fast_fail_open_block",
                "alpha6_score": "0.994",
                "alpha6_side": "buy",
                "f1_mom_5d": "0.2",
                "f2_mom_20d": "0.3",
                "f3_vol_adj_ret": "0.91",
                "f4_volume_expansion": "5.82",
                "f5_rsi_trend_confirm": "0.832",
                "expected_edge_bps": "180",
                "required_edge_bps": "45",
                "cost_gate_verified": "true",
                "cost_bps": "30",
            },
            {
                "candidate_id": "eth-weak",
                "run_id": "run_20260530_03",
                "ts_utc": ts,
                "symbol": "ETH-USDT",
                "final_score": "-0.1",
                "final_decision": "no_order",
                "alpha6_score": "0.5",
                "alpha6_side": "buy",
                "expected_edge_bps": "180",
                "required_edge_bps": "45",
                "cost_gate_verified": "true",
            },
            {
                "candidate_id": "btc-blocked",
                "run_id": "run_20260530_03",
                "ts_utc": ts,
                "symbol": "BTC-USDT",
                "final_score": "0.2",
                "final_decision": "blocked",
                "block_reason": "negative_expectancy_fast_fail_open_block",
                "alpha6_score": "0.95",
                "alpha6_side": "buy",
                "expected_edge_bps": "120",
                "required_edge_bps": "45",
                "cost_gate_verified": "true",
                "cost_bps": "30",
            },
        ]
    )
    market_bars = pl.DataFrame(
        [
            {"symbol": "BNB-USDT", "ts": ts, "close": 642.3},
            {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=4), "close": 675.0},
            {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=8), "close": 680.0},
            {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=12), "close": 690.0},
            {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=24), "close": 720.0},
            {"symbol": "BTC-USDT", "ts": ts, "close": 70000.0},
            {"symbol": "BTC-USDT", "ts": ts + timedelta(hours=4), "close": 71000.0},
        ]
    )
    negative_expectancy = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "negexp_closed_cycles": "3",
                "negexp_net_expectancy_bps": "-151.83",
                "adjusted_entry_expectancy_bps": "0",
                "entry_bad_cycles": "0",
                "exit_bad_cycles": "1",
                "min_hold_violation_cycles": "1",
            }
        ]
    )

    rows = _final_score_vs_alpha6_conflict_for_export(
        candidate_events=candidate_events,
        market_bars=market_bars,
        negative_expectancy=negative_expectancy,
    ).to_dicts()

    assert len(rows) == 2
    row = next(row for row in rows if row["symbol"] == "BNB-USDT")
    assert row["symbol"] == "BNB-USDT"
    assert row["alpha6_score"] == 0.994
    assert row["final_decision"] == "no_order"
    assert row["block_reason"] == "negative_expectancy_fast_fail_open_block"
    assert row["negative_expectancy_net_bps"] == "-151.83"
    assert row["negative_expectancy_fast_fail_net_bps"] is None
    assert row["future_4h_net_bps"] > 400.0
    assert row["future_24h_net_bps"] > 1100.0
    assert row["label_4h_status"] == "complete"
    assert row["label_8h_status"] == "complete"
    assert row["label_12h_status"] == "complete"
    assert row["label_24h_status"] == "complete"
    assert row["any_label_complete"] is True
    assert row["all_labels_complete"] is True
    assert row["label_status"] == "complete"
    assert row["material_profit_flag"] is True
    assert row["max_future_net_bps"] > 1100.0
    assert row["best_future_horizon_hours"] == 24
    assert row["missed_profit_flag"] is True
    blocked = next(row for row in rows if row["symbol"] == "BTC-USDT")
    assert blocked["final_decision"] == "blocked"
    assert blocked["negative_expectancy_net_bps"] is None
    assert blocked["negative_expectancy_fast_fail_net_bps"] is None

    summary = _final_score_vs_alpha6_conflict_summary_md(pl.DataFrame(rows))
    assert "conflict_count: 2" in summary
    assert "blocked_final_decision_count: 1" in summary
    assert "negative_expectancy_block_count: 2" in summary
    assert "partial_complete_count: 1" in summary
    assert "BNB-USDT" in summary
    assert "review_final_score_alpha6_conflict" in summary


def test_post_impulse_overextension_shadow_flags_late_breakout_failure():
    ts = datetime(2026, 5, 31, 12, tzinfo=UTC)
    candidates = pl.DataFrame(
        [
            {
                "run_id": "r_overextended_bnb",
                "ts_utc": ts,
                "symbol": "BNB-USDT",
                "alpha6_side": "buy",
                "alpha6_score": 0.98,
                "f3_vol_adj_ret": 12.0,
                "f4_volume_expansion": 2.0,
                "f5_rsi_trend_confirm": 0.7,
                "cost_bps": 30.0,
            }
        ]
    )
    market = pl.DataFrame(
        [
            {"symbol": "BNB-USDT", "ts": ts - timedelta(hours=48), "close": 100.0},
            {"symbol": "BNB-USDT", "ts": ts - timedelta(hours=24), "close": 110.0},
            {"symbol": "BNB-USDT", "ts": ts, "close": 160.0},
            {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=4), "close": 150.0},
            {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=8), "close": 148.0},
            {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=12), "close": 151.0},
        ]
    )

    overextension = _post_impulse_overextension_shadow_for_export(
        candidate_events=candidates,
        market_bars=market,
    )
    failure = _late_breakout_failure_shadow_for_export(overextension)

    row = overextension.to_dicts()[0]
    failure_row = failure.to_dicts()[0]
    assert row["symbol"] == "BNB-USDT"
    assert row["late_failure_flag"] is True
    assert "return_24h_ge_200bps" in row["overextension_reason"]
    assert row["why_not_triggered"] in (None, "")
    assert failure_row["diagnosis"] == "late_breakout_failure_after_overextension"
    assert failure_row["live_order_effect"] == "read_only_no_live_order"


def test_post_impulse_overextension_shadow_records_not_triggered_reason():
    ts = datetime(2026, 5, 31, 12, tzinfo=UTC)
    candidates = pl.DataFrame(
        [
            {
                "run_id": "r_strong_but_not_overextended",
                "ts_utc": ts,
                "symbol": "BNB-USDT",
                "alpha6_side": "buy",
                "alpha6_score": 0.8,
                "f3_vol_adj_ret": 3.0,
                "f4_volume_expansion": 0.2,
                "cost_bps": 30.0,
            }
        ]
    )
    market = pl.DataFrame(
        [
            {"symbol": "BNB-USDT", "ts": ts - timedelta(hours=24), "close": 100.0},
            {"symbol": "BNB-USDT", "ts": ts, "close": 101.0},
            {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=4), "close": 102.0},
        ]
    )

    overextension = _post_impulse_overextension_shadow_for_export(
        candidate_events=candidates,
        market_bars=market,
    )
    failure = _late_breakout_failure_shadow_for_export(overextension)

    row = overextension.to_dicts()[0]
    assert row["response_action"] == "diagnostic_only"
    assert row["overextension_reason"] in (None, "")
    assert row["why_not_triggered"] == "return_24h_lt_200bps"
    assert row["late_failure_flag"] is False
    assert failure.is_empty()


def test_consistency_dashboard_uses_bnb_paper_latest_view_not_raw_cumulative():
    latest = pl.DataFrame(
        [
            {
                "strategy_id": "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
                "paper_date": "2026-06-02",
                "symbol": "BNB-USDT",
                "entry_count": 89,
                "bundle_ts": datetime(2026, 6, 2, 12, tzinfo=UTC),
            }
        ]
    )
    raw_cumulative = pl.DataFrame(
        [
            {
                "strategy_id": "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
                "paper_date": "2026-06-01",
                "symbol": "BNB-USDT",
                "entry_count": 976,
                "bundle_ts": datetime(2026, 6, 1, 12, tzinfo=UTC),
            }
        ]
    )

    dashboard = _v5_quant_lab_consistency_dashboard_md(
        frames={
            "v5_bnb_paper_strategy_daily_latest": latest,
            "paper_strategy_daily": raw_cumulative,
        },
        final_score_alpha6_conflict=pl.DataFrame(),
        bnb_strong_alpha6_bypass_shadow=pl.DataFrame(),
        negative_expectancy_attribution=pl.DataFrame(),
        bnb_paper_daily=latest,
        risk_on_multi_buy_shadow=pl.DataFrame(),
        opportunity_advisory=pl.DataFrame(),
    )

    assert "bnb_paper_v5_entry_count: 89" in dashboard
    assert "bnb_paper_quant_lab_entry_count: 89" in dashboard
    assert "bnb_paper_quant_lab_raw_entry_count: 976" in dashboard
    assert "bnb_paper_entry_count_match: true" in dashboard
    assert "v5_bundle_lag_detected: false" in dashboard


def test_export_daily_pack_includes_final_score_alpha6_conflict(tmp_path):
    lake = tmp_path / "lake"
    ts = datetime(2026, 5, 30, 3, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "bnb-conflict",
                    "run_id": "run_20260530_03",
                    "ts_utc": ts,
                    "symbol": "BNB-USDT",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "final_score": "-0.17",
                    "final_decision": "no_order",
                    "alpha6_score": "0.994",
                    "alpha6_side": "buy",
                    "expected_edge_bps": "180",
                    "required_edge_bps": "45",
                    "cost_gate_verified": "true",
                    "cost_bps": "30",
                    "bundle_ts": ts,
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"symbol": "BNB-USDT", "ts": ts, "close": 642.3},
                {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=4), "close": 675.0},
                {"symbol": "BNB-USDT", "ts": ts + timedelta(hours=24), "close": 720.0},
            ]
        ),
        lake / "silver" / "market_bar",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "negexp_closed_cycles": "3",
                    "negexp_net_expectancy_bps": "-151.83",
                    "adjusted_entry_expectancy_bps": "0",
                    "entry_bad_cycles": "0",
                    "exit_bad_cycles": "1",
                    "min_hold_violation_cycles": "1",
                }
            ]
        ),
        lake / "silver" / "v5_negative_expectancy_consistency",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run_20260530_03",
                    "ts_utc": ts,
                    "symbol": "BNB-USDT",
                    "would_bypass": "true",
                    "alpha6_score": "0.994",
                    "f3": "12",
                    "f4": "5.82",
                    "f5": "0.832",
                    "expected_edge_bps": "180",
                    "required_edge_bps": "45",
                    "final_score": "-0.17",
                    "final_decision": "no_order",
                    "block_reason": "negative_expectancy_fast_fail_open_block",
                    "negative_expectancy_blocked": "true",
                    "future_4h_net_bps": "500",
                    "future_8h_net_bps": "700",
                    "future_12h_net_bps": "900",
                    "future_24h_net_bps": "1100",
                    "label_status": "complete",
                    "live_order_effect": "read_only_no_live_order",
                }
            ]
        ),
        lake / "gold" / "v5_bnb_strong_alpha6_bypass_shadow",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "cycle_index": "0",
                    "entry_bad": "false",
                    "exit_bad": "true",
                    "min_hold_violation": "true",
                    "would_unblock_if_adjusted": "true",
                    "block_attribution_conflict": "true",
                }
            ]
        ),
        lake / "gold" / "v5_negative_expectancy_attribution",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-30",
                    "proposal_id": "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "BNB-USDT",
                    "recommended_mode": "paper",
                    "entry_count": "1",
                    "live_eligible": "false",
                }
            ]
        ),
        lake / "gold" / "v5_bnb_paper_strategy_daily",
    )

    export = export_daily_pack(
        export_date="2026-05-30",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(export.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/final_score_vs_alpha6_conflict.csv" in names
        assert "reports/final_score_vs_alpha6_conflict_summary.md" in names
        assert "reports/bnb_strong_alpha6_bypass_shadow.csv" in names
        assert "reports/bnb_strong_alpha6_bypass_summary.md" in names
        assert "reports/negative_expectancy_attribution.csv" in names
        assert "reports/negative_expectancy_attribution_summary.md" in names
        assert "reports/bnb_paper_strategy_summary.md" in names
        assert "reports/v5_quant_lab_consistency_dashboard.md" in names
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/final_score_vs_alpha6_conflict.csv").decode("utf-8")
                )
            )
        )
        summary = archive.read("reports/final_score_vs_alpha6_conflict_summary.md").decode("utf-8")
        bypass_summary = archive.read("reports/bnb_strong_alpha6_bypass_summary.md").decode("utf-8")
        dashboard = archive.read("reports/v5_quant_lab_consistency_dashboard.md").decode("utf-8")
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BNB-USDT"
    assert rows[0]["negative_expectancy_net_bps"] == "-151.83"
    assert rows[0]["missed_profit_flag"] == "True"
    assert "conflict_count: 1" in summary
    assert "live_recommendation: none" in bypass_summary
    assert "final_score_vs_alpha6_conflict_count" in dashboard
    assert "would_unblock_if_adjusted_count" in dashboard
    assert "advisory_duplicate_key_count" in dashboard
    assert "alpha_factory_paper_ready_count_from_queue" in dashboard
    assert "bnb_paper_v5_entry_count: 1" in dashboard
    assert "bnb_paper_entry_count_match: true" in dashboard
    assert "bnb_paper_quant_lab_raw_entry_count" in dashboard
    assert "v5_bundle_lag_detected: false" in dashboard
    assert "negative_expectancy_metadata_missing_count" in dashboard
    assert "final_score_conflict_partial_complete_count" in dashboard
    assert "alpha_factory_source_mismatch_count" in dashboard


def test_sol_f4_synthetic_uses_sol_candidate_not_bnb_metrics():
    candidate_events = pl.DataFrame(
        [
            {
                "candidate_id": "bnb-candidate",
                "run_id": "run_20260527_01",
                "ts_utc": "2026-05-27T01:00:00Z",
                "symbol": "BNB-USDT",
                "strategy_candidate": "v5.f3_dominant_entry",
                "final_decision": "no_order",
                "alpha6_side": "buy",
                "f4_volume_expansion": "0.081574",
                "f5_rsi_trend_confirm": "-0.027027",
                "expected_edge_bps": "75",
                "required_edge_bps": "25",
                "cost_gate_verified": "true",
                "current_regime": "TREND_UP",
                "raw_payload_json": "{}",
            },
            {
                "candidate_id": "sol-candidate",
                "run_id": "run_20260527_01",
                "ts_utc": "2026-05-27T01:00:00Z",
                "symbol": "SOL-USDT",
                "strategy_candidate": "v5.f3_dominant_entry",
                "final_decision": "no_order",
                "alpha6_side": "sell",
                "f4_volume_expansion": "0.222222",
                "f5_rsi_trend_confirm": "0.333333",
                "expected_edge_bps": "75",
                "required_edge_bps": "25",
                "cost_gate_verified": "true",
                "current_regime": "TREND_UP",
                "raw_payload_json": "{}",
            },
        ]
    )

    rows = build_paper_strategy_runs_from_v5(pl.DataFrame(), candidate_events=candidate_events)
    report_rows = build_paper_strategy_runs_report_from_v5(
        pl.DataFrame(),
        candidate_events=candidate_events,
    )

    sol_rows = [
        row for row in rows.to_dicts()
        if row["proposal_id"] == "SOL_F4_VOLUME_EXPANSION_PAPER_V1"
    ]
    assert len(sol_rows) == 1
    run = sol_rows[0]
    assert run["proposal_id"] == "SOL_F4_VOLUME_EXPANSION_PAPER_V1"
    assert run["symbol"] == "SOL-USDT"
    assert run["source_candidate_symbol"] == "SOL-USDT"
    assert run["source_candidate_id"] == "sol-candidate"
    assert run["symbol_match_verified"] is True
    assert run["would_enter"] is False
    assert run["no_sample_reason"] == "alpha6_not_buy"
    report = next(
        row for row in report_rows.to_dicts()
        if row["proposal_id"] == "SOL_F4_VOLUME_EXPANSION_PAPER_V1"
    )
    assert report["alpha6_side"] == "sell"
    assert report["f4_volume_expansion"] == "0.222222"
    assert report["f5_rsi_trend_confirm"] == "0.333333"


def test_sol_f4_symbol_mismatch_blocks_synthetic_enter():
    rows = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-27",
                "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                "strategy_candidate": "v5.f4_volume_expansion_entry",
                "source_strategy_candidate": "v5.f3_dominant_entry",
                "source_candidate_symbol": "BNB-USDT",
                "source_candidate_id": "bnb-candidate",
                "symbol": "SOL-USDT",
                "would_enter": "true",
                "final_decision": "no_order",
                "alpha6_side": "buy",
                "f4_volume_expansion": "0.10",
                "expected_edge_bps": "80",
                "required_edge_bps": "25",
                "cost_gate_verified": "true",
                "current_regime": "TREND_UP",
                "raw_payload_json": "{}",
            }
        ]
    )

    run = build_paper_strategy_runs_from_v5(rows).to_dicts()[0]

    assert run["symbol_match_verified"] is False
    assert run["would_enter"] is False
    assert run["no_sample_reason"] == "symbol_mismatch"


def test_paper_daily_and_slippage_use_run_cost_source_mix(tmp_path):
    lake = tmp_path / "lake"
    run_rows = [
        {
            "as_of_date": f"2026-05-{day:02d}",
            "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
            "strategy_candidate": "v5.f4_volume_expansion_entry",
            "symbol": "SOL-USDT",
            "would_enter": "true",
            "paper_pnl_bps": "12.5",
            "arrival_bid": "172.10",
            "arrival_ask": "172.14",
            "arrival_mid": "172.12",
            "cost_source": "mixed_actual_proxy",
            "raw_payload_json": "{}",
            "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
        }
        for day in [17, 18]
    ]
    write_parquet_dataset(
        pl.DataFrame(run_rows),
        lake / "silver" / "v5_paper_strategy_run",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-18",
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "paper_days": "2",
                    "cost_source_mix": '{"missing":22}',
                    "raw_payload_json": "{}",
                    "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
                }
            ]
        ),
        lake / "silver" / "v5_paper_strategy_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-18",
                    "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "paper_days": "2",
                    "paper_slippage_coverage": "1.0",
                    "cost_source_mix": '{"public_spread_proxy":22}',
                    "raw_payload_json": "{}",
                    "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
                }
            ]
        ),
        lake / "silver" / "v5_paper_slippage_coverage",
    )

    build_and_publish_paper_strategy_tracking(lake, as_of_date="auto")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily").to_dicts()[0]
    slippage = read_parquet_dataset(lake / "gold" / "paper_slippage_coverage").to_dicts()[0]
    assert json.loads(daily["cost_source_mix"]) == {"mixed_actual_proxy": 2}
    assert json.loads(slippage["cost_source_mix"]) == {"mixed_actual_proxy": 2}
    assert daily["missing_cost_source_count"] == 0
    assert slippage["missing_cost_source_count"] == 0


def test_paper_daily_counts_horizon_level_pnl(tmp_path):
    lake = tmp_path / "lake"
    rows = [
        {
            "as_of_date": "2026-05-18",
            "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
            "strategy_candidate": "v5.f3_dominant_entry",
            "symbol": "ETH-USDT",
            "would_enter": "true",
            "paper_pnl_bps": "",
            "paper_pnl_bps_4h": "-50.0",
            "paper_pnl_bps_8h": "-31.3",
            "arrival_bid": "3600.0",
            "arrival_ask": "3600.5",
            "arrival_mid": "3600.25",
            "cost_source": "mixed_actual_proxy",
            "raw_payload_json": "{}",
            "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
        },
        {
            "as_of_date": "2026-05-19",
            "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
            "strategy_candidate": "v5.f3_dominant_entry",
            "symbol": "ETH-USDT",
            "would_enter": "true",
            "paper_pnl_bps": "",
            "paper_pnl_bps_4h": "-45.4",
            "arrival_bid": "3620.0",
            "arrival_ask": "3620.5",
            "arrival_mid": "3620.25",
            "cost_source": "mixed_actual_proxy",
            "raw_payload_json": "{}",
            "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
        },
    ]
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "v5_paper_strategy_run")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-19",
                    "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "paper_pnl_observed_count": "0",
                    "avg_paper_pnl_bps": "",
                    "paper_pnl_observed_count_by_horizon": "{}",
                    "complete_count_by_horizon": "{}",
                    "avg_paper_pnl_bps_by_horizon": "{}",
                    "win_rate_by_horizon": "{}",
                    "paper_pnl_day_count_by_horizon": "{}",
                    "raw_payload_json": "{}",
                    "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
                }
            ]
        ),
        lake / "silver" / "v5_paper_strategy_daily",
    )

    build_and_publish_paper_strategy_tracking(lake, as_of_date="auto")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily").to_dicts()[0]
    assert daily["paper_pnl_observed_count"] == 2
    assert daily["paper_pnl_day_count"] == 2
    assert daily["avg_paper_pnl_bps"] < 0
    assert json.loads(daily["paper_pnl_observed_count_by_horizon"]) == {
        "4h": 2,
        "8h": 1,
        "12h": 0,
        "24h": 0,
        "48h": 0,
        "72h": 0,
    }
    assert json.loads(daily["complete_count_by_horizon"]) == {
        "4h": 2,
        "8h": 1,
        "12h": 0,
        "24h": 0,
        "48h": 0,
        "72h": 0,
    }
    assert json.loads(daily["paper_pnl_day_count_by_horizon"]) == {
        "4h": 2,
        "8h": 1,
        "12h": 0,
        "24h": 0,
        "48h": 0,
        "72h": 0,
    }
    avg_by_horizon = json.loads(daily["avg_paper_pnl_bps_by_horizon"])
    assert avg_by_horizon["4h"] == -47.7
    assert avg_by_horizon["8h"] == -31.3
    assert avg_by_horizon["24h"] is None
    win_rate_by_horizon = json.loads(daily["win_rate_by_horizon"])
    assert win_rate_by_horizon["4h"] == 0.0
    assert win_rate_by_horizon["8h"] == 0.0
    assert win_rate_by_horizon["24h"] is None
    assert "waiting_for_longer_horizon_labels" in daily["live_block_reason"]
    assert daily["live_eligible"] is False


def test_eth_f3_negative_longer_horizon_downgrades_to_keep_shadow(tmp_path):
    lake = tmp_path / "lake"
    run_rows = [
        {
            "as_of_date": "2026-05-18",
            "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
            "strategy_candidate": "v5.f3_dominant_entry",
            "symbol": "ETH-USDT",
            "board_decision": "PAPER_READY",
            "would_enter": "true",
            "paper_pnl_bps_24h": "-12.0",
            "paper_pnl_bps_48h": "-25.0",
            "arrival_bid": "3600.0",
            "arrival_ask": "3600.5",
            "arrival_mid": "3600.25",
            "cost_source": "mixed_actual_proxy",
            "raw_payload_json": "{}",
            "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
        },
        {
            "as_of_date": "2026-05-19",
            "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
            "strategy_candidate": "v5.f3_dominant_entry",
            "symbol": "ETH-USDT",
            "board_decision": "PAPER_READY",
            "would_enter": "true",
            "paper_pnl_bps_24h": "-8.0",
            "paper_pnl_bps_48h": "-10.0",
            "arrival_bid": "3620.0",
            "arrival_ask": "3620.5",
            "arrival_mid": "3620.25",
            "cost_source": "mixed_actual_proxy",
            "raw_payload_json": "{}",
            "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
        },
    ]
    write_parquet_dataset(pl.DataFrame(run_rows), lake / "silver" / "v5_paper_strategy_run")

    build_and_publish_paper_strategy_tracking(lake, as_of_date="auto")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily").to_dicts()[0]
    assert daily["latest_board_decision"] == "KEEP_SHADOW"
    assert daily["negative_entry_day_count"] == 2
    assert daily["paper_negative_streak"] == 2
    assert daily["latest_paper_trend"] == "negative_24h_or_48h_streak"
    assert daily["live_eligible"] is False
    reasons = json.loads(daily["live_block_reason"])
    assert "paper_negative_24h_or_48h_streak" in reasons
    assert "eth_f3_48h_paper_pnl_negative" in reasons
    assert "keep_shadow_until_48h_recovers" in reasons


def test_eth_f3_short_horizon_positive_results_do_not_make_live_ready(tmp_path):
    lake = tmp_path / "lake"
    run_rows = [
        {
            "as_of_date": "2026-05-18",
            "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
            "strategy_candidate": "v5.f3_dominant_entry",
            "symbol": "ETH-USDT",
            "board_decision": "PAPER_READY",
            "would_enter": "true",
            "paper_pnl_bps_4h": "0.76",
            "paper_pnl_bps_8h": "12.13",
            "paper_pnl_bps_12h": "38.01",
            "arrival_bid": "3600.0",
            "arrival_ask": "3600.5",
            "arrival_mid": "3600.25",
            "cost_source": "mixed_actual_proxy",
            "raw_payload_json": "{}",
            "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
        }
    ]
    write_parquet_dataset(pl.DataFrame(run_rows), lake / "silver" / "v5_paper_strategy_run")

    build_and_publish_paper_strategy_tracking(lake, as_of_date="auto")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily").to_dicts()[0]
    assert daily["latest_board_decision"] == "PAPER_READY"
    assert daily["live_eligible"] is False
    assert json.loads(daily["win_rate_by_horizon"])["12h"] == 1.0
    reasons = json.loads(daily["live_block_reason"])
    assert "waiting_for_longer_horizon_labels" in reasons
    assert "eth_f3_paper_only_no_live" in reasons


def test_eth_f3_weak_short_horizons_keep_paper_when_48h_sample_is_positive(tmp_path):
    lake = tmp_path / "lake"
    run_rows = []
    for index in range(30):
        day = 1 + index
        run_rows.append(
            {
                "as_of_date": f"2026-05-{day:02d}",
                "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                "strategy_candidate": "v5.f3_dominant_entry",
                "symbol": "ETH-USDT",
                "board_decision": "LIVE_SMALL_READY" if index == 0 else "PAPER_READY",
                "would_enter": "true",
                "paper_pnl_bps_4h": "-8.0",
                "paper_pnl_bps_8h": "-5.0",
                "paper_pnl_bps_12h": "-3.0",
                "paper_pnl_bps_24h": "-1.0",
                "paper_pnl_bps_48h": "12.0",
                "arrival_bid": "3600.0",
                "arrival_ask": "3600.5",
                "arrival_mid": "3600.25",
                "cost_source": "mixed_actual_proxy",
                "raw_payload_json": "{}",
                "bundle_ts": datetime(2026, 5, 30, 12, tzinfo=UTC),
            }
        )
    write_parquet_dataset(pl.DataFrame(run_rows), lake / "silver" / "v5_paper_strategy_run")

    build_and_publish_paper_strategy_tracking(lake, as_of_date="auto")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily").to_dicts()[0]
    assert daily["latest_board_decision"] == "KEEP_SHADOW"
    assert daily["live_eligible"] is False
    assert daily["negative_entry_day_count"] == 30
    assert daily["paper_negative_streak"] == 30
    assert daily["latest_paper_trend"] == "negative_24h_or_48h_streak"
    assert json.loads(daily["complete_count_by_horizon"])["48h"] == 30
    assert json.loads(daily["avg_paper_pnl_bps_by_horizon"])["48h"] == 12.0
    reasons = json.loads(daily["live_block_reason"])
    assert "eth_f3_48h_positive_continue_paper" in reasons
    assert "paper_negative_24h_or_48h_streak" in reasons
    assert "eth_f3_paper_only_no_live" in reasons
    assert "eth_f3_48h_paper_pnl_negative" not in reasons


def test_eth_f3_downgraded_portfolio_disables_new_v5_paper_entries(tmp_path):
    lake = tmp_path / "lake"
    bundle_ts = datetime(2026, 5, 24, 12, tzinfo=UTC)
    v5_rows = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-24",
                "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                "strategy_candidate": "v5.f3_dominant_entry",
                "symbol": "ETH-USDT",
                "would_enter": "true",
                "would_size_usdt": "100.0",
                "final_decision": "no_order",
                "alpha6_side": "sell",
                "f3_vol_adj_ret": "-0.25",
                "final_score": "72.0",
                "current_regime": "TREND_UP",
                "paper_pnl_bps_24h": "-11.0",
                "paper_pnl_bps_48h": "-18.0",
                "raw_payload_json": "{}",
                "bundle_ts": bundle_ts,
            }
        ]
    )
    portfolio = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-24",
                "research_id": "ETH_F3_DOMINANT_ENTRY_PAPER_V1",
                "strategy_candidate": "v5.f3_dominant_entry",
                "symbol": "ETH-USDT",
                "status": "DOWNGRADED_FROM_PAPER",
                "created_at": datetime(2026, 5, 24, 0, tzinfo=UTC),
            }
        ]
    )
    write_parquet_dataset(v5_rows, lake / "silver" / "v5_paper_strategy_run")
    write_parquet_dataset(portfolio, lake / "gold" / "research_portfolio_status")

    build_and_publish_paper_strategy_tracking(lake, as_of_date="2026-05-24")

    run = read_parquet_dataset(lake / "gold" / "paper_strategy_runs").to_dicts()[0]
    assert run["would_enter"] is False
    assert run["would_size_usdt"] == 0.0
    assert run["paper_disabled_by_research_portfolio"] is True
    assert run["no_sample_reason"] == "downngraded_from_paper_no_new_entry"
    assert run["paper_pnl_bps_24h"] is None
    assert run["tracking_stage"] == "paper_review_disabled_by_research_portfolio"

    report = build_paper_strategy_runs_report_from_v5(
        v5_rows,
        research_portfolio=portfolio,
    ).to_dicts()[0]
    assert report["would_enter"] is False
    assert report["paper_disabled_by_research_portfolio"] is True
    assert report["no_sample_reason"] == "downngraded_from_paper_no_new_entry"


def test_eth_f3_new_v5_paper_entry_requires_buy_side_positive_f3_score_and_regime():
    bad_rows = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-24",
                "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                "strategy_candidate": "v5.f3_dominant_entry",
                "symbol": "ETH-USDT",
                "would_enter": "true",
                "final_decision": "no_order",
                "alpha6_side": "sell",
                "f3_vol_adj_ret": "0.5",
                "final_score": "72.0",
                "current_regime": "TREND_UP",
                "raw_payload_json": "{}",
            }
        ]
    )
    good_rows = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-24",
                "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                "strategy_candidate": "v5.f3_dominant_entry",
                "symbol": "ETH-USDT",
                "would_enter": "true",
                "final_decision": "paper_entry",
                "alpha6_side": "buy",
                "f3_vol_adj_ret": "0.5",
                "final_score": "72.0",
                "current_regime": "TREND_UP",
                "would_size_usdt": "100.0",
                "raw_payload_json": "{}",
            }
        ]
    )

    blocked = build_paper_strategy_runs_from_v5(bad_rows).to_dicts()[0]
    allowed = build_paper_strategy_runs_from_v5(good_rows).to_dicts()[0]

    assert blocked["would_enter"] is False
    assert blocked["no_sample_reason"] == "eth_f3_alpha6_side_not_buy_no_new_entry"
    assert blocked["tracking_stage"] == "paper_review_entry_conditions_not_met"
    assert allowed["would_enter"] is True
    assert allowed["would_size_usdt"] == 100.0


def test_sol_protect_negative_entry_days_downgrades_to_keep_shadow(tmp_path):
    lake = tmp_path / "lake"
    run_rows = [
        {
            "as_of_date": "2026-05-21",
            "proposal_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
            "symbol": "SOL-USDT",
            "board_decision": "PAPER_READY",
            "would_enter": "true",
            "paper_pnl_bps_24h": "-336.0",
            "arrival_bid": "170.0",
            "arrival_ask": "170.1",
            "arrival_mid": "170.05",
            "cost_source": "mixed_actual_proxy",
            "raw_payload_json": "{}",
            "bundle_ts": datetime(2026, 5, 22, 12, tzinfo=UTC),
        },
        {
            "as_of_date": "2026-05-22",
            "proposal_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
            "symbol": "SOL-USDT",
            "board_decision": "PAPER_READY",
            "would_enter": "true",
            "paper_pnl_bps_24h": "-333.0",
            "arrival_bid": "168.0",
            "arrival_ask": "168.1",
            "arrival_mid": "168.05",
            "cost_source": "mixed_actual_proxy",
            "raw_payload_json": "{}",
            "bundle_ts": datetime(2026, 5, 22, 12, tzinfo=UTC),
        },
    ]
    write_parquet_dataset(pl.DataFrame(run_rows), lake / "silver" / "v5_paper_strategy_run")

    build_and_publish_paper_strategy_tracking(lake, as_of_date="auto")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily").to_dicts()[0]
    assert daily["latest_board_decision"] == "KEEP_SHADOW"
    assert daily["negative_entry_day_count"] == 2
    assert daily["paper_negative_streak"] == 2
    assert daily["latest_paper_trend"] == "negative_24h_or_48h_streak"
    assert "paper_negative_24h_or_48h_streak" in json.loads(daily["live_block_reason"])


def test_v5_daily_negative_24h_48h_streak_downgrades_paper(tmp_path):
    lake = tmp_path / "lake"
    bundle_ts = datetime(2026, 5, 22, 12, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-21",
                    "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "latest_board_decision": "PAPER_READY",
                    "avg_paper_pnl_bps_by_horizon": json.dumps(
                        {"24h": -110.0, "48h": -352.0}
                    ),
                    "bundle_ts": bundle_ts,
                    "raw_payload_json": "{}",
                },
                {
                    "as_of_date": "2026-05-22",
                    "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "latest_board_decision": "PAPER_READY",
                    "avg_paper_pnl_bps_by_horizon": json.dumps(
                        {"4h": -20.0, "8h": -25.0, "12h": -30.0, "24h": -60.0}
                    ),
                    "bundle_ts": bundle_ts,
                    "raw_payload_json": "{}",
                },
                {
                    "as_of_date": "2026-05-23",
                    "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "latest_board_decision": "PAPER_READY",
                    "avg_paper_pnl_bps_by_horizon": "{}",
                    "bundle_ts": bundle_ts,
                    "raw_payload_json": "{}",
                },
            ]
        ),
        lake / "silver" / "v5_paper_strategy_daily",
    )

    build_and_publish_paper_strategy_tracking(lake, as_of_date="auto")

    rows = {
        row["as_of_date"]: row
        for row in read_parquet_dataset(lake / "gold" / "paper_strategy_daily").to_dicts()
    }
    latest = rows["2026-05-23"]
    assert latest["latest_board_decision"] == "KEEP_SHADOW"
    assert latest["negative_entry_day_count"] == 2
    assert latest["paper_negative_streak"] == 2
    assert latest["latest_paper_trend"] == "negative_24h_or_48h_streak"
    assert "paper_negative_24h_or_48h_streak" in json.loads(latest["live_block_reason"])


def test_v5_daily_eth_alias_merges_run_negative_streak(tmp_path):
    lake = tmp_path / "lake"
    bundle_ts = datetime(2026, 5, 23, 12, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-23",
                    "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.eth_f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "latest_board_decision": "PAPER_READY",
                    "avg_paper_pnl_bps_by_horizon": "{}",
                    "bundle_ts": bundle_ts,
                    "raw_payload_json": "{}",
                }
            ]
        ),
        lake / "silver" / "v5_paper_strategy_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-21",
                    "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "board_decision": "PAPER_READY",
                    "would_enter": "true",
                    "paper_pnl_bps_24h": "-125.0",
                    "paper_pnl_bps_48h": "-365.0",
                    "cost_source": "mixed_actual_proxy",
                    "bundle_ts": bundle_ts,
                    "raw_payload_json": "{}",
                },
                {
                    "as_of_date": "2026-05-22",
                    "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "board_decision": "PAPER_READY",
                    "would_enter": "true",
                    "paper_pnl_bps_24h": "-341.0",
                    "cost_source": "mixed_actual_proxy",
                    "bundle_ts": bundle_ts,
                    "raw_payload_json": "{}",
                },
            ]
        ),
        lake / "silver" / "v5_paper_strategy_run",
    )

    build_and_publish_paper_strategy_tracking(lake, as_of_date="auto")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily").to_dicts()[0]
    assert daily["strategy_candidate"] == "v5.eth_f3_dominant_entry"
    assert daily["latest_board_decision"] == "KEEP_SHADOW"
    assert daily["negative_entry_day_count"] == 2
    assert daily["paper_negative_streak"] == 2
    assert daily["latest_paper_trend"] == "negative_24h_or_48h_streak"


def test_eth_f3_negative_longer_horizon_downgrades_advisory(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "board_schema_version": "alpha_discovery_board.v0.1",
                    "as_of_date": "2026-05-18",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "candidate_name": "v5.f3_dominant_entry",
                    "source_type": "paper",
                    "symbol": "ETH-USDT",
                    "regime_state": "trend",
                    "horizon_hours": 24,
                    "sample_count": 80,
                    "complete_sample_count": 64,
                    "avg_net_bps": 31.5,
                    "p25_net_bps": -8.0,
                    "win_rate": 0.68,
                    "cost_source_mix": '{"mixed_actual_proxy":64}',
                    "decision": "PAPER_READY",
                    "decision_reasons": '["paper_ready_thresholds_met"]',
                    "created_at": datetime(2026, 5, 18, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "alpha_discovery_board",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-18",
                    "proposal_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "board_decision": "PAPER_READY",
                    "would_enter": "true",
                    "paper_pnl_bps_24h": "-12.0",
                    "paper_pnl_bps_48h": "-25.0",
                    "arrival_bid": "3600.0",
                    "arrival_ask": "3600.5",
                    "arrival_mid": "3600.25",
                    "cost_source": "mixed_actual_proxy",
                    "raw_payload_json": "{}",
                    "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
                }
            ]
        ),
        lake / "silver" / "v5_paper_strategy_run",
    )

    export = export_daily_pack(
        export_date="2026-05-18",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(export.zip_path) as archive:
        advisory = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
                )
            )
        )
    eth = next(row for row in advisory if row["strategy_candidate"] == "v5.f3_dominant_entry")
    assert eth["decision"] == "KEEP_SHADOW"
    assert eth["recommended_mode"] == "shadow"
    assert eth["max_live_notional_usdt"] == "0.0"


def test_paper_strategy_tracking_blocks_live_without_real_cost_quality(tmp_path):
    lake = tmp_path / "lake"
    rows = [
        {
            "paper_date": f"2026-05-{day:02d}",
            "strategy_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
            "experiment_name": "v5.f4_volume_expansion_entry",
            "symbol": "SOL-USDT",
            "recommended_mode": "paper",
            "would_enter": "true",
            "would_exit": "false",
            "would_size": "100",
            "paper_pnl": "0.1",
            "paper_pnl_bps": "1",
            "arrival_bid": "170.0",
            "arrival_ask": "170.1",
            "arrival_mid": "170.05",
            "estimated_spread_bps": "5.88",
            "cost_source_mix": '{"public_spread_proxy":1}',
            "required_paper_days": "14",
            "required_slippage_coverage": "0.8",
            "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
            "raw_payload_json": "{}",
        }
        for day in range(1, 15)
    ]
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "v5_paper_strategy_run")

    result = build_and_publish_paper_strategy_tracking(lake, as_of_date="auto")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily")
    row = daily.to_dicts()[0]
    assert result.paper_strategy_daily == 1
    assert row["paper_days"] == 0
    assert row["heartbeat_days"] == 0
    assert row["entry_day_count"] == 14
    assert row["paper_pnl_day_count"] == 14
    assert row["arrival_mid_coverage"] == 1.0
    assert row["live_eligible"] is False
    assert "cost_source_not_actual_or_mixed" in row["live_block_reason"]


def test_paper_strategy_tracking_can_mark_live_ready_after_mixed_cost_observations(tmp_path):
    lake = tmp_path / "lake"
    rows = [
        {
            "paper_date": f"2026-05-{day:02d}",
            "strategy_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
            "experiment_name": "v5.f4_volume_expansion_entry",
            "symbol": "SOL-USDT",
            "recommended_mode": "paper",
            "would_enter": "true",
            "would_exit": "false",
            "would_size": "100",
            "paper_pnl": "0.1",
            "paper_pnl_bps": "1",
            "arrival_bid": "170.0",
            "arrival_ask": "170.1",
            "arrival_mid": "170.05",
            "estimated_spread_bps": "5.88",
            "cost_source_mix": '{"mixed_actual_proxy":1}',
            "required_paper_days": "14",
            "required_slippage_coverage": "0.8",
            "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
            "raw_payload_json": "{}",
        }
        for day in range(1, 15)
    ]
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "v5_paper_strategy_run")

    build_and_publish_paper_strategy_tracking(lake, as_of_date="2026-05-18")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily")
    slippage = read_parquet_dataset(lake / "gold" / "paper_slippage_coverage")
    row = daily.to_dicts()[0]
    assert row["paper_days"] == 0
    assert row["heartbeat_days"] == 0
    assert row["entry_day_count"] == 14
    assert row["paper_pnl_day_count"] == 14
    assert row["arrival_mid_coverage"] == 1.0
    assert row["live_eligible"] is True
    assert json.loads(row["live_block_reason"]) == []
    assert slippage["arrival_mid_coverage"].to_list() == [1.0]
    assert slippage["spread_observation_coverage"].to_list() == [1.0]


def test_paper_strategy_tracking_blocks_live_on_mixed_cost_fallback_marker(tmp_path):
    lake = tmp_path / "lake"
    rows = [
        {
            "paper_date": f"2026-05-{day:02d}",
            "strategy_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
            "experiment_name": "v5.f4_volume_expansion_entry",
            "symbol": "SOL-USDT",
            "recommended_mode": "paper",
            "would_enter": "true",
            "would_exit": "false",
            "would_size": "100",
            "paper_pnl": "0.1",
            "paper_pnl_bps": "1",
            "arrival_bid": "170.0",
            "arrival_ask": "170.1",
            "arrival_mid": "170.05",
            "estimated_spread_bps": "5.88",
            "cost_source": "mixed_actual_proxy",
            "fallback_level": "SAMPLE_TOO_SMALL;SPREAD_PROXY",
            "required_paper_days": "14",
            "required_slippage_coverage": "0.8",
            "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
            "raw_payload_json": "{}",
        }
        for day in range(1, 15)
    ]
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "v5_paper_strategy_run")

    build_and_publish_paper_strategy_tracking(lake, as_of_date="2026-05-18")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily")
    row = daily.to_dicts()[0]
    reasons = json.loads(row["live_block_reason"])
    cost_mix = json.loads(row["cost_source_mix"])

    assert cost_mix["mixed_actual_proxy"] == 14
    assert cost_mix["fallback_not_live_safe"] == 14
    assert row["live_eligible"] is False
    assert "cost_source_not_trusted" in reasons
    assert "cost_source_not_actual_or_mixed" not in reasons


def test_export_daily_prefers_v5_paper_telemetry_over_pending_gold(tmp_path):
    lake = tmp_path / "lake"
    stale_pending = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-18",
                "proposal_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                "strategy_candidate": "v5.f4_volume_expansion_entry",
                "symbol": "SOL-USDT",
                "recommended_mode": "paper",
                "board_decision": "PAPER_READY",
                "suggested_horizon": "24h",
                "horizon_hours": 24,
                "would_enter": False,
                "would_exit": False,
                "would_size": 0.0,
                "would_size_usdt": 0.0,
                "paper_pnl_bps": None,
                "paper_pnl_usdt": None,
                "paper_tracking_status": "waiting_for_v5_paper_telemetry",
                "tracking_stage": "proposed_paper_strategy",
                "sample_count": 72,
                "complete_sample_count": 72,
                "avg_net_bps": 30.0,
                "p25_net_bps": -10.0,
                "win_rate": 0.7,
                "cost_source_mix": "public_spread_proxy",
                "live_block_reason": "[]",
                "required_paper_days": 14,
                "required_slippage_coverage": 0.8,
                "created_at": "2026-05-18T00:00:00Z",
                "source": "research.paper_strategy_tracking.v0.1",
                "schema_version": "paper_strategy_tracking.v1",
            }
        ]
    )
    write_parquet_dataset(stale_pending, lake / "gold" / "paper_strategy_runs")

    proposals = [
        (
            "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
            "v5.f4_volume_expansion_entry",
        ),
        (
            "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            "v5.sol_protect_alpha6_low_exception",
        ),
    ]
    run_rows = []
    for proposal_id, candidate in proposals:
        for index in range(3):
            run_rows.append(
                {
                    "paper_date": "2026-05-18",
                    "strategy_id": proposal_id,
                    "run_id": f"run_{index}",
                    "ts_utc": f"2026-05-18T0{index}:00:00Z",
                    "experiment_name": candidate,
                    "symbol": "SOL-USDT",
                    "recommended_mode": "paper",
                    "event_type": "heartbeat",
                    "would_enter": "false",
                    "would_exit": "false",
                    "would_size": "0",
                    "paper_pnl": "",
                    "paper_pnl_bps": "",
                    "final_decision": "heartbeat",
                    "no_sample_reason": "heartbeat_no_candidate",
                    "risk_level": "shadow",
                    "alpha6_score": "0.77",
                    "alpha6_side": "long",
                    "f4_volume_expansion": "true",
                    "f5_rsi_trend_confirm": "false",
                    "arrival_bid": "170.0",
                    "arrival_ask": "170.1",
                    "arrival_mid": "170.05",
                    "estimated_spread_bps": "5.88",
                    "cost_source": "public_spread_proxy",
                    "label_status": "heartbeat",
                    "required_paper_days": "14",
                    "required_slippage_coverage": "0.8",
                    "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
                    "raw_payload_json": json.dumps({"heartbeat_index": index}),
                }
            )
    run_rows.append(
        {
            "paper_date": "2026-05-17",
            "strategy_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
            "experiment_name": "v5.f4_volume_expansion_entry",
            "symbol": "SOL-USDT",
            "recommended_mode": "paper",
            "event_type": "heartbeat",
            "would_enter": "false",
            "would_exit": "false",
            "would_size": "0",
            "paper_pnl": "",
            "paper_pnl_bps": "",
            "required_paper_days": "14",
            "required_slippage_coverage": "0.8",
            "bundle_ts": datetime(2026, 5, 17, 12, tzinfo=UTC),
            "raw_payload_json": "{}",
        }
    )
    write_parquet_dataset(
        pl.DataFrame(run_rows),
        lake / "silver" / "v5_paper_strategy_run",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "paper_date": "2026-05-18",
                    "strategy_id": proposal_id,
                    "experiment_name": candidate,
                    "symbol": "SOL-USDT",
                    "recommended_mode": "paper",
                    "paper_days_to_date": "1",
                    "paper_pnl_usdt_sum": "0",
                    "required_paper_days": "14",
                    "required_slippage_coverage": "0.8",
                    "live_eligible": "false",
                    "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
                    "raw_payload_json": "{}",
                }
                for proposal_id, candidate in proposals
            ]
        ),
        lake / "silver" / "v5_paper_strategy_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy_id": proposal_id,
                    "experiment_name": candidate,
                    "symbol": "SOL-USDT",
                    "paper_days": "1",
                    "slippage_coverage": "0.0",
                    "required_slippage_coverage": "0.8",
                    "readiness_status": "insufficient_slippage_observations",
                    "bundle_ts": datetime(2026, 5, 18, 12, tzinfo=UTC),
                    "raw_payload_json": "{}",
                }
                for proposal_id, candidate in proposals
            ]
        ),
        lake / "silver" / "v5_paper_slippage_coverage",
    )

    export = export_daily_pack(
        export_date="2026-05-18",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(export.zip_path) as archive:
        runs = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/paper_strategy_runs.csv").decode("utf-8"))
            )
        )
        daily = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/paper_strategy_daily.csv").decode("utf-8"))
            )
        )

    assert len(runs) == 6
    assert {row["proposal_id"] for row in runs} == {
        "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
        "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
    }
    assert {row["strategy_candidate"] for row in runs} == {
        "v5.f4_volume_expansion_entry",
        "v5.sol_protect_alpha6_low_exception",
    }
    assert {row["paper_tracking_status"] for row in runs} == {
        "active"
    }
    assert {row["tracking_stage"] for row in runs} == {"active_paper_strategy"}
    assert {row["would_enter"] for row in runs} == {"False"}
    assert {row["paper_pnl_usdt"] for row in runs} == {""}
    assert {row["strategy_id"] for row in runs} == {
        "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
        "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
    }
    assert {row["final_decision"] for row in runs} == {"heartbeat"}
    assert {row["no_sample_reason"] for row in runs} == {"heartbeat_no_candidate"}
    assert {row["arrival_mid"] for row in runs} == {"170.05"}
    assert {row["cost_source"] for row in runs} == {"public_spread_proxy"}
    assert {row["label_status"] for row in runs} == {"heartbeat"}
    assert {row["paper_tracking_status"] for row in daily} == {
        "active"
    }
    assert {row["heartbeat_day_count"] for row in daily} == {"1"}
    assert {row["entry_day_count"] for row in daily} == {"0"}
    assert {row["would_enter_count"] for row in daily} == {"0"}
    assert {row["paper_pnl_observed_count"] for row in daily} == {"0"}
    assert {row["paper_pnl_day_count"] for row in daily} == {"0"}
    assert all("paper_active_but_no_entries_yet" in row["live_block_reason"] for row in daily)


def test_downgraded_paper_strategies_are_not_exported_as_paper(tmp_path):
    lake = tmp_path / "lake"
    board = pl.DataFrame(
        [
            _board_row(
                strategy_candidate="v5.f3_dominant_entry",
                symbol="ETH-USDT",
                source_type="candidate_event_label",
                avg_net_bps=35.0,
                decision="PAPER_READY",
                cost_source_mix='{"mixed_actual_proxy": 72}',
            ),
            _board_row(
                strategy_candidate="v5.sol_protect_alpha6_low_exception",
                symbol="SOL-USDT",
                source_type="candidate_event_label",
                avg_net_bps=45.0,
                decision="PAPER_READY",
                cost_source_mix='{"public_spread_proxy": 72}',
            ),
        ]
    )
    write_parquet_dataset(board, lake / "gold" / "alpha_discovery_board")
    paper_daily_rows = [
        _downgraded_paper_daily_row(
            proposal_id="ETH_F3_DOMINANT_ENTRY_PAPER_V1",
            candidate="v5.f3_dominant_entry",
            symbol="ETH-USDT",
        ),
        _downgraded_paper_daily_row(
            proposal_id="SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            candidate="v5.sol_protect_alpha6_low_exception",
            symbol="SOL-USDT",
        ),
    ]
    write_parquet_dataset(
        pl.DataFrame(paper_daily_rows),
        lake / "gold" / "paper_strategy_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_ts": "2026-05-09T00:00:00+00:00",
                    "generated_at": "2026-05-09T00:00:00+00:00",
                    "expires_at": "2026-05-09T03:00:00+00:00",
                    "contract_version": "v5.quant_lab.telemetry.v2",
                    "schema_version": "strategy_opportunity_advisory.v0.1",
                    "quant_lab_git_commit": "test",
                    "source_version": "test",
                    "would_block_if_enabled": False,
                    "would_enter": True,
                    "no_sample_reason": "",
                    "strategy_id": "ETH_USDT_V5_F3_DOMINANT_ENTRY",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "symbol": "ETH-USDT",
                    "v5_symbol": "ETH/USDT",
                    "decision": "PAPER_READY",
                    "recommended_mode": "paper",
                    "horizon_hours": 48,
                    "sample_count": 72,
                    "complete_sample_count": 72,
                    "avg_net_bps": 35.0,
                    "p25_net_bps": 10.0,
                    "win_rate": 0.7,
                    "cost_source_mix": '{"mixed_actual_proxy": 72}',
                    "cost_quality": "mixed",
                    "paper_days": 2,
                    "entry_day_count": 2,
                    "paper_pnl_observed_count": 2,
                    "slippage_coverage": 0.9,
                    "live_block_reasons": "[]",
                    "max_paper_notional_usdt": 100.0,
                    "max_live_notional_usdt": 0.0,
                }
            ]
        ),
        lake / "gold" / "strategy_opportunity_advisory",
    )

    export = export_daily_pack(
        export_date="2026-05-10",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(export.zip_path) as archive:
        proposals = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/paper_strategy_proposals.csv").decode("utf-8")
                )
            )
        )
        advisory = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
                )
            )
        )

    assert proposals == []
    by_candidate = {
        (row["strategy_candidate"], row["symbol"]): row
        for row in advisory
    }
    eth = by_candidate[("v5.f3_dominant_entry", "ETH-USDT")]
    sol = by_candidate[("v5.sol_protect_alpha6_low_exception", "SOL-USDT")]
    assert eth["recommended_mode"] == "shadow"
    assert sol["recommended_mode"] == "shadow"
    assert "downgraded_from_paper" in eth["live_block_reasons"]
    assert "downgraded_from_paper" in sol["live_block_reasons"]
    assert eth["max_paper_notional_usdt"] == "0.0"
    assert sol["max_paper_notional_usdt"] == "0.0"

    published = read_parquet_dataset(lake / "gold" / "strategy_opportunity_advisory")
    assert published.filter(pl.col("recommended_mode") == "paper").is_empty()


def test_strategy_opportunity_advisory_export_resets_expiry_from_generated_at():
    generated_at = datetime(2026, 5, 29, 2, 26, 22, tzinfo=UTC)
    board = pl.DataFrame(
        [
            {
                **_board_row(
                    strategy_candidate="v5.f4_volume_expansion_entry",
                    symbol="SOL-USDT",
                    source_type="candidate_event_label",
                    avg_net_bps=20.0,
                    decision="KEEP_SHADOW",
                    cost_source_mix='{"mixed_actual_proxy": 12}',
                ),
                "generated_at": generated_at,
                "expires_at": generated_at - timedelta(hours=1),
            }
        ]
    )

    advisory = _strategy_opportunity_advisory_for_export(
        alpha_discovery_board=board,
        strategy_evidence=pl.DataFrame(),
        paper_proposals=pl.DataFrame(),
        risk_permissions=pl.DataFrame(),
        cost_health=pl.DataFrame(),
        paper_daily=pl.DataFrame(),
        paper_slippage=pl.DataFrame(),
    )

    row = advisory.to_dicts()[0]
    assert row["generated_at"] == generated_at
    assert row["expires_at"] == generated_at + timedelta(
        seconds=STRATEGY_OPPORTUNITY_ADVISORY_TTL_SECONDS
    )


def test_research_portfolio_status_overrides_strategy_advisory_and_paper_proposals(tmp_path):
    board = pl.DataFrame(
        [
            _board_row(
                strategy_candidate="v5.af.failed_candidate",
                symbol="NEAR-USDT",
                source_type="alpha_factory",
                avg_net_bps=70.0,
                decision="PAPER_READY",
                cost_source_mix='{"mixed_actual_proxy": 72}',
            ),
            _board_row(
                strategy_candidate="v5.af.paused_candidate",
                symbol="WLD-USDT",
                source_type="alpha_factory",
                avg_net_bps=60.0,
                decision="PAPER_READY",
                cost_source_mix='{"mixed_actual_proxy": 72}',
            ),
            _board_row(
                strategy_candidate="v5.af.downgraded_candidate",
                symbol="OKB-USDT",
                source_type="alpha_factory",
                avg_net_bps=50.0,
                decision="PAPER_READY",
                cost_source_mix='{"mixed_actual_proxy": 72}',
            ),
            _board_row(
                strategy_candidate="v5.core.momentum",
                symbol="BTC-USDT",
                source_type="research_baseline",
                avg_net_bps=80.0,
                decision="PAPER_READY",
                cost_source_mix='{"mixed_actual_proxy": 72}',
            ),
        ]
    )
    portfolio = pl.DataFrame(
        [
            _portfolio_override_row("v5.af.failed_candidate", "KILL"),
            _portfolio_override_row(
                "v5.af.paused_candidate",
                "KILL",
                as_of_date="2026-05-09",
            ),
            _portfolio_override_row("v5.af.paused_candidate", "PAUSED"),
            _portfolio_override_row(
                "v5.af.downgraded_candidate",
                "DOWNGRADED_FROM_PAPER",
            ),
            _portfolio_override_row("v5.core.momentum", "BASELINE_ONLY"),
        ]
    )
    proposals = _paper_strategy_proposals_for_export(
        board,
        research_portfolio=portfolio,
    ).to_dicts()
    proposal_frame = pl.DataFrame(proposals) if proposals else pl.DataFrame()
    advisory = _strategy_opportunity_advisory_for_export(
        alpha_discovery_board=board,
        strategy_evidence=pl.DataFrame(),
        paper_proposals=proposal_frame,
        risk_permissions=pl.DataFrame(),
        cost_health=pl.DataFrame(),
        paper_daily=pl.DataFrame(),
        paper_slippage=pl.DataFrame(),
        research_portfolio=portfolio,
    ).to_dicts()

    assert proposals == []
    by_candidate = {row["strategy_candidate"]: row for row in advisory}
    killed = by_candidate["v5.af.failed_candidate"]
    paused = by_candidate["v5.af.paused_candidate"]
    downgraded = by_candidate["v5.af.downgraded_candidate"]
    baseline = by_candidate["v5.core.momentum"]
    assert killed["decision"] == "KILL"
    assert killed["recommended_mode"] == "none"
    assert "research_portfolio_kill" in killed["live_block_reasons"]
    assert paused["recommended_mode"] == "research"
    assert "research_paused" in paused["live_block_reasons"]
    assert downgraded["recommended_mode"] == "shadow"
    assert "downgraded_from_paper" in downgraded["live_block_reasons"]
    assert baseline["recommended_mode"] == "research"
    assert "baseline_only" in baseline["live_block_reasons"]
    assert all(float(row["max_paper_notional_usdt"] or 0.0) == 0.0 for row in advisory)


def test_strategy_advisory_uses_alpha_factory_promotion_queue_over_board():
    board = pl.DataFrame(
        [
            _board_row(
                strategy_candidate="v5.expanded_relative_strength_top1_shadow",
                symbol="TRX-USDT",
                source_type="alpha_factory",
                avg_net_bps=42.0,
                decision="PAPER_READY",
                cost_source_mix='{"mixed_actual_proxy": 72}',
            )
        ]
    )
    promotion_queue = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-25",
                "generated_at": datetime(2026, 5, 25, tzinfo=UTC),
                "strategy_candidate": "v5.expanded_relative_strength_top1_shadow",
                "symbol": "TRX-USDT",
                "horizon_hours": 24,
                "promotion_state": "KEEP_SHADOW",
                "recommended_mode": "shadow",
                "reasons": '["validation_not_paper_ready"]',
            }
        ]
    )

    advisory = _strategy_opportunity_advisory_for_export(
        alpha_discovery_board=board,
        strategy_evidence=pl.DataFrame(),
        paper_proposals=pl.DataFrame(),
        risk_permissions=pl.DataFrame(),
        cost_health=pl.DataFrame(),
        paper_daily=pl.DataFrame(),
        paper_slippage=pl.DataFrame(),
        alpha_factory_promotion_queue=promotion_queue,
    ).to_dicts()

    assert len(advisory) == 1
    row = advisory[0]
    assert row["decision"] == "KEEP_SHADOW"
    assert row["recommended_mode"] == "shadow"
    assert row["promotion_state"] == "KEEP_SHADOW"
    assert "alpha_factory_promotion_queue_not_paper_ready" in row["live_block_reasons"]
    assert row["max_paper_notional_usdt"] == 0.0


def test_strategy_advisory_caps_regime_router_alpha_factory_paper_ready():
    regime_advisory = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-25",
                "generated_at": datetime(2026, 5, 25, tzinfo=UTC),
                "current_regime": "ALT_IMPULSE",
                "recommended_mode": "paper",
                "allowed_strategy_candidates": '["v5.expanded_relative_strength_top1_shadow"]',
            }
        ]
    )
    promotion_queue = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-25",
                "generated_at": datetime(2026, 5, 25, tzinfo=UTC),
                "strategy_candidate": "v5.expanded_relative_strength_top1_shadow",
                "symbol": "UNKNOWN",
                "horizon_hours": None,
                "promotion_state": "KEEP_SHADOW",
                "recommended_mode": "shadow",
                "reasons": '["queue_kept_shadow"]',
            }
        ]
    )

    advisory = _strategy_opportunity_advisory_for_export(
        alpha_discovery_board=pl.DataFrame(),
        strategy_evidence=pl.DataFrame(),
        paper_proposals=pl.DataFrame(),
        risk_permissions=pl.DataFrame(),
        cost_health=pl.DataFrame(),
        paper_daily=pl.DataFrame(),
        paper_slippage=pl.DataFrame(),
        regime_strategy_advisory=regime_advisory,
        alpha_factory_promotion_queue=promotion_queue,
    ).to_dicts()

    assert len(advisory) == 1
    row = advisory[0]
    assert row["strategy_candidate"] == "regime_router:v5.expanded_relative_strength_top1_shadow"
    assert row["decision"] == "KEEP_SHADOW"
    assert row["recommended_mode"] == "shadow"
    assert row["would_enter"] is False
    assert row["max_paper_notional_usdt"] == 0.0
    assert "alpha_factory_promotion_queue_not_paper_ready" in row["live_block_reasons"]


def test_strategy_advisory_enriches_alpha_factory_score_from_results():
    board = pl.DataFrame(
        [
            _board_row(
                strategy_candidate="v5.expanded_relative_strength_top3_shadow",
                symbol="ONDO-USDT",
                source_type="alpha_factory",
                avg_net_bps=32.0,
                decision="KEEP_SHADOW",
                cost_source_mix='{"mixed_actual_proxy": 34}',
            )
        ]
    )
    alpha_factory_results = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-25",
                "generated_at": datetime(2026, 5, 25, tzinfo=UTC),
                "template_name": "expanded_relative_strength_v1",
                "candidate_id": "af-ondo-rs-top3-20260525",
                "strategy_candidate": "v5.expanded_relative_strength_top3_shadow",
                "symbol": "ONDO-USDT",
                "horizon_hours": 24,
                "alpha_factory_score": 74.5,
                "cost_quality_score": 0.67,
                "paper_ready_block_reasons": '["validation_not_paper_ready"]',
                "decision": "KEEP_SHADOW",
            }
        ]
    )
    promotion_queue = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-25",
                "generated_at": datetime(2026, 5, 25, tzinfo=UTC),
                "template_name": "expanded_relative_strength_v1",
                "candidate_id": "af-ondo-rs-top3-20260525",
                "strategy_candidate": "v5.expanded_relative_strength_top3_shadow",
                "symbol": "ONDO-USDT",
                "horizon_hours": 24,
                "promotion_state": "KEEP_SHADOW",
                "recommended_mode": "shadow",
                "reasons": '["validation_not_paper_ready"]',
            }
        ]
    )

    advisory = _strategy_opportunity_advisory_for_export(
        alpha_discovery_board=board,
        strategy_evidence=pl.DataFrame(),
        paper_proposals=pl.DataFrame(),
        risk_permissions=pl.DataFrame(),
        cost_health=pl.DataFrame(),
        paper_daily=pl.DataFrame(),
        paper_slippage=pl.DataFrame(),
        alpha_factory_results=alpha_factory_results,
        alpha_factory_promotion_queue=promotion_queue,
    ).to_dicts()

    assert len(advisory) == 1
    row = advisory[0]
    assert row["source_module"] == "alpha_factory"
    assert row["template_family"] == "expanded_relative_strength"
    assert row["candidate_id"] == "af-ondo-rs-top3-20260525"
    assert row["promotion_state"] == "KEEP_SHADOW"
    assert row["alpha_factory_score"] == 74.5
    assert row["cost_quality_score"] == 0.67
    assert "validation_not_paper_ready" in row["paper_ready_block_reasons"]


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
        candidate="v5.sol_protect_alpha6_low_exception",
        symbol="SOL-USDT",
        regime="protect",
        net_values=[26.0] * 72,
        cost_source="public_spread_proxy",
    )
    _add_labels(
        rows,
        start=start,
        candidate="v5.f4_volume_expansion_entry",
        symbol="SOL-USDT",
        regime="trend",
        net_values=[32.0] * 72,
        cost_source="public_spread_proxy",
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


def _board_row(
    *,
    strategy_candidate: str,
    symbol: str,
    source_type: str,
    avg_net_bps: float,
    decision: str,
    cost_source_mix: str,
) -> dict:
    return {
        "strategy": "v5",
        "board_schema_version": "alpha_discovery_board.v1",
        "as_of_date": "2026-05-10",
        "strategy_candidate": strategy_candidate,
        "candidate_name": strategy_candidate,
        "source_type": source_type,
        "symbol": symbol,
        "regime_state": "trend",
        "horizon_hours": 24,
        "sample_count": 72,
        "complete_sample_count": 72,
        "avg_net_bps": avg_net_bps,
        "median_net_bps": avg_net_bps,
        "p25_net_bps": 1.0,
        "win_rate": 0.72,
        "avg_mfe_bps": None,
        "avg_mae_bps": None,
        "cost_source_mix": cost_source_mix,
        "stability_by_day": "[]",
        "paper_days": 20,
        "cost_source_has_global_default": "global_default" in cost_source_mix,
        "decision": decision,
        "decision_reasons": "[]",
        "risk_permission": "ALLOW",
        "risk_permission_status": "ACTIVE_ALLOW",
        "enforce_readiness_status": "READY",
        "block_reason_mix": "{}",
        "final_decision_mix": "{}",
        "high_score_blocked_outcome_count": 0,
        "shadow_outcome_count": 0,
        "start_ts": datetime(2026, 5, 10, tzinfo=UTC),
        "end_ts": datetime(2026, 5, 10, tzinfo=UTC),
        "created_at": datetime(2026, 5, 10, tzinfo=UTC),
        "source": "test",
    }


def _downgraded_paper_daily_row(
    *,
    proposal_id: str,
    candidate: str,
    symbol: str,
) -> dict:
    return {
        "as_of_date": "2026-05-10",
        "proposal_id": proposal_id,
        "strategy_candidate": candidate,
        "symbol": symbol,
        "latest_board_decision": "PAPER_READY",
        "paper_negative_streak": 2,
        "latest_paper_trend": "negative",
        "live_block_reason": '["paper_negative_24h_or_48h_streak"]',
        "decision_reasons": "[]",
        "paper_days": 2,
        "entry_day_count": 2,
        "paper_pnl_observed_count": 2,
        "created_at": datetime(2026, 5, 10, 12, tzinfo=UTC),
    }


def _portfolio_override_row(
    strategy_candidate: str,
    status: str,
    *,
    as_of_date: str = "2026-05-10",
) -> dict:
    return {
        "schema_version": "research_portfolio_status.v0.1",
        "as_of_date": as_of_date,
        "research_id": strategy_candidate,
        "module": "alpha_factory",
        "strategy_candidate": strategy_candidate,
        "status": status,
        "action": f"{status}_ACTION",
        "reason": "test_portfolio_override",
        "created_at": datetime(2026, 5, 10, 13, tzinfo=UTC),
        "source": "test",
    }


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
