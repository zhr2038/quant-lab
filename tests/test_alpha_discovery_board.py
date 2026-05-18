import csv
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.alpha_discovery import (
    build_and_publish_alpha_discovery_board,
    normalize_alpha_discovery_board_decisions,
)
from quant_lab.research.paper_tracking import build_and_publish_paper_strategy_tracking


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
    ]

    normalized = normalize_alpha_discovery_board_decisions(pl.DataFrame(rows))
    output = {
        (row["strategy_candidate"], row["symbol"], row["source_type"]): row
        for row in normalized.to_dicts()
    }

    assert normalized.height == 2
    sol = output[("v5.f4_volume_expansion_entry", "SOL-USDT", "candidate_event_label")]
    assert sol["avg_net_bps"] == 30.0
    assert sol["decision"] == "PAPER_READY"
    bnb = output[("v5.swing_f4_f5_alpha6", "BNB-USDT", "candidate_event_label")]
    assert bnb["decision"] == "KEEP_SHADOW"


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
    assert set(daily["paper_days"].to_list()) == {1}
    assert set(daily["heartbeat_day_count"].to_list()) == {0}
    assert set(daily["entry_day_count"].to_list()) == {1}
    assert set(daily["would_enter_count"].to_list()) == {1}
    assert set(daily["paper_pnl_observed_count"].to_list()) == {1}
    assert set(daily["paper_tracking_status"].to_list()) == {"active"}
    assert set(daily["arrival_mid_coverage"].to_list()) == {1.0}
    assert set(daily["spread_observation_coverage"].to_list()) == {1.0}
    assert set(daily["live_eligible"].to_list()) == {False}
    assert set(slippage["paper_tracking_status"].to_list()) == {
        "active"
    }


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

    result = build_and_publish_paper_strategy_tracking(lake, as_of_date="2026-05-18")

    daily = read_parquet_dataset(lake / "gold" / "paper_strategy_daily")
    row = daily.to_dicts()[0]
    assert result.paper_strategy_daily == 1
    assert row["paper_days"] == 14
    assert row["entry_day_count"] == 14
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
    assert row["paper_days"] == 14
    assert row["entry_day_count"] == 14
    assert row["arrival_mid_coverage"] == 1.0
    assert row["live_eligible"] is True
    assert json.loads(row["live_block_reason"]) == []
    assert slippage["arrival_mid_coverage"].to_list() == [1.0]
    assert slippage["spread_observation_coverage"].to_list() == [1.0]


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
                    "experiment_name": candidate,
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
    assert {row["paper_tracking_status"] for row in daily} == {
        "active"
    }
    assert {row["heartbeat_day_count"] for row in daily} == {"1"}
    assert {row["entry_day_count"] for row in daily} == {"0"}
    assert {row["would_enter_count"] for row in daily} == {"0"}
    assert {row["paper_pnl_observed_count"] for row in daily} == {"0"}


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
