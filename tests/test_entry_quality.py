from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import app
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import (
    _late_entry_threshold_advisory_by_symbol_json,
    export_daily_pack,
)
from quant_lab.research.entry_quality import (
    _BuildContext,
    build_and_publish_entry_quality,
    build_and_publish_entry_quality_history,
    build_entry_quality_advisory,
)


def test_missed_low_audit_flags_late_chase_loss(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(lake, "BNB-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-1",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "BNB-USDT",
                    "side": "buy",
                    "action": "entry",
                    "price": 108.0,
                    "realized_net_bps": -60.0,
                    "exit_reason": "stop_loss",
                    "trade_id": "trade-1",
                }
            ]
        ),
        lake / "silver" / "v5_trade_event",
    )

    result = build_and_publish_entry_quality(lake, as_of_date="2026-05-10")

    assert result.missed_low_audit_rows == 1
    row = read_parquet_dataset(lake / "gold" / "v5_missed_low_audit").to_dicts()[0]
    assert row["diagnosis"] == "late_chase_loss"
    assert row["entry_vs_pre_24h_low_bps"] > 800
    by_symbol = read_parquet_dataset(lake / "gold" / "v5_missed_low_by_symbol").to_dicts()[0]
    assert by_symbol["late_chase_loss_count"] == 1


def test_late_entry_chase_shadow_counts_blocked_losses(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(lake, "SOL-USDT")
    candidate_ts = datetime(2026, 5, 10, 20, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand-sol-1",
                    "run_id": "run-1",
                    "ts_utc": candidate_ts,
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "entry_close": 108.0,
                    "f4_volume_expansion": 0.20,
                    "f5_rsi_trend_confirm": 0.10,
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand-sol-1",
                    "horizon_hours": 24,
                    "net_bps_after_cost": -75.0,
                }
            ]
        ),
        lake / "gold" / "v5_candidate_label",
    )

    build_and_publish_entry_quality(lake, as_of_date="2026-05-10")

    shadow = read_parquet_dataset(lake / "gold" / "v5_late_entry_chase_shadow")
    assert shadow.height == 1
    assert shadow.to_dicts()[0]["would_block_if_enabled"] is True
    threshold = read_parquet_dataset(
        lake / "gold" / "v5_late_entry_chase_threshold_advisory"
    )
    threshold_250 = threshold.filter(pl.col("threshold_bps") == 250).to_dicts()[0]
    assert threshold_250["would_block_loss_count"] == 1
    assert threshold_250["ready_for_live_guard"] is False
    by_symbol = read_parquet_dataset(
        lake / "gold" / "v5_late_entry_chase_threshold_by_symbol"
    )
    assert by_symbol.height == 24
    sol_100 = by_symbol.filter(
        (pl.col("symbol") == "SOL-USDT") & (pl.col("threshold_bps") == 100)
    ).to_dicts()[0]
    assert sol_100["would_block_loss_count"] == 1
    assert sol_100["ready_for_live_guard"] is False
    btc_100 = by_symbol.filter(
        (pl.col("symbol") == "BTC-USDT") & (pl.col("threshold_bps") == 100)
    ).to_dicts()[0]
    assert btc_100["would_block_count"] == 0


def test_late_entry_threshold_advisory_by_symbol_selects_shadow_thresholds():
    frame = pl.DataFrame(
        [
            {
                "symbol": "ETH-USDT",
                "threshold_bps": 100,
                "would_block_count": 10,
                "would_block_loss_count": 2,
                "would_block_profit_count": 8,
                "false_positive_rate": 0.8,
            },
            {
                "symbol": "ETH-USDT",
                "threshold_bps": 150,
                "would_block_count": 6,
                "would_block_loss_count": 2,
                "would_block_profit_count": 4,
                "false_positive_rate": 0.67,
            },
            {
                "symbol": "SOL-USDT",
                "threshold_bps": 150,
                "would_block_count": 5,
                "would_block_loss_count": 4,
                "would_block_profit_count": 1,
                "false_positive_rate": 0.2,
            },
            {
                "symbol": "SOL-USDT",
                "threshold_bps": 200,
                "would_block_count": 3,
                "would_block_loss_count": 3,
                "would_block_profit_count": 0,
                "false_positive_rate": 0.0,
            },
            {
                "symbol": "BTC-USDT",
                "threshold_bps": 100,
                "would_block_count": 2,
                "would_block_loss_count": 2,
                "would_block_profit_count": 0,
                "false_positive_rate": 0.0,
            },
            {
                "symbol": "BTC-USDT",
                "threshold_bps": 150,
                "would_block_count": 4,
                "would_block_loss_count": 2,
                "would_block_profit_count": 2,
                "false_positive_rate": 0.5,
            },
        ]
    )

    payload = _late_entry_threshold_advisory_by_symbol_json(frame)

    assert payload["ready_for_live_guard"] is False
    assert payload["hard_guard_allowed"] is False
    by_symbol = payload["by_symbol"]
    assert by_symbol["ETH-USDT"]["recommended_shadow_threshold_bps"] is None
    assert by_symbol["SOL-USDT"]["recommended_shadow_threshold_bps"] in {150, 200}
    assert by_symbol["SOL-USDT"]["ready_for_live_guard"] is False
    assert by_symbol["BTC-USDT"]["recommended_shadow_threshold_bps"] == 100
    assert by_symbol["BTC-USDT"]["block_loss_count"] == 2
    assert by_symbol["BTC-USDT"]["block_profit_count"] == 0


def test_pullback_reversal_shadow_outputs_positive_labels_without_live_ready(tmp_path):
    lake = tmp_path / "lake"
    _write_pullback_market_bars(lake, "ETH-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand-eth-pullback",
                    "run_id": "run-pullback",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "ETH-USDT",
                    "strategy_candidate": "portfolio",
                    "entry_close": 115.0,
                    "regime_state": "protect",
                    "risk_level": "normal",
                    "f4_volume_expansion": 0.0,
                    "f5_rsi_trend_confirm": 0.0,
                    "estimated_spread_bps": 2.0,
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )

    build_and_publish_entry_quality(lake, as_of_date="2026-05-10")

    shadow = read_parquet_dataset(lake / "gold" / "v5_pullback_reversal_shadow")
    rows_24h = shadow.filter(pl.col("horizon_hours") == 24).to_dicts()
    assert rows_24h
    assert rows_24h[0]["rule_version"] == "confirmed_reversal_v0.2"
    assert rows_24h[0]["close_reclaim_1h"] is True
    assert rows_24h[0]["current_close_gt_previous_close"] is True
    assert rows_24h[0]["btc_not_sharp_drop"] is True
    assert rows_24h[0]["spread_not_abnormal"] is True
    assert rows_24h[0]["net_bps_after_cost"] > 0
    comparison = read_parquet_dataset(
        lake / "gold" / "v5_pullback_reversal_rule_comparison"
    )
    assert {"old_rule", "new_rule"} <= set(comparison.get_column("rule_name").to_list())
    readiness = read_parquet_dataset(lake / "gold" / "v5_pullback_reversal_readiness")
    row = readiness.to_dicts()[0]
    assert row["ready_for_live_probe"] is False
    assert "insufficient_sample_count" in row["readiness_reasons"]


def test_daily_export_contains_entry_quality_reports(tmp_path):
    lake = tmp_path / "lake"
    out = tmp_path / "exports"
    _write_market_bars(lake, "BNB-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-1",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "BNB-USDT",
                    "side": "buy",
                    "action": "entry",
                    "price": 108.0,
                    "realized_net_bps": -60.0,
                    "trade_id": "trade-1",
                }
            ]
        ),
        lake / "silver" / "v5_trade_event",
    )
    build_and_publish_entry_quality(lake, as_of_date="2026-05-10")

    result = export_daily_pack(
        export_date="2026-05-10",
        lake_root=lake,
        out_dir=out,
        command_line=["qlab", "export-daily"],
        refresh_risk_permission=False,
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/missed_low_audit.csv" in names
        assert "reports/late_entry_chase_threshold_advisory.json" in names
        assert "reports/late_entry_chase_threshold_sensitivity_by_symbol.csv" in names
        assert "reports/late_entry_chase_threshold_advisory_by_symbol.json" in names
        assert "reports/threshold_advisory_by_symbol.json" in names
        assert "reports/pullback_reversal_rule_comparison.csv" in names
        assert "reports/pullback_reversal_readiness.json" in names
        assert "reports/old_v1_vs_v2_comparison.csv" in names
        assert "reports/pullback_reversal_v2_by_symbol.csv" in names
        assert "reports/pullback_reversal_v2_readiness.json" in names
        summary = archive.read("reports/entry_quality_summary.md").decode("utf-8")
        assert "read-only research" in summary
        advisory = archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
        assert "v5.entry_quality_missed_low_audit" in advisory
        advisory_rows = list(csv.DictReader(io.StringIO(advisory)))
        entry_row = next(
            row
            for row in advisory_rows
            if row["strategy_candidate"] == "v5.entry_quality_missed_low_audit"
        )
        assert entry_row["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
        assert entry_row["quant_lab_git_commit"] != "not_observable"
        assert entry_row["source_version"].startswith("entry_quality:")
        assert entry_row["would_block_if_enabled"] in {"False", "false", "0"}
        assert entry_row["would_enter"] in {"False", "false", "0"}
        assert entry_row["no_sample_reason"] == "audit_only"
        threshold = json.loads(
            archive.read("reports/late_entry_chase_threshold_advisory.json")
        )
        assert threshold["source"] == "quant_lab"
        by_symbol = json.loads(archive.read("reports/threshold_advisory_by_symbol.json"))
        assert by_symbol["ready_for_live_guard"] is False
        assert by_symbol["thresholds_bps"] == [50, 100, 150, 200, 250, 300]
        symbol_advisory = json.loads(
            archive.read("reports/late_entry_chase_threshold_advisory_by_symbol.json")
        )
        assert symbol_advisory["hard_guard_allowed"] is False
        assert symbol_advisory["ready_for_live_guard"] is False


def test_daily_export_contains_pullback_reversal_v2_reports(tmp_path):
    lake = tmp_path / "lake"
    out = tmp_path / "exports"
    _write_pullback_market_bars(lake, "ETH-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand-eth-pullback",
                    "run_id": "run-pullback",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "ETH-USDT",
                    "strategy_candidate": "portfolio",
                    "entry_close": 115.0,
                    "regime_state": "protect",
                    "risk_level": "normal",
                    "f4_volume_expansion": 0.0,
                    "f5_rsi_trend_confirm": 0.0,
                    "estimated_spread_bps": 2.0,
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )
    build_and_publish_entry_quality(lake, as_of_date="2026-05-10")

    result = export_daily_pack(
        export_date="2026-05-10",
        lake_root=lake,
        out_dir=out,
        command_line=["qlab", "export-daily"],
        refresh_risk_permission=False,
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        comparison = archive.read("reports/old_v1_vs_v2_comparison.csv").decode("utf-8")
        comparison_rows = list(csv.DictReader(io.StringIO(comparison)))
        assert comparison_rows
        assert {row["comparison_name"] for row in comparison_rows} == {
            "old_rule_vs_new_rule"
        }

        by_symbol = archive.read("reports/pullback_reversal_v2_by_symbol.csv").decode(
            "utf-8"
        )
        rows = list(csv.DictReader(io.StringIO(by_symbol)))
        assert len(rows) == 1
        row = rows[0]
        assert row["rule_version"] == "confirmed_reversal_v0.2"
        assert row["symbol"] == "ETH-USDT"
        assert row["decision"] == "RESEARCH_ONLY"
        reasons = set(json.loads(row["decision_reasons"]))
        assert {"not_live_validated", "paper_disabled_until_more_evidence"} <= reasons
        assert "insufficient_sample_count" in reasons

        readiness = json.loads(
            archive.read("reports/pullback_reversal_v2_readiness.json")
        )
        assert readiness["source"] == "quant_lab"
        assert readiness["mode"] == "advisory"
        assert readiness["row_count"] == 1
        assert readiness["rows"][0]["ready_for_live_probe"] is False


def test_entry_quality_publishes_strategy_opportunity_advisory_for_api(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    _write_market_bars(lake, "BNB-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-1",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "BNB-USDT",
                    "side": "buy",
                    "action": "entry",
                    "price": 108.0,
                    "realized_net_bps": -60.0,
                    "exit_reason": "stop_loss",
                    "trade_id": "trade-1",
                }
            ]
        ),
        lake / "silver" / "v5_trade_event",
    )

    result = build_and_publish_entry_quality(lake, as_of_date="2026-05-10")

    assert result.strategy_opportunity_advisory_rows >= 1
    gold_rows = read_parquet_dataset(
        lake / "gold" / "strategy_opportunity_advisory"
    ).to_dicts()
    missed = next(
        row
        for row in gold_rows
        if row["strategy_candidate"] == "v5.entry_quality_missed_low_audit"
    )
    pullback_candidates = {
        row["strategy_candidate"]
        for row in gold_rows
        if str(row.get("strategy_candidate") or "").startswith("v5.pullback_reversal_shadow_")
    }
    assert pullback_candidates == {
        "v5.pullback_reversal_shadow_bnb",
        "v5.pullback_reversal_shadow_btc",
        "v5.pullback_reversal_shadow_eth",
        "v5.pullback_reversal_shadow_sol",
    }
    pullback_btc = next(
        row
        for row in gold_rows
        if row["strategy_candidate"] == "v5.pullback_reversal_shadow_btc"
    )
    assert pullback_btc["recommended_mode"] == "research"
    assert pullback_btc["would_enter"] is False
    assert pullback_btc["would_block_if_enabled"] is False
    assert pullback_btc["no_sample_reason"] == "insufficient_sample_count"
    assert pullback_btc["max_live_notional_usdt"] == 0.0
    assert missed["recommended_mode"] == "research"
    assert missed["as_of_ts"] is not None
    assert missed["generated_at"] is not None
    assert missed["expires_at"] is not None
    assert missed["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
    assert missed["schema_version"] == "strategy_opportunity_advisory.v0.1"
    assert missed["source_version"]
    assert missed["would_block_if_enabled"] is False
    assert missed["would_enter"] is False
    assert missed["no_sample_reason"] == "audit_only"
    assert missed["max_live_notional_usdt"] == 0.0
    assert "shadow_only" in missed["live_block_reasons"]
    assert "not_live_validated" in missed["live_block_reasons"]

    response = TestClient(app).get("/v1/strategy-opportunity-advisory")
    assert response.status_code == 200
    api_row = response.json()[0]
    assert api_row["strategy_candidate"] == "v5.entry_quality_missed_low_audit"
    assert api_row["recommended_mode"] == "research"
    assert api_row["as_of_ts"]
    assert api_row["generated_at"]
    assert api_row["expires_at"]
    assert api_row["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
    assert api_row["source_version"]
    assert api_row["would_block_if_enabled"] is False
    assert api_row["would_enter"] is False
    assert api_row["no_sample_reason"] == "audit_only"
    assert api_row["max_live_notional_usdt"] == 0.0
    assert {"shadow_only", "not_live_validated"} <= set(api_row["live_block_reasons"])


def test_entry_quality_history_outputs_threshold_sensitivity_and_reports(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(lake, "SOL-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-1",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "SOL-USDT",
                    "side": "buy",
                    "action": "entry",
                    "price": 108.0,
                    "realized_net_bps": -80.0,
                    "trade_id": "trade-sol-1",
                }
            ]
        ),
        lake / "silver" / "v5_trade_event",
    )

    result = build_and_publish_entry_quality_history(
        lake,
        start_date="2026-05-10",
        end_date="2026-05-10",
        mode="full",
        cost_mode="conservative",
    )

    assert result.late_entry_threshold_sensitivity_rows == 6
    threshold = read_parquet_dataset(
        lake / "gold" / "v5_entry_quality_history_late_entry_chase_threshold_sensitivity"
    )
    row = threshold.filter(pl.col("threshold_bps") == 250).to_dicts()[0]
    assert row["would_block_loss_count"] == 1
    assert row["avg_net_bps_blocked"] == -80.0
    assert row["ready_for_live_guard"] is False
    assert (lake / "reports" / "late_entry_chase_threshold_sensitivity.csv").exists()
    assert (lake / "reports" / "entry_quality_historical_metrics.json").exists()


def test_entry_quality_history_recent_7d_filters_old_entries(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars_for_range(lake, "BNB-USDT", datetime(2026, 5, 1, tzinfo=UTC), 15 * 24)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "old",
                    "ts_utc": datetime(2026, 5, 2, 20, tzinfo=UTC),
                    "symbol": "BNB-USDT",
                    "side": "buy",
                    "action": "entry",
                    "price": 108.0,
                    "trade_id": "old-trade",
                },
                {
                    "run_id": "recent",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "BNB-USDT",
                    "side": "buy",
                    "action": "entry",
                    "price": 108.0,
                    "trade_id": "recent-trade",
                },
            ]
        ),
        lake / "silver" / "v5_trade_event",
    )

    build_and_publish_entry_quality_history(
        lake,
        start_date="2026-05-01",
        end_date="2026-05-10",
        mode="recent_7d",
        cost_mode="conservative",
    )

    missed = read_parquet_dataset(
        lake / "gold" / "v5_entry_quality_history_missed_low_audit"
    )
    assert missed.height == 1
    assert missed.to_dicts()[0]["run_id"] == "recent"


def test_entry_quality_history_pullback_by_symbol_and_anti_leakage(tmp_path):
    lake = tmp_path / "lake"
    _write_pullback_market_bars(lake, "ETH-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand-eth-pullback",
                    "run_id": "run-pullback",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "ETH-USDT",
                    "strategy_candidate": "portfolio",
                    "entry_close": 115.0,
                    "regime_state": "protect",
                    "risk_level": "normal",
                    "f4_volume_expansion": 0.0,
                    "f5_rsi_trend_confirm": 0.0,
                    "estimated_spread_bps": 2.0,
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )

    result = build_and_publish_entry_quality_history(
        lake,
        start_date="2026-05-10",
        end_date="2026-05-10",
        mode="walk_forward",
        cost_mode="conservative",
    )

    assert result.pullback_by_symbol_rows == 1
    by_symbol = read_parquet_dataset(
        lake / "gold" / "v5_entry_quality_history_pullback_by_symbol"
    ).to_dicts()[0]
    assert by_symbol["symbol"] == "ETH-USDT"
    assert by_symbol["decision"] in {"RESEARCH_ONLY", "KEEP_SHADOW"}
    assert by_symbol["decision"] != "LIVE_SMALL_READY"
    checks = read_parquet_dataset(
        lake / "gold" / "v5_entry_quality_history_anti_leakage_check"
    )
    assert set(checks.get_column("status").to_list()) == {"PASS"}


def test_entry_quality_history_empty_current_window_clears_stale_pullback(tmp_path):
    lake = tmp_path / "lake"
    _write_pullback_market_bars(lake, "ETH-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand-eth-pullback",
                    "run_id": "run-pullback",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "ETH-USDT",
                    "strategy_candidate": "portfolio",
                    "entry_close": 115.0,
                    "regime_state": "protect",
                    "risk_level": "normal",
                    "f4_volume_expansion": 0.0,
                    "f5_rsi_trend_confirm": 0.0,
                    "estimated_spread_bps": 2.0,
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )
    build_and_publish_entry_quality_history(
        lake,
        start_date="2026-05-10",
        end_date="2026-05-10",
        mode="full",
        cost_mode="conservative",
    )
    assert not read_parquet_dataset(
        lake / "gold" / "v5_entry_quality_history_pullback_by_symbol"
    ).is_empty()

    write_parquet_dataset(pl.DataFrame(), lake / "silver" / "v5_candidate_event")
    result = build_and_publish_entry_quality_history(
        lake,
        start_date="2026-05-11",
        end_date="2026-05-11",
        mode="full",
        cost_mode="conservative",
    )

    assert result.pullback_by_symbol_rows == 0
    assert read_parquet_dataset(
        lake / "gold" / "v5_entry_quality_history_pullback_by_symbol"
    ).is_empty()


def test_pullback_new_rule_rejects_falling_knife_candidate(tmp_path):
    lake = tmp_path / "lake"
    _write_pullback_market_bars(lake, "SOL-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand-sol-good",
                    "run_id": "run-good",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "portfolio",
                    "entry_close": 115.0,
                    "regime_state": "protect",
                    "risk_level": "normal",
                    "f4_volume_expansion": 0.0,
                    "f5_rsi_trend_confirm": 0.0,
                    "estimated_spread_bps": 2.0,
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )
    assert (
        build_and_publish_entry_quality(lake, as_of_date="2026-05-10")
        .pullback_reversal_shadow_rows
        > 0
    )

    _write_falling_pullback_market_bars(lake, "SOL-USDT")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand-sol-falling",
                    "run_id": "run-falling",
                    "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "portfolio",
                    "entry_close": 112.5,
                    "regime_state": "protect",
                    "risk_level": "normal",
                    "f4_volume_expansion": 0.0,
                    "f5_rsi_trend_confirm": 0.0,
                    "estimated_spread_bps": 2.0,
                }
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )

    result = build_and_publish_entry_quality(lake, as_of_date="2026-05-10")

    assert result.pullback_reversal_shadow_rows == 0
    assert read_parquet_dataset(lake / "gold" / "v5_pullback_reversal_shadow").height == 0
    opportunities = read_parquet_dataset(lake / "gold" / "strategy_opportunity_advisory")
    pullback_rows = [
        row
        for row in opportunities.to_dicts()
        if str(row.get("strategy_candidate") or "").startswith(
            "v5.pullback_reversal_shadow_"
        )
    ]
    assert len(pullback_rows) == 4
    assert all(row["recommended_mode"] == "research" for row in pullback_rows)
    assert all(row["would_enter"] is False for row in pullback_rows)
    assert all(row["max_live_notional_usdt"] == 0.0 for row in pullback_rows)
    assert {row["no_sample_reason"] for row in pullback_rows} == {
        "insufficient_sample_count"
    }
    comparison = read_parquet_dataset(
        lake / "gold" / "v5_pullback_reversal_rule_comparison"
    )
    rows = comparison.to_dicts()
    assert any(
        row["rule_name"] == "old_rule" and row["symbol"] == "SOL-USDT" for row in rows
    )
    assert not any(
        row["rule_name"] == "new_rule" and row["symbol"] == "SOL-USDT" for row in rows
    )


def test_pullback_bad_mae_and_negative_edge_stays_research_only():
    ctx = _BuildContext(
        as_of_date=datetime(2026, 5, 10, tzinfo=UTC).date(),
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        generated_from_bundle_id="bundle-1",
        window_hours=24,
    )
    readiness = pl.DataFrame(
        [
            {
                "strategy_candidate": "v5.pullback_reversal_shadow_eth",
                "symbol": "ETH-USDT",
                "sample_count": 60,
                "recent_7d_sample_count": 12,
                "avg_24h_net_bps": -40.0,
                "win_rate_24h": 0.30,
                "p25_24h_net_bps": -90.0,
                "avg_mae_bps": -180.0,
                "ready_for_paper": False,
                "ready_for_live_probe": False,
                "readiness_reasons": (
                    '["weak_24h_avg_net_bps","weak_24h_p25_net_bps",'
                    '"excessive_avg_mae_bps"]'
                ),
            }
        ]
    )

    advisory = build_entry_quality_advisory(
        missed_low=pl.DataFrame(),
        late_threshold=pl.DataFrame(),
        pullback_readiness=readiness,
        ctx=ctx,
    )
    row = next(
        item
        for item in advisory.to_dicts()
        if item["strategy_candidate"] == "v5.pullback_reversal_shadow_eth"
    )

    assert row["recommended_mode"] == "research"
    assert row["would_enter"] is False
    assert row["no_sample_reason"] == "weak_24h_avg_net_bps"


def test_pullback_v2_does_not_promote_to_paper_even_when_metrics_pass():
    ctx = _BuildContext(
        as_of_date=datetime(2026, 5, 10, tzinfo=UTC).date(),
        generated_at=datetime(2026, 5, 10, tzinfo=UTC),
        generated_from_bundle_id="bundle-1",
        window_hours=24,
    )
    readiness = pl.DataFrame(
        [
            {
                "strategy_candidate": "v5.pullback_reversal_shadow_sol",
                "symbol": "SOL-USDT",
                "sample_count": 80,
                "recent_7d_sample_count": 20,
                "avg_24h_net_bps": 75.0,
                "win_rate_24h": 0.62,
                "p25_24h_net_bps": -20.0,
                "avg_mae_bps": -70.0,
                "ready_for_paper": True,
                "ready_for_live_probe": False,
                "readiness_reasons": '["legacy_ready_for_paper"]',
            }
        ]
    )

    advisory = build_entry_quality_advisory(
        missed_low=pl.DataFrame(),
        late_threshold=pl.DataFrame(),
        pullback_readiness=readiness,
        ctx=ctx,
    )
    row = next(
        item
        for item in advisory.to_dicts()
        if item["strategy_candidate"] == "v5.pullback_reversal_shadow_sol"
    )

    assert row["recommended_mode"] == "shadow"
    assert row["readiness_status"] == "SHADOW_ONLY"
    assert row["would_enter"] is False
    assert row["ready_for_live"] is False


def _write_market_bars(lake, symbol: str) -> None:
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rows = []
    for hour in range(30):
        rows.append(
            {
                "venue": "okx",
                "symbol": symbol,
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": start + timedelta(hours=hour),
                "open": 100.0,
                "high": 110.0,
                "low": 100.0 if hour > 0 else 90.0,
                "close": 100.0,
                "volume": 1000.0,
                "quote_volume": 100000.0,
                "source": "test",
                "ingest_ts": start,
                "is_closed": True,
            }
        )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "market_bar")


def _write_market_bars_for_range(lake, symbol: str, start: datetime, hours: int) -> None:
    rows = []
    for hour in range(hours):
        rows.append(
            {
                "venue": "okx",
                "symbol": symbol,
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": start + timedelta(hours=hour),
                "open": 100.0,
                "high": 110.0,
                "low": 100.0 if hour > 0 else 90.0,
                "close": 100.0,
                "volume": 1000.0,
                "quote_volume": 100000.0,
                "source": "test",
                "ingest_ts": start,
                "is_closed": True,
            }
        )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "market_bar")


def _write_pullback_market_bars(lake, symbol: str) -> None:
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rows = []
    symbols = [symbol] if symbol == "BTC-USDT" else [symbol, "BTC-USDT"]
    for current_symbol in symbols:
        for hour in range(80):
            ts = start + timedelta(hours=hour)
            if current_symbol == "BTC-USDT" and symbol != "BTC-USDT":
                high, low, close = 101.0, 99.0, 100.0 + min(hour, 20) * 0.01
            elif hour < 18:
                high, low, close = 120.0, 108.0, 118.0
            elif hour == 18:
                high, low, close = 118.0, 112.0, 116.0
            elif hour == 19:
                high, low, close = 116.0, 112.0, 114.0
            elif hour == 20:
                high, low, close = 116.0, 112.0, 115.0
            else:
                high, low, close = 126.0, 114.0, 125.0
            rows.append(
                {
                    "venue": "okx",
                    "symbol": current_symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": ts,
                    "open": close,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                    "quote_volume": 100000.0,
                    "source": "test",
                    "ingest_ts": start,
                    "is_closed": True,
                }
            )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "market_bar")


def _write_falling_pullback_market_bars(lake, symbol: str) -> None:
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rows = []
    symbols = [symbol] if symbol == "BTC-USDT" else [symbol, "BTC-USDT"]
    for current_symbol in symbols:
        for hour in range(80):
            ts = start + timedelta(hours=hour)
            if current_symbol == "BTC-USDT" and symbol != "BTC-USDT":
                high, low, close = 101.0, 99.0, 100.0 + min(hour, 20) * 0.01
            elif hour < 18:
                high, low, close = 116.0, 108.0, 115.0
            elif hour < 20:
                high, low, close = 115.0, 110.0, 114.0
            elif hour == 20:
                high, low, close = 114.0, 110.0, 112.5
            else:
                high, low, close = 113.0, 107.0, 108.0
            rows.append(
                {
                    "venue": "okx",
                    "symbol": current_symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": ts,
                    "open": close,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                    "quote_volume": 100000.0,
                    "source": "test",
                    "ingest_ts": start,
                    "is_closed": True,
                }
            )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "market_bar")
