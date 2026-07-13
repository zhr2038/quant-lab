import csv
import gzip
import json
import zipfile
from io import StringIO

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.paper_promotion import build_and_publish_paper_strategy_pipeline
from quant_lab.research.paper_tracking import (
    build_and_publish_paper_strategy_tracking,
    build_paper_strategy_runs_from_v5,
)
from quant_lab.strategy_telemetry.ingest import (
    _event_key_fields,
    _event_key_from_fields,
    ingest_v5_bundle,
)
from tests.v5_bundle_fixture import make_tar, make_v5_bundle_fixture


def test_ingest_bundle_idempotent_by_sha256(tmp_path):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz")
    lake = tmp_path / "lake"

    first = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")
    second = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    manifests = read_parquet_dataset(lake / "bronze/strategy_telemetry/v5/bundle_manifest")
    decisions = read_parquet_dataset(lake / "silver/v5_decision_audit")

    assert first.skipped is False
    assert second.skipped is True
    assert manifests.height == 1
    assert decisions.height == 1


def test_ingest_parses_window_summary(tmp_path):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz")
    lake = tmp_path / "lake"

    ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    health = read_parquet_dataset(lake / "gold/strategy_health_daily")
    assert health.height == 1
    assert health["run_count_72h"][0] >= 1


def test_ingest_parses_state_files(tmp_path):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz")
    lake = tmp_path / "lake"

    ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    states = read_parquet_dataset(lake / "silver/v5_state_snapshot")
    assert set(states["state_type"].to_list()) >= {
        "kill_switch",
        "reconcile_status",
        "ledger_status",
        "auto_risk_eval",
    }


def test_sync_ingest_can_skip_large_historical_outcomes(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {
            "summaries/candidate_snapshot.csv": (
                "candidate_id,run_id,ts_utc,symbol,strategy_candidate,cost_source,"
                "quote_ts,quote_age_ms,quote_source\n"
                "cand_1,run_001,2026-05-10T01:00:00Z,SOL/USDT,"
                "f4_volume_expansion_entry,public_spread_proxy,"
                "2026-05-10T01:00:01Z,240,okx_books5\n"
            ),
            "raw/reports/quant_lab_requests.jsonl": (
                '{"ts":"2026-05-10T01:00:00Z","endpoint":"/v1/costs/estimate",'
                '"status_code":200,"success":true}\n'
            ),
            "summaries/high_score_blocked_outcomes.csv": (
                "candidate_id,symbol,label_4h_net_bps,label_status\nhist_1,SOL-USDT,12,complete\n"
            ),
            "summaries/alt_impulse_shadow_outcomes.csv": (
                "candidate_id,symbol,label_4h_net_bps,label_status\nshadow_1,SOL-USDT,10,complete\n"
            ),
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(
        bundle,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        run_analysis=False,
        refresh_candidate_gold=False,
        include_historical_outcomes=False,
    )

    candidate_events = read_parquet_dataset(lake / "silver/v5_candidate_event")
    requests = read_parquet_dataset(lake / "silver/v5_quant_lab_request")
    historical = read_parquet_dataset(lake / "silver/v5_high_score_blocked_outcome")
    shadow = read_parquet_dataset(lake / "silver/v5_shadow_outcome")
    assert candidate_events.height == 1
    candidate = candidate_events.to_dicts()[0]
    assert candidate["quote_ts"] == "2026-05-10T01:00:01Z"
    assert candidate["quote_age_ms"] == "240"
    assert candidate["quote_source"] == "okx_books5"
    assert requests.height == 1
    assert historical.is_empty()
    assert shadow.is_empty()
    assert any("skipped_historical_outcome_file" in warning for warning in result.warnings)
    redacted_root = tmp_path / "redacted" / "2026-05-10" / result.bundle_sha256 / "redacted_files"
    assert not (redacted_root / "summaries/high_score_blocked_outcomes.csv").exists()
    assert not (redacted_root / "summaries/alt_impulse_shadow_outcomes.csv").exists()


def test_ingest_routes_alt_impulse_readiness_json_to_state_snapshot(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {
            "summaries/alt_impulse_shadow_readiness.json": json.dumps(
                {
                    "schema_version": "v5.alt_impulse_shadow_readiness.v1",
                    "generated_ts_utc": "2026-05-10T01:00:00Z",
                    "ready_for_live_probe": False,
                    "blocking_reasons": ["no_symbol_ready_for_live_probe"],
                    "by_symbol": [
                        {
                            "symbol": "SOL/USDT",
                            "ready_for_live_probe": False,
                            "bnb_negative_expectancy_bps": None,
                        }
                    ],
                }
            )
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(
        bundle,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        run_analysis=False,
        refresh_candidate_gold=False,
    )

    states = read_parquet_dataset(lake / "silver/v5_state_snapshot")
    row = states.to_dicts()[0]
    payload = json.loads(row["raw_payload_json"])
    assert not any("failed to parse" in warning for warning in result.warnings)
    assert row["state_type"] == "alt_impulse_shadow_readiness"
    assert row["ok"] is False
    assert row["level"] == "BLOCKED"
    assert payload["by_symbol"][0]["bnb_negative_expectancy_bps"] is None


def test_csv_rows_normalize_extra_field_key_instead_of_parse_warning(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {
            "summaries/alt_impulse_shadow_outcomes.csv": (
                "candidate_id,symbol\nshadow_1,SOL-USDT,unexpected-extra-field\n"
            )
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(
        bundle,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        run_analysis=False,
        refresh_candidate_gold=False,
    )

    shadow = read_parquet_dataset(lake / "silver/v5_shadow_outcome")
    payload = json.loads(shadow.to_dicts()[0]["raw_payload_json"])
    assert not any("failed to parse" in warning for warning in result.warnings)
    assert payload["_extra_fields"] == ["unexpected-extra-field"]


def test_ingest_v5_trades_csv_normalizes_cost_schema(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T070500Z.tar.gz",
        {
            "raw/recent_runs/run_20260512_06/trades.csv": (
                "ts,symbol,side,action,qty,price,fee,fee_ccy,order_id,trade_id\n"
                "2026-05-12T06:01:00Z,BNB/USDT,buy,entry,0.5,620,-0.031,USDT,"
                "bnb-order-1,bnb-trade-1\n"
            ),
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    trades = read_parquet_dataset(lake / "silver/v5_trade_event").to_dicts()
    assert len(trades) == 1
    row = trades[0]
    assert row["run_id"] == "run_20260512_06"
    assert row["ts_utc"] == "2026-05-12T06:01:00Z"
    assert row["symbol"] == "BNB-USDT"
    assert row["normalized_symbol"] == "BNB-USDT"
    assert row["side"] == "buy"
    assert row["action"] == "entry"
    assert float(row["notional_usdt"]) == 310.0
    assert float(row["fee_usdt"]) == 0.031
    assert row["order_id"] == "bnb-order-1"
    assert row["trade_id"] == "bnb-trade-1"
    assert row["strategy_id"] == "v5"


def test_ingest_v5_trade_events_dedupes_overlapping_bundles(tmp_path):
    trade_csv = (
        "ts,symbol,side,action,qty,price,fee,fee_ccy,order_id,trade_id\n"
        "2026-05-12T06:01:00Z,BNB/USDT,buy,entry,0.5,620,-0.031,USDT,"
        "bnb-order-1,bnb-trade-1\n"
    )
    first = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T070500Z.tar.gz",
        {
            "raw/recent_runs/run_20260512_06/trades.csv": trade_csv,
            "raw/recent_runs/run_20260512_06/summary.json": '{"bundle_marker":"first"}',
        },
    )
    second = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T071500Z.tar.gz",
        {
            "raw/recent_runs/run_20260512_06/trades.csv": trade_csv,
            "raw/recent_runs/run_20260512_06/summary.json": '{"bundle_marker":"second"}',
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(first, lake, tmp_path / "restricted", tmp_path / "redacted")
    ingest_v5_bundle(second, lake, tmp_path / "restricted", tmp_path / "redacted")

    trades = read_parquet_dataset(lake / "silver/v5_trade_event")
    assert trades.height == 1
    row = trades.to_dicts()[0]
    assert row["trade_id"] == "bnb-trade-1"
    assert row["bundle_name"] == second.name
    assert row["stable_row_key"]


def test_ingest_exports_v5_pullback_reversal_artifacts(tmp_path):
    pullback_csv = (
        "as_of_date,rule_version,strategy_candidate,run_id,candidate_id,"
        "source_event_key,symbol,ts_utc,regime_state,risk_level,current_px,"
        "pre_24h_low,pre_24h_high,pullback_from_24h_high_bps,"
        "recent_2h_no_new_low,close_reclaim_1h,current_close_gt_previous_close,"
        "btc_not_sharp_drop,spread_not_abnormal,f4_volume_expansion,"
        "f5_rsi_trend_confirm,selected_roundtrip_cost_bps,cost_quality,"
        "horizon_hours,gross_bps,net_bps_after_cost,mfe_bps,mae_bps,win,"
        "label_status,mode,generated_at_utc,schema_version,contract_version\n"
        "2026-05-26,confirmed_reversal_v0.2,v5.pullback_reversal_shadow_bnb,"
        "run_1,cand_pb_1,key_1,BNB-USDT,2026-05-26T11:00:00Z,Trending,"
        "PROTECT,650,640,670,120,true,true,true,true,true,0.2,0.1,30,"
        "mixed_actual_proxy,24,80,50,110,-60,true,complete,shadow,"
        "2026-05-26T11:05:00Z,entry_quality.v0.1,v5.quant_lab.telemetry.v2\n"
    )
    readiness_json = json.dumps(
        {
            "row_count": 4,
            "rows": [
                {
                    "strategy_candidate": "v5.pullback_reversal_shadow_bnb",
                    "symbol": "BNB-USDT",
                    "sample_count": 1,
                    "ready_for_paper": False,
                }
            ],
        }
    )
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260526T110000Z.tar.gz",
        {
            "reports/pullback_reversal_shadow_outcomes.csv": pullback_csv,
            "reports/pullback_reversal_readiness.json": readiness_json,
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(
        bundle,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        run_analysis=False,
        refresh_candidate_gold=False,
    )
    assert result.silver_rows["v5_pullback_reversal_shadow"] == 1
    assert result.silver_rows["v5_pullback_reversal_readiness"] == 1

    export = export_daily_pack(
        export_date="2026-05-26",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily", "--no-pre-export-v5-refresh"],
        pre_export_v5_refresh=False,
    )
    with zipfile.ZipFile(export.zip_path) as archive:
        shadow_rows = list(
            csv.DictReader(
                StringIO(
                    archive.read("reports/pullback_reversal_shadow_outcomes.csv").decode("utf-8")
                )
            )
        )
        by_symbol_rows = list(
            csv.DictReader(
                StringIO(archive.read("reports/pullback_reversal_by_symbol.csv").decode("utf-8"))
            )
        )
        readiness = json.loads(archive.read("reports/pullback_reversal_readiness.json"))
        manifest = json.loads(archive.read("manifest.json"))

    assert shadow_rows
    assert shadow_rows[0]["symbol"] == "BNB-USDT"
    assert by_symbol_rows
    assert by_symbol_rows[0]["group_key"] == "BNB-USDT"
    assert readiness["row_count"] == 4
    files = {item["path"]: item for item in manifest["files"]}
    assert files["reports/pullback_reversal_shadow_outcomes.csv"]["rows"] == 1


def test_pullback_reversal_shadow_uses_semantic_dedupe_across_reports_and_summaries(
    tmp_path,
):
    header = (
        "as_of_date,rule_version,strategy_candidate,run_id,candidate_id,"
        "source_event_key,symbol,ts_utc,horizon_hours,net_bps_after_cost,"
        "label_status,generated_at_utc,schema_version,contract_version\n"
    )
    reports_csv = (
        header + "2026-05-26,confirmed_reversal_v0.2,v5.pullback_reversal_shadow_bnb,"
        "run_1,cand_pb_1,key_1,BNB-USDT,2026-05-26T11:00:00Z,24,50,"
        "complete,2026-05-26T11:05:00Z,entry_quality.v0.1,"
        "v5.quant_lab.telemetry.v2\n"
    )
    summaries_csv = (
        header + "2026-05-26,confirmed_reversal_v0.2,v5.pullback_reversal_shadow_bnb,"
        "run_1,cand_pb_1,key_1,BNB-USDT,2026-05-26T11:00:00Z,24,50,"
        "complete,2026-05-26T12:05:00Z,entry_quality.v0.1,"
        "v5.quant_lab.telemetry.v2\n"
        + "2026-05-26,confirmed_reversal_v0.2,v5.pullback_reversal_shadow_bnb,"
        "run_1,cand_pb_1,key_1,BNB-USDT,2026-05-26T11:00:00Z,48,70,"
        "complete,2026-05-26T12:05:00Z,entry_quality.v0.1,"
        "v5.quant_lab.telemetry.v2\n"
        + "2026-05-26,confirmed_reversal_v0.2,v5.pullback_reversal_shadow_sol,"
        "run_1,cand_pb_2,key_2,SOL-USDT,2026-05-26T11:00:00Z,24,30,"
        "complete,2026-05-26T12:05:00Z,entry_quality.v0.1,"
        "v5.quant_lab.telemetry.v2\n"
    )
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260526T110000Z.tar.gz",
        {
            "reports/pullback_reversal_shadow_outcomes.csv": reports_csv,
            "summaries/pullback_reversal_shadow_outcomes.csv": summaries_csv,
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(
        bundle,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        run_analysis=False,
        refresh_candidate_gold=False,
    )

    rows = read_parquet_dataset(lake / "gold/v5_pullback_reversal_shadow")
    assert rows.height == 3
    assert rows["stable_row_key"].n_unique() == 3
    grouped = rows.group_by(["symbol", "horizon_hours"]).len()
    assert {(row["symbol"], str(row["horizon_hours"])) for row in grouped.to_dicts()} == {
        ("BNB-USDT", "24"),
        ("BNB-USDT", "48"),
        ("SOL-USDT", "24"),
    }


def test_ingest_v5_paper_strategy_rows_dedupes_overlapping_summary(tmp_path):
    paper_csv = (
        "strategy_id,run_id,ts_utc,symbol,would_enter,final_decision,cost_source\n"
        "SOL_F4_VOLUME_EXPANSION_PAPER_V1,run_1,2026-05-12T06:01:00Z,SOL-USDT,false,no_order,mixed_actual_proxy\n"
        "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1,run_1,2026-05-12T06:02:00Z,ETH-USDT,true,paper,mixed_actual_proxy\n"
    )
    first = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T080500Z.tar.gz",
        {
            "summaries/paper_strategy_runs.csv": paper_csv,
            "raw/recent_runs/run_1/summary.json": '{"bundle_marker":"first"}',
        },
    )
    second = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T081500Z.tar.gz",
        {
            "summaries/paper_strategy_runs.csv": paper_csv,
            "raw/recent_runs/run_1/summary.json": '{"bundle_marker":"second"}',
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(first, lake, tmp_path / "restricted", tmp_path / "redacted")
    ingest_v5_bundle(second, lake, tmp_path / "restricted", tmp_path / "redacted")

    runs = read_parquet_dataset(lake / "silver/v5_paper_strategy_run")
    assert runs.height == 2
    assert runs["stable_row_key"].n_unique() == 2
    assert set(runs["bundle_name"].to_list()) == {second.name}


def test_ingest_and_export_v5_profitability_diagnostics_are_idempotent(tmp_path):
    funnel_header = (
        "schema_version,run_id,ts_utc,execution_mode,stage_order,stage,input_count,"
        "output_count,dropped_count,conversion_rate,entry_output_count,"
        "exit_output_count,primary_blocker,blocker_mix,count_source,live_order_effect\n"
    )
    exit_header = (
        "schema_version,proposal_id,strategy_id,strategy_version,symbol,"
        "closed_trade_count,avg_net_pnl_bps,avg_mfe_bps,avg_mae_bps,"
        "avg_profit_giveback_bps,avg_exit_efficiency,avg_holding_bars,"
        "high_profit_giveback_count,exit_reason_mix,exit_timing_state_mix,"
        "diagnosis,valid_for_live_orders,live_order_effect\n"
    )
    run_csv = (
        "schema_version,paper_trade_id,proposal_id,strategy_id,strategy_version,"
        "symbol,ts_utc,closed_at,net_pnl_bps,mfe_bps,mae_bps,"
        "profit_giveback_bps,exit_efficiency,exit_timing_bars,"
        "exit_timing_state,holding_period_seconds,exit_reason\n"
        "v5.generic_paper_runtime.v1,trade-1,proposal-1,strategy-1,1,SOL/USDT,"
        "2026-07-12T01:00:00Z,2026-07-12T01:00:00Z,25,80,-30,55,0.3125,"
        "4,profit_giveback,14400,max_holding_bars\n"
    )
    first = make_tar(
        tmp_path / "v5_live_followup_bundle_20260712T010000Z.tar.gz",
        {
            "summaries/trade_opportunity_funnel.csv": funnel_header
            + "v5.trade_opportunity_funnel.v1,run-1,2026-07-12T01:00:00Z,"
            "shadow,50,local_order_generation,4,0,4,0,0,0,"
            'protect_entry_rsi_confirm_too_weak,"{}",v5_order_intents,'
            "read_only_observability\n",
            "summaries/paper_strategy_exit_quality.csv": exit_header
            + "v5.generic_paper_runtime.v1,proposal-1,strategy-1,1,SOL/USDT,"
            '1,25,80,-30,55,0.3125,4,1,"{}","{}",high_profit_giveback,'
            "false,none\n",
            "summaries/paper_strategy_runs.csv": run_csv,
        },
    )
    second = make_tar(
        tmp_path / "v5_live_followup_bundle_20260712T020000Z.tar.gz",
        {
            "summaries/trade_opportunity_funnel.csv": funnel_header
            + "v5.trade_opportunity_funnel.v1,run-1,2026-07-12T02:00:00Z,"
            "shadow,50,local_order_generation,4,1,3,0.25,1,0,"
            'protect_entry_volume_confirm_negative,"{}",v5_order_intents,'
            "read_only_observability\n",
            "summaries/paper_strategy_exit_quality.csv": exit_header
            + "v5.generic_paper_runtime.v1,proposal-1,strategy-1,1,SOL/USDT,"
            '2,35,90,-25,55,0.3889,5,1,"{}","{}",high_profit_giveback,'
            "false,none\n",
            "summaries/paper_strategy_runs.csv": run_csv,
        },
    )
    lake = tmp_path / "lake"
    restricted = tmp_path / "restricted"
    redacted = tmp_path / "redacted"

    ingest_v5_bundle(
        first,
        lake,
        restricted,
        redacted,
        run_analysis=False,
        refresh_candidate_gold=False,
    )
    for _ in range(10):
        ingest_v5_bundle(
            second,
            lake,
            restricted,
            redacted,
            run_analysis=False,
            refresh_candidate_gold=False,
        )

    funnel = read_parquet_dataset(lake / "silver/v5_trade_opportunity_funnel")
    exit_quality = read_parquet_dataset(lake / "silver/v5_paper_strategy_exit_quality")
    runs = read_parquet_dataset(lake / "silver/v5_paper_strategy_run")
    assert funnel.height == 1
    assert funnel["output_count"][0] == "1"
    assert exit_quality.height == 1
    assert exit_quality["closed_trade_count"][0] == "2"
    assert runs.height == 1
    assert runs["mfe_bps"][0] == "80"
    assert runs["mae_bps"][0] == "-30"
    assert runs["profit_giveback_bps"][0] == "55"

    export = export_daily_pack(
        export_date="2026-07-12",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily", "--no-pre-export-v5-refresh"],
        pre_export_v5_refresh=False,
    )
    with zipfile.ZipFile(export.zip_path) as archive:
        names = set(archive.namelist())
        assert "v5/v5_trade_opportunity_funnel.csv" in names
        assert "v5/v5_paper_strategy_exit_quality.csv" in names
        exported_runs = list(
            csv.DictReader(
                StringIO(archive.read("v5/v5_paper_strategy_runs.csv").decode("utf-8"))
            )
        )
    assert exported_runs[0]["mfe_bps"] == "80"
    assert exported_runs[0]["profit_giveback_bps"] == "55"


def test_ingest_and_export_generic_paper_runtime_contract(tmp_path):
    proposal_id = "TRX_ALT_IMPULSE_48H_PAPER_V1"
    tracker_id = f"paper:{proposal_id}"
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260711T040000Z.tar.gz",
        {
            "summaries/paper_strategy_proposal_ack.csv": (
                "proposal_id,proposal_hash,paper_tracker_id,tracker_id,accepted,"
                "accepted_at,paper_only,recommended_mode,symbol,strategy_candidate,"
                "strategy_version,reject_reason,contract_version,rules_locked,"
                "live_order_effect,schema_version\n"
                f"{proposal_id},hash-trx,{tracker_id},{tracker_id},true,"
                "2026-07-11T03:00:00Z,true,paper,TRX/USDT,alt_impulse,1,,"
                "quant-lab.paper-strategy.v1,true,none,v5.generic_paper_runtime.v1\n"
            ),
            "summaries/paper_strategy_proposal_ack_current.csv": (
                "proposal_id,proposal_hash,paper_tracker_id,tracker_id,accepted,"
                "accepted_at,paper_only,recommended_mode,symbol,strategy_candidate,"
                "strategy_version,reject_reason,contract_version,rules_locked,"
                "live_order_effect,schema_version\n"
                f"{proposal_id},hash-trx,{tracker_id},{tracker_id},true,"
                "2026-07-11T03:00:00Z,true,paper,TRX/USDT,alt_impulse,1,,"
                "quant-lab.paper-strategy.v1,true,none,v5.generic_paper_runtime.v1\n"
            ),
            "summaries/paper_strategy_registry.csv": (
                "schema_version,proposal_id,proposal_hash,tracker_id,strategy_id,"
                "strategy_version,strategy_family,symbol,timeframe,state,rules_locked,"
                "paper_only,live_order_effect,created_at,updated_at\n"
                f"v5.generic_paper_runtime.v1,{proposal_id},hash-trx,{tracker_id},"
                "trx_alt_impulse,1,alt_impulse,TRX/USDT,1h,WAITING_SIGNAL,true,true,"
                "none,2026-07-11T03:00:00Z,2026-07-11T04:00:00Z\n"
            ),
            "summaries/paper_strategy_trackers_current.csv": (
                "schema_version,proposal_id,proposal_hash,tracker_id,strategy_id,"
                "strategy_version,strategy_family,symbol,timeframe,state,rules_locked,"
                "paper_only,live_order_effect,created_at,updated_at\n"
                f"v5.generic_paper_runtime.v1,{proposal_id},hash-trx,{tracker_id},"
                "trx_alt_impulse,1,alt_impulse,TRX/USDT,1h,WAITING_SIGNAL,true,true,"
                "none,2026-07-11T03:00:00Z,2026-07-11T04:00:00Z\n"
            ),
            "summaries/paper_strategy_state.csv": (
                "schema_version,tracker_id,proposal_id,strategy_id,symbol,state,"
                "paper_trade_id,open_paper_position,cooldown_remaining_bars,"
                "last_processed_bar_ts,updated_at\n"
                f"v5.generic_paper_runtime.v1,{tracker_id},{proposal_id},"
                "trx_alt_impulse,TRX/USDT,PAPER_CLOSED,trade-trx,false,0,"
                "2026-07-11T03:00:00Z,2026-07-11T04:00:00Z\n"
            ),
            "summaries/paper_strategy_signals.csv": (
                "schema_version,signal_id,proposal_id,tracker_id,strategy_id,"
                "strategy_version,symbol,signal_ts,decision_ts,signal_type,triggered,"
                "observability,arrival_bid,arrival_ask,arrival_mid,spread_bps,"
                "quote_timestamp,quote_age_seconds,price_source,fallback_level,"
                "valid_for_promotion\n"
                f"v5.generic_paper_runtime.v1,signal-trx,{proposal_id},{tracker_id},"
                "trx_alt_impulse,1,TRX/USDT,2026-07-11T03:00:00Z,"
                "2026-07-11T03:00:01Z,entry,true,OBSERVABLE,0.125,0.126,0.1255,"
                "79.68,2026-07-11T03:00:00Z,1,top_of_book,NONE,true\n"
            ),
            "summaries/paper_strategy_runs.csv": (
                "schema_version,paper_trade_id,proposal_id,strategy_id,strategy_version,"
                "symbol,direction,ts_utc,as_of_date,paper_tracker_id,strategy_candidate,"
                "recommended_mode,board_decision,suggested_horizon,horizon_hours,"
                "would_enter,would_exit,would_size,would_size_usdt,paper_source,"
                "paper_count_scope,entry_decision_ts,entry_arrival_mid,entry_bid,"
                "entry_ask,virtual_entry_price,paper_notional,virtual_quantity,"
                "exit_decision_ts,exit_arrival_mid,virtual_exit_price,fee_estimate,"
                "slippage_estimate,total_cost_bps,gross_pnl_bps,net_pnl_bps,"
                "paper_pnl_bps,paper_pnl_usdt,estimated_spread_bps,holding_bars,"
                "exit_reason,cost_source,cost_trust_level,market_regime,"
                "valid_for_promotion,closed_at\n"
                f"v5.generic_paper_runtime.v1,trade-trx,{proposal_id},trx_alt_impulse,"
                f"1,TRX/USDT,long,2026-07-11T04:00:00Z,2026-07-11,{tracker_id},"
                "alt_impulse,paper,PAPER_TRACKER_ACTIVE,48h,48,true,true,5,5,"
                "generic_contract_runtime,closed_trade,2026-07-11T03:00:01Z,0.1255,"
                "0.125,0.126,0.1261,5,39.65,2026-07-11T04:00:00Z,0.127,0.1269,"
                "2,2,83.68,120,112,112,0.056,79.68,1,structured_exit_rule,"
                "configured_conservative_paper,PAPER_ONLY,TREND_UP,true,"
                "2026-07-11T04:00:00Z\n"
            ),
            "summaries/paper_strategy_daily.csv": (
                "schema_version,paper_date,proposal_id,paper_tracker_id,strategy_id,"
                "symbol,paper_days,heartbeat_day_count,entry_day_count,"
                "daily_would_enter_count,cumulative_would_enter_count,would_enter_count,"
                "closed_entries,daily_paper_pnl_observed_count,"
                "cumulative_paper_pnl_observed_count,paper_pnl_observed_count,"
                "paper_pnl_day_count,avg_paper_pnl_bps,arrival_mid_coverage,"
                "spread_observation_coverage,cost_source_mix,created_at\n"
                f"v5.generic_paper_runtime.v1,2026-07-11,{proposal_id},{tracker_id},"
                "trx_alt_impulse,TRX/USDT,1,1,1,1,1,1,1,1,1,1,1,112,1,1,"
                '"[""configured_conservative_paper""]",2026-07-11T04:00:00Z\n'
            ),
            "summaries/paper_strategy_quote_coverage.csv": (
                "schema_version,proposal_id,strategy_id,symbol,candidate_signal_count,"
                "observable_quote_count,arrival_mid_coverage,stale_quote_rate,"
                "quote_fallback_rate,target_coverage\n"
                f"v5.generic_paper_runtime.v1,{proposal_id},trx_alt_impulse,TRX/USDT,"
                "1,1,1,0,0,0.95\n"
            ),
            "summaries/paper_strategy_cost_evidence.csv": (
                "schema_version,proposal_id,strategy_id,symbol,required_cost_trust_level,"
                "closed_trade_count,cost_observed_count,cost_source,cost_trust_level,"
                "valid_for_live_coverage\n"
                f"v5.generic_paper_runtime.v1,{proposal_id},trx_alt_impulse,TRX/USDT,"
                "CANARY,1,1,configured_conservative_paper,PAPER_ONLY,false\n"
            ),
            "summaries/paper_strategy_errors.csv": (
                "schema_version,ts_utc,proposal_id,error_code,error_type,error_message,"
                "live_order_effect\n"
                f"v5.generic_paper_runtime.v1,2026-07-11T02:00:00Z,{proposal_id},"
                "quote_not_observable,RuntimeError,stale quote,none\n"
            ),
            "summaries/paper_strategy_restart_recovery.csv": (
                "recovered_at,tracker_id,proposal_id,state,open_trade_preserved,"
                "schema_version\n"
                f"2026-07-11T03:30:00Z,{tracker_id},{proposal_id},PAPER_OPEN,true,"
                "v5.generic_paper_runtime.v1\n"
            ),
            "summaries/quant_lab_contract_status.json": json.dumps(
                {
                    "schema_version": "v5.generic_paper_runtime.v1",
                    "contract_version": "quant-lab.paper-strategy.v1",
                    "paper_runtime_enabled": True,
                    "paper_runtime_live_order_effect": "none",
                    "quant_lab_mode": "shadow",
                    "canary_enabled": False,
                    "active_tracker_count": 1,
                    "open_paper_position_count": 0,
                    "real_order_calls": 0,
                    "real_position_mutations": 0,
                    "generated_at": "2026-07-11T04:00:00Z",
                }
            ),
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(
        bundle,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        run_analysis=False,
        refresh_candidate_gold=False,
    )

    expected_counts = {
        "v5_paper_strategy_proposal_ack": 1,
        "v5_paper_strategy_registry": 1,
        "v5_paper_strategy_state": 1,
        "v5_paper_strategy_signal": 1,
        "v5_paper_strategy_run": 1,
        "v5_paper_strategy_daily": 1,
        "v5_paper_strategy_quote_coverage": 1,
        "v5_paper_strategy_cost_evidence": 1,
        "v5_paper_strategy_error": 1,
        "v5_paper_strategy_restart_recovery": 1,
        "v5_quant_lab_contract_status": 1,
    }
    for dataset, count in expected_counts.items():
        assert result.silver_rows[dataset] == count

    contract = read_parquet_dataset(lake / "silver/v5_quant_lab_contract_status").to_dicts()[0]
    assert contract["quant_lab_mode"] == "shadow"
    assert contract["paper_runtime_live_order_effect"] == "none"
    assert contract["canary_enabled"] is False
    assert contract["real_order_calls"] == 0
    assert contract["real_position_mutations"] == 0

    ingested_runs = read_parquet_dataset(lake / "silver/v5_paper_strategy_run")
    normalized_runs = build_paper_strategy_runs_from_v5(ingested_runs).to_dicts()
    assert normalized_runs[0]["would_enter"] is True
    assert normalized_runs[0]["paper_pnl_bps"] == 112.0
    assert normalized_runs[0]["paper_source"] == "generic_contract_runtime"
    assert normalized_runs[0]["arrival_bid"] == 0.125
    assert normalized_runs[0]["arrival_ask"] == 0.126
    assert normalized_runs[0]["arrival_mid"] == 0.1255
    assert normalized_runs[0]["virtual_entry_price"] == 0.1261
    assert normalized_runs[0]["virtual_exit_price"] == 0.1269
    assert normalized_runs[0]["cost_trust_level"] == "PAPER_ONLY"
    assert normalized_runs[0]["market_regime"] == "TREND_UP"
    assert normalized_runs[0]["exit_reason"] == "structured_exit_rule"
    assert normalized_runs[0]["holding_bars"] == 1
    assert normalized_runs[0]["observability"] == "OBSERVABLE"

    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-07-11",
                    "proposal_id": "LEGACY_PAPER_TRACKER",
                    "strategy_candidate": "legacy.paper",
                    "symbol": "BTC-USDT",
                    "paper_days": 1,
                    "paper_slippage_coverage": 0.5,
                    "arrival_mid_coverage": 0.5,
                    "spread_observation_coverage": 0.5,
                    "required_slippage_coverage": 0.8,
                }
            ]
        ),
        lake / "silver/v5_paper_slippage_coverage",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "contract_version": "quant_lab.paper_strategy.v1",
                    "proposal_id": proposal_id,
                    "proposal_hash": "hash-trx",
                    "strategy_id": "trx_alt_impulse",
                    "strategy_version": "1",
                    "strategy_family": "alt_impulse",
                    "strategy_candidate": "alt_impulse",
                    "symbol": "TRX/USDT",
                    "timeframe": "1h",
                    "max_holding_bars": 48,
                    "recommended_mode": "paper",
                    "entry_rule": '{"operator":"gt","field":"close","value":0}',
                    "exit_rule": '{"operator":"max_holding_bars","value":48}',
                }
            ]
        ),
        lake / "gold/paper_strategy_proposal",
    )
    tracking = build_and_publish_paper_strategy_tracking(lake, as_of_date="2026-07-11")
    pipeline = build_and_publish_paper_strategy_pipeline(lake, as_of_date="2026-07-11")
    daily = read_parquet_dataset(lake / "gold/paper_strategy_daily").to_dicts()[0]
    slippage = read_parquet_dataset(lake / "gold/paper_slippage_coverage")
    promotion = read_parquet_dataset(lake / "gold/paper_strategy_promotion_gate").to_dicts()[0]
    assert tracking.paper_strategy_runs == 1
    assert pipeline.paper_strategy_registry == 1
    assert daily["source_schema_version"] == "v5.generic_paper_runtime.v1"
    assert daily["paper_days"] == 1
    assert daily["heartbeat_day_count"] == 1
    assert daily["entry_day_count"] == 1
    assert daily["paper_pnl_observed_count"] == 1
    assert slippage.filter(pl.col("proposal_id") == proposal_id).height == 1
    assert promotion["lifecycle_state"] == "PAPER_EVIDENCE_INSUFFICIENT"
    assert promotion["paper_ready"] is False

    export = export_daily_pack(
        export_date="2026-07-11",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily", "--no-pre-export-v5-refresh"],
        pre_export_v5_refresh=False,
    )
    required_members = {
        "v5/v5_paper_strategy_proposal_ack.csv",
        "v5/v5_paper_strategy_registry.csv",
        "v5/v5_paper_strategy_state.csv",
        "v5/v5_paper_strategy_signals.csv",
        "v5/v5_paper_strategy_runs.csv",
        "v5/v5_paper_strategy_daily.csv",
        "v5/v5_paper_strategy_quote_coverage.csv",
        "v5/v5_paper_strategy_cost_evidence.csv",
        "v5/v5_paper_strategy_errors.csv",
        "v5/v5_paper_strategy_restart_recovery.csv",
        "v5/v5_quant_lab_contract_status.csv",
    }
    with zipfile.ZipFile(export.zip_path) as archive:
        assert required_members.issubset(archive.namelist())
        exported_contract = list(
            csv.DictReader(
                StringIO(archive.read("v5/v5_quant_lab_contract_status.csv").decode("utf-8"))
            )
        )[0]
        exported_runs = list(
            csv.DictReader(StringIO(archive.read("v5/v5_paper_strategy_runs.csv").decode("utf-8")))
        )
    assert exported_contract["quant_lab_mode"] == "shadow"
    assert exported_contract["canary_enabled"].lower() == "false"
    assert exported_runs[0]["paper_pnl_bps"] == "112"


def test_ingest_fill_bill_reconciliation_replaces_pending_with_late_bill(tmp_path):
    header = (
        "schema_version,generated_at,runtime_scope,symbol,order_id,cl_ord_id,"
        "trade_ids,last_fill_ts,selected_fee_usdt,bill_match_status,"
        "cost_evidence_status,cost_source\n"
    )
    pending = make_tar(
        tmp_path / "v5_live_followup_bundle_20260711T030000Z.tar.gz",
        {
            "summaries/fill_bill_cost_reconciliation.csv": (
                header + "v5.fill_bill_cost_reconciliation.v1,2026-07-11T03:00:00Z,"
                "fills,ETH-USDT,order-1,client-1,trade-1,2026-07-11T02:59:00Z,"
                "0.005,BILL_PENDING,PARTIAL,actual_fill_partial\n"
            )
        },
    )
    complete = make_tar(
        tmp_path / "v5_live_followup_bundle_20260711T031000Z.tar.gz",
        {
            "summaries/fill_bill_cost_reconciliation.csv": (
                header + "v5.fill_bill_cost_reconciliation.v1,2026-07-11T03:10:00Z,"
                "fills,ETH-USDT,order-1,client-1,trade-1,2026-07-11T02:59:00Z,"
                "0.005,PASS,ACTUAL,actual_fills_bills\n"
            )
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(
        pending,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        run_analysis=False,
        refresh_candidate_gold=False,
    )
    ingest_v5_bundle(
        complete,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        run_analysis=False,
        refresh_candidate_gold=False,
    )

    rows = read_parquet_dataset(lake / "silver/v5_fill_bill_cost_reconciliation")
    assert rows.height == 1
    row = rows.to_dicts()[0]
    assert row["bill_match_status"] == "PASS"
    assert row["cost_evidence_status"] == "ACTUAL"
    assert row["bundle_name"] == complete.name

    export = export_daily_pack(
        export_date="2026-07-11",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily", "--no-pre-export-v5-refresh"],
        pre_export_v5_refresh=False,
    )
    with zipfile.ZipFile(export.zip_path) as archive:
        exported = list(
            csv.DictReader(
                StringIO(archive.read("v5/v5_fill_bill_cost_reconciliation.csv").decode("utf-8"))
            )
        )
    assert exported[0]["bill_match_status"] == "PASS"
    assert exported[0]["cost_evidence_status"] == "ACTUAL"


def test_ingest_v5_expanded_universe_summary_rows(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T083000Z.tar.gz",
        {
            "summaries/expanded_universe_advisory_reader.csv": (
                "strategy_id,strategy_candidate,symbol,decision,recommended_mode,universe_type,generated_at\n"
                "WLD_EXPANDED_UNIVERSE_PAPER_V1,v5.expanded_universe_wld_paper,WLD-USDT,PAPER_READY,paper,expanded_paper,2026-05-14T08:30:00Z\n"
            ),
            "summaries/expanded_universe_paper_runs.csv": (
                "strategy_id,run_id,ts_utc,symbol,would_enter,no_sample_reason\n"
                "WLD_EXPANDED_UNIVERSE_PAPER_V1,run_wld,2026-05-14T08:30:00Z,WLD-USDT,false,no_wld_candidate\n"
            ),
            "summaries/expanded_universe_paper_daily.csv": (
                "strategy_id,paper_date,symbol,entry_count,no_sample_reason,generated_at\n"
                "WLD_EXPANDED_UNIVERSE_PAPER_V1,2026-05-14,WLD-USDT,0,no_wld_candidate,2026-05-14T08:30:00Z\n"
            ),
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    reader = read_parquet_dataset(lake / "silver/v5_expanded_universe_advisory_reader")
    runs = read_parquet_dataset(lake / "silver/v5_expanded_universe_paper_runs")
    daily = read_parquet_dataset(lake / "silver/v5_expanded_universe_paper_daily")

    assert reader.height == 1
    assert runs.height == 1
    assert daily.height == 1
    assert reader["symbol"].to_list() == ["WLD-USDT"]
    assert runs["no_sample_reason"].to_list() == ["no_wld_candidate"]
    assert daily["entry_count"].cast(str).to_list() == ["0"]


def test_ingest_rehydrates_empty_expanded_universe_tables_for_already_ingested_bundle(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T083000Z.tar.gz",
        {
            "summaries/expanded_universe_advisory_reader.csv": (
                "strategy_id,strategy_candidate,symbol,decision,recommended_mode,universe_type,generated_at\n"
                "WLD_EXPANDED_UNIVERSE_PAPER_V1,v5.expanded_universe_wld_paper,WLD-USDT,PAPER_READY,paper,expanded_paper,2026-05-14T08:30:00Z\n"
            ),
            "summaries/expanded_universe_paper_runs.csv": (
                "strategy_id,run_id,ts_utc,symbol,would_enter,no_sample_reason\n"
                "WLD_EXPANDED_UNIVERSE_PAPER_V1,run_wld,2026-05-14T08:30:00Z,WLD-USDT,false,no_wld_candidate\n"
            ),
            "summaries/expanded_universe_paper_daily.csv": (
                "strategy_id,paper_date,symbol,entry_count,no_sample_reason,generated_at\n"
                "WLD_EXPANDED_UNIVERSE_PAPER_V1,2026-05-14,WLD-USDT,0,no_wld_candidate,2026-05-14T08:30:00Z\n"
            ),
        },
    )
    lake = tmp_path / "lake"

    first = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")
    assert first.skipped is False
    for name in (
        "v5_expanded_universe_advisory_reader",
        "v5_expanded_universe_paper_runs",
        "v5_expanded_universe_paper_daily",
    ):
        write_parquet_dataset(
            pl.DataFrame(schema={"strategy": pl.Utf8}),
            lake / f"silver/{name}",
        )

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    reader = read_parquet_dataset(lake / "silver/v5_expanded_universe_advisory_reader")
    runs = read_parquet_dataset(lake / "silver/v5_expanded_universe_paper_runs")
    daily = read_parquet_dataset(lake / "silver/v5_expanded_universe_paper_daily")

    assert result.skipped is True
    assert result.silver_rows["v5_expanded_universe_advisory_reader"] == 1
    assert result.silver_rows["v5_expanded_universe_paper_runs"] == 1
    assert result.silver_rows["v5_expanded_universe_paper_daily"] == 1
    assert any(
        warning.startswith("rehydrated_empty_csv_refresh_datasets:") for warning in result.warnings
    )
    assert reader.height == 1
    assert runs.height == 1
    assert daily.height == 1
    assert reader["symbol"].to_list() == ["WLD-USDT"]
    assert runs["no_sample_reason"].to_list() == ["no_wld_candidate"]
    assert daily["entry_count"].cast(str).to_list() == ["0"]


def test_ingest_rehydrates_cost_probe_p3_preflight_for_already_ingested_bundle(tmp_path):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260514T083000Z.tar.gz")
    lake = tmp_path / "lake"

    first = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")
    assert first.skipped is False
    write_parquet_dataset(
        pl.DataFrame(schema={"strategy": pl.Utf8}),
        lake / "silver/v5_cost_probe_p3_preflight",
    )

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    p3_preflight = read_parquet_dataset(lake / "silver/v5_cost_probe_p3_preflight")
    row = p3_preflight.to_dicts()[0]
    assert result.skipped is True
    assert result.silver_rows["v5_cost_probe_p3_preflight"] == 1
    assert row["state"] == "NOT_READY"
    assert row["offline_plan_state"] == "NO_PLAN_ROWS"
    assert row["online_exchange_preflight_state"] == "NOT_READY"
    assert row["effective_preflight_state"] == "NOT_READY"
    assert row["approved_live_order_execution"] is False
    assert row["live_order_effect"] == "none_preflight_only_no_order"
    assert any(
        warning.startswith("rehydrated_empty_csv_refresh_datasets:") for warning in result.warnings
    )


def test_ingest_replaces_redacted_cost_probe_p3_preflight_for_already_ingested_bundle(
    tmp_path,
):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260514T083000Z.tar.gz")
    lake = tmp_path / "lake"

    first = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")
    assert first.skipped is False
    dataset_path = lake / "silver/v5_cost_probe_p3_preflight"
    damaged_rows = read_parquet_dataset(dataset_path).to_dicts()
    assert damaged_rows
    for row in damaged_rows:
        row["manual_authorization_required"] = False
        row["raw_payload_json"] = (
            '{"manual_authorization_required":"<REDACTED>","approved_live_order_execution":false}'
        )
    write_parquet_dataset(pl.DataFrame(damaged_rows, infer_schema_length=None), dataset_path)

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    rows = read_parquet_dataset(dataset_path).to_dicts()
    assert result.skipped is True
    assert result.silver_rows["v5_cost_probe_p3_preflight"] == len(damaged_rows)
    assert [row["manual_authorization_required"] for row in rows] == [True]
    assert all("<REDACTED>" not in row["raw_payload_json"] for row in rows)
    assert all('"manual_authorization_required": true' in row["raw_payload_json"] for row in rows)


def test_ingest_rehydrates_cost_probe_p3_preflight_schema_upgrade(
    tmp_path,
):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260514T083000Z.tar.gz")
    lake = tmp_path / "lake"

    first = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")
    assert first.skipped is False
    dataset_path = lake / "silver/v5_cost_probe_p3_preflight"
    old_rows = read_parquet_dataset(dataset_path).drop(
        [
            "offline_plan_state",
            "online_exchange_preflight_state",
            "effective_preflight_state",
        ]
    )
    write_parquet_dataset(old_rows, dataset_path)

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    rows = read_parquet_dataset(dataset_path).to_dicts()
    assert result.skipped is True
    assert result.silver_rows["v5_cost_probe_p3_preflight"] == 1
    assert rows[0]["offline_plan_state"] == "NO_PLAN_ROWS"
    assert rows[0]["online_exchange_preflight_state"] == "NOT_READY"
    assert rows[0]["effective_preflight_state"] == "NOT_READY"


def test_ingest_exports_cost_probe_order_and_roundtrip_events(tmp_path):
    order_events = (
        "\n".join(
            [
                json.dumps(
                    {
                        "event_ts": "2026-05-14T08:30:00Z",
                        "order_key": "probe-entry-1",
                        "symbol": "BTC/USDT",
                        "leg": "entry",
                        "side": "buy",
                        "intent": "entry",
                        "order_status": "submitted",
                        "client_order_id": "probe-entry-1",
                        "exchange_order_id": "okx-entry-1",
                        "live_enabled": True,
                        "dry_run": False,
                        "no_order_submitted": False,
                        "notional_usdt": "5.0",
                        "live_order_effect": "live_cost_probe_order",
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "event_ts": "2026-05-14T08:30:02Z",
                        "order_key": "probe-entry-1",
                        "symbol": "BTC/USDT",
                        "leg": "entry",
                        "side": "buy",
                        "intent": "entry",
                        "order_status": "filled",
                        "client_order_id": "probe-entry-1",
                        "exchange_order_id": "okx-entry-1",
                        "filled_qty": "0.00007",
                        "avg_px": "65000",
                        "fee_usdt": "0.005",
                        "live_enabled": True,
                        "dry_run": False,
                        "no_order_submitted": False,
                        "live_order_effect": "live_cost_probe_order",
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n"
    )
    roundtrip_events = (
        json.dumps(
            {
                "event_ts": "2026-05-14T08:30:20Z",
                "symbol": "BTC/USDT",
                "roundtrip_status": "closed",
                "roundtrip_id": "probe-roundtrip-1",
                "entry_order_id": "okx-entry-1",
                "exit_order_id": "okx-exit-1",
                "entry_order_status": "filled",
                "exit_order_status": "filled",
                "opened_at": "2026-05-14T08:30:00Z",
                "closed_at": "2026-05-14T08:30:20Z",
                "fees_usdt": "0.01",
                "net_pnl_usdt": "-0.02",
                "authorization_id": "auth-1",
                "execution_completed": True,
                "flat_verified": True,
                "exchange_flat_verified": True,
                "local_flat_verified": True,
                "reconcile_ok": True,
                "cost_evidence_complete": True,
                "eligible_for_cost_model": True,
                "eligible_for_live_cost_coverage": False,
                "sample_origin": "cost_probe",
                "source": "bootstrap_cost_probe",
                "entry_filled_qty": "0.00007",
                "exit_filled_qty": "0.00007",
                "entry_fee_usdt": "0.005",
                "exit_fee_usdt": "0.005",
                "roundtrip_cost_bps": "20.0",
                "fee_conversion_warnings": "",
                "bill_match_status": "bill_not_observed",
                "fee_match_status": "fill_fee_observed",
                "live_enabled": True,
                "dry_run": False,
                "no_order_submitted": False,
                "live_order_effect": "live_cost_probe_roundtrip",
            },
            sort_keys=True,
        )
        + "\n"
    )
    live_execution_status = json.dumps(
        {
            "schema_version": "v5.cost_probe_live_execution_status.v1",
            "generated_at": "2026-05-14T08:30:21Z",
            "status": "CLOSED_FLAT",
            "source_state": "COMPLETED",
            "manual_probe_symbol": "BTC/USDT",
            "authorization_id": "auth-1",
            "authorization_issued_at": "2026-05-14T08:29:00Z",
            "authorization_expires_at": "2026-05-14T08:34:00Z",
            "authorization_consumed_at": "2026-05-14T08:30:00Z",
            "authorization_age_sec": 81,
            "authorization_fresh": True,
            "authorization_validated": True,
            "authorization_consumed": True,
            "approved_live_order_execution": False,
            "live_order_effect": "live_cost_probe_roundtrip",
            "no_order_submitted": False,
            "recovery_required": False,
            "recovery_only": False,
            "entry_submit_intent": True,
            "entry_client_order_id": "probe-entry-1",
            "entry_order_id": "okx-entry-1",
            "entry_submitted": True,
            "entry_filled": True,
            "entry_filled_qty": "0.00007",
            "exit_submit_intent": True,
            "exit_client_order_id": "probe-exit-1",
            "exit_order_id": "okx-exit-1",
            "exit_submitted": True,
            "exit_filled": True,
            "exit_filled_qty": "0.00007",
            "execution_completed": True,
            "flat_verified": True,
            "exchange_flat_verified": True,
            "local_flat_verified": True,
            "reconcile_ok": True,
            "instrument_state": "live",
            "quote_balance_sufficient": True,
            "blockers": [],
        },
        sort_keys=True,
    )
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T083000Z.tar.gz",
        {
            "summaries/cost_probe_live_execution_status.json": live_execution_status,
            "raw/reports/cost_probe_live_execution_status.json": live_execution_status,
            "summaries/cost_probe_order_events.jsonl": order_events,
            "summaries/cost_probe_roundtrip_events.jsonl": roundtrip_events,
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(
        bundle,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        run_analysis=False,
        refresh_candidate_gold=False,
    )

    order_rows = read_parquet_dataset(lake / "silver/v5_cost_probe_order_event").to_dicts()
    roundtrip_rows = read_parquet_dataset(lake / "silver/v5_cost_probe_roundtrip_event").to_dicts()
    live_status_rows = read_parquet_dataset(
        lake / "silver/v5_cost_probe_live_execution_status"
    ).to_dicts()
    assert result.silver_rows["v5_cost_probe_live_execution_status"] == 1
    assert result.silver_rows["v5_cost_probe_order_event"] == 2
    assert result.silver_rows["v5_cost_probe_roundtrip_event"] == 1
    assert {row["order_status"] for row in order_rows} == {"submitted", "filled"}
    assert {row["event_id"] for row in order_rows} == {
        "probe-entry-1|order:entry:submitted|2026-05-14T08:30:00Z",
        "probe-entry-1|order:entry:filled|2026-05-14T08:30:02Z",
    }
    assert order_rows[0]["normalized_symbol"] == "BTC-USDT"
    assert live_status_rows[0]["status"] == "CLOSED_FLAT"
    assert live_status_rows[0]["authorization_fresh"] is True
    assert live_status_rows[0]["authorization_consumed"] is True
    assert live_status_rows[0]["recovery_required"] is False
    historical_duplicate = {
        **live_status_rows[0],
        "source_path_inside_bundle": "raw/reports/cost_probe_live_execution_status.json",
    }
    write_parquet_dataset(
        pl.DataFrame([*live_status_rows, historical_duplicate]),
        lake / "silver/v5_cost_probe_live_execution_status",
    )
    assert roundtrip_rows[0]["roundtrip_key"] == "probe-roundtrip-1"
    assert roundtrip_rows[0]["event_id"] == (
        "probe-roundtrip-1|roundtrip:closed|2026-05-14T08:30:20Z"
    )

    export = export_daily_pack(
        export_date="2026-05-14",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily", "--no-pre-export-v5-refresh"],
        pre_export_v5_refresh=False,
    )
    with zipfile.ZipFile(export.zip_path) as archive:
        order_csv = list(
            csv.DictReader(
                StringIO(archive.read("v5/v5_cost_probe_order_events.csv").decode("utf-8"))
            )
        )
        roundtrip_csv = list(
            csv.DictReader(
                StringIO(archive.read("v5/v5_cost_probe_roundtrip_events.csv").decode("utf-8"))
            )
        )
        roundtrip_canonical_csv = list(
            csv.DictReader(
                StringIO(archive.read("v5/v5_cost_probe_roundtrip_canonical.csv").decode("utf-8"))
            )
        )
        live_status_csv = list(
            csv.DictReader(
                StringIO(archive.read("v5/v5_cost_probe_live_execution_status.csv").decode("utf-8"))
            )
        )
        live_status_canonical_csv = list(
            csv.DictReader(
                StringIO(
                    archive.read("v5/v5_cost_probe_live_execution_status_canonical.csv").decode(
                        "utf-8"
                    )
                )
            )
        )
        manifest = json.loads(archive.read("manifest.json"))

    assert len(order_csv) == 2
    assert len(roundtrip_csv) == 1
    assert len(roundtrip_canonical_csv) == 1
    assert len(live_status_csv) == 1
    assert len(live_status_canonical_csv) == 1
    assert live_status_csv[0]["status"] == "CLOSED_FLAT"
    assert live_status_canonical_csv[0]["status"] == "CLOSED_FLAT"
    assert live_status_canonical_csv[0]["canonical"] == "True"
    assert live_status_canonical_csv[0]["canonical_priority"] == "100"
    assert live_status_canonical_csv[0]["revision"] == "1"
    assert live_status_csv[0]["authorization_fresh"] == "True"
    assert live_status_csv[0]["authorization_fresh_at_export"] == "False"
    assert int(live_status_csv[0]["authorization_age_sec_at_export"]) > 81
    assert int(live_status_csv[0]["authorization_seconds_to_expiry"]) < 0
    assert live_status_csv[0]["authorization_expired_at_export"] == "True"
    assert live_status_csv[0]["authorization_consumed"] == "True"
    assert live_status_csv[0]["recovery_required"] == "False"
    assert order_csv[-1]["event_type"] == "order:entry:filled"
    assert roundtrip_csv[0]["live_order_effect"] == "live_cost_probe_roundtrip"
    assert roundtrip_csv[0]["authorization_id"] == "auth-1"
    assert roundtrip_csv[0]["execution_completed"] == "True"
    assert roundtrip_csv[0]["flat_verified"] == "True"
    assert roundtrip_csv[0]["exchange_flat_verified"] == "True"
    assert roundtrip_csv[0]["local_flat_verified"] == "True"
    assert roundtrip_csv[0]["reconcile_ok"] == "True"
    assert roundtrip_csv[0]["cost_evidence_complete"] == "True"
    assert roundtrip_csv[0]["eligible_for_cost_model"] == "True"
    assert roundtrip_csv[0]["eligible_for_live_cost_coverage"] == "False"
    assert roundtrip_csv[0]["sample_origin"] == "cost_probe"
    assert roundtrip_csv[0]["source"] == "bootstrap_cost_probe"
    assert roundtrip_csv[0]["entry_filled_qty"] == "0.00007"
    assert roundtrip_csv[0]["exit_filled_qty"] == "0.00007"
    assert roundtrip_csv[0]["entry_fee_usdt"] == "0.005"
    assert roundtrip_csv[0]["exit_fee_usdt"] == "0.005"
    assert roundtrip_csv[0]["roundtrip_cost_bps"] == "20.0"
    assert roundtrip_csv[0]["bill_match_status"] == "bill_not_observed"
    assert roundtrip_csv[0]["fee_match_status"] == "fill_fee_observed"
    assert roundtrip_canonical_csv[0]["canonical"] == "True"
    assert roundtrip_canonical_csv[0]["terminal"] == "True"
    assert roundtrip_canonical_csv[0]["revision"] == "1"
    files = {item["path"]: item for item in manifest["files"]}
    assert files["v5/v5_cost_probe_live_execution_status.csv"]["rows"] == 1
    assert files["v5/v5_cost_probe_live_execution_status_canonical.csv"]["rows"] == 1
    assert files["v5/v5_cost_probe_order_events.csv"]["rows"] == 2
    assert files["v5/v5_cost_probe_roundtrip_events.csv"]["rows"] == 1
    assert files["v5/v5_cost_probe_roundtrip_canonical.csv"]["rows"] == 1


def test_ingest_v5_empty_expanded_universe_summaries_refresh_datasets(tmp_path):
    first = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T083000Z.tar.gz",
        {
            "summaries/expanded_universe_advisory_reader.csv": (
                "strategy_id,strategy_candidate,symbol,decision,recommended_mode,universe_type,generated_at\n"
                "WLD_EXPANDED_UNIVERSE_PAPER_V1,v5.expanded_universe_wld_paper,WLD-USDT,PAPER_READY,paper,expanded_paper,2026-05-14T08:30:00Z\n"
            ),
            "summaries/expanded_universe_paper_runs.csv": (
                "strategy_id,run_id,ts_utc,symbol,would_enter,no_sample_reason\n"
                "WLD_EXPANDED_UNIVERSE_PAPER_V1,run_wld,2026-05-14T08:30:00Z,WLD-USDT,false,no_wld_candidate\n"
            ),
            "summaries/expanded_universe_paper_daily.csv": (
                "strategy_id,paper_date,symbol,entry_count,no_sample_reason,generated_at\n"
                "WLD_EXPANDED_UNIVERSE_PAPER_V1,2026-05-14,WLD-USDT,0,no_wld_candidate,2026-05-14T08:30:00Z\n"
            ),
        },
    )
    second = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T084000Z.tar.gz",
        {
            "summaries/expanded_universe_advisory_reader.csv": (
                "strategy_id,strategy_candidate,symbol,decision,recommended_mode,universe_type,generated_at\n"
            ),
            "summaries/expanded_universe_paper_runs.csv": (
                "strategy_id,run_id,ts_utc,symbol,would_enter,no_sample_reason\n"
            ),
            "summaries/expanded_universe_paper_daily.csv": (
                "strategy_id,paper_date,symbol,entry_count,no_sample_reason,generated_at\n"
            ),
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(first, lake, tmp_path / "restricted", tmp_path / "redacted")
    result = ingest_v5_bundle(second, lake, tmp_path / "restricted", tmp_path / "redacted")

    reader = read_parquet_dataset(lake / "silver/v5_expanded_universe_advisory_reader")
    runs = read_parquet_dataset(lake / "silver/v5_expanded_universe_paper_runs")
    daily = read_parquet_dataset(lake / "silver/v5_expanded_universe_paper_daily")

    assert result.silver_rows["v5_expanded_universe_advisory_reader"] == 0
    assert result.silver_rows["v5_expanded_universe_paper_runs"] == 0
    assert result.silver_rows["v5_expanded_universe_paper_daily"] == 0
    assert reader.height == 0
    assert runs.height == 0
    assert daily.height == 0
    assert "strategy_id" in reader.columns
    assert "no_sample_reason" in runs.columns
    assert "entry_count" in daily.columns


def test_ingest_v5_stable_rows_tolerates_late_mixed_schema_values(tmp_path):
    rows = [
        (
            f"strategy_{index},run_1,2026-05-12T06:{index % 60:02d}:00Z,"
            f"SOL-USDT,false,no_order,{index}.5"
        )
        for index in range(120)
    ]
    rows.append(
        "strategy_not_observable,run_1,2026-05-12T08:01:00Z,SOL-USDT,false,no_order,not_observable"
    )
    paper_csv = (
        "strategy_id,run_id,ts_utc,symbol,would_enter,final_decision,alpha6_score\n"
        + "\n".join(rows)
        + "\n"
    )
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T082500Z.tar.gz",
        {"summaries/paper_strategy_runs.csv": paper_csv},
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    runs = read_parquet_dataset(lake / "silver/v5_paper_strategy_run")
    assert runs.height == 121
    assert "not_observable" in runs["alpha6_score"].cast(str).to_list()


def test_ingest_v5_order_lifecycle_computes_realized_cost_parts(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260515T010500Z.tar.gz",
        {
            "raw/recent_runs/run_lifecycle/order_lifecycle.csv": (
                "schema_version,lifecycle_id,run_id,ts_utc,symbol,normalized_symbol,side,intent,order_state,"
                "decision_ts,signal_price,arrival_bid,arrival_ask,arrival_mid,spread_bps_at_decision,"
                "submit_ts,order_type,order_px,cl_ord_id,exchange_order_id,first_fill_ts,last_fill_ts,"
                "fill_px,avg_fill_px,filled_qty,fee,fee_ccy,fee_usdt,notional_usdt,requested_notional_usdt,trade_ids,fill_count\n"
                "v5.order_lifecycle.v1,olc-1,run_lifecycle,2026-05-15T01:00:04Z,BNB/USDT,BNB-USDT,buy,OPEN_LONG,FILLED,"
                "2026-05-15T01:00:00Z,600,599,601,600,33.3333333333,"
                "2026-05-15T01:00:01Z,market,,clid-1,okx-1,2026-05-15T01:00:04Z,2026-05-15T01:00:04Z,"
                "602,602,0.2,-0.1204,USDT,0.1204,120.4,120,trade-1,1\n"
            ),
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.silver_rows["v5_order_lifecycle"] == 1
    rows = read_parquet_dataset(lake / "silver/v5_order_lifecycle").to_dicts()
    row = rows[0]
    assert row["symbol"] == "BNB-USDT"
    assert row["normalized_symbol"] == "BNB-USDT"
    assert float(row["arrival_slippage_bps"]) > 0
    assert float(row["spread_cost_bps"]) > 0
    assert float(row["fee_bps"]) > 0
    assert float(row["total_realized_cost_bps"]) > 0
    assert float(row["realized_total_cost_bps"]) > 0


def test_ingest_v5_order_lifecycle_accepts_fill_ts_and_spread_aliases(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260515T020500Z.tar.gz",
        {
            "raw/recent_runs/run_lifecycle/order_lifecycle.csv": (
                "schema_version,lifecycle_id,run_id,fill_ts,symbol,side,intent,order_state,"
                "decision_ts,signal_price,arrival_mid,spread,submit_ts,fill_px,fill_qty,"
                "fee_usdt,notional_usdt,exchange_order_id,trade_id,fill_count\n"
                "v5.order_lifecycle.v2,olc-btc-1,run_lifecycle,2026-05-15T02:00:04Z,"
                "BTC/USDT,buy,OPEN_LONG,FILLED,2026-05-15T02:00:00Z,60000,60005,"
                "2.5,2026-05-15T02:00:01Z,60020,0.01,0.3001,600.2,okx-btc-1,"
                "trade-btc-1,1\n"
            ),
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.silver_rows["v5_order_lifecycle"] == 1
    row = read_parquet_dataset(lake / "silver/v5_order_lifecycle").to_dicts()[0]
    assert row["ts_utc"] == "2026-05-15T02:00:04Z"
    assert row["symbol"] == "BTC-USDT"
    assert row["normalized_symbol"] == "BTC-USDT"
    assert float(row["arrival_slippage_bps"]) > 0
    assert float(row["delay_cost_bps"]) > 0
    assert float(row["spread_bps_at_decision"]) == 2.5
    assert float(row["fee_bps"]) > 0
    assert float(row["realized_total_cost_bps"]) > 0


def test_ingest_parses_quant_lab_usage_files(tmp_path):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz")
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.silver_rows["v5_quant_lab_usage"] == 1
    assert result.silver_rows["v5_quant_lab_request"] == 1
    assert result.silver_rows["v5_quant_lab_compliance"] == 1
    assert result.silver_rows["v5_quant_lab_cost_usage"] == 1
    assert result.silver_rows["v5_quant_lab_fallback"] == 1
    assert result.silver_rows["v5_paper_strategy_run"] == 1
    assert result.silver_rows["v5_paper_strategy_daily"] == 1
    assert result.silver_rows["v5_paper_slippage_coverage"] == 1
    assert result.silver_rows["v5_bnb_profit_lock_shadow"] == 1
    assert result.silver_rows["v5_bnb_negative_expectancy_attribution"] == 1
    assert result.silver_rows["v5_final_score_vs_alpha6_conflict"] == 1
    assert result.silver_rows["v5_bnb_strong_alpha6_bypass_shadow"] == 1
    assert result.silver_rows["v5_negative_expectancy_attribution"] == 1
    assert result.silver_rows["v5_bnb_paper_strategy_runs"] == 1
    assert result.silver_rows["v5_bnb_paper_strategy_daily"] == 1
    assert result.silver_rows["v5_negative_expectancy_consistency"] == 1
    assert result.silver_rows["v5_btc_probe_entry_quality_audit"] == 1
    assert result.silver_rows["v5_cost_probe_p3_preflight"] == 1
    assert read_parquet_dataset(lake / "silver/v5_quant_lab_usage").height == 1
    assert read_parquet_dataset(lake / "silver/v5_quant_lab_compliance").height == 1
    bnb_shadow = read_parquet_dataset(lake / "silver/v5_bnb_profit_lock_shadow").to_dicts()
    assert bnb_shadow[0]["symbol"] == "BNB-USDT"
    assert bnb_shadow[0]["best_shadow_exit_policy"] == "delayed_exit_12h"
    bnb_attr = read_parquet_dataset(
        lake / "silver/v5_bnb_negative_expectancy_attribution"
    ).to_dicts()
    assert bnb_attr[0]["symbol"] == "BNB-USDT"
    assert bnb_attr[0]["min_hold_violation"] == "true"
    conflict = read_parquet_dataset(lake / "silver/v5_final_score_vs_alpha6_conflict").to_dicts()
    assert conflict[0]["symbol"] == "BNB-USDT"
    assert conflict[0]["missed_profit_flag"] == "true"
    bypass = read_parquet_dataset(lake / "silver/v5_bnb_strong_alpha6_bypass_shadow").to_dicts()
    assert bypass[0]["live_order_effect"] == "read_only_no_live_order"
    generic_attr = read_parquet_dataset(
        lake / "silver/v5_negative_expectancy_attribution"
    ).to_dicts()
    assert generic_attr[0]["would_unblock_if_adjusted"] == "true"
    negexp = read_parquet_dataset(lake / "silver/v5_negative_expectancy_consistency").to_dicts()
    assert negexp[0]["symbol"] == "BNB-USDT"
    assert negexp[0]["min_hold_violation_cycles"] == "1"
    btc_probe = read_parquet_dataset(lake / "silver/v5_btc_probe_entry_quality_audit").to_dicts()
    assert btc_probe[0]["normalized_symbol"] == "BTC-USDT"
    assert btc_probe[0]["same_symbol_reentry_bypass"] == "probe_stop_loss_reentry_after_loss"
    assert btc_probe[0]["anti_chase_flag"] == "false"
    assert btc_probe[0]["live_order_effect"] == "none_read_only_v5_bundle_audit"
    p3_preflight = read_parquet_dataset(lake / "silver/v5_cost_probe_p3_preflight").to_dicts()
    assert p3_preflight[0]["state"] == "NOT_READY"
    assert p3_preflight[0]["offline_plan_state"] == "NO_PLAN_ROWS"
    assert p3_preflight[0]["online_exchange_preflight_state"] == "NOT_READY"
    assert p3_preflight[0]["effective_preflight_state"] == "NOT_READY"
    assert p3_preflight[0]["manual_authorization_required"] is True
    assert p3_preflight[0]["approved_live_order_execution"] is False
    assert p3_preflight[0]["live_order_effect"] == "none_preflight_only_no_order"
    assert '"manual_authorization_required": true' in p3_preflight[0]["raw_payload_json"]
    redacted_manual_auth = '"manual_authorization_required": "<REDACTED>"'
    assert redacted_manual_auth not in p3_preflight[0]["raw_payload_json"]
    assert "dry_run_plan_not_ready" in p3_preflight[0]["blockers_json"]
    for dataset in [
        "v5_bnb_profit_lock_shadow",
        "v5_bnb_strong_alpha6_bypass_shadow",
        "v5_bnb_paper_strategy_runs",
        "v5_bnb_paper_strategy_daily",
        "v5_cost_probe_p3_preflight",
    ]:
        dataset_path = lake / "silver" / dataset
        meta = json.loads((dataset_path / "_snapshot_meta.json").read_text(encoding="utf-8"))
        assert meta["dataset"] == dataset
        assert meta["row_count"] == read_parquet_dataset(dataset_path).height
        assert meta["generated_at"]
        assert meta["source_sha"]
    export = export_daily_pack(
        export_date="2026-05-10",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily", "--no-pre-export-v5-refresh"],
        pre_export_v5_refresh=False,
    )
    with zipfile.ZipFile(export.zip_path) as archive:
        exported_p3 = list(
            csv.DictReader(
                StringIO(archive.read("v5/v5_cost_probe_p3_preflight.csv").decode("utf-8"))
            )
        )
    assert exported_p3[0]["manual_authorization_required"].lower() == "true"
    assert exported_p3[0]["offline_plan_state"] == "NO_PLAN_ROWS"
    assert exported_p3[0]["online_exchange_preflight_state"] == "NOT_READY"
    assert exported_p3[0]["effective_preflight_state"] == "NOT_READY"
    assert "SOL/USDT" in exported_p3[0]["manual_allowed_symbols_json"]
    assert "latest_terminal_roundtrip_id" in exported_p3[0]
    assert "latest_terminal_roundtrip_ts" in exported_p3[0]
    assert "next_probe_allowed_at" in exported_p3[0]
    assert '"manual_authorization_required": true' in exported_p3[0]["raw_payload_json"]
    assert redacted_manual_auth not in exported_p3[0]["raw_payload_json"]
    paper_rows = read_parquet_dataset(lake / "silver/v5_paper_strategy_run").to_dicts()
    assert paper_rows[0]["proposal_id"] == "SOL_F4_VOLUME_EXPANSION_PAPER_V1"
    assert paper_rows[0]["recommended_mode"] == "paper"


def test_ingest_parses_quant_lab_usage_legacy_report_paths(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {
            "raw/reports/quant_lab_usage.jsonl": (
                '{"ts":"2026-05-10T01:00:00Z","mode":"enforce"}\n'
            ),
            "reports/quant_lab_requests.jsonl": (
                '{"ts":"2026-05-10T01:01:00Z","path":"/v1/health","status_code":200}\n'
            ),
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.silver_rows["v5_quant_lab_usage"] == 1
    assert result.silver_rows["v5_quant_lab_request"] == 1


def test_ingest_parses_quant_lab_usage_from_large_gzip_paths(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260606T082013Z.tar.gz",
        {
            "raw/large/reports/quant_lab_usage.jsonl.gz": gzip.compress(
                b'{"ts":"2026-06-06T08:20:13Z","mode":"shadow","endpoint":"/v1/risk/live-permission"}\n'
            ),
            "raw/large/reports/quant_lab_requests.jsonl.gz": gzip.compress(
                b'{"ts":"2026-06-06T08:20:13Z","path":"/v1/strategy-opportunity-advisory","status_code":200,"success":true}\n'
            ),
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.silver_rows["v5_quant_lab_usage"] == 1
    assert result.silver_rows["v5_quant_lab_request"] == 1
    usage = read_parquet_dataset(lake / "silver/v5_quant_lab_usage").to_dicts()
    requests = read_parquet_dataset(lake / "silver/v5_quant_lab_request").to_dicts()
    assert usage[0]["source_path_inside_bundle"] == ("raw/large/reports/quant_lab_usage.jsonl.gz")
    assert requests[0]["source_path_inside_bundle"] == (
        "raw/large/reports/quant_lab_requests.jsonl.gz"
    )
    assert requests[0]["event_type"] == "request"


def test_ingest_quant_lab_fallback_ignores_successful_200_requests(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {
            "summaries/quant_lab_fallbacks.csv": (
                "path,status_code,success,fallback_used,diagnosis,error,error_type\n"
                "/v1/costs/estimate,200,true,false,request_not_ok,http_200,\n"
                "/v1/risk/live-permission,503,false,false,request_failed,http_503,timeout\n"
            ),
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")
    fallbacks = read_parquet_dataset(lake / "silver/v5_quant_lab_fallback")

    assert result.silver_rows["v5_quant_lab_fallback"] == 1
    assert fallbacks.height == 1
    assert fallbacks["status_code"][0] == "503"


def test_ingest_quant_lab_requests_separates_success_errors_and_actual_fallbacks(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {
            "raw/reports/quant_lab_requests.jsonl": (
                '{"event_type":"request","path":"/v1/costs/estimate",'
                '"status_code":200,"success":true,"ok":true,'
                '"fallback_used":false,"diagnosis":"request_not_ok",'
                '"error":"http_200"}\n'
                '{"event_type":"request","path":"/v1/risk/live-permission",'
                '"status_code":0,"success":false,"fallback_used":true,'
                '"error_type":"QuantLabTimeout","error":"timeout"}\n'
            ),
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")
    requests = read_parquet_dataset(lake / "silver/v5_quant_lab_request")
    fallbacks = read_parquet_dataset(lake / "silver/v5_quant_lab_fallback")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")
    execution = read_parquet_dataset(lake / "gold/v5_execution_quality_daily")

    assert result.silver_rows["v5_quant_lab_request"] == 2
    assert result.silver_rows["v5_quant_lab_fallback"] == 1
    assert requests.height == 2
    assert fallbacks.height == 1
    assert "http_200" not in fallbacks["raw_payload_json"][0]
    assert "QuantLabTimeout" in fallbacks["raw_payload_json"][0]
    assert health["request_success_count"][0] == 1
    assert health["request_error_count"][0] == 1
    assert health["actual_fallback_count"][0] == 1
    assert health["unique_request_count"][0] == 2
    assert health["unique_success_count"][0] == 1
    assert health["unique_error_count"][0] == 1
    assert health["unique_actual_fallback_count"][0] == 1
    assert health["fallback_rate"][0] == 0.5
    assert health["degraded_reason"][0] == "actual_fallback_present"
    assert execution["fallback_count"][0] == 1


def test_ingest_quant_lab_fallback_csv_reads_nested_raw_json_without_double_count(
    tmp_path,
):
    csv_buffer = StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=["event_type", "fallback_used", "diagnosis", "error", "raw_json"],
    )
    writer.writeheader()
    writer.writerow(
        {
            "event_type": "request",
            "fallback_used": "false",
            "diagnosis": "request_not_ok",
            "error": "http_200",
            "raw_json": json.dumps(
                {"status_code": 200, "success": True, "error_type": None},
            ),
        },
    )
    writer.writerow(
        {
            "event_type": "fallback",
            "fallback_used": "true",
            "diagnosis": "quant_lab_unavailable_allow_sell_only",
            "error": "QuantLabTimeout",
            "raw_json": json.dumps(
                {
                    "status_code": 0,
                    "success": False,
                    "fallback_used": True,
                    "error_type": "QuantLabTimeout",
                },
            ),
        },
    )
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {
            "raw/reports/quant_lab_requests.jsonl": (
                '{"event_type":"request","status_code":200,"success":true,'
                '"fallback_used":false,"diagnosis":"request_not_ok",'
                '"error":"http_200"}\n'
                '{"event_type":"request","status_code":0,"success":false,'
                '"fallback_used":true,"error_type":"QuantLabTimeout",'
                '"error":"timeout"}\n'
            ),
            "summaries/quant_lab_fallbacks.csv": csv_buffer.getvalue(),
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")
    fallbacks = read_parquet_dataset(lake / "silver/v5_quant_lab_fallback")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert result.silver_rows["v5_quant_lab_request"] == 2
    assert result.silver_rows["v5_quant_lab_fallback"] == 1
    assert fallbacks.height == 1
    assert all("http_200" not in payload for payload in fallbacks["raw_payload_json"])
    assert health["request_success_count"][0] == 1
    assert health["request_error_count"][0] == 1
    assert health["actual_fallback_count"][0] == 1
    assert health["fallback_rate"][0] == 0.5
    assert health["raw_imported_rows"][0] == 4
    assert health["unique_event_rows"][0] == 2
    assert health["duplicate_event_count"][0] == 2
    assert health["duplicate_event_rows"][0] == 2


def test_ingest_event_key_prefers_trading_event_id(tmp_path):
    first = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T230100Z.tar.gz",
        {
            "raw/reports/quant_lab_requests.jsonl": (
                '{"event_id":"ql-timeout-1","event_type":"request",'
                '"run_id":"run_a","ts":"2026-05-14T23:01:00Z",'
                '"path":"/v1/risk/live-permission","request_id":"request-a",'
                '"status_code":0,"success":false,"fallback_used":true,'
                '"error_type":"QuantLabTimeout","error":"timeout"}\n'
            ),
        },
    )
    second = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T231000Z.tar.gz",
        {
            "raw/reports/quant_lab_requests.jsonl": (
                '{"event_id":"ql-timeout-1","event_type":"request",'
                '"run_id":"run_b","ts":"2026-05-14T23:09:00Z",'
                '"path":"/v1/risk/live-permission","request_id":"request-b",'
                '"status_code":0,"success":false,"fallback_used":true,'
                '"error_type":"QuantLabTimeout","error":"timeout"}\n'
            ),
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(first, lake, tmp_path / "restricted", tmp_path / "redacted")
    ingest_v5_bundle(second, lake, tmp_path / "restricted", tmp_path / "redacted")

    requests = read_parquet_dataset(lake / "silver/v5_quant_lab_request")
    fallbacks = read_parquet_dataset(lake / "silver/v5_quant_lab_fallback")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert requests.height == 1
    assert fallbacks.height == 1
    assert requests["event_id"][0] == "ql-timeout-1"
    assert int(requests["source_count"][0]) == 2
    assert int(requests["last_seen_source_count"][0]) == 1
    assert health["unique_actual_fallback_count"][0] == 1
    assert health["duplicate_event_count"][0] == 1


def test_quant_lab_exact_duplicate_reingest_keeps_single_canonical_event(tmp_path):
    request = (
        '{"event_type":"request","strategy_id":"v5","run_id":"run_same",'
        '"ts":"2026-05-14T23:01:00Z","path":"/v1/costs/estimate",'
        '"request_id":"request-a","status_code":200,"success":true,'
        '"fallback_used":false,"latency_ms":10}\n'
    )
    first = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T230100Z.tar.gz",
        {"raw/reports/quant_lab_requests.jsonl": request},
    )
    second = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T231000Z.tar.gz",
        {
            "raw/reports/quant_lab_requests.jsonl": request,
            "summaries/window_summary.json": '{"marker":"different_bundle_same_event"}',
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(first, lake, tmp_path / "restricted", tmp_path / "redacted")
    ingest_v5_bundle(second, lake, tmp_path / "restricted", tmp_path / "redacted")

    requests = read_parquet_dataset(lake / "silver/v5_quant_lab_request")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert requests.height == 1
    row = requests.to_dicts()[0]
    assert int(row["source_count"]) == 2
    assert int(row["last_seen_source_count"]) == 1
    assert int(row["payload_hash_count"]) == 1
    assert row["conflicting_duplicate"] is False
    assert health["unique_event_rows"][0] == 1
    assert health["duplicate_event_rows"][0] == 0
    assert health["exact_duplicate_event_rows"][0] == 0
    assert health["conflicting_duplicate_event_rows"][0] == 0


def test_quant_lab_latency_only_duplicate_keeps_same_event_key_without_conflict(
    tmp_path,
):
    first_request = (
        '{"event_type":"request","strategy_id":"v5","run_id":"run_same",'
        '"ts":"2026-05-14T23:01:00Z","path":"/v1/costs/estimate",'
        '"request_id":"request-a","status_code":200,"success":true,'
        '"fallback_used":false,"latency_ms":10,'
        '"response_summary":{"decision":"ALLOW","latency_ms":10}}\n'
    )
    latency_only_change = (
        '{"event_type":"request","strategy_id":"v5","run_id":"run_same",'
        '"ts":"2026-05-14T23:01:00Z","path":"/v1/costs/estimate",'
        '"request_id":"request-a","status_code":200,"success":true,'
        '"fallback_used":false,"latency_ms":25,'
        '"response_summary":{"decision":"ALLOW","latency_ms":25}}\n'
    )
    first = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T230100Z.tar.gz",
        {"raw/reports/quant_lab_requests.jsonl": first_request},
    )
    second = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T231000Z.tar.gz",
        {"raw/reports/quant_lab_requests.jsonl": latency_only_change},
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(first, lake, tmp_path / "restricted", tmp_path / "redacted")
    ingest_v5_bundle(second, lake, tmp_path / "restricted", tmp_path / "redacted")

    requests = read_parquet_dataset(lake / "silver/v5_quant_lab_request")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert requests.height == 1
    row = requests.to_dicts()[0]
    assert int(row["source_count"]) == 2
    assert int(row["last_seen_source_count"]) == 1
    assert int(row["payload_hash_count"]) == 1
    assert row["conflicting_duplicate"] is False
    assert health["unique_event_rows"][0] == 1
    assert health["duplicate_event_rows"][0] == 0
    assert health["exact_duplicate_event_rows"][0] == 0
    assert health["conflicting_duplicate_event_rows"][0] == 0


def test_quant_lab_semantic_duplicate_keeps_cumulative_conflict_without_current_duplicate(
    tmp_path,
):
    first_request = (
        '{"event_type":"request","strategy_id":"v5","run_id":"run_same",'
        '"ts":"2026-05-14T23:01:00Z","path":"/v1/costs/estimate",'
        '"request_id":"request-a","status_code":200,"success":true,'
        '"fallback_used":false,"latency_ms":10,'
        '"response_summary":{"decision":"ALLOW","reason":"healthy"}}\n'
    )
    changed_payload_same_identity = (
        '{"event_type":"request","strategy_id":"v5","run_id":"run_same",'
        '"ts":"2026-05-14T23:01:00Z","path":"/v1/costs/estimate",'
        '"request_id":"request-a","status_code":200,"success":true,'
        '"fallback_used":false,"latency_ms":25,'
        '"response_summary":{"decision":"ABORT","reason":"risk_blocked"}}\n'
    )
    first = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T230100Z.tar.gz",
        {"raw/reports/quant_lab_requests.jsonl": first_request},
    )
    second = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T231000Z.tar.gz",
        {"raw/reports/quant_lab_requests.jsonl": changed_payload_same_identity},
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(first, lake, tmp_path / "restricted", tmp_path / "redacted")
    ingest_v5_bundle(second, lake, tmp_path / "restricted", tmp_path / "redacted")

    requests = read_parquet_dataset(lake / "silver/v5_quant_lab_request")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert requests.height == 1
    row = requests.to_dicts()[0]
    assert int(row["source_count"]) == 2
    assert int(row["last_seen_source_count"]) == 1
    assert int(row["payload_hash_count"]) == 2
    assert row["conflicting_duplicate"] is True
    assert health["unique_event_rows"][0] == 1
    assert health["duplicate_event_rows"][0] == 0
    assert health["conflicting_duplicate_event_rows"][0] == 0
    assert health["conflicting_duplicate_event_key_count"][0] == 0
    assert health["duplicate_explanation"][0] == "none"


def test_latency_variation_does_not_change_event_key_or_conflict_hash():
    first_payload = {
        "event_type": "request",
        "strategy_id": "v5",
        "run_id": "run_same",
        "ts": "2026-05-14T23:01:00Z",
        "path": "/v1/costs/estimate",
        "request_id": "request-a",
        "status_code": 200,
        "success": True,
        "fallback_used": False,
        "latency_ms": 10,
    }
    second_payload = dict(first_payload)
    second_payload["latency_ms"] = 25

    first_fields = _event_key_fields(
        first_payload,
        first_payload,
        default_event_type="request",
    )
    second_fields = _event_key_fields(
        second_payload,
        second_payload,
        default_event_type="request",
    )

    assert first_fields["raw_payload_hash"] == second_fields["raw_payload_hash"]
    assert _event_key_from_fields(first_fields) == _event_key_from_fields(second_fields)


def test_ingest_overlapping_bundles_deduplicate_quant_lab_timeout(tmp_path):
    request = (
        '{"event_type":"request","run_id":"run_20260514_23",'
        '"ts":"2026-05-14T23:01:00Z","path":"/v1/risk/live-permission",'
        '"status_code":0,"success":false,"fallback_used":true,'
        '"error_type":"QuantLabTimeout","error":"timeout"}\n'
    )
    fallback_csv = (
        "event_type,ts,path,status_code,success,fallback_used,error_type,error,"
        "symbol,side,intent\n"
        "fallback,2026-05-14T23:01:00Z,/v1/risk/live-permission,0,false,true,"
        "QuantLabTimeout,timeout,NOT-OBSERVABLE,not_observable,not_observable\n"
    )
    first = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T230100Z.tar.gz",
        {
            "raw/reports/quant_lab_requests.jsonl": request,
            "summaries/quant_lab_fallbacks.csv": fallback_csv,
        },
    )
    second = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T231000Z.tar.gz",
        {
            "raw/reports/quant_lab_requests.jsonl": request,
            "summaries/quant_lab_fallbacks.csv": fallback_csv,
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(first, lake, tmp_path / "restricted", tmp_path / "redacted")
    ingest_v5_bundle(second, lake, tmp_path / "restricted", tmp_path / "redacted")

    requests = read_parquet_dataset(lake / "silver/v5_quant_lab_request")
    fallbacks = read_parquet_dataset(lake / "silver/v5_quant_lab_fallback")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert requests.height == 1
    assert fallbacks.height == 1
    assert int(fallbacks["source_count"][0]) == 4
    assert int(requests["last_seen_source_count"][0]) == 1
    assert int(fallbacks["last_seen_source_count"][0]) == 2
    assert health["request_success_count"][0] == 0
    assert health["request_error_count"][0] == 1
    assert health["actual_fallback_count"][0] == 1
    assert health["unique_request_count"][0] == 1
    assert health["unique_error_count"][0] == 1
    assert health["unique_actual_fallback_count"][0] == 1
    assert health["fallback_rate"][0] == 1.0
    assert health["raw_imported_rows"][0] == 3
    assert health["unique_event_rows"][0] == 1
    assert health["duplicate_event_count"][0] == 2
    assert health["duplicate_event_rows"][0] == 2
    assert health["exact_duplicate_event_rows"][0] == 2
    assert health["conflicting_duplicate_event_rows"][0] == 0
    assert (
        health["duplicate_explanation"][0]
        == "exact_duplicate_reingest_or_overlapping_followup_bundle"
    )
    assert health["first_seen_bundle_ts"][0].isoformat().startswith("2026-05-14T23:01:00")
    assert health["last_seen_bundle_ts"][0].isoformat().startswith("2026-05-14T23:10:00")


def test_ingest_latest_quant_lab_requests_counts_unique_health(tmp_path):
    request_lines = []
    for index in range(140):
        request_lines.append(
            json.dumps(
                {
                    "event_type": "request",
                    "run_id": "run_20260514_latest",
                    "ts": f"2026-05-14T10:{index % 60:02d}:00Z",
                    "path": "/v1/costs/estimate",
                    "request_id": f"ok-{index}",
                    "status_code": 200,
                    "success": True,
                    "fallback_used": False,
                }
            )
        )
    for index in range(2):
        request_lines.append(
            json.dumps(
                {
                    "event_type": "request",
                    "run_id": "run_20260514_latest",
                    "ts": f"2026-05-14T11:{index:02d}:00Z",
                    "path": "/v1/risk/live-permission",
                    "request_id": f"err-{index}",
                    "status_code": 400,
                    "success": False,
                    "fallback_used": False,
                    "error_type": "",
                    "error": "http_400",
                }
            )
        )
    fallback_csv = (
        "event_type,ts,path,request_id,status_code,success,fallback_used,error_type,error\n"
        + "\n".join(
            (
                f"fallback,2026-05-14T12:{index:02d}:00Z,/v1/risk/live-permission,"
                f"fb-{index},0,false,true,QuantLabTimeout,timeout"
            )
            for index in range(4)
        )
        + "\n"
    )
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260514T120500Z.tar.gz",
        {
            "raw/reports/quant_lab_requests.jsonl": "\n".join(request_lines) + "\n",
            "summaries/quant_lab_fallbacks.csv": fallback_csv,
        },
    )
    lake = tmp_path / "lake"

    ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    health = read_parquet_dataset(lake / "gold/strategy_health_daily")
    fallbacks = read_parquet_dataset(lake / "silver/v5_quant_lab_fallback")

    assert health["request_success_count"][0] == 140
    assert health["request_error_count"][0] == 2
    assert health["actual_fallback_count"][0] == 4
    assert health["unique_request_count"][0] == 142
    assert health["unique_success_count"][0] == 140
    assert health["unique_error_count"][0] == 2
    assert health["unique_actual_fallback_count"][0] == 4
    assert health["fallback_rate"][0] == 4 / 142
    assert fallbacks.height == 4
    assert not any("http_200" in payload for payload in fallbacks["raw_payload_json"])


def test_ingest_parses_official_bundle_top_level_dir(tmp_path):
    bundle = make_v5_bundle_fixture(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        top_level_dir=True,
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    states = read_parquet_dataset(lake / "silver/v5_state_snapshot")
    issues = read_parquet_dataset(lake / "silver/v5_issue")
    decisions = read_parquet_dataset(lake / "silver/v5_decision_audit")
    assert result.validation.detected_files
    assert "kill_switch" in set(states["state_type"].to_list())
    assert issues.height == 1
    assert decisions["run_id"][0] == "run_001"
