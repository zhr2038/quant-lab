import csv
import json
from io import StringIO

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle
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
                "candidate_id,run_id,ts_utc,symbol,strategy_candidate,cost_source\n"
                "cand_1,run_001,2026-05-10T01:00:00Z,SOL/USDT,"
                "f4_volume_expansion_entry,public_spread_proxy\n"
            ),
            "raw/reports/quant_lab_requests.jsonl": (
                '{"ts":"2026-05-10T01:00:00Z","endpoint":"/v1/costs/estimate",'
                '"status_code":200,"success":true}\n'
            ),
            "summaries/high_score_blocked_outcomes.csv": (
                "candidate_id,symbol,label_4h_net_bps,label_status\n"
                "hist_1,SOL-USDT,12,complete\n"
            ),
            "summaries/alt_impulse_shadow_outcomes.csv": (
                "candidate_id,symbol,label_4h_net_bps,label_status\n"
                "shadow_1,SOL-USDT,10,complete\n"
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
    assert requests.height == 1
    assert historical.is_empty()
    assert shadow.is_empty()
    assert any("skipped_historical_outcome_file" in warning for warning in result.warnings)
    redacted_root = (
        tmp_path
        / "redacted"
        / "2026-05-10"
        / result.bundle_sha256
        / "redacted_files"
    )
    assert not (redacted_root / "summaries/high_score_blocked_outcomes.csv").exists()
    assert not (redacted_root / "summaries/alt_impulse_shadow_outcomes.csv").exists()


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
    assert read_parquet_dataset(lake / "silver/v5_quant_lab_usage").height == 1
    assert read_parquet_dataset(lake / "silver/v5_quant_lab_compliance").height == 1
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
    assert health["unique_actual_fallback_count"][0] == 1
    assert health["duplicate_event_count"][0] == 3


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
    assert health["request_success_count"][0] == 0
    assert health["request_error_count"][0] == 1
    assert health["actual_fallback_count"][0] == 1
    assert health["unique_request_count"][0] == 1
    assert health["unique_error_count"][0] == 1
    assert health["unique_actual_fallback_count"][0] == 1
    assert health["fallback_rate"][0] == 1.0
    assert health["raw_imported_rows"][0] == 6
    assert health["unique_event_rows"][0] == 1
    assert health["duplicate_event_count"][0] == 5
    assert health["duplicate_event_rows"][0] == 5
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
