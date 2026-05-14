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


def test_ingest_parses_quant_lab_usage_files(tmp_path):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz")
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.silver_rows["v5_quant_lab_usage"] == 1
    assert result.silver_rows["v5_quant_lab_request"] == 1
    assert result.silver_rows["v5_quant_lab_compliance"] == 1
    assert result.silver_rows["v5_quant_lab_cost_usage"] == 1
    assert result.silver_rows["v5_quant_lab_fallback"] == 1
    assert read_parquet_dataset(lake / "silver/v5_quant_lab_usage").height == 1
    assert read_parquet_dataset(lake / "silver/v5_quant_lab_compliance").height == 1


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
    assert health["duplicate_event_rows"][0] == 2


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
    assert health["fallback_rate"][0] == 1.0
    assert health["raw_imported_rows"][0] == 6
    assert health["unique_event_rows"][0] == 1
    assert health["duplicate_event_rows"][0] == 5


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
