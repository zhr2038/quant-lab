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
