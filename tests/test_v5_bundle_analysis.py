from datetime import UTC, datetime

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.strategy_telemetry.analyze import analyze_v5_telemetry
from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle
from tests.v5_bundle_fixture import make_v5_bundle_fixture


def test_analyze_flags_kill_switch_critical(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_state(lake, "kill_switch", {"enabled": True})

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.status == "CRITICAL"
    assert "kill_switch_enabled" in result.critical_reasons


def test_analyze_flags_reconcile_failure(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_state(lake, "reconcile_status", {"ok": False})

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.status == "CRITICAL"
    assert "reconcile_not_ok" in result.critical_reasons


def test_analyze_flags_missing_recent_bundle(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake, bundle_ts=datetime(2026, 5, 8, tzinfo=UTC))

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.status == "WARNING"
    assert "latest bundle is older than 24 hours" in result.warnings


def test_analyze_detects_high_score_blocked_issue(tmp_path):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz")
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.validation.valid is True
    analysis = analyze_v5_telemetry(lake, date="2026-05-10")
    assert "review high_score_blocked_matured_without_label" in analysis.next_actions


def test_analyze_uses_window_summary_metrics_before_row_counts(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _run_summary_row(
                    "summaries/window_summary.json",
                    {
                        "run_count": 4,
                        "recent_24h_decision_audit_count": 5,
                        "latest_24h_trade_count": 6,
                        "last_72h_trade_count": 7,
                        "last_72h_roundtrip_count": 8,
                        "open_position_count": 9,
                        "dust_residual_position_count": 2,
                        "high_issue_count": 3,
                        "medium_issue_count": 1,
                    },
                ),
                _run_summary_row("raw/recent_runs/run_001/summary.json", {"run_id": "run_001"}),
            ]
        ),
        lake / "silver/v5_run_summary",
    )
    write_parquet_dataset(
        pl.DataFrame([_event_row("old") for _ in range(20)]),
        lake / "silver/v5_decision_audit",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.run_count_72h == 4
    assert result.decision_audit_count_24h == 5
    assert result.trade_count_24h == 6
    assert result.trade_count_72h == 7
    assert result.roundtrip_count_72h == 8
    assert result.open_position_count == 9
    assert result.dust_residual_position_count == 2
    assert result.high_issue_count == 3
    assert result.medium_issue_count == 1


def test_analyze_counts_recent_rows_by_bundle_ts_when_window_summary_missing(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _event_row("recent", bundle_ts=datetime(2026, 5, 10, 20, tzinfo=UTC)),
                _event_row("old", bundle_ts=datetime(2026, 5, 8, 20, tzinfo=UTC)),
            ]
        ),
        lake / "silver/v5_trade_event",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.trade_count_24h == 1
    assert result.trade_count_72h == 2


def test_analyze_reads_auto_risk_current_level(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_state(lake, "auto_risk_eval", {"current_level": "ELEVATED"})

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.auto_risk_level == "ELEVATED"


def _write_manifest(lake, bundle_ts=datetime(2026, 5, 10, 14, 2, 49, tzinfo=UTC)):
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "abc",
                    "bundle_name": "v5_live_followup_bundle_20260510T140249Z.tar.gz",
                    "bundle_ts": bundle_ts,
                    "ingest_ts": datetime(2026, 5, 10, 14, 3, tzinfo=UTC),
                    "schema_version": "test",
                }
            ]
        ),
        lake / "bronze/strategy_telemetry/v5/bundle_manifest",
    )


def _write_state(lake, state_type, payload):
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "abc",
                    "bundle_name": "bundle.tar.gz",
                    "bundle_ts": datetime(2026, 5, 10, tzinfo=UTC),
                    "ingest_ts": datetime(2026, 5, 10, tzinfo=UTC),
                    "schema_version": "test",
                    "source_path_inside_bundle": f"raw/state/{state_type}.json",
                    "run_id": None,
                    "row_index": 0,
                    "state_type": state_type,
                    "ok": payload.get("ok"),
                    "enabled": payload.get("enabled"),
                    "level": payload.get("current_level") or payload.get("level", ""),
                    "raw_payload_json": __import__("json").dumps(payload),
                }
            ]
        ),
        lake / "silver/v5_state_snapshot",
    )


def test_analysis_writes_gold(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    result = analyze_v5_telemetry(lake, date="2026-05-10")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert result.date == "2026-05-10"
    assert health.height == 1


def _run_summary_row(source_path, payload):
    return {
        "strategy": "v5",
        "bundle_sha256": "abc",
        "bundle_name": "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        "bundle_ts": datetime(2026, 5, 10, 14, 2, 49, tzinfo=UTC),
        "ingest_ts": datetime(2026, 5, 10, 14, 3, tzinfo=UTC),
        "schema_version": "test",
        "source_path_inside_bundle": source_path,
        "run_id": "run_001",
        "row_index": 0,
        "raw_payload_json": __import__("json").dumps(payload),
    }


def _event_row(label, bundle_ts=datetime(2026, 5, 10, 14, 2, 49, tzinfo=UTC)):
    return {
        "strategy": "v5",
        "bundle_sha256": f"abc-{label}",
        "bundle_name": f"{label}.tar.gz",
        "bundle_ts": bundle_ts,
        "ingest_ts": bundle_ts,
        "schema_version": "test",
        "source_path_inside_bundle": f"raw/recent_runs/{label}/decision_audit.json",
        "run_id": label,
        "row_index": 0,
        "raw_payload_json": "{}",
    }
