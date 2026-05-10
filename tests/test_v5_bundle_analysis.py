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
                    "level": payload.get("level", ""),
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
