from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_lab.ingest.v5_reports import (
    inspect_v5_reports,
    iter_v5_decision_audits,
    iter_v5_run_summaries,
    load_latest_v5_cost_stats,
)


def test_v5_report_inspection_with_temp_fixture(tmp_path):
    reports = tmp_path / "reports"
    (reports / "runs" / "run_001").mkdir(parents=True)
    (reports / "cost_stats").mkdir(parents=True)

    (reports / "alpha_snapshot.json").write_text('{"alpha_count": 1}', encoding="utf-8")
    (reports / "runs" / "run_001" / "decision_audit.json").write_text(
        '{"decision": "ALLOW"}', encoding="utf-8"
    )
    (reports / "runs" / "run_001" / "summary.json").write_text(
        '{"status": "complete"}', encoding="utf-8"
    )
    (reports / "cost_stats" / "daily_cost_stats_20260216.json").write_text(
        '{"day": "2026-02-16", "buckets": []}', encoding="utf-8"
    )

    inspection = inspect_v5_reports(reports)

    assert inspection.reports_dir == str(reports)
    assert inspection.has_alpha_snapshot is True
    assert inspection.decision_audit_count == 1
    assert inspection.summary_count == 1
    assert inspection.cost_stats_file_count == 1
    assert str(inspection.latest_cost_stats_day) == "2026-02-16"
    assert Path(inspection.latest_cost_stats_path).name == "daily_cost_stats_20260216.json"
    assert inspection.warnings == []

    latest_cost_stats = load_latest_v5_cost_stats(reports)
    assert latest_cost_stats == {"day": "2026-02-16", "buckets": []}

    decision_audits = list(iter_v5_decision_audits(reports))
    assert len(decision_audits) == 1
    assert decision_audits[0]["run_id"] == "run_001"
    assert Path(decision_audits[0]["source_path"]).name == "decision_audit.json"
    assert decision_audits[0]["loaded_at"].tzinfo == UTC
    assert isinstance(decision_audits[0]["loaded_at"], datetime)
    assert decision_audits[0]["raw"] == {"decision": "ALLOW"}

    summaries = list(iter_v5_run_summaries(reports))
    assert len(summaries) == 1
    assert summaries[0]["run_id"] == "run_001"
    assert Path(summaries[0]["source_path"]).name == "summary.json"
    assert summaries[0]["raw"] == {"status": "complete"}


def test_v5_report_inspection_allows_missing_optional_files(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()

    inspection = inspect_v5_reports(reports)

    assert inspection.has_alpha_snapshot is False
    assert inspection.decision_audit_count == 0
    assert inspection.summary_count == 0
    assert inspection.cost_stats_file_count == 0
    assert inspection.latest_cost_stats_day is None
    assert inspection.latest_cost_stats_path is None
    assert inspection.warnings == []
    assert load_latest_v5_cost_stats(reports) is None
    assert list(iter_v5_decision_audits(reports)) == []
    assert list(iter_v5_run_summaries(reports)) == []


def test_v5_report_inspection_warns_on_invalid_json(tmp_path):
    reports = tmp_path / "reports"
    (reports / "runs" / "run_001").mkdir(parents=True)
    (reports / "cost_stats").mkdir(parents=True)

    (reports / "alpha_snapshot.json").write_text("{invalid", encoding="utf-8")
    (reports / "runs" / "run_001" / "decision_audit.json").write_text(
        "{invalid", encoding="utf-8"
    )
    (reports / "runs" / "run_001" / "summary.json").write_text(
        '{"status": "complete"}', encoding="utf-8"
    )
    (reports / "cost_stats" / "daily_cost_stats_20260216.json").write_text(
        "{invalid", encoding="utf-8"
    )

    inspection = inspect_v5_reports(reports)

    assert inspection.has_alpha_snapshot is True
    assert inspection.decision_audit_count == 1
    assert inspection.summary_count == 1
    assert inspection.cost_stats_file_count == 1
    assert str(inspection.latest_cost_stats_day) == "2026-02-16"
    assert len(inspection.warnings) == 3
    assert all("Invalid JSON" in warning for warning in inspection.warnings)
    assert list(iter_v5_decision_audits(reports)) == []
    assert load_latest_v5_cost_stats(reports) is None


def test_v5_report_inspection_missing_reports_dir_error(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="V5 reports directory does not exist"):
        inspect_v5_reports(missing)


def test_static_v5_fixture_is_available_for_cli_acceptance():
    reports = Path("tests/fixtures/v5_reports")

    inspection = inspect_v5_reports(reports)

    assert inspection.has_alpha_snapshot is True
    assert inspection.decision_audit_count == 1
    assert inspection.summary_count == 1
    assert inspection.cost_stats_file_count == 2
    assert str(inspection.latest_cost_stats_day) == "2026-05-10"
