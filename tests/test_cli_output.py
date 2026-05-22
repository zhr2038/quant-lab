from datetime import UTC, datetime

from quant_lab.cli import (
    _compact_lake_health_payload,
    _compact_v5_sync_payload,
    _compact_v5_telemetry_payload,
)
from quant_lab.strategy_telemetry.models import V5TelemetryAnalysisResult


def test_compact_v5_telemetry_payload_is_journald_friendly():
    result = V5TelemetryAnalysisResult(
        date="2026-05-23",
        status="WARNING",
        latest_bundle_ts=datetime(2026, 5, 23, 1, tzinfo=UTC),
        quant_lab_mode="shadow",
        unique_request_count=100,
        request_success_count=98,
        request_error_count=2,
        actual_fallback_count=1,
        fallback_rate=0.01,
        duplicate_rate=0.9,
        warnings=[f"warning-{index}" for index in range(10)],
        critical_reasons=["critical"],
        next_actions=["review"],
    )

    payload = _compact_v5_telemetry_payload(result)

    assert payload["status"] == "WARNING"
    assert payload["latest_bundle_ts"] == "2026-05-23T01:00:00Z"
    assert payload["request_success_count"] == 98
    assert payload["warning_count"] == 10
    assert len(payload["warnings"]) == 5
    assert "router_reason_top" not in payload
    assert "latest_bundle_sha256" not in payload


def test_compact_v5_sync_payload_omits_nested_bundle_details():
    payload = _compact_v5_sync_payload(
        {
            "pull": {
                "pulled_files": ["latest.tar.gz"],
                "skipped_files": ["old.tar.gz"],
                "warnings": ["pull-warning"],
                "command_summary": ["ssh", "rsync"],
            },
            "inbox": {
                "processed": [
                    {
                        "bundle_name": "latest.tar.gz",
                        "validation": {"detected_files": ["a", "b"]},
                    }
                ],
                "skipped_files": ["already.tar.gz"],
                "warnings": ["max_scan_bundles_limit_applied:1_of_100", "inbox-warning"],
            },
            "analysis": {
                "status": "WARNING",
                "latest_bundle_ts": "2026-05-23T01:00:00Z",
                "warnings": ["large nested payload"],
            },
            "analysis_after_sync": False,
            "max_bundles": 1,
            "remote_max_files": 1,
            "max_scan_bundles": 1,
            "include_historical_outcomes": False,
        }
    )

    assert payload["pulled_count"] == 1
    assert payload["skipped_pull_count"] == 1
    assert payload["processed_count"] == 1
    assert payload["processed_bundles"] == ["latest.tar.gz"]
    assert payload["analysis_status"] == "WARNING"
    assert payload["warnings"] == ["pull-warning", "inbox-warning"]
    assert payload["scan_limited"] is True
    assert "pull" not in payload
    assert "inbox" not in payload
    assert "analysis" not in payload


def test_compact_lake_health_payload_omits_full_rows():
    payload = _compact_lake_health_payload(
        {
            "dataset_count": 2,
            "total_parquet_files": 123,
            "warning_count": 1,
            "rows": [
                {
                    "dataset": "orderbook_snapshot",
                    "parquet_file_count": 100,
                    "partition_dir_count": 20,
                    "status": "OK",
                    "path": "/large/path/that/should/not/be/printed",
                },
                {
                    "dataset": "api_request_metrics",
                    "parquet_file_count": 23,
                    "partition_dir_count": 4,
                    "status": "WARN",
                    "warning": "small-file ratio is high",
                },
            ],
        }
    )

    assert payload["dataset_count"] == 2
    assert payload["total_parquet_files"] == 123
    assert payload["warnings"] == [
        {
            "dataset": "api_request_metrics",
            "status": "WARN",
            "warning": "small-file ratio is high",
            "parquet_file_count": 23,
        }
    ]
    assert payload["top_file_count_datasets"][0]["dataset"] == "orderbook_snapshot"
    assert "rows" not in payload
    assert "path" not in str(payload)
