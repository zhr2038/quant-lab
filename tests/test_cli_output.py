from datetime import UTC, datetime

from quant_lab.cli import _compact_v5_telemetry_payload
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
