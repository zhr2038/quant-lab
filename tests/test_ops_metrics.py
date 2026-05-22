from datetime import UTC, datetime, timedelta

from quant_lab.ops.metrics import (
    api_metrics_summary,
    job_run_summary,
    record_api_request,
    record_job_run,
)


def test_api_request_metrics_are_micro_batched(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "100")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")

    for _ in range(10):
        record_api_request(
            lake_root=lake,
            method="GET",
            path="/v1/health",
            status_code=200,
            duration_seconds=0.01,
        )

    assert list((lake / "bronze" / "api_request_metrics").rglob("*.parquet")) == []

    summary = api_metrics_summary(lake)

    files = list((lake / "bronze" / "api_request_metrics").rglob("*.parquet"))
    assert len(files) == 1
    assert summary["request_count"] == 10
    assert summary["by_path"]["/v1/health"] == 10


def test_api_request_metrics_summary_uses_lazy_aggregation(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "2")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")

    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/health",
        status_code=200,
        duration_seconds=0.01,
    )
    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/health",
        status_code=500,
        duration_seconds=0.03,
    )

    def fail_full_read(*args, **kwargs):
        raise AssertionError("summary should not full-read api_request_metrics")

    monkeypatch.setattr("quant_lab.ops.metrics.read_parquet_dataset", fail_full_read)

    summary = api_metrics_summary(lake)

    assert summary["request_count"] == 2
    assert summary["by_path"]["/v1/health"] == 2
    assert summary["by_status_code"]["500"] == 1
    assert summary["latency_ms"]["max"] == 30.0
    assert summary["latency_by_path_ms"]["/v1/health"]["count"] == 2
    assert summary["latency_by_path_ms"]["/v1/health"]["max"] == 30.0
    assert summary["latency_by_path_ms"]["/v1/health"]["server_error_count"] == 1
    assert summary["slow_paths"][0]["path"] == "/v1/health"


def test_api_request_metrics_summary_reports_slowest_paths(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "10")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")

    for duration in [0.005, 0.006, 0.007]:
        record_api_request(
            lake_root=lake,
            method="GET",
            path="/v1/fast",
            status_code=200,
            duration_seconds=duration,
        )
    for duration in [0.100, 0.300, 0.500]:
        record_api_request(
            lake_root=lake,
            method="GET",
            path="/v1/slow",
            status_code=200,
            duration_seconds=duration,
        )
    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/missing",
        status_code=404,
        duration_seconds=0.020,
    )

    summary = api_metrics_summary(lake)

    assert summary["latency_by_path_ms"]["/v1/fast"]["count"] == 3
    assert summary["latency_by_path_ms"]["/v1/slow"]["max"] == 500.0
    assert summary["latency_by_path_ms"]["/v1/missing"]["client_error_count"] == 1
    assert summary["slow_paths"][0]["path"] == "/v1/slow"
    assert summary["slow_paths"][0]["p95"] >= 300.0


def test_job_run_summary_uses_lazy_aggregation(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    started = datetime(2026, 5, 23, 1, 0, tzinfo=UTC)
    record_job_run(
        lake_root=lake,
        job_name="export-daily",
        status="succeeded",
        started_at=started,
        finished_at=started + timedelta(seconds=5),
    )
    record_job_run(
        lake_root=lake,
        job_name="export-daily",
        status="failed",
        started_at=started + timedelta(minutes=10),
        finished_at=started + timedelta(minutes=10, seconds=7),
        error=RuntimeError("boom"),
    )

    def fail_full_read(*args, **kwargs):
        raise AssertionError("summary should not full-read job_run_history")

    monkeypatch.setattr("quant_lab.ops.metrics.read_parquet_dataset", fail_full_read)

    summary = job_run_summary(lake)

    assert summary["run_count"] == 2
    assert summary["jobs"] == [
        {
            "job_name": "export-daily",
            "run_count": 2,
            "failure_count": 1,
            "avg_s": 6.0,
            "p95_s": 7.0,
            "max_s": 7.0,
            "latest_duration_s": 7.0,
            "latest_status": "failed",
            "latest_finished_at": started + timedelta(minutes=10, seconds=7),
        }
    ]


def test_job_run_summary_day_auto_uses_current_utc_day(tmp_path):
    lake = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=2)
    record_job_run(
        lake_root=lake,
        job_name="sync-v5-telemetry",
        status="succeeded",
        started_at=old,
        finished_at=old + timedelta(seconds=30),
    )
    record_job_run(
        lake_root=lake,
        job_name="sync-v5-telemetry",
        status="succeeded",
        started_at=now,
        finished_at=now + timedelta(seconds=3),
    )

    summary = job_run_summary(lake, day="auto")

    assert summary["run_count"] == 1
    assert summary["jobs"][0]["job_name"] == "sync-v5-telemetry"
    assert summary["jobs"][0]["latest_duration_s"] == 3.0
