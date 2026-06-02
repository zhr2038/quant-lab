import time
from datetime import UTC, datetime, timedelta

import polars as pl

import quant_lab.ops.metrics as metrics_module
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


def test_api_request_metrics_default_flush_window_limits_small_files(monkeypatch):
    monkeypatch.delenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", raising=False)
    monkeypatch.delenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", raising=False)

    assert metrics_module._api_metrics_flush_rows() == 1_000
    assert metrics_module._api_metrics_flush_seconds() == 300.0


def test_api_request_metrics_can_flush_asynchronously(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "1")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_ASYNC_FLUSH", "1")
    scheduled: list[str] = []
    monkeypatch.setattr(
        metrics_module,
        "_schedule_api_request_metrics_flush",
        lambda lake_root: scheduled.append(str(lake_root)),
    )

    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/strategy-opportunity-advisory",
        status_code=200,
        duration_seconds=0.05,
    )

    assert scheduled == [str(lake)]
    assert list((lake / "bronze" / "api_request_metrics").rglob("*.parquet")) == []
    summary = api_metrics_summary(lake)
    assert summary["request_count"] == 1
    assert summary["by_path"]["/v1/strategy-opportunity-advisory"] == 1


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


def test_api_request_metrics_records_cache_and_payload_fields(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "1")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")

    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/strategy-opportunity-advisory",
        status_code=200,
        duration_seconds=0.01,
        cache_hit=True,
        rows_returned=233,
        response_bytes=12345,
        lake_scan_ms=0.0,
        serialize_ms=3.2,
        source_signature_ms=1.7,
        response_cache_hit=True,
        error_type=None,
    )

    summary = api_metrics_summary(lake)

    assert summary["cache_hit_count"] == 1
    assert summary["rows_returned_total"] == 233.0
    assert summary["response_bytes_total"] == 12345.0
    assert summary["serialize_ms_total"] == 3.2
    assert summary["source_signature_ms_total"] == 1.7
    assert summary["response_cache_hit_count"] == 1
    assert summary["by_error_type"] == {}


def test_api_request_metrics_summary_unions_evolved_parquet_schema(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "100")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")
    dataset = lake / "bronze" / "api_request_metrics"
    dataset.mkdir(parents=True)
    request_ts = datetime(2026, 6, 2, tzinfo=UTC)
    pl.DataFrame(
        [
            {
                "day": "2026-06-02",
                "request_ts": request_ts,
                "method": "GET",
                "path": "/v1/strategy-opportunity-advisory/v5-compact",
                "status_code": 200,
                "duration_ms": 10.0,
                "client_host": "127.0.0.1",
                "user_agent": "old-schema",
            }
        ]
    ).write_parquet(dataset / "old_schema.parquet")
    pl.DataFrame(
        [
            {
                "day": "2026-06-02",
                "request_ts": request_ts + timedelta(seconds=1),
                "method": "GET",
                "path": "/v1/strategy-opportunity-advisory/v5-compact",
                "status_code": 200,
                "duration_ms": 20.0,
                "client_host": "127.0.0.1",
                "user_agent": "new-schema",
                "cache_hit": True,
                "rows_returned": 201,
                "response_bytes": 313521,
                "source_signature_ms": 0.4,
                "response_cache_hit": True,
            }
        ]
    ).write_parquet(dataset / "new_schema.parquet")

    summary = api_metrics_summary(lake, day="2026-06-02")
    path_summary = summary["latency_by_path_ms"]["/v1/strategy-opportunity-advisory/v5-compact"]

    assert summary["request_count"] == 2
    assert summary["rows_returned_total"] == 201.0
    assert summary["response_bytes_total"] == 313521.0
    assert summary["source_signature_ms_total"] == 0.4
    assert summary["response_cache_hit_count"] == 1
    assert path_summary["rows_returned_total"] == 201.0
    assert path_summary["response_bytes_total"] == 313521.0


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


def test_api_request_metrics_do_not_partition_by_path(tmp_path, monkeypatch):
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
        path="/v1/strategy-opportunity-advisory",
        status_code=200,
        duration_seconds=0.02,
    )

    dataset = lake / "bronze" / "api_request_metrics"
    files = list(dataset.rglob("*.parquet"))
    assert len(files) == 1
    assert not any(part.startswith("path=") for file in files for part in file.parts)

    summary = api_metrics_summary(lake)

    assert summary["by_path"]["/v1/health"] == 1
    assert summary["by_path"]["/v1/strategy-opportunity-advisory"] == 1


def test_api_request_metrics_async_timer_flushes_without_threshold(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "100")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_ASYNC_FLUSH", "1")
    monkeypatch.setattr(metrics_module, "_api_metrics_flush_seconds", lambda: 0.01)

    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/strategy-opportunity-advisory",
        status_code=200,
        duration_seconds=0.01,
    )

    deadline = time.monotonic() + 1.0
    while (
        not list((lake / "bronze" / "api_request_metrics").rglob("*.parquet"))
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)

    summary = api_metrics_summary(lake)

    assert summary["request_count"] == 1
    assert summary["by_path"]["/v1/strategy-opportunity-advisory"] == 1


def test_api_request_metrics_summary_can_filter_recent_window(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "1")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")
    now = datetime.now(UTC)
    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/old",
        status_code=200,
        duration_seconds=0.100,
        request_ts=now - timedelta(hours=2),
    )
    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/current",
        status_code=200,
        duration_seconds=0.010,
        request_ts=now,
    )

    summary = api_metrics_summary(lake, since_minutes=60)

    assert summary["request_count"] == 1
    assert summary["by_path"] == {"/v1/current": 1}


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


def test_job_run_summary_can_filter_recent_window(tmp_path):
    lake = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(hours=2)
    record_job_run(
        lake_root=lake,
        job_name="old-job",
        status="succeeded",
        started_at=old,
        finished_at=old + timedelta(seconds=30),
    )
    record_job_run(
        lake_root=lake,
        job_name="current-job",
        status="succeeded",
        started_at=now,
        finished_at=now + timedelta(seconds=3),
    )

    summary = job_run_summary(lake, since_minutes=60)

    assert summary["run_count"] == 1
    assert summary["jobs"][0]["job_name"] == "current-job"
