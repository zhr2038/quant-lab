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
            "max_s": 7.0,
        }
    ]
