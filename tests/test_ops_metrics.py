from quant_lab.ops.metrics import api_metrics_summary, record_api_request


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
