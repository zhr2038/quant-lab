import json
from datetime import UTC, datetime
from types import SimpleNamespace

import polars as pl
from typer.testing import CliRunner

from quant_lab.cli import app
from quant_lab.data.lake import write_parquet_dataset

runner = CliRunner()


def test_data_quality_command_outputs_full_json(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-28",
                    "symbol": "BNB-USDT",
                    "regime": "Trending",
                    "event_type": "entry",
                    "notional_bucket": "all",
                    "sample_count": 30,
                    "total_cost_bps_p50": 1.0,
                    "total_cost_bps_p75": 1.5,
                    "total_cost_bps_p90": 2.0,
                    "cost_source": "mixed_actual_proxy",
                    "fallback_level": "NONE",
                    "created_at": datetime(2026, 5, 28, 2, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )

    result = runner.invoke(
        app,
        [
            "data-quality",
            "--lake-root",
            str(lake),
            "--dataset",
            "cost_bucket_daily",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dataset_count"] == 1
    assert payload["checks"]
    assert any(check["rule"] == "cost_negative_bps" for check in payload["checks"])


def test_export_daily_cli_does_not_refresh_risk_permission_by_default(monkeypatch, tmp_path):
    captured = {}

    def fake_run_with_job_metrics(*, lake_root, job_name, func):
        captured["job_name"] = job_name
        captured["lake_root"] = lake_root
        return func()

    def fake_export_daily_pack(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model_dump_json=lambda indent=None: json.dumps({"ok": True}))

    monkeypatch.setattr("quant_lab.cli.run_with_job_metrics", fake_run_with_job_metrics)
    monkeypatch.setattr("quant_lab.cli.export_daily_pack", fake_export_daily_pack)

    result = runner.invoke(
        app,
        [
            "export-daily",
            "--date",
            "2026-06-03",
            "--lake-root",
            str(tmp_path / "lake"),
            "--out-dir",
            str(tmp_path / "exports"),
        ],
    )

    assert result.exit_code == 0
    assert captured["job_name"] == "export-daily"
    assert captured["refresh_risk_permission"] is False


def test_data_quality_command_dataset_filter_runs_only_requested_dataset(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "venue": "okx",
                    "symbol": "BTC-USDT",
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": datetime(2026, 5, 28, 1, tzinfo=UTC),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1.0,
                    "source": "fixture",
                    "ingest_ts": datetime(2026, 5, 28, 1, 1, tzinfo=UTC),
                    "is_closed": True,
                }
            ]
        ),
        lake / "silver" / "market_bar",
    )

    result = runner.invoke(
        app,
        [
            "data-quality",
            "--lake-root",
            str(lake),
            "--dataset",
            "market_bar",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dataset_count"] == 1
    assert {check["dataset"] for check in payload["checks"]} == {"market_bar"}


def test_data_quality_command_compact_output_includes_top_failing_checks(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-28",
                    "symbol": "BNB-USDT",
                    "regime": "Trending",
                    "event_type": "entry",
                    "notional_bucket": "all",
                    "sample_count": 30,
                    "total_cost_bps_p50": -1.0,
                    "total_cost_bps_p75": 1.5,
                    "total_cost_bps_p90": 2.0,
                    "cost_source": "mixed_actual_proxy",
                    "fallback_level": "NONE",
                    "created_at": datetime(2026, 5, 28, 2, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )

    result = runner.invoke(
        app,
        [
            "data-quality",
            "--lake-root",
            str(lake),
            "--dataset",
            "cost_bucket_daily",
            "--compact-output",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert "checks" not in payload
    assert payload["failing_checks"][0]["rule"] == "cost_negative_bps"


def test_lake_health_include_quality_compact_outputs_quality_summary(tmp_path):
    lake = tmp_path / "lake"

    result = runner.invoke(
        app,
        [
            "lake-health",
            "--lake-root",
            str(lake),
            "--dataset",
            "market_bar",
            "--include-quality",
            "--compact-output",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "data_quality" in payload
    assert payload["data_quality"]["dataset_count"] == 1
    assert "checks" not in payload["data_quality"]
