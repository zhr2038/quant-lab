import json
from datetime import UTC, datetime

import polars as pl

from quant_lab.cli import data_quality_command
from quant_lab.data.lake import write_parquet_dataset


def test_data_quality_command_outputs_full_json(tmp_path, capsys):
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

    data_quality_command(
        lake_root=lake,
        dataset="cost_bucket_daily",
        compact_output=False,
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset_count"] == 1
    assert payload["checks"]
    assert any(check["rule"] == "cost_negative_bps" for check in payload["checks"])


def test_data_quality_command_compact_output_includes_top_failing_checks(tmp_path, capsys):
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

    data_quality_command(
        lake_root=lake,
        dataset="cost_bucket_daily",
        compact_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "FAIL"
    assert payload["failing_checks"][0]["rule"] == "cost_negative_bps"
