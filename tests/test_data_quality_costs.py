from datetime import UTC, datetime

import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.ops.data_quality import run_data_quality


def test_cost_quality_flags_global_default_as_hard_fallback(tmp_path):
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
                    "sample_count": 0,
                    "total_cost_bps_p50": 25.0,
                    "total_cost_bps_p75": 25.0,
                    "total_cost_bps_p90": 25.0,
                    "cost_source": "global_default",
                    "fallback_level": "GLOBAL_DEFAULT",
                    "created_at": datetime(2026, 5, 28, 2, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )

    result = run_data_quality(
        lake,
        dataset_names=["cost_bucket_daily"],
        reference_at=datetime(2026, 5, 28, 3, tzinfo=UTC),
    ).to_dict()

    hard_fallback = next(
        check for check in result["checks"] if check["rule"] == "cost_hard_fallback_visibility"
    )
    assert hard_fallback["status"] == "FAIL"
    assert hard_fallback["observed_value"] == "1"


def test_cost_quality_allows_symbol_level_public_proxy_but_keeps_visibility(tmp_path):
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
                    "sample_count": 20,
                    "total_cost_bps_p50": 1.0,
                    "total_cost_bps_p75": 1.5,
                    "total_cost_bps_p90": 2.0,
                    "cost_source": "public_spread_proxy",
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "created_at": datetime(2026, 5, 28, 2, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )

    result = run_data_quality(
        lake,
        dataset_names=["cost_bucket_daily"],
        reference_at=datetime(2026, 5, 28, 3, tzinfo=UTC),
    ).to_dict()

    hard_fallback = next(
        check for check in result["checks"] if check["rule"] == "cost_hard_fallback_visibility"
    )
    assert hard_fallback["status"] == "PASS"
