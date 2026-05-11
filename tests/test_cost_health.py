import json

import polars as pl

from quant_lab.costs.health import (
    build_cost_health_daily,
    publish_cost_health_daily,
    read_cost_health_daily,
)


def test_cost_health_ok_with_actual_rows(tmp_path):
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "BTC-USDT",
                    "source": "actual_okx_fills_and_bills",
                    "sample_count": 30,
                    "fallback_level": "NONE",
                    "cost_model_version": "costs-v1",
                }
            ]
        ),
        day="2026-05-10",
        min_sample_count=30,
        expected_symbols=["BTC-USDT"],
    )

    assert row.status == "OK"
    assert row.actual_rows == 1
    assert row.fallback_ratio == 0


def test_cost_health_proxy_only_is_warning():
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "BTC-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 12,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                }
            ]
        ),
        day="2026-05-10",
        min_sample_count=30,
    )

    assert row.status == "WARNING"
    assert row.proxy_rows == 1
    assert row.fallback_ratio == 1
    assert "all_rows_public_spread_proxy" in json.loads(row.warnings_json)


def test_cost_health_global_default_is_critical():
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "GLOBAL",
                    "source": "global_default",
                    "sample_count": 0,
                    "fallback_level": "GLOBAL_DEFAULT",
                }
            ]
        ),
        day="2026-05-10",
        min_sample_count=30,
    )

    assert row.status == "CRITICAL"
    assert row.global_default_rows == 1


def test_cost_health_publish_and_read(tmp_path):
    lake = tmp_path / "lake"
    row = build_cost_health_daily(pl.DataFrame(), day="2026-05-10", min_sample_count=30)

    publish_cost_health_daily(lake, row)
    payload = read_cost_health_daily(lake, day="2026-05-10")

    assert payload["status"] == "CRITICAL"
    assert payload["rows"] == 1
    assert payload["warnings"] == ["cost_bucket_daily empty"]
