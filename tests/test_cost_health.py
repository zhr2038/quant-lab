import json

import polars as pl

from quant_lab.costs.health import (
    build_cost_health_daily,
    publish_cost_health_daily,
    read_cost_health_daily,
    summarize_cost_api_usage,
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
        api_global_default_count=1,
        api_symbol_proxy_hit_count=2,
        api_regime_fallback_count=3,
        api_degraded_cost_count=4,
        api_cost_usage_rows=5,
    )

    assert row.status == "OK"
    assert row.actual_rows == 1
    assert row.mixed_rows == 0
    assert row.fallback_ratio == 0
    assert row.hard_fallback_ratio == 0
    assert row.soft_fallback_ratio == 0
    assert row.api_global_default_count == 1
    assert row.api_symbol_proxy_hit_count == 2
    assert row.api_regime_fallback_count == 3
    assert row.api_degraded_cost_count == 4
    assert row.api_cost_usage_rows == 5


def test_cost_health_summarizes_api_cost_usage_rows():
    stats = summarize_cost_api_usage(
        pl.DataFrame(
            [
                {
                    "cost_source": "global_default",
                    "fallback_level": "GLOBAL_DEFAULT",
                    "degraded_cost_model": "true",
                },
                {
                    "cost_source": "public_spread_proxy",
                    "fallback_level": "NONE",
                    "degraded_cost_model": "false",
                },
                {
                    "raw_payload_json": (
                        '{"response": {"cost_source": "public_spread_proxy", '
                        '"fallback_level": "REGIME_FALLBACK", '
                        '"degraded_cost_model": true}}'
                    )
                },
            ]
        )
    )

    assert stats["api_cost_usage_rows"] == 3
    assert stats["api_global_default_count"] == 1
    assert stats["api_symbol_proxy_hit_count"] == 2
    assert stats["api_regime_fallback_count"] == 1
    assert stats["api_degraded_cost_count"] == 2


def test_cost_health_counts_actual_fills_source():
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "BNB-USDT",
                    "source": "actual_fills",
                    "sample_count": 30,
                    "fallback_level": "NONE",
                    "cost_model_version": "costs-v1",
                },
                {
                    "day": "2026-05-10",
                    "symbol": "SOL-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 100,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "cost_model_version": "costs-v1",
                },
                {
                    "day": "2026-05-10",
                    "symbol": "GLOBAL",
                    "source": "global_default",
                    "sample_count": 0,
                    "fallback_level": "GLOBAL_DEFAULT",
                    "cost_model_version": "costs-v1",
                },
            ]
        ),
        day="2026-05-10",
        min_sample_count=30,
        expected_symbols=["BNB-USDT", "SOL-USDT"],
    )

    assert row.actual_rows == 1
    assert row.mixed_rows == 0
    assert row.proxy_rows == 1
    assert row.global_default_rows == 1
    assert row.hard_fallback_count == 1
    assert row.hard_fallback_ratio == 1 / 3
    assert row.soft_fallback_count == 1
    assert row.soft_fallback_ratio == 1 / 3
    assert row.proxy_only_count == 1
    assert row.global_default_count == 1
    assert row.symbols_with_actual_cost == ["BNB-USDT"]
    assert row.symbols_with_proxy_only == ["SOL-USDT"]
    assert row.symbols_proxy_only == ["SOL-USDT"]
    assert row.actual_sample_count_by_symbol == {"BNB-USDT": 30}


def test_cost_health_counts_mixed_actual_proxy_as_private_cost_available():
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "BNB-USDT",
                    "source": "mixed_actual_proxy",
                    "sample_count": 2,
                    "fallback_level": "SAMPLE_TOO_SMALL;SLIPPAGE_UNKNOWN;SPREAD_PROXY",
                    "cost_model_version": "costs-v1",
                },
                {
                    "day": "2026-05-10",
                    "symbol": "SOL-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 100,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "cost_model_version": "costs-v1",
                },
            ]
        ),
        day="2026-05-10",
        min_sample_count=30,
        expected_symbols=["BNB-USDT", "SOL-USDT"],
        private_fill_rows=2,
        private_bill_rows=1,
    )

    assert row.status == "WARNING"
    assert row.actual_rows == 0
    assert row.mixed_rows == 1
    assert row.hard_fallback_ratio == 0
    assert row.soft_fallback_ratio == 1
    assert row.proxy_only_count == 1
    assert row.symbols_with_mixed_cost == ["BNB-USDT"]
    assert row.symbols_with_proxy_only == ["SOL-USDT"]
    checks = json.loads(row.data_quality_checks_json)
    assert checks["private_fills_present_but_actual_cost_zero"] is True
    assert checks["actual_cost_symbol_coverage"] == "1/2"


def test_cost_health_flags_private_fills_without_actual_cost():
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "BNB-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 10,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "cost_model_version": "costs-v1",
                }
            ]
        ),
        day="2026-05-10",
        min_sample_count=30,
        expected_symbols=["BNB-USDT"],
        private_fill_rows=2,
    )

    assert row.status == "CRITICAL"
    assert "private_fills_present_but_actual_cost_zero" in json.loads(row.warnings_json)


def test_cost_health_keeps_cost_probe_private_fills_advisory():
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-06-25",
                    "symbol": "BNB-USDT",
                    "source": "bootstrap_cost_probe",
                    "sample_count": 2,
                    "fallback_level": "COST_PROBE_ONLY;SAMPLE_TOO_SMALL",
                    "cost_model_version": "costs-v1",
                }
            ]
        ),
        day="2026-06-25",
        min_sample_count=30,
        expected_symbols=["BNB-USDT"],
        private_fill_rows=2,
    )

    checks = json.loads(row.data_quality_checks_json)
    warnings = json.loads(row.warnings_json)
    assert row.status == "WARNING"
    assert checks["private_fills_present_but_actual_cost_zero"] is True
    assert "private_fills_present_but_actual_cost_zero" not in warnings


def test_cost_health_flags_v5_trades_without_actual_cost():
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "BNB-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 10,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "cost_model_version": "costs-v1",
                }
            ]
        ),
        day="2026-05-10",
        min_sample_count=30,
        expected_symbols=["BNB-USDT"],
        v5_trade_rows=2,
    )

    assert row.status == "CRITICAL"
    warnings = json.loads(row.warnings_json)
    checks = json.loads(row.data_quality_checks_json)
    assert "trades_present_but_not_in_cost_model" in warnings
    assert checks["trades_present_but_not_in_cost_model"] is False
    assert checks["fee_missing_rate"] == "0/2"


def test_cost_health_keeps_cost_probe_v5_trades_advisory():
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-06-25",
                    "symbol": "BNB-USDT",
                    "source": "bootstrap_cost_probe",
                    "sample_count": 2,
                    "fallback_level": "COST_PROBE_ONLY;SAMPLE_TOO_SMALL",
                    "cost_model_version": "costs-v1",
                }
            ]
        ),
        day="2026-06-25",
        min_sample_count=30,
        expected_symbols=["BNB-USDT"],
        v5_trade_rows=2,
    )

    checks = json.loads(row.data_quality_checks_json)
    warnings = json.loads(row.warnings_json)
    assert row.status == "WARNING"
    assert checks["trades_present_but_not_in_cost_model"] is True
    assert "trades_present_but_not_in_cost_model" not in warnings


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
    assert row.hard_fallback_ratio == 0
    assert row.soft_fallback_ratio == 1
    assert row.proxy_only_count == 1
    assert "all_rows_public_spread_proxy" in json.loads(row.warnings_json)


def test_cost_health_proxy_only_missing_research_symbols_is_warning():
    row = build_cost_health_daily(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "BTC-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 12,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                },
                {
                    "day": "2026-05-10",
                    "symbol": "ETH-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 12,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                },
            ]
        ),
        day="2026-05-10",
        min_sample_count=30,
        expected_symbols=["BTC-USDT", "ETH-USDT", "ALLO-USDT"],
    )

    assert row.status == "WARNING"
    assert row.symbols_missing_cost == ["ALLO-USDT"]
    warnings = json.loads(row.warnings_json)
    assert "all_rows_public_spread_proxy" in warnings
    assert "symbols_missing_cost" in warnings


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
    assert row.hard_fallback_count == 1
    assert row.hard_fallback_ratio == 1
    assert row.global_default_count == 1


def test_cost_health_publish_and_read(tmp_path):
    lake = tmp_path / "lake"
    row = build_cost_health_daily(pl.DataFrame(), day="2026-05-10", min_sample_count=30)

    publish_cost_health_daily(lake, row)
    payload = read_cost_health_daily(lake, day="2026-05-10")

    assert payload["status"] == "CRITICAL"
    assert payload["rows"] == 1
    assert payload["hard_fallback_ratio"] == 1.0
    assert payload["warnings"] == ["cost_bucket_daily empty"]
