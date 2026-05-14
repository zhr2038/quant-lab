from quant_lab.costs.model import (
    DEFAULT_FALLBACK_COST_BPS,
    CostBucket,
    estimate_cost_bps,
    estimate_cost_from_cost_bucket_daily_rows,
)


def test_cost_model_uses_exact_matching_bucket():
    estimate = estimate_cost_bps(
        "BTCUSDT",
        "normal",
        10_000,
        [
            CostBucket(
                bucket_id="btc-normal",
                symbol="BTCUSDT",
                regime="normal",
                min_notional_usdt=0,
                max_notional_usdt=50_000,
                cost_bps=4.5,
            )
        ],
    )

    assert estimate.cost_bps == 4.5
    assert estimate.symbol == "BTC-USDT"
    assert estimate.normalized_symbol == "BTC-USDT"
    assert estimate.bucket_id == "btc-normal"
    assert estimate.fallback_level == "NONE"


def test_cost_model_fallback_is_explicit_when_no_bucket_matches():
    estimate = estimate_cost_bps("ETHUSDT", "volatile", 10_000, [])

    assert estimate.cost_bps == DEFAULT_FALLBACK_COST_BPS
    assert estimate.bucket_id is None
    assert estimate.fallback_level == "DEFAULT_FALLBACK"


def test_cost_model_can_fallback_to_symbol_bucket():
    estimate = estimate_cost_bps(
        "BTCUSDT",
        "volatile",
        10_000,
        [
            {
                "bucket_id": "btc-any-regime",
                "symbol": "BTCUSDT",
                "regime": None,
                "min_notional_usdt": 0,
                "max_notional_usdt": None,
                "cost_bps": 6.0,
            }
        ],
    )

    assert estimate.cost_bps == 6.0
    assert estimate.bucket_id == "btc-any-regime"
    assert estimate.fallback_level == "REGIME_FALLBACK"


def test_cost_bucket_daily_estimate_uses_requested_quantile():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="normal",
        notional_usdt=5_000,
        quantile="p90",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "BTC-USDT",
                "regime": "normal",
                "event_type": "trade",
                "notional_bucket": "1k-10k",
                "sample_count": 42,
                "fee_bps_p50": 1.0,
                "fee_bps_p75": 1.5,
                "fee_bps_p90": 2.0,
                "slippage_bps_p50": 2.0,
                "slippage_bps_p75": 3.0,
                "slippage_bps_p90": 4.0,
                "spread_bps_p50": 0.5,
                "spread_bps_p75": 0.75,
                "spread_bps_p90": 1.0,
                "total_cost_bps_p50": 3.5,
                "total_cost_bps_p75": 5.25,
                "total_cost_bps_p90": 7.0,
                "fallback_level": "actual_okx_fills_and_bills",
                "source": "actual_okx_fills_and_bills",
                "cost_model_version": "costs-2026-05-10",
            }
        ],
    )

    assert estimate.quantile == "p90"
    assert estimate.requested_quantile == "p90"
    assert estimate.fee_bps == 2.0
    assert estimate.slippage_bps == 4.0
    assert estimate.spread_bps == 1.0
    assert estimate.total_cost_bps == 7.0
    assert estimate.total_cost_bps_p50 == 3.5
    assert estimate.total_cost_bps_p75 == 5.25
    assert estimate.total_cost_bps_p90 == 7.0
    assert estimate.cost_bps == 7.0
    assert estimate.fallback_level == "NONE"
    assert estimate.fallback_reason == "NONE"
    assert estimate.cost_source == "actual_okx_fills_and_bills"
    assert estimate.degraded_cost_model is False
    assert estimate.sample_count == 42
    assert estimate.sample_size == 42
    assert estimate.cost_model_version == "costs-2026-05-10"


def test_cost_bucket_daily_estimate_uses_global_fallback_when_no_bucket_matches():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="DOGE-USDT",
        regime="volatile",
        notional_usdt=5_000,
        quantile="p75",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "GLOBAL",
                "regime": "global",
                "event_type": "trade",
                "notional_bucket": "all",
                "sample_count": 8,
                "fee_bps_p50": 1.0,
                "fee_bps_p75": 2.0,
                "fee_bps_p90": 3.0,
                "slippage_bps_p50": 1.0,
                "slippage_bps_p75": 2.0,
                "slippage_bps_p90": 3.0,
                "spread_bps_p50": 1.0,
                "spread_bps_p75": 2.0,
                "spread_bps_p90": 3.0,
                "total_cost_bps_p50": 3.0,
                "total_cost_bps_p75": 6.0,
                "total_cost_bps_p90": 9.0,
                "fallback_level": "public_spread_proxy",
                "source": "public_spread_proxy",
            }
        ],
    )

    assert estimate.total_cost_bps == 6.0
    assert estimate.fallback_level == "GLOBAL_BUCKET_FALLBACK;public_spread_proxy"
    assert estimate.source == "public_spread_proxy"


def test_cost_bucket_daily_estimate_preserves_proxy_fallback_on_exact_match():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="public_proxy",
        notional_usdt=5_000,
        quantile="p75",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "BTC-USDT",
                "regime": "public_proxy",
                "event_type": "spread_proxy",
                "notional_bucket": "all",
                "sample_count": 8,
                "fee_bps_p75": 0.0,
                "slippage_bps_p75": 0.0,
                "spread_bps_p75": 2.0,
                "total_cost_bps_p75": 2.0,
                "fallback_level": "FEE_MISSING;SLIPPAGE_UNKNOWN;PUBLIC_SPREAD_PROXY",
                "source": "public_spread_proxy",
            }
        ],
    )

    assert estimate.total_cost_bps == 2.0
    assert estimate.fallback_level == "FEE_MISSING;SLIPPAGE_UNKNOWN;PUBLIC_SPREAD_PROXY"
    assert estimate.source == "public_spread_proxy"


def test_cost_bucket_daily_estimate_returns_explicit_global_default_without_rows():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="normal",
        notional_usdt=5_000,
        quantile="p50",
        rows=[],
    )

    assert estimate.total_cost_bps == DEFAULT_FALLBACK_COST_BPS
    assert estimate.fallback_level == "GLOBAL_DEFAULT"
    assert estimate.source == "global_default"
    assert estimate.sample_count == 0


def test_cost_bucket_daily_estimate_normalizes_strategy_symbol_to_cost_bucket():
    rows = [
        {
            "day": "2026-05-10",
            "symbol": "BNB-USDT",
            "regime": "public_proxy",
            "event_type": "spread_proxy",
            "notional_bucket": "all",
            "sample_count": 8,
            "fee_bps_p50": 0.0,
            "fee_bps_p75": 0.0,
            "fee_bps_p90": 0.0,
            "slippage_bps_p50": 0.0,
            "slippage_bps_p75": 0.0,
            "slippage_bps_p90": 0.0,
            "spread_bps_p50": 1.0,
            "spread_bps_p75": 2.0,
            "spread_bps_p90": 3.0,
            "total_cost_bps_p50": 1.0,
            "total_cost_bps_p75": 2.0,
            "total_cost_bps_p90": 3.0,
            "fallback_level": "FEE_MISSING;SLIPPAGE_UNKNOWN;PUBLIC_SPREAD_PROXY",
            "source": "public_spread_proxy",
            "cost_model_version": "costs-2026-05-10",
            "created_at": "2026-05-10T01:00:00Z",
        }
    ]

    slash = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BNB/USDT",
        regime="public_proxy",
        notional_usdt=5_000,
        quantile="p75",
        rows=rows,
    )
    hyphen = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BNB-USDT",
        regime="public_proxy",
        notional_usdt=5_000,
        quantile="p75",
        rows=rows,
    )

    assert slash.symbol == "BNB-USDT"
    assert slash.normalized_symbol == "BNB-USDT"
    assert slash.total_cost_bps == 2.0
    assert slash.source == "public_spread_proxy"
    assert slash.cost_source == "public_spread_proxy"
    assert slash.sample_size == 8
    assert slash.as_of_ts is not None
    assert hyphen.model_dump() == slash.model_dump()


def test_cost_bucket_daily_estimate_can_fallback_to_symbol_bucket_across_regime():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BNB/USDT",
        regime="normal",
        notional_usdt=5_000,
        quantile="p75",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "BNB-USDT",
                "regime": "public_proxy",
                "event_type": "spread_proxy",
                "notional_bucket": "all",
                "sample_count": 8,
                "total_cost_bps_p75": 2.0,
                "fallback_level": "PUBLIC_SPREAD_PROXY",
                "source": "public_spread_proxy",
            }
        ],
    )

    assert estimate.symbol == "BNB-USDT"
    assert estimate.total_cost_bps == 2.0
    assert estimate.source == "public_spread_proxy"
    assert estimate.fallback_level == "REGIME_FALLBACK;PUBLIC_SPREAD_PROXY"
    assert estimate.requested_regime == "normal"
    assert estimate.matched_regime == "public_proxy"
    assert estimate.cost_source == "public_spread_proxy"
    assert estimate.selected_total_cost_bps == 2.0
    assert estimate.fallback_reason in {"no_matching_regime", "cost_bucket_stale"}
    assert estimate.degraded_cost_model is True


def test_cost_bucket_daily_estimate_unknown_symbol_uses_global_default():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="UNKNOWN/USDT",
        regime="public_proxy",
        notional_usdt=5_000,
        quantile="p75",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "BNB-USDT",
                "regime": "public_proxy",
                "event_type": "spread_proxy",
                "notional_bucket": "all",
                "sample_count": 8,
                "total_cost_bps_p75": 2.0,
                "fallback_level": "PUBLIC_SPREAD_PROXY",
                "source": "public_spread_proxy",
            }
        ],
    )

    assert estimate.source == "global_default"
    assert estimate.cost_source == "global_default"
    assert estimate.fallback_level == "GLOBAL_DEFAULT"
    assert estimate.fallback_reason == "symbol_missing"
    assert estimate.degraded_reason == "global_default_cost"
    assert estimate.degraded_cost_model is True
    assert estimate.total_cost_bps == DEFAULT_FALLBACK_COST_BPS


def test_cost_bucket_daily_estimate_prefers_actual_fills_over_public_proxy():
    rows = [
        {
            "day": "2026-05-10",
            "symbol": "BNB-USDT",
            "regime": "realized",
            "event_type": "actual_fill",
            "notional_bucket": "all",
            "sample_count": 3,
            "total_cost_bps_p75": 4.0,
            "fallback_level": "SLIPPAGE_UNKNOWN",
            "source": "actual_fills",
        },
        {
            "day": "2026-05-10",
            "symbol": "BNB-USDT",
            "regime": "realized",
            "event_type": "spread_proxy",
            "notional_bucket": "all",
            "sample_count": 300,
            "total_cost_bps_p75": 1.0,
            "fallback_level": "PUBLIC_SPREAD_PROXY",
            "source": "public_spread_proxy",
        },
    ]

    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BNB/USDT",
        regime="realized",
        notional_usdt=5_000,
        quantile="p75",
        rows=rows,
    )

    assert estimate.total_cost_bps == 4.0
    assert estimate.source == "actual_fills"
    assert estimate.sample_size == 3
