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
    assert estimate.fee_bps == 2.0
    assert estimate.slippage_bps == 4.0
    assert estimate.spread_bps == 1.0
    assert estimate.total_cost_bps == 7.0
    assert estimate.cost_bps == 7.0
    assert estimate.fallback_level == "NONE"
    assert estimate.sample_count == 42
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
