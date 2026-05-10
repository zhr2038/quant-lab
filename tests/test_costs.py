from quant_lab.costs.model import DEFAULT_FALLBACK_COST_BPS, CostBucket, estimate_cost_bps


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

