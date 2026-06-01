from datetime import UTC, datetime

import polars as pl
from fastapi.testclient import TestClient

import quant_lab.api.main as api_main
from quant_lab.api.main import app
from quant_lab.costs.model import DEFAULT_FALLBACK_COST_BPS
from quant_lab.data.lake import write_parquet_dataset


def test_cost_estimate_api_reads_cost_bucket_daily_from_lake(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    regime="normal",
                    notional_bucket="1k-10k",
                    total_cost_bps_p75=5.25,
                    sample_count=42,
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={
            "symbol": "BTC-USDT",
            "regime": "normal",
            "notional_usdt": 5_000,
            "quantile": "p75",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "BTC-USDT"
    assert payload["regime"] == "normal"
    assert payload["quantile"] == "p75"
    assert payload["fee_bps"] == 1.5
    assert payload["fee_source"] == "actual_fills_bills"
    assert payload["slippage_bps"] == 3.0
    assert payload["slippage_source"] == "v5_order_lifecycle_arrival_mid"
    assert payload["spread_bps"] == 0.75
    assert payload["spread_source"] == "fresh_orderbook_p75"
    assert payload["total_cost_bps"] == 5.25
    assert payload["one_way_all_in_cost_bps"] == 5.25
    assert payload["roundtrip_all_in_cost_bps"] == 10.5
    assert payload["cost_quality"] == "actual"
    assert payload["cost_trusted_for_paper"] is True
    assert payload["cost_trusted_for_live"] is True
    assert payload["cost_trusted_for_live_canary"] is True
    assert payload["cost_trusted_for_live_scale"] is False
    assert payload["cost_trust_level"] == "CANARY"
    assert payload["fallback_level"] == "NONE"
    assert payload["source"] == "actual_okx_fills_and_bills"
    assert payload["normalized_symbol"] == "BTC-USDT"
    assert payload["cost_source"] == "actual_okx_fills_and_bills"
    assert payload["sample_size"] == 42
    assert payload["sample_count"] == 42
    assert payload["cost_model_version"] == "costs-2026-05-10"


def test_cost_estimate_api_reuses_server_cache_for_same_cost_key(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    api_main._COST_ESTIMATE_CACHE.clear()
    api_main._COST_BUCKET_CACHE.clear()
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    regime="normal",
                    notional_bucket="1k-10k",
                    total_cost_bps_p75=5.25,
                    sample_count=42,
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )
    client = TestClient(app)
    params = {
        "symbol": "BTC-USDT",
        "regime": "normal",
        "notional_usdt": 5_000,
        "quantile": "p75",
    }

    first = client.get("/v1/costs/estimate", params=params)

    def fail_cost_estimate(*_args, **_kwargs):
        raise AssertionError("same cost request should be served from server cache")

    monkeypatch.setattr(api_main, "_cost_bucket_rows_for_api", fail_cost_estimate)
    second = client.get("/v1/costs/estimate", params=params)
    api_main._COST_ESTIMATE_CACHE.clear()
    api_main._COST_BUCKET_CACHE.clear()

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["x-cost-cache-hit"] == "false"
    assert second.headers["x-cost-cache-hit"] == "true"
    assert second.headers["x-cost-bucket-cache-hit"] == "true"
    assert second.headers["x-quant-lab-api-cache-hit"] == "true"
    assert float(second.headers["x-quant-lab-lake-scan-ms"]) == 0.0
    assert second.json()["total_cost_bps"] == first.json()["total_cost_bps"]


def test_cost_estimate_api_uses_bucket_snapshot_when_request_cache_misses(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    api_main._COST_ESTIMATE_CACHE.clear()
    api_main._COST_BUCKET_CACHE.clear()
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    regime="normal",
                    notional_bucket="1k-10k",
                    total_cost_bps_p75=5.25,
                    sample_count=42,
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )
    client = TestClient(app)
    params = {
        "symbol": "BTC-USDT",
        "regime": "normal",
        "notional_usdt": 5_000,
        "quantile": "p75",
    }

    first = client.get("/v1/costs/estimate", params=params)
    api_main._COST_ESTIMATE_CACHE.clear()

    def fail_cost_bucket_loader(*_args, **_kwargs):
        raise AssertionError("cost bucket snapshot should avoid lake reload")

    monkeypatch.setattr(api_main, "_cost_bucket_rows_for_api", fail_cost_bucket_loader)
    second = client.get("/v1/costs/estimate", params={**params, "notional_usdt": 6_000})
    api_main._COST_ESTIMATE_CACHE.clear()
    api_main._COST_BUCKET_CACHE.clear()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers["x-cost-cache-hit"] == "false"
    assert second.headers["x-cost-bucket-cache-hit"] == "true"
    assert second.headers["x-quant-lab-api-cache-hit"] == "true"
    assert float(second.headers["x-quant-lab-lake-scan-ms"]) == 0.0
    assert second.json()["total_cost_bps"] == first.json()["total_cost_bps"]


def test_cost_estimate_api_uses_explicit_global_fallback_without_lake_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "empty-lake"))

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={
            "symbol": "BTC-USDT",
            "regime": "normal",
            "notional_usdt": 5_000,
            "quantile": "p90",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_cost_bps"] == DEFAULT_FALLBACK_COST_BPS
    assert payload["fallback_level"] == "GLOBAL_DEFAULT"
    assert payload["source"] == "global_default"
    assert payload["sample_count"] == 0
    assert payload["fee_source"] == "config_fee_bps"
    assert payload["spread_source"] == "global_default_config"
    assert payload["slippage_source"] == "config_slippage_bps"
    assert payload["one_way_all_in_cost_bps"] == DEFAULT_FALLBACK_COST_BPS
    assert payload["roundtrip_all_in_cost_bps"] == DEFAULT_FALLBACK_COST_BPS * 2.0
    assert payload["cost_quality"] == "global_default"
    assert payload["cost_trusted_for_paper"] is False
    assert payload["cost_trusted_for_live"] is False
    assert payload["cost_trust_level"] == "BLOCK"
    assert "source_global_default" in payload["cost_trust_block_reasons"]


def test_cost_estimate_api_can_match_requested_notional_bucket(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    regime="normal",
                    notional_bucket="0-1k",
                    total_cost_bps_p50=3.0,
                ),
                _cost_row(
                    symbol="BTC-USDT",
                    regime="normal",
                    notional_bucket="10k-100k",
                    total_cost_bps_p50=9.0,
                ),
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={
            "symbol": "BTC-USDT",
            "regime": "normal",
            "notional_usdt": 500,
            "quantile": "p50",
            "notional_bucket": "10k-100k",
        },
    )

    assert response.status_code == 200
    assert response.json()["total_cost_bps"] == 9.0


def test_cost_estimate_api_normalizes_slash_symbol(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BNB-USDT",
                    regime="public_proxy",
                    notional_bucket="all",
                    total_cost_bps_p75=2.25,
                    sample_count=11,
                    fallback_level="PUBLIC_SPREAD_PROXY",
                    source="public_spread_proxy",
                    created_at="2026-05-10T00:00:00Z",
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={
            "symbol": "BNB/USDT",
            "regime": "normal",
            "notional_usdt": 5_000,
            "quantile": "p75",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "BNB-USDT"
    assert payload["normalized_symbol"] == "BNB-USDT"
    assert payload["requested_regime"] == "normal"
    assert payload["matched_regime"] == "public_proxy"
    assert payload["source"] == "public_spread_proxy"
    assert payload["cost_source"] == "public_spread_proxy"
    assert payload["total_cost_bps"] == 2.25
    assert payload["selected_total_cost_bps"] == 2.25
    assert payload["fallback_level"] == "REGIME_FALLBACK;PUBLIC_SPREAD_PROXY"
    assert payload["fallback_reason"] == "cost_bucket_stale"
    assert payload["degraded_reason"] == "cost_bucket_stale"
    assert payload["requested_quantile"] == "p75"
    assert payload["degraded_cost_model"] is True


def test_cost_estimate_api_uses_same_symbol_public_proxy_for_trending_regime(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BNB-USDT",
                    regime="public_proxy",
                    notional_bucket="all",
                    total_cost_bps_p50=1.1,
                    total_cost_bps_p75=1.4969,
                    total_cost_bps_p90=2.2,
                    spread_bps_p50=1.1,
                    spread_bps_p75=1.4969,
                    spread_bps_p90=2.2,
                    sample_count=496,
                    fallback_level="PUBLIC_SPREAD_PROXY",
                    source="public_spread_proxy",
                    created_at=datetime.now(UTC).isoformat(),
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={
            "symbol": "BNB-USDT",
            "regime": "Trending",
            "notional_usdt": 5_000,
            "quantile": "p75",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "BNB-USDT"
    assert payload["normalized_symbol"] == "BNB-USDT"
    assert payload["requested_regime"] == "Trending"
    assert payload["matched_regime"] == "public_proxy"
    assert payload["source"] == "public_spread_proxy"
    assert payload["cost_source"] == "public_spread_proxy"
    assert payload["requested_quantile"] == "p75"
    assert payload["total_cost_bps_p75"] == 1.4969
    assert payload["selected_total_cost_bps"] == 1.4969
    assert payload["total_cost_bps"] == 1.4969
    assert payload["fee_bps"] == 10.0
    assert payload["fee_source"] == "config_fee_bps"
    assert payload["slippage_bps"] == 2.0
    assert payload["slippage_source"] == "config_slippage_bps"
    assert payload["spread_bps"] == 1.4969
    assert payload["spread_source"] == "fresh_orderbook_p75"
    assert payload["uncertainty_buffer_bps"] == 2.0
    assert payload["one_way_all_in_cost_bps"] == 15.4969
    assert payload["roundtrip_all_in_cost_bps"] == 30.9938
    assert payload["cost_quality"] == "public_proxy_only"
    assert payload["cost_trusted_for_paper"] is True
    assert payload["cost_trusted_for_live"] is False
    assert payload["cost_trust_level"] == "PAPER_ONLY"
    assert "source_public_proxy_only" in payload["cost_trust_block_reasons"]
    assert payload["sample_count"] == 496
    assert payload["fallback_level"] == "REGIME_FALLBACK;PUBLIC_SPREAD_PROXY"
    assert payload["fallback_reason"] == "no_matching_regime"
    assert payload["degraded_cost_model"] is True
    assert payload["source"] != "global_default"

    for symbol_variant in ["BNB/USDT", "BNBUSDT", "OKX:BNB-USDT"]:
        variant_response = TestClient(app).get(
            "/v1/costs/estimate",
            params={
                "symbol": symbol_variant,
                "regime": "Trending",
                "notional_usdt": 5_000,
                "quantile": "p75",
            },
        )
        assert variant_response.status_code == 200
        variant_payload = variant_response.json()
        assert variant_payload["normalized_symbol"] == "BNB-USDT"
        assert variant_payload["selected_total_cost_bps"] == payload["selected_total_cost_bps"]
        assert variant_payload["cost_source"] == "public_spread_proxy"
        assert variant_payload["source"] != "global_default"


def test_cost_estimate_api_uses_lazy_symbol_filtered_lake_read(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    regime="public_proxy",
                    notional_bucket="all",
                    total_cost_bps_p75=3.0,
                    source="public_spread_proxy",
                ),
                _cost_row(
                    symbol="BNB-USDT",
                    regime="public_proxy",
                    notional_bucket="all",
                    total_cost_bps_p75=1.75,
                    source="public_spread_proxy",
                ),
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    def fail_full_read(*args, **kwargs):
        raise AssertionError("cost estimate should lazy-filter cost_bucket_daily")

    monkeypatch.setattr("quant_lab.costs.model.read_parquet_dataset", fail_full_read)

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={
            "symbol": "OKX:BNB-USDT",
            "regime": "Trending",
            "notional_usdt": 5_000,
            "quantile": "p75",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_symbol"] == "BNB-USDT"
    assert payload["selected_total_cost_bps"] == 1.75
    assert payload["source"] == "public_spread_proxy"
    assert payload["source"] != "global_default"


def test_cost_estimate_api_unknown_symbol_uses_degraded_global_default(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BNB-USDT",
                    regime="public_proxy",
                    notional_bucket="all",
                    total_cost_bps_p75=1.4969,
                    fallback_level="PUBLIC_SPREAD_PROXY",
                    source="public_spread_proxy",
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={
            "symbol": "UNKNOWN-USDT",
            "regime": "Trending",
            "notional_usdt": 5_000,
            "quantile": "p75",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "global_default"
    assert payload["cost_source"] == "global_default"
    assert payload["fallback_level"] == "GLOBAL_DEFAULT"
    assert payload["fallback_reason"] == "symbol_missing"
    assert payload["degraded_reason"] == "global_default_cost"
    assert payload["degraded_cost_model"] is True
    assert payload["requested_quantile"] == "p75"


def test_cost_estimate_trust_blocks_stale_mixed_actual_proxy(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    sample_count=50,
                    source="mixed_actual_proxy",
                    fallback_level="SLIPPAGE_UNKNOWN",
                    created_at="2026-05-10T00:00:00Z",
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={"symbol": "BTC-USDT", "regime": "normal", "notional_usdt": 5_000},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cost_trust_level"] == "BLOCK"
    assert payload["cost_trusted_for_live"] is False
    assert payload["cost_trusted_for_live_canary"] is False
    assert "stale_cost_bucket" in payload["cost_trust_block_reasons"]


def test_cost_estimate_trust_allows_fresh_mixed_actual_proxy_canary(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    sample_count=30,
                    source="mixed_actual_proxy",
                    fallback_level="SLIPPAGE_UNKNOWN",
                    created_at=datetime.now(UTC).isoformat(),
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={"symbol": "BTC-USDT", "regime": "normal", "notional_usdt": 5_000},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cost_trust_level"] == "CANARY"
    assert payload["cost_trusted_for_live"] is True
    assert payload["cost_trusted_for_live_canary"] is True
    assert payload["cost_trusted_for_live_scale"] is False


def test_cost_estimate_trust_scales_actual_fills_with_actual_slippage(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    sample_count=100,
                    source="actual_fills",
                    fallback_level="NONE",
                    created_at=datetime.now(UTC).isoformat(),
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={"symbol": "BTC-USDT", "regime": "normal", "notional_usdt": 5_000},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["slippage_source"] == "v5_order_lifecycle_arrival_mid"
    assert payload["cost_trust_level"] == "SCALE_READY"
    assert payload["cost_trusted_for_live_canary"] is True
    assert payload["cost_trusted_for_live_scale"] is True


def test_cost_estimate_trust_actual_fills_sample_30_is_canary_not_scale(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    sample_count=30,
                    source="actual_fills",
                    fallback_level="NONE",
                    created_at=datetime.now(UTC).isoformat(),
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={"symbol": "BTC-USDT", "regime": "normal", "notional_usdt": 5_000},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cost_trust_level"] == "CANARY"
    assert payload["cost_trusted_for_live_canary"] is True
    assert payload["cost_trusted_for_live_scale"] is False


def test_cost_estimate_trust_regime_fallback_is_not_scale_ready(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    write_parquet_dataset(
        pl.DataFrame(
            [
                _cost_row(
                    symbol="BTC-USDT",
                    regime="realized",
                    sample_count=100,
                    source="actual_fills",
                    fallback_level="NONE",
                    created_at=datetime.now(UTC).isoformat(),
                )
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )

    response = TestClient(app).get(
        "/v1/costs/estimate",
        params={"symbol": "BTC-USDT", "regime": "Trending", "notional_usdt": 5_000},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback_level"] == "REGIME_FALLBACK"
    assert payload["cost_trust_level"] in {"CANARY", "PAPER_ONLY"}
    assert payload["cost_trusted_for_live_scale"] is False
    assert "fallback_not_live_safe" in payload["cost_trust_block_reasons"]


def _cost_row(**overrides):
    row = {
        "day": "2026-05-10",
        "symbol": "BTC-USDT",
        "regime": "normal",
        "event_type": "trade",
        "notional_bucket": "1k-10k",
        "sample_count": 10,
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
    row.update(overrides)
    return row
