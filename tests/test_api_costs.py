import polars as pl
from fastapi.testclient import TestClient

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
    assert payload["slippage_bps"] == 3.0
    assert payload["spread_bps"] == 0.75
    assert payload["total_cost_bps"] == 5.25
    assert payload["fallback_level"] == "NONE"
    assert payload["source"] == "actual_okx_fills_and_bills"
    assert payload["normalized_symbol"] == "BTC-USDT"
    assert payload["cost_source"] == "actual_okx_fills_and_bills"
    assert payload["sample_size"] == 42
    assert payload["sample_count"] == 42
    assert payload["cost_model_version"] == "costs-2026-05-10"


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
    assert payload["source"] == "public_spread_proxy"
    assert payload["total_cost_bps"] == 2.25
    assert payload["fallback_level"] == "REGIME_FALLBACK;PUBLIC_SPREAD_PROXY"


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
