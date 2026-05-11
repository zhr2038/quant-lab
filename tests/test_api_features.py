from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from quant_lab.api.main import app
from quant_lab.data.lake import write_market_bars
from quant_lab.features.publish import publish_features


def test_features_latest_returns_empty_rows_when_dataset_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))
    client = TestClient(app)

    response = client.get("/v1/features/latest?feature_set=core")

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == []
    assert payload["warnings"] == ["feature_value dataset is missing or empty"]


def test_features_latest_returns_latest_features(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    write_market_bars(
        lake,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": start + timedelta(hours=index),
                "open": 100.0 + index,
                "high": 102.0 + index,
                "low": 98.0 + index,
                "close": 100.0 + index,
                "volume": 10.0,
                "quote_volume": 1000.0 + index,
                "source": "test",
                "ingest_ts": start + timedelta(hours=index, minutes=1),
            }
            for index in range(30)
        ],
    )
    publish_features(lake, symbols=["BTC-USDT"])
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    client = TestClient(app)

    response = client.get(
        "/v1/features/latest",
        params={
            "feature_set": "core",
            "symbols": "BTC-USDT",
            "timeframe": "1H",
            "feature_names": "close_return_1,range_bps",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["feature_set"] == "core"
    assert payload["feature_version"] == "v0.1"
    assert payload["rows"][0]["symbol"] == "BTC-USDT"
    assert set(payload["rows"][0]["features"]) == {"close_return_1", "range_bps"}
