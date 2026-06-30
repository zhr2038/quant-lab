from datetime import UTC, datetime

from fastapi.testclient import TestClient

from quant_lab.api.main import app
from quant_lab.data.lake import write_market_bars


def test_market_bars_api_returns_expected_bars_from_temp_lake(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    write_market_bars(
        lake_root,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": datetime(2026, 5, 10, 1, tzinfo=UTC),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
                "volume": 12.0,
                "quote_volume": 1260.0,
                "source": "test_fixture",
                "ingest_ts": datetime(2026, 5, 10, 2, tzinfo=UTC),
            },
            {
                "venue": "okx",
                "symbol": "ETH-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": datetime(2026, 5, 10, 1, tzinfo=UTC),
                "open": 200.0,
                "high": 220.0,
                "low": 190.0,
                "close": 210.0,
                "volume": 4.0,
                "quote_volume": 840.0,
                "source": "test_fixture",
                "ingest_ts": datetime(2026, 5, 10, 2, tzinfo=UTC),
            },
        ],
    )
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake_root))

    response = TestClient(app).get(
        "/v1/market/bars",
        params={
            "venue": "okx",
            "symbol": "BTC-USDT",
            "timeframe": "1H",
            "start": "2026-05-10T00:00:00Z",
            "end": "2026-05-10T02:00:00Z",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["venue"] == "okx"
    assert payload[0]["symbol"] == "BTC-USDT"
    assert payload[0]["close"] == 105.0
    assert payload[0]["ts"] == "2026-05-10T01:00:00Z"


def test_market_bars_api_normalizes_venue_case(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    write_market_bars(
        lake_root,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": datetime(2026, 5, 10, 1, tzinfo=UTC),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
                "volume": 12.0,
                "quote_volume": 1260.0,
                "source": "test_fixture",
                "ingest_ts": datetime(2026, 5, 10, 2, tzinfo=UTC),
            }
        ],
    )
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake_root))

    response = TestClient(app).get(
        "/v1/market/bars",
        params={
            "venue": "OKX",
            "symbol": "BTC/USDT",
            "timeframe": "1H",
            "start": "2026-05-10T00:00:00Z",
            "end": "2026-05-10T02:00:00Z",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["venue"] == "okx"
    assert payload[0]["symbol"] == "BTC-USDT"
