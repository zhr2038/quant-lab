from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.ingest.okx_public import (
    OKXPublicAPIError,
    OKXPublicClient,
    OKXPublicConfig,
    OKXPublicTimeout,
    backfill_expanded_usdt_spot_market_bars,
    normalize_okx_candles_to_market_bars,
    publish_market_bars_to_lake,
    select_okx_usdt_spot_universe,
)


def test_public_methods_use_get_and_public_endpoints_without_auth_headers(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_request(self, method, url, params=None):
        calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "headers": dict(self.headers),
            }
        )
        request = httpx.Request(method, str(self.base_url.join(url)), params=params)
        return httpx.Response(
            200,
            json={"code": "0", "msg": "", "data": _data_for_url(url)},
            request=request,
        )

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXPublicClient(OKXPublicConfig(rate_limit_sleep_seconds=0))

    results = [
        client.get_instruments("SPOT"),
        client.get_tickers("SPOT"),
        client.get_ticker("BTC-USDT"),
        client.get_orderbook("BTC-USDT", sz=5),
        client.get_candles("BTC-USDT", "1H", limit=1),
        client.get_history_candles("BTC-USDT", "1H", before="1", limit=1),
        client.get_trades("BTC-USDT", limit=1),
        client.get_history_trades("BTC-USDT", after="1", before="2", limit=1),
    ]

    assert all(result for result in results)
    assert [call["method"] for call in calls] == ["GET"] * 8
    assert [call["url"] for call in calls] == [
        "/api/v5/public/instruments",
        "/api/v5/market/tickers",
        "/api/v5/market/ticker",
        "/api/v5/market/books",
        "/api/v5/market/candles",
        "/api/v5/market/history-candles",
        "/api/v5/market/trades",
        "/api/v5/market/history-trades",
    ]

    for call in calls:
        assert "/private/" not in str(call["url"]).lower()
        headers = {key.upper() for key in call["headers"]}
        for header in _forbidden_auth_headers():
            assert header not in headers


def test_get_candles_coerces_okx_arrays_to_dicts(monkeypatch):
    def fake_request(self, method, url, params=None):
        request = httpx.Request(method, str(self.base_url.join(url)), params=params)
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "",
                "data": [
                    [
                        "1760000000000",
                        "100",
                        "110",
                        "90",
                        "105",
                        "12",
                        "12",
                        "1260",
                        "1",
                    ]
                ],
            },
            request=request,
        )

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXPublicClient(OKXPublicConfig(rate_limit_sleep_seconds=0))

    assert client.get_candles("BTC-USDT", "1H", limit=1) == [
        {
            "ts": "1760000000000",
            "o": "100",
            "h": "110",
            "l": "90",
            "c": "105",
            "vol": "12",
            "volCcy": "12",
            "volCcyQuote": "1260",
            "confirm": "1",
        }
    ]


def test_empty_data_returns_empty_collections(monkeypatch):
    def fake_request(self, method, url, params=None):
        request = httpx.Request(method, str(self.base_url.join(url)), params=params)
        return httpx.Response(200, json={"code": "0", "msg": "", "data": []}, request=request)

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXPublicClient(OKXPublicConfig(rate_limit_sleep_seconds=0))

    assert client.get_instruments("SPOT") == []
    assert client.get_tickers("SPOT") == []
    assert client.get_ticker("BTC-USDT") == {}
    assert client.get_orderbook("BTC-USDT") == {}
    assert client.get_trades("BTC-USDT") == []


def test_normalize_okx_candles_to_market_bars():
    ingest_ts = datetime(2026, 2, 16, 2, tzinfo=UTC)
    candles = [
        {
            "ts": "1771200000000",
            "o": "100",
            "h": "110",
            "l": "90",
            "c": "105",
            "vol": "12",
            "volCcyQuote": "1260",
            "confirm": "1",
        }
    ]

    bars = normalize_okx_candles_to_market_bars(
        candles,
        inst_id="BTC-USDT",
        bar="1H",
        market_type="SPOT",
        ingest_ts=ingest_ts,
    )

    assert len(bars) == 1
    bar = bars[0]
    assert bar.venue == "okx"
    assert bar.symbol == "BTC-USDT"
    assert bar.market_type == "SPOT"
    assert bar.timeframe == "1H"
    assert bar.open == 100
    assert bar.high == 110
    assert bar.low == 90
    assert bar.close == 105
    assert bar.volume == 12
    assert bar.quote_volume == 1260
    assert bar.source == "okx_public_rest"
    assert bar.ingest_ts == ingest_ts
    assert bar.is_closed is True


def test_confirm_not_one_candle_is_filtered():
    bars = normalize_okx_candles_to_market_bars(
        [
            ["1771200000000", "100", "110", "90", "105", "12", "12", "1260", "0"],
            ["1771196400000", "99", "101", "98", "100", "8", "8", "800", "1"],
        ],
        inst_id="BTC-USDT",
        bar="1H",
        market_type="SPOT",
        ingest_ts=datetime(2026, 2, 16, 2, tzinfo=UTC),
    )

    assert len(bars) == 1
    assert bars[0].close == 100


def test_missing_confirm_uses_conservative_closed_bar_rule():
    now = datetime(2026, 2, 16, 2, tzinfo=UTC)
    closed_ts = int((now - timedelta(hours=2)).timestamp() * 1000)
    current_ts = int((now - timedelta(minutes=30)).timestamp() * 1000)

    bars = normalize_okx_candles_to_market_bars(
        [
            [str(closed_ts), "100", "110", "90", "105", "12", "12", "1260"],
            [str(current_ts), "105", "112", "104", "111", "10", "10", "1110"],
        ],
        inst_id="BTC-USDT",
        bar="1H",
        market_type="SPOT",
        ingest_ts=now,
        now=now,
    )

    assert len(bars) == 1
    assert bars[0].close == 105


def test_publish_okx_market_bars_to_lake(tmp_path):
    ingest_ts = datetime(2026, 2, 16, 2, tzinfo=UTC)
    bars = normalize_okx_candles_to_market_bars(
        [
            ["1771200000000", "100", "110", "90", "105", "12", "12", "1260", "1"],
        ],
        inst_id="BTC-USDT",
        bar="1H",
        market_type="SPOT",
        ingest_ts=ingest_ts,
    )

    rows_after_first_publish = publish_market_bars_to_lake(bars, tmp_path / "lake")
    rows_after_second_publish = publish_market_bars_to_lake(bars, tmp_path / "lake")
    dataset = read_parquet_dataset(tmp_path / "lake" / "silver" / "market_bar")

    assert rows_after_first_publish == 1
    assert rows_after_second_publish == 1
    assert dataset.height == 1
    assert dataset["source"][0] == "okx_public_rest"
    assert dataset["symbol"][0] == "BTC-USDT"


def test_okx_code_error_is_clear(monkeypatch):
    def fake_request(self, method, url, params=None):
        request = httpx.Request(method, str(self.base_url.join(url)), params=params)
        return httpx.Response(
            200,
            json={"code": "51000", "msg": "bad request", "data": []},
            request=request,
        )

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXPublicClient(OKXPublicConfig(rate_limit_sleep_seconds=0))

    with pytest.raises(OKXPublicAPIError, match="code=51000"):
        client.get_ticker("BTC-USDT")


def test_okx_timeout_error_is_clear(monkeypatch):
    def fake_request(self, method, url, params=None):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXPublicClient(OKXPublicConfig(max_retries=0, rate_limit_sleep_seconds=0))

    with pytest.raises(OKXPublicTimeout, match="timed out"):
        client.get_ticker("BTC-USDT")


def test_transient_error_retries_without_long_sleep(monkeypatch):
    attempts = 0

    def fake_request(self, method, url, params=None):
        nonlocal attempts
        attempts += 1
        request = httpx.Request(method, str(self.base_url.join(url)), params=params)
        status_code = 500 if attempts == 1 else 200
        return httpx.Response(
            status_code,
            json={"code": "0", "msg": "", "data": [{"instId": "BTC-USDT"}]},
            request=request,
        )

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXPublicClient(OKXPublicConfig(max_retries=1, rate_limit_sleep_seconds=0))

    assert client.get_ticker("BTC-USDT") == {"instId": "BTC-USDT"}
    assert attempts == 2


def test_select_okx_usdt_spot_universe_filters_risky_and_illiquid_symbols():
    instruments = [
        _instrument("BTC-USDT", "BTC", "USDT"),
        _instrument("XRP-USDT", "XRP", "USDT"),
        _instrument("DOGE-USDT", "DOGE", "USDT"),
        _instrument("USDC-USDT", "USDC", "USDT"),
        _instrument("ETH3L-USDT", "ETH3L", "USDT"),
        _instrument("THIN-USDT", "THIN", "USDT"),
        _instrument("WIDE-USDT", "WIDE", "USDT"),
    ]
    tickers = [
        _ticker("BTC-USDT", quote_volume=2_000_000, bid=100.0, ask=100.05),
        _ticker("XRP-USDT", quote_volume=5_000_000, bid=2.0, ask=2.001),
        _ticker("DOGE-USDT", quote_volume=9_000_000, bid=0.2, ask=0.2001),
        _ticker("USDC-USDT", quote_volume=9_000_000, bid=1.0, ask=1.0001),
        _ticker("ETH3L-USDT", quote_volume=9_000_000, bid=5.0, ask=5.001),
        _ticker("THIN-USDT", quote_volume=10_000, bid=10.0, ask=10.001),
        _ticker("WIDE-USDT", quote_volume=9_000_000, bid=10.0, ask=10.5),
    ]

    candidates = select_okx_usdt_spot_universe(
        instruments=instruments,
        tickers=tickers,
        max_symbols=10,
        min_quote_volume_24h=1_000_000,
        max_spread_bps=20,
    )

    assert [candidate.symbol for candidate in candidates] == ["BTC-USDT", "XRP-USDT"]
    assert candidates[0].is_current_v5_symbol is True
    assert candidates[1].quote_volume_24h == 5_000_000


def test_backfill_expanded_usdt_spot_market_bars_publishes_selected_symbols(tmp_path):
    lake_root = tmp_path / "lake"
    client = _FakeExpandedUniverseClient()

    result = backfill_expanded_usdt_spot_market_bars(
        lake_root=lake_root,
        client=client,
        max_symbols=2,
        history_pages=2,
        min_quote_volume_24h=1_000_000,
        max_spread_bps=20,
    )
    second = backfill_expanded_usdt_spot_market_bars(
        lake_root=lake_root,
        client=client,
        max_symbols=2,
        history_pages=2,
        min_quote_volume_24h=1_000_000,
        max_spread_bps=20,
    )
    market = read_parquet_dataset(lake_root / "silver" / "market_bar")
    candidates = read_parquet_dataset(
        lake_root / "bronze" / "okx_public_rest" / "spot_universe_candidates"
    )
    latest_by_symbol = (
        market.group_by("symbol")
        .agg(pl.col("ts").max().alias("latest_ts"))
        .sort("symbol")
        .to_dicts()
    )

    assert result.selected_symbols == ["BTC-USDT", "XRP-USDT"]
    assert result.published_market_bars == 4
    assert second.market_bar_rows == 4
    assert set(market["symbol"].to_list()) == {"BTC-USDT", "XRP-USDT"}
    assert latest_by_symbol == [
        {
            "symbol": "BTC-USDT",
            "latest_ts": datetime(2026, 2, 16, 1, tzinfo=UTC),
        },
        {
            "symbol": "XRP-USDT",
            "latest_ts": datetime(2026, 2, 16, 0, tzinfo=UTC),
        },
    ]
    assert set(candidates["symbol"].to_list()) == {"BTC-USDT", "XRP-USDT"}
    assert set(client.market_methods) == {"candles"}
    assert set(client.history_methods) == {"history_candles"}


def test_backfill_expanded_universe_deepens_only_symbols_with_internal_gaps(tmp_path):
    lake_root = tmp_path / "lake"
    seed = normalize_okx_candles_to_market_bars(
        [
            ["1771203600000", "101", "102", "100", "101.5", "11", "11", "1116.5", "1"],
            ["1771192800000", "100", "101", "99", "100.5", "10", "10", "1005", "1"],
        ],
        inst_id="BTC-USDT",
        bar="1H",
        market_type="SPOT",
    )
    publish_market_bars_to_lake(seed, lake_root)
    client = _GapRepairExpandedUniverseClient()

    result = backfill_expanded_usdt_spot_market_bars(
        lake_root=lake_root,
        client=client,
        max_symbols=2,
        history_pages=1,
        gap_repair_pages=3,
        limit=1,
        min_quote_volume_24h=1_000_000,
        max_spread_bps=20,
    )

    assert result.gap_repair_symbols == ["BTC-USDT"]
    assert result.gap_repair_history_pages == 3
    assert sum(symbol == "BTC-USDT" for symbol, _after in client.history_requests) == 3
    assert sum(symbol == "XRP-USDT" for symbol, _after in client.history_requests) == 1


def test_forbidden_keywords_do_not_appear_in_implementation_code():
    implementation_files = [
        Path("src/quant_lab/ingest/okx_public.py"),
        Path("src/quant_lab/cli.py"),
        Path("src/quant_lab/contracts/models.py"),
        Path("src/quant_lab/data/lake.py"),
    ]
    forbidden = [
        "OK-ACCESS-KEY",
        "OK-ACCESS-SIGN",
        "OK-ACCESS-PASSPHRASE",
        "place_order",
        "cancel_order",
        "amend_order",
        "withdraw",
        "transfer_funds",
    ]
    implementation_text = "\n".join(
        path.read_text(encoding="utf-8") for path in implementation_files
    )

    for keyword in forbidden:
        assert keyword not in implementation_text


def _data_for_url(url: str) -> list[dict[str, Any]] | list[list[str]]:
    responses: dict[str, list[dict[str, Any]] | list[list[str]]] = {
        "/api/v5/public/instruments": [{"instId": "BTC-USDT", "instType": "SPOT"}],
        "/api/v5/market/tickers": [{"instId": "BTC-USDT", "last": "100"}],
        "/api/v5/market/ticker": [{"instId": "BTC-USDT", "last": "100"}],
        "/api/v5/market/books": [{"asks": [["101", "1"]], "bids": [["100", "1"]]}],
        "/api/v5/market/candles": [
            ["1760000000000", "100", "110", "90", "105", "12", "12", "1260", "1"]
        ],
        "/api/v5/market/history-candles": [
            ["1759996400000", "98", "105", "97", "100", "10", "10", "1000", "1"]
        ],
        "/api/v5/market/trades": [{"tradeId": "1", "px": "100", "sz": "1"}],
        "/api/v5/market/history-trades": [{"tradeId": "0", "px": "99", "sz": "1"}],
    }
    return responses[url]


def _instrument(symbol: str, base: str, quote: str) -> dict[str, str]:
    return {
        "instType": "SPOT",
        "instId": symbol,
        "baseCcy": base,
        "quoteCcy": quote,
        "state": "live",
    }


def _ticker(symbol: str, *, quote_volume: float, bid: float, ask: float) -> dict[str, str]:
    return {
        "instId": symbol,
        "last": str((bid + ask) / 2.0),
        "bidPx": str(bid),
        "askPx": str(ask),
        "volCcy24h": str(quote_volume),
    }


class _FakeExpandedUniverseClient:
    def __init__(self) -> None:
        self.market_methods: list[str] = []
        self.history_methods: list[str] = []

    def get_instruments(self, inst_type: str) -> list[dict[str, str]]:
        assert inst_type == "SPOT"
        return [
            _instrument("BTC-USDT", "BTC", "USDT"),
            _instrument("XRP-USDT", "XRP", "USDT"),
            _instrument("DOGE-USDT", "DOGE", "USDT"),
        ]

    def get_tickers(self, inst_type: str) -> list[dict[str, str]]:
        assert inst_type == "SPOT"
        return [
            _ticker("BTC-USDT", quote_volume=2_000_000, bid=100.0, ask=100.05),
            _ticker("XRP-USDT", quote_volume=3_000_000, bid=2.0, ask=2.001),
            _ticker("DOGE-USDT", quote_volume=9_000_000, bid=0.2, ask=0.2001),
        ]

    def get_candles(
        self,
        inst_id: str,
        bar: str,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[list[str]]:
        assert bar == "1H"
        assert after is None
        assert before is None
        assert limit == 100
        self.market_methods.append("candles")
        ts = {
            "BTC-USDT": "1771203600000",
            "XRP-USDT": "1771200000000",
        }[inst_id]
        return [[ts, "101", "102", "100", "101.5", "11", "11", "1116.5", "1"]]

    def get_history_candles(
        self,
        inst_id: str,
        bar: str,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[list[str]]:
        assert bar == "1H"
        assert limit == 100
        self.history_methods.append("history_candles")
        if after is not None:
            ts = {
                "BTC-USDT": "1771200000000",
                "XRP-USDT": "1771196400000",
            }[inst_id]
            return [[ts, "100", "101", "99", "100.5", "10", "10", "1005", "1"]]
        assert before is None
        ts = {
            "BTC-USDT": "1771196400000",
            "XRP-USDT": "1771192800000",
        }[inst_id]
        return [[ts, "100", "101", "99", "100.5", "10", "10", "1005", "1"]]


class _GapRepairExpandedUniverseClient(_FakeExpandedUniverseClient):
    def __init__(self) -> None:
        super().__init__()
        self.history_requests: list[tuple[str, str | None]] = []

    def get_candles(
        self,
        inst_id: str,
        bar: str,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[list[str]]:
        assert bar == "1H"
        assert after is None
        assert before is None
        assert limit == 1
        self.market_methods.append("candles")
        ts = {
            "BTC-USDT": "1771203600000",
            "XRP-USDT": "1771200000000",
        }[inst_id]
        return [[ts, "101", "102", "100", "101.5", "11", "11", "1116.5", "1"]]

    def get_history_candles(
        self,
        inst_id: str,
        bar: str,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[list[str]]:
        assert bar == "1H"
        assert before is None
        assert limit == 1
        self.history_methods.append("history_candles")
        self.history_requests.append((inst_id, after))
        cursor = int(
            after
            or {
                "BTC-USDT": "1771203600000",
                "XRP-USDT": "1771200000000",
            }[inst_id]
        )
        ts = str(cursor - 60 * 60 * 1000)
        return [[ts, "100", "101", "99", "100.5", "10", "10", "1005", "1"]]


def _forbidden_auth_headers() -> list[str]:
    return [
        "-".join(["OK", "ACCESS", "KEY"]),
        "-".join(["OK", "ACCESS", "SIGN"]),
        "-".join(["OK", "ACCESS", "PASSPHRASE"]),
    ]
