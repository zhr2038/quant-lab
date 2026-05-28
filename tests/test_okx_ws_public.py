import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from quant_lab.data.lake import read_parquet_dataset, write_market_bars
from quant_lab.ingest.okx_ws_public import (
    COLLECTOR_HEALTH_DATASET,
    OKXPublicWSChannelError,
    OKXPublicWSClient,
    OKXPublicWSConfig,
    build_universe_subscription_payload,
    collect_okx_public_ws,
    collect_okx_public_ws_universe,
    normalize_okx_ws_candles_to_market_bars,
    normalize_okx_ws_orderbooks,
    normalize_okx_ws_trades,
    publish_okx_public_ws_messages_to_lake,
    subscribe_channel,
)


def test_public_subscriptions_do_not_send_login():
    fake_ws = FakeWebSocket(messages=[])
    client = OKXPublicWSClient(connect_factory=lambda url: fake_ws)

    async def scenario() -> None:
        await client.connect()
        await client.subscribe_tickers("BTC-USDT")
        await client.subscribe_candles("BTC-USDT", "1H")
        await client.subscribe_trades("BTC-USDT")
        await client.subscribe_books("BTC-USDT", "books5")

    asyncio.run(scenario())

    payloads = [json.loads(message) for message in fake_ws.sent]
    assert [payload["op"] for payload in payloads] == ["subscribe"] * 4
    assert {payload["args"][0]["channel"] for payload in payloads} == {
        "tickers",
        "candle1H",
        "trades",
        "books5",
    }
    assert "login" not in "\n".join(fake_ws.sent).lower()


def test_disallowed_channel_is_rejected():
    fake_ws = FakeWebSocket(messages=[])
    client = OKXPublicWSClient(connect_factory=lambda url: fake_ws)

    with pytest.raises(OKXPublicWSChannelError):
        asyncio.run(subscribe_channel(client, inst_id="BTC-USDT", channel="orders"))


def test_public_channel_router_subscribes_expected_payloads():
    fake_ws = FakeWebSocket(messages=[])
    client = OKXPublicWSClient(connect_factory=lambda url: fake_ws)

    async def scenario() -> None:
        await client.connect()
        for channel in ["tickers", "candle1H", "trades", "books"]:
            await subscribe_channel(client, inst_id="BTC-USDT", channel=channel)

    asyncio.run(scenario())

    channels = [json.loads(message)["args"][0]["channel"] for message in fake_ws.sent]
    assert channels == ["tickers", "candle1H", "trades", "books"]


def test_universe_subscription_payload_contains_all_symbols_and_channels():
    payload = build_universe_subscription_payload(
        symbols=["BTC-USDT", "ETH-USDT"],
        channels=["tickers", "trades", "books5"],
    )

    assert payload["op"] == "subscribe"
    assert payload["args"] == [
        {"channel": "tickers", "instId": "BTC-USDT"},
        {"channel": "trades", "instId": "BTC-USDT"},
        {"channel": "books5", "instId": "BTC-USDT"},
        {"channel": "tickers", "instId": "ETH-USDT"},
        {"channel": "trades", "instId": "ETH-USDT"},
        {"channel": "books5", "instId": "ETH-USDT"},
    ]


def test_universe_subscription_rejects_private_channel():
    with pytest.raises(OKXPublicWSChannelError):
        build_universe_subscription_payload(
            symbols=["BTC-USDT"],
            channels=["tickers", "orders"],
        )


def test_universe_subscription_sends_no_login_payload():
    fake_ws = FakeWebSocket(messages=[])
    client = OKXPublicWSClient(connect_factory=lambda url: fake_ws)

    async def scenario() -> None:
        await client.connect()
        await client.subscribe_public_channels(["BTC-USDT", "ETH-USDT"], ["tickers", "books5"])

    asyncio.run(scenario())

    assert len(fake_ws.sent) == 1
    sent = fake_ws.sent[0].lower()
    assert "login" not in sent
    assert json.loads(fake_ws.sent[0])["op"] == "subscribe"


def test_ws_message_standardization():
    messages = [_candle_message(confirm="1"), _trade_message(), _books_message()]

    bars = normalize_okx_ws_candles_to_market_bars(messages)
    trades = normalize_okx_ws_trades(messages)
    books = normalize_okx_ws_orderbooks(messages)

    assert len(bars) == 1
    assert bars[0].source == "okx_public_ws"
    assert bars[0].symbol == "BTC-USDT"
    assert bars[0].timeframe == "1H"
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "123"
    assert trades[0]["price"] == 100.5
    assert len(books) == 1
    assert books[0]["channel"] == "books5"
    assert books[0]["checksum"] == 42


def test_ws_trade_missing_timestamp_uses_ingest_time():
    message = _trade_message()
    del message["data"][0]["ts"]

    trades = normalize_okx_ws_trades([message])

    assert len(trades) == 1
    assert trades[0]["ts"]
    assert trades[0]["ts"].endswith("Z")
    assert trades[0]["day"] == trades[0]["ts"][:10]


def test_ws_orderbook_invalid_timestamp_uses_ingest_time():
    message = _books_message()
    message["data"][0]["ts"] = "not-a-timestamp"

    books = normalize_okx_ws_orderbooks([message])

    assert len(books) == 1
    assert books[0]["ts"]
    assert books[0]["ts"].endswith("Z")
    assert books[0]["day"] == books[0]["ts"][:10]


def test_ws_trade_and_orderbook_without_inst_id_do_not_write_silver_rows():
    trade = _trade_message()
    del trade["arg"]["instId"]
    book = _books_message()
    del book["arg"]["instId"]

    assert normalize_okx_ws_trades([trade]) == []
    assert normalize_okx_ws_orderbooks([book]) == []


def test_ws_candle_confirm_not_one_is_filtered():
    bars = normalize_okx_ws_candles_to_market_bars(
        [_candle_message(confirm="0"), _candle_message(confirm="1")]
    )

    assert len(bars) == 1
    assert bars[0].close == 105


def test_publish_ws_messages_to_lake(tmp_path):
    lake_root = tmp_path / "lake"

    result = publish_okx_public_ws_messages_to_lake(
        [_candle_message(confirm="1"), _trade_message(), _books_message()],
        lake_root=lake_root,
    )

    assert result == {
        "raw_ws_rows": 3,
        "market_bar_rows": 1,
        "trade_print_rows": 1,
        "orderbook_snapshot_rows": 1,
    }
    assert list((lake_root / "bronze" / "okx_public_ws").rglob("*.parquet"))
    assert list((lake_root / "silver" / "market_bar").rglob("*.parquet"))
    assert list((lake_root / "silver" / "trade_print").rglob("*.parquet"))
    assert list((lake_root / "silver" / "orderbook_snapshot").rglob("*.parquet"))

    trades = read_parquet_dataset(lake_root / "silver" / "trade_print")
    books = read_parquet_dataset(lake_root / "silver" / "orderbook_snapshot")
    raw = read_parquet_dataset(lake_root / "bronze" / "okx_public_ws")

    assert trades["symbol"][0] == "BTC-USDT"
    assert books["symbol"][0] == "BTC-USDT"
    assert raw.height == 3


def test_partitioned_ws_append_does_not_create_null_day_partition(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_LAB_WS_APPEND_PARTITIONED", "true")
    lake_root = tmp_path / "lake"
    trade = _trade_message(symbol="BTC-USDT", trade_id="missing-ts")
    del trade["data"][0]["ts"]
    book = _books_message(symbol="BTC-USDT")
    book["data"][0]["ts"] = "invalid-ts"
    ack = {"event": "subscribe", "data": []}

    result = publish_okx_public_ws_messages_to_lake([trade, book, ack], lake_root=lake_root)

    assert result["trade_print_rows"] == 1
    assert result["orderbook_snapshot_rows"] == 1
    parquet_paths = list(lake_root.rglob("*.parquet"))
    assert parquet_paths
    assert all("__null__" not in str(path) for path in parquet_paths)
    assert any("channel=unknown" in str(path) for path in parquet_paths)


def test_ws_stream_appends_large_datasets_without_rewriting_market_bars(tmp_path):
    lake_root = tmp_path / "lake"
    write_market_bars(
        lake_root,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": datetime(2026, 5, 13, 1, tzinfo=UTC),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "quote_volume": 1000.0,
                "source": "fixture",
                "ingest_ts": datetime(2026, 5, 13, 2, tzinfo=UTC),
            }
        ],
    )

    first = publish_okx_public_ws_messages_to_lake(
        [_trade_message(symbol="BTC-USDT", trade_id="1")],
        lake_root=lake_root,
    )
    second = publish_okx_public_ws_messages_to_lake(
        [_trade_message(symbol="ETH-USDT", trade_id="2")],
        lake_root=lake_root,
    )

    assert first["market_bar_rows"] == 0
    assert second["market_bar_rows"] == 0
    assert read_parquet_dataset(lake_root / "silver" / "market_bar").height == 1
    assert read_parquet_dataset(lake_root / "silver" / "trade_print").height == 2
    assert len(list((lake_root / "silver" / "trade_print").rglob("*.parquet"))) == 2
    assert not list((lake_root / "silver" / "trade_print").glob("day=*"))


def test_ws_batch_append_writes_one_file_per_dataset_flush(tmp_path):
    lake_root = tmp_path / "lake"
    publish_okx_public_ws_messages_to_lake(
        [
            _trade_message(symbol="BTC-USDT", trade_id="1"),
            _trade_message(symbol="ETH-USDT", trade_id="2"),
            _books_message(symbol="BTC-USDT"),
            _books_message(symbol="ETH-USDT"),
        ],
        lake_root=lake_root,
    )

    assert len(list((lake_root / "bronze" / "okx_public_ws").rglob("*.parquet"))) == 1
    assert len(list((lake_root / "silver" / "trade_print").rglob("*.parquet"))) == 1
    assert len(list((lake_root / "silver" / "orderbook_snapshot").rglob("*.parquet"))) == 1

    raw = read_parquet_dataset(lake_root / "bronze" / "okx_public_ws")
    trades = read_parquet_dataset(lake_root / "silver" / "trade_print")
    books = read_parquet_dataset(lake_root / "silver" / "orderbook_snapshot")

    assert raw.height == 4
    assert trades.height == 2
    assert books.height == 2


def test_read_messages_reconnects_without_long_sleep():
    first_ws = FakeWebSocket(messages=[], fail_once=True)
    second_ws = FakeWebSocket(messages=[json.dumps(_trade_message())])
    sockets = [first_ws, second_ws]
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def connect_factory(url: str) -> FakeWebSocket:
        return sockets.pop(0)

    client = OKXPublicWSClient(
        config=OKXPublicWSConfig(reconnect_max_attempts=1, reconnect_backoff_seconds=0),
        connect_factory=connect_factory,
        sleep=fake_sleep,
    )

    async def scenario() -> list[dict[str, Any]]:
        await client.connect()
        await client.subscribe_trades("BTC-USDT")
        messages = []
        async for message in client.read_messages(max_messages=1):
            messages.append(message)
        return messages

    messages = asyncio.run(scenario())

    assert len(messages) == 1
    assert messages[0]["arg"]["channel"] == "trades"
    assert sleeps == [0]
    assert len(first_ws.sent) == 1
    assert len(second_ws.sent) == 1


def test_read_messages_reconnects_after_receive_timeout():
    first_ws = HangingWebSocket()
    second_ws = FakeWebSocket(messages=[json.dumps(_trade_message())])
    sockets = [first_ws, second_ws]
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def connect_factory(url: str) -> Any:
        return sockets.pop(0)

    client = OKXPublicWSClient(
        config=OKXPublicWSConfig(
            receive_timeout_seconds=0.001,
            reconnect_max_attempts=1,
            reconnect_backoff_seconds=0,
        ),
        connect_factory=connect_factory,
        sleep=fake_sleep,
    )

    async def scenario() -> list[dict[str, Any]]:
        await client.connect()
        await client.subscribe_trades("BTC-USDT")
        messages = []
        async for message in client.read_messages(max_messages=1):
            messages.append(message)
        return messages

    messages = asyncio.run(scenario())

    assert len(messages) == 1
    assert messages[0]["arg"]["channel"] == "trades"
    assert client.reconnect_count == 1
    assert client.error_count == 1
    assert sleeps == [0]
    assert first_ws.closed is True
    assert len(first_ws.sent) == 1
    assert len(second_ws.sent) == 1


def test_collect_okx_public_ws_summary(tmp_path):
    fake_ws = FakeWebSocket(messages=[json.dumps(_trade_message())])

    summary = asyncio.run(
        collect_okx_public_ws(
            inst_id="BTC-USDT",
            channels=["trades"],
            lake_root=tmp_path / "lake",
            max_messages=1,
            connect_factory=lambda url: fake_ws,
            sleep=_no_sleep,
        )
    )

    assert summary.messages_read == 1
    assert summary.raw_ws_rows == 1
    assert summary.trade_print_rows == 1
    assert summary.datasets["trade_print"].endswith("silver\\trade_print") or summary.datasets[
        "trade_print"
    ].endswith("silver/trade_print")


def test_collect_universe_batches_flush_and_writes_health(tmp_path):
    lake_root = tmp_path / "lake"
    fake_ws = FakeWebSocket(
        messages=[
            json.dumps(_trade_message(symbol="BTC-USDT", trade_id="1")),
            json.dumps(_trade_message(symbol="ETH-USDT", trade_id="2")),
            json.dumps(_books_message(symbol="BTC-USDT")),
        ]
    )

    summary = asyncio.run(
        collect_okx_public_ws_universe(
            symbols=["BTC-USDT", "ETH-USDT"],
            channels=["trades", "books5"],
            lake_root=lake_root,
            flush_interval_seconds=100,
            flush_max_messages=3,
            max_messages=3,
            connect_factory=lambda url: fake_ws,
            sleep=_no_sleep,
        )
    )

    raw = read_parquet_dataset(lake_root / "bronze" / "okx_public_ws")
    trades = read_parquet_dataset(lake_root / "silver" / "trade_print")
    health = read_parquet_dataset(lake_root / COLLECTOR_HEALTH_DATASET)

    assert summary.messages_read == 3
    assert summary.raw_ws_rows == 3
    assert summary.trade_print_rows == 2
    assert raw.height == 3
    assert trades.height == 2
    assert health.height == 1
    assert health["last_message_at"][0]
    assert health["messages_read"][0] == 3
    assert health["status"][0] == "STOPPED"
    assert json.loads(health["subscribed_symbols"][0]) == ["BTC-USDT", "ETH-USDT"]


def test_forbidden_ws_terms_do_not_appear_in_implementation_code():
    implementation_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("src/quant_lab/ingest/okx_ws_public.py"),
            Path("src/quant_lab/cli.py"),
        ]
    )
    forbidden = [
        "OK-ACCESS-KEY",
        "OK-ACCESS-SIGN",
        "OK-ACCESS-PASSPHRASE",
        "place_order",
        "cancel_order",
        "amend_order",
    ]

    for keyword in forbidden:
        assert keyword not in implementation_text


class FakeWebSocket:
    def __init__(self, messages: list[str], fail_once: bool = False) -> None:
        self.messages = list(messages)
        self.fail_once = fail_once
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if self.fail_once:
            self.fail_once = False
            raise ConnectionError("temporary disconnect")
        if not self.messages:
            raise ConnectionError("no more test messages")
        return self.messages.pop(0)

    async def close(self) -> None:
        return None


class HangingWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        await asyncio.sleep(1)
        raise AssertionError("timeout should reconnect before recv returns")

    async def close(self) -> None:
        self.closed = True


async def _no_sleep(seconds: float) -> None:
    return None


def _candle_message(confirm: str) -> dict[str, Any]:
    return {
        "arg": {"channel": "candle1H", "instId": "BTC-USDT"},
        "data": [
            [
                "1771200000000",
                "100",
                "110",
                "90",
                "105",
                "12",
                "12",
                "1260",
                confirm,
            ]
        ],
    }


def _trade_message(symbol: str = "BTC-USDT", trade_id: str = "123") -> dict[str, Any]:
    return {
        "arg": {"channel": "trades", "instId": symbol},
        "data": [
            {
                "tradeId": trade_id,
                "px": "100.5",
                "sz": "0.2",
                "side": "buy",
                "ts": "1771200000000",
            }
        ],
    }


def _books_message(symbol: str = "BTC-USDT") -> dict[str, Any]:
    return {
        "arg": {"channel": "books5", "instId": symbol},
        "data": [
            {
                "asks": [["101", "1"]],
                "bids": [["100", "2"]],
                "checksum": 42,
                "ts": "1771200000000",
            }
        ],
    }
