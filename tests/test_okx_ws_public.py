import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.ingest.okx_ws_public import (
    OKXPublicWSChannelError,
    OKXPublicWSClient,
    OKXPublicWSConfig,
    collect_okx_public_ws,
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


def _trade_message() -> dict[str, Any]:
    return {
        "arg": {"channel": "trades", "instId": "BTC-USDT"},
        "data": [
            {
                "tradeId": "123",
                "px": "100.5",
                "sz": "0.2",
                "side": "buy",
                "ts": "1771200000000",
            }
        ],
    }


def _books_message() -> dict[str, Any]:
    return {
        "arg": {"channel": "books5", "instId": "BTC-USDT"},
        "data": [
            {
                "asks": [["101", "1"]],
                "bids": [["100", "2"]],
                "checksum": 42,
                "ts": "1771200000000",
            }
        ],
    }
