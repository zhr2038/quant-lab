import asyncio
import inspect
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import MarketBar
from quant_lab.data.lake import append_parquet_dataset, read_parquet_dataset, write_parquet_dataset
from quant_lab.ingest.okx_public import (
    MARKET_BAR_DATASET,
    normalize_okx_candles_to_market_bars,
    publish_market_bars_to_lake,
)
from quant_lab.symbols import normalize_symbol

OKX_PUBLIC_WS_SOURCE = "okx_public_ws"
BRONZE_WS_DATASET = Path("bronze") / "okx_public_ws"
TRADE_PRINT_DATASET = Path("silver") / "trade_print"
ORDERBOOK_SNAPSHOT_DATASET = Path("silver") / "orderbook_snapshot"
COLLECTOR_HEALTH_DATASET = Path("bronze") / "collector_health" / "okx_public_ws"
OKX_PUBLIC_WS_COLLECTOR_NAME = "okx_public_ws"

RAW_WS_SCHEMA = {
    "channel": pl.Utf8,
    "inst_id": pl.Utf8,
    "day": pl.Utf8,
    "received_at": pl.Utf8,
    "raw_json": pl.Utf8,
}

TRADE_PRINT_SCHEMA = {
    "venue": pl.Utf8,
    "symbol": pl.Utf8,
    "day": pl.Utf8,
    "trade_id": pl.Utf8,
    "price": pl.Float64,
    "size": pl.Float64,
    "side": pl.Utf8,
    "ts": pl.Utf8,
    "source": pl.Utf8,
    "ingest_ts": pl.Utf8,
    "raw_json": pl.Utf8,
}

ORDERBOOK_SNAPSHOT_SCHEMA = {
    "venue": pl.Utf8,
    "symbol": pl.Utf8,
    "day": pl.Utf8,
    "channel": pl.Utf8,
    "ts": pl.Utf8,
    "asks_json": pl.Utf8,
    "bids_json": pl.Utf8,
    "checksum": pl.Int64,
    "source": pl.Utf8,
    "ingest_ts": pl.Utf8,
    "raw_json": pl.Utf8,
}

COLLECTOR_HEALTH_SCHEMA = {
    "collector_name": pl.Utf8,
    "started_at": pl.Utf8,
    "last_message_at": pl.Utf8,
    "messages_read": pl.Int64,
    "reconnect_count": pl.Int64,
    "error_count": pl.Int64,
    "subscribed_symbols": pl.Utf8,
    "subscribed_channels": pl.Utf8,
    "status": pl.Utf8,
    "updated_at": pl.Utf8,
}


class OKXPublicWSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = "wss://ws.okx.com:8443/ws/v5/public"
    heartbeat_seconds: int = Field(default=20, gt=0)
    reconnect_max_attempts: int = Field(default=10, ge=0)
    reconnect_backoff_seconds: float = Field(default=1.0, ge=0)


class OKXPublicWSError(RuntimeError):
    pass


class OKXPublicWSChannelError(OKXPublicWSError):
    pass


class OKXWSCollectSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    inst_id: str
    channels: list[str]
    lake_root: str
    messages_read: int = Field(ge=0)
    raw_ws_rows: int = Field(ge=0)
    market_bar_rows: int = Field(ge=0)
    trade_print_rows: int = Field(ge=0)
    orderbook_snapshot_rows: int = Field(ge=0)
    datasets: dict[str, str]


class OKXWSUniverseCollectSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbols: list[str]
    channels: list[str]
    lake_root: str
    messages_read: int = Field(ge=0)
    raw_ws_rows: int = Field(ge=0)
    market_bar_rows: int = Field(ge=0)
    trade_print_rows: int = Field(ge=0)
    orderbook_snapshot_rows: int = Field(ge=0)
    collector_health_rows: int = Field(ge=0)
    reconnect_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    status: str
    datasets: dict[str, str]


ConnectFactory = Callable[[str], Any | Awaitable[Any]]
SleepFunc = Callable[[float], Awaitable[None]]


class OKXPublicWSClient:
    def __init__(
        self,
        config: OKXPublicWSConfig | None = None,
        connect_factory: ConnectFactory | None = None,
        sleep: SleepFunc | None = None,
    ) -> None:
        self.config = config or OKXPublicWSConfig()
        self._connect_factory = connect_factory
        self._sleep = sleep or asyncio.sleep
        self._ws: Any | None = None
        self._subscriptions: list[dict[str, Any]] = []
        self.reconnect_count = 0
        self.error_count = 0

    async def connect(self) -> None:
        self._ws = await self._open_connection()
        for payload in self._subscriptions:
            await self._send_payload(payload)

    async def subscribe_tickers(self, inst_id: str) -> None:
        await self._subscribe("tickers", inst_id)

    async def subscribe_candles(self, inst_id: str, bar: str) -> None:
        await self._subscribe(f"candle{bar}", inst_id)

    async def subscribe_trades(self, inst_id: str) -> None:
        await self._subscribe("trades", inst_id)

    async def subscribe_books(self, inst_id: str, depth: str = "books") -> None:
        await self._subscribe(depth, inst_id)

    async def subscribe_public_channels(
        self,
        symbols: Sequence[str],
        channels: Sequence[str],
    ) -> None:
        payload = build_universe_subscription_payload(symbols, channels)
        self._subscriptions.append(payload)
        if self._ws is not None:
            await self._send_payload(payload)

    async def read_messages(self, max_messages: int | None = None) -> AsyncIterator[dict[str, Any]]:
        emitted = 0
        reconnect_attempts = 0

        while max_messages is None or emitted < max_messages:
            if self._ws is None:
                await self.connect()

            try:
                raw_message = await self._ws.recv()
            except Exception:
                reconnect_attempts += 1
                self.reconnect_count += 1
                self.error_count += 1
                if reconnect_attempts > self.config.reconnect_max_attempts:
                    raise
                await self._sleep(self.config.reconnect_backoff_seconds)
                await self.connect()
                continue

            reconnect_attempts = 0
            if raw_message == "ping":
                await self._ws.send("pong")
                continue

            message = _loads_ws_message(raw_message)
            emitted += 1
            yield message

    async def _subscribe(self, channel: str, inst_id: str) -> None:
        _validate_public_channel(channel)
        payload = {"op": "subscribe", "args": [{"channel": channel, "instId": inst_id}]}
        self._subscriptions.append(payload)
        if self._ws is not None:
            await self._send_payload(payload)

    async def _send_payload(self, payload: Mapping[str, Any]) -> None:
        if self._ws is None:
            raise OKXPublicWSError("WebSocket is not connected")
        await self._ws.send(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    async def _open_connection(self) -> Any:
        if self._connect_factory is not None:
            connection = self._connect_factory(self.config.url)
            if inspect.isawaitable(connection):
                return await connection
            return connection

        import websockets

        return await websockets.connect(
            self.config.url,
            ping_interval=self.config.heartbeat_seconds,
        )


async def collect_okx_public_ws(
    inst_id: str,
    channels: Sequence[str],
    lake_root: str | Path,
    market_type: str = "SPOT",
    config: OKXPublicWSConfig | None = None,
    max_messages: int | None = None,
    connect_factory: ConnectFactory | None = None,
    sleep: SleepFunc | None = None,
) -> OKXWSCollectSummary:
    client = OKXPublicWSClient(config=config, connect_factory=connect_factory, sleep=sleep)
    await client.connect()
    for channel in channels:
        await subscribe_channel(client, inst_id=inst_id, channel=channel)

    messages_read = 0
    raw_ws_rows = 0
    market_bar_rows = 0
    trade_print_rows = 0
    orderbook_snapshot_rows = 0

    async for message in client.read_messages(max_messages=max_messages):
        messages_read += 1
        publish_result = publish_okx_public_ws_messages_to_lake(
            [message],
            lake_root=lake_root,
            market_type=market_type,
        )
        raw_ws_rows = publish_result["raw_ws_rows"]
        market_bar_rows = publish_result["market_bar_rows"]
        trade_print_rows = publish_result["trade_print_rows"]
        orderbook_snapshot_rows = publish_result["orderbook_snapshot_rows"]

    root = Path(lake_root)
    return OKXWSCollectSummary(
        inst_id=inst_id,
        channels=list(channels),
        lake_root=str(root),
        messages_read=messages_read,
        raw_ws_rows=raw_ws_rows,
        market_bar_rows=market_bar_rows,
        trade_print_rows=trade_print_rows,
        orderbook_snapshot_rows=orderbook_snapshot_rows,
        datasets=_dataset_paths(root),
    )


async def collect_okx_public_ws_universe(
    symbols: Sequence[str],
    channels: Sequence[str],
    lake_root: str | Path,
    market_type: str = "SPOT",
    flush_interval_seconds: float = 300.0,
    flush_max_messages: int = 50_000,
    config: OKXPublicWSConfig | None = None,
    max_messages: int | None = None,
    connect_factory: ConnectFactory | None = None,
    sleep: SleepFunc | None = None,
) -> OKXWSUniverseCollectSummary:
    normalized_symbols = _normalize_symbols(symbols)
    normalized_channels = _normalize_channels(channels)
    if flush_interval_seconds <= 0:
        raise ValueError("flush_interval_seconds must be positive")
    if flush_max_messages <= 0:
        raise ValueError("flush_max_messages must be positive")

    root = Path(lake_root)
    started_at = _utc_now_string()
    last_message_at: str | None = None
    messages_read = 0
    raw_ws_rows = 0
    market_bar_rows = 0
    trade_print_rows = 0
    orderbook_snapshot_rows = 0
    buffer: list[dict[str, Any]] = []
    status = "RUNNING"

    client = OKXPublicWSClient(config=config, connect_factory=connect_factory, sleep=sleep)
    await client.connect()
    await client.subscribe_public_channels(normalized_symbols, normalized_channels)
    collector_health_rows = publish_okx_public_ws_collector_health(
        lake_root=root,
        started_at=started_at,
        last_message_at=last_message_at,
        messages_read=messages_read,
        reconnect_count=client.reconnect_count,
        error_count=client.error_count,
        subscribed_symbols=normalized_symbols,
        subscribed_channels=normalized_channels,
        status=status,
    )

    loop = asyncio.get_running_loop()
    last_flush = loop.time()

    try:
        async for message in client.read_messages(max_messages=max_messages):
            messages_read += 1
            last_message_at = _utc_now_string()
            buffer.append(message)
            should_flush = (
                len(buffer) >= flush_max_messages
                or (loop.time() - last_flush) >= flush_interval_seconds
            )
            if should_flush:
                publish_result = publish_okx_public_ws_messages_to_lake(
                    buffer,
                    lake_root=root,
                    market_type=market_type,
                )
                buffer.clear()
                last_flush = loop.time()
                raw_ws_rows = publish_result["raw_ws_rows"]
                market_bar_rows = publish_result["market_bar_rows"]
                trade_print_rows = publish_result["trade_print_rows"]
                orderbook_snapshot_rows = publish_result["orderbook_snapshot_rows"]
                collector_health_rows = publish_okx_public_ws_collector_health(
                    lake_root=root,
                    started_at=started_at,
                    last_message_at=last_message_at,
                    messages_read=messages_read,
                    reconnect_count=client.reconnect_count,
                    error_count=client.error_count,
                    subscribed_symbols=normalized_symbols,
                    subscribed_channels=normalized_channels,
                    status=status,
                )
    except Exception:
        status = "ERROR"
        client.error_count += 1
        raise
    finally:
        if buffer:
            publish_result = publish_okx_public_ws_messages_to_lake(
                buffer,
                lake_root=root,
                market_type=market_type,
            )
            raw_ws_rows = publish_result["raw_ws_rows"]
            market_bar_rows = publish_result["market_bar_rows"]
            trade_print_rows = publish_result["trade_print_rows"]
            orderbook_snapshot_rows = publish_result["orderbook_snapshot_rows"]
        final_status = "STOPPED" if max_messages is not None and status == "RUNNING" else status
        collector_health_rows = publish_okx_public_ws_collector_health(
            lake_root=root,
            started_at=started_at,
            last_message_at=last_message_at,
            messages_read=messages_read,
            reconnect_count=client.reconnect_count,
            error_count=client.error_count,
            subscribed_symbols=normalized_symbols,
            subscribed_channels=normalized_channels,
            status=final_status,
        )
        status = final_status

    return OKXWSUniverseCollectSummary(
        symbols=normalized_symbols,
        channels=normalized_channels,
        lake_root=str(root),
        messages_read=messages_read,
        raw_ws_rows=raw_ws_rows,
        market_bar_rows=market_bar_rows,
        trade_print_rows=trade_print_rows,
        orderbook_snapshot_rows=orderbook_snapshot_rows,
        collector_health_rows=collector_health_rows,
        reconnect_count=client.reconnect_count,
        error_count=client.error_count,
        status=status,
        datasets=_dataset_paths(root),
    )


async def subscribe_channel(client: OKXPublicWSClient, inst_id: str, channel: str) -> None:
    _validate_public_channel(channel)
    if channel == "tickers":
        await client.subscribe_tickers(inst_id)
        return
    if channel.startswith("candle"):
        await client.subscribe_candles(inst_id, channel.removeprefix("candle"))
        return
    if channel == "trades":
        await client.subscribe_trades(inst_id)
        return
    await client.subscribe_books(inst_id, depth=channel)


def build_universe_subscription_payload(
    symbols: Sequence[str],
    channels: Sequence[str],
) -> dict[str, Any]:
    normalized_symbols = _normalize_symbols(symbols)
    normalized_channels = _normalize_channels(channels)
    return {
        "op": "subscribe",
        "args": [
            {"channel": channel, "instId": symbol}
            for symbol in normalized_symbols
            for channel in normalized_channels
        ],
    }


def publish_okx_public_ws_messages_to_lake(
    messages: Sequence[Mapping[str, Any]],
    lake_root: str | Path,
    market_type: str = "SPOT",
) -> dict[str, int]:
    root = Path(lake_root)
    raw_rows = _raw_ws_rows(messages)
    market_bars = normalize_okx_ws_candles_to_market_bars(messages, market_type=market_type)
    trade_rows = normalize_okx_ws_trades(messages)
    orderbook_rows = normalize_okx_ws_orderbooks(messages)

    raw_ws_rows = _append_frame(
        root / BRONZE_WS_DATASET,
        pl.DataFrame(raw_rows, schema=RAW_WS_SCHEMA, orient="row"),
        key_columns=["channel", "inst_id", "received_at", "raw_json"],
    )
    market_bar_rows = publish_market_bars_to_lake(market_bars, root) if market_bars else 0
    trade_print_rows = _append_frame(
        root / TRADE_PRINT_DATASET,
        pl.DataFrame(trade_rows, schema=TRADE_PRINT_SCHEMA, orient="row"),
        key_columns=["symbol", "trade_id", "ts"],
    )
    orderbook_snapshot_rows = _append_frame(
        root / ORDERBOOK_SNAPSHOT_DATASET,
        pl.DataFrame(orderbook_rows, schema=ORDERBOOK_SNAPSHOT_SCHEMA, orient="row"),
        key_columns=["symbol", "channel", "ts", "checksum"],
    )
    return {
        "raw_ws_rows": raw_ws_rows,
        "market_bar_rows": market_bar_rows,
        "trade_print_rows": trade_print_rows,
        "orderbook_snapshot_rows": orderbook_snapshot_rows,
    }


def publish_okx_public_ws_collector_health(
    *,
    lake_root: str | Path,
    started_at: str,
    last_message_at: str | None,
    messages_read: int,
    reconnect_count: int,
    error_count: int,
    subscribed_symbols: Sequence[str],
    subscribed_channels: Sequence[str],
    status: str,
) -> int:
    row = {
        "collector_name": OKX_PUBLIC_WS_COLLECTOR_NAME,
        "started_at": started_at,
        "last_message_at": last_message_at,
        "messages_read": messages_read,
        "reconnect_count": reconnect_count,
        "error_count": error_count,
        "subscribed_symbols": _json_dumps(list(subscribed_symbols)),
        "subscribed_channels": _json_dumps(list(subscribed_channels)),
        "status": status,
        "updated_at": _utc_now_string(),
    }
    return _upsert_frame(
        Path(lake_root) / COLLECTOR_HEALTH_DATASET,
        pl.DataFrame([row], schema=COLLECTOR_HEALTH_SCHEMA, orient="row"),
        key_columns=["collector_name"],
    )


def normalize_okx_ws_candles_to_market_bars(
    messages: Sequence[Mapping[str, Any]],
    market_type: str = "SPOT",
) -> list[MarketBar]:
    records: list[MarketBar] = []
    for message in messages:
        channel = _message_channel(message)
        if not channel or not channel.startswith("candle"):
            continue
        inst_id = _message_inst_id(message)
        if not inst_id:
            continue
        records.extend(
            normalize_okx_candles_to_market_bars(
                _message_data(message),
                inst_id=inst_id,
                bar=channel.removeprefix("candle"),
                market_type=market_type,
                source=OKX_PUBLIC_WS_SOURCE,
            )
        )
    return records


def normalize_okx_ws_trades(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ingest_ts = _utc_now_string()
    for message in messages:
        if _message_channel(message) != "trades":
            continue
        inst_id = _message_inst_id(message)
        for item in _message_data(message):
            if not isinstance(item, Mapping):
                continue
            rows.append(
                {
                    "venue": "okx",
                    "symbol": normalize_symbol(inst_id),
                    "day": _timestamp_ms_to_day(item.get("ts")),
                    "trade_id": _optional_string(item.get("tradeId")),
                    "price": _optional_float(item.get("px")),
                    "size": _optional_float(item.get("sz")),
                    "side": _optional_string(item.get("side")),
                    "ts": _timestamp_ms_to_utc_string(item.get("ts")),
                    "source": OKX_PUBLIC_WS_SOURCE,
                    "ingest_ts": ingest_ts,
                    "raw_json": _json_dumps(item),
                }
            )
    return rows


def normalize_okx_ws_orderbooks(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ingest_ts = _utc_now_string()
    for message in messages:
        channel = _message_channel(message)
        if channel not in {"books", "books5"}:
            continue
        inst_id = _message_inst_id(message)
        for item in _message_data(message):
            if not isinstance(item, Mapping):
                continue
            rows.append(
                {
                    "venue": "okx",
                    "symbol": normalize_symbol(inst_id),
                    "day": _timestamp_ms_to_day(item.get("ts")),
                    "channel": channel,
                    "ts": _timestamp_ms_to_utc_string(item.get("ts")),
                    "asks_json": _json_dumps(item.get("asks", [])),
                    "bids_json": _json_dumps(item.get("bids", [])),
                    "checksum": _optional_int(item.get("checksum")),
                    "source": OKX_PUBLIC_WS_SOURCE,
                    "ingest_ts": ingest_ts,
                    "raw_json": _json_dumps(item),
                }
            )
    return rows


def _validate_public_channel(channel: str) -> None:
    if channel in {"tickers", "trades", "books", "books5"}:
        return
    if channel.startswith("candle") and len(channel) > len("candle"):
        return
    raise OKXPublicWSChannelError(f"Disallowed OKX public WebSocket channel: {channel}")


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    normalized = [normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    if not normalized:
        raise ValueError("at least one symbol is required")
    return normalized


def _normalize_channels(channels: Sequence[str]) -> list[str]:
    normalized = [str(channel).strip() for channel in channels if str(channel).strip()]
    if not normalized:
        raise ValueError("at least one channel is required")
    for channel in normalized:
        _validate_public_channel(channel)
    return normalized


def _loads_ws_message(raw_message: Any) -> dict[str, Any]:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    if isinstance(raw_message, str):
        payload = json.loads(raw_message)
    elif isinstance(raw_message, Mapping):
        payload = dict(raw_message)
    else:
        raise OKXPublicWSError("Unexpected WebSocket message type")
    if not isinstance(payload, dict):
        raise OKXPublicWSError("Unexpected WebSocket message payload")
    return payload


def _raw_ws_rows(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    received_at = _utc_now_string()
    return [
        {
            "channel": _message_channel(message),
            "inst_id": _message_inst_id(message),
            "day": received_at[:10],
            "received_at": received_at,
            "raw_json": _json_dumps(message),
        }
        for message in messages
    ]


def _append_frame(dataset_path: Path, new_df: pl.DataFrame, key_columns: list[str]) -> int:
    if new_df.is_empty():
        return 0
    frame = _dedupe_frame(new_df, key_columns)
    partition_by = _append_partitions_for_dataset(dataset_path)
    result = append_parquet_dataset(
        frame,
        dataset_path,
        partition_by=partition_by,
        target_rows_per_file=_append_target_rows_for_dataset(dataset_path),
        file_prefix="batch",
    )
    return result.rows_written


def _upsert_frame(dataset_path: Path, new_df: pl.DataFrame, key_columns: list[str]) -> int:
    existing_df = read_parquet_dataset(dataset_path)
    if new_df.is_empty():
        return existing_df.height
    frames = [frame for frame in [existing_df, new_df] if not frame.is_empty()]
    combined = pl.concat(frames, how="diagonal_relaxed") if frames else new_df
    combined = _dedupe_frame(combined, key_columns)
    write_parquet_dataset(combined, dataset_path)
    return combined.height


def _dedupe_frame(df: pl.DataFrame, key_columns: list[str]) -> pl.DataFrame:
    if df.is_empty():
        return df
    available_keys = [column for column in key_columns if column in df.columns]
    if not available_keys:
        return df
    return df.unique(subset=available_keys, keep="last", maintain_order=True)


def _append_partitions_for_dataset(dataset_path: Path) -> list[str]:
    if os.environ.get("QUANT_LAB_WS_APPEND_PARTITIONED", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return []
    if _dataset_path_endswith(dataset_path, BRONZE_WS_DATASET):
        return ["day", "channel", "inst_id"]
    if _dataset_path_endswith(dataset_path, TRADE_PRINT_DATASET):
        return ["day", "symbol"]
    if _dataset_path_endswith(dataset_path, ORDERBOOK_SNAPSHOT_DATASET):
        return ["day", "symbol", "channel"]
    return []


def _append_target_rows_for_dataset(dataset_path: Path) -> int:
    raw_value = os.environ.get("QUANT_LAB_WS_APPEND_TARGET_ROWS", "50000")
    try:
        target = int(raw_value)
    except ValueError:
        target = 50_000
    return max(target, 1)


def _dataset_path_endswith(dataset_path: Path, relative: Path) -> bool:
    parts = dataset_path.parts
    suffix = relative.parts
    return len(parts) >= len(suffix) and parts[-len(suffix) :] == suffix


def _dataset_paths(root: Path) -> dict[str, str]:
    return {
        "okx_public_ws": str(root / BRONZE_WS_DATASET),
        "market_bar": str(root / MARKET_BAR_DATASET),
        "trade_print": str(root / TRADE_PRINT_DATASET),
        "orderbook_snapshot": str(root / ORDERBOOK_SNAPSHOT_DATASET),
        "collector_health": str(root / COLLECTOR_HEALTH_DATASET),
    }


def _message_channel(message: Mapping[str, Any]) -> str | None:
    arg = message.get("arg")
    if isinstance(arg, Mapping):
        channel = arg.get("channel")
        return str(channel) if channel is not None else None
    return None


def _message_inst_id(message: Mapping[str, Any]) -> str | None:
    arg = message.get("arg")
    if isinstance(arg, Mapping):
        inst_id = arg.get("instId")
        return str(inst_id) if inst_id is not None else None
    return None


def _message_data(message: Mapping[str, Any]) -> list[Any]:
    data = message.get("data", [])
    return data if isinstance(data, list) else []


def _timestamp_ms_to_utc_string(value: Any) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _timestamp_ms_to_day(value: Any) -> str | None:
    timestamp = _timestamp_ms_to_utc_string(value)
    return timestamp[:10] if timestamp else None


def _utc_now_string() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
