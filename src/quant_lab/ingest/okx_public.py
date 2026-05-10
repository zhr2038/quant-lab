import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import MarketBar
from quant_lab.data.lake import MARKET_BAR_DATASET as MARKET_BAR_DATASET
from quant_lab.data.lake import write_market_bars

OKX_PUBLIC_REST_SOURCE = "okx_public_rest"


class OKXPublicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = "https://www.okx.com"
    timeout_seconds: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    rate_limit_sleep_seconds: float = Field(default=0.2, ge=0)


class OKXPublicError(RuntimeError):
    pass


class OKXPublicTimeout(OKXPublicError):
    pass


class OKXPublicAPIError(OKXPublicError):
    pass


OKXPublicTimeoutError = OKXPublicTimeout
OKXPublicResponseError = OKXPublicAPIError


class OKXPublicClient:
    def __init__(
        self,
        config: OKXPublicConfig | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config or OKXPublicConfig()
        self._client = http_client or httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            headers={"Accept": "application/json"},
        )

    def get_instruments(self, inst_type: str) -> list[dict[str, Any]]:
        return self._get_list("/api/v5/public/instruments", {"instType": inst_type})

    def get_tickers(self, inst_type: str) -> list[dict[str, Any]]:
        return self._get_list("/api/v5/market/tickers", {"instType": inst_type})

    def get_candles(
        self,
        inst_id: str,
        bar: str,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        data = self._get_list(
            "/api/v5/market/candles",
            _drop_none(
                {
                    "instId": inst_id,
                    "bar": bar,
                    "after": after,
                    "before": before,
                    "limit": str(limit),
                }
            ),
        )
        return [_coerce_candle_item(item) for item in data]

    def get_history_candles(
        self,
        inst_id: str,
        bar: str,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        data = self._get_list(
            "/api/v5/market/history-candles",
            _drop_none(
                {
                    "instId": inst_id,
                    "bar": bar,
                    "after": after,
                    "before": before,
                    "limit": str(limit),
                }
            ),
        )
        return [_coerce_candle_item(item) for item in data]

    def get_ticker(self, inst_id: str) -> dict[str, Any]:
        data = self._get_list("/api/v5/market/ticker", {"instId": inst_id})
        return data[0] if data else {}

    def get_orderbook(self, inst_id: str, sz: int = 20) -> dict[str, Any]:
        data = self._get_list("/api/v5/market/books", {"instId": inst_id, "sz": str(sz)})
        return data[0] if data else {}

    def get_trades(self, inst_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._get_list(
            "/api/v5/market/trades",
            {"instId": inst_id, "limit": str(limit)},
        )

    def get_history_trades(
        self,
        inst_id: str,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._get_list(
            "/api/v5/market/history-trades",
            _drop_none(
                {
                    "instId": inst_id,
                    "after": after,
                    "before": before,
                    "limit": str(limit),
                }
            ),
        )

    def _get_list(self, path: str, params: Mapping[str, str]) -> list[dict[str, Any]]:
        payload = self._get(path, params=params)
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise OKXPublicAPIError(f"OKX public response data is not a list for {path}")
        return data

    def _get(self, path: str, params: Mapping[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        attempts = self.config.max_retries + 1

        for attempt in range(attempts):
            try:
                response = self._client.request("GET", path, params=dict(params))
                if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    self._sleep_before_retry()
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise OKXPublicAPIError("OKX public response is not a JSON object")
                if payload.get("code") != "0":
                    raise OKXPublicAPIError(
                        f"OKX public API error code={payload.get('code')} msg={payload.get('msg')}"
                    )
                return payload
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    self._sleep_before_retry()
                    continue
                raise OKXPublicTimeout(f"OKX public request timed out for {path}") from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    self._sleep_before_retry()
                    continue
                raise OKXPublicError(f"OKX public request failed for {path}: {exc}") from exc

        raise OKXPublicError(f"OKX public request failed for {path}: {last_error}")

    def _sleep_before_retry(self) -> None:
        if self.config.rate_limit_sleep_seconds > 0:
            time.sleep(self.config.rate_limit_sleep_seconds)


def normalize_okx_candles_to_market_bars(
    candles: Sequence[Sequence[Any] | Mapping[str, Any]],
    inst_id: str,
    bar: str,
    market_type: str,
    ingest_ts: datetime | None = None,
    now: datetime | None = None,
    source: str = OKX_PUBLIC_REST_SOURCE,
) -> list[MarketBar]:
    effective_ingest_ts = _ensure_utc(ingest_ts or datetime.now(UTC))
    effective_now = _ensure_utc(now or datetime.now(UTC))
    timeframe_ms = _timeframe_to_milliseconds(bar)

    market_bars: list[MarketBar] = []
    for candle in candles:
        normalized = _normalize_candle_payload(candle)
        if normalized is None:
            continue
        if not _is_closed_candle(normalized, timeframe_ms=timeframe_ms, now=effective_now):
            continue

        market_bars.append(
            MarketBar(
                venue="okx",
                symbol=inst_id.upper(),
                market_type=market_type.upper(),
                timeframe=bar,
                ts=_timestamp_ms_to_utc(normalized["ts"]),
                open=float(normalized["open"]),
                high=float(normalized["high"]),
                low=float(normalized["low"]),
                close=float(normalized["close"]),
                volume=float(normalized["volume"]),
                quote_volume=float(normalized["quote_volume"]),
                source=source,
                ingest_ts=effective_ingest_ts,
                is_closed=True,
            )
        )
    return sorted(market_bars, key=lambda record: record.ts)


def publish_market_bars_to_lake(records: Sequence[MarketBar], lake_root: str | Path) -> int:
    return write_market_bars(lake_root, records)


def _normalize_candle_payload(
    candle: Sequence[Any] | Mapping[str, Any],
) -> dict[str, Any] | None:
    if isinstance(candle, Mapping):
        ts = candle.get("ts")
        open_price = _first_present(candle, "o", "open")
        high = _first_present(candle, "h", "high")
        low = _first_present(candle, "l", "low")
        close = _first_present(candle, "c", "close")
        volume = _first_present(candle, "vol", "volume") or 0
        quote_volume = _first_present(candle, "volCcyQuote", "quote_volume") or 0
        confirm = candle.get("confirm")
    else:
        if len(candle) < 6:
            return None
        ts = candle[0]
        open_price = candle[1]
        high = candle[2]
        low = candle[3]
        close = candle[4]
        volume = candle[5]
        quote_volume = candle[7] if len(candle) > 7 else 0
        confirm = candle[8] if len(candle) > 8 else None

    if None in {ts, open_price, high, low, close}:
        return None

    return {
        "ts": str(ts),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": quote_volume,
        "confirm": None if confirm is None else str(confirm),
    }


def _is_closed_candle(candle: Mapping[str, Any], timeframe_ms: int | None, now: datetime) -> bool:
    confirm = candle.get("confirm")
    if confirm is not None:
        return confirm == "1"
    if timeframe_ms is None:
        return False
    candle_start_ms = int(candle["ts"])
    now_ms = int(now.timestamp() * 1000)
    return candle_start_ms + timeframe_ms <= now_ms


def _timeframe_to_milliseconds(bar: str) -> int | None:
    normalized = bar.strip()
    match normalized:
        case "1m":
            return 60_000
        case "3m":
            return 3 * 60_000
        case "5m":
            return 5 * 60_000
        case "15m":
            return 15 * 60_000
        case "30m":
            return 30 * 60_000
        case "1H":
            return 60 * 60_000
        case "2H":
            return 2 * 60 * 60_000
        case "4H":
            return 4 * 60 * 60_000
        case "6H":
            return 6 * 60 * 60_000
        case "12H":
            return 12 * 60 * 60_000
        case "1D" | "1Dutc":
            return 24 * 60 * 60_000
        case "1W" | "1Wutc":
            return 7 * 24 * 60 * 60_000
        case "1M" | "1Mutc":
            return 31 * 24 * 60 * 60_000
        case _:
            return None


def _timestamp_ms_to_utc(value: str) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    return value.astimezone(UTC)


def _drop_none(values: Mapping[str, str | None]) -> dict[str, str]:
    return {key: value for key, value in values.items() if value is not None}


def _coerce_candle_item(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    if not isinstance(item, Sequence):
        raise OKXPublicAPIError("OKX candle item is not an array or object")
    keys = ["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"]
    return {key: item[index] for index, key in enumerate(keys) if index < len(item)}


def _first_present(values: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values and values[key] is not None:
            return values[key]
    return None
