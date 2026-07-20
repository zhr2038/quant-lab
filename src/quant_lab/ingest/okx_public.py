import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import MarketBar
from quant_lab.data.lake import MARKET_BAR_DATASET as MARKET_BAR_DATASET
from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.symbols import normalize_symbol

OKX_PUBLIC_REST_SOURCE = "okx_public_rest"
OKX_EXPANDED_UNIVERSE_SOURCE = "okx_public_rest_expanded_universe"
OKX_FACTOR_RESEARCH_HISTORY_SOURCE = "okx_public_rest_factor_research_history"
OKX_SPOT_UNIVERSE_CANDIDATES_DATASET = (
    Path("bronze") / "okx_public_rest" / "spot_universe_candidates"
)
CURRENT_V5_UNIVERSE = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"}
STABLE_BASES = {
    "USDT",
    "USDC",
    "USD",
    "DAI",
    "FDUSD",
    "TUSD",
    "USDD",
    "USDE",
    "PYUSD",
    "BUSD",
    "RLUSD",
    "USDG",
    "EURT",
}
MEME_BASES = {
    "DOGE",
    "SHIB",
    "PEPE",
    "FLOKI",
    "BONK",
    "WIF",
    "MEME",
    "TURBO",
    "BABYDOGE",
    "AIDOGE",
}
LEVERAGED_SUFFIXES = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")

SPOT_UNIVERSE_CANDIDATE_SCHEMA = {
    "generated_at": pl.Datetime(time_zone="UTC"),
    "rank": pl.Int64,
    "symbol": pl.Utf8,
    "base_ccy": pl.Utf8,
    "quote_ccy": pl.Utf8,
    "quote_volume_24h": pl.Float64,
    "last_px": pl.Float64,
    "bid_px": pl.Float64,
    "ask_px": pl.Float64,
    "spread_bps": pl.Float64,
    "is_current_v5_symbol": pl.Boolean,
    "source": pl.Utf8,
}


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


class OKXSpotUniverseCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    base_ccy: str
    quote_ccy: str
    quote_volume_24h: float = Field(ge=0)
    last_px: float = Field(gt=0)
    bid_px: float = Field(gt=0)
    ask_px: float = Field(gt=0)
    spread_bps: float = Field(ge=0)
    is_current_v5_symbol: bool = False


class OKXExpandedUniverseBackfillResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    selected_symbols: list[str]
    candidates_considered: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    fetched_candles: int = Field(ge=0)
    published_market_bars: int = Field(ge=0)
    market_bar_rows: int = Field(ge=0)
    candidate_dataset_path: str
    market_bar_dataset_path: str
    gap_repair_symbols: list[str] = Field(default_factory=list)
    gap_repair_history_pages: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class OKXFactorResearchHistoryBackfillResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    symbols: list[str]
    bar: str
    start_at: datetime
    end_at: datetime
    expected_bars_per_symbol: int = Field(ge=1)
    fetched_pages: int = Field(ge=0)
    fetched_candles: int = Field(ge=0)
    published_market_bars: int = Field(ge=0)
    market_bar_rows: int = Field(ge=0)
    coverage_by_symbol: dict[str, float]
    earliest_by_symbol: dict[str, datetime | None]
    latest_by_symbol: dict[str, datetime | None]
    skipped_symbols: list[str]
    complete: bool
    minimum_coverage: float = Field(gt=0, le=1)
    warnings: list[str] = Field(default_factory=list)


def select_okx_usdt_spot_universe(
    *,
    instruments: Sequence[Mapping[str, Any]],
    tickers: Sequence[Mapping[str, Any]],
    max_symbols: int = 30,
    min_quote_volume_24h: float = 1_000_000.0,
    max_spread_bps: float = 20.0,
    min_price: float = 0.01,
    blacklist: Sequence[str] | None = None,
) -> list[OKXSpotUniverseCandidate]:
    barred = {normalize_symbol(item) for item in blacklist or []}
    live_instruments = _spot_instruments_by_symbol(instruments)
    candidates: list[OKXSpotUniverseCandidate] = []
    for ticker in tickers:
        symbol = normalize_symbol(ticker.get("instId"))
        instrument = live_instruments.get(symbol)
        if instrument is None:
            continue
        base_ccy, quote_ccy = _instrument_ccys(symbol, instrument)
        if not _is_allowed_spot_symbol(
            symbol=symbol,
            base_ccy=base_ccy,
            quote_ccy=quote_ccy,
            blacklist=barred,
        ):
            continue
        last_px = _float(ticker.get("last"))
        bid_px = _float(ticker.get("bidPx"))
        ask_px = _float(ticker.get("askPx"))
        if last_px is None or bid_px is None or ask_px is None:
            continue
        if last_px < min_price or bid_px <= 0 or ask_px <= bid_px:
            continue
        spread_bps = (ask_px - bid_px) / ((ask_px + bid_px) / 2.0) * 10_000.0
        if spread_bps > max_spread_bps:
            continue
        quote_volume = _ticker_quote_volume_24h(ticker, last_px=last_px)
        if quote_volume < min_quote_volume_24h:
            continue
        candidates.append(
            OKXSpotUniverseCandidate(
                symbol=symbol,
                base_ccy=base_ccy,
                quote_ccy=quote_ccy,
                quote_volume_24h=quote_volume,
                last_px=last_px,
                bid_px=bid_px,
                ask_px=ask_px,
                spread_bps=spread_bps,
                is_current_v5_symbol=symbol in CURRENT_V5_UNIVERSE,
            )
        )
    candidates.sort(
        key=lambda item: (
            item.is_current_v5_symbol,
            item.quote_volume_24h,
            -item.spread_bps,
        ),
        reverse=True,
    )
    return candidates[: max(max_symbols, 1)]


def backfill_expanded_usdt_spot_market_bars(
    *,
    lake_root: str | Path,
    client: OKXPublicClient | None = None,
    bar: str = "1H",
    market_type: str = "SPOT",
    max_symbols: int = 30,
    history_pages: int = 8,
    gap_repair_pages: int = 20,
    limit: int = 100,
    min_quote_volume_24h: float = 1_000_000.0,
    max_spread_bps: float = 20.0,
    min_price: float = 0.01,
    blacklist: Sequence[str] | None = None,
) -> OKXExpandedUniverseBackfillResult:
    root = Path(lake_root)
    effective_client = client or OKXPublicClient()
    generated_at = datetime.now(UTC)
    instruments = effective_client.get_instruments("SPOT")
    tickers = effective_client.get_tickers("SPOT")
    candidates = select_okx_usdt_spot_universe(
        instruments=instruments,
        tickers=tickers,
        max_symbols=max_symbols,
        min_quote_volume_24h=min_quote_volume_24h,
        max_spread_bps=max_spread_bps,
        min_price=min_price,
        blacklist=blacklist,
    )
    _write_spot_universe_candidates(root, candidates, generated_at=generated_at)
    gap_repair_symbols = _market_bar_symbols_with_internal_gaps(
        root,
        symbols=[candidate.symbol for candidate in candidates],
        bar=bar,
    )
    effective_gap_repair_pages = max(history_pages, gap_repair_pages)

    warnings: list[str] = []
    market_bars: list[MarketBar] = []
    fetched_candles = 0
    for candidate in candidates:
        after: str | None = None
        seen_cursors: set[str] = set()
        symbol_bar_count = 0
        latest_candles = effective_client.get_candles(
            candidate.symbol,
            bar,
            limit=limit,
        )
        if latest_candles:
            fetched_candles += len(latest_candles)
            market_bars.extend(
                normalized_bars := normalize_okx_candles_to_market_bars(
                    latest_candles,
                    inst_id=candidate.symbol,
                    bar=bar,
                    market_type=market_type,
                    source=OKX_EXPANDED_UNIVERSE_SOURCE,
                )
            )
            symbol_bar_count += len(normalized_bars)
            after = _oldest_candle_ts(latest_candles)
            if after is not None:
                seen_cursors.add(after)
        page_count = (
            effective_gap_repair_pages if candidate.symbol in gap_repair_symbols else history_pages
        )
        for _ in range(max(page_count, 1)):
            candles = effective_client.get_history_candles(
                candidate.symbol,
                bar,
                after=after,
                limit=limit,
            )
            if not candles:
                break
            fetched_candles += len(candles)
            market_bars.extend(
                normalized_bars := normalize_okx_candles_to_market_bars(
                    candles,
                    inst_id=candidate.symbol,
                    bar=bar,
                    market_type=market_type,
                    source=OKX_EXPANDED_UNIVERSE_SOURCE,
                )
            )
            symbol_bar_count += len(normalized_bars)
            next_after = _oldest_candle_ts(candles)
            if next_after is None or next_after in seen_cursors:
                break
            seen_cursors.add(next_after)
            after = next_after
            if len(candles) < limit:
                break
        if symbol_bar_count == 0:
            warnings.append(f"no_closed_market_bars:{candidate.symbol}")
    unique_bars = _dedupe_market_bars(market_bars)
    market_bar_rows = publish_market_bars_to_lake(unique_bars, root) if unique_bars else 0
    return OKXExpandedUniverseBackfillResult(
        lake_root=str(root),
        selected_symbols=[candidate.symbol for candidate in candidates],
        candidates_considered=len(tickers),
        selected_count=len(candidates),
        fetched_candles=fetched_candles,
        published_market_bars=len(unique_bars),
        market_bar_rows=market_bar_rows,
        candidate_dataset_path=str(root / OKX_SPOT_UNIVERSE_CANDIDATES_DATASET),
        market_bar_dataset_path=str(root / MARKET_BAR_DATASET),
        gap_repair_symbols=sorted(gap_repair_symbols),
        gap_repair_history_pages=effective_gap_repair_pages,
        warnings=warnings,
    )


def backfill_factor_research_market_history(
    *,
    lake_root: str | Path,
    start_at: datetime,
    end_at: datetime,
    symbols: Sequence[str] = tuple(sorted(CURRENT_V5_UNIVERSE)),
    client: OKXPublicClient | None = None,
    bar: str = "1H",
    market_type: str = "SPOT",
    max_pages_per_symbol: int = 100,
    limit: int = 300,
    minimum_coverage: float = 0.98,
    page_sleep_seconds: float = 0.11,
) -> OKXFactorResearchHistoryBackfillResult:
    """Idempotently deepen the locked Factor Research market history.

    This uses only OKX public history candles. Existing lake rows are the
    checkpoint: a retry resumes from each symbol's earliest stored bar and an
    atomic market-bar upsert preserves already completed symbols.
    """
    root = Path(lake_root)
    start = _ensure_utc(start_at)
    end = _ensure_utc(end_at)
    if end <= start:
        raise ValueError("factor_research_history_end_must_follow_start")
    if max_pages_per_symbol < 1:
        raise ValueError("max_pages_per_symbol must be positive")
    if not 1 <= limit <= 300:
        raise ValueError("limit must be between 1 and 300")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if page_sleep_seconds < 0:
        raise ValueError("page_sleep_seconds must be non-negative")
    timeframe_ms = _timeframe_to_milliseconds(bar)
    if timeframe_ms is None:
        raise ValueError(f"unsupported_factor_research_bar:{bar}")
    normalized_symbols = sorted(
        {normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)}
    )
    if not normalized_symbols:
        raise ValueError("factor_research_history_symbols_empty")

    expected_bars = max(
        1,
        math.ceil((end - start).total_seconds() * 1000 / timeframe_ms),
    )
    effective_client = client or OKXPublicClient()
    before = _market_bar_history_coverage(
        root,
        symbols=normalized_symbols,
        bar=bar,
        start_at=start,
        end_at=end,
        expected_bars=expected_bars,
    )
    fetched_pages = 0
    fetched_candles = 0
    published_market_bars = 0
    skipped_symbols: list[str] = []
    warnings: list[str] = []

    for symbol in normalized_symbols:
        existing_coverage = before["coverage_by_symbol"].get(symbol, 0.0)
        existing_earliest = before["earliest_by_symbol"].get(symbol)
        if existing_coverage >= minimum_coverage and (
            existing_earliest is not None
            and existing_earliest <= start + timedelta(milliseconds=timeframe_ms)
        ):
            skipped_symbols.append(symbol)
            continue
        has_window_start = (
            existing_earliest is not None
            and existing_earliest <= start + timedelta(milliseconds=timeframe_ms)
        )
        # If the start is already present but coverage is incomplete, the gap
        # is internal and pagination must restart from the latest page.
        # Otherwise continue backwards from the earliest durable checkpoint.
        after = (
            None
            if existing_earliest is None or has_window_start
            else str(int(existing_earliest.timestamp() * 1000))
        )
        seen_cursors = {after} if after is not None else set()
        symbol_records: list[MarketBar] = []
        reached_start = False
        for _ in range(max_pages_per_symbol):
            candles = effective_client.get_history_candles(
                symbol,
                bar,
                after=after,
                limit=limit,
            )
            fetched_pages += 1
            if not candles:
                break
            fetched_candles += len(candles)
            normalized = normalize_okx_candles_to_market_bars(
                candles,
                inst_id=symbol,
                bar=bar,
                market_type=market_type,
                source=OKX_FACTOR_RESEARCH_HISTORY_SOURCE,
            )
            symbol_records.extend(record for record in normalized if start <= record.ts < end)
            next_after = _oldest_candle_ts(candles)
            if next_after is None or next_after in seen_cursors:
                warnings.append(f"history_cursor_stalled:{symbol}")
                break
            oldest_at = datetime.fromtimestamp(int(next_after) / 1000.0, tz=UTC)
            if oldest_at <= start:
                reached_start = True
                break
            seen_cursors.add(next_after)
            after = next_after
            if len(candles) < limit:
                break
            if page_sleep_seconds > 0:
                time.sleep(page_sleep_seconds)
        unique_records = _dedupe_market_bars(symbol_records)
        if unique_records:
            publish_market_bars_to_lake(unique_records, root)
            published_market_bars += len(unique_records)
        if not reached_start and (
            existing_earliest is None
            or existing_earliest > start + timedelta(milliseconds=timeframe_ms)
        ):
            warnings.append(f"history_start_not_reached:{symbol}")

    coverage = _market_bar_history_coverage(
        root,
        symbols=normalized_symbols,
        bar=bar,
        start_at=start,
        end_at=end,
        expected_bars=expected_bars,
    )
    incomplete = [
        symbol
        for symbol in normalized_symbols
        if coverage["coverage_by_symbol"].get(symbol, 0.0) < minimum_coverage
    ]
    warnings.extend(f"history_coverage_below_threshold:{symbol}" for symbol in incomplete)
    return OKXFactorResearchHistoryBackfillResult(
        lake_root=str(root),
        symbols=normalized_symbols,
        bar=bar,
        start_at=start,
        end_at=end,
        expected_bars_per_symbol=expected_bars,
        fetched_pages=fetched_pages,
        fetched_candles=fetched_candles,
        published_market_bars=published_market_bars,
        market_bar_rows=coverage["market_bar_rows"],
        coverage_by_symbol=coverage["coverage_by_symbol"],
        earliest_by_symbol=coverage["earliest_by_symbol"],
        latest_by_symbol=coverage["latest_by_symbol"],
        skipped_symbols=skipped_symbols,
        complete=not incomplete,
        minimum_coverage=minimum_coverage,
        warnings=sorted(set(warnings)),
    )


def _market_bar_history_coverage(
    lake_root: Path,
    *,
    symbols: Sequence[str],
    bar: str,
    start_at: datetime,
    end_at: datetime,
    expected_bars: int,
) -> dict[str, Any]:
    try:
        market = read_parquet_dataset(lake_root / MARKET_BAR_DATASET)
    except Exception:
        market = pl.DataFrame()
    coverage = {symbol: 0.0 for symbol in symbols}
    earliest: dict[str, datetime | None] = {symbol: None for symbol in symbols}
    latest: dict[str, datetime | None] = {symbol: None for symbol in symbols}
    if not market.is_empty() and {"symbol", "timeframe", "ts"}.issubset(market.columns):
        selected = (
            market.with_columns(
                pl.col("symbol")
                .cast(pl.Utf8)
                .str.to_uppercase()
                .str.replace_all("/", "-")
                .alias("symbol")
            )
            .filter(
                pl.col("symbol").is_in(list(symbols))
                & (pl.col("timeframe").cast(pl.Utf8).str.to_lowercase() == bar.lower())
                & (pl.col("ts") >= start_at)
                & (pl.col("ts") < end_at)
            )
            .select("symbol", "ts")
            .drop_nulls()
            .unique(["symbol", "ts"])
            .group_by("symbol")
            .agg(
                pl.col("ts").min().alias("earliest"),
                pl.col("ts").max().alias("latest"),
                pl.len().alias("bars"),
            )
        )
        for row in selected.to_dicts():
            symbol = str(row["symbol"])
            coverage[symbol] = min(1.0, int(row["bars"]) / expected_bars)
            earliest[symbol] = row["earliest"]
            latest[symbol] = row["latest"]
    return {
        "coverage_by_symbol": coverage,
        "earliest_by_symbol": earliest,
        "latest_by_symbol": latest,
        "market_bar_rows": market.height,
    }


def _market_bar_symbols_with_internal_gaps(
    lake_root: Path,
    *,
    symbols: Sequence[str],
    bar: str,
) -> set[str]:
    timeframe_ms = _timeframe_to_milliseconds(bar)
    if timeframe_ms is None or not symbols:
        return set()
    try:
        market = read_parquet_dataset(lake_root / MARKET_BAR_DATASET)
    except Exception:
        return set()
    required = {"symbol", "timeframe", "ts"}
    if market.is_empty() or not required.issubset(market.columns):
        return set()
    selected = {normalize_symbol(symbol) for symbol in symbols}
    try:
        stats = (
            market.filter(
                pl.col("symbol").is_in(sorted(selected))
                & (pl.col("timeframe").cast(pl.Utf8) == bar)
            )
            .select("symbol", "ts")
            .unique()
            .group_by("symbol")
            .agg(
                pl.col("ts").min().alias("first_ts"),
                pl.col("ts").max().alias("last_ts"),
                pl.len().alias("actual_bars"),
            )
        )
    except Exception:
        return set()
    gaps: set[str] = set()
    for row in stats.to_dicts():
        first_ts = row.get("first_ts")
        last_ts = row.get("last_ts")
        actual_bars = int(row.get("actual_bars") or 0)
        if not isinstance(first_ts, datetime) or not isinstance(last_ts, datetime):
            continue
        expected_bars = int((last_ts - first_ts).total_seconds() * 1000 / timeframe_ms) + 1
        if expected_bars > actual_bars:
            gaps.add(str(row.get("symbol") or ""))
    return gaps


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
                symbol=normalize_symbol(inst_id),
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


def _write_spot_universe_candidates(
    lake_root: Path,
    candidates: Sequence[OKXSpotUniverseCandidate],
    *,
    generated_at: datetime,
) -> None:
    rows = [
        {
            "generated_at": generated_at,
            "rank": rank,
            "symbol": candidate.symbol,
            "base_ccy": candidate.base_ccy,
            "quote_ccy": candidate.quote_ccy,
            "quote_volume_24h": candidate.quote_volume_24h,
            "last_px": candidate.last_px,
            "bid_px": candidate.bid_px,
            "ask_px": candidate.ask_px,
            "spread_bps": candidate.spread_bps,
            "is_current_v5_symbol": candidate.is_current_v5_symbol,
            "source": OKX_EXPANDED_UNIVERSE_SOURCE,
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]
    frame = (
        pl.DataFrame(rows, schema=SPOT_UNIVERSE_CANDIDATE_SCHEMA, orient="row")
        if rows
        else pl.DataFrame(schema=SPOT_UNIVERSE_CANDIDATE_SCHEMA)
    )
    write_parquet_dataset(frame, lake_root / OKX_SPOT_UNIVERSE_CANDIDATES_DATASET)


def _dedupe_market_bars(records: Sequence[MarketBar]) -> list[MarketBar]:
    keyed: dict[tuple[str, str, str, datetime], MarketBar] = {}
    for record in records:
        keyed[(record.venue, record.symbol, record.timeframe, record.ts)] = record
    return sorted(keyed.values(), key=lambda item: (item.symbol, item.timeframe, item.ts))


def _oldest_candle_ts(candles: Sequence[Sequence[Any] | Mapping[str, Any]]) -> str | None:
    timestamps: list[int] = []
    for candle in candles:
        normalized = _normalize_candle_payload(candle)
        if normalized is None:
            continue
        try:
            timestamps.append(int(normalized["ts"]))
        except (TypeError, ValueError):
            continue
    return str(min(timestamps)) if timestamps else None


def _spot_instruments_by_symbol(
    instruments: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for instrument in instruments:
        symbol = normalize_symbol(instrument.get("instId"))
        if not symbol:
            continue
        if str(instrument.get("instType") or "SPOT").upper() != "SPOT":
            continue
        if str(instrument.get("state") or "live").lower() not in {"live", ""}:
            continue
        output[symbol] = instrument
    return output


def _instrument_ccys(symbol: str, instrument: Mapping[str, Any]) -> tuple[str, str]:
    base = str(instrument.get("baseCcy") or "").upper()
    quote = str(instrument.get("quoteCcy") or "").upper()
    if base and quote:
        return base, quote
    if "-" in symbol:
        parsed_base, parsed_quote = symbol.split("-", 1)
        return parsed_base, parsed_quote
    return symbol, ""


def _is_allowed_spot_symbol(
    *,
    symbol: str,
    base_ccy: str,
    quote_ccy: str,
    blacklist: set[str],
) -> bool:
    if symbol in blacklist:
        return False
    if quote_ccy != "USDT" or not symbol.endswith("-USDT"):
        return False
    if base_ccy in STABLE_BASES or base_ccy in MEME_BASES:
        return False
    return not base_ccy.endswith(LEVERAGED_SUFFIXES)


def _ticker_quote_volume_24h(ticker: Mapping[str, Any], *, last_px: float) -> float:
    direct = _float(
        ticker.get("volCcyQuote24h")
        or ticker.get("quoteVolume24h")
        or ticker.get("quoteVol24h")
    )
    if direct is not None:
        return direct
    volume_ccy = _float(ticker.get("volCcy24h"))
    if volume_ccy is not None:
        return volume_ccy
    base_volume = _float(ticker.get("vol24h") or ticker.get("vol"))
    return (base_volume or 0.0) * last_px


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


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
