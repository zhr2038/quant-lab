"""Small, read-only current snapshots; no lake scans or trading requests."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from quant_lab.decision.contracts import SYMBOLS, CostObservation, HourBar, InputSnapshot
from quant_lab.decision.current_cost_contracts import (
    CurrentCostDetail,
    CurrentCostObservation,
    FeeRate,
    SizeCost,
)
from quant_lab.ingest.okx_public import OKXPublicClient, OKXPublicConfig, OKXPublicError
from quant_lab.ingest.okx_readonly_private import (
    OKXReadOnlyClient,
    OKXReadOnlyConfig,
    OKXReadOnlyError,
)

FEE_REFRESH = timedelta(hours=6)
FEE_MAX_AGE = timedelta(hours=24)
BOOK_LIFETIME = timedelta(minutes=15)
UNCERTAINTY_ROUNDTRIP_BPS = 6.0  # Explicit research assumption, not a fitted percentile.
READ_ERRORS = (OKXPublicError, OKXReadOnlyError, ValueError, KeyError, TypeError, IndexError)


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp_ms(value: Any) -> datetime:
    return datetime.fromtimestamp(number(value) / 1000, UTC)


def number(value: Any) -> float:
    if value is None or isinstance(value, bool) or value == "":
        raise ValueError("missing numeric observation")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite observation")
    return result


def parse_fee(symbol: str, rows: list[dict], group: str, now: datetime) -> FeeRate:
    if len(rows) != 1 or rows[0].get("instType") != "SPOT" or not group:
        raise ValueError("fee response must identify one spot instrument group")
    row = rows[0]
    groups = row.get("feeGroup")
    if groups is not None:
        matched = [r for r in groups if str(r.get("groupId")) == group]
        if len(matched) != 1:
            raise ValueError("fee group not uniquely matched to public instrument")
        selected = matched[0]
    else:
        # Older endpoint version, called with this exact USDT spot instId.
        selected = row
    exchange_at = timestamp_ms(row["ts"])
    if not now - timedelta(minutes=5) <= exchange_at <= now:
        raise ValueError("fee response timestamp is stale or future")
    return FeeRate(
        symbol=symbol,
        group_id=group,
        maker_bps=-number(selected["maker"]) * 10_000,
        taker_bps=-number(selected["taker"]) * 10_000,
        fetched_at=now,
        exchange_at=exchange_at,
    )


def validate_book(book: dict, now: datetime) -> tuple[datetime, list, list]:
    ts = timestamp_ms(book["ts"])
    if not now - timedelta(seconds=60) <= ts <= now:
        raise ValueError("book observation stale or future")
    sides = []
    for side in ("asks", "bids"):
        raw = book[side]
        if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
            raise ValueError("invalid book depth")
        levels = [(number(row[0]), number(row[1])) for row in raw]
        if any(p <= 0 or q <= 0 for p, q in levels):
            raise ValueError("non-positive book level")
        prices = [p for p, _ in levels]
        if len(set(prices)) != len(prices) or prices != sorted(prices, reverse=side == "bids"):
            raise ValueError("unordered or duplicate book levels")
        sides.append(levels)
    asks, bids = sides
    if bids[0][0] >= asks[0][0]:
        raise ValueError("crossed or locked book")
    return ts, asks, bids


def vwap(levels: list, quantity: float) -> float | None:
    remaining, quote = quantity, 0.0
    for price, size in levels:
        taken = min(remaining, size)
        quote += price * taken
        remaining -= taken
        if remaining <= quantity * 1e-12:
            return quote / quantity
    return None


def estimate_sizes(asks, bids, fee: FeeRate, instrument: dict, notionals: list[float]):
    mid = (asks[0][0] + bids[0][0]) / 2
    minimum = number(instrument["minSz"])
    if minimum <= 0:
        raise ValueError("invalid instrument minimum")
    scenarios = []
    for notional in notionals:
        quantity = notional / mid
        buy, sell = vwap(asks, quantity), vwap(bids, quantity)
        status = "ESTIMATED"
        if quantity < minimum:
            status = "BELOW_MINIMUM_SIZE"
        elif buy is None or sell is None:
            status = "INSUFFICIENT_DEPTH"
        buy_bps = max(0, (buy / mid - 1) * 10_000) if buy is not None else None
        sell_bps = max(0, (1 - sell / mid) * 10_000) if sell is not None else None
        book_bps = buy_bps + sell_bps if buy_bps is not None and sell_bps is not None else None
        # Rebate is retained in the fee observation, but not credited before execution.
        fee_bps = 2 * max(0, fee.taker_bps)
        scenarios.append(
            SizeCost(
                notional_usdt=notional,
                base_quantity=quantity,
                status=status,
                buy_deviation_bps=buy_bps,
                sell_deviation_bps=sell_bps,
                book_roundtrip_bps=book_bps,
                fee_roundtrip_bps=fee_bps,
                uncertainty_bps=UNCERTAINTY_ROUNDTRIP_BPS,
                roundtrip_bps=(book_bps + fee_bps + UNCERTAINTY_ROUNDTRIP_BPS)
                if status == "ESTIMATED"
                else None,
            )
        )
    return scenarios


def current_cost(
    symbol: str,
    *,
    fee: FeeRate | None,
    book: dict,
    instrument: dict,
    now: datetime,
    anchor: CostObservation | None,
    notional_usdt: float,
    warnings: list[str],
) -> CurrentCostObservation:
    reasons = list(warnings)
    if fee is not None and not timedelta(0) <= now - fee.fetched_at < FEE_MAX_AGE:
        fee = None
        reasons.append("ACCOUNT_FEE_EXPIRED")
    book_at, sizes = None, []
    try:
        book_at, asks, bids = validate_book(book, now)
        if instrument.get("state") != "live":
            raise ValueError("instrument is not live")
        if fee is not None:
            if fee.group_id != str(instrument.get("groupId", "")):
                raise ValueError("cached fee group changed")
            sizes = estimate_sizes(
                asks, bids, fee, instrument, sorted({notional_usdt, 20.0, 50.0, 100.0})
            )
    except READ_ERRORS:
        reasons.append("CURRENT_BOOK_OR_INSTRUMENT_UNAVAILABLE")
    if fee is None:
        reasons.append("ACCOUNT_FEE_UNAVAILABLE")
    selected = next((s for s in sizes if s.notional_usdt == notional_usdt), None)
    total = selected.roundtrip_bps if selected else None
    if selected and selected.status != "ESTIMATED":
        reasons.append(selected.status)
    reasons.extend(["COST_ESTIMATE_UNCALIBRATED", "EXIT_BOOK_ASSUMED_CURRENT"])
    if fee is not None and fee.taker_bps < 0:
        reasons.append("REBATE_NOT_CREDITED_IN_ESTIMATE")
    detail = CurrentCostDetail(
        status="ESTIMATED" if total is not None else "UNAVAILABLE",
        fee=fee,
        book_as_of=book_at,
        valid_until=min(book_at + BOOK_LIFETIME, fee.fetched_at + FEE_MAX_AGE)
        if total is not None
        else None,
        sizes=sizes,
        historical_anchor=anchor,
    )
    return CurrentCostObservation(
        symbol=symbol,
        roundtrip_bps=total,
        notional_usdt=notional_usdt,
        source="okx_account_fee_and_book",
        quality="estimated_uncalibrated",
        version="current-cost-v1",
        as_of=book_at,
        missing_reasons=list(dict.fromkeys(reasons)),
        current=detail,
    )


def merge_closed_candles(symbol: str, rows: list[dict], old: list[HourBar], now: datetime):
    if len(rows) > 200:
        raise ValueError("candle response exceeds budget")
    merged = {b.ts: b for b in old if b.symbol == symbol}
    seen = set()
    for row in rows:
        if str(row["confirm"]) != "1":
            continue
        ts = timestamp_ms(row["ts"])
        if ts in seen:
            raise ValueError("duplicate closed candle in exchange response")
        seen.add(ts)
        if ts + timedelta(hours=1) > now:
            raise ValueError("confirmed candle is in the future")
        bar = HourBar(
            symbol=symbol,
            ts=ts,
            open=number(row["o"]),
            high=number(row["h"]),
            low=number(row["l"]),
            close=number(row["c"]),
            volume=number(row["vol"]),
            ingest_ts=now,
        )
        prior = merged.get(ts)
        if prior and all(
            getattr(prior, k) == getattr(bar, k) for k in ("open", "high", "low", "close", "volume")
        ):
            bar = prior  # Re-reading the same candle is not a new first observation.
        merged[ts] = bar
    return [merged[t] for t in sorted(merged) if now - timedelta(days=8) <= t]


@dataclass
class CurrentInputs:
    generated_at: datetime
    bars: list[HourBar]
    costs: list[CurrentCostObservation]
    warnings: list[str]


def collect_current_inputs(
    previous: InputSnapshot | None,
    *,
    notional_usdt: float = 20,
    clock: Callable[[], datetime] = utc_now,
    public: OKXPublicClient | None = None,
    private: OKXReadOnlyClient | None = None,
) -> CurrentInputs:
    own_public, own_private = public is None, False
    public = public or OKXPublicClient(OKXPublicConfig(timeout_seconds=4, max_retries=0))
    old_bars = previous.bars if previous else []
    old_costs = {c.symbol: c for c in previous.costs} if previous else {}
    warnings, bars, costs = [], [], []
    instruments, private_error = {}, None
    try:
        try:
            instruments = {r["instId"]: r for r in public.get_instruments("SPOT")}
        except READ_ERRORS:
            warnings.append("CURRENT_INSTRUMENTS_UNAVAILABLE")
        fees = {
            s: c.current.fee
            for s, c in old_costs.items()
            if isinstance(c, CurrentCostObservation) and c.current.fee is not None
        }
        refresh = {
            s
            for s in SYMBOLS
            if s not in fees or not timedelta(0) <= clock() - fees[s].fetched_at < FEE_REFRESH
        }
        if refresh:
            try:
                if private is None:
                    private = OKXReadOnlyClient(
                        OKXReadOnlyConfig.from_env().model_copy(
                            update={"timeout_seconds": 4, "max_retries": 0}
                        )
                    )
                    own_private = True
                permissions = private.get_account_config().get("perm", "")
                if set(permissions.split(",")) != {"read_only"}:
                    raise ValueError("fee reader requires a proven read-only key")
            except READ_ERRORS:
                private_error = "READONLY_FEE_ACCESS_UNAVAILABLE"
        for symbol in SYMBOLS:
            inst = symbol.removesuffix("USDT") + "-USDT"
            instrument = instruments.get(inst, {})
            cost_warnings = []
            if symbol in refresh:
                try:
                    if private_error:
                        raise ValueError(private_error)
                    rows = private.get_spot_fee_rates(inst)
                    fees[symbol] = parse_fee(symbol, rows, str(instrument["groupId"]), clock())
                except READ_ERRORS:
                    cost_warnings.append(private_error or "ACCOUNT_FEE_REFRESH_FAILED")
                    if symbol in fees:
                        cost_warnings.append("ACCOUNT_FEE_USING_PREVIOUS_OBSERVATION")
            try:
                rows = public.get_candles(inst, "1H", limit=200)
                recent = merge_closed_candles(symbol, rows, old_bars, clock())
                if not rows:
                    warnings.append(f"{symbol}:CURRENT_CANDLES_EMPTY")
            except READ_ERRORS:
                warnings.append(f"{symbol}:CURRENT_CANDLE_FETCH_FAILED")
                recent = merge_closed_candles(symbol, [], old_bars, clock())
            bars.extend(recent)
            if not recent or recent[-1].ts + timedelta(hours=2) <= clock():
                warnings.append(f"{symbol}:CURRENT_MARKET_STALE_OR_MISSING")
            try:
                book = public.get_orderbook(inst, sz=20)
            except READ_ERRORS:
                book = {}
                cost_warnings.append("CURRENT_BOOK_FETCH_FAILED")
            old = old_costs.get(symbol)
            anchor = (
                old.current.historical_anchor if isinstance(old, CurrentCostObservation) else old
            )
            cost = current_cost(
                symbol,
                fee=fees.get(symbol),
                book=book,
                instrument=instrument,
                now=clock(),
                anchor=anchor,
                notional_usdt=notional_usdt,
                warnings=cost_warnings,
            )
            costs.append(cost)
            warnings.extend(f"{symbol}:{w}" for w in cost_warnings)
            if cost.roundtrip_bps is None:
                warnings.append(f"{symbol}:CURRENT_COST_UNAVAILABLE")
        return CurrentInputs(generated_at=clock(), bars=bars, costs=costs, warnings=warnings)
    finally:
        if own_public:
            public._client.close()
        if own_private:
            private._client.close()
