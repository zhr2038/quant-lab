from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

DEFAULT_MARKET_BAR_TIMEFRAME = "1H"
_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([smhdwSMHDW])\s*$")


def market_bar_interval_seconds(timeframe: str | None = None) -> int:
    text = str(timeframe or DEFAULT_MARKET_BAR_TIMEFRAME).strip()
    match = _TIMEFRAME_RE.match(text)
    if not match:
        return 60 * 60
    amount = max(1, int(match.group(1)))
    unit = match.group(2).lower()
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    return amount * multipliers[unit]


def ensure_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def market_bar_close_ts(
    open_ts: Any,
    timeframe: str | None = None,
) -> datetime | None:
    opened_at = ensure_utc_datetime(open_ts)
    if opened_at is None:
        return None
    return opened_at + timedelta(seconds=market_bar_interval_seconds(timeframe))


def market_bar_freshness_seconds(
    latest_open_ts: Any,
    *,
    latest_close_ts: Any = None,
    timeframe: str | None = None,
    now: datetime | None = None,
) -> int | None:
    reference = ensure_utc_datetime(latest_close_ts) or market_bar_close_ts(
        latest_open_ts,
        timeframe=timeframe,
    )
    if reference is None:
        return None
    current = ensure_utc_datetime(now) or datetime.now(UTC)
    return max(0, int((current - reference).total_seconds()))
