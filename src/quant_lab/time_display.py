from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

BEIJING_TZ = ZoneInfo("Asia/Shanghai")
DISPLAY_TIMEZONE = "Asia/Shanghai"


def beijing_today(now: datetime | None = None) -> date:
    current = now or datetime.now(UTC)
    return _ensure_datetime(current).astimezone(BEIJING_TZ).date()


def format_beijing_time(value: Any) -> str:
    timestamp = coerce_datetime(value)
    if timestamp is None:
        return str(value)
    return f"{timestamp.astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')} {DISPLAY_TIMEZONE}"


def beijing_iso(value: Any) -> str | None:
    timestamp = coerce_datetime(value)
    if timestamp is None:
        return None
    return timestamp.astimezone(BEIJING_TZ).isoformat()


def coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _ensure_datetime(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "n/a", "na"}:
        return None
    if not _looks_like_datetime(text):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _ensure_datetime(parsed)


def is_time_column(column: str) -> bool:
    lowered = column.lower()
    return (
        lowered in {"date", "day"}
        or lowered.endswith("_at")
        or lowered.endswith("_ts")
        or lowered.endswith("_time")
        or lowered.endswith("_timestamp")
        or lowered in {"ts", "timestamp", "modified_at", "latest_timestamp"}
    )


def _ensure_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _looks_like_datetime(text: str) -> bool:
    if "T" in text:
        return True
    if "+" in text and ":" in text:
        return True
    if text.endswith("Z"):
        return True
    return False
