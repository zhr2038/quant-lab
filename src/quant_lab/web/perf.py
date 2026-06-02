from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any


@dataclass
class WebPerfEvent:
    event: str
    page_name: str = ""
    dataset_name: str = ""
    elapsed_ms: float = 0.0
    cache_hit: bool = False
    cache_miss: bool = False
    rglob_fallback: bool = False
    files_scanned: int = 0
    rows_rendered: int = 0
    warning: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "event": self.event,
            "page_name": self.page_name,
            "dataset_name": self.dataset_name,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "cache_hit": self.cache_hit,
            "cache_miss": self.cache_miss,
            "rglob_fallback": self.rglob_fallback,
            "files_scanned": self.files_scanned,
            "rows_rendered": self.rows_rendered,
            "warning": self.warning,
        }


_EVENTS: deque[WebPerfEvent] = deque(maxlen=500)


def record_event(event: str, **fields: Any) -> None:
    _EVENTS.append(WebPerfEvent(event=event, **fields))


@contextmanager
def timed(event: str, **fields: Any):
    start = perf_counter()
    try:
        yield
    finally:
        record_event(event, elapsed_ms=(perf_counter() - start) * 1000, **fields)


def recent_events(limit: int = 50) -> list[dict[str, Any]]:
    return [event.as_dict() for event in list(_EVENTS)[-limit:]]


def clear_events() -> None:
    _EVENTS.clear()
