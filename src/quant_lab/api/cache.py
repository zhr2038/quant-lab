from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class StrategyOpportunityAdvisorySnapshot(Generic[T]):
    signature: tuple[Any, ...]
    rows: tuple[T, ...]
    payload: bytes
    source_sha: str
    loaded_at: datetime
    lake_scan_ms: float
    serialize_ms: float


class StrategyOpportunityAdvisoryCache(Generic[T]):
    """In-memory advisory snapshot keyed by lake root and source signature."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[str, StrategyOpportunityAdvisorySnapshot[T]] = {}

    def clear(self) -> None:
        with self._lock:
            self._snapshots.clear()

    def get_snapshot(
        self,
        lake_root: Path,
        *,
        signature_builder: Callable[[Path], tuple[Any, ...]],
        loader: Callable[[Path], list[T]],
        serializer: Callable[[list[T]], bytes],
        monotonic_seconds: Callable[[], float],
    ) -> tuple[StrategyOpportunityAdvisorySnapshot[T], bool]:
        signature = signature_builder(lake_root)
        root_key = str(lake_root.resolve())
        with self._lock:
            cached = self._snapshots.get(root_key)
            if cached is not None and cached.signature == signature:
                return cached, True

        scan_started = monotonic_seconds()
        rows = loader(lake_root)
        lake_scan_ms = round((monotonic_seconds() - scan_started) * 1000.0, 3)
        serialize_started = monotonic_seconds()
        payload = serializer(rows)
        serialize_ms = round((monotonic_seconds() - serialize_started) * 1000.0, 3)
        snapshot = StrategyOpportunityAdvisorySnapshot(
            signature=signature,
            rows=tuple(rows),
            payload=payload,
            source_sha=_source_sha(signature),
            loaded_at=datetime.now(UTC),
            lake_scan_ms=lake_scan_ms,
            serialize_ms=serialize_ms,
        )
        with self._lock:
            current = self._snapshots.get(root_key)
            if current is not None and current.signature == signature:
                return current, True
            self._snapshots[root_key] = snapshot
        return snapshot, False


def _source_sha(signature: tuple[Any, ...]) -> str:
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()
