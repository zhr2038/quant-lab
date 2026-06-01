from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class StrategyOpportunityAdvisorySnapshot(Generic[T]):
    signature: tuple[Any, ...]
    rows: tuple[T, ...]
    payload: bytes
    source_sha: str
    loaded_at: datetime
    source_signature_ms: float
    lake_scan_ms: float
    serialize_ms: float


@dataclass(frozen=True)
class StrategyOpportunityAdvisoryResponse:
    key: tuple[Any, ...]
    payload: bytes
    etag: str
    row_count: int
    latest_generated_at: str
    serialize_ms: float
    created_at: datetime


@dataclass(frozen=True)
class CachedValue(Generic[T]):
    key: tuple[Any, ...]
    value: T
    created_at: datetime


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
    ) -> tuple[StrategyOpportunityAdvisorySnapshot[T], bool, float]:
        signature_started = monotonic_seconds()
        signature = signature_builder(lake_root)
        source_signature_ms = round((monotonic_seconds() - signature_started) * 1000.0, 3)
        root_key = str(lake_root.resolve())
        with self._lock:
            cached = self._snapshots.get(root_key)
            if cached is not None and cached.signature == signature:
                return cached, True, source_signature_ms

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
            source_signature_ms=source_signature_ms,
            lake_scan_ms=lake_scan_ms,
            serialize_ms=serialize_ms,
        )
        with self._lock:
            current = self._snapshots.get(root_key)
            if current is not None and current.signature == signature:
                return current, True, source_signature_ms
            self._snapshots[root_key] = snapshot
        return snapshot, False, source_signature_ms


class StrategyOpportunityAdvisoryResponseCache:
    """In-memory cache for filtered/serialized advisory responses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._responses: dict[tuple[Any, ...], StrategyOpportunityAdvisoryResponse] = {}

    def clear(self) -> None:
        with self._lock:
            self._responses.clear()

    def get(self, key: tuple[Any, ...]) -> StrategyOpportunityAdvisoryResponse | None:
        with self._lock:
            return self._responses.get(key)

    def set(
        self,
        key: tuple[Any, ...],
        *,
        payload: bytes,
        etag: str,
        row_count: int,
        latest_generated_at: str,
        serialize_ms: float,
    ) -> StrategyOpportunityAdvisoryResponse:
        response = StrategyOpportunityAdvisoryResponse(
            key=key,
            payload=payload,
            etag=etag,
            row_count=row_count,
            latest_generated_at=latest_generated_at,
            serialize_ms=serialize_ms,
            created_at=datetime.now(UTC),
        )
        with self._lock:
            self._responses[key] = response
        return response


class ExactKeyCache(Generic[T]):
    """Small process-local cache invalidated by exact source signatures."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[tuple[Any, ...], CachedValue[T]] = {}

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def get(self, key: tuple[Any, ...]) -> T | None:
        with self._lock:
            cached = self._values.get(key)
            return cached.value if cached is not None else None

    def set(self, key: tuple[Any, ...], value: T) -> None:
        with self._lock:
            self._values[key] = CachedValue(key=key, value=value, created_at=datetime.now(UTC))


def _source_sha(signature: tuple[Any, ...]) -> str:
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()
