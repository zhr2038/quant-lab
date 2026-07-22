from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


@contextmanager
def snapshot_payload_lock(
    queue_root: str | Path,
    snapshot_id: str,
    *,
    timeout_seconds: float = 60.0,
) -> Iterator[Path]:
    """Hold a process-local and OS advisory lock for one snapshot payload."""

    queue = Path(queue_root)
    lock_path = queue / "lease" / f"factor-factory-snapshot-{snapshot_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    with _LOCAL_LOCKS_GUARD:
        local_lock = _LOCAL_LOCKS.setdefault(key, threading.Lock())
    if timeout_seconds <= 0:
        acquired = local_lock.acquire(blocking=False)
    else:
        acquired = local_lock.acquire(timeout=timeout_seconds)
    if not acquired:
        raise TimeoutError(f"snapshot_payload_lock_busy:{snapshot_id}")
    try:
        with lock_path.open("a+b") as handle:
            if lock_path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            _acquire_os_lock(handle, snapshot_id=snapshot_id, timeout_seconds=timeout_seconds)
            try:
                yield lock_path
            finally:
                _release_os_lock(handle)
    finally:
        local_lock.release()


def _acquire_os_lock(handle, *, snapshot_id: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while True:
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except (BlockingIOError, OSError):
            if timeout_seconds <= 0 or time.monotonic() >= deadline:
                raise TimeoutError(f"snapshot_payload_lock_busy:{snapshot_id}") from None
            time.sleep(0.05)


def _release_os_lock(handle) -> None:
    with contextlib.suppress(OSError):
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
