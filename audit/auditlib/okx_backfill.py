"""OKX public history backfill: 1h candles + funding rate history.

Properties (per audit spec section 7):
- pagination backwards with `after` cursor
- token-bucket rate limit + exponential backoff with retry
- per-symbol checkpoint resume (idempotent, dedupe on ts)
- unclosed bars excluded (confirm == "1")
- UTC everywhere, per-symbol partitioned parquet: inst_id=/year=/month=
- raw API response metadata log per page (first page per symbol keeps full payload)
"""
from __future__ import annotations

import json
import time
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

from .paths import CANDLES_DIR, FUNDING_DIR, CHECKPOINTS, LOGS

OKX_BASE = "https://www.okx.com"
CANDLES_URL = OKX_BASE + "/api/v5/market/history-candles"
FUNDING_URL = OKX_BASE + "/api/v5/public/funding-rate-history"
RATE_LIMIT_RPS = 8.0  # OKX market data: 20 req / 2s; stay conservative
MAX_RETRIES = 6


class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


@dataclass
class PageLog:
    inst_id: str
    kind: str
    pages: int = 0
    rows: int = 0
    errors: list = field(default_factory=list)


def _get(url: str, params: dict, limiter: RateLimiter, logf, full_first: bool = False) -> list:
    """GET with retry/backoff. Returns data list (may be empty)."""
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        try:
            resp = requests.get(url, params=params, timeout=30)
            code = resp.status_code
            if code == 429 or code >= 500:
                raise RuntimeError(f"http {code}")
            body = resp.json()
            if str(body.get("code")) != "0":
                raise RuntimeError(f"okx code={body.get('code')} msg={body.get('msg')}")
            data = body.get("data") or []
            rec = {"params": params, "http": code, "rows": len(data), "attempt": attempt}
            if full_first:
                rec["payload"] = body
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            return data
        except Exception as e:  # noqa: BLE001
            logf.write(json.dumps({"params": params, "error": str(e)[:300], "attempt": attempt}) + "\n")
            logf.flush()
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return []


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _checkpoint_path(kind: str, inst_id: str) -> Path:
    d = CHECKPOINTS / "okx_backfill"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{kind}_{inst_id.replace('-', '_')}.json"


def _load_checkpoint(kind: str, inst_id: str) -> dict | None:
    p = _checkpoint_path(kind, inst_id)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_checkpoint(kind: str, inst_id: str, state: dict) -> None:
    state = dict(state)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _checkpoint_path(kind, inst_id).write_text(json.dumps(state, indent=2))


def backfill_candles(inst_id: str, start_ms: int, end_ms: int, limiter: RateLimiter) -> dict:
    """Backfill 1H candles for inst_id in [start_ms, end_ms). Backwards pagination."""
    ck = _load_checkpoint("candles", inst_id)
    if ck and ck.get("done") and ck.get("start_ms") == start_ms and ck.get("end_ms") == end_ms:
        return {"status": "skipped_done", "rows": ck.get("rows", 0)}

    raw_dir = LOGS / "okx_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logf = open(raw_dir / f"candles_{inst_id}.jsonl", "a")

    frames = []
    cursor_after = end_ms - 1  # first page: bars older than end
    rows_total = 0
    pages = 0
    first_page = True
    stop = False
    while not stop:
        params = {"instId": inst_id, "bar": "1H", "limit": "100", "after": str(cursor_after)}
        data = _get(CANDLES_URL, params, limiter, logf, full_first=first_page)
        first_page = False
        pages += 1
        if not data:
            break
        # rows newest-first: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        batch = []
        for r in data:
            ts = int(r[0])
            if ts < start_ms:
                stop = True
                continue
            if ts >= end_ms:
                continue
            batch.append(r)
        if not batch:
            break
        df = pl.DataFrame(
            {
                "ts": [int(r[0]) for r in batch],
                "open": [float(r[1]) for r in batch],
                "high": [float(r[2]) for r in batch],
                "low": [float(r[3]) for r in batch],
                "close": [float(r[4]) for r in batch],
                "volume": [float(r[5]) for r in batch],
                "vol_ccy": [float(r[6]) for r in batch],
                "quote_volume": [float(r[7]) for r in batch],
                "confirm": [str(r[8]) for r in batch],
            }
        )
        frames.append(df)
        rows_total += df.height
        oldest = min(int(r[0]) for r in batch)
        cursor_after = oldest - 1  # next page: strictly older
        if len(data) < 100:
            stop = True
        if oldest <= start_ms:
            stop = True
        # incremental checkpoint (resume support)
        _save_checkpoint("candles", inst_id, {
            "done": False, "start_ms": start_ms, "end_ms": end_ms,
            "oldest_fetched": oldest, "pages": pages, "rows": rows_total,
        })
    logf.close()

    if not frames:
        _save_checkpoint("candles", inst_id, {
            "done": True, "start_ms": start_ms, "end_ms": end_ms, "pages": pages, "rows": 0,
        })
        return {"status": "empty", "rows": 0}

    all_df = pl.concat(frames)
    # merge with previously written data for idempotent resume
    out_base = CANDLES_DIR / f"inst_id={inst_id}"
    if out_base.exists():
        try:
            prev = pl.scan_parquet(str(out_base / "**" / "*.parquet")).collect()
            all_df = pl.concat([all_df, prev])
        except Exception:
            pass
    # unclosed bars excluded; dedupe keep last write
    all_df = (
        all_df.filter(pl.col("confirm") == "1")
        .unique(subset=["ts"], keep="last")
        .sort("ts")
        .filter((pl.col("ts") >= start_ms) & (pl.col("ts") < end_ms))
        .with_columns(
            pl.from_epoch("ts", time_unit="ms").alias("ts_dt"),
            (pl.from_epoch("ts", time_unit="ms").dt.year()).alias("year"),
            (pl.from_epoch("ts", time_unit="ms").dt.month()).alias("month"),
        )
    )
    # write partitioned year/month
    for (y, m), part in all_df.group_by(["year", "month"], maintain_order=True):
        out_dir = out_base / f"year={y}" / f"month={m:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        part.drop(["year", "month"]).write_parquet(out_dir / "part-0.parquet", compression="zstd")

    _save_checkpoint("candles", inst_id, {
        "done": True, "start_ms": start_ms, "end_ms": end_ms,
        "pages": pages, "rows": all_df.height,
        "first_ts": int(all_df["ts"].min()), "last_ts": int(all_df["ts"].max()),
    })
    return {"status": "ok", "rows": all_df.height, "pages": pages}


def backfill_funding(inst_id_swap: str, start_ms: int, end_ms: int, limiter: RateLimiter) -> dict:
    """Backfill funding rate history for a SWAP instrument."""
    ck = _load_checkpoint("funding", inst_id_swap)
    if ck and ck.get("done") and ck.get("start_ms") == start_ms and ck.get("end_ms") == end_ms:
        return {"status": "skipped_done", "rows": ck.get("rows", 0)}

    raw_dir = LOGS / "okx_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logf = open(raw_dir / f"funding_{inst_id_swap}.jsonl", "a")

    frames = []
    cursor_after = end_ms - 1
    pages = 0
    rows_total = 0
    first_page = True
    stop = False
    while not stop:
        params = {"instId": inst_id_swap, "limit": "100", "after": str(cursor_after)}
        data = _get(FUNDING_URL, params, limiter, logf, full_first=first_page)
        first_page = False
        pages += 1
        if not data:
            break
        batch = []
        for r in data:
            ts = int(r["fundingTime"])
            if ts < start_ms:
                stop = True
                continue
            if ts >= end_ms:
                continue
            batch.append(r)
        if not batch:
            break
        df = pl.DataFrame(
            {
                "funding_time": [int(r["fundingTime"]) for r in batch],
                "funding_rate": [float(r["fundingRate"]) for r in batch],
                "realized_rate": [float(r.get("realizedRate") or r["fundingRate"]) for r in batch],
                "method": [str(r.get("method", "")) for r in batch],
            }
        )
        frames.append(df)
        rows_total += df.height
        oldest = min(int(r["fundingTime"]) for r in batch)
        cursor_after = oldest - 1
        if len(data) < 100:
            stop = True
        if oldest <= start_ms:
            stop = True
    logf.close()

    if not frames:
        _save_checkpoint("funding", inst_id_swap, {
            "done": True, "start_ms": start_ms, "end_ms": end_ms, "pages": pages, "rows": 0,
        })
        return {"status": "empty", "rows": 0}

    all_df = pl.concat(frames)
    out_base = FUNDING_DIR / f"inst_id={inst_id_swap}"
    if out_base.exists():
        try:
            prev = pl.scan_parquet(str(out_base / "**" / "*.parquet")).collect()
            all_df = pl.concat([all_df, prev])
        except Exception:
            pass
    all_df = (
        all_df.unique(subset=["funding_time"], keep="last")
        .sort("funding_time")
        .filter((pl.col("funding_time") >= start_ms) & (pl.col("funding_time") < end_ms))
        .with_columns(
            pl.from_epoch("funding_time", time_unit="ms").alias("ts_dt"),
            pl.from_epoch("funding_time", time_unit="ms").dt.year().alias("year"),
            pl.from_epoch("funding_time", time_unit="ms").dt.month().alias("month"),
        )
    )
    for (y, m), part in all_df.group_by(["year", "month"], maintain_order=True):
        out_dir = out_base / f"year={y}" / f"month={m:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        part.drop(["year", "month"]).write_parquet(out_dir / "part-0.parquet", compression="zstd")

    _save_checkpoint("funding", inst_id_swap, {
        "done": True, "start_ms": start_ms, "end_ms": end_ms,
        "pages": pages, "rows": all_df.height,
        "first_ts": int(all_df["funding_time"].min()), "last_ts": int(all_df["funding_time"].max()),
    })
    return {"status": "ok", "rows": all_df.height, "pages": pages}