"""Stage: OKX 2-year backfill of 1H candles + funding rates.

Usage: python -m audit.auditlib.run_backfill [--only BTC-USDT] [--kind candles|funding|both]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit.auditlib.okx_backfill import (  # noqa: E402
    RateLimiter,
    backfill_candles,
    backfill_funding,
)
from audit.auditlib.paths import AUDIT_END_DATE, AUDIT_START_DATE, CHECKPOINTS, WORK  # noqa: E402

EXTRA_MAJORS = [
    "DOGE-USDT",
    "SHIB-USDT",
    "PEPE-USDT",
    "DOT-USDT",
    "POL-USDT",
    "JUP-USDT",
    "PYTH-USDT",
    "STRK-USDT",
    "ZK-USDT",
    "WIF-USDT",
    "FLOKI-USDT",
    "BONK-USDT",
    "GRT-USDT",
    "IMX-USDT",
    "STX-USDT",
    "MKR-USDT",
    "OP-USDT",
    "ARB-USDT",
    "TIA-USDT",
    "SEI-USDT",
    "INJ-USDT",
    "LDO-USDT",
    "CRV-USDT",
    "SAND-USDT",
    "APE-USDT",
    "GMT-USDT",
    "BLUR-USDT",
    "W-USDT",
    "RENDER-USDT",
    "FET-USDT",
    "VIRTUAL-USDT",
    "ZRO-USDT",
    "ETHFI-USDT",
    "TON-USDT",
    "HBAR-USDT",
    "ICP-USDT",
    "TRX-USDT",
    "BCH-USDT",
    "AVAX-USDT",
    "DYDX-USDT",
]


def load_symbols() -> list[str]:
    sel = json.loads((WORK / "backfill_symbols.json").read_text())["symbols"]
    out = list(dict.fromkeys(list(sel) + EXTRA_MAJORS))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="single spot symbol, e.g. BTC-USDT")
    ap.add_argument("--kind", default="both", choices=["candles", "funding", "both"])
    ap.add_argument("--start", default=AUDIT_START_DATE)
    ap.add_argument("--end", default=AUDIT_END_DATE)
    args = ap.parse_args()

    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)

    symbols = [args.only] if args.only else load_symbols()
    limiter = RateLimiter(8.0)
    summary = {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "symbols": {},
        "started_at": datetime.now(UTC).isoformat(),
    }
    status_path = CHECKPOINTS / "okx_backfill" / "_run_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    for i, sym in enumerate(symbols, 1):
        t0 = time.time()
        entry = {}
        try:
            if args.kind in ("candles", "both"):
                r = backfill_candles(sym, start_ms, end_ms, limiter)
                entry["candles"] = r
                print(f"[{i}/{len(symbols)}] {sym} candles: {r}", flush=True)
            if args.kind in ("funding", "both"):
                rf = backfill_funding(sym.replace("-USDT", "-USDT-SWAP"), start_ms, end_ms, limiter)
                entry["funding"] = rf
                print(f"[{i}/{len(symbols)}] {sym} funding: {rf}", flush=True)
            entry["ok"] = True
        except Exception as e:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = str(e)[:500]
            print(f"[{i}/{len(symbols)}] {sym} ERROR: {e}", flush=True)
        entry["elapsed_s"] = round(time.time() - t0, 2)
        summary["symbols"][sym] = entry
        status_path.write_text(json.dumps(summary, indent=2))

    summary["finished_at"] = datetime.now(UTC).isoformat()
    status_path.write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for v in summary["symbols"].values() if v.get("ok"))
    print(f"BACKFILL_RUN_DONE ok={n_ok}/{len(symbols)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
