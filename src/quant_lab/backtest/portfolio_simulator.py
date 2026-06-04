from __future__ import annotations

from typing import Any

import polars as pl

from quant_lab.backtest.datasets import rows
from quant_lab.backtest.metrics import frame_with_schema

PORTFOLIO_EQUITY_FIELDS = [
    "ts_utc",
    "trade_index",
    "equity_bps",
    "drawdown_bps",
    "net_bps",
    "live_order_effect",
]


def simulate_equal_weight_equity(trades: pl.DataFrame) -> pl.DataFrame:
    """Build a diagnostic equity curve from replay trades.

    This is intentionally simple: it is a read-only research curve, not a live
    position sizing or portfolio allocation engine.
    """

    out: list[dict[str, Any]] = []
    equity = 0.0
    peak = 0.0
    for index, row in enumerate(rows(trades), start=1):
        net_bps = float(row.get("net_bps") or 0.0)
        equity += net_bps
        peak = max(peak, equity)
        out.append(
            {
                "ts_utc": row.get("exit_ts"),
                "trade_index": index,
                "equity_bps": equity,
                "drawdown_bps": peak - equity,
                "net_bps": net_bps,
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return frame_with_schema(out, PORTFOLIO_EQUITY_FIELDS)
