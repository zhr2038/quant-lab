from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

from quant_lab.backtest.cost_model import conservative_cost_for_symbol
from quant_lab.backtest.datasets import (
    coerce_dt,
    entry_price_from_row,
    first_value,
    float_or_none,
    future_net_bps_from_market,
    iso_utc,
    market_rows_by_symbol,
    normalize_strategy_symbol,
    rows,
)
from quant_lab.backtest.metrics import frame_with_schema, max_drawdown_bps, summarize_net_bps

V5_DECISION_REPLAY_TRADES_FIELDS = [
    "strategy_id",
    "symbol",
    "entry_ts",
    "exit_ts",
    "entry_px",
    "exit_px",
    "horizon_hours",
    "gross_bps",
    "cost_bps",
    "net_bps",
    "exit_reason",
    "data_leakage_check",
    "live_order_effect",
]

V5_DECISION_REPLAY_EQUITY_FIELDS = [
    "ts_utc",
    "trade_index",
    "equity_bps",
    "drawdown_bps",
    "symbol",
    "strategy_id",
    "net_bps",
]


def build_v5_decision_replay(
    *,
    candidate_snapshot: pl.DataFrame | None,
    decision_audit: pl.DataFrame | None,
    market_bars: pl.DataFrame | None,
    cost_bucket_daily: pl.DataFrame | None,
    max_hold_hours: int = 24,
    hard_stop_bps: float = -180.0,
) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    source_rows = _candidate_rows(candidate_snapshot, decision_audit)
    if not source_rows:
        trades = frame_with_schema([], V5_DECISION_REPLAY_TRADES_FIELDS)
        equity = frame_with_schema([], V5_DECISION_REPLAY_EQUITY_FIELDS)
        return trades, equity, _summary_md(trades, equity)
    bars_by_symbol = market_rows_by_symbol(market_bars)
    replay_rows: list[dict[str, Any]] = []
    for row in sorted(source_rows, key=lambda item: item["_ts"]):
        symbol = normalize_strategy_symbol(row.get("symbol"))
        if symbol == "UNKNOWN":
            continue
        if not _would_open(row):
            continue
        entry_ts = row["_ts"]
        entry_px = entry_price_from_row(row)
        if entry_px is None:
            entry_px = _entry_px_from_market(bars_by_symbol, symbol, entry_ts)
        cost = conservative_cost_for_symbol(cost_bucket_daily, symbol=symbol)
        net_bps = future_net_bps_from_market(
            bars_by_symbol=bars_by_symbol,
            symbol=symbol,
            ts=entry_ts,
            entry_px=entry_px or 0.0,
            horizon_hours=max_hold_hours,
            cost_bps=cost.cost_bps,
        )
        if net_bps is None:
            continue
        exit_px = _exit_px_from_net(entry_px or 0.0, net_bps + cost.cost_bps)
        exit_reason = "max_hold"
        if net_bps <= hard_stop_bps:
            exit_reason = "hard_stop_model"
        replay_rows.append(
            {
                "strategy_id": str(
                    first_value(
                        row,
                        (
                            "strategy_id",
                            "strategy_candidate",
                            "source_strategy_candidate",
                            "entry_reason",
                        ),
                    )
                    or "V5_DECISION_REPLAY_BACKTEST"
                ),
                "symbol": symbol,
                "entry_ts": iso_utc(entry_ts),
                "exit_ts": iso_utc(entry_ts + timedelta(hours=max_hold_hours)),
                "entry_px": entry_px,
                "exit_px": exit_px,
                "horizon_hours": max_hold_hours,
                "gross_bps": net_bps + cost.cost_bps,
                "cost_bps": cost.cost_bps,
                "net_bps": net_bps,
                "exit_reason": exit_reason,
                "data_leakage_check": "pass_sorted_visible_rows_only",
                "live_order_effect": "read_only_no_live_order",
            }
        )
    trades = frame_with_schema(replay_rows, V5_DECISION_REPLAY_TRADES_FIELDS)
    equity = _equity_curve(replay_rows)
    return trades, equity, _summary_md(trades, equity)


def _candidate_rows(
    candidate_snapshot: pl.DataFrame | None,
    decision_audit: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in (candidate_snapshot, decision_audit):
        for row in rows(source):
            ts = coerce_dt(first_value(row, ("ts_utc", "decision_ts", "run_ts", "timestamp")))
            if ts is None:
                continue
            item = dict(row)
            item["_ts"] = ts
            out.append(item)
    return out


def _would_open(row: dict[str, Any]) -> bool:
    final_decision = str(first_value(row, ("final_decision", "decision", "intent")) or "").lower()
    if final_decision in {"open_long", "allow", "allowed", "buy", "would_open", "paper_ready"}:
        return True
    if "open_long" in final_decision:
        return True
    return False


def _entry_px_from_market(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    symbol: str,
    ts: Any,
) -> float | None:
    parsed = coerce_dt(ts)
    if parsed is None:
        return None
    for row in bars_by_symbol.get(normalize_strategy_symbol(symbol), []):
        if row["_ts"] >= parsed:
            return float_or_none(row.get("close")) or row.get("_close")
    return None


def _exit_px_from_net(entry_px: float, gross_bps: float) -> float | None:
    if entry_px <= 0:
        return None
    return entry_px * (1.0 + gross_bps / 10_000.0)


def _equity_curve(trades: list[dict[str, Any]]) -> pl.DataFrame:
    rows_out: list[dict[str, Any]] = []
    equity = 0.0
    peak = 0.0
    for index, row in enumerate(trades, start=1):
        net = float_or_none(row.get("net_bps")) or 0.0
        equity += net
        peak = max(peak, equity)
        rows_out.append(
            {
                "ts_utc": row.get("exit_ts"),
                "trade_index": index,
                "equity_bps": equity,
                "drawdown_bps": peak - equity,
                "symbol": row.get("symbol"),
                "strategy_id": row.get("strategy_id"),
                "net_bps": net,
            }
        )
    return frame_with_schema(rows_out, V5_DECISION_REPLAY_EQUITY_FIELDS)


def _summary_md(trades: pl.DataFrame, equity: pl.DataFrame) -> str:
    stats = summarize_net_bps([row.get("net_bps") for row in rows(trades)])
    lines = [
        "# V5 Decision Replay Backtest",
        "",
        "Read-only replay approximation. Rows are sorted by decision timestamp and use only",
        "market bars at or after each decision timestamp. This report does not call V5 live logic.",
        "",
        f"- trade_count: {trades.height}",
        f"- complete_sample_count: {stats['complete_sample_count']}",
        f"- avg_net_bps: {stats['avg_net_bps']}",
        f"- win_rate: {stats['win_rate']}",
        f"- max_drawdown_bps: {max_drawdown_bps([row.get('net_bps') for row in rows(trades)])}",
        "- live_order_effect: read_only_no_live_order",
    ]
    if equity.is_empty():
        lines.append("- status: no replay trades")
    return "\n".join(lines) + "\n"
