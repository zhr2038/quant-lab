from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

from quant_lab.backtest.cost_model import conservative_cost_for_symbol
from quant_lab.backtest.datasets import (
    coerce_dt,
    first_value,
    float_or_none,
    iso_utc,
    market_rows_by_symbol,
    normalize_strategy_symbol,
    rows,
)
from quant_lab.backtest.metrics import frame_with_schema, max_drawdown_bps, summarize_net_bps

V5_DECISION_REPLAY_TRADES_FIELDS = [
    "strategy_id",
    "symbol",
    "decision_ts",
    "entry_ts",
    "exit_ts",
    "entry_px",
    "exit_px",
    "entry_price_source",
    "exit_price_source",
    "decision_delay_bars",
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
        decision_ts = row["_ts"]
        entry_bar = _next_closed_bar(bars_by_symbol, symbol, decision_ts)
        if entry_bar is None:
            continue
        entry_ts = entry_bar["_ts"]
        entry_px = float_or_none(entry_bar.get("close")) or entry_bar.get("_close")
        if entry_px is None or entry_px <= 0:
            continue
        cost = conservative_cost_for_symbol(cost_bucket_daily, symbol=symbol)
        exit_result = _exit_from_market(
            bars_by_symbol=bars_by_symbol,
            symbol=symbol,
            entry_ts=entry_ts,
            entry_px=entry_px,
            horizon_hours=max_hold_hours,
            hard_stop_bps=hard_stop_bps,
        )
        if exit_result is None:
            continue
        exit_ts, exit_px, gross_bps, exit_reason, exit_price_source = exit_result
        net_bps = gross_bps - cost.cost_bps
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
                "decision_ts": iso_utc(decision_ts),
                "entry_ts": iso_utc(entry_ts),
                "exit_ts": iso_utc(exit_ts),
                "entry_px": entry_px,
                "exit_px": exit_px,
                "entry_price_source": "next_closed_bar_close",
                "exit_price_source": exit_price_source,
                "decision_delay_bars": 1,
                "horizon_hours": max_hold_hours,
                "gross_bps": gross_bps,
                "cost_bps": cost.cost_bps,
                "net_bps": net_bps,
                "exit_reason": exit_reason,
                "data_leakage_check": "pass_next_closed_bar_one_bar_delay",
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


def _next_closed_bar(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    symbol: str,
    ts: Any,
) -> dict[str, Any] | None:
    parsed = coerce_dt(ts)
    if parsed is None:
        return None
    for row in bars_by_symbol.get(normalize_strategy_symbol(symbol), []):
        if row["_ts"] > parsed:
            return row
    return None


def _exit_from_market(
    *,
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    symbol: str,
    entry_ts: Any,
    entry_px: float,
    horizon_hours: int,
    hard_stop_bps: float,
) -> tuple[Any, float, float, str, str] | None:
    parsed = coerce_dt(entry_ts)
    if parsed is None or entry_px <= 0:
        return None
    horizon_ts = parsed + timedelta(hours=int(horizon_hours))
    stop_px = entry_px * (1.0 + float(hard_stop_bps) / 10_000.0)
    horizon_bar: dict[str, Any] | None = None
    for row in bars_by_symbol.get(normalize_strategy_symbol(symbol), []):
        row_ts = row["_ts"]
        if row_ts <= parsed:
            continue
        low = float_or_none(row.get("low"))
        if low is not None and low <= stop_px:
            open_px = float_or_none(row.get("open"))
            exit_px = min(stop_px, open_px) if open_px is not None else stop_px
            gross_bps = (exit_px / entry_px - 1.0) * 10_000.0
            return row_ts, exit_px, gross_bps, "hard_stop_model", "ohlc_stop_path"
        if row_ts >= horizon_ts:
            horizon_bar = row
            break
    if horizon_bar is None:
        return None
    exit_px = float_or_none(horizon_bar.get("close")) or horizon_bar.get("_close")
    if exit_px is None or exit_px <= 0:
        return None
    gross_bps = (exit_px / entry_px - 1.0) * 10_000.0
    return horizon_bar["_ts"], exit_px, gross_bps, "max_hold", "horizon_closed_bar_close"


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
        "the next closed market bar after each decision timestamp. This report does not call "
        "V5 live logic.",
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
