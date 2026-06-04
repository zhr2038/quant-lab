from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_lab.backtest.event_replay import build_v5_decision_replay
from quant_lab.backtest.label_backtest import build_label_backtest_summary


@dataclass(frozen=True)
class BacktestEngineResult:
    label_summary: pl.DataFrame
    replay_trades: pl.DataFrame
    replay_equity: pl.DataFrame
    replay_summary_md: str


class BacktestEngine:
    """Read-only orchestration for research backtests."""

    def run(self, frames: dict[str, pl.DataFrame]) -> BacktestEngineResult:
        label_summary = build_label_backtest_summary(frames)
        replay_trades, replay_equity, replay_summary_md = build_v5_decision_replay(
            candidate_snapshot=frames.get("v5_candidate_event", pl.DataFrame()),
            decision_audit=frames.get("v5_decision_audit", pl.DataFrame()),
            market_bars=frames.get("market_bar", pl.DataFrame()),
            cost_bucket_daily=frames.get("cost_bucket_daily", pl.DataFrame()),
        )
        return BacktestEngineResult(
            label_summary=label_summary,
            replay_trades=replay_trades,
            replay_equity=replay_equity,
            replay_summary_md=replay_summary_md,
        )
