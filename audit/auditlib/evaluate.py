"""Cross-sectional factor/label evaluation engine.

Conventions (mirrors production labels.py semantics):
- feature_ts: bar close time; every feature only uses data up to and
  including that bar.
- decision_ts = feature_ts + decision_delay bars (production default 1).
- entry price = close at decision_ts; exit = close at decision_ts + horizon.
- forward_return = exit/entry - 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from .ic_stats import ic_triad

BAR_SECONDS = 3600


@dataclass
class EvalResult:
    factor_id: str
    direction: int
    horizon_bars: int
    universe: str
    period: str  # e.g. "full", "research", "validate", "blind"
    n_periods: int
    n_symbol_obs: int
    ic_mean: float
    naive_tstat: float
    non_overlap_count: int
    non_overlap_ic_mean: float
    non_overlap_tstat: float
    hac_tstat: float
    hac_lag: int
    inflation_ratio: float
    rank_ic_by_ts: dict  # epoch -> ic (for downstream plotting)

    def flat_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "direction": self.direction,
            "horizon_bars": self.horizon_bars,
            "universe": self.universe,
            "period": self.period,
            "n_periods": self.n_periods,
            "n_symbol_obs": self.n_symbol_obs,
            "ic_mean": self.ic_mean,
            "naive_tstat": self.naive_tstat,
            "non_overlap_count": self.non_overlap_count,
            "non_overlap_ic_mean": self.non_overlap_ic_mean,
            "non_overlap_tstat": self.non_overlap_tstat,
            "hac_tstat": self.hac_tstat,
            "hac_lag": self.hac_lag,
            "inflation_ratio": self.inflation_ratio,
        }


def build_labels(
    bars: pl.DataFrame, horizon_bars: int, decision_delay_bars: int = 1
) -> pl.DataFrame:
    """Per-symbol forward returns.

    bars: [symbol, ts(datetime UTC), close] sorted per symbol.
    Returns [symbol, feature_ts, decision_ts, label_ts, entry_close, exit_close, forward_return].
    """
    future = decision_delay_bars + horizon_bars
    g = ["symbol"]
    out = (
        bars.sort(["symbol", "ts"])
        .with_columns(
            pl.col("ts").shift(-decision_delay_bars).over(g).alias("decision_ts"),
            pl.col("ts").shift(-future).over(g).alias("label_ts"),
            pl.col("close").shift(-decision_delay_bars).over(g).alias("entry_close"),
            pl.col("close").shift(-future).over(g).alias("exit_close"),
        )
        .rename({"ts": "feature_ts"})
        .with_columns((pl.col("exit_close") / pl.col("entry_close") - 1.0).alias("forward_return"))
        .select(
            [
                "symbol",
                "feature_ts",
                "decision_ts",
                "label_ts",
                "entry_close",
                "exit_close",
                "forward_return",
            ]
        )
        .filter(pl.col("forward_return").is_not_null())
    )
    return out


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float((xc * xc).sum() * (yc * yc).sum()))
    if denom == 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def evaluate_factor(
    signals: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    factor_id: str,
    direction: int,
    horizon_bars: int,
    universe_name: str,
    period_name: str = "full",
    min_cross: int = 5,
) -> EvalResult:
    """Compute per-decision-ts rank IC of direction*signal vs forward_return.

    signals: [symbol, feature_ts, signal]
    labels: from build_labels()
    """
    df = labels.join(
        signals.select(["symbol", "feature_ts", "signal"]),
        on=["symbol", "feature_ts"],
        how="inner",
    ).filter(pl.col("signal").is_not_null())

    panel = df.select(["decision_ts", "signal", "forward_return"])
    n_obs = panel.height
    # Rank and correlate in Polars. This avoids one Python group per hour for
    # every factor x horizon x universe x OOS slice while preserving ties.
    ic_frame = (
        panel.with_columns(
            (pl.col("signal") * direction)
            .rank("average")
            .over("decision_ts")
            .alias("_signal_rank"),
            pl.col("forward_return").rank("average").over("decision_ts").alias("_return_rank"),
        )
        .group_by("decision_ts")
        .agg(
            pl.len().alias("cross_count"),
            pl.corr("_signal_rank", "_return_rank").alias("ic"),
        )
        .filter((pl.col("cross_count") >= min_cross) & pl.col("ic").is_finite())
        .sort("decision_ts")
    )
    ics = ic_frame["ic"].to_list()
    tss = [int(value.timestamp()) for value in ic_frame["decision_ts"].to_list()]

    tri = ic_triad(np.asarray(ics), np.asarray(tss, dtype=np.int64), horizon_bars, BAR_SECONDS)
    return EvalResult(
        factor_id=factor_id,
        direction=direction,
        horizon_bars=horizon_bars,
        universe=universe_name,
        period=period_name,
        n_periods=tri.n_periods,
        n_symbol_obs=n_obs,
        ic_mean=tri.ic_mean,
        naive_tstat=tri.naive_tstat,
        non_overlap_count=tri.non_overlap_count,
        non_overlap_ic_mean=tri.non_overlap_ic_mean,
        non_overlap_tstat=tri.non_overlap_tstat,
        hac_tstat=tri.hac_tstat,
        hac_lag=tri.hac_lag,
        inflation_ratio=tri.inflation_ratio,
        rank_ic_by_ts=dict(zip(tss, ics, strict=True)),
    )
