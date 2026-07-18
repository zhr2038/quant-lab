"""Overlap-aware IC statistics for horizon labels.

Three estimators (audit spec section 11):
1. naive: per-period cross-sectional IC, plain t-stat (replicates production ic.py).
2. non-overlap: keep a decision timestamp only when it is >= last kept timestamp
   + horizon; t-stat over the surviving ICs.
3. HAC (Newey-West): t-stat for the mean IC of the full series with
   heteroskedasticity- and autocorrelation-consistent variance, lag matched to
   the horizon overlap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ICTriad:
    """One factor x horizon x universe x period evaluation."""

    n_periods: int
    ic_mean: float
    naive_tstat: float
    non_overlap_count: int
    non_overlap_ic_mean: float
    non_overlap_tstat: float
    hac_tstat: float
    hac_lag: int
    inflation_ratio: float  # naive_tstat / hac_tstat (sign-preserving, abs ratio)

    def as_dict(self) -> dict:
        return {
            "n_periods": self.n_periods,
            "ic_mean": self.ic_mean,
            "naive_tstat": self.naive_tstat,
            "non_overlap_count": self.non_overlap_count,
            "non_overlap_ic_mean": self.non_overlap_ic_mean,
            "non_overlap_tstat": self.non_overlap_tstat,
            "hac_tstat": self.hac_tstat,
            "hac_lag": self.hac_lag,
            "inflation_ratio": self.inflation_ratio,
        }


def plain_tstat(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return 0.0
    std = values.std(ddof=1)
    if std == 0:
        return 0.0
    return float(values.mean() / std * math.sqrt(n))


def select_non_overlapping(timestamps_sorted: np.ndarray, horizon_bars: int, bar_seconds: int = 3600) -> np.ndarray:
    """Return indices of kept samples under rule t_next >= t_last + horizon.

    timestamps_sorted: int64 epoch seconds (sorted ascending).
    horizon_bars: label horizon in bars of bar_seconds each.
    """
    if timestamps_sorted.size == 0:
        return np.empty(0, dtype=int)
    gap = horizon_bars * bar_seconds
    keep = [0]
    last = timestamps_sorted[0]
    for i in range(1, timestamps_sorted.size):
        if timestamps_sorted[i] - last >= gap:
            keep.append(i)
            last = timestamps_sorted[i]
    return np.asarray(keep, dtype=int)


def newey_west_tstat(values: np.ndarray, lag: int) -> float:
    """t-stat of the mean with Newey-West HAC variance (Bartlett kernel)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return 0.0
    mean = x.mean()
    e = x - mean
    lag = max(0, min(int(lag), n - 2))
    gamma0 = float(np.dot(e, e) / n)
    var_mean = gamma0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        gamma = float(np.dot(e[l:], e[:-l]) / n)
        var_mean += 2.0 * w * gamma
    if var_mean <= 0:
        return 0.0
    return float(mean / math.sqrt(var_mean / n))


def ic_triad(
    ic_by_ts: np.ndarray,
    decision_ts_epoch: np.ndarray,
    horizon_bars: int,
    bar_seconds: int = 3600,
    hac_lag: int | None = None,
) -> ICTriad:
    """Full triad for one factor evaluation.

    ic_by_ts: per-decision-timestamp cross-sectional IC (sorted by ts).
    decision_ts_epoch: matching epoch seconds, sorted ascending.
    horizon_bars: label horizon in bars.
    hac_lag: Newey-West lag; defaults to horizon_bars (covers the overlap window).
    """
    order = np.argsort(decision_ts_epoch)
    ts = np.asarray(decision_ts_epoch, dtype=np.int64)[order]
    ic = np.asarray(ic_by_ts, dtype=float)[order]
    mask = np.isfinite(ic)
    ts, ic = ts[mask], ic[mask]

    n = ic.size
    ic_mean = float(ic.mean()) if n else 0.0
    naive = plain_tstat(ic)

    keep = select_non_overlapping(ts, horizon_bars, bar_seconds)
    non_ic = ic[keep]
    non_mean = float(non_ic.mean()) if non_ic.size else 0.0
    non_t = plain_tstat(non_ic)

    lag = horizon_bars if hac_lag is None else hac_lag
    hac_t = newey_west_tstat(ic, lag)

    if hac_t != 0 and np.isfinite(hac_t):
        infl = abs(naive) / abs(hac_t)
    else:
        infl = float("inf") if naive != 0 else 1.0

    return ICTriad(
        n_periods=n,
        ic_mean=ic_mean,
        naive_tstat=naive,
        non_overlap_count=int(non_ic.size),
        non_overlap_ic_mean=non_mean,
        non_overlap_tstat=non_t,
        hac_tstat=hac_t,
        hac_lag=int(lag),
        inflation_ratio=float(infl),
    )