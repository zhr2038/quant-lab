import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ICStats:
    mean: float
    tstat: float
    period_count: int
    status: str = "ok"


def compute_ic(df: pl.DataFrame, feature_column: str = "alpha_score") -> ICStats:
    return _compute_period_stats(df, feature_column, rank=False)


def compute_rank_ic(df: pl.DataFrame, feature_column: str = "alpha_score") -> ICStats:
    return _compute_period_stats(df, feature_column, rank=True)


def compute_ic_tstat(values: list[float]) -> float:
    return _tstat(values)


def compute_rank_ic_tstat(values: list[float]) -> float:
    return _tstat(values)


def compute_by_symbol_ic(df: pl.DataFrame, feature_column: str = "alpha_score") -> dict[str, float]:
    if df.is_empty() or feature_column not in df.columns:
        return {}
    results: dict[str, float] = {}
    for symbol_key, group in df.group_by("symbol", maintain_order=True):
        symbol = symbol_key[0] if isinstance(symbol_key, tuple) else symbol_key
        pairs = _valid_pairs(group.to_dicts(), feature_column)
        results[str(symbol)] = _pearson([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    return results


# --- overlap-corrected estimators (audit/alpha-validity) ---------------------
#
# Adjacent decision timestamps with horizon_bars > 1 produce forward labels
# that share future returns. The plain t-stat treats every decision timestamp
# as an independent observation and overstates significance by roughly
# sqrt(horizon). The two estimators below provide corrected significance while
# the plain t-stat above is kept unchanged for backwards compatibility.


def _ics_by_decision_ts(df: pl.DataFrame, feature_column: str, *, rank: bool) -> tuple[list, list[float]]:
    """Per-decision-timestamp cross-sectional ICs, sorted by timestamp."""
    pairs_by_ts: dict[Any, list[tuple[float, float]]] = {}
    if df.is_empty() or feature_column not in df.columns:
        return [], []
    for row in df.select(["decision_ts", feature_column, "forward_return"]).iter_rows(named=True):
        x_value = row.get(feature_column)
        y_value = row.get("forward_return")
        if x_value is None or y_value is None:
            continue
        x = float(x_value)
        y = float(y_value)
        if math.isfinite(x) and math.isfinite(y):
            pairs_by_ts.setdefault(row["decision_ts"], []).append((x, y))
    ts_sorted = sorted(pairs_by_ts)
    ics: list[float] = []
    kept_ts: list[Any] = []
    for ts in ts_sorted:
        pairs = pairs_by_ts[ts]
        if len(pairs) < 3:
            continue
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]
        if rank:
            x_values = _ranks(x_values)
            y_values = _ranks(y_values)
        corr = _pearson(x_values, y_values)
        if math.isfinite(corr):
            kept_ts.append(ts)
            ics.append(corr)
    return kept_ts, ics


def _ts_epoch_seconds(value: Any) -> float:
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    if isinstance(value, (int, float)):
        # heuristic: microseconds/nanoseconds epoch or seconds
        v = float(value)
        if v > 1e14:
            return v / 1e9
        if v > 1e11:
            return v / 1e3
        return v
    raise TypeError(f"unsupported decision_ts type: {type(value)!r}")


def select_non_overlapping_indices(
    epoch_seconds: np.ndarray, horizon_bars: int, bar_seconds: int
) -> np.ndarray:
    """Indices of samples kept under rule t_next >= t_last_kept + horizon."""
    if epoch_seconds.size == 0:
        return np.empty(0, dtype=int)
    gap = horizon_bars * bar_seconds
    keep = [0]
    last = epoch_seconds[0]
    for i in range(1, epoch_seconds.size):
        if epoch_seconds[i] - last >= gap:
            keep.append(i)
            last = epoch_seconds[i]
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


def compute_rank_ic_nonoverlap(
    df: pl.DataFrame,
    feature_column: str = "alpha_score",
    *,
    horizon_bars: int,
    bar_seconds: int = 3600,
) -> ICStats:
    """Rank IC over non-overlapping decision timestamps only."""
    kept_ts, ics = _ics_by_decision_ts(df, feature_column, rank=True)
    if not ics:
        return ICStats(mean=0.0, tstat=0.0, period_count=0, status="insufficient_samples")
    epochs = np.array([_ts_epoch_seconds(ts) for ts in kept_ts], dtype=float)
    idx = select_non_overlapping_indices(epochs, horizon_bars, bar_seconds)
    selected = [ics[i] for i in idx]
    if not selected:
        return ICStats(mean=0.0, tstat=0.0, period_count=0, status="insufficient_samples")
    return ICStats(
        mean=sum(selected) / len(selected),
        tstat=_tstat(selected),
        period_count=len(selected),
    )


def compute_rank_ic_hac_tstat(
    df: pl.DataFrame,
    feature_column: str = "alpha_score",
    *,
    horizon_bars: int,
    bar_seconds: int = 3600,
    hac_lag: int | None = None,
) -> float:
    """Newey-West t-stat of mean rank IC; lag defaults to the horizon overlap."""
    _kept_ts, ics = _ics_by_decision_ts(df, feature_column, rank=True)
    if not ics:
        return 0.0
    lag = horizon_bars if hac_lag is None else hac_lag
    return newey_west_tstat(np.asarray(ics, dtype=float), lag)


def _compute_period_stats(df: pl.DataFrame, feature_column: str, *, rank: bool) -> ICStats:
    if df.is_empty() or feature_column not in df.columns:
        return ICStats(mean=0.0, tstat=0.0, period_count=0, status="insufficient_samples")
    values: list[float] = []
    for _period, group in df.group_by("decision_ts", maintain_order=True):
        pairs = _valid_pairs(group.to_dicts(), feature_column)
        if len(pairs) < 3:
            continue
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]
        if rank:
            x_values = _ranks(x_values)
            y_values = _ranks(y_values)
        corr = _pearson(x_values, y_values)
        if math.isfinite(corr):
            values.append(corr)
    if not values:
        return ICStats(mean=0.0, tstat=0.0, period_count=0, status="insufficient_samples")
    return ICStats(mean=sum(values) / len(values), tstat=_tstat(values), period_count=len(values))


def _valid_pairs(rows: list[dict[str, Any]], feature_column: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        x_value = row.get(feature_column)
        y_value = row.get("forward_return")
        if x_value is None or y_value is None:
            continue
        x = float(x_value)
        y = float(y_value)
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    return pairs


def _tstat(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(len(values))


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True))
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _ranks(values: list[float]) -> list[float]:
    grouped: dict[float, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[value].append(index)
    ranks = [0.0] * len(values)
    current = 1
    for value in sorted(grouped):
        indexes = grouped[value]
        average_rank = (current + current + len(indexes) - 1) / 2.0
        for index in indexes:
            ranks[index] = average_rank
        current += len(indexes)
    return ranks