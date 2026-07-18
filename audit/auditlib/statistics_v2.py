"""Predeclared statistical sensitivity tools for Alpha Audit v2.

The helpers in this module intentionally expose every requested bandwidth and
block size.  Callers cannot ask them to select the most favourable result.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
from scipy import stats


def automatic_hac_lag(n_observations: int) -> int:
    """Andrews/Newey-West style automatic lag for a mean estimate."""
    if n_observations < 2:
        return 0
    return max(1, int(math.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0))))


def hac_mean_estimate(values: np.ndarray, lag: int) -> dict[str, float | int]:
    """Mean, Newey-West standard error, t statistic and two-sided p value."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return {
            "estimate": float(x.mean()) if n else 0.0,
            "standard_error": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "lag": 0,
            "n_observations": n,
        }
    resolved_lag = max(0, min(int(lag), n - 2))
    estimate = float(x.mean())
    residual = x - estimate
    long_run_variance = float(np.dot(residual, residual) / n)
    for offset in range(1, resolved_lag + 1):
        weight = 1.0 - offset / (resolved_lag + 1.0)
        covariance = float(np.dot(residual[offset:], residual[:-offset]) / n)
        long_run_variance += 2.0 * weight * covariance
    standard_error = (
        math.sqrt(max(long_run_variance, 0.0) / n) if long_run_variance > 0 else 0.0
    )
    t_stat = estimate / standard_error if standard_error else 0.0
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(n - 1, 1)))
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "lag": resolved_lag,
        "n_observations": n,
    }


def hac_bandwidth_sensitivity(values: np.ndarray, horizon: int) -> list[dict]:
    """Return every predeclared HAC rule; never select a winner."""
    x = np.asarray(values, dtype=float)
    rules = [
        ("horizon_half", max(1, int(math.ceil(horizon / 2)))),
        ("horizon", max(1, int(horizon))),
        ("horizon_double", max(1, int(horizon * 2))),
        ("automatic", automatic_hac_lag(int(np.isfinite(x).sum()))),
    ]
    rows: list[dict] = []
    for rule, lag in rules:
        rows.append({"lag_rule": rule, **hac_mean_estimate(x, lag)})
    return rows


def _moving_block_sample(
    values: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    n = len(values)
    if n == 0:
        return values.copy()
    block = max(1, min(int(block_size), n))
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    return np.concatenate([values[start : start + block] for start in starts])[:n]


def compounded_return(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.prod(1.0 + values) - 1.0) if values.size else 0.0


def annualized_sharpe(values: np.ndarray, periods_per_year: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or float(values.std(ddof=1)) == 0.0:
        return 0.0
    return float(values.mean() / values.std(ddof=1) * math.sqrt(periods_per_year))


def moving_block_bootstrap(
    values: np.ndarray,
    *,
    block_sizes: Iterable[int],
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    seed: int = 20260718,
) -> list[dict]:
    """Moving-block bootstrap summary for every supplied block size."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rows: list[dict] = []
    for offset, block_size in enumerate(block_sizes):
        rng = np.random.default_rng(seed + offset)
        estimates = np.asarray(
            [
                statistic(_moving_block_sample(clean, int(block_size), rng))
                for _ in range(n_bootstrap)
            ],
            dtype=float,
        )
        rows.append(
            {
                "block_size": int(block_size),
                "bootstrap_count": int(n_bootstrap),
                "estimate": float(statistic(clean)),
                "ci_95_low": float(np.quantile(estimates, 0.025)),
                "ci_95_high": float(np.quantile(estimates, 0.975)),
                "median": float(np.median(estimates)),
                "probability_greater_than_zero": float(np.mean(estimates > 0.0)),
            }
        )
    return rows


def empirical_pvalues(
    observed: float, null_statistics: np.ndarray, null_max_abs: np.ndarray | None = None
) -> dict[str, float]:
    """Finite-sample corrected one/two-sided and max-stat empirical p values."""
    null = np.asarray(null_statistics, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {
            "empirical_p_value": 1.0,
            "two_sided_empirical_p_value": 1.0,
            "max_stat_adjusted_p_value": 1.0,
            "null_95_low": 0.0,
            "null_95_high": 0.0,
            "null_99_low": 0.0,
            "null_99_high": 0.0,
        }
    denominator = null.size + 1.0
    one_sided = (1.0 + float(np.sum(null >= observed))) / denominator
    two_sided = (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / denominator
    max_null = np.abs(null) if null_max_abs is None else np.asarray(null_max_abs, dtype=float)
    max_null = max_null[np.isfinite(max_null)]
    adjusted = (1.0 + float(np.sum(max_null >= abs(observed)))) / (max_null.size + 1.0)
    return {
        "empirical_p_value": one_sided,
        "two_sided_empirical_p_value": two_sided,
        "max_stat_adjusted_p_value": adjusted,
        "null_95_low": float(np.quantile(null, 0.025)),
        "null_95_high": float(np.quantile(null, 0.975)),
        "null_99_low": float(np.quantile(null, 0.005)),
        "null_99_high": float(np.quantile(null, 0.995)),
    }


def max_stat_adjusted_pvalue(
    observed_statistic: float, null_max_absolute_statistics: np.ndarray
) -> float:
    """Finite-sample corrected max-stat p value on a common statistic scale."""
    null = np.asarray(null_max_absolute_statistics, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed_statistic):
        return 1.0
    return float(
        (1.0 + np.sum(null >= abs(float(observed_statistic)))) / (null.size + 1.0)
    )
