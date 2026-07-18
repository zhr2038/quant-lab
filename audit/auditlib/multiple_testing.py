"""Multiple-testing corrections over the complete executed test family."""

from __future__ import annotations

import numpy as np
from scipy import stats


def adjust_pvalues(pvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Holm-Bonferroni and Benjamini-Hochberg adjusted p-values."""
    raw = np.asarray(pvalues, dtype=float)
    holm = np.full(raw.shape, np.nan, dtype=float)
    bh = np.full(raw.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(raw))
    if valid_indices.size == 0:
        return holm, bh

    clipped = np.clip(raw[valid_indices], 0.0, 1.0)
    order = np.argsort(clipped, kind="mergesort")
    ordered = clipped[order]
    m = ordered.size

    holm_ordered = np.maximum.accumulate(ordered * (m - np.arange(m)))
    holm_ordered = np.minimum(holm_ordered, 1.0)

    bh_ordered = ordered * m / np.arange(1, m + 1)
    bh_ordered = np.minimum.accumulate(bh_ordered[::-1])[::-1]
    bh_ordered = np.minimum(bh_ordered, 1.0)

    inverse = np.empty(m, dtype=int)
    inverse[order] = np.arange(m)
    holm[valid_indices] = holm_ordered[inverse]
    bh[valid_indices] = bh_ordered[inverse]
    return holm, bh


def two_sided_pvalue(tstat: float, degrees_of_freedom: int) -> float:
    if not np.isfinite(tstat) or degrees_of_freedom < 1:
        return 1.0
    return float(2.0 * stats.t.sf(abs(float(tstat)), df=int(degrees_of_freedom)))
