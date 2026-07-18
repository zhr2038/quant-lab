from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.multiple_testing import adjust_pvalues, two_sided_pvalue  # noqa: E402


def test_holm_and_bh_preserve_original_order() -> None:
    raw = np.array([0.01, 0.04, 0.03, 0.20])
    holm, bh = adjust_pvalues(raw)
    np.testing.assert_allclose(holm, [0.04, 0.09, 0.09, 0.20])
    np.testing.assert_allclose(bh, [0.04, 0.0533333333, 0.0533333333, 0.20])


def test_adjusted_pvalues_are_bounded_and_not_below_raw() -> None:
    raw = np.array([0.0, 0.4, 1.0, np.nan])
    holm, bh = adjust_pvalues(raw)
    for adjusted in (holm[:3], bh[:3]):
        assert ((adjusted >= raw[:3]) & (adjusted <= 1.0)).all()
    assert np.isnan(holm[3]) and np.isnan(bh[3])


def test_two_sided_pvalue_monotonic_in_absolute_tstat() -> None:
    assert two_sided_pvalue(3.0, 50) < two_sided_pvalue(2.0, 50)
    assert two_sided_pvalue(-2.0, 50) == two_sided_pvalue(2.0, 50)

