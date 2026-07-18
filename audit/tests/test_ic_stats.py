"""Tests for overlap-aware IC statistics (audit spec section 11)."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit.auditlib.ic_stats import (  # noqa: E402
    ic_triad,
    newey_west_tstat,
    plain_tstat,
    select_non_overlapping,
)

QL_SRC = Path(__file__).resolve().parents[2] / "src"


class TestNonOverlapSelection:
    def test_rule_next_ge_last_plus_horizon(self):
        ts = np.arange(0, 100 * 3600, 3600, dtype=np.int64)
        keep = select_non_overlapping(ts, horizon_bars=24)
        kept_ts = ts[keep]
        assert kept_ts[0] == ts[0]
        assert (np.diff(kept_ts) >= 24 * 3600).all()
        assert len(keep) == math.ceil(100 / 24)

    def test_horizon_one_keeps_everything(self):
        ts = np.arange(0, 10 * 3600, 3600, dtype=np.int64)
        assert len(select_non_overlapping(ts, horizon_bars=1)) == len(ts)

    def test_empty(self):
        assert select_non_overlapping(np.array([], dtype=np.int64), 24).size == 0

    def test_irregular_timestamps(self):
        ts = np.array([0, 3600, 3 * 3600, 30 * 3600, 31 * 3600], dtype=np.int64)
        keep = select_non_overlapping(ts, horizon_bars=24)
        assert ts[keep].tolist() == [0, 30 * 3600]


class TestNeweyWest:
    def test_iid_matches_plain_tstat_asymptotically(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.05, 0.3, size=2000)
        plain = plain_tstat(x)
        hac = newey_west_tstat(x, lag=10)
        assert abs(plain - hac) / max(abs(plain), 1e-9) < 0.15

    def test_overlapping_ma_process_shrinks_tstat(self):
        rng = np.random.default_rng(11)
        h = 24
        eps = rng.normal(0, 1, size=5000 + h)
        x = np.convolve(eps, np.ones(h) / h, mode="valid") + 0.004
        plain = plain_tstat(x)
        hac = newey_west_tstat(x, lag=h)
        assert abs(hac) < abs(plain)
        assert abs(plain) / max(abs(hac), 1e-9) > 2.0

    def test_lag_capped(self):
        x = np.array([0.1, 0.2, 0.15])
        assert newey_west_tstat(x, lag=100) != float("inf")

    def test_constant_series_zero(self):
        assert newey_west_tstat(np.ones(50), lag=5) == 0.0


class TestICTriad:
    def test_triad_fields(self):
        rng = np.random.default_rng(3)
        n = 500
        ts = np.arange(0, n * 3600, 3600, dtype=np.int64)
        ic = rng.normal(0.02, 0.1, size=n)
        tri = ic_triad(ic, ts, horizon_bars=24)
        d = tri.as_dict()
        assert d["n_periods"] == n
        assert 0 < d["non_overlap_count"] <= math.ceil(n / 24)
        assert set(d) == {
            "n_periods",
            "ic_mean",
            "naive_tstat",
            "non_overlap_count",
            "non_overlap_ic_mean",
            "non_overlap_tstat",
            "hac_tstat",
            "hac_lag",
            "inflation_ratio",
        }

    def test_unsorted_input_is_sorted(self):
        ts = np.array([3 * 3600, 0, 2 * 3600, 3600], dtype=np.int64)
        ic = np.array([0.3, 0.1, 0.2, 0.15])
        assert ic_triad(ic, ts, horizon_bars=1).n_periods == 4

    def test_nan_ics_dropped(self):
        ts = np.arange(0, 5 * 3600, 3600, dtype=np.int64)
        ic = np.array([np.nan, 0.1, np.nan, 0.2, 0.15])
        assert ic_triad(ic, ts, horizon_bars=1).n_periods == 3


class TestProductionICModuleHasCorrectedStats:
    """The production ic.py must expose overlap-corrected statistics (section 11).

    These tests FAIL before the fix and PASS after the minimal change.
    """

    def test_module_exposes_nonoverlap_and_hac(self):
        sys.path.insert(0, str(QL_SRC))
        from quant_lab.research import ic as prod_ic  # noqa: PLC0415

        assert hasattr(prod_ic, "compute_rank_ic_nonoverlap"), "missing non-overlap estimator"
        assert hasattr(prod_ic, "compute_rank_ic_hac_tstat"), "missing HAC estimator"

    @staticmethod
    def _overlapping_panel(n_times: int = 24 * 200, n_symbols: int = 40, h: int = 24):
        """Momentum signal vs h-bar forward returns.

        Both signal and label are h-window moving sums of the same return
        process, so per-period rank ICs are autocorrelated over ~h lags,
        exactly the situation created by adjacent hourly decisions with
        h-bar labels.
        """
        rng = np.random.default_rng(42)
        total = n_times + 2 * h + 2
        rets = rng.normal(0.0002, 0.004, size=(total, n_symbols))
        rows = []
        base_ts = 1_700_000_000
        for t in range(h, h + n_times):
            signal = rets[t - h : t].sum(axis=0)
            fwd = rets[t + 1 : t + 1 + h].sum(axis=0)
            decision = base_ts + t * 3600
            for s in range(n_symbols):
                rows.append(
                    {
                        "symbol": f"S{s}",
                        "decision_ts": decision,
                        "alpha_score": float(signal[s]),
                        "forward_return": float(fwd[s]),
                    }
                )
        return rows

    def test_overlapping_labels_naive_tstat_inflated(self):
        if str(QL_SRC) not in sys.path:
            sys.path.insert(0, str(QL_SRC))
        import polars as pl  # noqa: PLC0415

        from quant_lab.research import ic as prod_ic  # noqa: PLC0415

        h = 24
        df = pl.DataFrame(self._overlapping_panel(h=h))
        naive = prod_ic.compute_rank_ic(df)
        non = prod_ic.compute_rank_ic_nonoverlap(df, horizon_bars=h, bar_seconds=3600)
        hac_t = prod_ic.compute_rank_ic_hac_tstat(df, horizon_bars=h, bar_seconds=3600)
        assert non.period_count < naive.period_count
        assert abs(hac_t) <= abs(naive.tstat) + 1e-9
        assert abs(naive.tstat) / max(abs(hac_t), 1e-9) > 1.5
