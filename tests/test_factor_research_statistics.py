import random

import pytest

from quant_lab.research.factor_research.statistics import (
    adjust_multiple_testing,
    block_bootstrap_mean_ci,
    compute_overlap_aware_significance,
    newey_west_tstat,
    sign_flip_empirical_pvalue,
)


def test_overlapping_labels_cannot_use_naive_tstat_as_formal_evidence() -> None:
    randomizer = random.Random(17)
    values = []
    for _ in range(30):
        block_mean = 0.025 + randomizer.gauss(0.0, 0.04)
        values.extend(block_mean + randomizer.gauss(0.0, 0.002) for _ in range(24))

    result = compute_overlap_aware_significance(
        values,
        horizon=24,
        bootstrap_samples=300,
        permutation_samples=300,
        random_seed=9,
    )

    assert result.naive_tstat > result.confirmatory_hac_tstat
    assert result.hac_bandwidth_horizon == 24
    assert result.hac_bandwidth_horizon_half == 12
    assert result.hac_bandwidth_horizon_double == 48
    assert result.non_overlapping_count == 30


def test_hac_reports_bandwidth_sensitivity() -> None:
    values = [0.01 + (index % 12) * 0.001 for index in range(240)]

    short = newey_west_tstat(values, bandwidth=4)
    long = newey_west_tstat(values, bandwidth=24)

    assert short != pytest.approx(long)
    assert short > 0
    assert long > 0


def test_block_bootstrap_and_permutation_support_clear_positive_signal() -> None:
    values = [0.02 + ((index % 7) - 3) * 0.001 for index in range(210)]

    low, high = block_bootstrap_mean_ci(
        values,
        block_length=12,
        samples=400,
        random_seed=3,
    )
    pvalue = sign_flip_empirical_pvalue(values, samples=400, random_seed=3)

    assert 0 < low < high
    assert pvalue < 0.01


def test_holm_and_bh_adjust_all_trials_including_failures() -> None:
    adjusted = adjust_multiple_testing(
        {
            "strong": 0.001,
            "medium": 0.02,
            "weak": 0.20,
            "failed_trial": 1.0,
        }
    )

    assert adjusted["strong"].holm_adjusted_pvalue == pytest.approx(0.004)
    assert adjusted["medium"].holm_adjusted_pvalue == pytest.approx(0.06)
    assert adjusted["strong"].bh_fdr_qvalue == pytest.approx(0.004)
    assert adjusted["failed_trial"].test_count == 4
