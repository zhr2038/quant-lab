from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class OverlapAwareSignificance:
    period_count: int
    mean: float
    naive_tstat: float
    hac_tstat_horizon_half: float
    hac_tstat_horizon: float
    hac_tstat_horizon_double: float
    hac_tstat_auto: float
    hac_bandwidth_horizon_half: int
    hac_bandwidth_horizon: int
    hac_bandwidth_horizon_double: int
    hac_bandwidth_auto: int
    non_overlapping_mean: float
    non_overlapping_tstat: float
    non_overlapping_count: int
    non_overlapping_offset: int
    block_bootstrap_ci_low: float
    block_bootstrap_ci_high: float
    block_length: int
    permutation_empirical_pvalue: float
    raw_pvalue: float
    expected_direction: int

    @property
    def confirmatory_hac_tstat(self) -> float:
        return self.hac_tstat_horizon * self.expected_direction

    @property
    def non_overlapping_direction_consistent(self) -> bool:
        return self.non_overlapping_mean * self.expected_direction > 0

    @property
    def bootstrap_supports_direction(self) -> bool:
        if self.expected_direction > 0:
            return self.block_bootstrap_ci_low > 0
        return self.block_bootstrap_ci_high < 0


@dataclass(frozen=True)
class MultipleTestingAdjustment:
    test_id: str
    raw_pvalue: float
    holm_adjusted_pvalue: float
    bh_fdr_qvalue: float
    test_count: int


def compute_overlap_aware_significance(
    values: Iterable[float],
    *,
    horizon: int,
    expected_direction: int = 1,
    bootstrap_samples: int = 1000,
    permutation_samples: int = 1000,
    random_seed: int = 0,
) -> OverlapAwareSignificance:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if expected_direction not in {-1, 1}:
        raise ValueError("expected_direction must be -1 or 1")
    clean = [_finite_float(value) for value in values]
    sample = [value for value in clean if value is not None]
    half_bandwidth = max(1, horizon // 2)
    horizon_bandwidth = max(1, horizon)
    double_bandwidth = max(1, horizon * 2)
    auto_bandwidth = _automatic_bandwidth(len(sample))
    non_overlapping = sample[::horizon]
    bootstrap_low, bootstrap_high = block_bootstrap_mean_ci(
        sample,
        block_length=horizon,
        samples=bootstrap_samples,
        random_seed=random_seed,
    )
    directional = [value * expected_direction for value in sample]
    primary_tstat = newey_west_tstat(sample, bandwidth=horizon_bandwidth)
    return OverlapAwareSignificance(
        period_count=len(sample),
        mean=_mean(sample),
        naive_tstat=naive_tstat(sample),
        hac_tstat_horizon_half=newey_west_tstat(sample, bandwidth=half_bandwidth),
        hac_tstat_horizon=primary_tstat,
        hac_tstat_horizon_double=newey_west_tstat(sample, bandwidth=double_bandwidth),
        hac_tstat_auto=newey_west_tstat(sample, bandwidth=auto_bandwidth),
        hac_bandwidth_horizon_half=min(half_bandwidth, max(len(sample) - 1, 0)),
        hac_bandwidth_horizon=min(horizon_bandwidth, max(len(sample) - 1, 0)),
        hac_bandwidth_horizon_double=min(double_bandwidth, max(len(sample) - 1, 0)),
        hac_bandwidth_auto=min(auto_bandwidth, max(len(sample) - 1, 0)),
        non_overlapping_mean=_mean(non_overlapping),
        non_overlapping_tstat=naive_tstat(non_overlapping),
        non_overlapping_count=len(non_overlapping),
        non_overlapping_offset=0,
        block_bootstrap_ci_low=bootstrap_low,
        block_bootstrap_ci_high=bootstrap_high,
        block_length=min(horizon, len(sample)) if sample else 0,
        permutation_empirical_pvalue=sign_flip_empirical_pvalue(
            directional,
            samples=permutation_samples,
            random_seed=random_seed,
        ),
        raw_pvalue=_one_sided_normal_pvalue(primary_tstat * expected_direction),
        expected_direction=expected_direction,
    )


def naive_tstat(values: Iterable[float]) -> float:
    sample = _clean_values(values)
    if len(sample) < 2:
        return 0.0
    variance = sum((value - _mean(sample)) ** 2 for value in sample) / (len(sample) - 1)
    if variance <= 0:
        return 0.0
    return _mean(sample) / math.sqrt(variance / len(sample))


def newey_west_tstat(values: Iterable[float], *, bandwidth: int) -> float:
    sample = _clean_values(values)
    if len(sample) < 2:
        return 0.0
    lag_limit = min(max(int(bandwidth), 0), len(sample) - 1)
    mean = _mean(sample)
    centered = [value - mean for value in sample]
    n = len(centered)
    long_run_variance = sum(value * value for value in centered) / n
    for lag in range(1, lag_limit + 1):
        covariance = sum(
            centered[index] * centered[index - lag] for index in range(lag, n)
        ) / n
        weight = 1.0 - lag / (lag_limit + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return 0.0
    return mean / math.sqrt(long_run_variance / n)


def block_bootstrap_mean_ci(
    values: Iterable[float],
    *,
    block_length: int,
    samples: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> tuple[float, float]:
    sample = _clean_values(values)
    if not sample:
        return 0.0, 0.0
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    block = min(max(int(block_length), 1), len(sample))
    randomizer = random.Random(random_seed)
    means: list[float] = []
    for _ in range(samples):
        draw: list[float] = []
        while len(draw) < len(sample):
            start = randomizer.randrange(len(sample))
            draw.extend(sample[(start + offset) % len(sample)] for offset in range(block))
        means.append(_mean(draw[: len(sample)]))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return _quantile(means, tail), _quantile(means, 1.0 - tail)


def sign_flip_empirical_pvalue(
    values: Iterable[float],
    *,
    samples: int = 1000,
    random_seed: int = 0,
) -> float:
    sample = _clean_values(values)
    if not sample:
        return 1.0
    if samples < 100:
        raise ValueError("permutation samples must be at least 100")
    observed = _mean(sample)
    if observed <= 0:
        return 1.0
    randomizer = random.Random(random_seed)
    exceedances = 0
    for _ in range(samples):
        permuted = _mean(
            [value if randomizer.random() >= 0.5 else -value for value in sample]
        )
        if permuted >= observed:
            exceedances += 1
    return (exceedances + 1.0) / (samples + 1.0)


def adjust_multiple_testing(
    raw_pvalues: dict[str, float],
) -> dict[str, MultipleTestingAdjustment]:
    normalized = {
        str(test_id): min(max(float(pvalue), 0.0), 1.0)
        for test_id, pvalue in raw_pvalues.items()
    }
    if not normalized:
        return {}
    ordered = sorted(normalized, key=lambda test_id: (normalized[test_id], test_id))
    count = len(ordered)
    holm: dict[str, float] = {}
    running_holm = 0.0
    for rank, test_id in enumerate(ordered):
        adjusted = min((count - rank) * normalized[test_id], 1.0)
        running_holm = max(running_holm, adjusted)
        holm[test_id] = running_holm
    bh: dict[str, float] = {}
    running_bh = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        test_id = ordered[reverse_rank]
        rank = reverse_rank + 1
        adjusted = min(normalized[test_id] * count / rank, 1.0)
        running_bh = min(running_bh, adjusted)
        bh[test_id] = running_bh
    return {
        test_id: MultipleTestingAdjustment(
            test_id=test_id,
            raw_pvalue=normalized[test_id],
            holm_adjusted_pvalue=holm[test_id],
            bh_fdr_qvalue=bh[test_id],
            test_count=count,
        )
        for test_id in sorted(normalized)
    }


def _automatic_bandwidth(sample_count: int) -> int:
    if sample_count < 2:
        return 0
    return max(1, int(math.floor(4.0 * (sample_count / 100.0) ** (2.0 / 9.0))))


def _one_sided_normal_pvalue(tstat: float) -> float:
    return 0.5 * math.erfc(tstat / math.sqrt(2.0))


def _clean_values(values: Iterable[float]) -> list[float]:
    return [value for item in values if (value := _finite_float(item)) is not None]


def _finite_float(value: object) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = min(max(probability, 0.0), 1.0) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
