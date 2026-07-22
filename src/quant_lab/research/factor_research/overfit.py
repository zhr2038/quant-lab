from __future__ import annotations

import itertools
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OverfitDiagnostics:
    status: str
    pbo: float | None
    dsr_probability: float | None
    selection_degradation: float | None
    selected_variant_id: str | None
    variant_count: int
    period_count: int
    cscv_split_count: int
    blockers: tuple[str, ...]


def compute_overfit_diagnostics(
    variant_returns: dict[str, list[float]],
    *,
    cscv_subsets: int = 8,
) -> OverfitDiagnostics:
    clean = {
        str(variant_id): [float(value) for value in values if math.isfinite(float(value))]
        for variant_id, values in variant_returns.items()
    }
    clean = {variant_id: values for variant_id, values in clean.items() if values}
    variant_count = len(clean)
    period_count = min((len(values) for values in clean.values()), default=0)
    blockers: list[str] = []
    if variant_count < 2:
        blockers.append("at_least_two_variants_required_for_pbo")
    if period_count < max(cscv_subsets * 2, 20):
        blockers.append("insufficient_periods_for_cscv")
    if blockers:
        return OverfitDiagnostics(
            status="INCONCLUSIVE_OVERFIT_DIAGNOSTICS",
            pbo=None,
            dsr_probability=None,
            selection_degradation=None,
            selected_variant_id=None,
            variant_count=variant_count,
            period_count=period_count,
            cscv_split_count=0,
            blockers=tuple(blockers),
        )
    aligned = {
        variant_id: values[:period_count] for variant_id, values in sorted(clean.items())
    }
    selected_variant = max(aligned, key=lambda item: (_sharpe(aligned[item]), item))
    pbo, degradation, split_count = _cscv(
        aligned,
        subsets=min(cscv_subsets, period_count // 2),
    )
    dsr = deflated_sharpe_probability(
        aligned[selected_variant],
        trial_count=variant_count,
    )
    if pbo <= 0.20 and dsr >= 0.95:
        status = "PASS"
    else:
        status = "FAIL"
        if pbo > 0.20:
            blockers.append("pbo_above_0_20")
        if dsr < 0.95:
            blockers.append("dsr_probability_below_0_95")
    return OverfitDiagnostics(
        status=status,
        pbo=pbo,
        dsr_probability=dsr,
        selection_degradation=degradation,
        selected_variant_id=selected_variant,
        variant_count=variant_count,
        period_count=period_count,
        cscv_split_count=split_count,
        blockers=tuple(blockers),
    )


def deflated_sharpe_probability(values: list[float], *, trial_count: int) -> float:
    sample = [float(value) for value in values if math.isfinite(float(value))]
    if len(sample) < 3 or trial_count < 1:
        return 0.0
    sharpe = _sharpe(sample)
    skewness = _skewness(sample)
    kurtosis = _kurtosis(sample)
    variance_term = (
        1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe
    ) / max(len(sample) - 1, 1)
    if variance_term <= 0:
        return 0.0
    gamma = 0.5772156649015329
    if trial_count == 1:
        expected_maximum = 0.0
    else:
        expected_maximum = (
            (1.0 - gamma) * _normal_ppf(1.0 - 1.0 / trial_count)
            + gamma * _normal_ppf(1.0 - 1.0 / (trial_count * math.e))
        ) / math.sqrt(len(sample))
    z_score = (sharpe - expected_maximum) / math.sqrt(variance_term)
    return _normal_cdf(z_score)


def _cscv(
    variant_returns: dict[str, list[float]],
    *,
    subsets: int,
) -> tuple[float, float, int]:
    period_count = min(len(values) for values in variant_returns.values())
    subset_count = max(4, min(subsets, period_count))
    if subset_count % 2:
        subset_count -= 1
    boundaries = [round(index * period_count / subset_count) for index in range(subset_count + 1)]
    blocks = [
        list(range(boundaries[index], boundaries[index + 1]))
        for index in range(subset_count)
    ]
    negative_logits = 0
    degradations: list[float] = []
    split_count = 0
    variants = sorted(variant_returns)
    for in_sample_blocks in itertools.combinations(range(subset_count), subset_count // 2):
        in_sample_set = set(in_sample_blocks)
        in_indexes = [index for block in in_sample_blocks for index in blocks[block]]
        out_indexes = [
            index
            for block in range(subset_count)
            if block not in in_sample_set
            for index in blocks[block]
        ]
        in_scores = {
            variant: _sharpe([variant_returns[variant][index] for index in in_indexes])
            for variant in variants
        }
        out_scores = {
            variant: _sharpe([variant_returns[variant][index] for index in out_indexes])
            for variant in variants
        }
        selected = max(variants, key=lambda variant: (in_scores[variant], variant))
        out_order = sorted(variants, key=lambda variant: (out_scores[variant], variant))
        rank = out_order.index(selected) + 1
        percentile = (rank - 0.5) / len(variants)
        logit = math.log(percentile / (1.0 - percentile))
        negative_logits += int(logit <= 0.0)
        degradations.append(in_scores[selected] - out_scores[selected])
        split_count += 1
    return (
        negative_logits / split_count if split_count else 1.0,
        _mean(degradations),
        split_count,
    )


def _sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    standard_deviation = _std(values)
    if standard_deviation <= 0:
        return 0.0
    return _mean(values) / standard_deviation


def _skewness(values: list[float]) -> float:
    standard_deviation = _std(values)
    if standard_deviation <= 0:
        return 0.0
    mean = _mean(values)
    return _mean([((value - mean) / standard_deviation) ** 3 for value in values])


def _kurtosis(values: list[float]) -> float:
    standard_deviation = _std(values)
    if standard_deviation <= 0:
        return 3.0
    mean = _mean(values)
    return _mean([((value - mean) / standard_deviation) ** 4 for value in values])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_ppf(probability: float) -> float:
    probability = min(max(probability, 1e-12), 1.0 - 1e-12)
    coefficients_a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    coefficients_b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    coefficients_c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    coefficients_d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    lower = 0.02425
    upper = 1.0 - lower
    if probability < lower:
        q = math.sqrt(-2.0 * math.log(probability))
        return _rational(q, coefficients_c, coefficients_d)
    if probability > upper:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -_rational(q, coefficients_c, coefficients_d)
    q = probability - 0.5
    r = q * q
    numerator = _horner(r, coefficients_a)
    denominator = _horner(r, (*coefficients_b, 1.0))
    return numerator / denominator * q


def _rational(
    value: float,
    numerator_coefficients: tuple[float, ...],
    denominator_coefficients: tuple[float, ...],
) -> float:
    numerator = 0.0
    for coefficient in numerator_coefficients:
        numerator = numerator * value + coefficient
    denominator = 0.0
    for coefficient in denominator_coefficients:
        denominator = denominator * value + coefficient
    return numerator / (denominator * value + 1.0)


def _horner(value: float, coefficients: tuple[float, ...]) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result
