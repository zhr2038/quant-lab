"""Strict-null and robustness classifications for Audit v2.1."""

from __future__ import annotations

import numpy as np

STRICT_NULL_CONTROLS = (
    "cross_section_rank_permutation",
    "label_permutation",
    "independent_random_noise",
    "within_symbol_signal_permutation",
    "within_symbol_circular_shift",
)

ROBUSTNESS_PERTURBATIONS = (
    "time_within_factor_permutation",
    "wrong_lag",
    "random_universe",
    "drop_major_symbols",
    "horizon_shift",
    "decision_delay_change",
    "reverse_factor",
)


def classify_control(name: str) -> str:
    if name in STRICT_NULL_CONTROLS:
        return "STRICT_NULL_CONTROL"
    if name in ROBUSTNESS_PERTURBATIONS:
        return "ROBUSTNESS_PERTURBATION"
    raise ValueError(f"unclassified permutation or perturbation: {name}")


def _symbol_positions(symbol_ids: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for symbol in np.unique(symbol_ids):
        rows, columns = np.where(symbol_ids == symbol)
        order = np.argsort(rows, kind="mergesort")
        output[int(symbol)] = (rows[order], columns[order])
    return output


def within_symbol_signal_permutation(
    scores: np.ndarray, symbol_ids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Independently permute each symbol through time, preserving membership."""
    output = np.asarray(scores, dtype=float).copy()
    for rows, columns in _symbol_positions(symbol_ids).values():
        values = output[rows, columns].copy()
        output[rows, columns] = values[rng.permutation(len(values))]
    return output


def within_symbol_circular_shift(
    scores: np.ndarray,
    symbol_ids: np.ndarray,
    rng: np.random.Generator,
    *,
    minimum_shift_observations: int = 25,
) -> np.ndarray:
    """Shift each symbol independently by more than the 24h label horizon."""
    output = np.asarray(scores, dtype=float).copy()
    for rows, columns in _symbol_positions(symbol_ids).values():
        values = output[rows, columns].copy()
        count = len(values)
        if count < 3:
            continue
        minimum = min(max(2, int(minimum_shift_observations)), count - 1)
        maximum = count - minimum
        if maximum < minimum:
            shift = max(1, count // 2)
        else:
            shift = int(rng.integers(minimum, maximum + 1))
        output[rows, columns] = np.roll(values, shift)
    return output
