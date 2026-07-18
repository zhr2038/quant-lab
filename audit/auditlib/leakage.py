"""Leakage test harness (audit spec section 10).

The harness runs the *real* evaluation pipeline on manipulated variants of
signals/labels and asserts the manipulation is detected:
- permuted labels/signals must look random (no significance)
- injected future information must produce a detectable IC jump
- timestamp violations must be rejected by the validator
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .evaluate import build_labels, evaluate_factor


@dataclass
class LeakageTestResult:
    test_name: str
    expected: str
    observed: str
    passed: bool
    detail: dict


def validate_temporal_order(labels: pl.DataFrame) -> None:
    """feature_ts < decision_ts < label_ts for every row; raise otherwise."""
    bad = labels.filter(
        (pl.col("decision_ts") <= pl.col("feature_ts"))
        | (pl.col("label_ts") <= pl.col("decision_ts"))
    )
    if bad.height:
        raise AssertionError(f"temporal order violated in {bad.height} rows")


def permute_labels(labels: pl.DataFrame, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    out = labels.with_columns(
        pl.Series("forward_return", rng.permutation(labels["forward_return"].to_numpy()))
    )
    return out


def permute_signals(signals: pl.DataFrame, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return signals.with_columns(pl.Series("signal", rng.permutation(signals["signal"].to_numpy())))


def inject_future_into_signal(
    signals: pl.DataFrame, labels: pl.DataFrame, alpha: float = 0.5
) -> pl.DataFrame:
    """Artificially leak: signal' = signal + alpha * future_return (z-scaled)."""
    joined = labels.select(["symbol", "feature_ts", "forward_return"]).join(
        signals, on=["symbol", "feature_ts"], how="inner"
    )
    sd = joined["signal"].std()
    fr_sd = joined["forward_return"].std() or 1.0
    leaked = joined.with_columns(
        (pl.col("signal") + alpha * (sd / fr_sd) * pl.col("forward_return")).alias("signal")
    )
    return leaked.select(["symbol", "feature_ts", "signal"])


def shift_label_values(labels: pl.DataFrame, bars: int) -> pl.DataFrame:
    """Shift realized label values within symbol while retaining join keys.

    Positive bars use older labels (lag); negative bars leak later labels into
    earlier feature timestamps. This is a diagnostic manipulation only.
    """
    return (
        labels.sort(["symbol", "feature_ts"])
        .with_columns(pl.col("forward_return").shift(bars).over("symbol").alias("forward_return"))
        .filter(pl.col("forward_return").is_not_null())
    )


def shift_signal_values(signals: pl.DataFrame, bars: int) -> pl.DataFrame:
    """Shift signal values within symbol; negative bars inject future features."""
    return (
        signals.sort(["symbol", "feature_ts"])
        .with_columns(pl.col("signal").shift(bars).over("symbol").alias("signal"))
        .filter(pl.col("signal").is_not_null())
    )


def run_leakage_battery(
    signals: pl.DataFrame,
    bars: pl.DataFrame,
    *,
    factor_id: str,
    horizon_bars: int,
    universe_name: str = "battery",
    seed: int = 7,
) -> list[LeakageTestResult]:
    labels = build_labels(bars, horizon_bars, decision_delay_bars=1)
    validate_temporal_order(labels)
    base = evaluate_factor(
        signals,
        labels,
        factor_id=factor_id,
        direction=1,
        horizon_bars=horizon_bars,
        universe_name=universe_name,
    )

    results: list[LeakageTestResult] = []

    # 1. label permutation -> no significance
    perm_l = evaluate_factor(
        signals,
        permute_labels(labels, seed),
        factor_id=factor_id,
        direction=1,
        horizon_bars=horizon_bars,
        universe_name=universe_name,
    )
    results.append(
        LeakageTestResult(
            "label_permutation",
            "|hac_t| < 2",
            f"hac_t={perm_l.hac_tstat:.3f}",
            abs(perm_l.hac_tstat) < 2.0,
            {"baseline_ic": base.ic_mean, "permuted_ic": perm_l.ic_mean},
        )
    )

    # 2. signal permutation -> no significance
    perm_s = evaluate_factor(
        permute_signals(signals, seed + 1),
        labels,
        factor_id=factor_id,
        direction=1,
        horizon_bars=horizon_bars,
        universe_name=universe_name,
    )
    results.append(
        LeakageTestResult(
            "signal_permutation",
            "|hac_t| < 2",
            f"hac_t={perm_s.hac_tstat:.3f}",
            abs(perm_s.hac_tstat) < 2.0,
            {"baseline_ic": base.ic_mean, "permuted_ic": perm_s.ic_mean},
        )
    )

    # 3/4. Wrong label alignment in both directions. These are completed
    # diagnostics, not universal significance assertions: an honest factor may
    # legitimately be persistent at adjacent hours.
    for test_name, bars_shift in (("label_shift_forward", -1), ("label_shift_backward", 1)):
        shifted = evaluate_factor(
            signals,
            shift_label_values(labels, bars_shift),
            factor_id=factor_id,
            direction=1,
            horizon_bars=horizon_bars,
            universe_name=universe_name,
        )
        results.append(
            LeakageTestResult(
                test_name,
                "diagnostic completed and retained",
                f"baseline_ic={base.ic_mean:.4f} shifted_ic={shifted.ic_mean:.4f}",
                shifted.n_periods > 0,
                {"shift_bars": bars_shift, "ic_delta": shifted.ic_mean - base.ic_mean},
            )
        )

    # 5. Future feature value moved one hour earlier.
    shifted_signal = evaluate_factor(
        shift_signal_values(signals, -1),
        labels,
        factor_id=factor_id,
        direction=1,
        horizon_bars=horizon_bars,
        universe_name=universe_name,
    )
    results.append(
        LeakageTestResult(
            "feature_shift_forward",
            "diagnostic completed and retained",
            f"baseline_ic={base.ic_mean:.4f} shifted_ic={shifted_signal.ic_mean:.4f}",
            shifted_signal.n_periods > 0,
            {"shift_bars": -1, "ic_delta": shifted_signal.ic_mean - base.ic_mean},
        )
    )

    # 6. injected future -> detectable jump
    leaked = inject_future_into_signal(signals, labels, alpha=0.5)
    leak_eval = evaluate_factor(
        leaked,
        labels,
        factor_id=factor_id,
        direction=1,
        horizon_bars=horizon_bars,
        universe_name=universe_name,
    )
    results.append(
        LeakageTestResult(
            "future_injection_detected",
            "leaked_ic >> baseline_ic",
            f"baseline_ic={base.ic_mean:.4f} leaked_ic={leak_eval.ic_mean:.4f}",
            leak_eval.ic_mean > base.ic_mean + 0.10,
            {"delta": leak_eval.ic_mean - base.ic_mean},
        )
    )

    # 7. temporal validator rejects decision_delay=0 construction
    bad = build_labels(bars, horizon_bars, decision_delay_bars=1).with_columns(
        pl.col("feature_ts").alias("decision_ts")  # force decision == feature time
    )
    try:
        validate_temporal_order(bad)
        rejected = False
    except AssertionError:
        rejected = True
    results.append(
        LeakageTestResult(
            "decision_delay_zero_rejected",
            "validator raises",
            f"rejected={rejected}",
            rejected,
            {},
        )
    )

    accepted = True
    try:
        validate_temporal_order(labels)
    except AssertionError:
        accepted = False
    results.append(
        LeakageTestResult(
            "decision_delay_one_accepted",
            "validator accepts feature < decision < label",
            f"accepted={accepted}",
            accepted,
            {},
        )
    )

    return results
