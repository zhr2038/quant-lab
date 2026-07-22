from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from quant_lab.research.ic import compute_rank_ic


@dataclass(frozen=True)
class FactorAttributionResult:
    raw_rank_ic: float
    symbol_fixed_effect_null_ic: float
    beta_neutral_rank_ic: float
    liquidity_neutral_rank_ic: float
    momentum_neutral_rank_ic: float
    joint_residual_rank_ic: float
    incremental_ic: float
    attribution_type: str
    max_symbol_contribution_share: float
    dominant_symbol: str | None
    row_count: int
    period_count: int
    controls_available: tuple[str, ...]
    warnings: tuple[str, ...]


def compute_factor_attribution(
    dataset: pl.DataFrame,
    *,
    factor_id: str,
    factor_column: str = "alpha_score",
    return_column: str = "forward_return",
) -> FactorAttributionResult:
    required = {"symbol", "decision_ts", factor_column, return_column}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"factor attribution missing columns: {','.join(missing)}")
    clean = dataset.filter(
        pl.col(factor_column).is_not_null() & pl.col(return_column).is_not_null()
    )
    if clean.is_empty():
        return FactorAttributionResult(
            raw_rank_ic=0.0,
            symbol_fixed_effect_null_ic=0.0,
            beta_neutral_rank_ic=0.0,
            liquidity_neutral_rank_ic=0.0,
            momentum_neutral_rank_ic=0.0,
            joint_residual_rank_ic=0.0,
            incremental_ic=0.0,
            attribution_type="NO_INCREMENTAL_EDGE",
            max_symbol_contribution_share=0.0,
            dominant_symbol=None,
            row_count=0,
            period_count=0,
            controls_available=(),
            warnings=("insufficient_samples",),
        )
    working = (
        clean.rename({return_column: "forward_return"})
        if return_column != "forward_return"
        else clean
    )
    raw = _rank_ic(working, factor_column)
    symbol_residual = _demean_by_symbol(working, factor_column)
    symbol_fixed = _rank_ic(symbol_residual, "_factor_residual")
    controls = {
        "market_beta": "market_beta",
        "liquidity": "liquidity",
        "momentum": "momentum",
        "long_run_volatility": "long_run_volatility",
        "size_proxy": "size_proxy",
        "regime_score": "regime_score",
    }
    available = tuple(column for column in controls if column in working.columns)
    beta = _neutral_rank_ic(working, factor_column, ["market_beta"])
    liquidity = _neutral_rank_ic(working, factor_column, ["liquidity"])
    momentum = _neutral_rank_ic(working, factor_column, ["momentum"])
    joint_input = _demean_by_symbol(working, factor_column)
    joint = _neutral_rank_ic(
        joint_input,
        "_factor_residual",
        list(available),
    )
    max_share, dominant_symbol = _symbol_contribution(working, factor_column)
    warnings = tuple(
        f"missing_control:{column}"
        for column in controls
        if column not in working.columns
    )
    return FactorAttributionResult(
        raw_rank_ic=raw,
        symbol_fixed_effect_null_ic=symbol_fixed,
        beta_neutral_rank_ic=beta,
        liquidity_neutral_rank_ic=liquidity,
        momentum_neutral_rank_ic=momentum,
        joint_residual_rank_ic=joint,
        incremental_ic=joint,
        attribution_type=_attribution_type(factor_id, raw=raw, joint=joint),
        max_symbol_contribution_share=max_share,
        dominant_symbol=dominant_symbol,
        row_count=working.height,
        period_count=working["decision_ts"].n_unique(),
        controls_available=available,
        warnings=warnings,
    )


def decompose_low_volatility(
    market_bars: pl.DataFrame,
    *,
    recent_window: int,
    structural_window: int,
) -> pl.DataFrame:
    required = {"symbol", "ts", "close"}
    missing = sorted(required.difference(market_bars.columns))
    if missing:
        raise ValueError(f"low-vol decomposition missing columns: {','.join(missing)}")
    if recent_window < 2 or structural_window < recent_window:
        raise ValueError("low-vol windows are invalid")
    rows: list[dict[str, Any]] = []
    for symbol_key, group in market_bars.sort(["symbol", "ts"]).group_by(
        "symbol", maintain_order=True
    ):
        symbol = symbol_key[0] if isinstance(symbol_key, tuple) else symbol_key
        source = group.select(["ts", "close"]).to_dicts()
        returns: list[float | None] = [None]
        for previous, current in zip(source, source[1:], strict=False):
            previous_close = _float(previous.get("close"))
            current_close = _float(current.get("close"))
            if not previous_close or not current_close or previous_close <= 0 or current_close <= 0:
                returns.append(None)
            else:
                returns.append(math.log(current_close / previous_close))
        for index, row in enumerate(source):
            structural_sample = _window(returns, index, structural_window)
            recent_sample = _window(returns, index, recent_window)
            prior_sample = _prior_window(returns, index, recent_window)
            structural_vol = _sample_std(structural_sample)
            recent_vol = _sample_std(recent_sample)
            prior_vol = _sample_std(prior_sample)
            rows.append(
                {
                    "symbol": str(symbol),
                    "ts": row["ts"],
                    "structural_low_vol": -structural_vol if structural_vol is not None else None,
                    "dynamic_low_vol": (
                        prior_vol - recent_vol
                        if prior_vol is not None and recent_vol is not None
                        else None
                    ),
                    "recent_volatility": recent_vol,
                    "prior_volatility": prior_vol,
                    "structural_volatility": structural_vol,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "ts": pl.Datetime(time_zone="UTC"),
                "structural_low_vol": pl.Float64,
                "dynamic_low_vol": pl.Float64,
                "recent_volatility": pl.Float64,
                "prior_volatility": pl.Float64,
                "structural_volatility": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort(["symbol", "ts"])


def _neutral_rank_ic(
    frame: pl.DataFrame,
    factor_column: str,
    controls: list[str],
) -> float:
    available = [column for column in controls if column in frame.columns]
    if not available:
        return _rank_ic(frame, factor_column)
    residual_rows: list[dict[str, Any]] = []
    for _period, group in frame.sort("decision_ts").group_by(
        "decision_ts", maintain_order=True
    ):
        valid = group.drop_nulls([factor_column, "forward_return", *available])
        period_controls = [
            column for column in available if valid[column].n_unique() > 1
        ]
        if valid.height < max(3, len(period_controls) + 2):
            continue
        values = [_float(value) for value in valid[factor_column].to_list()]
        matrix = [
            [1.0, *[_float(row.get(column)) or 0.0 for column in period_controls]]
            for row in valid.select(period_controls).to_dicts()
        ]
        if any(value is None for value in values):
            continue
        coefficients = _least_squares(matrix, [float(value) for value in values])
        if coefficients is None:
            continue
        for row, x_values, factor_value in zip(
            valid.to_dicts(), matrix, values, strict=True
        ):
            fitted = sum(
                coefficient * value
                for coefficient, value in zip(coefficients, x_values, strict=True)
            )
            residual_rows.append(
                {
                    "symbol": row["symbol"],
                    "decision_ts": row["decision_ts"],
                    "forward_return": row["forward_return"],
                    "_factor_residual": float(factor_value) - fitted,
                }
            )
    if not residual_rows:
        return 0.0
    return _rank_ic(pl.DataFrame(residual_rows), "_factor_residual")


def _demean_by_symbol(frame: pl.DataFrame, factor_column: str) -> pl.DataFrame:
    return frame.with_columns(
        (
            pl.col(factor_column)
            - pl.col(factor_column).mean().over("symbol")
        ).alias("_factor_residual")
    )


def _rank_ic(frame: pl.DataFrame, factor_column: str) -> float:
    return compute_rank_ic(frame, feature_column=factor_column).mean


def _symbol_contribution(frame: pl.DataFrame, factor_column: str) -> tuple[float, str | None]:
    contributions = (
        frame.with_columns(
            (pl.col(factor_column) * pl.col("forward_return")).alias("_contribution")
        )
        .group_by("symbol")
        .agg(pl.col("_contribution").sum().alias("_contribution"))
        .with_columns(pl.col("_contribution").abs().alias("_absolute"))
        .sort("_absolute", descending=True)
    )
    if contributions.is_empty():
        return 0.0, None
    denominator = float(contributions["_absolute"].sum() or 0.0)
    first = contributions.row(0, named=True)
    return (
        float(first["_absolute"] or 0.0) / denominator if denominator > 0 else 0.0,
        str(first["symbol"]),
    )


def _attribution_type(factor_id: str, *, raw: float, joint: float) -> str:
    normalized = factor_id.lower()
    if raw <= 0.01 or joint <= 0.01:
        return "NO_INCREMENTAL_EDGE"
    if joint < raw * 0.5:
        return "BETA_OR_LIQUIDITY_PROXY"
    if "structural_low_vol" in normalized:
        return "STRUCTURAL_CROSS_SECTIONAL"
    if "dynamic_low_vol" in normalized or "timing" in normalized:
        return "DYNAMIC_TIMING"
    if "regime" in normalized:
        return "REGIME_CONDITIONAL"
    return "STRUCTURAL_CROSS_SECTIONAL"


def _least_squares(matrix: list[list[float]], values: list[float]) -> list[float] | None:
    if not matrix or len(matrix) != len(values):
        return None
    width = len(matrix[0])
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    target = [0.0 for _ in range(width)]
    for row, value in zip(matrix, values, strict=True):
        for left in range(width):
            target[left] += row[left] * value
            for right in range(width):
                gram[left][right] += row[left] * row[right]
    for index in range(width):
        gram[index][index] += 1e-10
    return _solve_linear_system(gram, target)


def _solve_linear_system(matrix: list[list[float]], target: list[float]) -> list[float] | None:
    size = len(target)
    augmented = [row[:] + [target[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[index][-1] for index in range(size)]


def _window(values: list[float | None], index: int, length: int) -> list[float]:
    start = max(0, index - length + 1)
    return [float(value) for value in values[start : index + 1] if value is not None]


def _prior_window(values: list[float | None], index: int, length: int) -> list[float]:
    end = max(0, index - length + 1)
    start = max(0, end - length)
    return [float(value) for value in values[start:end] if value is not None]


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _float(value: object) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None
