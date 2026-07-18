"""Long-only portfolio primitives with explicit turnover accounting."""

from __future__ import annotations

from collections.abc import Mapping


def portfolio_turnover(previous: Mapping[str, float], current: Mapping[str, float]) -> float:
    symbols = set(previous) | set(current)
    return 0.5 * sum(abs(float(current.get(s, 0.0)) - float(previous.get(s, 0.0))) for s in symbols)


def _cap_and_redistribute(weights: dict[str, float], cap: float) -> dict[str, float]:
    if not weights:
        return {}
    if cap <= 0 or cap * len(weights) < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for the selected symbol count")
    out = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(out.values())
    if total <= 0:
        out = {key: 1.0 / len(out) for key in out}
    else:
        out = {key: value / total for key, value in out.items()}

    fixed: set[str] = set()
    for _ in range(len(out) + 1):
        over = {key for key, value in out.items() if key not in fixed and value > cap + 1e-14}
        if not over:
            break
        for key in over:
            out[key] = cap
        fixed |= over
        remaining = 1.0 - sum(out[key] for key in fixed)
        free = [key for key in out if key not in fixed]
        if not free:
            break
        free_total = sum(out[key] for key in free)
        if free_total <= 0:
            for key in free:
                out[key] = remaining / len(free)
        else:
            for key in free:
                out[key] = remaining * out[key] / free_total
    return out


def long_only_weights(
    scores: Mapping[str, float],
    *,
    top_n: int,
    method: str = "equal",
    max_weight: float = 1.0,
) -> dict[str, float]:
    """Select highest-ranked symbols and return non-negative fully invested weights."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    selected = sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))[:top_n]
    if not selected:
        return {}
    if method == "equal":
        raw = {symbol: 1.0 for symbol, _ in selected}
    elif method == "score":
        minimum = min(float(score) for _, score in selected)
        raw = {symbol: float(score) - minimum + 1e-12 for symbol, score in selected}
    else:
        raise ValueError(f"unsupported weighting method: {method}")
    return _cap_and_redistribute(raw, float(max_weight))
