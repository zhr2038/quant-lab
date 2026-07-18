from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.costs import apply_costs  # noqa: E402
from audit.auditlib.portfolio import long_only_weights, portfolio_turnover  # noqa: E402


def test_cost_deduction_uses_roundtrip_one_way_cost() -> None:
    result = apply_costs(gross_return=0.02, turnover=1.0, one_way_cost_bps=15.0)
    assert result.cost_return == pytest.approx(0.003)
    assert result.net_return == pytest.approx(0.017)
    assert result.edge_cost_ratio == pytest.approx(0.02 / 0.003)


def test_turnover_is_half_l1_weight_change() -> None:
    assert portfolio_turnover({"A": 1.0}, {"B": 1.0}) == pytest.approx(1.0)
    assert portfolio_turnover({"A": 0.5, "B": 0.5}, {"A": 0.4, "B": 0.6}) == pytest.approx(0.1)


def test_long_only_weights_respect_top_n_and_cap() -> None:
    scores = {"A": 4.0, "B": 3.0, "C": 2.0, "D": -1.0}
    weights = long_only_weights(scores, top_n=3, method="score", max_weight=0.5)
    assert set(weights) == {"A", "B", "C"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) >= 0.0
    assert max(weights.values()) <= 0.5 + 1e-12

