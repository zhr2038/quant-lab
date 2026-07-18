from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.decisions_v2 import (  # noqa: E402
    validate_dashboard_contract,
    validate_layered_decisions,
)
from audit.auditlib.forward_paper import (  # noqa: E402
    upsert_by_decision_timestamp,
    validate_forward_timestamps,
)
from audit.auditlib.funding_v2 import (  # noqa: E402
    funding_fade_time_20d,
    funding_frequency_summary,
    time_based_funding_observations,
)
from audit.auditlib.portfolio_backtest import (  # noqa: E402
    MarketMatrix,
    performance_metrics,
    simulate_long_only,
)
from audit.auditlib.runtime_snapshot import (  # noqa: E402
    assert_no_sensitive_values,
    environment_presence,
)
from audit.auditlib.statistics_v2 import (  # noqa: E402
    empirical_pvalues,
    hac_bandwidth_sensitivity,
    max_stat_adjusted_pvalue,
    moving_block_bootstrap,
)


def _market(hours: int = 96) -> MarketMatrix:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(hours=index) for index in range(hours)]
    returns = np.zeros((hours, 3))
    returns[1:, 0] = 0.001
    returns[1:, 1] = 0.0005
    returns[1:, 2] = -0.0002
    signal = np.tile(np.asarray([3.0, 2.0, 1.0]), (hours, 1))
    members = {timestamp.date(): np.asarray([0, 1, 2]) for timestamp in timestamps}
    return MarketMatrix(
        timestamps=timestamps,
        symbols=["BTC-USDT", "A-USDT", "B-USDT"],
        returns=returns,
        signals={"low_vol": signal},
        universe_members={"top3": members},
        btc_up=np.ones(hours, dtype=bool),
        btc_returns=returns[:, 0],
    )


def test_24h_portfolio_rebalances_and_deducts_costs() -> None:
    market = _market()
    simulation = simulate_long_only(
        market,
        factor_id="low_vol",
        universe="top3",
        top_n=3,
        weighting="equal",
        max_weight=0.5,
        rebalance_hours=24,
        staggered=False,
        btc_filter=False,
    )
    assert np.flatnonzero(simulation.traded_fraction).tolist() == [1]
    metrics, _ = performance_metrics(
        simulation,
        market,
        one_way_cost_bps=15.0,
        fee_bps=10.0,
        start_index=0,
        end_index=len(market.timestamps),
        rebalance_hours=24,
    )
    assert metrics["turnover"] == pytest.approx(0.5)
    assert metrics["traded_fraction"] == pytest.approx(1.0)
    assert metrics["total_cost_return"] == pytest.approx(0.0015)
    assert metrics["total_return"] < metrics["gross_total_return"]


def _funding_rows() -> pl.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for symbol, step in (("A", 4), ("B", 8)):
        for hour in range(0, 24 * 22, step):
            rows.append(
                {
                    "symbol": symbol,
                    "funding_ts": start + timedelta(hours=hour),
                    "funding_rate": 0.001 + hour / 1_000_000,
                }
            )
    return pl.DataFrame(rows)


def test_time_based_funding_window_handles_different_frequencies() -> None:
    rolled = time_based_funding_observations(_funding_rows())
    latest = rolled.group_by("symbol").tail(1).sort("symbol")
    counts = dict(zip(latest["symbol"], latest["observation_count_20d"], strict=True))
    assert counts["A"] == 120
    assert counts["B"] == 60
    summary = funding_frequency_summary(_funding_rows()).sort("symbol")
    assert summary["median_settlement_frequency_hours"].to_list() == [4.0, 8.0]


def test_funding_window_includes_event_then_asof_excludes_exact_timestamp() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    funding = pl.DataFrame(
        {
            "symbol": ["A", "A"],
            "funding_ts": [start, start + timedelta(hours=8)],
            "funding_rate": [0.01, 0.03],
        }
    )
    rolled = time_based_funding_observations(funding).sort("funding_ts")
    assert rolled["funding_mean_20d"][0] == pytest.approx(0.01)
    assert rolled["funding_mean_20d"][1] == pytest.approx(0.02)
    bars = pl.DataFrame(
        {"symbol": ["A", "A"], "ts": [start, start + timedelta(hours=8)]}
    )
    signal = funding_fade_time_20d(bars, funding).sort("feature_ts")
    assert signal["signal"][0] is None
    assert signal["signal"][1] == pytest.approx(-0.01)


def test_hac_sensitivity_reports_all_predeclared_lags() -> None:
    rows = hac_bandwidth_sensitivity(np.linspace(-0.1, 0.2, 100), horizon=24)
    assert [row["lag_rule"] for row in rows] == [
        "horizon_half",
        "horizon",
        "horizon_double",
        "automatic",
    ]
    assert [row["lag"] for row in rows[:3]] == [12, 24, 48]
    assert all({"estimate", "standard_error", "t_stat", "p_value"} <= row.keys() for row in rows)


def test_moving_block_bootstrap_is_seeded_and_reports_probability() -> None:
    values = np.asarray([0.01, -0.005, 0.02, 0.0] * 30)
    first = moving_block_bootstrap(
        values,
        block_sizes=[24, 72],
        statistic=lambda sample: float(sample.mean()),
        n_bootstrap=100,
        seed=7,
    )
    second = moving_block_bootstrap(
        values,
        block_sizes=[24, 72],
        statistic=lambda sample: float(sample.mean()),
        n_bootstrap=100,
        seed=7,
    )
    assert first == second
    assert all(0.0 <= row["probability_greater_than_zero"] <= 1.0 for row in first)


def test_empirical_pvalue_and_max_stat_are_finite_sample_corrected() -> None:
    result = empirical_pvalues(2.0, np.asarray([-1.0, 0.0, 1.0]), np.asarray([1.0, 2.5, 3.0]))
    assert result["empirical_p_value"] == pytest.approx(0.25)
    assert result["two_sided_empirical_p_value"] == pytest.approx(0.25)
    assert result["max_stat_adjusted_p_value"] == pytest.approx(0.75)
    assert max_stat_adjusted_pvalue(2.0, np.asarray([1.0, 2.5, 3.0])) == pytest.approx(
        0.75
    )


def _decisions() -> dict:
    three = {
        "signal_validity": "INCONCLUSIVE",
        "portfolio_validity": "FAIL",
        "deployment_readiness": "FAIL",
    }
    return {
        "current_v5": {**three, "replayability": "PARTIALLY_REPLAYABLE"},
        "rev_xs_20d": dict(three),
        "funding_fade": dict(three),
        "low_vol_20d": {
            "signal_validity": "PASS",
            "locked_portfolio_validity": "FAIL",
            "deployment_readiness": "INCONCLUSIVE",
        },
        "low_vol_btc_trend": {
            "hypothesis_type": "POST_HOC_HYPOTHESIS",
            "parameters_locked": True,
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "INCONCLUSIVE",
        },
    }


def test_layered_decision_schema_rejects_post_hoc_pass() -> None:
    payload = _decisions()
    validate_layered_decisions(payload)
    payload["low_vol_btc_trend"]["portfolio_validity"] = "PASS"
    with pytest.raises(ValueError, match="post-hoc"):
        validate_layered_decisions(payload)


def test_forward_data_must_be_strictly_after_cutoff_and_resume_is_idempotent() -> None:
    cutoff = datetime(2026, 7, 17, 23, tzinfo=UTC)
    validate_forward_timestamps(pl.Series([cutoff + timedelta(hours=1)]), cutoff)
    with pytest.raises(ValueError, match="at or before"):
        validate_forward_timestamps(pl.Series([cutoff]), cutoff)
    rows = pl.DataFrame(
        {"decision_timestamp": [cutoff + timedelta(hours=1)], "rank": [1]}
    )
    assert upsert_by_decision_timestamp(rows, rows).height == 1


def test_runtime_snapshot_hashes_values_and_never_serializes_plaintext() -> None:
    environment = {"OKX_API_KEY": "super-secret-value", "MODE": "paper"}
    snapshot = environment_presence(environment)
    serialized = json.dumps(snapshot)
    assert "super-secret-value" not in serialized
    assert "paper" not in serialized
    assert snapshot["OKX_API_KEY"]["present"] is True
    assert snapshot["OKX_API_KEY"]["sensitive_name"] is True
    assert_no_sensitive_values(serialized, environment)


def test_dashboard_contract_contains_layered_fields(tmp_path: Path) -> None:
    content = (
        "Signal Validity Portfolio Validity Deployment Readiness POST_HOC "
        "FORWARD ONLY PARTIALLY_REPLAYABLE LIVE ALPHA FROZEN "
        "CONFIRMATORY EXPLORATORY"
    )
    dashboard = tmp_path / "audit_dashboard_v2.html"
    dashboard.write_text(content, encoding="utf-8")
    validate_dashboard_contract(dashboard.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="FORWARD ONLY"):
        validate_dashboard_contract(content.replace("FORWARD ONLY", ""))
