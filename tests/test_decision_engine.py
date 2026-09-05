from datetime import UTC, datetime, timedelta
from math import exp

import pytest
from pydantic import ValidationError

from quant_lab.decision.contracts import CostObservation, HourBar, InputSnapshot
from quant_lab.decision.engine import (
    build_advice,
    describe,
    historical_labels,
    prepare_history,
)


def series(n=1_600, rate=0.001, start=datetime(2025, 1, 1, tzinfo=UTC)):
    result = []
    for i in range(n):
        price = 100 * exp(rate * i)
        result.append(
            HourBar(
                symbol="BTCUSDT",
                ts=start + timedelta(hours=i),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=10,
                ingest_ts=start + timedelta(hours=i + 1),
            )
        )
    return result


def inputs(bars, now, cost=20):
    return InputSnapshot(
        snapshot_id="input-" + "a" * 64,
        generated_at=now,
        producer_commit="abcdef1",
        bars=bars[-192:],
        signature="test-signature",
        costs=[
            CostObservation(
                symbol="BTCUSDT",
                roundtrip_bps=cost,
                source="test",
                quality="test",
                version="test-v1",
                as_of=now,
                trusted_for_paper=True,
            )
        ],
    )


def test_future_and_unclosed_bars_cannot_enter_input():
    bars = series(30)
    now = bars[-1].ts + timedelta(minutes=59)
    assert len(prepare_history(bars, now)) == 29
    with pytest.raises(ValidationError, match="unclosed"):
        inputs(bars, now)


def test_late_ingest_is_not_available_yet():
    bars = series(30)
    now = bars[-1].ts + timedelta(hours=1)
    bars[-1] = bars[-1].model_copy(update={"ingest_ts": now + timedelta(seconds=1)})
    assert len(prepare_history(bars, now)) == 29
    with pytest.raises(ValidationError, match="not available"):
        inputs(bars, now)


def test_repeated_bars_do_not_add_samples_and_conflicts_fail():
    bars = series(30)
    now = bars[-1].ingest_ts
    assert len(prepare_history(bars + bars, now)) == 30
    changed = bars[-1].model_copy(update={"volume": 11})
    with pytest.raises(ValueError, match="conflicting duplicate"):
        prepare_history(bars + [changed], now)


def test_one_complete_bar_of_decision_delay():
    bars = series(32, rate=0, start=datetime(1970, 1, 1, tzinfo=UTC))
    for index, price in [(26, 200), (27, 112), (31, 126)]:
        bars[index] = bars[index].model_copy(
            update={"open": price, "close": price, "high": price, "low": price}
        )
    labels, _ = historical_labels(bars, horizon=4, context="trend_up")
    assert len(labels) == 1
    assert labels[0][0] == datetime(1970, 1, 2, 2, tzinfo=UTC)
    assert labels[0][2] == pytest.approx(1_250)


def test_gap_does_not_turn_into_a_valid_delayed_path():
    bars = series(32, rate=0, start=datetime(1970, 1, 1, tzinfo=UTC))
    del bars[28]
    labels, _ = historical_labels(bars, horizon=4, context="trend_up")
    assert labels == []


def test_cost_is_subtracted_exactly_once_and_stress_is_separate():
    bars = series()
    result = describe(bars, horizon=4, context="trend_up", cost_bps=20)
    expected_gross = (exp(0.004) - 1) * 10_000
    assert result.gross_mean_bps == pytest.approx(expected_gross)
    assert result.net_mean_bps == pytest.approx(expected_gross - 20)
    assert result.double_cost_mean_bps == pytest.approx(expected_gross - 40)
    assert result.non_overlapping_samples < result.samples / 4
    assert result.chronological_tail_samples >= 12


def test_advice_is_research_only_and_expires_before_delayed_entry():
    bars = series()
    now = bars[-1].ingest_ts + timedelta(minutes=15)
    advice = build_advice(bars, inputs=inputs(bars, now), symbol="BTCUSDT", horizon=4, now=now)
    assert advice.action == "REVIEW_ENTRY"
    assert advice.live_order_effect == "none"
    assert advice.calibrated_profit_probability is None
    assert advice.expires_at <= advice.reference_entry_at
    assert advice.reference_exit_at - advice.reference_entry_at == timedelta(hours=4)
    later = build_advice(
        bars,
        inputs=inputs(bars, now),
        symbol="BTCUSDT",
        horizon=4,
        now=now + timedelta(hours=1),
    )
    assert later.action == "NO_VIEW"
    assert "CURRENT_MARKET_STALE" in later.reason_codes


def test_one_opportunity_has_two_horizons_and_refresh_does_not_multiply_it():
    bars = series()
    now = bars[-1].ingest_ts + timedelta(minutes=15)
    snapshot = inputs(bars, now)
    four = build_advice(bars, inputs=snapshot, symbol="BTCUSDT", horizon=4, now=now)
    day = build_advice(bars, inputs=snapshot, symbol="BTCUSDT", horizon=24, now=now)
    refresh = build_advice(
        bars, inputs=snapshot, symbol="BTCUSDT", horizon=4, now=now + timedelta(minutes=1)
    )
    assert four.opportunity_id == day.opportunity_id == refresh.opportunity_id
    assert four.advice_id != refresh.advice_id
    assert four.advice_id != day.advice_id


def test_insufficient_history_is_explicit_not_a_zero_forecast():
    bars = series(40)
    now = bars[-1].ingest_ts + timedelta(minutes=10)
    advice = build_advice(bars, inputs=inputs(bars, now), symbol="BTCUSDT", horizon=24, now=now)
    assert advice.action == "NO_VIEW"
    assert "HISTORICAL_SAMPLE_INSUFFICIENT" in advice.reason_codes
    assert advice.distribution.net_mean_bps is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_cost_fails(value):
    with pytest.raises(ValidationError):
        CostObservation(symbol="BTCUSDT", roundtrip_bps=value, source="x", quality="x", version="x")
