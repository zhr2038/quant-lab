from __future__ import annotations

import sys

# ruff: noqa: E402
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jsonschema
import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.contribution_v21 import partition_symbol_contributions  # noqa: E402
from audit.auditlib.forward_v21 import (
    DECISION_SCHEMA,
    ENTRY_FEE_BPS,
    ENTRY_SLIPPAGE_BPS,
    EVENT_SCHEMA,
    EXIT_FEE_BPS,
    EXIT_SLIPPAGE_BPS,
    TRADE_SCHEMA,
    append_only_events,
    arithmetic_sum_return,
    build_cohort_equity,
    compounded_return,
    cost_adjusted_return,
    empty_frame,
    forward_available_days,
    forward_review_status,
    make_event,
    merge_trade_states,
    parameter_lock_digest,
    performance_counts,
    resolve_trade_state,
    select_next_bar_close,
    stable_decision_id,
    stable_trade_id,
    validate_new_cutoff,
)
from audit.auditlib.funding_v2 import (
    funding_fade_time_20d,
    time_based_funding_observations,
)
from audit.auditlib.permutation_v21 import (
    classify_control,
    within_symbol_circular_shift,
    within_symbol_signal_permutation,
)
from audit.auditlib.portfolio_backtest import MarketMatrix, Simulation
from audit.auditlib.report_v21 import validate_v21_report_contract
from audit.scripts.stage_v21_forward import build_new_decisions, run


def _base_trade(entry_ts: datetime, *, weight: float = 1.0) -> dict:
    decision_id = stable_decision_id("strategy", entry_ts - timedelta(hours=1), "snap", "lock")
    return {
        "trade_id": stable_trade_id(decision_id, "A-USDT", entry_ts),
        "decision_id": decision_id,
        "symbol": "A-USDT",
        "target_weight": weight,
        "entry_ts": entry_ts,
        "entry_price": 100.0,
        "scheduled_exit_ts": entry_ts + timedelta(hours=120),
        "actual_exit_ts": None,
        "exit_price": None,
        "exit_delay_bars": None,
        "mark_ts": entry_ts,
        "mark_price": 100.0,
        "entry_fee_bps": ENTRY_FEE_BPS,
        "entry_slippage_bps": ENTRY_SLIPPAGE_BPS,
        "exit_fee_bps": EXIT_FEE_BPS,
        "exit_slippage_bps": EXIT_SLIPPAGE_BPS,
        "gross_return": None,
        "net_return": None,
        "unrealized_gross_return": 0.0,
        "unrealized_net_return": -0.0015,
        "weighted_gross_contribution": None,
        "weighted_net_contribution": None,
        "status": "OPEN",
        "invalidation_reason": "",
        "parameter_lock_hash": "lock",
        "code_commit": "a" * 40,
        "snapshot_id": "snap",
        "runner_version": "quant_lab_low_vol_forward.v2.1",
    }


def _bars(rows: list[tuple[str, datetime, float]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["symbol", "ts", "close"], orient="row").with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC"))
    )


def test_fixed_120h_exit_uses_first_available_bar_and_never_reprices() -> None:
    entry = datetime(2026, 1, 1, 1, tzinfo=UTC)
    scheduled = entry + timedelta(hours=120)
    bars = _bars(
        [
            ("A-USDT", entry - timedelta(hours=1), 100.0),
            ("A-USDT", scheduled - timedelta(hours=2), 105.0),
            ("A-USDT", scheduled + timedelta(hours=1), 110.0),
            ("A-USDT", scheduled + timedelta(hours=19), 150.0),
        ]
    )
    open_state = resolve_trade_state(
        _base_trade(entry), bars, scheduled - timedelta(hours=1)
    )
    assert open_state["status"] == "OPEN"
    assert open_state["actual_exit_ts"] is None
    assert open_state["gross_return"] is None
    assert open_state["unrealized_gross_return"] == pytest.approx(0.05)

    closed = resolve_trade_state(_base_trade(entry), bars, scheduled + timedelta(hours=20))
    assert closed["status"] == "CLOSED"
    assert closed["actual_exit_ts"] == scheduled + timedelta(hours=2)
    assert closed["exit_delay_bars"] == 2
    assert closed["exit_price"] == 110.0
    assert resolve_trade_state(closed, bars, scheduled + timedelta(hours=40)) == closed


def test_entry_uses_exact_next_bar_close_and_rejects_a_gap() -> None:
    feature_cutoff = datetime(2026, 1, 1, 1, tzinfo=UTC)
    bars = _bars(
        [
            ("A-USDT", feature_cutoff - timedelta(hours=1), 100.0),
            ("A-USDT", feature_cutoff, 101.0),
            ("A-USDT", feature_cutoff + timedelta(hours=1), 102.0),
        ]
    )
    assert select_next_bar_close(bars, "A-USDT", feature_cutoff) == (
        feature_cutoff + timedelta(hours=1),
        101.0,
    )
    gap = bars.filter(pl.col("ts") != feature_cutoff)
    assert select_next_bar_close(gap, "A-USDT", feature_cutoff) is None


def test_round_trip_cost_is_recorded_on_both_sides_and_compounded() -> None:
    actual = cost_adjusted_return(0.10, 15.0, 15.0)
    expected = (1.0 - 0.0015) * 1.10 * (1.0 - 0.0015) - 1.0
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(0.10 - 0.003)


def test_period_returns_compound_and_arithmetic_sum_is_diagnostic_only() -> None:
    values = [0.10, -0.10]
    assert arithmetic_sum_return(values) == pytest.approx(0.0)
    assert compounded_return(values) == pytest.approx(-0.01)


def _decision(
    decision_id: str, entry_ts: datetime, *, status: str, cash_weight: float
) -> dict:
    return {
        "decision_id": decision_id,
        "strategy_id": "strategy",
        "decision_ts": entry_ts - timedelta(hours=1),
        "feature_cutoff_ts": entry_ts - timedelta(hours=1),
        "entry_ts": entry_ts,
        "available_data_cutoff": entry_ts,
        "btc_trend_state": "DOWN" if status == "CASH" else "UP",
        "universe": "[]",
        "factor_values": "[]",
        "ranked_symbols": "[]",
        "selected_symbols": "[]" if status == "CASH" else '["A-USDT"]',
        "rejected_symbols": "[]",
        "target_weights": "{}" if status == "CASH" else '{"A-USDT":1.0}',
        "cash_weight": cash_weight,
        "invested_weight": 1.0 - cash_weight,
        "decision_status": status,
        "decision_delay_bars": 1,
        "entry_price_rule": "next_bar_close",
        "parameter_lock_hash": "lock",
        "code_commit": "a" * 40,
        "snapshot_id": "snap",
        "hypothesis_type": "POST_HOC_HYPOTHESIS",
        "parameters_locked": True,
        "runner_version": "quant_lab_low_vol_forward.v2.1",
    }


def test_cash_decisions_are_not_counted_as_trades() -> None:
    entry = datetime(2026, 1, 1, 1, tzinfo=UTC)
    trade = _base_trade(entry)
    trade.update(
        {
            "status": "CLOSED",
            "actual_exit_ts": entry + timedelta(hours=120),
            "exit_price": 101.0,
            "exit_delay_bars": 0,
            "gross_return": 0.01,
            "net_return": 0.007,
            "unrealized_gross_return": None,
            "unrealized_net_return": None,
            "weighted_gross_contribution": 0.01,
            "weighted_net_contribution": 0.007,
        }
    )
    decisions = pl.DataFrame(
        [
            _decision("cash", entry, status="CASH", cash_weight=1.0),
            _decision(trade["decision_id"], entry, status="INVESTED", cash_weight=0.0),
        ],
        schema=DECISION_SCHEMA,
    )
    counts = performance_counts(
        decisions, pl.DataFrame([trade], schema=TRADE_SCHEMA)
    )
    assert counts == {
        "decision_count": 2,
        "cash_decision_count": 1,
        "entry_count": 1,
        "open_trade_count": 0,
        "closed_trade_count": 1,
        "completed_independent_period_count": 1,
    }


def test_multi_asset_weights_preserve_unfilled_cash() -> None:
    entry = datetime(2026, 1, 1, 1, tzinfo=UTC)
    first = _base_trade(entry, weight=0.50)
    second = {**_base_trade(entry, weight=0.30), "symbol": "B-USDT"}
    second["trade_id"] = stable_trade_id(
        second["decision_id"], second["symbol"], entry
    )
    for trade, gross in ((first, 0.10), (second, -0.05)):
        trade.update(
            {
                "status": "CLOSED",
                "actual_exit_ts": entry + timedelta(hours=120),
                "exit_price": trade["entry_price"] * (1.0 + gross),
                "exit_delay_bars": 0,
                "gross_return": gross,
                "net_return": cost_adjusted_return(gross, 15.0, 15.0),
                "unrealized_gross_return": None,
                "unrealized_net_return": None,
            }
        )
    from audit.auditlib.forward_v21 import portfolio_period_returns

    result = portfolio_period_returns([first, second], 0.20, realized=True)
    assert first["target_weight"] + second["target_weight"] + 0.20 == pytest.approx(1.0)
    assert result["gross_return"] == pytest.approx(0.50 * 0.10 + 0.30 * -0.05)
    assert result["entry_cost_return"] == pytest.approx(0.80 * 0.0015)
    assert result["exit_cost_return"] == pytest.approx(0.80 * 0.0015)


def test_benchmark_equity_is_nonzero_and_reproducible() -> None:
    entry = datetime(2026, 1, 1, 1, tzinfo=UTC)
    decision_id = stable_decision_id("benchmark", entry, "snap", "lock")
    decision = _decision(decision_id, entry, status="INVESTED", cash_weight=0.0)
    trade = _base_trade(entry)
    trade["decision_id"] = decision_id
    trade["trade_id"] = stable_trade_id(decision_id, "A-USDT", entry)
    bars = _bars(
        [
            ("A-USDT", entry - timedelta(hours=1), 100.0),
            ("A-USDT", entry, 110.0),
        ]
    )
    decisions = pl.DataFrame([decision], schema=DECISION_SCHEMA)
    trades = pl.DataFrame([trade], schema=TRADE_SCHEMA)
    timeline = [entry, entry + timedelta(hours=1)]
    first = build_cohort_equity(decisions, trades, bars, timeline)
    second = build_cohort_equity(decisions, trades, bars, timeline)
    assert first == second
    assert first[-1]["equity"] > 1.0


def test_partially_closed_cohort_never_reprices_the_closed_leg() -> None:
    entry = datetime(2026, 1, 1, 1, tzinfo=UTC)
    decision_id = stable_decision_id("strategy", entry, "snap", "lock")
    decision = _decision(
        decision_id, entry, status="INVESTED", cash_weight=0.0
    )
    first = _base_trade(entry, weight=0.5)
    first["decision_id"] = decision_id
    first["trade_id"] = stable_trade_id(decision_id, "A-USDT", entry)
    first.update(
        {
            "status": "CLOSED",
            "actual_exit_ts": entry + timedelta(hours=120),
            "exit_price": 110.0,
            "exit_delay_bars": 0,
            "gross_return": 0.10,
            "net_return": cost_adjusted_return(0.10, 15.0, 15.0),
            "unrealized_gross_return": None,
            "unrealized_net_return": None,
        }
    )
    second = {**_base_trade(entry, weight=0.5), "symbol": "B-USDT"}
    second["decision_id"] = decision_id
    second["trade_id"] = stable_trade_id(decision_id, "B-USDT", entry)
    second["mark_ts"] = entry + timedelta(hours=121)
    second["mark_price"] = 100.0
    bars = _bars(
        [
            ("A-USDT", entry - timedelta(hours=1), 100.0),
            ("B-USDT", entry - timedelta(hours=1), 100.0),
            ("A-USDT", entry + timedelta(hours=119), 110.0),
            ("A-USDT", entry + timedelta(hours=120), 1000.0),
            ("B-USDT", entry + timedelta(hours=120), 100.0),
        ]
    )
    rows = build_cohort_equity(
        pl.DataFrame([decision], schema=DECISION_SCHEMA),
        pl.DataFrame([first, second], schema=TRADE_SCHEMA),
        bars,
        [entry + timedelta(hours=121)],
    )
    assert rows[0]["equity"] < 1.1
    assert rows[0]["open_trade_count"] == 1


def test_events_and_closed_trades_are_idempotent_and_immutable() -> None:
    entry = datetime(2026, 1, 1, 1, tzinfo=UTC)
    event = make_event(
        event_type="ENTRY_RECORDED",
        event_ts=entry,
        decision_id="decision",
        trade_id="trade",
        payload={"price": 100.0},
        parameter_lock_hash="lock",
        code_commit="a" * 40,
    )
    events = append_only_events(
        empty_frame(EVENT_SCHEMA), pl.DataFrame([event], schema=EVENT_SCHEMA)
    )
    assert append_only_events(events, events).height == 1
    corrupted = dict(event)
    corrupted["payload_json"] = '{"price":101.0}'
    with pytest.raises(ValueError, match="immutable event"):
        append_only_events(events, pl.DataFrame([corrupted], schema=EVENT_SCHEMA))

    bars = _bars(
        [
            ("A-USDT", entry - timedelta(hours=1), 100.0),
            ("A-USDT", entry + timedelta(hours=119), 110.0),
        ]
    )
    closed = resolve_trade_state(
        _base_trade(entry), bars, entry + timedelta(hours=120)
    )
    changed = {**closed, "exit_price": 999.0}
    with pytest.raises(ValueError, match="closed trade"):
        merge_trade_states(
            pl.DataFrame([closed], schema=TRADE_SCHEMA),
            pl.DataFrame([changed], schema=TRADE_SCHEMA),
        )


def test_cutoff_and_day_metric_use_available_market_data() -> None:
    fix = datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="predates"):
        validate_new_cutoff(fix - timedelta(seconds=1), fix)
    validate_new_cutoff(fix, fix)
    assert forward_available_days(fix, fix + timedelta(hours=12)) == pytest.approx(0.5)


def test_forward_status_never_auto_promotes_to_live() -> None:
    ready = forward_review_status(
        available_days=31,
        completed_periods=6,
        entry_count=12,
        data_coverage=0.99,
        runner_error_count=0,
        unhandled_delay_count=0,
        strategy_net_return=0.1,
        excess_vs_btc=0.01,
        excess_vs_universe=0.01,
        drawdown_within_lock=True,
        concentration_within_lock=True,
    )
    assert ready == "PAPER_REVIEW_READY"
    assert "LIVE" not in ready
    insufficient = forward_review_status(
        available_days=29,
        completed_periods=6,
        entry_count=12,
        data_coverage=0.99,
        runner_error_count=0,
        unhandled_delay_count=0,
        strategy_net_return=0.1,
        excess_vs_btc=0.01,
        excess_vs_universe=0.01,
        drawdown_within_lock=True,
        concentration_within_lock=True,
    )
    assert insufficient == "INCONCLUSIVE_FORWARD_SAMPLE_INSUFFICIENT"


def test_funding_event_is_available_after_but_not_at_publication() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    funding = pl.DataFrame(
        {
            "symbol": ["A", "A"],
            "funding_ts": [start, start + timedelta(hours=8)],
            "funding_rate": [0.01, 0.03],
        }
    )
    rolled = time_based_funding_observations(funding).sort("funding_ts")
    assert rolled["observation_count_20d"].to_list() == [1, 2]
    assert rolled["funding_mean_20d"].to_list() == pytest.approx([0.01, 0.02])
    grid = pl.DataFrame(
        {
            "symbol": ["A"] * 4,
            "ts": [
                start,
                start + timedelta(seconds=1),
                start + timedelta(hours=8),
                start + timedelta(hours=8, seconds=1),
            ],
        }
    )
    signal = funding_fade_time_20d(grid, funding).sort("feature_ts")
    assert signal["signal"].to_list() == [None, -0.01, -0.01, -0.02]


def test_null_and_robustness_classes_are_disjoint() -> None:
    assert classify_control("within_symbol_signal_permutation") == "STRICT_NULL_CONTROL"
    assert classify_control("within_symbol_circular_shift") == "STRICT_NULL_CONTROL"
    assert classify_control("time_within_factor_permutation") == "ROBUSTNESS_PERTURBATION"
    with pytest.raises(ValueError, match="unclassified"):
        classify_control("friendly_but_unknown")


def test_within_symbol_nulls_preserve_membership_and_symbol_values() -> None:
    scores = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    symbol_ids = np.asarray([[1, 2], [1, 2], [1, 2], [1, 2]])
    permuted = within_symbol_signal_permutation(
        scores, symbol_ids, np.random.default_rng(7)
    )
    shifted = within_symbol_circular_shift(
        scores,
        symbol_ids,
        np.random.default_rng(7),
        minimum_shift_observations=2,
    )
    assert sorted(permuted[:, 0]) == sorted(scores[:, 0])
    assert sorted(permuted[:, 1]) == sorted(scores[:, 1])
    assert sorted(shifted[:, 0]) == sorted(scores[:, 0])
    assert sorted(shifted[:, 1]) == sorted(scores[:, 1])
    assert not np.array_equal(shifted, scores)


def test_symbol_concentration_is_computed_per_partition() -> None:
    timestamps = [datetime(2026, 1, 1, hour, tzinfo=UTC) for hour in range(4)]
    market = MarketMatrix(
        timestamps=timestamps,
        symbols=["A", "B"],
        returns=np.zeros((4, 2)),
        signals={},
        universe_members={},
        btc_up=np.zeros(4, dtype=bool),
        btc_returns=np.zeros(4),
    )
    contributions = np.asarray([[0.1, 0.0], [0.1, 0.0], [0.0, 0.2], [0.0, 0.2]])
    simulation = Simulation(
        gross_returns=contributions.sum(axis=1),
        turnover=np.zeros(4),
        traded_fraction=np.zeros(4),
        contribution_by_symbol=contributions.sum(axis=0),
        contribution_matrix=contributions,
        traded_by_symbol=np.zeros((4, 2)),
    )
    result = partition_symbol_contributions(
        simulation,
        market,
        {"research": (0, 2), "validation": (2, 4)},
        structure_id="fixture",
        factor_id="low_vol",
        one_way_cost_bps=15.0,
    )
    research = result.filter(pl.col("partition") == "research")
    validation = result.filter(pl.col("partition") == "validation")
    assert research["top_symbol"].unique().to_list() == ["A"]
    assert validation["top_symbol"].unique().to_list() == ["B"]


def test_btc_trend_off_creates_cash_decision_without_a_trade() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    symbols = ["BTC-USDT", *[f"S{index:02d}-USDT" for index in range(19)]]
    rows: list[dict] = []
    for hour in range(1504):
        timestamp = start + timedelta(hours=hour)
        for symbol_index, symbol in enumerate(symbols):
            close = (
                300.0 - hour * 0.01
                if symbol == "BTC-USDT"
                else 10.0 + symbol_index + hour * 0.0001
            )
            rows.append(
                {
                    "symbol": symbol,
                    "ts": timestamp,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 10_000.0,
                    "quote_volume": 2_000_000.0,
                }
            )
    combined = pl.DataFrame(rows)
    cutoff = start + timedelta(hours=1500, minutes=30)
    forward = combined.filter(pl.col("ts") > start + timedelta(hours=1499))
    lock = {
        "sha256": "b" * 64,
        "code_commit": "a" * 40,
        "data_snapshot_id": "snapshot",
    }
    decisions, trades, _events, _expected, _available = build_new_decisions(
        lock=lock,
        cutoff=cutoff,
        available_cutoff=start + timedelta(hours=1504),
        combined_bars=combined,
        forward_bars=forward,
        existing=empty_frame(DECISION_SCHEMA),
    )
    assert decisions.height == 1
    assert decisions["feature_cutoff_ts"][0] == start + timedelta(hours=1501)
    assert decisions["decision_ts"][0] == start + timedelta(hours=1501)
    assert decisions["entry_ts"][0] == start + timedelta(hours=1502)
    assert decisions["btc_trend_state"][0] == "DOWN"
    assert decisions["decision_status"][0] == "CASH"
    assert decisions["cash_weight"][0] == pytest.approx(1.0)
    assert trades.is_empty()


def _receipt_fixture() -> dict:
    evidence = {
        "storage": "inline",
        "schema_version": "fixture.v1",
        "value": {},
    }
    return {
        "schema_version": "v5_decision_receipt.v1",
        "receipt_id": "a" * 64,
        "receipt_status": "SUCCEEDED",
        "last_completed_stage": "EXECUTION_RECORDED",
        "created_at": "2026-07-18T12:00:00Z",
        "finalized_at": "2026-07-18T12:00:01Z",
        "decision_ts": "2026-07-18T12:00:00Z",
        "market_data_cutoff": "2026-07-18T11:00:00Z",
        "code_commit": "b" * 40,
        "config_hash": "c" * 64,
        "container_image_digest": f"sha256:{'d' * 64}",
        "strategy_version": "v5",
        **{
            key: dict(evidence)
            for key in (
                "universe",
                "raw_features",
                "factor_values",
                "factor_weights",
                "dynamic_ic_state",
                "regime_state",
                "hmm_state",
                "rss_state",
                "funding_state",
                "current_positions",
                "available_cash",
                "optimizer_inputs",
                "target_weights",
                "alpha_gate",
                "risk_permission",
                "fail_policy",
                "order_intents",
                "execution_results",
            )
        },
        "data_snapshot_id": "snapshot",
        "live_order_effect": "none",
        "writer_status": {
            "mode": "read_only",
            "atomic": True,
            "append_only": True,
            "write_latency_ms": 1.0,
        },
        "error": None,
    }


def test_v5_decision_receipt_schema_is_strict_and_rejects_sensitive_keys() -> None:
    schema_path = Path(__file__).resolve().parents[2] / "schemas/v5_decision_receipt.schema.json"
    schema = __import__("json").loads(schema_path.read_text(encoding="utf-8"))
    receipt = _receipt_fixture()
    jsonschema.Draft202012Validator(schema).validate(receipt)
    receipt["raw_features"]["value"] = {"api_key": "x"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(receipt)


def test_report_days_and_decisions_are_consistent_across_formats() -> None:
    days = 1.25
    status = {"forward_available_days": days}
    performance = {"forward_available_days": days}
    dashboard = (
        '<html data-forward-available-days="1.250000000">'
        "Forward Runner Funding 时间窗口 INVALIDATED PRODUCTION ALPHA · FROZEN "
        "已实现净复合 未实现净标记 BTC 基准 动态币池基准 Parameter lock "
        "PROVISIONAL INITIALIZATION · REJECTED INCONCLUSIVE</html>"
    )
    markdown = "| Forward available days | `1.250000000` |"
    decisions = {
        "forward_v21": {
            "available_days": days,
            "portfolio_validity": "INCONCLUSIVE",
            "deployment_readiness": "INCONCLUSIVE",
        },
        "current_v5": {"production_alpha": "FROZEN"},
        "live_or_live_small_permitted": False,
    }
    result = validate_v21_report_contract(
        status=status,
        performance=performance,
        dashboard_html=dashboard,
        forward_markdown=markdown,
        decisions=decisions,
    )
    assert result["consistent"] is True
    decisions["forward_v21"]["portfolio_validity"] = "PASS"
    with pytest.raises(ValueError, match="automatically promoted"):
        validate_v21_report_contract(
            status=status,
            performance=performance,
            dashboard_html=dashboard,
            forward_markdown=markdown,
            decisions=decisions,
        )


def test_mixed_funding_frequencies_share_natural_time_semantics_without_zero_fill() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for symbol, step in (("FOUR", 4), ("EIGHT", 8)):
        for hour in range(0, 24 * 21, step):
            if symbol == "FOUR" and hour == 100:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "funding_ts": start + timedelta(hours=hour),
                    "funding_rate": 0.001 if symbol == "FOUR" else 0.002,
                }
            )
    funding = pl.DataFrame(rows)
    rolled = time_based_funding_observations(funding)
    latest = rolled.group_by("symbol").agg(
        pl.col("observation_count_20d").last().alias("count"),
        pl.col("funding_mean_20d").last().alias("mean"),
    ).sort("symbol")
    counts = dict(zip(latest["symbol"], latest["count"], strict=True))
    means = dict(zip(latest["symbol"], latest["mean"], strict=True))
    assert counts["FOUR"] > counts["EIGHT"]
    assert means == pytest.approx({"FOUR": 0.001, "EIGHT": 0.002})
    assert rolled["funding_mean_20d"].null_count() == 0


def test_forward_runner_integration_writes_all_outputs_and_is_idempotent(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    symbols = ["BTC-USDT", *[f"S{index:02d}-USDT" for index in range(19)]]
    rows: list[dict] = []
    for hour in range(1504):
        timestamp = start + timedelta(hours=hour)
        for symbol_index, symbol in enumerate(symbols):
            close = (
                300.0 - hour * 0.01
                if symbol == "BTC-USDT"
                else 10.0 + symbol_index + hour * 0.0001
            )
            rows.append(
                {
                    "symbol": symbol,
                    "ts": timestamp,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 10_000.0,
                    "quote_volume": 2_000_000.0,
                }
            )
    bars = pl.DataFrame(rows)
    cutoff = start + timedelta(hours=1500, minutes=30)
    root = tmp_path / "v21"
    v1_root = tmp_path / "v1"
    (root / "manifests").mkdir(parents=True)
    (root / "artifacts").mkdir(parents=True)
    (root / "state").mkdir(parents=True)
    (v1_root / "data/silver").mkdir(parents=True)
    bars.filter(pl.col("ts") <= start + timedelta(hours=1499)).write_parquet(
        v1_root / "data/silver/bars_1h.parquet"
    )
    bars.filter(pl.col("ts") > start + timedelta(hours=1499)).write_parquet(
        root / "state/forward_v21_market_bars.parquet"
    )
    lock = {
        "strategy_id": "low_vol_btc_trend_top3_score_120h_v21",
        "hypothesis_type": "POST_HOC_HYPOTHESIS",
        "parameters_locked": True,
        "factor": "low_vol_20d",
        "factor_formula": "-rolling_std(ret_1h,480)",
        "factor_lookback_hours": 480,
        "universe": "top20",
        "universe_definition_hash": "c" * 64,
        "top_n": 3,
        "weighting": "score",
        "max_single_weight": 0.5,
        "rebalance_hours": 120,
        "holding_hours": 120,
        "rebalance_anchor_rule": "first_full_hour_strictly_after_cutoff",
        "btc_trend_filter": True,
        "btc_trend_lookback": "1440h",
        "btc_trend_lookback_hours": 1440,
        "btc_trend_formula": "BTC close >= MA",
        "decision_delay_bars": 1,
        "entry_price_rule": "next_bar_close",
        "exit_price_rule": "scheduled_exit_or_next_available_close",
        "strategy_type": "long_only_spot",
        "round_trip_cost_bps": 30.0,
        "cost_model": {},
        "cost_model_hash": "d" * 64,
        "maximum_drawdown_review_limit": 0.30,
        "maximum_symbol_contribution_share_review_limit": 0.50,
        "data_snapshot_id": "snapshot",
        "v1_snapshot_id": "v1",
        "code_commit": "a" * 40,
        "runner_version": "quant_lab_low_vol_forward.v2.1",
        "bar_timestamp_semantics": "bar_open_time",
        "bar_close_available_delay_hours": 1,
        "created_at": cutoff.isoformat(),
    }
    lock["sha256"] = parameter_lock_digest(lock)
    (root / "manifests/parameter_lock_v21.json").write_text(
        __import__("json").dumps(lock), encoding="utf-8"
    )
    cutoff_manifest = {
        "forward_v21_start_cutoff": cutoff.isoformat(),
        "parameter_lock_hash": lock["sha256"],
        "v1_data_cutoff": (start + timedelta(hours=1499)).isoformat(),
    }
    (root / "manifests/forward_v21_cutoff.json").write_text(
        __import__("json").dumps(cutoff_manifest), encoding="utf-8"
    )
    (root / "artifacts/legacy_forward_v2_status.json").write_text(
        __import__("json").dumps(
            {
                "invalidated_at": cutoff.isoformat(),
                "status": "INVALIDATED_BY_RUNNER_BUG",
                "preserved": True,
                "may_merge_with_v21": False,
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    first = run(
        root=root,
        v1_root=v1_root,
        requested_as_of=start + timedelta(hours=1504),
        resume=True,
        dry_run=False,
        no_fetch=True,
    )
    events_before = pl.read_parquet(
        root / "artifacts/forward_v21_events.parquet"
    )
    second = run(
        root=root,
        v1_root=v1_root,
        requested_as_of=start + timedelta(hours=1504),
        resume=True,
        dry_run=False,
        no_fetch=True,
    )
    events_after = pl.read_parquet(root / "artifacts/forward_v21_events.parquet")
    assert first == second
    assert events_before.equals(events_after)
    assert set(events_after["event_type"]) == {
        "LEGACY_V2_RECORD_INVALIDATED",
        "DECISION_CREATED",
    }
    assert first["decision_count"] == 1
    assert first["cash_decision_count"] == 1
    assert first["entry_count"] == 0
    assert first["btc_benchmark_compounded_return"] != 0.0
    for name in (
        "forward_v21_decisions.parquet",
        "forward_v21_trades.parquet",
        "forward_v21_events.parquet",
        "forward_v21_equity.parquet",
        "forward_v21_benchmarks.parquet",
        "forward_v21_performance.csv",
        "forward_v21_status.json",
    ):
        assert (root / "artifacts" / name).is_file()
