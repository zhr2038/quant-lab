from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.opportunity_cost.ledger import build_opportunity_cost_frames
from quant_lab.trade_level.bucket_policy import build_trade_level_bucket_policy
from quant_lab.trade_level.judgment import (
    build_trade_level_frames_from_sources,
    build_trade_level_judgments,
    build_trade_opportunity_events,
)
from quant_lab.trade_level.labels import build_trade_opportunity_labels
from quant_lab.trade_level.opportunity_queue import build_trade_level_opportunity_queue
from quant_lab.trade_level.similarity import build_trade_level_similarity_outcome


def test_sol_high_confidence_abort_becomes_micro_canary_review():
    frames = build_trade_level_frames_from_sources(
        candidate_events=pl.DataFrame([_sol_candidate()]),
        candidate_labels=pl.DataFrame(
            [
                {
                    "candidate_id": "sol-cand-1",
                    "run_id": "run-sol",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.local_alpha6",
                    "horizon_hours": 24,
                    "net_bps_after_cost": 42.0,
                    "mfe_bps": 90.0,
                    "mae_bps": -18.0,
                    "win": True,
                    "label_status": "complete",
                    "label_reason": "ok",
                }
            ]
        ),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
                    "live_block_reasons": (
                        '["no_strategy_live_small_ready",'
                        '"quant_lab_advisory_permission_not_allow",'
                        '"quant_lab_live_command_not_allowed",'
                        '"v5_local_live_not_controlled_by_quant_lab"]'
                    ),
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )

    judgment = frames["trade_level_judgment"].row(0, named=True)

    assert judgment["hard_safety_veto"] is False
    assert judgment["risk_permission_veto"] is True
    assert judgment["v5_high_confidence_opportunity"] is True
    assert judgment["trade_level_decision"] == "MICRO_CANARY_REVIEW"
    assert judgment["max_single_order_usdt"] == 0.0
    assert frames["quant_lab_false_block_audit"].row(0, named=True)["false_block"] is True
    sample = frames["v5_trade_learning_sample"].row(0, named=True)
    assert sample["sample_type"] == "COUNTERFACTUAL_SUCCESS"
    assert sample["quant_lab_false_block_candidate"] is True


def test_high_confidence_missing_arrival_mid_gets_observability_review_block():
    candidate = _sol_candidate(candidate_id="sol-missing-mid")
    candidate.pop("arrival_mid")
    frames = build_trade_level_frames_from_sources(
        candidate_events=pl.DataFrame([candidate]),
        candidate_labels=pl.DataFrame(
            [
                {
                    "candidate_id": "sol-missing-mid",
                    "run_id": "run-sol",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.local_alpha6",
                    "horizon_hours": 24,
                    "net_bps_after_cost": 42.0,
                    "label_status": "complete",
                    "label_reason": "ok",
                }
            ]
        ),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
                    "live_block_reasons": '["no_strategy_live_small_ready"]',
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )

    judgment = frames["trade_level_judgment"].row(0, named=True)

    assert judgment["v5_high_confidence_opportunity"] is True
    assert judgment["trade_level_decision"] == "MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY"
    assert "arrival_mid_missing" in judgment["reason"]
    assert "trade_level_not_live_ready" not in judgment["reason"]


def test_trade_opportunity_event_preserves_quote_metadata():
    frames = build_trade_level_frames_from_sources(
        candidate_events=pl.DataFrame(
            [
                _sol_candidate()
                | {
                    "quote_ts": "2026-06-29T08:05:01Z",
                    "quote_age_ms": 320.0,
                    "quote_source": "okx_books5",
                }
            ]
        ),
        candidate_labels=pl.DataFrame(),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
                    "live_block_reasons": '["no_strategy_live_small_ready"]',
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )

    event = frames["trade_opportunity_event"].row(0, named=True)

    assert event["quote_ts"] == "2026-06-29T08:05:01Z"
    assert event["quote_age_ms"] == 320.0
    assert event["quote_source"] == "okx_books5"


def test_trade_opportunity_label_records_source_and_derived_availability():
    decision_ts = datetime(2026, 7, 1, tzinfo=UTC)
    events = pl.DataFrame(
        [
            {
                "event_id": "event-a",
                "candidate_id": "candidate-a",
                "run_id": "run-a",
                "symbol": "BTC-USDT",
                "strategy_candidate": "strategy-a",
                "decision_ts": decision_ts,
            }
        ]
    )
    labels = build_trade_opportunity_labels(
        events,
        pl.DataFrame(
            [
                {
                    "candidate_id": "candidate-a",
                    "run_id": "run-a",
                    "symbol": "BTC-USDT",
                    "strategy_candidate": "strategy-a",
                    "horizon_hours": 4,
                    "decision_ts": decision_ts,
                    "label_ts": decision_ts + timedelta(hours=4, minutes=5),
                    "net_bps_after_cost": 4.0,
                    "label_status": "complete",
                },
                {
                    "candidate_id": "candidate-a",
                    "run_id": "run-a",
                    "symbol": "BTC-USDT",
                    "strategy_candidate": "strategy-a",
                    "horizon_hours": 8,
                    "decision_ts": decision_ts,
                    "net_bps_after_cost": 8.0,
                    "label_status": "complete",
                },
            ]
        ),
        created_at=decision_ts + timedelta(days=2),
    ).row(0, named=True)

    assert labels["schema_version"] == "trade_opportunity_label.v0.2"
    assert labels["label_4h_available_at"] == decision_ts + timedelta(hours=4, minutes=5)
    assert labels["label_4h_availability_source"] == "candidate_label_label_ts"
    assert labels["label_8h_available_at"] == decision_ts + timedelta(hours=8)
    assert labels["label_8h_availability_source"] == "derived_from_horizon"


@pytest.mark.parametrize(
    ("elapsed_hours", "expected_count", "expected_mean"),
    [
        (1, 0, None),
        (5, 1, 4.0),
        (9, 1, 8.0),
        (25, 1, 24.0),
    ],
)
def test_trade_level_similarity_uses_only_causally_available_outcome(
    elapsed_hours: int,
    expected_count: int,
    expected_mean: float | None,
):
    prior_ts = datetime(2026, 7, 1, tzinfo=UTC)
    current_ts = prior_ts + timedelta(hours=elapsed_hours)
    events = pl.DataFrame(
        [
            _similarity_event("prior", prior_ts),
            _similarity_event("current", current_ts),
        ]
    )
    labels = pl.DataFrame(
        [
            {
                "event_id": "prior",
                "decision_ts": prior_ts,
                "label_4h_after_cost_bps": 4.0,
                "label_8h_after_cost_bps": 8.0,
                "label_24h_after_cost_bps": 24.0,
                "label_4h_available_at": prior_ts + timedelta(hours=4),
                "label_8h_available_at": prior_ts + timedelta(hours=8),
                "label_24h_available_at": prior_ts + timedelta(hours=24),
                "label_4h_mae_bps": -4.0,
                "label_8h_mae_bps": -8.0,
                "label_24h_mae_bps": -24.0,
            }
        ]
    )

    current = (
        build_trade_level_similarity_outcome(events, labels, created_at=current_ts)
        .filter(pl.col("event_id") == "current")
        .row(0, named=True)
    )

    assert current["similar_sample_count"] == expected_count
    assert current["similar_mean_after_cost_bps"] == expected_mean
    assert current["similar_max_adverse_bps"] == (
        -expected_mean if expected_mean is not None else None
    )


def test_trade_level_similarity_excludes_same_timestamp_and_self():
    decision_ts = datetime(2026, 7, 1, tzinfo=UTC)
    events = pl.DataFrame(
        [
            _similarity_event("event-a", decision_ts),
            _similarity_event("event-b", decision_ts),
        ]
    )
    labels = pl.DataFrame(
        [
            {
                "event_id": event_id,
                "decision_ts": decision_ts,
                "label_4h_after_cost_bps": value,
                "label_4h_available_at": decision_ts,
                "label_4h_mae_bps": -value,
            }
            for event_id, value in (("event-a", 1.0), ("event-b", 2.0))
        ]
    )

    similarity = build_trade_level_similarity_outcome(
        events,
        labels,
        created_at=decision_ts,
    )

    assert similarity["similar_sample_count"].to_list() == [0, 0]


def test_trade_opportunity_event_uses_point_in_time_risk_permission():
    candidate = _sol_candidate()
    candidate["decision_ts"] = datetime(2026, 7, 1, 10, tzinfo=UTC)
    candidate["ts_utc"] = candidate["decision_ts"]
    events = build_trade_opportunity_events(
        pl.DataFrame([candidate]),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 7, 1, 9, tzinfo=UTC),
                    "live_block_reasons": '["no_strategy_live_small_ready"]',
                },
                {
                    "permission": "ALLOW",
                    "permission_status": "ACTIVE_ALLOW",
                    "as_of_ts": datetime(2026, 7, 1, 11, tzinfo=UTC),
                    "live_block_reasons": "[]",
                },
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 7, 1, 12, tzinfo=UTC),
    ).row(0, named=True)

    assert events["schema_version"] == "trade_opportunity_event.v0.3"
    assert events["quant_lab_permission"] == "ABORT"
    assert events["risk_permission_as_of_ts"] == datetime(2026, 7, 1, 9, tzinfo=UTC)
    assert events["risk_permission_source"] == "risk_permission_asof_join"
    assert events["risk_permission_status_at_decision"] == "ACTIVE_ABORT"


def test_trade_opportunity_event_prefers_signed_permission_context():
    candidate = _sol_candidate()
    candidate["decision_ts"] = datetime(2026, 7, 1, 10, tzinfo=UTC)
    candidate["ts_utc"] = candidate["decision_ts"]
    candidate["raw_payload_json"] = json.dumps(
        {
            "quant_lab_permission_context": {
                "permission": "ABORT",
                "permission_status": "ACTIVE_ABORT",
                "as_of_ts": "2026-07-01T09:00:00Z",
                "live_block_reasons": ["signed_context_block"],
                "allowed_live_modes": [],
            }
        }
    )
    event = build_trade_opportunity_events(
        pl.DataFrame([candidate]),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ALLOW",
                    "permission_status": "ACTIVE_ALLOW",
                    "as_of_ts": datetime(2026, 7, 1, 9, 30, tzinfo=UTC),
                    "live_block_reasons": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 7, 1, 12, tzinfo=UTC),
    ).row(0, named=True)

    assert event["quant_lab_permission"] == "ABORT"
    assert event["risk_permission_source"] == "candidate_event_signed_context"
    assert event["risk_permission_as_of_ts"] == datetime(2026, 7, 1, 9, tzinfo=UTC)
    assert "signed_context_block" in event["quant_lab_live_block_reasons"]


def test_missing_point_in_time_risk_permission_fails_closed():
    candidate = _sol_candidate()
    candidate["decision_ts"] = datetime(2026, 7, 1, 10, tzinfo=UTC)
    candidate["ts_utc"] = candidate["decision_ts"]
    events = build_trade_opportunity_events(
        pl.DataFrame([candidate]),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ALLOW",
                    "permission_status": "ACTIVE_ALLOW",
                    "as_of_ts": datetime(2026, 7, 1, 11, tzinfo=UTC),
                }
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 7, 1, 12, tzinfo=UTC),
    )
    event = events.row(0, named=True)
    judgment = build_trade_level_judgments(
        events,
        created_at=datetime(2026, 7, 1, 12, tzinfo=UTC),
    ).row(0, named=True)

    assert event["quant_lab_permission"] == "UNKNOWN"
    assert event["quant_lab_permission_status"] == "MISSING"
    assert event["risk_permission_source"] == "missing"
    assert judgment["trade_level_decision"] == "RISK_BLOCK"
    assert judgment["max_single_order_usdt"] == 0.0


def test_hard_safety_reason_always_hard_blocks():
    frames = build_trade_level_frames_from_sources(
        candidate_events=pl.DataFrame([_sol_candidate(candidate_id="sol-hard")]),
        candidate_labels=pl.DataFrame(
            [
                {
                    "candidate_id": "sol-cand-119",
                    "run_id": "run-sol-119",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.local_alpha6",
                    "horizon_hours": 24,
                    "net_bps_after_cost": -11.692423,
                    "mfe_bps": 5.0,
                    "mae_bps": -20.0,
                    "win": False,
                    "label_status": "complete",
                    "label_reason": "ok",
                }
            ]
        ),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
                    "live_block_reasons": '["reconcile_fail"]',
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )

    judgment = frames["trade_level_judgment"].row(0, named=True)

    assert judgment["hard_safety_veto"] is True
    assert judgment["trade_level_decision"] == "HARD_BLOCK"
    assert "reconcile_fail" in judgment["hard_safety_reasons"]


def test_supported_similar_sample_requires_bucket_policy_before_micro_canary_allow():
    events = build_trade_opportunity_events(
        pl.DataFrame([_sol_candidate(candidate_id="sol-allow")]),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
                    "live_block_reasons": '["no_strategy_live_small_ready"]',
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )
    event = events.row(0, named=True)
    similarity = pl.DataFrame(
        [
            {
                "event_id": event["event_id"],
                "decision_ts": event["decision_ts"],
                "symbol": "SOL-USDT",
                "similar_sample_count": 20,
                "similar_median_after_cost_bps": 12.0,
                "similar_p25_after_cost_bps": -10.0,
                "recent_7d_similar_mean": 5.0,
            }
        ]
    )

    judgments = build_trade_level_judgments(
        events,
        similarity=similarity,
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )
    judgment = judgments.row(0, named=True)

    assert judgment["trade_level_decision"] == "MICRO_CANARY_REVIEW"
    assert "similar_sample_supported_but_bucket_policy_required" in judgment["reason"]
    assert judgment["max_single_order_usdt"] == 0.0
    assert judgment["daily_trade_limit"] == 0


def test_bucket_policy_can_explicitly_promote_to_micro_canary_allow():
    events = build_trade_opportunity_events(
        pl.DataFrame([_sol_candidate(candidate_id="sol-policy-allow")]),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
                    "live_block_reasons": '["no_strategy_live_small_ready"]',
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )
    event = events.row(0, named=True)
    policy = pl.DataFrame(
        [
            {
                "bucket_key": "|".join(
                    [
                        event["symbol"],
                        event["strategy_candidate"],
                        "normal",
                        "normal",
                        "rank_1",
                        "alpha_ge_0_95",
                        "edge_ratio_ge_3",
                        "unknown_cost_source",
                        "cost_gate_verified",
                    ]
                ),
                "policy_action": "MICRO_CANARY_ALLOW",
                "policy_reason": "explicit_test_policy",
                "policy_confidence": "high",
                "max_single_order_usdt": 5.0,
                "daily_trade_limit": 1,
                "expires_at": datetime(2026, 6, 30, tzinfo=UTC),
                "created_at": datetime(2026, 6, 29, 8, tzinfo=UTC),
            }
        ]
    )

    judgments = build_trade_level_judgments(
        events,
        similarity=pl.DataFrame(
            [
                {
                    "event_id": event["event_id"],
                    "similar_sample_count": 24,
                    "similar_median_after_cost_bps": 12.0,
                    "similar_p25_after_cost_bps": -12.0,
                    "recent_7d_similar_mean": 8.0,
                }
            ]
        ),
        bucket_policy=policy,
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )
    judgment = judgments.row(0, named=True)

    assert judgment["trade_level_decision"] == "MICRO_CANARY_ALLOW"
    assert judgment["bucket_policy_action"] == "MICRO_CANARY_ALLOW"
    assert judgment["max_single_order_usdt"] == 5.0
    assert judgment["daily_trade_limit"] == 1


def test_sol_live_success_becomes_learning_sample_not_live_allow():
    candidate = _sol_candidate()
    candidate["actual_all_in_cost_bps"] = 40.0
    non_open_candidate = _sol_candidate(candidate_id="sol-no-open")
    non_open_candidate["run_id"] = "run-sol-no-open"
    non_open_candidate["intent"] = "BLOCKED"
    non_open_candidate["final_decision"] = "HOLD"
    non_open_candidate["target_weight_after_risk"] = 0.0
    frames = build_trade_level_frames_from_sources(
        candidate_events=pl.DataFrame([candidate, non_open_candidate]),
        candidate_labels=pl.DataFrame(
            [
                {
                    "candidate_id": "sol-cand-1",
                    "run_id": "run-sol",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.local_alpha6",
                    "horizon_hours": 24,
                    "net_bps_after_cost": 161.0,
                    "mfe_bps": 180.0,
                    "mae_bps": -12.0,
                    "win": True,
                    "label_status": "complete",
                    "label_reason": "ok",
                }
            ]
        ),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
                    "live_block_reasons": '["no_strategy_live_small_ready"]',
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(
            [
                {
                    "run_id": "run-sol",
                    "symbol": "SOL-USDT",
                    "ts_utc": datetime(2026, 6, 29, 8, 6, tzinfo=UTC),
                    "side": "buy",
                    "action": "entry_filled",
                    "price": "73.84",
                    "qty": "0.2135",
                },
                {
                    "run_id": "run-sol",
                    "symbol": "SOL-USDT",
                    "ts_utc": datetime(2026, 6, 29, 9, 12, tzinfo=UTC),
                    "side": "sell",
                    "action": "exit_filled",
                    "price": "75.02",
                    "qty": "0.2135",
                    "raw_payload_json": (
                        '{"exit_reason":"protect_profit_lock_trailing",'
                        '"net_bps":161.0,"net_pnl_usdt":0.25}'
                    ),
                },
            ]
        ),
        v5_roundtrips=pl.DataFrame(
            [
                {
                    "open_run_id": "run-sol",
                    "close_run_id": "run-sol-exit",
                    "symbol": "SOL-USDT",
                    "open_time_utc": datetime(2026, 6, 29, 13, 0, 51, tzinfo=UTC),
                    "close_time_utc": datetime(2026, 6, 29, 18, 1, tzinfo=UTC),
                    "net_bps": "9.66",
                    "net_pnl_usdt": "0.01527",
                }
            ]
        ),
        order_lifecycles=pl.DataFrame(
            [
                {
                    "run_id": "run-sol",
                    "symbol": "SOL-USDT",
                    "ts_utc": datetime(2026, 6, 29, 9, 12, tzinfo=UTC),
                    "raw_payload_json": (
                        '{"exit_reason":"protect_profit_lock_trailing",'
                        '"realized_total_cost_bps":40.0}'
                    ),
                }
            ]
        ),
        created_at=datetime(2026, 6, 29, 10, tzinfo=UTC),
    )

    judgment = frames["trade_level_judgment"].row(0, named=True)
    sample = frames["v5_trade_learning_sample"].row(0, named=True)
    attribution = frames["v5_trade_outcome_attribution"].row(0, named=True)
    audit = frames["quant_lab_false_block_audit"].row(0, named=True)
    opportunity = frames["quant_lab_opportunity_cost_event"].row(0, named=True)
    opportunity_daily = frames["quant_lab_opportunity_cost_daily"].row(0, named=True)
    opportunity_bucket = frames["opportunity_cost_by_bucket"].row(0, named=True)
    decision_regret = frames["quant_lab_decision_regret"].row(0, named=True)

    assert judgment["v5_high_confidence_opportunity"] is True
    assert judgment["trade_level_decision"] == "MICRO_CANARY_REVIEW"
    assert sample["sample_type"] == "LIVE_SUCCESS"
    assert sample["actual_order_submitted"] is True
    assert sample["actual_exit_reason"] == "protect_profit_lock_trailing"
    assert sample["actual_hold_minutes"] == 66.0
    assert sample["actual_roundtrip_net_bps"] == 161.0
    assert sample["actual_outcome_label"] == "PROFITABLE"
    assert sample["fixed_horizon_net_bps"] == 161.0
    assert sample["fixed_horizon_outcome_label"] == "PROFITABLE"
    assert sample["net_bps"] == 161.0
    assert sample["quant_lab_false_block_candidate"] is True
    assert sample["learning_eligible"] is True
    assert audit["sample_id"] == sample["sample_id"]
    assert audit["false_block"] is True
    assert attribution["entry_signal_quality"] == "PASS"
    assert attribution["exit_quality"] == "PASS"
    assert attribution["execution_quality"] == "WARN"
    assert attribution["cost_underestimated"] is True
    assert attribution["profit_lock_contribution"] is True
    assert opportunity["regret_type"] == "false_block"
    assert opportunity["cost_source"] == "unknown"
    assert opportunity["missed_profit_bps"] == 161.0
    assert opportunity["regret_bps"] == 161.0
    assert opportunity_daily["false_block_count"] == 1
    assert opportunity_daily["loss_saved_count"] == 0
    assert opportunity_daily["total_v5_would_open_count"] == 1
    assert opportunity_daily["quant_lab_would_block_count"] == 1
    assert opportunity_daily["veto_net_value_bps"] == -161.0
    assert opportunity_daily["opportunity_cost_status"] == "VETO_VALUE_NEGATIVE_REVIEW_EXCEPTIONS"
    assert opportunity_bucket["false_block_count"] == 1
    assert opportunity_bucket["cost_source"] == "unknown_cost_source"
    assert opportunity_bucket["missed_profit_bps_sum"] == 161.0
    assert decision_regret["regret_type"] == "false_block"
    assert decision_regret["best_hindsight_action"] == "ALLOW"
    assert decision_regret["regret_bps"] == 161.0


def test_live_sample_prefers_actual_roundtrip_over_negative_fixed_horizon_label():
    candidate = _sol_candidate()
    candidate["decision_ts"] = datetime(2026, 6, 29, 13, 0, 51, tzinfo=UTC)
    candidate["ts_utc"] = candidate["decision_ts"]
    frames = build_trade_level_frames_from_sources(
        candidate_events=pl.DataFrame([candidate]),
        candidate_labels=pl.DataFrame(
            [
                {
                    "candidate_id": "sol-cand-1",
                    "run_id": "run-sol",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.local_alpha6",
                    "horizon_hours": 24,
                    "net_bps_after_cost": -137.2,
                    "mfe_bps": 180.0,
                    "mae_bps": -170.0,
                    "win": False,
                    "label_status": "complete",
                    "label_reason": "ok",
                }
            ]
        ),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 12, tzinfo=UTC),
                    "live_block_reasons": '["no_strategy_live_small_ready"]',
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(
            [
                {
                    "run_id": "run-sol",
                    "symbol": "SOL-USDT",
                    "ts_utc": datetime(2026, 6, 29, 13, 0, 51, tzinfo=UTC),
                    "side": "buy",
                    "action": "entry_filled",
                    "price": "73.84",
                    "qty": "0.213705",
                    "notional_usdt": "15.7799772",
                    "fee_usdt": "0.0157799772",
                },
                {
                    "run_id": "run-sol-exit",
                    "symbol": "SOL-USDT",
                    "ts_utc": datetime(2026, 6, 29, 18, 1, tzinfo=UTC),
                    "side": "sell",
                    "action": "exit_filled",
                    "price": "75.18",
                    "qty": "0.213491",
                    "notional_usdt": "16.05025338",
                    "fee_usdt": "0.01605025338",
                },
            ]
        ),
        v5_roundtrips=pl.DataFrame(
            [
                {
                    "open_run_id": "run-sol",
                    "close_run_id": "run-sol-exit",
                    "symbol": "SOL-USDT",
                    "open_time_utc": datetime(2026, 6, 29, 13, 0, 51, tzinfo=UTC),
                    "close_time_utc": datetime(2026, 6, 29, 18, 1, tzinfo=UTC),
                    "net_bps": "9.66",
                    "net_pnl_usdt": "0.01527",
                }
            ]
        ),
        order_lifecycles=pl.DataFrame(
            [
                {
                    "run_id": "run-sol-exit",
                    "symbol": "SOL-USDT",
                    "ts_utc": datetime(2026, 6, 29, 18, 1, tzinfo=UTC),
                    "side": "sell",
                    "intent": "CLOSE_LONG",
                    "avg_fill_px": "75.18",
                    "filled_qty": "0.213491",
                    "notional_usdt": "16.05025338",
                    "fee_usdt": "0.01605025338",
                    "exit_reason": "protect_profit_lock_trailing",
                    "raw_payload_json": (
                        '{"exit_reason":"protect_profit_lock_trailing",'
                        '"realized_total_cost_bps":24.6,'
                        '"first_fill_ts":"2026-06-29T18:01:00Z",'
                        '"last_fill_ts":"2026-06-29T18:01:00Z"}'
                    ),
                }
            ]
        ),
        created_at=datetime(2026, 6, 29, 19, tzinfo=UTC),
    )

    sample = frames["v5_trade_learning_sample"].row(0, named=True)
    audit = frames["quant_lab_false_block_audit"].row(0, named=True)
    opportunity = frames["quant_lab_opportunity_cost_event"].row(0, named=True)
    expected_pnl = 16.05025338 - 15.7799772 - 0.0157799772 - 0.01605025338
    expected_bps = expected_pnl / 15.7799772 * 10_000.0

    assert sample["sample_type"] == "LIVE_SUCCESS"
    assert sample["outcome_label"] == "PROFITABLE"
    assert sample["actual_outcome_label"] == "PROFITABLE"
    assert sample["actual_exit_reason"] == "protect_profit_lock_trailing"
    assert sample["actual_roundtrip_net_bps"] == pytest.approx(expected_bps)
    assert sample["actual_roundtrip_net_pnl_usdt"] == pytest.approx(expected_pnl)
    assert sample["fill_to_fill_net_bps"] == pytest.approx(9.66)
    assert sample["fill_to_fill_net_pnl_usdt"] == pytest.approx(0.01527)
    assert sample["execution_adjusted_net_bps"] == pytest.approx(expected_bps)
    assert sample["execution_adjusted_net_pnl_usdt"] == pytest.approx(expected_pnl)
    assert sample["learning_net_bps"] == pytest.approx(expected_bps)
    assert sample["learning_net_pnl_usdt"] == pytest.approx(expected_pnl)
    assert sample["learning_return_basis"] == "actual_execution_adjusted_roundtrip"
    assert sample["actual_hold_minutes"] == 300.15
    assert sample["fixed_horizon_net_bps"] == -137.2
    assert sample["fixed_horizon_outcome_label"] == "UNPROFITABLE"
    assert sample["label_24h_after_cost_bps"] == -137.2
    assert sample["hold_minutes"] == 300.15
    assert sample["net_bps"] == pytest.approx(expected_bps)
    assert audit["actual_or_counterfactual_after_cost_bps"] == pytest.approx(expected_bps)
    assert audit["false_block"] is True
    assert opportunity["after_cost_bps"] == pytest.approx(expected_bps)
    assert opportunity["regret_type"] == "false_block"


def test_learning_sample_schema_does_not_infer_nullable_float_as_null():
    candidates = []
    for index in range(120):
        row = _sol_candidate(candidate_id=f"sol-cand-{index}")
        row["run_id"] = f"run-sol-{index}"
        candidates.append(row)
    frames = build_trade_level_frames_from_sources(
        candidate_events=pl.DataFrame(candidates),
        candidate_labels=pl.DataFrame(
            [
                {
                    "candidate_id": "sol-cand-119",
                    "run_id": "run-sol-119",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.local_alpha6",
                    "horizon_hours": 24,
                    "net_bps_after_cost": -11.692423,
                    "mfe_bps": 5.0,
                    "mae_bps": -20.0,
                    "win": False,
                    "label_status": "complete",
                    "label_reason": "ok",
                }
            ]
        ),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
                    "live_block_reasons": '["no_strategy_live_small_ready"]',
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(
            [
                {
                    "run_id": "run-sol-119",
                    "symbol": "SOL-USDT",
                    "ts_utc": datetime(2026, 6, 29, 8, 6, tzinfo=UTC),
                    "side": "buy",
                    "action": "entry_filled",
                    "price": "77383.7",
                    "qty": "0.001",
                }
            ]
        ),
        created_at=datetime(2026, 6, 29, 10, tzinfo=UTC),
    )

    samples = frames["v5_trade_learning_sample"]

    assert samples.height == 120
    assert samples["actual_fill_px"].drop_nulls().to_list() == [77383.7]
    assert frames["trade_opportunity_label"]["label_24h_after_cost_bps"].drop_nulls().to_list() == [
        -11.692423
    ]


def test_opportunity_cost_ignores_non_open_candidates_for_allow_outcomes():
    created = datetime(2026, 6, 29, 10, tzinfo=UTC)
    frames = build_opportunity_cost_frames(
        events=pl.DataFrame(
            [
                {
                    "event_id": "hold-candidate",
                    "decision_ts": datetime(2026, 6, 29, 8, 5, tzinfo=UTC),
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.local_alpha6",
                    "rank": 1,
                    "alpha6_score": 0.984,
                    "edge_required_ratio": 4.0,
                    "cost_gate_verified": True,
                    "cost_source": "bootstrap_cost_probe",
                    "v5_would_open": False,
                    "actual_submitted": False,
                }
            ]
        ),
        labels=pl.DataFrame(
            [
                {
                    "event_id": "hold-candidate",
                    "label_24h_after_cost_bps": -45.0,
                }
            ]
        ),
        judgments=pl.DataFrame(
            [
                {
                    "event_id": "hold-candidate",
                    "trade_level_decision": "MICRO_CANARY_ALLOW",
                    "v5_high_confidence_opportunity": True,
                }
            ]
        ),
        created_at=created,
    )

    event = frames["quant_lab_opportunity_cost_event"].row(0, named=True)
    daily = frames["quant_lab_opportunity_cost_daily"].row(0, named=True)
    regret = frames["quant_lab_decision_regret"].row(0, named=True)

    assert event["v5_would_open"] is False
    assert event["false_allow"] is False
    assert event["correct_allow"] is False
    assert event["regret_type"] == "not_v5_open"
    assert daily["total_v5_would_open_count"] == 0
    assert daily["quant_lab_would_allow_count"] == 0
    assert daily["false_allow_count"] == 0
    assert regret["regret_type"] == "not_v5_open"


def test_opportunity_bucket_recommends_risk_block_for_loss_saved_bucket():
    created = datetime(2026, 6, 29, 10, tzinfo=UTC)
    rows = []
    labels = []
    judgments = []
    for index in range(5):
        event_id = f"bnb-loss-saved-{index}"
        rows.append(
            {
                "event_id": event_id,
                "decision_ts": datetime(2026, 6, 29, 8, index, tzinfo=UTC),
                "symbol": "BNB-USDT",
                "strategy_candidate": "f3_dominant_entry",
                "rank": 1,
                "alpha6_score": 0.984,
                "edge_required_ratio": 2.0,
                "cost_gate_verified": True,
                "cost_source": "quant_lab_cached",
                "regime": "Trending",
                "risk_level": "PROTECT",
                "v5_would_open": True,
                "actual_submitted": False,
            }
        )
        labels.append(
            {"event_id": event_id, "label_24h_after_cost_bps": -45.0 if index < 3 else 0.0}
        )
        judgments.append(
            {
                "event_id": event_id,
                "trade_level_decision": "RISK_BLOCK",
                "v5_high_confidence_opportunity": True,
            }
        )
    frames = build_opportunity_cost_frames(
        events=pl.DataFrame(rows),
        labels=pl.DataFrame(labels),
        judgments=pl.DataFrame(judgments),
        created_at=created,
    )

    bucket = frames["opportunity_cost_by_bucket"].row(0, named=True)

    assert bucket["loss_saved_count"] == 3
    assert bucket["sample_count"] == 5
    assert bucket["veto_net_value_bps"] == 135.0
    assert bucket["opportunity_exception_candidate"] is False
    assert bucket["recommended_trade_level_decision"] == "RISK_BLOCK"
    assert bucket["policy_action"] == "RISK_BLOCK"
    assert bucket["high_confidence_loss_saved_count"] == 3
    assert bucket["loss_saved_bps_median"] == 45.0
    assert bucket["policy_basis"] == "loss_saved_bucket_positive_veto_value"


def test_opportunity_bucket_risk_blocks_weak_signal_loss_saved():
    created = datetime(2026, 6, 29, 10, tzinfo=UTC)
    rows = []
    labels = []
    judgments = []
    for index in range(5):
        event_id = f"eth-weak-loss-saved-{index}"
        rows.append(
            {
                "event_id": event_id,
                "decision_ts": datetime(2026, 6, 29, 8, index, tzinfo=UTC),
                "symbol": "ETH-USDT",
                "strategy_candidate": "f3_dominant_entry",
                "rank": 3,
                "alpha6_score": 0.42,
                "edge_required_ratio": 0.9,
                "cost_gate_verified": True,
                "cost_source": "public_spread_proxy",
                "regime": "Trending",
                "risk_level": "PROTECT",
                "v5_would_open": True,
                "actual_submitted": False,
            }
        )
        labels.append({"event_id": event_id, "label_24h_after_cost_bps": -120.0})
        judgments.append(
            {
                "event_id": event_id,
                "trade_level_decision": "RISK_BLOCK",
                "v5_high_confidence_opportunity": False,
            }
        )
    frames = build_opportunity_cost_frames(
        events=pl.DataFrame(rows),
        labels=pl.DataFrame(labels),
        judgments=pl.DataFrame(judgments),
        created_at=created,
    )

    bucket = frames["opportunity_cost_by_bucket"].row(0, named=True)
    policy = build_trade_level_bucket_policy(
        opportunity_cost_by_bucket=frames["opportunity_cost_by_bucket"],
        policy_date=created.date(),
        created_at=created,
    ).row(0, named=True)
    stale_bucket = pl.DataFrame(
        [
            bucket
            | {
                "schema_version": "opportunity_cost_by_bucket.v0.4",
                "policy_action": "PAPER_ONLY",
                "policy_confidence": "low",
                "policy_basis": "paper_only_until_bucket_policy_clear",
            }
        ]
    )
    upgraded_policy = build_trade_level_bucket_policy(
        opportunity_cost_by_bucket=stale_bucket,
        policy_date=created.date(),
        created_at=created,
    ).row(0, named=True)

    assert bucket["alpha6_bucket"] == "alpha_lt_0_85"
    assert bucket["loss_saved_count"] == 5
    assert bucket["high_confidence_loss_saved_count"] == 0
    assert bucket["policy_action"] == "RISK_BLOCK"
    assert bucket["policy_confidence"] == "medium"
    assert bucket["policy_basis"] == "weak_signal_loss_saved_bucket_positive_veto_value"
    assert policy["policy_action"] == "RISK_BLOCK"
    assert policy["policy_confidence"] == "medium"
    assert policy["policy_reason"] == "weak_signal_loss_saved_bucket_positive_veto_value"
    assert upgraded_policy["policy_action"] == "RISK_BLOCK"
    assert upgraded_policy["policy_reason"] == "weak_signal_loss_saved_bucket_positive_veto_value"


def test_explicit_micro_canary_allow_requires_similarity_evidence():
    events = build_trade_opportunity_events(
        pl.DataFrame([_sol_candidate()]),
        risk_permissions=pl.DataFrame(
            [
                {
                    "permission": "ABORT",
                    "permission_status": "ACTIVE_ABORT",
                    "as_of_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
                    "live_block_reasons": '["no_strategy_live_small_ready"]',
                    "allowed_live_modes": "[]",
                }
            ]
        ),
        v5_trades=pl.DataFrame(),
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )
    event = events.row(0, named=True)
    policy = pl.DataFrame(
        [
            {
                "bucket_key": event["event_id"] and "|".join(
                    [
                        event["symbol"],
                        event["strategy_candidate"],
                        "normal",
                        "normal",
                        "rank_1",
                        "alpha_ge_0_95",
                        "edge_ratio_ge_3",
                        "unknown_cost_source",
                        "cost_gate_verified",
                    ]
                ),
                "policy_action": "MICRO_CANARY_ALLOW",
                "policy_reason": "explicit_test_policy",
                "policy_confidence": "high",
                "max_single_order_usdt": 5.0,
                "daily_trade_limit": 1,
                "expires_at": datetime(2026, 6, 30, tzinfo=UTC),
                "created_at": datetime(2026, 6, 29, 8, tzinfo=UTC),
            }
        ]
    )

    judgments = build_trade_level_judgments(
        events,
        bucket_policy=policy,
        created_at=datetime(2026, 6, 29, 9, tzinfo=UTC),
    )
    judgment = judgments.row(0, named=True)

    assert judgment["trade_level_decision"] == "MICRO_CANARY_REVIEW"
    assert "bucket_policy_allow_requires_similarity_evidence" in judgment["reason"]
    assert judgment["max_single_order_usdt"] == 0.0


def test_bucket_policy_uses_only_prior_day_evidence():
    created = datetime(2026, 6, 29, 10, tzinfo=UTC)
    rows = []
    labels = []
    judgments = []
    for index in range(5):
        event_id = f"sol-prior-false-block-{index}"
        rows.append(
            {
                "event_id": event_id,
                "decision_ts": datetime(2026, 6, 28, 8, index, tzinfo=UTC),
                "symbol": "SOL-USDT",
                "strategy_candidate": "f3_dominant_entry",
                "rank": 1,
                "alpha6_score": 0.984,
                "edge_required_ratio": 4.0,
                "cost_gate_verified": True,
                "cost_source": "bootstrap_cost_probe",
                "regime": "normal",
                "risk_level": "normal",
                "v5_would_open": True,
                "actual_submitted": False,
            }
        )
        labels.append(
            {"event_id": event_id, "label_24h_after_cost_bps": 80.0 if index < 3 else 0.0}
        )
        judgments.append(
            {
                "event_id": event_id,
                "trade_level_decision": "PAPER_ONLY",
                "v5_high_confidence_opportunity": True,
            }
        )
    today_id = "sol-today-false-block"
    rows.append(
        {
            "event_id": today_id,
            "decision_ts": datetime(2026, 6, 29, 8, tzinfo=UTC),
            "symbol": "SOL-USDT",
            "strategy_candidate": "f3_dominant_entry",
            "rank": 1,
            "alpha6_score": 0.984,
            "edge_required_ratio": 4.0,
            "cost_gate_verified": True,
            "cost_source": "bootstrap_cost_probe",
            "regime": "normal",
            "risk_level": "normal",
            "v5_would_open": True,
            "actual_submitted": False,
        }
    )
    labels.append({"event_id": today_id, "label_24h_after_cost_bps": 999.0})
    judgments.append(
        {
            "event_id": today_id,
            "trade_level_decision": "PAPER_ONLY",
            "v5_high_confidence_opportunity": True,
        }
    )
    frames = build_opportunity_cost_frames(
        events=pl.DataFrame(rows),
        labels=pl.DataFrame(labels),
        judgments=pl.DataFrame(judgments),
        created_at=created,
    )

    policy = build_trade_level_bucket_policy(
        opportunity_cost_events=frames["quant_lab_opportunity_cost_event"],
        policy_date=datetime(2026, 6, 29, tzinfo=UTC).date(),
        created_at=created,
    )
    policy_row = policy.row(0, named=True)

    assert policy_row["policy_action"] == "MICRO_CANARY_REVIEW"
    assert policy_row["sample_count"] == 5
    assert policy_row["missed_profit_bps_sum"] == 240.0
    assert policy_row["min_arrival_mid_coverage"] == 0.8
    assert "false_block_bucket_negative_veto_value" in policy_row["policy_reason"]


def test_trade_level_opportunity_queue_marks_review_bucket_blocked_by_observability():
    now = datetime(2026, 6, 29, 10, tzinfo=UTC)
    bucket_key = (
        "SOL-USDT|sol_protect|Trending|PROTECT|rank_1|alpha_ge_0_95|"
        "edge_ratio_ge_3|quant_lab_cached|cost_gate_verified"
    )
    policy = pl.DataFrame(
        [
            {
                "policy_date": now.date(),
                "bucket_key": bucket_key,
                "symbol": "SOL-USDT",
                "strategy_candidate": "sol_protect",
                "regime": "Trending",
                "risk_level": "PROTECT",
                "sample_count": 42,
                "false_block_count": 40,
                "loss_saved_count": 2,
                "veto_net_value_bps": -14524.53,
                "policy_action": "MICRO_CANARY_REVIEW",
                "policy_reason": "false_block_bucket_negative_veto_value_manual_review_only",
                "policy_confidence": "high",
                "expires_at": datetime(2026, 6, 30, tzinfo=UTC),
            }
        ]
    )
    judgments = pl.DataFrame(
        [
            {
                "event_id": "evt-sol-1",
                "bucket_key": policy.row(0, named=True)["bucket_key"],
                "trade_level_decision": "MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY",
            }
        ]
    )

    queue = build_trade_level_opportunity_queue(policy, judgments, created_at=now)
    row = queue.row(0, named=True)

    assert row["next_action"] == "BLOCKED_BY_OBSERVABILITY"
    assert row["observability_status"] == "BLOCKED_BY_OBSERVABILITY"
    assert row["blocked_by_observability_count"] == 1


def test_trade_level_opportunity_queue_keeps_loss_saved_bucket_blocked():
    now = datetime(2026, 6, 29, 10, tzinfo=UTC)
    bucket_key = (
        "BNB-USDT|f3|Trending|PROTECT|rank_1|alpha_lt_0_85|"
        "edge_ratio_lt_1_5|public_spread_proxy|cost_gate_verified"
    )
    policy = pl.DataFrame(
        [
            {
                "policy_date": now.date(),
                "bucket_key": bucket_key,
                "symbol": "BNB-USDT",
                "strategy_candidate": "f3",
                "regime": "Trending",
                "risk_level": "PROTECT",
                "sample_count": 8,
                "false_block_count": 1,
                "loss_saved_count": 5,
                "veto_net_value_bps": 400.0,
                "policy_action": "RISK_BLOCK",
                "policy_reason": "loss_saved_bucket_positive_veto_value",
                "policy_confidence": "high",
                "expires_at": datetime(2026, 6, 30, tzinfo=UTC),
            }
        ]
    )

    queue = build_trade_level_opportunity_queue(policy, pl.DataFrame(), created_at=now)
    row = queue.row(0, named=True)

    assert row["next_action"] == "KEEP_BLOCKED"
    assert row["observability_status"] == "NOT_REQUIRED"


def _sol_candidate(candidate_id: str = "sol-cand-1") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "run_id": "run-sol",
        "ts_utc": datetime(2026, 6, 29, 8, 5, tzinfo=UTC),
        "decision_ts": datetime(2026, 6, 29, 8, 5, tzinfo=UTC),
        "symbol": "SOL/USDT",
        "side": "buy",
        "intent": "OPEN_LONG",
        "strategy_candidate": "v5.local_alpha6",
        "final_decision": "OPEN_LONG",
        "final_score": 0.98,
        "rank": 1,
        "alpha6_score": 0.984,
        "alpha6_side": "buy",
        "expected_edge_bps": 268.0,
        "required_edge_bps": 49.0,
        "cost_bps": 20.0,
        "cost_gate_verified": True,
        "would_block_by_cost": False,
        "arrival_mid": 73.84,
        "target_weight_after_risk": 0.15,
        "regime_state": "normal",
        "risk_level": "normal",
        "source_event_bundle_sha256": "sha-sol",
        "source_path_inside_bundle": "candidate_snapshot.csv",
        "created_at": datetime(2026, 6, 29, 8, 5, tzinfo=UTC) + timedelta(seconds=1),
    }


def _similarity_event(event_id: str, decision_ts: datetime) -> dict[str, object]:
    return {
        "event_id": event_id,
        "decision_ts": decision_ts,
        "symbol": "BTC-USDT",
        "strategy_candidate": "strategy-a",
        "regime": "normal",
        "risk_level": "normal",
        "alpha6_score": 0.9,
        "rank": 1,
        "cost_bps": 10.0,
    }
