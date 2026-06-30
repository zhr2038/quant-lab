from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.trade_level.judgment import (
    build_trade_level_frames_from_sources,
    build_trade_level_judgments,
    build_trade_opportunity_events,
)


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


def test_hard_safety_reason_always_hard_blocks():
    frames = build_trade_level_frames_from_sources(
        candidate_events=pl.DataFrame([_sol_candidate(candidate_id="sol-hard")]),
        candidate_labels=pl.DataFrame(),
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


def test_supported_similar_sample_can_promote_to_micro_canary_allow():
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

    assert judgment["trade_level_decision"] == "MICRO_CANARY_ALLOW"
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
                    "raw_payload_json": '{"exit_reason":"protect_profit_lock_trailing"}',
                },
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

    assert judgment["v5_high_confidence_opportunity"] is True
    assert judgment["trade_level_decision"] == "MICRO_CANARY_REVIEW"
    assert sample["sample_type"] == "LIVE_SUCCESS"
    assert sample["actual_order_submitted"] is True
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
    assert opportunity["missed_profit_bps"] == 161.0
    assert opportunity["regret_bps"] == 161.0
    assert opportunity_daily["false_block_count"] == 1
    assert opportunity_daily["loss_saved_count"] == 0
    assert opportunity_daily["total_v5_would_open_count"] == 1
    assert opportunity_daily["quant_lab_would_block_count"] == 1
    assert opportunity_daily["veto_net_value_bps"] == -161.0
    assert opportunity_daily["opportunity_cost_status"] == "VETO_VALUE_NEGATIVE_REVIEW_EXCEPTIONS"
    assert opportunity_bucket["false_block_count"] == 1
    assert opportunity_bucket["missed_profit_bps_sum"] == 161.0


def test_learning_sample_schema_does_not_infer_nullable_float_as_null():
    candidates = []
    for index in range(120):
        row = _sol_candidate(candidate_id=f"sol-cand-{index}")
        row["run_id"] = f"run-sol-{index}"
        candidates.append(row)
    frames = build_trade_level_frames_from_sources(
        candidate_events=pl.DataFrame(candidates),
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
