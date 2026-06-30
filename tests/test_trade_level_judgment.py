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
