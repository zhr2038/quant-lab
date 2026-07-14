from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.reports.ops_truthfulness import (
    build_api_auth_reports,
    build_complete_acceptance_status,
    build_paper_proposal_propagation_status,
    build_paper_runtime_freshness,
    build_post_fix_funnel_attribution,
)

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def test_auth_incident_uses_natural_no_recurrence_window() -> None:
    metrics = pl.DataFrame(
        [
            {
                "request_ts": (NOW - timedelta(hours=25)).isoformat(),
                "path": "/v1/strategy-opportunity-advisory/v5-compact",
                "client_id": "v5.quant_lab_client",
                "client_host": "113.226.191.85",
                "user_agent": "v5-quant-lab-client/1.0",
                "auth_result": "missing_bearer_token",
                "status_code": 401,
            }
        ]
    )

    incident = build_api_auth_reports(metrics, generated_at=NOW)["api_auth_incident"].to_dicts()[0]

    assert incident["current_status"] == "RECOVERED_24H_CLEAN"
    assert incident["recovered_at"] == ""
    assert incident["clean_hours_after_recovery"] == 25.0
    assert "retired_or_restarted" in incident["suspected_root_cause"]


def test_paper_runtime_no_trade_is_healthy_but_stale_open_trade_warns() -> None:
    fresh = NOW.isoformat()
    healthy = build_paper_runtime_freshness(
        {
            "v5_quant_lab_contract_status": pl.DataFrame([{"generated_at": fresh}]),
            "v5_paper_strategy_registry_current": pl.DataFrame([{"updated_at": fresh}]),
            "v5_paper_strategy_state": pl.DataFrame([{"updated_at": fresh}]),
        },
        generated_at=NOW,
    )
    healthy_status = {row["check_name"]: row["status"] for row in healthy.to_dicts()}
    assert healthy_status["paper_runtime_heartbeat_fresh"] == "PASS"
    assert healthy_status["paper_registry_fresh"] == "PASS"
    assert healthy_status["paper_state_fresh"] == "PASS"
    assert healthy_status["paper_signal_event_fresh"] == "NO_NEW_EVENT_EXPECTED"
    assert healthy_status["paper_trade_event_fresh"] == "NO_NEW_EVENT_EXPECTED"

    warning = build_paper_runtime_freshness(
        {
            "v5_quant_lab_contract_status": pl.DataFrame([{"generated_at": fresh}]),
            "v5_paper_strategy_registry_current": pl.DataFrame([{"updated_at": fresh}]),
            "v5_paper_strategy_state": pl.DataFrame(
                [
                    {
                        "state": "PAPER_OPEN",
                        "updated_at": (NOW - timedelta(hours=4)).isoformat(),
                    }
                ]
            ),
        },
        generated_at=NOW,
    )
    warning_status = {row["check_name"]: row["status"] for row in warning.to_dicts()}
    assert warning_status["paper_signal_event_fresh"] == "WARNING"
    assert warning_status["paper_trade_event_fresh"] == "WARNING"


def test_current_proposal_propagation_distinguishes_accepted_and_unseen() -> None:
    snapshot_id = "proposal-snapshot:test"
    snapshot_sha = "f" * 64
    proposals = pl.DataFrame(
        [
            {
                "proposal_id": "accepted:1",
                "proposal_hash": "a" * 64,
                "proposal_snapshot_id": snapshot_id,
                "proposal_snapshot_sha256": snapshot_sha,
                "snapshot_generated_at": (NOW - timedelta(minutes=10)).isoformat(),
                "created_at": (NOW - timedelta(minutes=10)).isoformat(),
            },
            {
                "proposal_id": "unseen:1",
                "proposal_hash": "b" * 64,
                "proposal_snapshot_id": snapshot_id,
                "proposal_snapshot_sha256": snapshot_sha,
                "snapshot_generated_at": (NOW - timedelta(minutes=10)).isoformat(),
                "created_at": (NOW - timedelta(minutes=5)).isoformat(),
            },
        ]
    )
    ack = pl.DataFrame(
        [
            {
                "proposal_id": "accepted:1",
                "proposal_hash": "a" * 64,
                "accepted": True,
                "source_proposal_snapshot_id": snapshot_id,
                "source_proposal_snapshot_sha256": snapshot_sha,
                "accepted_at": (NOW - timedelta(minutes=8)).isoformat(),
            }
        ]
    )
    trackers = pl.DataFrame(
        [
            {
                "proposal_id": "accepted:1",
                "proposal_hash": "a" * 64,
                "current_proposal_member": True,
                "source_proposal_snapshot_id": snapshot_id,
                "source_proposal_snapshot_sha256": snapshot_sha,
                "created_at": (NOW - timedelta(minutes=7)).isoformat(),
            }
        ]
    )

    rows = build_paper_proposal_propagation_status(
        proposals=proposals,
        ack_current=ack,
        ack_history=pl.DataFrame(),
        trackers_current=trackers,
        trackers_history=pl.DataFrame(),
        selected_v5_bundle_built_at=NOW - timedelta(minutes=1),
        v5_observed_proposal_snapshot_id=snapshot_id,
        v5_observed_proposal_snapshot_sha256=snapshot_sha,
        generated_at=NOW,
    ).to_dicts()
    statuses = {row["proposal_id"]: row["propagation_status"] for row in rows}

    assert statuses == {
        "accepted:1": "ACCEPTED_TRACKER_ACTIVE",
        "unseen:1": "SNAPSHOT_FETCHED_PENDING_PARSE",
    }


def test_expected_401_does_not_reset_production_auth_slo() -> None:
    reports = build_api_auth_reports(
        pl.DataFrame(
            [
                {
                    "request_ts": (NOW - timedelta(minutes=10)).isoformat(),
                    "path": "/v1/health/deep",
                    "client_id": "v5.quant_lab_client",
                    "user_agent": "v5-quant-lab-client/1.0",
                    "auth_result": "token_ok",
                    "status_code": 200,
                },
                {
                    "request_ts": (NOW - timedelta(minutes=5)).isoformat(),
                    "path": "/v1/health/deep",
                    "client_id": "quant-lab.auth-negative-probe",
                    "user_agent": "quant-lab-auth-negative-probe/1.0",
                    "auth_result": "missing_bearer_token",
                    "status_code": 401,
                    "authorization": "Bearer must-not-leak",
                }
            ]
        ),
        generated_at=NOW,
    )

    slo = reports["api_auth_production_slo"].to_dicts()[0]
    incident = reports["api_auth_incident"].to_dicts()[0]
    assert slo["production_auth_slo_status"] == "PASS"
    assert slo["unexpected_auth_failure_count"] == 0
    assert incident["incident_class"] == "EXPECTED_UNAUTHENTICATED_PROBE"
    assert incident["current_status"] == "EXPECTED_REJECTION_PASS"
    assert incident["affects_production_auth_slo"] is False
    assert "must-not-leak" not in repr(reports)


def test_trusted_v5_401_resets_production_auth_slo() -> None:
    reports = build_api_auth_reports(
        pl.DataFrame(
            [
                {
                    "request_ts": (NOW - timedelta(minutes=5)).isoformat(),
                    "path": "/v1/paper-strategy/proposals",
                    "client_id": "v5.quant_lab_client",
                    "user_agent": "v5-quant-lab-client/1.0",
                    "auth_result": "invalid_bearer_token",
                    "status_code": 401,
                }
            ]
        ),
        generated_at=NOW,
    )

    slo = reports["api_auth_production_slo"].to_dicts()[0]
    incident = reports["api_auth_incident"].to_dicts()[0]
    assert slo["production_auth_slo_status"] == "FAIL"
    assert slo["unexpected_auth_failure_count"] == 1
    assert incident["incident_class"] == "TRUSTED_CLIENT_TOKEN_INVALID"
    assert incident["unexpected_auth_failure"] is True


def test_manual_curl_missing_token_is_operator_error() -> None:
    reports = build_api_auth_reports(
        pl.DataFrame(
            [
                {
                    "request_ts": (NOW - timedelta(minutes=2)).isoformat(),
                    "path": "/v1/health/deep",
                    "client_id": "",
                    "user_agent": "curl/8.7.1",
                    "auth_result": "missing_bearer_token",
                    "status_code": 401,
                }
            ]
        ),
        generated_at=NOW,
    )

    manual = reports["api_auth_manual_probe"].to_dicts()[0]
    assert manual["incident_class"] == "MANUAL_PROBE_MISSING_TOKEN"
    assert manual["operator_error"] is True
    assert manual["affects_production_auth_slo"] is False


def test_proposal_published_after_bundle_has_no_first_seen_time() -> None:
    rows = build_paper_proposal_propagation_status(
        proposals=pl.DataFrame(
            [
                {
                    "proposal_id": "later:1",
                    "proposal_hash": "a" * 64,
                    "proposal_snapshot_id": "snapshot-later",
                    "proposal_snapshot_sha256": "b" * 64,
                    "snapshot_generated_at": NOW.isoformat(),
                }
            ]
        ),
        ack_current=pl.DataFrame(),
        ack_history=pl.DataFrame(),
        trackers_current=pl.DataFrame(),
        trackers_history=pl.DataFrame(),
        selected_v5_bundle_built_at=NOW - timedelta(minutes=1),
        v5_observed_proposal_snapshot_id="snapshot-later",
        v5_observed_proposal_snapshot_sha256="b" * 64,
        generated_at=NOW,
    ).to_dicts()

    assert rows[0]["propagation_status"] == "PUBLISHED_AFTER_SELECTED_BUNDLE"
    assert rows[0]["first_seen_by_v5_at"] == ""


def test_history_ack_cannot_prove_current_snapshot_seen() -> None:
    rows = build_paper_proposal_propagation_status(
        proposals=pl.DataFrame(
            [
                {
                    "proposal_id": "history:1",
                    "proposal_hash": "c" * 64,
                    "proposal_snapshot_id": "snapshot-current",
                    "proposal_snapshot_sha256": "d" * 64,
                    "snapshot_generated_at": (NOW - timedelta(minutes=10)).isoformat(),
                }
            ]
        ),
        ack_current=pl.DataFrame(),
        ack_history=pl.DataFrame(
            [
                {
                    "proposal_id": "history:1",
                    "proposal_hash": "c" * 64,
                    "accepted": True,
                    "accepted_at": (NOW - timedelta(minutes=20)).isoformat(),
                }
            ]
        ),
        trackers_current=pl.DataFrame(),
        trackers_history=pl.DataFrame(),
        selected_v5_bundle_built_at=NOW - timedelta(minutes=1),
        v5_observed_proposal_snapshot_id="snapshot-old",
        v5_observed_proposal_snapshot_sha256="e" * 64,
        generated_at=NOW,
    ).to_dicts()

    assert rows[0]["propagation_status"] == "HISTORICAL_ACK_ONLY"
    assert rows[0]["first_seen_by_v5_at"] == ""


def test_snapshot_id_match_with_sha_mismatch_is_rejected() -> None:
    rows = build_paper_proposal_propagation_status(
        proposals=pl.DataFrame(
            [
                {
                    "proposal_id": "mismatch:1",
                    "proposal_hash": "1" * 64,
                    "proposal_snapshot_id": "snapshot-one",
                    "proposal_snapshot_sha256": "2" * 64,
                    "snapshot_generated_at": (NOW - timedelta(minutes=10)).isoformat(),
                }
            ]
        ),
        ack_current=pl.DataFrame(),
        ack_history=pl.DataFrame(),
        trackers_current=pl.DataFrame(),
        trackers_history=pl.DataFrame(),
        selected_v5_bundle_built_at=NOW - timedelta(minutes=1),
        v5_observed_proposal_snapshot_id="snapshot-one",
        v5_observed_proposal_snapshot_sha256="3" * 64,
        generated_at=NOW,
    ).to_dicts()

    assert rows[0]["propagation_status"] == "HASH_MISMATCH"


def test_complete_acceptance_discloses_all_non_pass_and_funnel_eras() -> None:
    acceptance = pl.DataFrame(
        [
            {
                "check_name": "api_auth_error_rate_ok",
                "status": "FAIL",
                "observed_value": "3.24%",
            },
            {
                "check_name": "lake_growth",
                "status": "WARNING",
                "observed_value": "high",
            },
        ]
    )
    runtime = pl.DataFrame(
        [
            {"check_name": "paper_runtime_heartbeat_fresh", "status": "PASS"},
            {
                "check_name": "paper_trade_event_fresh",
                "status": "NO_NEW_EVENT_EXPECTED",
            },
        ]
    )
    propagation = pl.DataFrame(
        [{"proposal_id": "p", "propagation_status": "ACCEPTED_TRACKER_ACTIVE"}]
    )
    auth = pl.DataFrame([{"current_status": "RECOVERED_24H_CLEAN"}])
    result = build_complete_acceptance_status(
        system_acceptance=acceptance,
        data_quality={
            "status": "WARNING",
            "checks": [{"name": "dq_warn", "status": "WARN", "detail": "x"}],
            "risk_permission": {
                "permission": "ABORT",
                "permission_status": "ACTIVE_ABORT",
            },
            "quant_lab_enforce_readiness": {
                "entry_status": "ENTRY_BLOCKED",
                "scale_status": "SCALE_BLOCKED",
            },
        },
        paper_freshness=runtime,
        cohort=pl.DataFrame([{"cohort_version": 1, "status": "FORMING"}]),
        propagation=propagation,
        auth_incidents=auth,
        acceptance_context={"current_main_production_relationship": "MISMATCH"},
        generated_at=NOW,
    )

    assert {item["name"] for item in result["issues"]} == {
        "api_auth_error_rate_ok",
        "lake_growth",
        "dq_warn",
    }
    assert result["overall_system_health"] == "FAIL"
    assert result["permission_status"] == "ACTIVE_ABORT"
    assert result["current_main_coverage_verdict"] == "BLOCKED_CURRENT_MAIN_NOT_DEPLOYED"
    assert {
        "auth_verdict",
        "proposal_propagation_verdict",
        "expanded_universe_verdict",
        "paper_runtime_verdict",
        "cohort_verdict",
        "economic_evidence_verdict",
        "entry_verdict",
        "scale_verdict",
        "current_main_coverage_verdict",
    }.issubset(result)

    funnel = build_post_fix_funnel_attribution(
        pl.DataFrame(
            [
                {
                    "schema_version": "v5.trade_opportunity_funnel.v1",
                    "run_id": "old",
                    "primary_blocker": "unattributed_stage_loss",
                    "dropped_count": 3,
                },
                {
                    "schema_version": "v5.trade_opportunity_funnel.v2",
                    "run_id": "new",
                    "primary_blocker": "risk_blocked",
                    "dropped_count": 2,
                },
            ]
        )
    )
    assert funnel["pre_fix_unattributed_count"] == 3
    assert funnel["post_fix_unattributed_count"] == 0
    assert funnel["status"] == "PASS"
    assert funnel["historical_rows_rewritten"] is False
