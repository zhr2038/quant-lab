from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

AUTH_ERROR_RESULTS = {"missing_bearer_token", "invalid_bearer_token"}
CURRENT_FUNNEL_SCHEMA_VERSION = "v5.trade_opportunity_funnel.v2"
TRUSTED_PRODUCTION_CLIENT_IDS = {
    "v5.quant_lab_client",
    "v5.dashboard_proxy",
}
EXPECTED_UNAUTHENTICATED_PROBE_IDS = {
    "quant-lab.auth-negative-probe",
    "quant-lab.expected-unauthenticated-probe",
}

API_AUTH_INCIDENT_SCHEMA = {
    "endpoint": pl.Utf8,
    "client_id": pl.Utf8,
    "client_host": pl.Utf8,
    "user_agent": pl.Utf8,
    "auth_result": pl.Utf8,
    "client_class": pl.Utf8,
    "trusted_client": pl.Boolean,
    "expected_unauthenticated_probe": pl.Boolean,
    "incident_class": pl.Utf8,
    "affects_production_auth_slo": pl.Boolean,
    "affects_security_rejection_metric": pl.Boolean,
    "operator_error": pl.Boolean,
    "unexpected_auth_failure": pl.Boolean,
    "first_error_at": pl.Utf8,
    "last_error_at": pl.Utf8,
    "error_count": pl.Int64,
    "recovered_at": pl.Utf8,
    "clean_hours_after_recovery": pl.Float64,
    "current_status": pl.Utf8,
    "suspected_root_cause": pl.Utf8,
    "generated_at": pl.Utf8,
}

API_AUTH_PRODUCTION_SLO_SCHEMA = {
    "window_hours": pl.Int64,
    "trusted_request_count": pl.Int64,
    "unexpected_auth_failure_count": pl.Int64,
    "last_unexpected_auth_failure_at": pl.Utf8,
    "clean_hours_since_last_unexpected_failure": pl.Float64,
    "production_auth_slo_status": pl.Utf8,
    "detail": pl.Utf8,
    "generated_at": pl.Utf8,
}

API_AUTH_CLIENT_SUMMARY_SCHEMA = {
    "client_id": pl.Utf8,
    "client_host": pl.Utf8,
    "user_agent": pl.Utf8,
    "request_count_24h": pl.Int64,
    "auth_error_count_24h": pl.Int64,
    "auth_error_rate_24h": pl.Float64,
    "first_error_at": pl.Utf8,
    "last_error_at": pl.Utf8,
    "recovered_at": pl.Utf8,
    "current_status": pl.Utf8,
    "endpoints": pl.Utf8,
    "generated_at": pl.Utf8,
}

PAPER_RUNTIME_FRESHNESS_SCHEMA = {
    "check_name": pl.Utf8,
    "status": pl.Utf8,
    "latest_at": pl.Utf8,
    "age_seconds": pl.Int64,
    "detail": pl.Utf8,
    "generated_at": pl.Utf8,
}

PAPER_PROPOSAL_PROPAGATION_SCHEMA = {
    "proposal_id": pl.Utf8,
    "proposal_hash": pl.Utf8,
    "proposal_snapshot_id": pl.Utf8,
    "proposal_snapshot_sha256": pl.Utf8,
    "proposal_content_snapshot_id": pl.Utf8,
    "proposal_content_snapshot_sha256": pl.Utf8,
    "snapshot_generated_at": pl.Utf8,
    "selected_v5_bundle_built_at": pl.Utf8,
    "v5_observed_proposal_snapshot_id": pl.Utf8,
    "v5_observed_proposal_snapshot_sha256": pl.Utf8,
    "v5_observed_proposal_content_snapshot_id": pl.Utf8,
    "v5_observed_proposal_content_snapshot_sha256": pl.Utf8,
    "proposal_snapshot_match": pl.Boolean,
    "proposal_content_snapshot_match": pl.Boolean,
    "proposal_published_at": pl.Utf8,
    "first_seen_by_v5_at": pl.Utf8,
    "ack_at": pl.Utf8,
    "tracker_created_at": pl.Utf8,
    "propagation_status": pl.Utf8,
    "propagation_lag_seconds": pl.Int64,
    "exact_block_reason": pl.Utf8,
    "generated_at": pl.Utf8,
}


def build_api_auth_reports(
    metrics: pl.DataFrame,
    *,
    generated_at: datetime | None = None,
    incident_lookback_days: int = 30,
) -> dict[str, pl.DataFrame]:
    generated = _utc(generated_at)
    generated_text = generated.isoformat()
    rows = metrics.to_dicts() if not metrics.is_empty() else []
    parsed: list[tuple[dict[str, Any], datetime, dict[str, Any]]] = []
    cutoff = generated - timedelta(days=max(1, incident_lookback_days))
    for source in rows:
        timestamp = _row_time(source, ("request_ts", "ts_utc", "created_at"))
        if timestamp is None or timestamp < cutoff:
            continue
        parsed.append((source, timestamp, _classify_auth_event(source)))

    success_by_endpoint: dict[tuple[str, str, str, str], list[datetime]] = defaultdict(list)
    errors_by_key: dict[tuple[str, str, str, str, str, str], list[datetime]] = defaultdict(
        list
    )
    classifications_by_error: dict[
        tuple[str, str, str, str, str, str], dict[str, Any]
    ] = {}
    for row, timestamp, classification in parsed:
        endpoint = _text(row.get("path") or row.get("endpoint_path"))
        identity = (
            endpoint,
            _text(row.get("client_id")),
            _text(row.get("client_host")),
            _text(row.get("user_agent")),
        )
        auth_result = _text(row.get("auth_result"))
        status_code = _int(row.get("status_code"))
        if status_code == 401 or auth_result in AUTH_ERROR_RESULTS:
            error_key = (
                *identity,
                auth_result or "http_401",
                _text(classification.get("incident_class")),
            )
            errors_by_key[error_key].append(timestamp)
            classifications_by_error[error_key] = classification
        elif status_code is not None and 200 <= status_code < 400 and auth_result == "token_ok":
            success_by_endpoint[identity].append(timestamp)

    incident_rows: list[dict[str, Any]] = []
    for key, timestamps in sorted(errors_by_key.items()):
        endpoint, client_id, client_host, user_agent, auth_result, _incident_class = key
        classification = classifications_by_error[key]
        first_error = min(timestamps)
        last_error = max(timestamps)
        recoveries = [
            item
            for item in success_by_endpoint.get((endpoint, client_id, client_host, user_agent), [])
            if item > last_error
        ]
        recovered_at = min(recoveries) if recoveries else None
        clean_hours = max(0.0, (generated - last_error).total_seconds() / 3600.0)
        if _bool(classification.get("expected_unauthenticated_probe")):
            current_status = "EXPECTED_REJECTION_PASS"
        elif _bool(classification.get("operator_error")):
            current_status = "OPERATOR_ERROR_RECORDED"
        elif not _bool(classification.get("affects_production_auth_slo")):
            current_status = "SECURITY_REJECTION_RECORDED"
        elif clean_hours >= 24.0:
            current_status = "RECOVERED_24H_CLEAN"
        elif clean_hours <= 0.25 and recovered_at is None:
            current_status = "ACTIVE"
        else:
            current_status = "RECOVERED_AWAITING_24H"
        incident_rows.append(
            {
                "endpoint": endpoint,
                "client_id": client_id,
                "client_host": client_host,
                "user_agent": user_agent,
                "auth_result": auth_result,
                **classification,
                "first_error_at": first_error.isoformat(),
                "last_error_at": last_error.isoformat(),
                "error_count": len(timestamps),
                "recovered_at": recovered_at.isoformat() if recovered_at else "",
                "clean_hours_after_recovery": round(clean_hours, 6),
                "current_status": current_status,
                "suspected_root_cause": _auth_root_cause(
                    auth_result=auth_result,
                    recovered_at=recovered_at,
                    clean_hours=clean_hours,
                ),
                "generated_at": generated_text,
            }
        )

    recent_cutoff = generated - timedelta(hours=24)
    client_groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], datetime]]] = defaultdict(
        list
    )
    for row, timestamp, _classification in parsed:
        if timestamp < recent_cutoff:
            continue
        client_groups[
            (
                _text(row.get("client_id")),
                _text(row.get("client_host")),
                _text(row.get("user_agent")),
            )
        ].append((row, timestamp))

    client_rows: list[dict[str, Any]] = []
    for identity, client_events in sorted(client_groups.items()):
        client_id, client_host, user_agent = identity
        error_events = [
            (row, timestamp)
            for row, timestamp in client_events
            if _int(row.get("status_code")) == 401
            or _text(row.get("auth_result")) in AUTH_ERROR_RESULTS
        ]
        successes = [
            timestamp
            for row, timestamp in client_events
            if (_int(row.get("status_code")) or 999) < 400
            and _text(row.get("auth_result")) == "token_ok"
        ]
        first_error = min((timestamp for _row, timestamp in error_events), default=None)
        last_error = max((timestamp for _row, timestamp in error_events), default=None)
        recovered_at = min(
            (timestamp for timestamp in successes if last_error and timestamp > last_error),
            default=None,
        )
        clean_hours = (
            max(0.0, (generated - last_error).total_seconds() / 3600.0)
            if last_error is not None
            else None
        )
        if last_error is None:
            current_status = "HEALTHY"
        elif clean_hours is not None and clean_hours >= 24.0:
            current_status = "RECOVERED_24H_CLEAN"
        elif clean_hours is not None and clean_hours <= 0.25 and recovered_at is None:
            current_status = "ACTIVE"
        else:
            current_status = "RECOVERED_AWAITING_24H"
        endpoints = sorted(
            {
                _text(row.get("path") or row.get("endpoint_path"))
                for row, _timestamp in client_events
                if _text(row.get("path") or row.get("endpoint_path"))
            }
        )
        request_count = len(client_events)
        error_count = len(error_events)
        client_rows.append(
            {
                "client_id": client_id,
                "client_host": client_host,
                "user_agent": user_agent,
                "request_count_24h": request_count,
                "auth_error_count_24h": error_count,
                "auth_error_rate_24h": error_count / request_count if request_count else 0.0,
                "first_error_at": first_error.isoformat() if first_error else "",
                "last_error_at": last_error.isoformat() if last_error else "",
                "recovered_at": recovered_at.isoformat() if recovered_at else "",
                "current_status": current_status,
                "endpoints": json.dumps(endpoints, separators=(",", ":")),
                "generated_at": generated_text,
            }
        )

    recent_cutoff = generated - timedelta(hours=24)
    trusted_recent = [
        (row, timestamp, classification)
        for row, timestamp, classification in parsed
        if timestamp >= recent_cutoff and _bool(classification.get("trusted_client"))
    ]
    unexpected_failures = [
        timestamp
        for row, timestamp, classification in parsed
        if _bool(classification.get("unexpected_auth_failure"))
        and (
            _int(row.get("status_code")) == 401
            or _text(row.get("auth_result")) in AUTH_ERROR_RESULTS
        )
    ]
    last_unexpected = max(unexpected_failures, default=None)
    recent_unexpected = [item for item in unexpected_failures if item >= recent_cutoff]
    clean_hours = (
        max(0.0, (generated - last_unexpected).total_seconds() / 3600.0)
        if last_unexpected is not None
        else None
    )
    production_slo_status = (
        "FAIL"
        if recent_unexpected
        else "PASS"
        if trusted_recent
        else "NO_TRUSTED_TRAFFIC"
    )
    production_slo = _frame(
        [
            {
                "window_hours": 24,
                "trusted_request_count": len(trusted_recent),
                "unexpected_auth_failure_count": len(recent_unexpected),
                "last_unexpected_auth_failure_at": (
                    last_unexpected.isoformat() if last_unexpected else ""
                ),
                "clean_hours_since_last_unexpected_failure": clean_hours,
                "production_auth_slo_status": production_slo_status,
                "detail": (
                    "trusted production clients have no unexpected 401 in the last 24h"
                    if production_slo_status == "PASS"
                    else "trusted production client had an unexpected 401 in the last 24h"
                    if production_slo_status == "FAIL"
                    else "no trusted production client request was observed in the last 24h"
                ),
                "generated_at": generated_text,
            }
        ],
        API_AUTH_PRODUCTION_SLO_SCHEMA,
    )
    incident_frame = _frame(incident_rows, API_AUTH_INCIDENT_SCHEMA)
    security_rows = [
        row
        for row in incident_rows
        if _bool(row.get("affects_security_rejection_metric"))
    ]
    manual_rows = [row for row in incident_rows if _bool(row.get("operator_error"))]
    return {
        "api_auth_incident": incident_frame,
        "api_auth_error_timeline": incident_frame,
        "api_auth_client_summary": _frame(client_rows, API_AUTH_CLIENT_SUMMARY_SCHEMA),
        "api_auth_production_slo": production_slo,
        "api_auth_security_rejections": _frame(
            security_rows, API_AUTH_INCIDENT_SCHEMA
        ),
        "api_auth_manual_probe": _frame(manual_rows, API_AUTH_INCIDENT_SCHEMA),
    }


def build_paper_runtime_freshness(
    frames: Mapping[str, pl.DataFrame],
    *,
    generated_at: datetime | None = None,
    max_age_seconds: int = 3 * 60 * 60,
) -> pl.DataFrame:
    generated = _utc(generated_at)
    generated_text = generated.isoformat()
    sources = (
        (
            "paper_runtime_heartbeat_fresh",
            frames.get("v5_quant_lab_contract_status", pl.DataFrame()),
            ("generated_at", "ingest_ts"),
        ),
        (
            "paper_registry_fresh",
            _first_non_empty(
                frames,
                "v5_paper_strategy_registry_current",
                "v5_paper_strategy_registry",
            ),
            ("updated_at", "ingest_ts", "created_at"),
        ),
        (
            "paper_state_fresh",
            frames.get("v5_paper_strategy_state", pl.DataFrame()),
            ("updated_at", "ingest_ts"),
        ),
    )
    rows: list[dict[str, Any]] = []
    for check_name, frame, columns in sources:
        latest = _frame_latest_time(frame, columns)
        age = _age_seconds(generated, latest)
        status = "PASS" if age is not None and age <= max_age_seconds else "FAIL"
        rows.append(
            {
                "check_name": check_name,
                "status": status,
                "latest_at": latest.isoformat() if latest else "",
                "age_seconds": age,
                "detail": (
                    f"age_seconds={age};threshold_seconds={max_age_seconds}"
                    if age is not None
                    else "runtime evidence missing"
                ),
                "generated_at": generated_text,
            }
        )

    state = frames.get("v5_paper_strategy_state", pl.DataFrame())
    stale_open = _stale_open_paper_positions(
        state,
        generated=generated,
        max_age_seconds=max_age_seconds,
    )
    for check_name, frame, columns in (
        (
            "paper_signal_event_fresh",
            frames.get("v5_paper_strategy_signal", pl.DataFrame()),
            ("decision_ts", "signal_ts", "ingest_ts"),
        ),
        (
            "paper_trade_event_fresh",
            frames.get("v5_paper_strategy_run", pl.DataFrame()),
            ("closed_at", "ts_utc", "ingest_ts"),
        ),
    ):
        latest = _frame_latest_time(frame, columns)
        age = _age_seconds(generated, latest)
        if age is not None and age <= max_age_seconds:
            status = "PASS"
            detail = f"event_age_seconds={age};threshold_seconds={max_age_seconds}"
        elif stale_open:
            status = "WARNING"
            detail = (
                f"open_or_maturing_paper_position_has_stale_state;tracker_count={len(stale_open)}"
            )
        else:
            status = "NO_NEW_EVENT_EXPECTED"
            detail = "runtime heartbeat is evaluated separately; no new event required"
        rows.append(
            {
                "check_name": check_name,
                "status": status,
                "latest_at": latest.isoformat() if latest else "",
                "age_seconds": age,
                "detail": detail,
                "generated_at": generated_text,
            }
        )
    return _frame(rows, PAPER_RUNTIME_FRESHNESS_SCHEMA)


def paper_runtime_status(frame: pl.DataFrame) -> str:
    statuses = {
        _text(row.get("status")).upper()
        for row in (frame.to_dicts() if not frame.is_empty() else [])
    }
    if not statuses or "FAIL" in statuses:
        return "FAIL"
    if "WARNING" in statuses:
        return "WARNING"
    return "PASS"


def build_paper_proposal_propagation_status(
    *,
    proposals: pl.DataFrame,
    ack_current: pl.DataFrame,
    ack_history: pl.DataFrame,
    trackers_current: pl.DataFrame,
    trackers_history: pl.DataFrame,
    selected_v5_bundle_built_at: datetime | str | None = None,
    v5_observed_proposal_snapshot_id: str | None = None,
    v5_observed_proposal_snapshot_sha256: str | None = None,
    v5_observed_proposal_content_snapshot_id: str | None = None,
    v5_observed_proposal_content_snapshot_sha256: str | None = None,
    v5_snapshot_fetched_at: datetime | str | None = None,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = _utc(generated_at)
    generated_text = generated.isoformat()
    bundle_built_at = _as_utc_datetime(selected_v5_bundle_built_at)
    observed_snapshot_id = _text(v5_observed_proposal_snapshot_id)
    observed_snapshot_sha = _text(v5_observed_proposal_snapshot_sha256).lower()
    observed_content_snapshot_id = _text(
        v5_observed_proposal_content_snapshot_id
    )
    observed_content_snapshot_sha = _text(
        v5_observed_proposal_content_snapshot_sha256
    ).lower()
    observed_fetched_at = _as_utc_datetime(v5_snapshot_fetched_at)
    ack_current_rows = _rows(ack_current)
    ack_history_rows = _rows(ack_history)
    tracker_current_rows = _rows(trackers_current)
    tracker_history_rows = _rows(trackers_history)
    current_tracker_keys = {
        (_text(row.get("proposal_id")), _text(row.get("proposal_hash")))
        for row in (trackers_current.to_dicts() if not trackers_current.is_empty() else [])
    }
    output: list[dict[str, Any]] = []
    for proposal in proposals.to_dicts() if not proposals.is_empty() else []:
        proposal_id = _text(proposal.get("proposal_id"))
        proposal_hash = _text(proposal.get("proposal_hash"))
        snapshot_id = _text(proposal.get("proposal_snapshot_id"))
        snapshot_sha = _text(proposal.get("proposal_snapshot_sha256")).lower()
        content_snapshot_id = _text(
            proposal.get("proposal_content_snapshot_id")
        )
        content_snapshot_sha = _text(
            proposal.get("proposal_content_snapshot_sha256")
        ).lower()
        snapshot_at = _row_time(
            proposal,
            ("snapshot_generated_at", "published_at", "created_at"),
        )
        if not proposal_id:
            continue
        proposal_at = _row_time(
            proposal,
            (
                "snapshot_generated_at",
                "published_at",
                "created_at",
                "as_of_ts",
                "generated_at",
            ),
        )
        current_same_id_acks = [
            row for row in ack_current_rows if _text(row.get("proposal_id")) == proposal_id
        ]
        current_exact_acks = [
            row
            for row in current_same_id_acks
            if not proposal_hash or _text(row.get("proposal_hash")) == proposal_hash
        ]
        current_same_id_trackers = [
            row
            for row in tracker_current_rows
            if _text(row.get("proposal_id")) == proposal_id
        ]
        current_exact_trackers = [
            row
            for row in current_same_id_trackers
            if not proposal_hash or _text(row.get("proposal_hash")) == proposal_hash
        ]
        historical_exact = [
            row
            for row in [*ack_history_rows, *tracker_history_rows]
            if _text(row.get("proposal_id")) == proposal_id
            and (not proposal_hash or _text(row.get("proposal_hash")) == proposal_hash)
        ]
        snapshot_exact = bool(
            snapshot_id
            and snapshot_sha
            and observed_snapshot_id == snapshot_id
            and observed_snapshot_sha == snapshot_sha
        )
        content_snapshot_exact = bool(
            content_snapshot_id
            and content_snapshot_sha
            and observed_content_snapshot_id == content_snapshot_id
            and observed_content_snapshot_sha == content_snapshot_sha
        )
        snapshot_acks = [
            row
            for row in current_exact_acks
            if _row_matches_proposal_snapshot(row, snapshot_id, snapshot_sha)
        ]
        snapshot_trackers = [
            row
            for row in current_exact_trackers
            if _row_matches_proposal_snapshot(row, snapshot_id, snapshot_sha)
        ]
        exact_ack = max(snapshot_acks, key=_propagation_row_rank, default=None)
        exact_tracker = max(snapshot_trackers, key=_propagation_row_rank, default=None)
        ack_at = _row_time(
            exact_ack or {},
            ("accepted_at", "rejected_at", "ack_at", "created_at", "ingest_ts"),
        )
        tracker_at = _row_time(
            exact_tracker or {},
            ("tracker_created_at", "created_at", "updated_at", "ingest_ts"),
        )
        seen_times = [
            item
            for item in (observed_fetched_at, ack_at, tracker_at)
            if item is not None
        ]
        first_seen = min(seen_times) if seen_times else None
        reject_reason = _text((exact_ack or {}).get("reject_reason"))
        accepted = _bool((exact_ack or {}).get("accepted"))
        current_tracker = bool(
            exact_tracker
            and (
                _bool(exact_tracker.get("current_proposal_member"))
                or (
                    _text(exact_tracker.get("proposal_id")),
                    _text(exact_tracker.get("proposal_hash")),
                )
                in current_tracker_keys
            )
        )
        if bundle_built_at is None:
            status, block_reason = "UNKNOWN", "selected V5 bundle built_at is missing"
            first_seen = None
            ack_at = None
            tracker_at = None
        elif snapshot_at is None:
            status, block_reason = (
                "NOT_YET_PUBLISHED_AT_BUNDLE_TIME",
                "Current Proposal snapshot publication time is missing",
            )
            first_seen = None
            ack_at = None
            tracker_at = None
        elif snapshot_at > bundle_built_at:
            status, block_reason = (
                "PUBLISHED_AFTER_SELECTED_BUNDLE",
                "Current Proposal snapshot was published after the selected V5 bundle",
            )
            first_seen = None
            ack_at = None
            tracker_at = None
        elif observed_snapshot_id == snapshot_id and observed_snapshot_sha != snapshot_sha:
            status, block_reason = "HASH_MISMATCH", "V5 observed Snapshot ID with another SHA"
            first_seen = None
            ack_at = None
            tracker_at = None
        elif (
            observed_content_snapshot_id == content_snapshot_id
            and observed_content_snapshot_sha != content_snapshot_sha
        ):
            status, block_reason = (
                "HASH_MISMATCH",
                "V5 observed Proposal Content Snapshot ID with another SHA",
            )
            first_seen = None
            ack_at = None
            tracker_at = None
        elif not snapshot_exact:
            if historical_exact or current_exact_acks or current_exact_trackers:
                status, block_reason = (
                    "HISTORICAL_ACK_ONLY",
                    (
                        "same Proposal content was processed under another publication Snapshot; "
                        "Current publication Snapshot was not observed"
                        if content_snapshot_exact
                        else "only another Snapshot's ACK or Tracker evidence exists; "
                        "Current publication Snapshot was not observed"
                    ),
                )
            else:
                status, block_reason = (
                    "PUBLISHED_NOT_FETCHED",
                    "selected V5 bundle did not record the Current Proposal Snapshot ID and SHA",
                )
            first_seen = None
            ack_at = None
            tracker_at = None
        else:
            status, block_reason = _current_snapshot_propagation_state(
                proposal_hash=proposal_hash,
                current_same_id_acks=current_same_id_acks,
                current_same_id_trackers=current_same_id_trackers,
                exact_ack=exact_ack,
                exact_tracker=exact_tracker,
                accepted=accepted,
                current_tracker=current_tracker,
                reject_reason=reject_reason,
            )
        reached_at = first_seen or tracker_at or ack_at
        lag = (
            max(0, int((reached_at - proposal_at).total_seconds()))
            if proposal_at is not None and reached_at is not None
            else None
        )
        output.append(
            {
                "proposal_id": proposal_id,
                "proposal_hash": proposal_hash,
                "proposal_snapshot_id": snapshot_id,
                "proposal_snapshot_sha256": snapshot_sha,
                "proposal_content_snapshot_id": content_snapshot_id,
                "proposal_content_snapshot_sha256": content_snapshot_sha,
                "snapshot_generated_at": snapshot_at.isoformat() if snapshot_at else "",
                "selected_v5_bundle_built_at": (
                    bundle_built_at.isoformat() if bundle_built_at else ""
                ),
                "v5_observed_proposal_snapshot_id": observed_snapshot_id,
                "v5_observed_proposal_snapshot_sha256": observed_snapshot_sha,
                "v5_observed_proposal_content_snapshot_id": (
                    observed_content_snapshot_id
                ),
                "v5_observed_proposal_content_snapshot_sha256": (
                    observed_content_snapshot_sha
                ),
                "proposal_snapshot_match": snapshot_exact,
                "proposal_content_snapshot_match": content_snapshot_exact,
                "proposal_published_at": proposal_at.isoformat() if proposal_at else "",
                "first_seen_by_v5_at": first_seen.isoformat() if first_seen else "",
                "ack_at": ack_at.isoformat() if ack_at else "",
                "tracker_created_at": tracker_at.isoformat() if tracker_at else "",
                "propagation_status": status,
                "propagation_lag_seconds": lag,
                "exact_block_reason": block_reason,
                "generated_at": generated_text,
            }
        )
    return _frame(output, PAPER_PROPOSAL_PROPAGATION_SCHEMA)


def build_post_fix_funnel_attribution(frame: pl.DataFrame) -> dict[str, Any]:
    rows = frame.to_dicts() if not frame.is_empty() else []
    pre = [row for row in rows if _text(row.get("schema_version")) != CURRENT_FUNNEL_SCHEMA_VERSION]
    post = [
        row for row in rows if _text(row.get("schema_version")) == CURRENT_FUNNEL_SCHEMA_VERSION
    ]
    pre_count = sum(_unattributed_count(row) for row in pre)
    post_count = sum(_unattributed_count(row) for row in post)
    return {
        "current_schema_version": CURRENT_FUNNEL_SCHEMA_VERSION,
        "pre_fix_row_count": len(pre),
        "post_fix_row_count": len(post),
        "pre_fix_run_count": len(
            {_text(row.get("run_id")) for row in pre if _text(row.get("run_id"))}
        ),
        "post_fix_run_count": len(
            {_text(row.get("run_id")) for row in post if _text(row.get("run_id"))}
        ),
        "pre_fix_unattributed_count": pre_count,
        "post_fix_unattributed_count": post_count,
        "target_post_fix_unattributed_count": 0,
        "status": (
            "PASS"
            if post and post_count == 0
            else "PENDING_NEW_RUNTIME_ROWS"
            if not post
            else "FAIL"
        ),
        "historical_rows_rewritten": False,
    }


def build_complete_acceptance_status(
    *,
    system_acceptance: pl.DataFrame,
    data_quality: Mapping[str, Any],
    paper_freshness: pl.DataFrame,
    cohort: pl.DataFrame,
    propagation: pl.DataFrame,
    auth_incidents: pl.DataFrame,
    acceptance_context: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated = _utc(generated_at)
    issues: list[dict[str, Any]] = []
    for row in system_acceptance.to_dicts() if not system_acceptance.is_empty() else []:
        status = _text(row.get("status")).upper()
        if status in {"PASS", "INFO"}:
            continue
        issues.append(_issue("system_acceptance", row, status))
    for row in data_quality.get("checks", []) or []:
        if not isinstance(row, Mapping):
            continue
        status = _text(row.get("status")).upper()
        if status in {"PASS", "INFO", "N/A"}:
            continue
        issues.append(_issue("data_quality", row, status))

    readiness = data_quality.get("quant_lab_enforce_readiness", {}) or {}
    risk = data_quality.get("risk_permission", {}) or {}
    permission = _text(risk.get("permission") or "ABORT").upper()
    permission_status = _text(risk.get("permission_status") or "ACTIVE_ABORT").upper()
    safety_ok = permission in {"ABORT", "SELL_ONLY"} and permission_status in {
        "ACTIVE_ABORT",
        "ACTIVE_SELL_ONLY",
    }
    auth_rows = auth_incidents.to_dicts() if not auth_incidents.is_empty() else []
    auth_ok = bool(auth_rows) and all(
        _text(row.get("production_auth_slo_status")).upper() == "PASS"
        for row in auth_rows
    )
    runtime = paper_runtime_status(paper_freshness)
    propagation_rows = propagation.to_dicts() if not propagation.is_empty() else []
    propagation_ok = bool(propagation_rows) and all(
        _text(row.get("propagation_status")) == "ACCEPTED_TRACKER_ACTIVE"
        for row in propagation_rows
    )
    cohort_rows = cohort.to_dicts() if not cohort.is_empty() else []
    latest_cohort = max(
        cohort_rows,
        key=lambda row: _int(row.get("cohort_version")) or 0,
        default={},
    )
    context = dict(acceptance_context or {})
    expanded_ok = _independent_check_ok(
        system_acceptance,
        tokens=("expanded_universe",),
    )
    economic_ok = _independent_check_ok(
        system_acceptance,
        tokens=("economic", "cost_trust", "paper_evidence"),
    )
    cohort_ok = _bool(latest_cohort.get("formal_observation_eligible")) or _text(
        latest_cohort.get("status")
    ).upper() in {"OBSERVING", "REVIEW_READY"}
    main_relationship = _text(
        context.get("current_main_production_relationship")
    ).upper()
    propagation_states = {
        _text(row.get("propagation_status")) for row in propagation_rows
    }
    snapshot_mismatch_states = {
        "NOT_YET_PUBLISHED_AT_BUNDLE_TIME",
        "PUBLISHED_AFTER_SELECTED_BUNDLE",
        "PUBLISHED_NOT_FETCHED",
        "HASH_MISMATCH",
        "UNKNOWN",
    }
    current_binding_states = {
        "SNAPSHOT_FETCHED_PENDING_PARSE",
        "HISTORICAL_ACK_ONLY",
        "REJECTED_CONTRACT",
        "REJECTED_CAPACITY",
        "SUPERSEDED",
    }
    cohort_binding_ok = bool(
        cohort_ok
        and _bool(latest_cohort.get("snapshot_all_members_matched"))
        and _text(latest_cohort.get("snapshot_mismatch_proposal_ids"))
        in {"", "[]"}
    )
    acceptance_ok = bool(
        context.get("acceptance_set_id")
        and context.get("acceptance_set_matched") is True
        and context.get("formal_acceptance_eligible") is True
    )
    if not safety_ok:
        scoped_verdict = "BLOCKED_SAFETY"
    elif not auth_ok or runtime != "PASS":
        scoped_verdict = "BLOCKED_RUNTIME"
    elif propagation_states.intersection(snapshot_mismatch_states):
        scoped_verdict = "BLOCKED_PROPOSAL_SNAPSHOT_MISMATCH"
    elif propagation_states.intersection(current_binding_states) or not propagation_ok:
        scoped_verdict = "BLOCKED_CURRENT_ACK_TRACKER_SNAPSHOT"
    elif not cohort_binding_ok:
        scoped_verdict = "BLOCKED_COHORT_BINDING"
    elif not acceptance_ok:
        scoped_verdict = "BLOCKED_ACCEPTANCE_SET"
    else:
        scoped_verdict = "PASS_PROPOSAL_SNAPSHOT_AND_COHORT_BINDING"
    statuses = {_text(row.get("status")).upper() for row in issues}
    overall = (
        "FAIL" if statuses.intersection({"FAIL", "BLOCKED"}) else "WARNING" if statuses else "PASS"
    )
    return {
        "generated_at": generated.isoformat(),
        "scoped_verdict": scoped_verdict,
        "overall_system_health": overall,
        "data_quality_status": _text(data_quality.get("status") or "UNKNOWN").upper(),
        "paper_runtime_status": runtime,
        "cohort_status": _text(latest_cohort.get("status") or "MISSING"),
        "auth_verdict": "PASS" if auth_ok else "BLOCKED_API_AUTH",
        "proposal_propagation_verdict": (
            "PASS" if propagation_ok else "BLOCKED_PROPOSAL_PROPAGATION"
        ),
        "expanded_universe_verdict": (
            "PASS"
            if expanded_ok is True
            else "BLOCKED_EXPANDED_UNIVERSE"
            if expanded_ok is False
            else "UNKNOWN"
        ),
        "paper_runtime_verdict": (
            "PASS" if runtime == "PASS" else "BLOCKED_PAPER_RUNTIME"
        ),
        "cohort_verdict": "PASS" if cohort_ok else "BLOCKED_COHORT",
        "acceptance_set_verdict": (
            "PASS" if acceptance_ok else "BLOCKED_ACCEPTANCE_SET"
        ),
        "economic_evidence_verdict": (
            "PASS"
            if economic_ok is True
            else "BLOCKED_ECONOMIC_EVIDENCE"
            if economic_ok is False
            else "UNKNOWN"
        ),
        "entry_verdict": _text(readiness.get("entry_status") or "UNKNOWN"),
        "scale_verdict": _text(readiness.get("scale_status") or "UNKNOWN"),
        "current_main_coverage_verdict": (
            "PASS_CURRENT_MAIN_PRODUCTION_MATCH"
            if main_relationship == "MATCH"
            else "BLOCKED_CURRENT_MAIN_NOT_DEPLOYED"
            if main_relationship == "MISMATCH"
            else "UNKNOWN"
        ),
        "entry_status": _text(readiness.get("entry_status") or "UNKNOWN"),
        "scale_status": _text(readiness.get("scale_status") or "UNKNOWN"),
        "permission": permission,
        "permission_status": permission_status,
        "issue_count": len(issues),
        "system_acceptance_non_pass_count": sum(
            row["source"] == "system_acceptance" for row in issues
        ),
        "data_quality_non_pass_count": sum(row["source"] == "data_quality" for row in issues),
        "issues": issues,
    }


def _independent_check_ok(
    frame: pl.DataFrame,
    *,
    tokens: tuple[str, ...],
) -> bool | None:
    matching = [
        row
        for row in (frame.to_dicts() if not frame.is_empty() else [])
        if any(token in _text(row.get("check_name")).lower() for token in tokens)
    ]
    if not matching:
        return None
    return all(
        _text(row.get("status")).upper() in {"PASS", "INFO"} for row in matching
    )


def _current_snapshot_propagation_state(
    *,
    proposal_hash: str,
    current_same_id_acks: list[dict[str, Any]],
    current_same_id_trackers: list[dict[str, Any]],
    exact_ack: dict[str, Any] | None,
    exact_tracker: dict[str, Any] | None,
    accepted: bool,
    current_tracker: bool,
    reject_reason: str,
) -> tuple[str, str]:
    mismatched = any(
        proposal_hash and _text(row.get("proposal_hash")) not in {"", proposal_hash}
        for row in [*current_same_id_acks, *current_same_id_trackers]
    )
    if mismatched and exact_ack is None and exact_tracker is None:
        return "HASH_MISMATCH", "V5 evidence exists for proposal_id with another hash"
    if accepted and exact_tracker and current_tracker:
        return "ACCEPTED_TRACKER_ACTIVE", ""
    tracker_status = _text((exact_tracker or {}).get("supersession_status")).upper()
    if tracker_status == "SUPERSEDED" or not _bool(
        (exact_tracker or {}).get("current_proposal_member")
    ) and exact_tracker is not None:
        return "SUPERSEDED", "Current Snapshot Tracker is superseded or exit-only"
    reason = reject_reason.lower()
    if "capacity" in reason:
        return "REJECTED_CAPACITY", reject_reason
    if "hash" in reason or "version_conflict" in reason:
        return "HASH_MISMATCH", reject_reason
    if exact_ack is not None and not accepted:
        return "REJECTED_CONTRACT", reject_reason or "proposal rejected by V5"
    if exact_ack is not None or exact_tracker is not None:
        return "SNAPSHOT_FETCHED_PENDING_PARSE", "accepted ACK has no current active tracker"
    return (
        "SNAPSHOT_FETCHED_PENDING_PARSE",
        "V5 fetched the Snapshot but emitted no Snapshot-bound ACK or Tracker result",
    )


def _row_matches_proposal_snapshot(
    row: Mapping[str, Any],
    snapshot_id: str,
    snapshot_sha256: str,
) -> bool:
    return bool(
        snapshot_id
        and snapshot_sha256
        and _text(row.get("source_proposal_snapshot_id")) == snapshot_id
        and _text(row.get("source_proposal_snapshot_sha256")).lower()
        == snapshot_sha256.lower()
    )


def _stale_open_paper_positions(
    frame: pl.DataFrame,
    *,
    generated: datetime,
    max_age_seconds: int,
) -> list[dict[str, Any]]:
    output = []
    for row in frame.to_dicts() if not frame.is_empty() else []:
        open_position = _bool(row.get("open_paper_position")) or _text(
            row.get("state")
        ).upper() in {"PAPER_OPEN", "PAPER_EXIT_PENDING"}
        if not open_position:
            continue
        updated_at = _row_time(row, ("updated_at", "ingest_ts"))
        if updated_at is None or (generated - updated_at).total_seconds() > max_age_seconds:
            output.append(row)
    return output


def _issue(source: str, row: Mapping[str, Any], status: str) -> dict[str, Any]:
    name = _text(row.get("check_name") or row.get("name"))
    detail = _text(row.get("observed_value") or row.get("detail"))
    lowered = f"{name} {detail}".lower()
    paper = any(
        token in lowered for token in ("paper", "proposal", "tracker", "cohort", "api_auth")
    )
    scoped = paper or "api_auth" in lowered or "acceptance" in lowered
    return {
        "source": source,
        "name": name,
        "status": status,
        "detail": detail,
        "affects_scoped_verdict": scoped,
        "affects_paper": paper,
        "affects_canary": status in {"FAIL", "BLOCKED"} or paper,
        "affects_live": status in {"FAIL", "BLOCKED"},
    }


def _auth_root_cause(
    *,
    auth_result: str,
    recovered_at: datetime | None,
    clean_hours: float,
) -> str:
    if auth_result == "missing_bearer_token" and (recovered_at is not None or clean_hours > 0.25):
        return "retired_or_restarted_client_process_missing_token_environment"
    if auth_result == "missing_bearer_token":
        return "client_process_missing_token_environment"
    if auth_result == "invalid_bearer_token":
        return "client_token_mismatch_or_rotation_incomplete"
    return "http_authentication_failure"


def _classify_auth_event(row: Mapping[str, Any]) -> dict[str, Any]:
    client_id = _text(row.get("client_id")).lower()
    user_agent = _text(row.get("user_agent")).lower()
    explicit_class = _text(row.get("client_class")).lower()
    auth_result = _text(row.get("auth_result")).lower()
    expected_probe = _bool(row.get("expected_unauthenticated_probe")) or (
        client_id in EXPECTED_UNAUTHENTICATED_PROBE_IDS
        or explicit_class == "expected_unauthenticated_probe"
        or "auth-negative-probe" in user_agent
    )
    security_scan = (
        explicit_class == "security_scan"
        or "security-scan" in client_id
        or "security-scan" in user_agent
    )
    trusted = _bool(row.get("trusted_client")) or (
        client_id in TRUSTED_PRODUCTION_CLIENT_IDS
        or explicit_class == "trusted_production"
    )
    manual_probe = (
        explicit_class == "manual_probe"
        or client_id in {"manual.curl", "manual.probe", "operator.manual_probe"}
        or user_agent.startswith(("curl/", "httpie/", "postmanruntime/"))
        or "powershell" in user_agent
    )
    if expected_probe:
        client_class = "expected_unauthenticated_probe"
        incident_class = "EXPECTED_UNAUTHENTICATED_PROBE"
    elif security_scan:
        client_class = "security_scan"
        incident_class = "SECURITY_SCAN_REJECTED"
    elif trusted:
        client_class = "trusted_production"
        incident_class = (
            "TRUSTED_CLIENT_TOKEN_INVALID"
            if auth_result == "invalid_bearer_token"
            else "TRUSTED_CLIENT_TOKEN_MISSING"
        )
    elif manual_probe:
        client_class = "manual_probe"
        incident_class = "MANUAL_PROBE_MISSING_TOKEN"
    else:
        client_class = "unknown_external"
        incident_class = "UNKNOWN_EXTERNAL_UNAUTHENTICATED"
    operator_error = manual_probe and not trusted and not expected_probe
    unexpected_auth_failure = trusted and not expected_probe
    return {
        "client_class": client_class,
        "trusted_client": trusted,
        "expected_unauthenticated_probe": expected_probe,
        "incident_class": incident_class,
        "affects_production_auth_slo": unexpected_auth_failure,
        "affects_security_rejection_metric": (
            expected_probe or security_scan or (not trusted and not operator_error)
        ),
        "operator_error": operator_error,
        "unexpected_auth_failure": unexpected_auth_failure,
    }


def _unattributed_count(row: Mapping[str, Any]) -> int:
    total = 0
    if _text(row.get("primary_blocker")) == "unattributed_stage_loss":
        total += _int(row.get("dropped_count")) or 1
    raw = row.get("blocker_mix")
    try:
        values = json.loads(str(raw or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        values = {}
    if isinstance(values, Mapping):
        total = max(total, _int(values.get("unattributed_stage_loss")) or 0)
    return total


def _first_non_empty(frames: Mapping[str, pl.DataFrame], *names: str) -> pl.DataFrame:
    for name in names:
        frame = frames.get(name, pl.DataFrame())
        if not frame.is_empty():
            return frame
    return pl.DataFrame()


def _rows(*frames: pl.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frame in frames:
        for row in frame.to_dicts() if not frame.is_empty() else []:
            key = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            output.append(row)
    return output


def _propagation_row_rank(row: Mapping[str, Any]) -> tuple[datetime, int]:
    return (
        _row_time(
            row,
            (
                "updated_at",
                "accepted_at",
                "rejected_at",
                "created_at",
                "ingest_ts",
            ),
        )
        or datetime.min.replace(tzinfo=UTC),
        int(_bool(row.get("current_proposal_member"))),
    )


def _frame_latest_time(frame: pl.DataFrame, columns: Iterable[str]) -> datetime | None:
    latest = None
    for row in frame.to_dicts() if not frame.is_empty() else []:
        candidate = _row_time(row, columns)
        if candidate is not None and (latest is None or candidate > latest):
            latest = candidate
    return latest


def _row_time(row: Mapping[str, Any], columns: Iterable[str]) -> datetime | None:
    for column in columns:
        value = row.get(column)
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        text = _text(value)
        if not text:
            continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _as_utc_datetime(value: datetime | str | None) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _age_seconds(reference: datetime, value: datetime | None) -> int | None:
    if value is None:
        return None
    return max(0, int((reference - value).total_seconds()))


def _frame(rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=dict(schema))
    normalized = [{column: row.get(column) for column in schema} for row in rows]
    return pl.DataFrame(normalized, schema=dict(schema), orient="row")


def _utc(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    return current.astimezone(UTC) if current.tzinfo else current.replace(tzinfo=UTC)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "on"}
