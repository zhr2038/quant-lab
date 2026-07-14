from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

AUTH_ERROR_RESULTS = {"missing_bearer_token", "invalid_bearer_token"}
CURRENT_FUNNEL_SCHEMA_VERSION = "v5.trade_opportunity_funnel.v2"

API_AUTH_INCIDENT_SCHEMA = {
    "endpoint": pl.Utf8,
    "client_id": pl.Utf8,
    "client_host": pl.Utf8,
    "user_agent": pl.Utf8,
    "auth_result": pl.Utf8,
    "first_error_at": pl.Utf8,
    "last_error_at": pl.Utf8,
    "error_count": pl.Int64,
    "recovered_at": pl.Utf8,
    "clean_hours_after_recovery": pl.Float64,
    "current_status": pl.Utf8,
    "suspected_root_cause": pl.Utf8,
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
    parsed: list[tuple[dict[str, Any], datetime]] = []
    cutoff = generated - timedelta(days=max(1, incident_lookback_days))
    for source in rows:
        timestamp = _row_time(source, ("request_ts", "ts_utc", "created_at"))
        if timestamp is None or timestamp < cutoff:
            continue
        parsed.append((source, timestamp))

    success_by_endpoint: dict[tuple[str, str, str, str], list[datetime]] = defaultdict(list)
    errors_by_key: dict[tuple[str, str, str, str, str], list[datetime]] = defaultdict(list)
    for row, timestamp in parsed:
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
            errors_by_key[(*identity, auth_result or "http_401")].append(timestamp)
        elif status_code is not None and 200 <= status_code < 400 and auth_result == "token_ok":
            success_by_endpoint[identity].append(timestamp)

    incident_rows: list[dict[str, Any]] = []
    for key, timestamps in sorted(errors_by_key.items()):
        endpoint, client_id, client_host, user_agent, auth_result = key
        first_error = min(timestamps)
        last_error = max(timestamps)
        recoveries = [
            item
            for item in success_by_endpoint.get((endpoint, client_id, client_host, user_agent), [])
            if item > last_error
        ]
        recovered_at = min(recoveries) if recoveries else None
        clean_hours = max(0.0, (generated - last_error).total_seconds() / 3600.0)
        if clean_hours >= 24.0:
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
    for row, timestamp in parsed:
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

    return {
        "api_auth_incident": _frame(incident_rows, API_AUTH_INCIDENT_SCHEMA),
        "api_auth_error_timeline": _frame(incident_rows, API_AUTH_INCIDENT_SCHEMA),
        "api_auth_client_summary": _frame(client_rows, API_AUTH_CLIENT_SUMMARY_SCHEMA),
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
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = _utc(generated_at)
    generated_text = generated.isoformat()
    ack_rows = _rows(ack_current, ack_history)
    tracker_rows = _rows(trackers_current, trackers_history)
    current_tracker_keys = {
        (_text(row.get("proposal_id")), _text(row.get("proposal_hash")))
        for row in (trackers_current.to_dicts() if not trackers_current.is_empty() else [])
    }
    output: list[dict[str, Any]] = []
    for proposal in proposals.to_dicts() if not proposals.is_empty() else []:
        proposal_id = _text(proposal.get("proposal_id"))
        proposal_hash = _text(proposal.get("proposal_hash"))
        if not proposal_id:
            continue
        proposal_at = _row_time(
            proposal,
            ("published_at", "created_at", "as_of_ts", "generated_at"),
        )
        same_id_acks = [row for row in ack_rows if _text(row.get("proposal_id")) == proposal_id]
        exact_acks = [
            row
            for row in same_id_acks
            if not proposal_hash or _text(row.get("proposal_hash")) == proposal_hash
        ]
        exact_ack = max(exact_acks, key=_propagation_row_rank, default=None)
        same_id_trackers = [
            row for row in tracker_rows if _text(row.get("proposal_id")) == proposal_id
        ]
        exact_trackers = [
            row
            for row in same_id_trackers
            if not proposal_hash or _text(row.get("proposal_hash")) == proposal_hash
        ]
        exact_tracker = max(exact_trackers, key=_propagation_row_rank, default=None)
        ack_at = _row_time(
            exact_ack or {},
            ("accepted_at", "rejected_at", "ack_at", "created_at", "ingest_ts"),
        )
        tracker_at = _row_time(
            exact_tracker or {},
            ("tracker_created_at", "created_at", "updated_at", "ingest_ts"),
        )
        seen_times = [item for item in (ack_at, tracker_at) if item is not None]
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
        status, block_reason = _propagation_state(
            proposal_hash=proposal_hash,
            same_id_acks=same_id_acks,
            same_id_trackers=same_id_trackers,
            exact_ack=exact_ack,
            exact_tracker=exact_tracker,
            accepted=accepted,
            current_tracker=current_tracker,
            reject_reason=reject_reason,
        )
        reached_at = tracker_at or ack_at
        lag = (
            max(0, int((reached_at - proposal_at).total_seconds()))
            if proposal_at is not None and reached_at is not None
            else None
        )
        output.append(
            {
                "proposal_id": proposal_id,
                "proposal_hash": proposal_hash,
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
    auth_ok = all(
        _text(row.get("current_status")) == "RECOVERED_24H_CLEAN"
        for row in (auth_incidents.to_dicts() if not auth_incidents.is_empty() else [])
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
    if not safety_ok:
        scoped_verdict = "BLOCKED_SAFETY"
    elif not auth_ok:
        scoped_verdict = "BLOCKED_API_AUTH"
    elif runtime != "PASS":
        scoped_verdict = "BLOCKED_PAPER_RUNTIME_FRESHNESS"
    elif not propagation_ok:
        scoped_verdict = "BLOCKED_PROPOSAL_PROPAGATION"
    else:
        scoped_verdict = "PASS_OPS_TRUTHFULNESS_AND_PROPAGATION"
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


def _propagation_state(
    *,
    proposal_hash: str,
    same_id_acks: list[dict[str, Any]],
    same_id_trackers: list[dict[str, Any]],
    exact_ack: dict[str, Any] | None,
    exact_tracker: dict[str, Any] | None,
    accepted: bool,
    current_tracker: bool,
    reject_reason: str,
) -> tuple[str, str]:
    mismatched = any(
        proposal_hash and _text(row.get("proposal_hash")) not in {"", proposal_hash}
        for row in [*same_id_acks, *same_id_trackers]
    )
    if mismatched and exact_ack is None and exact_tracker is None:
        return "HASH_MISMATCH", "V5 evidence exists for proposal_id with another hash"
    if accepted and exact_tracker and current_tracker:
        return "ACCEPTED_TRACKER_ACTIVE", ""
    reason = reject_reason.lower()
    if "expired" in reason:
        return "REJECTED_EXPIRED", reject_reason
    if "capacity" in reason:
        return "REJECTED_CAPACITY", reject_reason
    if "hash" in reason or "version_conflict" in reason:
        return "HASH_MISMATCH", reject_reason
    if exact_ack is not None and not accepted:
        return "REJECTED_CONTRACT", reject_reason or "proposal rejected by V5"
    if exact_ack is not None or exact_tracker is not None:
        return "SEEN_PENDING_PARSE", "accepted ACK has no current active tracker"
    if same_id_acks or same_id_trackers:
        return "SUPERSEDED", "only historical or superseded V5 evidence exists"
    return "NOT_YET_SEEN_BY_V5", "no exact ACK or Tracker evidence"


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
