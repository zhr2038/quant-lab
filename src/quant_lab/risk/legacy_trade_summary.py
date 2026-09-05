"""Exact historical trade summary reader, isolated from retired producers."""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from quant_lab.strategy_telemetry.sanitize import safe_json_dumps

TRADE_LEVEL_RISK_SUMMARY_CURRENT_WINDOW_HOURS = 24

def trade_level_risk_summary(
    judgments: pl.DataFrame,
    false_block_audit: pl.DataFrame | None = None,
    opportunity_buckets: pl.DataFrame | None = None,
    bucket_policy: pl.DataFrame | None = None,
) -> dict[str, Any]:
    policy_source = _policy_summary_source(opportunity_buckets, bucket_policy)
    bucket_rows = _micro_canary_review_bucket_rows(policy_source)
    risk_block_bucket_count = _policy_action_count(policy_source, "RISK_BLOCK")
    if judgments.is_empty():
        return {
            "trade_level_decision_summary": safe_json_dumps({}),
            "micro_canary_review_count": 0,
            "micro_canary_review_bucket_count": len(bucket_rows),
            "reviewable_abort_count": 0,
            "micro_canary_review_ready_count": 0,
            "micro_canary_review_blocked_by_observability_count": 0,
            "micro_canary_allow_candidate_count": 0,
            "risk_block_bucket_count": risk_block_bucket_count,
            "recommended_next_permission_mode": _recommended_next_permission_mode(
                0,
                0,
                0,
                len(bucket_rows),
                risk_block_bucket_count,
            ),
            "blocked_by_observability_count": 0,
            "top_micro_canary_review_buckets": safe_json_dumps(bucket_rows[:5]),
            "false_block_rate": 0.0,
        }
    judgment_rows = _current_judgment_rows(judgments)
    decisions = Counter(
        _text(row.get("trade_level_decision")) or "UNKNOWN" for row in judgment_rows
    )
    review_count = decisions.get("MICRO_CANARY_REVIEW", 0) + decisions.get(
        "MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY", 0
    )
    review_ready_count = decisions.get("MICRO_CANARY_REVIEW", 0)
    blocked_by_observability = decisions.get("MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY", 0)
    allow_candidate_count = decisions.get("MICRO_CANARY_ALLOW", 0)
    false_block_rate = 0.0
    audit = false_block_audit if false_block_audit is not None else pl.DataFrame()
    if not audit.is_empty():
        current_event_ids = {
            event_id
            for row in judgment_rows
            if (event_id := _text(row.get("event_id") or row.get("sample_id")))
        }
        rows = [
            row
            for row in audit.to_dicts()
            if row.get("quant_lab_would_block")
            and (
                not current_event_ids
                or _text(row.get("event_id") or row.get("sample_id")) in current_event_ids
            )
        ]
        if rows:
            false_block_rate = sum(1 for row in rows if row.get("false_block")) / len(rows)
    return {
        "trade_level_decision_summary": safe_json_dumps(dict(sorted(decisions.items()))),
        "micro_canary_review_count": int(review_count),
        "micro_canary_review_bucket_count": len(bucket_rows),
        "reviewable_abort_count": int(review_count),
        "micro_canary_review_ready_count": int(review_ready_count),
        "micro_canary_review_blocked_by_observability_count": int(blocked_by_observability),
        "micro_canary_allow_candidate_count": int(allow_candidate_count),
        "risk_block_bucket_count": risk_block_bucket_count,
        "recommended_next_permission_mode": _recommended_next_permission_mode(
            int(review_ready_count),
            int(blocked_by_observability),
            int(allow_candidate_count),
            len(bucket_rows),
            risk_block_bucket_count,
        ),
        "blocked_by_observability_count": int(blocked_by_observability),
        "top_micro_canary_review_buckets": safe_json_dumps(bucket_rows[:5]),
        "false_block_rate": round(float(false_block_rate), 6),
    }

def _current_judgment_rows(judgments: pl.DataFrame) -> list[dict[str, Any]]:
    rows = judgments.to_dicts()
    stamped: list[tuple[dict[str, Any], datetime]] = []
    for row in rows:
        decision_ts = _timestamp(row.get("decision_ts"))
        if decision_ts is not None:
            stamped.append((row, decision_ts))
    if not stamped:
        return rows
    latest = max(ts for _, ts in stamped)
    cutoff = latest - timedelta(hours=TRADE_LEVEL_RISK_SUMMARY_CURRENT_WINDOW_HOURS)
    return [row for row, ts in stamped if ts >= cutoff]

def _micro_canary_review_bucket_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    rows = [
        row
        for row in frame.to_dicts()
        if _policy_action(row) == "MICRO_CANARY_REVIEW"
    ]
    rows.sort(
        key=lambda row: (
            _float(row.get("veto_net_value_bps")) or 0.0,
            -(_int(row.get("sample_count")) or 0),
        )
    )
    return [
        {
            "bucket_key": _text(row.get("bucket_key")),
            "symbol": _text(row.get("symbol")),
            "strategy_candidate": _text(row.get("strategy_candidate")),
            "sample_count": _int(row.get("sample_count")) or 0,
            "false_block_count": _int(row.get("false_block_count")) or 0,
            "loss_saved_count": _int(row.get("loss_saved_count")) or 0,
            "veto_net_value_bps": _float(row.get("veto_net_value_bps")) or 0.0,
            "policy_action": "MICRO_CANARY_REVIEW",
            "policy_reason": _text(row.get("policy_reason")),
            "recommended_trade_level_decision": "MICRO_CANARY_REVIEW",
        }
        for row in rows
    ]

def _policy_summary_source(
    opportunity_buckets: pl.DataFrame | None,
    bucket_policy: pl.DataFrame | None,
) -> pl.DataFrame:
    if bucket_policy is not None and not bucket_policy.is_empty():
        return bucket_policy
    if opportunity_buckets is not None:
        return opportunity_buckets
    return pl.DataFrame()

def _policy_action_count(frame: pl.DataFrame | None, action: str) -> int:
    if frame is None or frame.is_empty():
        return 0
    return sum(1 for row in frame.to_dicts() if _policy_action(row) == action)

def _recommended_next_permission_mode(
    review_ready_count: int,
    blocked_by_observability_count: int,
    allow_candidate_count: int,
    review_bucket_count: int,
    risk_block_bucket_count: int,
) -> str:
    if allow_candidate_count > 0:
        return "MICRO_CANARY_ALLOW_CANDIDATE_REVIEW_REQUIRED"
    if review_ready_count > 0:
        return "MICRO_CANARY_REVIEW_ONLY"
    if blocked_by_observability_count > 0:
        return "MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY"
    if review_bucket_count > 0:
        return "MICRO_CANARY_REVIEW_PENDING_MATCH"
    if risk_block_bucket_count > 0:
        return "RISK_BLOCK_POLICY_ACTIVE"
    return "PAPER_ONLY"

def _policy_action(row: dict[str, Any]) -> str:
    action = _text(row.get("policy_action") or row.get("recommended_trade_level_decision")).upper()
    return action if action in {"RISK_BLOCK", "MICRO_CANARY_REVIEW", "MICRO_CANARY_ALLOW"} else ""

def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value in (None, ""):
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)

def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text

def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None

def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None
