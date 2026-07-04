from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from quant_lab.strategy_telemetry.sanitize import safe_json_dumps

TRADE_LEVEL_OPPORTUNITY_QUEUE_SCHEMA_VERSION = "trade_level_opportunity_queue.v0.1"

TRADE_LEVEL_OPPORTUNITY_QUEUE_SCHEMA = {
    "schema_version": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "policy_date": pl.Date,
    "expires_at": pl.Datetime(time_zone="UTC"),
    "bucket_key": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "regime": pl.Utf8,
    "risk_level": pl.Utf8,
    "sample_count": pl.Int64,
    "false_block_count": pl.Int64,
    "loss_saved_count": pl.Int64,
    "veto_net_value_bps": pl.Float64,
    "policy_action": pl.Utf8,
    "policy_reason": pl.Utf8,
    "policy_confidence": pl.Utf8,
    "observability_status": pl.Utf8,
    "paper_tracking_status": pl.Utf8,
    "next_action": pl.Utf8,
    "review_ready_count": pl.Int64,
    "blocked_by_observability_count": pl.Int64,
    "micro_canary_allow_candidate_count": pl.Int64,
    "paper_only_count": pl.Int64,
    "risk_block_count": pl.Int64,
    "example_event_ids": pl.Utf8,
    "source": pl.Utf8,
}


def build_trade_level_opportunity_queue(
    bucket_policy: pl.DataFrame,
    judgments: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    created = created_at or datetime.now(UTC)
    if bucket_policy.is_empty():
        return pl.DataFrame(schema=TRADE_LEVEL_OPPORTUNITY_QUEUE_SCHEMA)
    judgments_by_bucket: dict[str, list[dict[str, Any]]] = {}
    if not judgments.is_empty() and "bucket_key" in judgments.columns:
        for row in judgments.to_dicts():
            bucket_key = _text(row.get("bucket_key"))
            if bucket_key:
                judgments_by_bucket.setdefault(bucket_key, []).append(row)
    rows = [
        _queue_row(policy, judgments_by_bucket.get(_text(policy.get("bucket_key")), []), created)
        for policy in bucket_policy.to_dicts()
    ]
    rows.sort(
        key=lambda row: (
            _next_action_priority(row.get("next_action")),
            _policy_priority(row.get("policy_action")),
            -abs(_float(row.get("veto_net_value_bps")) or 0.0),
            -(_int(row.get("sample_count")) or 0),
            _text(row.get("bucket_key")),
        )
    )
    return _frame(rows, TRADE_LEVEL_OPPORTUNITY_QUEUE_SCHEMA)


def _queue_row(
    policy: dict[str, Any],
    judgments: list[dict[str, Any]],
    created: datetime,
) -> dict[str, Any]:
    decisions = [_text(row.get("trade_level_decision")).upper() for row in judgments]
    review_ready_count = decisions.count("MICRO_CANARY_REVIEW")
    blocked_count = decisions.count("MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY")
    allow_candidate_count = decisions.count("MICRO_CANARY_ALLOW")
    paper_only_count = decisions.count("PAPER_ONLY")
    risk_block_count = decisions.count("RISK_BLOCK")
    policy_action = _policy_action(policy)
    observability_status = _observability_status(policy_action, review_ready_count, blocked_count)
    paper_tracking_status = _paper_tracking_status(policy_action, review_ready_count, blocked_count)
    next_action = _next_action(
        policy_action,
        observability_status,
        review_ready_count=review_ready_count,
        allow_candidate_count=allow_candidate_count,
    )
    examples = [
        _text(row.get("event_id"))
        for row in judgments
        if _text(row.get("event_id"))
    ][:5]
    return {
        "schema_version": TRADE_LEVEL_OPPORTUNITY_QUEUE_SCHEMA_VERSION,
        "created_at": created,
        "policy_date": policy.get("policy_date"),
        "expires_at": policy.get("expires_at"),
        "bucket_key": _text(policy.get("bucket_key")),
        "symbol": _text(policy.get("symbol")),
        "strategy_candidate": _text(policy.get("strategy_candidate")),
        "regime": _text(policy.get("regime")),
        "risk_level": _text(policy.get("risk_level")),
        "sample_count": _int(policy.get("sample_count")) or 0,
        "false_block_count": _int(policy.get("false_block_count")) or 0,
        "loss_saved_count": _int(policy.get("loss_saved_count")) or 0,
        "veto_net_value_bps": _float(policy.get("veto_net_value_bps")) or 0.0,
        "policy_action": policy_action,
        "policy_reason": _text(policy.get("policy_reason")),
        "policy_confidence": _text(policy.get("policy_confidence")),
        "observability_status": observability_status,
        "paper_tracking_status": paper_tracking_status,
        "next_action": next_action,
        "review_ready_count": review_ready_count,
        "blocked_by_observability_count": blocked_count,
        "micro_canary_allow_candidate_count": allow_candidate_count,
        "paper_only_count": paper_only_count,
        "risk_block_count": risk_block_count,
        "example_event_ids": safe_json_dumps(examples),
        "source": "quant_lab.trade_level.opportunity_queue",
    }


def _observability_status(
    policy_action: str,
    review_ready_count: int,
    blocked_count: int,
) -> str:
    if blocked_count > 0:
        return "BLOCKED_BY_OBSERVABILITY"
    if review_ready_count > 0:
        return "OBSERVABLE"
    if policy_action == "MICRO_CANARY_REVIEW":
        return "AWAITING_MATCH"
    return "NOT_REQUIRED"


def _paper_tracking_status(
    policy_action: str,
    review_ready_count: int,
    blocked_count: int,
) -> str:
    if policy_action == "PAPER_ONLY":
        return "PAPER_TRACKING_REQUIRED"
    if policy_action == "MICRO_CANARY_REVIEW" and not (review_ready_count or blocked_count):
        return "AWAITING_NEXT_MATCH"
    return "POLICY_EVIDENCE_READY"


def _next_action(
    policy_action: str,
    observability_status: str,
    *,
    review_ready_count: int,
    allow_candidate_count: int,
) -> str:
    if policy_action == "RISK_BLOCK":
        return "KEEP_BLOCKED"
    if policy_action == "MICRO_CANARY_ALLOW" and allow_candidate_count > 0:
        return "MICRO_CANARY_ALLOW_REVIEW_REQUIRED"
    if policy_action == "MICRO_CANARY_REVIEW":
        if observability_status == "BLOCKED_BY_OBSERVABILITY":
            return "BLOCKED_BY_OBSERVABILITY"
        if review_ready_count > 0:
            return "REVIEW_READY"
        return "AWAITING_NEXT_MATCH"
    return "PAPER_TRACKING_REQUIRED"


def _policy_action(row: dict[str, Any]) -> str:
    action = _text(row.get("policy_action")).upper()
    allowed = {"RISK_BLOCK", "PAPER_ONLY", "MICRO_CANARY_REVIEW", "MICRO_CANARY_ALLOW"}
    return action if action in allowed else "PAPER_ONLY"


def _next_action_priority(value: Any) -> int:
    return {
        "BLOCKED_BY_OBSERVABILITY": 0,
        "REVIEW_READY": 1,
        "MICRO_CANARY_ALLOW_REVIEW_REQUIRED": 2,
        "KEEP_BLOCKED": 3,
        "AWAITING_NEXT_MATCH": 4,
        "PAPER_TRACKING_REQUIRED": 5,
    }.get(_text(value).upper(), 9)


def _policy_priority(value: Any) -> int:
    return {
        "MICRO_CANARY_REVIEW": 0,
        "MICRO_CANARY_ALLOW": 1,
        "RISK_BLOCK": 2,
        "PAPER_ONLY": 3,
    }.get(_text(value).upper(), 9)


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row").select(
        [pl.col(name).cast(dtype, strict=False).alias(name) for name, dtype in schema.items()]
    )


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
