from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset, write_snapshot_meta
from quant_lab.opportunity_cost.ledger import (
    DECISION_REGRET_SCHEMA_VERSION,
    OPPORTUNITY_COST_BY_BUCKET_SCHEMA_VERSION,
    OPPORTUNITY_COST_DAILY_SCHEMA_VERSION,
    OPPORTUNITY_COST_EVENT_SCHEMA_VERSION,
    build_opportunity_cost_frames,
    opportunity_bucket_key,
)
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol
from quant_lab.trade_learning.attribution import (
    V5_TRADE_OUTCOME_ATTRIBUTION_SCHEMA_VERSION,
    build_v5_trade_outcome_attribution,
)
from quant_lab.trade_learning.samples import (
    V5_TRADE_LEARNING_SAMPLE_SCHEMA_VERSION,
    build_v5_trade_learning_samples,
)
from quant_lab.trade_level.bucket_policy import (
    TRADE_LEVEL_BUCKET_POLICY_SCHEMA_VERSION,
    build_trade_level_bucket_policy,
)
from quant_lab.trade_level.opportunity_queue import (
    TRADE_LEVEL_OPPORTUNITY_QUEUE_SCHEMA_VERSION,
    build_trade_level_opportunity_queue,
)

TRADE_LEVEL_SCHEMA_VERSION = "trade_level_judgment.v0.3"
TRADE_OPPORTUNITY_EVENT_SCHEMA_VERSION = "trade_opportunity_event.v0.2"
FALSE_BLOCK_AUDIT_SCHEMA_VERSION = "quant_lab_false_block_audit.v0.1"
TRADE_LEVEL_RISK_SUMMARY_CURRENT_WINDOW_HOURS = 24

TRADE_OPPORTUNITY_EVENT_DATASET = Path("gold") / "trade_opportunity_event"
TRADE_OPPORTUNITY_LABEL_DATASET = Path("gold") / "trade_opportunity_label"
TRADE_LEVEL_SIMILARITY_DATASET = Path("gold") / "trade_level_similarity_outcome"
TRADE_LEVEL_JUDGMENT_DATASET = Path("gold") / "trade_level_judgment"
TRADE_LEVEL_BUCKET_POLICY_DATASET = Path("gold") / "trade_level_bucket_policy"
TRADE_LEVEL_OPPORTUNITY_QUEUE_DATASET = Path("gold") / "trade_level_opportunity_queue"
FALSE_BLOCK_AUDIT_DATASET = Path("gold") / "quant_lab_false_block_audit"
V5_TRADE_LEARNING_SAMPLE_DATASET = Path("gold") / "v5_trade_learning_sample"
V5_TRADE_OUTCOME_ATTRIBUTION_DATASET = Path("gold") / "v5_trade_outcome_attribution"
OPPORTUNITY_COST_EVENT_DATASET = Path("gold") / "quant_lab_opportunity_cost_event"
OPPORTUNITY_COST_DAILY_DATASET = Path("gold") / "quant_lab_opportunity_cost_daily"
OPPORTUNITY_COST_BY_BUCKET_DATASET = Path("gold") / "opportunity_cost_by_bucket"
DECISION_REGRET_DATASET = Path("gold") / "quant_lab_decision_regret"

V5_CANDIDATE_EVENT_DATASET = Path("silver") / "v5_candidate_event"
V5_TRADE_EVENT_DATASET = Path("silver") / "v5_trade_event"
V5_ROUNDTRIP_DATASET = Path("silver") / "v5_roundtrip"
V5_ORDER_LIFECYCLE_DATASET = Path("silver") / "v5_order_lifecycle"
V5_CANDIDATE_LABEL_DATASET = Path("gold") / "v5_candidate_label"
RISK_PERMISSION_DATASET = Path("gold") / "risk_permission"

TRADE_OPPORTUNITY_EVENT_SCHEMA = {
    "schema_version": pl.Utf8,
    "event_id": pl.Utf8,
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "run_id": pl.Utf8,
    "candidate_id": pl.Utf8,
    "symbol": pl.Utf8,
    "side": pl.Utf8,
    "intent": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "v5_final_score": pl.Float64,
    "rank": pl.Int64,
    "alpha6_score": pl.Float64,
    "alpha6_side": pl.Utf8,
    "expected_edge_bps": pl.Float64,
    "required_edge_bps": pl.Float64,
    "edge_required_ratio": pl.Float64,
    "cost_bps": pl.Float64,
    "selected_cost_bps": pl.Float64,
    "actual_all_in_cost_bps": pl.Float64,
    "cost_source": pl.Utf8,
    "cost_gate_verified": pl.Boolean,
    "would_block_by_cost": pl.Boolean,
    "risk_level": pl.Utf8,
    "regime": pl.Utf8,
    "arrival_mid": pl.Float64,
    "quote_ts": pl.Utf8,
    "quote_age_ms": pl.Float64,
    "quote_source": pl.Utf8,
    "arrival_spread_bps": pl.Float64,
    "target_weight_after_risk": pl.Float64,
    "quant_lab_permission": pl.Utf8,
    "quant_lab_permission_status": pl.Utf8,
    "quant_lab_live_block_reasons": pl.Utf8,
    "allowed_live_modes": pl.Utf8,
    "v5_would_open": pl.Boolean,
    "actual_submitted": pl.Boolean,
    "source_bundle_sha256": pl.Utf8,
    "source_path_inside_bundle": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

TRADE_LEVEL_JUDGMENT_SCHEMA = {
    "schema_version": pl.Utf8,
    "event_id": pl.Utf8,
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "side": pl.Utf8,
    "intent": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "hard_safety_veto": pl.Boolean,
    "risk_permission_veto": pl.Boolean,
    "strategy_advisory_veto": pl.Boolean,
    "v5_high_confidence_opportunity": pl.Boolean,
    "similar_sample_count": pl.Int64,
    "similar_median_after_cost_bps": pl.Float64,
    "similar_p25_after_cost_bps": pl.Float64,
    "recent_7d_similar_mean": pl.Float64,
    "bucket_key": pl.Utf8,
    "bucket_policy_action": pl.Utf8,
    "bucket_policy_reason": pl.Utf8,
    "bucket_policy_confidence": pl.Utf8,
    "trade_level_decision": pl.Utf8,
    "max_single_order_usdt": pl.Float64,
    "daily_trade_limit": pl.Int64,
    "hard_safety_reasons": pl.Utf8,
    "risk_permission_reasons": pl.Utf8,
    "reason": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

FALSE_BLOCK_AUDIT_SCHEMA = {
    "schema_version": pl.Utf8,
    "sample_id": pl.Utf8,
    "event_id": pl.Utf8,
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "v5_would_open": pl.Boolean,
    "quant_lab_would_block": pl.Boolean,
    "trade_level_decision": pl.Utf8,
    "actual_or_counterfactual_after_cost_bps": pl.Float64,
    "label_horizon_hours": pl.Int64,
    "was_profitable": pl.Boolean,
    "false_block": pl.Boolean,
    "missed_profit_bps": pl.Float64,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

HARD_SAFETY_REASON_CODES = {
    "api_auth_fail",
    "api_auth_failed",
    "auth_fail",
    "auth_failed",
    "exchange_local_mismatch",
    "exchange_position_mismatch",
    "kill_switch",
    "kill_switch_on",
    "ledger_fail",
    "ledger_failed",
    "market_critical",
    "market_data_critical",
    "market_data_stale",
    "open_exposure_mismatch",
    "order_over_limit",
    "position_exposure_mismatch",
    "reconcile_fail",
    "reconcile_failed",
    "stale_market_data",
    "unmanaged_exposure",
    "unmanaged_position",
}
REVIEWABLE_ABORT_REASON_CODES = {
    "actual_or_mixed_cost_coverage_live_universe",
    "actual_or_mixed_cost_coverage_research_universe",
    "advisory_permission_not_allow",
    "baseline_not_global_strategy_gate",
    "cost_coverage_low",
    "cost_health_high_fallback",
    "cost_health_missing_or_critical",
    "no_paper_ready",
    "no_strategy_live_small_ready",
    "quant_lab_advisory_permission_not_allow",
    "quant_lab_live_command_not_allowed",
    "v5_local_live_not_controlled_by_quant_lab",
}


class TradeLevelBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    trade_opportunity_event_rows: int = Field(ge=0)
    trade_opportunity_label_rows: int = Field(ge=0)
    trade_level_similarity_rows: int = Field(ge=0)
    trade_level_judgment_rows: int = Field(ge=0)
    false_block_audit_rows: int = Field(ge=0)
    v5_trade_learning_sample_rows: int = Field(ge=0)
    v5_trade_outcome_attribution_rows: int = Field(ge=0)
    opportunity_cost_event_rows: int = Field(ge=0)
    opportunity_cost_daily_rows: int = Field(ge=0)
    opportunity_cost_by_bucket_rows: int = Field(ge=0)
    trade_level_bucket_policy_rows: int = Field(ge=0)
    trade_level_opportunity_queue_rows: int = Field(ge=0)
    decision_regret_rows: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_trade_level_judgment(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> TradeLevelBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    generated_at = datetime.now(UTC)
    frames = build_trade_level_frames_from_sources(
        candidate_events=read_parquet_dataset(root / V5_CANDIDATE_EVENT_DATASET),
        candidate_labels=read_parquet_dataset(root / V5_CANDIDATE_LABEL_DATASET),
        risk_permissions=read_parquet_dataset(root / RISK_PERMISSION_DATASET),
        v5_trades=read_parquet_dataset(root / V5_TRADE_EVENT_DATASET),
        v5_roundtrips=read_parquet_dataset(root / V5_ROUNDTRIP_DATASET),
        order_lifecycles=read_parquet_dataset(root / V5_ORDER_LIFECYCLE_DATASET),
        as_of_date=day,
        created_at=generated_at,
    )
    warnings: list[str] = []
    for dataset_name, relative_path in [
        ("trade_opportunity_event", TRADE_OPPORTUNITY_EVENT_DATASET),
        ("trade_opportunity_label", TRADE_OPPORTUNITY_LABEL_DATASET),
        ("trade_level_similarity_outcome", TRADE_LEVEL_SIMILARITY_DATASET),
        ("trade_level_judgment", TRADE_LEVEL_JUDGMENT_DATASET),
        ("quant_lab_false_block_audit", FALSE_BLOCK_AUDIT_DATASET),
        ("v5_trade_learning_sample", V5_TRADE_LEARNING_SAMPLE_DATASET),
        ("v5_trade_outcome_attribution", V5_TRADE_OUTCOME_ATTRIBUTION_DATASET),
        ("quant_lab_opportunity_cost_event", OPPORTUNITY_COST_EVENT_DATASET),
        ("quant_lab_opportunity_cost_daily", OPPORTUNITY_COST_DAILY_DATASET),
        ("opportunity_cost_by_bucket", OPPORTUNITY_COST_BY_BUCKET_DATASET),
        ("trade_level_bucket_policy", TRADE_LEVEL_BUCKET_POLICY_DATASET),
        ("trade_level_opportunity_queue", TRADE_LEVEL_OPPORTUNITY_QUEUE_DATASET),
        ("quant_lab_decision_regret", DECISION_REGRET_DATASET),
    ]:
        frame = frames[dataset_name]
        dataset_path = root / relative_path
        write_parquet_dataset(frame, dataset_path)
        write_snapshot_meta(
            dataset_path,
            dataset_name=dataset_name,
            frame=frame,
            schema_version=_schema_version_for_dataset(dataset_name),
            generated_at=generated_at,
        )
    if frames["trade_opportunity_event"].is_empty():
        warnings.append("v5_candidate_event_empty")
    return TradeLevelBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        trade_opportunity_event_rows=frames["trade_opportunity_event"].height,
        trade_opportunity_label_rows=frames["trade_opportunity_label"].height,
        trade_level_similarity_rows=frames["trade_level_similarity_outcome"].height,
        trade_level_judgment_rows=frames["trade_level_judgment"].height,
        false_block_audit_rows=frames["quant_lab_false_block_audit"].height,
        v5_trade_learning_sample_rows=frames["v5_trade_learning_sample"].height,
        v5_trade_outcome_attribution_rows=frames["v5_trade_outcome_attribution"].height,
        opportunity_cost_event_rows=frames["quant_lab_opportunity_cost_event"].height,
        opportunity_cost_daily_rows=frames["quant_lab_opportunity_cost_daily"].height,
        opportunity_cost_by_bucket_rows=frames["opportunity_cost_by_bucket"].height,
        trade_level_bucket_policy_rows=frames["trade_level_bucket_policy"].height,
        trade_level_opportunity_queue_rows=frames["trade_level_opportunity_queue"].height,
        decision_regret_rows=frames["quant_lab_decision_regret"].height,
        warnings=warnings,
    )


def build_trade_level_frames_from_sources(
    *,
    candidate_events: pl.DataFrame,
    candidate_labels: pl.DataFrame,
    risk_permissions: pl.DataFrame,
    v5_trades: pl.DataFrame,
    v5_roundtrips: pl.DataFrame | None = None,
    order_lifecycles: pl.DataFrame | None = None,
    as_of_date: str | date | None = None,
    created_at: datetime | None = None,
) -> dict[str, pl.DataFrame]:
    from quant_lab.trade_level.labels import build_trade_opportunity_labels
    from quant_lab.trade_level.similarity import build_trade_level_similarity_outcome

    generated_at = created_at or datetime.now(UTC)
    day = _parse_day(as_of_date) if as_of_date is not None else generated_at.date()
    events = build_trade_opportunity_events(
        candidate_events,
        risk_permissions=risk_permissions,
        v5_trades=v5_trades,
        order_lifecycles=order_lifecycles if order_lifecycles is not None else pl.DataFrame(),
        created_at=generated_at,
    )
    labels = build_trade_opportunity_labels(
        events,
        candidate_labels,
        created_at=generated_at,
    )
    similarity = build_trade_level_similarity_outcome(events, labels, created_at=generated_at)
    judgments = build_trade_level_judgments(
        events,
        similarity=similarity,
        created_at=generated_at,
    )
    samples = build_v5_trade_learning_samples(
        events,
        labels,
        judgments,
        v5_trades=v5_trades,
        v5_roundtrips=v5_roundtrips if v5_roundtrips is not None else pl.DataFrame(),
        order_lifecycles=order_lifecycles if order_lifecycles is not None else pl.DataFrame(),
        created_at=generated_at,
    )
    audit = build_false_block_audit(
        events,
        labels,
        judgments,
        samples=samples,
        created_at=generated_at,
    )
    attribution = build_v5_trade_outcome_attribution(samples, created_at=generated_at)
    opportunity_cost = build_opportunity_cost_frames(
        events,
        labels,
        judgments,
        samples=samples,
        created_at=generated_at,
    )
    bucket_policy = build_trade_level_bucket_policy(
        opportunity_cost_events=opportunity_cost["quant_lab_opportunity_cost_event"],
        opportunity_cost_by_bucket=opportunity_cost["opportunity_cost_by_bucket"],
        policy_date=day,
        created_at=generated_at,
    )
    judgments = build_trade_level_judgments(
        events,
        similarity=similarity,
        bucket_policy=bucket_policy,
        created_at=generated_at,
    )
    samples = build_v5_trade_learning_samples(
        events,
        labels,
        judgments,
        v5_trades=v5_trades,
        v5_roundtrips=v5_roundtrips if v5_roundtrips is not None else pl.DataFrame(),
        order_lifecycles=order_lifecycles if order_lifecycles is not None else pl.DataFrame(),
        created_at=generated_at,
    )
    audit = build_false_block_audit(
        events,
        labels,
        judgments,
        samples=samples,
        created_at=generated_at,
    )
    attribution = build_v5_trade_outcome_attribution(samples, created_at=generated_at)
    opportunity_cost = build_opportunity_cost_frames(
        events,
        labels,
        judgments,
        samples=samples,
        created_at=generated_at,
    )
    opportunity_queue = build_trade_level_opportunity_queue(
        bucket_policy,
        judgments,
        created_at=generated_at,
    )
    return {
        "trade_opportunity_event": events,
        "trade_opportunity_label": labels,
        "trade_level_similarity_outcome": similarity,
        "trade_level_judgment": judgments,
        "quant_lab_false_block_audit": audit,
        "v5_trade_learning_sample": samples,
        "v5_trade_outcome_attribution": attribution,
        "trade_level_bucket_policy": bucket_policy,
        "trade_level_opportunity_queue": opportunity_queue,
        **opportunity_cost,
    }


def build_trade_opportunity_events(
    candidate_events: pl.DataFrame,
    *,
    risk_permissions: pl.DataFrame | None = None,
    v5_trades: pl.DataFrame | None = None,
    order_lifecycles: pl.DataFrame | None = None,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    if candidate_events.is_empty():
        return pl.DataFrame(schema=TRADE_OPPORTUNITY_EVENT_SCHEMA)
    created = created_at or datetime.now(UTC)
    risk_frame = risk_permissions if risk_permissions is not None else pl.DataFrame()
    trade_frame = v5_trades if v5_trades is not None else pl.DataFrame()
    lifecycle_frame = order_lifecycles if order_lifecycles is not None else pl.DataFrame()
    risk_row = _latest_risk_permission_row(risk_frame)
    actual_lookup = _actual_submission_lookup(trade_frame, lifecycle_frame)
    rows: list[dict[str, Any]] = []
    for raw in candidate_events.to_dicts():
        payload = _payload(raw)
        row = _candidate_event_row(raw, payload, risk_row, actual_lookup, created)
        rows.append(row)
    return _frame(rows, TRADE_OPPORTUNITY_EVENT_SCHEMA)


def build_trade_level_judgments(
    events: pl.DataFrame,
    *,
    similarity: pl.DataFrame | None = None,
    bucket_policy: pl.DataFrame | None = None,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(schema=TRADE_LEVEL_JUDGMENT_SCHEMA)
    created = created_at or datetime.now(UTC)
    similarity_frame = similarity if similarity is not None else pl.DataFrame()
    similarity_by_event = {
        str(row.get("event_id") or ""): row for row in similarity_frame.to_dicts()
    }
    bucket_policy_by_key = _active_bucket_policy_by_key(
        bucket_policy if bucket_policy is not None else pl.DataFrame(),
        created,
    )
    rows = [
        _judgment_row(
            event,
            similarity_by_event.get(str(event.get("event_id") or "")),
            bucket_policy_by_key.get(opportunity_bucket_key(event)),
            created,
        )
        for event in events.to_dicts()
    ]
    return _frame(rows, TRADE_LEVEL_JUDGMENT_SCHEMA)


def build_false_block_audit(
    events: pl.DataFrame,
    labels: pl.DataFrame,
    judgments: pl.DataFrame,
    *,
    samples: pl.DataFrame | None = None,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    if events.is_empty() or judgments.is_empty():
        return pl.DataFrame(schema=FALSE_BLOCK_AUDIT_SCHEMA)
    created = created_at or datetime.now(UTC)
    events_by_id = {str(row.get("event_id") or ""): row for row in events.to_dicts()}
    labels_by_id = {str(row.get("event_id") or ""): row for row in labels.to_dicts()}
    samples_by_id = {
        str(row.get("event_id") or ""): row
        for row in (samples if samples is not None else pl.DataFrame()).to_dicts()
    }
    rows: list[dict[str, Any]] = []
    for judgment in judgments.to_dicts():
        event_id = str(judgment.get("event_id") or "")
        event = events_by_id.get(event_id, {})
        label = labels_by_id.get(event_id, {})
        sample = samples_by_id.get(event_id, {})
        value, horizon = _sample_or_label_value(sample, label)
        decision = _text(judgment.get("trade_level_decision"))
        quant_lab_would_block = decision not in {"MICRO_CANARY_ALLOW", "LIVE_SMALL_ALLOW"}
        was_profitable = value is not None and value > 0.0
        false_block = bool(event.get("v5_would_open")) and quant_lab_would_block and was_profitable
        rows.append(
            {
                "schema_version": FALSE_BLOCK_AUDIT_SCHEMA_VERSION,
                "sample_id": event_id,
                "event_id": event_id,
                "decision_ts": _timestamp(event.get("decision_ts")),
                "symbol": _text(event.get("symbol")),
                "strategy_candidate": _text(event.get("strategy_candidate")),
                "v5_would_open": bool(event.get("v5_would_open")),
                "quant_lab_would_block": quant_lab_would_block,
                "trade_level_decision": decision,
                "actual_or_counterfactual_after_cost_bps": value,
                "label_horizon_hours": horizon,
                "was_profitable": was_profitable if value is not None else None,
                "false_block": false_block,
                "missed_profit_bps": value if false_block else 0.0,
                "created_at": created,
                "source": "quant_lab.trade_level.false_block_audit",
            }
        )
    return _frame(rows, FALSE_BLOCK_AUDIT_SCHEMA)


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


def event_id_for_row(row: Mapping[str, Any]) -> str:
    parts = [
        _text(row.get("candidate_id")),
        _text(row.get("run_id")),
        _iso(_timestamp(row.get("decision_ts") or row.get("ts_utc") or row.get("ts"))),
        _symbol(row.get("symbol")),
        _text(row.get("strategy_candidate")),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"v5trade-{digest}"


def _candidate_event_row(
    raw: dict[str, Any],
    payload: dict[str, Any],
    risk_row: dict[str, Any],
    actual_lookup: set[tuple[str, str]],
    created: datetime,
) -> dict[str, Any]:
    decision_ts = _timestamp(_first(raw, payload, "decision_ts", "ts_utc", "ts", "timestamp"))
    symbol = _symbol(_first(raw, payload, "symbol", "instrument", "instId"))
    run_id = _text(_first(raw, payload, "run_id"))
    candidate_id = _text(_first(raw, payload, "candidate_id"))
    strategy = _text(_first(raw, payload, "strategy_candidate", "candidate_name", "candidate"))
    alpha6_side = _text(_first(raw, payload, "alpha6_side", "signal_side")).lower()
    side = _text(_first(raw, payload, "side", "order_side", "signal_side")) or alpha6_side
    side = side.lower()
    intent = _text(_first(raw, payload, "intent", "action", "final_decision")).upper()
    rank = _int(_first(raw, payload, "rank", "selected_rank"))
    expected = _float(_first(raw, payload, "expected_edge_bps"))
    required = _float(_first(raw, payload, "required_edge_bps"))
    edge_ratio = (
        expected / required if expected is not None and required not in (None, 0.0) else None
    )
    selected_cost = _float(_first(raw, payload, "selected_cost_bps", "cost_bps", "cost"))
    actual_cost = _float(_first(raw, payload, "actual_all_in_cost_bps", "actual_cost_bps"))
    arrival_mid = _float(
        _first(
            raw,
            payload,
            "arrival_mid",
            "arrival_mid_px",
            "mid_px_at_decision",
            "entry_reference_px",
        )
    )
    event = {
        "schema_version": TRADE_OPPORTUNITY_EVENT_SCHEMA_VERSION,
        "event_id": "",
        "decision_ts": decision_ts,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "symbol": symbol,
        "side": side,
        "intent": intent,
        "strategy_candidate": strategy,
        "v5_final_score": _float(_first(raw, payload, "v5_final_score", "final_score")),
        "rank": rank,
        "alpha6_score": _float(_first(raw, payload, "alpha6_score")),
        "alpha6_side": alpha6_side,
        "expected_edge_bps": expected,
        "required_edge_bps": required,
        "edge_required_ratio": edge_ratio,
        "cost_bps": selected_cost,
        "selected_cost_bps": selected_cost,
        "actual_all_in_cost_bps": actual_cost,
        "cost_source": _text(
            _first(raw, payload, "cost_source", "selected_cost_source", "actual_cost_source")
        ),
        "cost_gate_verified": _bool(_first(raw, payload, "cost_gate_verified")),
        "would_block_by_cost": _bool(_first(raw, payload, "would_block_by_cost")),
        "risk_level": _text(_first(raw, payload, "risk_level", "risk_state")),
        "regime": _text(_first(raw, payload, "regime", "regime_state")),
        "arrival_mid": arrival_mid,
        "quote_ts": _text(_first(raw, payload, "quote_ts", "arrival_quote_ts", "book_ts")),
        "quote_age_ms": _float(
            _first(raw, payload, "quote_age_ms", "arrival_quote_age_ms", "book_age_ms")
        ),
        "quote_source": _text(
            _first(raw, payload, "quote_source", "arrival_quote_source", "book_source")
        ),
        "arrival_spread_bps": _float(_first(raw, payload, "arrival_spread_bps", "spread_bps")),
        "target_weight_after_risk": _float(_first(raw, payload, "target_weight_after_risk")),
        "quant_lab_permission": _text(risk_row.get("permission")),
        "quant_lab_permission_status": _text(risk_row.get("permission_status")),
        "quant_lab_live_block_reasons": _json_list_text(risk_row.get("live_block_reasons")),
        "allowed_live_modes": _json_list_text(risk_row.get("allowed_live_modes")),
        "v5_would_open": _v5_would_open(raw, payload),
        "actual_submitted": (run_id, symbol) in actual_lookup,
        "source_bundle_sha256": _text(
            _first(raw, payload, "source_event_bundle_sha256", "bundle_sha256")
        ),
        "source_path_inside_bundle": _text(_first(raw, payload, "source_path_inside_bundle")),
        "created_at": created,
        "source": "quant_lab.trade_level.event",
    }
    event["event_id"] = event_id_for_row(event)
    return event


def _judgment_row(
    event: dict[str, Any],
    similarity: dict[str, Any] | None,
    bucket_policy: dict[str, Any] | None,
    created: datetime,
) -> dict[str, Any]:
    hard_reasons = _hard_safety_reasons(event)
    permission_reasons = _risk_reasons(event)
    hard = bool(hard_reasons)
    risk_veto = _risk_permission_veto(event)
    strategy_veto = _strategy_advisory_veto(event)
    high_confidence, high_reasons = _v5_high_confidence(event, hard_reasons)
    observability_reasons = _observability_reasons(event)
    similar_count = _int((similarity or {}).get("similar_sample_count")) or 0
    similar_median = _float((similarity or {}).get("similar_median_after_cost_bps"))
    similar_p25 = _float((similarity or {}).get("similar_p25_after_cost_bps"))
    recent_7d = _float((similarity or {}).get("recent_7d_similar_mean"))
    similar_ok = (
        similar_count >= 20
        and similar_median is not None
        and similar_median > 0.0
        and similar_p25 is not None
        and similar_p25 > -30.0
        and recent_7d is not None
        and recent_7d >= 0.0
    )
    paper_ready = _paper_ready(event)
    bucket_key = opportunity_bucket_key(event)
    policy_action = _policy_action(bucket_policy or {})
    policy_reason = _text((bucket_policy or {}).get("policy_reason"))
    policy_confidence = _text((bucket_policy or {}).get("policy_confidence"))
    reasons: list[str] = []
    if hard:
        decision = "HARD_BLOCK"
        reasons.extend(hard_reasons)
        max_order = 0.0
        daily_limit = 0
    elif risk_veto and not _risk_reasons_reviewable(permission_reasons):
        decision = "RISK_BLOCK"
        reasons.append("non_reviewable_risk_permission_veto")
        max_order = 0.0
        daily_limit = 0
    elif policy_action == "RISK_BLOCK":
        decision = "RISK_BLOCK"
        reasons.append("bucket_policy_risk_block")
        if policy_reason:
            reasons.append(policy_reason)
        max_order = 0.0
        daily_limit = 0
    elif policy_action == "MICRO_CANARY_REVIEW" and observability_reasons:
        decision = "MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY"
        reasons.append("bucket_policy_review_blocked_by_observability")
        reasons.extend(observability_reasons)
        max_order = 0.0
        daily_limit = 0
    elif policy_action == "MICRO_CANARY_REVIEW":
        decision = "MICRO_CANARY_REVIEW"
        reasons.append("bucket_policy_manual_micro_canary_review")
        if policy_reason:
            reasons.append(policy_reason)
        max_order = 0.0
        daily_limit = 0
    elif policy_action == "MICRO_CANARY_ALLOW" and observability_reasons:
        decision = "MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY"
        reasons.append("bucket_policy_allow_blocked_by_observability")
        reasons.extend(observability_reasons)
        max_order = 0.0
        daily_limit = 0
    elif policy_action == "MICRO_CANARY_ALLOW" and not similar_ok:
        decision = "MICRO_CANARY_REVIEW"
        reasons.append("bucket_policy_allow_requires_similarity_evidence")
        if not similar_ok:
            reasons.append("similar_sample_insufficient_for_auto_allow")
        max_order = 0.0
        daily_limit = 0
    elif (
        policy_action == "MICRO_CANARY_ALLOW"
        and similar_ok
        and not _daily_micro_canary_used(event)
        and (_float((bucket_policy or {}).get("max_single_order_usdt")) or 0.0) > 0.0
        and (_int((bucket_policy or {}).get("daily_trade_limit")) or 0) > 0
    ):
        decision = "MICRO_CANARY_ALLOW"
        reasons.append("bucket_policy_explicit_micro_canary_allow")
        max_order = min(_float((bucket_policy or {}).get("max_single_order_usdt")) or 5.0, 5.0)
        daily_limit = min(_int((bucket_policy or {}).get("daily_trade_limit")) or 1, 1)
    elif high_confidence and observability_reasons:
        decision = "MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY"
        reasons.append("high_confidence_observability_missing")
        reasons.extend(observability_reasons)
        max_order = 0.0
        daily_limit = 0
    elif high_confidence and similar_ok and not _daily_micro_canary_used(event):
        decision = "MICRO_CANARY_REVIEW"
        reasons.append("similar_sample_supported_but_bucket_policy_required")
        max_order = 0.0
        daily_limit = 0
    elif high_confidence and (risk_veto or strategy_veto or not similar_ok):
        decision = "MICRO_CANARY_REVIEW"
        reasons.append("high_confidence_requires_manual_micro_canary_review")
        if not similar_ok:
            reasons.append("similar_sample_insufficient_for_auto_allow")
        max_order = 0.0
        daily_limit = 0
    else:
        decision = "PAPER_ONLY"
        reasons.append("trade_level_not_live_ready")
        if paper_ready:
            reasons.append("paper_ready_still_requires_bucket_policy")
        reasons.extend(high_reasons)
        max_order = 0.0
        daily_limit = 0
    return {
        "schema_version": TRADE_LEVEL_SCHEMA_VERSION,
        "event_id": _text(event.get("event_id")),
        "decision_ts": _timestamp(event.get("decision_ts")),
        "symbol": _text(event.get("symbol")),
        "side": _text(event.get("side")),
        "intent": _text(event.get("intent")),
        "strategy_candidate": _text(event.get("strategy_candidate")),
        "hard_safety_veto": hard,
        "risk_permission_veto": risk_veto,
        "strategy_advisory_veto": strategy_veto,
        "v5_high_confidence_opportunity": high_confidence,
        "similar_sample_count": similar_count,
        "similar_median_after_cost_bps": similar_median,
        "similar_p25_after_cost_bps": similar_p25,
        "recent_7d_similar_mean": recent_7d,
        "bucket_key": bucket_key,
        "bucket_policy_action": policy_action,
        "bucket_policy_reason": policy_reason,
        "bucket_policy_confidence": policy_confidence,
        "trade_level_decision": decision,
        "max_single_order_usdt": max_order,
        "daily_trade_limit": daily_limit,
        "hard_safety_reasons": safe_json_dumps(sorted(set(hard_reasons))),
        "risk_permission_reasons": safe_json_dumps(sorted(set(permission_reasons))),
        "reason": ";".join(sorted(set(reasons))) or "ok",
        "created_at": created,
        "source": "quant_lab.trade_level.judgment",
    }


def _v5_high_confidence(
    event: dict[str, Any],
    hard_reasons: list[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if hard_reasons:
        reasons.append("hard_safety_veto")
    if _int(event.get("rank")) != 1:
        reasons.append("rank_not_1")
    if _text(event.get("alpha6_side")).lower() != "buy":
        reasons.append("alpha6_side_not_buy")
    alpha6 = _float(event.get("alpha6_score"))
    if alpha6 is None or alpha6 < 0.95:
        reasons.append("alpha6_score_lt_0_95")
    edge_ratio = _float(event.get("edge_required_ratio"))
    if edge_ratio is None or edge_ratio < 3.0:
        reasons.append("edge_required_ratio_lt_3")
    if _bool(event.get("cost_gate_verified")) is not True:
        reasons.append("cost_gate_not_verified")
    if _bool(event.get("would_block_by_cost")) is True:
        reasons.append("would_block_by_cost")
    if (
        _bool(event.get("unmanaged_position")) is True
        or _bool(event.get("unmanaged_exposure")) is True
    ):
        reasons.append("unmanaged_position")
    return not reasons, reasons


def _observability_reasons(event: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _float(event.get("arrival_mid")) is None:
        reasons.append("arrival_mid_missing")
    return reasons


def _hard_safety_reasons(event: dict[str, Any]) -> list[str]:
    reasons = set()
    for field, reason in [
        ("kill_switch_on", "kill_switch_on"),
        ("kill_switch_enabled", "kill_switch_on"),
        ("unmanaged_position", "unmanaged_position"),
        ("unmanaged_exposure", "unmanaged_exposure"),
        ("open_exposure_mismatch", "open_exposure_mismatch"),
        ("exchange_local_mismatch", "exchange_local_mismatch"),
        ("market_data_stale", "market_data_stale"),
        ("order_over_limit", "order_over_limit"),
    ]:
        if _bool(event.get(field)) is True:
            reasons.add(reason)
    for field, reason in [
        ("reconcile_status", "reconcile_fail"),
        ("ledger_status", "ledger_fail"),
        ("api_auth_status", "api_auth_fail"),
    ]:
        value = _text(event.get(field)).lower()
        if value and value not in {"ok", "pass", "passed", "healthy"}:
            reasons.add(reason)
    for reason in _risk_reasons(event):
        lowered = reason.lower()
        if lowered in HARD_SAFETY_REASON_CODES:
            reasons.add(lowered)
    return sorted(reasons)


def _risk_permission_veto(event: dict[str, Any]) -> bool:
    permission = _text(event.get("quant_lab_permission")).upper()
    status = _text(event.get("quant_lab_permission_status")).upper()
    return permission == "ABORT" or status.endswith("_ABORT")


def _risk_reasons(event: dict[str, Any]) -> list[str]:
    return _json_list(event.get("quant_lab_live_block_reasons"))


def _risk_reasons_reviewable(reasons: list[str]) -> bool:
    if not reasons:
        return True
    return all(reason.lower() in REVIEWABLE_ABORT_REASON_CODES for reason in reasons)


def _strategy_advisory_veto(event: dict[str, Any]) -> bool:
    reasons = _risk_reasons(event)
    return any("advisory" in reason.lower() or "paper" in reason.lower() for reason in reasons)


def _paper_ready(event: dict[str, Any]) -> bool:
    paper_days = _float(event.get("paper_days")) or 0.0
    closed_entries = _float(event.get("closed_entries") or event.get("closed_entry_count")) or 0.0
    arrival_mid_coverage = _float(event.get("arrival_mid_coverage")) or 0.0
    mean_net = _float(event.get("mean_after_cost_bps") or event.get("avg_net_bps"))
    median_net = _float(event.get("median_after_cost_bps") or event.get("median_net_bps"))
    return (
        paper_days >= 14
        and closed_entries >= 20
        and arrival_mid_coverage >= 0.8
        and mean_net is not None
        and mean_net > 0.0
        and median_net is not None
        and median_net > 0.0
    )


def _daily_micro_canary_used(event: dict[str, Any]) -> bool:
    return _bool(event.get("daily_micro_canary_used")) is True


def _active_bucket_policy_by_key(frame: pl.DataFrame, now: datetime) -> dict[str, dict[str, Any]]:
    if frame.is_empty() or "bucket_key" not in frame.columns:
        return {}
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        key = _text(row.get("bucket_key"))
        if not key:
            continue
        expires_at = _timestamp(row.get("expires_at"))
        if expires_at is not None and expires_at <= now:
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            _timestamp(row.get("policy_date") or row.get("created_at"))
            or datetime.min.replace(tzinfo=UTC),
            _timestamp(row.get("created_at")) or datetime.min.replace(tzinfo=UTC),
        ),
        reverse=True,
    )
    policies: dict[str, dict[str, Any]] = {}
    for row in rows:
        policies.setdefault(_text(row.get("bucket_key")), row)
    return policies


def _policy_action(row: dict[str, Any]) -> str:
    action = _text(row.get("policy_action") or row.get("recommended_trade_level_decision")).upper()
    return action if action in {"RISK_BLOCK", "MICRO_CANARY_REVIEW", "MICRO_CANARY_ALLOW"} else ""


def _v5_would_open(raw: dict[str, Any], payload: dict[str, Any]) -> bool:
    decision = _text(_first(raw, payload, "final_decision", "decision", "action", "intent")).lower()
    if any(token in decision for token in ("buy", "open", "enter", "allow", "submit")):
        return True
    target = _float(_first(raw, payload, "target_weight_after_risk", "target_weight_raw"))
    current = _float(_first(raw, payload, "current_weight"))
    if target is not None and target > max(current or 0.0, 0.0):
        return True
    return _bool(_first(raw, payload, "v5_would_open", "would_open")) is True


def _actual_submission_lookup(
    trades: pl.DataFrame,
    lifecycles: pl.DataFrame,
) -> set[tuple[str, str]]:
    lookup: set[tuple[str, str]] = set()
    for frame in [trades, lifecycles]:
        if frame.is_empty():
            continue
        for row in frame.to_dicts():
            symbol = _symbol(row.get("symbol"))
            run_id = _text(row.get("run_id"))
            action = _text(row.get("action") or row.get("event_type") or row.get("intent")).lower()
            if (
                symbol
                and run_id
                and any(
                    token in action for token in ("entry", "open", "buy", "submitted", "filled")
                )
            ):
                lookup.add((run_id, symbol))
    return lookup


def _latest_risk_permission_row(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {}
    rows = frame.to_dicts()
    return max(
        rows,
        key=lambda row: (
            _timestamp(row.get("as_of_ts") or row.get("created_at"))
            or datetime.min.replace(tzinfo=UTC)
        ),
    )


def _preferred_label_value(label: dict[str, Any]) -> tuple[float | None, int | None]:
    for horizon, field in [
        (24, "label_24h_after_cost_bps"),
        (8, "label_8h_after_cost_bps"),
        (4, "label_4h_after_cost_bps"),
    ]:
        value = _float(label.get(field))
        if value is not None:
            return value, horizon
    return None, None


def _sample_or_label_value(
    sample: dict[str, Any],
    label: dict[str, Any],
) -> tuple[float | None, int | None]:
    value = _float(sample.get("net_bps"))
    horizon = _int(sample.get("label_horizon_hours"))
    if value is not None:
        return value, horizon
    return _preferred_label_value(label)


def _schema_version_for_dataset(dataset_name: str) -> str:
    return {
        "trade_opportunity_event": TRADE_OPPORTUNITY_EVENT_SCHEMA_VERSION,
        "trade_opportunity_label": "trade_opportunity_label.v0.1",
        "trade_level_similarity_outcome": "trade_level_similarity_outcome.v0.1",
        "trade_level_judgment": TRADE_LEVEL_SCHEMA_VERSION,
        "quant_lab_false_block_audit": FALSE_BLOCK_AUDIT_SCHEMA_VERSION,
        "v5_trade_learning_sample": V5_TRADE_LEARNING_SAMPLE_SCHEMA_VERSION,
        "v5_trade_outcome_attribution": V5_TRADE_OUTCOME_ATTRIBUTION_SCHEMA_VERSION,
        "quant_lab_opportunity_cost_event": OPPORTUNITY_COST_EVENT_SCHEMA_VERSION,
        "quant_lab_opportunity_cost_daily": OPPORTUNITY_COST_DAILY_SCHEMA_VERSION,
        "opportunity_cost_by_bucket": OPPORTUNITY_COST_BY_BUCKET_SCHEMA_VERSION,
        "trade_level_bucket_policy": TRADE_LEVEL_BUCKET_POLICY_SCHEMA_VERSION,
        "trade_level_opportunity_queue": TRADE_LEVEL_OPPORTUNITY_QUEUE_SCHEMA_VERSION,
        "quant_lab_decision_regret": DECISION_REGRET_SCHEMA_VERSION,
    }.get(dataset_name, dataset_name)


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value and str(value).lower() != "auto":
        return date.fromisoformat(str(value))
    return datetime.now(UTC).date()


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    for field in ("raw_payload_json", "payload_json", "extra_json"):
        value = row.get(field)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _first(row: dict[str, Any], payload: dict[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
        value = payload.get(field)
        if value not in (None, ""):
            return value
    return None


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row").select(
        [pl.col(name).cast(dtype, strict=False).alias(name) for name, dtype in schema.items()]
    )


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [_text(part) for part in text.replace(";", ",").split(",") if _text(part)]
        if isinstance(parsed, list):
            return [_text(item) for item in parsed if _text(item)]
        if isinstance(parsed, str):
            return [_text(parsed)] if _text(parsed) else []
    return [_text(value)] if _text(value) else []


def _json_list_text(value: Any) -> str:
    return safe_json_dumps(_json_list(value))


def _symbol(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        return normalize_symbol(text)
    except Exception:
        return text.replace("/", "-").upper()


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


def _iso(value: datetime | None) -> str:
    return value.isoformat().replace("+00:00", "Z") if value else ""


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


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None
