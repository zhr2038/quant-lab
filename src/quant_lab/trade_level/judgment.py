from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset, write_snapshot_meta
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

TRADE_LEVEL_SCHEMA_VERSION = "trade_level_judgment.v0.1"
TRADE_OPPORTUNITY_EVENT_SCHEMA_VERSION = "trade_opportunity_event.v0.1"
FALSE_BLOCK_AUDIT_SCHEMA_VERSION = "quant_lab_false_block_audit.v0.1"

TRADE_OPPORTUNITY_EVENT_DATASET = Path("gold") / "trade_opportunity_event"
TRADE_OPPORTUNITY_LABEL_DATASET = Path("gold") / "trade_opportunity_label"
TRADE_LEVEL_SIMILARITY_DATASET = Path("gold") / "trade_level_similarity_outcome"
TRADE_LEVEL_JUDGMENT_DATASET = Path("gold") / "trade_level_judgment"
FALSE_BLOCK_AUDIT_DATASET = Path("gold") / "quant_lab_false_block_audit"

V5_CANDIDATE_EVENT_DATASET = Path("silver") / "v5_candidate_event"
V5_TRADE_EVENT_DATASET = Path("silver") / "v5_trade_event"
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
    "cost_gate_verified": pl.Boolean,
    "would_block_by_cost": pl.Boolean,
    "risk_level": pl.Utf8,
    "regime": pl.Utf8,
    "arrival_mid": pl.Float64,
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
        order_lifecycles=read_parquet_dataset(root / V5_ORDER_LIFECYCLE_DATASET),
        created_at=generated_at,
    )
    warnings: list[str] = []
    for dataset_name, relative_path in [
        ("trade_opportunity_event", TRADE_OPPORTUNITY_EVENT_DATASET),
        ("trade_opportunity_label", TRADE_OPPORTUNITY_LABEL_DATASET),
        ("trade_level_similarity_outcome", TRADE_LEVEL_SIMILARITY_DATASET),
        ("trade_level_judgment", TRADE_LEVEL_JUDGMENT_DATASET),
        ("quant_lab_false_block_audit", FALSE_BLOCK_AUDIT_DATASET),
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
        warnings=warnings,
    )


def build_trade_level_frames_from_sources(
    *,
    candidate_events: pl.DataFrame,
    candidate_labels: pl.DataFrame,
    risk_permissions: pl.DataFrame,
    v5_trades: pl.DataFrame,
    order_lifecycles: pl.DataFrame | None = None,
    created_at: datetime | None = None,
) -> dict[str, pl.DataFrame]:
    from quant_lab.trade_level.labels import build_trade_opportunity_labels
    from quant_lab.trade_level.similarity import build_trade_level_similarity_outcome

    generated_at = created_at or datetime.now(UTC)
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
    audit = build_false_block_audit(
        events,
        labels,
        judgments,
        created_at=generated_at,
    )
    return {
        "trade_opportunity_event": events,
        "trade_opportunity_label": labels,
        "trade_level_similarity_outcome": similarity,
        "trade_level_judgment": judgments,
        "quant_lab_false_block_audit": audit,
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
    created_at: datetime | None = None,
) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(schema=TRADE_LEVEL_JUDGMENT_SCHEMA)
    created = created_at or datetime.now(UTC)
    similarity_frame = similarity if similarity is not None else pl.DataFrame()
    similarity_by_event = {
        str(row.get("event_id") or ""): row for row in similarity_frame.to_dicts()
    }
    rows = [
        _judgment_row(event, similarity_by_event.get(str(event.get("event_id") or "")), created)
        for event in events.to_dicts()
    ]
    return _frame(rows, TRADE_LEVEL_JUDGMENT_SCHEMA)


def build_false_block_audit(
    events: pl.DataFrame,
    labels: pl.DataFrame,
    judgments: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    if events.is_empty() or judgments.is_empty():
        return pl.DataFrame(schema=FALSE_BLOCK_AUDIT_SCHEMA)
    created = created_at or datetime.now(UTC)
    events_by_id = {str(row.get("event_id") or ""): row for row in events.to_dicts()}
    labels_by_id = {str(row.get("event_id") or ""): row for row in labels.to_dicts()}
    rows: list[dict[str, Any]] = []
    for judgment in judgments.to_dicts():
        event_id = str(judgment.get("event_id") or "")
        event = events_by_id.get(event_id, {})
        label = labels_by_id.get(event_id, {})
        value, horizon = _preferred_label_value(label)
        decision = _text(judgment.get("trade_level_decision"))
        quant_lab_would_block = decision not in {"MICRO_CANARY_ALLOW", "LIVE_SMALL_ALLOW"}
        was_profitable = value is not None and value > 0.0
        false_block = bool(event.get("v5_would_open")) and quant_lab_would_block and was_profitable
        rows.append(
            {
                "schema_version": FALSE_BLOCK_AUDIT_SCHEMA_VERSION,
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
    judgments: pl.DataFrame, false_block_audit: pl.DataFrame | None = None
) -> dict[str, Any]:
    if judgments.is_empty():
        return {
            "trade_level_decision_summary": safe_json_dumps({}),
            "micro_canary_review_count": 0,
            "false_block_rate": 0.0,
        }
    decisions = Counter(
        _text(row.get("trade_level_decision")) or "UNKNOWN" for row in judgments.to_dicts()
    )
    review_count = decisions.get("MICRO_CANARY_REVIEW", 0)
    false_block_rate = 0.0
    audit = false_block_audit if false_block_audit is not None else pl.DataFrame()
    if not audit.is_empty():
        rows = [row for row in audit.to_dicts() if row.get("quant_lab_would_block")]
        if rows:
            false_block_rate = sum(1 for row in rows if row.get("false_block")) / len(rows)
    return {
        "trade_level_decision_summary": safe_json_dumps(dict(sorted(decisions.items()))),
        "micro_canary_review_count": int(review_count),
        "false_block_rate": round(float(false_block_rate), 6),
    }


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
        "cost_gate_verified": _bool(_first(raw, payload, "cost_gate_verified")),
        "would_block_by_cost": _bool(_first(raw, payload, "would_block_by_cost")),
        "risk_level": _text(_first(raw, payload, "risk_level", "risk_state")),
        "regime": _text(_first(raw, payload, "regime", "regime_state")),
        "arrival_mid": arrival_mid,
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
    event: dict[str, Any], similarity: dict[str, Any] | None, created: datetime
) -> dict[str, Any]:
    hard_reasons = _hard_safety_reasons(event)
    permission_reasons = _risk_reasons(event)
    hard = bool(hard_reasons)
    risk_veto = _risk_permission_veto(event)
    strategy_veto = _strategy_advisory_veto(event)
    high_confidence, high_reasons = _v5_high_confidence(event, hard_reasons)
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
    reasons: list[str] = []
    if hard:
        decision = "HARD_BLOCK"
        reasons.extend(hard_reasons)
        max_order = 0.0
        daily_limit = 0
    elif paper_ready and not risk_veto:
        decision = "LIVE_SMALL_ALLOW"
        reasons.append("paper_ready_trade_level_allow")
        max_order = max(_float(event.get("risk_max_single_order_usdt")) or 5.0, 5.0)
        daily_limit = 1
    elif high_confidence and similar_ok and not _daily_micro_canary_used(event):
        decision = "MICRO_CANARY_ALLOW"
        reasons.append("high_confidence_with_supported_similar_sample")
        max_order = 5.0
        daily_limit = 1
    elif high_confidence and (risk_veto or strategy_veto or not similar_ok):
        decision = "MICRO_CANARY_REVIEW"
        reasons.append("high_confidence_requires_manual_micro_canary_review")
        if not similar_ok:
            reasons.append("similar_sample_insufficient_for_auto_allow")
        max_order = 0.0
        daily_limit = 0
    elif risk_veto and not _risk_reasons_reviewable(permission_reasons):
        decision = "RISK_BLOCK"
        reasons.append("non_reviewable_risk_permission_veto")
        max_order = 0.0
        daily_limit = 0
    else:
        decision = "PAPER_ONLY"
        reasons.append("trade_level_not_live_ready")
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
    if _float(event.get("arrival_mid")) is None:
        reasons.append("arrival_mid_missing")
    selected = _float(event.get("selected_cost_bps"))
    actual = _float(event.get("actual_all_in_cost_bps"))
    if (
        actual is not None
        and selected is not None
        and actual > max(selected + 10.0, selected * 1.5)
    ):
        reasons.append("actual_cost_significantly_above_selected_cost")
    if (
        _bool(event.get("unmanaged_position")) is True
        or _bool(event.get("unmanaged_exposure")) is True
    ):
        reasons.append("unmanaged_position")
    return not reasons, reasons


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


def _schema_version_for_dataset(dataset_name: str) -> str:
    return {
        "trade_opportunity_event": TRADE_OPPORTUNITY_EVENT_SCHEMA_VERSION,
        "trade_opportunity_label": "trade_opportunity_label.v0.1",
        "trade_level_similarity_outcome": "trade_level_similarity_outcome.v0.1",
        "trade_level_judgment": TRADE_LEVEL_SCHEMA_VERSION,
        "quant_lab_false_block_audit": FALSE_BLOCK_AUDIT_SCHEMA_VERSION,
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
    return pl.DataFrame(rows, orient="row").select(
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
