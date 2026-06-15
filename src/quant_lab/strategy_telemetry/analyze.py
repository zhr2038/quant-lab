from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import (
    append_parquet_dataset,
    read_parquet_dataset,
    read_parquet_lazy,
    upsert_parquet_dataset,
    write_parquet_dataset,
)
from quant_lab.strategy_telemetry.models import V5TelemetryAnalysisResult
from quant_lab.symbols import normalize_symbol

GOLD_DATASETS = {
    "strategy_health_daily": Path("gold/strategy_health_daily"),
    "v5_execution_quality_daily": Path("gold/v5_execution_quality_daily"),
    "v5_gate_compliance_daily": Path("gold/v5_gate_compliance_daily"),
    "v5_missed_opportunity_daily": Path("gold/v5_missed_opportunity_daily"),
    "v5_config_health_daily": Path("gold/v5_config_health_daily"),
    "v5_issue_summary_daily": Path("gold/v5_issue_summary_daily"),
    "v5_quant_lab_mode_daily": Path("gold/v5_quant_lab_mode_daily"),
    "v5_quant_lab_enforcement_daily": Path("gold/v5_quant_lab_enforcement_daily"),
    "v5_final_score_vs_alpha6_conflict": Path("gold/v5_final_score_vs_alpha6_conflict"),
    "v5_bnb_strong_alpha6_bypass_shadow": Path("gold/v5_bnb_strong_alpha6_bypass_shadow"),
    "v5_negative_expectancy_attribution": Path("gold/v5_negative_expectancy_attribution"),
    "v5_bnb_paper_strategy_runs": Path("gold/v5_bnb_paper_strategy_runs"),
    "v5_bnb_paper_strategy_daily": Path("gold/v5_bnb_paper_strategy_daily"),
    "v5_bnb_paper_strategy_daily_latest": Path("gold/v5_bnb_paper_strategy_daily_latest"),
}

SILVER = {
    "manifest": Path("bronze/strategy_telemetry/v5/bundle_manifest"),
    "secret_scan": Path("bronze/strategy_telemetry/v5/secret_scan"),
    "run_summary": Path("silver/v5_run_summary"),
    "decision_audit": Path("silver/v5_decision_audit"),
    "trade_event": Path("silver/v5_trade_event"),
    "roundtrip": Path("silver/v5_roundtrip"),
    "router_decision": Path("silver/v5_router_decision"),
    "open_position": Path("silver/v5_open_position"),
    "state_snapshot": Path("silver/v5_state_snapshot"),
    "issue": Path("silver/v5_issue"),
    "config_audit": Path("silver/v5_config_audit"),
    "high_score_target": Path("silver/v5_high_score_blocked_target"),
    "high_score_outcome": Path("silver/v5_high_score_blocked_outcome"),
    "skipped": Path("silver/v5_skipped_candidate_outcome"),
    "quant_lab_usage": Path("silver/v5_quant_lab_usage"),
    "quant_lab_request": Path("silver/v5_quant_lab_request"),
    "quant_lab_compliance": Path("silver/v5_quant_lab_compliance"),
    "quant_lab_cost_usage": Path("silver/v5_quant_lab_cost_usage"),
    "quant_lab_fallback": Path("silver/v5_quant_lab_fallback"),
    "final_score_alpha6_conflict": Path("silver/v5_final_score_vs_alpha6_conflict"),
    "bnb_strong_alpha6_bypass_shadow": Path("silver/v5_bnb_strong_alpha6_bypass_shadow"),
    "negative_expectancy_attribution": Path("silver/v5_negative_expectancy_attribution"),
    "bnb_paper_strategy_runs": Path("silver/v5_bnb_paper_strategy_runs"),
    "bnb_paper_strategy_daily": Path("silver/v5_bnb_paper_strategy_daily"),
}

GOLD_TELEMETRY_MIRRORS = {
    "final_score_alpha6_conflict": "v5_final_score_vs_alpha6_conflict",
    "bnb_strong_alpha6_bypass_shadow": "v5_bnb_strong_alpha6_bypass_shadow",
    "negative_expectancy_attribution": "v5_negative_expectancy_attribution",
    "bnb_paper_strategy_runs": "v5_bnb_paper_strategy_runs",
    "bnb_paper_strategy_daily": "v5_bnb_paper_strategy_daily",
}

ANALYSIS_TIME_COLUMNS = ("bundle_ts", "ingest_ts", "ts_utc", "ts", "created_at")
DEFAULT_ANALYSIS_LOOKBACK_DAYS = 14


def _read_analysis_dataset(
    dataset_path: Path,
    columns: list[str],
    *,
    analysis_date: str | None = None,
    lookback_days: int | None = None,
) -> pl.DataFrame:
    try:
        lazy = read_parquet_lazy(dataset_path)
        available = lazy.collect_schema().names()
    except Exception:
        return pl.DataFrame()
    selected = [column for column in columns if column in available]
    if not selected:
        return pl.DataFrame()
    if analysis_date is not None and lookback_days is not None and lookback_days > 0:
        lazy = _filter_analysis_window(
            lazy,
            set(available),
            analysis_date=analysis_date,
            lookback_days=lookback_days,
        )
    try:
        return lazy.select(selected).collect()
    except Exception:
        return pl.DataFrame()


def _analysis_lookback_days() -> int:
    raw = os.environ.get("QUANT_LAB_V5_TELEMETRY_ANALYSIS_LOOKBACK_DAYS")
    if raw is None or not raw.strip():
        return DEFAULT_ANALYSIS_LOOKBACK_DAYS
    try:
        return max(int(raw), 1)
    except ValueError:
        return DEFAULT_ANALYSIS_LOOKBACK_DAYS


def _filter_analysis_window(
    lazy: pl.LazyFrame,
    available: set[str],
    *,
    analysis_date: str,
    lookback_days: int,
) -> pl.LazyFrame:
    time_expr = _coalesced_lazy_time(available)
    if time_expr is None:
        return lazy
    try:
        day = datetime.fromisoformat(str(analysis_date)[:10]).date()
    except ValueError:
        return lazy
    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC) - timedelta(
        days=lookback_days
    )
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    return lazy.with_columns(time_expr.alias("_analysis_ts")).filter(
        (pl.col("_analysis_ts") >= start) & (pl.col("_analysis_ts") < end)
    )


def _coalesced_lazy_time(available: set[str]) -> pl.Expr | None:
    expressions = [
        _lazy_datetime(column) for column in ANALYSIS_TIME_COLUMNS if column in available
    ]
    if not expressions:
        return None
    return pl.coalesce(expressions)


def _lazy_datetime(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .replace("", None)
        .str.to_datetime(
            time_zone="UTC",
            strict=False,
        )
    )


def analyze_v5_telemetry(
    lake_root: Path,
    date: str | None = None,
    *,
    refresh_candidate_gold: bool = True,
) -> V5TelemetryAnalysisResult:
    root = Path(lake_root)
    analysis_date = date or datetime.now(UTC).date().isoformat()
    lookback_days = _analysis_lookback_days()
    manifest = _read_analysis_dataset(
        root / SILVER["manifest"],
        ["bundle_ts", "bundle_sha256", "bundle_name", "ingest_ts"],
        analysis_date=analysis_date,
        lookback_days=max(lookback_days, 30),
    )
    secret_scan = _read_analysis_dataset(
        root / SILVER["secret_scan"],
        ["high_severity_count", "bundle_ts", "ingest_ts"],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    run_summary = _read_analysis_dataset(
        root / SILVER["run_summary"],
        ["raw_payload_json", "source_path_inside_bundle", "bundle_ts", "ingest_ts"],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    decisions = _read_analysis_dataset(
        root / SILVER["decision_audit"],
        ["raw_payload_json", "source_path_inside_bundle", "bundle_ts", "ingest_ts"],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    trades = _read_analysis_dataset(
        root / SILVER["trade_event"],
        ["bundle_ts", "ingest_ts", "ts_utc", "ts", "created_at", "source_path_inside_bundle"],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    roundtrips = _read_analysis_dataset(
        root / SILVER["roundtrip"],
        ["bundle_ts", "ingest_ts", "source_path_inside_bundle"],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    routers = _read_analysis_dataset(
        root / SILVER["router_decision"],
        ["reason", "router_reason", "decision_reason", "bundle_ts", "ingest_ts"],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    positions = _read_analysis_dataset(
        root / SILVER["open_position"],
        [
            "bundle_ts",
            "ingest_ts",
            "symbol",
            "size",
            "quantity",
            "dust",
            "is_dust",
            "raw_payload_json",
        ],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    states = _read_analysis_dataset(
        root / SILVER["state_snapshot"],
        [
            "state_type",
            "ok",
            "enabled",
            "level",
            "current_level",
            "risk_level",
            "raw_payload_json",
            "bundle_ts",
            "ingest_ts",
        ],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    issues = _read_analysis_dataset(
        root / SILVER["issue"],
        ["severity", "issue_type", "type", "message", "raw_payload_json", "bundle_ts", "ingest_ts"],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    config = _read_analysis_dataset(
        root / SILVER["config_audit"],
        [
            "consumed",
            "status",
            "issue_type",
            "key",
            "config_key",
            "path",
            "name",
            "blob",
            "raw_payload_json",
            "bundle_ts",
            "ingest_ts",
        ],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    targets = _read_analysis_dataset(
        root / SILVER["high_score_target"],
        ["bundle_ts", "ingest_ts"],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    outcomes = _read_analysis_dataset(
        root / SILVER["high_score_outcome"],
        [
            "matured",
            "is_matured",
            "maturity_reached",
            "profitable",
            "is_profitable",
            "bundle_ts",
            "ingest_ts",
        ],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    skipped = _read_analysis_dataset(
        root / SILVER["skipped"],
        ["matured", "is_matured", "maturity_reached", "bundle_ts", "ingest_ts"],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    quant_lab_usage = _read_analysis_dataset(
        root / SILVER["quant_lab_usage"],
        [
            "mode",
            "quant_lab_mode",
            "permission_gate_enforced",
            "apply_permission_gate",
            "cost_gate_enforced",
            "apply_cost_gate",
            "bundle_ts",
            "ingest_ts",
            "ts_utc",
            "ts",
            "created_at",
        ],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    quant_lab_request = _read_analysis_dataset(
        root / SILVER["quant_lab_request"],
        [
            "raw_payload_json",
            "raw_json",
            "source_path_inside_bundle",
            "event_key",
            "event_id",
            "strategy",
            "strategy_id",
            "run_id",
            "source_count",
            "last_seen_source_count",
            "payload_hash_count",
            "payload_hashes_json",
            "conflicting_duplicate",
            "raw_payload_hash",
            "first_seen_bundle_ts",
            "last_seen_bundle_ts",
            "bundle_ts",
            "ingest_ts",
            "ts_utc",
            "ts",
            "endpoint",
            "endpoint_path",
            "event_type",
            "status_code",
            "http_status",
            "error_type",
            "exception_type",
            "error",
            "exception",
            "fallback_used",
            "used_fallback",
            "local_fallback",
            "request_id",
            "trace_id",
            "symbol",
            "side",
            "intent",
            "success",
            "ok",
            "request_ok",
            "permission_status",
            "remote_permission_status",
            "risk_permission_status",
        ],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    quant_lab_compliance = _read_analysis_dataset(
        root / SILVER["quant_lab_compliance"],
        [
            "permission",
            "quant_lab_permission",
            "actual_violation",
            "hypothetical_violation",
            "reduce_only",
            "is_reduce_only",
            "new_risk",
            "risk_increasing",
            "raw_permission_decision",
            "raw_permission_status",
            "effective_permission_decision",
            "effective_permission_status",
            "permission_decision",
            "effective_decision",
            "order_decision",
            "permission_contract_violation",
            "would_block_if_enforced",
            "intent",
            "action",
            "router_intent",
            "side",
            "order_side",
            "mode",
            "quant_lab_mode",
            "permission_gate_enforced",
            "apply_permission_gate",
            "cost_gate_enforced",
            "apply_cost_gate",
            "bundle_ts",
            "ingest_ts",
            "ts_utc",
            "ts",
            "created_at",
        ],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    quant_lab_cost_usage = _read_analysis_dataset(
        root / SILVER["quant_lab_cost_usage"],
        [
            "mode",
            "quant_lab_mode",
            "cost_gate_enforced",
            "apply_cost_gate",
            "bundle_ts",
            "ingest_ts",
        ],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    quant_lab_fallback = _read_analysis_dataset(
        root / SILVER["quant_lab_fallback"],
        [
            "raw_payload_json",
            "raw_json",
            "source_path_inside_bundle",
            "event_key",
            "event_id",
            "strategy",
            "strategy_id",
            "run_id",
            "source_count",
            "last_seen_source_count",
            "payload_hash_count",
            "payload_hashes_json",
            "conflicting_duplicate",
            "raw_payload_hash",
            "first_seen_bundle_ts",
            "last_seen_bundle_ts",
            "bundle_ts",
            "ingest_ts",
            "ts_utc",
            "ts",
            "endpoint",
            "endpoint_path",
            "event_type",
            "status_code",
            "http_status",
            "error_type",
            "exception_type",
            "error",
            "exception",
            "diagnosis",
            "fallback_used",
            "used_fallback",
            "local_fallback",
            "request_id",
            "trace_id",
            "symbol",
            "side",
            "intent",
            "success",
            "ok",
            "request_ok",
        ],
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    window_summary = _window_summary_payload(run_summary)

    latest_bundle_ts = _latest_time(manifest, "bundle_ts")
    latest_sha = _latest_value(manifest, "bundle_sha256", "bundle_ts")
    critical: list[str] = []
    warnings: list[str] = []
    next_actions: list[str] = []

    kill_switch_enabled = _latest_state_bool(states, "kill_switch", "enabled")
    reconcile_ok = _latest_state_bool(states, "reconcile_status", "ok")
    ledger_ok = _latest_state_bool(states, "ledger_status", "ok")
    auto_risk_level = _latest_state_text(states, "auto_risk_eval", "level")

    if kill_switch_enabled is True:
        critical.append("kill_switch_enabled")
    if reconcile_ok is False:
        critical.append("reconcile_not_ok")
    if ledger_ok is False:
        critical.append("ledger_not_ok")

    high_issue_count = _summary_int(
        window_summary,
        ["high_issue_count"],
        fallback=_count_where(issues, "severity", "high"),
    )
    medium_issue_count = _summary_int(
        window_summary,
        ["medium_issue_count"],
        fallback=_count_where(issues, "severity", "medium"),
    )
    if high_issue_count > 0:
        warnings.append("high issues present")
    high_score_matured_issue_count = _high_score_blocked_matured_issue_count(issues)
    if high_score_matured_issue_count:
        next_actions.append("review high_score_blocked_matured_without_label")

    high_secret_count = _max_int(secret_scan, "high_severity_count")
    if high_secret_count > 0:
        critical.append("secret_scan_high_severity")

    if decisions.is_empty():
        warnings.append("no decision_audit in lake")
    if latest_bundle_ts is None and manifest.is_empty():
        warnings.append("no V5 bundle has been ingested")
    elif latest_bundle_ts is None:
        warnings.append("latest bundle timestamp is unavailable")
    elif _is_stale(latest_bundle_ts, analysis_date):
        warnings.append("latest bundle is older than 24 hours")

    config_summary = _config_not_consumed_summary(config)
    config_not_consumed_count = config_summary["count"]
    if config_summary["unknown"]:
        warnings.append("config_not_consumed_count_unknown")
    elif config_not_consumed_count > 0:
        warnings.append("config values not consumed at runtime")

    quant_lab_summary = _quant_lab_mode_summary(
        quant_lab_usage,
        quant_lab_request,
        quant_lab_compliance,
        quant_lab_cost_usage,
        quant_lab_fallback,
        decisions,
        trades,
        analysis_date=analysis_date,
    )
    gate_violations = (
        list(quant_lab_summary["actual_violations"])
        if quant_lab_summary["has_data"]
        else _gate_compliance_violations(decisions, trades)
    )
    if gate_violations:
        critical.append("gate_compliance_violation")
        next_actions.extend(gate_violations)
    if quant_lab_summary["hypothetical_violations"]:
        warnings.append("quant_lab hypothetical permission violations present")

    router_reason_top = _router_reason_top(routers)
    fallback_count = int(quant_lab_summary["actual_fallback_count"])
    if fallback_count:
        warnings.append(f"quant_lab actual fallback used {fallback_count} times")
    stale_permission_count = int(quant_lab_summary["stale_permission_consecutive_count"])
    if stale_permission_count >= 2:
        warnings.append(
            f"quant_lab stale or expired remote permission repeated {stale_permission_count} times"
        )
        next_actions.append("run quant-lab publish-risk-permission and verify risk timer")

    status = "OK"
    if warnings:
        status = "WARNING"
    if critical:
        status = "CRITICAL"

    high_score_blocked_matured_count = _summary_int(
        window_summary,
        ["high_score_blocked_matured_count"],
        fallback=max(_matured_count(outcomes), high_score_matured_issue_count),
    )
    result = V5TelemetryAnalysisResult(
        date=analysis_date,
        status=status,
        latest_bundle_ts=latest_bundle_ts,
        latest_bundle_sha256=latest_sha,
        run_count_72h=_summary_int(
            window_summary,
            ["run_count", "run_count_72h"],
            fallback=_count_recent_rows(
                run_summary,
                analysis_date,
                hours=72,
                exclude_source_path="summaries/window_summary.json",
            ),
        ),
        decision_audit_count_24h=_summary_int(
            window_summary,
            ["recent_24h_decision_audit_count", "decision_audit_count_24h"],
            fallback=_count_recent_rows(decisions, analysis_date, hours=24),
        ),
        trade_count_24h=_summary_int(
            window_summary,
            ["latest_24h_trade_count", "trade_count_24h"],
            fallback=_count_recent_rows(trades, analysis_date, hours=24),
        ),
        trade_count_72h=_summary_int(
            window_summary,
            ["last_72h_trade_count", "trade_count_72h"],
            fallback=_count_recent_rows(trades, analysis_date, hours=72),
        ),
        roundtrip_count_72h=_summary_int(
            window_summary,
            ["last_72h_roundtrip_count", "roundtrip_count_72h"],
            fallback=_count_recent_rows(roundtrips, analysis_date, hours=72),
        ),
        open_position_count=_summary_int(
            window_summary,
            ["open_position_count"],
            fallback=_count_rows(positions),
        ),
        dust_residual_position_count=_summary_int(
            window_summary,
            ["dust_residual_position_count"],
            fallback=_dust_count(positions),
        ),
        kill_switch_enabled=kill_switch_enabled,
        reconcile_ok=reconcile_ok,
        ledger_ok=ledger_ok,
        auto_risk_level=auto_risk_level,
        high_issue_count=high_issue_count,
        medium_issue_count=medium_issue_count,
        config_not_consumed_count=config_not_consumed_count,
        config_not_consumed_count_unknown=bool(config_summary["unknown"]),
        config_not_consumed_top_keys=list(config_summary["top_keys"]),
        high_score_blocked_count=_summary_int(
            window_summary,
            ["high_score_blocked_count"],
            fallback=_count_rows(targets),
        ),
        high_score_blocked_matured_count=high_score_blocked_matured_count,
        high_score_blocked_profitable_count=_profitable_count(outcomes),
        skipped_candidate_matured_count=_matured_count(skipped),
        router_reason_top=router_reason_top,
        quant_lab_mode=quant_lab_summary["mode"],
        permission_gate_enforced=quant_lab_summary["permission_gate_enforced"],
        cost_gate_enforced=quant_lab_summary["cost_gate_enforced"],
        quant_lab_usage_count=int(quant_lab_summary["usage_count"]),
        quant_lab_cost_usage_count=int(quant_lab_summary["cost_usage_count"]),
        quant_lab_fallback_count=int(quant_lab_summary["fallback_count"]),
        unique_request_count=int(quant_lab_summary["unique_request_count"]),
        unique_success_count=int(quant_lab_summary["unique_success_count"]),
        unique_error_count=int(quant_lab_summary["unique_error_count"]),
        unique_actual_fallback_count=int(quant_lab_summary["unique_actual_fallback_count"]),
        request_success_count=int(quant_lab_summary["request_success_count"]),
        request_error_count=int(quant_lab_summary["request_error_count"]),
        actual_fallback_count=int(quant_lab_summary["actual_fallback_count"]),
        fallback_rate=float(quant_lab_summary["fallback_rate"]),
        degraded_reason=str(quant_lab_summary["degraded_reason"]),
        raw_imported_rows=int(quant_lab_summary["raw_imported_rows"]),
        unique_event_rows=int(quant_lab_summary["unique_event_rows"]),
        duplicate_event_count=int(quant_lab_summary["duplicate_event_count"]),
        duplicate_event_rows=int(quant_lab_summary["duplicate_event_rows"]),
        duplicate_rate=float(quant_lab_summary["duplicate_rate"]),
        exact_duplicate_event_rows=int(quant_lab_summary["exact_duplicate_event_rows"]),
        conflicting_duplicate_event_rows=int(
            quant_lab_summary["conflicting_duplicate_event_rows"]
        ),
        conflicting_duplicate_event_key_count=int(
            quant_lab_summary["conflicting_duplicate_event_key_count"]
        ),
        duplicate_explanation=str(quant_lab_summary["duplicate_explanation"]),
        first_seen_bundle_ts=quant_lab_summary["first_seen_bundle_ts"],
        last_seen_bundle_ts=quant_lab_summary["last_seen_bundle_ts"],
        latest_permission_status=quant_lab_summary["latest_permission_status"],
        stale_permission_consecutive_count=stale_permission_count,
        quant_lab_actual_violation_count=len(quant_lab_summary["actual_violations"]),
        quant_lab_hypothetical_violation_count=len(
            quant_lab_summary["hypothetical_violations"]
        ),
        warnings=warnings,
        critical_reasons=critical,
        next_actions=sorted(set(next_actions)),
    )
    _write_gold(
        root,
        result,
        gate_violations=gate_violations,
        fallback_count=fallback_count,
        quant_lab_summary=quant_lab_summary,
    )
    _mirror_v5_consistency_gold(
        root,
        analysis_date=analysis_date,
        lookback_days=lookback_days,
    )
    if refresh_candidate_gold:
        _build_candidate_labels_safely(root, analysis_date)
        _build_alpha_discovery_board_safely(root, analysis_date)
        _build_strategy_evidence_safely(root, analysis_date)
        _build_entry_quality_safely(root, analysis_date)
    return result


def _mirror_v5_consistency_gold(
    lake_root: Path,
    *,
    analysis_date: str,
    lookback_days: int,
) -> None:
    for silver_name, gold_name in GOLD_TELEMETRY_MIRRORS.items():
        silver_path = lake_root / SILVER[silver_name]
        gold_path = lake_root / GOLD_DATASETS[gold_name]
        repaired = _repair_gold_mirror_if_needed(
            silver_path,
            gold_path,
            analysis_date=analysis_date,
        )
        if repaired:
            continue
        frame = _read_incremental_mirror_frame(
            silver_path,
            gold_path,
            analysis_date=analysis_date,
            lookback_days=lookback_days,
        )
        if frame.is_empty():
            continue
        _append_gold_mirror_rows(frame, gold_path, analysis_date)
        if gold_name == "v5_bnb_paper_strategy_daily":
            previous_latest = _safe_read_dataset(
                lake_root / GOLD_DATASETS["v5_bnb_paper_strategy_daily_latest"]
            )
            latest_input = (
                pl.concat([previous_latest, frame], how="diagonal_relaxed")
                if not previous_latest.is_empty()
                else frame
            )
            latest = _latest_bnb_paper_strategy_daily(latest_input)
            if not latest.is_empty():
                write_parquet_dataset(
                    latest,
                    lake_root / GOLD_DATASETS["v5_bnb_paper_strategy_daily_latest"],
                )


def _repair_gold_mirror_if_needed(
    silver_path: Path,
    gold_path: Path,
    *,
    analysis_date: str,
) -> bool:
    silver_rows = _dataset_row_count(silver_path)
    if silver_rows <= 0:
        return False
    gold_rows = _dataset_row_count(gold_path)
    silver_latest = _latest_dataset_time(silver_path)
    gold_latest = _latest_dataset_time(gold_path)
    inconsistent = gold_rows > silver_rows or (
        gold_rows == silver_rows
        and silver_latest is not None
        and gold_latest is not None
        and silver_latest != gold_latest
    )
    if not inconsistent:
        return False
    if silver_rows > _mirror_repair_max_rows():
        return False
    silver = _safe_read_dataset(silver_path)
    if silver.is_empty():
        return False
    write_parquet_dataset(
        _with_bundle_day(silver, analysis_date),
        gold_path,
        partition_by="bundle_day",
    )
    return True


def _mirror_repair_max_rows() -> int:
    raw = os.environ.get("QUANT_LAB_V5_MIRROR_REPAIR_MAX_ROWS")
    if raw is None or not raw.strip():
        return 200_000
    try:
        return max(int(raw), 0)
    except ValueError:
        return 200_000


def _read_incremental_mirror_frame(
    silver_path: Path,
    gold_path: Path,
    *,
    analysis_date: str,
    lookback_days: int,
) -> pl.DataFrame:
    try:
        lazy = read_parquet_lazy(silver_path)
        available = set(lazy.collect_schema().names())
    except Exception:
        return pl.DataFrame()
    time_expr = _coalesced_lazy_time(available)
    if time_expr is None:
        return pl.DataFrame()
    latest_gold_ts = _latest_dataset_time(gold_path)
    working = lazy.with_columns(time_expr.alias("_mirror_ts"))
    if latest_gold_ts is not None:
        working = working.filter(pl.col("_mirror_ts") > latest_gold_ts)
    else:
        working = _filter_analysis_window(
            working,
            available | {"_mirror_ts"},
            analysis_date=analysis_date,
            lookback_days=max(lookback_days, 30),
        )
    try:
        return working.drop("_mirror_ts").collect()
    except Exception:
        return pl.DataFrame()


def _latest_dataset_time(path: Path) -> datetime | None:
    try:
        lazy = read_parquet_lazy(path)
        available = set(lazy.collect_schema().names())
    except Exception:
        return None
    time_expr = _coalesced_lazy_time(available)
    if time_expr is None:
        return None
    try:
        value = lazy.select(time_expr.max().alias("latest_ts")).collect().item()
    except Exception:
        return None
    if not isinstance(value, datetime):
        return None
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def _dataset_row_count(path: Path) -> int:
    try:
        return int(read_parquet_lazy(path).select(pl.len().alias("rows")).collect().item())
    except Exception:
        return 0


def _append_gold_mirror_rows(frame: pl.DataFrame, path: Path, analysis_date: str) -> None:
    partitioned = _with_bundle_day(frame, analysis_date)
    append_parquet_dataset(
        partitioned,
        path,
        partition_by="bundle_day",
        target_rows_per_file=100_000,
        file_prefix="mirror",
    )


def _with_bundle_day(frame: pl.DataFrame, analysis_date: str) -> pl.DataFrame:
    if frame.is_empty() or "bundle_day" in frame.columns:
        return frame
    candidates = [
        column for column in ["bundle_ts", "ingest_ts", "ts_utc", "ts"] if column in frame.columns
    ]
    if not candidates:
        return frame.with_columns(pl.lit(str(analysis_date)[:10]).alias("bundle_day"))
    expressions = [
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.to_datetime(time_zone="UTC", strict=False)
        for column in candidates
    ]
    return frame.with_columns(
        pl.coalesce(expressions)
        .dt.strftime("%Y-%m-%d")
        .fill_null(str(analysis_date)[:10])
        .alias("bundle_day")
    )


def _safe_read_dataset(path: Path) -> pl.DataFrame:
    try:
        return read_parquet_dataset(path)
    except Exception:
        return pl.DataFrame()


def _latest_bnb_paper_strategy_daily(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    working = frame
    if "paper_date" not in working.columns and "as_of_date" in working.columns:
        working = working.with_columns(
            pl.col("as_of_date").cast(pl.Utf8, strict=False).alias("paper_date")
        )
    if "paper_date" not in working.columns:
        return working
    key_columns = [column for column in ["strategy_id", "paper_date"] if column in working.columns]
    if not key_columns:
        return working
    if "bundle_ts" in working.columns:
        sort_column = "bundle_ts"
    elif "ingest_ts" in working.columns:
        sort_column = "ingest_ts"
    else:
        sort_column = None
    if sort_column is None:
        return working.unique(subset=key_columns, keep="last", maintain_order=True)
    normalized = _normalize_time(working, sort_column)
    return (
        normalized.sort([*key_columns, sort_column])
        .group_by(key_columns, maintain_order=True)
        .tail(1)
    )


def _build_candidate_labels_safely(lake_root: Path, analysis_date: str) -> None:
    try:
        from quant_lab.research.candidate_labels import build_and_publish_candidate_labels

        build_and_publish_candidate_labels(lake_root, as_of_date=analysis_date)
    except Exception:
        # V5 telemetry health must remain available even if candidate snapshots
        # or forward market bars are incomplete.
        return


def _build_alpha_discovery_board_safely(lake_root: Path, analysis_date: str) -> None:
    try:
        from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board

        build_and_publish_alpha_discovery_board(lake_root, as_of_date=analysis_date)
    except Exception:
        # Strategy telemetry analysis should not fail if the candidate research
        # board is waiting on forward labels or cost context.
        return


def _build_strategy_evidence_safely(lake_root: Path, analysis_date: str) -> None:
    try:
        from quant_lab.research.strategy_evidence import build_and_publish_strategy_evidence

        build_and_publish_strategy_evidence(lake_root, as_of_date=analysis_date)
    except Exception:
        # V5 telemetry health must remain available even if research evidence inputs
        # are incomplete or a future bundle adds an unexpected telemetry shape.
        return


def _build_entry_quality_safely(lake_root: Path, analysis_date: str) -> None:
    try:
        from quant_lab.research.entry_quality import build_and_publish_entry_quality

        build_and_publish_entry_quality(lake_root, as_of_date=analysis_date)
    except Exception:
        # Entry quality is advisory-only and must not break telemetry health.
        return


def _write_gold(
    lake_root: Path,
    result: V5TelemetryAnalysisResult,
    gate_violations: list[str],
    fallback_count: int,
    quant_lab_summary: dict[str, Any],
) -> None:
    base = result.model_dump()
    base["router_reason_top_json"] = json.dumps(result.router_reason_top, sort_keys=True)
    base["config_not_consumed_top_keys_json"] = json.dumps(
        result.config_not_consumed_top_keys,
        sort_keys=True,
    )
    base["warnings_json"] = json.dumps(result.warnings, sort_keys=True)
    base["critical_reasons_json"] = json.dumps(result.critical_reasons, sort_keys=True)
    base["next_actions_json"] = json.dumps(result.next_actions, sort_keys=True)
    base.pop("router_reason_top", None)
    base.pop("config_not_consumed_top_keys", None)
    base.pop("warnings", None)
    base.pop("critical_reasons", None)
    base.pop("next_actions", None)

    rows_by_dataset = {
        "strategy_health_daily": [base],
        "v5_execution_quality_daily": [base | {"fallback_count": fallback_count}],
        "v5_gate_compliance_daily": [
            base
            | {
                "violation_count": len(gate_violations),
                "violations_json": json.dumps(gate_violations),
            }
        ],
        "v5_missed_opportunity_daily": [base],
        "v5_config_health_daily": [base],
        "v5_issue_summary_daily": [base],
        "v5_quant_lab_mode_daily": [
            {
                "strategy": result.strategy,
                "date": result.date,
                "mode": quant_lab_summary["mode"],
                "permission_gate_enforced": quant_lab_summary["permission_gate_enforced"],
                "cost_gate_enforced": quant_lab_summary["cost_gate_enforced"],
                "usage_count": quant_lab_summary["usage_count"],
                "unique_request_count": quant_lab_summary["unique_request_count"],
                "unique_success_count": quant_lab_summary["unique_success_count"],
                "unique_error_count": quant_lab_summary["unique_error_count"],
                "unique_actual_fallback_count": quant_lab_summary[
                    "unique_actual_fallback_count"
                ],
                "request_success_count": quant_lab_summary["request_success_count"],
                "request_error_count": quant_lab_summary["request_error_count"],
                "actual_fallback_count": quant_lab_summary["actual_fallback_count"],
                "fallback_rate": quant_lab_summary["fallback_rate"],
                "degraded_reason": quant_lab_summary["degraded_reason"],
                "raw_imported_rows": quant_lab_summary["raw_imported_rows"],
                "unique_event_rows": quant_lab_summary["unique_event_rows"],
                "duplicate_event_count": quant_lab_summary["duplicate_event_count"],
                "duplicate_event_rows": quant_lab_summary["duplicate_event_rows"],
                "duplicate_rate": quant_lab_summary["duplicate_rate"],
                "exact_duplicate_event_rows": quant_lab_summary["exact_duplicate_event_rows"],
                "conflicting_duplicate_event_rows": quant_lab_summary[
                    "conflicting_duplicate_event_rows"
                ],
                "conflicting_duplicate_event_key_count": quant_lab_summary[
                    "conflicting_duplicate_event_key_count"
                ],
                "duplicate_explanation": quant_lab_summary["duplicate_explanation"],
                "first_seen_bundle_ts": quant_lab_summary["first_seen_bundle_ts"],
                "last_seen_bundle_ts": quant_lab_summary["last_seen_bundle_ts"],
                "cost_usage_count": quant_lab_summary["cost_usage_count"],
                "fallback_count": quant_lab_summary["fallback_count"],
                "latest_bundle_ts": result.latest_bundle_ts,
                "latest_bundle_sha256": result.latest_bundle_sha256,
                "created_at": datetime.now(UTC),
            }
        ],
        "v5_quant_lab_enforcement_daily": [
            {
                "strategy": result.strategy,
                "date": result.date,
                "mode": quant_lab_summary["mode"],
                "permission_gate_enforced": quant_lab_summary["permission_gate_enforced"],
                "cost_gate_enforced": quant_lab_summary["cost_gate_enforced"],
                "actual_violation_count": len(quant_lab_summary["actual_violations"]),
                "hypothetical_violation_count": len(
                    quant_lab_summary["hypothetical_violations"]
                ),
                "actual_violations_json": json.dumps(
                    quant_lab_summary["actual_violations"],
                    sort_keys=True,
                ),
                "hypothetical_violations_json": json.dumps(
                    quant_lab_summary["hypothetical_violations"],
                    sort_keys=True,
                ),
                "latest_bundle_ts": result.latest_bundle_ts,
                "latest_bundle_sha256": result.latest_bundle_sha256,
                "created_at": datetime.now(UTC),
            }
        ],
    }
    for name, rows in rows_by_dataset.items():
        upsert_parquet_dataset(
            pl.DataFrame(rows),
            lake_root / GOLD_DATASETS[name],
            key_columns=["strategy", "date"],
        )


def _latest_time(df: pl.DataFrame, column: str) -> datetime | None:
    if df.is_empty() or column not in df.columns:
        return None
    normalized = _normalize_time(df, column)
    value = normalized.select(pl.col(column).max()).item()
    return value.astimezone(UTC) if isinstance(value, datetime) else None


def _latest_value(df: pl.DataFrame, value_column: str, time_column: str) -> str | None:
    if df.is_empty() or value_column not in df.columns:
        return None
    if time_column in df.columns:
        df = _normalize_time(df, time_column).sort(time_column)
    return str(df.tail(1)[value_column][0])


def _normalize_time(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if df.schema.get(column) == pl.String:
        return df.with_columns(pl.col(column).str.to_datetime(time_zone="UTC", strict=False))
    return df.with_columns(pl.col(column).cast(pl.Datetime(time_zone="UTC")).alias(column))


def _latest_state_bool(df: pl.DataFrame, state_type: str, field: str) -> bool | None:
    row = _latest_state_row(df, state_type)
    if row is None:
        return None
    value = row.get(field)
    if isinstance(value, bool):
        return value
    payload = _payload(row)
    value = payload.get(field)
    return value if isinstance(value, bool) else None


def _latest_state_text(df: pl.DataFrame, state_type: str, field: str) -> str | None:
    row = _latest_state_row(df, state_type)
    if row is None:
        return None
    fields = [field]
    if field == "level":
        fields = ["level", "current_level", "risk_level"]
    payload = _payload(row)
    for candidate in fields:
        if row.get(candidate):
            return str(row[candidate])
        value = payload.get(candidate)
        if value is not None and str(value):
            return str(value)
    return None


def _latest_state_row(df: pl.DataFrame, state_type: str) -> dict[str, Any] | None:
    if df.is_empty() or "state_type" not in df.columns:
        return None
    filtered = df.filter(pl.col("state_type") == state_type)
    if filtered.is_empty():
        return None
    if "ingest_ts" in filtered.columns:
        filtered = filtered.sort("ingest_ts")
    return filtered.tail(1).to_dicts()[0]


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_payload_json")
    if not isinstance(raw, str):
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _window_summary_payload(df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty() or "raw_payload_json" not in df.columns:
        return {}
    candidates = df
    if "source_path_inside_bundle" in df.columns:
        candidates = df.filter(
            pl.col("source_path_inside_bundle").cast(pl.Utf8).str.contains(
                "summaries/window_summary.json",
                literal=True,
            )
        )
    if candidates.is_empty():
        return {}
    for time_column in ["bundle_ts", "ingest_ts"]:
        if time_column in candidates.columns:
            candidates = _normalize_time(candidates, time_column).sort(time_column)
            break
    return _payload(candidates.tail(1).to_dicts()[0])


def _count_rows(df: pl.DataFrame) -> int:
    return 0 if df.is_empty() else df.height


def _summary_int(payload: dict[str, Any], keys: list[str], *, fallback: int) -> int:
    for key in keys:
        value = payload.get(key)
        if value is None or value == "":
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return fallback


def _count_recent_rows(
    df: pl.DataFrame,
    analysis_date: str,
    *,
    hours: int,
    exclude_source_path: str | None = None,
) -> int:
    if df.is_empty():
        return 0
    filtered = df
    if exclude_source_path and "source_path_inside_bundle" in filtered.columns:
        filtered = filtered.filter(
            ~pl.col("source_path_inside_bundle").cast(pl.Utf8).str.contains(
                exclude_source_path,
                literal=True,
            )
        )
    if filtered.is_empty():
        return 0
    time_column = "bundle_ts" if "bundle_ts" in filtered.columns else "ingest_ts"
    if time_column not in filtered.columns:
        return filtered.height
    normalized = _normalize_time(filtered, time_column)
    window_end = _analysis_window_end(analysis_date)
    window_start = window_end - timedelta(hours=hours)
    return normalized.filter(
        (pl.col(time_column) >= window_start) & (pl.col(time_column) < window_end)
    ).height


def _analysis_window_end(analysis_date: str) -> datetime:
    try:
        return datetime.fromisoformat(analysis_date).replace(tzinfo=UTC) + timedelta(days=1)
    except ValueError:
        return datetime.now(UTC)


def _count_where(df: pl.DataFrame, column: str, value: str) -> int:
    if df.is_empty() or column not in df.columns:
        return 0
    return df.filter(pl.col(column).str.to_lowercase() == value).height


def _issue_messages(df: pl.DataFrame) -> list[str]:
    if df.is_empty():
        return []
    messages = []
    for row in df.to_dicts():
        messages.append(str(row.get("issue_type", "")))
        messages.append(str(row.get("message", "")))
        messages.append(str(row.get("raw_payload_json", "")))
    return messages


def _max_int(df: pl.DataFrame, column: str) -> int:
    if df.is_empty() or column not in df.columns:
        return 0
    value = df.select(pl.col(column).max()).item()
    return int(value or 0)


def _is_stale(latest_bundle_ts: datetime, analysis_date: str) -> bool:
    try:
        date_end = datetime.fromisoformat(analysis_date).replace(tzinfo=UTC) + timedelta(days=1)
    except ValueError:
        date_end = datetime.now(UTC)
    return latest_bundle_ts < date_end - timedelta(hours=24)


def _config_not_consumed_summary(df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty():
        return {"count": 0, "unknown": False, "top_keys": []}
    useful_columns = {
        "consumed",
        "status",
        "issue_type",
        "key",
        "config_key",
        "path",
        "name",
        "raw_payload_json",
        "blob",
    }
    if not useful_columns.intersection(df.columns):
        return {"count": 0, "unknown": True, "top_keys": []}

    not_consumed_expr = _config_not_consumed_expr(df)
    key_candidates = [
        _non_empty_utf8_expr(column)
        for column in ["key", "config_key", "path", "name"]
        if column in df.columns
    ]
    key_expr = (
        pl.coalesce(key_candidates + [pl.lit(None, dtype=pl.Utf8)])
        if key_candidates
        else pl.lit(None, dtype=pl.Utf8)
    )
    flagged = df.select(
        [
            not_consumed_expr.alias("__not_consumed"),
            key_expr.alias("__config_key"),
        ]
    ).filter(pl.col("__not_consumed"))
    if flagged.is_empty():
        return {"count": 0, "unknown": False, "top_keys": []}
    keyed = flagged.filter(pl.col("__config_key").is_not_null())
    keys = (
        sorted(
            str(value)
            for value in keyed.select(pl.col("__config_key").unique()).to_series().to_list()
            if value is not None and str(value).strip()
        )
        if not keyed.is_empty()
        else []
    )
    unknown = keyed.height < flagged.height
    return {"count": len(keys), "unknown": unknown, "top_keys": keys[:20]}


def _config_not_consumed_expr(df: pl.DataFrame) -> pl.Expr:
    markers = ("not_consumed", "not consumed", "unused", "not-used")
    expressions: list[pl.Expr] = []
    if "consumed" in df.columns:
        consumed_text = _non_empty_utf8_expr("consumed").str.to_lowercase()
        expressions.append(consumed_text.is_in(["false", "0", "no"]))
        if df.schema.get("consumed") == pl.Boolean:
            expressions.append(pl.col("consumed").fill_null(True).not_())
    for column in ["status", "issue_type", "raw_payload_json", "blob"]:
        if column not in df.columns:
            continue
        text = _non_empty_utf8_expr(column).str.to_lowercase()
        marker_expr = pl.lit(False)
        for marker in markers:
            marker_expr = marker_expr | text.str.contains(marker, literal=True)
        expressions.append(marker_expr)
    if not expressions:
        return pl.lit(False)
    combined = expressions[0]
    for expression in expressions[1:]:
        combined = combined | expression
    return combined.fill_null(False)


def _non_empty_utf8_expr(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .replace("", None)
    )


def _first_text(
    row: dict[str, Any],
    payload: dict[str, Any],
    fields: list[str],
) -> str | None:
    for field in fields:
        value = row.get(field)
        if value is None:
            value = _nested_value(payload, field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _nested_value(payload: dict[str, Any], field: str) -> Any:
    value: Any = payload
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def _row_indicates_not_consumed(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    consumed = row.get("consumed", payload.get("consumed"))
    if isinstance(consumed, bool):
        return not consumed
    if consumed is not None and str(consumed).strip().lower() in {"false", "0", "no"}:
        return True
    status = str(row.get("status", payload.get("status", ""))).strip().lower()
    issue_type = str(row.get("issue_type", payload.get("issue_type", ""))).strip().lower()
    markers = ("not_consumed", "not consumed", "unused", "not-used")
    return any(marker in status or marker in issue_type for marker in markers)


def _gate_compliance_violations(decisions: pl.DataFrame, trades: pl.DataFrame) -> list[str]:
    if decisions.is_empty() or trades.is_empty():
        return []
    text = "\n".join(str(row.get("raw_payload_json", "")) for row in decisions.to_dicts()).lower()
    violations = []
    if '"permission": "abort"' in text or '"permission":"abort"' in text:
        violations.append("quant_lab ABORT permission with trade events present")
    if '"permission": "sell_only"' in text or '"permission":"sell_only"' in text:
        violations.append("quant_lab SELL_ONLY permission with trade events present")
    return violations


def _quant_lab_mode_summary(
    usage: pl.DataFrame,
    requests: pl.DataFrame,
    compliance: pl.DataFrame,
    cost_usage: pl.DataFrame,
    fallback: pl.DataFrame,
    decisions: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    analysis_date: str | None = None,
) -> dict[str, Any]:
    has_data = any(
        not frame.is_empty() for frame in [usage, requests, compliance, cost_usage, fallback]
    )
    mode = _latest_mode(usage, compliance, decisions)
    enforced = _gate_enforced(
        [usage, compliance, decisions],
        mode,
        ["permission_gate_enforced", "apply_permission_gate"],
    )
    cost_enforced = _gate_enforced(
        [usage, compliance, decisions, cost_usage],
        mode,
        ["cost_gate_enforced", "apply_cost_gate"],
    )
    actual: list[str] = []
    hypothetical: list[str] = []
    request_health = _quant_lab_request_health(
        requests,
        fallback,
        analysis_date=analysis_date,
    )
    compliance_health = _analysis_day_frame(compliance, analysis_date)
    trades_health = _analysis_day_frame(trades, analysis_date)
    if not compliance_health.is_empty():
        actual, hypothetical = _compliance_violation_messages(
            compliance_health,
            mode=mode,
            enforced=enforced,
        )
    elif (
        not trades_health.is_empty()
        and mode not in {None, "local_only", "cost_only"}
        and _mode_allows_actual_permission_check(mode)
    ):
        message = f"V5 trade events present without quant_lab compliance rows in mode={mode}"
        if mode == "shadow" or not enforced:
            hypothetical.append(message)
        else:
            actual.append(message)
    return {
        "has_data": has_data,
        "mode": mode or "unknown",
        "permission_gate_enforced": enforced,
        "cost_gate_enforced": cost_enforced,
        "usage_count": usage.height,
        **request_health,
        "cost_usage_count": cost_usage.height,
        "fallback_count": request_health["actual_fallback_count"],
        "latest_permission_status": request_health["latest_permission_status"],
        "stale_permission_consecutive_count": request_health[
            "stale_permission_consecutive_count"
        ],
        "actual_violations": sorted(set(actual)),
        "hypothetical_violations": sorted(set(hypothetical)),
    }


def _analysis_day_frame(frame: pl.DataFrame, analysis_date: str | None) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    rows = _analysis_day_event_rows(frame.to_dicts(), analysis_date)
    return pl.DataFrame(rows) if rows else frame.head(0)


def _quant_lab_request_health(
    requests: pl.DataFrame,
    fallback: pl.DataFrame,
    *,
    analysis_date: str | None = None,
) -> dict[str, Any]:
    success_count = 0
    error_count = 0
    request_fallback_keys: set[str] = set()
    all_request_rows = _unique_event_rows(requests)
    all_fallback_rows = _unique_event_rows(fallback)
    request_rows = _analysis_day_event_rows(all_request_rows.values(), analysis_date)
    fallback_rows = _analysis_day_event_rows(all_fallback_rows.values(), analysis_date)
    for row in request_rows:
        payload = _payload(row)
        status = _request_status(row, payload)
        if status == "success":
            success_count += 1
        elif status == "fallback":
            error_count += 1
            request_fallback_keys.add(_event_key(row))
        elif status == "error":
            error_count += 1
    fallback_keys = {
        _event_key(row)
        for row in fallback_rows
        if _fallback_table_row_is_actual(row, _payload(row))
    }
    actual_fallback_count = len(request_fallback_keys | fallback_keys)
    total_requests = success_count + error_count
    fallback_rate = actual_fallback_count / total_requests if total_requests else 0.0
    raw_imported_rows = _raw_event_row_count(requests) + _raw_event_row_count(fallback)
    unique_event_keys = set(all_request_rows) | set(all_fallback_rows)
    unique_event_rows = len(unique_event_keys)
    duplicate_event_rows = max(raw_imported_rows - unique_event_rows, 0)
    duplicate_rate = duplicate_event_rows / raw_imported_rows if raw_imported_rows else 0.0
    duplicate_breakdown = _duplicate_event_breakdown(
        [*all_request_rows.values(), *all_fallback_rows.values()],
        duplicate_event_rows=duplicate_event_rows,
    )
    first_seen, last_seen = _event_seen_range(
        list(all_request_rows.values()) + list(all_fallback_rows.values())
    )
    if actual_fallback_count:
        degraded_reason = "actual_fallback_present"
    elif error_count:
        degraded_reason = "request_errors_present"
    else:
        degraded_reason = "none"
    latest_permission_status, stale_permission_consecutive_count = (
        _permission_status_consecutive_health(request_rows)
    )
    return {
        "unique_request_count": total_requests,
        "unique_success_count": success_count,
        "unique_error_count": error_count,
        "unique_actual_fallback_count": actual_fallback_count,
        "request_success_count": success_count,
        "request_error_count": error_count,
        "actual_fallback_count": actual_fallback_count,
        "fallback_rate": fallback_rate,
        "degraded_reason": degraded_reason,
        "raw_imported_rows": raw_imported_rows,
        "unique_event_rows": unique_event_rows,
        "duplicate_event_count": duplicate_event_rows,
        "duplicate_event_rows": duplicate_event_rows,
        "duplicate_rate": duplicate_rate,
        "exact_duplicate_event_rows": duplicate_breakdown["exact_duplicate_event_rows"],
        "conflicting_duplicate_event_rows": duplicate_breakdown[
            "conflicting_duplicate_event_rows"
        ],
        "conflicting_duplicate_event_key_count": duplicate_breakdown[
            "conflicting_duplicate_event_key_count"
        ],
        "duplicate_explanation": duplicate_breakdown["duplicate_explanation"],
        "first_seen_bundle_ts": first_seen,
        "last_seen_bundle_ts": last_seen,
        "latest_permission_status": latest_permission_status,
        "stale_permission_consecutive_count": stale_permission_consecutive_count,
    }


def _analysis_day_event_rows(
    rows: Iterable[dict[str, Any]],
    analysis_date: str | None,
) -> list[dict[str, Any]]:
    if analysis_date is None:
        return list(rows)
    try:
        day = datetime.fromisoformat(str(analysis_date)[:10]).date()
    except ValueError:
        return list(rows)
    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    end = start + timedelta(days=1)
    return [
        row
        for row in rows
        if start <= _request_event_time(row) < end
    ]


def _duplicate_event_breakdown(
    rows: list[dict[str, Any]],
    *,
    duplicate_event_rows: int,
) -> dict[str, Any]:
    conflicting_rows = 0
    conflicting_keys = 0
    for row in rows:
        current_source_count = _row_current_source_count(row)
        if (
            current_source_count > 1
            and (_boolish(row.get("conflicting_duplicate")) or _payload_hash_count(row) > 1)
        ):
            conflicting_keys += 1
            conflicting_rows += max(current_source_count - 1, 0)
    exact_rows = max(duplicate_event_rows - conflicting_rows, 0)
    explanation = (
        "conflicting_duplicate_payloads_present"
        if conflicting_rows
        else "exact_duplicate_reingest_or_overlapping_followup_bundle"
        if duplicate_event_rows
        else "none"
    )
    return {
        "exact_duplicate_event_rows": exact_rows,
        "conflicting_duplicate_event_rows": conflicting_rows,
        "conflicting_duplicate_event_key_count": conflicting_keys,
        "duplicate_explanation": explanation,
    }


def _payload_hash_count(row: dict[str, Any]) -> int:
    value = row.get("payload_hash_count")
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    raw = row.get("payload_hashes_json")
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = []
        if isinstance(payload, list):
            return len({str(item) for item in payload if str(item).strip()})
    return 1


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _unique_event_rows(df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    if df.is_empty():
        return {}
    rows_by_key: dict[str, dict[str, Any]] = {}
    for row in df.to_dicts():
        key = _event_key(row)
        current = rows_by_key.get(key)
        if current is None or _row_seen_sort_value(row) >= _row_seen_sort_value(current):
            rows_by_key[key] = row
    return rows_by_key


def _permission_status_consecutive_health(
    rows: Any,
) -> tuple[str | None, int]:
    ordered = sorted(list(rows), key=_request_event_time)
    latest_status: str | None = None
    consecutive = 0
    for row in ordered:
        payload = _payload(row)
        status = _request_permission_status(row, payload)
        if not status:
            continue
        latest_status = status
        if _permission_status_is_stale_or_expired(status):
            consecutive += 1
        else:
            consecutive = 0
    return latest_status, consecutive


def _request_permission_status(row: dict[str, Any], payload: dict[str, Any]) -> str | None:
    value = _first_value(
        row,
        payload,
        [
            "permission_status",
            "remote_permission_status",
            "risk_permission_status",
            "response.permission_status",
            "response_json.permission_status",
            "response_body.permission_status",
            "body.permission_status",
            "json.permission_status",
            "quant_lab.permission_status",
            "risk_permission.permission_status",
        ],
    )
    if value is None:
        return None
    status = str(value).strip().upper()
    return status or None


def _permission_status_is_stale_or_expired(status: str | None) -> bool:
    normalized = str(status or "").strip().upper()
    return (
        normalized.startswith("EXPIRED_")
        or normalized.startswith("STALE_")
        or normalized == "NO_FRESH_PERMISSION"
    )


def _request_event_time(row: dict[str, Any]) -> datetime:
    payload = _payload(row)
    for field in [
        "ts_utc",
        "ts",
        "timestamp",
        "request_ts",
        "created_at",
        "bundle_ts",
        "ingest_ts",
    ]:
        value = _first_value(row, payload, [field])
        parsed = _parse_seen_time(value)
        if parsed is not None:
            return parsed
    return _row_seen_sort_value(row)


def _event_key(row: dict[str, Any]) -> str:
    payload = _payload(row)
    fields = _event_key_fields(row, payload)
    event_id = str(fields.get("event_id") or "").strip()
    if event_id:
        fields = {
            "strategy_id": str(fields.get("strategy_id") or "").strip(),
            "event_id": event_id,
        }
    elif (
        fields.get("endpoint_path")
        and fields.get("ts_utc")
        and fields.get("error_type")
    ):
        fields.pop("run_id", None)
        fields.pop("fallback_used", None)
    fields.pop("raw_payload_hash", None)
    rendered = json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _event_key_fields(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    source_path = _logical_source_path(str(row.get("source_path_inside_bundle") or ""))
    symbol = _event_clean_text(
        _first_value(
            row,
            payload,
            ["symbol", "normalized_symbol", "inst_id", "instId", "instrument", "pair"],
        )
    )
    strategy_id = _event_clean_text(
        _first_value(row, payload, ["strategy_id", "strategyId", "strategy"])
        or row.get("strategy")
    )
    event_id = _event_clean_text(
        _first_value(row, payload, ["event_id", "eventId", "source_event_id"])
    )
    status_code = _request_status_code(row, payload)
    endpoint_path = str(
        _first_value(
            row,
            payload,
            ["endpoint_path", "endpoint", "path", "url", "route", "api_path", "request_path"],
        )
        or ""
    ).strip()
    fields = {
        "event_id": event_id,
        "strategy_id": strategy_id,
        "run_id": str(_first_value(row, payload, ["run_id", "runId", "run"]) or ""),
        "event_type": _analysis_event_type(row, payload, source_path),
        "endpoint_path": endpoint_path,
        "ts_utc": _normalize_event_time(
            _first_value(
                row,
                payload,
                [
                    "ts_utc",
                    "ts",
                    "timestamp",
                    "created_at",
                    "time",
                    "request_ts",
                    "event_ts",
                ],
            )
        ),
        "status_code": "" if status_code is None else str(status_code),
        "error_type": str(
            _first_value(row, payload, ["error_type", "exception_type", "error", "exception"])
            or ""
        ).strip(),
        "request_id": str(
            _first_value(row, payload, ["request_id", "trace_id", "id", "uuid"]) or ""
        ).strip(),
        "symbol": normalize_symbol(symbol) if symbol else "",
        "side": _event_clean_text(_first_value(row, payload, ["side", "order_side"])).lower(),
        "intent": _event_clean_text(
            _first_value(row, payload, ["intent", "action", "router_intent"])
        ).lower(),
        "fallback_used": _first_bool(
            row,
            payload,
            ["fallback_used", "used_fallback", "local_fallback"],
        ),
    }
    fields["raw_payload_hash"] = _analysis_raw_payload_hash(row, payload, fields)
    return {key: value for key, value in fields.items() if value not in {"", None}}


def _analysis_raw_payload_hash(
    row: dict[str, Any],
    payload: dict[str, Any],
    fields: dict[str, Any],
) -> str:
    del fields
    source: Any = payload if payload else row
    if isinstance(source, dict):
        source = {
            key: _normalize_analysis_payload_conflict_value(value)
            for key, value in source.items()
            if key not in _ANALYSIS_EVENT_PAYLOAD_IDENTITY_ALIASES
            and not _analysis_payload_conflict_value_is_empty(value)
        }
    rendered = json.dumps(source, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


_ANALYSIS_EVENT_PAYLOAD_IDENTITY_ALIASES = {
    "event_id",
    "eventId",
    "source_event_id",
    "strategy_id",
    "strategyId",
    "strategy",
    "run_id",
    "runId",
    "run",
    "event_type",
    "type",
    "kind",
    "endpoint",
    "endpoint_path",
    "path",
    "url",
    "route",
    "api_path",
    "request_path",
    "source_path_inside_bundle",
    "bundle_sha256",
    "bundle_name",
    "bundle_ts",
    "ingest_ts",
    "schema_version",
    "row_index",
    "source_count",
    "last_seen_source_count",
    "first_seen_bundle_ts",
    "last_seen_bundle_ts",
    "ts_utc",
    "ts",
    "timestamp",
    "created_at",
    "time",
    "request_ts",
    "event_ts",
    "status_code",
    "error_type",
    "exception_type",
    "request_id",
    "trace_id",
    "id",
    "uuid",
    "symbol",
    "normalized_symbol",
    "inst_id",
    "instId",
    "instrument",
    "pair",
    "side",
    "order_side",
    "intent",
    "action",
    "router_intent",
    "fallback_used",
    "used_fallback",
    "local_fallback",
    "raw_payload_hash",
    "payload_hash",
    "payload_hashes_json",
    "payload_hash_count",
    "conflicting_duplicate",
    "event_key",
    "event_key_fields_json",
}


def _analysis_payload_conflict_value_is_empty(value: Any) -> bool:
    if value is None:
        return True
    rendered = str(value).strip().lower()
    return rendered in {"", "not_observable", "not-observable", "none", "null", "nan"}


def _normalize_analysis_payload_conflict_value(value: Any) -> Any:
    if isinstance(value, str):
        rendered = value.strip()
        lowered = rendered.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return rendered
    if isinstance(value, list):
        return [_normalize_analysis_payload_conflict_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_analysis_payload_conflict_value(item)
            for key, item in value.items()
            if not _analysis_payload_conflict_value_is_empty(item)
        }
    return value


def _analysis_event_type(
    row: dict[str, Any],
    payload: dict[str, Any],
    source_path: str,
) -> str:
    if source_path == "summaries/quant_lab_fallbacks.csv":
        return "request"
    value = _first_value(row, payload, ["event_type", "type", "kind"])
    if value is None and _first_value(row, payload, ["endpoint_path", "endpoint", "path", "url"]):
        value = "request"
    return str(value or "event").strip().lower()


def _event_clean_text(value: Any) -> str:
    if value is None:
        return ""
    rendered = str(value).strip()
    return (
        ""
        if rendered.lower()
        in {"none", "null", "nan", "unknown", "not_observable", "not-observable", "n/a", "na"}
        else rendered
    )


def _normalize_event_time(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")
    rendered = str(value).strip()
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError:
        return rendered
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _raw_event_row_count(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    total = 0
    for row in df.to_dicts():
        total += _row_current_source_count(row)
    return total


def _row_current_source_count(row: dict[str, Any]) -> int:
    value = row.get("last_seen_source_count")
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    first_seen = _parse_seen_time(row.get("first_seen_bundle_ts") or row.get("bundle_ts"))
    last_seen = _parse_seen_time(row.get("last_seen_bundle_ts") or row.get("bundle_ts"))
    if first_seen is not None and last_seen is not None and first_seen != last_seen:
        return 1
    return _row_source_count(row)


def _row_source_count(row: dict[str, Any]) -> int:
    value = row.get("source_count")
    try:
        return max(int(float(value)), 1)
    except (TypeError, ValueError):
        return 1


def _event_seen_range(rows: list[dict[str, Any]]) -> tuple[datetime | None, datetime | None]:
    seen_values = [
        value
        for row in rows
        for value in [
            row.get("first_seen_bundle_ts") or row.get("bundle_ts"),
            row.get("last_seen_bundle_ts") or row.get("bundle_ts"),
        ]
        if value is not None and value != ""
    ]
    parsed = [_parse_seen_time(value) for value in seen_values]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return None, None
    return min(parsed), max(parsed)


def _row_seen_sort_value(row: dict[str, Any]) -> datetime:
    return (
        _parse_seen_time(row.get("last_seen_bundle_ts"))
        or _parse_seen_time(row.get("bundle_ts"))
        or _parse_seen_time(row.get("ingest_ts"))
        or datetime.min.replace(tzinfo=UTC)
    )


def _parse_seen_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if value is None or value == "":
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _request_status(row: dict[str, Any], payload: dict[str, Any]) -> str:
    if _request_is_success(row, payload):
        return "success"
    if _request_is_actual_fallback(row, payload):
        return "fallback"
    if _request_is_error(row, payload):
        return "error"
    return "success"


def _request_is_success(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    if _request_fallback_used(row, payload):
        return False
    status_code = _request_status_code(row, payload)
    success = _first_bool(row, payload, ["success", "ok", "request_ok"])
    if status_code == 200 and success is not False:
        return True
    return success is True and (status_code is None or 200 <= status_code < 300)


def _request_is_error(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    status_code = _request_status_code(row, payload)
    if status_code is not None and status_code >= 400:
        return True
    success = _first_bool(row, payload, ["success", "ok", "request_ok"])
    if success is False:
        return True
    return _actual_error_text(row, payload)


def _request_is_actual_fallback(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    if _request_is_success(row, payload):
        return False
    if _request_fallback_used(row, payload):
        return True
    status_code = _request_status_code(row, payload)
    if status_code is not None and status_code >= 500:
        return True
    error_type = _first_value(row, payload, ["error_type", "exception_type"])
    if _actual_error_type(error_type):
        return True
    return _actual_error_text(row, payload)


def _request_fallback_used(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    return _first_bool(
        row,
        payload,
        ["fallback_used", "used_fallback", "local_fallback"],
    ) is True


def _request_status_code(row: dict[str, Any], payload: dict[str, Any]) -> int | None:
    value = _first_value(row, payload, ["status_code", "http_status", "status"])
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _actual_error_type(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in {
        "",
        "none",
        "null",
        "ok",
        "false",
        "0",
        "http_200",
        "200",
        "request_not_ok",
    }:
        return False
    if normalized.startswith("http_4") or normalized in {
        "400",
        "401",
        "403",
        "404",
        "409",
        "422",
        "429",
    }:
        return False
    return True


def _actual_error_text(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    rendered = " ".join(
        str(_first_value(row, payload, [field]) or "").lower()
        for field in [
            "error",
            "message",
            "exception",
            "diagnosis",
            "reason",
            "error_type",
            "exception_type",
        ]
    )
    if "http_200" in rendered or "http 200" in rendered:
        rendered = rendered.replace("request_not_ok", "")
    return any(
        marker in rendered
        for marker in ["timeout", "quantlabtimeout", "connection", "parse", "jsondecode"]
    )


def _non_request_fallback_count(fallback: pl.DataFrame, *, skip_request_sources: bool) -> int:
    if fallback.is_empty() or "source_path_inside_bundle" not in fallback.columns:
        return sum(
            1
            for row in fallback.to_dicts()
            if _fallback_table_row_is_actual(row, _payload(row))
        )
    count = 0
    for row in fallback.to_dicts():
        source_path = str(row.get("source_path_inside_bundle") or "")
        logical = _logical_source_path(source_path)
        if skip_request_sources and logical in {
            "raw/reports/quant_lab_requests.jsonl",
            "raw/quant_lab/quant_lab_requests.jsonl",
            "reports/quant_lab_requests.jsonl",
            "summaries/quant_lab_fallbacks.csv",
        }:
            continue
        payload = _payload(row)
        if _fallback_table_row_is_actual(row, payload):
            count += 1
    return count


def _fallback_table_row_is_actual(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    if _request_is_success(row, payload):
        return False
    if _request_is_actual_fallback(row, payload):
        return True
    count_value = _first_value(row, payload, ["count", "fallback_count"])
    try:
        return float(count_value) > 0
    except (TypeError, ValueError):
        return False


def _logical_source_path(source_path: str) -> str:
    parts = [part for part in source_path.split("/") if part]
    if len(parts) > 1 and parts[0].startswith("v5_live_followup_bundle_"):
        return "/".join(parts[1:])
    return source_path


def _compliance_violation_messages(
    frame: pl.DataFrame,
    *,
    mode: str | None,
    enforced: bool,
) -> tuple[list[str], list[str]]:
    if frame.is_empty():
        return [], []
    actual_messages: set[str] = set()
    hypothetical_messages: set[str] = set()

    explicit_actual = _boolean_series(frame, ["actual_violation"])
    explicit_hypothetical = _boolean_series(frame, ["hypothetical_violation"])
    if explicit_actual is not None and bool(explicit_actual.any()):
        message = _violation_message("explicit", mode)
        if mode != "shadow" and _mode_allows_actual_permission_check(mode):
            actual_messages.add(message)
        else:
            hypothetical_messages.add(message)
    if explicit_hypothetical is not None and bool(explicit_hypothetical.any()):
        hypothetical_messages.add(_violation_message("explicit", mode))

    permission = _coalesced_text_series(
        frame,
        [
            "permission",
            "quant_lab_permission",
            "raw_permission_decision",
            "raw_permission_status",
            "effective_permission_decision",
            "effective_permission_status",
            "permission_decision",
            "effective_decision",
        ],
    )
    if permission is None:
        return sorted(actual_messages), sorted(hypothetical_messages)

    risky = _risk_increasing_series(frame)
    permission_upper = permission.str.to_uppercase()
    permission_blocked = permission_upper.str.contains("ABORT") | permission_upper.str.contains(
        "SELL_ONLY"
    )
    blocked = frame.select((permission_blocked & risky).alias("blocked"))["blocked"]
    if not bool(blocked.any()):
        return sorted(actual_messages), sorted(hypothetical_messages)

    blocked_permissions = (
        frame.with_columns(permission_upper.alias("__permission"))
        .filter(permission_blocked & risky)
        .select(pl.col("__permission").unique())
        .to_series()
        .to_list()
    )
    target = (
        hypothetical_messages
        if mode == "shadow" or not enforced or not _mode_allows_actual_permission_check(mode)
        else actual_messages
    )
    for value in blocked_permissions:
        normalized = "SELL_ONLY" if "SELL_ONLY" in str(value) else "ABORT"
        target.add(_violation_message(normalized, mode))
    return sorted(actual_messages), sorted(hypothetical_messages)


def _violation_message(permission: str, mode: str | None) -> str:
    return (
        f"quant_lab {str(permission).upper()} permission with risk-increasing "
        f"V5 action in mode={mode or 'unknown'}"
    )


def _boolean_series(frame: pl.DataFrame, columns: list[str]) -> pl.Series | None:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return None
    exprs = [_bool_expr(column) for column in existing]
    combined = exprs[0]
    for expr in exprs[1:]:
        combined = combined | expr
    return frame.select(combined.fill_null(False).alias("__bool"))["__bool"]


def _bool_expr(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
        .is_in(["1", "true", "yes", "y", "on"])
    )


def _coalesced_text_series(frame: pl.DataFrame, columns: list[str]) -> pl.Series | None:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return None
    exprs = [
        pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
        for column in existing
    ]
    return frame.select(pl.coalesce(exprs).fill_null("").alias("__text"))["__text"]


def _risk_increasing_series(frame: pl.DataFrame) -> pl.Series:
    reduce_only = _boolean_series(frame, ["reduce_only", "is_reduce_only"])
    new_risk = _boolean_series(frame, ["new_risk", "risk_increasing"])
    side = _coalesced_text_series(frame, ["side", "order_side"])
    intent = _coalesced_text_series(frame, ["intent", "action", "router_intent"])
    order_decision = _coalesced_text_series(frame, ["order_decision", "effective_decision"])

    if new_risk is None:
        risk = pl.Series("__risk", [False] * frame.height)
    else:
        risk = new_risk
    for values in [side, intent, order_decision]:
        if values is None:
            continue
        lowered = values.str.to_lowercase()
        risk = risk | lowered.is_in(
            ["buy", "long", "open_long", "increase", "risk_increase", "enter"]
        )
        risk = risk & ~lowered.is_in(["reduce", "close", "close_long", "sell_only_reduce", "exit"])
    if reduce_only is not None:
        risk = risk & ~reduce_only
    return risk


def _latest_mode(*frames: pl.DataFrame) -> str | None:
    allowed = {"local_only", "shadow", "cost_only", "permission_only", "enforce"}
    for frame in frames:
        if frame.is_empty():
            continue
        for row in reversed(
            _recent_relevant_rows(
                frame,
                ["mode", "quant_lab_mode", "quant_lab.mode", "quant_lab.quant_lab_mode"],
            )
        ):
            payload = _payload(row)
            value = _first_text(
                row,
                payload,
                ["mode", "quant_lab_mode", "quant_lab.mode", "quant_lab.quant_lab_mode"],
            )
            normalized = str(value or "").strip().lower()
            if normalized in allowed:
                return normalized
    return None


def _gate_enforced(
    frames: list[pl.DataFrame],
    mode: str | None,
    field_names: list[str],
) -> bool:
    is_cost_gate = "cost_gate_enforced" in field_names or "apply_cost_gate" in field_names
    if mode in {"shadow", "local_only"} or (mode == "cost_only" and not is_cost_gate):
        return False
    expanded_fields = [
        field
        for name in field_names
        for field in [name, f"quant_lab.{name}"]
    ]
    for frame in frames:
        if frame.is_empty():
            continue
        for row in reversed(_recent_relevant_rows(frame, expanded_fields)):
            payload = _payload(row)
            value = _first_value(row, payload, expanded_fields)
            if value is None and "permission" in field_names[0]:
                value = _first_value(row, payload, ["enforced", "quant_lab.enforced"])
            parsed = _parse_bool(value)
            if parsed is not None:
                return parsed
    if is_cost_gate:
        return mode in {"enforce", "cost_only"}
    return mode in {"enforce", "permission_only"}


def _recent_relevant_rows(
    frame: pl.DataFrame,
    fields: list[str],
    *,
    limit: int = 500,
) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    root_fields = {field.split(".", maxsplit=1)[0] for field in fields}
    needed = [
        column
        for column in [
            *sorted(root_fields),
            "raw_payload_json",
            "raw_json",
            "ingest_ts",
            "bundle_ts",
        ]
        if column in frame.columns
    ]
    if not needed:
        return []
    candidates = frame.select(needed)
    scalar_fields = [column for column in root_fields if column in candidates.columns]
    mask: pl.Expr | None = None
    for column in scalar_fields:
        expr = (
            pl.col(column).is_not_null()
            & (pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars() != "")
            & (
                ~pl.col(column)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .str.to_lowercase()
                .is_in(["none", "null", "nan", "unknown", "not_observable"])
            )
        )
        mask = expr if mask is None else mask | expr
    if mask is not None:
        filtered = candidates.filter(mask)
        if not filtered.is_empty():
            candidates = filtered
    for time_column in ["ingest_ts", "bundle_ts"]:
        if time_column in candidates.columns:
            candidates = _normalize_time(candidates, time_column).sort(time_column)
            break
    return candidates.tail(max(limit, 1)).to_dicts()


def _first_value(row: dict[str, Any], payload: dict[str, Any], fields: list[str]) -> Any:
    raw_payload: dict[str, Any] | None = None
    for field in fields:
        value = row.get(field)
        if value is None:
            value = _nested_value(payload, field)
        if value is None:
            if raw_payload is None:
                raw_payload = _raw_json_payload(row, payload)
            value = _nested_value(raw_payload, field)
        if value is not None and str(value).strip():
            return value
    return None


def _mode_allows_actual_permission_check(mode: str | None) -> bool:
    return mode in {"enforce", "permission_only"}


def _first_bool(row: dict[str, Any], payload: dict[str, Any], fields: list[str]) -> bool | None:
    raw_payload: dict[str, Any] | None = None
    for field in fields:
        if field in row:
            parsed = _parse_bool(row.get(field))
            if parsed is not None:
                return parsed
        value = _nested_value(payload, field)
        if value is not None:
            parsed = _parse_bool(value)
            if parsed is not None:
                return parsed
        if raw_payload is None:
            raw_payload = _raw_json_payload(row, payload)
        value = _nested_value(raw_payload, field)
        if value is not None:
            parsed = _parse_bool(value)
            if parsed is not None:
                return parsed
    return None


def _raw_json_payload(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_json")
    if raw is None or not str(raw).strip():
        raw = payload.get("raw_json")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _is_risk_increasing_side(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"buy", "long", "open_long", "increase", "risk_increase", "enter"}


def _row_is_risk_increasing(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    reduce_only = _first_bool(row, payload, ["reduce_only", "is_reduce_only"])
    if reduce_only is True:
        return False
    new_risk = _first_bool(row, payload, ["new_risk", "risk_increasing"])
    if new_risk is not None:
        return new_risk
    intent = _first_text(row, payload, ["intent", "action", "router_intent"])
    if intent:
        normalized = intent.strip().lower()
        if normalized in {"reduce", "close", "close_long", "sell_only_reduce", "exit"}:
            return False
        if normalized in {"buy", "long", "open_long", "increase", "risk_increase", "enter"}:
            return True
    side = _first_text(row, payload, ["side", "order_side"])
    return _is_risk_increasing_side(side)


def _router_reason_top(df: pl.DataFrame) -> list[dict[str, Any]]:
    if df.is_empty():
        return []
    reason_column = next(
        (
            column
            for column in ["reason", "router_reason", "decision_reason"]
            if column in df.columns
        ),
        None,
    )
    if reason_column is None:
        return []
    return (
        df.group_by(reason_column)
        .len(name="count")
        .sort("count", descending=True)
        .head(10)
        .rename({reason_column: "reason"})
        .to_dicts()
    )


def _fallback_count(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    text = "\n".join(str(row.get("raw_payload_json", "")) for row in df.to_dicts()).lower()
    return text.count("fallback")


def _dust_count(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    text = "\n".join(json.dumps(row, sort_keys=True, default=str).lower() for row in df.to_dicts())
    return text.count("dust")


def _matured_count(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    count = _truthy_column_count(df, ["matured", "is_matured", "maturity_reached"])
    if count is not None:
        return count
    return _marker_text_row_count(df, ["matured", "maturity_reached"])


def _profitable_count(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    count = _truthy_column_count(df, ["profitable", "is_profitable"])
    if count is not None:
        return count
    return _marker_text_row_count(df, ["profitable"])


def _marker_text_row_count(df: pl.DataFrame, markers: list[str]) -> int:
    text_columns = [
        column
        for column in ["raw_payload_json", "message", "issue_type", "type", "status"]
        if column in df.columns
    ]
    if not text_columns:
        return 0
    marker_expr = pl.lit(False)
    for column in text_columns:
        text = (
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.to_lowercase()
        )
        for marker in markers:
            marker_expr = marker_expr | text.str.contains(marker, literal=True)
    return int(df.select(marker_expr.sum()).item() or 0)


def _truthy_column_count(df: pl.DataFrame, columns: list[str]) -> int | None:
    for column in columns:
        if column not in df.columns:
            continue
        if df.schema.get(column) == pl.Boolean:
            return int(df.select(pl.col(column).fill_null(False).sum()).item() or 0)
        truthy = (
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.to_lowercase()
            .is_in(["true", "1", "yes", "y"])
        )
        return int(df.select(truthy.sum()).item() or 0)
    return None


def _high_score_blocked_matured_issue_count(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    count = 0
    for row in df.to_dicts():
        text = " ".join(
            str(row.get(column, ""))
            for column in ["issue_type", "type", "message", "raw_payload_json"]
        ).lower()
        if "high_score_blocked_matured_without_label" in text:
            count += 1
    return count
