from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import polars as pl

from quant_lab.symbols import normalize_symbol

SYSTEM_ACCEPTANCE_FIELDS = [
    "check_name",
    "status",
    "observed_value",
    "expected_value",
    "owner",
    "next_action",
]

NO_TRIGGER_REASON_FIELDS = [
    "report_name",
    "generated_at",
    "source_row_count",
    "output_row_count",
    "missing_field_reason",
    "filtered_out_reason",
    "next_action",
    "live_order_effect",
]


def build_system_acceptance_dashboard(
    *,
    frames: Mapping[str, pl.DataFrame],
    report_frames: Mapping[str, pl.DataFrame],
    row_counts: Mapping[str, int],
    pre_export_v5: Mapping[str, Any] | None,
    data_quality_warnings: list[str] | tuple[str, ...],
    api_latency_summary: pl.DataFrame,
    lake_file_count: int | None,
    lake_file_growth_24h_count: int | None = None,
    lake_file_health: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    checks: list[dict[str, Any]] = []

    warnings_text = " ".join(str(item) for item in data_quality_warnings).lower()
    trade_conflict = any(
        token in warnings_text
        for token in (
            "trade_state_consistency",
            "lifecycle_close_filled_but_position_open",
            "reconcile_flat_but_open_positions_nonzero",
            "close_lifecycle_missing_trade_export",
        )
    )
    checks.append(
        _check(
            "v5_trade_state_consistency_ok",
            "FAIL" if trade_conflict else "PASS",
            "trade_state_warning_present" if trade_conflict else "no_trade_state_warning",
            "no trade-state consistency warning",
            "V5",
            (
                "fix V5 state reconciliation before trusting paper/live evidence"
                if trade_conflict
                else ""
            ),
        )
    )

    advisory = report_frames.get("strategy_opportunity_advisory", pl.DataFrame())
    advisory_status, advisory_observed, advisory_action = _advisory_freshness_status(
        advisory,
        generated_at=generated,
    )
    checks.append(
        _check(
            "advisory_freshness_ok",
            advisory_status,
            advisory_observed,
            "rows>0 and latest generated_at/expires_at fresh",
            "quant-lab",
            advisory_action,
        )
    )
    checks.append(
        _bottom_zone_paper_v5_rows_check(
            advisory,
            report_frames.get("bottom_zone_reversal_shadow", pl.DataFrame()),
        )
    )

    queue = frames.get("alpha_factory_promotion_queue", pl.DataFrame())
    queue_paper = _count_decision(queue, "PAPER_READY")
    advisory_alpha_factory_paper = _alpha_factory_advisory_paper_count(advisory)
    alpha_factory_ok = advisory_alpha_factory_paper <= queue_paper
    checks.append(
        _check(
            "alpha_factory_source_health_match",
            "PASS" if alpha_factory_ok else "FAIL",
            f"queue_paper_ready={queue_paper};advisory_alpha_factory_paper_ready={advisory_alpha_factory_paper}",
            "advisory PAPER_READY count must not exceed promotion queue PAPER_READY count",
            "quant-lab",
            "" if alpha_factory_ok else "dedupe advisory with alpha_factory_promotion_queue as cap",
        )
    )

    checks.append(
        _label_join_check(
            "bnb_bypass_label_join_ok",
            report_frames.get("bnb_strong_alpha6_bypass_shadow", pl.DataFrame()),
            owner="V5",
            next_action="verify V5 skipped_candidate_labels join for BNB strong alpha6 bypass",
        )
    )
    checks.append(
        _label_join_check(
            "final_score_conflict_label_join_ok",
            report_frames.get("final_score_alpha6_conflict", pl.DataFrame()),
            owner="V5",
            next_action="verify V5 skipped_candidate_labels join for final_score conflict",
        )
    )

    checks.append(
        _expanded_universe_v5_check(
            frames=frames,
            report_frames=report_frames,
        )
    )

    checks.append(
        _fast_microstructure_observability_check(
            _first_frame(
                frames,
                report_frames,
                (
                    "fast_microstructure_features",
                    "v5_fast_microstructure_features",
                ),
            )
        )
    )

    label_summary = report_frames.get("backtest_label_summary", pl.DataFrame())
    promotion = report_frames.get("research_promotion_decision", pl.DataFrame())
    replay = report_frames.get("v5_decision_replay_trades", pl.DataFrame())
    backtest_present = not label_summary.is_empty() and not promotion.is_empty()
    checks.append(
        _check(
            "backtest_reports_present",
            "PASS" if backtest_present else "FAIL",
            f"label_rows={label_summary.height};promotion_rows={promotion.height};replay_rows={replay.height}",
            "backtest_label_summary and research_promotion_decision rows > 0",
            "quant-lab",
            "" if backtest_present else "refresh V5 telemetry labels and rerun export-daily",
        )
    )

    web_warning_present = any(
        marker in warnings_text
        for marker in (
            "web_file_index_missing_fallback_rglob",
            "web_file_index_missing_refresh_required",
        )
    )
    checks.append(
        _check(
            "web_rglob_fallback_zero",
            "FAIL" if web_warning_present else "PASS",
            "web_file_index_warning_present" if web_warning_present else "0",
            "0 web file-index/rglob warnings",
            "quant-lab",
            (
                "run qlab refresh-web-file-index and lake-small-file-maintenance"
                if web_warning_present
                else ""
            ),
        )
    )

    lake_warn_threshold = _int_env("QUANT_LAB_ACCEPTANCE_LAKE_FILE_WARN_THRESHOLD", 5_000)
    lake_fail_threshold = _int_env("QUANT_LAB_ACCEPTANCE_LAKE_FILE_FAIL_THRESHOLD", 7_500)
    lake_health_ok = _lake_file_health_ok(lake_file_health)
    lake_health_suffix = _lake_file_health_observed_suffix(lake_file_health)
    if lake_file_count is None:
        lake_status = "WARNING"
        lake_observed = "not_observable"
        lake_action = "build bronze/lake_file_index before judging lake file-count health"
    else:
        if lake_health_ok:
            lake_status = "PASS"
        elif lake_file_count > lake_fail_threshold:
            lake_status = "FAIL"
        elif lake_file_count > lake_warn_threshold:
            lake_status = "WARNING"
        else:
            lake_status = "PASS"
        lake_observed = str(lake_file_count) + lake_health_suffix
        lake_action = (
            "" if lake_status == "PASS" else "run lake-small-file-maintenance on priority datasets"
        )
    checks.append(
        _check(
            "lake_parquet_file_count_under_threshold",
            lake_status,
            lake_observed,
            (
                f"warn <= {lake_warn_threshold}; fail <= {lake_fail_threshold}; "
                "or lake_health warning_count=0"
            ),
            "quant-lab",
            lake_action,
        )
    )
    lake_growth_threshold = _int_env("QUANT_LAB_ACCEPTANCE_LAKE_FILE_GROWTH_24H_WARN", 250)
    if lake_file_growth_24h_count is None:
        growth_status = "WARNING"
        growth_observed = "not_observable"
        growth_action = "refresh bronze/lake_file_index before judging 24h file growth"
    else:
        growth_status = (
            "PASS"
            if lake_health_ok or lake_file_growth_24h_count <= lake_growth_threshold
            else "WARNING"
        )
        growth_observed = str(lake_file_growth_24h_count) + lake_health_suffix
        growth_action = (
            ""
            if growth_status == "PASS"
            else "run compaction before small-file growth becomes a dashboard/API risk"
        )
    checks.append(
        _check(
            "lake_parquet_file_growth_24h_ok",
            growth_status,
            growth_observed,
            f"<= {lake_growth_threshold}; or lake_health warning_count=0",
            "quant-lab",
            growth_action,
        )
    )

    api_status, api_observed, api_expected, api_action = _api_latency_check(api_latency_summary)
    checks.append(
        _check(
            "api_latency_p95_ok",
            api_status,
            api_observed,
            api_expected,
            "quant-lab",
            api_action,
        )
    )
    checks.append(_api_auth_error_rate_check(api_latency_summary))
    auth_slo = frames.get("api_auth_production_slo", pl.DataFrame())
    auth_statuses = {
        str(row.get("production_auth_slo_status") or "UNKNOWN").upper()
        for row in (auth_slo.to_dicts() if not auth_slo.is_empty() else [])
    }
    auth_recovered = bool(auth_statuses) and auth_statuses == {"PASS"}
    checks.append(
        _check(
            "api_auth_incident_recovered_24h",
            "PASS" if auth_recovered else "FAIL",
            f"production_slo_statuses={sorted(auth_statuses)};rows={auth_slo.height}",
            "trusted production clients have no unexpected 401 in the last 24h",
            "V5/quant-lab",
            "observe from last 401 until 24h clean" if not auth_recovered else "",
        )
    )
    paper_freshness = frames.get("paper_runtime_freshness", pl.DataFrame())
    freshness_statuses = {
        str(row.get("status") or "FAIL").upper()
        for row in (paper_freshness.to_dicts() if not paper_freshness.is_empty() else [])
    }
    runtime_status = (
        "FAIL"
        if not freshness_statuses or "FAIL" in freshness_statuses
        else "WARNING"
        if "WARNING" in freshness_statuses
        else "PASS"
    )
    checks.append(
        _check(
            "paper_runtime_freshness_ok",
            runtime_status,
            f"statuses={sorted(freshness_statuses)};rows={paper_freshness.height}",
            "runtime/registry/state fresh; event streams may be NO_NEW_EVENT_EXPECTED",
            "V5/quant-lab",
            "inspect stale open Paper positions" if runtime_status != "PASS" else "",
        )
    )
    propagation = frames.get("paper_proposal_propagation_status", pl.DataFrame())
    propagation_statuses = {
        str(row.get("propagation_status") or "UNKNOWN").upper()
        for row in (propagation.to_dicts() if not propagation.is_empty() else [])
    }
    current_proposal_count = frames.get(
        "paper_strategy_proposals_current", pl.DataFrame()
    ).height
    propagation_ok = (
        current_proposal_count == 0
        or (
            propagation.height == current_proposal_count
            and propagation_statuses == {"ACCEPTED_TRACKER_ACTIVE"}
        )
    )
    checks.append(
        _check(
            "paper_proposal_propagation_complete",
            "PASS" if propagation_ok else "BLOCKED",
            (
                f"current_proposals={current_proposal_count};"
                f"propagation_rows={propagation.height};statuses={sorted(propagation_statuses)}"
            ),
            "every Current Proposal has exact accepted ACK and active Tracker",
            "V5/quant-lab",
            "allow V5 to consume Current Proposals naturally" if not propagation_ok else "",
        )
    )
    checks.append(_github_ci_status_check(dict(pre_export_v5 or {})))

    v5_context = dict(pre_export_v5 or {})
    checks.extend(
        _readiness_checks(
            data_quality_warnings,
            report_frames.get("v5_enforce_readiness", pl.DataFrame()),
            v5_context,
        )
    )
    authoritative = _truthy(v5_context.get("authoritative_snapshot"))
    stale = _truthy(v5_context.get("stale_v5_bundle"))
    v5_ok = not stale
    v5_status = "PASS" if v5_ok else "WARNING"
    if stale:
        v5_next_action = (
            "V5 bundle is stale; downgrade V5-derived conclusions and rerun telemetry sync"
        )
    elif not authoritative:
        v5_next_action = (
            "synced but pre-export V5 refresh was disabled; "
            "enable refresh for an authoritative snapshot"
        )
    else:
        v5_next_action = ""
    checks.append(
        _check(
            "v5_bundle_sync_ok",
            v5_status,
            f"authoritative_snapshot={str(authoritative).lower()};stale_v5_bundle={str(stale).lower()}",
            "stale_v5_bundle=false; authoritative_snapshot describes refresh provenance",
            "quant-lab",
            v5_next_action,
        )
    )

    return _frame(checks, SYSTEM_ACCEPTANCE_FIELDS)


def _readiness_checks(
    data_quality_warnings: list[str] | tuple[str, ...],
    readiness: pl.DataFrame,
    v5_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matches = [
        str(item)
        for item in data_quality_warnings
        if "quant_lab_enforce_readiness" in str(item).lower()
    ]
    observed = "; ".join(matches)
    latest = readiness.tail(1).to_dicts()[0] if not readiness.is_empty() else {}
    advisory_state = str(
        latest.get("readiness_status")
        or latest.get("advisory_status")
        or ("BLOCKED" if "readiness_status=blocked" in observed.lower() else "")
        or ("WARN" if "readiness_status=warn" in observed.lower() else "ADVISORY_READY")
    ).upper()
    entry_state = str(latest.get("entry_status") or "BLOCKED").upper()
    scale_state = str(latest.get("scale_status") or "BLOCKED").upper()
    permission = str(
        latest.get("permission_status")
        or v5_context.get("permission_status")
        or v5_context.get("risk_permission_status")
        or "ACTIVE_ABORT"
    ).upper()
    canary_enabled = _truthy(
        latest.get("canary_enabled") or v5_context.get("canary_enabled")
    )
    advisory_ok = advisory_state in {"ADVISORY_READY", "READY", "PASS"}
    entry_ok = (
        entry_state in {"ENTRY_READY", "READY", "PASS"}
        and permission not in {"ABORT", "ACTIVE_ABORT", "BLOCKED"}
        and canary_enabled
    )
    scale_ok = (
        scale_state in {"SCALE_READY", "READY", "PASS"}
        and entry_ok
    )
    return [
        _check(
            "advisory_readiness_ok",
            "PASS" if advisory_ok else ("WARNING" if advisory_state == "WARN" else "FAIL"),
            f"advisory_status={advisory_state}",
            "ADVISORY_READY",
            "quant-lab",
            "" if advisory_ok else "restore advisory data/service readiness",
        ),
        _check(
            "entry_readiness_ok",
            "PASS" if entry_ok else "BLOCKED",
            f"entry_status={entry_state};permission_status={permission};canary_enabled={str(canary_enabled).lower()}",
            "ENTRY_READY with non-abort permission and canary enabled",
            "quant-lab + V5",
            "keep shadow; do not enable Entry or Enforce" if not entry_ok else "",
        ),
        _check(
            "scale_readiness_ok",
            "PASS" if scale_ok else "BLOCKED",
            f"scale_status={scale_state};entry_ready={str(entry_ok).lower()}",
            "SCALE_READY after Entry readiness",
            "quant-lab + V5",
            "do not scale capital" if not scale_ok else "",
        ),
    ]


def build_no_trigger_reasons(
    *,
    report_name: str,
    source_row_count: int,
    output_row_count: int,
    missing_field_reason: str,
    filtered_out_reason: str,
    next_action: str,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    if output_row_count > 0:
        return _frame([], NO_TRIGGER_REASON_FIELDS)
    return _frame(
        [
            {
                "report_name": report_name,
                "generated_at": generated.isoformat().replace("+00:00", "Z"),
                "source_row_count": int(source_row_count),
                "output_row_count": int(output_row_count),
                "missing_field_reason": missing_field_reason,
                "filtered_out_reason": filtered_out_reason,
                "next_action": next_action,
                "live_order_effect": "read_only_no_live_order",
            }
        ],
        NO_TRIGGER_REASON_FIELDS,
    )


def system_acceptance_dashboard_md(frame: pl.DataFrame) -> str:
    rows = frame.to_dicts() if not frame.is_empty() else []
    counts: dict[str, int] = {"PASS": 0, "WARNING": 0, "FAIL": 0}
    for row in rows:
        status = str(row.get("status") or "WARNING").upper()
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# System Acceptance Dashboard",
        "",
        "Read-only end-to-end acceptance checks for V5 and quant-lab telemetry.",
        "This dashboard does not change live trading behavior.",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARNING: {counts.get('WARNING', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        "",
        "| check | status | observed | expected | owner | next action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_md_cell(row.get(column)) for column in SYSTEM_ACCEPTANCE_FIELDS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _label_join_check(
    check_name: str,
    frame: pl.DataFrame,
    *,
    owner: str,
    next_action: str,
) -> dict[str, Any]:
    if frame.is_empty():
        return _check(
            check_name,
            "WARNING",
            "rows=0",
            "if rows>0 then future label fields must be observable",
            owner,
            "collect candidate rows or verify upstream report generation",
        )
    rows = frame.to_dicts()
    observable = sum(1 for row in rows if _row_has_future_label(row))
    pending = len(rows) - observable
    pending_ratio = pending / len(rows) if rows else 1.0
    failure_reasons = _field_counts(rows, "label_join_failure_reason")
    match_types = _field_counts(rows, "label_join_match_type")
    status = "PASS" if observable > 0 and pending_ratio < 0.95 else "FAIL"
    return _check(
        check_name,
        status,
        (
            f"observable_label_rows={observable};pending_rows={pending};rows={len(rows)};"
            f"pending_ratio={pending_ratio:.6f};label_join_failure_reasons={failure_reasons};"
            f"label_join_match_types={match_types}"
        ),
        "pending_ratio < 0.95 and at least one report row has a future label",
        owner,
        "" if status == "PASS" else next_action,
    )


def _expanded_universe_v5_check(
    *,
    frames: Mapping[str, pl.DataFrame],
    report_frames: Mapping[str, pl.DataFrame],
) -> dict[str, Any]:
    ready_symbols = _expanded_ready_symbols(
        frames.get("expanded_universe_candidate_maturity", pl.DataFrame())
    )
    advisory = report_frames.get("strategy_opportunity_advisory", pl.DataFrame())
    ready_symbols.update(_expanded_advisory_symbols(advisory))
    v5_reader = _v5_paper_symbols(
        _first_frame(
            frames,
            report_frames,
            (
                "v5_expanded_universe_advisory_reader",
                "expanded_universe_advisory_reader",
            ),
        )
    )
    v5_runs = _v5_paper_symbols(
        _first_frame(
            frames,
            report_frames,
            (
                "v5_expanded_universe_paper_run",
                "v5_expanded_universe_paper_runs",
                "expanded_universe_paper_runs",
                "v5_paper_strategy_run",
                "paper_strategy_runs",
            ),
        )
    )
    v5_daily = _v5_paper_symbols(
        _first_frame(
            frames,
            report_frames,
            (
                "v5_expanded_universe_paper_daily",
                "expanded_universe_paper_daily",
            ),
        )
    )
    runs_frame = _first_frame(
        frames,
        report_frames,
        (
            "v5_expanded_universe_paper_run",
            "v5_expanded_universe_paper_runs",
            "expanded_universe_paper_runs",
            "v5_paper_strategy_run",
            "paper_strategy_runs",
        ),
    )
    daily_frame = _first_frame(
        frames,
        report_frames,
        (
            "v5_expanded_universe_paper_daily",
            "expanded_universe_paper_daily",
        ),
    )
    wanted = {symbol for symbol in ready_symbols if symbol in {"HYPE-USDT", "WLD-USDT"}}
    observed = wanted & v5_reader & v5_runs & v5_daily
    if not wanted:
        return _check(
            "expanded_universe_paper_v5_rows_ok",
            "PASS",
            "not_applicable; no HYPE/WLD PAPER_READY source rows",
            "when HYPE/WLD PAPER_READY source rows exist, they should produce V5 paper rows",
            "V5",
            "",
        )
    missing_reader = sorted(wanted - v5_reader)
    missing_runs = sorted(wanted - v5_runs)
    missing_daily = sorted(wanted - v5_daily)
    missing_no_sample_reason = sorted(
        symbol
        for symbol in wanted & v5_runs
        if _expanded_entry_count_zero_without_reason(symbol, runs_frame, daily_frame)
    )
    no_sample_reason_mix = _expanded_no_sample_reason_mix(wanted, runs_frame)
    status = "PASS" if wanted <= observed and not missing_no_sample_reason else "FAIL"
    if status == "PASS":
        next_action = ""
    elif missing_reader or missing_daily:
        next_action = (
            "generate a fresh V5 follow-up bundle after PAPER_READY advisory and rerun "
            "quant-lab V5 telemetry sync"
        )
    else:
        next_action = "fix V5 expanded_universe reader/runs/daily telemetry sync"
    return _check(
        "expanded_universe_paper_v5_rows_ok",
        status,
        (
            f"ready={sorted(wanted)};reader={sorted(v5_reader & wanted)};"
            f"runs={sorted(v5_runs & wanted)};daily={sorted(v5_daily & wanted)};"
            f"missing_reader={missing_reader};missing_runs={missing_runs};missing_daily={missing_daily};"
            f"entry_count_zero_missing_no_sample_reason={missing_no_sample_reason};"
            f"no_sample_reason_mix={no_sample_reason_mix}"
        ),
        (
            "all HYPE/WLD PAPER_READY rows appear in V5 expanded reader/runs/daily "
            "telemetry, and no-entry rows explain no_sample_reason"
        ),
        "V5",
        next_action,
    )


def _advisory_freshness_status(
    frame: pl.DataFrame,
    *,
    generated_at: datetime,
) -> tuple[str, str, str]:
    if frame.is_empty():
        return "FAIL", "rows=0", "refresh strategy_opportunity_advisory"
    latest_generated = _latest_dt(frame, ("generated_at", "as_of_ts"))
    latest_expiry = _latest_dt(frame, ("expires_at",))
    if latest_generated is None:
        return (
            "WARNING",
            f"rows={frame.height};generated_at=not_observable",
            "ensure advisory writes generated_at",
        )
    if latest_expiry is not None and latest_expiry < generated_at:
        return (
            "FAIL",
            f"rows={frame.height};latest_generated_at={_iso(latest_generated)};latest_expires_at={_iso(latest_expiry)}",
            "rerun advisory export before relying on paper/shadow rows",
        )
    age_sec = max(0, int((generated_at - latest_generated).total_seconds()))
    max_age = _int_env("QUANT_LAB_ACCEPTANCE_ADVISORY_MAX_AGE_SEC", 3 * 60 * 60)
    status = "PASS" if age_sec <= max_age else "FAIL"
    return (
        status,
        f"rows={frame.height};age_sec={age_sec};latest_generated_at={_iso(latest_generated)}",
        "" if status == "PASS" else "refresh advisory or inspect API/source lag",
    )


def _api_latency_check(frame: pl.DataFrame) -> tuple[str, str, str, str]:
    if frame.is_empty():
        return "WARNING", "rows=0", "p95_ms <= 1000", "collect API metrics"
    rows = frame.to_dicts()
    overall = next((row for row in rows if str(row.get("endpoint")) == "__all__"), rows[0])
    p95 = _float(overall.get("p95_ms"))
    threshold = float(_int_env("QUANT_LAB_ACCEPTANCE_API_P95_MS", 1000))
    if p95 is None:
        return (
            "WARNING",
            "p95_ms=not_observable",
            f"p95_ms <= {threshold:g}",
            "collect API latency metrics",
        )
    status = "PASS" if p95 <= threshold else "FAIL"
    return (
        status,
        f"p95_ms={p95:g}",
        f"p95_ms <= {threshold:g}",
        "" if status == "PASS" else "inspect API cache and lake-scan path",
    )


def _api_auth_error_rate_check(frame: pl.DataFrame) -> dict[str, Any]:
    threshold = float(os.environ.get("QUANT_LAB_ACCEPTANCE_API_AUTH_ERROR_RATE", "0.01"))
    if frame.is_empty():
        return _check(
            "api_auth_error_rate_ok",
            "WARNING",
            "rows=0",
            f"auth_error_rate < {threshold:g}",
            "quant-lab",
            "collect API auth metrics",
        )
    rows = frame.to_dicts()
    production = next(
        (
            row
            for row in rows
            if str(row.get("endpoint")) == "__production_v5__"
            and (_float(row.get("count")) or 0.0) > 0
        ),
        None,
    )
    overall = production or next(
        (row for row in rows if str(row.get("endpoint")) == "__all__"),
        rows[0],
    )
    scope = "production_v5" if production is not None else "all"
    request_count = _float(overall.get("count")) or 0.0
    auth_error_rate = _float(overall.get("auth_error_rate"))
    auth_error_count = _float(overall.get("auth_error_count")) or 0.0
    if request_count <= 0 or auth_error_rate is None:
        return _check(
            "api_auth_error_rate_ok",
            "WARNING",
            f"scope={scope};request_count={request_count:g};auth_error_rate=not_observable",
            f"auth_error_rate < {threshold:g}",
            "quant-lab",
            "collect API auth metrics",
        )
    status = "PASS" if auth_error_rate < threshold else "FAIL"
    return _check(
        "api_auth_error_rate_ok",
        status,
        (
            f"scope={scope};auth_error_count={auth_error_count:g};"
            f"request_count={request_count:g};auth_error_rate={auth_error_rate:.6f}"
        ),
        f"auth_error_rate < {threshold:g}",
        "quant-lab",
        "" if status == "PASS" else "fix unauthorized API callers or token rollout",
    )


def _github_ci_status_check(v5_context: Mapping[str, Any]) -> dict[str, Any]:
    summary = v5_context.get("github_ci_status")
    rows = v5_context.get("github_ci_status_rows")
    if not isinstance(summary, Mapping):
        return _check(
            "github_ci_status_ok",
            "WARNING",
            "github_ci_status=not_observable",
            "quant-lab and V5 current commits must have GitHub Actions success",
            "quant-lab/V5",
            "write GitHub CI status into the expert pack before veto/enforce promotion",
        )
    status = str(summary.get("overall_status") or "WARNING").upper()
    if status not in {"PASS", "WARNING", "FAIL"}:
        status = "WARNING"
    components = summary.get("components")
    component_parts: list[str] = []
    if isinstance(components, Mapping):
        for name in sorted(str(key) for key in components.keys()):
            item = components.get(name)
            if not isinstance(item, Mapping):
                continue
            conclusion = item.get("workflow_conclusion")
            commit = str(item.get("commit_sha") or "")
            short_commit = commit[:12] if commit else "missing"
            component_parts.append(f"{name}:{short_commit}:{conclusion}")
    if not component_parts and isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("component") or "")
            commit = str(row.get("commit_sha") or "")
            conclusion = row.get("workflow_conclusion")
            component_parts.append(f"{name}:{commit[:12] if commit else 'missing'}:{conclusion}")
    observed = ";".join(component_parts) if component_parts else "github_ci_status=empty"
    next_action = ""
    if status == "FAIL":
        next_action = "fix failing GitHub Actions before any enforce/live promotion"
    elif status == "WARNING":
        next_action = "collect current commit GitHub Actions success evidence"
    return _check(
        "github_ci_status_ok",
        status,
        observed,
        "quant-lab and V5 current commits must have GitHub Actions success",
        "quant-lab/V5",
        next_action,
    )


def _expanded_ready_symbols(frame: pl.DataFrame) -> set[str]:
    symbols: set[str] = set()
    for row in _rows(frame):
        state = str(
            row.get("expanded_universe_maturity_state")
            or row.get("maturity_state")
            or row.get("decision")
            or ""
        ).upper()
        if state == "PAPER_READY":
            symbols.add(normalize_symbol(row.get("symbol")))
    return symbols


def _expanded_advisory_symbols(frame: pl.DataFrame) -> set[str]:
    symbols: set[str] = set()
    for row in _rows(frame):
        mode = str(row.get("recommended_mode") or "").lower()
        decision = str(row.get("decision") or "").upper()
        universe = str(row.get("universe_type") or "").lower()
        symbol = normalize_symbol(row.get("symbol"))
        paper_ready = mode == "paper" or decision == "PAPER_READY" or universe == "expanded_paper"
        if symbol in {"HYPE-USDT", "WLD-USDT"} and paper_ready:
            symbols.add(symbol)
    return symbols


def _v5_paper_symbols(frame: pl.DataFrame) -> set[str]:
    symbols: set[str] = set()
    for row in _rows(frame):
        strategy_id = str(row.get("strategy_id") or "")
        symbol = normalize_symbol(row.get("symbol"))
        if symbol in {"HYPE-USDT", "WLD-USDT"} or strategy_id.startswith(("HYPE_", "WLD_")):
            symbols.add(symbol)
    return symbols


def _expanded_entry_count_zero_without_reason(
    symbol: str,
    runs_frame: pl.DataFrame,
    daily_frame: pl.DataFrame,
) -> bool:
    run_rows = [
        row
        for row in _rows(runs_frame)
        if normalize_symbol(row.get("symbol")) == symbol
        or str(row.get("strategy_id") or "").startswith(symbol.split("-", 1)[0])
    ]
    daily_rows = [
        row
        for row in _rows(daily_frame)
        if normalize_symbol(row.get("symbol")) == symbol
        or str(row.get("strategy_id") or "").startswith(symbol.split("-", 1)[0])
    ]
    if not run_rows and not daily_rows:
        return False
    any_entry = any(_truthy(row.get("would_enter")) for row in run_rows)
    any_entry = any_entry or any((_float(row.get("entry_count")) or 0.0) > 0 for row in daily_rows)
    if any_entry:
        return False
    return not any(str(row.get("no_sample_reason") or "").strip() for row in run_rows)


def _expanded_no_sample_reason_mix(symbols: set[str], runs_frame: pl.DataFrame) -> dict[str, int]:
    bases = {symbol.split("-", 1)[0] for symbol in symbols}
    rows = [
        row
        for row in _rows(runs_frame)
        if normalize_symbol(row.get("symbol")) in symbols
        or str(row.get("strategy_id") or "").split("_", 1)[0] in bases
    ]
    return _field_counts(rows, "no_sample_reason")


def _first_frame(
    frames: Mapping[str, pl.DataFrame],
    report_frames: Mapping[str, pl.DataFrame],
    names: tuple[str, ...],
) -> pl.DataFrame:
    for name in names:
        frame = frames.get(name, pl.DataFrame())
        if frame is not None and not frame.is_empty():
            return frame
        frame = report_frames.get(name, pl.DataFrame())
        if frame is not None and not frame.is_empty():
            return frame
    return pl.DataFrame()


def _alpha_factory_advisory_paper_count(frame: pl.DataFrame) -> int:
    count = 0
    for row in _rows(frame):
        strategy = str(row.get("strategy_candidate") or "")
        source = str(row.get("source_module") or "").lower()
        decision = str(row.get("decision") or "").upper()
        mode = str(row.get("recommended_mode") or "").lower()
        if (source == "alpha_factory" or strategy.startswith("v5.af.")) and (
            decision == "PAPER_READY" or mode == "paper"
        ):
            count += 1
    return count


def _count_decision(frame: pl.DataFrame, decision: str) -> int:
    if frame.is_empty():
        return 0
    wanted = decision.upper()
    count = 0
    for row in frame.to_dicts():
        value = str(
            row.get("decision") or row.get("promotion_state") or row.get("recommended_mode") or ""
        ).upper()
        if value == wanted:
            count += 1
    return count


def _row_has_future_label(row: Mapping[str, Any]) -> bool:
    for horizon in (4, 8, 12, 24, 48, 72):
        if _float(row.get(f"future_{horizon}h_net_bps")) is not None:
            return True
        if _float(row.get(f"label_{horizon}h_net_bps")) is not None:
            return True
        if _float(row.get(f"label_{horizon}h_after_cost_bps")) is not None:
            return True
        if _float(row.get(f"paper_pnl_bps_{horizon}h")) is not None:
            return True
    return False


def _field_counts(rows: list[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "").strip()
        if not value:
            value = "not_observable"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


FAST_MICROSTRUCTURE_CORE_FIELDS = (
    "orderbook_imbalance_1m",
    "orderbook_imbalance_5m",
    "taker_buy_sell_imbalance_5m",
    "cvd_5m",
    "cvd_divergence",
    "spread_bps_change_5m",
)


def _fast_microstructure_observability_check(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return _check(
            "fast_microstructure_core_observability_ok",
            "WARNING",
            "rows=0",
            "core fast microstructure fields not_observable_ratio < 0.30",
            "quant-lab",
            "refresh orderbook_spread_1m/trade_activity_1m rollups and rerun export",
        )
    rows = frame.to_dicts()
    missing = 0
    total = len(rows) * len(FAST_MICROSTRUCTURE_CORE_FIELDS)
    missing_fields: list[str] = []
    for field in FAST_MICROSTRUCTURE_CORE_FIELDS:
        if field not in frame.columns:
            missing += len(rows)
            missing_fields.append(field)
            continue
        for row in rows:
            if _float(row.get(field)) is None:
                missing += 1
    ratio = missing / total if total else 1.0
    status = "PASS" if ratio < 0.30 else "FAIL"
    return _check(
        "fast_microstructure_core_observability_ok",
        status,
        f"rows={len(rows)};missing_cells={missing};total_cells={total};not_observable_ratio={ratio:.6f};missing_fields={missing_fields}",
        "core fast microstructure fields not_observable_ratio < 0.30",
        "quant-lab",
        "" if status == "PASS" else "fix orderbook/trade rollups or feature field mapping",
    )


def _bottom_zone_paper_v5_rows_check(
    advisory: pl.DataFrame,
    bottom_zone_reversal_shadow: pl.DataFrame,
) -> dict[str, Any]:
    source_symbols = sorted(
        {
            normalize_symbol(row.get("symbol"))
            for row in _rows(bottom_zone_reversal_shadow)
            if str(row.get("bottom_zone_state") or "").upper() == "BOTTOM_PROBE_ALLOWED"
            and _truthy(row.get("would_probe_paper"))
            and normalize_symbol(row.get("symbol")) not in {"", "UNKNOWN"}
        }
    )
    advisory_rows = [
        row
        for row in _rows(advisory)
        if str(row.get("strategy_candidate") or "") == "v5.bottom_zone_probe_paper"
    ]
    valid_symbols = sorted(
        {
            normalize_symbol(row.get("symbol"))
            for row in advisory_rows
            if str(row.get("live_order_effect") or "") == "read_only_no_live_order"
            and (_float(row.get("max_live_notional_usdt")) or 0.0) == 0.0
            and str(row.get("decision") or "").upper() == "PAPER_READY"
            and str(row.get("recommended_mode") or "").lower() == "paper"
            and normalize_symbol(row.get("symbol")) not in {"", "UNKNOWN"}
        }
    )
    missing = [symbol for symbol in source_symbols if symbol not in valid_symbols]
    status = "PASS" if not missing else "FAIL"
    return _check(
        "bottom_zone_paper_v5_rows_ok",
        status,
        (
            f"source_probe_symbols={source_symbols};"
            f"valid_advisory_symbols={valid_symbols};missing={missing};"
            f"advisory_rows={len(advisory_rows)}"
        ),
        (
            "every BOTTOM_PROBE_ALLOWED would_probe_paper symbol has a "
            "read-only/no-live v5.bottom_zone_probe_paper PAPER_READY row"
        ),
        "quant-lab/V5",
        (
            ""
            if status == "PASS"
            else "publish final strategy_opportunity_advisory to gold before API/V5 export"
        ),
    )


def _latest_dt(frame: pl.DataFrame, columns: tuple[str, ...]) -> datetime | None:
    latest: datetime | None = None
    for row in _rows(frame):
        for column in columns:
            parsed = _dt(row.get(column))
            if parsed is not None and (latest is None or parsed > latest):
                latest = parsed
    return latest


def _dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "pass", "passed"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _lake_file_health_ok(lake_file_health: Mapping[str, Any] | None) -> bool:
    if not isinstance(lake_file_health, Mapping):
        return False
    if lake_file_health.get("ok") is False:
        return False
    try:
        return int(lake_file_health.get("warning_count") or 0) == 0
    except (TypeError, ValueError):
        return False


def _lake_file_health_observed_suffix(lake_file_health: Mapping[str, Any] | None) -> str:
    if not isinstance(lake_file_health, Mapping):
        return ""
    if "warning_count" not in lake_file_health:
        return ""
    warning_count = lake_file_health.get("warning_count")
    total = lake_file_health.get("total_parquet_files")
    if total in (None, ""):
        return f";lake_health_warning_count={warning_count}"
    return f";lake_health_warning_count={warning_count};lake_health_total={total}"


def _iso(value: datetime | None) -> str:
    return value.isoformat().replace("+00:00", "Z") if value else "not_observable"


def _rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.is_empty():
        return []
    return frame.to_dicts()


def _check(
    check_name: str,
    status: str,
    observed_value: Any,
    expected_value: Any,
    owner: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": str(status).upper(),
        "observed_value": str(observed_value),
        "expected_value": str(expected_value),
        "owner": owner,
        "next_action": next_action,
    }


def _frame(rows: list[dict[str, Any]], fields: list[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={field: pl.Utf8 for field in fields})
    normalized = pl.DataFrame(rows, infer_schema_length=None)
    for field in fields:
        if field not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=pl.Utf8).alias(field))
    return normalized.select(fields)


def _md_cell(value: Any) -> str:
    text = str(value or "")
    return text.replace("|", "\\|").replace("\n", " ")
