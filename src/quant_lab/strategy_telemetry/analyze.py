from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset
from quant_lab.strategy_telemetry.models import V5TelemetryAnalysisResult

GOLD_DATASETS = {
    "strategy_health_daily": Path("gold/strategy_health_daily"),
    "v5_execution_quality_daily": Path("gold/v5_execution_quality_daily"),
    "v5_gate_compliance_daily": Path("gold/v5_gate_compliance_daily"),
    "v5_missed_opportunity_daily": Path("gold/v5_missed_opportunity_daily"),
    "v5_config_health_daily": Path("gold/v5_config_health_daily"),
    "v5_issue_summary_daily": Path("gold/v5_issue_summary_daily"),
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
}


def analyze_v5_telemetry(lake_root: Path, date: str | None = None) -> V5TelemetryAnalysisResult:
    root = Path(lake_root)
    analysis_date = date or datetime.now(UTC).date().isoformat()
    manifest = read_parquet_dataset(root / SILVER["manifest"])
    secret_scan = read_parquet_dataset(root / SILVER["secret_scan"])
    run_summary = read_parquet_dataset(root / SILVER["run_summary"])
    decisions = read_parquet_dataset(root / SILVER["decision_audit"])
    trades = read_parquet_dataset(root / SILVER["trade_event"])
    roundtrips = read_parquet_dataset(root / SILVER["roundtrip"])
    routers = read_parquet_dataset(root / SILVER["router_decision"])
    positions = read_parquet_dataset(root / SILVER["open_position"])
    states = read_parquet_dataset(root / SILVER["state_snapshot"])
    issues = read_parquet_dataset(root / SILVER["issue"])
    config = read_parquet_dataset(root / SILVER["config_audit"])
    targets = read_parquet_dataset(root / SILVER["high_score_target"])
    outcomes = read_parquet_dataset(root / SILVER["high_score_outcome"])
    skipped = read_parquet_dataset(root / SILVER["skipped"])
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

    gate_violations = _gate_compliance_violations(decisions, trades)
    if gate_violations:
        critical.append("gate_compliance_violation")
        next_actions.extend(gate_violations)

    router_reason_top = _router_reason_top(routers)
    fallback_count = _fallback_count(decisions)
    if fallback_count:
        warnings.append(f"quant_lab fallback used {fallback_count} times")

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
        warnings=warnings,
        critical_reasons=critical,
        next_actions=sorted(set(next_actions)),
    )
    _write_gold(root, result, gate_violations=gate_violations, fallback_count=fallback_count)
    return result


def _write_gold(
    lake_root: Path,
    result: V5TelemetryAnalysisResult,
    gate_violations: list[str],
    fallback_count: int,
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
    useful_columns = {"consumed", "status", "issue_type", "key", "raw_payload_json"}
    if not useful_columns.intersection(df.columns):
        return {"count": 0, "unknown": True, "top_keys": []}

    keys: set[str] = set()
    unknown = False
    for row in df.to_dicts():
        payload = _payload(row)
        key = _first_text(row, payload, ["key", "config_key", "path", "name"])
        if _row_indicates_not_consumed(row, payload):
            if key:
                keys.add(key)
            else:
                unknown = True
    return {"count": len(keys), "unknown": unknown, "top_keys": sorted(keys)[:20]}


def _first_text(
    row: dict[str, Any],
    payload: dict[str, Any],
    fields: list[str],
) -> str | None:
    for field in fields:
        value = row.get(field)
        if value is None:
            value = payload.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


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
    text = "\n".join(json.dumps(row, sort_keys=True, default=str).lower() for row in df.to_dicts())
    return text.count("matured")


def _profitable_count(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    count = _truthy_column_count(df, ["profitable", "is_profitable"])
    if count is not None:
        return count
    text = "\n".join(json.dumps(row, sort_keys=True, default=str).lower() for row in df.to_dicts())
    return text.count("profitable")


def _truthy_column_count(df: pl.DataFrame, columns: list[str]) -> int | None:
    for column in columns:
        if column not in df.columns:
            continue
        count = 0
        for value in df[column].to_list():
            if isinstance(value, bool):
                count += int(value)
            elif str(value).strip().lower() in {"true", "1", "yes", "y"}:
                count += 1
        return count
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
