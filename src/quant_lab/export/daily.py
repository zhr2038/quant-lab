from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.data.lake import read_parquet_lazy
from quant_lab.risk.publish import (
    DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    is_permission_status_enforceable,
    permission_freshness_sec,
    permission_status,
    risk_permission_stale_vs_telemetry,
)
from quant_lab.strategy_telemetry.sanitize import SECRET_PATTERNS, safe_json_dumps
from quant_lab.symbols import normalize_symbol
from quant_lab.web import readers

SECTIONS = ["market", "features", "costs", "research", "risk", "anomalies", "v5", "charts"]
HEAVY_EXPORT_DATASET_LIMITS = {
    "okx_public_ws": 5_000,
    "trade_print": 20_000,
    "orderbook_snapshot": 20_000,
}
HEAVY_EXPORT_LOOKBACK_HOURS = 6

SECTION_DATASETS = {
    "market": ["market_bar", "trade_print", "orderbook_snapshot", "okx_public_ws"],
    "features": ["feature_value", "feature_coverage_daily", "feature_anomaly_daily"],
    "costs": [
        "cost_bucket_daily",
        "cost_health_daily",
        "okx_private_readonly_fills",
        "okx_private_readonly_bills",
    ],
    "research": ["alpha_evidence", "gate_decision"],
    "risk": ["risk_permission"],
    "anomalies": ["market_bar", "feature_value", "cost_bucket_daily", "gate_decision"],
    "v5": [
        "strategy_health_daily",
        "v5_execution_quality_daily",
        "v5_gate_compliance_daily",
        "v5_missed_opportunity_daily",
        "v5_config_health_daily",
        "v5_issue_summary_daily",
        "v5_quant_lab_mode_daily",
        "v5_quant_lab_enforcement_daily",
        "v5_quant_lab_usage",
        "v5_quant_lab_compliance",
        "v5_quant_lab_cost_usage",
        "v5_quant_lab_fallback",
    ],
    "charts": ["market_bar", "gate_decision", "cost_bucket_daily", "strategy_health_daily"],
}

REQUIRED_MEMBERS = [
    "README.md",
    "manifest.json",
    "provenance.json",
    "data_quality.json",
    "executive_summary.md",
    "expert_questions.md",
    "market/market_snapshot.csv",
    "market/market_bars_tail.csv",
    "market/trade_activity.csv",
    "market/orderbook_spread.csv",
    "market/funding_open_interest.csv",
    "features/feature_snapshot.csv",
    "features/feature_coverage.csv",
    "features/feature_anomalies.csv",
    "costs/cost_bucket_daily.csv",
    "costs/cost_health_daily.csv",
    "costs/cost_estimate_examples.json",
    "costs/cost_fallbacks.csv",
    "research/alpha_evidence.csv",
    "research/gate_decisions.csv",
    "research/research_conclusions.csv",
    "risk/risk_permission.json",
    "risk/risk_flags.csv",
    "anomalies/anomalies.csv",
    "anomalies/missing_data.csv",
    "anomalies/stale_data.csv",
    "anomalies/schema_violations.csv",
    "v5/v5_strategy_health.csv",
    "v5/v5_execution_quality.csv",
    "v5/v5_gate_compliance.csv",
    "v5/v5_missed_opportunity.csv",
    "v5/v5_config_health.csv",
    "v5/v5_issue_summary.csv",
    "v5/v5_quant_lab_mode.csv",
    "v5/v5_quant_lab_enforcement.csv",
    "v5/v5_quant_lab_usage.csv",
    "v5/v5_quant_lab_compliance.csv",
    "v5/v5_quant_lab_cost_usage.csv",
    "v5/v5_quant_lab_fallbacks.csv",
    "charts/market_close.png",
    "charts/market_returns.png",
    "charts/gate_status_counts.png",
    "charts/cost_distribution.png",
    "charts/v5_health_summary.png",
]

CSV_SCHEMAS: dict[str, list[str]] = {
    "costs/cost_bucket_daily.csv": [
        "day",
        "symbol",
        "regime",
        "event_type",
        "notional_bucket",
        "sample_count",
        "fee_bps_p50",
        "fee_bps_p75",
        "fee_bps_p90",
        "slippage_bps_p50",
        "slippage_bps_p75",
        "slippage_bps_p90",
        "spread_bps_p50",
        "spread_bps_p75",
        "spread_bps_p90",
        "total_cost_bps_p50",
        "total_cost_bps_p75",
        "total_cost_bps_p90",
        "fallback_level",
        "source",
    ],
    "costs/cost_health_daily.csv": [
        "day",
        "status",
        "cost_model_version",
        "actual_rows",
        "proxy_rows",
        "global_default_rows",
        "fallback_ratio",
        "symbols_with_actual_cost",
        "symbols_with_proxy_only",
        "symbols_missing_cost",
        "min_sample_count",
        "warnings_json",
        "created_at",
    ],
    "features/feature_snapshot.csv": [
        "feature_set",
        "feature_name",
        "feature_version",
        "symbol",
        "timeframe",
        "ts",
        "value",
        "lookback_bars",
        "input_dataset_version",
        "input_hash",
        "code_version",
        "created_at",
        "source",
        "is_valid",
        "invalid_reason",
    ],
    "features/feature_coverage.csv": [
        "day",
        "feature_set",
        "feature_name",
        "feature_version",
        "timeframe",
        "symbol",
        "total_rows",
        "valid_rows",
        "null_rows",
        "coverage",
        "min_ts",
        "max_ts",
        "created_at",
    ],
    "features/feature_anomalies.csv": [
        "day",
        "feature_set",
        "feature_name",
        "feature_version",
        "timeframe",
        "symbol",
        "anomaly_type",
        "anomaly_count",
        "severity",
        "example_ts",
        "created_at",
    ],
    "market/orderbook_spread.csv": ["symbol", "channel", "ts", "spread_bps"],
    "market/trade_activity.csv": ["symbol", "trade_count", "size_sum", "latest_trade_ts"],
    "research/alpha_evidence.csv": [
        "alpha_id",
        "version",
        "data_version",
        "feature_version",
        "cost_model_version",
        "universe_id",
        "start_ts",
        "end_ts",
        "coverage",
        "ic_mean",
        "ic_tstat",
        "rank_ic_mean",
        "rank_ic_tstat",
        "oos_sharpe",
        "oos_max_drawdown",
        "edge_cost_ratio",
        "paper_days",
        "created_at",
        "evidence_status",
    ],
    "research/gate_decisions.csv": [
        "alpha_id",
        "version",
        "gate_version",
        "status",
        "passed",
        "reasons",
        "metrics",
        "next_action",
        "created_at",
    ],
    "v5/v5_strategy_health.csv": [
        "strategy",
        "date",
        "status",
        "latest_bundle_ts",
        "latest_bundle_sha256",
        "run_count_72h",
        "decision_audit_count_24h",
        "trade_count_24h",
        "trade_count_72h",
        "roundtrip_count_72h",
        "open_position_count",
        "kill_switch_enabled",
        "reconcile_ok",
        "ledger_ok",
        "auto_risk_level",
        "high_issue_count",
        "medium_issue_count",
        "request_success_count",
        "request_error_count",
        "actual_fallback_count",
        "fallback_rate",
        "degraded_reason",
        "raw_imported_rows",
        "unique_event_rows",
        "duplicate_event_rows",
        "duplicate_rate",
        "first_seen_bundle_ts",
        "last_seen_bundle_ts",
    ],
    "v5/v5_execution_quality.csv": [
        "strategy",
        "date",
        "status",
        "fallback_count",
        "request_success_count",
        "request_error_count",
        "actual_fallback_count",
        "fallback_rate",
        "degraded_reason",
        "raw_imported_rows",
        "unique_event_rows",
        "duplicate_event_rows",
        "duplicate_rate",
    ],
    "v5/v5_gate_compliance.csv": [
        "strategy",
        "date",
        "status",
        "violation_count",
        "violations_json",
    ],
    "v5/v5_missed_opportunity.csv": ["strategy", "date", "status"],
    "v5/v5_config_health.csv": [
        "strategy",
        "date",
        "status",
        "config_not_consumed_count",
        "config_not_consumed_count_unknown",
        "config_not_consumed_top_keys_json",
    ],
    "v5/v5_issue_summary.csv": [
        "strategy",
        "date",
        "status",
        "high_issue_count",
        "medium_issue_count",
    ],
    "v5/v5_quant_lab_mode.csv": [
        "strategy",
        "date",
        "mode",
        "permission_gate_enforced",
        "cost_gate_enforced",
        "usage_count",
        "request_success_count",
        "request_error_count",
        "actual_fallback_count",
        "fallback_rate",
        "degraded_reason",
        "raw_imported_rows",
        "unique_event_rows",
        "duplicate_event_rows",
        "duplicate_rate",
        "first_seen_bundle_ts",
        "last_seen_bundle_ts",
        "cost_usage_count",
        "fallback_count",
        "latest_bundle_ts",
        "latest_bundle_sha256",
        "created_at",
    ],
    "v5/v5_quant_lab_enforcement.csv": [
        "strategy",
        "date",
        "mode",
        "permission_gate_enforced",
        "cost_gate_enforced",
        "actual_violation_count",
        "hypothetical_violation_count",
        "actual_violations_json",
        "hypothetical_violations_json",
        "latest_bundle_ts",
        "latest_bundle_sha256",
        "created_at",
    ],
    "v5/v5_quant_lab_usage.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "source_path_inside_bundle",
        "run_id",
        "row_index",
        "raw_payload_json",
    ],
    "v5/v5_quant_lab_compliance.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "source_path_inside_bundle",
        "permission",
        "side",
        "mode",
        "permission_gate_enforced",
        "raw_payload_json",
    ],
    "v5/v5_quant_lab_cost_usage.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "source_path_inside_bundle",
        "symbol",
        "cost_source",
        "count",
        "raw_payload_json",
    ],
    "v5/v5_quant_lab_fallbacks.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "source_path_inside_bundle",
        "event_key",
        "source_count",
        "first_seen_bundle_ts",
        "last_seen_bundle_ts",
        "endpoint",
        "event_type",
        "error_type",
        "fallback_used",
        "diagnosis",
        "degraded_reason",
        "fallback_reason",
        "count",
        "raw_payload_json",
    ],
}

_MemberPayload = str | bytes


class DailyExportResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    export_date: str
    profile: str
    zip_path: str
    sections: list[str]
    missing_sections: list[str]
    warnings: list[str]
    row_counts: dict[str, int] = Field(default_factory=dict)


class ExpertPackValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str | None
    valid: bool
    rejected: bool
    reasons: list[str]
    warnings: list[str]
    file_count: int = Field(ge=0)
    export_date: str | None = None


@dataclass(frozen=True)
class _DatasetSnapshot:
    frames: dict[str, pl.DataFrame]
    row_counts: dict[str, int]
    warnings: list[str]


def export_daily_pack(
    *,
    export_date: str | date,
    lake_root: str | Path,
    out_dir: str | Path,
    profile: str = "expert",
    command_line: list[str] | None = None,
) -> DailyExportResult:
    day = _parse_date(export_date)
    root = Path(lake_root)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(UTC)
    snapshot = _load_snapshot(root)
    missing_sections = _missing_sections(snapshot.row_counts)
    data_quality = _data_quality_payload(root, snapshot, day)
    warnings = sorted(set([*snapshot.warnings, *data_quality["warnings"]]))

    members: dict[str, _MemberPayload] = {}
    members.update(_dataset_members(snapshot.frames))
    members.update(_chart_members(snapshot.frames))
    members["README.md"] = _readme(day, root, profile)
    members["data_quality.json"] = _json_text(data_quality)
    members["executive_summary.md"] = _executive_summary(day, snapshot, data_quality)
    members["expert_questions.md"] = _expert_questions(snapshot, data_quality)
    members["provenance.json"] = _json_text(
        _provenance_payload(
            day=day,
            generated_at=generated_at,
            root=root,
            profile=profile,
            snapshot=snapshot,
            command_line=command_line or sys.argv,
        )
    )

    manifest = _manifest_payload(
        day=day,
        generated_at=generated_at,
        root=root,
        profile=profile,
        missing_sections=missing_sections,
        warnings=warnings,
        members=members,
        row_counts=snapshot.row_counts,
        command_line=command_line or sys.argv,
        snapshot=snapshot,
    )
    manifest["files"].append(
        {
            "path": "manifest.json",
            "kind": "json",
            "rows": None,
            "sha256": None,
        }
    )
    members["manifest.json"] = _json_text(manifest)

    _fail_on_secrets(members)

    zip_path = output_root / f"quant_lab_expert_pack_{day.isoformat()}.zip"
    _write_zip(zip_path, members)
    return DailyExportResult(
        export_date=day.isoformat(),
        profile=profile,
        zip_path=str(zip_path),
        sections=SECTIONS,
        missing_sections=missing_sections,
        warnings=warnings,
        row_counts=snapshot.row_counts,
    )


def validate_expert_pack(path: str | Path) -> ExpertPackValidationResult:
    pack_path = Path(path)
    reasons: list[str] = []
    warnings: list[str] = []
    file_count = 0
    export_date: str | None = None
    sha256 = _sha256_file(pack_path) if pack_path.exists() and pack_path.is_file() else None

    if not pack_path.exists():
        reasons.append("expert pack does not exist")
        return _validation_result(pack_path, sha256, reasons, warnings, file_count, export_date)
    if pack_path.suffix != ".zip":
        reasons.append("expert pack extension must be .zip")
        return _validation_result(pack_path, sha256, reasons, warnings, file_count, export_date)

    try:
        with zipfile.ZipFile(pack_path) as archive:
            names = archive.namelist()
            file_count = len(names)
            reasons.extend(_unsafe_zip_member_reasons(names))
            missing = sorted(set(REQUIRED_MEMBERS) - set(names))
            reasons.extend(f"missing required member: {name}" for name in missing)
            manifest = _read_zip_json(archive, "manifest.json", reasons)
            data_quality = _read_zip_json(archive, "data_quality.json", reasons)
            if isinstance(manifest, dict):
                value = manifest.get("export_date")
                export_date = str(value) if value else None
                manifest_files = manifest.get("files", [])
                if isinstance(manifest_files, list) and len(manifest_files) != file_count:
                    warnings.append("manifest file count does not match zip file count")
            if isinstance(data_quality, dict) and data_quality.get("status") == "FAIL":
                warnings.append("data_quality status is FAIL")
            reasons.extend(_zip_secret_reasons(archive, names))
    except zipfile.BadZipFile as exc:
        reasons.append(f"invalid zip: {exc}")

    return _validation_result(pack_path, sha256, reasons, warnings, file_count, export_date)


def _validation_result(
    pack_path: Path,
    sha256: str | None,
    reasons: list[str],
    warnings: list[str],
    file_count: int,
    export_date: str | None,
) -> ExpertPackValidationResult:
    unique_reasons = sorted(set(reasons))
    return ExpertPackValidationResult(
        path=str(pack_path),
        sha256=sha256,
        valid=not unique_reasons,
        rejected=bool(unique_reasons),
        reasons=unique_reasons,
        warnings=sorted(set(warnings)),
        file_count=file_count,
        export_date=export_date,
    )


def _load_snapshot(lake_root: Path) -> _DatasetSnapshot:
    frames: dict[str, pl.DataFrame] = {}
    row_counts: dict[str, int] = {}
    warnings: list[str] = []
    for name in sorted(readers.DATASET_PATHS):
        frame, row_count, warning = _load_export_frame(lake_root, name)
        frames[name] = frame
        row_counts[name] = row_count
        if warning:
            warnings.append(warning)
    for name, count in sorted(row_counts.items()):
        if count == 0:
            warnings.append(f"{name} dataset is missing or empty")
    return _DatasetSnapshot(frames=frames, row_counts=row_counts, warnings=warnings)


def _load_export_frame(
    lake_root: Path,
    dataset_name: str,
) -> tuple[pl.DataFrame, int, str | None]:
    if dataset_name not in HEAVY_EXPORT_DATASET_LIMITS:
        try:
            frame = readers.read_dataset(lake_root, dataset_name)
            return frame, frame.height, None
        except Exception as exc:
            return pl.DataFrame(), 0, f"{dataset_name} read failed: {exc}"

    dataset_path = readers.dataset_path_for(lake_root, dataset_name)
    try:
        lazy_frame = read_parquet_lazy(dataset_path)
        row_count = _lazy_row_count(lazy_frame)
        frame = _collect_recent_heavy_frame(
            lazy_frame,
            dataset_name,
            limit=HEAVY_EXPORT_DATASET_LIMITS[dataset_name],
        )
        return frame, row_count, None
    except Exception as exc:
        return pl.DataFrame(), 0, f"{dataset_name} sampled read failed: {exc}"


def _lazy_row_count(lazy_frame: pl.LazyFrame) -> int:
    try:
        return int(lazy_frame.select(pl.len().alias("rows")).collect().item())
    except Exception:
        return 0


def _collect_recent_heavy_frame(
    lazy_frame: pl.LazyFrame,
    dataset_name: str,
    *,
    limit: int,
) -> pl.DataFrame:
    columns = list(lazy_frame.collect_schema().names())
    timestamp_column = next(
        (
            column
            for column in readers.DATASET_TIMESTAMP_COLUMNS.get(dataset_name, ())
            if column in columns
        ),
        None,
    )
    if timestamp_column is None:
        return lazy_frame.tail(limit).collect()

    latest_frame = lazy_frame.select(
        pl.col(timestamp_column).max().alias(timestamp_column)
    ).collect()
    latest_value = latest_frame.item() if latest_frame.height else None
    latest_ts = readers._coerce_timestamp(latest_value)  # type: ignore[attr-defined]
    if latest_ts is None:
        return lazy_frame.tail(limit).collect()

    threshold = latest_ts - timedelta(hours=HEAVY_EXPORT_LOOKBACK_HOURS)
    try:
        frame = lazy_frame.filter(pl.col(timestamp_column) >= threshold).collect()
    except Exception:
        frame = lazy_frame.tail(limit).collect()
    if frame.height > limit:
        frame = _tail_by_time(frame, timestamp_column, limit=limit)
    return frame


def _dataset_members(frames: dict[str, pl.DataFrame]) -> dict[str, _MemberPayload]:
    market = frames.get("market_bar", pl.DataFrame())
    features = frames.get("feature_value", pl.DataFrame())
    feature_coverage = frames.get("feature_coverage_daily", pl.DataFrame())
    feature_anomalies = frames.get("feature_anomaly_daily", pl.DataFrame())
    costs = _normalize_symbol_frame(frames.get("cost_bucket_daily", pl.DataFrame()))
    cost_health = frames.get("cost_health_daily", pl.DataFrame())
    evidence = _alpha_evidence_for_export(frames.get("alpha_evidence", pl.DataFrame()))
    gates = frames.get("gate_decision", pl.DataFrame())
    risk = _risk_permissions_for_export(frames.get("risk_permission", pl.DataFrame()), frames)
    trades = _normalize_symbol_frame(frames.get("trade_print", pl.DataFrame()))
    books = _normalize_symbol_frame(frames.get("orderbook_snapshot", pl.DataFrame()))
    v5_health = frames.get("strategy_health_daily", pl.DataFrame())
    v5_execution = frames.get("v5_execution_quality_daily", pl.DataFrame())
    v5_gate = frames.get("v5_gate_compliance_daily", pl.DataFrame())
    v5_missed = frames.get("v5_missed_opportunity_daily", pl.DataFrame())
    v5_config = frames.get("v5_config_health_daily", pl.DataFrame())
    v5_issue = frames.get("v5_issue_summary_daily", pl.DataFrame())
    v5_mode = frames.get("v5_quant_lab_mode_daily", pl.DataFrame())
    v5_enforcement = frames.get("v5_quant_lab_enforcement_daily", pl.DataFrame())
    v5_usage = frames.get("v5_quant_lab_usage", pl.DataFrame())
    v5_compliance = frames.get("v5_quant_lab_compliance", pl.DataFrame())
    v5_cost_usage = frames.get("v5_quant_lab_cost_usage", pl.DataFrame())
    v5_fallbacks = frames.get("v5_quant_lab_fallback", pl.DataFrame())

    return {
        "market/market_snapshot.csv": _csv_member(
            "market/market_snapshot.csv", _latest_market_rows(market)
        ),
        "market/market_bars_tail.csv": _csv_member(
            "market/market_bars_tail.csv", _tail_by_time(market, "ts")
        ),
        "market/trade_activity.csv": _csv_member(
            "market/trade_activity.csv", readers.trade_activity_table(trades)
        ),
        "market/orderbook_spread.csv": _csv_member(
            "market/orderbook_spread.csv", readers.orderbook_spread_table(books)
        ),
        "market/funding_open_interest.csv": _csv_text(
            pl.DataFrame(schema={"symbol": pl.Utf8, "metric": pl.Utf8, "value": pl.Utf8})
        ),
        "features/feature_snapshot.csv": _csv_member(
            "features/feature_snapshot.csv", _tail_by_time(features, "ts")
        ),
        "features/feature_coverage.csv": _csv_member(
            "features/feature_coverage.csv",
            feature_coverage if not feature_coverage.is_empty() else _feature_coverage(features),
        ),
        "features/feature_anomalies.csv": _csv_member(
            "features/feature_anomalies.csv",
            feature_anomalies
            if not feature_anomalies.is_empty()
            else _feature_anomalies(features),
        ),
        "costs/cost_bucket_daily.csv": _csv_member("costs/cost_bucket_daily.csv", costs),
        "costs/cost_health_daily.csv": _csv_member(
            "costs/cost_health_daily.csv", cost_health
        ),
        "costs/cost_estimate_examples.json": _json_text(_cost_examples(costs)),
        "costs/cost_fallbacks.csv": _csv_text(_cost_fallbacks(costs)),
        "research/alpha_evidence.csv": _csv_member("research/alpha_evidence.csv", evidence),
        "research/gate_decisions.csv": _csv_member("research/gate_decisions.csv", gates),
        "research/research_conclusions.csv": _csv_text(
            pl.DataFrame(schema={"alpha_id": pl.Utf8, "conclusion": pl.Utf8, "source": pl.Utf8})
        ),
        "risk/risk_permission.json": _json_text({"rows": _rows(risk)}),
        "risk/risk_flags.csv": _csv_text(_risk_flags(risk)),
        "anomalies/anomalies.csv": _csv_text(_anomaly_rows(frames)),
        "anomalies/missing_data.csv": _csv_text(_missing_dataset_rows(frames)),
        "anomalies/stale_data.csv": _csv_text(_stale_rows(frames)),
        "anomalies/schema_violations.csv": _csv_text(_schema_violation_rows(market)),
        "v5/v5_strategy_health.csv": _csv_member("v5/v5_strategy_health.csv", v5_health),
        "v5/v5_execution_quality.csv": _csv_member("v5/v5_execution_quality.csv", v5_execution),
        "v5/v5_gate_compliance.csv": _csv_member("v5/v5_gate_compliance.csv", v5_gate),
        "v5/v5_missed_opportunity.csv": _csv_member("v5/v5_missed_opportunity.csv", v5_missed),
        "v5/v5_config_health.csv": _csv_member("v5/v5_config_health.csv", v5_config),
        "v5/v5_issue_summary.csv": _csv_member("v5/v5_issue_summary.csv", v5_issue),
        "v5/v5_quant_lab_mode.csv": _csv_member("v5/v5_quant_lab_mode.csv", v5_mode),
        "v5/v5_quant_lab_enforcement.csv": _csv_member(
            "v5/v5_quant_lab_enforcement.csv",
            v5_enforcement,
        ),
        "v5/v5_quant_lab_usage.csv": _csv_member("v5/v5_quant_lab_usage.csv", v5_usage),
        "v5/v5_quant_lab_compliance.csv": _csv_member(
            "v5/v5_quant_lab_compliance.csv",
            v5_compliance,
        ),
        "v5/v5_quant_lab_cost_usage.csv": _csv_member(
            "v5/v5_quant_lab_cost_usage.csv", v5_cost_usage
        ),
        "v5/v5_quant_lab_fallbacks.csv": _csv_member(
            "v5/v5_quant_lab_fallbacks.csv", v5_fallbacks
        ),
    }


def _chart_members(frames: dict[str, pl.DataFrame]) -> dict[str, _MemberPayload]:
    market = frames.get("market_bar", pl.DataFrame())
    gates = frames.get("gate_decision", pl.DataFrame())
    costs = frames.get("cost_bucket_daily", pl.DataFrame())
    v5_health = frames.get("strategy_health_daily", pl.DataFrame())

    closes = _numeric_values(_tail_by_time(market, "ts"), "close")
    returns = [
        (current / previous) - 1.0
        for previous, current in zip(closes, closes[1:], strict=False)
        if previous
    ]
    gate_counts = _value_counts(gates, "status")
    cost_values = _numeric_values(costs, "total_cost_bps_p75")
    v5_values = _v5_health_values(v5_health)

    return {
        "charts/market_close.png": _png_chart("market_close", closes),
        "charts/market_returns.png": _png_chart("market_returns", returns),
        "charts/gate_status_counts.png": _png_bar_chart("gate_status_counts", gate_counts),
        "charts/cost_distribution.png": _png_chart("cost_distribution", cost_values),
        "charts/v5_health_summary.png": _png_bar_chart("v5_health_summary", v5_values),
    }


def _manifest_payload(
    *,
    day: date,
    generated_at: datetime,
    root: Path,
    profile: str,
    missing_sections: list[str],
    warnings: list[str],
    members: dict[str, _MemberPayload],
    row_counts: dict[str, int],
    command_line: list[str],
    snapshot: _DatasetSnapshot,
) -> dict[str, Any]:
    git = _git_info()
    return {
        "export_date": day.isoformat(),
        "generated_at": generated_at.isoformat(),
        "profile": profile,
        "quant_lab_version": __version__,
        "git_commit": git["git_commit"],
        "git_branch": git["git_branch"],
        "dirty_worktree": git["dirty_worktree"],
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "export_command": " ".join(command_line),
        "lake_root": str(root),
        "sections": SECTIONS,
        "missing_sections": missing_sections,
        "warnings": warnings,
        "row_counts": row_counts,
        "dataset_freshness": {
            name: _dataset_freshness_payload(name, frame)
            for name, frame in sorted(snapshot.frames.items())
        },
        "files": [
            {
                "path": path,
                "kind": Path(path).suffix.lstrip(".") or "text",
                "rows": _member_row_count(path, text),
                "sha256": _sha256_payload(text),
            }
            for path, text in sorted(members.items())
        ],
    }


def _provenance_payload(
    *,
    day: date,
    generated_at: datetime,
    root: Path,
    profile: str,
    snapshot: _DatasetSnapshot,
    command_line: list[str],
) -> dict[str, Any]:
    git = _git_info()
    return {
        "export_date": day.isoformat(),
        "generated_at": generated_at.isoformat(),
        "quant_lab_version": __version__,
        "git_commit": git["git_commit"],
        "git_branch": git["git_branch"],
        "dirty_worktree": git["dirty_worktree"],
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "export_command": " ".join(command_line),
        "lake_root": str(root),
        "datasets": [
            {
                "name": name,
                "path": str(root / readers.DATASET_PATHS[name]),
                "row_count": snapshot.row_counts[name],
                "min_timestamp": _min_timestamp(name, frame),
                "max_timestamp": _max_timestamp(name, frame),
                **_dataset_freshness_payload(name, frame),
            }
            for name, frame in sorted(snapshot.frames.items())
        ],
        "export_profile": profile,
        "command_line": command_line,
        "code_version": __version__,
    }


def _data_quality_payload(
    lake_root: Path,
    snapshot: _DatasetSnapshot,
    day: date,
) -> dict[str, Any]:
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    try:
        health = readers.data_health_summary(lake_root)
    except Exception as exc:
        health = {"warnings": [f"data health summary failed: {exc}"]}
    warnings.extend(str(item) for item in health.get("warnings", []))

    market = snapshot.frames.get("market_bar", pl.DataFrame())
    checks.append(
        _check(
            "market_bar_present",
            market.height > 0,
            f"rows={market.height}",
            severity="critical",
        )
    )
    checks.append(
        _check(
            "market_bar_duplicate_primary_key",
            int(health.get("duplicate_bar_count", 0) or 0) == 0,
            f"duplicates={health.get('duplicate_bar_count', 0)}",
        )
    )
    checks.append(
        _check(
            "closed_bar_only",
            int(health.get("unclosed_bar_count", 0) or 0) == 0,
            f"unclosed={health.get('unclosed_bar_count', 0)}",
        )
    )
    schema_violations = health.get("schema_violations", []) or []
    checks.append(
        _check(
            "market_bar_schema",
            len(schema_violations) == 0,
            "; ".join(str(item) for item in schema_violations) or "ok",
        )
    )

    cost_api_enabled = _flag_enabled("QUANT_LAB_COST_API_ENABLED", default=True)
    v5_integration_enabled = _flag_enabled("QUANT_LAB_V5_INTEGRATION_ENABLED", default=True)
    research_enabled = _flag_enabled("QUANT_LAB_RESEARCH_ENABLED", default=False)

    costs = snapshot.frames.get("cost_bucket_daily", pl.DataFrame())
    fallback_rows = _cost_fallbacks(costs)
    checks.append(
        _check(
            "cost_bucket_daily_present",
            costs.height > 0,
            f"rows={costs.height}",
            severity="critical" if cost_api_enabled else "warning",
        )
    )
    if costs.is_empty():
        checks.append(
            _check(
                "cost_fallback_ratio",
                False,
                "cost_bucket_daily empty; fallback ratio not computed",
                status="N/A",
                severity="warning",
            )
        )
    else:
        fallback_ratio = fallback_rows.height / costs.height
        checks.append(
            _check(
                "cost_fallback_ratio",
                fallback_ratio <= 0.25,
                f"fallback_rows={fallback_rows.height}; ratio={fallback_ratio:.2%}",
                severity="warning",
            )
        )

    gates = snapshot.frames.get("gate_decision", pl.DataFrame())
    evidence = snapshot.frames.get("alpha_evidence", pl.DataFrame())
    checks.append(
        _check(
            "gate_decision_present",
            gates.height > 0,
            f"rows={gates.height}",
            severity="critical" if research_enabled else "warning",
        )
    )
    checks.append(
        _check(
            "alpha_evidence_present",
            evidence.height > 0,
            f"rows={evidence.height}",
            severity="critical" if research_enabled else "warning",
        )
    )
    checks.append(
        _check(
            "gate_has_alpha_evidence",
            _gates_have_evidence(gates, evidence),
            "gate_decision rows should have matching alpha_evidence when both datasets exist",
            warning_only=True,
        )
    )

    risk = _risk_permissions_for_export(
        snapshot.frames.get("risk_permission", pl.DataFrame()),
        snapshot.frames,
    )
    checks.append(
        _check(
            "risk_permission_present",
            risk.height > 0,
            f"rows={risk.height}",
            severity="critical" if v5_integration_enabled else "warning",
        )
    )
    checks.append(
        _check(
            "risk_permission_versions",
            _risk_has_versions(risk),
            "risk_permission requires gate_version and cost_model_version",
            warning_only=True,
        )
    )
    risk_quality = _risk_permission_quality(risk, snapshot.frames)
    risk_is_stale_vs_v5 = str(risk_quality["permission_status"]).startswith("STALE_")
    checks.append(
        _check(
            "risk_permission_fresh_vs_v5_telemetry",
            not risk_is_stale_vs_v5,
            _risk_permission_quality_detail(risk_quality),
            severity="critical",
        )
    )
    v5_versions = _v5_strategy_versions(snapshot.frames)
    risk_versions = _frame_values(risk, "version")
    version_mismatch = bool(
        v5_versions and risk_versions and not v5_versions.intersection(risk_versions)
    )
    checks.append(
        _check(
            "risk_permission_version_matches_v5",
            not version_mismatch,
            f"risk_versions={sorted(risk_versions)}; v5_versions={sorted(v5_versions)}"
            if version_mismatch
            else "ok",
            warning_only=True,
        )
    )
    fills = snapshot.frames.get("okx_private_readonly_fills", pl.DataFrame())
    bills = snapshot.frames.get("okx_private_readonly_bills", pl.DataFrame())
    proxy_only = _all_cost_rows_public_proxy(costs)
    checks.append(
        _check(
            "okx_private_actual_cost_available",
            not (proxy_only and fills.height < 5 and bills.height < 5),
            "actual cost unavailable; cost model is public_spread_proxy only",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "okx_ws_universe_complete",
            _okx_ws_universe_complete(snapshot.frames),
            "okx_ws_universe_incomplete",
            warning_only=True,
        )
    )

    stale = _stale_rows(snapshot.frames)
    checks.append(
        _check(
            "stale_dataset_check",
            stale.height == 0,
            f"stale_or_missing={stale.height}",
            severity=_stale_severity(stale),
        )
    )

    for check in checks:
        if check["status"] != "PASS":
            warnings.append(f"{check['name']}: {check['detail']}")

    status = _overall_quality_status(checks)

    return {
        "status": status,
        "export_date": day.isoformat(),
        "checks": checks,
        "warnings": sorted(set(warnings)),
        "risk_permission": risk_quality,
    }


def _check(
    name: str,
    passed: bool,
    detail: str,
    *,
    warning_only: bool = False,
    severity: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    if status is None:
        status = "PASS" if passed else ("WARN" if warning_only else "FAIL")
    if severity is None:
        severity = "warning" if status in {"WARN", "N/A"} or warning_only else "critical"
    return {"name": name, "status": status, "severity": severity, "detail": detail}


def _readme(day: date, root: Path, profile: str) -> str:
    return "\n".join(
        [
            f"# quant-lab expert pack {day.isoformat()}",
            "",
            "This package is a read-only daily export from the quant-lab lake/catalog.",
            "It is intended for expert review of data quality, costs, alpha gates, "
            "and risk permissions.",
            "",
            f"- export_date: {day.isoformat()}",
            f"- profile: {profile}",
            f"- lake_root: {root}",
            "",
            "Start with `executive_summary.md`, then inspect `data_quality.json`,",
            "`manifest.json`, and the section CSV/JSON files.",
            "",
        ]
    )


def _executive_summary(
    day: date,
    snapshot: _DatasetSnapshot,
    data_quality: dict[str, Any],
) -> str:
    gates = snapshot.frames.get("gate_decision", pl.DataFrame())
    risk = snapshot.frames.get("risk_permission", pl.DataFrame())
    gate_counts = _value_counts(gates, "status")
    risk_quality = data_quality.get("risk_permission", {})
    permission_counts = _value_counts(risk, "permission_status")
    if not permission_counts:
        permission_counts = _value_counts(risk, "permission")
    fallback_rows = _cost_fallbacks(snapshot.frames.get("cost_bucket_daily", pl.DataFrame()))
    questions = _question_lines(snapshot, data_quality)

    lines = [
        f"# Executive Summary - {day.isoformat()}",
        "",
        f"Data quality status: {data_quality.get('status', 'UNKNOWN')}",
        f"Dataset row counts: {safe_json_dumps(snapshot.row_counts)}",
        f"Cost fallback rows: {fallback_rows.height}",
        f"Gate status counts: {safe_json_dumps(gate_counts)}",
        f"Risk permission counts: {safe_json_dumps(permission_counts)}",
        f"Risk permission status: {risk_quality.get('permission_status', 'UNKNOWN')}",
        f"Risk permission as_of_ts: {risk_quality.get('as_of_ts')}",
        f"Latest V5 telemetry ts: {risk_quality.get('telemetry_latest_ts')}",
        f"Risk permission next action: {risk_quality.get('next_action', 'none')}",
        "",
        "Warnings:",
    ]
    warnings = data_quality.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings[:20])
    if not warnings:
        lines.append("- none")
    lines.extend(["", "Questions for expert review:"])
    lines.extend(questions)
    lines.append("")
    return "\n".join(lines)


def _expert_questions(snapshot: _DatasetSnapshot, data_quality: dict[str, Any]) -> str:
    return "\n".join([*_question_lines(snapshot, data_quality), ""])


def _question_lines(snapshot: _DatasetSnapshot, data_quality: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    costs = snapshot.frames.get("cost_bucket_daily", pl.DataFrame())
    if (
        snapshot.row_counts.get("trade_print", 0) == 0
        or snapshot.row_counts.get("orderbook_snapshot", 0) == 0
        or snapshot.row_counts.get("okx_public_ws", 0) == 0
    ):
        questions.append("是否已配置 OKX WebSocket trades/books 采集？")
    if (
        snapshot.row_counts.get("okx_private_readonly_fills", 0) == 0
        and snapshot.row_counts.get("okx_private_readonly_bills", 0) == 0
    ):
        questions.append("是否启用 OKX read-only fills/bills？")
    if snapshot.row_counts.get("cost_bucket_daily", 0) == 0:
        questions.append("为什么 cost_bucket_daily 为空？")
    elif _cost_model_is_proxy_or_fallback(costs):
        questions.append("成本模型仍是 public spread proxy，何时启用真实 fills/bills？")
    if snapshot.row_counts.get("feature_value", 0) == 0:
        questions.append("为什么 feature_value 为空？")
    if snapshot.row_counts.get("gate_decision", 0) == 0:
        questions.append("为什么 gate_decision / alpha_evidence 为空？")
    elif snapshot.row_counts.get("alpha_evidence", 0) == 0:
        questions.append("alpha_evidence 研究证据尚未生成：何时运行 walk-forward/evidence 任务？")
    if snapshot.row_counts.get("risk_permission", 0) == 0:
        questions.append("为什么 risk_permission 为空？")
    if snapshot.row_counts.get("strategy_health_daily", 0) == 0:
        questions.append("V5 telemetry 是否同步成功？")
    if snapshot.row_counts.get("v5_quant_lab_usage", 0) == 0:
        questions.append("V5 是否已接入 quant-lab API？当前 v5_quant_lab_usage 为空。")
    if _latest_v5_mode(snapshot.frames) == "unknown":
        questions.append(
            "V5 bundle is missing quant_lab mode fields; add mode and "
            "permission_gate_enforced to usage/compliance exports."
        )
    config_question = _config_not_consumed_question(
        snapshot.frames.get("v5_config_health_daily", pl.DataFrame())
    )
    if config_question:
        questions.append(config_question)
    warnings = data_quality.get("warnings", [])
    if warnings:
        questions.append("哪些 data_quality 告警需要先处理，才能继续研究复盘？")
    if not questions:
        questions.extend(
            [
                "是否有 gate status 变化需要人工复核？",
                "今日 cost fallback 比例是否可接受？",
                "是否需要在 V5/V7 消费前下调风险许可？",
            ]
        )
    return [f"{index}. {question}" for index, question in enumerate(questions[:10], start=1)]


def _cost_model_is_proxy_or_fallback(costs: pl.DataFrame) -> bool:
    if costs.is_empty():
        return False
    text_columns = [column for column in ["source", "fallback_level"] if column in costs.columns]
    if not text_columns:
        return False
    for row in costs.select(text_columns).to_dicts():
        rendered = " ".join(str(value).lower() for value in row.values())
        if "public_spread_proxy" in rendered or "global_default" in rendered:
            return True
    return _cost_fallbacks(costs).height == costs.height


def _all_cost_rows_public_proxy(costs: pl.DataFrame) -> bool:
    if costs.is_empty() or "source" not in costs.columns:
        return False
    return costs.filter(pl.col("source") == "public_spread_proxy").height == costs.height


def _alpha_evidence_for_export(evidence: pl.DataFrame) -> pl.DataFrame:
    if evidence.is_empty() and not evidence.columns:
        return evidence
    normalized = evidence
    if "evidence_status" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit("unknown").alias("evidence_status"))
    else:
        normalized = normalized.with_columns(
            pl.when(pl.col("evidence_status").is_null() | (pl.col("evidence_status") == ""))
            .then(pl.lit("unknown"))
            .otherwise(pl.col("evidence_status").cast(pl.Utf8))
            .alias("evidence_status")
        )
    return normalized


def _latest_v5_mode(frames: dict[str, pl.DataFrame]) -> str | None:
    mode = frames.get("v5_quant_lab_mode_daily", pl.DataFrame())
    if mode.is_empty() or "mode" not in mode.columns:
        return None
    row = _latest_by_dataset_time("v5_quant_lab_mode_daily", mode)
    return str(row.get("mode") or "").strip().lower() or None


def _latest_v5_bundle_ts(frames: dict[str, pl.DataFrame]) -> datetime | None:
    timestamps = [
        _latest_dataset_timestamp(name, frames.get(name, pl.DataFrame()))
        for name in [
            "strategy_health_daily",
            "v5_quant_lab_mode_daily",
            "v5_quant_lab_enforcement_daily",
            "v5_gate_compliance_daily",
        ]
    ]
    parsed = [value for value in timestamps if value is not None]
    return max(parsed) if parsed else None


def _risk_permission_quality(
    risk: pl.DataFrame,
    frames: dict[str, pl.DataFrame],
) -> dict[str, Any]:
    telemetry_latest_ts = _latest_v5_bundle_ts(frames)
    if risk.is_empty():
        return {
            "permission_status": "NO_FRESH_PERMISSION",
            "permission": None,
            "enforceable": False,
            "as_of_ts": None,
            "created_at": None,
            "source_bundle_ts": None,
            "telemetry_latest_ts": _iso_or_none(telemetry_latest_ts),
            "permission_freshness_sec": None,
            "threshold_seconds": DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
            "degraded_reason": "risk_permission_missing",
            "next_action": "run qlab publish-risk-permission after sync-v5-telemetry",
        }
    row = _latest_by_dataset_time("risk_permission", risk)
    permission = str(row.get("permission") or "ABORT").strip().upper()
    as_of_ts = _risk_row_timestamp(row, ["as_of_ts", "created_at"])
    created_at = _risk_row_timestamp(row, ["created_at"])
    row_telemetry_ts = _risk_row_timestamp(row, ["telemetry_latest_ts", "source_bundle_ts"])
    effective_telemetry_ts = max(
        [value for value in [telemetry_latest_ts, row_telemetry_ts] if value is not None],
        default=None,
    )
    stored_status = str(row.get("permission_status") or "").strip().upper()
    stale = risk_permission_stale_vs_telemetry(
        as_of_ts=as_of_ts,
        telemetry_latest_ts=effective_telemetry_ts,
        threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    )
    status = (
        stored_status
        if stored_status.startswith(("ACTIVE_", "STALE_"))
        and stored_status.endswith(permission)
        and stored_status.startswith("STALE_") == stale
        else permission_status(permission, stale=stale).value
    )
    freshness = permission_freshness_sec(
        as_of_ts=as_of_ts,
        telemetry_latest_ts=effective_telemetry_ts,
    )
    enforceable = is_permission_status_enforceable(status)
    return {
        "permission_status": status,
        "permission": permission,
        "enforceable": enforceable,
        "as_of_ts": _iso_or_none(as_of_ts),
        "created_at": _iso_or_none(created_at),
        "source_bundle_ts": _iso_or_none(row_telemetry_ts),
        "telemetry_latest_ts": _iso_or_none(effective_telemetry_ts),
        "permission_freshness_sec": freshness,
        "threshold_seconds": DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
        "degraded_reason": "stale_vs_v5_telemetry" if stale else "none",
        "next_action": (
            "run qlab publish-risk-permission after sync-v5-telemetry"
            if stale
            else "none"
        ),
    }


def _risk_permissions_for_export(
    risk: pl.DataFrame,
    frames: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    if risk.is_empty():
        return risk
    telemetry_latest_ts = _latest_v5_bundle_ts(frames)
    rows: list[dict[str, Any]] = []
    for row in risk.to_dicts():
        permission = str(row.get("permission") or "ABORT").strip().upper()
        as_of_ts = _risk_row_timestamp(row, ["as_of_ts", "created_at"])
        row_telemetry_ts = _risk_row_timestamp(row, ["telemetry_latest_ts", "source_bundle_ts"])
        effective_telemetry_ts = max(
            [value for value in [telemetry_latest_ts, row_telemetry_ts] if value is not None],
            default=None,
        )
        stale = risk_permission_stale_vs_telemetry(
            as_of_ts=as_of_ts,
            telemetry_latest_ts=effective_telemetry_ts,
            threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
        )
        status = permission_status(permission, stale=stale).value
        rows.append(
            row
            | {
                "as_of_ts": _iso_or_none(as_of_ts),
                "telemetry_latest_ts": _iso_or_none(effective_telemetry_ts),
                "permission_freshness_sec": permission_freshness_sec(
                    as_of_ts=as_of_ts,
                    telemetry_latest_ts=effective_telemetry_ts,
                ),
                "contract_version": row.get("contract_version") or "risk_permission.v0.2",
                "permission_status": status,
            }
        )
    return pl.DataFrame(rows) if rows else risk


def _risk_permission_quality_detail(risk_quality: dict[str, Any]) -> str:
    status = risk_quality.get("permission_status")
    if status == "NO_FRESH_PERMISSION":
        return (
            "NO_FRESH_PERMISSION; latest_v5_telemetry="
            f"{risk_quality.get('telemetry_latest_ts')}; next_action="
            f"{risk_quality.get('next_action')}"
        )
    if str(status).startswith("STALE_"):
        return (
            f"risk_permission_stale_vs_v5_telemetry; permission_status={status}; "
            f"as_of_ts={risk_quality.get('as_of_ts')}; "
            f"telemetry_latest_ts={risk_quality.get('telemetry_latest_ts')}; "
            f"permission_freshness_sec={risk_quality.get('permission_freshness_sec')}; "
            f"next_action={risk_quality.get('next_action')}"
        )
    return (
        f"ok; permission_status={status}; as_of_ts={risk_quality.get('as_of_ts')}; "
        f"telemetry_latest_ts={risk_quality.get('telemetry_latest_ts')}"
    )


def _risk_row_timestamp(row: dict[str, Any], fields: list[str]) -> datetime | None:
    for field in fields:
        value = readers._coerce_timestamp(row.get(field))  # type: ignore[attr-defined]
        if value is not None:
            return value
    return None


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _latest_dataset_timestamp(dataset_name: str, frame: pl.DataFrame) -> datetime | None:
    latest, _column = readers.latest_dataset_timestamp(dataset_name, frame)
    return latest


def _latest_by_dataset_time(dataset_name: str, frame: pl.DataFrame) -> dict[str, Any]:
    latest, column = readers.latest_dataset_timestamp(dataset_name, frame)
    if latest is not None and column in frame.columns:
        parsed = frame.with_columns(
            pl.col(column).map_elements(
                readers._coerce_timestamp,  # type: ignore[attr-defined]
                return_dtype=pl.Datetime(time_zone="UTC"),
            )
        )
        return parsed.sort(column).tail(1).to_dicts()[0]
    return frame.tail(1).to_dicts()[0]


def _frame_values(frame: pl.DataFrame, column: str) -> set[str]:
    if frame.is_empty() or column not in frame.columns:
        return set()
    return {
        str(value).strip()
        for value in frame[column].drop_nulls().to_list()
        if str(value).strip()
    }


def _v5_strategy_versions(frames: dict[str, pl.DataFrame]) -> set[str]:
    versions: set[str] = set()
    for name in [
        "v5_quant_lab_usage",
        "v5_quant_lab_compliance",
        "v5_quant_lab_cost_usage",
        "v5_quant_lab_fallback",
    ]:
        frame = frames.get(name, pl.DataFrame())
        if frame.is_empty():
            continue
        for row in frame.to_dicts():
            payload = _json_payload(row.get("raw_payload_json"))
            for field in [
                "strategy_version",
                "version",
                "v5_version",
                "quant_lab.strategy_version",
            ]:
                value = _row_or_payload_value(row, payload, field)
                if value is not None and str(value).strip():
                    versions.add(str(value).strip())
    return versions


def _json_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _row_or_payload_value(row: dict[str, Any], payload: dict[str, Any], field: str) -> Any:
    value = row.get(field)
    if value is not None and str(value).strip():
        return value
    nested: Any = payload
    for part in field.split("."):
        if not isinstance(nested, dict):
            return None
        nested = nested.get(part)
    return nested


def _okx_ws_universe_complete(frames: dict[str, pl.DataFrame]) -> bool:
    expected = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"}
    observed = _symbols(frames.get("trade_print", pl.DataFrame())) | _symbols(
        frames.get("orderbook_snapshot", pl.DataFrame())
    )
    return expected.issubset(observed)


def _symbols(frame: pl.DataFrame) -> set[str]:
    if frame.is_empty() or "symbol" not in frame.columns:
        return set()
    return {
        normalize_symbol(value)
        for value in frame["symbol"].drop_nulls().to_list()
        if str(value).strip()
    }


def _normalize_symbol_frame(frame: pl.DataFrame, column: str = "symbol") -> pl.DataFrame:
    if frame.is_empty() or column not in frame.columns:
        return frame
    return frame.with_columns(
        pl.col(column).map_elements(normalize_symbol, return_dtype=pl.Utf8).alias(column)
    )


def _config_not_consumed_question(config_health: pl.DataFrame) -> str | None:
    if config_health.is_empty() or "config_not_consumed_count" not in config_health.columns:
        return None
    row = config_health.tail(1).to_dicts()[0]
    try:
        count = int(row.get("config_not_consumed_count") or 0)
    except (TypeError, ValueError):
        count = 0
    unknown = str(row.get("config_not_consumed_count_unknown", "")).lower() in {
        "true",
        "1",
        "yes",
    }
    if count <= 0 and not unknown:
        return None
    top_keys = _config_top_keys(row.get("config_not_consumed_top_keys_json"))
    if top_keys:
        return (
            f"config_not_consumed_count 是否真实为 {count}？"
            f"需要复核 top keys: {', '.join(top_keys[:10])}。"
        )
    if unknown:
        return "config_not_consumed_count 无法结构化解析，需要列出 top keys。"
    return f"config_not_consumed_count 是否真实为 {count}？需要列出 top keys。"


def _config_top_keys(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload if str(item)]
    return []


def _latest_market_rows(market: pl.DataFrame) -> pl.DataFrame:
    if market.is_empty() or not {"symbol", "timeframe", "ts"}.issubset(market.columns):
        return pl.DataFrame(schema={"symbol": pl.Utf8, "timeframe": pl.Utf8, "latest_ts": pl.Utf8})
    return (
        _sort_frame(market, "ts")
        .group_by(["symbol", "timeframe"], maintain_order=True)
        .tail(1)
        .sort(["symbol", "timeframe"])
    )


def _tail_by_time(df: pl.DataFrame, column: str, limit: int = 500) -> pl.DataFrame:
    if df.is_empty():
        return df
    return _sort_frame(df, column).tail(limit) if column in df.columns else df.head(limit)


def _feature_coverage(features: pl.DataFrame) -> pl.DataFrame:
    if features.is_empty() or not {"feature_name", "symbol"}.issubset(features.columns):
        return _empty_csv_schema_frame("features/feature_coverage.csv")
    return (
        features.group_by(["feature_set", "feature_name", "feature_version", "timeframe", "symbol"])
        .agg(
            [
                pl.len().alias("total_rows"),
                pl.col("value").is_not_null().sum().cast(pl.Int64).alias("valid_rows"),
                pl.col("value").is_null().sum().cast(pl.Int64).alias("null_rows"),
                pl.col("ts").min().alias("min_ts"),
                pl.col("ts").max().alias("max_ts"),
                pl.col("created_at").max().alias("created_at"),
            ]
        )
        .with_columns(
            [
                pl.col("min_ts")
                .cast(pl.Datetime(time_zone="UTC"))
                .dt.date()
                .cast(pl.Utf8)
                .alias("day"),
                (pl.col("valid_rows") / pl.col("total_rows")).alias("coverage"),
            ]
        )
        .select(CSV_SCHEMAS["features/feature_coverage.csv"])
        .sort(["feature_name", "symbol"])
    )


def _feature_anomalies(features: pl.DataFrame) -> pl.DataFrame:
    if features.is_empty() or "value" not in features.columns:
        return _empty_csv_schema_frame("features/feature_anomalies.csv")
    rows = features.filter(pl.col("value").is_null())
    if rows.is_empty():
        return _empty_csv_schema_frame("features/feature_anomalies.csv")
    return (
        rows.with_columns(
            [
                pl.col("ts").cast(pl.Datetime(time_zone="UTC")).dt.date().cast(pl.Utf8).alias("day"),
                pl.lit("null_value").alias("anomaly_type"),
                pl.lit(1).alias("anomaly_count"),
                pl.lit("warning").alias("severity"),
                pl.col("ts").alias("example_ts"),
            ]
        )
        .group_by(
            [
                "day",
                "feature_set",
                "feature_name",
                "feature_version",
                "timeframe",
                "symbol",
                "anomaly_type",
                "severity",
            ]
        )
        .agg(
            [
                pl.col("anomaly_count").sum(),
                pl.col("example_ts").min(),
                pl.col("created_at").max(),
            ]
        )
        .select(CSV_SCHEMAS["features/feature_anomalies.csv"])
    )


def _empty_csv_schema_frame(path: str) -> pl.DataFrame:
    return pl.DataFrame(schema={column: pl.Utf8 for column in CSV_SCHEMAS[path]})


def _cost_examples(costs: pl.DataFrame) -> dict[str, Any]:
    return {"rows": _rows(costs.head(20) if not costs.is_empty() else costs)}


def _cost_fallbacks(costs: pl.DataFrame) -> pl.DataFrame:
    if costs.is_empty() or "fallback_level" not in costs.columns:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "fallback_level": pl.Utf8})
    return costs.filter(~pl.col("fallback_level").is_in(["actual_okx_fills_and_bills", "NONE"]))


def _risk_flags(risk: pl.DataFrame) -> pl.DataFrame:
    if risk.is_empty() or "permission" not in risk.columns:
        return pl.DataFrame(
            schema={
                "strategy": pl.Utf8,
                "permission": pl.Utf8,
                "permission_status": pl.Utf8,
                "enforceable": pl.Boolean,
                "reason": pl.Utf8,
            }
        )
    rows: list[dict[str, Any]] = []
    for row in risk.to_dicts():
        status = str(row.get("permission_status") or "").strip()
        permission = str(row.get("permission") or "").strip()
        if not status:
            status = permission_status(permission, stale=False).value
        stale = status.startswith("STALE_")
        risk_limited = permission in {"SELL_ONLY", "ABORT"}
        if stale or risk_limited:
            rows.append(
                {
                    "strategy": row.get("strategy"),
                    "permission": permission,
                    "permission_status": status,
                    "enforceable": is_permission_status_enforceable(status),
                    "reason": "stale_permission" if stale else "risk_limited_permission",
                }
            )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "strategy": pl.Utf8,
                "permission": pl.Utf8,
                "permission_status": pl.Utf8,
                "enforceable": pl.Boolean,
                "reason": pl.Utf8,
            }
        )
    )


def _anomaly_rows(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _missing_dataset_rows(frames).to_dicts():
        rows.append({"type": "missing_data", **item})
    for item in _schema_violation_rows(frames.get("market_bar", pl.DataFrame())).to_dicts():
        rows.append({"type": "schema_violation", **item})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"type": pl.Utf8})


def _missing_dataset_rows(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = [
        {"dataset": name, "reason": "missing_or_empty"}
        for name, frame in sorted(frames.items())
        if frame.is_empty()
    ]
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(schema={"dataset": pl.Utf8, "reason": pl.Utf8})
    )


def _stale_rows(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for name, frame in sorted(frames.items()):
        freshness = _dataset_freshness_payload(name, frame)
        status = freshness["freshness_status"]
        if status in {"missing", "unknown", "stale"}:
            rows.append(
                {
                    "dataset": name,
                    "status": status,
                    "row_count": frame.height,
                    "timestamp_column": freshness["timestamp_column"] or "",
                    "latest_timestamp": freshness["latest_timestamp"] or "",
                }
            )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "dataset": pl.Utf8,
                "status": pl.Utf8,
                "row_count": pl.Int64,
                "timestamp_column": pl.Utf8,
                "latest_timestamp": pl.Utf8,
            }
        )
    )


def _schema_violation_rows(market: pl.DataFrame) -> pl.DataFrame:
    violations = readers.market_bar_schema_violations(market) if not market.is_empty() else []
    rows = [{"dataset": "market_bar", "violation": item} for item in violations]
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(schema={"dataset": pl.Utf8, "violation": pl.Utf8})
    )


def _gates_have_evidence(gates: pl.DataFrame, evidence: pl.DataFrame) -> bool:
    if gates.is_empty() or evidence.is_empty():
        return False
    keys = [
        key for key in ["alpha_id", "version"] if key in gates.columns and key in evidence.columns
    ]
    if not keys:
        return False
    joined = gates.join(evidence.select(keys).unique(), on=keys, how="anti")
    return joined.is_empty()


def _risk_has_versions(risk: pl.DataFrame) -> bool:
    if risk.is_empty():
        return False
    required = {"gate_version", "cost_model_version"}
    return required.issubset(risk.columns) and risk.filter(
        pl.any_horizontal([pl.col(column).is_null() for column in sorted(required)])
    ).is_empty()


def _missing_sections(row_counts: dict[str, int]) -> list[str]:
    missing: list[str] = []
    for section, datasets in SECTION_DATASETS.items():
        if all(row_counts.get(name, 0) == 0 for name in datasets):
            missing.append(section)
    return missing


def _write_zip(path: Path, members: dict[str, _MemberPayload]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in sorted(members.items()):
            data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
            archive.writestr(member, data)


def _fail_on_secrets(members: dict[str, _MemberPayload]) -> None:
    findings: list[str] = []
    for member, text in members.items():
        if isinstance(text, bytes):
            continue
        high, medium = _secret_severity_counts(text)
        if high or medium:
            findings.append(
                f"{member}: {high} high, "
                f"{medium} medium"
            )
    if findings:
        rendered = "; ".join(findings)
        raise ValueError(f"expert pack contains possible secrets: {rendered}")


def _zip_secret_reasons(archive: zipfile.ZipFile, names: list[str]) -> list[str]:
    reasons: list[str] = []
    for name in names:
        if not _is_text_member(name):
            continue
        try:
            text = archive.read(name).decode("utf-8")
        except UnicodeDecodeError:
            continue
        high, medium = _secret_severity_counts(text)
        if high or medium:
            reasons.append(
                f"{name} contains possible secrets: "
                f"{high} high, {medium} medium"
            )
    return reasons


def _secret_severity_counts(text: str) -> tuple[int, int]:
    high = 0
    medium = 0
    for line in text.splitlines():
        for pattern, severity, _label in SECRET_PATTERNS:
            if not pattern.search(line):
                continue
            if severity == "high":
                high += 1
            elif severity == "medium":
                medium += 1
    return high, medium


def _unsafe_zip_member_reasons(names: list[str]) -> list[str]:
    reasons: list[str] = []
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or "\\" in name or ".." in pure.parts:
            reasons.append(f"unsafe zip member path: {name}")
    return reasons


def _read_zip_json(
    archive: zipfile.ZipFile,
    member: str,
    reasons: list[str],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(archive.read(member).decode("utf-8"))
    except KeyError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        reasons.append(f"{member} is invalid JSON: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def _csv_member(path: str, df: pl.DataFrame) -> str:
    return _csv_text(df, fixed_columns=CSV_SCHEMAS.get(path))


def _csv_text(df: pl.DataFrame, fixed_columns: list[str] | None = None) -> str:
    safe = _csv_frame_with_schema(readers.redact_frame(df), fixed_columns)
    columns = safe.columns if safe.columns else (fixed_columns or [])
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in _rows(safe):
        writer.writerow({column: _csv_value(row.get(column)) for column in columns})
    return handle.getvalue()


def _csv_frame_with_schema(df: pl.DataFrame, fixed_columns: list[str] | None) -> pl.DataFrame:
    if not fixed_columns:
        return df
    if df.is_empty() and not df.columns:
        return pl.DataFrame(schema={column: pl.Utf8 for column in fixed_columns})
    normalized = df
    for column in fixed_columns:
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    extra_columns = [column for column in normalized.columns if column not in fixed_columns]
    return normalized.select([*fixed_columns, *extra_columns])


def _rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    if df.is_empty():
        return []
    return [dict(row) for row in df.to_dicts()]


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat() if value.tzinfo else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict | list | tuple):
        return safe_json_dumps(value)
    return value


def _json_text(payload: Any) -> str:
    return safe_json_dumps(payload) + "\n"


def _value_counts(df: pl.DataFrame, column: str) -> dict[str, int]:
    if df.is_empty() or column not in df.columns:
        return {}
    return {
        str(row[column]): int(row["count"])
        for row in df.group_by(column).len(name="count").to_dicts()
    }


def _numeric_values(df: pl.DataFrame, column: str) -> list[float]:
    if df.is_empty() or column not in df.columns:
        return []
    values: list[float] = []
    for value in df[column].to_list():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        values.append(number)
    return values


def _v5_health_values(v5_health: pl.DataFrame) -> dict[str, int]:
    if v5_health.is_empty():
        return {}
    latest = v5_health.tail(1).to_dicts()[0]
    return {
        "high_issues": int(latest.get("high_issue_count") or 0),
        "medium_issues": int(latest.get("medium_issue_count") or 0),
        "open_positions": int(latest.get("open_position_count") or 0),
        "trades_24h": int(latest.get("trade_count_24h") or 0),
    }


def _png_chart(title: str, values: list[float]) -> bytes:
    if not values:
        return _png_placeholder(title, "missing data")
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return _minimal_png()

    width, height = 720, 360
    margin = 48
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill=(30, 30, 30))
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-12)
    points = []
    for index, value in enumerate(values):
        x = margin + (width - 2 * margin) * (index / max(len(values) - 1, 1))
        y = height - margin - ((value - min_value) / span) * (height - 2 * margin)
        points.append((x, y))
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(220, 220, 220))
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(30, 90, 180))
    else:
        draw.line(points, fill=(30, 90, 180), width=3)
    draw.text((margin, height - 34), f"n={len(values)}", fill=(80, 80, 80))
    return _image_bytes(image)


def _png_bar_chart(title: str, counts: dict[str, int]) -> bytes:
    if not counts:
        return _png_placeholder(title, "missing data")
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return _minimal_png()

    width, height = 720, 360
    margin = 48
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill=(30, 30, 30))
    items = list(counts.items())[:8]
    max_count = max(value for _key, value in items) or 1
    bar_width = max(28, (width - 2 * margin) // max(len(items), 1) - 12)
    for index, (label, value) in enumerate(items):
        x0 = margin + index * (bar_width + 12)
        x1 = x0 + bar_width
        y1 = height - margin
        y0 = y1 - int((height - 2 * margin) * (value / max_count))
        draw.rectangle((x0, y0, x1, y1), fill=(30, 120, 120))
        draw.text((x0, max(y0 - 18, 36)), str(value), fill=(40, 40, 40))
        draw.text((x0, height - 34), str(label)[:14], fill=(80, 80, 80))
    return _image_bytes(image)


def _png_placeholder(title: str, message: str) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return _minimal_png()

    image = Image.new("RGB", (720, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 696, 336), outline=(220, 220, 220))
    draw.text((40, 40), title, fill=(30, 30, 30))
    draw.text((40, 82), message, fill=(120, 80, 40))
    return _image_bytes(image)


def _image_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _minimal_png() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c49444154789c6360f8ffff3f0005fe02fea73581e40000000049454e44ae426082"
    )


def _sort_frame(df: pl.DataFrame, column: str) -> pl.DataFrame:
    try:
        return df.sort(column)
    except Exception:
        return df


def _min_timestamp(dataset_name: str, df: pl.DataFrame) -> str | None:
    value = _timestamp_value(dataset_name, df, "min")
    return value.isoformat() if isinstance(value, datetime) else None


def _max_timestamp(dataset_name: str, df: pl.DataFrame) -> str | None:
    value = _timestamp_value(dataset_name, df, "max")
    return value.isoformat() if isinstance(value, datetime) else None


def _dataset_freshness_payload(dataset_name: str, df: pl.DataFrame) -> dict[str, Any]:
    return readers.dataset_freshness_payload(dataset_name, df)


def _timestamp_value(dataset_name: str, df: pl.DataFrame, op: str) -> datetime | None:
    if df.is_empty():
        return None
    for column in readers.DATASET_TIMESTAMP_COLUMNS.get(
        dataset_name,
        ("ts", "created_at", "ingest_ts"),
    ):
        if column not in df.columns:
            continue
        parsed = [
            readers._coerce_timestamp(value)  # type: ignore[attr-defined]
            for value in df[column].to_list()
        ]
        values = [value for value in parsed if value is not None]
        if not values:
            continue
        return min(values) if op == "min" else max(values)
    return None


def _overall_quality_status(checks: list[dict[str, Any]]) -> str:
    if any(
        check["status"] == "FAIL" and check.get("severity") == "critical" for check in checks
    ):
        return "CRITICAL"
    if any(check["status"] in {"FAIL", "WARN", "N/A"} for check in checks):
        return "WARN"
    return "PASS"


def _stale_severity(stale: pl.DataFrame) -> str:
    if stale.is_empty() or "dataset" not in stale.columns:
        return "warning"
    critical = {"market_bar", "cost_bucket_daily", "risk_permission"}
    critical_rows = stale.filter(
        pl.col("dataset").is_in(sorted(critical))
        & pl.col("status").is_in(["missing", "stale", "missing_or_empty"])
    )
    return "critical" if not critical_rows.is_empty() else "warning"


def _flag_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _member_row_count(path: str, text: _MemberPayload) -> int | None:
    if isinstance(text, bytes):
        return None
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        rows = list(csv.reader(io.StringIO(text)))
        return max(len(rows) - 1, 0)
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return len(payload["rows"])
    return None


def _is_text_member(name: str) -> bool:
    return Path(name).suffix.lower() in {".csv", ".json", ".md", ".txt"}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: _MemberPayload) -> str:
    if isinstance(payload, bytes):
        return hashlib.sha256(payload).hexdigest()
    return _sha256_text(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    return {
        "git_commit": _git_command(["rev-parse", "--short", "HEAD"], root),
        "git_branch": _git_command(["branch", "--show-current"], root),
        "dirty_worktree": bool(_git_command(["status", "--porcelain"], root)),
    }


def _git_commit() -> str | None:
    return _git_info()["git_commit"]


def _git_command(args: list[str], root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=root,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value or None


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)
