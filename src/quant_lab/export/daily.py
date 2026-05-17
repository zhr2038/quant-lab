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
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.contracts.v5_quant_lab import (
    V5_QUANT_LAB_CONTRACT_VERSION,
    V5_TELEMETRY_DATASET_SCHEMA_VERSION,
)
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.reports.enforce_readiness import (
    ENFORCE_READINESS_CSV,
    ENFORCE_READINESS_JSON,
    build_enforce_readiness_report,
    enforce_readiness_members,
)
from quant_lab.research.alpha_discovery import normalize_alpha_discovery_board_decisions
from quant_lab.research.strategy_evidence import normalize_strategy_evidence_decisions
from quant_lab.risk.publish import (
    DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    is_permission_status_enforceable,
    permission_freshness_sec,
    permission_status,
    publish_risk_permission,
    risk_permission_stale_vs_telemetry,
)
from quant_lab.strategy_telemetry.analyze import _event_key as _v5_telemetry_event_key
from quant_lab.strategy_telemetry.sanitize import SECRET_PATTERNS, safe_json_dumps
from quant_lab.symbols import normalize_symbol
from quant_lab.time_display import BEIJING_TZ, DISPLAY_TIMEZONE, beijing_iso
from quant_lab.web import readers

SECTIONS = ["market", "features", "costs", "research", "risk", "anomalies", "v5", "charts"]
HEAVY_EXPORT_DATASET_LIMITS = {
    "okx_public_ws": 5_000,
    "feature_value": 10_000,
    "strategy_evidence_sample": 10_000,
    "trade_print": 10_000,
    "orderbook_snapshot": 10_000,
    "v5_decision_audit": 5_000,
    "v5_trade_event": 10_000,
    "v5_quant_lab_usage": 5_000,
    "v5_quant_lab_compliance": 5_000,
    "v5_quant_lab_cost_usage": 5_000,
    "v5_quant_lab_fallback": 5_000,
    "v5_candidate_event": 20_000,
    "v5_candidate_label": 20_000,
}
HEAVY_EXPORT_RECENT_FILE_LIMITS = {
    "okx_public_ws": 50,
    "feature_value": 100,
    "strategy_evidence_sample": 100,
    "trade_print": 50,
    "orderbook_snapshot": 50,
    "v5_decision_audit": 100,
    "v5_trade_event": 100,
    "v5_quant_lab_usage": 100,
    "v5_quant_lab_compliance": 100,
    "v5_quant_lab_cost_usage": 100,
    "v5_quant_lab_fallback": 100,
    "v5_candidate_event": 100,
    "v5_candidate_label": 100,
}
SECTION_DATASETS = {
    "market": ["market_bar", "trade_print", "orderbook_snapshot", "okx_public_ws"],
    "features": ["feature_value", "feature_coverage_daily", "feature_anomaly_daily"],
    "costs": [
        "cost_bucket_daily",
        "cost_health_daily",
        "okx_private_readonly_fills",
        "okx_private_readonly_bills",
    ],
    "research": [
        "alpha_evidence",
        "alpha_discovery_board",
        "strategy_evidence",
        "strategy_evidence_sample",
        "strategy_evidence_quality",
        "paper_strategy_runs",
        "paper_strategy_daily",
        "paper_slippage_coverage",
        "gate_decision",
    ],
    "risk": ["risk_permission"],
    "anomalies": ["market_bar", "feature_value", "cost_bucket_daily", "gate_decision"],
    "v5": [
        "v5_decision_audit",
        "v5_trade_event",
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
        "v5_candidate_event",
        "v5_candidate_label",
        "v5_candidate_quality_daily",
        "v5_candidate_outcome_summary",
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
    "research/strategy_evidence.csv",
    "research/strategy_evidence_samples.csv",
    "research/strategy_evidence_quality.csv",
    "research/gate_decisions.csv",
    "research/research_conclusions.csv",
    "risk/risk_permission.json",
    "risk/risk_flags.csv",
    "anomalies/anomalies.csv",
    "anomalies/missing_data.csv",
    "anomalies/stale_data.csv",
    "anomalies/schema_violations.csv",
    "v5/v5_strategy_health.csv",
    "v5/v5_decision_audit.csv",
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
    "v5/v5_candidate_events.csv",
    "v5/v5_candidate_labels.csv",
    "v5/v5_candidate_quality.csv",
    "v5/v5_candidate_outcome_summary.csv",
    "charts/market_close.png",
    "charts/market_returns.png",
    "charts/gate_status_counts.png",
    "charts/cost_distribution.png",
    "charts/v5_health_summary.png",
    "reports/alpha_discovery_board.csv",
    "reports/strategy_evidence_summary.md",
    "reports/candidate_kill_list.csv",
    "reports/candidate_shadow_watchlist.csv",
    "reports/candidate_paper_ready.csv",
    "reports/paper_strategy_proposals.csv",
    "reports/paper_strategy_runs.csv",
    "reports/paper_strategy_daily.csv",
    "reports/paper_slippage_coverage.csv",
    f"reports/{ENFORCE_READINESS_JSON}",
    f"reports/{ENFORCE_READINESS_CSV}",
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
        "cost_source",
        "actual_fill_count",
        "mixed_fill_count",
        "proxy_sample_count",
    ],
    "costs/cost_health_daily.csv": [
        "day",
        "status",
        "cost_model_version",
        "actual_rows",
        "proxy_rows",
        "global_default_rows",
        "fallback_ratio",
        "hard_fallback_count",
        "hard_fallback_ratio",
        "soft_fallback_count",
        "soft_fallback_ratio",
        "proxy_only_count",
        "global_default_count",
        "symbols_with_actual_cost",
        "symbols_with_mixed_cost",
        "symbols_with_proxy_only",
        "symbols_proxy_only",
        "symbols_missing_cost",
        "actual_sample_count_by_symbol",
        "data_quality_checks_json",
        "min_sample_count",
        "api_global_default_count",
        "api_symbol_proxy_hit_count",
        "api_regime_fallback_count",
        "api_degraded_cost_count",
        "api_cost_usage_rows",
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
    "research/strategy_evidence.csv": [
        "strategy",
        "evidence_version",
        "candidate_name",
        "as_of_date",
        "strategy_candidate",
        "symbol",
        "regime_state",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "median_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "decision",
        "decision_reasons",
        "start_ts",
        "end_ts",
        "created_at",
        "source",
    ],
    "research/strategy_evidence_samples.csv": [
        "strategy",
        "evidence_version",
        "as_of_date",
        "candidate_id",
        "run_id",
        "ts_utc",
        "symbol",
        "strategy_candidate",
        "candidate_name",
        "source_type",
        "sample_count",
        "complete_sample_count",
        "regime_state",
        "horizon_hours",
        "decision_ts",
        "label_ts",
        "entry_close",
        "label_close",
        "gross_bps",
        "net_bps_after_cost",
        "mfe_bps",
        "mae_bps",
        "win",
        "label_status",
        "label_reason",
        "cost_bps",
        "cost_source",
        "block_reason",
        "final_decision",
        "final_score",
        "expected_edge_bps",
        "required_edge_bps",
        "alpha6_score",
        "alpha6_side",
        "protect_level",
        "risk_level",
        "source_path_inside_bundle",
        "source_event_key",
        "source_bundle_ts",
        "created_at",
        "source",
    ],
    "research/strategy_evidence_quality.csv": [
        "strategy",
        "evidence_version",
        "as_of_date",
        "severity",
        "warning_type",
        "warning_count",
        "detail",
        "created_at",
        "source",
    ],
    "reports/alpha_discovery_board.csv": [
        "strategy",
        "board_schema_version",
        "as_of_date",
        "strategy_candidate",
        "candidate_name",
        "source_type",
        "symbol",
        "regime_state",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "median_net_bps",
        "p25_net_bps",
        "win_rate",
        "avg_mfe_bps",
        "avg_mae_bps",
        "cost_source_mix",
        "stability_by_day",
        "paper_days",
        "cost_source_has_global_default",
        "decision",
        "decision_reasons",
        "risk_permission",
        "risk_permission_status",
        "enforce_readiness_status",
        "block_reason_mix",
        "final_decision_mix",
        "high_score_blocked_outcome_count",
        "shadow_outcome_count",
        "start_ts",
        "end_ts",
        "created_at",
        "source",
    ],
    "reports/candidate_kill_list.csv": [
        "strategy_candidate",
        "candidate_name",
        "symbol",
        "regime_state",
        "horizon_hours",
        "as_of_date",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "median_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "decision",
        "decision_reasons",
    ],
    "reports/candidate_shadow_watchlist.csv": [
        "strategy_candidate",
        "candidate_name",
        "symbol",
        "regime_state",
        "horizon_hours",
        "as_of_date",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "median_net_bps",
        "p25_net_bps",
        "win_rate",
        "paper_days",
        "cost_source_mix",
        "decision",
        "decision_reasons",
    ],
    "reports/candidate_paper_ready.csv": [
        "strategy_candidate",
        "candidate_name",
        "symbol",
        "regime_state",
        "horizon_hours",
        "as_of_date",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "median_net_bps",
        "p25_net_bps",
        "win_rate",
        "paper_days",
        "cost_source_mix",
        "decision",
        "decision_reasons",
    ],
    "reports/paper_strategy_proposals.csv": [
        "proposal_id",
        "strategy_candidate",
        "symbol",
        "entry_conditions",
        "recommended_mode",
        "suggested_horizon",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "live_block_reason",
        "required_paper_days",
        "required_slippage_coverage",
        "as_of_date",
        "created_at",
    ],
    "reports/paper_strategy_runs.csv": [
        "as_of_date",
        "proposal_id",
        "strategy_candidate",
        "symbol",
        "recommended_mode",
        "board_decision",
        "suggested_horizon",
        "horizon_hours",
        "would_enter",
        "would_exit",
        "would_size",
        "would_size_usdt",
        "paper_pnl_bps",
        "paper_pnl_usdt",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "live_block_reason",
        "required_paper_days",
        "required_slippage_coverage",
        "created_at",
        "source",
        "schema_version",
    ],
    "reports/paper_strategy_daily.csv": [
        "as_of_date",
        "proposal_id",
        "strategy_candidate",
        "symbol",
        "recommended_mode",
        "paper_days",
        "latest_board_decision",
        "latest_horizon",
        "would_enter_count",
        "latest_paper_pnl_usdt",
        "cumulative_paper_pnl_usdt",
        "avg_paper_pnl_bps",
        "required_paper_days",
        "required_slippage_coverage",
        "live_eligible",
        "live_block_reason",
        "created_at",
        "source",
        "schema_version",
    ],
    "reports/paper_slippage_coverage.csv": [
        "as_of_date",
        "proposal_id",
        "strategy_candidate",
        "symbol",
        "paper_days",
        "observed_slippage_count",
        "required_observation_count",
        "paper_slippage_coverage",
        "required_slippage_coverage",
        "coverage_status",
        "created_at",
        "source",
        "schema_version",
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
        "unique_request_count",
        "unique_success_count",
        "unique_error_count",
        "unique_actual_fallback_count",
        "request_success_count",
        "request_error_count",
        "actual_fallback_count",
        "fallback_rate",
        "degraded_reason",
        "raw_imported_rows",
        "unique_event_rows",
        "duplicate_event_count",
        "duplicate_event_rows",
        "duplicate_rate",
        "first_seen_bundle_ts",
        "last_seen_bundle_ts",
    ],
    "v5/v5_decision_audit.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "source_path_inside_bundle",
        "run_id",
        "ingest_ts",
        "raw_payload_json",
    ],
    "v5/v5_execution_quality.csv": [
        "strategy",
        "date",
        "status",
        "fallback_count",
        "unique_request_count",
        "unique_success_count",
        "unique_error_count",
        "unique_actual_fallback_count",
        "request_success_count",
        "request_error_count",
        "actual_fallback_count",
        "fallback_rate",
        "degraded_reason",
        "raw_imported_rows",
        "unique_event_rows",
        "duplicate_event_count",
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
        "unique_request_count",
        "unique_success_count",
        "unique_error_count",
        "unique_actual_fallback_count",
        "request_success_count",
        "request_error_count",
        "actual_fallback_count",
        "fallback_rate",
        "degraded_reason",
        "raw_imported_rows",
        "unique_event_rows",
        "duplicate_event_count",
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
    "v5/v5_candidate_events.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "ingest_ts",
        "schema_version",
        "source_path_inside_bundle",
        "row_index",
        "event_type",
        "candidate_event_schema_version",
        "candidate_id",
        "run_id",
        "ts_utc",
        "symbol",
        "normalized_symbol",
        "candidate_quality_key",
        "regime_state",
        "risk_level",
        "current_position",
        "current_weight",
        "target_weight_raw",
        "target_weight_after_risk",
        "final_score",
        "rank",
        "f1_mom_5d",
        "f2_mom_20d",
        "f3_vol_adj_ret",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "alpha6_score",
        "alpha6_side",
        "ml_score",
        "mean_reversion_score",
        "expected_edge_bps",
        "required_edge_bps",
        "cost_bps",
        "selected_total_cost_bps",
        "cost_source",
        "cost_model_version",
        "cost_gate_verified",
        "would_block_by_cost",
        "eligible_before_filters",
        "final_decision",
        "block_reason",
        "strategy_candidate",
        "raw_payload_json",
    ],
    "v5/v5_candidate_labels.csv": [
        "strategy",
        "candidate_label_schema_version",
        "candidate_id",
        "run_id",
        "ts_utc",
        "symbol",
        "strategy_candidate",
        "block_reason",
        "final_decision",
        "horizon_hours",
        "decision_ts",
        "label_ts",
        "entry_close",
        "label_close",
        "gross_bps",
        "net_bps_after_cost",
        "mfe_bps",
        "mae_bps",
        "win",
        "label_status",
        "label_reason",
        "cost_bps",
        "cost_source",
        "alpha6_side",
        "regime_state",
        "risk_level",
        "protect_level",
        "final_score",
        "expected_edge_bps",
        "required_edge_bps",
        "source_event_bundle_sha256",
        "source_path_inside_bundle",
        "created_at",
        "source",
    ],
    "v5/v5_candidate_quality.csv": [
        "strategy",
        "date",
        "schema_version",
        "status",
        "candidate_event_rows",
        "run_count",
        "runs_with_candidate_event",
        "runs_without_candidate_event",
        "runs_without_candidate_event_json",
        "candidate_rows_by_run_json",
        "candidate_symbol_rows_by_run_json",
        "run_symbol_min_rows",
        "feature_completeness",
        "feature_completeness_by_field_json",
        "expected_label_rows",
        "label_rows",
        "complete_label_rows",
        "label_completeness",
        "cost_source_coverage",
        "cost_source_counts_json",
        "cost_source_quality_counts",
        "warnings_json",
        "created_at",
        "source",
    ],
    "v5/v5_candidate_outcome_summary.csv": [
        "strategy",
        "date",
        "schema_version",
        "block_reason",
        "strategy_candidate",
        "symbol",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_gross_bps",
        "avg_net_bps",
        "median_net_bps",
        "win_rate",
        "downside_p25_bps",
        "avg_mfe_bps",
        "avg_mae_bps",
        "label_status_counts_json",
        "created_at",
        "source",
    ],
    f"reports/{ENFORCE_READINESS_CSV}": [
        "as_of_ts",
        "strategy",
        "version",
        "readiness_status",
        "shadow_only_recommended",
        "blocked_reasons",
        "warning_reasons",
        "required_actions",
        "contract_version",
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
    refresh_risk_permission: bool = False,
    risk_strategy: str = "v5",
    risk_version: str = "5.0.0",
    pre_export_v5_refresh: bool = True,
    v5_telemetry_config: str | Path | None = None,
) -> DailyExportResult:
    day = _parse_date(export_date)
    root = Path(lake_root)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(UTC)
    pre_export_v5 = (
        _refresh_v5_before_export(root, day, config_path=v5_telemetry_config)
        if pre_export_v5_refresh
        else _observe_v5_before_export(root, config_path=v5_telemetry_config)
    )
    pre_export_v5_warnings = [str(warning) for warning in pre_export_v5.get("warnings", [])]
    pre_export_risk_warnings = (
        _refresh_risk_permission_before_export(root, strategy=risk_strategy, version=risk_version)
        if refresh_risk_permission
        else []
    )
    snapshot = _load_snapshot(root)
    missing_sections = _missing_sections(snapshot.row_counts)
    data_quality = _data_quality_payload(
        root,
        snapshot,
        day,
        generated_at,
        pre_export_risk_refresh_attempted=refresh_risk_permission,
        pre_export_v5=pre_export_v5,
        pre_export_warnings=pre_export_risk_warnings,
    )
    warnings = sorted(
        set([*snapshot.warnings, *pre_export_v5_warnings, *data_quality["warnings"]])
    )

    members: dict[str, _MemberPayload] = {}
    members.update(_dataset_members(snapshot.frames))
    members.update(_chart_members(snapshot.frames))
    members.update(enforce_readiness_members(root))
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
        pre_export_v5=pre_export_v5,
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

    zip_path = _unique_export_zip_path(output_root, day, generated_at)
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


def _unique_export_zip_path(output_root: Path, day: date, generated_at: datetime) -> Path:
    timestamp = generated_at.astimezone(BEIJING_TZ).strftime("%Y%m%dT%H%M%S%f+0800")
    base_name = f"quant_lab_expert_pack_{day.isoformat()}_{timestamp}"
    candidate = output_root / f"{base_name}.zip"
    counter = 1
    while candidate.exists():
        candidate = output_root / f"{base_name}_{counter}.zip"
        counter += 1
    return candidate


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
            warnings.append(f"{name} dataset is {_missing_dataset_reason(name)}")
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
        recent_files = _recent_heavy_dataset_files(
            dataset_path,
            max_files=HEAVY_EXPORT_RECENT_FILE_LIMITS.get(dataset_name, 50),
        )
        frame = _collect_recent_heavy_files(
            recent_files,
            dataset_name,
            limit=HEAVY_EXPORT_DATASET_LIMITS[dataset_name],
        )
        return frame, frame.height, None
    except Exception as exc:
        return pl.DataFrame(), 0, f"{dataset_name} sampled read failed: {exc}"


def _recent_heavy_dataset_files(dataset_path: Path, *, max_files: int) -> list[Path]:
    if not dataset_path.exists():
        return []
    if dataset_path.is_file():
        return [dataset_path]
    files = [path for path in dataset_path.rglob("*.parquet") if path.is_file()]
    if not files:
        return []
    files.sort(key=lambda path: path.stat().st_mtime)
    return files[-max_files:]


def _collect_recent_heavy_files(
    files: list[Path],
    dataset_name: str,
    *,
    limit: int,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    rows = 0
    for path in reversed(files):
        try:
            remaining = max(limit - rows, 1)
            frame = pl.scan_parquet(str(path)).tail(remaining).collect()
        except Exception:
            continue
        if frame.is_empty():
            continue
        frames.append(frame)
        rows += frame.height
        if rows >= limit:
            break
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(list(reversed(frames)), how="diagonal_relaxed")
    timestamp_column = next(
        (
            column
            for column in readers.DATASET_TIMESTAMP_COLUMNS.get(dataset_name, ())
            if column in combined.columns
        ),
        None,
    )
    if timestamp_column is None:
        return combined.tail(limit)
    return _tail_by_time(combined, timestamp_column, limit=limit)


def _snapshot_market_health(snapshot: _DatasetSnapshot) -> dict[str, Any]:
    market = snapshot.frames.get("market_bar", pl.DataFrame())
    warnings: list[str] = []
    if market.is_empty():
        return {
            "duplicate_bar_count": 0,
            "unclosed_bar_count": 0,
            "schema_violations": ["market_bar dataset missing or empty"],
            "warnings": ["market_bar dataset missing or empty"],
        }

    market = readers._normalize_market_frame(market)  # type: ignore[attr-defined]
    schema_violations = readers.market_bar_schema_violations(market)
    duplicate_count = readers.duplicate_market_bar_count(market)
    unclosed_count = readers.unclosed_market_bar_count(market)
    if duplicate_count:
        warnings.append(f"market_bar duplicate primary keys: {duplicate_count}")
    if unclosed_count:
        warnings.append(f"unclosed market_bar rows: {unclosed_count}")
    warnings.extend(schema_violations)
    return {
        "duplicate_bar_count": duplicate_count,
        "unclosed_bar_count": unclosed_count,
        "schema_violations": schema_violations,
        "warnings": warnings,
    }


def _dataset_members(frames: dict[str, pl.DataFrame]) -> dict[str, _MemberPayload]:
    market = frames.get("market_bar", pl.DataFrame())
    features = frames.get("feature_value", pl.DataFrame())
    feature_coverage = frames.get("feature_coverage_daily", pl.DataFrame())
    feature_anomalies = frames.get("feature_anomaly_daily", pl.DataFrame())
    costs = _normalize_symbol_frame(frames.get("cost_bucket_daily", pl.DataFrame()))
    cost_health = frames.get("cost_health_daily", pl.DataFrame())
    evidence = _alpha_evidence_for_export(frames.get("alpha_evidence", pl.DataFrame()))
    alpha_discovery_board = _alpha_discovery_board_for_export(
        frames.get("alpha_discovery_board", pl.DataFrame())
    )
    strategy_evidence = _strategy_evidence_for_export(
        frames.get("strategy_evidence", pl.DataFrame())
    )
    strategy_samples = _strategy_evidence_samples_for_export(
        frames.get("strategy_evidence_sample", pl.DataFrame())
    )
    paper_runs = frames.get("paper_strategy_runs", pl.DataFrame())
    paper_daily = frames.get("paper_strategy_daily", pl.DataFrame())
    paper_slippage = frames.get("paper_slippage_coverage", pl.DataFrame())
    gates = frames.get("gate_decision", pl.DataFrame())
    risk = _risk_permissions_for_export(frames.get("risk_permission", pl.DataFrame()), frames)
    trades = _normalize_symbol_frame(frames.get("trade_print", pl.DataFrame()))
    books = _normalize_symbol_frame(frames.get("orderbook_snapshot", pl.DataFrame()))
    v5_health = frames.get("strategy_health_daily", pl.DataFrame())
    v5_decisions = frames.get("v5_decision_audit", pl.DataFrame())
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
    v5_fallbacks = _dedupe_v5_event_frame(frames.get("v5_quant_lab_fallback", pl.DataFrame()))
    v5_candidate_events = frames.get("v5_candidate_event", pl.DataFrame())
    v5_candidate_labels = frames.get("v5_candidate_label", pl.DataFrame())
    v5_candidate_quality = frames.get("v5_candidate_quality_daily", pl.DataFrame())
    v5_candidate_outcomes = frames.get("v5_candidate_outcome_summary", pl.DataFrame())

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
        "research/strategy_evidence.csv": _csv_member(
            "research/strategy_evidence.csv",
            strategy_evidence,
        ),
        "research/strategy_evidence_samples.csv": _csv_member(
            "research/strategy_evidence_samples.csv",
            strategy_samples,
        ),
        "research/strategy_evidence_quality.csv": _csv_member(
            "research/strategy_evidence_quality.csv",
            frames.get("strategy_evidence_quality", pl.DataFrame()),
        ),
        "research/gate_decisions.csv": _csv_member("research/gate_decisions.csv", gates),
        "research/research_conclusions.csv": _csv_text(
            pl.DataFrame(schema={"alpha_id": pl.Utf8, "conclusion": pl.Utf8, "source": pl.Utf8})
        ),
        "reports/alpha_discovery_board.csv": _csv_member(
            "reports/alpha_discovery_board.csv",
            alpha_discovery_board,
        ),
        "reports/strategy_evidence_summary.md": _strategy_evidence_summary_md(
            alpha_discovery_board,
            strategy_evidence,
        ),
        "reports/candidate_kill_list.csv": _csv_member(
            "reports/candidate_kill_list.csv",
            _candidate_decision_rows(alpha_discovery_board, "KILL"),
        ),
        "reports/candidate_shadow_watchlist.csv": _csv_member(
            "reports/candidate_shadow_watchlist.csv",
            _candidate_decision_rows(alpha_discovery_board, "KEEP_SHADOW"),
        ),
        "reports/candidate_paper_ready.csv": _csv_member(
            "reports/candidate_paper_ready.csv",
            _candidate_decision_rows(alpha_discovery_board, "PAPER_READY"),
        ),
        "reports/paper_strategy_proposals.csv": _csv_member(
            "reports/paper_strategy_proposals.csv",
            _paper_strategy_proposals_for_export(alpha_discovery_board),
        ),
        "reports/paper_strategy_runs.csv": _csv_member(
            "reports/paper_strategy_runs.csv",
            paper_runs,
        ),
        "reports/paper_strategy_daily.csv": _csv_member(
            "reports/paper_strategy_daily.csv",
            paper_daily,
        ),
        "reports/paper_slippage_coverage.csv": _csv_member(
            "reports/paper_slippage_coverage.csv",
            paper_slippage,
        ),
        "risk/risk_permission.json": _json_text({"rows": _rows(risk)}),
        "risk/risk_flags.csv": _csv_text(_risk_flags(risk)),
        "anomalies/anomalies.csv": _csv_text(_anomaly_rows(frames)),
        "anomalies/missing_data.csv": _csv_text(_missing_dataset_rows(frames)),
        "anomalies/stale_data.csv": _csv_text(_stale_rows(frames)),
        "anomalies/schema_violations.csv": _csv_text(_schema_violation_rows(market)),
        "v5/v5_strategy_health.csv": _csv_member("v5/v5_strategy_health.csv", v5_health),
        "v5/v5_decision_audit.csv": _csv_member(
            "v5/v5_decision_audit.csv",
            v5_decisions,
        ),
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
        "v5/v5_candidate_events.csv": _csv_member(
            "v5/v5_candidate_events.csv",
            _tail_by_time(v5_candidate_events, "ts_utc", limit=50_000),
        ),
        "v5/v5_candidate_labels.csv": _csv_member(
            "v5/v5_candidate_labels.csv",
            _tail_by_time(v5_candidate_labels, "ts_utc", limit=100_000),
        ),
        "v5/v5_candidate_quality.csv": _csv_member(
            "v5/v5_candidate_quality.csv",
            v5_candidate_quality,
        ),
        "v5/v5_candidate_outcome_summary.csv": _csv_member(
            "v5/v5_candidate_outcome_summary.csv",
            v5_candidate_outcomes,
        ),
    }


def _dedupe_v5_event_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    rows_by_key: dict[str, dict[str, Any]] = {}
    for row in frame.to_dicts():
        key = _v5_telemetry_event_key(row)
        current = rows_by_key.get(key)
        if current is None or _v5_event_seen_time(row) >= _v5_event_seen_time(current):
            rows_by_key[key] = row
    return pl.DataFrame(list(rows_by_key.values()))


def _v5_event_seen_time(row: dict[str, Any]) -> datetime:
    for field in ["last_seen_bundle_ts", "bundle_ts", "ingest_ts"]:
        value = readers._coerce_timestamp(row.get(field))  # type: ignore[attr-defined]
        if value is not None:
            return value
    return datetime.min.replace(tzinfo=UTC)


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
    pre_export_v5: dict[str, Any] | None = None,
) -> dict[str, Any]:
    git = _git_info()
    risk_export_status = _risk_permission_export_status(snapshot.frames, generated_at)
    candidate_events = snapshot.frames.get("v5_candidate_event", pl.DataFrame())
    latest_ingested_bundle_ts = _latest_v5_bundle_ts(snapshot.frames)
    candidate_event_latest_ts = _latest_dataset_timestamp("v5_candidate_event", candidate_events)
    v5_bundle_lag_seconds = (
        max(0, int((generated_at - latest_ingested_bundle_ts).total_seconds()))
        if latest_ingested_bundle_ts is not None
        else None
    )
    v5_context = pre_export_v5 or {}
    return {
        "export_date": day.isoformat(),
        "generated_at": generated_at.isoformat(),
        "export_generated_at": generated_at.isoformat(),
        "display_timezone": DISPLAY_TIMEZONE,
        "generated_at_beijing": beijing_iso(generated_at),
        "export_generated_at_beijing": beijing_iso(generated_at),
        "risk_permission_generated_at": risk_export_status.get("risk_permission_generated_at"),
        "risk_permission_expires_at": risk_export_status.get("risk_permission_expires_at"),
        "permission_expired_at_export": risk_export_status.get(
            "permission_expired_at_export"
        ),
        "profile": profile,
        "quant_lab_version": __version__,
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
        "schema_version": V5_TELEMETRY_DATASET_SCHEMA_VERSION,
        "latest_v5_bundle_seen_at_export": v5_context.get(
            "latest_v5_bundle_seen_at_export"
        ),
        "latest_v5_bundle_ingested_at_export": _iso_or_none(latest_ingested_bundle_ts),
        "candidate_event_latest_ts": _iso_or_none(candidate_event_latest_ts),
        "latest_candidate_event_ts": _iso_or_none(candidate_event_latest_ts),
        "candidate_event_rows": candidate_events.height,
        "v5_bundle_lag_seconds": v5_bundle_lag_seconds,
        "pre_export_v5_refresh": v5_context,
        "git_commit": git["git_commit"],
        "git_branch": git["git_branch"],
        "dirty_worktree": git["dirty_worktree"],
        "git_dirty": git["dirty_worktree"],
        "provenance_status": git["provenance_status"],
        "code_provenance": git["code_provenance"],
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
        "risk_permission_export_status": risk_export_status,
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


def _refresh_risk_permission_before_export(
    lake_root: Path,
    *,
    strategy: str,
    version: str,
) -> list[str]:
    try:
        result = publish_risk_permission(lake_root=lake_root, strategy=strategy, version=version)
    except Exception as exc:  # pragma: no cover - exact errors depend on production lake state.
        return [
            "risk_permission_republish_failed: "
            f"{type(exc).__name__}: {_safe_warning_text(str(exc))}"
        ]
    return [
        f"risk_permission_republish_warning: {_safe_warning_text(str(warning))}"
        for warning in result.warnings
    ]


def _refresh_v5_before_export(
    lake_root: Path,
    export_day: date,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "enabled": True,
        "processed_bundle_count": 0,
        "skipped_bundle_count": 0,
        "latest_v5_bundle_seen_at_export": None,
        "local_inbox_dir": None,
        "warnings": [],
    }
    warnings: list[str] = []
    try:
        cfg = _load_export_v5_config(lake_root, config_path=config_path)
    except Exception as exc:  # pragma: no cover - production config errors vary.
        context["warnings"] = [
            "v5_pre_export_config_failed: "
            f"{type(exc).__name__}: {_safe_warning_text(str(exc))}"
        ]
        return context

    inbox_dir = Path(cfg["local_inbox_dir"])
    context["local_inbox_dir"] = str(inbox_dir)
    latest_seen = _latest_v5_bundle_ts_in_inbox(inbox_dir)
    context["latest_v5_bundle_seen_at_export"] = _iso_or_none(latest_seen)
    if not inbox_dir.exists():
        context["warnings"] = []
        return context

    try:
        from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle

        pending_paths = _pending_v5_bundle_paths_for_export(inbox_dir, lake_root)
        context["selected_bundle_count"] = len(pending_paths)
        context["selected_bundle_names"] = [path.name for path in pending_paths]
        processed_count = 0
        skipped_count = 0
        for bundle_path in pending_paths:
            result = ingest_v5_bundle(
                bundle_path=bundle_path,
                lake_root=lake_root,
                restricted_archive_dir=Path(cfg["restricted_archive_dir"]),
                redacted_archive_dir=Path(cfg["redacted_archive_dir"]),
                strategy=str(cfg["strategy"]),
                limits=cfg.get("limits"),
                run_analysis=False,
                refresh_candidate_gold=False,
            )
            if result.skipped:
                skipped_count += 1
            else:
                processed_count += 1
            warnings.extend(str(warning) for warning in result.warnings)
        context["processed_bundle_count"] = processed_count
        context["skipped_bundle_count"] = skipped_count
    except Exception as exc:  # pragma: no cover - exact production tar/parquet errors vary.
        warnings.append(
            "v5_pre_export_inbox_ingest_failed: "
            f"{type(exc).__name__}: {_safe_warning_text(str(exc))}"
        )

    if int(context.get("processed_bundle_count") or 0) > 0:
        warnings.extend(_refresh_v5_derived_outputs(lake_root, export_day))
    context["warnings"] = sorted(set(warnings))
    return context


def _load_export_v5_config(
    lake_root: Path,
    *,
    config_path: str | Path | None,
) -> dict[str, Any]:
    resolved = config_path or os.environ.get("QUANT_LAB_V5_TELEMETRY_CONFIG")
    if resolved is None:
        default_path = Path("/etc/quant-lab/v5_telemetry_remote.yaml")
        resolved = default_path if default_path.exists() else None
    if resolved is not None:
        from quant_lab.strategy_telemetry.config import load_v5_telemetry_remote_config

        cfg = load_v5_telemetry_remote_config(resolved)
        return {
            "strategy": cfg.strategy,
            "local_inbox_dir": cfg.local_inbox_dir,
            "restricted_archive_dir": cfg.restricted_archive_dir,
            "redacted_archive_dir": cfg.redacted_archive_dir,
            "limits": cfg.bundle_limits,
        }
    base = lake_root.parent if lake_root.name == "lake" else lake_root
    return {
        "strategy": "v5",
        "local_inbox_dir": base / "inbox" / "v5" / "bundles",
        "restricted_archive_dir": base / "archive_restricted" / "v5" / "bundles",
        "redacted_archive_dir": base / "archive" / "v5" / "bundles",
        "limits": None,
    }


def _observe_v5_before_export(
    lake_root: Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "enabled": False,
        "processed_bundle_count": 0,
        "skipped_bundle_count": 0,
        "selected_bundle_count": 0,
        "selected_bundle_names": [],
        "latest_v5_bundle_seen_at_export": None,
        "local_inbox_dir": None,
        "warnings": [],
    }
    try:
        cfg = _load_export_v5_config(lake_root, config_path=config_path)
    except Exception as exc:  # pragma: no cover - production config errors vary.
        context["warnings"] = [
            "v5_pre_export_observation_failed: "
            f"{type(exc).__name__}: {_safe_warning_text(str(exc))}"
        ]
        return context
    inbox_dir = Path(cfg["local_inbox_dir"])
    context["local_inbox_dir"] = str(inbox_dir)
    context["latest_v5_bundle_seen_at_export"] = _iso_or_none(
        _latest_v5_bundle_ts_in_inbox(inbox_dir)
    )
    return context


def _latest_v5_bundle_ts_in_inbox(inbox_dir: Path) -> datetime | None:
    if not inbox_dir.exists():
        return None
    from quant_lab.strategy_telemetry.bundle import parse_bundle_ts

    timestamps: list[datetime] = []
    for path in inbox_dir.glob("v5_live_followup_bundle_*.tar.gz"):
        parsed = parse_bundle_ts(path.name)
        if parsed is not None:
            timestamps.append(parsed)
            continue
        try:
            timestamps.append(datetime.fromtimestamp(path.stat().st_mtime, tz=UTC))
        except OSError:
            continue
    return max(timestamps) if timestamps else None


def _pending_v5_bundle_paths_for_export(inbox_dir: Path, lake_root: Path) -> list[Path]:
    max_pending = _export_v5_max_pending_bundles()
    max_scan = _export_v5_max_scan_bundles(max_pending)
    ingested = _ingested_bundle_sha256s(lake_root)
    from quant_lab.strategy_telemetry.bundle import compute_sha256, parse_bundle_ts

    def sort_key(path: Path) -> tuple[datetime, str]:
        parsed = parse_bundle_ts(path.name)
        if parsed is None:
            try:
                parsed = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError:
                parsed = datetime.min.replace(tzinfo=UTC)
        return parsed, path.name

    selected: list[Path] = []
    for scanned_count, path in enumerate(
        sorted(
            inbox_dir.glob("v5_live_followup_bundle_*.tar.gz"),
            key=sort_key,
            reverse=True,
        ),
        start=1,
    ):
        if scanned_count > max_scan:
            break
        try:
            sha256 = compute_sha256(path)
        except OSError:
            continue
        if sha256 in ingested:
            continue
        selected.append(path)
        if len(selected) >= max_pending:
            break
    return list(reversed(selected))


def _export_v5_max_pending_bundles() -> int:
    raw_value = os.environ.get("QUANT_LAB_EXPORT_V5_MAX_PENDING_BUNDLES", "5")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 5


def _export_v5_max_scan_bundles(max_pending: int) -> int:
    raw_value = os.environ.get("QUANT_LAB_EXPORT_V5_MAX_SCAN_BUNDLES", "20")
    try:
        return max(max_pending, int(raw_value))
    except ValueError:
        return max(max_pending, 20)


def _ingested_bundle_sha256s(lake_root: Path) -> set[str]:
    path = lake_root / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest"
    try:
        frame = read_parquet_dataset(path)
    except Exception:
        return set()
    if "bundle_sha256" not in frame.columns or frame.is_empty():
        return set()
    return {
        str(value)
        for value in frame.get_column("bundle_sha256").drop_nulls().unique().to_list()
        if value
    }


def _refresh_v5_derived_outputs(lake_root: Path, export_day: date) -> list[str]:
    warnings: list[str] = []
    steps = [
        (
            "analyze_v5_telemetry",
            lambda: __import__(
                "quant_lab.strategy_telemetry.analyze",
                fromlist=["analyze_v5_telemetry"],
            ).analyze_v5_telemetry(lake_root=lake_root, date=export_day.isoformat()),
        ),
        (
            "build_candidate_labels",
            lambda: __import__(
                "quant_lab.research.candidate_labels",
                fromlist=["build_and_publish_candidate_labels"],
            ).build_and_publish_candidate_labels(lake_root, as_of_date=export_day),
        ),
        (
            "build_strategy_evidence",
            lambda: __import__(
                "quant_lab.research.strategy_evidence",
                fromlist=["build_and_publish_strategy_evidence"],
            ).build_and_publish_strategy_evidence(
                lake_root, as_of_date=export_day.isoformat()
            ),
        ),
        (
            "build_alpha_discovery_board",
            lambda: __import__(
                "quant_lab.research.alpha_discovery",
                fromlist=["build_and_publish_alpha_discovery_board"],
            ).build_and_publish_alpha_discovery_board(lake_root, as_of_date=export_day),
        ),
        (
            "build_paper_strategy_tracking",
            lambda: __import__(
                "quant_lab.research.paper_tracking",
                fromlist=["build_and_publish_paper_strategy_tracking"],
            ).build_and_publish_paper_strategy_tracking(lake_root, as_of_date=export_day),
        ),
        (
            "write_enforce_readiness_report",
            lambda: __import__(
                "quant_lab.reports.enforce_readiness",
                fromlist=["write_enforce_readiness_report"],
            ).write_enforce_readiness_report(
                lake_root=lake_root,
                out_dir=readers.default_exports_root(lake_root),
                strategy="v5",
                version="5.0.0",
            ),
        ),
    ]
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            warnings.append(
                f"pre_export_v5_refresh_failed:{name}:"
                f"{type(exc).__name__}:{_safe_warning_text(str(exc))}"
            )
    return warnings


def _safe_warning_text(value: str, *, limit: int = 500) -> str:
    text = value.replace("\n", " ").replace("\r", " ").strip()
    for pattern, _severity, _label in SECRET_PATTERNS:
        text = pattern.sub("<REDACTED>", text)
    return text[:limit]


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
        "display_timezone": DISPLAY_TIMEZONE,
        "generated_at_beijing": beijing_iso(generated_at),
        "quant_lab_version": __version__,
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
        "schema_version": V5_TELEMETRY_DATASET_SCHEMA_VERSION,
        "git_commit": git["git_commit"],
        "git_branch": git["git_branch"],
        "dirty_worktree": git["dirty_worktree"],
        "git_dirty": git["dirty_worktree"],
        "provenance_status": git["provenance_status"],
        "code_provenance": git["code_provenance"],
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
    generated_at: datetime,
    *,
    pre_export_risk_refresh_attempted: bool = False,
    pre_export_v5: dict[str, Any] | None = None,
    pre_export_warnings: list[str] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []
    refresh_warnings = pre_export_warnings or []
    v5_refresh = pre_export_v5 or {}

    health = _snapshot_market_health(snapshot)
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
    decision_audit_quality = _decision_audit_quality(snapshot.frames)
    latest_v5_bundle_ts = _latest_v5_bundle_ts(snapshot.frames)
    v5_lag_seconds = (
        max(0, int((generated_at - latest_v5_bundle_ts).total_seconds()))
        if latest_v5_bundle_ts is not None
        else None
    )
    git = _git_info()
    checks.append(
        _check(
            "code_provenance_clean",
            not git["dirty_worktree"],
            (
                f"git_commit={git['git_commit']}; "
                f"git_branch={git['git_branch']}; "
                f"provenance_status={git['provenance_status']}"
            ),
            severity="warning",
        )
    )

    costs = snapshot.frames.get("cost_bucket_daily", pl.DataFrame())
    cost_health = snapshot.frames.get("cost_health_daily", pl.DataFrame())
    latest_cost_health = (
        cost_health.sort("day").tail(1).to_dicts()[0]
        if not cost_health.is_empty() and "day" in cost_health.columns
        else {}
    )
    fallback_stats = _cost_fallback_stats(costs, latest_cost_health)
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
        checks.append(
            _check(
                "cost_fallback_ratio",
                fallback_stats["fallback_ratio"] <= 0.25,
                (
                    f"fallback_rows={fallback_rows.height}; "
                    f"ratio={fallback_stats['fallback_ratio']:.2%}; "
                    "legacy metric includes soft/proxy fallbacks"
                ),
                warning_only=True,
            )
        )
        checks.append(
            _check(
                "cost_hard_fallback_ratio",
                fallback_stats["hard_fallback_ratio"] <= 0.25,
                (
                    f"hard_fallback_count={fallback_stats['hard_fallback_count']}; "
                    f"ratio={fallback_stats['hard_fallback_ratio']:.2%}; "
                    f"global_default_count={fallback_stats['global_default_count']}"
                ),
                severity="critical",
            )
        )
        checks.append(
            _check(
                "cost_soft_fallback_ratio",
                fallback_stats["soft_fallback_ratio"] <= 0.5,
                (
                    f"soft_fallback_count={fallback_stats['soft_fallback_count']}; "
                    f"ratio={fallback_stats['soft_fallback_ratio']:.2%}; "
                    f"proxy_only_count={fallback_stats['proxy_only_count']}"
                ),
                warning_only=True,
            )
        )

    gates = snapshot.frames.get("gate_decision", pl.DataFrame())
    evidence = snapshot.frames.get("alpha_evidence", pl.DataFrame())
    alpha_discovery_board = snapshot.frames.get("alpha_discovery_board", pl.DataFrame())
    strategy_evidence = snapshot.frames.get("strategy_evidence", pl.DataFrame())
    candidate_events = snapshot.frames.get("v5_candidate_event", pl.DataFrame())
    candidate_labels = snapshot.frames.get("v5_candidate_label", pl.DataFrame())
    candidate_quality = snapshot.frames.get("v5_candidate_quality_daily", pl.DataFrame())
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
            "strategy_evidence_present",
            strategy_evidence.height > 0 or alpha_discovery_board.height > 0,
            (
                f"rows={strategy_evidence.height}; "
                f"alpha_discovery_board_rows={alpha_discovery_board.height}; "
                "legacy_optional_when_alpha_discovery_board_present"
            ),
            severity=(
                "critical"
                if research_enabled and alpha_discovery_board.height == 0
                else "warning"
            ),
        )
    )
    checks.append(
        _check(
            "alpha_discovery_board_present",
            alpha_discovery_board.height > 0,
            f"rows={alpha_discovery_board.height}",
            severity="critical" if research_enabled else "warning",
        )
    )
    checks.append(
        _check(
            "alpha_discovery_board_live_rules",
            _alpha_discovery_live_rules_ok(alpha_discovery_board),
            "alt_impulse/sol protect cannot be LIVE_SMALL_READY; live rows need samples >= 60",
            severity="critical",
        )
    )
    checks.append(
        _check(
            "strategy_evidence_candidate_coverage",
            _strategy_evidence_has_required_candidates(strategy_evidence),
            "strategy_evidence should include all configured V5 strategy candidates",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "strategy_evidence_live_sample_floor",
            _strategy_evidence_live_sample_floor_ok(strategy_evidence),
            "LIVE_SMALL_READY requires sample_count >= 30",
            severity="critical",
        )
    )
    latest_candidate_quality = _latest_candidate_quality(candidate_quality)
    checks.append(
        _check(
            "v5_candidate_event_present",
            candidate_events.height > 0,
            f"rows={candidate_events.height}",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "v5_candidate_rows_by_run",
            int(latest_candidate_quality.get("runs_without_candidate_event") or 0) == 0
            and int(latest_candidate_quality.get("run_symbol_min_rows") or 0) >= 1,
            (
                "runs_without_candidate_event="
                f"{latest_candidate_quality.get('runs_without_candidate_event', 'n/a')}; "
                f"run_symbol_min_rows={latest_candidate_quality.get('run_symbol_min_rows', 'n/a')}"
            ),
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "v5_candidate_feature_completeness",
            float(latest_candidate_quality.get("feature_completeness") or 0.0) >= 0.8,
            f"feature_completeness={latest_candidate_quality.get('feature_completeness', 'n/a')}",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "v5_candidate_label_completeness",
            candidate_labels.height > 0
            and float(latest_candidate_quality.get("label_completeness") or 0.0) >= 0.8,
            f"label_completeness={latest_candidate_quality.get('label_completeness', 'n/a')}",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "v5_candidate_cost_source_coverage",
            float(latest_candidate_quality.get("cost_source_coverage") or 0.0) >= 0.8,
            (
                "cost_source_coverage="
                f"{latest_candidate_quality.get('cost_source_coverage', 'n/a')}"
            ),
            warning_only=True,
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
    checks.append(
        _check(
            "generic_decision_audit_present",
            bool(decision_audit_quality["generic_decision_audit_present"]),
            (
                "legacy generic silver/decision_audit is optional for V5; "
                f"rows={decision_audit_quality['generic_decision_audit_rows']}"
            ),
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "v5_decision_audit_present",
            bool(decision_audit_quality["v5_decision_audit_present"]),
            f"v5_decision_audit_count={decision_audit_quality['v5_decision_audit_count']}",
            severity="critical" if v5_integration_enabled else "warning",
        )
    )
    checks.append(
        _check(
            "v5_decision_audit_count",
            True,
            f"v5_decision_audit_count={decision_audit_quality['v5_decision_audit_count']}",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "v5_bundle_fresh_at_export",
            latest_v5_bundle_ts is not None
            and v5_lag_seconds is not None
            and v5_lag_seconds <= 3 * 60 * 60,
            (
                f"latest_v5_bundle_ts={_iso_or_none(latest_v5_bundle_ts)}; "
                f"lag_seconds={v5_lag_seconds}; "
                f"latest_seen_inbox={v5_refresh.get('latest_v5_bundle_seen_at_export')}"
            ),
            severity="critical" if v5_integration_enabled else "warning",
        )
    )

    risk = _risk_permissions_for_export(
        snapshot.frames.get("risk_permission", pl.DataFrame()),
        snapshot.frames,
        reference_at=generated_at,
    )
    checks.append(
        _check(
            "risk_permission_present",
            risk.height > 0,
            f"rows={risk.height}",
            severity="critical" if v5_integration_enabled else "warning",
        )
    )
    if pre_export_risk_refresh_attempted:
        checks.append(
            _check(
                "risk_permission_pre_export_refresh",
                not refresh_warnings,
                "; ".join(refresh_warnings) if refresh_warnings else "publish-risk-permission ok",
                warning_only=True,
            )
        )
        warnings.extend(refresh_warnings)
    checks.append(
        _check(
            "risk_permission_versions",
            _risk_has_versions(risk),
            "risk_permission requires gate_version and cost_model_version",
            warning_only=True,
        )
    )
    risk_quality = _risk_permission_quality(risk, snapshot.frames, reference_at=generated_at)
    risk_is_stale_vs_v5 = str(risk_quality["permission_status"]).startswith("STALE_")
    risk_expired_at_export = bool(risk_quality.get("permission_expired_at_export"))
    checks.append(
        _check(
            "risk_permission_fresh_vs_v5_telemetry",
            not risk_is_stale_vs_v5,
            _risk_permission_quality_detail(risk_quality),
            severity="critical",
        )
    )
    checks.append(
        _check(
            "risk_permission_not_expired_at_export",
            not risk_expired_at_export,
            _risk_permission_expired_detail(risk_quality),
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
    v5_trades = snapshot.frames.get("v5_trade_event", pl.DataFrame())
    proxy_only = _all_cost_rows_public_proxy(costs)
    health_checks = _jsonish_dict(latest_cost_health.get("data_quality_checks_json"))
    actual_rows = int(latest_cost_health.get("actual_rows") or 0)
    checks.append(
        _check(
            "okx_private_actual_cost_available",
            not (proxy_only and actual_rows == 0),
            "actual cost unavailable; cost model is public_spread_proxy only",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "private_fills_present_but_actual_cost_zero",
            health_checks.get("private_fills_present_but_actual_cost_zero", True) is not False,
            f"private_fills={fills.height}; actual_rows={actual_rows}",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "trades_present_but_not_in_cost_model",
            health_checks.get("trades_present_but_not_in_cost_model", True) is not False,
            f"v5_trades={v5_trades.height}; actual_rows={actual_rows}",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "bills_present_but_fee_bps_missing",
            health_checks.get("bills_present_but_fee_bps_missing", True) is not False,
            f"private_bills={bills.height}",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "fee_missing_rate",
            True,
            str(health_checks.get("fee_missing_rate", "n/a")),
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "actual_cost_symbol_coverage",
            actual_rows > 0 or (fills.height == 0 and v5_trades.height == 0),
            str(health_checks.get("actual_cost_symbol_coverage", "n/a")),
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
    enforce_readiness = build_enforce_readiness_report(lake_root)
    checks.append(
        _check(
            "quant_lab_enforce_readiness",
            enforce_readiness.readiness_status == "READY",
            (
                f"readiness_status={enforce_readiness.readiness_status}; "
                f"blocked={enforce_readiness.blocked_reasons}; "
                f"warnings={enforce_readiness.warning_reasons}"
            ),
            severity="critical" if enforce_readiness.readiness_status == "BLOCKED" else "warning",
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
        "generated_at": generated_at.isoformat(),
        "display_timezone": DISPLAY_TIMEZONE,
        "generated_at_beijing": beijing_iso(generated_at),
        "checks": checks,
        "warnings": sorted(set(warnings)),
        "risk_permission": risk_quality,
        "decision_audit": decision_audit_quality,
        "v5_pre_export": v5_refresh,
        "quant_lab_enforce_readiness": enforce_readiness.model_dump(mode="json"),
        "shadow_only_recommended": enforce_readiness.shadow_only_recommended,
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
    alpha_discovery_board = snapshot.frames.get("alpha_discovery_board", pl.DataFrame())
    strategy_evidence = snapshot.frames.get("strategy_evidence", pl.DataFrame())
    candidate_quality = _latest_candidate_quality(
        snapshot.frames.get("v5_candidate_quality_daily", pl.DataFrame())
    )
    risk = snapshot.frames.get("risk_permission", pl.DataFrame())
    gate_counts = _value_counts(gates, "status")
    alpha_discovery_counts = _value_counts(alpha_discovery_board, "decision")
    strategy_decision_counts = _value_counts(strategy_evidence, "decision")
    risk_quality = data_quality.get("risk_permission", {})
    quality_status = str(risk_quality.get("permission_status") or "").strip()
    permission_counts = (
        {quality_status: 1}
        if quality_status
        else _value_counts(risk, "permission_status") or _value_counts(risk, "permission")
    )
    fallback_rows = _cost_fallbacks(snapshot.frames.get("cost_bucket_daily", pl.DataFrame()))
    questions = _question_lines(snapshot, data_quality)
    readiness = data_quality.get("quant_lab_enforce_readiness", {})

    lines = [
        f"# Executive Summary - {day.isoformat()}",
        "",
        f"Data quality status: {data_quality.get('status', 'UNKNOWN')}",
        f"Dataset row counts: {safe_json_dumps(snapshot.row_counts)}",
        f"Cost fallback rows: {fallback_rows.height}",
        f"Gate status counts: {safe_json_dumps(gate_counts)}",
        f"Alpha discovery decision counts: {safe_json_dumps(alpha_discovery_counts)}",
        f"Strategy evidence decision counts: {safe_json_dumps(strategy_decision_counts)}",
        f"V5 candidate_event rows: {snapshot.row_counts.get('v5_candidate_event', 0)}",
        "V5 candidate label completeness: "
        f"{candidate_quality.get('label_completeness', 'n/a')}",
        "V5 candidate cost source coverage: "
        f"{candidate_quality.get('cost_source_coverage', 'n/a')}",
        f"Risk permission counts: {safe_json_dumps(permission_counts)}",
        f"Risk permission status: {risk_quality.get('permission_status', 'UNKNOWN')}",
        f"Risk permission as_of_ts: {risk_quality.get('as_of_ts')}",
        f"Latest V5 telemetry ts: {risk_quality.get('telemetry_latest_ts')}",
        f"Risk permission next action: {risk_quality.get('next_action', 'none')}",
        f"quant_lab_enforce_readiness: {readiness.get('readiness_status', 'UNKNOWN')}",
        f"shadow_only_recommended: {data_quality.get('shadow_only_recommended', True)}",
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
    if (
        snapshot.row_counts.get("strategy_evidence", 0) == 0
        and snapshot.row_counts.get("alpha_discovery_board", 0) == 0
    ):
        questions.append("Why is strategy_evidence empty for V5 candidate discovery?")
    if snapshot.row_counts.get("v5_candidate_event", 0) == 0:
        questions.append("Why is reports/candidate_snapshot.csv missing from V5 bundles?")
    if snapshot.row_counts.get("v5_candidate_label", 0) == 0:
        questions.append("Why are v5_candidate_label forward labels empty?")
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
    if snapshot.row_counts.get("alpha_discovery_board", 0) == 0:
        questions.append("Why is alpha_discovery_board empty for V5 candidate decisions?")
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


def _strategy_evidence_for_export(evidence: pl.DataFrame) -> pl.DataFrame:
    path = "research/strategy_evidence.csv"
    if evidence.is_empty() and not evidence.columns:
        return _empty_csv_schema_frame(path)
    normalized = normalize_strategy_evidence_decisions(evidence)
    sort_columns = [
        column
        for column in [
            "strategy_candidate",
            "symbol",
            "regime_state",
            "horizon_hours",
            "as_of_date",
            "created_at",
        ]
        if column in normalized.columns
    ]
    if sort_columns:
        normalized = normalized.sort(sort_columns)
    keys = [
        key
        for key in [
            "as_of_date",
            "strategy_candidate",
            "symbol",
            "horizon_hours",
            "source_type",
        ]
        if key in normalized.columns
    ]
    if keys:
        normalized = normalized.group_by(keys, maintain_order=True).tail(1)
    for column in CSV_SCHEMAS[path]:
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    final_sort = [
        column
        for column in ["strategy_candidate", "symbol", "regime_state", "horizon_hours"]
        if column in normalized.columns
    ]
    selected = normalized.select(CSV_SCHEMAS[path])
    return selected.sort(final_sort) if final_sort else selected


def _strategy_evidence_samples_for_export(samples: pl.DataFrame) -> pl.DataFrame:
    path = "research/strategy_evidence_samples.csv"
    if samples.is_empty() and not samples.columns:
        return _empty_csv_schema_frame(path)
    selected = _tail_by_time(samples, "ts_utc", limit=20_000)
    for column in CSV_SCHEMAS[path]:
        if column not in selected.columns:
            selected = selected.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    sort_columns = [
        column
        for column in ["strategy_candidate", "symbol", "horizon_hours", "ts_utc"]
        if column in selected.columns
    ]
    if sort_columns:
        selected = selected.sort(sort_columns)
    return selected.select(CSV_SCHEMAS[path])


def _alpha_discovery_board_for_export(board: pl.DataFrame) -> pl.DataFrame:
    path = "reports/alpha_discovery_board.csv"
    if board.is_empty() and not board.columns:
        return _empty_csv_schema_frame(path)
    selected = normalize_alpha_discovery_board_decisions(board)
    for column in CSV_SCHEMAS[path]:
        if column not in selected.columns:
            selected = selected.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    sort_columns = [
        column
        for column in ["decision", "strategy_candidate", "symbol", "horizon_hours"]
        if column in selected.columns
    ]
    selected = selected.sort(sort_columns) if sort_columns else selected
    keys = [
        key
        for key in [
            "strategy_candidate",
            "symbol",
            "regime_state",
            "horizon_hours",
            "as_of_date",
        ]
        if key in selected.columns
    ]
    if keys:
        selected = selected.group_by(keys, maintain_order=True).tail(1)
    return selected.select(CSV_SCHEMAS[path])


def _candidate_decision_rows(board: pl.DataFrame, decision: str) -> pl.DataFrame:
    paths = {
        "KILL": "reports/candidate_kill_list.csv",
        "KEEP_SHADOW": "reports/candidate_shadow_watchlist.csv",
        "PAPER_READY": "reports/candidate_paper_ready.csv",
    }
    path = paths.get(decision, "reports/candidate_shadow_watchlist.csv")
    if board.is_empty():
        return _empty_csv_schema_frame(path)
    if "decision" not in board.columns:
        return _empty_csv_schema_frame(path)
    selected = board.filter(pl.col("decision") == decision)
    if selected.is_empty():
        return _empty_csv_schema_frame(path)
    for column in CSV_SCHEMAS[path]:
        if column not in selected.columns:
            selected = selected.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    sort_columns = [
        column
        for column in ["strategy_candidate", "symbol", "horizon_hours"]
        if column in selected.columns
    ]
    selected = selected.sort(sort_columns) if sort_columns else selected
    return selected.select(CSV_SCHEMAS[path])


PAPER_PROPOSAL_IDS = {
    ("v5.sol_protect_alpha6_low_exception", "SOL-USDT"): (
        "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1"
    ),
    ("v5.f4_volume_expansion_entry", "SOL-USDT"): "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
}


def _paper_strategy_proposals_for_export(board: pl.DataFrame) -> pl.DataFrame:
    path = "reports/paper_strategy_proposals.csv"
    if board.is_empty() or "decision" not in board.columns:
        return _empty_csv_schema_frame(path)
    board_rows = board.to_dicts()
    latest_as_of_date = _latest_as_of_date(board_rows)
    rows = [
        row
        for row in board_rows
        if str(row.get("decision") or "").upper() == "PAPER_READY"
        and str(row.get("symbol") or "").strip().upper() != "UNKNOWN"
    ]
    if latest_as_of_date is not None:
        rows = [
            row
            for row in rows
            if str(row.get("as_of_date") or "").strip() == latest_as_of_date
        ]
    if not rows:
        return _empty_csv_schema_frame(path)

    best_by_candidate: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("strategy_candidate") or ""), str(row.get("symbol") or ""))
        current = best_by_candidate.get(key)
        if current is None or _paper_proposal_rank(row) > _paper_proposal_rank(current):
            best_by_candidate[key] = row

    created_at = datetime.now(UTC).isoformat()
    proposals = [
        {
            "proposal_id": _paper_proposal_id(row),
            "strategy_candidate": row.get("strategy_candidate"),
            "symbol": row.get("symbol"),
            "entry_conditions": safe_json_dumps(_paper_entry_conditions(row)),
            "recommended_mode": "paper",
            "suggested_horizon": _paper_suggested_horizon(row),
            "sample_count": _optional_int(row.get("sample_count")),
            "complete_sample_count": _optional_int(row.get("complete_sample_count")),
            "avg_net_bps": _optional_float(row.get("avg_net_bps")),
            "p25_net_bps": _optional_float(row.get("p25_net_bps")),
            "win_rate": _optional_float(row.get("win_rate")),
            "cost_source_mix": row.get("cost_source_mix"),
            "live_block_reason": safe_json_dumps(_paper_live_block_reasons(row)),
            "required_paper_days": 14,
            "required_slippage_coverage": 0.8,
            "as_of_date": row.get("as_of_date"),
            "created_at": created_at,
        }
        for row in sorted(
            best_by_candidate.values(),
            key=lambda item: (
                str(item.get("symbol") or ""),
                str(item.get("strategy_candidate") or ""),
            ),
        )
    ]
    return pl.DataFrame(proposals).select(CSV_SCHEMAS[path])


def _latest_as_of_date(rows: list[dict[str, Any]]) -> str | None:
    values = [
        str(row.get("as_of_date") or "").strip()
        for row in rows
        if str(row.get("as_of_date") or "").strip()
    ]
    return max(values) if values else None


def _paper_proposal_rank(row: dict[str, Any]) -> tuple[float, float, int, int, int]:
    return (
        _optional_float(row.get("avg_net_bps")) or float("-inf"),
        _optional_float(row.get("win_rate")) or float("-inf"),
        _optional_int(row.get("complete_sample_count")) or 0,
        _optional_int(row.get("sample_count")) or 0,
        _optional_int(row.get("horizon_hours")) or 0,
    )


def _paper_proposal_id(row: dict[str, Any]) -> str:
    candidate = str(row.get("strategy_candidate") or "")
    symbol = str(row.get("symbol") or "")
    mapped = PAPER_PROPOSAL_IDS.get((candidate, symbol))
    if mapped:
        return mapped
    stem = candidate.removeprefix("v5.")
    sanitized = "".join(char if char.isalnum() else "_" for char in f"{symbol}_{stem}")
    return f"{sanitized.upper()}_PAPER_V1"


def _paper_suggested_horizon(row: dict[str, Any]) -> str:
    horizon = _optional_int(row.get("horizon_hours"))
    return f"{horizon}h" if horizon is not None else ""


def _paper_entry_conditions(row: dict[str, Any]) -> dict[str, Any]:
    conditions: dict[str, Any] = {
        "strategy_candidate": row.get("strategy_candidate"),
        "symbol": row.get("symbol"),
        "board_decision": "PAPER_READY",
    }
    for source, target in [
        ("regime_state", "regime_state"),
        ("source_type", "evidence_source"),
        ("horizon_hours", "horizon_hours"),
        ("block_reason_mix", "block_reason_mix"),
        ("final_decision_mix", "final_decision_mix"),
    ]:
        value = row.get(source)
        if value not in (None, ""):
            conditions[target] = value
    return conditions


def _paper_live_block_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _cost_source_mix_has_actual_or_mixed(row.get("cost_source_mix")):
        reasons.append("cost_source_not_actual_or_mixed")
    paper_days = _optional_int(row.get("paper_days")) or 0
    if paper_days < 14:
        reasons.append("no_paper_days")
    reasons.append("no_live_slippage_coverage")
    return reasons


def _cost_source_mix_has_actual_or_mixed(value: Any) -> bool:
    text = safe_json_dumps(value) if isinstance(value, dict | list | tuple) else str(value or "")
    text = text.lower()
    return "actual" in text or "mixed_actual_proxy" in text


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _strategy_evidence_summary_md(
    alpha_discovery_board: pl.DataFrame,
    strategy_evidence: pl.DataFrame,
) -> str:
    lines = [
        "# Strategy Evidence Summary",
        "",
        "Alpha discovery evaluates V5 strategy candidates by candidate snapshot and "
        "forward labels, not only by `v5.core.momentum` feature gates.",
        "",
    ]
    if alpha_discovery_board.is_empty():
        lines.extend(["No alpha discovery board rows are available.", ""])
        if strategy_evidence.is_empty():
            lines.extend(["No legacy strategy evidence rows are available.", ""])
        return "\n".join(lines)

    counts = _value_counts(alpha_discovery_board, "decision")
    lines.extend(
        [
            f"Decision counts: {safe_json_dumps(counts)}",
            "",
            "| Candidate | Symbol | Regime | Horizon | Decision | Samples | Complete |",
        ]
    )
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: |")
    sort_columns = [
        column
        for column in ["strategy_candidate", "symbol", "regime_state", "horizon_hours"]
        if column in alpha_discovery_board.columns
    ]
    board = alpha_discovery_board.sort(sort_columns) if sort_columns else alpha_discovery_board
    for row in board.to_dicts():
        lines.append(
            "| {candidate} | {symbol} | {regime} | {horizon} | {decision} | "
            "{samples} | {complete} |".format(
                candidate=row.get("strategy_candidate") or row.get("candidate_name", ""),
                symbol=row.get("symbol", ""),
                regime=row.get("regime_state", ""),
                horizon=row.get("horizon_hours", ""),
                decision=row.get("decision", ""),
                samples=row.get("sample_count", 0),
                complete=row.get("complete_sample_count", 0),
            )
        )
    lines.append("")
    return "\n".join(lines)


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


def _decision_audit_quality(frames: dict[str, pl.DataFrame]) -> dict[str, Any]:
    generic = frames.get("decision_audit", pl.DataFrame())
    v5_decisions = frames.get("v5_decision_audit", pl.DataFrame())
    strategy_health = frames.get("strategy_health_daily", pl.DataFrame())
    health_count = 0
    if not strategy_health.is_empty():
        latest = _latest_by_dataset_time("strategy_health_daily", strategy_health)
        for field in ["decision_audit_count_24h", "decision_audit_count"]:
            value = latest.get(field)
            if value is None:
                continue
            try:
                health_count = max(health_count, int(float(value)))
            except (TypeError, ValueError):
                continue
    v5_count = max(v5_decisions.height, health_count)
    return {
        "generic_decision_audit_present": generic.height > 0,
        "generic_decision_audit_rows": generic.height,
        "v5_decision_audit_count": v5_count,
        "v5_decision_audit_present": v5_count > 0,
        "v5_decision_audit_rows": v5_decisions.height,
        "v5_decision_audit_count_source": (
            "strategy_health_daily"
            if health_count >= v5_decisions.height and health_count > 0
            else "silver/v5_decision_audit"
            if v5_decisions.height > 0
            else "none"
        ),
    }


def _risk_permission_quality(
    risk: pl.DataFrame,
    frames: dict[str, pl.DataFrame],
    *,
    reference_at: datetime | None = None,
) -> dict[str, Any]:
    checked_at = reference_at or datetime.now(UTC)
    telemetry_latest_ts = _latest_v5_bundle_ts(frames)
    if risk.is_empty():
        return {
            "permission_status": "NO_FRESH_PERMISSION",
            "permission": None,
            "enforceable": False,
            "as_of_ts": None,
            "api_permission_as_of_ts": None,
            "gold_permission_as_of_ts": None,
            "permission_api_lag_sec": None,
            "permission_api_consistent_with_gold": False,
            "created_at": None,
            "risk_permission_generated_at": None,
            "risk_permission_expires_at": None,
            "export_generated_at": checked_at.isoformat(),
            "permission_expired_at_export": False,
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
    expired = False
    expires_at = _risk_row_timestamp(row, ["expires_at"])
    if expires_at is not None:
        expired = expires_at < checked_at
    status = (
        stored_status
        if stored_status.startswith(("ACTIVE_", "STALE_", "EXPIRED_"))
        and stored_status.endswith(permission)
        and stored_status.startswith("STALE_") == (stale and not expired)
        and stored_status.startswith("EXPIRED_") == expired
        else permission_status(permission, stale=stale and not expired, expired=expired).value
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
        "api_permission_as_of_ts": _iso_or_none(as_of_ts),
        "gold_permission_as_of_ts": _iso_or_none(as_of_ts),
        "permission_api_lag_sec": 0,
        "permission_api_consistent_with_gold": True,
        "created_at": _iso_or_none(created_at),
        "risk_permission_generated_at": _iso_or_none(as_of_ts or created_at),
        "risk_permission_expires_at": _iso_or_none(expires_at),
        "export_generated_at": checked_at.isoformat(),
        "permission_expired_at_export": expired,
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
    *,
    reference_at: datetime | None = None,
) -> pl.DataFrame:
    if risk.is_empty():
        return risk
    checked_at = reference_at or datetime.now(UTC)
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
        expires_at = _risk_row_timestamp(row, ["expires_at"])
        expired = expires_at is not None and expires_at < checked_at
        status = permission_status(permission, stale=stale and not expired, expired=expired).value
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
                "enforceable": is_permission_status_enforceable(status) and not expired,
            }
        )
    return pl.DataFrame(rows) if rows else risk


def _risk_permission_export_status(
    frames: dict[str, pl.DataFrame],
    generated_at: datetime,
) -> dict[str, Any]:
    risk = _risk_permissions_for_export(
        frames.get("risk_permission", pl.DataFrame()),
        frames,
        reference_at=generated_at,
    )
    quality = _risk_permission_quality(risk, frames, reference_at=generated_at)
    return {
        "risk_permission_generated_at": quality.get("risk_permission_generated_at"),
        "risk_permission_expires_at": quality.get("risk_permission_expires_at"),
        "export_generated_at": quality.get("export_generated_at"),
        "permission_expired_at_export": quality.get("permission_expired_at_export"),
        "permission_status": quality.get("permission_status"),
        "next_action": quality.get("next_action"),
    }


def _risk_permission_expired_detail(risk_quality: dict[str, Any]) -> str:
    if not risk_quality.get("permission_expired_at_export"):
        return (
            f"ok; risk_permission_expires_at={risk_quality.get('risk_permission_expires_at')}; "
            f"export_generated_at={risk_quality.get('export_generated_at')}"
        )
    return (
        "risk_permission expired before expert export; "
        f"risk_permission_generated_at={risk_quality.get('risk_permission_generated_at')}; "
        f"risk_permission_expires_at={risk_quality.get('risk_permission_expires_at')}; "
        f"export_generated_at={risk_quality.get('export_generated_at')}; "
        "next_action=run qlab publish-risk-permission before qlab export-daily"
    )


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


def _latest_candidate_quality(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {}
    if "created_at" in frame.columns:
        return _latest_by_dataset_time("v5_candidate_quality_daily", frame)
    if "date" in frame.columns:
        return frame.sort("date").tail(1).to_dicts()[0]
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


def _jsonish_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
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


def _cost_fallback_stats(costs: pl.DataFrame, latest_cost_health: dict[str, Any]) -> dict[str, Any]:
    row_count = costs.height
    legacy_ratio = _float_or_none(latest_cost_health.get("fallback_ratio"))
    hard_ratio = _float_or_none(latest_cost_health.get("hard_fallback_ratio"))
    soft_ratio = _float_or_none(latest_cost_health.get("soft_fallback_ratio"))
    hard_count = _int_or_none(latest_cost_health.get("hard_fallback_count"))
    soft_count = _int_or_none(latest_cost_health.get("soft_fallback_count"))
    proxy_only_count = _int_or_none(latest_cost_health.get("proxy_only_count"))
    global_default_count = _int_or_none(latest_cost_health.get("global_default_count"))
    if row_count > 0 and (
        hard_ratio is None
        or soft_ratio is None
        or hard_count is None
        or soft_count is None
        or proxy_only_count is None
        or global_default_count is None
    ):
        rows = costs.to_dicts()
        hard_count = sum(1 for row in rows if _is_hard_cost_fallback(row))
        soft_count = sum(1 for row in rows if _is_soft_cost_fallback(row))
        proxy_only_count = sum(
            1
            for row in rows
            if str(row.get("source") or row.get("cost_source") or "").lower()
            == "public_spread_proxy"
        )
        global_default_count = sum(
            1
            for row in rows
            if str(row.get("source") or row.get("cost_source") or "").lower()
            == "global_default"
        )
        hard_ratio = hard_count / row_count
        soft_ratio = soft_count / row_count
    if legacy_ratio is None:
        legacy_ratio = 0.0 if row_count == 0 else _cost_fallbacks(costs).height / row_count
    return {
        "fallback_ratio": legacy_ratio,
        "hard_fallback_count": hard_count or 0,
        "hard_fallback_ratio": hard_ratio or 0.0,
        "soft_fallback_count": soft_count or 0,
        "soft_fallback_ratio": soft_ratio or 0.0,
        "proxy_only_count": proxy_only_count or 0,
        "global_default_count": global_default_count or 0,
    }


def _is_hard_cost_fallback(row: dict[str, Any]) -> bool:
    text = _cost_fallback_text(row)
    source = str(row.get("source") or row.get("cost_source") or "").lower()
    return source == "global_default" or any(
        token in text
        for token in [
            "GLOBAL_DEFAULT",
            "SYMBOL_MISSING",
            "SERVICE_UNAVAILABLE",
            "STALE_COST_BUCKET",
        ]
    )


def _is_soft_cost_fallback(row: dict[str, Any]) -> bool:
    if _is_hard_cost_fallback(row):
        return False
    text = _cost_fallback_text(row)
    return any(token in text for token in ["SAMPLE_TOO_SMALL", "SLIPPAGE_UNKNOWN", "SPREAD_PROXY"])


def _cost_fallback_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(column) or "").upper()
        for column in ["source", "cost_source", "fallback_level", "fallback_reason"]
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


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
        {"dataset": name, "reason": _missing_dataset_reason(name)}
        for name, frame in sorted(frames.items())
        if frame.is_empty()
    ]
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(schema={"dataset": pl.Utf8, "reason": pl.Utf8})
    )


def _missing_dataset_reason(dataset_name: str) -> str:
    if dataset_name == "decision_audit":
        return "legacy_optional_non_v5_research_missing"
    if dataset_name == "v5_decision_audit":
        return "v5_decision_audit_missing_or_empty"
    if dataset_name == "alpha_discovery_board":
        return "alpha_discovery_board_missing_or_empty"
    if dataset_name == "strategy_evidence":
        return "strategy_candidate_evidence_missing_or_empty"
    if dataset_name == "strategy_evidence_sample":
        return "strategy_candidate_samples_missing_or_empty"
    if dataset_name == "v5_candidate_event":
        return "v5_candidate_snapshot_missing_or_empty"
    if dataset_name == "v5_candidate_label":
        return "v5_candidate_forward_labels_missing_or_empty"
    if dataset_name == "v5_candidate_quality_daily":
        return "v5_candidate_quality_missing_or_empty"
    if dataset_name == "v5_candidate_outcome_summary":
        return "v5_candidate_outcome_summary_missing_or_empty"
    return "missing_or_empty"


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


def _strategy_evidence_has_required_candidates(strategy_evidence: pl.DataFrame) -> bool:
    if strategy_evidence.is_empty() or "candidate_name" not in strategy_evidence.columns:
        return False
    required = {
        "v5.btc_leadership_probe_strict",
        "v5.btc_leadership_blocked_relaxed",
        "v5.sol_protect_exception",
        "v5.alt_impulse_shadow",
        "v5.multi_position_k1",
        "v5.multi_position_k2",
        "v5.multi_position_k3",
        "v5.swing_f4_f5_alpha6",
        "v5.f3_dominant_entry",
        "v5.f4_volume_expansion_entry",
        "v5.mean_reversion_sideways",
    }
    observed = {str(value) for value in strategy_evidence["candidate_name"].drop_nulls()}
    return required.issubset(observed)


def _strategy_evidence_live_sample_floor_ok(strategy_evidence: pl.DataFrame) -> bool:
    if strategy_evidence.is_empty() or "decision" not in strategy_evidence.columns:
        return True
    if "sample_count" not in strategy_evidence.columns:
        return False
    live = strategy_evidence.filter(pl.col("decision") == "LIVE_SMALL_READY")
    if live.is_empty():
        return True
    return live.filter(pl.col("sample_count").cast(pl.Int64, strict=False) < 30).is_empty()


def _alpha_discovery_live_rules_ok(board: pl.DataFrame) -> bool:
    if board.is_empty() or "decision" not in board.columns:
        return False
    if "sample_count" not in board.columns:
        return False
    live = board.filter(pl.col("decision") == "LIVE_SMALL_READY")
    if live.is_empty():
        return True
    if not live.filter(pl.col("sample_count").cast(pl.Int64, strict=False) < 60).is_empty():
        return False
    if "strategy_candidate" in live.columns:
        blocked = live.filter(
            pl.col("strategy_candidate").is_in(
                [
                    "v5.alt_impulse_shadow",
                    "v5.sol_protect_exception",
                    "v5.multi_position_k2",
                    "v5.multi_position_k3",
                ]
            )
        )
        if not blocked.is_empty():
            return False
    if "cost_source_has_global_default" in live.columns:
        if bool(live["cost_source_has_global_default"].fill_null(True).any()):
            return False
    return True


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
    dirty = bool(_git_command(["status", "--porcelain"], root))
    return {
        "git_commit": _git_command(["rev-parse", "--short", "HEAD"], root),
        "git_branch": _git_command(["branch", "--show-current"], root),
        "dirty_worktree": dirty,
        "provenance_status": "git_dirty" if dirty else "git_clean",
        "code_provenance": "degraded" if dirty else "ok",
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
