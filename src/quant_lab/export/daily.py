from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

os.environ.setdefault("POLARS_MAX_THREADS", "2")

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.backtest.reports import (
    BACKTEST_CSV_SCHEMAS,
    FACTOR_FORWARD_VALIDATION_FIELDS,
    build_backtest_report_bundle,
    build_factor_forward_validation,
    factor_forward_validation_md,
)
from quant_lab.contracts.v5_quant_lab import (
    V5_QUANT_LAB_CONTRACT_VERSION,
    V5_TELEMETRY_DATASET_SCHEMA_VERSION,
)
from quant_lab.costs.calibrate import PRIVATE_COST_LOOKBACK_DAYS
from quant_lab.costs.model import (
    COST_BOOTSTRAP_READINESS_FIELDS,
    DEFAULT_LIVE_UNIVERSE_SYMBOLS,
    LIVE_UNIVERSE_COST_COVERAGE_FIELDS,
    build_cost_bootstrap_readiness,
    build_live_universe_cost_coverage,
    evaluate_live_universe_cost_coverage,
)
from quant_lab.costs.probe import (
    COST_PROBE_COST_DISAGREEMENT_FIELDS,
    COST_PROBE_FILL_BILL_MATCH_FIELDS,
    build_cost_probe_cost_disagreement,
    build_cost_probe_fill_bill_match,
    canonical_cost_probe_live_execution_status,
    canonical_cost_probe_roundtrip_events,
    cost_probe_private_fill_keys,
    private_fill_matches_cost_probe,
)
from quant_lab.data.file_index import build_lake_file_index
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.factors.composite_factory import (
    COMPOSITE_FACTOR_CANDIDATE_FIELDS,
    FACTOR_DEDUPE_DECISION_FIELDS,
    FACTOR_FAMILY_LEADERBOARD_FIELDS,
    FACTOR_PAPER_REVIEW_QUEUE_FIELDS,
    FACTOR_REGIME_EFFECTIVENESS_FIELDS,
    FACTOR_STRATEGY_BRIDGE_CANDIDATE_FIELDS,
    build_factor_factory_v2_reports,
)
from quant_lab.features.fast_microstructure import (
    FAST_MICROSTRUCTURE_FIELDS,
    FAST_MICROSTRUCTURE_FORWARD_TEST_FIELDS,
    FAST_MICROSTRUCTURE_STRATEGY_CANDIDATE_FIELDS,
    FAST_MICROSTRUCTURE_STRATEGY_REVIEW_FIELDS,
    build_fast_microstructure_features,
    build_fast_microstructure_forward_test,
    build_fast_microstructure_strategy_candidates,
    build_fast_microstructure_strategy_review,
    fast_microstructure_forward_summary_md,
)
from quant_lab.opportunity_cost.ledger import (
    DECISION_REGRET_SCHEMA,
    OPPORTUNITY_COST_BY_BUCKET_SCHEMA,
    OPPORTUNITY_COST_DAILY_SCHEMA,
    OPPORTUNITY_COST_EVENT_SCHEMA,
)
from quant_lab.ops.api_metrics import api_error_summary, api_metrics_summary
from quant_lab.ops.data_quality import run_data_quality
from quant_lab.ops.lake_health import lake_file_health_summary
from quant_lab.paper.proposals import PAPER_STRATEGY_MIGRATION_AUDIT_SCHEMA
from quant_lab.reports.enforce_readiness import (
    ENFORCE_READINESS_CSV,
    ENFORCE_READINESS_JSON,
    FALLBACK_RATE_BREAKDOWN_COLUMNS,
    FALLBACK_RATE_BREAKDOWN_CSV,
    build_enforce_readiness_report,
    build_fallback_rate_breakdown,
    enforce_readiness_members,
    write_enforce_readiness_report,
)
from quant_lab.reports.ops_truthfulness import (
    build_api_auth_reports,
    build_complete_acceptance_status,
    build_paper_proposal_propagation_status,
    build_paper_runtime_freshness,
    build_post_fix_funnel_attribution,
)
from quant_lab.reports.system_acceptance import (
    NO_TRIGGER_REASON_FIELDS,
    SYSTEM_ACCEPTANCE_FIELDS,
    build_no_trigger_reasons,
    build_system_acceptance_dashboard,
    system_acceptance_dashboard_md,
)
from quant_lab.research.advisory_overrides import (
    portfolio_overridden_decision_mode,
    portfolio_override_for_identifier_symbol,
    portfolio_override_for_row,
    portfolio_status_overrides_by_identifier,
)
from quant_lab.research.alpha_discovery import normalize_alpha_discovery_board_decisions
from quant_lab.research.alpha_factory import ALPHA_FACTORY_CANDIDATES, alpha_factory_daily_md
from quant_lab.research.baselines import (
    CORE_MOMENTUM_ALPHA_ID,
    RESEARCH_BASELINE_ROLE,
    alpha_role,
)
from quant_lab.research.bnb_swing_exit_policy import bnb_swing_exit_policy_summary_md
from quant_lab.research.bottom_zone_reversal import (
    BOTTOM_ZONE_FIELDS,
    bottom_zone_reversal_summary_md,
    build_bottom_zone_reversal_shadow,
)
from quant_lab.research.btc_probe_exit_policy import btc_probe_exit_policy_summary_md
from quant_lab.research.gate_effectiveness import (
    EFFECTIVENESS_SCHEMA,
    build_gate_effectiveness_report,
)
from quant_lab.research.market_pressure import (
    MARKET_PRESSURE_FIELDS,
    build_market_pressure_score,
    market_pressure_summary_md,
)
from quant_lab.research.paper_promotion import (
    PAPER_STRATEGY_IDENTITY_CONFLICT_SCHEMA,
    PAPER_STRATEGY_PROMOTION_GATE_SCHEMA,
    PAPER_STRATEGY_REGISTRY_SCHEMA,
    build_and_publish_paper_strategy_pipeline,
    build_paper_strategy_pipeline_frames,
)
from quant_lab.research.paper_tracking import (
    build_paper_slippage_coverage,
    build_paper_slippage_coverage_from_v5,
    build_paper_strategy_daily_from_runs,
    build_paper_strategy_daily_from_v5,
    build_paper_strategy_runs_from_v5,
    build_paper_strategy_runs_report_from_v5,
    enrich_paper_strategy_daily_from_runs,
    latest_v5_paper_frame,
    paper_strategy_summary_md,
)
from quant_lab.research.portfolio import (
    dedupe_research_portfolio_status,
    research_portfolio_summary_md,
)
from quant_lab.research.sol_protect_paper_loss import (
    sol_protect_paper_loss_summary_md,
)
from quant_lab.research.strategy_evidence import (
    normalize_strategy_evidence_decisions,
    strategy_evidence_decision_ladder,
)
from quant_lab.risk.publish import (
    DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    is_permission_status_enforceable,
    permission_freshness_sec,
    permission_status,
    publish_risk_permission,
    risk_permission_stale_vs_telemetry,
)
from quant_lab.strategy_telemetry.analyze import (
    _event_key as _v5_telemetry_event_key,
)
from quant_lab.strategy_telemetry.bundle import (
    compute_sha256,
    safe_extract_v5_bundle,
    validate_v5_bundle,
)
from quant_lab.strategy_telemetry.models import BundleLimits
from quant_lab.strategy_telemetry.sanitize import (
    SECRET_PATTERNS,
    redact_extracted_bundle,
    safe_json_dumps,
)
from quant_lab.symbols import normalize_symbol
from quant_lab.time_display import BEIJING_TZ, DISPLAY_TIMEZONE, beijing_iso
from quant_lab.trade_learning.attribution import V5_TRADE_OUTCOME_ATTRIBUTION_SCHEMA
from quant_lab.trade_learning.samples import V5_TRADE_LEARNING_SAMPLE_SCHEMA
from quant_lab.trade_level.bucket_policy import TRADE_LEVEL_BUCKET_POLICY_SCHEMA
from quant_lab.trade_level.judgment import (
    FALSE_BLOCK_AUDIT_SCHEMA,
    TRADE_LEVEL_JUDGMENT_SCHEMA,
    TRADE_OPPORTUNITY_EVENT_SCHEMA,
    build_trade_level_frames_from_sources,
)
from quant_lab.trade_level.labels import TRADE_OPPORTUNITY_LABEL_SCHEMA
from quant_lab.trade_level.opportunity_queue import TRADE_LEVEL_OPPORTUNITY_QUEUE_SCHEMA
from quant_lab.trade_level.similarity import TRADE_LEVEL_SIMILARITY_SCHEMA
from quant_lab.web import readers

STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION = "strategy_opportunity_advisory.v0.1"
WEB_DERIVED_SNAPSHOT_DATASETS = (
    "factor_strategy_bridge_candidates",
    "strategy_opportunity_advisory",
    "v5_missed_opportunity_audit",
    "v5_risk_on_multi_buy_shadow",
    "risk_on_multi_buy_shadow",
    "cost_bootstrap_readiness",
    "cost_probe_fill_bill_match",
    "cost_probe_cost_disagreement",
)
SNAPSHOT_META_DATASETS = {
    "cost_bucket_daily",
    "cost_health_daily",
    "cost_probe_cost_disagreement",
    "cost_probe_fill_bill_match",
    "factor_strategy_bridge_candidates",
    "gate_decision",
    "paper_strategy_migration_audit",
    "paper_strategy_promotion_gate",
    "paper_strategy_proposal",
    "paper_strategy_proposals_current",
    "paper_strategy_trackers_current",
    "paper_strategy_registry",
    "paper_strategy_registry_current",
    "paper_strategy_registry_history",
    "paper_strategy_ack_current",
    "paper_strategy_ack_history",
    "paper_strategy_identity_conflict",
    "paper_cohort_manifest",
    "api_auth_incident",
    "paper_runtime_freshness",
    "paper_proposal_propagation_status",
    "strategy_health_daily",
    "strategy_cost_trust",
    "strategy_opportunity_advisory",
    "trade_opportunity_event",
    "trade_opportunity_label",
    "trade_level_similarity_outcome",
    "trade_level_judgment",
    "trade_level_bucket_policy",
    "trade_level_opportunity_queue",
    "quant_lab_false_block_audit",
    "v5_trade_learning_sample",
    "v5_trade_outcome_attribution",
    "quant_lab_opportunity_cost_event",
    "quant_lab_opportunity_cost_daily",
    "opportunity_cost_by_bucket",
    "quant_lab_decision_regret",
    "v5_decision_audit",
    "v5_trade_event",
    "v5_roundtrip",
    "v5_open_position",
    "v5_state_snapshot",
    "v5_quant_lab_usage",
    "v5_quant_lab_request",
    "v5_quant_lab_compliance",
    "v5_quant_lab_cost_usage",
    "v5_quant_lab_fallback",
    "v5_candidate_event",
    "v5_cost_probe_p3_preflight",
    "v5_cost_probe_live_execution_status",
    "v5_cost_probe_order_event",
    "v5_cost_probe_roundtrip_event",
    "v5_fill_bill_cost_reconciliation",
    "v5_trade_opportunity_funnel",
    "v5_paper_strategy_run",
    "v5_paper_strategy_exit_quality",
    "v5_paper_strategy_proposal_ack",
    "v5_paper_strategy_daily",
    "v5_paper_strategy_registry",
    "v5_paper_strategy_state",
    "v5_paper_strategy_signal",
    "v5_paper_strategy_quote_coverage",
    "v5_paper_strategy_cost_evidence",
    "v5_paper_strategy_error",
    "v5_paper_strategy_restart_recovery",
    "v5_quant_lab_contract_status",
    "v5_paper_slippage_coverage",
    "v5_expanded_universe_advisory_reader",
    "v5_expanded_universe_paper_runs",
    "v5_expanded_universe_paper_daily",
    "v5_btc_probe_entry_quality_audit",
    "v5_bnb_profit_lock_shadow",
    "v5_bnb_negative_expectancy_attribution",
    "v5_final_score_vs_alpha6_conflict",
    "v5_final_score_vs_alpha6_conflict_silver",
    "v5_bnb_strong_alpha6_bypass_shadow",
    "v5_bnb_strong_alpha6_bypass_shadow_silver",
    "v5_negative_expectancy_attribution",
    "v5_negative_expectancy_attribution_silver",
    "v5_bnb_paper_strategy_runs",
    "v5_bnb_paper_strategy_runs_silver",
    "v5_bnb_paper_strategy_daily",
    "v5_bnb_paper_strategy_daily_silver",
    "v5_bnb_paper_strategy_daily_latest",
    "v5_negative_expectancy_consistency",
    "risk_permission",
    "risk_permission_api_dependency_meta",
    "research_portfolio_status",
    "alpha_factory_promotion_queue",
    "alpha_factory_result",
    "v5_gate_compliance_daily",
}
STRATEGY_OPPORTUNITY_ADVISORY_TTL_SECONDS = 3 * 60 * 60
UNOBSERVABLE_TEXT_VALUES = {"", "none", "null", "nan", "not_observable", "unknown"}
V5_CANDIDATE_CORE_SIGNAL_FIELDS = (
    "final_score",
    "f1_mom_5d",
    "f2_mom_20d",
    "f3_vol_adj_ret",
    "f4_volume_expansion",
    "f5_rsi_trend_confirm",
    "alpha6_score",
    "alpha6_side",
)
V5_CANDIDATE_EDGE_CONTEXT_FIELDS = (
    "expected_edge_bps",
    "required_edge_bps",
    "cost_source",
)
V5_CANDIDATE_OPTIONAL_SIGNAL_FIELDS = (
    "ml_score",
    "mean_reversion_score",
)

SECTIONS = ["market", "features", "costs", "research", "risk", "anomalies", "v5", "charts"]
HEAVY_EXPORT_DATASET_LIMITS = {
    "market_bar": 20_000,
    "api_request_metrics": 10_000,
    "okx_public_ws": 5_000,
    "feature_value": 10_000,
    "factor_value": 20_000,
    "factor_evidence": 20_000,
    "factor_candidate": 10_000,
    "factor_correlation_daily": 20_000,
    "job_run_history": 10_000,
    "strategy_evidence_sample": 10_000,
    "second_stage_alpha_factory_sample": 10_000,
    "expanded_relative_strength_decision_sample": 20_000,
    "alpha_factory_candidate": 10_000,
    "alpha_factory_result": 10_000,
    "alpha_factory_promotion_queue": 10_000,
    "trade_print": 10_000,
    "orderbook_snapshot": 10_000,
    # Fast microstructure forward validation needs more than a few hours of
    # all-symbol 1m rollups; 100k rows keeps roughly 40-50h on the current
    # production universe while staying well below full-dataset export size.
    "trade_activity_1m": 100_000,
    "orderbook_spread_1m": 100_000,
    "v5_decision_audit": 5_000,
    "v5_trade_event": 10_000,
    "v5_missed_opportunity_audit": 20_000,
    "v5_risk_on_multi_buy_shadow": 20_000,
    "risk_on_multi_buy_shadow": 20_000,
    "v5_quant_lab_usage": 5_000,
    "v5_quant_lab_compliance": 5_000,
    "v5_quant_lab_cost_usage": 5_000,
    "v5_quant_lab_fallback": 5_000,
    "v5_cost_probe_p3_preflight": 5_000,
    "v5_cost_probe_live_execution_status": 5_000,
    "v5_cost_probe_order_event": 10_000,
    "v5_cost_probe_roundtrip_event": 10_000,
    "v5_fill_bill_cost_reconciliation": 20_000,
    "v5_candidate_event": 20_000,
    "trade_opportunity_event": 20_000,
    "trade_opportunity_label": 20_000,
    "trade_level_similarity_outcome": 20_000,
    "trade_level_judgment": 20_000,
    "trade_level_bucket_policy": 20_000,
    "trade_level_opportunity_queue": 20_000,
    "quant_lab_false_block_audit": 20_000,
    "v5_trade_learning_sample": 20_000,
    "v5_trade_outcome_attribution": 20_000,
    "quant_lab_opportunity_cost_event": 20_000,
    "quant_lab_opportunity_cost_daily": 5_000,
    "opportunity_cost_by_bucket": 20_000,
    "quant_lab_decision_regret": 20_000,
    "v5_btc_probe_entry_quality_audit": 20_000,
    "v5_candidate_label": 20_000,
    "v5_missed_low_audit": 20_000,
    "v5_late_entry_chase_shadow": 20_000,
    "v5_pullback_reversal_shadow": 20_000,
    "v5_entry_quality_history_anti_leakage_check": 20_000,
    "expanded_universe_candidate_event": 20_000,
    "expanded_universe_candidate_label": 20_000,
    "expanded_crypto_universe_shadow": 20_000,
    "expanded_crypto_candidate_outcomes_by_symbol": 20_000,
    "v5_trade_opportunity_funnel": 20_000,
    "v5_paper_strategy_run": 20_000,
    "v5_paper_strategy_exit_quality": 20_000,
    "v5_paper_strategy_signal": 20_000,
    "v5_paper_strategy_error": 10_000,
    "v5_paper_strategy_restart_recovery": 10_000,
    "paper_strategy_runs": 20_000,
}
HEAVY_EXPORT_RECENT_FILE_LIMITS = {
    "market_bar": 120,
    "api_request_metrics": 50,
    "okx_public_ws": 50,
    "feature_value": 100,
    "factor_value": 100,
    "factor_evidence": 100,
    "factor_candidate": 100,
    "factor_correlation_daily": 100,
    "job_run_history": 50,
    "strategy_evidence_sample": 100,
    "second_stage_alpha_factory_sample": 100,
    "expanded_relative_strength_decision_sample": 100,
    "alpha_factory_candidate": 100,
    "alpha_factory_result": 100,
    "alpha_factory_promotion_queue": 100,
    "trade_print": 50,
    "orderbook_snapshot": 50,
    "v5_decision_audit": 100,
    "v5_trade_event": 100,
    "v5_missed_opportunity_audit": 100,
    "v5_risk_on_multi_buy_shadow": 100,
    "risk_on_multi_buy_shadow": 100,
    "v5_quant_lab_usage": 100,
    "v5_quant_lab_compliance": 100,
    "v5_quant_lab_cost_usage": 100,
    "v5_quant_lab_fallback": 100,
    "v5_cost_probe_p3_preflight": 100,
    "v5_cost_probe_order_event": 100,
    "v5_cost_probe_roundtrip_event": 100,
    "v5_fill_bill_cost_reconciliation": 100,
    "v5_candidate_event": 100,
    "trade_opportunity_event": 100,
    "trade_opportunity_label": 100,
    "trade_level_similarity_outcome": 100,
    "trade_level_judgment": 100,
    "trade_level_bucket_policy": 100,
    "trade_level_opportunity_queue": 100,
    "quant_lab_false_block_audit": 100,
    "v5_trade_learning_sample": 100,
    "v5_trade_outcome_attribution": 100,
    "quant_lab_opportunity_cost_event": 100,
    "quant_lab_opportunity_cost_daily": 100,
    "opportunity_cost_by_bucket": 100,
    "quant_lab_decision_regret": 100,
    "v5_btc_probe_entry_quality_audit": 100,
    "v5_candidate_label": 100,
    "v5_missed_low_audit": 100,
    "v5_late_entry_chase_shadow": 100,
    "v5_pullback_reversal_shadow": 100,
    "v5_entry_quality_history_anti_leakage_check": 100,
    "expanded_universe_candidate_event": 100,
    "expanded_universe_candidate_label": 100,
    "expanded_crypto_universe_shadow": 100,
    "expanded_crypto_candidate_outcomes_by_symbol": 100,
    "v5_trade_opportunity_funnel": 100,
    "v5_paper_strategy_run": 100,
    "v5_paper_strategy_exit_quality": 100,
    "v5_paper_strategy_signal": 100,
    "v5_paper_strategy_error": 100,
    "v5_paper_strategy_restart_recovery": 100,
    "paper_strategy_runs": 100,
}
DEFAULT_EXPORT_SAMPLED_ROW_LIMIT = 20_000
DEFAULT_EXPORT_RECENT_FILE_LIMIT = 100
DEFAULT_EXPORT_FULL_READ_MAX_FILES = 80
DEFAULT_EXPORT_FULL_READ_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_KEEP_EXPERT_PACKS = 5
EMBEDDED_V5_BUNDLE_DIR = "v5/followup_bundle"
EMBEDDED_V5_BUNDLE_MANIFEST = f"{EMBEDDED_V5_BUNDLE_DIR}/attachment_manifest.json"
V5_FOLLOWUP_BUNDLE_PREFIX = "v5_live_followup_bundle_"
V5_BUNDLE_SECRET_SCAN_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
V5_BUNDLE_SECRET_SCAN_MAX_MEMBER_BYTES = 8 * 1024 * 1024
LAKE_FILE_INDEX_DATASET = Path("bronze") / "lake_file_index"
_EXPORT_FILE_INDEX_CACHE: dict[Path, pl.DataFrame | None] = {}
SECRET_SCAN_PREFILTER_TOKENS = (
    "private key",
    "private_key",
    "private-key",
    "ssh-rsa",
    "ed25519",
    "ok-access",
    "api_key",
    "api-key",
    "apikey",
    "apisecret",
    "api_secret",
    "api-secret",
    "secret_key",
    "secret-key",
    "secretkey",
    "passphrase",
    "password",
    "token",
    "authorization",
)
HEAVY_REGISTRY_QUALITY_DATASETS = {
    "okx_public_ws",
    "trade_print",
    "orderbook_snapshot",
}
EVENT_DRIVEN_V5_DATASETS = readers.EVENT_DRIVEN_V5_DATASET_STATUSES
EVENT_DRIVEN_OKX_READONLY_DATASETS = readers.EVENT_DRIVEN_OKX_READONLY_DATASET_STATUSES
EVENT_DRIVEN_OK_STATUSES = readers.EVENT_DRIVEN_OK_STATUSES
DERIVED_LATEST_SOURCE_DATASETS = {
    "v5_bnb_paper_strategy_daily_latest": "v5_bnb_paper_strategy_daily",
}
DERIVED_LATEST_SOURCE_CURRENT_STATUS = "derived_latest_source_current"
SECTION_DATASETS = {
    "market": ["market_bar", "trade_print", "orderbook_snapshot", "okx_public_ws"],
    "features": ["feature_value", "feature_coverage_daily", "feature_anomaly_daily"],
    "costs": [
        "cost_bucket_daily",
        "cost_health_daily",
        "okx_private_readonly_fills",
        "okx_private_readonly_bills",
        "v5_fill_bill_cost_reconciliation",
    ],
    "research": [
        "alpha_evidence",
        "factor_definition",
        "factor_value",
        "factor_evidence",
        "factor_candidate",
        "factor_correlation_daily",
        "alpha_discovery_board",
        "strategy_evidence",
        "strategy_evidence_sample",
        "strategy_evidence_quality",
        "research_portfolio_status",
        "strategy_opportunity_advisory",
        "paper_strategy_proposal",
        "paper_strategy_proposals_current",
        "paper_strategy_trackers_current",
        "paper_strategy_migration_audit",
        "paper_strategy_identity_conflict",
        "paper_cohort_manifest",
        "strategy_cost_trust",
        "trade_opportunity_event",
        "trade_opportunity_label",
        "trade_level_similarity_outcome",
        "trade_level_judgment",
        "trade_level_bucket_policy",
        "trade_level_opportunity_queue",
        "quant_lab_false_block_audit",
        "quant_lab_decision_regret",
        "second_stage_alpha_factory_sample",
        "second_stage_alpha_factory_summary",
        "expanded_relative_strength_decision_sample",
        "alpha_factory_template_registry",
        "alpha_factory_candidate",
        "alpha_factory_result",
        "alpha_factory_promotion_queue",
        "market_regime_daily",
        "strategy_regime_matrix",
        "regime_strategy_advisory",
        "v5_missed_opportunity_audit",
        "v5_risk_on_multi_buy_shadow",
        "risk_on_multi_buy_shadow",
        "expanded_universe_candidate",
        "expanded_universe_quality",
        "expanded_universe_candidate_event",
        "expanded_universe_candidate_label",
        "expanded_universe_promotion_queue",
        "expanded_universe_candidate_maturity",
        "expanded_universe_watchlist",
        "expanded_crypto_universe_shadow",
        "symbol_quality_score",
        "expanded_crypto_candidate_outcomes_by_symbol",
        "expanded_crypto_recommendations",
        "paper_strategy_runs",
        "paper_strategy_daily",
        "paper_strategy_registry",
        "paper_strategy_registry_current",
        "paper_strategy_registry_history",
        "paper_strategy_promotion_gate",
        "paper_slippage_coverage",
        "sol_protect_paper_loss_attribution",
        "sol_protect_paper_loss_summary",
        "v5_missed_low_audit",
        "v5_missed_low_by_symbol",
        "v5_missed_low_by_entry_reason",
        "v5_late_entry_chase_shadow",
        "v5_late_entry_chase_threshold_advisory",
        "v5_late_entry_chase_threshold_by_symbol",
        "v5_pullback_reversal_shadow",
        "v5_pullback_reversal_readiness",
        "v5_pullback_reversal_rule_comparison",
        "btc_probe_exit_policy_review",
        "btc_probe_exit_policy_summary",
        "bnb_swing_exit_policy_review",
        "bnb_exit_policy_v5_vs_quant_lab_consistency",
        "bnb_swing_exit_policy_summary",
        "exit_policy_review_sample",
        "exit_policy_review_summary",
        "v5_entry_quality_advisory",
        "v5_entry_quality_history_late_entry_chase_threshold_sensitivity",
        "v5_entry_quality_history_pullback_by_symbol",
        "v5_entry_quality_history_pullback_by_regime",
        "v5_entry_quality_history_pullback_by_horizon",
        "v5_entry_quality_history_anti_leakage_check",
        "v5_entry_quality_history_metrics",
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
        "v5_cost_probe_p3_preflight",
        "v5_cost_probe_live_execution_status",
        "v5_cost_probe_order_event",
        "v5_cost_probe_roundtrip_event",
        "v5_fill_bill_cost_reconciliation",
        "v5_trade_opportunity_funnel",
        "v5_paper_strategy_run",
        "v5_paper_strategy_exit_quality",
        "v5_paper_strategy_proposal_ack",
        "v5_paper_strategy_daily",
        "v5_paper_strategy_registry",
        "v5_paper_strategy_state",
        "v5_paper_strategy_signal",
        "v5_paper_strategy_quote_coverage",
        "v5_paper_strategy_cost_evidence",
        "v5_paper_strategy_error",
        "v5_paper_strategy_restart_recovery",
        "v5_quant_lab_contract_status",
        "v5_candidate_event",
        "v5_btc_probe_entry_quality_audit",
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
    "reports/cost_bootstrap_readiness.csv",
    "reports/cost_probe_fill_bill_match.csv",
    "reports/cost_probe_cost_disagreement.csv",
    "reports/live_universe_cost_coverage.csv",
    "research/alpha_evidence.csv",
    "research/strategy_evidence.csv",
    "research/alt_impulse_shadow_by_regime.csv",
    "research/alt_impulse_shadow_by_symbol_regime_horizon.csv",
    "research/strategy_evidence_samples.csv",
    "research/strategy_evidence_quality.csv",
    "reports/research_portfolio_status.csv",
    "reports/research_portfolio_summary.md",
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
    "v5/v5_cost_probe_p3_preflight.csv",
    "v5/v5_cost_probe_live_execution_status.csv",
    "v5/v5_cost_probe_live_execution_status_canonical.csv",
    "v5/v5_cost_probe_order_events.csv",
    "v5/v5_cost_probe_roundtrip_events.csv",
    "v5/v5_cost_probe_roundtrip_canonical.csv",
    "v5/v5_fill_bill_cost_reconciliation.csv",
    "v5/v5_paper_strategy_proposal_ack.csv",
    "v5/v5_paper_strategy_registry.csv",
    "v5/v5_paper_strategy_state.csv",
    "v5/v5_paper_strategy_signals.csv",
    "v5/v5_paper_strategy_runs.csv",
    "v5/v5_paper_strategy_exit_quality.csv",
    "v5/v5_trade_opportunity_funnel.csv",
    "v5/v5_paper_strategy_daily.csv",
    "v5/v5_paper_strategy_quote_coverage.csv",
    "v5/v5_paper_strategy_cost_evidence.csv",
    "v5/v5_paper_strategy_errors.csv",
    "v5/v5_paper_strategy_restart_recovery.csv",
    "v5/v5_quant_lab_contract_status.csv",
    "v5/v5_candidate_events.csv",
    "v5/v5_btc_probe_entry_quality_audit.csv",
    "v5/v5_candidate_labels.csv",
    "v5/v5_candidate_quality.csv",
    "v5/v5_candidate_outcome_summary.csv",
    "v5/v5_expanded_universe_advisory_reader.csv",
    "v5/v5_expanded_universe_paper_runs.csv",
    "v5/v5_expanded_universe_paper_daily.csv",
    "charts/market_close.png",
    "charts/market_returns.png",
    "charts/gate_status_counts.png",
    "charts/cost_distribution.png",
    "charts/v5_health_summary.png",
    "reports/alpha_discovery_board.csv",
    "reports/strategy_evidence_summary.md",
    "reports/second_stage_alpha_factory_summary.csv",
    "reports/second_stage_alpha_factory_samples.csv",
    "reports/expanded_relative_strength_decision_samples.csv",
    "reports/alpha_factory_template_registry.csv",
    "reports/alpha_factory_candidates.csv",
    "reports/alpha_factory_results.csv",
    "reports/alpha_factory_promotion_queue.csv",
    "reports/alpha_factory_daily.md",
    "reports/factor_definitions.csv",
    "reports/factor_evidence.csv",
    "reports/factor_candidates.csv",
    "reports/factor_correlation_daily.csv",
    "reports/factor_dedupe_decision.csv",
    "reports/factor_family_leaderboard.csv",
    "reports/factor_paper_review_queue.csv",
    "reports/composite_factor_candidates.csv",
    "reports/factor_regime_effectiveness.csv",
    "reports/factor_forward_validation.csv",
    "reports/factor_forward_validation.md",
    "reports/factor_strategy_bridge_candidates.csv",
    "reports/factor_factory_summary.md",
    "reports/candidate_kill_list.csv",
    "reports/candidate_shadow_watchlist.csv",
    "reports/candidate_paper_ready.csv",
    "reports/historical_label_threshold_ready.csv",
    "reports/paper_strategy_proposals.csv",
    "reports/paper_strategy_proposal_ack.csv",
    "reports/paper_strategy_proposals_current.csv",
    "reports/paper_strategy_trackers_current.csv",
    "reports/paper_strategy_registry_current.csv",
    "reports/paper_strategy_registry_history.csv",
    "reports/paper_strategy_ack_current.csv",
    "reports/paper_strategy_ack_history.csv",
    "reports/paper_strategy_identity_conflict.csv",
    "reports/paper_strategy_identity_conflict.md",
    "reports/paper_cohort_manifest.json",
    "reports/paper_cohort_status.md",
    "reports/api_auth_error_timeline.csv",
    "reports/api_auth_client_summary.csv",
    "reports/paper_runtime_freshness.csv",
    "reports/paper_proposal_propagation_status.csv",
    "reports/system_acceptance_complete_status.json",
    "reports/post_fix_funnel_attribution.json",
    "reports/strategy_opportunity_advisory.csv",
    "reports/v5_opportunity_event.csv",
    "reports/v5_opportunity_label.csv",
    "reports/trade_opportunity_event.csv",
    "reports/trade_opportunity_label.csv",
    "reports/trade_level_similarity_outcome.csv",
    "reports/trade_level_judgment.csv",
    "reports/trade_level_bucket_policy.csv",
    "reports/trade_level_opportunity_queue.csv",
    "reports/quant_lab_false_block_audit.csv",
    "reports/v5_trade_learning_sample.csv",
    "reports/v5_trade_outcome_attribution.csv",
    "reports/quant_lab_opportunity_cost_event.csv",
    "reports/quant_lab_opportunity_cost_daily.csv",
    "reports/opportunity_cost_by_bucket.csv",
    "reports/gate_candidate_level_effectiveness.csv",
    "reports/gate_independent_event_effectiveness.csv",
    "reports/gate_paper_trade_effectiveness.csv",
    "reports/gate_live_trade_effectiveness.csv",
    "reports/duplicate_event_report.csv",
    "reports/duplicate_factor_report.csv",
    "reports/strategy_cost_trust.csv",
    "reports/quant_lab_decision_regret.csv",
    "reports/api_latency_summary.csv",
    "reports/api_latency_summary.md",
    "reports/api_error_summary.csv",
    f"reports/{FALLBACK_RATE_BREAKDOWN_CSV}",
    "reports/github_ci_status.csv",
    "reports/v5_bundle_sync_diagnostics.csv",
    "reports/v5_local_live_vs_quant_lab_shadow.csv",
    "reports/missed_opportunity_audit.csv",
    "reports/missed_opportunity_summary.md",
    "reports/bnb_missed_opportunity_samples.csv",
    "reports/bnb_missed_opportunity_summary.md",
    "reports/final_score_vs_alpha6_conflict.csv",
    "reports/final_score_vs_alpha6_conflict_summary.md",
    "reports/bnb_strong_alpha6_bypass_shadow.csv",
    "reports/bnb_strong_alpha6_bypass_summary.md",
    "reports/post_impulse_overextension_shadow.csv",
    "reports/post_impulse_overextension_no_trigger_reasons.csv",
    "reports/late_breakout_failure_shadow.csv",
    "reports/late_breakout_failure_protect_shadow.csv",
    "reports/bottom_zone_reversal_shadow.csv",
    "reports/bottom_zone_reversal_no_trigger_reasons.csv",
    "reports/bottom_zone_reversal_summary.md",
    "reports/bottom_zone_probe_paper_readiness.csv",
    "reports/fast_microstructure_features.csv",
    "reports/fast_microstructure_forward_test.csv",
    "reports/fast_microstructure_strategy_candidates.csv",
    "reports/fast_microstructure_strategy_review.csv",
    "reports/fast_microstructure_forward_summary.md",
    "reports/market_pressure_score.csv",
    "reports/market_pressure_summary.md",
    "reports/system_acceptance_dashboard.csv",
    "reports/system_acceptance_dashboard.md",
    "reports/backtest_label_summary.csv",
    "reports/backtest_label_summary.md",
    "reports/v5_decision_replay_trades.csv",
    "reports/v5_decision_replay_equity.csv",
    "reports/v5_decision_replay_summary.md",
    "reports/backtest_regime_breakdown.csv",
    "reports/bottom_zone_backtest.csv",
    "reports/bottom_zone_backtest_summary.md",
    "reports/research_promotion_decision.csv",
    "reports/research_promotion_decision.md",
    "reports/backtest_vs_paper_consistency.csv",
    "reports/backtest_vs_paper_consistency.md",
    "reports/backtest_vs_paper_gap_report.csv",
    "reports/backtest_vs_paper_gap_report.md",
    "reports/risk_on_multi_buy_shadow.csv",
    "reports/risk_on_multi_buy_summary.md",
    "reports/bnb_negative_expectancy_attribution.csv",
    "reports/negative_expectancy_attribution.csv",
    "reports/negative_expectancy_attribution_summary.md",
    "reports/market_regime_daily.csv",
    "reports/strategy_regime_matrix.csv",
    "reports/regime_strategy_advisory.csv",
    "reports/expanded_universe_candidates.csv",
    "reports/expanded_universe_quality.csv",
    "reports/expanded_universe_daily.md",
    "reports/expanded_universe_watchlist.csv",
    "reports/expanded_universe_candidate_maturity.csv",
    "reports/expanded_universe_promotion_summary.md",
    "reports/expanded_universe_promotion_queue.csv",
    "reports/expanded_universe_kill_list.csv",
    "reports/expanded_universe_replacement_candidates.csv",
    "reports/expanded_universe_strategy_evidence.csv",
    "reports/expanded_crypto_universe_shadow.csv",
    "reports/symbol_quality_score.csv",
    "reports/expanded_crypto_candidate_outcomes_by_symbol.csv",
    "reports/expanded_crypto_recommendations.json",
    "reports/strategy_level_dashboard.csv",
    "reports/paper_strategy_runs.csv",
    "reports/paper_strategy_daily.csv",
    "reports/paper_strategy_registry.csv",
    "reports/paper_strategy_promotion_gate.csv",
    "reports/paper_strategy_migration_audit.csv",
    "reports/paper_strategy_summary.md",
    "reports/bnb_paper_strategy_runs.csv",
    "reports/bnb_paper_strategy_daily.csv",
    "reports/bnb_paper_strategy_summary.md",
    "reports/v5_quant_lab_consistency_dashboard.md",
    "reports/paper_slippage_coverage.csv",
    "reports/sol_protect_paper_loss_attribution.csv",
    "reports/sol_protect_paper_loss_summary.md",
    "reports/missed_low_audit.csv",
    "reports/missed_low_by_symbol.csv",
    "reports/missed_low_by_entry_reason.csv",
    "reports/late_entry_chase_shadow.csv",
    "reports/late_entry_chase_threshold_advisory.json",
    "reports/late_entry_chase_threshold_sensitivity.csv",
    "reports/late_entry_chase_threshold_sensitivity_by_symbol.csv",
    "reports/late_entry_chase_threshold_advisory_by_symbol.json",
    "reports/threshold_advisory_by_symbol.json",
    "reports/pullback_reversal_shadow_outcomes.csv",
    "reports/old_v1_vs_v2_comparison.csv",
    "reports/pullback_reversal_v2_by_symbol.csv",
    "reports/pullback_reversal_v2_readiness.json",
    "reports/btc_probe_exit_policy_review.csv",
    "reports/btc_probe_exit_policy_summary.md",
    "reports/bnb_swing_exit_policy_review.csv",
    "reports/bnb_exit_policy_v5_vs_quant_lab_consistency.csv",
    "reports/bnb_swing_exit_policy_summary.md",
    "reports/exit_policy_review.csv",
    "reports/exit_policy_summary.md",
    "reports/pullback_reversal_by_symbol.csv",
    "reports/pullback_reversal_by_regime.csv",
    "reports/pullback_reversal_by_horizon.csv",
    "reports/pullback_reversal_rule_comparison.csv",
    "reports/pullback_reversal_readiness.json",
    "reports/anti_leakage_check.csv",
    "reports/entry_quality_summary.md",
    "reports/entry_quality_historical_summary.md",
    "reports/entry_quality_historical_metrics.json",
    f"reports/{ENFORCE_READINESS_JSON}",
    f"reports/{ENFORCE_READINESS_CSV}",
]

CSV_SCHEMAS: dict[str, list[str]] = {
    **BACKTEST_CSV_SCHEMAS,
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
        "cost_probe_fill_count",
        "strategy_live_fill_count",
        "private_fill_count",
        "sample_origin_mix",
        "eligible_for_live_cost_coverage",
    ],
    "costs/cost_health_daily.csv": [
        "day",
        "status",
        "cost_model_version",
        "actual_rows",
        "mixed_rows",
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
    f"reports/{FALLBACK_RATE_BREAKDOWN_CSV}": FALLBACK_RATE_BREAKDOWN_COLUMNS,
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
    "reports/factor_definitions.csv": [
        "factor_id",
        "factor_name",
        "factor_family",
        "factor_version",
        "description",
        "feature_set",
        "feature_version",
        "timeframe",
        "input_features_json",
        "template",
        "params_json",
        "expression_json",
        "expression_hash",
        "factor_hash",
        "canonical_factor_id",
        "factor_formula_hash",
        "formula_hash",
        "duplicate_of",
        "correlation_cluster_id",
        "effective_independence_weight",
        "independence_weight",
        "status",
        "lookback_bars",
        "availability_lag_bars",
        "warmup_bars",
        "required_bars",
        "causal",
        "normalization",
        "owner",
        "direction",
        "min_cross_section",
        "clip_abs",
        "enabled",
        "tags_json",
        "created_at",
        "source",
    ],
    "reports/factor_evidence.csv": [
        "as_of_date",
        "factor_id",
        "factor_name",
        "factor_family",
        "factor_version",
        "timeframe",
        "factor_hash",
        "canonical_factor_id",
        "formula_hash",
        "independence_weight",
        "horizon_bars",
        "decision_delay_bars",
        "sample_count",
        "valid_sample_count",
        "coverage",
        "ic_mean",
        "ic_tstat",
        "rank_ic_mean",
        "rank_ic_tstat",
        "long_only_mean_bps",
        "long_short_mean_bps",
        "win_rate",
        "turnover",
        "max_drawdown",
        "edge_cost_ratio",
        "decision",
        "score",
        "independence_adjusted_score",
        "reasons_json",
        "warnings_json",
        "start_ts",
        "end_ts",
        "created_at",
        "source",
    ],
    "reports/factor_candidates.csv": [
        "as_of_date",
        "factor_id",
        "factor_name",
        "factor_family",
        "factor_version",
        "timeframe",
        "factor_hash",
        "canonical_factor_id",
        "formula_hash",
        "independence_weight",
        "best_horizon_bars",
        "tested_horizon_count",
        "best_score",
        "avg_score",
        "independence_adjusted_best_score",
        "independence_adjusted_avg_score",
        "best_rank_ic_mean",
        "best_rank_ic_tstat",
        "best_long_short_mean_bps",
        "candidate_state",
        "recommended_action",
        "promotion_block_reasons_json",
        "manual_review_required",
        "created_at",
        "source",
    ],
    "reports/factor_correlation_daily.csv": [
        "as_of_date",
        "factor_id_left",
        "factor_id_right",
        "factor_version",
        "timeframe",
        "sample_count",
        "correlation",
        "created_at",
        "source",
    ],
    "reports/factor_dedupe_decision.csv": FACTOR_DEDUPE_DECISION_FIELDS,
    "reports/factor_family_leaderboard.csv": FACTOR_FAMILY_LEADERBOARD_FIELDS,
    "reports/factor_paper_review_queue.csv": FACTOR_PAPER_REVIEW_QUEUE_FIELDS,
    "reports/composite_factor_candidates.csv": COMPOSITE_FACTOR_CANDIDATE_FIELDS,
    "reports/factor_regime_effectiveness.csv": FACTOR_REGIME_EFFECTIVENESS_FIELDS,
    "reports/factor_forward_validation.csv": FACTOR_FORWARD_VALIDATION_FIELDS,
    "reports/factor_strategy_bridge_candidates.csv": FACTOR_STRATEGY_BRIDGE_CANDIDATE_FIELDS,
    "reports/live_universe_cost_coverage.csv": LIVE_UNIVERSE_COST_COVERAGE_FIELDS,
    "reports/cost_bootstrap_readiness.csv": COST_BOOTSTRAP_READINESS_FIELDS,
    "reports/cost_probe_fill_bill_match.csv": COST_PROBE_FILL_BILL_MATCH_FIELDS,
    "reports/cost_probe_cost_disagreement.csv": COST_PROBE_COST_DISAGREEMENT_FIELDS,
    "market/orderbook_spread.csv": ["symbol", "channel", "ts", "spread_bps"],
    "market/trade_activity.csv": ["symbol", "trade_count", "size_sum", "latest_trade_ts"],
    "reports/api_latency_summary.csv": [
        "endpoint",
        "count",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "success_p95_ms",
        "error_p95_ms",
        "p99_ms",
        "max_ms",
        "cache_hit_rate",
        "avg_rows_returned",
        "avg_response_bytes",
        "avg_lake_scan_ms",
        "avg_serialize_ms",
        "avg_source_signature_ms",
        "response_cache_hit_rate",
        "dependency_meta_missing_count",
        "dependency_meta_missing_rate",
        "error_count",
        "auth_error_count",
        "auth_error_rate",
    ],
    "reports/api_error_summary.csv": [
        "endpoint",
        "status_code",
        "auth_result",
        "client_id",
        "client_host",
        "user_agent",
        "error_count",
        "latest_error_ts",
        "error_rate",
    ],
    "reports/github_ci_status.csv": [
        "component",
        "repo",
        "commit_sha",
        "workflow_name",
        "workflow_run_id",
        "workflow_status",
        "workflow_conclusion",
        "event",
        "head_sha",
        "created_at",
        "updated_at",
        "html_url",
        "observed_at",
        "source",
        "error",
    ],
    "reports/v5_bundle_sync_diagnostics.csv": [
        "generated_at",
        "sync_attempted",
        "sync_success",
        "latest_local_bundle_ts",
        "latest_remote_bundle_ts",
        "latest_uploaded_bundle_ts",
        "latest_ingested_bundle_ts",
        "selected_v5_bundle_path",
        "selected_v5_bundle_ts",
        "selected_v5_bundle_sha",
        "selected_v5_bundle_sha256",
        "expected_v5_bundle_sha256",
        "expected_v5_bundle_matched",
        "stale_seconds",
        "why_stale",
        "failure_reason",
        "authoritative_snapshot",
        "stale_v5_bundle",
    ],
    "reports/system_acceptance_dashboard.csv": SYSTEM_ACCEPTANCE_FIELDS,
    "reports/post_impulse_overextension_no_trigger_reasons.csv": NO_TRIGGER_REASON_FIELDS,
    "reports/bottom_zone_reversal_no_trigger_reasons.csv": NO_TRIGGER_REASON_FIELDS,
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
        "role",
        "baseline_status",
        "not_live_eligible",
        "not_global_strategy_gate",
    ],
    "research/gate_decisions.csv": [
        "alpha_id",
        "version",
        "gate_version",
        "status",
        "passed",
        "role",
        "baseline_status",
        "not_live_eligible",
        "not_global_strategy_gate",
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
    "research/alt_impulse_shadow_by_regime.csv": [
        "as_of_date",
        "strategy_candidate",
        "candidate_name",
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
    "research/alt_impulse_shadow_by_symbol_regime_horizon.csv": [
        "as_of_date",
        "strategy_candidate",
        "candidate_name",
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
        "rank_score_bps",
        "rank_lookback_hours",
        "selected_rank",
        "top_k",
        "selection_reason",
        "anti_leakage_check",
        "futures_data_available",
        "funding_available",
        "funding_cost_bps",
        "mark_price_source",
        "liquidation_buffer_pct",
        "proxy_warning",
        "expected_edge_bps",
        "required_edge_bps",
        "alpha6_score",
        "alpha6_side",
        "protect_level",
        "risk_level",
        "btc_trend_state",
        "broad_market_positive_count",
        "funding_state",
        "volatility_bucket",
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
    "reports/second_stage_alpha_factory_summary.csv": [
        "strategy",
        "evidence_version",
        "as_of_date",
        "strategy_candidate",
        "candidate_name",
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
    "reports/second_stage_alpha_factory_samples.csv": [
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
        "rank_score_bps",
        "rank_lookback_hours",
        "selected_rank",
        "top_k",
        "selection_reason",
        "anti_leakage_check",
        "futures_data_available",
        "funding_available",
        "funding_cost_bps",
        "mark_price_source",
        "liquidation_buffer_pct",
        "proxy_warning",
        "expected_edge_bps",
        "required_edge_bps",
        "alpha6_score",
        "alpha6_side",
        "protect_level",
        "risk_level",
        "btc_trend_state",
        "broad_market_positive_count",
        "funding_state",
        "volatility_bucket",
        "source_path_inside_bundle",
        "source_event_key",
        "source_bundle_ts",
        "created_at",
        "source",
    ],
    "reports/alpha_factory_template_registry.csv": [
        "template_id",
        "template_family",
        "enabled",
        "description",
        "universe_scope",
        "allowed_regimes",
        "parameter_space_json",
        "max_candidates_per_day",
        "safety_mode",
        "created_at",
        "source",
    ],
    "reports/alpha_factory_candidates.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "template_name",
        "candidate_id",
        "strategy_candidate",
        "symbol",
        "regime_state",
        "horizon_hours",
        "parameter_json",
        "whitelist_json",
        "blacklist_json",
        "source_dataset",
        "candidate_state",
        "max_live_notional_usdt",
        "safety_mode",
        "source",
    ],
    "reports/alpha_factory_results.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "template_name",
        "candidate_id",
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
        "avg_mae_bps",
        "avg_mfe_bps",
        "cost_quality_score",
        "recent_sample_sufficient",
        "paper_ready_block_reasons",
        "train_metrics_json",
        "validation_metrics_json",
        "recent_7d_metrics_json",
        "stability_score",
        "tail_loss_penalty",
        "recent_degradation_penalty",
        "validation_failure_penalty",
        "alpha_factory_score",
        "decision",
        "decision_reasons",
        "recommended_mode",
        "start_ts",
        "end_ts",
        "max_live_notional_usdt",
        "source",
    ],
    "reports/alpha_factory_promotion_queue.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "template_name",
        "candidate_id",
        "strategy_candidate",
        "symbol",
        "horizon_hours",
        "promotion_state",
        "recommended_mode",
        "action",
        "reasons",
        "max_live_notional_usdt",
        "manual_live_approval_required",
        "source",
    ],
    "reports/research_portfolio_status.csv": [
        "schema_version",
        "as_of_date",
        "research_id",
        "module",
        "strategy_candidate",
        "status",
        "action",
        "reason",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "win_rate",
        "p25_net_bps",
        "paper_days",
        "entry_day_count",
        "paper_negative_streak",
        "downgrade_reason",
        "latest_paper_trend",
        "priority",
        "cost_source_mix",
        "last_review_date",
        "next_review_date",
        "recommended_new_research_slots",
        "freed_research_slots",
        "active_research_count",
        "killed_research_count",
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
    "reports/historical_label_threshold_ready.csv": [
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
        "contract_version",
        "proposal_hash",
        "strategy_id",
        "strategy_version",
        "strategy_family",
        "timeframe",
        "direction",
        "entry_rule",
        "exit_rule",
        "max_holding_bars",
        "min_holding_bars",
        "cooldown_bars",
        "signal_confirmation_bars",
        "cost_quantile",
        "minimum_expected_edge_bps",
        "paper_notional_usdt",
        "paper_only",
        "live_order_effect",
        "max_live_notional_usdt",
        "expires_at",
        "source_pack_sha256",
        "source_dataset_versions",
        "required_market_fields",
        "required_cost_trust_level",
        "lifecycle_state",
        "lifecycle_reason",
        "blocked_reasons",
        "next_required_actions",
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
    "reports/paper_strategy_proposal_ack.csv": [
        "proposal_id",
        "proposal_hash",
        "paper_tracker_id",
        "accepted",
        "accepted_at",
        "recommended_mode",
        "symbol",
        "strategy_candidate",
        "suggested_horizon",
        "proposal_source",
        "reject_reason",
        "live_order_effect",
        "source_pack_sha256",
        "source_v5_bundle_sha256",
    ],
    "reports/paper_strategy_ack_current.csv": [
        "proposal_id",
        "proposal_hash",
        "paper_tracker_id",
        "accepted",
        "accepted_at",
        "recommended_mode",
        "symbol",
        "strategy_candidate",
        "suggested_horizon",
        "proposal_source",
        "reject_reason",
        "live_order_effect",
        "source_pack_sha256",
        "source_v5_bundle_sha256",
    ],
    "reports/paper_strategy_ack_history.csv": [
        "proposal_id",
        "proposal_hash",
        "paper_tracker_id",
        "accepted",
        "accepted_at",
        "recommended_mode",
        "symbol",
        "strategy_candidate",
        "suggested_horizon",
        "proposal_source",
        "reject_reason",
        "live_order_effect",
        "source_pack_sha256",
        "source_v5_bundle_sha256",
    ],
    "reports/strategy_opportunity_advisory.csv": [
        "as_of_ts",
        "generated_at",
        "expires_at",
        "contract_version",
        "schema_version",
        "quant_lab_git_commit",
        "source_version",
        "would_block_if_enabled",
        "would_enter",
        "no_sample_reason",
        "strategy_id",
        "symbol",
        "v5_symbol",
        "strategy_candidate",
        "decision",
        "recommended_mode",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "cost_quality",
        "source_module",
        "template_family",
        "candidate_id",
        "promotion_state",
        "alpha_factory_score",
        "universe_type",
        "expanded_universe_maturity_state",
        "cost_quality_score",
        "paper_ready_block_reasons",
        "advisory_intent",
        "paper_days",
        "entry_day_count",
        "paper_pnl_observed_count",
        "slippage_coverage",
        "live_block_reasons",
        "max_paper_notional_usdt",
        "max_live_notional_usdt",
        "live_order_effect",
    ],
    "reports/v5_opportunity_event.csv": list(TRADE_OPPORTUNITY_EVENT_SCHEMA),
    "reports/v5_opportunity_label.csv": list(TRADE_OPPORTUNITY_LABEL_SCHEMA),
    "reports/trade_opportunity_event.csv": list(TRADE_OPPORTUNITY_EVENT_SCHEMA),
    "reports/trade_opportunity_label.csv": list(TRADE_OPPORTUNITY_LABEL_SCHEMA),
    "reports/trade_level_similarity_outcome.csv": list(TRADE_LEVEL_SIMILARITY_SCHEMA),
    "reports/trade_level_judgment.csv": list(TRADE_LEVEL_JUDGMENT_SCHEMA),
    "reports/trade_level_bucket_policy.csv": list(TRADE_LEVEL_BUCKET_POLICY_SCHEMA),
    "reports/trade_level_opportunity_queue.csv": list(TRADE_LEVEL_OPPORTUNITY_QUEUE_SCHEMA),
    "reports/quant_lab_false_block_audit.csv": list(FALSE_BLOCK_AUDIT_SCHEMA),
    "reports/v5_trade_learning_sample.csv": list(V5_TRADE_LEARNING_SAMPLE_SCHEMA),
    "reports/v5_trade_outcome_attribution.csv": list(V5_TRADE_OUTCOME_ATTRIBUTION_SCHEMA),
    "reports/quant_lab_opportunity_cost_event.csv": list(OPPORTUNITY_COST_EVENT_SCHEMA),
    "reports/quant_lab_opportunity_cost_daily.csv": list(OPPORTUNITY_COST_DAILY_SCHEMA),
    "reports/opportunity_cost_by_bucket.csv": list(OPPORTUNITY_COST_BY_BUCKET_SCHEMA),
    "reports/quant_lab_decision_regret.csv": list(DECISION_REGRET_SCHEMA),
    "reports/v5_local_live_vs_quant_lab_shadow.csv": [
        "generated_at",
        "schema_version",
        "trade_id",
        "order_id",
        "run_id",
        "entry_ts",
        "symbol",
        "side",
        "action",
        "entry_price",
        "matching_strategy_candidate",
        "matching_decision",
        "matching_recommended_mode",
        "matching_advisory_intent",
        "quant_lab_is_live_commander",
        "quant_lab_would_block_if_enabled",
        "quant_lab_block_reasons",
        "pnl_4h_bps",
        "pnl_8h_bps",
        "pnl_24h_bps",
        "block_outcome_4h",
        "block_outcome_8h",
        "block_outcome_24h",
        "block_outcome_summary",
        "source",
    ],
    "reports/missed_opportunity_audit.csv": [
        "generated_at",
        "schema_version",
        "run_id",
        "ts_utc",
        "symbol",
        "regime_state",
        "current_regime",
        "broad_market_positive_count",
        "final_score",
        "alpha6_side",
        "expected_edge_bps",
        "required_edge_bps",
        "v5_final_decision",
        "v5_block_reason",
        "actual_trade_opened",
        "quant_lab_would_block_live",
        "quant_lab_recommended_mode",
        "quant_lab_decision",
        "would_have_been_missed_by_quant_lab",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_24h_net_bps",
        "outcome_if_blocked",
        "source",
    ],
    "reports/bnb_missed_opportunity_samples.csv": [
        "run_id",
        "ts_utc",
        "entry_close",
        "alpha6_score",
        "f3",
        "f4",
        "f5",
        "final_score",
        "final_decision",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_12h_net_bps",
        "future_24h_net_bps",
        "missed_profit_flag",
    ],
    "reports/final_score_vs_alpha6_conflict.csv": [
        "run_id",
        "ts_utc",
        "symbol",
        "final_score",
        "alpha6_score",
        "alpha6_side",
        "f3_vol_adj_ret",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "expected_edge_bps",
        "required_edge_bps",
        "cost_gate_verified",
        "final_decision",
        "block_reason",
        "no_signal_reason",
        "negative_expectancy_net_bps",
        "negative_expectancy_fast_fail_net_bps",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_12h_net_bps",
        "future_24h_net_bps",
        "max_future_net_bps",
        "best_future_horizon_hours",
        "material_profit_flag",
        "label_4h_status",
        "label_8h_status",
        "label_12h_status",
        "label_24h_status",
        "any_label_complete",
        "all_labels_complete",
        "label_status",
        "missed_profit_flag",
    ],
    "reports/bnb_strong_alpha6_bypass_shadow.csv": [
        "run_id",
        "ts_utc",
        "would_bypass",
        "alpha6_score",
        "f3",
        "f4",
        "f5",
        "expected_edge_bps",
        "required_edge_bps",
        "final_score",
        "final_decision",
        "block_reason",
        "no_signal_reason",
        "negative_expectancy_blocked",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_12h_net_bps",
        "future_24h_net_bps",
        "max_future_net_bps",
        "best_future_horizon_hours",
        "material_profit_flag",
        "label_status",
        "outcome",
        "live_order_effect",
    ],
    "reports/post_impulse_overextension_shadow.csv": [
        "run_id",
        "ts_utc",
        "symbol",
        "alpha6_score",
        "alpha6_side",
        "f3_vol_adj_ret",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "entry_px",
        "return_24h_bps",
        "return_48h_bps",
        "overextension_reason",
        "why_not_triggered",
        "missing_field_reason",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_12h_net_bps",
        "max_future_net_bps",
        "worst_future_net_bps",
        "late_failure_flag",
        "response_action",
        "live_order_effect",
    ],
    "reports/late_breakout_failure_shadow.csv": [
        "run_id",
        "ts_utc",
        "symbol",
        "alpha6_score",
        "alpha6_side",
        "f3_vol_adj_ret",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "entry_px",
        "return_24h_bps",
        "return_48h_bps",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_12h_net_bps",
        "failure_horizon_hours",
        "failure_net_bps",
        "diagnosis",
        "response_action",
        "live_order_effect",
    ],
    "reports/late_breakout_failure_protect_shadow.csv": [
        "run_id",
        "ts_utc",
        "symbol",
        "alpha6_score",
        "alpha6_side",
        "f3_vol_adj_ret",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "entry_px",
        "return_24h_bps",
        "return_48h_bps",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_12h_net_bps",
        "failure_horizon_hours",
        "failure_net_bps",
        "diagnosis",
        "response_action",
        "live_order_effect",
    ],
    "reports/bottom_zone_reversal_shadow.csv": BOTTOM_ZONE_FIELDS,
    "reports/bottom_zone_probe_paper_readiness.csv": [
        "generated_at",
        "as_of_date",
        "strategy_id",
        "strategy_candidate",
        "symbol",
        "paper_days",
        "paper_entries",
        "avg_pnl_bps",
        "p25_pnl_bps",
        "required_paper_days",
        "required_paper_entries",
        "required_avg_pnl_bps",
        "required_p25_pnl_bps",
        "readiness_status",
        "blocking_reasons",
        "recommended_stage",
        "live_order_effect",
    ],
    "reports/fast_microstructure_features.csv": FAST_MICROSTRUCTURE_FIELDS,
    "reports/fast_microstructure_forward_test.csv": FAST_MICROSTRUCTURE_FORWARD_TEST_FIELDS,
    "reports/fast_microstructure_strategy_candidates.csv": (
        FAST_MICROSTRUCTURE_STRATEGY_CANDIDATE_FIELDS
    ),
    "reports/fast_microstructure_strategy_review.csv": (FAST_MICROSTRUCTURE_STRATEGY_REVIEW_FIELDS),
    "reports/market_pressure_score.csv": MARKET_PRESSURE_FIELDS,
    "reports/risk_on_multi_buy_shadow.csv": [
        "generated_at",
        "schema_version",
        "run_id",
        "decision_ts",
        "current_regime",
        "regime_source",
        "candidate_buy_count",
        "market_regime_daily_state",
        "v5_candidate_regime_state",
        "trigger_reason",
        "broad_market_positive_count",
        "btc_24h_return_bps",
        "strategy_candidate",
        "top_k",
        "selected_symbols",
        "selected_count",
        "would_buy_symbol",
        "would_buy",
        "would_size_usdt",
        "final_score",
        "expected_edge_bps",
        "required_edge_bps",
        "entry_px",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_24h_net_bps",
        "avg_portfolio_net_bps",
        "actual_v5_bought_symbols",
        "vs_actual_v5_net_bps",
        "missed_symbols",
        "recommended_mode",
        "max_live_notional_usdt",
        "source",
    ],
    "reports/bnb_negative_expectancy_attribution.csv": [
        "symbol",
        "cycle_index",
        "entry_ts",
        "exit_ts",
        "exit_reason",
        "exit_priority",
        "net_bps",
        "attribution",
        "entry_bad",
        "exit_bad",
        "min_hold_violation",
        "gave_back_profit",
        "trailing_too_early",
        "unknown",
        "exit_metadata_missing",
    ],
    "reports/negative_expectancy_attribution.csv": [
        "symbol",
        "cycle_index",
        "entry_ts",
        "exit_ts",
        "exit_reason",
        "exit_priority",
        "net_bps",
        "attribution",
        "entry_bad",
        "exit_bad",
        "min_hold_violation",
        "gave_back_profit",
        "trailing_too_early",
        "unknown",
        "exit_metadata_missing",
        "adjusted_entry_expectancy_bps",
        "raw_would_block",
        "adjusted_would_block",
        "would_unblock_if_adjusted",
        "block_attribution_conflict",
    ],
    "reports/market_regime_daily.csv": [
        "as_of_date",
        "current_regime",
        "active_regimes_json",
        "regime_confidence",
        "btc_24h_return_bps",
        "eth_24h_return_bps",
        "sol_24h_return_bps",
        "bnb_24h_return_bps",
        "broad_market_positive_count",
        "realized_vol_bps",
        "avg_spread_bps",
        "liquidity_thin",
        "created_at",
        "source",
        "schema_version",
    ],
    "reports/strategy_regime_matrix.csv": [
        "as_of_date",
        "strategy_candidate",
        "raw_strategy_candidate",
        "symbol",
        "regime_state",
        "canonical_regime",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "median_net_bps",
        "p25_net_bps",
        "win_rate",
        "mae_bps",
        "cost_source_mix",
        "decision",
        "decision_reasons",
        "created_at",
        "source",
        "schema_version",
    ],
    "reports/regime_strategy_advisory.csv": [
        "as_of_date",
        "current_regime",
        "active_regimes_json",
        "allowed_strategy_candidates",
        "blocked_strategy_candidates",
        "recommended_mode",
        "regime_confidence",
        "live_block_reasons",
        "created_at",
        "source",
        "schema_version",
    ],
    "reports/expanded_crypto_universe_shadow.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "rank",
        "symbol",
        "is_current_v5_symbol",
        "quote_volume_24h",
        "avg_spread_bps",
        "data_coverage",
        "btc_correlation",
        "quality_score",
        "recommendation",
        "blocking_reasons",
        "min_shadow_days_required",
        "notes",
        "source",
    ],
    "reports/symbol_quality_score.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "symbol",
        "quote_volume_24h",
        "avg_spread_bps",
        "min_notional_ok",
        "data_coverage",
        "avg_24h_net_bps",
        "avg_48h_net_bps",
        "win_rate_24h",
        "win_rate_48h",
        "f3_dominant_negative_score",
        "f4_confirmed_win_rate",
        "f5_confirmed_win_rate",
        "pullback_shadow_avg_24h",
        "late_chase_loss_rate",
        "negative_expectancy_bps",
        "btc_correlation",
        "quality_score",
        "recommendation",
        "blocking_reasons",
        "source",
    ],
    "reports/expanded_universe_candidates.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "symbol",
        "universe_type",
        "active_trading",
        "bar_coverage",
        "spread_bps_p75",
        "quote_volume_24h",
        "min_notional_ok",
        "candidate_state",
        "blocking_reasons",
        "source",
    ],
    "reports/expanded_universe_quality.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "symbol",
        "quote_volume_24h",
        "avg_spread_bps",
        "min_notional_ok",
        "data_coverage",
        "avg_24h_net_bps",
        "avg_48h_net_bps",
        "win_rate_24h",
        "win_rate_48h",
        "f3_dominant_negative_score",
        "f4_confirmed_win_rate",
        "f5_confirmed_win_rate",
        "pullback_shadow_avg_24h",
        "late_chase_loss_rate",
        "negative_expectancy_bps",
        "btc_correlation",
        "quality_score",
        "recommendation",
        "blocking_reasons",
        "source",
    ],
    "reports/expanded_universe_watchlist.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "watchlist_type",
        "symbol",
        "quality_score",
        "recommendation",
        "sample_count",
        "complete_sample_count",
        "positive_short_horizon_count",
        "best_short_horizon_hours",
        "best_short_avg_net_bps",
        "win_rate",
        "p25_net_bps",
        "maturity_state",
        "watch_reason",
        "source",
    ],
    "reports/expanded_universe_candidate_maturity.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "symbol",
        "strategy_candidate",
        "universe_type",
        "sample_count",
        "complete_sample_count",
        "positive_short_horizon_count",
        "positive_short_horizons",
        "best_short_horizon_hours",
        "best_short_avg_net_bps",
        "win_rate",
        "p25_net_bps",
        "maturity_state",
        "recommended_mode",
        "maturity_reasons",
        "cost_source_mix",
        "source",
    ],
    "reports/expanded_universe_promotion_queue.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "symbol",
        "strategy_candidate",
        "universe_type",
        "promotion_state",
        "recommended_mode",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "live_block_reasons",
        "replacement_target_candidate",
        "expansion_state",
        "min_shadow_days_required",
        "human_approval_required",
        "max_live_notional_usdt",
        "source",
    ],
    "reports/expanded_universe_kill_list.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "symbol",
        "strategy_candidate",
        "universe_type",
        "promotion_state",
        "recommended_mode",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "live_block_reasons",
        "replacement_target_candidate",
        "expansion_state",
        "min_shadow_days_required",
        "human_approval_required",
        "max_live_notional_usdt",
        "source",
    ],
    "reports/expanded_universe_replacement_candidates.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "symbol",
        "strategy_candidate",
        "universe_type",
        "promotion_state",
        "recommended_mode",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "live_block_reasons",
        "replacement_target_candidate",
        "expansion_state",
        "min_shadow_days_required",
        "human_approval_required",
        "max_live_notional_usdt",
        "source",
    ],
    "reports/expanded_universe_strategy_evidence.csv": [
        "strategy",
        "evidence_version",
        "as_of_date",
        "strategy_candidate",
        "candidate_name",
        "source_type",
        "symbol",
        "universe_type",
        "replacement_target_candidate",
        "expansion_state",
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
    "reports/expanded_relative_strength_decision_samples.csv": [
        "decision_ts",
        "symbol",
        "lookback_hours",
        "lookback_return_bps",
        "top_k",
        "selected_rank",
        "selected",
        "symbol_quality_score",
        "spread_bps",
        "btc_corr",
        "volume_24h_usdt",
        "selection_reason",
        "anti_leakage_check",
        "label_horizon_hours",
        "future_net_bps",
        "label_status",
    ],
    "reports/expanded_crypto_candidate_outcomes_by_symbol.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "symbol",
        "strategy_candidate",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "decision",
        "source",
    ],
    "reports/strategy_level_dashboard.csv": [
        "decision",
        "strategy_candidate",
        "symbol",
        "horizon_hours",
        "recommended_mode",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "live_block_reasons",
        "max_paper_notional_usdt",
        "max_live_notional_usdt",
        "as_of_ts",
    ],
    "reports/paper_strategy_runs.csv": [
        "as_of_date",
        "strategy_id",
        "proposal_id",
        "paper_tracker_id",
        "strategy_candidate",
        "run_id",
        "ts_utc",
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
        "final_decision",
        "no_sample_reason",
        "paper_disabled_by_research_portfolio",
        "paper_trigger_type",
        "paper_trigger_reason",
        "source_candidate_symbol",
        "source_candidate_id",
        "symbol_match_verified",
        "paper_source",
        "paper_count_scope",
        "risk_level",
        "alpha6_score",
        "alpha6_side",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "arrival_bid",
        "arrival_ask",
        "arrival_mid",
        "estimated_spread_bps",
        "expected_order_type",
        "estimated_fill_px",
        "cost_source",
        "paper_tracking_status",
        "tracking_stage",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "cost_source_mix",
        "live_block_reason",
        "required_paper_days",
        "required_slippage_coverage",
        "label_status",
        "source_path_inside_bundle",
        "bundle_ts",
        "ingest_ts",
        "created_at",
        "source",
        "schema_version",
    ],
    "reports/paper_strategy_daily.csv": [
        "as_of_date",
        "proposal_id",
        "paper_tracker_id",
        "strategy_candidate",
        "symbol",
        "recommended_mode",
        "paper_days",
        "heartbeat_days",
        "latest_board_decision",
        "latest_horizon",
        "heartbeat_day_count",
        "entry_day_count",
        "daily_v5_entry_count",
        "daily_synthetic_would_enter_count",
        "cumulative_v5_entry_count",
        "cumulative_synthetic_would_enter_count",
        "daily_would_enter_count",
        "cumulative_would_enter_count",
        "would_enter_count",
        "daily_paper_pnl_observed_count",
        "cumulative_paper_pnl_observed_count",
        "paper_pnl_observed_count",
        "count_scope",
        "paper_pnl_day_count",
        "latest_paper_pnl_usdt",
        "cumulative_paper_pnl_usdt",
        "avg_paper_pnl_bps",
        "paper_pnl_observed_count_by_horizon",
        "complete_count_by_horizon",
        "avg_paper_pnl_bps_by_horizon",
        "win_rate_by_horizon",
        "paper_pnl_day_count_by_horizon",
        "negative_entry_day_count",
        "paper_negative_streak",
        "latest_paper_trend",
        "paper_tracking_status",
        "tracking_stage",
        "required_paper_days",
        "required_entry_day_count",
        "required_slippage_coverage",
        "arrival_mid_coverage",
        "spread_observation_coverage",
        "cost_source_mix",
        "missing_cost_source_count",
        "live_eligible",
        "live_block_reason",
        "created_at",
        "source",
        "schema_version",
    ],
    "reports/paper_strategy_registry.csv": list(PAPER_STRATEGY_REGISTRY_SCHEMA),
    "reports/paper_strategy_registry_current.csv": list(
        PAPER_STRATEGY_REGISTRY_SCHEMA
    ),
    "reports/paper_strategy_registry_history.csv": list(
        PAPER_STRATEGY_REGISTRY_SCHEMA
    ),
    "reports/paper_strategy_identity_conflict.csv": list(
        PAPER_STRATEGY_IDENTITY_CONFLICT_SCHEMA
    ),
    "reports/paper_strategy_promotion_gate.csv": list(PAPER_STRATEGY_PROMOTION_GATE_SCHEMA),
    "reports/paper_strategy_migration_audit.csv": list(PAPER_STRATEGY_MIGRATION_AUDIT_SCHEMA),
    "reports/gate_candidate_level_effectiveness.csv": list(EFFECTIVENESS_SCHEMA),
    "reports/gate_independent_event_effectiveness.csv": list(EFFECTIVENESS_SCHEMA),
    "reports/gate_paper_trade_effectiveness.csv": list(EFFECTIVENESS_SCHEMA),
    "reports/gate_live_trade_effectiveness.csv": list(EFFECTIVENESS_SCHEMA),
    "reports/duplicate_event_report.csv": [
        "canonical_event_id",
        "strategy_evaluation_count",
        "strategy_ids",
        "event_independence_weight",
        "duplicate_evidence",
    ],
    "reports/duplicate_factor_report.csv": [
        "canonical_factor_id",
        "factor_formula_hash",
        "operator_graph_hash",
        "factor_count",
        "factor_ids",
        "duplicate_factor_ids",
        "effective_independence_weight",
    ],
    "reports/strategy_cost_trust.csv": [
        "strategy_id",
        "proposal_id",
        "proposal_hash",
        "strategy_version",
        "symbol",
        "horizon_hours",
        "cost_trust_level",
        "actual_sample_count",
        "bill_matched_sample_count",
        "arrival_observed_sample_count",
        "scale_ready_sample_count",
        "mixed_sample_count",
        "proxy_sample_count",
        "sample_age_seconds",
        "coverage_dimensions",
        "missing_dimensions",
        "evaluated_condition_count",
        "required_condition_count",
        "source",
        "created_at",
        "research_cost_usable",
        "paper_cost_usable",
        "canary_cost_usable",
        "live_cost_usable",
    ],
    "reports/paper_slippage_coverage.csv": [
        "as_of_date",
        "proposal_id",
        "paper_tracker_id",
        "strategy_candidate",
        "symbol",
        "paper_days",
        "observed_slippage_count",
        "required_observation_count",
        "paper_slippage_coverage",
        "required_slippage_coverage",
        "arrival_mid_coverage",
        "spread_observation_coverage",
        "cost_source_mix",
        "missing_cost_source_count",
        "coverage_status",
        "paper_tracking_status",
        "tracking_stage",
        "created_at",
        "source",
        "schema_version",
    ],
    "reports/sol_protect_paper_loss_attribution.csv": [
        "contract_version",
        "schema_version",
        "quant_lab_git_commit",
        "source_version",
        "generated_at_utc",
        "generated_from_bundle_id",
        "as_of_date",
        "proposal_id",
        "strategy_candidate",
        "strategy_id",
        "run_id",
        "entry_ts",
        "symbol",
        "entry_px",
        "alpha6_score",
        "alpha6_side",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "risk_level",
        "btc_trend_state",
        "market_regime",
        "regime_state",
        "entry_vs_24h_low_bps",
        "entry_position_in_24h_range",
        "paper_pnl_4h",
        "paper_pnl_8h",
        "paper_pnl_12h",
        "paper_pnl_24h",
        "mae_bps",
        "mfe_bps",
        "mae_bps_4h",
        "mae_bps_8h",
        "mae_bps_12h",
        "mae_bps_24h",
        "mfe_bps_4h",
        "mfe_bps_8h",
        "mfe_bps_12h",
        "mfe_bps_24h",
        "loss_horizons",
        "primary_loss_bps",
        "attribution_tags",
        "created_at",
        "source",
    ],
    "reports/missed_low_audit.csv": [
        "as_of_date",
        "run_id",
        "source_event_key",
        "symbol",
        "entry_ts",
        "entry_px",
        "entry_reason",
        "probe_type",
        "side",
        "intent",
        "entry_vs_pre_4h_low_bps",
        "entry_vs_pre_8h_low_bps",
        "entry_vs_pre_12h_low_bps",
        "entry_vs_pre_24h_low_bps",
        "entry_position_in_24h_range",
        "realized_net_bps",
        "exit_reason",
        "diagnosis",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/missed_low_by_symbol.csv": [
        "as_of_date",
        "group_key",
        "sample_count",
        "loss_count",
        "profit_count",
        "late_chase_loss_count",
        "late_but_trend_profitable_count",
        "avg_entry_vs_pre_24h_low_bps",
        "avg_entry_position_in_24h_range",
        "avg_realized_net_bps",
        "diagnosis_mix",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/missed_low_by_entry_reason.csv": [
        "as_of_date",
        "group_key",
        "sample_count",
        "loss_count",
        "profit_count",
        "late_chase_loss_count",
        "late_but_trend_profitable_count",
        "avg_entry_vs_pre_24h_low_bps",
        "avg_entry_position_in_24h_range",
        "avg_realized_net_bps",
        "diagnosis_mix",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/late_entry_chase_shadow.csv": [
        "as_of_date",
        "strategy_candidate",
        "source_type",
        "run_id",
        "candidate_id",
        "source_event_key",
        "symbol",
        "ts_utc",
        "entry_or_candidate_px",
        "recent_12h_low",
        "recent_24h_low",
        "entry_vs_12h_low_bps",
        "entry_vs_24h_low_bps",
        "entry_position_in_12h_range",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "late_chase_risk",
        "would_block_if_enabled",
        "realized_net_bps",
        "forward_24h_net_bps",
        "forward_48h_net_bps",
        "outcome_class",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/late_entry_chase_threshold_sensitivity.csv": [
        "start_date",
        "end_date",
        "window_mode",
        "cost_mode",
        "threshold_bps",
        "would_block_count",
        "would_block_loss_count",
        "would_block_profit_count",
        "false_positive_rate",
        "avg_net_bps_blocked",
        "avg_net_bps_not_blocked",
        "ready_for_live_guard",
        "advisory",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/late_entry_chase_threshold_sensitivity_by_symbol.csv": [
        "as_of_date",
        "symbol",
        "threshold_bps",
        "would_block_count",
        "would_block_loss_count",
        "would_block_profit_count",
        "false_positive_rate",
        "avg_net_bps_blocked",
        "avg_net_bps_not_blocked",
        "ready_for_live_guard",
        "advisory",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/pullback_reversal_shadow_outcomes.csv": [
        "as_of_date",
        "rule_version",
        "strategy_candidate",
        "run_id",
        "candidate_id",
        "source_event_key",
        "symbol",
        "ts_utc",
        "regime_state",
        "risk_level",
        "current_px",
        "pre_24h_low",
        "pre_24h_high",
        "pullback_from_24h_high_bps",
        "recent_2h_no_new_low",
        "close_reclaim_1h",
        "current_close_gt_previous_close",
        "btc_not_sharp_drop",
        "spread_not_abnormal",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "selected_roundtrip_cost_bps",
        "cost_quality",
        "horizon_hours",
        "gross_bps",
        "net_bps_after_cost",
        "mfe_bps",
        "mae_bps",
        "win",
        "label_status",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/pullback_reversal_rule_comparison.csv": [
        "as_of_date",
        "comparison_name",
        "rule_name",
        "rule_version",
        "symbol",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "win_rate",
        "avg_mae_bps",
        "p25_net_bps",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/old_v1_vs_v2_comparison.csv": [
        "as_of_date",
        "comparison_name",
        "rule_name",
        "rule_version",
        "symbol",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "win_rate",
        "avg_mae_bps",
        "p25_net_bps",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/pullback_reversal_v2_by_symbol.csv": [
        "as_of_date",
        "rule_version",
        "strategy_candidate",
        "symbol",
        "sample_count",
        "complete_sample_count",
        "avg_24h_net_bps",
        "win_rate_24h",
        "p25_24h_net_bps",
        "avg_mae_bps",
        "decision",
        "decision_reasons",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/btc_probe_exit_policy_review.csv": [
        "as_of_date",
        "strategy_candidate",
        "symbol",
        "run_id",
        "roundtrip_id",
        "entry_ts",
        "exit_ts",
        "entry_px",
        "exit_px",
        "actual_exit_net_bps",
        "would_hold_8h_net_bps",
        "would_hold_12h_net_bps",
        "would_hold_24h_net_bps",
        "would_hold_48h_net_bps",
        "mae_bps",
        "mfe_bps",
        "exit_reason",
        "probe_hold_hours",
        "selected_roundtrip_cost_bps",
        "label_source",
        "exit_policy_signal",
        "status",
        "generated_at_utc",
        "generated_from_bundle_id",
        "schema_version",
        "contract_version",
    ],
    "reports/bnb_swing_exit_policy_review.csv": [
        "as_of_date",
        "strategy_candidate",
        "symbol",
        "run_id",
        "source_entry_id",
        "entry_ts",
        "entry_px",
        "highest_px_after_entry",
        "max_unrealized_bps",
        "actual_exit_ts",
        "actual_exit_px",
        "actual_exit_net_bps",
        "fixed_hold_6h_from_entry_net_bps",
        "fixed_hold_12h_from_entry_net_bps",
        "fixed_hold_24h_from_entry_net_bps",
        "fixed_hold_4h_net_bps",
        "fixed_hold_8h_net_bps",
        "fixed_hold_12h_net_bps",
        "fixed_hold_24h_net_bps",
        "profit_lock_30bps_exit",
        "profit_lock_50bps_exit",
        "delayed_exit_6h_from_actual_exit_net_bps",
        "delayed_exit_12h_from_actual_exit_net_bps",
        "delayed_exit_24h_from_actual_exit_net_bps",
        "delayed_exit_6h_net_bps",
        "delayed_exit_12h_net_bps",
        "delayed_exit_24h_net_bps",
        "trailing_atr_exit",
        "best_exit_policy",
        "best_shadow_exit_policy",
        "best_exit_net_bps",
        "delta_vs_actual_bps",
        "exit_reason",
        "selected_roundtrip_cost_bps",
        "diagnosis",
        "status",
        "duplicate_group_key",
        "duplicate_row_count",
        "selected_for_summary",
        "summary_eligible",
        "review_row_source",
        "v5_vs_quant_lab_consistency_status",
        "v5_vs_quant_lab_mismatch_reason",
        "generated_at_utc",
        "generated_from_bundle_id",
        "schema_version",
        "contract_version",
    ],
    "reports/bnb_exit_policy_v5_vs_quant_lab_consistency.csv": [
        "as_of_date",
        "strategy_candidate",
        "symbol",
        "run_id",
        "source_entry_id",
        "entry_ts",
        "actual_exit_ts",
        "duplicate_group_key",
        "duplicate_row_count",
        "v5_shadow_row_present",
        "quant_lab_recomputed_row_present",
        "consistency_status",
        "mismatch_reason",
        "selected_for_summary_allowed",
        "compared_field_count",
        "mismatch_field_count",
        "v5_fixed_hold_6h_from_entry_net_bps",
        "quant_lab_fixed_hold_6h_from_entry_net_bps",
        "diff_fixed_hold_6h_from_entry_net_bps",
        "v5_fixed_hold_12h_from_entry_net_bps",
        "quant_lab_fixed_hold_12h_from_entry_net_bps",
        "diff_fixed_hold_12h_from_entry_net_bps",
        "v5_fixed_hold_24h_from_entry_net_bps",
        "quant_lab_fixed_hold_24h_from_entry_net_bps",
        "diff_fixed_hold_24h_from_entry_net_bps",
        "v5_delayed_exit_6h_from_actual_exit_net_bps",
        "quant_lab_delayed_exit_6h_from_actual_exit_net_bps",
        "diff_delayed_exit_6h_from_actual_exit_net_bps",
        "v5_delayed_exit_12h_from_actual_exit_net_bps",
        "quant_lab_delayed_exit_12h_from_actual_exit_net_bps",
        "diff_delayed_exit_12h_from_actual_exit_net_bps",
        "v5_delayed_exit_24h_from_actual_exit_net_bps",
        "quant_lab_delayed_exit_24h_from_actual_exit_net_bps",
        "diff_delayed_exit_24h_from_actual_exit_net_bps",
        "generated_at_utc",
        "generated_from_bundle_id",
        "schema_version",
        "contract_version",
    ],
    "reports/exit_policy_review.csv": [
        "as_of_date",
        "generated_at",
        "schema_version",
        "strategy_id",
        "strategy_candidate",
        "source_entry_id",
        "symbol",
        "entry_ts",
        "actual_exit_net_bps",
        "fixed_hold_4h_net_bps",
        "fixed_hold_8h_net_bps",
        "fixed_hold_12h_net_bps",
        "fixed_hold_24h_net_bps",
        "fixed_hold_48h_net_bps",
        "best_alternative_exit_policy",
        "best_alternative_net_bps",
        "delta_vs_actual_bps",
        "mfe_bps",
        "mae_bps",
        "exit_reason",
        "decision",
        "decision_reasons",
        "recommended_mode",
        "source",
    ],
    "reports/pullback_reversal_by_symbol.csv": [
        "start_date",
        "end_date",
        "window_mode",
        "cost_mode",
        "group_type",
        "group_key",
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
        "avg_mfe_bps",
        "avg_mae_bps",
        "cost_quality_mix",
        "decision",
        "decision_reasons",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/pullback_reversal_by_regime.csv": [
        "start_date",
        "end_date",
        "window_mode",
        "cost_mode",
        "group_type",
        "group_key",
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
        "avg_mfe_bps",
        "avg_mae_bps",
        "cost_quality_mix",
        "decision",
        "decision_reasons",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/pullback_reversal_by_horizon.csv": [
        "start_date",
        "end_date",
        "window_mode",
        "cost_mode",
        "group_type",
        "group_key",
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
        "avg_mfe_bps",
        "avg_mae_bps",
        "cost_quality_mix",
        "decision",
        "decision_reasons",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
    ],
    "reports/anti_leakage_check.csv": [
        "start_date",
        "end_date",
        "window_mode",
        "cost_mode",
        "check_name",
        "status",
        "violation_count",
        "detail",
        "mode",
        "generated_at_utc",
        "schema_version",
        "contract_version",
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
    "v5/v5_cost_probe_p3_preflight.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "ingest_ts",
        "schema_version",
        "source_path_inside_bundle",
        "row_index",
        "event_type",
        "generated_at_utc",
        "state",
        "offline_plan_state",
        "offline_plan_blocked_reasons_json",
        "online_exchange_preflight_state",
        "effective_preflight_state",
        "ready_to_request_manual_live_probe",
        "manual_authorization_required",
        "approved_live_order_execution",
        "live_enabled",
        "dry_run",
        "no_order_submitted",
        "live_order_effect",
        "manual_probe_symbol",
        "manual_allowed_symbols_json",
        "manual_max_notional_usdt",
        "manual_required_exit_policy",
        "manual_max_open_seconds",
        "planned_symbols_json",
        "blockers_json",
        "runtime_blockers_json",
        "guard_failures_json",
        "dry_run_plan_state",
        "latest_terminal_roundtrip_id",
        "latest_terminal_roundtrip_ts",
        "next_probe_allowed_at",
        "next_action",
        "post_probe_required_evidence_json",
        "raw_payload_json",
        "stable_row_key",
    ],
    "v5/v5_cost_probe_live_execution_status.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "ingest_ts",
        "schema_version",
        "source_path_inside_bundle",
        "row_index",
        "event_type",
        "generated_at_utc",
        "status",
        "source_state",
        "manual_probe_symbol",
        "authorization_id",
        "authorization_issued_at",
        "authorization_expires_at",
        "authorization_consumed_at",
        "authorization_age_sec",
        "authorization_fresh",
        "authorization_fresh_at_export",
        "authorization_age_sec_at_export",
        "authorization_seconds_to_expiry",
        "authorization_expired_at_export",
        "authorization_validated",
        "authorization_consumed",
        "approved_live_order_execution",
        "live_order_effect",
        "no_order_submitted",
        "recovery_required",
        "recovery_only",
        "entry_submit_intent",
        "entry_client_order_id",
        "entry_order_id",
        "entry_submitted",
        "entry_filled",
        "entry_filled_qty",
        "exit_submit_intent",
        "exit_client_order_id",
        "exit_order_id",
        "exit_submitted",
        "exit_filled",
        "exit_filled_qty",
        "execution_completed",
        "flat_verified",
        "exchange_flat_verified",
        "local_flat_verified",
        "reconcile_ok",
        "instrument_state",
        "quote_balance_sufficient",
        "blockers_json",
        "raw_payload_json",
        "stable_row_key",
    ],
    "v5/v5_fill_bill_cost_reconciliation.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "ingest_ts",
        "source_path_inside_bundle",
        "schema_version",
        "generated_at",
        "runtime_scope",
        "symbol",
        "order_leg",
        "side",
        "order_id",
        "cl_ord_id",
        "trade_ids",
        "fill_count",
        "first_fill_ts",
        "last_fill_ts",
        "liquidity_role",
        "fill_notional_usdt",
        "fee_currency",
        "fill_fee_raw",
        "fill_fee_usdt",
        "bill_ids",
        "bill_fee_usdt",
        "selected_fee_usdt",
        "rebate_usdt",
        "fee_diff_usdt",
        "fee_missing_count",
        "bill_delay_seconds",
        "bill_match_status",
        "cost_evidence_status",
        "cost_source",
        "fee_complete",
        "raw_payload_json",
        "stable_row_key",
    ],
    "v5/v5_cost_probe_live_execution_status_canonical.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "ingest_ts",
        "schema_version",
        "source_path_inside_bundle",
        "row_index",
        "event_type",
        "generated_at_utc",
        "status",
        "source_state",
        "manual_probe_symbol",
        "authorization_id",
        "authorization_issued_at",
        "authorization_expires_at",
        "authorization_consumed_at",
        "authorization_age_sec",
        "authorization_fresh",
        "authorization_fresh_at_export",
        "authorization_age_sec_at_export",
        "authorization_seconds_to_expiry",
        "authorization_expired_at_export",
        "authorization_validated",
        "authorization_consumed",
        "approved_live_order_execution",
        "live_order_effect",
        "no_order_submitted",
        "recovery_required",
        "recovery_only",
        "entry_submit_intent",
        "entry_client_order_id",
        "entry_order_id",
        "entry_submitted",
        "entry_filled",
        "entry_filled_qty",
        "exit_submit_intent",
        "exit_client_order_id",
        "exit_order_id",
        "exit_submitted",
        "exit_filled",
        "exit_filled_qty",
        "execution_completed",
        "flat_verified",
        "exchange_flat_verified",
        "local_flat_verified",
        "reconcile_ok",
        "instrument_state",
        "quote_balance_sufficient",
        "blockers_json",
        "canonical",
        "canonical_priority",
        "revision",
        "supersedes_stable_row_key",
        "raw_payload_json",
        "stable_row_key",
    ],
    "v5/v5_cost_probe_order_events.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "ingest_ts",
        "schema_version",
        "source_path_inside_bundle",
        "row_index",
        "event_key",
        "event_id",
        "event_type",
        "event_ts",
        "symbol",
        "normalized_symbol",
        "leg",
        "side",
        "intent",
        "order_status",
        "order_key",
        "order_id",
        "client_order_id",
        "exchange_order_id",
        "submitted_at",
        "filled_at",
        "dry_run",
        "live_enabled",
        "no_order_submitted",
        "notional_usdt",
        "filled_qty",
        "avg_px",
        "fee_usdt",
        "live_order_effect",
        "raw_payload_json",
    ],
    "v5/v5_cost_probe_roundtrip_events.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "ingest_ts",
        "schema_version",
        "source_path_inside_bundle",
        "row_index",
        "event_key",
        "event_id",
        "event_type",
        "event_ts",
        "symbol",
        "normalized_symbol",
        "roundtrip_key",
        "roundtrip_id",
        "roundtrip_status",
        "entry_order_id",
        "exit_order_id",
        "entry_order_status",
        "exit_order_status",
        "opened_at",
        "closed_at",
        "dry_run",
        "live_enabled",
        "no_order_submitted",
        "gross_pnl_usdt",
        "fees_usdt",
        "net_pnl_usdt",
        "authorization_id",
        "execution_completed",
        "flat_verified",
        "exchange_flat_verified",
        "local_flat_verified",
        "reconcile_ok",
        "cost_evidence_complete",
        "eligible_for_cost_model",
        "eligible_for_live_cost_coverage",
        "sample_origin",
        "source",
        "entry_filled_qty",
        "exit_filled_qty",
        "entry_fee_usdt",
        "exit_fee_usdt",
        "roundtrip_cost_bps",
        "fee_conversion_warnings",
        "bill_match_status",
        "fee_match_status",
        "live_order_effect",
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
    "v5/v5_btc_probe_entry_quality_audit.csv": [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "ingest_ts",
        "schema_version",
        "source_path_inside_bundle",
        "row_index",
        "run_id",
        "entry_ts",
        "entry_px",
        "final_score",
        "expected_edge_bps",
        "required_edge_bps",
        "btc_trend_score",
        "trend_buy_count",
        "alpha6_score",
        "alpha6_side",
        "bypassed_negative_expectancy_reason",
        "selected_symbol",
        "normalized_symbol",
        "selection_mode",
        "negative_expectancy_state",
        "same_symbol_reentry_bypass",
        "price_distance_from_recent_low_bps",
        "price_distance_from_recent_high_bps",
        "anti_chase_flag",
        "entry_quality_status",
        "live_order_effect",
        "raw_payload_json",
        "stable_row_key",
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
        "feature_denominator_rows",
        "no_signal_context_rows",
        "run_count",
        "runs_with_candidate_event",
        "runs_without_candidate_event",
        "runs_without_candidate_event_json",
        "candidate_rows_by_run_json",
        "candidate_symbol_rows_by_run_json",
        "run_symbol_min_rows",
        "feature_completeness",
        "feature_completeness_by_field_json",
        "required_feature_completeness",
        "required_feature_completeness_by_field_json",
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
    "reports/v5_candidate_feature_completeness_by_strategy.csv": [
        "strategy_candidate",
        "row_count",
        "feature_denominator_row_count",
        "no_signal_context_row_count",
        "core_signal_completeness",
        "edge_context_completeness",
        "optional_signal_completeness",
        "field_completeness_json",
        "missing_core_fields",
        "missing_edge_context_fields",
        "missing_optional_fields",
        "sample_symbols",
        "live_order_effect",
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
    "v5/v5_expanded_universe_advisory_reader.csv": [
        "run_id",
        "ts_utc",
        "source_path",
        "advisory_source",
        "advisory_fresh",
        "advisory_age_sec",
        "advisory_contract_match",
        "stale_advisory_used",
        "api_fallback_attempted",
        "api_fallback_success",
        "as_of_ts",
        "generated_at",
        "expires_at",
        "contract_version",
        "quant_lab_git_commit",
        "source_version",
        "universe_type",
        "symbol",
        "symbol_in_live_universe",
        "live_symbols_unchanged",
        "strategy_id",
        "strategy_candidate",
        "experiment_name",
        "decision",
        "recommended_mode",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "expanded_universe_maturity_state",
        "cost_source",
        "cost_source_quality",
        "cost_bps",
        "cost_model_version",
        "response_action",
        "negative_advisory",
        "paper_tracking_allowed",
        "shadow_tracking_allowed",
        "max_paper_notional_usdt",
        "max_live_notional_usdt",
        "max_live_notional_usdt_ignored",
        "live_block_reasons",
        "would_block_if_enabled",
        "would_enter",
        "no_sample_reason",
        "advisory_reason",
        "live_order_effect",
        "future_4h_net_bps",
        "label_4h_net_bps",
        "label_4h_after_cost_bps",
        "future_8h_net_bps",
        "label_8h_net_bps",
        "label_8h_after_cost_bps",
        "future_12h_net_bps",
        "label_12h_net_bps",
        "label_12h_after_cost_bps",
        "future_24h_net_bps",
        "label_24h_net_bps",
        "label_24h_after_cost_bps",
        "future_48h_net_bps",
        "label_48h_net_bps",
        "label_48h_after_cost_bps",
        "future_72h_net_bps",
        "label_72h_net_bps",
        "label_72h_after_cost_bps",
    ],
    "v5/v5_expanded_universe_paper_runs.csv": [
        "run_id",
        "ts_utc",
        "paper_date",
        "universe_type",
        "symbol",
        "symbol_in_live_universe",
        "live_symbols_unchanged",
        "strategy_id",
        "strategy_candidate",
        "experiment_name",
        "tracking_mode",
        "decision",
        "recommended_mode",
        "response_action",
        "negative_advisory",
        "would_enter",
        "would_size_usdt",
        "max_paper_notional_usdt",
        "max_live_notional_usdt_ignored",
        "no_sample_reason",
        "advisory_source",
        "advisory_source_path",
        "advisory_fresh",
        "advisory_contract_match",
        "live_block_reasons",
        "live_order_effect",
        "paper_pnl_bps_4h",
        "label_4h_status",
        "paper_pnl_bps_8h",
        "label_8h_status",
        "paper_pnl_bps_12h",
        "label_12h_status",
        "paper_pnl_bps_24h",
        "label_24h_status",
        "paper_pnl_bps_48h",
        "label_48h_status",
        "paper_pnl_bps_72h",
        "label_72h_status",
    ],
    "v5/v5_expanded_universe_paper_daily.csv": [
        "paper_date",
        "strategy_id",
        "experiment_name",
        "symbol",
        "row_count",
        "entry_count",
        "shadow_count",
        "negative_count",
        "avg_paper_pnl_bps_by_horizon",
        "paper_pnl_observed_count_by_horizon",
        "win_rate_by_horizon",
        "live_order_effect",
        "avg_paper_pnl_bps_4h",
        "avg_paper_pnl_bps_8h",
        "avg_paper_pnl_bps_12h",
        "avg_paper_pnl_bps_24h",
        "avg_paper_pnl_bps_48h",
        "avg_paper_pnl_bps_72h",
    ],
    f"reports/{ENFORCE_READINESS_CSV}": [
        "as_of_ts",
        "strategy",
        "version",
        "readiness_status",
        "shadow_only_recommended",
        "raw_imported_rows",
        "unique_event_rows",
        "duplicate_event_rows",
        "duplicate_rate",
        "dedupe_health_status",
        "dedupe_block_reason",
        "cost_symbol_hit_rate",
        "cost_live_symbol_hit_rate",
        "missing_live_cost_symbols",
        "live_universe_source",
        "actual_or_mixed_cost_coverage",
        "actual_or_mixed_cost_coverage_research_universe",
        "actual_or_mixed_cost_coverage_live_universe",
        "actual_or_mixed_cost_coverage_expanded_universe",
        "proxy_only_symbols_live",
        "proxy_only_symbols_expanded",
        "blocked_reasons",
        "warning_reasons",
        "required_actions",
        "contract_version",
    ],
}
CSV_SCHEMAS["v5/v5_cost_probe_roundtrip_canonical.csv"] = [
    *CSV_SCHEMAS["v5/v5_cost_probe_roundtrip_events.csv"],
    "revision",
    "supersedes_event_id",
    "terminal",
    "canonical",
    "canonical_priority",
]
CSV_SCHEMAS["reports/bnb_paper_strategy_runs.csv"] = CSV_SCHEMAS["reports/paper_strategy_runs.csv"]
CSV_SCHEMAS["reports/bnb_paper_strategy_daily.csv"] = list(
    dict.fromkeys(
        [
            "strategy_id",
            "entry_count",
            "complete_count",
            "pending_count",
            "avg_paper_pnl_bps_4h",
            "avg_paper_pnl_bps_8h",
            *CSV_SCHEMAS["reports/paper_strategy_daily.csv"],
        ]
    )
)

_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS = [
    "strategy",
    "bundle_sha256",
    "bundle_name",
    "bundle_ts",
    "ingest_ts",
    "source_path_inside_bundle",
    "row_index",
    "run_id",
]
_V5_PAPER_TELEMETRY_AUDIT_FIELDS = ["raw_payload_json", "stable_row_key"]
CSV_SCHEMAS.update(
    {
        "v5/v5_paper_strategy_proposal_ack.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "proposal_id",
            "proposal_hash",
            "paper_tracker_id",
            "tracker_id",
            "accepted",
            "accepted_at",
            "paper_only",
            "max_live_notional_usdt",
            "recommended_mode",
            "symbol",
            "strategy_candidate",
            "strategy_version",
            "suggested_horizon",
            "proposal_source",
            "reject_reason",
            "contract_version",
            "rules_locked",
            "live_order_effect",
            "expires_at",
            "source_v5_commit",
            "source_v5_bundle_sha256",
            "schema_version",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_paper_strategy_registry.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "proposal_id",
            "proposal_hash",
            "tracker_id",
            "strategy_id",
            "strategy_version",
            "strategy_family",
            "symbol",
            "timeframe",
            "state",
            "rules_locked",
            "paper_only",
            "live_order_effect",
            "created_at",
            "updated_at",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_paper_strategy_state.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "tracker_id",
            "proposal_id",
            "strategy_id",
            "symbol",
            "state",
            "paper_trade_id",
            "open_paper_position",
            "cooldown_remaining_bars",
            "last_processed_bar_ts",
            "updated_at",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_paper_strategy_signals.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "signal_id",
            "proposal_id",
            "tracker_id",
            "strategy_id",
            "strategy_version",
            "symbol",
            "signal_ts",
            "decision_ts",
            "signal_type",
            "triggered",
            "observability",
            "arrival_bid",
            "arrival_ask",
            "arrival_mid",
            "spread_bps",
            "quote_timestamp",
            "quote_age_seconds",
            "price_source",
            "fallback_level",
            "valid_for_promotion",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_paper_strategy_runs.csv": list(
            dict.fromkeys(
                [
                    *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
                    "schema_version",
                    "paper_trade_id",
                    "proposal_id",
                    "strategy_id",
                    "strategy_version",
                    "symbol",
                    "direction",
                    "entry_signal_ts",
                    "entry_decision_ts",
                    "entry_arrival_mid",
                    "entry_bid",
                    "entry_ask",
                    "virtual_entry_price",
                    "paper_notional",
                    "virtual_quantity",
                    "exit_signal_ts",
                    "exit_decision_ts",
                    "exit_arrival_mid",
                    "virtual_exit_price",
                    "fee_estimate",
                    "slippage_estimate",
                    "total_cost_bps",
                    "gross_pnl_bps",
                    "net_pnl_bps",
                    "max_favorable_excursion",
                    "max_adverse_excursion",
                    "mfe_bps",
                    "mae_bps",
                    "profit_giveback_bps",
                    "exit_efficiency",
                    "exit_timing_bars",
                    "exit_timing_state",
                    "holding_period_seconds",
                    "holding_bars",
                    "exit_reason",
                    "cost_source",
                    "required_cost_trust_level",
                    "cost_trust_level",
                    "cost_model_version",
                    "market_regime",
                    "spread_bps",
                    "quote_timestamp",
                    "quote_age_seconds",
                    "price_source",
                    "fallback_level",
                    "real_permission_would_allow",
                    "real_mode_would_allow",
                    "real_cost_canary_ready",
                    "real_funds_sufficient",
                    "valid_for_promotion",
                    "opened_at",
                    "closed_at",
                    "ts_utc",
                    "as_of_date",
                    "paper_tracker_id",
                    "strategy_candidate",
                    "recommended_mode",
                    "board_decision",
                    "suggested_horizon",
                    "horizon_hours",
                    "would_enter",
                    "would_exit",
                    "would_size",
                    "would_size_usdt",
                    "paper_source",
                    "paper_count_scope",
                    "paper_pnl_bps",
                    "paper_pnl_usdt",
                    "estimated_spread_bps",
                    *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
                ]
            )
        ),
        "v5/v5_paper_strategy_daily.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "paper_date",
            "proposal_id",
            "paper_tracker_id",
            "strategy_id",
            "symbol",
            "paper_days",
            "heartbeat_day_count",
            "entry_day_count",
            "daily_would_enter_count",
            "cumulative_would_enter_count",
            "would_enter_count",
            "closed_entries",
            "daily_paper_pnl_observed_count",
            "cumulative_paper_pnl_observed_count",
            "paper_pnl_observed_count",
            "paper_pnl_day_count",
            "avg_paper_pnl_bps",
            "arrival_mid_coverage",
            "spread_observation_coverage",
            "cost_source_mix",
            "created_at",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_paper_strategy_exit_quality.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "proposal_id",
            "strategy_id",
            "strategy_version",
            "symbol",
            "closed_trade_count",
            "avg_net_pnl_bps",
            "avg_mfe_bps",
            "avg_mae_bps",
            "avg_profit_giveback_bps",
            "avg_exit_efficiency",
            "avg_holding_bars",
            "high_profit_giveback_count",
            "exit_reason_mix",
            "exit_timing_state_mix",
            "diagnosis",
            "valid_for_live_orders",
            "live_order_effect",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_trade_opportunity_funnel.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "ts_utc",
            "execution_mode",
            "stage_order",
            "stage",
            "input_count",
            "output_count",
            "dropped_count",
            "conversion_rate",
            "entry_output_count",
            "exit_output_count",
            "primary_blocker",
            "blocker_mix",
            "count_source",
            "live_order_effect",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_paper_strategy_quote_coverage.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "proposal_id",
            "strategy_id",
            "symbol",
            "candidate_signal_count",
            "observable_quote_count",
            "arrival_mid_coverage",
            "stale_quote_rate",
            "quote_fallback_rate",
            "target_coverage",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_paper_strategy_cost_evidence.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "proposal_id",
            "strategy_id",
            "symbol",
            "required_cost_trust_level",
            "closed_trade_count",
            "cost_observed_count",
            "cost_source",
            "cost_trust_level",
            "valid_for_live_coverage",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_paper_strategy_errors.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "ts_utc",
            "proposal_id",
            "error_code",
            "error_type",
            "error_message",
            "live_order_effect",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_paper_strategy_restart_recovery.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "recovered_at",
            "tracker_id",
            "proposal_id",
            "state",
            "open_trade_preserved",
            "schema_version",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
        "v5/v5_quant_lab_contract_status.csv": [
            *_V5_PAPER_TELEMETRY_ENVELOPE_FIELDS,
            "schema_version",
            "contract_version",
            "paper_runtime_enabled",
            "paper_runtime_live_order_effect",
            "quant_lab_mode",
            "canary_enabled",
            "loaded_tracker_count",
            "current_active_tracker_count",
            "current_pending_tracker_count",
            "superseded_exit_only_count",
            "superseded_closed_count",
            "active_tracker_count",
            "active_tracker_count_deprecated",
            "active_tracker_count_semantics",
            "open_paper_position_count",
            "real_order_calls",
            "real_position_mutations",
            "generated_at",
            *_V5_PAPER_TELEMETRY_AUDIT_FIELDS,
        ],
    }
)

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


class WebDerivedSnapshotRefreshResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    live_order_effect: str = "none_read_only_lake_refresh"
    refreshed_datasets: list[str] = Field(default_factory=list)
    row_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _DatasetSnapshot:
    frames: dict[str, pl.DataFrame]
    row_counts: dict[str, int]
    warnings: list[str]
    transient_frames: dict[str, pl.DataFrame] = dataclass_field(default_factory=dict)


def _publish_strategy_opportunity_advisory_snapshot(
    root: Path,
    snapshot: _DatasetSnapshot,
    *,
    generated_at: datetime,
) -> _DatasetSnapshot:
    derived = _canonical_strategy_opportunity_frames(
        snapshot.frames,
        generated_at=generated_at,
    )
    opportunity = derived["strategy_opportunity_advisory"]
    frames = dict(snapshot.frames)
    frames["factor_strategy_bridge_candidates"] = derived["factor_strategy_bridge_candidates"]
    frames["strategy_opportunity_advisory"] = opportunity
    row_counts = dict(snapshot.row_counts)
    warnings = list(snapshot.warnings)
    _publish_export_frame(
        root,
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        dataset_name="factor_strategy_bridge_candidates",
        frame=derived["factor_strategy_bridge_candidates"],
    )
    _publish_export_frame(
        root,
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        dataset_name="strategy_opportunity_advisory",
        frame=opportunity,
    )
    return _DatasetSnapshot(
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        transient_frames={
            **snapshot.transient_frames,
            "bottom_zone_reversal_shadow": derived["bottom_zone_reversal_shadow"],
            "fast_microstructure_forward_test": derived["fast_microstructure_forward_test"],
        },
    )


PAPER_PIPELINE_EXPORT_DATASETS = (
    "paper_strategy_proposal",
    "paper_strategy_proposals_current",
    "paper_strategy_trackers_current",
    "paper_strategy_registry",
    "paper_strategy_registry_current",
    "paper_strategy_registry_history",
    "paper_strategy_ack_current",
    "paper_strategy_ack_history",
    "paper_strategy_identity_conflict",
    "paper_cohort_manifest",
    "paper_strategy_promotion_gate",
    "paper_strategy_migration_audit",
    "strategy_cost_trust",
)


def _publish_paper_strategy_pipeline_snapshot(
    root: Path,
    snapshot: _DatasetSnapshot,
    day: date,
) -> _DatasetSnapshot:
    frames = dict(snapshot.frames)
    row_counts = dict(snapshot.row_counts)
    warnings = [
        warning
        for warning in snapshot.warnings
        if not any(
            warning.startswith(f"{dataset_name} dataset is ")
            or warning.startswith(f"{dataset_name} read failed:")
            or warning.startswith(f"{dataset_name} sampled read failed:")
            for dataset_name in PAPER_PIPELINE_EXPORT_DATASETS
        )
    ]
    try:
        build_and_publish_paper_strategy_pipeline(root, as_of_date=day)
    except Exception as exc:
        warnings.append(
            "paper_strategy_pipeline publish skipped: "
            f"{type(exc).__name__}: {_safe_warning_text(str(exc))}"
        )
        return _DatasetSnapshot(
            frames=frames,
            row_counts=row_counts,
            warnings=warnings,
            transient_frames=snapshot.transient_frames,
        )

    for dataset_name in PAPER_PIPELINE_EXPORT_DATASETS:
        frame, row_count, warning = _load_export_frame(root, dataset_name)
        frames[dataset_name] = frame
        row_counts[dataset_name] = row_count
        if warning:
            warnings.append(warning)
    return _DatasetSnapshot(
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        transient_frames=snapshot.transient_frames,
    )


def _publish_ops_truthfulness_snapshot(
    root: Path,
    snapshot: _DatasetSnapshot,
    *,
    generated_at: datetime,
) -> _DatasetSnapshot:
    frames = dict(snapshot.frames)
    row_counts = dict(snapshot.row_counts)
    warnings = list(snapshot.warnings)
    auth = build_api_auth_reports(
        frames.get("api_request_metrics", pl.DataFrame()),
        generated_at=generated_at,
    )
    paper_runtime_freshness = build_paper_runtime_freshness(
        frames,
        generated_at=generated_at,
    )
    propagation = build_paper_proposal_propagation_status(
        proposals=frames.get("paper_strategy_proposals_current", pl.DataFrame()),
        ack_current=_prefer_frame(
            frames.get("paper_strategy_ack_current", pl.DataFrame()),
            frames.get("v5_paper_strategy_proposal_ack_current", pl.DataFrame()),
        ),
        ack_history=_prefer_frame(
            frames.get("paper_strategy_ack_history", pl.DataFrame()),
            frames.get("v5_paper_strategy_proposal_ack_history", pl.DataFrame()),
        ),
        trackers_current=_prefer_frame(
            frames.get("paper_strategy_trackers_current", pl.DataFrame()),
            frames.get("v5_paper_strategy_trackers_current", pl.DataFrame()),
        ),
        trackers_history=_prefer_frame(
            frames.get("paper_strategy_registry_history", pl.DataFrame()),
            frames.get("v5_paper_strategy_registry_history", pl.DataFrame()),
        ),
        generated_at=generated_at,
    )
    for dataset_name, frame in (
        ("api_auth_incident", auth["api_auth_incident"]),
        ("paper_runtime_freshness", paper_runtime_freshness),
        ("paper_proposal_propagation_status", propagation),
    ):
        _publish_export_frame(
            root,
            frames=frames,
            row_counts=row_counts,
            warnings=warnings,
            dataset_name=dataset_name,
            frame=frame,
        )
    return _DatasetSnapshot(
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        transient_frames={
            **snapshot.transient_frames,
            "api_auth_error_timeline": auth["api_auth_error_timeline"],
            "api_auth_client_summary": auth["api_auth_client_summary"],
            "paper_runtime_freshness": paper_runtime_freshness,
            "paper_proposal_propagation_status": propagation,
        },
    )


def _publish_trade_level_snapshot(
    root: Path,
    snapshot: _DatasetSnapshot,
    *,
    generated_at: datetime,
) -> _DatasetSnapshot:
    frames = dict(snapshot.frames)
    derived = build_trade_level_frames_from_sources(
        candidate_events=frames.get("v5_candidate_event", pl.DataFrame()),
        candidate_labels=frames.get("v5_candidate_label", pl.DataFrame()),
        risk_permissions=frames.get("risk_permission", pl.DataFrame()),
        v5_trades=frames.get("v5_trade_event", pl.DataFrame()),
        v5_roundtrips=frames.get("v5_roundtrip", pl.DataFrame()),
        order_lifecycles=_read_optional_lake_frame(root / "silver" / "v5_order_lifecycle"),
        as_of_date=generated_at.date(),
        created_at=generated_at,
    )
    row_counts = dict(snapshot.row_counts)
    warnings = [
        warning
        for warning in snapshot.warnings
        if not warning.startswith("trade_opportunity_event dataset is ")
        and not warning.startswith("trade_opportunity_label dataset is ")
        and not warning.startswith("trade_level_similarity_outcome dataset is ")
        and not warning.startswith("trade_level_judgment dataset is ")
        and not warning.startswith("trade_level_bucket_policy dataset is ")
        and not warning.startswith("trade_level_opportunity_queue dataset is ")
        and not warning.startswith("quant_lab_false_block_audit dataset is ")
        and not warning.startswith("v5_trade_learning_sample dataset is ")
        and not warning.startswith("v5_trade_outcome_attribution dataset is ")
        and not warning.startswith("quant_lab_opportunity_cost_event dataset is ")
        and not warning.startswith("quant_lab_opportunity_cost_daily dataset is ")
        and not warning.startswith("opportunity_cost_by_bucket dataset is ")
        and not warning.startswith("quant_lab_decision_regret dataset is ")
    ]
    for dataset_name, frame in derived.items():
        _publish_export_frame(
            root,
            frames=frames,
            row_counts=row_counts,
            warnings=warnings,
            dataset_name=dataset_name,
            frame=frame,
        )
    return _DatasetSnapshot(
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        transient_frames=snapshot.transient_frames,
    )


def _canonical_strategy_opportunity_frames(
    frames: Mapping[str, pl.DataFrame],
    *,
    generated_at: datetime,
) -> dict[str, pl.DataFrame]:
    market = frames.get("market_bar", pl.DataFrame())
    costs = _normalize_symbol_frame(frames.get("cost_bucket_daily", pl.DataFrame()))
    market_regime = frames.get("market_regime_daily", pl.DataFrame())
    factor_forward_validation = build_factor_forward_validation(
        factor_candidates=frames.get("factor_candidate", pl.DataFrame()),
        factor_values=frames.get("factor_value", pl.DataFrame()),
        market_bars=market,
        market_regime=market_regime,
        cost_bucket_daily=costs,
    )
    trades = _normalize_symbol_frame(
        _prefer_frame(
            frames.get("trade_activity_1m", pl.DataFrame()),
            frames.get("trade_print", pl.DataFrame()),
        )
    )
    books = _normalize_symbol_frame(
        _prefer_frame(
            frames.get("orderbook_spread_1m", pl.DataFrame()),
            frames.get("orderbook_snapshot", pl.DataFrame()),
        )
    )
    bottom_zone_reversal_shadow = build_bottom_zone_reversal_shadow(
        market_bars=market,
        orderbook_spread_1m=books,
        trade_activity_1m=trades,
        generated_at=generated_at,
    )
    fast_microstructure_forward_test = build_fast_microstructure_forward_test(
        market_bars=market,
        orderbook_spread_1m=books,
        trade_activity_1m=trades,
        market_regime=market_regime,
        cost_bucket_daily=costs,
        generated_at=generated_at,
    )
    factor_v2 = build_factor_factory_v2_reports(
        candidates=frames.get("factor_candidate", pl.DataFrame()),
        evidence=frames.get("factor_evidence", pl.DataFrame()),
        correlations=frames.get("factor_correlation_daily", pl.DataFrame()),
        factor_forward_validation=factor_forward_validation,
        fast_microstructure_forward_test=fast_microstructure_forward_test,
    )
    alpha_discovery_board = _alpha_discovery_board_for_export(
        frames.get("alpha_discovery_board", pl.DataFrame())
    )
    strategy_evidence = _strategy_evidence_for_export(
        frames.get("strategy_evidence", pl.DataFrame())
    )
    telemetry_latest_ts = _latest_v5_bundle_ts(dict(frames))
    risk = _risk_permissions_for_export(
        frames.get("risk_permission", pl.DataFrame()),
        frames,
        telemetry_latest_ts=telemetry_latest_ts,
    )
    _, paper_daily, paper_slippage = _paper_tracking_frames_for_export(dict(frames))
    research_portfolio = dedupe_research_portfolio_status(
        frames.get("research_portfolio_status", pl.DataFrame())
    )
    paper_proposals = _paper_strategy_proposals_for_export(
        alpha_discovery_board,
        research_portfolio=research_portfolio,
        persisted_proposals=frames.get("paper_strategy_proposal", pl.DataFrame()),
    )
    factor_strategy_bridge_candidates = factor_v2["factor_strategy_bridge_candidates"]
    opportunity_advisory = _strategy_opportunity_advisory_for_export(
        alpha_discovery_board=alpha_discovery_board,
        strategy_evidence=strategy_evidence,
        paper_proposals=paper_proposals,
        risk_permissions=risk,
        cost_health=frames.get("cost_health_daily", pl.DataFrame()),
        paper_daily=paper_daily,
        paper_slippage=paper_slippage,
        research_portfolio=research_portfolio,
        entry_quality_advisory=frames.get("v5_entry_quality_advisory", pl.DataFrame()),
        regime_strategy_advisory=frames.get("regime_strategy_advisory", pl.DataFrame()),
        risk_on_multi_buy_shadow=frames.get("v5_risk_on_multi_buy_shadow", pl.DataFrame()),
        alpha_factory_results=frames.get("alpha_factory_result", pl.DataFrame()),
        alpha_factory_promotion_queue=frames.get("alpha_factory_promotion_queue", pl.DataFrame()),
        expanded_universe_maturity=frames.get(
            "expanded_universe_candidate_maturity",
            pl.DataFrame(),
        ),
        bottom_zone_reversal_shadow=bottom_zone_reversal_shadow,
        factor_strategy_bridge_candidates=factor_strategy_bridge_candidates,
    )
    return {
        "bottom_zone_reversal_shadow": bottom_zone_reversal_shadow,
        "fast_microstructure_forward_test": fast_microstructure_forward_test,
        "factor_strategy_bridge_candidates": factor_strategy_bridge_candidates,
        "strategy_opportunity_advisory": opportunity_advisory,
    }


def _publish_risk_permission_dependency_meta_snapshot(
    root: Path,
    snapshot: _DatasetSnapshot,
) -> _DatasetSnapshot:
    frames = dict(snapshot.frames)
    risk = frames.get("risk_permission", pl.DataFrame())
    gates = frames.get("gate_decision", pl.DataFrame())
    cost_health = frames.get("cost_health_daily", pl.DataFrame())
    telemetry_latest_ts = _frame_latest_iso(
        risk,
        ("telemetry_latest_ts", "source_bundle_ts", "as_of_ts", "created_at"),
    )
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    dependency_source_sha = _dependency_source_sha(
        {
            "risk_permission": risk,
            "gate_decision": gates,
            "cost_health_daily": cost_health,
        },
        telemetry_latest_ts=telemetry_latest_ts,
    )
    strategies = _frame_unique_texts(risk, "strategy") or ["v5"]
    versions = _frame_unique_texts(risk, "version") or ["unknown"]
    rows = [
        {
            "strategy": strategy,
            "version": version,
            "risk_permission_source_sha": _frame_source_sha(
                "risk_permission",
                risk,
                _frame_latest_iso(risk, ("as_of_ts", "created_at")),
                _frame_latest_iso(risk, ("expires_at",)),
            ),
            "gate_decision_source_sha": _frame_source_sha(
                "gate_decision",
                gates,
                _frame_latest_iso(gates, ("created_at", "as_of_ts")),
                "",
            ),
            "cost_health_source_sha": _frame_source_sha(
                "cost_health_daily",
                cost_health,
                _frame_latest_iso(cost_health, ("created_at", "day")),
                "",
            ),
            "telemetry_latest_ts": telemetry_latest_ts,
            "source_sha": dependency_source_sha,
            "generated_at": generated_at,
        }
        for strategy in strategies
        for version in versions
    ]
    frame = pl.DataFrame(rows, infer_schema_length=None)
    row_counts = dict(snapshot.row_counts)
    warnings = list(snapshot.warnings)
    _publish_export_frame(
        root,
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        dataset_name="risk_permission_api_dependency_meta",
        frame=frame,
    )
    return _DatasetSnapshot(
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        transient_frames=snapshot.transient_frames,
    )


def _publish_cost_bootstrap_readiness_snapshot(
    root: Path,
    snapshot: _DatasetSnapshot,
    *,
    generated_at: datetime,
) -> _DatasetSnapshot:
    frames = dict(snapshot.frames)
    readiness = build_cost_bootstrap_readiness(
        _normalize_symbol_frame(frames.get("cost_bucket_daily", pl.DataFrame())),
        v5_order_lifecycle=_read_optional_lake_frame(root / "silver" / "v5_order_lifecycle"),
        v5_cost_probe_order_events=frames.get("v5_cost_probe_order_event", pl.DataFrame()),
        v5_cost_probe_roundtrip_events=frames.get(
            "v5_cost_probe_roundtrip_event",
            pl.DataFrame(),
        ),
        okx_private_readonly_fills=frames.get("okx_private_readonly_fills", pl.DataFrame()),
        okx_private_readonly_bills=frames.get("okx_private_readonly_bills", pl.DataFrame()),
        live_symbols=DEFAULT_LIVE_UNIVERSE_SYMBOLS,
        generated_at=generated_at,
    )
    fill_bill_match = build_cost_probe_fill_bill_match(
        frames.get("v5_cost_probe_order_event", pl.DataFrame()),
        frames.get("v5_cost_probe_roundtrip_event", pl.DataFrame()),
        frames.get("okx_private_readonly_fills", pl.DataFrame()),
        frames.get("okx_private_readonly_bills", pl.DataFrame()),
        generated_at=generated_at,
    )
    cost_disagreement = build_cost_probe_cost_disagreement(
        _normalize_symbol_frame(frames.get("cost_bucket_daily", pl.DataFrame())),
        frames.get("v5_cost_probe_order_event", pl.DataFrame()),
        frames.get("v5_cost_probe_roundtrip_event", pl.DataFrame()),
        frames.get("okx_private_readonly_fills", pl.DataFrame()),
        frames.get("okx_private_readonly_bills", pl.DataFrame()),
        generated_at=generated_at,
    )
    row_counts = dict(snapshot.row_counts)
    warnings = [
        warning
        for warning in snapshot.warnings
        if not warning.startswith("cost_probe_fill_bill_match dataset is ")
        and not warning.startswith("cost_probe_cost_disagreement dataset is ")
    ]
    _publish_export_frame(
        root,
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        dataset_name="cost_bootstrap_readiness",
        frame=readiness,
    )
    _publish_export_frame(
        root,
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        dataset_name="cost_probe_fill_bill_match",
        frame=fill_bill_match,
    )
    _publish_export_frame(
        root,
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        dataset_name="cost_probe_cost_disagreement",
        frame=cost_disagreement,
    )
    return _DatasetSnapshot(
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        transient_frames=snapshot.transient_frames,
    )


def _read_optional_lake_frame(path: Path) -> pl.DataFrame:
    try:
        return read_parquet_dataset(path)
    except Exception:
        return pl.DataFrame()


def _publish_export_frame(
    root: Path,
    *,
    frames: dict[str, pl.DataFrame],
    row_counts: dict[str, int],
    warnings: list[str],
    dataset_name: str,
    frame: pl.DataFrame,
) -> None:
    """Best-effort publication for export-derived datasets.

    Expert pack generation should remain available even when a production lake
    directory has drifted to read-only or mixed ownership. In that case the
    derived frame is still exported from memory and the failed lake publication
    is surfaced as an explicit warning for ops follow-up.
    """

    dataset_path = root / readers.DATASET_PATHS.get(dataset_name, Path("gold") / dataset_name)
    try:
        write_parquet_dataset(frame, dataset_path)
        if dataset_name in SNAPSHOT_META_DATASETS:
            _write_snapshot_meta(dataset_path, dataset_name=dataset_name, frame=frame)
        frames[dataset_name] = frame
    except Exception as exc:
        frames[dataset_name] = frame
        warnings.append(
            f"{dataset_name} publish skipped: {type(exc).__name__}:{_safe_warning_text(str(exc))}"
        )
    row_counts[dataset_name] = frame.height


def _write_snapshot_meta(dataset_path: Path, *, dataset_name: str, frame: pl.DataFrame) -> None:
    dataset_path.mkdir(parents=True, exist_ok=True)
    generated_at = _frame_latest_iso(
        frame,
        (
            "generated_at",
            "generated_at_utc",
            "created_at",
            "as_of_ts",
            "as_of_date",
            "latest_bundle_ts",
            "bundle_ts",
            "ingest_ts",
            "event_ts",
            "ts_utc",
            "ts",
            "entry_ts",
            "exit_ts",
            "paper_date",
            "date",
            "day",
        ),
    )
    expires_at = _frame_latest_iso(frame, ("expires_at",))
    schema_version = _frame_text_value(frame, "schema_version")
    parquet_count = sum(1 for path in dataset_path.rglob("*.parquet") if path.is_file())
    payload = {
        "dataset": dataset_name,
        "generated_at": generated_at,
        "expires_at": expires_at,
        "row_count": frame.height,
        "source_sha": _frame_source_sha(dataset_name, frame, generated_at, expires_at),
        "file_count": parquet_count,
        "schema_version": schema_version
        or (
            STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION
            if dataset_name == "strategy_opportunity_advisory"
            else dataset_name
        ),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    meta_path = dataset_path / "_snapshot_meta.json"
    tmp_path = dataset_path / "._snapshot_meta.tmp"
    tmp_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp_path.replace(meta_path)


def _frame_latest_iso(frame: pl.DataFrame, columns: Iterable[str]) -> str:
    if frame.is_empty():
        return ""
    for column in columns:
        if column not in frame.columns:
            continue
        try:
            value = frame.select(
                pl.col(column)
                .cast(pl.Utf8, strict=False)
                .str.to_datetime(time_zone="UTC", strict=False)
                .max()
                .alias(column)
            ).item()
        except Exception:
            continue
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
        if value not in (None, ""):
            return str(value)
    return ""


def _frame_text_value(frame: pl.DataFrame, column: str) -> str:
    if frame.is_empty() or column not in frame.columns:
        return ""
    try:
        values = [
            str(value)
            for value in frame.get_column(column).drop_nulls().unique().head(5).to_list()
            if str(value).strip()
        ]
    except Exception:
        return ""
    return ",".join(sorted(values))


def _frame_unique_texts(frame: pl.DataFrame, column: str) -> list[str]:
    if frame.is_empty() or column not in frame.columns:
        return []
    try:
        return [
            str(value).strip()
            for value in frame.get_column(column).drop_nulls().unique().to_list()
            if str(value).strip()
        ]
    except Exception:
        return []


def _frame_source_sha(
    dataset_name: str,
    frame: pl.DataFrame,
    generated_at: str,
    expires_at: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(dataset_name.encode("utf-8"))
    digest.update(str(frame.height).encode("ascii"))
    digest.update("|".join(frame.columns).encode("utf-8"))
    digest.update(str(generated_at or "").encode("utf-8"))
    digest.update(str(expires_at or "").encode("utf-8"))
    return digest.hexdigest()


def _dependency_source_sha(frames: dict[str, pl.DataFrame], *, telemetry_latest_ts: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(telemetry_latest_ts or "").encode("utf-8"))
    for dataset_name, frame in sorted(frames.items()):
        generated_at = _frame_latest_iso(
            frame,
            ("generated_at", "created_at", "as_of_ts", "date", "day"),
        )
        expires_at = _frame_latest_iso(frame, ("expires_at",))
        digest.update(dataset_name.encode("utf-8"))
        source_sha = _frame_source_sha(dataset_name, frame, generated_at, expires_at)
        digest.update(source_sha.encode("ascii"))
    return digest.hexdigest()


def _write_source_snapshot_metas(root: Path, frames: dict[str, pl.DataFrame]) -> list[str]:
    warnings: list[str] = []
    for dataset_name in sorted(SNAPSHOT_META_DATASETS):
        frame = frames.get(dataset_name, pl.DataFrame())
        dataset_path = root / readers.DATASET_PATHS.get(
            dataset_name,
            Path("gold") / dataset_name,
        )
        try:
            _write_snapshot_meta(dataset_path, dataset_name=dataset_name, frame=frame)
        except Exception as exc:
            warnings.append(
                f"{dataset_name} snapshot meta skipped: "
                f"{type(exc).__name__}:{_safe_warning_text(str(exc))}"
            )
    return warnings


def _publish_missed_opportunity_snapshot(
    root: Path,
    snapshot: _DatasetSnapshot,
) -> _DatasetSnapshot:
    frames = dict(snapshot.frames)
    opportunity = frames.get("strategy_opportunity_advisory", pl.DataFrame())
    audit = _v5_missed_opportunity_audit_for_export(
        candidate_events=frames.get("v5_candidate_event", pl.DataFrame()),
        v5_trades=frames.get("v5_trade_event", pl.DataFrame()),
        market_bars=frames.get("market_bar", pl.DataFrame()),
        opportunity_advisory=opportunity,
        market_regime=frames.get("market_regime_daily", pl.DataFrame()),
    )
    risk_on = _risk_on_multi_buy_shadow_for_export(
        candidate_events=frames.get("v5_candidate_event", pl.DataFrame()),
        v5_trades=frames.get("v5_trade_event", pl.DataFrame()),
        market_bars=frames.get("market_bar", pl.DataFrame()),
        market_regime=frames.get("market_regime_daily", pl.DataFrame()),
        cost_buckets=frames.get("cost_bucket_daily", pl.DataFrame()),
    )
    row_counts = dict(snapshot.row_counts)
    warnings = [
        warning
        for warning in snapshot.warnings
        if not warning.startswith("v5_missed_opportunity_audit dataset is ")
        and not warning.startswith("v5_risk_on_multi_buy_shadow dataset is ")
    ]
    _publish_export_frame(
        root,
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        dataset_name="v5_missed_opportunity_audit",
        frame=audit,
    )
    _publish_export_frame(
        root,
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        dataset_name="v5_risk_on_multi_buy_shadow",
        frame=risk_on,
    )
    _publish_export_frame(
        root,
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
        dataset_name="risk_on_multi_buy_shadow",
        frame=risk_on,
    )
    return _DatasetSnapshot(
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
    )


def _publish_research_portfolio_status_snapshot(
    root: Path,
    snapshot: _DatasetSnapshot,
    day: date,
) -> _DatasetSnapshot:
    frames = dict(snapshot.frames)
    row_counts = dict(snapshot.row_counts)
    warnings = list(snapshot.warnings)
    try:
        result = __import__(
            "quant_lab.research.portfolio",
            fromlist=["build_and_publish_research_portfolio_status"],
        ).build_and_publish_research_portfolio_status(root, as_of_date=day)
        frames["research_portfolio_status"] = dedupe_research_portfolio_status(
            read_parquet_dataset(root / "gold" / "research_portfolio_status")
        )
        row_counts["research_portfolio_status"] = result.rows_written
        warnings.extend(result.warnings)
    except Exception as exc:
        frames["research_portfolio_status"] = dedupe_research_portfolio_status(
            frames.get("research_portfolio_status", pl.DataFrame())
        )
        row_counts["research_portfolio_status"] = frames["research_portfolio_status"].height
        warnings.append(
            "research_portfolio_status publish skipped: "
            f"{type(exc).__name__}:{_safe_warning_text(str(exc))}"
        )
    return _DatasetSnapshot(
        frames=frames,
        row_counts=row_counts,
        warnings=warnings,
    )


def refresh_web_derived_snapshots(
    lake_root: str | Path,
    *,
    generated_at: datetime | None = None,
) -> WebDerivedSnapshotRefreshResult:
    """Refresh Web/expert-pack shared derived gold snapshots.

    Manual expert-pack generation publishes several display/audit tables that
    Web V2 also treats as first-class lake datasets. Keep those tables fresh
    from the normal read-only refresh path so the UI never depends on a manual
    ZIP export to clear stale-data warnings.
    """

    root = Path(lake_root)
    generated_at = generated_at or datetime.now(UTC)
    snapshot = _load_snapshot(root)
    snapshot = _publish_strategy_opportunity_advisory_snapshot(
        root,
        snapshot,
        generated_at=generated_at,
    )
    snapshot = _publish_missed_opportunity_snapshot(root, snapshot)
    snapshot = _publish_cost_bootstrap_readiness_snapshot(
        root,
        snapshot,
        generated_at=generated_at,
    )
    report_row_counts: dict[str, int] = {}
    try:
        write_enforce_readiness_report(root, strategy="v5", version="5.0.0")
        report_row_counts["v5_enforce_readiness"] = 1
    except Exception as exc:
        snapshot = _DatasetSnapshot(
            frames=snapshot.frames,
            row_counts=snapshot.row_counts,
            warnings=[
                *snapshot.warnings,
                "v5_enforce_readiness publish skipped: "
                f"{type(exc).__name__}:{_safe_warning_text(str(exc))}",
            ],
            transient_frames=snapshot.transient_frames,
        )
    try:
        fallback_breakdown = build_fallback_rate_breakdown(root)
        reports_dir = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / FALLBACK_RATE_BREAKDOWN_CSV).write_text(
            _csv_text(fallback_breakdown, fixed_columns=FALLBACK_RATE_BREAKDOWN_COLUMNS),
            encoding="utf-8",
        )
        report_row_counts["fallback_rate_breakdown"] = fallback_breakdown.height
    except Exception as exc:
        snapshot = _DatasetSnapshot(
            frames=snapshot.frames,
            row_counts=snapshot.row_counts,
            warnings=[
                *snapshot.warnings,
                "fallback_rate_breakdown publish skipped: "
                f"{type(exc).__name__}:{_safe_warning_text(str(exc))}",
            ],
            transient_frames=snapshot.transient_frames,
        )
    source_meta_warnings = _write_source_snapshot_metas(root, snapshot.frames)
    if source_meta_warnings:
        snapshot = _DatasetSnapshot(
            frames=snapshot.frames,
            row_counts=snapshot.row_counts,
            warnings=[*snapshot.warnings, *source_meta_warnings],
            transient_frames=snapshot.transient_frames,
        )
    return WebDerivedSnapshotRefreshResult(
        generated_at=generated_at,
        refreshed_datasets=list(WEB_DERIVED_SNAPSHOT_DATASETS),
        row_counts={
            dataset: int(snapshot.row_counts.get(dataset, 0))
            for dataset in WEB_DERIVED_SNAPSHOT_DATASETS
        }
        | report_row_counts,
        warnings=sorted(set(snapshot.warnings)),
    )


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
    allow_stale_v5: bool = False,
    expected_v5_bundle_sha256: str | None = None,
    acceptance_set_id: str | None = None,
) -> DailyExportResult:
    day = _parse_date(export_date)
    root = Path(lake_root)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(UTC)
    export_stage_timings: list[dict[str, Any]] = []
    stage_started = _export_stage_start("pre_export_v5")
    pre_export_v5 = (
        _refresh_v5_before_export(root, day, config_path=v5_telemetry_config)
        if pre_export_v5_refresh
        else _observe_v5_before_export(root, config_path=v5_telemetry_config)
    )
    pre_export_v5["allow_stale_v5"] = allow_stale_v5
    pre_export_v5["sync_attempted"] = bool(pre_export_v5_refresh)
    if expected_v5_bundle_sha256:
        pre_export_v5["expected_v5_bundle_sha256"] = str(expected_v5_bundle_sha256).strip()
    if acceptance_set_id:
        pre_export_v5["acceptance_set_id"] = str(acceptance_set_id).strip()
    _record_export_stage(export_stage_timings, "pre_export_v5", stage_started)
    stage_started = _export_stage_start("pre_export_risk_permission")
    pre_export_v5_warnings = [str(warning) for warning in pre_export_v5.get("warnings", [])]
    pre_export_risk_warnings = (
        _refresh_risk_permission_before_export(root, strategy=risk_strategy, version=risk_version)
        if refresh_risk_permission
        else []
    )
    _record_export_stage(export_stage_timings, "pre_export_risk_permission", stage_started)
    stage_started = _export_stage_start("load_snapshot")
    snapshot = _load_snapshot(root)
    _record_export_stage(export_stage_timings, "load_snapshot", stage_started)
    stage_started = _export_stage_start("v5_consistency")
    v5_consistency = _v5_export_consistency(
        snapshot.frames,
        pre_export_v5=pre_export_v5,
        pre_export_v5_refresh=pre_export_v5_refresh,
        allow_stale_v5=allow_stale_v5,
    )
    derived_outputs_stale = _v5_derived_outputs_stale(snapshot.frames, pre_export_v5)
    if (
        pre_export_v5_refresh
        and not allow_stale_v5
        and (v5_consistency["stale_v5_bundle"] or derived_outputs_stale)
    ):
        pre_export_v5["derived_refresh_retry_attempted"] = True
        retry_warnings = _refresh_v5_derived_outputs(root, day)
        if retry_warnings:
            pre_export_v5.setdefault("warnings", []).extend(
                f"pre_export_v5_derived_refresh_retry:{warning}" for warning in retry_warnings
            )
        snapshot = _load_snapshot(root)
        v5_consistency = _v5_export_consistency(
            snapshot.frames,
            pre_export_v5=pre_export_v5,
            pre_export_v5_refresh=pre_export_v5_refresh,
            allow_stale_v5=allow_stale_v5,
        )
    else:
        pre_export_v5["derived_refresh_retry_attempted"] = False
    pre_export_v5.update(v5_consistency)
    if (
        pre_export_v5_refresh
        and not allow_stale_v5
        and not v5_consistency["authoritative_snapshot"]
        and _v5_context_requires_authoritative_failure(pre_export_v5)
    ):
        raise RuntimeError(
            v5_consistency["warning_reason"]
            or v5_consistency["selected_v5_bundle_authoritative_reason"]
        )
    if v5_consistency["warning_reason"]:
        pre_export_v5.setdefault("warnings", []).append(v5_consistency["warning_reason"])
    _record_export_stage(export_stage_timings, "v5_consistency", stage_started)
    stage_started = _export_stage_start("github_ci_status")
    github_ci_status = _github_ci_status_for_export(pre_export_v5)
    pre_export_v5["github_ci_status_rows"] = github_ci_status.to_dicts()
    pre_export_v5["github_ci_status"] = _github_ci_status_summary(github_ci_status)
    _record_export_stage(export_stage_timings, "github_ci_status", stage_started)
    pre_export_v5_warnings = [str(warning) for warning in pre_export_v5.get("warnings", [])]
    stage_started = _export_stage_start("publish_research_portfolio_status")
    snapshot = _publish_research_portfolio_status_snapshot(root, snapshot, day)
    _record_export_stage(export_stage_timings, "publish_research_portfolio_status", stage_started)
    stage_started = _export_stage_start("publish_missed_opportunity")
    snapshot = _publish_missed_opportunity_snapshot(root, snapshot)
    _record_export_stage(export_stage_timings, "publish_missed_opportunity", stage_started)
    stage_started = _export_stage_start("publish_strategy_opportunity_advisory")
    snapshot = _publish_strategy_opportunity_advisory_snapshot(
        root,
        snapshot,
        generated_at=generated_at,
    )
    _record_export_stage(
        export_stage_timings,
        "publish_strategy_opportunity_advisory",
        stage_started,
    )
    stage_started = _export_stage_start("publish_paper_strategy_pipeline")
    snapshot = _publish_paper_strategy_pipeline_snapshot(root, snapshot, day)
    _record_export_stage(
        export_stage_timings,
        "publish_paper_strategy_pipeline",
        stage_started,
    )
    stage_started = _export_stage_start("publish_ops_truthfulness")
    snapshot = _publish_ops_truthfulness_snapshot(
        root,
        snapshot,
        generated_at=generated_at,
    )
    _record_export_stage(
        export_stage_timings,
        "publish_ops_truthfulness",
        stage_started,
    )
    stage_started = _export_stage_start("publish_trade_level_judgment")
    snapshot = _publish_trade_level_snapshot(
        root,
        snapshot,
        generated_at=generated_at,
    )
    _record_export_stage(export_stage_timings, "publish_trade_level_judgment", stage_started)
    stage_started = _export_stage_start("publish_risk_dependency_meta")
    snapshot = _publish_risk_permission_dependency_meta_snapshot(root, snapshot)
    _record_export_stage(export_stage_timings, "publish_risk_dependency_meta", stage_started)
    stage_started = _export_stage_start("publish_cost_bootstrap_readiness")
    snapshot = _publish_cost_bootstrap_readiness_snapshot(
        root,
        snapshot,
        generated_at=generated_at,
    )
    _record_export_stage(export_stage_timings, "publish_cost_bootstrap_readiness", stage_started)
    stage_started = _export_stage_start("write_source_snapshot_metas")
    meta_warnings = _write_source_snapshot_metas(root, snapshot.frames)
    if meta_warnings:
        snapshot = _DatasetSnapshot(
            frames=snapshot.frames,
            row_counts=snapshot.row_counts,
            warnings=[*snapshot.warnings, *meta_warnings],
            transient_frames=snapshot.transient_frames,
        )
    _record_export_stage(export_stage_timings, "write_source_snapshot_metas", stage_started)
    stage_started = _export_stage_start("data_quality_payload")
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
    _record_export_stage(export_stage_timings, "data_quality_payload", stage_started)
    stage_started = _export_stage_start("build_members")
    warnings = sorted(set([*snapshot.warnings, *pre_export_v5_warnings, *data_quality["warnings"]]))

    member_frames = {**snapshot.frames, **snapshot.transient_frames}
    members: dict[str, _MemberPayload] = {}
    members.update(
        _dataset_members(
            member_frames,
            root,
            data_quality=data_quality,
            pre_export_v5=pre_export_v5,
            row_counts=snapshot.row_counts,
            publish_warnings=warnings,
        )
    )
    members.update(_chart_members(member_frames))
    members.update(enforce_readiness_members(root))
    members[f"reports/{FALLBACK_RATE_BREAKDOWN_CSV}"] = _csv_member(
        f"reports/{FALLBACK_RATE_BREAKDOWN_CSV}",
        build_fallback_rate_breakdown(root),
    )
    _attach_selected_v5_bundle(members, pre_export_v5)
    _finalize_acceptance_set_context(pre_export_v5)
    members["reports/github_ci_status.csv"] = _csv_member(
        "reports/github_ci_status.csv",
        _github_ci_status_frame_from_context(pre_export_v5),
    )
    _promote_embedded_v5_summary_report(
        members,
        pre_export_v5,
        summary_member="summaries/paper_strategy_proposal_ack.csv",
        dest_member="reports/paper_strategy_proposal_ack.csv",
    )
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
            pre_export_v5=pre_export_v5,
        )
    )
    _record_export_stage(export_stage_timings, "build_members", stage_started)
    members["diagnostics/export_timing.csv"] = _csv_member(
        "diagnostics/export_timing.csv",
        _export_stage_timings_frame(export_stage_timings),
    )
    members["diagnostics/export_timing.json"] = _json_text(
        {
            "rows": export_stage_timings,
            "row_count": len(export_stage_timings),
            "generated_at": datetime.now(UTC).isoformat(),
        }
    )

    export_finished_at = datetime.now(UTC)
    manifest = _manifest_payload(
        day=day,
        generated_at=generated_at,
        export_finished_at=export_finished_at,
        root=root,
        profile=profile,
        missing_sections=missing_sections,
        warnings=warnings,
        members=members,
        row_counts=snapshot.row_counts,
        command_line=command_line or sys.argv,
        snapshot=snapshot,
        pre_export_v5=pre_export_v5,
        allow_stale_v5=allow_stale_v5,
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
    warnings.extend(_prune_old_export_packs(output_root, keep_count=_export_keep_pack_count()))
    index_warning = _write_export_index(
        output_root,
        latest_pack=zip_path,
        manifest=manifest,
        data_quality=data_quality,
        expert_questions=members["expert_questions.md"],
    )
    if index_warning:
        warnings.append(index_warning)
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


def _export_keep_pack_count() -> int:
    raw = os.environ.get("QUANT_LAB_EXPORT_KEEP_PACKS", str(DEFAULT_KEEP_EXPERT_PACKS))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_KEEP_EXPERT_PACKS
    return max(value, 1)


def _prune_old_export_packs(output_root: Path, *, keep_count: int) -> list[str]:
    if keep_count < 1 or not output_root.exists():
        return []
    try:
        packs = sorted(
            [path for path in output_root.glob("quant_lab_expert_pack_*.zip") if path.is_file()],
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
    except OSError as exc:
        return [f"export_pack_prune_failed:{exc}"]
    warnings: list[str] = []
    for pack in packs[keep_count:]:
        try:
            pack.unlink()
        except OSError as exc:
            warnings.append(f"export_pack_prune_failed:{pack.name}:{exc}")
    return warnings


def _write_export_index(
    output_root: Path,
    *,
    latest_pack: Path,
    manifest: dict[str, Any],
    data_quality: dict[str, Any],
    expert_questions: str,
) -> str | None:
    try:
        packs = sorted(
            [path for path in output_root.glob("quant_lab_expert_pack_*.zip") if path.is_file()],
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        authoritative_pack = _latest_authoritative_export_pack(packs)
        index_latest_pack = authoritative_pack or latest_pack
        if index_latest_pack == latest_pack:
            index_manifest = manifest
            index_data_quality = data_quality
            index_expert_questions = expert_questions
        else:
            index_manifest = _json_member_from_zip(index_latest_pack, "manifest.json")
            index_data_quality = _json_member_from_zip(index_latest_pack, "data_quality.json")
            index_expert_questions = _text_member_from_zip(
                index_latest_pack,
                "expert_questions.md",
            )
        pack_rows = []
        for path in packs:
            stat = path.stat()
            pack_manifest = (
                manifest
                if path == latest_pack
                else _json_member_from_zip(
                    path,
                    "manifest.json",
                )
            )
            pack_rows.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                    "authoritative_snapshot": bool(pack_manifest.get("authoritative_snapshot")),
                    "selected_v5_bundle_manifest_bundle_name": pack_manifest.get(
                        "selected_v5_bundle_manifest_bundle_name"
                    ),
                    "selected_v5_bundle_authoritative_reason": pack_manifest.get(
                        "selected_v5_bundle_authoritative_reason"
                    ),
                }
            )
        payload = {
            "schema_version": "quant_lab_export_index.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "latest_pack": str(index_latest_pack),
            "latest_pack_authoritative_snapshot": bool(
                index_manifest.get("authoritative_snapshot")
            ),
            "packs": pack_rows,
            "manifest_summary": index_manifest,
            "data_quality_summary": index_data_quality,
            "expert_questions": [
                line for line in index_expert_questions.splitlines() if line.strip()
            ][:20],
            "warnings": [],
        }
        tmp_path = output_root / ".export_index.json.tmp"
        index_path = output_root / "export_index.json"
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(index_path)
    except OSError as exc:
        return f"export_index_write_failed:{exc}"
    return None


def _latest_authoritative_export_pack(packs: list[Path]) -> Path | None:
    for path in packs:
        manifest = _json_member_from_zip(path, "manifest.json")
        if bool(manifest.get("authoritative_snapshot")):
            return path
    return None


def _json_member_from_zip(path: Path, member: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            payload = json.loads(archive.read(member).decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _text_member_from_zip(path: Path, member: str) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.read(member).decode("utf-8")
    except Exception:
        return ""


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
                embedded_path = manifest.get("embedded_v5_bundle_member_path")
                embedded_present = bool(manifest.get("embedded_v5_bundle_present"))
                if embedded_present:
                    if not embedded_path or str(embedded_path) not in names:
                        reasons.append("embedded V5 bundle is missing from zip")
                    else:
                        expected_sha = str(manifest.get("embedded_v5_bundle_sha256") or "").strip()
                        actual_sha = hashlib.sha256(archive.read(str(embedded_path))).hexdigest()
                        if expected_sha and actual_sha.lower() != expected_sha.lower():
                            reasons.append("embedded V5 bundle sha256 mismatch")
                acceptance_set_id = str(manifest.get("acceptance_set_id") or "").strip()
                if acceptance_set_id:
                    expected_sha = str(
                        manifest.get("expected_v5_bundle_sha256") or ""
                    ).strip().lower()
                    ingested_sha = str(
                        manifest.get("ingested_v5_bundle_sha256") or ""
                    ).strip().lower()
                    embedded_source_sha = str(
                        manifest.get("embedded_v5_bundle_source_sha256") or ""
                    ).strip().lower()
                    required = {
                        "expected_v5_bundle_sha256": expected_sha,
                        "ingested_v5_bundle_sha256": ingested_sha,
                        "embedded_v5_bundle_source_sha256": embedded_source_sha,
                        "quant_lab_commit": str(
                            manifest.get("quant_lab_commit") or ""
                        ).strip(),
                        "v5_commit": str(manifest.get("v5_commit") or "").strip(),
                    }
                    reasons.extend(
                        f"acceptance set missing required field: {name}"
                        for name, value in required.items()
                        if not value
                    )
                    if not (
                        expected_sha
                        and expected_sha == ingested_sha == embedded_source_sha
                    ):
                        reasons.append("acceptance set V5 bundle sha256 mismatch")
                    if manifest.get("acceptance_set_matched") is not True:
                        reasons.append("acceptance set is not marked matched")
                    attachment_manifest = _read_zip_json(
                        archive,
                        EMBEDDED_V5_BUNDLE_MANIFEST,
                        reasons,
                    )
                    if isinstance(attachment_manifest, dict):
                        attachment_source_sha = str(
                            attachment_manifest.get("source_sha256") or ""
                        ).strip().lower()
                        if attachment_source_sha != embedded_source_sha:
                            reasons.append(
                                "acceptance set embedded source sha256 mismatch"
                            )
            if isinstance(data_quality, dict):
                quality_status = str(data_quality.get("status") or "").upper()
                if quality_status in {"CRITICAL", "FAIL"}:
                    warnings.append(f"data_quality status is {quality_status}")
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
            empty_status = readers._empty_dataset_status(name)  # type: ignore[attr-defined]
            if empty_status in readers.OPTIONAL_EMPTY_DATASET_STATUSES:
                continue
            if _raw_export_rollup_name(lake_root, name):
                continue
            warnings.append(f"{name} dataset is {_missing_dataset_reason(name)}")
    return _DatasetSnapshot(frames=frames, row_counts=row_counts, warnings=warnings)


def _load_export_frame(
    lake_root: Path,
    dataset_name: str,
) -> tuple[pl.DataFrame, int, str | None]:
    dataset_path = readers.dataset_path_for(lake_root, dataset_name)
    rollup_name = _raw_export_rollup_name(lake_root, dataset_name)
    if rollup_name:
        row_count = _raw_export_row_count(
            dataset_path,
            allow_metadata_scan=dataset_name != "okx_public_ws",
        )
        warning = f"{dataset_name} export frame skipped; using {rollup_name} rollup"
        return pl.DataFrame(), row_count, warning
    if dataset_name not in HEAVY_EXPORT_DATASET_LIMITS and not _should_sample_export_dataset(
        dataset_path
    ):
        try:
            frame = readers.read_dataset(lake_root, dataset_name)
            return frame, frame.height, None
        except Exception as exc:
            return pl.DataFrame(), 0, f"{dataset_name} read failed: {exc}"

    try:
        limit = HEAVY_EXPORT_DATASET_LIMITS.get(dataset_name, DEFAULT_EXPORT_SAMPLED_ROW_LIMIT)
        max_files = HEAVY_EXPORT_RECENT_FILE_LIMITS.get(
            dataset_name,
            DEFAULT_EXPORT_RECENT_FILE_LIMIT,
        )
        all_files = (
            _export_parquet_files(dataset_path)
            if not dataset_path.is_file()
            else ([] if _is_internal_export_parquet_path(dataset_path) else [dataset_path])
        )
        recent_files = _recent_heavy_dataset_files(
            dataset_path,
            max_files=max_files,
        )
        frame = _collect_recent_heavy_files(
            recent_files,
            dataset_name,
            limit=limit,
        )
        row_count = _export_parquet_row_count(all_files, dataset_path=dataset_path)
        return frame, row_count, None
    except Exception as exc:
        return pl.DataFrame(), 0, f"{dataset_name} sampled read failed: {exc}"


def _raw_export_rollup_name(lake_root: Path, dataset_name: str) -> str | None:
    rollup_candidates = {
        "trade_print": ("trade_activity_1m",),
        "orderbook_snapshot": ("orderbook_spread_1m",),
        "okx_public_ws": (
            "okx_public_ws_health",
            "trade_activity_1m",
            "orderbook_spread_1m",
        ),
    }.get(dataset_name, ())
    for rollup_name in rollup_candidates:
        if readers.dataset_path_for(lake_root, rollup_name).exists():
            return rollup_name
    return None


def _raw_export_row_count(dataset_path: Path, *, allow_metadata_scan: bool) -> int:
    indexed_stats = _export_indexed_dataset_stats(dataset_path)
    if indexed_stats is not None:
        return indexed_stats[2]
    if not allow_metadata_scan:
        return 0
    return _export_parquet_row_count(
        _export_parquet_files(dataset_path),
        dataset_path=dataset_path,
    )


def _should_sample_export_dataset(dataset_path: Path) -> bool:
    if not dataset_path.exists():
        return False
    if dataset_path.is_file():
        if _is_internal_export_parquet_path(dataset_path):
            return False
        try:
            return dataset_path.stat().st_size > _export_full_read_max_bytes()
        except OSError:
            return True
    indexed_stats = _export_indexed_dataset_stats(dataset_path)
    if indexed_stats is not None:
        file_count, total_size, _row_count = indexed_stats
        return (
            file_count > _export_full_read_max_files() or total_size > _export_full_read_max_bytes()
        )
    files = _export_parquet_files(dataset_path)
    if len(files) > _export_full_read_max_files():
        return True
    total_size = 0
    for path in files:
        try:
            total_size += path.stat().st_size
        except OSError:
            return True
        if total_size > _export_full_read_max_bytes():
            return True
    return False


def _export_full_read_max_files() -> int:
    return max(
        int(os.environ.get("QUANT_LAB_EXPORT_FULL_READ_MAX_FILES", "80")),
        1,
    )


def _export_full_read_max_bytes() -> int:
    default_value = str(DEFAULT_EXPORT_FULL_READ_MAX_BYTES)
    return max(
        int(os.environ.get("QUANT_LAB_EXPORT_FULL_READ_MAX_BYTES", default_value)),
        1024,
    )


def _recent_heavy_dataset_files(dataset_path: Path, *, max_files: int) -> list[Path]:
    if not dataset_path.exists():
        return []
    if dataset_path.is_file():
        return [] if _is_internal_export_parquet_path(dataset_path) else [dataset_path]
    files = _export_parquet_files(dataset_path)
    if not files:
        return []
    files.sort(key=lambda path: path.stat().st_mtime)
    return files[-max_files:]


def _export_parquet_files(dataset_path: Path) -> list[Path]:
    indexed_files = _export_indexed_parquet_files(dataset_path)
    if indexed_files is not None:
        return indexed_files
    return sorted(
        path
        for path in dataset_path.rglob("*.parquet")
        if path.is_file() and not _is_internal_export_parquet_path(path)
    )


def _export_parquet_row_count(files: list[Path], *, dataset_path: Path | None = None) -> int:
    if not files:
        return 0
    indexed_count = _export_parquet_row_count_from_file_index(files, dataset_path=dataset_path)
    if indexed_count is not None:
        return indexed_count
    return _export_parquet_metadata_row_count(files)


def _export_parquet_row_count_from_file_index(
    files: list[Path],
    *,
    dataset_path: Path | None,
) -> int | None:
    if dataset_path is None:
        return None
    lake_root = _infer_lake_root(dataset_path)
    if lake_root is None:
        return None
    try:
        relative_dataset = str(dataset_path.relative_to(lake_root)).replace("\\", "/")
    except ValueError:
        return None
    index = _export_file_index(lake_root)
    if index is None:
        return None
    required = {"dataset", "path", "row_count", "mtime_ns", "file_size"}
    if index.is_empty() or not required.issubset(set(index.columns)):
        return None
    scoped = index.filter(pl.col("dataset") == relative_dataset)
    if scoped.is_empty():
        return None
    rows_by_path = {
        str(row.get("path") or ""): row for row in scoped.select(sorted(required)).to_dicts()
    }
    total = 0
    for file_path in files:
        try:
            relative_file = str(file_path.relative_to(lake_root)).replace("\\", "/")
            stat = file_path.stat()
        except OSError:
            return None
        except ValueError:
            return None
        row = rows_by_path.get(relative_file)
        if row is None:
            return None
        if _safe_int(row.get("mtime_ns")) != stat.st_mtime_ns:
            return None
        if _safe_int(row.get("file_size")) != stat.st_size:
            return None
        row_count = _safe_int(row.get("row_count"))
        if row_count is None:
            return None
        total += row_count
    return total


def _export_file_index(lake_root: Path) -> pl.DataFrame | None:
    root = lake_root.resolve()
    if root not in _EXPORT_FILE_INDEX_CACHE:
        index_path = root / LAKE_FILE_INDEX_DATASET
        try:
            _EXPORT_FILE_INDEX_CACHE[root] = _read_export_file_index(index_path)
        except Exception:
            _EXPORT_FILE_INDEX_CACHE[root] = None
    return _EXPORT_FILE_INDEX_CACHE[root]


def _read_export_file_index(index_path: Path) -> pl.DataFrame:
    if index_path.is_file():
        return pl.scan_parquet(str(index_path), hive_partitioning=False).collect()
    files = sorted(
        path
        for path in index_path.glob("*.parquet")
        if path.is_file() and not _is_internal_export_parquet_path(path)
    )
    if not files:
        return pl.DataFrame()
    return pl.scan_parquet(
        [str(path) for path in files],
        hive_partitioning=False,
        missing_columns="insert",
        extra_columns="ignore",
    ).collect()


def _export_index_rows_for_dataset(dataset_path: Path) -> tuple[Path, pl.DataFrame] | None:
    lake_root = _infer_lake_root(dataset_path)
    if lake_root is None:
        return None
    try:
        relative_dataset = str(dataset_path.relative_to(lake_root)).replace("\\", "/")
    except ValueError:
        return None
    index = _export_file_index(lake_root)
    required = {"dataset", "path", "row_count", "mtime_ns", "file_size"}
    if index is None or index.is_empty() or not required.issubset(set(index.columns)):
        return None
    scoped = index.filter(pl.col("dataset") == relative_dataset)
    if scoped.is_empty():
        return None
    return lake_root, scoped


def _export_indexed_parquet_files(dataset_path: Path) -> list[Path] | None:
    indexed = _export_index_rows_for_dataset(dataset_path)
    if indexed is None:
        return None
    lake_root, scoped = indexed
    rows = scoped.select(["path", "mtime_ns"]).sort(["mtime_ns", "path"]).to_dicts()
    files = []
    for row in rows:
        rel = str(row.get("path") or "")
        if not rel:
            continue
        path = lake_root / rel
        if path.is_file() and not _is_internal_export_parquet_path(path):
            files.append(path)
    return files


def _export_indexed_dataset_stats(dataset_path: Path) -> tuple[int, int, int] | None:
    indexed = _export_index_rows_for_dataset(dataset_path)
    if indexed is None:
        return None
    _lake_root, scoped = indexed
    try:
        stats = scoped.select(
            [
                pl.len().alias("file_count"),
                pl.col("file_size").cast(pl.Int64, strict=False).sum().alias("total_size"),
                pl.col("row_count").cast(pl.Int64, strict=False).sum().alias("row_count"),
            ]
        )
        return (
            int(stats.item(0, "file_count") or 0),
            int(stats.item(0, "total_size") or 0),
            int(stats.item(0, "row_count") or 0),
        )
    except Exception:
        return None


def _export_parquet_metadata_row_count(files: list[Path]) -> int:
    try:
        import pyarrow.parquet as pq

        return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in files)
    except Exception:
        pass

    try:
        return int(
            pl.scan_parquet(
                [str(path) for path in files],
                hive_partitioning=False,
                missing_columns="insert",
                extra_columns="ignore",
            )
            .select(pl.len().alias("rows"))
            .collect()
            .item()
        )
    except TypeError:
        try:
            return int(
                pl.scan_parquet([str(path) for path in files])
                .select(pl.len().alias("rows"))
                .collect()
                .item()
            )
        except Exception:
            pass
    except Exception:
        pass

    rows = 0
    for path in files:
        try:
            rows += int(pl.scan_parquet(str(path)).select(pl.len().alias("rows")).collect().item())
        except Exception:
            continue
    return rows


def _infer_lake_root(path: Path) -> Path | None:
    for idx, part in enumerate(path.parts):
        if part in {"bronze", "silver", "gold"} and idx > 0:
            return Path(*path.parts[:idx])
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_internal_export_parquet_path(path: Path) -> bool:
    return (
        any(part == "._tmp" or part.startswith("__") for part in path.parts)
        or path.name.startswith(".")
        or path.name.endswith(".tmp.parquet")
    )


def _collect_recent_heavy_files(
    files: list[Path],
    dataset_name: str,
    *,
    limit: int,
) -> pl.DataFrame:
    timestamp_columns = readers.DATASET_TIMESTAMP_COLUMNS.get(dataset_name, ())
    if files and timestamp_columns:
        try:
            scan = _scan_parquet_paths(files)
            columns = set(scan.collect_schema().names())
            timestamp_column = next(
                (column for column in timestamp_columns if column in columns),
                None,
            )
            if timestamp_column is not None:
                return scan.sort(timestamp_column).tail(limit).collect()
        except Exception:
            pass

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


def _scan_parquet_paths(files: list[Path]) -> pl.LazyFrame:
    paths = [str(path) for path in files]
    try:
        return pl.scan_parquet(
            paths,
            hive_partitioning=False,
            missing_columns="insert",
            extra_columns="ignore",
        )
    except TypeError:
        return pl.scan_parquet(paths)


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


def _paper_tracking_frames_for_export(
    frames: dict[str, pl.DataFrame],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    v5_runs_raw = latest_v5_paper_frame(frames.get("v5_paper_strategy_run", pl.DataFrame()))
    v5_daily_raw = latest_v5_paper_frame(frames.get("v5_paper_strategy_daily", pl.DataFrame()))
    v5_slippage_raw = latest_v5_paper_frame(
        frames.get("v5_paper_slippage_coverage", pl.DataFrame())
    )
    v5_candidate_raw = latest_v5_paper_frame(frames.get("v5_candidate_event", pl.DataFrame()))
    if any(
        not frame.is_empty()
        for frame in [v5_runs_raw, v5_daily_raw, v5_slippage_raw, v5_candidate_raw]
    ):
        research_portfolio = frames.get("research_portfolio_status", pl.DataFrame())
        report_runs = build_paper_strategy_runs_report_from_v5(
            v5_runs_raw,
            research_portfolio=research_portfolio,
            candidate_events=v5_candidate_raw,
        )
        runs = build_paper_strategy_runs_from_v5(
            v5_runs_raw,
            research_portfolio=research_portfolio,
            candidate_events=v5_candidate_raw,
        )
        export_day = _latest_paper_tracking_date(
            [runs, v5_daily_raw, v5_slippage_raw],
        )
        daily = build_paper_strategy_daily_from_v5(v5_daily_raw)
        if not runs.is_empty():
            daily = enrich_paper_strategy_daily_from_runs(
                daily,
                runs,
                as_of_date=export_day,
            )
        if daily.is_empty() and not runs.is_empty():
            daily = build_paper_strategy_daily_from_runs(runs, as_of_date=export_day)
        slippage = build_paper_slippage_coverage_from_v5(v5_slippage_raw, daily=daily)
        if slippage.is_empty() and not daily.is_empty():
            slippage = build_paper_slippage_coverage(daily, as_of_date=export_day)
        return report_runs if not report_runs.is_empty() else runs, daily, slippage

    return (
        frames.get("paper_strategy_runs", pl.DataFrame()),
        frames.get("paper_strategy_daily", pl.DataFrame()),
        frames.get("paper_slippage_coverage", pl.DataFrame()),
    )


def _filter_bnb_paper_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    strategy_columns = [
        column for column in ["proposal_id", "strategy_id"] if column in frame.columns
    ]
    if not strategy_columns:
        return frame.head(0)
    predicate = None
    bnb_ids = {"BNB_F3_DOMINANT_ENTRY_PAPER_V1", "BNB_RISK_ON_BUY_PAPER_V1"}
    for column in strategy_columns:
        expr = pl.col(column).cast(pl.Utf8, strict=False).is_in(bnb_ids)
        predicate = expr if predicate is None else predicate | expr
    if "symbol" in frame.columns:
        symbol_expr = (
            pl.col("symbol").cast(pl.Utf8, strict=False).str.to_uppercase().str.replace("-", "/")
            == "BNB/USDT"
        )
        predicate = predicate & symbol_expr if predicate is not None else symbol_expr
    return frame.filter(predicate) if predicate is not None else frame.head(0)


def _trade_activity_export_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    if "trade_count" in frame.columns:
        aggregations = [
            pl.col("trade_count").cast(pl.Int64, strict=False).sum().alias("trade_count")
        ]
        if "size_sum" in frame.columns:
            aggregations.append(
                pl.col("size_sum").cast(pl.Float64, strict=False).sum().alias("size_sum")
            )
        if "latest_trade_ts" in frame.columns:
            aggregations.append(pl.col("latest_trade_ts").max().alias("latest_trade_ts"))
        elif "minute_ts" in frame.columns:
            aggregations.append(pl.col("minute_ts").max().alias("latest_trade_ts"))
        return frame.group_by("symbol").agg(aggregations).sort("symbol")
    return readers.trade_activity_table(frame)


def _orderbook_spread_export_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    if (
        "spread_bps" in frame.columns
        and "asks_json" not in frame.columns
        and "bids_json" not in frame.columns
    ):
        working = frame
        if "ts" not in working.columns and "minute_ts" in working.columns:
            working = working.rename({"minute_ts": "ts"})
        sort_columns = [
            column for column in ["symbol", "channel", "ts"] if column in working.columns
        ]
        if sort_columns:
            group_columns = [
                column for column in ["symbol", "channel"] if column in working.columns
            ]
            if group_columns:
                return (
                    working.sort(sort_columns).group_by(group_columns, maintain_order=True).tail(1)
                )
        return working
    return readers.orderbook_spread_table(frame)


def _latest_paper_tracking_date(frames: list[pl.DataFrame]) -> date:
    values: list[str] = []
    for frame in frames:
        if frame.is_empty() or "as_of_date" not in frame.columns:
            continue
        values.extend(
            str(value)[:10]
            for value in frame.get_column("as_of_date").drop_nulls().to_list()
            if str(value).strip()
        )
    if not values:
        return datetime.now(UTC).date()
    return date.fromisoformat(max(values))


def _dataset_members(
    frames: dict[str, pl.DataFrame],
    root: Path,
    *,
    data_quality: dict[str, Any] | None = None,
    pre_export_v5: dict[str, Any] | None = None,
    row_counts: dict[str, int] | None = None,
    publish_warnings: list[str] | None = None,
) -> dict[str, _MemberPayload]:
    market = frames.get("market_bar", pl.DataFrame())
    features = frames.get("feature_value", pl.DataFrame())
    feature_coverage = frames.get("feature_coverage_daily", pl.DataFrame())
    feature_anomalies = frames.get("feature_anomaly_daily", pl.DataFrame())
    factor_definitions = frames.get("factor_definition", pl.DataFrame())
    factor_values = frames.get("factor_value", pl.DataFrame())
    factor_evidence = frames.get("factor_evidence", pl.DataFrame())
    factor_candidates = frames.get("factor_candidate", pl.DataFrame())
    factor_correlations = frames.get("factor_correlation_daily", pl.DataFrame())
    costs = _normalize_symbol_frame(frames.get("cost_bucket_daily", pl.DataFrame()))
    market_regime = frames.get("market_regime_daily", pl.DataFrame())
    factor_forward_validation = build_factor_forward_validation(
        factor_candidates=factor_candidates,
        factor_values=factor_values,
        market_bars=market,
        market_regime=market_regime,
        cost_bucket_daily=costs,
    )
    live_universe_cost_coverage = build_live_universe_cost_coverage(costs)
    cost_bootstrap_readiness = _prefer_frame(
        frames.get("cost_bootstrap_readiness", pl.DataFrame()),
        build_cost_bootstrap_readiness(
            costs,
            v5_order_lifecycle=_read_optional_lake_frame(root / "silver" / "v5_order_lifecycle"),
            v5_cost_probe_order_events=frames.get("v5_cost_probe_order_event", pl.DataFrame()),
            v5_cost_probe_roundtrip_events=frames.get(
                "v5_cost_probe_roundtrip_event",
                pl.DataFrame(),
            ),
            okx_private_readonly_fills=frames.get("okx_private_readonly_fills", pl.DataFrame()),
            okx_private_readonly_bills=frames.get("okx_private_readonly_bills", pl.DataFrame()),
        ),
    )
    cost_probe_fill_bill_match = build_cost_probe_fill_bill_match(
        frames.get("v5_cost_probe_order_event", pl.DataFrame()),
        frames.get("v5_cost_probe_roundtrip_event", pl.DataFrame()),
        frames.get("okx_private_readonly_fills", pl.DataFrame()),
        frames.get("okx_private_readonly_bills", pl.DataFrame()),
    )
    cost_probe_cost_disagreement = build_cost_probe_cost_disagreement(
        costs,
        frames.get("v5_cost_probe_order_event", pl.DataFrame()),
        frames.get("v5_cost_probe_roundtrip_event", pl.DataFrame()),
        frames.get("okx_private_readonly_fills", pl.DataFrame()),
        frames.get("okx_private_readonly_bills", pl.DataFrame()),
    )
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
    second_stage_summary = _strategy_evidence_for_export(
        frames.get("second_stage_alpha_factory_summary", pl.DataFrame())
    )
    second_stage_samples = _strategy_evidence_samples_for_export(
        frames.get("second_stage_alpha_factory_sample", pl.DataFrame())
    )
    relative_strength_decision_samples = frames.get(
        "expanded_relative_strength_decision_sample",
        pl.DataFrame(),
    )
    alpha_factory_registry = frames.get("alpha_factory_template_registry", pl.DataFrame())
    alpha_factory_candidates = frames.get("alpha_factory_candidate", pl.DataFrame())
    alpha_factory_results = frames.get("alpha_factory_result", pl.DataFrame())
    alpha_factory_promotion = frames.get("alpha_factory_promotion_queue", pl.DataFrame())
    research_portfolio = dedupe_research_portfolio_status(
        frames.get("research_portfolio_status", pl.DataFrame())
    )
    telemetry_latest_ts = _latest_v5_bundle_ts(frames)
    risk = _risk_permissions_for_export(
        frames.get("risk_permission", pl.DataFrame()),
        frames,
        telemetry_latest_ts=telemetry_latest_ts,
    )
    export_generated_at = (data_quality or {}).get("generated_at") or datetime.now(UTC)
    export_reference_at = _parse_export_ts(export_generated_at) or datetime.now(UTC)
    paper_runs, paper_daily, paper_slippage = _paper_tracking_frames_for_export(frames)
    bnb_paper_runs = _prefer_frame(
        frames.get("v5_bnb_paper_strategy_runs", pl.DataFrame()),
        _filter_bnb_paper_frame(paper_runs),
    )
    v5_bnb_paper_daily = _prefer_frame(
        frames.get("v5_bnb_paper_strategy_daily_latest", pl.DataFrame()),
        frames.get("v5_bnb_paper_strategy_daily", pl.DataFrame()),
    )
    bnb_paper_daily = _prefer_frame(
        v5_bnb_paper_daily,
        _filter_bnb_paper_frame(paper_daily),
    )
    sol_protect_loss_attribution = frames.get(
        "sol_protect_paper_loss_attribution",
        pl.DataFrame(),
    )
    sol_protect_loss_summary = frames.get("sol_protect_paper_loss_summary", pl.DataFrame())
    missed_low_audit = frames.get("v5_missed_low_audit", pl.DataFrame())
    missed_low_by_symbol = frames.get("v5_missed_low_by_symbol", pl.DataFrame())
    missed_low_by_entry_reason = frames.get("v5_missed_low_by_entry_reason", pl.DataFrame())
    late_entry_chase_shadow = frames.get("v5_late_entry_chase_shadow", pl.DataFrame())
    late_entry_threshold = frames.get(
        "v5_late_entry_chase_threshold_advisory",
        pl.DataFrame(),
    )
    late_entry_threshold_by_symbol = frames.get(
        "v5_late_entry_chase_threshold_by_symbol",
        pl.DataFrame(),
    )
    pullback_reversal_shadow = frames.get("v5_pullback_reversal_shadow", pl.DataFrame())
    pullback_reversal_readiness = frames.get(
        "v5_pullback_reversal_readiness",
        pl.DataFrame(),
    )
    pullback_reversal_rule_comparison = frames.get(
        "v5_pullback_reversal_rule_comparison",
        pl.DataFrame(),
    )
    btc_probe_exit_policy_review = frames.get("btc_probe_exit_policy_review", pl.DataFrame())
    btc_probe_exit_policy_summary = frames.get("btc_probe_exit_policy_summary", pl.DataFrame())
    bnb_swing_exit_policy_review = frames.get("bnb_swing_exit_policy_review", pl.DataFrame())
    bnb_negative_expectancy_attribution = frames.get(
        "v5_bnb_negative_expectancy_attribution",
        pl.DataFrame(),
    )
    negative_expectancy_attribution = _prefer_frame(
        frames.get("v5_negative_expectancy_attribution", pl.DataFrame()),
        bnb_negative_expectancy_attribution,
    )
    bnb_exit_policy_v5_vs_quant_lab_consistency = frames.get(
        "bnb_exit_policy_v5_vs_quant_lab_consistency",
        pl.DataFrame(),
    )
    bnb_swing_exit_policy_summary = frames.get("bnb_swing_exit_policy_summary", pl.DataFrame())
    exit_policy_review_sample = frames.get("exit_policy_review_sample", pl.DataFrame())
    exit_policy_review_summary = frames.get("exit_policy_review_summary", pl.DataFrame())
    entry_quality_advisory = frames.get("v5_entry_quality_advisory", pl.DataFrame())
    history_late_threshold = frames.get(
        "v5_entry_quality_history_late_entry_chase_threshold_sensitivity",
        pl.DataFrame(),
    )
    history_pullback_by_symbol = frames.get(
        "v5_entry_quality_history_pullback_by_symbol",
        pl.DataFrame(),
    )
    history_pullback_by_regime = frames.get(
        "v5_entry_quality_history_pullback_by_regime",
        pl.DataFrame(),
    )
    history_pullback_by_horizon = frames.get(
        "v5_entry_quality_history_pullback_by_horizon",
        pl.DataFrame(),
    )
    if history_pullback_by_symbol.is_empty() and not pullback_reversal_shadow.is_empty():
        history_pullback_by_symbol = _pullback_shadow_aggregate_for_export(
            pullback_reversal_shadow,
            group_type="symbol",
            path="reports/pullback_reversal_by_symbol.csv",
        )
    if history_pullback_by_regime.is_empty() and not pullback_reversal_shadow.is_empty():
        history_pullback_by_regime = _pullback_shadow_aggregate_for_export(
            pullback_reversal_shadow,
            group_type="regime",
            path="reports/pullback_reversal_by_regime.csv",
        )
    if history_pullback_by_horizon.is_empty() and not pullback_reversal_shadow.is_empty():
        history_pullback_by_horizon = _pullback_shadow_aggregate_for_export(
            pullback_reversal_shadow,
            group_type="horizon",
            path="reports/pullback_reversal_by_horizon.csv",
        )
    history_anti_leakage = frames.get(
        "v5_entry_quality_history_anti_leakage_check",
        pl.DataFrame(),
    )
    history_metrics = frames.get("v5_entry_quality_history_metrics", pl.DataFrame())
    paper_proposals = _paper_strategy_proposals_for_export(
        alpha_discovery_board,
        research_portfolio=research_portfolio,
        persisted_proposals=frames.get("paper_strategy_proposal", pl.DataFrame()),
    )
    paper_proposals_current = _prefer_frame(
        frames.get("paper_strategy_proposals_current", pl.DataFrame()),
        paper_proposals,
    )
    paper_pipeline = build_paper_strategy_pipeline_frames(
        proposals=paper_proposals_current,
        proposal_ack=frames.get("v5_paper_strategy_proposal_ack", pl.DataFrame()),
        runs=paper_runs,
        daily=paper_daily,
        strategy_cost_trust=frames.get("strategy_cost_trust", pl.DataFrame()),
        created_at=export_reference_at,
    )
    persisted_registry_current = frames.get(
        "paper_strategy_registry_current", pl.DataFrame()
    )
    paper_strategy_registry = (
        _csv_frame_with_schema(
            persisted_registry_current,
            CSV_SCHEMAS["reports/paper_strategy_registry.csv"],
        )
        if not persisted_registry_current.is_empty()
        else _stable_paper_strategy_lifecycle_frame(
            paper_pipeline["paper_strategy_registry"],
            frames.get("paper_strategy_registry", pl.DataFrame()),
            schema=CSV_SCHEMAS["reports/paper_strategy_registry.csv"],
        )
    )
    persisted_gate = frames.get("paper_strategy_promotion_gate", pl.DataFrame())
    paper_strategy_promotion_gate = (
        _csv_frame_with_schema(
            persisted_gate,
            CSV_SCHEMAS["reports/paper_strategy_promotion_gate.csv"],
        )
        if not persisted_gate.is_empty()
        else paper_pipeline["paper_strategy_promotion_gate"]
    )
    paper_strategy_registry_history = frames.get(
        "paper_strategy_registry_history", paper_strategy_registry
    )
    paper_strategy_trackers_current = frames.get(
        "paper_strategy_trackers_current", pl.DataFrame()
    )
    paper_strategy_identity_conflict = frames.get(
        "paper_strategy_identity_conflict", pl.DataFrame()
    )
    paper_cohort_manifest = frames.get("paper_cohort_manifest", pl.DataFrame())
    raw_paper_strategy_ack_current = frames.get(
        "paper_strategy_ack_current",
        frames.get("v5_paper_strategy_proposal_ack", pl.DataFrame()),
    )
    paper_strategy_proposal_ack_current = _paper_strategy_proposal_ack_for_export(
        raw_paper_strategy_ack_current,
        paper_strategy_registry=pl.DataFrame(),
    )
    paper_strategy_proposal_ack = _paper_strategy_proposal_ack_for_export(
        raw_paper_strategy_ack_current,
        paper_strategy_registry=paper_strategy_registry,
    )
    paper_strategy_ack_history = _paper_strategy_proposal_ack_for_export(
        frames.get("paper_strategy_ack_history", pl.DataFrame()),
        paper_strategy_registry=paper_strategy_registry_history,
    )
    paper_strategy_migration_audit = frames.get(
        "paper_strategy_migration_audit",
        pl.DataFrame(),
    )
    strategy_regime_matrix = frames.get("strategy_regime_matrix", pl.DataFrame())
    regime_strategy_advisory = frames.get("regime_strategy_advisory", pl.DataFrame())
    missed_opportunity_audit = frames.get("v5_missed_opportunity_audit", pl.DataFrame())
    risk_on_multi_buy_shadow = frames.get("v5_risk_on_multi_buy_shadow", pl.DataFrame())
    expanded_maturity = frames.get("expanded_universe_candidate_maturity", pl.DataFrame())
    bnb_missed_opportunity = _bnb_missed_opportunity_samples_for_export(
        candidate_events=frames.get("v5_candidate_event", pl.DataFrame()),
        market_bars=market,
    )
    final_score_alpha6_conflict = _prefer_frame(
        _normalize_final_score_alpha6_conflict_frame(
            frames.get("v5_final_score_vs_alpha6_conflict", pl.DataFrame())
        ),
        _final_score_vs_alpha6_conflict_for_export(
            candidate_events=frames.get("v5_candidate_event", pl.DataFrame()),
            market_bars=market,
            negative_expectancy=frames.get("v5_negative_expectancy_consistency", pl.DataFrame()),
        ),
    )
    bnb_strong_alpha6_bypass_shadow = _normalize_bnb_strong_alpha6_bypass_frame(
        frames.get("v5_bnb_strong_alpha6_bypass_shadow", pl.DataFrame())
    )
    post_impulse_overextension_shadow = _post_impulse_overextension_shadow_for_export(
        candidate_events=frames.get("v5_candidate_event", pl.DataFrame()),
        market_bars=market,
    )
    late_breakout_failure_shadow = _late_breakout_failure_shadow_for_export(
        post_impulse_overextension_shadow
    )
    trades = _normalize_symbol_frame(
        _prefer_frame(
            frames.get("trade_activity_1m", pl.DataFrame()),
            frames.get("trade_print", pl.DataFrame()),
        )
    )
    books = _normalize_symbol_frame(
        _prefer_frame(
            frames.get("orderbook_spread_1m", pl.DataFrame()),
            frames.get("orderbook_snapshot", pl.DataFrame()),
        )
    )
    fast_microstructure_features = build_fast_microstructure_features(
        market_bars=market,
        orderbook_spread_1m=books,
        trade_activity_1m=trades,
        generated_at=datetime.now(UTC),
    )
    bottom_zone_reversal_shadow = frames.get("bottom_zone_reversal_shadow", pl.DataFrame())
    if bottom_zone_reversal_shadow.is_empty():
        bottom_zone_reversal_shadow = build_bottom_zone_reversal_shadow(
            market_bars=market,
            orderbook_spread_1m=books,
            trade_activity_1m=trades,
            generated_at=datetime.now(UTC),
        )
    fast_microstructure_forward_test = frames.get(
        "fast_microstructure_forward_test",
        pl.DataFrame(),
    )
    if fast_microstructure_forward_test.is_empty():
        fast_microstructure_forward_test = build_fast_microstructure_forward_test(
            market_bars=market,
            orderbook_spread_1m=books,
            trade_activity_1m=trades,
            market_regime=market_regime,
            cost_bucket_daily=costs,
            generated_at=datetime.now(UTC),
        )
    fast_microstructure_strategy_candidates = build_fast_microstructure_strategy_candidates(
        fast_microstructure_forward_test
    )
    fast_microstructure_strategy_review = build_fast_microstructure_strategy_review(
        fast_microstructure_strategy_candidates,
        generated_at=export_generated_at,
    )
    factor_v2 = build_factor_factory_v2_reports(
        candidates=factor_candidates,
        evidence=factor_evidence,
        correlations=factor_correlations,
        factor_forward_validation=factor_forward_validation,
        fast_microstructure_forward_test=fast_microstructure_forward_test,
    )
    factor_strategy_bridge_candidates = frames.get(
        "factor_strategy_bridge_candidates",
        pl.DataFrame(),
    )
    if factor_strategy_bridge_candidates.is_empty():
        factor_strategy_bridge_candidates = factor_v2["factor_strategy_bridge_candidates"]
    opportunity_advisory = frames.get("strategy_opportunity_advisory", pl.DataFrame())
    if opportunity_advisory.is_empty():
        opportunity_advisory = _strategy_opportunity_advisory_for_export(
            alpha_discovery_board=alpha_discovery_board,
            strategy_evidence=strategy_evidence,
            paper_proposals=paper_proposals,
            risk_permissions=risk,
            cost_health=cost_health,
            paper_daily=paper_daily,
            paper_slippage=paper_slippage,
            research_portfolio=research_portfolio,
            entry_quality_advisory=entry_quality_advisory,
            regime_strategy_advisory=regime_strategy_advisory,
            risk_on_multi_buy_shadow=risk_on_multi_buy_shadow,
            alpha_factory_results=frames.get("alpha_factory_result", pl.DataFrame()),
            alpha_factory_promotion_queue=alpha_factory_promotion,
            expanded_universe_maturity=expanded_maturity,
            bottom_zone_reversal_shadow=bottom_zone_reversal_shadow,
            factor_strategy_bridge_candidates=factor_strategy_bridge_candidates,
        )
    strategy_level_dashboard = _strategy_level_dashboard_for_export(opportunity_advisory)
    gates = _gate_decisions_for_export(frames.get("gate_decision", pl.DataFrame()))
    market_pressure_score = build_market_pressure_score(
        bottom_zone_reversal_shadow=bottom_zone_reversal_shadow,
        fast_microstructure_features=fast_microstructure_features,
        generated_at=datetime.now(UTC),
    )
    bottom_zone_probe_paper_readiness = _bottom_zone_probe_paper_readiness_for_export(
        paper_runs=paper_runs,
        paper_daily=paper_daily,
        generated_at=datetime.now(UTC),
    )
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
    v5_fill_bill_cost_reconciliation = frames.get(
        "v5_fill_bill_cost_reconciliation",
        pl.DataFrame(),
    )
    v5_paper_strategy_proposal_ack = frames.get(
        "v5_paper_strategy_proposal_ack",
        pl.DataFrame(),
    )
    v5_paper_strategy_registry = frames.get(
        "v5_paper_strategy_registry",
        pl.DataFrame(),
    )
    v5_paper_strategy_state = frames.get("v5_paper_strategy_state", pl.DataFrame())
    v5_paper_strategy_signals = frames.get(
        "v5_paper_strategy_signal",
        pl.DataFrame(),
    )
    v5_paper_strategy_runs = frames.get("v5_paper_strategy_run", pl.DataFrame())
    v5_paper_strategy_exit_quality = frames.get(
        "v5_paper_strategy_exit_quality",
        pl.DataFrame(),
    )
    v5_trade_opportunity_funnel = frames.get(
        "v5_trade_opportunity_funnel",
        pl.DataFrame(),
    )
    v5_paper_strategy_daily = frames.get("v5_paper_strategy_daily", pl.DataFrame())
    v5_paper_strategy_quote_coverage = frames.get(
        "v5_paper_strategy_quote_coverage",
        pl.DataFrame(),
    )
    v5_paper_strategy_cost_evidence = frames.get(
        "v5_paper_strategy_cost_evidence",
        pl.DataFrame(),
    )
    v5_paper_strategy_errors = frames.get("v5_paper_strategy_error", pl.DataFrame())
    v5_paper_strategy_restart_recovery = frames.get(
        "v5_paper_strategy_restart_recovery",
        pl.DataFrame(),
    )
    v5_quant_lab_contract_status = frames.get(
        "v5_quant_lab_contract_status",
        pl.DataFrame(),
    )
    v5_cost_probe_p3_preflight = frames.get("v5_cost_probe_p3_preflight", pl.DataFrame())
    v5_cost_probe_live_execution_status = _dedupe_stable_row_key_frame(
        frames.get(
            "v5_cost_probe_live_execution_status",
            pl.DataFrame(),
        )
    )
    v5_cost_probe_live_execution_status = _with_cost_probe_authorization_export_freshness(
        v5_cost_probe_live_execution_status,
        reference_at=export_reference_at,
    )
    v5_cost_probe_live_execution_status_canonical = canonical_cost_probe_live_execution_status(
        v5_cost_probe_live_execution_status
    )
    v5_cost_probe_order_events = frames.get("v5_cost_probe_order_event", pl.DataFrame())
    v5_cost_probe_roundtrip_events = frames.get(
        "v5_cost_probe_roundtrip_event",
        pl.DataFrame(),
    )
    v5_cost_probe_roundtrip_canonical = canonical_cost_probe_roundtrip_events(
        v5_cost_probe_roundtrip_events
    )
    v5_candidate_events = frames.get("v5_candidate_event", pl.DataFrame())
    v5_btc_probe_entry_quality_audit = frames.get(
        "v5_btc_probe_entry_quality_audit",
        pl.DataFrame(),
    )
    v5_candidate_labels = frames.get("v5_candidate_label", pl.DataFrame())
    trade_opportunity_events = frames.get("trade_opportunity_event", pl.DataFrame())
    trade_opportunity_labels = frames.get("trade_opportunity_label", pl.DataFrame())
    trade_level_similarity = frames.get("trade_level_similarity_outcome", pl.DataFrame())
    trade_level_judgments = frames.get("trade_level_judgment", pl.DataFrame())
    trade_level_bucket_policy = frames.get("trade_level_bucket_policy", pl.DataFrame())
    trade_level_opportunity_queue = frames.get("trade_level_opportunity_queue", pl.DataFrame())
    false_block_audit = frames.get("quant_lab_false_block_audit", pl.DataFrame())
    v5_trade_learning_samples = frames.get("v5_trade_learning_sample", pl.DataFrame())
    v5_trade_outcome_attribution = frames.get("v5_trade_outcome_attribution", pl.DataFrame())
    opportunity_cost_events = frames.get("quant_lab_opportunity_cost_event", pl.DataFrame())
    opportunity_cost_daily = frames.get("quant_lab_opportunity_cost_daily", pl.DataFrame())
    opportunity_cost_by_bucket = frames.get("opportunity_cost_by_bucket", pl.DataFrame())
    decision_regret = frames.get("quant_lab_decision_regret", pl.DataFrame())
    gate_candidate_effectiveness = build_gate_effectiveness_report(
        opportunity_cost_events,
        evaluation_level="candidate_event",
        dedupe_field="strategy_evaluation_id",
    )
    gate_independent_event_effectiveness = build_gate_effectiveness_report(
        opportunity_cost_events,
        evaluation_level="independent_market_event",
        dedupe_field="canonical_event_id",
    )
    gate_paper_trade_effectiveness = build_gate_effectiveness_report(
        paper_runs,
        evaluation_level="paper_trade",
        dedupe_field="paper_trade_id",
    )
    gate_live_trade_effectiveness = build_gate_effectiveness_report(
        v5_trade_outcome_attribution,
        evaluation_level="v5_live_trade",
        dedupe_field="roundtrip_id",
    )
    duplicate_event_report = _duplicate_event_report(opportunity_cost_events)
    duplicate_factor_report = _duplicate_factor_report(factor_definitions)
    strategy_cost_trust = frames.get("strategy_cost_trust", pl.DataFrame())
    v5_candidate_quality = frames.get("v5_candidate_quality_daily", pl.DataFrame())
    v5_candidate_outcomes = frames.get("v5_candidate_outcome_summary", pl.DataFrame())
    v5_expanded_universe_advisory_reader = frames.get(
        "v5_expanded_universe_advisory_reader",
        pl.DataFrame(),
    )
    v5_expanded_universe_paper_runs = frames.get(
        "v5_expanded_universe_paper_runs",
        pl.DataFrame(),
    )
    v5_expanded_universe_paper_daily = frames.get(
        "v5_expanded_universe_paper_daily",
        pl.DataFrame(),
    )
    expanded_candidates = frames.get("expanded_universe_candidate", pl.DataFrame())
    expanded_quality = frames.get("expanded_universe_quality", pl.DataFrame())
    expanded_events = frames.get("expanded_universe_candidate_event", pl.DataFrame())
    expanded_labels = frames.get("expanded_universe_candidate_label", pl.DataFrame())
    expanded_promotion = frames.get("expanded_universe_promotion_queue", pl.DataFrame())
    expanded_watchlist = frames.get("expanded_universe_watchlist", pl.DataFrame())
    expanded_universe = frames.get("expanded_crypto_universe_shadow", pl.DataFrame())
    symbol_quality = frames.get("symbol_quality_score", pl.DataFrame())
    expanded_outcomes = frames.get(
        "expanded_crypto_candidate_outcomes_by_symbol",
        pl.DataFrame(),
    )
    expanded_recommendations = frames.get("expanded_crypto_recommendations", pl.DataFrame())
    v5_trades_for_shadow = frames.get("v5_trade_event", pl.DataFrame())
    local_live_vs_shadow = _v5_local_live_vs_quant_lab_shadow_for_export(
        v5_trades=v5_trades_for_shadow,
        market_bars=market,
        opportunity_advisory=opportunity_advisory,
    )
    backtest_frames = dict(frames)
    backtest_frames.update(
        {
            "market_bar": market,
            "cost_bucket_daily": costs,
            "strategy_opportunity_advisory": opportunity_advisory,
            "risk_on_multi_buy_shadow": risk_on_multi_buy_shadow,
            "final_score_vs_alpha6_conflict": final_score_alpha6_conflict,
            "bnb_strong_alpha6_bypass_shadow": bnb_strong_alpha6_bypass_shadow,
            "post_impulse_overextension_shadow": post_impulse_overextension_shadow,
            "bottom_zone_reversal_shadow": bottom_zone_reversal_shadow,
            "v5_candidate_event": v5_candidate_events,
            "v5_candidate_label": v5_candidate_labels,
            "v5_decision_audit": v5_decisions,
            "expanded_universe_candidate_label": expanded_labels,
            "expanded_universe_candidate_maturity": expanded_maturity,
            "paper_strategy_daily": paper_daily,
            "bnb_paper_strategy_daily": bnb_paper_daily,
        }
    )
    backtest_bundle = build_backtest_report_bundle(backtest_frames)
    api_latency_summary = _api_latency_summary_for_export(root)
    api_errors = _api_error_summary_for_export(root)
    post_impulse_no_trigger_reasons = build_no_trigger_reasons(
        report_name="post_impulse_overextension_shadow",
        source_row_count=v5_candidate_events.height,
        output_row_count=post_impulse_overextension_shadow.height,
        missing_field_reason=(
            "candidate_events_missing" if v5_candidate_events.is_empty() else "none"
        ),
        filtered_out_reason=(
            "no_alpha6_ge_0_75_or_f3_ge_10_or_f4_ge_1_candidate"
            if post_impulse_overextension_shadow.is_empty() and not v5_candidate_events.is_empty()
            else "none"
        ),
        next_action="inspect v5_candidate_event alpha6/f3/f4 coverage and thresholds",
    )
    bottom_zone_no_trigger_reasons = build_no_trigger_reasons(
        report_name="bottom_zone_reversal_shadow",
        source_row_count=market.height,
        output_row_count=bottom_zone_reversal_shadow.height,
        missing_field_reason="market_bars_missing" if market.is_empty() else "none",
        filtered_out_reason=(
            "no_valid_market_bars_or_no_symbols"
            if bottom_zone_reversal_shadow.is_empty() and not market.is_empty()
            else "none"
        ),
        next_action="verify market_bar symbol/timestamp/close fields and rollups",
    )
    lake_file_index_refresh = _refresh_lake_file_index_for_export(root)
    pre_export_v5["lake_file_index_refresh"] = lake_file_index_refresh
    if not lake_file_index_refresh.get("ok"):
        pre_export_v5.setdefault("warnings", []).append(
            "lake_file_index_refresh_failed:"
            + str(lake_file_index_refresh.get("error") or "unknown")
        )
    lake_file_health = _lake_file_health_for_export(root)
    pre_export_v5["lake_file_health"] = lake_file_health
    lake_file_count = _lake_file_index_count(root)
    lake_file_growth_24h_count = _lake_file_index_growth_24h_count(root)
    system_acceptance_dashboard = build_system_acceptance_dashboard(
        frames=frames,
        report_frames={
            "strategy_opportunity_advisory": opportunity_advisory,
            "bnb_strong_alpha6_bypass_shadow": bnb_strong_alpha6_bypass_shadow,
            "final_score_alpha6_conflict": final_score_alpha6_conflict,
            "fast_microstructure_features": fast_microstructure_features,
            "bottom_zone_reversal_shadow": bottom_zone_reversal_shadow,
            "paper_strategy_runs": paper_runs,
            "backtest_label_summary": backtest_bundle.label_summary,
            "research_promotion_decision": backtest_bundle.promotion_decision,
            "v5_decision_replay_trades": backtest_bundle.replay_trades,
            "expanded_universe_advisory_reader": v5_expanded_universe_advisory_reader,
            "expanded_universe_paper_runs": v5_expanded_universe_paper_runs,
            "expanded_universe_paper_daily": v5_expanded_universe_paper_daily,
        },
        row_counts=row_counts or {},
        pre_export_v5=pre_export_v5 or {},
        data_quality_warnings=list((data_quality or {}).get("warnings", [])),
        api_latency_summary=api_latency_summary,
        lake_file_count=lake_file_count,
        lake_file_growth_24h_count=lake_file_growth_24h_count,
        lake_file_health=lake_file_health,
    )
    api_auth_incident = frames.get("api_auth_incident", pl.DataFrame())
    api_auth_error_timeline = frames.get(
        "api_auth_error_timeline",
        api_auth_incident,
    )
    api_auth_client_summary = frames.get("api_auth_client_summary", pl.DataFrame())
    paper_runtime_freshness = frames.get(
        "paper_runtime_freshness",
        pl.DataFrame(),
    )
    paper_proposal_propagation_status = frames.get(
        "paper_proposal_propagation_status",
        pl.DataFrame(),
    )
    post_fix_funnel_attribution = build_post_fix_funnel_attribution(
        frames.get("v5_trade_opportunity_funnel", pl.DataFrame())
    )
    system_acceptance_complete_status = build_complete_acceptance_status(
        system_acceptance=system_acceptance_dashboard,
        data_quality=data_quality or {},
        paper_freshness=paper_runtime_freshness,
        cohort=paper_cohort_manifest,
        propagation=paper_proposal_propagation_status,
        auth_incidents=api_auth_incident,
        generated_at=export_reference_at,
    )

    return {
        "market/market_snapshot.csv": _csv_member(
            "market/market_snapshot.csv", _latest_market_rows(market)
        ),
        "market/market_bars_tail.csv": _csv_member(
            "market/market_bars_tail.csv", _tail_by_time(market, "ts")
        ),
        "market/trade_activity.csv": _csv_member(
            "market/trade_activity.csv", _trade_activity_export_table(trades)
        ),
        "market/orderbook_spread.csv": _csv_member(
            "market/orderbook_spread.csv", _orderbook_spread_export_table(books)
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
            feature_anomalies if not feature_anomalies.is_empty() else _feature_anomalies(features),
        ),
        "reports/factor_definitions.csv": _csv_member(
            "reports/factor_definitions.csv",
            factor_definitions,
        ),
        "reports/factor_evidence.csv": _csv_member(
            "reports/factor_evidence.csv",
            factor_evidence,
        ),
        "reports/factor_candidates.csv": _csv_member(
            "reports/factor_candidates.csv",
            factor_candidates,
        ),
        "reports/factor_correlation_daily.csv": _csv_member(
            "reports/factor_correlation_daily.csv",
            factor_correlations,
        ),
        "reports/factor_dedupe_decision.csv": _csv_member(
            "reports/factor_dedupe_decision.csv",
            factor_v2["factor_dedupe_decision"],
        ),
        "reports/factor_family_leaderboard.csv": _csv_member(
            "reports/factor_family_leaderboard.csv",
            factor_v2["factor_family_leaderboard"],
        ),
        "reports/factor_paper_review_queue.csv": _csv_member(
            "reports/factor_paper_review_queue.csv",
            factor_v2["factor_paper_review_queue"],
        ),
        "reports/composite_factor_candidates.csv": _csv_member(
            "reports/composite_factor_candidates.csv",
            factor_v2["composite_factor_candidates"],
        ),
        "reports/factor_regime_effectiveness.csv": _csv_member(
            "reports/factor_regime_effectiveness.csv",
            factor_v2["factor_regime_effectiveness"],
        ),
        "reports/factor_forward_validation.csv": _csv_member(
            "reports/factor_forward_validation.csv",
            factor_forward_validation,
        ),
        "reports/factor_forward_validation.md": factor_forward_validation_md(
            factor_forward_validation
        ),
        "reports/factor_strategy_bridge_candidates.csv": _csv_member(
            "reports/factor_strategy_bridge_candidates.csv",
            factor_strategy_bridge_candidates,
        ),
        "reports/factor_factory_summary.md": _factor_factory_summary_md(
            factor_candidates,
            factor_evidence,
            factor_correlations,
        ),
        "costs/cost_bucket_daily.csv": _csv_member("costs/cost_bucket_daily.csv", costs),
        "costs/cost_health_daily.csv": _csv_member("costs/cost_health_daily.csv", cost_health),
        "costs/cost_estimate_examples.json": _json_text(_cost_examples(costs)),
        "costs/cost_fallbacks.csv": _csv_text(_cost_fallbacks(costs)),
        "reports/cost_bootstrap_readiness.csv": _csv_member(
            "reports/cost_bootstrap_readiness.csv",
            cost_bootstrap_readiness,
        ),
        "reports/cost_probe_fill_bill_match.csv": _csv_member(
            "reports/cost_probe_fill_bill_match.csv",
            cost_probe_fill_bill_match,
        ),
        "reports/cost_probe_cost_disagreement.csv": _csv_member(
            "reports/cost_probe_cost_disagreement.csv",
            cost_probe_cost_disagreement,
        ),
        "reports/live_universe_cost_coverage.csv": _csv_member(
            "reports/live_universe_cost_coverage.csv",
            live_universe_cost_coverage,
        ),
        "research/alpha_evidence.csv": _csv_member("research/alpha_evidence.csv", evidence),
        "research/strategy_evidence.csv": _csv_member(
            "research/strategy_evidence.csv",
            strategy_evidence,
        ),
        "research/alt_impulse_shadow_by_regime.csv": _csv_member(
            "research/alt_impulse_shadow_by_regime.csv",
            _alt_impulse_shadow_by_regime_for_export(strategy_evidence),
        ),
        "research/alt_impulse_shadow_by_symbol_regime_horizon.csv": _csv_member(
            "research/alt_impulse_shadow_by_symbol_regime_horizon.csv",
            _alt_impulse_shadow_by_symbol_regime_horizon_for_export(strategy_evidence),
        ),
        "research/strategy_evidence_samples.csv": _csv_member(
            "research/strategy_evidence_samples.csv",
            strategy_samples,
        ),
        "research/strategy_evidence_quality.csv": _csv_member(
            "research/strategy_evidence_quality.csv",
            frames.get("strategy_evidence_quality", pl.DataFrame()),
        ),
        "reports/second_stage_alpha_factory_summary.csv": _csv_member(
            "reports/second_stage_alpha_factory_summary.csv",
            second_stage_summary,
        ),
        "reports/second_stage_alpha_factory_samples.csv": _csv_member(
            "reports/second_stage_alpha_factory_samples.csv",
            second_stage_samples,
        ),
        "reports/expanded_relative_strength_decision_samples.csv": _csv_member(
            "reports/expanded_relative_strength_decision_samples.csv",
            relative_strength_decision_samples,
        ),
        "reports/alpha_factory_template_registry.csv": _csv_member(
            "reports/alpha_factory_template_registry.csv",
            alpha_factory_registry,
        ),
        "reports/alpha_factory_candidates.csv": _csv_member(
            "reports/alpha_factory_candidates.csv",
            alpha_factory_candidates,
        ),
        "reports/alpha_factory_results.csv": _csv_member(
            "reports/alpha_factory_results.csv",
            alpha_factory_results,
        ),
        "reports/alpha_factory_promotion_queue.csv": _csv_member(
            "reports/alpha_factory_promotion_queue.csv",
            alpha_factory_promotion,
        ),
        "reports/alpha_factory_daily.md": alpha_factory_daily_md(
            candidates=alpha_factory_candidates,
            results=alpha_factory_results,
            promotion_queue=alpha_factory_promotion,
        ),
        "reports/research_portfolio_status.csv": _csv_member(
            "reports/research_portfolio_status.csv",
            research_portfolio,
        ),
        "reports/research_portfolio_summary.md": research_portfolio_summary_md(
            research_portfolio,
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
            _formal_candidate_paper_ready_rows(
                alpha_discovery_board,
                paper_strategy_promotion_gate=paper_strategy_promotion_gate,
            ),
        ),
        "reports/historical_label_threshold_ready.csv": _csv_member(
            "reports/historical_label_threshold_ready.csv",
            _candidate_decision_rows(alpha_discovery_board, "HISTORICAL_LABEL_THRESHOLD_READY"),
        ),
        "reports/paper_strategy_proposals.csv": _csv_member(
            "reports/paper_strategy_proposals.csv",
            paper_proposals,
        ),
        "reports/paper_strategy_proposal_ack.csv": _csv_member(
            "reports/paper_strategy_proposal_ack.csv",
            paper_strategy_proposal_ack,
        ),
        "reports/paper_strategy_proposals_current.csv": _csv_member(
            "reports/paper_strategy_proposals_current.csv",
            paper_proposals_current,
        ),
        "reports/paper_strategy_trackers_current.csv": _csv_member(
            "reports/paper_strategy_trackers_current.csv",
            paper_strategy_trackers_current,
        ),
        "reports/paper_strategy_ack_current.csv": _csv_member(
            "reports/paper_strategy_ack_current.csv",
            paper_strategy_proposal_ack_current,
        ),
        "reports/paper_strategy_ack_history.csv": _csv_member(
            "reports/paper_strategy_ack_history.csv",
            paper_strategy_ack_history,
        ),
        "reports/strategy_opportunity_advisory.csv": _csv_member(
            "reports/strategy_opportunity_advisory.csv",
            opportunity_advisory,
        ),
        "reports/v5_opportunity_event.csv": _csv_member(
            "reports/v5_opportunity_event.csv",
            _tail_by_time(trade_opportunity_events, "decision_ts", limit=50_000),
        ),
        "reports/v5_opportunity_label.csv": _csv_member(
            "reports/v5_opportunity_label.csv",
            _tail_by_time(trade_opportunity_labels, "decision_ts", limit=50_000),
        ),
        "reports/trade_opportunity_event.csv": _csv_member(
            "reports/trade_opportunity_event.csv",
            _tail_by_time(trade_opportunity_events, "decision_ts", limit=50_000),
        ),
        "reports/trade_opportunity_label.csv": _csv_member(
            "reports/trade_opportunity_label.csv",
            _tail_by_time(trade_opportunity_labels, "decision_ts", limit=50_000),
        ),
        "reports/trade_level_similarity_outcome.csv": _csv_member(
            "reports/trade_level_similarity_outcome.csv",
            _tail_by_time(trade_level_similarity, "decision_ts", limit=50_000),
        ),
        "reports/trade_level_judgment.csv": _csv_member(
            "reports/trade_level_judgment.csv",
            _tail_by_time(trade_level_judgments, "decision_ts", limit=50_000),
        ),
        "reports/trade_level_bucket_policy.csv": _csv_member(
            "reports/trade_level_bucket_policy.csv",
            _tail_by_time(trade_level_bucket_policy, "created_at", limit=50_000),
        ),
        "reports/trade_level_opportunity_queue.csv": _csv_member(
            "reports/trade_level_opportunity_queue.csv",
            _tail_by_time(trade_level_opportunity_queue, "created_at", limit=50_000),
        ),
        "reports/quant_lab_false_block_audit.csv": _csv_member(
            "reports/quant_lab_false_block_audit.csv",
            _tail_by_time(false_block_audit, "decision_ts", limit=50_000),
        ),
        "reports/v5_trade_learning_sample.csv": _csv_member(
            "reports/v5_trade_learning_sample.csv",
            _tail_by_time(v5_trade_learning_samples, "decision_ts", limit=50_000),
        ),
        "reports/v5_trade_outcome_attribution.csv": _csv_member(
            "reports/v5_trade_outcome_attribution.csv",
            _tail_by_time(v5_trade_outcome_attribution, "decision_ts", limit=50_000),
        ),
        "reports/quant_lab_opportunity_cost_event.csv": _csv_member(
            "reports/quant_lab_opportunity_cost_event.csv",
            _tail_by_time(opportunity_cost_events, "decision_ts", limit=50_000),
        ),
        "reports/quant_lab_opportunity_cost_daily.csv": _csv_member(
            "reports/quant_lab_opportunity_cost_daily.csv",
            _tail_by_time(opportunity_cost_daily, "day", limit=5_000),
        ),
        "reports/opportunity_cost_by_bucket.csv": _csv_member(
            "reports/opportunity_cost_by_bucket.csv",
            opportunity_cost_by_bucket,
        ),
        "reports/gate_candidate_level_effectiveness.csv": _csv_member(
            "reports/gate_candidate_level_effectiveness.csv",
            gate_candidate_effectiveness,
        ),
        "reports/gate_independent_event_effectiveness.csv": _csv_member(
            "reports/gate_independent_event_effectiveness.csv",
            gate_independent_event_effectiveness,
        ),
        "reports/gate_paper_trade_effectiveness.csv": _csv_member(
            "reports/gate_paper_trade_effectiveness.csv",
            gate_paper_trade_effectiveness,
        ),
        "reports/gate_live_trade_effectiveness.csv": _csv_member(
            "reports/gate_live_trade_effectiveness.csv",
            gate_live_trade_effectiveness,
        ),
        "reports/duplicate_event_report.csv": _csv_member(
            "reports/duplicate_event_report.csv",
            duplicate_event_report,
        ),
        "reports/duplicate_factor_report.csv": _csv_member(
            "reports/duplicate_factor_report.csv",
            duplicate_factor_report,
        ),
        "reports/strategy_cost_trust.csv": _csv_member(
            "reports/strategy_cost_trust.csv",
            strategy_cost_trust,
        ),
        "reports/quant_lab_decision_regret.csv": _csv_member(
            "reports/quant_lab_decision_regret.csv",
            _tail_by_time(decision_regret, "decision_ts", limit=50_000),
        ),
        "reports/api_latency_summary.csv": _csv_member(
            "reports/api_latency_summary.csv",
            api_latency_summary,
        ),
        "reports/api_latency_summary.md": _api_latency_summary_md(root),
        "reports/api_error_summary.csv": _csv_member(
            "reports/api_error_summary.csv",
            api_errors,
        ),
        "reports/api_auth_error_timeline.csv": _csv_member(
            "reports/api_auth_error_timeline.csv",
            api_auth_error_timeline,
        ),
        "reports/api_auth_client_summary.csv": _csv_member(
            "reports/api_auth_client_summary.csv",
            api_auth_client_summary,
        ),
        "reports/v5_bundle_sync_diagnostics.csv": _csv_member(
            "reports/v5_bundle_sync_diagnostics.csv",
            _v5_bundle_sync_diagnostics_frame(
                frames=frames,
                pre_export_v5=pre_export_v5 or {},
                generated_at=datetime.now(UTC),
            ),
        ),
        "reports/v5_local_live_vs_quant_lab_shadow.csv": _csv_member(
            "reports/v5_local_live_vs_quant_lab_shadow.csv",
            local_live_vs_shadow,
        ),
        "reports/missed_opportunity_audit.csv": _csv_member(
            "reports/missed_opportunity_audit.csv",
            missed_opportunity_audit,
        ),
        "reports/missed_opportunity_summary.md": _missed_opportunity_summary_md(
            missed_opportunity_audit,
            risk_on_multi_buy_shadow,
        ),
        "reports/bnb_missed_opportunity_samples.csv": _csv_member(
            "reports/bnb_missed_opportunity_samples.csv",
            bnb_missed_opportunity,
        ),
        "reports/bnb_missed_opportunity_summary.md": _bnb_missed_opportunity_summary_md(
            bnb_missed_opportunity
        ),
        "reports/final_score_vs_alpha6_conflict.csv": _csv_member(
            "reports/final_score_vs_alpha6_conflict.csv",
            final_score_alpha6_conflict,
        ),
        "reports/final_score_vs_alpha6_conflict_summary.md": (
            _final_score_vs_alpha6_conflict_summary_md(final_score_alpha6_conflict)
        ),
        "reports/bnb_strong_alpha6_bypass_shadow.csv": _csv_member(
            "reports/bnb_strong_alpha6_bypass_shadow.csv",
            bnb_strong_alpha6_bypass_shadow,
        ),
        "reports/bnb_strong_alpha6_bypass_summary.md": (
            _bnb_strong_alpha6_bypass_summary_md(bnb_strong_alpha6_bypass_shadow)
        ),
        "reports/post_impulse_overextension_shadow.csv": _csv_member(
            "reports/post_impulse_overextension_shadow.csv",
            post_impulse_overextension_shadow,
        ),
        "reports/post_impulse_overextension_no_trigger_reasons.csv": _csv_member(
            "reports/post_impulse_overextension_no_trigger_reasons.csv",
            post_impulse_no_trigger_reasons,
        ),
        "reports/late_breakout_failure_shadow.csv": _csv_member(
            "reports/late_breakout_failure_shadow.csv",
            late_breakout_failure_shadow,
        ),
        "reports/late_breakout_failure_protect_shadow.csv": _csv_member(
            "reports/late_breakout_failure_protect_shadow.csv",
            late_breakout_failure_shadow,
        ),
        "reports/bottom_zone_reversal_shadow.csv": _csv_member(
            "reports/bottom_zone_reversal_shadow.csv",
            bottom_zone_reversal_shadow,
        ),
        "reports/bottom_zone_reversal_no_trigger_reasons.csv": _csv_member(
            "reports/bottom_zone_reversal_no_trigger_reasons.csv",
            bottom_zone_no_trigger_reasons,
        ),
        "reports/bottom_zone_reversal_summary.md": bottom_zone_reversal_summary_md(
            bottom_zone_reversal_shadow
        ),
        "reports/bottom_zone_probe_paper_readiness.csv": _csv_member(
            "reports/bottom_zone_probe_paper_readiness.csv",
            bottom_zone_probe_paper_readiness,
        ),
        "reports/fast_microstructure_features.csv": _csv_member(
            "reports/fast_microstructure_features.csv",
            fast_microstructure_features,
        ),
        "reports/fast_microstructure_forward_test.csv": _csv_member(
            "reports/fast_microstructure_forward_test.csv",
            fast_microstructure_forward_test,
        ),
        "reports/fast_microstructure_strategy_candidates.csv": _csv_member(
            "reports/fast_microstructure_strategy_candidates.csv",
            fast_microstructure_strategy_candidates,
        ),
        "reports/fast_microstructure_strategy_review.csv": _csv_member(
            "reports/fast_microstructure_strategy_review.csv",
            fast_microstructure_strategy_review,
        ),
        "reports/fast_microstructure_forward_summary.md": fast_microstructure_forward_summary_md(
            fast_microstructure_forward_test
        ),
        "reports/market_pressure_score.csv": _csv_member(
            "reports/market_pressure_score.csv",
            market_pressure_score,
        ),
        "reports/market_pressure_summary.md": market_pressure_summary_md(market_pressure_score),
        "reports/system_acceptance_dashboard.csv": _csv_member(
            "reports/system_acceptance_dashboard.csv",
            system_acceptance_dashboard,
        ),
        "reports/system_acceptance_dashboard.md": system_acceptance_dashboard_md(
            system_acceptance_dashboard
        ),
        "reports/paper_runtime_freshness.csv": _csv_member(
            "reports/paper_runtime_freshness.csv",
            paper_runtime_freshness,
        ),
        "reports/paper_proposal_propagation_status.csv": _csv_member(
            "reports/paper_proposal_propagation_status.csv",
            paper_proposal_propagation_status,
        ),
        "reports/system_acceptance_complete_status.json": _json_text(
            system_acceptance_complete_status
        ),
        "reports/post_fix_funnel_attribution.json": _json_text(
            post_fix_funnel_attribution
        ),
        "reports/backtest_label_summary.csv": _csv_member(
            "reports/backtest_label_summary.csv",
            backtest_bundle.label_summary,
        ),
        "reports/backtest_label_summary.md": backtest_bundle.label_summary_md,
        "reports/v5_decision_replay_trades.csv": _csv_member(
            "reports/v5_decision_replay_trades.csv",
            backtest_bundle.replay_trades,
        ),
        "reports/v5_decision_replay_equity.csv": _csv_member(
            "reports/v5_decision_replay_equity.csv",
            backtest_bundle.replay_equity,
        ),
        "reports/v5_decision_replay_summary.md": backtest_bundle.replay_summary_md,
        "reports/backtest_regime_breakdown.csv": _csv_member(
            "reports/backtest_regime_breakdown.csv",
            backtest_bundle.regime_breakdown,
        ),
        "reports/bottom_zone_backtest.csv": _csv_member(
            "reports/bottom_zone_backtest.csv",
            backtest_bundle.bottom_zone_backtest,
        ),
        "reports/bottom_zone_backtest_summary.md": backtest_bundle.bottom_zone_summary_md,
        "reports/research_promotion_decision.csv": _csv_member(
            "reports/research_promotion_decision.csv",
            backtest_bundle.promotion_decision,
        ),
        "reports/research_promotion_decision.md": backtest_bundle.promotion_decision_md,
        "reports/backtest_vs_paper_consistency.csv": _csv_member(
            "reports/backtest_vs_paper_consistency.csv",
            backtest_bundle.backtest_vs_paper_consistency,
        ),
        "reports/backtest_vs_paper_consistency.md": (
            backtest_bundle.backtest_vs_paper_consistency_md
        ),
        "reports/backtest_vs_paper_gap_report.csv": _csv_member(
            "reports/backtest_vs_paper_gap_report.csv",
            backtest_bundle.backtest_vs_paper_gap_report,
        ),
        "reports/backtest_vs_paper_gap_report.md": (
            backtest_bundle.backtest_vs_paper_gap_report_md
        ),
        "reports/risk_on_multi_buy_shadow.csv": _csv_member(
            "reports/risk_on_multi_buy_shadow.csv",
            risk_on_multi_buy_shadow,
        ),
        "reports/risk_on_multi_buy_summary.md": _risk_on_multi_buy_summary_md(
            risk_on_multi_buy_shadow
        ),
        "reports/market_regime_daily.csv": _csv_member(
            "reports/market_regime_daily.csv",
            market_regime,
        ),
        "reports/strategy_regime_matrix.csv": _csv_member(
            "reports/strategy_regime_matrix.csv",
            strategy_regime_matrix,
        ),
        "reports/regime_strategy_advisory.csv": _csv_member(
            "reports/regime_strategy_advisory.csv",
            regime_strategy_advisory,
        ),
        "reports/expanded_universe_candidates.csv": _csv_member(
            "reports/expanded_universe_candidates.csv",
            expanded_candidates,
        ),
        "reports/expanded_universe_quality.csv": _csv_member(
            "reports/expanded_universe_quality.csv",
            expanded_quality if not expanded_quality.is_empty() else symbol_quality,
        ),
        "reports/expanded_universe_daily.md": _expanded_universe_daily_md(
            candidates=expanded_candidates,
            quality=expanded_quality if not expanded_quality.is_empty() else symbol_quality,
            events=expanded_events,
            labels=expanded_labels,
            promotion_queue=expanded_promotion,
            maturity=expanded_maturity,
            watchlist=expanded_watchlist,
        ),
        "reports/expanded_universe_watchlist.csv": _csv_member(
            "reports/expanded_universe_watchlist.csv",
            expanded_watchlist,
        ),
        "reports/expanded_universe_candidate_maturity.csv": _csv_member(
            "reports/expanded_universe_candidate_maturity.csv",
            expanded_maturity,
        ),
        "reports/expanded_universe_promotion_summary.md": _expanded_universe_promotion_summary_md(
            maturity=expanded_maturity,
            watchlist=expanded_watchlist,
            promotion_queue=expanded_promotion,
        ),
        "reports/expanded_universe_promotion_queue.csv": _csv_member(
            "reports/expanded_universe_promotion_queue.csv",
            expanded_promotion,
        ),
        "reports/expanded_universe_kill_list.csv": _csv_member(
            "reports/expanded_universe_kill_list.csv",
            _expanded_promotion_rows(expanded_promotion, "KILL"),
        ),
        "reports/expanded_universe_replacement_candidates.csv": _csv_member(
            "reports/expanded_universe_replacement_candidates.csv",
            _expanded_replacement_rows(expanded_promotion),
        ),
        "reports/expanded_universe_strategy_evidence.csv": _csv_member(
            "reports/expanded_universe_strategy_evidence.csv",
            _expanded_strategy_evidence_rows(strategy_evidence),
        ),
        "reports/expanded_crypto_universe_shadow.csv": _csv_member(
            "reports/expanded_crypto_universe_shadow.csv",
            expanded_universe,
        ),
        "reports/symbol_quality_score.csv": _csv_member(
            "reports/symbol_quality_score.csv",
            symbol_quality,
        ),
        "reports/expanded_crypto_candidate_outcomes_by_symbol.csv": _csv_member(
            "reports/expanded_crypto_candidate_outcomes_by_symbol.csv",
            expanded_outcomes,
        ),
        "reports/expanded_crypto_recommendations.json": _json_text(
            _expanded_recommendations_json(expanded_recommendations)
        ),
        "reports/strategy_level_dashboard.csv": _csv_member(
            "reports/strategy_level_dashboard.csv",
            strategy_level_dashboard,
        ),
        "reports/paper_strategy_runs.csv": _csv_member(
            "reports/paper_strategy_runs.csv",
            paper_runs,
        ),
        "reports/paper_strategy_daily.csv": _csv_member(
            "reports/paper_strategy_daily.csv",
            paper_daily,
        ),
        "reports/paper_strategy_registry.csv": _csv_member(
            "reports/paper_strategy_registry.csv",
            paper_strategy_registry,
        ),
        "reports/paper_strategy_registry_current.csv": _csv_member(
            "reports/paper_strategy_registry_current.csv",
            paper_strategy_registry,
        ),
        "reports/paper_strategy_registry_history.csv": _csv_member(
            "reports/paper_strategy_registry_history.csv",
            paper_strategy_registry_history,
        ),
        "reports/paper_strategy_identity_conflict.csv": _csv_member(
            "reports/paper_strategy_identity_conflict.csv",
            paper_strategy_identity_conflict,
        ),
        "reports/paper_strategy_identity_conflict.md": _paper_identity_conflict_md(
            paper_strategy_identity_conflict
        ),
        "reports/paper_cohort_manifest.json": _json_text(
            {"cohorts": paper_cohort_manifest.to_dicts()}
        ),
        "reports/paper_cohort_status.md": _paper_cohort_status_md(
            paper_cohort_manifest
        ),
        "reports/paper_strategy_promotion_gate.csv": _csv_member(
            "reports/paper_strategy_promotion_gate.csv",
            paper_strategy_promotion_gate,
        ),
        "reports/paper_strategy_migration_audit.csv": _csv_member(
            "reports/paper_strategy_migration_audit.csv",
            paper_strategy_migration_audit,
        ),
        "reports/bnb_paper_strategy_runs.csv": _csv_member(
            "reports/bnb_paper_strategy_runs.csv",
            bnb_paper_runs,
        ),
        "reports/bnb_paper_strategy_daily.csv": _csv_member(
            "reports/bnb_paper_strategy_daily.csv",
            bnb_paper_daily,
        ),
        "reports/bnb_paper_strategy_summary.md": _bnb_paper_strategy_summary_md(bnb_paper_daily),
        "reports/v5_quant_lab_consistency_dashboard.md": (
            _v5_quant_lab_consistency_dashboard_md(
                frames=frames,
                final_score_alpha6_conflict=final_score_alpha6_conflict,
                bnb_strong_alpha6_bypass_shadow=bnb_strong_alpha6_bypass_shadow,
                negative_expectancy_attribution=negative_expectancy_attribution,
                bnb_paper_daily=bnb_paper_daily,
                risk_on_multi_buy_shadow=risk_on_multi_buy_shadow,
                opportunity_advisory=opportunity_advisory,
            )
        ),
        "reports/paper_strategy_summary.md": paper_strategy_summary_md(paper_daily),
        "reports/paper_slippage_coverage.csv": _csv_member(
            "reports/paper_slippage_coverage.csv",
            paper_slippage,
        ),
        "reports/sol_protect_paper_loss_attribution.csv": _csv_member(
            "reports/sol_protect_paper_loss_attribution.csv",
            sol_protect_loss_attribution,
        ),
        "reports/sol_protect_paper_loss_summary.md": sol_protect_paper_loss_summary_md(
            sol_protect_loss_summary,
            sol_protect_loss_attribution,
        ),
        "reports/missed_low_audit.csv": _csv_member(
            "reports/missed_low_audit.csv",
            missed_low_audit,
        ),
        "reports/missed_low_by_symbol.csv": _csv_member(
            "reports/missed_low_by_symbol.csv",
            missed_low_by_symbol,
        ),
        "reports/missed_low_by_entry_reason.csv": _csv_member(
            "reports/missed_low_by_entry_reason.csv",
            missed_low_by_entry_reason,
        ),
        "reports/late_entry_chase_shadow.csv": _csv_member(
            "reports/late_entry_chase_shadow.csv",
            late_entry_chase_shadow,
        ),
        "reports/late_entry_chase_threshold_advisory.json": _json_text(
            _entry_quality_json(late_entry_threshold)
        ),
        "reports/late_entry_chase_threshold_sensitivity.csv": _csv_member(
            "reports/late_entry_chase_threshold_sensitivity.csv",
            history_late_threshold,
        ),
        "reports/late_entry_chase_threshold_sensitivity_by_symbol.csv": _csv_member(
            "reports/late_entry_chase_threshold_sensitivity_by_symbol.csv",
            late_entry_threshold_by_symbol,
        ),
        "reports/late_entry_chase_threshold_advisory_by_symbol.json": _json_text(
            _late_entry_threshold_advisory_by_symbol_json(late_entry_threshold_by_symbol)
        ),
        "reports/threshold_advisory_by_symbol.json": _json_text(
            _threshold_advisory_by_symbol_json(late_entry_threshold_by_symbol)
        ),
        "reports/pullback_reversal_shadow_outcomes.csv": _csv_member(
            "reports/pullback_reversal_shadow_outcomes.csv",
            pullback_reversal_shadow,
        ),
        "reports/pullback_reversal_by_symbol.csv": _csv_member(
            "reports/pullback_reversal_by_symbol.csv",
            history_pullback_by_symbol,
        ),
        "reports/pullback_reversal_by_regime.csv": _csv_member(
            "reports/pullback_reversal_by_regime.csv",
            history_pullback_by_regime,
        ),
        "reports/pullback_reversal_by_horizon.csv": _csv_member(
            "reports/pullback_reversal_by_horizon.csv",
            history_pullback_by_horizon,
        ),
        "reports/pullback_reversal_rule_comparison.csv": _csv_member(
            "reports/pullback_reversal_rule_comparison.csv",
            pullback_reversal_rule_comparison,
        ),
        "reports/old_v1_vs_v2_comparison.csv": _csv_member(
            "reports/old_v1_vs_v2_comparison.csv",
            pullback_reversal_rule_comparison,
        ),
        "reports/pullback_reversal_v2_by_symbol.csv": _csv_member(
            "reports/pullback_reversal_v2_by_symbol.csv",
            _pullback_v2_by_symbol_for_export(pullback_reversal_shadow),
        ),
        "reports/pullback_reversal_readiness.json": _json_text(
            _entry_quality_json(pullback_reversal_readiness)
        ),
        "reports/pullback_reversal_v2_readiness.json": _json_text(
            _entry_quality_json(pullback_reversal_readiness)
        ),
        "reports/btc_probe_exit_policy_review.csv": _csv_member(
            "reports/btc_probe_exit_policy_review.csv",
            btc_probe_exit_policy_review,
        ),
        "reports/btc_probe_exit_policy_summary.md": btc_probe_exit_policy_summary_md(
            btc_probe_exit_policy_summary,
            btc_probe_exit_policy_review,
        ),
        "reports/bnb_swing_exit_policy_review.csv": _csv_member(
            "reports/bnb_swing_exit_policy_review.csv",
            bnb_swing_exit_policy_review,
        ),
        "reports/bnb_negative_expectancy_attribution.csv": _csv_member(
            "reports/bnb_negative_expectancy_attribution.csv",
            bnb_negative_expectancy_attribution,
        ),
        "reports/negative_expectancy_attribution.csv": _csv_member(
            "reports/negative_expectancy_attribution.csv",
            negative_expectancy_attribution,
        ),
        "reports/negative_expectancy_attribution_summary.md": (
            _negative_expectancy_attribution_summary_md(negative_expectancy_attribution)
        ),
        "reports/bnb_exit_policy_v5_vs_quant_lab_consistency.csv": _csv_member(
            "reports/bnb_exit_policy_v5_vs_quant_lab_consistency.csv",
            bnb_exit_policy_v5_vs_quant_lab_consistency,
        ),
        "reports/bnb_swing_exit_policy_summary.md": bnb_swing_exit_policy_summary_md(
            bnb_swing_exit_policy_summary,
            bnb_swing_exit_policy_review,
        ),
        "reports/exit_policy_review.csv": _csv_member(
            "reports/exit_policy_review.csv",
            exit_policy_review_sample,
        ),
        "reports/exit_policy_summary.md": _exit_policy_summary_md(
            exit_policy_review_summary,
            exit_policy_review_sample,
        ),
        "reports/anti_leakage_check.csv": _csv_member(
            "reports/anti_leakage_check.csv",
            history_anti_leakage,
        ),
        "reports/entry_quality_summary.md": _entry_quality_summary_md(
            missed_low_audit=missed_low_audit,
            missed_low_by_symbol=missed_low_by_symbol,
            late_entry_chase_shadow=late_entry_chase_shadow,
            late_entry_threshold=late_entry_threshold,
            pullback_reversal_shadow=pullback_reversal_shadow,
            pullback_reversal_readiness=pullback_reversal_readiness,
            entry_quality_advisory=entry_quality_advisory,
        ),
        "reports/entry_quality_historical_summary.md": _entry_quality_history_summary_md(
            history_metrics
        ),
        "reports/entry_quality_historical_metrics.json": _json_text(
            _entry_quality_history_metrics_json(history_metrics)
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
        "v5/v5_quant_lab_fallbacks.csv": _csv_member("v5/v5_quant_lab_fallbacks.csv", v5_fallbacks),
        "v5/v5_fill_bill_cost_reconciliation.csv": _csv_member(
            "v5/v5_fill_bill_cost_reconciliation.csv",
            _tail_by_time(
                v5_fill_bill_cost_reconciliation,
                "last_fill_ts",
                limit=50_000,
            ),
        ),
        "v5/v5_paper_strategy_proposal_ack.csv": _csv_member(
            "v5/v5_paper_strategy_proposal_ack.csv",
            _tail_by_time(v5_paper_strategy_proposal_ack, "accepted_at", limit=20_000),
        ),
        "v5/v5_paper_strategy_registry.csv": _csv_member(
            "v5/v5_paper_strategy_registry.csv",
            _tail_by_time(v5_paper_strategy_registry, "updated_at", limit=20_000),
        ),
        "v5/v5_paper_strategy_state.csv": _csv_member(
            "v5/v5_paper_strategy_state.csv",
            _tail_by_time(v5_paper_strategy_state, "updated_at", limit=20_000),
        ),
        "v5/v5_paper_strategy_signals.csv": _csv_member(
            "v5/v5_paper_strategy_signals.csv",
            _tail_by_time(v5_paper_strategy_signals, "decision_ts", limit=50_000),
        ),
        "v5/v5_paper_strategy_runs.csv": _csv_member(
            "v5/v5_paper_strategy_runs.csv",
            _tail_by_time(v5_paper_strategy_runs, "closed_at", limit=50_000),
        ),
        "v5/v5_paper_strategy_exit_quality.csv": _csv_member(
            "v5/v5_paper_strategy_exit_quality.csv",
            _tail_by_time(v5_paper_strategy_exit_quality, "ingest_ts", limit=20_000),
        ),
        "v5/v5_trade_opportunity_funnel.csv": _csv_member(
            "v5/v5_trade_opportunity_funnel.csv",
            _tail_by_time(v5_trade_opportunity_funnel, "ts_utc", limit=50_000),
        ),
        "v5/v5_paper_strategy_daily.csv": _csv_member(
            "v5/v5_paper_strategy_daily.csv",
            _tail_by_time(v5_paper_strategy_daily, "created_at", limit=50_000),
        ),
        "v5/v5_paper_strategy_quote_coverage.csv": _csv_member(
            "v5/v5_paper_strategy_quote_coverage.csv",
            _tail_by_time(v5_paper_strategy_quote_coverage, "ingest_ts", limit=20_000),
        ),
        "v5/v5_paper_strategy_cost_evidence.csv": _csv_member(
            "v5/v5_paper_strategy_cost_evidence.csv",
            _tail_by_time(v5_paper_strategy_cost_evidence, "ingest_ts", limit=20_000),
        ),
        "v5/v5_paper_strategy_errors.csv": _csv_member(
            "v5/v5_paper_strategy_errors.csv",
            _tail_by_time(v5_paper_strategy_errors, "ts_utc", limit=20_000),
        ),
        "v5/v5_paper_strategy_restart_recovery.csv": _csv_member(
            "v5/v5_paper_strategy_restart_recovery.csv",
            _tail_by_time(
                v5_paper_strategy_restart_recovery,
                "recovered_at",
                limit=20_000,
            ),
        ),
        "v5/v5_quant_lab_contract_status.csv": _csv_member(
            "v5/v5_quant_lab_contract_status.csv",
            _tail_by_time(v5_quant_lab_contract_status, "generated_at", limit=20_000),
        ),
        "v5/v5_cost_probe_p3_preflight.csv": _csv_member(
            "v5/v5_cost_probe_p3_preflight.csv",
            _tail_by_time(v5_cost_probe_p3_preflight, "generated_at_utc", limit=10_000),
        ),
        "v5/v5_cost_probe_live_execution_status.csv": _csv_member(
            "v5/v5_cost_probe_live_execution_status.csv",
            _tail_by_time(
                v5_cost_probe_live_execution_status,
                "generated_at_utc",
                limit=10_000,
            ),
        ),
        "v5/v5_cost_probe_live_execution_status_canonical.csv": _csv_member(
            "v5/v5_cost_probe_live_execution_status_canonical.csv",
            _tail_by_time(
                v5_cost_probe_live_execution_status_canonical,
                "generated_at_utc",
                limit=10_000,
            ),
        ),
        "v5/v5_cost_probe_order_events.csv": _csv_member(
            "v5/v5_cost_probe_order_events.csv",
            _tail_by_time(v5_cost_probe_order_events, "event_ts", limit=50_000),
        ),
        "v5/v5_cost_probe_roundtrip_events.csv": _csv_member(
            "v5/v5_cost_probe_roundtrip_events.csv",
            _tail_by_time(v5_cost_probe_roundtrip_events, "event_ts", limit=50_000),
        ),
        "v5/v5_cost_probe_roundtrip_canonical.csv": _csv_member(
            "v5/v5_cost_probe_roundtrip_canonical.csv",
            _tail_by_time(v5_cost_probe_roundtrip_canonical, "event_ts", limit=50_000),
        ),
        "v5/v5_candidate_events.csv": _csv_member(
            "v5/v5_candidate_events.csv",
            _tail_by_time(v5_candidate_events, "ts_utc", limit=50_000),
        ),
        "v5/v5_btc_probe_entry_quality_audit.csv": _csv_member(
            "v5/v5_btc_probe_entry_quality_audit.csv",
            _tail_by_time(v5_btc_probe_entry_quality_audit, "ingest_ts", limit=50_000),
        ),
        "v5/v5_candidate_labels.csv": _csv_member(
            "v5/v5_candidate_labels.csv",
            _tail_by_time(v5_candidate_labels, "ts_utc", limit=100_000),
        ),
        "v5/v5_candidate_quality.csv": _csv_member(
            "v5/v5_candidate_quality.csv",
            v5_candidate_quality,
        ),
        "reports/v5_candidate_feature_completeness_by_strategy.csv": _csv_member(
            "reports/v5_candidate_feature_completeness_by_strategy.csv",
            _v5_candidate_feature_completeness_by_strategy(v5_candidate_events),
        ),
        "v5/v5_candidate_outcome_summary.csv": _csv_member(
            "v5/v5_candidate_outcome_summary.csv",
            v5_candidate_outcomes,
        ),
        "v5/v5_expanded_universe_advisory_reader.csv": _csv_member(
            "v5/v5_expanded_universe_advisory_reader.csv",
            v5_expanded_universe_advisory_reader,
        ),
        "v5/v5_expanded_universe_paper_runs.csv": _csv_member(
            "v5/v5_expanded_universe_paper_runs.csv",
            v5_expanded_universe_paper_runs,
        ),
        "v5/v5_expanded_universe_paper_daily.csv": _csv_member(
            "v5/v5_expanded_universe_paper_daily.csv",
            v5_expanded_universe_paper_daily,
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


def _dedupe_stable_row_key_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "stable_row_key" not in frame.columns:
        return frame
    rows_by_key: dict[str, dict[str, Any]] = {}
    for row in frame.to_dicts():
        key = str(row.get("stable_row_key") or "").strip()
        if not key:
            key = safe_json_dumps(row)
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


def _v5_candidate_feature_completeness_by_strategy(frame: pl.DataFrame) -> pl.DataFrame:
    path = "reports/v5_candidate_feature_completeness_by_strategy.csv"
    if frame.is_empty():
        return _empty_csv_schema_frame(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in frame.to_dicts():
        strategy = str(row.get("strategy_candidate") or "UNKNOWN").strip() or "UNKNOWN"
        grouped.setdefault(strategy, []).append(row)

    def field_completeness(
        group_rows: list[dict[str, Any]],
        fields: tuple[str, ...],
    ) -> dict[str, float]:
        if not group_rows:
            return {field: 0.0 for field in fields}
        return {
            field: sum(1 for row in group_rows if _candidate_feature_observed(row.get(field)))
            / len(group_rows)
            for field in fields
        }

    def mean(values: Iterable[float]) -> float:
        material = list(values)
        return float(sum(material) / len(material)) if material else 0.0

    output: list[dict[str, Any]] = []
    for strategy, group_rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        denominator_rows, no_signal_context_count = _v5_candidate_feature_denominator_rows(
            group_rows
        )
        core = field_completeness(denominator_rows, V5_CANDIDATE_CORE_SIGNAL_FIELDS)
        edge = field_completeness(denominator_rows, V5_CANDIDATE_EDGE_CONTEXT_FIELDS)
        optional = field_completeness(denominator_rows, V5_CANDIDATE_OPTIONAL_SIGNAL_FIELDS)
        all_fields = {**core, **edge, **optional}
        sample_symbols = sorted(
            {
                normalize_symbol(str(row.get("symbol") or ""))
                for row in group_rows
                if str(row.get("symbol") or "").strip()
            }
        )
        output.append(
            {
                "strategy_candidate": strategy,
                "row_count": len(group_rows),
                "feature_denominator_row_count": len(denominator_rows),
                "no_signal_context_row_count": no_signal_context_count,
                "core_signal_completeness": round(mean(core.values()), 6),
                "edge_context_completeness": round(mean(edge.values()), 6),
                "optional_signal_completeness": round(mean(optional.values()), 6),
                "field_completeness_json": safe_json_dumps(
                    {field: round(value, 6) for field, value in sorted(all_fields.items())}
                ),
                "missing_core_fields": ",".join(
                    field for field, value in core.items() if value < 0.8
                ),
                "missing_edge_context_fields": ",".join(
                    field for field, value in edge.items() if value < 0.8
                ),
                "missing_optional_fields": ",".join(
                    field for field, value in optional.items() if value < 0.8
                ),
                "sample_symbols": ",".join(sample_symbols[:12]),
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return pl.DataFrame(output, infer_schema_length=None).select(CSV_SCHEMAS[path])


def _v5_candidate_feature_denominator_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if not rows:
        return [], 0
    candidate_rows = [row for row in rows if _v5_candidate_row_needs_feature_coverage(row)]
    if candidate_rows:
        return candidate_rows, len(rows) - len(candidate_rows)
    return rows, 0


def _v5_candidate_row_needs_feature_coverage(row: dict[str, Any]) -> bool:
    eligible = str(row.get("eligible_before_filters") or "").strip().lower()
    if eligible in {"true", "1", "yes", "y"}:
        return True
    block_reason = str(row.get("block_reason") or "").strip()
    if block_reason:
        return True
    decision = str(row.get("final_decision") or "").strip().lower()
    if decision and decision != "no_order":
        return True
    if eligible in {"false", "0", "no", "n"}:
        return False
    for field in ("target_weight_raw", "target_weight_after_risk", "current_weight"):
        value = _float_or_none(row.get(field))
        if value is not None and abs(value) > 0.0:
            return True
    return any(_candidate_feature_observed(row.get(field)) for field in ("final_score", "rank"))


def _candidate_feature_observed(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() not in UNOBSERVABLE_TEXT_VALUES


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


def _attach_selected_v5_bundle(
    members: dict[str, _MemberPayload],
    v5_context: dict[str, Any],
) -> None:
    selected_path_raw = str(v5_context.get("selected_v5_bundle_path") or "").strip()
    if not selected_path_raw:
        v5_context.update(
            {
                "embedded_v5_bundle_present": False,
                "embedded_v5_bundle_member_path": None,
                "embedded_v5_bundle_name": None,
                "embedded_v5_bundle_sha256": None,
                "embedded_v5_bundle_size_bytes": 0,
                "embedded_v5_bundle_matches_selected": False,
                "embedded_v5_bundle_validation_status": "not_attached",
                "embedded_v5_bundle_reason": "selected_v5_bundle_path_missing",
            }
        )
        return

    selected_path = Path(selected_path_raw)
    bundle_name = selected_path.name
    if not _is_expected_v5_followup_bundle_name(bundle_name):
        raise RuntimeError(f"selected_v5_bundle_attachment_invalid_name:{bundle_name}")
    if not selected_path.is_file():
        raise RuntimeError(f"selected_v5_bundle_attachment_missing:{selected_path}")

    validation = validate_v5_bundle(selected_path, BundleLimits())
    if validation.rejected or not validation.sha256:
        reason = ";".join(validation.reasons) or "missing_sha256"
        raise RuntimeError(f"selected_v5_bundle_attachment_invalid:{reason}")

    actual_sha = validation.sha256.lower()
    selected_sha = str(v5_context.get("selected_v5_bundle_sha256") or "").strip().lower()
    manifest_sha = (
        str(v5_context.get("selected_v5_bundle_manifest_bundle_sha256") or "").strip().lower()
    )
    if selected_sha and actual_sha != selected_sha:
        raise RuntimeError(
            "selected_v5_bundle_attachment_sha_mismatch:"
            f"selected={selected_sha};actual={actual_sha}"
        )
    if manifest_sha and actual_sha != manifest_sha:
        raise RuntimeError(
            "selected_v5_bundle_attachment_manifest_sha_mismatch:"
            f"manifest={manifest_sha};actual={actual_sha}"
        )

    redacted_files_dir = _selected_v5_redacted_files_dir(
        v5_context,
        selected_path=selected_path,
        selected_sha256=validation.sha256,
    )
    redacted_source = "redacted_archive"
    redacted_source_dir = str(redacted_files_dir) if redacted_files_dir is not None else None
    if redacted_files_dir is None:
        payload, redacted_source = _transient_redacted_v5_bundle_bytes(selected_path)
    else:
        secret_reasons = _v5_redacted_files_secret_reasons(redacted_files_dir)
        if secret_reasons:
            raise ValueError(
                "selected redacted V5 bundle contains possible secrets: "
                + "; ".join(secret_reasons)
            )
        payload = _tar_gz_bytes_from_directory(redacted_files_dir)

    embedded_name = _redacted_v5_bundle_name(bundle_name)
    member_path = f"{EMBEDDED_V5_BUNDLE_DIR}/{embedded_name}"
    embedded_sha = hashlib.sha256(payload).hexdigest()
    metadata = {
        "schema_version": "quant_lab_embedded_v5_bundle.v1",
        "present": True,
        "member_path": member_path,
        "bundle_name": embedded_name,
        "original_bundle_name": bundle_name,
        "source_path": str(selected_path),
        "source_sha256": validation.sha256,
        "sha256": embedded_sha,
        "redacted": True,
        "redacted_source": redacted_source,
        "redacted_source_dir": redacted_source_dir,
        "size_bytes": len(payload),
        "selected_v5_bundle_sha256": v5_context.get("selected_v5_bundle_sha256"),
        "selected_v5_bundle_manifest_bundle_sha256": v5_context.get(
            "selected_v5_bundle_manifest_bundle_sha256"
        ),
        "selected_v5_bundle_built_at": v5_context.get("selected_v5_bundle_built_at"),
        "selected_v5_bundle_ingested_at": v5_context.get("selected_v5_bundle_ingested_at"),
        "selected_v5_bundle_manifest_match": bool(
            v5_context.get("selected_v5_bundle_manifest_match")
        ),
        "validation_status": "valid",
        "validation_file_count": validation.file_count,
        "validation_total_uncompressed_size_bytes": validation.total_uncompressed_size_bytes,
        "validation_detected_files": validation.detected_files,
        "matches_selected": not selected_sha or actual_sha == selected_sha,
        "matches_manifest": not manifest_sha or actual_sha == manifest_sha,
    }
    members[member_path] = payload
    members[EMBEDDED_V5_BUNDLE_MANIFEST] = _json_text(metadata)
    v5_context.update(
        {
            "embedded_v5_bundle_present": True,
            "embedded_v5_bundle_member_path": member_path,
            "embedded_v5_bundle_manifest_path": EMBEDDED_V5_BUNDLE_MANIFEST,
            "embedded_v5_bundle_name": embedded_name,
            "embedded_v5_bundle_original_name": bundle_name,
            "embedded_v5_bundle_sha256": embedded_sha,
            "embedded_v5_bundle_source_sha256": validation.sha256,
            "embedded_v5_bundle_redacted": True,
            "embedded_v5_bundle_redacted_source": redacted_source,
            "embedded_v5_bundle_redacted_source_dir": redacted_source_dir,
            "embedded_v5_bundle_size_bytes": metadata["size_bytes"],
            "embedded_v5_bundle_matches_selected": metadata["matches_selected"],
            "embedded_v5_bundle_matches_manifest": metadata["matches_manifest"],
            "embedded_v5_bundle_validation_status": "valid",
            "embedded_v5_bundle_file_count": validation.file_count,
            "embedded_v5_bundle_detected_files": validation.detected_files,
            "embedded_v5_bundle_reason": "attached_selected_v5_bundle",
        }
    )


def _finalize_acceptance_set_context(v5_context: dict[str, Any]) -> None:
    acceptance_set_id = str(v5_context.get("acceptance_set_id") or "").strip()
    expected_sha = str(v5_context.get("expected_v5_bundle_sha256") or "").strip().lower()
    ingested_sha = str(v5_context.get("selected_v5_bundle_sha256") or "").strip().lower()
    embedded_source_sha = str(
        v5_context.get("embedded_v5_bundle_source_sha256") or ""
    ).strip().lower()
    quant_lab_commit = _git_commit_full()
    v5_commit = _selected_v5_bundle_git_commit(v5_context)
    matched = bool(
        acceptance_set_id
        and expected_sha
        and expected_sha == ingested_sha == embedded_source_sha
        and quant_lab_commit
        and v5_commit
    )
    v5_context.update(
        {
            "ingested_v5_bundle_sha256": ingested_sha or None,
            "quant_lab_commit": quant_lab_commit,
            "v5_commit": v5_commit,
            "acceptance_set_matched": matched if acceptance_set_id else None,
            "acceptance_set_sha256_relationship": {
                "expected": expected_sha or None,
                "ingested": ingested_sha or None,
                "embedded_source": embedded_source_sha or None,
                "all_equal": matched if acceptance_set_id else None,
            },
        }
    )
    if not acceptance_set_id:
        return
    missing = [
        name
        for name, value in (
            ("expected_v5_bundle_sha256", expected_sha),
            ("ingested_v5_bundle_sha256", ingested_sha),
            ("embedded_v5_bundle_source_sha256", embedded_source_sha),
            ("quant_lab_commit", quant_lab_commit),
            ("v5_commit", v5_commit),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "acceptance_set_required_field_missing:" + ",".join(missing)
        )
    if not matched:
        raise RuntimeError(
            "acceptance_set_bundle_sha_mismatch:"
            f"expected={expected_sha};ingested={ingested_sha};"
            f"embedded_source={embedded_source_sha}"
        )


def _promote_embedded_v5_summary_report(
    members: dict[str, _MemberPayload],
    v5_context: dict[str, Any],
    *,
    summary_member: str,
    dest_member: str,
) -> None:
    schema = CSV_SCHEMAS.get(dest_member)
    if not schema:
        return
    empty = _csv_member(dest_member, _empty_csv_schema_frame(dest_member))
    existing_rows = _csv_member_rows(members.get(dest_member))
    embedded_member = str(v5_context.get("embedded_v5_bundle_member_path") or "").strip()
    payload = members.get(embedded_member)
    if not isinstance(payload, bytes):
        if not existing_rows:
            members[dest_member] = empty
        v5_context.setdefault("warnings", []).append(
            f"embedded_v5_summary_missing:{summary_member}:embedded_bundle_not_available"
        )
        return
    candidates = {
        summary_member.strip("/"),
        f"reports/{summary_member.strip('/')}",
        Path(summary_member).name,
    }
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            selected = None
            for member in archive.getmembers():
                normalized = member.name.replace("\\", "/").lstrip("./")
                if member.isfile() and normalized in candidates:
                    selected = member
                    break
            if selected is None:
                if not existing_rows:
                    members[dest_member] = empty
                v5_context.setdefault("warnings", []).append(
                    f"embedded_v5_summary_missing:{summary_member}:member_not_found"
                )
                return
            handle = archive.extractfile(selected)
            if handle is None:
                if not existing_rows:
                    members[dest_member] = empty
                v5_context.setdefault("warnings", []).append(
                    f"embedded_v5_summary_missing:{summary_member}:member_unreadable"
                )
                return
            text = handle.read().decode("utf-8-sig", "replace")
    except Exception as exc:
        if not existing_rows:
            members[dest_member] = empty
        v5_context.setdefault("warnings", []).append(
            f"embedded_v5_summary_missing:{summary_member}:{type(exc).__name__}"
        )
        return
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except Exception:
        rows = []
    if not rows:
        if not existing_rows:
            members[dest_member] = empty
        return
    frame = _paper_strategy_ack_report_frame([*existing_rows, *rows], path=dest_member)
    members[dest_member] = _csv_member(dest_member, frame)


def _csv_member_rows(payload: _MemberPayload | None) -> list[dict[str, Any]]:
    if not isinstance(payload, str) or not payload.strip():
        return []
    try:
        return list(csv.DictReader(io.StringIO(payload)))
    except Exception:
        return []


def _is_expected_v5_followup_bundle_name(name: str) -> bool:
    return name.startswith(V5_FOLLOWUP_BUNDLE_PREFIX) and (
        name.endswith(".tar.gz") or name.endswith(".tgz")
    )


def _selected_v5_redacted_files_dir(
    v5_context: dict[str, Any],
    *,
    selected_path: Path,
    selected_sha256: str,
) -> Path | None:
    redacted_archive_root = str(v5_context.get("redacted_archive_dir") or "").strip()
    if not redacted_archive_root:
        return None
    bundle_ts = (
        _parse_v5_context_ts(v5_context.get("selected_v5_bundle_built_at"))
        or _parse_v5_context_ts(v5_context.get("selected_v5_bundle_manifest_bundle_ts"))
        or _bundle_ts_for_path(selected_path)
    )
    if bundle_ts is None:
        return None
    candidate = (
        Path(redacted_archive_root)
        / bundle_ts.date().isoformat()
        / selected_sha256
        / "redacted_files"
    )
    return candidate if candidate.is_dir() else None


def _redacted_v5_bundle_name(bundle_name: str) -> str:
    if bundle_name.endswith(".tar.gz"):
        return bundle_name.removesuffix(".tar.gz") + ".redacted.tar.gz"
    if bundle_name.endswith(".tgz"):
        return bundle_name.removesuffix(".tgz") + ".redacted.tgz"
    return bundle_name + ".redacted.tar.gz"


def _v5_redacted_files_secret_reasons(path: Path) -> list[str]:
    reasons: list[str] = []
    for member in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = member.relative_to(path).as_posix()
        if not _is_v5_bundle_text_member(relative):
            continue
        try:
            high, medium = _secret_severity_counts_in_file(member)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            reasons.append(f"{relative} read failed: {exc}")
            continue
        if high or medium:
            reasons.append(f"{relative}: {high} high, {medium} medium")
    return reasons


def _secret_severity_counts_in_file(path: Path) -> tuple[int, int]:
    high = 0
    medium = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not _line_may_contain_secret(line):
                continue
            for pattern, severity, _label in SECRET_PATTERNS:
                if not pattern.search(line):
                    continue
                if severity == "high":
                    high += 1
                elif severity == "medium":
                    medium += 1
    return high, medium


def _transient_redacted_v5_bundle_bytes(path: Path) -> tuple[bytes, str]:
    limits = BundleLimits()
    with tempfile.TemporaryDirectory(prefix="quant_lab_embed_v5_") as temp_name:
        temp_root = Path(temp_name)
        extracted_dir = temp_root / "extracted"
        redacted_dir = temp_root / "redacted"
        safe_extract_v5_bundle(path, extracted_dir, limits)
        redact_extracted_bundle(extracted_dir, redacted_dir)
        secret_reasons = _v5_redacted_files_secret_reasons(redacted_dir)
        if secret_reasons:
            raise ValueError(
                "transient redacted V5 bundle contains possible secrets: "
                + "; ".join(secret_reasons)
            )
        return _tar_gz_bytes_from_directory(redacted_dir), "transient_redaction"


def _is_v5_bundle_text_member(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in V5_BUNDLE_SECRET_SCAN_TEXT_SUFFIXES


def _tar_gz_bytes_from_directory(source_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(item for item in source_dir.rglob("*") if item.is_file()):
            arcname = path.relative_to(source_dir).as_posix()
            info = archive.gettarinfo(str(path), arcname=arcname)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return buffer.getvalue()


def _manifest_payload(
    *,
    day: date,
    generated_at: datetime,
    export_finished_at: datetime,
    root: Path,
    profile: str,
    missing_sections: list[str],
    warnings: list[str],
    members: dict[str, _MemberPayload],
    row_counts: dict[str, int],
    command_line: list[str],
    snapshot: _DatasetSnapshot,
    pre_export_v5: dict[str, Any] | None = None,
    allow_stale_v5: bool = False,
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
    stale_v5_bundle = bool(v5_context.get("stale_v5_bundle"))
    authoritative_snapshot = bool(v5_context.get("authoritative_snapshot"))
    return {
        "export_date": day.isoformat(),
        "generated_at": generated_at.isoformat(),
        "export_generated_at": generated_at.isoformat(),
        "export_started_at": generated_at.isoformat(),
        "export_finished_at": export_finished_at.isoformat(),
        "display_timezone": DISPLAY_TIMEZONE,
        "generated_at_beijing": beijing_iso(generated_at),
        "export_generated_at_beijing": beijing_iso(generated_at),
        "export_started_at_beijing": beijing_iso(generated_at),
        "export_finished_at_beijing": beijing_iso(export_finished_at),
        "risk_permission_generated_at": risk_export_status.get("risk_permission_generated_at"),
        "risk_permission_expires_at": risk_export_status.get("risk_permission_expires_at"),
        "permission_expired_at_export": risk_export_status.get("permission_expired_at_export"),
        "profile": profile,
        "quant_lab_version": __version__,
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
        "schema_version": V5_TELEMETRY_DATASET_SCHEMA_VERSION,
        "latest_v5_bundle_seen_at_export": v5_context.get("latest_v5_bundle_seen_at_export"),
        "latest_v5_bundle_ingested_at_export": _iso_or_none(latest_ingested_bundle_ts),
        "authoritative_snapshot": authoritative_snapshot,
        "stale_v5_bundle": stale_v5_bundle,
        "allow_stale_v5": allow_stale_v5,
        "acceptance_set_id": v5_context.get("acceptance_set_id"),
        "expected_v5_bundle_sha256": v5_context.get("expected_v5_bundle_sha256"),
        "ingested_v5_bundle_sha256": v5_context.get("ingested_v5_bundle_sha256"),
        "acceptance_set_matched": v5_context.get("acceptance_set_matched"),
        "acceptance_set_sha256_relationship": v5_context.get(
            "acceptance_set_sha256_relationship"
        ),
        "quant_lab_commit": v5_context.get("quant_lab_commit") or git["git_commit"],
        "v5_commit": v5_context.get("v5_commit"),
        "expected_v5_bundle_matched": v5_context.get("expected_v5_bundle_matched"),
        "selected_v5_bundle_path": v5_context.get("selected_v5_bundle_path"),
        "selected_v5_bundle_sha256": v5_context.get("selected_v5_bundle_sha256"),
        "selected_v5_bundle_built_at": v5_context.get("selected_v5_bundle_built_at"),
        "selected_v5_bundle_ingested_at": v5_context.get("selected_v5_bundle_ingested_at"),
        "selected_v5_bundle_manifest_match": bool(
            v5_context.get("selected_v5_bundle_manifest_match")
        ),
        "selected_v5_bundle_authoritative_reason": v5_context.get(
            "selected_v5_bundle_authoritative_reason"
        ),
        "selected_v5_bundle_manifest_ingest_ts": v5_context.get(
            "selected_v5_bundle_manifest_ingest_ts"
        ),
        "selected_v5_bundle_manifest_bundle_ts": v5_context.get(
            "selected_v5_bundle_manifest_bundle_ts"
        ),
        "selected_v5_bundle_manifest_bundle_name": v5_context.get(
            "selected_v5_bundle_manifest_bundle_name"
        ),
        "selected_v5_bundle_manifest_bundle_sha256": v5_context.get(
            "selected_v5_bundle_manifest_bundle_sha256"
        )
        or v5_context.get("selected_v5_bundle_sha256"),
        "selected_v5_bundle_event_counts": v5_context.get(
            "selected_v5_bundle_event_counts",
            {},
        ),
        "embedded_v5_bundle_present": bool(v5_context.get("embedded_v5_bundle_present")),
        "embedded_v5_bundle_member_path": v5_context.get("embedded_v5_bundle_member_path"),
        "embedded_v5_bundle_manifest_path": v5_context.get("embedded_v5_bundle_manifest_path"),
        "embedded_v5_bundle_name": v5_context.get("embedded_v5_bundle_name"),
        "embedded_v5_bundle_original_name": v5_context.get("embedded_v5_bundle_original_name"),
        "embedded_v5_bundle_sha256": v5_context.get("embedded_v5_bundle_sha256"),
        "embedded_v5_bundle_source_sha256": v5_context.get("embedded_v5_bundle_source_sha256"),
        "embedded_v5_bundle_redacted": bool(v5_context.get("embedded_v5_bundle_redacted")),
        "embedded_v5_bundle_redacted_source": v5_context.get("embedded_v5_bundle_redacted_source"),
        "embedded_v5_bundle_size_bytes": v5_context.get(
            "embedded_v5_bundle_size_bytes",
            0,
        ),
        "embedded_v5_bundle_matches_selected": bool(
            v5_context.get("embedded_v5_bundle_matches_selected")
        ),
        "embedded_v5_bundle_matches_manifest": bool(
            v5_context.get("embedded_v5_bundle_matches_manifest")
        ),
        "embedded_v5_bundle_validation_status": v5_context.get(
            "embedded_v5_bundle_validation_status"
        ),
        "embedded_v5_bundle_reason": v5_context.get("embedded_v5_bundle_reason"),
        "github_ci_status": v5_context.get("github_ci_status"),
        "lake_file_index_refresh": v5_context.get("lake_file_index_refresh"),
        "lake_file_health": v5_context.get("lake_file_health"),
        "v5_export_consistency": {
            "authoritative_snapshot": authoritative_snapshot,
            "stale_v5_bundle": stale_v5_bundle,
            "warning_reason": v5_context.get("warning_reason"),
            "latest_v5_bundle_seen_at_export": v5_context.get("latest_v5_bundle_seen_at_export"),
            "latest_v5_bundle_ingested_at_export": _iso_or_none(latest_ingested_bundle_ts),
            "selected_v5_bundle_sha256": v5_context.get("selected_v5_bundle_sha256"),
            "acceptance_set_id": v5_context.get("acceptance_set_id"),
            "expected_v5_bundle_sha256": v5_context.get("expected_v5_bundle_sha256"),
            "ingested_v5_bundle_sha256": v5_context.get(
                "ingested_v5_bundle_sha256"
            ),
            "acceptance_set_matched": v5_context.get("acceptance_set_matched"),
            "acceptance_set_sha256_relationship": v5_context.get(
                "acceptance_set_sha256_relationship"
            ),
            "expected_v5_bundle_matched": v5_context.get("expected_v5_bundle_matched"),
            "selected_v5_bundle_manifest_match": bool(
                v5_context.get("selected_v5_bundle_manifest_match")
            ),
            "selected_v5_bundle_manifest_bundle_sha256": v5_context.get(
                "selected_v5_bundle_manifest_bundle_sha256"
            )
            or v5_context.get("selected_v5_bundle_sha256"),
            "selected_v5_bundle_authoritative_reason": v5_context.get(
                "selected_v5_bundle_authoritative_reason"
            ),
            "audit_bundle_count": v5_context.get("audit_bundle_count", 0),
            "scanned_bundle_count": v5_context.get("scanned_bundle_count", 0),
            "selected_bundle_count": v5_context.get("selected_bundle_count", 0),
            "unscanned_bundle_count": v5_context.get("unscanned_bundle_count", 0),
            "pending_uningested_bundle_count": v5_context.get(
                "pending_uningested_bundle_count",
                0,
            ),
            "pending_uningested_bundle_names": v5_context.get(
                "pending_uningested_bundle_names",
                [],
            ),
            "unreadable_bundle_count": v5_context.get("unreadable_bundle_count", 0),
            "unreadable_bundle_names": v5_context.get("unreadable_bundle_names", []),
            "historical_gap_detected": bool(v5_context.get("historical_gap_detected")),
            "historical_gap_reason": v5_context.get("historical_gap_reason"),
            "possible_historical_gap": bool(v5_context.get("possible_historical_gap")),
            "max_scan_bundles": v5_context.get("max_scan_bundles"),
            "max_pending_bundles": v5_context.get("max_pending_bundles"),
            "embedded_v5_bundle_present": bool(v5_context.get("embedded_v5_bundle_present")),
            "embedded_v5_bundle_member_path": v5_context.get("embedded_v5_bundle_member_path"),
            "embedded_v5_bundle_sha256": v5_context.get("embedded_v5_bundle_sha256"),
            "embedded_v5_bundle_source_sha256": v5_context.get("embedded_v5_bundle_source_sha256"),
            "embedded_v5_bundle_redacted": bool(v5_context.get("embedded_v5_bundle_redacted")),
            "embedded_v5_bundle_redacted_source": v5_context.get(
                "embedded_v5_bundle_redacted_source"
            ),
            "embedded_v5_bundle_matches_selected": bool(
                v5_context.get("embedded_v5_bundle_matches_selected")
            ),
            "embedded_v5_bundle_validation_status": v5_context.get(
                "embedded_v5_bundle_validation_status"
            ),
            "github_ci_status": v5_context.get("github_ci_status"),
            "lake_file_index_refresh": v5_context.get("lake_file_index_refresh"),
            "lake_file_health": v5_context.get("lake_file_health"),
        },
        "scanned_bundle_count": v5_context.get("scanned_bundle_count", 0),
        "audit_bundle_count": v5_context.get("audit_bundle_count", 0),
        "selected_bundle_count": v5_context.get("selected_bundle_count", 0),
        "unscanned_bundle_count": v5_context.get("unscanned_bundle_count", 0),
        "oldest_scanned_bundle_ts": v5_context.get("oldest_scanned_bundle_ts"),
        "newest_scanned_bundle_ts": v5_context.get("newest_scanned_bundle_ts"),
        "possible_historical_gap": bool(v5_context.get("possible_historical_gap")),
        "historical_gap_detected": bool(v5_context.get("historical_gap_detected")),
        "historical_gap_reason": v5_context.get("historical_gap_reason"),
        "pending_uningested_bundle_count": v5_context.get(
            "pending_uningested_bundle_count",
            0,
        ),
        "pending_uningested_bundle_names": v5_context.get(
            "pending_uningested_bundle_names",
            [],
        ),
        "pending_uningested_bundle_sha256s": v5_context.get(
            "pending_uningested_bundle_sha256s",
            [],
        ),
        "unreadable_bundle_count": v5_context.get("unreadable_bundle_count", 0),
        "unreadable_bundle_names": v5_context.get("unreadable_bundle_names", []),
        "max_scan_bundles": v5_context.get("max_scan_bundles"),
        "max_pending_bundles": v5_context.get("max_pending_bundles"),
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
        "restricted_archive_dir": None,
        "redacted_archive_dir": None,
        "selected_v5_bundle_path": None,
        "selected_v5_bundle_sha256": None,
        "selected_v5_bundle_built_at": None,
        "selected_v5_bundle_ingested_at": None,
        "selected_v5_bundle_manifest_match": False,
        "selected_v5_bundle_manifest_ingest_ts": None,
        "selected_v5_bundle_manifest_bundle_ts": None,
        "selected_v5_bundle_manifest_bundle_name": None,
        "selected_v5_bundle_manifest_bundle_sha256": None,
        "selected_v5_bundle_event_counts": {},
        "scanned_bundle_count": 0,
        "audit_bundle_count": 0,
        "selected_bundle_count": 0,
        "selected_bundle_names": [],
        "unscanned_bundle_count": 0,
        "oldest_scanned_bundle_ts": None,
        "newest_scanned_bundle_ts": None,
        "possible_historical_gap": False,
        "historical_gap_detected": False,
        "historical_gap_reason": "",
        "pending_uningested_bundle_count": 0,
        "pending_uningested_bundle_names": [],
        "pending_uningested_bundle_sha256s": [],
        "unreadable_bundle_count": 0,
        "unreadable_bundle_names": [],
        "max_scan_bundles": _export_v5_max_scan_bundles(_export_v5_max_pending_bundles()),
        "max_pending_bundles": _export_v5_max_pending_bundles(),
        "remote_pull_attempted": False,
        "remote_pull_success": None,
        "remote_pulled_count": 0,
        "remote_pulled_names": [],
        "remote_skipped_count": 0,
        "warnings": [],
    }
    warnings: list[str] = []
    try:
        cfg = _load_export_v5_config(lake_root, config_path=config_path)
    except Exception as exc:  # pragma: no cover - production config errors vary.
        context["warnings"] = [
            f"v5_pre_export_config_failed: {type(exc).__name__}: {_safe_warning_text(str(exc))}"
        ]
        return context

    context.update(_pull_latest_v5_remote_for_export(cfg))

    inbox_dir = Path(cfg["local_inbox_dir"])
    context["local_inbox_dir"] = str(inbox_dir)
    context["restricted_archive_dir"] = str(Path(cfg["restricted_archive_dir"]))
    context["redacted_archive_dir"] = str(Path(cfg["redacted_archive_dir"]))
    latest_seen_path = _latest_v5_bundle_path_in_inbox(inbox_dir)
    latest_seen = _bundle_ts_for_path(latest_seen_path) if latest_seen_path is not None else None
    context["latest_v5_bundle_seen_at_export"] = _iso_or_none(latest_seen)
    if latest_seen_path is not None:
        context["selected_v5_bundle_path"] = str(latest_seen_path)
        context["selected_v5_bundle_built_at"] = _iso_or_none(latest_seen)
        try:
            latest_seen_sha = compute_sha256(latest_seen_path)
            context["selected_v5_bundle_sha256"] = latest_seen_sha
            manifest_row = _bundle_manifest_row_by_sha(lake_root, latest_seen_sha)
            if manifest_row:
                _apply_selected_bundle_manifest_context(context, manifest_row)
        except OSError:
            latest_seen_sha = None
    if not inbox_dir.exists():
        context["warnings"] = []
        return context

    try:
        from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle

        scan = _pending_v5_bundle_scan_for_export(inbox_dir, lake_root)
        context.update(scan["context"])
        pending_paths = scan["pending_paths"]
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
            if result.bundle_sha256 == context.get("selected_v5_bundle_sha256"):
                context["selected_v5_bundle_event_counts"] = {
                    **result.silver_rows,
                    **result.gold_rows,
                }
                manifest_row = _bundle_manifest_row_by_sha(lake_root, result.bundle_sha256)
                if manifest_row:
                    _apply_selected_bundle_manifest_context(context, manifest_row)
            warnings.extend(str(warning) for warning in result.warnings)
        context["processed_bundle_count"] = processed_count
        context["skipped_bundle_count"] = skipped_count
        post_scan = _pending_v5_bundle_scan_for_export(inbox_dir, lake_root)
        _apply_post_refresh_v5_scan_context(context, post_scan["context"])
        selected_sha = str(context.get("selected_v5_bundle_sha256") or "").strip()
        if selected_sha:
            manifest_row = _bundle_manifest_row_by_sha(lake_root, selected_sha)
            if manifest_row:
                _apply_selected_bundle_manifest_context(context, manifest_row)
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
            "_remote_config": cfg,
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


def _pull_latest_v5_remote_for_export(cfg: dict[str, Any]) -> dict[str, Any]:
    if not _flag_enabled("QUANT_LAB_EXPORT_PULL_V5_REMOTE", default=False):
        return {
            "remote_pull_attempted": False,
            "remote_pull_success": None,
            "remote_pulled_count": 0,
            "remote_pulled_names": [],
            "remote_skipped_count": 0,
        }

    remote_config = cfg.get("_remote_config")
    if remote_config is None:
        raise RuntimeError(
            "v5_pre_export_remote_pull_failed: remote telemetry config is unavailable"
        )

    from quant_lab.strategy_telemetry.remote_pull import RemoteBundlePuller

    pull = RemoteBundlePuller().pull_bundles(remote_config, max_files=1)
    warnings = [str(warning) for warning in (pull.warnings or [])]
    if pull.dry_run:
        warnings.append("remote pull unexpectedly ran in dry-run mode")
    if warnings:
        detail = "; ".join(_safe_warning_text(warning) for warning in warnings[:5])
        raise RuntimeError(f"v5_pre_export_remote_pull_failed: {detail}")
    return {
        "remote_pull_attempted": True,
        "remote_pull_success": True,
        "remote_pulled_count": len(pull.pulled_files),
        "remote_pulled_names": [Path(path).name for path in pull.pulled_files],
        "remote_skipped_count": len(pull.skipped_files),
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
        "latest_v5_bundle_seen_at_export": None,
        "local_inbox_dir": None,
        "restricted_archive_dir": None,
        "redacted_archive_dir": None,
        "selected_v5_bundle_path": None,
        "selected_v5_bundle_sha256": None,
        "selected_v5_bundle_built_at": None,
        "selected_v5_bundle_ingested_at": None,
        "selected_v5_bundle_manifest_match": False,
        "selected_v5_bundle_manifest_ingest_ts": None,
        "selected_v5_bundle_manifest_bundle_ts": None,
        "selected_v5_bundle_manifest_bundle_name": None,
        "selected_v5_bundle_manifest_bundle_sha256": None,
        "selected_v5_bundle_event_counts": {},
        "scanned_bundle_count": 0,
        "audit_bundle_count": 0,
        "selected_bundle_count": 0,
        "selected_bundle_names": [],
        "unscanned_bundle_count": 0,
        "oldest_scanned_bundle_ts": None,
        "newest_scanned_bundle_ts": None,
        "possible_historical_gap": False,
        "historical_gap_detected": False,
        "historical_gap_reason": "",
        "pending_uningested_bundle_count": 0,
        "pending_uningested_bundle_names": [],
        "pending_uningested_bundle_sha256s": [],
        "unreadable_bundle_count": 0,
        "unreadable_bundle_names": [],
        "max_scan_bundles": _export_v5_max_scan_bundles(_export_v5_max_pending_bundles()),
        "max_pending_bundles": _export_v5_max_pending_bundles(),
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
    context["restricted_archive_dir"] = str(Path(cfg["restricted_archive_dir"]))
    context["redacted_archive_dir"] = str(Path(cfg["redacted_archive_dir"]))
    latest_seen_path = _latest_v5_bundle_path_in_inbox(inbox_dir)
    latest_seen = _bundle_ts_for_path(latest_seen_path) if latest_seen_path is not None else None
    context["latest_v5_bundle_seen_at_export"] = _iso_or_none(latest_seen)
    if latest_seen_path is not None:
        context["selected_v5_bundle_path"] = str(latest_seen_path)
        context["selected_v5_bundle_built_at"] = _iso_or_none(latest_seen)
        try:
            latest_seen_sha = compute_sha256(latest_seen_path)
            context["selected_v5_bundle_sha256"] = latest_seen_sha
            manifest_row = _bundle_manifest_row_by_sha(lake_root, latest_seen_sha)
            if manifest_row:
                _apply_selected_bundle_manifest_context(context, manifest_row)
        except OSError:
            pass
    if inbox_dir.exists():
        scan = _pending_v5_bundle_scan_for_export(inbox_dir, lake_root)
        context.update(scan["context"])
    return context


def _latest_v5_bundle_ts_in_inbox(inbox_dir: Path) -> datetime | None:
    latest_path = _latest_v5_bundle_path_in_inbox(inbox_dir)
    return _bundle_ts_for_path(latest_path) if latest_path is not None else None


def _latest_v5_bundle_path_in_inbox(inbox_dir: Path) -> Path | None:
    if not inbox_dir.exists():
        return None
    from quant_lab.strategy_telemetry.bundle import parse_bundle_ts

    candidates: list[tuple[datetime, Path]] = []
    for path in inbox_dir.glob("v5_live_followup_bundle_*.tar.gz"):
        parsed = parse_bundle_ts(path.name)
        if parsed is None:
            try:
                parsed = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError:
                continue
        candidates.append((parsed, path))
    return max(candidates, key=lambda item: (item[0], item[1].name))[1] if candidates else None


def _bundle_ts_for_path(path: Path | None) -> datetime | None:
    if path is None:
        return None
    from quant_lab.strategy_telemetry.bundle import parse_bundle_ts

    parsed = parse_bundle_ts(path.name)
    if parsed is not None:
        return parsed
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None


def _pending_v5_bundle_paths_for_export(inbox_dir: Path, lake_root: Path) -> list[Path]:
    return _pending_v5_bundle_scan_for_export(inbox_dir, lake_root)["pending_paths"]


def _pending_v5_bundle_scan_for_export(inbox_dir: Path, lake_root: Path) -> dict[str, Any]:
    max_pending = _export_v5_max_pending_bundles()
    max_scan = _export_v5_max_scan_bundles(max_pending)
    ingested_sha256s, ingested_names = _ingested_bundle_manifest_keys(lake_root)
    from quant_lab.strategy_telemetry.bundle import compute_sha256, parse_bundle_ts

    def sort_key(path: Path) -> tuple[datetime, str]:
        parsed = parse_bundle_ts(path.name)
        if parsed is None:
            try:
                parsed = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError:
                parsed = datetime.min.replace(tzinfo=UTC)
        return parsed, path.name

    all_paths = sorted(
        inbox_dir.glob("v5_live_followup_bundle_*.tar.gz"),
        key=sort_key,
        reverse=True,
    )
    scanned_paths = all_paths[:max_scan]
    scanned_ts = [sort_key(path)[0] for path in scanned_paths]
    pending: list[tuple[Path, str, datetime]] = []
    unreadable: list[Path] = []
    for path in scanned_paths:
        bundle_ts = sort_key(path)[0]
        try:
            sha256 = compute_sha256(path)
        except OSError:
            unreadable.append(path)
            continue
        if sha256 in ingested_sha256s:
            continue
        pending.append((path, sha256, bundle_ts))
    pending_scanned = pending
    selected = [item[0] for item in pending_scanned[:max_pending]]
    selected = selected[:max_pending]
    unscanned_count = max(len(all_paths) - len(scanned_paths), 0)
    unscanned_pending_names = [
        path.name for path in all_paths[max_scan:] if path.name not in ingested_names
    ]
    pending_uningested_count = len(pending_scanned) + len(unscanned_pending_names)
    pending_uningested_names = [
        *[item[0].name for item in pending_scanned],
        *unscanned_pending_names,
    ]
    pending_uningested_sha256s = [item[1] for item in pending_scanned]
    unreadable_names = [path.name for path in unreadable]
    pending_outside_selection_scan = unscanned_pending_names
    historical_gap_reasons: list[str] = []
    if unreadable:
        historical_gap_reasons.append("unreadable_bundle_sha256")
    if pending_uningested_count > max_pending:
        historical_gap_reasons.append("pending_uningested_bundle_count_exceeds_max_pending")
    if pending_outside_selection_scan:
        historical_gap_reasons.append("pending_uningested_bundle_outside_selection_scan")
    if pending_uningested_count > len(selected) and not historical_gap_reasons:
        historical_gap_reasons.append("pending_uningested_bundles_not_selected_for_refresh")
    historical_gap_detected = bool(historical_gap_reasons)
    historical_gap_reason = ";".join(historical_gap_reasons)
    context = {
        "scanned_bundle_count": len(scanned_paths),
        "audit_bundle_count": len(all_paths),
        "selected_bundle_count": len(selected),
        "selected_bundle_names": [path.name for path in selected],
        "unscanned_bundle_count": unscanned_count,
        "oldest_scanned_bundle_ts": _iso_or_none(min(scanned_ts)) if scanned_ts else None,
        "newest_scanned_bundle_ts": _iso_or_none(max(scanned_ts)) if scanned_ts else None,
        "possible_historical_gap": historical_gap_detected,
        "historical_gap_detected": historical_gap_detected,
        "historical_gap_reason": historical_gap_reason,
        "max_scan_bundles": max_scan,
        "max_pending_bundles": max_pending,
        "pending_scanned_bundle_count": len(pending_scanned),
        "pending_unscanned_bundle_count": len(unscanned_pending_names),
        "pending_uningested_bundle_count": pending_uningested_count,
        "pending_uningested_bundle_names": pending_uningested_names,
        "pending_uningested_bundle_sha256s": pending_uningested_sha256s,
        "unreadable_bundle_count": len(unreadable),
        "unreadable_bundle_names": unreadable_names,
        "total_inbox_bundle_count": len(all_paths),
    }
    return {"pending_paths": list(reversed(selected)), "context": context}


def _apply_post_refresh_v5_scan_context(
    context: dict[str, Any],
    scan_context: dict[str, Any],
) -> None:
    for key in (
        "audit_bundle_count",
        "scanned_bundle_count",
        "unscanned_bundle_count",
        "oldest_scanned_bundle_ts",
        "newest_scanned_bundle_ts",
        "possible_historical_gap",
        "historical_gap_detected",
        "historical_gap_reason",
        "max_scan_bundles",
        "max_pending_bundles",
        "pending_scanned_bundle_count",
        "pending_uningested_bundle_count",
        "pending_uningested_bundle_names",
        "pending_uningested_bundle_sha256s",
        "unreadable_bundle_count",
        "unreadable_bundle_names",
        "total_inbox_bundle_count",
    ):
        context[key] = scan_context.get(key)


def _apply_selected_bundle_manifest_context(
    context: dict[str, Any],
    manifest_row: dict[str, Any],
) -> None:
    context["selected_v5_bundle_ingested_at"] = _iso_or_none(
        _parse_v5_context_ts(manifest_row.get("ingest_ts"))
    )
    context["selected_v5_bundle_manifest_ingest_ts"] = context["selected_v5_bundle_ingested_at"]
    context["selected_v5_bundle_manifest_bundle_ts"] = _iso_or_none(
        _parse_v5_context_ts(manifest_row.get("bundle_ts"))
    )
    context["selected_v5_bundle_manifest_bundle_name"] = manifest_row.get("bundle_name")
    context["selected_v5_bundle_manifest_bundle_sha256"] = manifest_row.get(
        "bundle_sha256"
    ) or context.get("selected_v5_bundle_sha256")
    context["selected_v5_bundle_manifest_match"] = True


def _export_v5_max_pending_bundles() -> int:
    raw_value = os.environ.get("QUANT_LAB_EXPORT_V5_MAX_PENDING_BUNDLES", "1")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 1


def _export_v5_max_scan_bundles(max_pending: int) -> int:
    raw_value = os.environ.get("QUANT_LAB_EXPORT_V5_MAX_SCAN_BUNDLES", "1000")
    try:
        return max(max_pending, int(raw_value))
    except ValueError:
        return max(max_pending, 1000)


def _ingested_bundle_manifest_keys(lake_root: Path) -> tuple[set[str], set[str]]:
    path = lake_root / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest"
    try:
        frame = read_parquet_dataset(path)
    except Exception:
        return set(), set()
    if frame.is_empty():
        return set(), set()
    sha256s = set()
    names = set()
    if "bundle_sha256" in frame.columns:
        sha256s = {
            str(value)
            for value in frame.get_column("bundle_sha256").drop_nulls().unique().to_list()
            if value
        }
    if "bundle_name" in frame.columns:
        names = {
            str(value)
            for value in frame.get_column("bundle_name").drop_nulls().unique().to_list()
            if value
        }
    return sha256s, names


def _bundle_manifest_row_by_sha(lake_root: Path, sha256: str | None) -> dict[str, Any] | None:
    if not sha256:
        return None
    path = lake_root / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest"
    try:
        frame = read_parquet_dataset(path)
    except Exception:
        return None
    if frame.is_empty() or "bundle_sha256" not in frame.columns:
        return None
    rows = frame.filter(pl.col("bundle_sha256").cast(pl.Utf8) == str(sha256)).to_dicts()
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _parse_v5_context_ts(row.get("ingest_ts")) or datetime.min.replace(tzinfo=UTC)
        ),
    )


def _v5_export_consistency(
    frames: dict[str, pl.DataFrame],
    *,
    pre_export_v5: dict[str, Any],
    pre_export_v5_refresh: bool,
    allow_stale_v5: bool,
) -> dict[str, Any]:
    latest_seen = _parse_v5_context_ts(pre_export_v5.get("latest_v5_bundle_seen_at_export"))
    latest_local_ingested = _latest_v5_bundle_ts(frames)
    selected_sha = str(pre_export_v5.get("selected_v5_bundle_sha256") or "").strip()
    expected_sha = str(pre_export_v5.get("expected_v5_bundle_sha256") or "").strip()
    acceptance_set_id = str(pre_export_v5.get("acceptance_set_id") or "").strip()
    selected_ingested_at = _parse_v5_context_ts(pre_export_v5.get("selected_v5_bundle_ingested_at"))
    latest_ingested = _max_dt(latest_local_ingested, selected_ingested_at)
    manifest_match = bool(pre_export_v5.get("selected_v5_bundle_manifest_match"))
    historical_gap_detected = bool(
        pre_export_v5.get("historical_gap_detected") or pre_export_v5.get("possible_historical_gap")
    )
    historical_gap_reason = str(pre_export_v5.get("historical_gap_reason") or "").strip()
    pending_uningested_bundle_count = int(pre_export_v5.get("pending_uningested_bundle_count") or 0)
    has_v5_bundle_context = _v5_context_requires_authoritative_failure(pre_export_v5)
    stale, why_stale = _v5_bundle_stale_state(
        latest_remote=latest_seen,
        latest_ingested=latest_ingested,
    )
    reason = ""
    if stale:
        reason = (
            "stale_v5_bundle_at_export:"
            f"latest_seen={_iso_or_none(latest_seen)};"
            f"latest_ingested={_iso_or_none(latest_ingested)};"
            f"why_stale={why_stale};"
            f"pre_export_v5_refresh={pre_export_v5_refresh};"
            f"allow_stale_v5={allow_stale_v5}"
        )
    authoritative_blockers: list[str] = []
    if not pre_export_v5_refresh:
        authoritative_blockers.append("pre_export_v5_refresh_disabled")
    if stale:
        authoritative_blockers.append("stale_v5_bundle")
    if not selected_sha:
        authoritative_blockers.append("selected_v5_bundle_sha256_missing")
    if acceptance_set_id and not expected_sha:
        authoritative_blockers.append("acceptance_set_expected_v5_bundle_sha256_missing")
    if expected_sha and selected_sha and expected_sha.lower() != selected_sha.lower():
        authoritative_blockers.append(
            f"expected_v5_bundle_sha256_mismatch:expected={expected_sha};selected={selected_sha}"
        )
    if selected_ingested_at is None:
        authoritative_blockers.append("selected_v5_bundle_ingested_at_missing")
    if not manifest_match:
        authoritative_blockers.append("selected_v5_bundle_manifest_missing")
    if historical_gap_detected:
        authoritative_blockers.append(
            f"historical_gap_detected:{historical_gap_reason}"
            if historical_gap_reason
            else "historical_gap_detected"
        )
    if pending_uningested_bundle_count > 0:
        authoritative_blockers.append(
            f"pending_uningested_bundle_count={pending_uningested_bundle_count}"
        )
    authoritative = not authoritative_blockers
    authoritative_reason = (
        "selected_bundle_sha_matched_bundle_manifest"
        if authoritative
        else ";".join(authoritative_blockers)
    )
    if not reason and authoritative_blockers and has_v5_bundle_context and pre_export_v5_refresh:
        reason = (
            "non_authoritative_v5_snapshot:"
            f"reason={authoritative_reason};"
            f"latest_seen={_iso_or_none(latest_seen)};"
            f"latest_ingested={_iso_or_none(latest_ingested)};"
            f"pre_export_v5_refresh={pre_export_v5_refresh};"
            f"allow_stale_v5={allow_stale_v5}"
        )
    return {
        "authoritative_snapshot": authoritative,
        "stale_v5_bundle": stale,
        "why_stale": why_stale,
        "warning_reason": reason,
        "selected_v5_bundle_authoritative_reason": authoritative_reason,
        "selected_v5_bundle_manifest_match": manifest_match,
        "expected_v5_bundle_sha256": expected_sha or None,
        "expected_v5_bundle_matched": bool(
            expected_sha and selected_sha and expected_sha.lower() == selected_sha.lower()
        )
        if expected_sha
        else None,
        "acceptance_set_id": acceptance_set_id or None,
    }


def _v5_bundle_sync_diagnostics_frame(
    *,
    frames: dict[str, pl.DataFrame],
    pre_export_v5: dict[str, Any],
    generated_at: datetime,
) -> pl.DataFrame:
    path = "reports/v5_bundle_sync_diagnostics.csv"
    latest_local = _latest_v5_bundle_ts(frames)
    latest_remote = _parse_v5_context_ts(pre_export_v5.get("latest_v5_bundle_seen_at_export"))
    selected_bundle_ts = _parse_v5_context_ts(
        pre_export_v5.get("selected_v5_bundle_built_at")
        or pre_export_v5.get("selected_v5_bundle_manifest_bundle_ts")
    )
    selected_ingested = _parse_v5_context_ts(pre_export_v5.get("selected_v5_bundle_ingested_at"))
    latest_ingested = _max_dt(latest_local, selected_ingested)
    stale, why_stale = _v5_bundle_stale_state(
        latest_remote=latest_remote,
        latest_ingested=latest_ingested,
    )
    stale_seconds: int | str | None = None
    if latest_remote is not None and latest_ingested is not None:
        stale_seconds = max(0, int((latest_remote - latest_ingested).total_seconds()))
    elif latest_remote is not None and latest_ingested is None:
        stale_seconds = "not_observable_ingested_bundle_missing"
    authoritative = bool(pre_export_v5.get("authoritative_snapshot"))
    failure_reason = ""
    if stale or (bool(pre_export_v5.get("sync_attempted")) and not authoritative):
        failure_reason = str(
            pre_export_v5.get("warning_reason")
            or pre_export_v5.get("selected_v5_bundle_authoritative_reason")
            or ""
        )
    selected_sha = pre_export_v5.get("selected_v5_bundle_sha256")
    row = {
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "sync_attempted": bool(pre_export_v5.get("sync_attempted")),
        "sync_success": bool(authoritative and not stale),
        "latest_local_bundle_ts": _iso_or_none(latest_local),
        "latest_remote_bundle_ts": _iso_or_none(latest_remote),
        "latest_uploaded_bundle_ts": _iso_or_none(latest_remote),
        "latest_ingested_bundle_ts": _iso_or_none(latest_ingested),
        "selected_v5_bundle_path": pre_export_v5.get("selected_v5_bundle_path"),
        "selected_v5_bundle_ts": _iso_or_none(selected_bundle_ts),
        "selected_v5_bundle_sha": selected_sha,
        "selected_v5_bundle_sha256": selected_sha,
        "expected_v5_bundle_sha256": pre_export_v5.get("expected_v5_bundle_sha256"),
        "expected_v5_bundle_matched": pre_export_v5.get("expected_v5_bundle_matched"),
        "stale_seconds": stale_seconds,
        "why_stale": why_stale,
        "failure_reason": failure_reason,
        "authoritative_snapshot": authoritative,
        "stale_v5_bundle": stale,
    }
    return pl.DataFrame([row], infer_schema_length=None).select(CSV_SCHEMAS[path])


def _v5_bundle_stale_state(
    *,
    latest_remote: datetime | None,
    latest_ingested: datetime | None,
) -> tuple[bool, str]:
    if latest_remote is None:
        return False, "latest_remote_bundle_not_observable"
    if latest_ingested is None:
        return True, "latest_ingested_bundle_missing"
    if latest_ingested + timedelta(seconds=1) < latest_remote:
        return True, "remote_newer_than_ingested"
    return False, ""


def _max_dt(*values: datetime | None) -> datetime | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _v5_derived_outputs_stale(
    frames: dict[str, pl.DataFrame],
    pre_export_v5: dict[str, Any],
) -> bool:
    latest_seen = _parse_v5_context_ts(pre_export_v5.get("latest_v5_bundle_seen_at_export"))
    latest_local_ingested = _latest_v5_derived_output_ts(frames)
    stale, _why = _v5_bundle_stale_state(
        latest_remote=latest_seen,
        latest_ingested=latest_local_ingested,
    )
    return stale


def _v5_context_requires_authoritative_failure(pre_export_v5: dict[str, Any]) -> bool:
    return any(
        (
            bool(pre_export_v5.get("latest_v5_bundle_seen_at_export")),
            bool(pre_export_v5.get("selected_v5_bundle_path")),
            bool(pre_export_v5.get("selected_v5_bundle_sha256")),
            int(pre_export_v5.get("audit_bundle_count") or 0) > 0,
            int(pre_export_v5.get("pending_uningested_bundle_count") or 0) > 0,
            int(pre_export_v5.get("unreadable_bundle_count") or 0) > 0,
            bool(pre_export_v5.get("historical_gap_detected")),
            bool(pre_export_v5.get("possible_historical_gap")),
        )
    )


def _parse_v5_context_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC)


EXPORT_V5_DERIVED_LOOKBACK_DAYS = 8
EXPORT_V5_ALPHA_FACTORY_LOOKBACK_DAYS = 30
EXPORT_V5_ALPHA_FACTORY_MAX_CANDIDATES = 200


def _refresh_v5_derived_outputs(lake_root: Path, export_day: date) -> list[str]:
    if not _flag_enabled("QUANT_LAB_EXPORT_DERIVED_REFRESH_IN_PROCESS", default=False):
        return _refresh_v5_derived_outputs_subprocess(lake_root, export_day)
    warnings: list[str] = []
    steps = [
        (
            "analyze_v5_telemetry",
            lambda: __import__(
                "quant_lab.strategy_telemetry.analyze",
                fromlist=["analyze_v5_telemetry"],
            ).analyze_v5_telemetry(
                lake_root=lake_root,
                date=export_day.isoformat(),
                refresh_candidate_gold=False,
            ),
        ),
        (
            "build_candidate_labels",
            lambda: __import__(
                "quant_lab.research.candidate_labels",
                fromlist=["build_and_publish_candidate_labels"],
            ).build_and_publish_candidate_labels(
                lake_root,
                as_of_date=export_day,
                mode="incremental",
                lookback_days=EXPORT_V5_DERIVED_LOOKBACK_DAYS,
            ),
        ),
        (
            "build_strategy_evidence",
            lambda: __import__(
                "quant_lab.research.strategy_evidence",
                fromlist=["build_and_publish_strategy_evidence"],
            ).build_and_publish_strategy_evidence(
                lake_root,
                as_of_date=export_day.isoformat(),
                mode="incremental",
                lookback_days=EXPORT_V5_DERIVED_LOOKBACK_DAYS,
                include_historical_outcomes=False,
            ),
        ),
        (
            "build_trade_level_judgment",
            lambda: __import__(
                "quant_lab.trade_level.judgment",
                fromlist=["build_and_publish_trade_level_judgment"],
            ).build_and_publish_trade_level_judgment(lake_root, as_of_date=export_day),
        ),
        (
            "build_expanded_universe_automation",
            lambda: __import__(
                "quant_lab.research.expanded_universe",
                fromlist=["build_and_publish_expanded_crypto_universe_shadow"],
            ).build_and_publish_expanded_crypto_universe_shadow(
                lake_root,
                as_of_date=export_day.isoformat(),
            ),
        ),
        (
            "build_alpha_discovery_board",
            lambda: __import__(
                "quant_lab.research.alpha_discovery",
                fromlist=["build_and_publish_alpha_discovery_board"],
            ).build_and_publish_alpha_discovery_board(
                lake_root,
                as_of_date=export_day,
                include_legacy_outcome_counts=False,
            ),
        ),
        (
            "build_paper_strategy_tracking",
            lambda: __import__(
                "quant_lab.research.paper_tracking",
                fromlist=["build_and_publish_paper_strategy_tracking"],
            ).build_and_publish_paper_strategy_tracking(lake_root, as_of_date=export_day),
        ),
        (
            "build_paper_strategy_pipeline",
            lambda: __import__(
                "quant_lab.research.paper_promotion",
                fromlist=["build_and_publish_paper_strategy_pipeline"],
            ).build_and_publish_paper_strategy_pipeline(lake_root, as_of_date=export_day),
        ),
        (
            "build_research_portfolio_status_before_diagnostics",
            lambda: __import__(
                "quant_lab.research.portfolio",
                fromlist=["build_and_publish_research_portfolio_status"],
            ).build_and_publish_research_portfolio_status(lake_root, as_of_date=export_day),
        ),
        (
            "refresh_research_diagnostics",
            lambda: __import__(
                "quant_lab.research.diagnostics_refresh",
                fromlist=["refresh_research_diagnostics"],
            ).refresh_research_diagnostics(
                lake_root,
                as_of_date=export_day,
            ),
        ),
        (
            "build_entry_quality",
            lambda: __import__(
                "quant_lab.research.entry_quality",
                fromlist=["build_and_publish_entry_quality"],
            ).build_and_publish_entry_quality(lake_root, as_of_date=export_day),
        ),
        (
            "build_alpha_factory",
            lambda: __import__(
                "quant_lab.research.alpha_factory",
                fromlist=["build_and_publish_alpha_factory"],
            ).build_and_publish_alpha_factory(
                lake_root,
                as_of_date=export_day,
                lookback_days=EXPORT_V5_ALPHA_FACTORY_LOOKBACK_DAYS,
                max_candidates=EXPORT_V5_ALPHA_FACTORY_MAX_CANDIDATES,
            ),
        ),
        (
            "refresh_alpha_discovery_board_after_alpha_factory",
            lambda: __import__(
                "quant_lab.research.alpha_discovery",
                fromlist=["build_and_publish_alpha_discovery_board"],
            ).build_and_publish_alpha_discovery_board(
                lake_root,
                as_of_date=export_day,
                include_legacy_outcome_counts=False,
            ),
        ),
        (
            "build_regime_router",
            lambda: __import__(
                "quant_lab.research.regime_router",
                fromlist=["build_and_publish_regime_router"],
            ).build_and_publish_regime_router(lake_root, as_of_date=export_day),
        ),
        (
            "build_research_portfolio_status",
            lambda: __import__(
                "quant_lab.research.portfolio",
                fromlist=["build_and_publish_research_portfolio_status"],
            ).build_and_publish_research_portfolio_status(lake_root, as_of_date=export_day),
        ),
        (
            "write_enforce_readiness_report",
            lambda: __import__(
                "quant_lab.reports.enforce_readiness",
                fromlist=["write_enforce_readiness_report"],
            ).write_enforce_readiness_report(
                lake_root=lake_root,
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


def _refresh_v5_derived_outputs_subprocess(lake_root: Path, export_day: date) -> list[str]:
    warnings: list[str] = []
    for name, body in _v5_derived_refresh_step_bodies():
        stage = f"derived_refresh:{name}"
        started_at = time.perf_counter()
        _log_export_stage_event("start", stage)
        try:
            result = subprocess.run(
                [sys.executable, "-c", _v5_derived_refresh_subprocess_code(body)],
                cwd=str(Path(__file__).resolve().parents[3]),
                env={
                    **os.environ,
                    "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS", "2"),
                    "QUANT_LAB_DERIVED_REFRESH_LAKE_ROOT": str(lake_root),
                    "QUANT_LAB_DERIVED_REFRESH_DATE": export_day.isoformat(),
                    "QUANT_LAB_DERIVED_REFRESH_LOOKBACK_DAYS": str(EXPORT_V5_DERIVED_LOOKBACK_DAYS),
                    "QUANT_LAB_DERIVED_REFRESH_ALPHA_FACTORY_LOOKBACK_DAYS": str(
                        EXPORT_V5_ALPHA_FACTORY_LOOKBACK_DAYS
                    ),
                    "QUANT_LAB_DERIVED_REFRESH_ALPHA_FACTORY_MAX_CANDIDATES": str(
                        EXPORT_V5_ALPHA_FACTORY_MAX_CANDIDATES
                    ),
                    "QUANT_LAB_DERIVED_REFRESH_MEMORY_MB": os.environ.get(
                        "QUANT_LAB_EXPORT_DERIVED_REFRESH_STEP_MEMORY_MB",
                        os.environ.get("QUANT_LAB_WEB_EXPORT_MEMORY_LIMIT_MB", "0"),
                    ),
                },
                capture_output=True,
                text=True,
                timeout=int(
                    os.environ.get(
                        "QUANT_LAB_EXPORT_DERIVED_REFRESH_STEP_TIMEOUT_SEC",
                        "900",
                    )
                ),
            )
        except subprocess.TimeoutExpired as exc:
            _log_export_stage_event(
                "end",
                stage,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
                max_rss_mb=_process_max_rss_mb(),
            )
            warnings.append(f"pre_export_v5_refresh_failed:{name}:TimeoutExpired:{exc.timeout}s")
            continue
        _log_export_stage_event(
            "end",
            stage,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
            max_rss_mb=_process_max_rss_mb(),
        )
        if result.returncode != 0:
            detail = "\n".join(
                part for part in [result.stderr.strip(), result.stdout.strip()] if part
            )
            warnings.append(
                f"pre_export_v5_refresh_failed:{name}:"
                f"exit_{result.returncode}:{_safe_warning_text(detail)}"
            )
    return warnings


def _v5_derived_refresh_subprocess_code(body: str) -> str:
    return (
        "from __future__ import annotations\n"
        "import os\n"
        "from datetime import date\n"
        "from pathlib import Path\n"
        "memory_mb = int(os.environ.get('QUANT_LAB_DERIVED_REFRESH_MEMORY_MB', '0'))\n"
        "if memory_mb < 0:\n"
        "    raise ValueError('QUANT_LAB_DERIVED_REFRESH_MEMORY_MB must be >= 0')\n"
        "if memory_mb > 0:\n"
        "    try:\n"
        "        import resource\n"
        "        limit = memory_mb * 1024 * 1024\n"
        "        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))\n"
        "    except Exception:\n"
        "        pass\n"
        "lake_root = Path(os.environ['QUANT_LAB_DERIVED_REFRESH_LAKE_ROOT'])\n"
        "export_day = date.fromisoformat(os.environ['QUANT_LAB_DERIVED_REFRESH_DATE'])\n"
        "lookback_days = int(os.environ['QUANT_LAB_DERIVED_REFRESH_LOOKBACK_DAYS'])\n"
        "alpha_factory_lookback_days = int("
        "os.environ['QUANT_LAB_DERIVED_REFRESH_ALPHA_FACTORY_LOOKBACK_DAYS']"
        ")\n"
        "alpha_factory_max_candidates = int("
        "os.environ['QUANT_LAB_DERIVED_REFRESH_ALPHA_FACTORY_MAX_CANDIDATES']"
        ")\n"
        f"{body}\n"
    )


def _v5_derived_refresh_step_bodies() -> list[tuple[str, str]]:
    return [
        (
            "analyze_v5_telemetry",
            "from quant_lab.strategy_telemetry.analyze import analyze_v5_telemetry\n"
            "analyze_v5_telemetry("
            "lake_root=lake_root, "
            "date=export_day.isoformat(), "
            "refresh_candidate_gold=False"
            ")",
        ),
        (
            "build_candidate_labels",
            "from quant_lab.research.candidate_labels import build_and_publish_candidate_labels\n"
            "build_and_publish_candidate_labels("
            "lake_root, "
            "as_of_date=export_day, "
            "mode='incremental', "
            "lookback_days=lookback_days"
            ")",
        ),
        (
            "build_strategy_evidence",
            "from quant_lab.research.strategy_evidence import build_and_publish_strategy_evidence\n"
            "build_and_publish_strategy_evidence("
            "lake_root, "
            "as_of_date=export_day.isoformat(), "
            "mode='incremental', "
            "lookback_days=lookback_days, "
            "include_historical_outcomes=False"
            ")",
        ),
        (
            "build_trade_level_judgment",
            "from quant_lab.trade_level.judgment import "
            "build_and_publish_trade_level_judgment\n"
            "build_and_publish_trade_level_judgment(lake_root, as_of_date=export_day)",
        ),
        (
            "build_expanded_universe_automation",
            "from quant_lab.research.expanded_universe import "
            "build_and_publish_expanded_crypto_universe_shadow\n"
            "build_and_publish_expanded_crypto_universe_shadow("
            "lake_root, "
            "as_of_date=export_day.isoformat()"
            ")",
        ),
        (
            "build_alpha_discovery_board",
            "from quant_lab.research.alpha_discovery import "
            "build_and_publish_alpha_discovery_board\n"
            "build_and_publish_alpha_discovery_board("
            "lake_root, "
            "as_of_date=export_day, "
            "include_legacy_outcome_counts=False"
            ")",
        ),
        (
            "build_paper_strategy_tracking",
            "from quant_lab.research.paper_tracking import "
            "build_and_publish_paper_strategy_tracking\n"
            "build_and_publish_paper_strategy_tracking(lake_root, as_of_date=export_day)",
        ),
        (
            "build_paper_strategy_pipeline",
            "from quant_lab.research.paper_promotion import "
            "build_and_publish_paper_strategy_pipeline\n"
            "build_and_publish_paper_strategy_pipeline(lake_root, as_of_date=export_day)",
        ),
        (
            "build_research_portfolio_status_before_diagnostics",
            "from quant_lab.research.portfolio import build_and_publish_research_portfolio_status\n"
            "build_and_publish_research_portfolio_status(lake_root, as_of_date=export_day)",
        ),
        (
            "refresh_research_diagnostics",
            "from quant_lab.research.diagnostics_refresh import refresh_research_diagnostics\n"
            "refresh_research_diagnostics(lake_root, as_of_date=export_day)",
        ),
        (
            "build_entry_quality",
            "from quant_lab.research.entry_quality import build_and_publish_entry_quality\n"
            "build_and_publish_entry_quality(lake_root, as_of_date=export_day)",
        ),
        (
            "build_alpha_factory",
            "from quant_lab.research.alpha_factory import build_and_publish_alpha_factory\n"
            "build_and_publish_alpha_factory("
            "lake_root, "
            "as_of_date=export_day, "
            "lookback_days=alpha_factory_lookback_days, "
            "max_candidates=alpha_factory_max_candidates"
            ")",
        ),
        (
            "refresh_alpha_discovery_board_after_alpha_factory",
            "from quant_lab.research.alpha_discovery import "
            "build_and_publish_alpha_discovery_board\n"
            "build_and_publish_alpha_discovery_board("
            "lake_root, "
            "as_of_date=export_day, "
            "include_legacy_outcome_counts=False"
            ")",
        ),
        (
            "build_regime_router",
            "from quant_lab.research.regime_router import build_and_publish_regime_router\n"
            "build_and_publish_regime_router(lake_root, as_of_date=export_day)",
        ),
        (
            "build_research_portfolio_status",
            "from quant_lab.research.portfolio import build_and_publish_research_portfolio_status\n"
            "build_and_publish_research_portfolio_status(lake_root, as_of_date=export_day)",
        ),
        (
            "write_enforce_readiness_report",
            "from quant_lab.reports.enforce_readiness import write_enforce_readiness_report\n"
            "write_enforce_readiness_report("
            "lake_root=lake_root, "
            "strategy='v5', "
            "version='5.0.0'"
            ")",
        ),
    ]


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
    pre_export_v5: dict[str, Any] | None = None,
) -> dict[str, Any]:
    git = _git_info()
    v5_context = pre_export_v5 or {}
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
        "acceptance_set_id": v5_context.get("acceptance_set_id"),
        "expected_v5_bundle_sha256": v5_context.get("expected_v5_bundle_sha256"),
        "ingested_v5_bundle_sha256": v5_context.get("ingested_v5_bundle_sha256"),
        "acceptance_set_matched": v5_context.get("acceptance_set_matched"),
        "acceptance_set_sha256_relationship": v5_context.get(
            "acceptance_set_sha256_relationship"
        ),
        "quant_lab_commit": v5_context.get("quant_lab_commit") or git["git_commit"],
        "v5_commit": v5_context.get("v5_commit"),
        "expected_v5_bundle_matched": v5_context.get("expected_v5_bundle_matched"),
        "selected_v5_bundle_sha256": v5_context.get("selected_v5_bundle_sha256"),
        "selected_v5_bundle_manifest_bundle_sha256": v5_context.get(
            "selected_v5_bundle_manifest_bundle_sha256"
        )
        or v5_context.get("selected_v5_bundle_sha256"),
        "selected_v5_bundle_manifest_bundle_name": v5_context.get(
            "selected_v5_bundle_manifest_bundle_name"
        ),
        "embedded_v5_bundle_present": bool(v5_context.get("embedded_v5_bundle_present")),
        "embedded_v5_bundle_member_path": v5_context.get("embedded_v5_bundle_member_path"),
        "embedded_v5_bundle_name": v5_context.get("embedded_v5_bundle_name"),
        "embedded_v5_bundle_original_name": v5_context.get("embedded_v5_bundle_original_name"),
        "embedded_v5_bundle_sha256": v5_context.get("embedded_v5_bundle_sha256"),
        "embedded_v5_bundle_source_sha256": v5_context.get("embedded_v5_bundle_source_sha256"),
        "embedded_v5_bundle_redacted": bool(v5_context.get("embedded_v5_bundle_redacted")),
        "embedded_v5_bundle_redacted_source": v5_context.get("embedded_v5_bundle_redacted_source"),
        "embedded_v5_bundle_size_bytes": v5_context.get(
            "embedded_v5_bundle_size_bytes",
            0,
        ),
        "embedded_v5_bundle_matches_selected": bool(
            v5_context.get("embedded_v5_bundle_matches_selected")
        ),
        "embedded_v5_bundle_validation_status": v5_context.get(
            "embedded_v5_bundle_validation_status"
        ),
        "github_ci_status": v5_context.get("github_ci_status"),
        "lake_file_index_refresh": v5_context.get("lake_file_index_refresh"),
        "lake_file_health": v5_context.get("lake_file_health"),
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

    risk = _risk_permissions_for_export(
        snapshot.frames.get("risk_permission", pl.DataFrame()),
        snapshot.frames,
        reference_at=generated_at,
        telemetry_latest_ts=latest_v5_bundle_ts,
    )
    risk_quality = _risk_permission_quality(
        risk,
        snapshot.frames,
        reference_at=generated_at,
        telemetry_latest_ts=latest_v5_bundle_ts,
    )
    read_only_cost_advisory = _read_only_cost_advisory_context(risk_quality)

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
                True,
                (
                    f"legacy_fallback_rows={fallback_rows.height}; "
                    f"legacy_ratio={fallback_stats['fallback_ratio']:.2%}; "
                    f"hard_fallback_ratio={fallback_stats['hard_fallback_ratio']:.2%}; "
                    f"soft_fallback_ratio={fallback_stats['soft_fallback_ratio']:.2%}; "
                    "diagnostic legacy metric; hard/soft fallback checks carry status"
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
        soft_fallback_passed = fallback_stats["soft_fallback_ratio"] <= 0.5
        soft_fallback_detail = (
            f"soft_fallback_count={fallback_stats['soft_fallback_count']}; "
            f"ratio={fallback_stats['soft_fallback_ratio']:.2%}; "
            f"proxy_only_count={fallback_stats['proxy_only_count']}"
        )
        if (
            not soft_fallback_passed
            and read_only_cost_advisory
            and fallback_stats["hard_fallback_count"] == 0
            and fallback_stats["global_default_count"] == 0
        ):
            soft_fallback_passed = True
            soft_fallback_detail += (
                "; read_only_advisory=true; proxy_cost_not_actual=true; "
                "live_order_effect=read_only_no_live_order"
            )
        checks.append(
            _check(
                "cost_soft_fallback_ratio",
                soft_fallback_passed,
                soft_fallback_detail,
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
                "critical" if research_enabled and alpha_discovery_board.height == 0 else "warning"
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
    candidate_required_feature_completeness = _v5_candidate_required_feature_completeness(
        latest_candidate_quality
    )
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
            candidate_required_feature_completeness >= 0.8,
            (
                "required_feature_completeness="
                f"{candidate_required_feature_completeness}; "
                f"legacy_feature_completeness="
                f"{latest_candidate_quality.get('feature_completeness', 'n/a')}"
            ),
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
            (f"cost_source_coverage={latest_candidate_quality.get('cost_source_coverage', 'n/a')}"),
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
            bool(
                decision_audit_quality["generic_decision_audit_present"]
                or decision_audit_quality["v5_decision_audit_present"]
            ),
            (
                "legacy generic silver/decision_audit is optional for V5; "
                f"rows={decision_audit_quality['generic_decision_audit_rows']}; "
                f"v5_decision_audit_count={decision_audit_quality['v5_decision_audit_count']}"
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
    registry_dataset_names, registry_skipped_heavy = _registry_quality_dataset_names(snapshot)
    try:
        registry_quality = run_data_quality(
            lake_root,
            dataset_names=registry_dataset_names,
            reference_at=generated_at,
        ).to_dict(include_checks=True)
    except Exception as exc:
        registry_quality = {
            "status": "degraded",
            "generated_at": generated_at.isoformat(),
            "dataset_count": 0,
            "check_count": 0,
            "fail_count": 0,
            "warning_count": 1,
            "checks": [],
            "error_message": _safe_warning_text(f"{type(exc).__name__}:{exc}"),
        }
    registry_quality["skipped_heavy_datasets"] = registry_skipped_heavy
    registry_quality["selected_dataset_count"] = len(registry_dataset_names)
    dataset_governance = registry_quality
    checks.append(
        _check(
            "dataset_registry_quality",
            int(registry_quality.get("fail_count") or 0) == 0
            and str(registry_quality.get("status") or "").lower() != "degraded",
            (
                f"status={registry_quality.get('status')}; "
                f"fail_count={registry_quality.get('fail_count')}; "
                f"warning_count={registry_quality.get('warning_count')}"
            ),
            warning_only=True,
        )
    )
    if registry_skipped_heavy:
        checks.append(
            _check(
                "dataset_registry_heavy_raw_rollup_fast_path",
                True,
                (
                    "skipped_heavy_raw_datasets="
                    f"{','.join(registry_skipped_heavy)}; "
                    "rollup_or_health_dataset_used_for_export_quality"
                ),
                warning_only=True,
            )
        )

    risk = _risk_permissions_for_export(
        snapshot.frames.get("risk_permission", pl.DataFrame()),
        snapshot.frames,
        reference_at=generated_at,
        telemetry_latest_ts=latest_v5_bundle_ts,
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
    risk_quality = _risk_permission_quality(
        risk,
        snapshot.frames,
        reference_at=generated_at,
        telemetry_latest_ts=latest_v5_bundle_ts,
    )
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
    v5_cost_probe_order_events = snapshot.frames.get(
        "v5_cost_probe_order_event",
        pl.DataFrame(),
    )
    v5_cost_probe_roundtrip_events = snapshot.frames.get(
        "v5_cost_probe_roundtrip_event",
        pl.DataFrame(),
    )
    private_fill_probe_context = _private_fill_cost_probe_context(
        fills,
        order_events=v5_cost_probe_order_events,
        roundtrip_events=v5_cost_probe_roundtrip_events,
        export_day=day,
    )
    proxy_only = _all_cost_rows_public_proxy(costs)
    health_checks = _jsonish_dict(latest_cost_health.get("data_quality_checks_json"))
    actual_rows = int(latest_cost_health.get("actual_rows") or 0)
    mixed_rows = int(latest_cost_health.get("mixed_rows") or 0)
    actual_or_mixed_rows = actual_rows + mixed_rows
    actual_cost_symbol_coverage_passed, actual_cost_symbol_coverage_detail = (
        _actual_cost_symbol_coverage_check(
            export_day=day,
            costs=costs,
            latest_cost_health=latest_cost_health,
            health_checks=health_checks,
            actual_or_mixed_rows=actual_or_mixed_rows,
            fills=fills,
            v5_trades=v5_trades,
            cost_probe_private_fill_count=private_fill_probe_context[
                "cost_probe_private_fill_count"
            ],
        )
    )
    relevant_private_fills = _relevant_private_fill_rows(fills, export_day=day)
    relevant_v5_trades = _relevant_frame_rows(
        v5_trades,
        export_day=day,
        timestamp_columns=("ts_utc", "ts", "timestamp", "time", "created_at"),
    )
    private_fills_cost_passed, private_fills_cost_detail = _raw_fill_like_cost_check(
        health_checks=health_checks,
        health_check_name="private_fills_present_but_actual_cost_zero",
        raw_row_label="private_fills",
        raw_row_count=fills.height,
        relevant_row_count=relevant_private_fills,
        excluded_relevant_row_count=private_fill_probe_context["cost_probe_private_fill_count"],
        exclusion_detail=_private_fill_cost_probe_context_detail(private_fill_probe_context),
        actual_rows=actual_rows,
        mixed_rows=mixed_rows,
    )
    v5_trades_cost_passed, v5_trades_cost_detail = _raw_fill_like_cost_check(
        health_checks=health_checks,
        health_check_name="trades_present_but_not_in_cost_model",
        raw_row_label="v5_trades",
        raw_row_count=v5_trades.height,
        relevant_row_count=relevant_v5_trades,
        actual_rows=actual_rows,
        mixed_rows=mixed_rows,
    )
    checks.append(
        _check(
            "okx_private_actual_cost_available",
            not (proxy_only and actual_or_mixed_rows == 0),
            "actual cost unavailable; cost model is public_spread_proxy only",
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "private_fills_present_but_actual_cost_zero",
            private_fills_cost_passed,
            private_fills_cost_detail,
            warning_only=True,
        )
    )
    if private_fill_probe_context["cost_probe_private_fill_count"] > 0:
        checks.append(
            _check(
                "cost_probe_private_fills_excluded_from_live_coverage",
                True,
                _private_fill_cost_probe_context_detail(private_fill_probe_context),
                status="INFO",
                severity="info",
            )
        )
    checks.append(
        _check(
            "trades_present_but_not_in_cost_model",
            v5_trades_cost_passed,
            v5_trades_cost_detail,
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
            actual_cost_symbol_coverage_passed,
            actual_cost_symbol_coverage_detail,
            warning_only=True,
        )
    )
    live_cost_passed, live_cost_detail = _live_universe_stale_actual_or_mixed_check(costs)
    if read_only_cost_advisory and not live_cost_passed:
        live_cost_passed = True
        live_cost_detail += (
            "; read_only_advisory=true; proxy_cost_not_actual=true; "
            "not_used_for_live_strategy_promotion"
        )
    checks.append(
        _check(
            "live_universe_stale_actual_or_mixed_cost",
            live_cost_passed,
            live_cost_detail,
            warning_only=True,
        )
    )
    checks.append(
        _check(
            "okx_ws_universe_complete",
            *_okx_ws_universe_completeness(snapshot.frames),
            warning_only=True,
        )
    )
    enforce_readiness = build_enforce_readiness_report(lake_root)
    read_only_cost_coverage_block = _read_only_cost_coverage_readiness_block(
        enforce_readiness,
        risk_quality,
    )
    readiness_veto_ready = (
        str(getattr(enforce_readiness, "veto_status", "") or "").upper() == "VETO_READY"
    )
    readiness_contract_ok = enforce_readiness.readiness_status in {"READY", "ADVISORY_READY"} or (
        read_only_cost_coverage_block
    )
    readiness_ok = readiness_contract_ok or readiness_veto_ready
    readiness_check_status = (
        "PASS"
        if readiness_contract_ok
        else "WARN"
        if readiness_veto_ready
        else "WARN"
        if read_only_cost_coverage_block
        else "FAIL"
        if enforce_readiness.readiness_status == "BLOCKED"
        else "WARN"
    )
    readiness_detail = (
        f"readiness_status={enforce_readiness.readiness_status}; "
        f"veto_status={getattr(enforce_readiness, 'veto_status', 'UNKNOWN')}; "
        f"entry_status={getattr(enforce_readiness, 'entry_status', 'UNKNOWN')}; "
        f"scale_status={getattr(enforce_readiness, 'scale_status', 'UNKNOWN')}; "
        f"blocked={enforce_readiness.blocked_reasons}; "
        f"warnings={enforce_readiness.warning_reasons}"
    )
    if read_only_cost_coverage_block:
        readiness_detail += "; live_order_effect=read_only_no_live_order"
    elif readiness_veto_ready and enforce_readiness.readiness_status == "BLOCKED":
        readiness_detail += "; live_order_effect=entry_or_scale_block_only"
    checks.append(
        _check(
            "quant_lab_enforce_readiness",
            readiness_ok,
            readiness_detail,
            severity="critical" if readiness_check_status == "FAIL" else "warning",
            status=readiness_check_status,
        )
    )

    paper_runtime_freshness = snapshot.frames.get(
        "paper_runtime_freshness",
        pl.DataFrame(),
    )
    for row in (
        paper_runtime_freshness.to_dicts()
        if not paper_runtime_freshness.is_empty()
        else []
    ):
        check_name = str(row.get("check_name") or "paper_runtime_freshness")
        runtime_status = str(row.get("status") or "FAIL").upper()
        checks.append(
            _check(
                check_name,
                runtime_status in {"PASS", "NO_NEW_EVENT_EXPECTED"},
                str(row.get("detail") or "paper runtime evidence missing"),
                status=(
                    "INFO"
                    if runtime_status == "NO_NEW_EVENT_EXPECTED"
                    else "WARN"
                    if runtime_status == "WARNING"
                    else runtime_status
                ),
                severity=(
                    "warning"
                    if runtime_status in {"WARNING", "NO_NEW_EVENT_EXPECTED"}
                    else "critical"
                ),
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
        if check["status"] not in {"PASS", "INFO"}:
            warnings.append(f"{check['name']}: {check['detail']}")

    status = _overall_quality_status(checks)
    failures = _data_quality_failure_lines(checks)

    return {
        "status": status,
        "export_date": day.isoformat(),
        "generated_at": generated_at.isoformat(),
        "display_timezone": DISPLAY_TIMEZONE,
        "generated_at_beijing": beijing_iso(generated_at),
        "checks": checks,
        "failures": failures,
        "warnings": sorted(set(warnings)),
        "risk_permission": risk_quality,
        "decision_audit": decision_audit_quality,
        "v5_pre_export": v5_refresh,
        "dataset_governance": dataset_governance,
        "registry_quality": registry_quality,
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


def _read_only_cost_coverage_readiness_block(
    enforce_readiness: Any,
    risk_quality: dict[str, Any],
) -> bool:
    if getattr(enforce_readiness, "readiness_status", "") != "BLOCKED":
        return False
    blocked = {
        str(reason).strip()
        for reason in getattr(enforce_readiness, "blocked_reasons", [])
        if str(reason).strip()
    }
    if blocked != {"actual_or_mixed_cost_coverage_live_universe"}:
        return False
    permission = str(risk_quality.get("permission") or "").strip().upper()
    status = str(risk_quality.get("permission_status") or "").strip().upper()
    if permission in {"ABORT", "SELL_ONLY"}:
        return True
    if status in {"NO_FRESH_PERMISSION"} or status.startswith(("STALE_", "EXPIRED_")):
        return True
    return False


def _read_only_cost_advisory_context(risk_quality: dict[str, Any]) -> bool:
    permission = str(risk_quality.get("permission") or "").strip().upper()
    status = str(risk_quality.get("permission_status") or "").strip().upper()
    return permission in {"ABORT", "SELL_ONLY"} and status in {
        "ACTIVE_ABORT",
        "ACTIVE_SELL_ONLY",
    }


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
    research_portfolio = dedupe_research_portfolio_status(
        snapshot.frames.get("research_portfolio_status", pl.DataFrame())
    )
    paper_runs = snapshot.frames.get("paper_strategy_runs", pl.DataFrame())
    paper_daily = snapshot.frames.get("paper_strategy_daily", pl.DataFrame())
    candidate_quality = _latest_candidate_quality(
        snapshot.frames.get("v5_candidate_quality_daily", pl.DataFrame())
    )
    risk = snapshot.frames.get("risk_permission", pl.DataFrame())
    gate_counts = _value_counts(gates, "status")
    alpha_discovery_counts = _value_counts(alpha_discovery_board, "decision")
    strategy_decision_counts = _value_counts(strategy_evidence, "decision")
    research_portfolio_counts = _value_counts(research_portfolio, "status")
    strategy_dashboard_counts = _strategy_dashboard_decision_counts(
        snapshot.frames.get("strategy_opportunity_advisory", pl.DataFrame()),
        alpha_discovery_board,
        strategy_evidence,
    )
    baseline = _baseline_gate_summary(gates, snapshot.frames.get("alpha_evidence", pl.DataFrame()))
    paper_tracking_counts = _value_counts(paper_runs, "tracking_stage")
    paper_status_counts = _value_counts(paper_daily, "paper_tracking_status")
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
        "Core momentum baseline: "
        f"baseline_status={baseline.get('baseline_status', 'UNKNOWN')}, "
        f"role={baseline.get('role', RESEARCH_BASELINE_ROLE)}, "
        f"not_live_eligible={baseline.get('not_live_eligible', True)}, "
        f"not_global_strategy_gate={baseline.get('not_global_strategy_gate', True)}",
        f"Strategy-level dashboard decision counts: {safe_json_dumps(strategy_dashboard_counts)}",
        f"Research portfolio status counts: {safe_json_dumps(research_portfolio_counts)}",
        "Research portfolio priority: open `reports/research_portfolio_summary.md` "
        "before reading the full status CSV.",
        "Focus strategy candidates: "
        "SOL_PROTECT_MOMENTUM_CONTINUATION_PAPER_V1, "
        "ETH_F3_DOMINANT_ENTRY_PAPER_V1, "
        "ALT_IMPULSE_REGIME_SHADOW_V1, "
        "EXPANDED_CRYPTO_UNIVERSE_SHADOW",
        f"Research gate/baseline status counts: {safe_json_dumps(gate_counts)}",
        f"Alpha discovery decision counts: {safe_json_dumps(alpha_discovery_counts)}",
        f"Strategy evidence decision counts: {safe_json_dumps(strategy_decision_counts)}",
        f"Paper tracking stages: {safe_json_dumps(paper_tracking_counts)}",
        f"Paper tracking status counts: {safe_json_dumps(paper_status_counts)}",
        f"V5 candidate_event rows: {snapshot.row_counts.get('v5_candidate_event', 0)}",
        f"V5 candidate label completeness: {candidate_quality.get('label_completeness', 'n/a')}",
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
    elif cost_coverage_question := _live_cost_coverage_question(costs, data_quality):
        questions.append(cost_coverage_question)
    elif _cost_model_is_proxy_or_fallback(costs):
        questions.append("成本模型仍是 public spread proxy，何时启用真实 fills/bills？")
    if bootstrap_question := _cost_bootstrap_readiness_question(
        snapshot.frames.get("cost_bootstrap_readiness", pl.DataFrame())
    ):
        questions.append(bootstrap_question)
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
    if (
        snapshot.row_counts.get("strategy_evidence", 0) == 0
        and snapshot.row_counts.get("alpha_discovery_board", 0) == 0
    ):
        questions.append("Why is strategy_evidence empty for V5 candidate discovery?")
    if snapshot.row_counts.get("v5_candidate_event", 0) == 0:
        questions.append("Why is reports/candidate_snapshot.csv missing from V5 bundles?")
    if snapshot.row_counts.get("v5_candidate_label", 0) == 0:
        questions.append("Why are v5_candidate_label forward labels empty?")
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


def _cost_bootstrap_readiness_question(readiness: pl.DataFrame) -> str:
    if readiness.is_empty() or "actual_or_mixed_trusted_covered" not in readiness.columns:
        return ""
    rows = readiness.to_dicts()
    trusted = [
        row
        for row in rows
        if str(row.get("actual_or_mixed_trusted_covered") or "").lower() == "true"
        or row.get("actual_or_mixed_trusted_covered") is True
    ]
    if trusted:
        return ""
    states = sorted(
        {
            str(row.get("bootstrap_state") or "UNKNOWN")
            for row in rows
            if str(row.get("bootstrap_state") or "").strip()
        }
    )
    bill_matched = sum(
        1
        for row in rows
        if str(row.get("bill_match_status") or "").strip().upper() == "PASS"
        or (_safe_int(row.get("bill_matched_count")) or 0) > 0
        or (_safe_int(row.get("matched_bill_count")) or 0) > 0
    )
    bootstrap_ready = sum(
        1
        for row in rows
        if str(row.get("actual_or_mixed_bootstrap_covered") or "").lower() == "true"
        or row.get("actual_or_mixed_bootstrap_covered") is True
        or str(row.get("bootstrap_state") or "").strip().upper()
        in {"BOOTSTRAP_PROBE_AVAILABLE", "ACTUAL_FILLS_SMALL_SAMPLE", "ACTUAL_FILLS_TRUSTED"}
    )
    next_actions = sorted(
        {
            str(row.get("next_action") or "").strip()
            for row in rows
            if str(row.get("next_action") or "").strip()
        }
    )
    bill_detail = (
        f"bill_match 已通过 {bill_matched}/{len(rows)}"
        if bill_matched >= len(rows)
        else f"bill_match 仅通过 {bill_matched}/{len(rows)}，仍需补齐账单匹配"
    )
    bootstrap_detail = f"bootstrap coverage {bootstrap_ready}/{len(rows)}"
    action_detail = ""
    if next_actions:
        action_detail = "；next_action=" + " | ".join(next_actions[:4])
    return (
        "cost_bootstrap_readiness 仍无 trusted coverage；"
        f"{bill_detail}；{bootstrap_detail}；"
        "不要重复已完成的成本探针，下一步按 next_action 增加真实/混合 live 样本与策略证据。"
        "states=" + ",".join(states[:6]) + action_detail
    )


def _live_cost_coverage_question(costs: pl.DataFrame, data_quality: dict[str, Any]) -> str | None:
    if costs.is_empty():
        return None

    evaluation = evaluate_live_universe_cost_coverage(costs)
    coverage_rate = _float_or_none(evaluation.get("coverage_rate"))
    target_coverage = _float_or_none(evaluation.get("target_coverage"))
    coverage_status = str(evaluation.get("coverage_status") or "UNKNOWN").upper()
    stale_symbols = _question_symbol_list(evaluation.get("stale_actual_or_mixed_symbols", []))
    uncovered_symbols = _question_symbol_list(
        row.get("symbol")
        for row in evaluation.get("rows", [])
        if str(row.get("symbol") or "").strip() and not bool(row.get("actual_or_mixed_covered"))
    )
    proxy_only_symbols = _question_symbol_list(evaluation.get("proxy_only_symbols", []))
    direct_symbols = _question_symbol_list(evaluation.get("direct_symbols", []))
    readiness = data_quality.get("quant_lab_enforce_readiness", {})
    blocked_reasons = set(_json_listish(readiness.get("blocked_reasons")))
    coverage_blocked = "actual_or_mixed_cost_coverage_live_universe" in blocked_reasons
    should_ask = (
        coverage_status != "PASS"
        or bool(stale_symbols)
        or bool(uncovered_symbols)
        or bool(proxy_only_symbols)
        or coverage_blocked
    )
    if not should_ask:
        return None

    coverage_text = _rate_text(coverage_rate)
    target_text = _rate_text(target_coverage)
    return (
        "live universe actual/mixed 成本覆盖仅 "
        f"{coverage_text}（目标 {target_text}）；"
        f"未覆盖={_question_symbols_text(uncovered_symbols)}；"
        f"stale_actual_or_mixed={_question_symbols_text(stale_symbols)}；"
        f"direct_actual_or_mixed={_question_symbols_text(direct_symbols)}。"
        "是否刷新 OKX read-only fills/bills 或等待新成交后再评估？"
        "不要把 public_spread_proxy 当 actual/mixed。"
    )


def _question_symbol_list(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _question_symbols_text(symbols: list[str], *, limit: int = 6) -> str:
    if not symbols:
        return "none"
    shown = symbols[:limit]
    suffix = f", +{len(symbols) - limit} more" if len(symbols) > limit else ""
    return ", ".join(shown) + suffix


def _rate_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


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


def _actual_cost_symbol_coverage_check(
    *,
    export_day: date,
    costs: pl.DataFrame,
    latest_cost_health: dict[str, Any],
    health_checks: dict[str, Any],
    actual_or_mixed_rows: int,
    fills: pl.DataFrame,
    v5_trades: pl.DataFrame,
    cost_probe_private_fill_count: int = 0,
) -> tuple[bool, str]:
    latest_actual_or_mixed_symbols = set(
        _json_listish(latest_cost_health.get("symbols_with_actual_cost"))
    ) | set(_json_listish(latest_cost_health.get("symbols_with_mixed_cost")))
    latest_expected_symbols = set(latest_actual_or_mixed_symbols)
    for column in [
        "symbols_with_proxy_only",
        "symbols_proxy_only",
        "symbols_missing_cost",
    ]:
        latest_expected_symbols.update(_json_listish(latest_cost_health.get(column)))

    historical_actual_or_mixed_symbols: set[str] = set()
    historical_expected_symbols: set[str] = set()
    historical_actual_or_mixed_rows = 0
    if not costs.is_empty() and "symbol" in costs.columns:
        source_columns = [column for column in ["source", "cost_source"] if column in costs.columns]
        columns = ["symbol", *source_columns]
        for row in costs.select(columns).to_dicts():
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol or symbol == "GLOBAL":
                continue
            historical_expected_symbols.add(symbol)
            sources = {str(row.get(column) or "").strip().lower() for column in source_columns}
            if sources.intersection(
                {
                    "actual_okx_fills_and_bills",
                    "actual_fills",
                    "mixed_actual_proxy",
                }
            ):
                historical_actual_or_mixed_symbols.add(symbol)
                historical_actual_or_mixed_rows += 1

    health_detail = str(health_checks.get("actual_cost_symbol_coverage") or "").strip()
    if health_detail and health_detail.lower() != "n/a":
        latest_coverage = health_detail
        latest_source = "cost_health_daily"
    else:
        latest_denominator = max(
            len(latest_expected_symbols),
            len(latest_actual_or_mixed_symbols),
        )
        latest_coverage = (
            f"{len(latest_actual_or_mixed_symbols)}/{latest_denominator}"
            if latest_denominator > 0
            else "n/a"
        )
        latest_source = "computed_from_latest_cost_health"

    historical_denominator = max(
        len(historical_expected_symbols),
        len(historical_actual_or_mixed_symbols),
    )
    historical_coverage = (
        f"{len(historical_actual_or_mixed_symbols)}/{historical_denominator}"
        if historical_denominator > 0
        else "n/a"
    )

    relevant_private_fills = _relevant_private_fill_rows(fills, export_day=export_day)
    relevant_v5_trades = _relevant_frame_rows(
        v5_trades,
        export_day=export_day,
        timestamp_columns=("ts_utc", "ts", "timestamp", "time", "created_at"),
    )
    non_probe_private_fills = max(relevant_private_fills - cost_probe_private_fill_count, 0)
    fill_like_rows = non_probe_private_fills + relevant_v5_trades
    passed = actual_or_mixed_rows > 0 or fill_like_rows == 0
    detail = (
        f"latest_actual_or_mixed_symbols={latest_coverage}; "
        f"latest_actual_or_mixed_rows={actual_or_mixed_rows}; "
        f"historical_actual_or_mixed_symbols={historical_coverage}; "
        f"historical_actual_or_mixed_rows={historical_actual_or_mixed_rows}; "
        f"relevant_private_fills={relevant_private_fills}; "
        f"cost_probe_private_fill_count={cost_probe_private_fill_count}; "
        f"non_probe_private_fills={non_probe_private_fills}; "
        f"private_fills_total={fills.height}; "
        f"relevant_v5_trades={relevant_v5_trades}; "
        f"v5_trades_total={v5_trades.height}; "
        f"source={latest_source}+cost_bucket_daily_snapshot"
    )
    return passed, detail


def _private_fill_cost_probe_context(
    fills: pl.DataFrame,
    *,
    order_events: pl.DataFrame,
    roundtrip_events: pl.DataFrame,
    export_day: date,
) -> dict[str, int]:
    order_ids, trade_ids = cost_probe_private_fill_keys(order_events, roundtrip_events)
    relevant_private_fills = 0
    cost_probe_private_fill_count = 0
    if not fills.is_empty():
        for row in fills.to_dicts():
            timestamp = _first_timestamp_value(row, ("ts", "fillTime"))
            if timestamp is None and row.get("raw_json") is not None:
                timestamp = _raw_json_timestamp(row.get("raw_json"), ("ts", "fillTime"))
            if timestamp is None or not _timestamp_in_export_window(
                timestamp,
                export_day=export_day,
            ):
                continue
            relevant_private_fills += 1
            if private_fill_matches_cost_probe(
                row,
                order_ids=order_ids,
                trade_ids=trade_ids,
            ):
                cost_probe_private_fill_count += 1
    unclassified_private_fill_count = max(
        relevant_private_fills - cost_probe_private_fill_count,
        0,
    )
    return {
        "relevant_private_fills": relevant_private_fills,
        "strategy_private_fill_count": 0,
        "cost_probe_private_fill_count": cost_probe_private_fill_count,
        "unclassified_private_fill_count": unclassified_private_fill_count,
    }


def _private_fill_cost_probe_context_detail(context: Mapping[str, Any]) -> str:
    return (
        f"strategy_private_fill_count={context.get('strategy_private_fill_count', 0)}; "
        f"cost_probe_private_fill_count={context.get('cost_probe_private_fill_count', 0)}; "
        f"unclassified_private_fill_count={context.get('unclassified_private_fill_count', 0)}; "
        "cost_probe_private_fills_excluded_from_live_coverage=true"
    )


def _raw_fill_like_cost_check(
    *,
    health_checks: dict[str, Any],
    health_check_name: str,
    raw_row_label: str,
    raw_row_count: int,
    relevant_row_count: int,
    excluded_relevant_row_count: int = 0,
    exclusion_detail: str = "",
    actual_rows: int,
    mixed_rows: int,
) -> tuple[bool, str]:
    latest_health_passed = health_checks.get(health_check_name, True) is not False
    actual_or_mixed_rows = actual_rows + mixed_rows
    effective_relevant_row_count = max(relevant_row_count - excluded_relevant_row_count, 0)
    export_raw_without_latest_actual_or_mixed = (
        effective_relevant_row_count > 0 and actual_or_mixed_rows == 0
    )
    classification_overrides_latest_health = (
        excluded_relevant_row_count > 0 and effective_relevant_row_count == 0
    )
    passed = (
        latest_health_passed or classification_overrides_latest_health
    ) and not export_raw_without_latest_actual_or_mixed
    detail = (
        f"{raw_row_label}={raw_row_count}; "
        f"relevant_{raw_row_label}={relevant_row_count}; "
        f"excluded_relevant_{raw_row_label}={excluded_relevant_row_count}; "
        f"effective_relevant_{raw_row_label}={effective_relevant_row_count}; "
        f"actual_rows={actual_rows}; "
        f"mixed_rows={mixed_rows}; "
        f"latest_health_check_passed={str(latest_health_passed).lower()}; "
        "classification_overrides_latest_health="
        f"{str(classification_overrides_latest_health).lower()}; "
        "export_raw_rows_without_latest_actual_or_mixed="
        f"{str(export_raw_without_latest_actual_or_mixed).lower()}"
    )
    if exclusion_detail:
        detail += f"; {exclusion_detail}"
    return passed, detail


def _relevant_private_fill_rows(fills: pl.DataFrame, *, export_day: date) -> int:
    if fills.is_empty():
        return 0
    count = 0
    for row in fills.to_dicts():
        timestamp = _first_timestamp_value(row, ("ts", "fillTime"))
        if timestamp is None and row.get("raw_json") is not None:
            timestamp = _raw_json_timestamp(row.get("raw_json"), ("ts", "fillTime"))
        if timestamp is not None and _timestamp_in_export_window(timestamp, export_day=export_day):
            count += 1
    return count


def _relevant_frame_rows(
    frame: pl.DataFrame,
    *,
    export_day: date,
    timestamp_columns: tuple[str, ...],
) -> int:
    if frame.is_empty():
        return 0
    count = 0
    for row in frame.to_dicts():
        timestamp = _first_timestamp_value(row, timestamp_columns)
        if timestamp is not None and _timestamp_in_export_window(timestamp, export_day=export_day):
            count += 1
    return count


def _raw_json_timestamp(raw_json: Any, fields: tuple[str, ...]) -> datetime | None:
    try:
        loaded = json.loads(str(raw_json))
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(loaded, dict):
        timestamp = _first_timestamp_value(loaded, fields)
        if timestamp is not None:
            return timestamp
        data = loaded.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    timestamp = _first_timestamp_value(item, fields)
                    if timestamp is not None:
                        return timestamp
    return None


def _first_timestamp_value(row: dict[str, Any], fields: tuple[str, ...]) -> datetime | None:
    for field in fields:
        value = row.get(field)
        timestamp = _parse_cost_event_timestamp(value)
        if timestamp is not None:
            return timestamp
    return None


def _parse_cost_event_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        try:
            numeric = int(text)
        except ValueError:
            return None
        if numeric > 10_000_000_000:
            return datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
        return datetime.fromtimestamp(numeric, tz=UTC)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _timestamp_in_export_window(timestamp: datetime, *, export_day: date) -> bool:
    end = datetime.combine(export_day + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    start = end - timedelta(days=PRIVATE_COST_LOOKBACK_DAYS)
    return start <= timestamp.astimezone(UTC) < end


def _live_universe_stale_actual_or_mixed_check(costs: pl.DataFrame) -> tuple[bool, str]:
    evaluation = evaluate_live_universe_cost_coverage(costs)
    stale_symbols = [
        str(symbol)
        for symbol in evaluation.get("stale_actual_or_mixed_symbols", [])
        if str(symbol).strip()
    ]
    uncovered_symbols = [
        str(row.get("symbol"))
        for row in evaluation.get("rows", [])
        if str(row.get("symbol") or "").strip() and not bool(row.get("actual_or_mixed_covered"))
    ]
    proxy_only_symbols = [
        str(symbol) for symbol in evaluation.get("proxy_only_symbols", []) if str(symbol).strip()
    ]
    coverage_rate = _float_or_none(evaluation.get("coverage_rate"))
    coverage_text = "n/a" if coverage_rate is None else f"{coverage_rate:.2%}"
    detail = (
        f"stale_actual_or_mixed_symbols={sorted(stale_symbols)}; "
        f"uncovered_actual_or_mixed_symbols={sorted(uncovered_symbols)}; "
        f"proxy_only_symbols={sorted(proxy_only_symbols)}; "
        f"coverage_status={evaluation.get('coverage_status', 'UNKNOWN')}; "
        f"coverage_rate={coverage_text}; "
        "live_order_effect=read_only_no_live_order"
    )
    return not stale_symbols, detail


def _alpha_evidence_for_export(evidence: pl.DataFrame) -> pl.DataFrame:
    path = "research/alpha_evidence.csv"
    if evidence.is_empty() and not evidence.columns:
        return _empty_csv_schema_frame(path)
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
    if "role" not in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("alpha_id")
            .cast(pl.Utf8)
            .map_elements(alpha_role, return_dtype=pl.Utf8)
            .alias("role")
        )
    else:
        normalized = normalized.with_columns(
            pl.when(pl.col("role").is_null() | (pl.col("role") == ""))
            .then(pl.col("alpha_id").cast(pl.Utf8).map_elements(alpha_role, return_dtype=pl.Utf8))
            .otherwise(pl.col("role").cast(pl.Utf8))
            .alias("role")
        )
    normalized = normalized.with_columns(
        [
            pl.when(pl.col("role") == RESEARCH_BASELINE_ROLE)
            .then(pl.col("evidence_status").cast(pl.Utf8))
            .otherwise(pl.lit(""))
            .alias("baseline_status"),
            (pl.col("role") == RESEARCH_BASELINE_ROLE).alias("not_live_eligible"),
            (pl.col("role") == RESEARCH_BASELINE_ROLE).alias("not_global_strategy_gate"),
        ]
    )
    for column in CSV_SCHEMAS[path]:
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    return normalized.select(CSV_SCHEMAS[path])


def _gate_decisions_for_export(gates: pl.DataFrame) -> pl.DataFrame:
    path = "research/gate_decisions.csv"
    if gates.is_empty() and not gates.columns:
        return _empty_csv_schema_frame(path)
    normalized = gates
    if "role" not in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("alpha_id")
            .cast(pl.Utf8)
            .map_elements(alpha_role, return_dtype=pl.Utf8)
            .alias("role")
        )
    else:
        normalized = normalized.with_columns(
            pl.when(pl.col("role").is_null() | (pl.col("role") == ""))
            .then(pl.col("alpha_id").cast(pl.Utf8).map_elements(alpha_role, return_dtype=pl.Utf8))
            .otherwise(pl.col("role").cast(pl.Utf8))
            .alias("role")
        )
    normalized = normalized.with_columns(
        [
            pl.when(pl.col("role") == RESEARCH_BASELINE_ROLE)
            .then(pl.col("status").cast(pl.Utf8))
            .otherwise(pl.lit(""))
            .alias("baseline_status"),
            (pl.col("role") == RESEARCH_BASELINE_ROLE).alias("not_live_eligible"),
            (pl.col("role") == RESEARCH_BASELINE_ROLE).alias("not_global_strategy_gate"),
        ]
    )
    for column in CSV_SCHEMAS[path]:
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    return normalized.select(CSV_SCHEMAS[path])


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
            "regime_state",
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


def _alt_impulse_shadow_base_for_export(evidence: pl.DataFrame) -> pl.DataFrame:
    if evidence.is_empty() and not evidence.columns:
        return pl.DataFrame()
    normalized = normalize_strategy_evidence_decisions(evidence)
    if "strategy_candidate" not in normalized.columns:
        return pl.DataFrame()
    selected = normalized.filter(pl.col("strategy_candidate") == "v5.alt_impulse_shadow")
    if selected.is_empty():
        return pl.DataFrame()
    return selected


def _alt_impulse_shadow_by_symbol_regime_horizon_for_export(
    evidence: pl.DataFrame,
) -> pl.DataFrame:
    path = "research/alt_impulse_shadow_by_symbol_regime_horizon.csv"
    selected = _alt_impulse_shadow_base_for_export(evidence)
    if selected.is_empty():
        return _empty_csv_schema_frame(path)
    for column in CSV_SCHEMAS[path]:
        if column not in selected.columns:
            selected = selected.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    keys = [
        key
        for key in [
            "as_of_date",
            "strategy_candidate",
            "symbol",
            "regime_state",
            "horizon_hours",
        ]
        if key in selected.columns
    ]
    if keys:
        sort_keys = [*keys, "created_at"] if "created_at" in selected.columns else keys
        selected = selected.sort(sort_keys)
        selected = selected.group_by(keys, maintain_order=True).tail(1)
    sort_columns = [
        column
        for column in ["regime_state", "symbol", "horizon_hours", "as_of_date"]
        if column in selected.columns
    ]
    selected = selected.sort(sort_columns) if sort_columns else selected
    return selected.select(CSV_SCHEMAS[path])


def _alt_impulse_shadow_by_regime_for_export(evidence: pl.DataFrame) -> pl.DataFrame:
    path = "research/alt_impulse_shadow_by_regime.csv"
    selected = _alt_impulse_shadow_base_for_export(evidence)
    if selected.is_empty():
        return _empty_csv_schema_frame(path)
    rows = selected.to_dicts()
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("as_of_date") or ""),
            str(row.get("regime_state") or "UNKNOWN"),
            int(_export_float(row.get("horizon_hours")) or 0),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (as_of_date, regime_state, horizon_hours), group_rows in sorted(grouped.items()):
        sample_count = sum(int(_export_float(row.get("sample_count")) or 0) for row in group_rows)
        complete_count = sum(
            int(_export_float(row.get("complete_sample_count")) or 0) for row in group_rows
        )
        avg_net = _weighted_export_mean(group_rows, "avg_net_bps")
        median_net = _weighted_export_mean(group_rows, "median_net_bps")
        p25_net = _weighted_export_mean(group_rows, "p25_net_bps")
        win_rate = _weighted_export_mean(group_rows, "win_rate")
        decision, reasons = strategy_evidence_decision_ladder(
            sample_count=sample_count,
            complete_sample_count=complete_count,
            avg_net_bps=avg_net,
            p25_net_bps=p25_net,
            win_rate=win_rate,
            cost_source_mix=_merge_cost_source_mix(group_rows),
            candidate_name="v5.alt_impulse_shadow",
        )
        output.append(
            {
                "as_of_date": as_of_date,
                "strategy_candidate": "v5.alt_impulse_shadow",
                "candidate_name": "v5.alt_impulse_shadow",
                "regime_state": regime_state,
                "horizon_hours": horizon_hours,
                "sample_count": sample_count,
                "complete_sample_count": complete_count,
                "avg_net_bps": avg_net,
                "median_net_bps": median_net,
                "p25_net_bps": p25_net,
                "win_rate": win_rate,
                "cost_source_mix": safe_json_dumps(_merge_cost_source_mix(group_rows)),
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
                "start_ts": _min_export_value(group_rows, "start_ts"),
                "end_ts": _max_export_value(group_rows, "end_ts"),
                "created_at": _max_export_value(group_rows, "created_at"),
                "source": "export.alt_impulse_shadow_by_regime.v0.1",
            }
        )
    return pl.DataFrame(output).select(CSV_SCHEMAS[path])


def _weighted_export_mean(rows: list[dict[str, Any]], column: str) -> float | None:
    weighted: list[tuple[float, float]] = []
    for row in rows:
        value = _export_float(row.get(column))
        if value is None:
            continue
        weight = _export_float(row.get("complete_sample_count"))
        if weight is None or weight <= 0:
            weight = _export_float(row.get("sample_count")) or 1.0
        weighted.append((value, weight))
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in weighted) / total_weight


def _export_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _merge_cost_source_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in rows:
        for source, count in _cost_source_mix_items(row.get("cost_source_mix")).items():
            merged[source] = merged.get(source, 0) + count
    return merged


def _cost_source_mix_items(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            return _cost_source_mix_items(json.loads(value))
        except json.JSONDecodeError:
            cleaned = value.strip()
            return {cleaned: 1} if cleaned else {}
    if isinstance(value, dict):
        return {
            str(source or "MISSING"): int(_export_float(count) or 0)
            for source, count in value.items()
        }
    if isinstance(value, list):
        output: dict[str, int] = {}
        for item in value:
            if isinstance(item, dict):
                source = str(item.get("cost_source") or item.get("source") or "MISSING")
                count = int(_export_float(item.get("count") or item.get("sample_count")) or 1)
            else:
                source = str(item or "MISSING")
                count = 1
            output[source] = output.get(source, 0) + count
        return output
    return {}


def _min_export_value(rows: list[dict[str, Any]], column: str) -> Any:
    values = [row.get(column) for row in rows if row.get(column) is not None]
    return min(values) if values else None


def _max_export_value(rows: list[dict[str, Any]], column: str) -> Any:
    values = [row.get(column) for row in rows if row.get(column) is not None]
    return max(values) if values else None


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
        "HISTORICAL_LABEL_THRESHOLD_READY": "reports/historical_label_threshold_ready.csv",
    }
    path = paths.get(decision, "reports/candidate_shadow_watchlist.csv")
    if board.is_empty():
        return _empty_csv_schema_frame(path)
    if "decision" not in board.columns:
        return _empty_csv_schema_frame(path)
    source_decision = "PAPER_READY" if decision == "HISTORICAL_LABEL_THRESHOLD_READY" else decision
    if decision == "KEEP_SHADOW":
        selected = board.filter(pl.col("decision").is_in(["KEEP_SHADOW", "REGIME_SHADOW"]))
    else:
        selected = board.filter(pl.col("decision") == source_decision)
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


def _formal_candidate_paper_ready_rows(
    board: pl.DataFrame,
    *,
    paper_strategy_promotion_gate: pl.DataFrame | None = None,
) -> pl.DataFrame:
    path = "reports/candidate_paper_ready.csv"
    if board.is_empty() or "decision" not in board.columns:
        return _empty_csv_schema_frame(path)
    selected = board.filter(pl.col("decision") == "PAPER_READY")
    if selected.is_empty():
        return _empty_csv_schema_frame(path)

    gate_by_proposal = _paper_ready_gate_by_proposal(paper_strategy_promotion_gate)
    rows = []
    for row in selected.to_dicts():
        gate = gate_by_proposal.get(_paper_proposal_id(row), {})
        if gate and _optional_bool(gate.get("paper_ready")) is True:
            updated = dict(row)
            updated["paper_days"] = _optional_int(gate.get("paper_days"))
            if gate.get("cost_source"):
                updated["cost_source_mix"] = gate.get("cost_source")
            ready_reason = "paper_strategy_pipeline_ready"
            decision_reasons = set(_jsonish_list(updated.get("decision_reasons")))
            decision_reasons.add(ready_reason)
            updated["decision_reasons"] = safe_json_dumps(sorted(decision_reasons))
            rows.append(updated)
        elif _has_formal_paper_ready_evidence(row):
            rows.append(row)
    if not rows:
        return _empty_csv_schema_frame(path)

    output = pl.DataFrame(rows, infer_schema_length=None)
    for column in CSV_SCHEMAS[path]:
        if column not in output.columns:
            output = output.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    sort_columns = [
        column
        for column in ["strategy_candidate", "symbol", "horizon_hours"]
        if column in output.columns
    ]
    output = output.sort(sort_columns) if sort_columns else output
    return output.select(CSV_SCHEMAS[path])


def _paper_ready_gate_by_proposal(frame: pl.DataFrame | None) -> dict[str, dict[str, Any]]:
    if frame is None or frame.is_empty() or "proposal_id" not in frame.columns:
        return {}
    rows = {}
    for row in frame.to_dicts():
        proposal_id = str(row.get("proposal_id") or "").strip()
        if proposal_id:
            rows[proposal_id] = row
    return rows


def _jsonish_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [str(parsed)] if parsed else []


def _has_formal_paper_ready_evidence(row: dict[str, Any]) -> bool:
    if (_optional_int(row.get("paper_days")) or 0) < 14:
        return False
    if (_optional_int(row.get("paper_pnl_observed_count")) or 0) <= 0:
        return False
    arrival_mid_coverage = _optional_float(row.get("arrival_mid_coverage"))
    if arrival_mid_coverage is None or arrival_mid_coverage < 0.8:
        return False
    if not _cost_source_mix_has_actual_or_mixed(row.get("cost_source_mix")):
        return False
    cost_sources = {source.lower() for source in _cost_source_mix_items(row.get("cost_source_mix"))}
    blocked_sources = {
        "fallback_not_live_safe",
        "public_spread_proxy",
        "public_microstructure_proxy",
        "global_default",
        "local_estimate",
        "conservative_shadow_cost",
    }
    return not bool(cost_sources & blocked_sources)


PAPER_PROPOSAL_IDS = {
    ("v5.sol_protect_alpha6_low_exception", "SOL-USDT"): (
        "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1"
    ),
    ("v5.f4_volume_expansion_entry", "SOL-USDT"): "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
}


def _paper_strategy_proposals_for_export(
    board: pl.DataFrame,
    *,
    research_portfolio: pl.DataFrame | None = None,
    persisted_proposals: pl.DataFrame | None = None,
) -> pl.DataFrame:
    path = "reports/paper_strategy_proposals.csv"
    from quant_lab.paper.proposals import build_configured_proposals, proposal_export_rows
    from quant_lab.paper.service import proposal_from_storage_row, proposal_rule_fingerprint

    persisted = persisted_proposals if persisted_proposals is not None else pl.DataFrame()
    if board.is_empty() or "decision" not in board.columns:
        persisted_rows = []
        now = datetime.now(UTC)
        for row in persisted.to_dicts():
            try:
                proposal = proposal_from_storage_row(row)
            except Exception:
                continue
            if proposal.expires_at > now:
                persisted_rows.append((proposal, {}))
        if not persisted_rows:
            return _empty_csv_schema_frame(path)
        return _csv_frame_with_schema(
            pl.DataFrame(proposal_export_rows(persisted_rows), infer_schema_length=None),
            CSV_SCHEMAS[path],
        )
    portfolio_overrides = _portfolio_status_overrides_by_candidate_symbol(
        research_portfolio if research_portfolio is not None else pl.DataFrame()
    )
    board_rows = board.to_dicts()
    latest_as_of_date = _latest_as_of_date(board_rows)
    rows = [
        row
        for row in board_rows
        if str(row.get("decision") or "").upper() == "PAPER_READY"
        and str(row.get("symbol") or "").strip().upper() not in {"", "UNKNOWN", "ALL", "*"}
        and _portfolio_override_for_row(row, portfolio_overrides) is None
    ]
    if latest_as_of_date is not None:
        rows = [
            row for row in rows if str(row.get("as_of_date") or "").strip() == latest_as_of_date
        ]
    if not rows:
        return _empty_csv_schema_frame(path)

    eligible_board = pl.DataFrame(rows, infer_schema_length=None)
    configured = [
        (proposal, evidence)
        for proposal, evidence in build_configured_proposals(eligible_board)
        if _paper_proposal_legacy_key(evidence) not in PAPER_PROPOSAL_IDS
    ]
    if not persisted.is_empty():
        evidence_by_identity = {
            (proposal.strategy_id, proposal.strategy_version): evidence
            for proposal, evidence in configured
        }
        current_by_identity = {
            (proposal.strategy_id, proposal.strategy_version): (proposal, evidence)
            for proposal, evidence in configured
        }
        now = datetime.now(UTC)
        for row in persisted.to_dicts():
            try:
                proposal = proposal_from_storage_row(row)
            except Exception:
                continue
            if proposal.expires_at <= now:
                continue
            identity = (proposal.strategy_id, proposal.strategy_version)
            evidence = evidence_by_identity.get(identity)
            generated_row = current_by_identity.get(identity)
            if evidence is None or generated_row is None:
                continue
            generated = generated_row[0]
            if proposal_rule_fingerprint(generated) != proposal_rule_fingerprint(proposal):
                continue
            current_by_identity[identity] = (proposal, evidence)
        configured = list(current_by_identity.values())

    best_by_candidate: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("strategy_candidate") or ""), str(row.get("symbol") or ""))
        current = best_by_candidate.get(key)
        if current is None or _paper_proposal_rank(row) > _paper_proposal_rank(current):
            best_by_candidate[key] = row

    created_at = datetime.now(UTC).isoformat()
    legacy_proposals = [
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
        if _paper_proposal_legacy_key(row) in PAPER_PROPOSAL_IDS
    ]
    proposal_rows = [*proposal_export_rows(configured), *legacy_proposals]
    if not proposal_rows:
        return _empty_csv_schema_frame(path)
    return _csv_frame_with_schema(
        pl.DataFrame(proposal_rows, infer_schema_length=None),
        CSV_SCHEMAS[path],
    )


def _paper_proposal_legacy_key(row: Mapping[str, Any]) -> tuple[str, str]:
    candidate = str(row.get("strategy_candidate") or "").strip()
    symbol = str(row.get("symbol") or row.get("v5_symbol") or "").strip().upper()
    return candidate, symbol.replace("/", "-").replace("_", "-")


def _paper_strategy_proposal_ack_for_export(
    proposal_ack: pl.DataFrame,
    *,
    paper_strategy_registry: pl.DataFrame,
) -> pl.DataFrame:
    path = "reports/paper_strategy_proposal_ack.csv"
    rows: list[dict[str, Any]] = []
    rows.extend(
        _paper_strategy_ack_report_row(row, from_registry=False) for row in _rows(proposal_ack)
    )
    for row in _rows(paper_strategy_registry):
        if not _paper_strategy_registry_row_has_ack_evidence(row):
            continue
        rows.append(_paper_strategy_ack_report_row(row, from_registry=True))
    return _paper_strategy_ack_report_frame(rows, path=path)


def _stable_paper_strategy_lifecycle_frame(
    generated: pl.DataFrame,
    persisted: pl.DataFrame,
    *,
    schema: list[str],
) -> pl.DataFrame:
    rows = [*_rows(persisted), *_rows(generated)]
    if not rows:
        return pl.DataFrame(schema={column: pl.Utf8 for column in schema})
    merged = _merge_paper_strategy_rows(rows, schema)
    generated_identity = {
        _paper_strategy_row_key(row): row for row in _rows(generated)
    }
    immutable = {
        "proposal_id",
        "proposal_hash",
        "strategy_id",
        "strategy_version",
        "strategy_family",
        "symbol",
        "timeframe",
        "max_holding_bars",
        "contract_version",
    }
    for row in merged:
        canonical = generated_identity.get(_paper_strategy_row_key(row))
        if canonical is None:
            continue
        for field in immutable & set(schema):
            row[field] = canonical.get(field)
    return _csv_frame_with_schema(
        pl.DataFrame(merged, infer_schema_length=None),
        schema,
    )


def _merge_paper_strategy_rows(
    rows: list[dict[str, Any]],
    schema: list[str],
) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = {column: raw.get(column) for column in schema}
        key = _paper_strategy_row_key(row)
        if not key:
            continue
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        if _paper_strategy_row_rank(row) >= _paper_strategy_row_rank(current):
            best[key] = _fill_blank_values(row, current, schema)
        else:
            best[key] = _fill_blank_values(current, row, schema)
    return sorted(
        best.values(),
        key=lambda item: (
            _ack_text(item.get("symbol")),
            _ack_text(item.get("proposal_id")),
            _ack_text(item.get("paper_tracker_id")),
        ),
    )


def _paper_strategy_registry_row_has_ack_evidence(row: dict[str, Any]) -> bool:
    status = _ack_text(row.get("status")).upper()
    accepted = _optional_bool(row.get("accepted"))
    return bool(
        accepted is True
        or _ack_text(row.get("accepted_at"))
        or _ack_text(row.get("reject_reason"))
        or status in {"ACKED", "PAPER_TRACKING", "PAPER_REVIEW", "REJECTED_BY_V5"}
    )


def _paper_strategy_ack_report_row(
    row: dict[str, Any],
    *,
    from_registry: bool,
) -> dict[str, Any]:
    return {
        "proposal_id": _ack_text(row.get("proposal_id") or row.get("strategy_id")),
        "proposal_hash": _ack_text(row.get("proposal_hash")),
        "paper_tracker_id": _ack_text(row.get("paper_tracker_id")),
        "accepted": _ack_bool_text(row.get("accepted")),
        "accepted_at": _ack_text(
            row.get("accepted_at") or row.get("ingest_ts") or row.get("created_at")
        ),
        "recommended_mode": _ack_text(row.get("recommended_mode"))
        or ("paper" if _paper_only_ack_row(row) else ""),
        "symbol": normalize_symbol(row.get("symbol")) or _ack_text(row.get("symbol")),
        "strategy_candidate": _ack_text(row.get("strategy_candidate")),
        "suggested_horizon": _ack_text(row.get("suggested_horizon")),
        "proposal_source": _ack_text(row.get("proposal_source"))
        or ("paper_strategy_registry" if from_registry else "v5_paper_strategy_proposal_ack"),
        "reject_reason": _ack_text(row.get("reject_reason")),
        "live_order_effect": _ack_text(row.get("live_order_effect")) or "paper_only_no_live_order",
        "source_pack_sha256": _ack_text(row.get("source_pack_sha256")),
        "source_v5_bundle_sha256": _ack_text(
            row.get("source_v5_bundle_sha256")
            or row.get("bundle_sha256")
            or row.get("source_bundle_sha256")
        ),
    }


def _paper_strategy_ack_report_frame(
    rows: list[dict[str, Any]],
    *,
    path: str,
) -> pl.DataFrame:
    if not rows:
        return _empty_csv_schema_frame(path)
    schema = CSV_SCHEMAS[path]
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = {column: row.get(column) for column in schema}
        key = _paper_strategy_row_key(normalized)
        if not key:
            continue
        current = best.get(key)
        if current is None:
            best[key] = normalized
            continue
        if _paper_strategy_row_rank(normalized) >= _paper_strategy_row_rank(current):
            best[key] = _fill_blank_values(normalized, current, schema)
        else:
            best[key] = _fill_blank_values(current, normalized, schema)
    if not best:
        return _empty_csv_schema_frame(path)
    ordered = sorted(
        best.values(),
        key=lambda item: (
            _ack_text(item.get("symbol")),
            _ack_text(item.get("proposal_id")),
            _ack_text(item.get("paper_tracker_id")),
        ),
    )
    return pl.DataFrame(ordered, infer_schema_length=None).select(schema)


def _paper_strategy_row_key(row: dict[str, Any]) -> str:
    proposal_id = _ack_text(row.get("proposal_id") or row.get("strategy_id"))
    paper_tracker_id = _ack_text(row.get("paper_tracker_id"))
    if paper_tracker_id:
        return f"tracker:{paper_tracker_id}"
    if proposal_id:
        return f"id:{proposal_id}"
    candidate = _ack_text(row.get("strategy_candidate"))
    symbol = normalize_symbol(row.get("symbol")) or _ack_text(row.get("symbol"))
    horizon = _ack_text(row.get("suggested_horizon"))
    if candidate or symbol:
        return f"candidate:{candidate}|symbol:{symbol}|horizon:{horizon}"
    return ""


def _paper_strategy_row_rank(row: dict[str, Any]) -> tuple[int, datetime, int]:
    status = _ack_text(row.get("status")).upper()
    lifecycle_score = 0
    if _ack_bool_text(row.get("accepted")) == "true" or status in {
        "ACKED",
        "PAPER_TRACKING",
        "PAPER_REVIEW",
    }:
        lifecycle_score = 3
    elif _ack_text(row.get("reject_reason")) or status.startswith("REJECTED"):
        lifecycle_score = 2
    elif _ack_text(row.get("paper_tracker_id")):
        lifecycle_score = 1
    timestamp = _max_export_ts(
        row.get("accepted_at"),
        row.get("paper_start_at"),
        row.get("first_seen_at"),
        row.get("created_at"),
        row.get("ingest_ts"),
    ) or datetime.min.replace(tzinfo=UTC)
    completeness = sum(1 for value in row.values() if _ack_text(value))
    return lifecycle_score, timestamp, completeness


def _max_export_ts(*values: Any) -> datetime | None:
    parsed = [_parse_export_ts(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return max(parsed) if parsed else None


def _fill_blank_values(
    preferred: dict[str, Any],
    fallback: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    merged = dict(preferred)
    for column in columns:
        if not _ack_text(merged.get(column)) and _ack_text(fallback.get(column)):
            merged[column] = fallback.get(column)
    return merged


def _paper_only_ack_row(row: dict[str, Any]) -> bool:
    value = _optional_bool(row.get("paper_only"))
    if value is not None:
        return value
    effect = _ack_text(row.get("live_order_effect")).lower()
    return "paper_only" in effect


def _ack_bool_text(value: Any) -> str:
    parsed = _optional_bool(value)
    if parsed is True:
        return "true"
    if parsed is False:
        return "false"
    return _ack_text(value)


def _ack_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value).strip()


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


def _duplicate_event_report(events: pl.DataFrame) -> pl.DataFrame:
    path = "reports/duplicate_event_report.csv"
    if events.is_empty() or "canonical_event_id" not in events.columns:
        return _empty_csv_schema_frame(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in events.to_dicts():
        event_id = str(row.get("canonical_event_id") or "").strip()
        if event_id:
            grouped.setdefault(event_id, []).append(row)
    rows = []
    for event_id, group in grouped.items():
        strategies = sorted(
            {
                str(row.get("strategy_id") or row.get("strategy_candidate") or "")
                for row in group
                if str(row.get("strategy_id") or row.get("strategy_candidate") or "")
            }
        )
        count = max(len(strategies), 1)
        rows.append(
            {
                "canonical_event_id": event_id,
                "strategy_evaluation_count": len(group),
                "strategy_ids": safe_json_dumps(strategies),
                "event_independence_weight": 1.0 / count,
                "duplicate_evidence": len(group) > 1,
            }
        )
    return _csv_frame_with_schema(
        pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(),
        CSV_SCHEMAS[path],
    )


def _duplicate_factor_report(factors: pl.DataFrame) -> pl.DataFrame:
    path = "reports/duplicate_factor_report.csv"
    if factors.is_empty() or "canonical_factor_id" not in factors.columns:
        return _empty_csv_schema_frame(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in factors.to_dicts():
        canonical = str(row.get("canonical_factor_id") or "").strip()
        if canonical:
            grouped.setdefault(canonical, []).append(row)
    rows = []
    for canonical, group in grouped.items():
        factor_ids = sorted({str(row.get("factor_id") or "") for row in group})
        duplicate_ids = sorted(
            str(row.get("factor_id") or "") for row in group if str(row.get("duplicate_of") or "")
        )
        rows.append(
            {
                "canonical_factor_id": canonical,
                "factor_formula_hash": str(group[0].get("factor_formula_hash") or ""),
                "operator_graph_hash": str(group[0].get("operator_graph_hash") or ""),
                "factor_count": len(factor_ids),
                "factor_ids": safe_json_dumps(factor_ids),
                "duplicate_factor_ids": safe_json_dumps(duplicate_ids),
                "effective_independence_weight": 1.0 / max(len(factor_ids), 1),
            }
        )
    return _csv_frame_with_schema(
        pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(),
        CSV_SCHEMAS[path],
    )


def _lake_file_index_count(root: Path) -> int | None:
    try:
        frame = read_parquet_dataset(root / "bronze" / "lake_file_index")
    except Exception:
        return None
    if frame.is_empty():
        return None
    if "path" in frame.columns:
        return frame.select(pl.col("path").n_unique().alias("count")).item(0, "count")
    return frame.height


def _lake_file_index_growth_24h_count(root: Path) -> int | None:
    try:
        frame = read_parquet_dataset(root / "bronze" / "lake_file_index")
    except Exception:
        return None
    if frame.is_empty() or "mtime_ns" not in frame.columns:
        return None
    cutoff_ns = int((datetime.now(UTC) - timedelta(hours=24)).timestamp() * 1_000_000_000)
    try:
        return frame.select(
            (pl.col("mtime_ns").cast(pl.Int64, strict=False) >= cutoff_ns).sum().alias("count")
        ).item(0, "count")
    except Exception:
        return None


def _refresh_lake_file_index_for_export(root: Path) -> dict[str, Any]:
    dataset_paths = sorted(
        {
            str(path).replace("\\", "/")
            for path in readers.DATASET_PATHS.values()
            if str(path).strip()
        }
    )
    started_at = datetime.now(UTC)
    try:
        frame = build_lake_file_index(root, dataset_paths)
    except Exception as exc:
        return {
            "ok": False,
            "dataset_count": len(dataset_paths),
            "indexed_rows": 0,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "error": _safe_warning_text(f"{type(exc).__name__}:{exc}"),
        }
    return {
        "ok": True,
        "dataset_count": len(dataset_paths),
        "indexed_rows": frame.height,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "error": "",
    }


def _lake_file_health_for_export(root: Path) -> dict[str, Any]:
    try:
        summary = lake_file_health_summary(root)
    except Exception as exc:
        return {
            "ok": False,
            "error": _safe_warning_text(f"{type(exc).__name__}:{exc}"),
        }
    rows = list(summary.get("rows") or [])
    top_rows = sorted(
        rows,
        key=lambda row: int(row.get("parquet_file_count") or 0),
        reverse=True,
    )[:10]
    warning_rows = [
        row for row in rows if str(row.get("status") or "").upper() not in {"", "OK", "PASS"}
    ][:10]
    return {
        "ok": True,
        "dataset_count": int(summary.get("dataset_count") or 0),
        "total_parquet_files": int(summary.get("total_parquet_files") or 0),
        "warning_count": int(summary.get("warning_count") or 0),
        "top_file_count_datasets": [_lake_file_health_row_for_manifest(row) for row in top_rows],
        "warnings": [_lake_file_health_row_for_manifest(row) for row in warning_rows],
    }


def _lake_file_health_row_for_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": str(row.get("dataset") or ""),
        "parquet_file_count": int(row.get("parquet_file_count") or 0),
        "partition_dir_count": int(row.get("partition_dir_count") or 0),
        "small_file_count": int(row.get("small_file_count") or 0),
        "small_file_ratio": float(row.get("small_file_ratio") or 0.0),
        "status": str(row.get("status") or ""),
        "warning": row.get("warning"),
    }


def _github_ci_status_for_export(v5_context: Mapping[str, Any]) -> pl.DataFrame:
    path = "reports/github_ci_status.csv"
    observed_at = datetime.now(UTC).isoformat()
    quant_lab_commit = _git_commit_full()
    v5_commit = _selected_v5_bundle_git_commit(v5_context)
    rows: list[dict[str, Any]] = []
    targets = [
        (
            "quant-lab",
            os.environ.get("QUANT_LAB_GITHUB_REPO", "zhr2038/quant-lab"),
            quant_lab_commit,
        ),
        (
            "v5",
            os.environ.get("V5_GITHUB_REPO", "zhr2038/V5-prod"),
            v5_commit,
        ),
    ]
    enabled = _flag_enabled("QUANT_LAB_EXPORT_GITHUB_CI_STATUS", default=False)
    if not enabled:
        for component, repo, commit_sha in targets:
            rows.append(
                _github_ci_status_row(
                    component=component,
                    repo=repo,
                    commit_sha=commit_sha,
                    observed_at=observed_at,
                    workflow_status="not_checked",
                    workflow_conclusion="disabled",
                    source="env_disabled",
                    error="set QUANT_LAB_EXPORT_GITHUB_CI_STATUS=1 to query GitHub Actions",
                )
            )
        return pl.DataFrame(rows, infer_schema_length=None).select(CSV_SCHEMAS[path])

    timeout = float(os.environ.get("QUANT_LAB_EXPORT_GITHUB_CI_TIMEOUT_SECONDS", "8"))
    for component, repo, commit_sha in targets:
        if not commit_sha:
            rows.append(
                _github_ci_status_row(
                    component=component,
                    repo=repo,
                    commit_sha=commit_sha,
                    observed_at=observed_at,
                    workflow_status="not_observable",
                    workflow_conclusion="missing_commit",
                    source="local",
                    error="commit_sha_missing",
                )
            )
            continue
        try:
            run = _github_actions_latest_run(repo=repo, commit_sha=commit_sha, timeout=timeout)
        except Exception as exc:  # pragma: no cover - exercised by production network state.
            rows.append(
                _github_ci_status_row(
                    component=component,
                    repo=repo,
                    commit_sha=commit_sha,
                    observed_at=observed_at,
                    workflow_status="not_observable",
                    workflow_conclusion="query_error",
                    source="github_actions_api",
                    error=_safe_warning_text(f"{type(exc).__name__}:{exc}"),
                )
            )
            continue
        if not run:
            rows.append(
                _github_ci_status_row(
                    component=component,
                    repo=repo,
                    commit_sha=commit_sha,
                    observed_at=observed_at,
                    workflow_status="not_observable",
                    workflow_conclusion="no_run",
                    source="github_actions_api",
                    error="no workflow run found for commit",
                )
            )
            continue
        rows.append(
            _github_ci_status_row(
                component=component,
                repo=repo,
                commit_sha=commit_sha,
                observed_at=observed_at,
                workflow_name=run.get("name"),
                workflow_run_id=run.get("id"),
                workflow_status=run.get("status"),
                workflow_conclusion=run.get("conclusion"),
                event=run.get("event"),
                head_sha=run.get("head_sha"),
                created_at=run.get("created_at"),
                updated_at=run.get("updated_at"),
                html_url=run.get("html_url"),
                source="github_actions_api",
                error="",
            )
        )
    return pl.DataFrame(rows, infer_schema_length=None).select(CSV_SCHEMAS[path])


def _github_ci_status_frame_from_context(v5_context: Mapping[str, Any]) -> pl.DataFrame:
    rows = v5_context.get("github_ci_status_rows")
    if isinstance(rows, list):
        return pl.DataFrame(rows, infer_schema_length=None).select(
            CSV_SCHEMAS["reports/github_ci_status.csv"]
        )
    return _github_ci_status_for_export(v5_context)


def _github_ci_status_summary(frame: pl.DataFrame) -> dict[str, Any]:
    rows = frame.to_dicts() if not frame.is_empty() else []
    required = {"quant-lab", "v5"}
    present = {str(row.get("component") or "") for row in rows}
    missing = sorted(required - present)
    failures = [
        row for row in rows if str(row.get("workflow_conclusion") or "").lower() not in {"success"}
    ]
    overall = "PASS" if rows and not missing and not failures else "WARNING"
    known_bad = {
        "failure",
        "cancelled",
        "timed_out",
        "startup_failure",
        "action_required",
    }
    if any(str(row.get("workflow_conclusion") or "").lower() in known_bad for row in rows):
        overall = "FAIL"
    return {
        "overall_status": overall,
        "checked_components": sorted(present),
        "missing_components": missing,
        "failure_count": len(failures),
        "components": {
            str(row.get("component") or ""): {
                "repo": row.get("repo"),
                "commit_sha": row.get("commit_sha"),
                "workflow_status": row.get("workflow_status"),
                "workflow_conclusion": row.get("workflow_conclusion"),
                "workflow_run_id": row.get("workflow_run_id"),
                "html_url": row.get("html_url"),
                "error": row.get("error"),
            }
            for row in rows
        },
    }


def _github_ci_status_row(
    *,
    component: str,
    repo: str | None,
    commit_sha: str | None,
    observed_at: str,
    workflow_name: Any = None,
    workflow_run_id: Any = None,
    workflow_status: Any = None,
    workflow_conclusion: Any = None,
    event: Any = None,
    head_sha: Any = None,
    created_at: Any = None,
    updated_at: Any = None,
    html_url: Any = None,
    source: str,
    error: str,
) -> dict[str, Any]:
    return {
        "component": component,
        "repo": repo,
        "commit_sha": commit_sha,
        "workflow_name": workflow_name,
        "workflow_run_id": workflow_run_id,
        "workflow_status": workflow_status,
        "workflow_conclusion": workflow_conclusion,
        "event": event,
        "head_sha": head_sha,
        "created_at": created_at,
        "updated_at": updated_at,
        "html_url": html_url,
        "observed_at": observed_at,
        "source": source,
        "error": error,
    }


def _github_actions_latest_run(
    *,
    repo: str | None,
    commit_sha: str,
    timeout: float,
) -> dict[str, Any] | None:
    repo_text = str(repo or "").strip()
    if "/" not in repo_text:
        raise ValueError("github_repo_must_be_owner_slash_repo")
    owner, name = repo_text.split("/", 1)
    url = (
        "https://api.github.com/repos/"
        f"{urllib.parse.quote(owner)}/{urllib.parse.quote(name)}"
        f"/actions/runs?head_sha={urllib.parse.quote(commit_sha)}&per_page=10"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "quant-lab-expert-export/1.0",
    }
    token = os.environ.get("QUANT_LAB_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        return None
    for run in runs:
        if isinstance(run, dict) and str(run.get("head_sha") or "").lower() == commit_sha.lower():
            return run
    return runs[0] if runs and isinstance(runs[0], dict) else None


def _selected_v5_bundle_git_commit(v5_context: Mapping[str, Any]) -> str | None:
    selected_path_raw = str(v5_context.get("selected_v5_bundle_path") or "").strip()
    if not selected_path_raw:
        return None
    selected_path = Path(selected_path_raw)
    if not selected_path.is_file():
        return None
    try:
        with tarfile.open(selected_path, "r:gz") as archive:
            for member in archive.getmembers():
                parts = PurePosixPath(member.name).parts
                if len(parts) != 2 or parts[-1] != "manifest.json" or not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                payload = json.loads(extracted.read().decode("utf-8"))
                value = payload.get("git_commit")
                return str(value).strip() if value else None
    except (OSError, tarfile.TarError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return None


def _api_latency_summary_for_export(root: Path) -> pl.DataFrame:
    _flush_api_metrics_service_before_export()
    window_minutes = _api_metrics_export_window_minutes()
    summary = api_metrics_summary(root, since_minutes=window_minutes)
    rows: list[dict[str, Any]] = [_api_latency_summary_row("__all__", summary)]
    production_summary = _production_v5_api_metrics_summary(root, since_minutes=window_minutes)
    if production_summary is not None:
        rows.append(_api_latency_summary_row("__production_v5__", production_summary))
    by_path = (
        summary.get("latency_by_path_ms")
        if isinstance(summary.get("latency_by_path_ms"), dict)
        else {}
    )
    for endpoint, metrics in sorted(by_path.items()):
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "endpoint": endpoint,
                "count": int(metrics.get("count") or 0),
                "p50_ms": _float_or_blank(metrics.get("p50")),
                "p90_ms": _float_or_blank(metrics.get("p90")),
                "p95_ms": _float_or_blank(metrics.get("p95")),
                "success_p95_ms": _float_or_blank(metrics.get("success_p95")),
                "error_p95_ms": _float_or_blank(metrics.get("error_p95")),
                "p99_ms": _float_or_blank(metrics.get("p99")),
                "max_ms": _float_or_blank(metrics.get("max")),
                "cache_hit_rate": _safe_rate(metrics.get("cache_hit_count"), metrics.get("count")),
                "avg_rows_returned": _safe_avg(
                    metrics.get("rows_returned_total"),
                    metrics.get("count"),
                ),
                "avg_response_bytes": _safe_avg(
                    metrics.get("response_bytes_total"),
                    metrics.get("count"),
                ),
                "avg_lake_scan_ms": _safe_avg(
                    metrics.get("lake_scan_ms_total"),
                    metrics.get("count"),
                ),
                "avg_serialize_ms": _safe_avg(
                    metrics.get("serialize_ms_total"),
                    metrics.get("count"),
                ),
                "avg_source_signature_ms": _safe_avg(
                    metrics.get("source_signature_ms_total"),
                    metrics.get("count"),
                ),
                "response_cache_hit_rate": _safe_rate(
                    metrics.get("response_cache_hit_count"),
                    metrics.get("count"),
                ),
                "dependency_meta_missing_count": int(
                    metrics.get("dependency_meta_missing_count") or 0
                ),
                "dependency_meta_missing_rate": _safe_rate(
                    metrics.get("dependency_meta_missing_count"),
                    metrics.get("count"),
                ),
                "error_count": _api_latency_error_count(metrics),
                "auth_error_count": int(metrics.get("auth_error_count") or 0),
                "auth_error_rate": _safe_rate(
                    metrics.get("auth_error_count"),
                    metrics.get("count"),
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).select(
        CSV_SCHEMAS["reports/api_latency_summary.csv"]
    )


def _api_latency_summary_row(endpoint: str, summary: dict[str, Any]) -> dict[str, Any]:
    latency = summary.get("latency_ms") if isinstance(summary.get("latency_ms"), dict) else {}
    by_path = (
        summary.get("latency_by_path_ms")
        if isinstance(summary.get("latency_by_path_ms"), dict)
        else {}
    )
    overall_error_count = sum(
        _api_latency_error_count(metrics)
        for metrics in by_path.values()
        if isinstance(metrics, dict)
    )
    if overall_error_count <= 0:
        overall_error_count = sum(
            int(value or 0)
            for value in (
                summary.get("by_error_type")
                if isinstance(summary.get("by_error_type"), dict)
                else {}
            ).values()
        )
    request_count = int(summary.get("request_count") or 0)
    return {
        "endpoint": endpoint,
        "count": request_count,
        "p50_ms": _float_or_blank(latency.get("p50")),
        "p90_ms": _float_or_blank(latency.get("p90")),
        "p95_ms": _float_or_blank(latency.get("p95")),
        "success_p95_ms": _float_or_blank(latency.get("success_p95")),
        "error_p95_ms": _float_or_blank(latency.get("error_p95")),
        "p99_ms": _float_or_blank(latency.get("p99")),
        "max_ms": _float_or_blank(latency.get("max")),
        "cache_hit_rate": _safe_rate(summary.get("cache_hit_count"), request_count),
        "avg_rows_returned": _safe_avg(summary.get("rows_returned_total"), request_count),
        "avg_response_bytes": _safe_avg(summary.get("response_bytes_total"), request_count),
        "avg_lake_scan_ms": _safe_avg(summary.get("lake_scan_ms_total"), request_count),
        "avg_serialize_ms": _safe_avg(summary.get("serialize_ms_total"), request_count),
        "avg_source_signature_ms": _safe_avg(
            summary.get("source_signature_ms_total"), request_count
        ),
        "response_cache_hit_rate": _safe_rate(
            summary.get("response_cache_hit_count"), request_count
        ),
        "dependency_meta_missing_count": int(summary.get("dependency_meta_missing_count") or 0),
        "dependency_meta_missing_rate": _safe_rate(
            summary.get("dependency_meta_missing_count"), request_count
        ),
        "error_count": overall_error_count,
        "auth_error_count": int(summary.get("auth_error_count") or 0),
        "auth_error_rate": _safe_rate(summary.get("auth_error_count"), request_count),
    }


def _production_v5_api_metrics_summary(
    root: Path,
    *,
    since_minutes: int,
) -> dict[str, Any] | None:
    hosts = _csv_env_values("QUANT_LAB_API_METRICS_PRODUCTION_CLIENT_HOSTS")
    if not hosts:
        return None
    client_ids = _csv_env_values("QUANT_LAB_API_METRICS_PRODUCTION_CLIENT_IDS") or [
        "v5.quant_lab_client",
        "v5.dashboard_proxy",
    ]
    return api_metrics_summary(
        root,
        since_minutes=since_minutes,
        client_hosts=hosts,
        client_ids=client_ids,
    )


def _csv_env_values(name: str) -> list[str]:
    return [item.strip() for item in str(os.environ.get(name) or "").split(",") if item.strip()]


def _api_metrics_export_window_minutes() -> int:
    raw = str(os.environ.get("QUANT_LAB_API_METRICS_EXPORT_WINDOW_MINUTES", "")).strip()
    if not raw:
        return 24 * 60
    try:
        value = int(raw)
    except ValueError:
        return 24 * 60
    return value if value > 0 else 24 * 60


def _api_latency_error_count(metrics: dict[str, Any]) -> int:
    status_error_count = int(metrics.get("server_error_count") or 0) + int(
        metrics.get("client_error_count") or 0
    )
    if "error_count" not in metrics or metrics.get("error_count") is None:
        return status_error_count
    try:
        return max(int(float(metrics.get("error_count") or 0)), status_error_count)
    except Exception:
        return status_error_count


def _api_error_summary_for_export(root: Path) -> pl.DataFrame:
    path = "reports/api_error_summary.csv"
    rows = api_error_summary(root, since_minutes=24 * 60)
    if not rows:
        return _empty_csv_schema_frame(path)
    return pl.DataFrame(rows, infer_schema_length=None).select(CSV_SCHEMAS[path])


def _flush_api_metrics_service_before_export() -> None:
    enabled = str(os.environ.get("QUANT_LAB_API_METRICS_EXPORT_FLUSH_ENABLED", "1")).strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    token = str(os.environ.get("QUANT_LAB_API_TOKEN") or "").strip()
    if not token:
        return
    url = str(
        os.environ.get(
            "QUANT_LAB_API_METRICS_EXPORT_FLUSH_URL",
            "http://127.0.0.1:8027/v1/ops/api-metrics",
        )
    ).strip()
    if not url:
        return
    try:
        timeout = float(os.environ.get("QUANT_LAB_API_METRICS_EXPORT_FLUSH_TIMEOUT_SECONDS", "1.5"))
    except ValueError:
        timeout = 1.5
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "quant-lab-daily-export/1.0",
            "X-Quant-Lab-Client-Id": "quant-lab.daily_export",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=max(timeout, 0.1)) as response:
            response.read(1024)
    except Exception:
        return


def _api_latency_summary_md(root: Path) -> str:
    frame = _api_latency_summary_for_export(root)
    if frame.is_empty():
        return "# API Latency Summary\n\nNo API metrics observed.\n"
    lines = [
        "# API Latency Summary",
        "",
        (
            "Recent read-only API latency over the last "
            f"{_api_metrics_export_window_minutes()} minutes. "
            "`/v1/health` is intentionally light; lake freshness checks belong to "
            "`/v1/health/deep`."
        ),
        "",
        "| endpoint | count | p50 ms | p95 ms | max ms | cache hit | "
        "response cache hit | avg source signature ms | dependency meta missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.head(12).to_dicts():
        lines.append(
            f"| {row.get('endpoint') or ''} | {row.get('count') or 0} | "
            f"{row.get('p50_ms') or ''} | {row.get('p95_ms') or ''} | {row.get('max_ms') or ''} | "
            f"{row.get('cache_hit_rate') or ''} | {row.get('response_cache_hit_rate') or ''} | "
            f"{row.get('avg_source_signature_ms') or ''} | "
            f"{row.get('dependency_meta_missing_count') or 0} |"
        )
    return "\n".join(lines) + "\n"


def _float_or_blank(value: Any) -> float | str:
    try:
        return round(float(value), 3)
    except Exception:
        return ""


def _safe_rate(numerator: Any, denominator: Any) -> float | str:
    try:
        denom = float(denominator or 0)
        if denom <= 0:
            return ""
        return round(float(numerator or 0) / denom, 4)
    except Exception:
        return ""


def _safe_avg(total: Any, count: Any) -> float | str:
    try:
        denom = float(count or 0)
        if denom <= 0:
            return ""
        return round(float(total or 0) / denom, 3)
    except Exception:
        return ""


def _strategy_opportunity_advisory_for_export(
    *,
    alpha_discovery_board: pl.DataFrame,
    strategy_evidence: pl.DataFrame,
    paper_proposals: pl.DataFrame,
    risk_permissions: pl.DataFrame,
    cost_health: pl.DataFrame,
    paper_daily: pl.DataFrame,
    paper_slippage: pl.DataFrame,
    research_portfolio: pl.DataFrame | None = None,
    entry_quality_advisory: pl.DataFrame | None = None,
    regime_strategy_advisory: pl.DataFrame | None = None,
    risk_on_multi_buy_shadow: pl.DataFrame | None = None,
    alpha_factory_results: pl.DataFrame | None = None,
    alpha_factory_promotion_queue: pl.DataFrame | None = None,
    expanded_universe_maturity: pl.DataFrame | None = None,
    bottom_zone_reversal_shadow: pl.DataFrame | None = None,
    factor_strategy_bridge_candidates: pl.DataFrame | None = None,
) -> pl.DataFrame:
    path = "reports/strategy_opportunity_advisory.csv"
    source = (
        alpha_discovery_board
        if not alpha_discovery_board.is_empty()
        else _advisory_board_from_strategy_evidence(strategy_evidence)
    )
    proposal_by_key = _latest_rows_by_candidate_symbol(paper_proposals)
    paper_daily_by_key = _latest_rows_by_candidate_symbol(paper_daily)
    slippage_by_key = _latest_rows_by_candidate_symbol(paper_slippage)
    portfolio_overrides = _portfolio_status_overrides_by_candidate_symbol(
        research_portfolio if research_portfolio is not None else pl.DataFrame()
    )
    promotion_frame = (
        alpha_factory_promotion_queue
        if alpha_factory_promotion_queue is not None
        else pl.DataFrame()
    )
    alpha_factory_metadata = _alpha_factory_result_metadata_by_candidate_symbol(
        alpha_factory_results if alpha_factory_results is not None else pl.DataFrame()
    )
    alpha_factory_promotions = _alpha_factory_promotion_overrides_by_candidate_symbol(
        promotion_frame
    )
    risk_context = _advisory_risk_context(risk_permissions)
    latest_cost_health = _latest_cost_health_context(cost_health)
    git_commit = _git_commit()
    source_version = _source_version("strategy_opportunity_advisory", git_commit)

    rows: list[dict[str, Any]] = []
    source_rows = source.to_dicts() if not source.is_empty() else []
    latest_as_of_date = _latest_as_of_date(source_rows)
    if latest_as_of_date is not None:
        source_rows = [
            row
            for row in source_rows
            if str(row.get("as_of_date") or "").strip() == latest_as_of_date
        ]
    for row in source_rows:
        symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
        if symbol == "UNKNOWN":
            continue
        candidate = str(row.get("strategy_candidate") or row.get("candidate_name") or "").strip()
        key = (candidate, symbol)
        proposal = proposal_by_key.get(key, {})
        paper = paper_daily_by_key.get(key, {})
        slippage = slippage_by_key.get(key, {})
        decision = str(row.get("decision") or "RESEARCH_ONLY").strip().upper()
        paper_decision = str(paper.get("latest_board_decision") or "").strip().upper()
        if paper_decision in {"KEEP_SHADOW", "REGIME_SHADOW", "KILL"} and decision in {
            "PAPER_READY",
            "LIVE_SMALL_READY",
        }:
            decision = paper_decision
        if decision == "LIVE_SMALL_READY" and not bool(paper.get("live_eligible")):
            decision = "PAPER_READY"
        portfolio_override = _portfolio_override_for_candidate_symbol(
            portfolio_overrides,
            candidate,
            symbol,
        )
        decision, recommended_mode, portfolio_reasons = _portfolio_overridden_decision_mode(
            decision=decision,
            recommended_mode="",
            portfolio_override=portfolio_override,
        )
        recommended_mode = _advisory_recommended_mode(decision)
        cost_quality = _advisory_cost_quality(row.get("cost_source_mix"), latest_cost_health)
        slippage_coverage = _optional_float(
            slippage.get("paper_slippage_coverage")
            or slippage.get("arrival_mid_coverage")
            or paper.get("arrival_mid_coverage")
        )
        paper_days = _optional_int(paper.get("paper_days") or row.get("paper_days")) or 0
        entry_day_count = _optional_int(paper.get("entry_day_count")) or 0
        paper_pnl_observed_count = _optional_int(paper.get("paper_pnl_observed_count")) or 0
        live_block_reasons = _advisory_live_block_reasons(
            row=row,
            proposal=proposal,
            decision=decision,
            cost_quality=cost_quality,
            paper_days=paper_days,
            entry_day_count=entry_day_count,
            paper_pnl_observed_count=paper_pnl_observed_count,
            slippage_coverage=slippage_coverage,
            risk_context=risk_context,
        )
        if portfolio_reasons:
            live_block_reasons = sorted({*live_block_reasons, *portfolio_reasons})
        alpha_factory_meta = _alpha_factory_metadata_for_row(row, alpha_factory_metadata)
        source_module = row.get("source_module") or alpha_factory_meta.get("source_module")
        template_family = row.get("template_family") or alpha_factory_meta.get("template_family")
        candidate_id = row.get("candidate_id") or alpha_factory_meta.get("candidate_id")
        promotion_state = row.get("promotion_state") or alpha_factory_meta.get("promotion_state")
        alpha_factory_score = _optional_float(row.get("alpha_factory_score"))
        if alpha_factory_score is None:
            alpha_factory_score = _optional_float(alpha_factory_meta.get("alpha_factory_score"))
        cost_quality_score = _optional_float(row.get("cost_quality_score"))
        if cost_quality_score is None:
            cost_quality_score = _optional_float(alpha_factory_meta.get("cost_quality_score"))
        paper_ready_block_reasons = _json_listish(row.get("paper_ready_block_reasons"))
        if not paper_ready_block_reasons:
            paper_ready_block_reasons = _json_listish(
                alpha_factory_meta.get("paper_ready_block_reasons")
            )
        generated_at = _advisory_generated_at(row)
        rows.append(
            {
                "as_of_ts": _advisory_as_of_ts(row),
                "generated_at": generated_at,
                "expires_at": _advisory_expires_at(row, generated_at),
                "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
                "schema_version": str(
                    row.get("schema_version") or STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION
                ),
                "quant_lab_git_commit": _observable_text(row.get("quant_lab_git_commit"))
                or git_commit,
                "source_version": _observable_text(row.get("source_version")) or source_version,
                "would_block_if_enabled": _advisory_would_block_if_enabled(
                    decision=decision,
                    recommended_mode=recommended_mode,
                    sample_count=_optional_int(row.get("sample_count")),
                    row=row,
                ),
                "would_enter": _advisory_would_enter(
                    decision=decision,
                    recommended_mode=recommended_mode,
                    sample_count=_optional_int(row.get("sample_count")),
                    row=row,
                ),
                "no_sample_reason": _advisory_no_sample_reason(
                    decision=decision,
                    recommended_mode=recommended_mode,
                    sample_count=_optional_int(row.get("sample_count")),
                    row=row,
                ),
                "strategy_id": _advisory_strategy_id(candidate, symbol),
                "symbol": symbol,
                "v5_symbol": _v5_symbol(symbol),
                "strategy_candidate": candidate,
                "decision": decision,
                "recommended_mode": recommended_mode,
                "horizon_hours": _optional_int(row.get("horizon_hours")),
                "sample_count": _optional_int(row.get("sample_count")),
                "complete_sample_count": _optional_int(row.get("complete_sample_count")),
                "avg_net_bps": _optional_float(row.get("avg_net_bps")),
                "p25_net_bps": _optional_float(row.get("p25_net_bps")),
                "win_rate": _optional_float(row.get("win_rate")),
                "cost_source_mix": row.get("cost_source_mix"),
                "cost_quality": cost_quality,
                "source_module": source_module,
                "template_family": template_family,
                "candidate_id": candidate_id,
                "promotion_state": promotion_state,
                "alpha_factory_score": alpha_factory_score,
                "universe_type": row.get("universe_type"),
                "expanded_universe_maturity_state": (
                    row.get("expanded_universe_maturity_state")
                    or row.get("maturity_state")
                    or row.get("decision")
                ),
                "cost_quality_score": cost_quality_score,
                "paper_ready_block_reasons": safe_json_dumps(paper_ready_block_reasons),
                "advisory_intent": _export_advisory_intent(recommended_mode),
                "paper_days": paper_days,
                "entry_day_count": entry_day_count,
                "paper_pnl_observed_count": paper_pnl_observed_count,
                "slippage_coverage": slippage_coverage,
                "live_block_reasons": safe_json_dumps(live_block_reasons),
                "max_paper_notional_usdt": _advisory_max_paper_notional(recommended_mode),
                "max_live_notional_usdt": 0.0,
            }
        )
    rows.extend(
        _entry_quality_opportunity_rows(
            entry_quality_advisory if entry_quality_advisory is not None else pl.DataFrame()
        )
    )
    rows.extend(
        _regime_router_opportunity_rows(
            regime_strategy_advisory if regime_strategy_advisory is not None else pl.DataFrame()
        )
    )
    rows.extend(
        _risk_on_multi_buy_opportunity_rows(
            risk_on_multi_buy_shadow if risk_on_multi_buy_shadow is not None else pl.DataFrame()
        )
    )
    rows.extend(
        _expanded_universe_paper_opportunity_rows(
            (
                expanded_universe_maturity
                if expanded_universe_maturity is not None
                else pl.DataFrame()
            ),
            git_commit=git_commit,
            source_version=source_version,
        )
    )
    rows.extend(
        _bottom_zone_probe_paper_opportunity_rows(
            (
                bottom_zone_reversal_shadow
                if bottom_zone_reversal_shadow is not None
                else pl.DataFrame()
            ),
            git_commit=git_commit,
            source_version=source_version,
        )
    )
    rows.extend(
        _factor_strategy_bridge_review_opportunity_rows(
            (
                factor_strategy_bridge_candidates
                if factor_strategy_bridge_candidates is not None
                else pl.DataFrame()
            ),
            git_commit=git_commit,
            source_version=source_version,
        )
    )
    if not rows:
        return _empty_csv_schema_frame(path)
    for row in rows:
        row.setdefault("live_order_effect", "read_only_no_live_order")
    rows = _apply_alpha_factory_promotion_overrides(rows, alpha_factory_promotions)
    rows = _apply_research_portfolio_overrides(rows, portfolio_overrides)
    rows = _dedupe_strategy_opportunity_rows_by_logical_key(rows)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .sort(["strategy_candidate", "symbol", "horizon_hours"])
        .select(CSV_SCHEMAS[path])
    )


def _strategy_opportunity_advisory_from_frames(
    frames: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    alpha_discovery_board = _alpha_discovery_board_for_export(
        frames.get("alpha_discovery_board", pl.DataFrame())
    )
    strategy_evidence = _strategy_evidence_for_export(
        frames.get("strategy_evidence", pl.DataFrame())
    )
    telemetry_latest_ts = _latest_v5_bundle_ts(frames)
    risk = _risk_permissions_for_export(
        frames.get("risk_permission", pl.DataFrame()),
        frames,
        telemetry_latest_ts=telemetry_latest_ts,
    )
    _, paper_daily, paper_slippage = _paper_tracking_frames_for_export(frames)
    research_portfolio = dedupe_research_portfolio_status(
        frames.get("research_portfolio_status", pl.DataFrame())
    )
    paper_proposals = _paper_strategy_proposals_for_export(
        alpha_discovery_board,
        research_portfolio=research_portfolio,
        persisted_proposals=frames.get("paper_strategy_proposal", pl.DataFrame()),
    )
    return _strategy_opportunity_advisory_for_export(
        alpha_discovery_board=alpha_discovery_board,
        strategy_evidence=strategy_evidence,
        paper_proposals=paper_proposals,
        risk_permissions=risk,
        cost_health=frames.get("cost_health_daily", pl.DataFrame()),
        paper_daily=paper_daily,
        paper_slippage=paper_slippage,
        research_portfolio=research_portfolio,
        entry_quality_advisory=frames.get("v5_entry_quality_advisory", pl.DataFrame()),
        regime_strategy_advisory=frames.get("regime_strategy_advisory", pl.DataFrame()),
        risk_on_multi_buy_shadow=frames.get("v5_risk_on_multi_buy_shadow", pl.DataFrame()),
        alpha_factory_results=frames.get("alpha_factory_result", pl.DataFrame()),
        alpha_factory_promotion_queue=frames.get("alpha_factory_promotion_queue", pl.DataFrame()),
        expanded_universe_maturity=frames.get(
            "expanded_universe_candidate_maturity",
            pl.DataFrame(),
        ),
        bottom_zone_reversal_shadow=frames.get("bottom_zone_reversal_shadow", pl.DataFrame()),
        factor_strategy_bridge_candidates=frames.get(
            "factor_strategy_bridge_candidates",
            pl.DataFrame(),
        ),
    )


def _expanded_universe_paper_opportunity_rows(
    maturity: pl.DataFrame,
    *,
    git_commit: str,
    source_version: str,
) -> list[dict[str, Any]]:
    if maturity.is_empty():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in maturity.to_dicts():
        symbol = normalize_symbol(raw.get("symbol"))
        if symbol not in {"HYPE-USDT", "WLD-USDT"} or symbol in seen:
            continue
        state = str(
            raw.get("expanded_universe_maturity_state")
            or raw.get("maturity_state")
            or raw.get("decision")
            or ""
        ).upper()
        if state != "PAPER_READY":
            continue
        seen.add(symbol)
        base = symbol.split("-")[0]
        generated_at = _advisory_generated_at(raw)
        strategy_id = f"{base}_EXPANDED_UNIVERSE_PAPER_V1"
        strategy_candidate = f"v5.expanded_universe_{base.lower()}_paper"
        live_block_reasons = [
            "expanded_universe_not_live_approved",
            "quant_lab_live_command_not_allowed",
            "v5_local_live_not_controlled_by_quant_lab",
        ]
        rows.append(
            {
                "as_of_ts": _advisory_as_of_ts(raw),
                "generated_at": generated_at,
                "expires_at": _advisory_expires_at(raw, generated_at),
                "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
                "schema_version": STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION,
                "quant_lab_git_commit": git_commit,
                "source_version": source_version,
                "would_block_if_enabled": False,
                "would_enter": True,
                "no_sample_reason": None,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "v5_symbol": _v5_symbol(symbol),
                "strategy_candidate": strategy_candidate,
                "decision": "PAPER_READY",
                "recommended_mode": "paper",
                "horizon_hours": _optional_int(raw.get("horizon_hours")) or 24,
                "sample_count": _optional_int(raw.get("sample_count")),
                "complete_sample_count": _optional_int(raw.get("complete_sample_count")),
                "avg_net_bps": _optional_float(raw.get("avg_net_bps")),
                "p25_net_bps": _optional_float(raw.get("p25_net_bps")),
                "win_rate": _optional_float(raw.get("win_rate")),
                "cost_source_mix": raw.get("cost_source_mix") or raw.get("cost_source"),
                "cost_quality": raw.get("cost_quality") or raw.get("cost_source_quality"),
                "source_module": "expanded_universe",
                "template_family": "expanded_universe_paper",
                "candidate_id": raw.get("candidate_id") or strategy_id,
                "promotion_state": "PAPER_READY",
                "alpha_factory_score": None,
                "universe_type": "expanded_paper",
                "expanded_universe_maturity_state": "PAPER_READY",
                "cost_quality_score": _optional_float(raw.get("cost_quality_score")),
                "paper_ready_block_reasons": safe_json_dumps([]),
                "advisory_intent": "paper_shadow",
                "paper_days": _optional_int(raw.get("paper_days")) or 0,
                "entry_day_count": _optional_int(raw.get("entry_day_count")) or 0,
                "paper_pnl_observed_count": _optional_int(raw.get("paper_pnl_observed_count")) or 0,
                "slippage_coverage": _optional_float(raw.get("slippage_coverage")),
                "live_block_reasons": safe_json_dumps(live_block_reasons),
                "max_paper_notional_usdt": (
                    _optional_float(raw.get("max_paper_notional_usdt")) or 100.0
                ),
                "max_live_notional_usdt": 0.0,
            }
        )
    return rows


def _bottom_zone_probe_paper_opportunity_rows(
    bottom_zone_reversal_shadow: pl.DataFrame,
    *,
    git_commit: str,
    source_version: str,
) -> list[dict[str, Any]]:
    if bottom_zone_reversal_shadow.is_empty():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in bottom_zone_reversal_shadow.to_dicts():
        state = str(raw.get("bottom_zone_state") or "").strip().upper()
        if state != "BOTTOM_PROBE_ALLOWED":
            continue
        if _optional_bool(raw.get("would_probe_paper")) is False:
            continue
        symbol = normalize_symbol(raw.get("symbol"))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        generated_at = _advisory_generated_at(raw)
        live_block_reasons = [
            "bottom_zone_probe_paper_only_no_live",
            "bottom_zone_reversal_research_only",
            "quant_lab_live_command_not_allowed",
            "v5_local_live_not_controlled_by_quant_lab",
        ]
        rows.append(
            {
                "as_of_ts": _advisory_as_of_ts(raw),
                "generated_at": generated_at,
                "expires_at": _advisory_expires_at(raw, generated_at),
                "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
                "schema_version": STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION,
                "quant_lab_git_commit": git_commit,
                "source_version": source_version,
                "would_block_if_enabled": False,
                "would_enter": True,
                "no_sample_reason": None,
                "strategy_id": "BOTTOM_ZONE_PROBE_PAPER_V1",
                "symbol": symbol,
                "v5_symbol": _v5_symbol(symbol),
                "strategy_candidate": "v5.bottom_zone_probe_paper",
                "decision": "PAPER_READY",
                "recommended_mode": "paper",
                "horizon_hours": 24,
                "sample_count": None,
                "complete_sample_count": None,
                "avg_net_bps": _optional_float(raw.get("bounce_probability_4h")),
                "p25_net_bps": None,
                "win_rate": None,
                "cost_source_mix": "public_microstructure_proxy",
                "cost_quality": "bottom_zone_public_proxy",
                "source_module": "bottom_zone_reversal",
                "template_family": "bottom_zone_probe_paper",
                "candidate_id": f"BOTTOM_ZONE_PROBE_PAPER_V1:{symbol}",
                "promotion_state": "PAPER_REVIEW",
                "alpha_factory_score": _optional_float(raw.get("bottom_zone_score")),
                "universe_type": _v5_universe_type(symbol),
                "expanded_universe_maturity_state": "PAPER_READY",
                "cost_quality_score": None,
                "paper_ready_block_reasons": safe_json_dumps([]),
                "advisory_intent": "paper_shadow",
                "paper_days": 0,
                "entry_day_count": 0,
                "paper_pnl_observed_count": 0,
                "slippage_coverage": None,
                "live_block_reasons": safe_json_dumps(live_block_reasons),
                "max_paper_notional_usdt": 5.0,
                "max_live_notional_usdt": 0.0,
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return rows


def _bottom_zone_probe_paper_readiness_for_export(
    *,
    paper_runs: pl.DataFrame,
    paper_daily: pl.DataFrame,
    generated_at: datetime,
) -> pl.DataFrame:
    path = "reports/bottom_zone_probe_paper_readiness.csv"
    latest_daily = _latest_bottom_zone_paper_daily_rows(paper_daily)
    if not latest_daily:
        return _empty_csv_schema_frame(path)
    run_pnls = _bottom_zone_paper_pnls_by_key(paper_runs)
    rows: list[dict[str, Any]] = []
    for key, daily in sorted(latest_daily.items()):
        strategy_id, strategy_candidate, symbol = key
        values = run_pnls.get(key, [])
        paper_days = (
            _optional_int(
                daily.get("paper_days")
                or daily.get("paper_days_to_date")
                or daily.get("paper_pnl_day_count")
            )
            or 0
        )
        paper_entries = (
            _optional_int(
                daily.get("would_enter_count")
                or daily.get("cumulative_would_enter_count")
                or daily.get("entry_count")
                or daily.get("daily_would_enter_count")
            )
            or 0
        )
        avg_pnl = _optional_float(daily.get("avg_paper_pnl_bps"))
        if avg_pnl is None:
            avg_pnl = _float_mean(values)
        p25_pnl = _float_quantile(values, 0.25)
        reasons = _bottom_zone_paper_readiness_reasons(
            paper_days=paper_days,
            paper_entries=paper_entries,
            avg_pnl=avg_pnl,
            p25_pnl=p25_pnl,
        )
        rows.append(
            {
                "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
                "as_of_date": str(daily.get("as_of_date") or daily.get("paper_date") or ""),
                "strategy_id": strategy_id,
                "strategy_candidate": strategy_candidate,
                "symbol": symbol,
                "paper_days": paper_days,
                "paper_entries": paper_entries,
                "avg_pnl_bps": _round_float(avg_pnl),
                "p25_pnl_bps": _round_float(p25_pnl),
                "required_paper_days": 14,
                "required_paper_entries": 20,
                "required_avg_pnl_bps": 0.0,
                "required_p25_pnl_bps": -50.0,
                "readiness_status": "READY_FOR_REVIEW" if not reasons else "BLOCKED",
                "blocking_reasons": safe_json_dumps(reasons),
                "recommended_stage": "PAPER_REVIEW" if not reasons else "KEEP_PAPER_SHADOW",
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).select(CSV_SCHEMAS[path])


def _latest_bottom_zone_paper_daily_rows(
    paper_daily: pl.DataFrame,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if paper_daily.is_empty():
        return rows
    for row in paper_daily.to_dicts():
        if not _is_bottom_zone_paper_row(row):
            continue
        symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
        if symbol == "UNKNOWN":
            continue
        strategy_id = str(row.get("strategy_id") or "BOTTOM_ZONE_PROBE_PAPER_V1").strip()
        strategy_candidate = str(
            row.get("strategy_candidate") or "v5.bottom_zone_probe_paper"
        ).strip()
        key = (strategy_id, strategy_candidate, symbol)
        current = rows.get(key)
        if current is None or _paper_daily_row_time(row) >= _paper_daily_row_time(current):
            rows[key] = row
    return rows


def _bottom_zone_paper_pnls_by_key(
    paper_runs: pl.DataFrame,
) -> dict[tuple[str, str, str], list[float]]:
    values: dict[tuple[str, str, str], list[float]] = {}
    if paper_runs.is_empty():
        return values
    for row in paper_runs.to_dicts():
        if not _is_bottom_zone_paper_row(row):
            continue
        symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
        if symbol == "UNKNOWN":
            continue
        strategy_id = str(row.get("strategy_id") or "BOTTOM_ZONE_PROBE_PAPER_V1").strip()
        strategy_candidate = str(
            row.get("strategy_candidate")
            or row.get("source_strategy_candidate")
            or "v5.bottom_zone_probe_paper"
        ).strip()
        key = (strategy_id, strategy_candidate, symbol)
        row_values = _paper_run_pnl_values(row)
        if row_values:
            values.setdefault(key, []).extend(row_values)
    return values


def _paper_run_pnl_values(row: dict[str, Any]) -> list[float]:
    primary = _optional_float(row.get("paper_pnl_bps"))
    if primary is not None:
        return [primary]
    values: list[float] = []
    for key, value in row.items():
        if not str(key).startswith("paper_pnl_bps_"):
            continue
        parsed = _optional_float(value)
        if parsed is not None:
            values.append(parsed)
    return values


def _bottom_zone_paper_readiness_reasons(
    *,
    paper_days: int,
    paper_entries: int,
    avg_pnl: float | None,
    p25_pnl: float | None,
) -> list[str]:
    reasons: list[str] = []
    if paper_days < 14:
        reasons.append("paper_days_lt_14")
    if paper_entries < 20:
        reasons.append("paper_entries_lt_20")
    if avg_pnl is None or avg_pnl <= 0:
        reasons.append("avg_pnl_not_positive")
    if p25_pnl is None or p25_pnl <= -50:
        reasons.append("p25_not_above_minus_50")
    return reasons


def _is_bottom_zone_paper_row(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(field) or "").lower()
        for field in (
            "strategy_id",
            "strategy_candidate",
            "source_strategy_candidate",
            "experiment_name",
        )
    )
    return "bottom_zone_probe_paper" in text or "bottom_zone_reversal" in text


def _paper_daily_row_time(row: dict[str, Any]) -> datetime:
    for field in ("as_of_date", "paper_date", "created_at"):
        value = readers._coerce_timestamp(row.get(field))  # type: ignore[attr-defined]
        if value is not None:
            return value
    return datetime.min.replace(tzinfo=UTC)


def _round_float(value: float | None) -> float | str:
    return "" if value is None else round(float(value), 6)


def _v5_universe_type(symbol: str) -> str:
    live_symbols = {normalize_symbol(item) for item in DEFAULT_LIVE_UNIVERSE_SYMBOLS}
    return "v5_live_universe" if normalize_symbol(symbol) in live_symbols else "expanded_paper"


def _factor_strategy_bridge_review_opportunity_rows(
    bridge_candidates: pl.DataFrame,
    *,
    git_commit: str,
    source_version: str,
) -> list[dict[str, Any]]:
    if bridge_candidates.is_empty():
        return []
    rows: list[dict[str, Any]] = []
    for raw in bridge_candidates.to_dicts():
        action = str(raw.get("recommended_action") or "").strip().upper()
        eligible = str(raw.get("eligible_for_alpha_factory") or "").strip().lower()
        candidate_id = str(raw.get("bridge_candidate_id") or "").strip()
        if action != "REVIEW_FOR_ALPHA_FACTORY_STRATEGY":
            continue
        if eligible != "strategy_review_pending":
            continue
        if not candidate_id.startswith(("v5.factor_bridge.", "v5.fast_microstructure_bridge.")):
            continue
        symbol = normalize_symbol(raw.get("symbol")) or "ALL"
        generated_at = _advisory_generated_at(raw)
        horizon_hours = _optional_int(raw.get("horizon_hours"))
        blocking_reasons = sorted(
            {
                *_json_listish(raw.get("blocking_reasons")),
                "strategy_review_pending",
                "shadow_review_only",
                "not_live_validated",
                "quant_lab_live_command_not_allowed",
                "v5_local_live_not_controlled_by_quant_lab",
            }
        )
        is_fast_bridge = candidate_id.startswith("v5.fast_microstructure_bridge.")
        rows.append(
            {
                "as_of_ts": _advisory_as_of_ts(raw),
                "generated_at": generated_at,
                "expires_at": _advisory_expires_at(raw, generated_at),
                "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
                "schema_version": STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION,
                "quant_lab_git_commit": git_commit,
                "source_version": source_version,
                "would_block_if_enabled": False,
                "would_enter": False,
                "no_sample_reason": "strategy_review_pending_shadow_only",
                "strategy_id": _advisory_strategy_id(candidate_id, symbol),
                "symbol": symbol,
                "v5_symbol": _v5_symbol(symbol),
                "strategy_candidate": candidate_id,
                "decision": "KEEP_SHADOW",
                "recommended_mode": "shadow",
                "horizon_hours": horizon_hours,
                "sample_count": _optional_int(raw.get("forward_sample_count")),
                "complete_sample_count": _optional_int(raw.get("forward_sample_count")),
                "avg_net_bps": _optional_float(raw.get("forward_cost_adjusted_score")),
                "p25_net_bps": None,
                "win_rate": None,
                "cost_source_mix": "forward_validation_cost_adjusted",
                "cost_quality": "strategy_review_pending_cost_validation",
                "source_module": "factor_strategy_bridge",
                "template_family": (
                    "fast_microstructure_bridge" if is_fast_bridge else "factor_strategy_bridge"
                ),
                "candidate_id": candidate_id,
                "promotion_state": "STRATEGY_REVIEW_PENDING",
                "alpha_factory_score": None,
                "universe_type": "v5_live_universe",
                "expanded_universe_maturity_state": None,
                "cost_quality_score": None,
                "paper_ready_block_reasons": safe_json_dumps(blocking_reasons),
                "advisory_intent": _export_advisory_intent("shadow"),
                "paper_days": 0,
                "entry_day_count": 0,
                "paper_pnl_observed_count": 0,
                "slippage_coverage": None,
                "live_block_reasons": safe_json_dumps(blocking_reasons),
                "max_paper_notional_usdt": 0.0,
                "max_live_notional_usdt": 0.0,
                "live_order_effect": "read_only_no_live_order",
            }
        )
    return rows


def _entry_quality_opportunity_rows(entry_quality_advisory: pl.DataFrame) -> list[dict[str, Any]]:
    if entry_quality_advisory.is_empty():
        return []
    rows: list[dict[str, Any]] = []
    git_commit = _git_commit()
    source_version = _source_version("entry_quality_advisory", git_commit)
    latest_as_of_date = _latest_as_of_date(entry_quality_advisory.to_dicts())
    for row in entry_quality_advisory.to_dicts():
        if latest_as_of_date and str(row.get("as_of_date") or "") != latest_as_of_date:
            continue
        candidate = str(row.get("strategy_candidate") or "").strip()
        mode = _entry_quality_recommended_mode(candidate, row.get("recommended_mode"))
        if mode == "paper":
            decision = "PAPER_READY"
        elif mode == "shadow":
            decision = "KEEP_SHADOW"
        else:
            decision = "RESEARCH_ONLY"
        symbol = normalize_symbol(row.get("symbol")) if row.get("symbol") != "ALL" else "ALL"
        live_block_reasons = _entry_quality_live_block_reasons(row, mode)
        generated_at = _advisory_generated_at(row)
        rows.append(
            {
                "as_of_ts": _entry_quality_as_of_ts(row),
                "generated_at": generated_at,
                "expires_at": _advisory_expires_at(row, generated_at),
                "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
                "schema_version": str(
                    row.get("schema_version") or STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION
                ),
                "quant_lab_git_commit": _observable_text(row.get("quant_lab_git_commit"))
                or git_commit,
                "source_version": _observable_text(row.get("source_version")) or source_version,
                "would_block_if_enabled": _optional_bool(row.get("would_block_if_enabled")) is True,
                "would_enter": _optional_bool(row.get("would_enter")) is True,
                "no_sample_reason": str(row.get("no_sample_reason") or "").strip()
                or _entry_quality_export_no_sample_reason(row, mode),
                "strategy_id": _advisory_strategy_id(
                    str(row.get("strategy_candidate") or ""),
                    symbol or "ALL",
                ),
                "symbol": symbol or "ALL",
                "v5_symbol": _v5_symbol(symbol) if symbol and symbol != "ALL" else "ALL",
                "strategy_candidate": row.get("strategy_candidate"),
                "decision": decision,
                "recommended_mode": mode,
                "horizon_hours": None,
                "sample_count": _optional_int(row.get("sample_count")),
                "complete_sample_count": _optional_int(row.get("sample_count")),
                "avg_net_bps": _optional_float(row.get("avg_net_bps")),
                "p25_net_bps": None,
                "win_rate": _optional_float(row.get("win_rate")),
                "cost_source_mix": None,
                "cost_quality": "entry_quality_research",
                "source_module": "entry_quality",
                "template_family": None,
                "candidate_id": None,
                "promotion_state": None,
                "alpha_factory_score": None,
                "universe_type": None,
                "expanded_universe_maturity_state": None,
                "cost_quality_score": None,
                "paper_ready_block_reasons": "[]",
                "advisory_intent": _export_advisory_intent(mode),
                "paper_days": 0,
                "entry_day_count": 0,
                "paper_pnl_observed_count": 0,
                "slippage_coverage": None,
                "live_block_reasons": safe_json_dumps(live_block_reasons),
                "max_paper_notional_usdt": _advisory_max_paper_notional(mode),
                "max_live_notional_usdt": 0.0,
            }
        )
    return rows


def _entry_quality_recommended_mode(candidate: str, value: Any) -> str:
    if candidate == "v5.entry_quality_missed_low_audit":
        return "research"
    mode = str(value or "").strip().lower()
    if mode == "audit":
        return "research"
    if mode in {"paper", "shadow", "research"}:
        return mode
    return "shadow"


def _regime_router_opportunity_rows(regime_advisory: pl.DataFrame) -> list[dict[str, Any]]:
    if regime_advisory.is_empty():
        return []
    rows: list[dict[str, Any]] = []
    git_commit = _git_commit()
    source_version = _source_version("regime_router", git_commit)
    latest_as_of_date = _latest_as_of_date(regime_advisory.to_dicts())
    for row in regime_advisory.to_dicts():
        if latest_as_of_date and str(row.get("as_of_date") or "") != latest_as_of_date:
            continue
        current_regime = str(row.get("current_regime") or "UNKNOWN").strip().upper()
        mode = str(row.get("recommended_mode") or "research").strip().lower()
        if mode not in {"research", "shadow", "paper"}:
            mode = "research"
        generated_at = _advisory_generated_at(row)
        base_reasons = set(_json_listish(row.get("live_block_reasons")))
        base_reasons.update(
            {
                "regime_router_read_only",
                "not_live_validated",
                "no_live_small_from_regime_router",
            }
        )
        for candidate in _json_listish(row.get("allowed_strategy_candidates")):
            decision = "PAPER_READY" if mode == "paper" else "KEEP_SHADOW"
            if mode == "research":
                decision = "RESEARCH_ONLY"
            rows.append(
                _regime_router_opportunity_row(
                    row=row,
                    candidate=candidate,
                    decision=decision,
                    recommended_mode=mode,
                    current_regime=current_regime,
                    generated_at=generated_at,
                    git_commit=git_commit,
                    source_version=source_version,
                    reasons=sorted(base_reasons),
                )
            )
        for candidate in _json_listish(row.get("blocked_strategy_candidates")):
            rows.append(
                _regime_router_opportunity_row(
                    row=row,
                    candidate=candidate,
                    decision="KILL",
                    recommended_mode="none",
                    current_regime=current_regime,
                    generated_at=generated_at,
                    git_commit=git_commit,
                    source_version=source_version,
                    reasons=sorted({*base_reasons, "blocked_in_current_regime"}),
                )
            )
    return rows


def _regime_router_opportunity_row(
    *,
    row: dict[str, Any],
    candidate: str,
    decision: str,
    recommended_mode: str,
    current_regime: str,
    generated_at: datetime,
    git_commit: str,
    source_version: str,
    reasons: list[str],
) -> dict[str, Any]:
    strategy_candidate = f"regime_router:{candidate}"
    return {
        "as_of_ts": _entry_quality_as_of_ts(row),
        "generated_at": generated_at,
        "expires_at": _advisory_expires_at(row, generated_at),
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
        "schema_version": STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION,
        "quant_lab_git_commit": git_commit,
        "source_version": source_version,
        "would_block_if_enabled": decision == "KILL",
        "would_enter": decision == "PAPER_READY",
        "no_sample_reason": (
            "blocked_in_current_regime"
            if decision == "KILL"
            else f"regime_router_{current_regime.lower()}_{recommended_mode}"
        ),
        "strategy_id": _advisory_strategy_id(strategy_candidate, "ALL"),
        "symbol": "ALL",
        "v5_symbol": "ALL",
        "strategy_candidate": strategy_candidate,
        "decision": decision,
        "recommended_mode": recommended_mode,
        "horizon_hours": None,
        "sample_count": None,
        "complete_sample_count": None,
        "avg_net_bps": None,
        "p25_net_bps": None,
        "win_rate": None,
        "cost_source_mix": None,
        "cost_quality": "regime_router",
        "source_module": "regime_router",
        "template_family": None,
        "candidate_id": None,
        "promotion_state": None,
        "alpha_factory_score": None,
        "universe_type": None,
        "expanded_universe_maturity_state": None,
        "cost_quality_score": None,
        "paper_ready_block_reasons": "[]",
        "advisory_intent": _export_advisory_intent(recommended_mode),
        "paper_days": 0,
        "entry_day_count": 0,
        "paper_pnl_observed_count": 0,
        "slippage_coverage": None,
        "live_block_reasons": safe_json_dumps(reasons),
        "max_paper_notional_usdt": _advisory_max_paper_notional(recommended_mode),
        "max_live_notional_usdt": 0.0,
    }


def _risk_on_multi_buy_opportunity_rows(risk_on_shadow: pl.DataFrame) -> list[dict[str, Any]]:
    if risk_on_shadow.is_empty():
        return []
    rows: list[dict[str, Any]] = []
    git_commit = _git_commit()
    source_version = _source_version("risk_on_multi_buy_shadow", git_commit)
    for row in risk_on_shadow.to_dicts():
        candidate = str(row.get("strategy_candidate") or "").strip()
        if candidate not in {
            "v5.risk_on_multi_buy_top1_shadow",
            "v5.risk_on_multi_buy_top2_shadow",
            "v5.risk_on_multi_buy_top3_shadow",
        }:
            continue
        generated_at = _advisory_generated_at(row)
        live_block_reasons = [
            "risk_on_multi_buy_shadow_only",
            "not_live_validated",
            "shadow_only",
            "quant_lab_live_command_not_allowed",
            "v5_local_live_not_controlled_by_quant_lab",
        ]
        rows.append(
            {
                "as_of_ts": _entry_quality_as_of_ts(row),
                "generated_at": generated_at,
                "expires_at": _advisory_expires_at(row, generated_at),
                "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
                "schema_version": STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION,
                "quant_lab_git_commit": git_commit,
                "source_version": source_version,
                "would_block_if_enabled": False,
                "would_enter": bool(row.get("would_buy")),
                "no_sample_reason": "risk_on_shadow_collect_more_samples",
                "strategy_id": _advisory_strategy_id(candidate, "MULTI"),
                "symbol": "MULTI",
                "v5_symbol": "MULTI",
                "strategy_candidate": candidate,
                "decision": "KEEP_SHADOW",
                "recommended_mode": "shadow",
                "horizon_hours": 24,
                "sample_count": _optional_int(row.get("selected_count")),
                "complete_sample_count": _optional_int(row.get("selected_count")),
                "avg_net_bps": _optional_float(row.get("avg_portfolio_net_bps")),
                "p25_net_bps": None,
                "win_rate": None,
                "cost_source_mix": '{"conservative_shadow_cost":1}',
                "cost_quality": "shadow_research",
                "source_module": "risk_on_multi_buy_shadow",
                "template_family": "risk_on_multi_buy_shadow",
                "candidate_id": str(row.get("run_id") or ""),
                "promotion_state": "SHADOW",
                "alpha_factory_score": None,
                "universe_type": "v5_major_spot",
                "expanded_universe_maturity_state": None,
                "cost_quality_score": None,
                "paper_ready_block_reasons": safe_json_dumps(live_block_reasons),
                "advisory_intent": _export_advisory_intent("shadow"),
                "paper_days": 0,
                "entry_day_count": 0,
                "paper_pnl_observed_count": 0,
                "slippage_coverage": None,
                "live_block_reasons": safe_json_dumps(live_block_reasons),
                "max_paper_notional_usdt": 0.0,
                "max_live_notional_usdt": 0.0,
            }
        )
    return rows


def _entry_quality_live_block_reasons(row: dict[str, Any], recommended_mode: str) -> list[str]:
    reasons = set(_json_listish(row.get("advisory_reasons")))
    reasons.add("shadow_only")
    reasons.add("not_live_validated")
    reasons.add("entry_quality_advisory_only")
    if recommended_mode != "paper":
        reasons.add("not_paper_candidate")
    return sorted(reason for reason in reasons if reason)


RISK_ON_MULTI_BUY_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"}
RISK_ON_MULTI_BUY_COST_BPS = 30.0
RISK_ON_MULTI_BUY_REGIMES = {"ALT_IMPULSE", "TREND_UP"}


def _v5_missed_opportunity_audit_for_export(
    *,
    candidate_events: pl.DataFrame,
    v5_trades: pl.DataFrame,
    market_bars: pl.DataFrame,
    opportunity_advisory: pl.DataFrame,
    market_regime: pl.DataFrame,
) -> pl.DataFrame:
    path = "reports/missed_opportunity_audit.csv"
    if candidate_events.is_empty():
        return _empty_csv_schema_frame(path)
    market_by_symbol = _market_close_rows_by_symbol(market_bars)
    advisory_by_symbol = _latest_advisory_by_symbol(opportunity_advisory)
    trade_opened = _actual_open_trade_symbols_by_run(v5_trades)
    regime_by_date = _market_regime_context_by_date(market_regime)
    latest_regime_context = _latest_market_regime_context(market_regime)
    generated_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for candidate in candidate_events.to_dicts():
        symbol = normalize_symbol(candidate.get("symbol"))
        ts = readers._coerce_timestamp(candidate.get("ts_utc") or candidate.get("ts"))
        if not symbol or symbol == "UNKNOWN" or ts is None:
            continue
        regime_context = _candidate_market_regime_context(
            candidate,
            ts,
            regime_by_date,
            latest_regime_context,
            market_by_symbol=market_by_symbol,
        )
        run_id = str(candidate.get("run_id") or "").strip()
        entry_price = _candidate_entry_price(candidate, market_by_symbol.get(symbol, []), ts)
        future = {
            horizon: _future_net_bps_after_shadow_cost(
                market_by_symbol.get(symbol, []),
                ts.astimezone(UTC),
                entry_price,
                horizon,
            )
            for horizon in (4, 8, 24)
        }
        advisory = _matching_quant_lab_advisory(candidate, symbol, advisory_by_symbol)
        quant_lab_mode = str(advisory.get("recommended_mode") if advisory else "research")
        quant_lab_decision = str(advisory.get("decision") if advisory else "RESEARCH_ONLY")
        quant_lab_would_block = quant_lab_mode not in {"live", "live_small"}
        actual_opened = (run_id, symbol) in trade_opened
        strong_candidate = _candidate_is_strong_buy(candidate)
        outcome = _missed_opportunity_outcome(
            actual_trade_opened=actual_opened,
            quant_lab_would_block_live=quant_lab_would_block,
            strong_candidate=strong_candidate,
            future_net_bps=future,
        )
        rows.append(
            {
                "generated_at": generated_at,
                "schema_version": "v5_missed_opportunity_audit.v0.1",
                "run_id": run_id,
                "ts_utc": ts.astimezone(UTC).isoformat(),
                "symbol": symbol,
                "regime_state": candidate.get("regime_state"),
                "current_regime": regime_context.get("current_regime"),
                "broad_market_positive_count": _optional_int(
                    candidate.get("broad_market_positive_count")
                )
                or regime_context.get("broad_market_positive_count"),
                "final_score": _optional_float(candidate.get("final_score")),
                "alpha6_side": candidate.get("alpha6_side"),
                "expected_edge_bps": _optional_float(candidate.get("expected_edge_bps")),
                "required_edge_bps": _optional_float(candidate.get("required_edge_bps")),
                "v5_final_decision": candidate.get("final_decision"),
                "v5_block_reason": candidate.get("block_reason"),
                "actual_trade_opened": actual_opened,
                "quant_lab_would_block_live": quant_lab_would_block,
                "quant_lab_recommended_mode": quant_lab_mode,
                "quant_lab_decision": quant_lab_decision,
                "would_have_been_missed_by_quant_lab": outcome
                == "quant_lab_would_have_missed_profit",
                "future_4h_net_bps": future[4],
                "future_8h_net_bps": future[8],
                "future_24h_net_bps": future[24],
                "outcome_if_blocked": outcome,
                "source": "quant_lab_missed_opportunity_audit",
            }
        )
    if not rows:
        return _empty_csv_schema_frame(path)
    return pl.DataFrame(rows, infer_schema_length=None).select(CSV_SCHEMAS[path])


BNB_MISSED_OPPORTUNITY_START_TS = datetime(2026, 5, 30, tzinfo=UTC)
FINAL_SCORE_ALPHA6_CONFLICT_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"}


def _bnb_missed_opportunity_samples_for_export(
    *,
    candidate_events: pl.DataFrame,
    market_bars: pl.DataFrame,
) -> pl.DataFrame:
    path = "reports/bnb_missed_opportunity_samples.csv"
    if candidate_events.is_empty():
        return _empty_csv_schema_frame(path)
    market_by_symbol = _market_close_rows_by_symbol(market_bars)
    bnb_market = market_by_symbol.get("BNB-USDT", [])
    rows: list[dict[str, Any]] = []
    for candidate in candidate_events.to_dicts():
        symbol = normalize_symbol(candidate.get("symbol"))
        if symbol != "BNB-USDT":
            continue
        ts = readers._coerce_timestamp(candidate.get("ts_utc") or candidate.get("ts"))
        if ts is None or ts.astimezone(UTC) < BNB_MISSED_OPPORTUNITY_START_TS:
            continue
        side = str(candidate.get("alpha6_side") or "").strip().lower()
        if side != "buy":
            continue
        decision = str(candidate.get("final_decision") or "").strip().lower()
        if decision not in {"no_order", "blocked"}:
            continue
        expected = _optional_float(candidate.get("expected_edge_bps"))
        required = _optional_float(candidate.get("required_edge_bps"))
        if expected is None or required is None or expected <= required:
            continue
        entry_close = _candidate_entry_price(candidate, bnb_market, ts)
        future = {
            horizon: _future_net_bps_after_shadow_cost(
                bnb_market,
                ts.astimezone(UTC),
                entry_close,
                horizon,
            )
            for horizon in (4, 8, 12, 24)
        }
        observed = [value for value in future.values() if value is not None]
        best_future = (
            max(
                ((horizon, value) for horizon, value in future.items() if value is not None),
                key=lambda item: item[1],
            )
            if observed
            else (None, None)
        )
        rows.append(
            {
                "run_id": str(candidate.get("run_id") or "").strip(),
                "ts_utc": ts.astimezone(UTC).isoformat(),
                "entry_close": entry_close,
                "alpha6_score": _optional_float(candidate.get("alpha6_score")),
                "f3": _candidate_factor_value(
                    candidate,
                    "f3",
                    "f3_vol_adj_ret",
                    "f3_score",
                    "f3_dominant",
                    "f3_dominant_score",
                ),
                "f4": _candidate_factor_value(
                    candidate,
                    "f4",
                    "f4_volume_expansion",
                    "f4_score",
                ),
                "f5": _candidate_factor_value(
                    candidate,
                    "f5",
                    "f5_rsi_trend_confirm",
                    "f5_score",
                ),
                "final_score": _optional_float(candidate.get("final_score")),
                "final_decision": candidate.get("final_decision"),
                "future_4h_net_bps": future[4],
                "future_8h_net_bps": future[8],
                "future_12h_net_bps": future[12],
                "future_24h_net_bps": future[24],
                "missed_profit_flag": bool(best_future[1] is not None and best_future[1] >= 50.0),
            }
        )
    if not rows:
        return _empty_csv_schema_frame(path)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .select(CSV_SCHEMAS[path])
        .sort(["ts_utc", "run_id"])
    )


def _candidate_factor_value(candidate: dict[str, Any], *field_names: str) -> float | None:
    for field in field_names:
        value = _optional_float(candidate.get(field))
        if value is not None:
            return value
    return None


def _post_impulse_overextension_shadow_for_export(
    *,
    candidate_events: pl.DataFrame,
    market_bars: pl.DataFrame,
) -> pl.DataFrame:
    path = "reports/post_impulse_overextension_shadow.csv"
    if candidate_events.is_empty():
        return _empty_csv_schema_frame(path)
    market_by_symbol = _market_close_rows_by_symbol(market_bars)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_events.to_dicts():
        side = str(candidate.get("alpha6_side") or "").strip().lower()
        if side and side != "buy":
            continue
        alpha6_score = _optional_float(candidate.get("alpha6_score"))
        f3 = _candidate_factor_value(candidate, "f3_vol_adj_ret", "f3", "f3_score")
        f4 = _candidate_factor_value(candidate, "f4_volume_expansion", "f4", "f4_score")
        signal_reasons = []
        if alpha6_score is not None and alpha6_score >= 0.75:
            signal_reasons.append("alpha6_score_ge_0_75")
        if f3 is not None and f3 >= 10.0:
            signal_reasons.append("f3_ge_10")
        if f4 is not None and f4 >= 1.0:
            signal_reasons.append("f4_ge_1")
        if not signal_reasons:
            continue
        missing_reasons = []
        if not side:
            missing_reasons.append("alpha6_side_missing")
        if alpha6_score is None:
            missing_reasons.append("alpha6_score_missing")
        if f3 is None:
            missing_reasons.append("f3_missing")
        if f4 is None:
            missing_reasons.append("f4_missing")
        symbol = normalize_symbol(candidate.get("symbol"))
        ts = readers._coerce_timestamp(candidate.get("ts_utc") or candidate.get("ts"))
        if not symbol or symbol == "UNKNOWN" or ts is None:
            continue
        market_rows = market_by_symbol.get(symbol, [])
        if not market_rows:
            missing_reasons.append("market_bars_missing")
        ts_utc = ts.astimezone(UTC)
        entry_px = _candidate_entry_price(candidate, market_rows, ts_utc)
        return_24h = _historical_return_bps(market_rows, ts_utc, 24)
        return_48h = _historical_return_bps(market_rows, ts_utc, 48)
        overextended_24h = return_24h is not None and return_24h >= 200.0
        overextended_48h = return_48h is not None and return_48h >= 500.0
        why_not_triggered = []
        if return_24h is None:
            why_not_triggered.append("return_24h_not_observable")
            missing_reasons.append("return_24h_missing")
        elif not overextended_24h:
            why_not_triggered.append("return_24h_lt_200bps")
        if return_48h is None:
            missing_reasons.append("return_48h_missing")
        shadow_cost_bps = _candidate_shadow_cost_bps(candidate)
        future = {
            horizon: _future_net_bps_after_shadow_cost(
                market_rows,
                ts_utc,
                entry_px,
                horizon,
                shadow_cost_bps=shadow_cost_bps,
            )
            for horizon in (4, 8, 12)
        }
        observed = [value for value in future.values() if value is not None]
        reasons = []
        if overextended_24h:
            reasons.append("return_24h_ge_200bps")
        if overextended_48h:
            reasons.append("return_48h_ge_500bps")
        is_overextended = bool(reasons)
        rows.append(
            {
                "run_id": str(candidate.get("run_id") or "").strip(),
                "ts_utc": ts_utc.isoformat(),
                "symbol": symbol,
                "alpha6_score": alpha6_score,
                "alpha6_side": side,
                "f3_vol_adj_ret": f3,
                "f4_volume_expansion": f4,
                "f5_rsi_trend_confirm": _candidate_factor_value(
                    candidate,
                    "f5_rsi_trend_confirm",
                    "f5",
                    "f5_score",
                ),
                "entry_px": entry_px,
                "return_24h_bps": return_24h,
                "return_48h_bps": return_48h,
                "overextension_reason": ";".join(reasons),
                "why_not_triggered": ";".join(why_not_triggered if not is_overextended else []),
                "missing_field_reason": ";".join(sorted(set(missing_reasons))),
                "future_4h_net_bps": future[4],
                "future_8h_net_bps": future[8],
                "future_12h_net_bps": future[12],
                "max_future_net_bps": max(observed) if observed else None,
                "worst_future_net_bps": min(observed) if observed else None,
                "late_failure_flag": bool(is_overextended and observed and min(observed) <= -50.0),
                "response_action": "shadow_tracking" if is_overextended else "diagnostic_only",
                "live_order_effect": "read_only_no_live_order",
            }
        )
    if not rows:
        return _empty_csv_schema_frame(path)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .select(CSV_SCHEMAS[path])
        .sort(["ts_utc", "symbol", "run_id"])
    )


def _late_breakout_failure_shadow_for_export(overextension: pl.DataFrame) -> pl.DataFrame:
    path = "reports/late_breakout_failure_shadow.csv"
    if overextension.is_empty():
        return _empty_csv_schema_frame(path)
    rows: list[dict[str, Any]] = []
    for row in overextension.to_dicts():
        if not str(row.get("overextension_reason") or "").strip():
            continue
        future = {
            4: _optional_float(row.get("future_4h_net_bps")),
            8: _optional_float(row.get("future_8h_net_bps")),
            12: _optional_float(row.get("future_12h_net_bps")),
        }
        observed = [(horizon, value) for horizon, value in future.items() if value is not None]
        if not observed:
            continue
        failure_horizon, failure_value = min(observed, key=lambda item: item[1])
        if failure_value > -50.0:
            continue
        rows.append(
            {
                "run_id": row.get("run_id"),
                "ts_utc": row.get("ts_utc"),
                "symbol": row.get("symbol"),
                "alpha6_score": row.get("alpha6_score"),
                "alpha6_side": row.get("alpha6_side"),
                "f3_vol_adj_ret": row.get("f3_vol_adj_ret"),
                "f4_volume_expansion": row.get("f4_volume_expansion"),
                "f5_rsi_trend_confirm": row.get("f5_rsi_trend_confirm"),
                "entry_px": row.get("entry_px"),
                "return_24h_bps": row.get("return_24h_bps"),
                "return_48h_bps": row.get("return_48h_bps"),
                "future_4h_net_bps": row.get("future_4h_net_bps"),
                "future_8h_net_bps": row.get("future_8h_net_bps"),
                "future_12h_net_bps": row.get("future_12h_net_bps"),
                "failure_horizon_hours": failure_horizon,
                "failure_net_bps": failure_value,
                "diagnosis": "late_breakout_failure_after_overextension",
                "response_action": "shadow_tracking",
                "live_order_effect": "read_only_no_live_order",
            }
        )
    if not rows:
        return _empty_csv_schema_frame(path)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .select(CSV_SCHEMAS[path])
        .sort(["ts_utc", "symbol", "run_id"])
    )


def _final_score_vs_alpha6_conflict_for_export(
    *,
    candidate_events: pl.DataFrame,
    market_bars: pl.DataFrame,
    negative_expectancy: pl.DataFrame | None = None,
) -> pl.DataFrame:
    path = "reports/final_score_vs_alpha6_conflict.csv"
    if candidate_events.is_empty():
        return _empty_csv_schema_frame(path)
    market_by_symbol = _market_close_rows_by_symbol(market_bars)
    negative_expectancy_by_symbol = _negative_expectancy_stats_by_symbol(
        negative_expectancy if negative_expectancy is not None else pl.DataFrame()
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidate_events.to_dicts():
        symbol = normalize_symbol(candidate.get("symbol"))
        if symbol not in FINAL_SCORE_ALPHA6_CONFLICT_SYMBOLS:
            continue
        if not _is_final_score_alpha6_conflict_candidate(candidate):
            continue
        ts = readers._coerce_timestamp(candidate.get("ts_utc") or candidate.get("ts"))
        if ts is None:
            continue
        symbol_market = market_by_symbol.get(symbol, [])
        entry_close = _candidate_entry_price(candidate, symbol_market, ts)
        shadow_cost_bps = _candidate_shadow_cost_bps(candidate)
        future = {
            horizon: _future_net_bps_after_shadow_cost(
                symbol_market,
                ts.astimezone(UTC),
                entry_close,
                horizon,
                shadow_cost_bps=shadow_cost_bps,
            )
            for horizon in (4, 8, 12, 24)
        }
        label_statuses = {
            horizon: ("complete" if _optional_float(future[horizon]) is not None else "pending")
            for horizon in (4, 8, 12, 24)
        }
        any_label_complete = any(status == "complete" for status in label_statuses.values())
        all_labels_complete = all(status == "complete" for status in label_statuses.values())
        if all_labels_complete:
            label_status = "complete"
        elif any_label_complete:
            label_status = "partial_complete"
        else:
            label_status = "pending"
        observed = [value for value in future.values() if value is not None]
        best_future = (
            max(
                ((horizon, value) for horizon, value in future.items() if value is not None),
                key=lambda item: item[1],
            )
            if observed
            else (None, None)
        )
        material_profit = bool(best_future[1] is not None and best_future[1] >= 50.0)
        rows.append(
            {
                "run_id": str(candidate.get("run_id") or "").strip(),
                "ts_utc": ts.astimezone(UTC).isoformat(),
                "symbol": symbol,
                "final_score": _optional_float(candidate.get("final_score")),
                "alpha6_score": _optional_float(candidate.get("alpha6_score")),
                "alpha6_side": candidate.get("alpha6_side"),
                "f3_vol_adj_ret": _candidate_factor_value(
                    candidate,
                    "f3_vol_adj_ret",
                    "f3",
                    "f3_score",
                ),
                "f4_volume_expansion": _candidate_factor_value(
                    candidate,
                    "f4_volume_expansion",
                    "f4",
                    "f4_score",
                ),
                "f5_rsi_trend_confirm": _candidate_factor_value(
                    candidate,
                    "f5_rsi_trend_confirm",
                    "f5",
                    "f5_score",
                ),
                "expected_edge_bps": _optional_float(candidate.get("expected_edge_bps")),
                "required_edge_bps": _optional_float(candidate.get("required_edge_bps")),
                "cost_gate_verified": _optional_bool(candidate.get("cost_gate_verified")),
                "final_decision": candidate.get("final_decision"),
                "block_reason": candidate.get("block_reason"),
                "no_signal_reason": candidate.get("no_signal_reason"),
                "negative_expectancy_net_bps": _negative_expectancy_value(
                    negative_expectancy_by_symbol,
                    symbol,
                    "negexp_net_expectancy_bps",
                    "net_expectancy_bps",
                ),
                "negative_expectancy_fast_fail_net_bps": _negative_expectancy_value(
                    negative_expectancy_by_symbol,
                    symbol,
                    "negexp_fast_fail_net_expectancy_bps",
                    "fast_fail_net_expectancy_bps",
                ),
                "future_4h_net_bps": future[4],
                "future_8h_net_bps": future[8],
                "future_12h_net_bps": future[12],
                "future_24h_net_bps": future[24],
                "max_future_net_bps": best_future[1],
                "best_future_horizon_hours": best_future[0],
                "material_profit_flag": material_profit,
                "label_4h_status": label_statuses[4],
                "label_8h_status": label_statuses[8],
                "label_12h_status": label_statuses[12],
                "label_24h_status": label_statuses[24],
                "any_label_complete": any_label_complete,
                "all_labels_complete": all_labels_complete,
                "label_status": label_status,
                "missed_profit_flag": material_profit,
            }
        )
    if not rows:
        return _empty_csv_schema_frame(path)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .select(CSV_SCHEMAS[path])
        .sort(["ts_utc", "symbol", "run_id"])
    )


def _is_final_score_alpha6_conflict_candidate(candidate: dict[str, Any]) -> bool:
    side = str(candidate.get("alpha6_side") or "").strip().lower()
    if side != "buy":
        return False
    alpha6_score = _optional_float(candidate.get("alpha6_score"))
    if alpha6_score is None or alpha6_score < 0.9:
        return False
    expected = _optional_float(candidate.get("expected_edge_bps"))
    required = _optional_float(candidate.get("required_edge_bps"))
    if expected is None or required is None or expected <= required:
        return False
    if _optional_bool(candidate.get("cost_gate_verified")) is not True:
        return False
    final_score = _optional_float(candidate.get("final_score"))
    final_decision = str(candidate.get("final_decision") or "").strip().lower()
    return (final_score is not None and final_score < 0.0) or final_decision in {
        "no_order",
        "blocked",
    }


def _negative_expectancy_stats_by_symbol(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.is_empty():
        return {}
    fields = (
        "negexp_closed_cycles",
        "negexp_net_expectancy_bps",
        "adjusted_entry_expectancy_bps",
        "negexp_fast_fail_net_expectancy_bps",
        "adjusted_fast_fail_net_expectancy_bps",
        "entry_bad_cycles",
        "exit_bad_cycles",
        "min_hold_violation_cycles",
        "premature_soft_exit_count",
        "excluded_from_fast_fail_count",
        "diagnosis",
    )
    out: dict[str, dict[str, Any]] = {}
    for row in frame.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        payload: dict[str, Any] = {}
        for field in fields:
            value = row.get(field)
            if value is None or value == "":
                continue
            payload[field] = value
        if payload:
            out[symbol] = payload
    return out


def _negative_expectancy_value(
    stats_by_symbol: dict[str, dict[str, Any]],
    symbol: str,
    *fields: str,
) -> Any:
    payload = stats_by_symbol.get(symbol, {})
    for field in fields:
        value = payload.get(field)
        if value is not None and value != "":
            return value
    return None


def _candidate_shadow_cost_bps(candidate: dict[str, Any]) -> float:
    for field in (
        "selected_entry_gate_cost_bps",
        "roundtrip_all_in_cost_bps",
        "selected_total_cost_bps",
        "cost_bps",
    ):
        value = _optional_float(candidate.get(field))
        if value is not None:
            return max(value, 0.0)
    return RISK_ON_MULTI_BUY_COST_BPS


def _conflict_avg_optional(values: Iterable[Any]) -> float | None:
    observed = [_optional_float(value) for value in values]
    observed = [value for value in observed if value is not None]
    if not observed:
        return None
    return sum(observed) / len(observed)


def _conflict_display_number(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "not_observable"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _final_score_vs_alpha6_conflict_summary_md(conflicts: pl.DataFrame) -> str:
    rows = conflicts.to_dicts() if not conflicts.is_empty() else []
    conflict_count = len(rows)
    avg_4h = _conflict_avg_optional(row.get("future_4h_net_bps") for row in rows)
    avg_8h = _conflict_avg_optional(row.get("future_8h_net_bps") for row in rows)
    avg_24h = _conflict_avg_optional(row.get("future_24h_net_bps") for row in rows)
    symbol_counts: dict[str, int] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "UNKNOWN")
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    missed_profit_count = sum(
        1 for row in rows if _optional_bool(row.get("missed_profit_flag")) is True
    )
    blocked_final_decision_count = sum(
        1 for row in rows if str(row.get("final_decision") or "").strip().lower() == "blocked"
    )
    negative_expectancy_block_count = sum(
        1
        for row in rows
        if "negative_expectancy"
        in " ".join(
            [
                str(row.get("block_reason") or ""),
                str(row.get("no_signal_reason") or ""),
                str(row.get("negative_expectancy_net_bps") or ""),
                str(row.get("negative_expectancy_fast_fail_net_bps") or ""),
            ]
        ).lower()
    )
    partial_complete_count = sum(
        1 for row in rows if str(row.get("label_status") or "").strip() == "partial_complete"
    )
    if conflict_count == 0:
        recommendation = "no_conflict_observed"
    elif missed_profit_count > 0:
        recommendation = "review_final_score_alpha6_conflict"
    else:
        recommendation = "collect_more_samples"
    return "\n".join(
        [
            "# Final score vs Alpha6 conflict",
            "",
            "Diagnostic only. This report does not change V5 live orders.",
            "",
            f"- conflict_count: {conflict_count}",
            f"- avg_future_4h_net_bps: {_conflict_display_number(avg_4h)}",
            f"- avg_future_8h_net_bps: {_conflict_display_number(avg_8h)}",
            f"- avg_future_24h_net_bps: {_conflict_display_number(avg_24h)}",
            f"- symbol_breakdown: {safe_json_dumps(symbol_counts)}",
            f"- blocked_final_decision_count: {blocked_final_decision_count}",
            f"- negative_expectancy_block_count: {negative_expectancy_block_count}",
            f"- partial_complete_count: {partial_complete_count}",
            f"- recommendation: {recommendation}",
            "",
        ]
    )


def _bnb_strong_alpha6_bypass_summary_md(frame: pl.DataFrame) -> str:
    rows = frame.to_dicts() if not frame.is_empty() else []
    profitable_count = sum(
        1 for row in rows if str(row.get("outcome") or "").strip() == "profitable_shadow"
    )
    negative_expectancy_blocked_count = sum(
        1 for row in rows if _optional_bool(row.get("negative_expectancy_blocked")) is True
    )
    return "\n".join(
        [
            "# BNB strong Alpha6 bypass shadow",
            "",
            "Shadow only. This report never creates a live recommendation.",
            "",
            f"- row_count: {len(rows)}",
            f"- profitable_shadow_count: {profitable_count}",
            f"- negative_expectancy_blocked_count: {negative_expectancy_blocked_count}",
            "- live_order_effect: read_only_no_live_order",
            "- live_recommendation: none",
            "",
        ]
    )


def _negative_expectancy_attribution_summary_md(frame: pl.DataFrame) -> str:
    rows = frame.to_dicts() if not frame.is_empty() else []

    def count_true(field: str) -> int:
        return sum(1 for row in rows if _optional_bool(row.get(field)) is True)

    return "\n".join(
        [
            "# Negative expectancy attribution",
            "",
            "Diagnostic only. Attribution separates entry losses from exit/min-hold failures.",
            "",
            f"- row_count: {len(rows)}",
            f"- entry_bad_count: {count_true('entry_bad')}",
            f"- exit_bad_count: {count_true('exit_bad')}",
            f"- min_hold_violation_count: {count_true('min_hold_violation')}",
            f"- gave_back_profit_count: {count_true('gave_back_profit')}",
            f"- trailing_too_early_count: {count_true('trailing_too_early')}",
            "",
        ]
    )


def _v5_quant_lab_consistency_dashboard_md(
    *,
    frames: dict[str, pl.DataFrame],
    final_score_alpha6_conflict: pl.DataFrame,
    bnb_strong_alpha6_bypass_shadow: pl.DataFrame,
    negative_expectancy_attribution: pl.DataFrame,
    bnb_paper_daily: pl.DataFrame,
    risk_on_multi_buy_shadow: pl.DataFrame,
    opportunity_advisory: pl.DataFrame,
) -> str:
    live_state_ok = _live_state_consistency_ok(frames)
    conflict_rows = (
        final_score_alpha6_conflict.to_dicts() if not final_score_alpha6_conflict.is_empty() else []
    )
    bnb_conflicts = [
        row for row in conflict_rows if normalize_symbol(row.get("symbol")) == "BNB-USDT"
    ]
    bnb_future_positive = any(
        (_optional_float(row.get("future_24h_net_bps")) or 0.0) > 0
        or (_optional_float(row.get("future_12h_net_bps")) or 0.0) > 0
        for row in bnb_conflicts
    )
    attribution_rows = (
        negative_expectancy_attribution.to_dicts()
        if not negative_expectancy_attribution.is_empty()
        else []
    )
    entry_bad_count = sum(
        1 for row in attribution_rows if _optional_bool(row.get("entry_bad")) is True
    )
    exit_bad_count = sum(
        1 for row in attribution_rows if _optional_bool(row.get("exit_bad")) is True
    )
    min_hold_count = sum(
        1 for row in attribution_rows if _optional_bool(row.get("min_hold_violation")) is True
    )
    metadata_missing_count = sum(
        1
        for row in attribution_rows
        if _optional_bool(row.get("exit_metadata_missing")) is True
        or "exit_metadata_missing" in str(row.get("attribution") or "")
    )
    would_unblock_count = sum(
        1
        for row in attribution_rows
        if _optional_bool(row.get("would_unblock_if_adjusted")) is True
    )
    partial_complete_count = sum(
        1
        for row in conflict_rows
        if str(row.get("label_status") or "").strip() == "partial_complete"
        or (
            _optional_bool(row.get("any_label_complete")) is True
            and _optional_bool(row.get("all_labels_complete")) is not True
        )
    )
    mainly_entry_bad = entry_bad_count > max(exit_bad_count, min_hold_count)
    risk_on_observable = _risk_on_selected_symbols_observable(risk_on_multi_buy_shadow)
    paper_ready_from_advisory = _alpha_factory_paper_ready_count_from_advisory(opportunity_advisory)
    paper_ready_from_queue = _alpha_factory_paper_ready_count_from_queue(
        frames.get("alpha_factory_promotion_queue", pl.DataFrame())
    )
    bnb_paper_v5_entry_count = _bnb_paper_today_entry_count(bnb_paper_daily)
    quant_lab_bnb_paper_daily = _filter_bnb_paper_frame(
        frames.get("paper_strategy_daily", pl.DataFrame())
    )
    quant_lab_bnb_paper_daily_latest = _prefer_frame(
        frames.get("v5_bnb_paper_strategy_daily_latest", pl.DataFrame()),
        bnb_paper_daily,
    )
    bnb_paper_quant_lab_raw_entry_count = _bnb_paper_today_entry_count(quant_lab_bnb_paper_daily)
    bnb_paper_quant_lab_entry_count = _bnb_paper_today_entry_count(quant_lab_bnb_paper_daily_latest)
    if bnb_paper_quant_lab_entry_count == 0 and bnb_paper_v5_entry_count > 0:
        bnb_paper_quant_lab_entry_count = bnb_paper_v5_entry_count
    source_mismatch_count = _alpha_factory_source_mismatch_count(
        opportunity_advisory,
        frames.get("alpha_factory_promotion_queue", pl.DataFrame()),
    )
    advisory_duplicate_count = _advisory_duplicate_key_count(opportunity_advisory)
    bnb_paper_count_match = bnb_paper_v5_entry_count == bnb_paper_quant_lab_entry_count
    v5_bundle_lag_detected = not bnb_paper_count_match
    return "\n".join(
        [
            "# V5 / Quant Lab consistency dashboard",
            "",
            "Read-only dashboard. It does not modify V5 live behavior.",
            "",
            f"- live_state_consistency_ok: {str(live_state_ok).lower()}",
            f"- advisory_duplicate_key_count: {advisory_duplicate_count}",
            f"- final_score_vs_alpha6_conflict_count: {len(conflict_rows)}",
            f"- final_score_conflict_partial_complete_count: {partial_complete_count}",
            f"- bnb_conflict_future_pnl_positive: {str(bnb_future_positive).lower()}",
            f"- negative_expectancy_entry_bad_count: {entry_bad_count}",
            f"- negative_expectancy_exit_bad_count: {exit_bad_count}",
            f"- negative_expectancy_min_hold_violation_count: {min_hold_count}",
            f"- negative_expectancy_metadata_missing_count: {metadata_missing_count}",
            f"- negative_expectancy_mainly_entry_bad: {str(mainly_entry_bad).lower()}",
            f"- would_unblock_if_adjusted_count: {would_unblock_count}",
            f"- bnb_paper_today_entry_count: {_bnb_paper_today_entry_count(bnb_paper_daily)}",
            f"- bnb_paper_v5_entry_count: {bnb_paper_v5_entry_count}",
            f"- bnb_paper_quant_lab_entry_count: {bnb_paper_quant_lab_entry_count}",
            f"- bnb_paper_quant_lab_raw_entry_count: {bnb_paper_quant_lab_raw_entry_count}",
            f"- bnb_paper_entry_count_match: {str(bnb_paper_count_match).lower()}",
            f"- v5_bundle_lag_detected: {str(v5_bundle_lag_detected).lower()}",
            f"- risk_on_selected_symbols_observable: {str(risk_on_observable).lower()}",
            f"- alpha_factory_paper_ready_count: {paper_ready_from_queue}",
            f"- alpha_factory_paper_ready_count_from_queue: {paper_ready_from_queue}",
            f"- alpha_factory_paper_ready_count_from_advisory: {paper_ready_from_advisory}",
            f"- alpha_factory_source_mismatch_count: {source_mismatch_count}",
            f"- stale_advisory_count: {_stale_advisory_count(opportunity_advisory)}",
            f"- bnb_strong_alpha6_bypass_shadow_rows: {bnb_strong_alpha6_bypass_shadow.height}",
            "",
        ]
    )


def _live_state_consistency_ok(frames: dict[str, pl.DataFrame]) -> bool:
    issues = frames.get("v5_issue", pl.DataFrame())
    if issues.is_empty():
        return True
    texts: list[str] = []
    for row in issues.to_dicts():
        texts.extend(
            str(row.get(field) or "").lower() for field in ("issue_type", "type", "code", "message")
        )
    blocked_markers = (
        "lifecycle_close_filled_but_position_open",
        "reconcile_flat_but_open_positions_nonzero",
        "close_lifecycle_missing_trade_export",
    )
    return not any(any(marker in text for marker in blocked_markers) for text in texts)


def _bnb_paper_today_entry_count(frame: pl.DataFrame) -> int:
    if frame.is_empty():
        return 0
    if "entry_count" in frame.columns:
        return sum(
            int(_optional_float(value) or 0) for value in frame.get_column("entry_count").to_list()
        )
    if "would_enter_count" in frame.columns:
        return sum(
            int(_optional_float(value) or 0)
            for value in frame.get_column("would_enter_count").to_list()
        )
    return 0


def _bnb_paper_strategy_summary_md(frame: pl.DataFrame) -> str:
    rows = frame.to_dicts() if not frame.is_empty() else []
    entry_count = _bnb_paper_today_entry_count(frame)
    complete_count = sum(int(_optional_float(row.get("complete_count")) or 0) for row in rows)
    pending_count = sum(int(_optional_float(row.get("pending_count")) or 0) for row in rows)
    strategy_count = len(
        {
            strategy
            for strategy in (
                str(row.get("strategy_id") or row.get("proposal_id") or "").strip() for row in rows
            )
            if strategy
        }
    )
    return "\n".join(
        [
            "# BNB paper strategy summary",
            "",
            (
                "Read-only V5 paper telemetry. Entry count is sourced from "
                "v5_bnb_paper_strategy_daily.entry_count."
            ),
            "",
            f"- entry_count: {entry_count}",
            f"- complete_count: {complete_count}",
            f"- pending_count: {pending_count}",
            f"- strategy_count: {strategy_count}",
            "",
        ]
    )


def _risk_on_selected_symbols_observable(frame: pl.DataFrame) -> bool:
    if frame.is_empty() or "selected_symbols" not in frame.columns:
        return False
    for value in frame.get_column("selected_symbols").to_list():
        text = str(value or "").strip().lower()
        if text and text not in {"not_observable", "[]", "null", "none"}:
            return True
    return False


def _count_decision_rows(frame: pl.DataFrame, decision: str) -> int:
    if frame.is_empty() or "decision" not in frame.columns:
        return 0
    target = decision.strip().upper()
    return sum(
        1
        for value in frame.get_column("decision").to_list()
        if str(value or "").strip().upper() == target
    )


def _is_alpha_factory_row_dict(row: dict[str, Any]) -> bool:
    candidate = str(row.get("strategy_candidate") or "").strip()
    candidate_key = (
        candidate.split(":", 1)[1] if candidate.startswith("regime_router:") else candidate
    )
    source_module = str(row.get("source_module") or "").strip().lower()
    template_family = str(row.get("template_family") or "").strip()
    return (
        source_module == "alpha_factory"
        or candidate_key in ALPHA_FACTORY_CANDIDATES
        or candidate_key.startswith("v5.af.")
        or candidate_key.startswith("v5.expanded_relative_strength")
        or bool(template_family)
        or row.get("alpha_factory_score") not in (None, "")
    )


def _alpha_factory_paper_ready_count_from_advisory(frame: pl.DataFrame) -> int:
    if frame.is_empty():
        return 0
    return sum(
        1
        for row in frame.to_dicts()
        if _is_alpha_factory_row_dict(row)
        and str(row.get("decision") or "").strip().upper() == "PAPER_READY"
    )


def _alpha_factory_paper_ready_count_from_queue(frame: pl.DataFrame) -> int:
    if frame.is_empty():
        return 0
    count = 0
    for row in frame.to_dicts():
        state = str(row.get("promotion_state") or row.get("decision") or "").strip().upper()
        if state == "PAPER_READY":
            count += 1
    return count


def _alpha_factory_queue_state_map(frame: pl.DataFrame) -> dict[tuple[str, str, int | None], str]:
    out: dict[tuple[str, str, int | None], str] = {}
    if frame.is_empty():
        return out
    for row in frame.to_dicts():
        candidate = str(row.get("strategy_candidate") or "").strip()
        if not candidate:
            continue
        key = (
            candidate,
            normalize_symbol(row.get("symbol")) or "UNKNOWN",
            _optional_int(row.get("horizon_hours")),
        )
        state = str(row.get("promotion_state") or row.get("decision") or "").strip().upper()
        out[key] = state
    return out


def _alpha_factory_source_mismatch_count(advisory: pl.DataFrame, queue: pl.DataFrame) -> int:
    if advisory.is_empty():
        return 0
    queue_states = _alpha_factory_queue_state_map(queue)
    count = 0
    for row in advisory.to_dicts():
        if not _is_alpha_factory_row_dict(row):
            continue
        if str(row.get("decision") or "").strip().upper() != "PAPER_READY":
            continue
        candidate = str(row.get("strategy_candidate") or "").strip()
        candidate_key = (
            candidate.split(":", 1)[1] if candidate.startswith("regime_router:") else candidate
        )
        key = (
            candidate_key,
            normalize_symbol(row.get("symbol")) or "UNKNOWN",
            _optional_int(row.get("horizon_hours")),
        )
        state = (
            queue_states.get(key)
            or queue_states.get((key[0], key[1], None))
            or queue_states.get((key[0], "UNKNOWN", key[2]))
            or queue_states.get((key[0], "UNKNOWN", None))
        )
        if state != "PAPER_READY":
            count += 1
    return count


def _stale_advisory_count(frame: pl.DataFrame) -> int:
    if frame.is_empty():
        return 0
    count = 0
    for row in frame.to_dicts():
        fresh = row.get("advisory_fresh")
        source_stale = row.get("selected_source_is_stale")
        stale_reason = str(row.get("stale_reason") or "").strip()
        if _optional_bool(fresh) is False or _optional_bool(source_stale) is True or stale_reason:
            count += 1
    return count


def _risk_on_multi_buy_shadow_for_export(
    *,
    candidate_events: pl.DataFrame,
    v5_trades: pl.DataFrame,
    market_bars: pl.DataFrame,
    market_regime: pl.DataFrame,
    cost_buckets: pl.DataFrame | None = None,
) -> pl.DataFrame:
    path = "reports/risk_on_multi_buy_shadow.csv"
    if candidate_events.is_empty():
        return _empty_csv_schema_frame(path)
    market_by_symbol = _market_close_rows_by_symbol(market_bars)
    regime_by_date = _market_regime_context_by_date(market_regime)
    latest_regime_context = _latest_market_regime_context(market_regime)
    actual_opened = _actual_open_trade_symbols_by_run(v5_trades)
    shadow_cost_by_symbol = _risk_on_shadow_cost_by_symbol(
        cost_buckets if cost_buckets is not None else pl.DataFrame()
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidate_events.to_dicts():
        symbol = normalize_symbol(candidate.get("symbol"))
        if symbol not in RISK_ON_MULTI_BUY_SYMBOLS:
            continue
        ts = readers._coerce_timestamp(candidate.get("ts_utc") or candidate.get("ts"))
        if ts is None:
            continue
        market_regime_context = _market_regime_context_for_ts(
            ts,
            regime_by_date,
            latest_regime_context,
        )
        regime_context = _candidate_market_regime_context(
            candidate,
            ts,
            regime_by_date,
            latest_regime_context,
            market_by_symbol=market_by_symbol,
        )
        run_id = str(candidate.get("run_id") or "").strip() or ts.astimezone(UTC).isoformat()
        grouped.setdefault(run_id, []).append(
            candidate
            | {
                "_ts": ts.astimezone(UTC),
                "_regime_context": regime_context,
                "_market_regime_context": market_regime_context,
                "_v5_candidate_regime_state": _risk_on_candidate_v5_regime(candidate),
            }
        )
    generated_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for run_id, raw_candidates in grouped.items():
        candidate_buy_count = _risk_on_candidate_buy_count(raw_candidates)
        candidates = [
            row
            for row in raw_candidates
            if _risk_on_candidate_passes(row, candidate_buy_count=candidate_buy_count)
        ]
        if not candidates:
            continue
        candidates = sorted(
            candidates,
            key=lambda row: _optional_float(row.get("final_score")) or float("-inf"),
            reverse=True,
        )
        for top_k, strategy_candidate in [
            (1, "v5.risk_on_multi_buy_top1_shadow"),
            (2, "v5.risk_on_multi_buy_top2_shadow"),
            (3, "v5.risk_on_multi_buy_top3_shadow"),
        ]:
            selected = candidates[:top_k]
            if not selected:
                continue
            selected_symbols = [normalize_symbol(row.get("symbol")) for row in selected]
            selected_symbols = [symbol for symbol in selected_symbols if symbol]
            decision_ts = min(row["_ts"] for row in selected)
            selected_regime_context = selected[0].get("_regime_context") or latest_regime_context
            selected_market_regime_context = (
                selected[0].get("_market_regime_context") or latest_regime_context
            )
            trigger = _risk_on_trigger_context(selected[0], candidate_buy_count)
            selected_metrics: list[dict[str, Any]] = []
            for row in selected:
                symbol = normalize_symbol(row.get("symbol"))
                symbol_market = market_by_symbol.get(symbol, [])
                entry_px = _candidate_entry_price(row, symbol_market, row["_ts"])
                shadow_cost_bps = shadow_cost_by_symbol.get(
                    symbol,
                    RISK_ON_MULTI_BUY_COST_BPS,
                )
                selected_metrics.append(
                    {
                        "row": row,
                        "symbol": symbol,
                        "entry_px": entry_px,
                        "future_4h_net_bps": _future_net_bps_after_shadow_cost(
                            symbol_market,
                            row["_ts"],
                            entry_px,
                            4,
                            shadow_cost_bps=shadow_cost_bps,
                        ),
                        "future_8h_net_bps": _future_net_bps_after_shadow_cost(
                            symbol_market,
                            row["_ts"],
                            entry_px,
                            8,
                            shadow_cost_bps=shadow_cost_bps,
                        ),
                        "future_24h_net_bps": _future_net_bps_after_shadow_cost(
                            symbol_market,
                            row["_ts"],
                            entry_px,
                            24,
                            shadow_cost_bps=shadow_cost_bps,
                        ),
                    }
                )
            pnl_by_horizon = {
                horizon: _average_optional(
                    [
                        _optional_float(metric.get(f"future_{horizon}h_net_bps"))
                        for metric in selected_metrics
                    ]
                )
                for horizon in (4, 8, 24)
            }
            actual_symbols = sorted(
                symbol for candidate_run, symbol in actual_opened if candidate_run == run_id
            )
            actual_net = _average_optional(
                [
                    _future_net_bps_after_shadow_cost(
                        market_by_symbol.get(symbol, []),
                        decision_ts,
                        _market_close_at_or_after(market_by_symbol.get(symbol, []), decision_ts),
                        24,
                    )
                    for symbol in actual_symbols
                ]
            )
            missed_symbols = sorted(set(selected_symbols) - set(actual_symbols))
            for metric in selected_metrics:
                selected_row = metric["row"]
                rows.append(
                    {
                        "generated_at": generated_at,
                        "schema_version": "v5_risk_on_multi_buy_shadow.v0.2",
                        "run_id": run_id,
                        "decision_ts": decision_ts.isoformat(),
                        "current_regime": selected_regime_context.get("current_regime"),
                        "regime_source": trigger["regime_source"],
                        "candidate_buy_count": candidate_buy_count,
                        "market_regime_daily_state": selected_market_regime_context.get(
                            "current_regime"
                        ),
                        "v5_candidate_regime_state": selected_row.get("_v5_candidate_regime_state"),
                        "trigger_reason": trigger["trigger_reason"],
                        "broad_market_positive_count": selected_regime_context.get(
                            "broad_market_positive_count"
                        ),
                        "btc_24h_return_bps": selected_regime_context.get("btc_24h_return_bps"),
                        "strategy_candidate": strategy_candidate,
                        "top_k": top_k,
                        "selected_symbols": safe_json_dumps(selected_symbols),
                        "selected_count": len(selected_symbols),
                        "would_buy_symbol": metric["symbol"],
                        "would_buy": bool(metric["symbol"]),
                        "would_size_usdt": 1000.0,
                        "final_score": _optional_float(selected_row.get("final_score")),
                        "expected_edge_bps": _optional_float(selected_row.get("expected_edge_bps")),
                        "required_edge_bps": _optional_float(selected_row.get("required_edge_bps")),
                        "entry_px": metric["entry_px"],
                        "future_4h_net_bps": metric["future_4h_net_bps"],
                        "future_8h_net_bps": metric["future_8h_net_bps"],
                        "future_24h_net_bps": metric["future_24h_net_bps"],
                        "avg_portfolio_net_bps": pnl_by_horizon[24],
                        "actual_v5_bought_symbols": safe_json_dumps(actual_symbols),
                        "vs_actual_v5_net_bps": (
                            None
                            if actual_net is None or pnl_by_horizon[24] is None
                            else pnl_by_horizon[24] - actual_net
                        ),
                        "missed_symbols": safe_json_dumps(missed_symbols),
                        "recommended_mode": "shadow",
                        "max_live_notional_usdt": 0.0,
                        "source": "quant_lab_risk_on_multi_buy_shadow",
                    }
                )
    if not rows:
        return _empty_csv_schema_frame(path)
    return pl.DataFrame(rows, infer_schema_length=None).select(CSV_SCHEMAS[path])


def _missed_opportunity_summary_md(audit: pl.DataFrame, risk_on_shadow: pl.DataFrame) -> str:
    lines = [
        "# Missed Opportunity Audit",
        "",
        "Quant Lab is a research/advisory/audit system, not the V5 live commander.",
        (
            "The audit estimates whether advisory-only blocking would have missed profit "
            "or avoided loss."
        ),
        "",
    ]
    if audit.is_empty():
        lines.append("- No candidate events were available for missed-opportunity analysis.")
    else:
        counts = Counter(
            str(row.get("outcome_if_blocked") or "unknown") for row in audit.to_dicts()
        )
        for key, value in sorted(counts.items()):
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Risk-on Multi-buy Shadow", ""])
    if risk_on_shadow.is_empty():
        lines.append("- No ALT_IMPULSE/TREND_UP multi-buy shadow candidates were observed.")
    else:
        latest_rows = sorted(
            risk_on_shadow.to_dicts(),
            key=lambda row: str(row.get("decision_ts") or ""),
            reverse=True,
        )[:5]
        for row in latest_rows:
            lines.append(
                "- "
                f"{row.get('strategy_candidate')} selected {row.get('selected_symbols')} "
                f"24h_net={row.get('future_24h_net_bps')}"
            )
    return "\n".join(lines) + "\n"


def _paper_identity_conflict_md(frame: pl.DataFrame) -> str:
    rows = frame.to_dicts() if not frame.is_empty() else []
    active = [
        row for row in rows if _optional_bool(row.get("active_conflict")) is True
    ]
    lines = [
        "# Paper Strategy Identity Conflicts",
        "",
        f"- total_conflicts: {len(rows)}",
        f"- active_conflicts: {len(active)}",
        "- fail_closed: true",
        "",
    ]
    for row in active[:50]:
        lines.append(
            "- "
            f"{row.get('proposal_id')} | {row.get('conflict_field')} | "
            f"canonical={row.get('canonical_value')} | observed={row.get('observed_value')} | "
            f"source={row.get('source_dataset')}"
        )
    if not active:
        lines.append("No active identity conflicts. Historical resolved rows remain auditable.")
    return "\n".join(lines) + "\n"


def _paper_cohort_status_md(frame: pl.DataFrame) -> str:
    rows = frame.sort("cohort_version").to_dicts() if not frame.is_empty() else []
    lines = [
        "# Paper Cohort Status",
        "",
        "Cohort membership is immutable. New proposal sets create a new cohort version.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('cohort_id')}",
                f"- version: {row.get('cohort_version')}",
                f"- status: {row.get('status')}",
                f"- observation_start_at: {row.get('observation_start_at')}",
                f"- proposal_ids: {row.get('proposal_ids')}",
                f"- raw_closed_trade_count: {row.get('raw_closed_trade_count')}",
                "- independent_closed_trade_count: "
                f"{row.get('independent_closed_trade_count')}",
                f"- horizon_variant_count: {row.get('horizon_variant_count')}",
                "",
            ]
        )
    if not rows:
        lines.append("No cohort has been admitted.")
    return "\n".join(lines) + "\n"


def _bnb_missed_opportunity_summary_md(samples: pl.DataFrame) -> str:
    lines = [
        "# BNB Missed Opportunity Summary",
        "",
        (
            "Scope: BNB-USDT candidate events from 2026-05-30 onward where "
            "alpha6_side=buy, final_decision is no_order/blocked, and "
            "expected_edge_bps exceeded required_edge_bps."
        ),
        (
            "Forward returns are conservative shadow net bps using the same 30 bps "
            "roundtrip cost floor as the risk-on shadow reports."
        ),
        "",
    ]
    if samples.is_empty():
        lines.append("- No qualifying BNB alpha6 buy no-order/blocked samples were found.")
        return "\n".join(lines) + "\n"

    rows = samples.to_dicts()
    missed_profit_count = sum(
        1 for row in rows if _optional_bool(row.get("missed_profit_flag")) is True
    )
    lines.extend(
        [
            f"- qualifying_samples: {len(rows)}",
            f"- missed_profit_samples: {missed_profit_count}",
        ]
    )
    for horizon in (4, 8, 12, 24):
        values = [_optional_float(row.get(f"future_{horizon}h_net_bps")) for row in rows]
        observed = [value for value in values if value is not None]
        if not observed:
            lines.append(f"- avg_future_{horizon}h_net_bps: pending")
        else:
            lines.append(
                f"- avg_future_{horizon}h_net_bps: "
                f"{sum(observed) / len(observed):.2f} "
                f"(observed={len(observed)})"
            )

    latest_rows = sorted(rows, key=lambda row: str(row.get("ts_utc") or ""), reverse=True)[:10]
    lines.extend(["", "## Latest qualifying samples", ""])
    for row in latest_rows:
        lines.append(
            "- "
            f"{row.get('ts_utc')} run={row.get('run_id')} "
            f"decision={row.get('final_decision')} "
            f"score={row.get('final_score')} "
            f"4h={row.get('future_4h_net_bps')} "
            f"8h={row.get('future_8h_net_bps')} "
            f"12h={row.get('future_12h_net_bps')} "
            f"24h={row.get('future_24h_net_bps')} "
            f"missed_profit={row.get('missed_profit_flag')}"
        )
    return "\n".join(lines) + "\n"


def _risk_on_multi_buy_summary_md(risk_on_shadow: pl.DataFrame) -> str:
    lines = [
        "# Risk-on Multi-buy Shadow",
        "",
        (
            "This is a shadow-only research report for ALT_IMPULSE/TREND_UP "
            "multi-symbol buying. It does not change V5 live execution."
        ),
        "",
    ]
    if risk_on_shadow.is_empty():
        lines.append("- No qualifying risk-on multi-buy shadow samples were observed.")
        return "\n".join(lines) + "\n"

    rows = sorted(
        risk_on_shadow.to_dicts(),
        key=lambda row: (
            str(row.get("decision_ts") or ""),
            _optional_int(row.get("top_k")) or 0,
        ),
        reverse=True,
    )
    latest_by_top_k: dict[int, dict[str, Any]] = {}
    for row in rows:
        top_k = _optional_int(row.get("top_k"))
        if top_k is None or top_k in latest_by_top_k:
            continue
        latest_by_top_k[top_k] = row

    lines.append("## Latest portfolios")
    for top_k in sorted(latest_by_top_k):
        row = latest_by_top_k[top_k]
        lines.append(
            "- "
            f"top{top_k}: selected={row.get('selected_symbols')}; "
            f"24h_portfolio_net_bps={_format_summary_number(row.get('avg_portfolio_net_bps'))}; "
            f"vs_actual_v5_net_bps={_format_summary_number(row.get('vs_actual_v5_net_bps'))}; "
            f"actual_v5_bought={row.get('actual_v5_bought_symbols')}; "
            f"missed={row.get('missed_symbols')}"
        )

    symbol_counts = Counter(
        str(row.get("would_buy_symbol") or "unknown") for row in risk_on_shadow.to_dicts()
    )
    lines.extend(["", "## Selected symbol counts"])
    for symbol, count in sorted(symbol_counts.items()):
        lines.append(f"- {symbol}: {count}")
    return "\n".join(lines) + "\n"


def _format_summary_number(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "not_observable"
    return f"{number:.2f}"


def _v5_local_live_vs_quant_lab_shadow_for_export(
    *,
    v5_trades: pl.DataFrame,
    market_bars: pl.DataFrame,
    opportunity_advisory: pl.DataFrame,
) -> pl.DataFrame:
    path = "reports/v5_local_live_vs_quant_lab_shadow.csv"
    if v5_trades.is_empty():
        return _empty_csv_schema_frame(path)
    market_by_symbol = _market_close_rows_by_symbol(market_bars)
    advisory_by_symbol = _latest_advisory_by_symbol(opportunity_advisory)
    generated_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for trade in v5_trades.to_dicts():
        if not _is_open_long_trade_event(trade):
            continue
        symbol = normalize_symbol(trade.get("normalized_symbol") or trade.get("symbol"))
        entry_ts = readers._coerce_timestamp(  # type: ignore[attr-defined]
            trade.get("ts_utc") or trade.get("ts") or trade.get("created_at")
        )
        entry_price = _optional_float(
            trade.get("price") or trade.get("fill_px") or trade.get("avg_fill_px")
        )
        if not symbol or symbol == "UNKNOWN" or entry_ts is None or not entry_price:
            continue
        advisory = _matching_quant_lab_advisory(trade, symbol, advisory_by_symbol)
        pnl = {
            horizon: _future_long_pnl_bps(
                market_by_symbol.get(symbol, []),
                entry_ts.astimezone(UTC),
                entry_price,
                horizon,
            )
            for horizon in (4, 8, 24)
        }
        would_block = _advisory_would_block_hypothetically(advisory)
        block_reasons = _json_listish(advisory.get("live_block_reasons")) if advisory else []
        if not advisory:
            block_reasons = ["no_matching_quant_lab_advisory"]
        rows.append(
            {
                "generated_at": generated_at,
                "schema_version": "v5_local_live_vs_quant_lab_shadow.v0.1",
                "trade_id": trade.get("trade_id"),
                "order_id": trade.get("order_id") or trade.get("exchange_order_id"),
                "run_id": trade.get("run_id"),
                "entry_ts": entry_ts.astimezone(UTC).isoformat(),
                "symbol": symbol,
                "side": str(trade.get("side") or "").lower(),
                "action": trade.get("action") or trade.get("intent"),
                "entry_price": entry_price,
                "matching_strategy_candidate": (
                    advisory.get("strategy_candidate") if advisory else None
                ),
                "matching_decision": advisory.get("decision") if advisory else None,
                "matching_recommended_mode": advisory.get("recommended_mode") if advisory else None,
                "matching_advisory_intent": advisory.get("advisory_intent") if advisory else None,
                "quant_lab_is_live_commander": False,
                "quant_lab_would_block_if_enabled": would_block,
                "quant_lab_block_reasons": safe_json_dumps(block_reasons),
                "pnl_4h_bps": pnl[4],
                "pnl_8h_bps": pnl[8],
                "pnl_24h_bps": pnl[24],
                "block_outcome_4h": _block_outcome(would_block, pnl[4]),
                "block_outcome_8h": _block_outcome(would_block, pnl[8]),
                "block_outcome_24h": _block_outcome(would_block, pnl[24]),
                "block_outcome_summary": _block_outcome_summary(would_block, pnl),
                "source": "quant_lab_shadow_audit",
            }
        )
    if not rows:
        return _empty_csv_schema_frame(path)
    return pl.DataFrame(rows, infer_schema_length=None).select(CSV_SCHEMAS[path])


def _market_close_rows_by_symbol(market_bars: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if market_bars.is_empty() or "close" not in market_bars.columns:
        return {}
    output: dict[str, list[dict[str, Any]]] = {}
    for row in market_bars.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        ts = readers._coerce_timestamp(row.get("ts"))  # type: ignore[attr-defined]
        close = _optional_float(row.get("close"))
        if not symbol or symbol == "UNKNOWN" or ts is None or close is None:
            continue
        output.setdefault(symbol, []).append({"ts": ts.astimezone(UTC), "close": close})
    for rows in output.values():
        rows.sort(key=lambda row: row["ts"])
    return output


def _actual_open_trade_symbols_by_run(v5_trades: pl.DataFrame) -> set[tuple[str, str]]:
    opened: set[tuple[str, str]] = set()
    if v5_trades.is_empty():
        return opened
    for trade in v5_trades.to_dicts():
        if not _is_open_long_trade_event(trade):
            continue
        run_id = str(trade.get("run_id") or "").strip()
        symbol = normalize_symbol(trade.get("normalized_symbol") or trade.get("symbol"))
        if run_id and symbol and symbol != "UNKNOWN":
            opened.add((run_id, symbol))
    return opened


def _latest_market_regime_context(market_regime: pl.DataFrame) -> dict[str, Any]:
    if market_regime.is_empty():
        return _empty_market_regime_context()
    row = max(market_regime.to_dicts(), key=_advisory_row_time)
    return _market_regime_context_from_row(row)


def _market_regime_context_by_date(market_regime: pl.DataFrame) -> dict[str, dict[str, Any]]:
    if market_regime.is_empty():
        return {}
    latest_by_date: dict[str, dict[str, Any]] = {}
    for row in market_regime.to_dicts():
        as_of = _market_regime_row_date(row)
        if not as_of:
            continue
        current = latest_by_date.get(as_of)
        if current is None or _advisory_row_time(row) >= _advisory_row_time(current):
            latest_by_date[as_of] = row
    return {day: _market_regime_context_from_row(row) for day, row in latest_by_date.items()}


def _market_regime_context_for_ts(
    ts: datetime,
    regime_by_date: dict[str, dict[str, Any]],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    day = ts.astimezone(UTC).date().isoformat()
    return regime_by_date.get(day) or fallback or _empty_market_regime_context()


def _candidate_market_regime_context(
    candidate: dict[str, Any],
    ts: datetime,
    regime_by_date: dict[str, dict[str, Any]],
    fallback: dict[str, Any],
    *,
    market_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    base = dict(_market_regime_context_for_ts(ts, regime_by_date, fallback))
    candidate_regime = candidate.get("current_regime") or candidate.get("regime_state")
    if str(candidate_regime or "").strip():
        base["current_regime"] = _risk_on_regime_name(candidate_regime)
    candidate_broad = _optional_int(candidate.get("broad_market_positive_count"))
    if candidate_broad is not None:
        base["broad_market_positive_count"] = candidate_broad
    candidate_btc = _optional_float(candidate.get("btc_24h_return_bps"))
    if candidate_btc is not None:
        base["btc_24h_return_bps"] = candidate_btc
    elif market_by_symbol:
        computed_btc = _historical_return_bps(
            market_by_symbol.get("BTC-USDT", []),
            ts.astimezone(UTC),
            24,
        )
        if computed_btc is not None:
            base["btc_24h_return_bps"] = computed_btc
    return base


def _market_regime_row_date(row: dict[str, Any]) -> str | None:
    raw = row.get("as_of_date") or row.get("date") or row.get("day")
    if raw:
        text = str(raw).strip()
        if text:
            return text[:10]
    ts = readers._coerce_timestamp(
        row.get("created_at") or row.get("latest_bundle_ts") or row.get("ts")
    )
    if ts is None:
        return None
    return ts.astimezone(UTC).date().isoformat()


def _market_regime_context_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_regime": _risk_on_regime_name(row.get("current_regime") or "UNKNOWN"),
        "broad_market_positive_count": _optional_int(row.get("broad_market_positive_count")) or 0,
        "btc_24h_return_bps": _optional_float(row.get("btc_24h_return_bps")),
    }


def _empty_market_regime_context() -> dict[str, Any]:
    return {
        "current_regime": "UNKNOWN",
        "broad_market_positive_count": 0,
        "btc_24h_return_bps": None,
    }


def _candidate_entry_price(
    candidate: dict[str, Any],
    market_rows: list[dict[str, Any]],
    ts: datetime,
) -> float | None:
    for field in [
        "entry_close",
        "close",
        "current_px",
        "price",
        "reference_price",
        "mark_price",
    ]:
        value = _optional_float(candidate.get(field))
        if value is not None and value > 0:
            return value
    return _market_close_at_or_after(market_rows, ts.astimezone(UTC))


def _market_close_at_or_after(
    market_rows: list[dict[str, Any]],
    ts: datetime,
) -> float | None:
    if not market_rows:
        return None
    for row in market_rows:
        if row["ts"] >= ts:
            return _optional_float(row.get("close"))
    return _optional_float(market_rows[-1].get("close"))


def _market_close_at_or_before(
    market_rows: list[dict[str, Any]],
    ts: datetime,
) -> float | None:
    if not market_rows:
        return None
    previous: float | None = None
    for row in market_rows:
        if row["ts"] > ts:
            break
        previous = _optional_float(row.get("close"))
    return previous


def _historical_return_bps(
    market_rows: list[dict[str, Any]],
    ts: datetime,
    lookback_hours: int,
) -> float | None:
    current = _market_close_at_or_before(market_rows, ts)
    previous = _market_close_at_or_before(market_rows, ts - timedelta(hours=lookback_hours))
    if current is None or previous is None or previous <= 0:
        return None
    return (current / previous - 1.0) * 10000.0


def _future_net_bps_after_shadow_cost(
    market_rows: list[dict[str, Any]],
    entry_ts: datetime,
    entry_price: float | None,
    horizon_hours: int,
    *,
    shadow_cost_bps: float = RISK_ON_MULTI_BUY_COST_BPS,
) -> float | None:
    if entry_price is None or entry_price <= 0:
        return None
    gross = _future_long_pnl_bps(market_rows, entry_ts, entry_price, horizon_hours)
    if gross is None:
        return None
    return gross - shadow_cost_bps


def _candidate_is_strong_buy(candidate: dict[str, Any]) -> bool:
    side = str(candidate.get("alpha6_side") or candidate.get("side") or "").lower()
    if side and "buy" not in side and "long" not in side:
        return False
    final_score = _optional_float(candidate.get("final_score"))
    expected = _optional_float(candidate.get("expected_edge_bps"))
    required = _optional_float(candidate.get("required_edge_bps"))
    if expected is not None and required is not None:
        return expected > required
    return final_score is not None and final_score > 0


def _missed_opportunity_outcome(
    *,
    actual_trade_opened: bool,
    quant_lab_would_block_live: bool,
    strong_candidate: bool,
    future_net_bps: dict[int, float | None],
) -> str:
    observed = [value for value in future_net_bps.values() if value is not None]
    if not observed:
        return "pending"
    best_future = max(observed)
    worst_future = min(observed)
    if actual_trade_opened and quant_lab_would_block_live and best_future > 0:
        return "quant_lab_would_have_missed_profit"
    if not actual_trade_opened and strong_candidate and best_future > 0:
        return "v5_missed_profit_opportunity"
    if quant_lab_would_block_live and worst_future < 0:
        return "quant_lab_correctly_avoided_loss"
    if not actual_trade_opened and strong_candidate and worst_future < 0:
        return "v5_correctly_skipped_loss"
    return "pending"


def _risk_on_candidate_passes(
    candidate: dict[str, Any],
    *,
    candidate_buy_count: int = 0,
) -> bool:
    trigger = _risk_on_trigger_context(candidate, candidate_buy_count)
    if not trigger["triggered"]:
        return False
    if not _candidate_is_strong_buy(candidate):
        return False
    if _optional_bool(candidate.get("cost_gate_verified")) is not True:
        return False
    return _optional_float(candidate.get("final_score")) is not None


def _risk_on_trigger_context(
    candidate: dict[str, Any],
    candidate_buy_count: int,
) -> dict[str, Any]:
    market_context = candidate.get("_market_regime_context") or {}
    market_regime = _risk_on_regime_name(market_context.get("current_regime") or "")
    broad = int(market_context.get("broad_market_positive_count") or 0)
    btc_return = _optional_float(market_context.get("btc_24h_return_bps"))
    market_trigger = (
        market_regime in RISK_ON_MULTI_BUY_REGIMES
        and broad >= 3
        and btc_return is not None
        and btc_return > 0
    )
    v5_regime = _risk_on_candidate_v5_regime(candidate)
    intraday_trigger = v5_regime in RISK_ON_MULTI_BUY_REGIMES and candidate_buy_count >= 2
    if market_trigger:
        return {
            "triggered": True,
            "regime_source": "market_regime_daily",
            "trigger_reason": "market_regime_daily_risk_on",
        }
    if intraday_trigger:
        return {
            "triggered": True,
            "regime_source": "v5_candidate_event",
            "trigger_reason": "intraday_v5_candidate_risk_on",
        }
    return {
        "triggered": False,
        "regime_source": "none",
        "trigger_reason": "risk_on_conditions_not_met",
    }


def _risk_on_candidate_buy_count(candidates: list[dict[str, Any]]) -> int:
    symbols = {
        normalize_symbol(candidate.get("symbol"))
        for candidate in candidates
        if _risk_on_candidate_v5_regime(candidate) in RISK_ON_MULTI_BUY_REGIMES
        and _risk_on_candidate_is_buy(candidate)
    }
    return sum(1 for symbol in symbols if symbol and symbol != "UNKNOWN")


def _risk_on_candidate_is_buy(candidate: dict[str, Any]) -> bool:
    side = str(candidate.get("alpha6_side") or candidate.get("side") or "").lower()
    return "buy" in side or "long" in side


def _risk_on_candidate_v5_regime(candidate: dict[str, Any]) -> str:
    return _risk_on_regime_name(
        candidate.get("_v5_candidate_regime_state")
        or candidate.get("current_regime")
        or candidate.get("regime_state")
        or ""
    )


def _risk_on_regime_name(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"TRENDING", "TREND", "BULL", "BULLISH"}:
        return "TREND_UP"
    return text


def _risk_on_shadow_cost_by_symbol(cost_buckets: pl.DataFrame) -> dict[str, float]:
    if cost_buckets.is_empty():
        return {}
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for row in cost_buckets.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol or symbol == "UNKNOWN":
            continue
        current = latest_by_symbol.get(symbol)
        if current is None or _advisory_row_time(row) >= _advisory_row_time(current):
            latest_by_symbol[symbol] = row
    output: dict[str, float] = {}
    for symbol, row in latest_by_symbol.items():
        cost = (
            _optional_float(row.get("roundtrip_all_in_cost_bps"))
            or _optional_float(row.get("total_cost_bps_p75"))
            or _optional_float(row.get("selected_total_cost_bps"))
            or _optional_float(row.get("cost_bps"))
        )
        if cost is not None and cost >= 0:
            output[symbol] = max(cost, RISK_ON_MULTI_BUY_COST_BPS)
    return output


def _average_optional(values: list[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return sum(observed) / len(observed)


def _latest_advisory_by_symbol(frame: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if frame.is_empty():
        return {}
    output: dict[str, list[dict[str, Any]]] = {}
    for row in frame.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol or symbol == "UNKNOWN":
            continue
        output.setdefault(symbol, []).append(row)
    for rows in output.values():
        rows.sort(key=_advisory_row_time, reverse=True)
    return output


def _matching_quant_lab_advisory(
    trade: dict[str, Any],
    symbol: str,
    advisory_by_symbol: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    candidates = advisory_by_symbol.get(symbol, [])
    if not candidates:
        return None
    trade_candidate = str(
        trade.get("strategy_candidate")
        or trade.get("candidate_name")
        or trade.get("strategy_id")
        or ""
    ).lower()
    if trade_candidate:
        for row in candidates:
            if trade_candidate in str(row.get("strategy_candidate") or "").lower():
                return row
    return candidates[0]


def _is_open_long_trade_event(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(field) or "").lower() for field in ["action", "intent", "event_type", "side"]
    )
    if any(token in text for token in ["exit", "close", "sell_only"]):
        return False
    return any(token in text for token in ["open_long", "entry", "open", "buy"])


def _future_long_pnl_bps(
    market_rows: list[dict[str, Any]],
    entry_ts: datetime,
    entry_price: float,
    horizon_hours: int,
) -> float | None:
    target = entry_ts + timedelta(hours=horizon_hours)
    for row in market_rows:
        if row["ts"] >= target:
            return (float(row["close"]) / entry_price - 1.0) * 10_000.0
    return None


def _advisory_would_block_hypothetically(advisory: dict[str, Any] | None) -> bool:
    if advisory is None:
        return False
    if _optional_bool(advisory.get("would_block_if_enabled")) is True:
        return True
    return str(advisory.get("decision") or "").upper() == "KILL"


def _block_outcome(would_block: bool, pnl_bps: float | None) -> str:
    if not would_block:
        return "not_blocked_by_quant_lab"
    if pnl_bps is None:
        return "blocked_outcome_pending"
    return "missed_profit_if_blocked" if pnl_bps > 0 else "avoided_loss_if_blocked"


def _block_outcome_summary(would_block: bool, pnl_by_horizon: dict[int, float | None]) -> str:
    if not would_block:
        return "quant_lab_would_not_block"
    observed = [value for value in pnl_by_horizon.values() if value is not None]
    if not observed:
        return "blocked_outcome_pending"
    return "would_miss_profit" if max(observed) > 0 else "would_avoid_loss"


def _advisory_strategy_id(strategy_candidate: str, symbol: str) -> str:
    raw = f"{symbol}_{strategy_candidate}"
    normalized = "".join(character if character.isalnum() else "_" for character in raw.upper())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _entry_quality_as_of_ts(row: dict[str, Any]) -> str:
    value = row.get("generated_at_utc") or row.get("as_of_date")
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat() if value.tzinfo else value.isoformat()
    raw = str(value or "").strip()
    if len(raw) == 10 and raw[4] == "-":
        return f"{raw}T00:00:00+00:00"
    return raw or datetime.now(UTC).isoformat()


def _strategy_level_dashboard_for_export(opportunity_advisory: pl.DataFrame) -> pl.DataFrame:
    path = "reports/strategy_level_dashboard.csv"
    if opportunity_advisory.is_empty():
        return _empty_csv_schema_frame(path)
    normalized = opportunity_advisory
    for column in CSV_SCHEMAS[path]:
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    rows = normalized.select(CSV_SCHEMAS[path]).to_dicts()
    decision_rank = {
        "PAPER_READY": 0,
        "LIVE_SMALL_READY": 1,
        "KEEP_SHADOW": 2,
        "REGIME_SHADOW": 3,
        "KILL": 4,
        "RESEARCH_ONLY": 5,
    }
    rows.sort(
        key=lambda row: (
            decision_rank.get(str(row.get("decision") or "").upper(), 99),
            str(row.get("strategy_candidate") or ""),
            str(row.get("symbol") or ""),
            int(row.get("horizon_hours") or 0),
        )
    )
    return pl.DataFrame(rows, infer_schema_length=None).select(CSV_SCHEMAS[path])


def _advisory_board_from_strategy_evidence(strategy_evidence: pl.DataFrame) -> pl.DataFrame:
    if strategy_evidence.is_empty():
        return strategy_evidence
    return normalize_strategy_evidence_decisions(strategy_evidence)


def _latest_rows_by_candidate_symbol(frame: pl.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if frame.is_empty():
        return rows_by_key
    for row in frame.to_dicts():
        candidate = str(row.get("strategy_candidate") or row.get("candidate_name") or "").strip()
        symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
        if not candidate or symbol == "UNKNOWN":
            continue
        key = (candidate, symbol)
        current = rows_by_key.get(key)
        if current is None or _advisory_row_time(row) >= _advisory_row_time(current):
            rows_by_key[key] = row
    return rows_by_key


def _portfolio_status_overrides_by_candidate_symbol(
    research_portfolio: pl.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    return portfolio_status_overrides_by_identifier(research_portfolio)


def _portfolio_override_for_row(
    row: dict[str, Any],
    overrides: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    return portfolio_override_for_row(row, overrides)


def _portfolio_override_for_candidate_symbol(
    overrides: dict[tuple[str, str], dict[str, Any]],
    candidate: str,
    symbol: str,
) -> dict[str, Any] | None:
    return portfolio_override_for_identifier_symbol(overrides, candidate, symbol)


def _portfolio_overridden_decision_mode(
    *,
    decision: str,
    recommended_mode: str,
    portfolio_override: dict[str, Any] | None,
) -> tuple[str, str, list[str]]:
    return portfolio_overridden_decision_mode(
        decision=decision,
        recommended_mode=recommended_mode,
        portfolio_override=portfolio_override,
    )


def _alpha_factory_promotion_overrides_by_candidate_symbol(
    promotion_queue: pl.DataFrame,
) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    overrides: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    if promotion_queue.is_empty():
        return overrides
    rows = promotion_queue.to_dicts()
    latest_as_of_date = _latest_as_of_date(rows)
    if latest_as_of_date:
        rows = [
            row for row in rows if str(row.get("as_of_date") or "").strip() == latest_as_of_date
        ]
    for row in rows:
        candidate = str(row.get("strategy_candidate") or "").strip()
        symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
        if not candidate:
            continue
        key = (candidate, symbol, _optional_int(row.get("horizon_hours")))
        current = overrides.get(key)
        if current is None or _advisory_row_time(row) >= _advisory_row_time(current):
            overrides[key] = row
    return overrides


def _alpha_factory_result_metadata_by_candidate_symbol(
    results: pl.DataFrame,
) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    metadata: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    if results.is_empty():
        return metadata
    rows = results.to_dicts()
    latest_as_of_date = _latest_as_of_date(rows)
    if latest_as_of_date:
        rows = [
            row for row in rows if str(row.get("as_of_date") or "").strip() == latest_as_of_date
        ]
    for row in rows:
        candidate = str(row.get("strategy_candidate") or "").strip()
        symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
        if not candidate:
            continue
        enriched = {
            "source_module": "alpha_factory",
            "template_family": _alpha_factory_template_family(row),
            "candidate_id": row.get("candidate_id"),
            "promotion_state": row.get("promotion_state") or row.get("decision"),
            "alpha_factory_score": _optional_float(row.get("alpha_factory_score")),
            "cost_quality_score": _optional_float(row.get("cost_quality_score")),
            "paper_ready_block_reasons": row.get("paper_ready_block_reasons"),
        }
        key = (candidate, symbol, _optional_int(row.get("horizon_hours")))
        current = metadata.get(key)
        if current is None or _advisory_row_time(row) >= _advisory_row_time(current):
            metadata[key] = {**row, **enriched}
    return metadata


def _alpha_factory_template_family(row: dict[str, Any]) -> str:
    explicit = str(row.get("template_family") or "").strip()
    if explicit:
        return explicit
    template_name = str(row.get("template_name") or "").strip()
    if not template_name:
        return ""
    if "_v" in template_name:
        prefix, suffix = template_name.rsplit("_v", 1)
        if prefix and suffix.isdigit():
            return prefix
    return template_name


def _alpha_factory_metadata_for_row(
    row: dict[str, Any],
    metadata: dict[tuple[str, str, int | None], dict[str, Any]],
) -> dict[str, Any]:
    candidate = str(row.get("strategy_candidate") or "").strip()
    symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
    horizon = _optional_int(row.get("horizon_hours"))
    for key in (
        (candidate, symbol, horizon),
        (candidate, symbol, None),
        (candidate, "UNKNOWN", horizon),
        (candidate, "UNKNOWN", None),
    ):
        value = metadata.get(key)
        if value is not None:
            return value
    return {}


def _alpha_factory_promotion_for_row(
    row: dict[str, Any],
    promotions: dict[tuple[str, str, int | None], dict[str, Any]],
) -> dict[str, Any] | None:
    candidate = str(row.get("strategy_candidate") or "").strip()
    candidate_variants = [candidate]
    if candidate.startswith("regime_router:"):
        candidate_variants.append(candidate.split(":", 1)[1])
    symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
    horizon = _optional_int(row.get("horizon_hours"))
    for candidate_key in candidate_variants:
        for key in [
            (candidate_key, symbol, horizon),
            (candidate_key, symbol, None),
            (candidate_key, "UNKNOWN", horizon),
            (candidate_key, "UNKNOWN", None),
        ]:
            promotion = promotions.get(key)
            if promotion is not None:
                return promotion
    return None


def _is_alpha_factory_advisory_row(row: dict[str, Any]) -> bool:
    candidate = str(row.get("strategy_candidate") or "").strip()
    source_module = str(row.get("source_module") or "").strip().lower()
    if source_module == "regime_router":
        return False
    if source_module == "expanded_universe":
        return False
    if source_module == "bottom_zone_reversal":
        return False
    template_family = str(row.get("template_family") or "").strip()
    return (
        candidate in ALPHA_FACTORY_CANDIDATES
        or candidate.startswith("v5.af.")
        or candidate.startswith("v5.expanded_relative_strength")
        or source_module == "alpha_factory"
        or bool(template_family)
        or row.get("alpha_factory_score") not in (None, "")
    )


def _is_alpha_factory_promotion_capped_row(row: dict[str, Any]) -> bool:
    candidate = str(row.get("strategy_candidate") or "").strip()
    candidate_key = (
        candidate.split(":", 1)[1] if candidate.startswith("regime_router:") else candidate
    )
    source_module = str(row.get("source_module") or "").strip().lower()
    if source_module == "expanded_universe":
        return False
    if source_module == "bottom_zone_reversal":
        return False
    template_family = str(row.get("template_family") or "").strip()
    return (
        candidate_key in ALPHA_FACTORY_CANDIDATES
        or candidate_key.startswith("v5.af.")
        or candidate_key.startswith("v5.expanded_relative_strength")
        or source_module == "alpha_factory"
        or bool(template_family)
        or row.get("alpha_factory_score") not in (None, "")
    )


def _alpha_factory_promotion_decision_mode(
    promotion: dict[str, Any],
) -> tuple[str, str, list[str]]:
    state = (
        str(promotion.get("promotion_state") or promotion.get("decision") or "RESEARCH")
        .strip()
        .upper()
    )
    decision = "RESEARCH_ONLY" if state == "RESEARCH" else state
    mode = str(promotion.get("recommended_mode") or "").strip().lower()
    if not mode:
        mode = _advisory_recommended_mode(decision)
    reasons = _json_listish(promotion.get("reasons"))
    if decision != "PAPER_READY":
        reasons.append("alpha_factory_promotion_queue_not_paper_ready")
    return decision, mode, sorted(set(reasons))


def _apply_alpha_factory_promotion_overrides(
    rows: list[dict[str, Any]],
    promotions: dict[tuple[str, str, int | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if not _is_alpha_factory_promotion_capped_row(row):
            output.append(row)
            continue
        promotion = _alpha_factory_promotion_for_row(row, promotions)
        if promotion is None:
            if str(row.get("decision") or "").strip().upper() not in {
                "PAPER_READY",
                "LIVE_SMALL_READY",
            } and str(row.get("recommended_mode") or "").strip().lower() not in {
                "paper",
                "live_small",
            }:
                output.append(row)
                continue
            decision = "KEEP_SHADOW"
            mode = "shadow"
            reasons = ["alpha_factory_promotion_queue_missing"]
            promotion_state = row.get("promotion_state") or "KEEP_SHADOW"
        else:
            decision, mode, reasons = _alpha_factory_promotion_decision_mode(promotion)
            promotion_state = (
                str(promotion.get("promotion_state") or promotion.get("decision") or decision)
                .strip()
                .upper()
            )
        live_reasons = sorted({*_json_listish(row.get("live_block_reasons")), *reasons})
        paper_ready_reasons = sorted(
            {
                *_json_listish(row.get("paper_ready_block_reasons")),
                *(reason for reason in reasons if decision != "PAPER_READY"),
            }
        )
        updated = {
            **row,
            "decision": decision,
            "recommended_mode": mode,
            "source_module": row.get("source_module") or "alpha_factory",
            "template_family": row.get("template_family")
            or _alpha_factory_template_family(promotion or {}),
            "candidate_id": row.get("candidate_id") or (promotion or {}).get("candidate_id"),
            "promotion_state": promotion_state,
            "live_block_reasons": safe_json_dumps(live_reasons),
            "paper_ready_block_reasons": safe_json_dumps(paper_ready_reasons),
            "advisory_intent": _export_advisory_intent(mode),
            "max_paper_notional_usdt": _advisory_max_paper_notional(mode),
            "max_live_notional_usdt": 0.0,
            "would_enter": (
                False if mode in {"none", "research", "shadow"} else row.get("would_enter")
            ),
        }
        if mode in {"research", "shadow"} and not str(row.get("no_sample_reason") or "").strip():
            updated["no_sample_reason"] = "shadow_only" if mode == "shadow" else "research_only"
        output.append(updated)
    return output


def _apply_research_portfolio_overrides(
    rows: list[dict[str, Any]],
    overrides: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not overrides:
        return rows
    output: list[dict[str, Any]] = []
    for row in rows:
        override = _portfolio_override_for_row(row, overrides)
        decision, mode, reasons = _portfolio_overridden_decision_mode(
            decision=str(row.get("decision") or "RESEARCH_ONLY").strip().upper(),
            recommended_mode=str(row.get("recommended_mode") or "").strip(),
            portfolio_override=override,
        )
        if not reasons:
            output.append(row)
            continue
        live_reasons = sorted({*_json_listish(row.get("live_block_reasons")), *reasons})
        updated = {
            **row,
            "decision": decision,
            "recommended_mode": mode,
            "live_block_reasons": safe_json_dumps(live_reasons),
            "advisory_intent": _export_advisory_intent(mode),
            "max_paper_notional_usdt": _advisory_max_paper_notional(mode),
            "max_live_notional_usdt": 0.0,
            "would_block_if_enabled": (
                True if decision == "KILL" else row.get("would_block_if_enabled")
            ),
            "would_enter": False if mode in {"none", "research"} else row.get("would_enter"),
        }
        if mode == "research":
            no_sample_reason = str(row.get("no_sample_reason") or "research_only")
            if "research_paused" in reasons:
                no_sample_reason = "research_paused"
            elif "baseline_only" in reasons:
                no_sample_reason = "baseline_only"
            updated["no_sample_reason"] = no_sample_reason
        output.append(updated)
    return output


_ADVISORY_DECISION_PRIORITY = {
    "LIVE_SMALL_READY": 5,
    "PAPER_READY": 4,
    "REGIME_SHADOW": 3,
    "KEEP_SHADOW": 2,
    "RESEARCH_ONLY": 1,
    "KEEP_RESEARCH": 1,
    "KILL": 0,
}


def _strategy_opportunity_logical_key(row: dict[str, Any]) -> tuple[str, str, int | None, str, str]:
    return (
        str(row.get("strategy_candidate") or "").strip(),
        normalize_symbol(row.get("symbol")) or "UNKNOWN",
        _optional_int(row.get("horizon_hours")),
        str(row.get("source_module") or "").strip(),
        str(row.get("candidate_id") or "").strip(),
    )


def _strategy_opportunity_row_preferred(new: dict[str, Any], current: dict[str, Any]) -> bool:
    new_time = _advisory_row_time(new)
    current_time = _advisory_row_time(current)
    if new_time != current_time:
        return new_time > current_time
    new_priority = _ADVISORY_DECISION_PRIORITY.get(
        str(new.get("decision") or "").strip().upper(), -1
    )
    current_priority = _ADVISORY_DECISION_PRIORITY.get(
        str(current.get("decision") or "").strip().upper(), -1
    )
    if new_priority != current_priority:
        return new_priority > current_priority
    return str(new.get("source_version") or "") >= str(current.get("source_version") or "")


def _dedupe_strategy_opportunity_rows_by_logical_key(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, int | None, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _strategy_opportunity_logical_key(row)
        current = selected.get(key)
        if current is None or _strategy_opportunity_row_preferred(row, current):
            selected[key] = dict(row)
    return sorted(
        selected.values(),
        key=lambda row: (
            str(row.get("strategy_candidate") or ""),
            normalize_symbol(row.get("symbol")) or "UNKNOWN",
            _optional_int(row.get("horizon_hours")) or -1,
            str(row.get("source_module") or ""),
            str(row.get("candidate_id") or ""),
        ),
    )


def _advisory_duplicate_key_count(frame: pl.DataFrame) -> int:
    if frame.is_empty():
        return 0
    counts: dict[tuple[str, str, int | None, str, str], int] = {}
    for row in frame.to_dicts():
        key = _strategy_opportunity_logical_key(row)
        counts[key] = counts.get(key, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def _advisory_row_time(row: dict[str, Any]) -> datetime:
    for field in ["created_at", "as_of_ts", "as_of_date", "end_ts", "ts_utc"]:
        value = readers._coerce_timestamp(row.get(field))  # type: ignore[attr-defined]
        if value is not None:
            return value
    return datetime.min.replace(tzinfo=UTC)


def _advisory_as_of_ts(row: dict[str, Any]) -> str:
    value = _advisory_row_time(row)
    if value == datetime.min.replace(tzinfo=UTC):
        raw_day = str(row.get("as_of_date") or "").strip()
        if raw_day:
            return f"{raw_day[:10]}T00:00:00+00:00"
        return datetime.now(UTC).isoformat()
    return value.isoformat()


def _advisory_generated_at(row: dict[str, Any]) -> datetime:
    for field in ["generated_at", "generated_at_utc", "created_at", "as_of_ts"]:
        value = readers._coerce_timestamp(row.get(field))  # type: ignore[attr-defined]
        if value is not None:
            return value.astimezone(UTC)
    as_of = _advisory_row_time(row)
    if as_of != datetime.min.replace(tzinfo=UTC):
        return as_of.astimezone(UTC)
    return datetime.now(UTC)


def _advisory_expires_at(row: dict[str, Any], generated_at: datetime) -> datetime:
    del row
    return generated_at.astimezone(UTC) + timedelta(
        seconds=STRATEGY_OPPORTUNITY_ADVISORY_TTL_SECONDS
    )


def _v5_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    if "-" in normalized:
        base, quote, *_rest = normalized.split("-")
        return f"{base}/{quote}"
    return normalized


def _advisory_recommended_mode(decision: str) -> str:
    return {
        "PAPER_READY": "paper",
        "KEEP_SHADOW": "shadow",
        "REGIME_SHADOW": "shadow",
        "KILL": "none",
        "LIVE_SMALL_READY": "live_small",
    }.get(decision, "research")


def _export_advisory_intent(recommended_mode: str) -> str:
    mode = str(recommended_mode or "").strip().lower()
    if mode in {"paper", "shadow", "live_small"}:
        return "paper_shadow"
    return "research_only"


def _advisory_would_block_if_enabled(
    *,
    decision: str,
    recommended_mode: str,
    sample_count: int | None,
    row: dict[str, Any],
) -> bool:
    explicit = _optional_bool(row.get("would_block_if_enabled"))
    if explicit is not None:
        return explicit
    if decision == "KILL":
        return True
    return recommended_mode == "none" and (sample_count or 0) > 0


def _advisory_would_enter(
    *,
    decision: str,
    recommended_mode: str,
    sample_count: int | None,
    row: dict[str, Any],
) -> bool:
    explicit = _optional_bool(row.get("would_enter"))
    if explicit is not None:
        return explicit
    if (sample_count or 0) <= 0:
        return False
    return decision in {"PAPER_READY", "LIVE_SMALL_READY"} or recommended_mode in {
        "paper",
        "live_small",
    }


def _advisory_no_sample_reason(
    *,
    decision: str,
    recommended_mode: str,
    sample_count: int | None,
    row: dict[str, Any],
) -> str | None:
    explicit = str(row.get("no_sample_reason") or "").strip()
    if explicit:
        return explicit
    if (sample_count or 0) <= 0:
        return "no_strategy_evidence_sample"
    if decision == "KILL":
        return "killed_candidate"
    if recommended_mode == "research":
        return "research_only"
    if recommended_mode == "shadow":
        return "shadow_only"
    return None


def _entry_quality_export_no_sample_reason(
    row: dict[str, Any],
    recommended_mode: str,
) -> str | None:
    candidate = str(row.get("strategy_candidate") or "").strip()
    if candidate == "v5.entry_quality_missed_low_audit":
        return "audit_only"
    if candidate == "v5.late_entry_chase_guard_shadow":
        return "guard_shadow_only"
    if recommended_mode == "paper":
        return "not_live_validated"
    if recommended_mode == "shadow":
        return "shadow_only_collect_more_samples"
    if recommended_mode == "research":
        return "research_only"
    return None


def _advisory_cost_quality(cost_source_mix: Any, latest_cost_health: dict[str, Any]) -> str:
    sources = {source.lower() for source in _cost_source_mix_items(cost_source_mix)}
    if not sources:
        status = str(latest_cost_health.get("status") or "").strip().upper()
        return "unknown" if not status else f"unknown_cost_mix_health_{status.lower()}"
    if "global_default" in sources:
        return "global_default"
    if any("mixed_actual_proxy" in source or "actual" in source for source in sources):
        return "actual_or_mixed"
    if any(source in {"public_spread_proxy", "public_proxy"} for source in sources):
        return "public_spread_proxy"
    if "cost_not_requested_no_order" in sources:
        return "cost_not_requested_no_order"
    return "unknown"


def _latest_cost_health_context(cost_health: pl.DataFrame) -> dict[str, Any]:
    if cost_health.is_empty():
        return {}
    return max(cost_health.to_dicts(), key=_advisory_row_time)


def _advisory_risk_context(risk_permissions: pl.DataFrame) -> dict[str, Any]:
    if risk_permissions.is_empty():
        return {
            "permission": "UNKNOWN",
            "permission_status": "NO_FRESH_PERMISSION",
            "enforceable": False,
            "max_single_order_usdt": 0.0,
        }
    row = max(risk_permissions.to_dicts(), key=_advisory_row_time)
    return {
        "permission": str(row.get("permission") or "ABORT").strip().upper(),
        "permission_status": str(row.get("permission_status") or "").strip().upper(),
        "enforceable": bool(row.get("enforceable")),
        "max_single_order_usdt": _optional_float(row.get("max_single_order_usdt")) or 0.0,
    }


def _advisory_live_block_reasons(
    *,
    row: dict[str, Any],
    proposal: dict[str, Any],
    decision: str,
    cost_quality: str,
    paper_days: int,
    entry_day_count: int,
    paper_pnl_observed_count: int,
    slippage_coverage: float | None,
    risk_context: dict[str, Any],
) -> list[str]:
    reasons = set(_json_listish(row.get("decision_reasons")))
    reasons.update(_json_listish(proposal.get("live_block_reason")))
    if str(row.get("universe_type") or "") == "expanded_paper":
        reasons.add("expanded_universe_not_live_approved")
    if decision != "LIVE_SMALL_READY":
        reasons.add(f"decision_{decision.lower()}")
    if cost_quality not in {"actual_or_mixed"}:
        reasons.add("cost_source_not_actual_or_mixed")
    if paper_days < 14:
        reasons.add("no_paper_days")
    if entry_day_count <= 0:
        reasons.add("no_paper_entries")
    if paper_pnl_observed_count <= 0:
        reasons.add("no_paper_pnl_observations")
    if slippage_coverage is None or slippage_coverage < 0.8:
        reasons.add("no_live_slippage_coverage")
    reasons.add("quant_lab_live_command_not_allowed")
    reasons.add("v5_local_live_not_controlled_by_quant_lab")
    if risk_context.get("permission") != "ALLOW" or not risk_context.get("enforceable"):
        reasons.add("quant_lab_advisory_permission_not_allow")
    if decision == "KILL":
        reasons.add("candidate_killed")
    return sorted(reason for reason in reasons if reason)


def _json_listish(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return _json_listish(parsed)
    return [str(value)]


def _advisory_max_paper_notional(recommended_mode: str) -> float:
    return 100.0 if recommended_mode == "paper" else 0.0


def _advisory_max_live_notional(
    *,
    decision: str,
    risk_context: dict[str, Any],
    live_block_reasons: list[str],
) -> float:
    if decision != "LIVE_SMALL_READY" or live_block_reasons:
        return 0.0
    if risk_context.get("permission") != "ALLOW" or not risk_context.get("enforceable"):
        return 0.0
    return max(float(risk_context.get("max_single_order_usdt") or 0.0), 0.0)


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


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _fmt_float(value: Any, *, digits: int = 2) -> str:
    number = _optional_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _factor_factory_summary_md(
    candidates: pl.DataFrame,
    evidence: pl.DataFrame,
    correlations: pl.DataFrame,
) -> str:
    lines = [
        "# Factor Factory Summary",
        "",
        "Factor Factory is read-only research telemetry. It publishes factor definitions, "
        "values, evidence, candidates, and correlation diagnostics for manual review.",
        "",
        "- live_order_effect: none_read_only_research",
        "- historical_label_threshold_ready: historical label threshold only, "
        "not formal paper-ready",
        "- candidate_paper_ready: requires paper PnL, arrival-mid coverage, "
        "and actual/mixed cost evidence",
        "- decision_delay_bars: at least one closed bar before label evaluation",
        "",
    ]
    if candidates.is_empty():
        lines.extend(
            [
                "No factor candidates are available yet.",
                "",
                "Next action: run `qlab publish-features` and then "
                "`qlab build-factor-factory --apply` after confirming feature coverage.",
            ]
        )
        return "\n".join(lines) + "\n"

    counts = _value_counts(candidates, "candidate_state")
    paper_ready_count = counts.get("PAPER_READY", 0)
    lines.extend(
        [
            f"Candidate state counts: {safe_json_dumps(counts)}",
            f"PAPER_READY count: {paper_ready_count}",
            "",
            "| Factor | Family | State | Horizon | Score | Rank IC | Long-short bps | Action |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    sort_columns = [
        column
        for column in ["candidate_state", "best_score", "factor_id"]
        if column in candidates.columns
    ]
    ordered = candidates.sort(sort_columns, descending=[False, True, False][: len(sort_columns)])
    for row in ordered.head(20).to_dicts():
        lines.append(
            "| {factor} | {family} | {state} | {horizon} | {score} | {rank_ic} | "
            "{spread} | {action} |".format(
                factor=row.get("factor_id", ""),
                family=row.get("factor_family", ""),
                state=row.get("candidate_state", ""),
                horizon=row.get("best_horizon_bars", ""),
                score=_fmt_float(row.get("best_score"), digits=3),
                rank_ic=_fmt_float(row.get("best_rank_ic_mean"), digits=4),
                spread=_fmt_float(row.get("best_long_short_mean_bps"), digits=2),
                action=row.get("recommended_action", ""),
            )
        )
    lines.append("")
    if not evidence.is_empty():
        decision_counts = _value_counts(evidence, "decision")
        lines.extend(
            [
                f"Evidence decision counts by horizon: {safe_json_dumps(decision_counts)}",
                "",
            ]
        )
    high_corr_count = 0
    if not correlations.is_empty() and "correlation" in correlations.columns:
        high_corr_count = correlations.filter(
            pl.col("correlation").cast(pl.Float64, strict=False).abs() >= 0.90
        ).height
    lines.extend(
        [
            f"High-correlation pair count (abs >= 0.90): {high_corr_count}",
            "",
            "Manual review should suppress redundant factors before any paper proposal is "
            "promoted.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    manifest_ts = _latest_dataset_timestamp(
        "v5_bundle_manifest",
        frames.get("v5_bundle_manifest", pl.DataFrame()),
    )
    if manifest_ts is not None:
        return manifest_ts
    return _latest_v5_derived_output_ts(frames)


def _latest_v5_derived_output_ts(frames: dict[str, pl.DataFrame]) -> datetime | None:
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
    telemetry_latest_ts: datetime | None = None,
) -> dict[str, Any]:
    checked_at = reference_at or datetime.now(UTC)
    telemetry_latest_ts = telemetry_latest_ts or _latest_v5_bundle_ts(frames)
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
            "run qlab publish-risk-permission after sync-v5-telemetry" if stale else "none"
        ),
    }


def _risk_permissions_for_export(
    risk: pl.DataFrame,
    frames: dict[str, pl.DataFrame],
    *,
    reference_at: datetime | None = None,
    telemetry_latest_ts: datetime | None = None,
) -> pl.DataFrame:
    if risk.is_empty():
        return risk
    checked_at = reference_at or datetime.now(UTC)
    telemetry_latest_ts = telemetry_latest_ts or _latest_v5_bundle_ts(frames)
    rows: list[dict[str, Any]] = []
    for row in _current_risk_permission_rows(risk.to_dicts()):
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


def _current_risk_permission_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formal_strategies = {
        str(row.get("strategy") or "").strip()
        for row in rows
        if not _is_bootstrap_risk_permission(row)
    }
    if not formal_strategies:
        return rows
    return [
        row
        for row in rows
        if not (
            _is_bootstrap_risk_permission(row)
            and str(row.get("strategy") or "").strip() in formal_strategies
        )
    ]


def _is_bootstrap_risk_permission(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(column) or "").strip().lower()
        for column in ["version", "gate_version", "source", "fallback_level"]
    )
    return "bootstrap" in text


def _risk_permission_export_status(
    frames: dict[str, pl.DataFrame],
    generated_at: datetime,
) -> dict[str, Any]:
    telemetry_latest_ts = _latest_v5_bundle_ts(frames)
    risk = _risk_permissions_for_export(
        frames.get("risk_permission", pl.DataFrame()),
        frames,
        reference_at=generated_at,
        telemetry_latest_ts=telemetry_latest_ts,
    )
    quality = _risk_permission_quality(
        risk,
        frames,
        reference_at=generated_at,
        telemetry_latest_ts=telemetry_latest_ts,
    )
    return {
        "risk_permission_generated_at": quality.get("risk_permission_generated_at"),
        "risk_permission_expires_at": quality.get("risk_permission_expires_at"),
        "export_generated_at": quality.get("export_generated_at"),
        "permission_expired_at_export": quality.get("permission_expired_at_export"),
        "permission_status": quality.get("permission_status"),
        "next_action": quality.get("next_action"),
    }


def _registry_quality_dataset_names(snapshot: _DatasetSnapshot) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    skipped_heavy: list[str] = []
    for name, row_count in sorted(snapshot.row_counts.items()):
        if row_count <= 0:
            continue
        if name in HEAVY_REGISTRY_QUALITY_DATASETS and _can_skip_heavy_registry_quality(
            name,
            snapshot.row_counts,
        ):
            skipped_heavy.append(name)
            continue
        selected.append(name)
    return selected, skipped_heavy


def _can_skip_heavy_registry_quality(
    dataset_name: str,
    row_counts: dict[str, int],
) -> bool:
    if dataset_name == "trade_print":
        return row_counts.get("trade_activity_1m", 0) > 0
    if dataset_name == "orderbook_snapshot":
        return row_counts.get("orderbook_spread_1m", 0) > 0
    if dataset_name == "okx_public_ws":
        return row_counts.get("okx_public_ws_health", 0) > 0 and (
            row_counts.get("trade_activity_1m", 0) > 0
            or row_counts.get("orderbook_spread_1m", 0) > 0
        )
    return False


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


def _v5_candidate_required_feature_completeness(row: dict[str, Any]) -> float:
    value = row.get("required_feature_completeness")
    if value not in {None, ""}:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
    by_field = _json_payload(row.get("required_feature_completeness_by_field_json"))
    if not by_field:
        by_field = _json_payload(row.get("feature_completeness_by_field_json"))
    required_values = []
    for field in ("final_score", "expected_edge_bps", "required_edge_bps"):
        try:
            required_values.append(float(by_field[field]))
        except (KeyError, TypeError, ValueError):
            required_values = []
            break
    if required_values:
        return float(sum(required_values) / len(required_values))
    value = row.get("feature_completeness")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _frame_values(frame: pl.DataFrame, column: str) -> set[str]:
    if frame.is_empty() or column not in frame.columns:
        return set()
    return {
        str(value).strip() for value in frame[column].drop_nulls().to_list() if str(value).strip()
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
    return _okx_ws_universe_completeness(frames)[0]


def _okx_ws_universe_completeness(frames: dict[str, pl.DataFrame]) -> tuple[bool, str]:
    expected = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"}
    observed_by_source = {
        "trade_print": _symbols(frames.get("trade_print", pl.DataFrame())),
        "orderbook_snapshot": _symbols(frames.get("orderbook_snapshot", pl.DataFrame())),
        "trade_activity_1m": _symbols(frames.get("trade_activity_1m", pl.DataFrame())),
        "orderbook_spread_1m": _symbols(frames.get("orderbook_spread_1m", pl.DataFrame())),
    }
    observed = set().union(*observed_by_source.values())
    missing = sorted(expected.difference(observed))
    detail = (
        f"expected={len(expected)}; "
        f"observed={len(expected.intersection(observed))}; "
        f"missing={missing}; "
        "sources="
        + ",".join(f"{name}:{len(symbols)}" for name, symbols in observed_by_source.items())
    )
    return not missing, detail


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
                pl.col("ts")
                .cast(pl.Datetime(time_zone="UTC"))
                .dt.date()
                .cast(pl.Utf8)
                .alias("day"),
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
            if str(row.get("source") or row.get("cost_source") or "").lower() == "global_default"
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
                    "reason": "stale_permission"
                    if stale
                    else "quant_lab_advisory_risk_flag_not_v5_live_command",
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
        if frame.is_empty() and not _empty_frame_covered_by_rollup(name, frames)
    ]
    return (
        pl.DataFrame(rows) if rows else pl.DataFrame(schema={"dataset": pl.Utf8, "reason": pl.Utf8})
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


def _empty_frame_covered_by_rollup(
    dataset_name: str,
    frames: dict[str, pl.DataFrame],
) -> bool:
    rollup_candidates = {
        "trade_print": ("trade_activity_1m",),
        "orderbook_snapshot": ("orderbook_spread_1m",),
        "okx_public_ws": (
            "okx_public_ws_health",
            "trade_activity_1m",
            "orderbook_spread_1m",
        ),
    }.get(dataset_name, ())
    return any(not frames.get(name, pl.DataFrame()).is_empty() for name in rollup_candidates)


def _derived_latest_status_from_frames(
    dataset_name: str,
    status: str,
    frames: dict[str, pl.DataFrame],
) -> str:
    source_dataset = DERIVED_LATEST_SOURCE_DATASETS.get(dataset_name)
    if not source_dataset or status != "stale":
        return status
    source_frame = frames.get(source_dataset, pl.DataFrame())
    if source_frame.is_empty():
        return status
    source_freshness = _dataset_freshness_payload(source_dataset, source_frame)
    source_status = str(source_freshness.get("freshness_status") or "")
    if source_status in {"fresh", "delayed"}:
        return DERIVED_LATEST_SOURCE_CURRENT_STATUS
    return status


def _stale_rows(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    v5_telemetry_is_current = _v5_telemetry_is_current_from_frames(frames)
    okx_readonly_private_is_current = _okx_readonly_private_is_current_from_frames(frames)
    expanded_universe_automation_is_active = _expanded_universe_automation_is_active_from_frames(
        frames
    )
    closed_research_keys = readers.research_portfolio_closed_keys(
        frames.get("research_portfolio_status", pl.DataFrame())
    )
    for name, frame in sorted(frames.items()):
        freshness = _dataset_freshness_payload(name, frame)
        status = freshness["freshness_status"]
        if frame.is_empty():
            empty_status = readers._empty_dataset_status(name)  # type: ignore[attr-defined]
            if (
                empty_status in readers.OPTIONAL_EMPTY_DATASET_STATUSES
                or empty_status in EVENT_DRIVEN_OK_STATUSES
            ):
                continue
            if _empty_frame_covered_by_rollup(name, frames):
                continue
            if name == "expanded_crypto_universe_shadow" and expanded_universe_automation_is_active:
                continue
        if readers._dataset_belongs_to_closed_research(name, closed_research_keys):  # type: ignore[attr-defined]
            status = "closed_research_snapshot"
        if name in EVENT_DRIVEN_V5_DATASETS and status == "stale" and v5_telemetry_is_current:
            status = EVENT_DRIVEN_V5_DATASETS[name]
        if (
            name in EVENT_DRIVEN_OKX_READONLY_DATASETS
            and status == "stale"
            and okx_readonly_private_is_current
        ):
            status = EVENT_DRIVEN_OKX_READONLY_DATASETS[name]
        if name in readers.HISTORICAL_RESEARCH_DATASETS and status == "stale":
            status = "historical_research_snapshot"
        status = _derived_latest_status_from_frames(name, status, frames)
        status = readers._optional_stale_status_from_registry(name, status)  # type: ignore[attr-defined]
        if (
            status in {"missing", "unknown", "stale", "future"}
            and status not in EVENT_DRIVEN_OK_STATUSES
        ):
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


def _expanded_universe_automation_is_active_from_frames(frames: dict[str, pl.DataFrame]) -> bool:
    for dataset_name in (
        "expanded_universe_candidate",
        "expanded_universe_quality",
        "expanded_universe_candidate_event",
        "expanded_universe_watchlist",
    ):
        frame = frames.get(dataset_name, pl.DataFrame())
        if frame.is_empty():
            continue
        status = str(_dataset_freshness_payload(dataset_name, frame)["freshness_status"] or "")
        if status in {"fresh", "delayed"}:
            return True
    return False


def _v5_telemetry_is_current_from_frames(frames: dict[str, pl.DataFrame]) -> bool:
    health = frames.get("strategy_health_daily", pl.DataFrame())
    if health.is_empty():
        return False
    freshness = _dataset_freshness_payload("strategy_health_daily", health)
    return str(freshness.get("freshness_status") or "") in {"fresh", "delayed"}


def _okx_readonly_private_is_current_from_frames(frames: dict[str, pl.DataFrame]) -> bool:
    job_history = frames.get("job_run_history", pl.DataFrame())
    if _recent_successful_job_from_frame(job_history, "okx-backfill-readonly"):
        return True
    if _recent_successful_job_from_frame(job_history, "okx-fetch-bills"):
        return True
    for dataset_name in ("okx_private_readonly_fills", "okx_private_readonly_bills"):
        frame = frames.get(dataset_name, pl.DataFrame())
        if frame.is_empty():
            continue
        freshness = _dataset_freshness_payload(dataset_name, frame)
        if str(freshness.get("freshness_status") or "") in {"fresh", "delayed"}:
            return True
    return False


def _recent_successful_job_from_frame(
    frame: pl.DataFrame,
    job_name: str,
    *,
    max_age_seconds: int = 24 * 60 * 60,
) -> bool:
    if frame.is_empty() or not {"job_name", "status", "finished_at"}.issubset(frame.columns):
        return False
    try:
        filtered = frame.filter(
            (pl.col("job_name").cast(pl.Utf8) == job_name)
            & (pl.col("status").cast(pl.Utf8).str.to_lowercase() == "succeeded")
        )
    except Exception:
        return False
    latest = _timestamp_value("job_run_history", filtered, "max")
    if latest is None:
        return False
    age_seconds = max(int((datetime.now(UTC) - latest).total_seconds()), 0)
    return age_seconds <= max_age_seconds


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
    gates = _non_bootstrap_gate_rows(gates)
    if gates.is_empty():
        return True
    keys = [
        key for key in ["alpha_id", "version"] if key in gates.columns and key in evidence.columns
    ]
    if not keys:
        return False
    joined = gates.join(evidence.select(keys).unique(), on=keys, how="anti")
    return joined.is_empty()


def _non_bootstrap_gate_rows(gates: pl.DataFrame) -> pl.DataFrame:
    filtered = gates
    for column in ["version", "source", "fallback_level"]:
        if column not in filtered.columns:
            continue
        filtered = filtered.filter(
            ~pl.col(column).fill_null("").cast(pl.Utf8).str.to_lowercase().str.contains("bootstrap")
        )
    if "alpha_id" in filtered.columns and "version" in filtered.columns:
        filtered = filtered.filter(
            ~(
                pl.col("alpha_id").fill_null("").cast(pl.Utf8).str.ends_with(".core")
                & (pl.col("version").fill_null("").cast(pl.Utf8).str.to_lowercase() == "bootstrap")
            )
        )
    return filtered


def _prefer_frame(primary: pl.DataFrame, fallback: pl.DataFrame) -> pl.DataFrame:
    return primary if not primary.is_empty() else fallback


def _normalize_final_score_alpha6_conflict_frame(frame: pl.DataFrame) -> pl.DataFrame:
    path = "reports/final_score_vs_alpha6_conflict.csv"
    if frame.is_empty():
        return _empty_csv_schema_frame(path)
    normalized = frame
    rename_map = {
        "f3": "f3_vol_adj_ret",
        "f4": "f4_volume_expansion",
        "f5": "f5_rsi_trend_confirm",
        "negexp_net_expectancy_bps": "negative_expectancy_net_bps",
        "negexp_fast_fail_net_expectancy_bps": "negative_expectancy_fast_fail_net_bps",
    }
    for old, new in rename_map.items():
        if old in normalized.columns and new not in normalized.columns:
            normalized = normalized.rename({old: new})
    return _csv_frame_with_schema(normalized, CSV_SCHEMAS[path])


def _normalize_bnb_strong_alpha6_bypass_frame(frame: pl.DataFrame) -> pl.DataFrame:
    path = "reports/bnb_strong_alpha6_bypass_shadow.csv"
    if frame.is_empty():
        return _empty_csv_schema_frame(path)
    normalized = frame
    if (
        "negative_expectancy_blocked" not in normalized.columns
        and "would_bypass_negative_expectancy" in normalized.columns
    ):
        normalized = normalized.rename(
            {"would_bypass_negative_expectancy": "negative_expectancy_blocked"}
        )
    if "would_bypass" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit(True).alias("would_bypass"))
    if "live_order_effect" not in normalized.columns:
        normalized = normalized.with_columns(
            pl.lit("read_only_no_live_order").alias("live_order_effect")
        )
    return _csv_frame_with_schema(normalized, CSV_SCHEMAS[path])


def _strategy_evidence_has_required_candidates(strategy_evidence: pl.DataFrame) -> bool:
    if strategy_evidence.is_empty() or "candidate_name" not in strategy_evidence.columns:
        return False
    # Keep this list to active V5 candidate families with current telemetry or
    # historical outcome sources. Do not include dormant/planned candidates here:
    # otherwise daily export reports a false coverage warning even when V5 is not
    # emitting those candidates.
    required = {
        "v5.sol_protect_exception",
        "v5.alt_impulse_shadow",
        "v5.swing_f4_f5_alpha6",
        "v5.f3_dominant_entry",
        "v5.f4_volume_expansion_entry",
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
                    "v5.btc_leadership_blocked_relaxed",
                    "v5.btc_leadership_alpha6_low_blocked",
                    "v5.btc_leadership_f5_low_blocked",
                    "v5.btc_leadership_no_breakout_blocked",
                    "v5.multi_position_k1",
                    "v5.multi_position_k2",
                    "v5.multi_position_k3",
                    "v5.portfolio_trend_following",
                    "v5.pullback_reversal_shadow",
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
    return (
        required.issubset(risk.columns)
        and risk.filter(
            pl.any_horizontal([pl.col(column).is_null() for column in sorted(required)])
        ).is_empty()
    )


def _missing_sections(row_counts: dict[str, int]) -> list[str]:
    missing: list[str] = []
    for section, datasets in SECTION_DATASETS.items():
        if all(row_counts.get(name, 0) == 0 for name in datasets):
            missing.append(section)
    return missing


def _write_zip(path: Path, members: dict[str, _MemberPayload]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in sorted(members.items()):
            with archive.open(member, "w") as handle:
                for chunk in _payload_byte_chunks(payload):
                    handle.write(chunk)


def _fail_on_secrets(members: dict[str, _MemberPayload]) -> None:
    findings: list[str] = []
    for member, text in members.items():
        if isinstance(text, bytes):
            continue
        high, medium = _secret_severity_counts(text)
        if high or medium:
            findings.append(f"{member}: {high} high, {medium} medium")
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
            reasons.append(f"{name} contains possible secrets: {high} high, {medium} medium")
    return reasons


def _secret_severity_counts(text: str) -> tuple[int, int]:
    high = 0
    medium = 0
    for line in _iter_text_lines(text):
        if not _line_may_contain_secret(line):
            continue
        for pattern, severity, _label in SECRET_PATTERNS:
            if not pattern.search(line):
                continue
            if severity == "high":
                high += 1
            elif severity == "medium":
                medium += 1
    return high, medium


def _line_may_contain_secret(line: str) -> bool:
    lowered = line.lower()
    return any(token in lowered for token in SECRET_SCAN_PREFILTER_TOKENS)


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


def _entry_quality_json(df: pl.DataFrame) -> dict[str, Any]:
    row_count = 0 if df.is_empty() else df.height
    if not df.is_empty() and "row_count" in df.columns:
        counts = [
            value
            for value in (_optional_int(item) for item in df["row_count"].to_list())
            if value is not None
        ]
        if counts:
            row_count = max(row_count, max(counts))
    return {
        "rows": _rows(df),
        "row_count": row_count,
        "source": "quant_lab",
        "mode": "advisory",
    }


def _pullback_shadow_aggregate_for_export(
    pullback: pl.DataFrame,
    *,
    group_type: str,
    path: str,
) -> pl.DataFrame:
    if pullback.is_empty():
        return _empty_csv_schema_frame(path)
    rows = pullback.to_dicts()
    if group_type == "symbol":
        keys = sorted({str(row.get("symbol") or "") for row in rows if row.get("symbol")})
    elif group_type == "regime":
        keys = sorted({str(row.get("regime_state") or "UNKNOWN") for row in rows})
    elif group_type == "horizon":
        keys = sorted(
            {
                int(value)
                for value in (_optional_int(row.get("horizon_hours")) for row in rows)
                if value is not None
            }
        )
    else:
        return _empty_csv_schema_frame(path)
    output: list[dict[str, Any]] = []
    ts_values = [
        parsed
        for parsed in (_parse_export_ts(row.get("ts_utc")) for row in rows)
        if parsed is not None
    ]
    start_date = min(ts_values).date().isoformat() if ts_values else None
    end_date = max(ts_values).date().isoformat() if ts_values else None
    for key in keys:
        if group_type == "symbol":
            group = [row for row in rows if str(row.get("symbol") or "") == key]
        elif group_type == "regime":
            group = [row for row in rows if str(row.get("regime_state") or "UNKNOWN") == key]
        else:
            group = [row for row in rows if _optional_int(row.get("horizon_hours")) == key]
        complete = [
            row for row in group if str(row.get("label_status") or "").strip().lower() == "complete"
        ]
        net_values = [
            value
            for value in (_optional_float(row.get("net_bps_after_cost")) for row in complete)
            if value is not None
        ]
        mfe_values = [
            value
            for value in (_optional_float(row.get("mfe_bps")) for row in complete)
            if value is not None
        ]
        mae_values = [
            value
            for value in (_optional_float(row.get("mae_bps")) for row in complete)
            if value is not None
        ]
        first = group[0]
        avg_net = _float_mean(net_values)
        win_rate = (
            sum(1 for value in net_values if value > 0.0) / len(net_values) if net_values else None
        )
        decision = "RESEARCH_ONLY"
        reasons = ["v5_bundle_pullback_import"]
        if len(net_values) >= 5 and avg_net is not None:
            if avg_net > 0 and (win_rate or 0.0) >= 0.5:
                decision = "KEEP_SHADOW"
            elif avg_net < 0 and (win_rate or 0.0) < 0.45:
                decision = "KILL"
        output.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "window_mode": "v5_bundle",
                "cost_mode": "reported",
                "group_type": group_type,
                "group_key": str(key),
                "strategy_candidate": first.get("strategy_candidate")
                or "v5.pullback_reversal_shadow",
                "symbol": first.get("symbol") if group_type != "horizon" else "ALL",
                "regime_state": first.get("regime_state") if group_type != "horizon" else "ALL",
                "horizon_hours": key if group_type == "horizon" else 0,
                "sample_count": len(group),
                "complete_sample_count": len(net_values),
                "avg_net_bps": avg_net,
                "median_net_bps": _float_quantile(net_values, 0.5),
                "p25_net_bps": _float_quantile(net_values, 0.25),
                "win_rate": win_rate,
                "avg_mfe_bps": _float_mean(mfe_values),
                "avg_mae_bps": _float_mean(mae_values),
                "cost_quality_mix": safe_json_dumps(
                    dict(Counter(str(row.get("cost_quality") or "reported") for row in group))
                ),
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
                "mode": "shadow",
                "generated_at_utc": first.get("generated_at_utc"),
                "schema_version": first.get("schema_version"),
                "contract_version": first.get("contract_version"),
            }
        )
    return pl.DataFrame(output, infer_schema_length=None).select(CSV_SCHEMAS[path])


def _parse_export_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if value is None or value == "":
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _with_cost_probe_authorization_export_freshness(
    frame: pl.DataFrame,
    *,
    reference_at: datetime,
) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return frame
    reference = reference_at.astimezone(UTC)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        normalized = dict(row)
        issued_at = _parse_export_ts(row.get("authorization_issued_at"))
        expires_at = _parse_export_ts(row.get("authorization_expires_at"))
        if issued_at is None or expires_at is None:
            normalized.update(
                {
                    "authorization_fresh_at_export": None,
                    "authorization_age_sec_at_export": None,
                    "authorization_seconds_to_expiry": None,
                    "authorization_expired_at_export": None,
                }
            )
        else:
            normalized.update(
                {
                    "authorization_fresh_at_export": issued_at <= reference <= expires_at,
                    "authorization_age_sec_at_export": max(
                        int((reference - issued_at).total_seconds()),
                        0,
                    ),
                    "authorization_seconds_to_expiry": int(
                        (expires_at - reference).total_seconds()
                    ),
                    "authorization_expired_at_export": reference > expires_at,
                }
            )
        rows.append(normalized)
    return pl.DataFrame(rows, infer_schema_length=None)


def _pullback_v2_by_symbol_for_export(pullback: pl.DataFrame) -> pl.DataFrame:
    path = "reports/pullback_reversal_v2_by_symbol.csv"
    if pullback.is_empty():
        return _empty_csv_schema_frame(path)
    required = {"rule_version", "symbol", "horizon_hours"}
    if not required.issubset(pullback.columns):
        return _empty_csv_schema_frame(path)
    rows = [
        row
        for row in pullback.to_dicts()
        if str(row.get("rule_version") or "") == "confirmed_reversal_v0.2"
        and _optional_int(row.get("horizon_hours")) == 24
    ]
    if not rows:
        return _empty_csv_schema_frame(path)
    output: list[dict[str, Any]] = []
    symbols = sorted({str(row.get("symbol") or "") for row in rows if row.get("symbol")})
    for symbol in symbols:
        group = [row for row in rows if str(row.get("symbol") or "") == symbol]
        complete = [
            row for row in group if str(row.get("label_status") or "").strip().lower() == "complete"
        ]
        net_values = [
            value
            for value in (_optional_float(row.get("net_bps_after_cost")) for row in complete)
            if value is not None
        ]
        mae_values = [
            value
            for value in (_optional_float(row.get("mae_bps")) for row in complete)
            if value is not None
        ]
        avg_net = _float_mean(net_values)
        win_rate = (
            sum(1 for value in net_values if value > 0.0) / len(net_values) if net_values else None
        )
        p25 = _float_quantile(net_values, 0.25)
        avg_mae = _float_mean(mae_values)
        decision, reasons = _pullback_v2_decision(
            sample_count=len(group),
            complete_sample_count=len(net_values),
            avg_net_bps=avg_net,
            win_rate=win_rate,
            p25_net_bps=p25,
            avg_mae_bps=avg_mae,
        )
        first = group[0]
        output.append(
            {
                "as_of_date": first.get("as_of_date"),
                "rule_version": "confirmed_reversal_v0.2",
                "strategy_candidate": first.get("strategy_candidate"),
                "symbol": symbol,
                "sample_count": len(group),
                "complete_sample_count": len(net_values),
                "avg_24h_net_bps": avg_net,
                "win_rate_24h": win_rate,
                "p25_24h_net_bps": p25,
                "avg_mae_bps": avg_mae,
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
                "mode": "shadow",
                "generated_at_utc": first.get("generated_at_utc"),
                "schema_version": first.get("schema_version"),
                "contract_version": first.get("contract_version"),
            }
        )
    return pl.DataFrame(output, infer_schema_length=None).select(CSV_SCHEMAS[path])


def _pullback_v2_decision(
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    win_rate: float | None,
    p25_net_bps: float | None,
    avg_mae_bps: float | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = ["not_live_validated", "paper_disabled_until_more_evidence"]
    if sample_count < 50:
        reasons.append("insufficient_sample_count")
    if complete_sample_count < 10:
        reasons.append("insufficient_complete_samples")
    if avg_net_bps is None or avg_net_bps <= 0.0:
        reasons.append("weak_24h_avg_net_bps")
    if win_rate is None or win_rate <= 0.45:
        reasons.append("weak_24h_win_rate")
    if p25_net_bps is None or p25_net_bps <= -50.0:
        reasons.append("weak_24h_p25_net_bps")
    if avg_mae_bps is None or avg_mae_bps <= -120.0:
        reasons.append("excessive_avg_mae_bps")
    if len(reasons) > 2:
        return "RESEARCH_ONLY", reasons
    return "KEEP_SHADOW", ["v2_not_broadly_negative", *reasons]


def _float_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _float_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _expanded_recommendations_json(df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty():
        return {
            "rows": [],
            "row_count": 0,
            "source": "quant_lab",
            "mode": "shadow_research",
            "min_stable_output_days": 7,
        }
    rows = _rows(df)
    latest = rows[-1]
    return {
        "rows": rows,
        "latest": latest,
        "row_count": len(rows),
        "source": "quant_lab",
        "mode": "shadow_research",
        "min_stable_output_days": latest.get("min_stable_output_days", 7),
    }


def _expanded_universe_daily_md(
    *,
    candidates: pl.DataFrame,
    quality: pl.DataFrame,
    events: pl.DataFrame,
    labels: pl.DataFrame,
    promotion_queue: pl.DataFrame,
    maturity: pl.DataFrame,
    watchlist: pl.DataFrame,
) -> str:
    lines = [
        "# Expanded Crypto Universe Automation",
        "",
        "该报告只做 research/shadow/paper 研究，不自动加入 V5 live symbols。",
        "",
        f"- candidates: {candidates.height}",
        f"- quality rows: {quality.height}",
        f"- candidate events: {events.height}",
        f"- labels: {labels.height}",
        f"- promotion queue rows: {promotion_queue.height}",
        f"- maturity rows: {maturity.height}",
        f"- watchlist rows: {watchlist.height}",
        "",
        "## Promotion counts",
    ]
    counts = _value_counts(promotion_queue, "promotion_state")
    if counts:
        lines.extend(f"- {state}: {count}" for state, count in sorted(counts.items()))
    else:
        lines.append("- no promotion rows")
    lines.extend(
        [
            "",
            "## Safety",
            "- max_live_notional_usdt is forced to 0 for expanded universe rows.",
            "- LIVE_SMALL_CANDIDATE requires manual enablement and is not emitted in stage one.",
        ]
    )
    return "\n".join(lines) + "\n"


def _expanded_universe_promotion_summary_md(
    *,
    maturity: pl.DataFrame,
    watchlist: pl.DataFrame,
    promotion_queue: pl.DataFrame,
) -> str:
    lines = [
        "# Expanded Universe Promotion Summary",
        "",
        "扩展币池只做 research/shadow/paper，不自动替换 ETH/BNB，也不改变 V5 live symbols。",
        "",
        "## Maturity counts",
    ]
    maturity_counts = _value_counts(maturity, "maturity_state")
    if maturity_counts:
        lines.extend(f"- {state}: {count}" for state, count in sorted(maturity_counts.items()))
    else:
        lines.append("- no maturity rows")
    lines.extend(["", "## Watchlists"])
    watch_counts = _value_counts(watchlist, "watchlist_type")
    if watch_counts:
        lines.extend(f"- {kind}: {count}" for kind, count in sorted(watch_counts.items()))
    else:
        lines.append("- no watchlist rows")
    lines.extend(["", "## Promotion states"])
    promotion_counts = _value_counts(promotion_queue, "promotion_state")
    if promotion_counts:
        lines.extend(f"- {state}: {count}" for state, count in sorted(promotion_counts.items()))
    else:
        lines.append("- no promotion rows")
    lines.extend(
        [
            "",
            "## Rules",
            "- complete_sample_count < 10: RESEARCH.",
            "- complete_sample_count >= 10 and at least two positive short horizons: KEEP_SHADOW.",
            "- complete_sample_count >= 30, win_rate > 55%, p25_net_bps > -50: PAPER_READY.",
            "- LIVE_SMALL_READY is not emitted by expanded universe automation.",
            "- expanded universe does not output ETH/BNB replacement advice; all rows remain "
            "research/shadow/paper only.",
        ]
    )
    return "\n".join(lines) + "\n"


def _expanded_promotion_rows(frame: pl.DataFrame, state: str) -> pl.DataFrame:
    path = "reports/expanded_universe_promotion_queue.csv"
    if frame.is_empty() or "promotion_state" not in frame.columns:
        return _empty_csv_schema_frame(path)
    filtered = frame.filter(pl.col("promotion_state") == state)
    return filtered if not filtered.is_empty() else _empty_csv_schema_frame(path)


def _expanded_replacement_rows(frame: pl.DataFrame) -> pl.DataFrame:
    path = "reports/expanded_universe_replacement_candidates.csv"
    if frame.is_empty():
        return _empty_csv_schema_frame(path)
    if "replacement_target_candidate" not in frame.columns:
        return _empty_csv_schema_frame(path)
    filtered = frame.filter(
        pl.col("replacement_target_candidate").fill_null("").cast(pl.Utf8).str.len_chars() > 0
    )
    return filtered if not filtered.is_empty() else _empty_csv_schema_frame(path)


def _expanded_strategy_evidence_rows(strategy_evidence: pl.DataFrame) -> pl.DataFrame:
    path = "reports/expanded_universe_strategy_evidence.csv"
    if strategy_evidence.is_empty() or "universe_type" not in strategy_evidence.columns:
        return _empty_csv_schema_frame(path)
    filtered = strategy_evidence.filter(pl.col("universe_type") == "expanded_paper")
    return filtered if not filtered.is_empty() else _empty_csv_schema_frame(path)


def _threshold_advisory_by_symbol_json(df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty():
        return {
            "rows": [],
            "by_symbol": {},
            "advisory_by_symbol": {},
            "ready_for_live_guard": False,
            "source": "quant_lab",
            "mode": "shadow",
        }
    rows = _rows(df)
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_symbol.setdefault(str(row.get("symbol") or "UNKNOWN"), []).append(row)
    return {
        "rows": rows,
        "by_symbol": by_symbol,
        "advisory_by_symbol": _late_entry_threshold_advisory_rows(by_symbol),
        "row_count": len(rows),
        "thresholds_bps": [50, 100, 150, 200, 250, 300],
        "ready_for_live_guard": False,
        "source": "quant_lab",
        "mode": "shadow",
    }


def _late_entry_threshold_advisory_by_symbol_json(df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty():
        return {
            "rows": [],
            "by_symbol": {},
            "thresholds_bps": [50, 100, 150, 200, 250, 300],
            "ready_for_live_guard": False,
            "hard_guard_allowed": False,
            "source": "quant_lab",
            "mode": "shadow_research",
        }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(df):
        grouped.setdefault(str(row.get("symbol") or "UNKNOWN"), []).append(row)
    advisory = _late_entry_threshold_advisory_rows(grouped)
    return {
        "rows": list(advisory.values()),
        "by_symbol": advisory,
        "thresholds_bps": [50, 100, 150, 200, 250, 300],
        "ready_for_live_guard": False,
        "hard_guard_allowed": False,
        "source": "quant_lab",
        "mode": "shadow_research",
        "notes": [
            "By-symbol thresholds are read-only shadow research.",
            "No hard guard is enabled by quant-lab.",
        ],
    }


def _late_entry_threshold_advisory_rows(
    by_symbol: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for symbol, rows in sorted(by_symbol.items()):
        selected = _select_late_entry_threshold(rows)
        output[symbol] = _late_entry_symbol_advisory(symbol, selected, rows)
    return output


def _select_late_entry_threshold(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        block_loss = int(row.get("would_block_loss_count") or 0)
        block_profit = int(row.get("would_block_profit_count") or 0)
        would_block = int(row.get("would_block_count") or 0)
        false_positive = _float_or_none(row.get("false_positive_rate"))
        if would_block <= 0 or block_loss <= 0:
            continue
        if block_loss <= block_profit:
            continue
        if false_positive is not None and false_positive > 0.40:
            continue
        candidates.append(row)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            int(row.get("would_block_loss_count") or 0)
            - int(row.get("would_block_profit_count") or 0),
            -int(row.get("would_block_profit_count") or 0),
            -int(row.get("threshold_bps") or 0),
        ),
        reverse=True,
    )[0]


def _late_entry_symbol_advisory(
    symbol: str,
    selected: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if selected is None:
        best_observed = _best_observed_late_entry_row(rows)
        return {
            "symbol": symbol,
            "recommended_shadow_threshold_bps": None,
            "ready_for_live_guard": False,
            "hard_guard_allowed": False,
            "false_positive_rate": (
                _float_or_none(best_observed.get("false_positive_rate")) if best_observed else None
            ),
            "block_loss_count": (
                int(best_observed.get("would_block_loss_count") or 0) if best_observed else 0
            ),
            "block_profit_count": (
                int(best_observed.get("would_block_profit_count") or 0) if best_observed else 0
            ),
            "advisory": "research_only_no_shadow_threshold",
            "reason": "no_symbol_threshold_with_acceptable_false_positive_rate",
        }
    return {
        "symbol": symbol,
        "recommended_shadow_threshold_bps": int(selected.get("threshold_bps") or 0),
        "ready_for_live_guard": False,
        "hard_guard_allowed": False,
        "false_positive_rate": _float_or_none(selected.get("false_positive_rate")),
        "block_loss_count": int(selected.get("would_block_loss_count") or 0),
        "block_profit_count": int(selected.get("would_block_profit_count") or 0),
        "advisory": "shadow_only_research_threshold",
        "reason": "loss_filtering_potential_with_guard_disabled",
    }


def _best_observed_late_entry_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("would_block_loss_count") or 0),
            -int(row.get("would_block_profit_count") or 0),
            -int(row.get("threshold_bps") or 0),
        ),
        reverse=True,
    )[0]


def _exit_policy_summary_md(summary: pl.DataFrame, sample: pl.DataFrame) -> str:
    lines = [
        "# Exit Policy Review",
        "",
        "This section is read-only research. It compares actual exits with "
        "fixed-hold alternatives.",
        "It does not change V5 exit policy.",
        "",
        f"- sample_rows: {sample.height}",
        f"- summary_rows: {summary.height}",
    ]
    if summary.is_empty():
        lines.extend(["", "- status: no exit policy samples"])
        return "\n".join(lines) + "\n"
    for row in summary.to_dicts():
        lines.extend(
            [
                "",
                f"## {row.get('strategy_id') or row.get('strategy_candidate')}",
                f"- symbol: {row.get('symbol')}",
                f"- decision: {row.get('decision')}",
                f"- sample_count: {row.get('sample_count')}",
                f"- actual_exit_count: {row.get('actual_exit_count')}",
                f"- stop_loss_too_early_count: {row.get('stop_loss_too_early_count')}",
                "- hold_24h_better_than_actual_count: "
                f"{row.get('hold_24h_better_than_actual_count')}",
                f"- avg_delta_hold24h_vs_actual: {row.get('avg_delta_hold24h_vs_actual')}",
                f"- best_alternative_exit_policy: {row.get('best_alternative_exit_policy')}",
                f"- decision_reasons: {row.get('decision_reasons')}",
            ]
        )
    return "\n".join(lines) + "\n"


def _entry_quality_summary_md(
    *,
    missed_low_audit: pl.DataFrame,
    missed_low_by_symbol: pl.DataFrame,
    late_entry_chase_shadow: pl.DataFrame,
    late_entry_threshold: pl.DataFrame,
    pullback_reversal_shadow: pl.DataFrame,
    pullback_reversal_readiness: pl.DataFrame,
    entry_quality_advisory: pl.DataFrame,
) -> str:
    late_block_count = 0
    if (
        not late_entry_chase_shadow.is_empty()
        and "would_block_if_enabled" in late_entry_chase_shadow.columns
    ):
        late_block_count = int(
            late_entry_chase_shadow.filter(pl.col("would_block_if_enabled") == True).height  # noqa: E712
        )
    ready_for_paper = 0
    if (
        not pullback_reversal_readiness.is_empty()
        and "ready_for_paper" in pullback_reversal_readiness.columns
    ):
        ready_for_paper = int(
            pullback_reversal_readiness.filter(pl.col("ready_for_paper") == True).height  # noqa: E712
        )
    lines = [
        "# Entry Quality Research",
        "",
        "This section is read-only research. It audits entries and builds "
        "shadow/advisory diagnostics only.",
        "It does not place orders, block orders, or mutate V5 state.",
        "",
        f"- missed_low_audit_rows: {missed_low_audit.height}",
        f"- missed_low_by_symbol_rows: {missed_low_by_symbol.height}",
        f"- late_entry_chase_shadow_rows: {late_entry_chase_shadow.height}",
        f"- late_entry_chase_would_block_rows: {late_block_count}",
        f"- late_entry_threshold_rows: {late_entry_threshold.height}",
        f"- pullback_reversal_shadow_rows: {pullback_reversal_shadow.height}",
        f"- pullback_reversal_ready_for_paper_symbols: {ready_for_paper}",
        f"- entry_quality_advisory_rows: {entry_quality_advisory.height}",
        "",
        "Operational interpretation:",
        "- missed-low audit quantifies how far actual OPEN_LONG entries were from recent lows.",
        "- late-entry chase is a shadow guard sensitivity study, not a live guard.",
        "- pullback reversal is shadow/paper research and is not live-ready in v0.1.",
    ]
    return "\n".join(lines) + "\n"


def _entry_quality_history_metrics_json(metrics: pl.DataFrame) -> dict[str, Any]:
    if metrics.is_empty():
        return {"rows": [], "row_count": 0, "source": "quant_lab", "mode": "audit"}
    rows = _rows(metrics)
    latest = rows[-1] if rows else {}
    parsed: dict[str, Any] = {}
    text = str(latest.get("metrics_json") or "{}")
    try:
        payload = json.loads(text)
    except ValueError:
        payload = {}
    if isinstance(payload, dict):
        parsed = payload
    return {
        "rows": rows,
        "row_count": len(rows),
        "source": "quant_lab",
        "mode": "audit",
        "latest_metrics": parsed,
    }


def _entry_quality_history_summary_md(metrics: pl.DataFrame) -> str:
    payload = _entry_quality_history_metrics_json(metrics).get("latest_metrics", {})
    if not isinstance(payload, dict):
        payload = {}
    lines = [
        "# Entry Quality Historical Analysis",
        "",
        "This section is read-only historical research. It does not place orders, "
        "block orders, mutate V5 state, or change risk permission.",
        "",
        f"- start_date: {payload.get('start_date', '')}",
        f"- end_date: {payload.get('end_date', '')}",
        f"- mode: {payload.get('mode', '')}",
        f"- cost_mode: {payload.get('cost_mode', '')}",
        f"- missed_low_audit_rows: {payload.get('missed_low_audit_rows', 0)}",
        f"- late_entry_chase_shadow_rows: {payload.get('late_entry_chase_shadow_rows', 0)}",
        f"- pullback_reversal_shadow_rows: {payload.get('pullback_reversal_shadow_rows', 0)}",
        f"- anti_leakage_status: {payload.get('anti_leakage_status', '')}",
        "",
        "Allowed historical decisions: RESEARCH_ONLY, KEEP_SHADOW, PAPER_READY.",
        "LIVE_SMALL_READY is intentionally unavailable in this historical builder.",
    ]
    return "\n".join(lines) + "\n"


def _csv_member(path: str, df: pl.DataFrame) -> str:
    return _csv_text(df, fixed_columns=CSV_SCHEMAS.get(path))


def _csv_text(df: pl.DataFrame, fixed_columns: list[str] | None = None) -> str:
    safe = _csv_frame_with_schema(readers.redact_frame(df), fixed_columns)
    safe = _normalize_bool_columns_for_csv(safe)
    if _can_use_native_csv_writer(safe):
        return safe.write_csv()
    return _python_csv_text(safe, fixed_columns=fixed_columns)


def _can_use_native_csv_writer(df: pl.DataFrame) -> bool:
    for dtype in df.dtypes:
        base_type = dtype.base_type() if hasattr(dtype, "base_type") else dtype
        if base_type in {pl.Object, pl.List, pl.Struct}:
            return False
    return True


def _normalize_bool_columns_for_csv(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() and not df.columns:
        return df
    expressions = []
    for column, dtype in zip(df.columns, df.dtypes, strict=False):
        base_type = dtype.base_type() if hasattr(dtype, "base_type") else dtype
        if base_type == pl.Boolean:
            expressions.append(
                pl.when(pl.col(column).is_null())
                .then(pl.lit(None, dtype=pl.Utf8))
                .when(pl.col(column))
                .then(pl.lit("True"))
                .otherwise(pl.lit("False"))
                .alias(column)
            )
    if not expressions:
        return df
    return df.with_columns(expressions)


def _python_csv_text(df: pl.DataFrame, fixed_columns: list[str] | None = None) -> str:
    columns = df.columns if df.columns else (fixed_columns or [])
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in _rows(df):
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
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, dict | list | tuple):
        return safe_json_dumps(value)
    return value


def _json_text(payload: Any) -> str:
    return safe_json_dumps(payload) + "\n"


def _export_stage_start(stage: str | None = None) -> float:
    if stage:
        _log_export_stage_event("start", stage)
    return time.perf_counter()


def _record_export_stage(
    rows: list[dict[str, Any]],
    stage: str,
    started_at: float,
) -> None:
    elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    max_rss_mb = _process_max_rss_mb()
    rows.append(
        {
            "stage": stage,
            "elapsed_ms": round(elapsed_ms, 3),
            "max_rss_mb": max_rss_mb,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    )
    _log_export_stage_event(
        "end",
        stage,
        elapsed_ms=round(elapsed_ms, 3),
        max_rss_mb=max_rss_mb,
    )


def _log_export_stage_event(
    event: str,
    stage: str,
    **extra: Any,
) -> None:
    log_path = os.environ.get("QUANT_LAB_EXPORT_STAGE_LOG_PATH")
    if not log_path:
        return
    payload = {
        "event": event,
        "stage": stage,
        "recorded_at": datetime.now(UTC).isoformat(),
        "max_rss_mb": _process_max_rss_mb(),
        **extra,
    }
    try:
        with Path(log_path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        return


def _export_stage_timings_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "stage": pl.Utf8,
                "elapsed_ms": pl.Float64,
                "max_rss_mb": pl.Float64,
                "recorded_at": pl.Utf8,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).select(
        ["stage", "elapsed_ms", "max_rss_mb", "recorded_at"]
    )


def _process_max_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return round(rss / (1024 * 1024), 3)
    return round(rss / 1024, 3)


def _value_counts(df: pl.DataFrame, column: str) -> dict[str, int]:
    if df.is_empty() or column not in df.columns:
        return {}
    return {
        str(row[column]): int(row["count"])
        for row in df.group_by(column).len(name="count").to_dicts()
    }


def _strategy_dashboard_decision_counts(
    opportunity_advisory: pl.DataFrame,
    alpha_discovery_board: pl.DataFrame,
    strategy_evidence: pl.DataFrame,
) -> dict[str, int]:
    source = opportunity_advisory
    if source.is_empty():
        source = alpha_discovery_board
    if source.is_empty():
        source = strategy_evidence
    counts = _value_counts(source, "decision")
    ordered: dict[str, int] = {}
    for decision in ["PAPER_READY", "KEEP_SHADOW", "REGIME_SHADOW", "KILL", "LIVE_SMALL_READY"]:
        ordered[decision] = counts.get(decision, 0)
    return ordered


def _baseline_gate_summary(gates: pl.DataFrame, evidence: pl.DataFrame) -> dict[str, Any]:
    status = "UNKNOWN"
    if not gates.is_empty() and "alpha_id" in gates.columns:
        rows = [
            row
            for row in gates.to_dicts()
            if str(row.get("alpha_id") or "") == CORE_MOMENTUM_ALPHA_ID
        ]
        if rows:
            rows.sort(key=lambda row: str(row.get("created_at") or ""))
            status = str(rows[-1].get("status") or "UNKNOWN")
    if status == "UNKNOWN" and not evidence.is_empty() and "alpha_id" in evidence.columns:
        rows = [
            row
            for row in evidence.to_dicts()
            if str(row.get("alpha_id") or "") == CORE_MOMENTUM_ALPHA_ID
        ]
        if rows:
            rows.sort(key=lambda row: str(row.get("created_at") or ""))
            status = str(rows[-1].get("evidence_status") or "UNKNOWN")
    return {
        "alpha_id": CORE_MOMENTUM_ALPHA_ID,
        "role": RESEARCH_BASELINE_ROLE,
        "baseline_status": status,
        "not_live_eligible": True,
        "not_global_strategy_gate": True,
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
    if any(check["status"] == "FAIL" and check.get("severity") == "critical" for check in checks):
        return "CRITICAL"
    if any(check["status"] in {"FAIL", "WARN", "N/A"} for check in checks):
        return "WARN"
    return "PASS"


def _data_quality_failure_lines(checks: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for check in checks:
        if check.get("status") != "FAIL":
            continue
        if check.get("severity") != "critical":
            continue
        failures.append(f"{check.get('name')}: {check.get('detail')}")
    return sorted(set(failures))


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
        return max(_line_count(text) - 1, 0)
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
    digest = hashlib.sha256()
    for chunk in _text_chunks(text):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _sha256_payload(payload: _MemberPayload) -> str:
    if isinstance(payload, bytes):
        return hashlib.sha256(payload).hexdigest()
    return _sha256_text(payload)


def _payload_byte_chunks(payload: _MemberPayload, chunk_size: int = 1024 * 1024) -> Any:
    if isinstance(payload, bytes):
        for index in range(0, len(payload), chunk_size):
            yield payload[index : index + chunk_size]
        return
    for chunk in _text_chunks(payload, chunk_size):
        yield chunk.encode("utf-8")


def _text_chunks(text: str, chunk_size: int = 1024 * 1024) -> Any:
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


def _iter_text_lines(text: str) -> Any:
    start = 0
    while start < len(text):
        end = text.find("\n", start)
        if end == -1:
            yield text[start:]
            return
        yield text[start : end + 1]
        start = end + 1


def _line_count(text: str) -> int:
    if not text:
        return 0
    count = text.count("\n")
    if not text.endswith("\n"):
        count += 1
    return count


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


def _git_commit_full() -> str | None:
    root = Path(__file__).resolve().parents[3]
    return _git_command(["rev-parse", "HEAD"], root)


def _source_version(component: str, git_commit: str | None) -> str:
    version = git_commit or __version__
    return f"{component}:{version}"


def _observable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in UNOBSERVABLE_TEXT_VALUES else text


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
