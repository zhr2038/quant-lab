from __future__ import annotations

import hashlib
import json
import os
import time as time_module
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.costs.model import (
    COST_BOOTSTRAP_READINESS_FIELDS,
    DEFAULT_LIVE_UNIVERSE_SYMBOLS,
    LIVE_UNIVERSE_COST_COVERAGE_FIELDS,
    evaluate_live_universe_cost_coverage,
)
from quant_lab.data.lake import invalid_parquet_files, read_parquet_dataset
from quant_lab.data.market_bar_time import (
    DEFAULT_MARKET_BAR_TIMEFRAME,
    market_bar_close_ts,
)
from quant_lab.ops.dataset_registry import get_dataset_spec
from quant_lab.research.portfolio import research_portfolio_closed_keys
from quant_lab.symbols import normalize_symbol
from quant_lab.web import perf
from quant_lab.web.pages._common import display_value

DATASET_PATHS = {
    "market_bar": Path("silver") / "market_bar",
    "feature_value": Path("gold") / "feature_value",
    "feature_coverage_daily": Path("gold") / "feature_coverage_daily",
    "feature_anomaly_daily": Path("gold") / "feature_anomaly_daily",
    "factor_definition": Path("gold") / "factor_definition",
    "factor_value": Path("gold") / "factor_value",
    "factor_evidence": Path("gold") / "factor_evidence",
    "factor_candidate": Path("gold") / "factor_candidate",
    "factor_correlation_daily": Path("gold") / "factor_correlation_daily",
    "research_hypothesis_registry": Path("gold") / "research_hypothesis_registry",
    "research_trial_ledger": Path("gold") / "research_trial_ledger",
    "factor_attribution": Path("gold") / "factor_attribution",
    "factor_portfolio_validation": Path("gold") / "factor_portfolio_validation",
    "factor_retirement": Path("gold") / "factor_retirement",
    "factor_external_audit_evidence": Path("gold") / "factor_external_audit_evidence",
    "factor_strategy_bridge_candidates": Path("gold") / "factor_strategy_bridge_candidates",
    "cost_bucket_daily": Path("gold") / "cost_bucket_daily",
    "cost_health_daily": Path("gold") / "cost_health_daily",
    "cost_bootstrap_readiness": Path("gold") / "cost_bootstrap_readiness",
    "alpha_evidence": Path("gold") / "alpha_evidence",
    "alpha_discovery_board": Path("gold") / "alpha_discovery_board",
    "strategy_evidence": Path("gold") / "strategy_evidence",
    "strategy_evidence_sample": Path("gold") / "strategy_evidence_sample",
    "strategy_evidence_quality": Path("gold") / "strategy_evidence_quality",
    "research_portfolio_status": Path("gold") / "research_portfolio_status",
    "strategy_opportunity_advisory": Path("gold") / "strategy_opportunity_advisory",
    "trade_opportunity_event": Path("gold") / "trade_opportunity_event",
    "trade_opportunity_label": Path("gold") / "trade_opportunity_label",
    "trade_level_similarity_outcome": Path("gold") / "trade_level_similarity_outcome",
    "trade_level_judgment": Path("gold") / "trade_level_judgment",
    "trade_level_bucket_policy": Path("gold") / "trade_level_bucket_policy",
    "trade_level_opportunity_queue": Path("gold") / "trade_level_opportunity_queue",
    "quant_lab_false_block_audit": Path("gold") / "quant_lab_false_block_audit",
    "v5_trade_learning_sample": Path("gold") / "v5_trade_learning_sample",
    "v5_trade_outcome_attribution": Path("gold") / "v5_trade_outcome_attribution",
    "quant_lab_opportunity_cost_event": Path("gold") / "quant_lab_opportunity_cost_event",
    "quant_lab_opportunity_cost_daily": Path("gold") / "quant_lab_opportunity_cost_daily",
    "opportunity_cost_by_bucket": Path("gold") / "opportunity_cost_by_bucket",
    "quant_lab_decision_regret": Path("gold") / "quant_lab_decision_regret",
    "v5_missed_opportunity_audit": Path("gold") / "v5_missed_opportunity_audit",
    "v5_risk_on_multi_buy_shadow": Path("gold") / "v5_risk_on_multi_buy_shadow",
    "risk_on_multi_buy_shadow": Path("gold") / "risk_on_multi_buy_shadow",
    "alpha_factory_candidate": Path("gold") / "alpha_factory_candidate",
    "alpha_factory_result": Path("gold") / "alpha_factory_result",
    "alpha_factory_promotion_queue": Path("gold") / "alpha_factory_promotion_queue",
    "second_stage_alpha_factory_sample": Path("gold") / "second_stage_alpha_factory_sample",
    "second_stage_alpha_factory_summary": Path("gold") / "second_stage_alpha_factory_summary",
    "expanded_relative_strength_decision_sample": Path("gold")
    / "expanded_relative_strength_decision_sample",
    "alpha_factory_template_registry": Path("gold") / "alpha_factory_template_registry",
    "market_regime_daily": Path("gold") / "market_regime_daily",
    "strategy_regime_matrix": Path("gold") / "strategy_regime_matrix",
    "regime_strategy_advisory": Path("gold") / "regime_strategy_advisory",
    "expanded_universe_candidate": Path("gold") / "expanded_universe_candidate",
    "expanded_universe_quality": Path("gold") / "expanded_universe_quality",
    "expanded_universe_candidate_event": Path("gold") / "expanded_universe_candidate_event",
    "expanded_universe_candidate_label": Path("gold") / "expanded_universe_candidate_label",
    "expanded_universe_promotion_queue": Path("gold") / "expanded_universe_promotion_queue",
    "expanded_universe_candidate_maturity": Path("gold") / "expanded_universe_candidate_maturity",
    "expanded_universe_watchlist": Path("gold") / "expanded_universe_watchlist",
    "expanded_crypto_universe_shadow": Path("gold") / "expanded_crypto_universe_shadow",
    "symbol_quality_score": Path("gold") / "symbol_quality_score",
    "expanded_crypto_candidate_outcomes_by_symbol": Path("gold")
    / "expanded_crypto_candidate_outcomes_by_symbol",
    "expanded_crypto_recommendations": Path("gold") / "expanded_crypto_recommendations",
    "paper_strategy_runs": Path("gold") / "paper_strategy_runs",
    "paper_strategy_daily": Path("gold") / "paper_strategy_daily",
    "paper_strategy_proposal": Path("gold") / "paper_strategy_proposal",
    "paper_strategy_proposals_current": Path("gold") / "paper_strategy_proposals_current",
    "paper_strategy_proposal_snapshot": Path("gold") / "paper_strategy_proposal_snapshot",
    "paper_strategy_trackers_current": Path("gold") / "paper_strategy_trackers_current",
    "paper_strategy_registry": Path("gold") / "paper_strategy_registry",
    "paper_strategy_registry_current": Path("gold") / "paper_strategy_registry_current",
    "paper_strategy_registry_history": Path("gold") / "paper_strategy_registry_history",
    "paper_strategy_ack_current": Path("gold") / "paper_strategy_ack_current",
    "paper_strategy_ack_history": Path("gold") / "paper_strategy_ack_history",
    "paper_strategy_identity_conflict": Path("gold") / "paper_strategy_identity_conflict",
    "paper_cohort_manifest": Path("gold") / "paper_cohort_manifest",
    "api_auth_incident": Path("gold") / "api_auth_incident",
    "api_auth_production_slo": Path("gold") / "api_auth_production_slo",
    "api_auth_security_rejections": Path("gold") / "api_auth_security_rejections",
    "api_auth_manual_probe": Path("gold") / "api_auth_manual_probe",
    "paper_runtime_freshness": Path("gold") / "paper_runtime_freshness",
    "paper_proposal_propagation_status": Path("gold") / "paper_proposal_propagation_status",
    "ai_research_run": Path("gold") / "ai_research_run",
    "ai_research_finding": Path("gold") / "ai_research_finding",
    "ai_factor_proposal": Path("gold") / "ai_factor_proposal",
    "ai_paper_strategy_draft": Path("gold") / "ai_paper_strategy_draft",
    "ai_experiment_proposal": Path("gold") / "ai_experiment_proposal",
    "ai_research_hypothesis_draft": Path("gold") / "ai_research_hypothesis_draft",
    "ai_data_collection_proposal": Path("gold") / "ai_data_collection_proposal",
    "ai_attribution_experiment": Path("gold") / "ai_attribution_experiment",
    "ai_code_review_target": Path("gold") / "ai_code_review_target",
    "paper_strategy_migration_audit": Path("gold") / "paper_strategy_migration_audit",
    "paper_strategy_promotion_gate": Path("gold") / "paper_strategy_promotion_gate",
    "strategy_cost_trust": Path("gold") / "strategy_cost_trust",
    "paper_slippage_coverage": Path("gold") / "paper_slippage_coverage",
    "v5_final_score_vs_alpha6_conflict": Path("gold") / "v5_final_score_vs_alpha6_conflict",
    "v5_bnb_strong_alpha6_bypass_shadow": Path("gold") / "v5_bnb_strong_alpha6_bypass_shadow",
    "v5_negative_expectancy_attribution": Path("gold") / "v5_negative_expectancy_attribution",
    "v5_bnb_paper_strategy_runs": Path("gold") / "v5_bnb_paper_strategy_runs",
    "v5_bnb_paper_strategy_daily": Path("gold") / "v5_bnb_paper_strategy_daily",
    "v5_bnb_paper_strategy_daily_latest": Path("gold") / "v5_bnb_paper_strategy_daily_latest",
    "sol_protect_paper_loss_attribution": Path("gold") / "sol_protect_paper_loss_attribution",
    "sol_protect_paper_loss_summary": Path("gold") / "sol_protect_paper_loss_summary",
    "v5_missed_low_audit": Path("gold") / "v5_missed_low_audit",
    "v5_missed_low_by_symbol": Path("gold") / "v5_missed_low_by_symbol",
    "v5_missed_low_by_entry_reason": Path("gold") / "v5_missed_low_by_entry_reason",
    "v5_late_entry_chase_shadow": Path("gold") / "v5_late_entry_chase_shadow",
    "v5_late_entry_chase_threshold_advisory": Path("gold")
    / "v5_late_entry_chase_threshold_advisory",
    "v5_late_entry_chase_threshold_by_symbol": Path("gold")
    / "v5_late_entry_chase_threshold_by_symbol",
    "v5_pullback_reversal_shadow": Path("gold") / "v5_pullback_reversal_shadow",
    "v5_pullback_reversal_readiness": Path("gold") / "v5_pullback_reversal_readiness",
    "v5_pullback_reversal_rule_comparison": Path("gold") / "v5_pullback_reversal_rule_comparison",
    "v5_btc_probe_entry_quality_audit": Path("silver") / "v5_btc_probe_entry_quality_audit",
    "btc_probe_exit_policy_review": Path("gold") / "btc_probe_exit_policy_review",
    "btc_probe_exit_policy_summary": Path("gold") / "btc_probe_exit_policy_summary",
    "bnb_swing_exit_policy_review": Path("gold") / "bnb_swing_exit_policy_review",
    "bnb_exit_policy_v5_vs_quant_lab_consistency": Path("gold")
    / "bnb_exit_policy_v5_vs_quant_lab_consistency",
    "bnb_swing_exit_policy_summary": Path("gold") / "bnb_swing_exit_policy_summary",
    "exit_policy_review_sample": Path("gold") / "exit_policy_review_sample",
    "exit_policy_review_summary": Path("gold") / "exit_policy_review_summary",
    "v5_entry_quality_advisory": Path("gold") / "v5_entry_quality_advisory",
    "v5_entry_quality_history_late_entry_chase_threshold_sensitivity": Path("gold")
    / "v5_entry_quality_history_late_entry_chase_threshold_sensitivity",
    "v5_entry_quality_history_pullback_by_symbol": Path("gold")
    / "v5_entry_quality_history_pullback_by_symbol",
    "v5_entry_quality_history_pullback_by_regime": Path("gold")
    / "v5_entry_quality_history_pullback_by_regime",
    "v5_entry_quality_history_pullback_by_horizon": Path("gold")
    / "v5_entry_quality_history_pullback_by_horizon",
    "v5_entry_quality_history_anti_leakage_check": Path("gold")
    / "v5_entry_quality_history_anti_leakage_check",
    "v5_entry_quality_history_metrics": Path("gold") / "v5_entry_quality_history_metrics",
    "gate_decision": Path("gold") / "gate_decision",
    "risk_permission": Path("gold") / "risk_permission",
    "risk_permission_api_dependency_meta": Path("gold") / "risk_permission_api_dependency_meta",
    "v5_bundle_manifest": Path("bronze") / "strategy_telemetry" / "v5" / "bundle_manifest",
    "api_request_metrics": Path("bronze") / "api_request_metrics",
    "job_run_history": Path("gold") / "job_run_history",
    "lake_file_health_daily": Path("gold") / "lake_file_health_daily",
    "strategy_health_daily": Path("gold") / "strategy_health_daily",
    "okx_public_ws_health": Path("bronze") / "collector_health" / "okx_public_ws",
    "v5_execution_quality_daily": Path("gold") / "v5_execution_quality_daily",
    "v5_gate_compliance_daily": Path("gold") / "v5_gate_compliance_daily",
    "v5_missed_opportunity_daily": Path("gold") / "v5_missed_opportunity_daily",
    "v5_config_health_daily": Path("gold") / "v5_config_health_daily",
    "v5_issue_summary_daily": Path("gold") / "v5_issue_summary_daily",
    "v5_quant_lab_mode_daily": Path("gold") / "v5_quant_lab_mode_daily",
    "v5_quant_lab_enforcement_daily": Path("gold") / "v5_quant_lab_enforcement_daily",
    "v5_quant_lab_usage": Path("silver") / "v5_quant_lab_usage",
    "v5_quant_lab_request": Path("silver") / "v5_quant_lab_request",
    "v5_quant_lab_compliance": Path("silver") / "v5_quant_lab_compliance",
    "v5_quant_lab_cost_usage": Path("silver") / "v5_quant_lab_cost_usage",
    "v5_quant_lab_fallback": Path("silver") / "v5_quant_lab_fallback",
    "v5_cost_probe_p3_preflight": Path("silver") / "v5_cost_probe_p3_preflight",
    "v5_cost_probe_live_execution_status": Path("silver") / "v5_cost_probe_live_execution_status",
    "v5_cost_probe_order_event": Path("silver") / "v5_cost_probe_order_event",
    "v5_cost_probe_roundtrip_event": Path("silver") / "v5_cost_probe_roundtrip_event",
    "v5_trade_opportunity_funnel": Path("silver") / "v5_trade_opportunity_funnel",
    "v5_paper_strategy_run": Path("silver") / "v5_paper_strategy_run",
    "v5_paper_strategy_exit_quality": Path("silver") / "v5_paper_strategy_exit_quality",
    "v5_paper_strategy_proposal_ack": Path("silver") / "v5_paper_strategy_proposal_ack",
    "v5_paper_strategy_proposal_ack_current": Path("silver")
    / "v5_paper_strategy_proposal_ack_current",
    "v5_paper_strategy_proposal_ack_history": Path("silver")
    / "v5_paper_strategy_proposal_ack_history",
    "v5_paper_strategy_daily": Path("silver") / "v5_paper_strategy_daily",
    "v5_paper_slippage_coverage": Path("silver") / "v5_paper_slippage_coverage",
    "v5_paper_strategy_registry": Path("silver") / "v5_paper_strategy_registry",
    "v5_paper_strategy_registry_current": Path("silver") / "v5_paper_strategy_registry_current",
    "v5_paper_strategy_registry_history": Path("silver") / "v5_paper_strategy_registry_history",
    "v5_paper_strategy_trackers_current": Path("silver") / "v5_paper_strategy_trackers_current",
    "v5_paper_strategy_state": Path("silver") / "v5_paper_strategy_state",
    "v5_paper_strategy_signal": Path("silver") / "v5_paper_strategy_signal",
    "v5_paper_strategy_quote_coverage": Path("silver") / "v5_paper_strategy_quote_coverage",
    "v5_paper_strategy_cost_evidence": Path("silver") / "v5_paper_strategy_cost_evidence",
    "v5_paper_strategy_error": Path("silver") / "v5_paper_strategy_error",
    "v5_paper_strategy_restart_recovery": Path("silver") / "v5_paper_strategy_restart_recovery",
    "v5_quant_lab_contract_status": Path("silver") / "v5_quant_lab_contract_status",
    "v5_fill_bill_cost_reconciliation": Path("silver") / "v5_fill_bill_cost_reconciliation",
    "v5_expanded_universe_advisory_reader": Path("silver") / "v5_expanded_universe_advisory_reader",
    "v5_expanded_universe_paper_runs": Path("silver") / "v5_expanded_universe_paper_runs",
    "v5_expanded_universe_paper_daily": Path("silver") / "v5_expanded_universe_paper_daily",
    "v5_bnb_profit_lock_shadow": Path("silver") / "v5_bnb_profit_lock_shadow",
    "v5_bnb_negative_expectancy_attribution": Path("silver")
    / "v5_bnb_negative_expectancy_attribution",
    "v5_final_score_vs_alpha6_conflict_silver": Path("silver")
    / "v5_final_score_vs_alpha6_conflict",
    "v5_bnb_strong_alpha6_bypass_shadow_silver": Path("silver")
    / "v5_bnb_strong_alpha6_bypass_shadow",
    "v5_negative_expectancy_attribution_silver": Path("silver")
    / "v5_negative_expectancy_attribution",
    "v5_bnb_paper_strategy_runs_silver": Path("silver") / "v5_bnb_paper_strategy_runs",
    "v5_bnb_paper_strategy_daily_silver": Path("silver") / "v5_bnb_paper_strategy_daily",
    "v5_negative_expectancy_consistency": Path("silver") / "v5_negative_expectancy_consistency",
    "v5_candidate_event": Path("silver") / "v5_candidate_event",
    "v5_candidate_label": Path("gold") / "v5_candidate_label",
    "v5_candidate_quality_daily": Path("gold") / "v5_candidate_quality_daily",
    "v5_candidate_outcome_summary": Path("gold") / "v5_candidate_outcome_summary",
    "cost_probe_fill_bill_match": Path("gold") / "cost_probe_fill_bill_match",
    "cost_probe_cost_disagreement": Path("gold") / "cost_probe_cost_disagreement",
    "v5_decision_audit": Path("silver") / "v5_decision_audit",
    "v5_trade_event": Path("silver") / "v5_trade_event",
    "trade_print": Path("silver") / "trade_print",
    "orderbook_snapshot": Path("silver") / "orderbook_snapshot",
    "trade_activity_1m": Path("silver") / "trade_activity_1m",
    "orderbook_spread_1m": Path("silver") / "orderbook_spread_1m",
    "okx_public_ws": Path("bronze") / "okx_public_ws",
    "okx_private_readonly_fills": Path("bronze") / "okx_private_readonly" / "fills_history",
    "okx_private_readonly_bills": Path("bronze") / "okx_private_readonly" / "bills",
    "decision_audit": Path("silver") / "decision_audit",
}
WEB_FILE_INDEX_FALLBACK_WARNING = "web_file_index_missing_refresh_required"
WEB_SNAPSHOT_META_MISSING_WARNING = "web_snapshot_meta_missing"
WEB_CACHE_DEFAULT_TTL_SECONDS = 30
V5_PAPER_TELEMETRY_DATASETS = {
    "v5_bnb_paper_strategy_daily",
    "v5_bnb_paper_strategy_daily_latest",
    "v5_bnb_paper_strategy_runs",
    "v5_paper_strategy_run",
    "v5_paper_strategy_exit_quality",
    "v5_paper_strategy_proposal_ack",
    "v5_paper_strategy_daily",
    "v5_paper_slippage_coverage",
    "v5_paper_strategy_registry",
    "v5_paper_strategy_state",
    "v5_paper_strategy_signal",
    "v5_paper_strategy_quote_coverage",
    "v5_paper_strategy_cost_evidence",
    "v5_paper_strategy_error",
    "v5_paper_strategy_restart_recovery",
    "v5_quant_lab_contract_status",
    "v5_expanded_universe_advisory_reader",
    "v5_expanded_universe_paper_runs",
    "v5_expanded_universe_paper_daily",
}
DERIVED_LATEST_SOURCE_CURRENT_STATUS = "derived_latest_source_current"
OPTIONAL_EMPTY_DATASET_STATUSES = {
    "closed_research_snapshot",
    "export_derived_optional",
    "expanded_universe_automation_active",
    "legacy_optional",
    "waiting_for_v5_paper_telemetry",
    "waiting_for_v5_bundle_manifest",
    "entry_quality_optional",
    "trade_level_optional",
    "historical_research_snapshot",
    "cost_probe_optional_audit",
    "registry_optional_empty",
    "registry_freshness_not_required",
    "registry_freshness_within_sla",
    DERIVED_LATEST_SOURCE_CURRENT_STATUS,
    "event_driven_no_recent_cost_probe_p3_preflight",
    "event_driven_no_recent_cost_probe_order_event",
    "event_driven_no_recent_cost_probe_roundtrip_event",
}
ENTRY_QUALITY_DATASETS = {
    "v5_missed_low_audit",
    "v5_missed_low_by_symbol",
    "v5_missed_low_by_entry_reason",
    "v5_late_entry_chase_shadow",
    "v5_late_entry_chase_threshold_advisory",
    "v5_late_entry_chase_threshold_by_symbol",
    "v5_pullback_reversal_shadow",
    "v5_pullback_reversal_readiness",
    "v5_pullback_reversal_rule_comparison",
    "v5_btc_probe_entry_quality_audit",
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
}
HISTORICAL_RESEARCH_DATASETS = {
    "v5_paper_strategy_proposal_ack_history",
    "v5_paper_strategy_registry_history",
    "v5_entry_quality_history_late_entry_chase_threshold_sensitivity",
    "v5_entry_quality_history_pullback_by_symbol",
    "v5_entry_quality_history_pullback_by_regime",
    "v5_entry_quality_history_pullback_by_horizon",
    "v5_entry_quality_history_anti_leakage_check",
    "v5_entry_quality_history_metrics",
}
RESEARCH_DIAGNOSTIC_DATASET_KEYS: dict[str, tuple[str, ...]] = {
    "sol_protect_paper_loss_attribution": (
        "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
        "v5.sol_protect_alpha6_low_exception",
    ),
    "sol_protect_paper_loss_summary": (
        "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
        "v5.sol_protect_alpha6_low_exception",
    ),
    "btc_probe_exit_policy_review": (
        "BTC_STRICT_PROBE_EXIT_POLICY_REVIEW",
        "v5.btc_strict_probe_exit_policy_review",
    ),
    "btc_probe_exit_policy_summary": (
        "BTC_STRICT_PROBE_EXIT_POLICY_REVIEW",
        "v5.btc_strict_probe_exit_policy_review",
    ),
    "bnb_swing_exit_policy_review": (
        "BNB_SWING_EXIT_POLICY_REVIEW",
        "v5.bnb_swing_exit_policy_review",
    ),
    "bnb_exit_policy_v5_vs_quant_lab_consistency": (
        "BNB_SWING_EXIT_POLICY_REVIEW",
        "v5.bnb_swing_exit_policy_review",
    ),
    "bnb_swing_exit_policy_summary": (
        "BNB_SWING_EXIT_POLICY_REVIEW",
        "v5.bnb_swing_exit_policy_review",
    ),
}
EVENT_DRIVEN_V5_DATASET_STATUSES = {
    "v5_trade_event": "event_driven_no_recent_trade",
    "v5_trade_opportunity_funnel": "event_driven_no_recent_trade_opportunity_funnel",
    "v5_paper_strategy_run": "NO_NEW_EVENT_EXPECTED",
    "v5_paper_strategy_signal": "NO_NEW_EVENT_EXPECTED",
    "v5_bnb_profit_lock_shadow": "event_driven_no_recent_bnb_profit_lock",
    "v5_quant_lab_usage": "event_driven_no_recent_quant_lab_usage",
    "v5_quant_lab_request": "event_driven_no_recent_quant_lab_request",
    "v5_quant_lab_cost_usage": "event_driven_no_recent_quant_lab_cost_usage",
    "v5_quant_lab_fallback": "event_driven_no_recent_quant_lab_fallback",
    "v5_cost_probe_p3_preflight": "event_driven_no_recent_cost_probe_p3_preflight",
    "v5_cost_probe_live_execution_status": (
        "event_driven_no_recent_cost_probe_live_execution_status"
    ),
    "v5_cost_probe_order_event": "event_driven_no_recent_cost_probe_order_event",
    "v5_cost_probe_roundtrip_event": "event_driven_no_recent_cost_probe_roundtrip_event",
    "v5_fill_bill_cost_reconciliation": "event_driven_no_recent_fill_bill_reconciliation",
    "v5_bnb_negative_expectancy_attribution": (
        "event_driven_no_recent_bnb_negative_expectancy_attribution"
    ),
    "v5_negative_expectancy_attribution": (
        "event_driven_no_recent_negative_expectancy_attribution"
    ),
    "v5_negative_expectancy_attribution_silver": (
        "event_driven_no_recent_negative_expectancy_attribution"
    ),
}
EVENT_DRIVEN_OKX_READONLY_DATASET_STATUSES = {
    "okx_private_readonly_fills": "event_driven_no_recent_okx_readonly_fills",
    "okx_private_readonly_bills": "event_driven_no_recent_okx_readonly_bills",
}
EVENT_DRIVEN_OK_STATUSES = set(EVENT_DRIVEN_V5_DATASET_STATUSES.values()) | set(
    EVENT_DRIVEN_OKX_READONLY_DATASET_STATUSES.values()
)
DERIVED_ROLLUP_SOURCE_DATASETS = {
    "trade_activity_1m": "trade_print",
    "orderbook_spread_1m": "orderbook_snapshot",
}
DERIVED_LATEST_SOURCE_DATASETS = {
    "v5_bnb_paper_strategy_daily_latest": "v5_bnb_paper_strategy_daily",
}
DERIVED_ROLLUP_PENDING_STATUS = "derived_rollup_pending"
STALE_DATASET_SCHEMA: dict[str, Any] = {
    "takeaway": pl.Utf8,
    "severity": pl.Utf8,
    "next_action": pl.Utf8,
    "dataset": pl.Utf8,
    "rows": pl.Int64,
    "status": pl.Utf8,
    "path": pl.Utf8,
    "freshness_seconds": pl.Int64,
    "timestamp_column": pl.Utf8,
    "latest_timestamp": pl.Utf8,
    "warning": pl.Utf8,
}

DATASET_TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    "market_bar": ("ts",),
    "okx_public_ws": ("received_at",),
    "trade_print": ("ts", "ingest_ts"),
    "orderbook_snapshot": ("ts", "ingest_ts"),
    "trade_activity_1m": ("latest_trade_ts", "minute_ts"),
    "orderbook_spread_1m": ("ts", "minute_ts"),
    "cost_bucket_daily": ("created_at", "day"),
    "cost_health_daily": ("created_at", "day"),
    "cost_bootstrap_readiness": ("generated_at",),
    "cost_probe_fill_bill_match": ("generated_at",),
    "cost_probe_cost_disagreement": ("generated_at",),
    "gate_decision": ("created_at",),
    "risk_permission": ("as_of_ts", "created_at"),
    "risk_permission_api_dependency_meta": ("generated_at", "telemetry_latest_ts"),
    "api_request_metrics": ("request_ts",),
    "api_auth_incident": ("generated_at", "last_error_at"),
    "api_auth_error_timeline": ("generated_at", "last_error_at"),
    "api_auth_client_summary": ("generated_at", "last_error_at"),
    "api_auth_production_slo": (
        "generated_at",
        "last_unexpected_auth_failure_at",
    ),
    "api_auth_security_rejections": ("generated_at", "last_error_at"),
    "api_auth_manual_probe": ("generated_at", "last_error_at"),
    "paper_strategy_proposal_snapshot": (
        "last_evaluated_at",
        "snapshot_generated_at",
    ),
    "paper_runtime_freshness": ("generated_at", "latest_at"),
    "paper_proposal_propagation_status": (
        "generated_at",
        "tracker_created_at",
        "ack_at",
        "proposal_published_at",
    ),
    "paper_cohort_manifest": ("last_evaluated_at", "created_at"),
    "ai_research_run": ("completed_at", "created_at"),
    "ai_research_finding": ("completed_at",),
    "ai_factor_proposal": ("completed_at",),
    "ai_paper_strategy_draft": ("completed_at",),
    "ai_experiment_proposal": ("completed_at",),
    "ai_research_hypothesis_draft": ("completed_at",),
    "ai_data_collection_proposal": ("completed_at",),
    "ai_attribution_experiment": ("completed_at",),
    "ai_code_review_target": ("completed_at",),
    "job_run_history": ("finished_at", "started_at"),
    "lake_file_health_daily": ("created_at", "day"),
    "strategy_health_daily": ("latest_bundle_ts", "date"),
    "v5_bundle_manifest": ("bundle_ts", "ingest_ts", "created_at"),
    "v5_execution_quality_daily": ("latest_bundle_ts", "date"),
    "v5_gate_compliance_daily": ("latest_bundle_ts", "date"),
    "v5_missed_opportunity_daily": ("latest_bundle_ts", "date"),
    "v5_config_health_daily": ("latest_bundle_ts", "date"),
    "v5_issue_summary_daily": ("latest_bundle_ts", "date"),
    "v5_quant_lab_mode_daily": ("latest_bundle_ts", "created_at", "date"),
    "v5_quant_lab_enforcement_daily": ("latest_bundle_ts", "created_at", "date"),
    "okx_private_readonly_fills": ("ingest_ts",),
    "okx_private_readonly_bills": ("ingest_ts",),
    "feature_value": ("created_at", "ts"),
    "feature_coverage_daily": ("created_at", "max_ts", "day"),
    "feature_anomaly_daily": ("created_at", "example_ts", "day"),
    "factor_definition": ("created_at",),
    "factor_value": ("created_at", "ts"),
    "factor_evidence": ("created_at", "end_ts", "as_of_date"),
    "factor_candidate": ("created_at", "as_of_date"),
    "factor_correlation_daily": ("created_at", "as_of_date"),
    "research_hypothesis_registry": ("updated_at", "created_at"),
    "research_trial_ledger": ("finished_at", "started_at", "submitted_at"),
    "factor_attribution": ("created_at", "as_of_date"),
    "factor_portfolio_validation": ("created_at", "as_of_date"),
    "factor_retirement": ("recorded_at",),
    "factor_external_audit_evidence": ("imported_at",),
    "factor_strategy_bridge_candidates": ("generated_at", "created_at", "as_of_date"),
    "alpha_evidence": ("created_at", "end_ts"),
    "alpha_discovery_board": ("created_at", "end_ts", "start_ts", "as_of_date"),
    "strategy_evidence": ("created_at", "end_ts", "as_of_date"),
    "strategy_evidence_sample": ("created_at", "ts_utc"),
    "strategy_evidence_quality": ("created_at", "as_of_date"),
    "research_portfolio_status": ("created_at", "last_review_date"),
    "strategy_opportunity_advisory": ("as_of_ts", "created_at"),
    "trade_opportunity_event": ("decision_ts", "created_at"),
    "trade_opportunity_label": ("decision_ts", "created_at"),
    "trade_level_similarity_outcome": ("decision_ts", "created_at"),
    "trade_level_judgment": ("decision_ts", "created_at"),
    "trade_level_bucket_policy": ("created_at", "policy_date", "expires_at"),
    "trade_level_opportunity_queue": ("created_at", "policy_date", "expires_at"),
    "quant_lab_false_block_audit": ("decision_ts", "created_at"),
    "v5_trade_learning_sample": ("decision_ts", "created_at"),
    "v5_trade_outcome_attribution": ("decision_ts", "created_at"),
    "quant_lab_opportunity_cost_event": ("decision_ts", "created_at"),
    "quant_lab_opportunity_cost_daily": ("day", "created_at"),
    "opportunity_cost_by_bucket": ("created_at",),
    "quant_lab_decision_regret": ("decision_ts", "created_at"),
    "v5_missed_opportunity_audit": ("ts_utc", "generated_at"),
    "v5_risk_on_multi_buy_shadow": ("generated_at", "decision_ts"),
    "risk_on_multi_buy_shadow": ("generated_at", "decision_ts"),
    "second_stage_alpha_factory_sample": ("created_at", "ts_utc"),
    "second_stage_alpha_factory_summary": ("created_at", "end_ts", "as_of_date"),
    "expanded_relative_strength_decision_sample": ("created_at", "decision_ts"),
    "alpha_factory_template_registry": ("created_at",),
    "alpha_factory_candidate": ("generated_at", "as_of_date"),
    "alpha_factory_result": ("generated_at", "end_ts", "as_of_date"),
    "alpha_factory_promotion_queue": ("generated_at", "as_of_date"),
    "market_regime_daily": ("created_at", "as_of_date"),
    "strategy_regime_matrix": ("created_at", "as_of_date"),
    "regime_strategy_advisory": ("created_at", "as_of_date"),
    "expanded_universe_candidate": ("generated_at", "as_of_date"),
    "expanded_universe_quality": ("generated_at", "as_of_date"),
    "expanded_universe_candidate_event": ("ts_utc", "generated_at"),
    "expanded_universe_candidate_label": ("generated_at", "label_ts"),
    "expanded_universe_promotion_queue": ("generated_at", "as_of_date"),
    "expanded_universe_candidate_maturity": ("generated_at", "as_of_date"),
    "expanded_universe_watchlist": ("generated_at", "as_of_date"),
    "expanded_crypto_universe_shadow": ("generated_at", "as_of_date"),
    "symbol_quality_score": ("generated_at", "as_of_date"),
    "expanded_crypto_candidate_outcomes_by_symbol": ("generated_at", "as_of_date"),
    "expanded_crypto_recommendations": ("generated_at", "as_of_date"),
    "paper_strategy_runs": ("created_at", "as_of_date"),
    "paper_strategy_daily": ("created_at", "as_of_date"),
    "paper_strategy_proposal": ("created_at", "expires_at"),
    "paper_strategy_registry": ("created_at", "accepted_at", "paper_start_at"),
    "paper_strategy_migration_audit": ("created_at",),
    "paper_strategy_promotion_gate": ("created_at",),
    "strategy_cost_trust": ("created_at",),
    "paper_slippage_coverage": ("created_at", "as_of_date"),
    "sol_protect_paper_loss_attribution": (
        "generated_at_utc",
        "entry_ts",
        "as_of_date",
    ),
    "sol_protect_paper_loss_summary": ("generated_at_utc", "as_of_date"),
    "v5_missed_low_audit": ("generated_at_utc", "entry_ts", "as_of_date"),
    "v5_missed_low_by_symbol": ("generated_at_utc", "as_of_date"),
    "v5_missed_low_by_entry_reason": ("generated_at_utc", "as_of_date"),
    "v5_late_entry_chase_shadow": ("generated_at_utc", "ts_utc", "as_of_date"),
    "v5_late_entry_chase_threshold_advisory": ("generated_at_utc", "as_of_date"),
    "v5_late_entry_chase_threshold_by_symbol": ("generated_at_utc", "as_of_date"),
    "v5_pullback_reversal_shadow": ("generated_at_utc", "ts_utc", "as_of_date"),
    "v5_pullback_reversal_readiness": ("generated_at_utc", "as_of_date"),
    "v5_pullback_reversal_rule_comparison": ("generated_at_utc", "as_of_date"),
    "btc_probe_exit_policy_review": ("generated_at_utc", "entry_ts", "as_of_date"),
    "btc_probe_exit_policy_summary": ("generated_at_utc", "as_of_date"),
    "bnb_swing_exit_policy_review": ("generated_at_utc", "entry_ts", "as_of_date"),
    "bnb_exit_policy_v5_vs_quant_lab_consistency": (
        "generated_at_utc",
        "entry_ts",
        "as_of_date",
    ),
    "bnb_swing_exit_policy_summary": ("generated_at_utc", "as_of_date"),
    "exit_policy_review_sample": ("generated_at", "entry_ts", "as_of_date"),
    "exit_policy_review_summary": ("generated_at", "as_of_date"),
    "v5_entry_quality_advisory": ("generated_at_utc", "as_of_date"),
    "v5_entry_quality_history_late_entry_chase_threshold_sensitivity": (
        "generated_at_utc",
        "end_date",
    ),
    "v5_entry_quality_history_pullback_by_symbol": ("generated_at_utc", "end_date"),
    "v5_entry_quality_history_pullback_by_regime": ("generated_at_utc", "end_date"),
    "v5_entry_quality_history_pullback_by_horizon": ("generated_at_utc", "end_date"),
    "v5_entry_quality_history_anti_leakage_check": ("generated_at_utc", "end_date"),
    "v5_entry_quality_history_metrics": ("generated_at_utc", "end_date"),
    "okx_public_ws_health": ("last_message_at", "updated_at", "started_at"),
    "decision_audit": ("ingest_ts", "loaded_at"),
    "v5_quant_lab_usage": ("ingest_ts", "bundle_ts"),
    "v5_quant_lab_request": ("ts_utc", "ts", "ingest_ts", "bundle_ts"),
    "v5_quant_lab_compliance": ("ingest_ts", "bundle_ts"),
    "v5_quant_lab_cost_usage": ("ingest_ts", "bundle_ts"),
    "v5_quant_lab_fallback": ("ingest_ts", "bundle_ts"),
    "v5_cost_probe_p3_preflight": ("generated_at_utc", "ingest_ts", "bundle_ts"),
    "v5_cost_probe_live_execution_status": (
        "generated_at_utc",
        "ingest_ts",
        "bundle_ts",
    ),
    "v5_cost_probe_order_event": ("event_ts", "ingest_ts", "bundle_ts"),
    "v5_cost_probe_roundtrip_event": ("event_ts", "ingest_ts", "bundle_ts"),
    "v5_trade_opportunity_funnel": ("ts_utc", "ingest_ts", "bundle_ts"),
    "v5_fill_bill_cost_reconciliation": (
        "last_fill_ts",
        "generated_at",
        "ingest_ts",
        "bundle_ts",
    ),
    "v5_paper_strategy_run": ("created_at", "ingest_ts", "bundle_ts", "as_of_date"),
    "v5_paper_strategy_exit_quality": ("ingest_ts", "bundle_ts"),
    "v5_paper_strategy_proposal_ack": ("created_at", "ingest_ts", "bundle_ts"),
    "v5_paper_strategy_proposal_ack_current": (
        "created_at",
        "ingest_ts",
        "bundle_ts",
    ),
    "v5_paper_strategy_proposal_ack_history": (
        "created_at",
        "ingest_ts",
        "bundle_ts",
    ),
    "v5_paper_strategy_daily": ("created_at", "ingest_ts", "bundle_ts", "as_of_date"),
    "v5_paper_strategy_registry": ("updated_at", "created_at", "ingest_ts", "bundle_ts"),
    "v5_paper_strategy_registry_current": (
        "updated_at",
        "created_at",
        "ingest_ts",
        "bundle_ts",
    ),
    "v5_paper_strategy_registry_history": (
        "updated_at",
        "created_at",
        "ingest_ts",
        "bundle_ts",
    ),
    "v5_paper_strategy_trackers_current": (
        "updated_at",
        "created_at",
        "ingest_ts",
        "bundle_ts",
    ),
    "v5_paper_strategy_state": ("updated_at", "ingest_ts", "bundle_ts"),
    "v5_paper_strategy_signal": ("decision_ts", "signal_ts", "ingest_ts", "bundle_ts"),
    "v5_paper_strategy_quote_coverage": ("ingest_ts", "bundle_ts"),
    "v5_paper_strategy_cost_evidence": ("ingest_ts", "bundle_ts"),
    "v5_paper_strategy_error": ("ts_utc", "ingest_ts", "bundle_ts"),
    "v5_paper_strategy_restart_recovery": ("recovered_at", "ingest_ts", "bundle_ts"),
    "v5_quant_lab_contract_status": ("generated_at", "ingest_ts", "bundle_ts"),
    "v5_paper_slippage_coverage": ("created_at", "ingest_ts", "bundle_ts", "as_of_date"),
    "v5_expanded_universe_advisory_reader": (
        "generated_at",
        "ingest_ts",
        "bundle_ts",
        "ts_utc",
    ),
    "v5_expanded_universe_paper_runs": (
        "generated_at",
        "ingest_ts",
        "bundle_ts",
        "ts_utc",
    ),
    "v5_expanded_universe_paper_daily": (
        "generated_at",
        "ingest_ts",
        "bundle_ts",
        "paper_date",
        "as_of_date",
    ),
    "v5_bnb_profit_lock_shadow": ("ingest_ts", "bundle_ts", "exit_ts", "entry_ts"),
    "v5_bnb_negative_expectancy_attribution": ("ingest_ts", "bundle_ts", "exit_ts"),
    "v5_negative_expectancy_attribution": ("ingest_ts", "bundle_ts", "exit_ts"),
    "v5_negative_expectancy_attribution_silver": ("ingest_ts", "bundle_ts", "exit_ts"),
    "v5_btc_probe_entry_quality_audit": ("ingest_ts", "bundle_ts", "ts_utc", "created_at"),
    "v5_candidate_event": ("ts_utc", "ingest_ts", "bundle_ts"),
    "v5_candidate_label": ("created_at", "decision_ts", "ts_utc", "label_ts"),
    "v5_candidate_quality_daily": ("created_at", "date"),
    "v5_candidate_outcome_summary": ("created_at", "date"),
    "v5_decision_audit": ("ingest_ts", "bundle_ts"),
    "v5_trade_event": ("ts_utc", "ts", "ingest_ts", "bundle_ts"),
}

DATASET_FRESHNESS_TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    # `day` is the business aggregation date for opportunity-cost reports.
    # Freshness should track when the daily rollup was rebuilt.
    "quant_lab_opportunity_cost_daily": ("created_at", "day"),
    # `as_of_ts` is the advisory business date and may be midnight UTC for the
    # next Asia/Shanghai export day. Freshness should track production time.
    "strategy_opportunity_advisory": ("generated_at", "created_at", "as_of_ts"),
    # This is a current-view projection rebuilt from the selected V5 bundle.
    # Tracker creation is immutable and must not make a freshly ingested view stale.
    "paper_strategy_trackers_current": (
        "ingest_ts",
        "bundle_ts",
        "updated_at",
        "created_at",
    ),
}

CORE_DIAGNOSTIC_DATASETS = {
    "silver/market_bar": Path("silver") / "market_bar",
    "gold/cost_bucket_daily": Path("gold") / "cost_bucket_daily",
    "gold/cost_health_daily": Path("gold") / "cost_health_daily",
    "gold/gate_decision": Path("gold") / "gate_decision",
    "gold/risk_permission": Path("gold") / "risk_permission",
    "gold/strategy_health_daily": Path("gold") / "strategy_health_daily",
    "gold/v5_gate_compliance_daily": Path("gold") / "v5_gate_compliance_daily",
}

MARKET_BOOTSTRAP_COMMAND = (
    "qlab okx-fetch-candles --inst-id BTC-USDT --bar 1H --market-type SPOT "
    "--lake-root {lake_root} --limit 100"
)
V5_TELEMETRY_SYNC_COMMAND = (
    "qlab sync-v5-telemetry --config /etc/quant-lab/v5_telemetry_remote.yaml"
)
OKX_WS_UNIVERSE_SYMBOLS = (
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "BNB-USDT",
    "ADA-USDT",
    "ASTER-USDT",
    "BASED-USDT",
    "CHZ-USDT",
    "DASH-USDT",
    "FIL-USDT",
    "GRASS-USDT",
    "HYPE-USDT",
    "ICP-USDT",
    "IP-USDT",
    "JTO-USDT",
    "LINK-USDT",
    "LIT-USDT",
    "LPT-USDT",
    "LTC-USDT",
    "MON-USDT",
    "NEAR-USDT",
    "OKB-USDT",
    "ONDO-USDT",
    "PAXG-USDT",
    "PROS-USDT",
    "SUI-USDT",
    "TON-USDT",
    "TRUMP-USDT",
    "TRX-USDT",
    "WLD-USDT",
    "XAUT-USDT",
    "XRP-USDT",
    "ZEC-USDT",
)
OKX_WS_COLLECT_COMMAND = (
    f"qlab okx-ws-collect-universe --symbols {','.join(OKX_WS_UNIVERSE_SYMBOLS)} "
    "--channels tickers,trades,books5 --market-type SPOT --lake-root {lake_root} "
    "--flush-interval-seconds 60 --flush-max-messages 10000"
)
EXPERT_PACK_MISSING_MESSAGE = "qlab export-daily 尚未实现或尚未运行。"
BOOTSTRAP_GOLD_HEALTH_COMMAND = (
    "qlab bootstrap-gold-health --lake-root {lake_root} "
    "--strategy v5 --version bootstrap --day auto"
)
BOOTSTRAP_PLACEHOLDER_WARNING = "bootstrap 保守占位数据，不是 live-ready 证据"
DISPLAY_LIMIT = 500
MARKET_REGIME_CURRENT_LAG_HOURS = 2
WEB_HEAVY_DATASET_LIMITS = {
    "market_bar": 20_000,
    "trade_print": 20_000,
    "orderbook_snapshot": 20_000,
    "okx_public_ws": 5_000,
}
WEB_RESEARCH_SAMPLE_DATASETS = {
    "strategy_evidence_sample",
    "v5_candidate_event",
    "v5_candidate_label",
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
    "expanded_relative_strength_decision_sample",
    "expanded_universe_candidate_event",
    "expanded_universe_candidate_label",
    "expanded_crypto_universe_shadow",
}
WEB_FEATURE_SAMPLE_DATASETS = {"feature_value"}
WEB_AUDIT_SAMPLE_DATASETS = {"decision_audit"}
WEB_COST_SAMPLE_DATASETS = {"cost_bucket_daily"}
WEB_ADVISORY_SAMPLE_DATASETS = {
    "strategy_opportunity_advisory",
    "v5_entry_quality_advisory",
    "v5_missed_opportunity_audit",
    "v5_risk_on_multi_buy_shadow",
    "risk_on_multi_buy_shadow",
}
WEB_RECENT_LOOKBACK_HOURS = {
    "market_bar": 24 * 14,
    "trade_print": 6,
    "orderbook_snapshot": 6,
    "okx_public_ws": 6,
    "cost_bucket_daily": 24 * 180,
    "decision_audit": 24 * 14,
    "feature_value": 24 * 14,
    "strategy_opportunity_advisory": 24 * 14,
    "v5_missed_opportunity_audit": 24 * 14,
    "v5_risk_on_multi_buy_shadow": 24 * 14,
    "risk_on_multi_buy_shadow": 24 * 14,
    "v5_entry_quality_advisory": 24 * 14,
    "strategy_evidence_sample": 24 * 14,
    "v5_candidate_event": 24 * 14,
    "v5_candidate_label": 24 * 14,
    "trade_opportunity_event": 24 * 14,
    "trade_opportunity_label": 24 * 14,
    "trade_level_similarity_outcome": 24 * 14,
    "trade_level_judgment": 24 * 14,
    "trade_level_bucket_policy": 24 * 14,
    "trade_level_opportunity_queue": 24 * 14,
    "quant_lab_false_block_audit": 24 * 14,
    "v5_trade_learning_sample": 24 * 14,
    "v5_trade_outcome_attribution": 24 * 14,
    "quant_lab_opportunity_cost_event": 24 * 14,
    "quant_lab_opportunity_cost_daily": 24 * 14,
    "opportunity_cost_by_bucket": 24 * 14,
    "quant_lab_decision_regret": 24 * 14,
    "expanded_relative_strength_decision_sample": 24 * 14,
    "expanded_universe_candidate_event": 24 * 14,
    "expanded_universe_candidate_label": 24 * 14,
    "expanded_crypto_universe_shadow": 24 * 14,
}
WEB_HEAVY_METADATA_DATASETS = {"okx_public_ws", "trade_print", "orderbook_snapshot"}
WEB_RECENT_FILE_SAMPLE_DATASETS = (
    WEB_HEAVY_METADATA_DATASETS
    | WEB_RESEARCH_SAMPLE_DATASETS
    | WEB_FEATURE_SAMPLE_DATASETS
    | WEB_AUDIT_SAMPLE_DATASETS
    | WEB_COST_SAMPLE_DATASETS
    | WEB_ADVISORY_SAMPLE_DATASETS
)
WEB_HEAVY_EXACT_ROW_COUNT_FILE_LIMIT = 64
WEB_HEAVY_ROW_COUNT_SAMPLE_FILES = 32
WEB_HEAVY_INDEX_FALLBACK_FILE_LIMIT = 512
WEB_FULL_VALIDATION_FILE_LIMIT = 512
WEB_RECENT_FILE_LIMITS = {
    "trade_print": 6,
    "orderbook_snapshot": 6,
    "okx_public_ws": 384,
    "cost_bucket_daily": 384,
    "decision_audit": 384,
    "feature_value": 384,
    "strategy_opportunity_advisory": 384,
    "v5_missed_opportunity_audit": 384,
    "v5_risk_on_multi_buy_shadow": 384,
    "risk_on_multi_buy_shadow": 384,
    "v5_entry_quality_advisory": 384,
    "strategy_evidence_sample": 384,
    "v5_candidate_event": 384,
    "v5_candidate_label": 384,
    "trade_opportunity_event": 384,
    "trade_opportunity_label": 384,
    "trade_level_similarity_outcome": 384,
    "trade_level_judgment": 384,
    "trade_level_bucket_policy": 384,
    "trade_level_opportunity_queue": 384,
    "quant_lab_false_block_audit": 384,
    "v5_trade_learning_sample": 384,
    "v5_trade_outcome_attribution": 384,
    "quant_lab_opportunity_cost_event": 384,
    "quant_lab_opportunity_cost_daily": 384,
    "opportunity_cost_by_bucket": 384,
    "quant_lab_decision_regret": 384,
    "expanded_relative_strength_decision_sample": 384,
    "expanded_universe_candidate_event": 384,
    "expanded_universe_candidate_label": 384,
    "expanded_crypto_universe_shadow": 384,
}
PARQUET_MAGIC = b"PAR1"
MIN_PARQUET_SIZE_BYTES = 12
RESEARCH_STATUS_PRIORITY = {
    "DOWNGRADED_FROM_PAPER": 0,
    "REVIEW": 1,
    "PAPER": 2,
    "SHADOW": 3,
    "ACTIVE": 4,
    "ACTIVE_DIAGNOSTIC": 5,
    "PAUSED": 6,
    "BASELINE_ONLY": 7,
    "KILL": 8,
}
RESEARCH_ACTION_PRIORITY = {
    "CONTINUE_PAPER": 0,
    "CONTINUE_SHADOW_OR_REVIEW": 1,
    "CONTINUE_SHADOW": 2,
    "ACTIVE_DIAGNOSTIC": 3,
    "KEEP_RESEARCH": 4,
    "PAUSED_TO_WEEKLY": 5,
    "BASELINE_ONLY": 6,
    "CLOSE_RESEARCH": 7,
}
DECISION_PRIORITY = {
    "PAPER_READY": 0,
    "KEEP_SHADOW": 1,
    "REGIME_SHADOW": 2,
    "RESEARCH_ONLY": 3,
    "KILL": 4,
}
RECOMMENDED_MODE_PRIORITY = {"paper": 0, "shadow": 1, "research": 2, "audit": 3, "none": 4}


def _format_bps(value: Any) -> str:
    parsed = _as_float(value)
    return "n/a" if parsed is None else f"{parsed:.1f}bps"


def _format_pct(value: Any) -> str:
    parsed = _as_float(value)
    return "n/a" if parsed is None else f"{parsed * 100:.1f}%"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _friendly_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "候选"
    if text.startswith("regime_router:"):
        return "行情路由：" + _friendly_label(text.removeprefix("regime_router:"))
    displayed = display_value(text)
    return str(displayed) if displayed != text else text


@dataclass(frozen=True)
class DatasetState:
    name: str
    path: Path
    rows: int
    exists: bool
    parquet_file_count: int = 0
    warning: str | None = None


@dataclass(frozen=True)
class DatasetSnapshot:
    rows: int
    exists: bool
    parquet_file_count: int
    freshness: dict[str, Any]
    warning: str | None = None


_WEB_DATASET_CACHE: dict[tuple[Any, ...], tuple[float, Any]] = {}


def clear_web_cache() -> None:
    _WEB_DATASET_CACHE.clear()


def _web_cache_ttl_seconds() -> int:
    raw = os.environ.get("QUANT_LAB_WEB_DATASET_CACHE_TTL_SECONDS", "")
    try:
        value = int(raw) if raw else WEB_CACHE_DEFAULT_TTL_SECONDS
    except ValueError:
        return WEB_CACHE_DEFAULT_TTL_SECONDS
    return max(value, 0)


def _web_cache_get(key: tuple[Any, ...], *, event: str, dataset_name: str = "") -> Any | None:
    ttl = _web_cache_ttl_seconds()
    if ttl <= 0:
        perf.record_event(event, dataset_name=dataset_name, cache_miss=True)
        return None
    cached = _WEB_DATASET_CACHE.get(key)
    now = time_module.monotonic()
    if cached is None or now - cached[0] > ttl:
        perf.record_event(event, dataset_name=dataset_name, cache_miss=True)
        return None
    perf.record_event(event, dataset_name=dataset_name, cache_hit=True)
    return cached[1]


def _web_cache_set(key: tuple[Any, ...], value: Any) -> Any:
    if _web_cache_ttl_seconds() > 0:
        _WEB_DATASET_CACHE[key] = (time_module.monotonic(), value)
    return value


def read_dataset(lake_root: str | Path, dataset_name: str) -> pl.DataFrame:
    df, _warning = read_dataset_with_warning(lake_root, dataset_name)
    return df


def read_dataset_with_warning(
    lake_root: str | Path,
    dataset_name: str,
) -> tuple[pl.DataFrame, str | None]:
    dataset_path = dataset_path_for(lake_root, dataset_name)
    return _read_parquet_dataset_with_warning(dataset_path, dataset_name)


def read_recent_dataset_with_warning(
    lake_root: str | Path,
    dataset_name: str,
    *,
    limit: int | None = None,
    lookback_hours: int | None = None,
) -> tuple[pl.DataFrame, str | None]:
    dataset_path = dataset_path_for(lake_root, dataset_name)
    row_limit = limit or WEB_HEAVY_DATASET_LIMITS.get(dataset_name, DISPLAY_LIMIT)
    hours = lookback_hours or WEB_RECENT_LOOKBACK_HOURS.get(dataset_name, 6)
    cache_key = (
        "read_recent_dataset_with_warning",
        str(Path(lake_root).resolve()),
        dataset_name,
        row_limit,
        hours,
        _web_dataset_source_signature(dataset_path),
    )
    cached = _web_cache_get(
        cache_key,
        event="read_recent_dataset_with_warning",
        dataset_name=dataset_name,
    )
    if cached is not None:
        return cached

    if dataset_name in WEB_RECENT_FILE_SAMPLE_DATASETS:
        files, warning = _recent_valid_parquet_files(dataset_path, dataset_name)
    else:
        files, warning = _valid_parquet_files_with_warning(dataset_path, dataset_name)
    if not files:
        return _web_cache_set(cache_key, (pl.DataFrame(), warning))

    try:
        with perf.timed(
            "dataset_read",
            dataset_name=dataset_name,
            files_scanned=len(files),
        ):
            lazy = _scan_parquet_files(files)
            frame = _collect_recent_lazy_frame(
                lazy,
                dataset_name,
                limit=row_limit,
                lookback_hours=hours,
            )
        perf.record_event("dataset_rows", dataset_name=dataset_name, rows_rendered=frame.height)
        return _web_cache_set(cache_key, (frame, warning))
    except Exception as exc:
        return _web_cache_set(cache_key, (pl.DataFrame(), f"{dataset_name} 抽样读取失败：{exc}"))


def _read_web_display_dataset_with_warning(
    lake_root: str | Path,
    dataset_name: str,
) -> tuple[pl.DataFrame, str | None]:
    dataset_path = dataset_path_for(lake_root, dataset_name)
    cache_key = (
        "_read_web_display_dataset_with_warning",
        str(Path(lake_root).resolve()),
        dataset_name,
        _web_dataset_source_signature(dataset_path),
    )
    cached = _web_cache_get(
        cache_key,
        event="_read_web_display_dataset_with_warning",
        dataset_name=dataset_name,
    )
    if cached is not None:
        return cached

    sampled_datasets = (
        WEB_RESEARCH_SAMPLE_DATASETS
        | WEB_FEATURE_SAMPLE_DATASETS
        | WEB_AUDIT_SAMPLE_DATASETS
        | WEB_COST_SAMPLE_DATASETS
        | WEB_ADVISORY_SAMPLE_DATASETS
    )
    if dataset_name in sampled_datasets:
        result = read_recent_dataset_with_warning(lake_root, dataset_name)
    else:
        result = read_dataset_with_warning(lake_root, dataset_name)
    return _web_cache_set(cache_key, result)


def dataset_path_for(lake_root: str | Path, dataset_name: str) -> Path:
    relative = DATASET_PATHS.get(dataset_name)
    if relative is None:
        relative = Path(dataset_name)
    return Path(lake_root) / relative


def dataset_freshness_payload(
    dataset_name: str,
    df: pl.DataFrame,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if df.is_empty():
        return {
            "freshness_seconds": None,
            "freshness_status": "missing",
            "timestamp_column": None,
            "latest_timestamp": None,
        }
    latest, column = latest_dataset_timestamp(
        dataset_name,
        df,
        timestamp_columns=DATASET_FRESHNESS_TIMESTAMP_COLUMNS.get(dataset_name),
    )
    if latest is None:
        return {
            "freshness_seconds": None,
            "freshness_status": "unknown",
            "timestamp_column": column,
            "latest_timestamp": None,
        }
    reference_time = now.astimezone(UTC) if now else datetime.now(UTC)
    freshness_reference = latest
    extras: dict[str, Any] = {}
    if dataset_name == "market_bar":
        timeframe = _latest_market_bar_timeframe(df, latest) or DEFAULT_MARKET_BAR_TIMEFRAME
        close_ts = market_bar_close_ts(latest, timeframe)
        if close_ts is not None:
            freshness_reference = close_ts
            extras.update(
                {
                    "latest_close_timestamp": close_ts.isoformat(),
                    "timeframe": timeframe,
                    "freshness_reference": "bar_close",
                }
            )
    seconds = int((reference_time - freshness_reference).total_seconds())
    if seconds < -5 * 60:
        status = "future"
        extras["future_seconds"] = abs(seconds)
        extras["freshness_warning"] = "timestamp_after_export_reference"
    elif seconds <= 6 * 60 * 60:
        seconds = max(seconds, 0)
        status = "fresh"
    elif seconds <= 24 * 60 * 60:
        status = "delayed"
    else:
        status = "stale"
    return {
        "freshness_seconds": seconds,
        "freshness_status": status,
        "timestamp_column": column,
        "latest_timestamp": latest.isoformat(),
        **extras,
    }


def latest_dataset_timestamp(
    dataset_name: str,
    df: pl.DataFrame,
    *,
    timestamp_columns: tuple[str, ...] | None = None,
) -> tuple[datetime | None, str | None]:
    if df.is_empty():
        return None, None
    columns = timestamp_columns or DATASET_TIMESTAMP_COLUMNS.get(
        dataset_name,
        ("ts", "created_at", "ingest_ts"),
    )
    seen_timestamp_column: str | None = None
    for column in columns:
        if column not in df.columns:
            continue
        seen_timestamp_column = column
        values = [_coerce_timestamp(value) for value in df[column].to_list()]
        parsed = [value for value in values if value is not None]
        if parsed:
            return max(parsed), column
    return None, seen_timestamp_column


def _freshness_timestamp_columns(
    dataset_name: str,
    override: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if override is not None:
        return override
    return DATASET_FRESHNESS_TIMESTAMP_COLUMNS.get(dataset_name)


def _latest_market_bar_timeframe(df: pl.DataFrame, latest_ts: datetime | None) -> str | None:
    if (
        latest_ts is None
        or df.is_empty()
        or "timeframe" not in df.columns
        or "ts" not in df.columns
    ):
        return None
    try:
        normalized = df
        if normalized.schema.get("ts") == pl.String:
            normalized = normalized.with_columns(
                pl.col("ts").str.to_datetime(time_zone="UTC", strict=False)
            )
        else:
            normalized = normalized.with_columns(
                pl.col("ts").cast(pl.Datetime(time_zone="UTC")).alias("ts")
            )
        latest_utc = (
            latest_ts.astimezone(UTC) if latest_ts.tzinfo else latest_ts.replace(tzinfo=UTC)
        )
        row = normalized.filter(pl.col("ts") == latest_utc).tail(1)
        if row.is_empty():
            row = normalized.sort("ts").tail(1)
        value = row.item(0, "timeframe")
    except Exception:
        return None
    text = str(value or "").strip()
    return text or None


def _read_parquet_dataset_with_warning(
    dataset_path: str | Path,
    dataset_label: str,
) -> tuple[pl.DataFrame, str | None]:
    path = Path(dataset_path)
    invalid_files = invalid_parquet_files(path)
    try:
        df = read_parquet_dataset(path)
    except Exception as exc:
        return pl.DataFrame(), (
            f"{dataset_label} 数据集读取失败，已按空数据处理：{path}；错误：{exc}"
        )
    if invalid_files:
        rendered = ", ".join(str(file_path) for file_path in invalid_files[:5])
        suffix = "" if len(invalid_files) <= 5 else f"（另有 {len(invalid_files) - 5} 个）"
        return df, (f"{dataset_label} 已忽略无效 Parquet 文件：{rendered}{suffix}")
    return df, None


def _dataset_snapshot(
    lake_root: str | Path,
    dataset_name: str,
    *,
    timestamp_columns: tuple[str, ...] | None = None,
    now: datetime | None = None,
) -> DatasetSnapshot:
    if dataset_name in WEB_HEAVY_METADATA_DATASETS and timestamp_columns is None:
        return _heavy_dataset_snapshot(lake_root, dataset_name, now=now)

    path = dataset_path_for(lake_root, dataset_name)
    meta_snapshot = _dataset_snapshot_from_meta(path, now=now)
    if meta_snapshot is not None:
        return meta_snapshot

    files, warning = _valid_parquet_files_with_warning(path, dataset_name)
    if warning:
        perf.record_event("dataset_snapshot_warning", dataset_name=dataset_name, warning=warning)
    perf.record_event(
        "dataset_snapshot_warning",
        dataset_name=dataset_name,
        warning=WEB_SNAPSHOT_META_MISSING_WARNING,
    )
    if not files:
        return DatasetSnapshot(
            rows=0,
            exists=path.exists(),
            parquet_file_count=0,
            freshness=_freshness_payload(None, None, is_empty=True, now=now),
            warning=warning,
        )

    try:
        lazy = _scan_parquet_files(files)
        schema = lazy.collect_schema()
        rows = int(lazy.select(pl.len().alias("rows")).collect().item())
        latest, column = _latest_lazy_timestamp(
            dataset_name,
            lazy,
            schema,
            timestamp_columns=_freshness_timestamp_columns(dataset_name, timestamp_columns),
        )
    except Exception as exc:
        return DatasetSnapshot(
            rows=0,
            exists=path.exists(),
            parquet_file_count=len(files),
            freshness=_freshness_payload(None, None, is_empty=True, now=now),
            warning=f"{dataset_name} 元数据读取失败：{exc}",
        )

    return DatasetSnapshot(
        rows=rows,
        exists=path.exists(),
        parquet_file_count=len(files),
        freshness=_freshness_payload(latest, column, is_empty=rows == 0, now=now),
        warning=warning,
    )


def _scan_parquet_files(files: list[Path]) -> pl.LazyFrame:
    try:
        return pl.scan_parquet(
            [str(file_path) for file_path in files],
            hive_partitioning=False,
            missing_columns="insert",
            extra_columns="ignore",
        )
    except TypeError:
        return pl.scan_parquet([str(file_path) for file_path in files])


def _heavy_dataset_snapshot(
    lake_root: str | Path,
    dataset_name: str,
    *,
    now: datetime | None = None,
) -> DatasetSnapshot:
    path = dataset_path_for(lake_root, dataset_name)
    meta_snapshot = _dataset_snapshot_from_meta(path, now=now)
    if meta_snapshot is not None:
        return meta_snapshot
    files, warning = _metadata_snapshot_files(path, dataset_name)
    if warning:
        perf.record_event("dataset_snapshot_warning", dataset_name=dataset_name, warning=warning)
    perf.record_event(
        "dataset_snapshot_warning",
        dataset_name=dataset_name,
        warning=WEB_SNAPSHOT_META_MISSING_WARNING,
    )
    if not files:
        return DatasetSnapshot(
            rows=0,
            exists=path.exists(),
            parquet_file_count=0,
            freshness=_freshness_payload(None, None, is_empty=True, now=now),
            warning=warning,
        )

    rows = _parquet_metadata_row_count(files)
    latest = _latest_file_mtime(files)
    return DatasetSnapshot(
        rows=rows,
        exists=path.exists(),
        parquet_file_count=len(files),
        freshness=_freshness_payload(latest, "file_mtime", is_empty=rows == 0, now=now),
        warning=warning,
    )


def _metadata_snapshot_files(path: Path, dataset_name: str) -> tuple[list[Path], str | None]:
    candidates = _parquet_file_candidates(path)
    if len(candidates) > WEB_FULL_VALIDATION_FILE_LIMIT:
        return candidates, None
    invalid_files = [
        file_path for file_path in candidates if not _is_valid_parquet_file_path(file_path)
    ]
    valid_files = [file_path for file_path in candidates if file_path not in set(invalid_files)]
    return valid_files, _invalid_parquet_warning(dataset_name, invalid_files)


def _parquet_metadata_row_count(files: list[Path]) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return len(files)

    if len(files) > WEB_HEAVY_EXACT_ROW_COUNT_FILE_LIMIT:
        return _sampled_parquet_metadata_row_count(files, pq)

    total = 0
    for file_path in files:
        try:
            total += int(pq.ParquetFile(file_path).metadata.num_rows)
        except Exception:
            total += 0
    return total


def _sampled_parquet_metadata_row_count(files: list[Path], pq: Any) -> int:
    sample_size = min(WEB_HEAVY_ROW_COUNT_SAMPLE_FILES, len(files))
    half = max(sample_size // 2, 1)
    sampled = files[:half] + files[-half:]
    sampled = list(dict.fromkeys(sampled))
    counts: list[int] = []
    for file_path in sampled:
        try:
            counts.append(int(pq.ParquetFile(file_path).metadata.num_rows))
        except Exception:
            continue
    if not counts:
        return len(files)
    average_rows = sum(counts) / len(counts)
    return max(int(average_rows * len(files)), len(counts))


def _latest_file_mtime(files: list[Path]) -> datetime | None:
    if not files:
        return None
    try:
        latest = max(file_path.stat().st_mtime for file_path in files)
    except OSError:
        return None
    return datetime.fromtimestamp(latest, tz=UTC)


def _recent_valid_parquet_files(path: Path, dataset_name: str) -> tuple[list[Path], str | None]:
    candidates, source_warning = _recent_parquet_file_candidates_with_warning(path)
    max_files = WEB_RECENT_FILE_LIMITS.get(dataset_name, WEB_FULL_VALIDATION_FILE_LIMIT)
    selected = candidates[-max_files:] if len(candidates) > max_files else candidates
    invalid_files = [
        file_path for file_path in selected if not _is_valid_parquet_file_path(file_path)
    ]
    invalid = set(invalid_files)
    valid_files = [file_path for file_path in selected if file_path not in invalid]
    return valid_files, _combine_warnings(
        source_warning,
        _invalid_parquet_warning(dataset_name, invalid_files),
    )


def _recent_parquet_file_candidates(path: Path) -> list[Path]:
    candidates, _warning = _recent_parquet_file_candidates_with_warning(path)
    return candidates


def _recent_parquet_file_candidates_with_warning(path: Path) -> tuple[list[Path], str | None]:
    candidates, warning = _parquet_file_candidates_with_warning(path)
    batch_files = [file_path for file_path in candidates if file_path.name.startswith("batch_")]
    if batch_files:
        candidates = batch_files
    try:
        return (
            sorted(candidates, key=lambda file_path: (file_path.stat().st_mtime, file_path.name)),
            warning,
        )
    except OSError:
        return sorted(candidates), warning


def _latest_lazy_timestamp(
    dataset_name: str,
    lazy: pl.LazyFrame,
    schema: pl.Schema,
    *,
    timestamp_columns: tuple[str, ...] | None = None,
) -> tuple[datetime | None, str | None]:
    columns = timestamp_columns or DATASET_TIMESTAMP_COLUMNS.get(
        dataset_name,
        ("ts", "created_at", "ingest_ts"),
    )
    seen_timestamp_column: str | None = None
    for column in columns:
        if column not in schema:
            continue
        seen_timestamp_column = column
        value = (
            lazy.select(_lazy_timestamp_expr(schema, column).max().alias(column)).collect().item()
        )
        latest = _coerce_timestamp(value)
        if latest is not None:
            return latest, column
    return None, seen_timestamp_column


def _collect_recent_lazy_frame(
    lazy: pl.LazyFrame,
    dataset_name: str,
    *,
    limit: int,
    lookback_hours: int,
) -> pl.DataFrame:
    schema = lazy.collect_schema()
    timestamp_column = next(
        (
            column
            for column in (
                _freshness_timestamp_columns(dataset_name, None)
                or DATASET_TIMESTAMP_COLUMNS.get(dataset_name, ("ts", "created_at"))
            )
            if column in schema
        ),
        None,
    )
    if timestamp_column is None:
        return lazy.tail(limit).collect()

    latest, _column = _latest_lazy_timestamp(
        dataset_name,
        lazy,
        schema,
        timestamp_columns=(timestamp_column,),
    )
    if latest is None:
        return lazy.tail(limit).collect()

    threshold = latest - timedelta(hours=lookback_hours)
    timestamp_expr = _lazy_timestamp_expr(schema, timestamp_column).alias("__qlab_recent_ts")
    try:
        frame = (
            lazy.with_columns(timestamp_expr)
            .filter(pl.col("__qlab_recent_ts") >= threshold)
            .drop("__qlab_recent_ts")
            .collect()
        )
    except Exception:
        frame = lazy.tail(limit).collect()
    if frame.height > limit:
        frame = _tail_recent_sample(frame, timestamp_column, limit)
    return frame


def _lazy_timestamp_expr(schema: pl.Schema, column: str) -> pl.Expr:
    expression = pl.col(column)
    if schema.get(column) == pl.String:
        return expression.str.to_datetime(time_zone="UTC", strict=False)
    return expression.cast(pl.Datetime(time_zone="UTC"), strict=False)


def _tail_recent_sample(df: pl.DataFrame, column: str, limit: int) -> pl.DataFrame:
    if "symbol" not in df.columns:
        return _tail_by_column(df, column, limit)
    try:
        symbol_count = max(df["symbol"].n_unique(), 1)
        per_symbol_limit = max(limit // symbol_count, 1)
        return (
            df.sort(["symbol", column])
            .group_by("symbol", maintain_order=True)
            .tail(per_symbol_limit)
            .sort(column)
        )
    except Exception:
        return _tail_by_column(df, column, limit)


def _tail_by_column(df: pl.DataFrame, column: str, limit: int) -> pl.DataFrame:
    if column not in df.columns:
        return df.tail(limit)
    try:
        return df.sort(column).tail(limit)
    except Exception:
        return df.tail(limit)


def _snapshot_latest_timestamp(snapshot: DatasetSnapshot) -> datetime | None:
    value = snapshot.freshness.get("latest_timestamp")
    return _coerce_timestamp(value)


def _freshness_payload(
    latest: datetime | None,
    column: str | None,
    *,
    is_empty: bool,
    now: datetime | None = None,
    latest_close: datetime | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    if is_empty:
        return {
            "freshness_seconds": None,
            "freshness_status": "missing",
            "timestamp_column": column,
            "latest_timestamp": None,
        }
    if latest is None:
        return {
            "freshness_seconds": None,
            "freshness_status": "unknown",
            "timestamp_column": column,
            "latest_timestamp": None,
        }
    reference_time = now.astimezone(UTC) if now else datetime.now(UTC)
    freshness_reference = latest_close or latest
    seconds = max(int((reference_time - freshness_reference).total_seconds()), 0)
    if seconds <= 6 * 60 * 60:
        status = "fresh"
    elif seconds <= 24 * 60 * 60:
        status = "delayed"
    else:
        status = "stale"
    payload = {
        "freshness_seconds": seconds,
        "freshness_status": status,
        "timestamp_column": column,
        "latest_timestamp": latest.isoformat(),
    }
    if latest_close is not None:
        payload["latest_close_timestamp"] = latest_close.isoformat()
        payload["freshness_reference"] = "bar_close"
    if timeframe:
        payload["timeframe"] = timeframe
    return payload


def _valid_parquet_files(path: Path, *, invalid_files: list[Path] | None = None) -> list[Path]:
    if invalid_files is None:
        files, _warning = _valid_parquet_files_with_warning(path, _dataset_name_from_path(path))
        return files
    candidates = _parquet_file_candidates(path)
    invalid = set(invalid_files if invalid_files is not None else invalid_parquet_files(path))
    return [file_path for file_path in candidates if file_path not in invalid]


def _valid_parquet_files_with_warning(
    path: Path,
    dataset_name: str,
) -> tuple[list[Path], str | None]:
    candidates, source_warning = _parquet_file_candidates_with_warning(path)
    invalid_files = [
        file_path for file_path in candidates if not _is_valid_parquet_file_path(file_path)
    ]
    warning = _combine_warnings(
        source_warning,
        _invalid_parquet_warning(dataset_name, invalid_files),
    )
    return [file_path for file_path in candidates if file_path not in set(invalid_files)], warning


def _parquet_file_candidates_with_warning(path: Path) -> tuple[list[Path], str | None]:
    if path.is_file() and path.suffix == ".parquet":
        return ([] if _is_internal_lake_path(path) else [path]), None

    indexed = _indexed_files_for_web(path)
    if indexed is not None:
        if not indexed and path.exists():
            return _parquet_file_index_fallback_candidates(path)
        if path.exists() and _web_file_index_appears_stale(path, indexed):
            return _merge_indexed_with_fallback_candidates(path, indexed), None
        perf.record_event(
            "web_file_index_lookup",
            dataset_name=_dataset_name_from_path(path),
            files_scanned=len(indexed),
        )
        return indexed, None

    lake_root = _infer_lake_root_from_path(path)
    if lake_root is not None:
        if path.exists():
            return _parquet_file_index_fallback_candidates(path)
        return [], None

    candidates = _parquet_file_candidates_rglob(path)
    if path.exists():
        perf.record_event(
            "web_file_index_lookup",
            dataset_name=_dataset_name_from_path(path),
            rglob_fallback=True,
            files_scanned=len(candidates),
            warning=WEB_FILE_INDEX_FALLBACK_WARNING,
        )
    return candidates, WEB_FILE_INDEX_FALLBACK_WARNING if path.exists() else None


def _parquet_file_index_fallback_candidates(path: Path) -> tuple[list[Path], str | None]:
    dataset_name = _dataset_name_from_path(path)
    if dataset_name in WEB_HEAVY_METADATA_DATASETS:
        candidates, truncated = _bounded_parquet_file_candidates_rglob(
            path,
            max_files=WEB_HEAVY_INDEX_FALLBACK_FILE_LIMIT,
        )
        perf.record_event(
            "web_file_index_fallback_rglob",
            dataset_name=dataset_name,
            files_scanned=len(candidates),
            rglob_fallback=True,
            warning=WEB_FILE_INDEX_FALLBACK_WARNING if truncated or not candidates else None,
        )
        if candidates and not truncated:
            return candidates, None
        return candidates, WEB_FILE_INDEX_FALLBACK_WARNING
    candidates = _parquet_file_candidates_rglob(path)
    perf.record_event(
        "web_file_index_fallback_rglob",
        dataset_name=dataset_name,
        files_scanned=len(candidates),
    )
    if candidates:
        return candidates, None
    return [], WEB_FILE_INDEX_FALLBACK_WARNING


def _merge_indexed_with_fallback_candidates(path: Path, indexed: list[Path]) -> list[Path]:
    dataset_name = _dataset_name_from_path(path)
    if dataset_name in WEB_HEAVY_METADATA_DATASETS:
        return indexed
    discovered = _parquet_file_candidates_rglob(path)
    merged = sorted({*indexed, *discovered})
    perf.record_event(
        "web_file_index_fallback_rglob",
        dataset_name=dataset_name,
        files_scanned=len(discovered),
        rglob_fallback=True,
        warning=(
            "stale_index_merge:"
            f"indexed={len(indexed)};discovered={len(discovered)};merged={len(merged)}"
        ),
    )
    return merged


def _web_file_index_appears_stale(path: Path, indexed: list[Path]) -> bool:
    if not indexed:
        return False
    dataset_name = _dataset_name_from_path(path)
    if dataset_name in WEB_HEAVY_METADATA_DATASETS:
        return False
    latest_indexed_mtime = _latest_existing_file_mtime(indexed)
    latest_path_hint_mtime = _path_parquet_mtime_hint(path)
    if latest_indexed_mtime is None or latest_path_hint_mtime is None:
        return False
    return latest_path_hint_mtime > latest_indexed_mtime + 1e-6


def _latest_existing_file_mtime(files: list[Path]) -> float | None:
    latest: float | None = None
    for file_path in files:
        try:
            if not file_path.exists() or not file_path.is_file():
                continue
            mtime = file_path.stat().st_mtime
        except OSError:
            continue
        latest = mtime if latest is None else max(latest, mtime)
    return latest


def _path_parquet_mtime_hint(
    path: Path,
    *,
    max_depth: int = 4,
    max_entries: int = 512,
) -> float | None:
    latest: float | None = None
    seen = 0

    def visit(candidate: Path, depth: int) -> None:
        nonlocal latest, seen
        if seen >= max_entries:
            return
        try:
            if _is_internal_lake_path(candidate):
                return
            stat = candidate.stat()
        except OSError:
            return
        seen += 1
        if candidate.is_file() and candidate.suffix == ".parquet":
            latest = stat.st_mtime if latest is None else max(latest, stat.st_mtime)
        if depth >= max_depth or not candidate.is_dir():
            return
        try:
            children = sorted(candidate.iterdir())
        except OSError:
            return
        for child in children:
            visit(child, depth + 1)

    visit(path, 0)
    return latest


def _parquet_file_candidates(path: Path) -> list[Path]:
    candidates, _warning = _parquet_file_candidates_with_warning(path)
    return candidates


def _parquet_file_candidates_rglob(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [] if _is_internal_lake_path(path) else [path]
    if not path.exists():
        return []
    return sorted(
        candidate for candidate in path.rglob("*.parquet") if not _is_internal_lake_path(candidate)
    )


def _bounded_parquet_file_candidates_rglob(
    path: Path,
    *,
    max_files: int,
) -> tuple[list[Path], bool]:
    if path.is_file() and path.suffix == ".parquet":
        return ([] if _is_internal_lake_path(path) else [path]), False
    if not path.exists():
        return [], False
    candidates: list[Path] = []
    truncated = False
    for candidate in path.rglob("*.parquet"):
        if _is_internal_lake_path(candidate):
            continue
        if len(candidates) >= max_files:
            truncated = True
            break
        candidates.append(candidate)
    return sorted(candidates), truncated


def _indexed_files_for_web(path: Path) -> list[Path] | None:
    lake_root = _infer_lake_root_from_path(path)
    if lake_root is None:
        return None
    try:
        relative_dataset = str(path.relative_to(lake_root)).replace("\\", "/")
    except ValueError:
        return None
    paths_by_dataset = _web_file_index_paths_by_dataset(lake_root)
    if not paths_by_dataset:
        return None
    files = paths_by_dataset.get(relative_dataset)
    if not files:
        return None
    return sorted(
        file_path
        for file_path in files
        if file_path.is_file() and not _is_internal_lake_path(file_path)
    )


def _web_file_index_paths_by_dataset(lake_root: Path) -> dict[str, list[Path]]:
    cache_key = (
        "web_file_index_paths_by_dataset",
        str(lake_root.resolve()),
        _lake_file_index_signature(lake_root),
    )
    cached = _web_cache_get(cache_key, event="web_file_index_paths_by_dataset")
    if cached is not None:
        return cached
    try:
        index = read_parquet_dataset(lake_root / "bronze" / "lake_file_index")
    except Exception:
        return _web_cache_set(cache_key, {})
    required = {"dataset", "path"}
    if index.is_empty() or not required.issubset(set(index.columns)):
        return _web_cache_set(cache_key, {})
    normalized = index.select(
        [
            pl.col("dataset").cast(pl.Utf8, strict=False).alias("dataset"),
            pl.col("path").cast(pl.Utf8, strict=False).alias("path"),
        ]
    ).drop_nulls()
    paths_by_dataset: dict[str, list[Path]] = {}
    for row in normalized.iter_rows(named=True):
        dataset = str(row.get("dataset") or "")
        relative_path = str(row.get("path") or "")
        if not dataset or not relative_path:
            continue
        paths_by_dataset.setdefault(dataset, []).append(lake_root / relative_path)
    return _web_cache_set(cache_key, paths_by_dataset)


def _web_file_index_dataset_signature(path: Path) -> tuple[Any, ...] | None:
    lake_root = _infer_lake_root_from_path(path)
    if lake_root is None:
        return None
    try:
        relative_dataset = str(path.relative_to(lake_root)).replace("\\", "/")
    except ValueError:
        return None
    signatures = _web_file_index_dataset_signatures(lake_root)
    return signatures.get(relative_dataset)


def _web_file_index_dataset_signatures(lake_root: Path) -> dict[str, tuple[Any, ...]]:
    cache_key = (
        "web_file_index_dataset_signatures",
        str(lake_root.resolve()),
        _lake_file_index_signature(lake_root),
    )
    cached = _web_cache_get(cache_key, event="web_file_index_dataset_signatures")
    if cached is not None:
        return cached
    try:
        index = read_parquet_dataset(lake_root / "bronze" / "lake_file_index")
    except Exception:
        return _web_cache_set(cache_key, {})
    required = {"dataset", "path"}
    if index.is_empty() or not required.issubset(set(index.columns)):
        return _web_cache_set(cache_key, {})
    available = set(index.columns)
    selected_columns = [
        column
        for column in (
            "dataset",
            "path",
            "source_sha",
            "mtime_ns",
            "file_size",
            "row_count",
            "min_ts",
            "max_ts",
            "indexed_at",
            "index_version",
        )
        if column in available
    ]
    normalized = (
        index.select([pl.col(column).cast(pl.Utf8, strict=False) for column in selected_columns])
        .drop_nulls(subset=["dataset", "path"])
        .sort(["dataset", "path"])
    )
    accumulators: dict[str, dict[str, Any]] = {}
    for row in normalized.iter_rows(named=True):
        dataset = str(row.get("dataset") or "")
        path_text = str(row.get("path") or "")
        if not dataset or not path_text:
            continue
        accumulator = accumulators.setdefault(
            dataset,
            {
                "count": 0,
                "row_count_sum": 0,
                "file_size_sum": 0,
                "max_mtime_ns": "",
                "max_ts": "",
                "max_indexed_at": "",
                "digest": hashlib.blake2b(digest_size=16),
            },
        )
        accumulator["count"] += 1
        for source_column, target_key in (
            ("row_count", "row_count_sum"),
            ("file_size", "file_size_sum"),
        ):
            try:
                accumulator[target_key] += int(row.get(source_column) or 0)
            except (TypeError, ValueError):
                pass
        for source_column, target_key in (
            ("mtime_ns", "max_mtime_ns"),
            ("max_ts", "max_ts"),
            ("indexed_at", "max_indexed_at"),
        ):
            value = str(row.get(source_column) or "")
            if value > accumulator[target_key]:
                accumulator[target_key] = value
        digest = accumulator["digest"]
        digest.update(
            "|".join(str(row.get(column) or "") for column in selected_columns).encode(
                "utf-8", "surrogatepass"
            )
        )
        digest.update(b"\n")
    signatures = {
        dataset: (
            "lake_file_index_dataset",
            values["count"],
            values["row_count_sum"],
            values["file_size_sum"],
            values["max_mtime_ns"],
            values["max_ts"],
            values["max_indexed_at"],
            values["digest"].hexdigest(),
        )
        for dataset, values in accumulators.items()
    }
    return _web_cache_set(cache_key, signatures)


def _infer_lake_root_from_path(path: Path) -> Path | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part in {"bronze", "silver", "gold"} and index > 0:
            return Path(*parts[:index])
    return None


def _dataset_name_from_path(path: Path) -> str:
    lake_root = _infer_lake_root_from_path(path)
    if lake_root is None:
        return str(path)
    try:
        relative = str(path.relative_to(lake_root)).replace("\\", "/")
    except ValueError:
        return str(path)
    for name, dataset_path in DATASET_PATHS.items():
        if str(dataset_path).replace("\\", "/") == relative:
            return name
    return relative


def _dataset_snapshot_from_meta(
    path: Path,
    *,
    now: datetime | None = None,
) -> DatasetSnapshot | None:
    meta = _snapshot_meta_for_dataset(path)
    if not meta:
        return None
    rows = _int_or_zero(meta.get("row_count", meta.get("rows")))
    file_count = _int_or_zero(meta.get("file_count", meta.get("parquet_file_count")))
    latest = _coerce_timestamp(
        meta.get("latest_timestamp")
        or meta.get("max_timestamp")
        or meta.get("latest_ts")
        or meta.get("generated_at")
    )
    dataset_name = _dataset_name_from_path(path)
    if (
        latest is None
        and rows > 0
        and DATASET_TIMESTAMP_COLUMNS.get(dataset_name)
    ):
        return None
    timeframe = None
    latest_close = None
    if dataset_name == "market_bar":
        timeframe = str(meta.get("latest_timeframe") or DEFAULT_MARKET_BAR_TIMEFRAME)
        latest_close = _coerce_timestamp(
            meta.get("latest_close_timestamp") or meta.get("latest_close_ts")
        )
        if latest_close is None and latest is not None:
            latest_close = market_bar_close_ts(latest, timeframe)
    column = str(meta.get("timestamp_column") or "snapshot_meta")
    return DatasetSnapshot(
        rows=rows,
        exists=path.exists(),
        parquet_file_count=file_count,
        freshness=_freshness_payload(
            latest,
            column,
            is_empty=rows == 0,
            now=now,
            latest_close=latest_close,
            timeframe=timeframe,
        ),
        warning=None,
    )


def _snapshot_meta_for_dataset(path: Path) -> dict[str, Any] | None:
    meta_path = path / "_snapshot_meta.json"
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def web_dataset_source_signature(path: Path) -> tuple[Any, ...]:
    """Return the bounded source signature used by Web cache invalidation."""
    return _web_dataset_source_signature(path)


def _web_dataset_source_signature(path: Path) -> tuple[Any, ...]:
    meta = _snapshot_meta_for_dataset(path)
    if meta:
        meta_path = path / "_snapshot_meta.json"
        try:
            meta_stat = meta_path.stat()
        except OSError:
            meta_stat = None
        try:
            root_stat = path.stat()
        except OSError:
            root_stat = None
        meta_fields = (
            "source_sha",
            "generated_at",
            "expires_at",
            "row_count",
            "rows",
            "file_count",
            "parquet_file_count",
            "latest_timestamp",
            "max_timestamp",
            "latest_ts",
            "latest_close_timestamp",
            "latest_close_ts",
            "timestamp_column",
            "schema_version",
            "created_at",
        )
        return (
            "snapshot_meta",
            meta_stat.st_mtime_ns if meta_stat is not None else None,
            meta_stat.st_size if meta_stat is not None else None,
            root_stat.st_mtime_ns if root_stat is not None else None,
            root_stat.st_size if root_stat is not None else None,
            tuple((field, str(meta.get(field) or "")) for field in meta_fields),
        )
    dataset_name = _dataset_name_from_path(path)
    if dataset_name in WEB_HEAVY_METADATA_DATASETS:
        indexed_signature = _web_file_index_dataset_signature(path)
        if indexed_signature is not None:
            return indexed_signature
    indexed = _indexed_files_for_web(path)
    if indexed is not None:
        newest = 0
        size_sum = 0
        for file_path in indexed:
            try:
                stat = file_path.stat()
            except OSError:
                continue
            newest = max(newest, stat.st_mtime_ns)
            size_sum += stat.st_size
        return ("lake_file_index", len(indexed), newest, size_sum)
    if not path.exists():
        return ("missing", str(path))
    try:
        stat = path.stat()
    except OSError:
        return ("unreadable", str(path))
    return ("path", stat.st_mtime_ns, stat.st_size)


def _combine_warnings(*warnings: str | None) -> str | None:
    values = [warning for warning in warnings if warning]
    if not values:
        return None
    deduped = list(dict.fromkeys(values))
    return "; ".join(deduped)


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_internal_lake_path(path: Path) -> bool:
    return (
        any(part == "._tmp" or part.startswith("__") for part in path.parts)
        or path.name.startswith(".")
        or path.name.endswith(".tmp.parquet")
    )


def _is_valid_parquet_file_path(path: Path) -> bool:
    try:
        if path.stat().st_size < MIN_PARQUET_SIZE_BYTES:
            return False
        with path.open("rb") as file:
            header = file.read(len(PARQUET_MAGIC))
            file.seek(-len(PARQUET_MAGIC), 2)
            footer = file.read(len(PARQUET_MAGIC))
        return header == PARQUET_MAGIC and footer == PARQUET_MAGIC
    except OSError:
        return False


def _invalid_parquet_warning(dataset_label: str, invalid_files: list[Path]) -> str | None:
    if not invalid_files:
        return None
    rendered = ", ".join(str(file_path) for file_path in invalid_files[:5])
    suffix = "" if len(invalid_files) <= 5 else f"（另有 {len(invalid_files) - 5} 个）"
    return f"{dataset_label} 已忽略无效 Parquet 文件：{rendered}{suffix}"


def dataset_states(lake_root: str | Path) -> list[DatasetState]:
    root = Path(lake_root)
    cache_key = (
        "dataset_states",
        str(root.resolve()),
        _lake_file_index_signature(root),
    )
    cached = _web_cache_get(cache_key, event="dataset_states")
    if cached is not None:
        return cached

    states: list[DatasetState] = []
    for name in sorted(DATASET_PATHS):
        path = dataset_path_for(root, name)
        snapshot = _dataset_snapshot(lake_root, name)
        states.append(
            DatasetState(
                name=name,
                path=path,
                rows=snapshot.rows,
                exists=snapshot.exists,
                parquet_file_count=snapshot.parquet_file_count,
                warning=snapshot.warning,
            )
        )
    return _web_cache_set(cache_key, states)


def _lake_file_index_signature(lake_root: Path) -> tuple[Any, ...]:
    index_path = lake_root / "bronze" / "lake_file_index"
    meta = _snapshot_meta_for_dataset(index_path)
    if meta:
        return (
            "lake_file_index_meta",
            meta.get("source_sha") or meta.get("generated_at") or meta.get("row_count"),
        )
    try:
        newest = max(
            (file_path.stat().st_mtime_ns for file_path in index_path.rglob("*.parquet")),
            default=0,
        )
    except OSError:
        newest = 0
    return ("lake_file_index", newest)


def dashboard_overview(lake_root: str | Path) -> dict[str, Any]:
    diagnostics = lake_diagnostics(lake_root)
    market_health = _overview_market_health(lake_root)
    costs = cost_model_summary(lake_root)
    gates = alpha_gate_summary(lake_root)
    consumers = strategy_consumer_summary(lake_root)
    ws_status, ws_warnings = _overview_okx_ws_status(lake_root)
    experts = expert_export_summary(default_exports_root(lake_root))
    advisory, advisory_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "strategy_opportunity_advisory",
    )
    entry_quality, entry_quality_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "v5_entry_quality_advisory",
    )

    warnings = [
        *diagnostics["warnings"],
        *market_health["warnings"],
        *costs["warnings"],
        *gates["warnings"],
        *consumers["warnings"],
        *ws_warnings,
        *experts["warnings"],
        *[warning for warning in [advisory_warning, entry_quality_warning] if warning],
    ]
    status = "OK"
    if warnings:
        status = "WARNING"
    if market_health["schema_violation_count"] > 0 or market_health["unclosed_bar_count"] > 0:
        status = "CRITICAL"

    overview = {
        "status": status,
        "v5_permission": consumers["permissions"].get("v5", "UNKNOWN"),
        "okx_public_rest_status": "OK" if market_health["latest_market_bar_ts"] else "WARNING",
        "okx_public_ws_status": ws_status,
        "okx_readonly_status": readonly_private_status(lake_root),
        "latest_market_bar_ts": market_health["latest_market_bar_ts"],
        "missing_bar_ratio": market_health["missing_bar_ratio"],
        "cost_fallback_ratio": costs["fallback_ratio"],
        "cost_hard_fallback_ratio": costs["hard_fallback_ratio"],
        "cost_soft_fallback_ratio": costs["soft_fallback_ratio"],
        "alpha_gate_counts": gates["counts"],
        "latest_expert_pack": experts["latest_pack"],
        "strategy_opportunity_advisory": redact_frame(_strategy_opportunity_table(advisory)).head(
            30
        ),
        "entry_quality_advisory": redact_frame(_entry_quality_table(entry_quality)).head(30),
        "diagnostics": diagnostics,
        "warnings": warnings,
    }
    if "v7" in consumers["permissions"]:
        overview["v7_permission"] = consumers["permissions"]["v7"]
    return overview


def _select_existing_columns(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    selected = [column for column in columns if column in frame.columns]
    return frame.select(selected) if selected else frame


def _sort_by_priority(
    frame: pl.DataFrame,
    column: str,
    priority: dict[str, int],
    secondary: list[str] | None = None,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    sort_columns = [column for column in (secondary or []) if column in frame.columns]
    if column not in frame.columns:
        return frame.sort(sort_columns) if sort_columns else frame
    priority_column = "__web_display_priority"
    fallback_priority = len(priority) + 100
    table = frame.with_columns(
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .map_elements(
            lambda value: priority.get(str(value), fallback_priority),
            return_dtype=pl.Int64,
        )
        .alias(priority_column)
    )
    return table.sort([priority_column, *sort_columns]).drop(priority_column)


def _dedupe_latest_web_rows(
    frame: pl.DataFrame,
    keys: list[str],
    timestamp_columns: list[str],
) -> pl.DataFrame:
    if frame.is_empty() or not all(key in frame.columns for key in keys):
        return frame
    sort_columns = [column for column in timestamp_columns if column in frame.columns]
    if sort_columns:
        frame = frame.sort(sort_columns)
    return frame.unique(subset=keys, keep="last", maintain_order=True)


def _strategy_opportunity_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "decision",
        "recommended_mode",
        "strategy_candidate",
        "symbol",
        "horizon_hours",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "complete_sample_count",
        "source_module",
        "promotion_state",
        "universe_type",
        "alpha_factory_score",
        "cost_quality_score",
        "paper_ready_block_reasons",
        "live_block_reasons",
        "max_live_notional_usdt",
        "as_of_ts",
    ]
    table = _select_existing_columns(frame, columns)
    table = _dedupe_latest_web_rows(
        table,
        keys=["strategy_candidate", "symbol", "horizon_hours"],
        timestamp_columns=["as_of_ts", "created_at"],
    )
    table = _with_strategy_opportunity_focus(table)
    table = _sort_by_priority(
        table,
        "recommended_mode",
        RECOMMENDED_MODE_PRIORITY,
        ["decision", "strategy_candidate", "symbol", "horizon_hours"],
    )
    table = _sort_by_priority(
        table,
        "decision",
        DECISION_PRIORITY,
        ["recommended_mode", "strategy_candidate", "symbol", "horizon_hours"],
    )
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "recommended_mode",
            "strategy_candidate",
            "symbol",
            "horizon_hours",
            "key_metrics",
            "alpha_factory_score",
            "cost_quality_score",
            "paper_ready_block_reasons",
            "live_block_reasons",
            "source_module",
            "promotion_state",
            "universe_type",
            "as_of_ts",
        ],
    )


def _with_strategy_opportunity_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "decision",
            "recommended_mode",
            "strategy_candidate",
            "symbol",
            "horizon_hours",
            "complete_sample_count",
            "avg_net_bps",
            "p25_net_bps",
            "win_rate",
            "alpha_factory_score",
            "max_live_notional_usdt",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_strategy_opportunity_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_strategy_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _with_alpha_factory_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "decision",
            "promotion_state",
            "recommended_mode",
            "strategy_candidate",
            "symbol",
            "horizon_hours",
            "complete_sample_count",
            "avg_net_bps",
            "p25_net_bps",
            "win_rate",
            "alpha_factory_score",
            "cost_quality_score",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_alpha_factory_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_strategy_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _with_research_result_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "decision",
            "strategy_candidate",
            "candidate_name",
            "symbol",
            "horizon_hours",
            "sample_count",
            "complete_sample_count",
            "avg_net_bps",
            "p25_net_bps",
            "win_rate",
            "avg_mae_bps",
            "avg_mfe_bps",
            "paper_days",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_research_result_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_strategy_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _strategy_opportunity_takeaway(row: dict[str, Any]) -> str:
    decision = str(row.get("decision") or "").upper()
    mode = str(row.get("recommended_mode") or "").lower()
    candidate = _friendly_label(row.get("strategy_candidate"))
    live_notional = _as_float(row.get("max_live_notional_usdt")) or 0.0
    if decision == "KILL" or mode == "none":
        return f"不推荐：{candidate} 已淘汰或当前无正边际"
    if mode == "paper":
        suffix = "，实盘名义仍为 0" if live_notional <= 0 else ""
        return f"纸面观察：{candidate} 可继续做 paper{suffix}"
    if mode == "shadow":
        return f"影子观察：{candidate} 继续收集样本，不放实盘"
    if mode == "research":
        return f"研究观察：{candidate} 证据不足，暂不进 paper"
    if mode == "audit":
        return f"审计：{candidate} 仅用于诊断"
    return f"待判断：{candidate}"


def _alpha_factory_takeaway(row: dict[str, Any]) -> str:
    decision = str(row.get("decision") or row.get("promotion_state") or "").upper()
    candidate = _friendly_label(row.get("strategy_candidate"))
    if decision == "KILL":
        return f"淘汰：{candidate} 未通过 Alpha Factory"
    if decision == "PAPER_READY":
        return f"纸面候选：{candidate} 通过基础筛选，但仍只允许 paper"
    if decision == "KEEP_SHADOW":
        return f"继续影子：{candidate} 有苗头但未达 paper 门槛"
    return f"研究中：{candidate} 继续等待样本和验证"


def _research_result_takeaway(row: dict[str, Any]) -> str:
    decision = str(row.get("decision") or "").upper()
    candidate = _friendly_label(row.get("strategy_candidate") or row.get("candidate_name"))
    if decision == "KILL":
        return f"淘汰：{candidate} 负收益或尾部风险不合格"
    if decision == "PAPER_READY":
        return f"纸面就绪：{candidate} 可观察 paper，不能直接 live"
    if decision == "KEEP_SHADOW":
        return f"继续影子：{candidate} 有正边际但仍需验证"
    if decision == "REGIME_SHADOW":
        return f"分行情观察：{candidate} 只在特定 regime 下研究"
    return f"仅研究：{candidate} 样本或成本质量不足"


def _strategy_key_metrics(row: dict[str, Any]) -> str:
    parts: list[str] = []
    complete = _as_int(row.get("complete_sample_count"))
    sample = _as_int(row.get("sample_count"))
    if complete is not None:
        parts.append(f"完整样本 {complete}")
    elif sample is not None:
        parts.append(f"样本 {sample}")
    avg = _as_float(row.get("avg_net_bps"))
    if avg is not None:
        parts.append(f"平均净收益 {_format_bps(avg)}")
    p25 = _as_float(row.get("p25_net_bps"))
    if p25 is not None:
        parts.append(f"P25 {_format_bps(p25)}")
    win_rate = _as_float(row.get("win_rate"))
    if win_rate is not None:
        parts.append(f"胜率 {_format_pct(win_rate)}")
    score = _as_float(row.get("alpha_factory_score"))
    if score is not None:
        parts.append(f"工厂分 {score:.2f}")
    cost_score = _as_float(row.get("cost_quality_score"))
    if cost_score is not None:
        parts.append(f"成本分 {cost_score:.2f}")
    paper_days = _as_int(row.get("paper_days"))
    if paper_days is not None:
        parts.append(f"paper {paper_days}天")
    negative_streak = _as_int(row.get("paper_negative_streak"))
    if negative_streak:
        parts.append(f"连续转弱 {negative_streak}天")
    return "；".join(parts) if parts else "暂无关键指标"


def _entry_quality_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "strategy_candidate",
        "symbol",
        "recommended_mode",
        "readiness_status",
        "sample_count",
        "avg_net_bps",
        "win_rate",
        "advisory_reasons",
        "ready_for_live",
        "generated_at_utc",
    ]
    selected = [column for column in columns if column in frame.columns]
    table = frame.select(selected) if selected else frame
    return _sort_by_priority(
        table,
        "recommended_mode",
        RECOMMENDED_MODE_PRIORITY,
        ["strategy_candidate", "symbol"],
    )


def _overview_market_health(lake_root: str | Path) -> dict[str, Any]:
    warnings: list[str] = []
    market_health = _market_bar_lazy_health(lake_root)
    if market_health["warning"]:
        warnings.append(market_health["warning"])
    if market_health["row_count"] == 0:
        return {
            "latest_market_bar_ts": None,
            "latest_market_bar_close_ts": None,
            "market_bar_timeframe": DEFAULT_MARKET_BAR_TIMEFRAME,
            "missing_bar_ratio": 0.0,
            "schema_violation_count": 1,
            "unclosed_bar_count": 0,
            "warnings": [*warnings, "market_bar 数据集缺失或为空"],
        }
    schema_violations = market_health["schema_violations"]
    unclosed_count = market_health["unclosed_bar_count"]
    missing_bars = market_health["missing_bars"]
    return {
        "latest_market_bar_ts": market_health["latest_market_bar_ts"],
        "latest_market_bar_close_ts": market_health.get("latest_market_bar_close_ts"),
        "market_bar_timeframe": market_health.get("market_bar_timeframe"),
        "missing_bar_ratio": _missing_ratio(missing_bars, market_health["row_count"]),
        "schema_violation_count": len(schema_violations),
        "unclosed_bar_count": unclosed_count,
        "warnings": [*warnings, *schema_violations],
    }


def _overview_okx_ws_status(lake_root: str | Path) -> tuple[str, list[str]]:
    health, health_warning = read_dataset_with_warning(lake_root, "okx_public_ws_health")
    warnings = [health_warning] if health_warning else []
    latest_health = _latest_collector_health(health)
    if not latest_health:
        return "WARNING", [*warnings, "OKX 公共 WebSocket 采集器健康数据缺失或为空"]
    return str(latest_health.get("status") or "UNKNOWN"), warnings


def lake_diagnostics(lake_root: str | Path) -> dict[str, Any]:
    root = Path(lake_root)
    parquet_count = _parquet_file_count(root)
    latest_market_bar_ts: datetime | None = None
    warnings: list[str] = []
    suggested_commands: list[dict[str, str]] = []
    dataset_rows: list[dict[str, Any]] = []
    dataset_row_counts: dict[str, int] = {}

    if not root.exists():
        warnings.append(f"Lake 根目录不存在：{root}")
    if parquet_count == 0:
        warnings.append(f"Lake 根目录下没有 Parquet 文件：{root}")

    for dataset_name, relative_path in CORE_DIAGNOSTIC_DATASETS.items():
        path = root / relative_path
        warning = None
        canonical_name = _canonical_dataset_name(dataset_name)
        snapshot = _dataset_snapshot(
            root,
            dataset_name,
            timestamp_columns=DATASET_TIMESTAMP_COLUMNS.get(canonical_name),
        )
        row_count = snapshot.rows
        warning = snapshot.warning
        freshness = snapshot.freshness
        if warning:
            warnings.append(warning)
        if dataset_name == "silver/market_bar":
            latest_market_bar_ts = _coerce_timestamp(freshness.get("latest_timestamp"))

        exists = path.exists()
        dataset_row_counts[dataset_name] = row_count
        if not exists or row_count == 0:
            warnings.append(f"{dataset_name} 数据集缺失或为空：{path}")
        elif _dataset_has_bootstrap_placeholder(path, canonical_name):
            warning = BOOTSTRAP_PLACEHOLDER_WARNING
            warnings.append(f"{dataset_name}: {warning}")

        dataset_rows.append(
            {
                "dataset": dataset_name,
                "exists": exists,
                "parquet_file_count": snapshot.parquet_file_count,
                "rows": row_count,
                "path": str(path),
                "freshness_seconds": freshness["freshness_seconds"],
                "freshness_status": freshness["freshness_status"],
                "timestamp_column": freshness["timestamp_column"] or "",
                "latest_timestamp": freshness["latest_timestamp"] or "",
                "warning": warning or "",
            }
        )

    if dataset_row_counts.get("silver/market_bar", 0) == 0:
        command = MARKET_BOOTSTRAP_COMMAND.format(lake_root=root)
        warnings.append(
            f"market_bar 为空。建议命令：{MARKET_BOOTSTRAP_COMMAND.format(lake_root=root)}"
        )

        suggested_commands.append({"purpose": "补齐 market_bar", "command": command})

    gold_health_missing = any(
        dataset_row_counts.get(dataset_name, 0) == 0
        for dataset_name in [
            "gold/cost_bucket_daily",
            "gold/gate_decision",
            "gold/risk_permission",
        ]
    )
    if gold_health_missing:
        command = BOOTSTRAP_GOLD_HEALTH_COMMAND.format(lake_root=root)
        warnings.append(f"建议初始化 gold 健康数据：{command}")
        suggested_commands.append({"purpose": "初始化 gold 健康数据", "command": command})

    ws_health, ws_health_warning = _read_parquet_dataset_with_warning(
        root / DATASET_PATHS["okx_public_ws_health"],
        "okx_public_ws_health",
    )
    if ws_health_warning:
        warnings.append(ws_health_warning)
    if ws_health.is_empty():
        command = OKX_WS_COLLECT_COMMAND.format(lake_root=root)
        warnings.append(f"OKX 公共 WebSocket 采集器健康为空。建议命令：{command}")
        suggested_commands.append({"purpose": "启动 OKX 公共 WebSocket", "command": command})

    if dataset_row_counts.get("gold/strategy_health_daily", 0) == 0:
        warnings.append(f"V5 遥测为空。建议命令：{V5_TELEMETRY_SYNC_COMMAND}")

        suggested_commands.append({"purpose": "同步 V5 遥测", "command": V5_TELEMETRY_SYNC_COMMAND})

    experts = expert_export_summary(default_exports_root(root))
    if not experts["latest_pack"]:
        warnings.append(EXPERT_PACK_MISSING_MESSAGE)
        suggested_commands.append(
            {
                "purpose": "生成专家包",
                "command": (
                    "qlab export-daily --date YYYY-MM-DD "
                    f"--lake-root {root} --out-dir {default_exports_root(root)}"
                ),
            }
        )

    return {
        "lake_root": str(root),
        "lake_root_exists": root.exists(),
        "parquet_file_count": parquet_count,
        "latest_market_bar_ts": latest_market_bar_ts,
        "latest_v5_bundle_ts": _latest_v5_bundle_ts(root),
        "latest_expert_pack": experts["latest_pack"],
        "datasets": pl.DataFrame(dataset_rows),
        "suggested_commands": pl.DataFrame(suggested_commands),
        "warnings": _dedupe_strings(warnings),
    }


def data_health_summary(lake_root: str | Path) -> dict[str, Any]:
    warnings: list[str] = []
    market_health = _market_bar_lazy_health(lake_root)
    if market_health["warning"]:
        warnings.append(market_health["warning"])
    if market_health["row_count"] == 0:
        return {
            "latest_per_symbol": pl.DataFrame(),
            "missing_bars": pl.DataFrame(),
            "duplicate_bar_count": 0,
            "unclosed_bar_count": 0,
            "schema_violations": ["market_bar 数据集缺失或为空"],
            "schema_violation_count": 1,
            "stale_datasets": _stale_dataset_rows(lake_root),
            "latest_market_bar_ts": None,
            "latest_market_bar_close_ts": None,
            "market_bar_timeframe": DEFAULT_MARKET_BAR_TIMEFRAME,
            "missing_bar_ratio": 0.0,
            "warnings": [*warnings, "market_bar 数据集缺失或为空"],
        }

    schema_violations = market_health["schema_violations"]
    duplicate_count = market_health["duplicate_bar_count"]
    unclosed_count = market_health["unclosed_bar_count"]
    latest_per_symbol = market_health["latest_per_symbol"]
    missing_bars = market_health["missing_bars"]
    missing_ratio = _missing_ratio(missing_bars, market_health["row_count"])
    latest_ts = market_health["latest_market_bar_ts"]
    latest_close_ts = market_health.get("latest_market_bar_close_ts")
    timeframe = market_health.get("market_bar_timeframe")

    if duplicate_count:
        warnings.append(f"market_bar 主键重复：{duplicate_count}")
    if unclosed_count:
        warnings.append(f"未闭合 market_bar：{unclosed_count}")
    warnings.extend(schema_violations)

    return {
        "latest_per_symbol": latest_per_symbol,
        "missing_bars": missing_bars,
        "duplicate_bar_count": duplicate_count,
        "unclosed_bar_count": unclosed_count,
        "schema_violations": schema_violations,
        "schema_violation_count": len(schema_violations),
        "stale_datasets": _stale_dataset_rows(lake_root),
        "latest_market_bar_ts": latest_ts,
        "latest_market_bar_close_ts": latest_close_ts,
        "market_bar_timeframe": timeframe,
        "missing_bar_ratio": missing_ratio,
        "warnings": warnings,
    }


def _market_bar_lazy_health(lake_root: str | Path) -> dict[str, Any]:
    path = dataset_path_for(lake_root, "market_bar")
    cache_key = (
        "market_bar_lazy_health",
        str(Path(lake_root).resolve()),
        _web_dataset_source_signature(path),
    )
    cached = _web_cache_get(
        cache_key,
        event="market_bar_lazy_health",
        dataset_name="market_bar",
    )
    if cached is not None:
        return cached

    snapshot = _dataset_snapshot(lake_root, "market_bar")
    files = _valid_parquet_files(path, invalid_files=invalid_parquet_files(path))
    if not files:
        return _web_cache_set(
            cache_key,
            {
                "row_count": 0,
                "warning": snapshot.warning,
                "latest_per_symbol": pl.DataFrame(),
                "missing_bars": pl.DataFrame(),
                "duplicate_bar_count": 0,
                "unclosed_bar_count": 0,
                "schema_violations": ["market_bar 数据集缺失或为空"],
                "latest_market_bar_ts": None,
                "latest_market_bar_close_ts": None,
                "market_bar_timeframe": DEFAULT_MARKET_BAR_TIMEFRAME,
            },
        )
    try:
        lazy = _scan_parquet_files(files)
        schema = lazy.collect_schema()
    except Exception as exc:
        return _web_cache_set(
            cache_key,
            {
                "row_count": 0,
                "warning": f"market_bar 元数据读取失败：{exc}",
                "latest_per_symbol": pl.DataFrame(),
                "missing_bars": pl.DataFrame(),
                "duplicate_bar_count": 0,
                "unclosed_bar_count": 0,
                "schema_violations": ["market_bar 数据集读取失败"],
                "latest_market_bar_ts": None,
                "latest_market_bar_close_ts": None,
                "market_bar_timeframe": DEFAULT_MARKET_BAR_TIMEFRAME,
            },
        )

    latest_ts = _coerce_timestamp(snapshot.freshness.get("latest_timestamp"))
    latest_close_ts = _coerce_timestamp(snapshot.freshness.get("latest_close_timestamp"))
    timeframe = str(snapshot.freshness.get("timeframe") or DEFAULT_MARKET_BAR_TIMEFRAME)
    if latest_close_ts is None:
        latest_close_ts = market_bar_close_ts(latest_ts, timeframe)
    return _web_cache_set(
        cache_key,
        {
            "row_count": snapshot.rows,
            "warning": snapshot.warning,
            "latest_per_symbol": _latest_market_bars_lazy(lazy, schema),
            "missing_bars": _missing_bar_table_lazy(lazy, schema),
            "duplicate_bar_count": _duplicate_market_bar_count_lazy(lazy, schema),
            "unclosed_bar_count": _unclosed_market_bar_count_lazy(lazy, schema),
            "schema_violations": _market_bar_schema_violations_lazy(lazy, schema),
            "latest_market_bar_ts": latest_ts,
            "latest_market_bar_close_ts": latest_close_ts,
            "market_bar_timeframe": timeframe,
        },
    )


def okx_collector_summary(lake_root: str | Path) -> dict[str, Any]:
    market_snapshot = _dataset_snapshot(
        lake_root,
        "market_bar",
        timestamp_columns=("ingest_ts",),
    )
    ws_raw_snapshot = _dataset_snapshot(lake_root, "okx_public_ws")
    trades_snapshot = _dataset_snapshot(lake_root, "trade_print")
    books_snapshot = _dataset_snapshot(lake_root, "orderbook_snapshot")
    ws_health, ws_health_warning = read_dataset_with_warning(lake_root, "okx_public_ws_health")

    latest_rest = _snapshot_latest_timestamp(market_snapshot)
    latest_ws = _snapshot_latest_timestamp(ws_raw_snapshot)
    latest_health = _latest_collector_health(ws_health)
    if latest_health:
        latest_ws = _parse_datetime(latest_health.get("last_message_at")) or latest_ws

    rows = [
        {
            "collector": "OKX 公共 REST",
            "success_count": market_snapshot.rows,
            "error_count": 0,
            "latest_success_ts": latest_rest,
            "reconnect_count": None,
            "rate_limit_warnings": 0,
            "lag": _lag_label(latest_rest),
        },
        {
            "collector": "OKX 公共 WebSocket",
            "success_count": int(latest_health.get("messages_read") or ws_raw_snapshot.rows)
            if latest_health
            else ws_raw_snapshot.rows,
            "error_count": int(latest_health.get("error_count") or 0) if latest_health else 0,
            "latest_success_ts": latest_ws,
            "reconnect_count": int(latest_health.get("reconnect_count") or 0)
            if latest_health
            else 0,
            "rate_limit_warnings": 0,
            "lag": _lag_label(latest_ws),
            "status": str(latest_health.get("status") or "UNKNOWN") if latest_health else "UNKNOWN",
        },
    ]
    warnings = [
        warning
        for warning in [
            market_snapshot.warning,
            ws_raw_snapshot.warning,
            ws_health_warning,
            trades_snapshot.warning,
            books_snapshot.warning,
        ]
        if warning
    ]
    if market_snapshot.rows == 0:
        warnings.append("OKX 公共 REST market_bar 数据集缺失或为空")
    if ws_raw_snapshot.rows == 0:
        warnings.append("OKX 公共 WebSocket bronze 数据集缺失或为空")

    if ws_health.is_empty():
        warnings.append("OKX 公共 WebSocket 采集器健康数据缺失或为空")

    return {
        "collectors": pl.DataFrame(rows),
        "collector_health": redact_frame(ws_health).head(DISPLAY_LIMIT),
        "okx_public_ws_status": str(latest_health.get("status") or "UNKNOWN")
        if latest_health
        else "WARNING",
        "trade_print_rows": trades_snapshot.rows,
        "orderbook_snapshot_rows": books_snapshot.rows,
        "warnings": warnings,
    }


def market_regime_summary(lake_root: str | Path) -> dict[str, Any]:
    market, market_warning = read_recent_dataset_with_warning(lake_root, "market_bar")
    spread_bps, books_warning = _latest_orderbook_spread_for_market_status(lake_root)
    trade_activity, trades_warning = _recent_trade_activity_for_market_status(lake_root)
    ws_health, ws_health_warning = read_dataset_with_warning(lake_root, "okx_public_ws_health")
    warnings = [
        warning
        for warning in [market_warning, books_warning, trades_warning, ws_health_warning]
        if warning
    ]
    if market.is_empty():
        return {
            "regimes": pl.DataFrame(),
            "spread_bps": spread_bps,
            "trade_activity": trade_activity,
            "abnormal_symbols": pl.DataFrame(),
            "warnings": [*warnings, "market_bar 数据集缺失或为空"],
        }

    market = _normalize_market_frame(market)
    regimes = (
        market.sort(["symbol", "timeframe", "ts"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(1).over(["symbol", "timeframe"])) - 1.0)
            .abs()
            .alias("abs_return")
        )
        .group_by(["symbol", "timeframe"])
        .agg(
            pl.col("abs_return").mean().fill_null(0.0).alias("mean_abs_return"),
            pl.col("volume").mean().fill_null(0.0).alias("avg_volume"),
            pl.col("ts").max().alias("latest_ts"),
        )
        .with_columns(
            pl.when(pl.col("mean_abs_return") > 0.03)
            .then(pl.lit("高波动"))
            .when(pl.col("mean_abs_return") > 0.01)
            .then(pl.lit("正常"))
            .otherwise(pl.lit("低波动"))
            .alias("volatility_regime")
        )
    )
    global_latest_ts = regimes.select(pl.col("latest_ts").max()).item()
    stale_regime_count = 0
    if isinstance(global_latest_ts, datetime):
        current_cutoff = global_latest_ts - timedelta(hours=MARKET_REGIME_CURRENT_LAG_HOURS)
        current_regimes = regimes.filter(pl.col("latest_ts") >= current_cutoff)
        stale_regime_count = regimes.height - current_regimes.height
        if not current_regimes.is_empty():
            regimes = current_regimes
    regimes = regimes.sort(["latest_ts", "symbol", "timeframe"], descending=[True, False, False])

    warnings.extend(
        _market_universe_warnings(
            market,
            spread_bps,
            trade_activity,
            ws_subscription=_latest_ws_subscription(ws_health),
        )
    )

    return {
        "regimes": redact_frame(regimes),
        "spread_bps": spread_bps,
        "trade_activity": trade_activity,
        "abnormal_symbols": regimes.filter(pl.col("mean_abs_return") > 0.03),
        "hidden_stale_regime_rows": stale_regime_count,
        "market_regime_latest_ts": global_latest_ts,
        "warnings": warnings,
    }


def _latest_orderbook_spread_for_market_status(
    lake_root: str | Path,
) -> tuple[pl.DataFrame, str | None]:
    dataset_name = "orderbook_snapshot"
    dataset_path = dataset_path_for(lake_root, dataset_name)
    files, warning = _recent_valid_parquet_files(dataset_path, dataset_name)
    if not files:
        return pl.DataFrame(), warning
    try:
        lazy = _scan_parquet_files(files)
        schema = lazy.collect_schema()
        required = {"symbol", "asks_json", "bids_json"}
        if not required.issubset(schema.names()):
            return pl.DataFrame(), warning
        timestamp_column = _web_status_timestamp_column(schema)
        selected_columns = ["symbol", "asks_json", "bids_json"]
        if "channel" in schema:
            selected_columns.append("channel")
        if timestamp_column is None:
            frame = lazy.select(selected_columns).tail(DISPLAY_LIMIT).collect()
            return orderbook_spread_table(frame), warning

        ts_expr = _lazy_timestamp_expr(schema, timestamp_column).alias("__qlab_ts")
        latest, _column = _latest_lazy_timestamp(
            dataset_name,
            lazy,
            schema,
            timestamp_columns=(timestamp_column,),
        )
        filtered = lazy.with_columns(ts_expr)
        if latest is not None:
            filtered = filtered.filter(
                pl.col("__qlab_ts")
                >= latest - timedelta(hours=WEB_RECENT_LOOKBACK_HOURS.get(dataset_name, 6))
            )
        group_keys = ["symbol", "channel"] if "channel" in schema else ["symbol"]
        frame = (
            filtered.select([*selected_columns, "__qlab_ts"])
            .sort([*group_keys, "__qlab_ts"])
            .group_by(group_keys, maintain_order=True)
            .tail(1)
            .rename({"__qlab_ts": "ts"})
            .collect()
        )
        return orderbook_spread_table(frame), warning
    except Exception as exc:
        return pl.DataFrame(), f"{dataset_name} 市场状态聚合失败：{exc}"


def _recent_trade_activity_for_market_status(
    lake_root: str | Path,
) -> tuple[pl.DataFrame, str | None]:
    dataset_name = "trade_print"
    dataset_path = dataset_path_for(lake_root, dataset_name)
    files, warning = _recent_valid_parquet_files(dataset_path, dataset_name)
    if not files:
        return pl.DataFrame(), warning
    try:
        lazy = _scan_parquet_files(files)
        schema = lazy.collect_schema()
        if "symbol" not in schema.names():
            return pl.DataFrame(), warning
        timestamp_column = _web_status_timestamp_column(schema)
        filtered = lazy
        aggregations = [pl.len().alias("trade_count")]
        if "size" in schema:
            aggregations.append(
                pl.col("size").cast(pl.Float64, strict=False).sum().alias("size_sum")
            )
        if timestamp_column is not None:
            filtered = filtered.with_columns(
                _lazy_timestamp_expr(schema, timestamp_column).alias("__qlab_ts")
            )
            latest, _column = _latest_lazy_timestamp(
                dataset_name,
                lazy,
                schema,
                timestamp_columns=(timestamp_column,),
            )
            if latest is not None:
                filtered = filtered.filter(
                    pl.col("__qlab_ts")
                    >= latest - timedelta(hours=WEB_RECENT_LOOKBACK_HOURS.get(dataset_name, 6))
                )
            aggregations.append(pl.col("__qlab_ts").max().alias("latest_trade_ts"))
        return filtered.group_by("symbol").agg(aggregations).sort("symbol").collect(), warning
    except Exception as exc:
        return pl.DataFrame(), f"{dataset_name} 市场状态聚合失败：{exc}"


def _web_status_timestamp_column(schema: pl.Schema) -> str | None:
    if "ts" in schema:
        return "ts"
    if "ingest_ts" in schema:
        return "ingest_ts"
    return None


def _market_universe_warnings(
    market: pl.DataFrame,
    orderbooks: pl.DataFrame,
    trade_activity: pl.DataFrame,
    *,
    ws_subscription: tuple[set[str], set[str]] | None = None,
) -> list[str]:
    configured_ws_symbols = set(OKX_WS_UNIVERSE_SYMBOLS)
    market_symbols = _symbols_from_frame(market) & configured_ws_symbols
    if not market_symbols:
        return []
    if ws_subscription is not None:
        subscribed_symbols, subscribed_channels = ws_subscription
        has_books = bool({"books", "books5"} & subscribed_channels)
        has_trades = "trades" in subscribed_channels
        if market_symbols.issubset(subscribed_symbols) and has_books and has_trades:
            return []
    warnings: list[str] = []
    spread_symbols = _symbols_from_frame(orderbooks)
    trade_symbols = _symbols_from_frame(trade_activity)
    if spread_symbols and (missing_spread := sorted(market_symbols - spread_symbols)):
        warnings.append(
            "OKX WebSocket universe 不完整：订单簿缺少 " + ", ".join(missing_spread[:8])
        )
    if trade_symbols and (missing_trades := sorted(market_symbols - trade_symbols)):
        warnings.append("OKX WebSocket universe 不完整：成交缺少 " + ", ".join(missing_trades[:8]))
    return warnings


def _symbols_from_frame(frame: pl.DataFrame) -> set[str]:
    if frame.is_empty() or "symbol" not in frame.columns:
        return set()
    return {str(symbol) for symbol in frame["symbol"].drop_nulls().unique().to_list()}


def cost_model_summary(lake_root: str | Path) -> dict[str, Any]:
    costs, costs_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "cost_bucket_daily",
    )
    costs = _normalize_symbol_frame(costs)
    health, health_warning = read_dataset_with_warning(lake_root, "cost_health_daily")
    bootstrap, bootstrap_warning = read_dataset_with_warning(
        lake_root,
        "cost_bootstrap_readiness",
    )
    disagreement, disagreement_warning = read_dataset_with_warning(
        lake_root,
        "cost_probe_cost_disagreement",
    )
    live_coverage, live_coverage_warning = _live_universe_cost_coverage_table(costs)
    warnings = [
        warning
        for warning in [
            costs_warning,
            health_warning,
            bootstrap_warning,
            disagreement_warning,
            live_coverage_warning,
        ]
        if warning
    ]
    warnings.extend(_cost_probe_disagreement_warnings(disagreement))
    if costs.is_empty():
        return {
            "costs": pl.DataFrame(),
            "cost_health": redact_frame(_cost_health_table(health)).head(DISPLAY_LIMIT),
            "cost_bootstrap_readiness": redact_frame(
                _cost_bootstrap_readiness_table(bootstrap)
            ).head(DISPLAY_LIMIT),
            "cost_probe_cost_disagreement": redact_frame(
                _cost_probe_cost_disagreement_table(disagreement)
            ).head(DISPLAY_LIMIT),
            "live_universe_cost_coverage": live_coverage,
            "actual_rows": 0,
            "mixed_rows": 0,
            "bootstrap_probe_rows": 0,
            "proxy_rows": 0,
            "global_default_rows": 0,
            "fallback_ratio": None,
            "hard_fallback_ratio": None,
            "soft_fallback_ratio": None,
            "proxy_only_count": 0,
            "fallback_ratio_status": "N/A",
            "warnings": [*warnings, "cost_bucket_daily 数据集缺失或为空"],
        }

    fallback_ratio = 0.0
    if "fallback_level" in costs.columns and costs.height:
        fallback_rows = costs.filter(
            ~pl.col("fallback_level").is_in(["actual_okx_fills_and_bills", "NONE"])
        ).height
        fallback_ratio = fallback_rows / costs.height

    latest_health = health.sort("day").tail(1).to_dicts()[0] if not health.is_empty() else {}
    effective_quality = _effective_cost_quality_counts(costs, bootstrap, live_coverage)
    fallback_ratio = latest_health.get("fallback_ratio", fallback_ratio)
    return {
        "costs": redact_frame(_cost_bucket_table(costs)).head(DISPLAY_LIMIT),
        "cost_health": redact_frame(_cost_health_table(health)).head(DISPLAY_LIMIT),
        "cost_bootstrap_readiness": redact_frame(_cost_bootstrap_readiness_table(bootstrap)).head(
            DISPLAY_LIMIT
        ),
        "cost_probe_cost_disagreement": redact_frame(
            _cost_probe_cost_disagreement_table(disagreement)
        ).head(DISPLAY_LIMIT),
        "live_universe_cost_coverage": live_coverage,
        "actual_rows": effective_quality["actual_rows"],
        "mixed_rows": effective_quality["mixed_rows"],
        "bootstrap_probe_rows": effective_quality["bootstrap_probe_rows"],
        "proxy_rows": effective_quality["proxy_rows"],
        "global_default_rows": effective_quality["global_default_rows"],
        "cost_quality_basis": effective_quality["basis"],
        "hard_fallback_count": int(latest_health.get("hard_fallback_count") or 0),
        "hard_fallback_ratio": latest_health.get("hard_fallback_ratio"),
        "soft_fallback_count": int(latest_health.get("soft_fallback_count") or 0),
        "soft_fallback_ratio": latest_health.get("soft_fallback_ratio"),
        "proxy_only_count": int(latest_health.get("proxy_only_count") or 0),
        "symbols_with_actual_cost": latest_health.get("symbols_with_actual_cost"),
        "symbols_with_proxy_only": latest_health.get("symbols_with_proxy_only"),
        "fallback_ratio": fallback_ratio,
        "fallback_ratio_status": (
            "OK" if float(latest_health.get("hard_fallback_ratio") or 0.0) <= 0.25 else "FAIL"
        ),
        "warnings": warnings,
    }


def _effective_cost_quality_counts(
    costs: pl.DataFrame,
    bootstrap: pl.DataFrame,
    live_coverage: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Count the best current cost evidence once per symbol for the Web V2 card."""
    counts = {
        "actual_rows": 0,
        "mixed_rows": 0,
        "bootstrap_probe_rows": 0,
        "proxy_rows": 0,
        "global_default_rows": 0,
        "basis": "effective_symbol_source",
    }
    if costs.is_empty() and bootstrap.is_empty():
        return counts

    candidates: dict[str, dict[str, Any]] = {}

    def add_candidate(
        symbol: Any,
        source: Any,
        *,
        time_key: Any = "",
        sample_count: Any = None,
    ) -> None:
        normalized_symbol = normalize_symbol(str(symbol or ""))
        if not normalized_symbol or normalized_symbol == "GLOBAL":
            return
        normalized_source = _cost_source_bucket(source)
        if normalized_source is None:
            return
        priority = _cost_source_priority(normalized_source)
        candidate = {
            "source": normalized_source,
            "priority": priority,
            "time_key": str(time_key or ""),
            "sample_count": _as_int(sample_count) or 0,
        }
        existing = candidates.get(normalized_symbol)
        if existing is None or _cost_quality_candidate_is_better(candidate, existing):
            candidates[normalized_symbol] = candidate

    latest_costs = costs
    if "day" in latest_costs.columns and not latest_costs.is_empty():
        latest_day = latest_costs.select(pl.col("day").cast(pl.String, strict=False).max()).item()
        latest_costs = latest_costs.filter(
            pl.col("day").cast(pl.String, strict=False) == latest_day
        )

    for row in latest_costs.to_dicts():
        add_candidate(
            row.get("symbol"),
            row.get("cost_source") or row.get("source"),
            time_key=row.get("created_at") or row.get("day"),
            sample_count=row.get("sample_count"),
        )

    for row in bootstrap.to_dicts():
        source = _bootstrap_readiness_cost_source(row)
        add_candidate(
            row.get("symbol"),
            source,
            time_key=(
                row.get("latest_fill_ts")
                or row.get("latest_probe_ts")
                or row.get("latest_probe_fill_ts")
                or row.get("generated_at")
            ),
            sample_count=row.get("sample_count"),
        )

    _apply_live_universe_cost_quality_overrides(candidates, live_coverage)

    for row in candidates.values():
        source = row["source"]
        if source == "actual":
            counts["actual_rows"] += 1
        elif source == "mixed":
            counts["mixed_rows"] += 1
        elif source == "bootstrap":
            counts["bootstrap_probe_rows"] += 1
        elif source == "proxy":
            counts["proxy_rows"] += 1
        elif source == "global_default":
            counts["global_default_rows"] += 1
    return counts


def _apply_live_universe_cost_quality_overrides(
    candidates: dict[str, dict[str, Any]],
    live_coverage: pl.DataFrame | None,
) -> None:
    """Use live-universe effective sources to prevent stale bootstrap rows from winning."""
    if live_coverage is None or live_coverage.is_empty():
        return
    if (
        "symbol" not in live_coverage.columns
        or "effective_cost_source" not in live_coverage.columns
    ):
        return
    for row in live_coverage.to_dicts():
        normalized_symbol = normalize_symbol(str(row.get("symbol") or ""))
        if not normalized_symbol or normalized_symbol == "GLOBAL":
            continue
        source = _live_coverage_effective_source(row)
        normalized_source = _cost_source_bucket(source)
        if normalized_source is None:
            continue
        candidates[normalized_symbol] = {
            "source": normalized_source,
            "priority": _cost_source_priority(normalized_source),
            "time_key": row.get("latest_created_at") or row.get("generated_at") or "",
            "sample_count": _as_int(row.get("sample_count")) or 0,
        }


def _live_coverage_effective_source(row: dict[str, Any]) -> Any:
    tier = str(row.get("cost_evidence_tier") or "").strip().lower()
    if "proxy" in tier:
        return "public_spread_proxy"
    if "bootstrap_cost_probe" in tier:
        return "bootstrap_cost_probe"
    return row.get("effective_cost_source")


def _cost_probe_disagreement_warnings(frame: pl.DataFrame) -> list[str]:
    if frame.is_empty() or "status" not in frame.columns:
        return []
    status_text = pl.col("status").cast(pl.Utf8, strict=False).str.to_uppercase()
    fail_count = frame.filter(status_text == "FAIL").height
    warn_count = frame.filter(status_text.is_in(["WARN", "WARNING"])).height
    warnings: list[str] = []
    if fail_count:
        warnings.append(f"cost_probe_cost_disagreement_fail: fail_count={fail_count}")
    if warn_count:
        warnings.append(f"cost_probe_cost_disagreement_warn: warn_count={warn_count}")
    return warnings


def _cost_quality_candidate_is_better(candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
    candidate_priority = _candidate_priority(candidate)
    existing_priority = _candidate_priority(existing)
    if candidate_priority != existing_priority:
        return candidate_priority < existing_priority
    candidate_samples = int(candidate.get("sample_count") or 0)
    existing_samples = int(existing.get("sample_count") or 0)
    if candidate_samples != existing_samples:
        return candidate_samples > existing_samples
    return str(candidate.get("time_key") or "") > str(existing.get("time_key") or "")


def _candidate_priority(candidate: dict[str, Any]) -> int:
    value = candidate.get("priority")
    return 99 if value is None else int(value)


def _cost_source_bucket(value: Any) -> str | None:
    source = str(value or "").strip().lower()
    if not source:
        return None
    if "actual" in source and "mixed" not in source:
        return "actual"
    if "mixed" in source:
        return "mixed"
    if "bootstrap_cost_probe" in source:
        return "bootstrap"
    if "public_spread_proxy" in source or source == "public_proxy":
        return "proxy"
    if "global_default" in source:
        return "global_default"
    return None


def _cost_source_priority(source: str) -> int:
    return {
        "actual": 0,
        "mixed": 1,
        "bootstrap": 2,
        "proxy": 3,
        "global_default": 4,
    }.get(source, 99)


def _bootstrap_readiness_cost_source(row: dict[str, Any]) -> str | None:
    source_text = " ".join(
        str(row.get(column) or "").strip().lower()
        for column in ["latest_cost_source", "cost_evidence_tier", "bootstrap_state"]
    )
    actual_count = (_as_int(row.get("actual_fill_count")) or 0) + (
        _as_int(row.get("strategy_live_fill_count")) or 0
    )
    mixed_count = _as_int(row.get("mixed_fill_count")) or 0
    probe_count = _as_int(row.get("cost_probe_fill_count")) or 0
    if actual_count > 0 or "actual_fills" in source_text:
        return "actual_fills"
    if mixed_count > 0 or "mixed_actual_proxy" in source_text:
        return "mixed_actual_proxy"
    if probe_count > 0 or "bootstrap_cost_probe" in source_text:
        return "bootstrap_cost_probe"
    return None


def _live_universe_cost_coverage_table(costs: pl.DataFrame) -> tuple[pl.DataFrame, str | None]:
    empty = pl.DataFrame(schema={field: pl.Utf8 for field in LIVE_UNIVERSE_COST_COVERAGE_FIELDS})
    if costs.is_empty():
        return empty, None
    try:
        rows = evaluate_live_universe_cost_coverage(costs).get("rows", [])
    except Exception as exc:
        return empty, f"live_universe_cost_coverage 计算失败：{exc}"
    if not rows:
        return empty, None
    return redact_frame(pl.DataFrame(rows, infer_schema_length=None)).select(
        LIVE_UNIVERSE_COST_COVERAGE_FIELDS
    ), None


def _cost_bootstrap_readiness_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema={field: pl.Utf8 for field in COST_BOOTSTRAP_READINESS_FIELDS})
    normalized = frame
    for column in COST_BOOTSTRAP_READINESS_FIELDS:
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None).alias(column))
    return normalized.select(COST_BOOTSTRAP_READINESS_FIELDS)


def _cost_health_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "day",
        "status",
        "actual_rows",
        "mixed_rows",
        "proxy_rows",
        "global_default_rows",
        "hard_fallback_ratio",
        "soft_fallback_ratio",
        "proxy_only_count",
        "api_global_default_count",
        "api_symbol_proxy_hit_count",
        "api_regime_fallback_count",
        "api_degraded_cost_count",
        "symbols_with_actual_cost",
        "symbols_with_mixed_cost",
        "symbols_with_proxy_only",
        "warnings_json",
        "created_at",
    ]
    table = _select_existing_columns(frame, columns)
    table = _with_cost_health_focus(table)
    sort_columns = [column for column in ["day", "created_at"] if column in table.columns]
    if sort_columns:
        table = table.sort(sort_columns, descending=True)
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "severity",
            "next_action",
            "key_metrics",
            "day",
            "status",
            "symbols_with_actual_cost",
            "symbols_with_mixed_cost",
            "symbols_with_proxy_only",
            "created_at",
        ],
    )


def _with_cost_health_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "status",
            "actual_rows",
            "mixed_rows",
            "proxy_rows",
            "global_default_rows",
            "hard_fallback_ratio",
            "soft_fallback_ratio",
            "proxy_only_count",
            "api_global_default_count",
            "api_symbol_proxy_hit_count",
            "api_regime_fallback_count",
            "api_degraded_cost_count",
            "warnings_json",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_cost_health_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_cost_health_severity, return_dtype=pl.Utf8)
        .alias("severity"),
        pl.struct(fields)
        .map_elements(_cost_health_next_action, return_dtype=pl.Utf8)
        .alias("next_action"),
        pl.struct(fields)
        .map_elements(_cost_health_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _cost_health_takeaway(row: dict[str, Any]) -> str:
    hard_ratio = _as_float(row.get("hard_fallback_ratio")) or 0.0
    soft_ratio = _as_float(row.get("soft_fallback_ratio")) or 0.0
    actual = _as_int(row.get("actual_rows")) or 0
    mixed = _as_int(row.get("mixed_rows")) or 0
    proxy = _as_int(row.get("proxy_rows")) or 0
    global_default = _as_int(row.get("global_default_rows")) or 0
    api_global = _as_int(row.get("api_global_default_count")) or 0
    if hard_ratio > 0.25 or global_default or api_global:
        return "硬回退偏高：存在全局默认或成本服务缺口，不能按这些成本做实盘判断"
    if actual + mixed > 0 and proxy > 0:
        return "成本可用但不完整：已有真实/混合成本，仍有标的只靠盘口代理"
    if proxy > 0 and actual + mixed == 0:
        return "仅代理成本：当前主要是 public spread proxy，只适合 paper/shadow 参考"
    if soft_ratio > 0:
        return "软回退较多：样本或滑点还不充分，先看 canary/paper"
    return "成本健康：没有明显硬回退"


def _cost_health_severity(row: dict[str, Any]) -> str:
    hard_ratio = _as_float(row.get("hard_fallback_ratio")) or 0.0
    global_default = _as_int(row.get("global_default_rows")) or 0
    api_global = _as_int(row.get("api_global_default_count")) or 0
    if hard_ratio > 0.25 or global_default or api_global:
        return "CRITICAL"
    soft_ratio = _as_float(row.get("soft_fallback_ratio")) or 0.0
    proxy = _as_int(row.get("proxy_rows")) or 0
    actual = _as_int(row.get("actual_rows")) or 0
    mixed = _as_int(row.get("mixed_rows")) or 0
    if soft_ratio > 0 or (proxy and not (actual + mixed)):
        return "WARNING"
    return "OK"


def _cost_health_next_action(row: dict[str, Any]) -> str:
    hard_ratio = _as_float(row.get("hard_fallback_ratio")) or 0.0
    global_default = _as_int(row.get("global_default_rows")) or 0
    api_global = _as_int(row.get("api_global_default_count")) or 0
    if hard_ratio > 0.25 or global_default or api_global:
        return "优先排查 symbol 命中、成本桶新鲜度和 global_default API 命中"
    actual = _as_int(row.get("actual_rows")) or 0
    mixed = _as_int(row.get("mixed_rows")) or 0
    proxy = _as_int(row.get("proxy_rows")) or 0
    if proxy and not (actual + mixed):
        return "接入 V5 order_lifecycle 或 OKX fills/bills，补真实费用和滑点样本"
    if proxy:
        return "补齐代理标的的真实或混合成本，降低 soft fallback"
    return "保持成本刷新与样本积累"


def _cost_health_key_metrics(row: dict[str, Any]) -> str:
    parts: list[str] = []
    actual = _as_int(row.get("actual_rows"))
    mixed = _as_int(row.get("mixed_rows"))
    proxy = _as_int(row.get("proxy_rows"))
    global_default = _as_int(row.get("global_default_rows"))
    if actual is not None:
        parts.append(f"真实 {actual}")
    if mixed is not None:
        parts.append(f"混合 {mixed}")
    if proxy is not None:
        parts.append(f"代理 {proxy}")
    if global_default is not None:
        parts.append(f"全局默认 {global_default}")
    hard = _as_float(row.get("hard_fallback_ratio"))
    if hard is not None:
        parts.append(f"硬回退 {_format_pct(hard)}")
    soft = _as_float(row.get("soft_fallback_ratio"))
    if soft is not None:
        parts.append(f"软回退 {_format_pct(soft)}")
    api_global = _as_int(row.get("api_global_default_count"))
    if api_global:
        parts.append(f"API 全局默认 {api_global}")
    api_proxy = _as_int(row.get("api_symbol_proxy_hit_count"))
    if api_proxy:
        parts.append(f"API 标的代理命中 {api_proxy}")
    return "；".join(parts) if parts else "暂无关键指标"


def _cost_bucket_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "symbol",
        "regime",
        "notional_bucket",
        "source",
        "cost_source",
        "sample_count",
        "actual_fill_count",
        "mixed_fill_count",
        "proxy_sample_count",
        "cost_probe_fill_count",
        "strategy_live_fill_count",
        "private_fill_count",
        "sample_origin_mix",
        "eligible_for_live_cost_coverage",
        "fee_bps_p75",
        "spread_bps_p75",
        "slippage_bps_p75",
        "total_cost_bps_p50",
        "total_cost_bps_p75",
        "total_cost_bps_p90",
        "one_way_all_in_cost_bps",
        "roundtrip_all_in_cost_bps",
        "cost_trust_level",
        "cost_trusted_for_live_canary",
        "cost_trusted_for_live_scale",
        "fallback_level",
        "fallback_reason",
        "created_at",
        "day",
    ]
    table = _select_existing_columns(frame, columns)
    table = _with_cost_bucket_focus(table)
    table = _latest_cost_bucket_display_rows(table)
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "cost_trust_level",
            "symbol",
            "regime",
            "key_metrics",
            "source",
            "cost_source",
            "sample_count",
            "sample_origin_mix",
            "eligible_for_live_cost_coverage",
            "fallback_level",
            "fallback_reason",
            "day",
            "created_at",
        ],
    )


def _latest_cost_bucket_display_rows(table: pl.DataFrame) -> pl.DataFrame:
    if table.is_empty() or "symbol" not in table.columns:
        return table

    working = table
    required_columns = ["cost_source", "source", "regime", "notional_bucket", "created_at", "day"]
    for column in required_columns:
        if column not in working.columns:
            working = working.with_columns(pl.lit("").alias(column))

    source_text = pl.concat_str(
        [
            pl.col("cost_source").cast(pl.String, strict=False).fill_null(""),
            pl.lit("|"),
            pl.col("source").cast(pl.String, strict=False).fill_null(""),
        ]
    ).str.to_lowercase()
    time_text = pl.coalesce(
        [
            pl.col("created_at").cast(pl.String, strict=False),
            pl.col("day").cast(pl.String, strict=False),
            pl.lit(""),
        ]
    )
    working = working.with_columns(
        pl.col("symbol")
        .cast(pl.String, strict=False)
        .is_in(list(DEFAULT_LIVE_UNIVERSE_SYMBOLS))
        .cast(pl.Int8)
        .alias("_display_live_priority"),
        pl.when(source_text.str.contains("actual"))
        .then(0)
        .when(source_text.str.contains("mixed"))
        .then(1)
        .when(source_text.str.contains("bootstrap_cost_probe"))
        .then(2)
        .when(source_text.str.contains("public_spread_proxy"))
        .then(3)
        .when(source_text.str.contains("global_default"))
        .then(4)
        .otherwise(5)
        .alias("_display_source_priority"),
        time_text.alias("_display_time_key"),
    )
    key_columns = [
        column
        for column in ["symbol", "cost_source", "source", "regime"]
        if column in working.columns
    ]
    return (
        working.sort(
            ["_display_live_priority", "_display_source_priority", "_display_time_key", "symbol"],
            descending=[True, False, True, False],
        )
        .unique(subset=key_columns, keep="first", maintain_order=True)
        .drop(["_display_live_priority", "_display_source_priority", "_display_time_key"])
    )


def _cost_probe_cost_disagreement_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "generated_at",
        "symbol",
        "status",
        "diff_bps",
        "v5_roundtrip_cost_bps",
        "quant_lab_roundtrip_cost_bps",
        "okx_bill_roundtrip_cost_bps",
        "cost_bucket_source",
        "bill_match_status",
        "reason",
        "authorization_id",
        "roundtrip_id",
    ]
    table = _select_existing_columns(frame, columns)
    if "status" in table.columns:
        status_priority = {
            "FAIL": 0,
            "WARN": 1,
            "WARNING": 1,
            "NOT_EVALUATED": 2,
            "PASS": 3,
        }
        table = table.with_columns(
            pl.col("status")
            .cast(pl.Utf8, strict=False)
            .str.to_uppercase()
            .map_elements(
                lambda value: status_priority.get(str(value or ""), 4),
                return_dtype=pl.Int64,
            )
            .alias("_status_priority")
        )
        sort_columns = ["_status_priority"]
        descending = [False]
        if "generated_at" in table.columns:
            sort_columns.append("generated_at")
            descending.append(True)
        if "symbol" in table.columns:
            sort_columns.append("symbol")
            descending.append(False)
        table = table.sort(sort_columns, descending=descending).drop("_status_priority")
    return table


def _with_cost_bucket_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "symbol",
            "source",
            "cost_source",
            "sample_count",
            "actual_fill_count",
            "mixed_fill_count",
            "proxy_sample_count",
            "cost_probe_fill_count",
            "strategy_live_fill_count",
            "private_fill_count",
            "sample_origin_mix",
            "eligible_for_live_cost_coverage",
            "fee_bps_p75",
            "spread_bps_p75",
            "slippage_bps_p75",
            "total_cost_bps_p50",
            "total_cost_bps_p75",
            "total_cost_bps_p90",
            "one_way_all_in_cost_bps",
            "roundtrip_all_in_cost_bps",
            "cost_trust_level",
            "cost_trusted_for_live_canary",
            "cost_trusted_for_live_scale",
            "fallback_level",
            "fallback_reason",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_cost_bucket_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_cost_bucket_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _cost_bucket_takeaway(row: dict[str, Any]) -> str:
    symbol = str(row.get("symbol") or "未知标的")
    source = str(row.get("cost_source") or row.get("source") or "").lower()
    trust = str(row.get("cost_trust_level") or "").upper()
    sample_count = _as_int(row.get("sample_count")) or 0
    fallback = str(row.get("fallback_level") or row.get("fallback_reason") or "")
    if "global_default" in source or "GLOBAL_DEFAULT" in fallback:
        return f"硬回退：{symbol} 使用全局默认成本，不能作为 paper/live 晋级证据"
    if "public_spread_proxy" in source:
        return f"代理成本：{symbol} 只有盘口价差代理，适合观察但不是完整交易成本"
    if "bootstrap_cost_probe" in source:
        return f"探针样本：{symbol} 仅可用于 bootstrap，不能作为 live actual/mixed 覆盖"
    if "mixed" in source:
        if trust == "CANARY":
            return f"混合成本：{symbol} 可作为 canary 级参考，放大前还要补真实滑点"
        return f"混合成本：{symbol} 有真实费用但滑点仍依赖代理"
    if "actual" in source:
        if trust == "SCALE_READY":
            return f"真实成本：{symbol} 样本和滑点质量达到 scale 级"
        if sample_count >= 30:
            return f"真实成本：{symbol} 可作为 canary 级参考"
        return f"真实成本样本少：{symbol} 还需积累到至少 30 笔"
    return f"成本桶：{symbol} 请关注来源、样本数和 fallback"


def _cost_bucket_key_metrics(row: dict[str, Any]) -> str:
    parts: list[str] = []
    sample_count = _as_int(row.get("sample_count"))
    if sample_count is not None:
        parts.append(f"样本 {sample_count}")
    p50 = _as_float(row.get("total_cost_bps_p50"))
    p75 = _as_float(row.get("total_cost_bps_p75"))
    p90 = _as_float(row.get("total_cost_bps_p90"))
    if p75 is not None:
        pieces = []
        if p50 is not None:
            pieces.append(f"P50 {_format_bps(p50)}")
        pieces.append(f"P75 {_format_bps(p75)}")
        if p90 is not None:
            pieces.append(f"P90 {_format_bps(p90)}")
        parts.append("总成本 " + " / ".join(pieces))
    roundtrip = _as_float(row.get("roundtrip_all_in_cost_bps"))
    if roundtrip is not None:
        parts.append(f"往返 all-in {_format_bps(roundtrip)}")
    fee = _as_float(row.get("fee_bps_p75"))
    spread = _as_float(row.get("spread_bps_p75"))
    slip = _as_float(row.get("slippage_bps_p75"))
    components = []
    if fee is not None:
        components.append(f"费 {_format_bps(fee)}")
    if spread is not None:
        components.append(f"价差 {_format_bps(spread)}")
    if slip is not None:
        components.append(f"滑点 {_format_bps(slip)}")
    if components:
        parts.append(" / ".join(components))
    canary = row.get("cost_trusted_for_live_canary")
    scale = row.get("cost_trusted_for_live_scale")
    if scale is True:
        parts.append("scale 可信")
    elif canary is True:
        parts.append("canary 可信")
    return "；".join(parts) if parts else "暂无关键指标"


def _normalize_symbol_frame(frame: pl.DataFrame, column: str = "symbol") -> pl.DataFrame:
    if frame.is_empty() or column not in frame.columns:
        return frame
    return frame.with_columns(
        pl.col(column).map_elements(normalize_symbol, return_dtype=pl.Utf8).alias(column)
    )


def alpha_gate_summary(lake_root: str | Path) -> dict[str, Any]:
    gates, gates_warning = read_dataset_with_warning(lake_root, "gate_decision")
    evidence, evidence_warning = read_dataset_with_warning(lake_root, "alpha_evidence")
    discovery_board, discovery_board_warning = read_dataset_with_warning(
        lake_root,
        "alpha_discovery_board",
    )
    strategy_evidence, strategy_evidence_warning = read_dataset_with_warning(
        lake_root,
        "strategy_evidence",
    )
    strategy_samples, strategy_samples_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "strategy_evidence_sample",
    )
    research_portfolio, research_portfolio_warning = read_dataset_with_warning(
        lake_root,
        "research_portfolio_status",
    )
    candidate_events, candidate_events_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "v5_candidate_event",
    )
    candidate_labels, candidate_labels_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "v5_candidate_label",
    )
    candidate_quality, candidate_quality_warning = read_dataset_with_warning(
        lake_root,
        "v5_candidate_quality_daily",
    )
    candidate_outcomes, candidate_outcomes_warning = read_dataset_with_warning(
        lake_root,
        "v5_candidate_outcome_summary",
    )
    strategy_opportunities, strategy_opportunities_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "strategy_opportunity_advisory",
    )
    trade_opportunities, trade_opportunities_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "trade_opportunity_event",
    )
    trade_labels, trade_labels_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "trade_opportunity_label",
    )
    trade_similarity, trade_similarity_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "trade_level_similarity_outcome",
    )
    trade_judgments, trade_judgments_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "trade_level_judgment",
    )
    trade_bucket_policy, trade_bucket_policy_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "trade_level_bucket_policy",
    )
    trade_opportunity_queue, trade_opportunity_queue_warning = (
        _read_web_display_dataset_with_warning(
            lake_root,
            "trade_level_opportunity_queue",
        )
    )
    false_block_audit, false_block_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "quant_lab_false_block_audit",
    )
    v5_trade_learning, v5_trade_learning_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "v5_trade_learning_sample",
    )
    v5_trade_attribution, v5_trade_attribution_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "v5_trade_outcome_attribution",
    )
    opportunity_cost_events, opportunity_cost_events_warning = (
        _read_web_display_dataset_with_warning(
            lake_root,
            "quant_lab_opportunity_cost_event",
        )
    )
    opportunity_cost_daily, opportunity_cost_daily_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "quant_lab_opportunity_cost_daily",
    )
    opportunity_cost_by_bucket, opportunity_cost_by_bucket_warning = (
        _read_web_display_dataset_with_warning(
            lake_root,
            "opportunity_cost_by_bucket",
        )
    )
    decision_regret, decision_regret_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "quant_lab_decision_regret",
    )
    alpha_factory_results, alpha_factory_results_warning = read_dataset_with_warning(
        lake_root,
        "alpha_factory_result",
    )
    alpha_factory_promotion, alpha_factory_promotion_warning = read_dataset_with_warning(
        lake_root,
        "alpha_factory_promotion_queue",
    )
    missed_opportunity, missed_opportunity_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "v5_missed_opportunity_audit",
    )
    risk_on_multi_buy, risk_on_multi_buy_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "risk_on_multi_buy_shadow",
    )
    if risk_on_multi_buy.is_empty():
        risk_on_multi_buy, risk_on_multi_buy_warning = _read_web_display_dataset_with_warning(
            lake_root,
            "v5_risk_on_multi_buy_shadow",
        )
    expanded_candidates, expanded_candidates_warning = read_dataset_with_warning(
        lake_root,
        "expanded_universe_candidate",
    )
    expanded_quality, expanded_quality_warning = read_dataset_with_warning(
        lake_root,
        "expanded_universe_quality",
    )
    expanded_events, expanded_events_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "expanded_universe_candidate_event",
    )
    expanded_labels, expanded_labels_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "expanded_universe_candidate_label",
    )
    expanded_promotion, expanded_promotion_warning = read_dataset_with_warning(
        lake_root,
        "expanded_universe_promotion_queue",
    )
    expanded_maturity, expanded_maturity_warning = read_dataset_with_warning(
        lake_root,
        "expanded_universe_candidate_maturity",
    )
    expanded_watchlist, expanded_watchlist_warning = read_dataset_with_warning(
        lake_root,
        "expanded_universe_watchlist",
    )
    expanded_universe, expanded_universe_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "expanded_crypto_universe_shadow",
    )
    symbol_quality, symbol_quality_warning = read_dataset_with_warning(
        lake_root,
        "symbol_quality_score",
    )
    expanded_recommendations, expanded_recommendations_warning = read_dataset_with_warning(
        lake_root,
        "expanded_crypto_recommendations",
    )
    warnings = [
        warning
        for warning in [
            gates_warning,
            evidence_warning,
            discovery_board_warning,
            strategy_evidence_warning,
            strategy_samples_warning,
            research_portfolio_warning,
            candidate_events_warning,
            candidate_labels_warning,
            candidate_quality_warning,
            candidate_outcomes_warning,
            strategy_opportunities_warning,
            trade_opportunities_warning,
            trade_labels_warning,
            trade_similarity_warning,
            trade_judgments_warning,
            trade_bucket_policy_warning,
            trade_opportunity_queue_warning,
            false_block_warning,
            v5_trade_learning_warning,
            v5_trade_attribution_warning,
            opportunity_cost_events_warning,
            opportunity_cost_daily_warning,
            opportunity_cost_by_bucket_warning,
            decision_regret_warning,
            alpha_factory_results_warning,
            alpha_factory_promotion_warning,
            missed_opportunity_warning,
            risk_on_multi_buy_warning,
            expanded_candidates_warning,
            expanded_quality_warning,
            expanded_events_warning,
            expanded_labels_warning,
            expanded_promotion_warning,
            expanded_maturity_warning,
            expanded_watchlist_warning,
            expanded_universe_warning,
            symbol_quality_warning,
            expanded_recommendations_warning,
        ]
        if warning
    ]
    if gates.is_empty():
        warnings.append("gate_decision 数据集缺失或为空")
    if evidence.is_empty():
        warnings.append("alpha_evidence 研究证据尚未生成")
    if discovery_board.is_empty():
        warnings.append("alpha_discovery_board 候选决策面板尚未生成")
    if strategy_evidence.is_empty() and discovery_board.is_empty():
        warnings.append("strategy_evidence 候选发现证据尚未生成")
    if candidate_events.is_empty():
        warnings.append("v5_candidate_event 候选快照尚未入湖")
    if candidate_labels.is_empty():
        warnings.append("v5_candidate_label 前向标签尚未生成")
    if trade_judgments.is_empty():
        warnings.append("trade_level_judgment 逐笔判断尚未生成")
    if trade_opportunity_queue.is_empty():
        warnings.append("trade_level_opportunity_queue 机会队列尚未生成")
    counts: dict[str, int] = {}
    if "status" in gates.columns:
        counts = {
            row["status"]: row["count"]
            for row in gates.group_by("status").len(name="count").sort("status").to_dicts()
        }
    strategy_counts: dict[str, int] = {}
    if "decision" in strategy_evidence.columns:
        strategy_counts = {
            row["decision"]: row["count"]
            for row in strategy_evidence.group_by("decision")
            .len(name="count")
            .sort("decision")
            .to_dicts()
        }
    discovery_counts: dict[str, int] = {}
    if "decision" in discovery_board.columns:
        discovery_counts = {
            row["decision"]: row["count"]
            for row in discovery_board.group_by("decision")
            .len(name="count")
            .sort("decision")
            .to_dicts()
        }
    joined = gates
    if not gates.is_empty() and not evidence.is_empty():
        join_keys = [
            key
            for key in ["alpha_id", "version"]
            if key in gates.columns and key in evidence.columns
        ]
        if join_keys:
            joined = gates.join(evidence, on=join_keys, how="left", suffix="_evidence")
    return {
        "gates": redact_frame(joined).head(DISPLAY_LIMIT),
        "counts": counts,
        "alpha_discovery_board": redact_frame(_alpha_discovery_board_table(discovery_board)).head(
            DISPLAY_LIMIT
        ),
        "strategy_opportunity_advisory": redact_frame(
            _strategy_opportunity_table(strategy_opportunities)
        ).head(DISPLAY_LIMIT),
        "trade_opportunity_event": redact_frame(_trade_opportunity_table(trade_opportunities)).head(
            DISPLAY_LIMIT
        ),
        "trade_opportunity_label": redact_frame(trade_labels).head(DISPLAY_LIMIT),
        "trade_level_similarity_outcome": redact_frame(trade_similarity).head(DISPLAY_LIMIT),
        "trade_level_judgment": redact_frame(_trade_level_judgment_table(trade_judgments)).head(
            DISPLAY_LIMIT
        ),
        "trade_level_bucket_policy": redact_frame(trade_bucket_policy).head(DISPLAY_LIMIT),
        "trade_level_opportunity_queue": redact_frame(trade_opportunity_queue).head(DISPLAY_LIMIT),
        "quant_lab_false_block_audit": redact_frame(false_block_audit).head(DISPLAY_LIMIT),
        "v5_trade_learning_sample": redact_frame(v5_trade_learning).head(DISPLAY_LIMIT),
        "v5_trade_outcome_attribution": redact_frame(v5_trade_attribution).head(DISPLAY_LIMIT),
        "quant_lab_opportunity_cost_event": redact_frame(opportunity_cost_events).head(
            DISPLAY_LIMIT
        ),
        "quant_lab_opportunity_cost_daily": redact_frame(opportunity_cost_daily).head(
            DISPLAY_LIMIT
        ),
        "opportunity_cost_by_bucket": redact_frame(opportunity_cost_by_bucket).head(DISPLAY_LIMIT),
        "quant_lab_decision_regret": redact_frame(decision_regret).head(DISPLAY_LIMIT),
        "alpha_factory_result": redact_frame(
            _alpha_factory_result_table(alpha_factory_results)
        ).head(DISPLAY_LIMIT),
        "alpha_factory_promotion_queue": redact_frame(
            _alpha_factory_promotion_table(alpha_factory_promotion)
        ).head(DISPLAY_LIMIT),
        "missed_opportunity_audit": redact_frame(
            _missed_opportunity_table(missed_opportunity)
        ).head(DISPLAY_LIMIT),
        "risk_on_multi_buy_shadow": redact_frame(_risk_on_multi_buy_table(risk_on_multi_buy)).head(
            DISPLAY_LIMIT
        ),
        "expanded_universe_candidate": redact_frame(
            _expanded_candidate_table(expanded_candidates)
        ).head(DISPLAY_LIMIT),
        "expanded_universe_quality": redact_frame(
            _symbol_quality_table(
                expanded_quality if not expanded_quality.is_empty() else symbol_quality
            )
        ).head(DISPLAY_LIMIT),
        "expanded_universe_candidate_event": redact_frame(expanded_events).head(DISPLAY_LIMIT),
        "expanded_universe_candidate_label": redact_frame(expanded_labels).head(DISPLAY_LIMIT),
        "expanded_universe_promotion_queue": redact_frame(
            _expanded_promotion_table(expanded_promotion)
        ).head(DISPLAY_LIMIT),
        "expanded_universe_candidate_maturity": redact_frame(expanded_maturity).head(DISPLAY_LIMIT),
        "expanded_universe_watchlist": redact_frame(
            _expanded_watchlist_table(expanded_watchlist)
        ).head(DISPLAY_LIMIT),
        "expanded_crypto_universe_shadow": redact_frame(
            _expanded_universe_table(expanded_universe)
        ).head(DISPLAY_LIMIT),
        "symbol_quality_score": redact_frame(_symbol_quality_table(symbol_quality)).head(
            DISPLAY_LIMIT
        ),
        "expanded_crypto_recommendations": redact_frame(expanded_recommendations).tail(1),
        "strategy_evidence": redact_frame(_strategy_evidence_table(strategy_evidence)).head(
            DISPLAY_LIMIT
        ),
        "research_portfolio_status": redact_frame(
            _research_portfolio_table(research_portfolio)
        ).head(DISPLAY_LIMIT),
        "strategy_samples": redact_frame(_strategy_sample_table(strategy_samples)).head(
            DISPLAY_LIMIT
        ),
        "candidate_events": redact_frame(_candidate_event_table(candidate_events)).head(
            DISPLAY_LIMIT
        ),
        "candidate_labels": redact_frame(_candidate_label_table(candidate_labels)).head(
            DISPLAY_LIMIT
        ),
        "candidate_quality": redact_frame(candidate_quality.tail(5)),
        "candidate_outcomes": redact_frame(_candidate_outcome_table(candidate_outcomes)).head(
            DISPLAY_LIMIT
        ),
        "strategy_counts": strategy_counts,
        "alpha_discovery_counts": discovery_counts,
        "warnings": warnings,
    }


def _expanded_universe_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "rank",
        "symbol",
        "is_current_v5_symbol",
        "quality_score",
        "recommendation",
        "quote_volume_24h",
        "avg_spread_bps",
        "data_coverage",
        "btc_correlation",
        "blocking_reasons",
        "notes",
        "generated_at",
    ]
    selected = [column for column in columns if column in frame.columns]
    table = frame.select(selected) if selected else frame
    return table.sort("rank") if "rank" in table.columns else table


def _alpha_factory_result_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "decision",
        "strategy_candidate",
        "template_family",
        "universe_type",
        "horizon_hours",
        "alpha_factory_score",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "complete_sample_count",
        "recent_sample_sufficient",
        "cost_quality_score",
        "paper_ready_block_reasons",
        "generated_at",
    ]
    table = _select_existing_columns(frame, columns)
    table = _with_alpha_factory_focus(table)
    table = _sort_by_priority(
        table,
        "decision",
        DECISION_PRIORITY,
        ["strategy_candidate", "horizon_hours"],
    )
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "decision",
            "strategy_candidate",
            "horizon_hours",
            "key_metrics",
            "recent_sample_sufficient",
            "cost_quality_score",
            "paper_ready_block_reasons",
            "generated_at",
        ],
    )


def _alpha_factory_promotion_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "promotion_state",
        "recommended_mode",
        "strategy_candidate",
        "symbol",
        "horizon_hours",
        "alpha_factory_score",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "complete_sample_count",
        "live_block_reasons",
        "generated_at",
    ]
    table = _select_existing_columns(frame, columns)
    table = _with_alpha_factory_focus(table)
    table = _sort_by_priority(
        table,
        "recommended_mode",
        RECOMMENDED_MODE_PRIORITY,
        ["promotion_state", "strategy_candidate", "symbol", "horizon_hours"],
    )
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "recommended_mode",
            "strategy_candidate",
            "symbol",
            "horizon_hours",
            "key_metrics",
            "live_block_reasons",
            "generated_at",
        ],
    )


def _missed_opportunity_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "outcome_if_blocked",
        "symbol",
        "current_regime",
        "final_score",
        "alpha6_side",
        "v5_final_decision",
        "actual_trade_opened",
        "quant_lab_recommended_mode",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_24h_net_bps",
        "ts_utc",
    ]
    table = _select_existing_columns(frame, columns)
    table = _with_missed_opportunity_focus(table)
    sort_columns = [
        column for column in ["outcome_if_blocked", "symbol", "ts_utc"] if column in table.columns
    ]
    table = table.sort(sort_columns) if sort_columns else table
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "outcome_if_blocked",
            "symbol",
            "current_regime",
            "key_metrics",
            "v5_final_decision",
            "actual_trade_opened",
            "quant_lab_recommended_mode",
            "ts_utc",
        ],
    )


def _with_missed_opportunity_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "outcome_if_blocked",
            "symbol",
            "current_regime",
            "final_score",
            "actual_trade_opened",
            "quant_lab_recommended_mode",
            "future_4h_net_bps",
            "future_8h_net_bps",
            "future_24h_net_bps",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_missed_opportunity_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_missed_opportunity_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _missed_opportunity_takeaway(row: dict[str, Any]) -> str:
    symbol = str(row.get("symbol") or "未知标的")
    outcome = str(row.get("outcome_if_blocked") or "")
    if outcome == "quant_lab_would_have_missed_profit":
        return f"中台偏保守：若硬拦 {symbol} 会错过盈利"
    if outcome == "quant_lab_correctly_avoided_loss":
        return f"中台有效避险：拦截 {symbol} 会避开亏损"
    if outcome == "v5_missed_profit_opportunity":
        return f"V5 可能错过机会：{symbol} 后续为正"
    if outcome == "v5_correctly_skipped_loss":
        return f"V5 跳过正确：{symbol} 后续为负"
    return f"待观察：{symbol} 结果尚未成熟"


def _missed_opportunity_key_metrics(row: dict[str, Any]) -> str:
    parts: list[str] = []
    score = _as_float(row.get("final_score"))
    if score is not None:
        parts.append(f"最终分 {score:.2f}")
    for label, column in [
        ("4h", "future_4h_net_bps"),
        ("8h", "future_8h_net_bps"),
        ("24h", "future_24h_net_bps"),
    ]:
        value = _as_float(row.get(column))
        if value is not None:
            parts.append(f"{label} {_format_bps(value)}")
    mode = row.get("quant_lab_recommended_mode")
    if mode:
        parts.append(f"中台建议 {display_value(mode)}")
    opened = row.get("actual_trade_opened")
    if opened is not None:
        parts.append(f"V5开仓 {display_value(opened)}")
    return "；".join(parts) if parts else "暂无关键指标"


def _risk_on_multi_buy_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "ts_utc",
        "current_regime",
        "regime_source",
        "top_k",
        "selected_symbols",
        "would_buy_symbol",
        "final_score",
        "future_4h_net_bps",
        "future_8h_net_bps",
        "future_24h_net_bps",
        "portfolio_avg_net_bps",
        "actual_v5_bought_symbols",
        "missed_symbols",
        "vs_actual_v5_net_bps",
    ]
    table = _select_existing_columns(frame, columns)
    table = _with_risk_on_multi_buy_focus(table)
    sort_columns = [
        column for column in ["ts_utc", "top_k", "would_buy_symbol"] if column in table.columns
    ]
    table = table.sort(sort_columns, descending=True) if sort_columns else table
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "current_regime",
            "regime_source",
            "selected_symbols",
            "top_k",
            "key_metrics",
            "actual_v5_bought_symbols",
            "missed_symbols",
            "ts_utc",
        ],
    )


def _with_risk_on_multi_buy_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "current_regime",
            "regime_source",
            "top_k",
            "selected_symbols",
            "would_buy_symbol",
            "final_score",
            "future_4h_net_bps",
            "future_8h_net_bps",
            "future_24h_net_bps",
            "portfolio_avg_net_bps",
            "actual_v5_bought_symbols",
            "missed_symbols",
            "vs_actual_v5_net_bps",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_risk_on_multi_buy_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_risk_on_multi_buy_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _risk_on_multi_buy_takeaway(row: dict[str, Any]) -> str:
    top_k = _as_int(row.get("top_k")) or 0
    selected = row.get("selected_symbols")
    selected_text = str(display_value(selected)) if selected is not None else "暂无标的"
    portfolio = _as_float(row.get("portfolio_avg_net_bps"))
    missed = row.get("missed_symbols")
    if portfolio is not None and portfolio > 0:
        return f"Risk-on 影子：Top{top_k} {selected_text} 后验为正，继续比较是否漏买"
    if missed not in (None, "", "[]", []):
        return f"Risk-on 影子：发现未买标的 {display_value(missed)}，需要继续观察"
    return f"Risk-on 影子：Top{top_k} {selected_text} 仅作多币买入研究"


def _risk_on_multi_buy_key_metrics(row: dict[str, Any]) -> str:
    parts: list[str] = []
    score = _as_float(row.get("final_score"))
    if score is not None:
        parts.append(f"最终分 {score:.2f}")
    for label, column in [
        ("组合", "portfolio_avg_net_bps"),
        ("相对V5", "vs_actual_v5_net_bps"),
        ("4h", "future_4h_net_bps"),
        ("8h", "future_8h_net_bps"),
        ("24h", "future_24h_net_bps"),
    ]:
        value = _as_float(row.get(column))
        if value is not None:
            parts.append(f"{label} {_format_bps(value)}")
    return "；".join(parts) if parts else "暂无关键指标"


def _expanded_candidate_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "symbol",
        "universe_type",
        "candidate_state",
        "bar_coverage",
        "spread_bps_p75",
        "quote_volume_24h",
        "min_notional_ok",
        "active_trading",
        "blocking_reasons",
        "generated_at",
    ]
    selected = [column for column in columns if column in frame.columns]
    return frame.select(selected) if selected else frame


def _expanded_watchlist_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "watchlist_type",
        "symbol",
        "quality_score",
        "recommendation",
        "complete_sample_count",
        "positive_short_horizon_count",
        "best_short_horizon_hours",
        "best_short_avg_net_bps",
        "win_rate",
        "p25_net_bps",
        "maturity_state",
        "watch_reason",
        "generated_at",
    ]
    table = _select_existing_columns(frame, columns)
    table = _with_expanded_watchlist_focus(table)
    sort_columns = [column for column in ["watchlist_type", "symbol"] if column in table.columns]
    if sort_columns:
        table = table.sort(sort_columns)
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "watchlist_type",
            "symbol",
            "recommendation",
            "key_metrics",
            "maturity_state",
            "watch_reason",
            "generated_at",
        ],
    )


def _with_expanded_watchlist_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "watchlist_type",
            "symbol",
            "quality_score",
            "recommendation",
            "complete_sample_count",
            "positive_short_horizon_count",
            "best_short_horizon_hours",
            "best_short_avg_net_bps",
            "win_rate",
            "p25_net_bps",
            "maturity_state",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_expanded_watchlist_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_expanded_watchlist_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _expanded_watchlist_takeaway(row: dict[str, Any]) -> str:
    symbol = str(row.get("symbol") or "未知标的")
    watchlist_type = str(row.get("watchlist_type") or "")
    if watchlist_type == "outcome_watchlist":
        return f"收益苗头：{symbol} 短周期表现值得继续 shadow"
    if watchlist_type == "quality_watchlist":
        return f"质量观察：{symbol} 流动性/点差较好，先补收益样本"
    if watchlist_type == "reject_list":
        return f"低优先：{symbol} 当前不进扩展重点"
    return f"扩展币池观察：{symbol}"


def _expanded_watchlist_key_metrics(row: dict[str, Any]) -> str:
    parts: list[str] = []
    quality = _as_float(row.get("quality_score"))
    if quality is not None:
        parts.append(f"质量分 {quality:.1f}")
    complete = _as_int(row.get("complete_sample_count"))
    if complete is not None:
        parts.append(f"完整样本 {complete}")
    positive_count = _as_int(row.get("positive_short_horizon_count"))
    if positive_count is not None:
        parts.append(f"短周期为正 {positive_count}")
    best_horizon = _as_int(row.get("best_short_horizon_hours"))
    best_net = _as_float(row.get("best_short_avg_net_bps"))
    if best_horizon is not None and best_net is not None:
        parts.append(f"最佳 {best_horizon}h {_format_bps(best_net)}")
    win_rate = _as_float(row.get("win_rate"))
    if win_rate is not None:
        parts.append(f"胜率 {_format_pct(win_rate)}")
    p25 = _as_float(row.get("p25_net_bps"))
    if p25 is not None:
        parts.append(f"P25 {_format_bps(p25)}")
    return "；".join(parts) if parts else "暂无关键指标"


def _expanded_promotion_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "symbol",
        "strategy_candidate",
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
        "max_live_notional_usdt",
        "generated_at",
    ]
    selected = [column for column in columns if column in frame.columns]
    table = frame.select(selected) if selected else frame
    table = _with_expanded_promotion_focus(table)
    return (
        _select_existing_columns(
            table.sort(["promotion_state", "symbol", "strategy_candidate"]),
            [
                "takeaway",
                "recommended_mode",
                "symbol",
                "strategy_candidate",
                "horizon_hours",
                "key_metrics",
                "live_block_reasons",
                "replacement_target_candidate",
                "generated_at",
            ],
        )
        if {"promotion_state", "symbol", "strategy_candidate"}.issubset(table.columns)
        else table
    )


def _symbol_quality_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "symbol",
        "quality_score",
        "recommendation",
        "quote_volume_24h",
        "avg_spread_bps",
        "data_coverage",
        "avg_24h_net_bps",
        "avg_48h_net_bps",
        "win_rate_24h",
        "win_rate_48h",
        "btc_correlation",
        "blocking_reasons",
    ]
    selected = [column for column in columns if column in frame.columns]
    table = frame.select(selected) if selected else frame
    table = _with_symbol_quality_focus(table)
    if "quality_score" in table.columns:
        table = table.sort("quality_score", descending=True)
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "symbol",
            "recommendation",
            "key_metrics",
            "blocking_reasons",
        ],
    )


def _with_expanded_promotion_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "promotion_state",
            "recommended_mode",
            "symbol",
            "strategy_candidate",
            "horizon_hours",
            "complete_sample_count",
            "avg_net_bps",
            "p25_net_bps",
            "win_rate",
            "cost_source_mix",
            "max_live_notional_usdt",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_expanded_promotion_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_strategy_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _expanded_promotion_takeaway(row: dict[str, Any]) -> str:
    symbol = str(row.get("symbol") or "未知标的")
    candidate = _friendly_label(row.get("strategy_candidate"))
    state = str(row.get("promotion_state") or "").upper()
    mode = str(row.get("recommended_mode") or "").lower()
    if state == "KILL" or mode == "none":
        return f"扩展币池淘汰：{symbol} {candidate} 暂不继续"
    if state == "PAPER_READY" or mode == "paper":
        return f"扩展币池 paper：{symbol} {candidate} 只允许纸面观察，实盘为 0"
    if state == "KEEP_SHADOW" or mode == "shadow":
        return f"扩展币池影子：{symbol} {candidate} 有苗头但样本仍不足"
    return f"扩展币池研究：{symbol} {candidate} 等待更多样本"


def _with_symbol_quality_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "symbol",
            "quality_score",
            "recommendation",
            "quote_volume_24h",
            "avg_spread_bps",
            "data_coverage",
            "avg_24h_net_bps",
            "avg_48h_net_bps",
            "win_rate_24h",
            "win_rate_48h",
            "btc_correlation",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_symbol_quality_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_symbol_quality_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _symbol_quality_takeaway(row: dict[str, Any]) -> str:
    symbol = str(row.get("symbol") or "未知标的")
    recommendation = str(row.get("recommendation") or "")
    quality = _as_float(row.get("quality_score"))
    if "reject" in recommendation:
        return f"暂不扩展：{symbol} 被过滤或优先级低"
    if "outcome" in recommendation:
        return f"收益苗头观察：{symbol} 需要继续补样本"
    if "quality" in recommendation or (quality is not None and quality >= 75):
        return f"质量观察：{symbol} 流动性/点差较好，但还不是替换建议"
    return f"扩展观察：{symbol} 等待质量和收益证据"


def _symbol_quality_key_metrics(row: dict[str, Any]) -> str:
    parts: list[str] = []
    quality = _as_float(row.get("quality_score"))
    if quality is not None:
        parts.append(f"质量分 {quality:.1f}")
    spread = _as_float(row.get("avg_spread_bps"))
    if spread is not None:
        parts.append(f"平均价差 {_format_bps(spread)}")
    coverage = _as_float(row.get("data_coverage"))
    if coverage is not None:
        parts.append(f"K线覆盖 {_format_pct(coverage)}")
    avg_24 = _as_float(row.get("avg_24h_net_bps"))
    if avg_24 is not None:
        parts.append(f"24h {_format_bps(avg_24)}")
    avg_48 = _as_float(row.get("avg_48h_net_bps"))
    if avg_48 is not None:
        parts.append(f"48h {_format_bps(avg_48)}")
    win_24 = _as_float(row.get("win_rate_24h"))
    if win_24 is not None:
        parts.append(f"24h胜率 {_format_pct(win_24)}")
    corr = _as_float(row.get("btc_correlation"))
    if corr is not None:
        parts.append(f"BTC相关 {corr:.2f}")
    return "；".join(parts) if parts else "暂无关键指标"


def _alpha_discovery_board_table(board: pl.DataFrame) -> pl.DataFrame:
    if board.is_empty():
        return board
    columns = [
        "decision",
        "strategy_candidate",
        "symbol",
        "regime_state",
        "horizon_hours",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "complete_sample_count",
        "sample_count",
        "avg_mfe_bps",
        "avg_mae_bps",
        "cost_source_mix",
        "paper_days",
        "decision_reasons",
        "as_of_date",
        "created_at",
    ]
    table = _select_existing_columns(board, columns)
    table = _dedupe_latest_web_rows(
        table,
        keys=["strategy_candidate", "symbol", "horizon_hours"],
        timestamp_columns=["as_of_date", "created_at"],
    )
    table = _with_research_result_focus(table)
    table = _sort_by_priority(
        table,
        "decision",
        DECISION_PRIORITY,
        ["strategy_candidate", "symbol", "horizon_hours"],
    )
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "decision",
            "strategy_candidate",
            "symbol",
            "horizon_hours",
            "key_metrics",
            "decision_reasons",
            "cost_source_mix",
            "as_of_date",
        ],
    )


def _strategy_evidence_table(strategy_evidence: pl.DataFrame) -> pl.DataFrame:
    if strategy_evidence.is_empty():
        return strategy_evidence
    columns = [
        "candidate_name",
        "strategy_candidate",
        "symbol",
        "horizon_hours",
        "decision",
        "avg_net_bps",
        "p25_net_bps",
        "win_rate",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps_by_horizon",
        "win_rate_by_horizon",
        "downside_p25_by_horizon",
        "cost_sensitivity",
        "regime_breakdown",
        "decision_reasons",
        "as_of_date",
        "created_at",
    ]
    table = _select_existing_columns(strategy_evidence, columns)
    candidate_column = (
        "strategy_candidate" if "strategy_candidate" in table.columns else "candidate_name"
    )
    table = _with_research_result_focus(table)
    table = _sort_by_priority(table, "decision", DECISION_PRIORITY, [candidate_column])
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "decision",
            candidate_column,
            "symbol",
            "horizon_hours",
            "key_metrics",
            "decision_reasons",
            "regime_breakdown",
            "as_of_date",
        ],
    )


def _research_portfolio_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "status",
        "action",
        "strategy_candidate",
        "reason",
        "avg_net_bps",
        "win_rate",
        "complete_sample_count",
        "paper_days",
        "entry_day_count",
        "paper_negative_streak",
        "next_review_date",
        "module",
        "research_id",
        "created_at",
    ]
    table = _select_existing_columns(frame, columns)
    table = _dedupe_latest_web_rows(
        table,
        keys=["research_id"],
        timestamp_columns=["as_of_date", "created_at", "last_review_date"],
    )
    table = _with_normalized_research_action(table)
    table = _with_research_portfolio_focus(table)
    table = _sort_by_priority(table, "status", RESEARCH_STATUS_PRIORITY, ["module", "research_id"])
    table = _sort_by_priority(table, "action", RESEARCH_ACTION_PRIORITY, ["status", "research_id"])
    return _select_existing_columns(
        table,
        [
            "takeaway",
            "action",
            "strategy_candidate",
            "key_metrics",
            "reason",
            "downgrade_reason",
            "next_review_date",
            "module",
            "research_id",
        ],
    )


def _with_research_portfolio_focus(table: pl.DataFrame) -> pl.DataFrame:
    fields = [
        column
        for column in [
            "status",
            "action",
            "research_id",
            "strategy_candidate",
            "complete_sample_count",
            "avg_net_bps",
            "win_rate",
            "paper_days",
            "entry_day_count",
            "paper_negative_streak",
            "priority",
        ]
        if column in table.columns
    ]
    if not fields:
        return table
    return table.with_columns(
        pl.struct(fields)
        .map_elements(_research_portfolio_takeaway, return_dtype=pl.Utf8)
        .alias("takeaway"),
        pl.struct(fields)
        .map_elements(_research_portfolio_key_metrics, return_dtype=pl.Utf8)
        .alias("key_metrics"),
    )


def _research_portfolio_takeaway(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "").upper()
    action = str(row.get("action") or "").upper()
    research_id = _friendly_label(row.get("strategy_candidate") or row.get("research_id"))
    if status == "KILL" or action == "CLOSE_RESEARCH":
        return f"关闭：{research_id} 不再进入重点日报或晋级队列"
    if status == "DOWNGRADED_FROM_PAPER":
        return f"降级：{research_id} 停止 paper 推荐，回到 shadow/复查"
    if status == "PAPER":
        return f"继续纸面：{research_id} 只做 paper，不放 live"
    if status == "SHADOW" or action in {"CONTINUE_SHADOW", "REGIME_SHADOW"}:
        return f"继续影子：{research_id} 收集更多样本"
    if status == "ACTIVE" or action == "ACTIVE_DIAGNOSTIC":
        return f"诊断中：{research_id} 用于定位问题，不做交易建议"
    if status == "BASELINE_ONLY":
        return f"基线：{research_id} 只作对照，不再当全局 gate"
    if status == "PAUSED":
        return f"暂停：{research_id} 等待样本、触发或成本质量改善"
    return f"继续观察：{research_id}"


def _research_portfolio_key_metrics(row: dict[str, Any]) -> str:
    parts: list[str] = []
    complete = _as_int(row.get("complete_sample_count"))
    if complete is not None:
        parts.append(f"完整样本 {complete}")
    avg = _as_float(row.get("avg_net_bps"))
    if avg is not None:
        parts.append(f"平均净收益 {_format_bps(avg)}")
    win_rate = _as_float(row.get("win_rate"))
    if win_rate is not None:
        parts.append(f"胜率 {_format_pct(win_rate)}")
    entry_days = _as_int(row.get("entry_day_count"))
    if entry_days:
        parts.append(f"入场日 {entry_days}")
    paper_days = _as_int(row.get("paper_days"))
    if paper_days:
        parts.append(f"paper {paper_days}天")
    negative_streak = _as_int(row.get("paper_negative_streak"))
    if negative_streak:
        parts.append(f"连续转弱 {negative_streak}天")
    priority = str(row.get("priority") or "").upper()
    if priority == "LOW":
        parts.append("优先级低")
    return "；".join(parts) if parts else "暂无关键指标"


def _with_normalized_research_action(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "action" not in frame.columns:
        return frame
    fields = [column for column in ["status", "action"] if column in frame.columns]
    if not fields:
        return frame
    return frame.with_columns(
        pl.struct(fields)
        .map_elements(_normalize_research_action_row, return_dtype=pl.Utf8)
        .alias("action")
    )


def _normalize_research_action_row(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "").strip().upper()
    action = str(row.get("action") or "").strip()
    lowered = action.lower()
    if status == "KILL" or lowered.startswith("close") or "closed" in lowered:
        return "CLOSE_RESEARCH"
    if status == "DOWNGRADED_FROM_PAPER":
        return "CONTINUE_SHADOW_OR_REVIEW"
    if status == "PAUSED":
        return "PAUSED_TO_WEEKLY"
    if status == "BASELINE_ONLY":
        return "BASELINE_ONLY"
    if status == "PAPER":
        return "CONTINUE_PAPER"
    if status in {"SHADOW", "REGIME_SHADOW"}:
        return "CONTINUE_SHADOW"
    if status in {"ACTIVE", "ACTIVE_DIAGNOSTIC"}:
        return "ACTIVE_DIAGNOSTIC"
    return action or "KEEP_RESEARCH"


def _strategy_sample_table(strategy_samples: pl.DataFrame) -> pl.DataFrame:
    if strategy_samples.is_empty():
        return strategy_samples
    columns = [
        "ts_utc",
        "symbol",
        "candidate_name",
        "entry_condition_signal",
        "block_reason",
        "final_score",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "alpha6_score",
        "alpha6_side",
        "regime_state",
        "protect_level",
        "expected_edge_bps",
        "required_edge_bps",
        "cost_source",
        "cost_bps",
        "label_status",
        "net_bps_after_cost_24h",
        "win_24h",
        "drawdown_proxy_bps_24h",
        "net_bps_after_cost_72h",
        "win_72h",
        "drawdown_proxy_bps_72h",
        "source_dataset",
    ]
    selected = [column for column in columns if column in strategy_samples.columns]
    if not selected:
        return strategy_samples
    table = strategy_samples.select(selected)
    sort_columns = [
        column for column in ["candidate_name", "symbol", "ts_utc"] if column in table.columns
    ]
    return table.sort(sort_columns) if sort_columns else table


def _candidate_event_table(candidate_events: pl.DataFrame) -> pl.DataFrame:
    if candidate_events.is_empty():
        return candidate_events
    columns = [
        "ts_utc",
        "run_id",
        "symbol",
        "strategy_candidate",
        "final_decision",
        "block_reason",
        "final_score",
        "rank",
        "f1_mom_5d",
        "f2_mom_20d",
        "f3_vol_adj_ret",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "alpha6_score",
        "alpha6_side",
        "expected_edge_bps",
        "required_edge_bps",
        "cost_bps",
        "cost_source",
        "candidate_id",
    ]
    selected = [column for column in columns if column in candidate_events.columns]
    table = candidate_events.select(selected) if selected else candidate_events
    sort_columns = [
        column for column in ["run_id", "symbol", "strategy_candidate"] if column in table.columns
    ]
    return table.sort(sort_columns) if sort_columns else table


def _trade_opportunity_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "decision_ts",
        "symbol",
        "side",
        "intent",
        "strategy_candidate",
        "rank",
        "alpha6_score",
        "expected_edge_bps",
        "required_edge_bps",
        "edge_required_ratio",
        "cost_gate_verified",
        "would_block_by_cost",
        "quant_lab_permission",
        "quant_lab_permission_status",
        "v5_would_open",
        "actual_submitted",
        "event_id",
    ]
    table = _select_existing_columns(frame, columns)
    sort_columns = [column for column in ["decision_ts", "symbol"] if column in table.columns]
    descending = [column == "decision_ts" for column in sort_columns]
    return table.sort(sort_columns, descending=descending) if sort_columns else table


def _trade_level_judgment_table(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    columns = [
        "decision_ts",
        "symbol",
        "trade_level_decision",
        "hard_safety_veto",
        "risk_permission_veto",
        "strategy_advisory_veto",
        "v5_high_confidence_opportunity",
        "similar_sample_count",
        "similar_median_after_cost_bps",
        "similar_p25_after_cost_bps",
        "max_single_order_usdt",
        "reason",
        "event_id",
    ]
    table = _select_existing_columns(frame, columns)
    sort_columns = [column for column in ["decision_ts", "symbol"] if column in table.columns]
    descending = [column == "decision_ts" for column in sort_columns]
    return table.sort(sort_columns, descending=descending) if sort_columns else table


def _candidate_label_table(candidate_labels: pl.DataFrame) -> pl.DataFrame:
    if candidate_labels.is_empty():
        return candidate_labels
    columns = [
        "ts_utc",
        "symbol",
        "strategy_candidate",
        "block_reason",
        "horizon_hours",
        "gross_bps",
        "net_bps_after_cost",
        "mfe_bps",
        "mae_bps",
        "win",
        "label_status",
        "label_reason",
        "cost_bps",
        "cost_source",
        "candidate_id",
    ]
    selected = [column for column in columns if column in candidate_labels.columns]
    table = candidate_labels.select(selected) if selected else candidate_labels
    sort_columns = [
        column
        for column in ["strategy_candidate", "symbol", "horizon_hours"]
        if column in table.columns
    ]
    return table.sort(sort_columns) if sort_columns else table


def _candidate_outcome_table(candidate_outcomes: pl.DataFrame) -> pl.DataFrame:
    if candidate_outcomes.is_empty():
        return candidate_outcomes
    columns = [
        "block_reason",
        "strategy_candidate",
        "symbol",
        "horizon_hours",
        "sample_count",
        "complete_sample_count",
        "avg_net_bps",
        "median_net_bps",
        "win_rate",
        "downside_p25_bps",
        "avg_mfe_bps",
        "avg_mae_bps",
        "date",
    ]
    selected = [column for column in columns if column in candidate_outcomes.columns]
    table = candidate_outcomes.select(selected) if selected else candidate_outcomes
    sort_columns = [
        column
        for column in ["strategy_candidate", "symbol", "horizon_hours"]
        if column in table.columns
    ]
    return table.sort(sort_columns) if sort_columns else table


def feature_summary(lake_root: str | Path) -> dict[str, Any]:
    features, feature_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "feature_value",
    )
    coverage, coverage_warning = read_dataset_with_warning(lake_root, "feature_coverage_daily")
    anomalies, anomaly_warning = read_dataset_with_warning(lake_root, "feature_anomaly_daily")
    warnings = [
        warning for warning in [feature_warning, coverage_warning, anomaly_warning] if warning
    ]
    if features.is_empty():
        warnings.append("feature_value 数据集缺失或为空")
    latest = _latest_features_table(features)
    null_ratio = 0.0
    if not features.is_empty() and "value" in features.columns:
        null_ratio = float(features["value"].null_count()) / features.height
    display_features = features.tail(DISPLAY_LIMIT) if not features.is_empty() else features
    return {
        "features": redact_frame(display_features),
        "coverage": redact_frame(coverage),
        "anomalies": redact_frame(anomalies),
        "latest": redact_frame(latest),
        "null_ratio": null_ratio,
        "top_anomalies": _top_feature_anomalies(anomalies),
        "warnings": warnings,
    }


def strategy_consumer_summary(lake_root: str | Path) -> dict[str, Any]:
    permissions, permissions_warning = read_dataset_with_warning(lake_root, "risk_permission")
    audits, audits_warning = _read_web_display_dataset_with_warning(
        lake_root,
        "decision_audit",
    )
    warnings = [warning for warning in [permissions_warning, audits_warning] if warning]
    if permissions.is_empty():
        warnings.append("risk_permission 数据集缺失或为空")
    permission_rows = _dedupe_latest_web_rows(
        permissions,
        keys=["strategy"],
        timestamp_columns=["as_of_ts", "created_at", "source_bundle_ts"],
    )
    if not permission_rows.is_empty():
        for sort_column in ("as_of_ts", "created_at", "source_bundle_ts"):
            if sort_column in permission_rows.columns:
                permission_rows = permission_rows.sort(
                    sort_column,
                    descending=True,
                    nulls_last=True,
                )
                break

    latest_by_strategy: dict[str, str] = {}
    if not permission_rows.is_empty() and {"strategy", "permission"}.issubset(
        permission_rows.columns
    ):
        sort_column = "as_of_ts" if "as_of_ts" in permission_rows.columns else "created_at"
        if sort_column not in permission_rows.columns:
            sort_column = "strategy"
        for row in permission_rows.sort(sort_column).to_dicts():
            status = row.get("permission_status")
            latest_by_strategy[str(row["strategy"]).lower()] = str(status or row["permission"])

    return {
        "permissions": latest_by_strategy,
        "permission_rows": redact_frame(permission_rows).head(DISPLAY_LIMIT),
        "fallback_rows": _fallback_rows(audits),
        "warnings": warnings,
    }


def v5_telemetry_summary(lake_root: str | Path) -> dict[str, Any]:
    health, health_warning = read_dataset_with_warning(lake_root, "strategy_health_daily")
    gate, gate_warning = read_dataset_with_warning(lake_root, "v5_gate_compliance_daily")
    mode, mode_warning = read_dataset_with_warning(lake_root, "v5_quant_lab_mode_daily")
    enforcement, enforcement_warning = read_dataset_with_warning(
        lake_root,
        "v5_quant_lab_enforcement_daily",
    )
    p3_preflight, p3_preflight_warning = read_dataset_with_warning(
        lake_root,
        "v5_cost_probe_p3_preflight",
    )
    live_execution_status, live_execution_status_warning = read_dataset_with_warning(
        lake_root,
        "v5_cost_probe_live_execution_status",
    )
    live_execution_status = _with_cost_probe_authorization_read_freshness(
        live_execution_status,
        reference_at=datetime.now(UTC),
    )
    probe_order_events, probe_order_events_warning = read_dataset_with_warning(
        lake_root,
        "v5_cost_probe_order_event",
    )
    probe_roundtrip_events, probe_roundtrip_events_warning = read_dataset_with_warning(
        lake_root,
        "v5_cost_probe_roundtrip_event",
    )
    warnings = [
        warning
        for warning in [
            health_warning,
            gate_warning,
            mode_warning,
            enforcement_warning,
            p3_preflight_warning,
            live_execution_status_warning,
            probe_order_events_warning,
            probe_roundtrip_events_warning,
        ]
        if warning
    ]
    p3_latest = _latest_v5_cost_probe_p3_preflight(p3_preflight)
    latest_live_execution_status = _latest_v5_cost_probe_live_execution_status(
        live_execution_status
    )
    latest_order_event = _latest_v5_event_frame(probe_order_events)
    latest_roundtrip_event = _latest_v5_event_frame(probe_roundtrip_events)
    if health.is_empty():
        latest = {}
        if p3_latest:
            latest["cost_probe_p3_preflight"] = p3_latest
        if latest_live_execution_status:
            latest["cost_probe_live_execution_status"] = latest_live_execution_status
        if latest_order_event:
            latest["cost_probe_order_event"] = latest_order_event
        if latest_roundtrip_event:
            latest["cost_probe_roundtrip_event"] = latest_roundtrip_event
        return {
            "latest": latest,
            "health_rows": pl.DataFrame(),
            "gate_compliance_rows": gate,
            "quant_lab_mode_rows": mode,
            "quant_lab_enforcement_rows": enforcement,
            "cost_probe_p3_preflight_rows": p3_preflight.head(DISPLAY_LIMIT),
            "cost_probe_live_execution_status_rows": live_execution_status.head(DISPLAY_LIMIT),
            "cost_probe_order_event_rows": probe_order_events.head(DISPLAY_LIMIT),
            "cost_probe_roundtrip_event_rows": probe_roundtrip_events.head(DISPLAY_LIMIT),
            "warnings": [*warnings, "strategy_health_daily 数据集缺失或为空"],
        }
    health = _sort_v5_freshness_frame(health)
    latest = health.tail(1).to_dicts()[0]
    if p3_latest:
        latest = dict(latest)
        latest["cost_probe_p3_preflight"] = p3_latest
    if latest_live_execution_status:
        latest = dict(latest)
        latest["cost_probe_live_execution_status"] = latest_live_execution_status
    if latest_order_event:
        latest = dict(latest)
        latest["cost_probe_order_event"] = latest_order_event
    if latest_roundtrip_event:
        latest = dict(latest)
        latest["cost_probe_roundtrip_event"] = latest_roundtrip_event
    return {
        "latest": latest,
        "health_rows": health.reverse().head(DISPLAY_LIMIT),
        "gate_compliance_rows": gate.head(DISPLAY_LIMIT),
        "quant_lab_mode_rows": mode.head(DISPLAY_LIMIT),
        "quant_lab_enforcement_rows": enforcement.head(DISPLAY_LIMIT),
        "cost_probe_p3_preflight_rows": p3_preflight.head(DISPLAY_LIMIT),
        "cost_probe_live_execution_status_rows": live_execution_status.head(DISPLAY_LIMIT),
        "cost_probe_order_event_rows": probe_order_events.head(DISPLAY_LIMIT),
        "cost_probe_roundtrip_event_rows": probe_roundtrip_events.head(DISPLAY_LIMIT),
        "warnings": warnings,
    }


def _latest_v5_cost_probe_p3_preflight(frame: pl.DataFrame) -> dict[str, Any] | None:
    if frame.is_empty():
        return None
    sort_column = "generated_at_utc" if "generated_at_utc" in frame.columns else "ingest_ts"
    if sort_column not in frame.columns:
        sort_column = frame.columns[0]
    row = frame.sort(sort_column).tail(1).to_dicts()[0]
    return redact_frame(pl.DataFrame([row])).to_dicts()[0]


def _latest_v5_cost_probe_live_execution_status(
    frame: pl.DataFrame,
) -> dict[str, Any] | None:
    if frame.is_empty():
        return None
    sort_column = "generated_at_utc" if "generated_at_utc" in frame.columns else "ingest_ts"
    if sort_column not in frame.columns:
        sort_column = frame.columns[0]
    row = frame.sort(sort_column).tail(1).to_dicts()[0]
    return redact_frame(pl.DataFrame([row])).to_dicts()[0]


def _with_cost_probe_authorization_read_freshness(
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
        issued_at = _parse_datetime(row.get("authorization_issued_at"))
        expires_at = _parse_datetime(row.get("authorization_expires_at"))
        if issued_at is None or expires_at is None:
            normalized.update(
                {
                    "authorization_fresh_at_read": None,
                    "authorization_age_sec_at_read": None,
                    "authorization_seconds_to_expiry_at_read": None,
                    "authorization_expired_at_read": None,
                }
            )
        else:
            normalized.update(
                {
                    "authorization_fresh_at_read": issued_at <= reference <= expires_at,
                    "authorization_age_sec_at_read": max(
                        int((reference - issued_at).total_seconds()),
                        0,
                    ),
                    "authorization_seconds_to_expiry_at_read": int(
                        (expires_at - reference).total_seconds()
                    ),
                    "authorization_expired_at_read": reference > expires_at,
                }
            )
        rows.append(normalized)
    return pl.DataFrame(rows, infer_schema_length=None)


def _latest_v5_event_frame(frame: pl.DataFrame) -> dict[str, Any] | None:
    if frame.is_empty():
        return None
    sort_column = "event_ts" if "event_ts" in frame.columns else "ingest_ts"
    if sort_column not in frame.columns:
        sort_column = frame.columns[0]
    row = frame.sort(sort_column).tail(1).to_dicts()[0]
    return redact_frame(pl.DataFrame([row])).to_dicts()[0]


def expert_export_summary(exports_root: str | Path) -> dict[str, Any]:
    if _bool_env_value("QUANT_LAB_NAS_EXPORT_ENABLED", default=False):
        return _nas_expert_export_summary()
    root = Path(exports_root)
    cache_key = (
        "expert_export_summary",
        str(root.resolve()),
        _expert_export_source_signature(root),
    )
    cached = _web_cache_get(cache_key, event="expert_export_summary")
    if cached is not None:
        return cached

    indexed = _expert_export_summary_from_index(root)
    if indexed is not None:
        return _web_cache_set(cache_key, indexed)

    packs = _expert_pack_paths(root)
    if not packs:
        return _web_cache_set(
            cache_key,
            {
                "latest_pack": None,
                "packs": pl.DataFrame(),
                "manifest_summary": {},
                "data_quality_summary": {},
                "expert_questions": [],
                "warnings": [f"未在目录下找到专家包：{root}"],
            },
        )

    latest = _latest_authoritative_expert_pack(packs) or packs[0]
    manifest = _read_json_from_zip(latest, "manifest.json")
    data_quality = _read_json_from_zip(latest, "data_quality.json")
    questions = _read_text_from_zip(latest, "expert_questions.md").splitlines()
    pack_rows = [
        {
            "path": str(path),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
            "authoritative_snapshot": _expert_pack_authoritative_snapshot(path),
        }
        for path in packs
    ]
    return _web_cache_set(
        cache_key,
        {
            "latest_pack": str(latest),
            "packs": pl.DataFrame(pack_rows),
            "manifest_summary": manifest,
            "data_quality_summary": data_quality,
            "expert_questions": [line for line in questions if line.strip()][:20],
            "warnings": [WEB_EXPORT_INDEX_MISSING_WARNING],
        },
    )


def _nas_expert_export_summary() -> dict[str, Any]:
    from quant_lab.export_plane.cloud_index import export_plane_status

    queue_root = Path(
        os.environ.get("QUANT_LAB_EXPORT_QUEUE_ROOT", "/var/lib/quant-lab/export_queue")
    )
    payload = export_plane_status(queue_root)
    latest = payload.get("latest_pack") if isinstance(payload.get("latest_pack"), dict) else {}
    rows = payload.get("packs") if isinstance(payload.get("packs"), list) else []
    pack_rows = [
        {
            "path": None,
            "name": row.get("pack_name"),
            "pack_id": row.get("pack_id"),
            "size_bytes": row.get("pack_size_bytes"),
            "modified_at": row.get("accepted_at"),
            "authoritative_snapshot": row.get("authoritative_input_snapshot"),
            "nas_artifact_validated": row.get("nas_artifact_validated"),
            "control_plane_receipt_verified": row.get("control_plane_receipt_verified"),
            "download_ready": row.get("download_ready"),
            "download_url": row.get("download_url"),
            "pack_sha256": row.get("pack_sha256"),
        }
        for row in rows
    ]
    return {
        "latest_pack": latest.get("pack_name"),
        "latest_pack_source": "nas_signed_receipt" if latest else None,
        "packs": pl.DataFrame(pack_rows) if pack_rows else pl.DataFrame(),
        "manifest_summary": latest.get("manifest_summary") or {},
        "data_quality_summary": latest.get("data_quality_summary") or {},
        "expert_questions": latest.get("expert_questions") or [],
        "manual_state": payload.get("state"),
        "manual_status": payload.get("task") or payload.get("request") or {},
        "warnings": [],
        "storage_location": "nas_only",
    }


def _bool_env_value(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


WEB_EXPORT_INDEX_MISSING_WARNING = "web_export_index_missing_scan_zip_fallback"


def _expert_export_summary_from_index(root: Path) -> dict[str, Any] | None:
    index_path = root / "export_index.json"
    if not index_path.is_file():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    manifest_summary = payload.get("manifest_summary") or {}
    if (
        isinstance(manifest_summary, dict)
        and manifest_summary.get("authoritative_snapshot") is False
    ):
        return None
    pack_rows = payload.get("packs")
    if not isinstance(pack_rows, list):
        pack_rows = []
    normalized_pack_rows = _existing_expert_index_pack_rows(root, pack_rows)
    if not normalized_pack_rows and _expert_pack_paths(root):
        return None
    latest_pack = _existing_expert_index_pack_path(root, payload.get("latest_pack"))
    if latest_pack is None and normalized_pack_rows:
        # The index is stale enough that its manifest/data-quality summary may
        # no longer describe the latest real pack; fall back to the zip scan.
        return None
    filesystem_packs = _expert_pack_paths(root)
    if filesystem_packs and normalized_pack_rows:
        indexed_paths = {
            Path(str(row.get("path"))).resolve() for row in normalized_pack_rows if row.get("path")
        }
        newest_filesystem_pack = filesystem_packs[0].resolve()
        if newest_filesystem_pack not in indexed_paths:
            # A newly generated pack can appear before export_index.json is
            # refreshed.  The index summaries then describe an older pack, so
            # scan the zips and let the mtime/authoritative rules choose.
            return None
        indexed_latest = (
            _latest_authoritative_index_pack(normalized_pack_rows)
            or Path(str(normalized_pack_rows[0]["path"])).resolve()
        )
        if latest_pack is not None and latest_pack.resolve() != indexed_latest:
            return None
    return {
        "latest_pack": str(latest_pack) if latest_pack is not None else None,
        "packs": (pl.DataFrame(normalized_pack_rows) if normalized_pack_rows else pl.DataFrame()),
        "manifest_summary": manifest_summary,
        "data_quality_summary": payload.get("data_quality_summary") or {},
        "expert_questions": payload.get("expert_questions") or [],
        "warnings": payload.get("warnings") or [],
    }


def _existing_expert_index_pack_rows(
    root: Path,
    pack_rows: list[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_row in pack_rows:
        if not isinstance(raw_row, dict):
            continue
        path = _existing_expert_index_pack_path(
            root,
            raw_row.get("path"),
            raw_row.get("name"),
        )
        if path is None or path in seen:
            continue
        seen.add(path)
        try:
            stat = path.stat()
        except OSError:
            continue
        row = dict(raw_row)
        row["path"] = str(path)
        row["name"] = path.name
        row["size_bytes"] = stat.st_size
        row["modified_at"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("modified_at") or ""),
            str(row.get("name") or ""),
        ),
        reverse=True,
    )
    return rows


def _latest_authoritative_index_pack(rows: list[dict[str, Any]]) -> Path | None:
    for row in rows:
        value = row.get("authoritative_snapshot")
        if value is True or str(value).strip().lower() in {"true", "1", "yes"}:
            path = row.get("path")
            if path:
                return Path(str(path)).resolve()
    return None


def _existing_expert_index_pack_path(
    root: Path,
    value: Any,
    name: Any | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if value:
        candidates.append(Path(str(value)))
    if name:
        candidates.append(root / str(name))
    for candidate in candidates:
        pack_name = candidate.name
        if not (pack_name.startswith("quant_lab_expert_pack_") and pack_name.endswith(".zip")):
            continue
        path = candidate if candidate.is_absolute() else root / candidate
        try:
            resolved = path.resolve()
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    return None


def _expert_export_source_signature(root: Path) -> tuple[Any, ...]:
    index_path = root / "export_index.json"
    if index_path.is_file():
        try:
            stat = index_path.stat()
            return (
                "export_index",
                stat.st_mtime_ns,
                stat.st_size,
                _latest_expert_pack_signature(root),
            )
        except OSError:
            return ("export_index", "unreadable")
    latest = _latest_expert_pack_signature(root)
    return ("zip_scan", latest)


def _latest_expert_pack_signature(root: Path) -> tuple[Any, ...]:
    packs = _expert_pack_paths(root)
    if not packs:
        return ("missing",)
    signatures: list[tuple[str, Any, Any]] = []
    for path in packs[:24]:
        try:
            stat = path.stat()
        except OSError:
            signatures.append((path.name, "unreadable", "unreadable"))
            continue
        signatures.append((path.name, stat.st_mtime_ns, stat.st_size))
    return ("packs", len(packs), tuple(signatures))


def _expert_pack_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        root.glob("quant_lab_expert_pack_*.zip"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def _latest_authoritative_expert_pack(packs: list[Path]) -> Path | None:
    for path in packs:
        if _expert_pack_authoritative_snapshot(path):
            return path
    return None


def _expert_pack_authoritative_snapshot(path: Path) -> bool:
    manifest = _read_json_from_zip(path, "manifest.json")
    return bool(manifest.get("authoritative_snapshot"))


def default_exports_root(lake_root: str | Path) -> Path:
    root = Path(lake_root)
    if root.name == "lake":
        return root.parent / "exports"
    return root / "exports"


def _latest_v5_bundle_ts(lake_root: str | Path) -> datetime | None:
    candidates = [
        _latest_dataset_timestamp_by_path(
            dataset_path_for(lake_root, "v5_bundle_manifest"),
            "v5_bundle_manifest",
            timestamp_columns=DATASET_TIMESTAMP_COLUMNS["v5_bundle_manifest"],
        ),
        _latest_dataset_timestamp_by_path(
            dataset_path_for(lake_root, "strategy_health_daily"),
            "strategy_health_daily",
            timestamp_columns=("latest_bundle_ts", "created_at", "date"),
        ),
        _latest_dataset_timestamp_by_path(
            Path(lake_root) / "bronze/strategy_telemetry/v5/bundle_manifest",
            "bronze/strategy_telemetry/v5/bundle_manifest",
            timestamp_columns=("bundle_ts", "ingest_ts", "created_at"),
        ),
    ]
    return max((value for value in candidates if value is not None), default=None)


def _sort_v5_freshness_frame(frame: pl.DataFrame) -> pl.DataFrame:
    sort_columns = [
        column for column in ("latest_bundle_ts", "created_at", "date") if column in frame.columns
    ]
    if not sort_columns:
        sort_columns = [frame.columns[0]]
    return frame.sort(sort_columns)


def latest_v5_bundle_ts_for_web(lake_root: str | Path) -> datetime | None:
    return _latest_v5_bundle_ts(lake_root)


def current_enforce_readiness_summary(lake_root: str | Path) -> dict[str, Any]:
    path = Path(lake_root) / "reports" / "v5_enforce_readiness.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "generated_at": payload.get("generated_at"),
        "readiness_status": payload.get("readiness_status"),
        "veto_status": payload.get("veto_status"),
        "entry_status": payload.get("entry_status"),
        "scale_status": payload.get("scale_status"),
        "blocked_reasons": payload.get("blocked_reasons") or [],
        "warning_reasons": payload.get("warning_reasons") or [],
        "entry_blocked_reasons": payload.get("entry_blocked_reasons") or [],
        "scale_blocked_reasons": payload.get("scale_blocked_reasons") or [],
    }


def _latest_dataset_timestamp_by_path(
    path: str | Path,
    dataset_name: str,
    *,
    timestamp_columns: tuple[str, ...],
) -> datetime | None:
    files = _valid_parquet_files(Path(path), invalid_files=invalid_parquet_files(path))
    if not files:
        return None
    try:
        lazy = _scan_parquet_files(files)
        schema = lazy.collect_schema()
        latest, _column = _latest_lazy_timestamp(
            dataset_name,
            lazy,
            schema,
            timestamp_columns=timestamp_columns,
        )
        return latest
    except Exception:
        return None


def _parquet_file_count(path: str | Path) -> int:
    root = Path(path)
    try:
        if root.is_file():
            return 1 if root.suffix == ".parquet" and not _is_internal_lake_path(root) else 0
        if not root.exists():
            return 0
        return len(_parquet_file_candidates(root))
    except OSError:
        return 0


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _is_bootstrap_placeholder(df: pl.DataFrame) -> bool:
    if df.is_empty():
        return False
    return _frame_is_bootstrap_placeholder(_latest_placeholder_slice(df))


def _dataset_has_bootstrap_placeholder(path: Path, dataset_name: str) -> bool:
    latest = _latest_dataset_slice(path, dataset_name)
    return _frame_is_bootstrap_placeholder(latest)


def _latest_dataset_slice(path: Path, dataset_name: str) -> pl.DataFrame:
    files = _valid_parquet_files(path, invalid_files=invalid_parquet_files(path))
    if not files:
        return pl.DataFrame()
    try:
        lazy = _scan_parquet_files(files)
        schema = lazy.collect_schema()
    except Exception:
        return pl.DataFrame()

    columns = DATASET_TIMESTAMP_COLUMNS.get(dataset_name, ("as_of_ts", "created_at", "ingest_ts"))
    for column in columns:
        if column not in schema:
            continue
        try:
            timestamp_expr = _lazy_timestamp_expr(schema, column).alias("__qlab_latest_ts")
            latest_frame = lazy.select(timestamp_expr.max().alias("__latest_ts")).collect()
            latest = latest_frame.item(0, "__latest_ts")
            if latest is None:
                continue
            return (
                lazy.with_columns(timestamp_expr)
                .filter(pl.col("__qlab_latest_ts") == latest)
                .drop("__qlab_latest_ts")
                .collect()
            )
        except Exception:
            continue
    try:
        return lazy.limit(100).collect()
    except Exception:
        return pl.DataFrame()


def _latest_placeholder_slice(df: pl.DataFrame) -> pl.DataFrame:
    for column in ["as_of_ts", "created_at", "ingest_ts"]:
        if column not in df.columns:
            continue
        try:
            non_null = df.filter(pl.col(column).is_not_null())
            if non_null.is_empty():
                continue
            latest_value = non_null.select(pl.col(column).max()).item()
            return non_null.filter(pl.col(column) == latest_value)
        except Exception:
            continue
    return df


def _frame_is_bootstrap_placeholder(df: pl.DataFrame) -> bool:
    if df.is_empty():
        return False
    if "source" in df.columns:
        source_values = {str(value) for value in df["source"].drop_nulls().to_list()}
        if source_values == {"bootstrap_gold_health"} or source_values == {"global_default"}:
            return True
    if "gate_version" in df.columns:
        gate_versions = {str(value) for value in df["gate_version"].drop_nulls().to_list()}
        if gate_versions and all(version.startswith("bootstrap.") for version in gate_versions):
            return True
    if "fallback_level" in df.columns:
        fallback_values = {str(value) for value in df["fallback_level"].drop_nulls().to_list()}
        if fallback_values in [{"BOOTSTRAP_CONSERVATIVE"}, {"GLOBAL_DEFAULT"}]:
            return True
    return False


def latest_market_bars(market: pl.DataFrame) -> pl.DataFrame:
    if market.is_empty():
        return pl.DataFrame()
    return (
        market.group_by(["symbol", "timeframe"])
        .agg(pl.col("ts").max().alias("latest_ts"), pl.len().alias("rows"))
        .sort(["symbol", "timeframe"])
    )


def _latest_market_bars_lazy(lazy: pl.LazyFrame, schema: pl.Schema) -> pl.DataFrame:
    if not {"symbol", "timeframe", "ts"}.issubset(schema):
        return pl.DataFrame()
    try:
        return (
            lazy.with_columns(_lazy_timestamp_expr(schema, "ts").alias("ts"))
            .group_by(["symbol", "timeframe"])
            .agg(pl.col("ts").max().alias("latest_ts"), pl.len().alias("rows"))
            .sort(["symbol", "timeframe"])
            .collect()
        )
    except Exception:
        return pl.DataFrame()


def duplicate_market_bar_count(market: pl.DataFrame) -> int:
    keys = ["venue", "symbol", "timeframe", "ts"]
    if not set(keys).issubset(market.columns):
        return 0
    return market.group_by(keys).len(name="count").filter(pl.col("count") > 1).height


def _duplicate_market_bar_count_lazy(lazy: pl.LazyFrame, schema: pl.Schema) -> int:
    keys = ["venue", "symbol", "timeframe", "ts"]
    if not set(keys).issubset(schema):
        return 0
    try:
        return int(
            lazy.group_by(keys)
            .len(name="count")
            .filter(pl.col("count") > 1)
            .select(pl.len().alias("duplicate_groups"))
            .collect()
            .item(0, "duplicate_groups")
            or 0
        )
    except Exception:
        return 0


def unclosed_market_bar_count(market: pl.DataFrame) -> int:
    if "is_closed" not in market.columns:
        return 0
    return market.filter(pl.col("is_closed") == False).height  # noqa: E712


def _unclosed_market_bar_count_lazy(lazy: pl.LazyFrame, schema: pl.Schema) -> int:
    if "is_closed" not in schema:
        return 0
    try:
        return int(
            lazy.filter(pl.col("is_closed") == False)  # noqa: E712
            .select(pl.len().alias("rows"))
            .collect()
            .item(0, "rows")
            or 0
        )
    except Exception:
        return 0


def market_bar_schema_violations(market: pl.DataFrame) -> list[str]:
    required = {
        "venue",
        "symbol",
        "market_type",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "ingest_ts",
    }
    violations = [f"缺少字段：{column}" for column in sorted(required - set(market.columns))]
    if violations:
        return violations
    invalid_rows = market.filter(
        (pl.col("open") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("high") < pl.col("low"))
        | (pl.col("volume") < 0)
    ).height
    if invalid_rows:
        violations.append(f"OHLCV 无效行数：{invalid_rows}")
    return violations


def _market_bar_schema_violations_lazy(lazy: pl.LazyFrame, schema: pl.Schema) -> list[str]:
    required = {
        "venue",
        "symbol",
        "market_type",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "ingest_ts",
    }
    violations = [f"缺少字段：{column}" for column in sorted(required - set(schema.names()))]
    if violations:
        return violations
    try:
        invalid_rows = int(
            lazy.filter(
                (pl.col("open").cast(pl.Float64, strict=False) <= 0)
                | (pl.col("close").cast(pl.Float64, strict=False) <= 0)
                | (
                    pl.col("high").cast(pl.Float64, strict=False)
                    < pl.col("low").cast(pl.Float64, strict=False)
                )
                | (pl.col("volume").cast(pl.Float64, strict=False) < 0)
            )
            .select(pl.len().alias("rows"))
            .collect()
            .item(0, "rows")
            or 0
        )
    except Exception:
        invalid_rows = 0
    if invalid_rows:
        violations.append(f"OHLCV 无效行数：{invalid_rows}")
    return violations


def missing_bar_table(market: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    if market.is_empty() or not {"symbol", "timeframe", "ts"}.issubset(market.columns):
        return pl.DataFrame(rows)
    for group in market.group_by(["symbol", "timeframe"], maintain_order=True):
        keys, group_df = group
        symbol, timeframe = keys
        step = _timeframe_delta(str(timeframe))
        if step is None or group_df.height < 2:
            continue
        timestamps = sorted(set(group_df["ts"].to_list()))
        expected = int((timestamps[-1] - timestamps[0]) / step) + 1
        missing = max(expected - len(timestamps), 0)
        if missing:
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "expected_bars": expected,
                    "actual_bars": len(timestamps),
                    "missing_bars": missing,
                }
            )
    return pl.DataFrame(rows)


def _missing_bar_table_lazy(lazy: pl.LazyFrame, schema: pl.Schema) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    if not {"symbol", "timeframe", "ts"}.issubset(schema):
        return pl.DataFrame(rows)
    try:
        stats = (
            lazy.with_columns(_lazy_timestamp_expr(schema, "ts").alias("__qlab_ts"))
            .group_by(["symbol", "timeframe"])
            .agg(
                pl.col("__qlab_ts").min().alias("min_ts"),
                pl.col("__qlab_ts").max().alias("max_ts"),
                pl.col("__qlab_ts").n_unique().alias("actual_bars"),
            )
            .collect()
        )
    except Exception:
        return pl.DataFrame(rows)
    for row in stats.to_dicts():
        symbol = row.get("symbol")
        timeframe = str(row.get("timeframe") or "")
        step = _timeframe_delta(timeframe)
        start = _coerce_timestamp(row.get("min_ts"))
        end = _coerce_timestamp(row.get("max_ts"))
        if step is None or start is None or end is None:
            continue
        actual = int(row.get("actual_bars") or 0)
        if actual < 2:
            continue
        expected = int((end - start) / step) + 1
        missing = max(expected - actual, 0)
        if missing:
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "expected_bars": expected,
                    "actual_bars": actual,
                    "missing_bars": missing,
                }
            )
    return pl.DataFrame(rows)


def orderbook_spread_table(books: pl.DataFrame) -> pl.DataFrame:
    if books.is_empty() or not {"asks_json", "bids_json"}.issubset(books.columns):
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in books.to_dicts():
        ask = _best_price(row.get("asks_json"))
        bid = _best_price(row.get("bids_json"))
        if ask is None or bid is None or ask <= bid:
            continue
        mid = (ask + bid) / 2.0
        rows.append(
            {
                "symbol": row.get("symbol"),
                "channel": row.get("channel"),
                "ts": row.get("ts"),
                "spread_bps": ((ask - bid) / mid) * 10_000,
            }
        )
    if not rows:
        return pl.DataFrame()
    spread = pl.DataFrame(rows)
    if "ts" not in spread.columns:
        return spread.sort("symbol") if "symbol" in spread.columns else spread
    return (
        spread.with_columns(
            pl.col("ts")
            .map_elements(_coerce_timestamp, return_dtype=pl.Datetime(time_zone="UTC"))
            .alias("ts")
        )
        .sort("ts")
        .group_by(["symbol", "channel"], maintain_order=True)
        .tail(1)
        .sort(["symbol", "channel"])
    )


def trade_activity_table(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty() or "symbol" not in trades.columns:
        return pl.DataFrame()
    aggregations = [pl.len().alias("trade_count")]
    if "size" in trades.columns:
        aggregations.append(pl.col("size").cast(pl.Float64, strict=False).sum().alias("size_sum"))
    if "ts" in trades.columns:
        aggregations.append(pl.col("ts").max().alias("latest_trade_ts"))
    return trades.group_by("symbol").agg(aggregations).sort("symbol")


def redact_frame(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    expressions = []
    for column in df.columns:
        if _sensitive_column(column):
            expressions.append(pl.lit("[REDACTED]").alias(column))
        else:
            expressions.append(
                pl.col(column).map_elements(redact_text, return_dtype=df.schema[column])
            )
    try:
        return df.with_columns(expressions)
    except Exception:
        return df


def redact_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if _sensitive_text(lowered):
        return "[REDACTED]"
    return value


def okx_ws_latest(lake_root: str | Path) -> datetime | None:
    return _snapshot_latest_timestamp(_dataset_snapshot(lake_root, "okx_public_ws"))


def _latest_collector_health(health: pl.DataFrame) -> dict[str, Any] | None:
    if health.is_empty():
        return None
    sort_column = "updated_at" if "updated_at" in health.columns else health.columns[0]
    return health.sort(sort_column).tail(1).to_dicts()[0]


def _latest_ws_subscription(health: pl.DataFrame) -> tuple[set[str], set[str]] | None:
    latest = _latest_collector_health(health)
    if not latest:
        return None
    symbols = {
        normalize_symbol(value) for value in _json_string_list(latest.get("subscribed_symbols"))
    }
    channels = {
        str(value).strip() for value in _json_string_list(latest.get("subscribed_channels"))
    }
    if not symbols and not channels:
        return None
    return symbols, channels


def _json_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return _json_string_list(parsed)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def readonly_private_status(lake_root: str | Path) -> str:
    fills = _dataset_snapshot(lake_root, "okx_private_readonly_fills")
    bills = _dataset_snapshot(lake_root, "okx_private_readonly_bills")
    if fills.rows == 0 and bills.rows == 0:
        return "NOT_CONFIGURED"
    return "OK"


def _fallback_rows(audits: pl.DataFrame) -> pl.DataFrame:
    if audits.is_empty():
        return pl.DataFrame()
    text_columns = [column for column in audits.columns if audits.schema[column] == pl.String]
    if not text_columns:
        return pl.DataFrame()
    predicate = None
    for column in text_columns:
        expression = pl.col(column).str.contains("fallback", literal=True)
        predicate = expression if predicate is None else predicate | expression
    return redact_frame(audits.filter(predicate)) if predicate is not None else pl.DataFrame()


def _latest_features_table(features: pl.DataFrame) -> pl.DataFrame:
    required = {"symbol", "ts", "feature_name", "value"}
    if features.is_empty() or not required.issubset(features.columns):
        return pl.DataFrame()
    normalized = _normalize_optional_time(features, "ts")
    latest_ts = normalized.group_by("symbol").agg(pl.col("ts").max().alias("ts"))
    return normalized.join(latest_ts, on=["symbol", "ts"], how="inner").sort(
        ["symbol", "feature_name"]
    )


def _top_feature_anomalies(anomalies: pl.DataFrame) -> pl.DataFrame:
    if anomalies.is_empty() or "anomaly_type" not in anomalies.columns:
        return pl.DataFrame()
    count_expr = (
        pl.col("anomaly_count").sum().alias("anomaly_count")
        if "anomaly_count" in anomalies.columns
        else pl.len().alias("anomaly_count")
    )
    return (
        anomalies.group_by(["anomaly_type", "severity"])
        .agg(count_expr)
        .sort("anomaly_count", descending=True)
        .head(20)
    )


def _stale_dataset_rows(lake_root: str | Path) -> pl.DataFrame:
    root = Path(lake_root)
    cache_key = (
        "stale_dataset_rows",
        str(root.resolve()),
        _lake_file_index_signature(root),
    )
    cached = _web_cache_get(cache_key, event="stale_dataset_rows")
    if cached is not None:
        return cached

    rows = []
    v5_telemetry_is_current = _v5_telemetry_is_current(lake_root)
    okx_readonly_private_is_current = _okx_readonly_private_is_current(lake_root)
    expanded_universe_automation_is_active = _expanded_universe_automation_is_active(lake_root)
    closed_research_keys = _closed_research_keys(lake_root)
    for name in sorted(DATASET_PATHS):
        path = dataset_path_for(lake_root, name)
        snapshot = _dataset_snapshot(lake_root, name)
        freshness = snapshot.freshness
        status = freshness["freshness_status"]
        if snapshot.rows == 0:
            status = _empty_dataset_status(name)
        if (
            name == "expanded_crypto_universe_shadow"
            and snapshot.rows == 0
            and expanded_universe_automation_is_active
        ):
            status = "expanded_universe_automation_active"
        if _dataset_belongs_to_closed_research(name, closed_research_keys):
            status = "closed_research_snapshot"
        if (
            name in EVENT_DRIVEN_V5_DATASET_STATUSES
            and status == "stale"
            and v5_telemetry_is_current
        ):
            status = EVENT_DRIVEN_V5_DATASET_STATUSES[name]
        if name in V5_PAPER_TELEMETRY_DATASETS and status == "stale" and v5_telemetry_is_current:
            status = "waiting_for_v5_paper_telemetry"
        if (
            name in EVENT_DRIVEN_OKX_READONLY_DATASET_STATUSES
            and status == "stale"
            and okx_readonly_private_is_current
        ):
            status = EVENT_DRIVEN_OKX_READONLY_DATASET_STATUSES[name]
        if name in HISTORICAL_RESEARCH_DATASETS and status == "stale":
            status = "historical_research_snapshot"
        status = _optional_stale_status_from_registry(
            name,
            status,
            freshness_seconds=freshness["freshness_seconds"],
        )
        status = _derived_rollup_status(lake_root, name, snapshot, status)
        status = _derived_latest_status(lake_root, name, snapshot, status)
        if _should_show_stale_dataset_row(snapshot, status):
            status_text = str(status)
            rows.append(
                {
                    "takeaway": _stale_dataset_takeaway(name, status_text, snapshot),
                    "severity": _stale_dataset_severity(name, status_text, snapshot),
                    "next_action": _stale_dataset_next_action(name, status_text, snapshot),
                    "dataset": name,
                    "rows": snapshot.rows,
                    "status": status_text,
                    "path": str(path),
                    "freshness_seconds": freshness["freshness_seconds"],
                    "timestamp_column": freshness["timestamp_column"] or "",
                    "latest_timestamp": freshness["latest_timestamp"] or "",
                    "warning": snapshot.warning or "",
                }
            )
    if not rows:
        return _web_cache_set(cache_key, pl.DataFrame(schema=STALE_DATASET_SCHEMA))
    return _web_cache_set(
        cache_key,
        pl.DataFrame(rows, schema=STALE_DATASET_SCHEMA, orient="row"),
    )


def _closed_research_keys(lake_root: str | Path) -> set[str]:
    try:
        frame = read_parquet_dataset(dataset_path_for(lake_root, "research_portfolio_status"))
    except Exception:
        return set()
    return research_portfolio_closed_keys(frame)


def _dataset_belongs_to_closed_research(dataset_name: str, closed_keys: set[str]) -> bool:
    if not closed_keys:
        return False
    keys = RESEARCH_DIAGNOSTIC_DATASET_KEYS.get(dataset_name)
    if not keys:
        return False
    return bool(set(keys).intersection(closed_keys))


def _expanded_universe_automation_is_active(lake_root: str | Path) -> bool:
    for dataset_name in (
        "expanded_universe_candidate",
        "expanded_universe_quality",
        "expanded_universe_candidate_event",
        "expanded_universe_watchlist",
    ):
        snapshot = _dataset_snapshot(lake_root, dataset_name)
        status = str(snapshot.freshness.get("freshness_status") or "")
        if snapshot.rows > 0 and status in {"fresh", "delayed"} and not snapshot.warning:
            return True
    return False


def _stale_dataset_takeaway(dataset_name: str, status: str, snapshot: DatasetSnapshot) -> str:
    dataset_label = _dataset_display_name(dataset_name)
    status_label = str(display_value(status))
    if snapshot.warning:
        return f"{dataset_label} 读取异常：{snapshot.warning}"
    if status == DERIVED_ROLLUP_PENDING_STATUS:
        source_label = _dataset_display_name(DERIVED_ROLLUP_SOURCE_DATASETS[dataset_name])
        return f"{dataset_label} 暂无派生数据：{source_label} 已存在，需运行 1m rollup 构建"
    if snapshot.rows == 0:
        return f"{dataset_label} 暂无数据：{status_label}"
    if status == "stale":
        return f"{dataset_label} 已过期：下游页面可能看到旧结果"
    if status == "missing":
        return f"{dataset_label} 缺失：对应链路尚未产出"
    if status == "unknown":
        return f"{dataset_label} 时间不可判定：缺少可用时间列"
    return f"{dataset_label} 需要关注：{status_label}"


def _stale_dataset_severity(
    dataset_name: str,
    status: str,
    snapshot: DatasetSnapshot,
) -> str:
    if snapshot.warning:
        return "CRITICAL"
    if status == DERIVED_ROLLUP_PENDING_STATUS:
        return "WARNING"
    if dataset_name in {"market_bar", "cost_bucket_daily", "risk_permission"}:
        return "CRITICAL"
    if status in {"missing", "unknown"}:
        return "CRITICAL"
    if dataset_name in {"feature_value", "alpha_evidence", "gate_decision"}:
        return "WARNING"
    return "WARNING"


def _stale_dataset_next_action(
    dataset_name: str,
    status: str,
    snapshot: DatasetSnapshot,
) -> str:
    if snapshot.warning:
        return "先修复数据集读取异常，再重新打开页面验证"
    if status == DERIVED_ROLLUP_PENDING_STATUS or dataset_name in DERIVED_ROLLUP_SOURCE_DATASETS:
        return (
            "运行 qlab build-market-data-rollups --lake-root /var/lib/quant-lab/lake "
            "--lookback-hours 24 --apply 刷新 1m 派生表"
        )
    if dataset_name == "market_bar":
        return "检查 OKX REST/WS 采集任务并补齐 market_bar"
    if dataset_name == "feature_value":
        return "运行 qlab publish-features 重新生成核心特征"
    if dataset_name == "alpha_evidence":
        return "运行 qlab build-alpha-evidence 生成研究证据"
    if dataset_name in {"cost_bucket_daily", "cost_health_daily"}:
        return "运行 qlab calibrate-costs 刷新成本模型"
    if dataset_name == "risk_permission":
        return "运行 qlab publish-risk-permission 刷新风险许可"
    if dataset_name in {"gate_decision", "strategy_health_daily", "v5_gate_compliance_daily"}:
        return "先同步 V5 telemetry，再刷新 gate/risk/report"
    if dataset_name.startswith("v5_") or dataset_name.startswith("strategy_"):
        return "检查 V5 telemetry sync 与对应 research refresh 任务"
    if status == "stale":
        return "检查对应 systemd timer 是否运行，并按需手动刷新"
    return "确认该数据集是否仍需要；需要则运行对应构建命令"


def _dataset_display_name(dataset_name: str) -> str:
    labels = {
        "market_bar": "行情 K 线",
        "feature_value": "特征值",
        "alpha_evidence": "Alpha 证据",
        "gate_decision": "门控决策",
        "risk_permission": "风险许可",
        "cost_bucket_daily": "成本桶",
        "cost_health_daily": "成本健康",
        "strategy_health_daily": "V5 策略健康",
        "v5_gate_compliance_daily": "V5 门控合规",
        "v5_candidate_event": "V5 候选事件",
        "v5_candidate_label": "V5 候选标签",
        "v5_trade_opportunity_funnel": "V5 交易机会漏斗",
        "v5_cost_probe_p3_preflight": "V5 成本探针 P3 预检",
        "v5_cost_probe_live_execution_status": "V5 成本探针执行状态",
        "v5_cost_probe_order_event": "V5 成本探针订单事件",
        "v5_cost_probe_roundtrip_event": "V5 成本探针回合事件",
        "strategy_opportunity_advisory": "策略机会建议",
        "research_portfolio_status": "研究组合裁剪",
        "strategy_evidence": "策略证据",
        "strategy_evidence_sample": "策略证据样本",
        "paper_strategy_runs": "纸面策略运行",
        "paper_strategy_daily": "纸面策略日汇总",
        "paper_strategy_registry": "纸面策略注册表",
        "paper_strategy_migration_audit": "旧纸面策略迁移审计",
        "paper_strategy_promotion_gate": "纸面策略晋级闸门",
        "v5_paper_strategy_proposal_ack": "V5 纸面策略 ACK",
        "v5_paper_strategy_registry": "V5 通用纸面策略注册表",
        "v5_paper_strategy_state": "V5 通用纸面策略状态",
        "v5_paper_strategy_signal": "V5 通用纸面策略信号",
        "v5_paper_strategy_quote_coverage": "V5 纸面策略盘口覆盖",
        "v5_paper_strategy_cost_evidence": "V5 纸面策略成本证据",
        "v5_paper_strategy_error": "V5 纸面策略错误",
        "v5_paper_strategy_restart_recovery": "V5 纸面策略重启恢复",
        "v5_quant_lab_contract_status": "V5 与中台契约状态",
        "v5_fill_bill_cost_reconciliation": "V5 成交与账单成本对账",
        "paper_slippage_coverage": "纸面滑点覆盖",
        "trade_print": "OKX 成交流",
        "orderbook_snapshot": "OKX 订单簿",
        "trade_activity_1m": "成交活跃度 1m 聚合",
        "orderbook_spread_1m": "订单簿价差 1m 聚合",
        "okx_public_ws": "OKX WebSocket 原始数据",
    }
    if dataset_name in labels:
        return labels[dataset_name]
    return dataset_name.replace("_", " ")


def _empty_dataset_status(dataset_name: str) -> str:
    if dataset_name == "alpha_evidence":
        return "研究证据尚未生成"
    if dataset_name == "feature_value":
        return "特征尚未发布"
    if dataset_name == "factor_strategy_bridge_candidates":
        return "export_derived_optional"
    if dataset_name == "cost_bootstrap_readiness":
        return "export_derived_optional"
    if dataset_name in {"cost_probe_fill_bill_match", "cost_probe_cost_disagreement"}:
        return "cost_probe_optional_audit"
    event_driven_status = EVENT_DRIVEN_V5_DATASET_STATUSES.get(dataset_name)
    if event_driven_status:
        return event_driven_status
    if dataset_name == "decision_audit":
        return "legacy_optional"
    if dataset_name == "v5_bundle_manifest":
        return "waiting_for_v5_bundle_manifest"
    if dataset_name in V5_PAPER_TELEMETRY_DATASETS:
        return "waiting_for_v5_paper_telemetry"
    if dataset_name in RESEARCH_DIAGNOSTIC_DATASET_KEYS:
        return "waiting_for_v5_paper_telemetry"
    if dataset_name in ENTRY_QUALITY_DATASETS:
        return "entry_quality_optional"
    if dataset_name in {
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
    }:
        return "trade_level_optional"
    spec = get_dataset_spec(dataset_name)
    if spec is not None and not spec.required and int(spec.min_rows or 0) == 0:
        return "registry_optional_empty"
    return "missing"


def _optional_stale_status_from_registry(
    dataset_name: str,
    status: str,
    *,
    freshness_seconds: int | None = None,
) -> str:
    if status != "stale":
        return status
    spec = get_dataset_spec(dataset_name)
    if spec is None:
        return status
    if spec.freshness_seconds is None and "freshness" not in spec.quality_rules:
        if dataset_name in ENTRY_QUALITY_DATASETS:
            return "entry_quality_optional"
        return "registry_freshness_not_required"
    if (
        spec.freshness_seconds is not None
        and freshness_seconds is not None
        and freshness_seconds <= spec.freshness_seconds
    ):
        return "registry_freshness_within_sla"
    return status


def _derived_rollup_status(
    lake_root: str | Path,
    dataset_name: str,
    snapshot: DatasetSnapshot,
    status: str,
) -> str:
    source_dataset = DERIVED_ROLLUP_SOURCE_DATASETS.get(dataset_name)
    if not source_dataset:
        return status
    if snapshot.rows > 0:
        return status
    if status not in {"missing", "unknown"}:
        return status
    source_snapshot = _dataset_snapshot(lake_root, source_dataset)
    if (source_snapshot.rows > 0 and not source_snapshot.warning) or _dataset_has_parquet_files(
        dataset_path_for(lake_root, source_dataset)
    ):
        return DERIVED_ROLLUP_PENDING_STATUS
    return status


def _derived_latest_status(
    lake_root: str | Path,
    dataset_name: str,
    snapshot: DatasetSnapshot,
    status: str,
) -> str:
    source_dataset = DERIVED_LATEST_SOURCE_DATASETS.get(dataset_name)
    if not source_dataset or status != "stale" or snapshot.warning:
        return status
    source_snapshot = _dataset_snapshot(lake_root, source_dataset)
    source_status = str(source_snapshot.freshness.get("freshness_status") or "")
    if (
        source_snapshot.rows > 0
        and not source_snapshot.warning
        and source_status in {"fresh", "delayed"}
    ):
        return DERIVED_LATEST_SOURCE_CURRENT_STATUS
    return status


def _dataset_has_parquet_files(path: Path) -> bool:
    if path.is_file():
        return path.suffix == ".parquet" and not _is_internal_lake_path(path)
    if not path.exists() or not path.is_dir():
        return False
    try:
        return any(
            candidate.is_file() and not _is_internal_lake_path(candidate)
            for candidate in path.glob("*.parquet")
        )
    except OSError:
        return False


def _should_show_stale_dataset_row(snapshot: DatasetSnapshot, status: str) -> bool:
    if snapshot.warning:
        return True
    if status in OPTIONAL_EMPTY_DATASET_STATUSES or status in EVENT_DRIVEN_OK_STATUSES:
        return False
    if status in {"missing", "unknown", "stale"}:
        return True
    return snapshot.rows == 0


def _v5_telemetry_is_current(lake_root: str | Path) -> bool:
    snapshot = _dataset_snapshot(lake_root, "strategy_health_daily")
    return str(snapshot.freshness.get("freshness_status") or "") in {"fresh", "delayed"}


def _okx_readonly_private_is_current(lake_root: str | Path) -> bool:
    for job_name in ("okx-backfill-readonly", "okx-fetch-bills"):
        if _recent_successful_job(lake_root, job_name):
            return True
    for dataset_name in ("okx_private_readonly_fills", "okx_private_readonly_bills"):
        snapshot = _dataset_snapshot(lake_root, dataset_name)
        if (
            snapshot.rows > 0
            and not snapshot.warning
            and str(snapshot.freshness.get("freshness_status") or "") in {"fresh", "delayed"}
        ):
            return True
    return False


def _recent_successful_job(
    lake_root: str | Path,
    job_name: str,
    *,
    max_age_seconds: int = 24 * 60 * 60,
) -> bool:
    try:
        frame = read_parquet_dataset(dataset_path_for(lake_root, "job_run_history"))
    except Exception:
        return False
    if frame.is_empty() or not {"job_name", "status", "finished_at"}.issubset(frame.columns):
        return False
    try:
        filtered = frame.filter(
            (pl.col("job_name").cast(pl.Utf8) == job_name)
            & (pl.col("status").cast(pl.Utf8).str.to_lowercase() == "succeeded")
        )
    except Exception:
        return False
    if filtered.is_empty():
        return False
    latest, _column = latest_dataset_timestamp("job_run_history", filtered)
    if latest is None:
        return False
    age_seconds = max(int((datetime.now(UTC) - latest).total_seconds()), 0)
    return age_seconds <= max_age_seconds


def _canonical_dataset_name(dataset_name: str) -> str:
    if dataset_name in DATASET_PATHS:
        return dataset_name
    normalized = dataset_name.replace("\\", "/")
    for name, relative_path in DATASET_PATHS.items():
        if str(relative_path).replace("\\", "/") == normalized:
            return name
    return dataset_name


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            if len(text) == 10:
                parsed_date = date.fromisoformat(text)
                return datetime.combine(parsed_date, time.min, tzinfo=UTC)
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _normalize_market_frame(df: pl.DataFrame) -> pl.DataFrame:
    normalized = _normalize_optional_time(df, "ts")
    normalized = _normalize_optional_time(normalized, "ingest_ts")
    return normalized


def _normalize_optional_time(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if df.is_empty() or column not in df.columns:
        return df
    if df.schema[column] == pl.String:
        return df.with_columns(pl.col(column).str.to_datetime(time_zone="UTC", strict=False))
    try:
        return df.with_columns(pl.col(column).cast(pl.Datetime(time_zone="UTC")).alias(column))
    except Exception:
        return df


def _max_datetime(df: pl.DataFrame, column: str) -> datetime | None:
    if df.is_empty() or column not in df.columns:
        return None
    value = df.select(pl.col(column).max()).item()
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return None


def _missing_ratio(missing_bars: pl.DataFrame, actual_rows: int) -> float:
    if missing_bars.is_empty() or "missing_bars" not in missing_bars.columns:
        return 0.0
    missing = float(missing_bars["missing_bars"].sum())
    denominator = actual_rows + missing
    return missing / denominator if denominator else 0.0


def _lag_label(timestamp: datetime | None) -> str:
    if timestamp is None:
        return "未知"
    lag = datetime.now(UTC) - timestamp
    if lag < timedelta(hours=2):
        return "新鲜"
    if lag < timedelta(hours=24):
        return "延迟"
    return "过期"


def _timeframe_delta(timeframe: str) -> timedelta | None:
    match timeframe:
        case "1m":
            return timedelta(minutes=1)
        case "3m":
            return timedelta(minutes=3)
        case "5m":
            return timedelta(minutes=5)
        case "15m":
            return timedelta(minutes=15)
        case "30m":
            return timedelta(minutes=30)
        case "1H":
            return timedelta(hours=1)
        case "2H":
            return timedelta(hours=2)
        case "4H":
            return timedelta(hours=4)
        case "6H":
            return timedelta(hours=6)
        case "12H":
            return timedelta(hours=12)
        case "1D" | "1Dutc":
            return timedelta(days=1)
        case _:
            return None


def _best_price(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    try:
        values = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        return float(values[0][0]) if values else None
    except Exception:
        return None


def _read_json_from_zip(path: Path, member: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text_from_zip(path: Path, member: str) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as handle:
                return handle.read().decode("utf-8")
    except Exception:
        return ""


def _sensitive_column(column: str) -> bool:
    lowered = column.lower()
    return (
        "secret" in lowered
        or "passphrase" in lowered
        or "credential" in lowered
        or "token" in lowered
        or ("key" in lowered and ("api" in lowered or "access" in lowered))
    )


def _sensitive_text(lowered: str) -> bool:
    return (
        "ok-access-" in lowered
        or "passphrase=" in lowered
        or "secret=" in lowered
        or "credential=" in lowered
    )
