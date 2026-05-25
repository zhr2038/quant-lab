from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import invalid_parquet_files, read_parquet_dataset
from quant_lab.symbols import normalize_symbol

DATASET_PATHS = {
    "market_bar": Path("silver") / "market_bar",
    "feature_value": Path("gold") / "feature_value",
    "feature_coverage_daily": Path("gold") / "feature_coverage_daily",
    "feature_anomaly_daily": Path("gold") / "feature_anomaly_daily",
    "cost_bucket_daily": Path("gold") / "cost_bucket_daily",
    "cost_health_daily": Path("gold") / "cost_health_daily",
    "alpha_evidence": Path("gold") / "alpha_evidence",
    "alpha_discovery_board": Path("gold") / "alpha_discovery_board",
    "strategy_evidence": Path("gold") / "strategy_evidence",
    "strategy_evidence_sample": Path("gold") / "strategy_evidence_sample",
    "strategy_evidence_quality": Path("gold") / "strategy_evidence_quality",
    "research_portfolio_status": Path("gold") / "research_portfolio_status",
    "strategy_opportunity_advisory": Path("gold") / "strategy_opportunity_advisory",
    "v5_missed_opportunity_audit": Path("gold") / "v5_missed_opportunity_audit",
    "v5_risk_on_multi_buy_shadow": Path("gold") / "v5_risk_on_multi_buy_shadow",
    "risk_on_multi_buy_shadow": Path("gold") / "risk_on_multi_buy_shadow",
    "alpha_factory_candidate": Path("gold") / "alpha_factory_candidate",
    "alpha_factory_result": Path("gold") / "alpha_factory_result",
    "alpha_factory_promotion_queue": Path("gold") / "alpha_factory_promotion_queue",
    "second_stage_alpha_factory_sample": Path("gold")
    / "second_stage_alpha_factory_sample",
    "second_stage_alpha_factory_summary": Path("gold")
    / "second_stage_alpha_factory_summary",
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
    "paper_slippage_coverage": Path("gold") / "paper_slippage_coverage",
    "sol_protect_paper_loss_attribution": Path("gold")
    / "sol_protect_paper_loss_attribution",
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
    "btc_probe_exit_policy_review": Path("gold") / "btc_probe_exit_policy_review",
    "btc_probe_exit_policy_summary": Path("gold") / "btc_probe_exit_policy_summary",
    "bnb_swing_exit_policy_review": Path("gold") / "bnb_swing_exit_policy_review",
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
    "v5_quant_lab_compliance": Path("silver") / "v5_quant_lab_compliance",
    "v5_quant_lab_cost_usage": Path("silver") / "v5_quant_lab_cost_usage",
    "v5_quant_lab_fallback": Path("silver") / "v5_quant_lab_fallback",
    "v5_paper_strategy_run": Path("silver") / "v5_paper_strategy_run",
    "v5_paper_strategy_daily": Path("silver") / "v5_paper_strategy_daily",
    "v5_paper_slippage_coverage": Path("silver") / "v5_paper_slippage_coverage",
    "v5_candidate_event": Path("silver") / "v5_candidate_event",
    "v5_candidate_label": Path("gold") / "v5_candidate_label",
    "v5_candidate_quality_daily": Path("gold") / "v5_candidate_quality_daily",
    "v5_candidate_outcome_summary": Path("gold") / "v5_candidate_outcome_summary",
    "v5_decision_audit": Path("silver") / "v5_decision_audit",
    "v5_trade_event": Path("silver") / "v5_trade_event",
    "trade_print": Path("silver") / "trade_print",
    "orderbook_snapshot": Path("silver") / "orderbook_snapshot",
    "okx_public_ws": Path("bronze") / "okx_public_ws",
    "okx_private_readonly_fills": Path("bronze") / "okx_private_readonly" / "fills_history",
    "okx_private_readonly_bills": Path("bronze") / "okx_private_readonly" / "bills",
    "decision_audit": Path("silver") / "decision_audit",
}
V5_PAPER_TELEMETRY_DATASETS = {
    "v5_paper_strategy_run",
    "v5_paper_strategy_daily",
    "v5_paper_slippage_coverage",
}
OPTIONAL_EMPTY_DATASET_STATUSES = {
    "legacy_optional",
    "waiting_for_v5_paper_telemetry",
    "entry_quality_optional",
    "historical_research_snapshot",
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
    "btc_probe_exit_policy_review",
    "btc_probe_exit_policy_summary",
    "bnb_swing_exit_policy_review",
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
    "v5_entry_quality_history_late_entry_chase_threshold_sensitivity",
    "v5_entry_quality_history_pullback_by_symbol",
    "v5_entry_quality_history_pullback_by_regime",
    "v5_entry_quality_history_pullback_by_horizon",
    "v5_entry_quality_history_anti_leakage_check",
    "v5_entry_quality_history_metrics",
}
EVENT_DRIVEN_OK_STATUSES = {"event_driven_no_recent_trade"}

DATASET_TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    "market_bar": ("ts",),
    "okx_public_ws": ("received_at",),
    "trade_print": ("ts", "ingest_ts"),
    "orderbook_snapshot": ("ts", "ingest_ts"),
    "cost_bucket_daily": ("created_at", "day"),
    "cost_health_daily": ("created_at", "day"),
    "gate_decision": ("created_at",),
    "risk_permission": ("as_of_ts", "created_at"),
    "api_request_metrics": ("request_ts",),
    "job_run_history": ("finished_at", "started_at"),
    "lake_file_health_daily": ("created_at", "day"),
    "strategy_health_daily": ("latest_bundle_ts", "date"),
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
    "alpha_evidence": ("created_at", "end_ts"),
    "alpha_discovery_board": ("created_at", "end_ts", "start_ts", "as_of_date"),
    "strategy_evidence": ("created_at", "end_ts", "as_of_date"),
    "strategy_evidence_sample": ("created_at", "ts_utc"),
    "strategy_evidence_quality": ("created_at", "as_of_date"),
    "research_portfolio_status": ("created_at", "last_review_date"),
    "strategy_opportunity_advisory": ("as_of_ts", "created_at"),
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
    "expanded_universe_candidate_label": ("label_ts", "generated_at"),
    "expanded_universe_promotion_queue": ("generated_at", "as_of_date"),
    "expanded_universe_candidate_maturity": ("generated_at", "as_of_date"),
    "expanded_universe_watchlist": ("generated_at", "as_of_date"),
    "expanded_crypto_universe_shadow": ("generated_at", "as_of_date"),
    "symbol_quality_score": ("generated_at", "as_of_date"),
    "expanded_crypto_candidate_outcomes_by_symbol": ("generated_at", "as_of_date"),
    "expanded_crypto_recommendations": ("generated_at", "as_of_date"),
    "paper_strategy_runs": ("created_at", "as_of_date"),
    "paper_strategy_daily": ("created_at", "as_of_date"),
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
    "v5_quant_lab_compliance": ("ingest_ts", "bundle_ts"),
    "v5_quant_lab_cost_usage": ("ingest_ts", "bundle_ts"),
    "v5_quant_lab_fallback": ("ingest_ts", "bundle_ts"),
    "v5_paper_strategy_run": ("created_at", "as_of_date", "ingest_ts", "bundle_ts"),
    "v5_paper_strategy_daily": ("created_at", "as_of_date", "ingest_ts", "bundle_ts"),
    "v5_paper_slippage_coverage": ("created_at", "as_of_date", "ingest_ts", "bundle_ts"),
    "v5_candidate_event": ("ts_utc", "ingest_ts", "bundle_ts"),
    "v5_candidate_label": ("label_ts", "decision_ts", "ts_utc", "created_at"),
    "v5_candidate_quality_daily": ("created_at", "date"),
    "v5_candidate_outcome_summary": ("created_at", "date"),
    "v5_decision_audit": ("ingest_ts", "bundle_ts"),
    "v5_trade_event": ("ts_utc", "ts", "ingest_ts", "bundle_ts"),
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
    "--lake-root {lake_root} --history --limit 100"
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
    "--flush-interval-seconds 600 --flush-max-messages 50000"
)
EXPERT_PACK_MISSING_MESSAGE = "qlab export-daily 尚未实现或尚未运行。"
BOOTSTRAP_GOLD_HEALTH_COMMAND = (
    "qlab bootstrap-gold-health --lake-root {lake_root} "
    "--strategy v5 --version bootstrap --day auto"
)
BOOTSTRAP_PLACEHOLDER_WARNING = "bootstrap 保守占位数据，不是 live-ready 证据"
DISPLAY_LIMIT = 500
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
WEB_FULL_VALIDATION_FILE_LIMIT = 512
WEB_RECENT_FILE_LIMITS = {
    "trade_print": 384,
    "orderbook_snapshot": 384,
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
    if dataset_name in WEB_RECENT_FILE_SAMPLE_DATASETS:
        files, warning = _recent_valid_parquet_files(dataset_path, dataset_name)
    else:
        invalid_files = invalid_parquet_files(dataset_path)
        files = _valid_parquet_files(dataset_path, invalid_files=invalid_files)
        warning = _invalid_parquet_warning(dataset_name, invalid_files)
    if not files:
        return pl.DataFrame(), warning

    row_limit = limit or WEB_HEAVY_DATASET_LIMITS.get(dataset_name, DISPLAY_LIMIT)
    hours = lookback_hours or WEB_RECENT_LOOKBACK_HOURS.get(dataset_name, 6)
    try:
        lazy = _scan_parquet_files(files)
        frame = _collect_recent_lazy_frame(
            lazy,
            dataset_name,
            limit=row_limit,
            lookback_hours=hours,
        )
        return frame, warning
    except Exception as exc:
        return pl.DataFrame(), f"{dataset_name} 抽样读取失败：{exc}"


def _read_web_display_dataset_with_warning(
    lake_root: str | Path,
    dataset_name: str,
) -> tuple[pl.DataFrame, str | None]:
    sampled_datasets = (
        WEB_RESEARCH_SAMPLE_DATASETS
        | WEB_FEATURE_SAMPLE_DATASETS
        | WEB_AUDIT_SAMPLE_DATASETS
        | WEB_COST_SAMPLE_DATASETS
        | WEB_ADVISORY_SAMPLE_DATASETS
    )
    if dataset_name in sampled_datasets:
        return read_recent_dataset_with_warning(lake_root, dataset_name)
    return read_dataset_with_warning(lake_root, dataset_name)


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
    latest, column = latest_dataset_timestamp(dataset_name, df)
    if latest is None:
        return {
            "freshness_seconds": None,
            "freshness_status": "unknown",
            "timestamp_column": column,
            "latest_timestamp": None,
        }
    reference_time = now.astimezone(UTC) if now else datetime.now(UTC)
    seconds = max(int((reference_time - latest).total_seconds()), 0)
    if seconds <= 6 * 60 * 60:
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
    }


def latest_dataset_timestamp(
    dataset_name: str,
    df: pl.DataFrame,
) -> tuple[datetime | None, str | None]:
    if df.is_empty():
        return None, None
    columns = DATASET_TIMESTAMP_COLUMNS.get(dataset_name, ("ts", "created_at", "ingest_ts"))
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
    invalid_files = invalid_parquet_files(path)
    files = _valid_parquet_files(path, invalid_files=invalid_files)
    warning = _invalid_parquet_warning(dataset_name, invalid_files)
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
            timestamp_columns=timestamp_columns,
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
    files, warning = _metadata_snapshot_files(path, dataset_name)
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
    candidates = _recent_parquet_file_candidates(path)
    max_files = WEB_RECENT_FILE_LIMITS.get(dataset_name, WEB_FULL_VALIDATION_FILE_LIMIT)
    selected = candidates[-max_files:] if len(candidates) > max_files else candidates
    invalid_files = [
        file_path for file_path in selected if not _is_valid_parquet_file_path(file_path)
    ]
    invalid = set(invalid_files)
    valid_files = [file_path for file_path in selected if file_path not in invalid]
    return valid_files, _invalid_parquet_warning(dataset_name, invalid_files)


def _recent_parquet_file_candidates(path: Path) -> list[Path]:
    candidates = _parquet_file_candidates(path)
    batch_files = [file_path for file_path in candidates if file_path.name.startswith("batch_")]
    if batch_files:
        candidates = batch_files
    try:
        return sorted(candidates, key=lambda file_path: (file_path.stat().st_mtime, file_path.name))
    except OSError:
        return sorted(candidates)


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
            for column in DATASET_TIMESTAMP_COLUMNS.get(dataset_name, ("ts", "created_at"))
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
    seconds = max(int((reference_time - latest).total_seconds()), 0)
    if seconds <= 6 * 60 * 60:
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
    }


def _valid_parquet_files(path: Path, *, invalid_files: list[Path] | None = None) -> list[Path]:
    candidates = _parquet_file_candidates(path)
    invalid = set(invalid_files if invalid_files is not None else invalid_parquet_files(path))
    return [file_path for file_path in candidates if file_path not in invalid]


def _parquet_file_candidates(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [] if _is_internal_lake_path(path) else [path]
    if not path.exists():
        return []
    return sorted(
        candidate for candidate in path.rglob("*.parquet") if not _is_internal_lake_path(candidate)
    )


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
    states: list[DatasetState] = []
    for name in sorted(DATASET_PATHS):
        path = dataset_path_for(lake_root, name)
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
    return states


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

    return {
        "status": status,
        "v5_permission": consumers["permissions"].get("v5", "UNKNOWN"),
        "v7_permission": consumers["permissions"].get("v7", "UNKNOWN"),
        "okx_public_rest_status": "OK" if market_health["latest_market_bar_ts"] else "WARNING",
        "okx_public_ws_status": ws_status,
        "okx_readonly_status": readonly_private_status(lake_root),
        "latest_market_bar_ts": market_health["latest_market_bar_ts"],
        "missing_bar_ratio": market_health["missing_bar_ratio"],
        "cost_fallback_ratio": costs["fallback_ratio"],
        "alpha_gate_counts": gates["counts"],
        "latest_expert_pack": experts["latest_pack"],
        "strategy_opportunity_advisory": redact_frame(_strategy_opportunity_table(advisory)).head(
            30
        ),
        "entry_quality_advisory": redact_frame(_entry_quality_table(entry_quality)).head(30),
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


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
    table = _sort_by_priority(
        table,
        "recommended_mode",
        RECOMMENDED_MODE_PRIORITY,
        ["decision", "strategy_candidate", "symbol", "horizon_hours"],
    )
    return _sort_by_priority(
        table,
        "decision",
        DECISION_PRIORITY,
        ["recommended_mode", "strategy_candidate", "symbol", "horizon_hours"],
    )


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
        "missing_bar_ratio": missing_ratio,
        "warnings": warnings,
    }


def _market_bar_lazy_health(lake_root: str | Path) -> dict[str, Any]:
    snapshot = _dataset_snapshot(lake_root, "market_bar")
    path = dataset_path_for(lake_root, "market_bar")
    files = _valid_parquet_files(path, invalid_files=invalid_parquet_files(path))
    if not files:
        return {
            "row_count": 0,
            "warning": snapshot.warning,
            "latest_per_symbol": pl.DataFrame(),
            "missing_bars": pl.DataFrame(),
            "duplicate_bar_count": 0,
            "unclosed_bar_count": 0,
            "schema_violations": ["market_bar 数据集缺失或为空"],
            "latest_market_bar_ts": None,
        }
    try:
        lazy = _scan_parquet_files(files)
        schema = lazy.collect_schema()
    except Exception as exc:
        return {
            "row_count": 0,
            "warning": f"market_bar 元数据读取失败：{exc}",
            "latest_per_symbol": pl.DataFrame(),
            "missing_bars": pl.DataFrame(),
            "duplicate_bar_count": 0,
            "unclosed_bar_count": 0,
            "schema_violations": ["market_bar 数据集读取失败"],
            "latest_market_bar_ts": None,
        }

    latest_ts = _coerce_timestamp(snapshot.freshness.get("latest_timestamp"))
    return {
        "row_count": snapshot.rows,
        "warning": snapshot.warning,
        "latest_per_symbol": _latest_market_bars_lazy(lazy, schema),
        "missing_bars": _missing_bar_table_lazy(lazy, schema),
        "duplicate_bar_count": _duplicate_market_bar_count_lazy(lazy, schema),
        "unclosed_bar_count": _unclosed_market_bar_count_lazy(lazy, schema),
        "schema_violations": _market_bar_schema_violations_lazy(lazy, schema),
        "latest_market_bar_ts": latest_ts,
    }


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
    books, books_warning = read_recent_dataset_with_warning(lake_root, "orderbook_snapshot")
    trades, trades_warning = read_recent_dataset_with_warning(lake_root, "trade_print")
    ws_health, ws_health_warning = read_dataset_with_warning(lake_root, "okx_public_ws_health")
    warnings = [
        warning
        for warning in [market_warning, books_warning, trades_warning, ws_health_warning]
        if warning
    ]
    if market.is_empty():
        return {
            "regimes": pl.DataFrame(),
            "spread_bps": pl.DataFrame(),
            "trade_activity": pl.DataFrame(),
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
        .sort(["symbol", "timeframe"])
    )

    spread_bps = orderbook_spread_table(books)
    trade_activity = trade_activity_table(trades)
    warnings.extend(
        _market_universe_warnings(
            market,
            books,
            trade_activity,
            ws_subscription=_latest_ws_subscription(ws_health),
        )
    )

    return {
        "regimes": redact_frame(regimes),
        "spread_bps": spread_bps,
        "trade_activity": trade_activity,
        "abnormal_symbols": regimes.filter(pl.col("mean_abs_return") > 0.03),
        "warnings": warnings,
    }


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
    warnings = [warning for warning in [costs_warning, health_warning] if warning]
    if costs.is_empty():
        return {
            "costs": pl.DataFrame(),
            "cost_health": redact_frame(health).head(DISPLAY_LIMIT),
            "actual_rows": 0,
            "mixed_rows": 0,
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
    fallback_ratio = latest_health.get("fallback_ratio", fallback_ratio)
    return {
        "costs": redact_frame(costs).head(DISPLAY_LIMIT),
        "cost_health": redact_frame(health).head(DISPLAY_LIMIT),
        "actual_rows": int(latest_health.get("actual_rows") or 0),
        "mixed_rows": int(latest_health.get("mixed_rows") or 0),
        "proxy_rows": int(latest_health.get("proxy_rows") or 0),
        "global_default_rows": int(latest_health.get("global_default_rows") or 0),
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
        "alpha_factory_result": redact_frame(
            _alpha_factory_result_table(alpha_factory_results)
        ).head(DISPLAY_LIMIT),
        "alpha_factory_promotion_queue": redact_frame(
            _alpha_factory_promotion_table(alpha_factory_promotion)
        ).head(DISPLAY_LIMIT),
        "missed_opportunity_audit": redact_frame(
            _missed_opportunity_table(missed_opportunity)
        ).head(DISPLAY_LIMIT),
        "risk_on_multi_buy_shadow": redact_frame(
            _risk_on_multi_buy_table(risk_on_multi_buy)
        ).head(DISPLAY_LIMIT),
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
        "expanded_universe_candidate_maturity": redact_frame(expanded_maturity).head(
            DISPLAY_LIMIT
        ),
        "expanded_universe_watchlist": redact_frame(expanded_watchlist).head(DISPLAY_LIMIT),
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
    return _sort_by_priority(
        table,
        "decision",
        DECISION_PRIORITY,
        ["strategy_candidate", "horizon_hours"],
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
    return _sort_by_priority(
        table,
        "recommended_mode",
        RECOMMENDED_MODE_PRIORITY,
        ["promotion_state", "strategy_candidate", "symbol", "horizon_hours"],
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
    sort_columns = [
        column for column in ["outcome_if_blocked", "symbol", "ts_utc"] if column in table.columns
    ]
    return table.sort(sort_columns) if sort_columns else table


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
    sort_columns = [
        column
        for column in ["ts_utc", "top_k", "would_buy_symbol"]
        if column in table.columns
    ]
    return table.sort(sort_columns, descending=True) if sort_columns else table


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
    return (
        table.sort(["promotion_state", "symbol", "strategy_candidate"])
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
    if "quality_score" in table.columns:
        return table.sort("quality_score", descending=True)
    return table


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
    return _sort_by_priority(
        table,
        "decision",
        DECISION_PRIORITY,
        ["strategy_candidate", "symbol", "horizon_hours"],
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
    return _sort_by_priority(table, "decision", DECISION_PRIORITY, [candidate_column])


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
    table = _sort_by_priority(table, "status", RESEARCH_STATUS_PRIORITY, ["module", "research_id"])
    table = _sort_by_priority(table, "action", RESEARCH_ACTION_PRIORITY, ["status", "research_id"])
    hidden = [column for column in ["created_at"] if column in table.columns]
    return table.drop(hidden) if hidden else table


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
    status = str(row.get("status") or "").strip()
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

    latest_by_strategy: dict[str, str] = {}
    if not permissions.is_empty() and {"strategy", "permission"}.issubset(permissions.columns):
        sort_column = "as_of_ts" if "as_of_ts" in permissions.columns else "created_at"
        if sort_column not in permissions.columns:
            sort_column = "strategy"
        for row in permissions.sort(sort_column).to_dicts():
            status = row.get("permission_status")
            latest_by_strategy[str(row["strategy"]).lower()] = str(status or row["permission"])

    return {
        "permissions": latest_by_strategy,
        "permission_rows": redact_frame(permissions).head(DISPLAY_LIMIT),
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
    warnings = [
        warning
        for warning in [health_warning, gate_warning, mode_warning, enforcement_warning]
        if warning
    ]
    if health.is_empty():
        return {
            "latest": {},
            "health_rows": pl.DataFrame(),
            "gate_compliance_rows": gate,
            "quant_lab_mode_rows": mode,
            "quant_lab_enforcement_rows": enforcement,
            "warnings": [*warnings, "strategy_health_daily 数据集缺失或为空"],
        }
    sort_column = "date" if "date" in health.columns else health.columns[0]
    latest = health.sort(sort_column).tail(1).to_dicts()[0]
    return {
        "latest": latest,
        "health_rows": health.sort(sort_column, descending=True).head(DISPLAY_LIMIT),
        "gate_compliance_rows": gate.head(DISPLAY_LIMIT),
        "quant_lab_mode_rows": mode.head(DISPLAY_LIMIT),
        "quant_lab_enforcement_rows": enforcement.head(DISPLAY_LIMIT),
        "warnings": warnings,
    }


def expert_export_summary(exports_root: str | Path) -> dict[str, Any]:
    root = Path(exports_root)
    packs = _expert_pack_paths(root)
    if not packs:
        return {
            "latest_pack": None,
            "packs": pl.DataFrame(),
            "manifest_summary": {},
            "data_quality_summary": {},
            "expert_questions": [],
            "warnings": [f"未在目录下找到专家包：{root}"],
        }

    latest = packs[0]
    manifest = _read_json_from_zip(latest, "manifest.json")
    data_quality = _read_json_from_zip(latest, "data_quality.json")
    questions = _read_text_from_zip(latest, "expert_questions.md").splitlines()
    pack_rows = [
        {
            "path": str(path),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        }
        for path in packs
    ]
    return {
        "latest_pack": str(latest),
        "packs": pl.DataFrame(pack_rows),
        "manifest_summary": manifest,
        "data_quality_summary": data_quality,
        "expert_questions": [line for line in questions if line.strip()][:20],
        "warnings": [],
    }


def _expert_pack_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        root.glob("quant_lab_expert_pack_*.zip"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def default_exports_root(lake_root: str | Path) -> Path:
    root = Path(lake_root)
    if root.name == "lake":
        return root.parent / "exports"
    return root / "exports"


def _latest_v5_bundle_ts(lake_root: str | Path) -> datetime | None:
    latest = _latest_dataset_timestamp_by_path(
        dataset_path_for(lake_root, "strategy_health_daily"),
        "strategy_health_daily",
        timestamp_columns=("latest_bundle_ts", "created_at", "date"),
    )
    if latest is not None:
        return latest
    return _latest_dataset_timestamp_by_path(
        Path(lake_root) / "bronze/strategy_telemetry/v5/bundle_manifest",
        "bronze/strategy_telemetry/v5/bundle_manifest",
        timestamp_columns=("bundle_ts", "ingest_ts", "created_at"),
    )


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
            return 1 if root.suffix == ".parquet" else 0
        if not root.exists():
            return 0
        return sum(1 for _ in root.rglob("*.parquet"))
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
    rows = []
    v5_telemetry_is_current = _v5_telemetry_is_current(lake_root)
    for name in sorted(DATASET_PATHS):
        path = dataset_path_for(lake_root, name)
        snapshot = _dataset_snapshot(lake_root, name)
        freshness = snapshot.freshness
        status = freshness["freshness_status"]
        if snapshot.rows == 0:
            status = _empty_dataset_status(name)
        if name == "v5_trade_event" and status == "stale" and v5_telemetry_is_current:
            status = "event_driven_no_recent_trade"
        if name in HISTORICAL_RESEARCH_DATASETS and status == "stale":
            status = "historical_research_snapshot"
        if _should_show_stale_dataset_row(snapshot, status):
            rows.append(
                {
                    "dataset": name,
                    "rows": snapshot.rows,
                    "status": status,
                    "path": str(path),
                    "freshness_seconds": freshness["freshness_seconds"],
                    "timestamp_column": freshness["timestamp_column"] or "",
                    "latest_timestamp": freshness["latest_timestamp"] or "",
                    "warning": snapshot.warning or "",
                }
            )
    return pl.DataFrame(rows)


def _empty_dataset_status(dataset_name: str) -> str:
    if dataset_name == "alpha_evidence":
        return "研究证据尚未生成"
    if dataset_name == "feature_value":
        return "特征尚未发布"
    if dataset_name == "decision_audit":
        return "legacy_optional"
    if dataset_name in V5_PAPER_TELEMETRY_DATASETS:
        return "waiting_for_v5_paper_telemetry"
    if dataset_name in ENTRY_QUALITY_DATASETS:
        return "entry_quality_optional"
    return "missing"


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
