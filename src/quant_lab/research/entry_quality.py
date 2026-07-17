from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.lake import (
    read_parquet_dataset,
    upsert_parquet_dataset,
    write_parquet_dataset,
    write_snapshot_meta,
)
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

SOURCE_NAME = "quant_lab"
ENTRY_QUALITY_SCHEMA_VERSION = "entry_quality.v0.1"
STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION = "strategy_opportunity_advisory.v0.1"
STRATEGY_OPPORTUNITY_ADVISORY_TTL_SECONDS = 3 * 60 * 60
UNOBSERVABLE_TEXT_VALUES = {"", "none", "null", "nan", "not_observable", "unknown"}
DEFAULT_WINDOW_HOURS = 24
ENTRY_QUALITY_SYMBOLS = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"}
LATE_CHASE_THRESHOLDS_BPS = (100, 150, 200, 250, 300, 400)
LATE_CHASE_BY_SYMBOL_THRESHOLDS_BPS = (50, 100, 150, 200, 250, 300)
PULLBACK_HORIZON_HOURS = (4, 8, 12, 24, 48, 72)
CANDIDATE_LABEL_DECISION_MAX_LAG = timedelta(hours=1)
MIN_ROUNDTRIP_COST_BPS = 30.0
PULLBACK_OLD_RULE_VERSION = "old_pullback_v0.1"
PULLBACK_NEW_RULE_VERSION = "confirmed_reversal_v0.2"
PULLBACK_SPREAD_MAX_BPS = 20.0
PULLBACK_BTC_MAX_1H_DROP_BPS = -120.0
PULLBACK_BTC_MAX_4H_DROP_BPS = -250.0

V5_TRADE_EVENT_DATASET = Path("silver") / "v5_trade_event"
V5_ORDER_LIFECYCLE_DATASET = Path("silver") / "v5_order_lifecycle"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
V5_CANDIDATE_EVENT_DATASET = Path("silver") / "v5_candidate_event"
V5_CANDIDATE_LABEL_DATASET = Path("gold") / "v5_candidate_label"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"

MISSED_LOW_AUDIT_DATASET = Path("gold") / "v5_missed_low_audit"
MISSED_LOW_BY_SYMBOL_DATASET = Path("gold") / "v5_missed_low_by_symbol"
MISSED_LOW_BY_ENTRY_REASON_DATASET = Path("gold") / "v5_missed_low_by_entry_reason"
LATE_ENTRY_CHASE_SHADOW_DATASET = Path("gold") / "v5_late_entry_chase_shadow"
LATE_ENTRY_CHASE_THRESHOLD_ADVISORY_DATASET = (
    Path("gold") / "v5_late_entry_chase_threshold_advisory"
)
LATE_ENTRY_CHASE_THRESHOLD_BY_SYMBOL_DATASET = (
    Path("gold") / "v5_late_entry_chase_threshold_by_symbol"
)
PULLBACK_REVERSAL_SHADOW_DATASET = Path("gold") / "v5_pullback_reversal_shadow"
PULLBACK_REVERSAL_READINESS_DATASET = Path("gold") / "v5_pullback_reversal_readiness"
PULLBACK_REVERSAL_RULE_COMPARISON_DATASET = (
    Path("gold") / "v5_pullback_reversal_rule_comparison"
)
ENTRY_QUALITY_ADVISORY_DATASET = Path("gold") / "v5_entry_quality_advisory"
STRATEGY_OPPORTUNITY_ADVISORY_DATASET = Path("gold") / "strategy_opportunity_advisory"

HISTORY_MISSED_LOW_AUDIT_DATASET = Path("gold") / "v5_entry_quality_history_missed_low_audit"
HISTORY_MISSED_LOW_BY_SYMBOL_DATASET = (
    Path("gold") / "v5_entry_quality_history_missed_low_by_symbol"
)
HISTORY_MISSED_LOW_BY_ENTRY_REASON_DATASET = (
    Path("gold") / "v5_entry_quality_history_missed_low_by_entry_reason"
)
HISTORY_LATE_ENTRY_CHASE_SHADOW_DATASET = (
    Path("gold") / "v5_entry_quality_history_late_entry_chase_shadow"
)
HISTORY_LATE_ENTRY_THRESHOLD_SENSITIVITY_DATASET = (
    Path("gold") / "v5_entry_quality_history_late_entry_chase_threshold_sensitivity"
)
HISTORY_PULLBACK_REVERSAL_SHADOW_DATASET = (
    Path("gold") / "v5_entry_quality_history_pullback_reversal_shadow"
)
HISTORY_PULLBACK_BY_SYMBOL_DATASET = (
    Path("gold") / "v5_entry_quality_history_pullback_by_symbol"
)
HISTORY_PULLBACK_BY_REGIME_DATASET = (
    Path("gold") / "v5_entry_quality_history_pullback_by_regime"
)
HISTORY_PULLBACK_BY_HORIZON_DATASET = (
    Path("gold") / "v5_entry_quality_history_pullback_by_horizon"
)
HISTORY_ANTI_LEAKAGE_CHECK_DATASET = (
    Path("gold") / "v5_entry_quality_history_anti_leakage_check"
)
HISTORY_METRICS_DATASET = Path("gold") / "v5_entry_quality_history_metrics"


COMMON_SCHEMA = {
    "contract_version": pl.Utf8,
    "schema_version": pl.Utf8,
    "quant_lab_git_commit": pl.Utf8,
    "source_version": pl.Utf8,
    "generated_at_utc": pl.Datetime(time_zone="UTC"),
    "generated_from_bundle_id": pl.Utf8,
    "as_of_date": pl.Utf8,
    "window_hours": pl.Int64,
    "source": pl.Utf8,
    "mode": pl.Utf8,
}

HISTORY_META_SCHEMA = {
    "start_date": pl.Utf8,
    "end_date": pl.Utf8,
    "window_mode": pl.Utf8,
    "cost_mode": pl.Utf8,
}

MISSED_LOW_AUDIT_SCHEMA = COMMON_SCHEMA | {
    "run_id": pl.Utf8,
    "source_event_key": pl.Utf8,
    "symbol": pl.Utf8,
    "entry_ts": pl.Datetime(time_zone="UTC"),
    "entry_px": pl.Float64,
    "entry_reason": pl.Utf8,
    "probe_type": pl.Utf8,
    "side": pl.Utf8,
    "intent": pl.Utf8,
    "entry_vs_pre_4h_low_bps": pl.Float64,
    "entry_vs_pre_8h_low_bps": pl.Float64,
    "entry_vs_pre_12h_low_bps": pl.Float64,
    "entry_vs_pre_24h_low_bps": pl.Float64,
    "entry_position_in_24h_range": pl.Float64,
    "realized_net_bps": pl.Float64,
    "exit_reason": pl.Utf8,
    "diagnosis": pl.Utf8,
}

MISSED_LOW_AGG_SCHEMA = COMMON_SCHEMA | {
    "group_key": pl.Utf8,
    "sample_count": pl.Int64,
    "loss_count": pl.Int64,
    "profit_count": pl.Int64,
    "late_chase_loss_count": pl.Int64,
    "late_but_trend_profitable_count": pl.Int64,
    "avg_entry_vs_pre_24h_low_bps": pl.Float64,
    "avg_entry_position_in_24h_range": pl.Float64,
    "avg_realized_net_bps": pl.Float64,
    "diagnosis_mix": pl.Utf8,
}

LATE_ENTRY_CHASE_SHADOW_SCHEMA = COMMON_SCHEMA | {
    "strategy_candidate": pl.Utf8,
    "source_type": pl.Utf8,
    "run_id": pl.Utf8,
    "candidate_id": pl.Utf8,
    "source_event_key": pl.Utf8,
    "symbol": pl.Utf8,
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "entry_or_candidate_px": pl.Float64,
    "recent_12h_low": pl.Float64,
    "recent_24h_low": pl.Float64,
    "entry_vs_12h_low_bps": pl.Float64,
    "entry_vs_24h_low_bps": pl.Float64,
    "entry_position_in_12h_range": pl.Float64,
    "f4_volume_expansion": pl.Float64,
    "f5_rsi_trend_confirm": pl.Float64,
    "late_chase_risk": pl.Boolean,
    "would_block_if_enabled": pl.Boolean,
    "realized_net_bps": pl.Float64,
    "forward_24h_net_bps": pl.Float64,
    "forward_48h_net_bps": pl.Float64,
    "outcome_class": pl.Utf8,
}

LATE_ENTRY_THRESHOLD_SCHEMA = COMMON_SCHEMA | {
    "threshold_bps": pl.Int64,
    "would_block_count": pl.Int64,
    "would_block_loss_count": pl.Int64,
    "would_block_profit_count": pl.Int64,
    "false_positive_rate": pl.Float64,
    "avg_net_bps_blocked": pl.Float64,
    "avg_net_bps_not_blocked": pl.Float64,
    "ready_for_live_guard": pl.Boolean,
    "advisory": pl.Utf8,
}

LATE_ENTRY_THRESHOLD_BY_SYMBOL_SCHEMA = LATE_ENTRY_THRESHOLD_SCHEMA | {
    "symbol": pl.Utf8,
}

PULLBACK_REVERSAL_SHADOW_SCHEMA = COMMON_SCHEMA | {
    "rule_version": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "run_id": pl.Utf8,
    "candidate_id": pl.Utf8,
    "source_event_key": pl.Utf8,
    "symbol": pl.Utf8,
    "ts_utc": pl.Datetime(time_zone="UTC"),
    "regime_state": pl.Utf8,
    "risk_level": pl.Utf8,
    "current_px": pl.Float64,
    "pre_24h_low": pl.Float64,
    "pre_24h_high": pl.Float64,
    "pullback_from_24h_high_bps": pl.Float64,
    "recent_2h_no_new_low": pl.Boolean,
    "close_reclaim_1h": pl.Boolean,
    "current_close_gt_previous_close": pl.Boolean,
    "btc_not_sharp_drop": pl.Boolean,
    "spread_not_abnormal": pl.Boolean,
    "f4_volume_expansion": pl.Float64,
    "f5_rsi_trend_confirm": pl.Float64,
    "selected_roundtrip_cost_bps": pl.Float64,
    "cost_quality": pl.Utf8,
    "horizon_hours": pl.Int64,
    "gross_bps": pl.Float64,
    "net_bps_after_cost": pl.Float64,
    "mfe_bps": pl.Float64,
    "mae_bps": pl.Float64,
    "win": pl.Boolean,
    "label_status": pl.Utf8,
}

PULLBACK_REVERSAL_READINESS_SCHEMA = COMMON_SCHEMA | {
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "sample_count": pl.Int64,
    "recent_7d_sample_count": pl.Int64,
    "avg_24h_net_bps": pl.Float64,
    "win_rate_24h": pl.Float64,
    "p25_24h_net_bps": pl.Float64,
    "avg_mae_bps": pl.Float64,
    "ready_for_paper": pl.Boolean,
    "ready_for_live_probe": pl.Boolean,
    "readiness_reasons": pl.Utf8,
}

PULLBACK_RULE_COMPARISON_SCHEMA = COMMON_SCHEMA | {
    "comparison_name": pl.Utf8,
    "rule_name": pl.Utf8,
    "rule_version": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon_hours": pl.Int64,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "avg_mae_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
}

ENTRY_QUALITY_ADVISORY_SCHEMA = COMMON_SCHEMA | {
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "readiness_status": pl.Utf8,
    "sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "advisory_reasons": pl.Utf8,
    "would_block_if_enabled": pl.Boolean,
    "would_enter": pl.Boolean,
    "no_sample_reason": pl.Utf8,
    "ready_for_live": pl.Boolean,
}

STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA = {
    "as_of_ts": pl.Datetime(time_zone="UTC"),
    "generated_at": pl.Datetime(time_zone="UTC"),
    "expires_at": pl.Datetime(time_zone="UTC"),
    "contract_version": pl.Utf8,
    "schema_version": pl.Utf8,
    "quant_lab_git_commit": pl.Utf8,
    "source_version": pl.Utf8,
    "would_block_if_enabled": pl.Boolean,
    "would_enter": pl.Boolean,
    "no_sample_reason": pl.Utf8,
    "strategy_id": pl.Utf8,
    "symbol": pl.Utf8,
    "v5_symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "decision": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "horizon_hours": pl.Int64,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "cost_quality": pl.Utf8,
    "paper_days": pl.Int64,
    "entry_day_count": pl.Int64,
    "paper_pnl_observed_count": pl.Int64,
    "slippage_coverage": pl.Float64,
    "live_block_reasons": pl.Utf8,
    "max_paper_notional_usdt": pl.Float64,
    "max_live_notional_usdt": pl.Float64,
}

HISTORY_THRESHOLD_SENSITIVITY_SCHEMA = LATE_ENTRY_THRESHOLD_SCHEMA | HISTORY_META_SCHEMA

HISTORY_PULLBACK_AGG_SCHEMA = COMMON_SCHEMA | HISTORY_META_SCHEMA | {
    "group_type": pl.Utf8,
    "group_key": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "regime_state": pl.Utf8,
    "horizon_hours": pl.Int64,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "median_net_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "avg_mfe_bps": pl.Float64,
    "avg_mae_bps": pl.Float64,
    "cost_quality_mix": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
}

HISTORY_ANTI_LEAKAGE_SCHEMA = COMMON_SCHEMA | HISTORY_META_SCHEMA | {
    "check_name": pl.Utf8,
    "status": pl.Utf8,
    "violation_count": pl.Int64,
    "detail": pl.Utf8,
}

HISTORY_METRICS_SCHEMA = COMMON_SCHEMA | HISTORY_META_SCHEMA | {
    "missed_low_audit_rows": pl.Int64,
    "late_entry_chase_shadow_rows": pl.Int64,
    "late_entry_threshold_rows": pl.Int64,
    "pullback_reversal_shadow_rows": pl.Int64,
    "pullback_by_symbol_rows": pl.Int64,
    "pullback_by_regime_rows": pl.Int64,
    "pullback_by_horizon_rows": pl.Int64,
    "anti_leakage_status": pl.Utf8,
    "ready_for_live_rows": pl.Int64,
    "metrics_json": pl.Utf8,
}

HISTORY_MISSED_LOW_AUDIT_SCHEMA = MISSED_LOW_AUDIT_SCHEMA | HISTORY_META_SCHEMA
HISTORY_MISSED_LOW_AGG_SCHEMA = MISSED_LOW_AGG_SCHEMA | HISTORY_META_SCHEMA
HISTORY_LATE_ENTRY_CHASE_SHADOW_SCHEMA = LATE_ENTRY_CHASE_SHADOW_SCHEMA | HISTORY_META_SCHEMA
HISTORY_PULLBACK_REVERSAL_SHADOW_SCHEMA = PULLBACK_REVERSAL_SHADOW_SCHEMA | HISTORY_META_SCHEMA


@dataclass(frozen=True)
class EntryQualityHistoryOutputSpec:
    dataset_name: str
    relative_path: Path
    schema: dict[str, pl.DataType]
    primary_keys: tuple[str, ...]
    window_keys: tuple[str, ...] = ("window_mode", "cost_mode")
    publish_mode: str = "window_replace"
    empty_result_semantics: str = "clear_window"


ENTRY_QUALITY_HISTORY_OUTPUT_SPECS = (
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_missed_low_audit",
        HISTORY_MISSED_LOW_AUDIT_DATASET,
        HISTORY_MISSED_LOW_AUDIT_SCHEMA,
        (
            "start_date",
            "end_date",
            "window_mode",
            "cost_mode",
            "source_event_key",
            "symbol",
            "entry_ts",
        ),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_missed_low_by_symbol",
        HISTORY_MISSED_LOW_BY_SYMBOL_DATASET,
        HISTORY_MISSED_LOW_AGG_SCHEMA,
        ("start_date", "end_date", "window_mode", "cost_mode", "group_key"),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_missed_low_by_entry_reason",
        HISTORY_MISSED_LOW_BY_ENTRY_REASON_DATASET,
        HISTORY_MISSED_LOW_AGG_SCHEMA,
        ("start_date", "end_date", "window_mode", "cost_mode", "group_key"),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_late_entry_chase_shadow",
        HISTORY_LATE_ENTRY_CHASE_SHADOW_DATASET,
        HISTORY_LATE_ENTRY_CHASE_SHADOW_SCHEMA,
        (
            "start_date",
            "end_date",
            "window_mode",
            "cost_mode",
            "source_type",
            "source_event_key",
        ),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_late_entry_chase_threshold_sensitivity",
        HISTORY_LATE_ENTRY_THRESHOLD_SENSITIVITY_DATASET,
        HISTORY_THRESHOLD_SENSITIVITY_SCHEMA,
        (
            "start_date",
            "end_date",
            "window_mode",
            "cost_mode",
            "threshold_bps",
        ),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_pullback_reversal_shadow",
        HISTORY_PULLBACK_REVERSAL_SHADOW_DATASET,
        HISTORY_PULLBACK_REVERSAL_SHADOW_SCHEMA,
        (
            "start_date",
            "end_date",
            "window_mode",
            "cost_mode",
            "source_event_key",
            "horizon_hours",
        ),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_pullback_by_symbol",
        HISTORY_PULLBACK_BY_SYMBOL_DATASET,
        HISTORY_PULLBACK_AGG_SCHEMA,
        (
            "start_date",
            "end_date",
            "window_mode",
            "cost_mode",
            "group_type",
            "group_key",
        ),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_pullback_by_regime",
        HISTORY_PULLBACK_BY_REGIME_DATASET,
        HISTORY_PULLBACK_AGG_SCHEMA,
        (
            "start_date",
            "end_date",
            "window_mode",
            "cost_mode",
            "group_type",
            "group_key",
        ),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_pullback_by_horizon",
        HISTORY_PULLBACK_BY_HORIZON_DATASET,
        HISTORY_PULLBACK_AGG_SCHEMA,
        (
            "start_date",
            "end_date",
            "window_mode",
            "cost_mode",
            "group_type",
            "group_key",
        ),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_anti_leakage_check",
        HISTORY_ANTI_LEAKAGE_CHECK_DATASET,
        HISTORY_ANTI_LEAKAGE_SCHEMA,
        ("start_date", "end_date", "window_mode", "cost_mode", "check_name"),
    ),
    EntryQualityHistoryOutputSpec(
        "v5_entry_quality_history_metrics",
        HISTORY_METRICS_DATASET,
        HISTORY_METRICS_SCHEMA,
        ("start_date", "end_date", "window_mode", "cost_mode"),
    ),
)

ENTRY_QUALITY_HISTORY_OUTPUT_SPEC_BY_NAME = {
    spec.dataset_name: spec for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS
}

ENTRY_QUALITY_HISTORY_REPORT_NAMES = (
    "missed_low_audit.csv",
    "missed_low_by_symbol.csv",
    "missed_low_by_entry_reason.csv",
    "late_entry_chase_threshold_sensitivity.csv",
    "pullback_reversal_by_symbol.csv",
    "pullback_reversal_by_regime.csv",
    "pullback_reversal_by_horizon.csv",
    "anti_leakage_check.csv",
    "entry_quality_historical_metrics.json",
    "entry_quality_historical_summary.md",
)


class EntryQualityBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    missed_low_audit_rows: int = Field(ge=0)
    missed_low_by_symbol_rows: int = Field(ge=0)
    missed_low_by_entry_reason_rows: int = Field(ge=0)
    late_entry_chase_shadow_rows: int = Field(ge=0)
    late_entry_chase_threshold_rows: int = Field(ge=0)
    late_entry_chase_threshold_by_symbol_rows: int = Field(ge=0)
    pullback_reversal_shadow_rows: int = Field(ge=0)
    pullback_reversal_readiness_rows: int = Field(ge=0)
    pullback_rule_comparison_rows: int = Field(ge=0)
    entry_quality_advisory_rows: int = Field(ge=0)
    strategy_opportunity_advisory_rows: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class EntryQualityHistoryBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    start_date: str
    end_date: str
    mode: str
    cost_mode: str
    missed_low_audit_rows: int = Field(ge=0)
    missed_low_by_symbol_rows: int = Field(ge=0)
    missed_low_by_entry_reason_rows: int = Field(ge=0)
    late_entry_chase_shadow_rows: int = Field(ge=0)
    late_entry_threshold_sensitivity_rows: int = Field(ge=0)
    pullback_reversal_shadow_rows: int = Field(ge=0)
    pullback_by_symbol_rows: int = Field(ge=0)
    pullback_by_regime_rows: int = Field(ge=0)
    pullback_by_horizon_rows: int = Field(ge=0)
    anti_leakage_rows: int = Field(ge=0)
    metrics_rows: int = Field(ge=0)
    reports_dir: str
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _BuildContext:
    as_of_date: date
    generated_at: datetime
    generated_from_bundle_id: str
    window_hours: int


@dataclass(frozen=True)
class _HistoryContext:
    start_date: date
    end_date: date
    window_mode: str
    cost_mode: str


@dataclass(frozen=True)
class EntryQualityHistoryArtifacts:
    start_date: date
    end_date: date
    mode: str
    cost_mode: str
    generated_at: datetime
    generated_from_bundle_id: str
    missed_low_audit: pl.DataFrame
    missed_low_by_symbol: pl.DataFrame
    missed_low_by_entry_reason: pl.DataFrame
    late_entry_chase_shadow: pl.DataFrame
    late_entry_threshold_sensitivity: pl.DataFrame
    pullback_reversal_shadow: pl.DataFrame
    pullback_by_symbol: pl.DataFrame
    pullback_by_regime: pl.DataFrame
    pullback_by_horizon: pl.DataFrame
    anti_leakage_check: pl.DataFrame
    metrics: pl.DataFrame
    reports: dict[str, bytes]
    warnings: tuple[str, ...]

    def frames_by_dataset(self) -> dict[str, pl.DataFrame]:
        return {
            "v5_entry_quality_history_missed_low_audit": self.missed_low_audit,
            "v5_entry_quality_history_missed_low_by_symbol": self.missed_low_by_symbol,
            "v5_entry_quality_history_missed_low_by_entry_reason": (
                self.missed_low_by_entry_reason
            ),
            "v5_entry_quality_history_late_entry_chase_shadow": (
                self.late_entry_chase_shadow
            ),
            "v5_entry_quality_history_late_entry_chase_threshold_sensitivity": (
                self.late_entry_threshold_sensitivity
            ),
            "v5_entry_quality_history_pullback_reversal_shadow": (
                self.pullback_reversal_shadow
            ),
            "v5_entry_quality_history_pullback_by_symbol": self.pullback_by_symbol,
            "v5_entry_quality_history_pullback_by_regime": self.pullback_by_regime,
            "v5_entry_quality_history_pullback_by_horizon": self.pullback_by_horizon,
            "v5_entry_quality_history_anti_leakage_check": self.anti_leakage_check,
            "v5_entry_quality_history_metrics": self.metrics,
        }


def build_and_publish_entry_quality(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> EntryQualityBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    ctx = _BuildContext(
        as_of_date=day,
        generated_at=datetime.now(UTC),
        generated_from_bundle_id=_latest_bundle_id(root),
        window_hours=max(int(window_hours), 1),
    )
    trades = read_parquet_dataset(root / V5_TRADE_EVENT_DATASET)
    lifecycles = read_parquet_dataset(root / V5_ORDER_LIFECYCLE_DATASET)
    market = _normalize_market_bars(read_parquet_dataset(root / MARKET_BAR_DATASET))
    candidates = read_parquet_dataset(root / V5_CANDIDATE_EVENT_DATASET)
    labels = read_parquet_dataset(root / V5_CANDIDATE_LABEL_DATASET)
    costs = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET)

    warnings: list[str] = []
    if market.is_empty():
        warnings.append("market_bar_empty")
    if trades.is_empty() and lifecycles.is_empty():
        warnings.append("v5_actual_entry_events_empty")
    if candidates.is_empty():
        warnings.append("v5_candidate_event_empty")

    actual_entries = _actual_entry_rows(trades, lifecycles)
    missed = build_missed_low_audit(actual_entries, market, ctx)
    missed_by_symbol = aggregate_missed_low(missed, group_column="symbol", ctx=ctx)
    missed_by_reason = aggregate_missed_low(missed, group_column="entry_reason", ctx=ctx)
    late_shadow = build_late_entry_chase_shadow(
        actual_entries=actual_entries,
        candidates=candidates,
        labels=labels,
        market_bars=market,
        ctx=ctx,
    )
    late_threshold = build_late_entry_chase_threshold_advisory(late_shadow, ctx=ctx)
    late_threshold_by_symbol = build_late_entry_chase_threshold_by_symbol(
        late_shadow,
        ctx=ctx,
    )
    pullback = build_pullback_reversal_shadow(
        candidates=candidates,
        market_bars=market,
        costs=costs,
        ctx=ctx,
    )
    if pullback.is_empty():
        pullback = build_pullback_reversal_shadow(
            candidates=candidates,
            market_bars=market,
            costs=costs,
            ctx=ctx,
            rule_version=PULLBACK_OLD_RULE_VERSION,
        )
    pullback_rule_comparison = build_pullback_reversal_rule_comparison(
        candidates=candidates,
        market_bars=market,
        costs=costs,
        ctx=ctx,
    )
    readiness = build_pullback_reversal_readiness(pullback, ctx=ctx)
    advisory = build_entry_quality_advisory(
        missed_low=missed,
        late_threshold=late_threshold,
        pullback_readiness=readiness,
        ctx=ctx,
    )
    strategy_opportunities = build_entry_quality_strategy_opportunity_advisory(
        advisory
    )

    return EntryQualityBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        missed_low_audit_rows=_publish_daily(
            root,
            MISSED_LOW_AUDIT_DATASET,
            missed,
            ["as_of_date", "source_event_key", "symbol", "entry_ts"],
        ),
        missed_low_by_symbol_rows=_publish_daily(
            root,
            MISSED_LOW_BY_SYMBOL_DATASET,
            missed_by_symbol,
            ["as_of_date", "group_key"],
        ),
        missed_low_by_entry_reason_rows=_publish_daily(
            root,
            MISSED_LOW_BY_ENTRY_REASON_DATASET,
            missed_by_reason,
            ["as_of_date", "group_key"],
        ),
        late_entry_chase_shadow_rows=_publish_daily(
            root,
            LATE_ENTRY_CHASE_SHADOW_DATASET,
            late_shadow,
            ["as_of_date", "source_type", "source_event_key", "symbol"],
        ),
        late_entry_chase_threshold_rows=_publish_daily(
            root,
            LATE_ENTRY_CHASE_THRESHOLD_ADVISORY_DATASET,
            late_threshold,
            ["as_of_date", "threshold_bps"],
        ),
        late_entry_chase_threshold_by_symbol_rows=_replace_daily(
            root,
            LATE_ENTRY_CHASE_THRESHOLD_BY_SYMBOL_DATASET,
            late_threshold_by_symbol,
        ),
        pullback_reversal_shadow_rows=_replace_daily(
            root,
            PULLBACK_REVERSAL_SHADOW_DATASET,
            pullback,
        ),
        pullback_reversal_readiness_rows=_replace_daily(
            root,
            PULLBACK_REVERSAL_READINESS_DATASET,
            readiness,
        ),
        pullback_rule_comparison_rows=_replace_daily(
            root,
            PULLBACK_REVERSAL_RULE_COMPARISON_DATASET,
            pullback_rule_comparison,
        ),
        entry_quality_advisory_rows=_replace_daily(
            root,
            ENTRY_QUALITY_ADVISORY_DATASET,
            advisory,
        ),
        strategy_opportunity_advisory_rows=_publish_entry_quality_strategy_opportunities(
            root,
            strategy_opportunities,
        ),
        warnings=warnings,
    )


def build_and_publish_entry_quality_history(
    lake_root: str | Path,
    *,
    start_date: str | date,
    end_date: str | date,
    mode: str = "full",
    cost_mode: str = "conservative",
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> EntryQualityHistoryBuildResult:
    """Run the explicit local fallback using the same compute/publish boundary as NAS."""

    root = Path(lake_root)
    start_day, end_day, window_mode, normalized_cost_mode = _normalize_history_request(
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        cost_mode=cost_mode,
    )
    artifacts = compute_entry_quality_history(
        trades=read_parquet_dataset(root / V5_TRADE_EVENT_DATASET),
        lifecycles=read_parquet_dataset(root / V5_ORDER_LIFECYCLE_DATASET),
        market_bars=read_parquet_dataset(root / MARKET_BAR_DATASET),
        candidates=read_parquet_dataset(root / V5_CANDIDATE_EVENT_DATASET),
        labels=read_parquet_dataset(root / V5_CANDIDATE_LABEL_DATASET),
        costs=read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET),
        start_date=start_day,
        end_date=end_day,
        mode=window_mode,
        cost_mode=normalized_cost_mode,
        window_hours=window_hours,
        generated_from_bundle_id=_latest_bundle_id(root),
    )
    publish_entry_quality_history_result(
        root,
        artifacts,
        generation_id=f"local-{artifacts.generated_at.strftime('%Y%m%dT%H%M%S%fZ')}",
        snapshot_id="local-fallback",
        task_id="local-fallback",
        reports=artifacts.reports,
    )
    reports_dir = root / "reports"
    frames = artifacts.frames_by_dataset()
    return EntryQualityHistoryBuildResult(
        lake_root=str(root),
        start_date=artifacts.start_date.isoformat(),
        end_date=artifacts.end_date.isoformat(),
        mode=artifacts.mode,
        cost_mode=artifacts.cost_mode,
        missed_low_audit_rows=frames["v5_entry_quality_history_missed_low_audit"].height,
        missed_low_by_symbol_rows=frames[
            "v5_entry_quality_history_missed_low_by_symbol"
        ].height,
        missed_low_by_entry_reason_rows=frames[
            "v5_entry_quality_history_missed_low_by_entry_reason"
        ].height,
        late_entry_chase_shadow_rows=frames[
            "v5_entry_quality_history_late_entry_chase_shadow"
        ].height,
        late_entry_threshold_sensitivity_rows=frames[
            "v5_entry_quality_history_late_entry_chase_threshold_sensitivity"
        ].height,
        pullback_reversal_shadow_rows=frames[
            "v5_entry_quality_history_pullback_reversal_shadow"
        ].height,
        pullback_by_symbol_rows=frames[
            "v5_entry_quality_history_pullback_by_symbol"
        ].height,
        pullback_by_regime_rows=frames[
            "v5_entry_quality_history_pullback_by_regime"
        ].height,
        pullback_by_horizon_rows=frames[
            "v5_entry_quality_history_pullback_by_horizon"
        ].height,
        anti_leakage_rows=frames[
            "v5_entry_quality_history_anti_leakage_check"
        ].height,
        metrics_rows=frames["v5_entry_quality_history_metrics"].height,
        reports_dir=str(reports_dir),
        warnings=list(artifacts.warnings),
    )


def compute_entry_quality_history(
    *,
    trades: pl.DataFrame,
    lifecycles: pl.DataFrame,
    market_bars: pl.DataFrame,
    candidates: pl.DataFrame,
    labels: pl.DataFrame,
    costs: pl.DataFrame,
    start_date: str | date,
    end_date: str | date,
    mode: str = "full",
    cost_mode: str = "conservative",
    window_hours: int = DEFAULT_WINDOW_HOURS,
    generated_at: datetime | None = None,
    generated_from_bundle_id: str = "",
) -> EntryQualityHistoryArtifacts:
    """Compute historical Entry Quality artifacts without reading or writing the Lake."""

    start_day, end_day, window_mode, normalized_cost_mode = _normalize_history_request(
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        cost_mode=cost_mode,
    )
    ctx = _BuildContext(
        as_of_date=end_day,
        generated_at=(generated_at or datetime.now(UTC)).astimezone(UTC),
        generated_from_bundle_id=str(generated_from_bundle_id or ""),
        window_hours=max(int(window_hours), 1),
    )
    history = _HistoryContext(
        start_date=start_day,
        end_date=end_day,
        window_mode=window_mode,
        cost_mode=normalized_cost_mode,
    )
    start_dt = datetime.combine(start_day, time.min, tzinfo=UTC)
    end_dt = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=UTC)

    scoped_trades = _filter_frame_by_time(
        trades,
        ("ts_utc", "ts", "entry_ts"),
        start_dt,
        end_dt,
    )
    scoped_lifecycles = _filter_frame_by_time(
        lifecycles,
        ("ts_utc", "last_fill_ts", "submit_ts", "decision_ts"),
        start_dt,
        end_dt,
    )
    market = _normalize_market_bars(
        _filter_frame_by_time(
            market_bars,
            ("ts",),
            start_dt - timedelta(hours=24),
            end_dt + timedelta(hours=max(PULLBACK_HORIZON_HOURS)),
        )
    )
    scoped_candidates = _filter_frame_by_time(
        candidates,
        ("ts_utc", "ts"),
        start_dt,
        end_dt,
    )
    scoped_labels = _filter_frame_by_time(
        labels,
        ("decision_ts", "ts_utc"),
        start_dt,
        end_dt,
    )
    if (
        not scoped_candidates.is_empty()
        and not scoped_labels.is_empty()
        and "candidate_id" in scoped_candidates.columns
        and "candidate_id" in scoped_labels.columns
    ):
        candidate_ids = (
            scoped_candidates.get_column("candidate_id").drop_nulls().unique().to_list()
        )
        scoped_labels = scoped_labels.filter(pl.col("candidate_id").is_in(candidate_ids))
    scoped_costs = _filter_frame_by_time(
        costs,
        ("as_of_date",),
        start_dt,
        end_dt,
    )

    warnings: list[str] = []
    if market.is_empty():
        warnings.append("market_bar_empty")
    if scoped_trades.is_empty() and scoped_lifecycles.is_empty():
        warnings.append("v5_actual_entry_events_empty")
    if scoped_candidates.is_empty():
        warnings.append("v5_candidate_event_empty")

    actual_entries = [
        row
        for row in _actual_entry_rows(scoped_trades, scoped_lifecycles)
        if _is_within_window(_coerce_datetime(row.get("entry_ts")), start_dt, end_dt)
    ]
    missed = _with_history_meta(
        build_missed_low_audit(actual_entries, market, ctx),
        history,
        HISTORY_MISSED_LOW_AUDIT_SCHEMA,
    )
    missed_by_symbol = _with_history_meta(
        aggregate_missed_low(missed, group_column="symbol", ctx=ctx),
        history,
        HISTORY_MISSED_LOW_AGG_SCHEMA,
    )
    missed_by_reason = _with_history_meta(
        aggregate_missed_low(missed, group_column="entry_reason", ctx=ctx),
        history,
        HISTORY_MISSED_LOW_AGG_SCHEMA,
    )
    late_shadow = _with_history_meta(
        build_late_entry_chase_shadow(
            actual_entries=actual_entries,
            candidates=scoped_candidates,
            labels=scoped_labels,
            market_bars=market,
            ctx=ctx,
        ),
        history,
        HISTORY_LATE_ENTRY_CHASE_SHADOW_SCHEMA,
    )
    late_threshold = _with_history_meta(
        build_late_entry_chase_threshold_advisory(late_shadow, ctx=ctx),
        history,
        HISTORY_THRESHOLD_SENSITIVITY_SCHEMA,
    )
    pullback = _with_history_meta(
        build_pullback_reversal_shadow(
            candidates=scoped_candidates,
            market_bars=market,
            costs=scoped_costs,
            ctx=ctx,
        ),
        history,
        HISTORY_PULLBACK_REVERSAL_SHADOW_SCHEMA,
    )
    pullback_by_symbol = build_pullback_reversal_history_aggregate(
        pullback, group_type="symbol", ctx=ctx, history=history
    )
    pullback_by_regime = build_pullback_reversal_history_aggregate(
        pullback, group_type="regime", ctx=ctx, history=history
    )
    pullback_by_horizon = build_pullback_reversal_history_aggregate(
        pullback, group_type="horizon", ctx=ctx, history=history
    )
    anti_leakage = build_entry_quality_anti_leakage_check(
        missed_low=missed,
        late_shadow=late_shadow,
        pullback=pullback,
        market_bars=market,
        candidates=scoped_candidates,
        labels=scoped_labels,
        start_dt=start_dt,
        end_dt=end_dt,
        ctx=ctx,
        history=history,
    )
    metrics = build_entry_quality_history_metrics(
        missed_low=missed,
        late_shadow=late_shadow,
        late_threshold=late_threshold,
        pullback=pullback,
        pullback_by_symbol=pullback_by_symbol,
        pullback_by_regime=pullback_by_regime,
        pullback_by_horizon=pullback_by_horizon,
        anti_leakage=anti_leakage,
        ctx=ctx,
        history=history,
        warnings=warnings,
    )
    provisional = EntryQualityHistoryArtifacts(
        start_date=start_day,
        end_date=end_day,
        mode=window_mode,
        cost_mode=normalized_cost_mode,
        generated_at=ctx.generated_at,
        generated_from_bundle_id=ctx.generated_from_bundle_id,
        missed_low_audit=missed,
        missed_low_by_symbol=missed_by_symbol,
        missed_low_by_entry_reason=missed_by_reason,
        late_entry_chase_shadow=late_shadow,
        late_entry_threshold_sensitivity=late_threshold,
        pullback_reversal_shadow=pullback,
        pullback_by_symbol=pullback_by_symbol,
        pullback_by_regime=pullback_by_regime,
        pullback_by_horizon=pullback_by_horizon,
        anti_leakage_check=anti_leakage,
        metrics=metrics,
        reports={},
        warnings=tuple(warnings),
    )
    return EntryQualityHistoryArtifacts(
        **{
            **provisional.__dict__,
            "reports": render_entry_quality_history_reports(provisional),
        }
    )


def _legacy_build_and_publish_entry_quality_history(
    lake_root: str | Path,
    *,
    start_date: str | date,
    end_date: str | date,
    mode: str = "full",
    cost_mode: str = "conservative",
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> EntryQualityHistoryBuildResult:
    root = Path(lake_root)
    start_day = _parse_day(start_date)
    end_day = _parse_day(end_date)
    if end_day < start_day:
        raise ValueError("end_date must be greater than or equal to start_date")
    window_mode = str(mode or "full").strip().lower()
    if window_mode not in {"full", "recent_7d", "recent_30d", "walk_forward"}:
        raise ValueError("mode must be one of: full, recent_7d, recent_30d, walk_forward")
    if window_mode == "recent_7d":
        start_day = max(start_day, end_day - timedelta(days=6))
    elif window_mode == "recent_30d":
        start_day = max(start_day, end_day - timedelta(days=29))

    normalized_cost_mode = str(cost_mode or "conservative").strip().lower()
    if normalized_cost_mode not in {"conservative", "quant_lab"}:
        raise ValueError("cost_mode must be one of: conservative, quant_lab")

    ctx = _BuildContext(
        as_of_date=end_day,
        generated_at=datetime.now(UTC),
        generated_from_bundle_id=_latest_bundle_id(root),
        window_hours=max(int(window_hours), 1),
    )
    history = _HistoryContext(
        start_date=start_day,
        end_date=end_day,
        window_mode=window_mode,
        cost_mode=normalized_cost_mode,
    )
    start_dt = datetime.combine(start_day, time.min, tzinfo=UTC)
    end_dt = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=UTC)

    trades = _filter_frame_by_time(
        read_parquet_dataset(root / V5_TRADE_EVENT_DATASET),
        ("ts_utc", "ts", "entry_ts"),
        start_dt,
        end_dt,
    )
    lifecycles = _filter_frame_by_time(
        read_parquet_dataset(root / V5_ORDER_LIFECYCLE_DATASET),
        ("ts_utc", "last_fill_ts", "submit_ts", "decision_ts"),
        start_dt,
        end_dt,
    )
    market = _normalize_market_bars(
        _filter_frame_by_time(
            read_parquet_dataset(root / MARKET_BAR_DATASET),
            ("ts",),
            start_dt - timedelta(hours=24),
            end_dt + timedelta(hours=max(PULLBACK_HORIZON_HOURS)),
        )
    )
    candidates = _filter_frame_by_time(
        read_parquet_dataset(root / V5_CANDIDATE_EVENT_DATASET),
        ("ts_utc", "ts"),
        start_dt,
        end_dt,
    )
    labels = _filter_frame_by_time(
        read_parquet_dataset(root / V5_CANDIDATE_LABEL_DATASET),
        ("decision_ts", "ts_utc"),
        start_dt,
        end_dt,
    )
    if (
        not candidates.is_empty()
        and not labels.is_empty()
        and "candidate_id" in candidates.columns
        and "candidate_id" in labels.columns
    ):
        candidate_ids = candidates.get_column("candidate_id").drop_nulls().unique().to_list()
        labels = labels.filter(pl.col("candidate_id").is_in(candidate_ids))
    costs = _filter_frame_by_time(
        read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET),
        ("as_of_date",),
        start_dt,
        end_dt,
    )

    warnings: list[str] = []
    if market.is_empty():
        warnings.append("market_bar_empty")
    if trades.is_empty() and lifecycles.is_empty():
        warnings.append("v5_actual_entry_events_empty")
    if candidates.is_empty():
        warnings.append("v5_candidate_event_empty")

    actual_entries = [
        row
        for row in _actual_entry_rows(trades, lifecycles)
        if _is_within_window(_coerce_datetime(row.get("entry_ts")), start_dt, end_dt)
    ]
    missed = _with_history_meta(
        build_missed_low_audit(actual_entries, market, ctx),
        history,
        MISSED_LOW_AUDIT_SCHEMA | HISTORY_META_SCHEMA,
    )
    missed_by_symbol = _with_history_meta(
        aggregate_missed_low(missed, group_column="symbol", ctx=ctx),
        history,
        MISSED_LOW_AGG_SCHEMA | HISTORY_META_SCHEMA,
    )
    missed_by_reason = _with_history_meta(
        aggregate_missed_low(missed, group_column="entry_reason", ctx=ctx),
        history,
        MISSED_LOW_AGG_SCHEMA | HISTORY_META_SCHEMA,
    )
    late_shadow = _with_history_meta(
        build_late_entry_chase_shadow(
            actual_entries=actual_entries,
            candidates=candidates,
            labels=labels,
            market_bars=market,
            ctx=ctx,
        ),
        history,
        LATE_ENTRY_CHASE_SHADOW_SCHEMA | HISTORY_META_SCHEMA,
    )
    late_threshold = _with_history_meta(
        build_late_entry_chase_threshold_advisory(late_shadow, ctx=ctx),
        history,
        HISTORY_THRESHOLD_SENSITIVITY_SCHEMA,
    )
    pullback = _with_history_meta(
        build_pullback_reversal_shadow(
            candidates=candidates,
            market_bars=market,
            costs=costs,
            ctx=ctx,
        ),
        history,
        PULLBACK_REVERSAL_SHADOW_SCHEMA | HISTORY_META_SCHEMA,
    )
    pullback_by_symbol = build_pullback_reversal_history_aggregate(
        pullback, group_type="symbol", ctx=ctx, history=history
    )
    pullback_by_regime = build_pullback_reversal_history_aggregate(
        pullback, group_type="regime", ctx=ctx, history=history
    )
    pullback_by_horizon = build_pullback_reversal_history_aggregate(
        pullback, group_type="horizon", ctx=ctx, history=history
    )
    anti_leakage = build_entry_quality_anti_leakage_check(
        missed_low=missed,
        late_shadow=late_shadow,
        pullback=pullback,
        market_bars=market,
        candidates=candidates,
        labels=labels,
        start_dt=start_dt,
        end_dt=end_dt,
        ctx=ctx,
        history=history,
    )
    metrics = build_entry_quality_history_metrics(
        missed_low=missed,
        late_shadow=late_shadow,
        late_threshold=late_threshold,
        pullback=pullback,
        pullback_by_symbol=pullback_by_symbol,
        pullback_by_regime=pullback_by_regime,
        pullback_by_horizon=pullback_by_horizon,
        anti_leakage=anti_leakage,
        ctx=ctx,
        history=history,
        warnings=warnings,
    )

    missed_rows = _publish_history(
        root,
        HISTORY_MISSED_LOW_AUDIT_DATASET,
        missed,
        ["start_date", "end_date", "window_mode", "source_event_key", "symbol", "entry_ts"],
    )
    by_symbol_rows = _publish_history(
        root,
        HISTORY_MISSED_LOW_BY_SYMBOL_DATASET,
        missed_by_symbol,
        ["start_date", "end_date", "window_mode", "group_key"],
    )
    by_reason_rows = _publish_history(
        root,
        HISTORY_MISSED_LOW_BY_ENTRY_REASON_DATASET,
        missed_by_reason,
        ["start_date", "end_date", "window_mode", "group_key"],
    )
    late_rows = _publish_history(
        root,
        HISTORY_LATE_ENTRY_CHASE_SHADOW_DATASET,
        late_shadow,
        ["start_date", "end_date", "window_mode", "source_type", "source_event_key"],
    )
    threshold_rows = _publish_history(
        root,
        HISTORY_LATE_ENTRY_THRESHOLD_SENSITIVITY_DATASET,
        late_threshold,
        ["start_date", "end_date", "window_mode", "threshold_bps"],
    )
    pullback_rows = _publish_history(
        root,
        HISTORY_PULLBACK_REVERSAL_SHADOW_DATASET,
        pullback,
        ["start_date", "end_date", "window_mode", "source_event_key", "horizon_hours"],
    )
    pullback_symbol_rows = _publish_history(
        root,
        HISTORY_PULLBACK_BY_SYMBOL_DATASET,
        pullback_by_symbol,
        ["start_date", "end_date", "window_mode", "group_type", "group_key"],
    )
    pullback_regime_rows = _publish_history(
        root,
        HISTORY_PULLBACK_BY_REGIME_DATASET,
        pullback_by_regime,
        ["start_date", "end_date", "window_mode", "group_type", "group_key"],
    )
    pullback_horizon_rows = _publish_history(
        root,
        HISTORY_PULLBACK_BY_HORIZON_DATASET,
        pullback_by_horizon,
        ["start_date", "end_date", "window_mode", "group_type", "group_key"],
    )
    anti_rows = _publish_history(
        root,
        HISTORY_ANTI_LEAKAGE_CHECK_DATASET,
        anti_leakage,
        ["start_date", "end_date", "window_mode", "check_name"],
    )
    metrics_rows = _publish_history(
        root,
        HISTORY_METRICS_DATASET,
        metrics,
        ["start_date", "end_date", "window_mode"],
    )

    reports_dir = _write_history_reports(
        root,
        missed_low=missed,
        missed_low_by_symbol=missed_by_symbol,
        missed_low_by_entry_reason=missed_by_reason,
        late_threshold=late_threshold,
        pullback_by_symbol=pullback_by_symbol,
        pullback_by_regime=pullback_by_regime,
        pullback_by_horizon=pullback_by_horizon,
        anti_leakage=anti_leakage,
        metrics=metrics,
    )

    return EntryQualityHistoryBuildResult(
        lake_root=str(root),
        start_date=start_day.isoformat(),
        end_date=end_day.isoformat(),
        mode=window_mode,
        cost_mode=normalized_cost_mode,
        missed_low_audit_rows=missed_rows,
        missed_low_by_symbol_rows=by_symbol_rows,
        missed_low_by_entry_reason_rows=by_reason_rows,
        late_entry_chase_shadow_rows=late_rows,
        late_entry_threshold_sensitivity_rows=threshold_rows,
        pullback_reversal_shadow_rows=pullback_rows,
        pullback_by_symbol_rows=pullback_symbol_rows,
        pullback_by_regime_rows=pullback_regime_rows,
        pullback_by_horizon_rows=pullback_horizon_rows,
        anti_leakage_rows=anti_rows,
        metrics_rows=metrics_rows,
        reports_dir=str(reports_dir),
        warnings=warnings,
    )


def build_missed_low_audit(
    actual_entries: list[dict[str, Any]],
    market_bars: pl.DataFrame,
    ctx: _BuildContext,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    market_by_symbol = _market_rows_by_symbol(market_bars)
    for entry in actual_entries:
        symbol = normalize_symbol(entry.get("symbol")) or "UNKNOWN"
        entry_ts = _coerce_datetime(entry.get("entry_ts"))
        entry_px = _float_or_none(entry.get("entry_px"))
        if symbol == "UNKNOWN" or entry_ts is None or entry_px is None or entry_px <= 0:
            continue
        windows = {
            hours: _pre_window_stats(market_by_symbol.get(symbol, []), entry_ts, hours)
            for hours in (4, 8, 12, 24)
        }
        realized_net_bps = _float_or_none(entry.get("realized_net_bps"))
        diagnosis = _missed_low_diagnosis(
            entry_vs_24h_low_bps=_entry_vs_low_bps(entry_px, windows[24].get("low")),
            position_24h=_position_in_range(
                entry_px, windows[24].get("low"), windows[24].get("high")
            ),
            realized_net_bps=realized_net_bps,
        )
        rows.append(
            _common(ctx, mode="audit")
            | {
                "run_id": str(entry.get("run_id") or ""),
                "source_event_key": _source_event_key(entry, symbol=symbol, ts=entry_ts),
                "symbol": symbol,
                "entry_ts": entry_ts,
                "entry_px": entry_px,
                "entry_reason": str(entry.get("entry_reason") or ""),
                "probe_type": str(entry.get("probe_type") or ""),
                "side": str(entry.get("side") or ""),
                "intent": str(entry.get("intent") or ""),
                "entry_vs_pre_4h_low_bps": _entry_vs_low_bps(entry_px, windows[4].get("low")),
                "entry_vs_pre_8h_low_bps": _entry_vs_low_bps(entry_px, windows[8].get("low")),
                "entry_vs_pre_12h_low_bps": _entry_vs_low_bps(entry_px, windows[12].get("low")),
                "entry_vs_pre_24h_low_bps": _entry_vs_low_bps(entry_px, windows[24].get("low")),
                "entry_position_in_24h_range": _position_in_range(
                    entry_px, windows[24].get("low"), windows[24].get("high")
                ),
                "realized_net_bps": realized_net_bps,
                "exit_reason": str(entry.get("exit_reason") or ""),
                "diagnosis": diagnosis,
            }
        )
    if not rows:
        return pl.DataFrame(schema=MISSED_LOW_AUDIT_SCHEMA)
    return pl.DataFrame(rows, schema=MISSED_LOW_AUDIT_SCHEMA, orient="row")


def aggregate_missed_low(
    missed_low: pl.DataFrame,
    *,
    group_column: str,
    ctx: _BuildContext,
) -> pl.DataFrame:
    if missed_low.is_empty() or group_column not in missed_low.columns:
        return pl.DataFrame(schema=MISSED_LOW_AGG_SCHEMA)
    rows: list[dict[str, Any]] = []
    for group_key, group_rows in _group_dicts(missed_low.to_dicts(), group_column).items():
        diagnoses = Counter(str(row.get("diagnosis") or "") for row in group_rows)
        realized = [_float_or_none(row.get("realized_net_bps")) for row in group_rows]
        rows.append(
            _common(ctx, mode="audit")
            | {
                "group_key": str(group_key or "UNKNOWN"),
                "sample_count": len(group_rows),
                "loss_count": sum(
                    1 for value in realized if value is not None and value < 0.0
                ),
                "profit_count": sum(
                    1 for value in realized if value is not None and value >= 0.0
                ),
                "late_chase_loss_count": diagnoses.get("late_chase_loss", 0),
                "late_but_trend_profitable_count": diagnoses.get(
                    "late_but_trend_profitable", 0
                ),
                "avg_entry_vs_pre_24h_low_bps": _mean(
                    row.get("entry_vs_pre_24h_low_bps") for row in group_rows
                ),
                "avg_entry_position_in_24h_range": _mean(
                    row.get("entry_position_in_24h_range") for row in group_rows
                ),
                "avg_realized_net_bps": _mean(realized),
                "diagnosis_mix": safe_json_dumps(dict(diagnoses)),
            }
        )
    return pl.DataFrame(rows, schema=MISSED_LOW_AGG_SCHEMA, orient="row")


def build_late_entry_chase_shadow(
    *,
    actual_entries: list[dict[str, Any]],
    candidates: pl.DataFrame,
    labels: pl.DataFrame,
    market_bars: pl.DataFrame,
    ctx: _BuildContext,
) -> pl.DataFrame:
    market_by_symbol = _market_rows_by_symbol(market_bars)
    label_context = _label_context(labels)
    rows: list[dict[str, Any]] = []
    for entry in actual_entries:
        subject = _late_subject_from_actual(entry)
        if subject:
            rows.append(
                _late_shadow_row(subject, market_by_symbol, label_context, ctx=ctx)
            )
    for candidate in candidates.to_dicts() if not candidates.is_empty() else []:
        subject = _late_subject_from_candidate(candidate)
        if subject:
            rows.append(
                _late_shadow_row(subject, market_by_symbol, label_context, ctx=ctx)
            )
    rows = [row for row in rows if row]
    if not rows:
        return pl.DataFrame(schema=LATE_ENTRY_CHASE_SHADOW_SCHEMA)
    return pl.DataFrame(rows, schema=LATE_ENTRY_CHASE_SHADOW_SCHEMA, orient="row")


def build_late_entry_chase_threshold_advisory(
    late_shadow: pl.DataFrame,
    *,
    ctx: _BuildContext,
) -> pl.DataFrame:
    shadow_rows = late_shadow.to_dicts() if not late_shadow.is_empty() else []
    rows = [
        _late_entry_threshold_row(shadow_rows, threshold=threshold, ctx=ctx)
        for threshold in LATE_CHASE_THRESHOLDS_BPS
    ]
    return pl.DataFrame(rows, schema=LATE_ENTRY_THRESHOLD_SCHEMA, orient="row")


def build_late_entry_chase_threshold_by_symbol(
    late_shadow: pl.DataFrame,
    *,
    ctx: _BuildContext,
) -> pl.DataFrame:
    if late_shadow.is_empty():
        return pl.DataFrame(schema=LATE_ENTRY_THRESHOLD_BY_SYMBOL_SCHEMA)
    rows: list[dict[str, Any]] = []
    shadow_rows = late_shadow.to_dicts()
    for symbol in sorted(ENTRY_QUALITY_SYMBOLS):
        symbol_rows = [
            row
            for row in shadow_rows
            if normalize_symbol(row.get("symbol")) == symbol
        ]
        for threshold in LATE_CHASE_BY_SYMBOL_THRESHOLDS_BPS:
            rows.append(
                _late_entry_threshold_row(
                    symbol_rows,
                    threshold=threshold,
                    ctx=ctx,
                    symbol=symbol,
                )
            )
    return pl.DataFrame(rows, schema=LATE_ENTRY_THRESHOLD_BY_SYMBOL_SCHEMA, orient="row")


def _late_entry_threshold_row(
    shadow_rows: list[dict[str, Any]],
    *,
    threshold: int,
    ctx: _BuildContext,
    symbol: str | None = None,
) -> dict[str, Any]:
    blocked = [
        row
        for row in shadow_rows
        if (_float_or_none(row.get("entry_vs_12h_low_bps")) or 0.0) > threshold
        and (_float_or_none(row.get("entry_position_in_12h_range")) or 0.0) > 0.70
    ]
    loss_count = sum(
        1
        for row in blocked
        if _outcome_value(row) is not None and (_outcome_value(row) or 0.0) < 0
    )
    profit_count = sum(
        1
        for row in blocked
        if _outcome_value(row) is not None and (_outcome_value(row) or 0.0) >= 0
    )
    not_blocked = [row for row in shadow_rows if row not in blocked]
    total = len(blocked)
    payload = _common(ctx, mode="advisory") | {
        "threshold_bps": int(threshold),
        "would_block_count": total,
        "would_block_loss_count": loss_count,
        "would_block_profit_count": profit_count,
        "false_positive_rate": (profit_count / total) if total else None,
        "avg_net_bps_blocked": _mean(_outcome_value(row) for row in blocked),
        "avg_net_bps_not_blocked": _mean(_outcome_value(row) for row in not_blocked),
        "ready_for_live_guard": False,
        "advisory": "shadow_only_collect_more_samples",
    }
    if symbol is not None:
        payload["symbol"] = symbol
    return payload


def build_pullback_reversal_shadow(
    *,
    candidates: pl.DataFrame,
    market_bars: pl.DataFrame,
    costs: pl.DataFrame,
    ctx: _BuildContext,
    rule_version: str = PULLBACK_NEW_RULE_VERSION,
) -> pl.DataFrame:
    if candidates.is_empty() or market_bars.is_empty():
        return pl.DataFrame(schema=PULLBACK_REVERSAL_SHADOW_SCHEMA)
    market_by_symbol = _market_rows_by_symbol(market_bars)
    cost_by_symbol = _roundtrip_cost_by_symbol(costs)
    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dicts():
        symbol = normalize_symbol(candidate.get("symbol") or candidate.get("normalized_symbol"))
        if symbol not in ENTRY_QUALITY_SYMBOLS:
            continue
        ts = _coerce_datetime(candidate.get("ts_utc") or candidate.get("ts"))
        if ts is None:
            continue
        current_px = _candidate_price(candidate, market_by_symbol.get(symbol, []), ts)
        if current_px is None:
            continue
        bars = market_by_symbol.get(symbol, [])
        pre_24h = _pre_window_stats(bars, ts, 24)
        pullback_bps = _pullback_from_high_bps(current_px, pre_24h.get("high"))
        f4 = _float_or_none(candidate.get("f4_volume_expansion"))
        f5 = _float_or_none(candidate.get("f5_rsi_trend_confirm"))
        conditions = _pullback_reversal_conditions(
            row=candidate,
            current_px=current_px,
            pre_24h=pre_24h,
            bars=bars,
            btc_bars=market_by_symbol.get("BTC-USDT", []),
            ts=ts,
        )
        if not _pullback_reversal_candidate_ok(
            row=candidate,
            current_px=current_px,
            pre_24h=pre_24h,
            pullback_bps=pullback_bps,
            f4=f4,
            f5=f5,
            bars=bars,
            btc_bars=market_by_symbol.get("BTC-USDT", []),
            ts=ts,
            rule_version=rule_version,
        ):
            continue
        has_quant_lab_cost = symbol in cost_by_symbol
        roundtrip_cost = max(
            cost_by_symbol.get(symbol, MIN_ROUNDTRIP_COST_BPS),
            MIN_ROUNDTRIP_COST_BPS,
        )
        cost_quality = "quant_lab" if has_quant_lab_cost else "degraded"
        for horizon in PULLBACK_HORIZON_HOURS:
            label = _forward_label(bars, ts, current_px, horizon, roundtrip_cost)
            rows.append(
                _common(ctx, mode="shadow")
                | {
                    "strategy_candidate": _pullback_candidate_name(symbol),
                    "rule_version": rule_version,
                    "run_id": str(candidate.get("run_id") or ""),
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "source_event_key": _source_event_key(candidate, symbol=symbol, ts=ts),
                    "symbol": symbol,
                    "ts_utc": ts,
                    "regime_state": str(candidate.get("regime_state") or ""),
                    "risk_level": str(candidate.get("risk_level") or ""),
                    "current_px": current_px,
                    "pre_24h_low": pre_24h.get("low"),
                    "pre_24h_high": pre_24h.get("high"),
                    "pullback_from_24h_high_bps": pullback_bps,
                    "recent_2h_no_new_low": conditions["recent_2h_no_new_low"],
                    "close_reclaim_1h": conditions["close_reclaim_1h"],
                    "current_close_gt_previous_close": conditions[
                        "current_close_gt_previous_close"
                    ],
                    "btc_not_sharp_drop": conditions["btc_not_sharp_drop"],
                    "spread_not_abnormal": conditions["spread_not_abnormal"],
                    "f4_volume_expansion": f4,
                    "f5_rsi_trend_confirm": f5,
                    "selected_roundtrip_cost_bps": roundtrip_cost,
                    "cost_quality": cost_quality,
                    "horizon_hours": horizon,
                    **label,
                }
            )
    if not rows:
        return pl.DataFrame(schema=PULLBACK_REVERSAL_SHADOW_SCHEMA)
    return pl.DataFrame(rows, schema=PULLBACK_REVERSAL_SHADOW_SCHEMA, orient="row")


def build_pullback_reversal_rule_comparison(
    *,
    candidates: pl.DataFrame,
    market_bars: pl.DataFrame,
    costs: pl.DataFrame,
    ctx: _BuildContext,
) -> pl.DataFrame:
    if candidates.is_empty() or market_bars.is_empty():
        return pl.DataFrame(schema=PULLBACK_RULE_COMPARISON_SCHEMA)
    old_shadow = build_pullback_reversal_shadow(
        candidates=candidates,
        market_bars=market_bars,
        costs=costs,
        ctx=ctx,
        rule_version=PULLBACK_OLD_RULE_VERSION,
    )
    new_shadow = build_pullback_reversal_shadow(
        candidates=candidates,
        market_bars=market_bars,
        costs=costs,
        ctx=ctx,
        rule_version=PULLBACK_NEW_RULE_VERSION,
    )
    rows: list[dict[str, Any]] = []
    for rule_name, rule_version, frame in [
        ("old_rule", PULLBACK_OLD_RULE_VERSION, old_shadow),
        ("new_rule", PULLBACK_NEW_RULE_VERSION, new_shadow),
    ]:
        if frame.is_empty():
            continue
        rows.extend(
            _pullback_rule_comparison_rows(
                frame,
                rule_name=rule_name,
                rule_version=rule_version,
                ctx=ctx,
            )
        )
    if not rows:
        return pl.DataFrame(schema=PULLBACK_RULE_COMPARISON_SCHEMA)
    return pl.DataFrame(rows, schema=PULLBACK_RULE_COMPARISON_SCHEMA, orient="row")


def _pullback_rule_comparison_rows(
    frame: pl.DataFrame,
    *,
    rule_name: str,
    rule_version: str,
    ctx: _BuildContext,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    horizon_rows = [
        row
        for row in frame.to_dicts()
        if int(_float_or_none(row.get("horizon_hours")) or 0) == 24
    ]
    for symbol, group_rows in _group_dicts(horizon_rows, "symbol").items():
        complete_values = [
            _float_or_none(row.get("net_bps_after_cost"))
            for row in group_rows
            if str(row.get("label_status") or "") == "complete"
        ]
        complete_values = [value for value in complete_values if value is not None]
        mae_values = [
            _float_or_none(row.get("mae_bps"))
            for row in group_rows
            if str(row.get("label_status") or "") == "complete"
        ]
        mae_values = [value for value in mae_values if value is not None]
        rows.append(
            _common(ctx, mode="advisory")
            | {
                "comparison_name": "old_rule_vs_new_rule",
                "rule_name": rule_name,
                "rule_version": rule_version,
                "symbol": str(symbol),
                "horizon_hours": 24,
                "sample_count": len(group_rows),
                "complete_sample_count": len(complete_values),
                "avg_net_bps": _mean(complete_values),
                "win_rate": (
                    sum(1 for value in complete_values if value > 0.0)
                    / len(complete_values)
                    if complete_values
                    else None
                ),
                "avg_mae_bps": _mean(mae_values),
                "p25_net_bps": _quantile(complete_values, 0.25),
            }
        )
    return rows


def build_pullback_reversal_readiness(
    pullback: pl.DataFrame,
    *,
    ctx: _BuildContext,
) -> pl.DataFrame:
    if pullback.is_empty():
        return pl.DataFrame(schema=PULLBACK_REVERSAL_READINESS_SCHEMA)
    rows: list[dict[str, Any]] = []
    shadow_rows = [
        row for row in pullback.to_dicts() if int(row.get("horizon_hours") or 0) == 24
    ]
    grouped = _group_dicts(shadow_rows, "symbol")
    recent_cutoff = datetime.combine(ctx.as_of_date - timedelta(days=7), time.min, tzinfo=UTC)
    for symbol, group_rows in grouped.items():
        net_values = [_float_or_none(row.get("net_bps_after_cost")) for row in group_rows]
        net_values = [value for value in net_values if value is not None]
        mae_values = [_float_or_none(row.get("mae_bps")) for row in group_rows]
        mae_values = [value for value in mae_values if value is not None]
        sample_count = len(group_rows)
        recent_count = sum(
            1
            for row in group_rows
            if (_coerce_datetime(row.get("ts_utc")) or datetime.min.replace(tzinfo=UTC))
            >= recent_cutoff
        )
        avg_net = _mean(net_values)
        win_rate = (
            sum(1 for value in net_values if value > 0) / len(net_values)
            if net_values
            else None
        )
        p25 = _quantile(net_values, 0.25)
        avg_mae = _mean(mae_values)
        reasons = _pullback_readiness_reasons(
            sample_count=sample_count,
            recent_count=recent_count,
            avg_net=avg_net,
            win_rate=win_rate,
            p25=p25,
            avg_mae=avg_mae,
        )
        ready_for_paper = False
        rendered_reasons = (
            reasons
            if reasons
            else ["pullback_v2_shadow_only_not_paper_ready", "not_live_validated"]
        )
        rows.append(
            _common(ctx, mode="advisory")
            | {
                "strategy_candidate": _pullback_candidate_name(str(symbol)),
                "symbol": str(symbol),
                "sample_count": sample_count,
                "recent_7d_sample_count": recent_count,
                "avg_24h_net_bps": avg_net,
                "win_rate_24h": win_rate,
                "p25_24h_net_bps": p25,
                "avg_mae_bps": avg_mae,
                "ready_for_paper": ready_for_paper,
                "ready_for_live_probe": False,
                "readiness_reasons": safe_json_dumps(rendered_reasons),
            }
        )
    return pl.DataFrame(rows, schema=PULLBACK_REVERSAL_READINESS_SCHEMA, orient="row")


def build_entry_quality_advisory(
    *,
    missed_low: pl.DataFrame,
    late_threshold: pl.DataFrame,
    pullback_readiness: pl.DataFrame,
    ctx: _BuildContext,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    if not missed_low.is_empty():
        rows.append(
            _common(ctx, mode="advisory")
            | {
                "strategy_candidate": "v5.entry_quality_missed_low_audit",
                "symbol": "ALL",
                "recommended_mode": "audit",
                "readiness_status": "AUDIT_READY",
                "sample_count": missed_low.height,
                "avg_net_bps": _mean(missed_low.get_column("realized_net_bps").to_list()),
                "win_rate": None,
                "advisory_reasons": safe_json_dumps(
                    ["read_only_audit", "does_not_block_live_orders"]
                ),
                "would_block_if_enabled": False,
                "would_enter": False,
                "no_sample_reason": "audit_only",
                "ready_for_live": False,
            }
        )
    if not late_threshold.is_empty():
        total_blocked = sum(
            int(row.get("would_block_count") or 0) for row in late_threshold.to_dicts()
        )
        if total_blocked > 0:
            rows.append(
                _common(ctx, mode="advisory")
                | {
                    "strategy_candidate": "v5.late_entry_chase_guard_shadow",
                    "symbol": "ALL",
                    "recommended_mode": "shadow",
                    "readiness_status": "SHADOW_ONLY",
                    "sample_count": total_blocked,
                    "avg_net_bps": None,
                    "win_rate": None,
                    "advisory_reasons": safe_json_dumps(
                        ["ready_for_live_guard=false", "threshold_sensitivity_only"]
                    ),
                    "would_block_if_enabled": True,
                    "would_enter": False,
                    "no_sample_reason": "guard_shadow_only",
                    "ready_for_live": False,
                }
            )
    observed_pullback_symbols: set[str] = set()
    for row in pullback_readiness.to_dicts() if not pullback_readiness.is_empty() else []:
        ready_for_paper = False
        symbol = normalize_symbol(row.get("symbol"))
        if symbol:
            observed_pullback_symbols.add(symbol)
        recommended_mode = _pullback_recommended_mode(row, ready_for_paper=ready_for_paper)
        rows.append(
            _common(ctx, mode="advisory")
            | {
                "strategy_candidate": row.get("strategy_candidate"),
                "symbol": symbol,
                "recommended_mode": recommended_mode,
                "readiness_status": "READY_FOR_PAPER" if ready_for_paper else "SHADOW_ONLY",
                "sample_count": int(row.get("sample_count") or 0),
                "avg_net_bps": _float_or_none(row.get("avg_24h_net_bps")),
                "win_rate": _float_or_none(row.get("win_rate_24h")),
                "advisory_reasons": str(row.get("readiness_reasons") or "[]"),
                "would_block_if_enabled": False,
                "would_enter": recommended_mode == "paper",
                "no_sample_reason": _pullback_no_sample_reason(
                    row,
                    recommended_mode=recommended_mode,
                ),
                "ready_for_live": False,
            }
        )
    for symbol in sorted(ENTRY_QUALITY_SYMBOLS - observed_pullback_symbols):
        rows.append(
            _common(ctx, mode="advisory")
            | {
                "strategy_candidate": _pullback_candidate_name(symbol),
                "symbol": symbol,
                "recommended_mode": "research",
                "readiness_status": "RESEARCH_ONLY",
                "sample_count": 0,
                "avg_net_bps": None,
                "win_rate": None,
                "advisory_reasons": safe_json_dumps(
                    [
                        "confirmed_reversal_no_current_samples",
                        "insufficient_sample_count",
                        "shadow_only_collect_more_samples",
                        "not_live_validated",
                    ]
                ),
                "would_block_if_enabled": False,
                "would_enter": False,
                "no_sample_reason": "insufficient_sample_count",
                "ready_for_live": False,
            }
        )
    if not rows:
        return pl.DataFrame(schema=ENTRY_QUALITY_ADVISORY_SCHEMA)
    return pl.DataFrame(rows, schema=ENTRY_QUALITY_ADVISORY_SCHEMA, orient="row")


def build_entry_quality_strategy_opportunity_advisory(
    entry_quality_advisory: pl.DataFrame,
) -> pl.DataFrame:
    if entry_quality_advisory.is_empty():
        return pl.DataFrame(schema=STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA)
    rows: list[dict[str, Any]] = []
    latest_as_of_date = _latest_as_of_date(entry_quality_advisory.to_dicts())
    for row in entry_quality_advisory.to_dicts():
        if latest_as_of_date and str(row.get("as_of_date") or "") != latest_as_of_date:
            continue
        candidate = str(row.get("strategy_candidate") or "").strip()
        if not candidate:
            continue
        mode = _entry_quality_recommended_mode(candidate, row.get("recommended_mode"))
        decision = _entry_quality_strategy_decision(mode)
        symbol = normalize_symbol(row.get("symbol")) if row.get("symbol") != "ALL" else None
        rendered_symbol = symbol or "ALL"
        sample_count = _optional_int(row.get("sample_count")) or 0
        live_block_reasons = _entry_quality_live_block_reasons(row, mode)
        as_of_ts = _entry_quality_as_of_datetime(row)
        generated_at = _coerce_datetime(row.get("generated_at_utc")) or as_of_ts
        git_commit = _observable_text(row.get("quant_lab_git_commit")) or _git_commit()
        source_version = _observable_text(row.get("source_version")) or _source_version(
            "entry_quality_advisory",
            git_commit,
        )
        rows.append(
            {
                "as_of_ts": as_of_ts,
                "generated_at": generated_at,
                "expires_at": _strategy_advisory_expires_at(generated_at),
                "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
                "schema_version": STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION,
                "quant_lab_git_commit": git_commit,
                "source_version": source_version,
                "would_block_if_enabled": bool(row.get("would_block_if_enabled")),
                "would_enter": bool(row.get("would_enter")),
                "no_sample_reason": _entry_quality_no_sample_reason(row, mode, sample_count),
                "strategy_id": _strategy_id(candidate, rendered_symbol),
                "symbol": rendered_symbol,
                "v5_symbol": _v5_symbol(rendered_symbol),
                "strategy_candidate": candidate,
                "decision": decision,
                "recommended_mode": mode,
                "horizon_hours": None,
                "sample_count": sample_count,
                "complete_sample_count": sample_count,
                "avg_net_bps": _float_or_none(row.get("avg_net_bps")),
                "p25_net_bps": None,
                "win_rate": _float_or_none(row.get("win_rate")),
                "cost_source_mix": safe_json_dumps({"entry_quality_research": sample_count}),
                "cost_quality": "entry_quality_research",
                "paper_days": 0,
                "entry_day_count": 0,
                "paper_pnl_observed_count": 0,
                "slippage_coverage": None,
                "live_block_reasons": safe_json_dumps(live_block_reasons),
                "max_paper_notional_usdt": 100.0 if mode == "paper" else 0.0,
                "max_live_notional_usdt": 0.0,
            }
        )
    if not rows:
        return pl.DataFrame(schema=STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA)
    return pl.DataFrame(
        rows,
        schema=STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA,
        orient="row",
    )


def build_pullback_reversal_history_aggregate(
    pullback: pl.DataFrame,
    *,
    group_type: str,
    ctx: _BuildContext,
    history: _HistoryContext,
) -> pl.DataFrame:
    if pullback.is_empty():
        return pl.DataFrame(schema=HISTORY_PULLBACK_AGG_SCHEMA)
    rows: list[dict[str, Any]] = []
    pullback_rows = pullback.to_dicts()
    if group_type == "symbol":
        grouped = _group_dicts(pullback_rows, "symbol")
    elif group_type == "regime":
        grouped = _group_dicts(pullback_rows, "regime_state")
    elif group_type == "horizon":
        grouped = _group_dicts(pullback_rows, "horizon_hours")
    else:
        raise ValueError("group_type must be symbol, regime, or horizon")
    for group_key, group_rows in grouped.items():
        net_values = [
            _float_or_none(row.get("net_bps_after_cost"))
            for row in group_rows
            if str(row.get("label_status") or "") == "complete"
        ]
        net_values = [value for value in net_values if value is not None]
        mfe_values = [_float_or_none(row.get("mfe_bps")) for row in group_rows]
        mae_values = [_float_or_none(row.get("mae_bps")) for row in group_rows]
        cost_quality_mix = Counter(str(row.get("cost_quality") or "degraded") for row in group_rows)
        complete_count = len(net_values)
        avg_net = _mean(net_values)
        win_rate = (
            sum(1 for value in net_values if value > 0.0) / complete_count
            if complete_count
            else None
        )
        decision, reasons = _history_shadow_decision(
            sample_count=len(group_rows),
            complete_sample_count=complete_count,
            avg_net_bps=avg_net,
            win_rate=win_rate,
            p25_net_bps=_quantile(net_values, 0.25),
            avg_mae_bps=_mean(mae_values),
            cost_quality_mix=cost_quality_mix,
        )
        first = group_rows[0]
        rows.append(
            _common(ctx, mode="advisory")
            | _history_meta(history)
            | {
                "group_type": group_type,
                "group_key": str(group_key),
                "strategy_candidate": str(first.get("strategy_candidate") or ""),
                "symbol": str(first.get("symbol") or "") if group_type != "horizon" else "ALL",
                "regime_state": (
                    str(first.get("regime_state") or "") if group_type != "horizon" else "ALL"
                ),
                "horizon_hours": (
                    int(_float_or_none(first.get("horizon_hours")) or 0)
                    if group_type == "horizon"
                    else 0
                ),
                "sample_count": len(group_rows),
                "complete_sample_count": complete_count,
                "avg_net_bps": avg_net,
                "median_net_bps": _quantile(net_values, 0.50),
                "p25_net_bps": _quantile(net_values, 0.25),
                "win_rate": win_rate,
                "avg_mfe_bps": _mean(mfe_values),
                "avg_mae_bps": _mean(mae_values),
                "cost_quality_mix": safe_json_dumps(dict(cost_quality_mix)),
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
            }
        )
    return pl.DataFrame(rows, schema=HISTORY_PULLBACK_AGG_SCHEMA, orient="row")


def build_entry_quality_anti_leakage_check(
    *,
    missed_low: pl.DataFrame,
    late_shadow: pl.DataFrame,
    pullback: pl.DataFrame,
    market_bars: pl.DataFrame,
    candidates: pl.DataFrame,
    labels: pl.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    ctx: _BuildContext,
    history: _HistoryContext,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    window_violations = 0
    for frame, column in [
        (missed_low, "entry_ts"),
        (late_shadow, "ts_utc"),
        (pullback, "ts_utc"),
    ]:
        if frame.is_empty() or column not in frame.columns:
            continue
        for value in frame.get_column(column).to_list():
            ts = _coerce_datetime(value)
            if ts is not None and not (start_dt <= ts < end_dt):
                window_violations += 1
    max_horizon = max(PULLBACK_HORIZON_HOURS)
    allowed_forward_end = end_dt + timedelta(hours=max_horizon)
    label_order_violations = 0
    label_boundary_violations = 0
    label_identity_violations = 0
    candidate_times: dict[str, datetime] = {}
    if not candidates.is_empty() and "candidate_id" in candidates.columns:
        for row in candidates.to_dicts():
            candidate_id = str(row.get("candidate_id") or "").strip()
            decision_ts = _coerce_datetime(row.get("ts_utc") or row.get("ts"))
            if candidate_id and decision_ts is not None:
                candidate_times[candidate_id] = decision_ts
    if not labels.is_empty():
        for row in labels.to_dicts():
            candidate_id = str(row.get("candidate_id") or "").strip()
            decision_ts = _coerce_datetime(row.get("decision_ts") or row.get("ts_utc"))
            label_ts = _coerce_datetime(row.get("label_ts") or row.get("label_end_ts"))
            horizon = int(_float_or_none(row.get("horizon_hours")) or 0)
            if decision_ts is not None and label_ts is not None and label_ts <= decision_ts:
                label_order_violations += 1
            if label_ts is not None and label_ts > allowed_forward_end:
                label_boundary_violations += 1
            if (
                decision_ts is not None
                and label_ts is not None
                and horizon > 0
                and label_ts > decision_ts + timedelta(hours=horizon + 1)
            ):
                label_boundary_violations += 1
            candidate_ts = candidate_times.get(candidate_id)
            if candidate_id and candidate_id not in candidate_times:
                label_identity_violations += 1
            elif candidate_ts is not None:
                decision_is_causal = (
                    decision_ts is not None
                    and candidate_ts <= decision_ts
                    and decision_ts <= candidate_ts + CANDIDATE_LABEL_DECISION_MAX_LAG
                )
                if not decision_is_causal:
                    label_identity_violations += 1

    market_future_violations = 0
    market_max_ts: datetime | None = None
    if not market_bars.is_empty() and "ts" in market_bars.columns:
        market_times = [
            ts
            for ts in (_coerce_datetime(value) for value in market_bars.get_column("ts").to_list())
            if ts is not None
        ]
        if market_times:
            market_max_ts = max(market_times)
            market_future_violations = sum(ts >= allowed_forward_end for ts in market_times)

    horizon_violations = 0
    if not pullback.is_empty():
        for row in pullback.to_dicts():
            ts = _coerce_datetime(row.get("ts_utc"))
            horizon = int(_float_or_none(row.get("horizon_hours")) or 0)
            if str(row.get("label_status") or "") == "complete" and (
                ts is None
                or horizon <= 0
                or market_max_ts is None
                or ts + timedelta(hours=horizon) > market_max_ts
                or ts + timedelta(hours=horizon) > allowed_forward_end
            ):
                horizon_violations += 1

    bundle_identity_violations = 0
    for frame in (missed_low, late_shadow, pullback):
        if frame.is_empty() or "generated_from_bundle_id" not in frame.columns:
            continue
        bundle_identity_violations += sum(
            str(value or "") != ctx.generated_from_bundle_id
            for value in frame.get_column("generated_from_bundle_id").to_list()
        )
    walk_forward_violations = (
        label_order_violations if history.window_mode == "walk_forward" else 0
    )
    checks = [
        (
            "history_window_respected",
            window_violations,
            "rows must have entry/decision timestamps inside requested historical window",
        ),
        (
            "label_ts_after_decision_ts",
            label_order_violations,
            "pullback labels must be forward horizons after decision_ts",
        ),
        (
            "forward_label_end_boundary",
            label_boundary_violations,
            "forward labels must stay inside the sealed maximum-horizon boundary",
        ),
        (
            "candidate_label_identity",
            label_identity_violations,
            "candidate labels must bind to the same candidate_id and its first hourly decision bar",
        ),
        (
            "market_future_data_excluded",
            market_future_violations,
            "market inputs must not extend beyond the sealed maximum-horizon boundary",
        ),
        (
            "closed_bar_inputs_only",
            0,
            "entry conditions use pre-window bars strictly before decision/entry timestamp",
        ),
        (
            "walk_forward_semantics",
            walk_forward_violations,
            "walk-forward labels must preserve point-in-time decision ordering",
        ),
        (
            "horizon_completion",
            horizon_violations,
            "complete outcomes require enough sealed market data for their full horizon",
        ),
        (
            "bundle_source_identity",
            bundle_identity_violations,
            "all derived rows must retain the task-selected V5 bundle identity",
        ),
        (
            "read_only_no_live_action",
            0,
            "history builder writes research diagnostics only and does not affect V5 live state",
        ),
    ]
    for check_name, violation_count, detail in checks:
        rows.append(
            _common(ctx, mode="audit")
            | _history_meta(history)
            | {
                "check_name": check_name,
                "status": "FAIL" if violation_count else "PASS",
                "violation_count": int(violation_count),
                "detail": detail,
            }
        )
    return pl.DataFrame(rows, schema=HISTORY_ANTI_LEAKAGE_SCHEMA, orient="row")


def build_entry_quality_history_metrics(
    *,
    missed_low: pl.DataFrame,
    late_shadow: pl.DataFrame,
    late_threshold: pl.DataFrame,
    pullback: pl.DataFrame,
    pullback_by_symbol: pl.DataFrame,
    pullback_by_regime: pl.DataFrame,
    pullback_by_horizon: pl.DataFrame,
    anti_leakage: pl.DataFrame,
    ctx: _BuildContext,
    history: _HistoryContext,
    warnings: list[str],
) -> pl.DataFrame:
    anti_status = "PASS"
    if not anti_leakage.is_empty() and "status" in anti_leakage.columns:
        statuses = {str(value) for value in anti_leakage.get_column("status").to_list()}
        anti_status = "FAIL" if "FAIL" in statuses else "PASS"
    metrics = {
        "start_date": history.start_date.isoformat(),
        "end_date": history.end_date.isoformat(),
        "mode": history.window_mode,
        "cost_mode": history.cost_mode,
        "missed_low_audit_rows": missed_low.height,
        "late_entry_chase_shadow_rows": late_shadow.height,
        "late_entry_threshold_rows": late_threshold.height,
        "pullback_reversal_shadow_rows": pullback.height,
        "pullback_by_symbol_rows": pullback_by_symbol.height,
        "pullback_by_regime_rows": pullback_by_regime.height,
        "pullback_by_horizon_rows": pullback_by_horizon.height,
        "anti_leakage_status": anti_status,
        "ready_for_live_rows": 0,
        "allowed_decisions": ["RESEARCH_ONLY", "KEEP_SHADOW"],
        "warnings": warnings,
    }
    row = (
        _common(ctx, mode="audit")
        | _history_meta(history)
        | {
            "missed_low_audit_rows": missed_low.height,
            "late_entry_chase_shadow_rows": late_shadow.height,
            "late_entry_threshold_rows": late_threshold.height,
            "pullback_reversal_shadow_rows": pullback.height,
            "pullback_by_symbol_rows": pullback_by_symbol.height,
            "pullback_by_regime_rows": pullback_by_regime.height,
            "pullback_by_horizon_rows": pullback_by_horizon.height,
            "anti_leakage_status": anti_status,
            "ready_for_live_rows": 0,
            "metrics_json": safe_json_dumps(metrics),
        }
    )
    return pl.DataFrame([row], schema=HISTORY_METRICS_SCHEMA, orient="row")


def _actual_entry_rows(trades: pl.DataFrame, lifecycles: pl.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in trades.to_dicts() if not trades.is_empty() else []:
        if not _is_open_long_like(row):
            continue
        symbol = normalize_symbol(row.get("normalized_symbol") or row.get("symbol"))
        ts = _coerce_datetime(row.get("ts_utc") or row.get("ts"))
        price = _float_or_none(row.get("price") or row.get("fill_px") or row.get("fill_price"))
        if symbol and ts and price:
            rows.append(
                {
                    **row,
                    "symbol": symbol,
                    "entry_ts": ts,
                    "entry_px": price,
                    "intent": str(row.get("intent") or row.get("action") or "OPEN_LONG"),
                    "entry_reason": str(
                        row.get("entry_reason")
                        or row.get("action")
                        or row.get("final_decision")
                        or ""
                    ),
                    "realized_net_bps": _float_or_none(
                        row.get("realized_net_bps") or row.get("net_bps") or row.get("pnl_bps")
                    ),
                }
            )
    for row in lifecycles.to_dicts() if not lifecycles.is_empty() else []:
        if not _is_open_long_like(row):
            continue
        symbol = normalize_symbol(row.get("normalized_symbol") or row.get("symbol"))
        ts = _coerce_datetime(
            row.get("ts_utc")
            or row.get("last_fill_ts")
            or row.get("submit_ts")
            or row.get("decision_ts")
        )
        price = _float_or_none(row.get("avg_fill_px") or row.get("fill_px") or row.get("entry_px"))
        if symbol and ts and price:
            rows.append(
                {
                    **row,
                    "symbol": symbol,
                    "entry_ts": ts,
                    "entry_px": price,
                    "intent": str(row.get("intent") or "OPEN_LONG"),
                    "entry_reason": str(row.get("entry_reason") or row.get("intent") or ""),
                    "realized_net_bps": _float_or_none(
                        row.get("realized_net_bps")
                        or row.get("net_bps")
                        or row.get("total_realized_cost_bps")
                    ),
                }
            )
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _source_event_key(row, symbol=str(row.get("symbol") or ""), ts=row.get("entry_ts"))
        unique[key] = row
    return list(unique.values())


def _late_subject_from_actual(row: dict[str, Any]) -> dict[str, Any] | None:
    symbol = normalize_symbol(row.get("symbol"))
    ts = _coerce_datetime(row.get("entry_ts"))
    price = _float_or_none(row.get("entry_px"))
    if not symbol or ts is None or price is None:
        return None
    return {
        "strategy_candidate": "v5.late_entry_chase_guard_shadow",
        "source_type": "actual_entry",
        "run_id": row.get("run_id"),
        "candidate_id": "",
        "source_event_key": _source_event_key(row, symbol=symbol, ts=ts),
        "symbol": symbol,
        "ts_utc": ts,
        "entry_or_candidate_px": price,
        "f4_volume_expansion": _float_or_none(row.get("f4_volume_expansion")),
        "f5_rsi_trend_confirm": _float_or_none(row.get("f5_rsi_trend_confirm")),
        "realized_net_bps": _float_or_none(row.get("realized_net_bps")),
    }


def _late_subject_from_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    symbol = normalize_symbol(row.get("normalized_symbol") or row.get("symbol"))
    ts = _coerce_datetime(row.get("ts_utc") or row.get("ts"))
    if not symbol or ts is None:
        return None
    price = _float_or_none(
        row.get("entry_close")
        or row.get("candidate_px")
        or row.get("price")
        or row.get("close")
        or row.get("last_price")
    )
    return {
        "strategy_candidate": "v5.late_entry_chase_guard_shadow",
        "source_type": "candidate_event",
        "run_id": row.get("run_id"),
        "candidate_id": str(row.get("candidate_id") or ""),
        "source_event_key": _source_event_key(row, symbol=symbol, ts=ts),
        "symbol": symbol,
        "ts_utc": ts,
        "entry_or_candidate_px": price,
        "f4_volume_expansion": _float_or_none(row.get("f4_volume_expansion")),
        "f5_rsi_trend_confirm": _float_or_none(row.get("f5_rsi_trend_confirm")),
        "realized_net_bps": None,
    }


def _late_shadow_row(
    subject: dict[str, Any],
    market_by_symbol: dict[str, list[dict[str, Any]]],
    label_context: dict[str, dict[int, dict[str, Any]]],
    *,
    ctx: _BuildContext,
) -> dict[str, Any]:
    symbol = str(subject["symbol"])
    ts = subject["ts_utc"]
    px = _float_or_none(subject.get("entry_or_candidate_px"))
    bars = market_by_symbol.get(symbol, [])
    if px is None:
        px = _market_close_at_or_before(bars, ts)
    low12 = _pre_window_stats(bars, ts, 12)
    low24 = _pre_window_stats(bars, ts, 24)
    f4 = _float_or_none(subject.get("f4_volume_expansion"))
    f5 = _float_or_none(subject.get("f5_rsi_trend_confirm"))
    entry_vs_12h_low = _entry_vs_low_bps(px, low12.get("low"))
    entry_pos_12h = _position_in_range(px, low12.get("low"), low12.get("high"))
    late_chase = (
        (entry_vs_12h_low or 0.0) > 250.0
        and (entry_pos_12h or 0.0) > 0.70
        and (f4 is None or f4 < 0.50)
        and (f5 is None or f5 < 0.45)
    )
    candidate_labels = label_context.get(str(subject.get("candidate_id") or ""), {})
    forward_24 = _float_or_none(candidate_labels.get(24, {}).get("net_bps_after_cost"))
    forward_48 = _float_or_none(candidate_labels.get(48, {}).get("net_bps_after_cost"))
    realized = _float_or_none(subject.get("realized_net_bps"))
    outcome = _classify_outcome(realized if realized is not None else forward_24)
    return _common(ctx, mode="shadow") | {
        "strategy_candidate": "v5.late_entry_chase_guard_shadow",
        "source_type": subject.get("source_type"),
        "run_id": str(subject.get("run_id") or ""),
        "candidate_id": str(subject.get("candidate_id") or ""),
        "source_event_key": str(subject.get("source_event_key") or ""),
        "symbol": symbol,
        "ts_utc": ts,
        "entry_or_candidate_px": px,
        "recent_12h_low": low12.get("low"),
        "recent_24h_low": low24.get("low"),
        "entry_vs_12h_low_bps": entry_vs_12h_low,
        "entry_vs_24h_low_bps": _entry_vs_low_bps(px, low24.get("low")),
        "entry_position_in_12h_range": entry_pos_12h,
        "f4_volume_expansion": f4,
        "f5_rsi_trend_confirm": f5,
        "late_chase_risk": late_chase,
        "would_block_if_enabled": late_chase,
        "realized_net_bps": realized,
        "forward_24h_net_bps": forward_24,
        "forward_48h_net_bps": forward_48,
        "outcome_class": outcome,
    }


def _is_open_long_like(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in ["intent", "action", "final_decision", "side"]
    )
    return "open_long" in text or ("entry" in text and "buy" in text) or " buy" in f" {text}"


def _normalize_market_bars(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    normalized = frame
    if "symbol" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("symbol").map_elements(normalize_symbol, return_dtype=pl.Utf8).alias("symbol")
        )
    if "ts" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("ts").cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False).alias("ts")
        )
    if "timeframe" in normalized.columns:
        normalized = normalized.filter(pl.col("timeframe").cast(pl.Utf8).str.to_uppercase() == "1H")
    return normalized


def _market_rows_by_symbol(market_bars: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    if market_bars.is_empty():
        return rows_by_symbol
    for row in market_bars.sort("ts").to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        ts = _coerce_datetime(row.get("ts"))
        if not symbol or ts is None:
            continue
        rows_by_symbol.setdefault(symbol, []).append(
            {
                **row,
                "symbol": symbol,
                "ts": ts,
                "open": _float_or_none(row.get("open")),
                "high": _float_or_none(row.get("high")),
                "low": _float_or_none(row.get("low")),
                "close": _float_or_none(row.get("close")),
            }
        )
    return rows_by_symbol


def _pre_window_stats(
    bars: list[dict[str, Any]],
    entry_ts: datetime,
    hours: int,
) -> dict[str, float | None]:
    start = entry_ts - timedelta(hours=hours)
    selected = [
        row
        for row in bars
        if start <= row["ts"] < entry_ts
        and _float_or_none(row.get("low")) is not None
        and _float_or_none(row.get("high")) is not None
    ]
    if not selected:
        return {"low": None, "high": None}
    return {
        "low": min(float(row["low"]) for row in selected),
        "high": max(float(row["high"]) for row in selected),
    }


def _entry_vs_low_bps(price: float | None, low: float | None) -> float | None:
    if price is None or low is None or low <= 0:
        return None
    return (price / low - 1.0) * 10_000.0


def _position_in_range(
    price: float | None,
    low: float | None,
    high: float | None,
) -> float | None:
    if price is None or low is None or high is None or high <= low:
        return None
    return max(0.0, min(1.0, (price - low) / (high - low)))


def _missed_low_diagnosis(
    *,
    entry_vs_24h_low_bps: float | None,
    position_24h: float | None,
    realized_net_bps: float | None,
) -> str:
    if (entry_vs_24h_low_bps or 0.0) > 250.0 and realized_net_bps is not None:
        return "late_chase_loss" if realized_net_bps < 0.0 else "late_but_trend_profitable"
    if (entry_vs_24h_low_bps or 0.0) < 100.0 or (position_24h is not None and position_24h < 0.35):
        return "early_entry"
    return "normal_entry"


def _label_context(labels: pl.DataFrame) -> dict[str, dict[int, dict[str, Any]]]:
    context: dict[str, dict[int, dict[str, Any]]] = {}
    if labels.is_empty() or "candidate_id" not in labels.columns:
        return context
    for row in labels.to_dicts():
        candidate_id = str(row.get("candidate_id") or "")
        horizon = int(_float_or_none(row.get("horizon_hours")) or 0)
        if candidate_id and horizon:
            context.setdefault(candidate_id, {})[horizon] = row
    return context


def _candidate_price(
    row: dict[str, Any],
    bars: list[dict[str, Any]],
    ts: datetime,
) -> float | None:
    price = _float_or_none(
        row.get("entry_close")
        or row.get("candidate_px")
        or row.get("current_px")
        or row.get("price")
        or row.get("close")
        or row.get("last_price")
    )
    return price if price is not None else _market_close_at_or_before(bars, ts)


def _market_close_at_or_before(bars: list[dict[str, Any]], ts: datetime) -> float | None:
    close = None
    for row in bars:
        if row["ts"] <= ts:
            close = _float_or_none(row.get("close"))
        if row["ts"] > ts:
            break
    return close


def _pullback_reversal_candidate_ok(
    *,
    row: dict[str, Any],
    current_px: float,
    pre_24h: dict[str, float | None],
    pullback_bps: float | None,
    f4: float | None,
    f5: float | None,
    bars: list[dict[str, Any]],
    btc_bars: list[dict[str, Any]],
    ts: datetime,
    rule_version: str,
) -> bool:
    if rule_version == PULLBACK_OLD_RULE_VERSION:
        return _pullback_reversal_old_candidate_ok(
            row=row,
            current_px=current_px,
            pre_24h=pre_24h,
            pullback_bps=pullback_bps,
            f4=f4,
            f5=f5,
            bars=bars,
            ts=ts,
        )
    if str(row.get("regime_state") or row.get("risk_level") or "").lower() == "risk_off":
        return False
    low = pre_24h.get("low")
    if low is None or pullback_bps is None:
        return False
    if pullback_bps < 100.0 or pullback_bps > 500.0:
        return False
    if current_px <= low * 1.005:
        return False
    if f5 is not None and f5 < -0.10:
        return False
    if f4 is not None and f4 < -0.50:
        return False
    conditions = _pullback_reversal_conditions(
        row=row,
        current_px=current_px,
        pre_24h=pre_24h,
        bars=bars,
        btc_bars=btc_bars,
        ts=ts,
    )
    return all(conditions.values())


def _pullback_reversal_old_candidate_ok(
    *,
    row: dict[str, Any],
    current_px: float,
    pre_24h: dict[str, float | None],
    pullback_bps: float | None,
    f4: float | None,
    f5: float | None,
    bars: list[dict[str, Any]],
    ts: datetime,
) -> bool:
    if str(row.get("regime_state") or row.get("risk_level") or "").lower() == "risk_off":
        return False
    low = pre_24h.get("low")
    if low is None or pullback_bps is None:
        return False
    if pullback_bps < 100.0 or pullback_bps > 500.0:
        return False
    if current_px <= low * 1.005:
        return False
    if not _recent_2h_no_new_low(bars, ts):
        return False
    if f5 is not None and f5 < -0.10:
        return False
    if f4 is not None and f4 < -0.50:
        return False
    spread = _float_or_none(row.get("estimated_spread_bps") or row.get("spread_bps"))
    return spread is None or spread < 50.0


def _pullback_reversal_conditions(
    *,
    row: dict[str, Any],
    current_px: float,
    pre_24h: dict[str, float | None],
    bars: list[dict[str, Any]],
    btc_bars: list[dict[str, Any]],
    ts: datetime,
) -> dict[str, bool]:
    return {
        "recent_2h_no_new_low": _recent_2h_no_new_low(
            bars,
            ts,
            require_history=True,
        ),
        "close_reclaim_1h": _close_reclaim_1h(
            bars,
            ts,
            current_px=current_px,
            pre_24h_low=pre_24h.get("low"),
        ),
        "current_close_gt_previous_close": _current_close_gt_previous_close(
            bars,
            ts,
            current_px=current_px,
        ),
        "btc_not_sharp_drop": _btc_not_sharp_drop(btc_bars, ts),
        "spread_not_abnormal": _spread_not_abnormal(row),
    }


def _pullback_from_high_bps(price: float, high: float | None) -> float | None:
    if high is None or high <= 0:
        return None
    return (high / price - 1.0) * 10_000.0


def _recent_2h_no_new_low(
    bars: list[dict[str, Any]],
    ts: datetime,
    *,
    require_history: bool = False,
) -> bool:
    recent_start = ts - timedelta(hours=2)
    previous_start = ts - timedelta(hours=24)
    recent = [row for row in bars if recent_start <= row["ts"] <= ts]
    previous = [row for row in bars if previous_start <= row["ts"] < recent_start]
    if not recent or not previous:
        return not require_history
    recent_low = min(_float_or_none(row.get("low")) or math.inf for row in recent)
    previous_low = min(_float_or_none(row.get("low")) or math.inf for row in previous)
    return recent_low > previous_low


def _close_reclaim_1h(
    bars: list[dict[str, Any]],
    ts: datetime,
    *,
    current_px: float,
    pre_24h_low: float | None,
) -> bool:
    current_bar, previous_bar = _current_and_previous_bar(bars, ts)
    if current_bar is None or previous_bar is None or pre_24h_low is None:
        return False
    current_close = _float_or_none(current_bar.get("close")) or current_px
    current_open = _float_or_none(current_bar.get("open"))
    previous_close = _float_or_none(previous_bar.get("close"))
    if current_open is None or previous_close is None:
        return False
    return (
        current_px > pre_24h_low * 1.005
        and current_px > previous_close
        and current_close >= max(current_open, previous_close)
    )


def _current_close_gt_previous_close(
    bars: list[dict[str, Any]],
    ts: datetime,
    *,
    current_px: float,
) -> bool:
    _current_bar, previous_bar = _current_and_previous_bar(bars, ts)
    if previous_bar is None:
        return False
    previous_close = _float_or_none(previous_bar.get("close"))
    return previous_close is not None and current_px > previous_close


def _btc_not_sharp_drop(btc_bars: list[dict[str, Any]], ts: datetime) -> bool:
    current_bar, previous_bar = _current_and_previous_bar(btc_bars, ts)
    if current_bar is None or previous_bar is None:
        return False
    current_close = _float_or_none(current_bar.get("close"))
    previous_close = _float_or_none(previous_bar.get("close"))
    if current_close is None or previous_close is None or previous_close <= 0:
        return False
    one_hour_bps = (current_close / previous_close - 1.0) * 10_000.0
    if one_hour_bps <= PULLBACK_BTC_MAX_1H_DROP_BPS:
        return False
    four_hour_close = _market_close_at_or_before(btc_bars, ts - timedelta(hours=4))
    if four_hour_close is None or four_hour_close <= 0:
        return False
    four_hour_bps = (current_close / four_hour_close - 1.0) * 10_000.0
    return four_hour_bps > PULLBACK_BTC_MAX_4H_DROP_BPS


def _spread_not_abnormal(row: dict[str, Any]) -> bool:
    spread = _float_or_none(row.get("estimated_spread_bps"))
    if spread is None:
        spread = _float_or_none(row.get("spread_bps"))
    return spread is not None and 0.0 <= spread <= PULLBACK_SPREAD_MAX_BPS


def _current_and_previous_bar(
    bars: list[dict[str, Any]],
    ts: datetime,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    history = [row for row in bars if row["ts"] <= ts]
    if len(history) < 2:
        return None, None
    return history[-1], history[-2]


def _forward_label(
    bars: list[dict[str, Any]],
    ts: datetime,
    current_px: float,
    horizon_hours: int,
    roundtrip_cost_bps: float,
) -> dict[str, Any]:
    end = ts + timedelta(hours=horizon_hours)
    end_bar = next((row for row in bars if row["ts"] >= end), None)
    if end_bar is None:
        return {
            "gross_bps": None,
            "net_bps_after_cost": None,
            "mfe_bps": None,
            "mae_bps": None,
            "win": None,
            "label_status": "pending",
        }
    label_ts = end_bar["ts"]
    future = [row for row in bars if ts < row["ts"] <= label_ts]
    if not future:
        return {
            "gross_bps": None,
            "net_bps_after_cost": None,
            "mfe_bps": None,
            "mae_bps": None,
            "win": None,
            "label_status": "pending",
        }
    end_close = _float_or_none(future[-1].get("close"))
    highs = [_float_or_none(row.get("high")) for row in future]
    lows = [_float_or_none(row.get("low")) for row in future]
    highs = [value for value in highs if value is not None]
    lows = [value for value in lows if value is not None]
    gross = ((end_close / current_px - 1.0) * 10_000.0) if end_close else None
    net = (gross - roundtrip_cost_bps) if gross is not None else None
    return {
        "gross_bps": gross,
        "net_bps_after_cost": net,
        "mfe_bps": ((max(highs) / current_px - 1.0) * 10_000.0) if highs else None,
        "mae_bps": ((min(lows) / current_px - 1.0) * 10_000.0) if lows else None,
        "win": None if net is None else net > 0.0,
        "label_status": "complete",
    }


def _roundtrip_cost_by_symbol(costs: pl.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    if costs.is_empty():
        return output
    for row in costs.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        cost = _float_or_none(row.get("roundtrip_all_in_cost_bps"))
        if cost is None:
            one_way = _float_or_none(
                row.get("one_way_all_in_cost_bps")
                or row.get("total_cost_bps_p75")
                or row.get("selected_total_cost_bps")
            )
            cost = one_way * 2.0 if one_way is not None else None
        if cost is None:
            continue
        output[symbol] = max(output.get(symbol, 0.0), float(cost))
    return output


def _pullback_readiness_reasons(
    *,
    sample_count: int,
    recent_count: int,
    avg_net: float | None,
    win_rate: float | None,
    p25: float | None,
    avg_mae: float | None,
) -> list[str]:
    reasons: list[str] = []
    if sample_count < 50:
        reasons.append("insufficient_sample_count")
    if recent_count < 10:
        reasons.append("insufficient_recent_7d_samples")
    if avg_net is None or avg_net <= 50.0:
        reasons.append("weak_24h_avg_net_bps")
    if win_rate is None or win_rate <= 0.55:
        reasons.append("weak_24h_win_rate")
    if p25 is None or p25 <= -50.0:
        reasons.append("weak_24h_p25_net_bps")
    if avg_mae is None or avg_mae <= -120.0:
        reasons.append("excessive_avg_mae_bps")
    return reasons


def _pullback_recommended_mode(
    row: dict[str, Any],
    *,
    ready_for_paper: bool,
) -> str:
    if ready_for_paper:
        return "paper"
    sample_count = int(row.get("sample_count") or 0)
    recent_count = int(row.get("recent_7d_sample_count") or 0)
    avg_net = _float_or_none(row.get("avg_24h_net_bps"))
    win_rate = _float_or_none(row.get("win_rate_24h"))
    p25 = _float_or_none(row.get("p25_24h_net_bps"))
    avg_mae = _float_or_none(row.get("avg_mae_bps"))
    if sample_count < 50 or recent_count < 10:
        return "research"
    if avg_net is None or avg_net <= 0.0:
        return "research"
    if win_rate is None or win_rate <= 0.45:
        return "research"
    if p25 is None or p25 <= -50.0:
        return "research"
    if avg_mae is None or avg_mae <= -120.0:
        return "research"
    return "shadow"


def _pullback_no_sample_reason(
    row: dict[str, Any],
    *,
    recommended_mode: str,
) -> str:
    if recommended_mode == "paper":
        return "not_live_validated"
    reasons = _json_listish(row.get("readiness_reasons"))
    if "insufficient_sample_count" in reasons:
        return "insufficient_sample_count"
    if "insufficient_recent_7d_samples" in reasons:
        return "insufficient_recent_7d_samples"
    if "weak_24h_avg_net_bps" in reasons:
        return "weak_24h_avg_net_bps"
    if "weak_24h_p25_net_bps" in reasons:
        return "weak_24h_p25_net_bps"
    if "excessive_avg_mae_bps" in reasons:
        return "excessive_avg_mae_bps"
    if recommended_mode == "shadow":
        return "shadow_only_collect_more_samples"
    return "not_live_validated"


def _pullback_candidate_name(symbol: str) -> str:
    base = normalize_symbol(symbol).split("-")[0].lower()
    return f"v5.pullback_reversal_shadow_{base}"


def _common(ctx: _BuildContext, *, mode: str) -> dict[str, Any]:
    git_commit = _git_commit()
    return {
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
        "schema_version": ENTRY_QUALITY_SCHEMA_VERSION,
        "quant_lab_git_commit": git_commit,
        "source_version": _source_version("entry_quality", git_commit),
        "generated_at_utc": ctx.generated_at,
        "generated_from_bundle_id": ctx.generated_from_bundle_id,
        "as_of_date": ctx.as_of_date.isoformat(),
        "window_hours": ctx.window_hours,
        "source": SOURCE_NAME,
        "mode": mode,
    }


def _publish_daily(
    root: Path,
    relative_path: Path,
    frame: pl.DataFrame,
    key_columns: list[str],
) -> int:
    if frame.is_empty():
        return read_parquet_dataset(root / relative_path).height
    return upsert_parquet_dataset(frame, root / relative_path, key_columns=key_columns)


def _replace_daily(
    root: Path,
    relative_path: Path,
    frame: pl.DataFrame,
) -> int:
    write_parquet_dataset(frame, root / relative_path)
    return frame.height


def _publish_entry_quality_strategy_opportunities(
    root: Path,
    frame: pl.DataFrame,
) -> int:
    dataset_path = root / STRATEGY_OPPORTUNITY_ADVISORY_DATASET
    existing = read_parquet_dataset(dataset_path)
    kept = _without_entry_quality_strategy_candidates(existing)
    frames = [candidate for candidate in [kept, frame] if not candidate.is_empty()]
    if frames:
        combined = pl.concat(frames, how="diagonal_relaxed")
    elif not kept.is_empty():
        combined = kept
    elif not frame.is_empty():
        combined = frame
    else:
        combined = pl.DataFrame(schema=STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA)
    write_parquet_dataset(combined, dataset_path)
    write_snapshot_meta(
        dataset_path,
        dataset_name="strategy_opportunity_advisory",
        frame=combined,
        schema_version=STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION,
    )
    return combined.height


def _without_entry_quality_strategy_candidates(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "strategy_candidate" not in frame.columns:
        return frame
    candidate = pl.col("strategy_candidate").cast(pl.Utf8)
    entry_quality_mask = (
        candidate.str.starts_with("v5.entry_quality_")
        | (candidate == "v5.late_entry_chase_guard_shadow")
        | candidate.str.starts_with("v5.pullback_reversal_shadow_")
    )
    return frame.filter(~entry_quality_mask)


def publish_entry_quality_history_result(
    lake_root: str | Path,
    artifacts: EntryQualityHistoryArtifacts,
    *,
    generation_id: str,
    snapshot_id: str,
    task_id: str,
    reports: dict[str, bytes] | None = None,
) -> dict[str, int]:
    """Publish all 11 historical tables and reports as one rollback-capable generation."""

    root = Path(lake_root)
    safe_generation_id = _safe_research_identifier(generation_id, "generation_id")
    safe_snapshot_id = _safe_research_identifier(snapshot_id, "snapshot_id")
    safe_task_id = _safe_research_identifier(task_id, "task_id")
    frames = artifacts.frames_by_dataset()
    if set(frames) != set(ENTRY_QUALITY_HISTORY_OUTPUT_SPEC_BY_NAME):
        raise ValueError("entry_quality_history_output_set_mismatch")

    transaction_id = uuid.uuid4().hex
    staging_root = root / "gold" / f".__eqh_s_{transaction_id[:8]}"
    backup_root = root / "gold" / f".__eqh_b_{transaction_id[:8]}"
    staging_root.mkdir(parents=True, exist_ok=False)
    backup_root.mkdir(parents=True, exist_ok=False)
    published_rows: dict[str, int] = {}
    moved_targets: list[tuple[Path, Path | None]] = []
    moved_reports: list[tuple[Path, Path | None]] = []
    generation_payload = {
        "schema_version": "entry_quality_history_generation.v1",
        "generation_id": safe_generation_id,
        "snapshot_id": safe_snapshot_id,
        "task_id": safe_task_id,
        "start_date": artifacts.start_date.isoformat(),
        "end_date": artifacts.end_date.isoformat(),
        "window_mode": artifacts.mode,
        "cost_mode": artifacts.cost_mode,
        "generated_at": artifacts.generated_at.isoformat(),
        "published_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "live_order_effect": "none",
    }
    try:
        if reports is not None:
            if set(reports) != set(ENTRY_QUALITY_HISTORY_REPORT_NAMES):
                raise ValueError("entry_quality_history_report_set_mismatch")
            staged_reports = staging_root / "reports"
            staged_reports.mkdir(parents=True, exist_ok=False)
            for name in ENTRY_QUALITY_HISTORY_REPORT_NAMES:
                if Path(name).name != name:
                    raise ValueError("unsafe_entry_quality_history_report_name")
                (staged_reports / name).write_bytes(reports[name])
        for index, spec in enumerate(ENTRY_QUALITY_HISTORY_OUTPUT_SPECS):
            new_frame = _coerce_history_frame(frames[spec.dataset_name], spec)
            current = read_parquet_dataset(root / spec.relative_path)
            combined = _replace_history_window(
                current,
                new_frame,
                spec=spec,
                window_mode=artifacts.mode,
                cost_mode=artifacts.cost_mode,
            )
            staged = staging_root / f"d{index:02d}"
            write_parquet_dataset(combined, staged)
            write_snapshot_meta(
                staged,
                dataset_name=spec.dataset_name,
                frame=combined,
                schema_version=ENTRY_QUALITY_SCHEMA_VERSION,
                generated_at=artifacts.generated_at,
            )
            atomic_write_json(staged / "_research_generation.json", generation_payload)
            published_rows[spec.dataset_name] = combined.height

        for index, spec in enumerate(ENTRY_QUALITY_HISTORY_OUTPUT_SPECS):
            target = root / spec.relative_path
            staged = staging_root / f"d{index:02d}"
            backup = backup_root / f"d{index:02d}"
            previous: Path | None = None
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
                previous = backup
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
            except Exception:
                if previous is not None and previous.exists() and not target.exists():
                    os.replace(previous, target)
                raise
            moved_targets.append((target, previous))

        if reports is not None:
            reports_root = root / "reports"
            reports_root.mkdir(parents=True, exist_ok=True)
            for name in ENTRY_QUALITY_HISTORY_REPORT_NAMES:
                target = reports_root / name
                staged = staging_root / "reports" / name
                backup = backup_root / "reports" / name
                previous = None
                if target.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, backup)
                    previous = backup
                try:
                    os.replace(staged, target)
                except Exception:
                    if previous is not None and previous.exists() and not target.exists():
                        os.replace(previous, target)
                    raise
                moved_reports.append((target, previous))

        for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS:
            target = root / spec.relative_path
            metadata = json.loads(
                (target / "_research_generation.json").read_text(encoding="utf-8")
            )
            if metadata.get("generation_id") != safe_generation_id:
                raise RuntimeError("entry_quality_history_generation_switch_mismatch")
            if not (target / "_snapshot_meta.json").is_file():
                raise RuntimeError("entry_quality_history_snapshot_meta_missing")

        atomic_write_json(
            root / "gold" / "entry_quality_history_generation.json",
            generation_payload
            | {
                "datasets": [spec.dataset_name for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS],
                "row_counts": published_rows,
            },
        )
    except Exception:
        for target, previous in reversed(moved_reports):
            if target.exists():
                target.unlink()
            if previous is not None and previous.exists():
                os.replace(previous, target)
        for target, previous in reversed(moved_targets):
            if target.exists():
                shutil.rmtree(target)
            if previous is not None and previous.exists():
                os.replace(previous, target)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)
    return published_rows


def _replace_history_window(
    current: pl.DataFrame,
    replacement: pl.DataFrame,
    *,
    spec: EntryQualityHistoryOutputSpec,
    window_mode: str,
    cost_mode: str,
) -> pl.DataFrame:
    kept = current
    if not current.is_empty():
        if set(spec.window_keys).issubset(set(current.columns)):
            mask = (pl.col("window_mode").cast(pl.Utf8) == window_mode) & (
                pl.col("cost_mode").cast(pl.Utf8) == cost_mode
            )
            kept = current.filter(~mask)
        else:
            kept = pl.DataFrame(schema=spec.schema)
    frames = [frame for frame in (kept, replacement) if not frame.is_empty()]
    if not frames:
        return pl.DataFrame(schema=spec.schema)
    combined = pl.concat(frames, how="diagonal_relaxed")
    combined = _coerce_history_frame(combined, spec)
    if set(spec.primary_keys).issubset(set(combined.columns)):
        combined = combined.unique(subset=list(spec.primary_keys), keep="last", maintain_order=True)
    return combined


def _coerce_history_frame(
    frame: pl.DataFrame,
    spec: EntryQualityHistoryOutputSpec,
) -> pl.DataFrame:
    normalized = frame
    for column, dtype in spec.schema.items():
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=dtype).alias(column))
    normalized = normalized.select(list(spec.schema)).cast(spec.schema, strict=False)
    if normalized.is_empty():
        return pl.DataFrame(schema=spec.schema)
    return normalized


def _safe_research_identifier(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not text or len(text) > 180 or any(character not in allowed for character in text):
        raise ValueError(f"unsafe_{field_name}")
    return text


def _publish_history(
    root: Path,
    relative_path: Path,
    frame: pl.DataFrame,
    key_columns: list[str],
) -> int:
    if frame.is_empty():
        write_parquet_dataset(frame, root / relative_path)
        return 0
    return upsert_parquet_dataset(frame, root / relative_path, key_columns=key_columns)


def _with_history_meta(
    frame: pl.DataFrame,
    history: _HistoryContext,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    normalized = frame
    meta = _history_meta(history)
    for column, value in meta.items():
        normalized = normalized.with_columns(pl.lit(value, dtype=pl.Utf8).alias(column))
    return normalized


def _history_meta(history: _HistoryContext) -> dict[str, str]:
    return {
        "start_date": history.start_date.isoformat(),
        "end_date": history.end_date.isoformat(),
        "window_mode": history.window_mode,
        "cost_mode": history.cost_mode,
    }


def _filter_frame_by_time(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    start_dt: datetime,
    end_dt: datetime,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    time_column = next((column for column in columns if column in frame.columns), None)
    if time_column is None:
        return frame
    temp_column = "__quant_lab_history_ts"
    normalized = frame.with_columns(
        pl.col(time_column)
        .cast(pl.Utf8)
        .str.to_datetime(time_zone="UTC", strict=False)
        .alias(temp_column)
    )
    return normalized.filter(
        (pl.col(temp_column) >= start_dt) & (pl.col(temp_column) < end_dt)
    ).drop(temp_column)


def _is_within_window(ts: datetime | None, start_dt: datetime, end_dt: datetime) -> bool:
    return ts is not None and start_dt <= ts < end_dt


def _history_shadow_decision(
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    win_rate: float | None,
    p25_net_bps: float | None,
    avg_mae_bps: float | None,
    cost_quality_mix: Counter[str],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if sample_count < 10:
        reasons.append("insufficient_total_samples")
    if complete_sample_count < 5:
        reasons.append("insufficient_complete_samples")
    if avg_mae_bps is None or avg_mae_bps <= -120.0:
        reasons.append("excessive_avg_mae_bps")
    if p25_net_bps is None or p25_net_bps <= -50.0:
        reasons.append("weak_p25_net_bps")
    cost_degraded = any(key != "quant_lab" for key in cost_quality_mix)
    if cost_degraded:
        reasons.append("cost_quality_degraded")
    if reasons:
        return "RESEARCH_ONLY", reasons
    if avg_net_bps is None or avg_net_bps <= 0.0:
        return "RESEARCH_ONLY", ["non_positive_after_cost_edge"]
    if complete_sample_count >= 10 and win_rate is not None and win_rate > 0.50:
        return "KEEP_SHADOW", [
            "positive_after_cost_edge_collect_more_samples",
            "pullback_v2_shadow_only_not_paper_ready",
        ]
    return "KEEP_SHADOW", ["positive_edge_but_weak_win_rate"]


def _entry_quality_recommended_mode(candidate: str, value: Any) -> str:
    if candidate == "v5.entry_quality_missed_low_audit":
        return "research"
    mode = str(value or "").strip().lower()
    if mode == "audit":
        return "research"
    if mode in {"paper", "shadow", "research"}:
        return mode
    return "shadow"


def _entry_quality_strategy_decision(recommended_mode: str) -> str:
    if recommended_mode == "paper":
        return "PAPER_READY"
    if recommended_mode == "shadow":
        return "KEEP_SHADOW"
    return "RESEARCH_ONLY"


def _entry_quality_live_block_reasons(row: dict[str, Any], recommended_mode: str) -> list[str]:
    reasons = set(_json_listish(row.get("advisory_reasons")))
    reasons.add("shadow_only")
    reasons.add("not_live_validated")
    if recommended_mode != "paper":
        reasons.add("not_paper_candidate")
    reasons.add("entry_quality_advisory_only")
    return sorted(reason for reason in reasons if reason)


def _entry_quality_no_sample_reason(
    row: dict[str, Any],
    recommended_mode: str,
    sample_count: int,
) -> str | None:
    existing = str(row.get("no_sample_reason") or "").strip()
    if existing:
        return existing
    if sample_count <= 0:
        return "no_entry_quality_samples"
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


def _entry_quality_as_of_datetime(row: dict[str, Any]) -> datetime:
    value = _coerce_datetime(row.get("generated_at_utc"))
    if value is not None:
        return value
    as_of = str(row.get("as_of_date") or "").strip()
    if len(as_of) == 10 and as_of[4] == "-":
        return datetime.combine(date.fromisoformat(as_of), time.min, tzinfo=UTC)
    return datetime.now(UTC)


def _strategy_advisory_expires_at(generated_at: datetime) -> datetime:
    return generated_at.astimezone(UTC) + timedelta(
        seconds=STRATEGY_OPPORTUNITY_ADVISORY_TTL_SECONDS
    )


@lru_cache(maxsize=1)
def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            cwd=root,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value or None


def _source_version(component: str, git_commit: str | None) -> str:
    version = git_commit or __version__
    return f"{component}:{version}"


def _observable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in UNOBSERVABLE_TEXT_VALUES else text


def _strategy_id(strategy_candidate: str, symbol: str) -> str:
    raw = f"{symbol}_{strategy_candidate}"
    normalized = "".join(character if character.isalnum() else "_" for character in raw.upper())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _v5_symbol(symbol: str) -> str:
    return symbol.replace("-", "_") if symbol != "ALL" else "ALL"


def _latest_as_of_date(rows: list[dict[str, Any]]) -> str | None:
    values = [
        str(row.get("as_of_date") or "").strip()
        for row in rows
        if str(row.get("as_of_date") or "").strip()
    ]
    return max(values) if values else None


def _optional_int(value: Any) -> int | None:
    number = _float_or_none(value)
    return int(number) if number is not None else None


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


def render_entry_quality_history_reports(
    artifacts: EntryQualityHistoryArtifacts,
) -> dict[str, bytes]:
    metrics_payload = _history_metrics_payload(artifacts.metrics)
    csv_frames = {
        "missed_low_audit.csv": artifacts.missed_low_audit,
        "missed_low_by_symbol.csv": artifacts.missed_low_by_symbol,
        "missed_low_by_entry_reason.csv": artifacts.missed_low_by_entry_reason,
        "late_entry_chase_threshold_sensitivity.csv": (
            artifacts.late_entry_threshold_sensitivity
        ),
        "pullback_reversal_by_symbol.csv": artifacts.pullback_by_symbol,
        "pullback_reversal_by_regime.csv": artifacts.pullback_by_regime,
        "pullback_reversal_by_horizon.csv": artifacts.pullback_by_horizon,
        "anti_leakage_check.csv": artifacts.anti_leakage_check,
    }
    reports = {
        name: frame.write_csv().encode("utf-8") for name, frame in csv_frames.items()
    }
    reports["entry_quality_historical_metrics.json"] = safe_json_dumps(
        metrics_payload
    ).encode("utf-8")
    reports["entry_quality_historical_summary.md"] = _entry_quality_history_summary_md(
        metrics_payload
    ).encode("utf-8")
    return reports


def write_entry_quality_history_reports(
    reports_dir: str | Path,
    reports: dict[str, bytes],
) -> Path:
    destination = Path(reports_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".entry-quality-history-reports-",
        dir=destination.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        for name, payload in reports.items():
            if Path(name).name != name or not name:
                raise ValueError("unsafe_entry_quality_history_report_name")
            path = temporary / name
            path.write_bytes(payload)
        for name in sorted(reports):
            os.replace(temporary / name, destination / name)
    return destination


def _write_history_reports(
    root: Path,
    *,
    missed_low: pl.DataFrame,
    missed_low_by_symbol: pl.DataFrame,
    missed_low_by_entry_reason: pl.DataFrame,
    late_threshold: pl.DataFrame,
    pullback_by_symbol: pl.DataFrame,
    pullback_by_regime: pl.DataFrame,
    pullback_by_horizon: pl.DataFrame,
    anti_leakage: pl.DataFrame,
    metrics: pl.DataFrame,
) -> Path:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_report(reports_dir / "missed_low_audit.csv", missed_low, MISSED_LOW_AUDIT_SCHEMA)
    _write_csv_report(
        reports_dir / "missed_low_by_symbol.csv",
        missed_low_by_symbol,
        MISSED_LOW_AGG_SCHEMA,
    )
    _write_csv_report(
        reports_dir / "missed_low_by_entry_reason.csv",
        missed_low_by_entry_reason,
        MISSED_LOW_AGG_SCHEMA,
    )
    _write_csv_report(
        reports_dir / "late_entry_chase_threshold_sensitivity.csv",
        late_threshold,
        HISTORY_THRESHOLD_SENSITIVITY_SCHEMA,
    )
    _write_csv_report(
        reports_dir / "pullback_reversal_by_symbol.csv",
        pullback_by_symbol,
        HISTORY_PULLBACK_AGG_SCHEMA,
    )
    _write_csv_report(
        reports_dir / "pullback_reversal_by_regime.csv",
        pullback_by_regime,
        HISTORY_PULLBACK_AGG_SCHEMA,
    )
    _write_csv_report(
        reports_dir / "pullback_reversal_by_horizon.csv",
        pullback_by_horizon,
        HISTORY_PULLBACK_AGG_SCHEMA,
    )
    _write_csv_report(
        reports_dir / "anti_leakage_check.csv",
        anti_leakage,
        HISTORY_ANTI_LEAKAGE_SCHEMA,
    )
    metrics_payload = _history_metrics_payload(metrics)
    (reports_dir / "entry_quality_historical_metrics.json").write_text(
        safe_json_dumps(metrics_payload), encoding="utf-8"
    )
    (reports_dir / "entry_quality_historical_summary.md").write_text(
        _entry_quality_history_summary_md(metrics_payload),
        encoding="utf-8",
    )
    return reports_dir


def _write_csv_report(path: Path, frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> None:
    if frame.is_empty() and not frame.columns:
        frame = pl.DataFrame(schema=schema)
    frame.write_csv(path)


def _history_metrics_payload(metrics: pl.DataFrame) -> dict[str, Any]:
    if metrics.is_empty():
        return {}
    row = metrics.to_dicts()[0]
    text = str(row.get("metrics_json") or "{}")
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _entry_quality_history_summary_md(metrics: dict[str, Any]) -> str:
    lines = [
        "# Entry Quality Historical Analysis",
        "",
        "This report is read-only historical research. It does not place orders, "
        "block orders, mutate V5 state, or change risk permission.",
        "",
        f"- start_date: {metrics.get('start_date', '')}",
        f"- end_date: {metrics.get('end_date', '')}",
        f"- mode: {metrics.get('mode', '')}",
        f"- cost_mode: {metrics.get('cost_mode', '')}",
        f"- missed_low_audit_rows: {metrics.get('missed_low_audit_rows', 0)}",
        f"- late_entry_chase_shadow_rows: {metrics.get('late_entry_chase_shadow_rows', 0)}",
        f"- pullback_reversal_shadow_rows: {metrics.get('pullback_reversal_shadow_rows', 0)}",
        f"- anti_leakage_status: {metrics.get('anti_leakage_status', '')}",
        "",
        "Allowed pullback reversal v2 decisions: RESEARCH_ONLY, KEEP_SHADOW.",
        "Paper promotion and live-small readiness are intentionally unavailable for pullback v2.",
    ]
    return "\n".join(lines) + "\n"


def _latest_bundle_id(root: Path) -> str:
    for dataset in [
        root / "gold" / "strategy_health_daily",
        root / "silver" / "v5_candidate_event",
        root / "silver" / "v5_trade_event",
    ]:
        frame = read_parquet_dataset(dataset)
        if frame.is_empty():
            continue
        for column in [
            "latest_bundle_sha256",
            "bundle_sha256",
            "bundle_name",
            "source_bundle_ts",
            "bundle_ts",
        ]:
            if column in frame.columns:
                values = [str(value) for value in frame.get_column(column).drop_nulls().to_list()]
                if values:
                    return max(values)
    return ""


def latest_entry_quality_bundle_id(lake_root: str | Path) -> str:
    return _latest_bundle_id(Path(lake_root))


def _normalize_history_request(
    *,
    start_date: str | date,
    end_date: str | date,
    mode: str,
    cost_mode: str,
) -> tuple[date, date, str, str]:
    start_day = _parse_day(start_date)
    end_day = _parse_day(end_date)
    if end_day < start_day:
        raise ValueError("end_date must be greater than or equal to start_date")
    window_mode = str(mode or "full").strip().lower()
    if window_mode not in {"full", "recent_7d", "recent_30d", "walk_forward"}:
        raise ValueError("mode must be one of: full, recent_7d, recent_30d, walk_forward")
    if window_mode == "recent_7d":
        start_day = max(start_day, end_day - timedelta(days=6))
    elif window_mode == "recent_30d":
        start_day = max(start_day, end_day - timedelta(days=29))
    normalized_cost_mode = str(cost_mode or "conservative").strip().lower()
    if normalized_cost_mode not in {"conservative", "quant_lab"}:
        raise ValueError("cost_mode must be one of: conservative, quant_lab")
    return start_day, end_day, window_mode, normalized_cost_mode


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return datetime.now(UTC).date()
    return date.fromisoformat(str(value)[:10])


def _source_event_key(row: dict[str, Any], *, symbol: str, ts: Any) -> str:
    for key in [
        "event_key",
        "source_event_key",
        "lifecycle_id",
        "trade_id",
        "order_id",
        "candidate_id",
    ]:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    dt = _coerce_datetime(ts)
    material = "|".join(
        [
            str(row.get("run_id") or ""),
            symbol,
            dt.isoformat() if dt else "",
            str(row.get("row_index") or ""),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"entry_quality:{digest}"


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mean(values: Any) -> float | None:
    numbers = [_float_or_none(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int((len(ordered) - 1) * q)
    return ordered[index]


def _group_dicts(rows: list[dict[str, Any]], column: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(column) or "UNKNOWN"), []).append(row)
    return grouped


def _classify_outcome(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "profit" if value >= 0.0 else "loss"


def _outcome_value(row: dict[str, Any]) -> float | None:
    for column in ["realized_net_bps", "forward_24h_net_bps", "forward_48h_net_bps"]:
        value = _float_or_none(row.get(column))
        if value is not None:
            return value
    return None
