from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.advisory_overrides import portfolio_status_overrides_by_identifier
from quant_lab.research.alpha_discovery import normalize_alpha_discovery_board_decisions
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

PAPER_STRATEGY_RUNS_DATASET = Path("gold") / "paper_strategy_runs"
PAPER_STRATEGY_DAILY_DATASET = Path("gold") / "paper_strategy_daily"
PAPER_SLIPPAGE_COVERAGE_DATASET = Path("gold") / "paper_slippage_coverage"
ALPHA_DISCOVERY_BOARD_DATASET = Path("gold") / "alpha_discovery_board"
RESEARCH_PORTFOLIO_STATUS_DATASET = Path("gold") / "research_portfolio_status"
V5_PAPER_STRATEGY_RUN_DATASET = Path("silver") / "v5_paper_strategy_run"
V5_PAPER_STRATEGY_DAILY_DATASET = Path("silver") / "v5_paper_strategy_daily"
V5_PAPER_SLIPPAGE_COVERAGE_DATASET = Path("silver") / "v5_paper_slippage_coverage"
V5_CANDIDATE_EVENT_DATASET = Path("silver") / "v5_candidate_event"
PAPER_TRACKING_SOURCE = "research.paper_strategy_tracking.v0.1"
V5_PAPER_TRACKING_SOURCE = "v5.paper_strategy_telemetry"
V5_PAPER_TRACKING_STATUS = "active"
PAPER_TRACKING_SCHEMA_VERSION = "paper_strategy_tracking.v1"
DEFAULT_REQUIRED_ENTRY_DAYS_FOR_LIVE = 3
COST_FALLBACK_NOT_LIVE_SAFE = "fallback_not_live_safe"
COST_DEGRADED_MODEL = "degraded_cost_model"
LIVE_BLOCKING_COST_SOURCES = {
    "global_default",
    "cost_not_requested_no_order",
    COST_FALLBACK_NOT_LIVE_SAFE,
    COST_DEGRADED_MODEL,
}
SAFE_COST_FALLBACK_TOKENS = {"", "none", "actual_okx_fills_and_bills"}
PAPER_PNL_HORIZON_HOURS = (4, 8, 12, 24, 48, 72)
SOL_F4_PAPER_PROPOSAL_ID = "SOL_F4_VOLUME_EXPANSION_PAPER_V1"
SOL_F4_STRATEGY_CANDIDATE = "v5.f4_volume_expansion_entry"
SOL_F4_SYMBOL = "SOL-USDT"
SOL_F3_PAPER_PROPOSAL_ID = "SOL_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
SOL_F3_STRATEGY_CANDIDATE = "v5.f3_dominant_entry"
SOL_F3_SYMBOL = "SOL-USDT"
ETH_F3_PAPER_PROPOSAL_ID = "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
ETH_F3_STRATEGY_CANDIDATE = "v5.f3_dominant_entry"
ETH_F3_SYMBOL = "ETH-USDT"
ETH_F3_PRIMARY_REVIEW_HORIZON = "48h"
ETH_F3_MIN_PRIMARY_COMPLETE_COUNT = 30
ETH_F3_DISABLED_NO_SAMPLE_REASON = "downngraded_from_paper_no_new_entry"
ETH_F4_VOLUME_EXPANSION_PAPER_PROPOSAL_ID = "ETH_USDT_F4_VOLUME_EXPANSION_ENTRY_PAPER_V1"
ETH_F4_VOLUME_EXPANSION_STRATEGY_CANDIDATE = "v5.f4_volume_expansion_entry"
ETH_F4_VOLUME_EXPANSION_SYMBOL = "ETH-USDT"
BNB_F3_PAPER_PROPOSAL_ID = "BNB_F3_DOMINANT_ENTRY_PAPER_V1"
BNB_F3_CURRENT_PROPOSAL_ID = "BNB_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
BNB_RISK_ON_BUY_PAPER_PROPOSAL_ID = "BNB_RISK_ON_BUY_PAPER_V1"
BNB_F3_STRATEGY_CANDIDATE = "v5.bnb_f3_dominant_entry"
BNB_F3_GENERIC_STRATEGY_CANDIDATE = "v5.f3_dominant_entry"
BNB_RISK_ON_BUY_STRATEGY_CANDIDATE = "v5.bnb_risk_on_buy"
BNB_SYMBOL = "BNB-USDT"
BNB_ALLOWED_PAPER_REGIMES = {"ALT_IMPULSE", "TREND_UP", "TRENDING"}
BNB_F3_SOURCE_CANDIDATES = {
    "f3_dominant_entry",
    "v5.f3_dominant_entry",
    "v5.bnb_f3_dominant_entry",
}

PAPER_RUN_REPORT_SCHEMA = {
    "as_of_date": pl.Utf8,
    "strategy_id": pl.Utf8,
    "proposal_id": pl.Utf8,
    "paper_tracker_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "run_id": pl.Utf8,
    "ts_utc": pl.Utf8,
    "symbol": pl.Utf8,
    "would_enter": pl.Boolean,
    "would_exit": pl.Boolean,
    "final_decision": pl.Utf8,
    "no_sample_reason": pl.Utf8,
    "paper_disabled_by_research_portfolio": pl.Boolean,
    "paper_trigger_type": pl.Utf8,
    "paper_trigger_reason": pl.Utf8,
    "source_candidate_symbol": pl.Utf8,
    "source_candidate_id": pl.Utf8,
    "symbol_match_verified": pl.Boolean,
    "paper_source": pl.Utf8,
    "paper_count_scope": pl.Utf8,
    "risk_level": pl.Utf8,
    "alpha6_score": pl.Float64,
    "alpha6_side": pl.Utf8,
    "f4_volume_expansion": pl.Utf8,
    "f5_rsi_trend_confirm": pl.Utf8,
    "arrival_bid": pl.Float64,
    "arrival_ask": pl.Float64,
    "arrival_mid": pl.Float64,
    "estimated_spread_bps": pl.Float64,
    "expected_order_type": pl.Utf8,
    "estimated_fill_px": pl.Float64,
    "cost_source": pl.Utf8,
    "cost_source_mix": pl.Utf8,
    "paper_pnl_bps": pl.Float64,
    "paper_pnl_usdt": pl.Float64,
    "label_status": pl.Utf8,
    "paper_tracking_status": pl.Utf8,
    "tracking_stage": pl.Utf8,
    "source_path_inside_bundle": pl.Utf8,
    "bundle_ts": pl.Utf8,
    "ingest_ts": pl.Utf8,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}

PAPER_RUN_SCHEMA = {
    "as_of_date": pl.Utf8,
    "paper_trade_id": pl.Utf8,
    "strategy_id": pl.Utf8,
    "strategy_version": pl.Utf8,
    "proposal_id": pl.Utf8,
    "paper_tracker_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "board_decision": pl.Utf8,
    "suggested_horizon": pl.Utf8,
    "horizon_hours": pl.Int64,
    "no_sample_reason": pl.Utf8,
    "would_enter": pl.Boolean,
    "would_exit": pl.Boolean,
    "would_size": pl.Float64,
    "would_size_usdt": pl.Float64,
    "paper_disabled_by_research_portfolio": pl.Boolean,
    "paper_trigger_type": pl.Utf8,
    "paper_trigger_reason": pl.Utf8,
    "source_candidate_symbol": pl.Utf8,
    "source_candidate_id": pl.Utf8,
    "symbol_match_verified": pl.Boolean,
    "paper_source": pl.Utf8,
    "paper_count_scope": pl.Utf8,
    "paper_pnl_bps": pl.Float64,
    "paper_pnl_bps_4h": pl.Float64,
    "paper_pnl_bps_8h": pl.Float64,
    "paper_pnl_bps_12h": pl.Float64,
    "paper_pnl_bps_24h": pl.Float64,
    "paper_pnl_bps_48h": pl.Float64,
    "paper_pnl_bps_72h": pl.Float64,
    "paper_pnl_usdt": pl.Float64,
    "arrival_bid": pl.Float64,
    "arrival_ask": pl.Float64,
    "arrival_mid": pl.Float64,
    "entry_decision_ts": pl.Utf8,
    "exit_decision_ts": pl.Utf8,
    "quote_timestamp": pl.Utf8,
    "quote_age_seconds": pl.Float64,
    "estimated_spread_bps": pl.Float64,
    "expected_order_type": pl.Utf8,
    "estimated_fill_px": pl.Float64,
    "virtual_entry_price": pl.Float64,
    "virtual_exit_price": pl.Float64,
    "virtual_quantity": pl.Float64,
    "cost_source": pl.Utf8,
    "cost_trust_level": pl.Utf8,
    "market_regime": pl.Utf8,
    "exit_reason": pl.Utf8,
    "gross_pnl_bps": pl.Float64,
    "net_pnl_bps": pl.Float64,
    "max_favorable_excursion": pl.Float64,
    "max_adverse_excursion": pl.Float64,
    "holding_bars": pl.Int64,
    "observability": pl.Utf8,
    "not_observable_reason": pl.Utf8,
    "valid_for_promotion": pl.Boolean,
    "paper_tracking_status": pl.Utf8,
    "tracking_stage": pl.Utf8,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "live_block_reason": pl.Utf8,
    "required_paper_days": pl.Int64,
    "required_slippage_coverage": pl.Float64,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}

PAPER_DAILY_SCHEMA = {
    "as_of_date": pl.Utf8,
    "source_schema_version": pl.Utf8,
    "proposal_id": pl.Utf8,
    "paper_tracker_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "paper_days": pl.Int64,
    "heartbeat_days": pl.Int64,
    "latest_board_decision": pl.Utf8,
    "latest_horizon": pl.Utf8,
    "heartbeat_day_count": pl.Int64,
    "entry_day_count": pl.Int64,
    "daily_v5_entry_count": pl.Int64,
    "daily_synthetic_would_enter_count": pl.Int64,
    "cumulative_v5_entry_count": pl.Int64,
    "cumulative_synthetic_would_enter_count": pl.Int64,
    "daily_would_enter_count": pl.Int64,
    "cumulative_would_enter_count": pl.Int64,
    "would_enter_count": pl.Int64,
    "daily_paper_pnl_observed_count": pl.Int64,
    "cumulative_paper_pnl_observed_count": pl.Int64,
    "paper_pnl_observed_count": pl.Int64,
    "count_scope": pl.Utf8,
    "paper_pnl_day_count": pl.Int64,
    "latest_paper_pnl_usdt": pl.Float64,
    "cumulative_paper_pnl_usdt": pl.Float64,
    "avg_paper_pnl_bps": pl.Float64,
    "paper_pnl_observed_count_by_horizon": pl.Utf8,
    "complete_count_by_horizon": pl.Utf8,
    "avg_paper_pnl_bps_by_horizon": pl.Utf8,
    "win_rate_by_horizon": pl.Utf8,
    "paper_pnl_day_count_by_horizon": pl.Utf8,
    "negative_entry_day_count": pl.Int64,
    "paper_negative_streak": pl.Int64,
    "latest_paper_trend": pl.Utf8,
    "paper_tracking_status": pl.Utf8,
    "tracking_stage": pl.Utf8,
    "required_paper_days": pl.Int64,
    "required_entry_day_count": pl.Int64,
    "required_slippage_coverage": pl.Float64,
    "arrival_mid_coverage": pl.Float64,
    "spread_observation_coverage": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "missing_cost_source_count": pl.Int64,
    "live_eligible": pl.Boolean,
    "live_block_reason": pl.Utf8,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}

PAPER_SLIPPAGE_SCHEMA = {
    "as_of_date": pl.Utf8,
    "proposal_id": pl.Utf8,
    "paper_tracker_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "paper_days": pl.Int64,
    "observed_slippage_count": pl.Int64,
    "required_observation_count": pl.Int64,
    "paper_slippage_coverage": pl.Float64,
    "required_slippage_coverage": pl.Float64,
    "arrival_mid_coverage": pl.Float64,
    "spread_observation_coverage": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "missing_cost_source_count": pl.Int64,
    "coverage_status": pl.Utf8,
    "paper_tracking_status": pl.Utf8,
    "tracking_stage": pl.Utf8,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}


@dataclass(frozen=True)
class PaperStrategyConfig:
    proposal_id: str
    strategy_candidate: str
    symbol: str
    paper_tracker_id: str = ""
    paper_size_usdt: float = 100.0
    required_paper_days: int = 14
    required_entry_day_count: int = DEFAULT_REQUIRED_ENTRY_DAYS_FOR_LIVE
    required_slippage_coverage: float = 0.8


PAPER_STRATEGIES = [
    PaperStrategyConfig(
        proposal_id="SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
        strategy_candidate="v5.sol_protect_alpha6_low_exception",
        symbol="SOL-USDT",
    ),
    PaperStrategyConfig(
        proposal_id=SOL_F4_PAPER_PROPOSAL_ID,
        strategy_candidate=SOL_F4_STRATEGY_CANDIDATE,
        symbol=SOL_F4_SYMBOL,
    ),
    PaperStrategyConfig(
        proposal_id=SOL_F3_PAPER_PROPOSAL_ID,
        strategy_candidate=SOL_F3_STRATEGY_CANDIDATE,
        symbol=SOL_F3_SYMBOL,
    ),
    PaperStrategyConfig(
        proposal_id=ETH_F3_PAPER_PROPOSAL_ID,
        strategy_candidate=ETH_F3_STRATEGY_CANDIDATE,
        symbol=ETH_F3_SYMBOL,
    ),
    PaperStrategyConfig(
        proposal_id=ETH_F4_VOLUME_EXPANSION_PAPER_PROPOSAL_ID,
        strategy_candidate=ETH_F4_VOLUME_EXPANSION_STRATEGY_CANDIDATE,
        symbol=ETH_F4_VOLUME_EXPANSION_SYMBOL,
    ),
    PaperStrategyConfig(
        proposal_id=BNB_F3_CURRENT_PROPOSAL_ID,
        strategy_candidate=BNB_F3_GENERIC_STRATEGY_CANDIDATE,
        symbol=BNB_SYMBOL,
        paper_tracker_id=BNB_F3_PAPER_PROPOSAL_ID,
    ),
    PaperStrategyConfig(
        proposal_id=BNB_F3_PAPER_PROPOSAL_ID,
        strategy_candidate=BNB_F3_STRATEGY_CANDIDATE,
        symbol=BNB_SYMBOL,
    ),
    PaperStrategyConfig(
        proposal_id=BNB_RISK_ON_BUY_PAPER_PROPOSAL_ID,
        strategy_candidate=BNB_RISK_ON_BUY_STRATEGY_CANDIDATE,
        symbol=BNB_SYMBOL,
    ),
]


class PaperStrategyTrackingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    paper_strategy_runs: int = Field(ge=0)
    paper_strategy_daily: int = Field(ge=0)
    paper_slippage_coverage: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_paper_strategy_tracking(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> PaperStrategyTrackingResult:
    root = Path(lake_root)
    board = read_parquet_dataset(root / ALPHA_DISCOVERY_BOARD_DATASET)
    day = _resolve_as_of_date(board, as_of_date)
    warnings: list[str] = []
    v5_runs_raw = latest_v5_paper_frame(read_parquet_dataset(root / V5_PAPER_STRATEGY_RUN_DATASET))
    v5_daily_raw = latest_v5_paper_frame(
        read_parquet_dataset(root / V5_PAPER_STRATEGY_DAILY_DATASET)
    )
    v5_slippage_raw = latest_v5_paper_frame(
        read_parquet_dataset(root / V5_PAPER_SLIPPAGE_COVERAGE_DATASET)
    )
    v5_candidate_raw = latest_v5_paper_frame(
        read_parquet_dataset(root / V5_CANDIDATE_EVENT_DATASET)
    )
    research_portfolio = read_parquet_dataset(root / RESEARCH_PORTFOLIO_STATUS_DATASET)
    if any(
        not frame.is_empty()
        for frame in [v5_runs_raw, v5_daily_raw, v5_slippage_raw, v5_candidate_raw]
    ):
        runs = build_paper_strategy_runs_from_v5(
            v5_runs_raw,
            research_portfolio=research_portfolio,
            candidate_events=v5_candidate_raw,
        )
        daily = build_paper_strategy_daily_from_v5(v5_daily_raw)
        if not runs.is_empty():
            daily = enrich_paper_strategy_daily_from_runs(
                daily,
                runs,
                as_of_date=day,
            )
        if daily.is_empty() and not runs.is_empty():
            daily = build_paper_strategy_daily_from_runs(runs, as_of_date=day)
        coverage = build_paper_slippage_coverage_from_v5(v5_slippage_raw, daily=daily)
        if not daily.is_empty():
            coverage = _replace_by_key(
                build_paper_slippage_coverage(daily, as_of_date=day),
                coverage,
                ["proposal_id", "strategy_candidate", "symbol"],
            )
        if runs.is_empty():
            warnings.append("paper_tracking_v5_runs_missing")
    else:
        runs, daily, coverage, warnings = build_pending_paper_strategy_tracking(
            board,
            as_of_date=day,
        )
    run_rows = _publish_runs(root, runs)
    daily_rows = _publish_daily(root, daily)
    coverage_rows = _publish_slippage(root, coverage)
    return PaperStrategyTrackingResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        paper_strategy_runs=run_rows,
        paper_strategy_daily=daily_rows,
        paper_slippage_coverage=coverage_rows,
        warnings=warnings,
    )


def build_paper_strategy_runs(
    board: pl.DataFrame,
    *,
    as_of_date: date,
) -> tuple[pl.DataFrame, list[str]]:
    warnings: list[str] = []
    if board.is_empty():
        return _empty_frame(PAPER_RUN_SCHEMA), ["paper_tracking_alpha_discovery_board_empty"]
    normalized = normalize_alpha_discovery_board_decisions(board)
    if normalized.is_empty():
        return _empty_frame(PAPER_RUN_SCHEMA), ["paper_tracking_alpha_discovery_board_empty"]
    rows = [
        row
        for row in normalized.to_dicts()
        if str(row.get("as_of_date") or "") == as_of_date.isoformat()
        and str(row.get("decision") or "").upper() == "PAPER_READY"
    ]
    if not rows:
        return _empty_frame(PAPER_RUN_SCHEMA), ["paper_tracking_no_paper_ready_rows"]

    by_key = {(cfg.strategy_candidate, cfg.symbol): cfg for cfg in PAPER_STRATEGIES}
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("strategy_candidate") or ""), str(row.get("symbol") or ""))
        if key not in by_key:
            continue
        current = selected.get(key)
        if current is None or _proposal_rank(row) > _proposal_rank(current):
            selected[key] = row

    if not selected:
        return _empty_frame(PAPER_RUN_SCHEMA), ["paper_tracking_no_configured_sol_candidate_ready"]

    created_at = datetime.now(UTC).isoformat()
    run_rows = [
        _pending_run_row(row, by_key[key], created_at) for key, row in sorted(selected.items())
    ]
    return pl.DataFrame(run_rows, schema=PAPER_RUN_SCHEMA, orient="row"), warnings


def build_pending_paper_strategy_tracking(
    board: pl.DataFrame,
    *,
    as_of_date: date,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str]]:
    runs, warnings = build_paper_strategy_runs(board, as_of_date=as_of_date)
    daily = build_pending_paper_strategy_daily(runs, as_of_date=as_of_date)
    coverage = build_paper_slippage_coverage(daily, as_of_date=as_of_date)
    if not runs.is_empty():
        warnings.append("paper_tracking_status=waiting_for_v5_paper_telemetry")
    return runs, daily, coverage, warnings


def build_pending_paper_strategy_daily(
    pending_runs: pl.DataFrame,
    *,
    as_of_date: date,
) -> pl.DataFrame:
    if pending_runs.is_empty():
        return _empty_frame(PAPER_DAILY_SCHEMA)
    created_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for row in pending_runs.to_dicts():
        cfg = _config_for(row.get("proposal_id"), row.get("strategy_candidate"), row.get("symbol"))
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "proposal_id": row.get("proposal_id"),
                "strategy_candidate": row.get("strategy_candidate"),
                "symbol": row.get("symbol"),
                "recommended_mode": "paper",
                "paper_days": 0,
                "heartbeat_days": 0,
                "latest_board_decision": str(row.get("board_decision") or "PAPER_READY"),
                "latest_horizon": str(row.get("suggested_horizon") or ""),
                "heartbeat_day_count": 0,
                "entry_day_count": 0,
                "daily_v5_entry_count": 0,
                "daily_synthetic_would_enter_count": 0,
                "cumulative_v5_entry_count": 0,
                "cumulative_synthetic_would_enter_count": 0,
                "daily_would_enter_count": 0,
                "cumulative_would_enter_count": 0,
                "would_enter_count": 0,
                "daily_paper_pnl_observed_count": 0,
                "cumulative_paper_pnl_observed_count": 0,
                "paper_pnl_observed_count": 0,
                "count_scope": "cumulative",
                "paper_pnl_day_count": 0,
                "latest_paper_pnl_usdt": None,
                "cumulative_paper_pnl_usdt": 0.0,
                "avg_paper_pnl_bps": None,
                "paper_pnl_observed_count_by_horizon": "{}",
                "complete_count_by_horizon": "{}",
                "avg_paper_pnl_bps_by_horizon": "{}",
                "win_rate_by_horizon": "{}",
                "paper_pnl_day_count_by_horizon": "{}",
                "negative_entry_day_count": 0,
                "paper_negative_streak": 0,
                "latest_paper_trend": "waiting_for_v5_paper_telemetry",
                "paper_tracking_status": "waiting_for_v5_paper_telemetry",
                "tracking_stage": "proposed_paper_strategy",
                "required_paper_days": cfg.required_paper_days,
                "required_entry_day_count": cfg.required_entry_day_count,
                "required_slippage_coverage": cfg.required_slippage_coverage,
                "arrival_mid_coverage": 0.0,
                "spread_observation_coverage": 0.0,
                "cost_source_mix": str(row.get("cost_source_mix") or ""),
                "missing_cost_source_count": 0
                if str(row.get("cost_source_mix") or "").strip()
                else 1,
                "live_eligible": False,
                "live_block_reason": safe_json_dumps(
                    sorted(
                        {
                            *_json_list(row.get("live_block_reason")),
                            "waiting_for_v5_paper_telemetry",
                            "no_paper_days",
                            "insufficient_entry_days",
                            "no_live_slippage_coverage",
                            "insufficient_arrival_mid_coverage",
                        }
                    )
                ),
                "created_at": created_at,
                "source": PAPER_TRACKING_SOURCE,
                "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
            }
        )
    return pl.DataFrame(rows, schema=PAPER_DAILY_SCHEMA, orient="row")


def build_paper_strategy_daily_from_runs(
    runs: pl.DataFrame,
    *,
    as_of_date: date,
) -> pl.DataFrame:
    return _daily_from_runs(runs, as_of_date=as_of_date)


def build_paper_strategy_daily(
    lake_root: str | Path,
    new_runs: pl.DataFrame,
    *,
    as_of_date: date,
) -> pl.DataFrame:
    return _daily_from_runs(new_runs, as_of_date=as_of_date)


def _daily_from_runs(runs: pl.DataFrame, *, as_of_date: date) -> pl.DataFrame:
    if runs.is_empty():
        return _empty_frame(PAPER_DAILY_SCHEMA)
    created_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for group in _group_run_rows(runs.to_dicts()).values():
        cfg = _config_for(
            group["proposal_id"],
            group["strategy_candidate"],
            group["symbol"],
        )
        run_rows = group["rows"]
        entry_rows = [row for row in run_rows if bool(row.get("would_enter")) is True]
        v5_entry_rows = [row for row in entry_rows if _paper_source(row) == "v5_telemetry"]
        synthetic_entry_rows = [row for row in entry_rows if _paper_source(row) != "v5_telemetry"]
        if not run_rows:
            continue
        run_rows.sort(key=lambda row: str(row.get("as_of_date") or ""))
        latest = run_rows[-1]
        as_of_text = as_of_date.isoformat()
        daily_run_rows = [
            row for row in run_rows if str(row.get("as_of_date") or "")[:10] == as_of_text
        ]
        heartbeat_days = _heartbeat_day_count(run_rows)
        entry_days = len({str(row.get("as_of_date") or "") for row in entry_rows})
        daily_entry_rows = [row for row in daily_run_rows if bool(row.get("would_enter")) is True]
        daily_v5_entry_rows = [
            row for row in daily_entry_rows if _paper_source(row) == "v5_telemetry"
        ]
        daily_synthetic_entry_rows = [
            row for row in daily_entry_rows if _paper_source(row) != "v5_telemetry"
        ]
        pnl_rows = _paper_pnl_rows(run_rows)
        daily_pnl_rows = _paper_pnl_rows(daily_run_rows)
        paper_pnl_days = _paper_pnl_day_count(pnl_rows)
        horizon_stats = _paper_pnl_horizon_stats(pnl_rows)
        negative_trend = _paper_negative_trend_stats(run_rows)
        latest_reason = _paper_block_reason_list(latest.get("live_block_reason"))
        quality = _paper_cost_quality(run_rows)
        block_reasons = _paper_live_block_reasons(
            heartbeat_day_count=heartbeat_days,
            entry_day_count=entry_days,
            paper_pnl_observed_count=len(pnl_rows),
            paper_pnl_day_count=paper_pnl_days,
            required_paper_days=cfg.required_paper_days,
            required_entry_day_count=cfg.required_entry_day_count,
            required_slippage_coverage=cfg.required_slippage_coverage,
            arrival_mid_coverage=quality["arrival_mid_coverage"],
            paper_slippage_coverage=quality["paper_slippage_coverage"],
            cost_source_mix=quality["cost_source_mix"],
        )
        latest_board_decision = str(latest.get("board_decision") or "")
        review = _paper_strategy_review(
            cfg=cfg,
            latest_board_decision=latest_board_decision,
            horizon_stats=horizon_stats,
            paper_negative_streak=negative_trend["paper_negative_streak"],
        )
        latest_board_decision = review["latest_board_decision"]
        block_reasons = [*block_reasons, *review["live_block_reasons"]]
        if (
            "paper_negative_24h_or_48h_streak" in latest_reason
            or "paper_negative_24h_or_48h_streak" in block_reasons
        ):
            negative_trend = {
                **negative_trend,
                "negative_entry_day_count": max(
                    int(negative_trend.get("negative_entry_day_count") or 0),
                    2,
                ),
                "paper_negative_streak": max(
                    int(negative_trend.get("paper_negative_streak") or 0),
                    2,
                ),
                "latest_paper_trend": "negative_24h_or_48h_streak",
            }
        live_eligible = not block_reasons
        tracking_stage = (
            "completed_paper_observations"
            if paper_pnl_days >= cfg.required_paper_days
            else "active_paper_strategy"
        )
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "proposal_id": cfg.proposal_id,
                "strategy_candidate": cfg.strategy_candidate,
                "symbol": cfg.symbol,
                "recommended_mode": "paper",
                "paper_days": heartbeat_days,
                "heartbeat_days": heartbeat_days,
                "latest_board_decision": latest_board_decision,
                "latest_horizon": str(latest.get("suggested_horizon") or ""),
                "heartbeat_day_count": heartbeat_days,
                "entry_day_count": entry_days,
                "daily_v5_entry_count": len(daily_v5_entry_rows),
                "daily_synthetic_would_enter_count": len(daily_synthetic_entry_rows),
                "cumulative_v5_entry_count": len(v5_entry_rows),
                "cumulative_synthetic_would_enter_count": len(synthetic_entry_rows),
                "daily_would_enter_count": len(daily_entry_rows),
                "cumulative_would_enter_count": len(entry_rows),
                "would_enter_count": len(entry_rows),
                "daily_paper_pnl_observed_count": len(daily_pnl_rows),
                "cumulative_paper_pnl_observed_count": len(pnl_rows),
                "paper_pnl_observed_count": len(pnl_rows),
                "count_scope": "cumulative",
                "paper_pnl_day_count": paper_pnl_days,
                "latest_paper_pnl_usdt": _optional_float(pnl_rows[-1].get("paper_pnl_usdt"))
                if pnl_rows
                else None,
                "cumulative_paper_pnl_usdt": sum(
                    _optional_float(row.get("paper_pnl_usdt")) or 0.0 for row in pnl_rows
                ),
                "avg_paper_pnl_bps": _mean(
                    value for row in pnl_rows for value in _paper_pnl_bps_values(row)
                ),
                "paper_pnl_observed_count_by_horizon": safe_json_dumps(
                    horizon_stats["observed_count_by_horizon"]
                ),
                "complete_count_by_horizon": safe_json_dumps(
                    horizon_stats["complete_count_by_horizon"]
                ),
                "avg_paper_pnl_bps_by_horizon": safe_json_dumps(
                    horizon_stats["avg_bps_by_horizon"]
                ),
                "win_rate_by_horizon": safe_json_dumps(horizon_stats["win_rate_by_horizon"]),
                "paper_pnl_day_count_by_horizon": safe_json_dumps(
                    horizon_stats["day_count_by_horizon"]
                ),
                "negative_entry_day_count": negative_trend["negative_entry_day_count"],
                "paper_negative_streak": negative_trend["paper_negative_streak"],
                "latest_paper_trend": negative_trend["latest_paper_trend"],
                "paper_tracking_status": V5_PAPER_TRACKING_STATUS,
                "tracking_stage": tracking_stage,
                "required_paper_days": cfg.required_paper_days,
                "required_entry_day_count": cfg.required_entry_day_count,
                "required_slippage_coverage": cfg.required_slippage_coverage,
                "arrival_mid_coverage": quality["arrival_mid_coverage"],
                "spread_observation_coverage": quality["spread_observation_coverage"],
                "cost_source_mix": safe_json_dumps(quality["cost_source_mix"]),
                "missing_cost_source_count": quality["missing_cost_source_count"],
                "live_eligible": live_eligible,
                "live_block_reason": safe_json_dumps(sorted({*latest_reason, *block_reasons})),
                "created_at": created_at,
                "source": PAPER_TRACKING_SOURCE,
                "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
            }
        )
    return pl.DataFrame(rows, schema=PAPER_DAILY_SCHEMA, orient="row")


def build_paper_slippage_coverage(
    daily: pl.DataFrame,
    *,
    as_of_date: date,
) -> pl.DataFrame:
    if daily.is_empty():
        return _empty_frame(PAPER_SLIPPAGE_SCHEMA)
    created_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for row in daily.to_dicts():
        paper_days = int(_optional_float(row.get("paper_days")) or 0)
        required = max(paper_days, 1)
        arrival_mid_coverage = _optional_float(row.get("arrival_mid_coverage")) or 0.0
        spread_observation_coverage = _optional_float(row.get("spread_observation_coverage")) or 0.0
        coverage = min(arrival_mid_coverage, spread_observation_coverage)
        observed = int(round(coverage * required))
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "proposal_id": row.get("proposal_id"),
                "strategy_candidate": row.get("strategy_candidate"),
                "symbol": row.get("symbol"),
                "paper_days": paper_days,
                "observed_slippage_count": observed,
                "required_observation_count": required,
                "paper_slippage_coverage": coverage,
                "required_slippage_coverage": _optional_float(row.get("required_slippage_coverage"))
                or 0.8,
                "arrival_mid_coverage": arrival_mid_coverage,
                "spread_observation_coverage": spread_observation_coverage,
                "cost_source_mix": str(row.get("cost_source_mix") or ""),
                "missing_cost_source_count": int(
                    _optional_float(row.get("missing_cost_source_count")) or 0
                ),
                "coverage_status": _coverage_status(row, coverage),
                "paper_tracking_status": str(
                    row.get("paper_tracking_status") or V5_PAPER_TRACKING_STATUS
                ),
                "tracking_stage": str(row.get("tracking_stage") or "active_paper_strategy"),
                "created_at": created_at,
                "source": PAPER_TRACKING_SOURCE,
                "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
            }
        )
    return pl.DataFrame(rows, schema=PAPER_SLIPPAGE_SCHEMA, orient="row")


def _publish_runs(lake_root: Path, runs: pl.DataFrame) -> int:
    path = lake_root / PAPER_STRATEGY_RUNS_DATASET
    write_parquet_dataset(runs, path)
    return runs.height


def _publish_daily(lake_root: Path, daily: pl.DataFrame) -> int:
    path = lake_root / PAPER_STRATEGY_DAILY_DATASET
    write_parquet_dataset(daily, path)
    return daily.height


def _publish_slippage(lake_root: Path, coverage: pl.DataFrame) -> int:
    path = lake_root / PAPER_SLIPPAGE_COVERAGE_DATASET
    write_parquet_dataset(coverage, path)
    return coverage.height


def _combine_unique_runs(existing: pl.DataFrame, new_runs: pl.DataFrame) -> pl.DataFrame:
    return _replace_by_key(
        existing,
        new_runs,
        ["proposal_id", "as_of_date", "suggested_horizon"],
    )


def _replace_by_key(
    existing: pl.DataFrame,
    new_frame: pl.DataFrame,
    keys: list[str],
) -> pl.DataFrame:
    frames = [frame for frame in [existing, new_frame] if not frame.is_empty()]
    if not frames:
        schema = dict(new_frame.schema) if new_frame.schema else None
        return pl.DataFrame(schema=schema) if schema else pl.DataFrame()
    combined = pl.concat(frames, how="diagonal_relaxed")
    available = [key for key in keys if key in combined.columns]
    if available:
        combined = combined.unique(subset=available, keep="last", maintain_order=True)
    return combined


def _pending_run_row(
    row: dict[str, Any],
    cfg: PaperStrategyConfig,
    created_at: str,
) -> dict[str, Any]:
    avg_net_bps = _optional_float(row.get("avg_net_bps"))
    cost_source_mix = str(row.get("cost_source_mix") or "")
    return {
        "as_of_date": str(row.get("as_of_date") or ""),
        "proposal_id": cfg.proposal_id,
        "strategy_candidate": cfg.strategy_candidate,
        "symbol": cfg.symbol,
        "recommended_mode": "paper",
        "board_decision": "PAPER_READY",
        "suggested_horizon": _suggested_horizon(row),
        "horizon_hours": int(_optional_float(row.get("horizon_hours")) or 0),
        "no_sample_reason": "waiting_for_v5_paper_telemetry",
        "would_enter": False,
        "would_exit": False,
        "would_size": 0.0,
        "would_size_usdt": 0.0,
        "paper_disabled_by_research_portfolio": False,
        "paper_trigger_type": "paper_ready_proposal",
        "paper_trigger_reason": "waiting_for_v5_paper_telemetry",
        "paper_source": "opportunity_paper",
        "paper_count_scope": "daily",
        "paper_pnl_bps": None,
        "paper_pnl_bps_4h": None,
        "paper_pnl_bps_8h": None,
        "paper_pnl_bps_12h": None,
        "paper_pnl_bps_24h": None,
        "paper_pnl_bps_48h": None,
        "paper_pnl_bps_72h": None,
        "paper_pnl_usdt": None,
        "arrival_bid": None,
        "arrival_ask": None,
        "arrival_mid": None,
        "estimated_spread_bps": None,
        "expected_order_type": "",
        "estimated_fill_px": None,
        "cost_source": "",
        "paper_tracking_status": "waiting_for_v5_paper_telemetry",
        "tracking_stage": "proposed_paper_strategy",
        "sample_count": int(_optional_float(row.get("sample_count")) or 0),
        "complete_sample_count": int(_optional_float(row.get("complete_sample_count")) or 0),
        "avg_net_bps": avg_net_bps,
        "p25_net_bps": _optional_float(row.get("p25_net_bps")),
        "win_rate": _optional_float(row.get("win_rate")),
        "cost_source_mix": cost_source_mix,
        "live_block_reason": safe_json_dumps(_live_block_reasons(row)),
        "required_paper_days": cfg.required_paper_days,
        "required_slippage_coverage": cfg.required_slippage_coverage,
        "created_at": created_at,
        "source": PAPER_TRACKING_SOURCE,
        "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
    }


def build_paper_strategy_runs_from_v5(
    frame: pl.DataFrame,
    *,
    research_portfolio: pl.DataFrame | None = None,
    candidate_events: pl.DataFrame | None = None,
) -> pl.DataFrame:
    candidate_frame = candidate_events if candidate_events is not None else pl.DataFrame()
    if frame.is_empty() and candidate_frame.is_empty():
        return _empty_frame(PAPER_RUN_SCHEMA)
    created_at = datetime.now(UTC).isoformat()
    portfolio_frame = research_portfolio if research_portfolio is not None else pl.DataFrame()
    portfolio_overrides = _paper_portfolio_overrides(portfolio_frame)
    rows = (
        [
            _v5_run_row(row, created_at, portfolio_overrides=portfolio_overrides)
            for row in frame.to_dicts()
        ]
        if not frame.is_empty()
        else []
    )
    rows.extend(_sol_f4_factor_candidate_run_rows(candidate_frame, created_at, rows))
    rows.extend(_bnb_paper_candidate_run_rows(candidate_frame, created_at, rows))
    if not rows:
        return _empty_frame(PAPER_RUN_SCHEMA)
    return pl.DataFrame(rows, schema=PAPER_RUN_SCHEMA, orient="row")


def build_paper_strategy_runs_report_from_v5(
    frame: pl.DataFrame,
    *,
    research_portfolio: pl.DataFrame | None = None,
    candidate_events: pl.DataFrame | None = None,
) -> pl.DataFrame:
    candidate_frame = candidate_events if candidate_events is not None else pl.DataFrame()
    if frame.is_empty() and candidate_frame.is_empty():
        return _empty_frame(PAPER_RUN_REPORT_SCHEMA)
    created_at = datetime.now(UTC).isoformat()
    portfolio_frame = research_portfolio if research_portfolio is not None else pl.DataFrame()
    portfolio_overrides = _paper_portfolio_overrides(portfolio_frame)
    rows = (
        [
            _v5_run_report_row(row, created_at, portfolio_overrides=portfolio_overrides)
            for row in frame.to_dicts()
        ]
        if not frame.is_empty()
        else []
    )
    rows.extend(_sol_f4_factor_candidate_report_rows(candidate_frame, created_at, rows))
    rows.extend(_bnb_paper_candidate_report_rows(candidate_frame, created_at, rows))
    if not rows:
        return _empty_frame(PAPER_RUN_REPORT_SCHEMA)
    return pl.DataFrame(rows, schema=PAPER_RUN_REPORT_SCHEMA, orient="row")


def latest_v5_paper_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    timestamp_column = next(
        (column for column in ["bundle_ts", "ingest_ts"] if column in frame.columns),
        None,
    )
    if timestamp_column is None:
        return frame
    try:
        latest = frame.select(pl.col(timestamp_column).max().alias("latest"))["latest"][0]
    except Exception:
        return frame
    if latest is None or str(latest).strip() == "":
        return frame
    return frame.filter(pl.col(timestamp_column) == latest)


def build_paper_strategy_daily_from_v5(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_frame(PAPER_DAILY_SCHEMA)
    created_at = datetime.now(UTC).isoformat()
    rows = [_v5_daily_row(row, created_at) for row in frame.to_dicts()]
    return _enrich_daily_negative_streaks(
        pl.DataFrame(rows, schema=PAPER_DAILY_SCHEMA, orient="row")
    )


def paper_strategy_summary_md(daily: pl.DataFrame) -> str:
    lines = [
        "# 纸面策略日汇总",
        "",
        (
            "口径说明：`daily_*` 是当天新增，`cumulative_*` 和兼容字段 "
            "`would_enter_count` / `paper_pnl_observed_count` 是累计总值；"
            "V5 telemetry entry 与中台 synthetic would-enter 分开统计。"
        ),
        "",
    ]
    if daily.is_empty():
        lines.append("- 当前没有纸面策略日汇总数据。")
        return "\n".join(lines).rstrip() + "\n"
    rows = daily.to_dicts()
    latest_as_of_date = max(str(row.get("as_of_date") or "")[:10] for row in rows)
    display_rows = [
        row for row in rows if str(row.get("as_of_date") or "")[:10] == latest_as_of_date
    ] or rows
    daily_entries = sum(
        int(_optional_float(row.get("daily_would_enter_count")) or 0) for row in display_rows
    )
    cumulative_entries = sum(
        int(_optional_float(row.get("cumulative_would_enter_count")) or 0) for row in display_rows
    )
    daily_pnl = sum(
        int(_optional_float(row.get("daily_paper_pnl_observed_count")) or 0) for row in display_rows
    )
    cumulative_pnl = sum(
        int(_optional_float(row.get("cumulative_paper_pnl_observed_count")) or 0)
        for row in display_rows
    )
    daily_v5_entries = sum(
        int(_optional_float(row.get("daily_v5_entry_count")) or 0) for row in display_rows
    )
    daily_synthetic_entries = sum(
        int(_optional_float(row.get("daily_synthetic_would_enter_count")) or 0)
        for row in display_rows
    )
    lines.extend(
        [
            f"- 今日新 entry 数: {daily_entries}",
            f"- 今日 V5 实际 paper entry: {daily_v5_entries}",
            f"- 今日中台 synthetic would_enter: {daily_synthetic_entries}",
            f"- 累计 entry 数: {cumulative_entries}",
            f"- 今日新增 PnL 标签数: {daily_pnl}",
            f"- 累计 PnL 标签数: {cumulative_pnl}",
            "",
            (
                "| 策略 | 标的 | 今日 V5 entry | 今日 synthetic | 累计 entry | "
                "今日 PnL 标签 | 累计 PnL 标签 | 状态 | 趋势 |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in sorted(display_rows, key=lambda item: str(item.get("proposal_id") or "")):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("proposal_id") or row.get("strategy_candidate") or ""),
                    str(row.get("symbol") or ""),
                    str(int(_optional_float(row.get("daily_v5_entry_count")) or 0)),
                    str(int(_optional_float(row.get("daily_synthetic_would_enter_count")) or 0)),
                    str(int(_optional_float(row.get("cumulative_would_enter_count")) or 0)),
                    str(int(_optional_float(row.get("daily_paper_pnl_observed_count")) or 0)),
                    str(int(_optional_float(row.get("cumulative_paper_pnl_observed_count")) or 0)),
                    str(row.get("latest_board_decision") or ""),
                    str(row.get("latest_paper_trend") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _enrich_daily_negative_streaks(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in frame.to_dicts():
        groups.setdefault(_paper_daily_key(row), []).append(row)
    enriched: list[dict[str, Any]] = []
    for group_rows in groups.values():
        negative_count = 0
        streak = 0
        for row in sorted(group_rows, key=lambda item: str(item.get("as_of_date") or "")):
            updated = dict(row)
            observed = _daily_row_has_24h_48h_pnl(updated)
            is_negative = _daily_row_24h_48h_negative(updated)
            if observed:
                if is_negative:
                    negative_count += 1
                    streak += 1
                else:
                    streak = 0
                updated["negative_entry_day_count"] = negative_count
                updated["paper_negative_streak"] = streak
                if streak >= 2:
                    updated["latest_paper_trend"] = "negative_24h_or_48h_streak"
                elif is_negative:
                    updated["latest_paper_trend"] = "latest_entry_day_negative"
                else:
                    updated["latest_paper_trend"] = "latest_entry_day_non_negative"
            else:
                existing_trend = str(updated.get("latest_paper_trend") or "")
                existing_streak = int(_optional_float(updated.get("paper_negative_streak")) or 0)
                existing_count = int(_optional_float(updated.get("negative_entry_day_count")) or 0)
                carry_streak = max(streak, existing_streak)
                carry_count = max(negative_count, existing_count)
                if existing_trend == "negative_24h_or_48h_streak" and carry_streak < 2:
                    carry_streak = 2
                    carry_count = max(carry_count, 2)
                updated["negative_entry_day_count"] = carry_count
                updated["paper_negative_streak"] = carry_streak
                if carry_streak >= 2:
                    updated["latest_paper_trend"] = "negative_24h_or_48h_streak"
                elif existing_trend in {"", "not_observable", "waiting_for_24h_48h_labels"}:
                    updated["latest_paper_trend"] = "waiting_for_24h_48h_labels"
            updated = _with_paper_strategy_review(
                updated,
                cfg=_config_for(
                    updated.get("proposal_id"),
                    updated.get("strategy_candidate"),
                    updated.get("symbol"),
                ),
            )
            enriched.append(updated)
    return pl.DataFrame(enriched, schema=PAPER_DAILY_SCHEMA, orient="row")


def enrich_paper_strategy_daily_from_runs(
    daily: pl.DataFrame,
    runs: pl.DataFrame,
    *,
    as_of_date: date,
) -> pl.DataFrame:
    run_daily = build_paper_strategy_daily_from_runs(runs, as_of_date=as_of_date)
    if daily.is_empty():
        return run_daily
    if run_daily.is_empty():
        return daily
    metrics: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in run_daily.to_dicts():
        for key in _paper_daily_key_aliases(row):
            metrics[key] = row
    enriched: list[dict[str, Any]] = []
    metric_fields = [
        "paper_days",
        "heartbeat_days",
        "heartbeat_day_count",
        "entry_day_count",
        "daily_synthetic_would_enter_count",
        "cumulative_v5_entry_count",
        "cumulative_synthetic_would_enter_count",
        "cumulative_would_enter_count",
        "would_enter_count",
        "cumulative_paper_pnl_observed_count",
        "paper_pnl_observed_count",
        "paper_pnl_day_count",
        "negative_entry_day_count",
        "paper_negative_streak",
        "missing_cost_source_count",
    ]
    fallback_fields = [
        "latest_paper_pnl_usdt",
        "cumulative_paper_pnl_usdt",
        "avg_paper_pnl_bps",
        "arrival_mid_coverage",
        "spread_observation_coverage",
        "cost_source_mix",
    ]
    run_preferred_fields = [
        "paper_pnl_observed_count_by_horizon",
        "complete_count_by_horizon",
        "avg_paper_pnl_bps_by_horizon",
        "win_rate_by_horizon",
        "paper_pnl_day_count_by_horizon",
    ]
    for row in daily.to_dicts():
        metric: dict[str, Any] = {}
        for key in _paper_daily_key_aliases(row):
            metric = metrics.get(key, {})
            if metric:
                break
        updated = dict(row)
        authoritative_daily = (
            str(updated.get("source_schema_version") or "") == "v5.generic_paper_runtime.v1"
        )
        row_as_of_date = str(updated.get("as_of_date") or "")[:10]
        updated["daily_v5_entry_count"] = int(
            _optional_float(updated.get("daily_v5_entry_count"))
            or _optional_float(updated.get("daily_would_enter_count"))
            or 0
        )
        existing_cumulative_v5_count = int(
            _optional_float(updated.get("cumulative_v5_entry_count"))
            or _optional_float(updated.get("cumulative_would_enter_count"))
            or 0
        )
        for field in metric_fields:
            existing_value = int(_optional_float(updated.get(field)) or 0)
            metric_value = int(_optional_float(metric.get(field)) or 0)
            updated[field] = (
                max(existing_value, metric_value) if authoritative_daily else metric_value
            )
        if not updated.get("strategy_candidate") and metric.get("strategy_candidate"):
            updated["strategy_candidate"] = metric.get("strategy_candidate")
        if row_as_of_date != as_of_date.isoformat():
            updated["daily_synthetic_would_enter_count"] = 0
        if metric:
            metric_v5_count = _optional_float(metric.get("cumulative_v5_entry_count"))
            updated["cumulative_v5_entry_count"] = max(
                existing_cumulative_v5_count,
                int(metric_v5_count or 0),
            )
            total_would_enter_count = int(
                _optional_float(updated.get("cumulative_v5_entry_count")) or 0
            ) + int(_optional_float(updated.get("cumulative_synthetic_would_enter_count")) or 0)
            updated["cumulative_would_enter_count"] = total_would_enter_count
            updated["would_enter_count"] = total_would_enter_count
        for field in ["daily_would_enter_count", "daily_paper_pnl_observed_count"]:
            if updated.get(field) in {None, ""}:
                updated[field] = int(_optional_float(metric.get(field)) or 0)
        updated["count_scope"] = "cumulative"
        for field in fallback_fields:
            if updated.get(field) in {None, ""} and metric.get(field) not in {None, ""}:
                updated[field] = metric.get(field)
        for field in run_preferred_fields:
            if metric.get(field) not in {None, "", "{}"}:
                updated[field] = metric.get(field)
        for field in [
            "arrival_mid_coverage",
            "spread_observation_coverage",
            "cost_source_mix",
            "latest_paper_trend",
        ]:
            if metric.get(field) not in {None, ""}:
                updated[field] = metric.get(field)
        updated["missing_cost_source_count"] = (
            1 if _is_missing_cost_source(updated.get("cost_source_mix")) else 0
        )
        required_days = int(_optional_float(updated.get("required_paper_days")) or 14)
        required_entries = int(
            _optional_float(updated.get("required_entry_day_count"))
            or DEFAULT_REQUIRED_ENTRY_DAYS_FOR_LIVE
        )
        required_coverage = _optional_float(updated.get("required_slippage_coverage")) or 0.8
        block_reasons = _paper_live_block_reasons(
            heartbeat_day_count=int(_optional_float(updated.get("heartbeat_day_count")) or 0),
            entry_day_count=int(_optional_float(updated.get("entry_day_count")) or 0),
            paper_pnl_observed_count=int(
                _optional_float(updated.get("paper_pnl_observed_count")) or 0
            ),
            paper_pnl_day_count=int(_optional_float(updated.get("paper_pnl_day_count")) or 0),
            required_paper_days=required_days,
            required_entry_day_count=required_entries,
            required_slippage_coverage=required_coverage,
            arrival_mid_coverage=_optional_float(updated.get("arrival_mid_coverage")) or 0.0,
            paper_slippage_coverage=min(
                _optional_float(updated.get("arrival_mid_coverage")) or 0.0,
                _optional_float(updated.get("spread_observation_coverage")) or 0.0,
            ),
            cost_source_mix=updated.get("cost_source_mix"),
        )
        review = _paper_strategy_review(
            cfg=_config_for(
                updated.get("proposal_id"),
                updated.get("strategy_candidate"),
                updated.get("symbol"),
            ),
            latest_board_decision=str(updated.get("latest_board_decision") or ""),
            horizon_stats={
                "observed_count_by_horizon": _json_dict(
                    updated.get("paper_pnl_observed_count_by_horizon")
                ),
                "complete_count_by_horizon": _json_dict(updated.get("complete_count_by_horizon")),
                "avg_bps_by_horizon": _json_dict(updated.get("avg_paper_pnl_bps_by_horizon")),
                "day_count_by_horizon": _json_dict(updated.get("paper_pnl_day_count_by_horizon")),
            },
            paper_negative_streak=int(_optional_float(updated.get("paper_negative_streak")) or 0),
        )
        updated["latest_board_decision"] = review["latest_board_decision"]
        block_reasons = [*block_reasons, *review["live_block_reasons"]]
        updated["live_eligible"] = not block_reasons
        updated["live_block_reason"] = safe_json_dumps(
            sorted({*_paper_block_reason_list(updated.get("live_block_reason")), *block_reasons})
        )
        enriched.append(updated)
    return pl.DataFrame(enriched, schema=PAPER_DAILY_SCHEMA, orient="row")


def build_paper_slippage_coverage_from_v5(
    frame: pl.DataFrame,
    *,
    daily: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_frame(PAPER_SLIPPAGE_SCHEMA)
    created_at = datetime.now(UTC).isoformat()
    rows = [_v5_slippage_row(row, created_at) for row in frame.to_dicts()]
    coverage = pl.DataFrame(rows, schema=PAPER_SLIPPAGE_SCHEMA, orient="row")
    return _align_slippage_cost_source_mix(coverage, daily)


def _align_slippage_cost_source_mix(
    coverage: pl.DataFrame,
    daily: pl.DataFrame | None,
) -> pl.DataFrame:
    if coverage.is_empty() or daily is None or daily.is_empty():
        return coverage
    daily_by_key = {_paper_daily_key(row): row for row in daily.to_dicts()}
    rows: list[dict[str, Any]] = []
    for row in coverage.to_dicts():
        updated = dict(row)
        source = daily_by_key.get(_paper_daily_key(row))
        if source is not None:
            updated["cost_source_mix"] = str(source.get("cost_source_mix") or "")
            updated["missing_cost_source_count"] = int(
                _optional_float(source.get("missing_cost_source_count")) or 0
            )
        rows.append(updated)
    return pl.DataFrame(rows, schema=PAPER_SLIPPAGE_SCHEMA, orient="row")


def _paper_portfolio_overrides(
    research_portfolio: pl.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    if research_portfolio.is_empty():
        return {}
    return portfolio_status_overrides_by_identifier(research_portfolio)


def _paper_entry_policy(
    *,
    row: dict[str, Any],
    payload: dict[str, Any],
    proposal_id: str,
    candidate: str,
    symbol: str,
    raw_would_enter: bool,
    portfolio_overrides: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    if _is_sol_f4_tracking_row(proposal_id, candidate, symbol):
        if not _sol_f4_symbol_match_verified(row, payload, symbol):
            return {
                "would_enter": False,
                "no_sample_reason": "symbol_mismatch",
                "paper_disabled_by_research_portfolio": False,
                "tracking_stage": "paper_review_entry_conditions_not_met",
                "clear_paper_pnl": True,
                "paper_trigger_type": "",
                "paper_trigger_reason": "source_candidate_symbol_mismatch",
            }
        sol_no_sample_reason = _sol_f4_factor_no_sample_reason(row, payload, symbol)
        if sol_no_sample_reason == "alpha6_not_buy":
            return {
                "would_enter": False,
                "no_sample_reason": sol_no_sample_reason,
                "paper_disabled_by_research_portfolio": False,
                "tracking_stage": "paper_review_entry_conditions_not_met",
                "clear_paper_pnl": True,
                "paper_trigger_type": "",
                "paper_trigger_reason": sol_no_sample_reason,
            }
    if _eth_f3_disabled_by_research_portfolio(
        row=row,
        payload=payload,
        proposal_id=proposal_id,
        candidate=candidate,
        symbol=symbol,
        portfolio_overrides=portfolio_overrides,
    ):
        return {
            "would_enter": False,
            "no_sample_reason": ETH_F3_DISABLED_NO_SAMPLE_REASON,
            "paper_disabled_by_research_portfolio": True,
            "tracking_stage": "paper_review_disabled_by_research_portfolio",
            "clear_paper_pnl": True,
            "paper_trigger_type": "",
            "paper_trigger_reason": ETH_F3_DISABLED_NO_SAMPLE_REASON,
        }
    sol_factor_reason = _sol_f4_factor_trigger_reason(row, payload, proposal_id, candidate, symbol)
    if sol_factor_reason:
        return {
            "would_enter": True,
            "no_sample_reason": "",
            "paper_disabled_by_research_portfolio": False,
            "tracking_stage": "active_paper_strategy",
            "clear_paper_pnl": False,
            "paper_trigger_type": "factor_condition_match",
            "paper_trigger_reason": sol_factor_reason,
        }
    bnb_factor_reason = _bnb_paper_trigger_reason(row, payload, proposal_id)
    if bnb_factor_reason:
        return {
            "would_enter": True,
            "no_sample_reason": "",
            "paper_disabled_by_research_portfolio": False,
            "tracking_stage": "active_paper_strategy",
            "clear_paper_pnl": False,
            "paper_trigger_type": "bnb_factor_condition_match",
            "paper_trigger_reason": bnb_factor_reason,
        }
    if raw_would_enter and _sol_f4_source_candidate_matches(
        proposal_id,
        candidate,
        symbol,
        payload,
    ):
        return {
            "would_enter": True,
            "no_sample_reason": "",
            "paper_disabled_by_research_portfolio": False,
            "tracking_stage": "",
            "clear_paper_pnl": False,
            "paper_trigger_type": "source_candidate_match",
            "paper_trigger_reason": "source_strategy_candidate_matches_sol_f4",
        }
    if (
        raw_would_enter
        and _is_eth_f3_tracking_row(proposal_id, candidate, symbol)
        and _eth_f3_has_restore_gate_fields(row, payload)
    ):
        restore_reason = _eth_f3_restore_block_reason(row, payload)
        if restore_reason:
            return {
                "would_enter": False,
                "no_sample_reason": restore_reason,
                "paper_disabled_by_research_portfolio": False,
                "tracking_stage": "paper_review_entry_conditions_not_met",
                "clear_paper_pnl": True,
                "paper_trigger_type": "",
                "paper_trigger_reason": restore_reason,
            }
    return {
        "would_enter": raw_would_enter,
        "no_sample_reason": "",
        "paper_disabled_by_research_portfolio": False,
        "tracking_stage": "",
        "clear_paper_pnl": False,
        "paper_trigger_type": "source_candidate_match" if raw_would_enter else "",
        "paper_trigger_reason": "explicit_v5_would_enter" if raw_would_enter else "",
    }


def _sol_f4_factor_candidate_run_rows(
    candidate_events: pl.DataFrame,
    created_at: str,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if candidate_events.is_empty():
        return []
    existing_keys = _sol_f4_existing_candidate_keys(existing_rows)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_events.to_dicts():
        candidate_payload = _payload(candidate)
        if not _sol_f4_candidate_symbol_matches(candidate, candidate_payload):
            continue
        synthetic = _sol_f4_candidate_event_as_paper_row(candidate)
        payload = _payload(synthetic)
        trigger_reason = _sol_f4_factor_trigger_reason(
            synthetic,
            payload,
            SOL_F4_PAPER_PROPOSAL_ID,
            SOL_F4_STRATEGY_CANDIDATE,
            SOL_F4_SYMBOL,
        )
        key = _sol_f4_candidate_key(synthetic, payload)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        if trigger_reason:
            synthetic["paper_trigger_reason"] = trigger_reason
            synthetic["would_enter"] = "true"
        else:
            synthetic["paper_trigger_reason"] = ""
            synthetic["would_enter"] = "false"
            synthetic["no_sample_reason"] = _sol_f4_factor_no_sample_reason(
                synthetic,
                payload,
                SOL_F4_SYMBOL,
            )
        rows.append(_v5_run_row(synthetic, created_at))
    return rows


def _sol_f4_factor_candidate_report_rows(
    candidate_events: pl.DataFrame,
    created_at: str,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if candidate_events.is_empty():
        return []
    existing_keys = _sol_f4_existing_candidate_keys(existing_rows)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_events.to_dicts():
        candidate_payload = _payload(candidate)
        if not _sol_f4_candidate_symbol_matches(candidate, candidate_payload):
            continue
        synthetic = _sol_f4_candidate_event_as_paper_row(candidate)
        payload = _payload(synthetic)
        trigger_reason = _sol_f4_factor_trigger_reason(
            synthetic,
            payload,
            SOL_F4_PAPER_PROPOSAL_ID,
            SOL_F4_STRATEGY_CANDIDATE,
            SOL_F4_SYMBOL,
        )
        key = _sol_f4_candidate_key(synthetic, payload)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        if trigger_reason:
            synthetic["paper_trigger_reason"] = trigger_reason
            synthetic["would_enter"] = "true"
        else:
            synthetic["paper_trigger_reason"] = ""
            synthetic["would_enter"] = "false"
            synthetic["no_sample_reason"] = _sol_f4_factor_no_sample_reason(
                synthetic,
                payload,
                SOL_F4_SYMBOL,
            )
        rows.append(_v5_run_report_row(synthetic, created_at))
    return rows


def _sol_f4_existing_candidate_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        proposal_id = str(row.get("proposal_id") or row.get("strategy_id") or "")
        symbol = normalize_symbol(row.get("symbol"))
        if proposal_id != SOL_F4_PAPER_PROPOSAL_ID or symbol != SOL_F4_SYMBOL:
            continue
        payload = _payload(row)
        keys.add(_sol_f4_candidate_key(row, payload))
    return keys


def _sol_f4_candidate_key(
    row: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[str, str, str, str]:
    return (
        _field(row, payload, "candidate_id"),
        _field(row, payload, "run_id"),
        _field(row, payload, "ts_utc", "ts", "timestamp", "created_at"),
        normalize_symbol(_field(row, payload, "symbol", "normalized_symbol")) or SOL_F4_SYMBOL,
    )


def _sol_f4_candidate_event_as_paper_row(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = _payload(candidate)
    source_candidate = _field(
        candidate,
        payload,
        "source_strategy_candidate",
        "strategy_candidate",
        "candidate",
    )
    source_symbol = normalize_symbol(
        _field(candidate, payload, "source_candidate_symbol", "symbol", "normalized_symbol")
    )
    source_candidate_id = _field(candidate, payload, "source_candidate_id", "candidate_id")
    ts_utc = _field(candidate, payload, "ts_utc", "ts", "timestamp", "created_at")
    merged_payload = {
        **payload,
        **{
            key: value
            for key, value in candidate.items()
            if key not in {"raw_payload_json"} and value is not None
        },
        "source_strategy_candidate": source_candidate,
        "source_candidate_symbol": source_symbol,
        "source_candidate_id": source_candidate_id,
        "symbol_match_verified": source_symbol == SOL_F4_SYMBOL,
        "paper_trigger_type": "factor_condition_match",
    }
    return {
        **candidate,
        "as_of_date": _as_of_date(candidate, payload),
        "proposal_id": SOL_F4_PAPER_PROPOSAL_ID,
        "strategy_id": SOL_F4_PAPER_PROPOSAL_ID,
        "strategy_candidate": SOL_F4_STRATEGY_CANDIDATE,
        "source_strategy_candidate": source_candidate,
        "source_candidate_symbol": source_symbol,
        "source_candidate_id": source_candidate_id,
        "symbol_match_verified": source_symbol == SOL_F4_SYMBOL,
        "symbol": SOL_F4_SYMBOL,
        "normalized_symbol": SOL_F4_SYMBOL,
        "recommended_mode": "paper",
        "board_decision": "PAPER_READY",
        "suggested_horizon": "24h",
        "horizon_hours": "24",
        "would_enter": "false",
        "would_exit": "false",
        "would_size": "100",
        "would_size_usdt": "100",
        "final_decision": _field(candidate, payload, "final_decision", "decision"),
        "ts_utc": ts_utc,
        "paper_tracking_status": V5_PAPER_TRACKING_STATUS,
        "tracking_stage": "active_paper_strategy",
        "paper_trigger_type": "factor_condition_match",
        "paper_source": "quant_lab_synthetic",
        "paper_count_scope": "daily",
        "raw_payload_json": safe_json_dumps(merged_payload),
    }


def _sol_f4_factor_trigger_reason(
    row: dict[str, Any],
    payload: dict[str, Any],
    proposal_id: str,
    candidate: str,
    symbol: str,
) -> str:
    if not _is_sol_f4_tracking_row(proposal_id, candidate, symbol):
        return ""
    if not _sol_f4_symbol_match_verified(row, payload, symbol):
        return ""
    row_symbol = normalize_symbol(_field(row, payload, "symbol", "normalized_symbol") or symbol)
    if row_symbol != SOL_F4_SYMBOL:
        return ""
    alpha6_side = str(_field(row, payload, "alpha6_side") or "").strip().lower()
    if alpha6_side != "buy":
        return ""
    f4_value = _factor_float(_field(row, payload, "f4_volume_expansion"))
    if f4_value is None or f4_value < 0:
        return ""
    expected = _optional_float(_field(row, payload, "expected_edge_bps", "edge_bps"))
    required = _optional_float(_field(row, payload, "required_edge_bps", "required_edge"))
    if expected is None or required is None or expected <= required:
        return ""
    if _optional_bool(_field(row, payload, "cost_gate_verified", "cost.gate_verified")) is not True:
        return ""
    if _risk_off_for_sol_f4(row, payload):
        return ""
    return (
        "sol_f4_factor_condition_match:"
        "alpha6_buy,f4_non_negative,expected_edge_gt_required_edge,"
        "cost_gate_verified,not_risk_off"
    )


def _sol_f4_factor_no_sample_reason(
    row: dict[str, Any],
    payload: dict[str, Any],
    symbol: str,
) -> str:
    if not _sol_f4_symbol_match_verified(row, payload, symbol):
        return "symbol_mismatch"
    alpha6_side = str(_field(row, payload, "alpha6_side") or "").strip().lower()
    if alpha6_side and alpha6_side not in {"buy", "long"}:
        return "alpha6_not_buy"
    f4_value = _factor_float(_field(row, payload, "f4_volume_expansion"))
    if f4_value is None or f4_value < 0:
        return "f4_not_expanding"
    expected = _optional_float(_field(row, payload, "expected_edge_bps", "edge_bps"))
    required = _optional_float(_field(row, payload, "required_edge_bps", "required_edge"))
    if expected is None or required is None or expected <= required:
        return "edge_not_above_required"
    if _optional_bool(_field(row, payload, "cost_gate_verified", "cost.gate_verified")) is not True:
        return "cost_gate_not_verified"
    if _risk_off_for_sol_f4(row, payload):
        return "risk_off"
    return ""


def _sol_f4_source_candidate_symbol(
    row: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    return normalize_symbol(
        _field(row, payload, "source_candidate_symbol", "candidate_symbol")
        or _field(row, payload, "symbol", "normalized_symbol")
    )


def _sol_f4_symbol_match_verified(
    row: dict[str, Any],
    payload: dict[str, Any],
    symbol: str,
) -> bool:
    paper_symbol = normalize_symbol(symbol) or normalize_symbol(
        _field(row, payload, "symbol", "normalized_symbol")
    )
    source_symbol = _sol_f4_source_candidate_symbol(row, payload)
    return paper_symbol == SOL_F4_SYMBOL and source_symbol == SOL_F4_SYMBOL


def _sol_f4_candidate_symbol_matches(candidate: dict[str, Any], payload: dict[str, Any]) -> bool:
    return _sol_f4_source_candidate_symbol(candidate, payload) == SOL_F4_SYMBOL


def _is_sol_f4_tracking_row(proposal_id: str, candidate: str, symbol: str) -> bool:
    normalized_symbol = normalize_symbol(symbol)
    normalized_candidate = str(candidate or "").strip()
    return normalized_symbol == SOL_F4_SYMBOL and (
        proposal_id == SOL_F4_PAPER_PROPOSAL_ID
        or normalized_candidate in {SOL_F4_STRATEGY_CANDIDATE, "f4_volume_expansion_entry"}
    )


def _sol_f4_source_candidate_matches(
    proposal_id: str,
    candidate: str,
    symbol: str,
    payload: dict[str, Any],
) -> bool:
    if not _is_sol_f4_tracking_row(proposal_id, candidate, symbol):
        return False
    source_candidate = str(
        payload.get("source_strategy_candidate")
        or payload.get("strategy_candidate")
        or candidate
        or ""
    ).strip()
    return source_candidate in {SOL_F4_STRATEGY_CANDIDATE, "f4_volume_expansion_entry"}


def _risk_off_for_sol_f4(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    explicit = _optional_bool(_field(row, payload, "risk_off", "is_risk_off"))
    if explicit is not None:
        return explicit
    for key in ["current_regime", "regime_state", "market_regime", "risk_level"]:
        value = str(_field(row, payload, key) or "").strip().upper()
        if value == "RISK_OFF":
            return True
    return False


def _bnb_paper_candidate_run_rows(
    candidate_events: pl.DataFrame,
    created_at: str,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if candidate_events.is_empty():
        return []
    existing_keys = _bnb_existing_candidate_keys(existing_rows)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_events.to_dicts():
        payload = _payload(candidate)
        symbol = normalize_symbol(_field(candidate, payload, "symbol", "normalized_symbol"))
        if symbol != BNB_SYMBOL:
            continue
        for proposal_id, strategy_candidate in (
            (BNB_F3_PAPER_PROPOSAL_ID, BNB_F3_STRATEGY_CANDIDATE),
            (BNB_RISK_ON_BUY_PAPER_PROPOSAL_ID, BNB_RISK_ON_BUY_STRATEGY_CANDIDATE),
        ):
            synthetic = _bnb_candidate_event_as_paper_row(
                candidate,
                proposal_id=proposal_id,
                strategy_candidate=strategy_candidate,
            )
            synthetic_payload = _payload(synthetic)
            key = _bnb_candidate_key(synthetic, synthetic_payload)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            trigger_reason = _bnb_paper_trigger_reason(
                synthetic,
                synthetic_payload,
                proposal_id,
            )
            if trigger_reason:
                synthetic["paper_trigger_reason"] = trigger_reason
                synthetic["would_enter"] = "true"
            else:
                synthetic["paper_trigger_reason"] = ""
                synthetic["would_enter"] = "false"
                synthetic["no_sample_reason"] = _bnb_paper_no_sample_reason(
                    synthetic,
                    synthetic_payload,
                    proposal_id,
                )
            rows.append(_v5_run_row(synthetic, created_at))
    return rows


def _bnb_paper_candidate_report_rows(
    candidate_events: pl.DataFrame,
    created_at: str,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if candidate_events.is_empty():
        return []
    existing_keys = _bnb_existing_candidate_keys(existing_rows)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_events.to_dicts():
        payload = _payload(candidate)
        symbol = normalize_symbol(_field(candidate, payload, "symbol", "normalized_symbol"))
        if symbol != BNB_SYMBOL:
            continue
        for proposal_id, strategy_candidate in (
            (BNB_F3_PAPER_PROPOSAL_ID, BNB_F3_STRATEGY_CANDIDATE),
            (BNB_RISK_ON_BUY_PAPER_PROPOSAL_ID, BNB_RISK_ON_BUY_STRATEGY_CANDIDATE),
        ):
            synthetic = _bnb_candidate_event_as_paper_row(
                candidate,
                proposal_id=proposal_id,
                strategy_candidate=strategy_candidate,
            )
            synthetic_payload = _payload(synthetic)
            key = _bnb_candidate_key(synthetic, synthetic_payload)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            trigger_reason = _bnb_paper_trigger_reason(
                synthetic,
                synthetic_payload,
                proposal_id,
            )
            if trigger_reason:
                synthetic["paper_trigger_reason"] = trigger_reason
                synthetic["would_enter"] = "true"
            else:
                synthetic["paper_trigger_reason"] = ""
                synthetic["would_enter"] = "false"
                synthetic["no_sample_reason"] = _bnb_paper_no_sample_reason(
                    synthetic,
                    synthetic_payload,
                    proposal_id,
                )
            rows.append(_v5_run_report_row(synthetic, created_at))
    return rows


def _bnb_existing_candidate_keys(
    rows: list[dict[str, Any]],
) -> set[tuple[str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        proposal_id = str(row.get("proposal_id") or row.get("strategy_id") or "")
        if not (
            _is_bnb_f3_tracking_row(proposal_id) or proposal_id == BNB_RISK_ON_BUY_PAPER_PROPOSAL_ID
        ):
            continue
        if normalize_symbol(row.get("symbol")) != BNB_SYMBOL:
            continue
        keys.add(_bnb_candidate_key(row, _payload(row)))
    return keys


def _bnb_candidate_key(
    row: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    return (
        str(_field(row, payload, "proposal_id", "strategy_id") or ""),
        _field(row, payload, "candidate_id", "source_candidate_id"),
        _field(row, payload, "run_id"),
        _field(row, payload, "ts_utc", "ts", "timestamp", "created_at"),
        normalize_symbol(_field(row, payload, "symbol", "normalized_symbol")) or BNB_SYMBOL,
    )


def _bnb_candidate_event_as_paper_row(
    candidate: dict[str, Any],
    *,
    proposal_id: str,
    strategy_candidate: str,
) -> dict[str, Any]:
    payload = _payload(candidate)
    source_candidate = _field(
        candidate,
        payload,
        "source_strategy_candidate",
        "strategy_candidate",
        "candidate",
    )
    source_candidate_id = _field(candidate, payload, "source_candidate_id", "candidate_id")
    ts_utc = _field(candidate, payload, "ts_utc", "ts", "timestamp", "created_at")
    merged_payload = {
        **payload,
        **{
            key: value
            for key, value in candidate.items()
            if key not in {"raw_payload_json"} and value is not None
        },
        "source_strategy_candidate": source_candidate,
        "source_candidate_symbol": BNB_SYMBOL,
        "source_candidate_id": source_candidate_id,
        "symbol_match_verified": True,
        "paper_trigger_type": "bnb_factor_condition_match",
    }
    return {
        **candidate,
        "as_of_date": _as_of_date(candidate, payload),
        "proposal_id": proposal_id,
        "strategy_id": proposal_id,
        "paper_tracker_id": _paper_tracker_id_for(proposal_id, strategy_candidate, BNB_SYMBOL),
        "strategy_candidate": strategy_candidate,
        "source_strategy_candidate": source_candidate,
        "source_candidate_symbol": BNB_SYMBOL,
        "source_candidate_id": source_candidate_id,
        "symbol_match_verified": True,
        "symbol": BNB_SYMBOL,
        "normalized_symbol": BNB_SYMBOL,
        "recommended_mode": "paper",
        "board_decision": "PAPER_READY",
        "suggested_horizon": "24h",
        "horizon_hours": "24",
        "would_enter": "false",
        "would_exit": "false",
        "would_size": "100",
        "would_size_usdt": "100",
        "final_decision": _field(candidate, payload, "final_decision", "decision"),
        "ts_utc": ts_utc,
        "paper_tracking_status": V5_PAPER_TRACKING_STATUS,
        "tracking_stage": "active_paper_strategy",
        "paper_trigger_type": "bnb_factor_condition_match",
        "paper_source": "quant_lab_synthetic",
        "paper_count_scope": "daily",
        "live_block_reason": safe_json_dumps(
            ["bnb_paper_only_no_live", "bnb_negative_expectancy_recovery_research_only"]
        ),
        "raw_payload_json": safe_json_dumps(merged_payload),
    }


def _bnb_paper_trigger_reason(
    row: dict[str, Any],
    payload: dict[str, Any],
    proposal_id: str,
) -> str:
    if normalize_symbol(_field(row, payload, "symbol", "normalized_symbol")) != BNB_SYMBOL:
        return ""
    if _is_bnb_f3_tracking_row(proposal_id):
        source_candidate = str(
            _field(row, payload, "source_strategy_candidate", "strategy_candidate") or ""
        ).strip()
        if source_candidate not in BNB_F3_SOURCE_CANDIDATES:
            return ""
    alpha6_side = str(_field(row, payload, "alpha6_side") or "").strip().lower()
    if alpha6_side != "buy":
        return ""
    alpha6_score = _optional_float(_field(row, payload, "alpha6_score"))
    if alpha6_score is None or alpha6_score < 0.9:
        return ""
    expected = _optional_float(_field(row, payload, "expected_edge_bps", "edge_bps"))
    required = _optional_float(_field(row, payload, "required_edge_bps", "required_edge"))
    if expected is None or required is None or expected <= required:
        return ""
    if _optional_bool(_field(row, payload, "cost_gate_verified", "cost.gate_verified")) is not True:
        return ""
    if _bnb_regime(row, payload) not in BNB_ALLOWED_PAPER_REGIMES:
        return ""
    return (
        "bnb_factor_condition_match:"
        "alpha6_buy,alpha6_score_gte_0_9,expected_edge_gt_required_edge,"
        "cost_gate_verified,risk_on_or_trending_regime"
    )


def _bnb_paper_no_sample_reason(
    row: dict[str, Any],
    payload: dict[str, Any],
    proposal_id: str,
) -> str:
    if normalize_symbol(_field(row, payload, "symbol", "normalized_symbol")) != BNB_SYMBOL:
        return "symbol_mismatch"
    if _is_bnb_f3_tracking_row(proposal_id):
        source_candidate = str(
            _field(row, payload, "source_strategy_candidate", "strategy_candidate") or ""
        ).strip()
        if source_candidate not in BNB_F3_SOURCE_CANDIDATES:
            return "bnb_f3_strategy_candidate_not_matched"
    alpha6_side = str(_field(row, payload, "alpha6_side") or "").strip().lower()
    if alpha6_side != "buy":
        return "alpha6_not_buy"
    alpha6_score = _optional_float(_field(row, payload, "alpha6_score"))
    if alpha6_score is None or alpha6_score < 0.9:
        return "alpha6_score_below_threshold"
    expected = _optional_float(_field(row, payload, "expected_edge_bps", "edge_bps"))
    required = _optional_float(_field(row, payload, "required_edge_bps", "required_edge"))
    if expected is None or required is None or expected <= required:
        return "edge_not_above_required"
    if _optional_bool(_field(row, payload, "cost_gate_verified", "cost.gate_verified")) is not True:
        return "cost_gate_not_verified"
    if _bnb_regime(row, payload) not in BNB_ALLOWED_PAPER_REGIMES:
        return "regime_not_allowed"
    return ""


def _is_bnb_f3_tracking_row(proposal_id: str) -> bool:
    return proposal_id in {BNB_F3_PAPER_PROPOSAL_ID, BNB_F3_CURRENT_PROPOSAL_ID}


def _bnb_regime(row: dict[str, Any], payload: dict[str, Any]) -> str:
    return (
        str(_field(row, payload, "current_regime", "regime_state", "market_regime") or "")
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _factor_float(value: Any) -> float | None:
    parsed = _optional_float(value)
    if parsed is not None:
        return parsed
    boolean = _optional_bool(value)
    if boolean is not None:
        return 1.0 if boolean else 0.0
    return None


def _eth_f3_disabled_by_research_portfolio(
    *,
    row: dict[str, Any],
    payload: dict[str, Any],
    proposal_id: str,
    candidate: str,
    symbol: str,
    portfolio_overrides: dict[tuple[str, str], dict[str, Any]],
) -> bool:
    if not _is_eth_f3_tracking_row(proposal_id, candidate, symbol):
        return False
    override = _eth_f3_portfolio_override(portfolio_overrides, proposal_id, candidate, symbol)
    if override is None:
        return False
    status = str(override.get("status") or "").strip().upper()
    if status != "DOWNGRADED_FROM_PAPER":
        return False
    return _paper_row_is_on_or_after_override(row, payload, override)


def _eth_f3_portfolio_override(
    portfolio_overrides: dict[tuple[str, str], dict[str, Any]],
    proposal_id: str,
    candidate: str,
    symbol: str,
) -> dict[str, Any] | None:
    normalized_symbol = normalize_symbol(symbol) or ETH_F3_SYMBOL
    identifiers = {
        proposal_id,
        candidate,
        ETH_F3_PAPER_PROPOSAL_ID,
        "ETH_F3_DOMINANT_ENTRY_PAPER_V1",
        ETH_F3_STRATEGY_CANDIDATE,
        "v5.eth_f3_dominant_entry",
    }
    symbols = {normalized_symbol, ETH_F3_SYMBOL}
    for identifier in identifiers:
        if not identifier:
            continue
        for item_symbol in symbols:
            override = portfolio_overrides.get((identifier, item_symbol))
            if override is not None:
                return override
    return None


def _paper_row_is_on_or_after_override(
    row: dict[str, Any],
    payload: dict[str, Any],
    override: dict[str, Any],
) -> bool:
    row_day = _safe_date(_as_of_date(row, payload))
    override_day = _safe_date(override.get("as_of_date") or override.get("last_review_date"))
    if row_day is None or override_day is None:
        return True
    return row_day >= override_day


def _is_eth_f3_tracking_row(proposal_id: str, candidate: str, symbol: str) -> bool:
    normalized_symbol = normalize_symbol(symbol)
    return normalized_symbol == ETH_F3_SYMBOL and (
        proposal_id in {ETH_F3_PAPER_PROPOSAL_ID, "ETH_F3_DOMINANT_ENTRY_PAPER_V1"}
        or candidate in {ETH_F3_STRATEGY_CANDIDATE, "v5.eth_f3_dominant_entry"}
    )


def _eth_f3_has_restore_gate_fields(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    return any(
        _field(row, payload, field) != ""
        for field in [
            "alpha6_side",
            "f3_vol_adj_ret",
            "final_score",
            "current_regime",
            "regime_state",
            "market_regime",
            "final_decision",
            "decision",
        ]
    )


def _eth_f3_restore_block_reason(row: dict[str, Any], payload: dict[str, Any]) -> str | None:
    alpha6_side = str(_field(row, payload, "alpha6_side") or "").strip().lower()
    if alpha6_side != "buy":
        return "eth_f3_alpha6_side_not_buy_no_new_entry"
    f3_value = _optional_float(_field(row, payload, "f3_vol_adj_ret", "f3"))
    if f3_value is None or f3_value <= 0:
        return "eth_f3_f3_vol_adj_ret_not_positive_no_new_entry"
    if _optional_float(_field(row, payload, "final_score")) is None:
        return "eth_f3_final_score_missing_no_new_entry"
    regime = (
        str(_field(row, payload, "current_regime", "regime_state", "market_regime") or "")
        .strip()
        .upper()
    )
    if regime == "RISK_OFF":
        return "eth_f3_risk_off_no_new_entry"
    final_decision = str(_field(row, payload, "final_decision", "decision") or "").lower()
    if final_decision in {"no_order", "none", "skip", "skipped"}:
        return "eth_f3_final_decision_no_order_no_new_entry"
    return None


def _safe_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.astimezone(UTC).date() if value.tzinfo else value.date()
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _paper_source_for_run(
    row: dict[str, Any],
    payload: dict[str, Any],
    enter_policy: dict[str, Any],
    raw_would_enter: bool,
) -> str:
    explicit = str(_field(row, payload, "paper_source") or "").strip()
    if explicit:
        return explicit
    trigger_type = str(enter_policy.get("paper_trigger_type") or "").strip()
    if trigger_type == "factor_condition_match" and not raw_would_enter:
        return "quant_lab_synthetic"
    return "v5_telemetry"


def _paper_source(row: dict[str, Any]) -> str:
    source = str(row.get("paper_source") or "").strip()
    if source == "generic_contract_runtime":
        return "v5_telemetry"
    return source or "v5_telemetry"


def _source_candidate_metadata(
    row: dict[str, Any],
    payload: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    raw_source_symbol = _field(row, payload, "source_candidate_symbol", "candidate_symbol")
    source_symbol = normalize_symbol(raw_source_symbol) if raw_source_symbol else ""
    normalized_symbol = normalize_symbol(symbol)
    explicit_verified = _optional_bool(_field(row, payload, "symbol_match_verified"))
    if explicit_verified is None:
        symbol_match_verified = (
            not source_symbol or not normalized_symbol or source_symbol == normalized_symbol
        )
    else:
        symbol_match_verified = explicit_verified
    return {
        "source_candidate_symbol": source_symbol,
        "source_candidate_id": _field(row, payload, "source_candidate_id", "candidate_id"),
        "symbol_match_verified": symbol_match_verified,
    }


def _v5_run_report_row(
    row: dict[str, Any],
    created_at: str,
    *,
    portfolio_overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = _payload(row)
    candidate = _field(
        row,
        payload,
        "strategy_candidate",
        "experiment_name",
        "candidate",
        "source_strategy_candidate",
    )
    symbol = _field(row, payload, "symbol", "normalized_symbol")
    strategy_id = _field(row, payload, "strategy_id")
    proposal_id = _field(row, payload, "proposal_id") or _proposal_id_for(candidate, symbol)
    paper_tracker_id = (
        _field(row, payload, "paper_tracker_id")
        or strategy_id
        or _paper_tracker_id_for(proposal_id, candidate, symbol)
    )
    raw_would_enter = _optional_bool(_field(row, payload, "would_enter", "enter")) is True
    enter_policy = _paper_entry_policy(
        row=row,
        payload=payload,
        proposal_id=proposal_id,
        candidate=candidate,
        symbol=symbol,
        raw_would_enter=raw_would_enter,
        portfolio_overrides=portfolio_overrides or {},
    )
    would_exit = _optional_bool(_field(row, payload, "would_exit", "exit")) is True
    paper_pnl_bps = _optional_float(_field(row, payload, "paper_pnl_bps", "pnl_bps"))
    paper_pnl_usdt = _optional_float(
        _field(row, payload, "paper_pnl_usdt", "paper_pnl", "pnl_usdt")
    )
    if enter_policy["clear_paper_pnl"]:
        paper_pnl_bps = None
        paper_pnl_usdt = None
    paper_source = _paper_source_for_run(row, payload, enter_policy, raw_would_enter)
    source_candidate = _source_candidate_metadata(row, payload, str(symbol or ""))
    return {
        "as_of_date": _as_of_date(row, payload),
        "strategy_id": strategy_id or paper_tracker_id or proposal_id,
        "proposal_id": proposal_id,
        "paper_tracker_id": paper_tracker_id,
        "strategy_candidate": candidate,
        "run_id": _field(row, payload, "run_id"),
        "ts_utc": _field(row, payload, "ts_utc", "ts", "timestamp", "created_at"),
        "symbol": symbol,
        "would_enter": enter_policy["would_enter"],
        "would_exit": would_exit,
        "final_decision": _field(row, payload, "final_decision", "decision"),
        "no_sample_reason": enter_policy["no_sample_reason"]
        or _field(row, payload, "no_sample_reason", "reason"),
        "paper_disabled_by_research_portfolio": enter_policy[
            "paper_disabled_by_research_portfolio"
        ],
        "paper_trigger_type": enter_policy["paper_trigger_type"]
        or _field(row, payload, "paper_trigger_type"),
        "paper_trigger_reason": enter_policy["paper_trigger_reason"]
        or _field(row, payload, "paper_trigger_reason"),
        **source_candidate,
        "paper_source": paper_source,
        "paper_count_scope": _field(row, payload, "paper_count_scope", default="daily"),
        "risk_level": _field(row, payload, "risk_level"),
        "alpha6_score": _optional_float(_field(row, payload, "alpha6_score")),
        "alpha6_side": _field(row, payload, "alpha6_side"),
        "f4_volume_expansion": _field(row, payload, "f4_volume_expansion"),
        "f5_rsi_trend_confirm": _field(row, payload, "f5_rsi_trend_confirm"),
        "arrival_bid": _optional_float(_field(row, payload, "arrival_bid", "bid_at_arrival")),
        "arrival_ask": _optional_float(_field(row, payload, "arrival_ask", "ask_at_arrival")),
        "arrival_mid": _optional_float(_field(row, payload, "arrival_mid", "mid_at_arrival")),
        "estimated_spread_bps": _optional_float(
            _field(
                row,
                payload,
                "estimated_spread_bps",
                "arrival_spread_bps",
                "spread_bps_at_decision",
                "spread_bps",
            )
        ),
        "expected_order_type": _field(
            row,
            payload,
            "expected_order_type",
            "order_type",
            "paper_order_type",
        ),
        "estimated_fill_px": _optional_float(
            _field(row, payload, "estimated_fill_px", "estimated_fill_price", "paper_fill_px")
        ),
        "cost_source": _cost_source_with_live_safety_markers(
            _field(row, payload, "cost_source"),
            row=row,
            payload=payload,
        ),
        "cost_source_mix": _cost_source_with_live_safety_markers(
            _field(row, payload, "cost_source_mix"),
            row=row,
            payload=payload,
        ),
        "paper_pnl_bps": paper_pnl_bps,
        "paper_pnl_usdt": paper_pnl_usdt,
        "label_status": _field(row, payload, "label_status"),
        "paper_tracking_status": V5_PAPER_TRACKING_STATUS,
        "tracking_stage": enter_policy["tracking_stage"]
        or ("completed_paper_observations" if would_exit else "active_paper_strategy"),
        "source_path_inside_bundle": _field(row, payload, "source_path_inside_bundle"),
        "bundle_ts": _field(row, payload, "bundle_ts"),
        "ingest_ts": _field(row, payload, "ingest_ts"),
        "created_at": created_at,
        "source": V5_PAPER_TRACKING_SOURCE,
        "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
    }


def _v5_run_row(
    row: dict[str, Any],
    created_at: str,
    *,
    portfolio_overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = _payload(row)
    candidate = _field(
        row,
        payload,
        "strategy_candidate",
        "experiment_name",
        "candidate",
        "source_strategy_candidate",
    )
    symbol = _field(row, payload, "symbol", "normalized_symbol")
    proposal_id = _field(row, payload, "proposal_id", "strategy_id") or _proposal_id_for(
        candidate,
        symbol,
    )
    paper_tracker_id = _field(
        row, payload, "paper_tracker_id", "strategy_id"
    ) or _paper_tracker_id_for(proposal_id, candidate, symbol)
    would_exit = _optional_bool(_field(row, payload, "would_exit", "exit")) is True
    raw_would_enter = _optional_bool(_field(row, payload, "would_enter", "enter")) is True
    enter_policy = _paper_entry_policy(
        row=row,
        payload=payload,
        proposal_id=proposal_id,
        candidate=candidate,
        symbol=symbol,
        raw_would_enter=raw_would_enter,
        portfolio_overrides=portfolio_overrides or {},
    )
    effective_would_enter = bool(enter_policy["would_enter"])
    paper_pnl_bps = _optional_float(_field(row, payload, "paper_pnl_bps", "pnl_bps"))
    paper_pnl_bps_4h = _optional_float(_field(row, payload, "paper_pnl_bps_4h"))
    paper_pnl_bps_8h = _optional_float(_field(row, payload, "paper_pnl_bps_8h"))
    paper_pnl_bps_12h = _optional_float(_field(row, payload, "paper_pnl_bps_12h"))
    paper_pnl_bps_24h = _optional_float(_field(row, payload, "paper_pnl_bps_24h"))
    paper_pnl_bps_48h = _optional_float(_field(row, payload, "paper_pnl_bps_48h"))
    paper_pnl_bps_72h = _optional_float(_field(row, payload, "paper_pnl_bps_72h"))
    paper_pnl_usdt = _optional_float(
        _field(row, payload, "paper_pnl_usdt", "paper_pnl", "pnl_usdt")
    )
    if enter_policy["clear_paper_pnl"]:
        paper_pnl_bps = None
        paper_pnl_bps_4h = None
        paper_pnl_bps_8h = None
        paper_pnl_bps_12h = None
        paper_pnl_bps_24h = None
        paper_pnl_bps_48h = None
        paper_pnl_bps_72h = None
        paper_pnl_usdt = None
    paper_source = _paper_source_for_run(row, payload, enter_policy, raw_would_enter)
    source_candidate = _source_candidate_metadata(row, payload, str(symbol or ""))
    return {
        "as_of_date": _as_of_date(row, payload),
        "paper_trade_id": _field(row, payload, "paper_trade_id"),
        "strategy_id": _field(row, payload, "strategy_id"),
        "strategy_version": _field(row, payload, "strategy_version"),
        "proposal_id": proposal_id,
        "paper_tracker_id": paper_tracker_id,
        "strategy_candidate": candidate,
        "symbol": symbol,
        "recommended_mode": _field(row, payload, "recommended_mode", default="paper"),
        "board_decision": _field(row, payload, "board_decision", default="PAPER_READY"),
        "suggested_horizon": _field(row, payload, "suggested_horizon", "horizon"),
        "horizon_hours": int(_optional_float(_field(row, payload, "horizon_hours")) or 0),
        "no_sample_reason": enter_policy["no_sample_reason"]
        or _field(row, payload, "no_sample_reason", "reason"),
        "would_enter": effective_would_enter,
        "would_exit": would_exit,
        "would_size": 0.0
        if not effective_would_enter
        else _optional_float(_field(row, payload, "would_size", "would_size_notional", "size"))
        or 0.0,
        "would_size_usdt": 0.0
        if not effective_would_enter
        else _optional_float(
            _field(
                row,
                payload,
                "would_size_usdt",
                "would_size_notional",
                "paper_size_usdt",
                "would_size",
            )
        )
        or 0.0,
        "paper_disabled_by_research_portfolio": enter_policy[
            "paper_disabled_by_research_portfolio"
        ],
        "paper_trigger_type": enter_policy["paper_trigger_type"]
        or _field(row, payload, "paper_trigger_type"),
        "paper_trigger_reason": enter_policy["paper_trigger_reason"]
        or _field(row, payload, "paper_trigger_reason"),
        **source_candidate,
        "paper_source": paper_source,
        "paper_count_scope": _field(row, payload, "paper_count_scope", default="daily"),
        "paper_pnl_bps": paper_pnl_bps,
        "paper_pnl_bps_4h": paper_pnl_bps_4h,
        "paper_pnl_bps_8h": paper_pnl_bps_8h,
        "paper_pnl_bps_12h": paper_pnl_bps_12h,
        "paper_pnl_bps_24h": paper_pnl_bps_24h,
        "paper_pnl_bps_48h": paper_pnl_bps_48h,
        "paper_pnl_bps_72h": paper_pnl_bps_72h,
        "paper_pnl_usdt": paper_pnl_usdt,
        "arrival_bid": _optional_float(
            _field(row, payload, "arrival_bid", "entry_bid", "bid_at_arrival")
        ),
        "arrival_ask": _optional_float(
            _field(row, payload, "arrival_ask", "entry_ask", "ask_at_arrival")
        ),
        "arrival_mid": _optional_float(
            _field(row, payload, "arrival_mid", "entry_arrival_mid", "mid_at_arrival")
        ),
        "entry_decision_ts": _field(row, payload, "entry_decision_ts"),
        "exit_decision_ts": _field(row, payload, "exit_decision_ts"),
        "quote_timestamp": _field(row, payload, "quote_timestamp"),
        "quote_age_seconds": _optional_float(_field(row, payload, "quote_age_seconds")),
        "estimated_spread_bps": _optional_float(
            _field(
                row,
                payload,
                "estimated_spread_bps",
                "arrival_spread_bps",
                "spread_bps_at_decision",
                "spread_bps",
            )
        ),
        "expected_order_type": _field(
            row,
            payload,
            "expected_order_type",
            "order_type",
            "paper_order_type",
        ),
        "estimated_fill_px": _optional_float(
            _field(
                row,
                payload,
                "estimated_fill_px",
                "estimated_fill_price",
                "paper_fill_px",
                "virtual_entry_price",
            )
        ),
        "virtual_entry_price": _optional_float(_field(row, payload, "virtual_entry_price")),
        "virtual_exit_price": _optional_float(_field(row, payload, "virtual_exit_price")),
        "virtual_quantity": _optional_float(_field(row, payload, "virtual_quantity")),
        "paper_tracking_status": _field(
            row,
            payload,
            "paper_tracking_status",
            default=V5_PAPER_TRACKING_STATUS,
        ),
        "tracking_stage": enter_policy["tracking_stage"]
        or ("completed_paper_observations" if would_exit else "active_paper_strategy"),
        "sample_count": int(_optional_float(_field(row, payload, "sample_count")) or 0),
        "complete_sample_count": int(
            _optional_float(_field(row, payload, "complete_sample_count")) or 0
        ),
        "avg_net_bps": _optional_float(_field(row, payload, "avg_net_bps")),
        "p25_net_bps": _optional_float(_field(row, payload, "p25_net_bps")),
        "win_rate": _optional_float(_field(row, payload, "win_rate")),
        "cost_source": _cost_source_with_live_safety_markers(
            _field(row, payload, "cost_source"),
            row=row,
            payload=payload,
        ),
        "cost_trust_level": _field(row, payload, "cost_trust_level"),
        "market_regime": _field(row, payload, "market_regime"),
        "exit_reason": _field(row, payload, "exit_reason"),
        "gross_pnl_bps": _optional_float(_field(row, payload, "gross_pnl_bps")),
        "net_pnl_bps": _optional_float(_field(row, payload, "net_pnl_bps")),
        "max_favorable_excursion": _optional_float(_field(row, payload, "max_favorable_excursion")),
        "max_adverse_excursion": _optional_float(_field(row, payload, "max_adverse_excursion")),
        "holding_bars": int(_optional_float(_field(row, payload, "holding_bars")) or 0),
        "observability": (
            "OBSERVABLE"
            if _optional_float(_field(row, payload, "arrival_mid", "entry_arrival_mid")) is not None
            else _field(row, payload, "observability", default="NOT_OBSERVABLE")
        ),
        "not_observable_reason": _field(
            row,
            payload,
            "not_observable_reason",
            "no_sample_reason",
        ),
        "valid_for_promotion": _optional_bool(_field(row, payload, "valid_for_promotion")) is True,
        "cost_source_mix": _cost_source_with_live_safety_markers(
            _field(row, payload, "cost_source_mix"),
            row=row,
            payload=payload,
        ),
        "live_block_reason": _jsonish_text(_field(row, payload, "live_block_reason")),
        "required_paper_days": int(
            _optional_float(_field(row, payload, "required_paper_days")) or 14
        ),
        "required_slippage_coverage": _optional_float(
            _field(row, payload, "required_slippage_coverage")
        )
        or 0.8,
        "created_at": created_at,
        "source": V5_PAPER_TRACKING_SOURCE,
        "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
    }


def _v5_daily_row(row: dict[str, Any], created_at: str) -> dict[str, Any]:
    payload = _payload(row)
    candidate = _field(row, payload, "strategy_candidate", "experiment_name", "candidate")
    symbol = _field(row, payload, "symbol", "normalized_symbol")
    proposal_id = _field(row, payload, "proposal_id", "strategy_id") or _proposal_id_for(
        candidate,
        symbol,
    )
    paper_tracker_id = _field(
        row, payload, "paper_tracker_id", "strategy_id"
    ) or _paper_tracker_id_for(proposal_id, candidate, symbol)
    raw_paper_days = int(
        _optional_float(_field(row, payload, "paper_days", "paper_days_to_date")) or 0
    )
    required_days = int(_optional_float(_field(row, payload, "required_paper_days")) or 14)
    heartbeat_days = int(
        _optional_float(_field(row, payload, "heartbeat_days", "heartbeat_day_count"))
        or raw_paper_days
    )
    entry_day_count = int(_optional_float(_field(row, payload, "entry_day_count")) or 0)
    daily_would_enter_count = int(
        _optional_float(_field(row, payload, "daily_would_enter_count", "entry_count")) or 0
    )
    cumulative_would_raw = _field(
        row,
        payload,
        "cumulative_would_enter_count",
        "would_enter_count",
    )
    cumulative_would_value = _optional_float(cumulative_would_raw)
    cumulative_would_enter_count = (
        daily_would_enter_count if cumulative_would_value is None else int(cumulative_would_value)
    )
    daily_paper_pnl_observed_count = int(
        _optional_float(
            _field(
                row,
                payload,
                "daily_paper_pnl_observed_count",
                "complete_count",
            )
        )
        or 0
    )
    cumulative_pnl_raw = _field(
        row,
        payload,
        "cumulative_paper_pnl_observed_count",
        "paper_pnl_observed_count",
    )
    cumulative_pnl_value = _optional_float(cumulative_pnl_raw)
    cumulative_paper_pnl_observed_count = (
        daily_paper_pnl_observed_count
        if cumulative_pnl_value is None
        else int(cumulative_pnl_value)
    )
    paper_pnl_day_count = int(
        _optional_float(_field(row, payload, "paper_pnl_day_count", "paper_pnl_days")) or 0
    )
    required_entry_day_count = int(
        _optional_float(_field(row, payload, "required_entry_day_count"))
        or DEFAULT_REQUIRED_ENTRY_DAYS_FOR_LIVE
    )
    required_slippage_coverage = (
        _optional_float(_field(row, payload, "required_slippage_coverage")) or 0.8
    )
    arrival_mid_coverage = _optional_float(_field(row, payload, "arrival_mid_coverage")) or 0.0
    spread_observation_coverage = (
        _optional_float(_field(row, payload, "spread_observation_coverage")) or 0.0
    )
    cost_source_mix = _cost_source_with_live_safety_markers(
        _field(row, payload, "cost_source_mix"),
        row=row,
        payload=payload,
    )
    block_reasons = _paper_live_block_reasons(
        heartbeat_day_count=heartbeat_days,
        entry_day_count=entry_day_count,
        paper_pnl_observed_count=cumulative_paper_pnl_observed_count,
        paper_pnl_day_count=paper_pnl_day_count,
        required_paper_days=required_days,
        required_entry_day_count=required_entry_day_count,
        required_slippage_coverage=required_slippage_coverage,
        arrival_mid_coverage=arrival_mid_coverage,
        paper_slippage_coverage=min(arrival_mid_coverage, spread_observation_coverage),
        cost_source_mix=cost_source_mix,
    )
    row_out = {
        "as_of_date": _as_of_date(row, payload),
        "source_schema_version": _field(row, payload, "schema_version"),
        "proposal_id": proposal_id,
        "paper_tracker_id": paper_tracker_id,
        "strategy_candidate": candidate,
        "symbol": symbol,
        "recommended_mode": _field(row, payload, "recommended_mode", default="paper"),
        "paper_days": heartbeat_days,
        "heartbeat_days": heartbeat_days,
        "latest_board_decision": _field(
            row,
            payload,
            "latest_board_decision",
            default="PAPER_READY",
        ),
        "latest_horizon": _field(row, payload, "latest_horizon", "suggested_horizon", "horizon"),
        "heartbeat_day_count": heartbeat_days,
        "entry_day_count": entry_day_count,
        "daily_v5_entry_count": daily_would_enter_count,
        "daily_synthetic_would_enter_count": int(
            _optional_float(_field(row, payload, "daily_synthetic_would_enter_count")) or 0
        ),
        "cumulative_v5_entry_count": cumulative_would_enter_count,
        "cumulative_synthetic_would_enter_count": int(
            _optional_float(_field(row, payload, "cumulative_synthetic_would_enter_count")) or 0
        ),
        "daily_would_enter_count": daily_would_enter_count,
        "cumulative_would_enter_count": cumulative_would_enter_count,
        "would_enter_count": cumulative_would_enter_count,
        "daily_paper_pnl_observed_count": daily_paper_pnl_observed_count,
        "cumulative_paper_pnl_observed_count": cumulative_paper_pnl_observed_count,
        "paper_pnl_observed_count": cumulative_paper_pnl_observed_count,
        "count_scope": "cumulative",
        "paper_pnl_day_count": paper_pnl_day_count,
        "latest_paper_pnl_usdt": _optional_float(
            _field(row, payload, "latest_paper_pnl_usdt", "paper_pnl_usdt")
        ),
        "cumulative_paper_pnl_usdt": _optional_float(
            _field(
                row,
                payload,
                "cumulative_paper_pnl_usdt",
                "paper_pnl_usdt_sum",
                "paper_pnl_usdt",
            )
        )
        or 0.0,
        "avg_paper_pnl_bps": _optional_float(
            _field(row, payload, "avg_paper_pnl_bps", "paper_pnl_bps")
        ),
        "paper_pnl_observed_count_by_horizon": _jsonish_text(
            _field(row, payload, "paper_pnl_observed_count_by_horizon", default="{}")
        ),
        "complete_count_by_horizon": _jsonish_text(
            _field(
                row,
                payload,
                "complete_count_by_horizon",
                "paper_pnl_complete_count_by_horizon",
                default="{}",
            )
        ),
        "avg_paper_pnl_bps_by_horizon": _jsonish_text(
            _field(row, payload, "avg_paper_pnl_bps_by_horizon", default="{}")
        ),
        "win_rate_by_horizon": _jsonish_text(
            _field(row, payload, "win_rate_by_horizon", "paper_win_rate_by_horizon", default="{}")
        ),
        "paper_pnl_day_count_by_horizon": _jsonish_text(
            _field(row, payload, "paper_pnl_day_count_by_horizon", default="{}")
        ),
        "negative_entry_day_count": int(
            _optional_float(_field(row, payload, "negative_entry_day_count")) or 0
        ),
        "paper_negative_streak": int(
            _optional_float(_field(row, payload, "paper_negative_streak")) or 0
        ),
        "latest_paper_trend": _field(
            row,
            payload,
            "latest_paper_trend",
            default="not_observable",
        ),
        "paper_tracking_status": _field(
            row,
            payload,
            "paper_tracking_status",
            default=V5_PAPER_TRACKING_STATUS,
        ),
        "tracking_stage": "completed_paper_observations"
        if paper_pnl_day_count >= required_days
        else "active_paper_strategy",
        "required_paper_days": required_days,
        "required_entry_day_count": required_entry_day_count,
        "required_slippage_coverage": required_slippage_coverage,
        "arrival_mid_coverage": arrival_mid_coverage,
        "spread_observation_coverage": spread_observation_coverage,
        "cost_source_mix": cost_source_mix,
        "missing_cost_source_count": 1 if _is_missing_cost_source(cost_source_mix) else 0,
        "live_eligible": not block_reasons,
        "live_block_reason": safe_json_dumps(
            sorted({*_json_list(_field(row, payload, "live_block_reason")), *block_reasons})
        ),
        "created_at": created_at,
        "source": V5_PAPER_TRACKING_SOURCE,
        "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
    }
    return _with_paper_strategy_review(
        row_out,
        cfg=_config_for(proposal_id, candidate, symbol),
        extra_live_block_reasons=block_reasons,
    )


def _v5_slippage_row(row: dict[str, Any], created_at: str) -> dict[str, Any]:
    payload = _payload(row)
    candidate = _field(row, payload, "strategy_candidate", "experiment_name", "candidate")
    symbol = _field(row, payload, "symbol", "normalized_symbol")
    proposal_id = _field(row, payload, "proposal_id", "strategy_id") or _proposal_id_for(
        candidate,
        symbol,
    )
    paper_tracker_id = _field(
        row, payload, "paper_tracker_id", "strategy_id"
    ) or _paper_tracker_id_for(proposal_id, candidate, symbol)
    paper_days = int(_optional_float(_field(row, payload, "paper_days")) or 0)
    coverage = (
        _optional_float(_field(row, payload, "paper_slippage_coverage", "slippage_coverage")) or 0.0
    )
    arrival_mid_coverage = _optional_float(
        _field(row, payload, "arrival_mid_coverage", "mid_coverage")
    )
    spread_observation_coverage = _optional_float(
        _field(row, payload, "spread_observation_coverage", "spread_coverage")
    )
    cost_source_mix = _cost_source_with_live_safety_markers(
        _field(row, payload, "cost_source_mix"),
        row=row,
        payload=payload,
    )
    return {
        "as_of_date": _as_of_date(row, payload),
        "proposal_id": proposal_id,
        "paper_tracker_id": paper_tracker_id,
        "strategy_candidate": candidate,
        "symbol": symbol,
        "paper_days": paper_days,
        "observed_slippage_count": int(
            _optional_float(
                _field(row, payload, "observed_slippage_count", "slippage_covered_rows")
            )
            or 0
        ),
        "required_observation_count": int(
            _optional_float(_field(row, payload, "required_observation_count", "total_rows")) or 1
        ),
        "paper_slippage_coverage": coverage,
        "required_slippage_coverage": _optional_float(
            _field(row, payload, "required_slippage_coverage")
        )
        or 0.8,
        "arrival_mid_coverage": arrival_mid_coverage
        if arrival_mid_coverage is not None
        else coverage,
        "spread_observation_coverage": spread_observation_coverage
        if spread_observation_coverage is not None
        else coverage,
        "cost_source_mix": cost_source_mix,
        "missing_cost_source_count": 1 if _is_missing_cost_source(cost_source_mix) else 0,
        "coverage_status": _field(
            row,
            payload,
            "coverage_status",
            "readiness_status",
            default="insufficient_slippage_observations",
        ),
        "paper_tracking_status": _field(
            row,
            payload,
            "paper_tracking_status",
            default=V5_PAPER_TRACKING_STATUS,
        ),
        "tracking_stage": "completed_paper_observations"
        if coverage >= 0.8
        else "active_paper_strategy",
        "created_at": created_at,
        "source": V5_PAPER_TRACKING_SOURCE,
        "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
    }


def _resolve_as_of_date(board: pl.DataFrame, value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        normalized = value.strip().lower()
        if normalized != "auto":
            return date.fromisoformat(value.strip())
    if not board.is_empty() and "as_of_date" in board.columns:
        values = [
            str(item).strip()
            for item in board.get_column("as_of_date").drop_nulls().unique().to_list()
            if str(item).strip()
        ]
        if values:
            return date.fromisoformat(max(values))
    return datetime.now(UTC).date()


def _config_for(
    proposal_id: Any,
    strategy_candidate: Any,
    symbol: Any,
) -> PaperStrategyConfig:
    for cfg in PAPER_STRATEGIES:
        if proposal_id and str(proposal_id) == cfg.proposal_id:
            return cfg
        if (
            str(strategy_candidate or "") == cfg.strategy_candidate
            and str(symbol or "") == cfg.symbol
        ):
            return cfg
    return PaperStrategyConfig(
        proposal_id=str(proposal_id or ""),
        strategy_candidate=str(strategy_candidate or ""),
        symbol=str(symbol or ""),
    )


def _paper_tracker_id_for(
    proposal_id: Any,
    strategy_candidate: Any,
    symbol: Any,
) -> str:
    cfg = _config_for(proposal_id, strategy_candidate, symbol)
    return cfg.paper_tracker_id or cfg.proposal_id or str(proposal_id or "")


def _proposal_id_for(strategy_candidate: Any, symbol: Any) -> str:
    cfg = _config_for(None, strategy_candidate, symbol)
    return cfg.proposal_id


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_payload_json")
    if not raw:
        return {}
    try:
        loaded = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _field(
    row: dict[str, Any],
    payload: dict[str, Any],
    *keys: str,
    default: str = "",
) -> str:
    for key in keys:
        for source in (row, payload):
            value = source.get(key)
            if value is not None and str(value).strip() != "":
                return str(value)
    return default


def _as_of_date(row: dict[str, Any], payload: dict[str, Any]) -> str:
    value = _field(row, payload, "as_of_date", "paper_date", "date")
    if value:
        return value[:10]
    ts_value = _field(row, payload, "ts_utc", "created_at", "ingest_ts", "bundle_ts")
    if ts_value:
        return ts_value[:10]
    return datetime.now(UTC).date().isoformat()


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _jsonish_text(value: Any) -> str:
    if isinstance(value, list | dict):
        return safe_json_dumps(value)
    text = str(value or "")
    if not text:
        return "[]"
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return safe_json_dumps([text])
    return safe_json_dumps(loaded)


def _coverage_status(row: dict[str, Any], coverage: float) -> str:
    status = str(row.get("coverage_status") or "").strip()
    if status:
        return status
    if str(row.get("paper_tracking_status") or "") == "waiting_for_v5_paper_telemetry":
        return "waiting_for_v5_paper_telemetry"
    required = _optional_float(row.get("required_slippage_coverage")) or 0.8
    return "ok" if coverage >= required else "insufficient_slippage_observations"


def _paper_daily_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("proposal_id") or ""),
        str(row.get("strategy_candidate") or ""),
        str(row.get("symbol") or ""),
    )


def _paper_daily_key_aliases(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    proposal_id, candidate, symbol = _paper_daily_key(row)
    proposals = {proposal_id}
    candidates = {candidate}
    if proposal_id == BNB_F3_CURRENT_PROPOSAL_ID:
        proposals.add(BNB_F3_PAPER_PROPOSAL_ID)
        candidates.add(BNB_F3_STRATEGY_CANDIDATE)
    if proposal_id == BNB_F3_PAPER_PROPOSAL_ID:
        proposals.add(BNB_F3_CURRENT_PROPOSAL_ID)
        candidates.add(BNB_F3_GENERIC_STRATEGY_CANDIDATE)
    if proposal_id == SOL_F3_PAPER_PROPOSAL_ID:
        candidates.add(SOL_F3_STRATEGY_CANDIDATE)
        candidates.add("v5.sol_f3_dominant_entry")
    if proposal_id == ETH_F4_VOLUME_EXPANSION_PAPER_PROPOSAL_ID:
        candidates.add(ETH_F4_VOLUME_EXPANSION_STRATEGY_CANDIDATE)
        candidates.add("v5.eth_f4_volume_expansion_entry")
    if proposal_id == "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1":
        proposals.add("ETH_F3_DOMINANT_ENTRY_PAPER_V1")
    if proposal_id == "ETH_F3_DOMINANT_ENTRY_PAPER_V1":
        proposals.add("ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1")
    if candidate == "v5.eth_f3_dominant_entry":
        candidates.add("v5.f3_dominant_entry")
    if candidate == "v5.f3_dominant_entry":
        candidates.add("v5.eth_f3_dominant_entry")
    aliases = [
        (proposal, strategy_candidate, symbol)
        for proposal in proposals
        for strategy_candidate in candidates
    ]
    if proposal_id:
        aliases.append((proposal_id, "", symbol))
    return list(dict.fromkeys(aliases))


def _group_run_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _paper_daily_key(row)
        group = groups.setdefault(
            key,
            {
                "proposal_id": key[0],
                "strategy_candidate": key[1],
                "symbol": key[2],
                "rows": [],
            },
        )
        group["rows"].append(row)
    return groups


def _heartbeat_day_count(rows: list[dict[str, Any]]) -> int:
    heartbeat_days = {
        str(row.get("as_of_date") or "")
        for row in rows
        if not bool(row.get("would_enter"))
        and _optional_float(row.get("paper_pnl_usdt")) is None
        and _optional_float(row.get("paper_pnl_bps")) is None
        and not _paper_pnl_horizon_values(row)
    }
    heartbeat_days.discard("")
    return len(heartbeat_days)


def _paper_pnl_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _optional_float(row.get("paper_pnl_usdt")) is not None
        or _optional_float(row.get("paper_pnl_bps")) is not None
        or bool(_paper_pnl_horizon_values(row))
    ]


def _paper_pnl_day_count(rows: list[dict[str, Any]]) -> int:
    days = {str(row.get("as_of_date") or "") for row in rows}
    days.discard("")
    return len(days)


def _paper_pnl_bps_values(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    main_value = _optional_float(row.get("paper_pnl_bps"))
    if main_value is not None:
        values.append(main_value)
    values.extend(_paper_pnl_horizon_values(row).values())
    return values


def _paper_pnl_horizon_values(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for horizon in PAPER_PNL_HORIZON_HOURS:
        value = _optional_float(row.get(f"paper_pnl_bps_{horizon}h"))
        if value is not None:
            values[f"{horizon}h"] = value
    return values


def _paper_pnl_horizon_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values_by_horizon: dict[str, list[float]] = {}
    days_by_horizon: dict[str, set[str]] = {}
    for row in rows:
        row_day = str(row.get("as_of_date") or "")
        for horizon, value in _paper_pnl_horizon_values(row).items():
            values_by_horizon.setdefault(horizon, []).append(value)
            if row_day:
                days_by_horizon.setdefault(horizon, set()).add(row_day)
    horizon_keys = [f"{horizon}h" for horizon in PAPER_PNL_HORIZON_HOURS]
    return {
        "observed_count_by_horizon": {
            horizon: len(values_by_horizon.get(horizon, [])) for horizon in horizon_keys
        },
        "complete_count_by_horizon": {
            horizon: len(values_by_horizon.get(horizon, [])) for horizon in horizon_keys
        },
        "avg_bps_by_horizon": {
            horizon: (
                round(sum(values_by_horizon[horizon]) / len(values_by_horizon[horizon]), 10)
                if values_by_horizon.get(horizon)
                else None
            )
            for horizon in horizon_keys
        },
        "win_rate_by_horizon": {
            horizon: (
                round(
                    sum(1 for value in values_by_horizon[horizon] if value > 0)
                    / len(values_by_horizon[horizon]),
                    10,
                )
                if values_by_horizon.get(horizon)
                else None
            )
            for horizon in horizon_keys
        },
        "day_count_by_horizon": {
            horizon: len(days_by_horizon.get(horizon, set())) for horizon in horizon_keys
        },
    }


def _paper_negative_trend_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_day_horizon: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if bool(row.get("would_enter")) is not True:
            continue
        row_day = str(row.get("as_of_date") or "")
        if not row_day:
            continue
        for horizon in ["24h", "48h"]:
            value = _paper_pnl_horizon_values(row).get(horizon)
            if value is None:
                continue
            values_by_day_horizon.setdefault(row_day, {}).setdefault(horizon, []).append(value)
    negative_days: dict[str, bool] = {}
    for row_day, horizons in values_by_day_horizon.items():
        negative_days[row_day] = any(
            values and (sum(values) / len(values)) < 0.0 for values in horizons.values()
        )
    ordered_days = sorted(negative_days)
    streak = 0
    for row_day in reversed(ordered_days):
        if negative_days[row_day]:
            streak += 1
        else:
            break
    if not ordered_days:
        trend = "waiting_for_24h_48h_labels"
    elif streak >= 2:
        trend = "negative_24h_or_48h_streak"
    elif negative_days[ordered_days[-1]]:
        trend = "latest_entry_day_negative"
    else:
        trend = "latest_entry_day_non_negative"
    return {
        "negative_entry_day_count": sum(1 for is_negative in negative_days.values() if is_negative),
        "paper_negative_streak": streak,
        "latest_paper_trend": trend,
    }


def _daily_row_has_24h_48h_pnl(row: dict[str, Any]) -> bool:
    averages = _json_dict(row.get("avg_paper_pnl_bps_by_horizon"))
    return any(_optional_float(averages.get(horizon)) is not None for horizon in ["24h", "48h"])


def _daily_row_24h_48h_negative(row: dict[str, Any]) -> bool:
    averages = _json_dict(row.get("avg_paper_pnl_bps_by_horizon"))
    return any(
        (value is not None and value < 0.0)
        for value in (_optional_float(averages.get(horizon)) for horizon in ["24h", "48h"])
    )


def _with_paper_strategy_review(
    row: dict[str, Any],
    *,
    cfg: PaperStrategyConfig,
    extra_live_block_reasons: list[str] | None = None,
) -> dict[str, Any]:
    horizon_stats = {
        "observed_count_by_horizon": _json_dict(row.get("paper_pnl_observed_count_by_horizon")),
        "complete_count_by_horizon": _json_dict(row.get("complete_count_by_horizon")),
        "avg_bps_by_horizon": _json_dict(row.get("avg_paper_pnl_bps_by_horizon")),
        "day_count_by_horizon": _json_dict(row.get("paper_pnl_day_count_by_horizon")),
    }
    review = _paper_strategy_review(
        cfg=cfg,
        latest_board_decision=str(row.get("latest_board_decision") or ""),
        horizon_stats=horizon_stats,
        paper_negative_streak=int(_optional_float(row.get("paper_negative_streak")) or 0),
    )
    reasons = [
        *_paper_block_reason_list(row.get("live_block_reason")),
        *(extra_live_block_reasons or []),
        *review["live_block_reasons"],
    ]
    updated = dict(row)
    updated["latest_board_decision"] = review["latest_board_decision"]
    updated["live_block_reason"] = safe_json_dumps(sorted(set(reasons)))
    updated["live_eligible"] = bool(updated.get("live_eligible")) and not reasons
    return updated


def _paper_strategy_review(
    *,
    cfg: PaperStrategyConfig,
    latest_board_decision: str,
    horizon_stats: dict[str, dict[str, Any]],
    paper_negative_streak: int = 0,
) -> dict[str, Any]:
    decision = latest_board_decision or "PAPER_READY"
    reasons: list[str] = []
    if paper_negative_streak >= 2:
        if decision.upper() in {"PAPER_READY", "LIVE_SMALL_READY"}:
            decision = "KEEP_SHADOW"
        reasons.extend(
            [
                "paper_negative_24h_or_48h_streak",
                "review_before_resuming_paper_promotion",
            ]
        )
    if not _is_eth_f3_config(cfg):
        return {"latest_board_decision": decision, "live_block_reasons": reasons}

    observed_by_horizon = horizon_stats.get("observed_count_by_horizon", {})
    complete_by_horizon = horizon_stats.get("complete_count_by_horizon", {})
    avg_by_horizon = horizon_stats.get("avg_bps_by_horizon", {})
    primary_avg = _optional_float(avg_by_horizon.get(ETH_F3_PRIMARY_REVIEW_HORIZON))
    primary_complete_count = int(
        _optional_float(complete_by_horizon.get(ETH_F3_PRIMARY_REVIEW_HORIZON))
        or _optional_float(observed_by_horizon.get(ETH_F3_PRIMARY_REVIEW_HORIZON))
        or 0
    )
    if decision.upper() == "LIVE_SMALL_READY":
        decision = "PAPER_READY"
        reasons.append("eth_f3_not_live_ready_use_paper_review")
    if primary_avg is not None and primary_avg < 0.0:
        if decision.upper() in {"PAPER_READY", "LIVE_SMALL_READY"}:
            decision = "KEEP_SHADOW"
        reasons.extend(
            [
                "eth_f3_48h_paper_pnl_negative",
                "keep_shadow_until_48h_recovers",
            ]
        )
    elif primary_complete_count >= ETH_F3_MIN_PRIMARY_COMPLETE_COUNT and (
        primary_avg is not None and primary_avg > 0.0
    ):
        if decision.upper() not in {"KEEP_SHADOW", "KILL"}:
            decision = "PAPER_READY"
        reasons.append("eth_f3_48h_positive_continue_paper")
    elif primary_complete_count <= 0:
        reasons.append("waiting_for_longer_horizon_labels")
    elif primary_complete_count < ETH_F3_MIN_PRIMARY_COMPLETE_COUNT:
        reasons.append("waiting_for_48h_min_complete_samples")
    elif primary_avg is None:
        reasons.append("waiting_for_48h_avg_net_bps")
    reasons.append("eth_f3_paper_only_no_live")
    return {
        "latest_board_decision": decision,
        "live_block_reasons": sorted(set(reasons)),
    }


def _is_eth_f3_config(cfg: PaperStrategyConfig) -> bool:
    return cfg.proposal_id == ETH_F3_PAPER_PROPOSAL_ID or (
        cfg.strategy_candidate == ETH_F3_STRATEGY_CANDIDATE and cfg.symbol == ETH_F3_SYMBOL
    )


def _proposal_rank(row: dict[str, Any]) -> tuple[float, float, int, int, int]:
    return (
        _optional_float(row.get("avg_net_bps")) or float("-inf"),
        _optional_float(row.get("win_rate")) or float("-inf"),
        int(_optional_float(row.get("complete_sample_count")) or 0),
        int(_optional_float(row.get("sample_count")) or 0),
        int(_optional_float(row.get("horizon_hours")) or 0),
    )


def _suggested_horizon(row: dict[str, Any]) -> str:
    horizon = int(_optional_float(row.get("horizon_hours")) or 0)
    return f"{horizon}h" if horizon else ""


def _paper_cost_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entry_rows = [row for row in rows if bool(row.get("would_enter")) is True]
    observed_rows = entry_rows or rows
    denominator = max(len(observed_rows), 1)
    arrival_mid_count = sum(1 for row in observed_rows if _optional_float(row.get("arrival_mid")))
    spread_count = sum(1 for row in observed_rows if _has_spread_observation(row))
    cost_mix = _cost_source_mix_counts(observed_rows)
    missing_cost_source_count = _missing_cost_source_count(observed_rows)
    arrival_mid_coverage = arrival_mid_count / denominator
    spread_observation_coverage = spread_count / denominator
    return {
        "arrival_mid_coverage": arrival_mid_coverage,
        "spread_observation_coverage": spread_observation_coverage,
        "paper_slippage_coverage": min(arrival_mid_coverage, spread_observation_coverage),
        "cost_source_mix": cost_mix,
        "missing_cost_source_count": missing_cost_source_count,
    }


def _has_spread_observation(row: dict[str, Any]) -> bool:
    if _optional_float(row.get("estimated_spread_bps")) is not None:
        return True
    return (
        _optional_float(row.get("arrival_bid")) is not None
        and _optional_float(row.get("arrival_ask")) is not None
    )


def _paper_live_block_reasons(
    *,
    heartbeat_day_count: int,
    entry_day_count: int,
    paper_pnl_observed_count: int,
    paper_pnl_day_count: int,
    required_paper_days: int,
    required_entry_day_count: int,
    required_slippage_coverage: float,
    arrival_mid_coverage: float,
    paper_slippage_coverage: float,
    cost_source_mix: Any,
) -> list[str]:
    reasons: set[str] = set()
    cost_sources = _cost_source_mix_sources(cost_source_mix)
    has_actual_or_mixed = bool(cost_sources & {"actual_fills", "mixed_actual_proxy"})
    has_global_default = "global_default" in cost_sources
    has_live_blocking_cost = bool(cost_sources & LIVE_BLOCKING_COST_SOURCES)
    if heartbeat_day_count > 0 and entry_day_count == 0:
        reasons.add("paper_active_but_no_entries_yet")
    if entry_day_count < required_entry_day_count:
        reasons.add("insufficient_entry_days")
    if paper_pnl_observed_count <= 0:
        reasons.add("no_paper_pnl_observations")
    if paper_pnl_day_count < required_paper_days:
        reasons.add("insufficient_paper_pnl_days")
    if arrival_mid_coverage < required_slippage_coverage:
        reasons.add("insufficient_arrival_mid_coverage")
    if paper_slippage_coverage < required_slippage_coverage:
        reasons.add("no_live_slippage_coverage")
    if has_global_default:
        reasons.add("cost_source_global_default")
    if has_live_blocking_cost:
        reasons.add("cost_source_not_trusted")
    if not has_actual_or_mixed:
        reasons.add("cost_source_not_actual_or_mixed")
    return sorted(reasons)


def _live_block_reasons(row: dict[str, Any]) -> list[str]:
    reasons: set[str] = {
        "insufficient_paper_pnl_days",
        "insufficient_entry_days",
        "no_paper_pnl_observations",
        "no_live_slippage_coverage",
        "insufficient_arrival_mid_coverage",
    }
    if not _cost_source_mix_has_actual_or_mixed(row.get("cost_source_mix")):
        reasons.add("cost_source_not_actual_or_mixed")
    if _cost_source_mix_has_live_blocking(row.get("cost_source_mix")):
        reasons.add("cost_source_not_trusted")
    return sorted(reasons)


def _paper_block_reason_list(value: Any) -> list[str]:
    normalized: list[str] = []
    for reason in _json_list(value):
        for part in str(reason).replace(",", ";").split(";"):
            text = part.strip()
            if not text:
                continue
            if text == "no_paper_days":
                text = "insufficient_paper_pnl_days"
            if text not in normalized:
                normalized.append(text)
    return normalized


def _cost_source_mix_has_actual_or_mixed(value: Any) -> bool:
    return bool(_cost_source_mix_sources(value) & {"actual_fills", "mixed_actual_proxy"})


def _cost_source_mix_has_live_blocking(value: Any) -> bool:
    return bool(_cost_source_mix_sources(value) & LIVE_BLOCKING_COST_SOURCES)


def _cost_source_mix_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        sources = set(_cost_source_mix_sources(row.get("cost_source_mix")))
        source_tokens = set(_cost_source_mix_sources(row.get("cost_source")))
        if not sources:
            sources = source_tokens or {"missing"}
        elif source_tokens and sources & LIVE_BLOCKING_COST_SOURCES:
            sources.update(source_tokens)
        sources.update(_cost_live_safety_markers(row))
        for source in sources:
            if source:
                counts[source] = counts.get(source, 0) + 1
    return counts


def _cost_source_with_live_safety_markers(
    value: Any,
    *,
    row: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> str:
    payload = payload or {}
    original = str(
        value
        or row.get("cost_source")
        or payload.get("cost_source")
        or row.get("source")
        or payload.get("source")
        or ""
    ).strip()
    tokens = list(_cost_source_mix_sources(original))
    markers = _cost_live_safety_markers(row, payload)
    if not markers:
        return original
    tokens.extend(sorted(markers))
    deduped = list(dict.fromkeys(token for token in tokens if token))
    return ";".join(deduped)


def _missing_cost_source_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if _is_missing_cost_source(row.get("cost_source_mix"))
        and _is_missing_cost_source(row.get("cost_source"))
    )


def _is_missing_cost_source(value: Any) -> bool:
    sources = _cost_source_mix_sources(value)
    if not sources:
        return True
    return sources <= {"missing", "none", "null"}


def _cost_source_mix_sources(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        return {str(key).strip().lower() for key in value if str(key).strip()}
    if isinstance(value, list | tuple | set):
        sources: set[str] = set()
        for item in value:
            if isinstance(item, dict):
                item = item.get("cost_source") or item.get("source")
            text = str(item or "").strip().lower()
            if text:
                sources.add(text)
        return sources
    text = str(value or "").strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            source
            for source in (str(part or "").strip().lower() for part in re.split(r"[;,+|]", text))
            if source
        }
    return _cost_source_mix_sources(parsed)


def _cost_live_safety_markers(
    row: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> set[str]:
    data = {**(payload or {}), **row}
    markers: set[str] = set()
    fallback_level = data.get("fallback_level") or data.get("cost_fallback_level")
    fallback_reason = data.get("fallback_reason") or data.get("cost_fallback_reason")
    degraded_reason = data.get("degraded_reason") or data.get("cost_degraded_reason")
    degraded_cost_model = data.get("degraded_cost_model") or data.get("cost_degraded_model")
    if _cost_fallback_not_live_safe(fallback_level, fallback_reason):
        markers.add(COST_FALLBACK_NOT_LIVE_SAFE)
    if _truthy_cost_flag(degraded_cost_model) or _cost_degraded_reason_not_none(degraded_reason):
        markers.add(COST_DEGRADED_MODEL)
    return markers


def _cost_fallback_not_live_safe(fallback_level: Any, fallback_reason: Any) -> bool:
    level = _normalize_cost_fallback_token(fallback_level)
    reason = _normalize_cost_fallback_token(fallback_reason)
    return level not in SAFE_COST_FALLBACK_TOKENS or reason not in {"", "none"}


def _truthy_cost_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _cost_degraded_reason_not_none(value: Any) -> bool:
    return _normalize_cost_fallback_token(value) not in {
        "",
        "none",
        "null",
        "unknown",
        "n/a",
    }


def _normalize_cost_fallback_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
    return []


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Any) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)
