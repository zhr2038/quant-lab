from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.alpha_discovery import normalize_alpha_discovery_board_decisions
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps

PAPER_STRATEGY_RUNS_DATASET = Path("gold") / "paper_strategy_runs"
PAPER_STRATEGY_DAILY_DATASET = Path("gold") / "paper_strategy_daily"
PAPER_SLIPPAGE_COVERAGE_DATASET = Path("gold") / "paper_slippage_coverage"
ALPHA_DISCOVERY_BOARD_DATASET = Path("gold") / "alpha_discovery_board"
V5_PAPER_STRATEGY_RUN_DATASET = Path("silver") / "v5_paper_strategy_run"
V5_PAPER_STRATEGY_DAILY_DATASET = Path("silver") / "v5_paper_strategy_daily"
V5_PAPER_SLIPPAGE_COVERAGE_DATASET = Path("silver") / "v5_paper_slippage_coverage"
PAPER_TRACKING_SOURCE = "research.paper_strategy_tracking.v0.1"
V5_PAPER_TRACKING_SOURCE = "v5.paper_strategy_telemetry"
V5_PAPER_TRACKING_STATUS = "active"
PAPER_TRACKING_SCHEMA_VERSION = "paper_strategy_tracking.v1"

PAPER_RUN_SCHEMA = {
    "as_of_date": pl.Utf8,
    "proposal_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "board_decision": pl.Utf8,
    "suggested_horizon": pl.Utf8,
    "horizon_hours": pl.Int64,
    "would_enter": pl.Boolean,
    "would_exit": pl.Boolean,
    "would_size": pl.Float64,
    "would_size_usdt": pl.Float64,
    "paper_pnl_bps": pl.Float64,
    "paper_pnl_usdt": pl.Float64,
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
    "proposal_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "paper_days": pl.Int64,
    "latest_board_decision": pl.Utf8,
    "latest_horizon": pl.Utf8,
    "heartbeat_day_count": pl.Int64,
    "entry_day_count": pl.Int64,
    "would_enter_count": pl.Int64,
    "paper_pnl_observed_count": pl.Int64,
    "latest_paper_pnl_usdt": pl.Float64,
    "cumulative_paper_pnl_usdt": pl.Float64,
    "avg_paper_pnl_bps": pl.Float64,
    "paper_tracking_status": pl.Utf8,
    "tracking_stage": pl.Utf8,
    "required_paper_days": pl.Int64,
    "required_slippage_coverage": pl.Float64,
    "live_eligible": pl.Boolean,
    "live_block_reason": pl.Utf8,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}

PAPER_SLIPPAGE_SCHEMA = {
    "as_of_date": pl.Utf8,
    "proposal_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "paper_days": pl.Int64,
    "observed_slippage_count": pl.Int64,
    "required_observation_count": pl.Int64,
    "paper_slippage_coverage": pl.Float64,
    "required_slippage_coverage": pl.Float64,
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
    paper_size_usdt: float = 100.0
    required_paper_days: int = 14
    required_slippage_coverage: float = 0.8


PAPER_STRATEGIES = [
    PaperStrategyConfig(
        proposal_id="SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
        strategy_candidate="v5.sol_protect_alpha6_low_exception",
        symbol="SOL-USDT",
    ),
    PaperStrategyConfig(
        proposal_id="SOL_F4_VOLUME_EXPANSION_PAPER_V1",
        strategy_candidate="v5.f4_volume_expansion_entry",
        symbol="SOL-USDT",
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
    v5_runs_raw = latest_v5_paper_frame(
        read_parquet_dataset(root / V5_PAPER_STRATEGY_RUN_DATASET)
    )
    v5_daily_raw = latest_v5_paper_frame(
        read_parquet_dataset(root / V5_PAPER_STRATEGY_DAILY_DATASET)
    )
    v5_slippage_raw = latest_v5_paper_frame(
        read_parquet_dataset(root / V5_PAPER_SLIPPAGE_COVERAGE_DATASET)
    )
    if any(
        not frame.is_empty()
        for frame in [v5_runs_raw, v5_daily_raw, v5_slippage_raw]
    ):
        runs = build_paper_strategy_runs_from_v5(v5_runs_raw)
        daily = build_paper_strategy_daily_from_v5(v5_daily_raw)
        if not runs.is_empty():
            daily = enrich_paper_strategy_daily_from_runs(
                daily,
                runs,
                as_of_date=day,
            )
        if daily.is_empty() and not runs.is_empty():
            daily = build_paper_strategy_daily_from_runs(runs, as_of_date=day)
        coverage = build_paper_slippage_coverage_from_v5(v5_slippage_raw)
        if coverage.is_empty() and not daily.is_empty():
            coverage = build_paper_slippage_coverage(daily, as_of_date=day)
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
                "latest_board_decision": str(row.get("board_decision") or "PAPER_READY"),
                "latest_horizon": str(row.get("suggested_horizon") or ""),
                "heartbeat_day_count": 0,
                "entry_day_count": 0,
                "would_enter_count": 0,
                "paper_pnl_observed_count": 0,
                "latest_paper_pnl_usdt": None,
                "cumulative_paper_pnl_usdt": 0.0,
                "avg_paper_pnl_bps": None,
                "paper_tracking_status": "waiting_for_v5_paper_telemetry",
                "tracking_stage": "proposed_paper_strategy",
                "required_paper_days": cfg.required_paper_days,
                "required_slippage_coverage": cfg.required_slippage_coverage,
                "live_eligible": False,
                "live_block_reason": safe_json_dumps(
                    sorted(
                        {
                            *_json_list(row.get("live_block_reason")),
                            "waiting_for_v5_paper_telemetry",
                            "no_paper_days",
                            "no_live_slippage_coverage",
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
        if not run_rows:
            continue
        run_rows.sort(key=lambda row: str(row.get("as_of_date") or ""))
        latest = run_rows[-1]
        paper_days = len({str(row.get("as_of_date") or "") for row in run_rows})
        heartbeat_days = _heartbeat_day_count(run_rows)
        entry_days = len({str(row.get("as_of_date") or "") for row in entry_rows})
        pnl_rows = _paper_pnl_rows(entry_rows)
        latest_reason = _json_list(latest.get("live_block_reason"))
        coverage = 0.0
        live_eligible = (
            paper_days >= cfg.required_paper_days
            and coverage >= cfg.required_slippage_coverage
            and not latest_reason
        )
        tracking_stage = (
            "completed_paper_observations"
            if paper_days >= cfg.required_paper_days
            else "active_paper_strategy"
        )
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "proposal_id": cfg.proposal_id,
                "strategy_candidate": cfg.strategy_candidate,
                "symbol": cfg.symbol,
                "recommended_mode": "paper",
                "paper_days": paper_days,
                "latest_board_decision": str(latest.get("board_decision") or ""),
                "latest_horizon": str(latest.get("suggested_horizon") or ""),
                "heartbeat_day_count": heartbeat_days,
                "entry_day_count": entry_days,
                "would_enter_count": len(entry_rows),
                "paper_pnl_observed_count": len(pnl_rows),
                "latest_paper_pnl_usdt": _optional_float(pnl_rows[-1].get("paper_pnl_usdt"))
                if pnl_rows
                else None,
                "cumulative_paper_pnl_usdt": sum(
                    _optional_float(row.get("paper_pnl_usdt")) or 0.0
                    for row in pnl_rows
                ),
                "avg_paper_pnl_bps": _mean(
                    _optional_float(row.get("paper_pnl_bps")) for row in pnl_rows
                ),
                "paper_tracking_status": V5_PAPER_TRACKING_STATUS,
                "tracking_stage": tracking_stage,
                "required_paper_days": cfg.required_paper_days,
                "required_slippage_coverage": cfg.required_slippage_coverage,
                "live_eligible": live_eligible,
                "live_block_reason": safe_json_dumps(
                    sorted(
                        {
                            *latest_reason,
                            "cost_source_not_actual_or_mixed",
                            "no_live_slippage_coverage",
                        }
                    )
                ),
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
        observed = 0
        required = max(paper_days, 1)
        coverage = observed / required
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
                "required_slippage_coverage": _optional_float(
                    row.get("required_slippage_coverage")
                )
                or 0.8,
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
    return {
        "as_of_date": str(row.get("as_of_date") or ""),
        "proposal_id": cfg.proposal_id,
        "strategy_candidate": cfg.strategy_candidate,
        "symbol": cfg.symbol,
        "recommended_mode": "paper",
        "board_decision": "PAPER_READY",
        "suggested_horizon": _suggested_horizon(row),
        "horizon_hours": int(_optional_float(row.get("horizon_hours")) or 0),
        "would_enter": False,
        "would_exit": False,
        "would_size": 0.0,
        "would_size_usdt": 0.0,
        "paper_pnl_bps": None,
        "paper_pnl_usdt": None,
        "paper_tracking_status": "waiting_for_v5_paper_telemetry",
        "tracking_stage": "proposed_paper_strategy",
        "sample_count": int(_optional_float(row.get("sample_count")) or 0),
        "complete_sample_count": int(_optional_float(row.get("complete_sample_count")) or 0),
        "avg_net_bps": avg_net_bps,
        "p25_net_bps": _optional_float(row.get("p25_net_bps")),
        "win_rate": _optional_float(row.get("win_rate")),
        "cost_source_mix": str(row.get("cost_source_mix") or ""),
        "live_block_reason": safe_json_dumps(_live_block_reasons(row)),
        "required_paper_days": cfg.required_paper_days,
        "required_slippage_coverage": cfg.required_slippage_coverage,
        "created_at": created_at,
        "source": PAPER_TRACKING_SOURCE,
        "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
    }


def build_paper_strategy_runs_from_v5(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_frame(PAPER_RUN_SCHEMA)
    created_at = datetime.now(UTC).isoformat()
    rows = [_v5_run_row(row, created_at) for row in frame.to_dicts()]
    return pl.DataFrame(rows, schema=PAPER_RUN_SCHEMA, orient="row")


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
    return pl.DataFrame(rows, schema=PAPER_DAILY_SCHEMA, orient="row")


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
    metrics = {
        _paper_daily_key(row): row
        for row in run_daily.to_dicts()
    }
    enriched: list[dict[str, Any]] = []
    metric_fields = [
        "heartbeat_day_count",
        "entry_day_count",
        "would_enter_count",
        "paper_pnl_observed_count",
    ]
    fallback_fields = [
        "latest_paper_pnl_usdt",
        "cumulative_paper_pnl_usdt",
        "avg_paper_pnl_bps",
    ]
    for row in daily.to_dicts():
        metric = metrics.get(_paper_daily_key(row), {})
        updated = dict(row)
        for field in metric_fields:
            updated[field] = int(_optional_float(metric.get(field)) or 0)
        for field in fallback_fields:
            if updated.get(field) in {None, ""} and metric.get(field) not in {None, ""}:
                updated[field] = metric.get(field)
        enriched.append(updated)
    return pl.DataFrame(enriched, schema=PAPER_DAILY_SCHEMA, orient="row")


def build_paper_slippage_coverage_from_v5(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_frame(PAPER_SLIPPAGE_SCHEMA)
    created_at = datetime.now(UTC).isoformat()
    rows = [_v5_slippage_row(row, created_at) for row in frame.to_dicts()]
    return pl.DataFrame(rows, schema=PAPER_SLIPPAGE_SCHEMA, orient="row")


def _v5_run_row(row: dict[str, Any], created_at: str) -> dict[str, Any]:
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
    would_exit = _optional_bool(_field(row, payload, "would_exit", "exit")) is True
    return {
        "as_of_date": _as_of_date(row, payload),
        "proposal_id": proposal_id,
        "strategy_candidate": candidate,
        "symbol": symbol,
        "recommended_mode": _field(row, payload, "recommended_mode", default="paper"),
        "board_decision": _field(row, payload, "board_decision", default="PAPER_READY"),
        "suggested_horizon": _field(row, payload, "suggested_horizon", "horizon"),
        "horizon_hours": int(_optional_float(_field(row, payload, "horizon_hours")) or 0),
        "would_enter": _optional_bool(_field(row, payload, "would_enter", "enter")) is True,
        "would_exit": would_exit,
        "would_size": _optional_float(
            _field(row, payload, "would_size", "would_size_notional", "size")
        )
        or 0.0,
        "would_size_usdt": _optional_float(
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
        "paper_pnl_bps": _optional_float(_field(row, payload, "paper_pnl_bps", "pnl_bps")),
        "paper_pnl_usdt": _optional_float(
            _field(row, payload, "paper_pnl_usdt", "paper_pnl", "pnl_usdt")
        ),
        "paper_tracking_status": _field(
            row,
            payload,
            "paper_tracking_status",
            default=V5_PAPER_TRACKING_STATUS,
        ),
        "tracking_stage": "completed_paper_observations"
        if would_exit
        else "active_paper_strategy",
        "sample_count": int(_optional_float(_field(row, payload, "sample_count")) or 0),
        "complete_sample_count": int(
            _optional_float(_field(row, payload, "complete_sample_count")) or 0
        ),
        "avg_net_bps": _optional_float(_field(row, payload, "avg_net_bps")),
        "p25_net_bps": _optional_float(_field(row, payload, "p25_net_bps")),
        "win_rate": _optional_float(_field(row, payload, "win_rate")),
        "cost_source_mix": _field(row, payload, "cost_source_mix"),
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
    paper_days = int(
        _optional_float(_field(row, payload, "paper_days", "paper_days_to_date")) or 0
    )
    required_days = int(_optional_float(_field(row, payload, "required_paper_days")) or 14)
    return {
        "as_of_date": _as_of_date(row, payload),
        "proposal_id": proposal_id,
        "strategy_candidate": candidate,
        "symbol": symbol,
        "recommended_mode": _field(row, payload, "recommended_mode", default="paper"),
        "paper_days": paper_days,
        "latest_board_decision": _field(
            row,
            payload,
            "latest_board_decision",
            default="PAPER_READY",
        ),
        "latest_horizon": _field(row, payload, "latest_horizon", "suggested_horizon", "horizon"),
        "heartbeat_day_count": int(
            _optional_float(_field(row, payload, "heartbeat_day_count")) or 0
        ),
        "entry_day_count": int(
            _optional_float(_field(row, payload, "entry_day_count")) or 0
        ),
        "would_enter_count": int(
            _optional_float(_field(row, payload, "would_enter_count", "entry_count")) or 0
        ),
        "paper_pnl_observed_count": int(
            _optional_float(_field(row, payload, "paper_pnl_observed_count", "complete_count"))
            or 0
        ),
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
        "paper_tracking_status": _field(
            row,
            payload,
            "paper_tracking_status",
            default=V5_PAPER_TRACKING_STATUS,
        ),
        "tracking_stage": "completed_paper_observations"
        if paper_days >= required_days
        else "active_paper_strategy",
        "required_paper_days": required_days,
        "required_slippage_coverage": _optional_float(
            _field(row, payload, "required_slippage_coverage")
        )
        or 0.8,
        "live_eligible": _optional_bool(
            _field(row, payload, "live_eligible", "live_small_ready")
        )
        is True,
        "live_block_reason": _jsonish_text(_field(row, payload, "live_block_reason")),
        "created_at": created_at,
        "source": V5_PAPER_TRACKING_SOURCE,
        "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
    }


def _v5_slippage_row(row: dict[str, Any], created_at: str) -> dict[str, Any]:
    payload = _payload(row)
    candidate = _field(row, payload, "strategy_candidate", "experiment_name", "candidate")
    symbol = _field(row, payload, "symbol", "normalized_symbol")
    proposal_id = _field(row, payload, "proposal_id", "strategy_id") or _proposal_id_for(
        candidate,
        symbol,
    )
    paper_days = int(_optional_float(_field(row, payload, "paper_days")) or 0)
    coverage = _optional_float(
        _field(row, payload, "paper_slippage_coverage", "slippage_coverage")
    ) or 0.0
    return {
        "as_of_date": _as_of_date(row, payload),
        "proposal_id": proposal_id,
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
    }
    heartbeat_days.discard("")
    return len(heartbeat_days)


def _paper_pnl_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _optional_float(row.get("paper_pnl_usdt")) is not None
        or _optional_float(row.get("paper_pnl_bps")) is not None
    ]


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


def _live_block_reasons(row: dict[str, Any]) -> list[str]:
    reasons: set[str] = {"no_paper_days", "no_live_slippage_coverage"}
    if not _cost_source_mix_has_actual_or_mixed(row.get("cost_source_mix")):
        reasons.add("cost_source_not_actual_or_mixed")
    return sorted(reasons)


def _cost_source_mix_has_actual_or_mixed(value: Any) -> bool:
    text = safe_json_dumps(value) if isinstance(value, dict | list | tuple) else str(value or "")
    text = text.lower()
    return "actual_fills" in text or "mixed_actual_proxy" in text


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
