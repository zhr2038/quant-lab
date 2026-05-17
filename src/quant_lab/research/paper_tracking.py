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
PAPER_TRACKING_SOURCE = "research.paper_strategy_tracking.v0.1"
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
    "would_size_usdt": pl.Float64,
    "paper_pnl_bps": pl.Float64,
    "paper_pnl_usdt": pl.Float64,
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
    "would_enter_count": pl.Int64,
    "latest_paper_pnl_usdt": pl.Float64,
    "cumulative_paper_pnl_usdt": pl.Float64,
    "avg_paper_pnl_bps": pl.Float64,
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
    runs, warnings = build_paper_strategy_runs(board, as_of_date=day)
    run_rows = _publish_runs(root, runs)
    daily = build_paper_strategy_daily(root, runs, as_of_date=day)
    daily_rows = _publish_daily(root, daily)
    coverage = build_paper_slippage_coverage(daily, as_of_date=day)
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
    run_rows = [_run_row(row, by_key[key], created_at) for key, row in sorted(selected.items())]
    return pl.DataFrame(run_rows, schema=PAPER_RUN_SCHEMA, orient="row"), warnings


def build_paper_strategy_daily(
    lake_root: str | Path,
    new_runs: pl.DataFrame,
    *,
    as_of_date: date,
) -> pl.DataFrame:
    root = Path(lake_root)
    existing_runs = read_parquet_dataset(root / PAPER_STRATEGY_RUNS_DATASET)
    all_runs = _combine_unique_runs(existing_runs, new_runs)
    if all_runs.is_empty():
        return _empty_frame(PAPER_DAILY_SCHEMA)

    created_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for cfg in PAPER_STRATEGIES:
        proposal_rows = [
            row
            for row in all_runs.to_dicts()
            if row.get("proposal_id") == cfg.proposal_id
            and bool(row.get("would_enter")) is True
        ]
        if not proposal_rows:
            continue
        proposal_rows.sort(key=lambda row: str(row.get("as_of_date") or ""))
        latest = proposal_rows[-1]
        paper_days = len({str(row.get("as_of_date") or "") for row in proposal_rows})
        latest_reason = _json_list(latest.get("live_block_reason"))
        coverage = 0.0
        live_eligible = (
            paper_days >= cfg.required_paper_days
            and coverage >= cfg.required_slippage_coverage
            and not latest_reason
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
                "would_enter_count": len(proposal_rows),
                "latest_paper_pnl_usdt": _optional_float(latest.get("paper_pnl_usdt")),
                "cumulative_paper_pnl_usdt": sum(
                    _optional_float(row.get("paper_pnl_usdt")) or 0.0
                    for row in proposal_rows
                ),
                "avg_paper_pnl_bps": _mean(
                    _optional_float(row.get("paper_pnl_bps")) for row in proposal_rows
                ),
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
                "coverage_status": "insufficient_slippage_observations",
                "created_at": created_at,
                "source": PAPER_TRACKING_SOURCE,
                "schema_version": PAPER_TRACKING_SCHEMA_VERSION,
            }
        )
    return pl.DataFrame(rows, schema=PAPER_SLIPPAGE_SCHEMA, orient="row")


def _publish_runs(lake_root: Path, runs: pl.DataFrame) -> int:
    path = lake_root / PAPER_STRATEGY_RUNS_DATASET
    existing = read_parquet_dataset(path)
    combined = _combine_unique_runs(existing, runs)
    write_parquet_dataset(combined, path)
    return combined.height


def _publish_daily(lake_root: Path, daily: pl.DataFrame) -> int:
    path = lake_root / PAPER_STRATEGY_DAILY_DATASET
    existing = read_parquet_dataset(path)
    combined = _replace_by_key(existing, daily, ["proposal_id", "as_of_date"])
    write_parquet_dataset(combined, path)
    return combined.height


def _publish_slippage(lake_root: Path, coverage: pl.DataFrame) -> int:
    path = lake_root / PAPER_SLIPPAGE_COVERAGE_DATASET
    existing = read_parquet_dataset(path)
    combined = _replace_by_key(existing, coverage, ["proposal_id", "as_of_date"])
    write_parquet_dataset(combined, path)
    return combined.height


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


def _run_row(row: dict[str, Any], cfg: PaperStrategyConfig, created_at: str) -> dict[str, Any]:
    avg_net_bps = _optional_float(row.get("avg_net_bps"))
    paper_pnl_usdt = cfg.paper_size_usdt * (avg_net_bps or 0.0) / 10_000.0
    return {
        "as_of_date": str(row.get("as_of_date") or ""),
        "proposal_id": cfg.proposal_id,
        "strategy_candidate": cfg.strategy_candidate,
        "symbol": cfg.symbol,
        "recommended_mode": "paper",
        "board_decision": "PAPER_READY",
        "suggested_horizon": _suggested_horizon(row),
        "horizon_hours": int(_optional_float(row.get("horizon_hours")) or 0),
        "would_enter": True,
        "would_exit": False,
        "would_size_usdt": cfg.paper_size_usdt,
        "paper_pnl_bps": avg_net_bps,
        "paper_pnl_usdt": paper_pnl_usdt,
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
