from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.strategy_evidence import (
    _canonical_candidate_name,
    strategy_evidence_decision_ladder,
)
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

ALPHA_DISCOVERY_BOARD_DATASET = Path("gold") / "alpha_discovery_board"
SOURCE_NAME = "research.alpha_discovery_board.v0.1"
BOARD_SCHEMA_VERSION = "alpha_discovery_board.v1"

LABEL_DATASET = Path("gold") / "v5_candidate_label"
EVENT_DATASET = Path("silver") / "v5_candidate_event"
STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"
HIGH_SCORE_BLOCKED_OUTCOME_DATASET = Path("silver") / "v5_high_score_blocked_outcome"
SHADOW_OUTCOME_DATASET = Path("silver") / "v5_shadow_outcome"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
TRADE_EVENT_DATASET = Path("silver") / "v5_trade_event"
RISK_PERMISSION_DATASET = Path("gold") / "risk_permission"
V5_ENFORCEMENT_DATASET = Path("gold") / "v5_quant_lab_enforcement_daily"

DECISIONS = (
    "KILL",
    "RESEARCH_ONLY",
    "KEEP_SHADOW",
    "PAPER_READY",
    "LIVE_SMALL_READY",
)

BOARD_SCHEMA: dict[str, Any] = {
    "strategy": pl.Utf8,
    "board_schema_version": pl.Utf8,
    "as_of_date": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "candidate_name": pl.Utf8,
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
    "cost_source_mix": pl.Utf8,
    "stability_by_day": pl.Utf8,
    "paper_days": pl.Int64,
    "cost_source_has_global_default": pl.Boolean,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "risk_permission": pl.Utf8,
    "risk_permission_status": pl.Utf8,
    "enforce_readiness_status": pl.Utf8,
    "block_reason_mix": pl.Utf8,
    "final_decision_mix": pl.Utf8,
    "high_score_blocked_outcome_count": pl.Int64,
    "shadow_outcome_count": pl.Int64,
    "start_ts": pl.Datetime(time_zone="UTC"),
    "end_ts": pl.Datetime(time_zone="UTC"),
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


class AlphaDiscoveryBoardBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of_date: str
    alpha_discovery_board_rows: int = Field(ge=0)
    candidate_label_rows: int = Field(ge=0)
    candidate_event_rows: int = Field(ge=0)
    decision_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_alpha_discovery_board(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> AlphaDiscoveryBoardBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    created_at = datetime.now(UTC)

    labels = read_parquet_dataset(root / LABEL_DATASET)
    events = read_parquet_dataset(root / EVENT_DATASET)
    strategy_evidence = read_parquet_dataset(root / STRATEGY_EVIDENCE_DATASET)
    cost_bucket_daily = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET)
    trades = read_parquet_dataset(root / TRADE_EVENT_DATASET)
    risk_permission = read_parquet_dataset(root / RISK_PERMISSION_DATASET)
    blocked_outcomes = read_parquet_dataset(root / HIGH_SCORE_BLOCKED_OUTCOME_DATASET)
    shadow_outcomes = read_parquet_dataset(root / SHADOW_OUTCOME_DATASET)
    readiness = _enforce_readiness_context(root)

    if not strategy_evidence.is_empty():
        board = build_alpha_discovery_board_from_strategy_evidence(
            strategy_evidence=strategy_evidence,
            events=events,
            trades=trades,
            risk_permission=risk_permission,
            blocked_outcomes=blocked_outcomes,
            shadow_outcomes=shadow_outcomes,
            as_of_date=day,
            created_at=created_at,
            readiness=readiness,
        )
    else:
        board = build_alpha_discovery_board(
            labels=labels,
            events=events,
            cost_bucket_daily=cost_bucket_daily,
            trades=trades,
            risk_permission=risk_permission,
            blocked_outcomes=blocked_outcomes,
            shadow_outcomes=shadow_outcomes,
            as_of_date=day,
            created_at=created_at,
            readiness=readiness,
        )
    rows = _upsert_if_not_empty(
        board,
        root / ALPHA_DISCOVERY_BOARD_DATASET,
        [
            "strategy",
            "board_schema_version",
            "as_of_date",
            "strategy_candidate",
            "symbol",
            "regime_state",
            "horizon_hours",
        ],
    )
    warnings: list[str] = []
    if labels.is_empty() and strategy_evidence.is_empty():
        warnings.append("v5_candidate_label_empty")
    if board.is_empty():
        warnings.append("alpha_discovery_board_empty")
    return AlphaDiscoveryBoardBuildResult(
        as_of_date=day.isoformat(),
        alpha_discovery_board_rows=rows,
        candidate_label_rows=labels.height,
        candidate_event_rows=events.height,
        decision_counts=_decision_counts(board),
        warnings=warnings,
    )


def build_alpha_discovery_board(
    *,
    labels: pl.DataFrame,
    events: pl.DataFrame,
    cost_bucket_daily: pl.DataFrame,
    trades: pl.DataFrame,
    risk_permission: pl.DataFrame,
    blocked_outcomes: pl.DataFrame,
    shadow_outcomes: pl.DataFrame,
    as_of_date: date,
    created_at: datetime | None = None,
    readiness: dict[str, Any] | None = None,
) -> pl.DataFrame:
    if labels.is_empty():
        return pl.DataFrame(schema=BOARD_SCHEMA)

    cutoff = datetime.combine(as_of_date + timedelta(days=1), time.min, tzinfo=UTC)
    cost_sources = _cost_sources_by_symbol(cost_bucket_daily)
    normalized_rows = [
        _normalize_label_row(row, cost_sources)
        for row in labels.to_dicts()
        if _within_as_of(row.get("ts_utc"), cutoff)
    ]
    if not normalized_rows:
        return pl.DataFrame(schema=BOARD_SCHEMA)

    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        groups[
            (
                row["strategy_candidate"],
                row["symbol"],
                row["regime_state"],
                row["horizon_hours"],
            )
        ].append(row)

    paper_days = _paper_days_by_group(events, trades, as_of_date=as_of_date)
    risk = _risk_context(risk_permission)
    readiness = readiness or {"readiness_status": "UNKNOWN"}
    blocked_counts = _legacy_counts(blocked_outcomes, default_candidate="UNKNOWN")
    shadow_counts = _legacy_counts(
        shadow_outcomes,
        default_candidate="v5.alt_impulse_shadow",
    )

    created = created_at or datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for (candidate, symbol, regime, horizon), group_rows in sorted(groups.items()):
        rows.append(
            _board_row(
                candidate=candidate,
                symbol=symbol,
                regime=regime,
                horizon=horizon,
                rows=group_rows,
                as_of_date=as_of_date,
                created_at=created,
                paper_days=paper_days.get((candidate, symbol, regime), 0),
                risk=risk,
                readiness=readiness,
                blocked_count=blocked_counts.get((candidate, symbol), 0),
                shadow_count=shadow_counts.get((candidate, symbol), 0),
            )
        )
    return pl.DataFrame(rows, schema=BOARD_SCHEMA, orient="row")


def build_alpha_discovery_board_from_strategy_evidence(
    *,
    strategy_evidence: pl.DataFrame,
    events: pl.DataFrame,
    trades: pl.DataFrame,
    risk_permission: pl.DataFrame,
    blocked_outcomes: pl.DataFrame,
    shadow_outcomes: pl.DataFrame,
    as_of_date: date,
    created_at: datetime | None = None,
    readiness: dict[str, Any] | None = None,
) -> pl.DataFrame:
    if strategy_evidence.is_empty():
        return pl.DataFrame(schema=BOARD_SCHEMA)
    strategy_evidence = _latest_strategy_evidence_as_of(strategy_evidence, as_of_date)
    cutoff = datetime.combine(as_of_date + timedelta(days=1), time.min, tzinfo=UTC)
    rows = [
        row
        for row in strategy_evidence.to_dicts()
        if _within_as_of(
            row.get("end_ts") or row.get("created_at") or row.get("as_of_date"),
            cutoff,
        )
    ]
    if not rows:
        return pl.DataFrame(schema=BOARD_SCHEMA)
    paper_days = _paper_days_by_group(events, trades, as_of_date=as_of_date)
    risk = _risk_context(risk_permission)
    readiness = readiness or {"readiness_status": "UNKNOWN"}
    blocked_counts = _legacy_counts(blocked_outcomes, default_candidate="UNKNOWN")
    shadow_counts = _legacy_counts(shadow_outcomes, default_candidate="v5.alt_impulse_shadow")
    created = created_at or datetime.now(UTC)
    board_rows: list[dict[str, Any]] = []
    for row in rows:
        candidate = _clean_text(row.get("strategy_candidate") or row.get("candidate_name"))
        symbol = normalize_symbol(_clean_text(row.get("symbol"))) or "UNKNOWN"
        regime = _clean_text(row.get("regime_state")) or "UNKNOWN"
        horizon = int(_finite_float(row.get("horizon_hours")) or 0)
        decision, reasons = _strategy_evidence_decision(row)
        board_rows.append(
            {
                "strategy": _clean_text(row.get("strategy")) or "v5",
                "board_schema_version": BOARD_SCHEMA_VERSION,
                "as_of_date": as_of_date.isoformat(),
                "strategy_candidate": candidate,
                "candidate_name": _clean_text(row.get("candidate_name")) or candidate,
                "symbol": symbol,
                "regime_state": regime,
                "horizon_hours": horizon,
                "sample_count": int(_finite_float(row.get("sample_count")) or 0),
                "complete_sample_count": int(_finite_float(row.get("complete_sample_count")) or 0),
                "avg_net_bps": _finite_float(row.get("avg_net_bps")),
                "median_net_bps": _finite_float(row.get("median_net_bps")),
                "p25_net_bps": _finite_float(row.get("p25_net_bps")),
                "win_rate": _finite_float(row.get("win_rate")),
                "avg_mfe_bps": None,
                "avg_mae_bps": None,
                "cost_source_mix": _clean_text(row.get("cost_source_mix")) or "[]",
                "stability_by_day": "[]",
                "paper_days": paper_days.get((candidate, symbol, regime), 0),
                "cost_source_has_global_default": "global_default"
                in (_clean_text(row.get("cost_source_mix")).lower()),
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
                "risk_permission": str(risk.get("permission") or "UNKNOWN"),
                "risk_permission_status": str(risk.get("permission_status") or "UNKNOWN"),
                "enforce_readiness_status": str(readiness.get("readiness_status") or "UNKNOWN"),
                "block_reason_mix": "{}",
                "final_decision_mix": "{}",
                "high_score_blocked_outcome_count": blocked_counts.get((candidate, symbol), 0),
                "shadow_outcome_count": shadow_counts.get((candidate, symbol), 0),
                "start_ts": _coerce_timestamp(row.get("start_ts")),
                "end_ts": _coerce_timestamp(row.get("end_ts")),
                "created_at": created,
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(board_rows, schema=BOARD_SCHEMA, orient="row")


def _board_row(
    *,
    candidate: str,
    symbol: str,
    regime: str,
    horizon: int,
    rows: list[dict[str, Any]],
    as_of_date: date,
    created_at: datetime,
    paper_days: int,
    risk: dict[str, Any],
    readiness: dict[str, Any],
    blocked_count: int,
    shadow_count: int,
) -> dict[str, Any]:
    complete = [row for row in rows if row.get("label_status") == "complete"]
    net_values = _float_values(complete, "net_bps_after_cost")
    wins = [bool(row.get("win")) for row in complete if row.get("win") is not None]
    avg_net = _mean(net_values)
    median_net = _median(net_values)
    p25_net = _quantile(net_values, 0.25)
    win_rate = (sum(wins) / len(wins)) if wins else None
    cost_source_counts = Counter(_clean_text(row.get("cost_source")) or "MISSING" for row in rows)
    has_global_default = any(source.lower() == "global_default" for source in cost_source_counts)
    decision, reasons = _decision(
        sample_count=len(rows),
        complete_sample_count=len(complete),
        avg_net_bps=avg_net,
        win_rate=win_rate,
        p25_net_bps=p25_net,
        paper_days=paper_days,
        cost_source_counts=cost_source_counts,
    )
    ts_values = [
        ts
        for ts in (_coerce_timestamp(row.get("ts_utc")) for row in rows)
        if ts is not None
    ]
    return {
        "strategy": "v5",
        "board_schema_version": BOARD_SCHEMA_VERSION,
        "as_of_date": as_of_date.isoformat(),
        "strategy_candidate": candidate,
        "candidate_name": candidate,
        "symbol": symbol,
        "regime_state": regime,
        "horizon_hours": horizon,
        "sample_count": len(rows),
        "complete_sample_count": len(complete),
        "avg_net_bps": avg_net,
        "median_net_bps": median_net,
        "p25_net_bps": p25_net,
        "win_rate": win_rate,
        "avg_mfe_bps": _mean(_float_values(complete, "mfe_bps")),
        "avg_mae_bps": _mean(_float_values(complete, "mae_bps")),
        "cost_source_mix": _cost_source_mix_json(cost_source_counts, len(rows)),
        "stability_by_day": _stability_by_day_json(complete),
        "paper_days": paper_days,
        "cost_source_has_global_default": has_global_default,
        "decision": decision,
        "decision_reasons": safe_json_dumps(reasons),
        "risk_permission": str(risk.get("permission") or "UNKNOWN"),
        "risk_permission_status": str(risk.get("permission_status") or "UNKNOWN"),
        "enforce_readiness_status": str(readiness.get("readiness_status") or "UNKNOWN"),
        "block_reason_mix": _value_mix_json(rows, "block_reason", fallback="NONE"),
        "final_decision_mix": _value_mix_json(rows, "final_decision", fallback="UNKNOWN"),
        "high_score_blocked_outcome_count": blocked_count,
        "shadow_outcome_count": shadow_count,
        "start_ts": min(ts_values) if ts_values else None,
        "end_ts": max(ts_values) if ts_values else None,
        "created_at": created_at,
        "source": SOURCE_NAME,
    }


def _decision(
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    win_rate: float | None,
    p25_net_bps: float | None,
    paper_days: int,
    cost_source_counts: Counter[str],
) -> tuple[str, list[str]]:
    return strategy_evidence_decision_ladder(
        sample_count=sample_count,
        complete_sample_count=complete_sample_count,
        avg_net_bps=avg_net_bps,
        p25_net_bps=p25_net_bps,
        win_rate=win_rate,
        paper_days=paper_days,
        cost_source_mix=cost_source_counts,
    )


def _strategy_evidence_decision(row: dict[str, Any]) -> tuple[str, list[str]]:
    sample_count = int(_finite_float(row.get("sample_count")) or 0)
    complete_sample_count = int(_finite_float(row.get("complete_sample_count")) or 0)
    avg_net = _finite_float(row.get("avg_net_bps"))
    win_rate = _finite_float(row.get("win_rate"))
    p25 = _finite_float(row.get("p25_net_bps"))
    paper_days = int(_finite_float(row.get("paper_days")) or 0)
    return strategy_evidence_decision_ladder(
        sample_count=sample_count,
        complete_sample_count=complete_sample_count,
        avg_net_bps=avg_net,
        win_rate=win_rate,
        p25_net_bps=p25,
        paper_days=paper_days,
        cost_source_mix=row.get("cost_source_mix"),
    )


def _normalize_label_row(
    row: dict[str, Any],
    cost_sources: dict[str, str],
) -> dict[str, Any]:
    symbol = normalize_symbol(_clean_text(row.get("symbol"))) or "UNKNOWN"
    cost_source = (
        _clean_text(row.get("cost_source"))
        or cost_sources.get(symbol)
        or cost_sources.get("GLOBAL")
        or "MISSING"
    )
    return {
        "strategy_candidate": _clean_text(row.get("strategy_candidate"))
        or _clean_text(row.get("candidate_name"))
        or "UNKNOWN",
        "symbol": symbol,
        "regime_state": _clean_text(row.get("regime_state")) or "UNKNOWN",
        "horizon_hours": int(_finite_float(row.get("horizon_hours")) or 0),
        "label_status": _clean_text(row.get("label_status")) or "unknown",
        "net_bps_after_cost": _finite_float(row.get("net_bps_after_cost")),
        "mfe_bps": _finite_float(row.get("mfe_bps")),
        "mae_bps": _finite_float(row.get("mae_bps")),
        "win": row.get("win"),
        "cost_source": cost_source,
        "block_reason": _clean_text(row.get("block_reason")),
        "final_decision": _clean_text(row.get("final_decision")),
        "ts_utc": _coerce_timestamp(row.get("ts_utc")),
    }


def _paper_days_by_group(
    events: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    as_of_date: date,
) -> dict[tuple[str, str, str], int]:
    days: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    cutoff = datetime.combine(as_of_date + timedelta(days=1), time.min, tzinfo=UTC)
    for row in events.to_dicts() if not events.is_empty() else []:
        ts = _coerce_timestamp(row.get("ts_utc"))
        if ts is None or ts >= cutoff:
            continue
        final_decision = _clean_text(row.get("final_decision")).lower()
        if "paper" not in final_decision:
            continue
        key = _candidate_symbol_regime_key(row)
        if key is not None:
            days[key].add(ts.date().isoformat())
    for row in trades.to_dicts() if not trades.is_empty() else []:
        ts = _coerce_timestamp(row.get("ts_utc") or row.get("ts") or row.get("ingest_ts"))
        if ts is None or ts >= cutoff:
            continue
        text = " ".join(
            _clean_text(row.get(field)).lower()
            for field in ["mode", "trade_mode", "final_decision", "decision", "raw_payload_json"]
        )
        if "paper" not in text:
            continue
        key = _candidate_symbol_regime_key(row)
        if key is not None:
            days[key].add(ts.date().isoformat())
    return {key: len(value) for key, value in days.items()}


def _candidate_symbol_regime_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    candidate = (
        _clean_text(row.get("strategy_candidate"))
        or _clean_text(row.get("candidate_name"))
        or _clean_text(row.get("candidate"))
    )
    symbol = normalize_symbol(_clean_text(row.get("symbol"))) or ""
    if not candidate or not symbol:
        return None
    regime = _clean_text(row.get("regime_state") or row.get("regime")) or "UNKNOWN"
    return candidate, symbol, regime


def _legacy_counts(
    frame: pl.DataFrame,
    *,
    default_candidate: str,
) -> dict[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in frame.to_dicts() if not frame.is_empty() else []:
        candidate = (
            _clean_text(row.get("strategy_candidate"))
            or _clean_text(row.get("candidate_name"))
            or _clean_text(row.get("candidate"))
            or default_candidate
        )
        symbol = normalize_symbol(_clean_text(row.get("symbol"))) or "UNKNOWN"
        counts[(candidate, symbol)] += 1
    return dict(counts)


def _cost_sources_by_symbol(cost_bucket_daily: pl.DataFrame) -> dict[str, str]:
    sources: dict[str, str] = {}
    for row in cost_bucket_daily.to_dicts() if not cost_bucket_daily.is_empty() else []:
        symbol = normalize_symbol(_clean_text(row.get("symbol"))) or "GLOBAL"
        source = (
            _clean_text(row.get("cost_source"))
            or _clean_text(row.get("source"))
            or _clean_text(row.get("fallback_level"))
        )
        if source:
            sources[symbol] = source
    if "GLOBAL" in sources:
        for symbol in list(sources):
            sources.setdefault(symbol, sources["GLOBAL"])
    return sources


def _risk_context(risk_permission: pl.DataFrame) -> dict[str, Any]:
    if risk_permission.is_empty():
        return {"permission": "UNKNOWN", "permission_status": "UNKNOWN"}
    rows = risk_permission.to_dicts()
    selected = max(
        rows,
        key=lambda row: _coerce_timestamp(row.get("as_of_ts"))
        or datetime.min.replace(tzinfo=UTC),
    )
    return {
        "permission": _clean_text(selected.get("permission")) or "UNKNOWN",
        "permission_status": _clean_text(selected.get("permission_status")) or "UNKNOWN",
    }


def _enforce_readiness_context(root: Path) -> dict[str, Any]:
    frame = read_parquet_dataset(root / V5_ENFORCEMENT_DATASET)
    if frame.is_empty():
        return {"readiness_status": "UNKNOWN"}
    rows = frame.to_dicts()
    selected = max(
        rows,
        key=lambda row: _coerce_timestamp(
            row.get("created_at") or row.get("latest_bundle_ts") or row.get("date")
        )
        or datetime.min.replace(tzinfo=UTC),
    )
    status = _clean_text(selected.get("status")).upper()
    readiness_status = {
        "OK": "READY",
        "WARNING": "WARN",
        "CRITICAL": "BLOCKED",
    }.get(status, status or "UNKNOWN")
    return {
        "readiness_status": readiness_status,
        "shadow_only_recommended": readiness_status != "READY",
        "blocked_reasons": _json_list(selected.get("critical_reasons_json")),
        "warning_reasons": _json_list(selected.get("warnings_json")),
    }


def _stability_by_day_json(rows: list[dict[str, Any]]) -> str:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ts = _coerce_timestamp(row.get("ts_utc"))
        if ts is None:
            continue
        by_day[ts.date().isoformat()].append(row)
    output: list[dict[str, Any]] = []
    for day, day_rows in sorted(by_day.items()):
        net_values = _float_values(day_rows, "net_bps_after_cost")
        wins = [bool(row.get("win")) for row in day_rows if row.get("win") is not None]
        output.append(
            {
                "date": day,
                "sample_count": len(day_rows),
                "avg_net_bps": _mean(net_values),
                "win_rate": (sum(wins) / len(wins)) if wins else None,
            }
        )
    return safe_json_dumps(output)


def _cost_source_mix_json(counts: Counter[str], total: int) -> str:
    rows = [
        {
            "cost_source": source,
            "count": count,
            "ratio": (count / total) if total else 0.0,
        }
        for source, count in sorted(counts.items())
    ]
    return safe_json_dumps(rows)


def _value_mix_json(
    rows: list[dict[str, Any]],
    column: str,
    *,
    fallback: str,
) -> str:
    counts = Counter(_clean_text(row.get(column)) or fallback for row in rows)
    return safe_json_dumps(dict(sorted(counts.items())))


def _within_as_of(value: Any, cutoff: datetime) -> bool:
    ts = _coerce_timestamp(value)
    return ts is not None and ts < cutoff


def _upsert_if_not_empty(df: pl.DataFrame, dataset_path: Path, _keys: list[str]) -> int:
    if df.is_empty():
        return read_parquet_dataset(dataset_path).height
    existing = read_parquet_dataset(dataset_path)
    combined = _replace_matching_as_of_dates(existing, df)
    normalized = normalize_alpha_discovery_board_decisions(combined)
    write_parquet_dataset(normalized, dataset_path)
    return normalized.height


def _latest_strategy_evidence_as_of(
    strategy_evidence: pl.DataFrame,
    as_of_date: date,
) -> pl.DataFrame:
    if "as_of_date" not in strategy_evidence.columns:
        return strategy_evidence
    rows = strategy_evidence.to_dicts()
    dated_rows: list[tuple[date, dict[str, Any]]] = []
    for row in rows:
        row_day = _date_from_value(row.get("as_of_date"))
        if row_day is not None and row_day <= as_of_date:
            dated_rows.append((row_day, row))
    if not dated_rows:
        return pl.DataFrame(schema=strategy_evidence.schema)
    latest_day = max(row_day for row_day, _ in dated_rows)
    latest_rows = [row for row_day, row in dated_rows if row_day == latest_day]
    return pl.DataFrame(latest_rows, schema=strategy_evidence.schema, orient="row")


def _replace_matching_as_of_dates(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    if existing.is_empty():
        return incoming
    if incoming.is_empty():
        return existing
    if "as_of_date" not in existing.columns or "as_of_date" not in incoming.columns:
        return pl.concat([existing, incoming], how="diagonal_relaxed")
    dates = {
        str(value)
        for value in incoming["as_of_date"].drop_nulls().cast(pl.Utf8).unique().to_list()
        if str(value).strip()
    }
    if not dates:
        return pl.concat([existing, incoming], how="diagonal_relaxed")
    retained = existing.filter(~pl.col("as_of_date").cast(pl.Utf8).is_in(sorted(dates)))
    if retained.is_empty():
        return incoming
    return pl.concat([retained, incoming], how="diagonal_relaxed")


def _date_from_value(value: Any) -> date | None:
    ts = _coerce_timestamp(value)
    return ts.date() if ts is not None else None


def normalize_alpha_discovery_board_decisions(board: pl.DataFrame) -> pl.DataFrame:
    if board.is_empty() or "decision" not in board.columns:
        return board
    rows: list[dict[str, Any]] = []
    for row in board.to_dicts():
        candidate = _canonical_candidate_name(
            row.get("strategy_candidate") or row.get("candidate_name"),
            dataset_name="alpha_discovery_board",
        )
        if candidate:
            row["strategy_candidate"] = candidate
            row["candidate_name"] = candidate
        decision, reasons = strategy_evidence_decision_ladder(
            sample_count=int(_finite_float(row.get("sample_count")) or 0),
            complete_sample_count=int(_finite_float(row.get("complete_sample_count")) or 0),
            avg_net_bps=_finite_float(row.get("avg_net_bps")),
            p25_net_bps=_finite_float(row.get("p25_net_bps")),
            win_rate=_finite_float(row.get("win_rate")),
            paper_days=int(_finite_float(row.get("paper_days")) or 0),
            cost_source_mix=row.get("cost_source_mix"),
        )
        row["decision"] = decision
        row["decision_reasons"] = safe_json_dumps(reasons)
        rows.append(row)
    normalized = _drop_invalid_alpha_discovery_rows(
        _drop_unknown_symbol_rows(pl.DataFrame(rows, schema=board.schema, orient="row"))
    )
    keys = [
        column
        for column in [
            "strategy",
            "board_schema_version",
            "as_of_date",
            "strategy_candidate",
            "symbol",
            "regime_state",
            "horizon_hours",
        ]
        if column in normalized.columns
    ]
    return normalized.unique(subset=keys, keep="last", maintain_order=True) if keys else normalized


def _drop_unknown_symbol_rows(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "symbol" not in frame.columns:
        return frame
    symbol = pl.col("symbol").fill_null("").cast(pl.Utf8).str.to_uppercase()
    return frame.filter(symbol != "UNKNOWN")


def _drop_invalid_alpha_discovery_rows(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    required = {"strategy_candidate", "sample_count", "complete_sample_count"}
    if not required.issubset(frame.columns):
        return frame
    candidate = pl.col("strategy_candidate").fill_null("").cast(pl.Utf8)
    sample_count = pl.col("sample_count").fill_null(0).cast(pl.Int64, strict=False)
    complete_count = pl.col("complete_sample_count").fill_null(0).cast(pl.Int64, strict=False)
    invalid_alt_impulse = (
        (candidate == "v5.alt_impulse_shadow")
        & (sample_count > 10)
        & (complete_count == 0)
    )
    return frame.filter(~invalid_alt_impulse)


def _decision_counts(board: pl.DataFrame) -> dict[str, int]:
    if board.is_empty() or "decision" not in board.columns:
        return {}
    return {
        str(row["decision"]): int(row["count"])
        for row in board.group_by("decision").len(name="count").to_dicts()
    }


def _float_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    return [
        value
        for value in (_finite_float(row.get(column)) for row in rows)
        if value is not None
    ]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(math.ceil(q * len(ordered)) - 1, 0), len(ordered) - 1)
    return ordered[index]


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    rendered = str(value).strip()
    return "" if rendered.lower() in {"", "none", "null", "nan", "n/a", "na"} else rendered


def _coerce_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value and str(value).strip().lower() != "auto":
        return date.fromisoformat(str(value))
    return datetime.now(UTC).date()


def _json_list(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]
