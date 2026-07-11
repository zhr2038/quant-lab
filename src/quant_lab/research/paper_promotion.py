from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.paper.contracts import (
    PAPER_STRATEGY_CONTRACT_VERSION,
    LifecycleState,
    legacy_lifecycle_state,
)
from quant_lab.paper.cost_trust import build_strategy_cost_trust_frame
from quant_lab.paper.proposals import (
    PAPER_STRATEGY_MIGRATION_AUDIT_SCHEMA,
    build_configured_proposals,
    build_legacy_proposal_migration_audit,
)
from quant_lab.paper.service import PAPER_STRATEGY_PROPOSAL_DATASET, publish_proposals
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

PAPER_STRATEGY_REGISTRY_DATASET = Path("gold") / "paper_strategy_registry"
PAPER_STRATEGY_PROMOTION_GATE_DATASET = Path("gold") / "paper_strategy_promotion_gate"
PAPER_STRATEGY_PROPOSAL_ACK_DATASET = Path("silver") / "v5_paper_strategy_proposal_ack"
PAPER_STRATEGY_RUNS_DATASET = Path("gold") / "paper_strategy_runs"
PAPER_STRATEGY_DAILY_DATASET = Path("gold") / "paper_strategy_daily"
STRATEGY_OPPORTUNITY_ADVISORY_DATASET = Path("gold") / "strategy_opportunity_advisory"
ALPHA_DISCOVERY_BOARD_DATASET = Path("gold") / "alpha_discovery_board"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
STRATEGY_COST_TRUST_DATASET = Path("gold") / "strategy_cost_trust"
PAPER_STRATEGY_MIGRATION_AUDIT_DATASET = Path("gold") / "paper_strategy_migration_audit"

PAPER_PIPELINE_SOURCE = "research.paper_strategy_promotion.v0.1"
PAPER_PIPELINE_SCHEMA_VERSION = "paper_strategy_pipeline.v1"
MIN_PAPER_DAYS = 14
MIN_CLOSED_ENTRIES = 20
MIN_ENTRY_DAY_COUNT = 7
MIN_ARRIVAL_MID_COVERAGE = 0.8
MIN_P25_AFTER_COST_BPS = -30.0
TRUSTED_PAPER_COST_SOURCES = {"actual_fills", "mixed_actual_proxy", "bootstrap_cost_probe"}
BLOCKED_PAPER_COST_SOURCES = {
    "fallback_not_live_safe",
    "public_spread_proxy",
    "public_microstructure_proxy",
    "global_default",
    "local_estimate",
    "conservative_shadow_cost",
}

PAPER_STRATEGY_REGISTRY_SCHEMA = {
    "strategy_id": pl.Utf8,
    "strategy_version": pl.Utf8,
    "proposal_id": pl.Utf8,
    "proposal_hash": pl.Utf8,
    "paper_tracker_id": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "status": pl.Utf8,
    "lifecycle_state": pl.Utf8,
    "lifecycle_reason": pl.Utf8,
    "blocked_reasons": pl.Utf8,
    "next_required_actions": pl.Utf8,
    "contract_version": pl.Utf8,
    "paper_start_at": pl.Utf8,
    "rules_locked": pl.Boolean,
    "accepted": pl.Boolean,
    "reject_reason": pl.Utf8,
    "first_seen_at": pl.Utf8,
    "accepted_at": pl.Utf8,
    "expires_at": pl.Utf8,
    "source_pack_sha256": pl.Utf8,
    "source_v5_bundle_sha256": pl.Utf8,
    "paper_only": pl.Boolean,
    "max_live_notional_usdt": pl.Float64,
    "live_order_effect": pl.Utf8,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}

PAPER_STRATEGY_PROMOTION_GATE_SCHEMA = {
    "strategy_id": pl.Utf8,
    "strategy_version": pl.Utf8,
    "proposal_id": pl.Utf8,
    "proposal_hash": pl.Utf8,
    "paper_tracker_id": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "lifecycle_state": pl.Utf8,
    "accepted": pl.Boolean,
    "paper_tracker_created": pl.Boolean,
    "paper_tracker_effective": pl.Boolean,
    "paper_tracker_status": pl.Utf8,
    "paper_runs": pl.Int64,
    "paper_days": pl.Int64,
    "closed_entries": pl.Int64,
    "entry_day_count": pl.Int64,
    "arrival_mid_coverage": pl.Float64,
    "mean_after_cost_bps": pl.Float64,
    "median_after_cost_bps": pl.Float64,
    "p25_after_cost_bps": pl.Float64,
    "recent_7d_mean_bps": pl.Float64,
    "recent_7d_hit_rate": pl.Float64,
    "max_drawdown_bps": pl.Float64,
    "max_day_concentration": pl.Float64,
    "regime_count": pl.Int64,
    "rules_locked": pl.Boolean,
    "cost_source": pl.Utf8,
    "strategy_cost_trust_level": pl.Utf8,
    "required_cost_trust_level": pl.Utf8,
    "cost_trusted_for_paper": pl.Boolean,
    "backtest_paper_conflict": pl.Boolean,
    "promotion_score": pl.Float64,
    "promotion_confidence": pl.Utf8,
    "evidence_quality": pl.Utf8,
    "tail_risk_status": pl.Utf8,
    "regime_coverage_status": pl.Utf8,
    "cost_evidence_status": pl.Utf8,
    "paper_ready": pl.Boolean,
    "block_reason": pl.Utf8,
    "promotion_block_reasons": pl.Utf8,
    "lifecycle_reason": pl.Utf8,
    "blocked_reasons": pl.Utf8,
    "next_required_actions": pl.Utf8,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}


class PaperStrategyPipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    paper_strategy_registry: int = Field(ge=0)
    paper_strategy_promotion_gate: int = Field(ge=0)
    paper_strategy_migration_audit: int = Field(ge=0)


@dataclass(frozen=True)
class _StrategyKey:
    proposal_id: str
    paper_tracker_id: str
    strategy_candidate: str
    symbol: str


def build_and_publish_paper_strategy_pipeline(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> PaperStrategyPipelineResult:
    root = Path(lake_root)
    day = _resolve_as_of_date(as_of_date)
    advisory = read_parquet_dataset(root / STRATEGY_OPPORTUNITY_ADVISORY_DATASET)
    legacy_evidence = read_parquet_dataset(root / ALPHA_DISCOVERY_BOARD_DATASET)
    if legacy_evidence.is_empty():
        legacy_evidence = advisory
    existing_registry = read_parquet_dataset(root / PAPER_STRATEGY_REGISTRY_DATASET)
    existing_gate = read_parquet_dataset(root / PAPER_STRATEGY_PROMOTION_GATE_DATASET)
    existing_proposals = read_parquet_dataset(root / PAPER_STRATEGY_PROPOSAL_DATASET)
    existing_migration_audit = read_parquet_dataset(root / PAPER_STRATEGY_MIGRATION_AUDIT_DATASET)
    configured = build_configured_proposals(advisory)
    proposal_frame = existing_proposals if not existing_proposals.is_empty() else advisory
    if configured:
        published = publish_proposals(root, [proposal for proposal, _evidence in configured])
        proposal_frame = published
    existing_registry = _filter_superseded_structured_rows(
        existing_registry, proposal_frame
    )
    existing_gate = _filter_superseded_structured_rows(existing_gate, proposal_frame)
    migration_audit = _merge_migration_audit(
        existing_migration_audit,
        build_legacy_proposal_migration_audit(legacy_evidence, configured),
    )
    write_parquet_dataset(
        migration_audit,
        root / PAPER_STRATEGY_MIGRATION_AUDIT_DATASET,
    )
    strategy_cost_trust = build_strategy_cost_trust_frame(
        proposal_frame,
        read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET),
    )
    write_parquet_dataset(strategy_cost_trust, root / STRATEGY_COST_TRUST_DATASET)
    paper_ack = _filter_superseded_structured_rows(
        read_parquet_dataset(root / PAPER_STRATEGY_PROPOSAL_ACK_DATASET),
        proposal_frame,
    )
    paper_runs = _filter_superseded_structured_rows(
        read_parquet_dataset(root / PAPER_STRATEGY_RUNS_DATASET),
        proposal_frame,
    )
    paper_daily = _filter_superseded_structured_rows(
        read_parquet_dataset(root / PAPER_STRATEGY_DAILY_DATASET),
        proposal_frame,
    )
    frames = build_paper_strategy_pipeline_frames(
        proposals=proposal_frame,
        proposal_ack=paper_ack,
        runs=paper_runs,
        daily=paper_daily,
        strategy_cost_trust=strategy_cost_trust,
    )
    registry = _merge_lifecycle_frame(
        existing_registry,
        frames["paper_strategy_registry"],
        schema=PAPER_STRATEGY_REGISTRY_SCHEMA,
        rank=_registry_persistence_rank,
    )
    gate = _promotion_gate_frame_from_registry(
        registry,
        daily=paper_daily,
        runs=paper_runs,
        proposals=proposal_frame,
        strategy_cost_trust=strategy_cost_trust,
    )
    gate = _merge_lifecycle_frame(
        existing_gate,
        gate,
        schema=PAPER_STRATEGY_PROMOTION_GATE_SCHEMA,
        rank=_gate_persistence_rank,
    )
    write_parquet_dataset(registry, root / PAPER_STRATEGY_REGISTRY_DATASET)
    write_parquet_dataset(gate, root / PAPER_STRATEGY_PROMOTION_GATE_DATASET)
    return PaperStrategyPipelineResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        paper_strategy_registry=registry.height,
        paper_strategy_promotion_gate=gate.height,
        paper_strategy_migration_audit=migration_audit.height,
    )


def _merge_migration_audit(existing: pl.DataFrame, generated: pl.DataFrame) -> pl.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for frame in (existing, generated):
        if frame.is_empty():
            continue
        for row in frame.to_dicts():
            key = _text(row.get("legacy_row_id"))
            if key:
                rows[key] = row
    return _frame(list(rows.values()), PAPER_STRATEGY_MIGRATION_AUDIT_SCHEMA)


def _merge_lifecycle_frame(
    existing: pl.DataFrame,
    generated: pl.DataFrame,
    *,
    schema: dict[str, pl.DataType],
    rank: Any,
) -> pl.DataFrame:
    best: dict[str, dict[str, Any]] = {}
    for frame in (existing, generated):
        if frame.is_empty():
            continue
        for source in frame.to_dicts():
            row = {column: source.get(column) for column in schema}
            key = _persistent_lifecycle_key(row)
            if not key:
                continue
            current = best.get(key)
            if current is None:
                best[key] = row
                continue
            if rank(row) >= rank(current):
                best[key] = _fill_blank_lifecycle_values(row, current, schema)
            else:
                best[key] = _fill_blank_lifecycle_values(current, row, schema)
    return _frame(list(best.values()), schema)


def _promotion_gate_frame_from_registry(
    registry: pl.DataFrame,
    *,
    daily: pl.DataFrame,
    runs: pl.DataFrame,
    proposals: pl.DataFrame,
    strategy_cost_trust: pl.DataFrame,
) -> pl.DataFrame:
    created_at = datetime.now(UTC).isoformat()
    daily_rows = _latest_rows_by_strategy(daily)
    run_rows = runs.to_dicts() if not runs.is_empty() else []
    proposal_rows = _proposal_rows(proposals)
    cost_trust_rows = strategy_cost_trust.to_dicts() if not strategy_cost_trust.is_empty() else []
    rows: list[dict[str, Any]] = []
    for registry_row in registry.to_dicts() if not registry.is_empty() else []:
        proposal_id, paper_tracker_id, strategy_candidate, symbol = _row_key_tuple(registry_row)
        key = _StrategyKey(
            proposal_id=proposal_id,
            paper_tracker_id=paper_tracker_id,
            strategy_candidate=strategy_candidate,
            symbol=symbol,
        )
        rows.append(
            _promotion_gate_row(
                registry=registry_row,
                daily=_match_strategy_row(daily_rows, key),
                runs=[row for row in run_rows if _row_matches_key(row, key)],
                proposal=_match_strategy_row(proposal_rows, key),
                strategy_cost_trust=_match_cost_trust_row(cost_trust_rows, registry_row),
                created_at=created_at,
            )
        )
    return _frame(rows, PAPER_STRATEGY_PROMOTION_GATE_SCHEMA)


def _persistent_lifecycle_key(row: Mapping[str, Any]) -> str:
    return (
        _text(row.get("proposal_id"))
        or _text(row.get("paper_tracker_id"))
        or "|".join(
            part
            for part in (
                _text(row.get("strategy_id")),
                _symbol(row),
                _text(row.get("strategy_candidate")),
            )
            if part
        )
    )


def _fill_blank_lifecycle_values(
    primary: dict[str, Any],
    fallback: Mapping[str, Any],
    schema: Mapping[str, pl.DataType],
) -> dict[str, Any]:
    output = dict(primary)
    for column in schema:
        if column == "reject_reason" and _bool(primary.get("accepted")) is True:
            output[column] = ""
            continue
        if output.get(column) in (None, "") and fallback.get(column) not in (None, ""):
            output[column] = fallback.get(column)
    return output


def _filter_superseded_structured_rows(
    frame: pl.DataFrame,
    proposals: pl.DataFrame,
) -> pl.DataFrame:
    if frame.is_empty() or proposals.is_empty():
        return frame
    active: dict[tuple[str, str], str] = {}
    for row in proposals.to_dicts():
        strategy_id = _text(row.get("strategy_id"))
        strategy_version = _text(row.get("strategy_version"))
        proposal_id = _text(row.get("proposal_id"))
        if strategy_id and strategy_version and proposal_id:
            active[(strategy_id, strategy_version)] = proposal_id
    if not active:
        return frame
    rows = []
    for row in frame.to_dicts():
        proposal_id = _text(row.get("proposal_id"))
        identity = _structured_proposal_identity(proposal_id)
        if identity in active and proposal_id != active[identity]:
            continue
        rows.append(row)
    if not rows:
        return frame.head(0)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .cast(frame.schema, strict=False)
        .select(frame.columns)
    )


def _structured_proposal_identity(proposal_id: str) -> tuple[str, str] | None:
    parts = proposal_id.rsplit(":", 2)
    if len(parts) != 3 or len(parts[2]) != 12:
        return None
    if any(char not in "0123456789abcdef" for char in parts[2].lower()):
        return None
    return parts[0], parts[1]


def _registry_persistence_rank(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    rejected = (
        bool(_text(row.get("reject_reason"))) or "REJECTED" in _text(row.get("status")).upper()
    )
    accepted = _bool(row.get("accepted")) is True
    tracker = bool(_text(row.get("paper_tracker_id")))
    return int(rejected or accepted), int(tracker), int(accepted), _row_sort_ts(row)


def _gate_persistence_rank(row: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        _int(row.get("paper_runs")) or 0,
        _int(row.get("paper_days")) or 0,
        _int(row.get("closed_entries")) or 0,
        int(_bool(row.get("accepted")) is True),
        _row_sort_ts(row),
    )


def build_paper_strategy_pipeline_frames(
    *,
    proposals: pl.DataFrame | None = None,
    proposal_ack: pl.DataFrame | None = None,
    runs: pl.DataFrame | None = None,
    daily: pl.DataFrame | None = None,
    strategy_cost_trust: pl.DataFrame | None = None,
    created_at: datetime | None = None,
) -> dict[str, pl.DataFrame]:
    created = (created_at or datetime.now(UTC)).isoformat()
    proposal_frame = proposals if proposals is not None else pl.DataFrame()
    ack_frame = proposal_ack if proposal_ack is not None else pl.DataFrame()
    run_frame = runs if runs is not None else pl.DataFrame()
    daily_frame = daily if daily is not None else pl.DataFrame()
    ack_frame = _filter_superseded_structured_rows(ack_frame, proposal_frame)
    run_frame = _filter_superseded_structured_rows(run_frame, proposal_frame)
    daily_frame = _filter_superseded_structured_rows(daily_frame, proposal_frame)
    proposal_rows = _proposal_rows(proposal_frame)
    ack_rows = _latest_rows_by_strategy(
        ack_frame
    )
    run_rows = run_frame.to_dicts()
    daily_rows = _latest_rows_by_strategy(daily_frame)
    cost_trust_rows = (
        strategy_cost_trust.to_dicts()
        if strategy_cost_trust is not None and not strategy_cost_trust.is_empty()
        else []
    )

    keys = _collect_strategy_keys(proposal_rows, ack_rows, run_rows, daily_rows)
    registry_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for key in sorted(
        keys,
        key=lambda item: (item.symbol, item.proposal_id, item.paper_tracker_id),
    ):
        proposal = _match_strategy_row(proposal_rows, key)
        ack = _match_strategy_row(ack_rows, key)
        daily_row = _match_strategy_row(daily_rows, key)
        matched_runs = [row for row in run_rows if _row_matches_key(row, key)]
        registry = _registry_row(
            key=key,
            proposal=proposal,
            ack=ack,
            daily=daily_row,
            runs=matched_runs,
            created_at=created,
        )
        gate = _promotion_gate_row(
            registry=registry,
            daily=daily_row,
            runs=matched_runs,
            proposal=proposal,
            strategy_cost_trust=_match_cost_trust_row(cost_trust_rows, registry),
            created_at=created,
        )
        registry_rows.append(registry)
        gate_rows.append(gate)
    return {
        "paper_strategy_registry": _frame(registry_rows, PAPER_STRATEGY_REGISTRY_SCHEMA),
        "paper_strategy_promotion_gate": _frame(
            gate_rows,
            PAPER_STRATEGY_PROMOTION_GATE_SCHEMA,
        ),
    }


def _proposal_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    rows = []
    for row in frame.to_dicts():
        recommended = _text(row.get("recommended_mode")).lower()
        decision = _text(row.get("decision")).upper()
        proposal_id = _text(row.get("proposal_id")) or _text(row.get("strategy_id"))
        if not proposal_id and decision not in {"PAPER_READY", "LIVE_SMALL_READY"}:
            continue
        if recommended not in {"", "paper", "paper_only"} and decision != "PAPER_READY":
            continue
        rows.append(row)
    return rows


def _latest_rows_by_strategy(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in frame.to_dicts():
        key = _row_key_tuple(row)
        if key == ("", "", "", ""):
            continue
        current = best.get(key)
        if current is None or _row_sort_ts(row) >= _row_sort_ts(current):
            best[key] = row
    return list(best.values())


def _collect_strategy_keys(
    proposals: list[dict[str, Any]],
    ack_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    daily_rows: list[dict[str, Any]],
) -> set[_StrategyKey]:
    by_proposal: dict[str, _StrategyKey] = {}
    keys_without_proposal: set[_StrategyKey] = set()
    for row in [*proposals, *ack_rows, *run_rows, *daily_rows]:
        proposal_id, paper_tracker_id, strategy_candidate, symbol = _row_key_tuple(row)
        if not any([proposal_id, paper_tracker_id, strategy_candidate, symbol]):
            continue
        key = _StrategyKey(
            proposal_id=proposal_id,
            paper_tracker_id=paper_tracker_id,
            strategy_candidate=strategy_candidate,
            symbol=symbol,
        )
        if not proposal_id:
            keys_without_proposal.add(key)
            continue
        current = by_proposal.get(proposal_id)
        if current is None:
            by_proposal[proposal_id] = key
            continue
        by_proposal[proposal_id] = _StrategyKey(
            proposal_id=proposal_id,
            paper_tracker_id=current.paper_tracker_id or key.paper_tracker_id,
            strategy_candidate=current.strategy_candidate or key.strategy_candidate,
            symbol=current.symbol or key.symbol,
        )
    return {*by_proposal.values(), *keys_without_proposal}


def _registry_row(
    *,
    key: _StrategyKey,
    proposal: Mapping[str, Any],
    ack: Mapping[str, Any],
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    accepted = _bool(ack.get("accepted"))
    reject_reason = "" if accepted is True else _text(ack.get("reject_reason"))
    proposal_id = (
        key.proposal_id or _text(ack.get("proposal_id")) or _text(proposal.get("proposal_id"))
    )
    explicit_paper_tracker_id = (
        key.paper_tracker_id
        or _text(ack.get("paper_tracker_id"))
        or _text(daily.get("paper_tracker_id"))
    )
    paper_tracker_id = explicit_paper_tracker_id or (
        proposal_id if accepted and (daily or runs) else ""
    )
    strategy_id = (
        _text(proposal.get("strategy_id"))
        or paper_tracker_id
        or proposal_id
        or _text(daily.get("strategy_id"))
    )
    paper_only = _paper_only(ack, proposal)
    max_live_notional = _float(
        ack.get("max_live_notional_usdt")
        or ack.get("advisory_max_live_notional_usdt")
        or proposal.get("max_live_notional_usdt")
    )
    if max_live_notional is None and paper_only:
        max_live_notional = 0.0
    status = _registry_status(
        accepted=accepted,
        reject_reason=reject_reason,
        paper_tracker_id=paper_tracker_id,
        daily=daily,
        runs=runs,
    )
    lifecycle = legacy_lifecycle_state(
        status,
        accepted=accepted,
        tracker_effective=bool(accepted and paper_tracker_id),
    )
    lifecycle_blocked = _registry_block_reasons(
        accepted=accepted,
        reject_reason=reject_reason,
        paper_tracker_id=paper_tracker_id,
    )
    return {
        "strategy_id": strategy_id,
        "strategy_version": _text(ack.get("strategy_version"))
        or _text(proposal.get("strategy_version"))
        or _strategy_version(strategy_id or proposal_id),
        "proposal_id": proposal_id,
        "proposal_hash": _text(ack.get("proposal_hash"))
        or _text(proposal.get("proposal_hash"))
        or _proposal_hash(proposal),
        "paper_tracker_id": paper_tracker_id,
        "symbol": key.symbol or _symbol(ack) or _symbol(proposal) or _symbol(daily),
        "strategy_candidate": (
            key.strategy_candidate
            or _text(ack.get("strategy_candidate"))
            or _text(proposal.get("strategy_candidate"))
            or _text(daily.get("strategy_candidate"))
        ),
        "status": status,
        "lifecycle_state": lifecycle.value,
        "lifecycle_reason": _lifecycle_reason(lifecycle, lifecycle_blocked),
        "blocked_reasons": safe_json_dumps(lifecycle_blocked),
        "next_required_actions": safe_json_dumps(
            _next_required_actions(lifecycle, lifecycle_blocked)
        ),
        "contract_version": _text(proposal.get("contract_version"))
        or _text(ack.get("contract_version"))
        or PAPER_STRATEGY_CONTRACT_VERSION,
        "paper_start_at": _paper_start_at(ack=ack, daily=daily, runs=runs),
        "rules_locked": bool(accepted and paper_only and paper_tracker_id),
        "accepted": accepted,
        "reject_reason": reject_reason,
        "first_seen_at": _first_seen_at(proposal=proposal, ack=ack, runs=runs, daily=daily),
        "accepted_at": _text(ack.get("accepted_at")) or _text(ack.get("ingest_ts")),
        "expires_at": _text(ack.get("expires_at")) or _text(proposal.get("expires_at")),
        "source_pack_sha256": _text(ack.get("source_pack_sha256"))
        or _text(proposal.get("source_pack_sha256")),
        "source_v5_bundle_sha256": _text(ack.get("source_v5_bundle_sha256"))
        or _text(ack.get("bundle_sha256")),
        "paper_only": paper_only,
        "max_live_notional_usdt": max_live_notional,
        "live_order_effect": _text(ack.get("live_order_effect"))
        or _text(proposal.get("live_order_effect"))
        or "paper_only_no_live_order",
        "created_at": created_at,
        "source": PAPER_PIPELINE_SOURCE,
        "schema_version": PAPER_PIPELINE_SCHEMA_VERSION,
    }


def _promotion_gate_row(
    *,
    registry: Mapping[str, Any],
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
    proposal: Mapping[str, Any],
    strategy_cost_trust: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    values = _paper_pnl_values(runs)
    recent_values = _recent_paper_pnl_values(runs)
    paper_days = _int(daily.get("paper_days")) or _int(daily.get("heartbeat_day_count")) or 0
    closed_entries = (
        _int(daily.get("paper_pnl_observed_count"))
        or _int(daily.get("cumulative_paper_pnl_observed_count"))
        or len(values)
    )
    entry_day_count = _int(daily.get("entry_day_count")) or 0
    arrival_mid_coverage = _float(daily.get("arrival_mid_coverage")) or 0.0
    mean_bps = _float(daily.get("avg_paper_pnl_bps")) or _mean(values)
    median_bps = _median(values)
    p25_bps = _percentile(values, 0.25)
    recent_mean = _mean(recent_values)
    recent_hit_rate = _hit_rate(recent_values)
    max_drawdown_bps = _max_drawdown(values)
    max_day_concentration = _max_day_concentration(runs)
    regimes = _paper_regimes(runs)
    single_regime_strategy = _bool(daily.get("single_regime_strategy")) is True
    tail_risk_status = _tail_risk_status(values)
    regime_coverage_status = (
        "PASS" if len(regimes) >= 2 or single_regime_strategy else "INSUFFICIENT"
    )
    cost_sources = _cost_sources(daily.get("cost_source_mix"))
    cost_trusted = bool(cost_sources & TRUSTED_PAPER_COST_SOURCES) and not bool(
        cost_sources & BLOCKED_PAPER_COST_SOURCES
    )
    accepted = bool(registry.get("accepted"))
    reject_reason = _text(registry.get("reject_reason"))
    paper_tracker_created = bool(_text(registry.get("paper_tracker_id")))
    paper_tracker_effective = accepted and paper_tracker_created
    rules_locked = bool(registry.get("rules_locked"))
    backtest_paper_conflict = _bool(daily.get("backtest_paper_conflict")) is True
    paper_tracker_status = _paper_tracker_status(
        accepted=accepted,
        reject_reason=reject_reason,
        paper_tracker_created=paper_tracker_created,
    )
    block_reasons = _gate_block_reasons(
        accepted=accepted,
        reject_reason=reject_reason,
        paper_tracker_created=paper_tracker_created,
        paper_days=paper_days,
        closed_entries=closed_entries,
        entry_day_count=entry_day_count,
        arrival_mid_coverage=arrival_mid_coverage,
        mean_bps=mean_bps,
        median_bps=median_bps,
        p25_bps=p25_bps,
        recent_mean=recent_mean,
        cost_trusted=cost_trusted,
        max_drawdown_bps=max_drawdown_bps,
        max_drawdown_budget_bps=_float(daily.get("max_drawdown_budget_bps")) or 500.0,
        max_day_concentration=max_day_concentration,
        tail_risk_status=tail_risk_status,
        regime_coverage_status=regime_coverage_status,
        rules_locked=rules_locked,
        backtest_paper_conflict=backtest_paper_conflict,
        daily_block_reasons=_json_list(daily.get("live_block_reason")),
    )
    required_cost_trust_level = _text(proposal.get("required_cost_trust_level")) or "PAPER_ONLY"
    strategy_cost_trust_level = (
        _text(strategy_cost_trust.get("cost_trust_level")) or "NOT_EVALUATED"
    )
    dimensional_cost_trusted = True
    if strategy_cost_trust:
        dimensional_cost_trusted = _cost_trust_rank(strategy_cost_trust_level) >= (
            _cost_trust_rank(required_cost_trust_level)
        )
        if not dimensional_cost_trusted:
            block_reasons = sorted(
                {
                    *block_reasons,
                    "strategy_cost_trust_below_required:"
                    f"actual={strategy_cost_trust_level};"
                    f"required={required_cost_trust_level}",
                }
            )
    effective_cost_trusted = cost_trusted and dimensional_cost_trusted
    paper_ready = not block_reasons
    lifecycle = (
        LifecycleState.PAPER_PROMOTION_READY
        if paper_ready
        else (
            LifecycleState.PAPER_EVIDENCE_INSUFFICIENT
            if paper_tracker_effective
            else legacy_lifecycle_state(registry.get("lifecycle_state"))
        )
    )
    promotion_score = _promotion_score(block_reasons)
    promotion_confidence = _promotion_confidence(
        score=promotion_score,
        closed_entries=closed_entries,
        paper_days=paper_days,
    )
    cost_evidence_status = (
        "PASS"
        if effective_cost_trusted
        else (
            f"{strategy_cost_trust_level}_REQUIRED_{required_cost_trust_level}"
            if strategy_cost_trust
            else "INSUFFICIENT"
        )
    )
    return {
        "strategy_id": _text(registry.get("strategy_id")),
        "strategy_version": _text(registry.get("strategy_version")),
        "proposal_id": _text(registry.get("proposal_id")),
        "proposal_hash": _text(registry.get("proposal_hash")),
        "paper_tracker_id": _text(registry.get("paper_tracker_id")),
        "symbol": _text(registry.get("symbol")),
        "strategy_candidate": _text(registry.get("strategy_candidate")),
        "lifecycle_state": lifecycle.value,
        "accepted": accepted,
        "paper_tracker_created": paper_tracker_created,
        "paper_tracker_effective": paper_tracker_effective,
        "paper_tracker_status": paper_tracker_status,
        "paper_runs": len(runs),
        "paper_days": paper_days,
        "closed_entries": closed_entries,
        "entry_day_count": entry_day_count,
        "arrival_mid_coverage": arrival_mid_coverage,
        "mean_after_cost_bps": mean_bps,
        "median_after_cost_bps": median_bps,
        "p25_after_cost_bps": p25_bps,
        "recent_7d_mean_bps": recent_mean,
        "recent_7d_hit_rate": recent_hit_rate,
        "max_drawdown_bps": max_drawdown_bps,
        "max_day_concentration": max_day_concentration,
        "regime_count": len(regimes),
        "rules_locked": rules_locked,
        "cost_source": safe_json_dumps(sorted(cost_sources)),
        "strategy_cost_trust_level": strategy_cost_trust_level,
        "required_cost_trust_level": required_cost_trust_level,
        "cost_trusted_for_paper": effective_cost_trusted,
        "backtest_paper_conflict": backtest_paper_conflict,
        "promotion_score": promotion_score,
        "promotion_confidence": promotion_confidence,
        "evidence_quality": _evidence_quality(
            closed_entries=closed_entries,
            arrival_mid_coverage=arrival_mid_coverage,
            cost_trusted=effective_cost_trusted,
        ),
        "tail_risk_status": tail_risk_status,
        "regime_coverage_status": regime_coverage_status,
        "cost_evidence_status": cost_evidence_status,
        "paper_ready": paper_ready,
        "block_reason": safe_json_dumps(block_reasons),
        "promotion_block_reasons": safe_json_dumps(block_reasons),
        "lifecycle_reason": _lifecycle_reason(lifecycle, block_reasons),
        "blocked_reasons": safe_json_dumps(block_reasons),
        "next_required_actions": safe_json_dumps(_next_required_actions(lifecycle, block_reasons)),
        "created_at": created_at,
        "source": PAPER_PIPELINE_SOURCE,
        "schema_version": PAPER_PIPELINE_SCHEMA_VERSION,
    }


def _gate_block_reasons(
    *,
    accepted: bool,
    reject_reason: str,
    paper_tracker_created: bool,
    paper_days: int,
    closed_entries: int,
    entry_day_count: int,
    arrival_mid_coverage: float,
    mean_bps: float | None,
    median_bps: float | None,
    p25_bps: float | None,
    recent_mean: float | None,
    cost_trusted: bool,
    max_drawdown_bps: float,
    max_drawdown_budget_bps: float,
    max_day_concentration: float,
    tail_risk_status: str,
    regime_coverage_status: str,
    rules_locked: bool,
    backtest_paper_conflict: bool,
    daily_block_reasons: list[str],
) -> list[str]:
    reasons: set[str] = set()
    if not accepted:
        reasons.add("proposal_not_acked")
    if reject_reason:
        reasons.add(f"v5_rejected:{reject_reason}")
    if paper_tracker_created and not accepted:
        reasons.add("paper_tracker_not_effective_without_ack")
    if not paper_tracker_created:
        reasons.add("paper_tracker_missing")
    if paper_days < MIN_PAPER_DAYS:
        reasons.add("insufficient_paper_days")
    if closed_entries < MIN_CLOSED_ENTRIES:
        reasons.add("insufficient_closed_entries")
    if entry_day_count < MIN_ENTRY_DAY_COUNT:
        reasons.add("insufficient_entry_day_count")
    if arrival_mid_coverage < MIN_ARRIVAL_MID_COVERAGE:
        reasons.add("insufficient_arrival_mid_coverage")
    if mean_bps is None or mean_bps <= 0.0:
        reasons.add("mean_after_cost_not_positive")
    if median_bps is None or median_bps <= 0.0:
        reasons.add("median_after_cost_not_positive")
    if p25_bps is None or p25_bps < MIN_P25_AFTER_COST_BPS:
        reasons.add("p25_after_cost_too_weak")
    if recent_mean is None or recent_mean < 0.0:
        reasons.add("recent_7d_mean_negative_or_missing")
    if not cost_trusted:
        reasons.add("cost_not_trusted_for_paper")
    if max_drawdown_bps > max_drawdown_budget_bps:
        reasons.add("max_drawdown_exceeds_budget")
    if max_day_concentration > 0.5:
        reasons.add("single_day_concentration_too_high")
    if tail_risk_status != "PASS":
        reasons.add("right_tail_dependence")
    if regime_coverage_status != "PASS":
        reasons.add("insufficient_regime_coverage")
    if not rules_locked:
        reasons.add("paper_rules_not_locked")
    if backtest_paper_conflict:
        reasons.add("unresolved_backtest_paper_conflict")
    reasons.update(_paper_block_reason_subset(daily_block_reasons))
    return sorted(reasons)


def _registry_block_reasons(
    *,
    accepted: bool,
    reject_reason: str,
    paper_tracker_id: str,
) -> list[str]:
    reasons: list[str] = []
    if not accepted:
        reasons.append("v5_ack_required")
    if reject_reason:
        reasons.append(f"v5_rejected:{reject_reason}")
    if accepted and not paper_tracker_id:
        reasons.append("paper_tracker_missing")
    return reasons


def _lifecycle_reason(state: LifecycleState, blocked_reasons: list[str]) -> str:
    if blocked_reasons:
        return blocked_reasons[0]
    reasons = {
        LifecycleState.PAPER_PROPOSAL_READY: "structured_proposal_ready",
        LifecycleState.PAPER_ACK_PENDING: "waiting_for_v5_ack",
        LifecycleState.PAPER_TRACKER_ACTIVE: "v5_paper_tracker_active",
        LifecycleState.PAPER_EVIDENCE_INSUFFICIENT: "paper_tracker_active_evidence_pending",
        LifecycleState.PAPER_PROMOTION_READY: "paper_promotion_gates_passed",
    }
    return reasons.get(state, state.value.lower())


def _next_required_actions(state: LifecycleState, blocked_reasons: list[str]) -> list[str]:
    actions: list[str] = []
    mapping = {
        "v5_ack_required": "sync_proposal_to_v5",
        "paper_tracker_missing": "create_v5_paper_tracker",
        "insufficient_paper_days": "continue_paper_tracking",
        "insufficient_closed_entries": "collect_more_closed_paper_trades",
        "insufficient_arrival_mid_coverage": "improve_top_of_book_coverage",
        "cost_not_trusted_for_paper": "collect_strategy_dimensional_cost_evidence",
        "paper_rules_not_locked": "lock_acked_rule_version",
        "insufficient_regime_coverage": "collect_additional_regime_evidence",
    }
    for reason in blocked_reasons:
        base = reason.split(":", 1)[0]
        action = mapping.get(base)
        if action and action not in actions:
            actions.append(action)
    if not actions and state == LifecycleState.PAPER_PROMOTION_READY:
        actions.append("manual_canary_review")
    return actions


def _paper_tracker_status(
    *,
    accepted: bool,
    reject_reason: str,
    paper_tracker_created: bool,
) -> str:
    if accepted and paper_tracker_created:
        return "EFFECTIVE"
    if reject_reason and paper_tracker_created:
        return "REJECTED_BY_V5"
    if paper_tracker_created:
        return "AWAITING_ACK"
    if accepted:
        return "ACKED_TRACKER_MISSING"
    if reject_reason:
        return "REJECTED_TRACKER_MISSING"
    return "MISSING"


def _paper_block_reason_subset(reasons: list[str]) -> set[str]:
    ignored = {
        "no_live_slippage_coverage",
        "cost_source_not_actual_or_mixed",
        "paper_active_but_no_entries_yet",
    }
    return {reason for reason in reasons if reason and reason not in ignored}


def _registry_status(
    *,
    accepted: bool,
    reject_reason: str,
    paper_tracker_id: str,
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
) -> str:
    if reject_reason and not accepted:
        return "REJECTED_BY_V5"
    if not accepted:
        return "PROPOSED_AWAITING_ACK"
    if not paper_tracker_id:
        return "ACKED_TRACKER_MISSING"
    if bool(daily.get("live_eligible")):
        return "PAPER_REVIEW"
    if daily or runs:
        return "PAPER_TRACKING"
    return "ACKED"


def _match_strategy_row(rows: list[Mapping[str, Any]], key: _StrategyKey) -> Mapping[str, Any]:
    matches = [row for row in rows if _row_matches_key(row, key)]
    if not matches:
        return {}
    return max(
        matches,
        key=lambda row: (
            int(_bool(row.get("accepted")) is True),
            int(bool(_text(row.get("paper_tracker_id")))),
            int(_bool(row.get("rules_locked")) is True),
            int(not bool(_text(row.get("reject_reason")))),
            _row_sort_ts(row),
        ),
    )


def _match_cost_trust_row(
    rows: list[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    aliases = {
        _text(registry.get("strategy_id")),
        _text(registry.get("proposal_id")),
    }
    matches = [row for row in rows if _text(row.get("strategy_id")) in aliases - {""}]
    if not matches:
        return {}
    return max(matches, key=_row_sort_ts)


def _cost_trust_rank(value: Any) -> int:
    return {
        "BLOCK": 0,
        "PAPER_ONLY": 1,
        "CANARY": 2,
        "SCALE_READY": 3,
    }.get(_text(value).upper(), -1)


def _row_matches_key(row: Mapping[str, Any], key: _StrategyKey) -> bool:
    aliases = {
        _text(row.get("proposal_id")),
        _text(row.get("paper_tracker_id")),
        _text(row.get("strategy_id")),
    }
    key_aliases = {key.proposal_id, key.paper_tracker_id}
    if aliases & key_aliases - {""}:
        return True
    return (
        bool(key.strategy_candidate)
        and bool(key.symbol)
        and _text(row.get("strategy_candidate")) == key.strategy_candidate
        and _symbol(row) == key.symbol
    )


def _row_key_tuple(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    proposal_id = _text(row.get("proposal_id")) or _text(row.get("strategy_id"))
    paper_tracker_id = _text(row.get("paper_tracker_id"))
    structured_proposal = bool(
        _text(row.get("proposal_hash"))
        and row.get("entry_rule") not in (None, "")
        and row.get("exit_rule") not in (None, "")
    )
    if not paper_tracker_id and not structured_proposal:
        strategy_id = _text(row.get("strategy_id"))
        if proposal_id != strategy_id:
            paper_tracker_id = strategy_id
    return (
        proposal_id,
        paper_tracker_id,
        _text(row.get("strategy_candidate")),
        _symbol(row),
    )


def _paper_start_at(
    *,
    ack: Mapping[str, Any],
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
) -> str:
    run_times = [_row_sort_ts(row) for row in runs if _row_sort_ts(row)]
    if run_times:
        return min(run_times)
    return (
        _text(daily.get("paper_start_at"))
        or _text(ack.get("accepted_at"))
        or _text(ack.get("ingest_ts"))
    )


def _first_seen_at(
    *,
    proposal: Mapping[str, Any],
    ack: Mapping[str, Any],
    runs: list[dict[str, Any]],
    daily: Mapping[str, Any],
) -> str:
    values = [
        _text(proposal.get("created_at")),
        _text(proposal.get("generated_at")),
        _text(ack.get("first_seen_at")),
        _text(ack.get("ingest_ts")),
        _text(daily.get("created_at")),
        *[_row_sort_ts(row) for row in runs],
    ]
    values = [value for value in values if value]
    return min(values) if values else ""


def _paper_only(ack: Mapping[str, Any], proposal: Mapping[str, Any]) -> bool:
    live_effect = _text(ack.get("live_order_effect")).lower()
    mode = _text(ack.get("recommended_mode") or proposal.get("recommended_mode")).lower()
    max_notional = _float(
        ack.get("max_live_notional_usdt")
        or ack.get("advisory_max_live_notional_usdt")
        or proposal.get("max_live_notional_usdt")
    )
    return (
        "paper" in live_effect
        or mode in {"paper", "paper_only"}
        or max_notional == 0.0
        or max_notional is None
    )


def _paper_pnl_values(rows: list[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for key in (
            "paper_pnl_bps",
            "paper_pnl_bps_4h",
            "paper_pnl_bps_8h",
            "paper_pnl_bps_12h",
            "paper_pnl_bps_24h",
            "paper_pnl_bps_48h",
            "paper_pnl_bps_72h",
        ):
            value = _float(row.get(key))
            if value is not None:
                values.append(value)
    return values


def _recent_paper_pnl_values(rows: list[Mapping[str, Any]]) -> list[float]:
    dated: list[tuple[date, float]] = []
    for row in rows:
        row_date = _row_date(row)
        if row_date is None:
            continue
        for value in _paper_pnl_values([row]):
            dated.append((row_date, value))
    if not dated:
        return []
    latest = max(day for day, _value in dated)
    cutoff = latest - timedelta(days=6)
    return [value for day, value in dated if day >= cutoff]


def _max_drawdown(values: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    maximum = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _max_day_concentration(rows: list[Mapping[str, Any]]) -> float:
    by_day: dict[date, float] = {}
    for row in rows:
        day = _row_date(row)
        if day is None:
            continue
        by_day[day] = by_day.get(day, 0.0) + sum(abs(value) for value in _paper_pnl_values([row]))
    total = sum(by_day.values())
    return max(by_day.values(), default=0.0) / total if total > 0 else 0.0


def _paper_regimes(rows: list[Mapping[str, Any]]) -> set[str]:
    return {
        _text(row.get("market_regime") or row.get("regime") or row.get("regime_state"))
        for row in rows
        if _text(row.get("market_regime") or row.get("regime") or row.get("regime_state"))
    }


def _tail_risk_status(values: list[float]) -> str:
    positive = sorted((value for value in values if value > 0.0), reverse=True)
    if len(values) < MIN_CLOSED_ENTRIES or not positive:
        return "INSUFFICIENT"
    total = sum(positive)
    return "PASS" if total > 0 and positive[0] / total <= 0.5 else "CONCENTRATED"


def _promotion_score(block_reasons: list[str]) -> float:
    required_gates = 16
    return max(0.0, min(1.0, (required_gates - len(set(block_reasons))) / required_gates))


def _promotion_confidence(*, score: float, closed_entries: int, paper_days: int) -> str:
    if score >= 1.0 and closed_entries >= 30 and paper_days >= 21:
        return "HIGH"
    if score >= 0.8 and closed_entries >= MIN_CLOSED_ENTRIES:
        return "MEDIUM"
    return "LOW"


def _evidence_quality(
    *,
    closed_entries: int,
    arrival_mid_coverage: float,
    cost_trusted: bool,
) -> str:
    if closed_entries >= 30 and arrival_mid_coverage >= 0.95 and cost_trusted:
        return "HIGH"
    if closed_entries >= MIN_CLOSED_ENTRIES and arrival_mid_coverage >= 0.8 and cost_trusted:
        return "MEDIUM"
    return "LOW"


def _row_date(row: Mapping[str, Any]) -> date | None:
    text = _text(row.get("as_of_date")) or _text(row.get("ts_utc"))[:10]
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _cost_sources(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {
                part.strip().lower() for part in text.replace(";", ",").split(",") if part.strip()
            }
        items = parsed if isinstance(parsed, list) else [parsed]
    sources = set()
    for item in items:
        if isinstance(item, dict):
            source = item.get("cost_source") or item.get("source")
        else:
            source = item
        text = str(source or "").strip().lower()
        if text:
            sources.add(text)
    return sources


def _json_list(value: Any) -> list[str]:
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


def _proposal_hash(row: Mapping[str, Any]) -> str:
    if not row:
        return ""
    payload = {
        key: row.get(key)
        for key in sorted(
            {
                "proposal_id",
                "strategy_candidate",
                "symbol",
                "entry_conditions",
                "recommended_mode",
                "suggested_horizon",
                "cost_source_mix",
                "required_paper_days",
                "required_slippage_coverage",
            }
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _strategy_version(strategy_id: str) -> str:
    if not strategy_id:
        return ""
    marker = "_V"
    if marker not in strategy_id:
        return "V1"
    suffix = strategy_id.rsplit(marker, 1)[-1]
    return f"V{suffix}" if suffix else "V1"


def _resolve_as_of_date(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value and str(value).lower() != "auto":
        return date.fromisoformat(str(value)[:10])
    return datetime.now(UTC).date()


def _row_sort_ts(row: Mapping[str, Any]) -> str:
    for key in (
        "accepted_at",
        "created_at",
        "generated_at",
        "as_of_ts",
        "ts_utc",
        "as_of_date",
        "ingest_ts",
        "bundle_ts",
    ):
        value = _text(row.get(key))
        if value:
            return value
    return ""


def _symbol(row: Mapping[str, Any]) -> str:
    value = _text(row.get("symbol") or row.get("v5_symbol") or row.get("normalized_symbol"))
    return normalize_symbol(value) if value else ""


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan", "nat"} else text


def _float(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    parsed = _float(value)
    return None if parsed is None else int(parsed)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    return text in {"1", "true", "yes", "y", "accepted"}


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * q)))
    return ordered[index]


def _hit_rate(values: list[float]) -> float | None:
    return None if not values else sum(1 for value in values if value > 0.0) / len(values)


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row")
