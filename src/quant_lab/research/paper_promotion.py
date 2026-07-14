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
from quant_lab.paper.service import (
    PAPER_STRATEGY_PROPOSAL_DATASET,
    publish_canonical_proposal_snapshot,
    publish_proposals,
)
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

PAPER_STRATEGY_REGISTRY_DATASET = Path("gold") / "paper_strategy_registry"
PAPER_STRATEGY_TRACKERS_CURRENT_DATASET = (
    Path("gold") / "paper_strategy_trackers_current"
)
PAPER_STRATEGY_REGISTRY_CURRENT_DATASET = (
    Path("gold") / "paper_strategy_registry_current"
)
PAPER_STRATEGY_REGISTRY_HISTORY_DATASET = (
    Path("gold") / "paper_strategy_registry_history"
)
PAPER_STRATEGY_ACK_CURRENT_DATASET = Path("gold") / "paper_strategy_ack_current"
PAPER_STRATEGY_ACK_HISTORY_DATASET = Path("gold") / "paper_strategy_ack_history"
PAPER_STRATEGY_IDENTITY_CONFLICT_DATASET = (
    Path("gold") / "paper_strategy_identity_conflict"
)
PAPER_COHORT_MANIFEST_DATASET = Path("gold") / "paper_cohort_manifest"
PAPER_STRATEGY_PROMOTION_GATE_DATASET = Path("gold") / "paper_strategy_promotion_gate"
PAPER_STRATEGY_PROPOSAL_ACK_DATASET = Path("silver") / "v5_paper_strategy_proposal_ack"
PAPER_STRATEGY_RUNS_DATASET = Path("gold") / "paper_strategy_runs"
PAPER_STRATEGY_DAILY_DATASET = Path("gold") / "paper_strategy_daily"
STRATEGY_OPPORTUNITY_ADVISORY_DATASET = Path("gold") / "strategy_opportunity_advisory"
ALPHA_DISCOVERY_BOARD_DATASET = Path("gold") / "alpha_discovery_board"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
STRATEGY_COST_TRUST_DATASET = Path("gold") / "strategy_cost_trust"
PAPER_STRATEGY_MIGRATION_AUDIT_DATASET = Path("gold") / "paper_strategy_migration_audit"
V5_PAPER_STRATEGY_COST_EVIDENCE_DATASET = (
    Path("silver") / "v5_paper_strategy_cost_evidence"
)
V5_PAPER_STRATEGY_REGISTRY_DATASET = Path("silver") / "v5_paper_strategy_registry"
V5_PAPER_STRATEGY_REGISTRY_CURRENT_DATASET = (
    Path("silver") / "v5_paper_strategy_registry_current"
)
V5_PAPER_STRATEGY_REGISTRY_HISTORY_DATASET = (
    Path("silver") / "v5_paper_strategy_registry_history"
)
V5_PAPER_STRATEGY_ACK_CURRENT_DATASET = (
    Path("silver") / "v5_paper_strategy_proposal_ack_current"
)
V5_PAPER_STRATEGY_ACK_HISTORY_DATASET = (
    Path("silver") / "v5_paper_strategy_proposal_ack_history"
)
V5_PAPER_STRATEGY_TRACKERS_CURRENT_DATASET = (
    Path("silver") / "v5_paper_strategy_trackers_current"
)

PAPER_PIPELINE_SOURCE = "research.paper_strategy_promotion.v0.1"
PAPER_PIPELINE_SCHEMA_VERSION = "paper_strategy_pipeline.v1"
MIN_PAPER_DAYS = 14
MIN_CLOSED_ENTRIES = 20
MIN_ENTRY_DAY_COUNT = 7
MIN_ARRIVAL_MID_COVERAGE = 0.8
MIN_P25_AFTER_COST_BPS = -30.0
TRUSTED_PAPER_COST_SOURCES = {
    "actual_fills",
    "mixed_actual_proxy",
    "bootstrap_cost_probe",
    "configured_conservative_paper",
    "public_spread_proxy",
}
BLOCKED_PAPER_COST_SOURCES = {
    "fallback_not_live_safe",
    "public_microstructure_proxy",
    "global_default",
    "local_estimate",
    "conservative_shadow_cost",
}
CANARY_COST_SOURCES = {"actual_fills", "mixed_actual_proxy"}
SCALE_COST_SOURCES = {"actual_fills"}

IMMUTABLE_IDENTITY_FIELDS = (
    "proposal_id",
    "proposal_hash",
    "strategy_id",
    "strategy_version",
    "strategy_family",
    "symbol",
    "timeframe",
    "max_holding_bars",
    "contract_version",
)

PAPER_STRATEGY_REGISTRY_SCHEMA = {
    "strategy_id": pl.Utf8,
    "strategy_version": pl.Utf8,
    "strategy_family": pl.Utf8,
    "proposal_id": pl.Utf8,
    "proposal_hash": pl.Utf8,
    "paper_tracker_id": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "max_holding_bars": pl.Int64,
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
    "current_ack_present": pl.Boolean,
    "current_tracker_present": pl.Boolean,
    "paper_tracker_effective": pl.Boolean,
    "current_tracker_effective": pl.Boolean,
    "current_runtime_eligible": pl.Boolean,
    "evidence_from_history": pl.Boolean,
    "current_activation_at": pl.Utf8,
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
    "identity_conflict": pl.Boolean,
    "identity_conflict_fields": pl.Utf8,
    "current_proposal_member": pl.Boolean,
    "current_cohort_member": pl.Boolean,
    "supersession_status": pl.Utf8,
    "new_entry_allowed": pl.Boolean,
    "exit_allowed": pl.Boolean,
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
    "current_ack_present": pl.Boolean,
    "current_tracker_present": pl.Boolean,
    "current_runtime_eligible": pl.Boolean,
    "historical_evidence_present": pl.Boolean,
    "historical_closed_trade_count": pl.Int64,
    "current_contract_closed_trade_count": pl.Int64,
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
    "paper_cost_model_usable": pl.Boolean,
    "closed_trade_cost_observation_count": pl.Int64,
    "closed_trade_cost_missing_count": pl.Int64,
    "closed_trade_cost_coverage": pl.Float64,
    "closed_trade_cost_sources": pl.Utf8,
    "dimensional_cost_trust_level": pl.Utf8,
    "dimensional_cost_trust_matched": pl.Boolean,
    "strategy_cost_trust_level": pl.Utf8,
    "required_cost_trust_level": pl.Utf8,
    "cost_trusted_for_paper": pl.Boolean,
    "cost_trusted_for_canary": pl.Boolean,
    "cost_trusted_for_scale": pl.Boolean,
    "raw_closed_trade_count": pl.Int64,
    "independent_closed_trade_count": pl.Int64,
    "canonical_opportunity_id": pl.Utf8,
    "shared_entry_event_id": pl.Utf8,
    "evidence_independence_weight": pl.Float64,
    "horizon_variant_count": pl.Int64,
    "win_rate": pl.Float64,
    "avg_mfe_bps": pl.Float64,
    "avg_mae_bps": pl.Float64,
    "avg_profit_giveback_bps": pl.Float64,
    "avg_exit_efficiency": pl.Float64,
    "backtest_paper_conflict": pl.Boolean,
    "promotion_score": pl.Float64,
    "promotion_confidence": pl.Utf8,
    "evidence_quality": pl.Utf8,
    "tail_risk_status": pl.Utf8,
    "regime_coverage_status": pl.Utf8,
    "cost_evidence_status": pl.Utf8,
    "identity_conflict": pl.Boolean,
    "identity_conflict_fields": pl.Utf8,
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

PAPER_STRATEGY_IDENTITY_CONFLICT_SCHEMA = {
    "proposal_id": pl.Utf8,
    "proposal_hash": pl.Utf8,
    "tracker_id": pl.Utf8,
    "conflict_field": pl.Utf8,
    "canonical_value": pl.Utf8,
    "observed_value": pl.Utf8,
    "source_dataset": pl.Utf8,
    "source_row": pl.Utf8,
    "active_conflict": pl.Boolean,
    "resolution_status": pl.Utf8,
    "created_at": pl.Utf8,
}

PAPER_COHORT_MANIFEST_SCHEMA = {
    "cohort_id": pl.Utf8,
    "cohort_version": pl.Int64,
    "proposal_ids": pl.Utf8,
    "proposal_hashes": pl.Utf8,
    "strategy_ids": pl.Utf8,
    "symbols": pl.Utf8,
    "horizons": pl.Utf8,
    "admitted_at": pl.Utf8,
    "observation_start_at": pl.Utf8,
    "status": pl.Utf8,
    "proposal_count": pl.Int64,
    "accepted_ack_count": pl.Int64,
    "active_tracker_count": pl.Int64,
    "current_runtime_eligible_count": pl.Int64,
    "pending_proposal_ids": pl.Utf8,
    "ack_missing_proposal_ids": pl.Utf8,
    "tracker_missing_proposal_ids": pl.Utf8,
    "identity_conflict_proposal_ids": pl.Utf8,
    "all_members_admitted": pl.Boolean,
    "formal_observation_eligible": pl.Boolean,
    "raw_closed_trade_count": pl.Int64,
    "independent_closed_trade_count": pl.Int64,
    "horizon_variant_count": pl.Int64,
    "created_at": pl.Utf8,
    "last_evaluated_at": pl.Utf8,
    "last_evidence_at": pl.Utf8,
    "evidence_signature": pl.Utf8,
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
    existing_registry_current = read_parquet_dataset(
        root / PAPER_STRATEGY_REGISTRY_CURRENT_DATASET
    )
    if existing_registry_current.is_empty():
        existing_registry_current = existing_registry
    existing_registry_history = read_parquet_dataset(
        root / PAPER_STRATEGY_REGISTRY_HISTORY_DATASET
    )
    if existing_registry_history.is_empty():
        existing_registry_history = existing_registry
    existing_proposals = read_parquet_dataset(root / PAPER_STRATEGY_PROPOSAL_DATASET)
    existing_migration_audit = read_parquet_dataset(root / PAPER_STRATEGY_MIGRATION_AUDIT_DATASET)
    configured = build_configured_proposals(
        _proposal_source_evidence(legacy_evidence, advisory)
    )
    proposal_frame = existing_proposals if not existing_proposals.is_empty() else advisory
    if configured:
        published = publish_proposals(root, [proposal for proposal, _evidence in configured])
        proposal_frame = published
    proposal_frame, _proposal_snapshot = publish_canonical_proposal_snapshot(
        root,
        proposal_frame,
    )
    raw_ack = read_parquet_dataset(root / V5_PAPER_STRATEGY_ACK_CURRENT_DATASET)
    raw_ack_history = read_parquet_dataset(root / V5_PAPER_STRATEGY_ACK_HISTORY_DATASET)
    if raw_ack_history.is_empty():
        raw_ack_history = read_parquet_dataset(root / PAPER_STRATEGY_PROPOSAL_ACK_DATASET)
    raw_trackers = read_parquet_dataset(
        root / V5_PAPER_STRATEGY_TRACKERS_CURRENT_DATASET
    )
    current_identity_conflicts = _build_identity_conflicts(
        proposal_frame,
        (
            ("paper_strategy_registry", existing_registry, False),
            (
                "v5_paper_strategy_proposal_ack",
                raw_ack,
                True,
            ),
            (
                "v5_paper_strategy_trackers_current",
                raw_trackers,
                True,
            ),
        ),
    )
    identity_conflicts = _merge_identity_conflicts(
        read_parquet_dataset(root / PAPER_STRATEGY_IDENTITY_CONFLICT_DATASET),
        current_identity_conflicts,
    )
    write_parquet_dataset(
        identity_conflicts,
        root / PAPER_STRATEGY_IDENTITY_CONFLICT_DATASET,
    )
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
        paper_runtime_cost_evidence=read_parquet_dataset(
            root / V5_PAPER_STRATEGY_COST_EVIDENCE_DATASET
        ),
    )
    write_parquet_dataset(strategy_cost_trust, root / STRATEGY_COST_TRUST_DATASET)
    paper_ack = _filter_superseded_structured_rows(raw_ack, proposal_frame)
    paper_ack = _latest_rows_frame(paper_ack)
    ack_history = _merge_ack_history(raw_ack_history, raw_ack)
    write_parquet_dataset(ack_history, root / PAPER_STRATEGY_ACK_HISTORY_DATASET)
    write_parquet_dataset(paper_ack, root / PAPER_STRATEGY_ACK_CURRENT_DATASET)
    trackers_current = _latest_rows_frame(
        _filter_superseded_structured_rows(raw_trackers, proposal_frame)
    )
    write_parquet_dataset(
        trackers_current,
        root / PAPER_STRATEGY_TRACKERS_CURRENT_DATASET,
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
        trackers_current=trackers_current,
        runs=paper_runs,
        daily=paper_daily,
        strategy_cost_trust=strategy_cost_trust,
    )
    registry = frames["paper_strategy_registry"]
    registry = _canonicalize_registry_identity(registry, proposal_frame)
    registry = _apply_active_identity_conflicts(registry, identity_conflicts)
    gate = _promotion_gate_frame_from_registry(
        registry,
        daily=paper_daily,
        runs=paper_runs,
        proposals=proposal_frame,
        strategy_cost_trust=strategy_cost_trust,
    )
    gate = _canonicalize_gate_identity(gate, proposal_frame)
    registry_history = _merge_lifecycle_frame(
        _merge_lifecycle_frame(
            existing_registry_history,
            existing_registry_current,
            schema=PAPER_STRATEGY_REGISTRY_SCHEMA,
            rank=_registry_persistence_rank,
        ),
        registry,
        schema=PAPER_STRATEGY_REGISTRY_SCHEMA,
        rank=_registry_persistence_rank,
    )
    registry_history = _as_registry_history(registry_history)
    write_parquet_dataset(registry, root / PAPER_STRATEGY_REGISTRY_DATASET)
    write_parquet_dataset(registry, root / PAPER_STRATEGY_REGISTRY_CURRENT_DATASET)
    write_parquet_dataset(
        registry_history,
        root / PAPER_STRATEGY_REGISTRY_HISTORY_DATASET,
    )
    write_parquet_dataset(gate, root / PAPER_STRATEGY_PROMOTION_GATE_DATASET)
    cohort = _build_paper_cohort_manifest(
        read_parquet_dataset(root / PAPER_COHORT_MANIFEST_DATASET),
        proposals=proposal_frame,
        registry=registry,
        runs=paper_runs,
        created_at=datetime.now(UTC),
    )
    write_parquet_dataset(cohort, root / PAPER_COHORT_MANIFEST_DATASET)
    return PaperStrategyPipelineResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        paper_strategy_registry=registry.height,
        paper_strategy_promotion_gate=gate.height,
        paper_strategy_migration_audit=migration_audit.height,
    )


def _proposal_source_evidence(
    discovery: pl.DataFrame,
    advisory: pl.DataFrame,
) -> pl.DataFrame:
    """Break the pre-Paper circular gate while honoring explicit downgrades."""
    if discovery.is_empty() or advisory.is_empty():
        return discovery
    latest_advisory: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in advisory.to_dicts():
        key = _proposal_evidence_key(row)
        if key == ("", "", 0):
            continue
        current = latest_advisory.get(key)
        if current is None or _row_sort_ts(row) >= _row_sort_ts(current):
            latest_advisory[key] = row
    rows = []
    for row in discovery.to_dicts():
        advisory_row = latest_advisory.get(_proposal_evidence_key(row), {})
        decision = _text(advisory_row.get("decision")).upper()
        reasons = {reason.lower() for reason in _json_list(advisory_row.get("live_block_reasons"))}
        if "downgraded_from_paper" in reasons or decision in {
            "DEAD",
            "KILL",
            "QUARANTINE",
        }:
            continue
        rows.append(row)
    if not rows:
        return discovery.head(0)
    return pl.DataFrame(rows, infer_schema_length=None).cast(discovery.schema, strict=False).select(
        discovery.columns
    )


def _proposal_evidence_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    candidate = _text(
        row.get("strategy_candidate") or row.get("candidate_name")
    ).lower()
    raw_symbol = _text(row.get("symbol") or row.get("v5_symbol"))
    symbol = normalize_symbol(raw_symbol) if raw_symbol else ""
    return candidate, symbol, _int(row.get("horizon_hours") or row.get("suggested_horizon")) or 0


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


def _merge_ack_history(history: pl.DataFrame, current: pl.DataFrame) -> pl.DataFrame:
    if history.is_empty():
        return current
    if current.is_empty():
        return history
    combined = pl.concat([history, current], how="diagonal_relaxed")
    key_columns = [
        column
        for column in (
            "proposal_id",
            "proposal_hash",
            "tracker_id",
            "paper_tracker_id",
            "accepted_at",
            "ingest_ts",
        )
        if column in combined.columns
    ]
    return combined.unique(subset=key_columns, keep="last", maintain_order=True)


def _latest_rows_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    rows = _latest_rows_by_strategy(frame)
    if not rows:
        return frame.head(0)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .cast(frame.schema, strict=False)
        .select(frame.columns)
    )


def _build_identity_conflicts(
    proposals: pl.DataFrame,
    sources: tuple[tuple[str, pl.DataFrame, bool], ...],
) -> pl.DataFrame:
    if proposals.is_empty():
        return _frame([], PAPER_STRATEGY_IDENTITY_CONFLICT_SCHEMA)
    canonical = {
        _text(row.get("proposal_id")): row
        for row in proposals.to_dicts()
        if _text(row.get("proposal_id"))
    }
    created_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    tracker_owners: dict[str, list[tuple[str, str, bool, Mapping[str, Any]]]] = {}
    for source_name, frame, active_conflict in sources:
        if frame.is_empty():
            continue
        for observed in frame.to_dicts():
            proposal_id = _text(observed.get("proposal_id"))
            proposal = canonical.get(proposal_id)
            if proposal is None:
                continue
            tracker_id = _text(
                observed.get("tracker_id") or observed.get("paper_tracker_id")
            )
            if tracker_id:
                tracker_owners.setdefault(tracker_id, []).append(
                    (proposal_id, source_name, active_conflict, observed)
                )
                encoded_proposal_id = (
                    tracker_id.removeprefix("paper:")
                    if tracker_id.startswith("paper:")
                    else ""
                )
                if encoded_proposal_id and encoded_proposal_id != proposal_id:
                    rows.append(
                        {
                            "proposal_id": proposal_id,
                            "proposal_hash": _text(proposal.get("proposal_hash")),
                            "tracker_id": tracker_id,
                            "conflict_field": "tracker_id_to_proposal_id",
                            "canonical_value": proposal_id,
                            "observed_value": encoded_proposal_id,
                            "source_dataset": source_name,
                            "source_row": safe_json_dumps(observed),
                            "active_conflict": active_conflict,
                            "resolution_status": (
                                "BLOCKED_ACTIVE_CONFLICT"
                                if active_conflict
                                else "RESOLVED_BY_CANONICAL_PROPOSAL"
                            ),
                            "created_at": created_at,
                        }
                    )
            for field in IMMUTABLE_IDENTITY_FIELDS:
                expected = _identity_value(proposal, field)
                actual = _identity_value(observed, field)
                if not actual or expected == actual:
                    continue
                rows.append(
                    {
                        "proposal_id": proposal_id,
                        "proposal_hash": _text(proposal.get("proposal_hash")),
                        "tracker_id": _text(
                            observed.get("tracker_id")
                            or observed.get("paper_tracker_id")
                        ),
                        "conflict_field": field,
                        "canonical_value": expected,
                        "observed_value": actual,
                        "source_dataset": source_name,
                        "source_row": safe_json_dumps(observed),
                        "active_conflict": active_conflict,
                        "resolution_status": (
                            "BLOCKED_ACTIVE_CONFLICT"
                            if active_conflict
                            else "RESOLVED_BY_CANONICAL_PROPOSAL"
                        ),
                        "created_at": created_at,
                    }
                )
    for tracker_id, owners in tracker_owners.items():
        proposal_ids = sorted({owner[0] for owner in owners})
        if len(proposal_ids) <= 1:
            continue
        for proposal_id, source_name, active_conflict, observed in owners:
            proposal = canonical[proposal_id]
            rows.append(
                {
                    "proposal_id": proposal_id,
                    "proposal_hash": _text(proposal.get("proposal_hash")),
                    "tracker_id": tracker_id,
                    "conflict_field": "tracker_id_to_proposal_id",
                    "canonical_value": proposal_id,
                    "observed_value": safe_json_dumps(proposal_ids),
                    "source_dataset": f"{source_name}:cross_source_tracker_mapping",
                    "source_row": safe_json_dumps(observed),
                    "active_conflict": active_conflict,
                    "resolution_status": (
                        "BLOCKED_ACTIVE_CONFLICT"
                        if active_conflict
                        else "RESOLVED_BY_CANONICAL_PROPOSAL"
                    ),
                    "created_at": created_at,
                }
            )
    return _frame(rows, PAPER_STRATEGY_IDENTITY_CONFLICT_SCHEMA)


def _merge_identity_conflicts(
    existing: pl.DataFrame,
    current: pl.DataFrame,
) -> pl.DataFrame:
    current_rows = current.to_dicts() if not current.is_empty() else []
    active_keys = {
        (
            _text(row.get("proposal_id")),
            _text(row.get("source_dataset")),
            _text(row.get("conflict_field")),
            _text(row.get("observed_value")),
        )
        for row in current_rows
    }
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for raw in existing.to_dicts() if not existing.is_empty() else []:
        row = dict(raw)
        key = (
            _text(row.get("proposal_id")),
            _text(row.get("source_dataset")),
            _text(row.get("conflict_field")),
            _text(row.get("observed_value")),
        )
        if _bool(row.get("active_conflict")) and key not in active_keys:
            row["active_conflict"] = False
            row["resolution_status"] = "RESOLVED_SOURCE_CORRECTED"
        rows[key] = row
    for row in current_rows:
        key = (
            _text(row.get("proposal_id")),
            _text(row.get("source_dataset")),
            _text(row.get("conflict_field")),
            _text(row.get("observed_value")),
        )
        rows[key] = row
    return _frame(list(rows.values()), PAPER_STRATEGY_IDENTITY_CONFLICT_SCHEMA)


def _identity_value(row: Mapping[str, Any], field: str) -> str:
    if field == "symbol":
        return _symbol(row)
    if field == "max_holding_bars":
        return str(
            _int(
                row.get("max_holding_bars")
                or row.get("horizon_hours")
                or row.get("suggested_horizon")
            )
            or ""
        )
    return _text(row.get(field))


def _canonicalize_registry_identity(
    registry: pl.DataFrame,
    proposals: pl.DataFrame,
) -> pl.DataFrame:
    if registry.is_empty() or proposals.is_empty():
        return registry
    canonical = {
        _text(row.get("proposal_id")): row
        for row in proposals.to_dicts()
        if _text(row.get("proposal_id"))
    }
    rows: list[dict[str, Any]] = []
    for source in registry.to_dicts():
        row = dict(source)
        proposal = canonical.get(_text(row.get("proposal_id")))
        if proposal is not None:
            row.update(
                {
                    "proposal_id": _text(proposal.get("proposal_id")),
                    "proposal_hash": _text(proposal.get("proposal_hash")),
                    "strategy_id": _text(proposal.get("strategy_id")),
                    "strategy_version": _text(proposal.get("strategy_version")),
                    "strategy_family": _text(proposal.get("strategy_family")),
                    "symbol": _symbol(proposal),
                    "timeframe": _text(proposal.get("timeframe")),
                    "max_holding_bars": _int(
                        proposal.get("max_holding_bars")
                        or proposal.get("horizon_hours")
                        or proposal.get("suggested_horizon")
                    )
                    or 0,
                    "contract_version": _text(proposal.get("contract_version"))
                    or PAPER_STRATEGY_CONTRACT_VERSION,
                    "identity_conflict": False,
                    "identity_conflict_fields": "[]",
                    "current_proposal_member": True,
                    "current_cohort_member": True,
                    "exit_allowed": True,
                }
            )
        rows.append(row)
    return _frame(rows, PAPER_STRATEGY_REGISTRY_SCHEMA)


def _as_registry_history(registry: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in registry.to_dicts() if not registry.is_empty() else []:
        row = dict(source)
        row.update(
            {
                "current_ack_present": False,
                "current_tracker_present": False,
                "paper_tracker_effective": False,
                "current_tracker_effective": False,
                "current_runtime_eligible": False,
                "evidence_from_history": True,
                "current_activation_at": "",
                "current_proposal_member": False,
                "current_cohort_member": False,
                "supersession_status": "HISTORY_ONLY",
                "new_entry_allowed": False,
                "exit_allowed": True,
            }
        )
        rows.append(row)
    return _frame(rows, PAPER_STRATEGY_REGISTRY_SCHEMA)


def _apply_active_identity_conflicts(
    registry: pl.DataFrame,
    conflicts: pl.DataFrame,
) -> pl.DataFrame:
    if registry.is_empty() or conflicts.is_empty():
        return registry
    fields_by_proposal: dict[str, set[str]] = {}
    for row in conflicts.to_dicts():
        if not _bool(row.get("active_conflict")):
            continue
        fields_by_proposal.setdefault(_text(row.get("proposal_id")), set()).add(
            _text(row.get("conflict_field"))
        )
    rows = []
    for source in registry.to_dicts():
        row = dict(source)
        fields = sorted(fields_by_proposal.get(_text(row.get("proposal_id")), set()))
        if fields:
            row["identity_conflict"] = True
            row["identity_conflict_fields"] = safe_json_dumps(fields)
            row["status"] = "IDENTITY_CONFLICT"
            row["lifecycle_state"] = "IDENTITY_CONFLICT"
            row["lifecycle_reason"] = "canonical_identity_conflict"
            row["blocked_reasons"] = safe_json_dumps(["canonical_identity_conflict"])
            row["paper_tracker_effective"] = False
            row["current_tracker_effective"] = False
            row["current_runtime_eligible"] = False
            row["supersession_status"] = "IDENTITY_CONFLICT"
            row["new_entry_allowed"] = False
        rows.append(row)
    return _frame(rows, PAPER_STRATEGY_REGISTRY_SCHEMA)


def _canonicalize_gate_identity(
    gate: pl.DataFrame,
    proposals: pl.DataFrame,
) -> pl.DataFrame:
    if gate.is_empty() or proposals.is_empty():
        return gate
    canonical = {
        _text(row.get("proposal_id")): row
        for row in proposals.to_dicts()
        if _text(row.get("proposal_id"))
    }
    rows = []
    for source in gate.to_dicts():
        row = dict(source)
        proposal = canonical.get(_text(row.get("proposal_id")))
        if proposal is not None:
            row["proposal_hash"] = _text(proposal.get("proposal_hash"))
            row["strategy_id"] = _text(proposal.get("strategy_id"))
            row["strategy_version"] = _text(proposal.get("strategy_version"))
            row["symbol"] = _symbol(proposal)
        rows.append(row)
    return _frame(rows, PAPER_STRATEGY_PROMOTION_GATE_SCHEMA)


def _build_paper_cohort_manifest(
    existing: pl.DataFrame,
    *,
    proposals: pl.DataFrame,
    registry: pl.DataFrame,
    runs: pl.DataFrame,
    created_at: datetime,
) -> pl.DataFrame:
    existing_rows = existing.to_dicts() if not existing.is_empty() else []
    evaluated_at = created_at.astimezone(UTC).isoformat()
    proposal_rows = sorted(
        proposals.to_dicts() if not proposals.is_empty() else [],
        key=lambda row: _text(row.get("proposal_id")),
    )
    if not proposal_rows:
        for row in existing_rows:
            row["created_at"] = _text(row.get("created_at")) or _text(
                row.get("admitted_at")
            )
            row["last_evaluated_at"] = evaluated_at
            row["last_evidence_at"] = (
                _text(row.get("last_evidence_at"))
                or _text(row.get("created_at"))
            )
        return _frame(existing_rows, PAPER_COHORT_MANIFEST_SCHEMA)
    proposal_ids = [_text(row.get("proposal_id")) for row in proposal_rows]
    proposal_id_set = set(proposal_ids)
    signature = safe_json_dumps(proposal_ids)
    registry_rows = [
        row
        for row in (registry.to_dicts() if not registry.is_empty() else [])
        if _text(row.get("proposal_id")) in proposal_id_set
    ]
    by_proposal = {
        _text(row.get("proposal_id")): row
        for row in registry_rows
        if _text(row.get("proposal_id"))
    }
    accepted_ack_ids = sorted(
        proposal_id
        for proposal_id in proposal_ids
        if _bool(by_proposal.get(proposal_id, {}).get("current_ack_present"))
        and _bool(by_proposal.get(proposal_id, {}).get("accepted"))
    )
    active_tracker_ids = sorted(
        proposal_id
        for proposal_id in proposal_ids
        if _bool(by_proposal.get(proposal_id, {}).get("current_tracker_present"))
    )
    runtime_eligible_ids = sorted(
        proposal_id
        for proposal_id in proposal_ids
        if _bool(by_proposal.get(proposal_id, {}).get("current_runtime_eligible"))
    )
    ack_missing_ids = sorted(set(proposal_ids) - set(accepted_ack_ids))
    tracker_missing_ids = sorted(set(proposal_ids) - set(active_tracker_ids))
    identity_conflict_ids = sorted(
        proposal_id
        for proposal_id in proposal_ids
        if _bool(by_proposal.get(proposal_id, {}).get("identity_conflict"))
    )
    pending_ids = sorted(set(proposal_ids) - set(runtime_eligible_ids))
    all_members_admitted = (
        len(runtime_eligible_ids) == len(proposal_ids)
        and not identity_conflict_ids
    )
    activation_times = [
        _text(by_proposal[proposal_id].get("current_activation_at"))
        for proposal_id in runtime_eligible_ids
        if _text(by_proposal[proposal_id].get("current_activation_at"))
    ]
    observation_start_at = (
        max(activation_times)
        if all_members_admitted and len(activation_times) == len(proposal_ids)
        else None
    )
    admission = {
        "proposal_count": len(proposal_ids),
        "accepted_ack_count": len(accepted_ack_ids),
        "active_tracker_count": len(active_tracker_ids),
        "current_runtime_eligible_count": len(runtime_eligible_ids),
        "pending_proposal_ids": safe_json_dumps(pending_ids),
        "ack_missing_proposal_ids": safe_json_dumps(ack_missing_ids),
        "tracker_missing_proposal_ids": safe_json_dumps(tracker_missing_ids),
        "identity_conflict_proposal_ids": safe_json_dumps(identity_conflict_ids),
        "all_members_admitted": all_members_admitted,
        "formal_observation_eligible": all_members_admitted,
    }
    run_rows = [
        row
        for row in (runs.to_dicts() if not runs.is_empty() else [])
        if _text(row.get("proposal_id")) in proposal_id_set
    ]
    event_ids = {_paper_event_id(row) for row in run_rows if _paper_event_id(row)}
    horizons = sorted(
        {
            _int(
                row.get("max_holding_bars")
                or row.get("horizon_hours")
                or row.get("suggested_horizon")
            )
            or 0
            for row in proposal_rows
        }
    )
    member_states = sorted(
        (
            proposal_id,
            _text(by_proposal.get(proposal_id, {}).get("lifecycle_state")),
            _bool(by_proposal.get(proposal_id, {}).get("current_ack_present")),
            _bool(by_proposal.get(proposal_id, {}).get("current_tracker_present")),
            _bool(by_proposal.get(proposal_id, {}).get("current_runtime_eligible")),
        )
        for proposal_id in proposal_ids
    )
    evidence_signature = hashlib.sha256(
        safe_json_dumps(
            {
                **admission,
                "raw_closed_trade_count": len(run_rows),
                "independent_closed_trade_count": len(event_ids),
                "horizon_variant_count": len(horizons),
                "member_states": member_states,
            }
        ).encode()
    ).hexdigest()
    latest = max(existing_rows, key=lambda row: _int(row.get("cohort_version")) or 0, default=None)
    if latest is not None and _text(latest.get("proposal_ids")) == signature:
        previous_evidence_signature = _text(latest.get("evidence_signature"))
        latest["raw_closed_trade_count"] = len(run_rows)
        latest["independent_closed_trade_count"] = len(event_ids)
        latest["horizon_variant_count"] = len(horizons)
        latest.update(admission)
        if _text(latest.get("status")) not in {"FROZEN", "INVALID"}:
            if all_members_admitted:
                latest["status"] = (
                    "REVIEW_READY"
                    if _text(latest.get("status")) == "REVIEW_READY"
                    else "OBSERVING"
                )
                latest["observation_start_at"] = (
                    _text(latest.get("observation_start_at"))
                    or observation_start_at
                )
            else:
                latest["status"] = "FORMING"
                latest["observation_start_at"] = None
        else:
            latest["formal_observation_eligible"] = False
        latest["created_at"] = _text(latest.get("created_at")) or _text(
            latest.get("admitted_at")
        )
        latest["last_evaluated_at"] = evaluated_at
        latest["evidence_signature"] = evidence_signature
        if previous_evidence_signature != evidence_signature:
            latest["last_evidence_at"] = evaluated_at
        else:
            latest["last_evidence_at"] = (
                _text(latest.get("last_evidence_at"))
                or _text(latest.get("created_at"))
            )
        return _frame(existing_rows, PAPER_COHORT_MANIFEST_SCHEMA)
    for row in existing_rows:
        row["last_evaluated_at"] = evaluated_at
        if _text(row.get("status")) not in {"FROZEN", "INVALID"}:
            row["status"] = "FROZEN"
            row["formal_observation_eligible"] = False
            row["last_evidence_at"] = evaluated_at
    version = max((_int(row.get("cohort_version")) or 0 for row in existing_rows), default=0) + 1
    admitted = created_at.astimezone(UTC).isoformat()
    material = "|".join(
        f"{row.get('proposal_id')}:{row.get('proposal_hash')}" for row in proposal_rows
    )
    existing_rows.append(
        {
            "cohort_id": f"paper-cohort-{hashlib.sha256(material.encode()).hexdigest()[:16]}",
            "cohort_version": version,
            "proposal_ids": signature,
            "proposal_hashes": safe_json_dumps(
                [_text(row.get("proposal_hash")) for row in proposal_rows]
            ),
            "strategy_ids": safe_json_dumps(
                [_text(row.get("strategy_id")) for row in proposal_rows]
            ),
            "symbols": safe_json_dumps(sorted({_symbol(row) for row in proposal_rows})),
            "horizons": safe_json_dumps(horizons),
            "admitted_at": admitted,
            "observation_start_at": observation_start_at,
            "status": "OBSERVING" if all_members_admitted else "FORMING",
            **admission,
            "raw_closed_trade_count": len(run_rows),
            "independent_closed_trade_count": len(event_ids),
            "horizon_variant_count": len(horizons),
            "created_at": admitted,
            "last_evaluated_at": evaluated_at,
            "last_evidence_at": evaluated_at,
            "evidence_signature": evidence_signature,
        }
    )
    return _frame(existing_rows, PAPER_COHORT_MANIFEST_SCHEMA)


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
    event_variants = _event_variant_counts(run_rows)
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
                event_variants=event_variants,
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
    active_hash_by_id: dict[str, str] = {}
    structured_contract_present = False
    for row in proposals.to_dicts():
        strategy_id = _text(row.get("strategy_id"))
        strategy_version = _text(row.get("strategy_version"))
        proposal_id = _text(row.get("proposal_id"))
        proposal_hash = _text(row.get("proposal_hash"))
        structured = bool(
            _text(row.get("contract_version")) == PAPER_STRATEGY_CONTRACT_VERSION
            or (
                proposal_hash
                and row.get("entry_rule") not in (None, "")
                and row.get("exit_rule") not in (None, "")
            )
        )
        structured_contract_present = structured_contract_present or structured
        if strategy_id and strategy_version and proposal_id:
            active[(strategy_id, strategy_version)] = proposal_id
        if structured and proposal_id:
            active_hash_by_id[proposal_id] = proposal_hash
    if not active and not active_hash_by_id:
        return frame
    rows = []
    for row in frame.to_dicts():
        proposal_id = _text(row.get("proposal_id"))
        identity = _structured_proposal_identity(proposal_id)
        if identity in active and proposal_id != active[identity]:
            continue
        if structured_contract_present:
            if proposal_id not in active_hash_by_id:
                continue
            expected_hash = active_hash_by_id[proposal_id]
            observed_hash = _text(row.get("proposal_hash"))
            if expected_hash and observed_hash and observed_hash != expected_hash:
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


def build_paper_strategy_pipeline_frames(
    *,
    proposals: pl.DataFrame | None = None,
    proposal_ack: pl.DataFrame | None = None,
    trackers_current: pl.DataFrame | None = None,
    runs: pl.DataFrame | None = None,
    daily: pl.DataFrame | None = None,
    strategy_cost_trust: pl.DataFrame | None = None,
    created_at: datetime | None = None,
) -> dict[str, pl.DataFrame]:
    created = (created_at or datetime.now(UTC)).isoformat()
    proposal_frame = proposals if proposals is not None else pl.DataFrame()
    ack_frame = proposal_ack if proposal_ack is not None else pl.DataFrame()
    tracker_frame = trackers_current if trackers_current is not None else pl.DataFrame()
    run_frame = runs if runs is not None else pl.DataFrame()
    daily_frame = daily if daily is not None else pl.DataFrame()
    ack_frame = _filter_superseded_structured_rows(ack_frame, proposal_frame)
    tracker_frame = _filter_superseded_structured_rows(tracker_frame, proposal_frame)
    run_frame = _filter_superseded_structured_rows(run_frame, proposal_frame)
    daily_frame = _filter_superseded_structured_rows(daily_frame, proposal_frame)
    proposal_rows = _proposal_rows(proposal_frame)
    ack_rows = _latest_rows_by_strategy(
        ack_frame
    )
    tracker_rows = _latest_rows_by_strategy(tracker_frame)
    run_rows = run_frame.to_dicts()
    event_variants = _event_variant_counts(run_rows)
    daily_rows = _latest_rows_by_strategy(daily_frame)
    cost_trust_rows = (
        strategy_cost_trust.to_dicts()
        if strategy_cost_trust is not None and not strategy_cost_trust.is_empty()
        else []
    )

    registry_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for proposal in sorted(
        proposal_rows,
        key=lambda row: (_symbol(row), _text(row.get("proposal_id"))),
    ):
        proposal_id = _text(proposal.get("proposal_id"))
        proposal_hash = _text(proposal.get("proposal_hash"))
        ack = _latest_exact_current_row(ack_rows, proposal_id, proposal_hash)
        tracker = _latest_exact_current_row(tracker_rows, proposal_id, proposal_hash)
        key = _StrategyKey(
            proposal_id=proposal_id,
            paper_tracker_id=_text(
                tracker.get("tracker_id") or tracker.get("paper_tracker_id")
            ),
            strategy_candidate=_text(proposal.get("strategy_candidate")),
            symbol=_symbol(proposal),
        )
        daily_row = _match_strategy_row(daily_rows, key)
        matched_runs = [row for row in run_rows if _row_matches_key(row, key)]
        registry = _registry_row(
            key=key,
            proposal=proposal,
            ack=ack,
            tracker=tracker,
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
            event_variants=event_variants,
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


def _latest_exact_current_row(
    rows: list[dict[str, Any]],
    proposal_id: str,
    proposal_hash: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if _text(row.get("proposal_id")) == proposal_id
        and _text(row.get("proposal_hash")) == proposal_hash
    ]
    if not matches:
        return {}
    return max(matches, key=_row_sort_ts)


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
    tracker: Mapping[str, Any],
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    current_ack_present = bool(ack)
    current_tracker_present = bool(tracker)
    accepted = current_ack_present and _bool(ack.get("accepted")) is True
    reject_reason = "" if accepted is True else _text(ack.get("reject_reason"))
    proposal_id = (
        key.proposal_id or _text(ack.get("proposal_id")) or _text(proposal.get("proposal_id"))
    )
    paper_tracker_id = _text(
        tracker.get("tracker_id") or tracker.get("paper_tracker_id")
    )
    strategy_id = (
        _text(proposal.get("strategy_id"))
        or paper_tracker_id
        or proposal_id
        or _text(daily.get("strategy_id"))
    )
    paper_only = _paper_only(ack, proposal) and _paper_only(tracker, proposal)
    max_live_notional = _float(
        ack.get("max_live_notional_usdt")
        or ack.get("advisory_max_live_notional_usdt")
        or proposal.get("max_live_notional_usdt")
    )
    if max_live_notional is None and paper_only:
        max_live_notional = 0.0
    current_runtime_eligible = bool(
        accepted
        and current_tracker_present
        and paper_tracker_id
        and _bool(tracker.get("rules_locked")) is True
        and paper_only
    )
    if accepted and not current_tracker_present:
        status = "CURRENT_ACKED_TRACKER_MISSING"
    elif current_tracker_present and not accepted:
        status = "CURRENT_TRACKER_ACK_MISSING"
    elif current_runtime_eligible:
        status = "CURRENT_ACTIVE"
    elif reject_reason:
        status = "REJECTED_BY_V5"
    else:
        status = "CURRENT_PENDING_ACK"
    lifecycle = legacy_lifecycle_state(
        status,
        accepted=accepted,
        tracker_effective=current_runtime_eligible,
    )
    lifecycle_blocked = _registry_block_reasons(
        accepted=accepted,
        reject_reason=reject_reason,
        paper_tracker_id=paper_tracker_id,
    )
    return {
        "strategy_id": strategy_id,
        "strategy_version": _text(proposal.get("strategy_version"))
        or _text(ack.get("strategy_version"))
        or _strategy_version(strategy_id or proposal_id),
        "strategy_family": _text(proposal.get("strategy_family"))
        or _text(proposal.get("strategy_candidate"))
        or key.strategy_candidate,
        "proposal_id": proposal_id,
        "proposal_hash": _text(proposal.get("proposal_hash"))
        or _text(ack.get("proposal_hash"))
        or _proposal_hash(proposal),
        "paper_tracker_id": paper_tracker_id,
        "symbol": _symbol(proposal) or key.symbol or _symbol(ack) or _symbol(daily),
        "timeframe": _text(proposal.get("timeframe")),
        "max_holding_bars": _int(
            proposal.get("max_holding_bars")
            or proposal.get("horizon_hours")
            or proposal.get("suggested_horizon")
        )
        or 0,
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
        "rules_locked": current_runtime_eligible,
        "accepted": accepted,
        "current_ack_present": current_ack_present,
        "current_tracker_present": current_tracker_present,
        "paper_tracker_effective": current_tracker_present,
        "current_tracker_effective": current_tracker_present,
        "current_runtime_eligible": current_runtime_eligible,
        "evidence_from_history": bool(daily or runs),
        "current_activation_at": (
            _current_activation_at(ack, tracker) if current_runtime_eligible else ""
        ),
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
        "identity_conflict": False,
        "identity_conflict_fields": "[]",
        "current_proposal_member": True,
        "current_cohort_member": True,
        "supersession_status": (
            status if status.startswith("CURRENT_") else "CURRENT_PENDING_ACK"
        ),
        "new_entry_allowed": current_runtime_eligible,
        "exit_allowed": True,
    }


def _promotion_gate_row(
    *,
    registry: Mapping[str, Any],
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
    proposal: Mapping[str, Any],
    strategy_cost_trust: Mapping[str, Any],
    event_variants: Mapping[str, set[str]],
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
    cost_sources, cost_source_errors = parse_cost_source_mix(
        daily.get("cost_source_mix")
    )
    paper_cost_model_usable = bool(cost_sources) and not cost_source_errors and (
        cost_sources <= TRUSTED_PAPER_COST_SOURCES
    )
    closed_cost_sources: set[str] = set()
    closed_trade_cost_observation_count = 0
    closed_trade_cost_invalid_count = 0
    for run in runs:
        run_sources, run_errors = parse_cost_source_mix(
            run.get("cost_source") or run.get("cost_source_mix")
        )
        if run_sources and not run_errors:
            closed_trade_cost_observation_count += 1
            closed_cost_sources.update(run_sources)
        else:
            closed_trade_cost_invalid_count += 1
    closed_trade_cost_missing_count = max(
        closed_entries - closed_trade_cost_observation_count,
        closed_trade_cost_invalid_count,
        0,
    )
    closed_trade_cost_coverage = (
        closed_trade_cost_observation_count / closed_entries
        if closed_entries > 0
        else 1.0
    )
    accepted = bool(registry.get("accepted"))
    reject_reason = _text(registry.get("reject_reason"))
    current_ack_present = _bool(registry.get("current_ack_present")) is True
    current_tracker_present = _bool(registry.get("current_tracker_present")) is True
    current_runtime_eligible = _bool(registry.get("current_runtime_eligible")) is True
    paper_tracker_created = current_tracker_present
    paper_tracker_effective = current_tracker_present
    historical_evidence_present = bool(runs) or bool(daily)
    historical_closed_trade_count = len(runs)
    proposal_id = _text(registry.get("proposal_id"))
    proposal_hash = _text(registry.get("proposal_hash"))
    current_contract_closed_trade_count = sum(
        1
        for run in runs
        if _text(run.get("proposal_id")) == proposal_id
        and (
            not _text(run.get("proposal_hash"))
            or _text(run.get("proposal_hash")) == proposal_hash
        )
    )
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
        cost_trusted=False,
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
    cost_match_status = _text(strategy_cost_trust.get("__match_status")) or "MISSING"
    dimensional_cost_trust_matched = cost_match_status == "UNIQUE"
    strategy_cost_trust_level = (
        _text(strategy_cost_trust.get("cost_trust_level"))
        if dimensional_cost_trust_matched
        else "NOT_EVALUATED"
    ) or "NOT_EVALUATED"
    dimensional_paper_usable = dimensional_cost_trust_matched and (
        _bool(strategy_cost_trust.get("paper_cost_usable"))
        if strategy_cost_trust.get("paper_cost_usable") not in (None, "")
        else _cost_trust_rank(strategy_cost_trust_level) >= _cost_trust_rank("PAPER_ONLY")
    )
    dimensional_canary_usable = dimensional_cost_trust_matched and _bool(
        strategy_cost_trust.get("canary_cost_usable")
    )
    dimensional_scale_usable = dimensional_cost_trust_matched and _bool(
        strategy_cost_trust.get("live_cost_usable")
    )
    closed_cost_complete = closed_trade_cost_missing_count == 0
    effective_cost_trusted = (
        paper_cost_model_usable
        and closed_cost_complete
        and dimensional_paper_usable
    )
    cost_trusted_for_canary = (
        effective_cost_trusted
        and dimensional_canary_usable
        and bool(closed_cost_sources)
        and closed_cost_sources <= CANARY_COST_SOURCES
    )
    cost_trusted_for_scale = (
        cost_trusted_for_canary
        and dimensional_scale_usable
        and closed_cost_sources <= SCALE_COST_SOURCES
    )
    block_reasons = [
        reason for reason in block_reasons if reason != "cost_not_trusted_for_paper"
    ]
    if not effective_cost_trusted:
        block_reasons.append("cost_not_trusted_for_paper")
    if cost_match_status == "MISSING":
        block_reasons.append("cost_trust_row_missing")
    elif cost_match_status == "AMBIGUOUS":
        block_reasons.append("cost_trust_row_ambiguous")
    if closed_trade_cost_missing_count > 0:
        block_reasons.append("closed_trade_cost_source_missing")
    block_reasons.extend(
        f"cost_source_data_quality_error:{error}" for error in cost_source_errors
    )
    identity_conflict = _bool(registry.get("identity_conflict"))
    if identity_conflict:
        block_reasons.append("canonical_identity_conflict")
    if not current_runtime_eligible:
        block_reasons.append("current_runtime_not_eligible")
    block_reasons = sorted(set(block_reasons))
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
    if cost_match_status == "MISSING":
        cost_evidence_status = "COST_TRUST_ROW_MISSING"
    elif cost_match_status == "AMBIGUOUS":
        cost_evidence_status = "COST_TRUST_ROW_AMBIGUOUS"
    elif closed_trade_cost_missing_count > 0:
        cost_evidence_status = "CLOSED_TRADE_COST_SOURCE_MISSING"
    elif cost_source_errors:
        cost_evidence_status = "COST_SOURCE_DATA_QUALITY_ERROR"
    else:
        cost_evidence_status = "PASS" if effective_cost_trusted else "COST_NOT_USABLE"
    event_ids = [_paper_event_id(row) for row in runs if _paper_event_id(row)]
    independence_weight = sum(
        1.0 / max(len(event_variants.get(event_id, set())), 1)
        for event_id in event_ids
    )
    horizon_variant_count = max(
        (len(event_variants.get(event_id, set())) for event_id in event_ids),
        default=0,
    )
    mfe_values = _numeric_run_values(runs, "mfe_bps", "max_favorable_excursion")
    mae_values = _numeric_run_values(runs, "mae_bps", "max_adverse_excursion")
    giveback_values = _numeric_run_values(runs, "profit_giveback_bps")
    efficiency_values = _numeric_run_values(runs, "exit_efficiency")
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
        "current_ack_present": current_ack_present,
        "current_tracker_present": current_tracker_present,
        "current_runtime_eligible": current_runtime_eligible,
        "historical_evidence_present": historical_evidence_present,
        "historical_closed_trade_count": historical_closed_trade_count,
        "current_contract_closed_trade_count": current_contract_closed_trade_count,
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
        "paper_cost_model_usable": paper_cost_model_usable,
        "closed_trade_cost_observation_count": closed_trade_cost_observation_count,
        "closed_trade_cost_missing_count": closed_trade_cost_missing_count,
        "closed_trade_cost_coverage": closed_trade_cost_coverage,
        "closed_trade_cost_sources": safe_json_dumps(sorted(closed_cost_sources)),
        "dimensional_cost_trust_level": strategy_cost_trust_level,
        "dimensional_cost_trust_matched": dimensional_cost_trust_matched,
        "strategy_cost_trust_level": strategy_cost_trust_level,
        "required_cost_trust_level": required_cost_trust_level,
        "cost_trusted_for_paper": effective_cost_trusted,
        "cost_trusted_for_canary": cost_trusted_for_canary,
        "cost_trusted_for_scale": cost_trusted_for_scale,
        "raw_closed_trade_count": len(runs),
        "independent_closed_trade_count": len(set(event_ids)),
        "canonical_opportunity_id": safe_json_dumps(sorted(set(event_ids))),
        "shared_entry_event_id": safe_json_dumps(
            sorted(
                event_id
                for event_id in set(event_ids)
                if len(event_variants.get(event_id, set())) > 1
            )
        ),
        "evidence_independence_weight": independence_weight,
        "horizon_variant_count": horizon_variant_count,
        "win_rate": _hit_rate(values),
        "avg_mfe_bps": _mean(mfe_values),
        "avg_mae_bps": _mean(mae_values),
        "avg_profit_giveback_bps": _mean(giveback_values),
        "avg_exit_efficiency": _mean(efficiency_values),
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
        "identity_conflict": identity_conflict,
        "identity_conflict_fields": _text(registry.get("identity_conflict_fields"))
        or "[]",
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
    priorities = (
        (
            "proposal_id",
            lambda row: _text(row.get("proposal_id"))
            == _text(registry.get("proposal_id")),
        ),
        (
            "proposal_hash",
            lambda row: _text(row.get("proposal_hash"))
            == _text(registry.get("proposal_hash")),
        ),
        (
            "strategy_id_version",
            lambda row: (
                _text(row.get("strategy_id")) == _text(registry.get("strategy_id"))
                and _text(row.get("strategy_version"))
                == _text(registry.get("strategy_version"))
            ),
        ),
    )
    for match_level, predicate in priorities:
        expected_present = {
            "proposal_id": bool(_text(registry.get("proposal_id"))),
            "proposal_hash": bool(_text(registry.get("proposal_hash"))),
            "strategy_id_version": bool(
                _text(registry.get("strategy_id"))
                and _text(registry.get("strategy_version"))
            ),
        }[match_level]
        if not expected_present:
            continue
        matches = [row for row in rows if predicate(row)]
        if not matches:
            continue
        if len(matches) != 1:
            return {"__match_status": "AMBIGUOUS", "__match_level": match_level}
        return {
            **dict(matches[0]),
            "__match_status": "UNIQUE",
            "__match_level": match_level,
        }
    return {"__match_status": "MISSING", "__match_level": "none"}


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
    if aliases - {""}:
        return False
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


def _current_activation_at(
    ack: Mapping[str, Any],
    tracker: Mapping[str, Any],
) -> str:
    values = [
        _text(ack.get("accepted_at") or ack.get("ingest_ts")),
        _text(tracker.get("created_at") or tracker.get("updated_at")),
    ]
    present = [value for value in values if value]
    if not present:
        return ""
    return max(present, key=_timestamp_sort_value)


def _timestamp_sort_value(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


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


def parse_cost_source_mix(value: Any) -> tuple[set[str], list[str]]:
    if value in (None, ""):
        return set(), []
    payload = value
    if isinstance(value, str):
        text = value.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            if text.startswith(("{", "[")):
                return set(), ["invalid_json"]
            return (
                {
                    part.strip().lower()
                    for part in text.replace(";", ",").split(",")
                    if part.strip()
                },
                [],
            )
    sources: set[str] = set()
    errors: list[str] = []
    if isinstance(payload, Mapping):
        if "cost_source" in payload or "source" in payload:
            items: list[Any] = [payload]
        else:
            items = [
                {"cost_source": source, "count": count}
                for source, count in payload.items()
            ]
    elif isinstance(payload, list):
        items = payload
    else:
        items = [payload]
    for item in items:
        count: Any = 1
        if isinstance(item, dict):
            source = item.get("cost_source") or item.get("source")
            count = item.get("count", 1)
        else:
            source = item
        text = str(source or "").strip().lower()
        if not text:
            continue
        try:
            numeric = float(count)
            parsed_count = int(numeric)
            if numeric != parsed_count:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"invalid_count:{text}")
            continue
        if parsed_count < 0:
            errors.append(f"negative_count:{text}")
            continue
        if parsed_count > 0:
            sources.add(text)
    return sources, sorted(set(errors))


def _cost_sources(value: Any) -> set[str]:
    return parse_cost_source_mix(value)[0]


def _paper_event_id(row: Mapping[str, Any]) -> str:
    explicit = _text(
        row.get("canonical_opportunity_id") or row.get("shared_entry_event_id")
    )
    if explicit:
        return explicit
    symbol = _symbol(row)
    entry_ts = _text(
        row.get("entry_signal_ts")
        or row.get("entry_decision_ts")
        or row.get("signal_ts")
    )
    if entry_ts:
        material = f"{symbol or 'UNKNOWN'}|{entry_ts}"
        return f"paper-event:{hashlib.sha256(material.encode()).hexdigest()[:24]}"
    return _text(row.get("paper_trade_id"))


def _event_variant_counts(
    rows: list[Mapping[str, Any]],
) -> dict[str, set[str]]:
    variants: dict[str, set[str]] = {}
    for row in rows:
        event_id = _paper_event_id(row)
        proposal_id = _text(row.get("proposal_id"))
        if event_id and proposal_id:
            variants.setdefault(event_id, set()).add(proposal_id)
    return variants


def _numeric_run_values(
    rows: list[Mapping[str, Any]],
    *fields: str,
) -> list[float]:
    output: list[float] = []
    for row in rows:
        value = next(
            (
                _float(row.get(field))
                for field in fields
                if _float(row.get(field)) is not None
            ),
            None,
        )
        if value is not None:
            output.append(value)
    return output


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
