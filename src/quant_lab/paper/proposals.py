from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.paper.contracts import PaperRule, PaperStrategyProposal

DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "paper_strategy_proposals.yaml"
)

PAPER_STRATEGY_MIGRATION_AUDIT_SCHEMA = {
    "legacy_row_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon_hours": pl.Int64,
    "legacy_decision": pl.Utf8,
    "legacy_recommended_mode": pl.Utf8,
    "migration_status": pl.Utf8,
    "migration_reason": pl.Utf8,
    "canonical_proposal_id": pl.Utf8,
    "canonical_proposal_hash": pl.Utf8,
    "lifecycle_state": pl.Utf8,
    "created_at": pl.Utf8,
    "schema_version": pl.Utf8,
}


class ProposalTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str
    strategy_version: str
    strategy_family: str
    symbol: str
    timeframe: str
    direction: str = "long"
    source_candidates: list[str] = Field(min_length=1)
    source_horizon_hours: int = Field(ge=1)
    entry_rule: PaperRule
    exit_rule: PaperRule
    max_holding_bars: int = Field(ge=1)
    min_holding_bars: int = Field(default=0, ge=0)
    cooldown_bars: int = Field(default=0, ge=0)
    signal_confirmation_bars: int = Field(default=1, ge=1)
    cost_quantile: str = "p75"
    minimum_expected_edge_bps: float = 0.0
    paper_notional_usdt: float = Field(default=20.0, gt=0)
    required_market_fields: list[str] = Field(default_factory=list)
    required_cost_trust_level: str = "PAPER_ONLY"
    expires_after_days: int = Field(default=30, ge=1, le=365)


class ProposalTemplateFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    proposals: list[ProposalTemplate]


def load_proposal_templates(path: str | Path | None = None) -> list[ProposalTemplate]:
    source = Path(path) if path is not None else DEFAULT_TEMPLATE_PATH
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return ProposalTemplateFile.model_validate(payload).proposals


def build_configured_proposals(
    evidence: pl.DataFrame,
    *,
    templates_path: str | Path | None = None,
    created_at: datetime | None = None,
    source_pack_sha256: str = "",
    source_dataset_versions: dict[str, str] | None = None,
) -> list[tuple[PaperStrategyProposal, dict[str, Any]]]:
    if evidence.is_empty():
        return []
    created = (created_at or datetime.now(UTC)).astimezone(UTC)
    evidence_rows = evidence.to_dicts()
    proposals: list[tuple[PaperStrategyProposal, dict[str, Any]]] = []
    for template in load_proposal_templates(templates_path):
        matching = [row for row in evidence_rows if _matches_template(row, template)]
        if not matching:
            continue
        best = max(matching, key=_evidence_rank)
        proposal = PaperStrategyProposal(
            strategy_id=template.strategy_id,
            strategy_version=template.strategy_version,
            strategy_family=template.strategy_family,
            symbol=template.symbol,
            timeframe=template.timeframe,
            direction=template.direction,
            entry_rule=template.entry_rule,
            exit_rule=template.exit_rule,
            max_holding_bars=template.max_holding_bars,
            min_holding_bars=template.min_holding_bars,
            cooldown_bars=template.cooldown_bars,
            signal_confirmation_bars=template.signal_confirmation_bars,
            cost_quantile=template.cost_quantile,
            minimum_expected_edge_bps=template.minimum_expected_edge_bps,
            paper_notional_usdt=template.paper_notional_usdt,
            created_at=created,
            expires_at=created + timedelta(days=template.expires_after_days),
            source_pack_sha256=source_pack_sha256,
            source_dataset_versions=source_dataset_versions
            or {"alpha_discovery_board": str(best.get("schema_version") or "legacy")},
            required_market_fields=template.required_market_fields,
            required_cost_trust_level=template.required_cost_trust_level,
        )
        proposals.append((proposal, best))
    return proposals


def proposal_export_rows(
    proposals: list[tuple[PaperStrategyProposal, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proposal, evidence in proposals:
        model = proposal.model_dump(mode="json")
        rows.append(
            {
                **model,
                "entry_rule": _json(model["entry_rule"]),
                "exit_rule": _json(model["exit_rule"]),
                "source_dataset_versions": _json(model["source_dataset_versions"]),
                "required_market_fields": _json(model["required_market_fields"]),
                "blocked_reasons": _json(model["blocked_reasons"]),
                "next_required_actions": _json(model["next_required_actions"]),
                # Legacy fields remain readable while consumers migrate to the contract fields.
                "strategy_candidate": str(evidence.get("strategy_candidate") or ""),
                "entry_conditions": _json(model["entry_rule"]),
                "recommended_mode": "paper",
                "suggested_horizon": f"{proposal.max_holding_bars}h",
                "sample_count": evidence.get("sample_count"),
                "complete_sample_count": evidence.get("complete_sample_count"),
                "avg_net_bps": evidence.get("avg_net_bps"),
                "p25_net_bps": evidence.get("p25_net_bps"),
                "win_rate": evidence.get("win_rate"),
                "cost_source_mix": evidence.get("cost_source_mix"),
                "live_block_reason": _json(
                    ["paper_only", "v5_ack_required", "canary_disabled_by_default"]
                ),
                "required_paper_days": 14,
                "required_slippage_coverage": 0.8,
                "as_of_date": evidence.get("as_of_date"),
            }
        )
    return rows


def build_legacy_proposal_migration_audit(
    evidence: pl.DataFrame,
    proposals: list[tuple[PaperStrategyProposal, dict[str, Any]]],
    *,
    templates_path: str | Path | None = None,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    if evidence.is_empty():
        return pl.DataFrame(schema=PAPER_STRATEGY_MIGRATION_AUDIT_SCHEMA)
    created = (created_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    templates = load_proposal_templates(templates_path)
    proposal_by_strategy = {proposal.strategy_id: proposal for proposal, _row in proposals}
    selected_keys = {_legacy_identity(row): proposal.strategy_id for proposal, row in proposals}
    rows: list[dict[str, Any]] = []
    for source in evidence.to_dicts():
        decision = str(source.get("decision") or "").strip().upper()
        recommended = str(source.get("recommended_mode") or "").strip().lower()
        if decision not in {"PAPER_READY", "LIVE_SMALL_READY"} and recommended not in {
            "paper",
            "paper_only",
        }:
            continue
        candidate = str(source.get("strategy_candidate") or source.get("candidate_name") or "")
        symbol = str(source.get("symbol") or source.get("v5_symbol") or "")
        horizon = _int(source.get("horizon_hours") or source.get("suggested_horizon"))
        identity = _legacy_identity(source)
        selected_strategy = selected_keys.get(identity)
        matching_templates = [
            template for template in templates if _matches_template(source, template)
        ]
        canonical = proposal_by_strategy.get(selected_strategy or "") or next(
            (
                proposal_by_strategy.get(template.strategy_id)
                for template in matching_templates
                if proposal_by_strategy.get(template.strategy_id) is not None
            ),
            None,
        )
        if not candidate or not symbol or horizon <= 0:
            status = "INVALID_LEGACY_ROW"
            reason = "missing_strategy_candidate_symbol_or_horizon"
            lifecycle = "REJECTED"
        elif selected_strategy:
            status = "MIGRATED_TO_V1_CONTRACT"
            reason = "selected_first_batch_evidence"
            lifecycle = "PAPER_PROPOSAL_READY"
        elif canonical is not None:
            status = "DEDUPED_TO_CANONICAL_PROPOSAL"
            reason = "shared_strategy_family_event_evidence"
            lifecycle = "RESEARCH_ONLY"
        else:
            status = "NOT_SELECTED_FIRST_BATCH"
            reason = "outside_explicit_first_batch_allowlist"
            lifecycle = "RESEARCH_ONLY"
        row_id = hashlib.sha256(
            json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "legacy_row_id": row_id,
                "strategy_candidate": candidate,
                "symbol": symbol.upper().replace("/", "-"),
                "horizon_hours": horizon,
                "legacy_decision": decision,
                "legacy_recommended_mode": recommended,
                "migration_status": status,
                "migration_reason": reason,
                "canonical_proposal_id": canonical.proposal_id if canonical else "",
                "canonical_proposal_hash": canonical.proposal_hash if canonical else "",
                "lifecycle_state": lifecycle,
                "created_at": created,
                "schema_version": "paper_strategy_migration_audit.v1",
            }
        )
    if not rows:
        return pl.DataFrame(schema=PAPER_STRATEGY_MIGRATION_AUDIT_SCHEMA)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .cast(PAPER_STRATEGY_MIGRATION_AUDIT_SCHEMA, strict=False)
        .select(list(PAPER_STRATEGY_MIGRATION_AUDIT_SCHEMA))
        .unique(subset=["legacy_row_id"], keep="last", maintain_order=True)
    )


def _matches_template(row: dict[str, Any], template: ProposalTemplate) -> bool:
    symbol = str(row.get("symbol") or row.get("v5_symbol") or "").upper().replace("/", "-")
    expected_symbol = template.symbol.upper().replace("/", "-")
    if symbol != expected_symbol:
        return False
    horizon = _int(row.get("horizon_hours") or row.get("suggested_horizon"))
    if horizon != template.source_horizon_hours:
        return False
    candidate = str(row.get("strategy_candidate") or row.get("candidate_name") or "").lower()
    return any(source.lower() in candidate for source in template.source_candidates)


def _legacy_identity(row: dict[str, Any]) -> tuple[str, str, int]:
    candidate = str(row.get("strategy_candidate") or row.get("candidate_name") or "").lower()
    symbol = str(row.get("symbol") or row.get("v5_symbol") or "").upper().replace("/", "-")
    horizon = _int(row.get("horizon_hours") or row.get("suggested_horizon"))
    return candidate, symbol, horizon


def _evidence_rank(row: dict[str, Any]) -> tuple[int, float, float, float, str]:
    return (
        _int(row.get("complete_sample_count")),
        _float(row.get("p25_net_bps")),
        _float(row.get("avg_net_bps")),
        _float(row.get("win_rate")),
        str(row.get("as_of_date") or ""),
    )


def _int(value: Any) -> int:
    text = str(value or "").lower().removesuffix("h")
    try:
        return int(float(text))
    except ValueError:
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
