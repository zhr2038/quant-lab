from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.paper.contracts import PaperStrategyAck, PaperStrategyProposal

PAPER_STRATEGY_PROPOSAL_DATASET = Path("gold") / "paper_strategy_proposal"
PAPER_STRATEGY_ACK_DATASET = Path("silver") / "v5_paper_strategy_proposal_ack"
PAPER_STRATEGY_PROMOTION_DATASET = Path("gold") / "paper_strategy_promotion_gate"
STRATEGY_COST_TRUST_DATASET = Path("gold") / "strategy_cost_trust"


def read_proposals(lake_root: str | Path) -> list[dict[str, Any]]:
    frame = read_parquet_dataset(Path(lake_root) / PAPER_STRATEGY_PROPOSAL_DATASET)
    return _json_rows(frame)


def read_proposal(lake_root: str | Path, proposal_id: str) -> dict[str, Any] | None:
    for row in read_proposals(lake_root):
        if str(row.get("proposal_id") or "") == proposal_id:
            return row
    return None


def publish_proposals(
    lake_root: str | Path,
    proposals: list[PaperStrategyProposal],
) -> pl.DataFrame:
    existing = read_parquet_dataset(Path(lake_root) / PAPER_STRATEGY_PROPOSAL_DATASET)
    rows = existing.to_dicts() if not existing.is_empty() else []
    for proposal in proposals:
        identity = (proposal.strategy_id, proposal.strategy_version)
        same_version = [
            row
            for row in rows
            if (str(row.get("strategy_id") or ""), str(row.get("strategy_version") or ""))
            == identity
        ]
        if same_version:
            fingerprint = proposal_rule_fingerprint(proposal)
            compatible = [
                row for row in same_version if proposal_rule_fingerprint(row) == fingerprint
            ]
            if not compatible:
                raise ValueError(
                    "strategy_version_rule_conflict: "
                    f"{proposal.strategy_id}:{proposal.strategy_version}"
                )
            canonical = proposal_from_storage_row(compatible[0])
            canonical = canonical.model_copy(
                update={
                    "created_at": proposal.created_at,
                    "expires_at": proposal.expires_at,
                    "source_pack_sha256": proposal.source_pack_sha256,
                    "lifecycle_state": proposal.lifecycle_state,
                    "lifecycle_reason": proposal.lifecycle_reason,
                    "blocked_reasons": proposal.blocked_reasons,
                    "next_required_actions": proposal.next_required_actions,
                }
            )
            rows = [
                row
                for row in rows
                if (
                    str(row.get("strategy_id") or ""),
                    str(row.get("strategy_version") or ""),
                )
                != identity
            ]
            rows.append(_proposal_storage_row(canonical))
            continue
        rows.append(_proposal_storage_row(proposal))
    merged = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    write_parquet_dataset(merged, Path(lake_root) / PAPER_STRATEGY_PROPOSAL_DATASET)
    return merged


def proposal_from_storage_row(row: dict[str, Any]) -> PaperStrategyProposal:
    return PaperStrategyProposal.model_validate(_json_row(dict(row)))


def proposal_rule_fingerprint(value: PaperStrategyProposal | dict[str, Any]) -> str:
    proposal = (
        value if isinstance(value, PaperStrategyProposal) else proposal_from_storage_row(value)
    )
    payload = proposal.model_dump(mode="json")
    locked_fields = (
        "contract_version",
        "strategy_id",
        "strategy_version",
        "strategy_family",
        "symbol",
        "timeframe",
        "direction",
        "entry_rule",
        "exit_rule",
        "max_holding_bars",
        "min_holding_bars",
        "cooldown_bars",
        "signal_confirmation_bars",
        "cost_quantile",
        "minimum_expected_edge_bps",
        "paper_notional_usdt",
        "paper_only",
        "live_order_effect",
        "max_live_notional_usdt",
        "required_market_fields",
        "required_cost_trust_level",
    )
    canonical = {field: payload.get(field) for field in locked_fields}
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def record_ack(lake_root: str | Path, ack: PaperStrategyAck) -> tuple[dict[str, Any], bool]:
    root = Path(lake_root)
    proposal = read_proposal(root, ack.proposal_id)
    if proposal is None:
        raise ValueError("unknown_proposal_id")
    if str(proposal.get("proposal_hash") or "") != ack.proposal_hash:
        raise ValueError("proposal_hash_mismatch")
    frame = read_parquet_dataset(root / PAPER_STRATEGY_ACK_DATASET)
    rows = frame.to_dicts() if not frame.is_empty() else []
    for row in rows:
        if str(row.get("proposal_id") or "") != ack.proposal_id:
            continue
        if str(row.get("proposal_hash") or "") != ack.proposal_hash:
            raise ValueError("duplicate_version_conflict")
        return _json_row(row), True
    row = {
        **ack.model_dump(mode="json"),
        "paper_tracker_id": ack.tracker_id,
        "ingest_ts": datetime.now(UTC).isoformat(),
        "schema_version": "v5_paper_strategy_proposal_ack.v2",
    }
    rows.append(row)
    output = pl.DataFrame(rows, infer_schema_length=None)
    write_parquet_dataset(output, root / PAPER_STRATEGY_ACK_DATASET)
    return _json_row(row), False


def status_rows(lake_root: str | Path) -> list[dict[str, Any]]:
    root = Path(lake_root)
    proposals = read_parquet_dataset(root / PAPER_STRATEGY_PROPOSAL_DATASET)
    acks = read_parquet_dataset(root / PAPER_STRATEGY_ACK_DATASET)
    promotion = read_parquet_dataset(root / PAPER_STRATEGY_PROMOTION_DATASET)
    ack_by_id = _latest_by(acks, "proposal_id")
    promotion_by_id = _latest_by(promotion, "proposal_id")
    rows: list[dict[str, Any]] = []
    for proposal in proposals.to_dicts() if not proposals.is_empty() else []:
        proposal_id = str(proposal.get("proposal_id") or "")
        rows.append(
            {
                **_json_row(proposal),
                "ack": _json_row(ack_by_id.get(proposal_id, {})),
                "promotion": _json_row(promotion_by_id.get(proposal_id, {})),
            }
        )
    return rows


def promotion_rows(lake_root: str | Path) -> list[dict[str, Any]]:
    return _json_rows(
        read_parquet_dataset(Path(lake_root) / PAPER_STRATEGY_PROMOTION_DATASET)
    )


def cost_trust_rows(lake_root: str | Path, strategy_id: str) -> list[dict[str, Any]]:
    rows = _json_rows(read_parquet_dataset(Path(lake_root) / STRATEGY_COST_TRUST_DATASET))
    return [row for row in rows if str(row.get("strategy_id") or "") == strategy_id]


def canary_readiness(lake_root: str | Path) -> dict[str, Any]:
    promotion = promotion_rows(lake_root)
    ready = [row for row in promotion if bool(row.get("paper_ready"))]
    return {
        "canary_ready": False,
        "mode": "shadow",
        "reason": "canary_disabled_by_default",
        "paper_promotion_ready_count": len(ready),
        "required_permission_status": "ACTIVE_ALLOW",
        "required_cost_trust_level": "CANARY",
        "live_order_effect": "none",
    }


def _upsert_rows(
    existing: pl.DataFrame,
    incoming: pl.DataFrame,
    *,
    key_fields: tuple[str, ...],
) -> pl.DataFrame:
    rows: dict[tuple[str, ...], dict[str, Any]] = {}
    for frame in (existing, incoming):
        if frame.is_empty():
            continue
        for row in frame.to_dicts():
            key = tuple(str(row.get(field) or "") for field in key_fields)
            rows[key] = row
    return pl.DataFrame(list(rows.values()), infer_schema_length=None) if rows else pl.DataFrame()


def _latest_by(frame: pl.DataFrame, field: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if frame.is_empty():
        return latest
    for row in frame.to_dicts():
        key = str(row.get(field) or "")
        if key:
            latest[key] = row
    return latest


def _json_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return [_json_row(row) for row in (frame.to_dicts() if not frame.is_empty() else [])]


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        str(key): value.isoformat() if isinstance(value, datetime) else value
        for key, value in row.items()
    }
    for field in (
        "entry_rule",
        "exit_rule",
        "source_dataset_versions",
        "required_market_fields",
        "blocked_reasons",
        "next_required_actions",
    ):
        value = output.get(field)
        if not isinstance(value, str):
            continue
        try:
            output[field] = json.loads(value)
        except json.JSONDecodeError:
            pass
    return output


def _proposal_storage_row(proposal: PaperStrategyProposal) -> dict[str, Any]:
    row = proposal.model_dump(mode="json")
    for field in (
        "entry_rule",
        "exit_rule",
        "source_dataset_versions",
        "required_market_fields",
        "blocked_reasons",
        "next_required_actions",
    ):
        row[field] = json.dumps(
            row.get(field), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
    return row
