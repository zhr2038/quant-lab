from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.paper.contracts import (
    PAPER_STRATEGY_CONTRACT_VERSION,
    PaperStrategyAck,
    PaperStrategyProposal,
)

PAPER_STRATEGY_PROPOSAL_DATASET = Path("gold") / "paper_strategy_proposal"
PAPER_STRATEGY_PROPOSALS_CURRENT_DATASET = (
    Path("gold") / "paper_strategy_proposals_current"
)
PAPER_STRATEGY_PROPOSAL_SNAPSHOT_DATASET = (
    Path("gold") / "paper_strategy_proposal_snapshot"
)
PAPER_STRATEGY_ACK_DATASET = Path("silver") / "v5_paper_strategy_proposal_ack"
PAPER_STRATEGY_PROMOTION_DATASET = Path("gold") / "paper_strategy_promotion_gate"
STRATEGY_COST_TRUST_DATASET = Path("gold") / "strategy_cost_trust"
PROPOSAL_COMPILER_VERSION = "quant_lab.paper_strategy_proposal_compiler.v1"


def read_proposals(lake_root: str | Path) -> list[dict[str, Any]]:
    root = Path(lake_root)
    frame = read_parquet_dataset(root / PAPER_STRATEGY_PROPOSALS_CURRENT_DATASET)
    if frame.is_empty():
        frame = read_parquet_dataset(root / PAPER_STRATEGY_PROPOSAL_DATASET)
    return _json_rows(frame)


def read_proposal_snapshot(lake_root: str | Path) -> dict[str, Any]:
    root = Path(lake_root)
    proposals = read_proposals(root)
    frame = read_parquet_dataset(root / PAPER_STRATEGY_PROPOSAL_SNAPSHOT_DATASET)
    if not frame.is_empty():
        row = _json_row(frame.tail(1).to_dicts()[0])
        row["proposal_ids"] = _json_string_list(row.get("proposal_ids"))
        row["proposal_hashes"] = _json_string_list(row.get("proposal_hashes"))
        return row
    current = pl.DataFrame(proposals, infer_schema_length=None) if proposals else pl.DataFrame()
    _enriched, metadata = build_canonical_proposal_snapshot(
        current,
        source_quant_lab_commit=_git_commit_full(),
    )
    return metadata


def read_proposal_snapshot_payload(lake_root: str | Path) -> dict[str, Any]:
    metadata = read_proposal_snapshot(lake_root)
    return {**metadata, "proposals": read_proposals(lake_root)}


def publish_canonical_proposal_snapshot(
    lake_root: str | Path,
    proposals: pl.DataFrame,
    *,
    generated_at: datetime | None = None,
    source_quant_lab_commit: str | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    root = Path(lake_root)
    prior = read_parquet_dataset(root / PAPER_STRATEGY_PROPOSAL_SNAPSHOT_DATASET)
    enriched, metadata = build_canonical_proposal_snapshot(
        proposals,
        generated_at=generated_at,
        source_quant_lab_commit=source_quant_lab_commit or _git_commit_full(),
        prior_snapshot=prior,
    )
    write_parquet_dataset(enriched, root / PAPER_STRATEGY_PROPOSALS_CURRENT_DATASET)
    storage = dict(metadata)
    storage["proposal_ids"] = json.dumps(metadata["proposal_ids"], separators=(",", ":"))
    storage["proposal_hashes"] = json.dumps(
        metadata["proposal_hashes"], separators=(",", ":")
    )
    write_parquet_dataset(
        pl.DataFrame([storage], infer_schema_length=None),
        root / PAPER_STRATEGY_PROPOSAL_SNAPSHOT_DATASET,
    )
    return enriched, metadata


def build_canonical_proposal_snapshot(
    proposals: pl.DataFrame,
    *,
    generated_at: datetime | None = None,
    source_quant_lab_commit: str,
    prior_snapshot: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    generated = generated_at or datetime.now(UTC)
    generated = generated.astimezone(UTC) if generated.tzinfo else generated.replace(tzinfo=UTC)
    members = sorted(
        (
            str(row.get("proposal_id") or "").strip(),
            str(row.get("proposal_hash") or "").strip().lower(),
        )
        for row in (proposals.to_dicts() if not proposals.is_empty() else [])
        if str(row.get("proposal_id") or "").strip()
    )
    proposal_ids = [proposal_id for proposal_id, _proposal_hash in members]
    proposal_hashes = [proposal_hash for _proposal_id, proposal_hash in members]
    contract_versions = sorted(
        {
            str(row.get("contract_version") or "").strip()
            for row in (proposals.to_dicts() if not proposals.is_empty() else [])
        }
        - {""}
    )
    if len(contract_versions) > 1:
        raise ValueError("proposal_snapshot_contract_version_mismatch")
    proposal_contract_version = (
        contract_versions[0]
        if contract_versions
        else PAPER_STRATEGY_CONTRACT_VERSION
    )
    content_material = {
        "contract_version": proposal_contract_version,
        "proposal_ids": sorted(proposal_ids),
        "proposal_hashes": sorted(proposal_hashes),
        "proposal_count": len(members),
    }
    content_snapshot_sha = hashlib.sha256(
        json.dumps(
            content_material, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    content_snapshot_id = f"proposal-content-snapshot:{content_snapshot_sha[:24]}"
    material = {
        **content_material,
        "proposal_ids": proposal_ids,
        "proposal_hashes": proposal_hashes,
        "source_quant_lab_commit": str(source_quant_lab_commit or "").strip(),
    }
    snapshot_sha = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    snapshot_id = f"proposal-snapshot:{snapshot_sha[:24]}"
    generated_at_text = generated.isoformat()
    snapshot_generated_at = generated_at_text
    if prior_snapshot is not None and not prior_snapshot.is_empty():
        prior = prior_snapshot.tail(1).to_dicts()[0]
        prior_content_sha = str(
            prior.get("proposal_content_snapshot_sha256")
            or prior.get("proposal_snapshot_sha256")
            or ""
        ).lower()
        if prior_content_sha == content_snapshot_sha:
            snapshot_generated_at = str(prior.get("snapshot_generated_at") or "").strip()
            if not snapshot_generated_at:
                snapshot_generated_at = generated_at_text
    metadata = {
        "proposal_snapshot_id": snapshot_id,
        "proposal_snapshot_sha256": snapshot_sha,
        "proposal_content_snapshot_id": content_snapshot_id,
        "proposal_content_snapshot_sha256": content_snapshot_sha,
        "snapshot_generated_at": snapshot_generated_at,
        "generated_at": generated_at_text,
        "proposal_count": len(members),
        "proposal_ids": proposal_ids,
        "proposal_hashes": proposal_hashes,
        "source_quant_lab_commit": str(source_quant_lab_commit or "").strip(),
        "proposal_compiler_version": PROPOSAL_COMPILER_VERSION,
        "proposal_contract_version": proposal_contract_version,
        "quant_lab_contract_version": proposal_contract_version,
    }
    if proposals.is_empty():
        enriched = proposals.clone()
        for name in (
            "proposal_snapshot_id",
            "proposal_snapshot_sha256",
            "proposal_content_snapshot_id",
            "proposal_content_snapshot_sha256",
            "snapshot_generated_at",
            "source_quant_lab_commit",
            "proposal_compiler_version",
            "proposal_contract_version",
        ):
            enriched = enriched.with_columns(pl.lit(None, dtype=pl.Utf8).alias(name))
    else:
        enriched = proposals.with_columns(
            pl.lit(snapshot_id).alias("proposal_snapshot_id"),
            pl.lit(snapshot_sha).alias("proposal_snapshot_sha256"),
            pl.lit(content_snapshot_id).alias("proposal_content_snapshot_id"),
            pl.lit(content_snapshot_sha).alias(
                "proposal_content_snapshot_sha256"
            ),
            pl.lit(snapshot_generated_at).alias("snapshot_generated_at"),
            pl.lit(str(source_quant_lab_commit or "").strip()).alias(
                "source_quant_lab_commit"
            ),
            pl.lit(PROPOSAL_COMPILER_VERSION).alias("proposal_compiler_version"),
            pl.lit(proposal_contract_version).alias("proposal_contract_version"),
        )
    return enriched, metadata


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
    current_snapshot_id = str(proposal.get("proposal_snapshot_id") or "")
    current_snapshot_sha = str(proposal.get("proposal_snapshot_sha256") or "").lower()
    current_content_snapshot_id = str(
        proposal.get("proposal_content_snapshot_id") or ""
    )
    current_content_snapshot_sha = str(
        proposal.get("proposal_content_snapshot_sha256") or ""
    ).lower()
    if ack.source_proposal_snapshot_id and (
        ack.source_proposal_snapshot_id != current_snapshot_id
        or ack.source_proposal_snapshot_sha256.lower() != current_snapshot_sha
    ):
        raise ValueError("proposal_snapshot_mismatch")
    if ack.source_proposal_content_snapshot_id and (
        ack.source_proposal_content_snapshot_id != current_content_snapshot_id
        or ack.source_proposal_content_snapshot_sha256.lower()
        != current_content_snapshot_sha
    ):
        raise ValueError("proposal_content_snapshot_mismatch")
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
        proposal_row = _json_row(proposal)
        ack_row = _json_row(ack_by_id.get(proposal_id, {}))
        promotion_row = _json_row(promotion_by_id.get(proposal_id, {}))
        effective_row = promotion_row or proposal_row
        rows.append(
            {
                **proposal_row,
                "proposal_lifecycle_state": proposal_row.get("lifecycle_state"),
                "proposal_lifecycle_reason": proposal_row.get("lifecycle_reason"),
                "proposal_blocked_reasons": proposal_row.get("blocked_reasons", []),
                "proposal_next_required_actions": proposal_row.get(
                    "next_required_actions", []
                ),
                "lifecycle_state": effective_row.get("lifecycle_state"),
                "lifecycle_reason": effective_row.get("lifecycle_reason"),
                "blocked_reasons": effective_row.get("blocked_reasons", []),
                "next_required_actions": effective_row.get(
                    "next_required_actions", []
                ),
                "accepted": promotion_row.get("accepted", ack_row.get("accepted")),
                "rules_locked": promotion_row.get(
                    "rules_locked", ack_row.get("rules_locked")
                ),
                "paper_tracker_id": promotion_row.get("paper_tracker_id")
                or ack_row.get("paper_tracker_id")
                or ack_row.get("tracker_id"),
                "paper_ready": promotion_row.get("paper_ready", False),
                "status_source": (
                    "paper_strategy_promotion_gate"
                    if promotion_row
                    else "paper_strategy_proposal"
                ),
                "ack": ack_row,
                "promotion": promotion_row,
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


def _json_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _git_commit_full() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    value = result.stdout.strip().lower()
    return value if len(value) == 40 and all(char in "0123456789abcdef" for char in value) else ""
