from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.contracts.models import GateDecision, RiskAction, RiskPermission
from quant_lab.data.lake import read_parquet_dataset

CORE_ALPHA_ID = "v5.core.momentum"
ADVISORY_DECISIONS = {"KEEP_SHADOW", "REGIME_SHADOW", "PAPER_READY", "LIVE_SMALL_READY"}
SHADOW_DECISIONS = {"KEEP_SHADOW", "REGIME_SHADOW"}
PAPER_DECISIONS = {"PAPER_READY", "LIVE_SMALL_READY"}
LIVE_DECISIONS = {"LIVE_SMALL_READY"}


def build_risk_advisory_context(
    lake_root: str | Path,
    *,
    gate_decisions: list[GateDecision],
    data_health: dict[str, Any],
    cost_health: dict[str, Any],
    telemetry_reasons: list[str],
) -> dict[str, Any]:
    opportunity = _strategy_opportunity_context(Path(lake_root))
    core_status = _core_alpha_gate_status(gate_decisions)
    core_dead = core_status == "DEAD"
    data_reasons = _data_block_reasons(data_health)
    cost_reasons = _cost_block_reasons(cost_health)
    telemetry_block_reasons = [str(reason) for reason in telemetry_reasons if str(reason)]
    safe_for_advisory = (
        not data_reasons
        and not _hard_cost_reasons(cost_reasons)
        and not telemetry_block_reasons
    )

    allowed_advisory_modes: list[str] = []
    if safe_for_advisory and opportunity["strategy_opportunities_available"]:
        if opportunity["shadow_available"] or opportunity["paper_available"]:
            allowed_advisory_modes.append("shadow")
        if opportunity["paper_available"]:
            allowed_advisory_modes.append("paper")

    live_block_reasons: list[str] = []
    live_block_reasons.extend(data_reasons)
    live_block_reasons.extend(cost_reasons)
    live_block_reasons.extend(telemetry_block_reasons)
    if core_dead:
        live_block_reasons.append("core_alpha_dead")
    if not opportunity["live_small_ready_available"]:
        live_block_reasons.append("no_strategy_live_small_ready")

    if not safe_for_advisory:
        system_status = "BLOCKED"
    elif opportunity["live_small_ready_available"] and not core_dead and not live_block_reasons:
        system_status = "SAFE_FOR_LIVE_SMALL_REVIEW"
    else:
        system_status = "SAFE_FOR_ADVISORY"

    return {
        "system_safety_status": system_status,
        "core_alpha_gate_status": core_status,
        "core_alpha_dead": core_dead,
        "strategy_opportunities_available": opportunity["strategy_opportunities_available"],
        "allowed_advisory_modes": allowed_advisory_modes,
        "base_allowed_live_modes": ["live_small"]
        if system_status == "SAFE_FOR_LIVE_SMALL_REVIEW"
        else [],
        "base_live_block_reasons": _dedupe(live_block_reasons),
    }


def apply_risk_advisory_context(
    permission: RiskPermission,
    context: dict[str, Any],
) -> RiskPermission:
    live_block_reasons = [
        *list(permission.live_block_reasons or []),
        *list(context.get("base_live_block_reasons") or []),
    ]
    allowed_live_modes = list(context.get("base_allowed_live_modes") or [])
    if permission.permission != RiskAction.ALLOW:
        live_block_reasons.append("global_permission_not_allow")
        allowed_live_modes = []
    if permission.permission == RiskAction.ALLOW and "live_small" not in allowed_live_modes:
        allowed_live_modes = []
    return permission.model_copy(
        update={
            "system_safety_status": str(context.get("system_safety_status") or "UNKNOWN"),
            "core_alpha_gate_status": str(context.get("core_alpha_gate_status") or "UNKNOWN"),
            "core_alpha_dead": bool(context.get("core_alpha_dead")),
            "strategy_opportunities_available": bool(
                context.get("strategy_opportunities_available")
            ),
            "allowed_advisory_modes": list(context.get("allowed_advisory_modes") or []),
            "allowed_live_modes": allowed_live_modes,
            "live_block_reasons": _dedupe(live_block_reasons),
        }
    )


def _strategy_opportunity_context(root: Path) -> dict[str, Any]:
    board = read_parquet_dataset(root / "gold" / "alpha_discovery_board")
    if board.is_empty():
        board = read_parquet_dataset(root / "gold" / "strategy_evidence")
    decisions = _latest_decisions(board)
    return {
        "strategy_opportunities_available": bool(decisions & ADVISORY_DECISIONS),
        "shadow_available": bool(decisions & SHADOW_DECISIONS),
        "paper_available": bool(decisions & PAPER_DECISIONS),
        "live_small_ready_available": bool(decisions & LIVE_DECISIONS),
        "decisions": sorted(decisions),
    }


def _latest_decisions(frame: pl.DataFrame) -> set[str]:
    if frame.is_empty() or "decision" not in frame.columns:
        return set()
    rows = frame.to_dicts()
    latest_day = max(
        (
            str(row.get("as_of_date") or "")[:10]
            for row in rows
            if str(row.get("as_of_date") or "").strip()
        ),
        default=None,
    )
    if latest_day:
        rows = [row for row in rows if str(row.get("as_of_date") or "")[:10] == latest_day]
    return {str(row.get("decision") or "").strip().upper() for row in rows}


def _core_alpha_gate_status(gate_decisions: list[GateDecision]) -> str:
    core = [decision for decision in gate_decisions if decision.alpha_id == CORE_ALPHA_ID]
    if not core:
        return "UNKNOWN"
    latest = max(core, key=lambda decision: decision.created_at)
    return latest.status.value


def _data_block_reasons(data_health: dict[str, Any]) -> list[str]:
    if not (
        bool(data_health.get("is_critical"))
        or str(data_health.get("status") or "").lower() == "critical"
    ):
        return []
    reasons = data_health.get("reasons")
    if isinstance(reasons, list):
        return [str(reason) for reason in reasons if str(reason)]
    return ["data_health_critical"]


def _cost_block_reasons(cost_health: dict[str, Any]) -> list[str]:
    status = str(cost_health.get("status") or "").lower()
    reasons: list[str] = []
    if not cost_health or bool(cost_health.get("missing")) or status in {"missing", "critical"}:
        reasons.append("cost_health_missing_or_critical")
    if bool(cost_health.get("stale")) or status == "stale":
        reasons.append("cost_health_stale")
    if bool(cost_health.get("high_fallback")):
        reasons.append("cost_health_high_fallback")
    return reasons


def _hard_cost_reasons(reasons: list[str]) -> list[str]:
    return [
        reason
        for reason in reasons
        if reason in {"cost_health_missing_or_critical", "cost_health_stale"}
    ]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return json_list(parsed)
    return [str(value)]
