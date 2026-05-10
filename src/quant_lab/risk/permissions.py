from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from quant_lab.contracts.models import GateDecision, GateStatus, RiskAction, RiskPermission

DEFAULT_LIVE_ALLOWED_MODES = ["paper", "live_canary"]
PAPER_ONLY_ALLOWED_MODES = ["paper"]
SELL_ONLY_ALLOWED_MODES = ["sell_only"]


def evaluate_live_permission(
    strategy: str,
    version: str,
    gate_decisions: Sequence[GateDecision],
    cost_health: Mapping[str, Any] | None,
    data_health: Mapping[str, Any] | None,
) -> RiskPermission:
    decisions = list(gate_decisions)
    created_at = _created_at(decisions)
    cost_model_version = _health_value(cost_health, "cost_model_version", default="unknown")
    gate_version = _gate_version(decisions)

    if _is_data_critical(data_health):
        return _permission(
            strategy=strategy,
            version=version,
            permission=RiskAction.ABORT,
            allowed_modes=[],
            reasons=["data_health_critical"],
            cost_model_version=cost_model_version,
            gate_version=gate_version,
            created_at=created_at,
        )

    if not decisions:
        return _permission(
            strategy=strategy,
            version=version,
            permission=RiskAction.ABORT,
            allowed_modes=[],
            reasons=["no_required_gate_decisions"],
            cost_model_version=cost_model_version,
            gate_version=gate_version,
            created_at=created_at,
        )

    if any(decision.status == GateStatus.DEAD for decision in decisions):
        return _permission(
            strategy=strategy,
            version=version,
            permission=RiskAction.ABORT,
            allowed_modes=[],
            reasons=["required_alpha_gate_dead"],
            cost_model_version=cost_model_version,
            gate_version=gate_version,
            created_at=created_at,
        )

    cost_reasons = _cost_health_reasons(cost_health)
    if cost_reasons:
        return _permission(
            strategy=strategy,
            version=version,
            permission=RiskAction.SELL_ONLY,
            allowed_modes=SELL_ONLY_ALLOWED_MODES,
            reasons=cost_reasons,
            cost_model_version=cost_model_version,
            gate_version=gate_version,
            created_at=created_at,
        )

    if any(decision.status == GateStatus.QUARANTINE for decision in decisions):
        return _permission(
            strategy=strategy,
            version=version,
            permission=RiskAction.SELL_ONLY,
            allowed_modes=SELL_ONLY_ALLOWED_MODES,
            reasons=["required_alpha_gate_quarantine"],
            cost_model_version=cost_model_version,
            gate_version=gate_version,
            created_at=created_at,
        )

    if any(decision.status == GateStatus.PAPER_READY for decision in decisions):
        return _permission(
            strategy=strategy,
            version=version,
            permission=RiskAction.ALLOW,
            allowed_modes=PAPER_ONLY_ALLOWED_MODES,
            reasons=["required_alpha_gate_paper_ready"],
            cost_model_version=cost_model_version,
            gate_version=gate_version,
            created_at=created_at,
        )

    return _permission(
        strategy=strategy,
        version=version,
        permission=RiskAction.ALLOW,
        allowed_modes=_configured_live_modes(data_health),
        reasons=["all_required_alpha_gates_live_ready"],
        cost_model_version=cost_model_version,
        gate_version=gate_version,
        created_at=created_at,
        max_gross_exposure=_health_float(data_health, "max_gross_exposure", default=0.25),
        max_single_weight=_health_float(data_health, "max_single_weight", default=0.05),
    )


def _permission(
    *,
    strategy: str,
    version: str,
    permission: RiskAction,
    allowed_modes: Sequence[str],
    reasons: Sequence[str],
    cost_model_version: str,
    gate_version: str,
    created_at: datetime,
    max_gross_exposure: float = 0.0,
    max_single_weight: float = 0.0,
) -> RiskPermission:
    return RiskPermission(
        strategy=strategy,
        version=version,
        permission=permission,
        allowed_modes=list(allowed_modes),
        max_gross_exposure=max_gross_exposure,
        max_single_weight=max_single_weight,
        cost_model_version=cost_model_version,
        gate_version=gate_version,
        reasons=list(reasons),
        created_at=created_at,
    )


def _is_data_critical(data_health: Mapping[str, Any] | None) -> bool:
    if not data_health:
        return False
    if bool(data_health.get("is_critical")):
        return True
    status = str(data_health.get("status") or data_health.get("level") or "").lower()
    return status == "critical"


def _cost_health_reasons(cost_health: Mapping[str, Any] | None) -> list[str]:
    if not cost_health:
        return ["cost_health_missing"]
    status = str(cost_health.get("status") or "").lower()
    reasons: list[str] = []
    if status == "missing" or bool(cost_health.get("missing")):
        reasons.append("cost_health_missing")
    if status == "stale" or bool(cost_health.get("stale")) or bool(cost_health.get("is_stale")):
        reasons.append("cost_health_stale")
    return reasons


def _configured_live_modes(data_health: Mapping[str, Any] | None) -> list[str]:
    if not data_health:
        return list(DEFAULT_LIVE_ALLOWED_MODES)
    modes = data_health.get("allowed_modes")
    if not isinstance(modes, Sequence) or isinstance(modes, str):
        return list(DEFAULT_LIVE_ALLOWED_MODES)
    normalized = [str(mode) for mode in modes if str(mode)]
    return normalized or list(DEFAULT_LIVE_ALLOWED_MODES)


def _health_value(
    health: Mapping[str, Any] | None,
    key: str,
    *,
    default: str,
) -> str:
    if not health:
        return default
    value = health.get(key)
    if value is None:
        return default
    return str(value)


def _health_float(
    health: Mapping[str, Any] | None,
    key: str,
    *,
    default: float,
) -> float:
    if not health or health.get(key) is None:
        return default
    return float(health[key])


def _gate_version(gate_decisions: Sequence[GateDecision]) -> str:
    versions = sorted({decision.gate_version for decision in gate_decisions})
    return "+".join(versions) if versions else "unknown"


def _created_at(gate_decisions: Sequence[GateDecision]) -> datetime:
    if not gate_decisions:
        return datetime.now(UTC)
    return max(decision.created_at for decision in gate_decisions)
