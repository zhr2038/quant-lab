from datetime import UTC, datetime

from quant_lab.contracts.models import RiskAction, RiskPermission, RiskPermissionStatus
from quant_lab.risk.advisory import apply_risk_advisory_context
from quant_lab.risk.publish import is_permission_status_enforceable


def _permission(*, status: RiskPermissionStatus | None) -> RiskPermission:
    return RiskPermission(
        strategy="v5",
        version="5.0.0",
        permission=RiskAction.ALLOW,
        allowed_modes=["paper", "live_canary"],
        max_gross_exposure=0.0,
        max_single_weight=0.0,
        cost_model_version="costs-test",
        gate_version="gate-test",
        reasons=[],
        created_at=datetime(2026, 6, 16, tzinfo=UTC),
        permission_status=status,
        enforceable=True,
    )


def test_permission_status_enforceable_requires_known_active_status() -> None:
    assert is_permission_status_enforceable(RiskPermissionStatus.ACTIVE_ALLOW)
    assert is_permission_status_enforceable("ACTIVE_SELL_ONLY")
    assert not is_permission_status_enforceable("ACTIVE_UNKNOWN")
    assert not is_permission_status_enforceable(None)


def test_advisory_context_requires_status_before_legacy_allowed_modes() -> None:
    result = apply_risk_advisory_context(
        _permission(status=None),
        {
            "system_safety_status": "SAFE_FOR_ADVISORY",
            "allowed_advisory_modes": ["shadow", "paper"],
            "base_live_block_reasons": [],
        },
    )

    assert result.allowed_modes == []


def test_advisory_context_keeps_non_live_modes_for_known_active_status() -> None:
    result = apply_risk_advisory_context(
        _permission(status=RiskPermissionStatus.ACTIVE_ALLOW),
        {
            "system_safety_status": "SAFE_FOR_ADVISORY",
            "allowed_advisory_modes": ["shadow", "paper"],
            "base_live_block_reasons": [],
        },
    )

    assert result.allowed_modes == ["paper", "shadow"]
    assert result.allowed_live_modes == []


def test_advisory_context_preserves_review_only_published_modes() -> None:
    permission = _permission(status=RiskPermissionStatus.ACTIVE_ABORT).model_copy(
        update={
            "permission": RiskAction.ABORT,
            "allowed_advisory_modes": ["micro_canary_review", "live_canary"],
        }
    )

    result = apply_risk_advisory_context(
        permission,
        {
            "system_safety_status": "SAFE_FOR_ADVISORY",
            "allowed_advisory_modes": ["shadow", "paper"],
            "base_live_block_reasons": [],
        },
    )

    assert result.allowed_advisory_modes == ["shadow", "paper", "micro_canary_review"]
    assert result.allowed_live_modes == []
