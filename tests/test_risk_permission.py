from datetime import UTC, datetime

from quant_lab.contracts.models import GateDecision, GateStatus, RiskAction
from quant_lab.risk.permissions import evaluate_live_permission

CREATED_AT = datetime(2026, 5, 10, tzinfo=UTC)


def gate(status: GateStatus, alpha_id: str = "alpha-test") -> GateDecision:
    return GateDecision(
        alpha_id=alpha_id,
        version="v1",
        gate_version="default-v0.1",
        status=status,
        passed=status == GateStatus.LIVE_READY,
        reasons=["test_gate_status"],
        metrics={"ic_tstat": 3.0},
        next_action="test_next_action",
        created_at=CREATED_AT,
    )


def test_critical_data_health_aborts():
    permission = evaluate_live_permission(
        strategy="v5",
        version="v1",
        gate_decisions=[gate(GateStatus.LIVE_READY)],
        cost_health={"status": "ok", "cost_model_version": "costs-v1"},
        data_health={"status": "critical"},
    )

    assert permission.permission == RiskAction.ABORT
    assert permission.allowed_modes == []
    assert permission.max_gross_exposure == 0
    assert permission.reasons == ["data_health_critical"]


def test_stale_cost_health_is_sell_only():
    permission = evaluate_live_permission(
        strategy="v5",
        version="v1",
        gate_decisions=[gate(GateStatus.LIVE_READY)],
        cost_health={"status": "stale", "cost_model_version": "costs-v1"},
        data_health={"status": "ok"},
    )

    assert permission.permission == RiskAction.SELL_ONLY
    assert permission.allowed_modes == ["sell_only"]
    assert permission.reasons == ["cost_health_stale"]


def test_missing_cost_health_is_sell_only():
    permission = evaluate_live_permission(
        strategy="v5",
        version="v1",
        gate_decisions=[gate(GateStatus.LIVE_READY)],
        cost_health=None,
        data_health={"status": "ok"},
    )

    assert permission.permission == RiskAction.SELL_ONLY
    assert permission.reasons == ["cost_health_missing"]


def test_public_spread_proxy_only_cost_is_sell_only_not_live_allow():
    permission = evaluate_live_permission(
        strategy="v5",
        version="v1",
        gate_decisions=[gate(GateStatus.LIVE_READY)],
        cost_health={
            "status": "warning",
            "cost_model_version": "cost_bucket_daily:2026-05-12",
            "fallback_ratio": 1.0,
            "high_fallback": True,
        },
        data_health={"status": "ok"},
    )

    assert permission.permission == RiskAction.SELL_ONLY
    assert permission.reasons == ["cost_health_high_fallback"]


def test_dead_gate_aborts():
    permission = evaluate_live_permission(
        strategy="v5",
        version="v1",
        gate_decisions=[gate(GateStatus.DEAD)],
        cost_health={"status": "ok", "cost_model_version": "costs-v1"},
        data_health={"status": "ok"},
    )

    assert permission.permission == RiskAction.ABORT
    assert permission.allowed_modes == []
    assert permission.reasons == ["required_alpha_gate_dead"]


def test_quarantine_gate_is_sell_only():
    permission = evaluate_live_permission(
        strategy="v5",
        version="v1",
        gate_decisions=[gate(GateStatus.QUARANTINE)],
        cost_health={"status": "ok", "cost_model_version": "costs-v1"},
        data_health={"status": "ok"},
    )

    assert permission.permission == RiskAction.SELL_ONLY
    assert permission.allowed_modes == ["sell_only"]
    assert permission.reasons == ["required_alpha_gate_quarantine"]


def test_paper_ready_allows_paper_only():
    permission = evaluate_live_permission(
        strategy="v5",
        version="v1",
        gate_decisions=[gate(GateStatus.PAPER_READY)],
        cost_health={"status": "ok", "cost_model_version": "costs-v1"},
        data_health={"status": "ok", "allowed_modes": ["paper", "live_canary"]},
    )

    assert permission.permission == RiskAction.ALLOW
    assert permission.allowed_modes == ["paper"]
    assert permission.max_gross_exposure == 0
    assert permission.reasons == ["required_alpha_gate_paper_ready"]


def test_live_ready_does_not_emit_live_modes():
    permission = evaluate_live_permission(
        strategy="v5",
        version="v1",
        gate_decisions=[gate(GateStatus.LIVE_READY)],
        cost_health={"status": "ok", "cost_model_version": "costs-v1"},
        data_health={
            "status": "ok",
            "allowed_modes": ["paper", "live_canary"],
            "max_gross_exposure": 0.12,
            "max_single_weight": 0.03,
        },
    )

    assert permission.permission == RiskAction.ALLOW
    assert permission.allowed_modes == ["paper"]
    assert permission.max_gross_exposure == 0
    assert permission.max_single_weight == 0
    assert permission.cost_model_version == "costs-v1"
    assert permission.gate_version == "default-v0.1"
    assert permission.created_at == CREATED_AT
    assert permission.reasons == ["all_required_alpha_gates_live_ready"]
