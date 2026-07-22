from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from quant_lab.research_plane.contracts import (
    RESEARCH_RECEIPT_ADAPTER,
    RESEARCH_RESULT_ADAPTER,
    RESEARCH_SNAPSHOT_ADAPTER,
    RESEARCH_TASK_ADAPTER,
    FactorResearchResultManifest,
    FactorResearchSnapshotManifest,
    FactorResearchTask,
    FactorResearchWorkerReceipt,
)

COMMIT = "a" * 40
SHA = "b" * 64
GENERATED_AT = datetime(2026, 7, 19, 1, 2, 3, tzinfo=UTC)
HYPOTHESIS_IDS = ("hyp.defensive-low-vol.v1",)
TRIAL_IDS = ("trial.structural-low-vol.20d.v1",)


def _snapshot_payload() -> dict[str, object]:
    return {
        "schema_version": "quant_lab_factor_research_snapshot.v1",
        "task_type": "factor_research",
        "snapshot_id": "factor-research-snapshot-test",
        "generated_at": GENERATED_AT.isoformat(),
        "quant_lab_commit": COMMIT,
        "selected_v5_bundle_id": f"v5-bundle-sha256:{SHA}",
        "factor_research_schema_version": "factor_research.v2",
        "hypothesis_registry_digest": SHA,
        "trial_ledger_digest": SHA,
        "source_input_digest": SHA,
        "as_of_date": "2026-07-19",
        "start_date": "2024-07-20",
        "end_date": "2026-07-19",
        "max_history_days": 1095,
        "hypothesis_ids": list(HYPOTHESIS_IDS),
        "trial_ids": list(TRIAL_IDS),
        "test_count": 1,
        "datasets": ["silver/market_bar"],
        "files": [],
        "total_input_bytes": 0,
        "total_input_rows": 0,
        "manifest_sha256": SHA,
        "signature_key_id": "cloud-research-v1",
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "signature": "test",
    }


def _task_payload() -> dict[str, object]:
    return {
        "schema_version": "quant_lab_factor_research_task.v1",
        "task_type": "factor_research",
        "task_id": "factor-research-task-test",
        "snapshot_id": "factor-research-snapshot-test",
        "as_of_date": "2026-07-19",
        "start_date": "2024-07-20",
        "end_date": "2026-07-19",
        "max_history_days": 1095,
        "hypothesis_ids": list(HYPOTHESIS_IDS),
        "trial_ids": list(TRIAL_IDS),
        "test_count": 1,
        "quant_lab_commit": COMMIT,
        "factor_research_schema_version": "factor_research.v2",
        "hypothesis_registry_digest": SHA,
        "trial_ledger_digest": SHA,
        "source_input_digest": SHA,
        "selected_v5_bundle_id": f"v5-bundle-sha256:{SHA}",
        "snapshot_manifest_sha256": SHA,
        "requested_at": (GENERATED_AT + timedelta(minutes=1)).isoformat(),
        "lease_seconds": 7200,
        "max_attempts": 3,
        "signature_key_id": "cloud-research-v1",
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "signature": "test",
    }


def _result_payload() -> dict[str, object]:
    return {
        "schema_version": "quant_lab_factor_research_result.v1",
        "task_type": "factor_research",
        "task_id": "factor-research-task-test",
        "snapshot_id": "factor-research-snapshot-test",
        "snapshot_manifest_sha256": SHA,
        "selected_v5_bundle_id": f"v5-bundle-sha256:{SHA}",
        "quant_lab_commit": COMMIT,
        "worker_commit": COMMIT,
        "factor_research_schema_version": "factor_research.v2",
        "hypothesis_registry_digest": SHA,
        "trial_ledger_digest": SHA,
        "source_input_digest": SHA,
        "as_of_date": "2026-07-19",
        "start_date": "2024-07-20",
        "end_date": "2026-07-19",
        "max_history_days": 1095,
        "hypothesis_ids": list(HYPOTHESIS_IDS),
        "trial_ids": list(TRIAL_IDS),
        "test_count": 1,
        "multiple_testing_family": "defensive.v1",
        "generation_id": "factor-generation-test",
        "generated_at": (GENERATED_AT + timedelta(hours=1)).isoformat(),
        "completed_at": (GENERATED_AT + timedelta(hours=2)).isoformat(),
        "outputs": [],
        "reports": [],
        "anti_leakage_status": "PASS",
        "anti_leakage_violation_count": 0,
        "warnings": [],
        "input_bytes": 0,
        "cache_hit_bytes": 0,
        "downloaded_bytes": 0,
        "output_bytes": 0,
        "peak_rss_bytes": 0,
        "compute_duration_seconds": 1.0,
        "worker_key_id": "nas-research-v1",
        "research_only": True,
        "requires_cloud_validation": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "signature": "test",
    }


def _receipt_payload() -> dict[str, object]:
    return {
        "schema_version": "quant_lab_factor_research_receipt.v1",
        "task_type": "factor_research",
        "task_id": "factor-research-task-test",
        "snapshot_id": "factor-research-snapshot-test",
        "worker_id": "nas-research-worker-01",
        "worker_commit": COMMIT,
        "state": "completed",
        "claimed_at": GENERATED_AT.isoformat(),
        "completed_at": (GENERATED_AT + timedelta(hours=2)).isoformat(),
        "result_manifest_sha256": SHA,
        "output_rows": 0,
        "input_bytes": 0,
        "downloaded_bytes": 0,
        "cache_hit_bytes": 0,
        "anti_leakage_status": "PASS",
        "anti_leakage_violation_count": 0,
        "error_code": None,
        "worker_key_id": "nas-research-v1",
        "research_only": True,
        "live_order_effect": "none",
        "signature": "test",
    }


def test_factor_research_discriminated_union_round_trip() -> None:
    snapshot = RESEARCH_SNAPSHOT_ADAPTER.validate_python(_snapshot_payload())
    task = RESEARCH_TASK_ADAPTER.validate_python(_task_payload())
    result = RESEARCH_RESULT_ADAPTER.validate_python(_result_payload())
    receipt = RESEARCH_RECEIPT_ADAPTER.validate_python(_receipt_payload())

    assert isinstance(snapshot, FactorResearchSnapshotManifest)
    assert isinstance(task, FactorResearchTask)
    assert isinstance(result, FactorResearchResultManifest)
    assert isinstance(receipt, FactorResearchWorkerReceipt)
    assert task.parameters.hypothesis_ids == HYPOTHESIS_IDS
    assert task.parameters.trial_ids == TRIAL_IDS


@pytest.mark.parametrize(
    ("payload", "field", "value", "adapter"),
    [
        (_task_payload(), "research_only", False, RESEARCH_TASK_ADAPTER),
        (_task_payload(), "live_order_effect", "orders", RESEARCH_TASK_ADAPTER),
        (_task_payload(), "automatic_promotion", True, RESEARCH_TASK_ADAPTER),
        (_task_payload(), "max_live_notional_usdt", 1, RESEARCH_TASK_ADAPTER),
        (_result_payload(), "anti_leakage_status", "WARN", RESEARCH_RESULT_ADAPTER),
        (_result_payload(), "anti_leakage_violation_count", 1, RESEARCH_RESULT_ADAPTER),
    ],
)
def test_factor_research_safety_literals_fail_closed(
    payload: dict[str, object],
    field: str,
    value: object,
    adapter: object,
) -> None:
    with pytest.raises(ValidationError):
        adapter.validate_python(payload | {field: value})  # type: ignore[attr-defined]


def test_factor_research_budget_and_identity_are_strict() -> None:
    duplicate_trials = _task_payload() | {
        "trial_ids": [TRIAL_IDS[0], TRIAL_IDS[0]],
        "test_count": 2,
    }
    too_many_hypotheses = _task_payload() | {
        "hypothesis_ids": [f"hyp.{index}" for index in range(7)]
    }
    mismatched_test_count = _task_payload() | {"test_count": 2}

    for payload in (duplicate_trials, too_many_hypotheses, mismatched_test_count):
        with pytest.raises(ValidationError):
            RESEARCH_TASK_ADAPTER.validate_python(payload)


def test_factor_research_rejects_cross_task_fields() -> None:
    with pytest.raises(ValidationError):
        RESEARCH_TASK_ADAPTER.validate_python(
            _task_payload() | {"alpha_factory_schema_version": "alpha_factory.v0.1"}
        )


def test_factor_research_window_is_bounded() -> None:
    payload = _task_payload() | {
        "start_date": date(2023, 7, 19).isoformat(),
        "end_date": date(2026, 7, 19).isoformat(),
        "max_history_days": 365,
    }
    with pytest.raises(ValidationError):
        RESEARCH_TASK_ADAPTER.validate_python(payload)
