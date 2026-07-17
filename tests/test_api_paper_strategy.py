from datetime import UTC, datetime

import polars as pl
import pytest
from fastapi.testclient import TestClient

from quant_lab.api.main import create_app
from quant_lab.data.lake import write_parquet_dataset
from quant_lab.paper.contracts import PaperStrategyAck, PaperStrategyProposal
from quant_lab.paper.service import (
    build_canonical_proposal_snapshot,
    publish_canonical_proposal_snapshot,
    publish_proposals,
    read_proposals,
)


def _proposal() -> PaperStrategyProposal:
    return PaperStrategyProposal(
        strategy_id="TEST_PAPER",
        strategy_version="1.0.0",
        strategy_family="test",
        symbol="TRX/USDT",
        timeframe="1h",
        entry_rule={"operator": "momentum_gt", "field": "momentum_24", "value": 0},
        exit_rule={"operator": "max_holding_bars", "value": 48},
        max_holding_bars=48,
        created_at=datetime(2026, 7, 10, tzinfo=UTC),
        expires_at=datetime(2026, 8, 10, tzinfo=UTC),
        required_market_fields=["bid", "ask", "mid", "momentum_24"],
        required_cost_trust_level="PAPER_ONLY",
    )


def test_paper_api_gets_and_ack_is_disabled_by_default(monkeypatch, tmp_path):
    lake = tmp_path / "lake"
    proposal = _proposal()
    published = publish_proposals(lake, [proposal])
    _snapshot_frame, snapshot = publish_canonical_proposal_snapshot(
        lake,
        published,
        generated_at=datetime(2026, 7, 10, 0, 30, tzinfo=UTC),
        source_quant_lab_commit="a" * 40,
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "cohort_id": "paper-cohort-api",
                    "cohort_version": 3,
                    "observation_start_at": "2026-07-10T01:00:00Z",
                    "status": "OBSERVING",
                    "proposal_content_snapshot_sha256": snapshot[
                        "proposal_content_snapshot_sha256"
                    ],
                    "last_evaluated_at": "2026-07-10T01:05:00Z",
                }
            ]
        ),
        lake / "gold/paper_cohort_manifest",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_content_snapshot_sha256": snapshot[
                        "proposal_content_snapshot_sha256"
                    ],
                    "proposal_snapshot_fetched_at": "2026-07-10T01:06:00Z",
                }
            ]
        ),
        lake / "silver/v5_quant_lab_contract_status",
    )
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_PAPER_ACK_WRITE_ENABLED", raising=False)
    client = TestClient(create_app())

    response = client.get("/v1/paper-strategy/proposals")
    assert response.status_code == 200
    payload = response.json()
    assert payload["proposal_snapshot_id"].startswith("proposal-snapshot:")
    assert len(payload["proposal_snapshot_sha256"]) == 64
    assert payload["proposal_content_snapshot_id"].startswith(
        "proposal-content-snapshot:"
    )
    assert len(payload["proposal_content_snapshot_sha256"]) == 64
    assert payload["snapshot_generated_at"] == "2026-07-10T00:30:00+00:00"
    assert payload["proposal_count"] == 1
    assert payload["proposal_ids"] == [proposal.proposal_id]
    assert payload["proposal_hashes"] == [proposal.proposal_hash]
    assert payload["source_quant_lab_commit"] == "a" * 40
    assert payload["proposal_contract_version"] == "quant_lab.paper_strategy.v1"
    assert payload["proposal_compiler_version"]
    assert payload["cohort_id"] == "paper-cohort-api"
    assert payload["cohort_version"] == 3
    assert payload["cohort_observation_start_at"] == "2026-07-10T01:00:00Z"
    assert payload["cohort_status"] == "OBSERVING"
    assert payload["cohort_proposal_content_snapshot_sha256"] == payload[
        "proposal_content_snapshot_sha256"
    ]
    assert payload["cohort_last_evaluated_at"] == "2026-07-10T01:05:00Z"
    assert payload["last_evaluated_at"]
    assert payload["last_consumed_by_v5_at"] == "2026-07-10T01:06:00Z"
    assert payload["proposals"][0]["proposal_snapshot_id"] == payload[
        "proposal_snapshot_id"
    ]
    detail = client.get(f"/v1/paper-strategy/proposals/{proposal.proposal_id}")
    assert detail.status_code == 200
    ack = PaperStrategyAck(
        proposal_id=proposal.proposal_id,
        proposal_hash=proposal.proposal_hash,
        accepted=True,
        tracker_id=f"paper:{proposal.proposal_id}",
        strategy_version=proposal.strategy_version,
        rules_locked=True,
        accepted_at=datetime(2026, 7, 10, 1, tzinfo=UTC),
        expires_at=proposal.expires_at,
    )
    response = client.post("/v1/paper-strategy/ack", json=ack.model_dump(mode="json"))
    assert response.status_code == 503


def test_paper_ack_write_is_authenticated_and_idempotent(monkeypatch, tmp_path):
    lake = tmp_path / "lake"
    proposal = _proposal()
    publish_proposals(lake, [proposal])
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_PAPER_ACK_WRITE_ENABLED", "true")
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "paper-token")
    client = TestClient(create_app())
    payload = PaperStrategyAck(
        proposal_id=proposal.proposal_id,
        proposal_hash=proposal.proposal_hash,
        accepted=True,
        tracker_id=f"paper:{proposal.proposal_id}",
        strategy_version=proposal.strategy_version,
        rules_locked=True,
        accepted_at=datetime(2026, 7, 10, 1, tzinfo=UTC),
        expires_at=proposal.expires_at,
    ).model_dump(mode="json")

    assert client.post("/v1/paper-strategy/ack", json=payload).status_code == 401
    headers = {"Authorization": "Bearer paper-token"}
    first = client.post("/v1/paper-strategy/ack", json=payload, headers=headers)
    second = client.post("/v1/paper-strategy/ack", json=payload, headers=headers)

    assert first.status_code == 200
    assert first.json()["exchange_state_mutated"] is False
    assert first.json()["idempotent"] is False
    assert second.json()["idempotent"] is True


def test_publish_reuses_canonical_id_when_only_provenance_changes(tmp_path):
    lake = tmp_path / "lake"
    original = _proposal()
    payload = original.model_dump(mode="json")
    payload.update(
        {
            "proposal_id": "",
            "proposal_hash": "",
            "created_at": datetime(2026, 7, 11, tzinfo=UTC),
            "expires_at": datetime(2026, 8, 11, tzinfo=UTC),
            "source_dataset_versions": {"alpha_discovery_board": "v2"},
        }
    )
    refreshed = PaperStrategyProposal.model_validate(payload)
    assert refreshed.proposal_hash != original.proposal_hash

    publish_proposals(lake, [original])
    publish_proposals(lake, [refreshed])

    rows = read_proposals(lake)
    assert len(rows) == 1
    assert rows[0]["proposal_id"] == original.proposal_id
    assert rows[0]["proposal_hash"] == original.proposal_hash
    assert rows[0]["expires_at"] == "2026-08-11T00:00:00Z"


def test_publish_rejects_rule_change_without_strategy_version_bump(tmp_path):
    lake = tmp_path / "lake"
    original = _proposal()
    payload = original.model_dump(mode="json")
    payload.update(
        {
            "proposal_id": "",
            "proposal_hash": "",
            "entry_rule": {
                "operator": "momentum_gt",
                "field": "momentum_24",
                "value": 1,
            },
        }
    )
    changed = PaperStrategyProposal.model_validate(payload)
    publish_proposals(lake, [original])

    with pytest.raises(ValueError, match="strategy_version_rule_conflict"):
        publish_proposals(lake, [changed])


def test_canonical_proposal_snapshot_is_stable_for_identical_members(tmp_path):
    lake = tmp_path / "lake"
    published = publish_proposals(lake, [_proposal()])
    first_frame, first = publish_canonical_proposal_snapshot(
        lake,
        published,
        generated_at=datetime(2026, 7, 10, tzinfo=UTC),
        source_quant_lab_commit="b" * 40,
    )
    second_frame, second = publish_canonical_proposal_snapshot(
        lake,
        published,
        generated_at=datetime(2026, 7, 11, tzinfo=UTC),
        source_quant_lab_commit="a" * 40,
    )

    assert second["proposal_snapshot_id"] != first["proposal_snapshot_id"]
    assert second["proposal_snapshot_sha256"] != first["proposal_snapshot_sha256"]
    assert second["proposal_content_snapshot_id"] == first[
        "proposal_content_snapshot_id"
    ]
    assert second["proposal_content_snapshot_sha256"] == first[
        "proposal_content_snapshot_sha256"
    ]
    assert second["snapshot_generated_at"] == first["snapshot_generated_at"]
    assert first["last_evaluated_at"] == "2026-07-10T00:00:00+00:00"
    assert second["last_evaluated_at"] == "2026-07-11T00:00:00+00:00"
    assert second["last_consumed_by_v5_at"] == ""
    assert first_frame["proposal_content_snapshot_id"].to_list() == second_frame[
        "proposal_content_snapshot_id"
    ].to_list()


def test_proposal_content_snapshot_changes_for_membership_hash_or_contract() -> None:
    proposal = _proposal().model_dump(mode="json")
    base = pl.DataFrame([proposal], infer_schema_length=None)
    _frame, initial = build_canonical_proposal_snapshot(
        base,
        generated_at=datetime(2026, 7, 10, tzinfo=UTC),
        source_quant_lab_commit="a" * 40,
    )
    changed_hash = base.with_columns(pl.lit("f" * 64).alias("proposal_hash"))
    _frame, hash_changed = build_canonical_proposal_snapshot(
        changed_hash,
        generated_at=datetime(2026, 7, 10, tzinfo=UTC),
        source_quant_lab_commit="a" * 40,
    )
    changed_contract = base.with_columns(
        pl.lit("quant_lab.paper_strategy.v2").alias("contract_version")
    )
    _frame, contract_changed = build_canonical_proposal_snapshot(
        changed_contract,
        generated_at=datetime(2026, 7, 10, tzinfo=UTC),
        source_quant_lab_commit="a" * 40,
    )

    assert hash_changed["proposal_content_snapshot_sha256"] != initial[
        "proposal_content_snapshot_sha256"
    ]
    assert contract_changed["proposal_content_snapshot_sha256"] != initial[
        "proposal_content_snapshot_sha256"
    ]


def test_paper_status_surfaces_effective_promotion_lifecycle(monkeypatch, tmp_path):
    lake = tmp_path / "lake"
    proposal = _proposal()
    publish_proposals(lake, [proposal])
    tracker_id = f"paper:{proposal.proposal_id}"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": proposal.proposal_id,
                    "proposal_hash": proposal.proposal_hash,
                    "paper_tracker_id": tracker_id,
                    "accepted": True,
                    "rules_locked": True,
                }
            ]
        ),
        lake / "silver/v5_paper_strategy_proposal_ack",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": proposal.proposal_id,
                    "proposal_hash": proposal.proposal_hash,
                    "paper_tracker_id": tracker_id,
                    "accepted": True,
                    "rules_locked": True,
                    "paper_ready": False,
                    "lifecycle_state": "PAPER_EVIDENCE_INSUFFICIENT",
                    "lifecycle_reason": "insufficient_paper_days",
                    "blocked_reasons": '["insufficient_paper_days"]',
                    "next_required_actions": '["continue_paper_tracking"]',
                }
            ]
        ),
        lake / "gold/paper_strategy_promotion_gate",
    )
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)

    response = TestClient(create_app()).get("/v1/paper-strategy/status")

    assert response.status_code == 200
    row = response.json()[0]
    assert row["proposal_lifecycle_state"] == "PAPER_PROPOSAL_READY"
    assert row["lifecycle_state"] == "PAPER_EVIDENCE_INSUFFICIENT"
    assert row["lifecycle_reason"] == "insufficient_paper_days"
    assert row["blocked_reasons"] == ["insufficient_paper_days"]
    assert row["next_required_actions"] == ["continue_paper_tracking"]
    assert row["accepted"] is True
    assert row["rules_locked"] is True
    assert row["paper_tracker_id"] == tracker_id
    assert row["paper_ready"] is False
    assert row["status_source"] == "paper_strategy_promotion_gate"
