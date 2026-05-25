from datetime import UTC, datetime

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import app
from quant_lab.contracts.models import AlphaEvidence, GateDecision, GateStatus
from quant_lab.data.lake import write_parquet_dataset


def test_gate_example_returns_conservative_decision():
    response = TestClient(app).get("/v1/gates/example")

    assert response.status_code == 200
    payload = response.json()
    assert payload["alpha_id"] == "example-alpha"
    assert payload["version"] == "example"
    assert payload["gate_version"] == "example-conservative-v0.1"
    assert payload["status"] == "QUARANTINE"
    assert payload["passed"] is False
    assert payload["reasons"] == ["example_not_live_ready_evidence"]
    assert payload["metrics"] == {}


def test_gate_decision_route_returns_conservative_missing_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))

    response = TestClient(app).get("/v1/gates/decision/demo-alpha")

    assert response.status_code == 200
    payload = response.json()
    assert payload["alpha_id"] == "demo-alpha"
    assert payload["status"] == "QUARANTINE"
    assert payload["reasons"] == ["missing_gate_decision"]
    assert payload["next_action"] == "build_alpha_evidence_before_gate"


def test_gate_decision_route_reads_lake(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    decision = GateDecision(
        alpha_id="v5.core.momentum",
        version="v0.1",
        gate_version="default-v0.1",
        status=GateStatus.PAPER_READY,
        passed=False,
        reasons=["needs_paper_observation"],
        metrics={"paper_days": 0},
        next_action="continue_paper_observation",
        created_at=datetime(2026, 5, 11, tzinfo=UTC),
    ).model_dump(mode="json")
    write_parquet_dataset(
        pl.DataFrame([decision | {"strategy": "v5"}]),
        lake / "gold" / "gate_decision",
    )

    response = TestClient(app).get("/v1/gates/decision/v5.core.momentum")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "PAPER_READY"
    assert payload["reasons"] == ["needs_paper_observation"]


def test_research_alpha_route_returns_evidence_and_gate(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    evidence = AlphaEvidence.example_live_ready().model_copy(
        update={"alpha_id": "v5.core.momentum", "version": "v0.1"}
    )
    decision = GateDecision(
        alpha_id="v5.core.momentum",
        version="v0.1",
        gate_version="default-v0.1",
        status=GateStatus.LIVE_READY,
        passed=True,
        reasons=["all_default_gates_passed"],
        metrics={"ic_tstat": 3.1},
        next_action="eligible_for_strategy_consumer_review",
        created_at=datetime(2026, 5, 11, tzinfo=UTC),
    ).model_dump(mode="json")
    write_parquet_dataset(
        pl.DataFrame([{**evidence.model_dump(mode="json"), "source": "test"}]),
        lake / "gold" / "alpha_evidence",
    )
    write_parquet_dataset(
        pl.DataFrame([decision | {"strategy": "v5"}]),
        lake / "gold" / "gate_decision",
    )

    response = TestClient(app).get("/v1/research/alpha/v5.core.momentum")

    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence"]["alpha_id"] == "v5.core.momentum"
    assert payload["gate_decision"]["status"] == "LIVE_READY"
    assert payload["warnings"] == []


def test_research_alpha_route_uses_lazy_alpha_evidence_lookup(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    older = AlphaEvidence.example_live_ready().model_copy(
        update={
            "alpha_id": "v5.core.momentum",
            "version": "old",
            "created_at": datetime(2026, 5, 10, tzinfo=UTC),
        }
    )
    newer = AlphaEvidence.example_live_ready().model_copy(
        update={
            "alpha_id": "v5.core.momentum",
            "version": "new",
            "created_at": datetime(2026, 5, 11, tzinfo=UTC),
        }
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {**older.model_dump(mode="json"), "source": "test"},
                {**newer.model_dump(mode="json"), "source": "test"},
            ]
        ),
        lake / "gold" / "alpha_evidence",
    )

    def fail_full_read(*args, **kwargs):
        raise AssertionError("research alpha endpoint should lazy-filter alpha_evidence")

    monkeypatch.setattr("quant_lab.api.main.read_parquet_dataset", fail_full_read)

    response = TestClient(app).get("/v1/research/alpha/v5.core.momentum")

    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence"]["alpha_id"] == "v5.core.momentum"
    assert payload["evidence"]["version"] == "new"
    assert payload["warnings"] == ["gate_decision missing for alpha_id"]
