from datetime import UTC, datetime

import polars as pl

from quant_lab.contracts.models import AlphaEvidence, GateStatus
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.publish import publish_gate_decisions_from_evidence


def test_publish_gate_decisions_from_evidence_writes_paper_ready_not_live_without_paper(tmp_path):
    lake = tmp_path / "lake"
    _write_evidence(lake, _evidence(paper_days=0, paper_slippage_coverage=0.0))

    result = publish_gate_decisions_from_evidence(lake, strategy="v5")
    gates = read_parquet_dataset(lake / "gold" / "gate_decision")

    assert result.status_counts == {GateStatus.PAPER_READY.value: 1}
    assert gates["status"][0] == GateStatus.PAPER_READY.value


def test_publish_gate_decisions_live_ready_requires_paper_evidence(tmp_path):
    lake = tmp_path / "lake"
    _write_evidence(lake, _evidence(paper_days=14, paper_slippage_coverage=0.8))

    result = publish_gate_decisions_from_evidence(lake, strategy="v5")

    assert result.status_counts == {GateStatus.LIVE_READY.value: 1}


def test_publish_gate_decisions_negative_ic_dead(tmp_path):
    lake = tmp_path / "lake"
    evidence = _evidence(paper_days=21, paper_slippage_coverage=0.9).model_copy(
        update={"ic_mean": -0.01}
    )
    _write_evidence(lake, evidence)

    result = publish_gate_decisions_from_evidence(lake, strategy="v5")

    assert result.status_counts == {GateStatus.DEAD.value: 1}


def _write_evidence(lake, evidence: AlphaEvidence) -> None:
    write_parquet_dataset(
        pl.DataFrame([{**evidence.model_dump(mode="json"), "source": "test"}]),
        lake / "gold" / "alpha_evidence",
    )


def _evidence(*, paper_days: int, paper_slippage_coverage: float) -> AlphaEvidence:
    return AlphaEvidence(
        alpha_id="v5.core.momentum",
        version="v0.1",
        data_version="market_bar:test",
        feature_version="core:v0.1:close_return_24",
        cost_model_version="costs-test",
        universe_id="okx-major-spot",
        start_ts=datetime(2026, 5, 10, tzinfo=UTC),
        end_ts=datetime(2026, 5, 11, tzinfo=UTC),
        coverage=0.99,
        ic_mean=0.05,
        ic_tstat=3.0,
        rank_ic_mean=0.05,
        rank_ic_tstat=3.0,
        edge_cost_ratio=2.0,
        oos_sharpe=1.0,
        oos_sortino=None,
        oos_cagr=None,
        oos_max_drawdown=0.05,
        profit_factor=1.5,
        turnover=0.1,
        cost_ratio=0.2,
        profitable_folds_ratio=0.8,
        train_oos_decay=0.2,
        pbo_score=None,
        paper_days=paper_days,
        paper_slippage_coverage=paper_slippage_coverage,
        created_at=datetime(2026, 5, 11, tzinfo=UTC),
    )
