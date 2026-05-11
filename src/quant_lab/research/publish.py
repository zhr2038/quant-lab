import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import AlphaEvidence, AlphaResearchSpec, GateDecision
from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset
from quant_lab.gates.defaults import evaluate_alpha_gate
from quant_lab.research.evidence import build_alpha_evidence

ALPHA_EVIDENCE_DATASET = Path("gold") / "alpha_evidence"
GATE_DECISION_DATASET = Path("gold") / "gate_decision"
RISK_PERMISSION_DATASET = Path("gold") / "risk_permission"

ALPHA_EVIDENCE_SCHEMA = {
    "alpha_id": pl.Utf8,
    "version": pl.Utf8,
    "data_version": pl.Utf8,
    "feature_version": pl.Utf8,
    "cost_model_version": pl.Utf8,
    "universe_id": pl.Utf8,
    "start_ts": pl.Utf8,
    "end_ts": pl.Utf8,
    "coverage": pl.Float64,
    "ic_mean": pl.Float64,
    "ic_tstat": pl.Float64,
    "rank_ic_mean": pl.Float64,
    "rank_ic_tstat": pl.Float64,
    "oos_sharpe": pl.Float64,
    "oos_sortino": pl.Float64,
    "oos_cagr": pl.Float64,
    "oos_max_drawdown": pl.Float64,
    "profit_factor": pl.Float64,
    "turnover": pl.Float64,
    "cost_ratio": pl.Float64,
    "edge_cost_ratio": pl.Float64,
    "profitable_folds_ratio": pl.Float64,
    "train_oos_decay": pl.Float64,
    "pbo_score": pl.Float64,
    "paper_days": pl.Int64,
    "paper_slippage_coverage": pl.Float64,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
}

GATE_DECISION_SCHEMA = {
    "strategy": pl.Utf8,
    "alpha_id": pl.Utf8,
    "version": pl.Utf8,
    "gate_version": pl.Utf8,
    "status": pl.Utf8,
    "passed": pl.Boolean,
    "reasons": pl.Utf8,
    "metrics": pl.Utf8,
    "next_action": pl.Utf8,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "fallback_level": pl.Utf8,
}


class AlphaEvidencePublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    alpha_id: str
    version: str
    strategy: str
    status: str
    dataset_rows: int = Field(ge=0)
    alpha_evidence_rows: int = Field(ge=0)
    gate_decision_rows: int = Field(ge=0)
    gate_status: str | None = None
    warnings: list[str] = Field(default_factory=list)


class GatePublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    strategy: str
    alpha_evidence_rows: int = Field(ge=0)
    gate_decision_rows: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ResearchHealthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    date: str | None = None
    alpha_evidence_rows: int = Field(ge=0)
    gate_decision_rows: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    latest_risk_permission: dict[str, Any] | None = None
    missing_prerequisites: list[str] = Field(default_factory=list)


def build_and_publish_alpha_evidence(
    lake_root: str | Path,
    spec: AlphaResearchSpec,
) -> AlphaEvidencePublishResult:
    root = Path(lake_root)
    result = build_alpha_evidence(root, spec)
    warnings = list(result.warnings)
    if result.evidence is None:
        return AlphaEvidencePublishResult(
            lake_root=str(root),
            alpha_id=spec.alpha_id,
            version=spec.version,
            strategy=spec.strategy,
            status=result.status,
            dataset_rows=result.dataset.height,
            alpha_evidence_rows=read_parquet_dataset(root / ALPHA_EVIDENCE_DATASET).height,
            gate_decision_rows=read_parquet_dataset(root / GATE_DECISION_DATASET).height,
            gate_status=None,
            warnings=warnings,
        )

    alpha_rows = publish_alpha_evidence(root, [result.evidence])
    decision = evaluate_alpha_gate(result.evidence)
    gate_rows = publish_gate_decision(root, spec.strategy, decision)
    return AlphaEvidencePublishResult(
        lake_root=str(root),
        alpha_id=spec.alpha_id,
        version=spec.version,
        strategy=spec.strategy,
        status=result.status,
        dataset_rows=result.dataset.height,
        alpha_evidence_rows=alpha_rows,
        gate_decision_rows=gate_rows,
        gate_status=decision.status.value,
        warnings=warnings,
    )


def publish_alpha_evidence(lake_root: str | Path, rows: list[AlphaEvidence]) -> int:
    if not rows:
        return read_parquet_dataset(Path(lake_root) / ALPHA_EVIDENCE_DATASET).height
    frame = pl.DataFrame(
        [_alpha_evidence_row(row) for row in rows],
        schema=ALPHA_EVIDENCE_SCHEMA,
        orient="row",
    )
    return upsert_parquet_dataset(
        frame,
        Path(lake_root) / ALPHA_EVIDENCE_DATASET,
        key_columns=[
            "alpha_id",
            "version",
            "data_version",
            "feature_version",
            "cost_model_version",
            "universe_id",
            "start_ts",
            "end_ts",
        ],
    )


def publish_gate_decision(
    lake_root: str | Path,
    strategy: str,
    decision: GateDecision,
) -> int:
    frame = pl.DataFrame(
        [_gate_decision_row(strategy, decision)],
        schema=GATE_DECISION_SCHEMA,
        orient="row",
    )
    return upsert_parquet_dataset(
        frame,
        Path(lake_root) / GATE_DECISION_DATASET,
        key_columns=["strategy", "alpha_id", "version", "gate_version"],
    )


def publish_gate_decisions_from_evidence(
    lake_root: str | Path,
    strategy: str = "v5",
) -> GatePublishResult:
    root = Path(lake_root)
    evidence_rows = _load_alpha_evidence(root)
    warnings: list[str] = []
    if not evidence_rows:
        warnings.append("alpha_evidence missing or empty")
        return GatePublishResult(
            lake_root=str(root),
            strategy=strategy,
            alpha_evidence_rows=0,
            gate_decision_rows=read_parquet_dataset(root / GATE_DECISION_DATASET).height,
            status_counts={},
            warnings=warnings,
        )

    decisions = [evaluate_alpha_gate(evidence) for evidence in evidence_rows]
    frame = pl.DataFrame(
        [_gate_decision_row(strategy, decision) for decision in decisions],
        schema=GATE_DECISION_SCHEMA,
        orient="row",
    )
    gate_rows = upsert_parquet_dataset(
        frame,
        root / GATE_DECISION_DATASET,
        key_columns=["strategy", "alpha_id", "version", "gate_version"],
    )
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.status.value] = counts.get(decision.status.value, 0) + 1
    return GatePublishResult(
        lake_root=str(root),
        strategy=strategy,
        alpha_evidence_rows=len(evidence_rows),
        gate_decision_rows=gate_rows,
        status_counts=counts,
        warnings=[],
    )


def research_health(lake_root: str | Path, date: str | None = None) -> ResearchHealthResult:
    root = Path(lake_root)
    evidence = read_parquet_dataset(root / ALPHA_EVIDENCE_DATASET)
    gates = read_parquet_dataset(root / GATE_DECISION_DATASET)
    risk = read_parquet_dataset(root / RISK_PERMISSION_DATASET)
    missing = []
    if evidence.is_empty():
        missing.append("alpha_evidence")
    if gates.is_empty():
        missing.append("gate_decision")
    if risk.is_empty():
        missing.append("risk_permission")
    if date and "created_at" in gates.columns:
        gates = gates.filter(pl.col("created_at").cast(pl.Utf8).str.starts_with(date))
    status_counts = {}
    if not gates.is_empty() and "status" in gates.columns:
        status_counts = {
            str(row["status"]): int(row["count"])
            for row in gates.group_by("status").len(name="count").to_dicts()
        }
    latest_risk = None if risk.is_empty() else _latest_row(risk, "created_at")
    return ResearchHealthResult(
        lake_root=str(root),
        date=date,
        alpha_evidence_rows=evidence.height,
        gate_decision_rows=gates.height,
        status_counts=status_counts,
        latest_risk_permission=latest_risk,
        missing_prerequisites=missing,
    )


def _load_alpha_evidence(root: Path) -> list[AlphaEvidence]:
    df = read_parquet_dataset(root / ALPHA_EVIDENCE_DATASET)
    if df.is_empty():
        return []
    evidence: list[AlphaEvidence] = []
    for row in df.to_dicts():
        cleaned = dict(row)
        cleaned.pop("source", None)
        try:
            evidence.append(AlphaEvidence.model_validate(cleaned))
        except Exception:
            continue
    return evidence


def _alpha_evidence_row(evidence: AlphaEvidence) -> dict[str, Any]:
    return {**evidence.model_dump(mode="json"), "source": "research.alpha_evidence.v0.1"}


def _gate_decision_row(strategy: str, decision: GateDecision) -> dict[str, Any]:
    return {
        **decision.model_dump(mode="json"),
        "strategy": strategy,
        "reasons": _json(decision.reasons),
        "metrics": _json(decision.metrics),
        "source": "research.alpha_evidence.v0.1",
        "fallback_level": "NONE",
    }


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _latest_row(df: pl.DataFrame, column: str) -> dict[str, Any]:
    if column in df.columns:
        return df.sort(column).tail(1).to_dicts()[0]
    return df.tail(1).to_dicts()[0]
