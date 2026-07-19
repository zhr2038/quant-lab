from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import (
    count_parquet_rows,
    read_parquet_dataset,
    write_parquet_dataset,
    write_snapshot_meta,
)
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.research.factor_research.contracts import (
    FactorResearchDecision,
    HypothesisStatus,
)
from quant_lab.research.factor_research.outputs import FACTOR_RESEARCH_OUTPUT_SPECS
from quant_lab.research.factor_research.registry import (
    FACTOR_EXTERNAL_AUDIT_EVIDENCE_DATASET,
    FACTOR_EXTERNAL_AUDIT_EVIDENCE_SCHEMA,
    FACTOR_RETIREMENT_DATASET,
    FACTOR_RETIREMENT_SCHEMA,
    HYPOTHESIS_REGISTRY_SCHEMA,
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
    TRIAL_LEDGER_SCHEMA,
    factor_external_audit_evidence_frame,
    factor_retirement_registry_frame,
)
from quant_lab.research_plane.atomic_publish import (
    AtomicPublishItem,
    commit_atomic_research_generation,
    recover_atomic_research_generation,
)
from quant_lab.research_plane.result import ValidatedFactorResearchResult

FACTOR_RESEARCH_GENERATION_POINTER = Path("gold") / "factor_research_generation.json"
FACTOR_RESEARCH_GENERATION_SCHEMA = "factor_research_generation.v2"
FACTOR_RESEARCH_TRANSACTION_NAME = "factor_research"
FACTOR_RESEARCH_SOURCE = "factor_research.nas.v2"
FACTOR_RESEARCH_GENERATION_FRESH_DAYS = 7


def current_factor_research_generation_binding(
    lake_root: str | Path,
    *,
    alpha_as_of_date: date,
) -> dict[str, Any]:
    """Return one verified, immutable Factor Generation identity for Alpha research."""
    root = Path(lake_root)
    pointer_path = root / FACTOR_RESEARCH_GENERATION_POINTER
    if not pointer_path.is_file():
        raise RuntimeError("factor_research_generation_unavailable")
    try:
        pointer = json.loads(pointer_path.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("factor_research_generation_pointer_invalid") from exc
    required = (
        "generation_id",
        "factor_generation_digest",
        "hypothesis_registry_digest",
        "trial_ledger_digest",
        "as_of_date",
        "published_at",
        "hypothesis_ids",
    )
    missing = [field for field in required if not pointer.get(field)]
    if missing:
        raise RuntimeError(
            "factor_research_generation_binding_incomplete:" + ",".join(missing)
        )
    generation_id = str(pointer["generation_id"])
    for field in (
        "factor_generation_digest",
        "hypothesis_registry_digest",
        "trial_ledger_digest",
    ):
        value = str(pointer[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise RuntimeError(f"factor_research_generation_{field}_invalid")
    try:
        generation_as_of_date = date.fromisoformat(str(pointer["as_of_date"]))
        published_at = datetime.fromisoformat(str(pointer["published_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("factor_research_generation_timestamp_invalid") from exc
    if published_at.tzinfo is None or published_at.utcoffset() != UTC.utcoffset(published_at):
        raise RuntimeError("factor_research_generation_published_at_not_utc")
    if generation_as_of_date > alpha_as_of_date:
        raise RuntimeError("factor_research_generation_from_future")
    verify_factor_research_generation(root, generation_id)
    return {
        "factor_generation_id": generation_id,
        "factor_generation_digest": str(pointer["factor_generation_digest"]),
        "factor_generation_as_of_date": generation_as_of_date,
        "factor_generation_published_at": published_at,
        "hypothesis_registry_digest": str(pointer["hypothesis_registry_digest"]),
        "trial_ledger_digest": str(pointer["trial_ledger_digest"]),
        "factor_generation_fresh": (
            alpha_as_of_date - generation_as_of_date
        ).days
        <= FACTOR_RESEARCH_GENERATION_FRESH_DAYS,
        "factor_generation_hypothesis_ids": tuple(
            sorted(str(item) for item in pointer["hypothesis_ids"])
        ),
    }


def publish_factor_research_generation(
    lake_root: str | Path,
    validated: ValidatedFactorResearchResult,
) -> dict[str, int]:
    """Atomically publish cloud-owned Factor Research v2 state and compute outputs."""
    root = Path(lake_root)
    recover_factor_research_publication(root)
    manifest = validated.manifest
    evidence = pl.read_parquet(validated.output_paths["factor_evidence"])
    existing_registry = read_parquet_dataset(root / RESEARCH_HYPOTHESIS_REGISTRY_DATASET)
    existing_ledger = read_parquet_dataset(root / RESEARCH_TRIAL_LEDGER_DATASET)
    updated_registry = _updated_hypothesis_registry(
        existing_registry,
        evidence=evidence,
        hypothesis_ids=set(manifest.hypothesis_ids),
        completed_at=manifest.completed_at,
    )
    updated_ledger = _updated_trial_ledger(
        existing_ledger,
        evidence=evidence,
        trial_ids=set(manifest.trial_ids),
        started_at=manifest.generated_at,
        completed_at=manifest.completed_at,
    )
    retirement = read_parquet_dataset(root / FACTOR_RETIREMENT_DATASET)
    if retirement.is_empty():
        retirement = factor_retirement_registry_frame(recorded_at=manifest.completed_at)
    external_audit = read_parquet_dataset(root / FACTOR_EXTERNAL_AUDIT_EVIDENCE_DATASET)
    if external_audit.is_empty():
        external_audit = factor_external_audit_evidence_frame(imported_at=manifest.completed_at)

    generation_digest = _generation_digest(validated)
    generation_payload = {
        "schema_version": FACTOR_RESEARCH_GENERATION_SCHEMA,
        "generation_id": manifest.generation_id,
        "factor_generation_digest": generation_digest,
        "task_id": manifest.task_id,
        "snapshot_id": manifest.snapshot_id,
        "hypothesis_ids": list(manifest.hypothesis_ids),
        "trial_ids": list(manifest.trial_ids),
        "quant_lab_commit": manifest.quant_lab_commit,
        "worker_commit": manifest.worker_commit,
        "data_snapshot_digest": manifest.source_input_digest,
        "hypothesis_registry_digest": manifest.hypothesis_registry_digest,
        "trial_ledger_digest": manifest.trial_ledger_digest,
        "test_count": manifest.test_count,
        "multiple_testing_family": manifest.multiple_testing_family,
        "as_of_date": manifest.as_of_date.isoformat(),
        "published_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
    }
    transaction_id = uuid.uuid4().hex
    staging_root = root / "gold" / f".__factor_research_stage_{transaction_id[:8]}"
    staging_root.mkdir(parents=True, exist_ok=False)
    items: list[AtomicPublishItem] = []
    row_counts: dict[str, int] = {}
    try:
        control_frames = (
            (
                "research_hypothesis_registry",
                RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
                updated_registry,
                HYPOTHESIS_REGISTRY_SCHEMA,
            ),
            (
                "research_trial_ledger",
                RESEARCH_TRIAL_LEDGER_DATASET,
                updated_ledger,
                TRIAL_LEDGER_SCHEMA,
            ),
            (
                "factor_retirement",
                FACTOR_RETIREMENT_DATASET,
                retirement,
                FACTOR_RETIREMENT_SCHEMA,
            ),
            (
                "factor_external_audit_evidence",
                FACTOR_EXTERNAL_AUDIT_EVIDENCE_DATASET,
                external_audit,
                FACTOR_EXTERNAL_AUDIT_EVIDENCE_SCHEMA,
            ),
        )
        for index, (name, target, frame, schema) in enumerate(control_frames):
            normalized = _normalize_frame(frame, schema)
            staged = staging_root / f"control-{index:02d}"
            _stage_dataset(
                staged,
                dataset_name=name,
                frame=normalized,
                generated_at=manifest.generated_at,
                generation_payload=generation_payload,
            )
            row_counts[name] = normalized.height
            items.append(AtomicPublishItem(target=target, staged=staged.relative_to(root)))

        for index, spec in enumerate(FACTOR_RESEARCH_OUTPUT_SPECS):
            incoming = pl.read_parquet(validated.output_paths[spec.dataset_name])
            existing = read_parquet_dataset(root / spec.relative_path)
            merged = _merge_managed_factor_rows(
                existing,
                incoming,
                schema=spec.schema,
                hypothesis_ids=set(manifest.hypothesis_ids),
                as_of_date=manifest.as_of_date.isoformat(),
            )
            staged = staging_root / f"dataset-{index:02d}"
            _stage_dataset(
                staged,
                dataset_name=spec.dataset_name,
                frame=merged,
                generated_at=manifest.generated_at,
                generation_payload=generation_payload,
            )
            row_counts[spec.dataset_name] = merged.height
            items.append(
                AtomicPublishItem(target=spec.relative_path, staged=staged.relative_to(root))
            )

        reports_staged = staging_root / "reports"
        reports_staged.mkdir(parents=True, exist_ok=False)
        for name, payload in sorted(validated.reports.items()):
            if Path(name).name != name:
                raise ValueError("unsafe_factor_research_report_name")
            staged = reports_staged / name
            staged.write_bytes(payload)
            items.append(
                AtomicPublishItem(
                    target=Path("reports") / name,
                    staged=staged.relative_to(root),
                )
            )

        pointer = generation_payload | {
            "datasets": list(row_counts),
            "row_counts": row_counts,
        }
        commit_atomic_research_generation(
            root,
            transaction_name=FACTOR_RESEARCH_TRANSACTION_NAME,
            generation_payload=pointer,
            pointer_path=FACTOR_RESEARCH_GENERATION_POINTER,
            items=items,
            post_commit_validate=lambda: verify_factor_research_generation(
                root, manifest.generation_id, row_counts
            ),
        )
        return row_counts
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def recover_factor_research_publication(lake_root: str | Path) -> bool:
    return recover_atomic_research_generation(
        lake_root,
        transaction_name=FACTOR_RESEARCH_TRANSACTION_NAME,
        pointer_path=FACTOR_RESEARCH_GENERATION_POINTER,
    )


def verify_factor_research_generation(
    lake_root: str | Path,
    generation_id: str,
    expected_rows: dict[str, int] | None = None,
) -> dict[str, int]:
    root = Path(lake_root)
    pointer = json.loads((root / FACTOR_RESEARCH_GENERATION_POINTER).read_text("utf-8"))
    if pointer.get("generation_id") != generation_id:
        raise RuntimeError("factor_research_generation_pointer_mismatch")
    if pointer.get("schema_version") != FACTOR_RESEARCH_GENERATION_SCHEMA:
        raise RuntimeError("factor_research_generation_schema_mismatch")
    for field in (
        "factor_generation_digest",
        "hypothesis_registry_digest",
        "trial_ledger_digest",
    ):
        value = str(pointer.get(field) or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise RuntimeError(f"factor_research_generation_{field}_invalid")
    if (
        pointer.get("research_only") is not True
        or pointer.get("live_order_effect") != "none"
        or pointer.get("automatic_promotion") is not False
        or int(pointer.get("max_live_notional_usdt") or 0) != 0
    ):
        raise RuntimeError("factor_research_generation_safety_mismatch")
    rows = {str(key): int(value) for key, value in dict(pointer.get("row_counts") or {}).items()}
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError("factor_research_generation_row_count_mismatch")
    targets = {
        "research_hypothesis_registry": RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
        "research_trial_ledger": RESEARCH_TRIAL_LEDGER_DATASET,
        "factor_retirement": FACTOR_RETIREMENT_DATASET,
        "factor_external_audit_evidence": FACTOR_EXTERNAL_AUDIT_EVIDENCE_DATASET,
        **{spec.dataset_name: spec.relative_path for spec in FACTOR_RESEARCH_OUTPUT_SPECS},
    }
    if set(rows) != set(targets):
        raise RuntimeError("factor_research_generation_dataset_set_mismatch")
    for dataset_name, target in targets.items():
        metadata = json.loads(
            (root / target / "_research_generation.json").read_text("utf-8")
        )
        if metadata.get("generation_id") != generation_id:
            raise RuntimeError(f"factor_research_dataset_generation_mismatch:{target}")
        if (
            metadata.get("factor_generation_digest")
            != pointer.get("factor_generation_digest")
        ):
            raise RuntimeError(f"factor_research_dataset_digest_mismatch:{target}")
        if count_parquet_rows(root / target) != rows[dataset_name]:
            raise RuntimeError(f"factor_research_dataset_row_count_mismatch:{target}")
    return rows


def _updated_trial_ledger(
    existing: pl.DataFrame,
    *,
    evidence: pl.DataFrame,
    trial_ids: set[str],
    started_at: datetime,
    completed_at: datetime,
) -> pl.DataFrame:
    if existing.is_empty():
        raise ValueError("factor_research_trial_ledger_missing_on_publish")
    decisions = evidence.select("trial_id", "decision")
    if set(decisions.get_column("trial_id").to_list()) != trial_ids:
        raise ValueError("factor_research_trial_decisions_incomplete")
    updated = existing.join(
        decisions.rename({"decision": "_result_decision"}), on="trial_id", how="left"
    ).with_columns(
        pl.when(pl.col("trial_id").is_in(sorted(trial_ids)))
        .then(pl.lit("COMPLETED"))
        .otherwise(pl.col("status"))
        .alias("status"),
        pl.when(pl.col("trial_id").is_in(sorted(trial_ids)))
        .then(pl.col("_result_decision"))
        .otherwise(pl.col("decision"))
        .alias("decision"),
        pl.when(pl.col("trial_id").is_in(sorted(trial_ids)))
        .then(pl.coalesce(pl.col("started_at"), pl.lit(started_at)))
        .otherwise(pl.col("started_at"))
        .alias("started_at"),
        pl.when(pl.col("trial_id").is_in(sorted(trial_ids)))
        .then(pl.lit(completed_at))
        .otherwise(pl.col("finished_at"))
        .alias("finished_at"),
    ).drop("_result_decision")
    return _normalize_frame(updated, TRIAL_LEDGER_SCHEMA).sort("trial_id")


def _updated_hypothesis_registry(
    existing: pl.DataFrame,
    *,
    evidence: pl.DataFrame,
    hypothesis_ids: set[str],
    completed_at: datetime,
) -> pl.DataFrame:
    if existing.is_empty():
        raise ValueError("factor_research_hypothesis_registry_missing_on_publish")
    statuses: dict[str, str] = {}
    for hypothesis_id in hypothesis_ids:
        rows = evidence.filter(pl.col("hypothesis_id") == hypothesis_id)
        statuses[hypothesis_id] = _hypothesis_status(rows)
    result = existing.with_columns(
        pl.when(pl.col("hypothesis_id").is_in(sorted(hypothesis_ids)))
        .then(
            pl.col("hypothesis_id").replace_strict(
                statuses, default=pl.col("status"), return_dtype=pl.Utf8
            )
        )
        .otherwise(pl.col("status"))
        .alias("status"),
        pl.when(pl.col("hypothesis_id").is_in(sorted(hypothesis_ids)))
        .then(pl.lit(completed_at))
        .otherwise(pl.col("updated_at"))
        .alias("updated_at"),
    )
    return _normalize_frame(result, HYPOTHESIS_REGISTRY_SCHEMA).sort(
        ["hypothesis_id", "hypothesis_version"]
    )


def _hypothesis_status(evidence: pl.DataFrame) -> str:
    decisions = set(evidence.get_column("decision").drop_nulls().to_list())
    signal_validities = set(
        evidence.get_column("signal_validity").drop_nulls().to_list()
    )
    if FactorResearchDecision.PAPER_CANDIDATE.value in decisions:
        return HypothesisStatus.PAPER_CANDIDATE.value
    if "PASS" in signal_validities:
        if FactorResearchDecision.PORTFOLIO_FAIL.value in decisions:
            return HypothesisStatus.PORTFOLIO_FAIL.value
        return HypothesisStatus.SIGNAL_VALID.value
    if FactorResearchDecision.SIGNAL_CANDIDATE.value in decisions:
        return HypothesisStatus.RUNNING.value
    if decisions and decisions.issubset(
        {
            FactorResearchDecision.INCONCLUSIVE.value,
            FactorResearchDecision.INCONCLUSIVE_OVERFIT_DIAGNOSTICS.value,
        }
    ):
        return HypothesisStatus.INCONCLUSIVE.value
    return HypothesisStatus.REJECTED.value


def _merge_managed_factor_rows(
    existing: pl.DataFrame,
    incoming: pl.DataFrame,
    *,
    schema: dict[str, pl.DataType],
    hypothesis_ids: set[str],
    as_of_date: str,
) -> pl.DataFrame:
    current = _normalize_frame(existing, schema)
    replacement = _normalize_frame(incoming, schema)
    if not current.is_empty() and "hypothesis_id" in current.columns:
        managed = pl.col("hypothesis_id").is_in(sorted(hypothesis_ids))
        if "as_of_date" in current.columns:
            managed &= pl.col("as_of_date") == as_of_date
        current = current.filter(~managed.fill_null(False))
    merged = pl.concat([current, replacement], how="vertical_relaxed")
    return merged.select(list(schema)).cast(schema, strict=True)


def _normalize_frame(
    frame: pl.DataFrame,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    normalized = frame
    for column, dtype in schema.items():
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return normalized.select(list(schema)).cast(schema, strict=True)


def _stage_dataset(
    staged: Path,
    *,
    dataset_name: str,
    frame: pl.DataFrame,
    generated_at: datetime,
    generation_payload: dict[str, Any],
) -> None:
    write_parquet_dataset(frame, staged)
    write_snapshot_meta(
        staged,
        dataset_name=dataset_name,
        frame=frame,
        schema_version="factor_research.v2",
        generated_at=generated_at,
    )
    atomic_write_json(staged / "_research_generation.json", generation_payload)


def _generation_digest(validated: ValidatedFactorResearchResult) -> str:
    manifest = validated.manifest
    payload = {
        "task_id": manifest.task_id,
        "snapshot_id": manifest.snapshot_id,
        "hypothesis_ids": list(manifest.hypothesis_ids),
        "trial_ids": list(manifest.trial_ids),
        "quant_lab_commit": manifest.quant_lab_commit,
        "worker_commit": manifest.worker_commit,
        "source_input_digest": manifest.source_input_digest,
        "hypothesis_registry_digest": manifest.hypothesis_registry_digest,
        "trial_ledger_digest": manifest.trial_ledger_digest,
        "outputs": [
            {"dataset_name": item.dataset_name, "sha256": item.sha256}
            for item in sorted(manifest.outputs, key=lambda item: item.dataset_name)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
