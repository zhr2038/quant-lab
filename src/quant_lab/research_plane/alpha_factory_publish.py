from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_lab.data.lake import (
    count_parquet_rows,
    read_parquet_dataset,
    write_parquet_dataset,
    write_snapshot_meta,
)
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.research.alpha_factory.factory import (
    ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS,
    ALPHA_FACTORY_PROMOTION_QUEUE_DATASET,
    PROMOTION_SCHEMA,
    SCHEMA_VERSION,
    STRATEGY_EVIDENCE_DATASET,
    derive_alpha_factory_cloud_outputs,
    merge_alpha_factory_managed_evidence,
)
from quant_lab.research.strategy_evidence import STRATEGY_EVIDENCE_SAMPLE_DATASET
from quant_lab.research_plane.atomic_publish import (
    AtomicPublishItem,
    commit_atomic_research_generation,
    recover_atomic_research_generation,
)
from quant_lab.research_plane.result import ValidatedAlphaFactoryResult

ALPHA_FACTORY_GENERATION_POINTER = Path("gold") / "alpha_factory_generation.json"
ALPHA_FACTORY_TRANSACTION_NAME = "alpha_factory"


def publish_alpha_factory_generation(
    lake_root: str | Path,
    validated: ValidatedAlphaFactoryResult,
) -> dict[str, int]:
    """Derive cloud-owned outputs and atomically publish one Alpha generation."""
    root = Path(lake_root)
    recover_alpha_factory_publication(root)
    manifest = validated.manifest
    result = pl.read_parquet(validated.output_paths["alpha_factory_result"])
    second_stage_samples = pl.read_parquet(
        validated.output_paths["second_stage_alpha_factory_sample"]
    )
    derivations = derive_alpha_factory_cloud_outputs(
        second_stage_samples=second_stage_samples,
        alpha_results=result,
        generated_at=manifest.generated_at,
    )
    del result
    del second_stage_samples

    existing_sample = read_parquet_dataset(root / STRATEGY_EVIDENCE_SAMPLE_DATASET)
    merged_sample = merge_alpha_factory_managed_evidence(
        existing_sample,
        derivations.strategy_evidence_sample,
        as_of_date=manifest.as_of_date,
        sample=True,
    )
    del existing_sample
    existing_summary = read_parquet_dataset(root / STRATEGY_EVIDENCE_DATASET)
    merged_summary = merge_alpha_factory_managed_evidence(
        existing_summary,
        derivations.strategy_evidence,
        as_of_date=manifest.as_of_date,
        sample=False,
    )
    del existing_summary

    transaction_id = uuid.uuid4().hex
    staging_root = root / "gold" / f".__alpha_factory_stage_{transaction_id[:8]}"
    staging_root.mkdir(parents=True, exist_ok=False)
    generation_payload = {
        "schema_version": "alpha_factory_generation.v1",
        "generation_id": manifest.generation_id,
        "task_id": manifest.task_id,
        "snapshot_id": manifest.snapshot_id,
        "commit": manifest.quant_lab_commit,
        "registry_digest": manifest.template_registry_digest,
        "factor_generation_id": manifest.factor_generation_id,
        "factor_generation_digest": manifest.factor_generation_digest,
        "factor_generation_as_of_date": (
            manifest.factor_generation_as_of_date.isoformat()
            if manifest.factor_generation_as_of_date is not None
            else None
        ),
        "factor_generation_published_at": (
            manifest.factor_generation_published_at.isoformat()
            if manifest.factor_generation_published_at is not None
            else None
        ),
        "hypothesis_registry_digest": manifest.hypothesis_registry_digest,
        "trial_ledger_digest": manifest.trial_ledger_digest,
        "factor_generation_fresh": manifest.factor_generation_fresh,
        "factor_generation_hypothesis_ids": list(
            manifest.factor_generation_hypothesis_ids or ()
        ),
        "as_of_date": manifest.as_of_date.isoformat(),
        "published_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
    }
    items: list[AtomicPublishItem] = []
    row_counts: dict[str, int] = {}
    try:
        for index, spec in enumerate(ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS):
            frame = pl.read_parquet(validated.output_paths[spec.dataset_name])
            staged = staging_root / f"dataset-{index:02d}"
            write_parquet_dataset(frame, staged)
            write_snapshot_meta(
                staged,
                dataset_name=spec.dataset_name,
                frame=frame,
                schema_version=SCHEMA_VERSION,
                generated_at=manifest.generated_at,
            )
            atomic_write_json(staged / "_research_generation.json", generation_payload)
            row_counts[spec.dataset_name] = frame.height
            items.append(
                AtomicPublishItem(
                    target=spec.relative_path,
                    staged=staged.relative_to(root),
                )
            )
            del frame

        promotion = derivations.promotion_queue.select(list(PROMOTION_SCHEMA)).cast(
            PROMOTION_SCHEMA,
            strict=True,
        )
        promotion_staged = staging_root / "promotion"
        write_parquet_dataset(promotion, promotion_staged)
        write_snapshot_meta(
            promotion_staged,
            dataset_name="alpha_factory_promotion_queue",
            frame=promotion,
            schema_version=SCHEMA_VERSION,
            generated_at=manifest.generated_at,
        )
        atomic_write_json(
            promotion_staged / "_research_generation.json",
            generation_payload,
        )
        row_counts["alpha_factory_promotion_queue"] = promotion.height
        items.append(
            AtomicPublishItem(
                target=ALPHA_FACTORY_PROMOTION_QUEUE_DATASET,
                staged=promotion_staged.relative_to(root),
            )
        )

        for name, target, frame in (
            (
                "strategy_evidence_sample",
                STRATEGY_EVIDENCE_SAMPLE_DATASET,
                merged_sample,
            ),
            ("strategy_evidence", STRATEGY_EVIDENCE_DATASET, merged_summary),
        ):
            staged = staging_root / name
            write_parquet_dataset(frame, staged)
            write_snapshot_meta(
                staged,
                dataset_name=name,
                frame=frame,
                schema_version=SCHEMA_VERSION,
                generated_at=manifest.generated_at,
            )
            atomic_write_json(staged / "_research_generation.json", generation_payload)
            candidate_evidence_sidecar = (
                root / target / "_v5_candidate_evidence_generation.json"
            )
            if candidate_evidence_sidecar.is_file():
                shutil.copy2(
                    candidate_evidence_sidecar,
                    staged / candidate_evidence_sidecar.name,
                )
            row_counts[name] = frame.height
            items.append(
                AtomicPublishItem(
                    target=target,
                    staged=staged.relative_to(root),
                )
            )

        reports_staged = staging_root / "reports"
        reports_staged.mkdir(parents=True, exist_ok=False)
        for name, payload in sorted(validated.reports.items()):
            if Path(name).name != name:
                raise ValueError("unsafe_alpha_factory_report_name")
            staged = reports_staged / name
            staged.write_bytes(payload)
            items.append(
                AtomicPublishItem(
                    target=Path("reports") / name,
                    staged=staged.relative_to(root),
                )
            )

        pointer = generation_payload | {
            "datasets": [
                *[spec.dataset_name for spec in ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS],
                "alpha_factory_promotion_queue",
                "strategy_evidence_sample",
                "strategy_evidence",
            ],
            "row_counts": row_counts,
        }
        commit_atomic_research_generation(
            root,
            transaction_name=ALPHA_FACTORY_TRANSACTION_NAME,
            generation_payload=pointer,
            pointer_path=ALPHA_FACTORY_GENERATION_POINTER,
            items=items,
            post_commit_validate=lambda: verify_alpha_factory_generation(
                root,
                manifest.generation_id,
                row_counts,
            ),
        )
        return row_counts
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def recover_alpha_factory_publication(lake_root: str | Path) -> bool:
    return recover_atomic_research_generation(
        lake_root,
        transaction_name=ALPHA_FACTORY_TRANSACTION_NAME,
        pointer_path=ALPHA_FACTORY_GENERATION_POINTER,
    )


def verify_alpha_factory_generation(
    lake_root: str | Path,
    generation_id: str,
    expected_rows: dict[str, int] | None = None,
) -> dict[str, int]:
    root = Path(lake_root)
    pointer = json.loads((root / ALPHA_FACTORY_GENERATION_POINTER).read_text("utf-8"))
    if pointer.get("generation_id") != generation_id:
        raise RuntimeError("alpha_factory_generation_pointer_mismatch")
    if (
        pointer.get("research_only") is not True
        or pointer.get("live_order_effect") != "none"
        or pointer.get("automatic_promotion") is not False
    ):
        raise RuntimeError("alpha_factory_generation_safety_mismatch")
    factor_binding = (
        pointer.get("factor_generation_id"),
        pointer.get("factor_generation_digest"),
        pointer.get("factor_generation_as_of_date"),
        pointer.get("factor_generation_published_at"),
        pointer.get("hypothesis_registry_digest"),
        pointer.get("trial_ledger_digest"),
        pointer.get("factor_generation_fresh"),
        pointer.get("factor_generation_hypothesis_ids"),
    )
    if any(value is None for value in factor_binding):
        raise RuntimeError("alpha_factory_generation_factor_binding_missing")
    rows = {str(key): int(value) for key, value in dict(pointer.get("row_counts") or {}).items()}
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError("alpha_factory_generation_row_count_mismatch")
    targets = {
        **{
            spec.dataset_name: spec.relative_path
            for spec in ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS
        },
        "alpha_factory_promotion_queue": ALPHA_FACTORY_PROMOTION_QUEUE_DATASET,
        "strategy_evidence_sample": STRATEGY_EVIDENCE_SAMPLE_DATASET,
        "strategy_evidence": STRATEGY_EVIDENCE_DATASET,
    }
    if set(rows) != set(targets):
        raise RuntimeError("alpha_factory_generation_dataset_set_mismatch")
    for dataset_name, target in targets.items():
        metadata = json.loads(
            (root / target / "_research_generation.json").read_text("utf-8")
        )
        if metadata.get("generation_id") != generation_id:
            raise RuntimeError(f"alpha_factory_dataset_generation_mismatch:{target}")
        if count_parquet_rows(root / target) != rows[dataset_name]:
            raise RuntimeError(f"alpha_factory_dataset_row_count_mismatch:{target}")
    return rows
