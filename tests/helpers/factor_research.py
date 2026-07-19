from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.factor_research.outputs import FACTOR_RESEARCH_OUTPUT_SPECS
from quant_lab.research.factor_research.registry import (
    FACTOR_RETIREMENT_DATASET,
    FACTOR_RETIREMENT_SCHEMA,
    HYPOTHESIS_REGISTRY_SCHEMA,
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
    TRIAL_LEDGER_SCHEMA,
)
from quant_lab.research_plane.factor_research_publish import (
    FACTOR_RESEARCH_GENERATION_POINTER,
)


def seed_verified_factor_generation(
    lake_root: Path,
    *,
    as_of_date: date,
    hypothesis_ids: tuple[str, ...] = ("hypothesis.test",),
    generation_id: str = "factor-research-test-generation",
    generation_digest: str = "a" * 64,
) -> None:
    """Write a minimal verified generation around any existing factor fixture rows."""
    data_snapshot_id = "factor-input-" + "b" * 24
    targets = {
        "research_hypothesis_registry": (
            RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
            HYPOTHESIS_REGISTRY_SCHEMA,
        ),
        "research_trial_ledger": (
            RESEARCH_TRIAL_LEDGER_DATASET,
            TRIAL_LEDGER_SCHEMA,
        ),
        "factor_retirement": (FACTOR_RETIREMENT_DATASET, FACTOR_RETIREMENT_SCHEMA),
        **{
            spec.dataset_name: (spec.relative_path, spec.schema)
            for spec in FACTOR_RESEARCH_OUTPUT_SPECS
        },
    }
    row_counts: dict[str, int] = {}
    for dataset_name, (relative_path, schema) in targets.items():
        existing = read_parquet_dataset(lake_root / relative_path)
        frame = _normalize(existing, schema)
        if dataset_name == "factor_candidate" and not frame.is_empty():
            frame = frame.with_columns(
                pl.lit(hypothesis_ids[0]).alias("hypothesis_id"),
                pl.lit(1).alias("hypothesis_version"),
                pl.lit(data_snapshot_id).alias("data_snapshot_id"),
                pl.lit(as_of_date.isoformat()).alias("as_of_date"),
                pl.lit("SIGNAL_VALID").alias("candidate_state"),
                pl.lit("PASS").alias("signal_validity"),
                pl.lit("PASS").alias("portfolio_validity"),
                pl.lit("RESEARCH_ONLY").alias("deployment_readiness"),
                pl.lit(True).alias("research_only"),
                pl.lit("none").alias("live_order_effect"),
                pl.lit(False).alias("automatic_promotion"),
                pl.lit(0.0).alias("max_live_notional_usdt"),
            )
        if dataset_name == "factor_value" and not frame.is_empty():
            frame = frame.with_columns(
                pl.lit(hypothesis_ids[0]).alias("hypothesis_id"),
                pl.lit(1).alias("hypothesis_version"),
                pl.lit("test-recipe").alias("feature_recipe_id"),
                pl.lit(data_snapshot_id).alias("data_snapshot_id"),
                pl.lit(True).alias("research_only"),
                pl.lit("none").alias("live_order_effect"),
            )
        frame = frame.select(list(schema)).cast(schema, strict=False)
        write_parquet_dataset(frame, lake_root / relative_path)
        row_counts[dataset_name] = frame.height

    payload = {
        "schema_version": "factor_research_generation.v1",
        "generation_id": generation_id,
        "factor_generation_digest": generation_digest,
        "task_id": "factor-research-test-task",
        "snapshot_id": "factor-research-test-snapshot",
        "hypothesis_ids": list(hypothesis_ids),
        "trial_ids": ["trial.test"],
        "quant_lab_commit": "c" * 40,
        "worker_commit": "c" * 40,
        "data_snapshot_digest": "b" * 64,
        "hypothesis_registry_digest": "d" * 64,
        "trial_ledger_digest": "e" * 64,
        "test_count": 1,
        "multiple_testing_family": "test-family",
        "as_of_date": as_of_date.isoformat(),
        "published_at": datetime.combine(
            as_of_date,
            datetime.min.time(),
            tzinfo=UTC,
        ).isoformat(),
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "datasets": list(row_counts),
        "row_counts": row_counts,
    }
    for _dataset_name, (relative_path, _schema) in targets.items():
        (lake_root / relative_path / "_research_generation.json").write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    pointer = lake_root / FACTOR_RESEARCH_GENERATION_POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _normalize(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    normalized = frame
    for column, dtype in schema.items():
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return normalized.select(list(schema)).cast(schema, strict=False)
