from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from quant_lab.data.lake import count_parquet_rows, read_parquet_dataset
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.factors.factory import (
    FACTOR_CANDIDATE_DATASET,
    FACTOR_CANDIDATE_SCHEMA,
    FACTOR_CORRELATION_DAILY_DATASET,
    FACTOR_DEFINITION_DATASET,
    FACTOR_EVIDENCE_DATASET,
    FACTOR_VALUE_DATASET,
    build_factor_definition_frame,
    derive_factor_candidate_frame,
)
from quant_lab.research.factor_research.outputs import FACTOR_RESEARCH_OUTPUT_SPECS
from quant_lab.research.factor_research.registry import (
    FACTOR_EXTERNAL_AUDIT_EVIDENCE_DATASET,
    FACTOR_RETIREMENT_DATASET,
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
)
from quant_lab.research_plane.atomic_publish import (
    AtomicPublishItem,
    commit_atomic_research_generation,
    recover_atomic_research_generation,
)
from quant_lab.research_plane.factor_factory_result import ValidatedFactorFactoryResult
from quant_lab.research_plane.factor_research_publish import (
    FACTOR_RESEARCH_GENERATION_POINTER,
    FACTOR_RESEARCH_GENERATION_SCHEMA,
    FACTOR_RESEARCH_SOURCE,
    verify_factor_research_generation,
)

FACTOR_FACTORY_GENERATION_POINTER = Path("gold") / "factor_factory_generation.json"
FACTOR_FACTORY_GENERATION_SCHEMA = "factor_factory_generation.v1"
FACTOR_FACTORY_TRANSACTION_NAME = "factor_factory"
FACTOR_FACTORY_DATASETS = {
    "factor_definition": FACTOR_DEFINITION_DATASET,
    "factor_value": FACTOR_VALUE_DATASET,
    "factor_evidence": FACTOR_EVIDENCE_DATASET,
    "factor_candidate": FACTOR_CANDIDATE_DATASET,
    "factor_correlation_daily": FACTOR_CORRELATION_DAILY_DATASET,
}
FACTOR_FACTORY_PRIMARY_KEYS = {
    "factor_definition": ("factor_id", "factor_version"),
    "factor_value": ("factor_id", "factor_version", "symbol", "timeframe", "ts"),
    "factor_evidence": (
        "as_of_date",
        "factor_id",
        "factor_version",
        "timeframe",
        "horizon_bars",
        "decision_delay_bars",
    ),
    "factor_candidate": (
        "as_of_date",
        "factor_id",
        "factor_version",
        "timeframe",
    ),
    "factor_correlation_daily": (
        "as_of_date",
        "factor_id_left",
        "factor_id_right",
        "factor_version",
        "timeframe",
    ),
}


def publish_factor_factory_generation(
    lake_root: str | Path,
    validated: ValidatedFactorFactoryResult,
) -> dict[str, Any]:
    """Atomically upsert exactly five Factor Gold datasets and their generation pointer."""

    root = Path(lake_root).resolve(strict=True)
    recover_factor_factory_publication(root)
    manifest = validated.manifest
    if manifest.completed_no_update:
        return {
            "published": False,
            "completed_no_update": True,
            "no_update_reason": manifest.no_update_reason,
            "generation_id": manifest.generation_id,
            "row_counts": {
                name: count_parquet_rows(root / path)
                for name, path in FACTOR_FACTORY_DATASETS.items()
            },
        }
    _validate_previous_generation_binding(root, manifest)
    candidate = derive_factor_candidate_frame(
        pl.read_parquet(validated.output_paths["factor_evidence"]),
        as_of_date=manifest.as_of_date,
        created_at=manifest.completed_at,
    )
    _validate_cloud_candidate(candidate)
    transaction_id = uuid.uuid4().hex
    staging_root = root / "gold" / f".__factor_factory_stage_{transaction_id[:8]}"
    staging_root.mkdir(parents=True, exist_ok=False)
    items: list[AtomicPublishItem] = []
    row_counts: dict[str, int] = {}
    try:
        candidate_path = staging_root / "incoming-factor-candidate.parquet"
        candidate.select(list(FACTOR_CANDIDATE_SCHEMA)).cast(
            FACTOR_CANDIDATE_SCHEMA, strict=True
        ).write_parquet(candidate_path, compression="zstd")
        incoming_by_dataset = {
            "factor_definition": [staging_root / "incoming-factor-definition.parquet"],
            "factor_value": list(validated.value_partition_paths),
            "factor_evidence": [validated.output_paths["factor_evidence"]],
            "factor_candidate": [candidate_path],
            "factor_correlation_daily": [validated.output_paths["factor_correlation_daily"]],
        }
        definitions = build_factor_definition_frame(
            validated.snapshot.factor_plan.factor_spec_models(),
            created_at=manifest.generated_at,
        )
        definitions.select(list(definitions.schema)).write_parquet(
            incoming_by_dataset["factor_definition"][0], compression="zstd"
        )
        staged_by_dataset: dict[str, Path] = {}
        for index, (dataset_name, target) in enumerate(FACTOR_FACTORY_DATASETS.items()):
            staged = staging_root / f"dataset-{index:02d}-{dataset_name}"
            rows = _stage_streaming_upsert(
                root / target,
                incoming_by_dataset[dataset_name],
                staged,
                key_columns=FACTOR_FACTORY_PRIMARY_KEYS[dataset_name],
                replace_scope=(
                    {
                        "as_of_date": manifest.as_of_date.isoformat(),
                        "source": "factors.factory.v0.1",
                    }
                    if dataset_name
                    in {
                        "factor_evidence",
                        "factor_candidate",
                        "factor_correlation_daily",
                    }
                    else None
                ),
            )
            _validate_staged_dataset(
                staged,
                dataset_name=dataset_name,
                key_columns=FACTOR_FACTORY_PRIMARY_KEYS[dataset_name],
            )
            staged_by_dataset[dataset_name] = staged
            row_counts[dataset_name] = rows
            items.append(AtomicPublishItem(target=target, staged=staged.relative_to(root)))

        dataset_hashes = {
            name: _dataset_digest(path) for name, path in staged_by_dataset.items()
        }
        generation_digest = _generation_digest(validated, candidate_path)
        generation_payload = {
            "schema_version": FACTOR_FACTORY_GENERATION_SCHEMA,
            "generation_id": manifest.generation_id,
            "generation_digest": generation_digest,
            "task_id": manifest.task_id,
            "snapshot_id": manifest.snapshot_id,
            "quant_lab_commit": manifest.quant_lab_commit,
            "worker_commit": manifest.worker_commit,
            "factor_plan_digest": manifest.factor_plan_digest,
            "feature_set": manifest.feature_set,
            "feature_version": manifest.feature_version,
            "factor_version": manifest.factor_version,
            "timeframe": manifest.timeframe,
            "horizon_bars": list(manifest.horizon_bars),
            "decision_delay_bars": manifest.decision_delay_bars,
            "min_samples": validated.snapshot.min_samples,
            "top_quantile": validated.snapshot.top_quantile,
            "cost_quantile": validated.snapshot.cost_quantile,
            "source_input_digest": manifest.source_input_digest,
            "cost_input_digest": manifest.cost_input_digest,
            "previous_generation_id": manifest.previous_generation_id,
            "previous_generation_digest": manifest.previous_generation_digest,
            "as_of_date": manifest.as_of_date.isoformat(),
            "factor_ids": list(manifest.factor_ids),
            "factor_count": manifest.factor_count,
            "result_mode": manifest.result_mode,
            "history_mode": manifest.history_mode,
            "datasets": list(FACTOR_FACTORY_DATASETS),
            "row_counts": row_counts,
            "dataset_hashes": dataset_hashes,
            "primary_keys": {
                name: list(keys) for name, keys in FACTOR_FACTORY_PRIMARY_KEYS.items()
            },
            "published_at": datetime.now(UTC).isoformat(),
            "diagnostic_only": True,
            "research_only": True,
            "live_order_effect": "none_read_only_research",
            "automatic_promotion": False,
            "max_live_notional_usdt": 0,
            "manual_review_required": True,
        }
        for dataset_name, staged in staged_by_dataset.items():
            _write_generation_sidecars(
                staged,
                dataset_name=dataset_name,
                generation_payload=generation_payload,
                row_count=row_counts[dataset_name],
            )

        factor_research_migration = _stage_factor_research_coownership_pointer(
            root,
            staging_root,
            staged_by_dataset=staged_by_dataset,
            row_counts=row_counts,
            coowner_generation_id=manifest.generation_id,
        )
        if factor_research_migration is not None:
            items.append(
                AtomicPublishItem(
                    target=FACTOR_RESEARCH_GENERATION_POINTER,
                    staged=factor_research_migration.relative_to(root),
                )
            )

        def post_commit_validate() -> None:
            verify_factor_factory_generation(
                root,
                manifest.generation_id,
                expected_rows=row_counts,
            )
            if factor_research_migration is not None:
                pointer = json.loads((root / FACTOR_RESEARCH_GENERATION_POINTER).read_text("utf-8"))
                verify_factor_research_generation(root, str(pointer["generation_id"]))

        commit_atomic_research_generation(
            root,
            transaction_name=FACTOR_FACTORY_TRANSACTION_NAME,
            generation_payload=generation_payload,
            pointer_path=FACTOR_FACTORY_GENERATION_POINTER,
            items=items,
            post_commit_validate=post_commit_validate,
        )
        return {
            "published": True,
            "completed_no_update": False,
            "generation_id": manifest.generation_id,
            "generation_digest": generation_digest,
            "row_counts": row_counts,
        }
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def recover_factor_factory_publication(lake_root: str | Path) -> bool:
    return recover_atomic_research_generation(
        lake_root,
        transaction_name=FACTOR_FACTORY_TRANSACTION_NAME,
        pointer_path=FACTOR_FACTORY_GENERATION_POINTER,
    )


def verify_factor_factory_generation(
    lake_root: str | Path,
    generation_id: str,
    *,
    expected_rows: dict[str, int] | None = None,
) -> dict[str, int]:
    root = Path(lake_root)
    pointer = json.loads((root / FACTOR_FACTORY_GENERATION_POINTER).read_text("utf-8"))
    if pointer.get("schema_version") != FACTOR_FACTORY_GENERATION_SCHEMA:
        raise RuntimeError("factor_factory_generation_schema_mismatch")
    if pointer.get("generation_id") != generation_id:
        raise RuntimeError("factor_factory_generation_pointer_mismatch")
    digest = str(pointer.get("generation_digest") or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError("factor_factory_generation_digest_invalid")
    safety = {
        "diagnostic_only": True,
        "research_only": True,
        "live_order_effect": "none_read_only_research",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "manual_review_required": True,
    }
    if any(pointer.get(field) != value for field, value in safety.items()):
        raise RuntimeError("factor_factory_generation_safety_mismatch")
    if (
        pointer.get("result_mode") != "PARITY_FULL"
        or pointer.get("history_mode") != "bootstrap_full"
    ):
        raise RuntimeError("factor_factory_generation_history_mode_mismatch")
    rows = {str(key): int(value) for key, value in dict(pointer.get("row_counts") or {}).items()}
    dataset_hashes = {
        str(key): str(value) for key, value in dict(pointer.get("dataset_hashes") or {}).items()
    }
    if set(rows) != set(FACTOR_FACTORY_DATASETS):
        raise RuntimeError("factor_factory_generation_dataset_set_mismatch")
    if set(dataset_hashes) != set(FACTOR_FACTORY_DATASETS):
        raise RuntimeError("factor_factory_generation_hash_set_mismatch")
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError("factor_factory_generation_row_count_mismatch")
    for dataset_name, target in FACTOR_FACTORY_DATASETS.items():
        dataset_root = root / target
        sidecar = json.loads((dataset_root / "_factor_factory_generation.json").read_text("utf-8"))
        if sidecar.get("generation_id") != generation_id:
            raise RuntimeError(f"factor_factory_dataset_generation_mismatch:{dataset_name}")
        if sidecar.get("generation_digest") != digest:
            raise RuntimeError(f"factor_factory_dataset_digest_mismatch:{dataset_name}")
        if count_parquet_rows(dataset_root) != rows[dataset_name]:
            raise RuntimeError(f"factor_factory_dataset_row_count_mismatch:{dataset_name}")
        if _dataset_digest(dataset_root) != dataset_hashes[dataset_name]:
            raise RuntimeError(f"factor_factory_dataset_digest_mismatch:{dataset_name}")
        _validate_staged_dataset(
            dataset_root,
            dataset_name=dataset_name,
            key_columns=FACTOR_FACTORY_PRIMARY_KEYS[dataset_name],
        )
    candidates = read_parquet_dataset(root / FACTOR_CANDIDATE_DATASET)
    _validate_published_candidates(
        candidates,
        as_of_date=str(pointer["as_of_date"]),
    )
    return rows


def _validate_previous_generation_binding(root: Path, manifest: Any) -> None:
    pointer = root / FACTOR_FACTORY_GENERATION_POINTER
    if manifest.previous_generation_id is None:
        if pointer.exists():
            raise ValueError("factor_factory_result_superseded_by_generation")
        return
    if not pointer.is_file():
        raise ValueError("factor_factory_previous_generation_missing")
    payload = json.loads(pointer.read_text("utf-8"))
    if (
        payload.get("generation_id") != manifest.previous_generation_id
        or payload.get("generation_digest") != manifest.previous_generation_digest
    ):
        raise ValueError("factor_factory_result_superseded_by_generation")


def _stage_streaming_upsert(
    existing_root: Path,
    incoming_paths: list[Path],
    staged_root: Path,
    *,
    key_columns: tuple[str, ...],
    replace_scope: dict[str, object] | None = None,
) -> int:
    if not incoming_paths:
        raise ValueError("factor_factory_publish_incoming_missing")
    staged_root.mkdir(parents=True, exist_ok=False)
    output_path = staged_root / "data.parquet"
    temp_directory = staged_root / ".duckdb_tmp"
    temp_directory.mkdir(parents=True, exist_ok=False)
    existing_paths = (
        sorted(path for path in existing_root.rglob("*.parquet") if path.is_file())
        if existing_root.is_dir()
        else []
    )
    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        connection.execute("SET threads = 2")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET memory_limit = '768MB'")
        connection.execute(f"SET temp_directory = {_sql_literal(temp_directory)}")
        incoming_sql = _read_parquet_sql(incoming_paths)
        if existing_paths:
            existing_sql = _read_parquet_sql(existing_paths)
            if replace_scope:
                predicate = " AND ".join(
                    f"{_identifier(column)} = {_sql_literal(str(value))}"
                    for column, value in sorted(replace_scope.items())
                )
                existing_sql = f"SELECT * FROM ({existing_sql}) WHERE NOT ({predicate})"
            using = ",".join(_identifier(column) for column in key_columns)
            query = f"""
                WITH incoming AS ({incoming_sql}),
                retained AS (
                    SELECT existing.*
                    FROM ({existing_sql}) AS existing
                    ANTI JOIN incoming USING ({using})
                )
                SELECT * FROM retained
                UNION ALL BY NAME
                SELECT * FROM incoming
            """
        else:
            query = incoming_sql
        connection.execute(
            f"COPY ({query}) TO {_sql_literal(output_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
    finally:
        connection.close()
        shutil.rmtree(temp_directory, ignore_errors=True)
    _copy_existing_sidecars(existing_root, staged_root)
    return count_parquet_rows(staged_root)


def _validate_staged_dataset(
    dataset_root: Path,
    *,
    dataset_name: str,
    key_columns: tuple[str, ...],
) -> None:
    files = sorted(path for path in dataset_root.rglob("*.parquet") if path.is_file())
    if not files:
        raise ValueError(f"factor_factory_publish_dataset_missing:{dataset_name}")
    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        source = _read_parquet_sql(files)
        columns = {str(row[0]) for row in connection.execute(f"DESCRIBE {source}").fetchall()}
        missing = sorted(set(key_columns) - columns)
        if missing:
            raise ValueError(
                f"factor_factory_publish_primary_key_missing:{dataset_name}:{','.join(missing)}"
            )
        key_sql = ",".join(_identifier(column) for column in key_columns)
        duplicate = connection.execute(
            f"SELECT 1 FROM ({source}) GROUP BY {key_sql} HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate is not None:
            raise ValueError(f"factor_factory_publish_duplicate_key:{dataset_name}")
        null_predicate = " OR ".join(f"{_identifier(column)} IS NULL" for column in key_columns)
        null_row = connection.execute(
            f"SELECT 1 FROM ({source}) WHERE {null_predicate} LIMIT 1"
        ).fetchone()
        if null_row is not None:
            raise ValueError(f"factor_factory_publish_null_key:{dataset_name}")
    finally:
        connection.close()


def _validate_cloud_candidate(
    frame: pl.DataFrame,
    *,
    enforce_row_limit: bool = True,
) -> None:
    if frame.is_empty():
        return
    if enforce_row_limit and frame.height > 200:
        raise ValueError("factor_factory_candidate_limit_exceeded")
    required = {"as_of_date", "candidate_state", "manual_review_required", "source"}
    if not required.issubset(frame.columns):
        raise ValueError("factor_factory_candidate_schema_incomplete")
    allowed = {"KILL", "RESEARCH", "KEEP_SHADOW", "PAPER_READY"}
    invalid = frame.filter(
        ~pl.col("manual_review_required").fill_null(False)
        | ~pl.col("candidate_state").is_in(sorted(allowed))
        | (pl.col("source") != "factors.factory.v0.1")
    )
    if not invalid.is_empty():
        raise ValueError("factor_factory_candidate_safety_mismatch")


def _validate_published_candidates(frame: pl.DataFrame, *, as_of_date: str) -> None:
    if frame.is_empty():
        return
    if "source" not in frame.columns:
        raise ValueError("factor_factory_candidate_schema_incomplete")
    managed = frame.filter(pl.col("source") == "factors.factory.v0.1")
    _validate_cloud_candidate(managed, enforce_row_limit=False)
    current = managed.filter(pl.col("as_of_date") == as_of_date)
    _validate_cloud_candidate(current)


def _write_generation_sidecars(
    staged: Path,
    *,
    dataset_name: str,
    generation_payload: dict[str, Any],
    row_count: int,
) -> None:
    atomic_write_json(staged / "_factor_factory_generation.json", generation_payload)
    atomic_write_json(
        staged / "_snapshot_meta.json",
        {
            "dataset": dataset_name,
            "generated_at": generation_payload["published_at"],
            "row_count": row_count,
            "source_sha": generation_payload["generation_digest"],
            "file_count": sum(1 for path in staged.rglob("*.parquet") if path.is_file()),
            "schema_version": FACTOR_FACTORY_GENERATION_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def _stage_factor_research_coownership_pointer(
    root: Path,
    staging_root: Path,
    *,
    staged_by_dataset: dict[str, Path],
    row_counts: dict[str, int],
    coowner_generation_id: str,
) -> Path | None:
    pointer_path = root / FACTOR_RESEARCH_GENERATION_POINTER
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text("utf-8"))
    if pointer.get("schema_version") != FACTOR_RESEARCH_GENERATION_SCHEMA:
        raise RuntimeError("factor_research_generation_pointer_invalid_during_coownership")
    factor_targets = {
        spec.dataset_name: spec.relative_path for spec in FACTOR_RESEARCH_OUTPUT_SPECS
    }
    all_targets = {
        "research_hypothesis_registry": RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
        "research_trial_ledger": RESEARCH_TRIAL_LEDGER_DATASET,
        "factor_retirement": FACTOR_RETIREMENT_DATASET,
        "factor_external_audit_evidence": FACTOR_EXTERNAL_AUDIT_EVIDENCE_DATASET,
        **factor_targets,
    }
    total_counts = {
        str(name): int(value) for name, value in dict(pointer.get("row_counts") or {}).items()
    }
    managed_counts = {
        str(name): int(value)
        for name, value in dict(pointer.get("managed_row_counts") or {}).items()
    }
    for name, target in all_targets.items():
        dataset_root = staged_by_dataset.get(name, root / target)
        if name in row_counts:
            total_counts[name] = row_counts[name]
        elif name not in total_counts:
            total_counts[name] = count_parquet_rows(dataset_root)
        if name in factor_targets:
            managed_counts[name] = _count_source_rows(dataset_root, FACTOR_RESEARCH_SOURCE)
        elif name not in managed_counts:
            managed_counts[name] = count_parquet_rows(dataset_root)
    updated = pointer | {
        "row_counts": total_counts,
        "managed_row_counts": managed_counts,
        "shared_gold_ownership": True,
        "last_factor_factory_generation_id": coowner_generation_id,
        "coownership_observed_at": datetime.now(UTC).isoformat(),
    }
    staged = staging_root / "factor_research_generation.coowned.json"
    atomic_write_json(staged, updated)
    return staged


def _count_source_rows(dataset_root: Path, source: str) -> int:
    files = sorted(path for path in dataset_root.rglob("*.parquet") if path.is_file())
    if not files:
        return 0
    lazy = pl.scan_parquet([str(path) for path in files], extra_columns="ignore")
    if "source" not in lazy.collect_schema().names():
        return 0
    return int(
        lazy.filter(pl.col("source") == source).select(pl.len()).collect(engine="streaming").item()
    )


def _copy_existing_sidecars(existing_root: Path, staged_root: Path) -> None:
    if not existing_root.is_dir():
        return
    for source in existing_root.rglob("*"):
        if source.is_symlink():
            raise ValueError("factor_factory_publish_existing_symlink_forbidden")
        if not source.is_file() or source.suffix == ".parquet":
            continue
        relative = source.relative_to(existing_root)
        if relative.name in {"_snapshot_meta.json", "_factor_factory_generation.json"}:
            continue
        destination = staged_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _generation_digest(
    validated: ValidatedFactorFactoryResult,
    candidate_path: Path,
) -> str:
    manifest = validated.manifest
    payload = {
        "task_id": manifest.task_id,
        "snapshot_id": manifest.snapshot_id,
        "factor_plan_digest": manifest.factor_plan_digest,
        "source_input_digest": manifest.source_input_digest,
        "cost_input_digest": manifest.cost_input_digest,
        "previous_generation_id": manifest.previous_generation_id,
        "previous_generation_digest": manifest.previous_generation_digest,
        "factor_ids": list(manifest.factor_ids),
        "outputs": [
            [item.dataset_name, item.sha256]
            for item in sorted(manifest.outputs, key=lambda value: value.dataset_name)
        ],
        "value_partitions": [
            [item.relative_path, item.sha256]
            for item in sorted(manifest.value_partitions, key=lambda value: value.relative_path)
        ],
        "factor_candidate_sha256": _sha256_file(candidate_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_parquet_sql(paths: list[Path]) -> str:
    values = ",".join(_sql_literal(path) for path in paths)
    return f"SELECT * FROM read_parquet([{values}], union_by_name=true)"


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_digest(root: Path) -> str:
    payload = [
        [str(path.relative_to(root)).replace("\\", "/"), _sha256_file(path)]
        for path in sorted(
            candidate for candidate in root.rglob("*.parquet") if candidate.is_file()
        )
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
