from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from quant_lab.data.lake import (
    count_parquet_rows,
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
)
from quant_lab.research.alpha_factory.factory import (
    SOURCE_NAME as ALPHA_FACTORY_SOURCE_NAME,
)
from quant_lab.research.second_stage_alpha_factory import (
    SOURCE_NAME as SECOND_STAGE_ALPHA_FACTORY_SOURCE_NAME,
)
from quant_lab.research.strategy_evidence import (
    SAMPLE_SCHEMA,
    STRATEGY_EVIDENCE_SAMPLE_DATASET,
    STRATEGY_EVIDENCE_SAMPLE_KEY_COLUMNS,
    SUMMARY_SCHEMA,
    normalize_strategy_evidence_decisions,
)
from quant_lab.research_plane.atomic_publish import (
    AtomicPublishItem,
    commit_atomic_research_generation,
    recover_atomic_research_generation,
)
from quant_lab.research_plane.result import ValidatedAlphaFactoryResult
from quant_lab.research_plane.signatures import model_content_sha256

ALPHA_FACTORY_GENERATION_POINTER = Path("gold") / "alpha_factory_generation.json"
ALPHA_FACTORY_GENERATION_SCHEMA = "alpha_factory_generation.v2"
ALPHA_FACTORY_TRANSACTION_NAME = "alpha_factory"
ALPHA_FACTORY_PUBLISH_DUCKDB_MEMORY_LIMIT = "1GB"
ALPHA_FACTORY_SHARED_MANAGED_SCOPES = {
    "strategy_evidence_sample": {
        "scope": "managed_source",
        "source": SECOND_STAGE_ALPHA_FACTORY_SOURCE_NAME,
    },
    "strategy_evidence": {
        "scope": "managed_source",
        "source": ALPHA_FACTORY_SOURCE_NAME,
    },
}
ALPHA_FACTORY_SHARED_PRIMARY_KEYS = {
    "strategy_evidence_sample": (
        "source",
        *STRATEGY_EVIDENCE_SAMPLE_KEY_COLUMNS,
    ),
    "strategy_evidence": (
        "source",
        "strategy",
        "evidence_version",
        "as_of_date",
        "strategy_candidate",
        "symbol",
        "regime_state",
        "horizon_hours",
    ),
}


def publish_alpha_factory_generation(
    lake_root: str | Path,
    validated: ValidatedAlphaFactoryResult,
) -> dict[str, int]:
    """Derive cloud-owned outputs and atomically publish one Alpha generation."""
    root = Path(lake_root)
    recover_alpha_factory_publication(root)
    _remove_orphan_alpha_factory_staging(root)
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

    transaction_id = uuid.uuid4().hex
    staging_root = root / "gold" / f".__alpha_factory_stage_{transaction_id[:8]}"
    staging_root.mkdir(parents=True, exist_ok=False)
    generation_payload = {
        "schema_version": ALPHA_FACTORY_GENERATION_SCHEMA,
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
    managed_dataset_hashes: dict[str, str] = {}
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
                derivations.strategy_evidence_sample,
            ),
            (
                "strategy_evidence",
                STRATEGY_EVIDENCE_DATASET,
                normalize_strategy_evidence_decisions(derivations.strategy_evidence),
            ),
        ):
            schema = SAMPLE_SCHEMA if name == "strategy_evidence_sample" else SUMMARY_SCHEMA
            incoming = frame.select(list(schema)).cast(schema, strict=False)
            incoming_path = staging_root / f"incoming-{name}.parquet"
            incoming.write_parquet(incoming_path, compression="zstd")
            staged = staging_root / name
            total_rows = _stage_alpha_factory_shared_evidence(
                root / target,
                incoming_path,
                staged,
                dataset_name=name,
                as_of_date=manifest.as_of_date.isoformat(),
            )
            _write_streaming_snapshot_meta(
                staged,
                dataset_name=name,
                row_count=total_rows,
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
            managed_rows, managed_digest = _alpha_factory_managed_shared_metrics(
                staged,
                name,
            )
            row_counts[name] = managed_rows
            managed_dataset_hashes[name] = managed_digest
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
            "row_count_scopes": _alpha_factory_row_count_scopes(),
            "managed_dataset_hashes": managed_dataset_hashes,
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


def _remove_orphan_alpha_factory_staging(root: Path) -> None:
    gold = root / "gold"
    if not gold.is_dir():
        return
    for candidate in gold.glob(".__alpha_factory_stage_*"):
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate, ignore_errors=True)


def _stage_alpha_factory_shared_evidence(
    existing_root: Path,
    incoming_path: Path,
    staged_root: Path,
    *,
    dataset_name: str,
    as_of_date: str,
) -> int:
    """Replace the current day and upsert Alpha-owned history with bounded memory."""
    try:
        managed_source = ALPHA_FACTORY_SHARED_MANAGED_SCOPES[dataset_name]["source"]
    except KeyError as exc:
        raise ValueError(
            f"alpha_factory_unknown_shared_dataset:{dataset_name}"
        ) from exc
    schema = SAMPLE_SCHEMA if dataset_name == "strategy_evidence_sample" else SUMMARY_SCHEMA
    staged_root.mkdir(parents=True, exist_ok=False)
    output_path = staged_root / "data.parquet"
    spill = staged_root / ".duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=False)
    existing_files = _alpha_factory_parquet_files(existing_root)
    columns = ",".join(_sql_identifier(column) for column in schema)
    incoming_sql = (
        f"SELECT {columns} FROM read_parquet("
        f"{_sql_literal(incoming_path)}, hive_partitioning=false)"
    )
    if existing_files:
        existing_sql = (
            f"SELECT {columns} FROM read_parquet("
            f"[{','.join(_sql_literal(path) for path in existing_files)}], "
            "union_by_name=true, hive_partitioning=false)"
        )
        managed_existing_predicate = (
            f"{_sql_identifier('source')} = {_sql_literal(managed_source)} "
            f"AND CAST({_sql_identifier('as_of_date')} AS VARCHAR) "
            f"IS DISTINCT FROM {_sql_literal(as_of_date)}"
        )
        unmanaged_existing_predicate = (
            f"{_sql_identifier('source')} IS DISTINCT FROM "
            f"{_sql_literal(managed_source)}"
        )
        key_columns = ",".join(
            _sql_identifier(column)
            for column in ALPHA_FACTORY_SHARED_PRIMARY_KEYS[dataset_name]
        )
        priority_column = _sql_identifier("__alpha_incoming_priority")
        managed_rows = (
            f"SELECT {columns}, 0 AS {priority_column} "
            f"FROM ({existing_sql}) WHERE {managed_existing_predicate} "
            "UNION ALL BY NAME "
            f"SELECT {columns}, 1 AS {priority_column} FROM ({incoming_sql})"
        )
        deduplicated_managed_rows = (
            f"SELECT {columns} FROM ("
            f"SELECT {columns}, ROW_NUMBER() OVER ("
            f"PARTITION BY {key_columns} "
            f"ORDER BY {priority_column} DESC, "
            f"{_sql_identifier('created_at')} DESC NULLS LAST"
            ") AS __alpha_row_number "
            f"FROM ({managed_rows})"
            ") WHERE __alpha_row_number = 1"
        )
        query = (
            f"SELECT {columns} FROM ({existing_sql}) "
            f"WHERE {unmanaged_existing_predicate} "
            "UNION ALL BY NAME "
            f"SELECT {columns} FROM ({deduplicated_managed_rows})"
        )
    else:
        query = incoming_sql
    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        connection.execute("SET threads = 1")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(
            f"SET memory_limit = '{ALPHA_FACTORY_PUBLISH_DUCKDB_MEMORY_LIMIT}'"
        )
        connection.execute(f"SET temp_directory = {_sql_literal(spill)}")
        connection.execute(
            f"COPY ({query}) TO {_sql_literal(output_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
    return count_parquet_rows(staged_root)


def _write_streaming_snapshot_meta(
    staged_root: Path,
    *,
    dataset_name: str,
    row_count: int,
    schema_version: str,
    generated_at: datetime,
) -> None:
    files = _alpha_factory_parquet_files(staged_root)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    atomic_write_json(
        staged_root / "_snapshot_meta.json",
        {
            "dataset": dataset_name,
            "generated_at": generated_at.isoformat(),
            "expires_at": None,
            "row_count": row_count,
            "source_sha": digest.hexdigest(),
            "file_count": len(files),
            "schema_version": schema_version,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )


def _alpha_factory_parquet_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*.parquet")
        if path.is_file()
        and all(not part.startswith((".", "__")) for part in path.relative_to(root).parts)
    )


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def verify_alpha_factory_generation(
    lake_root: str | Path,
    generation_id: str,
    expected_rows: dict[str, int] | None = None,
) -> dict[str, int]:
    root = Path(lake_root)
    pointer = json.loads((root / ALPHA_FACTORY_GENERATION_POINTER).read_text("utf-8"))
    if pointer.get("schema_version") != ALPHA_FACTORY_GENERATION_SCHEMA:
        raise RuntimeError("alpha_factory_generation_pointer_schema_mismatch")
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
    row_count_scopes = {
        str(key): value
        for key, value in dict(pointer.get("row_count_scopes") or {}).items()
    }
    expected_scopes = _alpha_factory_row_count_scopes()
    if row_count_scopes != expected_scopes:
        raise RuntimeError("alpha_factory_generation_row_count_scope_mismatch")
    managed_dataset_hashes = {
        str(key): str(value)
        for key, value in dict(pointer.get("managed_dataset_hashes") or {}).items()
    }
    if (
        set(managed_dataset_hashes) != set(ALPHA_FACTORY_SHARED_MANAGED_SCOPES)
        or any(len(value) != 64 for value in managed_dataset_hashes.values())
    ):
        raise RuntimeError("alpha_factory_generation_managed_hash_set_mismatch")
    if set(rows) != set(targets):
        raise RuntimeError("alpha_factory_generation_dataset_set_mismatch")
    for dataset_name, target in targets.items():
        metadata = json.loads(
            (root / target / "_research_generation.json").read_text("utf-8")
        )
        if metadata.get("generation_id") != generation_id:
            raise RuntimeError(f"alpha_factory_dataset_generation_mismatch:{target}")
        if dataset_name in ALPHA_FACTORY_SHARED_MANAGED_SCOPES:
            actual_rows, actual_digest = _alpha_factory_managed_shared_metrics(
                root / target,
                dataset_name,
            )
        else:
            actual_rows = count_parquet_rows(root / target)
            actual_digest = None
        if actual_rows != rows[dataset_name]:
            raise RuntimeError(f"alpha_factory_dataset_row_count_mismatch:{target}")
        if (
            actual_digest is not None
            and actual_digest != managed_dataset_hashes[dataset_name]
        ):
            raise RuntimeError(f"alpha_factory_dataset_managed_hash_mismatch:{target}")
    return rows


def _alpha_factory_row_count_scopes() -> dict[str, dict[str, str]]:
    scopes = {
        spec.dataset_name: {"scope": "dataset_total"}
        for spec in ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS
    }
    scopes["alpha_factory_promotion_queue"] = {"scope": "dataset_total"}
    scopes.update(ALPHA_FACTORY_SHARED_MANAGED_SCOPES)
    return scopes


def _alpha_factory_managed_shared_frame(
    frame: pl.DataFrame,
    dataset_name: str,
) -> pl.DataFrame:
    try:
        scope = ALPHA_FACTORY_SHARED_MANAGED_SCOPES[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"alpha_factory_unknown_shared_dataset:{dataset_name}"
        ) from exc
    schema = SAMPLE_SCHEMA if dataset_name == "strategy_evidence_sample" else SUMMARY_SCHEMA
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    missing = sorted(set(schema) - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"alpha_factory_shared_dataset_columns_missing:{dataset_name}:"
            + ",".join(missing)
        )
    return (
        frame.filter(
            pl.col("source").cast(pl.Utf8, strict=False) == scope["source"]
        )
        .select(list(schema))
        .cast(schema, strict=False)
    )


def _alpha_factory_managed_shared_metrics(
    dataset_root: Path,
    dataset_name: str,
) -> tuple[int, str]:
    """Stream aggregate Alpha-owned rows without materializing shared history."""
    try:
        scope = ALPHA_FACTORY_SHARED_MANAGED_SCOPES[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"alpha_factory_unknown_shared_dataset:{dataset_name}"
        ) from exc
    schema = SAMPLE_SCHEMA if dataset_name == "strategy_evidence_sample" else SUMMARY_SCHEMA
    files = sorted(path for path in dataset_root.rglob("*.parquet") if path.is_file())
    if not files:
        empty = pl.DataFrame(schema=schema)
        return 0, _alpha_factory_managed_shared_digest(dataset_name, empty)
    duplicate_rows = _alpha_factory_managed_shared_duplicate_rows(
        dataset_root,
        dataset_name,
        files=files,
    )
    if duplicate_rows:
        raise RuntimeError(
            "alpha_factory_shared_dataset_duplicate_primary_key:"
            f"{dataset_name}:{duplicate_rows}"
        )
    lazy = pl.scan_parquet([str(path) for path in files], extra_columns="ignore")
    missing = sorted(set(schema) - set(lazy.collect_schema().names()))
    if missing:
        raise RuntimeError(
            f"alpha_factory_shared_dataset_columns_missing:{dataset_name}:"
            + ",".join(missing)
        )
    managed = (
        lazy.filter(
            pl.col("source").cast(pl.Utf8, strict=False) == scope["source"]
        )
        .select(
            [
                pl.col(column).cast(dtype, strict=False).alias(column)
                for column, dtype in schema.items()
            ]
        )
    )
    columns = sorted(schema)
    row = pl.struct(columns)
    values = (
        managed.select(
            [
                pl.len().alias("row_count"),
                row.hash(seed=0).sum().alias("hash_sum_0"),
                row.hash(seed=1).sum().alias("hash_sum_1"),
                row.hash(seed=2).min().alias("hash_min"),
                row.hash(seed=3).max().alias("hash_max"),
            ]
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    aggregates = {name: int(value or 0) for name, value in values.items()}
    digest = model_content_sha256(
        {
            "schema_version": "alpha_factory_managed_dataset_digest.v1",
            "dataset_name": dataset_name,
            "managed_scope": ALPHA_FACTORY_SHARED_MANAGED_SCOPES[dataset_name],
            "schema": [(column, str(schema[column])) for column in columns],
            **aggregates,
        }
    )
    return aggregates["row_count"], digest


def _alpha_factory_managed_shared_duplicate_rows(
    dataset_root: Path,
    dataset_name: str,
    *,
    files: list[Path] | None = None,
) -> int:
    try:
        managed_source = ALPHA_FACTORY_SHARED_MANAGED_SCOPES[dataset_name]["source"]
        primary_keys = ALPHA_FACTORY_SHARED_PRIMARY_KEYS[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"alpha_factory_unknown_shared_dataset:{dataset_name}"
        ) from exc
    parquet_files = files or _alpha_factory_parquet_files(dataset_root)
    if not parquet_files:
        return 0
    source = (
        "read_parquet("
        f"[{','.join(_sql_literal(path) for path in parquet_files)}], "
        "union_by_name=true, hive_partitioning=false)"
    )
    keys = ",".join(_sql_identifier(column) for column in primary_keys)
    with tempfile.TemporaryDirectory(prefix="quant-lab-alpha-verify-") as temp_root:
        connection = duckdb.connect(database=":memory:", read_only=False)
        try:
            connection.execute("SET threads = 1")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute("SET enable_progress_bar = false")
            connection.execute(
                f"SET memory_limit = '{ALPHA_FACTORY_PUBLISH_DUCKDB_MEMORY_LIMIT}'"
            )
            connection.execute(
                f"SET temp_directory = {_sql_literal(Path(temp_root) / 'duckdb')}"
            )
            row = connection.execute(
                "SELECT COALESCE(SUM(row_count - 1), 0) FROM ("
                f"SELECT COUNT(*) AS row_count FROM {source} "
                f"WHERE {_sql_identifier('source')} = {_sql_literal(managed_source)} "
                f"GROUP BY {keys} HAVING COUNT(*) > 1"
                ")"
            ).fetchone()
        finally:
            connection.close()
    return int(row[0] or 0)


def _alpha_factory_managed_shared_digest(
    dataset_name: str,
    frame: pl.DataFrame,
) -> str:
    columns = sorted(frame.columns)
    schema = [(column, str(frame.schema[column])) for column in columns]
    if frame.is_empty():
        aggregates = {
            "row_count": 0,
            "hash_sum_0": 0,
            "hash_sum_1": 0,
            "hash_min": 0,
            "hash_max": 0,
        }
    else:
        row = pl.struct(columns)
        values = frame.select(
            [
                pl.len().alias("row_count"),
                row.hash(seed=0).sum().alias("hash_sum_0"),
                row.hash(seed=1).sum().alias("hash_sum_1"),
                row.hash(seed=2).min().alias("hash_min"),
                row.hash(seed=3).max().alias("hash_max"),
            ]
        ).row(0, named=True)
        aggregates = {name: int(value or 0) for name, value in values.items()}
    return model_content_sha256(
        {
            "schema_version": "alpha_factory_managed_dataset_digest.v1",
            "dataset_name": dataset_name,
            "managed_scope": ALPHA_FACTORY_SHARED_MANAGED_SCOPES[dataset_name],
            "schema": schema,
            **aggregates,
        }
    )
