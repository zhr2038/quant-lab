from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from quant_lab.data.lake import count_parquet_rows
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.research.candidate_labels import (
    CANDIDATE_LABEL_DATASET,
    CANDIDATE_OUTCOME_SUMMARY_DATASET,
    CANDIDATE_QUALITY_DATASET,
    derive_candidate_outcome_summary,
    derive_candidate_quality,
)
from quant_lab.research.candidate_labels import (
    SOURCE_NAME as CANDIDATE_LABEL_SOURCE,
)
from quant_lab.research.strategy_evidence import (
    SAMPLE_SCHEMA,
    STRATEGY_EVIDENCE_DATASET,
    STRATEGY_EVIDENCE_QUALITY_DATASET,
    STRATEGY_EVIDENCE_SAMPLE_DATASET,
    STRATEGY_EVIDENCE_SAMPLE_KEY_COLUMNS,
    SUMMARY_SCHEMA,
    derive_strategy_evidence_quality,
    normalize_strategy_evidence_samples,
    summarize_strategy_evidence,
)
from quant_lab.research.strategy_evidence import (
    SOURCE_NAME as STRATEGY_EVIDENCE_SOURCE,
)
from quant_lab.research_plane.atomic_publish import (
    AtomicPublishItem,
    commit_atomic_research_generation,
    recover_atomic_research_generation,
)
from quant_lab.research_plane.signatures import model_content_sha256
from quant_lab.research_plane.v5_candidate_evidence_result import (
    ValidatedV5CandidateEvidenceResult,
)

V5_CANDIDATE_EVIDENCE_GENERATION_POINTER = (
    Path("gold") / "v5_candidate_evidence_generation.json"
)
V5_CANDIDATE_EVIDENCE_GENERATION_SCHEMA = "v5_candidate_evidence_generation.v1"
V5_CANDIDATE_EVIDENCE_TRANSACTION_NAME = "v5_candidate_evidence"
V5_CANDIDATE_EVIDENCE_SIDECAR = "_v5_candidate_evidence_generation.json"
V5_CANDIDATE_EVIDENCE_DATASETS = {
    "v5_candidate_label": CANDIDATE_LABEL_DATASET,
    "v5_candidate_quality_daily": CANDIDATE_QUALITY_DATASET,
    "v5_candidate_outcome_summary": CANDIDATE_OUTCOME_SUMMARY_DATASET,
    "strategy_evidence_sample": STRATEGY_EVIDENCE_SAMPLE_DATASET,
    "strategy_evidence": STRATEGY_EVIDENCE_DATASET,
    "strategy_evidence_quality": STRATEGY_EVIDENCE_QUALITY_DATASET,
}
V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS = {
    "v5_candidate_label": ("strategy", "candidate_id", "horizon_hours"),
    "v5_candidate_quality_daily": ("strategy", "date"),
    "v5_candidate_outcome_summary": (
        "strategy",
        "date",
        "block_reason",
        "strategy_candidate",
        "symbol",
        "horizon_hours",
    ),
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
    "strategy_evidence_quality": (
        "source",
        "strategy",
        "evidence_version",
        "as_of_date",
        "severity",
        "warning_type",
    ),
}
V5_CANDIDATE_EVIDENCE_MANAGED_SCOPES = {
    "v5_candidate_label": {
        "strategy": "v5",
        "source": CANDIDATE_LABEL_SOURCE,
    },
    "v5_candidate_quality_daily": {
        "strategy": "v5",
        "source": CANDIDATE_LABEL_SOURCE,
    },
    "v5_candidate_outcome_summary": {
        "strategy": "v5",
        "source": CANDIDATE_LABEL_SOURCE,
    },
    "strategy_evidence_sample": {
        "strategy": "v5",
        "source": STRATEGY_EVIDENCE_SOURCE,
    },
    "strategy_evidence": {
        "strategy": "v5",
        "source": STRATEGY_EVIDENCE_SOURCE,
    },
    "strategy_evidence_quality": {
        "strategy": "v5",
        "source": STRATEGY_EVIDENCE_SOURCE,
    },
}
V5_CANDIDATE_EVIDENCE_ALLOWED_CLOUD_DECISIONS = frozenset(
    {
        "RESEARCH_ONLY",
        "KEEP_SHADOW",
        "REGIME_SHADOW",
        "KILL",
        "PAPER_READY",
        "LIVE_SMALL_READY",
    }
)


def publish_v5_candidate_evidence_generation(
    lake_root: str | Path,
    validated: ValidatedV5CandidateEvidenceResult,
    *,
    snapshot_root: str | Path,
) -> dict[str, Any]:
    """Cloud-derive control outputs and atomically publish exactly six Gold tables."""

    root = Path(lake_root).resolve(strict=True)
    recover_v5_candidate_evidence_publication(root)
    _validate_previous_generation_binding(root, validated)
    manifest = validated.manifest
    snapshot_files = Path(snapshot_root).resolve(strict=True) / "files"
    events = _read_snapshot_dataset(
        snapshot_files,
        validated.snapshot,
        "silver/v5_candidate_event",
    )
    run_summary = _read_snapshot_dataset(
        snapshot_files,
        validated.snapshot,
        "silver/v5_run_summary",
    )
    labels = _collect_paths(validated.label_paths)
    candidate_quality = derive_candidate_quality(
        events,
        labels,
        run_summary,
        as_of_date=manifest.as_of_date,
        created_at=manifest.completed_at,
    )
    candidate_outcome = derive_candidate_outcome_summary(
        labels,
        as_of_date=manifest.as_of_date,
        created_at=manifest.completed_at,
    )

    transaction_id = uuid.uuid4().hex
    staging_root = root / "gold" / f".__v5_candidate_evidence_stage_{transaction_id[:8]}"
    staging_root.mkdir(parents=True, exist_ok=False)
    staged_by_dataset: dict[str, Path] = {}
    items: list[AtomicPublishItem] = []
    try:
        sample_staged = staging_root / "dataset-03-strategy_evidence_sample"
        _stage_streaming_upsert(
            root / STRATEGY_EVIDENCE_SAMPLE_DATASET,
            list(validated.sample_paths),
            sample_staged,
            key_columns=V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS["strategy_evidence_sample"],
        )
        staged_by_dataset["strategy_evidence_sample"] = sample_staged

        summary_samples = _recent_managed_samples(
            sample_staged,
            as_of_date=manifest.as_of_date,
            lookback_days=manifest.lookback_days,
        )
        summary_rows = summarize_strategy_evidence(
            summary_samples,
            as_of_date=manifest.as_of_date,
        )
        summary = pl.DataFrame(summary_rows, schema=SUMMARY_SCHEMA, orient="row")
        quality = derive_strategy_evidence_quality(
            manifest.as_of_date,
            list(manifest.warnings),
            created_at=manifest.completed_at,
        )

        incoming_frames = {
            "v5_candidate_quality_daily": candidate_quality,
            "v5_candidate_outcome_summary": candidate_outcome,
            "strategy_evidence": summary,
            "strategy_evidence_quality": quality,
        }
        incoming_paths: dict[str, list[Path]] = {
            "v5_candidate_label": list(validated.label_paths),
            "strategy_evidence_sample": list(validated.sample_paths),
        }
        for name, frame in incoming_frames.items():
            path = staging_root / f"incoming-{name}.parquet"
            frame.write_parquet(path, compression="zstd")
            incoming_paths[name] = [path]

        for index, (dataset_name, target) in enumerate(
            V5_CANDIDATE_EVIDENCE_DATASETS.items()
        ):
            if dataset_name == "strategy_evidence_sample":
                staged = sample_staged
            else:
                staged = staging_root / f"dataset-{index:02d}-{dataset_name}"
                replace_scope = None
                if dataset_name in {"strategy_evidence", "strategy_evidence_quality"}:
                    incoming_has_rows = not incoming_frames[dataset_name].is_empty()
                    if incoming_has_rows:
                        replace_scope = {
                            **V5_CANDIDATE_EVIDENCE_MANAGED_SCOPES[dataset_name],
                            "as_of_date": manifest.as_of_date.isoformat(),
                        }
                _stage_streaming_upsert(
                    root / target,
                    incoming_paths[dataset_name],
                    staged,
                    key_columns=V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS[dataset_name],
                    replace_scope=replace_scope,
                )
                staged_by_dataset[dataset_name] = staged
            _validate_staged_dataset(
                staged,
                dataset_name=dataset_name,
                key_columns=V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS[dataset_name],
                managed_scope=V5_CANDIDATE_EVIDENCE_MANAGED_SCOPES[dataset_name],
            )
            items.append(AtomicPublishItem(target=target, staged=staged.relative_to(root)))

        managed_row_counts = {
            name: _managed_row_count(staged_by_dataset[name], name)
            for name in V5_CANDIDATE_EVIDENCE_DATASETS
        }
        dataset_hashes = {
            name: _managed_dataset_digest(staged_by_dataset[name], name)
            for name in V5_CANDIDATE_EVIDENCE_DATASETS
        }
        generation_digest = model_content_sha256(
            {
                "schema_version": "v5_candidate_evidence_generation_digest.v1",
                "task_id": manifest.task_id,
                "snapshot_id": manifest.snapshot_id,
                "previous_generation_id": manifest.previous_generation_id,
                "previous_generation_digest": manifest.previous_generation_digest,
                "result_outputs": [
                    [item.relative_path, item.sha256]
                    for item in sorted(manifest.outputs, key=lambda item: item.relative_path)
                ],
                "dataset_hashes": dataset_hashes,
                "managed_row_counts": managed_row_counts,
            }
        )
        generation_payload = {
            "schema_version": V5_CANDIDATE_EVIDENCE_GENERATION_SCHEMA,
            "generation_id": manifest.generation_id,
            "generation_digest": generation_digest,
            "task_id": manifest.task_id,
            "snapshot_id": manifest.snapshot_id,
            "quant_lab_commit": manifest.quant_lab_commit,
            "worker_commit": manifest.worker_commit,
            "candidate_event_digest": manifest.candidate_event_digest,
            "market_bar_digest": manifest.market_bar_digest,
            "run_summary_digest": manifest.run_summary_digest,
            "input_fingerprint_digest": manifest.input_fingerprint_digest,
            "previous_generation_id": manifest.previous_generation_id,
            "previous_generation_digest": manifest.previous_generation_digest,
            "as_of_date": manifest.as_of_date.isoformat(),
            "mode": manifest.mode,
            "lookback_days": manifest.lookback_days,
            "horizon_hours": list(manifest.horizon_hours),
            "candidate_label_schema_version": manifest.candidate_label_schema_version,
            "strategy_evidence_version": manifest.strategy_evidence_version,
            "datasets": list(V5_CANDIDATE_EVIDENCE_DATASETS),
            "row_counts": managed_row_counts,
            "dataset_hashes": dataset_hashes,
            "primary_keys": {
                name: list(keys)
                for name, keys in V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS.items()
            },
            "published_at": datetime.now(UTC).isoformat(),
            "diagnostic_only": True,
            "research_only": True,
            "live_order_effect": "none_read_only_research",
            "automatic_promotion": False,
            "max_live_notional_usdt": 0,
        }
        for dataset_name, staged in staged_by_dataset.items():
            _write_generation_sidecars(
                staged,
                dataset_name=dataset_name,
                generation_payload=generation_payload,
                managed_row_count=managed_row_counts[dataset_name],
                managed_dataset_hash=dataset_hashes[dataset_name],
            )

        commit_atomic_research_generation(
            root,
            transaction_name=V5_CANDIDATE_EVIDENCE_TRANSACTION_NAME,
            generation_payload=generation_payload,
            pointer_path=V5_CANDIDATE_EVIDENCE_GENERATION_POINTER,
            items=items,
            post_commit_validate=lambda: verify_v5_candidate_evidence_generation_fast(
                root,
                manifest.generation_id,
                expected_input_fingerprint=manifest.input_fingerprint_digest,
            ),
        )
        return {
            "published": True,
            "generation_id": manifest.generation_id,
            "generation_digest": generation_digest,
            "row_counts": managed_row_counts,
            "dataset_hashes": dataset_hashes,
        }
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def verify_v5_candidate_evidence_generation_fast(
    lake_root: str | Path,
    generation_id: str,
    *,
    expected_input_fingerprint: str | None = None,
) -> dict[str, int]:
    root = Path(lake_root).resolve(strict=True)
    pointer_path = root / V5_CANDIDATE_EVIDENCE_GENERATION_POINTER
    try:
        pointer = json.loads(pointer_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("v5_candidate_evidence_generation_pointer_invalid") from exc
    if pointer.get("schema_version") != V5_CANDIDATE_EVIDENCE_GENERATION_SCHEMA:
        raise RuntimeError("v5_candidate_evidence_generation_pointer_schema_mismatch")
    if pointer.get("generation_id") != generation_id:
        raise RuntimeError("v5_candidate_evidence_generation_pointer_mismatch")
    digest = str(pointer.get("generation_digest") or "")
    if len(digest) != 64:
        raise RuntimeError("v5_candidate_evidence_generation_digest_invalid")
    if expected_input_fingerprint is not None and pointer.get(
        "input_fingerprint_digest"
    ) != expected_input_fingerprint:
        raise RuntimeError("v5_candidate_evidence_generation_fingerprint_mismatch")
    if (
        pointer.get("diagnostic_only") is not True
        or pointer.get("research_only") is not True
        or pointer.get("live_order_effect") != "none_read_only_research"
        or pointer.get("automatic_promotion") is not False
        or pointer.get("max_live_notional_usdt") != 0
    ):
        raise RuntimeError("v5_candidate_evidence_generation_safety_mismatch")
    row_counts = {
        str(name): int(value) for name, value in dict(pointer.get("row_counts") or {}).items()
    }
    dataset_hashes = {
        str(name): str(value)
        for name, value in dict(pointer.get("dataset_hashes") or {}).items()
    }
    primary_keys = {
        str(name): tuple(value)
        for name, value in dict(pointer.get("primary_keys") or {}).items()
    }
    expected_names = set(V5_CANDIDATE_EVIDENCE_DATASETS)
    if (
        set(pointer.get("datasets") or []) != expected_names
        or set(row_counts) != expected_names
        or set(dataset_hashes) != expected_names
        or primary_keys != V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS
    ):
        raise RuntimeError("v5_candidate_evidence_generation_dataset_set_mismatch")
    for dataset_name, target in V5_CANDIDATE_EVIDENCE_DATASETS.items():
        dataset_root = root / target
        if not dataset_root.is_dir():
            raise RuntimeError(
                f"v5_candidate_evidence_generation_dataset_missing:{dataset_name}"
            )
        sidecar_path = dataset_root / V5_CANDIDATE_EVIDENCE_SIDECAR
        try:
            sidecar = json.loads(sidecar_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"v5_candidate_evidence_generation_sidecar_invalid:{dataset_name}"
            ) from exc
        if (
            sidecar.get("generation_id") != generation_id
            or sidecar.get("generation_digest") != digest
            or sidecar.get("dataset_name") != dataset_name
            or sidecar.get("dataset_hash") != dataset_hashes[dataset_name]
            or sidecar.get("row_count") != row_counts[dataset_name]
            or tuple(sidecar.get("primary_keys") or ())
            != V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS[dataset_name]
        ):
            raise RuntimeError(
                f"v5_candidate_evidence_generation_sidecar_mismatch:{dataset_name}"
            )
        _validate_staged_dataset(
            dataset_root,
            dataset_name=dataset_name,
            key_columns=V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS[dataset_name],
            managed_scope=V5_CANDIDATE_EVIDENCE_MANAGED_SCOPES[dataset_name],
        )
        if _managed_row_count(dataset_root, dataset_name) != row_counts[dataset_name]:
            raise RuntimeError(
                f"v5_candidate_evidence_generation_row_count_mismatch:{dataset_name}"
            )
        if _managed_dataset_digest(dataset_root, dataset_name) != dataset_hashes[dataset_name]:
            raise RuntimeError(
                f"v5_candidate_evidence_generation_dataset_hash_mismatch:{dataset_name}"
            )
    _validate_cloud_decision_safety(root / STRATEGY_EVIDENCE_DATASET)
    return row_counts


def recover_v5_candidate_evidence_publication(lake_root: str | Path) -> bool:
    return recover_atomic_research_generation(
        lake_root,
        transaction_name=V5_CANDIDATE_EVIDENCE_TRANSACTION_NAME,
        pointer_path=V5_CANDIDATE_EVIDENCE_GENERATION_POINTER,
    )


def _validate_previous_generation_binding(
    root: Path,
    validated: ValidatedV5CandidateEvidenceResult,
) -> None:
    pointer_path = root / V5_CANDIDATE_EVIDENCE_GENERATION_POINTER
    previous_id = validated.manifest.previous_generation_id
    previous_digest = validated.manifest.previous_generation_digest
    if previous_id is None:
        if pointer_path.is_file():
            raise ValueError("v5_candidate_evidence_result_superseded_by_generation")
        return
    if not pointer_path.is_file():
        raise ValueError("v5_candidate_evidence_previous_generation_missing")
    pointer = json.loads(pointer_path.read_text("utf-8"))
    if (
        pointer.get("generation_id") != previous_id
        or pointer.get("generation_digest") != previous_digest
    ):
        raise ValueError("v5_candidate_evidence_result_superseded_by_generation")


def _recent_managed_samples(
    dataset_root: Path,
    *,
    as_of_date: date,
    lookback_days: int,
) -> pl.DataFrame:
    files = _parquet_files(dataset_root)
    if not files:
        return pl.DataFrame(schema=SAMPLE_SCHEMA)
    start = datetime.combine(as_of_date - timedelta(days=lookback_days), time.min, tzinfo=UTC)
    end = datetime.combine(as_of_date + timedelta(days=1), time.min, tzinfo=UTC)
    lazy = pl.scan_parquet([str(path) for path in files], extra_columns="ignore")
    filtered = lazy.filter(
        (pl.col("strategy") == "v5")
        & (pl.col("source") == STRATEGY_EVIDENCE_SOURCE)
        & pl.col("ts_utc").cast(pl.Datetime(time_zone="UTC"), strict=False).is_between(
            start,
            end,
            closed="left",
        )
    )
    frame = filtered.collect(engine="streaming")
    if frame.is_empty():
        return pl.DataFrame(schema=SAMPLE_SCHEMA)
    return normalize_strategy_evidence_samples(frame).unique(
        subset=STRATEGY_EVIDENCE_SAMPLE_KEY_COLUMNS,
        keep="last",
        maintain_order=True,
    )


def _read_snapshot_dataset(
    files_root: Path,
    snapshot: Any,
    dataset_name: str,
) -> pl.DataFrame:
    paths = [
        files_root / item.relative_path
        for item in snapshot.files
        if item.dataset_name == dataset_name
    ]
    return _collect_paths(tuple(paths))


def _collect_paths(paths: tuple[Path, ...]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    return pl.scan_parquet([str(path) for path in paths], extra_columns="ignore").collect(
        engine="streaming"
    )


def _stage_streaming_upsert(
    existing_root: Path,
    incoming_paths: list[Path],
    staged_root: Path,
    *,
    key_columns: tuple[str, ...],
    replace_scope: dict[str, object] | None = None,
) -> int:
    if not incoming_paths:
        raise ValueError("v5_candidate_evidence_publish_incoming_missing")
    staged_root.mkdir(parents=True, exist_ok=False)
    output_path = staged_root / "data.parquet"
    temp_directory = staged_root / ".duckdb_tmp"
    temp_directory.mkdir(parents=True, exist_ok=False)
    _require_writable_spill_directory(temp_directory)
    existing_paths = _parquet_files(existing_root)
    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        _configure_duckdb(connection, temp_directory)
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
    managed_scope: dict[str, object],
) -> None:
    files = _parquet_files(dataset_root)
    if not files:
        raise ValueError(f"v5_candidate_evidence_publish_dataset_missing:{dataset_name}")
    temporary = dataset_root.parent / f".duckdb-validate-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        _require_writable_spill_directory(temporary)
        connection = duckdb.connect(database=":memory:", read_only=False)
        _configure_duckdb(connection, temporary)
        source = _read_parquet_sql(files)
        columns = {str(row[0]) for row in connection.execute(f"DESCRIBE {source}").fetchall()}
        missing = sorted(set(key_columns) - columns)
        if missing:
            raise ValueError(
                f"v5_candidate_evidence_publish_primary_key_missing:{dataset_name}:"
                + ",".join(missing)
            )
        managed_source = (
            f"SELECT * FROM ({source}) WHERE {_scope_sql(managed_scope)}"
        )
        key_sql = ",".join(_identifier(column) for column in key_columns)
        duplicate = connection.execute(
            f"SELECT 1 FROM ({managed_source}) "
            f"GROUP BY {key_sql} HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate is not None:
            raise ValueError(f"v5_candidate_evidence_publish_duplicate_key:{dataset_name}")
        null_predicate = " OR ".join(
            f"{_identifier(column)} IS NULL" for column in key_columns
        )
        if connection.execute(
            f"SELECT 1 FROM ({managed_source}) WHERE {null_predicate} LIMIT 1"
        ).fetchone():
            raise ValueError(f"v5_candidate_evidence_publish_null_key:{dataset_name}")
    finally:
        if connection is not None:
            connection.close()
        shutil.rmtree(temporary, ignore_errors=True)


def _managed_row_count(dataset_root: Path, dataset_name: str) -> int:
    files = _parquet_files(dataset_root)
    if not files:
        return 0
    source = _read_parquet_sql(files)
    predicate = _scope_sql(V5_CANDIDATE_EVIDENCE_MANAGED_SCOPES[dataset_name])
    connection = duckdb.connect(database=":memory:")
    try:
        row = connection.execute(
            f"SELECT COUNT(*) FROM ({source}) WHERE {predicate}"
        ).fetchone()
        return int(row[0])
    finally:
        connection.close()


def _managed_dataset_digest(dataset_root: Path, dataset_name: str) -> str:
    files = _parquet_files(dataset_root)
    if not files:
        return model_content_sha256({"dataset_name": dataset_name, "schema": [], "rows": 0})
    source = _read_parquet_sql(files)
    predicate = _scope_sql(V5_CANDIDATE_EVIDENCE_MANAGED_SCOPES[dataset_name])
    connection = duckdb.connect(database=":memory:")
    try:
        description = connection.execute(f"DESCRIBE {source}").fetchall()
        columns = [str(row[0]) for row in description]
        schema = [(str(row[0]), str(row[1])) for row in description]
        hash_sql = "hash(" + ",".join(_identifier(column) for column in columns) + ")"
        row = connection.execute(
            "SELECT COUNT(*), "
            f"COALESCE(SUM(CAST({hash_sql} AS HUGEINT)), 0), "
            f"COALESCE(BIT_XOR({hash_sql}), 0), "
            f"COALESCE(MIN({hash_sql}), 0), COALESCE(MAX({hash_sql}), 0) "
            f"FROM ({source}) WHERE {predicate}"
        ).fetchone()
    finally:
        connection.close()
    return model_content_sha256(
        {
            "schema_version": "v5_candidate_evidence_managed_dataset_digest.v1",
            "dataset_name": dataset_name,
            "scope": V5_CANDIDATE_EVIDENCE_MANAGED_SCOPES[dataset_name],
            "schema": schema,
            "row_count": int(row[0]),
            "hash_sum": str(row[1]),
            "hash_xor": str(row[2]),
            "hash_min": str(row[3]),
            "hash_max": str(row[4]),
        }
    )


def _write_generation_sidecars(
    staged: Path,
    *,
    dataset_name: str,
    generation_payload: dict[str, Any],
    managed_row_count: int,
    managed_dataset_hash: str,
) -> None:
    atomic_write_json(
        staged / V5_CANDIDATE_EVIDENCE_SIDECAR,
        {
            "schema_version": V5_CANDIDATE_EVIDENCE_GENERATION_SCHEMA,
            "generation_id": generation_payload["generation_id"],
            "generation_digest": generation_payload["generation_digest"],
            "task_id": generation_payload["task_id"],
            "snapshot_id": generation_payload["snapshot_id"],
            "dataset_name": dataset_name,
            "dataset_hash": managed_dataset_hash,
            "row_count": managed_row_count,
            "primary_keys": generation_payload["primary_keys"][dataset_name],
            "managed_scope": V5_CANDIDATE_EVIDENCE_MANAGED_SCOPES[dataset_name],
            "research_only": True,
            "live_order_effect": "none_read_only_research",
            "automatic_promotion": False,
            "max_live_notional_usdt": 0,
        },
    )
    atomic_write_json(
        staged / "_snapshot_meta.json",
        {
            "dataset": dataset_name,
            "generated_at": generation_payload["published_at"],
            "row_count": count_parquet_rows(staged),
            "managed_row_count": managed_row_count,
            "source_sha": generation_payload["generation_digest"],
            "file_count": len(_parquet_files(staged)),
            "schema_version": V5_CANDIDATE_EVIDENCE_GENERATION_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def _validate_cloud_decision_safety(dataset_root: Path) -> None:
    files = _parquet_files(dataset_root)
    if not files:
        return
    lazy = pl.scan_parquet([str(path) for path in files], extra_columns="ignore")
    required = {"strategy", "source", "decision"}
    if not required.issubset(lazy.collect_schema().names()):
        raise RuntimeError("v5_candidate_evidence_strategy_evidence_schema_incomplete")
    invalid = (
        lazy.filter(
            (pl.col("strategy") == "v5")
            & (pl.col("source") == STRATEGY_EVIDENCE_SOURCE)
            & ~pl.col("decision").is_in(sorted(V5_CANDIDATE_EVIDENCE_ALLOWED_CLOUD_DECISIONS))
        )
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if invalid:
        raise RuntimeError("v5_candidate_evidence_cloud_decision_safety_mismatch")


def _copy_existing_sidecars(existing_root: Path, staged_root: Path) -> None:
    if not existing_root.is_dir():
        return
    for source in existing_root.rglob("*"):
        if source.is_symlink():
            raise ValueError("v5_candidate_evidence_existing_symlink_forbidden")
        if not source.is_file() or source.suffix == ".parquet":
            continue
        relative = source.relative_to(existing_root)
        if relative.name in {"_snapshot_meta.json", V5_CANDIDATE_EVIDENCE_SIDECAR}:
            continue
        destination = staged_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _require_writable_spill_directory(path: Path) -> None:
    probe = path / f".write-probe-{uuid.uuid4().hex}"
    try:
        with probe.open("xb") as handle:
            handle.write(b"v5-candidate-evidence-duckdb-spill-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RuntimeError("v5_candidate_evidence_duckdb_spill_not_writable") from exc
    finally:
        probe.unlink(missing_ok=True)


def _configure_duckdb(connection: duckdb.DuckDBPyConnection, temporary: Path) -> None:
    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET memory_limit = '384MB'")
    connection.execute(f"SET temp_directory = {_sql_literal(temporary)}")


def _parquet_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def _read_parquet_sql(paths: list[Path]) -> str:
    values = ",".join(_sql_literal(path) for path in paths)
    return (
        f"SELECT * FROM read_parquet([{values}], union_by_name=true, "
        "hive_partitioning=false)"
    )


def _scope_sql(scope: dict[str, object]) -> str:
    return " AND ".join(
        f"{_identifier(column)} = {_sql_literal(str(value))}"
        for column, value in sorted(scope.items())
    )


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
