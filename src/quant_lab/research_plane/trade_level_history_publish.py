from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from quant_lab.data.lake import count_parquet_rows
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.research_plane.atomic_publish import (
    AtomicPublishItem,
    commit_atomic_research_generation,
    recover_atomic_research_generation,
)
from quant_lab.research_plane.signatures import model_content_sha256
from quant_lab.research_plane.trade_level_history_contracts import (
    TRADE_LEVEL_HISTORY_GENERATION_SCHEMA,
    TRADE_LEVEL_HISTORY_PRIMARY_KEYS,
)
from quant_lab.research_plane.trade_level_history_result import (
    ValidatedTradeLevelHistoryResult,
)
from quant_lab.research_plane.v5_candidate_evidence_publish import (
    V5_CANDIDATE_EVIDENCE_GENERATION_POINTER,
    verify_v5_candidate_evidence_generation_fast,
)
from quant_lab.trade_level.judgment import TRADE_OPPORTUNITY_EVENT_SCHEMA
from quant_lab.trade_level.labels import TRADE_OPPORTUNITY_LABEL_SCHEMA
from quant_lab.trade_level.similarity import TRADE_LEVEL_SIMILARITY_SCHEMA

TRADE_LEVEL_HISTORY_GENERATION_POINTER = (
    Path("gold") / "trade_level_history_generation.json"
)
TRADE_LEVEL_HISTORY_TRANSACTION_NAME = "trade_level_history"
TRADE_LEVEL_HISTORY_SIDECAR = "_trade_level_history_generation.json"
TRADE_LEVEL_HISTORY_DATASETS = {
    "trade_opportunity_event": Path("gold") / "trade_opportunity_event",
    "trade_opportunity_label": Path("gold") / "trade_opportunity_label",
    "trade_level_similarity_outcome": (
        Path("gold") / "trade_level_similarity_outcome"
    ),
}
TRADE_LEVEL_HISTORY_DATASET_SCHEMAS = {
    "trade_opportunity_event": TRADE_OPPORTUNITY_EVENT_SCHEMA,
    "trade_opportunity_label": TRADE_OPPORTUNITY_LABEL_SCHEMA,
    "trade_level_similarity_outcome": TRADE_LEVEL_SIMILARITY_SCHEMA,
}
TRADE_LEVEL_HISTORY_ALL_PRIMARY_KEYS = {
    "trade_opportunity_event": ("event_id",),
    **TRADE_LEVEL_HISTORY_PRIMARY_KEYS,
}


def publish_trade_level_history_generation(
    lake_root: str | Path,
    validated: ValidatedTradeLevelHistoryResult,
    *,
    snapshot_root: str | Path,
) -> dict[str, Any]:
    """Atomically replace the three full-history Gold datasets."""

    root = Path(lake_root).resolve(strict=True)
    recover_trade_level_history_publication(root)
    manifest = validated.manifest
    current = _read_optional_pointer(
        root / TRADE_LEVEL_HISTORY_GENERATION_POINTER
    )
    if current.get("generation_id") == manifest.generation_id:
        row_counts = verify_trade_level_history_generation_fast(
            root,
            manifest.generation_id,
            expected_input_fingerprint=manifest.input_fingerprint_digest,
            expected_candidate_generation_id=(
                manifest.candidate_evidence_generation_id
            ),
            expected_candidate_generation_digest=(
                manifest.candidate_evidence_generation_digest
            ),
        )
        return {
            "published": False,
            "idempotent": True,
            "generation_id": manifest.generation_id,
            "generation_digest": current["generation_digest"],
            "row_counts": row_counts,
            "dataset_hashes": current["dataset_hashes"],
        }

    _validate_previous_generation_binding(root, validated)
    _validate_candidate_generation_binding(root, validated)
    snapshot_files = Path(snapshot_root).resolve(strict=True) / "files"
    event_paths = [
        snapshot_files / item.relative_path
        for item in validated.snapshot.files
        if item.dataset_name == "cloud/trade_opportunity_event"
    ]
    if len(event_paths) != 1:
        raise ValueError(
            "trade_level_history_publish_event_snapshot_file_set_mismatch"
        )
    source_paths = {
        "trade_opportunity_event": event_paths,
        "trade_opportunity_label": list(validated.label_paths),
        "trade_level_similarity_outcome": list(
            validated.similarity_paths
        ),
    }
    if any(not paths for paths in source_paths.values()):
        raise ValueError("trade_level_history_publish_input_files_missing")

    transaction_id = uuid.uuid4().hex
    staging_root = (
        root
        / "gold"
        / f".__trade_level_history_stage_{transaction_id[:8]}"
    )
    staging_root.mkdir(parents=True, exist_ok=False)
    staged_by_dataset: dict[str, Path] = {}
    items: list[AtomicPublishItem] = []
    try:
        for index, (dataset_name, target) in enumerate(
            TRADE_LEVEL_HISTORY_DATASETS.items()
        ):
            staged = staging_root / f"dataset-{index:02d}-{dataset_name}"
            _stage_full_dataset(
                source_paths[dataset_name],
                staged,
                schema=TRADE_LEVEL_HISTORY_DATASET_SCHEMAS[dataset_name],
            )
            _validate_dataset(
                staged,
                dataset_name=dataset_name,
                expected_schema=TRADE_LEVEL_HISTORY_DATASET_SCHEMAS[
                    dataset_name
                ],
                primary_keys=TRADE_LEVEL_HISTORY_ALL_PRIMARY_KEYS[
                    dataset_name
                ],
            )
            staged_by_dataset[dataset_name] = staged
            items.append(
                AtomicPublishItem(
                    target=target,
                    staged=staged.relative_to(root),
                )
            )

        row_counts = {
            name: count_parquet_rows(path)
            for name, path in staged_by_dataset.items()
        }
        if not (
            row_counts["trade_opportunity_event"]
            == row_counts["trade_opportunity_label"]
            == row_counts["trade_level_similarity_outcome"]
        ):
            raise ValueError(
                "trade_level_history_publish_cardinality_mismatch"
            )
        managed_columns = {
            name: list(TRADE_LEVEL_HISTORY_DATASET_SCHEMAS[name])
            for name in TRADE_LEVEL_HISTORY_DATASETS
        }
        dataset_hashes = {
            name: _dataset_digest(
                staged_by_dataset[name],
                dataset_name=name,
                managed_columns=managed_columns[name],
            )
            for name in TRADE_LEVEL_HISTORY_DATASETS
        }
        result_outputs = [
            [item.relative_path, item.sha256]
            for item in sorted(
                manifest.outputs,
                key=lambda item: item.relative_path,
            )
        ]
        primary_keys = {
            name: list(keys)
            for name, keys in TRADE_LEVEL_HISTORY_ALL_PRIMARY_KEYS.items()
        }
        schema_versions = {
            "trade_opportunity_event": manifest.trade_event_schema_version,
            "trade_opportunity_label": manifest.trade_label_schema_version,
            "trade_level_similarity_outcome": (
                manifest.similarity_schema_version
            ),
        }
        digest_identity = _generation_digest_identity(
            task_id=manifest.task_id,
            snapshot_id=manifest.snapshot_id,
            previous_generation_id=manifest.previous_generation_id,
            previous_generation_digest=manifest.previous_generation_digest,
            quant_lab_commit=manifest.quant_lab_commit,
            worker_commit=manifest.worker_commit,
            candidate_evidence_generation_id=(
                manifest.candidate_evidence_generation_id
            ),
            candidate_evidence_generation_digest=(
                manifest.candidate_evidence_generation_digest
            ),
            candidate_evidence_input_fingerprint=(
                manifest.candidate_evidence_input_fingerprint
            ),
            derived_event_digest=manifest.derived_event_digest,
            candidate_label_dataset_hash=(
                manifest.candidate_label_dataset_hash
            ),
            input_fingerprint_digest=manifest.input_fingerprint_digest,
            history_mode=manifest.history_mode,
            schema_versions=schema_versions,
            similarity_availability_policy=(
                manifest.similarity_availability_policy
            ),
            result_outputs=result_outputs,
            row_counts=row_counts,
            dataset_hashes=dataset_hashes,
            managed_columns=managed_columns,
            primary_keys=primary_keys,
        )
        generation_digest = model_content_sha256(digest_identity)
        generation_payload = {
            "schema_version": TRADE_LEVEL_HISTORY_GENERATION_SCHEMA,
            "generation_id": manifest.generation_id,
            "generation_digest": generation_digest,
            "task_id": manifest.task_id,
            "snapshot_id": manifest.snapshot_id,
            "quant_lab_commit": manifest.quant_lab_commit,
            "worker_commit": manifest.worker_commit,
            "candidate_evidence_generation_id": (
                manifest.candidate_evidence_generation_id
            ),
            "candidate_evidence_generation_digest": (
                manifest.candidate_evidence_generation_digest
            ),
            "candidate_evidence_input_fingerprint": (
                manifest.candidate_evidence_input_fingerprint
            ),
            "derived_event_digest": manifest.derived_event_digest,
            "candidate_label_dataset_hash": (
                manifest.candidate_label_dataset_hash
            ),
            "input_fingerprint_digest": (
                manifest.input_fingerprint_digest
            ),
            "previous_generation_id": manifest.previous_generation_id,
            "previous_generation_digest": (
                manifest.previous_generation_digest
            ),
            "as_of_date": manifest.as_of_date.isoformat(),
            "history_mode": manifest.history_mode,
            "schema_versions": schema_versions,
            "similarity_availability_policy": (
                manifest.similarity_availability_policy
            ),
            "datasets": list(TRADE_LEVEL_HISTORY_DATASETS),
            "row_counts": row_counts,
            "dataset_hashes": dataset_hashes,
            "managed_columns": managed_columns,
            "primary_keys": primary_keys,
            "result_outputs": result_outputs,
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
            )
        commit_atomic_research_generation(
            root,
            transaction_name=TRADE_LEVEL_HISTORY_TRANSACTION_NAME,
            generation_payload=generation_payload,
            pointer_path=TRADE_LEVEL_HISTORY_GENERATION_POINTER,
            items=items,
            post_commit_validate=lambda: (
                verify_trade_level_history_generation_fast(
                    root,
                    manifest.generation_id,
                    expected_input_fingerprint=(
                        manifest.input_fingerprint_digest
                    ),
                    expected_candidate_generation_id=(
                        manifest.candidate_evidence_generation_id
                    ),
                    expected_candidate_generation_digest=(
                        manifest.candidate_evidence_generation_digest
                    ),
                )
            ),
        )
        return {
            "published": True,
            "idempotent": False,
            "generation_id": manifest.generation_id,
            "generation_digest": generation_digest,
            "row_counts": row_counts,
            "dataset_hashes": dataset_hashes,
        }
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def verify_trade_level_history_generation_fast(
    lake_root: str | Path,
    generation_id: str,
    *,
    expected_input_fingerprint: str | None = None,
    expected_candidate_generation_id: str | None = None,
    expected_candidate_generation_digest: str | None = None,
) -> dict[str, int]:
    """Read-only integrity verification of the pointer and all three Golds."""

    root = Path(lake_root).resolve(strict=True)
    pointer = _read_pointer(root / TRADE_LEVEL_HISTORY_GENERATION_POINTER)
    if pointer.get("schema_version") != (
        TRADE_LEVEL_HISTORY_GENERATION_SCHEMA
    ):
        raise RuntimeError(
            "trade_level_history_generation_pointer_schema_mismatch"
        )
    if pointer.get("generation_id") != generation_id:
        raise RuntimeError(
            "trade_level_history_generation_pointer_mismatch"
        )
    if (
        expected_input_fingerprint is not None
        and pointer.get("input_fingerprint_digest")
        != expected_input_fingerprint
    ):
        raise RuntimeError(
            "trade_level_history_generation_fingerprint_mismatch"
        )
    if (
        expected_candidate_generation_id is not None
        and pointer.get("candidate_evidence_generation_id")
        != expected_candidate_generation_id
    ):
        raise RuntimeError(
            "trade_level_history_generation_candidate_id_mismatch"
        )
    if (
        expected_candidate_generation_digest is not None
        and pointer.get("candidate_evidence_generation_digest")
        != expected_candidate_generation_digest
    ):
        raise RuntimeError(
            "trade_level_history_generation_candidate_digest_mismatch"
        )
    if (
        pointer.get("diagnostic_only") is not True
        or pointer.get("research_only") is not True
        or pointer.get("live_order_effect")
        != "none_read_only_research"
        or pointer.get("automatic_promotion") is not False
        or pointer.get("max_live_notional_usdt") != 0
    ):
        raise RuntimeError(
            "trade_level_history_generation_safety_mismatch"
        )

    expected_names = set(TRADE_LEVEL_HISTORY_DATASETS)
    row_counts = {
        str(name): int(value)
        for name, value in dict(pointer.get("row_counts") or {}).items()
    }
    dataset_hashes = {
        str(name): str(value)
        for name, value in dict(
            pointer.get("dataset_hashes") or {}
        ).items()
    }
    managed_columns = {
        str(name): tuple(str(column) for column in value)
        for name, value in dict(
            pointer.get("managed_columns") or {}
        ).items()
    }
    primary_keys = {
        str(name): tuple(str(column) for column in value)
        for name, value in dict(pointer.get("primary_keys") or {}).items()
    }
    expected_columns = {
        name: tuple(TRADE_LEVEL_HISTORY_DATASET_SCHEMAS[name])
        for name in TRADE_LEVEL_HISTORY_DATASETS
    }
    if (
        set(pointer.get("datasets") or []) != expected_names
        or set(row_counts) != expected_names
        or set(dataset_hashes) != expected_names
        or managed_columns != expected_columns
        or primary_keys != TRADE_LEVEL_HISTORY_ALL_PRIMARY_KEYS
    ):
        raise RuntimeError(
            "trade_level_history_generation_dataset_set_mismatch"
        )
    schema_versions = dict(pointer.get("schema_versions") or {})
    if schema_versions != {
        "trade_opportunity_event": "trade_opportunity_event.v0.3",
        "trade_opportunity_label": "trade_opportunity_label.v0.2",
        "trade_level_similarity_outcome": (
            "trade_level_similarity_outcome.v0.2"
        ),
    }:
        raise RuntimeError(
            "trade_level_history_generation_schema_version_mismatch"
        )
    if pointer.get("history_mode") != "PARITY_FULL":
        raise RuntimeError(
            "trade_level_history_generation_history_mode_mismatch"
        )
    if pointer.get("similarity_availability_policy") != (
        "largest_available_horizon"
    ):
        raise RuntimeError(
            "trade_level_history_generation_availability_policy_mismatch"
        )
    digest_identity = _generation_digest_identity(
        task_id=str(pointer.get("task_id") or ""),
        snapshot_id=str(pointer.get("snapshot_id") or ""),
        previous_generation_id=pointer.get("previous_generation_id"),
        previous_generation_digest=pointer.get(
            "previous_generation_digest"
        ),
        quant_lab_commit=str(pointer.get("quant_lab_commit") or ""),
        worker_commit=str(pointer.get("worker_commit") or ""),
        candidate_evidence_generation_id=str(
            pointer.get("candidate_evidence_generation_id") or ""
        ),
        candidate_evidence_generation_digest=str(
            pointer.get("candidate_evidence_generation_digest") or ""
        ),
        candidate_evidence_input_fingerprint=str(
            pointer.get("candidate_evidence_input_fingerprint") or ""
        ),
        derived_event_digest=str(
            pointer.get("derived_event_digest") or ""
        ),
        candidate_label_dataset_hash=str(
            pointer.get("candidate_label_dataset_hash") or ""
        ),
        input_fingerprint_digest=str(
            pointer.get("input_fingerprint_digest") or ""
        ),
        history_mode=str(pointer.get("history_mode") or ""),
        schema_versions=schema_versions,
        similarity_availability_policy=str(
            pointer.get("similarity_availability_policy") or ""
        ),
        result_outputs=list(pointer.get("result_outputs") or []),
        row_counts=row_counts,
        dataset_hashes=dataset_hashes,
        managed_columns={
            name: list(columns)
            for name, columns in managed_columns.items()
        },
        primary_keys={
            name: list(keys) for name, keys in primary_keys.items()
        },
    )
    digest = model_content_sha256(digest_identity)
    if pointer.get("generation_digest") != digest:
        raise RuntimeError(
            "trade_level_history_generation_digest_mismatch"
        )

    candidate_pointer = _read_pointer(
        root / V5_CANDIDATE_EVIDENCE_GENERATION_POINTER
    )
    if (
        candidate_pointer.get("generation_id")
        != pointer.get("candidate_evidence_generation_id")
        or candidate_pointer.get("generation_digest")
        != pointer.get("candidate_evidence_generation_digest")
        or candidate_pointer.get("input_fingerprint_digest")
        != pointer.get("candidate_evidence_input_fingerprint")
        or dict(candidate_pointer.get("dataset_hashes") or {}).get(
            "v5_candidate_label"
        )
        != pointer.get("candidate_label_dataset_hash")
    ):
        raise RuntimeError(
            "trade_level_history_generation_candidate_binding_mismatch"
        )
    verify_v5_candidate_evidence_generation_fast(
        root,
        str(pointer["candidate_evidence_generation_id"]),
        expected_input_fingerprint=str(
            pointer["candidate_evidence_input_fingerprint"]
        ),
    )

    for dataset_name, target in TRADE_LEVEL_HISTORY_DATASETS.items():
        dataset_root = root / target
        if not dataset_root.is_dir():
            raise RuntimeError(
                f"trade_level_history_generation_dataset_missing:"
                f"{dataset_name}"
            )
        sidecar = _read_pointer(
            dataset_root / TRADE_LEVEL_HISTORY_SIDECAR
        )
        if (
            sidecar.get("generation_id") != generation_id
            or sidecar.get("generation_digest") != digest
            or sidecar.get("dataset_name") != dataset_name
            or sidecar.get("dataset_hash")
            != dataset_hashes[dataset_name]
            or sidecar.get("row_count") != row_counts[dataset_name]
            or tuple(sidecar.get("primary_keys") or ())
            != TRADE_LEVEL_HISTORY_ALL_PRIMARY_KEYS[dataset_name]
            or tuple(sidecar.get("managed_columns") or ())
            != managed_columns[dataset_name]
            or sidecar.get("research_only") is not True
            or sidecar.get("live_order_effect")
            != "none_read_only_research"
            or sidecar.get("automatic_promotion") is not False
            or sidecar.get("max_live_notional_usdt") != 0
        ):
            raise RuntimeError(
                f"trade_level_history_generation_sidecar_mismatch:"
                f"{dataset_name}"
            )
        try:
            _validate_dataset(
                dataset_root,
                dataset_name=dataset_name,
                expected_schema=TRADE_LEVEL_HISTORY_DATASET_SCHEMAS[
                    dataset_name
                ],
                primary_keys=TRADE_LEVEL_HISTORY_ALL_PRIMARY_KEYS[
                    dataset_name
                ],
            )
            if (
                count_parquet_rows(dataset_root)
                != row_counts[dataset_name]
            ):
                raise RuntimeError(
                    f"trade_level_history_generation_row_count_mismatch:"
                    f"{dataset_name}"
                )
            if _dataset_digest(
                dataset_root,
                dataset_name=dataset_name,
                managed_columns=list(managed_columns[dataset_name]),
            ) != dataset_hashes[dataset_name]:
                raise RuntimeError(
                    f"trade_level_history_generation_dataset_hash_mismatch:"
                    f"{dataset_name}"
                )
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith(
                "trade_level_history_generation_"
            ):
                raise
            raise RuntimeError(
                f"trade_level_history_generation_dataset_integrity_failed:"
                f"{dataset_name}"
            ) from exc
    if not (
        row_counts["trade_opportunity_event"]
        == row_counts["trade_opportunity_label"]
        == row_counts["trade_level_similarity_outcome"]
    ):
        raise RuntimeError(
            "trade_level_history_generation_cardinality_mismatch"
        )
    return row_counts


def recover_trade_level_history_publication(
    lake_root: str | Path,
) -> bool:
    return recover_atomic_research_generation(
        lake_root,
        transaction_name=TRADE_LEVEL_HISTORY_TRANSACTION_NAME,
        pointer_path=TRADE_LEVEL_HISTORY_GENERATION_POINTER,
    )


def _validate_previous_generation_binding(
    root: Path,
    validated: ValidatedTradeLevelHistoryResult,
) -> None:
    pointer_path = root / TRADE_LEVEL_HISTORY_GENERATION_POINTER
    previous_id = validated.manifest.previous_generation_id
    previous_digest = validated.manifest.previous_generation_digest
    if previous_id is None:
        if pointer_path.is_file():
            raise ValueError(
                "trade_level_history_result_superseded_by_generation"
            )
        return
    pointer = _read_pointer(pointer_path)
    if (
        pointer.get("generation_id") != previous_id
        or pointer.get("generation_digest") != previous_digest
    ):
        raise ValueError(
            "trade_level_history_result_superseded_by_generation"
        )


def _validate_candidate_generation_binding(
    root: Path,
    validated: ValidatedTradeLevelHistoryResult,
) -> None:
    manifest = validated.manifest
    pointer = _read_pointer(
        root / V5_CANDIDATE_EVIDENCE_GENERATION_POINTER
    )
    if (
        pointer.get("generation_id")
        != manifest.candidate_evidence_generation_id
        or pointer.get("generation_digest")
        != manifest.candidate_evidence_generation_digest
        or pointer.get("input_fingerprint_digest")
        != manifest.candidate_evidence_input_fingerprint
        or dict(pointer.get("dataset_hashes") or {}).get(
            "v5_candidate_label"
        )
        != manifest.candidate_label_dataset_hash
    ):
        raise ValueError(
            "trade_level_history_result_superseded_by_candidate_generation"
        )
    verify_v5_candidate_evidence_generation_fast(
        root,
        manifest.candidate_evidence_generation_id,
        expected_input_fingerprint=(
            manifest.candidate_evidence_input_fingerprint
        ),
    )


def _stage_full_dataset(
    source_paths: list[Path],
    staged_root: Path,
    *,
    schema: dict[str, Any],
) -> None:
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    staged_root.mkdir(parents=True, exist_ok=False)
    output = staged_root / "data.parquet"
    temporary = staged_root / ".duckdb_tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        _configure_duckdb(connection, temporary)
        source = _read_parquet_sql(source_paths)
        columns = ",".join(_identifier(name) for name in schema)
        connection.execute(
            f"COPY (SELECT {columns} FROM ({source})) "
            f"TO {_sql_literal(output)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
    finally:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)


def _validate_dataset(
    dataset_root: Path,
    *,
    dataset_name: str,
    expected_schema: dict[str, Any],
    primary_keys: tuple[str, ...],
) -> None:
    files = _parquet_files(dataset_root)
    if not files:
        raise ValueError(
            f"trade_level_history_publish_dataset_missing:{dataset_name}"
        )
    actual_schema = pl.read_parquet_schema(files[0])
    if list(actual_schema.items()) != list(expected_schema.items()):
        raise ValueError(
            f"trade_level_history_publish_schema_mismatch:{dataset_name}"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f"quant-lab-trade-level-{dataset_name}-validate-"
        )
    )
    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        _configure_duckdb(connection, temporary)
        source = _read_parquet_sql(files)
        description = connection.execute(
            f"DESCRIBE {source}"
        ).fetchall()
        columns = [str(row[0]) for row in description]
        if columns != list(expected_schema):
            raise ValueError(
                f"trade_level_history_publish_schema_mismatch:"
                f"{dataset_name}"
            )
        key_sql = ",".join(
            _identifier(column) for column in primary_keys
        )
        duplicate = connection.execute(
            f"SELECT 1 FROM ({source}) GROUP BY {key_sql} "
            "HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        null_predicate = " OR ".join(
            f"{_identifier(column)} IS NULL" for column in primary_keys
        )
        null_key = connection.execute(
            f"SELECT 1 FROM ({source}) WHERE {null_predicate} LIMIT 1"
        ).fetchone()
        if duplicate is not None or null_key is not None:
            raise ValueError(
                f"trade_level_history_publish_primary_key_invalid:"
                f"{dataset_name}"
            )
    finally:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)


def _dataset_digest(
    dataset_root: Path,
    *,
    dataset_name: str,
    managed_columns: list[str],
) -> str:
    files = _parquet_files(dataset_root)
    if not files:
        raise RuntimeError(
            f"trade_level_history_dataset_missing:{dataset_name}"
        )
    source = _read_parquet_sql(files)
    connection = duckdb.connect(database=":memory:", read_only=False)
    try:
        description = connection.execute(
            f"DESCRIBE {source}"
        ).fetchall()
        schema_by_column = {
            str(row[0]): str(row[1]) for row in description
        }
        if set(managed_columns) - set(schema_by_column):
            raise RuntimeError(
                f"trade_level_history_managed_columns_missing:"
                f"{dataset_name}"
            )
        schema = [
            (column, schema_by_column[column])
            for column in managed_columns
        ]
        hash_sql = "hash(" + ",".join(
            _identifier(column) for column in managed_columns
        ) + ")"
        row = connection.execute(
            "SELECT COUNT(*), "
            f"COALESCE(SUM(CAST({hash_sql} AS HUGEINT)), 0), "
            f"COALESCE(BIT_XOR({hash_sql}), 0), "
            f"COALESCE(MIN({hash_sql}), 0), "
            f"COALESCE(MAX({hash_sql}), 0) "
            f"FROM ({source})"
        ).fetchone()
    finally:
        connection.close()
    return model_content_sha256(
        {
            "schema_version": (
                "trade_level_history_dataset_digest.v1"
            ),
            "dataset_name": dataset_name,
            "managed_columns": managed_columns,
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
) -> None:
    atomic_write_json(
        staged / TRADE_LEVEL_HISTORY_SIDECAR,
        {
            "schema_version": TRADE_LEVEL_HISTORY_GENERATION_SCHEMA,
            "generation_id": generation_payload["generation_id"],
            "generation_digest": generation_payload[
                "generation_digest"
            ],
            "task_id": generation_payload["task_id"],
            "snapshot_id": generation_payload["snapshot_id"],
            "dataset_name": dataset_name,
            "dataset_hash": generation_payload["dataset_hashes"][
                dataset_name
            ],
            "row_count": generation_payload["row_counts"][dataset_name],
            "primary_keys": generation_payload["primary_keys"][
                dataset_name
            ],
            "managed_columns": generation_payload["managed_columns"][
                dataset_name
            ],
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
            "row_count": generation_payload["row_counts"][dataset_name],
            "source_sha": generation_payload["generation_digest"],
            "file_count": len(_parquet_files(staged)),
            "schema_version": TRADE_LEVEL_HISTORY_GENERATION_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def _generation_digest_identity(
    *,
    task_id: str,
    snapshot_id: str,
    previous_generation_id: object,
    previous_generation_digest: object,
    quant_lab_commit: str,
    worker_commit: str,
    candidate_evidence_generation_id: str,
    candidate_evidence_generation_digest: str,
    candidate_evidence_input_fingerprint: str,
    derived_event_digest: str,
    candidate_label_dataset_hash: str,
    input_fingerprint_digest: str,
    history_mode: str,
    schema_versions: dict[str, Any],
    similarity_availability_policy: str,
    result_outputs: list[Any],
    row_counts: dict[str, int],
    dataset_hashes: dict[str, str],
    managed_columns: dict[str, list[str]],
    primary_keys: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "schema_version": "trade_level_history_generation_digest.v1",
        "task_id": task_id,
        "snapshot_id": snapshot_id,
        "previous_generation_id": previous_generation_id,
        "previous_generation_digest": previous_generation_digest,
        "quant_lab_commit": quant_lab_commit,
        "worker_commit": worker_commit,
        "candidate_evidence_generation_id": (
            candidate_evidence_generation_id
        ),
        "candidate_evidence_generation_digest": (
            candidate_evidence_generation_digest
        ),
        "candidate_evidence_input_fingerprint": (
            candidate_evidence_input_fingerprint
        ),
        "derived_event_digest": derived_event_digest,
        "candidate_label_dataset_hash": candidate_label_dataset_hash,
        "input_fingerprint_digest": input_fingerprint_digest,
        "history_mode": history_mode,
        "schema_versions": schema_versions,
        "similarity_availability_policy": (
            similarity_availability_policy
        ),
        "result_outputs": result_outputs,
        "row_counts": row_counts,
        "dataset_hashes": dataset_hashes,
        "managed_columns": managed_columns,
        "primary_keys": primary_keys,
    }


def _read_pointer(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"trade_level_history_generation_pointer_invalid:{path.name}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"trade_level_history_generation_pointer_invalid:{path.name}"
        )
    return payload


def _read_optional_pointer(path: Path) -> dict[str, Any]:
    return _read_pointer(path) if path.is_file() else {}


def _configure_duckdb(
    connection: duckdb.DuckDBPyConnection,
    temporary: Path,
) -> None:
    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET memory_limit = '512MB'")
    connection.execute(
        f"SET temp_directory = {_sql_literal(temporary)}"
    )


def _parquet_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.parquet") if path.is_file()
    )


def _read_parquet_sql(paths: list[Path]) -> str:
    values = ",".join(_sql_literal(path) for path in paths)
    return (
        "SELECT * FROM read_parquet("
        f"[{values}], union_by_name=true, hive_partitioning=false)"
    )


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
