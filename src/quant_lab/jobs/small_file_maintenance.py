from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.file_index import LAKE_FILE_INDEX, build_lake_file_index
from quant_lab.data.lake import CompactParquetResult, compact_parquet_directory_files

DEFAULT_PRIORITY_DATASETS = (
    "silver/orderbook_snapshot",
    "silver/trade_print",
    "bronze/okx_public_ws",
    "silver/v5_quant_lab_request",
    "silver/v5_candidate_event",
    "gold/v5_candidate_label",
    "silver/v5_decision_audit",
    "bronze/api_request_metrics",
)


@dataclass(frozen=True)
class SmallFileGroup:
    dataset: str
    partition_dir: str
    file_count: int
    direct_source_file_count: int
    compact_file_count: int
    avg_file_size: float
    total_size: int
    priority_rank: int
    include_existing_compact_files: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SmallFileMaintenanceResult:
    lake_root: str
    dry_run: bool
    started_at: datetime
    finished_at: datetime | None = None
    indexed_rows: int = 0
    candidate_group_count: int = 0
    processed_group_count: int = 0
    compacted_group_count: int = 0
    source_file_count: int = 0
    output_file_count: int = 0
    groups: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["started_at"] = self.started_at.isoformat()
        payload["finished_at"] = self.finished_at.isoformat() if self.finished_at else None
        return payload


def lake_small_file_maintenance(
    lake_root: str | Path,
    *,
    min_files: int = 16,
    max_avg_file_size_mb: float = 8.0,
    max_groups: int = 50,
    target_rows_per_file: int = 250_000,
    max_source_files_per_batch: int = 64,
    max_source_files_per_group: int | None = 64,
    max_source_batch_bytes: int | None = 268_435_456,
    priority_datasets: tuple[str, ...] = DEFAULT_PRIORITY_DATASETS,
    dry_run: bool = False,
) -> SmallFileMaintenanceResult:
    root = Path(lake_root)
    started = datetime.now(UTC)
    result = SmallFileMaintenanceResult(
        lake_root=str(root),
        dry_run=bool(dry_run),
        started_at=started,
    )
    try:
        index_delta = build_lake_file_index(root, priority_datasets)
        result.indexed_rows = index_delta.height
    except Exception as exc:
        result.warnings.append(f"build_lake_file_index_failed:{type(exc).__name__}:{exc}")
        result.finished_at = datetime.now(UTC)
        return result

    groups = small_file_groups(
        root,
        min_files=min_files,
        max_avg_file_size_mb=max_avg_file_size_mb,
        priority_datasets=priority_datasets,
    )
    result.candidate_group_count = len(groups)
    selected = groups[: max(int(max_groups), 0)]
    for group in selected:
        entry: dict[str, Any] = group.to_dict()
        if dry_run:
            entry["action"] = "would_compact"
            result.groups.append(entry)
            result.processed_group_count += 1
            continue
        try:
            compact_result = compact_parquet_directory_files(
                root / group.partition_dir,
                target_rows_per_file=target_rows_per_file,
                max_source_files_per_batch=max_source_files_per_batch,
                max_source_batch_bytes=max_source_batch_bytes,
                include_existing_compact_files=group.include_existing_compact_files,
                max_total_source_files=max_source_files_per_group,
            )
        except Exception as exc:
            entry["action"] = "compact_failed"
            entry["error"] = f"{type(exc).__name__}:{exc}"
            result.warnings.append(
                f"compact_failed:{group.partition_dir}:{type(exc).__name__}:{exc}"
            )
            result.groups.append(entry)
            result.processed_group_count += 1
            continue
        entry.update(_compact_result_payload(compact_result))
        entry["action"] = "compacted"
        result.groups.append(entry)
        result.processed_group_count += 1
        result.compacted_group_count += 1 if compact_result.source_file_count else 0
        result.source_file_count += compact_result.source_file_count
        result.output_file_count += compact_result.output_file_count

    result.finished_at = datetime.now(UTC)
    return result


def small_file_groups(
    lake_root: str | Path,
    *,
    min_files: int = 16,
    max_avg_file_size_mb: float = 8.0,
    priority_datasets: tuple[str, ...] = DEFAULT_PRIORITY_DATASETS,
) -> list[SmallFileGroup]:
    root = Path(lake_root)
    index = _read_file_index(root)
    required = {"dataset", "path", "file_size"}
    if index.is_empty() or not required.issubset(set(index.columns)):
        return []
    priority_rank = {dataset: rank for rank, dataset in enumerate(priority_datasets)}
    max_avg_bytes = max(float(max_avg_file_size_mb), 0.0) * 1024 * 1024
    rows: list[dict[str, Any]] = []
    for row in index.to_dicts():
        dataset = str(row.get("dataset") or "")
        if dataset not in priority_rank:
            continue
        relative_path = str(row.get("path") or "")
        file_kind = _compaction_file_kind(relative_path)
        if file_kind is None:
            continue
        file_size = _int_value(row.get("file_size"))
        if file_size is None or file_size <= 0:
            continue
        partition_dir = str(Path(relative_path).parent).replace("\\", "/")
        rows.append(
            {
                "dataset": dataset,
                "partition_dir": partition_dir,
                "file_size": file_size,
                "is_direct_source": file_kind == "direct",
                "is_compact_output": file_kind == "compact",
                "priority_rank": priority_rank[dataset],
            }
        )
    if not rows:
        return []
    frame = pl.DataFrame(rows)
    grouped = (
        frame.group_by(["dataset", "partition_dir", "priority_rank"])
        .agg(
            [
                pl.len().alias("file_count"),
                pl.col("is_direct_source").sum().alias("direct_source_file_count"),
                pl.col("is_compact_output").sum().alias("compact_file_count"),
                pl.col("file_size").sum().alias("total_size"),
                pl.col("file_size").mean().alias("avg_file_size"),
            ]
        )
        .filter(
            (
                (pl.col("direct_source_file_count") >= max(int(min_files), 1))
                | (pl.col("compact_file_count") >= max(int(min_files), 1))
            )
            & (pl.col("avg_file_size") < max_avg_bytes)
        )
        .sort(["priority_rank", "file_count", "total_size"], descending=[False, True, True])
    )
    return [
        SmallFileGroup(
            dataset=str(row["dataset"]),
            partition_dir=str(row["partition_dir"]),
            file_count=int(row["file_count"]),
            direct_source_file_count=int(row["direct_source_file_count"]),
            compact_file_count=int(row["compact_file_count"]),
            avg_file_size=float(row["avg_file_size"]),
            total_size=int(row["total_size"]),
            priority_rank=int(row["priority_rank"]),
            include_existing_compact_files=int(row["compact_file_count"]) >= max(
                int(min_files),
                1,
            ),
        )
        for row in grouped.to_dicts()
    ]


def _read_file_index(root: Path) -> pl.DataFrame:
    try:
        from quant_lab.data.lake import read_parquet_dataset

        return read_parquet_dataset(root / LAKE_FILE_INDEX)
    except Exception:
        return pl.DataFrame()


def _compaction_file_kind(relative_path: str) -> str | None:
    name = Path(relative_path).name
    if not name.endswith(".parquet"):
        return None
    if name.startswith(".") or name == "data.parquet":
        return None
    return "compact" if name.startswith("compact_") else "direct"


def _compact_result_payload(result: CompactParquetResult) -> dict[str, Any]:
    return {
        "compact_source_file_count": result.source_file_count,
        "compact_output_file_count": result.output_file_count,
        "compact_rows": result.rows,
        "compact_target_rows_per_file": result.target_rows_per_file,
        "compact_max_source_files_per_batch": result.max_source_files_per_batch,
        "compact_max_source_batch_bytes": result.max_source_batch_bytes,
    }


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
