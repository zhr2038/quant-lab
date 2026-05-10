"""Lake and catalog access helpers."""

from quant_lab.data.lake import (
    LakeConfig,
    query_dataset_sql,
    read_parquet_dataset,
    read_parquet_lazy,
    scan_parquet_with_duckdb,
    upsert_parquet_dataset,
    write_parquet_dataset,
)

__all__ = [
    "LakeConfig",
    "query_dataset_sql",
    "read_parquet_dataset",
    "read_parquet_lazy",
    "scan_parquet_with_duckdb",
    "upsert_parquet_dataset",
    "write_parquet_dataset",
]
