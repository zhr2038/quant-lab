from pathlib import Path

import polars as pl

from quant_lab.features.publish import (
    FEATURE_ANOMALY_DATASET,
    FEATURE_COVERAGE_DATASET,
    compute_feature_anomalies,
    compute_feature_coverage,
    feature_health,
)

__all__ = [
    "FEATURE_ANOMALY_DATASET",
    "FEATURE_COVERAGE_DATASET",
    "compute_feature_anomalies",
    "compute_feature_coverage",
    "feature_health",
    "read_feature_anomalies",
    "read_feature_coverage",
]


def read_feature_coverage(lake_root: str | Path) -> pl.DataFrame:
    from quant_lab.data.lake import read_parquet_dataset

    return read_parquet_dataset(Path(lake_root) / FEATURE_COVERAGE_DATASET)


def read_feature_anomalies(lake_root: str | Path) -> pl.DataFrame:
    from quant_lab.data.lake import read_parquet_dataset

    return read_parquet_dataset(Path(lake_root) / FEATURE_ANOMALY_DATASET)
