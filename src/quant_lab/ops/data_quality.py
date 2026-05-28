from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import (
    count_parquet_rows,
    invalid_parquet_files,
    read_parquet_lazy,
)
from quant_lab.ops.dataset_registry import DatasetSpec, dataset_registry


@dataclass(frozen=True)
class DataQualityCheck:
    dataset: str
    rule: str
    status: str
    severity: str
    detail: str
    owner: str
    path: str
    next_action: str = ""
    observed_value: str | None = None
    expected_value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "rule": self.rule,
            "status": self.status,
            "severity": self.severity,
            "detail": self.detail,
            "owner": self.owner,
            "path": self.path,
            "next_action": self.next_action,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
        }


@dataclass(frozen=True)
class DataQualitySummary:
    status: str
    generated_at: datetime
    dataset_count: int
    check_count: int
    fail_count: int
    warning_count: int
    checks: tuple[DataQualityCheck, ...]

    def to_dict(self, *, include_checks: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "generated_at": self.generated_at.isoformat(),
            "dataset_count": self.dataset_count,
            "check_count": self.check_count,
            "fail_count": self.fail_count,
            "warning_count": self.warning_count,
        }
        if include_checks:
            payload["checks"] = [check.to_dict() for check in self.checks]
        return payload


def run_data_quality(
    lake_root: str | Path,
    *,
    dataset_names: list[str] | tuple[str, ...] | set[str] | None = None,
    reference_at: datetime | None = None,
    registry: dict[str, DatasetSpec] | None = None,
) -> DataQualitySummary:
    root = Path(lake_root)
    generated_at = _utc(reference_at or datetime.now(UTC))
    specs = registry or dataset_registry()
    selected_names = sorted(dataset_names) if dataset_names is not None else sorted(specs)
    checks: list[DataQualityCheck] = []
    for name in selected_names:
        spec = specs.get(name)
        if spec is None:
            spec = DatasetSpec(
                dataset_id=name,
                layer="unknown",
                relative_path=Path(name),
                owner="unknown",
                required=False,
                min_rows=0,
            )
        checks.extend(_dataset_checks(root, spec, reference_at=generated_at))

    fail_count = sum(1 for check in checks if check.status == "FAIL")
    warning_count = sum(1 for check in checks if check.status == "WARN")
    status = "FAIL" if fail_count else ("WARN" if warning_count else "PASS")
    return DataQualitySummary(
        status=status,
        generated_at=generated_at,
        dataset_count=len(selected_names),
        check_count=len(checks),
        fail_count=fail_count,
        warning_count=warning_count,
        checks=tuple(checks),
    )


def _dataset_checks(
    root: Path,
    spec: DatasetSpec,
    *,
    reference_at: datetime,
) -> list[DataQualityCheck]:
    path = root / spec.relative_path
    checks: list[DataQualityCheck] = []
    invalid_files = invalid_parquet_files(path)
    if invalid_files:
        checks.append(
            _check(
                spec,
                "parquet_valid",
                False,
                f"invalid_parquet_files={len(invalid_files)}",
                path,
                severity="critical",
                next_action="repair or remove invalid parquet files before reading the dataset",
            )
        )
        return checks

    row_count = count_parquet_rows(path)
    checks.append(
        _check(
            spec,
            "row_count_min",
            row_count >= spec.min_rows,
            f"rows={row_count}; min_rows={spec.min_rows}",
            path,
            severity="critical" if spec.required else "warning",
            status="PASS" if row_count >= spec.min_rows else ("FAIL" if spec.required else "WARN"),
            observed_value=str(row_count),
            expected_value=str(spec.min_rows),
            next_action="run the producer job or inspect upstream source if rows are missing",
        )
    )
    if row_count <= 0:
        return checks

    lazy = read_parquet_lazy(path)
    try:
        schema = lazy.collect_schema()
    except Exception as exc:
        checks.append(
            _check(
                spec,
                "schema_readable",
                False,
                f"collect_schema_failed={type(exc).__name__}:{exc}",
                path,
                severity="critical",
                next_action="repair the parquet dataset or regenerate it from source",
            )
        )
        return checks

    available_columns = set(schema.names())
    missing_columns = [
        column for column in spec.required_columns if column not in available_columns
    ]
    checks.append(
        _check(
            spec,
            "schema_required_columns",
            not missing_columns,
            "ok" if not missing_columns else f"missing_columns={missing_columns}",
            path,
            severity="critical",
            next_action="publish the dataset with the current contract schema",
        )
    )
    checks.extend(_utc_column_checks(spec, path, schema))
    checks.extend(_primary_key_checks(spec, path, lazy, available_columns))
    checks.extend(_closed_bar_checks(spec, path, lazy, available_columns))
    checks.extend(_freshness_checks(spec, path, lazy, available_columns, reference_at))
    checks.extend(_cost_checks(spec, path, lazy, available_columns))
    checks.extend(_risk_permission_checks(spec, path, lazy, available_columns, reference_at))
    return checks


def _utc_column_checks(
    spec: DatasetSpec,
    path: Path,
    schema: pl.Schema,
) -> list[DataQualityCheck]:
    checks: list[DataQualityCheck] = []
    for column in spec.utc_timestamp_columns:
        if column not in schema:
            continue
        dtype = schema[column]
        dtype_text = str(dtype)
        if dtype_text.startswith("Datetime"):
            passed = "UTC" in dtype_text
            checks.append(
                _check(
                    spec,
                    f"utc_timestamp:{column}",
                    passed,
                    f"{column} dtype={dtype_text}",
                    path,
                    severity="critical",
                    next_action=f"normalize {column} to timezone-aware UTC before writing",
                )
            )
    return checks


def _primary_key_checks(
    spec: DatasetSpec,
    path: Path,
    lazy: pl.LazyFrame,
    available_columns: set[str],
) -> list[DataQualityCheck]:
    if not spec.primary_key or not set(spec.primary_key).issubset(available_columns):
        return []
    try:
        duplicate_count = (
            lazy.group_by(list(spec.primary_key))
            .len()
            .filter(pl.col("len") > 1)
            .select(pl.len().alias("duplicate_key_count"))
            .collect()
            .item()
        )
    except Exception as exc:
        return [
            _check(
                spec,
                "primary_key_unique",
                False,
                f"duplicate_check_failed={type(exc).__name__}:{exc}",
                path,
                severity="warning",
                status="WARN",
                next_action="rerun lake-health after compaction or inspect schema drift",
            )
        ]
    return [
        _check(
            spec,
            "primary_key_unique",
            int(duplicate_count or 0) == 0,
            f"duplicate_key_count={int(duplicate_count or 0)}",
            path,
            severity="critical",
            next_action="dedupe/upsert the dataset using the registered primary key",
            observed_value=str(int(duplicate_count or 0)),
            expected_value="0",
        )
    ]


def _closed_bar_checks(
    spec: DatasetSpec,
    path: Path,
    lazy: pl.LazyFrame,
    available_columns: set[str],
) -> list[DataQualityCheck]:
    if spec.closed_bar_column is None or spec.closed_bar_column not in available_columns:
        return []
    try:
        unclosed_count = (
            lazy.filter(pl.col(spec.closed_bar_column) != True)  # noqa: E712
            .select(pl.len().alias("unclosed_count"))
            .collect()
            .item()
        )
    except Exception as exc:
        return [
            _check(
                spec,
                "closed_bar_only",
                False,
                f"closed_bar_check_failed={type(exc).__name__}:{exc}",
                path,
                severity="critical",
                next_action="repair market_bar schema before feature publishing",
            )
        ]
    return [
        _check(
            spec,
            "closed_bar_only",
            int(unclosed_count or 0) == 0,
            f"unclosed_count={int(unclosed_count or 0)}",
            path,
            severity="critical",
            next_action="publish only closed bars; never compute features on open bars",
            observed_value=str(int(unclosed_count or 0)),
            expected_value="0",
        )
    ]


def _freshness_checks(
    spec: DatasetSpec,
    path: Path,
    lazy: pl.LazyFrame,
    available_columns: set[str],
    reference_at: datetime,
) -> list[DataQualityCheck]:
    if spec.timestamp_column is None or spec.timestamp_column not in available_columns:
        return []
    try:
        latest = (
            lazy.select(
                pl.col(spec.timestamp_column)
                .cast(pl.Utf8)
                .str.to_datetime(time_zone="UTC", strict=False)
                .max()
                .alias("latest_ts")
            )
            .collect()
            .item()
        )
    except Exception as exc:
        return [
            _check(
                spec,
                "freshness",
                False,
                f"freshness_check_failed={type(exc).__name__}:{exc}",
                path,
                severity="warning",
                status="WARN",
                next_action="inspect timestamp column parsing and schema drift",
            )
        ]
    if not isinstance(latest, datetime):
        return [
            _check(
                spec,
                "freshness",
                False,
                "latest_ts not observable",
                path,
                severity="warning",
                status="WARN",
                next_action="ensure the registered timestamp column is populated",
            )
        ]
    latest_utc = _utc(latest)
    age_seconds = max(0, int((reference_at - latest_utc).total_seconds()))
    threshold = spec.freshness_seconds
    passed = threshold is None or age_seconds <= threshold
    return [
        _check(
            spec,
            "freshness",
            passed,
            f"latest_ts={latest_utc.isoformat()}; age_seconds={age_seconds}; threshold={threshold}",
            path,
            severity="critical" if spec.required else "warning",
            status="PASS" if passed else ("FAIL" if spec.required else "WARN"),
            next_action="run the producer job or inspect the upstream V5/OKX sync",
            observed_value=str(age_seconds),
            expected_value=str(threshold) if threshold is not None else None,
        )
    ]


def _cost_checks(
    spec: DatasetSpec,
    path: Path,
    lazy: pl.LazyFrame,
    available_columns: set[str],
) -> list[DataQualityCheck]:
    if spec.dataset_id != "cost_bucket_daily":
        return []
    columns = {"cost_source", "source", "fallback_level"}.intersection(available_columns)
    if not columns:
        return []
    source_expr = pl.concat_str([pl.col(column).cast(pl.Utf8) for column in sorted(columns)])
    try:
        hard_fallback_count = (
            lazy.with_columns(source_expr.str.to_lowercase().alias("_cost_text"))
            .filter(
                pl.col("_cost_text").str.contains("global_default")
                | pl.col("_cost_text").str.contains("service_unavailable")
                | pl.col("_cost_text").str.contains("symbol_missing")
            )
            .select(pl.len().alias("hard_fallback_count"))
            .collect()
            .item()
        )
    except Exception as exc:
        return [
            _check(
                spec,
                "cost_hard_fallback_visibility",
                False,
                f"cost_fallback_check_failed={type(exc).__name__}:{exc}",
                path,
                severity="warning",
                status="WARN",
                next_action="inspect cost_bucket_daily fallback schema",
            )
        ]
    return [
        _check(
            spec,
            "cost_hard_fallback_visibility",
            int(hard_fallback_count or 0) == 0,
            f"hard_fallback_count={int(hard_fallback_count or 0)}",
            path,
            severity="critical",
            next_action="calibrate symbol-level costs before trusting live cost estimates",
            observed_value=str(int(hard_fallback_count or 0)),
            expected_value="0",
        )
    ]


def _risk_permission_checks(
    spec: DatasetSpec,
    path: Path,
    lazy: pl.LazyFrame,
    available_columns: set[str],
    reference_at: datetime,
) -> list[DataQualityCheck]:
    if spec.dataset_id != "risk_permission" or "expires_at" not in available_columns:
        return []
    try:
        latest = (
            lazy.select(
                pl.col("expires_at")
                .cast(pl.Utf8)
                .str.to_datetime(time_zone="UTC", strict=False)
                .max()
                .alias("latest_expires_at")
            )
            .collect()
            .item()
        )
    except Exception as exc:
        return [
            _check(
                spec,
                "risk_permission_not_expired",
                False,
                f"expires_at_check_failed={type(exc).__name__}:{exc}",
                path,
                severity="critical",
                next_action="rerun publish-risk-permission before export-daily",
            )
        ]
    passed = isinstance(latest, datetime) and _utc(latest) > reference_at
    detail = (
        f"latest_expires_at={_utc(latest).isoformat()}"
        if isinstance(latest, datetime)
        else "latest_expires_at not observable"
    )
    return [
        _check(
            spec,
            "risk_permission_not_expired",
            passed,
            detail,
            path,
            severity="critical",
            next_action="run qlab publish-risk-permission before qlab export-daily",
        )
    ]


def _check(
    spec: DatasetSpec,
    rule: str,
    passed: bool,
    detail: str,
    path: Path,
    *,
    severity: str,
    status: str | None = None,
    next_action: str = "",
    observed_value: str | None = None,
    expected_value: str | None = None,
) -> DataQualityCheck:
    if status is None:
        status = "PASS" if passed else ("FAIL" if severity == "critical" else "WARN")
    return DataQualityCheck(
        dataset=spec.dataset_id,
        rule=rule,
        status=status,
        severity=severity,
        detail=detail,
        owner=spec.owner,
        path=str(path),
        next_action=next_action,
        observed_value=observed_value,
        expected_value=expected_value,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
