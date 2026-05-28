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
    checks.extend(_market_bar_ohlc_checks(spec, path, lazy, available_columns))
    checks.extend(_feature_value_checks(spec, path, lazy, available_columns))
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
    checks: list[DataQualityCheck] = []
    if "sample_count" in available_columns:
        try:
            low_sample_count = (
                lazy.filter(pl.col("sample_count").cast(pl.Int64, strict=False) < 30)
                .select(pl.len().alias("low_sample_count"))
                .collect()
                .item()
            )
            checks.append(
                _check(
                    spec,
                    "cost_low_sample_count",
                    int(low_sample_count or 0) == 0,
                    f"low_sample_count={int(low_sample_count or 0)}; threshold=30",
                    path,
                    severity="warning",
                    status="PASS" if int(low_sample_count or 0) == 0 else "WARN",
                    next_action="collect more actual/mixed cost samples before trusting live cost",
                    observed_value=str(int(low_sample_count or 0)),
                    expected_value="0",
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    spec,
                    "cost_low_sample_count",
                    False,
                    f"cost_low_sample_check_failed={type(exc).__name__}:{exc}",
                    path,
                    severity="warning",
                    status="WARN",
                    next_action="inspect cost_bucket_daily sample_count schema",
                )
            )
    bps_columns = [
        column
        for column in available_columns
        if column.endswith("_bps")
        or "_bps_" in column
        or column in {"cost_bps", "selected_total_cost_bps"}
    ]
    if bps_columns:
        try:
            negative_conditions = [
                pl.col(column).cast(pl.Float64, strict=False) < 0 for column in bps_columns
            ]
            negative_bps_count = (
                lazy.filter(pl.any_horizontal(negative_conditions))
                .select(pl.len().alias("negative_bps_count"))
                .collect()
                .item()
            )
            checks.append(
                _check(
                    spec,
                    "cost_negative_bps",
                    int(negative_bps_count or 0) == 0,
                    f"negative_bps_count={int(negative_bps_count or 0)}; columns={bps_columns}",
                    path,
                    severity="critical",
                    next_action=(
                        "repair cost calibration output; cost bps fields must be non-negative"
                    ),
                    observed_value=str(int(negative_bps_count or 0)),
                    expected_value="0",
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    spec,
                    "cost_negative_bps",
                    False,
                    f"cost_negative_bps_check_failed={type(exc).__name__}:{exc}",
                    path,
                    severity="warning",
                    status="WARN",
                    next_action="inspect cost bps schema drift",
                )
            )
    columns = {"cost_source", "source", "fallback_level"}.intersection(available_columns)
    if not columns:
        return checks
    source_expr = pl.concat_str([pl.col(column).cast(pl.Utf8) for column in sorted(columns)])
    try:
        with_cost_text = lazy.with_columns(source_expr.str.to_lowercase().alias("_cost_text"))
        hard_fallback_count = (
            with_cost_text
            .filter(
                pl.col("_cost_text").str.contains("global_default")
                | pl.col("_cost_text").str.contains("service_unavailable")
                | pl.col("_cost_text").str.contains("symbol_missing")
            )
            .select(pl.len().alias("hard_fallback_count"))
            .collect()
            .item()
        )
        public_proxy_count = (
            with_cost_text
            .filter(pl.col("_cost_text").str.contains("public_spread_proxy"))
            .select(pl.len().alias("public_proxy_count"))
            .collect()
            .item()
        )
    except Exception as exc:
        checks.append(
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
        )
        return checks
    checks.extend(
        [
            _check(
                spec,
                "cost_public_proxy_visibility",
                int(public_proxy_count or 0) == 0,
                f"public_proxy_count={int(public_proxy_count or 0)}",
                path,
                severity="warning",
                status="PASS" if int(public_proxy_count or 0) == 0 else "WARN",
                next_action=(
                    "replace public spread proxy with actual/mixed cost evidence when possible"
                ),
                observed_value=str(int(public_proxy_count or 0)),
                expected_value="0",
            ),
        ]
    )
    checks.append(
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
    )
    return checks


def _market_bar_ohlc_checks(
    spec: DatasetSpec,
    path: Path,
    lazy: pl.LazyFrame,
    available_columns: set[str],
) -> list[DataQualityCheck]:
    if spec.dataset_id != "market_bar":
        return []
    required = {"open", "high", "low", "close"}
    if not required.issubset(available_columns):
        return []
    try:
        invalid_count = (
            lazy.filter(
                (pl.col("open") <= 0)
                | (pl.col("high") <= 0)
                | (pl.col("low") <= 0)
                | (pl.col("close") <= 0)
                | (pl.col("high") < pl.col("low"))
                | (pl.col("high") < pl.max_horizontal("open", "close"))
                | (pl.col("low") > pl.min_horizontal("open", "close"))
            )
            .select(pl.len().alias("invalid_ohlc_count"))
            .collect()
            .item()
        )
    except Exception as exc:
        return [
            _check(
                spec,
                "market_bar_ohlc_valid",
                False,
                f"ohlc_check_failed={type(exc).__name__}:{exc}",
                path,
                severity="critical",
                next_action="repair market_bar numeric schema before feature publishing",
            )
        ]
    return [
        _check(
            spec,
            "market_bar_ohlc_valid",
            int(invalid_count or 0) == 0,
            f"invalid_ohlc_count={int(invalid_count or 0)}",
            path,
            severity="critical",
            next_action="drop or repair invalid OHLC bars before feature publishing",
            observed_value=str(int(invalid_count or 0)),
            expected_value="0",
        )
    ]


def _feature_value_checks(
    spec: DatasetSpec,
    path: Path,
    lazy: pl.LazyFrame,
    available_columns: set[str],
) -> list[DataQualityCheck]:
    if spec.dataset_id != "feature_value":
        return []
    checks: list[DataQualityCheck] = []
    if "value" in available_columns:
        try:
            infinite_count = (
                lazy.filter(
                    pl.col("value")
                    .cast(pl.Float64, strict=False)
                    .is_infinite()
                    .fill_null(False)
                )
                .select(pl.len().alias("infinite_count"))
                .collect()
                .item()
            )
            checks.append(
                _check(
                    spec,
                    "feature_value_no_infinite",
                    int(infinite_count or 0) == 0,
                    f"infinite_count={int(infinite_count or 0)}",
                    path,
                    severity="critical",
                    next_action="repair feature computation; values must be finite or null",
                    observed_value=str(int(infinite_count or 0)),
                    expected_value="0",
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    spec,
                    "feature_value_no_infinite",
                    False,
                    f"infinite_check_failed={type(exc).__name__}:{exc}",
                    path,
                    severity="warning",
                    status="WARN",
                    next_action="inspect feature_value value dtype",
                )
            )
    if {"value", "is_valid"}.issubset(available_columns):
        invalid_reason_expr = (
            pl.col("invalid_reason").cast(pl.Utf8).fill_null("")
            if "invalid_reason" in available_columns
            else pl.lit("")
        )
        try:
            null_valid_count = (
                lazy.with_columns(invalid_reason_expr.alias("_invalid_reason"))
                .filter(
                    pl.col("value").is_null()
                    & (pl.col("is_valid") == True)  # noqa: E712
                    & (pl.col("_invalid_reason").str.strip_chars() == "")
                )
                .select(pl.len().alias("null_valid_count"))
                .collect()
                .item()
            )
            checks.append(
                _check(
                    spec,
                    "feature_value_null_valid_consistency",
                    int(null_valid_count or 0) == 0,
                    f"null_valid_without_reason_count={int(null_valid_count or 0)}",
                    path,
                    severity="critical",
                    next_action="mark null feature values invalid or provide invalid_reason",
                    observed_value=str(int(null_valid_count or 0)),
                    expected_value="0",
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    spec,
                    "feature_value_null_valid_consistency",
                    False,
                    f"null_valid_check_failed={type(exc).__name__}:{exc}",
                    path,
                    severity="warning",
                    status="WARN",
                    next_action="inspect feature_value is_valid/value schema",
                )
            )
    if {"invalid_reason", "is_valid"}.issubset(available_columns):
        try:
            reason_valid_count = (
                lazy.filter(
                    (pl.col("is_valid") == True)  # noqa: E712
                    & (pl.col("invalid_reason").cast(pl.Utf8).fill_null("").str.strip_chars() != "")
                )
                .select(pl.len().alias("reason_valid_count"))
                .collect()
                .item()
            )
            checks.append(
                _check(
                    spec,
                    "feature_value_invalid_reason_consistency",
                    int(reason_valid_count or 0) == 0,
                    f"valid_with_invalid_reason_count={int(reason_valid_count or 0)}",
                    path,
                    severity="critical",
                    next_action="set is_valid=false when invalid_reason is populated",
                    observed_value=str(int(reason_valid_count or 0)),
                    expected_value="0",
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    spec,
                    "feature_value_invalid_reason_consistency",
                    False,
                    f"invalid_reason_check_failed={type(exc).__name__}:{exc}",
                    path,
                    severity="warning",
                    status="WARN",
                    next_action="inspect feature_value invalid_reason schema",
                )
            )
    return checks


def _risk_permission_checks(
    spec: DatasetSpec,
    path: Path,
    lazy: pl.LazyFrame,
    available_columns: set[str],
    reference_at: datetime,
) -> list[DataQualityCheck]:
    if spec.dataset_id != "risk_permission" or "expires_at" not in available_columns:
        return []
    checks: list[DataQualityCheck] = []
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
    checks.append(
        _check(
            spec,
            "risk_permission_not_expired",
            passed,
            detail,
            path,
            severity="critical",
            next_action="run qlab publish-risk-permission before qlab export-daily",
        )
    )
    if {"permission_status", "expires_at"}.issubset(available_columns):
        try:
            active_expired_count = (
                lazy.with_columns(
                    pl.col("expires_at")
                    .cast(pl.Utf8)
                    .str.to_datetime(time_zone="UTC", strict=False)
                    .alias("_expires_at")
                )
                .filter(
                    pl.col("permission_status")
                    .cast(pl.Utf8)
                    .str.starts_with("ACTIVE_")
                    & (pl.col("_expires_at") <= reference_at)
                )
                .select(pl.len().alias("active_expired_count"))
                .collect()
                .item()
            )
            checks.append(
                _check(
                    spec,
                    "risk_permission_active_not_expired",
                    int(active_expired_count or 0) == 0,
                    f"active_expired_count={int(active_expired_count or 0)}",
                    path,
                    severity="critical",
                    next_action="republish risk_permission; ACTIVE_* rows must not be expired",
                    observed_value=str(int(active_expired_count or 0)),
                    expected_value="0",
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    spec,
                    "risk_permission_active_not_expired",
                    False,
                    f"active_expiry_check_failed={type(exc).__name__}:{exc}",
                    path,
                    severity="critical",
                    next_action="inspect risk_permission permission_status/expires_at schema",
                )
            )
    if {"permission_status", "enforceable"}.issubset(available_columns):
        try:
            non_active_enforceable_count = (
                lazy.filter(
                    ~pl.col("permission_status").cast(pl.Utf8).str.starts_with("ACTIVE_")
                    & (pl.col("enforceable") == True)  # noqa: E712
                )
                .select(pl.len().alias("non_active_enforceable_count"))
                .collect()
                .item()
            )
            checks.append(
                _check(
                    spec,
                    "risk_permission_enforceable_consistency",
                    int(non_active_enforceable_count or 0) == 0,
                    f"non_active_enforceable_count={int(non_active_enforceable_count or 0)}",
                    path,
                    severity="critical",
                    next_action="set enforceable=false for STALE/EXPIRED/NO_FRESH permissions",
                    observed_value=str(int(non_active_enforceable_count or 0)),
                    expected_value="0",
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    spec,
                    "risk_permission_enforceable_consistency",
                    False,
                    f"enforceable_consistency_check_failed={type(exc).__name__}:{exc}",
                    path,
                    severity="critical",
                    next_action="inspect risk_permission enforceable schema",
                )
            )
    return checks


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
