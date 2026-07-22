from __future__ import annotations

import gc
import hashlib
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as datetime_time
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset  # noqa: F401
from quant_lab.factors.factory import (
    FACTOR_CORRELATION_SCHEMA,
    FACTOR_EVIDENCE_SCHEMA,
    FACTOR_VALUE_SCHEMA,
    LEGACY_MAIN_DECISION_POLICY,
    FactorFactoryPureResult,
    build_factor_correlation_frame,
    build_factor_definition_frame,
    build_factor_evidence_from_labels,
    build_factor_value_frame,
    build_latest_symbol_cost_frame,
)
from quant_lab.research.labels import build_forward_return_labels
from quant_lab.research_plane.contracts import (
    DEFAULT_FACTOR_FACTORY_MAX_INPUT_UNCOMPRESSED_BYTES,
    FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1,
    FactorFactorySnapshotManifest,
    FactorFactoryTask,
)
from quant_lab.research_plane.factor_factory_snapshot import (
    COST_BUCKET_DAILY_DATASET,
    FEATURE_VALUE_DATASET,
    MARKET_BAR_DATASET,
)

FACTOR_FACTORY_ALLOWED_DECISIONS = frozenset({"KILL", "RESEARCH", "KEEP_SHADOW", "PAPER_READY"})
FACTOR_FACTORY_ANTI_LEAKAGE_CHECKS = (
    "task_snapshot_identity_matches",
    "task_snapshot_commit_matches",
    "snapshot_manifest_digest_matches_task",
    "factor_plan_digest_matches",
    "factor_plan_scope_matches_task",
    "factor_plan_membership_is_frozen",
    "source_input_digest_matches",
    "cost_input_digest_matches",
    "previous_generation_binding_matches",
    "result_mode_is_parity_full",
    "history_mode_is_bootstrap_full",
    "snapshot_dataset_allowlist_exact",
    "snapshot_files_are_manifest_bound",
    "feature_scope_matches_plan",
    "feature_timestamps_within_signed_bounds",
    "feature_primary_keys_are_unique",
    "feature_rows_are_valid_only",
    "market_timeframe_matches_task",
    "market_rows_are_closed_only",
    "market_timestamps_within_signed_bounds",
    "market_primary_keys_are_unique",
    "cost_symbol_selection_is_unique",
    "cost_rows_match_signed_snapshot",
    "decision_delay_is_positive",
    "horizons_are_positive_sorted_unique",
    "factor_values_use_planned_factor_ids_only",
    "factor_value_primary_keys_are_unique",
    "factor_available_time_not_before_event_time",
    "factor_available_time_matches_plan_lag",
    "factor_event_time_matches_ts",
    "factor_value_keys_exist_in_snapshot_features",
    "factor_value_timestamps_within_feature_bounds",
    "evidence_uses_requested_horizons_only",
    "evidence_decisions_are_research_only",
    "evidence_primary_keys_are_unique",
    "correlation_primary_keys_are_unique",
    "nas_does_not_derive_factor_candidates",
    "automatic_promotion_is_disabled",
    "live_notional_is_zero",
    "live_order_effect_is_none_read_only_research",
)


@dataclass(frozen=True)
class FactorFactoryComputeArtifacts:
    generated_at: datetime
    definitions: pl.DataFrame
    values: StagedFactorValueSet
    evidence: pl.DataFrame
    correlations: pl.DataFrame
    anti_leakage: dict[str, Any]
    worker_report: dict[str, Any]
    warnings: tuple[str, ...]
    no_update_reason: str | None
    factor_ids: tuple[str, ...]


@dataclass(frozen=True)
class StagedFactorValueSet:
    paths: tuple[Path, ...]
    row_count: int
    uncompressed_bytes: int

    @property
    def height(self) -> int:
        return self.row_count

    def is_empty(self) -> bool:
        return self.row_count == 0

    def collect(self) -> pl.DataFrame:
        if not self.paths:
            return pl.DataFrame(schema=FACTOR_VALUE_SCHEMA)
        return (
            pl.scan_parquet([str(path) for path in self.paths], extra_columns="ignore")
            .select(list(FACTOR_VALUE_SCHEMA))
            .collect(engine="streaming")
        )


def compute_factor_factory_result(
    snapshot_root: str | Path,
    manifest: FactorFactorySnapshotManifest,
    task: FactorFactoryTask,
    stage_callback: Callable[[str], None] | None = None,
    max_input_uncompressed_bytes: int = (DEFAULT_FACTOR_FACTORY_MAX_INPUT_UNCOMPRESSED_BYTES),
    work_dir: str | Path | None = None,
) -> FactorFactoryComputeArtifacts:
    """Execute the immutable plan with bounded, staged Factor Value computation."""

    started = time.perf_counter()
    _validate_task_manifest_binding(task, manifest)
    root = Path(snapshot_root) / "files"
    estimated_uncompressed_bytes = (
        manifest.estimated_uncompressed_input_bytes
        if manifest.estimated_uncompressed_input_bytes is not None
        else manifest.estimated_uncompressed_bytes
    )
    if manifest.schema_version == FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1:
        estimated_uncompressed_bytes = sum(
            _parquet_uncompressed_bytes(root / reference.relative_path)
            for reference in manifest.files
        )
    if estimated_uncompressed_bytes > max_input_uncompressed_bytes:
        raise ValueError("factor_factory_input_uncompressed_size_limit_exceeded")

    stage_root = _prepare_stage_root(snapshot_root, task.task_id, work_dir=work_dir)
    generated_at = datetime.now(UTC)
    specs = manifest.factor_plan.factor_spec_models()
    available_feature_names = tuple(
        sorted({name for spec in specs for name in spec.input_features})
    )
    definitions = build_factor_definition_frame(specs, created_at=generated_at)
    factor_ids = tuple(sorted(item.factor_id for item in manifest.factor_plan.factor_specs))
    stage_durations: dict[str, float] = {}
    stage_release_events: list[str] = []
    warnings: list[str] = []
    current_stage: str | None = None
    current_stage_started = time.perf_counter()
    peak_rss_bytes = _peak_rss_bytes()
    temporary_disk_peak_bytes = 0

    def record_stage(stage: str) -> None:
        nonlocal current_stage, current_stage_started, peak_rss_bytes
        now = time.perf_counter()
        if current_stage is not None:
            stage_durations[current_stage] = stage_durations.get(current_stage, 0.0) + (
                now - current_stage_started
            )
        current_stage = stage
        current_stage_started = now
        peak_rss_bytes = max(peak_rss_bytes, _peak_rss_bytes())
        if stage_callback is not None:
            stage_callback(stage)

    violations = _empty_streaming_violations()
    value_paths: list[Path] = []
    value_rows = 0
    value_uncompressed_bytes = 0
    label_partition_count = 0
    no_update_reason: str | None = None
    evidence = pl.DataFrame(schema=FACTOR_EVIDENCE_SCHEMA)
    correlations = pl.DataFrame(schema=FACTOR_CORRELATION_SCHEMA)
    costs = pl.DataFrame()
    try:
        feature_paths = _snapshot_dataset_paths(
            root, manifest, _dataset_name(FEATURE_VALUE_DATASET)
        )
        market_paths = _snapshot_dataset_paths(root, manifest, _dataset_name(MARKET_BAR_DATASET))
        cost_paths = _snapshot_dataset_paths(
            root, manifest, _dataset_name(COST_BUCKET_DAILY_DATASET)
        )
        if not feature_paths:
            no_update_reason = "feature_value_missing_or_empty"
            warnings.append("feature_value missing or empty for factor factory")
        else:
            record_stage("computing_values")
            feature_scan = _scan_paths(feature_paths)
            feature_dates = (
                feature_scan.select(_utc_expr("ts").dt.date().alias("partition_date"))
                .unique()
                .sort("partition_date")
                .collect(engine="streaming")
                .get_column("partition_date")
                .drop_nulls()
                .to_list()
            )
            input_dataset_version = _lazy_text_mode(
                feature_scan, "input_dataset_version", "feature_value:unknown"
            )
            input_hash = _lazy_text_mode(feature_scan, "input_hash", "sha256:unknown")
            for partition_date in feature_dates:
                feature_frame = _collect_utc_date(feature_scan, partition_date)
                _accumulate_feature_violations(
                    violations,
                    feature_frame,
                    manifest=manifest,
                    task=task,
                )
                value_frame = build_factor_value_frame(
                    feature_frame,
                    specs,
                    created_at=generated_at,
                    input_dataset_version=input_dataset_version,
                    input_hash=input_hash,
                    available_feature_names=available_feature_names,
                )
                _accumulate_value_violations(
                    violations,
                    value_frame,
                    feature_frame,
                    manifest=manifest,
                )
                if not value_frame.is_empty():
                    value_path = (
                        stage_root
                        / "factor-values"
                        / f"date={partition_date.isoformat()}"
                        / "part-00000.parquet"
                    )
                    value_path.parent.mkdir(parents=True, exist_ok=True)
                    value_frame.write_parquet(value_path, compression="zstd")
                    value_paths.append(value_path)
                    value_rows += value_frame.height
                    value_uncompressed_bytes += _parquet_uncompressed_bytes(value_path)
                del value_frame, feature_frame
                gc.collect()
                stage_release_events.append(f"feature_partition_released:{partition_date}")
                peak_rss_bytes = max(peak_rss_bytes, _peak_rss_bytes())
                temporary_disk_peak_bytes = max(
                    temporary_disk_peak_bytes, _directory_size_bytes(stage_root)
                )
            del feature_scan
            gc.collect()
            stage_release_events.append("feature_scan_released")
            if value_rows == 0:
                no_update_reason = "factor_value_empty"
                warnings.append("no factor values computed")

        staged_values = StagedFactorValueSet(
            paths=tuple(value_paths),
            row_count=value_rows,
            uncompressed_bytes=value_uncompressed_bytes,
        )
        if no_update_reason is None:
            if not market_paths:
                no_update_reason = "market_bar_missing_or_empty"
                warnings.append("market_bar missing or empty for factor evidence")
            else:
                record_stage("computing_labels")
                label_paths, label_partition_count = _stage_forward_labels(
                    market_paths,
                    stage_root=stage_root,
                    task=task,
                    manifest=manifest,
                    violations=violations,
                    horizons=task.horizon_bars,
                    release_events=stage_release_events,
                )
                temporary_disk_peak_bytes = max(
                    temporary_disk_peak_bytes, _directory_size_bytes(stage_root)
                )
                costs = _read_paths(cost_paths)
                cost_frame = build_latest_symbol_cost_frame(
                    costs,
                    cost_quantile=task.cost_quantile,
                    warnings=warnings,
                )
                record_stage("computing_evidence")
                evidence_frames: list[pl.DataFrame] = []
                value_scan = _scan_paths(value_paths)
                for horizon in task.horizon_bars:
                    horizon_paths = label_paths.get(horizon, ())
                    if not horizon_paths:
                        warnings.append(f"horizon_{horizon}_labels_empty")
                        continue
                    labels = _scan_paths(list(horizon_paths)).collect(engine="streaming")
                    for factor_id in factor_ids:
                        factor_values = value_scan.filter(pl.col("factor_id") == factor_id).collect(
                            engine="streaming"
                        )
                        if factor_values.is_empty():
                            continue
                        frame = build_factor_evidence_from_labels(
                            factor_values,
                            labels,
                            cost_frame,
                            as_of_date=task.as_of_date,
                            horizon_bars=horizon,
                            decision_delay_bars=task.decision_delay_bars,
                            min_samples=task.min_samples,
                            top_quantile=task.top_quantile,
                            created_at=generated_at,
                            decision_policy=LEGACY_MAIN_DECISION_POLICY,
                        )
                        if not frame.is_empty():
                            evidence_frames.append(frame)
                        del factor_values, frame
                    del labels
                    gc.collect()
                    stage_release_events.append(f"label_horizon_released:{horizon}")
                evidence = (
                    pl.concat(evidence_frames, how="vertical_relaxed")
                    .select(list(FACTOR_EVIDENCE_SCHEMA))
                    .sort(
                        [
                            "as_of_date",
                            "factor_id",
                            "factor_version",
                            "timeframe",
                            "horizon_bars",
                        ]
                    )
                    if evidence_frames
                    else pl.DataFrame(schema=FACTOR_EVIDENCE_SCHEMA)
                )
                del evidence_frames, value_scan, cost_frame
                gc.collect()
                stage_release_events.append("evidence_inputs_released")

                record_stage("computing_correlation")
                bounded_values = (
                    _scan_paths(value_paths)
                    .filter(pl.col("is_valid") & pl.col("value").is_not_null())
                    .sort("ts", descending=True)
                    .head(250_000)
                    .collect(engine="streaming")
                )
                correlations = build_factor_correlation_frame(
                    bounded_values,
                    as_of_date=task.as_of_date,
                    factor_version=task.factor_version,
                    timeframe=task.timeframe,
                    created_at=generated_at,
                )
                del bounded_values
                gc.collect()
                stage_release_events.append("correlation_input_released")

        pure_for_validation = FactorFactoryPureResult(
            generated_at=generated_at,
            definitions=definitions,
            values=pl.DataFrame(schema=FACTOR_VALUE_SCHEMA),
            evidence=evidence,
            correlations=correlations,
            warnings=tuple(warnings),
            no_update_reason=no_update_reason,
        )
        anti_started = time.perf_counter()
        anti_leakage = _anti_leakage_report_staged(
            manifest=manifest,
            task=task,
            costs=costs,
            result=pure_for_validation,
            staged_violations=violations,
        )
        if anti_leakage["status"] != "PASS" or anti_leakage["violation_count"] != 0:
            failed = ",".join(
                item["check_name"]
                for item in anti_leakage["checks"]
                if item["status"] != "PASS"
            )
            raise ValueError(f"factor_factory_anti_leakage_not_pass:{failed}")
        stage_durations["anti_leakage"] = time.perf_counter() - anti_started
        now = time.perf_counter()
        if current_stage is not None:
            stage_durations[current_stage] = stage_durations.get(current_stage, 0.0) + (
                now - current_stage_started
            )
        peak_rss_bytes = max(peak_rss_bytes, _peak_rss_bytes())
        temporary_disk_peak_bytes = max(
            temporary_disk_peak_bytes, _directory_size_bytes(stage_root)
        )
        worker_report = {
            "schema_version": "quant_lab.factor_factory_worker_report.v2",
            "task_id": task.task_id,
            "snapshot_id": task.snapshot_id,
            "factor_plan_digest": task.factor_plan_digest,
            "source_input_digest": task.source_input_digest,
            "cost_input_digest": task.cost_input_digest,
            "previous_generation_id": task.previous_generation_id,
            "previous_generation_digest": task.previous_generation_digest,
            "result_mode": task.result_mode,
            "history_mode": task.history_mode,
            "min_samples": task.min_samples,
            "top_quantile": task.top_quantile,
            "cost_quantile": task.cost_quantile,
            "factor_count": len(factor_ids),
            "factor_ids": list(factor_ids),
            "output_rows": {
                "factor_definition_preview": definitions.height,
                "factor_value": staged_values.height,
                "factor_evidence": evidence.height,
                "factor_correlation_daily": correlations.height,
            },
            "compressed_input_bytes": manifest.total_input_bytes,
            "estimated_uncompressed_input_bytes": estimated_uncompressed_bytes,
            "streaming_enabled": True,
            "factor_value_stage_partition_count": len(value_paths),
            "label_stage_partition_count": label_partition_count,
            "stage_release_events": stage_release_events,
            "peak_rss_bytes_observed": peak_rss_bytes,
            "temporary_disk_peak_bytes": temporary_disk_peak_bytes,
            "completed_no_update": no_update_reason is not None,
            "no_update_reason": no_update_reason,
            "compute_duration_seconds": time.perf_counter() - started,
            "stage_durations_seconds": {
                name: round(value, 6) for name, value in sorted(stage_durations.items())
            },
            "diagnostic_only": True,
            "research_only": True,
            "live_order_effect": "none_read_only_research",
            "automatic_promotion": False,
            "max_live_notional_usdt": 0,
        }
        return FactorFactoryComputeArtifacts(
            generated_at=generated_at,
            definitions=definitions,
            values=staged_values,
            evidence=evidence,
            correlations=correlations,
            anti_leakage=anti_leakage,
            worker_report=worker_report,
            warnings=tuple(dict.fromkeys(warnings)),
            no_update_reason=no_update_reason,
            factor_ids=factor_ids,
        )
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


STREAMING_PRECOMPUTED_CHECKS = (
    "feature_scope_matches_plan",
    "feature_timestamps_within_signed_bounds",
    "feature_primary_keys_are_unique",
    "feature_rows_are_valid_only",
    "market_timeframe_matches_task",
    "market_rows_are_closed_only",
    "market_timestamps_within_signed_bounds",
    "market_primary_keys_are_unique",
    "factor_values_use_planned_factor_ids_only",
    "factor_value_primary_keys_are_unique",
    "factor_available_time_not_before_event_time",
    "factor_available_time_matches_plan_lag",
    "factor_event_time_matches_ts",
    "factor_value_keys_exist_in_snapshot_features",
    "factor_value_timestamps_within_feature_bounds",
)


def _prepare_stage_root(
    snapshot_root: str | Path,
    task_id: str,
    *,
    work_dir: str | Path | None,
) -> Path:
    if work_dir is None:
        base = Path(snapshot_root).parent.parent / "work" / task_id
        stage_root = base / "factor-factory-stage"
    else:
        base = Path(work_dir)
        stage_root = base / "factor-factory-stage"
    base.mkdir(parents=True, exist_ok=True)
    if stage_root.is_symlink():
        raise ValueError("factor_factory_stage_path_symlink")
    resolved_base = base.resolve(strict=True)
    resolved_stage = stage_root.resolve(strict=False)
    if resolved_base not in resolved_stage.parents:
        raise ValueError("factor_factory_stage_path_escape")
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=False)
    return stage_root


def _snapshot_dataset_paths(
    root: Path,
    manifest: FactorFactorySnapshotManifest,
    dataset_name: str,
) -> list[Path]:
    return sorted(
        root / reference.relative_path
        for reference in manifest.files
        if reference.dataset_name == dataset_name
    )


def _scan_paths(paths: list[Path]) -> pl.LazyFrame:
    if not paths:
        raise ValueError("factor_factory_staged_paths_empty")
    return pl.scan_parquet(
        [str(path) for path in paths],
        extra_columns="ignore",
        hive_partitioning=False,
    )


def _read_paths(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    return _scan_paths(paths).collect(engine="streaming")


def _utc_expr(column: str) -> pl.Expr:
    return pl.coalesce(
        pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
        pl.col(column).cast(pl.Utf8, strict=False).str.to_datetime(
            time_zone="UTC", strict=False
        ),
    )


def _collect_utc_date(lazy: pl.LazyFrame, partition_date: date) -> pl.DataFrame:
    lower = datetime.combine(partition_date, datetime_time.min, tzinfo=UTC)
    upper = lower + timedelta(days=1)
    return (
        lazy.filter((_utc_expr("ts") >= lower) & (_utc_expr("ts") < upper))
        .with_columns(_utc_expr("ts").alias("ts"))
        .collect(engine="streaming")
    )


def _lazy_text_mode(lazy: pl.LazyFrame, column: str, fallback: str) -> str:
    if column not in lazy.collect_schema().names():
        return fallback
    modes = (
        lazy.select(pl.col(column).cast(pl.Utf8, strict=False).alias(column))
        .filter(pl.col(column).is_not_null() & (pl.col(column).str.len_chars() > 0))
        .group_by(column)
        .len()
        .sort(["len", column], descending=[True, False])
        .head(1)
        .collect(engine="streaming")
    )
    return str(modes.item(0, column)) if modes.height else fallback


def _empty_streaming_violations() -> dict[str, int]:
    return {name: 0 for name in STREAMING_PRECOMPUTED_CHECKS}


def _accumulate_feature_violations(
    counts: dict[str, int],
    features: pl.DataFrame,
    *,
    manifest: FactorFactorySnapshotManifest,
    task: FactorFactoryTask,
) -> None:
    counts["feature_scope_matches_plan"] += _filter_count(
        features,
        (pl.col("feature_set") != task.feature_set)
        | (pl.col("feature_version") != task.feature_version)
        | (pl.col("timeframe") != task.timeframe),
    )
    counts["feature_timestamps_within_signed_bounds"] += _frame_bound_violations(
        features,
        "ts",
        lower=manifest.feature_min_ts,
        upper=manifest.feature_max_ts,
    )
    counts["feature_primary_keys_are_unique"] += _duplicate_count(
        features,
        ["feature_set", "feature_name", "feature_version", "symbol", "timeframe", "ts"],
    )
    counts["feature_rows_are_valid_only"] += _false_count(features, "is_valid")


def _accumulate_value_violations(
    counts: dict[str, int],
    values: pl.DataFrame,
    features: pl.DataFrame,
    *,
    manifest: FactorFactorySnapshotManifest,
) -> None:
    planned_ids = {item.factor_id for item in manifest.factor_plan.factor_specs}
    counts["factor_values_use_planned_factor_ids_only"] += _membership_violation_count(
        values, "factor_id", planned_ids
    )
    counts["factor_value_primary_keys_are_unique"] += _duplicate_count(
        values,
        ["factor_id", "factor_version", "symbol", "timeframe", "ts"],
    )
    counts["factor_available_time_not_before_event_time"] += _filter_count(
        values, pl.col("available_time") < pl.col("event_time")
    )
    counts["factor_available_time_matches_plan_lag"] += _availability_lag_violations(
        values, manifest
    )
    counts["factor_event_time_matches_ts"] += _filter_count(
        values, pl.col("event_time") != pl.col("ts")
    )
    counts["factor_value_keys_exist_in_snapshot_features"] += (
        _factor_value_feature_key_violations(values, features)
    )
    counts["factor_value_timestamps_within_feature_bounds"] += _result_bound_violations(
        values, manifest
    )


def _stage_forward_labels(
    market_paths: list[Path],
    *,
    stage_root: Path,
    task: FactorFactoryTask,
    manifest: FactorFactorySnapshotManifest,
    violations: dict[str, int],
    horizons: tuple[int, ...],
    release_events: list[str],
) -> tuple[dict[int, tuple[Path, ...]], int]:
    market_scan = _scan_paths(market_paths).with_columns(_utc_expr("ts").alias("ts"))
    symbols = (
        market_scan.select(pl.col("symbol").cast(pl.Utf8))
        .unique()
        .sort("symbol")
        .collect(engine="streaming")
        .get_column("symbol")
        .drop_nulls()
        .to_list()
    )
    label_paths: dict[int, list[Path]] = {horizon: [] for horizon in horizons}
    count = 0
    for symbol in symbols:
        market = (
            market_scan.filter(pl.col("symbol") == symbol)
            .sort(["symbol", "timeframe", "ts"])
            .collect(engine="streaming")
        )
        violations["market_timeframe_matches_task"] += _filter_count(
            market, pl.col("timeframe") != task.timeframe
        )
        violations["market_rows_are_closed_only"] += _false_count(market, "is_closed")
        violations["market_timestamps_within_signed_bounds"] += _frame_bound_violations(
            market,
            "ts",
            lower=manifest.market_min_ts,
            upper=manifest.market_max_ts,
        )
        violations["market_primary_keys_are_unique"] += _duplicate_count(
            market, ["symbol", "timeframe", "ts"]
        )
        symbol_key = hashlib.sha256(str(symbol).encode("utf-8")).hexdigest()[:16]
        for horizon in horizons:
            labels = build_forward_return_labels(
                market,
                horizon_bars=horizon,
                decision_delay_bars=task.decision_delay_bars,
            )
            if labels.is_empty():
                continue
            lookahead_violations = labels.filter(
                (pl.col("decision_ts") <= pl.col("feature_ts"))
                | (pl.col("label_ts") <= pl.col("decision_ts"))
            ).height
            if lookahead_violations:
                raise ValueError("factor_factory_label_lookahead_detected")
            path = (
                stage_root
                / "forward-labels"
                / f"horizon={horizon}"
                / f"symbol-{symbol_key}.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            labels.write_parquet(path, compression="zstd")
            label_paths[horizon].append(path)
            count += 1
            del labels
        del market
        gc.collect()
        release_events.append(f"market_symbol_released:{symbol_key}")
    del market_scan
    gc.collect()
    release_events.append("market_scan_released")
    return {key: tuple(value) for key, value in label_paths.items()}, count


def _anti_leakage_report_staged(
    *,
    manifest: FactorFactorySnapshotManifest,
    task: FactorFactoryTask,
    costs: pl.DataFrame,
    result: FactorFactoryPureResult,
    staged_violations: dict[str, int],
) -> dict[str, Any]:
    report = _anti_leakage_report(
        manifest=manifest,
        task=task,
        features=pl.DataFrame(),
        market=pl.DataFrame(),
        costs=costs,
        result=result,
    )
    checks: list[dict[str, Any]] = []
    for item in report["checks"]:
        row = dict(item)
        if row["check_name"] in staged_violations:
            count = staged_violations[row["check_name"]]
            row["violation_count"] = count
            row["status"] = "PASS" if count == 0 else "FAIL"
        checks.append(row)
    violations = sum(int(item["violation_count"]) for item in checks)
    return {
        **report,
        "checks": checks,
        "status": "PASS" if violations == 0 else "FAIL",
        "violation_count": violations,
    }


def _peak_rss_bytes() -> int:
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        value = 0
    else:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if value > 0:
            return value if value > 10_000_000 else value * 1024
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        try:
            import ctypes  # noqa: PLC0415
            from ctypes import wintypes  # noqa: PLC0415

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_memory_info.restype = wintypes.BOOL
            handle = get_current_process()
            if get_memory_info(
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError):
            pass
        return value
    return int(psutil.Process().memory_info().rss)


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file() and not candidate.is_symlink():
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return total


def _dataset_name(path: Path) -> str:
    return str(path).replace("\\", "/")


def _validate_task_manifest_binding(
    task: FactorFactoryTask,
    manifest: FactorFactorySnapshotManifest,
) -> None:
    pairs = {
        "snapshot_id": (task.snapshot_id, manifest.snapshot_id),
        "quant_lab_commit": (task.quant_lab_commit, manifest.quant_lab_commit),
        "snapshot_manifest_sha256": (
            task.snapshot_manifest_sha256,
            manifest.manifest_sha256,
        ),
        "factor_plan_digest": (task.factor_plan_digest, manifest.factor_plan_digest),
        "source_input_digest": (task.source_input_digest, manifest.source_input_digest),
        "cost_input_digest": (task.cost_input_digest, manifest.cost_input_digest),
    }
    mismatch = [name for name, values in pairs.items() if values[0] != values[1]]
    if mismatch:
        raise ValueError(f"factor_factory_task_snapshot_mismatch:{','.join(mismatch)}")
    if task.parameters.model_dump(exclude={"as_of_date"}) != manifest_parameters(manifest):
        raise ValueError("factor_factory_task_snapshot_parameter_mismatch")
    if manifest.schema_version == FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1 and (
        task.as_of_date != manifest.as_of_date
        or task.previous_generation_id != manifest.previous_generation_id
        or task.previous_generation_digest != manifest.previous_generation_digest
    ):
        raise ValueError("factor_factory_task_snapshot_legacy_binding_mismatch")


def manifest_parameters(manifest: FactorFactorySnapshotManifest) -> dict[str, Any]:
    return {
        "feature_set": manifest.feature_set,
        "feature_version": manifest.feature_version,
        "factor_version": manifest.factor_version,
        "timeframe": manifest.timeframe,
        "horizon_bars": manifest.horizon_bars,
        "decision_delay_bars": manifest.decision_delay_bars,
        "max_factors": manifest.max_factors,
        "min_samples": manifest.min_samples,
        "top_quantile": manifest.top_quantile,
        "cost_quantile": manifest.cost_quantile,
        "result_mode": manifest.result_mode,
        "history_mode": manifest.history_mode,
    }


def _parquet_uncompressed_bytes(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(row_group).column(column).total_uncompressed_size
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _anti_leakage_report(
    *,
    manifest: FactorFactorySnapshotManifest,
    task: FactorFactoryTask,
    features: pl.DataFrame,
    market: pl.DataFrame,
    costs: pl.DataFrame,
    result: FactorFactoryPureResult,
) -> dict[str, Any]:
    planned_ids = {item.factor_id for item in manifest.factor_plan.factor_specs}
    expected_datasets = {
        "gold/feature_value",
        "silver/market_bar",
        "gold/cost_bucket_daily",
    }
    checks: dict[str, tuple[int, str]] = {
        "task_snapshot_identity_matches": (
            int(task.snapshot_id != manifest.snapshot_id),
            "task and snapshot ids are identical",
        ),
        "task_snapshot_commit_matches": (
            int(task.quant_lab_commit != manifest.quant_lab_commit),
            "task, snapshot, and worker code identity are bound",
        ),
        "snapshot_manifest_digest_matches_task": (
            int(task.snapshot_manifest_sha256 != manifest.manifest_sha256),
            "task references the sealed snapshot manifest",
        ),
        "factor_plan_digest_matches": (
            int(task.factor_plan_digest != manifest.factor_plan.plan_digest),
            "embedded plan digest is immutable",
        ),
        "factor_plan_scope_matches_task": (
            int(
                manifest.factor_plan.feature_set != task.feature_set
                or manifest.factor_plan.feature_version != task.feature_version
                or manifest.factor_plan.factor_version != task.factor_version
                or manifest.factor_plan.timeframe != task.timeframe
            ),
            "plan scope exactly matches task scope",
        ),
        "factor_plan_membership_is_frozen": (
            int(manifest.factor_plan.factor_count != len(planned_ids)),
            "NAS consumed only the embedded factor membership",
        ),
        "source_input_digest_matches": (
            int(task.source_input_digest != manifest.source_input_digest),
            "feature and market input identity is signed",
        ),
        "cost_input_digest_matches": (
            int(task.cost_input_digest != manifest.cost_input_digest),
            "cost input identity is signed",
        ),
        "previous_generation_binding_matches": (
            int(
                (
                    manifest.schema_version == FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1
                    and (
                        task.previous_generation_id != manifest.previous_generation_id
                        or task.previous_generation_digest != manifest.previous_generation_digest
                    )
                )
                or (
                    manifest.schema_version != FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1
                    and any(
                        value is not None
                        for value in (
                            manifest.previous_generation_id,
                            manifest.previous_generation_digest,
                            manifest.previous_generation_manifest,
                        )
                    )
                )
            ),
            "v1 previous Gold matches Task; v2 keeps previous Gold task-only",
        ),
        "result_mode_is_parity_full": (
            int(task.result_mode != "PARITY_FULL"),
            "full parity result mode is mandatory",
        ),
        "history_mode_is_bootstrap_full": (
            int(task.history_mode != "bootstrap_full"),
            "full-history bootstrap mode is mandatory",
        ),
        "snapshot_dataset_allowlist_exact": (
            int(set(manifest.datasets) != expected_datasets),
            "only feature, market, and cost datasets are available",
        ),
        "snapshot_files_are_manifest_bound": (
            int(len({item.relative_path for item in manifest.files}) != len(manifest.files)),
            "all input files have unique signed manifest paths",
        ),
        "feature_scope_matches_plan": (
            _filter_count(
                features,
                (pl.col("feature_set") != task.feature_set)
                | (pl.col("feature_version") != task.feature_version)
                | (pl.col("timeframe") != task.timeframe),
            ),
            "every feature row is in the signed plan scope",
        ),
        "feature_timestamps_within_signed_bounds": (
            _frame_bound_violations(
                features,
                "ts",
                lower=manifest.feature_min_ts,
                upper=manifest.feature_max_ts,
            ),
            "feature timestamps stay inside the signed full-history bounds",
        ),
        "feature_primary_keys_are_unique": (
            _duplicate_count(
                features,
                ["feature_set", "feature_name", "feature_version", "symbol", "timeframe", "ts"],
            ),
            "feature primary keys are unique",
        ),
        "feature_rows_are_valid_only": (
            _false_count(features, "is_valid"),
            "snapshot contains valid feature rows only",
        ),
        "market_timeframe_matches_task": (
            _filter_count(market, pl.col("timeframe") != task.timeframe),
            "market rows match task timeframe",
        ),
        "market_rows_are_closed_only": (
            _false_count(market, "is_closed"),
            "market rows are closed bars only",
        ),
        "market_timestamps_within_signed_bounds": (
            _frame_bound_violations(
                market,
                "ts",
                lower=manifest.market_min_ts,
                upper=manifest.market_max_ts,
            ),
            "market timestamps stay inside the signed label boundary",
        ),
        "market_primary_keys_are_unique": (
            _duplicate_count(market, ["symbol", "timeframe", "ts"]),
            "market primary keys are unique",
        ),
        "cost_symbol_selection_is_unique": (
            _duplicate_count(costs, ["symbol"]),
            "cloud fixed exactly one cost row per symbol",
        ),
        "cost_rows_match_signed_snapshot": (
            _cost_snapshot_mismatch(costs, manifest, task.cost_quantile),
            "latest-per-symbol cost rows exactly match signed snapshot metadata",
        ),
        "decision_delay_is_positive": (
            int(task.decision_delay_bars < 1),
            "decision delay is at least one completed bar",
        ),
        "horizons_are_positive_sorted_unique": (
            int(
                task.horizon_bars != tuple(sorted(set(task.horizon_bars)))
                or any(value <= 0 for value in task.horizon_bars)
            ),
            "forward horizons are positive, sorted, and unique",
        ),
        "factor_values_use_planned_factor_ids_only": (
            _membership_violation_count(result.values, "factor_id", planned_ids),
            "computed values contain no NAS-discovered factor",
        ),
        "factor_value_primary_keys_are_unique": (
            _duplicate_count(
                result.values,
                ["factor_id", "factor_version", "symbol", "timeframe", "ts"],
            ),
            "factor value primary keys are unique",
        ),
        "factor_available_time_not_before_event_time": (
            _filter_count(result.values, pl.col("available_time") < pl.col("event_time")),
            "availability never precedes event time",
        ),
        "factor_available_time_matches_plan_lag": (
            _availability_lag_violations(result.values, manifest),
            "availability is exactly event time plus the planned lag",
        ),
        "factor_event_time_matches_ts": (
            _filter_count(result.values, pl.col("event_time") != pl.col("ts")),
            "factor event_time exactly equals its feature timestamp",
        ),
        "factor_value_keys_exist_in_snapshot_features": (
            _factor_value_feature_key_violations(result.values, features),
            "every factor value symbol, timeframe, and timestamp exists in signed features",
        ),
        "factor_value_timestamps_within_feature_bounds": (
            _result_bound_violations(result.values, manifest),
            "factor values stay within signed feature bounds",
        ),
        "evidence_uses_requested_horizons_only": (
            _membership_violation_count(
                result.evidence,
                "horizon_bars",
                set(task.horizon_bars),
            ),
            "evidence uses only requested label horizons",
        ),
        "evidence_decisions_are_research_only": (
            _membership_violation_count(
                result.evidence,
                "decision",
                FACTOR_FACTORY_ALLOWED_DECISIONS,
            ),
            "decisions are limited to non-live research states",
        ),
        "evidence_primary_keys_are_unique": (
            _duplicate_count(
                result.evidence,
                [
                    "as_of_date",
                    "factor_id",
                    "factor_version",
                    "timeframe",
                    "horizon_bars",
                    "decision_delay_bars",
                ],
            ),
            "evidence primary keys are unique",
        ),
        "correlation_primary_keys_are_unique": (
            _duplicate_count(
                result.correlations,
                [
                    "as_of_date",
                    "factor_id_left",
                    "factor_id_right",
                    "factor_version",
                    "timeframe",
                ],
            ),
            "correlation primary keys are unique",
        ),
        "nas_does_not_derive_factor_candidates": (
            0,
            "candidate derivation is reserved for cloud publication",
        ),
        "automatic_promotion_is_disabled": (
            int(task.automatic_promotion is not False),
            "automatic promotion is hard-disabled",
        ),
        "live_notional_is_zero": (
            int(task.max_live_notional_usdt != 0),
            "live notional is exactly zero",
        ),
        "live_order_effect_is_none_read_only_research": (
            int(task.live_order_effect != "none_read_only_research"),
            "task has no live order effect",
        ),
    }
    if tuple(checks) != FACTOR_FACTORY_ANTI_LEAKAGE_CHECKS:
        raise RuntimeError("factor_factory_anti_leakage_check_registry_mismatch")
    rows = [
        {
            "check_name": name,
            "status": "PASS" if count == 0 else "FAIL",
            "violation_count": count,
            "detail": detail,
        }
        for name, (count, detail) in checks.items()
    ]
    violations = sum(item["violation_count"] for item in rows)
    return {
        "schema_version": "quant_lab.factor_factory_anti_leakage.v1",
        "task_id": task.task_id,
        "snapshot_id": task.snapshot_id,
        "factor_plan_digest": task.factor_plan_digest,
        "status": "PASS" if violations == 0 else "FAIL",
        "violation_count": violations,
        "checks": rows,
        "diagnostic_only": True,
        "research_only": True,
        "live_order_effect": "none_read_only_research",
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
    }


def _filter_count(frame: pl.DataFrame, predicate: pl.Expr) -> int:
    if frame.is_empty():
        return 0
    try:
        return frame.filter(predicate.fill_null(True)).height
    except pl.exceptions.ColumnNotFoundError:
        return frame.height


def _false_count(frame: pl.DataFrame, column: str) -> int:
    if frame.is_empty():
        return 0
    if column not in frame.columns:
        return frame.height
    return frame.filter(~pl.col(column).cast(pl.Boolean, strict=False).fill_null(False)).height


def _duplicate_count(frame: pl.DataFrame, columns: list[str]) -> int:
    if frame.is_empty():
        return 0
    if not set(columns).issubset(frame.columns):
        return frame.height
    return frame.height - frame.unique(subset=columns).height


def _frame_bound_violations(
    frame: pl.DataFrame,
    column: str,
    *,
    lower: datetime | None,
    upper: datetime | None,
) -> int:
    if frame.is_empty():
        return 0
    if column not in frame.columns:
        return frame.height
    timestamp = pl.coalesce(
        pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
        pl.col(column).cast(pl.Utf8, strict=False).str.to_datetime(time_zone="UTC", strict=False),
    )
    if lower is None or upper is None:
        return frame.height
    return frame.filter(timestamp.is_null() | (timestamp < lower) | (timestamp > upper)).height


def _membership_violation_count(
    frame: pl.DataFrame,
    column: str,
    allowed: set[Any] | frozenset[Any],
) -> int:
    if frame.is_empty():
        return 0
    if column not in frame.columns:
        return frame.height
    return frame.filter(pl.col(column).is_null() | ~pl.col(column).is_in(sorted(allowed))).height


def _cost_snapshot_mismatch(
    frame: pl.DataFrame,
    manifest: FactorFactorySnapshotManifest,
    cost_quantile: str,
) -> int:
    if frame.is_empty():
        return int(bool(manifest.cost_snapshot))
    column = f"total_cost_bps_{cost_quantile}"
    if column not in frame.columns or "symbol" not in frame.columns:
        return frame.height
    date_column = "day" if "day" in frame.columns else "as_of_date"
    source_column = "cost_source" if "cost_source" in frame.columns else "source"
    observed = sorted(
        (
            str(row["symbol"]),
            str(row.get(date_column)) if row.get(date_column) is not None else None,
            str(row.get("cost_model_version") or "unknown"),
            str(row.get(source_column) or "unknown"),
            float(row[column]),
        )
        for row in frame.to_dicts()
    )
    expected = sorted(
        (
            item.symbol,
            item.cost_date,
            item.cost_model_version,
            item.cost_source,
            item.cost_bps,
        )
        for item in manifest.cost_snapshot
    )
    return int(observed != expected)


def _availability_lag_violations(
    values: pl.DataFrame,
    manifest: FactorFactorySnapshotManifest,
) -> int:
    if values.is_empty():
        return 0
    lag_rows = [
        {
            "factor_id": item.factor_id,
            "_expected_lag_seconds": (
                _timeframe_seconds(item.timeframe) * item.availability_lag_bars
            ),
        }
        for item in manifest.factor_plan.factor_specs
    ]
    if not lag_rows:
        return values.height
    checked = values.join(pl.DataFrame(lag_rows), on="factor_id", how="left")
    actual_seconds = (pl.col("available_time") - pl.col("event_time")).dt.total_seconds()
    return checked.filter(
        pl.col("_expected_lag_seconds").is_null()
        | actual_seconds.is_null()
        | (actual_seconds != pl.col("_expected_lag_seconds"))
    ).height


def _factor_value_feature_key_violations(
    values: pl.DataFrame,
    features: pl.DataFrame,
) -> int:
    if values.is_empty():
        return 0
    keys = ["symbol", "timeframe", "ts"]
    if not set(keys).issubset(features.columns):
        return values.height
    available = features.select(keys).unique()
    return values.select(keys).unique().join(available, on=keys, how="anti").height


def _timeframe_seconds(timeframe: str) -> int:
    value = timeframe.strip().lower()
    if len(value) < 2 or not value[:-1].isdigit():
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    amount = int(value[:-1])
    unit = value[-1]
    if amount <= 0 or unit not in multipliers:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return amount * multipliers[unit]


def _result_bound_violations(
    values: pl.DataFrame,
    manifest: FactorFactorySnapshotManifest,
) -> int:
    if values.is_empty():
        return 0
    if manifest.feature_min_ts is None or manifest.feature_max_ts is None:
        return values.height
    timestamp = pl.col("ts").cast(pl.Datetime(time_zone="UTC"), strict=False)
    return values.filter(
        timestamp.is_null()
        | (timestamp < manifest.feature_min_ts)
        | (timestamp > manifest.feature_max_ts)
    ).height
