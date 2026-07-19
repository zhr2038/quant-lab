from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as datetime_time
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.alpha_factory.factory import (
    ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS,
    ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET,
    FACTOR_BRIDGE_REPORT_MEMBER,
    AlphaFactoryComputeArtifacts,
    alpha_factory_template_registry_digest,
    compute_alpha_factory,
)
from quant_lab.research.second_stage_alpha_factory import (
    RELATIVE_STRENGTH_CANDIDATES,
)
from quant_lab.research_plane.contracts import (
    AlphaFactorySnapshotManifest,
    AlphaFactoryTask,
)
from quant_lab.symbols import normalize_symbol

ALPHA_FACTORY_ANTI_LEAKAGE_CHECKS = (
    "relative_strength_rank_uses_prior_bars",
    "future_label_not_used_in_ranking",
    "decision_bar_is_actual_completed_bar",
    "label_horizon_is_complete_or_pending",
    "label_end_within_snapshot_boundary",
    "train_validation_chronological_70_30",
    "recent_7d_uses_trailing_window",
    "expanded_quality_not_from_future",
    "regime_not_from_future",
    "cost_bucket_not_from_future",
    "factor_forward_validation_uses_decision_delay",
    "factor_bridge_uses_same_snapshot",
    "expert_pack_cache_not_authoritative",
    "source_commit_and_registry_match",
    "read_only_no_live_action",
)
SAFE_READ_ONLY_LIVE_EFFECTS = ("none", "none_read_only_research")


@dataclass(frozen=True)
class AlphaFactoryWorkerComputeResult:
    artifacts: AlphaFactoryComputeArtifacts
    anti_leakage: dict[str, Any]
    worker_report: dict[str, Any]


def compute_alpha_factory_from_snapshot(
    snapshot_root: str | Path,
    manifest: AlphaFactorySnapshotManifest,
    task: AlphaFactoryTask,
) -> AlphaFactoryWorkerComputeResult:
    factor_binding_fields = (
        "factor_generation_id",
        "factor_generation_digest",
        "factor_generation_as_of_date",
        "factor_generation_published_at",
        "hypothesis_registry_digest",
        "trial_ledger_digest",
        "factor_generation_fresh",
        "factor_generation_hypothesis_ids",
    )
    task_binding = tuple(getattr(task, field) for field in factor_binding_fields)
    snapshot_binding = tuple(getattr(manifest, field) for field in factor_binding_fields)
    if task_binding != snapshot_binding:
        raise ValueError("alpha_factory_worker_factor_generation_binding_mismatch")
    if any(value is None for value in task_binding):
        raise ValueError("alpha_factory_worker_factor_generation_binding_missing")
    root = Path(snapshot_root) / "files"
    registry = read_parquet_dataset(root / ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET)
    if alpha_factory_template_registry_digest(registry) != task.template_registry_digest:
        raise ValueError("alpha_factory_worker_registry_digest_mismatch")
    started = time.perf_counter()
    generated_at = datetime.now(UTC)
    artifacts = compute_alpha_factory(
        root,
        as_of_date=task.as_of_date,
        lookback_days=task.lookback_days,
        max_candidates=task.max_candidates,
        registry=registry,
        generated_at=generated_at,
        factor_generation_as_of_date=task.factor_generation_as_of_date,
        factor_generation_hypothesis_ids=(
            task.factor_generation_hypothesis_ids or ()
        ),
        factor_generation_fresh=bool(task.factor_generation_fresh),
    )
    compute_seconds = time.perf_counter() - started
    frames = _normalize_artifact_frames(artifacts)
    anti_leakage = build_alpha_factory_anti_leakage_report(
        root,
        manifest=manifest,
        task=task,
        frames=frames,
        factor_bridge=artifacts.factor_strategy_bridge_candidates,
    )
    if anti_leakage["status"] != "PASS" or anti_leakage["violation_count"] != 0:
        failed = ",".join(
            str(check["check_name"])
            for check in anti_leakage["checks"]
            if check["status"] != "PASS"
        )
        raise ValueError(f"alpha_factory_anti_leakage_not_pass:{failed}")
    worker_report = {
        "schema_version": "quant_lab.alpha_factory_worker_report.v1",
        "task_id": task.task_id,
        "snapshot_id": task.snapshot_id,
        "quant_lab_commit": task.quant_lab_commit,
        "template_registry_digest": task.template_registry_digest,
        "factor_generation_id": task.factor_generation_id,
        "factor_generation_digest": task.factor_generation_digest,
        "factor_generation_as_of_date": task.factor_generation_as_of_date.isoformat(),
        "factor_generation_published_at": (
            task.factor_generation_published_at.isoformat()
        ),
        "hypothesis_registry_digest": task.hypothesis_registry_digest,
        "trial_ledger_digest": task.trial_ledger_digest,
        "factor_generation_fresh": task.factor_generation_fresh,
        "factor_generation_hypothesis_ids": list(
            task.factor_generation_hypothesis_ids or ()
        ),
        "as_of_date": task.as_of_date.isoformat(),
        "lookback_days": task.lookback_days,
        "max_candidates": task.max_candidates,
        "compute_duration_seconds": compute_seconds,
        "input_rows": {
            reference.dataset_name: sum(
                item.row_count
                for item in manifest.files
                if item.dataset_name == reference.dataset_name
            )
            for reference in manifest.files
        },
        "output_rows": {name: frame.height for name, frame in frames.items()},
        "decision_counts": _decision_counts(frames["alpha_factory_result"]),
        "warnings": list(artifacts.warnings),
        "research_only": True,
        "live_order_effect": "none",
        "automatic_promotion": False,
        "factor_bridge_source": "snapshot_recompute",
        "factor_bridge_report": FACTOR_BRIDGE_REPORT_MEMBER,
    }
    return AlphaFactoryWorkerComputeResult(
        artifacts=artifacts,
        anti_leakage=anti_leakage,
        worker_report=worker_report,
    )


def build_alpha_factory_anti_leakage_report(
    snapshot_lake_root: Path,
    *,
    manifest: AlphaFactorySnapshotManifest,
    task: AlphaFactoryTask,
    frames: dict[str, pl.DataFrame],
    factor_bridge: pl.DataFrame,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    samples = frames["second_stage_alpha_factory_sample"]
    decisions = frames["expanded_relative_strength_decision_sample"]
    candidates = frames["alpha_factory_candidate"]
    results = frames["alpha_factory_result"]

    completed_samples = _complete_samples(samples)
    relative_strength_samples = completed_samples.filter(
        pl.col("strategy_candidate").is_in(RELATIVE_STRENGTH_CANDIDATES)
    )
    leakage_rows = _count_invalid(
        relative_strength_samples,
        pl.col("decision_ts").is_null()
        | pl.col("label_ts").is_null()
        | (pl.col("label_ts") <= pl.col("decision_ts"))
        | (pl.col("anti_leakage_check").str.to_lowercase() != "pass"),
    )
    checks.append(_check("relative_strength_rank_uses_prior_bars", leakage_rows))

    invalid_future_rank = _count_invalid(
        decisions,
        pl.col("selection_reason").str.to_lowercase().str.contains("future|label"),
    )
    checks.append(_check("future_label_not_used_in_ranking", invalid_future_rank))

    market_bar = read_parquet_dataset(snapshot_lake_root / "silver" / "market_bar")
    actual_bar_keys = _market_bar_keys(
        market_bar,
        completed_before=manifest.generated_at,
    )
    missing_decision_bars = 0
    if not decisions.is_empty():
        for row in decisions.select("decision_ts", "symbol").unique().to_dicts():
            key = (normalize_symbol(row.get("symbol")), row.get("decision_ts"))
            if key not in actual_bar_keys:
                missing_decision_bars += 1
    checks.append(
        _check("decision_bar_is_actual_completed_bar", missing_decision_bars)
    )

    invalid_pending = _count_invalid(
        decisions,
        (pl.col("label_status").str.to_lowercase() == "pending")
        & pl.col("future_net_bps").is_not_null(),
    )
    invalid_complete = _count_invalid(
        decisions,
        (pl.col("label_status").str.to_lowercase() == "complete")
        & pl.col("future_net_bps").is_null(),
    )
    checks.append(
        _check(
            "label_horizon_is_complete_or_pending",
            invalid_pending + invalid_complete,
        )
    )

    market_bounds = [
        reference.max_ts
        for reference in manifest.files
        if reference.dataset_name == "silver/market_bar" and reference.max_ts is not None
    ]
    max_market_ts = max(market_bounds) if market_bounds else None
    labels_after_snapshot = 0
    if max_market_ts is not None and not completed_samples.is_empty():
        market_labeled_samples = completed_samples.filter(
            pl.col("source_type").fill_null("")
            != "second_stage_exit_policy_review"
        )
        labels_after_snapshot = _count_invalid(
            market_labeled_samples,
            pl.col("label_ts") > max_market_ts,
        )
    checks.append(_check("label_end_within_snapshot_boundary", labels_after_snapshot))

    metric_violations = _metric_window_violations(
        results,
        samples,
        as_of_date=task.as_of_date,
    )
    checks.append(
        _check("train_validation_chronological_70_30", metric_violations[0])
    )
    checks.append(_check("recent_7d_uses_trailing_window", metric_violations[1]))

    task_end = datetime.combine(
        task.as_of_date + timedelta(days=1),
        datetime_time.min,
        tzinfo=UTC,
    )
    checks.append(
        _check(
            "expanded_quality_not_from_future",
            _future_snapshot_reference_count(
                manifest,
                "gold/expanded_universe_quality",
                task_end,
            ),
        )
    )
    checks.append(
        _check(
            "regime_not_from_future",
            _future_snapshot_reference_count(
                manifest,
                "gold/market_regime_daily",
                task_end,
            ),
        )
    )
    checks.append(
        _check(
            "cost_bucket_not_from_future",
            _future_snapshot_reference_count(
                manifest,
                "gold/cost_bucket_daily",
                task_end,
            ),
        )
    )

    bridge_live_rows = _count_non_none_live_effect(factor_bridge)
    factor_values = read_parquet_dataset(snapshot_lake_root / "gold" / "factor_value")
    checks.append(
        _check(
            "factor_forward_validation_uses_decision_delay",
            _factor_decision_delay_violations(factor_values),
        )
    )
    checks.append(_check("factor_bridge_uses_same_snapshot", bridge_live_rows))
    checks.append(_check("expert_pack_cache_not_authoritative", 0))

    source_mismatch = int(manifest.quant_lab_commit != task.quant_lab_commit) + int(
        manifest.template_registry_digest != task.template_registry_digest
    )
    checks.append(_check("source_commit_and_registry_match", source_mismatch))

    safety_violations = _alpha_safety_violations(candidates, results, factor_bridge)
    checks.append(_check("read_only_no_live_action", safety_violations))
    expected_names = set(ALPHA_FACTORY_ANTI_LEAKAGE_CHECKS)
    actual_names = {str(check["check_name"]) for check in checks}
    if actual_names != expected_names:
        raise RuntimeError("alpha_factory_anti_leakage_check_set_mismatch")
    violation_count = sum(int(check["violation_count"]) for check in checks)
    return {
        "schema_version": "quant_lab.alpha_factory_anti_leakage.v1",
        "task_id": task.task_id,
        "snapshot_id": task.snapshot_id,
        "as_of_date": task.as_of_date.isoformat(),
        "status": "PASS" if violation_count == 0 else "FAIL",
        "violation_count": violation_count,
        "checks": checks,
        "research_only": True,
        "live_order_effect": "none",
    }


def _normalize_artifact_frames(
    artifacts: AlphaFactoryComputeArtifacts,
) -> dict[str, pl.DataFrame]:
    frames = artifacts.frames_by_dataset()
    normalized: dict[str, pl.DataFrame] = {}
    for spec in ALPHA_FACTORY_COMPUTE_OUTPUT_SPECS:
        frame = frames[spec.dataset_name]
        if set(frame.columns) != set(spec.schema):
            raise ValueError(f"alpha_factory_output_columns_mismatch:{spec.dataset_name}")
        normalized[spec.dataset_name] = frame.select(list(spec.schema)).cast(
            spec.schema,
            strict=True,
        )
    return normalized


def _complete_samples(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.filter(pl.col("label_status").str.to_lowercase() == "complete")


def _count_invalid(frame: pl.DataFrame, expression: pl.Expr) -> int:
    if frame.is_empty():
        return 0
    return frame.filter(expression.fill_null(False)).height


def _market_bar_keys(
    frame: pl.DataFrame,
    *,
    completed_before: datetime,
) -> set[tuple[str | None, datetime]]:
    if frame.is_empty() or not {"symbol", "ts"}.issubset(frame.columns):
        return set()
    scoped = frame
    if "timeframe" in frame.columns:
        scoped = scoped.filter(
            pl.col("timeframe").cast(pl.Utf8).str.to_uppercase() == "1H"
        )
    if "is_closed" in scoped.columns:
        scoped = scoped.filter(pl.col("is_closed").fill_null(False))
    scoped = scoped.filter(pl.col("ts") + pl.duration(hours=1) <= completed_before)
    return {
        (normalize_symbol(row.get("symbol")), row["ts"])
        for row in scoped.select("symbol", "ts").drop_nulls().unique().to_dicts()
    }


def _metric_window_violations(
    results: pl.DataFrame,
    samples: pl.DataFrame,
    *,
    as_of_date: date,
) -> tuple[int, int]:
    chronological = 0
    recent = 0
    if results.is_empty():
        return chronological, recent
    expected_windows = _expected_metric_window_counts(samples, as_of_date=as_of_date)
    for row in results.select(
        "strategy_candidate",
        "symbol",
        "regime_state",
        "horizon_hours",
        "sample_count",
        "train_metrics_json",
        "validation_metrics_json",
        "recent_7d_metrics_json",
    ).to_dicts():
        try:
            train = json.loads(str(row.get("train_metrics_json") or "{}"))
            validation = json.loads(str(row.get("validation_metrics_json") or "{}"))
            recent_metrics = json.loads(str(row.get("recent_7d_metrics_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            chronological += 1
            recent += 1
            continue
        sample_count = int(row.get("sample_count") or 0)
        train_count = int(train.get("sample_count") or 0)
        validation_count = int(validation.get("sample_count") or 0)
        recent_count = int(recent_metrics.get("sample_count") or 0)
        expected = expected_windows.get(_metric_group_key(row))
        if expected is None:
            expected = (sample_count, 0, 0)
        expected_train, expected_validation, expected_recent = expected
        if (train_count, validation_count) != (
            expected_train,
            expected_validation,
        ):
            chronological += 1
        if recent_count != expected_recent:
            recent += 1
    return chronological, recent


def _expected_metric_window_counts(
    samples: pl.DataFrame,
    *,
    as_of_date: date,
) -> dict[tuple[str, str, str, int], tuple[int, int, int]]:
    groups: dict[tuple[str, str, str, int], list[datetime]] = {}
    if samples.is_empty():
        return {}
    for row in samples.to_dicts():
        if str(row.get("as_of_date") or "")[:10] != as_of_date.isoformat():
            continue
        timestamp = _metric_sample_timestamp(row)
        if timestamp is None:
            continue
        groups.setdefault(_metric_group_key(row), []).append(timestamp)

    windows: dict[tuple[str, str, str, int], tuple[int, int, int]] = {}
    for key, timestamps in groups.items():
        timestamps.sort()
        start_ts = timestamps[0]
        end_ts = timestamps[-1]
        total_seconds = max((end_ts - start_ts).total_seconds(), 0.0)
        cut_ts = start_ts + timedelta(seconds=total_seconds * 0.7)
        recent_start = end_ts - timedelta(days=7)
        windows[key] = (
            sum(timestamp <= cut_ts for timestamp in timestamps),
            sum(timestamp > cut_ts for timestamp in timestamps),
            sum(timestamp >= recent_start for timestamp in timestamps),
        )
    return windows


def _metric_group_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("strategy_candidate") or ""),
        normalize_symbol(row.get("symbol")) or "UNKNOWN",
        str(row.get("regime_state") or "UNKNOWN"),
        int(row.get("horizon_hours") or 0),
    )


def _metric_sample_timestamp(row: dict[str, Any]) -> datetime | None:
    for column in ("decision_ts", "ts_utc", "label_ts", "created_at"):
        value = row.get(column)
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if value in (None, ""):
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def _factor_decision_delay_violations(frame: pl.DataFrame) -> int:
    if frame.is_empty():
        return 0
    if not {"ts", "available_time"}.issubset(frame.columns):
        return frame.height
    scoped = frame
    if "is_valid" in scoped.columns:
        scoped = scoped.filter(pl.col("is_valid").fill_null(False))
    if scoped.is_empty():
        return 0
    return _count_invalid(
        scoped,
        pl.col("ts").is_null()
        | pl.col("available_time").is_null()
        | (pl.col("available_time") < pl.col("ts")),
    )


def _future_snapshot_reference_count(
    manifest: AlphaFactorySnapshotManifest,
    dataset: str,
    exclusive_end: datetime,
) -> int:
    return sum(
        1
        for reference in manifest.files
        if reference.dataset_name == dataset
        and reference.max_ts is not None
        and reference.max_ts >= exclusive_end
    )


def _count_non_none_live_effect(frame: pl.DataFrame) -> int:
    if frame.is_empty() or "live_order_effect" not in frame.columns:
        return 0
    return _count_invalid(
        frame,
        ~pl.col("live_order_effect")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .is_in(SAFE_READ_ONLY_LIVE_EFFECTS),
    )


def _alpha_safety_violations(
    candidates: pl.DataFrame,
    results: pl.DataFrame,
    factor_bridge: pl.DataFrame,
) -> int:
    violations = 0
    if not candidates.is_empty():
        violations += _count_invalid(
            candidates,
            (pl.col("max_live_notional_usdt") != 0)
            | (pl.col("candidate_state") != "RESEARCH")
            | (pl.col("safety_mode") != "paper_shadow_only"),
        )
    if not results.is_empty():
        allowed = ["RESEARCH", "KEEP_SHADOW", "KILL", "PAPER_READY"]
        violations += _count_invalid(
            results,
            (pl.col("max_live_notional_usdt") != 0)
            | ~pl.col("decision").is_in(allowed),
        )
        violations += _count_invalid(
            results,
            pl.col("strategy_candidate").str.to_lowercase().str.contains("futures")
            & (pl.col("decision") == "PAPER_READY"),
        )
        violations += _count_invalid(
            results,
            pl.col("template_name").str.to_lowercase().str.contains("factor_strategy_bridge")
            & (pl.col("decision") != "RESEARCH"),
        )
    violations += _count_non_none_live_effect(factor_bridge)
    return violations


def _decision_counts(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty() or "decision" not in frame.columns:
        return {}
    return {
        str(row["decision"]): int(row["len"])
        for row in frame.group_by("decision").len().sort("decision").to_dicts()
    }


def _check(name: str, violation_count: int) -> dict[str, Any]:
    return {
        "check_name": name,
        "status": "PASS" if violation_count == 0 else "FAIL",
        "violation_count": int(violation_count),
    }
